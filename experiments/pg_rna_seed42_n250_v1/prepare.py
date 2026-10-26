"""Freeze independent seed-42 query and private labels for every inventoried assay."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
AA = set("ACDEFGHIKLMNPQRSTVWY")
RNA = set("ACGU")
SEED = 42


def sha(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canon(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True,
                       separators=(",", ":"), allow_nan=False) + "\n").encode()


def freeze(path: Path, value, mode=0o644):
    payload = canon(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError(f"Frozen file differs: {path}")
        return
    with path.open("xb") as stream:
        stream.write(payload)
    path.chmod(mode)


def source_rows(item, domain):
    path = Path(item["source_csv"]["path"])
    if sha(path) != item["source_csv"]["sha256"]:
        raise ValueError("Source CSV changed: " + str(path))
    wt = item["wt_sequence"].strip().upper().replace("T", "U") if domain == "rna" else item["wt_sequence"]
    seen = {}
    excluded = Counter()
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for raw in csv.DictReader(stream):
            try:
                score = float(raw["DMS_score"])
                if not math.isfinite(score):
                    raise ValueError("nonfinite_score")
                if domain == "protein":
                    mutation = raw["mutant"]
                    seq = raw["mutated_sequence"]
                    rebuilt = list(wt)
                    positions = []
                    for part in mutation.split(":"):
                        match = re.fullmatch(r"([A-Z])(\d+)([A-Z])", part)
                        if match is None:
                            raise ValueError("invalid_mutation")
                        old, pos, new = match.groups()
                        pos = int(pos)
                        if new not in AA or pos < 1 or pos > len(wt) or wt[pos - 1] != old:
                            raise ValueError("WT_or_alphabet_mismatch")
                        rebuilt[pos - 1] = new
                        positions.append(pos)
                    if seq != "".join(rebuilt) or seq == wt:
                        raise ValueError("mutated_sequence_mismatch")
                else:
                    seq = raw["sequence"].strip().upper().replace("T", "U")
                    mutation = str(raw["mutant"])
                    if len(seq) != len(wt) or set(seq) - RNA or seq == wt:
                        raise ValueError("RNA_sequence_invalid")
                    positions = [i + 1 for i, (a, b) in enumerate(zip(wt, seq)) if a != b]
                    if not positions:
                        raise ValueError("RNA_no_change")
                if seq in seen:
                    if seen[seq]["score"] != score:
                        raise ValueError("duplicate_sequence_conflicting_labels")
                    excluded["duplicate_sequence"] += 1
                    continue
                seen[seq] = {"mutation": mutation, "sequence": seq,
                             "changed_positions": positions, "score": score}
            except (KeyError, ValueError, IndexError, TypeError) as exc:
                excluded[str(exc)] += 1
    return wt, list(seen.values()), dict(excluded)


def main():
    inventory_path = HERE / "INVENTORY.json"
    inventory = json.loads(inventory_path.read_text())
    summary = []
    for domain, key in (("protein", "protein"), ("rna", "rna")):
        for item in inventory[key]:
            assay = item["assay"]
            wt, rows, exclusions = source_rows(item, domain)
            if len(rows) < 3:
                raise ValueError(f"Fewer than three usable variants: {domain}/{assay}")
            # Stable sequence-only ordering makes the sampling independent of labels.
            ordered = sorted(rows, key=lambda x: (x["mutation"], x["sequence"]))
            permutation = np.random.RandomState(SEED).permutation(len(ordered))
            n = min(250, len(ordered))
            chosen = [ordered[int(i)] for i in permutation[:n]]
            candidates, labels = [], {}
            for index, row in enumerate(chosen, 1):
                cid = f"M{index:03d}"
                candidates.append({"id": cid, "mutation": row["mutation"],
                                   "sequence": row["sequence"],
                                   "changed_positions": row["changed_positions"]})
                labels[cid] = row["score"]
            folder = HERE / "queries" / domain / assay
            query = {"schema": "pg-rna-seed42-query-v1", "domain": domain, "assay": assay,
                     "seed": SEED, "n": n, "wt_sequence": wt,
                     "candidates": candidates,
                     "assay_context": item.get("metadata") if domain == "protein" else
                         json.loads(Path(item["input_metadata"]["path"]).read_text())["metadata"]}
            freeze(folder / "query.json", query)
            freeze(folder / "evaluation_labels.json",
                   {"assay": assay, "higher_is_better": True, "labels": labels}, 0o600)
            manifest = {"schema": "pg-rna-seed42-query-manifest-v1", "domain": domain,
                        "assay": assay, "seed": SEED, "n": n, "source_usable_n": len(rows),
                        "sampling": "RandomState(42) permutation of mutation-and-sequence sorted, label-free valid records",
                        "source_csv": item["source_csv"], "query_sha256": sha(folder / "query.json"),
                        "evaluation_labels_sha256": sha(folder / "evaluation_labels.json"),
                        "exclusions": exclusions,
                        "msa_source": item.get("msa_profile") if domain == "protein" else item.get("msa_record"),
                        "structure_source": item.get("structure_features") if domain == "protein" else item.get("structure")}
            freeze(folder / "manifest.json", manifest)
            summary.append({"domain": domain, "assay": assay, "n": n,
                            "source_usable_n": len(rows),
                            "msa_resource": manifest["msa_source"] is not None,
                            "structure_resource": manifest["structure_source"] is not None,
                            "query": str(folder / "query.json"),
                            "manifest": str(folder / "manifest.json")})
    freeze(HERE / "PREPARATION.json", {"schema": "pg-rna-seed42-preparation-v1",
           "seed": SEED, "target_n": 250, "inventory_sha256": sha(inventory_path),
           "assays": summary,
           "counts": {d: {"all": sum(x["domain"] == d for x in summary),
                           "n250": sum(x["domain"] == d and x["n"] == 250 for x in summary),
                           "short": sum(x["domain"] == d and x["n"] < 250 for x in summary)}
                      for d in ("protein", "rna")}})
    print(json.dumps(json.loads((HERE / "PREPARATION.json").read_text())["counts"]))


if __name__ == "__main__":
    main()
