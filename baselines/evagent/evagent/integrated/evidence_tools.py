"""Deterministic evidence preparation; optional tools never load in prediction."""
from __future__ import annotations

import json
import math
from pathlib import Path

from .bundles import SCHEMA, file_hash, validate_bundle
from .gateway import digest


def canonical(sequence: str) -> str:
    result = sequence.upper().replace("T", "U")
    if not result or set(result) - set("ACGU"):
        raise ValueError("Evidence requires an unambiguous RNA sequence")
    return result


def differences(wt: str, sequence: str) -> list[int]:
    if len(wt) != len(sequence):
        raise ValueError("This evidence version requires substitution-only candidates")
    return [i+1 for i, (a, b) in enumerate(zip(wt, sequence)) if a != b]


def empty_source(reason: str) -> dict:
    return {"status": "unavailable", "reason": reason, "task": {}, "positions": {},
            "candidates": {}, "provenance": [], "limitations": []}


def read_alignment(path: Path) -> list[tuple[str, str]]:
    text = path.read_text()
    records: dict[str, str] = {}
    if text.lstrip().startswith("# STOCKHOLM"):
        for line in text.splitlines():
            parts = line.split()
            if len(parts) >= 2 and not line.startswith(("#", "//")):
                records[parts[0]] = records.get(parts[0], "") + parts[1]
    else:
        name = None
        for line in text.splitlines():
            if line.startswith(">"):
                name = line[1:].strip()
                if name in records:
                    raise ValueError("Duplicate alignment sequence identifier")
                records[name] = ""
            elif line.strip():
                if name is None:
                    raise ValueError("Invalid alignment format")
                # A3M lower-case insertions do not occupy reference columns.
                records[name] += "".join(c for c in line.strip() if not c.islower())
    if not records or len({len(s) for s in records.values()}) != 1:
        raise ValueError("Alignment is empty or ragged")
    return [(name, s.upper().replace("T", "U").replace(".", "-")) for name, s in records.items()]


def msa_features(wt: str, variants: list[dict], path: Path, audit: dict) -> dict:
    if audit.get("origin") != "natural_homologs" or audit.get("reviewed") is not True:
        raise ValueError("Alignment origin must be reviewed natural homologs")
    records = read_alignment(path)
    if len({s for _, s in records}) < 2:
        raise ValueError("Query-only alignment is not homolog evidence")
    matches = [s for _, s in records if s.replace("-", "") == wt]
    if not matches:
        raise ValueError("MSA lacks an exact WT query; align WT with the frozen family CM")
    query = matches[0]
    if any(s != query for s in matches):
        raise ValueError("Ambiguous WT alignment mapping")
    columns = [i for i, c in enumerate(query) if c != "-"]
    # Stable, label-independent cap and exact sequence deduplication.
    sequences = sorted({s for name, s in records if name != audit.get("query_identifier", "WT_query")})[:2048]
    if len(sequences) < 2 or any(set(s)-set('ACGU-') for s in sequences):
        raise ValueError("Need at least two unambiguous natural homolog sequences")
    import numpy as np
    a = np.array([list(s) for s in sequences])[:, columns]
    weights = np.zeros(len(a), dtype=float)
    for start in range(0, len(a), 64):
        chunk = a[start:start+64]
        valid = (chunk[:, None, :] != "-") & (a[None, :, :] != "-")
        denom = valid.sum(axis=2)
        same = ((chunk[:, None, :] == a[None, :, :]) & valid).sum(axis=2)
        neighbors = ((same / np.maximum(denom, 1) >= .8) & (denom >= .5*len(wt))).sum(axis=1)
        weights[start:start+len(chunk)] = 1 / np.maximum(neighbors, 1)
    neff = float(weights.sum())
    positions = {}
    for i, base in enumerate(wt):
        counts = {b: float(weights[a[:, i] == b].sum()) for b in "ACGU"}
        valid_weight = sum(counts.values())
        frequency = {b: (counts[b]+.25)/(valid_weight+1) for b in "ACGU"}
        gap = float(weights[a[:, i] == "-"].sum()/max(neff, 1e-12))
        entropy = -sum(p*math.log2(p) for p in frequency.values())
        positions[str(i+1)] = {"wt": base, "frequency": frequency, "entropy_bits": round(entropy, 5),
                               "gap_fraction": round(gap, 5), "neff": round(neff, 3),
                               "non_gap_effective_depth": round(valid_weight, 3),
                               "support": "low" if valid_weight < 20 or gap > .5 else "adequate"}
    candidates = {}
    for row in variants:
        s = canonical(row["sequence"])
        changed = differences(wt, s)
        candidates[digest(row["sequence"])] = {
            "changed_positions": changed,
            "site_log_frequency_ratios": {str(i): round(math.log(positions[str(i)]["frequency"][s[i-1]]/
                                                        positions[str(i)]["frequency"][wt[i-1]]), 5) for i in changed},
        }
    return {"status": "available", "task": {"sequences_used": len(a), "sequences_total": len(records),
            "neff": round(neff, 3), "weighting": "inverse >=80pct identity neighbors, >=50pct pairwise coverage",
            "pseudocount_total": 1, "mapping": "exact aligned WT; 1-based ungapped WT"},
            "positions": positions, "candidates": candidates, "candidate_specific": True,
            "provenance": [], "limitations": ["Conservation is context-dependent evidence, not a deterministic mutation effect.",
                "Independent-site frequencies do not model epistasis; related sequences are reweighted, not statistically independent."]}


def secondary_features(wt: str, variants: list[dict]) -> dict:
    import numpy as np
    import RNA

    def fold(sequence):
        md = RNA.md()
        md.temperature = 37.0
        fc = RNA.fold_compound(sequence, md)
        dot, energy = fc.mfe()
        fc.exp_params_rescale(energy)
        _, ensemble = fc.pf()
        bpp = np.asarray(fc.bpp())[1:, 1:]
        bpp = bpp + bpp.T
        return {"dot_bracket": dot, "mfe_kcal_mol": energy, "ensemble_kcal_mol": ensemble,
                "ensemble_diversity": fc.mean_bp_distance()}, bpp

    summary, wt_bpp = fold(wt)
    cache = {wt: (summary, wt_bpp)}
    candidates = {}
    for row in variants:
        s = canonical(row["sequence"])
        changed = differences(wt, s)
        if s not in cache:
            cache[s] = fold(s)
        result, bpp = cache[s]
        candidates[digest(row["sequence"])] = {
            "changed_positions": changed, **{k: round(v, 5) for k, v in result.items() if k != "dot_bracket"},
            "delta_mfe_kcal_mol": round(result["mfe_kcal_mol"]-summary["mfe_kcal_mol"], 5),
            "base_pair_distance": RNA.bp_distance(summary["dot_bracket"], result["dot_bracket"]),
            "mean_pair_probability_change": round(float(np.abs(bpp-wt_bpp).sum()/len(wt)), 5),
            "site_unpaired_probability": {str(i): round(float(max(0, 1-bpp[i-1].sum())), 5) for i in changed},
        }
    positions = {str(i+1): {"wt_unpaired_probability": round(float(max(0, 1-wt_bpp[i].sum())), 5),
                           "wt_pairing_partners": [[j+1, round(float(p), 4)] for j, p in enumerate(wt_bpp[i]) if p >= .1]}
                 for i in range(len(wt))}
    return {"status": "available", "task": {"secondary_structure": summary, "temperature_c": 37.0},
            "positions": positions, "candidates": candidates, "candidate_specific": True,
            "provenance": [], "limitations": ["Equilibrium, pseudoknot-free secondary-structure model; fixed computational temperature.",
                "Does not model RNA modifications, co-transcriptional folding, binding partners or assay-specific kinetics."]}


def prepare_bundle(config: dict, *, base: Path, output: Path) -> None:
    if config.get("source_audit", {}).get("status") != "admitted":
        raise ValueError("Source audit has not admitted this task")
    def path(key):
        p = Path(config[key])
        return (base / p).resolve() if not p.is_absolute() else p
    task = json.loads(path("task_file").read_text())
    # Reject complete benchmark rows: worker inputs must be physically stripped.
    from .gateway import assert_public
    assert_public(task)
    wt = canonical(task["wt_sequence"])
    sources = {name: empty_source("not_available_in_frozen_sources") for name in ("msa", "structure", "literature")}
    if config.get("msa_file"):
        sources["msa"] = msa_features(wt, task["variants"], path("msa_file"), config["msa_audit"])
        sources["msa"]["provenance"] = [{"id": "msa", "sha256": file_hash(path("msa_file")),
                                           "tool": "evagent-msa", "version": "v1", **config["msa_audit"]}]
    if config.get("secondary_file"):
        sources["structure"] = json.loads(path("secondary_file").read_text())
    elif config.get("compute_secondary", False):
        import RNA
        sources["structure"] = secondary_features(wt, task["variants"])
        sources["structure"]["provenance"] = [{"id": "secondary", "sha256": digest(sources["structure"]),
            "tool": "ViennaRNA", "version": RNA.__version__, "parameters": {"temperature_c": 37.0}}]
    if config.get("structure_file"):
        record = json.loads(path("structure_file").read_text())
        if record.get("wt_sha256") != digest(task["wt_sequence"]):
            raise ValueError("3D structure WT mismatch")
        sources["structure"]["task"]["tertiary_structure"] = record["task"]
        sources["structure"]["provenance"].extend(record["provenance"])
        for pos, value in record.get("positions", {}).items():
            sources["structure"]["positions"].setdefault(pos, {}).update(value)
        sources["structure"]["status"] = "available"
    if config.get("literature_file"):
        lit = json.loads(path("literature_file").read_text())
        if lit.get("reviewed") is not True or lit.get("assay") != task["assay"]:
            raise ValueError("Literature requires reviewed assay-matched excerpts")
        records = lit["records"]
        denied = {s.lower() for s in config.get("denied_dois", [])}
        import re
        def normalize(text):
            return re.sub(r"[^a-z0-9]", "", text.lower())
        denied_titles = {normalize(s) for s in config.get("denied_titles", [])}
        for record in records:
            if not record.get("url") or not record.get("date") or not record.get("background_only"):
                raise ValueError("Literature lacks applicability review")
            if (record.get("doi", "").lower() in denied or record.get("target_experiment")
                    or normalize(record.get("title", "")) in denied_titles
                    or record.get("direct_variant_effects") or record.get("benchmark_derived")):
                raise ValueError("Target experiment literature is prohibited")
            if any(word in record["summary"].lower() for word in ["dms_score", "experimental_score", "ranking:", "fitness ="]):
                raise ValueError("Contaminated literature text")
        if records:
            sources["literature"] = {"status": "available", "task": {"records": records},
                "positions": {}, "candidates": {}, "provenance": [{"id": "literature", "sha256": file_hash(path("literature_file")),
                    "tool": "reviewed-background-extraction", "version": "v1"}],
                "limitations": ["External retrieval controls do not rule out pretrained model exposure."]}
    for key, source in (("msa_file", "msa"), ("secondary_file", "structure"),
                        ("structure_file", "structure"), ("literature_file", "literature")):
        if config.get(key):
            artifact = path(key).resolve()
            relative = artifact.relative_to(output.parent.resolve())
            sources[source]["provenance"].append({"id": key, "sha256": file_hash(artifact),
                "artifact": str(relative), "tool": "frozen-input", "version": "v1"})
    bundle = {"schema": SCHEMA, "assay": task["assay"], "wt_sha256": digest(task["wt_sequence"]),
              "source_status": "admitted", "source_audit": config["source_audit"], "sources": sources}
    bundle["sha256"] = digest(bundle)
    validate_bundle(bundle, root=output.parent)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and json.loads(output.read_text()) != bundle:
        raise ValueError("Refusing to replace a frozen evidence bundle")
    output.write_text(json.dumps(bundle, indent=2, allow_nan=False)+"\n")
