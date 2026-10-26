"""Data-custodian stage. Labels are written separately and never sent to predictors."""
import csv
import math
import random
import sys
from pathlib import Path

from .common import freeze, load, normalize, sha, validate_input, mutations


def prepare(spec_path, out):
    spec = load(spec_path)
    source = Path(spec["source_csv"]).resolve()
    modality = spec["modality"]
    base = normalize(spec["base"], modality)
    with source.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    usable, excluded, seen = [], {}, set()
    mapped_positions = None
    if spec.get('require_msa_mapped', False):
        from .models import read_alignment
        _, positions, _, _, _, _ = read_alignment({'base': base, 'modality': modality, 'evidence': spec['evidence']})
        mapped_positions = set(positions)
    # Keep first valid source occurrence, regardless of label value; no averaging or label ranking.
    for index, row in enumerate(rows):
        try:
            seq = normalize(row[spec["sequence_column"]], modality)
            label = float(row[spec["label_column"]])
            if not math.isfinite(label):
                raise ValueError("nonfinite_label")
            if modality != "ligand" and len(seq) != len(base):
                raise ValueError("indel")
            if seq == base:
                raise ValueError("reference")
            if mapped_positions is not None and any(i not in mapped_positions for i, _, _ in mutations(base, seq)):
                raise ValueError('mutation_outside_msa_supported_coordinates')
            if seq in seen:
                raise ValueError("duplicate_sequence")
            seen.add(seq)
            usable.append((seq, label, index + 2))
        except (ValueError, KeyError, TypeError) as e:
            reason = str(e)
            excluded[reason] = excluded.get(reason, 0) + 1
    if len(usable) < 250:
        raise ValueError(f"Only {len(usable)} unique valid candidates; cannot satisfy N=250")
    # Exact protocol: lexical canonical sequence order, then Python Random(42).sample.
    chosen = random.Random(42).sample(sorted(usable, key=lambda x: x[0]), 250)
    item = {"schema": "singlechain-v1", "assay": spec["assay"], "modality": modality,
            "n": 250, "seed": 42, "base": base, "endpoint": spec["endpoint"],
            "candidates": [{"id": i, "sequence": x[0]} for i, x in enumerate(chosen, 1)],
            "evidence": spec.get("evidence", {}), "smoke_only": False}
    validate_input(item)
    out = Path(out)
    freeze(out / "input.json", item)
    freeze(out / "evaluation_labels.json", {
        "input_sha256": sha(out / "input.json"), "labels": [
            {"id": i, "raw_label": x[1], "label": x[1] * spec["endpoint"]["label_sign"],
             "source_row": x[2]} for i, x in enumerate(chosen, 1)]})
    freeze(out / "manifest.json", {
        "source_csv": str(source), "source_sha256": sha(source),
        "spec_sha256": sha(spec_path), "input_sha256": sha(out / "input.json"),
        "labels_sha256": sha(out / "evaluation_labels.json"),
        "n": 250, "seed": 42, "source_n": len(rows), "usable_unique_n": len(usable),
        "excluded": excluded, "sampling": "canonical sequence lexical sort; random.Random(42).sample(k=250)",
        "source": spec.get("source", "local source; see source_csv"),
        "require_msa_mapped": spec.get('require_msa_mapped', False),
        "python": sys.version,
        "preparation_code_sha256": {p.name: sha(p) for p in [Path(__file__), Path(__file__).with_name('common.py'), Path(__file__).with_name('models.py')]},
        "scope": spec.get("scope", "user-specified assay"),
        "labels_used_for_selection": "finite-label eligibility only; no value-based selection"})
    return {"assay": spec["assay"], "n": 250, "out": str(out)}
