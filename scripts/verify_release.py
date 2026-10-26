"""Verify exported source hashes, frozen metadata and saved numeric packets."""
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def verify():
    manifest = json.loads((ROOT / "benchmark/source_manifest.json").read_text())
    for record in manifest["files"]:
        relative = Path(record["destination"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Invalid manifest path")
        path = ROOT / relative
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != record["exported_sha256"]:
            raise ValueError("Export hash mismatch: " + str(relative))
    catalog = json.loads((ROOT / "benchmark/singlechain_catalog.json").read_text())["assays"]
    identities = {(row["domain"], row["assay"]) for row in catalog}
    if len(identities) != len(catalog):
        raise ValueError("Duplicate catalog identity")
    for row in catalog:
        if row["n"] != len(row["candidate_ids"]) or row["n"] != len(set(row["candidate_ids"])):
            raise ValueError("Invalid frozen candidate order")
    metrics = json.loads((ROOT / "results/single_chain_metrics.json").read_text())["rows"]
    cells = set()
    for row in metrics:
        identity = tuple(row.get(k) for k in ("domain", "assay", "model_key", "arm", "aggregation"))
        if identity in cells:
            raise ValueError("Duplicate saved metric cell")
        cells.add(identity)
        if row["status"] != "complete" or row["n_expected"] != row["n_predicted"]:
            raise ValueError("Saved complete-metric table contains an incomplete cell")
        if row["value"] is not None and not math.isfinite(row["value"]):
            raise ValueError("Nonfinite metric")
    figures = json.loads((ROOT / "benchmark/figure_cohorts.json").read_text())["figures"]
    for entry in figures:
        packet = json.loads((ROOT / entry["packet"]).read_text())
        if packet["cohort"] != entry["cohort"]:
            raise ValueError("Figure cohort mismatch")
        for bar in packet["bars"]:
            if (bar["n"] != len(packet["cohort"]) or len(bar["assay_ids"]) != bar["n"]
                or set(bar["assay_ids"]) != set(packet["cohort"]) or len(bar["values"]) != bar["n"]
                or not math.isclose(math.fsum(bar["values"]) / bar["n"], bar["mean"], abs_tol=1e-12)):
                raise ValueError("Figure values/mean differ: " + bar["name"])
    complexes = json.loads((ROOT / "results/complex_metrics.json").read_text())["rows"]
    for row in complexes:
        if row["status"] == "complete" and row["n_expected"] != row["n_predicted"]:
            raise ValueError("Complex complete status has incomplete coverage")
        if row["status"] != "complete" and row["spearman"] is not None:
            raise ValueError("Incomplete complex cell was scored")
    return dict(status="PASS", exported_artifacts=len(manifest["files"]), registered_assays=len(catalog),
                saved_metric_cells=len(metrics), saved_complex_cells=len(complexes), figures=len(figures), model_calls=0)


if __name__ == "__main__":
    print(json.dumps(verify()))
