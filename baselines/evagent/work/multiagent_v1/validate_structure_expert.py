"""Offline actual-library acceptance on all admitted frozen tasks. No model client."""

from __future__ import annotations

import argparse
import datetime
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

from evagent.integrated.gateway import assert_public
from evagent.integrated.research.campaign import prepare, run_campaign
from evagent.integrated.research.contracts import load_snapshot, save
from evagent.integrated.research.structure_analysis import canonical, changed
from evagent.integrated.research.structure_review import DESIGN, ActiveReviewTools
from evagent.integrated.tasks import load_task


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = args.source_config.resolve()
    out = args.output.resolve()
    raw = json.loads(source.read_text())
    for row in raw["tasks"]:
        for field in ("sample_file", "context_file", "snapshot_file"):
            row[field] = str((source.parent / row[field]).resolve())
    for field in ("coverage_file", "environment_file"):
        if raw.get(field):
            raw[field] = str((source.parent / raw[field]).resolve())
    for ledger in raw.get("prior_ledgers", []):
        ledger["path"] = str((source.parent / ledger["path"]).resolve())
    raw.update(
        analysis_profile="tool_review_v3",
        analysis=DESIGN,
        repeats=1,
        output_dir=str(out / "not_launched_campaign"),
    )
    raw["model_config"]["research_tool_protocol"] = "native_structure_v3"
    raw.pop("execution", None)
    raw.pop("resume_from", None)
    config_path = out / "candidate_config.json"
    save(config_path, raw)
    dry = run_campaign(config_path, dry_run=True)
    manifest, _ = prepare(config_path)
    save(out / "dry_run.json", dry)
    save(out / "candidate_manifest.json", manifest)
    coverage = []
    for spec in manifest["tasks"]:
        task = load_task(spec)
        snapshot = load_snapshot(Path(spec["snapshot_file"]), task)
        # Fixed input order, no labels or historical performance used for acceptance selection.
        selected = [task.variants[0]]
        multiple = next(
            (
                v
                for v in task.variants
                if 2
                <= len(changed(canonical(task.wt_sequence), canonical(v["sequence"])) or [])
                <= 12
            ),
            None,
        )
        if multiple and multiple["variant_id"] != selected[0]["variant_id"]:
            selected.append(multiple)
        public = {
            "assay": task.assay,
            "wt_sequence": task.wt_sequence,
            "variants": selected,
            "metadata": task.metadata,
        }
        assert_public(public)
        model = SimpleNamespace(
            gateway=SimpleNamespace(run_dir=out / "actual_tool_acceptance" / "provider"),
            metadata={
                "assay": task.assay,
                "role": "review_structure",
                "condition": "offline_acceptance",
            },
        )
        tools = ActiveReviewTools(
            public, snapshot, "review_structure", model, snapshot_path=spec["snapshot_file"]
        )
        results = {}
        for mode in ("pairing", "ensemble"):
            results["secondary_" + mode] = tools.call("rna_secondary", {"mode": mode})
        if multiple:
            results["joint_changes"] = tools.call(
                "rna_secondary",
                {"mode": "joint_changes", "candidate_ids": [multiple["variant_id"]]},
            )
        # Compute raw WT annotations for each available asset, then verify model-visible filters.
        geometry = tools.backend.geometry()
        results["contacts"] = tools.call("rna_contacts", {})
        results["annotations"] = tools.call("rna_annotations", {})
        for result in results.values():
            assert_public(result)
        coverage.append(
            {
                "assay": task.assay,
                "family": spec["family"],
                "candidates_in_frozen_task": spec["maximum_n"],
                "candidates_exercised": [v["variant_id"] for v in selected],
                "snapshot_sha256": snapshot["sha256"],
                "preflight": spec["structure_preflight"],
                "raw_geometry_status": geometry["status"],
                "raw_Barnaba_annotation_count": len(geometry.get("annotations", [])),
                "raw_Barnaba_torsion_positions": sum(
                    bool(p.get("backbone_angles_radians"))
                    for p in geometry.get("positions", {}).values()
                ),
                "model_visible_results": {
                    k: dict(Counter(c["status"] for c in r["evidence"])) for k, r in results.items()
                },
                "tool_calls": len(tools.records),
                "tool_elapsed_s": sum(t["elapsed_s"] for t in tools.records),
            }
        )
        save(out / "actual_tool_coverage.json", coverage)
        print(
            json.dumps(
                {
                    "assay": task.assay,
                    "raw_geometry": geometry["status"],
                    "eligible_positions": len(
                        spec["structure_preflight"]["eligible_geometry_positions"]
                    ),
                }
            ),
            flush=True,
        )
    result = {
        "status": "passed",
        "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "api_calls": 0,
        "launched_campaign": False,
        "tasks": len(coverage),
        "dry_run": dry,
        "geometry_statuses": dict(Counter(r["preflight"]["geometry_status"] for r in coverage)),
        "candidate_input_count": sum(r["candidates_in_frozen_task"] for r in coverage),
        "actual_tool_calls": sum(r["tool_calls"] for r in coverage),
        "raw_Barnaba_annotations": sum(r["raw_Barnaba_annotation_count"] for r in coverage),
        "raw_Barnaba_torsion_positions": sum(r["raw_Barnaba_torsion_positions"] for r in coverage),
        "configuration_note": "Offline candidate only. Refresh prior ledger lineage after the currently running source campaign closes before any paid launch.",
    }
    save(out / "acceptance.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
