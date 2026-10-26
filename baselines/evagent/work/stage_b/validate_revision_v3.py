"""Offline release checks for the fixed revision; never opens an API client."""

import json
import runpy
import subprocess
import sys
import types
from pathlib import Path

from evagent.integrated.bundles import file_hash
from evagent.integrated.campaign import prepare_campaign, read_prior_ledgers
from evagent.integrated.config import ExperimentConfig
from evagent.integrated.experiment import prepare, run_experiment
from evagent.integrated.gateway import assert_public, digest

ROOT = Path("/data/yanjie_huang/EVAgent/experiments/cr_ncrna_evidence_b_v3")


def main():
    manifest, _ = prepare_campaign(ROOT / "campaign.json")
    assets = ROOT / "assets"
    frozen = json.loads((ROOT / "assets_manifest.json").read_text())["sha256"]
    for relative, expected in frozen.items():
        assert file_hash(assets / relative) == expected, relative
    common = {}
    rows = []
    total = 0
    for cell in manifest["cells"]:
        if cell["status"] != "eligible":
            continue
        config = ExperimentConfig.load(Path(cell["config"]))
        task, request_manifest, _ = prepare(config)
        assert_public(task.metadata)
        assert_public(task.assay_memory)
        assert not task.assay_memory  # no new unreviewed summaries or memory access
        signature = {
            k: v
            for k, v in config.values.items()
            if k not in {"output_dir", "evidence_mode", "evidence_sources", "evidence_bundle_file"}
        }
        key = (cell["assay"], cell["n"])
        assert common.setdefault(key, signature) == signature
        assert config.values["prompt_version"] == "v3"
        for request in request_manifest["initial_requests"]:
            text = request["system"] + request["user"]
            assert "use only the supplied assay metadata and sequences" not in text
            assert "any frozen additional evidence provided in this request" in text
            if cell["condition"] != "cr":
                assert "Frozen multi-source evidence" in text
            assert "10.1038/418222a" not in text  # previous generic review removed
            assert "10.1038/346818a0" not in text
            total += 1
        result = run_experiment(config, dry_run=True)
        assert result["status"] == "dry_run"
        assert not Path(config.values["output_dir"]).exists()
        rows.append({k: result[k] for k in ("assay", "n", "fingerprint", "initial_requests")})
    assert len({r["fingerprint"] for r in rows}) == len(rows)
    # Check frozen v1/v2 prompt bytes against the pre-audit source implementation.
    historical = subprocess.check_output(
        [
            "git",
            "show",
            "d150e27ddaf3535274119ef8e09c24673edac811:evagent/integrated/chunked_scoring.py",
        ],
        text=True,
    )
    module = types.ModuleType("evagent.integrated._historical_scoring")
    module.__package__ = "evagent.integrated"
    sys.modules[module.__name__] = module
    exec(compile(historical, "<historical scoring>", "exec"), module.__dict__)
    from evagent.integrated.chunked_scoring import build_scoring_prompt, make_scoring_chunks

    case = runpy.run_path("tests/test_integrated_protocols.py")["synthetic_case"]()
    legacy = {}
    for version in ("v1", "v2"):
        for memory in ({}, {"summary": "Label-free task context"}):
            old = module.build_scoring_prompt(
                case,
                module.make_scoring_chunks(case)[0],
                assay_memory=memory,
                prompt_variant=version,
            )
            new = build_scoring_prompt(
                case, make_scoring_chunks(case)[0], assay_memory=memory, prompt_variant=version
            )
            assert new == old
            legacy[f"{version}-memory-{bool(memory)}"] = digest(new)
    prior = read_prior_ledgers(manifest["prior_ledgers"], ROOT)
    assert prior["reported_tokens"] == 1119985
    report = {
        "status": "passed",
        "api_calls": 0,
        "eligible_cells": len(rows),
        "initial_requests": total,
        "task_n_groups": len(common),
        "asset_files_verified": len(frozen),
        "legacy_prompt_hashes": legacy,
        "prior_usage": prior,
        "cells": rows,
        "checks": [
            "same model/sample/repair settings across conditions",
            "all v3 prompts permit frozen evidence",
            "no old generic literature",
            "unique fresh cache fingerprints",
            "no prediction directories created",
            "legacy prompt byte equivalence",
            "complete historical budget lineage",
            "asset hashes",
        ],
    }
    (ROOT / "offline_revision_acceptance.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: v
                for k, v in report.items()
                if k not in {"cells", "prior_usage", "legacy_prompt_hashes"}
            }
        )
    )


if __name__ == "__main__":
    main()
