"""Evaluator-only coverage, provenance, and mock exclusion checks."""

from __future__ import annotations

import itertools
import json

import pytest

from evagent.fewshot.data import freeze_episode, load_landscape, write_episode
from evagent.fewshot.protocol import all_requests
from evagent.fewshot.reporting import report, run_baselines
from evagent.fewshot.runner import freeze_plan, load, save


@pytest.fixture
def campaign(tmp_path):
    source = tmp_path / "landscape.json"
    source.write_text(json.dumps([
        {"sequence": "".join(sequence), "fitness": index / 127}
        for index, sequence in enumerate(itertools.product("AC", repeat=7))
    ]))
    landscape = load_landscape(source, {
        "assay_id": "offline_report_fixture", "domain": "rna", "wt_sequence": "AAAAAAA",
        "assay_description": "Synthetic assay for offline reporting tests",
        "endpoint": "activity", "readout": "measured activity", "higher_is_better": True,
    })
    episodes_root = tmp_path / "episodes"
    artifacts = []
    for seed in (1, 2):
        frozen = freeze_episode(landscape, "random", seed, query_size=50)
        write_episode(frozen, episodes_root / frozen["episode"]["episode_id"])
        artifacts.append(frozen)
    run_dir = tmp_path / "run"
    freeze_plan(episodes_root, run_dir, {
        "model": "offline-report-test", "thinking_level": "medium", "max_output_tokens": 1024,
    })
    save(run_dir / "execution_config.json", {"mode": "live", "context_preflight": "provider_count_tokens"})
    return episodes_root, run_dir, tmp_path / "baselines", tmp_path / "report", artifacts


def write_condition(run_dir, frozen, condition, *, mode="live", partial=False):
    labels = frozen["evaluation_labels"]["labels"]
    for request in all_requests(frozen["episode"], frozen["support"]):
        if request["condition"]["name"] != condition or partial and request["chunk_index"] > 0:
            continue
        sign = -1 if condition == "numeric_0" else 1
        save(run_dir / "predictions" / request["request_hash"] / "result.json", {
            "request_hash": request["request_hash"], "episode_id": request["episode_id"],
            "condition": request["condition"], "chunk_index": request["chunk_index"],
            "expected_ids": request["expected_ids"], "execution_mode": mode, "status": "ok",
            "value": {variant_id: sign * labels[variant_id] for variant_id in request["expected_ids"]},
            "usage_summary": {"total_tokens": 10},
        })


def make_report(campaign):
    episodes_root, run_dir, baselines, output, _artifacts = campaign
    result = report(episodes_root, run_dir, baselines, output, plot=False)
    return result, load(output / "results.json")


@pytest.mark.parametrize("unit_mode", ["offline_test", "live"])
def test_report_excludes_offline_predictions_but_keeps_real_numerical_baselines(campaign, unit_mode):
    episodes_root, run_dir, baselines, _output, artifacts = campaign
    for frozen in artifacts:
        write_condition(run_dir, frozen, "numeric_0", mode=unit_mode)
    save(run_dir / "execution_config.json", {"mode": "offline_test", "context_preflight": "offline_test_injection"})
    assert run_baselines(episodes_root, baselines)["new_baseline_cells"] == 16
    assert run_baselines(episodes_root, baselines)["new_baseline_cells"] == 0
    result, table = make_report(campaign)
    assert result["llm_complete_cells"] == 0
    assert result["complete_cells"] == 16
    zero = [row for row in table["rows"] if row["condition"] == "numeric_0"]
    assert all(row["status"] == "offline_test_excluded" for row in zero)
    assert all("spearman" not in row for row in zero)
    assert table["paired_vs_zero_shot"] == []
    assert all(row["coverage"] == 1 and row["currency_cost"] == 0
               for row in table["rows"] if row["method"] in {"ridge", "nearest_neighbor"})


def test_complete_only_metrics_seed_denominators_and_paired_differences(campaign):
    _episodes_root, run_dir, _baselines, output, artifacts = campaign
    for frozen in artifacts:
        write_condition(run_dir, frozen, "numeric_0")
    write_condition(run_dir, artifacts[0], "numeric_16")
    write_condition(run_dir, artifacts[0], "numeric_4", partial=True)
    result, table = make_report(campaign)
    assert result["llm_complete_cells"] == 3
    incomplete = next(row for row in table["rows"] if row["condition"] == "numeric_4" and row["seed"] == 1)
    assert incomplete["status"] == "incomplete_or_invalid"
    assert incomplete["coverage"] == 0.5
    assert incomplete["n_predicted"] == 25
    assert "spearman" not in incomplete
    summary = {row["condition"]: row for row in table["summary"]}
    assert summary["numeric_0"]["planned_seeds"] == 2
    assert summary["numeric_0"]["complete_seeds"] == 2
    assert summary["numeric_0"]["mean_spearman"] == pytest.approx(-1)
    assert summary["numeric_16"]["planned_seeds"] == 2
    assert summary["numeric_16"]["complete_seeds"] == 1
    assert summary["numeric_16"]["mean_spearman"] == pytest.approx(1)
    assert summary["numeric_16"]["mean_top25_precision"] == 1
    assert summary["numeric_16"]["reported_tokens"] == 20
    assert len(table["paired_vs_zero_shot"]) == 1
    paired = table["paired_vs_zero_shot"][0]
    assert paired["condition"] == "numeric_16"
    assert paired["delta_spearman"] == pytest.approx(2)
    assert paired["delta_top25_precision"] == 1
    assert len(load(output / "cases.json")["cases"]) == 6


@pytest.mark.parametrize("filename", ["episode.json", "support.json", "evaluation_labels.json"])
def test_report_checks_frozen_prediction_and_evaluation_input_hashes(campaign, filename):
    episodes_root, _run_dir, _baselines, _output, artifacts = campaign
    path = episodes_root / artifacts[0]["episode"]["episode_id"] / filename
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="changed|provenance|hash|mismatch"):
        make_report(campaign)


@pytest.mark.parametrize("alteration", ["request_identity", "run_fingerprint", "missing_episode"])
def test_report_rejects_prediction_or_run_provenance_mismatch(campaign, alteration):
    episodes_root, run_dir, _baselines, _output, artifacts = campaign
    write_condition(run_dir, artifacts[0], "numeric_0")
    if alteration == "request_identity":
        path = next((run_dir / "predictions").glob("*/result.json"))
        prediction = load(path)
        prediction["request_hash"] = "unrelated-request"
        save(path, prediction)
    elif alteration == "run_fingerprint":
        path = run_dir / "plan.json"
        plan = load(path)
        plan["max_repairs"] += 1
        save(path, plan)
    else:
        episode_folder = episodes_root / artifacts[1]["episode"]["episode_id"]
        episode_folder.rename(episodes_root.parent / "unexpectedly_removed_episode")
    with pytest.raises(ValueError, match="mismatch|missing|set|denominator"):
        make_report(campaign)
