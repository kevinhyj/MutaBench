"""Synthetic unit tests; fixtures are temporary and never scientific results."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sqlite3
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location("paired_report", Path(__file__).with_name("paired_analysis.py"))
reporter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reporter)


def row(seed, condition, value, *, status="ok"):
    return {"assay_id": "synthetic", "split": "random", "episode_id": f"synthetic__seed{seed}",
            "seed": seed, "condition": condition, "status": status, "coverage": 1.0 if status == "ok" else 0,
            "n_valid": 250 if status == "ok" else 0, "n_expected": 250,
            "spearman": value, "top_k_overlap_precision": value,
            "reported_tokens": 0}


def matched_rows():
    return [row(seed, condition, value) for seed, value in ((1, .2), (2, .8))
            for condition in reporter.NUMERIC + ("nearest_neighbor_4", "ridge_4")]


def test_common_numeric_cohort_and_baselines_use_same_seed_set():
    rows = matched_rows()
    rows = [item for item in rows if not (item["seed"] == 2 and item["condition"] == "numeric_32")]
    curves = reporter.common_curves(rows)
    for item in curves:
        assert item["common_metric_seeds"] == 1
        assert item["episode_ids"] == ["synthetic__seed1"]
        if item["method"] == "numeric" or item["shots"] == 4:
            assert item["mean"] == pytest.approx(.2)
    available = reporter.available_table(rows)
    unpaired_zero = next(item for item in available if item["condition"] == "numeric_0")
    assert unpaired_zero["mean_spearman"] == pytest.approx(.5)


def test_undefined_correlations_and_baselines_never_shrink_denominator():
    rows = matched_rows()
    next(item for item in rows if item["seed"] == 2 and item["condition"] == "numeric_16")["spearman"] = None
    next(item for item in rows if item["seed"] == 1 and item["condition"] == "ridge_4")["spearman"] = None
    curves = reporter.common_curves(rows)
    rho = next(item for item in curves if item["metric"] == "spearman" and item["condition"] == "numeric_0")
    top = next(item for item in curves if item["metric"] == "top_k_overlap_precision" and item["condition"] == "numeric_0")
    ridge = next(item for item in curves if item["metric"] == "spearman" and item["condition"] == "ridge_4")
    assert rho["common_metric_seeds"] == 1
    assert top["common_metric_seeds"] == 2
    assert ridge["mean"] is None
    assert ridge["status"] == "undefined_or_missing_on_common_seeds"


def test_no_numeric_results_is_missing_not_zero():
    rows = [row(1, condition, None, status="not_run") for condition in reporter.NUMERIC]
    curves = reporter.common_curves(rows)
    assert all(item["mean"] is None and item["common_metric_seeds"] == 0 for item in curves)


def test_four_control_cohort_and_pairwise_deltas_have_explicit_denominators():
    rows = [row(seed, condition, .1 * index) for seed in (1, 2)
            for index, condition in enumerate(reporter.CONTROLS)]
    rows = [item for item in rows if not (item["seed"] == 2 and item["condition"] == "shuffled_16")]
    result = reporter.control_comparisons(rows)
    assert all(item["common_seeds"] == 1 for item in result["common_summary"])
    ordinal = [item for item in result["paired_deltas"] if item["condition"] == "ordinal_16"]
    assert len(ordinal) == 4  # Two seeds x two metrics.
    assert all(item["pairwise_complete_seeds"] == 2 for item in ordinal)
    assert all(item["delta_control_minus_numeric"] == pytest.approx(.1) for item in ordinal)


def test_average_ranks_do_not_depend_on_id_order():
    assert reporter.average_ranks({"z": 2, "a": 2, "b": 1}) == {"a": 1.5, "z": 1.5, "b": 3}


def test_cases_require_genuine_signed_average_rank_changes():
    episode_id = "synthetic__seed1"
    episode = {"episode_id": episode_id, "assay_id": "synthetic", "split": "random", "higher_is_better": True,
               "query": [{"variant_id": key, "sequence": sequence, "mutation": "test"}
                         for key, sequence in (("a", "AAA"), ("b", "AAC"), ("c", "ACA"))]}
    labels = {"a": 2, "b": 2, "c": 0}
    rows = [row(1, condition, .5) for condition in ("numeric_0", "numeric_16", "numeric_32")]
    scores = {(episode_id, "numeric_0"): {"a": 0, "b": 0, "c": 1},
              (episode_id, "numeric_16"): {"b": 0, "a": 0, "c": 1},
              (episode_id, "numeric_32"): labels}
    result = reporter.tied_cases(rows, {episode_id: (episode, labels)}, scores)
    assert not any(case["condition"] == "numeric_16" for case in result["cases"])
    assert all(case["selection"] == "improved" and case["rank_error_reduction"] > 0 for case in result["cases"])
    assert all("sequence" in case and "mutation" in case and "measured_label" in case for case in result["cases"])
    assert next(item for item in result["availability"] if item["condition"] == "numeric_16")["unchanged_variants"] == 3


def test_readonly_ledger_counts_unarchived_unknown_and_prior_usage(tmp_path):
    path = tmp_path / "ledger.sqlite"
    connection = sqlite3.connect(path)
    connection.executescript("CREATE TABLE run_state(key TEXT,value TEXT); CREATE TABLE attempts(status TEXT,tokens INTEGER,request_json TEXT);")
    connection.execute("INSERT INTO run_state VALUES (?,?)", ("prior_usage", json.dumps({"reported_tokens": 5, "provider_attempts": 1})))
    for status, tokens in (("ok", 10), ("inflight", None), ("failed_unknown_cost", None), ("not_submitted", 900)):
        connection.execute("INSERT INTO attempts VALUES (?,?,?)", (status, tokens, json.dumps({"metadata": {"condition": "numeric_4"}})))
    connection.commit()
    connection.close()
    before = path.read_bytes()
    result = reporter.ledger_snapshot(path, {})
    assert result["reported_tokens"] == 15
    assert result["provider_attempts"] == 4
    assert result["unknown_usage_attempts"] == 2
    assert result["usage_is_lower_bound"] is True
    assert path.read_bytes() == before


def test_validity_separates_initial_and_repaired_and_excludes_mock_results(tmp_path):
    for index, mode in enumerate(("live", "offline_test")):
        output = tmp_path / "predictions" / str(index) / "result.json"
        output.parent.mkdir(parents=True)
        value = {"execution_mode": mode, "episode_id": "synthetic", "condition": {"name": "numeric_4"},
                 "status": "ok", "expected_ids": ["M1"], "value": {"M1": .5},
                 "attempts": [{"validation": {"valid": False}}, {"validation": {"valid": True}}],
                 "usage_summary": {"total_tokens": 10}}
        output.write_text(json.dumps(value))
    result, scores = reporter.inspect_predictions(tmp_path)
    assert result["totals"]["archived_units"] == 1
    assert result["totals"]["first_attempt_valid_fraction"] == 0
    assert result["totals"]["final_valid_fraction_of_archived_units"] == 1
    assert result["totals"]["units_with_repairs"] == 1
    assert scores["synthetic", "numeric_4"] == {"M1": .5}


def test_zero_llm_snapshot_keeps_inputs_unchanged_and_refuses_overwrite(tmp_path):
    experiment, output = tmp_path / "experiment", tmp_path / "snapshot"
    episode_dir = experiment / "episodes/synthetic"
    episode_dir.mkdir(parents=True)
    episode = {"episode_id": "synthetic__seed1", "higher_is_better": True, "query": []}
    payloads = {"episode.json": episode, "support.json": {"examples": []},
                "evaluation_labels.json": {"episode_id": "synthetic__seed1", "higher_is_better": True, "labels": {}}}
    hashes = {}
    for filename, value in payloads.items():
        encoded = json.dumps(value).encode()
        (episode_dir / filename).write_bytes(encoded)
        hashes[filename] = hashlib.sha256(encoded).hexdigest()
    (episode_dir / "manifest.json").write_text(json.dumps({"files": hashes}))
    analysis = experiment / "analysis/results.json"
    analysis.parent.mkdir()
    analysis.write_text(json.dumps({"rows": [row(1, condition, None, status="not_run") for condition in reporter.NUMERIC],
                                    "run_status": {"status": "not_started"}}))
    before = {str(path): path.read_bytes() for path in experiment.rglob("*") if path.is_file()}
    result = reporter.build_snapshot(experiment, output, plots=False)
    assert result["model_calls"] == 0
    snapshot = json.loads((output / "snapshot.json").read_text())
    assert snapshot["ledger"]["reported_tokens"] is None
    assert snapshot["provenance"]["campaign_identity"]["status"] == "baseline_only_unverified"
    assert all(item["mean"] is None for item in snapshot["common_curves"])
    assert before == {str(path): path.read_bytes() for path in experiment.rglob("*") if path.is_file()}
    with pytest.raises(FileExistsError):
        reporter.build_snapshot(experiment, output, plots=False)
    with pytest.raises(ValueError, match="frozen"):
        reporter.build_snapshot(experiment, episode_dir / "new", plots=False)


def test_campaign_fingerprints_require_three_way_agreement(tmp_path):
    plan = {"schema": "synthetic", "model": "example", "requests": []}
    plan["fingerprint"] = reporter.digest(plan)
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan))
    source = {"run_fingerprint": plan["fingerprint"]}
    ledger = {"basis": "SQLite mode=ro, one read transaction", "run_fingerprint": plan["fingerprint"]}
    result = reporter.verify_campaign_identity(source, tmp_path, ledger, has_llm_results=True)
    assert result["status"] == "verified"
    with pytest.raises(ValueError, match="Analysis/run-plan"):
        reporter.verify_campaign_identity({"run_fingerprint": "different"}, tmp_path, ledger, has_llm_results=True)
    with pytest.raises(ValueError, match="Ledger/run-plan"):
        reporter.verify_campaign_identity(source, tmp_path, dict(ledger, run_fingerprint="different"), has_llm_results=True)
    with pytest.raises(ValueError, match="ledger"):
        reporter.verify_campaign_identity(source, tmp_path, {}, has_llm_results=True)
    plan["model"] = "changed"
    path.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="digest"):
        reporter.verify_campaign_identity(source, tmp_path, ledger, has_llm_results=True)


def test_missing_plan_allowed_only_for_explicit_baseline_only_report(tmp_path):
    result = reporter.verify_campaign_identity({"run_fingerprint": "old_preview"}, tmp_path, {}, has_llm_results=False)
    assert result["status"] == "baseline_only_unverified"
    with pytest.raises(ValueError, match="matching frozen run plan"):
        reporter.verify_campaign_identity({}, tmp_path, {}, has_llm_results=True)
