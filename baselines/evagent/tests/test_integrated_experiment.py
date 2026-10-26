"""End-to-end synthetic experiments. Network transports are always injected."""

import json

import pytest

from evagent.integrated.config import ExperimentConfig
from evagent.integrated.experiment import prepare, run_experiment
from evagent.integrated.ranking import final_json_object
from evagent.integrated.tasks import load_task


def setup_config(tmp_path, *, method="cr", backend="direct", **options):
    (tmp_path / "sample.json").write_text(json.dumps({
        "assay": "synthetic_contract", "size": 7,
        "variants": [{"id": f"synthetic-{i}", "seq": "ACGUACGU", "score": 9876500 + i}
                     for i in range(7)],
    }))
    (tmp_path / "context.json").write_text(json.dumps({
        "assay": "synthetic_contract", "wt_sequence": "ACGUACGU",
        "metadata": {"DMS_ID": "synthetic_contract", "ASSAY_DESCRIPTION": "Synthetic software fixture",
                     "ground_truth": "NEVER_SEND_CONTEXT_LABEL"},
    }))
    path = tmp_path / "config.json"
    path.write_text(json.dumps({
        "method": method, "backend": backend, "model": "gemini-fake",
        "thinking_level": "medium", "sample_file": "sample.json", "context_file": "context.json",
        "output_dir": "results", "token_budget": 10000, "group_size": 4, "anchor_count": 1,
        **options,
    }))
    return ExperimentConfig.load(path)


def _after(text, marker):
    return json.JSONDecoder().raw_decode(text.split(marker, 1)[1].lstrip())[0]


def synthetic_transport(request):
    import re

    user = request["messages"][-1]["content"]
    if "Candidates with local WT/mutant sequence context:" in user:
        rows = _after(user, "Candidates with local WT/mutant sequence context:")
        answer = {"variants": [{"variant_id": row["variant_id"], "mutation": row["mutation"],
                               "direction": "neutral", "predicted_score": int(row["variant_id"][1:]) / 10,
                               "mechanisms": ["synthetic fixture"], "evidence": [], "confidence": 0.5}
                              for row in rows]}
    elif "Current uncertain region, already listed approximately best-to-worst:" in user:
        ids = _after(user, "Current uncertain region, already listed approximately best-to-worst:")
        answer = {"ranking": sorted(ids, reverse=True), "rationale": []}
    elif "Variant evidence:" in user:
        rows = _after(user, "Variant evidence:")
        answer = {"ranking": sorted([row["variant_id"] for row in rows], reverse=True), "rationale": []}
    elif "Analyze this assay and return exactly:" in user:
        answer = {"assay_summary": "Synthetic fixture", "property_direction": "higher",
                  "functional_hypotheses": [], "sequence_features": []}
    else:
        ids = re.findall(r"^(M\d+) \[", user, re.MULTILINE)
        assert ids and len(ids) == len(set(ids))
        if "candidate mutants to score" in user:
            answer = {"variants": [{"variant_id": value, "predicted_score": int(value[1:]) / 10}
                                   for value in ids]}
        else:
            answer = {"ranking": sorted(ids, reverse=True)}
    return {"final_text": json.dumps(answer), "finish_reason": "STOP", "response_metadata": {},
            "thought_summary": "", "usage_summary": {"model_calls": 1, "input_tokens": 20,
               "output_tokens": 10, "reasoning_tokens": 0, "total_tokens": 30}}


@pytest.mark.parametrize("method", ["global", "cr", "acr", "workflow"])
@pytest.mark.parametrize("backend", ["direct", "deepagents"])
def test_complete_pipeline_no_labels_and_ledger(tmp_path, method, backend):
    if backend == "deepagents":
        pytest.importorskip("deepagents")
    config = setup_config(tmp_path, method=method, backend=backend)
    requests = []
    def transport(request):
        requests.append(request)
        return synthetic_transport(request)
    result = run_experiment(config, transport=transport)
    assert result["status"] == "ok", result
    assert set(result["ranking"]) == {f"M{i:02d}" for i in range(1, 8)}
    assert result["metrics"]["n"] == 7
    assert result["ledger"]["provider_attempts"] == len(requests)
    assert result["ledger"]["reported_tokens"] == len(requests) * 30
    for request in requests:
        text = json.dumps(request)
        assert "987650" not in text
        assert "NEVER_SEND_CONTEXT_LABEL" not in text
        assert request["thinking_level"] == "medium"
    again = run_experiment(config, transport=lambda _: pytest.fail("Completed run must not regenerate"))
    assert again["reused_saved_result"]
    assert not (tmp_path / "state").exists()


def test_dry_run_requires_no_key_writes_or_budget(tmp_path, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    config = setup_config(tmp_path, token_budget=None)
    before = sorted(str(path) for path in tmp_path.rglob("*"))
    result = run_experiment(config, dry_run=True)
    assert result["api_calls"] == result["writes"] == 0
    assert sorted(str(path) for path in tmp_path.rglob("*")) == before
    with pytest.raises(ValueError, match="token_budget"):
        run_experiment(config, transport=lambda _: pytest.fail("No budget"))


def test_changed_input_and_modes_cannot_reuse_run(tmp_path):
    config = setup_config(tmp_path)
    run_experiment(config, transport=synthetic_transport)
    raw = json.loads(config.source.read_text())
    raw["max_repairs"] = 0
    config.source.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="different frozen"):
        run_experiment(ExperimentConfig.load(config.source), transport=lambda _: pytest.fail("Changed config"))


def test_memory_snapshot_changes_fingerprint(tmp_path):
    from evagent.integrated.memory import initialize_memory, write_basic_memory

    db = tmp_path / "basic.sqlite"
    initialize_memory(basic_db=db)
    write_basic_memory(db, "Prefer concise validated output.")
    config = setup_config(tmp_path, memory_mode="basic", basic_memory_db=str(db))
    first = prepare(config)[1]["fingerprint"]
    write_basic_memory(db, "Prefer complete validated output.")
    assert prepare(config)[1]["fingerprint"] != first


@pytest.mark.parametrize("mode", ["skill", "literature", "structure", "curated"])
def test_evidence_modes_reach_prepared_context_and_do_not_write(tmp_path, mode):
    import csv
    reference = tmp_path / "rg/data/RNAGym/fitness/reference_sheet_final.csv"
    reference.parent.mkdir(parents=True)
    with reference.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["DMS_ID", "RNA_TYPE", "ASSAY_DESCRIPTION", "RAW_CONSTRUCT_SEQ"])
        writer.writeheader()
        writer.writerow({"DMS_ID": "synthetic_contract", "RNA_TYPE": "aptamer",
                         "ASSAY_DESCRIPTION": "Synthetic software fixture", "RAW_CONSTRUCT_SEQ": "ACGUACGU"})
    config = setup_config(tmp_path, rg_llm_root=str(tmp_path / "rg"), evidence_mode=mode)
    _, manifest, context_text = prepare(config)
    assert "assay_evidence" in context_text
    assert manifest["evidence_representation"] == "text"
    dry = run_experiment(config, dry_run=True)
    assert dry["writes"] == dry["api_calls"] == 0
    assert not (tmp_path / "results").exists()


def test_pause_preserved_and_does_not_resubmit(tmp_path):
    config = setup_config(tmp_path)
    result = run_experiment(config, transport=lambda _: {"finish_reason": None})
    assert result["status"] == "paused"
    result = run_experiment(config, transport=lambda _: pytest.fail("Paused run must not resubmit"))
    assert result["status"] == "paused"
    assert result["ledger"]["provider_attempts"] == 1


def test_workflow_missing_evidence_is_invalid_not_neutral(tmp_path):
    config = setup_config(tmp_path, method="workflow")
    def missing(request):
        result = synthetic_transport(request)
        value = final_json_object(result["final_text"])
        if "variants" in value:
            value["variants"] = value["variants"][:-1]
            result["final_text"] = json.dumps(value)
        return result
    result = run_experiment(config, transport=missing)
    assert result["status"] == "invalid_output"
    assert result["ranking"] is None
    assert result["metrics"]["status"] == "not_evaluated_incomplete"
    assert result["attempts"][-1]["validation"]["valid"] is False


def test_workflow_blocked_remains_provider_status(tmp_path):
    config = setup_config(tmp_path, method="workflow", max_repairs=1)
    result = run_experiment(config, transport=lambda request: {
        **synthetic_transport(request), "finish_reason": "SAFETY"})
    assert result["status"] == "blocked"
    assert result["ledger"]["provider_attempts"] == 1


def test_bad_sample_metadata_fails_before_calls(tmp_path):
    config = setup_config(tmp_path)
    data = json.loads((tmp_path / "sample.json").read_text())
    data["size"] += 1
    (tmp_path / "sample.json").write_text(json.dumps(data))
    with pytest.raises(ValueError, match="size"):
        load_task(config.values)


def test_protocol_options_unknown_and_secret_fields_rejected(tmp_path):
    with pytest.raises(ValueError, match="Unknown config"):
        setup_config(tmp_path, api_key="must-not-be-stored")


@pytest.mark.parametrize("options", [
    {"protocol": "integrated-acr-v1"}, {"max_tokens": -1}, {"temperature": "invalid"},
    {"temperature": float("nan")}, {"timeout_s": 0},
])
def test_invalid_config_fails_before_output_or_provider(tmp_path, options):
    with pytest.raises(ValueError):
        setup_config(tmp_path, **options)
    assert not (tmp_path / "results").exists()


def test_invalid_labels_preserve_completed_predictions(tmp_path):
    config = setup_config(tmp_path, method="global")
    sample = tmp_path / "sample.json"
    value = json.loads(sample.read_text())
    value["variants"][0]["score"] = "not-a-number"
    sample.write_text(json.dumps(value))
    result = run_experiment(config, transport=synthetic_transport)
    assert result["status"] == "evaluation_error"
    assert len(result["ranking"]) == 7
    assert result["prediction_status"] == "ok"
    assert (tmp_path / "results/result.json").exists()
    repeated = run_experiment(config, transport=lambda _: pytest.fail("Never regenerate paid prediction"))
    assert repeated["reused_saved_result"]


def test_modified_sample_during_request_preserves_prediction(tmp_path):
    config = setup_config(tmp_path, method="global")
    def changing(request):
        sample = tmp_path / "sample.json"
        value = json.loads(sample.read_text())
        value["variants"][0]["score"] += 1
        sample.write_text(json.dumps(value))
        return synthetic_transport(request)
    result = run_experiment(config, transport=changing)
    assert result["status"] == "evaluation_error"
    assert len(result["ranking"]) == 7
    assert result["ledger"]["provider_attempts"] == 1


def test_corrupted_completed_result_is_rejected(tmp_path):
    config = setup_config(tmp_path, method="global")
    run_experiment(config, transport=synthetic_transport)
    path = tmp_path / "results/result.json"
    saved = json.loads(path.read_text())
    saved["ranking"].pop()
    path.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="integrity"):
        run_experiment(config, transport=lambda _: pytest.fail("Corrupt result cannot trigger requests"))


def test_output_directory_lease_precedes_manifest_writes(tmp_path):
    from evagent.runtime import ExecutionPaused
    config = setup_config(tmp_path, method="global")
    def recursive(request):
        with pytest.raises(ExecutionPaused, match="owns the experiment"):
            run_experiment(config, transport=lambda _: pytest.fail("Concurrent owner must not generate"))
        return synthetic_transport(request)
    result = run_experiment(config, transport=recursive)
    assert result["status"] == "ok"
    manifest = json.loads((tmp_path / "results/manifest.json").read_text())
    assert manifest["fingerprint"] == result["fingerprint"]


def test_frozen_assay_context_survives_long_term_memory_none(tmp_path):
    config = setup_config(tmp_path, method="workflow", assay_memory_file="assay_memory.json")
    (tmp_path / "assay_memory.json").write_text(json.dumps({
        "status": "complete", "data": {"assay_summary": "FROZEN_CONTEXT_SENTINEL",
                                        "property_direction": "higher"},
    }))
    requests = []
    def transport(request):
        requests.append(request)
        return synthetic_transport(request)
    result = run_experiment(config, transport=transport)
    assert result["status"] == "ok"
    assert result["memory_mode"] == "none"
    assert all("FROZEN_CONTEXT_SENTINEL" in json.dumps(request) for request in requests)
