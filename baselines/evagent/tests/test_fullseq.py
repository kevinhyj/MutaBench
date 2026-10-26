import json
import math

import pytest

from evagent.fullseq import Ledger, TerminalRequest, full_prompt, validate_evidence


def candidate(i=1):
    return {
        "variant_id": f"M{i:03d}",
        "mutation": f"A{i}G",
        "sequence": "A" * 6000 + "G",
        "experimental_score": 123456789,
    }


def evidence(v):
    return {
        "variant_id": v["variant_id"],
        "mutation": v["mutation"],
        "predicted_score": 0.2,
        "direction": "beneficial",
        "confidence": 0.7,
        "mechanisms": ["test"],
        "evidence": [],
    }


def test_full_sequences_not_truncated_and_labels_excluded():
    v = candidate()
    wt = "A" * 6001
    prompt = full_prompt(
        {"DMS_ID": "test", "DMS_score": 987654321}, {"assay_summary": "test"}, wt, [v]
    )
    assert wt in prompt and v["sequence"] in prompt
    assert "123456789" not in prompt and "987654321" not in prompt
    assert "experimental_score" not in prompt


@pytest.mark.parametrize("bad", [None, True, "0.2", math.nan, math.inf, 1.1, -1.1])
def test_score_validation_never_fabricates_or_clips(bad):
    v = candidate()
    raw = evidence(v)
    raw["predicted_score"] = bad
    with pytest.raises(ValueError):
        validate_evidence({"variants": [raw]}, [v])


def test_strict_coverage_and_id_normalization():
    v = candidate()
    raw = evidence(v)
    raw["variant_id"] = "M01"
    assert validate_evidence({"variants": [raw]}, [v])[0]["variant_id"] == "M001"
    with pytest.raises(ValueError):
        validate_evidence({"variants": []}, [v])
    with pytest.raises(ValueError):
        validate_evidence({"variants": [raw, raw]}, [v])
    raw["variant_id"] = "M999"
    with pytest.raises(ValueError):
        validate_evidence({"variants": [raw]}, [v])


def test_shared_ledger_accounts_protocol_and_reuses_once(tmp_path):
    from evagent.runtime import ExecutionPaused, RequestStore

    old = RequestStore(tmp_path / "calls.sqlite", 100)
    old.invoke("legacy", lambda: {"usage": {"totalTokenCount": 5}})
    old.db.close()
    new = Ledger(tmp_path / "calls.sqlite", 12, "fullseq")
    calls = []

    def paid():
        calls.append(1)
        return {"usage": {"totalTokenCount": 8}, "answer": json.dumps({"ok": True})}

    new.invoke("new", paid)
    new.invoke("new", paid)
    assert len(calls) == 1 and new.accounting()["reported_tokens"] == 13
    with pytest.raises(ExecutionPaused):
        new.invoke("another", paid)
    assert {x["protocol"] for x in new.accounting()["by_protocol"]} == {"legacy", "fullseq"}


def test_transport_retry_limit_survives_restart(tmp_path):
    store = Ledger(tmp_path / "calls.sqlite", 100, "fullseq")
    for i in range(5):
        store.db.execute(
            "INSERT INTO requests VALUES (?, ?, NULL, NULL, 1, 2)",
            (f"key:attempt:{i}", "failed_unknown_cost"),
        )
    store.db.commit()
    with pytest.raises(TerminalRequest, match="exhausted"):
        store.invoke("key", lambda: pytest.fail("sixth wire attempt"))


def test_auth_failure_pauses_and_redacts_error(tmp_path, monkeypatch):
    from evagent.runtime import ExecutionPaused

    monkeypatch.setenv("GEMINI_API_KEY", "fake-secret-for-test")
    store = Ledger(tmp_path / "calls.sqlite", 100, "fullseq")

    def fail():
        raise RuntimeError("HTTP 403 fake-secret-for-test")

    with pytest.raises(ExecutionPaused):
        store.invoke("auth", fail)
    assert store.stop.is_set()
    raw = store.db.execute("SELECT raw FROM requests WHERE key='auth'").fetchone()[0]
    assert "fake-secret-for-test" not in raw and "[REDACTED]" in raw


def test_retry_delays_require_non_negative_seconds(tmp_path, monkeypatch):
    monkeypatch.setenv("EVAGENT_RETRY_DELAYS_S", "0,-1")
    store = Ledger(tmp_path / "calls.sqlite", 100, "fullseq")
    with pytest.raises(ValueError, match="non-negative"):
        store.invoke("bad-delays", lambda: {})


def test_smoke_gate_requires_six_cells_and_five_valid_cr(tmp_path):
    from evagent.fullseq import save, smoke_gate

    for i in range(6):
        save(
            tmp_path / f"assays/a{i}/evaluations/n50/evaluation.json",
            {"cr_status": "complete" if i < 5 else "incomplete"},
        )
    assert smoke_gate(tmp_path)
    save(tmp_path / "failures.json", [{"fatal": True}])
    assert not smoke_gate(tmp_path)


def test_mock_cell_report_preserves_ties_and_missing_global(tmp_path):
    import time
    from types import SimpleNamespace

    import pandas as pd

    from evagent.fullseq import Experiment

    run = Experiment.__new__(Experiment)
    run.root = tmp_path
    run.started = time.time()
    run.source = {"assays": [{"assay": "mock", "n_valid": 500}]}
    run.metadata = {"mock": {"RAW_CONSTRUCT_SEQ": "AAAA"}}
    run.classes = {"mock": {"rna_type": "mock", "protein_stability_source": False}}
    run.store = SimpleNamespace(accounting=lambda: {"reported_tokens": 0})
    variants = [{**candidate(i), "experimental_score": float(i)} for i in range(1, 51)]
    run.variants = lambda assay, n: variants
    run.global_ranking = lambda *args: {"status": "token_preflight_unavailable"}
    run.memory = lambda assay: {"status": "complete", "data": {}}
    run.chunk = lambda assay, memory, rows, index: {
        "status": "complete",
        "data": [evidence(v) for v in rows],
    }
    run.cell("mock", 50)
    progress = run.report()
    frame = pd.read_csv(tmp_path / "analysis/experiment_long.csv")
    assert frame.spearman.isna().all()  # Constant CR scores are undefined, not zero.
    assert progress["cr_complete_cells"] == 1
    assert progress["global_valid_cells"] == 0
    assert progress["infrastructure_pending_method_cells"] == 1
