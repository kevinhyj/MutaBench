"""Budget partition and cache preservation for an explicitly requested continuation."""
import importlib.util
import sqlite3
from pathlib import Path

import pytest

from evagent.fewshot.protocol import digest
from evagent.fewshot.runner import load, save

spec = importlib.util.spec_from_file_location("scale_continue_tool", Path(__file__).parents[1] / "tools/fewshot_scale/continue_run.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_remaining_allocation_retains_unknown_and_model():
    prior = {"budget": 200_000_000, "fingerprint": "old", "model": {"seed": 1, "max_output_tokens": 32768}}
    remaining = module.continuation_config(prior, {"inflight": 0, "total_debit": 127_055_069}, {})
    assert remaining["budget"] == 72_944_931
    assert remaining["model"] == prior["model"]
    assert prior["budget"] == 200_000_000 and prior["fingerprint"] == "old"
    assert remaining["fingerprint"] == digest({k: v for k, v in remaining.items() if k != "fingerprint"})
    with pytest.raises(ValueError):
        module.continuation_config(prior, {"inflight": 1, "total_debit": 100}, {})
    with pytest.raises(ValueError):
        module.continuation_config(prior, {"inflight": 0, "total_debit": 200_000_000}, {})


def ledger(path, rows):
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE attempts(status TEXT,tokens INTEGER,unknown INTEGER,reserve INTEGER)")
        db.executemany("INSERT INTO attempts VALUES(?,?,?,?)", rows)


def test_partial_unknown_cost_is_not_double_counted_or_cleared(tmp_path):
    path = tmp_path / "campaign.sqlite"
    ledger(path, [("ok", 100, 0, 200), ("failed_unknown_cost", 20, 1, 150), ("blocked", 0, 1, 250)])
    result = module.debit(path)
    assert result["known_tokens"] == 120
    assert result["unknown_reserve"] == 380
    assert result["total_debit"] == 500


def test_prepare_reuses_success_and_blocks_but_renews_only_reviewed_failures(tmp_path, monkeypatch):
    parent, child = tmp_path / "parent", tmp_path / "child"
    parent.mkdir()
    monkeypatch.setattr(module, "verify", lambda *a: None)
    save(parent / "DONE.json", {})
    save(parent / "cron_status.json", {"enabled": False})
    save(parent / "FROZEN.json", {"budget": 1000, "fingerprint": "parent", "model": {}, "input_hashes": {}})
    ledger(parent / "campaign.sqlite", [("ok", 100, 0, 200), ("failed_unknown_cost", 0, 1, 200)])
    (parent / "inputs").mkdir()
    for name in ("COHORT.json", "screening.json", "model_config.json", "preflight.json"):
        save(parent / name, {})
    for key, status in [("good", "ok"), ("block", "blocked"), ("retry", "transport_attempts_exhausted")]:
        folder = parent / "predictions" / key
        req = {"system": "system", "user": "user", "expected_ids": ["M001"], "request_hash": key, "protocol": "fixture"}
        result = {"status": status, "request_hash": key, "execution_mode": "live", "value": {"M001": .1}}
        result["result_digest"] = digest(result)
        save(folder / "request.json", req)
        save(folder / "result.json", result)
    inputs = {"task_id": "assay/static/zero"}
    save(parent / "tasks/assay/static/zero/inputs.json", inputs)
    save(parent / "tasks/assay/static/zero/result.json", {"status": "complete", "input_digest": digest(inputs), "requests": ["good"]})
    before = (parent / "predictions/retry/result.json").read_bytes()
    module.prepare(parent, child)
    assert load(child / "FROZEN.json")["budget"] == 700
    assert (child / "predictions/good").is_symlink()
    assert (child / "predictions/block").is_symlink()
    assert not (child / "predictions/retry").exists()
    assert (parent / "predictions/retry/result.json").read_bytes() == before
    assert (parent / "DONE.json").exists() and (child / "PAUSE").exists()
    with pytest.raises(ValueError, match="allocated"):
        module.prepare(parent, tmp_path / "second_child")


def test_parent_changes_pause_the_child(tmp_path):
    parent, child = tmp_path / "parent", tmp_path / "child"
    parent.mkdir()
    child.mkdir()
    ledger(parent / "campaign.sqlite", [("ok", 100, 0, 200)])
    save(child / "CONTINUATION.json", {"parent": str(parent), "parent_debit": module.debit(parent / "campaign.sqlite"), "original_shared_cap": 1000})
    ledger(child / "campaign.sqlite", [("ok", 20, 0, 150)])
    assert module.cumulative_report(child)["remaining"] == 880
    with sqlite3.connect(parent / "campaign.sqlite") as db:
        db.execute("INSERT INTO attempts VALUES('ok',30,0,100)")
    with pytest.raises(ValueError, match="Parent charges changed"):
        module.cumulative_report(child)
    assert (child / "PAUSE").exists()
