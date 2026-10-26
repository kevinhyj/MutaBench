"""Offline fault injection and scientific isolation contracts for scale-v3."""
import json
import re
import sqlite3
from concurrent.futures import ThreadPoolExecutor

import pytest

from evagent.fewshot.data import Landscape
from evagent.fewshot.protocol import Condition, make_request
from evagent.fewshot.runner import load
from evagent.fewshot.scale import campaign, watchdog
from evagent.fewshot.scale.execution import Executor, Journal, StopCampaign, reserve_for
from evagent.fewshot.scale.prepare import freeze
from evagent.fewshot.scale.workflow import (
    acquire,
    build_task,
    cached,
    measured_support,
    requests,
    task_requests,
)
from evagent.integrated.gateway import ProviderStreamInterrupted

MODEL = {"model": "offline-fixture", "thinking_level": "medium", "max_output_tokens": 1024,
         "temperature": 0, "generation_seed": 1, "retries": 0}


@pytest.fixture
def bundle(tmp_path):
    meta = {"assay_id": "fixture", "domain": "rna", "wt_sequence": "A" * 10,
            "assay_description": "Measured cleavage", "endpoint": "cleavage", "readout": "fraction",
            "higher_is_better": True}
    records = tuple({"sequence": "".join("AC"[int(v)] for v in f"{i:010b}"),
                     "fitness": float(i), "mutation": str(i), "mutation_order": i.bit_count()}
                    for i in range(1, 800))
    freeze(Landscape(meta, records, {}), tmp_path / "inputs/fixture")
    return tmp_path


def one_request(root, arm="anchored32"):
    p = load(root / "inputs/fixture/public.json")
    return task_requests(build_task(root, "fixture", "static", arm, selected=p["random_order"][:32]), MODEL)[0]


def good_response(request):
    text = request["messages"][-1]["content"]
    ids = re.search(r"Required IDs: ([M\d, ]+)", text).group(1).split(", ")
    return {"final_text": json.dumps({"variants": [{"variant_id": i, "predicted_score": n / 25}
                                                   for n, i in enumerate(ids)]}),
            "finish_reason": "STOP", "usage_summary": {"input_tokens": 80, "output_tokens": 20,
            "total_tokens": 110, "reasoning_tokens": 10, "model_calls": 1}}


def checker(system, user):
    return {"input_tokens": len(user), "limit": 1_000_000, "basis": "offline"}


def test_freeze_is_label_independent_and_disjoint(bundle):
    p = load(bundle / "inputs/fixture/public.json")
    oracle = load(bundle / "inputs/fixture/oracle.json")
    truth = load(bundle / "inputs/fixture/evaluation.json")
    assert len(p["test"]) == len(p["pool"]) == 250
    assert set(oracle).isdisjoint(truth)
    assert not ({r["sequence"] for r in p["test"]} & {r["sequence"] for r in p["pool"]})
    assert "fitness" not in json.dumps(p)
    selected = p["random_order"][:4]
    sup = measured_support(p, oracle, selected)
    assert [s["fitness"] for s in sup] == [oracle[i] for i in selected]
    assert [s["acquisition_id"] for s in sup] == selected


def test_static_random32_and_initial_predictions_exactly_reuse(bundle):
    p = load(bundle / "inputs/fixture/public.json")
    static = build_task(bundle, "fixture", "static", "anchored32", selected=p["random_order"][:32])
    final = build_task(bundle, "fixture", "round/random", "32", selected=p["random_order"][:32])
    assert [r["request_hash"] for r in task_requests(static, MODEL)] == [r["request_hash"] for r in task_requests(final, MODEL)]
    a = build_task(bundle, "fixture", "round/random", "4", selected=p["random_order"][:4])
    b = build_task(bundle, "fixture", "round/greedy", "4", selected=p["random_order"][:4])
    assert [r["request_hash"] for r in task_requests(a, MODEL)] == [r["request_hash"] for r in task_requests(b, MODEL)]
    assert len(task_requests(a, MODEL)) == 20


def test_fixed_test_chunks_and_all_supports_no_label_leak(bundle):
    p = load(bundle / "inputs/fixture/public.json")
    chunks = []
    for k in (4, 8, 16, 32):
        task = build_task(bundle, "fixture", "round/random", str(k), selected=p["random_order"][:k])
        for q in task["test"]:
            q["fitness"] = "HIDDEN_SENTINEL"
        reqs = task_requests(task, MODEL)
        chunks.append([r["expected_ids"] for r in reqs if r["role"] == "test"])
        for r in reqs:
            assert len(r["support_ids"]) == k
            assert f"S{k:03}" in r["user"]
            assert "HIDDEN_SENTINEL" not in r["user"]
    assert all(c == chunks[0] for c in chunks)


@pytest.mark.parametrize("no_wt", [False, True])
def test_anchored_scale_support_only_and_raw_legacy_equivalence(bundle, no_wt):
    p = load(bundle / "inputs/fixture/public.json")
    sup = measured_support(p, load(bundle / "inputs/fixture/oracle.json"), p["random_order"][:32])
    meta = dict(p["metadata"], wt_sequence=None if no_wt else p["metadata"]["wt_sequence"])
    req = requests(meta, p["test"], sup, "raw32", "identity", MODEL)[0]
    legacy = make_request(dict(meta, episode_id="identity", query=p["test"]),
                          {"episode_id": "identity", "examples": sup}, Condition("numeric_32", 32, "numeric"), 0)
    assert req["user"] == legacy["user"] and req["system"] == legacy["system"]
    anchored = requests(meta, p["test"], sup, "anchored32", "identity", MODEL)[0]
    assert "atan" in anchored["user"] and "NOT neutrality" in anchored["user"]
    assert len(anchored["support_ids"]) == 32


def test_greedy_uses_only_complete_pool_predictions(bundle):
    p = load(bundle / "inputs/fixture/public.json")
    selected = p["random_order"][:4]
    predictions = {i: n for n, i in enumerate(p["random_order"]) if i not in selected}
    new = acquire(p, selected, 8, "greedy", predictions)
    assert new[:4] == selected
    assert new[4:] == list(reversed(p["random_order"]))[:4]
    with pytest.raises(ValueError):
        acquire(p, selected, 8, "greedy", {})
    assert acquire(p, selected, 8, "random") == p["random_order"][:8]


def test_duplicate_requests_have_one_paid_attempt(bundle):
    j = Journal(bundle / "run", 1_000_000, "fixture")
    e = Executor(j, MODEL, checker, transport=good_response, retry_wait=0)
    r = one_request(bundle)
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda _: e.run(r, 1), range(8)))
    assert j.summary()["attempts"] == 1
    assert j.summary()["known_tokens"] == 110
    assert cached(j.root, r, "offline_test")["status"] == "ok"
    with pytest.raises(ValueError):
        cached(j.root, r)


def test_network_retry_preserves_unknown_reservations(bundle):
    j = Journal(bundle / "run", 1_000_000, "fixture")
    calls = []
    def transport(req):
        calls.append(req)
        if len(calls) < 3:
            raise ProviderStreamInterrupted({"response_metadata": {"transport_error": {"exception_type": "ReadError"}}})
        return good_response(req)
    e = Executor(j, MODEL, checker, transport=transport, retry_wait=0)
    r = one_request(bundle)
    e.run(r, 1)
    summary = j.summary()
    assert len(calls) == 3 and summary["known_tokens"] == 110
    assert summary["unknown_reserve"] == 2 * reserve_for(r["system"], r["user"], MODEL)
    assert summary["inflight"] == 0


def test_retry_cap_and_independent_request_continuation(bundle):
    j = Journal(bundle / "run", 1_000_000, "fixture")
    def fail(req):
        raise ProviderStreamInterrupted({"response_metadata": {"transport_error": {"status_code": 503}}})
    r = one_request(bundle)
    e = Executor(j, MODEL, checker, transport=fail, retry_wait=0)
    e.run(r, 1)
    assert j.summary()["attempts"] == 3
    assert cached(j.root, r, "offline_test")["status"] == "transport_attempts_exhausted"
    e.transport = good_response
    e.run(one_request(bundle, "zero"), 2)
    assert j.summary()["attempts"] == 4


def test_permanent_block_not_retried_and_unknown_reserve_kept(bundle):
    j = Journal(bundle / "run", 1_000_000, "fixture")
    e = Executor(j, MODEL, checker, transport=lambda req: {
        "response_metadata": {"prompt_feedback": {"block_reason": "OTHER"}}}, retry_wait=0)
    r = one_request(bundle)
    e.run(r, 1)
    assert j.summary()["attempts"] == 1 and j.summary()["unknown_reserve"] > 0
    assert cached(j.root, r, "offline_test")["status"] == "blocked"


def test_concurrent_budget_admission_pause_and_snapshot(tmp_path):
    j = Journal(tmp_path, 100, "fixture")
    def admit(n):
        try:
            return j.admit(str(n), str(n), 40, 1)
        except StopCampaign:
            return None
    with ThreadPoolExecutor(max_workers=8) as pool:
        ids = list(pool.map(admit, range(8)))
    assert len([i for i in ids if i]) == 2
    assert j.summary()["total_debit"] == 80
    assert (tmp_path / "STOP.json").exists()
    with sqlite3.connect(j.snapshot()) as db:
        assert db.execute("PRAGMA integrity_check").fetchone()[0] == "ok"


def test_manual_pause_stops_without_submitting(bundle):
    j = Journal(bundle / "run", 1_000_000, "fixture")
    (j.root / "PAUSE").touch()
    e = Executor(j, MODEL, checker, transport=good_response)
    with pytest.raises(StopCampaign):
        e.run(one_request(bundle), 1)
    assert j.summary()["attempts"] == 0


def test_process_exit_recovers_completed_leaf_without_call(bundle):
    j = Journal(bundle / "run", 1_000_000, "fixture")
    r = one_request(bundle)
    e = Executor(j, MODEL, checker, transport=good_response, retry_wait=0)
    first = e.generate(r, r["user"], 1)
    j.db.execute("UPDATE attempts SET status='inflight',unknown=1,tokens=0")
    j.db.commit()
    assert j.recover() == [1]
    e.transport = lambda _: pytest.fail("Must recover, not regenerate")
    second = e.generate(r, r["user"], 2)
    assert first == second and j.summary()["known_tokens"] == 110


def test_interrupted_leaf_keeps_reservation_and_is_retryable(tmp_path):
    j = Journal(tmp_path, 1_000_000, "fixture")
    aid = j.admit("logical", "prompt", 1234, 1)
    leaf = j.leaf(aid)
    leaf.mkdir(parents=True)
    with sqlite3.connect(leaf / "request_ledger.sqlite") as db:
        db.execute("CREATE TABLE attempts(id INTEGER, status TEXT, response_json TEXT,tokens INTEGER,error_type TEXT)")
        db.execute("INSERT INTO attempts VALUES(1,'inflight',NULL,NULL,NULL)")
    j.recover()
    assert j.summary()["unknown_reserve"] == 1234
    assert j.history("logical")[0]["retryable"] == 1


def test_adaptive_workers_drain_and_restore(tmp_path):
    j = Journal(tmp_path, 1_000_000, "fixture")
    assert j.concurrency() == 8
    batch = j.start_batch(8)
    for n in range(20):
        j.admit(str(n), str(n), 100, batch)
    with pytest.raises(ValueError):
        j.start_batch(15)
    j.db.execute("UPDATE attempts SET status='ok',unknown=0,tokens=10")
    j.db.commit()
    j.end_batch(batch)
    assert j.concurrency() == 15
    batch2 = j.start_batch(15)
    j.admit("error", "error", 100, batch2)
    j.db.execute("UPDATE attempts SET status='failed_unknown_cost',retryable=1,http_code=429 WHERE logical='error'")
    j.db.commit()
    j.end_batch(batch2)
    assert j.concurrency() == 7


def test_integrity_stop_on_source_change(tmp_path, monkeypatch):
    path = tmp_path / "source.py"
    path.write_text("original")
    monkeypatch.setattr(campaign, "source_files", lambda: [path])
    from evagent.fewshot.protocol import digest
    from evagent.fewshot.runner import sha
    config = {"source_hashes": {str(path): sha(path)}, "input_hashes": {}, "dependencies": {}}
    config["fingerprint"] = digest(config)
    campaign.verify(tmp_path, config)
    path.write_text("changed")
    with pytest.raises(ValueError, match="source"):
        campaign.verify(tmp_path, config)


def test_watchdog_rejects_reused_pid_and_wrong_command(tmp_path, monkeypatch):
    record = {"pid": 9, "start_ticks": "111"}
    monkeypatch.setattr(watchdog, "identity", lambda pid: {"pid": pid, "start_ticks": "222", "state": "S", "command": []})
    assert not watchdog.alive(record, tmp_path)
    monkeypatch.setattr(watchdog, "identity", lambda pid: {"pid": pid, "start_ticks": "111", "state": "S", "command": [b"wrong"]})
    assert not watchdog.alive(record, tmp_path)


def test_cron_preserves_other_jobs_and_never_restarts_pause(tmp_path, monkeypatch):
    from types import SimpleNamespace
    written = []
    def command(args, **kw):
        if args == ["crontab", "-l"]:
            return SimpleNamespace(returncode=0, stdout="17 2 * * * existing-backup\n")
        written.append(kw["input"])
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(watchdog.subprocess, "run", command)
    watchdog.cron(tmp_path, True)
    assert "existing-backup" in written[0] and "* * * * *" in written[0]
    (tmp_path / "PAUSE").touch()
    monkeypatch.setattr(watchdog.subprocess, "Popen", lambda *a, **k: pytest.fail("PAUSE must not launch"))
    watchdog.check(tmp_path)


def test_worker_lifetime_lock_prevents_duplicate_without_pid_record(tmp_path, monkeypatch):
    import fcntl
    lease = (tmp_path / "worker.lock").open("a+")
    fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
    monkeypatch.setattr(watchdog.subprocess, "Popen", lambda *a, **k: pytest.fail("Duplicate worker"))
    watchdog.check(tmp_path)
    lease.close()


def test_full_feedback_replay_and_paired_report(bundle):
    from evagent.fewshot.protocol import digest
    from evagent.fewshot.runner import save
    from evagent.fewshot.scale.report import report
    from evagent.fewshot.scale.workflow import scan_phase
    save(bundle / "COHORT.json", {"assays": [{"metadata": {"assay_id": "fixture", "domain": "rna"},
         "family": "fixture", "study": "fixture", "development": False, "development_background": False,
         "multi_round_eligible": True, "extrapolation": []}]})
    submitted = []
    for phase in ("static", "rounds"):
        for _ in range(20):
            pending, _ = scan_phase(bundle, ["fixture"], phase, MODEL)
            if not pending:
                break
            for r in pending:
                # Synthetic artifacts stay in pytest's temporary directory only.
                result = {"request_hash": r["request_hash"], "execution_mode": "live", "status": "ok",
                          "value": {v: int(v[1:]) / 1000 for v in r["expected_ids"]}}
                result["result_digest"] = digest(result)
                path = bundle / "predictions" / r["request_hash"]
                save(path / "request.json", r)
                save(path / "result.json", result)
                submitted.append(r["request_hash"])
        else:
            pytest.fail("Feedback dependencies did not terminate")
    assert len(submitted) == len(set(submitted))
    assert len(list((bundle / "tasks").rglob("result.json"))) == 11
    r32 = load(bundle / "tasks/fixture/round/random/32/result.json")
    static = load(bundle / "tasks/fixture/static/anchored32/result.json")
    assert r32["requests"] == static["requests"]
    oracle = load(bundle / "inputs/fixture/oracle.json")
    for path in (bundle / "tasks/fixture/round").rglob("inputs.json"):
        task = load(path)
        assert all(s["fitness"] == oracle[s["acquisition_id"]] for s in task["support"])
    j = Journal(bundle, 1_000_000, "fixture")
    progress = report(bundle, j.summary(), "complete")
    assert progress["tasks"] == {"complete": 11}
    metrics = load(bundle / "analysis/metrics.json")
    assert len(metrics["paired_rounds"]) == 4
    assert (bundle / "analysis/REPORT.md").exists()


def test_systemd_fallback_disables_only_own_timer(tmp_path, monkeypatch):
    from evagent.fewshot.runner import save
    calls = []
    save(tmp_path / "scheduler.json", {"backend": "systemd_user_timer", "timer": "own-experiment.timer"})
    monkeypatch.setattr(watchdog.subprocess, "run", lambda args, **kw: calls.append(args))
    watchdog.cron(tmp_path, False)
    assert calls == [["systemctl", "--user", "disable", "--now", "own-experiment.timer"]]


def test_systemd_watchdog_starts_worker_without_local_ssh_parent(tmp_path, monkeypatch):
    from evagent.fewshot.runner import save
    calls = []
    save(tmp_path / "scheduler.json", {"backend": "systemd_user_timer", "worker": "own-worker.service"})
    save(tmp_path / "FROZEN.json", {})
    monkeypatch.setattr(watchdog.subprocess, "run", lambda args, **kw: calls.append(args))
    monkeypatch.setattr(watchdog.subprocess, "Popen", lambda *a, **kw: pytest.fail("Must use server manager"))
    watchdog.check(tmp_path)
    assert calls == [["systemctl", "--user", "start", "--no-block", "own-worker.service"]]
