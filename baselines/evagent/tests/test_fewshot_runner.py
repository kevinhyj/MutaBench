"""Offline contracts for measured-reference prompts and the real request ledger."""

from __future__ import annotations

import copy
import io
import json
import re
import sqlite3
import threading
import time
import urllib.request
from types import SimpleNamespace

import pytest

from evagent.fewshot import runner
from evagent.fewshot.protocol import CONDITIONS, all_requests, make_request, measured_examples
from evagent.runtime import ExecutionPaused

MODEL = {
    "model": "offline-fewshot-model",
    "thinking_level": "medium",
    "max_output_tokens": 1024,
    "temperature": 0.0,
    "retries": 0,
}


def dna(number, alphabet="AT"):
    return "".join(alphabet[int(bit)] for bit in f"{number:08b}")


@pytest.fixture
def episode_files(tmp_path):
    episode = {
        "episode_id": "example-random-seed1",
        "assay_id": "offline_assay",
        "domain": "rna",
        "assay_description": "An offline functional-activity fixture",
        "endpoint": "cleavage activity",
        "readout": "fraction cleaved",
        "higher_is_better": True,
        "wt_sequence": "AAAAAAAA",
        "query": [
            {"variant_id": f"M{i + 1:03}", "mutation": f"A{i + 1}T", "sequence": dna(i)}
            for i in range(25)
        ],
    }
    support = {
        "episode_id": episode["episode_id"],
        "examples": [
            {"variant_id": f"S{i + 1:03}", "mutation": f"C{i + 1}G",
             "sequence": dna(i, "CG"), "fitness": 0.001 * (i + 1)}
            for i in range(32)
        ],
    }
    folder = tmp_path / "episodes" / episode["episode_id"]
    runner.save(folder / "episode.json", episode)
    runner.save(folder / "support.json", support)
    # Deliberately unreadable hidden truth: prediction must never open it.
    (folder / "evaluation.json").write_text("THIS IS HIDDEN; DO NOT READ")
    return folder.parent, episode, support


def context_check(system, user):
    return {"input_tokens": len(system) + len(user), "limit": 100_000,
            "basis": "offline_test_only"}


def response(text):
    return {"final_text": text, "finish_reason": "STOP",
            "usage_summary": {"input_tokens": 8, "output_tokens": 2, "total_tokens": 10,
                              "reasoning_tokens": 0, "model_calls": 1}}


def valid_response(request):
    user = request["messages"][-1]["content"]
    required = re.search(r"Required IDs: ([M\d, ]+)", user).group(1).split(", ")
    return response(json.dumps({"variants": [
        {"variant_id": vid, "predicted_score": (index - 12) / 25}
        for index, vid in enumerate(required)
    ]}))


def execute(root, run_dir, transport, **kwargs):
    return runner.execute(root, run_dir, MODEL, token_budget=1_000_000,
                          transport=transport, context_check=context_check,
                          workers=kwargs.pop("workers", 1), **kwargs)


def results(run_dir):
    return [runner.load(path) for path in sorted((run_dir / "predictions").glob("*/result.json"))]


def test_selected_breadth_conditions_preserve_prompts_and_all_32_unlabeled_examples(episode_files, tmp_path):
    from evagent.fewshot.protocol import resolve_conditions
    root, episode, support = episode_files
    names = ['numeric_0','numeric_16','numeric_32','unlabeled_32']
    original = {r['condition']['name']:r for r in all_requests(episode,support)}
    selected = {r['condition']['name']:r for r in all_requests(episode,support,names)}
    for name in names[:3]:
        assert selected[name] == original[name]
    unlabeled = selected['unlabeled_32']
    assert len(unlabeled['support_ids']) == 32
    assert 'S032' in unlabeled['user']
    examples = measured_examples(episode,support,resolve_conditions(['unlabeled_32'])[0])
    assert len(examples)==32 and all('fitness' not in row for row in examples)
    run_dir = tmp_path/'breadth'
    result = execute(root,run_dir,valid_response,conditions=names)
    assert result['new_request_units']==4
    assert [c['name'] for c in runner.load(run_dir/'plan.json')['conditions']]==names
    with pytest.raises(ValueError,match='Frozen'):
        execute(root,run_dir,valid_response,conditions=['numeric_0'])


def test_all_shots_reach_every_chunk_and_query_labels_never_render(episode_files):
    _root, episode, support = episode_files
    episode["query"] += [
        {"variant_id": f"M{i + 1:03}", "mutation": f"A{i + 1}T", "sequence": dna(i)}
        for i in range(25, 50)
    ]
    episode["hidden_labels"] = {"M001": "hidden-label-sentinel"}
    for query in episode["query"]:
        query["fitness"] = "hidden-label-sentinel"
        query["experimental_score"] = 9876543.125
    requests = all_requests(episode, support)
    assert len(requests) == 16
    assert len({request["system"] for request in requests}) == 1
    for condition in CONDITIONS:
        by_chunk = [r for r in requests if r["condition"]["name"] == condition.name]
        assert len(by_chunk) == 2
        assert by_chunk[0]["support_ids"] == by_chunk[1]["support_ids"]
        assert len(by_chunk[0]["support_ids"]) == condition.shots
        for request in by_chunk:
            assert "hidden-label-sentinel" not in request["user"]
            assert "9876543.125" not in request["user"]
            assert request["expected_ids"] == [
                q["variant_id"] for q in episode["query"][request["chunk_index"] * 25:
                                                          request["chunk_index"] * 25 + 25]
            ]
            for row in support["examples"][:condition.shots]:
                assert row["sequence"] in request["user"]
            for row in support["examples"][condition.shots:]:
                assert row["sequence"] not in request["user"]
        assert make_request(episode, support, condition, 0) == by_chunk[0]
    # Candidate rows and the scoring contract remain identical across all arms.
    for chunk in range(2):
        assert len({r["user"].split("**25 candidate mutants to score**", 1)[1]
                    for r in requests if r["chunk_index"] == chunk}) == 1


def test_controls_share_sequences_and_do_not_leak_numeric_labels(episode_files):
    _root, episode, support = episode_files
    controls = {c.name: measured_examples(episode, support, c) for c in CONDITIONS}
    numeric = controls["numeric_16"]
    for name in ("unlabeled_16", "ordinal_16", "shuffled_16"):
        assert [(r["variant_id"], r["mutation"], r["sequence"]) for r in controls[name]] == [
            (r["variant_id"], r["mutation"], r["sequence"]) for r in numeric
        ]
    assert all(set(row) == {"variant_id", "mutation", "sequence"}
               for row in controls["unlabeled_16"])
    assert all("measured_value" not in row for row in controls["ordinal_16"])
    assert [r["measured_rank"] for r in controls["ordinal_16"]] == list(range(16, 0, -1))
    shuffled = [row["measured_value"] for row in controls["shuffled_16"]]
    ordered = [row["measured_value"] for row in numeric]
    assert sorted(shuffled) == sorted(ordered)
    assert shuffled != ordered
    episode["higher_is_better"] = False
    condition = next(c for c in CONDITIONS if c.name == "ordinal_16")
    assert [r["measured_rank"] for r in measured_examples(episode, support, condition)] == list(range(1, 17))


@pytest.mark.parametrize("overlap", ["variant_id", "sequence"])
def test_query_support_overlap_rejected(episode_files, overlap):
    _root, episode, support = episode_files
    support["examples"][0][overlap] = episode["query"][0][overlap]
    with pytest.raises(ValueError, match="overlap"):
        make_request(episode, support, CONDITIONS[1], 0)


def test_no_wt_and_no_silent_shot_reduction(episode_files):
    _root, episode, support = episode_files
    episode["wt_sequence"] = None
    request = make_request(episode, support, CONDITIONS[4], 0)
    assert "Unavailable; do not invent a WT." in request["user"]
    assert "relative to wild type and the assay definition" not in request["user"]
    support["examples"] = support["examples"][:31]
    with pytest.raises(ValueError, match="Insufficient"):
        make_request(episode, support, CONDITIONS[4], 0)


def test_real_gateway_offline_run_and_resume_use_cached_results(episode_files, tmp_path):
    root, _episode, _support = episode_files
    calls = []
    run_dir = tmp_path / "run"
    first = execute(root, run_dir, lambda request: calls.append(request) or valid_response(request))
    assert len(calls) == 8
    assert first["ledger"]["provider_attempts"] == 8
    assert first["ledger"]["reported_tokens"] == 80
    assert first["execution_mode"] == "offline_test"
    assert all(row["status"] == "ok" and len(row["value"]) == 25 for row in results(run_dir))
    second = execute(root, run_dir, lambda request: pytest.fail("Resume generated a new request"))
    assert second["new_request_units"] == 0
    assert second["ledger"]["provider_attempts"] == 8
    assert all("hidden" not in json.dumps(call["metadata"]) for call in calls)


def test_missing_result_resumes_from_durable_provider_cache(episode_files, tmp_path):
    root, _episode, _support = episode_files
    run_dir = tmp_path / "run"
    execute(root, run_dir, valid_response)
    missing = next((run_dir / "predictions").glob("*/result.json"))
    missing.unlink()
    resumed = execute(root, run_dir, lambda request: pytest.fail("Ledger cache was bypassed"))
    assert resumed["new_request_units"] == 1
    assert resumed["ledger"]["provider_attempts"] == 8
    assert runner.load(missing)["attempts"][0]["cache_hit"]


def test_context_limit_prevents_every_generation_without_truncating(episode_files, tmp_path):
    root, _episode, _support = episode_files
    checked = []

    def too_long(system, user):
        checked.append(user)
        raise runner.ContextTooLong("offline over-limit fixture")

    run_dir = tmp_path / "run"
    summary = runner.execute(root, run_dir, MODEL, token_budget=1_000_000,
                             transport=lambda _: pytest.fail("Over-limit request reached provider"),
                             context_check=too_long, workers=1)
    assert len(checked) == 8
    assert summary["ledger"]["provider_attempts"] == 0
    assert summary["ledger"]["reported_tokens"] == 0
    assert all(row["status"] == "context_limit_exceeded" for row in results(run_dir))
    largest = next(r for r in all_requests(_episode, _support) if r["condition"]["shots"] == 32)
    assert largest["user"] in checked


def test_repair_context_failure_preserves_initial_attempt_cost(episode_files, tmp_path):
    root, _episode, _support = episode_files
    checks = []

    def repair_too_long(system, user):
        checks.append(user)
        if len(checks) == 2:
            raise runner.ContextTooLong("repair does not fit")
        return context_check(system, user)

    run_dir = tmp_path / "run"
    summary = runner.execute(root, run_dir, MODEL, token_budget=1_000_000,
                             transport=lambda _: response("not valid JSON"),
                             context_check=repair_too_long, max_units=1, workers=1)
    assert summary["ledger"]["provider_attempts"] == 1
    assert summary["ledger"]["reported_tokens"] == 10
    record = results(run_dir)[0]
    assert record["status"] == "context_limit_exceeded"
    assert len(record["attempts"]) == 1
    assert record["usage_summary"]["total_tokens"] == 10


def test_unknown_cost_pauses_and_cannot_retry_on_resume(episode_files, tmp_path):
    root, _episode, _support = episode_files
    calls = []

    def ambiguous(request):
        calls.append(request)
        return {"final_text": "unknown", "finish_reason": "STOP"}

    run_dir = tmp_path / "run"
    with pytest.raises(ExecutionPaused):
        execute(root, run_dir, ambiguous)
    status = runner.load(run_dir / "status.json")
    assert status["status"] == "paused"
    assert status["ledger"]["unknown_usage_attempts"] == 1
    assert status["ledger"]["usage_is_lower_bound"]
    with pytest.raises(ExecutionPaused):
        execute(root, run_dir, lambda _: pytest.fail("Ambiguous request was retried"))
    assert len(calls) == 1
    with sqlite3.connect(run_dir / "provider/request_ledger.sqlite") as db:
        assert db.execute("SELECT COUNT(*) FROM attempts").fetchone()[0] == 1


@pytest.mark.parametrize("filename", ["episode.json", "support.json"])
def test_changed_frozen_inputs_refuse_resume(episode_files, tmp_path, filename):
    root, episode, _support = episode_files
    run_dir = tmp_path / "run"
    execute(root, run_dir, valid_response, max_units=1)
    path = root / episode["episode_id"] / filename
    changed = runner.load(path)
    if filename == "episode.json":
        changed["endpoint"] = "different endpoint"
    else:
        changed["examples"][0]["fitness"] += 1
    runner.save(path, changed)
    with pytest.raises(ValueError, match="changed"):
        execute(root, run_dir, lambda _: pytest.fail("Changed input reached provider"))


def test_changed_model_refuses_same_run_directory(episode_files, tmp_path):
    root, _episode, _support = episode_files
    run_dir = tmp_path / "run"
    execute(root, run_dir, valid_response, max_units=1)
    changed = {**MODEL, "temperature": 0.2}
    with pytest.raises(ValueError, match="changed"):
        runner.execute(root, run_dir, changed, token_budget=1_000_000,
                       transport=lambda _: pytest.fail("Changed model reached provider"),
                       context_check=context_check, workers=1)


def test_offline_predictions_cannot_be_promoted_to_live_on_resume(episode_files, tmp_path, monkeypatch):
    root, _episode, _support = episode_files
    run_dir = tmp_path / "run"
    execute(root, run_dir, valid_response, max_units=1)
    monkeypatch.setattr(runner, "GoogleContextCheck", lambda *args: pytest.fail(
        "Live context checking started before rejecting offline artifacts"))
    # max_units=0 guarantees even the pre-fix implementation cannot make a live call.
    with pytest.raises(ValueError, match="mode|offline|live"):
        runner.execute(root, run_dir, MODEL, token_budget=1_000_000, live=True, max_units=0, workers=1)
    assert results(run_dir)[0]["execution_mode"] == "offline_test"


def test_protocol_projects_only_fields_not_arbitrary_input_objects(episode_files):
    _root, episode, support = episode_files
    clean = make_request(episode, support, CONDITIONS[1], 0)
    injected = copy.deepcopy(episode)
    injected["query"][0]["private_label_object"] = object()
    injected["truth"] = {"must_never_be_serialized": object()}
    support["examples"][0]["other_assay_scores"] = object()
    assert make_request(injected, support, CONDITIONS[1], 0) == clean


def test_google_context_rest_counts_system_and_caches_without_credentials(tmp_path, monkeypatch):
    captured = []
    responses = iter([{"inputTokenLimit": 100, "outputTokenLimit": 2048},
                      {"totalTokens": 100}, {"totalTokens": 101}])

    def open_request(request, *, timeout):
        captured.append(request)
        assert timeout == 60
        return io.BytesIO(json.dumps(next(responses)).encode())

    monkeypatch.setenv("GEMINI_API_KEY", "test-credential-never-persist")
    monkeypatch.setattr(urllib.request, "build_opener", lambda *args: SimpleNamespace(open=open_request))
    checker = runner.GoogleContextCheck(MODEL, tmp_path / "token_checks")
    first = checker("full system instruction", "all support and query sequences")
    assert first["input_tokens"] == 100
    assert checker("full system instruction", "all support and query sequences") == first
    assert len(captured) == 2
    request = captured[1]
    assert request.full_url.endswith("/v1beta/models/offline-fewshot-model:countTokens")
    assert request.get_header("X-goog-api-key") == "test-credential-never-persist"
    payload = json.loads(request.data)
    assert payload == {"generateContentRequest": {
        "model": "models/offline-fewshot-model",
        "systemInstruction": {"parts": [{"text": "full system instruction"}]},
        "contents": [{"role": "user", "parts": [{"text": "all support and query sequences"}]}],
    }}
    with pytest.raises(runner.ContextTooLong, match="no examples or sequences truncated"):
        checker("different system instruction", "all support and query sequences")
    assert len(captured) == 3
    with pytest.raises(runner.ContextTooLong):
        checker("different system instruction", "all support and query sequences")
    assert len(captured) == 3
    persisted = "\n".join(path.read_text() for path in (tmp_path / "token_checks").glob("*.json"))
    assert "test-credential-never-persist" not in persisted
    assert "X-goog-api-key" not in persisted
    assert "full system instruction" not in persisted
    checker.close()


@pytest.mark.parametrize("count", [None, -1, True, 1.5])
def test_google_context_rest_rejects_missing_or_invalid_counts(tmp_path, monkeypatch, count):
    responses = iter([{"inputTokenLimit": 100, "outputTokenLimit": 2048}, {"totalTokens": count}])
    monkeypatch.setenv("GEMINI_API_KEY", "offline-test-only")
    monkeypatch.setattr(urllib.request, "build_opener", lambda *args: SimpleNamespace(
        open=lambda *args, **kwargs: io.BytesIO(json.dumps(next(responses)).encode())))
    checker = runner.GoogleContextCheck(MODEL, tmp_path / "counts")
    with pytest.raises(ValueError, match="count unavailable"):
        checker("system", "user")
    assert [p.name for p in (tmp_path / "counts").iterdir()] == ["model_limits.json"]


@pytest.mark.parametrize('workers', [4, 64])
def test_workers_share_ledger_and_bound_inflight_calls(episode_files, tmp_path, workers):
    root, episode, _support = episode_files
    if workers == 64:
        episode['query'] = [{"variant_id": f"M{i+1:03}", "mutation": f"A{i+1}T", "sequence": dna(i)} for i in range(200)]
        runner.save(root / episode['episode_id'] / 'episode.json', episode)
    expected_calls = 64 if workers == 64 else 8
    lock = threading.Lock()
    rendezvous = threading.Barrier(workers)
    active = peak = calls = 0

    def concurrent_response(request):
        nonlocal active, peak, calls
        with lock:
            active += 1
            calls += 1
            peak = max(peak, active)
        rendezvous.wait(timeout=20)
        result = valid_response(request)
        with lock:
            active -= 1
        return result

    run_dir = tmp_path / "concurrent_run"
    summary = execute(root, run_dir, concurrent_response, workers=workers)
    assert calls == expected_calls
    assert peak == workers
    assert active == 0
    assert summary["workers"] == workers
    assert summary["new_request_units"] == expected_calls
    assert summary["ledger"]["reported_tokens"] == expected_calls * 10
    assert summary["ledger"]["provider_attempts"] == expected_calls
    assert summary["ledger"]["peak_concurrent_requests"] == workers
    assert summary["ledger"]["inflight_requests"] == 0
    assert not list(run_dir.rglob("*.tmp"))
    resumed = execute(root, run_dir, lambda _: pytest.fail("Concurrent resume made a new call"), workers=workers)
    assert resumed["new_request_units"] == 0
    assert resumed["ledger"]["reported_tokens"] == expected_calls * 10
    with pytest.raises(ValueError, match="workers"):
        execute(root, run_dir, valid_response, workers=2)


def test_concurrent_unknown_cost_stops_new_admission_and_settles_inflight(episode_files, tmp_path):
    root, _episode, _support = episode_files
    lock = threading.Lock()
    rendezvous = threading.Barrier(4)
    calls = completed = 0

    def fail_one(request):
        nonlocal calls, completed
        with lock:
            ordinal = calls
            calls += 1
        rendezvous.wait(timeout=3)
        if ordinal == 0:
            return {"final_text": "ambiguous", "finish_reason": "STOP"}
        time.sleep(0.05)
        with lock:
            completed += 1
        return valid_response(request)

    run_dir = tmp_path / "concurrent_run"
    with pytest.raises(ExecutionPaused):
        execute(root, run_dir, fail_one, workers=4)
    assert calls == 4
    assert completed == 3
    summary = runner.load(run_dir / "status.json")
    assert summary["status"] == "paused"
    assert summary["new_request_units"] == 3
    assert summary["ledger"]["reported_tokens"] == 30
    assert summary["ledger"]["unknown_usage_attempts"] == 1
    assert summary["ledger"]["inflight_requests"] == 0
    assert len(results(run_dir)) == 3
    with pytest.raises(ExecutionPaused):
        execute(root, run_dir, lambda _: pytest.fail("Paused concurrent run admitted another call"), workers=4)


def test_concurrent_budget_reserves_for_inflight_requests(episode_files, tmp_path):
    root, _episode, _support = episode_files
    lock = threading.Lock()
    calls = 0
    requests = all_requests(_episode, _support)
    reserves = [len((r["system"] + r["user"]).encode()) + 2 * MODEL["max_output_tokens"] + 1024
                for r in requests]
    budget = max(reserves) + 1
    assert 2 * min(reserves) > budget

    def slow_response(request):
        nonlocal calls
        with lock:
            calls += 1
        time.sleep(0.05)
        return valid_response(request)

    run_dir = tmp_path / "budget_run"
    with pytest.raises(ExecutionPaused):
        runner.execute(root, run_dir, MODEL, token_budget=budget, transport=slow_response,
                       context_check=context_check, workers=4)
    summary = runner.load(run_dir / "status.json")
    assert calls == 1
    assert summary["ledger"]["provider_attempts"] == 1
    assert summary["ledger"]["reported_tokens"] == 10
    assert summary["ledger"]["inflight_requests"] == 0
    assert summary["ledger"]["reported_tokens"] <= budget
    assert summary["ledger"]["paused"]


def test_concurrent_max_units_does_not_over_admit(episode_files, tmp_path):
    root, _episode, _support = episode_files
    summary = execute(root, tmp_path / "limited", valid_response, workers=4, max_units=3)
    assert summary["status"] == "unit_limit_reached"
    assert summary["new_request_units"] == 3
    assert summary["ledger"]["provider_attempts"] == 3


def test_episode_schedule_interleaves_assays_within_seed_and_split(tmp_path):
    for seed in (2, 1):
        for split in ("low_order_to_high_order", "random"):
            for assay in ("RNA", "Protein"):
                identity = f"{assay}-{split}-{seed}"
                folder = tmp_path / identity
                runner.save(folder / "episode.json", {"episode_id": identity, "assay_id": assay,
                                                      "seed": seed, "split": split})
                runner.save(folder / "support.json", {"episode_id": identity})
    order = [(episode["seed"], episode["split"], episode["assay_id"])
             for _folder, episode, _support in runner.episodes(tmp_path)]
    assert order == [(seed, split, assay) for seed in (1, 2)
                     for split in ("random", "low_order_to_high_order")
                     for assay in ("Protein", "RNA")]


def test_execution_rechecks_bytes_before_later_episode(episode_files, tmp_path):
    root, episode, support = episode_files
    next_episode = dict(episode, episode_id="next-episode", seed=2)
    next_support = copy.deepcopy(support)
    next_support["episode_id"] = next_episode["episode_id"]
    folder = root / next_episode["episode_id"]
    runner.save(folder / "episode.json", next_episode)
    runner.save(folder / "support.json", next_support)
    calls = 0
    run_dir = tmp_path / "changing_input"

    def mutate_later_episode(request):
        nonlocal calls
        assert runner.load(run_dir / "status.json")["status"] == "running"
        calls += 1
        if calls == 1:
            next_support["examples"][0]["fitness"] += 100
            runner.save(folder / "support.json", next_support)
        return valid_response(request)

    with pytest.raises(ValueError, match="changed during execution"):
        execute(root, run_dir, mutate_later_episode)
    assert calls == 8
    status = runner.load(run_dir / "status.json")
    assert status["status"] == "error"
    assert status["new_request_units"] == 8
