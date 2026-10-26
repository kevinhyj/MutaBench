"""Offline admission, unknown-cost preservation, and provider contract tests."""

import json
from types import SimpleNamespace

import pytest

from evagent.integrated.gateway import (
    ProviderStreamInterrupted,
    RequestGateway,
    public_model_config,
)
from evagent.runtime import ExecutionPaused

MODEL = {"model": "gemini-test", "thinking_level": "high"}


def response(**changes):
    return {"final_text": '{"ranking":["v1"]}', "finish_reason": "STOP",
            "usage_summary": {"input_tokens": 8, "output_tokens": 2, "total_tokens": 10,
                              "reasoning_tokens": 0, "model_calls": 1},
            "response_metadata": {}, "thought_summary": "", **changes}


def gateway(tmp_path, transport, **kwargs):
    return RequestGateway(model_config=MODEL, run_dir=tmp_path, fingerprint="frozen-v1",
                          token_budget=kwargs.pop("token_budget", 100), transport=transport,
                          **kwargs)


def generate(instance, user="task", **kwargs):
    return instance.generate(system="system", user=user, run_name="offline", **kwargs)


def test_cache_actual_prompt_hash_and_all_attempts(tmp_path):
    calls = []
    instance = gateway(tmp_path, lambda request: calls.append(request) or response())
    first = generate(instance)
    assert generate(instance)["cache_hit"]
    second = generate(instance, "different")
    assert first["request_sha256"] != second["request_sha256"]
    assert calls[0]["model"] == MODEL["model"]
    assert calls[0]["thinking_level"] == "high"
    assert instance.summary()["provider_attempts"] == 2
    assert instance.summary()["reported_tokens"] == 20
    instance.close()
    resumed = gateway(tmp_path, lambda _: pytest.fail("A cached request must not call provider"))
    assert generate(resumed)["cache_hit"]
    resumed.close()


@pytest.mark.parametrize("bad", [
    None, {}, response(finish_reason=None), response(finish_reason="UNRECOGNIZED_NEW_STATE"),
    response(final_text=""), response(usage_summary=None), response(cost_known=False),
    response(usage_summary={"input_tokens": 8, "output_tokens": 2, "total_tokens": None}),
    response(usage_summary={"input_tokens": 8, "output_tokens": 2, "total_tokens": 1}),
    response(usage_summary={"input_tokens": 8, "output_tokens": 2, "total_tokens": 10,
                            "model_calls": 2}),
])
def test_ambiguous_response_pauses_and_never_resubmits_after_restart(tmp_path, bad):
    calls = []
    instance = gateway(tmp_path, lambda request: calls.append(request) or bad)
    with pytest.raises(ExecutionPaused):
        generate(instance)
    with pytest.raises(ExecutionPaused):
        generate(instance, "new request")
    assert len(calls) == 1
    assert instance.summary()["paused"]
    assert instance.db.execute("SELECT response_json FROM attempts").fetchone()[0]
    instance.close()
    resumed = gateway(tmp_path, lambda _: pytest.fail("Unknown requests cannot be retried"))
    with pytest.raises(ExecutionPaused):
        generate(resumed)
    with pytest.raises(ExecutionPaused):
        generate(resumed, "still new")
    resumed.close()


def test_transport_exception_is_one_attempt_without_persisting_secret(tmp_path):
    def fail(_):
        raise ConnectionError("Bearer private-token must never be logged")
    instance = gateway(tmp_path, fail)
    with pytest.raises(ExecutionPaused):
        generate(instance)
    row = instance.db.execute("SELECT status,error_type,response_json FROM attempts").fetchone()
    assert row == ("failed_unknown_cost", "ConnectionError", None)
    assert "private-token" not in json.dumps(instance.summary())
    instance.close()


def test_interrupted_stream_retains_partial_output_and_reported_tokens(tmp_path):
    def fail(_):
        raise ProviderStreamInterrupted(response(final_text="partial answer", finish_reason=None))
    instance = gateway(tmp_path, fail)
    with pytest.raises(ExecutionPaused):
        generate(instance)
    saved = json.loads(instance.db.execute("SELECT response_json FROM attempts").fetchone()[0])
    assert saved["final_text"] == "partial answer"
    assert instance.summary()["reported_tokens"] == 10
    assert instance.summary()["paused"]
    instance.close()


@pytest.mark.parametrize("finish,status", [
    ("SAFETY", "blocked"), ("MAX_TOKENS", "truncated"), ("OTHER", "terminal_failure"),
])
def test_known_terminal_response_remains_terminal_and_cached(tmp_path, finish, status):
    calls = []
    instance = gateway(tmp_path, lambda req: calls.append(req) or response(finish_reason=finish))
    result = generate(instance)
    assert result["status"] == status
    assert generate(instance)["status"] == status
    assert len(calls) == 1
    assert instance.summary()["reported_tokens"] == 10
    instance.close()


def test_tool_response_never_becomes_valid_text(tmp_path):
    instance = gateway(tmp_path, lambda _: response(tool_calls=[{"name": "task"}]))
    assert generate(instance)["status"] == "terminal_failure"
    instance.close()


def test_soft_budget_admits_one_overage_then_stops(tmp_path):
    instance = gateway(tmp_path, lambda _: response(), token_budget=5)
    assert generate(instance)["status"] == "ok"
    assert generate(instance)["cache_hit"]
    with pytest.raises(ExecutionPaused):
        generate(instance, "second")
    assert instance.summary()["reported_tokens"] == 10
    instance.close()


def test_restart_detects_interrupted_attempt_without_overwriting(tmp_path):
    instance = gateway(tmp_path, lambda _: response())
    instance.db.execute(
        "INSERT INTO attempts(request_sha256,run_name,status,request_json,started) "
        "VALUES ('interrupted','test','inflight','{}',1)"
    )
    instance.db.commit()
    instance.close()
    resumed = gateway(tmp_path, lambda _: pytest.fail("Interrupted request must pause admission"))
    with pytest.raises(ExecutionPaused):
        generate(resumed)
    assert resumed.summary()["requests_by_status"] == {"interrupted_unknown_cost": 1}
    resumed.close()


def test_run_fingerprint_and_budget_cannot_be_reused(tmp_path):
    instance = gateway(tmp_path, lambda _: response())
    instance.close()
    for overrides in ({"fingerprint": "different"}, {"token_budget": 101},
                      {"model_config": {**MODEL, "thinking_level": "low"}}):
        with pytest.raises(ValueError):
            RequestGateway(**{"model_config": MODEL, "run_dir": tmp_path,
                              "fingerprint": "frozen-v1", "token_budget": 100,
                              "transport": lambda _: response(), **overrides})


def test_exclusive_run_lease_prevents_concurrent_admission(tmp_path):
    instance = gateway(tmp_path, lambda _: response())
    with pytest.raises(ExecutionPaused, match="Another generator"):
        gateway(tmp_path, lambda _: response())
    instance.close()


@pytest.mark.parametrize("metadata", [{"labels": [1]}, {"nested": {"api_key": "secret"}},
                                       {"fitness_score": 0.5}])
def test_labels_and_credentials_rejected_before_admission(tmp_path, metadata):
    instance = gateway(tmp_path, lambda _: pytest.fail("Forbidden metadata reached provider"))
    with pytest.raises(ValueError):
        generate(instance, metadata=metadata)
    assert instance.summary()["provider_attempts"] == 0
    instance.close()


@pytest.mark.parametrize("config", [
    {"model": "gemini-test"}, {"thinking": "high"}, {**MODEL, "api_key": "private"},
    {**MODEL, "retries": 2}, {**MODEL, "proxy": "http://user:password@host:1234"},
    {**MODEL, "timeout_s": float("nan")}, {**MODEL, "temperature": float("inf")},
    {**MODEL, "temperature": -1},
])
def test_model_config_must_be_explicit_public_and_retry_free(config):
    with pytest.raises(ValueError):
        public_model_config(config)


def test_ledger_contains_exact_complete_long_prompt(tmp_path):
    long_prompt = "full-sequence-" * 20000
    instance = gateway(tmp_path, lambda request: response())
    generate(instance, long_prompt)
    saved = json.loads(instance.db.execute("SELECT request_json FROM attempts").fetchone()[0])
    assert saved["messages"][1]["content"] == long_prompt
    instance.close()


def test_google_sdk_contract_streaming_no_retries_no_afc_no_network(tmp_path, monkeypatch):
    genai = pytest.importorskip("google.genai")
    from evagent.integrated.gateway import google_stream_transport

    captured = {}

    class FakeModels:
        def generate_content_stream(self, **kwargs):
            captured["generation"] = kwargs
            yield SimpleNamespace(
                usage_metadata=SimpleNamespace(prompt_token_count=8, candidates_token_count=2,
                                               total_token_count=12, thoughts_token_count=2),
                candidates=[SimpleNamespace(
                    finish_reason="STOP", content=SimpleNamespace(parts=[
                        SimpleNamespace(text="thought", thought=True, function_call=None),
                        SimpleNamespace(text="answer", thought=False, function_call=None),
                    ]))], prompt_feedback=None, model_version="fake-version",
            )

    class FakeClient:
        def __init__(self, **kwargs):
            captured["client"] = kwargs
            self.models = FakeModels()

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    monkeypatch.setattr(genai, "Client", FakeClient)
    monkeypatch.setenv("GEMINI_API_KEY", "synthetic-secret")
    instance = gateway(tmp_path, google_stream_transport)
    result = generate(instance)
    assert result["status"] == "ok"
    assert result["final_text"] == "answer"
    assert result["thought_summary"] == "thought"
    assert result["usage_summary"]["total_tokens"] == 12
    assert captured["client"]["http_options"].retry_options.attempts == 1
    assert captured["generation"]["config"].automatic_function_calling.disable is True
    assert captured["generation"]["model"] == "gemini-test"
    saved = instance.db.execute("SELECT request_json,response_json FROM attempts").fetchone()
    assert "synthetic-secret" not in "".join(saved)
    instance.close()
