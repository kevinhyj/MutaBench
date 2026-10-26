"""Both backends use fake transport only; Deep Agents tests require the agent extra."""

import sys

import pytest

from evagent.integrated.backends import _instance_profile_factory, build_generator
from evagent.runtime import ExecutionPaused


def fake_response():
    return {"final_text": '{"ranking":["v1"]}', "thought_summary": "",
            "finish_reason": "STOP", "response_metadata": {},
            "usage_summary": {"input_tokens": 5, "output_tokens": 3, "total_tokens": 8,
                              "reasoning_tokens": 0, "model_calls": 1}}


def _harness_profile_for_model(model, spec):
    return "original"


def _toy_factory(*, model=None):
    return _harness_profile_for_model(model, None)


def test_factory_profile_isolation_without_global_registry():
    isolated = _instance_profile_factory(_toy_factory, "private")
    assert isolated() == "private"
    assert _toy_factory() == "original"


def test_public_future_profile_interface_is_preferred():
    def factory(*, harness_profile, model=None):
        return harness_profile
    assert _instance_profile_factory(factory, "private")() == "private"


def test_unknown_factory_fails_closed():
    with pytest.raises(RuntimeError, match="isolated harness"):
        _instance_profile_factory(lambda **kwargs: None, object())


@pytest.mark.parametrize("backend", ["direct", "deepagents"])
def test_backend_exact_public_config_long_prompt_and_cached_gateway(tmp_path, backend):
    if backend == "deepagents":
        pytest.importorskip("deepagents")
    requests = []
    generator = build_generator(
        backend, model_config={"model": "gemini-fake", "thinking_level": "high"},
        run_dir=tmp_path, fingerprint="synthetic-offline", token_budget=100,
        transport=lambda req: requests.append(req) or fake_response(),
    )
    long_prompt = "synthetic-input-" * 24000
    result = generator.generate(system="synthetic system", user=long_prompt, run_name="test")
    assert result["status"] == "ok"
    assert requests[0]["model"] == "gemini-fake"
    assert requests[0]["thinking_level"] == "high"
    assert requests[0]["messages"] == [
        {"role": "system", "content": "synthetic system"},
        {"role": "user", "content": long_prompt},
    ]
    assert generator.generate(system="synthetic system", user=long_prompt,
                              run_name="test")["cache_hit"]
    assert len(requests) == 1
    assert generator.summary()["provider_attempts"] == 1
    generator.close()


def test_deepagents_profile_tool_and_memory_state_is_local(tmp_path):
    pytest.importorskip("deepagents")
    import deepagents.graph as graph_module
    original_resolver = graph_module._harness_profile_for_model
    generator = build_generator(
        "deepagents", model_config={"model": "gemini-fake", "thinking_level": None},
        run_dir=tmp_path, fingerprint="synthetic", token_budget=100,
        transport=lambda _: fake_response(),
    )
    assert graph_module._harness_profile_for_model is original_resolver
    assert generator.profile.general_purpose_subagent.enabled is False
    assert generator.filesystem_middleware._human_message_token_limit_before_evict is None
    assert generator.filesystem_middleware._tool_token_limit_before_evict is None
    with pytest.raises(ValueError, match="Tools are disabled"):
        generator.model.bind_tools([{"name": "task"}])
    result = generator.generate(system="plain context", user="synthetic", run_name="test")
    assert result["status"] == "ok"
    assert graph_module._harness_profile_for_model is original_resolver
    assert not any("Summarization" in name or "Memory" in name
                   for name in generator.graph.get_graph().nodes)
    # The required filesystem scaffolding may compile an unreachable tools node;
    # successful invocation plus bind_tools rejection proves no tool was exposed.
    assert len(result["model_call_records"]) == 1
    generator.close()


def test_deepagents_provider_pause_is_not_intercepted(tmp_path):
    pytest.importorskip("deepagents")
    generator = build_generator(
        "deepagents", model_config={"model": "gemini-fake", "thinking_level": "high"},
        run_dir=tmp_path, fingerprint="synthetic", token_budget=100,
        transport=lambda _: {**fake_response(), "finish_reason": None},
    )
    with pytest.raises(ExecutionPaused):
        generator.generate(system="system", user="synthetic", run_name="test")
    assert generator.summary()["provider_attempts"] == 1
    generator.close()


def test_direct_import_does_not_load_deepagents(tmp_path):
    # Module-level imports are intentionally absent; basic installation may be
    # Python 3.10 and have no Deep Agents distribution at all.
    import subprocess
    source = "import evagent.integrated.backends; import sys; assert 'deepagents' not in sys.modules"
    subprocess.run([sys.executable, "-c", source], check=True)
