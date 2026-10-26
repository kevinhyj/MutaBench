"""Offline checks: future seed=42 profile reaches the SDK, old seeds stay explicit."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from evagent.integrated import gateway


PROFILE = Path(__file__).resolve().parents[1] / "examples/gemini_flash_medium_seed42.json"


def test_seed42_profile_preserves_current_settings():
    cfg = json.loads(PROFILE.read_text())
    clean = gateway.public_model_config(cfg)
    assert clean == cfg
    assert clean["generation_seed"] == 42
    assert clean["model"] == "gemini-3.8-flash"
    assert clean["thinking_level"] == "medium"
    assert clean["temperature"] == 0
    assert clean["max_output_tokens"] == 65536
    assert clean["proxy"] == "http://127.0.0.1:7777"
    assert clean["api_key_env"] == "GEMINI_API_KEY"
    assert "seed" not in clean  # Do not accidentally substitute a data seed.


@pytest.mark.parametrize("mode", ["stream", "nonstream"])
@pytest.mark.parametrize("seed", [42, 1])
def test_seed_reaches_sdk_without_overriding_legacy(monkeypatch, mode, seed):
    from google import genai

    captured = []

    class Client:
        def __init__(self, **kwargs):
            self.models = self

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def generate_content(self, **kwargs):
            captured.append(kwargs)
            return SimpleNamespace(
                usage_metadata=SimpleNamespace(prompt_token_count=1, candidates_token_count=1,
                    total_token_count=2, thoughts_token_count=0),
                candidates=[SimpleNamespace(finish_reason="STOP", content=SimpleNamespace(
                    parts=[SimpleNamespace(text="OK", thought=False, function_call=None)]))],
                prompt_feedback=None, model_version="offline-mock")

        def generate_content_stream(self, **kwargs):
            yield self.generate_content(**kwargs)

    monkeypatch.setattr(genai, "Client", Client)
    monkeypatch.setenv("OFFLINE_SEED_TEST_KEY", "not-a-real-key")
    cfg = json.loads(PROFILE.read_text())
    cfg.update(api_key_env="OFFLINE_SEED_TEST_KEY", transport_mode=mode,
               generation_seed=seed)
    cfg = gateway.public_model_config(cfg)
    # Two independent calls also represent a retry: no counter-derived seed.
    for _ in range(2):
        result = gateway.google_stream_transport({
            "model": cfg["model"], "model_config": cfg,
            "messages": [{"role": "user", "content": "Return OK."}]})
        assert result["final_text"] == "OK"
    assert len(captured) == 2
    assert all(call["config"].seed == seed for call in captured)
