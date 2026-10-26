import types

from evagent.model import _provider_generate


def test_provider_reconnects_after_transient_disconnect(monkeypatch, tmp_path):
    calls = []

    class Result:
        def __init__(self, error=None):
            self.status = "error" if error else "ok"
            self.error = error
            self.transport_error = error
            self.retryable = bool(error)
            self.text = "{}"
            self.thinking_text = ""
            self.usage = {"totalTokenCount": 7}
            self.elapsed_s = 0.1
            self.attempts = 1
            self.model_version = "test"
            self.finish_reason = "STOP"

    def fake_chat(*args, **kwargs):
        del args, kwargs
        calls.append(1)
        return Result("RemoteDisconnected: tunnel closed") if len(calls) < 3 else Result()

    monkeypatch.setitem(
        __import__("sys").modules,
        "rg_llm.client",
        types.SimpleNamespace(chat_gemini=fake_chat),
    )
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("EVAGENT_DISCONNECT_RETRIES", "3")
    monkeypatch.setenv("EVAGENT_RECONNECT_BACKOFF_S", "0")
    result = _provider_generate(
        public_config={
            "api_key_env": "GEMINI_API_KEY",
            "model_id": "test-model",
            "retries": 1,
            "timeout_s": 1,
            "proxy": None,
        },
        rg_llm_root=tmp_path,
        system="system",
        user="user",
        run_name="test",
        metadata={},
    )
    assert len(calls) == 3
    assert result["transport_reconnects"] == 2
    assert result["attempts"] == 3
