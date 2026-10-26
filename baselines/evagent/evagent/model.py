"""Gemini adapter with safe tracing and robust structured-output parsing."""

from __future__ import annotations

import json
import os
import re
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langsmith import traceable

from . import runtime
from .config import ModelConfig
from .api_timeouts import generation_timeout_seconds
from .schemas import ModelCallRecord


class ModelError(RuntimeError):
    def __init__(self, message: str, calls=None):
        super().__init__(message)
        self.calls = list(calls or [])


_API_SEMAPHORE: threading.BoundedSemaphore | None = None


def configure_api_concurrency(limit: int) -> None:
    """Set a process-wide cap shared by all assay workflows."""
    global _API_SEMAPHORE
    if limit < 1:
        raise ValueError("API concurrency must be positive")
    _API_SEMAPHORE = threading.BoundedSemaphore(limit)


@dataclass
class StructuredResponse:
    data: Any
    calls: list[ModelCallRecord]


def ensure_rg_llm_importable(root: Path) -> None:
    src = str(root / "src")
    if src not in sys.path:
        sys.path.insert(0, src)


def _provider_generate(
    *,
    public_config: dict[str, Any],
    rg_llm_root: str,
    system: str,
    user: str,
    run_name: str,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Trace-safe boundary: the API key is read inside and is never an argument."""
    del run_name, metadata  # They remain visible to tracing but not the provider.
    key_env = str(public_config["api_key_env"])
    api_key = os.environ.get(key_env)
    if not api_key:
        raise ModelError(f"Required environment variable {key_env} is not set")

    ensure_rg_llm_importable(Path(rg_llm_root))
    from rg_llm.client import chat_gemini

    provider_retries = int(public_config.get("retries", 3))
    reconnect_retries = max(0, int(os.environ.get("EVAGENT_DISCONNECT_RETRIES", "3")))
    backoff_s = max(0.0, float(os.environ.get("EVAGENT_RECONNECT_BACKOFF_S", "4")))
    result = None
    outer_attempt = 0
    for outer_attempt in range(reconnect_retries + 1):
        # This is an operational socket deadline, deliberately kept out of
        # public_config so it cannot alter request-cache keys or experiment
        # protocol fingerprints.
        timeout_override = os.environ.get("EVAGENT_TIMEOUT_S")
        request_timeout = generation_timeout_seconds(timeout_override or public_config.get("timeout_s"))
        result = chat_gemini(
            str(public_config["model_id"]),
            system,
            user,
            api_key=api_key,
            endpoint=public_config.get("endpoint"),
            max_tokens=public_config.get("max_tokens"),
            temperature=float(public_config.get("temperature", 0.0)),
            timeout=request_timeout,
            retries=provider_retries,
            proxy_url=public_config.get("proxy"),
            thinking_level=public_config.get("thinking_level"),
            include_thoughts=bool(public_config.get("include_thoughts", True)),
        )
        if result.status != "error":
            break
        # chat_gemini creates a fresh urllib opener for every call. Repeating the
        # call therefore reconnects the proxy tunnel instead of reusing a dead
        # CONNECT socket. Authentication and other non-retryable errors stop
        # immediately; transient disconnects/5xx get an outer backoff retry.
        if not bool(getattr(result, "retryable", False)) or outer_attempt >= reconnect_retries:
            break
        time.sleep(backoff_s * (2**outer_attempt))
    assert result is not None
    if result.status == "error":
        detail = str(result.error or getattr(result, "transport_error", "") or "unknown")
        detail = detail.replace(api_key, "[REDACTED]")
        raise runtime.ExecutionPaused(
            f"Gemini request failed after {outer_attempt + 1} connection attempt(s): {detail[:600]}"
        )
    internal_attempts = int(getattr(result, "attempts", 1) or 1)
    return {
        "answer": result.text,
        "thought_summary": result.thinking_text,
        "usage": result.usage,
        "elapsed_s": result.elapsed_s,
        "attempts": outer_attempt * provider_retries + internal_attempts,
        "transport_reconnects": outer_attempt,
        "effective_timeout_s": request_timeout,
        "model_version": result.model_version,
        "finish_reason": result.finish_reason,
        "status": result.status,
        # Response body only: never request headers or credentials. New protocols
        # inspect this before interpreting an empty answer as a scientific failure.
        "provider_raw": getattr(result, "raw", None),
    }


@traceable(name="gemini_generate", run_type="llm")
def _gemini_generate(**kwargs: Any) -> dict[str, Any]:
    key = runtime.digest(kwargs)
    if runtime.STORE:
        cached = runtime.STORE.cached(key)
        if cached is not None:
            return {**cached, "cache_hit": True, "request_sha256": key, "queue_s": 0.0}
    if _API_SEMAPHORE is None:
        configure_api_concurrency(12)
    started = time.perf_counter()
    assert _API_SEMAPHORE is not None
    with _API_SEMAPHORE:
        queued = time.perf_counter() - started
        raw = (
            runtime.STORE.invoke(key, lambda: _provider_generate(**kwargs))
            if runtime.STORE
            else _provider_generate(**kwargs)
        )
    return {**raw, "queue_s": queued, "request_sha256": key}


def extract_json(text: str) -> Any:
    """Parse JSON even when wrapped in prose/fences or containing trailing commas."""
    if not text or not text.strip():
        raise ValueError("empty model output")
    cleaned = text.strip().replace("“", '"').replace("”", '"').replace("’", "'")
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.I | re.S)
    cleaned = re.sub(r",\s*([}\]])", r"\1", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        for index, char in enumerate(cleaned):
            if char not in "[{":
                continue
            try:
                value, _ = decoder.raw_decode(cleaned[index:])
                return value
            except json.JSONDecodeError:
                continue
    raise ValueError("no valid JSON value found in model output")


class GeminiAdapter:
    def __init__(
        self,
        config: ModelConfig,
        rg_llm_root: Path,
        schema_retries: int = 1,
        namespace: str = "default",
    ):
        self.config = config
        self.rg_llm_root = rg_llm_root
        self.schema_retries = schema_retries
        self.namespace = namespace

    def _call(
        self, system: str, user: str, run_name: str, metadata: dict[str, Any]
    ) -> ModelCallRecord:
        raw = _gemini_generate(
            public_config=self.config.public_dict(),
            rg_llm_root=str(self.rg_llm_root),
            system=system,
            user=user,
            run_name=run_name,
            metadata={**metadata, "cache_namespace": self.namespace},
        )
        return ModelCallRecord(run_name=run_name, **{k: v for k, v in raw.items() if k != "status"})

    def generate_json(
        self,
        *,
        system: str,
        user: str,
        run_name: str,
        metadata: dict[str, Any] | None = None,
    ) -> StructuredResponse:
        calls: list[ModelCallRecord] = []
        call = self._call(system, user, run_name, metadata or {})
        calls.append(call)
        try:
            return StructuredResponse(extract_json(call.answer), calls)
        except ValueError as first_error:
            for retry in range(self.schema_retries):
                repair_user = (
                    "Convert the following response to valid JSON only. Preserve all IDs and values; "
                    "do not add commentary.\n\n" + call.answer
                )
                call = self._call(
                    "You repair malformed JSON without changing its semantic content.",
                    repair_user,
                    f"{run_name}.json_repair_{retry + 1}",
                    metadata or {},
                )
                calls.append(call)
                try:
                    return StructuredResponse(extract_json(call.answer), calls)
                except ValueError:
                    pass
            raise ModelError(f"Unable to parse structured output: {first_error}", calls=calls)
