"""Integration-owned request ledger; preserves EVAgent's ExecutionPaused boundary.

Unlike the legacy RequestStore, each provider attempt is retained independently
and ambiguous attempts close admission unless a scoped continuation policy is enabled.
Token cost means reported provider tokens, not an estimated currency charge.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlsplit

from evagent.runtime import ExecutionPaused


class ProviderStreamInterrupted(RuntimeError):
    """A failed stream retains only normalized response fields, never HTTP headers."""

    def __init__(self, partial_response: dict):
        super().__init__("Provider stream interrupted")
        self.partial_response = partial_response


class ProviderAttemptsExhausted(ExecutionPaused):
    """A failed provider request must not enter semantic repair."""


class AdmissionPaused(ExecutionPaused):
    """An outer admission controller stopped a call before provider submission."""


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


_FORBIDDEN_KEYS = {
    "api_key", "apikey", "authorization", "password", "secret", "access_token",
    "labels", "label", "answer_key", "fitness", "fitness_score", "true_score",
    "target", "targets", "ground_truth",
    "experimental_score", "dms_score", "measured_score", "score", "scores",
}


def assert_public(value: Any) -> None:
    """Reject credentials and benchmark labels in structured request metadata."""
    if isinstance(value, dict):
        for key, item in value.items():
            if str(key).lower() in _FORBIDDEN_KEYS:
                raise ValueError(f"Forbidden request metadata field: {key}")
            assert_public(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            assert_public(item)


def public_model_config(config: dict) -> dict:
    assert_public(config)
    model = config.get("model", config.get("model_id"))
    if not isinstance(model, str) or not model.strip():
        raise ValueError("model_config requires an explicit model/model_id")
    if "thinking_level" not in config and "thinking" not in config:
        raise ValueError("model_config requires explicit thinking_level/thinking")
    thinking = config.get("thinking_level", config.get("thinking"))
    if thinking is not None and (not isinstance(thinking, str) or not thinking.strip()):
        raise ValueError("thinking_level must be a nonempty string or null")
    known = {
        "model", "model_id", "thinking", "thinking_level", "temperature", "max_tokens",
        "max_output_tokens", "timeout_s", "endpoint", "proxy", "api_key_env",
        "include_thoughts", "retries", "provider", "generation_seed",
    }
    if set(config) - known:
        raise ValueError(f"Unsupported model options: {sorted(set(config) - known)}")
    if config.get("provider", "google") not in ("google", "gemini", "google_genai"):
        raise ValueError("The integrated provider currently supports Google Gemini only")
    if config.get("retries", 0) != 0:
        raise ValueError("Integrated requests require retries=0; retries are ledger-owned")
    for key in ("endpoint", "proxy"):
        if config.get(key):
            parsed = urlsplit(config[key])
            if parsed.username or parsed.password or parsed.query or parsed.fragment:
                raise ValueError(f"{key} must not embed credentials or query parameters")
    seed = config.get("generation_seed")
    if seed is not None and (type(seed) is not int or not 0 <= seed <= 2147483647):
        raise ValueError("generation_seed must be a nonnegative int32")
    public = {
        "model": model,
        "thinking_level": thinking,
        "temperature": float(config.get("temperature", 0.0)),
        "max_output_tokens": config.get("max_output_tokens", config.get("max_tokens")),
        "timeout_s": float(config.get("timeout_s", 1200)),
        "include_thoughts": bool(config.get("include_thoughts", True)),
        "api_key_env": str(config.get("api_key_env", "GEMINI_API_KEY")),
        "endpoint": config.get("endpoint"),
        "proxy": config.get("proxy"),
        "provider": "google_genai",
        "retries": 0, "generation_seed": seed,
    }
    if not math.isfinite(public["timeout_s"]) or public["timeout_s"] <= 0:
        raise ValueError("timeout_s must be finite and positive")
    if not math.isfinite(public["temperature"]) or not 0 <= public["temperature"] <= 2:
        raise ValueError("temperature must be finite and between 0 and 2")
    maximum = public["max_output_tokens"]
    if maximum is not None and (type(maximum) is not int or maximum < 1):
        raise ValueError("max_output_tokens must be a positive integer")
    return public


class RequestGateway:
    """Serial, durable admission for all provider calls made by one experiment."""

    def __init__(self, *, model_config: dict, run_dir: Path, fingerprint: str,
                 token_budget: int, transport: Callable[[dict], dict] | None = None,
                 reserve_budget: bool = False, prior_usage: dict | None = None):
        if type(token_budget) is not int or token_budget <= 0:
            raise ValueError("A positive explicit token_budget is required")
        if not fingerprint:
            raise ValueError("A nonempty experiment fingerprint is required")
        self.model_config = public_model_config(model_config)
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.continuation_policy = None
        for parent in (self.run_dir, *self.run_dir.parents):
            policy_path = parent / "unknown_response_policy.json"
            if policy_path.exists():
                policy = json.loads(policy_path.read_text())
                if policy != {"mode": "retry_then_continue", "max_attempts": 3, "backoff_seconds": [15, 30]}:
                    raise ValueError("Unsupported unknown response policy")
                self.continuation_policy = policy
                break
        self._lease = (self.run_dir / "request_ledger.lock").open("a+")
        try:
            fcntl.flock(self._lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self._lease.close()
            raise ExecutionPaused("Another generator owns this experiment ledger") from exc
        self.path = self.run_dir / "request_ledger.sqlite"
        self.fingerprint = fingerprint
        self.token_budget = token_budget
        self.reserve_budget = reserve_budget
        self.prior_usage = prior_usage or {}
        for key in ('reported_tokens', 'provider_attempts'):
            if type(self.prior_usage.get(key, 0)) is not int or self.prior_usage.get(key, 0) < 0:
                raise ValueError('Invalid prior usage debit')
        self.transport = transport or google_stream_transport
        self.lock = threading.RLock()
        self.db = sqlite3.connect(self.path, check_same_thread=False, timeout=30)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS run_state (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS attempts (
                id INTEGER PRIMARY KEY AUTOINCREMENT, request_sha256 TEXT NOT NULL,
                run_name TEXT NOT NULL, status TEXT NOT NULL, request_json TEXT NOT NULL,
                response_json TEXT, tokens INTEGER, started REAL NOT NULL, ended REAL,
                elapsed_s REAL, error_type TEXT
            );
            CREATE UNIQUE INDEX IF NOT EXISTS unique_attempt ON attempts(request_sha256);
        """)
        state = dict(self.db.execute("SELECT key,value FROM run_state"))
        if state and (state.get("fingerprint") != fingerprint
                      or state.get("model_sha256") != digest(self.model_config)):
            self.close()
            raise ValueError("Run directory belongs to a different experiment/model fingerprint")
        if state and int(state["token_budget"]) != token_budget:
            self.close()
            raise ValueError("Run token budget is immutable; use a new run directory")
        if state and json.loads(state.get('prior_usage', '{}')) != self.prior_usage:
            self.close()
            raise ValueError('Prior campaign usage debit is immutable')
        for key, value in (("fingerprint", fingerprint),
                           ("model_sha256", digest(self.model_config)),
                           ("token_budget", str(token_budget)),
                           ('prior_usage', json.dumps(self.prior_usage, sort_keys=True))):
            self.db.execute("INSERT OR IGNORE INTO run_state VALUES (?,?)", (key, value))
        abandoned = self.db.execute(
            "UPDATE attempts SET status='interrupted_unknown_cost', ended=? "
            "WHERE status='inflight'", (time.time(),)
        ).rowcount
        if abandoned:
            self._pause("Interrupted provider attempt has unknown cost")
        self.db.commit()
        self._acknowledge_unknown_pause()

    def _pause(self, reason: str) -> None:
        self.db.execute(
            "INSERT OR REPLACE INTO run_state VALUES ('pause_reason',?)", (reason,)
        )

    def pause(self, reason: str) -> None:
        """Close admission for an adapter invariant failure without another API attempt."""
        with self.lock:
            self._pause(reason)
            self.db.commit()
        raise ExecutionPaused(reason)

    def _acknowledge_unknown_pause(self) -> bool:
        if not self.continuation_policy:
            return False
        row = self.db.execute("SELECT value FROM run_state WHERE key='pause_reason'").fetchone()
        if not row:
            return True
        allowed = {
            "Interrupted provider attempt has unknown cost",
            "Provider attempt failed with unknown response/cost",
            "Unrecognized provider response",
            "Provider response has no usage; token cost is unknown",
            "Incomplete/invalid provider usage; token cost is unknown",
            "Inconsistent/unknown provider token cost",
            "Provider reported hidden calls/retries",
            "Provider reasoning-token usage is invalid",
            "Provider termination marker missing or unknown",
            "Unrecognized provider termination marker",
            "Provider returned no identifiable final answer",
        }
        if row[0] not in allowed:
            return False
        self.db.execute("INSERT INTO run_state VALUES (?,?)",
                        ("unknown_pause_ack:" + str(time.time_ns()),
                         json.dumps({"reason": row[0], "policy": self.continuation_policy})))
        self.db.execute("DELETE FROM run_state WHERE key='pause_reason'")
        self.db.commit()
        return True

    def summary(self) -> dict:
        with self.lock:
            rows = self.db.execute(
                "SELECT status,COUNT(*),COALESCE(SUM(tokens),0) FROM attempts WHERE status!='not_submitted' GROUP BY status"
            ).fetchall()
            paused = self.db.execute(
                "SELECT value FROM run_state WHERE key='pause_reason'"
            ).fetchone()
            current_total = sum(row[2] for row in rows)
            total = current_total + self.prior_usage.get('reported_tokens', 0)
            unknown_reserve = 0
            unknown_attempts = 0
            for request_json, status, tokens in self.db.execute(
                    "SELECT request_json,status,tokens FROM attempts WHERE status!='not_submitted'"):
                if status in {"failed_unknown_cost", "interrupted_unknown_cost", "unknown_cost"} or (tokens is None and status != "inflight"):
                    unknown_attempts += 1
                    request = json.loads(request_json)
                    maximum = request.get("model_config", self.model_config).get("max_output_tokens") or 32768
                    estimate = sum(len(m["content"].encode("utf-8")) for m in request.get("messages", [])) + 2*maximum + 1024
                    unknown_reserve += max(0, estimate - (tokens or 0))
            budget_used = total + self.prior_usage.get("unknown_cost_reserve_tokens", 0) + (unknown_reserve if self.continuation_policy else 0)
            elapsed = self.db.execute("SELECT COALESCE(SUM(elapsed_s),0) FROM attempts WHERE status!='not_submitted'").fetchone()[0]
            statuses = dict(self.prior_usage.get('requests_by_status', {}))
            for status, count, _ in rows:
                statuses[status] = statuses.get(status, 0)+count
            return {
                "soft_limit": self.token_budget, "token_budget": self.token_budget,
                "reported_tokens": total, "current_run_reported_tokens": current_total,
                "prior_reported_tokens": self.prior_usage.get('reported_tokens', 0),
                "prior_ledgers": self.prior_usage.get('ledgers', []),
                "provider_attempts": sum(row[1] for row in rows) + self.prior_usage.get('provider_attempts', 0),
                "provider_elapsed_s": elapsed + self.prior_usage.get('provider_elapsed_s', 0),
                "requests_by_status": statuses,
                "unknown_cost_attempts": unknown_attempts + self.prior_usage.get("unknown_cost_attempts", 0),
                "current_unknown_cost_attempts": unknown_attempts,
                "unknown_cost_reserve_tokens": unknown_reserve + self.prior_usage.get("unknown_cost_reserve_tokens", 0),
                "current_unknown_cost_reserve_tokens": unknown_reserve,
                "budget_accounted_tokens": budget_used,
                "continuation_policy": self.continuation_policy,
                "paused": bool(paused) or budget_used >= self.token_budget,
                "pause_reason": paused[0] if paused else (
                    "Token soft budget exhausted" if budget_used >= self.token_budget else None
                ),
                "ledger_path": str(self.path),
            }

    def generate(self, *, system: str, user: str, run_name: str,
                 metadata: dict | None = None) -> dict:
        return self.invoke_messages(
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            run_name=run_name, metadata=metadata,
        )

    def invoke_messages(self, *, messages: list[dict], run_name: str,
                        metadata: dict | None = None) -> dict:
        if not self.continuation_policy:
            return self._invoke_once(messages=messages, run_name=run_name, metadata=metadata)
        for index in range(self.continuation_policy["max_attempts"]):
            retry_metadata = dict(metadata or {})
            if index:
                retry_metadata["gateway_transport_attempt"] = index + 1
            try:
                return self._invoke_once(messages=messages, run_name=run_name, metadata=retry_metadata)
            except ProviderAttemptsExhausted:
                raise
            except AdmissionPaused:
                raise
            except ExecutionPaused:
                if (self.run_dir / "PAUSE").exists() or not self._acknowledge_unknown_pause():
                    raise
                if self.summary()["paused"]:
                    raise
                if index + 1 < self.continuation_policy["max_attempts"]:
                    time.sleep(self.continuation_policy["backoff_seconds"][index])
        raise ProviderAttemptsExhausted("Provider attempts exhausted; continue campaign with API failure")

    def _invoke_once(self, *, messages: list[dict], run_name: str,
                     metadata: dict | None = None) -> dict:
        metadata = metadata or {}
        assert_public(metadata)
        # Snapshot before hashing: a caller cannot mutate an admitted request later.
        request = json.loads(json.dumps({
            "model": self.model_config["model"],
            "thinking_level": self.model_config["thinking_level"],
            "model_config": self.model_config, "messages": messages,
            "metadata": metadata, "fingerprint": self.fingerprint,
        }, allow_nan=False))
        key = digest(request)
        started = time.time()
        with self.lock:
            # The process lease and thread lock prevent stale-budget admission.
            self.db.execute("BEGIN IMMEDIATE")
            previous = self.db.execute(
                "SELECT status,response_json FROM attempts WHERE request_sha256=?", (key,)
            ).fetchone()
            if previous and previous[0] in ("ok", "blocked", "truncated", "terminal_failure"):
                self.db.commit()
                return {**json.loads(previous[1]), "cache_hit": True}
            info = self.summary()
            if self.reserve_budget:
                maximum = self.model_config["max_output_tokens"]
                if maximum is None:
                    raise ValueError("Campaign admission requires explicit max_output_tokens")
                # Conservative text-only allowance: UTF-8 bytes plus output/thought
                # headroom. Actual billing always uses reported provider tokens.
                reserve = sum(len(m["content"].encode("utf-8")) for m in messages) + 2*maximum + 1024
                if info["budget_accounted_tokens"] + reserve > self.token_budget:
                    self._pause("Insufficient remaining budget for conservative request reserve")
                    self.db.commit()
                    raise AdmissionPaused("Campaign token reserve exhausted before provider submission")
            if previous or info["paused"] or (self.run_dir / "PAUSE").exists():
                self.db.commit()
                raise ExecutionPaused(info["pause_reason"] or "Prior unresolved attempt or PAUSE marker")
            cursor = self.db.execute(
                "INSERT INTO attempts(request_sha256,run_name,status,request_json,started) "
                "VALUES (?,?,'inflight',?,?)", (key, run_name, json.dumps(request), started)
            )
            attempt = cursor.lastrowid
            # Commit before network I/O so a process crash leaves durable inflight evidence.
            self.db.commit()
            try:
                raw = self.transport(request)
            except ProviderAttemptsExhausted:
                self.db.execute(
                    "UPDATE attempts SET status='not_submitted',tokens=0,ended=?,error_type=? WHERE id=?",
                    (time.time(), "ProviderAttemptsExhausted", attempt),
                )
                self.db.commit()
                raise
            except AdmissionPaused:
                self.db.execute(
                    "UPDATE attempts SET status='not_submitted',tokens=0,ended=? WHERE id=?",
                    (time.time(), attempt),
                )
                self._pause("Campaign stopped admission before provider submission")
                self.db.commit()
                raise
            except BaseException as exc:
                partial = exc.partial_response if isinstance(exc, ProviderStreamInterrupted) else None
                partial_tokens = ((partial.get("usage_summary") or {}).get("total_tokens")
                                  if partial else None)
                if type(partial_tokens) is not int or partial_tokens < 0:
                    partial_tokens = None
                self.db.execute(
                    "UPDATE attempts SET status='failed_unknown_cost',ended=?,elapsed_s=?,"
                    "error_type=?,response_json=?,tokens=? WHERE id=?",
                    (time.time(), time.time() - started, type(exc).__name__,
                     json.dumps(partial) if partial else None, partial_tokens, attempt),
                )
                self._pause("Provider attempt failed with unknown response/cost")
                self.db.commit()
                # Cancellation must not be converted into an automatic transport retry.
                if not isinstance(exc, Exception):
                    raise
                # Exception messages may contain headers/URLs/credentials. Do not persist them.
                raise ExecutionPaused(
                    f"Provider attempt paused ({type(exc).__name__}); inspect ledger before new work"
                ) from exc
            elapsed = time.time() - started
            reason, tokens, status = _classify(raw)
            serializable = raw if isinstance(raw, dict) else {"response_type": type(raw).__name__}
            result = {
                **serializable, "request_sha256": key, "cache_hit": False,
                "elapsed_s": elapsed, "status": status,
            }
            self.db.execute(
                "UPDATE attempts SET status=?,response_json=?,tokens=?,ended=?,elapsed_s=? WHERE id=?",
                (status, json.dumps(result, default=str), tokens, time.time(), elapsed, attempt),
            )
            if reason:
                self._pause(reason)
            self.db.commit()
            if reason:
                raise ExecutionPaused(reason)
            return result

    def close(self) -> None:
        with self.lock:
            self.db.close()
            fcntl.flock(self._lease, fcntl.LOCK_UN)
            self._lease.close()


def _classify(raw: Any) -> tuple[str | None, int | None, str]:
    if not isinstance(raw, dict):
        return "Unrecognized provider response", None, "unknown_response"
    usage = raw.get("usage_summary")
    if not isinstance(usage, dict):
        return "Provider response has no usage; token cost is unknown", None, "unknown_cost"
    required = [usage.get(key) for key in ("input_tokens", "output_tokens", "total_tokens")]
    if any(type(value) is not int or value < 0 for value in required):
        return "Incomplete/invalid provider usage; token cost is unknown", None, "unknown_cost"
    tokens = usage["total_tokens"]
    if tokens < usage["input_tokens"] + usage["output_tokens"] or raw.get("cost_known") is False:
        return "Inconsistent/unknown provider token cost", tokens, "unknown_cost"
    if usage.get("model_calls", 1) != 1:
        return "Provider reported hidden calls/retries", tokens, "unknown_cost"
    reasoning = usage.get("reasoning_tokens", 0)
    if type(reasoning) is not int or reasoning < 0 or reasoning > tokens:
        return "Provider reasoning-token usage is invalid", tokens, "unknown_cost"
    finish = raw.get("finish_reason")
    if not isinstance(finish, str) or finish.upper() in ("", "UNKNOWN", "NONE", "FINISH_REASON_UNSPECIFIED"):
        return "Provider termination marker missing or unknown", tokens, "unknown_finish"
    finish = finish.upper().split(".")[-1]
    if finish in ("SAFETY", "RECITATION", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII", "IMAGE_SAFETY"):
        return None, tokens, "blocked"
    if finish in ("MAX_TOKENS", "LENGTH"):
        return None, tokens, "truncated"
    if finish in ("MALFORMED_FUNCTION_CALL", "UNEXPECTED_TOOL_CALL", "OTHER", "LANGUAGE", "TOOL_CALLS"):
        return None, tokens, "terminal_failure"
    if finish not in ("STOP", "END_TURN"):
        return "Unrecognized provider termination marker", tokens, "unknown_finish"
    if raw.get("tool_calls"):
        return None, tokens, "terminal_failure"
    if raw.get("status") in ("error", "failed", "blocked", "truncated"):
        return None, tokens, "terminal_failure"
    if not isinstance(raw.get("final_text"), str) or not raw["final_text"].strip():
        return "Provider returned no identifiable final answer", tokens, "unknown_response"
    return None, tokens, "ok"


def google_stream_transport(request: dict) -> dict:
    """One Google SDK stream with retries and automatic function calls disabled."""
    from google import genai
    from google.genai import types

    config = request["model_config"]
    api_key = os.environ.get(config["api_key_env"])
    if not api_key:
        raise RuntimeError(f"Missing environment variable {config['api_key_env']}")
    http = {"timeout": int(config["timeout_s"] * 1000),
            "retry_options": types.HttpRetryOptions(attempts=1)}
    if config.get("endpoint"):
        http["base_url"] = config["endpoint"]
    if config.get("proxy"):
        http["client_args"] = {"proxy": config["proxy"]}
    generation = {
        "temperature": config["temperature"],
        "automatic_function_calling": types.AutomaticFunctionCallingConfig(disable=True),
        "system_instruction": "\n\n".join(
            message["content"] for message in request["messages"] if message["role"] == "system"
        ),
    }
    if config["generation_seed"] is not None:
        generation["seed"] = config["generation_seed"]
    if config["max_output_tokens"] is not None:
        generation["max_output_tokens"] = config["max_output_tokens"]
    if config["thinking_level"] is not None:
        generation["thinking_config"] = types.ThinkingConfig(
            thinking_level=config["thinking_level"], include_thoughts=config["include_thoughts"]
        )
    contents = [
        types.Content(role="model" if message["role"] == "assistant" else "user",
                      parts=[types.Part(text=message["content"])])
        for message in request["messages"] if message["role"] != "system"
    ]
    text_parts, thoughts, tool_calls = [], [], []
    finish, usage, response_metadata = None, None, {}

    def snapshot():
        normalized_usage = None if usage is None else {
            "input_tokens": usage.prompt_token_count,
            "output_tokens": usage.candidates_token_count,
            "total_tokens": usage.total_token_count,
            "reasoning_tokens": usage.thoughts_token_count or 0,
            "model_calls": 1,
        }
        return {
            "final_text": "".join(text_parts), "thought_summary": "".join(thoughts),
            "finish_reason": finish, "usage_summary": normalized_usage,
            "response_metadata": response_metadata, "tool_calls": tool_calls,
        }

    try:
        with genai.Client(api_key=api_key, http_options=types.HttpOptions(**http)) as client:
            for chunk in client.models.generate_content_stream(
                model=request["model"], contents=contents,
                config=types.GenerateContentConfig(**generation),
            ):
                if chunk.usage_metadata is not None:
                    usage = chunk.usage_metadata
                for candidate in chunk.candidates or []:
                    if candidate.finish_reason is not None:
                        finish = getattr(candidate.finish_reason, "value", str(candidate.finish_reason))
                    for part in (candidate.content.parts if candidate.content else []) or []:
                        if part.text:
                            (thoughts if part.thought else text_parts).append(part.text)
                        if part.function_call:
                            tool_calls.append(part.function_call.model_dump(mode="json"))
                if chunk.prompt_feedback:
                    response_metadata["prompt_feedback"] = chunk.prompt_feedback.model_dump(mode="json")
                if chunk.model_version:
                    response_metadata["model_version"] = chunk.model_version
    except Exception as exc:
        raise ProviderStreamInterrupted(snapshot()) from exc
    return snapshot()
