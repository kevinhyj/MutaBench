"""Integration-owned request ledger; preserves EVAgent's ExecutionPaused boundary.

Unlike the legacy RequestStore, each provider attempt is retained independently
and every ambiguous attempt permanently closes admission for this run directory.
Token cost means reported provider tokens, not an estimated currency charge.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import re
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Callable
from urllib.parse import quote, urlsplit

from evagent.runtime import ExecutionPaused
from evagent.api_timeouts import generation_timeout_seconds


class ProviderStreamInterrupted(RuntimeError):
    """A failed stream retains only normalized response fields, never HTTP headers."""

    def __init__(self, partial_response: dict):
        super().__init__("Provider stream interrupted")
        self.partial_response = partial_response


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
        "include_thoughts", "retries", "provider", "generation_seed", "transport_mode", "function_calling_mode",
        "response_mime_type",
        "research_tool_protocol",
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
    public = {
        "model": model,
        "thinking_level": thinking,
        "temperature": float(config.get("temperature", 0.0)),
        "max_output_tokens": config.get("max_output_tokens", config.get("max_tokens")),
        "timeout_s": float(config.get("timeout_s", 3600)),
        "include_thoughts": bool(config.get("include_thoughts", True)),
        "api_key_env": str(config.get("api_key_env", "GEMINI_API_KEY")),
        "endpoint": config.get("endpoint"),
        "proxy": config.get("proxy"),
        "provider": "google_genai",
        "retries": 0,
    }
    if not math.isfinite(public["timeout_s"]) or public["timeout_s"] <= 0:
        raise ValueError("timeout_s must be finite and positive")
    if not math.isfinite(public["temperature"]) or not 0 <= public["temperature"] <= 2:
        raise ValueError("temperature must be finite and between 0 and 2")
    if "generation_seed" in config:
        if type(config["generation_seed"]) is not int or config["generation_seed"] < 0:
            raise ValueError("generation_seed must be a nonnegative integer")
        public["generation_seed"] = config["generation_seed"]
    if "transport_mode" in config:
        if config["transport_mode"] not in {"stream", "nonstream"}:
            raise ValueError("transport_mode must be stream or nonstream")
        public["transport_mode"] = config["transport_mode"]
    if "function_calling_mode" in config:
        if config['function_calling_mode'] != 'NONE':
            raise ValueError('Integrated JSON tools require function_calling_mode=NONE')
        public['function_calling_mode'] = 'NONE'
    if 'response_mime_type' in config:
        if config['response_mime_type'] != 'application/json':
            raise ValueError('Supported explicit response_mime_type is application/json')
        public['response_mime_type'] = 'application/json'
    if 'research_tool_protocol' in config:
        if config['research_tool_protocol'] not in {'native_v1', 'native_review_v2', 'native_structure_v3', 'native_framework_v4'} or 'function_calling_mode' in config or 'response_mime_type' in config:
            raise ValueError('Native research tools require an explicit protocol without global output/function overrides')
        public['research_tool_protocol'] = config['research_tool_protocol']
    maximum = public["max_output_tokens"]
    if maximum is not None and (type(maximum) is not int or maximum < 1):
        raise ValueError("max_output_tokens must be a positive integer")
    return public


def validate_connection_retry_policy(policy, budget_mode):
    if policy is not None and (
        budget_mode != "unlimited" or not isinstance(policy, dict)
        or set(policy) != {'max_attempts', 'backoff_seconds', 'authorization'}
        or policy['max_attempts'] != 3 or policy['backoff_seconds'] != [15, 30]
        or not isinstance(policy['authorization'], str) or not policy['authorization'].strip()
    ):
        raise ValueError('Invalid explicit connection retry policy')


class RequestGateway:
    """Serial, durable admission for all provider calls made by one experiment."""

    def __init__(self, *, model_config: dict, run_dir: Path, fingerprint: str,
                 token_budget: int | None, transport: Callable[[dict], dict] | None = None,
                 reserve_budget: bool = False, prior_usage: dict | None = None,
                 budget_mode: str = "limited", connection_retry_policy: dict | None = None):
        if budget_mode not in {"limited", "unlimited"}:
            raise ValueError("Invalid budget_mode")
        if budget_mode == "unlimited" and token_budget is not None:
            raise ValueError("Unlimited budget requires token_budget=null")
        self.budget_mode = budget_mode
        validate_connection_retry_policy(connection_retry_policy, budget_mode)
        self.connection_retry_policy = connection_retry_policy
        if budget_mode == "limited" and (type(token_budget) is not int or token_budget <= 0):
            raise ValueError("A positive explicit token_budget is required")
        if not fingerprint:
            raise ValueError("A nonempty experiment fingerprint is required")
        self.model_config = public_model_config(model_config)
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
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
        if budget_mode == "limited" and self.prior_usage.get('unknown_usage_attempts', 0):
            self._lease.close()
            raise ValueError('Unresolved historical cost is incompatible with a finite budget')
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
        if state and (state["token_budget"] != str(token_budget) or state.get("budget_mode", "limited") != budget_mode):
            self.close()
            raise ValueError("Run token budget is immutable; use a new run directory")
        if state and json.loads(state.get('prior_usage', '{}')) != self.prior_usage:
            self.close()
            raise ValueError('Prior campaign usage debit is immutable')
        if state and json.loads(state.get('connection_retry_policy', 'null')) != connection_retry_policy:
            self.close()
            raise ValueError('Connection retry policy is immutable')
        for key, value in (("fingerprint", fingerprint),
                           ("model_sha256", digest(self.model_config)),
                           ("token_budget", str(token_budget)),
                           ("budget_mode", budget_mode),
                           ('prior_usage', json.dumps(self.prior_usage, sort_keys=True))):
            self.db.execute("INSERT OR IGNORE INTO run_state VALUES (?,?)", (key, value))
        if connection_retry_policy is not None:
            self.db.execute("INSERT OR IGNORE INTO run_state VALUES (?,?)",
                            ('connection_retry_policy', json.dumps(connection_retry_policy, sort_keys=True)))
        abandoned = self.db.execute(
            "UPDATE attempts SET status='interrupted_unknown_cost', ended=? "
            "WHERE status='inflight'", (time.time(),)
        ).rowcount
        if abandoned:
            self._pause("Interrupted provider attempt has unknown cost")
        self.db.commit()

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
            elapsed = self.db.execute("SELECT COALESCE(SUM(elapsed_s),0) FROM attempts WHERE status!='not_submitted'").fetchone()[0]
            statuses = dict(self.prior_usage.get('requests_by_status', {}))
            for status, count, _ in rows:
                statuses[status] = statuses.get(status, 0)+count
            return {
                "soft_limit": self.token_budget, "token_budget": self.token_budget,
                "budget_mode": self.budget_mode,
                "reported_tokens": total, "current_run_reported_tokens": current_total,
                "prior_reported_tokens": self.prior_usage.get('reported_tokens', 0),
                "prior_ledgers": self.prior_usage.get('ledgers', []),
                "unknown_usage_attempts": self.prior_usage.get('unknown_usage_attempts', 0)
                + sum(count for status, count, _ in rows if "unknown" in status or status == "inflight"),
                "usage_is_lower_bound": bool(self.prior_usage.get('unknown_usage_attempts', 0)
                or any("unknown" in status or status == "inflight" for status, _, _ in rows)),
                "provider_attempts": sum(row[1] for row in rows) + self.prior_usage.get('provider_attempts', 0),
                "provider_elapsed_s": elapsed + self.prior_usage.get('provider_elapsed_s', 0),
                "requests_by_status": statuses,
                "paused": bool(paused) or (self.token_budget is not None and total >= self.token_budget),
                "pause_reason": paused[0] if paused else (
                    "Token soft budget exhausted" if (self.token_budget is not None and total >= self.token_budget) else None
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
                        metadata: dict | None = None,
                        tool_declarations: list[dict] | None = None,
                        tool_choice: str = 'auto') -> dict:
        options = dict(tool_declarations=tool_declarations, tool_choice=tool_choice)
        if self.connection_retry_policy is None:
            return self._invoke_messages_once(messages=messages, run_name=run_name, metadata=metadata, **options)
        with self.lock:
            return self._invoke_with_connection_retry(messages=messages, run_name=run_name, metadata=metadata, **options)

    def _invoke_with_connection_retry(self, *, messages, run_name, metadata,
                                      tool_declarations=None, tool_choice='auto'):
        metadata = metadata or {}
        cap = metadata.get('provider_attempt_limit', 3)
        if type(cap) is not int or cap < 1:
            raise ValueError('Invalid provider attempt allowance')
        records = []
        for ordinal in range(min(3, cap)):
            if (self.run_dir.parent / 'PAUSE').exists():
                self.pause('Campaign PAUSE marker before provider submission')
            scoped = {**metadata, 'transport_attempt': ordinal + 1}
            key = digest(self._request(messages, scoped, tool_declarations, tool_choice))
            try:
                result = self._invoke_messages_once(
                    messages=messages, run_name=run_name, metadata=scoped,
                    tool_declarations=tool_declarations, tool_choice=tool_choice)
                records.append({k: result.get(k) for k in (
                    'request_sha256', 'status', 'cache_hit', 'usage_summary', 'elapsed_s')})
                return {**result, 'transport_attempt_records': records}
            except ExecutionPaused:
                with self.lock:
                    row = self.db.execute(
                        'SELECT status,response_json,elapsed_s FROM attempts WHERE request_sha256=?', (key,)
                    ).fetchone()
                    raw = json.loads(row[1] or '{}') if row else {}
                    diagnostic = raw.get('response_metadata', {}).get('transport_error', {})
                    pause = self.db.execute("SELECT value FROM run_state WHERE key='pause_reason'").fetchone()
                    latest = self.db.execute('SELECT request_sha256 FROM attempts ORDER BY id DESC LIMIT 1').fetchone()
                    if (ordinal + 1 >= min(3, cap) or not row or row[0] != 'failed_unknown_cost'
                            or diagnostic.get('exception_type') not in {
                                'RemoteProtocolError', 'ReadError', 'WriteError', 'ConnectError', 'ConnectionError'}
                            or (self.run_dir / 'PAUSE').exists()
                            or (self.run_dir.parent / 'PAUSE').exists()
                            or (pause and (pause[0] != 'Provider attempt failed with unknown response/cost'
                                           or latest[0] != key))):
                        raise
                    records.append({'request_sha256': key, 'status': row[0], 'cache_hit': False,
                                    'usage_summary': raw.get('usage_summary'), 'elapsed_s': row[2]})
                    self.db.execute('INSERT OR IGNORE INTO run_state VALUES (?,?)',
                                    ('connection_retry_ack:' + key, json.dumps({
                                        'policy': self.connection_retry_policy, 'time': time.time(),
                                        'usage_status': 'unresolved_lower_bound', 'next_attempt': ordinal + 2})))
                    self.db.execute("DELETE FROM run_state WHERE key='pause_reason'")
                    self.db.commit()
                try:
                    time.sleep(self.connection_retry_policy['backoff_seconds'][ordinal])
                except BaseException as error:
                    self._pause('Connection retry interrupted before next submission')
                    self.db.commit()
                    raise ExecutionPaused('Connection retry interrupted before next submission') from error
        raise AssertionError('Unreachable retry state')

    def _request(self, messages, metadata, tool_declarations=None, tool_choice='auto'):
        request = {
            'model': self.model_config['model'],
            'thinking_level': self.model_config['thinking_level'],
            'model_config': self.model_config, 'messages': messages,
            'metadata': metadata, 'fingerprint': self.fingerprint,
        }
        if self.model_config.get('research_tool_protocol') == 'native_framework_v4':
            declarations = tool_declarations or []
            if not isinstance(declarations, list):
                raise ValueError('Framework tool declarations must be a list')
            names = []
            for row in declarations:
                if (not isinstance(row, dict) or not isinstance(row.get('name'), str)
                        or not row['name'] or not isinstance(row.get('parameters', {}), dict)
                        or set(row) - {'name', 'description', 'parameters'}):
                    raise ValueError('Invalid framework function declaration')
                names.append(row['name'])
            if len(names) != len(set(names)):
                raise ValueError('Duplicate framework function declarations')
            if tool_choice not in {'auto', 'any', 'none', *names}:
                raise ValueError('Unsupported framework tool choice')
            request.update(tool_declarations=declarations, tool_choice=tool_choice)
        elif tool_declarations is not None or tool_choice != 'auto':
            raise ValueError('Dynamic tool declarations require native_framework_v4')
        return json.loads(json.dumps(request, allow_nan=False))

    def _invoke_messages_once(self, *, messages: list[dict], run_name: str,
                              metadata: dict | None = None,
                              tool_declarations: list[dict] | None = None,
                              tool_choice: str = 'auto') -> dict:
        metadata = metadata or {}
        assert_public(metadata)
        # Snapshot before hashing: a caller cannot mutate an admitted request later.
        request = self._request(messages, metadata, tool_declarations, tool_choice)
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
            if self.reserve_budget and self.token_budget is not None:
                maximum = self.model_config["max_output_tokens"]
                if maximum is None:
                    raise ValueError("Campaign admission requires explicit max_output_tokens")
                # Conservative text-only allowance: UTF-8 bytes plus output/thought
                # headroom. Actual billing always uses reported provider tokens.
                reserve = (len(json.dumps(request, ensure_ascii=False).encode('utf-8'))
                           if self.model_config.get('research_tool_protocol') == 'native_framework_v4'
                           else sum(len(m["content"].encode("utf-8")) for m in messages))
                reserve += 2*maximum + 1024
                if info["reported_tokens"] + reserve > self.token_budget:
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
                # Exception messages may contain headers/URLs/credentials. Do not persist them.
                raise ExecutionPaused(
                    f"Provider attempt paused ({type(exc).__name__}); inspect ledger before new work"
                ) from exc
            elapsed = time.time() - started
            permitted_call = False
            if (self.model_config.get('research_tool_protocol') == 'native_framework_v4'
                    and isinstance(raw, dict) and isinstance(raw.get('tool_calls'), list)
                    and raw['tool_calls'] and request['tool_choice'] != 'none'):
                names = {row['name'] for row in request['tool_declarations']}
                ids = [call.get('id') for call in raw['tool_calls'] if isinstance(call, dict)
                       and call.get('id') is not None]
                permitted_call = (all(isinstance(call, dict) and isinstance(call.get('name'), str)
                                      and call['name'] in names
                                      and isinstance(call.get('args'), dict)
                                      and (call.get('id') is None or
                                           isinstance(call['id'], str) and bool(call['id']))
                                      for call in raw['tool_calls'])
                                  and len(ids) == len(set(ids)))
            if (self.model_config.get('research_tool_protocol') in {'native_review_v2', 'native_structure_v3'}
                    and metadata.get('phase') == 'analysis' and isinstance(raw, dict)
                    and isinstance(raw.get('tool_calls'), list) and raw['tool_calls']):
                if self.model_config['research_tool_protocol'] == 'native_structure_v3':
                    from .research.structure_review import normalize_calls
                else:
                    from .research.review import normalize_calls
                action = normalize_calls(raw['tool_calls'], metadata.get('role'))
                if action is not None:
                    raw = {**raw, 'provider_final_text': raw.get('final_text', ''),
                           'final_text': json.dumps(action)}
                    permitted_call = True
            if (self.model_config.get('research_tool_protocol') == 'native_v1'
                    and metadata.get('phase') == 'analysis' and isinstance(raw, dict)
                    and isinstance(raw.get('tool_calls'), list) and len(raw['tool_calls']) == 1):
                from .research.native_tools import declarations
                call = raw['tool_calls'][0]
                if (isinstance(call, dict) and call.get('name') in {
                        d['name'] for d in declarations(metadata.get('role'), metadata)}
                        and isinstance(call.get('args'), dict)):
                    raw = {**raw, 'provider_final_text': raw.get('final_text', ''),
                           'final_text': json.dumps({'tool': call['name'], 'arguments': call['args']})}
                    permitted_call = True
            reason, tokens, status = _classify(raw, allow_tool_call=permitted_call)
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


def _classify(raw: Any, *, allow_tool_call=False) -> tuple[str | None, int | None, str]:
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
    if raw.get("tool_calls") and not allow_tool_call:
        return None, tokens, "terminal_failure"
    if raw.get("status") in ("error", "failed", "blocked", "truncated"):
        return None, tokens, "terminal_failure"
    if (not isinstance(raw.get("final_text"), str)
            or not raw["final_text"].strip() and not allow_tool_call):
        return "Provider returned no identifiable final answer", tokens, "unknown_response"
    return None, tokens, "ok"


def _safe_api_error_diagnostic(exc, config):
    """Retain bounded provider diagnostics, never response bodies or headers.

    Only call for the SDK's APIError family. Generic exception strings can embed
    arbitrary request data, so they remain excluded from the provider ledger.
    """
    result = {}
    status = getattr(exc, 'status', None)
    if isinstance(status, str) and re.fullmatch(r'[A-Z][A-Z0-9_]{0,63}', status):
        result['provider_status'] = status
    message = getattr(exc, 'message', None)
    if not isinstance(message, str) or not message:
        return result
    credential = os.environ.get(config['api_key_env'])
    if credential:
        for value in {credential, quote(credential, safe='')}:
            message = message.replace(value, '[REDACTED_CREDENTIAL]')
    message = re.sub(r'https?://[^\s\"\'<>]+', '[REDACTED_URL]', message, flags=re.I)
    message = re.sub(
        r'(?im)^\s*(?:authorization|proxy-authorization|cookie|set-cookie|x-goog-api-key|x-api-key)\s*[:=].*$',
        '[REDACTED_HEADER]', message)
    message = re.sub(r'(?i)\bbearer\s+[^\s\"\',;]+', 'Bearer [REDACTED_CREDENTIAL]', message)
    message = re.sub(
        r'''(?ix)(["']?(?:api[-_]?key|authorization|access[-_]?token|password|secret)["']?\s*[:=]\s*)
            (?:"[^"]*"|'[^']*'|[^\s,;]+)''',
        r'\1[REDACTED_CREDENTIAL]', message)
    result['message'] = ' '.join(message.split())[:1600]
    return result


def google_stream_transport(request: dict) -> dict:
    """One Google SDK stream with retries and automatic function calls disabled."""
    from google import genai
    from google.genai import errors, types

    config = request["model_config"]
    api_key = os.environ.get(config["api_key_env"])
    if not api_key:
        raise RuntimeError(f"Missing environment variable {config['api_key_env']}")
    http = {"timeout": int(generation_timeout_seconds(config["timeout_s"]) * 1000),
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
    if "generation_seed" in config:
        generation["seed"] = config["generation_seed"]
    if config.get('function_calling_mode') == 'NONE':
        generation['tool_config'] = types.ToolConfig(
            function_calling_config=types.FunctionCallingConfig(mode='NONE'))
    if config.get('response_mime_type'):
        generation['response_mime_type'] = config['response_mime_type']
    framework = config.get('research_tool_protocol') == 'native_framework_v4'
    if framework:
        rows = request.get('tool_declarations', [])
        choice = request.get('tool_choice', 'auto')
        tool_config = {'mode': 'NONE'}
        if rows:
            # parameters_json_schema preserves the exact LangChain/Pydantic schema,
            # including nested objects, without the legacy Google Schema coercion.
            generation['tools'] = [types.Tool(function_declarations=[
                types.FunctionDeclaration(
                    name=row['name'], description=row.get('description', ''),
                    parameters_json_schema=row.get('parameters', {'type': 'object'}))
                for row in rows])]
            tool_config = {'mode': {'auto': 'AUTO', 'any': 'ANY', 'none': 'NONE'}.get(choice, 'ANY')}
            if choice not in {'auto', 'any', 'none'}:
                tool_config['allowed_function_names'] = [choice]
        generation['tool_config'] = types.ToolConfig(
            function_calling_config=types.FunctionCallingConfig(**tool_config))
    if config.get('research_tool_protocol') in {'native_v1', 'native_review_v2', 'native_structure_v3'}:
        if request.get('metadata', {}).get('phase') == 'analysis':
            if config['research_tool_protocol'] == 'native_structure_v3':
                from .research.structure_review import declarations
            elif config['research_tool_protocol'] == 'native_review_v2':
                from .research.review import declarations
            else:
                from .research.native_tools import declarations
            rows = (declarations(request['metadata'].get('role'))
                    if config['research_tool_protocol'] in {'native_review_v2', 'native_structure_v3'}
                    else declarations(request['metadata'].get('role'), request['metadata']))
            generation['tools'] = [types.Tool(function_declarations=[
                types.FunctionDeclaration(**row) for row in rows])]
            generation['tool_config'] = types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(mode='AUTO'))
        else:
            generation['tool_config'] = types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(mode='NONE'))
    if config["max_output_tokens"] is not None:
        generation["max_output_tokens"] = config["max_output_tokens"]
    if config["thinking_level"] is not None:
        generation["thinking_config"] = types.ThinkingConfig(
            thinking_level=config["thinking_level"], include_thoughts=config["include_thoughts"]
        )
    if framework:
        contents = []
        previous_role = None
        for message in request['messages']:
            if message['role'] == 'system':
                continue
            parts = [types.Part.model_validate(part) for part in message.get('native_parts', [])]
            if not parts:
                parts = [types.Part(text=message['content'])]
            # All results from one parallel model turn form one user Content.
            # Never flatten model parts: thought signatures belong to exact parts.
            if message['role'] == 'tool' and previous_role == 'tool':
                contents[-1].parts.extend(parts)
            else:
                contents.append(types.Content(
                    role='model' if message['role'] == 'assistant' else 'user', parts=parts))
            previous_role = message['role']
    else:
        contents = [
            types.Content(role="model" if message["role"] == "assistant" else "user",
                          parts=[types.Part(text=message["content"])])
            for message in request["messages"] if message["role"] != "system"
        ]
    text_parts, thoughts, tool_calls = [], [], []
    native_parts = []
    finish, usage, response_metadata = None, None, {"effective_timeout_s": http["timeout"] / 1000}

    def snapshot():
        normalized_usage = None if usage is None else {
            "input_tokens": usage.prompt_token_count,
            "output_tokens": usage.candidates_token_count,
            "total_tokens": usage.total_token_count,
            "reasoning_tokens": usage.thoughts_token_count or 0,
            "model_calls": 1,
        }
        result = {
            "final_text": "".join(text_parts), "thought_summary": "".join(thoughts),
            "finish_reason": finish, "usage_summary": normalized_usage,
            "response_metadata": response_metadata, "tool_calls": tool_calls,
        }
        if framework:
            result['native_parts'] = native_parts
        return result

    try:
        with genai.Client(api_key=api_key, http_options=types.HttpOptions(**http)) as client:
            arguments = dict(model=request["model"], contents=contents,
                             config=types.GenerateContentConfig(**generation))
            chunks = (
                [client.models.generate_content(**arguments)]
                if config.get("transport_mode", "stream") == "nonstream"
                else client.models.generate_content_stream(**arguments)
            )
            for chunk in chunks:
                if chunk.usage_metadata is not None:
                    usage = chunk.usage_metadata
                for candidate in chunk.candidates or []:
                    if candidate.finish_reason is not None:
                        finish = getattr(candidate.finish_reason, "value", str(candidate.finish_reason))
                    for part in (candidate.content.parts if candidate.content else []) or []:
                        if framework:
                            native_parts.append(part.model_dump(mode='json', exclude_none=True))
                        if part.text:
                            (thoughts if part.thought else text_parts).append(part.text)
                        if part.function_call:
                            tool_calls.append(part.function_call.model_dump(mode="json", exclude_none=True)
                                              if framework else part.function_call.model_dump(mode="json"))
                if chunk.prompt_feedback:
                    response_metadata["prompt_feedback"] = chunk.prompt_feedback.model_dump(mode="json")
                if chunk.model_version:
                    response_metadata["model_version"] = chunk.model_version
                if getattr(chunk, "response_id", None):
                    response_metadata["response_id"] = chunk.response_id
    except Exception as exc:
        # SDK API errors expose a separate diagnostic message; sanitize it without
        # serializing the full error/details/HTTP response. Other exceptions retain
        # types and status codes only. A failed request is never retried here.
        response_metadata["transport_error"] = {
            "exception_type": type(exc).__name__,
            "cause_type": type(exc.__cause__).__name__ if exc.__cause__ else None,
        }
        code = getattr(exc, "code", None)
        if type(code) is int:
            response_metadata["transport_error"]["status_code"] = code
        if isinstance(exc, errors.APIError):
            response_metadata['transport_error'].update(_safe_api_error_diagnostic(exc, config))
        raise ProviderStreamInterrupted(snapshot()) from exc
    return snapshot()
