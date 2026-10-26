"""Durable request cache and shared soft budget; no credentials are persisted."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Callable


class ExecutionPaused(RuntimeError):
    """Infrastructure/budget stop; deliberately not a scientific ModelError."""


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


class RequestStore:
    def __init__(self, path: Path, limit: int):
        self.pause_path = path.parent / "PAUSE"
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS requests (key TEXT PRIMARY KEY, "
            "status TEXT, raw TEXT, tokens INTEGER, started REAL, ended REAL)"
        )
        # A prior in-flight request may have been billed; never treat it as zero-cost.
        self.db.execute(
            "UPDATE requests SET status='interrupted_unknown_cost' WHERE status='inflight'"
        )
        self.db.commit()
        self.lock = threading.RLock()
        self.limit = limit
        self.stop = threading.Event()

    def cached(self, key: str) -> dict | None:
        with self.lock:
            row = self.db.execute(
                "SELECT raw FROM requests WHERE key=? AND status='complete'", (key,)
            ).fetchone()
            return json.loads(row[0]) if row else None

    def summary(self) -> dict:
        with self.lock:
            rows = self.db.execute(
                "SELECT status,COUNT(*),SUM(tokens) FROM requests GROUP BY status"
            ).fetchall()
            retry_unknown = self.db.execute(
                "SELECT COALESCE(SUM(MAX(COALESCE(json_extract(raw, '$.attempts'),1)-1,0)),0) "
                "FROM requests WHERE status='complete'"
            ).fetchone()[0]
        return {
            "soft_limit": self.limit,
            "reported_tokens": sum(r[2] or 0 for r in rows),
            "requests_by_status": {r[0]: r[1] for r in rows},
            "unreported_retry_attempts": retry_unknown,
        }

    def invoke(self, key: str, call: Callable[[], dict]) -> dict:
        with self.lock:
            cached = self.cached(key)
            if cached is not None:
                return {**cached, "cache_hit": True, "request_sha256": key}
            if (
                self.pause_path.exists()
                or self.stop.is_set()
                or self.summary()["reported_tokens"] >= self.limit
            ):
                raise ExecutionPaused("Request admission paused: infrastructure or token budget")
            prior = self.db.execute("SELECT status FROM requests WHERE key=?", (key,)).fetchone()
            if prior and prior[0] != "complete":
                # Retain the earlier unknown-cost attempt rather than overwrite it.
                self.db.execute(
                    "UPDATE requests SET key=? WHERE key=?",
                    (key + ":attempt:" + str(time.time_ns()), key),
                )
            self.db.execute(
                "INSERT INTO requests VALUES (?, 'inflight', NULL, NULL, ?, NULL)",
                (key, time.time()),
            )
            self.db.commit()
        try:
            raw = call()
        except Exception as error:
            self.stop.set()
            with self.lock:
                self.db.execute(
                    "UPDATE requests SET status='failed_unknown_cost', ended=? WHERE key=?",
                    (time.time(), key),
                )
                self.db.commit()
            detail = str(error) if isinstance(error, ExecutionPaused) else type(error).__name__
            raise ExecutionPaused(f"Provider request paused: {detail}") from error
        usage = raw.get("usage") or {}
        tokens = usage.get("totalTokenCount")
        with self.lock:
            self.db.execute(
                "UPDATE requests SET status='complete', raw=?, tokens=?, ended=? WHERE key=?",
                (json.dumps(raw), tokens, time.time(), key),
            )
            self.db.commit()
            if tokens is None:
                self.stop.set()
        return {**raw, "cache_hit": False, "request_sha256": key}


STORE: RequestStore | None = None


def configure_store(path: Path, limit: int = 5_000_000) -> RequestStore:
    global STORE
    STORE = RequestStore(path, limit)
    return STORE
