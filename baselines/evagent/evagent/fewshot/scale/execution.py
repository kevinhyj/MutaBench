"""Durable shared admission; each wire attempt retains its own immutable gateway ledger.

Unknown costs retain reservations across restarts. A request has at most three
wire submissions in total (including its optional format repair), a stricter
bound than three attempts for each repair independently.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path

from evagent.fewshot.protocol import digest
from evagent.fewshot.runner import ContextTooLong, repair_prompt, save
from evagent.integrated.chunked_scoring import ScoringResponseValidator
from evagent.integrated.gateway import RequestGateway
from evagent.integrated.self_check import SelfCheckPolicy, run_self_check
from evagent.runtime import ExecutionPaused

RETRYABLE = {"RemoteDisconnected", "RemoteProtocolError", "ReadError", "WriteError",
             "ConnectError", "ConnectionError", "ConnectTimeout", "ReadTimeout",
             "TimeoutError", "PoolTimeout", "ServerError", "URLError", "TimeoutException"}
BLOCKS = {"OTHER", "SAFETY", "BLOCKLIST", "PROHIBITED_CONTENT", "IMAGE_SAFETY", "RECITATION", "SPII"}


class StopCampaign(Exception):
    pass


def reserve_for(system, prompt, model):
    return len(system.encode()) + len(prompt.encode()) + 2 * model["max_output_tokens"] + 1024


def read_leaf(path):
    if not path.exists():
        return None
    db = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=30)
    db.row_factory = sqlite3.Row
    try:
        db.execute("PRAGMA query_only=ON")
        row = db.execute("SELECT * FROM attempts ORDER BY id DESC LIMIT 1").fetchone()
        return dict(row) if row else None
    finally:
        db.close()


def classify_leaf(row):
    if row is None:
        return {"status": "not_submitted", "tokens": 0, "unknown": 0, "retryable": 1,
                "error_type": "not_submitted", "response": {}}
    response = json.loads(row.get("response_json") or "{}")
    metadata = response.get("response_metadata") or {}
    error = metadata.get("transport_error") or {}
    kind = error.get("exception_type") or row.get("error_type")
    code = error.get("status_code")
    status = row["status"]
    if status == "inflight":
        status, kind = "interrupted_unknown_cost", "ProcessExit"
    block = (metadata.get("prompt_feedback") or {}).get("block_reason")
    blocked = block in BLOCKS or status == "blocked"
    retry = not blocked and (kind in RETRYABLE or kind == "ProcessExit" or
                            code in {408, 429, 500, 502, 503, 504})
    tokens = row.get("tokens")
    tokens = tokens if type(tokens) is int and tokens >= 0 else 0
    # Unknown finish may have complete token usage, but preserve the upper bound
    # whenever the gateway did not certify a terminal response.
    unknown = int("unknown" in status or status == "inflight")
    return {"status": "blocked" if blocked else status, "tokens": tokens,
            "unknown": unknown, "retryable": int(retry), "error_type": kind,
            "response": response, "http_code": code}


class Journal:
    def __init__(self, root, budget, fingerprint):
        self.root, self.budget, self.fingerprint = Path(root), budget, fingerprint
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.db = sqlite3.connect(self.root / "campaign.sqlite", check_same_thread=False, timeout=30)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS config (id INTEGER PRIMARY KEY, budget INTEGER, fingerprint TEXT);
            CREATE TABLE IF NOT EXISTS attempts (
              id INTEGER PRIMARY KEY, logical TEXT, prompt TEXT, batch INTEGER,
              reserve INTEGER, tokens INTEGER DEFAULT 0, unknown INTEGER DEFAULT 1,
              status TEXT, retryable INTEGER DEFAULT 0, started REAL, ended REAL,
              error_type TEXT, http_code INTEGER);
            CREATE TABLE IF NOT EXISTS batches (
              id INTEGER PRIMARY KEY, workers INTEGER, started REAL, ended REAL);
        """)
        old = self.db.execute("SELECT * FROM config").fetchone()
        if old and (old["budget"] != budget or old["fingerprint"] != fingerprint):
            raise ValueError("Campaign budget/fingerprint changed")
        self.db.execute("INSERT OR IGNORE INTO config VALUES (1,?,?)", (budget, fingerprint))
        self.db.commit()

    def check(self):
        if (self.root / "PAUSE").exists():
            raise StopCampaign("operator_pause")
        if (self.root / "STOP.json").exists():
            raise StopCampaign("terminal_stop")

    def summary(self):
        with self.lock:
            all_rows = list(self.db.execute("SELECT * FROM attempts"))
        known = sum(r["tokens"] for r in all_rows)
        unknown = sum(max(0, r["reserve"] - r["tokens"]) for r in all_rows
                      if r["unknown"] and r["status"] != "inflight")
        inflight = sum(max(0, r["reserve"] - r["tokens"]) for r in all_rows if r["status"] == "inflight")
        from collections import Counter
        return {"known_tokens": known, "unknown_reserve": unknown, "inflight_reserve": inflight,
                "total_debit": known + unknown + inflight, "budget": self.budget,
                "remaining": self.budget - known - unknown - inflight,
                "attempts": len(all_rows), "inflight": sum(r["status"] == "inflight" for r in all_rows),
                "unknown_attempts": sum(r["unknown"] and r["status"] != "inflight" for r in all_rows),
                "attempt_statuses": dict(Counter(r["status"] for r in all_rows)),
                "error_types": dict(Counter(r["error_type"] for r in all_rows if r["error_type"]))}

    def leaf(self, aid):
        return self.root / "attempts" / f"{aid:07d}" / "provider"

    def history(self, logical, prompt=None):
        with self.lock:
            return [dict(r) for r in self.db.execute("SELECT * FROM attempts WHERE logical=? ORDER BY id", (logical,))
                    if prompt is None or r["prompt"] == prompt]

    def admit(self, logical, prompt, reserve, batch):
        with self.lock:
            self.check()
            self.db.execute("BEGIN IMMEDIATE")
            if sum(r["status"] != "not_submitted" for r in self.history(logical)) >= 3:
                self.db.rollback()
                return None
            if self.summary()["remaining"] < reserve:
                self.db.rollback()
                save(self.root / "STOP.json", {"reason": "budget_reservation_exhausted", "time": time.time()})
                raise StopCampaign("budget_reservation_exhausted")
            cursor = self.db.execute(
                "INSERT INTO attempts(logical,prompt,batch,reserve,status,started) VALUES(?,?,?,?, 'inflight',?)",
                (logical, prompt, batch, reserve, time.time()))
            aid = cursor.lastrowid
            self.db.commit()
            return aid

    def settle(self, aid):
        value = classify_leaf(read_leaf(self.leaf(aid) / "request_ledger.sqlite"))
        with self.lock:
            self.db.execute("UPDATE attempts SET status=?,tokens=?,unknown=?,retryable=?,ended=?,error_type=?,http_code=? WHERE id=?",
                            (value["status"], value["tokens"], value["unknown"], value["retryable"], time.time(),
                             value["error_type"], value.get("http_code"), aid))
            self.db.commit()
        return value

    def recover(self):
        # Only called under the process-lifetime lock, after the old process is gone.
        with self.lock:
            ids = [r[0] for r in self.db.execute("SELECT id FROM attempts WHERE status='inflight'")]
        for aid in ids:
            self.settle(aid)
        with self.lock:
            self.db.execute("UPDATE batches SET ended=? WHERE ended IS NULL", (time.time(),))
            self.db.commit()
        return ids

    def start_batch(self, workers):
        with self.lock:
            if self.summary()["inflight"]:
                raise ValueError("Cannot change workers with inflight attempts")
            cur = self.db.execute("INSERT INTO batches(workers,started) VALUES(?,?)", (workers, time.time()))
            self.db.commit()
            return cur.lastrowid

    def end_batch(self, batch):
        with self.lock:
            self.db.execute("UPDATE batches SET ended=? WHERE id=?", (time.time(), batch))
            self.db.commit()

    def concurrency(self):
        with self.lock:
            batch = self.db.execute("SELECT * FROM batches ORDER BY id DESC LIMIT 1").fetchone()
            recent = list(self.db.execute("SELECT * FROM attempts WHERE status!='not_submitted' ORDER BY id DESC LIMIT 50"))
        workers = batch["workers"] if batch else 8
        if batch:
            current_errors = [r for r in recent if r["batch"] == batch["id"] and
                              (r["retryable"] or r["http_code"] == 429)]
            if current_errors and (any(r["http_code"] == 429 for r in current_errors) or
                                   (len(recent) == 50 and sum(bool(r["retryable"]) for r in recent) / 50 >= .1)):
                return max(2, workers // 2)
        if len(recent) >= 20 and all(r["status"] == "ok" for r in recent[:20]):
            return 15
        return workers

    def snapshot(self):
        destination = self.root / "snapshots" / (str(int(time.time())) + ".sqlite")
        destination.parent.mkdir(exist_ok=True)
        with self.lock, sqlite3.connect(destination) as target:
            self.db.backup(target)
        return destination


class Executor:
    def __init__(self, journal, model, checker, *, transport=None, retry_wait=60):
        self.journal, self.model, self.checker = journal, model, checker
        self.transport, self.retry_wait = transport, retry_wait
        self.key_locks = {}
        self.lock = threading.Lock()

    def generate(self, request, prompt, batch):
        logical = request["request_hash"]
        prompt_hash = digest([request["system"], prompt])
        history = self.journal.history(logical, prompt_hash)
        for old in history:
            if old["status"] == "ok":
                return classify_leaf(read_leaf(self.journal.leaf(old["id"]) / "request_ledger.sqlite"))["response"]
        while True:
            self.journal.check()
            history = self.journal.history(logical, prompt_hash)
            if history:
                last = history[-1]
                if not last["retryable"]:
                    return {"status": last["status"], "error_type": last["error_type"]}
                # Respect the backoff after process restarts as well as in-process errors.
                until = (last["ended"] or last["started"]) + self.retry_wait
                while time.time() < until:
                    self.journal.check()
                    time.sleep(min(1, until - time.time()))
            if sum(r["status"] != "not_submitted" for r in self.journal.history(logical)) >= 3:
                return {"status": "transport_attempts_exhausted"}
            try:
                check = self.checker(request["system"], prompt)
            except ContextTooLong:
                return {"status": "context_limit_exceeded"}
            save(self.journal.root / "predictions" / logical / (prompt_hash + ".context.json"), check)
            reservation = reserve_for(request["system"], prompt, self.model)
            aid = self.journal.admit(logical, prompt_hash, reservation, batch)
            if aid is None:
                return {"status": "transport_attempts_exhausted"}
            gateway = None
            try:
                gateway = RequestGateway(model_config=self.model, run_dir=self.journal.leaf(aid),
                         fingerprint=self.journal.fingerprint, token_budget=reservation,
                         reserve_budget=True, transport=self.transport)
                gateway.generate(system=request["system"], user=prompt, run_name="fewshot_scale_v3",
                                 metadata={"logical_request": logical, "batch": batch})
            except ExecutionPaused:
                pass  # The durable leaf, not an exception string, determines recovery.
            finally:
                if gateway is not None:
                    gateway.close()
            result = self.journal.settle(aid)
            if result["status"] == "ok":
                return result["response"]
            if not result["retryable"]:
                return {**result["response"], "status": result["status"], "error_type": result["error_type"]}

    def run(self, request, batch):
        key = request["request_hash"]
        with self.lock:
            mutex = self.key_locks.setdefault(key, threading.Lock())
        with mutex:
            self.journal.check()
            folder = self.journal.root / "predictions" / key
            path = folder / "result.json"
            from .workflow import cached
            mode = "live" if self.transport is None else "offline_test"
            if cached(self.journal.root, request, mode) is not None:
                return
            save(folder / "request.json", request)
            result = run_self_check(original_prompt=request["user"],
                     generate=lambda prompt, attempt: self.generate(request, prompt, batch),
                     validator=ScoringResponseValidator(request["expected_ids"]),
                     policy=SelfCheckPolicy(max_repairs=1), repair_prompt_builder=repair_prompt)
            result.update(request_hash=key, execution_mode="live" if self.transport is None else "offline_test")
            result["result_digest"] = digest(result)
            save(path, result)
