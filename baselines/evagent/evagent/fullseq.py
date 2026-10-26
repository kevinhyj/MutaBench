"""Full-sequence CR, fixed B25, durable request-level execution and shared budget.

python -m evagent.fullseq --output-root experiments/cr_fullseq_b25_v1
"""

from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import math
import os
import re
import sqlite3
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from . import model as model_module
from . import runtime
from .compact import validate_frozen_data
from .experiments import audit_baseline, correlation, reference
from .graph import ASSAY_PROMPT, ASSAY_SYSTEM, VARIANT_SYSTEM, _canonical_id, model_metadata
from .main import (
    _clean_metadata,
    _sample_hash_from_subset,
    add_shared_arguments,
    build_config,
    configure_tracing,
    nested_subsample_split,
)
from .model import GeminiAdapter, configure_api_concurrency, extract_json
from .schemas import AssayMemory, VariantEvidence

PROTOCOL = "cr_fullseq_b25_v1"
SIZES = (50, 100, 250, 500)
CR_PROMPT = """Assay metadata:
{metadata}

Assay memory:
{memory}

Complete wild-type sequence:
{wt}

Analyze every candidate below. Return exactly:
{{"variants":[{{"variant_id":"M001","mutation":"...",
"direction":"beneficial|neutral|deleterious|uncertain","predicted_score":0.0,
"mechanisms":["..."],"evidence":[{{"evidence_id":"sequence:M001",
"evidence_type":"sequence","claim":"..."}}],"confidence":0.0}}]}}

Candidates with complete mutant sequences (not truncated):
{candidates}"""


def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{threading.get_ident()}.tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str, allow_nan=False))
    tmp.replace(path)


def clean(value):
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def validate_evidence(data, variants):
    if not isinstance(data, dict) or not isinstance(data.get("variants"), list):
        raise ValueError("Missing variants array")
    expected = {v["variant_id"]: v for v in variants}
    parsed = {}
    for raw in data["variants"]:
        if not isinstance(raw, dict):
            raise ValueError("Variant must be an object")
        vid = _canonical_id(raw.get("variant_id"), set(expected))
        if vid is None or vid in parsed:
            raise ValueError("Unknown or duplicate ID")
        for key, lo, hi in (("predicted_score", -1, 1), ("confidence", 0, 1)):
            value = raw.get(key)
            if isinstance(value, bool) or not isinstance(value, (float, int)):
                raise ValueError(f"Missing/non-numeric {key}")
            if not math.isfinite(value) or not lo <= value <= hi:
                raise ValueError(f"Invalid {key}")
        if raw.get("mutation") != expected[vid]["mutation"]:
            raise ValueError("Mutation does not match supplied candidate")
        value = VariantEvidence.model_validate({**raw, "variant_id": vid})
        if value.fallback:
            raise ValueError("Fallback is not model evidence")
        parsed[vid] = value.model_dump()
    if set(parsed) != set(expected):
        raise ValueError("Missing variant IDs")
    return [parsed[v["variant_id"]] for v in variants]


def full_prompt(metadata, memory, wt, variants):
    # Explicit safe projection: never serialize experimental labels to a model.
    candidates = [{k: v[k] for k in ("variant_id", "mutation", "sequence")} for v in variants]
    return CR_PROMPT.format(
        metadata=canonical(model_metadata(metadata)),
        memory=canonical(memory),
        wt=wt,
        candidates=canonical(candidates),
    )


class TerminalRequest(Exception):
    """Known unsupported input or exhausted transport attempts, not a ranking fallback."""


class Ledger(runtime.RequestStore):
    """One retry layer; every wire attempt persisted, including unknown-cost failures."""

    def __init__(self, path, limit, protocol):
        super().__init__(path, limit)
        self.protocol = protocol
        self.protocol_pause = None
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS request_protocol (key TEXT PRIMARY KEY, protocol TEXT)"
        )
        self.db.execute("INSERT OR IGNORE INTO request_protocol SELECT key, 'legacy' FROM requests")
        self.db.commit()
        self.failures = 0
        self.cool_until = 0.0

    def check_admission(self):
        if (
            self.stop.is_set()
            or self.pause_path.exists()
            or (self.protocol_pause and self.protocol_pause.exists())
            or self.summary()["reported_tokens"] >= self.limit
        ):
            raise runtime.ExecutionPaused("Budget/manual/fatal pause")

    def invoke(self, key, call):
        # A transport retry must not monopolize a benchmark worker for an hour.
        # The attempt count remains durable in SQLite, so a restart cannot bypass
        # the five-attempt ceiling. This is operational policy, not model logic.
        delays = tuple(
            int(value)
            for value in os.environ.get("EVAGENT_RETRY_DELAYS_S", "0,15,60,180,300").split(",")
        )
        if not delays or any(delay < 0 for delay in delays):
            raise ValueError("EVAGENT_RETRY_DELAYS_S must be comma-separated non-negative seconds")
        for delay in delays:
            with self.lock:
                cached = self.cached(key)
                if cached:
                    return {**cached, "cache_hit": True, "request_sha256": key}
                attempts = self.db.execute(
                    "SELECT COUNT(*) FROM requests WHERE key=? OR key LIKE ?",
                    (key, key + ":attempt:%"),
                ).fetchone()[0]
                if attempts >= 5:
                    raise TerminalRequest("transport_retry_exhausted")
                self.check_admission()
            until = max(time.time() + delay, self.cool_until)
            while time.time() < until:
                self.check_admission()
                time.sleep(min(2, max(0, until - time.time())))
            with self.lock:
                self.check_admission()
                prior = self.db.execute(
                    "SELECT status FROM requests WHERE key=?", (key,)
                ).fetchone()
                if prior:
                    renamed = key + ":attempt:" + str(time.time_ns())
                    self.db.execute("UPDATE requests SET key=? WHERE key=?", (renamed, key))
                    self.db.execute("UPDATE request_protocol SET key=? WHERE key=?", (renamed, key))
                self.db.execute(
                    "INSERT INTO requests VALUES (?, 'inflight', NULL, NULL, ?, NULL)",
                    (key, time.time()),
                )
                self.db.execute("INSERT INTO request_protocol VALUES (?, ?)", (key, self.protocol))
                self.db.commit()
            try:
                raw = call()
            except Exception as error:
                message = str(error)
                secret = os.environ.get("GEMINI_API_KEY")
                safe_message = message.replace(secret, "[REDACTED]") if secret else message
                with self.lock:
                    self.db.execute(
                        "UPDATE requests SET status='failed_unknown_cost',raw=?,ended=? WHERE key=?",
                        (
                            json.dumps(
                                {"error_type": type(error).__name__, "error": safe_message[:600]}
                            ),
                            time.time(),
                            key,
                        ),
                    )
                    self.db.commit()
                    self.failures += 1
                    if self.failures >= 6:
                        self.cool_until = max(self.cool_until, time.time() + 300)
                if any(
                    s in message.lower()
                    for s in (
                        "401",
                        "403",
                        "api_key_invalid",
                        "api key not valid",
                        "permission_denied",
                        "not set",
                    )
                ):
                    self.stop.set()
                    raise runtime.ExecutionPaused(
                        "Authentication/configuration failure; manual attention required"
                    ) from error
                if any(
                    s in message.lower()
                    for s in ("http 400", "http error 400", "invalid_argument", "404")
                ):
                    raise TerminalRequest("nonretryable_provider_request") from error
                continue
            with self.lock:
                tokens = (raw.get("usage") or {}).get("totalTokenCount")
                self.db.execute(
                    "UPDATE requests SET status='complete',raw=?,tokens=?,ended=? WHERE key=?",
                    (json.dumps(raw), tokens, time.time(), key),
                )
                self.db.commit()
                self.failures = 0
                if tokens is None:
                    self.stop.set()
            return {**raw, "cache_hit": False, "request_sha256": key}
        raise TerminalRequest("transport_retry_exhausted")

    def accounting(self):
        with self.lock:
            rows = self.db.execute(
                "SELECT p.protocol,r.status,COUNT(*),SUM(r.tokens) FROM requests r LEFT JOIN request_protocol p ON p.key=r.key GROUP BY p.protocol,r.status"
            ).fetchall()
        return {
            **self.summary(),
            "by_protocol": [
                {"protocol": p, "status": s, "requests": n, "tokens": t} for p, s, n, t in rows
            ],
        }


class Service:
    def __init__(self, config, root, store):
        self.config, self.root, self.store = config, root, store
        self.adapter = GeminiAdapter(config.model, config.rg_llm_root, namespace=PROTOCOL)
        self.limit = 1_048_576

    def metadata_request(self, suffix, payload=None):
        endpoint = self.config.model.endpoint.removesuffix(":generateContent") + suffix
        opener = urllib.request.build_opener(
            urllib.request.ProxyHandler(
                {"http": self.config.model.proxy, "https": self.config.model.proxy}
            )
        )
        req = urllib.request.Request(
            endpoint,
            data=None if payload is None else json.dumps(payload).encode(),
            headers={
                "x-goog-api-key": os.environ[self.config.model.api_key_env],
                "Content-Type": "application/json",
            },
        )
        for attempt in range(3):
            try:
                with model_module._API_SEMAPHORE:
                    with opener.open(req, timeout=45) as response:
                        return json.load(response)
            except urllib.error.HTTPError as error:
                if error.code in (401, 403):
                    self.store.stop.set()
                    raise runtime.ExecutionPaused("Metadata authentication failure") from error
                if error.code < 500 and error.code != 429:
                    raise TerminalRequest(f"metadata_http_{error.code}") from error
            except OSError:
                pass
            if attempt < 2:
                time.sleep(5 * (attempt + 1))
        raise TerminalRequest("token_preflight_unavailable")

    def preflight(self):
        info = self.metadata_request("")
        self.limit = int(info["inputTokenLimit"])
        save(
            self.root / "provider_limits.json",
            {k: info.get(k) for k in ("name", "version", "inputTokenLimit", "outputTokenLimit")},
        )

    def count(self, system, user):
        digest = runtime.digest([self.config.model.model_id, system, user])
        path = self.root / "token_checks" / (digest + ".json")
        if path.exists():
            count = json.loads(path.read_text())["totalTokens"]
        else:
            response = self.metadata_request(
                ":countTokens",
                {
                    "generateContentRequest": {
                        "model": "models/" + self.config.model.model_id,
                        "systemInstruction": {"parts": [{"text": system}]},
                        "contents": [{"role": "user", "parts": [{"text": user}]}],
                    }
                },
            )
            count = int(response["totalTokens"])
            save(path, {"input_sha256": digest, "totalTokens": count, "limit": self.limit})
        if count > self.limit:
            raise TerminalRequest("context_limit_exceeded")
        return count

    def call(self, path, system, user, name, metadata):
        request = {
            "protocol": PROTOCOL,
            "model": self.config.model.public_dict(),
            "system": system,
            "user": user,
            "name": name,
            "metadata": metadata,
        }
        fingerprint = runtime.digest(request)
        request_path = path.with_name(path.stem + "_request.json")
        if path.exists():
            result = json.loads(path.read_text())
            if result["request_hash"] != fingerprint:
                raise runtime.ExecutionPaused("Saved request hash mismatch")
            return result["call"]
        save(request_path, {"request_hash": fingerprint, **request})
        self.count(system, user)
        response = self.adapter._call(system, user, name, metadata).model_dump()
        save(path, {"request_hash": fingerprint, "call": response})
        return response

    def structured(self, folder, system, user, name, metadata, validator):
        calls = []
        error = ""
        for attempt in range(2):
            if attempt:
                system = "Repair JSON formatting only; preserve all supplied IDs and values. Do not invent missing biological predictions."
                user = "Return valid JSON without changing semantic content.\n" + calls[0]["answer"]
            call = self.call(
                folder / f"response_{attempt}.json",
                system,
                user,
                name + f".response{attempt}",
                metadata,
            )
            calls.append(call)
            try:
                data = validator(extract_json(call["answer"]))
                return {"status": "complete", "data": data, "calls": calls}
            except (ValueError, TypeError, KeyError) as exc:
                error = str(exc)
        return {"status": "invalid_output", "error": error, "calls": calls}


class Experiment:
    def __init__(self, args, config, source, store):
        self.args, self.config, self.source, self.store = args, config, source, store
        self.root = Path(args.output_root)
        self.service = Service(config, self.root, store)
        self.metadata = {
            str(r["DMS_ID"]): _clean_metadata(r) for r in reference(config).to_dict("records")
        }
        self.classes = {a["assay"]: a for a in source["assays"]}
        self.started = time.time()

    def prepare(self):
        maximums = {}
        for a in self.source["assays"]:
            maximum = max(n for n in SIZES if n <= a["n_valid"])
            maximums[a["assay"]] = maximum
            subset = nested_subsample_split(
                self.config, a["assay"], maximum, 1, self.root / "samples"
            )
            # Check old frozen prefixes without modifying them.
            for n in SIZES:
                old = Path(self.args.legacy_root) / "splits" / a["assay"] / f"n{n}_b1.json"
                if old.exists() and n <= maximum:
                    old_rows = [
                        (v["id"], v["seq"], float(v["score"]))
                        for v in json.loads(old.read_text())["variants"]
                    ]
                    if _sample_hash_from_subset(old_rows) != _sample_hash_from_subset(subset[:n]):
                        raise runtime.ExecutionPaused("Historical sample prefix mismatch")
        manifest = {
            "protocol": PROTOCOL,
            "sizes": list(SIZES),
            "chunk_size": 25,
            "seed": 1,
            "model": self.config.model.public_dict(),
            "source_manifest_hash": runtime.digest(self.source),
            "prompt_hash": runtime.digest(
                [ASSAY_SYSTEM, str(ASSAY_PROMPT), VARIANT_SYSTEM, CR_PROMPT]
            ),
            "budget_limit": self.args.token_budget,
            "maximums": maximums,
            "eligible_cells": sum(
                sum(n <= a["n_valid"] for n in SIZES) for a in self.source["assays"]
            ),
        }
        path = self.root / "protocol.json"
        if path.exists() and json.loads(path.read_text()) != manifest:
            raise runtime.ExecutionPaused("Frozen protocol changed")
        save(path, manifest)
        self.manifest = manifest

    def variants(self, assay, n):
        sample = json.loads((self.root / "samples" / assay / f"n{n}_b1.json").read_text())
        return [
            {
                "variant_id": f"M{i + 1:03d}",
                "mutation": v["id"],
                "sequence": v["seq"],
                "experimental_score": float(v["score"]),
            }
            for i, v in enumerate(sample["variants"])
        ]

    def memory(self, assay):
        folder = self.root / "assays" / assay
        path = folder / "memory.json"
        if path.exists():
            return json.loads(path.read_text())
        meta = self.metadata[assay]
        wt = str(meta["RAW_CONSTRUCT_SEQ"]).upper()
        user = ASSAY_PROMPT.format(
            metadata=canonical(model_metadata(meta)), length=len(wt), wt_sequence=wt
        )

        def validator(data):
            return AssayMemory.model_validate({"assay_id": assay, **data}).model_dump()

        # Memory reuse requires exact original system/user/config cache evidence; otherwise call once.
        result = self.service.structured(
            folder / "analyst", ASSAY_SYSTEM, user, "assay_analyst", {"assay": assay}, validator
        )
        save(path, result)
        return result

    def chunk(self, assay, memory, variants, index):
        folder = self.root / "assays" / assay / "chunks" / f"c{index:03d}"
        path = folder / "result.json"
        meta = self.metadata[assay]
        wt = str(meta["RAW_CONSTRUCT_SEQ"]).upper()
        fingerprint = runtime.digest(
            [
                memory["data"],
                [{k: v[k] for k in ("variant_id", "mutation", "sequence")} for v in variants],
            ]
        )
        if path.exists():
            result = json.loads(path.read_text())
            if result["input_hash"] != fingerprint:
                raise runtime.ExecutionPaused("Chunk/evidence mismatch")
            return result
        try:
            result = self.service.structured(
                folder,
                VARIANT_SYSTEM,
                full_prompt(meta, memory["data"], wt, variants),
                "CR",
                {"assay": assay, "chunk": index},
                lambda d: validate_evidence(d, variants),
            )
        except TerminalRequest as error:
            result = {"status": str(error), "calls": []}
        result.update(input_hash=fingerprint, chunk=index)
        # Temporary transport/preflight failures are not permanently cached scientific results.
        if result["status"] not in ("token_preflight_unavailable", "transport_retry_exhausted"):
            save(path, result)
        return result

    def global_ranking(self, assay, n, variants):
        from rg_llm import prompt

        folder = self.root / "assays" / assay / "evaluations" / f"n{n}"
        path = folder / "global.json"
        if path.exists():
            return json.loads(path.read_text())
        subset = [(v["mutation"], v["sequence"], v["experimental_score"]) for v in variants]
        meta = self.metadata[assay]
        user, ids = prompt.build_user_prompt(
            model_metadata(meta), str(meta["RAW_CONSTRUCT_SEQ"]).upper(), subset
        )
        expected = runtime.digest([prompt.SYSTEM_PROMPT, user])
        old = (
            Path(self.args.legacy_root)
            / "baseline"
            / self.config.model.model_id
            / f"n{n}"
            / "b1"
            / f"{assay}.json"
        )
        record = None
        if old.exists():
            candidate = json.loads(old.read_text())
            if (
                candidate.get("prompt_sha256") == expected
                and candidate.get("sample_sha256") == _sample_hash_from_subset(subset)
                and candidate.get("model") == self.config.model.model_id
                and candidate.get("thinking_level") == "medium"
            ):
                # Compare generation config through the recorded hash of the historical protocol.
                historical = copy.copy(self.args)
                for repeats, verify in ((3, 16), (1, 0)):
                    historical.repeats, historical.verify_count = repeats, verify
                    if candidate.get("config_sha256") == build_config(historical).fingerprint():
                        record = candidate
                        break
        if record is not None:
            call = {
                "usage": record.get("usage", {}),
                "elapsed_s": record.get("elapsed_s", 0),
                "model_version": record.get("model_version"),
                "request_sha256": record.get("request_sha256"),
            }
            reuse = {
                "source_path": str(old),
                "source_hash": hashlib.sha256(old.read_bytes()).hexdigest(),
            }
        else:
            reuse = None
            try:
                call = self.service.call(
                    folder / "global_response.json",
                    prompt.SYSTEM_PROMPT,
                    user,
                    "Global",
                    {"assay": assay, "N": n},
                )
                record = {"ids": ids, "response_text": call["answer"]}
            except TerminalRequest as error:
                result = {
                    "status": str(error),
                    "calls": [],
                    "sample_hash": _sample_hash_from_subset(subset),
                }
                if str(error) not in ("token_preflight_unavailable", "transport_retry_exhausted"):
                    save(path, result)
                return result
        audit = audit_baseline(record)
        # Original IDs vary in zero padding; normalize by verified candidate order.
        mapping = {original: v["variant_id"] for original, v in zip(ids, variants, strict=True)}
        ranking = [mapping[i] for i in (audit.get("ranking") or [])]
        returned = []
        for token in re.findall(r"\bM\d+\b", record["response_text"], flags=re.I):
            vid = _canonical_id(token, set(ids))
            if vid is not None and mapping[vid] not in returned:
                returned.append(mapping[vid])
        # Prefer the explicit ranking array, not IDs occurring in prose/reasoning.
        try:
            raw = extract_json(record["response_text"])
            values = (
                raw.get("ranking", [])
                if isinstance(raw, dict)
                else raw
                if isinstance(raw, list)
                else []
            )
            returned = []
            for token in values:
                vid = _canonical_id(token, set(ids))
                if vid is not None and mapping[vid] not in returned:
                    returned.append(mapping[vid])
        except (ValueError, TypeError):
            returned = []  # Do not infer rankings from arbitrary explanatory prose.
        if audit["valid"]:
            returned = ranking
        result = {
            "status": "complete" if audit["valid"] else "invalid_output",
            "audit": audit,
            "ranking": ranking,
            "returned_ranking": returned,
            "calls": [call],
            "reuse": reuse,
            "sample_hash": _sample_hash_from_subset(subset),
        }
        save(path, result)
        return result

    def cell(self, assay, n):
        started = time.time()
        variants = self.variants(assay, n)
        folder = self.root / "assays" / assay / "evaluations" / f"n{n}"
        global_result = self.global_ranking(assay, n, variants)
        memory = self.memory(assay)
        chunks = []
        if memory["status"] == "complete":
            with ThreadPoolExecutor(max_workers=4) as pool:
                chunks = list(
                    pool.map(
                        lambda i: self.chunk(assay, memory, variants[i : i + 25], i // 25),
                        range(0, n, 25),
                    )
                )
        evidence = [v for c in chunks if c["status"] == "complete" for v in c["data"]]
        truth = {v["variant_id"]: v["experimental_score"] for v in variants}
        scores = {v["variant_id"]: v["predicted_score"] for v in evidence}
        complete = len(scores) == n and set(scores) == set(truth)
        cr = {
            "status": "complete" if complete else "incomplete",
            "coverage": len(scores) / n,
            "scores": scores,
            "chunk_statuses": [c["status"] for c in chunks],
            "calls": memory.get("calls", [])
            + [call for c in chunks for call in c.get("calls", [])],
            "sample_hash": _sample_hash_from_subset(
                [(v["mutation"], v["sequence"], v["experimental_score"]) for v in variants]
            ),
        }
        save(folder / "cr.json", cr)
        pd.DataFrame(
            [{**v, "predicted_score": scores.get(v["variant_id"])} for v in variants]
        ).drop(columns=["sequence"]).to_csv(folder / "predictions.csv", index=False)
        save(
            folder / "evaluation.json",
            {
                "assay": assay,
                "N": n,
                "protocol": PROTOCOL,
                "seed": 1,
                "B": 25,
                "cr_status": cr["status"],
                "global_status": global_result["status"],
                "wall_s": time.time() - started,
                "completed_at": time.time(),
                "truth": truth,
                "sample_hash": cr["sample_hash"],
            },
        )
        print(f"[CR] {assay} N={n} CR={cr['status']} Global={global_result['status']}", flush=True)

    def report(self):
        rows = []
        returned_rows = []
        probe_rows = []
        reuse = []
        for path in self.root.glob("assays/*/evaluations/n*/evaluation.json"):
            e = json.loads(path.read_text())
            truth = e["truth"]
            folder = path.parent
            cr = json.loads((folder / "cr.json").read_text())
            g = (
                json.loads((folder / "global.json").read_text())
                if (folder / "global.json").exists()
                else {"status": e["global_status"], "calls": []}
            )
            common = {
                "assay": e["assay"],
                "N": e["N"],
                "protocol": PROTOCOL,
                "seed": 1,
                "sample_hash": e["sample_hash"],
                "rna_type": self.classes[e["assay"]]["rna_type"],
                "protein_stability_source": self.classes[e["assay"]]["protein_stability_source"],
                "wt_length": len(str(self.metadata[e["assay"]]["RAW_CONSTRUCT_SEQ"])),
            }
            for method, result in (("CR", cr), ("Global", g)):
                scores = (
                    result.get("scores", {})
                    if method == "CR"
                    else {vid: -i for i, vid in enumerate(result.get("ranking", []))}
                )
                metrics = (
                    correlation(scores, truth)
                    if result["status"] == "complete"
                    else {"spearman": math.nan, "kendall": math.nan}
                )
                calls = result.get("calls", [])
                rows.append(
                    {
                        **common,
                        "method": method,
                        "status": result["status"],
                        "strict_valid": result["status"] == "complete",
                        **metrics,
                        "coverage": result.get(
                            "coverage", len(result.get("returned_ranking", [])) / e["N"]
                        ),
                        "attributed_tokens": sum(
                            (c.get("usage") or {}).get("totalTokenCount", 0) for c in calls
                        ),
                        "attributed_calls": len(calls),
                        "summed_request_s": sum(c.get("elapsed_s", 0) for c in calls),
                        "evaluation_wall_s": e["wall_s"],
                        "model_versions": ";".join(
                            sorted(
                                {
                                    str(c.get("model_version"))
                                    for c in calls
                                    if c.get("model_version")
                                }
                            )
                        ),
                    }
                )
            ids = g.get("returned_ranking", [])
            if len(ids) > 1:
                gs = {v: -i for i, v in enumerate(ids)}
                gm = correlation(gs, truth, ids)
                cm = (
                    correlation(cr["scores"], truth, ids)
                    if set(ids) <= set(cr["scores"])
                    else {"spearman": math.nan, "kendall": math.nan}
                )
                returned_rows.append(
                    {
                        **common,
                        "returned_count": len(ids),
                        "coverage": len(ids) / e["N"],
                        "global_spearman": gm["spearman"],
                        "cr_spearman": cm["spearman"],
                    }
                )
            if g.get("reuse"):
                reuse.append({**common, **g["reuse"]})
            probe = list(truth)[:50]
            for method, values in (
                ("CR", cr.get("scores", {})),
                ("Global", {v: -i for i, v in enumerate(g.get("returned_ranking", []))}),
            ):
                present = [v for v in probe if v in values]
                metrics = (
                    correlation(values, truth, present)
                    if len(present) > 1
                    else {"spearman": math.nan, "kendall": math.nan}
                )
                probe_rows.append(
                    {
                        **common,
                        "method": method,
                        "probe_coverage": len(present) / 50,
                        "full_probe": len(present) == 50,
                        **metrics,
                    }
                )
        out = self.root / "analysis"
        out.mkdir(exist_ok=True)
        frame = pd.DataFrame(rows)
        if len(frame):
            frame.to_csv(out / "experiment_long.csv", index=False)
            frame.groupby(["N", "method", "status"]).size().rename("cells").to_csv(
                out / "status_counts.csv"
            )
            pd.DataFrame(returned_rows).to_csv(out / "returned_subset.csv", index=False)
            pd.DataFrame(probe_rows).to_csv(out / "fixed_probe.csv", index=False)
            paired = []
            scopes = {"all": frame, "non_protein_stability": frame[~frame.protein_stability_source]}
            scopes.update({"rna_type:" + str(k): v for k, v in frame.groupby("rna_type")})
            for scope, data in scopes.items():
                for n, block in data.groupby("N"):
                    pivot = block.pivot(index="assay", columns="method", values="spearman")
                    delta = (pivot.CR - pivot.Global).dropna().to_numpy()
                    if not len(delta):
                        continue
                    ci = (
                        np.quantile(
                            np.random.default_rng(1)
                            .choice(delta, (10000, len(delta)), replace=True)
                            .mean(axis=1),
                            [0.025, 0.975],
                        )
                        if len(delta) > 1
                        else [np.nan, np.nan]
                    )
                    paired.append(
                        {
                            "scope": scope,
                            "N": n,
                            "n_assays": len(delta),
                            "mean_delta": float(delta.mean()),
                            "bootstrap95_low": ci[0],
                            "bootstrap95_high": ci[1],
                        }
                    )
            pd.DataFrame(paired).to_csv(out / "paired_comparisons.csv", index=False)
            all_n = frame.pivot(index="assay", columns=["method", "N"], values="spearman")
            expected_columns = pd.MultiIndex.from_product([["CR", "Global"], SIZES])
            matched_ids = all_n.reindex(columns=expected_columns).dropna().index
            matched = frame[frame.assay.isin(matched_ids)]
            matched.to_csv(out / "all_N_matched_subset.csv", index=False)
            matched.groupby(["N", "method"]).agg(
                n_assays=("assay", "count"), mean_spearman=("spearman", "mean")
            ).to_csv(out / "all_N_matched_curve.csv")
            frame.groupby(["N", "method"]).agg(
                cells=("assay", "count"),
                computable=("spearman", "count"),
                mean_spearman=("spearman", "mean"),
                median_spearman=("spearman", "median"),
                strict_valid_rate=("strict_valid", "mean"),
                mean_coverage=("coverage", "mean"),
            ).to_csv(out / "macro_summary.csv")
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            common_assays = {a["assay"] for a in self.source["assays"] if a["n_valid"] >= 500}
            for field in (
                "spearman",
                "kendall",
                "coverage",
                "strict_valid",
                "attributed_tokens",
                "summed_request_s",
            ):
                fig, ax = plt.subplots(figsize=(6, 4))
                for method, b in frame[frame.assay.isin(common_assays)].groupby("method"):
                    v = b.groupby("N")[field].mean()
                    ax.plot(v.index, v.values, "o-", label=method)
                ax.set(
                    xlabel="N",
                    ylabel=field,
                    title="Common eligible cohort; valid-only accuracy; see denominators",
                )
                if ax.lines:
                    ax.legend()
                fig.tight_layout()
                for suffix in ("png", "pdf"):
                    fig.savefig(out / f"{field}_vs_N.{suffix}")
                plt.close(fig)
        save(self.root / "reuse_manifest.json", reuse)
        completed = len(rows) // 2
        progress = {
            "protocol": PROTOCOL,
            "evaluated_cells": completed,
            "eligible_cells": 273,
            "remaining_cells": 273 - completed,
            "cr_complete_cells": sum(r["method"] == "CR" and r["strict_valid"] for r in rows),
            "global_valid_cells": sum(r["method"] == "Global" and r["strict_valid"] for r in rows),
            "infrastructure_pending_method_cells": sum(
                r["status"] in ("token_preflight_unavailable", "transport_retry_exhausted")
                for r in rows
            ),
            "updated_at": time.time(),
            "elapsed_s": time.time() - self.started,
            **self.store.accounting(),
        }
        if time.time() - self.started >= 43200 and not (self.root / "checkpoint_12h.json").exists():
            save(self.root / "checkpoint_12h.json", progress)
        save(self.root / "progress.json", progress)
        return progress

    def run(self):
        self.service.preflight()
        self.prepare()
        failures = []
        for n in SIZES:
            jobs = iter(a["assay"] for a in self.source["assays"] if a["n_valid"] >= n)
            if self.args.smoke_only:
                jobs = iter(list(jobs)[:6])
            with ThreadPoolExecutor(max_workers=6) as pool:
                pending = {}
                for assay in list(__import__("itertools").islice(jobs, 6)):
                    pending[pool.submit(self.cell, assay, n)] = assay
                while pending:
                    finished, _ = wait(pending, timeout=60, return_when=FIRST_COMPLETED)
                    if not finished:
                        save(
                            self.root / "live_budget.json",
                            {"updated_at": time.time(), **self.store.accounting()},
                        )
                    for future in finished:
                        assay = pending.pop(future)
                        try:
                            future.result()
                        except runtime.ExecutionPaused as error:
                            self.store.stop.set()
                            failures.append(
                                {"assay": assay, "N": n, "error": str(error), "fatal": True}
                            )
                        except TerminalRequest as error:
                            failures.append(
                                {"assay": assay, "N": n, "error": str(error), "fatal": False}
                            )
                        except Exception as error:
                            self.store.stop.set()
                            failures.append(
                                {"assay": assay, "N": n, "error": str(error), "fatal": True}
                            )
                        save(self.root / "failures.json", failures)
                        if not self.store.stop.is_set():
                            following = next(jobs, None)
                            if following is not None:
                                pending[pool.submit(self.cell, following, n)] = following
                    if finished:
                        self.report()
            if self.store.stop.is_set() or self.args.smoke_only:
                break
        return self.report()


def smoke_gate(root):
    cells = [
        json.loads(p.read_text()) for p in root.glob("assays/*/evaluations/n50/evaluation.json")
    ]
    failures_path = root / "failures.json"
    failures = json.loads(failures_path.read_text()) if failures_path.exists() else []
    return (
        len(cells) >= 6
        and sum(c["cr_status"] == "complete" for c in cells) >= 5
        and not any(f.get("fatal") for f in failures)
    )


def supervise(args):
    """Detached smoke-to-full supervisor. Never kills an in-flight paid request."""
    root, legacy = Path(args.output_root), Path(args.legacy_root)
    with (root / "supervisor.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state_path = root / "supervisor_status.json"
        state = (
            json.loads(state_path.read_text())
            if state_path.exists()
            else {"started_at": time.time(), "stage": "smoke", "launches": 0}
        )
        if state["stage"] in ("complete", "paused"):
            return
        while True:
            if (root / "PAUSE").exists() or (legacy / "PAUSE").exists():
                state.update(stage="paused", reason="manual_pause")
                save(state_path, state)
                return
            command = [
                sys.executable,
                "-u",
                "-m",
                "evagent.fullseq",
                *[v for v in sys.argv[1:] if v != "--supervise"],
            ]
            if state["stage"] == "smoke" and "--smoke-only" not in command:
                command.append("--smoke-only")
            state["launches"] += 1
            with (root / "worker.log").open("a") as log:
                child = subprocess.Popen(
                    command, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL
                )
                state.update(worker_pid=child.pid, supervisor_pid=os.getpid())
                while child.poll() is None:
                    state["updated_at"] = time.time()
                    save(state_path, state)
                    if (
                        time.time() - state["started_at"] >= 43200
                        and not (root / "checkpoint_12h.json").exists()
                    ):
                        progress = root / "progress.json"
                        save(
                            root / "checkpoint_12h.json",
                            {
                                "supervisor": state,
                                "progress": json.loads(progress.read_text())
                                if progress.exists()
                                else None,
                            },
                        )
                    time.sleep(30)
            outcome_path = root / "worker_outcome.json"
            outcome = (
                json.loads(outcome_path.read_text())
                if outcome_path.exists()
                else {"status": "unexpected_exit"}
            )
            if outcome.get("status") == "preflight_unavailable" and state["launches"] < 4:
                time.sleep(60)
                continue
            if child.returncode != 0 or outcome.get("status") != "finished":
                state.update(stage="paused", reason=outcome, exit_code=child.returncode)
            elif state["stage"] == "smoke":
                if smoke_gate(root):
                    state.update(stage="full", smoke_passed_at=time.time())
                    save(state_path, state)
                    continue
                state.update(stage="paused", reason="smoke_gate_failed")
            else:
                progress = json.loads((root / "progress.json").read_text())
                state.update(
                    stage="complete"
                    if progress["evaluated_cells"] == 273
                    and progress["cr_complete_cells"] == 273
                    and progress.get("infrastructure_pending_method_cells", 0) == 0
                    else "paused",
                    reason="queue_finished; inspect validity and unresolved cells",
                )
            state["updated_at"] = time.time()
            save(state_path, state)
            return


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_shared_arguments(parser)
    parser.set_defaults(
        output_root="experiments/" + PROTOCOL, seed=1, grouping_seed=1, trace_mode="off"
    )
    parser.add_argument("--legacy-root", default="runs_expansion")
    parser.add_argument("--token-budget", type=int, default=300_000_000)
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--supervise", action="store_true")
    args = parser.parse_args()
    root = Path(args.output_root)
    legacy = Path(args.legacy_root)
    root.mkdir(parents=True, exist_ok=True)
    configure_tracing("off", PROTOCOL)
    os.environ["EVAGENT_DISCONNECT_RETRIES"] = "0"
    config = build_config(args)
    config = replace(config, model=replace(config.model, retries=1))
    if not os.environ.get(config.model.api_key_env):
        raise SystemExit("GEMINI_API_KEY must be injected into the launch environment")
    if args.supervise:
        supervise(args)
        return
    save(root / "worker_outcome.json", {"status": "starting", "started_at": time.time()})
    with (legacy / "runner.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        source = json.loads((legacy / "frozen_manifest.json").read_text())
        reference(config)
        validate_frozen_data(config, source)
        backup = root / "ledger_before_start.sqlite"
        if not backup.exists():
            with (
                sqlite3.connect(legacy / "requests.sqlite") as src,
                sqlite3.connect(backup) as dest,
            ):
                src.backup(dest)
        store = Ledger(legacy / "requests.sqlite", args.token_budget, PROTOCOL)
        store.protocol_pause = root / "PAUSE"
        runtime.STORE = store
        configure_api_concurrency(12)
        try:
            result = Experiment(args, config, source, store).run()
        except TerminalRequest as error:
            save(
                root / "worker_outcome.json",
                {
                    "status": "preflight_unavailable"
                    if str(error) == "token_preflight_unavailable"
                    else "paused",
                    "reason": str(error),
                },
            )
            raise
        save(
            root / "worker_outcome.json",
            {
                "status": "paused"
                if store.stop.is_set() or result.get("infrastructure_pending_method_cells", 0)
                else "finished",
                "ended_at": time.time(),
            },
        )
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
