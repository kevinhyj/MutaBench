"""Isolated full-sequence CR repair and N1000/2000 extension; no v1 writes."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import sqlite3
import subprocess
import sys
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from . import fullseq as old
from . import runtime
from .graph import ASSAY_PROMPT, ASSAY_SYSTEM, VARIANT_SYSTEM, model_metadata
from .main import (
    _sample_hash_from_subset,
    add_shared_arguments,
    build_config,
    configure_tracing,
    nested_subsample_split,
)
from .model import GeminiAdapter, configure_api_concurrency, extract_json
from .schemas import AssayMemory

PROTOCOL = "cr_fullseq_b25_largeN_repair_v2"
SIZES = (50, 100, 250, 500, 1000, 2000)
LARGE = (1000, 2000)
TERMINAL = {"complete", "invalid_output", "context_limit_exceeded", "dependency_failed"}
save, digest, clean = old.save, runtime.digest, old.clean


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def freeze(path, value):
    if Path(path).exists() and read(path) != value:
        raise runtime.ExecutionPaused(f"Frozen artifact mismatch: {path}")
    save(path, value)


def generation(config):
    # Transport policy is recorded separately; it is not a generation parameter.
    return {
        k: config[k]
        for k in (
            "model_id",
            "endpoint",
            "temperature",
            "thinking_level",
            "include_thoughts",
            "max_tokens",
        )
    }


def signature(config, system, user):
    return digest([generation(config), system, user])


def unwrap(answer, variant=False):
    """Only unwrap unambiguous envelopes; never fill scores, IDs or evidence."""
    value = extract_json(answer)
    for _ in range(4):
        if isinstance(value, str):
            value = extract_json(value)
        elif (
            isinstance(value, dict)
            and len(value) == 1
            and next(iter(value)) in {"result", "response", "output", "data"}
        ):
            value = next(iter(value.values()))
        else:
            break
    if variant and isinstance(value, list):
        value = {"variants": value}
    return value


def validate_memory(value, assay):
    if not isinstance(value, dict) or value.get("assay_id", assay) != assay:
        raise ValueError("Wrong assay memory identity")
    parsed = AssayMemory.model_validate({"assay_id": assay, **value})
    if not parsed.assay_summary.strip() or not parsed.property_direction.strip():
        raise ValueError("Empty assay memory")
    return parsed.model_dump()


def extend_samples(config, assay, sizes, root, source_root):
    """Explicitly materialize every eligible requested size, including intermediate N."""
    rows = nested_subsample_split(config, assay, max(sizes), 1, root)
    family = digest(rows)
    for n in sizes:
        path = root / assay / f"n{n}_b1.json"
        selected = rows[:n]
        if len(selected) != n:
            raise ValueError("No repeated sampling to meet N")
        for previous in (path, source_root / "samples" / assay / f"n{n}_b1.json"):
            if previous.exists():
                saved = [(v["id"], v["seq"], float(v["score"])) for v in read(previous)["variants"]]
                if saved != selected:
                    raise runtime.ExecutionPaused("Nested sample prefix/order mismatch")
        if not path.exists():
            save(
                path,
                {
                    "schema_version": 2,
                    "assay": assay,
                    "size": n,
                    "seed": 1,
                    "batch": 1,
                    "parent_n": max(sizes),
                    "family_sha256": family,
                    "variants": [
                        {"id": m, "seq": seq, "score": score} for m, seq, score in selected
                    ],
                },
            )
    return {str(n): _sample_hash_from_subset(rows[:n]) for n in sizes}


class Service(old.Service):
    def __init__(self, config, root, store):
        super().__init__(config, root, store)
        self.adapter = GeminiAdapter(config.model, config.rg_llm_root, namespace=PROTOCOL)

    def call(self, path, system, user, name, metadata):
        request = {
            "protocol": PROTOCOL,
            "model": self.config.model.public_dict(),
            "system": system,
            "user": user,
            "name": name,
            "metadata": metadata,
        }
        fingerprint = digest(request)
        if path.exists():
            result = read(path)
            if result["request_hash"] != fingerprint:
                raise runtime.ExecutionPaused("Saved request hash mismatch")
            self.check_version(result["call"])
            return result["call"]
        freeze(
            path.with_name(path.stem + "_request.json"), {"request_hash": fingerprint, **request}
        )
        self.store.check_admission()
        self.count(system, user)
        response = self.adapter._call(system, user, name, metadata).model_dump()
        save(path, {"request_hash": fingerprint, "call": response})
        self.check_version(response)
        return response

    def check_version(self, response):
        actual = response.get("model_version")
        if actual and actual != self.config.model.model_id:
            self.store.stop.set()
            raise runtime.ExecutionPaused(
                "Provider version changed; raw result retained, manual review required"
            )

    def predict(self, folder, system, user, name, metadata, validator, repair_once=False):
        # The fixed response_0 request is the single durable semantic regeneration slot.
        if repair_once:
            freeze(
                folder / "regeneration_slot.json",
                {
                    "allowed_generations": 1,
                    "signature": signature(self.config.model.public_dict(), system, user),
                },
            )
        calls = []
        error = ""
        for attempt in range(1 if repair_once else 2):
            call = self.call(
                folder / f"response_{attempt}.json",
                system,
                user,
                name + f".response{attempt}",
                metadata,
            )
            calls.append(call)
            try:
                return {
                    "status": "complete",
                    "data": validator(unwrap(call["answer"], variant=name.startswith("CR"))),
                    "calls": calls,
                }
            except (ValueError, TypeError, KeyError) as exc:
                error = str(exc)
            # Empty output cannot be recovered through a format-only paid request.
            if not call["answer"].strip():
                break
            system = "Repair JSON formatting only; preserve all supplied IDs and values. Do not invent missing biological predictions."
            user = "Return valid JSON without changing semantic content.\n" + calls[0]["answer"]
        return {"status": "invalid_output", "error": error, "calls": calls}


class Experiment(old.Experiment):
    def __init__(self, args, config, source, store):
        super().__init__(args, config, source, store)
        self.origin = Path(args.source_root)
        self.service = Service(config, self.root, store)
        self.phase = "preparation"

    def prepare(self):
        from rg_llm import prompt

        source_protocol = read(self.origin / "protocol.json")
        if generation(source_protocol["model"]) != generation(self.config.model.public_dict()):
            raise runtime.ExecutionPaused("Source generation configuration changed")
        maximums, samples, queue = {}, {}, []
        for a in self.source["assays"]:
            assay = a["assay"]
            eligible = [n for n in SIZES if n <= a["n_valid"]]
            maximums[assay] = max(eligible)
            samples[assay] = extend_samples(
                self.config, assay, eligible, self.root / "samples", self.origin
            )
            for n in SIZES:
                queue.append(
                    {
                        "assay": assay,
                        "N": n,
                        "eligible": n in eligible,
                        "status": "queued" if n in eligible else "insufficient_variants",
                    }
                )
        source_files = {
            str(p.relative_to(self.origin)): sha(p)
            for pattern in (
                "assays/*/memory.json",
                "assays/*/chunks/*/result.json",
                "assays/*/evaluations/*/global.json",
            )
            for p in self.origin.glob(pattern)
        }
        code = {str(p): sha(p) for p in sorted(Path(__file__).parent.glob("*.py"))}
        manifest = {
            "protocol": PROTOCOL,
            "source_root": str(self.origin.resolve()),
            "source_protocol_sha256": sha(self.origin / "protocol.json"),
            "source_results": source_files,
            "sizes": list(SIZES),
            "large_sizes": list(LARGE),
            "B": 25,
            "seed": 1,
            "input_representation": "full_sequence",
            "model": self.config.model.public_dict(),
            "code_sha256": code,
            "prompt_sha256": digest(
                [
                    ASSAY_SYSTEM,
                    str(ASSAY_PROMPT),
                    VARIANT_SYSTEM,
                    old.CR_PROMPT,
                    prompt.SYSTEM_PROMPT,
                ]
            ),
            "maximums": maximums,
            "samples": samples,
            "eligible_cells": sum(q["eligible"] for q in queue),
            "large_cells": sum(q["eligible"] and q["N"] in LARGE for q in queue),
            "budget_limit": self.args.token_budget,
            "retry_delays_s": os.environ.get("EVAGENT_RETRY_DELAYS_S", "0,15,60,180,300"),
            "repair_generations": 1,
            "effective_timeout_s": max(3600, int(
                os.environ.get("EVAGENT_TIMEOUT_S", self.config.model.timeout_s)
            )),
            "expected_provider_version": self.config.model.model_id,
        }
        freeze(self.root / "protocol.json", manifest)
        freeze(self.root / "queue_manifest.json", queue)
        self.manifest = manifest

    def check_origin(self, path):
        key = str(path.relative_to(self.origin))
        if self.manifest["source_results"].get(key) != sha(path):
            raise runtime.ExecutionPaused("Historical result changed since freeze")

    def memory_prompt(self, assay):
        meta = self.metadata[assay]
        wt = str(meta["RAW_CONSTRUCT_SEQ"]).upper()
        return ASSAY_PROMPT.format(
            metadata=old.canonical(model_metadata(meta)), length=len(wt), wt_sequence=wt
        )

    def import_result(self, path, original, request_path, system, user, validator, variant=False):
        sig = signature(self.config.model.public_dict(), system, user)
        if path.exists():
            result = read(path)
            if result["input_signature"] != sig:
                raise runtime.ExecutionPaused("Cached evidence/input mismatch")
            return result
        if not original.exists():
            return None
        self.check_origin(original)
        request = read(request_path)
        if signature(request["model"], request["system"], request["user"]) != sig:
            raise runtime.ExecutionPaused("Historical prompt/config/dependency mismatch")
        result = read(original)
        provenance = {
            "source_path": str(original),
            "source_sha256": sha(original),
            "source_request_sha256": sha(request_path),
        }
        if result["status"] == "complete":
            validator({"variants": result["data"]} if variant else result["data"])
            result = {**result, "origin": "valid_reuse"}
        else:
            # Reparse only the original response, never infer predictions from a repair hallucination.
            try:
                data = validator(unwrap(result.get("calls", [{}])[0].get("answer", ""), variant))
                result = {
                    **result,
                    "status": "complete",
                    "data": data,
                    "origin": "offline_parse_repair",
                }
            except (ValueError, TypeError, KeyError, IndexError):
                result = {**result, "status": "needs_regeneration", "origin": "repair_pending"}
        result.update(input_signature=sig, provenance=provenance)
        save(path, result)
        return result

    def memory(self, assay, offline=False):
        folder = self.root / "assays" / assay
        source = self.origin / "assays" / assay
        user = self.memory_prompt(assay)

        def validator(value):
            return validate_memory(value, assay)

        result = self.import_result(
            folder / "memory.json",
            source / "memory.json",
            source / "analyst/response_0_request.json",
            ASSAY_SYSTEM,
            user,
            validator,
        )
        if result and result["status"] in TERMINAL:
            return result
        if offline:
            return result
        repair = result is not None and bool(result.get("provenance"))
        try:
            predicted = self.service.predict(
                folder / ("repair_memory" if repair else "analyst"),
                ASSAY_SYSTEM,
                user,
                "assay_analyst",
                {"assay": assay},
                validator,
                repair_once=repair,
            )
        except old.TerminalRequest as exc:
            predicted = {"status": str(exc), "calls": []}
        predicted.update(
            input_signature=signature(self.config.model.public_dict(), ASSAY_SYSTEM, user),
            origin="regenerated_once" if repair else "new",
            provenance=(result or {}).get("provenance"),
            historical_calls=(result or {}).get(
                "historical_calls", (result or {}).get("calls", [])
            ),
        )
        save(folder / "memory.json", predicted)
        return predicted

    def chunk(self, assay, memory, variants, index, offline=False):
        folder = self.root / "assays" / assay / "chunks" / f"c{index:03d}"
        source = self.origin / "assays" / assay / "chunks" / f"c{index:03d}"
        user = old.full_prompt(
            self.metadata[assay],
            memory["data"],
            str(self.metadata[assay]["RAW_CONSTRUCT_SEQ"]).upper(),
            variants,
        )

        def validator(value):
            return old.validate_evidence(value, variants)

        result = self.import_result(
            folder / "result.json",
            source / "result.json",
            source / "response_0_request.json",
            VARIANT_SYSTEM,
            user,
            validator,
            variant=True,
        )
        if result and result["status"] in TERMINAL:
            return result
        if offline:
            return result
        repair = result is not None and bool(result.get("provenance"))
        try:
            predicted = self.service.predict(
                folder / ("repair_once" if repair else "fresh"),
                VARIANT_SYSTEM,
                user,
                "CR.repair" if repair else "CR",
                {"assay": assay, "chunk": index},
                validator,
                repair_once=repair,
            )
        except old.TerminalRequest as exc:
            predicted = {"status": str(exc), "calls": []}
        predicted.update(
            input_signature=signature(self.config.model.public_dict(), VARIANT_SYSTEM, user),
            origin="regenerated_once" if repair else "new",
            chunk=index,
            provenance=(result or {}).get("provenance"),
            historical_calls=(result or {}).get(
                "historical_calls", (result or {}).get("calls", [])
            ),
        )
        save(folder / "result.json", predicted)
        return predicted

    def offline(self):
        counts = {}
        for a in self.source["assays"]:
            assay = a["assay"]
            memory = self.memory(assay, offline=True)
            counts[memory["origin"]] = counts.get(memory["origin"], 0) + 1
            if memory["status"] != "complete":
                continue
            variants = self.variants(assay, min(500, self.manifest["maximums"][assay]))
            for i in range(0, len(variants), 25):
                result = self.chunk(assay, memory, variants[i : i + 25], i // 25, offline=True)
                if result:
                    counts[result["origin"]] = counts.get(result["origin"], 0) + 1
        save(
            self.root / "offline_audit.json", {"counts": counts, "at": time.time(), "paid_calls": 0}
        )

    def repair_assay(self, assay):
        memory = self.memory(assay)
        if memory["status"] != "complete":
            return
        variants = self.variants(assay, min(500, self.manifest["maximums"][assay]))
        # Only repair previously attempted chunks here; missing chunks are backfilled later.
        for i in range(0, len(variants), 25):
            path = self.root / "assays" / assay / "chunks" / f"c{i // 25:03d}" / "result.json"
            if path.exists() and read(path)["status"] not in TERMINAL:
                self.chunk(assay, memory, variants[i : i + 25], i // 25)

    def global_ranking(self, assay, n, variants):
        from rg_llm import prompt

        path = self.root / "assays" / assay / "evaluations" / f"n{n}" / "global.json"
        if path.exists() and read(path)["status"] in TERMINAL:
            return read(path)
        source = self.origin / "assays" / assay / "evaluations" / f"n{n}" / "global.json"
        subset = [(v["mutation"], v["sequence"], v["experimental_score"]) for v in variants]
        user, _ = prompt.build_user_prompt(
            model_metadata(self.metadata[assay]),
            str(self.metadata[assay]["RAW_CONSTRUCT_SEQ"]).upper(),
            subset,
        )
        if source.exists():
            self.check_origin(source)
            result = read(source)
            if result["sample_hash"] != _sample_hash_from_subset(subset):
                raise runtime.ExecutionPaused("Global source sample mismatch")
            request_path = source.with_name("global_response_request.json")
            if request_path.exists():
                request = read(request_path)
                if signature(request["model"], request["system"], request["user"]) != signature(
                    self.config.model.public_dict(), prompt.SYSTEM_PROMPT, user
                ):
                    raise runtime.ExecutionPaused("Global source prompt mismatch")
            elif result.get("reuse"):
                original = Path(result["reuse"]["source_path"])
                if sha(original) != result["reuse"]["source_hash"] or read(original)[
                    "prompt_sha256"
                ] != digest([prompt.SYSTEM_PROMPT, user]):
                    raise runtime.ExecutionPaused("Legacy Global source mismatch")
            else:
                raise runtime.ExecutionPaused("Global source lacks request provenance")
            if result["status"] in TERMINAL:
                result = {
                    **result,
                    "origin": "valid_reuse"
                    if result["status"] == "complete"
                    else "historical_failure_preserved",
                    "provenance": {"source_path": str(source), "source_sha256": sha(source)},
                }
                save(path, result)
                return result
        # Re-evaluate pending requests with durable v2 request keys; never delete old records.
        if path.exists():
            pending = read(path)
            if pending["status"] not in TERMINAL:
                return self.new_global(assay, n, variants, path)
        return self.new_global(assay, n, variants, path)

    def new_global(self, assay, n, variants, path):
        return self._global_request(assay, n, variants, path)

    def _global_request(self, assay, n, variants, path):
        # Failed transport entries have no completed response. Reusing the exact request path
        # enforces its durable attempt cap. Audit uses the same v1 strict parser.
        from rg_llm import prompt

        subset = [(v["mutation"], v["sequence"], v["experimental_score"]) for v in variants]
        user, ids = prompt.build_user_prompt(
            model_metadata(self.metadata[assay]),
            str(self.metadata[assay]["RAW_CONSTRUCT_SEQ"]).upper(),
            subset,
        )
        try:
            call = self.service.call(
                path.with_name("global_response.json"),
                prompt.SYSTEM_PROMPT,
                user,
                "Global",
                {"assay": assay, "N": n},
            )
        except old.TerminalRequest as exc:
            result = {
                "status": str(exc),
                "calls": [],
                "sample_hash": _sample_hash_from_subset(subset),
            }
        else:
            audit = old.audit_baseline({"ids": ids, "response_text": call["answer"]})
            mapping = {key: v["variant_id"] for key, v in zip(ids, variants, strict=True)}
            ranking = [mapping[x] for x in (audit.get("ranking") or [])]
            returned = []
            try:
                raw = extract_json(call["answer"])
                values = (
                    raw.get("ranking", [])
                    if isinstance(raw, dict)
                    else raw
                    if isinstance(raw, list)
                    else []
                )
                for value in values:
                    canonical = old._canonical_id(value, set(ids))
                    if canonical is not None and mapping[canonical] not in returned:
                        returned.append(mapping[canonical])
            except (ValueError, TypeError):
                pass
            result = {
                "status": "complete" if audit["valid"] else "invalid_output",
                "audit": audit,
                "ranking": ranking,
                "returned_ranking": ranking if audit["valid"] else returned,
                "calls": [call],
                "sample_hash": _sample_hash_from_subset(subset),
            }
        result["origin"] = "new_or_infrastructure_recovery"
        save(path, result)
        return result

    def cell(self, assay, n):
        started = time.time()
        folder = self.root / "assays" / assay / "evaluations" / f"n{n}"
        if (folder / "evaluation.json").exists() and read(folder / "evaluation.json").get(
            "terminal"
        ):
            return
        variants = self.variants(assay, n)
        global_result = self.global_ranking(assay, n, variants)
        memory = self.memory(assay)
        chunks = []
        if memory["status"] == "complete":
            existing = [
                self.chunk(assay, memory, variants[i : i + 25], i // 25, offline=True)
                for i in range(0, n, 25)
            ]
            blocked = any(c and c["status"] != "complete" for c in existing)
            if blocked:
                chunks = [c or {"status": "dependency_failed"} for c in existing]
            else:
                halted = threading.Event()

                def run_chunk(i):
                    cached = existing[i // 25]
                    if cached:
                        return cached
                    if halted.is_set():
                        return {"status": "dependency_failed"}
                    value = self.chunk(assay, memory, variants[i : i + 25], i // 25)
                    if value["status"] != "complete":
                        halted.set()
                    return value

                with ThreadPoolExecutor(max_workers=4) as pool:
                    chunks = list(pool.map(run_chunk, range(0, n, 25)))
        evidence = [v for c in chunks if c["status"] == "complete" for v in c["data"]]
        scores = {v["variant_id"]: v["predicted_score"] for v in evidence}
        complete = len(scores) == n and set(scores) == {v["variant_id"] for v in variants}
        failures = [c["status"] for c in chunks if c["status"] != "complete"]
        if memory["status"] != "complete":
            failures.append(memory["status"])
        pending = any(status not in TERMINAL for status in failures)
        cr = {
            "status": "complete"
            if complete
            else "dependency_pending"
            if pending
            else "dependency_failed",
            "coverage": len(scores) / n,
            "scores": scores,
            "chunk_statuses": [c["status"] for c in chunks],
            "calls": memory.get("calls", [])
            + [call for c in chunks for call in c.get("calls", [])],
            "historical_repair_calls": memory.get("historical_calls", [])
            + [call for c in chunks for call in c.get("historical_calls", [])],
            "sample_hash": self.manifest["samples"][assay][str(n)],
        }
        save(folder / "cr.json", cr)
        table = pd.DataFrame(
            [{k: v[k] for k in ("variant_id", "mutation", "experimental_score")} for v in variants]
        )
        table["predicted_score"] = table.variant_id.map(scores)
        table["predicted_rank_average"] = table.predicted_score.rank(
            method="average", ascending=False
        )
        table.to_csv(folder / "predictions.csv", index=False)
        save(
            folder / "evaluation.json",
            {
                "protocol": PROTOCOL,
                "assay": assay,
                "N": n,
                "B": 25,
                "seed": 1,
                "sample_hash": cr["sample_hash"],
                "cr_status": cr["status"],
                "global_status": global_result["status"],
                "terminal": not pending and global_result["status"] in TERMINAL,
                "truth": {v["variant_id"]: v["experimental_score"] for v in variants},
                "wall_s": time.time() - started,
                "completed_at": time.time(),
            },
        )
        print(
            f"[{PROTOCOL}] {assay} N={n} CR={cr['status']} Global={global_result['status']}",
            flush=True,
        )

    def phase_run(self, name, assays, function):
        self.phase = name
        save(self.root / "phase.json", {"phase": name, "at": time.time()})
        jobs = iter(assays)
        errors = []
        with ThreadPoolExecutor(max_workers=6) as pool:
            pending = {}
            for _ in range(6):
                assay = next(jobs, None)
                if assay is not None:
                    pending[pool.submit(function, assay)] = assay
            while pending:
                finished, _ = wait(pending, timeout=60, return_when=FIRST_COMPLETED)
                for task in finished:
                    assay = pending.pop(task)
                    try:
                        task.result()
                    except Exception as exc:
                        self.store.stop.set()
                        errors.append(
                            {
                                "phase": name,
                                "assay": assay,
                                "type": type(exc).__name__,
                                "error": str(exc)[:600],
                            }
                        )
                        save(self.root / "failures.json", errors)
                    if not self.store.stop.is_set():
                        following = next(jobs, None)
                        if following is not None:
                            pending[pool.submit(function, following)] = following
                self.report()
        if self.store.stop.is_set():
            raise runtime.ExecutionPaused("Phase paused; inspect failures and budget")

    def run(self):
        self.prepare()
        self.offline()
        if self.args.prepare_only:
            return self.report()
        self.service.preflight()
        self.phase_run(
            "bounded_repair", [a["assay"] for a in self.source["assays"]], self.repair_assay
        )
        for n in SIZES:
            self.phase_run(
                f"N{n}",
                [a["assay"] for a in self.source["assays"] if a["n_valid"] >= n],
                lambda assay: self.cell(assay, n),
            )
        self.phase = "queue_finished"
        return self.report()

    def report(self):
        out = self.root / "analysis"
        out.mkdir(exist_ok=True)
        rows, probes, overlaps, evaluations = [], [], [], []
        for path in self.root.glob("assays/*/evaluations/n*/evaluation.json"):
            e = read(path)
            evaluations.append(e)
            truth = e["truth"]
            cr, global_result = read(path.with_name("cr.json")), read(path.with_name("global.json"))
            common = {k: e[k] for k in ("assay", "N", "seed", "B", "sample_hash")}
            common.update(
                protocol=PROTOCOL,
                rna_type=self.classes[e["assay"]]["rna_type"],
                protein_stability_source=self.classes[e["assay"]]["protein_stability_source"],
                wt_length=len(str(self.metadata[e["assay"]]["RAW_CONSTRUCT_SEQ"])),
            )
            for method, result in (("CR", cr), ("Global", global_result)):
                scores = (
                    cr["scores"]
                    if method == "CR"
                    else {v: -i for i, v in enumerate(result.get("ranking", []))}
                )
                metrics = (
                    old.correlation(scores, truth)
                    if result["status"] == "complete"
                    else {"spearman": math.nan, "kendall": math.nan}
                )
                calls = result.get("calls", [])
                versions = sorted(
                    {str(c.get("model_version")) for c in calls if c.get("model_version")}
                )
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
                            c.get("usage", {}).get("totalTokenCount", 0) for c in calls
                        ),
                        "historical_repair_tokens": sum(
                            c.get("usage", {}).get("totalTokenCount", 0)
                            for c in result.get("historical_repair_calls", [])
                        ),
                        "attributed_calls": len(calls),
                        "summed_request_s": sum(c.get("elapsed_s", 0) for c in calls),
                        "summed_queue_s": sum(c.get("queue_s", 0) for c in calls),
                        "evaluation_wall_s": e["wall_s"],
                        "model_versions": ";".join(versions) or "unavailable",
                    }
                )
                returned = (
                    scores
                    if method == "CR"
                    else {v: -i for i, v in enumerate(result.get("returned_ranking", []))}
                )
                probe_ids = [v for v in list(truth)[:50] if v in returned]
                pm = (
                    old.correlation(returned, truth, probe_ids)
                    if len(probe_ids) > 1
                    else {"spearman": math.nan, "kendall": math.nan}
                )
                probes.append({**common, "method": method, "coverage": len(probe_ids) / 50, **pm})
            ids = [v for v in global_result.get("returned_ranking", []) if v in cr["scores"]]
            gscore = {v: -i for i, v in enumerate(global_result.get("returned_ranking", []))}
            for method, scores in (("CR", cr["scores"]), ("Global", gscore)):
                metrics = (
                    old.correlation(scores, truth, ids)
                    if len(ids) > 1
                    else {"spearman": math.nan, "kendall": math.nan}
                )
                overlaps.append(
                    {
                        **common,
                        "method": method,
                        "overlap_count": len(ids),
                        "overlap_coverage": len(ids) / e["N"],
                        **metrics,
                    }
                )
        frame = pd.DataFrame(rows)
        repairs = []
        for pattern in ("assays/*/memory.json", "assays/*/chunks/*/result.json"):
            for path in self.root.glob(pattern):
                result = read(path)
                repairs.append(
                    {
                        "path": str(path),
                        "status": result["status"],
                        "origin": result.get("origin"),
                        "input_signature": result.get("input_signature"),
                        "source_path": (result.get("provenance") or {}).get("source_path"),
                        "source_sha256": (result.get("provenance") or {}).get("source_sha256"),
                    }
                )
        pd.DataFrame(repairs).to_csv(out / "repair_and_reuse.csv", index=False)
        if len(frame):
            frame.to_csv(out / "experiment_long.csv", index=False)
            pd.DataFrame(probes).to_csv(out / "fixed_probe.csv", index=False)
            pd.DataFrame(overlaps).to_csv(out / "returned_overlap.csv", index=False)
            frame.groupby(["N", "method", "status"]).size().rename("cells").to_csv(
                out / "status_counts.csv"
            )
            frame.groupby(["N", "method"]).agg(
                evaluated=("assay", "count"),
                valid=("strict_valid", "sum"),
                computable=("spearman", "count"),
                mean_spearman=("spearman", "mean"),
                median_spearman=("spearman", "median"),
                mean_coverage=("coverage", "mean"),
            ).to_csv(out / "macro_summary.csv")
            self.comparisons(frame, out)
        counts = {str(n): sum(e["N"] == n for e in evaluations) for n in SIZES}
        expected = {str(n): sum(a["n_valid"] >= n for a in self.source["assays"]) for n in SIZES}
        known = {(e["assay"], e["N"]): e for e in evaluations}
        pending = []
        for item in read(self.root / "queue_manifest.json"):
            if item["eligible"]:
                e = known.get((item["assay"], item["N"]))
                if e is None or not e["terminal"]:
                    pending.append(
                        {
                            "assay": item["assay"],
                            "N": item["N"],
                            "status": "not_evaluated" if e is None else "infrastructure_pending",
                        }
                    )
        save(self.root / "pending_queue.json", pending)
        accounting = self.store.accounting()
        progress = {
            "protocol": PROTOCOL,
            "phase": self.phase,
            "updated_at": time.time(),
            "elapsed_s": time.time() - self.started,
            "eligible_cells": self.manifest["eligible_cells"],
            "large_eligible_cells": self.manifest["large_cells"],
            "evaluated_cells": len(evaluations),
            "terminal_cells": sum(e["terminal"] for e in evaluations),
            "large_evaluated_cells": sum(e["N"] in LARGE for e in evaluations),
            "large_terminal_cells": sum(e["N"] in LARGE and e["terminal"] for e in evaluations),
            "cr_complete_cells": sum(e["cr_status"] == "complete" for e in evaluations),
            "global_valid_cells": sum(e["global_status"] == "complete" for e in evaluations),
            "pending_cells": len(pending),
            "by_N_evaluated": counts,
            "by_N_eligible": expected,
            **accounting,
        }
        save(self.root / "progress.json", progress)
        return progress

    def comparisons(self, frame, out):
        paired = []
        scopes = {
            "all": frame,
            "non_protein_stability": frame[~frame.protein_stability_source],
            "common_45_eligible": frame[
                frame.assay.isin(
                    [a["assay"] for a in self.source["assays"] if a["n_valid"] >= 2000]
                )
            ],
        }
        scopes.update({"rna_type:" + str(k): b for k, b in frame.groupby("rna_type")})
        scopes.update({"provider_version:" + str(k): b for k, b in frame.groupby("model_versions")})
        for scope, data in scopes.items():
            for n, block in data.groupby("N"):
                pivot = block.pivot(index="assay", columns="method", values="spearman").reindex(
                    columns=["CR", "Global"]
                )
                delta = (pivot.CR - pivot.Global).dropna().to_numpy()
                if len(delta):
                    ci = bootstrap(delta[:, None])[0]
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
        table = frame.pivot(index="assay", columns=["method", "N"], values="spearman")
        for sizes, name in ((SIZES, "six_N"), (LARGE, "large_N")):
            both = table.reindex(
                columns=pd.MultiIndex.from_product([["CR", "Global"], sizes])
            ).dropna()
            chosen = frame[frame.assay.isin(both.index) & frame.N.isin(sizes)]
            chosen.to_csv(out / f"{name}_matched_subset.csv", index=False)
            if len(both):
                delta = both["CR"].to_numpy() - both["Global"].to_numpy()
                # One resampled assay-index matrix is shared by every N and N contrast.
                values = np.column_stack([delta, delta[:, -1] - delta[:, 0]])
                ci = bootstrap(values)
                pd.DataFrame(
                    [
                        {
                            "contrast": str(n),
                            "n_assays": len(both),
                            "mean_delta": float(values[:, i].mean()),
                            "bootstrap95_low": ci[i][0],
                            "bootstrap95_high": ci[i][1],
                        }
                        for i, n in enumerate([*sizes, "last_N_minus_first_N_delta"])
                    ]
                ).to_csv(out / f"{name}_joint_bootstrap.csv", index=False)
        before = self.origin / "analysis/experiment_long.csv"
        if before.exists():
            original = pd.read_csv(before)
            original["version"] = "v1_original"
            after = frame[frame.N <= 500].copy()
            after["version"] = PROTOCOL
            pd.concat([original, after], ignore_index=True).to_csv(
                out / "repair_before_after.csv", index=False
            )
        self.plots(frame, out)

    def plots(self, frame, out):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        cohort = frame[
            frame.assay.isin([a["assay"] for a in self.source["assays"] if a["n_valid"] >= 2000])
        ]
        for metric in (
            "spearman",
            "coverage",
            "strict_valid",
            "attributed_tokens",
            "summed_request_s",
        ):
            fig, ax = plt.subplots(figsize=(6.4, 4.2))
            for method, color, label in (
                ("CR", "#728AB9", "Chunked ranking"),
                ("Global", "#F5B5BF", "Global ranking"),
            ):
                b = cohort[cohort.method == method].groupby("N")[metric]
                value = b.mean()
                ax.plot(value.index, value.values, "o-", color=color, label=label)
            ax.set(
                xlabel="Library size N",
                ylabel=metric,
                title="Common eligible cohort; accuracy is valid-only",
            )
            ax.legend()
            fig.tight_layout()
            for ext in ("png", "pdf"):
                fig.savefig(out / f"{metric}_vs_N.{ext}")
            plt.close(fig)


def bootstrap(values):
    values = np.asarray(values)
    if len(values) < 2:
        return np.full((values.shape[1], 2), np.nan)
    indices = np.random.default_rng(1).integers(0, len(values), size=(10000, len(values)))
    means = values[indices].mean(axis=1)
    return np.quantile(means, [0.025, 0.975], axis=0).T


def supervise(args):
    root = Path(args.output_root)
    with (root / "supervisor.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state_path = root / "supervisor_status.json"
        state = (
            read(state_path)
            if state_path.exists()
            else {"started_at": time.time(), "stage": "starting", "launches": 0}
        )
        if state["stage"] in ("paused", "complete"):
            return
        while True:
            if (root / "PAUSE").exists() or (Path(args.legacy_root) / "PAUSE").exists():
                state.update(stage="paused", reason="manual_pause")
                save(state_path, state)
                return
            command = [
                sys.executable,
                "-u",
                "-m",
                "evagent.large_n",
                *[x for x in sys.argv[1:] if x != "--supervise"],
            ]
            state["launches"] += 1
            with (root / "worker.log").open("a") as log:
                child = subprocess.Popen(
                    command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT
                )
                state.update(stage="running", supervisor_pid=os.getpid(), worker_pid=child.pid)
                while child.poll() is None:
                    state["updated_at"] = time.time()
                    save(state_path, state)
                    if (
                        time.time() - state["started_at"] >= 43200
                        and not (root / "checkpoint_12h.json").exists()
                    ):
                        save(
                            root / "checkpoint_12h.json",
                            {
                                "supervisor": state,
                                "progress": read(root / "progress.json")
                                if (root / "progress.json").exists()
                                else None,
                            },
                        )
                    time.sleep(30)
            outcome = (
                read(root / "worker_outcome.json")
                if (root / "worker_outcome.json").exists()
                else {"status": "unexpected_exit"}
            )
            if outcome["status"] == "preflight_unavailable" and state["launches"] < 4:
                time.sleep(60)
                continue
            state.update(
                stage="complete"
                if child.returncode == 0 and outcome["status"] == "complete"
                else "paused",
                reason=outcome,
                updated_at=time.time(),
            )
            save(state_path, state)
            return


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_shared_arguments(parser)
    parser.set_defaults(
        output_root="experiments/" + PROTOCOL, seed=1, grouping_seed=1, trace_mode="off"
    )
    parser.add_argument("--source-root", default="experiments/cr_fullseq_b25_v1")
    parser.add_argument("--legacy-root", default="runs_expansion")
    parser.add_argument("--token-budget", type=int, default=300_000_000)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--supervise", action="store_true")
    args = parser.parse_args()
    if args.seed != 1 or args.grouping_seed != 1 or args.token_budget > 300_000_000:
        raise SystemExit("Protocol seed or authorized budget mismatch")
    root, legacy = Path(args.output_root), Path(args.legacy_root)
    if root.resolve() == Path(args.source_root).resolve():
        raise SystemExit("v2 output must not overwrite v1")
    root.mkdir(parents=True, exist_ok=True)
    configure_tracing("off", PROTOCOL)
    os.environ["EVAGENT_DISCONNECT_RETRIES"] = "0"
    config = build_config(args)
    config = replace(config, model=replace(config.model, retries=1))
    if not args.prepare_only and not os.environ.get(config.model.api_key_env):
        raise SystemExit("GEMINI_API_KEY must be injected through environment")
    if args.supervise:
        supervise(args)
        return
    save(root / "worker_outcome.json", {"status": "starting", "at": time.time()})
    try:
        with (legacy / "runner.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with sqlite3.connect(f"file:{legacy / 'requests.sqlite'}?mode=ro", uri=True) as db:
                if db.execute("SELECT COUNT(*) FROM requests WHERE status='inflight'").fetchone()[
                    0
                ]:
                    raise runtime.ExecutionPaused(
                        "Unresolved in-flight ledger entries; inspect before launch"
                    )
                backup = root / "ledger_before_start.sqlite"
                if not backup.exists():
                    with sqlite3.connect(backup) as dest:
                        db.backup(dest)
            source = read(legacy / "frozen_manifest.json")
            old.reference(config)
            old.validate_frozen_data(config, source)
            store = old.Ledger(legacy / "requests.sqlite", args.token_budget, PROTOCOL)
            store.protocol_pause = root / "PAUSE"
            runtime.STORE = store
            configure_api_concurrency(12)
            result = Experiment(args, config, source, store).run()
            status = (
                "prepared"
                if args.prepare_only
                else "complete"
                if result["pending_cells"] == 0
                else "pending_infrastructure"
            )
            save(root / "worker_outcome.json", {"status": status, "at": time.time()})
            print(json.dumps(result), flush=True)
    except Exception as exc:
        secret = os.environ.get("GEMINI_API_KEY", "")
        message = str(exc).replace(secret, "[REDACTED]") if secret else str(exc)
        save(
            root / "worker_outcome.json",
            {
                "status": "preflight_unavailable"
                if isinstance(exc, old.TerminalRequest) and message == "token_preflight_unavailable"
                else "paused",
                "error_type": type(exc).__name__,
                "reason": message[:600],
                "at": time.time(),
            },
        )
        raise


if __name__ == "__main__":
    main()
