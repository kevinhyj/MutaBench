"""Isolated full-sequence ACR, G25/A4; no CR evidence or score fallback."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import random
import sqlite3
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from . import fullseq as old
from . import model, runtime
from .experiments import correlation, mean_local_rank
from .graph import _canonical_id, model_metadata
from .main import add_shared_arguments, build_config, configure_tracing
from .ranking import _pl_objective, _quantile_anchors

PROTOCOL = "acr_seq_g25_a4_v1"
SIZES = (50, 100, 250, 500)
PILOT = (
    "Andreasson_2020_ribozyme", "Domingo_2018_tRNA", "Tome_2014_NELFE_aptamer",
    "Ke_2017_mRNA", "BCHB_CHLTE_Tsuboyama_2023_2KRU", "A0A2Z5U3Z0_9INFA_Doud_2016",
)
SYSTEM = """You predict mutation effects from complete biological sequences and assay metadata.
Compare all supplied candidates for the measured assay phenotype. Rank best/highest
fitness first and worst/lowest fitness last, interpreting fitness in this assay's
biological context. Use sequence-grounded mechanistic comparisons; distinguish
observations from hypotheses. Do not claim experimental measurements or external
tools that were not supplied. Return a strict complete permutation of candidate IDs.
Return JSON only: {"ranking":["M001","M002"],"comparisons":[{"variant_ids":["M001","M002"],"reason":"brief mechanism and uncertainty"}]}.
Do not emit a separate numeric score or omit uncertain candidates."""


def read(path):
    return json.loads(Path(path).read_text())


def freeze(path, value):
    if Path(path).exists() and read(path) != value:
        raise runtime.ExecutionPaused(f"Frozen artifact mismatch: {path}")
    old.save(path, value)


def groups(assay, ids, anchors):
    if len(anchors) != 4 or len(set(anchors)) != 4 or not set(anchors) <= set(ids):
        raise ValueError("Four unique shared anchors required")
    remaining = [v for v in ids if v not in anchors]
    result = []
    for offset in range(0, len(remaining), 21):
        members = list(anchors) + remaining[offset:offset + 21]
        random.Random(runtime.digest([assay, 1, members])).shuffle(members)
        result.append({"variant_ids": members, "anchor_ids": anchors,
                       "group_hash": runtime.digest([assay, members])})
    if set().union(*(set(g["variant_ids"]) for g in result)) != set(ids):
        raise ValueError("Incomplete grouping")
    return result


def prompt(metadata, wt, variants):
    candidates = [{k: v[k] for k in ("variant_id", "mutation", "sequence")} for v in variants]
    return ("Assay metadata:\n" + old.canonical(model_metadata(metadata))
            + "\nComplete wild-type sequence:\n" + wt
            + "\nCandidates with complete mutant sequences (not truncated):\n"
            + old.canonical(candidates))


def validate_answer(answer, ids):
    data = model.extract_json(answer)
    if not isinstance(data, dict) or not isinstance(data.get("ranking"), list):
        raise ValueError("Missing ranking array")
    ranking = [_canonical_id(v, set(ids)) for v in data["ranking"]]
    if len(ranking) != len(ids) or set(ranking) != set(ids) or len(set(ranking)) != len(ids):
        raise ValueError("Ranking is not a strict permutation")
    # Mechanistic text is stored, never used to repair or complete the ranking.
    return {"ranking": ranking, "comparisons": data.get("comparisons", [])}


def classify(raw):
    envelope = raw.get("provider_raw") or {}
    feedback = envelope.get("promptFeedback") or {}
    candidates = envelope.get("candidates") or []
    reasons = {str(c.get("finishReason", "")).upper() for c in candidates}
    if feedback.get("blockReason") or reasons & {
        "SAFETY", "RECITATION", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII", "IMAGE_SAFETY"
    }:
        return "provider_blocked"
    if not candidates or not any(reasons):
        return "provider_unclassified"
    return "response_received"


def wire_request(config, system, user):
    # Exact representation used by rg_llm.client.chat_gemini, not systemInstruction.
    generation = {"temperature": config.temperature,
                  "thinkingConfig": {"thinkingLevel": config.thinking_level,
                                     "includeThoughts": config.include_thoughts}}
    if config.max_tokens is not None:
        generation["maxOutputTokens"] = config.max_tokens
    return {"contents": [{"parts": [{"text": system + "\n\n" + user}]}],
            "generationConfig": generation}


def fit(ids, rankings):
    index = {v: i for i, v in enumerate(ids)}
    connected = {ids[0]}
    for item in rankings:
        r = item["ranking"]
        if len(r) != len(set(r)) or not set(r) <= set(ids):
            raise ValueError("Invalid PL observation")
    for _ in rankings:
        for item in rankings:
            if connected.intersection(item["ranking"]):
                connected.update(item["ranking"])
    if connected != set(ids):
        raise ValueError("Disconnected ranking graph")
    orders = [(np.array([index[v] for v in r["ranking"]]), 1.0) for r in rankings]
    x = np.zeros(len(ids))
    for iterations in (500, 5000):
        result = minimize(lambda z: _pl_objective(z, orders, 1e-3), x,
                          method="L-BFGS-B", jac=True,
                          options={"maxiter": iterations, "ftol": 1e-12, "gtol": 1e-6})
        x = result.x
        if result.success and np.isfinite(x).all() and np.isfinite(result.fun):
            x = x - x.mean()
            if x.std() > 1e-12:
                x /= x.std()
            return dict(zip(ids, map(float, x))), {
                "success": True, "message": str(result.message),
                "objective": float(result.fun), "gradient_max": float(np.abs(result.jac).max()),
            }
    raise ValueError("PL did not converge with unchanged objective")


class Ledger(old.Ledger):
    def __init__(self, path, limit, root, experiment_limit):
        super().__init__(path, limit, PROTOCOL)
        self.protocol_pause = root / "PAUSE"
        self.experiment_limit = experiment_limit

    def own_tokens(self):
        with self.lock:
            return self.db.execute(
                "SELECT COALESCE(SUM(r.tokens),0) FROM requests r JOIN request_protocol p "
                "ON r.key=p.key WHERE p.protocol=?", (PROTOCOL,)).fetchone()[0]

    def check_admission(self):
        super().check_admission()
        if self.own_tokens() >= self.experiment_limit:
            raise runtime.ExecutionPaused("ACR experiment token soft cap reached")

    def invoke(self, key, call):
        # The parent persists each wire attempt and usage before this classification.
        # Unclassified envelopes are recoverable, never silently cached as rankings.
        def guarded_call():
            try:
                raw = call()
                secret = os.environ.get("GEMINI_API_KEY", "")
                return json.loads(json.dumps(raw).replace(secret, "[REDACTED]")) if secret else raw
            except Exception as exc:
                if any(s in str(exc).lower() for s in ("429", "resource_exhausted", "quota exhausted")):
                    self.stop.set()
                raise
        while True:
            raw = super().invoke(key, guarded_call)
            kind = classify(raw)
            if kind != "provider_unclassified":
                return {**raw, "response_class": kind}
            with self.lock:
                self.db.execute("UPDATE requests SET status='provider_unclassified' WHERE key=?", (key,))
                self.db.commit()
            self.check_admission()
            # Avoid hammering an ambiguous service; durable parent ceiling is five.
            for _ in range(15):
                self.check_admission()
                time.sleep(1)


class Service(old.Service):
    def check_version(self, raw):
        actual = (raw.get("provider_raw") or {}).get("modelVersion")
        with self.store.lock:
            version_path = self.root / "observed_model_version.json"
            if actual:
                if version_path.exists() and read(version_path)["model_version"] != actual:
                    self.store.stop.set()
                    raise runtime.ExecutionPaused("Actual provider model version changed; raw response saved")
                freeze(version_path, {"model_version": actual})

    def count(self, system, user):
        body = wire_request(self.config.model, system, user)
        digest = runtime.digest([self.config.model.public_dict(), body])
        path = self.root / "token_checks" / (digest + ".json")
        self.store.check_admission()
        if path.exists():
            result = read(path)
        else:
            result = self.metadata_request(":countTokens", {"generateContentRequest": {
                "model": "models/" + self.config.model.model_id, **body}})
            old.save(path, {"totalTokens": int(result["totalTokens"]), "input_hash": digest})
        if int(result["totalTokens"]) > self.limit:
            raise old.TerminalRequest("context_limit_exceeded")
        return int(result["totalTokens"])

    def request(self, folder, user, metadata):
        args = {"public_config": self.config.model.public_dict(),
                "rg_llm_root": str(self.config.rg_llm_root), "system": SYSTEM, "user": user,
                "run_name": "ACR-seq", "metadata": {"protocol": PROTOCOL, **metadata}}
        freeze(folder / "request.json", args)
        fingerprint = runtime.digest(args)
        result_path = folder / "response.json"
        if result_path.exists():
            raw = read(result_path)
            if raw["request_sha256"] != fingerprint:
                raise runtime.ExecutionPaused("Response cache mismatch")
            self.check_version(raw)
            return raw
        count = self.count(SYSTEM, user)
        raw = model._gemini_generate(**args)
        raw["preflight_input_tokens"] = count
        # Never store a secret accidentally echoed in a body/diagnostic.
        secret = os.environ.get(self.config.model.api_key_env, "")
        if secret:
            raw = json.loads(json.dumps(raw).replace(secret, "[REDACTED]"))
        old.save(result_path, raw)
        self.check_version(raw)
        return raw


class Experiment:
    def __init__(self, args, config, store):
        self.args, self.config, self.store = args, config, store
        self.root, self.source = Path(args.output_root), Path(args.source_root)
        self.frozen = read(Path(args.legacy_root) / "frozen_manifest.json")
        self.assays = self.frozen["assays"]
        self.classes = {a["assay"]: a for a in self.assays}
        self.metadata = {str(r["DMS_ID"]): old._clean_metadata(r)
                         for r in old.reference(config).to_dict("records")}
        self.service = Service(config, self.root, store)
        self.started = time.time()

    def variants(self, assay, n):
        s = read(self.root / "samples" / assay / f"n{n}_b1.json")
        return [{"variant_id": f"M{i+1:03d}", "mutation": v["id"], "sequence": v["seq"],
                 "experimental_score": float(v["score"])} for i, v in enumerate(s["variants"])]

    def anchors(self, assay, policy):
        ids = [v["variant_id"] for v in self.variants(assay, 50)]
        if policy == "random":
            return random.Random(runtime.digest([assay, 1, "anchors"])).sample(ids, 4)
        cr = read(self.source / "assays" / assay / "evaluations/n50/cr.json")
        if cr["status"] != "complete" or set(cr["scores"]) != set(ids):
            return None
        return _quantile_anchors(sorted(ids, key=lambda v: (cr["scores"][v], v)), 4)

    def prepare(self):
        old.validate_frozen_data(self.config, self.frozen)
        source_protocol = read(self.source / "protocol.json")
        if source_protocol["model"] != self.config.model.public_dict():
            raise runtime.ExecutionPaused("CR/ACR generation configuration mismatch")
        inventory = []
        for a in self.assays:
            assay = a["assay"]
            freeze(self.root / "input_metadata" / f"{assay}.json", {
                "metadata": model_metadata(self.metadata[assay]),
                "wt_sequence": str(self.metadata[assay]["RAW_CONSTRUCT_SEQ"]).upper()})
            largest = max(n for n in SIZES if n <= a["n_valid"])
            previous = []
            for n in SIZES:
                if n > largest:
                    inventory.append({"assay": assay, "N": n, "status": "not_applicable"})
                    continue
                sample = read(self.source / "samples" / assay / f"n{n}_b1.json")
                if len(sample["variants"]) != n or sample["variants"][:len(previous)] != previous:
                    raise ValueError("Non-nested source samples")
                previous = sample["variants"]
                freeze(self.root / "samples" / assay / f"n{n}_b1.json", sample)
                inventory.append({"assay": assay, "N": n, "status": "queued"})
            for policy in (["random", "quantile"] if assay in PILOT else ["random"]):
                anchors = self.anchors(assay, policy)
                freeze(self.root / "anchors" / policy / f"{assay}.json", anchors)
                if anchors is not None:
                    for n in SIZES:
                        if n <= largest:
                            g = groups(assay, [v["variant_id"] for v in self.variants(assay, n)], anchors)
                            freeze(self.root / "groups" / policy / assay / f"n{n}.json", g)
        manifest = {"protocol": PROTOCOL, "source_protocol_hash": runtime.digest(source_protocol),
                    "source_manifest_hash": runtime.digest(self.frozen), "sizes": list(SIZES),
                    "seed": 1, "grouping_seed": 1, "group_size": 25, "anchors": 4,
                    "prompt_hash": runtime.digest(SYSTEM), "model": self.config.model.public_dict(),
                    "eligible_cells": sum(r["status"] == "queued" for r in inventory),
                    "pilot": list(PILOT), "new_token_limit": self.args.experiment_budget,
                    "cumulative_token_limit": self.args.token_budget,
                    "code_hashes": {p.name: runtime.digest(p.read_text()) for p in
                                    [Path(__file__), Path(model.__file__), Path(old.__file__)]}}
        freeze(self.root / "protocol.json", manifest)
        freeze(self.root / "queue.json", inventory)

    def group(self, assay, policy, group, by_id):
        folder = self.root / "requests" / policy / assay / group["group_hash"]
        variants = [by_id[v] for v in group["variant_ids"]]
        user = prompt(self.metadata[assay], str(self.metadata[assay]["RAW_CONSTRUCT_SEQ"]).upper(), variants)
        expected = runtime.digest([SYSTEM, user, self.config.model.public_dict(), group])
        path = folder / "result.json"
        if path.exists():
            result = read(path)
            if result["input_hash"] != expected:
                raise runtime.ExecutionPaused("Group dependency hash mismatch")
            return result
        result = {"input_hash": expected, "group_hash": group["group_hash"], "calls": []}
        try:
            raw = self.service.request(folder, user, {"assay": assay, "policy": policy,
                                                      "group_hash": group["group_hash"]})
            result["calls"] = [raw]
            if classify(raw) == "provider_blocked":
                result["status"] = "provider_blocked"
            else:
                result.update(validate_answer(raw["answer"], group["variant_ids"]), status="complete")
        except (ValueError, TypeError, KeyError) as exc:
            result.update(status="invalid_output", error=str(exc))
        except old.TerminalRequest as exc:
            result.update(status=str(exc))
        if result["status"] not in {"token_preflight_unavailable", "transport_retry_exhausted"}:
            old.save(path, result)
        return result

    def cell(self, assay, n, policy):
        folder = self.root / "evaluations" / policy / assay / f"n{n}"
        variants = self.variants(assay, n)
        truth = {v["variant_id"]: v["experimental_score"] for v in variants}
        sample_hash = old._sample_hash_from_subset([(v["mutation"], v["sequence"],
                                                    v["experimental_score"]) for v in variants])
        anchors = read(self.root / "anchors" / policy / f"{assay}.json")
        if anchors is None:
            old.save(folder / "result.json", {"status": "ablation_not_applicable", "assay": assay,
                                              "N": n, "policy": policy, "sample_hash": sample_hash})
            return
        gs = read(self.root / "groups" / policy / assay / f"n{n}.json")
        started = time.time()
        by_id = {v["variant_id"]: v for v in variants}
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(lambda g: self.group(assay, policy, g, by_id), gs))
        complete = all(r["status"] == "complete" for r in results)
        valid = [r for r in results if r["status"] == "complete"]
        seen = set().union(*(set(r["ranking"]) for r in valid)) if valid else set()
        result = {"assay": assay, "N": n, "policy": policy, "sample_hash": sample_hash,
                  "truth": truth, "anchors": anchors, "groups": [r["group_hash"] for r in results],
                  "group_statuses": [r["status"] for r in results], "coverage": len(seen)/n,
                  "status": "complete" if complete else "incomplete", "fallback": False,
                  "wall_s": time.time()-started, "completed_at": time.time(),
                  "calls": [c for r in results for c in r["calls"]]}
        anchor_orders = [[v for v in r["ranking"] if v in anchors] for r in valid]
        result["anchor_orders"] = anchor_orders
        agreements = []
        for i, first in enumerate(anchor_orders):
            for second in anchor_orders[i+1:]:
                agreements.append(correlation({v: -j for j, v in enumerate(first)},
                                              {v: -j for j, v in enumerate(second)})["kendall"])
        result["anchor_pairwise_kendall"] = float(np.mean(agreements)) if agreements else None
        if any(r["status"] in {"token_preflight_unavailable", "transport_retry_exhausted"} for r in results):
            result["status"] = "pending_infrastructure"
        if complete:
            try:
                tick = time.perf_counter()
                result["scores"], result["optimizer"] = fit(list(truth), valid)
                result["aggregation_s"] = time.perf_counter()-tick
                result["mean_local_scores"] = mean_local_rank(list(truth), valid)
                result["metrics"] = old.clean(correlation(result["scores"], truth))
                result["nonanchor_metrics"] = old.clean(correlation(result["scores"], truth,
                                                        [v for v in truth if v not in anchors]))
            except ValueError as exc:
                result.update(status="aggregation_failed", error=str(exc))
        old.save(folder / "result.json", result)
        print(f"[ACR] {policy} {assay} N={n}: {result['status']}", flush=True)

    def run_assay(self, assay, policy):
        for n in SIZES:
            if n <= self.classes[assay]["n_valid"]:
                self.store.check_admission()
                self.cell(assay, n, policy)

    def phase(self, assays, policy, name):
        old.save(self.root / "phase.json", {"phase": name, "at": time.time()})
        # Six assay workers, two local requests each, capped globally at twelve.
        with ThreadPoolExecutor(max_workers=6) as pool:
            futures = [pool.submit(self.run_assay, a, policy) for a in assays]
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception:
                    self.store.stop.set()
                    raise
                self.report()

    def gate(self):
        outputs = [read(p) for p in (self.root / "evaluations/random").glob("*/n*/result.json")
                   if p.parent.parent.name in PILOT]
        requests = [read(p) for a in PILOT for p in
                    (self.root / "requests/random" / a).glob("*/result.json")]
        success = sum(r["status"] == "complete" for r in requests)
        valid_rate = success / max(1, len(requests))
        complete_assays = sum(sum(r["assay"] == a and r["status"] == "complete"
                                  for r in outputs) == 4 for a in PILOT)
        pending = any(r["status"] == "pending_infrastructure" for r in outputs)
        gate = {"passed": len(outputs) == 24 and complete_assays >= 5 and valid_rate >= .95 and not pending,
                "complete_assays": complete_assays, "local_valid_rate": valid_rate,
                "cells": len(outputs), "accuracy_not_used_for_gate": True}
        old.save(self.root / "engineering_gate.json", gate)
        if not gate["passed"]:
            raise runtime.ExecutionPaused("Engineering gate failed; inspect saved diagnostics")

    def report(self):
        rows, pending = [], []
        for entry in read(self.root / "queue.json"):
            if entry["status"] == "not_applicable":
                continue
            assay, n = entry["assay"], entry["N"]
            path = self.root / "evaluations/random" / assay / f"n{n}/result.json"
            if not path.exists():
                pending.append(entry)
                continue
            r = read(path)
            if r["status"] == "pending_infrastructure":
                pending.append({**entry, "status": r["status"]})
            truth = r["truth"]
            common = {"assay": assay, "N": n, "rna_type": self.classes[assay]["rna_type"],
                      "protein_stability_source": self.classes[assay]["protein_stability_source"],
                      "sample_hash": r["sample_hash"]}
            methods = [("ACR-seq", r, r.get("scores", {})),
                       ("ACR-mean-local", r, r.get("mean_local_scores", {}))]
            for method, filename in (("CR-v1", "cr.json"), ("Global-v1", "global.json")):
                source = self.source / "assays" / assay / "evaluations" / f"n{n}" / filename
                if source.exists():
                    old_r = read(source)
                    if old_r["sample_hash"] != r["sample_hash"]:
                        raise runtime.ExecutionPaused("Matched comparison sample mismatch")
                    scores = old_r.get("scores", {}) if method == "CR-v1" else {
                        v: -i for i, v in enumerate(old_r.get("ranking") or [])}
                    methods.append((method, old_r, scores))
                else:
                    methods.append((method, {"status": "source_pending_infrastructure", "coverage": 0}, {}))
            for method, record, scores in methods:
                valid = record["status"] == "complete" and set(scores) == set(truth)
                calls = record.get("calls", [])
                rows.append({**common, "method": method, "status": record["status"],
                             "strict_valid": valid, "coverage": record.get("coverage", int(valid)),
                             "fallback": False, "calls": len(calls),
                             "attributed_tokens": sum((c.get("usage") or {}).get("totalTokenCount", 0) or 0 for c in calls),
                             "summed_request_s": sum(c.get("elapsed_s", 0) or 0 for c in calls),
                             "queue_s": sum(c.get("queue_s", 0) or 0 for c in calls),
                             "wall_s": record.get("wall_s"),
                             **(correlation(scores, truth) if valid else {"spearman": None, "kendall": None})})
        out = self.root / "analysis"
        out.mkdir(exist_ok=True)
        if rows:
            pd.DataFrame(rows).to_csv(out / "experiment_long.csv", index=False)
        eligible = sum(e["status"] != "not_applicable" for e in read(self.root / "queue.json"))
        progress = {"eligible_cells": eligible,
                    "terminal_cells": eligible-len(pending), "pending_cells": len(pending),
                    "complete_acr_cells": sum(r["method"] == "ACR-seq" and r["strict_valid"] for r in rows),
                    "new_reported_tokens": self.store.own_tokens(), "at": time.time(),
                    **self.store.accounting()}
        old.save(self.root / "pending_queue.json", pending)
        old.save(self.root / "progress.json", progress)
        return progress

    def analysis(self):
        out = self.root / "analysis"
        path = out / "experiment_long.csv"
        if not path.exists():
            return
        frame = pd.read_csv(path)
        primary = frame[frame.method == "ACR-seq"]
        matched, diagnostics, ablation, anchor_diagnostics = [], [], [], []
        for _, r in primary.iterrows():
            a, n = r.assay, int(r.N)
            current = read(self.root / "evaluations/random" / a / f"n{n}/result.json")
            anchor_diagnostics.append({"assay": a, "N": n, "status": current["status"],
                                       "anchor_pairwise_kendall": current.get("anchor_pairwise_kendall")})
            alternate = self.root / "evaluations/quantile" / a / f"n{n}/result.json"
            if alternate.exists():
                q = read(alternate)
                ablation.append({"assay": a, "N": n, "random_status": current["status"],
                                 "quantile_status": q["status"],
                                 "delta_quantile_minus_random":
                                 (q.get("metrics", {}).get("spearman") or 0) - r.spearman
                                 if q["status"] == "complete" and pd.notna(q.get("metrics", {}).get("spearman"))
                                 and pd.notna(r.spearman) else None})
            for comparator in ("CR-v1", "Global-v1"):
                b = frame[(frame.assay == a) & (frame.N == n) & (frame.method == comparator)]
                if len(b) and pd.notna(r.spearman) and pd.notna(b.iloc[0].spearman):
                    matched.append({"assay": a, "N": n, "comparator": comparator,
                                    "rna_type": r.rna_type,
                                    "protein_stability_source": r.protein_stability_source,
                                    "delta": r.spearman-b.iloc[0].spearman})
            if current["status"] != "complete":
                continue
            cr_path = self.source / "assays" / a / "evaluations" / f"n{n}/cr.json"
            cr = read(cr_path)
            if cr["status"] != "complete":
                continue
            truth, scores = current["truth"], current["scores"]
            ids = list(truth)
            for name, prediction in (("ACR-seq", scores), ("CR-v1", cr["scores"])):
                buckets = {"within_cr_chunk": [], "between_cr_chunks": []}
                for i, vi in enumerate(ids):
                    for j in range(i+1, len(ids)):
                        vj = ids[j]
                        d = np.sign(truth[vi]-truth[vj])
                        if d == 0:
                            continue
                        correct = np.sign(prediction[vi]-prediction[vj])*d
                        buckets["within_cr_chunk" if i//25 == j//25 else "between_cr_chunks"].append((correct+1)/2)
                diagnostics.append({"assay": a, "N": n, "method": name,
                                    **{k: float(np.mean(v)) if v else None for k, v in buckets.items()},
                                    **{"nonanchor_"+k: v for k, v in correlation(prediction, truth,
                                      [v for v in ids if v not in current["anchors"]]).items()}})
        pd.DataFrame(matched).to_csv(out / "paired_deltas.csv", index=False)
        pd.DataFrame(diagnostics).to_csv(out / "chunk_and_nonanchor_diagnostics.csv", index=False)
        pd.DataFrame(ablation).to_csv(out / "anchor_ablation.csv", index=False)
        pd.DataFrame(anchor_diagnostics).to_csv(out / "anchor_consistency.csv", index=False)
        ci = []
        if matched:
            m = pd.DataFrame(matched)
            common_eligible = {a["assay"] for a in self.assays if a["n_valid"] >= 500}
            for comparator, data in m.groupby("comparator"):
                complete_ids = set(data.groupby("assay").N.nunique().loc[lambda s: s == 4].index)
                cohorts = {"all_available": set(data.assay), "common_eligible_65": common_eligible,
                           "all_four_N_valid": complete_ids}
                cohorts.update({"rna_type:"+str(t): set(g.assay) for t, g in data.groupby("rna_type")})
                cohorts["protein_stability"] = set(data.loc[data.protein_stability_source, "assay"])
                cohorts["not_protein_stability"] = set(data.loc[~data.protein_stability_source, "assay"])
                for cohort, allowed in cohorts.items():
                    matrix = data[data.assay.isin(allowed)].pivot(index="assay", columns="N", values="delta").reindex(columns=SIZES)
                    if matrix.empty:
                        continue
                    values = matrix.to_numpy()
                    rng = np.random.default_rng(1)
                    # Same assay resampling indices across N; not seed variance.
                    indices = rng.integers(0, len(values), (5000, len(values)))
                    samples = values[indices]
                    counts = np.isfinite(samples).sum(axis=1)
                    sums = np.nansum(samples, axis=1)
                    means = np.divide(sums, counts, out=np.full_like(sums, np.nan), where=counts > 0)
                    for j, n in enumerate(SIZES):
                        valid_values = values[:, j][np.isfinite(values[:, j])]
                        if not len(valid_values):
                            continue
                        lo, hi = np.nanquantile(means[:, j], [.025, .975])
                        ci.append({"comparator": comparator, "cohort": cohort, "N": n,
                                   "n_assays": len(valid_values), "mean_delta": float(valid_values.mean()),
                                   "median_delta": float(np.median(valid_values)), "ci95_low": lo, "ci95_high": hi})
        pd.DataFrame(ci).to_csv(out / "paired_bootstrap_ci.csv", index=False)
        frame.groupby(["method", "N"]).agg(n=("assay", "size"), valid=("strict_valid", "sum"),
                mean_spearman=("spearman", "mean"), median_spearman=("spearman", "median"),
                attributed_tokens=("attributed_tokens", "mean"), latency=("summed_request_s", "mean")).to_csv(out / "summary.csv")
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        allowed = {a["assay"] for a in self.assays if a["n_valid"] >= 500}
        for field in ("spearman", "strict_valid", "attributed_tokens", "summed_request_s"):
            fig, ax = plt.subplots(figsize=(6, 4))
            for method_name, data in frame[frame.assay.isin(allowed)].groupby("method"):
                avg = data.groupby("N")[field].mean()
                ax.plot(avg.index, avg.values, "o-", label=method_name)
            ax.set(xlabel="N", ylabel=field, title="Common eligible assays; accuracy valid-only (see denominators)")
            ax.legend(fontsize=8)
            fig.tight_layout()
            for suffix in ("pdf", "svg", "png"):
                fig.savefig(out / f"{field}_vs_N.{suffix}")
            plt.close(fig)
        # A fixed all-four-N valid cohort prevents a changing valid subset from
        # being mistaken for an accuracy scaling curve.
        required = frame[frame.method.isin(["ACR-seq", "CR-v1"])]
        counts = required[required.spearman.notna()].groupby("assay").size()
        fixed = set(counts[counts == 8].index) & allowed
        old.save(out / "fixed_acr_cr_cohort.json", sorted(fixed))
        if fixed:
            fig, ax = plt.subplots(figsize=(6, 4))
            for method_name, data in required[required.assay.isin(fixed)].groupby("method"):
                avg = data.groupby("N").spearman.mean()
                ax.plot(avg.index, avg.values, "o-", label=method_name)
            ax.set(xlabel="N", ylabel="Spearman", title=f"Same {len(fixed)} assays valid at all four N")
            ax.legend()
            fig.tight_layout()
            for suffix in ("pdf", "svg", "png"):
                fig.savefig(out / f"spearman_fixed_cohort.{suffix}")
            plt.close(fig)

    def run(self):
        self.prepare()
        if self.args.prepare_only:
            return self.report()
        self.service.preflight()
        self.phase(PILOT, "random", "engineering_pilot")
        self.gate()
        self.phase(PILOT, "quantile", "pilot_anchor_ablation")
        self.phase([a["assay"] for a in self.assays if a["assay"] not in PILOT], "random", "full_70_assays")
        result = self.report()
        self.analysis()
        return result


def supervise(args):
    root = Path(args.output_root)
    with (root / "supervisor.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        command = [sys.executable, "-u", "-m", "evagent.acr_seq"] + [a for a in sys.argv[1:] if a != "--supervise"]
        started = time.time()
        with (root / "worker.log").open("a") as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            while process.poll() is None:
                status = {"status": "running", "supervisor_pid": os.getpid(), "worker_pid": process.pid,
                          "elapsed_s": time.time()-started, "at": time.time()}
                old.save(root / "supervisor_status.json", status)
                if status["elapsed_s"] >= 43200 and not (root / "checkpoint_12h.json").exists():
                    old.save(root / "checkpoint_12h.json", status)
                time.sleep(30)
        old.save(root / "supervisor_status.json", {"status": "exited", "worker_exit_code": process.returncode,
                  "outcome": read(root / "worker_outcome.json"), "at": time.time()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_shared_arguments(parser)
    parser.set_defaults(output_root="experiments/"+PROTOCOL, seed=1, grouping_seed=1, trace_mode="off")
    parser.add_argument("--source-root", default="experiments/cr_fullseq_b25_v1")
    parser.add_argument("--legacy-root", default="runs_expansion")
    parser.add_argument("--token-budget", type=int, default=300_000_000)
    parser.add_argument("--experiment-budget", type=int, default=30_000_000)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--supervise", action="store_true")
    parser.add_argument("--analysis-only", action="store_true")
    args = parser.parse_args()
    if args.seed != 1 or args.grouping_seed != 1 or args.token_budget > 300_000_000 or args.experiment_budget > 30_000_000:
        raise SystemExit("Frozen protocol/authorized budget mismatch")
    root, legacy = Path(args.output_root), Path(args.legacy_root)
    if root.resolve() == Path(args.source_root).resolve():
        raise SystemExit("Output cannot overwrite source")
    root.mkdir(parents=True, exist_ok=True)
    if args.supervise:
        supervise(args)
        return
    configure_tracing("off", PROTOCOL)
    os.environ["EVAGENT_DISCONNECT_RETRIES"] = "0"
    config = build_config(args)
    config = replace(config, model=replace(config.model, retries=1))
    exp = None
    try:
        with (legacy / "runner.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with sqlite3.connect(f"file:{legacy/'requests.sqlite'}?mode=ro", uri=True) as db:
                if db.execute("SELECT COUNT(*) FROM requests WHERE status='inflight'").fetchone()[0]:
                    raise runtime.ExecutionPaused("Unresolved in-flight requests")
            store = Ledger(legacy / "requests.sqlite", args.token_budget, root, args.experiment_budget)
            runtime.STORE = store
            model.configure_api_concurrency(12)
            exp = Experiment(args, config, store)
            old.save(root / "worker_outcome.json", {"status": "running", "pid": os.getpid(), "at": time.time()})
            if args.analysis_only:
                result = exp.report()
                exp.analysis()
            else:
                result = exp.run()
            old.save(root / "worker_outcome.json", {"status": "prepared" if args.prepare_only else
                     "complete" if not result["pending_cells"] else "pending_infrastructure", "at": time.time()})
    except Exception as exc:
        if exp:
            exp.report()
        secret = os.environ.get("GEMINI_API_KEY", "")
        message = str(exc).replace(secret, "[REDACTED]") if secret else str(exc)
        old.save(root / "worker_outcome.json", {"status": "paused", "error_type": type(exc).__name__,
                                               "reason": message[:600], "at": time.time()})
        raise


if __name__ == "__main__":
    main()
