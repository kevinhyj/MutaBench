"""Pure request construction and real-measurement replay; never reads test labels."""
from __future__ import annotations

import numpy as np

from evagent.fewshot.evaluation import ridge_predict, sequence_nearest_neighbor
from evagent.fewshot.protocol import Condition, digest, make_request
from evagent.fewshot.runner import load, save

from . import PROTOCOL

SHOTS = (4, 8, 16, 32)
ARMS = ("zero", "raw32", "anchored32")


def measured_support(public, oracle, selected):
    by_id = {r["variant_id"]: r for r in public["pool"]}
    if len(selected) != len(set(selected)) or not set(selected) <= set(by_id):
        raise ValueError("Invalid acquisition IDs")
    test_sequences = {r["sequence"] for r in public["test"]}
    if any(by_id[i]["sequence"] in test_sequences for i in selected):
        raise ValueError("Support/test overlap")
    return [dict(by_id[i], variant_id=f"S{n:03}", acquisition_id=i, fitness=oracle[i])
            for n, i in enumerate(selected, 1)]


def acquire(public, prior_selected, count, policy, predictions=None):
    remaining = [i for i in public["random_order"] if i not in prior_selected]
    if policy == "greedy":
        if predictions is None or set(predictions) != set(remaining):
            raise ValueError("Greedy acquisition requires the complete unmeasured pool")
        remaining = sorted(remaining, key=lambda i: (-predictions[i], i))
    elif policy != "random":
        raise ValueError("Unknown acquisition policy")
    result = prior_selected + remaining[:count - len(prior_selected)]
    if len(result) != count:
        raise ValueError("Insufficient acquisition candidates")
    return result


def requests(meta, query, support, arm, identity, model):
    k = 0 if arm == "zero" else len(support)
    ep = dict(meta, episode_id=identity, query=query)
    sup = {"episode_id": identity, "examples": support if k else []}
    result = []
    for chunk in range((len(query) + 24) // 25):
        r = make_request(ep, sup, Condition(f"numeric_{k}", k, "numeric"), chunk)
        if arm.startswith("anchored"):
            values = np.asarray([s["fitness"] for s in support], dtype=float)
            center = float(np.median(values))
            scale = float(np.percentile(values, 75) - np.percentile(values, 25))
            scale = scale if scale > 0 else float(np.std(values)) or 1.0
            if not meta["higher_is_better"]:
                raise ValueError("Scale-v3 admitted data must use higher-is-better units")
            start = r["user"].index("Use one shared scale across independent chunks:")
            end = r["user"].index("for the measured property.", start) + len("for the measured property.")
            r["user"] = r["user"][:start] + (
                "Use one shared scale across independent chunks, anchored ONLY to the measured support. "
                f"First estimate the raw assay readout y; return score=(2/pi)*atan((y-({center:.17g}))/({scale:.17g})). "
                "The same constants apply to every candidate and every chunk. Zero represents the support median, "
                "NOT neutrality relative to WT. Higher raw measurements imply higher scores for this assay. "
                "The measured examples below remain in RAW assay units, not transformed scores."
            ) + r["user"][end:]
            r["user"] = r["user"].replace(
                "Estimate each mutant relative to wild type and the assay definition",
                "Estimate each mutant using the assay definition and the fixed support-derived transformation")
        r["protocol"] = PROTOCOL
        r["arm"] = arm
        # Cache only by actual request + interpretation/model; scheduling IDs never
        # prevent reuse of identical initial trajectories or static/random32 calls.
        r["request_hash"] = digest({"system": r["system"], "user": r["user"],
                                    "expected_ids": r["expected_ids"], "model": model,
                                    "protocol": PROTOCOL})
        result.append(r)
    return result


def cached(root, request, mode="live"):
    path = root / "predictions" / request["request_hash"] / "result.json"
    if not path.exists():
        return None
    result = load(path)
    if result["request_hash"] != request["request_hash"] or result["execution_mode"] != mode:
        raise ValueError("Cached result identity mismatch")
    if result.get("result_digest") != digest({k: v for k, v in result.items() if k != "result_digest"}):
        raise ValueError("Cached result checksum mismatch")
    original = load(path.with_name("request.json"))
    if any(original[k] != request[k] for k in ("system", "user", "expected_ids", "request_hash", "protocol")):
        raise ValueError("Cached prompt identity mismatch")
    if result["status"] == "ok":
        values = result["value"]
        if set(values) != set(request["expected_ids"]) or not all(np.isfinite(v) for v in values.values()):
            raise ValueError("Invalid complete cache")
    return result


def task_path(root, task_id):
    return root / "tasks" / task_id / "result.json"


def completed(root, task):
    path = task_path(root, task["task_id"])
    if path.exists():
        result = load(path)
        if result["input_digest"] != digest(task):
            raise ValueError("Task replay changed")
        return result
    return None


def build_task(root, assay, mode, arm, selected=None, support=None, split=None):
    base = root / "inputs" / assay
    public = load(base / (split or "") / "public.json")
    if support is None:
        support = measured_support(public, load(base / "oracle.json"), selected)
    name = f"{assay}/{mode}/{arm}"
    task = {"task_id": name, "assay": assay, "mode": mode, "arm": arm,
            "split": split or "random", "metadata": public["metadata"],
            "selected": selected, "support": support, "test": public["test"]}
    if mode.startswith("round/") and len(support) < 32:
        task["pool"] = [r for r in public["pool"] if r["variant_id"] not in selected]
    return task


def task_requests(task, model):
    # Only named sequence/metadata fields are projected by the existing protocol.
    all_requests = []
    arm = task["arm"]
    if task["mode"].startswith("round/"):
        arm = f"anchored{len(task['support'])}"
    for role in ("test", "pool"):
        if role in task:
            for r in requests(task["metadata"], task[role], task["support"], arm,
                              task["task_id"] + "/" + role, model):
                r["role"] = role
                all_requests.append(r)
    return all_requests


def finish_task(root, task, reqs):
    results = [cached(root, r) for r in reqs]
    if any(r is None for r in results):
        return None
    valid = all(r["status"] == "ok" for r in results)
    predictions = {role: {} for role in ("test", "pool") if role in task}
    for request, result in zip(reqs, results):
        if result["status"] == "ok":
            predictions[request["role"]].update(result["value"])
    baselines = {}
    if task["support"] and task["arm"] != "zero":
        for name, predict in (("nearest", sequence_nearest_neighbor), ("ridge", ridge_predict)):
            baselines[name] = predict(task["support"], task["test"])
    result = {"task_id": task["task_id"], "input_digest": digest(task),
              "assay": task["assay"], "mode": task["mode"], "arm": task["arm"],
              "split": task["split"], "selected": task["selected"],
              "shots": 0 if task["arm"] == "zero" else len(task["support"]),
              "status": "complete" if valid else "paused_task",
              "predictions": predictions, "baselines": baselines,
              "requests": [r["request_hash"] for r in reqs],
              "failures": [r["status"] for r in results if r["status"] != "ok"],
              "best_measured": max((s["fitness"] for s in task["support"]), default=None)
              if task["arm"] != "zero" else None}
    save(task_path(root, task["task_id"]).with_name("inputs.json"), task)
    save(task_path(root, task["task_id"]), result)
    return result


def phase_tasks(root, assays, phase):
    for assay in assays:
        public = load(root / "inputs" / assay / "public.json")
        if phase == "static":
            for arm in ARMS:
                yield build_task(root, assay, "static", arm, selected=public["random_order"][:32])
        elif phase == "extrapolation":
            for split in ("1_to_2", "12_to_3plus"):
                path = root / "inputs" / assay / split / "public.json"
                if path.exists():
                    p = load(path)
                    for arm in ARMS:
                        yield build_task(root, assay, "extrapolation/" + split, arm,
                                         support=p["support"], split=split)
        elif phase == "rounds" and public["multi_round_eligible"]:
            for policy in ("random", "greedy"):
                selected = public["random_order"][:4]
                for k in SHOTS:
                    task = build_task(root, assay, "round/" + policy, str(k), selected=selected)
                    yield task
                    old = completed(root, task)
                    if old is None or old["status"] != "complete":
                        break
                    if k != 32:
                        next_k = SHOTS[SHOTS.index(k) + 1]
                        selected = acquire(public, selected, next_k, policy,
                                           old["predictions"].get("pool"))


def scan_phase(root, assays, phase, model):
    pending, results = {}, []
    for task in phase_tasks(root, assays, phase):
        result = completed(root, task)
        if result is None:
            reqs = task_requests(task, model)
            result = finish_task(root, task, reqs)
            if result is None:
                for r in reqs:
                    if cached(root, r) is None:
                        pending.setdefault(r["request_hash"], r)
        if result is not None:
            results.append(result)
    # Interleave assays first and conditions second, rather than spending an
    # entire early budget on the first biological background.
    ordered = sorted(pending.values(), key=lambda r: (r["chunk_index"], r["episode_id"]))
    return ordered, results
