"""Explicitly authorized continuation of a completed scale-v3 campaign.

Preserves the parent and partitions its remaining budget. No predictor package,
model setting, prompt, input, historical ledger or terminal block is changed.
"""
from __future__ import annotations

import argparse
import collections
import copy
import fcntl
import json
import sqlite3
import time
from pathlib import Path

from evagent.fewshot.protocol import digest
from evagent.fewshot.runner import load, save, sha
from evagent.fewshot.scale.campaign import verify
from evagent.fewshot.scale.workflow import cached

RETRY_STATUSES = {"truncated", "transport_attempts_exhausted", "unknown_cost"}
INHERIT_STATUSES = {"ok", "blocked"}


def debit(path):
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as db:
        db.execute("PRAGMA query_only=ON")
        db.execute("BEGIN")
        rows = db.execute("SELECT status,tokens,unknown,reserve FROM attempts").fetchall()
    known = sum(tokens for _, tokens, _, _ in rows)
    unknown = sum(max(0, reserve - tokens) for status, tokens, flag, reserve in rows
                  if flag and status != "inflight")
    inflight = sum(max(0, reserve - tokens) for status, tokens, _, reserve in rows if status == "inflight")
    return {"known_tokens": known, "unknown_reserve": unknown, "inflight_reserve": inflight,
            "total_debit": known + unknown + inflight,
            "inflight": sum(status == "inflight" for status, *_ in rows), "attempts": len(rows)}


def continuation_config(parent_config, parent_debit, manifest):
    if parent_debit["inflight"]:
        raise ValueError("Parent still has inflight requests")
    remaining = parent_config["budget"] - parent_debit["total_debit"]
    if remaining <= 0:
        raise ValueError("No remaining authorized budget")
    config = copy.deepcopy(parent_config)
    config.pop("fingerprint")
    config["budget"] = remaining
    config["phases"] = ["static", "rounds", "extrapolation"]
    config["authorization"] = "User explicitly requested continuation after reviewing completed/failed tasks; one bounded recovery epoch within the original 200M cap."
    config["budget_scope"] = "Remaining allocation after all parent known tokens and unknown reserves; no additional 200M budget."
    config["continuation"] = manifest
    config["fingerprint"] = digest(config)
    return config


def prepare(parent, root):
    parent, root = parent.resolve(), root.resolve()
    if root == parent or root.parent != parent.parent:
        raise ValueError("Continuation must be a distinct sibling experiment")
    # This same parent lock is held for the lifetime of the continuation worker.
    with (parent / "worker.lock").open("a+") as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if root.exists():
            raise ValueError("Continuation directory already exists; inspect instead of overwriting")
        if not (parent / "DONE.json").exists() or (parent / "PAUSE").exists():
            raise ValueError("Parent must be completed and not operator-paused")
        if load(parent / "cron_status.json")["enabled"]:
            raise ValueError("Disable parent scheduler before allocating its remainder")
        claim = parent / "CONTINUATION_CHILD.json"
        if claim.exists():
            raise ValueError("Parent remaining budget is already allocated to a child")
        original = load(parent / "FROZEN.json")
        verify(parent, original)
        accounting = debit(parent / "campaign.sqlite")
        # Read and validate every reusable artifact before exposing a runnable root.
        predictions, counts = [], collections.Counter()
        for path in sorted((parent / "predictions").glob("*/result.json")):
            request = load(path.with_name("request.json"))
            result = cached(parent, request)
            status = result["status"]
            if status not in RETRY_STATUSES | INHERIT_STATUSES:
                raise ValueError("Unreviewed terminal response status")
            counts[status] += 1
            predictions.append({"request_hash": request["request_hash"], "status": status,
                                "result_sha256": sha(path), "request_sha256": sha(path.with_name("request.json")),
                                "action": "reuse" if status in INHERIT_STATUSES else "bounded_retry"})
        manifest = {"parent": str(parent), "root": str(root), "time": time.time(),
                    "parent_fingerprint": original["fingerprint"], "parent_debit": accounting,
                    "original_shared_cap": original["budget"],
                    "parent_ledger_sha256": sha(parent / "campaign.sqlite"),
                    "new_attempt_limit_per_logical_request": 3,
                    "retry_authorization": "User: 继续跑呗; renew one bounded epoch only, without erasing previous attempts",
                    "unchanged": ["model", "output limit", "seed", "prompts", "support/query split", "parser", "acquisition policies"],
                    "prior_status_counts": dict(counts), "predictions": predictions,
                    "preparation_script_sha256": sha(__file__)}
        config = continuation_config(original, accounting, manifest)
        root.mkdir()
        (root / "PAUSE").write_text("Preparation/verification gate; remove only after scheduler validation.\n")
        save(root / "CONTINUATION.json", manifest)
        for name in ("inputs", "COHORT.json", "screening.json", "model_config.json", "preflight.json"):
            (root / name).symlink_to(parent / name, target_is_directory=(parent / name).is_dir())
        (root / "predictions").mkdir()
        for entry in predictions:
            if entry["action"] == "reuse":
                key = entry["request_hash"]
                (root / "predictions" / key).symlink_to(parent / "predictions" / key, target_is_directory=True)
        task_count = 0
        # Copy the frozen task inputs/result files, leaving failed tasks to be
        # reconstructed against the new overlay of cached/retried predictions.
        for path in sorted((parent / "tasks").rglob("result.json")):
            task = load(path)
            if task["status"] != "complete":
                continue
            inputs = load(path.with_name("inputs.json"))
            if task["input_digest"] != digest(inputs):
                raise ValueError("Inherited task input identity mismatch")
            if any(load(parent / "predictions" / h / "result.json")["status"] != "ok" for h in task["requests"]):
                raise ValueError("Inherited complete task has invalid predictions")
            dest = root / "tasks" / path.parent.relative_to(parent / "tasks")
            dest.mkdir(parents=True, exist_ok=True)
            for name in ("result.json", "inputs.json"):
                (dest / name).symlink_to(path.parent / name)
            task_count += 1
        config["input_hashes"]["CONTINUATION.json"] = sha(root / "CONTINUATION.json")
        config["fingerprint"] = digest({k: v for k, v in config.items() if k != "fingerprint"})
        save(root / "FROZEN.json", config)
        verify(root, config)
        save(root / "preparation_result.json", {"complete_tasks_reused": task_count,
             "prediction_counts": dict(counts), "new_token_cap": config["budget"], "generation_calls": 0})
        # Single-allocation record: do not create concurrent children each using
        # the same parent remainder. The parent remains DONE and disabled.
        with claim.open("x") as handle:
            json.dump({"child": str(root), "allocation": config["budget"], "parent_debit": accounting,
                       "time": time.time()}, handle, indent=2)
        print(json.dumps(load(root / "preparation_result.json")), flush=True)


def cumulative_report(root):
    manifest = load(root / "CONTINUATION.json")
    parent = Path(manifest["parent"])
    prior = debit(parent / "campaign.sqlite")
    if prior != manifest["parent_debit"]:
        (root / "PAUSE").touch()
        raise ValueError("Parent charges changed; pause continuation for budget reconciliation")
    local = debit(root / "campaign.sqlite") if (root / "campaign.sqlite").exists() else {
        k: 0 for k in prior}
    combined = {k: prior[k] + local[k] for k in ("known_tokens", "unknown_reserve", "inflight_reserve", "total_debit")}
    combined.update(original_cap=manifest["original_shared_cap"],
                    remaining=manifest["original_shared_cap"] - combined["total_debit"],
                    parent=prior, continuation=local, time=time.time())
    save(root / "TOTAL_BUDGET.json", combined)
    report_path = root / "analysis/REPORT.md"
    if report_path.exists():
        old = report_path.read_text()
        corrected = old.replace(
            "Old campaigns are excluded from the new 200M cap and remain intact.",
            "This is a continuation allocation within the ORIGINAL 200M cap. The local ledger below excludes parent usage; TOTAL_BUDGET.json contains parent plus continuation, including all unknown reserves.")
        if corrected != old:
            temp = report_path.with_suffix(".continuation.tmp")
            temp.write_text(corrected)
            temp.replace(report_path)
    text = ("# Continuation budget and provenance\n\n"
            f"Original shared cap: {combined['original_cap']:,} tokens. Combined known: {combined['known_tokens']:,}; "
            f"unknown reserve: {combined['unknown_reserve']:,}; inflight reserve: {combined['inflight_reserve']:,}. "
            f"Remaining: {combined['remaining']:,}.\n\n"
            "Complete parent predictions are reused; parent failures and fees remain unchanged. "
            "The user authorized one new bounded recovery epoch; each logical request has at most three NEW submissions including repair, "
            "in addition to explicitly retained prior attempts. Permanent blocks are not resubmitted. "
            "Model, output ceiling, seed, prompts, labels and evaluation are unchanged.\n\n"
            "The continuation analysis/metrics.json includes inherited complete tasks and newly completed tasks, deduplicated by task identity. "
            "Compare coverage to the original 567 complete / 642 planned units; do not add the two report row counts.\n")
    (root / "CONTINUATION_REPORT.md").write_text(text)
    return combined


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "report"))
    parser.add_argument("--parent", type=Path)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        if args.parent is None:
            parser.error("prepare requires --parent")
        prepare(args.parent, args.root)
    else:
        print(json.dumps(cumulative_report(args.root)), flush=True)
