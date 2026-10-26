"""Single-seed, two-size benchmark using the existing cumulative request ledger."""

from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from . import runtime
from .experiments import (
    audit_baseline,
    baseline_one,
    classify,
    correlation,
    diagnose_cell,
    paired_statistics,
    preflight,
    reference,
    save,
    state_at,
)
from .graph import prompt_fingerprint
from .main import (
    _sample_hash,
    _sample_hash_from_subset,
    add_shared_arguments,
    build_config,
    configure_tracing,
    nested_subsample_split,
    run_one,
)
from .model import configure_api_concurrency
from .ranking import make_anchor_groups, validate_group_coverage


def validate_frozen_data(config, source):
    from rg_llm.dataset import resolve_assay_path

    if (
        runtime.digest(reference(config).fillna("").to_dict("records"))
        != source["reference_sha256"]
    ):
        raise ValueError("Frozen reference sheet changed")
    for item in source["assays"]:
        path = resolve_assay_path(item["assay"], config.data_root)
        if hashlib.sha256(path.read_bytes()).hexdigest() != item["data_sha256"]:
            raise ValueError(f"Frozen assay data changed: {item['assay']}")


def protocol(source):
    return {
        "protocol": "compact-v1",
        "sizes": [250, 50],
        "seeds": [1],
        "partitions": 1,
        "verifier": False,
        "source_sha256": runtime.digest(source),
        "assays": source["assays"],
        "matrix": [j for j in source["matrix"] if j["seed"] == 1 and j["size"] in (250, 50)],
    }


def compatible_source(args, assay, size):
    """Authorize cache namespace reuse only for the unchanged historical r3 protocol."""
    cfg = build_config(args)
    path = Path(args.output_root) / cfg.model.model_id / assay / f"n{size}" / "seed1" / "r3"
    if not (path / "run_manifest.json").exists():
        return None
    old = copy.copy(args)
    old.repeats, old.verify_count = 3, 16
    manifest = json.loads((path / "run_manifest.json").read_text())
    expected_sample = _sample_hash_from_subset(
        nested_subsample_split(cfg, assay, size, 1, Path(args.output_root) / "splits")
    )
    if (
        manifest["config_sha256"] != build_config(old).fingerprint()
        or manifest["prompt_sha256"] != prompt_fingerprint()
        or manifest["sample_sha256"] != expected_sample
    ):
        raise ValueError(f"Incompatible historical cache: {assay} n={size}")
    state = state_at(path)
    if _sample_hash(state["variants"]) != expected_sample:
        raise ValueError("Historical checkpoint sample mismatch")
    local = [r for r in state.get("local_rankings", []) if r["partition_id"] == 0]
    if local:
        groups = make_anchor_groups(
            state["variant_evidence"],
            group_size=args.group_size,
            anchor_count=args.anchors,
            repeats=1,
            seed=1,
        )
        fields = ("partition_id", "group_id", "variant_ids", "anchor_ids")

        def project(rows):
            return [{k: r[k] for k in fields} for r in rows]

        if project(local) != project(groups):
            raise ValueError("Historical first partition differs")
        validate_group_coverage(groups, [v["variant_id"] for v in state["variants"]], 1)
    # Each replay additionally verifies the complete request/evidence hash in RequestStore.
    return manifest["run_id"]


def report(args, manifest, store=None):
    root = Path(args.output_root)
    out = root / "compact_analysis"
    out.mkdir(exist_ok=True)
    cfg = build_config(args)
    classes = {str(r["DMS_ID"]): classify(r) for r in reference(cfg).to_dict("records")}
    rows, statuses, audits = [], [], []
    for job in manifest["matrix"]:
        assay, size = job["assay"], job["size"]
        run = root / cfg.model.model_id / assay / f"n{size}" / "seed1" / "r1"
        base = root / "baseline" / cfg.model.model_id / f"n{size}" / "b1" / f"{assay}.json"
        complete = (run / "run_summary.json").exists() and base.exists()
        statuses.append(
            {
                **job,
                "status": "not_applicable"
                if not job["eligible"]
                else "complete"
                if complete
                else "pending",
            }
        )
        if not complete:
            continue
        summary = json.loads((run / "run_summary.json").read_text())
        cache = out / "cells" / (runtime.digest(summary) + ".json")
        if cache.exists():
            result = json.loads(cache.read_text())
        else:
            result, _ = diagnose_cell(run, "compact-v1", classes[assay])
            save(cache, result)
        selected = [r for r in result if r["method"] in ("direct-score", "PL:0")]
        record = json.loads(base.read_text())
        audit = audit_baseline(record)
        audits.append({"assay": assay, "size": size, **audit})
        if record["sample_sha256"] != summary["sample_sha256"]:
            raise ValueError("Matched sample hash mismatch")
        state = state_at(run)
        # RG-LLM uses M01 while EVAgent uses M001. Match by the verified input order.
        truth = {
            vid: v["experimental_score"]
            for vid, v in zip(record["ids"], state["variants"], strict=True)
        }
        scores = {vid: -i for i, vid in enumerate(audit["ranking"] or [])}
        template = next(r for r in selected if r["subset"] == "all")
        global_row = {
            **template,
            "method": "Gemini-global",
            "valid": audit["valid"],
            "spearman": np.nan,
            "kendall": np.nan,
            "model_calls": 1,
            "total_tokens": (record.get("usage") or {}).get("totalTokenCount"),
            "summed_request_s": record.get("elapsed_s"),
            "queue_s": record.get("queue_s"),
            "observed_session_wall_s": np.nan,
            "aggregation_s": 0,
            "variant_fallback_count": 0,
            "local_fallback_count": 0,
            "evidence_sha256": None,
            "groups_sha256": None,
            "trace_id": None,
            "config_sha256": record["config_sha256"],
            "prompt_sha256": record["prompt_sha256"],
            "provider_model_versions": record.get("model_version"),
            "partitions": 0,
        }
        if audit["valid"]:
            global_row.update(correlation(scores, truth))
        rows.extend(selected + [global_row])
    save(root / "compact_queue_status.json", statuses)
    save(out / "baseline_audit.json", audits)
    frame = pd.DataFrame(rows)
    if len(frame):
        frame.to_csv(out / "experiment_long.csv", index=False)
        pd.DataFrame(paired_statistics(frame)).to_csv(out / "paired_comparisons.csv", index=False)
        common = {r["assay"] for r in manifest["matrix"] if r["size"] == 250 and r["eligible"]}
        all_rows = frame[frame.subset == "all"]
        all_rows.groupby(["size", "method"]).agg(
            n=("assay", "count"),
            spearman=("spearman", "mean"),
            valid_rate=("valid", "mean"),
            total_tokens=("total_tokens", "sum"),
        ).to_csv(out / "macro_summary.csv")
        # The scale curve uses only assays with both sizes completed, among the eligible 68.
        counts = all_rows.groupby("assay")["size"].nunique()
        paired_assays = common & set(counts[counts == 2].index)
        scale = all_rows[all_rows.assay.isin(paired_assays)]
        scale.to_csv(out / "scale_matched_cells.csv", index=False)
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        for field in ("spearman", "total_tokens", "summed_request_s", "valid"):
            fig, ax = plt.subplots(figsize=(6, 4))
            for method, block in scale.groupby("method"):
                values = block.groupby("size")[field].mean()
                ax.plot(values.index, values.values, "o-", label=method)
            ax.set(
                xlabel="N",
                ylabel=field,
                title=f"Matched size cohort: {len(paired_assays)}/68 assays",
            )
            if len(scale):
                ax.legend()
            fig.tight_layout()
            for suffix in ("png", "pdf"):
                fig.savefig(out / f"{field}_vs_n.{suffix}")
            plt.close(fig)
    done = sum(r["status"] == "complete" for r in statuses)
    budget = store.summary() if store else {}
    # Attribution is not new billing: historical cached r3 expenditures remain in the ledger.
    estimates = {}
    if len(frame):
        principal = frame[(frame.subset == "all") & frame.method.isin(["PL:0", "Gemini-global"])]
        for size, block in principal.groupby("size"):
            estimates[int(size)] = float(block.groupby("assay").total_tokens.sum().mean())
    remaining_estimate = sum(
        estimates.get(j["size"], estimates.get(250, 0) * j["size"] / 250)
        for j in statuses
        if j["status"] == "pending"
    )
    save(
        root / "compact_progress.json",
        {
            **budget,
            "complete_cells": done,
            "eligible_cells": 138,
            "remaining_cells": 138 - done,
            "remaining_attributed_tokens_estimate": remaining_estimate,
            "estimate_note": "Historical per-cell mean; N50 extrapolated until observed; excludes unknown retries",
            "updated_at_unix": time.time(),
        },
    )
    print(
        f"[compact] {done}/138 complete; reported tokens={budget.get('reported_tokens')}",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_shared_arguments(parser)
    parser.set_defaults(
        output_root="runs_expansion",
        repeats=1,
        verify_count=0,
        seed=1,
        grouping_seed=1,
        trace_mode="off",
        assay_concurrency=6,
        api_concurrency=12,
    )
    parser.add_argument("--token-budget", type=int, default=30_000_000)
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--wait-for-idle", action="store_true")
    args = parser.parse_args()
    if (args.repeats, args.verify_count, args.seed, args.grouping_seed) != (1, 0, 1, 1):
        parser.error("compact-v1 fixes repeats=1, verify_count=0 and both seeds=1")
    root = Path(args.output_root)
    source = json.loads((root / "frozen_manifest.json").read_text())
    manifest = protocol(source)
    if args.report_only:
        report(args, manifest)
        return
    # Same lock as the old runner. Never open RequestStore while that runner is alive.
    with (root / "runner.lock").open("a") as lock:
        print("[compact] waiting for exclusive runner lock", flush=True)
        fcntl.flock(lock, fcntl.LOCK_EX | (0 if args.wait_for_idle else fcntl.LOCK_NB))
        reference(build_config(args))  # Initialize RG-LLM import path before validation.
        validate_frozen_data(build_config(args), source)
        path = root / "compact_manifest.json"
        if path.exists() and json.loads(path.read_text()) != manifest:
            raise ValueError("Compact frozen protocol changed")
        save(path, manifest)
        preflight(build_config(args), include_trace=args.trace_mode == "hosted")
        configure_tracing(args.trace_mode, args.langsmith_project)
        configure_api_concurrency(args.api_concurrency)
        store = runtime.configure_store(root / "requests.sqlite", args.token_budget)
        historical = copy.copy(args)
        historical.repeats, historical.verify_count = 3, 16
        args.compatible_baseline_config_sha256 = build_config(historical).fingerprint()

        def cell(job):
            options = copy.copy(args)
            options.request_namespace = compatible_source(options, job["assay"], job["size"])
            baseline_one(options, job["assay"], job["size"])
            run_one(options, job["assay"], job["size"])

        jobs = [j for j in manifest["matrix"] if j["eligible"]]
        for offset in range(0, len(jobs), args.assay_concurrency):
            errors = []
            with ThreadPoolExecutor(max_workers=args.assay_concurrency) as pool:
                futures = {
                    pool.submit(cell, j): j for j in jobs[offset : offset + args.assay_concurrency]
                }
                for future in as_completed(futures):
                    try:
                        future.result()
                    except Exception as error:
                        store.stop.set()
                        errors.append({"job": futures[future], "error": str(error)})
            report(args, manifest, store)
            if errors:
                save(root / "compact_pause.json", errors)
                return
            if store.stop.is_set() or (root / "PAUSE").exists():
                return


if __name__ == "__main__":
    main()
