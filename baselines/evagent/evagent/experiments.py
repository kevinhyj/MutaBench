"""Offline ablations and budgeted matched RNAGym experiments.

python -m evagent.experiments --help
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import math
import random
import re
import sqlite3
import time
from collections import Counter, defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from langgraph.checkpoint.sqlite import SqliteSaver
from scipy.stats import kendalltau, spearmanr, t

from . import runtime
from .config import PILOT_ASSAYS
from .graph import RANK_PROMPT, RANK_SYSTEM, _strict_ranking, model_metadata
from .main import (
    _clean_metadata,
    _sample_hash,
    _sample_hash_from_subset,
    _utc_now,
    add_shared_arguments,
    build_config,
    configure_tracing,
    nested_subsample_split,
    run_one,
)
from .model import (
    GeminiAdapter,
    ModelError,
    configure_api_concurrency,
    ensure_rg_llm_importable,
    extract_json,
)
from .ranking import aggregate_rankings


def save(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2, default=str, allow_nan=True))
    temp.replace(path)


def state_at(run_dir: Path) -> dict:
    manifest = json.loads((run_dir / "run_manifest.json").read_text())
    with sqlite3.connect(f"file:{run_dir / 'checkpoint.sqlite'}?mode=ro", uri=True) as con:
        item = SqliteSaver(con).get_tuple({"configurable": {"thread_id": manifest["run_id"]}})
    if item is None:
        raise ValueError(f"No checkpoint at {run_dir}")
    return item.checkpoint["channel_values"]


def correlation(scores: dict, truth: dict, ids=None) -> dict:
    ids = list(truth if ids is None else ids)
    if len(ids) < 2:
        return {"spearman": math.nan, "kendall": math.nan}
    x, y = [scores[i] for i in ids], [truth[i] for i in ids]
    if len(set(x)) < 2 or len(set(y)) < 2:
        return {"spearman": math.nan, "kendall": math.nan}
    return {
        "spearman": float(spearmanr(x, y).statistic),
        "kendall": float(kendalltau(x, y).statistic),
    }


def mean_local_rank(ids, rankings):
    observations = defaultdict(list)
    for item in rankings:
        order = item["ranking"]
        for rank, vid in enumerate(order):
            observations[vid].append(1.0 - rank / max(1, len(order) - 1))
    if set(observations) != set(ids):
        raise ValueError("Local ranking coverage incomplete")
    return {vid: float(np.mean(observations[vid])) for vid in ids}


def audit_baseline(record):
    """Normalize zero padding only; never silently discard unknown IDs."""
    expected = record["ids"]
    idmap = {int(re.fullmatch(r"M(\d+)", x).group(1)): x for x in expected}
    text = record.get("response_text", "") or ""
    old_blob = re.search(r'"ranking"\s*:\s*\[(.*?)\]', text, re.S)
    old_tokens = re.findall(r"M\d{1,3}(?!\d)", old_blob.group(1) if old_blob else text)
    old_order = [f"M{int(token[1:]):02d}" for token in old_tokens]
    old_order = [token for token in old_order if token in set(expected)]
    original_valid = len(old_order) == len(expected) and len(set(old_order)) == len(expected)
    structural = "json"
    try:
        obj = extract_json(text)
        ranking = obj.get("ranking") if isinstance(obj, dict) else obj
        if not isinstance(ranking, list):
            raise ValueError("not ranking list")
    except (ValueError, TypeError):
        # Preserve order and duplicates in the stated ranking section.
        blob = re.search(r'"ranking"\s*:\s*\[(.*?)\]', text, re.S)
        ranking = re.findall(r"\bM\d+\b", blob.group(1)) if blob else []
        structural = "format" if ranking else "unparseable"
    normalized, unknown = [], []
    for value in ranking:
        match = re.fullmatch(r"M(\d+)", str(value).strip())
        if match and int(match.group(1)) in idmap:
            normalized.append(idmap[int(match.group(1))])
        else:
            unknown.append(str(value))
    counts = Counter(normalized)
    duplicates = [k for k, count in counts.items() if count > 1]
    missing = sorted(set(expected) - set(normalized))
    valid = len(normalized) == len(expected) and not (unknown or duplicates or missing)
    return {
        "valid": valid,
        "format": structural,
        "duplicates": duplicates,
        "missing": missing,
        "unknown": unknown,
        "ranking": normalized if valid else None,
        "original_parsed": bool(record.get("parsed")),
        "original_parser_valid_recomputed": original_valid,
    }


def call_cost(records):
    calls = [r for r in records if r.get("record_type") == "model_call"]
    return {
        "total_tokens": sum((r.get("usage") or {}).get("totalTokenCount", 0) for r in calls),
        "model_calls": len(calls),
        "summed_request_s": sum(r.get("elapsed_s", 0) for r in calls),
        "queue_s": sum(r.get("queue_s", 0) for r in calls),
        "unknown_usage_calls": sum("totalTokenCount" not in (r.get("usage") or {}) for r in calls),
    }


def diagnose_cell(run_dir: Path, cohort: str, classification: dict):
    summary = json.loads((run_dir / "run_summary.json").read_text())
    state = state_at(run_dir)
    variants = state["variants"]
    truth = {v["variant_id"]: v["experimental_score"] for v in variants}
    prior = {v["variant_id"]: v["predicted_score"] for v in state["variant_evidence"]}
    ids = list(truth)
    rankings = [r for r in state["local_rankings"] if r["partition_id"] >= 0]
    anchors = set(itertools.chain.from_iterable(r.get("anchor_ids", []) for r in rankings))
    records = state.get("reasoning_records", [])
    fixed = [r for r in records if r.get("stage") in {"assay_analyst", "variant_reasoner"}]
    sessions_path = run_dir / "execution_sessions.jsonl"
    sessions = (
        [json.loads(line) for line in sessions_path.read_text().splitlines()]
        if sessions_path.exists()
        else []
    )
    base = {
        "assay": summary["assay"],
        "size": len(ids),
        "seed": summary["seed"],
        "grouping_seed": summary.get("grouping_seed", summary["seed"]),
        "cohort": cohort,
        "sample_sha256": summary["sample_sha256"],
        "config_sha256": summary["config_sha256"],
        "prompt_sha256": summary["prompt_sha256"],
        "trace_id": summary["trace_id"],
        "observed_session_wall_s": sum(r["wall_s"] for r in sessions) if sessions else math.nan,
        "aggregation_s": summary.get("aggregation_s", math.nan),
        "verification_aggregation_s": summary.get("verification_aggregation_s", math.nan),
        "provider_model_versions": ";".join(
            sorted({r.get("model_version", "") for r in records if r.get("model_version")})
        ),
        "variant_fallback_count": sum(bool(r.get("fallback")) for r in state["variant_evidence"]),
        "local_fallback_count": sum(bool(r.get("fallback")) for r in rankings),
        "evidence_sha256": runtime.digest(state["variant_evidence"]),
        "groups_sha256": runtime.digest(
            [
                {k: r[k] for k in ("partition_id", "group_id", "variant_ids", "anchor_ids")}
                for r in rankings
            ]
        ),
        **classification,
    }
    rows, score_sets = [], {}

    def add(method, scores, cost_records, group_count, combination="", extra=None):
        score_sets[method] = scores
        for subset, selected in [
            ("all", ids),
            ("anchors", sorted(anchors)),
            ("nonanchors", [i for i in ids if i not in anchors]),
        ]:
            rows.append(
                {
                    **base,
                    "method": method,
                    "subset": subset,
                    "n_variants": len(selected),
                    "partitions": group_count,
                    "combination": combination,
                    "valid": set(scores) == set(ids) and all(np.isfinite(list(scores.values()))),
                    **correlation(scores, truth, selected),
                    **call_cost(cost_records),
                    **(extra or {}),
                }
            )

    add("direct-score", prior, fixed, 0)
    partitions = sorted({r["partition_id"] for r in rankings})
    for count in range(1, len(partitions) + 1):
        for combination in itertools.combinations(partitions, count):
            observations = [r for r in rankings if r["partition_id"] in combination]
            cost_records = fixed + [
                r
                for r in records
                if r.get("stage") == "local_ranker"
                and any(str(r.get("unit", "")).startswith(f"p{p}.") for p in combination)
            ]
            suffix = "+".join(map(str, combination))
            scores = aggregate_rankings(ids, observations, initial_scores=prior)
            add("PL:" + suffix, scores, cost_records, count, suffix)
            add(
                "MeanRank:" + suffix,
                mean_local_rank(ids, observations),
                cost_records,
                count,
                suffix,
            )
    unverified = score_sets["PL:" + "+".join(map(str, partitions))]
    verified = state["aggregate_scores"]
    add("verified", verified, [r for r in records if r.get("record_type") == "model_call"], 3)
    verification = next((r for r in state["local_rankings"] if r["partition_id"] < 0), None)
    candidates = verification["variant_ids"] if verification else []
    improved = worsened = original_wrong = raw_improved = raw_worsened = non_tied_pairs = 0
    raw_verify = {vid: -i for i, vid in enumerate(verification["ranking"])} if verification else {}
    for a, b in itertools.combinations(candidates, 2):
        target = np.sign(truth[a] - truth[b])
        if target == 0:
            continue
        non_tied_pairs += 1
        before, after = np.sign(unverified[a] - unverified[b]), np.sign(verified[a] - verified[b])
        original_wrong += int(before != target)
        improved += int(before != target and after == target)
        worsened += int(before == target and after != target)
        raw_after = np.sign(raw_verify[a] - raw_verify[b])
        raw_improved += int(before != target and raw_after == target)
        raw_worsened += int(before == target and raw_after != target)
    verifier = {
        **base,
        "candidate_count": len(candidates),
        "original_wrong_pairs": original_wrong,
        "corrected_pairs": improved,
        "worsened_pairs": worsened,
        "non_tied_pairs": non_tied_pairs,
        "raw_corrected_pairs": raw_improved,
        "raw_worsened_pairs": raw_worsened,
        "raw_verifier_kendall": correlation(raw_verify, truth, candidates)["kendall"],
        "local_kendall_before": correlation(unverified, truth, candidates)["kendall"],
        "local_kendall_after": correlation(verified, truth, candidates)["kendall"],
        "delta_spearman": correlation(verified, truth)["spearman"]
        - correlation(unverified, truth)["spearman"],
        "verification_present": verification is not None,
    }
    return rows, verifier


def reference(config):
    ensure_rg_llm_importable(config.rg_llm_root)
    from rg_llm.dataset import load_reference_sheet

    return load_reference_sheet(
        config.rg_llm_root / "data/RNAGym/fitness/reference_sheet_final.csv"
    )


def classify(row):
    protein = "protein folding stability" in str(row.get("TITLE", "")).lower()
    return {
        "rna_type": str(row["RNA_TYPE"]),
        "protein_stability_source": protein,
        "source_family": "protein_stability" if protein else "other_rnagym",
    }


def freeze(args):
    config = build_config(args)
    ref = reference(config)
    from rg_llm.dataset import load_assay, resolve_assay_path

    result = []
    for row in ref.to_dict("records"):
        assay = str(row["DMS_ID"])
        frame = load_assay(assay, config.data_root)
        frame = frame.dropna(subset=["mutant", "sequence", "DMS_score"])
        frame = frame[np.isfinite(pd.to_numeric(frame.DMS_score, errors="coerce"))]
        n = len(frame.drop_duplicates(subset=["mutant", "sequence"]))
        path = resolve_assay_path(assay, config.data_root)
        result.append(
            {
                "assay": assay,
                "n_valid": n,
                **classify(row),
                "data_sha256": __import__("hashlib").sha256(path.read_bytes()).hexdigest(),
                "eligible_sizes": [n0 for n0 in (250, 100, 50) if n >= n0],
            }
        )
    # Round-robin biological types, deterministic within type, independent of labels.
    buckets = defaultdict(list)
    for item in sorted(result, key=lambda r: r["assay"]):
        buckets[item["rna_type"]].append(item)
    rng = random.Random(1)
    for items in buckets.values():
        rng.shuffle(items)
    queues = [deque(buckets[k]) for k in sorted(buckets)]
    ordered = []
    while any(queues):
        for queue in queues:
            if queue:
                ordered.append(queue.popleft())
    payload = {
        "schema_version": 2,
        "metadata_policy": "whitelist-v2",
        "assays": ordered,
        "reference_sha256": runtime.digest(ref.fillna("").to_dict("records")),
        "matrix": [
            {
                "assay": r["assay"],
                "size": n,
                "seed": seed,
                "grouping_seed": seed,
                "eligible": n in r["eligible_sizes"],
            }
            for seed in (1, 2, 3)
            for n in (250, 100, 50)
            for r in ordered
        ],
    }
    path = Path(args.output_root) / "frozen_manifest.json"
    if path.exists() and json.loads(path.read_text()) != payload:
        raise RuntimeError("Frozen data manifest changed; cannot resume")
    save(path, payload)
    return payload


def paired_statistics(frame):
    all_rows = frame[frame.subset == "all"]
    output = []
    comparisons = [
        ("PL-budget-mean:3", "PL-budget-mean:1"),
        ("PL-budget-mean:1", "direct-score"),
        ("PL:0", "direct-score"),
        ("PL:0+1+2", "PL:0"),
        ("verified", "PL:0+1+2"),
        ("PL:0+1+2", "MeanRank:0+1+2"),
        ("PL:0", "Gemini-global"),
        ("direct-score", "Gemini-global"),
        ("PL:0+1+2", "Gemini-global"),
    ]
    for (cohort, size), block in all_rows.groupby(["cohort", "size"]):
        scopes = {
            "all": block,
            "non_protein_stability": block[~block.protein_stability_source.astype(bool)],
        }
        scopes.update({"type:" + str(k): g for k, g in block.groupby("rna_type")})
        for scope, sub in scopes.items():
            for seed_scope, part in [("pooled", sub)] + [
                (str(k), g) for k, g in sub.groupby("seed")
            ]:
                pivot = part.pivot_table(
                    index=["assay", "seed"], columns="method", values="spearman"
                )
                for a, b in comparisons:
                    if a not in pivot or b not in pivot:
                        continue
                    pairs = pivot[[a, b]].dropna()
                    delta = (pairs[a] - pairs[b]).groupby(level="assay").mean().to_numpy()
                    if not len(delta):
                        continue
                    mean = float(delta.mean())
                    rng = np.random.default_rng(1)
                    ci = (
                        np.quantile(
                            rng.choice(delta, (10000, len(delta)), replace=True).mean(axis=1),
                            [0.025, 0.975],
                        )
                        if len(delta) > 1
                        else [math.nan] * 2
                    )
                    se = (
                        float(delta.std(ddof=1) / np.sqrt(len(delta)))
                        if len(delta) > 1
                        else math.nan
                    )
                    ci90 = (
                        [
                            mean - t.ppf(0.95, len(delta) - 1) * se,
                            mean + t.ppf(0.95, len(delta) - 1) * se,
                        ]
                        if len(delta) > 1
                        else [math.nan] * 2
                    )
                    output.append(
                        {
                            "cohort": cohort,
                            "size": int(size),
                            "scope": scope,
                            "seed_scope": seed_scope,
                            "method": a,
                            "reference": b,
                            "n_assays": len(delta),
                            "n_matched_cells": len(pairs),
                            "mean_delta": mean,
                            "median_delta": float(np.median(delta)),
                            "bootstrap95_low": ci[0],
                            "bootstrap95_high": ci[1],
                            "wins": int(sum(delta > 0)),
                            "losses": int(sum(delta < 0)),
                            "tost90_low": ci90[0],
                            "tost90_high": ci90[1],
                            "equivalent_margin_001": bool(ci90[0] > -0.01 and ci90[1] < 0.01),
                        }
                    )
    return output


def diagnose(args):
    config = build_config(args)
    classes = {str(r["DMS_ID"]): classify(r) for r in reference(config).to_dict("records")}
    output = Path(args.output_root) / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    rows, verifier_rows, audit_rows = [], [], []
    for root in [Path(args.legacy_root), Path(args.output_root)]:
        cohort = "pilot-v1" if root == Path(args.legacy_root) else "whitelist-v2"
        for summary_path in sorted(root.glob("gemini*/*/n*/seed*/r3/run_summary.json")):
            summary = json.loads(summary_path.read_text())
            cache_key = runtime.digest({"v": 5, "summary": summary, "cohort": cohort})
            cache = output / "cells" / (cache_key + ".json")
            if cache.exists():
                result = json.loads(cache.read_text())
            else:
                cell_rows, cell_verifier = diagnose_cell(
                    summary_path.parent, cohort, classes[summary["assay"]]
                )
                result = {"rows": cell_rows, "verifier": cell_verifier}
                save(cache, result)
            rows.extend(result["rows"])
            verifier_rows.append(result["verifier"])
        for path in sorted(root.glob("baseline/gemini*/n*/b*/*.json")):
            record = json.loads(path.read_text())
            audit = audit_baseline(record)
            base = {
                "assay": record["assay"],
                "size": record["size"],
                "seed": record["seed"],
                "cohort": cohort,
                **classes[record["assay"]],
            }
            audit_rows.append({**base, **{k: v for k, v in audit.items() if k != "ranking"}})
            split_path = (
                root
                / "splits"
                / record["assay"].replace("/", "_")
                / f"n{record['size']}_b{record['seed']}.json"
            )
            if not split_path.exists():
                # Resolve the actual split naming convention used by main.py.
                from .main import _split_path

                split_path = _split_path(
                    root / "splits", record["assay"], record["size"], record["seed"]
                )
            split = json.loads(split_path.read_text())["variants"]
            truth = {vid: float(x["score"]) for vid, x in zip(record["ids"], split)}
            scores = {vid: -i for i, vid in enumerate(audit["ranking"] or [])}
            rows.append(
                {
                    **base,
                    "method": "Gemini-global",
                    "subset": "all",
                    "partitions": 0,
                    "sample_sha256": record["sample_sha256"],
                    "valid": audit["valid"],
                    "spearman": correlation(scores, truth)["spearman"]
                    if audit["valid"]
                    else math.nan,
                    "kendall": correlation(scores, truth)["kendall"]
                    if audit["valid"]
                    else math.nan,
                    "total_tokens": (record.get("usage") or {}).get("totalTokenCount", 0),
                    "model_calls": 1,
                    "summed_request_s": record.get("elapsed_s", 0),
                }
            )
    if not rows:
        return
    frame = pd.DataFrame(rows)
    # Expected performance over partition choices, not a new combined prediction.
    pl_rows = frame[frame.method.str.startswith("PL:")]
    choice_rows = []
    for _, choices in pl_rows.groupby(["cohort", "assay", "size", "seed", "subset", "partitions"]):
        row = choices.iloc[0].to_dict()
        row["method"] = f"PL-budget-mean:{int(row['partitions'])}"
        row["combination"] = "all_combinations"
        row["partition_choice_sd"] = float(choices.spearman.std(ddof=0))
        for field in ("spearman", "kendall", "total_tokens", "model_calls", "summed_request_s"):
            row[field] = float(choices[field].mean())
        choice_rows.append(row)
    frame = pd.concat([frame, pd.DataFrame(choice_rows)], ignore_index=True)
    frame.to_csv(output / "experiment_long.csv", index=False)
    pd.DataFrame(verifier_rows).to_csv(output / "verifier_diagnostics.csv", index=False)
    save(output / "baseline_audit.json", audit_rows)
    numeric_rows = []
    for path in sorted((Path(args.output_root) / "numeric_ablation").glob("*.json")):
        record = json.loads(path.read_text())
        source = state_at(
            Path(args.legacy_root) / config.model.model_id / record["assay"] / "n250/seed1/r3"
        )
        canonical = _sample_hash(source["variants"])
        if record["sample_sha256"] not in (canonical, runtime.digest(source["variants"])):
            raise ValueError("Numeric ablation sample mismatch")
        numeric_rows.append(
            {
                "assay": record["assay"],
                "cohort": "pilot-v1",
                "size": 250,
                "seed": 1,
                "sample_sha256": canonical,
                "full_spearman": record["full_metrics"]["spearman"],
                "numeric_spearman": record["numeric_metrics"]["spearman"],
                "delta_numeric_minus_full": record["numeric_metrics"]["spearman"]
                - record["full_metrics"]["spearman"],
                "fallback_groups": sum(bool(r.get("fallback")) for r in record["local_rankings"]),
                **call_cost([{**r, "record_type": "model_call"} for r in record["calls"]]),
            }
        )
    if numeric_rows:
        pd.DataFrame(numeric_rows).to_csv(output / "numeric_evidence_comparison.csv", index=False)
    paired = paired_statistics(frame)
    pd.DataFrame(paired).to_csv(output / "paired_comparisons.csv", index=False)
    # Partition selection variance within each cell, separate from sampling variability.
    partition = frame[(frame.subset == "all") & frame.method.str.startswith("PL:")]
    partition.groupby(["cohort", "assay", "size", "seed", "partitions"]).spearman.agg(
        ["mean", "std", "min", "max"]
    ).reset_index().to_csv(output / "partition_variability.csv", index=False)
    all_rows = frame[frame.subset == "all"]
    conflicts = all_rows.groupby(["cohort", "assay", "size", "seed"]).sample_sha256.nunique()
    if (conflicts > 1).any():
        raise ValueError("Matched sample hash mismatch")
    save(
        output / "analysis_summary.json",
        {
            "generated_at": _utc_now(),
            "n_cells": len(verifier_rows),
            "hash_mismatches": int(sum(conflicts > 1)),
            "paired": paired,
            "notes": [
                "Cohorts are separate because metadata/prompt policy changed.",
                "Baseline accuracy is conditional on strict output validity.",
                "Bootstrap unit is assay after averaging seeds; 6-assay pilot is exploratory.",
            ],
        },
    )
    figures(frame, pd.DataFrame(verifier_rows), output)
    print(f"[analysis] {len(verifier_rows)} cells -> {output}", flush=True)


def figures(frame, verifier, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "pdf.fonttype": 42})
    chosen = ["Gemini-global", "direct-score", "PL:0", "PL:0+1+2", "verified"]
    for cohort, block in frame[frame.subset == "all"].groupby("cohort"):
        for name, x, y in [
            ("spearman_vs_n", "size", "spearman"),
            ("validity_vs_n", "size", "valid"),
            ("tokens_vs_performance", "total_tokens", "spearman"),
            ("latency_vs_n", "size", "summed_request_s"),
        ]:
            fig, ax = plt.subplots(figsize=(6, 4))
            for method in chosen:
                sub = block[block.method == method]
                if sub.empty:
                    continue
                # Mean seeds within assay before macro averaging assays.
                fields = list(dict.fromkeys([v for v in [x, y] if v != "size"]))
                by_assay = (
                    sub.groupby(["assay", "size"])[fields].mean(numeric_only=True).reset_index()
                )
                means = by_assay.groupby("size")[fields].mean().reset_index()
                ax.plot(means[x], means[y], "o-", label=method)
            ax.set(xlabel=x, ylabel=y, title=f"{cohort} (baseline accuracy: valid outputs only)")
            ax.legend(fontsize=8)
            ax.grid(alpha=0.2)
            fig.tight_layout()
            for ext in ("png", "pdf"):
                fig.savefig(output / f"{cohort}_{name}.{ext}", dpi=180)
            plt.close(fig)
        fig, ax = plt.subplots(figsize=(6, 4))
        selected = block[block.method.str.startswith("PL:")]
        for n, sub in selected.groupby("size"):
            means = (
                sub.groupby(["assay", "seed", "partitions"])
                .spearman.mean()
                .groupby(["assay", "partitions"])
                .mean()
                .groupby("partitions")
                .mean()
            )
            ax.plot(means.index, means.values, "o-", label=f"N={n}")
        ax.set(
            xlabel="Number of partitions (all combinations averaged)",
            ylabel="Macro Spearman",
            title=cohort,
        )
        ax.legend()
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(output / f"{cohort}_partition_budget.{ext}", dpi=180)
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(6, 4))
        selected = verifier[verifier.cohort == cohort]
        for n, sub in selected.groupby("size"):
            ax.scatter([n] * len(sub), sub.delta_spearman, label=f"N={n}", alpha=0.7)
        ax.axhline(0, color="black", lw=0.7)
        ax.set(xlabel="N", ylabel="Verifier delta Spearman", title=cohort)
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(output / f"{cohort}_verifier_delta.{ext}", dpi=180)
        plt.close(fig)


def baseline_one(args, assay, size):
    config = build_config(args)
    from rg_llm import prompt

    subset = nested_subsample_split(
        config, assay, size, args.seed, Path(args.output_root) / "splits"
    )
    ref = reference(config)
    row = _clean_metadata(ref[ref.DMS_ID == assay].iloc[0].to_dict())
    user, ids = prompt.build_user_prompt(
        model_metadata(row), str(row["RAW_CONSTRUCT_SEQ"]).upper(), subset
    )
    path = (
        Path(args.output_root)
        / "baseline"
        / config.model.model_id
        / f"n{size}"
        / f"b{args.seed}"
        / f"{assay}.json"
    )
    expected_hash = _sample_hash_from_subset(subset)
    prompt_hash = runtime.digest([prompt.SYSTEM_PROMPT, user])
    if path.exists():
        result = json.loads(path.read_text())
        if (
            result["sample_sha256"] != expected_hash
            or result["prompt_sha256"] != prompt_hash
            or result.get("config_sha256")
            not in {
                config.fingerprint(),
                getattr(args, "compatible_baseline_config_sha256", config.fingerprint()),
            }
        ):
            raise ValueError("Baseline cache provenance mismatch")
        return result
    adapter = GeminiAdapter(
        config.model, config.rg_llm_root, 0, namespace=f"baseline-v2:{assay}:{size}:{args.seed}"
    )
    call = adapter._call(
        prompt.SYSTEM_PROMPT,
        user,
        "global_ranking",
        {"assay": assay, "size": size, "seed": args.seed},
    )
    result = {
        "assay": assay,
        "size": size,
        "seed": args.seed,
        "model": config.model.model_id,
        "thinking_level": config.model.thinking_level,
        "ids": ids,
        "sample_sha256": expected_hash,
        "prompt_sha256": prompt_hash,
        "config_sha256": config.fingerprint(),
        "metadata_policy": "whitelist-v2",
        "system_prompt": prompt.SYSTEM_PROMPT,
        "user_prompt": user,
        "response_text": call.answer,
        "thinking_text": call.thought_summary,
        "usage": call.usage,
        "elapsed_s": call.elapsed_s,
        "queue_s": call.queue_s,
        "model_version": call.model_version,
        "attempts": call.attempts,
        "finish_reason": call.finish_reason,
        "request_sha256": call.request_sha256,
        "completed_at_utc": _utc_now(),
    }
    audit = audit_baseline(result)
    result.update(parsed=audit["valid"], ranking=audit["ranking"], status="ok")
    truth = {vid: x[2] for vid, x in zip(ids, subset)}
    scores = {vid: -i for i, vid in enumerate(audit["ranking"] or [])}
    result.update(
        correlation(scores, truth) if audit["valid"] else {"spearman": None, "kendall": None}
    )
    save(path, result)
    return result


def preflight(config, include_trace=True):
    import urllib.request

    proxy = config.model.proxy
    opener = urllib.request.build_opener(
        urllib.request.ProxyHandler({"http": proxy, "https": proxy} if proxy else {})
    )
    urls = ["https://generativelanguage.googleapis.com"]
    if include_trace:
        urls.append("https://api.smith.langchain.com/info")
    for url in urls:
        try:
            opener.open(url, timeout=15).close()
        except urllib.error.HTTPError as e:
            if e.code != 404:
                raise runtime.ExecutionPaused(f"Preflight HTTP {e.code}") from e
        except OSError as e:
            raise runtime.ExecutionPaused("Proxy/service preflight failed") from e


def run_ablation(args):
    config = build_config(args)
    for assay in PILOT_ASSAYS:
        state = state_at(Path(args.legacy_root) / config.model.model_id / assay / "n250/seed1/r3")
        path = Path(args.output_root) / "numeric_ablation" / (assay + ".json")
        if path.exists():
            continue
        evidence = {
            v["variant_id"]: {k: v[k] for k in ("variant_id", "predicted_score", "confidence")}
            for v in state["variant_evidence"]
        }
        groups = [r for r in state["local_rankings"] if r["partition_id"] == 0]
        adapter = GeminiAdapter(config.model, config.rg_llm_root, namespace=f"numeric-v1:{assay}")
        results, calls = [], []
        for group in groups:
            user = RANK_PROMPT.format(
                assay_memory=json.dumps(state["assay_memory"]),
                evidence=json.dumps([evidence[i] for i in group["variant_ids"]]),
            )
            response = None
            fallback = False
            try:
                response = adapter.generate_json(
                    system=RANK_SYSTEM,
                    user=user,
                    run_name="numeric_local_ranker",
                    metadata={"assay": assay, "group": group["group_id"]},
                )
                ranking, rationale = _strict_ranking(response.data, group["variant_ids"])
            except (ModelError, ValueError, TypeError) as error:
                fallback = True
                ranking = sorted(
                    group["variant_ids"], key=lambda vid: (-evidence[vid]["predicted_score"], vid)
                )
                rationale = [
                    "Malformed ranking; same numeric-prior fallback as full-evidence ranker."
                ]
                if response is None:
                    calls.extend(c.model_dump() for c in getattr(error, "calls", []))
            results.append(
                {**group, "ranking": ranking, "rationale": rationale, "fallback": fallback}
            )
            if response is not None:
                calls.extend(c.model_dump() for c in response.calls)
        prior = {k: v["predicted_score"] for k, v in evidence.items()}
        truth = {v["variant_id"]: v["experimental_score"] for v in state["variants"]}
        scores = aggregate_rankings(list(truth), results, initial_scores=prior)
        full = aggregate_rankings(list(truth), groups, initial_scores=prior)
        save(
            path,
            {
                "assay": assay,
                "sample_sha256": _sample_hash(state["variants"]),
                "evidence_sha256": runtime.digest(evidence),
                "group_sha256": runtime.digest(groups),
                "scores": scores,
                "numeric_metrics": correlation(scores, truth),
                "full_metrics": correlation(full, truth),
                "local_rankings": results,
                "calls": calls,
            },
        )


def run_expansion(args):
    import fcntl

    root = Path(args.output_root)
    root.mkdir(parents=True, exist_ok=True)
    lock_handle = (root / "runner.lock").open("a")
    fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    config = build_config(args)
    preflight(config)
    configure_tracing(args.trace_mode, args.langsmith_project)
    configure_api_concurrency(args.api_concurrency)
    store = runtime.configure_store(root / "requests.sqlite", args.token_budget)
    manifest = freeze(args)
    jobs = [job for job in manifest["matrix"] if job["eligible"]]
    started = time.time()
    statuses = (
        json.loads((root / "queue_status.json").read_text())
        if (root / "queue_status.json").exists()
        else {}
    )

    def key(job):
        return f"{job['assay']}:n{job['size']}:s{job['seed']}"

    def cell(job):
        options = copy.copy(args)
        options.seed = job["seed"]
        options.grouping_seed = job["grouping_seed"]
        baseline_one(options, job["assay"], job["size"])
        run_one(options, job["assay"], job["size"])
        return "complete"

    paused = False
    for seed in (1, 2, 3):
        for size in (250, 100, 50):
            selected = [j for j in jobs if j["seed"] == seed and j["size"] == size]
            for offset in range(0, len(selected), args.assay_concurrency):
                batch = selected[offset : offset + args.assay_concurrency]
                for job in batch:
                    if statuses.get(key(job)) != "complete":
                        statuses[key(job)] = "running"
                save(root / "queue_status.json", statuses)
                with ThreadPoolExecutor(max_workers=args.assay_concurrency) as executor:
                    futures = {executor.submit(cell, job): job for job in batch}
                    for future in as_completed(futures):
                        job = futures[future]
                        try:
                            statuses[key(job)] = future.result()
                        except Exception as error:
                            statuses[key(job)] = "paused:" + type(error).__name__
                            store.stop.set()
                            paused = True
                            print(f"[paused] {key(job)} {error}", flush=True)
                        save(root / "queue_status.json", statuses)
                        save(
                            root / "budget_status.json",
                            {
                                **store.summary(),
                                "updated_at": _utc_now(),
                                "wall_s": time.time() - started,
                                "complete_cells": sum(v == "complete" for v in statuses.values()),
                                "eligible_cells": len(jobs),
                                "remaining_cells": len(jobs)
                                - sum(v == "complete" for v in statuses.values()),
                            },
                        )
                diagnose(args)
                if paused or store.summary()["reported_tokens"] >= args.token_budget:
                    return
                # The first batch is followed by the preregistered pilot reasoning ablation.
                if seed == 1 and size == 250 and offset == 0:
                    try:
                        run_ablation(args)
                    except runtime.ExecutionPaused:
                        save(root / "budget_status.json", store.summary())
                        return


def checkpoint_report(args):
    """Read the live ledger without creating a store or changing request states."""
    import fcntl

    root = Path(args.output_root)
    with sqlite3.connect(f"file:{root / 'requests.sqlite'}?mode=ro", uri=True) as con:
        statuses = con.execute(
            "SELECT status,COUNT(*),SUM(tokens) FROM requests GROUP BY status"
        ).fetchall()
        first = con.execute("SELECT MIN(started) FROM requests").fetchone()[0]
        retries = con.execute(
            "SELECT COALESCE(SUM(MAX(COALESCE(json_extract(raw,'$.attempts'),1)-1,0)),0) FROM requests WHERE status='complete'"
        ).fetchone()[0]
    active = False
    with (root / "runner.lock").open("r") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(handle, fcntl.LOCK_UN)
        except BlockingIOError:
            active = True
    completed = []
    for path in sorted(root.glob("gemini*/*/n*/seed*/r3/run_summary.json")):
        result = json.loads(path.read_text())
        completed.append(
            {
                k: result[k]
                for k in ("assay", "size", "seed", "metrics", "fallback_count", "valid_permutation")
            }
        )
    manifest = json.loads((root / "frozen_manifest.json").read_text())
    report = {
        "checked_at": _utc_now(),
        "runner_active": active,
        "elapsed_since_first_request_s": time.time() - first,
        "reported_tokens": sum(row[2] or 0 for row in statuses),
        "soft_limit": args.token_budget,
        "requests_by_status": {row[0]: row[1] for row in statuses},
        "unreported_retry_attempts": retries,
        "complete_cells": len(completed),
        "remaining_cells": sum(j["eligible"] for j in manifest["matrix"]) - len(completed),
        "numeric_ablation_assays_completed": len(list((root / "numeric_ablation").glob("*.json"))),
        "completed": completed,
    }
    save(root / "checkpoint_report.json", report)
    if (
        report["elapsed_since_first_request_s"] >= 1800
        and not (root / "first_30min_checkpoint.json").exists()
    ):
        save(root / "first_30min_checkpoint.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "completed"}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["diagnose", "freeze", "expand", "ablation", "status"])
    add_shared_arguments(parser)
    parser.set_defaults(
        output_root="runs_expansion",
        trace_mode="hosted",
        langsmith_project="evagent-rnagym-expansion",
    )
    parser.add_argument("--legacy-root", default="runs")
    parser.add_argument("--token-budget", type=int, default=5_000_000)
    args = parser.parse_args()
    if args.action == "status":
        checkpoint_report(args)
    elif args.action == "diagnose":
        diagnose(args)
    elif args.action == "freeze":
        manifest = freeze(args)
        print(
            "assays",
            len(manifest["assays"]),
            "eligible cells",
            sum(j["eligible"] for j in manifest["matrix"]),
        )
    elif args.action == "ablation":
        preflight(build_config(args))
        configure_tracing(args.trace_mode, args.langsmith_project)
        configure_api_concurrency(args.api_concurrency)
        runtime.configure_store(Path(args.output_root) / "requests.sqlite", args.token_budget)
        run_ablation(args)
    else:
        run_expansion(args)


if __name__ == "__main__":
    main()
