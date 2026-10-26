"""CLI, RNAGym loading, resumable execution, and experiment artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sqlite3
import time
import uuid
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from langgraph.checkpoint.sqlite import SqliteSaver
from scipy.stats import kendalltau, spearmanr

from .config import PILOT_ASSAYS, AppConfig, PipelineConfig
from .graph import EVAgentWorkflow, model_metadata, prompt_fingerprint
from .model import GeminiAdapter, configure_api_concurrency, ensure_rg_llm_importable


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _json_default(value: Any):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if pd.isna(value):
        return None
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _write_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default))
    temporary.replace(path)


def _clean_metadata(row: dict[str, Any]) -> dict[str, Any]:
    clean: dict[str, Any] = {}
    for key, value in row.items():
        if value is None or (not isinstance(value, (list, dict)) and pd.isna(value)):
            clean[key] = None
        elif isinstance(value, np.generic):
            clean[key] = value.item()
        else:
            clean[key] = value
    return clean


def configure_tracing(mode: str, project: str) -> str:
    if mode not in {"auto", "hosted", "off"}:
        raise ValueError(f"unknown trace mode {mode}")
    has_key = bool(os.environ.get("LANGSMITH_API_KEY"))
    if mode == "hosted" and not has_key:
        raise RuntimeError("--trace-mode hosted requires LANGSMITH_API_KEY")
    enabled = mode == "hosted" or (mode == "auto" and has_key)
    os.environ["LANGSMITH_TRACING"] = "true" if enabled else "false"
    if enabled:
        os.environ.setdefault("LANGSMITH_PROJECT", project)
        return "hosted"
    return "off"


def _split_path(root: Path, assay: str, size: int, seed: int) -> Path:
    return root / assay.replace("/", "_") / f"n{size}_b{seed}.json"


def _variant_key(mutation: str, sequence: str, score: float) -> tuple[str, str, float]:
    return str(mutation), str(sequence), round(float(score), 12)


def _nested_variant_order(
    rows: list[tuple[str, str, float]],
    prefix: list[tuple[str, str, float]],
    *,
    assay: str,
    seed: int,
) -> list[tuple[str, str, float]]:
    """Preserve a frozen prefix, then deterministically shuffle unused rows."""
    positions: dict[tuple[str, str, float], deque[int]] = defaultdict(deque)
    for index, row in enumerate(rows):
        positions[_variant_key(*row)].append(index)
    used: set[int] = set()
    ordered: list[tuple[str, str, float]] = []
    for row in prefix:
        candidates = positions.get(_variant_key(*row))
        if not candidates:
            raise ValueError(f"Frozen N=50 row no longer exists in assay {assay}: {row[0]}")
        index = candidates.popleft()
        used.add(index)
        ordered.append(rows[index])
    remaining = [index for index in range(len(rows)) if index not in used]
    random.Random(f"evagent-nested-v1:{assay}:{seed}").shuffle(remaining)
    ordered.extend(rows[index] for index in remaining)
    return ordered


def nested_subsample_split(
    config: AppConfig,
    assay: str,
    size: int,
    seed: int,
    splits_root: Path,
) -> list[tuple[str, str, float]]:
    """Materialize nested N=50/100/250 prefixes shared by baseline and EVAgent."""
    ensure_rg_llm_importable(config.rg_llm_root)
    from rg_llm.dataset import load_assay

    frame = load_assay(assay, config.data_root)
    frame = frame.dropna(subset=["mutant", "DMS_score", "sequence"])
    frame = frame[np.isfinite(pd.to_numeric(frame["DMS_score"], errors="coerce"))]
    frame = frame.drop_duplicates(subset=["mutant", "sequence"]).reset_index(drop=True)
    rows = [
        (str(row["mutant"]), str(row["sequence"]), float(row["DMS_score"]))
        for _, row in frame.iterrows()
    ]
    requested_sizes = sorted({n for n in (50, 100, 250, int(size)) if n <= len(rows)})
    if size > len(rows):
        raise ValueError(f"Requested nested size {size} > assay size {len(rows)} for {assay}")

    source_path = _split_path(config.rg_llm_root / "splits", assay, 50, seed)
    prefix: list[tuple[str, str, float]] = []
    if source_path.exists():
        source = json.loads(source_path.read_text())
        prefix = [(str(x["id"]), str(x["seq"]), float(x["score"])) for x in source["variants"]]
        if len(prefix) != 50:
            raise ValueError(f"Expected a 50-variant frozen prefix at {source_path}")
    ordered = _nested_variant_order(rows, prefix, assay=assay, seed=seed)
    family_hash = hashlib.sha256(
        json.dumps(ordered[: requested_sizes[-1]], separators=(",", ":")).encode()
    ).hexdigest()

    assay_dir = splits_root / assay.replace("/", "_")
    assay_dir.mkdir(parents=True, exist_ok=True)
    for nested_size in requested_sizes:
        path = _split_path(splits_root, assay, nested_size, seed)
        variants = [
            {"id": mutation, "seq": sequence, "score": score}
            for mutation, sequence, score in ordered[:nested_size]
        ]
        payload = {
            "schema_version": 2,
            "strategy": "evagent-nested-v1",
            "assay": assay,
            "size": nested_size,
            "batch": seed,
            "seed": seed,
            "n": nested_size,
            "parent_n": requested_sizes[-1],
            "family_sha256": family_hash,
            "source_n50": str(source_path) if source_path.exists() else None,
            "variants": variants,
        }
        if path.exists():
            current = json.loads(path.read_text())
            current_rows = [
                (str(x["id"]), str(x["seq"]), float(x["score"]))
                for x in current.get("variants", [])
            ]
            if current_rows != ordered[:nested_size]:
                raise RuntimeError(
                    f"Existing split {path} conflicts with nested-v1; move it aside before running"
                )
        else:
            _write_json(path, payload)
    return ordered[:size]


def load_initial_state(
    config: AppConfig,
    assay: str,
    size: int,
    seed: int,
    splits_root: Path,
    trace_id: str,
    run_id: str,
) -> dict[str, Any]:
    ensure_rg_llm_importable(config.rg_llm_root)
    from rg_llm.dataset import get_meta_row, load_reference_sheet

    reference = load_reference_sheet(
        config.rg_llm_root / "data" / "RNAGym" / "fitness" / "reference_sheet_final.csv"
    )
    meta = _clean_metadata(get_meta_row(reference, assay))
    wt = str(meta["RAW_CONSTRUCT_SEQ"]).upper()
    subset = nested_subsample_split(config, assay, size, seed, splits_root)
    width = max(3, len(str(size)))
    variants = [
        {
            "variant_id": f"M{i + 1:0{width}d}",
            "mutation": mutation,
            "sequence": sequence,
            "experimental_score": float(score),
        }
        for i, (mutation, sequence, score) in enumerate(subset)
    ]
    return {
        "assay_id": assay,
        "metadata": model_metadata(meta),
        "wt_sequence": wt,
        "variants": variants,
        "trace_id": trace_id,
        "run_id": run_id,
        "status": "initialized",
    }


def _score_map(scores: dict[str, float], variants: list[dict[str, Any]]) -> dict[str, float]:
    predicted = [float(scores[x["variant_id"]]) for x in variants]
    observed = [float(x["experimental_score"]) for x in variants]
    rho = spearmanr(predicted, observed).statistic
    tau = kendalltau(predicted, observed).statistic
    return {
        "spearman": float(rho) if np.isfinite(rho) else 0.0,
        "kendall": float(tau) if np.isfinite(tau) else 0.0,
    }


def _usage_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    totals: dict[str, int] = {}
    elapsed = 0.0
    calls = 0
    attempts = 0
    for record in records:
        if record.get("record_type") != "model_call":
            continue
        calls += 1
        elapsed += float(record.get("elapsed_s", 0.0))
        attempts += int(record.get("attempts", 1))
        for key, value in record.get("usage", {}).items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                totals[key] = totals.get(key, 0) + int(value)
    return {
        "model_calls": calls,
        "request_attempts": attempts,
        "summed_model_latency_s": elapsed,
        "tokens": totals,
    }


def _record_in_budget(record: dict[str, Any], budget: int, verified: bool) -> bool:
    stage = record.get("stage")
    if stage in {"assay_analyst", "variant_reasoner"}:
        return True
    if stage == "uncertainty_verifier":
        return verified
    if stage != "local_ranker":
        return False
    unit = str(record.get("unit", ""))
    if not unit.startswith("p") or ".g" not in unit:
        return False
    try:
        partition = int(unit[1:].split(".g", 1)[0])
    except ValueError:
        return False
    return 0 <= partition < budget


def _budget_summaries(
    records: list[dict[str, Any]], repeats: int
) -> tuple[dict[str, dict[str, Any]], dict[str, int]]:
    usage: dict[str, dict[str, Any]] = {}
    fallback: dict[str, int] = {}
    for budget in range(1, repeats + 1):
        selected = [x for x in records if _record_in_budget(x, budget, False)]
        usage[str(budget)] = _usage_summary(selected)
        fallback[str(budget)] = sum(x.get("record_type") == "fallback" for x in selected)
    selected = [x for x in records if _record_in_budget(x, repeats, True)]
    usage["verified"] = _usage_summary(selected)
    fallback["verified"] = sum(x.get("record_type") == "fallback" for x in selected)
    return usage, fallback


def _sample_hash(variants: list[dict[str, Any]]) -> str:
    payload = [(x["variant_id"], x["mutation"], x["sequence"]) for x in variants]
    return hashlib.sha256(json.dumps(payload, separators=(",", ":")).encode()).hexdigest()


def _sample_hash_from_subset(subset: list[tuple[str, str, float]]) -> str:
    width = max(3, len(str(len(subset))))
    variants = [
        {"variant_id": f"M{i + 1:0{width}d}", "mutation": row[0], "sequence": row[1]}
        for i, row in enumerate(subset)
    ]
    return _sample_hash(variants)


def write_artifacts(
    run_dir: Path,
    state: dict[str, Any],
    config: AppConfig,
    trace_mode: str,
    started_at: str,
) -> dict[str, Any]:
    predictions = pd.DataFrame(state["predictions"])
    predictions.insert(0, "assay", state["assay_id"])
    predictions.to_csv(run_dir / "predictions.csv", index=False)

    with (run_dir / "reasoning.jsonl").open("w", encoding="utf-8") as handle:
        for record in state.get("reasoning_records", []):
            handle.write(json.dumps(record, ensure_ascii=False, default=_json_default) + "\n")

    metrics = {
        label: _score_map(scores, state["variants"])
        for label, scores in state.get("budget_scores", {}).items()
    }
    strict_ids = [x["variant_id"] for x in state["predictions"]]
    expected_ids = [x["variant_id"] for x in state["variants"]]
    valid_permutation = len(strict_ids) == len(set(strict_ids)) and set(strict_ids) == set(
        expected_ids
    )
    budget_usage, budget_fallback = _budget_summaries(
        state.get("reasoning_records", []), config.pipeline.repeats
    )
    budget_validity = {
        label: set(scores) == set(expected_ids) and len(scores) == len(expected_ids)
        for label, scores in state.get("budget_scores", {}).items()
    }
    summary = {
        "schema_version": 1,
        "status": state["status"],
        "assay": state["assay_id"],
        "size": len(state["variants"]),
        "seed": config.pipeline.seed,
        "grouping_seed": config.pipeline.grouping_seed
        if config.pipeline.grouping_seed is not None
        else config.pipeline.seed,
        "model": config.model.model_id,
        "thinking_level": config.model.thinking_level,
        "repeats": config.pipeline.repeats,
        "config_sha256": config.fingerprint(),
        "prompt_sha256": prompt_fingerprint(),
        "sample_sha256": _sample_hash(state["variants"]),
        "trace_mode": trace_mode,
        "trace_id": state["trace_id"],
        "run_id": state["run_id"],
        "started_at_utc": started_at,
        "completed_at_utc": _utc_now(),
        "valid_permutation": valid_permutation,
        "budget_validity": budget_validity,
        "metrics": metrics,
        "usage": _usage_summary(state.get("usage_records", [])),
        "budget_usage": budget_usage,
        "budget_fallback_count": budget_fallback,
        "fallback_count": sum(
            1 for x in state.get("reasoning_records", []) if x.get("record_type") == "fallback"
        ),
        "aggregation_s": state.get("aggregation_s"),
        "verification_aggregation_s": state.get("verification_aggregation_s"),
        "provider_model_versions": sorted(
            {
                x.get("model_version", "")
                for x in state.get("usage_records", [])
                if x.get("model_version")
            }
        ),
    }
    _write_json(run_dir / "run_summary.json", summary)
    return summary


def build_config(args: argparse.Namespace) -> AppConfig:
    root = Path(args.rg_llm_root).resolve()
    base = AppConfig.defaults(root)
    pipeline = PipelineConfig(
        seed=args.seed,
        grouping_seed=getattr(args, "grouping_seed", None),
        reasoning_batch_size=args.reasoning_batch_size,
        local_group_size=args.group_size,
        anchor_count=args.anchors,
        repeats=args.repeats,
        max_concurrency=args.concurrency,
        verification_candidates=args.verify_count,
        schema_retries=args.schema_retries,
    )
    return AppConfig(
        model=base.model,
        pipeline=pipeline,
        rg_llm_root=root,
        data_root=Path(args.data_root),
    )


def run_one(args: argparse.Namespace, assay: str, size: int) -> dict[str, Any]:
    config = build_config(args)
    trace_mode = configure_tracing(args.trace_mode, args.langsmith_project)
    output_root = Path(args.output_root).resolve()
    run_dir = (
        output_root
        / config.model.model_id
        / assay
        / f"n{size}"
        / f"seed{config.pipeline.seed}"
        / f"r{config.pipeline.repeats}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    summary_path = run_dir / "run_summary.json"
    current_sample = _sample_hash_from_subset(
        nested_subsample_split(config, assay, size, config.pipeline.seed, output_root / "splits")
    )
    if summary_path.exists() and not args.overwrite:
        summary = json.loads(summary_path.read_text())
        if (
            summary.get("status") == "complete"
            and summary.get("config_sha256") == config.fingerprint()
            and summary.get("prompt_sha256") == prompt_fingerprint()
            and summary.get("sample_sha256") == current_sample
        ):
            print(f"[skip] {assay} n={size}: complete")
            return summary

    manifest_path = run_dir / "run_manifest.json"
    if manifest_path.exists() and not args.overwrite:
        manifest = json.loads(manifest_path.read_text())
    else:
        manifest = {
            "run_id": str(uuid.uuid4()),
            "trace_id": str(uuid.uuid4()),
            "created_at_utc": _utc_now(),
            "config_sha256": config.fingerprint(),
            "prompt_sha256": prompt_fingerprint(),
            "sample_sha256": current_sample,
            "request_namespace": getattr(args, "request_namespace", None),
        }
        _write_json(manifest_path, manifest)
    if manifest.get("config_sha256") != config.fingerprint():
        raise RuntimeError(
            f"Existing run at {run_dir} uses another config; pass --overwrite or change parameters"
        )
    if (
        manifest.get("prompt_sha256") != prompt_fingerprint()
        or manifest.get("sample_sha256") != current_sample
    ):
        raise RuntimeError("Checkpoint prompt/sample version mismatch; use an isolated output root")

    started_at = _utc_now()
    state = load_initial_state(
        config,
        assay,
        size,
        config.pipeline.seed,
        output_root / "splits",
        manifest["trace_id"],
        manifest["run_id"],
    )
    model = GeminiAdapter(
        config.model,
        config.rg_llm_root,
        schema_retries=config.pipeline.schema_retries,
        namespace=manifest.get("request_namespace") or manifest["run_id"],
    )
    checkpoint_path = run_dir / "checkpoint.sqlite"
    connection = sqlite3.connect(checkpoint_path, check_same_thread=False)
    session_started = time.perf_counter()
    try:
        graph = EVAgentWorkflow(model, config.pipeline).compile(SqliteSaver(connection))
        graph_config = {
            "configurable": {"thread_id": manifest["run_id"]},
            "run_id": uuid.UUID(manifest["trace_id"]),
            "run_name": f"EVAgent:{assay}:n{size}:r{config.pipeline.repeats}",
            "tags": ["evagent", "rnagym", f"n{size}", f"r{config.pipeline.repeats}"],
            "metadata": {
                "assay": assay,
                "size": size,
                "seed": config.pipeline.seed,
                "model": config.model.model_id,
            },
            "recursion_limit": 30,
        }
        snapshot = graph.get_state(graph_config)
        if snapshot.values and snapshot.next and not args.overwrite:
            print(f"[resume] {assay} n={size} from {snapshot.next}")
            final_state = graph.invoke(None, graph_config)
        elif snapshot.values and not snapshot.next and snapshot.values.get("status") == "complete":
            final_state = snapshot.values
        else:
            print(f"[run] {assay} n={size} repeats={config.pipeline.repeats}")
            final_state = graph.invoke(state, graph_config)
    finally:
        connection.close()
        with (run_dir / "execution_sessions.jsonl").open("a") as handle:
            handle.write(
                json.dumps(
                    {
                        "started_at": started_at,
                        "ended_at": _utc_now(),
                        "wall_s": time.perf_counter() - session_started,
                    }
                )
                + "\n"
            )

    summary = write_artifacts(run_dir, final_state, config, trace_mode, started_at)
    print(
        f"[done] {assay} n={size} valid={summary['valid_permutation']} "
        f"spearman={summary['metrics'].get('verified', summary['metrics'].get(str(config.pipeline.repeats), {})).get('spearman', float('nan')):+.3f}"
    )
    return summary


def run_baseline(args: argparse.Namespace, assays: list[str], sizes: list[int]) -> None:
    config = build_config(args)
    api_key = os.environ.get(config.model.api_key_env)
    if not api_key:
        raise RuntimeError(f"{config.model.api_key_env} is required for baseline calls")
    ensure_rg_llm_importable(config.rg_llm_root)
    from rg_llm.dataset import load_reference_sheet
    from rg_llm.runner import run_cell

    reference = load_reference_sheet(
        config.rg_llm_root / "data" / "RNAGym" / "fitness" / "reference_sheet_final.csv"
    )
    output_root = Path(args.output_root).resolve()
    spec = config.model.public_dict()

    def run_cell_for_assay(assay: str, size: int) -> dict[str, Any]:
        subset = nested_subsample_split(
            config, assay, size, config.pipeline.seed, output_root / "splits"
        )
        result = run_cell(
            model=config.model.model_id,
            spec=spec,
            api_key=api_key,
            assay=assay,
            size=size,
            seed=config.pipeline.seed,
            ref_sheet=reference,
            results_root=output_root / "baseline",
            splits_root=output_root / "splits",
            data_root=config.data_root,
            timeout=max(3600, config.model.timeout_s),
        )
        result["sample_sha256"] = _sample_hash_from_subset(subset)
        result["thinking_level"] = config.model.thinking_level
        result_path = (
            output_root
            / "baseline"
            / config.model.model_id
            / f"n{size}"
            / f"b{config.pipeline.seed}"
            / f"{assay}.json"
        )
        _write_json(result_path, result)
        return result

    for size in sizes:
        with ThreadPoolExecutor(max_workers=min(args.assay_concurrency, len(assays))) as executor:
            futures = {executor.submit(run_cell_for_assay, assay, size): assay for assay in assays}
            for future in as_completed(futures):
                assay = futures[future]
                result = future.result()
                print(
                    f"[baseline] {assay} n={size} spearman={result.get('spearman')}",
                    flush=True,
                )


def _method_name(budget: str, repeats: int) -> str:
    return f"EVAgent-r{repeats}+verify" if budget == "verified" else f"EVAgent-r{budget}"


def _bootstrap_mean_ci(values: list[float], iterations: int = 10_000) -> list[float] | None:
    if not values:
        return None
    array = np.asarray(values, dtype=float)
    rng = np.random.default_rng(1)
    draws = rng.choice(array, size=(iterations, len(array)), replace=True).mean(axis=1)
    return [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]


def write_pilot_summary(output_root: Path) -> Path:
    rows: list[dict[str, Any]] = []
    for path in output_root.glob("*/**/run_summary.json"):
        if "baseline" in path.parts:
            continue
        summary = json.loads(path.read_text())
        for budget, metrics in summary.get("metrics", {}).items():
            budget_usage = summary.get("budget_usage", {}).get(budget, {})
            rows.append(
                {
                    "method": _method_name(budget, int(summary["repeats"])),
                    "assay": summary["assay"],
                    "size": summary["size"],
                    "seed": summary["seed"],
                    "spearman": metrics["spearman"],
                    "kendall": metrics["kendall"],
                    "total_tokens": budget_usage.get("tokens", {}).get("totalTokenCount"),
                    "model_calls": budget_usage.get("model_calls"),
                    "request_attempts": budget_usage.get("request_attempts"),
                    "latency_s": budget_usage.get("summed_model_latency_s"),
                    "fallback_count": summary.get("budget_fallback_count", {}).get(budget),
                    "valid": summary.get("budget_validity", {}).get(budget, False),
                    "sample_sha256": summary.get("sample_sha256"),
                    "trace_id": summary.get("trace_id"),
                }
            )
    for path in (output_root / "baseline").glob("*/n*/b*/*.json"):
        record = json.loads(path.read_text())
        rows.append(
            {
                "method": "Gemini-global",
                "assay": record.get("assay"),
                "size": record.get("size"),
                "seed": record.get("seed"),
                "spearman": record.get("spearman"),
                "kendall": None,
                "total_tokens": record.get("usage", {}).get("totalTokenCount"),
                "model_calls": 1,
                "request_attempts": record.get("attempts"),
                "latency_s": record.get("elapsed_s"),
                "fallback_count": 0 if record.get("parsed") else 1,
                "valid": record.get("parsed"),
                "sample_sha256": record.get("sample_sha256"),
                "trace_id": None,
            }
        )
    out = output_root / "pilot_summary.csv"
    frame = pd.DataFrame(rows)
    if frame.empty:
        frame.to_csv(out, index=False)
        return out
    baseline_hash = {
        (row.assay, int(row.size), int(row.seed)): row.sample_sha256
        for row in frame.itertuples()
        if row.method == "Gemini-global"
    }
    frame["sample_matches_baseline"] = [
        True
        if row.method == "Gemini-global"
        else row.sample_sha256 == baseline_hash.get((row.assay, int(row.size), int(row.seed)))
        for row in frame.itertuples()
    ]
    frame = frame.sort_values(["size", "assay", "method"])
    frame.to_csv(out, index=False)

    comparison = frame.pivot_table(
        index=["size", "seed", "assay"], columns="method", values="spearman", aggfunc="first"
    ).reset_index()
    for method in [x for x in frame["method"].unique() if x != "Gemini-global"]:
        if "Gemini-global" in comparison and method in comparison:
            comparison[f"delta:{method}"] = comparison[method] - comparison["Gemini-global"]
    comparison.to_csv(output_root / "pilot_assay_comparison.csv", index=False)

    aggregate: dict[str, Any] = {"generated_at_utc": _utc_now(), "sizes": {}}
    for size, size_frame in frame.groupby("size"):
        size_report: dict[str, Any] = {"methods": {}, "paired_deltas": {}}
        for method, method_frame in size_frame.groupby("method"):
            values = method_frame["spearman"].dropna().astype(float).tolist()
            size_report["methods"][method] = {
                "n_assays": len(values),
                "mean_spearman": float(np.mean(values)) if values else None,
                "median_spearman": float(np.median(values)) if values else None,
                "valid_rate": float(method_frame["valid"].fillna(False).mean()),
                "fallback_rate": float(
                    method_frame["fallback_count"].fillna(0).sum()
                    / max(1, method_frame["model_calls"].fillna(0).sum())
                ),
            }
        baseline = size_frame[size_frame["method"] == "Gemini-global"].set_index("assay")
        for method in [x for x in size_frame["method"].unique() if x != "Gemini-global"]:
            candidate = size_frame[size_frame["method"] == method].set_index("assay")
            shared = baseline.index.intersection(candidate.index)
            deltas = [
                float(candidate.loc[assay, "spearman"] - baseline.loc[assay, "spearman"])
                for assay in shared
                if pd.notna(candidate.loc[assay, "spearman"])
                and pd.notna(baseline.loc[assay, "spearman"])
            ]
            size_report["paired_deltas"][method] = {
                "n_assays": len(deltas),
                "mean_delta_spearman": float(np.mean(deltas)) if deltas else None,
                "bootstrap_95ci": _bootstrap_mean_ci(deltas),
            }
        aggregate["sizes"][str(int(size))] = size_report
    _write_json(output_root / "pilot_aggregate.json", aggregate)
    return out


def add_shared_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--rg-llm-root", default="/data/yanjie_huang/RG-LLM")
    parser.add_argument("--data-root", default=str(AppConfig().data_root))
    parser.add_argument("--output-root", default="runs")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--grouping-seed", type=int)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--reasoning-batch-size", type=int, default=24)
    parser.add_argument("--group-size", type=int, default=32)
    parser.add_argument("--anchors", type=int, default=8)
    parser.add_argument("--verify-count", type=int, default=16)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--assay-concurrency", type=int, default=6)
    parser.add_argument("--api-concurrency", type=int, default=12)
    parser.add_argument("--schema-retries", type=int, default=1)
    parser.add_argument("--trace-mode", choices=["auto", "hosted", "off"], default="auto")
    parser.add_argument("--langsmith-project", default="evagent-rnagym-pilot")
    parser.add_argument("--overwrite", action="store_true")


def smoke(args: argparse.Namespace) -> None:
    config = build_config(args)
    configure_tracing(args.trace_mode, args.langsmith_project)
    model = GeminiAdapter(config.model, config.rg_llm_root, schema_retries=0)
    response = model.generate_json(
        system="Return valid JSON only.",
        user='Return exactly {"status":"ok"}.',
        run_name="gemini_smoke",
        metadata={"smoke": True},
    )
    print(
        json.dumps(
            {
                "parsed": response.data,
                "usage": response.calls[-1].usage,
                "model_version": response.calls[-1].model_version,
            },
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    experiment_parser = subparsers.add_parser("experiment", help="run an isolated integrated protocol")
    experiment_parser.add_argument("--config", type=Path, required=True)
    experiment_parser.add_argument("--dry-run", action="store_true")

    campaign_parser = subparsers.add_parser("campaign", help="run a frozen, globally budgeted matrix")
    campaign_parser.add_argument("--config", type=Path, required=True)
    campaign_parser.add_argument("--dry-run", action="store_true")
    evidence_parser = subparsers.add_parser("evidence", help="prepare or validate frozen evidence")
    evidence_parser.add_argument("action", choices=["prepare", "validate"])
    evidence_parser.add_argument("--config", type=Path, required=True)

    memory_parser = subparsers.add_parser("memory", help="explicit memory administration")
    memory_parser.add_argument("action", choices=["init", "show", "write-basic", "import-basic"])
    memory_parser.add_argument("--basic-db", type=Path)
    memory_parser.add_argument("--learning-db", type=Path)
    memory_parser.add_argument("--file", type=Path)
    memory_parser.add_argument("--source-db", type=Path)

    run_parser = subparsers.add_parser("run", help="run one assay/size cell")
    add_shared_arguments(run_parser)
    run_parser.add_argument("--assay", required=True)
    run_parser.add_argument("--size", type=int, required=True)

    pilot_parser = subparsers.add_parser("pilot", help="run the fixed six-assay pilot")
    add_shared_arguments(pilot_parser)
    pilot_parser.add_argument("--sizes", default="50,100,250")
    pilot_parser.add_argument("--include-baseline", action="store_true")

    baseline_parser = subparsers.add_parser("baseline", help="run RG-LLM global-ranking baseline")
    add_shared_arguments(baseline_parser)
    baseline_parser.add_argument("--assays", default=",".join(PILOT_ASSAYS))
    baseline_parser.add_argument("--sizes", default="50,100,250")

    smoke_parser = subparsers.add_parser("smoke", help="make one minimal live Gemini request")
    add_shared_arguments(smoke_parser)

    args = parser.parse_args()
    if args.command in {"campaign", "experiment", "evidence"}:
        schema = json.loads(args.config.read_text()).get("schema", "")
        if args.command == "evidence" and schema == "evagent-evidence-agents-v2":
            from .integrated.research.evidence_agents import command
            raise SystemExit(command(args.config, args.action))
        if schema.startswith("evagent-research-"):
            if args.command == "evidence":
                from .integrated.research.prepare import command
                raise SystemExit(command(args.config, args.action))
            from .integrated.research.campaign import command
            raise SystemExit(command(args.config, dry_run=args.dry_run))
    if args.command == "campaign":
        from .integrated.campaign import command
        raise SystemExit(command(args.config, dry_run=args.dry_run))
    if args.command == "evidence":
        from .integrated.bundles import command
        raise SystemExit(command(args.config, args.action))
    if args.command == "experiment":
        from .integrated.experiment import command
        raise SystemExit(command(args.config, dry_run=args.dry_run))
    if args.command == "memory":
        from .integrated.experiment import memory_command
        raise SystemExit(memory_command(args))
    configure_api_concurrency(args.api_concurrency)
    if args.command == "run":
        run_one(args, args.assay, args.size)
    elif args.command == "baseline":
        run_baseline(args, args.assays.split(","), [int(x) for x in args.sizes.split(",")])
        print(write_pilot_summary(Path(args.output_root).resolve()))
    elif args.command == "smoke":
        smoke(args)
    else:
        sizes = [int(x) for x in args.sizes.split(",")]
        if args.include_baseline:
            run_baseline(args, list(PILOT_ASSAYS), sizes)
        for size in sizes:
            with ThreadPoolExecutor(
                max_workers=min(args.assay_concurrency, len(PILOT_ASSAYS))
            ) as executor:
                futures = {
                    executor.submit(run_one, args, assay, size): assay for assay in PILOT_ASSAYS
                }
                for future in as_completed(futures):
                    future.result()
            print(write_pilot_summary(Path(args.output_root).resolve()))


if __name__ == "__main__":
    main()
