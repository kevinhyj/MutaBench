"""Anchor grouping, Plackett-Luce aggregation, and uncertainty estimates."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Any, Iterable

import numpy as np
from scipy.optimize import minimize


def _quantile_anchors(ordered_ids: list[str], count: int) -> list[str]:
    count = min(count, len(ordered_ids))
    if count == 0:
        return []
    raw = np.linspace(0, len(ordered_ids) - 1, count)
    indices: list[int] = []
    for value in raw:
        idx = int(round(float(value)))
        if idx not in indices:
            indices.append(idx)
    # Rounding can collapse positions in tiny sets; fill deterministically.
    indices.extend(i for i in range(len(ordered_ids)) if i not in indices)
    return [ordered_ids[i] for i in indices[:count]]


def make_anchor_groups(
    evidence: list[dict[str, Any]],
    *,
    group_size: int,
    anchor_count: int,
    repeats: int,
    seed: int,
) -> list[dict[str, Any]]:
    if not evidence:
        return []
    ids = [str(item["variant_id"]) for item in evidence]
    if len(ids) <= group_size:
        return [
            {"partition_id": partition, "group_id": 0, "anchor_ids": [], "variant_ids": ids}
            for partition in range(repeats)
        ]

    scored = sorted(evidence, key=lambda x: (float(x["predicted_score"]), x["variant_id"]))
    anchors = _quantile_anchors([str(x["variant_id"]) for x in scored], anchor_count)
    anchor_set = set(anchors)
    nonanchors = [item for item in ids if item not in anchor_set]
    payload_size = group_size - len(anchors)
    if payload_size < 1:
        raise ValueError("group_size must exceed the number of anchors")

    groups: list[dict[str, Any]] = []
    for partition in range(repeats):
        shuffled = list(nonanchors)
        random.Random(seed + 1009 * partition).shuffle(shuffled)
        chunks = [shuffled[i : i + payload_size] for i in range(0, len(shuffled), payload_size)]
        # Avoid a nearly empty final comparison by rebalancing the last two chunks.
        if len(chunks) > 1 and len(chunks[-1]) < max(2, payload_size // 3):
            merged = chunks[-2] + chunks[-1]
            split = math.ceil(len(merged) / 2)
            chunks[-2:] = [merged[:split], merged[split:]]
        for group_id, chunk in enumerate(chunks):
            groups.append(
                {
                    "partition_id": partition,
                    "group_id": group_id,
                    "anchor_ids": anchors,
                    "variant_ids": anchors + chunk,
                }
            )
    return groups


def validate_group_coverage(groups: list[dict[str, Any]], ids: Iterable[str], repeats: int) -> None:
    expected = set(ids)
    for partition in range(repeats):
        present: set[str] = set()
        partition_groups = [g for g in groups if int(g["partition_id"]) == partition]
        if not partition_groups:
            raise ValueError(f"partition {partition} has no groups")
        for group in partition_groups:
            members = list(group["variant_ids"])
            if len(members) != len(set(members)):
                raise ValueError("a local group contains duplicate variants")
            present.update(members)
        if present != expected:
            raise ValueError(f"partition {partition} does not cover the full variant set")


def _pl_objective(
    scores: np.ndarray,
    orders: list[tuple[np.ndarray, float]],
    l2: float,
) -> tuple[float, np.ndarray]:
    loss = 0.5 * l2 * float(np.dot(scores, scores))
    grad = l2 * scores.copy()
    for order, weight in orders:
        local = scores[order]
        # Stable suffix denominators; vectorized cumulative derivative of PL.
        normalizers = np.logaddexp.accumulate(local[::-1])[::-1]
        loss += weight * float(np.sum(normalizers[:-1] - local[:-1]))
        log_prefix = np.logaddexp.accumulate(-normalizers[:-1])
        terms = np.exp(local + np.r_[log_prefix, log_prefix[-1]])
        terms[:-1] -= 1.0
        np.add.at(grad, order, weight * terms)
    return loss, grad


def aggregate_rankings(
    variant_ids: list[str],
    rankings: list[dict[str, Any]],
    *,
    initial_scores: dict[str, float] | None = None,
    l2: float = 1e-3,
) -> dict[str, float]:
    if not variant_ids:
        return {}
    index = {variant_id: i for i, variant_id in enumerate(variant_ids)}
    orders: list[tuple[np.ndarray, float]] = []
    for item in rankings:
        ranking = list(item["ranking"])
        if len(ranking) < 2 or any(variant_id not in index for variant_id in ranking):
            continue
        orders.append(
            (np.array([index[x] for x in ranking], dtype=int), float(item.get("weight", 1.0)))
        )

    x0 = np.array(
        [float((initial_scores or {}).get(variant_id, 0.0)) for variant_id in variant_ids],
        dtype=float,
    )
    x0 -= x0.mean()
    if not orders:
        fitted = x0
    else:
        result = minimize(
            lambda x: _pl_objective(x, orders, l2),
            x0,
            method="L-BFGS-B",
            jac=True,
            options={"maxiter": 500, "ftol": 1e-10},
        )
        if not result.success and not np.isfinite(result.fun):
            raise RuntimeError(f"Plackett-Luce optimization failed: {result.message}")
        fitted = np.asarray(result.x, dtype=float)

    fitted -= fitted.mean()
    std = float(fitted.std())
    if std > 1e-12:
        fitted /= std
    return {variant_id: float(fitted[i]) for i, variant_id in enumerate(variant_ids)}


def estimate_uncertainty(
    variant_ids: list[str],
    rankings: list[dict[str, Any]],
    *,
    initial_scores: dict[str, float],
    repeats: int,
    l2: float,
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    per_budget: dict[str, dict[str, float]] = {}
    partition_scores: list[dict[str, float]] = []
    available = sorted({int(x["partition_id"]) for x in rankings if int(x["partition_id"]) >= 0})
    for partition in available:
        subset = [x for x in rankings if int(x["partition_id"]) == partition]
        partition_scores.append(
            aggregate_rankings(variant_ids, subset, initial_scores=initial_scores, l2=l2)
        )

    for budget in range(1, min(repeats, len(partition_scores)) + 1):
        per_budget[str(budget)] = aggregate_rankings(
            variant_ids,
            [x for x in rankings if 0 <= int(x["partition_id"]) < budget],
            initial_scores=initial_scores,
            l2=l2,
        )

    rank_samples: dict[str, list[float]] = defaultdict(list)
    for scores in partition_scores:
        ordered = sorted(variant_ids, key=lambda x: (-scores[x], x))
        positions = {variant_id: i for i, variant_id in enumerate(ordered)}
        for variant_id in variant_ids:
            rank_samples[variant_id].append(float(positions[variant_id]))

    scale = max(1.0, float(len(variant_ids) - 1))
    uncertainty: dict[str, float] = {}
    for variant_id in variant_ids:
        samples = rank_samples.get(variant_id, [])
        uncertainty[variant_id] = (
            min(1.0, float(np.std(samples)) / scale) if len(samples) > 1 else 0.0
        )
    return uncertainty, per_budget


def verification_candidates(
    variant_ids: list[str],
    scores: dict[str, float],
    uncertainty: dict[str, float],
    evidence_confidence: dict[str, float],
    count: int,
    max_group_size: int,
) -> list[str]:
    ordered = sorted(variant_ids, key=lambda x: (-scores[x], x))
    priority = sorted(
        variant_ids,
        key=lambda x: (
            -(uncertainty.get(x, 0.0) + 0.25 * (1 - evidence_confidence.get(x, 0.0))),
            x,
        ),
    )
    selected: set[str] = set(priority[: min(count, len(priority))])
    positions = {variant_id: i for i, variant_id in enumerate(ordered)}
    for variant_id in list(selected):
        pos = positions[variant_id]
        if pos > 0:
            selected.add(ordered[pos - 1])
        if pos + 1 < len(ordered):
            selected.add(ordered[pos + 1])
    return [x for x in ordered if x in selected][:max_group_size]
