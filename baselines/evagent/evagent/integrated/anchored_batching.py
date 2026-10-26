"""Leak-free anchor-consistent batching for large RNAgym ranking prompts.

This module adapts the useful parts of EVAgent's ACR protocol to the existing
Deep Agents benchmark harness.  A fixed small set of candidates occurs in
every local group; local rankings are then joined with a Plackett--Luce fit.
The model never sees which candidates are anchors and never receives labels,
previous rankings, or experimental scores.
"""

from __future__ import annotations

import hashlib
import math
import random
import re
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from scipy.optimize import minimize

from .ranking import RankingCase, ensure_label_free_case

ACR_PROTOCOL = "ypxia-acr-fullseq-tail-rebalanced-v1"


@dataclass(frozen=True)
class AnchorBatch:
    """One local ranking request in an anchor-connected comparison graph."""

    group_id: int
    variant_ids: list[str]
    anchor_ids: list[str]


def _seed(*parts: object) -> int:
    encoded = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big")


def make_anchor_batches(
    assay: str,
    variant_ids: Sequence[str],
    *,
    group_size: int = 25,
    anchor_count: int = 4,
    seed: int = 1,
) -> list[AnchorBatch]:
    """Create deterministic, connected local groups.

    This follows EVAgent's ACR invariants: fixed shared anchors, bounded local
    group size, deterministic randomized presentation, and full coverage.  A
    small last payload is rebalanced so that all local comparisons remain
    substantive; it does not alter which candidates are evaluated.
    """

    ids = list(variant_ids)
    if len(ids) != len(set(ids)) or not ids:
        raise ValueError("variant IDs must be non-empty and unique")
    if group_size < 2:
        raise ValueError("group_size must be at least two")
    if len(ids) <= group_size:
        return [AnchorBatch(group_id=0, variant_ids=ids, anchor_ids=[])]
    if not 1 <= anchor_count < group_size:
        raise ValueError("anchor_count must be in [1, group_size)")
    if anchor_count >= len(ids):
        raise ValueError("anchor_count must be smaller than the candidate count")

    anchors = random.Random(_seed(assay, seed, "anchors")).sample(ids, anchor_count)
    anchor_set = set(anchors)
    payload = [variant_id for variant_id in ids if variant_id not in anchor_set]
    payload_size = group_size - anchor_count
    chunks = [payload[offset : offset + payload_size] for offset in range(0, len(payload), payload_size)]

    # EVAgent's generic grouping routine avoids a very small final comparison
    # group.  For N=50, B=25, A=4 this turns 21/21/4 into 21/13/12 payloads.
    if len(chunks) > 1 and len(chunks[-1]) < max(2, payload_size // 3):
        merged = chunks[-2] + chunks[-1]
        split = math.ceil(len(merged) / 2)
        chunks[-2:] = [merged[:split], merged[split:]]

    batches: list[AnchorBatch] = []
    for group_id, chunk in enumerate(chunks):
        members = anchors + chunk
        random.Random(_seed(assay, seed, "group", group_id, *members)).shuffle(members)
        batches.append(
            AnchorBatch(group_id=group_id, variant_ids=members, anchor_ids=list(anchors))
        )
    validate_anchor_batches(batches, ids)
    return batches


def validate_anchor_batches(batches: Sequence[AnchorBatch], variant_ids: Iterable[str]) -> None:
    """Check the overlap graph and exact candidate coverage before inference."""

    expected = set(variant_ids)
    if not batches:
        raise ValueError("no batches were created")
    seen: set[str] = set()
    common_anchors = set(batches[0].anchor_ids)
    for batch in batches:
        members = batch.variant_ids
        if len(members) != len(set(members)):
            raise ValueError("a local batch contains duplicate IDs")
        if not set(members) <= expected:
            raise ValueError("a local batch contains an unknown ID")
        if batch.anchor_ids and not set(batch.anchor_ids) <= set(members):
            raise ValueError("a local batch omits an anchor")
        if set(batch.anchor_ids) != common_anchors:
            raise ValueError("all local batches must share the same anchors")
        seen.update(members)
    if seen != expected:
        raise ValueError("local batches do not cover every candidate")


def _split_prompt_candidates(case: RankingCase) -> tuple[list[str], dict[str, str]]:
    """Extract candidate lines from the frozen public RNAgym prompt."""

    ensure_label_free_case(case)
    lines = case.user_prompt.splitlines()
    try:
        heading = next(
            index
            for index, line in enumerate(lines)
            if "candidate mutants to rank" in line.lower()
        )
        instruction = next(
            index
            for index, line in enumerate(lines[heading + 1 :], start=heading + 1)
            if line.startswith("Rank all ")
        )
    except StopIteration as error:
        raise ValueError(f"Unsupported RNAgym prompt layout for {case.assay}") from error

    rows: dict[str, str] = {}
    extracted_ids: list[str] = []
    for line in lines[heading + 1 : instruction]:
        match = re.match(r"^(M\d+)\s+\[.*\]:\s+.+$", line)
        if match:
            rows[match.group(1)] = line
            extracted_ids.append(match.group(1))
    if set(rows) != set(case.ids) or len(extracted_ids) != len(case.ids):
        raise ValueError(f"Candidate extraction mismatch for {case.assay}")
    return lines[:heading], rows


def build_local_ranking_prompt(case: RankingCase, batch: AnchorBatch) -> str:
    """Render a valid local ranking task without revealing anchor membership."""

    preamble, rows = _split_prompt_candidates(case)
    selected = "\n".join(rows[variant_id] for variant_id in batch.variant_ids)
    user = "\n".join(preamble) + "\n\n"
    user += (
        f"**{len(batch.variant_ids)} candidate mutants to rank** "
        "(id, mutation(s) relative to WT, full sequence):\n"
    )
    user += selected + "\n\n"
    user += (
        f"Rank all {len(batch.variant_ids)} mutants from the one you predict has the MOST "
        "favorable effect on the assayed RNA property (highest fitness) to the LEAST. "
        "Reason through the ordering, then on the last line output ONLY the JSON object:\n"
        '{"ranking": ["M03", "M27", ... all '
        f"{len(batch.variant_ids)} ids, best to worst]}}"
    )
    return (
        "Follow this benchmark instruction exactly. Do not use tools to inspect labels or "
        "previous answers.\n\n"
        f"{case.system_prompt}\n\n{user}"
    )


def _pl_objective(
    scores: np.ndarray, orders: Sequence[np.ndarray], l2: float
) -> tuple[float, np.ndarray]:
    """Negative Plackett--Luce likelihood and gradient for full local rankings."""

    loss = 0.5 * l2 * float(np.dot(scores, scores))
    gradient = l2 * scores.copy()
    for order in orders:
        local = scores[order]
        # At position i, the chosen item must beat every still-available item.
        for position in range(len(order) - 1):
            tail = local[position:]
            maximum = float(np.max(tail))
            probabilities = np.exp(tail - maximum)
            probabilities /= probabilities.sum()
            loss += maximum + math.log(float(np.exp(tail - maximum).sum())) - float(
                local[position]
            )
            gradient[order[position:]] += probabilities
            gradient[order[position]] -= 1.0
    return loss, gradient


def aggregate_anchor_rankings(
    variant_ids: Sequence[str], rankings: Sequence[Sequence[str]], *, l2: float = 1e-3
) -> dict[str, float]:
    """Fit one global latent score per candidate from overlapping local orders."""

    ids = list(variant_ids)
    if len(ids) != len(set(ids)) or not ids:
        raise ValueError("variant IDs must be non-empty and unique")
    index = {variant_id: offset for offset, variant_id in enumerate(ids)}
    orders: list[np.ndarray] = []
    graph: dict[str, set[str]] = {variant_id: set() for variant_id in ids}
    for ranking in rankings:
        order = list(ranking)
        if len(order) < 2 or len(order) != len(set(order)) or not set(order) <= set(index):
            raise ValueError("invalid local ranking")
        orders.append(np.asarray([index[variant_id] for variant_id in order], dtype=int))
        for variant_id in order[1:]:
            graph[order[0]].add(variant_id)
            graph[variant_id].add(order[0])
    if not orders:
        raise ValueError("at least one local ranking is required")
    connected = {ids[0]}
    frontier = [ids[0]]
    while frontier:
        current = frontier.pop()
        for neighbor in graph[current] - connected:
            connected.add(neighbor)
            frontier.append(neighbor)
    if connected != set(ids):
        raise ValueError("local ranking graph is disconnected")

    result = minimize(
        lambda values: _pl_objective(values, orders, l2),
        np.zeros(len(ids), dtype=float),
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": 1000, "ftol": 1e-12, "gtol": 1e-7},
    )
    if not result.success or not np.isfinite(result.x).all():
        raise RuntimeError(f"Plackett--Luce aggregation failed: {result.message}")
    scores = np.asarray(result.x, dtype=float)
    scores -= scores.mean()
    scale = float(scores.std())
    if scale > 1e-12:
        scores /= scale
    return {variant_id: float(scores[offset]) for offset, variant_id in enumerate(ids)}


def ranking_from_scores(scores: dict[str, float]) -> list[str]:
    """Produce a deterministic descending global order without arbitrary repair."""

    return sorted(scores, key=lambda variant_id: (-scores[variant_id], variant_id))
