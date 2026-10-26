"""Backend-independent execution of migrated, versioned ranking protocols.

The gateway is injected as a callback; no model SDK, label loader, persistent
memory database, or historical-result fallback is imported here.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Callable

from .anchored_batching import (
    ACR_PROTOCOL,
    aggregate_anchor_rankings,
    build_local_ranking_prompt,
    make_anchor_batches,
    ranking_from_scores,
)
from .chunked_scoring import (
    ScoringResponseValidator,
    build_scoring_prompt,
    make_scoring_chunks,
    ranking_from_scoring_scores,
)
from .ranking import (
    RankingCase,
    RankingResponseValidator,
    ensure_label_free_case,
    ensure_label_free_context,
)
from .self_check import SelfCheckPolicy, _sum_usage, run_self_check

GLOBAL_PROTOCOL = "integrated-global-strict-json-v1"


@dataclass(frozen=True)
class ProtocolTask:
    system: str
    user: str
    expected_ids: list[str]
    metadata: dict[str, Any]


def _append_context(user: str, assay_memory: dict[str, Any], context_text: str) -> str:
    if assay_memory:
        user += "\n\nFrozen assay context (generated without candidate labels):\n"
        user += json.dumps(assay_memory, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    if context_text:
        user += "\n\nFrozen additional context:\n" + context_text
    return user


def build_protocol_tasks(
    method: str,
    *,
    case: RankingCase,
    assay_memory: dict[str, Any],
    context_text: str = "",
    prompt_version: str = "v2",
    group_size: int = 25,
    anchor_count: int = 4,
    seed: int = 1,
) -> list[ProtocolTask]:
    """Prepare every initial request for dry-run, fingerprinting, or execution."""
    ensure_label_free_case(case)
    ensure_label_free_context(assay_memory)
    if method not in {"global", "cr", "acr"}:
        raise ValueError(f"Unsupported protocol method: {method}")
    if prompt_version not in {"v1", "v2", "v3"}:
        raise ValueError("prompt_version must be v1, v2 or v3")
    tasks: list[ProtocolTask] = []
    system = case.system_prompt
    if method == "cr" and prompt_version in {"v2", "v3"}:
        system = re.sub(
            r"Rank the mutants by their predicted effect on the assayed RNA property, then output the ranking in the requested JSON format\.?",
            "Estimate an independent continuous fitness score for every supplied mutant.",
            system,
            flags=re.IGNORECASE,
        )

    def add(user: str, ids: list[str], protocol: str, index: int, **extra: Any) -> None:
        metadata = {
            "method": method,
            "protocol": protocol,
            "unit_index": index,
            "assay": case.assay,
            "expected_ids": list(ids),
            "group_size": group_size,
            "anchor_count": anchor_count if method == "acr" else 0,
            "seed": seed,
            **extra,
        }
        metadata["prompt_sha256"] = hashlib.sha256((system + "\n\n" + user).encode()).hexdigest()
        tasks.append(ProtocolTask(system, user, list(ids), metadata))

    if method == "cr":
        # Group size belongs in the protocol name; b25 source names are not
        # reused for a different chunk size or the stricter migrated validator.
        protocol = f"ypxia-cr-fullseq-b{group_size}-{prompt_version}-strict-v1"
        for chunk in make_scoring_chunks(case, chunk_size=group_size):
            user = build_scoring_prompt(case, chunk, assay_memory=assay_memory, prompt_variant=prompt_version)
            selected_context = context_text(chunk.ids) if callable(context_text) else context_text
            add(_append_context(user, {}, selected_context), chunk.ids, protocol, chunk.chunk_index, unit_kind="chunk", prompt_version=prompt_version)
    elif method == "acr":
        for batch in make_anchor_batches(case.assay, case.ids, group_size=group_size, anchor_count=anchor_count, seed=seed):
            user = _append_context(build_local_ranking_prompt(case, batch), assay_memory, context_text)
            add(user, batch.variant_ids, ACR_PROTOCOL, batch.group_id, unit_kind="group", anchor_ids=batch.anchor_ids)
    else:
        user = _append_context(case.agent_prompt, assay_memory, context_text)
        add(user, case.ids, GLOBAL_PROTOCOL, 0, unit_kind="global")
    return tasks


def execute_protocol(
    method: str,
    *,
    case: RankingCase,
    generate: Callable[..., dict[str, Any]],
    assay_memory: dict[str, Any],
    context_text: str = "",
    max_repairs: int = 0,
    prompt_version: str = "v2",
    group_size: int = 25,
    anchor_count: int = 4,
    seed: int = 1,
) -> dict[str, Any]:
    """Execute a complete cell through the caller's accounting gateway.

    Semantic-invalid groups are retained and invalidate the whole cell. A
    provider failure prevents additional calls. No exception from ``generate``
    is caught, so an ambiguous-response ExecutionPaused cannot become a repair.
    """
    policy = SelfCheckPolicy(max_repairs=max_repairs)
    tasks = build_protocol_tasks(
        method, case=case, assay_memory=assay_memory, context_text=context_text,
        prompt_version=prompt_version, group_size=group_size,
        anchor_count=anchor_count, seed=seed,
    )
    units: list[dict[str, Any]] = []
    attempts: list[dict[str, Any]] = []
    provider_failure: str | None = None
    for task in tasks:
        if provider_failure:
            units.append({
                "status": "not_run", "reason": "previous_provider_failure", "value": None,
                "metadata": task.metadata, "attempts": [], "repair_attempts_used": 0,
                "usage_summary": _sum_usage([]),
            })
            continue
        validator = ScoringResponseValidator(task.expected_ids) if method == "cr" else RankingResponseValidator(task.expected_ids)

        def generate_attempt(user: str, attempt_index: int) -> dict[str, Any]:
            metadata = {
                **task.metadata,
                "attempt_index": attempt_index,
                "attempt_kind": "initial" if attempt_index == 0 else "repair",
                "prompt_sha256": hashlib.sha256((task.system + "\n\n" + user).encode()).hexdigest(),
            }
            generated = dict(generate(
                system=task.system, user=user,
                run_name=f"{case.assay}-{method}-{task.metadata['unit_index']:04d}-attempt-{attempt_index}",
                metadata=metadata,
            ))
            generated["protocol_metadata"] = metadata
            return generated

        checked = run_self_check(
            original_prompt=task.user, generate=generate_attempt,
            validator=validator, policy=policy,
        )
        checked["metadata"] = task.metadata
        units.append(checked)
        attempts.extend(checked["attempts"])
        if checked["status"] not in {"ok", "invalid_output"}:
            provider_failure = checked["status"]

    complete = all(unit["status"] == "ok" for unit in units)
    status = "ok" if complete else (provider_failure or "invalid_output")
    ranking: list[str] | None = None
    scores: dict[str, float] | None = None
    aggregation_error: str | None = None
    if complete:
        if method == "cr":
            scores = {variant_id: score for unit in units for variant_id, score in unit["value"].items()}
            # Absolute scores are merged verbatim, preserving cross-chunk ties.
            ranking = ranking_from_scoring_scores(scores)
        elif method == "global":
            ranking = list(units[0]["value"])
        elif len(case.ids) == 1:
            ranking, scores = list(case.ids), {case.ids[0]: 0.0}
        else:
            try:
                scores = aggregate_anchor_rankings(case.ids, [unit["value"] for unit in units])
                ranking = ranking_from_scores(scores)
            except (ValueError, RuntimeError) as error:
                status = "aggregation_error"
                aggregation_error = str(error)

    return {
        "status": status,
        "ranking": ranking,
        "scores": scores,
        "attempts": attempts,
        "units": units,
        "groups": units,
        "usage_summary": _sum_usage(attempts),
        "repair_attempts_used": sum(unit["repair_attempts_used"] for unit in units),
        "protocol": tasks[0].metadata["protocol"],
        "method": method,
        "assay": case.assay,
        "n_candidates": len(case.ids),
        "valid_units": sum(unit["status"] == "ok" for unit in units),
        "total_units": len(units),
        "aggregation_error": aggregation_error,
    }
