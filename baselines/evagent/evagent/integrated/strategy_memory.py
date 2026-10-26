"""Audited, development-derived strategy memory for held-out benchmarks.

This is deliberately separate from the agent's durable ``AGENTS.md``.  It is
for explicitly non-zero-shot experiments in which development labels may be
used to derive general reasoning heuristics, then a frozen artifact is tested
on source-disjoint assays.  The validator prevents common forms of answer
leakage from entering an inference prompt.
"""

# Adapted from ypxia/src/rnagym_deep_agent/strategy_memory.py; see provenance manifest.

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

MAX_STRATEGY_CHARS = 8_000
_CANDIDATE_ID = re.compile(r"\bM\d{1,4}\b", re.IGNORECASE)
_LONG_RNA_SEQUENCE = re.compile(r"\b[ACGUT]{20,}\b", re.IGNORECASE)


@dataclass(frozen=True)
class StrategyMemory:
    """A validated immutable strategy artifact injected into benchmark context."""

    text: str
    sha256: str
    source_path: Path


def validate_strategy_memory(text: str, *, forbidden_assays: Iterable[str]) -> str:
    """Accept only concise, assay-independent strategy text.

    This is a leakage guard, not a claim that arbitrary prose is safe.  It
    rejects direct benchmark identifiers and sequence-like content; experiment
    manifests additionally guarantee that the held-out assays were never part
    of the reflection inputs.
    """

    normalized = text.replace("\r\n", "\n").strip()
    if not normalized:
        raise ValueError("strategy memory is empty")
    if len(normalized) > MAX_STRATEGY_CHARS:
        raise ValueError(f"strategy memory exceeds {MAX_STRATEGY_CHARS} characters")
    if _CANDIDATE_ID.search(normalized):
        raise ValueError("strategy memory must not contain benchmark candidate IDs")
    if _LONG_RNA_SEQUENCE.search(normalized):
        raise ValueError("strategy memory must not contain RNA sequences")

    folded = normalized.casefold()
    mentioned = [assay for assay in forbidden_assays if assay.casefold() in folded]
    if mentioned:
        raise ValueError(
            "strategy memory must not contain benchmark assay identifiers: "
            + ", ".join(sorted(mentioned))
        )
    return normalized


def load_strategy_memory(path: Path, *, forbidden_assays: Iterable[str]) -> StrategyMemory:
    """Read, validate, and fingerprint a frozen strategy artifact."""

    resolved = path.resolve()
    text = validate_strategy_memory(
        resolved.read_text(encoding="utf-8"), forbidden_assays=forbidden_assays
    )
    return StrategyMemory(
        text=text,
        sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        source_path=resolved,
    )
