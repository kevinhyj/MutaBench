"""Leak-aware learning memory for development-calibrated agents.

Raw benchmark histories are deliberately not model-facing memory.  A separate
reflection stage converts one development episode into a small, de-identified
lesson.  Lessons are consolidated into knowledge cards, validated on a
selection split, and only active cards may be retrieved for inference.
"""

# Adapted from ypxia/src/rnagym_deep_agent/learning_memory.py; see provenance manifest.

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass, replace
from typing import Any, Iterable, Sequence

from .strategy_memory import validate_strategy_memory

LESSON_OUTCOMES = {"improved", "degraded", "neutral", "format_failure"}
CARD_KINDS = {"procedural", "semantic", "failure_guard"}
CARD_STATUSES = {"candidate", "validated", "active", "rejected", "retired"}
_TAG_RE = re.compile(r"^[a-z0-9_]+(?::[a-z0-9_]+)?$")
_CONTEXT_KEYS = {
    "task_family",
    "rna_type",
    "phenotype",
    "length",
    "mutation_count",
    "structure",
    "output_mode",
}


def _slug(value: str) -> str:
    text = re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")
    return text or "unknown"


def _hash(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class TaskContext:
    """Coarse task features that are safe to use for memory retrieval."""

    task_family: str = "rna_fitness"
    rna_type: str = "unknown"
    phenotype: str = "unknown"
    length_bucket: str = "unknown"
    mutation_count_bucket: str = "unknown"
    structure_available: bool = False
    output_mode: str = "ranking"

    def tags(self) -> tuple[str, ...]:
        return (
            "all",
            f"task_family:{_slug(self.task_family)}",
            f"rna_type:{_slug(self.rna_type)}",
            f"phenotype:{_slug(self.phenotype)}",
            f"length:{_slug(self.length_bucket)}",
            f"mutation_count:{_slug(self.mutation_count_bucket)}",
            f"structure:{'available' if self.structure_available else 'unavailable'}",
            f"output_mode:{_slug(self.output_mode)}",
        )


@dataclass(frozen=True)
class EpisodeLesson:
    """A de-identified reflection produced from one development episode."""

    episode_id: str
    split: str
    source_group_hash: str
    artifact_sha256: str
    context: TaskContext
    outcome: str
    strategy_used: str
    what_worked: str
    what_failed: str
    condition: str
    proposed_rule: str
    contraindication: str
    tags: tuple[str, ...]
    confidence: float


@dataclass(frozen=True)
class KnowledgeCard:
    """A versioned, retrievable unit of learned procedural knowledge."""

    card_id: str
    version: int
    kind: str
    status: str
    condition: str
    guidance: str
    contraindication: str
    tags: tuple[str, ...]
    confidence: float
    support_count: int
    source_episode_ids: tuple[str, ...]


@dataclass(frozen=True)
class PromotionEvidence:
    """Selection-set evidence used to move a card through its lifecycle."""

    paired_assays: int
    mean_delta: float
    ci_low: float | None
    ci_high: float | None
    pass_rate_delta: float


def build_episode_reflection_prompt(
    *,
    history_summary: str,
    context: TaskContext,
    outcome: str,
    artifact_sha256: str,
    max_history_chars: int = 20_000,
) -> str:
    """Create the isolated reflection request for one development episode.

    The request may inspect a bounded historical summary, but its output must
    pass :func:`parse_episode_reflection` before entering learning memory.
    """

    if outcome not in LESSON_OUTCOMES:
        raise ValueError(f"unsupported lesson outcome: {outcome}")
    bounded = history_summary.strip()[:max_history_chars]
    context_json = json.dumps(asdict(context), ensure_ascii=False, sort_keys=True)
    return "\n".join(
        [
            "You are an offline memory reflection worker.",
            "Extract one general lesson from this DEVELOPMENT episode.",
            "Do not reproduce assay names, candidate IDs, sequences, rankings, labels, or numeric ground truth.",
            "State when the lesson applies, when it should not be used, and keep claims proportional to the evidence.",
            f"Task context: {context_json}",
            f"Outcome category: {outcome}",
            f"Source artifact SHA256: {artifact_sha256}",
            "Historical reasoning summary:",
            bounded,
            "Output ONLY one JSON object with keys: strategy_used, what_worked, what_failed, condition, proposed_rule, contraindication, tags, confidence.",
            "tags must be a JSON list of lowercase scope tags such as all, rna_type:ribozyme, structure:available, or output_mode:ranking.",
            "confidence must be between 0 and 1.",
        ]
    )


def _json_objects(text: str) -> list[dict[str, Any]]:
    decoder = json.JSONDecoder()
    values: list[dict[str, Any]] = []
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, _end = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            values.append(value)
    return values


def _required_text(value: Any, field: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"reflection field is empty: {field}")
    return text


def _validated_tags(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError("reflection tags must be a non-empty JSON list")
    tags = tuple(dict.fromkeys(str(item).strip().casefold() for item in value))
    if any(not _TAG_RE.fullmatch(tag) for tag in tags):
        raise ValueError("reflection contains an invalid scope tag")
    return tags


def parse_episode_reflection(
    text: str,
    *,
    context: TaskContext,
    split: str,
    source_group: str,
    artifact_sha256: str,
    outcome: str,
    forbidden_assays: Iterable[str],
) -> EpisodeLesson:
    """Validate reflection output and turn it into a safe episode lesson."""

    if split != "development":
        raise ValueError("learning memory may only be derived from development episodes")
    if outcome not in LESSON_OUTCOMES:
        raise ValueError(f"unsupported lesson outcome: {outcome}")
    objects = _json_objects(text)
    required = {
        "strategy_used",
        "what_worked",
        "what_failed",
        "condition",
        "proposed_rule",
        "contraindication",
        "tags",
        "confidence",
    }
    candidates = [value for value in objects if required <= set(value)]
    if not candidates:
        raise ValueError("no complete reflection JSON object found")
    value = candidates[-1]
    fields = {
        field: _required_text(value.get(field), field)
        for field in required - {"tags", "confidence"}
    }
    tags = _validated_tags(value["tags"])
    confidence = float(value["confidence"])
    if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        raise ValueError("reflection confidence must be finite and in [0, 1]")
    # This gate blocks the common direct leakage paths before any text can be
    # consolidated or retrieved into a held-out prompt.
    validate_strategy_memory(
        "\n".join((*fields.values(), *tags)), forbidden_assays=forbidden_assays
    )
    identity = {
        "artifact_sha256": artifact_sha256,
        "context": asdict(context),
        "outcome": outcome,
        "rule": fields["proposed_rule"],
    }
    return EpisodeLesson(
        episode_id=f"exp_{_hash(identity)[:16]}",
        split=split,
        source_group_hash=_hash(source_group)[:16],
        artifact_sha256=artifact_sha256,
        context=context,
        outcome=outcome,
        strategy_used=fields["strategy_used"],
        what_worked=fields["what_worked"],
        what_failed=fields["what_failed"],
        condition=fields["condition"],
        proposed_rule=fields["proposed_rule"],
        contraindication=fields["contraindication"],
        tags=tags,
        confidence=confidence,
    )


def consolidate_lessons(
    lessons: Sequence[EpisodeLesson], *, kind: str = "procedural"
) -> list[KnowledgeCard]:
    """Deterministically merge duplicate lesson rules into candidate cards.

    Semantic rewriting belongs to the isolated reflection worker.  This step is
    intentionally mechanical so the same accepted lessons always produce the
    same registry.
    """

    if kind not in CARD_KINDS:
        raise ValueError(f"unsupported card kind: {kind}")
    grouped: dict[tuple[str, tuple[str, ...]], list[EpisodeLesson]] = {}
    for lesson in lessons:
        if lesson.split != "development":
            raise ValueError("non-development lesson cannot be consolidated")
        rule_key = " ".join(lesson.proposed_rule.casefold().split())
        # Scope is part of identity: an otherwise identical rule learned for a
        # ribozyme must not silently absorb evidence from a coding-RNA task.
        key = (rule_key, tuple(sorted(lesson.tags)))
        grouped.setdefault(key, []).append(lesson)
    cards: list[KnowledgeCard] = []
    for (rule_key, scope_key), group in sorted(grouped.items()):
        source_ids = tuple(sorted({lesson.episode_id for lesson in group}))
        tags = scope_key
        contraindications = tuple(
            dict.fromkeys(lesson.contraindication for lesson in group)
        )
        confidence = sum(lesson.confidence for lesson in group) / len(group)
        identity = {"kind": kind, "rule": rule_key, "tags": tags}
        cards.append(
            KnowledgeCard(
                card_id=f"mem_{_hash(identity)[:16]}",
                version=1,
                kind=kind,
                status="candidate",
                condition=group[0].condition,
                guidance=group[0].proposed_rule,
                contraindication="; ".join(contraindications),
                tags=tags,
                confidence=confidence,
                support_count=len(source_ids),
                source_episode_ids=source_ids,
            )
        )
    return cards


def advance_card(
    card: KnowledgeCard,
    evidence: PromotionEvidence,
    *,
    min_paired_assays: int = 5,
    require_positive_ci_for_active: bool = True,
) -> KnowledgeCard:
    """Advance a candidate using explicit selection-set evidence.

    Accuracy gain must be positive and validity must not fall.  By default an
    active card additionally requires a positive lower confidence bound;
    otherwise it remains validated but not model-facing.
    """

    if card.status not in CARD_STATUSES:
        raise ValueError(f"invalid card status: {card.status}")
    if (
        isinstance(evidence.paired_assays, bool)
        or not isinstance(evidence.paired_assays, int)
        or evidence.paired_assays < 0
        or isinstance(min_paired_assays, bool)
        or not isinstance(min_paired_assays, int)
        or min_paired_assays < 1
    ):
        raise ValueError("promotion assay counts must be nonnegative integers with a positive minimum")
    if not all(math.isfinite(value) for value in (evidence.mean_delta, evidence.pass_rate_delta)):
        raise ValueError("promotion deltas must be finite")
    if (evidence.ci_low is None) != (evidence.ci_high is None):
        raise ValueError("promotion confidence bounds must both be present or both absent")
    if evidence.ci_low is not None and evidence.ci_high is not None:
        if not all(math.isfinite(value) for value in (evidence.ci_low, evidence.ci_high)):
            raise ValueError("promotion confidence bounds must be finite")
        if evidence.ci_low > evidence.ci_high:
            raise ValueError("promotion confidence bounds are reversed")
    if card.status in {"rejected", "retired"}:
        return card
    qualifies = (
        evidence.paired_assays >= min_paired_assays
        and evidence.mean_delta > 0
        and evidence.pass_rate_delta >= 0
    )
    if not qualifies:
        return replace(card, status="rejected", version=card.version + 1)
    positive_ci = evidence.ci_low is not None and evidence.ci_low > 0
    status = "active" if positive_ci or not require_positive_ci_for_active else "validated"
    return replace(card, status=status, version=card.version + 1)


def _tag_map(tags: Iterable[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for tag in tags:
        if ":" in tag:
            key, value = tag.split(":", 1)
            result[key] = value
    return result


def retrieve_cards(
    cards: Sequence[KnowledgeCard],
    context: TaskContext,
    *,
    max_cards: int = 5,
) -> list[KnowledgeCard]:
    """Return a deterministic, context-matched set of active cards."""

    if max_cards < 1:
        raise ValueError("max_cards must be positive")
    context_tags = set(context.tags())
    context_map = _tag_map(context_tags)
    ranked: list[tuple[float, KnowledgeCard]] = []
    for card in cards:
        if card.status != "active":
            continue
        card_map = _tag_map(card.tags)
        if any(
            key in _CONTEXT_KEYS
            and key in context_map
            and context_map[key] != value
            for key, value in card_map.items()
        ):
            continue
        matches = len((set(card.tags) - {"all"}) & context_tags)
        score = 3.0 * matches + card.confidence + math.log1p(card.support_count)
        ranked.append((score, card))
    ranked.sort(key=lambda item: (-item[0], item[1].card_id, -item[1].version))
    return [card for _score, card in ranked[:max_cards]]


def compile_strategy_text(
    cards: Sequence[KnowledgeCard],
    *,
    forbidden_assays: Iterable[str],
    max_chars: int = 4_000,
) -> str:
    """Compile retrieved cards into the existing frozen-strategy interface."""

    if not cards:
        raise ValueError("no active memory cards were selected")
    lines = ["# Retrieved learning memory", ""]
    for card in cards:
        if card.status != "active":
            raise ValueError("only active cards may enter an inference strategy")
        block = (
            f"- When {card.condition}: {card.guidance} "
            f"Countercheck: {card.contraindication}"
        )
        if len("\n".join((*lines, block))) > max_chars:
            break
        lines.append(block)
    text = "\n".join(lines).strip()
    if len(lines) == 2:
        raise ValueError("memory token budget excluded every selected card")
    return validate_strategy_memory(text, forbidden_assays=forbidden_assays)


def registry_payload(cards: Sequence[KnowledgeCard]) -> dict[str, Any]:
    """Return a canonical, auditable registry representation."""

    rows = [asdict(card) for card in sorted(cards, key=lambda item: (item.card_id, item.version))]
    payload: dict[str, Any] = {"schema_version": 1, "cards": rows}
    payload["registry_sha256"] = _hash(payload)
    return payload
