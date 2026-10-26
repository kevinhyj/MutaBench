"""CR-style fixed-size scoring chunks for large RNAgym libraries.

Each request predicts an absolute score for 25 candidates.  Consecutive,
non-overlapping chunks can therefore be merged directly and reused across the
frozen N=100/250/500 prefixes.  Experimental scores are never rendered into a
model prompt.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from .ranking import (
    RankingCase,
    ensure_label_free_case,
    ensure_label_free_context,
    final_json_object,
)
from .self_check import ValidationIssue, ValidationReport

CR_PROTOCOL = "agent-cr-fullseq-b25-repro-v1"
CR_PROTOCOL_V2 = "agent-cr-fullseq-b25-repro-v2"
_ROW_RE = re.compile(r"^(M\d+)\s+\[(.*)\]:\s+(.+)$")


@dataclass(frozen=True)
class ScoringCandidate:
    variant_id: str
    mutation: str
    sequence: str


@dataclass(frozen=True)
class ScoringChunk:
    chunk_index: int
    candidates: list[ScoringCandidate]

    @property
    def ids(self) -> list[str]:
        return [candidate.variant_id for candidate in self.candidates]


def _split_case(case: RankingCase) -> tuple[list[str], list[ScoringCandidate]]:
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

    candidates: list[ScoringCandidate] = []
    for line in lines[heading + 1 : instruction]:
        match = _ROW_RE.match(line)
        if match:
            candidates.append(
                ScoringCandidate(
                    variant_id=match.group(1),
                    mutation=match.group(2),
                    sequence=match.group(3),
                )
            )
    if [candidate.variant_id for candidate in candidates] != case.ids:
        raise ValueError(f"Candidate extraction mismatch for {case.assay}")
    return lines[:heading], candidates


def make_scoring_chunks(case: RankingCase, *, chunk_size: int = 25) -> list[ScoringChunk]:
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    _preamble, candidates = _split_case(case)
    return [
        ScoringChunk(chunk_index=offset // chunk_size, candidates=candidates[offset : offset + chunk_size])
        for offset in range(0, len(candidates), chunk_size)
    ]


def load_frozen_assay_memory(assays_root: Path, assay: str) -> dict[str, Any]:
    path = assays_root / assay / "memory.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("status") != "complete" or not isinstance(record.get("data"), dict):
        # Four historical analyst calls are invalid. The benchmark prompt
        # already contains assay metadata and WT, so run them without an
        # invented summary and record that fallback in the chunk artifact.
        return {}
    # The source protocol generated these summaries without candidate labels.
    # Reject obvious accidental leakage before inserting one into a new prompt.
    serialized = json.dumps(record["data"], ensure_ascii=False)
    if re.search(r'"(?:score|ranking|fitness_values?)"\s*:', serialized, flags=re.I):
        raise ValueError(f"Frozen assay memory contains a prohibited result field: {path}")
    return record["data"]


def build_scoring_prompt(
    case: RankingCase,
    chunk: ScoringChunk,
    *,
    assay_memory: dict[str, Any],
    prompt_variant: str = "v1",
) -> str:
    """Render one score-calibrated B=25 request independent of total library N.

    v1 is frozen for the completed benchmark. v2 removes the inherited ranking
    instruction and makes the score bounds/completeness requirements explicit.
    v3 retains the v2 score contract and explicitly permits frozen evidence.
    The caller must version runs with changed prompts separately.
    """

    if prompt_variant not in {"v1", "v2", "v3"}:
        raise ValueError(f"Unknown scoring prompt variant: {prompt_variant}")
    ensure_label_free_context(assay_memory)

    preamble, _candidates = _split_case(case)
    rows = "\n".join(
        f"{candidate.variant_id} [{candidate.mutation}]: {candidate.sequence}"
        for candidate in chunk.candidates
    )
    memory = json.dumps(
        assay_memory, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    expected = ", ".join(chunk.ids)
    task_system_prompt = case.system_prompt
    if prompt_variant in {"v2", "v3"}:
        # The public RNAgym prompt is a ranking prompt. It remains useful for
        # assay metadata, but its final ranking instruction conflicts with CR.
        task_system_prompt = re.sub(
            r"Rank the mutants by their predicted effect on the assayed RNA property, then output the ranking in the requested JSON format\.?",
            "Estimate an independent continuous fitness score for every supplied mutant.",
            task_system_prompt,
            flags=re.IGNORECASE,
        )
    contract = (
        "On the final line output ONLY one JSON object. It must contain exactly "
        f"{len(chunk.candidates)} objects in the variants array, one for each required ID, "
        "with no missing, duplicate, or extra IDs. Each object must contain only "
        "variant_id and predicted_score. Every predicted_score must be a finite "
        "JSON number in the closed interval [-1.0, 1.0]."
        if prompt_variant in {"v2", "v3"}
        else "On the final line output ONLY this JSON shape, with every supplied ID exactly once and no extra IDs:"
    )
    output_example = (
        "Output JSON only; do not include prose after it."
        if prompt_variant in {"v2", "v3"}
        else '{"variants":[{"variant_id":"M01","predicted_score":0.0}]}'
    )
    return "\n".join(
        [
            "Follow this benchmark instruction exactly. Do not inspect labels, previous answers, or result files.",
            (
                "Predict a calibrated fitness score for every supplied mutant. Use one shared scale across independent chunks: -1 is strongly deleterious, 0 is approximately neutral, and +1 is strongly beneficial for the measured property. The interval [-1.0, 1.0] is a hard output constraint; never emit a value outside it."
                if prompt_variant in {"v2", "v3"}
                else "Predict a calibrated fitness score for every supplied mutant. Use one shared scale across independent chunks: -1 is strongly deleterious, 0 is approximately neutral, and +1 is strongly beneficial for the measured property."
            ),
            "Do not rank only within this chunk. Estimate each mutant relative to wild type and the assay definition so scores remain comparable with candidates evaluated in other chunks.",
            "",
            task_system_prompt,
            "",
            *preamble,
            "",
            (
                "Frozen assay memory (generated without candidate labels or experimental scores):"
                if assay_memory
                else ("No frozen assay summary is supplied. Use the supplied assay metadata and sequences, together with any frozen additional evidence provided in this request. Respect the applicability and uncertainty stated for each source; absent evidence is not evidence of no effect:"
                      if prompt_variant == "v3"
                      else "Frozen assay memory unavailable; use only the supplied assay metadata and sequences:")
            ),
            memory if assay_memory else "{}",
            "",
            f"**{len(chunk.candidates)} candidate mutants to score** (id, mutation(s) relative to WT, full sequence):",
            rows,
            "",
            "Reason through sequence, structural, motif, and assay effects. " + contract,
            output_example,
            f"Required IDs: {expected}",
        ]
    )


def _canonical_id(value: Any, expected: set[str]) -> str | None:
    match = re.fullmatch(r"M(\d+)", str(value).strip(), flags=re.I)
    if not match:
        return None
    number = int(match.group(1))
    return next((item for item in expected if int(item[1:]) == number), None)


class ScoringResponseValidator:
    """Validate one CR response and explain every repairable contract error."""

    def __init__(self, expected_ids: Sequence[str]) -> None:
        if not expected_ids or len(set(expected_ids)) != len(expected_ids):
            raise ValueError("expected_ids must be a non-empty unique sequence")
        self.expected_ids = tuple(expected_ids)
        self.expected = set(expected_ids)

    def validate(self, text: str) -> ValidationReport[dict[str, float]]:
        if not text:
            return ValidationReport(
                valid=False,
                value=None,
                issues=(
                    ValidationIssue(
                        code="empty_response",
                        message="The response contains no final text.",
                    ),
                ),
            )

        try:
            value = final_json_object(text)
        except ValueError:
            value = None
        if not isinstance(value, dict) or set(value) != {"variants"} or not isinstance(value.get("variants"), list):
            return ValidationReport(
                valid=False,
                value=None,
                issues=(
                    ValidationIssue(
                        code="no_parseable_variants",
                        message="The final response must be a JSON object containing only a variants array.",
                    ),
                ),
            )

        rows = value["variants"]
        parsed: dict[str, float] = {}
        seen_ids: set[str] = set()
        duplicate_ids: list[str] = []
        unknown_ids: list[str] = []
        invalid_rows: list[int] = []
        extra_fields: list[int] = []
        missing_scores: list[str] = []
        non_numeric_scores: list[str] = []
        non_finite_scores: list[str] = []
        out_of_range: dict[str, float] = {}

        for row_index, row in enumerate(rows):
            if not isinstance(row, dict):
                invalid_rows.append(row_index)
                continue
            if set(row) - {"variant_id", "predicted_score"}:
                extra_fields.append(row_index)
            raw_id = row.get("variant_id")
            variant_id = _canonical_id(raw_id, self.expected)
            if variant_id is None:
                unknown_ids.append(str(raw_id))
                continue
            if variant_id in seen_ids:
                duplicate_ids.append(variant_id)
                continue
            seen_ids.add(variant_id)
            if "predicted_score" not in row:
                missing_scores.append(variant_id)
                continue
            score = row.get("predicted_score")
            if isinstance(score, bool) or not isinstance(score, (int, float)):
                non_numeric_scores.append(variant_id)
                continue
            numeric = float(score)
            if not math.isfinite(numeric):
                non_finite_scores.append(variant_id)
                continue
            parsed[variant_id] = numeric
            if not -1.0 <= numeric <= 1.0:
                out_of_range[variant_id] = numeric

        missing_ids = [variant_id for variant_id in self.expected_ids if variant_id not in parsed]
        issues: list[ValidationIssue] = []
        if extra_fields:
            issues.append(ValidationIssue(
                code="extra_fields",
                message="Candidate objects may contain only variant_id and predicted_score.",
                details={"row_indices": extra_fields},
            ))
        if invalid_rows:
            issues.append(
                ValidationIssue(
                    code="non_object_rows",
                    message="Every variants entry must be a JSON object.",
                    details={"row_indices": invalid_rows},
                )
            )
        if unknown_ids:
            issues.append(
                ValidationIssue(
                    code="unknown_ids",
                    message="The response contains IDs not supplied by the task.",
                    details={"ids": unknown_ids},
                )
            )
        if duplicate_ids:
            issues.append(
                ValidationIssue(
                    code="duplicate_ids",
                    message="Every required ID must occur exactly once.",
                    details={"ids": sorted(set(duplicate_ids))},
                )
            )
        if missing_scores:
            issues.append(
                ValidationIssue(
                    code="missing_scores",
                    message="Some candidate objects omit predicted_score.",
                    details={"ids": missing_scores},
                )
            )
        if non_numeric_scores:
            issues.append(
                ValidationIssue(
                    code="non_numeric_scores",
                    message="Every predicted_score must be a JSON number.",
                    details={"ids": non_numeric_scores},
                )
            )
        if non_finite_scores:
            issues.append(
                ValidationIssue(
                    code="non_finite_scores",
                    message="Every predicted_score must be finite.",
                    details={"ids": non_finite_scores},
                )
            )
        if out_of_range:
            issues.append(
                ValidationIssue(
                    code="score_out_of_range",
                    message="Every predicted_score must be inside the closed interval [-1.0, 1.0].",
                    details={"scores": out_of_range},
                )
            )
        if missing_ids:
            issues.append(
                ValidationIssue(
                    code="missing_ids",
                    message="The response does not contain every required candidate ID with a valid score.",
                    details={"ids": missing_ids},
                )
            )

        if issues:
            return ValidationReport(valid=False, value=None, issues=tuple(issues))
        if len(rows) != len(self.expected_ids) or set(parsed) != self.expected:
            return ValidationReport(
                valid=False,
                value=None,
                issues=(
                    ValidationIssue(
                        code="cardinality_mismatch",
                        message="The variants array must contain exactly the required IDs once each.",
                        details={
                            "expected": len(self.expected_ids),
                            "received": len(rows),
                        },
                    ),
                ),
            )
        return ValidationReport(
            valid=True,
            value={variant_id: parsed[variant_id] for variant_id in self.expected_ids},
        )


def validate_scoring_scores(
    text: str, expected_ids: Sequence[str]
) -> ValidationReport[dict[str, float]]:
    """Return normalized scores or detailed, label-free validation issues."""

    return ScoringResponseValidator(expected_ids).validate(text)


def parse_scoring_scores(text: str, expected_ids: Sequence[str]) -> dict[str, float] | None:
    """Strictly parse a complete score map; never infer or fill missing values."""

    return validate_scoring_scores(text, expected_ids).value


def ranking_from_scoring_scores(scores: dict[str, float]) -> list[str]:
    return sorted(scores, key=lambda variant_id: (-scores[variant_id], int(variant_id[1:])))
