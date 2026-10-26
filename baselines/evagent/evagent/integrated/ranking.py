"""Label-free ranking inputs and strict validation of the final JSON artifact.

RankingCase and the prompt wrapper originate in the collaborator's ranking.py.
Unlike its historical parser, this validator never discards unknown IDs or
extracts IDs from prose. Evaluation and label loading live outside protocols.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Sequence

from .self_check import ValidationIssue, ValidationReport


@dataclass(frozen=True)
class RankingCase:
    assay: str
    ids: list[str]
    system_prompt: str
    user_prompt: str
    prompt_sha256: str = ""
    # Compatibility field only. Protocol entrypoints reject populated labels.
    ground_truth_scores: list[float] | None = None
    source: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if not self.ids or any(not isinstance(value, str) or not value for value in self.ids):
            raise ValueError("ids must be non-empty strings")
        if len(self.ids) != len(set(self.ids)):
            raise ValueError("ids must be unique")
        if not self.prompt_sha256:
            object.__setattr__(
                self,
                "prompt_sha256",
                hashlib.sha256((self.system_prompt + "\n\n" + self.user_prompt).encode()).hexdigest(),
            )

    @property
    def agent_prompt(self) -> str:
        ensure_label_free_case(self)
        return (
            "Follow this benchmark instruction exactly. Do not use tools to inspect labels or "
            "previous answers.\n\n"
            f"{self.system_prompt}\n\n{self.user_prompt}"
        )


def ensure_label_free_case(case: RankingCase) -> None:
    if case.ground_truth_scores is not None:
        raise ValueError("Model-facing RankingCase must not contain ground_truth_scores")


def ensure_label_free_context(value: Any) -> None:
    prohibited = {
        "ground_truth", "ground_truth_scores", "ground_truth_score", "labels", "label",
        "experimental_scores", "fitness_values", "fitness_value", "ranking", "scores", "score",
    }
    if isinstance(value, dict):
        for key, nested in value.items():
            if str(key).lower() in prohibited:
                raise ValueError(f"Frozen assay context contains a prohibited result field: {key}")
            ensure_label_free_context(nested)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            ensure_label_free_context(nested)


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_json_constant(constant: str) -> None:
    raise ValueError(f"Invalid JSON constant: {constant}")


def final_json_object(text: str) -> dict[str, Any]:
    """Decode a final JSON object, permitting preceding reasoning/code fences.

    No token/ID extraction or guessed JSON repair is performed. Trailing prose,
    duplicate keys, nonstandard numeric constants, and non-object roots fail.
    """
    trimmed = text.strip()
    if trimmed.endswith("```"):
        trimmed = trimmed[:-3].rstrip()
    decoder = json.JSONDecoder(
        object_pairs_hook=_unique_object,
        parse_constant=_reject_json_constant,
    )
    for index, character in enumerate(trimmed):
        if character != "{":
            continue
        try:
            value, end = decoder.raw_decode(trimmed[index:])
        except (ValueError, json.JSONDecodeError):
            continue
        if not trimmed[index + end :].strip() and isinstance(value, dict):
            return value
    raise ValueError("Expected a complete final JSON object with no trailing prose")


class RankingResponseValidator:
    def __init__(self, expected_ids: Sequence[str]) -> None:
        if not expected_ids or len(expected_ids) != len(set(expected_ids)):
            raise ValueError("expected_ids must be non-empty and unique")
        self.expected_ids = tuple(expected_ids)
        self.expected = set(expected_ids)

    def validate(self, text: str) -> ValidationReport[list[str]]:
        try:
            value = final_json_object(text)
        except ValueError as error:
            return ValidationReport(False, None, (ValidationIssue("invalid_json", str(error)),))
        if set(value) != {"ranking"} or not isinstance(value.get("ranking"), list):
            return ValidationReport(False, None, (ValidationIssue(
                "invalid_ranking_shape", "Return one JSON object containing only a ranking array."
            ),))
        order = value["ranking"]
        issues: list[ValidationIssue] = []
        non_strings = [index for index, item in enumerate(order) if not isinstance(item, str)]
        known_strings = [item for item in order if isinstance(item, str)]
        unknown = [item for item in known_strings if item not in self.expected]
        duplicates = sorted({item for item in known_strings if known_strings.count(item) > 1})
        missing = [item for item in self.expected_ids if item not in known_strings]
        if non_strings:
            issues.append(ValidationIssue("invalid_ids", "Every ranking entry must be a string ID.", {"indices": non_strings}))
        if unknown:
            issues.append(ValidationIssue("unknown_ids", "Do not include IDs outside the supplied candidates.", {"ids": unknown}))
        if duplicates:
            issues.append(ValidationIssue("duplicate_ids", "Every supplied ID must occur exactly once.", {"ids": duplicates}))
        if missing:
            issues.append(ValidationIssue("missing_ids", "The ranking must include every supplied ID.", {"ids": missing}))
        if len(order) != len(self.expected_ids):
            issues.append(ValidationIssue("cardinality_mismatch", "The ranking length must equal the candidate count.", {"expected": len(self.expected_ids), "received": len(order)}))
        return ValidationReport(not issues, list(order) if not issues else None, tuple(issues))


def validate_ranking(text: str, ids: Sequence[str]) -> ValidationReport[list[str]]:
    return RankingResponseValidator(ids).validate(text)


def parse_ranking(text: str, ids: Sequence[str]) -> list[str] | None:
    return validate_ranking(text, ids).value
