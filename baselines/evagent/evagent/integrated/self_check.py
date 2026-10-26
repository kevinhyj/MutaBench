"""Bounded, auditable self-check loops for Deep Agents tasks.

The loop deliberately separates deterministic validation from model critique.
Validators never need benchmark labels: they inspect only the task contract and
the generated artifact. Every generation and repair is retained for audit.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Generic, Protocol, TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class ValidationIssue:
    """One machine-readable contract violation."""

    code: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)
    repairable: bool = True

    def as_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "details": self.details,
            "repairable": self.repairable,
        }


@dataclass(frozen=True)
class ValidationReport(Generic[T]):
    """Validator output with an optional normalized value."""

    valid: bool
    value: T | None
    issues: tuple[ValidationIssue, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "issues": [issue.as_dict() for issue in self.issues],
        }


class TextValidator(Protocol[T]):
    def validate(self, text: str) -> ValidationReport[T]:
        """Validate generated text without consulting task labels."""


@dataclass(frozen=True)
class SelfCheckPolicy:
    """Safety and cost bounds for one task episode."""

    max_repairs: int = 1
    stop_on_nonrepairable: bool = True

    def __post_init__(self) -> None:
        if self.max_repairs < 0:
            raise ValueError("max_repairs must be non-negative")


Generation = dict[str, Any]
Generate = Callable[[str, int], Generation]
RepairPromptBuilder = Callable[[str, str, ValidationReport[Any], int], str]


def default_repair_prompt(
    original_prompt: str,
    previous_text: str,
    report: ValidationReport[Any],
    repair_number: int,
) -> str:
    """Build a label-free correction request from deterministic diagnostics."""

    diagnostics = json.dumps(
        [issue.as_dict() for issue in report.issues],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return "\n".join(
        [
            "A previous response failed deterministic validation.",
            f"Repair attempt: {repair_number}.",
            "Correct only the reported contract violations. Preserve valid prior values and judgments unless a reported violation requires changing them.",
            "Do not inspect or request labels, answer keys, benchmark scores, previous tasks, or result files.",
            "Re-read the original task and return a complete final answer satisfying its output contract.",
            "",
            "VALIDATION_DIAGNOSTICS:",
            diagnostics,
            "",
            "PREVIOUS_RESPONSE:",
            previous_text,
            "",
            "ORIGINAL_TASK:",
            original_prompt,
        ]
    )


def _sum_usage(attempts: list[dict[str, Any]]) -> dict[str, int]:
    fields = (
        "model_calls",
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "reasoning_tokens",
        "cache_read_tokens",
    )
    return {
        field: sum(
            int((attempt.get("usage_summary") or {}).get(field) or 0)
            for attempt in attempts
        )
        for field in fields
    }


def run_self_check(
    *,
    original_prompt: str,
    generate: Generate,
    validator: TextValidator[T],
    policy: SelfCheckPolicy | None = None,
    repair_prompt_builder: RepairPromptBuilder = default_repair_prompt,
) -> dict[str, Any]:
    """Generate, validate, and perform bounded repair attempts.

    The generate callback receives the prompt and zero-based attempt index. It
    should return final_text plus any audit metadata. A generation-level
    status=api_error is treated as non-repairable here; transport retries belong
    below this semantic self-check layer.
    """

    active_policy = policy or SelfCheckPolicy()
    prompt = original_prompt
    attempts: list[dict[str, Any]] = []
    value: T | None = None
    terminal_status = "invalid_output"

    for attempt_index in range(active_policy.max_repairs + 1):
        generation = dict(generate(prompt, attempt_index))
        generation["attempt_index"] = attempt_index
        generation["attempt_kind"] = "initial" if attempt_index == 0 else "repair"
        generation["attempt_prompt_sha256"] = hashlib.sha256(
            prompt.encode("utf-8")
        ).hexdigest()

        if generation.get("status") not in {"ok", "generated"}:
            failure_status = str(generation.get("status") or "generation_error")
            report: ValidationReport[T] = ValidationReport(
                valid=False,
                value=None,
                issues=(
                    ValidationIssue(
                        code=failure_status,
                        message="Generation did not produce a complete provider response; semantic repair is disabled.",
                        details={
                            "error_type": generation.get("error_type"),
                            "error": generation.get("error"),
                        },
                        repairable=False,
                    ),
                ),
            )
            terminal_status = failure_status
        else:
            report = validator.validate(str(generation.get("final_text") or ""))
            terminal_status = "ok" if report.valid else "invalid_output"

        generation["validation"] = report.as_dict()
        attempts.append(generation)
        if report.valid:
            value = report.value
            break
        if attempt_index >= active_policy.max_repairs:
            break
        if active_policy.stop_on_nonrepairable and any(
            not issue.repairable for issue in report.issues
        ):
            break
        prompt = repair_prompt_builder(
            original_prompt,
            str(generation.get("final_text") or ""),
            report,
            attempt_index + 1,
        )

    return {
        "status": terminal_status,
        "value": value,
        "attempts": attempts,
        "repair_attempts_used": max(0, len(attempts) - 1),
        "usage_summary": _sum_usage(attempts),
    }
