from __future__ import annotations

import unittest

from evagent.integrated.self_check import (
    SelfCheckPolicy,
    ValidationIssue,
    ValidationReport,
    run_self_check,
)


class _ExactTextValidator:
    def validate(self, text: str) -> ValidationReport[str]:
        if text == "valid":
            return ValidationReport(valid=True, value=text)
        return ValidationReport(
            valid=False,
            value=None,
            issues=(
                ValidationIssue(
                    code="wrong_text",
                    message="Response must equal valid.",
                ),
            ),
        )


class SelfCheckTest(unittest.TestCase):
    def test_repairs_once_and_retains_both_attempts(self) -> None:
        prompts: list[str] = []

        def generate(prompt: str, attempt_index: int) -> dict:
            prompts.append(prompt)
            return {
                "status": "generated",
                "final_text": "invalid" if attempt_index == 0 else "valid",
                "usage_summary": {"model_calls": 1, "total_tokens": 10},
            }

        result = run_self_check(
            original_prompt="original task",
            generate=generate,
            validator=_ExactTextValidator(),
            policy=SelfCheckPolicy(max_repairs=1),
        )

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["value"], "valid")
        self.assertEqual(len(result["attempts"]), 2)
        self.assertEqual(result["repair_attempts_used"], 1)
        self.assertEqual(result["usage_summary"]["model_calls"], 2)
        self.assertEqual(result["usage_summary"]["total_tokens"], 20)
        self.assertIn("wrong_text", prompts[1])
        self.assertIn("original task", prompts[1])
        self.assertIn("invalid", prompts[1])

    def test_api_error_does_not_trigger_semantic_repair(self) -> None:
        calls = 0

        def generate(_prompt: str, _attempt_index: int) -> dict:
            nonlocal calls
            calls += 1
            return {
                "status": "api_error",
                "final_text": "",
                "error_type": "TimeoutError",
                "error": "timed out",
            }

        result = run_self_check(
            original_prompt="original task",
            generate=generate,
            validator=_ExactTextValidator(),
            policy=SelfCheckPolicy(max_repairs=2),
        )

        self.assertEqual(result["status"], "api_error")
        self.assertEqual(calls, 1)
        self.assertEqual(
            result["attempts"][0]["validation"]["issues"][0]["code"],
            "api_error",
        )


if __name__ == "__main__":
    unittest.main()
