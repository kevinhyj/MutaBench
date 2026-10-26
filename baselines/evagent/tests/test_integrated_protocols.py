"""Synthetic offline contract tests; no experimental labels or API requests."""

import json
import unittest
from dataclasses import replace

from evagent.integrated.chunked_scoring import ScoringResponseValidator
from evagent.integrated.protocols import build_protocol_tasks, execute_protocol
from evagent.integrated.ranking import RankingCase, parse_ranking, validate_ranking


def synthetic_case(size: int = 6) -> RankingCase:
    ids = [f"M{index:02d}" for index in range(1, size + 1)]
    return RankingCase(
        assay="synthetic", ids=ids,
        system_prompt="Rank the mutants by their predicted effect on the assayed RNA property, then output the ranking in the requested JSON format.",
        user_prompt=(
            "Synthetic fixture, no measured outcomes.\n"
            f"**{size} candidate mutants to rank** (id, mutation, full sequence):\n"
            + "\n".join(f"{candidate} [synthetic]: ACGU" for candidate in ids)
            + f"\nRank all {size} mutants from best to worst.\n"
            + '{"ranking":["M01", ...]}'
        ),
    )


def complete_generation(**kwargs):
    ids = kwargs["metadata"]["expected_ids"]
    if kwargs["metadata"]["method"] == "cr":
        artifact = {"variants": [{"variant_id": candidate, "predicted_score": 0.125} for candidate in ids]}
    else:
        artifact = {"ranking": sorted(ids)}
    return {"status": "ok", "final_text": json.dumps(artifact),
            "usage_summary": {"model_calls": 1, "input_tokens": 7, "output_tokens": 3, "total_tokens": 10}}


class IntegratedProtocolTest(unittest.TestCase):
    def test_strict_ranking_accepts_final_json(self):
        self.assertEqual(parse_ranking('Reasoning.\n{"ranking":["M02","M01"]}', ["M01", "M02"]), ["M02", "M01"])

    def test_strict_ranking_rejects_extra_unknown_duplicate_and_nonstring_ids(self):
        for order in (["M01", "M02", "M99"], ["M01", "M01"], ["M01"], ["M01", 2], ["M001", "M002"]):
            with self.subTest(order=order):
                self.assertIsNone(parse_ranking(json.dumps({"ranking": order}), ["M01", "M02"]))
        report = validate_ranking('{"ranking":["M01","M02","M99"]}', ["M01", "M02"])
        self.assertIn("unknown_ids", [issue.code for issue in report.issues])

    def test_strict_ranking_rejects_regex_only_malformed_or_ambiguous_json(self):
        for output in (
            "M01 M02", '{"ranking": [M01, M02]}', '{"ranking":["M01","M02"],}',
            '{"ranking":["M01","M02"]} trailing',
            '{"ranking":["M01"],"ranking":["M01","M02"]}',
            '{"ranking":["M01","M02"],"extra":0}',
        ):
            with self.subTest(output=output):
                self.assertIsNone(parse_ranking(output, ["M01", "M02"]))

    def test_score_validator_rejects_invalid_numbers_and_fields(self):
        validator = ScoringResponseValidator(["M01"])
        for value in (True, "0.2", 2.0, float("nan"), float("inf")):
            with self.subTest(value=value):
                self.assertFalse(validator.validate(json.dumps({"variants": [{"variant_id": "M01", "predicted_score": value}]})).valid)
        self.assertFalse(validator.validate('{"variants":[{"variant_id":"M01","predicted_score":0.1,"label":1}]}').valid)

    def test_both_cr_prompt_versions_are_available_and_v2_clears_outer_conflict(self):
        case = synthetic_case()
        for version in ("v1", "v2"):
            task = build_protocol_tasks("cr", case=case, assay_memory={}, prompt_version=version, group_size=3)[0]
            self.assertIn(version, task.metadata["protocol"])
            if version == "v2":
                self.assertNotIn("Rank the mutants by", task.system)
                self.assertNotIn("Rank the mutants by", task.user)
                self.assertIn("closed interval [-1.0, 1.0]", task.user)
            else:
                self.assertIn("Rank the mutants by", task.system)

    def test_cr_absolute_scores_are_not_normalized(self):
        result = execute_protocol("cr", case=synthetic_case(), generate=complete_generation, assay_memory={}, group_size=3)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["scores"], {candidate: 0.125 for candidate in synthetic_case().ids})
        self.assertEqual(result["ranking"], synthetic_case().ids)
        self.assertEqual(result["usage_summary"]["total_tokens"], 20)

    def test_all_protocols_complete_offline(self):
        for method in ("global", "cr", "acr"):
            with self.subTest(method=method):
                result = execute_protocol(method, case=synthetic_case(), generate=complete_generation, assay_memory={}, group_size=4, anchor_count=1)
                self.assertEqual(result["status"], "ok")
                self.assertEqual(set(result["ranking"]), set(synthetic_case().ids))
                self.assertEqual(len(result["attempts"]), result["total_units"])

    def test_any_invalid_chunk_invalidates_entire_cell_without_filling(self):
        def generate(**kwargs):
            result = complete_generation(**kwargs)
            if kwargs["metadata"]["unit_index"] == 0:
                result["final_text"] = '{"variants":[]}'
            return result
        result = execute_protocol("cr", case=synthetic_case(), generate=generate, assay_memory={}, group_size=3)
        self.assertEqual(result["status"], "invalid_output")
        self.assertIsNone(result["ranking"])
        self.assertIsNone(result["scores"])
        self.assertEqual(result["valid_units"], 1)
        self.assertEqual(len(result["units"]), 2)
        self.assertEqual(result["usage_summary"]["model_calls"], 2)

    def test_bounded_repairs_include_all_usage_and_preserve_diagnostics(self):
        calls = []
        def generate(**kwargs):
            calls.append(kwargs)
            result = complete_generation(**kwargs)
            if kwargs["metadata"]["attempt_index"] == 0:
                result["final_text"] = '{"variants":[]}'
            return result
        result = execute_protocol("cr", case=synthetic_case(2), generate=generate, assay_memory={}, max_repairs=1)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["usage_summary"]["total_tokens"], 20)
        self.assertEqual(result["repair_attempts_used"], 1)
        self.assertEqual([attempt["attempt_kind"] for attempt in result["attempts"]], ["initial", "repair"])
        self.assertIn("VALIDATION_DIAGNOSTICS", calls[1]["user"])
        self.assertNotEqual(calls[0]["metadata"]["prompt_sha256"], calls[1]["metadata"]["prompt_sha256"])

    def test_invalid_response_stops_at_repair_limit(self):
        calls = []
        def generate(**kwargs):
            calls.append(kwargs)
            return {"status": "ok", "final_text": "invalid", "usage_summary": {"model_calls": 1, "total_tokens": 1}}
        result = execute_protocol("global", case=synthetic_case(), generate=generate, assay_memory={}, max_repairs=2)
        self.assertEqual(len(calls), 3)
        self.assertEqual(result["status"], "invalid_output")
        self.assertEqual(result["usage_summary"]["model_calls"], 3)

    def test_provider_failures_never_trigger_semantic_repair_or_later_groups(self):
        for failure in ("api_error", "blocked", "truncated", "budget_exhausted"):
            calls = []
            def generate(**kwargs):
                calls.append(kwargs)
                return {"status": failure, "final_text": "", "usage_summary": {"model_calls": 1, "total_tokens": 1}}
            with self.subTest(failure=failure):
                result = execute_protocol("cr", case=synthetic_case(), generate=generate, assay_memory={}, max_repairs=2, group_size=3)
                self.assertEqual(len(calls), 1)
                self.assertEqual(result["status"], failure)
                self.assertEqual(result["units"][1]["status"], "not_run")

    def test_execution_pause_exception_propagates_unchanged(self):
        class ExecutionPaused(RuntimeError):
            pass
        pause = ExecutionPaused("ambiguous response")
        def generate(**kwargs):
            raise pause
        with self.assertRaises(ExecutionPaused) as raised:
            execute_protocol("cr", case=synthetic_case(), generate=generate, assay_memory={}, max_repairs=2)
        self.assertIs(raised.exception, pause)

    def test_populated_label_field_rejected_before_callback(self):
        case = replace(synthetic_case(), ground_truth_scores=[99.0] * 6)
        with self.assertRaisesRegex(ValueError, "ground_truth_scores"):
            execute_protocol("global", case=case, generate=complete_generation, assay_memory={})

    def test_assay_result_fields_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "prohibited result field"):
            build_protocol_tasks("cr", case=synthetic_case(), assay_memory={"nested": {"scores": [99.0]}})

    def test_assay_context_and_additional_evidence_enter_all_methods(self):
        for method in ("global", "cr", "acr"):
            tasks = build_protocol_tasks(method, case=synthetic_case(), assay_memory={"summary": "Assay summary marker."}, context_text="Frozen evidence marker.")
            self.assertTrue(all("Assay summary marker." in task.user and "Frozen evidence marker." in task.user for task in tasks))

    def test_prompt_fingerprint_changes_with_context(self):
        first = build_protocol_tasks("global", case=synthetic_case(), assay_memory={}, context_text="a")[0]
        second = build_protocol_tasks("global", case=synthetic_case(), assay_memory={}, context_text="b")[0]
        self.assertNotEqual(first.metadata["prompt_sha256"], second.metadata["prompt_sha256"])

    def test_v3_allows_evidence_and_preserves_frozen_v2(self):
        case = synthetic_case()
        old = build_protocol_tasks("cr", case=case, assay_memory={}, prompt_version="v2")[0]
        new = build_protocol_tasks("cr", case=case, assay_memory={}, prompt_version="v3",
                                   context_text="Reviewed evidence marker")[0]
        self.assertIn("use only the supplied assay metadata and sequences", old.user)
        self.assertNotIn("use only the supplied assay metadata and sequences", new.user)
        self.assertIn("any frozen additional evidence provided in this request", new.user)
        self.assertIn("Reviewed evidence marker", new.user)
        self.assertNotIn("Rank the mutants", new.system)
        self.assertNotEqual(old.metadata["prompt_sha256"], new.metadata["prompt_sha256"])
        result = execute_protocol("cr", case=case, assay_memory={}, prompt_version="v3",
                                  context_text="Reviewed evidence marker", generate=complete_generation)
        self.assertEqual(result['status'], 'ok')


if __name__ == "__main__":
    unittest.main()
