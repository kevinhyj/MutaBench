from __future__ import annotations

import unittest

from evagent.integrated.chunked_scoring import (
    build_scoring_prompt,
    make_scoring_chunks,
    parse_scoring_scores,
    ranking_from_scoring_scores,
    validate_scoring_scores,
)
from evagent.integrated.ranking import RankingCase


class ChunkedScoringTest(unittest.TestCase):
    def setUp(self) -> None:
        self.ids = [f"M{index:02d}" for index in range(1, 101)]
        rows = "\n".join(
            f"M{index:02d} [A{index}U]: AUGC" for index in range(1, 101)
        )
        self.case = RankingCase(
            assay="synthetic",
            ids=self.ids,
            system_prompt="Predict RNA fitness.",
            user_prompt=(
                "**RNA element:** synthetic\n"
                "**Wild-type sequence (4 nt):**\nAUGC\n\n"
                "**100 candidate mutants to rank** (id, mutation(s) relative to WT, full sequence):\n"
                + rows
                + "\n\nRank all 100 mutants from best to worst."
            ),
            prompt_sha256="unused",
        )

    def test_fixed_chunks_cover_once(self) -> None:
        chunks = make_scoring_chunks(self.case)
        self.assertEqual([len(chunk.candidates) for chunk in chunks], [25, 25, 25, 25])
        self.assertEqual(
            [variant_id for chunk in chunks for variant_id in chunk.ids], self.ids
        )

    def test_prompt_is_n_independent_and_has_no_scores(self) -> None:
        chunk = make_scoring_chunks(self.case)[1]
        prompt = build_scoring_prompt(
            self.case,
            chunk,
            assay_memory={"assay_summary": "Synthetic activity assay."},
        )
        self.assertIn("**25 candidate mutants to score**", prompt)
        self.assertNotIn("100 candidate", prompt)
        self.assertIn("M26 [A26U]", prompt)
        self.assertNotIn("M01 [A1U]", prompt)

    def test_v2_prompt_removes_ranking_conflict_and_adds_hard_contract(self) -> None:
        chunk = make_scoring_chunks(self.case)[1]
        prompt = build_scoring_prompt(
            self.case,
            chunk,
            assay_memory={},
            prompt_variant="v2",
        )
        self.assertNotIn(
            "Rank the mutants by their predicted effect on the assayed RNA property",
            prompt,
        )
        self.assertIn("closed interval [-1.0, 1.0]", prompt)
        self.assertIn("exactly 25 objects", prompt)
        self.assertNotIn('{"variants":[{"variant_id":"M01"', prompt)

    def test_strict_score_parser_accepts_padding_and_rejects_partial(self) -> None:
        expected = ["M01", "M02", "M03"]
        valid = (
            'analysis\n{"variants":['
            '{"variant_id":"M001","predicted_score":0.5},'
            '{"variant_id":"M002","predicted_score":0.0},'
            '{"variant_id":"M003","predicted_score":-0.5}]}'
        )
        self.assertEqual(
            parse_scoring_scores(valid, expected),
            {"M01": 0.5, "M02": 0.0, "M03": -0.5},
        )
        self.assertIsNone(
            parse_scoring_scores(
                '{"variants":[{"variant_id":"M01","predicted_score":0.5}]}',
                expected,
            )
        )

    def test_score_validator_reports_actionable_contract_errors(self) -> None:
        expected = ["M01", "M02", "M03"]
        report = validate_scoring_scores(
            '{"variants":['
            '{"variant_id":"M01","predicted_score":-1.2},'
            '{"variant_id":"M01","predicted_score":0.1},'
            '{"variant_id":"M03","predicted_score":0.2}]}',
            expected,
        )
        self.assertFalse(report.valid)
        issues = {issue.code: issue for issue in report.issues}
        self.assertEqual(issues["duplicate_ids"].details["ids"], ["M01"])
        self.assertEqual(issues["missing_ids"].details["ids"], ["M02"])
        self.assertEqual(
            issues["score_out_of_range"].details["scores"],
            {"M01": -1.2},
        )

    def test_scores_merge_without_per_chunk_normalization(self) -> None:
        scores = {"M01": -0.4, "M02": 0.7, "M03": 0.1}
        self.assertEqual(ranking_from_scoring_scores(scores), ["M02", "M03", "M01"])


if __name__ == "__main__":
    unittest.main()
