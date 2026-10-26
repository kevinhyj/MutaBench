from __future__ import annotations

import json
import unittest

from evagent.integrated.learning_memory import (
    EpisodeLesson,
    KnowledgeCard,
    PromotionEvidence,
    TaskContext,
    advance_card,
    compile_strategy_text,
    consolidate_lessons,
    parse_episode_reflection,
    registry_payload,
    retrieve_cards,
)


class LearningMemoryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.context = TaskContext(
            rna_type="ribozyme",
            phenotype="self cleavage",
            length_bucket="short",
            mutation_count_bucket="double",
            structure_available=True,
            output_mode="ranking",
        )
        self.response = json.dumps(
            {
                "strategy_used": "Compared the expected output contract with the returned identifiers.",
                "what_worked": "Explicit checks reduced omitted identifiers.",
                "what_failed": "A valid JSON wrapper did not establish complete coverage.",
                "condition": "a complete ranking is required",
                "proposed_rule": "Verify identifier coverage and uniqueness before returning the output.",
                "contraindication": "Do not infer validity solely from valid JSON.",
                "tags": ["rna_type:ribozyme", "structure:available", "output_mode:ranking"],
                "confidence": 0.7,
            }
        )

    def lesson(self, suffix: str = "a") -> EpisodeLesson:
        return parse_episode_reflection(
            self.response,
            context=self.context,
            split="development",
            source_group=f"source-{suffix}",
            artifact_sha256=suffix * 64,
            outcome="improved",
            forbidden_assays=["Heldout_2026_ribozyme"],
        )

    def test_reflection_is_deidentified_and_development_only(self) -> None:
        lesson = self.lesson()
        self.assertTrue(lesson.episode_id.startswith("exp_"))
        self.assertNotIn("source-a", lesson.source_group_hash)
        with self.assertRaisesRegex(ValueError, "development"):
            parse_episode_reflection(
                self.response,
                context=self.context,
                split="test",
                source_group="source-a",
                artifact_sha256="a" * 64,
                outcome="improved",
                forbidden_assays=[],
            )

    def test_reflection_rejects_direct_leakage(self) -> None:
        value = json.loads(self.response)
        value["proposed_rule"] = "For Heldout_2026_ribozyme rank M12 first."
        with self.assertRaises(ValueError):
            parse_episode_reflection(
                json.dumps(value),
                context=self.context,
                split="development",
                source_group="source-a",
                artifact_sha256="a" * 64,
                outcome="improved",
                forbidden_assays=["Heldout_2026_ribozyme"],
            )

    def test_consolidate_promote_retrieve_and_compile(self) -> None:
        cards = consolidate_lessons([self.lesson("a"), self.lesson("b")])
        self.assertEqual(len(cards), 1)
        self.assertEqual(cards[0].support_count, 2)
        active = advance_card(
            cards[0],
            PromotionEvidence(
                paired_assays=10,
                mean_delta=0.03,
                ci_low=0.01,
                ci_high=0.05,
                pass_rate_delta=0.0,
            ),
        )
        self.assertEqual(active.status, "active")
        selected = retrieve_cards([active], self.context)
        self.assertEqual(selected, [active])
        text = compile_strategy_text(selected, forbidden_assays=[])
        self.assertIn("Verify identifier coverage", text)
        payload = registry_payload([active])
        self.assertEqual(payload["schema_version"], 1)
        self.assertEqual(len(payload["registry_sha256"]), 64)

    def test_retrieval_excludes_conflicting_scope(self) -> None:
        card = KnowledgeCard(
            card_id="mem_x",
            version=1,
            kind="procedural",
            status="active",
            condition="a coding sequence is evaluated",
            guidance="Inspect coding consequences.",
            contraindication="Do not apply to noncoding RNA.",
            tags=("rna_type:mrna_coding",),
            confidence=0.9,
            support_count=10,
            source_episode_ids=("exp_x",),
        )
        self.assertEqual(retrieve_cards([card], self.context), [])

    def test_same_rule_with_different_scope_is_not_merged(self) -> None:
        first = self.lesson("a")
        other_context = TaskContext(rna_type="mrna_coding", output_mode="ranking")
        value = json.loads(self.response)
        value["tags"] = ["rna_type:mrna_coding", "output_mode:ranking"]
        second = parse_episode_reflection(
            json.dumps(value),
            context=other_context,
            split="development",
            source_group="source-b",
            artifact_sha256="b" * 64,
            outcome="improved",
            forbidden_assays=[],
        )
        self.assertEqual(len(consolidate_lessons([first, second])), 2)

    def test_failed_selection_rejects_card(self) -> None:
        card = consolidate_lessons([self.lesson()])[0]
        rejected = advance_card(
            card,
            PromotionEvidence(
                paired_assays=10,
                mean_delta=-0.01,
                ci_low=-0.03,
                ci_high=0.01,
                pass_rate_delta=0.0,
            ),
        )
        self.assertEqual(rejected.status, "rejected")
        self.assertEqual(retrieve_cards([rejected], self.context), [])

    def test_promotion_rejects_nonfinite_or_incoherent_evidence(self) -> None:
        card = consolidate_lessons([self.lesson()])[0]
        invalid = [
            PromotionEvidence(10, float("nan"), 0.01, 0.03, 0.0),
            PromotionEvidence(10, 0.02, 0.01, float("inf"), 0.0),
            PromotionEvidence(10, 0.02, 0.03, 0.01, 0.0),
            PromotionEvidence(10, 0.02, 0.01, None, 0.0),
            PromotionEvidence(True, 0.02, 0.01, 0.03, 0.0),
        ]
        for evidence in invalid:
            with self.subTest(evidence=evidence), self.assertRaises(ValueError):
                advance_card(card, evidence)


if __name__ == "__main__":
    unittest.main()
