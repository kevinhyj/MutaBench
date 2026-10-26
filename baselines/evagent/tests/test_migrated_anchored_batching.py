from __future__ import annotations

import unittest

from evagent.integrated.anchored_batching import (
    aggregate_anchor_rankings,
    build_local_ranking_prompt,
    make_anchor_batches,
)
from evagent.integrated.ranking import RankingCase


class AnchorBatchingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.ids = [f"M{index:02d}" for index in range(1, 51)]

    def test_n50_groups_are_bounded_connected_and_deterministic(self) -> None:
        first = make_anchor_batches("synthetic", self.ids)
        second = make_anchor_batches("synthetic", self.ids)

        self.assertEqual(first, second)
        self.assertEqual([len(batch.variant_ids) for batch in first], [25, 17, 16])
        self.assertEqual(len(first[0].anchor_ids), 4)
        self.assertTrue(all(batch.anchor_ids == first[0].anchor_ids for batch in first))
        self.assertEqual(set().union(*(set(batch.variant_ids) for batch in first)), set(self.ids))

    def test_pl_preserves_high_rank_agreement_from_overlapping_groups(self) -> None:
        batches = make_anchor_batches("synthetic", self.ids)
        truth_order = list(reversed(self.ids))
        positions = {variant_id: index for index, variant_id in enumerate(truth_order)}
        local_orders = [
            sorted(batch.variant_ids, key=lambda variant_id: positions[variant_id])
            for batch in batches
        ]

        scores = aggregate_anchor_rankings(self.ids, local_orders)
        observed = sorted(self.ids, key=lambda variant_id: (-scores[variant_id], variant_id))

        # Candidates occurring in different groups but between the same two
        # anchors have no direct comparison, so exact recovery is not an ACR
        # invariant.  The aggregate must nevertheless preserve strong global
        # rank agreement and produce one score for every candidate.
        observed_positions = {variant_id: index for index, variant_id in enumerate(observed)}
        squared_error = sum(
            (observed_positions[variant_id] - positions[variant_id]) ** 2
            for variant_id in self.ids
        )
        rho = 1 - 6 * squared_error / (len(self.ids) * (len(self.ids) ** 2 - 1))
        self.assertGreater(rho, 0.9)
        self.assertEqual(set(scores), set(self.ids))

    def test_local_prompt_only_contains_selected_candidates(self) -> None:
        rows = "\n".join(
            f"M{index:02d} [A{index}U]: AU" for index in range(1, 51)
        )
        case = RankingCase(
            assay="synthetic",
            ids=self.ids,
            system_prompt="Rank RNA mutants.",
            user_prompt=(
                "**RNA element:** synthetic\n\n"
                "**50 candidate mutants to rank** (id, mutation(s) relative to WT, full sequence):\n"
                + rows
                + "\n\nRank all 50 mutants from best to worst."
            ),
            prompt_sha256="not-used",
        )
        batch = make_anchor_batches("synthetic", self.ids)[0]
        prompt = build_local_ranking_prompt(case, batch)

        self.assertIn("**25 candidate mutants to rank**", prompt)
        self.assertIn("Rank all 25 mutants", prompt)
        present = {variant_id for variant_id in self.ids if f"{variant_id} [" in prompt}
        self.assertEqual(present, set(batch.variant_ids))


if __name__ == "__main__":
    unittest.main()
