"""MSA sampling must preserve evidence independently of input order/labels."""
import copy
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

from evagent.offline_ablation import render_evidence
import test_offline_ablation_features as fixture_checks

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools/offline_ablation'))
from rna_features import ProteinRNAFeatureBuilder, inverse_identity_weights, selected_alignment


class ProteinRNAMSASelectionTests(unittest.TestCase):
    def test_sha256_selection_has_both_fragment_regions_and_is_order_invariant(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'msa.a3m'
            words = [''.join('C' if value & (1 << bit) else 'A' for bit in range(5)) for value in range(32)]
            homologs = ['-----' + word for word in words] + [word + '-----' for word in words]
            self.assertTrue(all(row.startswith('-----') for row in sorted(homologs)[:16]))
            def write(rows):
                path.write_text('>WT\nAAAAAAAAAA\n' + ''.join(f'>h{i}\n{row}\n' for i, row in enumerate(rows)))
            write(homologs)
            array, positions, first = selected_alignment(path, 'AAAAAAAAAA', cap=16)
            self.assertEqual(positions, list(range(10)))
            self.assertTrue((array[:, 0] >= 0).any())
            self.assertTrue((array[:, 9] >= 0).any())
            self.assertEqual(first['full_pool_support_by_observed_index']['1'], 32)
            write(list(reversed(homologs)))
            other, other_positions, second = selected_alignment(path, 'AAAAAAAAAA', cap=16)
            np.testing.assert_array_equal(array, other)
            self.assertEqual(positions, other_positions)
            self.assertEqual(first['selected_homologs_sha256'], second['selected_homologs_sha256'])

    def test_neighbor_weights_match_manual_pairwise_counts(self):
        array = np.asarray([[0, 0, 0, 0], [0, 0, 0, 0], [1, 1, 1, 1], [-1, -1, 1, 1]], dtype=np.int8)
        np.testing.assert_allclose(inverse_identity_weights(array), [.5, .5, .5, .5])

    def test_low_neff_statistics_score_all_candidates_and_keep_strict_sidecar(self):
        with tempfile.TemporaryDirectory() as directory:
            builder, task, observed, partners = fixture_checks.FrozenFeatureTests().fixture(Path(directory), low_neff=True)
            builder.__class__ = ProteinRNAFeatureBuilder
            (builder.msa_root / 'synthetic.a3m').write_text('>WT\nA\n>one\nC\n>two\nD\n>three\nE\n>ambiguous\nX\n')
            original = copy.deepcopy(task)
            features, mapping = builder.build(task, observed, partners)
            self.assertEqual(task, original)
            self.assertTrue(features['full_msa_coverage'])
            self.assertFalse(features['legacy_strict_full_msa_coverage'])
            self.assertEqual(features['legacy_strict_independent_msa_scores'], {'one': None, 'two': None})
            self.assertTrue(all(value is not None for value in features['independent_msa_scores'].values()))
            rendered = render_evidence(task, ['one', 'two'], features, 'msa_prompt')
            self.assertEqual(rendered['msa_sites']['one'][0]['status'], 'global_neff_below_frozen_20')
            self.assertEqual(rendered['msa_chains'][0]['global_neff'], 3)
            self.assertEqual(len(mapping['sites']), 1)


if __name__ == '__main__':
    unittest.main()
