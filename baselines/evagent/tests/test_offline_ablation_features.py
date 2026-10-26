"""Synthetic geometry/probability checks with no labels or provider imports."""
import importlib.util
import json
import math
from pathlib import Path
import tempfile
import types
import unittest

import numpy as np

from evagent.offline_ablation import digest, file_hash, render_evidence

FEATURE_PATH = Path(__file__).resolve().parents[1] / 'tools/offline_ablation/features.py'
spec = importlib.util.spec_from_file_location('ablation_feature_checks', FEATURE_PATH)
features_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(features_module)


class FrozenFeatureTests(unittest.TestCase):
    def fixture(self, root, *, incomplete=False, low_neff=False):
        atoms = [('N', (0, 0, 0), 'N'), ('CA', (1, 0, 0), 'C'),
                 ('C', (2, 0, 0), 'C'), ('O', (3, 0, 0), 'O'),
                 ('CB', (1, -1, 0), 'C')]
        if incomplete:
            atoms = atoms[:-1]
        lines = []
        for serial, (name, xyz, element) in enumerate(atoms, 1):
            x, y, z = xyz
            lines.append(f'ATOM  {serial:5d} {name:>4s} ALA A   1    {x:8.3f}{y:8.3f}{z:8.3f}{1:6.2f}{10:6.2f}          {element:>2s}')
        lines += ['TER', 'ATOM    101    P   U B   1       6.000   0.000   0.000  1.00 10.00           P',
                  'TER', 'ATOM    102   CA ALA C   1       0.100   0.000   0.000  1.00 10.00           C', 'END']
        pdb = root / 'source.pdb'
        pdb.write_text('\n'.join(lines) + '\n')
        msa_root = root / 'msa'
        (msa_root / 'mappings').mkdir(parents=True)
        (msa_root / 'synthetic.a3m').write_text('>WT\nA\n>natural\nC\n')
        (msa_root / 'mappings/synthetic__SYN__A.json').write_text(json.dumps(dict(
            columns=[dict(position=1, insertion_code='', msa_column=1)], msa_path='synthetic.a3m')))
        size = 4 if low_neff else 40
        array = np.array([[value] for value in [0, 1, -1, -2] for _ in range(size // 4)], dtype=np.int8)
        builder = features_module.FeatureBuilder.__new__(features_module.FeatureBuilder)
        builder.root, builder.output = root, root / 'out'
        builder.msa_root, builder.profiles, builder.structures = msa_root, {}, {}
        builder.models = types.SimpleNamespace(alignment=lambda item: (
            array, np.ones(size), [0], features_module.AA, {}))
        task = dict(source='synthetic', group_id='synthetic_assay', reference_complex_id='SYN',
                    candidates=[dict(id=cid, sites=[dict(chain='A', position=1, insertion_code='',
                                                         wt='A', mutant=mutant, sequence_index_1based=1)])
                                for cid, mutant in [('one', 'C'), ('two', 'D')]])
        candidate_hash = digest(task['candidates'])
        observed = dict(source='synthetic', reference_complex_id='SYN', candidate_hash=candidate_hash,
                        structure_source=dict(path=str(pdb), sha256=file_hash(pdb)),
                        wt_chains=[dict(chain='A', residue_ids=['1'], resolved_sequence='A'),
                                   dict(chain='B', residue_ids=['1'], resolved_sequence=None)])
        partners = dict(candidate_hash=candidate_hash, groups=[['A'], ['B']],
                        evidence={'definition': 'synthetic fixed protein/RNA groups'}, excluded_extra_chains=['C'])
        return builder, task, observed, partners

    def test_weighted_frequencies_gaps_unknowns_and_designated_RNA_partner(self):
        with tempfile.TemporaryDirectory() as directory:
            builder, task, observed, partners = self.fixture(Path(directory))
            features, mapping = builder.build(task, observed, partners)
            rendered = render_evidence(task, ['one', 'two'], features, 'msa_prompt_structure')
            row = rendered['msa_sites']['two'][0]
            self.assertAlmostEqual(row['p_wt'], 10.05 / 21)
            self.assertAlmostEqual(row['p_mutant'], 0.05 / 21)
            self.assertAlmostEqual(row['log_probability_ratio'], math.log(0.05 / 10.05))
            self.assertEqual(row['weighted_valid_count'], 20)
            self.assertEqual(row['weighted_gap_fraction'], .25)
            self.assertEqual(row['weighted_unknown_fraction'], .25)
            geometry = rendered['structure_sites']['one'][0]
            self.assertAlmostEqual(geometry['min_partner_heavy_distance_A'], 3)
            self.assertEqual(geometry['partner_contact_count_5A'], 1)
            self.assertEqual(geometry['partner_chain_ids'], ['B'])
            self.assertEqual(geometry['sasa_status'], 'ok')
            self.assertGreaterEqual(geometry['delta_sasa_partner_A2'], 0)
            self.assertEqual(geometry['site_id'], row['site_id'])
            self.assertEqual(len(mapping['sites']), 1)
            table = json.loads(next((builder.output / 'frozen_probability_tables').glob('*.json')).read_text())
            self.assertAlmostEqual(sum(table['rows'][0]['probabilities_20']), 1)
            self.assertNotIn('probabilities_20', json.dumps(rendered))

    def test_missing_sidechain_keeps_distances_but_null_SASA(self):
        with tempfile.TemporaryDirectory() as directory:
            builder, task, observed, partners = self.fixture(Path(directory), incomplete=True)
            features, _ = builder.build(task, observed, partners)
            row = render_evidence(task, ['one', 'two'], features, 'structure')['structure_sites']['one'][0]
            self.assertEqual(row['site_status'], 'ok')
            self.assertEqual(row['partner_status'], 'ok')
            self.assertEqual(row['sasa_status'], 'missing_WT_heavy_atoms')
            self.assertIsNone(row['sasa_bound_A2'])
            self.assertIsNone(row['delta_sasa_partner_A2'])
            self.assertEqual(row['min_partner_heavy_distance_A'], 3)

    def test_unavailable_structure_and_absent_column_stay_null(self):
        with tempfile.TemporaryDirectory() as directory:
            builder, task, observed, partners = self.fixture(Path(directory))
            features, _ = builder.build(task, observed, partners, unavailable_structure_reason='specified DNA chains are absent')
            row = render_evidence(task, ['one'], features, 'structure')['structure_sites']['one'][0]
            self.assertIsNone(row['min_partner_heavy_distance_A'])
            self.assertIsNone(row['partner_contact_count_5A'])
            self.assertIsNone(row['sasa_bound_A2'])
            self.assertIsNone(row['delta_sasa_partner_A2'])
            self.assertEqual(row['partner_status'], 'missing_partner')
            self.assertEqual(row['partner_chain_ids'], ['B'])
            self.assertEqual(features['structure_definition']['unavailable_reason'], 'specified DNA chains are absent')
            features['msa']['one'][row['site_id']] = dict(
                p_wt=None, p_mutant=None, log_probability_ratio=None, weighted_valid_count=None,
                weighted_gap_fraction=None, weighted_unknown_fraction=None, status='no_valid_homolog_at_column')
            rendered = render_evidence(task, ['one'], features, 'msa_prompt')
            absent = rendered['msa_sites']['one'][0]
            self.assertIsNone(absent['weighted_valid_count'])
            self.assertNotIn('probabilities_20', json.dumps(rendered))

    def test_low_neff_keeps_prompt_statistics_but_independent_score_missing(self):
        with tempfile.TemporaryDirectory() as directory:
            builder, task, observed, partners = self.fixture(Path(directory), low_neff=True)
            features, _ = builder.build(task, observed, partners)
            rendered = render_evidence(task, ['one', 'two'], features, 'msa_prompt')
            row = rendered['msa_sites']['one'][0]
            self.assertEqual(row['status'], 'global_neff_below_frozen_20')
            self.assertIsNotNone(row['p_wt'])
            self.assertIsNotNone(row['p_mutant'])
            self.assertEqual(features['independent_msa_scores'], {'one': None, 'two': None})
            self.assertFalse(features['full_msa_coverage'])


if __name__ == '__main__':
    unittest.main()
