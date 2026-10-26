"""Offline regression checks; synthetic inputs and no provider connection."""
import copy
import json
import tempfile
import unittest
from pathlib import Path

from evagent.offline_ablation import (
    MSA_FIELDS, STRUCTURE_FIELDS, base_sites, file_hash, fuse_rankings,
    matching_cache, midranks, render_evidence, site_identity,
)


class OfflineAblationTests(unittest.TestCase):
    def task(self):
        return dict(group_id='synthetic_assay', source='synthetic', candidates=[
            dict(id='one', sites=[dict(chain='A', position=5, insertion_code='',
                                      wt='A', mutant='C', sequence_index_1based=1)]),
            dict(id='two', sites=[dict(chain='A', position=5, insertion_code='',
                                      wt='A', mutant='D', sequence_index_1based=1)]),
        ])

    def test_average_ties_and_direction(self):
        self.assertEqual(midranks([2, 2, 4]), [1.5, 1.5, 3])
        self.assertEqual(midranks([2, 2, 4], descending=True), [2.5, 2.5, 1])
        rows = fuse_rankings(['a', 'b', 'c'], dict(a=2, b=2, c=4), dict(a=0, b=0, c=8))
        self.assertEqual([r['gemini_rank_percentile'] for r in rows], [0.25, 0.25, 1])
        self.assertEqual([r['final_rank'] for r in rows], [2.5, 2.5, 1])

    def test_raw_score_scale_cannot_change_fusion(self):
        ids = ['a', 'b', 'c']
        a = fuse_rankings(ids, dict(a=-1000, b=0, c=1), dict(a=100, b=80, c=-100))
        b = fuse_rankings(ids, dict(a=-1, b=0, c=1), dict(a=3, b=2, c=1))
        self.assertEqual([r['fusion_value'] for r in a], [r['fusion_value'] for r in b])
        self.assertEqual([r['final_rank'] for r in a], [2, 2, 2])

    def test_n250_equal_rank_sums_do_not_split_by_float_roundoff(self):
        ids = [str(index) for index in range(250)]
        g = {cid: index + 1 for index, cid in enumerate(ids)}
        m = dict(zip(ids, [4, 3, 1, 2] + list(range(5, 251))))
        rows = fuse_rankings(ids, g, m)
        self.assertEqual(rows[0]['fusion_value'], rows[1]['fusion_value'])
        self.assertEqual(rows[0]['final_rank'], rows[1]['final_rank'])

    def test_missing_extra_nonfinite_and_singleton_rejected(self):
        for scores in (dict(a=1), dict(a=1, b=None), dict(a=1, b=float('nan')),
                       dict(a=1, b=True), dict(a=1, b=2, extra=3)):
            with self.assertRaises(ValueError):
                fuse_rankings(['a', 'b'], scores, dict(a=1, b=2))
        with self.assertRaises(ValueError):
            fuse_rankings(['a'], dict(a=1), dict(a=1))

    def test_site_ids_share_locus_but_preserve_chain_and_insertion(self):
        task = self.task()
        mapping, sites = base_sites(task)
        self.assertEqual(len(mapping), 1)
        self.assertEqual(sites['one'][0]['site_id'], sites['two'][0]['site_id'])
        locus = task['candidates'][0]['sites'][0]
        self.assertNotEqual(site_identity('synthetic_assay', locus),
                            site_identity('synthetic_assay', {**locus, 'insertion_code': 'A'}))
        task['candidates'][1]['sites'][0]['chain'] = 'B'
        _, sites = base_sites(task)
        self.assertIn('insertion_code', sites['one'][0])
        self.assertNotEqual(sites['one'][0]['site_id'], sites['two'][0]['site_id'])

    def features(self, task):
        _, sites = base_sites(task)
        feature = dict(msa={}, structure={}, msa_chains={'A': dict(homolog_count=50, global_neff=22)},
                       structure_definition={'units': {'distance': 'A', 'sasa': 'A2'}})
        for cid, rows in sites.items():
            sid = rows[0]['site_id']
            feature['msa'][cid] = {sid: dict(zip(MSA_FIELDS, [0.2, 0.1, -0.6931471805599453, 20, 0.1, 0, 'ok']))}
            feature['msa'][cid][sid].update(msa_column=42, source_hash='audit-only', probabilities_20=[0.05]*20)
            feature['structure'][cid] = {sid: dict(zip(STRUCTURE_FIELDS,
                [None, None, None, None, 'missing_coordinates', 'missing_partner',
                 'unavailable', ['B'], 'frozen WT coordinates']))}
        return feature

    def test_prompt_deduplicates_chain_and_excludes_full_probability_table(self):
        task = self.task()
        rendered = render_evidence(task, ['one', 'two'], self.features(task), 'msa_prompt')
        self.assertEqual(len(rendered['msa_chains']), 1)
        text = json.dumps(rendered)
        for key in ('msa_column', 'source_hash', 'probabilities_20'):
            self.assertNotIn(key, text)
        self.assertEqual(rendered['msa_sites']['one'][0]['site_id'],
                         rendered['msa_sites']['two'][0]['site_id'])

    def test_structure_missing_is_null_and_site_identity_matches(self):
        task = self.task()
        rendered = render_evidence(task, ['one', 'two'], self.features(task), 'msa_prompt_structure')
        a = rendered['structure_sites']['one'][0]
        self.assertIsNone(a['sasa_bound_A2'])
        self.assertIsNone(a['min_partner_heavy_distance_A'])
        self.assertEqual(a['site_id'], rendered['msa_sites']['one'][0]['site_id'])

    def test_missing_evidence_field_fails_before_prompt(self):
        task = self.task()
        features = self.features(task)
        sid = next(iter(features['msa']['one']))
        del features['msa']['one'][sid]['weighted_gap_fraction']
        with self.assertRaises(ValueError):
            render_evidence(task, ['one', 'two'], features, 'msa_prompt')

    def test_unavailable_sasa_and_geometry_cannot_be_zero_imputed(self):
        task = self.task()
        for key in STRUCTURE_FIELDS[:4]:
            features = self.features(task)
            sid = next(iter(features['structure']['one']))
            features['structure']['one'][sid][key] = 0
            with self.assertRaises(ValueError):
                render_evidence(task, ['one', 'two'], features, 'structure')

    def test_extra_candidate_or_site_evidence_rejected(self):
        task = self.task()
        for mode in ('candidate', 'site'):
            features = self.features(task)
            if mode == 'candidate':
                features['msa']['extra'] = features['msa']['one']
            else:
                features['msa']['one']['unrelated_site'] = next(iter(features['msa']['one'].values()))
            with self.assertRaises(ValueError):
                render_evidence(task, ['one', 'two'], features, 'msa_prompt')

    def test_nonfinite_chain_depth_and_duplicate_prompt_ids_rejected(self):
        task = self.task()
        features = self.features(task)
        features['msa_chains']['A']['global_neff'] = float('nan')
        with self.assertRaises(ValueError):
            render_evidence(task, ['one', 'two'], features, 'msa_prompt')
        with self.assertRaises(ValueError):
            render_evidence(task, ['one', 'one'], self.features(task), 'msa_prompt')

    def test_postprocessing_never_becomes_prompt_condition(self):
        task = self.task()
        with self.assertRaises(ValueError):
            render_evidence(task, ['one', 'two'], self.features(task), 'msa_gemini_rank5050')

    def cache_fixture(self, root):
        task = self.task()
        request = dict(contents=[{'text': 'frozen prompt'}], generationConfig={'seed': 42})
        task_path, request_path, response_path = [root / x for x in ('task.json', 'request.json', 'response.json')]
        task_path.write_text(json.dumps(task)); request_path.write_text(json.dumps(request))
        unit = dict(id='u', source='synthetic', group_id='synthetic_assay', expected_ids=['one', 'two'],
                    task_sha256=file_hash(task_path), request_sha256=file_hash(request_path))
        response = dict(unit_id='u', status='ok', request_sha256=unit['request_sha256'],
                        scores={'one': 0.1, 'two': -0.1}, model='gemini-3.8-flash')
        response_path.write_text(json.dumps(response))
        return unit, task_path, request_path, response_path, request

    def test_cache_exact_match_and_changed_prompt_config_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.cache_fixture(Path(directory))
            scores, origin = matching_cache(*args)
            self.assertEqual(scores['one'], 0.1)
            self.assertEqual(origin['match'], 'exact_assay_candidates_prompt_configuration')
            for changed in (dict(contents=[{'text': 'changed MSA prompt'}], generationConfig={'seed': 42}),
                            dict(contents=[{'text': 'frozen prompt'}], generationConfig={'seed': 1})):
                with self.assertRaises(ValueError):
                    matching_cache(*args[:-1], changed)

    def test_cache_candidate_order_assay_and_amended_config_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.cache_fixture(Path(directory))
            for changed in ({**args[0], 'expected_ids': ['two', 'one']}, {**args[0], 'group_id': 'other'}):
                with self.assertRaises(ValueError):
                    matching_cache(changed, *args[1:])
            response_path = args[3]
            response = json.loads(response_path.read_text()); response['actual_request_sha256'] = 'changed'
            response_path.write_text(json.dumps(response))
            with self.assertRaises(ValueError):
                matching_cache(*args)

    def test_amendment_requires_explicit_exact_actual_request_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.cache_fixture(Path(directory))
            original_hash = args[0]['request_sha256']
            amended = copy.deepcopy(args[-1])
            amended['generationConfig']['responseMimeType'] = 'application/json'
            args[2].write_text(json.dumps(amended))
            actual_hash = file_hash(args[2])
            response = json.loads(args[3].read_text()); response['actual_request_sha256'] = actual_hash
            args[3].write_text(json.dumps(response))
            unit = {**args[0], 'request_sha256': actual_hash, 'original_request_sha256': original_hash}
            scores, origin = matching_cache(unit, *args[1:-1], amended)
            self.assertEqual(scores['one'], 0.1)
            self.assertEqual(origin['original_request_sha256'], original_hash)
            with self.assertRaises(ValueError):
                matching_cache(unit, *args[1:-1], args[-1])

    def test_cache_other_model_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.cache_fixture(Path(directory))
            response = json.loads(args[3].read_text())
            response['model'] = 'gemini-3.8-pro'
            args[3].write_text(json.dumps(response))
            with self.assertRaises(ValueError):
                matching_cache(*args)


if __name__ == '__main__':
    unittest.main()
