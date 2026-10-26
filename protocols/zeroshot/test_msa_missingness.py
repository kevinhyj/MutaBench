import contextlib
import copy
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import prepare_singlechain
from prepare_ppi_missingness import prepare as prepare_ppi
from msa_missingness import (COMPLEX_FIELDS, MSA_MARKER, UNALIGNED, encode,
                             rebuild_ppi_user, rebuild_protein_msa_unit,
                             repair_complex_features, repair_complex_row,
                             repair_protein_packet, sha, split_ppi_user)


def protein_fixture():
    query = dict(domain='protein', assay='synthetic', wt_sequence='ACD', candidates=[
        dict(id='M001', sequence='GCD', mutation='A1G', changed_positions=[1]),
        dict(id='M002', sequence='ADD', mutation='C2D', changed_positions=[2]),
        dict(id='M003', sequence='GDD', mutation='A1G:C2D', changed_positions=[1, 2]),
        dict(id='M004', sequence='ACE', mutation='D3E', changed_positions=[3])])
    profile = dict(positions={str(i): dict(wt=wt, non_gap_effective_depth=depth, gap_fraction=gap,
                                         support='low') for i, wt, depth, gap in
                              [(1, 'A', 0., 1.), (2, 'C', 2., 0.), (3, 'D', 0., 1.)]})
    profile['positions']['1']['homolog_column'] = UNALIGNED
    evidence = {}
    for c in query['candidates']:
        evidence[c['id']] = [dict(position=pos, wt=query['wt_sequence'][pos-1], mutant=c['sequence'][pos-1],
                                  site_log_frequency_ratio=0., non_gap_effective_depth=profile['positions'][str(pos)]['non_gap_effective_depth'],
                                  gap_fraction=profile['positions'][str(pos)]['gap_fraction'], support='low')
                              for pos in c['changed_positions']]
    packet = dict(status='available', assay='synthetic', source_status='precomputed_ProteinGym_profile',
                  caveat='Frequency is not fitness.', site_evidence=evidence,
                  msa_only_scores={c['id']: 0. for c in query['candidates']})
    return query, packet, profile


def complex_fixture():
    return dict(p_wt=.2, p_mutant=.2, log_probability_ratio=0., weighted_valid_count=2.,
                weighted_gap_fraction=0., weighted_unknown_fraction=0., status='global_neff_below_frozen_20')


class MissingnessTests(unittest.TestCase):
    def test_unaligned_is_null_and_aligned_no_observations_preserves_real_counts(self):
        q, p, f = protein_fixture()
        r = repair_protein_packet(q, p, f)
        missing = r['site_evidence']['M001'][0]
        self.assertIsNone(missing['site_log_frequency_ratio'])
        self.assertIsNone(missing['non_gap_effective_depth'])
        self.assertIsNone(missing['gap_fraction'])
        self.assertEqual(missing['missing_reason'], 'query_insertion_unaligned')
        aligned = r['site_evidence']['M004'][0]
        self.assertIsNone(aligned['site_log_frequency_ratio'])
        self.assertEqual((aligned['non_gap_effective_depth'], aligned['gap_fraction']), (0., 1.))
        self.assertEqual(r['status'], 'partial')
        self.assertEqual(p['site_evidence']['M001'][0]['site_log_frequency_ratio'], 0.)

    def test_true_zero_low_support_and_tiny_positive_unrounded_depth_remain_valid(self):
        q, p, f = protein_fixture()
        f['positions']['2']['non_gap_effective_depth'] = .00001
        p['site_evidence']['M002'][0]['non_gap_effective_depth'] = 0.
        p['site_evidence']['M003'][1]['non_gap_effective_depth'] = 0.
        f['positions']['2']['gap_fraction'] = -4.44e-16
        r = repair_protein_packet(q, p, f)
        self.assertEqual(r['site_evidence']['M002'][0]['site_log_frequency_ratio'], 0.)
        self.assertEqual(r['site_evidence']['M002'][0]['support'], 'low')
        self.assertEqual(r['site_evidence']['M002'][0]['status'], 'available')
        self.assertEqual(r['msa_only_scores']['M002'], 0.)

    def test_multisite_score_is_not_partial_sum(self):
        q, p, f = protein_fixture()
        p['msa_only_scores']['M003'] = 4.5
        r = repair_protein_packet(q, p, f)
        self.assertIsNone(r['msa_only_scores']['M003'])
        self.assertEqual(r['candidate_msa_status']['M003'], 'partial')

    def test_missing_profile_position_and_wholly_unavailable_packet(self):
        q, p, f = protein_fixture()
        f['positions'].pop('2')
        r = repair_protein_packet(q, p, f)
        self.assertEqual(r['status'], 'unavailable')
        self.assertEqual(r['site_evidence']['M002'][0]['missing_reason'], 'position_absent_from_profile')

    def test_unknown_provenance_or_wt_mismatch_stops(self):
        q, p, f = protein_fixture()
        f['positions']['1']['homolog_column'] = 'unknown_mapping'
        with self.assertRaisesRegex(ValueError, 'Unknown homolog'):
            repair_protein_packet(q, p, f)
        f['positions']['1']['wt'] = 'V'
        with self.assertRaisesRegex(ValueError, 'WT mismatch'):
            repair_protein_packet(q, p, f)

    def test_singlechain_actual_prompt_and_identity_updated(self):
        q, p, f = protein_fixture()
        r = repair_protein_packet(q, p, f)
        unit = dict(logical_id='old', template=dict(system='system', expected_ids=['M001'],
                    user='prefix\n\n' + MSA_MARKER + encode({'site_evidence': {'M001': p['site_evidence']['M001']}}) + '\n\nstructure unchanged'))
        new = rebuild_protein_msa_unit(unit, r)
        block = json.loads(new['template']['user'].split(MSA_MARKER)[1].split('\n\n')[0])
        self.assertIsNone(block['site_evidence']['M001'][0]['site_log_frequency_ratio'])
        self.assertEqual(block['status'], 'unavailable')
        self.assertTrue(new['template']['user'].endswith('\n\nstructure unchanged'))
        self.assertNotEqual(new['logical_id'], 'old')

    def test_ppi_missing_placeholder_fields_null_but_low_neff_row_unchanged(self):
        missing = dict.fromkeys(COMPLEX_FIELDS, 0.)
        missing.update(status='no_valid_homolog_at_column', weighted_gap_fraction=1.)
        self.assertTrue(all(repair_complex_row(missing)[k] is None for k in COMPLEX_FIELDS))
        observed = complex_fixture()
        self.assertEqual(repair_complex_row(observed), observed)

    def test_ppi_unsupported_status_and_invalid_observed_values_rejected(self):
        row = complex_fixture()
        row['status'] = 'unknown'
        with self.assertRaisesRegex(ValueError, 'Unknown complex'):
            repair_complex_row(row)
        row['status'] = 'ok'; row['weighted_valid_count'] = 0
        with self.assertRaisesRegex(ValueError, 'no valid support'):
            repair_complex_row(row)

    def test_ppi_entrypoint_updates_saved_feature_request_and_queue(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); old = root/'old'; old.mkdir()
            group = 'synthetic'; missing = dict(complex_fixture(), status='no_valid_homolog_at_column',
                                              weighted_valid_count=0., weighted_gap_fraction=1.,
                                              p_wt=None, p_mutant=None, log_probability_ratio=None)
            features = dict(msa={'M001': {'s1': missing, 's2': complex_fixture()}},
                            structure={'unchanged': True}, msa_chains={'A': {'global_neff': 2}},
                            independent_msa_scores={'M001': None}, full_msa_coverage=False)
            task = dict(candidates=[dict(id='M001', sites=[])])
            context = dict(candidates=task['candidates'], binding_partners={'unchanged': True},
                           evidence={'msa_sites': {'M001': [dict(site_id=sid, **r) for sid, r in features['msa']['M001'].items()]}})
            request = dict(contents=[dict(parts=[dict(text='prefix\n' + encode(context))])], generationConfig={'unchanged': True})
            artifacts = {'prepared/synthetic.json': task, 'features/synthetic.json': features,
                         'requests/unit.json': request}
            for name, data in artifacts.items():
                p = old/name; p.parent.mkdir(parents=True, exist_ok=True); p.write_text(encode(data))
            plan = [dict(id='unit', group_id=group, condition='msa_prompt', batch=0, expected_ids=['M001'],
                         request_path='requests/unit.json', request_sha256=sha(old/'requests/unit.json'))]
            (old/'plan.json').write_text(encode(plan))
            hashes = {name: sha(old/name) for name in [*artifacts, 'plan.json']}
            (old/'freeze.json').write_text(encode(dict(artifact_hashes=hashes)))
            manifest = prepare_ppi(root/'new', [old])
            queue = json.loads((root/'new/QUEUE.json').read_text())
            _, shown, _ = split_ppi_user(queue[0]['template']['user'])
            self.assertIsNone(shown['evidence']['msa_sites']['M001'][0]['weighted_gap_fraction'])
            self.assertEqual({k: shown['evidence']['msa_sites']['M001'][1][k] for k in complex_fixture()}, complex_fixture())
            self.assertTrue(all(sha(root/'new'/name) == h for name, h in manifest['files'].items()))
            self.assertTrue(all(sha(old/name) == h for name, h in hashes.items()))
            with self.assertRaises(FileExistsError):
                prepare_ppi(root/'new', [old])

    def test_singlechain_entrypoint_serializes_partial_msa_without_bypassing_rna_fix(self):
        q, p, f = protein_fixture()
        # One synthetic multi-site candidate; real preparation sampler/cleaner.
        q['candidates'] = [dict(q['candidates'][2], id='M001')]
        p['site_evidence'] = {'M001': p['site_evidence']['M003']}; p['msa_only_scores'] = {'M001': 0.}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); old, here = root/'old', root/'pipeline'; old.mkdir(); here.mkdir()
            shutil.copyfile(prepare_singlechain.OLD/'prepare.py', old/'prepare.py')
            for name in ['rna_confidence.py', 'msa_missingness.py', 'sampling.py', 'sampling_contract.json']:
                shutil.copyfile(prepare_singlechain.HERE/name, here/name)
            profile = old/'profile.json'; profile.write_text(encode(f))
            p['source_profile'] = {'path': str(profile), 'sha256': sha(profile)}
            csv = old/'source.csv'; csv.write_text('mutant,mutated_sequence,DMS_score\nA1G:C2D,GDD,1.0\n')
            item = dict(assay='synthetic', wt_sequence='ACD', msa_profile=p['source_profile'],
                        source_csv={'path': str(csv), 'sha256': sha(csv)})
            qpath = old/'queries/protein/synthetic/query.json'; qpath.parent.mkdir(parents=True)
            qpath.write_text(encode(q)); (qpath.parent/'evaluation_labels.json').write_text('{"labels":{"M001":1.0}}')
            ep = old/'evidence/protein/synthetic'; ep.mkdir(parents=True)
            (ep/'msa.json').write_text(encode(p)); (ep/'structure.json').write_text('{"status":"unavailable"}')
            (old/'INVENTORY.json').write_text(encode({'protein': [item], 'rna': []}))
            unit = dict(domain='protein', assay='synthetic', arm='sequence_msa_prompt', chunk_index=0,
                        logical_id='old', template=dict(system='system', expected_ids=['M001'],
                        user=MSA_MARKER + encode({'site_evidence': p['site_evidence']}) + '\n\noutput'))
            (old/'QUEUE.json').write_text(encode([unit]))
            (old/'PLAN.json').write_text(encode(dict(queue_sha256=sha(old/'QUEUE.json'), input_hashes={
                'protein/synthetic': {'query': sha(qpath), 'msa': sha(ep/'msa.json'), 'structure': sha(ep/'structure.json')}})))
            with patch.object(prepare_singlechain, 'HERE', here), patch.object(prepare_singlechain, 'OLD', old), \
                    patch('sys.argv', ['prepare_singlechain.py', '--version', 'v1']), contextlib.redirect_stdout(io.StringIO()):
                prepare_singlechain.main()
            new = json.loads((here/'inputs/v1/evidence/protein/synthetic/msa.json').read_text())
            self.assertEqual(new['status'], 'partial'); self.assertIsNone(new['msa_only_scores']['M001'])
            queue = json.loads((here/'inputs/v1/QUEUE.json').read_text())
            unit = next(u for u in queue if u['arm']=='msa_prompt')
            shown = json.loads(unit['template']['user'])['evidence']['msa']
            self.assertEqual(shown['status'], 'partial')
            self.assertEqual([s['features'] for s in shown['variants'][0]['sites']],
                [{k:v for k,v in s.items() if k not in {'position','wt','mutant'}} for s in new['site_evidence']['M001']])


if __name__ == '__main__':
    unittest.main()
