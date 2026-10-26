import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import protein_msa_guard as guard
import test_msa_missingness as fixtures
from msa_missingness import repair_protein_packet
from singlechain_catalog import sha


class ProteinGuardTests(unittest.TestCase):
    def fixture(self):
        q, p, f = fixtures.protein_fixture()
        return q, repair_protein_packet(q, p, f), f

    def test_repaired_nulls_and_observed_zero_pass(self):
        q, p, f = self.fixture()
        report = guard.inspect_packet(q, p, f)
        self.assertTrue(report['valid'])
        self.assertEqual(report['counts']['candidates_requiring_null'], 3)
        self.assertEqual(report['counts']['observed_zero_ratios'], 2)
        self.assertEqual(p['msa_only_scores']['M002'], 0.)

    def test_missing_ratio_zero_is_rejected(self):
        q, p, f = self.fixture()
        for cid in ('M001', 'M004'):
            bad = copy.deepcopy(p); bad['site_evidence'][cid][0]['site_log_frequency_ratio'] = 0.
            self.assertFalse(guard.inspect_packet(q, bad, f)['valid'])

    def test_partial_sum_rejected_even_when_prompt_site_fields_are_correct(self):
        q, p, f = self.fixture(); p['msa_only_scores']['M003'] = 4.5
        report = guard.inspect_packet(q, p, f)
        self.assertFalse(report['valid'])
        self.assertEqual(report['errors']['partial_evidence_has_candidate_score'], 1)

    def test_unmapped_numeric_placeholder_rejected_but_aligned_zero_depth_kept(self):
        q, p, f = self.fixture()
        self.assertEqual(p['site_evidence']['M004'][0]['non_gap_effective_depth'],0)
        self.assertTrue(guard.inspect_packet(q, p, f)['valid'])
        p['site_evidence']['M001'][0]['non_gap_effective_depth'] = 0
        self.assertFalse(guard.inspect_packet(q, p, f)['valid'])

    def test_unrounded_positive_depth_not_deleted(self):
        q, p, f = self.fixture(); f['positions']['2']['non_gap_effective_depth'] = .00001
        for cid, index in [('M002',0),('M003',1)]:
            p['site_evidence'][cid][index]['non_gap_effective_depth'] = 0.
        self.assertTrue(guard.inspect_packet(q, p, f)['valid'])

    def test_no_evidence_cannot_hide_numeric_scores(self):
        q, _, _ = self.fixture()
        self.assertTrue(guard.inspect_packet(q, {'status':'unavailable'}, None)['valid'])
        self.assertFalse(guard.inspect_packet(q, {'status':'unavailable','msa_only_scores':{'M001':0}}, None)['valid'])

    def test_evaluator_rejects_partial_sum_before_opening_labels(self):
        import evaluate_monomer_msa as evaluator
        q, p, f = self.fixture(); p['msa_only_scores']['M003'] = 4.5
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); bundle = root/'inputs/v'; ev = root/'evaluator/v'
            qp = bundle/'queries/protein/synthetic/query.json'; qp.parent.mkdir(parents=True)
            qp.write_text(json.dumps(q)); profile = root/'profile.json'; profile.write_text(json.dumps(f))
            p['source_profile'] = {'path':str(profile),'sha256':sha(profile)}
            ep = bundle/'evidence/protein/synthetic/msa.json'; ep.parent.mkdir(parents=True); ep.write_text(json.dumps(p))
            ev.mkdir(parents=True); (ev/'LABEL_REFERENCES.json').write_text(json.dumps([
                dict(domain='protein',assay='synthetic',query_path=str(qp),query_sha256=sha(qp),
                     label_path=str(root/'must_not_open_labels'),label_sha256='unused')]))
            with patch.object(evaluator,'validate_bundle',return_value=({'version':'v'}, [])):
                with self.assertRaisesRegex(ValueError,'Invalid Protein MSA'):
                    evaluator.evaluate(bundle, root/'result')
            self.assertFalse((root/'result').exists())


if __name__ == '__main__': unittest.main()
