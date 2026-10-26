import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from rna_msa_missingness import repair, load_and_repair, MISSING_REASON
from rna_msa_guard import inspect_packet, check_packet
from singlechain_catalog import sha


def fixture():
    query=dict(domain='rna',assay='synthetic',wt_sequence='AA',candidates=[
        dict(id='M001',sequence='CC',mutation='A1C,A2C',changed_positions=[1,2]),
        dict(id='M002',sequence='AC',mutation='A2C',changed_positions=[2])])
    profile=dict(status='available',positions={str(i):dict(wt='A',frequency=dict.fromkeys('ACGU',.25),
        non_gap_effective_depth=depth,gap_fraction=gap) for i,depth,gap in [(1,0.,1.),(2,2.,0.)]})
    sites={c['id']:[dict(position=p,wt='A',mutant='C',site_log_frequency_ratio=0.,
        non_gap_effective_depth=profile['positions'][str(p)]['non_gap_effective_depth'],
        gap_fraction=profile['positions'][str(p)]['gap_fraction'],support='low') for p in c['changed_positions']]
        for c in query['candidates']}
    packet=dict(assay='synthetic',status='available',site_evidence=sites,msa_only_scores={'M001':0.,'M002':0.})
    return query,packet,profile


class RNAGuardTests(unittest.TestCase):
    def test_pseudocount_zero_removed_observed_zero_kept(self):
        q,p,f=fixture();r=repair(q,p,f)
        self.assertEqual(r['site_evidence']['M001'][0]['reason'],MISSING_REASON)
        self.assertNotIn('site_log_frequency_ratio',r['site_evidence']['M001'][0])
        self.assertIsNone(r['msa_only_scores']['M001']);self.assertEqual(r['msa_only_scores']['M002'],0.)
        self.assertTrue(inspect_packet(q,r,f)['valid']);self.assertEqual(p['msa_only_scores']['M001'],0.)

    def test_repair_idempotent_for_partial_and_all_missing(self):
        q,p,f=fixture()
        for depth in (2.,0.):
            f['positions']['2']['non_gap_effective_depth']=depth
            for rows in p['site_evidence'].values():
                for s in rows:
                    if s['position']==2:s['non_gap_effective_depth']=depth
            r=repair(q,p,f);self.assertEqual(repair(q,r,f),r)

    def test_positive_raw_depth_rounded_to_zero_is_recovered(self):
        q,p,f=fixture();f['positions']['2']['non_gap_effective_depth']=.00001
        for rows in p['site_evidence'].values():
            for s in rows:
                if s['position']==2:s['non_gap_effective_depth']=0.
        r=repair(q,p,f)
        self.assertEqual(r['site_evidence']['M002'][0]['non_gap_effective_depth'],.00001)
        self.assertEqual(r['msa_only_scores']['M002'],0.)
        self.assertEqual(repair(q,r,f),r);self.assertTrue(inspect_packet(q,r,f)['valid'])

    def test_partial_score_and_numeric_missing_fields_rejected(self):
        q,p,f=fixture();r=repair(q,p,f)
        bad=copy.deepcopy(r);bad['msa_only_scores']['M001']=1.2
        self.assertFalse(inspect_packet(q,bad,f)['valid'])
        bad=copy.deepcopy(r);bad['site_evidence']['M001'][0]['site_log_frequency_ratio']=0.
        self.assertFalse(inspect_packet(q,bad,f)['valid'])

    def test_status_alone_does_not_bypass_repair(self):
        q,p,f=fixture();p['status']='unavailable'
        r=repair(q,p,f);self.assertIsNone(r['msa_only_scores']['M001'])
        self.assertTrue(inspect_packet(q,r,f)['valid'])

    def test_wholly_unavailable_packet_cannot_hide_score(self):
        q,_,_=fixture();p={'status':'unavailable'}
        self.assertEqual(repair(q,p,None),p)
        with self.assertRaisesRegex(ValueError,'aggregate without evidence'):
            repair(q,dict(p,msa_only_scores={'M001':0}),None)

    def test_hash_checked_nested_profile_and_repeat_load(self):
        q,p,f=fixture()
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'profile.json';path.write_text(json.dumps({'sources':{'msa':f}}))
            p['source_profile']={'path':str(path),'sha256':sha(path)}
            r=load_and_repair(q,p,Path);self.assertEqual(load_and_repair(q,r,Path),r)
            self.assertTrue(check_packet(q,r)['valid']);path.write_text('{}')
            with self.assertRaisesRegex(ValueError,'changed'):check_packet(q,r)

    def test_identity_mismatch_rejected(self):
        q,p,f=fixture();p['site_evidence']['M001'][0]['wt']='G'
        with self.assertRaisesRegex(ValueError,'WT mismatch'):repair(q,p,f)

    def test_evaluator_rejects_before_reading_labels(self):
        import evaluate_monomer_msa as evaluator
        q,p,f=fixture()
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);bundle=root/'inputs/v';ev=root/'evaluator/v'
            qp=bundle/'queries/rna/synthetic/query.json';qp.parent.mkdir(parents=True);qp.write_text(json.dumps(q))
            fp=root/'profile.json';fp.write_text(json.dumps(f));p['source_profile']={'path':str(fp),'sha256':sha(fp)}
            ep=bundle/'evidence/rna/synthetic/msa.json';ep.parent.mkdir(parents=True);ep.write_text(json.dumps(p))
            ev.mkdir(parents=True);(ev/'LABEL_REFERENCES.json').write_text(json.dumps([
                dict(domain='rna',assay='synthetic',query_path=str(qp),query_sha256=sha(qp),
                     label_path=str(root/'must_not_open'),label_sha256='unused')]))
            with patch.object(evaluator,'validate_bundle',return_value=({'version':'v'},[])):
                with self.assertRaisesRegex(ValueError,'Invalid RNA MSA'):evaluator.evaluate(bundle,root/'result')
            self.assertFalse((root/'result').exists())


if __name__=='__main__':unittest.main()
