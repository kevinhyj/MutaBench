"""Independent numerical, identity, source-replay and no-network integration tests."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import admet_protocol as admet
import monomer_campaign as campaign
from evaluate_monomers import metric, merge_predictions, paired_summary, evaluate
from shared_protocol import save, load, sha


def fixture_tasks(n=26):
    # Two different row IDs deliberately share a SMILES.
    return [dict(endpoint='synthetic',kind='classification',description='Synthetic positive class.',
                 candidates=[dict(id=f'M{i:03d}',smiles='CCO') for i in range(n)])]


def fixture_shapes(tasks):
    return {t['endpoint']:{c['id']:dict(status='available',conformers=1,optimization='MMFF94',
        shape=dict(NPR1=.2,NPR2=.8,RadiusOfGyration=1.5,Asphericity=.1,SpherocityIndex=.3,PBF=.2))
        for c in t['candidates']} for t in tasks}


def fixture_groups(tasks):
    return [dict(endpoint=t['endpoint'],round=r,group=g,ids=[c['id'] for c in t['candidates'][s:s+25]])
            for t in tasks for r in range(3) for g,s in enumerate(range(0,len(t['candidates']),25))]


class MetricTests(unittest.TestCase):
    def test_auc_known_ties_and_direction(self):
        self.assertEqual(metric('classification',[0,0,1,1],[0,1,1,2])[0],.875)
        self.assertEqual(metric('classification',[0,1,0,1],[0,0,0,0])[0],.5)
        self.assertEqual(metric('classification',[0,1],[1,-1])[0],0)
        self.assertIsNone(metric('classification',[1,1],[0,1])[0])

    def test_auc_against_independent_pairwise_reference(self):
        import numpy as np
        rng=np.random.RandomState(42)
        for _ in range(40):
            labels=rng.randint(0,2,51).tolist();scores=rng.randint(-3,4,51).tolist()
            positive=[s for y,s in zip(labels,scores) if y==1]
            negative=[s for y,s in zip(labels,scores) if y==0]
            expected=sum((a>b)+.5*(a==b) for a in positive for b in negative)/(len(positive)*len(negative))
            self.assertAlmostEqual(metric('classification',labels,scores)[0],expected,places=14)

    def test_spearman_signed_constant_nonfinite(self):
        self.assertAlmostEqual(metric('regression',[1,2,3],[3,2,1])[0],-1)
        self.assertIsNone(metric('regression',[1,2],[0,0])[0])
        for bad in [True,float('nan'),float('inf')]:
            with self.assertRaises(ValueError):metric('regression',[1,2],[0,bad])
        with self.assertRaises(ValueError):metric('classification',[0,2],[0,1])

    def test_common_query_and_zero_coverage_model(self):
        rows=[dict(model_key=m,domain='rna',arm='sequence',aggregation='single_pass',metric='spearman',
            status='complete',cohort='N250',assay='a',query_sha256=h,value=v)
            for m,h,v in [('m1','same',-.5),('m2','different',.8)]]
        self.assertEqual(paired_summary(rows)[0]['common_assays'],0)
        rows[1]['query_sha256']='same'
        self.assertEqual(paired_summary(rows)[0]['models']['m1'],-.5)
        rows[1]['status']='incomplete';rows[1]['value']=None
        self.assertEqual(paired_summary(rows)[0]['common_assays'],0)

    def test_rank_fusion_ties_identity_and_missing(self):
        from evaluate_monomer_msa import rank_fusion
        self.assertEqual(rank_fusion(['a','b','c'],{'a':0,'b':0,'c':1},{'a':3,'b':2,'c':1}),
                         {'a':2.25,'b':1.75,'c':2.0})
        with self.assertRaises(ValueError):rank_fusion(['a'],{'a':0},{'a':None})
        with self.assertRaises(ValueError):rank_fusion(['a'],{'a':0},{'b':0})


class RNAMissingnessTests(unittest.TestCase):
    def test_zero_depth_is_unknown_but_observed_zero_ratio_stays(self):
        from rna_msa_missingness import repair
        q=dict(domain='rna',assay='synthetic',wt_sequence='AA',candidates=[dict(id='M1',sequence='CC',changed_positions=[1,2])])
        packet=dict(status='available',assay='synthetic',msa_only_scores={'M1':0},site_evidence={'M1':[
            dict(position=i,wt='A',mutant='C',non_gap_effective_depth=d,site_log_frequency_ratio=0,gap_fraction=0,support='low')
            for i,d in [(1,0),(2,1)]]})
        profile=dict(positions={str(i):dict(wt='A',frequency={'A':.25,'C':.25},non_gap_effective_depth=d) for i,d in [(1,0),(2,1)]})
        repaired=repair(q,packet,profile)
        self.assertEqual(repaired['site_evidence']['M1'][0]['status'],'unavailable')
        self.assertEqual(repaired['site_evidence']['M1'][1]['site_log_frequency_ratio'],0)
        self.assertIsNone(repaired['msa_only_scores']['M1'])
        self.assertEqual(packet['msa_only_scores']['M1'],0)
        profile['positions']['1']['non_gap_effective_depth']=.0004
        repaired=repair(q,packet,profile)
        self.assertEqual(repaired['site_evidence']['M1'][0]['non_gap_effective_depth'],.0004)
        self.assertEqual(repaired['msa_only_scores']['M1'],0)


class RenderingTests(unittest.TestCase):
    def setUp(self):
        self.tasks=fixture_tasks();self.shapes=fixture_shapes(self.tasks);self.groups=fixture_groups(self.tasks)

    def render(self): return admet.render(self.tasks,self.shapes,self.groups,'same-system')

    def test_complete_paired_rounds_duplicates_and_last_chunk(self):
        queue=self.render();self.assertEqual(len(queue),12)
        for a,b in zip(queue[::2],queue[1::2]):
            x,y=json.loads(a['template']['user']),json.loads(b['template']['user'])
            x.pop('evidence');y.pop('evidence');self.assertEqual(x,y)
        self.assertEqual(len(queue[2]['template']['expected_ids']),1)
        self.assertNotIn('low-energy',queue[1]['template']['user'])
        self.assertEqual(len({c['smiles'] for c in self.tasks[0]['candidates']}),1)
        self.assertEqual(len(queue[0]['template']['expected_ids']),25)

    def test_reject_missing_round_duplicate_group_and_labels(self):
        for mutate in [lambda:self.groups.pop(),lambda:self.groups.append(copy.deepcopy(self.groups[0])),
            lambda:self.tasks[0]['candidates'][0].update(y=1),lambda:self.groups[0]['ids'].append('M000')]:
            self.setUp();mutate()
            with self.assertRaises(ValueError):self.render()

    def test_shape_missing_zero_bad_ranges(self):
        self.shapes['synthetic']['M000']=dict(status='unavailable',reason='invalid_smiles')
        self.shapes['synthetic']['M001']['shape']['PBF']=0
        self.assertEqual(json.loads(self.render()[1]['template']['user'])['evidence']['structure']['status'],'partial')
        self.shapes['synthetic']['M001']['shape']['NPR1']=1.5
        with self.assertRaises(ValueError):self.render()


class IntegrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        self.here=self.root/'pipeline';self.here.mkdir()
        tasks=fixture_tasks();shapes=fixture_shapes(tasks);groups=fixture_groups(tasks)
        data=self.root/'data/admet_anewdde';save(data/'frozen/tasks.public.json',tasks)
        labels={'synthetic':{'y':{c['id']:i%2 for i,c in enumerate(tasks[0]['candidates'])}}}
        save(data/'labels/labels.local.json',labels)
        save(data/'manifest.json',{'hashes':{n:sha(data/n) for n in ['frozen/tasks.public.json','labels/labels.local.json']}})
        baseline=self.root/'experiments/admet_cr_elo_v1/tasks.public.json';save(baseline,tasks)
        for part in ['run','remaining']:
            folder=self.root/'experiments/admet_cr_structure_v1'/part
            save(folder/'features.json',shapes if part=='run' else {})
            save(folder/'queue.json',[dict(r,method='cr') for r in groups] if part=='run' else [])
            save(folder/'plan.json',dict(hashes={n:sha(folder/n) for n in ['features.json','queue.json']},
                                      baseline_hashes={'tasks.public.json':sha(baseline)}))
        self.patch=patch.object(admet,'ROOT',self.root);self.patch.start();self.addCleanup(self.patch.stop)
        self.source=self.here/'test_source.py';self.source.write_text('# test implementation\n')
        self.source_map={str(self.source):sha(self.source)}
        self.patch2=patch.object(campaign,'HERE',self.here);self.patch2.start();self.addCleanup(self.patch2.stop)
        self.patch3=patch.object(campaign,'sources',return_value=self.source_map);self.patch3.start();self.addCleanup(self.patch3.stop)
        self.model=dict(provider='google',model='synthetic',thinking_level='medium',temperature=0,
            seed=42,timeout_s=3600,automatic_retries=0,stream=True,max_output_tokens=32,endpoint='https://example.invalid')

    def prepare(self):
        # No label file may be parsed by the builder or campaign prepare.
        original=admet.load
        def guarded(p):
            if 'labels.local' in str(p):raise AssertionError('predictor read labels')
            return original(p)
        with patch.object(admet,'load',side_effect=guarded):admet.prepare('synthetic',here=self.here)
        self.bundle=self.here/'inputs/synthetic';self.run=self.here/'runs/test'
        campaign.prepare(self.run,[self.bundle],{'model':self.model},2)

    def test_prepare_replay_tamper_and_complete_evaluation(self):
        self.prepare();plan,requests=campaign.validate(self.run)
        self.assertEqual(plan['planned_requests'],12)
        for r in requests:
            value={i:float(int(i[1:])%2) for i in r['expected_ids']}
            campaign.write_unit(self.run,'model',r,dict(status='ok',failure_category='ok',
                request_id=r['request_id'],messages_sha256=r['messages_sha256']),value)
        rows=evaluate(self.run)
        self.assertEqual(len(rows),4)
        self.assertTrue(all(r['metric']=='auroc' and r['value']==1 for r in rows))
        path=self.run/'models/model/units'/(requests[-1]['request_id']+'.json')
        path.unlink();rows=evaluate(self.run)
        self.assertEqual(sum(r['status']=='incomplete' for r in rows),1)
        self.assertIsNone(next(r for r in rows if r['status']=='incomplete')['value'])
        (self.bundle/'groups.json').write_text('[]')
        with self.assertRaises(ValueError):campaign.validate(self.run)

    def test_changed_source_rejected_even_with_rehashed_normalized_file(self):
        self.prepare();m=load(self.bundle/'MANIFEST.json')
        shapes=load(self.bundle/'shapes.json');shapes['synthetic']['M000']['shape']['PBF']=.8
        save(self.bundle/'shapes.json',shapes)
        m['files']['shapes.json']=sha(self.bundle/'shapes.json');save(self.bundle/'MANIFEST.json',m)
        (self.bundle/'MANIFEST.sha256').write_text(sha(self.bundle/'MANIFEST.json'))
        with self.assertRaisesRegex(ValueError,'source'):admet.validate_bundle(self.bundle)

    def test_saved_unsettled_attempt_never_resubmitted(self):
        self.prepare();_,requests=campaign.validate(self.run);r=requests[0]
        folder=self.run/'models/model/attempts'/r['request_id'];folder.mkdir(parents=True)
        with patch('multillm_transport.submit',side_effect=AssertionError('must not send')):
            result=campaign.submit_one(self.run,'model',self.model,r)
        self.assertEqual(result['failure_category'],'unsettled_saved_attempt')

    def test_long_wait_and_credential_fields(self):
        for update in [dict(timeout_s=180),dict(api_key='not-a-key'),dict(seed=1),dict(automatic_retries=1)]:
            with self.assertRaises(ValueError):campaign.validate_models({'x':dict(self.model,**update)})

    def test_dispatch_restart_reuses_attempts_and_preflight(self):
        import threading
        self.prepare();calls=[];counter=[0,0];lock=threading.Lock()
        def fake_submit(folder,model,request):
            with lock:
                counter[0]+=1;counter[1]=max(counter)
            folder.mkdir(parents=True,exist_ok=False);calls.append(request['request_id'])
            account=dict(status='ok',failure_category='ok',request_id=request['request_id'],
                         messages_sha256=request['messages_sha256'])
            values={i:0.0 for i in request['expected_ids']}
            save(folder/'ACCOUNTING.json',account);save(folder/'RESULT.json',dict(status='ok',value=values))
            with lock:counter[0]-=1
            return account,values
        with patch('multillm_transport.submit',side_effect=fake_submit):
            campaign.run_campaign(self.run)
            self.assertEqual(len(calls),13)
            campaign.run_campaign(self.run)
        self.assertEqual(len(calls),13)
        self.assertLessEqual(counter[1],2)
        self.assertEqual(load(self.run/'STATUS.json')['state'],'finished')

    def test_pause_prevents_all_submissions(self):
        self.prepare();(self.run/'PAUSE').write_text('test')
        with patch('multillm_transport.submit',side_effect=AssertionError('must not send')):
            campaign.run_campaign(self.run)
        status=load(self.run/'STATUS.json')
        self.assertEqual(status['completed'],0);self.assertEqual(status['held'],12)


if __name__=='__main__':unittest.main()
