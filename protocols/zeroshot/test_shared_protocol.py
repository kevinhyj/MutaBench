"""Synthetic six-domain integration and adversarial acceptance tests, no API."""
import argparse
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import shared_protocol as protocol
import prepare_shared_prompts as builder
import run_frozen as runner
import multillm_campaign as campaign
import multillm_supervise as supervisor


def single_sources(domain='protein', n=4):
    wt = 'A'*n
    candidates = [dict(id=f'M{i:03}', mutation=f'A{i}G', sequence=wt[:i-1]+'G'+wt[i:],
                       changed_positions=[i]) for i in range(1,n+1)]
    query = dict(domain=domain,assay='synthetic',wt_sequence=wt,candidates=candidates,assay_context={})
    msa = dict(status='available',site_evidence={c['id']:[dict(position=i,wt='A',mutant='G',
        site_log_frequency_ratio=0.,non_gap_effective_depth=2.,gap_fraction=0.,support='low',
        **(dict(status='available',missing_reason=None) if domain=='protein' else {}))]
        for i,c in enumerate(candidates,1)})
    structure = dict(status='unavailable',reason='synthetic_no_structure')
    return dict(query=query,msa=msa,structure=structure)


def complex_sources(domain):
    from prepare_protein_rna_contact import base_sites
    query=dict(group_id='synthetic',wt_chains=[dict(chain='A',polymer_type='protein',resolved_sequence='A',residue_ids=['1'])],
        candidates=[dict(id='M001',sites=[dict(chain='A',position=1,sequence_index_1based=1,wt='A',mutant='G')])])
    _,sites=base_sites(query);sid=sites['M001'][0]['site_id']
    context=dict(task=dict(dataset_id='synthetic',assay_id='synthetic',modality=domain,
        score_target=dict(higher_score_means='higher synthetic readout')),
        reference=dict(chains=[dict(chain_id='A',molecule_type='protein',sequence='A')],
            partners=dict(group_a=['A'],group_b=['B'])))
    if domain=='protein_ligand':context['reference']['ligand']=dict(smiles='C',ccd_id='SYN')
    msa=dict(msa={'M001':{sid:dict(status='ok',p_wt=.2,p_mutant=.2,log_probability_ratio=0.,
        weighted_valid_count=2.,weighted_gap_fraction=0.,weighted_unknown_fraction=0.)}})
    structure=dict(structure={'M001':{sid:dict(site_status='ok',partner_status='ok',sasa_status='ok',
        min_partner_heavy_distance_A=9.,partner_contact_count_5A=0,sasa_bound_A2=0.,delta_sasa_partner_A2=0.)}})
    if domain=='protein_rna':
        from zeroshot_msa import POLICY_VERSION
        msa['msa_scoring_policy'] = {'version': POLICY_VERSION}
        structure=dict(sites=[dict(site_id=sid,interface_geometry_status='computed',partner_contact_count_5A=0,
            min_partner_heavy_distance_A=9.,rna_moiety_status='complete',
            rna_contact_count_5A_by_moiety=dict(base=0,ribose=0,phosphate=0))])
    return dict(query=query,msa=msa,structure=structure,context=context)


def make_bundle(root, domains=('protein',), n=4):
    bundle=root/'inputs'/'v'; names=[];queue=[];refs=[]
    for domain in domains:
        source=single_sources(domain,n) if domain in {'protein','rna'} else complex_sources(domain)
        name,units=builder.add_assay(bundle,domain,'synthetic',source)
        names.append(name);queue.extend(units)
        qp=bundle/'queries'/domain/'synthetic'/'query.json'
        refs.append(dict(domain=domain,assay='synthetic',query_path=str(qp),query_sha256=protocol.sha(qp),
                         label_path='/never/open/labels.json',label_sha256='f'*64))
    builder.finish(bundle,'v',names,queue,{'synthetic_source':'0'*64})
    protocol.save(root/'evaluator/v/LABEL_REFERENCES.json',refs)
    return bundle


def rehash_manifest(bundle):
    m=protocol.load(bundle/'MANIFEST.json')
    for name in m['files']:m['files'][name]=protocol.sha(bundle/name)
    m['acceptance']['sha256']=protocol.sha(bundle/'ACCEPTANCE.json')
    protocol.save(bundle/'MANIFEST.json',m)
    (bundle/'MANIFEST.sha256').write_text(protocol.sha(bundle/'MANIFEST.json')+'\n')


class SharedTests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup);self.root=Path(temp.name)

    def test_six_domains_same_system_four_arms_and_fixed_chunks(self):
        bundle=make_bundle(self.root,protocol.DOMAINS,n=26)
        manifest,queue=protocol.validate_bundle(bundle)
        self.assertEqual(set(manifest['domains']),set(protocol.DOMAINS))
        self.assertEqual(len(queue),32)  # Two singlechains x8, four complexes x4.
        for domain in protocol.DOMAINS:
            for chunk in {q['chunk_index'] for q in queue if q['domain']==domain}:
                rows=[q for q in queue if (q['domain'],q['chunk_index'])==(domain,chunk)]
                self.assertEqual({r['arm'] for r in rows},set(protocol.ARMS))
                users=[json.loads(r['template']['user']) for r in rows]
                bases=[{k:v for k,v in u.items() if k!='evidence'} for u in users]
                self.assertTrue(all(b==bases[0] for b in bases))
                for row,user in zip(rows,users):
                    self.assertEqual(row['template']['system'],protocol.SYSTEM_PATH.read_text())
                    for kind in ('msa','structure'):
                        self.assertEqual(user['evidence'][kind]['status']=='not_requested',kind not in row['arm'])
        self.assertEqual(len(protocol.load_requests(bundle)),32)

    def test_ready_must_be_boolean_true_and_acceptance_required(self):
        bundle=make_bundle(self.root);original=protocol.load(bundle/'MANIFEST.json')
        for value in (None,False,1,'true'):
            m=copy.deepcopy(original)
            if value is None:m.pop('fully_ready')
            else:m['fully_ready']=value
            protocol.save(bundle/'MANIFEST.json',m);rehash_manifest(bundle)
            with self.assertRaisesRegex(ValueError,'fully_ready'):protocol.validate_bundle(bundle)
        m=copy.deepcopy(original);m.pop('acceptance');protocol.save(bundle/'MANIFEST.json',m)
        (bundle/'MANIFEST.sha256').write_text(protocol.sha(bundle/'MANIFEST.json'))
        with self.assertRaisesRegex(ValueError,'acceptance'):protocol.validate_bundle(bundle)

    def test_rehashed_legacy_prompt_and_missing_arm_rejected(self):
        bundle=make_bundle(self.root);original=protocol.load(bundle/'QUEUE.json')
        for mode in ('legacy','missing_arm','changed_evidence'):
            queue=copy.deepcopy(original)
            if mode=='legacy':queue[0]['template']['user']='old copied prompt'
            elif mode=='missing_arm':queue.pop()
            else:
                user=json.loads(queue[1]['template']['user']);user['evidence']['msa']['variants'][0]['sites'][0]['features']['site_log_frequency_ratio']=1
                queue[1]['template']['user']=protocol.encode(user)
            protocol.save(bundle/'QUEUE.json',queue);rehash_manifest(bundle)
            with self.assertRaisesRegex(ValueError,'Requests differ'):protocol.validate_bundle(bundle)

    def test_stale_report_rejected_even_with_new_manifest_hash(self):
        bundle=make_bundle(self.root);report=protocol.load(bundle/'ACCEPTANCE.json');report['feature_validation']=False
        protocol.save(bundle/'ACCEPTANCE.json',report);rehash_manifest(bundle)
        with self.assertRaisesRegex(ValueError,'stale'):protocol.validate_bundle(bundle)

    def test_record_is_replayed_from_sources(self):
        bundle=make_bundle(self.root);path=bundle/protocol.load(bundle/'RECORDS.json')[0]
        record=protocol.load(path);record['base']['task']['score_target']['higher_score_means']='opposite direction'
        protocol.save(path,record);rehash_manifest(bundle)
        with self.assertRaisesRegex(ValueError,'source adapter'):protocol.validate_bundle(bundle)

    def test_missingness_confidence_contact_and_semantics_are_enforced(self):
        source=single_sources();record=builder.record_from_sources('protein','synthetic',source)
        f=record['evidence']['msa']['variants'][0]['sites'][0]['features'];f.update(homolog_column='query_insertion_unaligned_uniform_pseudocount_only')
        with self.assertRaisesRegex(ValueError,'placeholders'):protocol.render(record)
        source=single_sources('rna');source['structure']=dict(status='available',kind='predicted_WT',site_evidence={
            c['id']:[dict(position=i,plddt=0,anchor_atom='P',nonlocal_neighbors_8a=0,nearest_nonlocal_positions=[])]
            for i,c in enumerate(source['query']['candidates'],1)})
        with self.assertRaisesRegex(ValueError,'Unverified RNA'):protocol.render(builder.record_from_sources('rna','synthetic',source))
        source=complex_sources('protein_rna');record=builder.record_from_sources('protein_rna','synthetic',source)
        self.assertIs(record['evidence']['structure']['variants'][0]['sites'][0]['features']['direct_rna_contact'],False)
        record['evidence']['structure']['definition']['fields']['direct_rna_contact']='sidechain only'
        with self.assertRaisesRegex(ValueError,'definitions'):protocol.render(record)

    def test_source_spec_builder_and_evaluator_reference_binding(self):
        source=complex_sources('protein_protein');source['task']=source['query']
        source['query']=dict(domain='protein_protein',assay='synthetic',candidates=source['task']['candidates'])
        descs={}
        for key,value in source.items():
            p=self.root/(key+'.json');protocol.save(p,value);descs[key]=dict(path=p.name,sha256=protocol.sha(p))
        ref=self.root/'refs.json';protocol.save(ref,[dict(domain='protein_protein',assay='synthetic',query_sha256=descs['query']['sha256'],label_path='/unread',label_sha256='f'*64)])
        spec=self.root/'spec.json';protocol.save(spec,dict(assays=[dict(domain='protein_protein',assay='synthetic',sources=descs)],evaluator_references=dict(path=ref.name,sha256=protocol.sha(ref))))
        m=builder.prepare(spec,'new',self.root);self.assertIs(m['fully_ready'],True)
        protocol.validate_bundle(self.root/'inputs/new')
        with self.assertRaisesRegex(ValueError,'new input'):builder.prepare(spec,'new',self.root)

    def test_task_cannot_change_evaluator_candidate_identity(self):
        source=complex_sources('protein_protein');source['task']=copy.deepcopy(source['query'])
        source['task']['candidates'][0]['sites'][0]['mutant']='C'
        with self.assertRaisesRegex(ValueError,'frozen evaluator query'):
            builder.record_from_sources('protein_protein','synthetic',source)

    def test_runner_prepare_and_resume_gate_before_credentials(self):
        bundle=make_bundle(self.root);run=self.root/'runs/new';config=self.root/'model.json'
        protocol.save(config,dict(model='gemini-3.8-flash',thinking_level='medium',generation_seed=42,
            temperature=0,max_output_tokens=100,timeout_s=3600,retries=0))
        args=argparse.Namespace(inputs=str(bundle),run=str(run),domains=None,arms='sequence',max_units=None,
            model_config=str(config),credentials='/never/read',unlimited=False,token_budget=10000,workers=1,max_repairs=0)
        with patch.object(runner,'HERE',self.root),patch.object(runner,'source_hashes',return_value={}):
            runner.prepare(args);runner.validate(run)
        m=protocol.load(bundle/'MANIFEST.json');m.pop('fully_ready');protocol.save(bundle/'MANIFEST.json',m)
        plan=protocol.load(run/'PLAN.json');plan['input_manifest_sha256']=protocol.sha(bundle/'MANIFEST.json')
        plan['fingerprint']=runner.digest({k:v for k,v in plan.items() if k!='fingerprint'});protocol.save(run/'PLAN.json',plan)
        with patch('run_frozen.credential') as credential,patch('run_frozen.ConcurrentGateway') as gateway:
            with self.assertRaisesRegex(ValueError,'fully_ready'):runner.worker(run)
            credential.assert_not_called();gateway.assert_not_called()

    def test_multimodel_prepare_validate_and_no_legacy_default(self):
        bundle=make_bundle(self.root);run=self.root/'runs/multi'
        with patch.object(campaign,'HERE',self.root),patch.object(campaign,'source_paths',return_value=[]),patch.object(campaign,'credential'):
            campaign.prepare(run,bundle);p,requests,tasks=campaign.validate(run)
            self.assertEqual(len(requests),4);self.assertEqual(len(tasks),16)
            self.assertEqual(p['domains'],{'protein':1})
            with self.assertRaisesRegex(ValueError,'Explicit --inputs'):campaign.prepare(self.root/'runs/legacy')

    def test_campaign_worker_rejects_unaccepted_input_before_preflight(self):
        bundle=make_bundle(self.root);run=self.root/'runs/multi'
        with patch.object(campaign,'HERE',self.root),patch.object(campaign,'source_paths',return_value=[]),patch.object(campaign,'credential'):
            campaign.prepare(run,bundle)
        m=protocol.load(bundle/'MANIFEST.json');m.pop('fully_ready');protocol.save(bundle/'MANIFEST.json',m)
        p=protocol.load(run/'PLAN.json');p['input_manifest_sha256']=protocol.sha(bundle/'MANIFEST.json')
        protocol.save(run/'PLAN.json',p);(run/'PLAN.sha256').write_text(protocol.sha(run/'PLAN.json'))
        with patch.object(campaign,'submit') as submit:
            with self.assertRaisesRegex(ValueError,'fully_ready'):campaign.run_campaign(run)
            submit.assert_not_called()

    def test_run_subset_cannot_replace_an_accepted_request(self):
        bundle=make_bundle(self.root);_,accepted=protocol.validate_bundle(bundle)
        protocol.validate_subset(accepted[:1],accepted)
        changed=copy.deepcopy(accepted[:1]);changed[0]['template']['user']='old prompt'
        with self.assertRaisesRegex(ValueError,'not an accepted'):protocol.validate_subset(changed,accepted)
        with self.assertRaisesRegex(ValueError,'duplicate'):protocol.validate_subset(accepted[:1]*2,accepted)

    def test_missingness_preserves_observed_zero_and_low_support(self):
        for domain in ('protein_protein','protein_rna'):
            source=complex_sources(domain)
            f=next(iter(source['msa']['msa']['M001'].values()));f['status']='global_neff_below_frozen_20'
            record=builder.record_from_sources(domain,'synthetic',source);protocol.render(record)
            out=record['evidence']['msa']['variants'][0]['sites'][0]['features']
            self.assertEqual(out['log_probability_ratio'],0.)
            self.assertEqual(out['weighted_valid_count'],2.)
            f.update(status='no_valid_homolog_at_column',weighted_valid_count=0,weighted_gap_fraction=1.)
            record=builder.record_from_sources(domain,'synthetic',source);protocol.render(record)
            out=record['evidence']['msa']['variants'][0]['sites'][0]['features']
            self.assertIsNone(out['weighted_valid_count']);self.assertIsNone(out['weighted_gap_fraction'])

    def test_multichunk_progress_averages_each_assay_once(self):
        run=self.root/'run';key='argon_high';child=run/'models'/key
        tasks=[dict(model_key=key,request_id=str(i),domain='protein',assay=assay,arm='sequence')
               for i,assay in enumerate(('one','one','two'))]
        protocol.save(run/'PLAN.json',dict(models={key:{}},workers=20,inputs='/synthetic'))
        protocol.save(run/'STATUS.json',dict(state='complete',valid=3,completed=3))
        protocol.save(run/'TASKS.json',tasks)
        metrics=[dict(domain='protein',assay=assay,arm='sequence',status='complete',spearman=rho,n_predicted=n,n_expected=n)
                 for assay,rho,n in [('one',.8,26),('two',.2,1)]]
        protocol.save(child/'evaluation/METRICS.json',dict(rows=metrics))
        for task in tasks:protocol.save(child/'units'/(task['request_id']+'.json'),dict(status='ok'))
        with patch.object(supervisor,'HERE',self.root),patch.object(supervisor.subprocess,'run',return_value=argparse.Namespace(returncode=0)):
            supervisor.evaluate_and_publish(run)
        summary=protocol.load(run/'evaluation/SUMMARY.json')['rows']
        row=next(r for r in summary if r['arm']=='sequence')
        self.assertEqual(row['planned_assays'],2);self.assertEqual(row['planned_requests'],3)
        self.assertEqual(row['evaluated_assays'],2);self.assertAlmostEqual(row['mean_signed_spearman'],.5)
        self.assertIn('3/3',(run/'PROGRESS.md').read_text())


if __name__=='__main__':unittest.main()
