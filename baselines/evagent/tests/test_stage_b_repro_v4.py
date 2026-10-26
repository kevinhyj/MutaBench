"""Offline acceptance for seeded transport, repeated draws and nested evaluation."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import pytest

HERE=Path(__file__).resolve().parents[1]/'work/stage_b/reproducible_v4'
spec=importlib.util.spec_from_file_location('stage_b_repro_runner',HERE/'runner.py')
r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
g=r.gw


def response(ids=None):
    ids=ids or ['M01','M02']
    return {'final_text':json.dumps({'variants':[{'variant_id':v,'predicted_score':i/100} for i,v in enumerate(ids)]}),
            'thought_summary':'','finish_reason':'STOP','tool_calls':[],
            'usage_summary':{'input_tokens':8,'output_tokens':2,'reasoning_tokens':0,'total_tokens':10,'model_calls':1},
            'response_metadata':{'model_version':'mock'}}


def gateway(tmp_path,transport,budget=100000,prior=None):
    return g.RequestGateway(model_config={'model':'mock','thinking_level':'medium','max_tokens':100,'generation_seed':1},
        run_dir=tmp_path/'provider',fingerprint='f',token_budget=budget,transport=transport,prior_usage=prior)


def task(n=50):
    t={'assay':'benign_mock','family':'mock','source_status':'admitted','templates':{'cr':[],'all':[]},'samples':{}}
    for i in range(n//25):
        for co in ('cr','all'):
            t['templates'][co].append({'system':'frozen','user':co+'-'+str(i),'expected_ids':[f'M{j:02d}' for j in range(i*25+1,i*25+26)]})
    return t


def config():return {'conditions':['cr','all'],'repeats':3,'max_repairs':1,'model_config':{'model':'mock','generation_seed':1}}


def test_generation_seed_reaches_sdk(monkeypatch):
    import google.genai
    captured={}
    class Client:
        def __init__(self,**kwargs):self.models=self
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def generate_content_stream(self,**kwargs):
            captured.update(kwargs)
            yield SimpleNamespace(usage_metadata=SimpleNamespace(prompt_token_count=8,candidates_token_count=2,total_token_count=10,thoughts_token_count=0),
                candidates=[SimpleNamespace(finish_reason='STOP',content=SimpleNamespace(parts=[SimpleNamespace(text='OK',thought=False,function_call=None)]))],prompt_feedback=None,model_version='mock')
    monkeypatch.setattr(google.genai,'Client',Client);monkeypatch.setenv('MOCK_KEY','not-a-real-key')
    cfg=g.public_model_config({'model':'mock','thinking_level':'medium','generation_seed':1,'max_tokens':100,'api_key_env':'MOCK_KEY'})
    out=g.google_stream_transport({'model':'mock','model_config':cfg,'messages':[{'role':'user','content':'OK'}]})
    assert out['final_text']=='OK';assert captured['config'].seed==1


def test_replicates_fresh_but_restart_cached(tmp_path):
    calls=[];gt=gateway(tmp_path,lambda q:calls.append(q) or response(q['metadata']['expected_ids']));t=task();cfg=config();template=t['templates']['cr'][0]
    a=r.run_unit(gt,t,'cr',1,0,template,cfg,tmp_path)
    b=r.run_unit(gt,t,'cr',1,0,template,cfg,tmp_path)
    c=r.run_unit(gt,t,'cr',2,0,template,cfg,tmp_path)
    assert a==b;assert a['logical_id']!=c['logical_id'];assert len(calls)==2
    assert all(q['model_config']['generation_seed']==1 for q in calls)
    gt.close()


def test_nested_predictions_are_same_values_without_calls(tmp_path):
    t=task(250);cfg=config();rows=[{'id':str(i),'seq':'A','score':i} for i in range(250)]
    for n in (50,250):
        p=tmp_path/f'n{n}.json';r.save(p,{'variants':rows[:n]});t['samples'][str(n)]={'path':str(p),'sha256':r.file_hash(p)}
    calls=[];gt=gateway(tmp_path,lambda q:calls.append(q) or response(q['metadata']['expected_ids']))
    for co in ('cr','all'):
        for i,template in enumerate(t['templates'][co]):r.run_unit(gt,t,co,1,i,template,cfg,tmp_path)
    before=len(calls);out=r.result_rows([t],cfg,tmp_path)
    assert before==20 and len(calls)==before
    assert len([v for v in out if v['status']=='ok'])==4
    assert len([v for v in out if v['n']==50 and v['status']=='ok'])==2
    gt.close()


def test_unknown_retries_counted_and_api_failure_not_repaired(tmp_path,monkeypatch):
    monkeypatch.setattr(g.time,'sleep',lambda _:None)
    r.save(tmp_path/'unknown_response_policy.json',{'mode':'retry_then_continue','max_attempts':3,'backoff_seconds':[15,30]})
    calls=[];gt=gateway(tmp_path,lambda q:calls.append(q) or None);t=task();cfg=config()
    out=r.run_unit(gt,t,'cr',1,0,t['templates']['cr'][0],cfg,tmp_path)
    assert out['status']=='api_failed';assert len(calls)==3;assert out['repair_attempts_used']==0
    assert not gt.summary()['paused'];assert gt.summary()['unknown_cost_reserve_tokens']>0
    gt.close()


def test_operator_pause_never_becomes_api_failure(tmp_path,monkeypatch):
    monkeypatch.setattr(g.time,'sleep',lambda _:None)
    r.save(tmp_path/'unknown_response_policy.json',{'mode':'retry_then_continue','max_attempts':3,'backoff_seconds':[15,30]})
    gt=gateway(tmp_path,lambda q:pytest.fail('No call under PAUSE'))
    (tmp_path/'provider/PAUSE').write_text('operator stop')
    t=task()
    with pytest.raises(r.ExecutionPaused):r.run_unit(gt,t,'cr',1,0,t['templates']['cr'][0],config(),tmp_path)
    assert not list((tmp_path/'units').glob('*.json'))
    gt.close()


def test_prior_unknown_reserve_blocks_budget(tmp_path):
    gt=gateway(tmp_path,lambda q:pytest.fail('Prior reserve must block admission'),budget=1000,
               prior={'reported_tokens':10,'provider_attempts':1,'unknown_cost_reserve_tokens':990})
    with pytest.raises(r.ExecutionPaused):gt.generate(system='x',user='y',run_name='z')
    assert gt.summary()['budget_accounted_tokens']==1000
    gt.close()


def test_prompt_or_evidence_change_invalidates_key():
    t=task();cfg=config();a=t['templates']['all'][0]
    assert r.unit_key(t,'all',1,0,a,cfg)!=r.unit_key(t,'all',1,0,{**a,'user':a['user']+'new evidence'},cfg)


def test_full_three_repeats_required_for_primary_summary():
    rows=[]
    for rep in range(1,4):
        for co in ('cr','all'):
            rows.append({'assay':'mock','family':'mock','n':250,'replicate':rep,'condition':co,'status':'ok','metrics':{'spearman':.1 if co=='cr' else .2,'kendall_tau':.1 if co=='cr' else .2}})
    assert r.summarize(rows[:-2])[0]['assays_with_all_three_repeats']==0
    assert r.summarize(rows)[0]['family_macro_delta_spearman']==pytest.approx(.1)
