import importlib.util,json
from pathlib import Path
import pytest
path=Path(__file__).resolve().parents[1]/'work/stage_b/simple_structure_v5/probe.py'
spec=importlib.util.spec_from_file_location('simple_v5',path);v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v)

def setup(monkeypatch,tmp_path):
 raw={'mock':{'wt_secondary_structure':{'dot_bracket':'(..)'},'wt_tertiary_structure':{'kind':'predicted_WT','residues':[{'position':1,'base':'A','atoms':{'P':[1.,2.,3.]}}]},'variant_secondary_structures':{'M01':{'dot_bracket':'....'},'M02':{'dot_bracket':'(..)'}}}}
 v.r.save(tmp_path/'raw_structures.json',raw);monkeypatch.setattr(v,'HERE',tmp_path)
 return {'system':'fixed task','user':'fixed context and sequence','expected_ids':['M01'],'metadata':{}}

def test_raw_input_has_only_requested_structures(monkeypatch,tmp_path):
 t=setup(monkeypatch,tmp_path);out,_=v.transform(t,'structure_raw','mock',0)
 data=json.loads(out['user'].split('\n')[-1])
 assert set(data)=={'wt_secondary_structure','wt_tertiary_structure','variant_secondary_structures'}
 assert set(data['variant_secondary_structures'])=={'M01'}
 assert data['variant_secondary_structures']['M01']=={'dot_bracket':'....'}
 assert out['system']==t['system'] and out['user'].startswith(t['user'])
 assert all(x not in out['user'] for x in ['mfe_kcal','ensemble','mean_pair_probability','site_log_frequency','literature','plddt'])

def test_baseline_unchanged(monkeypatch,tmp_path):
 t=setup(monkeypatch,tmp_path);assert v.transform(t,'cr','mock',0)==(t,{})

def test_two_conditions_gateway_and_cache(monkeypatch,tmp_path):
 t=setup(monkeypatch,tmp_path);calls=[]
 def transport(q):
  calls.append(q)
  return {'final_text':json.dumps({'variants':[{'variant_id':'M01','predicted_score':.2}]}),'thought_summary':'','finish_reason':'STOP','tool_calls':[], 'usage_summary':{'input_tokens':8,'output_tokens':2,'reasoning_tokens':0,'total_tokens':10,'model_calls':1},'response_metadata':{'model_version':'mock'}}
 g=v.r.gw.RequestGateway(model_config={'model':'mock','max_tokens':100,'generation_seed':1,'thinking_level':'medium'},run_dir=tmp_path/'provider',fingerprint='simple',token_budget=100000,transport=transport)
 for _ in range(2):
  for arm in v.ARMS:
   temp,_=v.transform(t,arm,'mock',0)
   result=v.r.run_unit(g,{'assay':'mock'},arm,1,0,temp,{'max_repairs':1,'model_config':{'model':'mock'}},tmp_path)
   assert result['status']=='ok'
 assert len(calls)==2 and g.summary()['reported_tokens']==20
 g.close()
