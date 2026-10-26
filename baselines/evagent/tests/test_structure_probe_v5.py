import collections,importlib.util,json
from pathlib import Path
import pytest
p=Path(__file__).resolve().parents[1]/'work/stage_b/structure_probe_v5/probe.py';spec=importlib.util.spec_from_file_location('probe_v5_test',p);v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v)

def template():
 c={}
 for i in range(1,5):c[f'M{i:02d}']={'changed_positions':[i],'site_unpaired_probability':{str(i):.2},**{f:i*.1 for f in v.FIELDS}}
 e={'structure':{'candidates':c,'task':{'wt':'frozen'},'positions':{'1':{'wt_unpaired_probability':.5}}},'msa':{'status':'unavailable'},'literature':{'task':{'records':['frozen']}}}
 return {'system':'original','user':'original prompt\n'+v.MARKER+json.dumps(e,sort_keys=True,separators=(',',':')),'metadata':{'unit_index':0},'expected_ids':list(c)}

def parse(t):return json.loads(t['user'].split(v.MARKER)[1])

def test_full_control_is_byte_identical():
 t=template();assert v.transform(t,'all','mock',0)==(t,{})

def test_deletion_only_changes_declared_scalar_fields():
 t=template();z,_=v.transform(t,v.ARMS[1],'mock',0);a=parse(t);b=parse(z)
 assert a['msa']==b['msa'];assert a['literature']==b['literature'];assert a['structure']['positions']==b['structure']['positions']
 for key in a['structure']['candidates']:
  old=a['structure']['candidates'][key];new=b['structure']['candidates'][key]
  assert {k:x for k,x in old.items() if k not in v.FIELDS}==new
 assert t['system']==z['system'] and t['expected_ids']==z['expected_ids']

def test_shuffle_preserves_length_and_scalar_tuples_without_fixed_points():
 t=template();z,m=v.transform(t,v.ARMS[2],'mock',0)
 assert len(t['user'].encode())==len(z['user'].encode());assert all(k!=donor for k,donor in m.items())
 a=parse(t)['structure']['candidates'];b=parse(z)['structure']['candidates']
 assert collections.Counter(tuple(x[f] for f in v.FIELDS) for x in a.values())==collections.Counter(tuple(x[f] for f in v.FIELDS) for x in b.values())
 for k in a:
  assert b[k]['changed_positions']==a[k]['changed_positions'];assert b[k]['site_unpaired_probability']==a[k]['site_unpaired_probability']
 assert v.transform(t,v.ARMS[2],'mock',0)==(z,m)

def test_no_overlap_with_active_v4(tmp_path):
 source=tmp_path/'old';source.mkdir();v.r.save(source/'progress.json',{'status':'running'})
 with pytest.raises(RuntimeError,match='Wait for v4 completion'):v.freeze(tmp_path/'new',source)
 assert not (tmp_path/'new').exists()


def test_all_arms_use_gateway_and_restart_without_calls(tmp_path):
 calls=[]
 def transport(q):
  calls.append(q)
  return {'final_text':json.dumps({'variants':[{'variant_id':vid,'predicted_score':i/10} for i,vid in enumerate(q['metadata']['expected_ids'])]}),'thought_summary':'','finish_reason':'STOP','tool_calls':[], 'usage_summary':{'input_tokens':8,'output_tokens':2,'reasoning_tokens':0,'total_tokens':10,'model_calls':1},'response_metadata':{'model_version':'mock'}}
 gt=v.r.gw.RequestGateway(model_config={'model':'mock','thinking_level':'medium','max_tokens':100,'generation_seed':1},run_dir=tmp_path/'provider',fingerprint='probe',token_budget=100000,transport=transport)
 task={'assay':'mock'};cfg={'model_config':{'model':'mock','generation_seed':1},'max_repairs':1}
 for _ in range(2):
  for arm in v.ARMS:
   t,_=v.transform(template(),arm,'mock',0)
   out=v.r.run_unit(gt,task,arm,1,0,t,cfg,tmp_path)
   assert out['status']=='ok'
 assert len(calls)==3 and gt.summary()['reported_tokens']==30
 assert {q['metadata']['condition'] for q in calls}==set(v.ARMS)
 gt.close()
