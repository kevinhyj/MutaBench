"""User-authorized scheduler change: reuse V5 units/ledger, submit structure only."""
import argparse,collections,datetime,fcntl,json,os
from pathlib import Path
import probe
r=probe.r
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]/'experiments/cr_ncrna_simple_structure_v5'

def report(tasks,cfg,root,gateway,status,current):
 import numpy as np
 rows=r.result_rows(tasks,{**cfg,'conditions':['structure_raw']},root)
 for row in rows:
  row['evidence_availability']={'structure':'available'}
  row['evidence_bundle_sha256']=r.file_hash(HERE/'raw_structures.json')
 baseline=r.load(root/'historical_v4_progress.json')['rows'];comparisons=[]
 for n in (50,250):
  pairs=[]
  for a in rows:
   if a['n']!=n or a['status']!='ok':continue
   bs=[b for b in baseline if b['assay']==a['assay'] and b['n']==n and b['condition']=='cr' and b['status']=='ok']
   if len(bs)!=3:continue
   pairs.append({'assay':a['assay'],'family':a['family'],'historical_cr_repeats':3,'structure_repeats':1,'delta_spearman':a['metrics']['spearman']-float(np.mean([b['metrics']['spearman'] for b in bs])),'delta_kendall':a['metrics']['kendall_tau']-float(np.mean([b['metrics']['kendall_tau'] for b in bs])),'delta_vs_each_historical_cr':[{'replicate':b['replicate'],'spearman':a['metrics']['spearman']-b['metrics']['spearman'],'kendall':a['metrics']['kendall_tau']-b['metrics']['kendall_tau']} for b in bs]})
  c={'n':n,'pairs':pairs,'paired_assays':len(pairs),'comparison':'one new structure prediction versus mean metric of three historical CR runs; noncontemporaneous; three differences are dependent'}
  if pairs:
   fs=sorted({p['family'] for p in pairs});c['families']=len(fs)
   for metric in ('spearman','kendall'):c['family_macro_delta_'+metric]=float(np.mean([np.mean([p['delta_'+metric] for p in pairs if p['family']==f]) for f in fs]))
  comparisons.append(c)
 units=[r.load(p) for p in (root/'units').glob('*.json')]
 value={'status':status,'timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'current':current,'execution_conditions':['structure_raw'],'completed_units':sum(u['condition']=='structure_raw' for u in units),'planned_units':sum(len(t['templates']['structure_raw']) for t in tasks),'preserved_cr_units':sum(u['condition']=='cr' for u in units),'statuses':dict(collections.Counter(x['status'] for x in rows)),'rows':rows,'comparisons':comparisons,'ledger':gateway.summary()}
 r.save(root/'progress.json',value)

def execute(gateway,tasks,cfg,root,callback):
 for task in tasks:
  for i,temp in enumerate(task['templates']['structure_raw']):
   current={'assay':task['assay'],'arm':'structure_raw','unit':i}
   result=r.run_unit(gateway,task,'structure_raw',1,i,temp,cfg,root)
   print(json.dumps({**current,'status':result['status']}),flush=True)
   callback(current)

def run(dry=False):
 cfg,tasks,root=probe.validate(ROOT/'campaign.json');manifest=r.load(root/'structure_only_manifest.json')
 for f,h in manifest['hashes'].items():
  if r.file_hash(Path(f))!=h:raise ValueError('Continuation or baseline changed: '+f)
 if manifest['execution_conditions']!=['structure_raw']:raise ValueError('Unexpected conditions')
 if dry:print(json.dumps({'status':'dry_run','planned_structure_units':202,'new_cr_requests':0}));return
 for line in Path(cfg['credentials_file']).read_text().splitlines():
  line=line.strip().removeprefix('export ')
  if line.startswith('GEMINI_API_KEY='):os.environ['GEMINI_API_KEY']=line.split('=',1)[1].strip().strip('\"\'')
 with (root/'campaign.lock').open('a+') as lease:
  fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
  gateway=r.gw.RequestGateway(model_config=cfg['model_config'],run_dir=root/'provider',fingerprint=r.gw.digest(cfg),token_budget=cfg['token_budget'],reserve_budget=True,prior_usage=cfg['prior_usage'])
  state='complete';current=None
  def progress(c):
   nonlocal current
   current=c;report(tasks,cfg,root,gateway,'running',current)
  try:
   report(tasks,cfg,root,gateway,'running',current)
   execute(gateway,tasks,cfg,root,progress)
  except Exception as exc:
   state='paused' if isinstance(exc,r.ExecutionPaused) else 'error'
   r.save(root/'structure_only_failure.json',{'type':type(exc).__name__,'current':current})
   if state=='error':
    import traceback;traceback.print_exc()
  finally:report(tasks,cfg,root,gateway,state,current);gateway.close()
  if state!='complete':raise SystemExit(2)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--dry-run',action='store_true');run(p.parse_args().dry_run)
