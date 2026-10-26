"""Read-only diagnostic of complete V4 triplicates; stdout is a reproducible snapshot."""
import json,datetime
import numpy as np
from scipy.stats import rankdata,spearmanr
from pathlib import Path
import probe
r=probe.r
root=Path(__file__).resolve().parents[3]/'experiments/cr_ncrna_evidence_b_repro_v4'
cfg,manifest,_,_=r.validate_plan(root/'campaign.json')

def rho(a,b):
 z=float(spearmanr(a,b).statistic)
 return z if np.isfinite(z) else None

def partial(a,b,x):
 ar=rankdata(a);br=rankdata(b)
 return rho_pearson(ar-x@np.linalg.lstsq(x,ar,rcond=None)[0],br-x@np.linalg.lstsq(x,br,rcond=None)[0])

def rho_pearson(a,b):return float(np.corrcoef(a,b)[0,1])

rows=[]
for t in manifest['tasks']:
 if '250' not in t['samples']:continue
 values={};hashes={};complete=True
 for arm in ('cr','all'):
  vectors=[]
  for rep in range(1,4):
   pred={}
   for i,temp in enumerate(t['templates'][arm]):
    key=r.unit_key(t,arm,rep,i,temp,cfg);p=root/'units'/f'{key}.json'
    if not p.exists():complete=False;continue
    if p.with_suffix('.sha256').read_text().strip()!=r.file_hash(p):raise ValueError('Unit hash changed')
    u=r.load(p);hashes[p.name]=r.file_hash(p)
    if u['status']!='ok':complete=False;continue
    pred.update(u['value'])
   if len(pred)==250:vectors.append([pred[f'M{i:02d}'] for i in range(1,251)])
  if len(vectors)==3:values[arm]=np.mean(vectors,axis=0)
 if not complete or len(values)!=2:continue
 features={}
 for temp in t['templates']['all']:features.update(json.loads(temp['user'].split(probe.MARKER)[1])['structure']['candidates'])
 ordered=[features[f'M{i:02d}'] for i in range(1,251)]
 f=np.array([-a['mean_pair_probability_change'] for a in ordered]);counts=np.array([len(a['changed_positions']) for a in ordered]);groups=np.arange(250)//25
 y=np.array([a['score'] for a in r.load(t['samples']['250']['path'])['variants']])
 x=np.column_stack([np.ones(250)]+[(counts==c).astype(float) for c in sorted(set(counts))[1:]]+[(groups==g).astype(float) for g in range(1,10)])
 strata=[]
 for c in sorted(set(counts)):
  ix=counts==c
  if ix.sum()>=20:strata.append({'mutation_count':int(c),'n':int(ix.sum()),'cr_rho':rho(values['cr'][ix],y[ix]),'all_rho':rho(values['all'][ix],y[ix])})
 rows.append({'assay':t['assay'],'family':t['family'],'feature':'negative mean_pair_probability_change','raw_feature_rho':{k:rho(f,a) for k,a in {'truth':y,**values}.items()},'partial_feature_correlation_controlling_categorical_count_and_group':{k:partial(f,a,x) for k,a in {'truth':y,**values}.items()},'count_strata_min_n20':strata,'within_group_rho':{k:float(np.nanmean([rho(a[i:i+25],y[i:i+25]) for i in range(0,250,25)])) for k,a in values.items()},'unit_hashes':hashes})
print(json.dumps({'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'campaign_sha256':r.file_hash(root/'campaign.json'),'interpretation':'Exploratory posthoc diagnostics of three-repeat mean predictions; not an ensemble benchmark or causal effect. Labels only used here, never in intervention construction. Excludes incomplete triplicates.','tasks':rows},indent=2,allow_nan=False))
