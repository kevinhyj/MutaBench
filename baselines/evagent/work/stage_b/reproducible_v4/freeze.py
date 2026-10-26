"""Freeze the predeclared paired follow-up without selecting on outcomes."""
import fcntl
import json
from pathlib import Path
import sqlite3
import datetime
from evagent.integrated.config import ExperimentConfig
from evagent.integrated.experiment import prepare
from evagent.integrated.tasks import file_hash
from runner import save,load,HERE


def main():
    project=Path('/data/yanjie_huang/EVAgent');old=project/'experiments/cr_ncrna_evidence_b_v3';root=project/'experiments/cr_ncrna_evidence_b_repro_v4'
    if (root/'campaign.json').exists():raise RuntimeError('Freeze already exists; do not overwrite')
    root.mkdir(parents=True,exist_ok=True)
    # A campaign lease and a gateway lease prove no old provider admission remains.
    with (old/'campaign/campaign.lock').open('a+') as outer,(old/'campaign/provider/request_ledger.lock').open('a+') as inner:
        fcntl.flock(outer,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(inner,fcntl.LOCK_EX|fcntl.LOCK_NB)
        source=sqlite3.connect(f'file:{old}/campaign/provider/request_ledger.sqlite?mode=ro',uri=True)
        if source.execute("select count(*) from attempts where status='inflight'").fetchone()[0]:raise RuntimeError('Old provider still in flight')
        snapshot=root/'prior_v3_ledger.sqlite';dest=sqlite3.connect(snapshot);source.backup(dest);dest.close()
        prior=json.loads(source.execute("select value from run_state where key='prior_usage'").fetchone()[0]);prior['ledgers'].append({'path':str(snapshot),'sha256':file_hash(snapshot)})
        unknown_reserve=0;unknown_count=0
        for status,tokens,elapsed,request_json in source.execute("select status,tokens,elapsed_s,request_json from attempts where status!='not_submitted'"):
            prior['provider_attempts']+=1;prior['reported_tokens']+=tokens or 0;prior['provider_elapsed_s']+=elapsed or 0
            prior['requests_by_status'][status]=prior['requests_by_status'].get(status,0)+1
            if status in {'failed_unknown_cost','interrupted_unknown_cost','unknown_cost'} or tokens is None:
                q=json.loads(request_json);reserve=sum(len(m['content'].encode('utf-8')) for m in q['messages'])+2*q['model_config']['max_output_tokens']+1024
                unknown_reserve+=max(0,reserve-(tokens or 0));unknown_count+=1
        prior['unknown_cost_reserve_tokens']=unknown_reserve;prior['unknown_cost_attempts']=unknown_count
        source.close()
    raw=load(old/'campaign.json');by_assay={}
    for row in raw['cells']:
        if row['status']=='eligible' and row['source_status']=='admitted' and row['condition'] in {'cr','all'}:
            by_assay.setdefault(row['assay'],{})[(row['n'],row['condition'])]=row
    tasks=[]
    for assay,cells in sorted(by_assay.items()):
        ns=sorted({n for n,co in cells if (n,'cr') in cells and (n,'all') in cells});maximum=max(ns)
        task={'assay':assay,'family':cells[(maximum,'cr')]['family'],'source_status':'admitted','maximum_n':maximum,'samples':{},'templates':{}}
        for n in ns:
            cfg=ExperimentConfig.load(Path(cells[(n,'cr')]['config']));sp=Path(cfg.values['sample_file'])
            task['samples'][str(n)]={'path':str(sp),'sha256':file_hash(sp)}
        for co in ('cr','all'):
            cfg=ExperimentConfig.load(Path(cells[(maximum,co)]['config']));_,m,_=prepare(cfg)
            task['templates'][co]=m['initial_requests']
            for n in ns:
                if n == maximum:continue
                small_cfg=ExperimentConfig.load(Path(cells[(n,co)]['config']));_,small,_=prepare(small_cfg)
                if small['initial_requests']!=m['initial_requests'][:n//25]:raise RuntimeError('Original nested prompts differ')
            if co=='all':task['evidence_sha256']=m['evidence']['sha256'];task['evidence_sources']={s:v['status'] for s,v in m['evidence']['sources'].items()}
        tasks.append(task)
    if len(tasks)!=21:raise RuntimeError('Expected all 21 previously admitted tasks')
    save(root/'tasks.json',{'tasks':tasks,'selection':'All source-admitted CR/all cells, independent of prior outcomes','excluded_coverage':[{k:r[k] for k in ('assay','n','condition','status','reason') if k in r} for r in raw['cells'] if r['status']!='eligible']})
    assets={str(f):file_hash(f) for f in (old/'assets').rglob('*') if f.is_file()}
    for f in (project/'evagent').rglob('*.py'):assets[str(f)]=file_hash(f)
    for f in (root/'tasks.json',snapshot):assets[str(f)]=file_hash(f)
    model={**raw['model_config'],'generation_seed':1}
    cfg={'schema':'paired-stage-b-repro-v4','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
         'token_budget':50000000,'conditions':['cr','all'],'repeats':3,'model_config':model,'max_repairs':1,
         'credentials_file':'/home/yanjie/.config/evagent/credentials.env','prior_usage':prior,'asset_hashes':assets,
         'runtime_hashes':{name:file_hash(HERE/name) for name in ('runner.py','gateway.py','freeze.py')},
         'design':{'generation_seed':1,'sampling_seed':1,'replicate_unit':'fresh model draw with fixed seed','n50':'same first two predicted groups of maximum N','condition_order':'alternates by repeat, task, and group','memory_mode':'none','evidence':'unchanged frozen v3 bundles','primary_endpoint':'family-macro paired delta Spearman at N250 averaged across 3 repeats per assay','secondary_endpoints':['Kendall','N50 nested evaluation','repeat variability','failure rate','reported tokens and unknown-cost reserve'],'no_outcome_dependent_prompt_changes':True}}
    save(root/'campaign.json',cfg)
    save(root/'unknown_response_policy.json',{'mode':'retry_then_continue','max_attempts':3,'backoff_seconds':[15,30]})
    print(json.dumps({'root':str(root),'tasks':len(tasks),'predicted_units':sum(t['maximum_n']//25 for t in tasks)*2*3,'prior_reported_tokens':prior['reported_tokens'],'prior_unknown_reserve':unknown_reserve,'asset_files':len(assets)}))

if __name__=='__main__':main()
