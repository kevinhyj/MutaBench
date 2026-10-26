"""Frozen, paired CR vs all-evidence experiment with shared nested predictions.

One immutable gateway ledger owns every actual call. Labels are only loaded by
post-prediction evaluation. A replicate is an intentional fresh draw, even with
identical generation_seed; N=50 reuses the first two N=250 groups verbatim.
"""
from __future__ import annotations
import argparse
import collections
import datetime
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('repro_v4_gateway', HERE/'gateway.py')
gw = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gw)
from evagent.integrated.chunked_scoring import ScoringResponseValidator, ranking_from_scoring_scores
from evagent.integrated.self_check import SelfCheckPolicy, run_self_check
from evagent.integrated.tasks import file_hash, evaluate_result
from evagent.runtime import ExecutionPaused


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+'\n')
    temp.replace(path)


def load(path):
    return json.loads(Path(path).read_text())


def validate_plan(path):
    config=load(path);root=path.parent
    if config['schema']!='paired-stage-b-repro-v4':raise ValueError('Unknown plan schema')
    if config['conditions']!=['cr','all'] or config['repeats']!=3:raise ValueError('Frozen paired design changed')
    if config['model_config'].get('generation_seed')!=1:raise ValueError('Explicit frozen generation seed required')
    for relative,sha in config['runtime_hashes'].items():
        if file_hash(HERE/relative)!=sha:raise ValueError('Runtime changed: '+relative)
    for filename,sha in config['asset_hashes'].items():
        if file_hash(Path(filename))!=sha:raise ValueError('Frozen asset changed: '+filename)
    manifest=load(root/'tasks.json')
    for task in manifest['tasks']:
        if task['source_status']!='admitted':raise ValueError('Non-admitted task')
        ns=sorted(int(n) for n in task['samples'])
        maximum=max(ns);full=load(task['samples'][str(maximum)]['path'])['variants']
        for n in ns:
            if load(task['samples'][str(n)]['path'])['variants']!=full[:n]:raise ValueError('Samples not exact nested prefixes')
        for cond in config['conditions']:
            templates=task['templates'][cond]
            expected=[f'M{i:02d}' for i in range(1,maximum+1)]
            if [i for t in templates for i in t['expected_ids']]!=expected:raise ValueError('Group identity mismatch')
            for t in templates:
                if len(t['expected_ids'])!=25:raise ValueError('Unexpected group size')
                if 'Frozen multi-source evidence' in t['user'] and cond=='cr':raise ValueError('CR received evidence')
    fingerprint=gw.digest(config)
    return config,manifest,root,fingerprint


def unit_key(task,cond,rep,index,template,config):
    return gw.digest({'assay':task['assay'],'condition':cond,'replicate':rep,
                     'unit_index':index,'system':template['system'],'user':template['user'],
                     'expected_ids':template['expected_ids'],'model_config':config['model_config'],
                     'max_repairs':config['max_repairs'],'template_schema':'v4'})


def run_unit(gateway,task,cond,rep,index,template,config,root):
    key=unit_key(task,cond,rep,index,template,config);path=root/'units'/f'{key}.json'
    if path.exists():
        value=load(path)
        if value.get('logical_id')!=key:raise ValueError('Unit identity mismatch')
        checksum=path.with_suffix('.sha256')
        if not checksum.exists() or checksum.read_text().strip()!=file_hash(path):raise ValueError('Unit checksum mismatch')
        return value
    validator=ScoringResponseValidator(template['expected_ids'])
    def generate(user,attempt_index):
        return gateway.generate(system=template['system'],user=user,
            run_name=f"{task['assay']}-{cond}-r{rep}-g{index}-a{attempt_index}",
            metadata={'assay':task['assay'],'condition':cond,'replicate_id':rep,
                      'unit_index':index,'logical_id':key,'expected_ids':template['expected_ids'],
                      'attempt_index':attempt_index,'attempt_kind':'initial' if not attempt_index else 'repair'})
    try:
        checked=run_self_check(original_prompt=template['user'],generate=generate,
                              validator=validator,policy=SelfCheckPolicy(max_repairs=config['max_repairs']))
        value={k:checked[k] for k in ('status','value','repair_attempts_used','usage_summary')}
        value['provider_request_hashes']=[a.get('request_sha256') for a in checked['attempts']]
    except gw.ProviderAttemptsExhausted as error:
        value={'status':'api_failed','value':None,'reason':str(error),'repair_attempts_used':0}
    value.update(logical_id=key,expected_ids=template['expected_ids'],assay=task['assay'],condition=cond,
                 replicate=rep,unit_index=index)
    save(path,value);path.with_suffix('.sha256').write_text(file_hash(path)+'\n')
    return value


def result_rows(tasks,config,root):
    rows=[]
    for task in tasks:
        for rep in range(1,config['repeats']+1):
            for cond in config['conditions']:
                units=[]
                for index,template in enumerate(task['templates'][cond]):
                    key=unit_key(task,cond,rep,index,template,config);p=root/'units'/f'{key}.json'
                    if p.exists():
                        if p.with_suffix('.sha256').read_text().strip()!=file_hash(p):raise ValueError('Saved unit integrity mismatch')
                        units.append(load(p))
                    else:units.append({'status':'not_run'})
                for n,sample in sorted(task['samples'].items(),key=lambda t:int(t[0])):
                    n=int(n);selected=units[:n//25];status='ok' if all(u['status']=='ok' for u in selected) else ('incomplete' if any(u['status']=='not_run' for u in selected) else 'failed')
                    row={'assay':task['assay'],'family':task['family'],'n':n,'replicate':rep,'condition':cond,'status':status,
                         'units':collections.Counter(u['status'] for u in selected),
                         'evidence_availability':task.get('evidence_sources',{}) if cond=='all' else {},
                         'evidence_bundle_sha256':task.get('evidence_sha256') if cond=='all' else None}
                    if status=='ok':
                        scores={vid:v for u in selected for vid,v in u['value'].items()}
                        ids=[f'M{i:02d}' for i in range(1,n+1)]
                        if set(scores)!=set(ids):raise ValueError('Incomplete aggregate')
                        prediction={'status':'ok','scores':scores,'ranking':ranking_from_scoring_scores(scores)}
                        row['metrics']=evaluate_result(Path(sample['path']),sample['sha256'],ids,prediction)
                        if row['metrics']['status']!='ok':row['status']='undefined_metric'
                    rows.append(row)
    return rows


def summarize(rows):
    import numpy as np
    comparisons=[]
    for n in sorted({r['n'] for r in rows}):
        lookup={(r['assay'],r['replicate'],r['condition']):r for r in rows if r['n']==n and r['status']=='ok'}
        pairs=[]
        for assay,rep,cond in lookup:
            if cond!='cr' or (assay,rep,'all') not in lookup:continue
            a=lookup[(assay,rep,'all')];b=lookup[(assay,rep,'cr')]
            pairs.append({'assay':assay,'family':a['family'],'replicate':rep,
                'delta_spearman':a['metrics']['spearman']-b['metrics']['spearman'],
                'delta_kendall':a['metrics']['kendall_tau']-b['metrics']['kendall_tau']})
        assays={a: [r for r in pairs if r['assay']==a] for a in sorted({r['assay'] for r in pairs})}
        complete={a:rr for a,rr in assays.items() if len(rr)==3}
        desc={'n':n,'paired_replicates':len(pairs),'assays_with_pairs':len(assays),'assays_with_all_three_repeats':len(complete),'pairs':pairs}
        if pairs:desc['exploratory_mean_delta_spearman']=float(np.mean([r['delta_spearman'] for r in pairs]))
        if complete:
            means=[{'assay':a,'family':rr[0]['family'],
                    'delta_spearman':float(np.mean([r['delta_spearman'] for r in rr])),
                    'delta_kendall':float(np.mean([r['delta_kendall'] for r in rr])),
                    'repeat_sd_delta_spearman':float(np.std([r['delta_spearman'] for r in rr],ddof=1)),
                    'positive_repeats':sum(r['delta_spearman']>0 for r in rr)} for a,rr in complete.items()]
            families=sorted({r['family'] for r in means})
            fm=[float(np.mean([r['delta_spearman'] for r in means if r['family']==f])) for f in families]
            desc.update(assay_repeat_summaries=means,families=len(families),family_macro_delta_spearman=float(np.mean(fm)),
                        family_macro_delta_kendall=float(np.mean([np.mean([r['delta_kendall'] for r in means if r['family']==f]) for f in families])))
            if len(fm)>=3:
                draws=np.random.default_rng(1).choice(fm,(2000,len(fm)),replace=True).mean(axis=1)
                desc['family_bootstrap_95pct']=[float(v) for v in np.quantile(draws,[.025,.975])]
        comparisons.append(desc)
    return comparisons


def report(tasks,config,root,gateway,state,current=None):
    rows=result_rows(tasks,config,root);summary={'timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'status':state,'current':current,'rows':rows,'statuses':dict(collections.Counter(r['status'] for r in rows)),
        'comparisons':summarize(rows),'ledger':gateway.summary(),
        'completed_units':len(list((root/'units').glob('*.json'))),'planned_units':sum(len(t['templates']['cr'])*2*config['repeats'] for t in tasks),
        'coverage':{'admitted_tasks':len(tasks),'evidence_available_tasks':{src:sum(t.get('evidence_sources',{}).get(src)=='available' for t in tasks) for src in ('msa','structure','literature')}},
        'interpretation':'All means all AVAILABLE frozen evidence with explicit missingness. N=50 is a nested evaluation of the same predictions; repeats are fresh draws with fixed generation_seed. Seed is best-effort, not a determinism guarantee.'}
    save(root/'progress.json',summary)
    lines=['# Stage B reproducible paired experiment','',f"State: {state}",f"Units completed: {summary['completed_units']} / {summary['planned_units']}",
           f"Reported tokens including prior campaigns: {summary['ledger']['reported_tokens']:,}",
           f"Budget accounted including unknown-cost reserves: {summary['ledger']['budget_accounted_tokens']:,} / {config['token_budget']:,}",
           '', 'Completed pairs only; failed conditions are never imputed. Primary summaries require all three repeats per assay. N=50 and N=250 are dependent nested evaluations.', '']
    for comp in summary['comparisons']:
        lines.append(f"N={comp['n']}: {comp['paired_replicates']} paired repeats; {comp['assays_with_all_three_repeats']} assays with all 3 repeats.")
        if 'family_macro_delta_spearman' in comp:lines.append(f"Family macro delta Spearman: {comp['family_macro_delta_spearman']:.5f}")
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')
    return summary


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True);parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args();config,manifest,root,fingerprint=validate_plan(args.config.resolve());tasks=manifest['tasks']
    if args.dry_run:
        print(json.dumps({'status':'dry_run','tasks':len(tasks),'initial_requests':sum(len(t['templates']['cr'])*2*3 for t in tasks),'generation_seed':1,'api_calls':0,'fingerprint':fingerprint}));return
    for line in Path(config['credentials_file']).read_text().splitlines():
        line=line.strip().removeprefix('export ')
        if line.startswith('GEMINI_API_KEY='):os.environ['GEMINI_API_KEY']=line.split('=',1)[1].strip().strip('\"\'')
    with (root/'campaign.lock').open('a+') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        gateway=gw.RequestGateway(model_config=config['model_config'],run_dir=root/'provider',fingerprint=fingerprint,
                                  token_budget=config['token_budget'],reserve_budget=True,prior_usage=config['prior_usage'])
        state='complete';current=None
        try:
            smoke=gateway.generate(system='API connectivity and explicit generation seed check.',user='Reply with exactly OK.',run_name='seed-smoke',metadata={'phase':'generation_seed_smoke'})
            if smoke['status']!='ok' or smoke['final_text'].strip()!='OK':raise RuntimeError('Seed smoke did not return OK')
            save(root/'seed_smoke.json',{'status':'accepted','generation_seed':1,'request_sha256':smoke['request_sha256'],
                'model_version':smoke.get('response_metadata',{}).get('model_version'),'note':'Accepted API request; does not prove deterministic decoding.'})
            for rep in range(1,4):
                for task_index,task in enumerate(tasks):
                    for index in range(len(task['templates']['cr'])):
                        order=['cr','all'] if (rep+task_index+index)%2 else ['all','cr']
                        for cond in order:
                            current={'assay':task['assay'],'replicate':rep,'unit':index,'condition':cond}
                            value=run_unit(gateway,task,cond,rep,index,task['templates'][cond][index],config,root)
                            print(json.dumps({**current,'status':value['status'],'ledger_tokens':gateway.summary()['reported_tokens']}),flush=True)
                        report(tasks,config,root,gateway,'running',current)
        except Exception as exc:
            state='paused' if isinstance(exc,ExecutionPaused) else 'error'
            save(root/'failure.json',{'type':type(exc).__name__,'reason':str(exc) if isinstance(exc,ExecutionPaused) else 'Inspect local traceback; no automatic scientific changes','current':current})
            if state=='error':
                import traceback
                traceback.print_exc()
        finally:
            result=report(tasks,config,root,gateway,state,current);gateway.close()
        print(json.dumps({'status':state,'completed_units':result['completed_units'],'statuses':result['statuses']}),flush=True)
        if state!='complete':raise SystemExit(2)

if __name__=='__main__':main()
