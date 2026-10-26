"""Outcome-independent structural-summary ablation and shuffled-input control."""
import argparse,collections,copy,datetime,fcntl,hashlib,importlib.util,json,os,sqlite3
from pathlib import Path
HERE=Path(__file__).resolve().parent
BASE=HERE.parent/'reproducible_v4'
spec=importlib.util.spec_from_file_location('v5_base_runner',BASE/'runner.py');r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
ARMS=['cr','structure_raw']
FIELDS=['mfe_kcal_mol','ensemble_kcal_mol','ensemble_diversity','delta_mfe_kcal_mol','base_pair_distance','mean_pair_probability_change']
MARKER='Frozen multi-source evidence (text/numeric; observations, not fitness labels):\n'


def transform(template,arm,assay,index):
    t=copy.deepcopy(template)
    if arm=='cr':return t,{}
    if arm!='structure_raw':raise ValueError('Unknown arm')
    raw=r.load(HERE/'raw_structures.json')[assay]
    evidence={k:copy.deepcopy(raw[k]) for k in ('wt_secondary_structure','wt_tertiary_structure')}
    evidence['variant_secondary_structures']={vid:raw['variant_secondary_structures'][vid] for vid in t['expected_ids']}
    t['user']+='\n\nStructural input: WT heavy-atom Cartesian coordinates in angstroms and predicted secondary structures. Dot-bracket positions follow the corresponding full RNA sequence.\n'+json.dumps(evidence,sort_keys=True,separators=(',',':'))
    t['metadata']={**t.get('metadata',{}),'evidence_mode':'raw_structure_only','prompt_sha256':r.gw.digest([t['system'],t['user']])}
    return t,{}


def freeze(root,source):
    if (root/'campaign.json').exists():raise RuntimeError('Never overwrite frozen probe')
    if r.load(source/'progress.json')['status']!='complete':raise RuntimeError('Wait for v4 completion; do not stop or overlap it')
    r.validate_plan(source/'campaign.json')
    root.mkdir(parents=True,exist_ok=True)
    with (source/'campaign.lock').open('a+') as lease,(source/'provider/request_ledger.lock').open('a+') as gl:
        fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(gl,fcntl.LOCK_EX|fcntl.LOCK_NB)
        db=sqlite3.connect(f'file:{source}/provider/request_ledger.sqlite?mode=ro',uri=True)
        if db.execute("select count(*) from attempts where status='inflight'").fetchone()[0]:raise RuntimeError('Provider in flight')
        snapshot=root/'prior_v4_ledger.sqlite';dest=sqlite3.connect(snapshot);db.backup(dest);dest.close()
        prior=json.loads(db.execute("select value from run_state where key='prior_usage'").fetchone()[0])
        prior['ledgers'].append({'path':str(snapshot),'sha256':r.file_hash(snapshot)})
        for status,tokens,elapsed,request_json in db.execute("select status,tokens,elapsed_s,request_json from attempts where status!='not_submitted'"):
            prior['provider_attempts']+=1;prior['reported_tokens']+=tokens or 0;prior['provider_elapsed_s']+=elapsed or 0
            prior['requests_by_status'][status]=prior['requests_by_status'].get(status,0)+1
            if status in {'failed_unknown_cost','interrupted_unknown_cost','unknown_cost'} or tokens is None:
                q=json.loads(request_json);reserve=sum(len(m['content'].encode()) for m in q['messages'])+2*q['model_config']['max_output_tokens']+1024
                prior['unknown_cost_reserve_tokens']=prior.get('unknown_cost_reserve_tokens',0)+max(0,reserve-(tokens or 0))
                prior['unknown_cost_attempts']=prior.get('unknown_cost_attempts',0)+1
        db.close()
    old=r.load(source/'campaign.json');tasks=r.load(source/'tasks.json')['tasks'];receipts=[]
    for task in tasks:
        original=task['templates']['cr'];task['templates']={arm:[] for arm in ARMS}
        for index,t in enumerate(original):
            for arm in ARMS:
                transformed,mapping=transform(t,arm,task['assay'],index);task['templates'][arm].append(transformed)
                receipts.append({'assay':task['assay'],'unit':index,'arm':arm,'donors':mapping,'original_prompt_sha256':r.gw.digest([t['system'],t['user']]),'prompt_sha256':r.gw.digest([transformed['system'],transformed['user']]),'original_bytes':len(t['user'].encode()),'transformed_bytes':len(transformed['user'].encode())})
    r.save(root/'tasks.json',{'tasks':tasks});r.save(root/'interventions.json',receipts)
    prior_accounted=prior['reported_tokens']+prior.get('unknown_cost_reserve_tokens',0)
    if prior_accounted>=50000000:raise RuntimeError('Global budget exhausted')
    cfg={**old,'schema':'stage-b-simple-structure-v5','conditions':ARMS,'repeats':1,'prior_usage':prior,
         'token_budget':min(50000000,prior_accounted+15000000),'global_token_cap':50000000,'incremental_budget_cap':15000000,
         'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
         'runtime_hashes':{str(f):r.file_hash(f) for f in (HERE/'probe.py',HERE/'prepare.py',BASE/'runner.py',BASE/'gateway.py')},
         'asset_hashes':{**old['asset_hashes'],str(HERE/'raw_structures.json'):r.file_hash(HERE/'raw_structures.json'),str(HERE/'preparation_audit.json'):r.file_hash(HERE/'preparation_audit.json'),str(HERE/'requirements.lock'):r.file_hash(HERE/'requirements.lock'),str(root/'tasks.json'):r.file_hash(root/'tasks.json'),str(root/'interventions.json'):r.file_hash(root/'interventions.json'),str(snapshot):r.file_hash(snapshot)},
         'design':{'selection':'All 21 source-admitted V4 tasks; no outcome selection','primary':'single-repeat paired structure_raw minus contemporaneous CR, family-macro N250 Spearman','input':'task context and full sequences; WT heavy-atom XYZ coordinates and full WT/candidate dot-bracket only','excluded':'MSA, literature, energies, ensemble metrics, scalar differences, confidence scores and contact summaries','N50':'same predictions, first two groups','initial_requests':404,'repeats':1,'limitation':'textual atomic coordinates, no native images; secondary structures are pseudoknot-free; prior WT 3D may depend on MSA; one-repeat exploratory design'}}

    r.save(root/'campaign.json',cfg);r.save(root/'unknown_response_policy.json',{'mode':'retry_then_continue','max_attempts':3,'backoff_seconds':[15,30]})
    print(json.dumps({'status':'frozen','tasks':len(tasks),'initial_requests':sum(len(t['templates']['cr'])*len(ARMS) for t in tasks),'prior_accounted':prior_accounted,'effective_total_budget':cfg['token_budget']}))


def validate(path):
    cfg=r.load(path);root=path.parent
    if cfg['schema']!='stage-b-simple-structure-v5' or cfg['conditions']!=ARMS or cfg['repeats']!=1:raise ValueError('Probe design changed')
    for records in (cfg['runtime_hashes'],cfg['asset_hashes']):
        for f,sha in records.items():
            if r.file_hash(Path(f))!=sha:raise ValueError('Frozen file changed: '+f)
    tasks=r.load(root/'tasks.json')['tasks']
    if len(tasks)!=21 or any(t['source_status']!='admitted' for t in tasks):raise ValueError('Source scope changed')
    for t in tasks:
        for i,original in enumerate(t['templates']['cr']):
            for arm in ARMS:
                if transform(original,arm,t['assay'],i)[0]!=t['templates'][arm][i]:raise ValueError('Intervention mismatch')
    return cfg,tasks,root


def report(tasks,cfg,root,gateway,state,current):
    import numpy as np
    rows=r.result_rows(tasks,cfg,root);comp=[]
    task_map={t['assay']:t for t in tasks}
    for row in rows:
        row['evidence_availability']={'structure':'available'} if row['condition']=='structure_raw' else {}
        row['evidence_bundle_sha256']=r.file_hash(HERE/'raw_structures.json') if row['condition']=='structure_raw' else None
        row['intervention']=row['condition']
    for n in sorted({x['n'] for x in rows}):
        lookup={(x['assay'],x['condition']):x for x in rows if x['n']==n and x['status']=='ok'}
        for arm in ARMS[1:]:
            pairs=[]
            for task in tasks:
                a=lookup.get((task['assay'],arm));b=lookup.get((task['assay'],'cr'))
                if a and b:pairs.append({'assay':task['assay'],'family':task['family'],'delta_spearman':a['metrics']['spearman']-b['metrics']['spearman'],'delta_kendall':a['metrics']['kendall_tau']-b['metrics']['kendall_tau']})
            out={'n':n,'contrast':arm+' minus cr','paired_assays':len(pairs),'pairs':pairs,'interpretation':'single-repeat mechanistic pilot; families and N sizes are not independent repetitions'}
            if pairs:
                fs=sorted({x['family'] for x in pairs});means=[float(np.mean([x['delta_spearman'] for x in pairs if x['family']==f])) for f in fs]
                out.update(families=len(fs),family_macro_delta_spearman=float(np.mean(means)))
                if len(means)>=3:
                    boot=np.random.default_rng(1).choice(means,(2000,len(means)),replace=True).mean(axis=1);out['family_bootstrap_95pct']=[float(x) for x in np.quantile(boot,[.025,.975])]
            comp.append(out)
    value={'status':state,'timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'current':current,'completed_units':len(list((root/'units').glob('*.json'))),'planned_units':sum(len(t['templates']['cr'])*len(ARMS) for t in tasks),'statuses':dict(collections.Counter(x['status'] for x in rows)),'rows':rows,'comparisons':comp,'ledger':gateway.summary()}
    r.save(root/'progress.json',value);return value


def run(path,dry=False):
    cfg,tasks,root=validate(path)
    if dry:print(json.dumps({'status':'dry_run','tasks':len(tasks),'initial_requests':sum(len(t['templates']['cr'])*len(ARMS) for t in tasks),'api_calls':0}));return
    for line in Path(cfg['credentials_file']).read_text().splitlines():
        line=line.strip().removeprefix('export ')
        if line.startswith('GEMINI_API_KEY='):os.environ['GEMINI_API_KEY']=line.split('=',1)[1].strip().strip('\"\'')
    with (root/'campaign.lock').open('a+') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        gateway=r.gw.RequestGateway(model_config=cfg['model_config'],run_dir=root/'provider',fingerprint=r.gw.digest(cfg),token_budget=cfg['token_budget'],reserve_budget=True,prior_usage=cfg['prior_usage'])
        state='complete';current=None
        try:
            smoke=gateway.generate(system='API connectivity and explicit generation seed check.',user='Reply with exactly OK.',run_name='probe-smoke')
            if smoke['status']!='ok' or smoke['final_text'].strip()!='OK':raise RuntimeError('Connectivity failure')
            for ti,task in enumerate(tasks):
                for i in range(len(task['templates']['cr'])):
                    k=(ti+i)%len(ARMS);order=ARMS[k:]+ARMS[:k]
                    for arm in order:
                        current={'assay':task['assay'],'arm':arm,'unit':i};v=r.run_unit(gateway,task,arm,1,i,task['templates'][arm][i],cfg,root)
                        print(json.dumps({**current,'status':v['status']}),flush=True)
                    report(tasks,cfg,root,gateway,'running',current)
        except Exception as exc:
            state='paused' if isinstance(exc,r.ExecutionPaused) else 'error';r.save(root/'failure.json',{'type':type(exc).__name__,'current':current})
            if state=='error':
                import traceback;traceback.print_exc()
        finally:
            report(tasks,cfg,root,gateway,state,current);gateway.close()
        if state!='complete':raise SystemExit(2)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['freeze','run']);parser.add_argument('--root',type=Path,required=True);parser.add_argument('--source',type=Path);parser.add_argument('--dry-run',action='store_true');args=parser.parse_args()
    if args.action=='freeze':freeze(args.root.resolve(),args.source.resolve())
    else:run(args.root.resolve()/'campaign.json',args.dry_run)
