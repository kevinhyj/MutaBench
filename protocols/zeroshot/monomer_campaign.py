"""Versioned multi-model monomer campaigns. Prepare/validate never use credentials.

Run requires --execute, enforces long waits, durable single attempts and exact
accepted messages. It never reads labels. Use evaluate_monomers in a separate
process for complete-cohort metrics.
"""
import argparse
import collections
import concurrent.futures as cf
import copy
import fcntl
import json
from pathlib import Path
import shutil
import sys

from shared_protocol import save, sha, digest, require
from monomer_integrity import (strict_load as load, VERSION as INTEGRITY_VERSION, IDENTITY,
    task_metric, check_request, check_request_set, check_receipt, check_result_inventory)
from frozen_prompt_bundle import message_sha256
from multillm_transport import now

HERE = Path(__file__).resolve().parent
DEFAULT_MODELS = ('flash38_low','flash38_medium','flash38_high','argon_high','qwen38_27b_high','sol61_max','deepseek41_flash_max')


def bundle_load(path):
    schema = load(path/'MANIFEST.json')['schema']
    if schema == 'evagent-monomer-sequences-3':
        from monomer_sequence_semantics_v3 import validate_bundle
    elif schema == 'evagent-admet-molecules-2':
        from admet_protocol_v2 import validate_bundle
    elif schema == 'evagent-admet-molecules-1':
        from admet_protocol import validate_bundle
    else:
        from shared_protocol import validate_bundle
    return validate_bundle(path)


def sources():
    names = ('monomer_campaign.py','monomer_sequence_semantics_v3.py','admet_protocol.py','admet_protocol_v2.py','admet_endpoints_v2.json',
        'evaluate_monomers.py','monomer_integrity.py','prepare_monomer_sequences.py',
        'shared_protocol.py','prepare_shared_prompts.py','prompt_semantics.py','msa_missingness.py','rna_confidence.py',
        'prepare_protein_rna_contact.py','zeroshot_msa.py','scoring_validation.py','frozen_prompt_bundle.py',
        'multillm_transport.py','deepseek_official_transport.py','rna_msa_missingness.py','monomer_progress.py')
    paths = [HERE/n for n in names] + [HERE/'prompts/admet_v2/system.txt', HERE/'prompts/shared_fields_v1_draft/system_prompt.txt',
        HERE/'audits/feature_contract_20261014_v1/PROFILES.json']
    return {str(p):sha(p) for p in paths}


def requests_from_queue(queue):
    result = []
    for q in queue:
        messages = [dict(role='system',content=q['template']['system']),dict(role='user',content=q['template']['user'])]
        result.append(dict(request_id=q['logical_id'],domain=q['domain'],assay=q['assay'],arm=q['arm'],
            chunk_index=q['chunk_index'],round_index=q.get('round_index',0),expected_ids=q['template']['expected_ids'],
            messages=messages,messages_sha256=message_sha256(messages)))
        check_request(result[-1])
    return result


def check_admet_references(path, manifest, references):
    if 'admet' not in manifest['domains']:
        return
    tasks = {t['endpoint']:t for t in load(path/'tasks.json')}
    rows = [r for r in references if r['domain']=='admet']
    require(len(rows)==len(tasks) and {r['assay'] for r in rows}==set(tasks),
            'ADMET evaluator scope mismatch')
    for ref in rows:
        kind=tasks[ref['assay']]['kind']
        require(ref['kind']==kind and ref['metric']==('auroc' if kind=='classification' else 'spearman')
                and ref['label_container']=='admet_endpoint_y' and ref['label_key']==ref['assay'],
                'ADMET evaluator semantics differ from endpoint task')


def validate_models(models):
    require(bool(models), 'Empty model matrix')
    for key, model in models.items():
        require(key and model['provider'] in {'google','openai_compatible'}, 'Unknown model provider')
        require(type(model['timeout_s']) is int and model['timeout_s'] >= 3600, 'Generation wait must be >=3600s')
        require(model['seed'] == 42 and model['automatic_retries'] == 0, 'Unexpected seed/retry contract')
        require(model['stream'] is True and model['endpoint'].startswith('https://'), 'Streaming HTTPS required')
        require(not {'api_key','key','token','authorization'} & set(model), 'Credentials must not enter PLAN')
        if key == 'sol61_max': require(model['thinking_level'] == 'max', 'Sol requires max')


def prepare(run, inputs, models, workers):
    run = Path(run).resolve()
    require(not run.exists(), 'New run directory required')
    require(type(workers) is int and 1 <= workers <= 200, 'Invalid explicit concurrency')
    validate_models(models)
    requests, refs, bundles, identities = [], [], [], set()
    for path in inputs:
        path = Path(path).resolve(); manifest, queue = bundle_load(path)
        require(set(manifest['domains']) <= {'protein','rna','admet'}, 'Monomer domains only')
        rp = HERE/'evaluator'/manifest['version']/'LABEL_REFERENCES.json'
        references = load(rp); by_key = {(r['domain'],r['assay']):r for r in references}
        check_admet_references(path, manifest, references)
        require(len(by_key) == len(references), 'Duplicate evaluator reference')
        keys = {(u['domain'],u['assay']) for u in queue}
        require(set(by_key) == keys and not keys & identities, 'Overlapping/missing assay scope')
        identities.update(keys)
        for key, ref in by_key.items():
            task_metric(key[0],ref)
            qp = path/'queries'/key[0]/key[1]/'query.json'
            require(Path(ref['query_path']).resolve() == qp and sha(qp) == ref['query_sha256'], 'Evaluator query mismatch')
            if key[0] == 'admet':
                require(ref['metric'] == ('auroc' if ref['kind'] == 'classification' else 'spearman'), 'Endpoint metric mismatch')
            refs.append(ref)
        bundles.append(dict(path=str(path),manifest_sha256=sha(path/'MANIFEST.json'),
                            references_path=str(rp),references_sha256=sha(rp)))
        requests.extend(requests_from_queue(queue))
    check_request_set(requests)
    # Validate provider payload construction without opening a credential file.
    for model in models.values():
        if model['model'] == 'deepseek-flash': from deepseek_official_transport import payload
        else: from multillm_transport import payload
        for request in requests: payload(model, request)
    run.mkdir(parents=True)
    save(run/'REQUESTS.json', requests); save(run/'EVALUATOR_REFERENCES.json',refs)
    source_hashes = sources()
    for path in source_hashes:
        source = Path(path); dest = run/'source_snapshot'/source.relative_to(HERE)
        dest.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(source,dest)
    plan = dict(schema='monomer-campaign-2',integrity_version=INTEGRITY_VERSION,
        created_utc=now(),models=models,workers=workers,bundles=bundles,
        requests_sha256=sha(run/'REQUESTS.json'),references_sha256=sha(run/'EVALUATOR_REFERENCES.json'),
        source_hashes=source_hashes,planned_requests=len(requests)*len(models),
        preflight_requests=len(models),domains=dict(collections.Counter(d for d,_ in identities)),
        retries=0,prompt_repair=False,provider_fallback=False,labels_in_predictor=False,
        protocol='Frozen shared sequence four arms; ADMET paired SMILES/3D with exact historical 3-round groups',
        generation_seed=42,admet_conformer_seed=1,
        stop_policy='PAUSE stops new dispatch; fatal model errors or 16 technical errors in last20 stop that model; drain active requests')
    save(run/'PLAN.json',plan); (run/'PLAN.sha256').write_text(sha(run/'PLAN.json')+'\n')
    save(run/'STATUS.json',dict(state='prepared_not_started',updated_utc=now(),planned=plan['planned_requests'],active=0,completed=0))
    validate(run)
    return plan


def validate(run):
    run = Path(run); plan = load(run/'PLAN.json')
    require(plan.get('schema')=='monomer-campaign-2' and plan.get('integrity_version')==INTEGRITY_VERSION,
            'New campaign version required for strict result integrity')
    require(sha(run/'PLAN.json') == (run/'PLAN.sha256').read_text().strip(), 'Plan changed')
    require(plan['source_hashes'] == sources(), 'Campaign implementation changed')
    validate_models(plan['models'])
    require(sha(run/'REQUESTS.json') == plan['requests_sha256'], 'Requests changed')
    require(sha(run/'EVALUATOR_REFERENCES.json') == plan['references_sha256'], 'Evaluator references changed')
    expected = []; refs = []
    for descriptor in plan['bundles']:
        p = Path(descriptor['path'])
        require(sha(p/'MANIFEST.json') == descriptor['manifest_sha256'], 'Input manifest changed')
        manifest, queue = bundle_load(p); expected.extend(requests_from_queue(queue))
        rp = Path(descriptor['references_path'])
        require(sha(rp) == descriptor['references_sha256'], 'Source evaluator references changed')
        references = load(rp)
        check_admet_references(p, manifest, references)
        require(len({(r['domain'],r['assay']) for r in references})==len(references),'Duplicate evaluator reference')
        for ref in references:
            task_metric(ref['domain'],ref)
            qp=p/'queries'/ref['domain']/ref['assay']/'query.json'
            require(Path(ref['query_path']).resolve()==qp and sha(qp)==ref['query_sha256'],'Evaluator query mismatch')
        refs.extend(references)
    require(expected == load(run/'REQUESTS.json'), 'Run requests differ from accepted inputs')
    require(refs == load(run/'EVALUATOR_REFERENCES.json'), 'Copied evaluator references differ')
    require(len({(r['domain'],r['assay']) for r in refs})==len(refs),'Overlapping evaluator scope')
    require({(r['domain'],r['assay']) for r in refs}=={(r['domain'],r['assay']) for r in expected},
            'Evaluator scope differs from predictor scope')
    check_request_set(expected)
    return plan, expected


def write_unit(run, key, request, account, value):
    check_request(request)
    require(account.get('request_id')==request['request_id'] and
            account.get('messages_sha256')==request['messages_sha256'],'Transport receipt belongs to another request')
    row=dict(integrity_version=INTEGRITY_VERSION,model_key=key,identity={k:request[k] for k in IDENTITY},
        request_id=request['request_id'],messages_sha256=request['messages_sha256'],
        status=account['status'],failure_category=account.get('failure_category'),value=value)
    check_receipt(row,request,key)
    path = run/'models'/key/'units'/(request['request_id']+'.json')
    if path.exists():
        require(path.with_suffix('.sha256').is_file() and sha(path)==path.with_suffix('.sha256').read_text().strip()
                and load(path)==row,'Conflicting existing result; new attempt/version required')
        return
    save(path,row)
    path.with_suffix('.sha256').write_text(sha(path)+'\n')


def submit_one(run, key, model, request):
    check_request(request)
    folder = run/'models'/key/'attempts'/request['request_id']
    if folder.exists():
        # Restart may recover a settled receipt, but never repeat an ambiguous attempt.
        if not (folder/'ACCOUNTING.json').exists() or not (folder/'RESULT.json').exists():
            return dict(status='failed',failure_category='unsettled_saved_attempt')
        account = load(folder/'ACCOUNTING.json'); result = load(folder/'RESULT.json')
        require(account['messages_sha256'] == request['messages_sha256'], 'Saved attempt input mismatch')
        write_unit(run,key,request,account,result['value']); return account
    if model['model'] == 'deepseek-flash': from deepseek_official_transport import submit
    else: from multillm_transport import submit
    account,value = submit(folder,model,request)
    write_unit(run,key,request,account,value)
    return account


def run_campaign(run):
    run = Path(run).resolve()
    with (run/'runner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        plan, requests = validate(run)
        for key in plan['models']:check_result_inventory(run/'models'/key/'units',requests)
        stopped = {}; recent = collections.defaultdict(lambda:collections.deque(maxlen=20))
        # One durable neutral preflight per model; reuse only a completed successful one.
        for key, model in plan['models'].items():
            if (run/'PAUSE').exists(): stopped[key]='paused'; continue
            folder = run/'preflight'/key
            if folder.exists():
                account = load(folder/'ACCOUNTING.json') if (folder/'ACCOUNTING.json').exists() else {'status':'unknown'}
            else:
                messages=[dict(role='system',content='Return only the requested JSON.'),
                    dict(role='user',content='Return {"variants":[{"variant_id":"neutral","predicted_score":0.0}]}')]
                request=dict(request_id='neutral',messages=messages,messages_sha256=message_sha256(messages),expected_ids=['neutral'])
                if model['model']=='deepseek-flash': from deepseek_official_transport import submit
                else: from multillm_transport import submit
                account,_=submit(folder,model,request)
            if account['status']!='ok': stopped[key]='preflight_failed_or_unknown'
        tasks=collections.deque((k,r) for r in requests for k in plan['models'])
        active={}; completed=valid=held=0
        def report(state):
            save(run/'STATUS.json',dict(state=state,updated_utc=now(),planned=plan['planned_requests'],
                completed=completed,valid=valid,active=len(active),pending=len(tasks),held=held,stopped_models=stopped))
        report('running')
        with cf.ThreadPoolExecutor(max_workers=plan['workers']) as pool:
            while tasks or active:
                if (run/'PAUSE').exists():
                    for k in plan['models']: stopped[k]='paused'
                while tasks and len(active)<plan['workers']:
                    key,request=tasks.popleft()
                    if key in stopped: held+=1; continue
                    if plan['source_hashes'] != sources():
                        stopped.update({k:'implementation_changed' for k in plan['models']}); break
                    future=pool.submit(submit_one,run,key,plan['models'][key],request);active[future]=(key,request)
                if not active: break
                done,_=cf.wait(active,timeout=5,return_when=cf.FIRST_COMPLETED)
                for future in done:
                    key,request=active.pop(future)
                    try: account=future.result()
                    except Exception as exc:
                        account=dict(status='failed',failure_category='local_error',error=type(exc).__name__+': '+str(exc))
                        save(run/'local_errors'/key/(request['request_id']+'.json'),account)
                    completed+=1;valid+=account['status']=='ok'
                    category=account.get('failure_category');recent[key].append(category)
                    if category in {'authentication_or_permission','quota','invalid_request','model_identity_mismatch','local_error','unsettled_saved_attempt','explicit_block'}:
                        stopped[key]=category
                    elif len(recent[key])==20 and sum(c not in {None,'ok'} for c in recent[key])>=16:
                        stopped[key]='technical_failure_circuit_breaker'
                report('running')
        report('finished' if valid==plan['planned_requests'] else 'stopped_incomplete')


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['prepare','validate','run']);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--inputs',type=Path,nargs='+');p.add_argument('--model-keys',nargs='+',default=list(DEFAULT_MODELS))
    p.add_argument('--workers',type=int,default=20);p.add_argument('--execute',action='store_true')
    a=p.parse_args()
    if a.action=='prepare':
        p.error('--inputs required') if not a.inputs else None
        models=load(HERE/'llm_models.json')['models']
        plan=prepare(a.run,a.inputs,{k:copy.deepcopy(models[k]) for k in a.model_keys},a.workers)
        print(dict(status='prepared_not_started',requests=plan['planned_requests'],models=list(plan['models'])))
    elif a.action=='validate': print(dict(status='PASS',requests=len(validate(a.run)[1])))
    elif a.execute: run_campaign(a.run)
    else: print('Preview only. Add --execute to submit the frozen campaign.')
