"""Frozen multi-provider scoring campaign; prediction never opens evaluator labels."""
import argparse
import collections
import concurrent.futures as cf
import fcntl
import json
import os
from pathlib import Path
import shutil
import sys

from frozen_prompt_bundle import message_sha256
from shared_protocol import load_requests, validate_bundle, validate_subset
from run_frozen import validate_evaluator
from multillm_transport import load, save, sha, now, submit, credential

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
NAME = 'multillm_final_four_arm_c20_20261015_v1'
FATAL = {'authentication_or_permission','quota','invalid_request','model_identity_mismatch','runner_exception'}
TECHNICAL = {'network_ambiguous','timeout_ambiguous','rate_limit','provider_error','malformed_stream','incomplete_output','format','empty_final'}


def configured_models(profile='legacy_four'):
    models = {}
    configurations = [('argon_high','gemini-4-argon','high'),
                             ('qwen38_27b_xhigh','qwen3.8-27b','xhigh'),
                             ('flash38_low','gemini-3.8-flash','low'),
                             ('flash38_high','gemini-3.8-flash','high')]
    if profile in {'ppi_rna_five','ppi_rna_six'}:
        configurations = [('flash38_low','gemini-3.8-flash','low'),
                          ('flash38_medium','gemini-3.8-flash','medium'),
                          ('flash38_high','gemini-3.8-flash','high'),
                          ('argon_high','gemini-4-argon','high'),
                          ('qwen38_27b_high','qwen3.8-27b','high')]
        if profile=='ppi_rna_six':
            configurations.append(('sol61_max','gpt-6.1-sol','max'))
    elif profile != 'legacy_four':
        raise ValueError('Unknown model profile')
    for key,model,level in configurations:
        google = model.startswith('gemini')
        models[key] = dict(model=model,thinking_level=level,provider='google' if google else 'openai_compatible',
            credential_profile='existing_google_GEMINI_API_KEY' if google else 'ypxia_27b',
            credential_path=str(ROOT/('private_config/evagent/credentials.env' if google else 'private_config/evagent/private/ypxia_27b.json')),
            endpoint='https://generativelanguage.googleapis.com/v1beta/models/'+model+':streamGenerateContent?alt=sse' if google else 'https://llm-ukpk3ktp21jd8tmd.cn-beijing.maas.aliyuncs.com/compatible-mode/v1/chat/completions',
            proxy='http://127.0.0.1:7777' if google else None,seed=42,temperature=0,
            timeout_s=3600 if google else 7200,max_output_tokens=65536 if google else 32768,
            stream=True,automatic_retries=0,enable_thinking=True,
            thinking_basis=f'Explicit {level} requested; actual provider acceptance recorded by neutral preflight')
        if key=='sol61_max':
            models[key].update(credential_profile='hyj_pro',credential_path=str(ROOT/'private_config/evagent/private/hyj_pro.json'),
                endpoint='https://114.55.8.157/codex-ws/v1/chat/completions',proxy=None,temperature=None,
                timeout_s=3600,max_output_tokens=128000,request_style='sol_chat',enable_thinking=None,
                thinking_basis='User requested reuse of the currently running Sol few-shot configuration: hyj_pro, reasoning_effort=max; no fallback route')
    return models


def source_paths():
    return [HERE/n for n in ['multillm_campaign.py','multillm_transport.py','multillm_supervise.py','frozen_prompt_bundle.py','evaluate_frozen.py','scoring_validation.py','revalidate_campaign.py','shared_protocol.py','prepare_shared_prompts.py','run_frozen.py']] + [
        ROOT/'code/evagent/integrated'/n for n in ['chunked_scoring.py','ranking.py','self_check.py']]


def prepare(run, inputs=None, model_profile='legacy_four', resume_from=None):
    os.umask(0o077)
    if run.exists(): raise FileExistsError('Use a new campaign version')
    if not run.is_relative_to(HERE/'runs'): raise ValueError('Pipeline runs directory required')
    if inputs is None: raise ValueError('Explicit --inputs with current shared acceptance required; historical registry is not a new-run default')
    inputs=Path(inputs).resolve()
    manifest,queue=validate_bundle(inputs)
    requests = load_requests(inputs)
    refs=HERE/'evaluator'/manifest['version']/'LABEL_REFERENCES.json'
    validate_evaluator(refs,inputs,queue)
    models = configured_models(model_profile)
    if resume_from is not None:
        resume_from=Path(resume_from).resolve()
        old=load(resume_from/'PLAN.json')
        if sha(resume_from/'PLAN.json')!=(resume_from/'PLAN.sha256').read_text().strip():raise ValueError('Previous plan changed')
        if not set(old['models'])<=set(models) or any(models[k]!=v for k,v in old['models'].items()) or old['input_manifest_sha256']!=sha(inputs/'MANIFEST.json'):
            raise ValueError('Existing model configurations and input must match exactly; only new models may be added')
    for model in models.values(): credential(model)  # Validate availability without printing or persisting values.
    run.mkdir(parents=True)
    save(run/'REQUESTS.json',requests)
    tasks = [dict(model_key=k,request_id=r['request_id'],domain=r['domain'],assay=r['assay'],arm=r['arm']) for r in requests for k in models]
    save(run/'TASKS.json',tasks)
    system_tasks={}
    if 'sol61_max' in models:
        for domain in sorted({r['domain'] for r in requests}):
            selected=[r for r in requests if r['domain']==domain]
            path=run/'system_tasks'/('sol61_max_'+domain+'.json')
            save(path,dict(model_key='sol61_max',domain=domain,assays=len({r['assay'] for r in selected}),
                chunk_size=25,requests=len(selected),request_ids=[r['request_id'] for r in selected],
                arms=sorted({r['arm'] for r in selected}),candidate_count=sum(len(r['expected_ids']) for r in selected),
                scoring='Independent continuous scores, no within-chunk renormalization; join frozen IDs before full-assay evaluation',
                inputs=str(inputs),input_manifest_sha256=sha(inputs/'MANIFEST.json')))
            system_tasks[domain]=dict(path=str(path),sha256=sha(path))
    for key,model in models.items():
        child=run/'models'/key
        save(child/'QUEUE.json',queue)
        save(child/'PLAN.json',dict(schema='frozen-multillm-model-run-v1',model_key=key,model_config=model,
            inputs=str(inputs),input_manifest_sha256=sha(inputs/'MANIFEST.json'),queue_sha256=sha(child/'QUEUE.json'),
            evaluator_references=str(refs),evaluator_references_sha256=sha(refs),initial_units=len(queue),
            protocol_version=manifest['protocol_version'],acceptance=manifest['acceptance'],
            candidate_resampling=False,test_labels_in_model_inputs=False,max_attempts_per_unit=1))
    sources = {str(p):sha(p) for p in source_paths()}
    for p in source_paths():
        rel=p.relative_to(ROOT);target=run/'source_snapshot'/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target)
    plan=dict(schema='frozen-multillm-campaign-v1',created_utc=now(),models=models,workers=20,
        model_profile=model_profile,
        concurrency_scope='Global total across all model configurations in this campaign; not per model and not shared with unrelated experiments',
        inputs=str(inputs),input_manifest_sha256=sha(inputs/'MANIFEST.json'),
        protocol_version=manifest['protocol_version'],acceptance=manifest['acceptance'],
        requests_sha256=sha(run/'REQUESTS.json'),tasks_sha256=sha(run/'TASKS.json'),
        model_plan_hashes={k:sha(run/'models'/k/'PLAN.json') for k in models},
        source_hashes=sources,planned_scoring_requests=len(tasks),planned_preflight_requests=len(set(models)-set(old['models'])) if resume_from else len(models),
        added_model_keys=sorted(set(models)-set(old['models'])) if resume_from else list(models),system_task_indexes=system_tasks,
        resume_from=str(resume_from) if resume_from else None,
        resume_plan_sha256=sha(resume_from/'PLAN.json') if resume_from else None,
        resume_policy='Wait for prior calls to drain, locally revalidate raw responses, reuse attempts and preflight, submit only never-attempted requests' if resume_from else None,
        preflight_policy='One neutral JSON score-format test per model configuration before its first biological scoring request',
        sampling='Accepted frozen query order, seed42; CR25, no resampling or label-dependent selection',
        dispatch='Round-robin model order within each frozen request; continuously refill up to global20',
        stop_policy='Stop affected model on authentication/quota/invalid request/identity/runner error; or >=16 technical failures in its last20 settled calls. Drain active calls. Do not retry explicit blocks or ambiguous calls.',
        retry_policy='Single attempt, no implicit transport retry, no prompt repair, no fallback provider. Saved attempts are never resubmitted by restart.',
        domains=manifest['domains'],
        admet=dict(status='separate_endpoint_protocol_required',reason='No MSA four-arm task; classification requires AUROC, not binding Spearman'),
        execution_scope='Explicit accepted input bundle; prepare does not submit requests')
    save(run/'PLAN.json',plan);(run/'PLAN.sha256').write_text(sha(run/'PLAN.json')+'\n')
    save(run/'STATUS.json',dict(state='prepared',updated_utc=now(),planned=len(tasks),completed=0,valid=0,active=0))
    index=dict(schema='zeroshot-llm-model-index-v1',updated_utc=now(),models=models,
        sources=dict(argon=str(ROOT/'experiments/gemini4_argon_plain_both_fewshot_n250_cr25_seed42_c30_20261014_v1/PLAN.json'),
                     qwen=str(ROOT/'experiments/qwen38_27b_ypxia_evidence_resume_c20_seed42_20261014_v1/PLAN.json'),
                     sol=str(ROOT/'experiments/sol61_hyj_pro_remaining1080_max_c30_seed42_20261014_v1/PLAN.json'),
                     sol_live_fewshot=str(ROOT/'experiments/sol61_hyj_pro_settled_failures_retry4_max_c36_seed42_20261015_v1/PLAN.json')),
        current_campaign=str(run),prompt_registry=str(HERE/'prompts/FINAL_PROMPTS.json'),
        runtime_interpreter=sys.executable,runner=str(HERE/'multillm_campaign.py'),supervisor=str(HERE/'multillm_supervise.py'),
        progress=str(HERE/'LLM_SCORING_PROGRESS.md'),credentials_in_index=False,system_task_indexes=system_tasks,
        access_status='Configured paths validated; fresh preflight results live in campaign/preflight/<model>/ACCOUNTING.json')
    save(HERE/'llm_models.json',index)
    save(run/'OFFLINE_VALIDATION.json',dict(status='PASS',unique_request_ids=len({r['request_id'] for r in requests}),
        tasks=len(tasks),models=len(models),assays=sum(manifest['domains'].values()),arms=4,source_manifest_verified=True,all_payload_messages_equal=True,
        labels_read=False,model_calls=0,credential_paths_present=True,global_concurrency=20))
    print(json.dumps(dict(status='prepared',run=str(run),models=list(models),scoring_requests=len(tasks),global_concurrency=20)))


def validate(run):
    if sha(run/'PLAN.json') != (run/'PLAN.sha256').read_text().strip(): raise ValueError('Campaign plan changed')
    p=load(run/'PLAN.json')
    for file,h in p['source_hashes'].items():
        if sha(file)!=h: raise ValueError('Frozen implementation changed: '+file)
    if sha(run/'REQUESTS.json')!=p['requests_sha256'] or sha(run/'TASKS.json')!=p['tasks_sha256']:
        raise ValueError('Frozen tasks changed')
    for ref in p.get('system_task_indexes',{}).values():
        if sha(ref['path'])!=ref['sha256']:raise ValueError('System task index changed')
    inputs=Path(p['inputs'])
    if sha(inputs/'MANIFEST.json')!=p['input_manifest_sha256']: raise ValueError('Input manifest changed')
    requests=load_requests(inputs)
    manifest,accepted=validate_bundle(inputs)
    if p.get('protocol_version')!=manifest['protocol_version'] or p.get('acceptance')!=manifest['acceptance']:raise ValueError('Campaign lacks current protocol acceptance')
    if requests!=load(run/'REQUESTS.json'): raise ValueError('Campaign requests differ from bundle')
    for k,h in p['model_plan_hashes'].items():
        child=run/'models'/k
        if sha(child/'PLAN.json')!=h: raise ValueError('Model plan changed')
        cp=load(child/'PLAN.json')
        if sha(child/'QUEUE.json')!=cp['queue_sha256']: raise ValueError('Model queue changed')
        validate_subset(load(child/'QUEUE.json'),accepted)
        if load(child/'QUEUE.json')!=accepted:raise ValueError('Model queue scope mismatch')
        if cp.get('protocol_version')!=manifest['protocol_version'] or cp.get('acceptance')!=manifest['acceptance']:raise ValueError('Model lacks current acceptance')
        refs=Path(cp['evaluator_references'])
        if sha(refs)!=cp['evaluator_references_sha256']:raise ValueError('Evaluator references changed')
        validate_evaluator(refs,inputs,accepted)
    tasks=load(run/'TASKS.json')
    expected=[dict(model_key=k,request_id=r['request_id'],domain=r['domain'],assay=r['assay'],arm=r['arm']) for r in requests for k in p['models']]
    if tasks!=expected:raise ValueError('Campaign tasks differ from accepted scope')
    return p,{r['request_id']:r for r in requests},tasks


def preflight(run, key):
    """One neutral probe, reusable by the campaign; never resubmit a saved attempt."""
    os.umask(0o077)
    plan=load(run/'PLAN.json')
    if sha(run/'PLAN.json')!=(run/'PLAN.sha256').read_text().strip():raise ValueError('Plan changed')
    for file,h in plan['source_hashes'].items():
        if sha(file)!=h:raise ValueError('Frozen implementation changed: '+file)
    messages=[dict(role='system',content='Return only the requested JSON.'),dict(role='user',content='Return exactly {"variants":[{"variant_id":"M001","predicted_score":0}]}')]
    request=dict(request_id='neutral_preflight',messages=messages,messages_sha256=message_sha256(messages),expected_ids=['M001'])
    accounting,value=submit(run/'preflight'/key,plan['models'][key],request)
    print(json.dumps({k:accounting.get(k) for k in ['status','failure_category','http_status','returned_models','thinking_level','elapsed_s']}))
    if accounting['status']!='ok' or value!={'M001':0.0}:raise SystemExit(1)


def run_one(run, task, model, request):
    root=run/'models'/task['model_key'];folder=root/'attempts'/task['request_id']
    accounting,value=submit(folder,model,request)
    unit=dict(logical_id=task['request_id'],domain=task['domain'],assay=task['assay'],arm=task['arm'],chunk_index=request.get('chunk_index',0),
        expected_ids=request['expected_ids'],status=accounting['status'],value=value,
        failure_category=accounting['failure_category'],messages_sha256=request['messages_sha256'],
        request_sha256=accounting['request_sha256'],accounting_path=str(folder/'ACCOUNTING.json'))
    path=root/'units'/(task['request_id']+'.json');save(path,unit);path.with_suffix('.sha256').write_text(sha(path)+'\n')
    return accounting


def run_campaign(run):
    os.umask(0o077)
    plan,requests,tasks=validate(run)
    lease=(run/'RUN.lock').open('a');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if plan.get('resume_from'):
        from revalidate_campaign import import_settled
        import_settled(run,plan)
    save(run/'LAUNCH.json',dict(pid=os.getpid(),started_utc=now()))
    previous=load(run/'STATUS.json') if (run/'STATUS.json').exists() else {}
    stopped=dict(previous.get('stop_reasons',{}));recent={k:collections.deque(maxlen=20) for k in plan['models']};completed={};pending=[]
    for task in tasks:
        key=task['model_key'];uid=task['request_id'];child=run/'models'/key;p=child/'units'/(uid+'.json')
        if p.exists():
            if sha(p)!=p.with_suffix('.sha256').read_text().strip():raise ValueError('Saved prediction changed')
            completed[key,uid]=load(p)
        elif (child/'attempts'/uid).exists():
            stopped[key]='prior_attempt_not_finalized_no_automatic_replay'
        else:pending.append(task)
    smoke_messages=[dict(role='system',content='Return only the requested JSON.'),dict(role='user',content='Return exactly {"variants":[{"variant_id":"M001","predicted_score":0}]}')]
    smoke=dict(request_id='neutral_preflight',messages=smoke_messages,messages_sha256=message_sha256(smoke_messages),expected_ids=['M001'])
    save(run/'STATUS.json',dict(state='preflight',updated_utc=now(),planned=len(tasks),completed=len(completed),valid=sum(r['status']=='ok' for r in completed.values()),active=0,stop_reasons=stopped))
    with cf.ThreadPoolExecutor(max_workers=min(plan['workers'],len(plan['models']))) as pool:
        probes={}
        for key,model in plan['models'].items():
            folder=run/'preflight'/key
            if (folder/'RESULT.json').exists():
                if load(folder/'RESULT.json')['status']!='ok':stopped[key]='preflight_failed'
            elif folder.exists():stopped[key]='unresolved_preflight_no_replay'
            else:probes[pool.submit(submit,folder,model,smoke)]=key
        for f in cf.as_completed(probes):
            key=probes[f]
            try:
                a,v=f.result()
                if a['status']!='ok' or v!={'M001':0.0}:stopped[key]='preflight_'+a['failure_category']
            except Exception as e:
                stopped[key]='preflight_runner_'+type(e).__name__
                save(run/'preflight'/key/'ERROR.json',dict(type=type(e).__name__,time=now()))
    active={};peak=0
    def snapshot(state):
        model_rows={}
        for key in plan['models']:
            units=[r for (k,u),r in completed.items() if k==key];running=[t for t in active.values() if t['model_key']==key]
            categories=collections.Counter(r['failure_category'] for r in units if r['status']!='ok')
            planned=sum(t['model_key']==key for t in tasks)
            model_rows[key]=dict(planned=planned,completed=len(units),valid=sum(r['status']=='ok' for r in units),
                failed=sum(r['status']!='ok' for r in units),active=len(running),not_submitted=planned-len(units)-len(running),
                stop_reason=stopped.get(key),failure_categories=dict(categories))
        save(run/'STATUS.json',dict(state=state,updated_utc=now(),pid=os.getpid(),planned=len(tasks),completed=len(completed),
            valid=sum(r['status']=='ok' for r in completed.values()),active=len(active),workers=20,peak_concurrency=peak,
            models=model_rows,stop_reasons=stopped,active_tasks=list(active.values())))
    with cf.ThreadPoolExecutor(max_workers=plan['workers']) as pool:
        while pending or active:
            if (run/'PAUSE').exists():
                for key in plan['models']:stopped.setdefault(key,'pause_file')
            while pending and len(active)<plan['workers']:
                task=pending.pop(0);key=task['model_key']
                if key in stopped:continue
                f=pool.submit(run_one,run,task,plan['models'][key],requests[task['request_id']]);active[f]=task
                peak=max(peak,len(active))
            snapshot('running' if not stopped else 'running_with_stopped_models')
            if not active:break
            done,_=cf.wait(active,timeout=5,return_when=cf.FIRST_COMPLETED)
            for f in done:
                task=active.pop(f);key=task['model_key'];uid=task['request_id']
                try:
                    a=f.result();result=load(run/'models'/key/'units'/(uid+'.json'))
                    completed[key,uid]=result;category=a['failure_category'];recent[key].append(category)
                    if category in FATAL:stopped[key]=category
                    if len(recent[key])==20 and sum(c in TECHNICAL for c in recent[key])>=16:stopped[key]='technical_failure_circuit_breaker'
                except Exception as e:
                    stopped[key]='runner_exception'
                    save(run/'errors'/(key+'_'+uid+'.json'),dict(type=type(e).__name__,time=now(),request_id=uid))
    state='complete' if len(completed)==len(tasks) and all(r['status']=='ok' for r in completed.values()) else 'finished_with_failures' if len(completed)==len(tasks) else 'stopped_incomplete'
    snapshot(state);lease.close()


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','validate','run','preflight']);p.add_argument('--run',type=Path,default=HERE/'runs'/NAME);p.add_argument('--inputs',type=Path);p.add_argument('--model-profile',choices=['legacy_four','ppi_rna_five','ppi_rna_six'],default='legacy_four');p.add_argument('--resume-from',type=Path);p.add_argument('--model-key');a=p.parse_args();run=a.run.resolve()
    if a.action=='prepare':prepare(run,a.inputs,a.model_profile,a.resume_from)
    elif a.action=='validate':plan,requests,tasks=validate(run);print(json.dumps(dict(status='PASS',requests=len(requests),tasks=len(tasks),model_calls=0)))
    elif a.action=='preflight':preflight(run,a.model_key)
    else:run_campaign(run)


if __name__=='__main__':main()
