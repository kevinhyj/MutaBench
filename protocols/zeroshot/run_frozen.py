"""Versioned execution of frozen label-free CR requests using the existing durable gateway."""
import argparse
import concurrent.futures
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import sys
import time

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'code'))
from evagent.integrated.research.concurrent_gateway import ConcurrentGateway
from evagent.integrated.gateway import public_model_config,digest
from scoring_validation import ScoringResponseValidator
from evagent.integrated.self_check import SelfCheckPolicy,run_self_check
from evagent.runtime import ExecutionPaused
from shared_protocol import validate_bundle, validate_subset


def load(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    temp=p.with_suffix(p.suffix+'.tmp');temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');temp.replace(p)
def freeze(p,value):
    p=Path(p)
    if p.exists():raise FileExistsError(str(p))
    save(p,value)


def source_hashes():
    paths=[HERE/'run_frozen.py',HERE/'evaluate_frozen.py',HERE/'scoring_validation.py',HERE/'sampling.py',HERE/'sampling_contract.json',
           ROOT/'code/evagent/integrated/gateway.py',ROOT/'code/evagent/integrated/research/concurrent_gateway.py',
           ROOT/'code/evagent/integrated/chunked_scoring.py',ROOT/'code/evagent/integrated/self_check.py',
           ROOT/'code/evagent/api_timeouts.py', HERE/'shared_protocol.py', HERE/'prepare_shared_prompts.py']
    return {str(p):sha(p) for p in paths}


def prepare(args):
    inputs=Path(args.inputs).resolve();run=Path(args.run).resolve()
    if not run.is_relative_to((HERE/'runs').resolve()) or run.exists():raise ValueError('New pipeline runs directory required')
    manifest,queue=validate_bundle(inputs)
    if args.domains and not set(args.domains.split(','))<=set(q['domain'] for q in queue):raise ValueError('Unknown requested domain')
    if args.arms and not set(args.arms.split(','))<=set(q['arm'] for q in queue):raise ValueError('Unknown requested arm')
    if args.max_units is not None and args.max_units<1:raise ValueError('Positive max-units required')
    queue=[q for q in queue if (not args.domains or q['domain'] in args.domains.split(',')) and (not args.arms or q['arm'] in args.arms.split(','))]
    if args.max_units:queue=queue[:args.max_units]
    if not queue:raise ValueError('Empty requested cohort')
    config=public_model_config(load(args.model_config))
    if config.get('generation_seed')!=42:raise ValueError('New runs require generation_seed42')
    config['timeout_s']=max(3600,config['timeout_s'])
    if not args.unlimited and (args.token_budget is None or args.token_budget<1):raise ValueError('Explicit positive budget required')
    if args.unlimited and args.token_budget is not None:raise ValueError('Choose a finite budget or explicit unlimited')
    if not 1<=args.workers<=64:raise ValueError('Workers must be 1..64')
    evaluator=HERE/'evaluator'/manifest['version']/'LABEL_REFERENCES.json'
    if not evaluator.is_file():raise ValueError('Independent evaluator references not frozen')
    validate_evaluator(evaluator,inputs,queue)
    run.mkdir(parents=True)
    freeze(run/'QUEUE.json',queue)
    plan=dict(schema='pipeline-frozen-cr-run-v1',inputs=str(inputs),input_manifest_sha256=sha(inputs/'MANIFEST.json'),
              evaluator_references=str(evaluator),evaluator_references_sha256=sha(evaluator),
              queue_sha256=sha(run/'QUEUE.json'),model_config=config,token_budget=args.token_budget,budget_mode='unlimited' if args.unlimited else 'limited',workers=args.workers,
              max_repairs=args.max_repairs,source_hashes=source_hashes(),initial_units=len(queue),
              maximum_generation_requests=len(queue)*(1+args.max_repairs),candidate_resampling=False,
              test_labels_in_model_inputs=False,credentials_path=str(Path(args.credentials).resolve()),
              protocol_version=manifest['protocol_version'],acceptance=manifest['acceptance'],
              excluded_by_domain_or_arm=True if args.domains or args.arms else False)
    plan['fingerprint']=digest(plan);freeze(run/'PLAN.json',plan)
    save(run/'STATUS.json',dict(status='prepared_not_started',planned=len(queue),submitted=0,completed=0,model_calls=0))
    print(json.dumps({'status':'prepared_not_started','run':str(run),'units':len(queue),'maximum_generation_requests':plan['maximum_generation_requests'],'fingerprint':plan['fingerprint']}))


def validate(run):
    plan=load(run/'PLAN.json')
    if digest({k:v for k,v in plan.items() if k!='fingerprint'})!=plan['fingerprint']:raise ValueError('Plan fingerprint changed')
    for name,h in plan['source_hashes'].items():
        if sha(name)!=h:raise ValueError('Frozen implementation changed: '+name)
    inputs=Path(plan['inputs'])
    if sha(inputs/'MANIFEST.json')!=plan['input_manifest_sha256']:raise ValueError('Input manifest changed')
    m,accepted=validate_bundle(inputs)
    if sha(run/'QUEUE.json')!=plan['queue_sha256']:raise ValueError('Queue changed')
    queue=load(run/'QUEUE.json')
    if plan.get('protocol_version')!=m['protocol_version'] or plan.get('acceptance')!=m['acceptance']:raise ValueError('Run lacks current protocol acceptance')
    validate_subset(queue,accepted)
    evaluator=Path(plan['evaluator_references'])
    if sha(evaluator)!=plan['evaluator_references_sha256']:raise ValueError('Evaluator references changed')
    validate_evaluator(evaluator,inputs,queue)
    return plan,queue


def validate_evaluator(path,inputs,queue):
    """Validate reference metadata without opening label files."""
    rows=load(path);indexed={(r['domain'],r['assay']):r for r in rows}
    if len(indexed)!=len(rows):raise ValueError('Duplicate evaluator reference')
    for domain,assay in {(q['domain'],q['assay']) for q in queue}:
        ref=indexed.get((domain,assay));qp=inputs/'queries'/domain/assay/'query.json'
        if ref is None or Path(ref['query_path']).resolve()!=qp.resolve() or ref['query_sha256']!=sha(qp):
            raise ValueError('Evaluator query differs from accepted input')
        if not ref.get('label_path') or not isinstance(ref.get('label_sha256'),str) or len(ref['label_sha256'])!=64:
            raise ValueError('Missing evaluator label provenance')


def credential(path):
    for line in Path(path).read_text().splitlines():
        line=line.strip().removeprefix('export ')
        if line.startswith('GEMINI_API_KEY='):
            values=shlex.split(line.split('=',1)[1])
            if len(values)!=1:raise ValueError('Credential parse error')
            os.environ['GEMINI_API_KEY']=values[0];return
    raise ValueError('Gemini credential unavailable')


def one(run,plan,unit,gateway):
    p=run/'units'/(unit['logical_id']+'.json')
    if p.exists():
        result=load(p)
        if result['logical_id']!=unit['logical_id'] or sha(p)!=p.with_suffix('.sha256').read_text().strip():raise ValueError('Saved result changed')
        return result
    template=unit['template'];validator=ScoringResponseValidator(template['expected_ids'])
    def generate(user,attempt):
        return gateway.generate(system=template['system'],user=user,run_name='zeroshot-'+unit['logical_id'],
            metadata=dict(domain=unit['domain'],assay=unit['assay'],arm=unit['arm'],chunk_index=unit['chunk_index'],
                          logical_id=unit['logical_id'],expected_ids=template['expected_ids'],attempt_index=attempt))
    checked=run_self_check(original_prompt=template['user'],generate=generate,validator=validator,policy=SelfCheckPolicy(max_repairs=plan['max_repairs']))
    result=dict(logical_id=unit['logical_id'],domain=unit['domain'],assay=unit['assay'],arm=unit['arm'],chunk_index=unit['chunk_index'],
                expected_ids=template['expected_ids'],status=checked['status'],value=checked['value'],
                repair_attempts_used=checked['repair_attempts_used'],usage_summary=checked['usage_summary'],
                provider_request_hashes=[x.get('request_sha256') for x in checked['attempts']])
    freeze(p,result);p.with_suffix('.sha256').write_text(sha(p)+'\n');return result


def worker(run,transport=None):
    plan,queue=validate(run)
    with (run/'.run.lock').open('a+') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if transport is None:credential(plan['credentials_path'])
        gateway=ConcurrentGateway(model_config=plan['model_config'],run_dir=run/'provider',fingerprint=plan['fingerprint'],
            token_budget=plan['token_budget'],budget_mode=plan.get('budget_mode','limited'),reserve_budget=True,max_concurrent_requests=plan['workers'],transport=transport)
        pending=[q for q in queue if not (run/'units'/(q['logical_id']+'.json')).exists()]
        stop=False
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=plan['workers']) as pool:
                for start in range(0,len(pending),plan['workers']):
                    if (run/'PAUSE').exists():stop=True;break
                    futures=[pool.submit(one,run,plan,q,gateway) for q in pending[start:start+plan['workers']]]
                    while futures:
                        done,not_done=concurrent.futures.wait(futures,timeout=5,return_when=concurrent.futures.FIRST_COMPLETED)
                        for future in done:
                            try:future.result()
                            except ExecutionPaused:stop=True
                            except Exception as error:
                                save(run/'ERROR.json',dict(type=type(error).__name__,time=time.time()));stop=True
                        futures=list(not_done)
                        existing=list((run/'units').glob('*.json'));ok=sum(load(p).get('status')=='ok' for p in existing)
                        save(run/'STATUS.json',dict(status='draining' if stop else 'running',planned=len(queue),completed=len(existing),valid=ok,
                                                  inflight=len(futures),ledger=gateway.summary(),updated_unix=time.time()))
                    if stop:break
            state=gateway.summary();files=list((run/'units').glob('*.json'));ok=sum(load(p).get('status')=='ok' for p in files)
            save(run/'STATUS.json',dict(status='paused_requires_attention' if stop else 'finished',planned=len(queue),completed=len(files),valid=ok,
                                      inflight=0,ledger=state,updated_unix=time.time()))
        finally:gateway.close()


def main():
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='action',required=True)
    a=sub.add_parser('prepare');a.add_argument('--inputs',required=True);a.add_argument('--run',required=True);a.add_argument('--model-config',required=True)
    a.add_argument('--credentials',required=True);a.add_argument('--token-budget',type=int);a.add_argument('--unlimited',action='store_true');a.add_argument('--workers',type=int,default=8)
    a.add_argument('--max-repairs',type=int,choices=[0,1],default=1);a.add_argument('--domains');a.add_argument('--arms');a.add_argument('--max-units',type=int)
    for name in ['validate','run']:
        a=sub.add_parser(name);a.add_argument('--run',required=True)
    args=p.parse_args()
    if args.action=='prepare':prepare(args)
    elif args.action=='validate':plan,q=validate(Path(args.run));print(json.dumps({'status':'valid','units':len(q),'model_calls':0}))
    else:worker(Path(args.run))


if __name__=='__main__':main()
