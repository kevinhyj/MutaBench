"""Reuse settled raw responses after a local validator repair, without API calls."""
import collections
import hashlib
from pathlib import Path
import shutil
import time

from multillm_transport import load, save, sha, parse
from scoring_validation import VALIDATION_VERSION


def import_settled(run, plan):
    source=Path(plan['resume_from'])
    if (run/'REVALIDATION.json').exists():
        return
    # Do not kill long generations or import a concurrently changing campaign.
    while not (source/'SUPERVISOR_RESULT.json').exists():
        save(run/'STATUS.json',dict(state='waiting_for_previous_inflight_calls',
            planned=plan['planned_scoring_requests'],completed=0,valid=0,active=0,
            previous_run=str(source),previous_status=load(source/'STATUS.json')))
        time.sleep(30)
    if sha(source/'PLAN.json') != plan['resume_plan_sha256']:
        raise ValueError('Previous frozen plan changed')
    old=load(source/'PLAN.json')
    if not set(old['models'])<=set(plan['models']) or any(plan['models'][k]!=v for k,v in old['models'].items()) or old['input_manifest_sha256']!=plan['input_manifest_sha256']:
        raise ValueError('Resume models/input differ')
    retained_tasks=[t for t in load(run/'TASKS.json') if t['model_key'] in old['models']]
    if load(source/'REQUESTS.json')!=load(run/'REQUESTS.json') or load(source/'TASKS.json')!=retained_tasks:
        raise ValueError('Resume messages or task order differ')
    for path,h in old['source_hashes'].items():
        root=Path(__file__).resolve().parent.parents[1]
        snapshot=source/'source_snapshot'/Path(path).relative_to(root)
        if sha(snapshot)!=h:raise ValueError('Previous implementation snapshot changed')
    state=load(source/'STATUS.json')
    if state.get('active') or state.get('state') not in {'complete','finished_with_failures','stopped_incomplete'}:
        raise ValueError('Previous requests have not drained')
    requests={r['request_id']:r for r in load(run/'REQUESTS.json')}
    outcomes=collections.Counter();records=[]
    for key,model in old['models'].items():
        probe=source/'preflight'/key
        if load(probe/'RESULT.json')['status']!='ok':raise ValueError('Cannot reuse failed preflight')
        shutil.copytree(probe,run/'preflight'/key)
        attempts=source/'models'/key/'attempts'
        for folder in sorted(attempts.iterdir()) if attempts.exists() else []:
            uid=folder.name;request=requests[uid]
            original=load(folder/'ACCOUNTING.json')
            unit_path=source/'models'/key/'units'/(uid+'.json')
            if sha(unit_path)!=unit_path.with_suffix('.sha256').read_text().strip():raise ValueError('Previous result changed')
            raw=(folder/'response.raw').read_bytes() if (folder/'response.raw').exists() else b''
            if hashlib.sha256(raw).hexdigest()!=original['response_sha256'] or sha(folder/'REQUEST.json')!=original['request_sha256']:
                raise ValueError('Previous raw response or request changed')
            parsed=parse(raw,model,request['expected_ids'],original['http_status'],original['curl_returncode'])
            value=parsed.pop('value');visible=parsed.pop('visible_output')
            if visible!=(folder/'FINAL.txt').read_text():raise ValueError('Revalidation changed model output')
            prior=load(unit_path)
            if prior['status']=='ok' and (parsed['status']!='ok' or value!=prior['value']):
                raise ValueError('Revalidation changed a previously accepted prediction')
            target=run/'models'/key/'attempts'/uid
            shutil.copytree(folder,target)
            accounting=dict(original,**parsed,revalidation=dict(source=str(folder),
                source_accounting_sha256=sha(folder/'ACCOUNTING.json'),model_calls=0,
                validator_version=VALIDATION_VERSION,
                reason='Local cosmetic-key/metadata tolerance; exact IDs and numeric scores unchanged; no raw output or prompt edits'))
            save(target/'ACCOUNTING.json',accounting)
            save(target/'RESULT.json',dict(status=parsed['status'],failure_category=parsed['failure_category'],value=value))
            unit=dict(logical_id=uid,domain=request['domain'],assay=request['assay'],arm=request['arm'],
                chunk_index=request['chunk_index'],expected_ids=request['expected_ids'],status=parsed['status'],
                value=value,failure_category=parsed['failure_category'],messages_sha256=request['messages_sha256'],
                request_sha256=original['request_sha256'],accounting_path=str(target/'ACCOUNTING.json'),
                imported_from=str(unit_path),source_unit_sha256=sha(unit_path))
            dest=run/'models'/key/'units'/(uid+'.json');save(dest,unit);dest.with_suffix('.sha256').write_text(sha(dest)+'\n')
            outcomes[parsed['failure_category']]+=1
            records.append(dict(model_key=key,request_id=uid,previous_status=original['failure_category'],
                                status=parsed['failure_category'],source=str(folder)))
    save(run/'REVALIDATION.json',dict(source=str(source),source_plan_sha256=plan['resume_plan_sha256'],
        imported_requests=len(records),outcomes=dict(outcomes),new_model_calls=0,records=records))
