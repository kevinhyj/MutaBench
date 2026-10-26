"""Durable single-attempt Google/Qwen streaming transport for frozen messages."""
import json
import os
from pathlib import Path
import re
import shlex
import stat
import subprocess
import sys
import time
from datetime import datetime, timezone

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / 'code'))
from scoring_validation import ScoringResponseValidator, VALIDATION_VERSION
from frozen_prompt_bundle import check_actual_messages, message_sha256, sha_bytes

REFUSAL = re.compile(r"(?:I (?:cannot|can't|won't|am unable to) (?:assist|help|provide|score|predict|comply)|(?:content|safety) policy|我(?:不能|无法)(?:协助|帮助|提供|预测|评分))", re.I)


def now(): return datetime.now(timezone.utc).isoformat()
def load(p): return json.loads(Path(p).read_text())
def sha(p): return sha_bytes(Path(p).read_bytes())
def save(p, obj):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    temporary = p.with_name(p.name + '.tmp')
    temporary.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    temporary.replace(p)


def payload(model, request):
    messages = request['messages']
    if model['provider'] == 'google':
        body = dict(systemInstruction={'parts':[{'text':messages[0]['content']}]},
                    contents=[{'role':'user','parts':[{'text':messages[1]['content']}]}],
                    generationConfig=dict(temperature=model['temperature'], seed=model['seed'],
                        maxOutputTokens=model['max_output_tokens'],
                        thinkingConfig=dict(thinkingLevel=model['thinking_level'].upper(), includeThoughts=True)))
        actual = [dict(role='system', content=body['systemInstruction']['parts'][0]['text']),
                  dict(role='user', content=body['contents'][0]['parts'][0]['text'])]
    elif model['provider'] == 'openai_compatible':
        if model.get('request_style')=='sol_chat':
            body=dict(model=model['model'],messages=messages,stream=True,stream_options={'include_usage':True},
                seed=model['seed'],reasoning_effort=model['thinking_level'],max_completion_tokens=model['max_output_tokens'])
        else:body = dict(model=model['model'], messages=messages, stream=True,
                    stream_options={'include_usage':True}, seed=model['seed'], temperature=model['temperature'],
                    reasoning_effort=model['thinking_level'], enable_thinking=True,
                    max_tokens=model['max_output_tokens'])
        actual = body['messages']
    else:
        raise ValueError('Unsupported configured provider')
    check_actual_messages(request, actual)
    return body


def credential(model):
    path = Path(model['credential_path'])
    if stat.S_IMODE(path.stat().st_mode) & 0o077:
        raise ValueError('Credential file permissions are not private')
    if model['provider'] == 'google':
        key = None
        for line in path.read_text().splitlines():
            line = line.strip().removeprefix('export ')
            if line.startswith('GEMINI_API_KEY='):
                values = shlex.split(line.split('=',1)[1])
                if len(values) != 1: raise ValueError('Credential parse error')
                key = values[0]
    else:
        cfg = load(path)
        if cfg['profile'] != model['credential_profile'] or cfg['base_url'].rstrip('/')+'/chat/completions' != model['endpoint']:
            raise ValueError('Configured provider identity changed')
        key = cfg.get('api_key')
    if not key or any(c in key for c in '\r\n'):
        raise ValueError('Missing or invalid credential')
    return key


def stream_events(raw):
    text = raw.decode('utf-8', errors='replace').lstrip('\ufeff')
    if text.lstrip().startswith(('{','[')):
        try:
            obj = json.loads(text)
            return (obj if isinstance(obj,list) else [obj]), False, 0
        except ValueError:
            return [], False, 1
    events = []; done = False; malformed = 0; buffer = []
    for line in (text.replace('\r\n','\n')+'\n\n').splitlines():
        if line.startswith('data:'):
            buffer.append(line[5:].lstrip(' '))
        elif not line and buffer:
            item = '\n'.join(buffer); buffer = []
            if item.strip() == '[DONE]': done = True; continue
            try:
                obj = json.loads(item)
                if isinstance(obj,dict): events.append(obj)
                else: malformed += 1
            except ValueError: malformed += 1
    return events, done, malformed


def parse(raw, model, expected_ids, http_status, returncode):
    events, done, malformed = stream_events(raw)
    errors = [e['error'] for e in events if e.get('error')]
    models = []; finishes = []; text = []; reasoning = []; usage = {}; blocked = False
    if model['provider'] == 'google':
        for e in events:
            if e.get('modelVersion'): models.append(e['modelVersion'])
            if e.get('usageMetadata'): usage = e['usageMetadata']
            blocked |= bool(e.get('promptFeedback',{}).get('blockReason'))
            for c in e.get('candidates',[]):
                if c.get('index',0) != 0: malformed += 1; continue
                if c.get('finishReason'): finishes.append(c['finishReason'])
                for part in c.get('content',{}).get('parts',[]):
                    if isinstance(part.get('text'),str):
                        (reasoning if part.get('thought') else text).append(part['text'])
        i, v, r, total = (usage.get(k) for k in ['promptTokenCount','candidatesTokenCount','thoughtsTokenCount','totalTokenCount'])
        if r is None and all(type(x) is int for x in (i,v,total)) and i+v == total: r = 0
        output = v+r if type(v) is int and type(r) is int else None
        tokens = dict(input_tokens=i,output_tokens=output,reasoning_tokens=r,total_tokens=total)
        blocked |= any(f in {'SAFETY','PROHIBITED_CONTENT','RECITATION','BLOCKLIST','SPII'} for f in finishes)
        complete = bool(finishes and finishes[-1]=='STOP')
    else:
        for e in events:
            if e.get('model'): models.append(e['model'])
            if e.get('usage'): usage = e['usage']
            for c in e.get('choices',[]):
                if c.get('index',0) != 0: malformed += 1; continue
                d = c.get('delta') or c.get('message') or {}
                blocked |= bool(d.get('refusal'))
                if isinstance(d.get('content'),str): text.append(d['content'])
                if isinstance(d.get('reasoning_content'),str): reasoning.append(d['reasoning_content'])
                if c.get('finish_reason'): finishes.append(c['finish_reason'])
        tokens = dict(input_tokens=usage.get('prompt_tokens'),output_tokens=usage.get('completion_tokens'),
                      total_tokens=usage.get('total_tokens'),reasoning_tokens=(usage.get('completion_tokens_details') or {}).get('reasoning_tokens'))
        blocked |= any(f in {'content_filter','refusal'} for f in finishes)
        complete = bool(done and finishes and finishes[-1]=='stop')
    visible = ''.join(text)
    blocked |= bool(REFUSAL.search(visible))
    error_text = json.dumps(errors,ensure_ascii=False).lower()
    if returncode: category = 'timeout_ambiguous' if returncode==28 else 'network_ambiguous'
    elif http_status in (401,403): category = 'authentication_or_permission'
    elif http_status==402 or any(s in error_text for s in ('insufficient_quota','insufficient balance','余额不足','额度不足')): category = 'quota'
    elif http_status==429: category = 'rate_limit'
    elif http_status in (400,404,405,422): category = 'invalid_request'
    elif not 200<=http_status<300 or errors: category = 'provider_error'
    elif blocked: category = 'explicit_block'
    elif malformed: category = 'malformed_stream'
    elif set(models) != {model['model']}: category = 'model_identity_mismatch'
    elif not complete: category = 'incomplete_output'
    elif not visible: category = 'empty_final'
    else: category = 'ok'
    report = ScoringResponseValidator(expected_ids).validate(visible) if category=='ok' else None
    if report and not report.valid: category = 'format'
    for k,v in tokens.items():
        if type(v) is not int or v<0: tokens[k] = None
    vals = [tokens[k] for k in ('input_tokens','output_tokens','total_tokens')]
    return dict(status='ok' if category=='ok' else 'failed',failure_category=category,
        value=report.value if category=='ok' else None,visible_output=visible,
        returned_models=sorted(set(models)),finish_reasons=finishes,stream_done_marker=done,
        malformed_events=malformed,api_errors=errors,raw_usage=usage,**tokens,
        usage_consistent=vals[0]+vals[1]==vals[2] if all(v is not None for v in vals) else None,
        validation_version=VALIDATION_VERSION,format_validation=report.as_dict() if report else None)


def submit(folder, model, request):
    """An existing attempt directory is never automatically resubmitted."""
    folder = Path(folder);folder.mkdir(parents=True,exist_ok=False)
    body = payload(model,request); key = credential(model)
    if key in json.dumps(body): raise ValueError('Credential in payload')
    save(folder/'REQUEST.json',body)
    start = dict(started_utc=now(),model=model['model'],thinking_level=model['thinking_level'],
        credential_profile=model['credential_profile'],credential_path=model['credential_path'],endpoint=model['endpoint'],
        timeout_s=model['timeout_s'],request_id=request['request_id'],messages_sha256=request['messages_sha256'],
        request_sha256=sha(folder/'REQUEST.json'),seed=model['seed'],automatic_retries=0,
        model_identity='Provider-reported model ID; backend weights and actual thinking effort are not independently verified')
    save(folder/'START.json',start)
    header = 'x-goog-api-key: '+key if model['provider']=='google' else 'Authorization: Bearer '+key
    auth = 'header = '+json.dumps(header)+'\nheader = "Content-Type: application/json"\n'
    args = ['curl','--silent','--show-error','--no-buffer','--http1.1','--proto','=https',
            '--connect-timeout','30','--max-time',str(max(3600,model['timeout_s'])),'--config','-',
            '--data-binary','@'+str(folder/'REQUEST.json'),'--output',str(folder/'response.raw'),
            '--dump-header',str(folder/'response.headers'),'--write-out','%{http_code}']
    if model.get('proxy'): args += ['--proxy',model['proxy'],'--noproxy','']
    else: args += ['--noproxy','*']
    args.append(model['endpoint'])
    begun = time.monotonic()
    try:
        result = subprocess.run(args,input=auth,text=True,capture_output=True,timeout=model['timeout_s']+60)
        code = result.returncode; status = int(result.stdout.strip() or 0)
        error = result.stderr.replace(key,'[REDACTED]')[-2000:]
    except subprocess.TimeoutExpired:
        code,status,error = 28,0,'Outer transport deadline exceeded; request may have reached provider'
    raw = (folder/'response.raw').read_bytes() if (folder/'response.raw').exists() else b''
    parsed = parse(raw,model,request['expected_ids'],status,code)
    value = parsed.pop('value');visible = parsed.pop('visible_output')
    (folder/'FINAL.txt').write_text(visible)
    accounting = dict(start,**parsed,finished_utc=now(),elapsed_s=time.monotonic()-begun,
                      http_status=status,curl_returncode=code,transport_error=error,response_sha256=sha_bytes(raw))
    if model['provider']=='google': accounting['monetary_accounting'] = False
    else: accounting['price_status'] = 'Workspace pricing not verified; retain raw usage, do not reuse another provider price'
    save(folder/'ACCOUNTING.json',accounting)
    save(folder/'RESULT.json',dict(status=accounting['status'],failure_category=accounting['failure_category'],value=value))
    return accounting,value
