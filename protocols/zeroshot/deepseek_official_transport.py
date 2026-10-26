"""Official DeepSeek adapter; exact frozen messages, one durable attempt."""
import json
from pathlib import Path
import subprocess
import time

from frozen_prompt_bundle import check_actual_messages, sha_bytes
from multillm_transport import credential, load, now, parse, save, sha


def payload(model, request):
    if model['model'] != 'deepseek-flash' or model['endpoint'] != 'https://api.deepseek.com/chat/completions':
        raise ValueError('This adapter is pinned to the official DeepSeek Flash route')
    if model['thinking_level'] != 'max' or model['timeout_s'] < 3600:
        raise ValueError('Expected max effort and long generation deadline')
    body = dict(model=model['model'], messages=request['messages'],
                thinking={'type': 'enabled'}, reasoning_effort='max', seed=42,
                stream=True, stream_options={'include_usage': True},
                max_tokens=model['max_output_tokens'])
    check_actual_messages(request, body['messages'])
    return body


def submit(folder, model, request):
    folder = Path(folder)
    body = payload(model, request)
    key = credential(model)
    folder.mkdir(parents=True, exist_ok=False)
    if key in json.dumps(body):
        raise ValueError('Credential in payload')
    save(folder / 'REQUEST.json', body)
    start = dict(started_utc=now(), model=model['model'], thinking_level='max',
                 thinking=body['thinking'], credential_profile=model['credential_profile'],
                 credential_path=model['credential_path'], endpoint=model['endpoint'],
                 timeout_s=model['timeout_s'], request_id=request['request_id'],
                 messages_sha256=request['messages_sha256'], request_sha256=sha(folder / 'REQUEST.json'),
                 seed=42, seed_enforcement='Requested; provider determinism not independently verified',
                 automatic_retries=0)
    save(folder / 'START.json', start)
    auth = 'header = ' + json.dumps('Authorization: Bearer ' + key) + '\nheader = "Content-Type: application/json"\n'
    args = ['curl', '--silent', '--show-error', '--no-buffer', '--http1.1', '--proto', '=https',
            '--noproxy', '*', '--connect-timeout', '30', '--max-time', str(model['timeout_s']),
            '--config', '-', '--data-binary', '@' + str(folder / 'REQUEST.json'),
            '--output', str(folder / 'response.raw'), '--dump-header', str(folder / 'response.headers'),
            '--write-out', '%{http_code}', model['endpoint']]
    begun = time.monotonic()
    try:
        result = subprocess.run(args, input=auth, text=True, capture_output=True, timeout=model['timeout_s'] + 60)
        code, http = result.returncode, int(result.stdout.strip() or 0)
        error = result.stderr.replace(key, '[REDACTED]')[-2000:]
    except subprocess.TimeoutExpired:
        code, http, error = 28, 0, 'Outer deadline exceeded; upstream completion unknown'
    raw = (folder / 'response.raw').read_bytes() if (folder / 'response.raw').exists() else b''
    parsed = parse(raw, model, request['expected_ids'], http, code)
    value, visible = parsed.pop('value'), parsed.pop('visible_output')
    (folder / 'FINAL.txt').write_text(visible)
    account = dict(start, **parsed, finished_utc=now(), elapsed_s=time.monotonic() - begun,
                   http_status=http, curl_returncode=code, transport_error=error,
                   response_sha256=sha_bytes(raw),
                   price_status='Retain raw usage; no assumption about actual provider debit')
    save(folder / 'ACCOUNTING.json', account)
    save(folder / 'RESULT.json', dict(status=account['status'], failure_category=account['failure_category'], value=value))
    return account, value
