"""Explicit opt-in Chat Completions transport with durable, fail-closed accounting.

Existing Gemini campaigns are unchanged. Secrets are read only at submission and
passed to curl over stdin, never via argv or experiment artifacts. No retries.
Dollar estimates use caller-supplied reference rates, not verified vendor bills.
"""
from __future__ import annotations
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
from .api_timeouts import generation_timeout_seconds
import time

class ProviderPaused(RuntimeError): pass

def save(path, value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    with tmp.open('w') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
    tmp.replace(path)

def usage_cost(usage, rates):
    required=['prompt_tokens','completion_tokens','total_tokens']
    if any(type(usage.get(k)) is not int or usage[k]<0 for k in required):
        raise ProviderPaused('Missing or invalid token usage')
    if usage['total_tokens']!=usage['prompt_tokens']+usage['completion_tokens']:
        raise ProviderPaused('Inconsistent token usage')
    cached=usage.get('prompt_tokens_details',{}).get('cached_tokens',0)
    if type(cached) is not int or not 0<=cached<=usage['prompt_tokens']:
        raise ProviderPaused('Invalid cache usage')
    # No cache discount assumed: conservative reference price, not vendor charge.
    return (usage['prompt_tokens']*rates['input_per_million']+usage['completion_tokens']*rates['output_per_million'])/1e6

class OpenAICompatible:
    def __init__(self, config, run_dir):
        self.config=dict(config);self.root=Path(run_dir);self.root.mkdir(parents=True,exist_ok=True)
        for k in ['token_budget','reference_usd_budget','max_requests','max_completion_tokens']:
            if not isinstance(config[k],(int,float)) or not math.isfinite(config[k]) or config[k]<=0: raise ValueError('Positive finite '+k+' required')
        if config['base_url']!='https://api.yunjintao.com/v1': raise ValueError('Unapproved endpoint')
        self.lock=(self.root/'provider.lock').open('a+')
        fcntl.flock(self.lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        self.state_path=self.root/'ledger.json'
        self.state=json.loads(self.state_path.read_text()) if self.state_path.exists() else {'config':self.config,'attempts':[],'paused':False}
        if self.state['config']!=self.config:
            self.close()
            raise ProviderPaused('Frozen provider config changed')
        if self.state['paused'] or any(a['status']=='inflight' for a in self.state['attempts']):
            self.close()
            raise ProviderPaused('Prior ambiguous attempt or pause requires reconciliation; no automatic retry')

    def generate(self, *, system, user, name):
        c=self.config
        request_timeout=generation_timeout_seconds(c.get("timeout_s"))
        payload={'model':c['model'],'messages':[{'role':'system','content':system},{'role':'user','content':user}],
                 'reasoning_effort':c['reasoning_effort'],'max_completion_tokens':c['max_completion_tokens'],'stream':False}
        if c.get('seed') is not None: payload['seed']=c['seed']
        if c.get('temperature') is not None: payload['temperature']=c['temperature']
        if c['model']=='gpt-6-astra' and 'temperature' in payload: raise ValueError('Astra does not support temperature')
        raw=json.dumps(payload,ensure_ascii=False).encode(); fingerprint=hashlib.sha256(raw).hexdigest()
        if any(a['request_sha256']==fingerprint for a in self.state['attempts']): raise ProviderPaused('Duplicate request; reuse saved result')
        if (self.root/'PAUSE').exists(): raise ProviderPaused('Operator pause')
        spent=sum(a.get('tokens',a['reserved_tokens']) for a in self.state['attempts'])
        dollars=sum(a.get('reference_usd',a['reserved_reference_usd']) for a in self.state['attempts'])
        # UTF-8 bytes plus a cushion for observed gateway prompt overhead.
        inp=len(raw)+8192; reserve=inp+c['max_completion_tokens']
        reference=(inp*c['rates']['input_per_million']+c['max_completion_tokens']*c['rates']['output_per_million'])/1e6
        if len(self.state['attempts'])>=c['max_requests'] or spent+reserve>c['token_budget'] or dollars+reference>c['reference_usd_budget']:
            raise ProviderPaused('Admission budget exhausted')
        key=Path(c['credential_file']).expanduser().read_text().strip()
        if not key or any(x in key for x in ['\n','\r','"','\\']): raise ValueError('Malformed credential')
        index=len(self.state['attempts']); request=self.root/f'{index:03d}.request.json'
        save(request,payload)
        attempt={'name':name,'request_sha256':fingerprint,'status':'inflight','started':time.time(),
                 'reserved_tokens':reserve,'reserved_reference_usd':reference,
                 'effective_timeout_s':request_timeout}
        self.state['attempts'].append(attempt);save(self.state_path,self.state)
        try:
            result=subprocess.run(['curl','--silent','--show-error','--max-time',str(request_timeout),
                '--config','-','-H','Content-Type: application/json','--data-binary','@'+str(request),
                '-w','\n%{http_code}',c['base_url']+'/chat/completions'],
                input='header = "Authorization: Bearer '+key+'"\n',text=True,capture_output=True,
                timeout=request_timeout+10)
            body,_,http=result.stdout.rpartition('\n')
            (self.root/f'{index:03d}.response.json').write_text(body.replace(key,'[REDACTED]'))
            attempt.update(http_status=http,curl_exit=result.returncode,elapsed_s=time.time()-attempt['started'])
            if result.returncode or http!='200': raise ProviderPaused('Transport/API failure; cost unknown')
            data=json.loads(body);usage=data.get('usage',{})
            cost=usage_cost(usage,c['rates'])
            attempt.update(tokens=usage['total_tokens'],reference_usd=cost,usage=usage,returned_model=data.get('model'))
            if data.get('model')!=c['model']: raise ProviderPaused('Returned model differs from requested model')
            choice=data['choices'][0]
            if choice.get('finish_reason')!='stop': raise ProviderPaused('Response did not finish normally')
            content=choice['message'].get('content')
            if not isinstance(content,str) or not content: raise ProviderPaused('Missing response content')
            attempt.update(status='ok',finished=time.time());save(self.state_path,self.state)
            return {'text':content,'usage':usage,'request_sha256':fingerprint,'reference_usd':cost,'model':data['model']}
        except Exception as e:
            attempt.update(status='paused',finished=time.time(),error_type=type(e).__name__)
            self.state['paused']=True;save(self.state_path,self.state)
            raise ProviderPaused(str(e).replace(key,'[REDACTED]')) from None

    def close(self):
        fcntl.flock(self.lock,fcntl.LOCK_UN);self.lock.close()
