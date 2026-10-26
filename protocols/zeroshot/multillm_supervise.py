"""Background supervision and independent periodic evaluation of a frozen campaign."""
import argparse
import collections
import csv
import fcntl
import io
import json
import math
from pathlib import Path
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone, timedelta

from multillm_transport import load, save, sha, now

HERE=Path(__file__).resolve().parent
ARMS={'sequence':'序列','msa_prompt':'序列＋MSA','structure':'序列＋结构','msa_prompt_structure':'序列＋MSA＋结构'}
NAMES={'argon_high':'Gemini 4 Argon · high','qwen38_27b_xhigh':'Qwen 3.8 27B · xhigh','flash38_low':'Gemini 3.8 Flash · low','flash38_high':'Gemini 3.8 Flash · high'}
NAMES.update(flash38_medium='Gemini 3.8 Flash · medium', qwen38_27b_high='Qwen 3.8 27B · high')
NAMES.update(sol61_max='GPT 6.1 Sol · max')


def atomic_text(p,text):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp');tmp.write_text(text);tmp.replace(p)


def evaluate_and_publish(run):
    plan=load(run/'PLAN.json');status=load(run/'STATUS.json');tasks=load(run/'TASKS.json');metrics={};eval_errors={}
    if status['state']=='waiting_for_previous_inflight_calls':
        previous=status['previous_status']
        while previous.get('state')=='waiting_for_previous_inflight_calls' and 'previous_status' in previous:
            previous=previous['previous_status']
        text='\n'.join(['# Zero-shot 多模型打分进度','',
            '状态：等待旧批次在途请求自然结束，随后自动导入响应、离线重验并继续未提交任务。',
            f'更新时间：{datetime.now(timezone(timedelta(hours=8))).isoformat()}。',
            f'旧批次已结算 {previous.get("completed",0)}/{plan["planned_scoring_requests"]}，有效 {previous.get("valid",0)}，仍在途 {previous.get("active",0)}。',
            '承接已有响应并按统一字段容错规则重验；已有模型的Prompt、候选及配置保持不变，不重发已有请求。新增模型读取相同冻结Prompt。',
            f'新运行目录：`{run}`。',f'旧批次明细：`{status["previous_run"]}/PROGRESS.md`。',''])
        for key in plan.get('added_model_keys',[]):
            probe=run/'preflight'/key/'ACCOUNTING.json'
            a=load(probe) if probe.exists() else {}
            text+=f'新增模型：{NAMES[key]}；计划 {sum(t["model_key"]==key for t in tasks)} 请求；预检：{a.get("failure_category","待提交")}；打分等待队列衔接。\n'
        atomic_text(run/'PROGRESS.md',text);atomic_text(HERE/'LLM_SCORING_PROGRESS.md',text)
        return
    for key in plan['models']:
        child=run/'models'/key
        result=subprocess.run([sys.executable,'-B',str(HERE/'evaluate_frozen.py'),'--run',str(child)],capture_output=True,text=True)
        if result.returncode:
            eval_errors[key]=result.stderr[-1500:];continue
        for r in load(child/'evaluation/METRICS.json')['rows']:
            metrics[key,r['domain'],r['assay'],r['arm']]=r
    groups=collections.defaultdict(list);details=[];active={(t['model_key'],t['request_id']) for t in status.get('active_tasks',[])}
    for t in tasks:
        key=t['model_key'];uid=t['request_id'];child=run/'models'/key;p=child/'units'/(uid+'.json')
        unit=load(p) if p.exists() else {};metric=metrics.get((key,t['domain'],t['assay'],t['arm']),{})
        valid=unit.get('status')=='ok';running=(key,uid) in active
        stop=status.get('stop_reasons',{}).get(key)
        state='complete' if valid else 'failed' if unit else 'running' if running else 'held' if stop else 'queued'
        rho=metric.get('spearman') if metric.get('status')=='complete' else None
        detail=dict(**t,status=state,reason=unit.get('failure_category') if unit and not valid else stop if state=='held' else '',spearman=rho,
                    n_predicted=metric.get('n_predicted',0),n_expected=metric.get('n_expected'))
        details.append(detail);groups[key,t['domain'],t['arm']].append(detail)
    summary=[]
    for key in plan['models']:
        for domain in sorted({t['domain'] for t in tasks}):
            for arm in ARMS:
                group=groups[key,domain,arm];counts=collections.Counter(r['status'] for r in group)
                assay_values={r['assay']:r['spearman'] for r in group if r['spearman'] is not None and math.isfinite(r['spearman'])}
                values=list(assay_values.values())
                reasons=collections.Counter(r['reason'] for r in group if r['reason'])
                summary.append(dict(model_key=key,model=NAMES[key],domain=domain,arm=arm,planned_assays=len({r['assay'] for r in group}),planned_requests=len(group),valid_predictions=counts['complete'],
                    evaluated_assays=len(values),mean_signed_spearman=sum(values)/len(values) if values else None,
                    active=counts['running'],queued=counts['queued'],failed=counts['failed'],held=counts['held'],reasons=dict(reasons)))
    out=run/'evaluation';out.mkdir(exist_ok=True)
    for name,rows in [('per_request.csv',details),('summary.csv',summary)]:
        stream=io.StringIO();w=csv.DictWriter(stream,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows);atomic_text(out/name,stream.getvalue())
    save(out/'SUMMARY.json',dict(updated_utc=now(),rows=summary,evaluator_errors=eval_errors,partial_not_scored=True,undefined_not_zero=True))
    tokens={};preflight={}
    for key in plan['models']:
        records=[load(p) for p in (run/'models'/key/'attempts').glob('*/ACCOUNTING.json')]
        tokens[key]={field:dict(reported_sum=sum(a[field] for a in records if type(a.get(field)) is int),known_attempts=sum(type(a.get(field)) is int for a in records),unknown_attempts=sum(type(a.get(field)) is not int for a in records)) for field in ['input_tokens','output_tokens','reasoning_tokens','total_tokens']}
        p=run/'preflight'/key/'ACCOUNTING.json'
        if p.exists():
            a=load(p);preflight[key]={k:a.get(k) for k in ['status','failure_category','http_status','returned_models','thinking_level','elapsed_s','reasoning_tokens']}
    save(run/'USAGE.json',dict(updated_utc=now(),models=tokens,preflight=preflight,gemini_monetary_accounting=False))
    beijing=datetime.now(timezone(timedelta(hours=8))).isoformat()
    lines=['# Zero-shot 多模型打分进度','',f'更新：{beijing}；状态：{status["state"]}。',
        f'本批总并发上限 **{plan["workers"]}**，当前进行中 **{status.get("active",0)}**；有效 **{status.get("valid",0)}/{len(tasks)}**，已结算 **{status.get("completed",0)}/{len(tasks)}**。预检不计入打分分母；本版本新增{plan.get("planned_preflight_requests",len(plan["models"]))}次，其余沿用原记录。','',
        f'当前范围：{len({(t["domain"],t["assay"]) for t in tasks})}个assay，四臂，{len(plan["models"])}个模型配置；候选数量按各冻结query。全部读取验收后的prompt，候选与顺序不变。','',
        f'运行目录：`{run}`。',f'模型与配置路径索引：`{HERE / "llm_models.json"}`。',f'冻结输入：`{plan["inputs"]}`。','',
        '| 模型 | 配置预检 | 已结算 | 有效 | 活跃 | 未提交 | 停止原因 |','| --- | --- | ---: | ---: | ---: | ---: | --- |']
    for key in plan['models']:
        s=status.get('models',{}).get(key,{});probe=preflight.get(key,{})
        planned=sum(t['model_key']==key for t in tasks)
        lines.append(f'| {NAMES[key]} | {probe.get("failure_category","等待／进行中")} | {s.get("completed",0)}/{planned} | {s.get("valid",0)} | {s.get("active",0)} | {s.get("not_submitted",planned)} | {s.get("stop_reason") or "无"} |')
    for domain in sorted({t['domain'] for t in tasks}):
        title=domain
        lines += ['',f'## {title}','','| 模型 | Arm | 有效请求 | 完整且相关有定义 | Spearman均值 | 活跃/排队 | 失败/暂缓及原因 |','| --- | --- | ---: | ---: | ---: | --- | --- |']
        for s in summary:
            if s['domain']!=domain:continue
            mean=f'{s["mean_signed_spearman"]:.4f}' if s['mean_signed_spearman'] is not None else '—'
            reasons='；'.join(f'{k}: {n}' for k,n in s['reasons'].items()) or '无'
            lines.append(f'| {s["model"]} | {ARMS[s["arm"]]} | {s["valid_predictions"]}/{s["planned_requests"]} | {s["evaluated_assays"]} | {mean} | {s["active"]}/{s["queued"]} | {s["failed"]}/{s["held"]}；{reasons} |')
    lines += ['','均值仅包含完整冻结query、独立评估且相关有定义的assay；多个CR25块按assay只计一次；未完成不填零，各行完成集合可能不同。这里只显示预检接受的参数及供应商返回的模型名称，不能独立验证服务端实际思考强度。','',
        '## 后续体系','','本批范围由已验收输入包决定；新增范围需新版本。ADMET没有MSA四臂，需独立端点评估协议。','',
        f'逐请求进度与assay指标：`{out / "per_request.csv"}`。',f'各模型预测：`{run / "models"}/<model_key>/units/`；实际请求响应：同模型目录`attempts/`；token：`{run / "USAGE.json"}`。',
        '创建运行目录下的`PAUSE`文件只停止后续提交，已提交请求继续等待；删除该文件不会自动重启或重发。','']
    scope_path=Path(plan['inputs'])/'SCOPE.json'
    if scope_path.exists():
        scope=load(scope_path);held=[r for r in scope['rows'] if r['status']!='included']
        lines += ['## 原定范围与未提交项','',
            f'原定范围：{scope["original_counts"]}。本批执行范围以以上实际分母为准；{len(held)}项未纳入提交，完整记录：`{scope_path}`。',
            '未提交项不计为模型失败、缺数据或已完成。本轮不对病原体受体结合／抗体逃逸条目生成逐变体功能预测。','']
    text='\n'.join(lines);atomic_text(run/'PROGRESS.md',text);atomic_text(HERE/'LLM_SCORING_PROGRESS.md',text)


def supervise(run):
    with (run/'SUPERVISOR.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        plan=load(run/'PLAN.json')
        if plan.get('resume_from'):
            from multillm_campaign import preflight
            for key in plan.get('added_model_keys',[]):
                if (run/'preflight'/key).exists():continue
                save(run/'STATUS.json',dict(state='preflight',updated_utc=now(),planned=plan['planned_scoring_requests'],completed=0,valid=0,active=0,preflight_model=key))
                try:preflight(run,key)
                except (Exception,SystemExit) as error:
                    save(run/'preflight'/key/'ERROR.json',dict(type=type(error).__name__,time=now()))
        with (run/'scoring.log').open('a') as log:
            worker=subprocess.Popen([sys.executable,'-B',str(HERE/'multillm_campaign.py'),'run','--run',str(run)],stdout=log,stderr=subprocess.STDOUT)
            while worker.poll() is None:
                try:evaluate_and_publish(run)
                except Exception as e:save(run/'MONITOR_ERROR.json',dict(type=type(e).__name__,time=now()))
                try:worker.wait(timeout=30)
                except subprocess.TimeoutExpired:pass
            if worker.returncode:
                s=load(run/'STATUS.json');s.update(state='worker_failed',worker_returncode=worker.returncode,updated_utc=now());save(run/'STATUS.json',s)
            evaluate_and_publish(run)
            save(run/'SUPERVISOR_RESULT.json',dict(returncode=worker.returncode,finished_utc=now()))


def launch(run, execute):
    from multillm_campaign import validate
    validate(run)
    unit='evagent-zs-multillm-'+sha(run/'PLAN.json')[:12]
    argv=['systemd-run','--user','--unit',unit,'--property=Restart=no','--property=WorkingDirectory='+str(HERE),
        '--property=Environment=PYTHONUNBUFFERED=1','--property=UMask=0077',
        '--property=StandardOutput=append:'+str(run/'supervisor.log'),'--property=StandardError=append:'+str(run/'supervisor.log'),
        sys.executable,'-B',str(Path(__file__).resolve()),'supervise','--run',str(run)]
    if not execute:print(shlex.join(argv));return
    active=subprocess.run(['systemctl','--user','is-active',unit+'.service'],capture_output=True,text=True)
    if active.stdout.strip()=='active':raise ValueError('Service already active')
    result=subprocess.run(argv,capture_output=True,text=True)
    save(run/'SERVICE.json',dict(unit=unit+'.service',argv=argv,returncode=result.returncode,stdout=result.stdout,stderr=result.stderr))
    print(json.dumps(dict(unit=unit,returncode=result.returncode,progress=str(HERE/'LLM_SCORING_PROGRESS.md'))))
    if result.returncode:raise SystemExit(result.returncode)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['launch','supervise','refresh']);p.add_argument('--run',type=Path,required=True);p.add_argument('--execute',action='store_true');a=p.parse_args();run=a.run.resolve()
    if a.action=='launch':launch(run,a.execute)
    elif a.action=='refresh':evaluate_and_publish(run)
    else:supervise(run)
