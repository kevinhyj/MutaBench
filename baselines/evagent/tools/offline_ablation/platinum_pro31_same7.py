"""Same-7 protein–ligand ablation on gemini-3.1-pro-preview.

Prompts are the frozen 3.8 texts. Sequence is resubmitted because the model
changed. Fusion stays postprocessing. The five scope-boundary assays are absent.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import os
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE))
from evagent.offline_ablation import digest, file_hash, validate_scores, midranks
from evagent.integrated.chunked_scoring import ScoringResponseValidator
from evagent.integrated.research.concurrent_gateway import ConcurrentGateway
from evagent.integrated.self_check import SelfCheckPolicy, run_self_check
from features import save
from platinum import EXCLUDED, credential_value, fusion_artifact, load

PROJECT = CODE.parent
V3 = PROJECT / 'experiments/platinum_n10_cr25_ablation_20261004_v3'
FLASH = PROJECT / 'paper/data/zero_shot/complex/protein-ligand/gemini_ablation_same7_20261005'
DEFAULT_OUT = PROJECT / 'experiments/platinum_n10_cr25_pro31_same7_20261006'
PAPER = PROJECT / 'paper/data/zero_shot/complex/protein-ligand/gemini_ablation_same7_pro31_20261006'
HISTORY = PROJECT / 'work/final_mutation_gemini_cr25_seed42_20260928'
CREDENTIAL = PROJECT / 'private_config/evagent/credentials.env'
SAME7 = (
    'platinum_0ab21de83028e2b4',
    'platinum_1ba63813aabda8b0',
    'platinum_230fd561905fea66',
    'platinum_69a9636e43bdc4e3',
    'platinum_979d0884de082efa',
    'platinum_f2b7041bd5fedfed',
    'platinum_f60012ec5b476d7c',
)
CONDITIONS = ('sequence', 'msa_prompt', 'structure', 'msa_prompt_structure')
PROMPT_CONDITIONS = ('msa_prompt', 'structure', 'msa_prompt_structure')
METHODS = ('sequence', 'msa_gemini_rank5050', 'msa_prompt', 'structure',
           'msa_prompt_structure', 'msa_gemini_rank5050_structure')
LABELS = {
    'sequence': '原序列',
    'msa_gemini_rank5050': '原序列+MSA 50/50',
    'msa_prompt': 'MSA prompt',
    'structure': 'Structure prompt',
    'msa_prompt_structure': 'MSA+Structure prompt',
    'msa_gemini_rank5050_structure': 'Structure+MSA 50/50',
}
MODEL = {
    'model': 'gemini-3.1-pro-preview',
    'thinking_level': 'medium',
    'temperature': 0.0,
    'max_output_tokens': 65536,
    'timeout_s': 900.0,
    'include_thoughts': True,
    'api_key_env': 'GEMINI_API_KEY',
    'endpoint': None,
    'proxy': 'http://127.0.0.1:7777',
    'provider': 'google_genai',
    'retries': 0,
    'generation_seed': 42,
    'transport_mode': 'stream',
}


def prompt_sha(system, user):
    return hashlib.sha256((system + '\n' + user).encode()).hexdigest()


def prepare(out):
    if out.exists() and any(out.iterdir()):
        raise ValueError('Preparation requires a new empty output directory')
    overlap = set(SAME7) & set(EXCLUDED)
    if overlap or len(SAME7) != 7 or len(set(SAME7)) != 7:
        raise ValueError('Same-7 set is not the intended complete-MSA subset')
    plan_v3 = load(V3 / 'plan.json')
    by_id = {row['assay_id']: row for row in plan_v3['assays']}
    queue = []
    assays = []
    for eid in SAME7:
        record = by_id[eid]
        if record['status'] != 'prepared' or not record['msa_complete']:
            raise ValueError('Assay is outside the complete-MSA same-7 set: ' + eid)
        base = load(V3 / 'assays' / eid / 'base_prompt.json')
        features = load(V3 / 'assays' / eid / 'features.json')
        ids = list(base['expected_ids'])
        if ids != [f'M{i:03d}' for i in range(1, 11)]:
            raise ValueError('Candidate ids changed')
        if not features['full_msa_coverage']:
            raise ValueError('MSA coverage is incomplete')
        validate_scores(features['independent_msa_scores'], ids)
        specs = [('sequence', base['system'], base['user'], str(V3 / 'assays' / eid / 'base_prompt.json'))]
        for condition in PROMPT_CONDITIONS:
            unit = next(item for item in load(V3 / 'queue.json')
                        if item['assay_id'] == eid and item['condition'] == condition)
            request = load(V3 / 'requests' / f"{unit['request_hash']}.json")
            if request['user'] == base['user'] or request['system'] != base['system']:
                raise ValueError('Frozen prompt contract differs')
            if request['expected_ids'] != ids:
                raise ValueError('Candidate set differs from the sequence prompt')
            specs.append((condition, request['system'], request['user'],
                          str(V3 / 'requests' / f"{unit['request_hash']}.json")))
        for condition, system, user, source in specs:
            body = dict(assay_id=eid, condition=condition, chunk_index=0, CR=25, N=10,
                        expected_ids=ids, system=system, user=user,
                        prompt_sha256=prompt_sha(system, user), source_prompt=source,
                        model='gemini-3.1-pro-preview', thinking_level='medium', generation_seed=42)
            body['request_hash'] = digest(body)
            save(out / 'requests' / f"{body['request_hash']}.json", body)
            queue.append(dict(request_hash=body['request_hash'], assay_id=eid, condition=condition))
        assays.append(dict(assay_id=eid, reference_complex_id=record['reference_complex_id'],
                           group_id=record['group_id'], N=10, status='prepared', msa_complete=True))
        print(json.dumps(dict(event='prepared', assay=eid, complex=record['reference_complex_id'])), flush=True)
    save(out / 'queue.json', queue)
    plan = dict(schema='platinum-n10-cr25-pro31-same7-v1', model_config=MODEL, assays=assays,
                conditions=list(CONDITIONS), CR=25, N=10, sample_seed=42, generation_seed=42,
                weights=dict(gemini=0.5, msa=0.5), workers=4, max_repairs=1, token_budget=8_000_000,
                source_v3=str(V3), labels_read_by_predictor=False,
                excluded_not_queued=sorted(EXCLUDED),
                planned_new_requests=len(queue))
    plan['fingerprint'] = digest(plan)
    save(out / 'plan.json', plan)
    print(json.dumps(dict(event='plan_ready', requests=len(queue))), flush=True)


def _one(out, plan, gateway, unit):
    key = unit['request_hash']
    target = out / 'results' / f'{key}.json'
    if target.exists():
        result = load(target)
        if result.get('status') == 'ok':
            validate_scores(result['value'], result['expected_ids'], bounded=True)
            return result
        raise ValueError('Existing non-ok result is left in place: ' + key)
    request = load(out / 'requests' / f'{key}.json')
    attempts = []

    class PromptBlocked(Exception):
        pass

    def generate(prompt, attempt):
        response = gateway.generate(
            system=request['system'], user=prompt, run_name='platinum_cr25_pro31_same7',
            metadata=dict(assay_id=request['assay_id'], condition=request['condition'],
                          chunk_index=0, attempt=attempt, request_hash=key))
        attempts.append(response)
        if (response.get('response_metadata') or {}).get('prompt_feedback', {}).get('block_reason'):
            raise PromptBlocked()
        version = str((response.get('response_metadata') or {}).get('model_version', ''))
        if response.get('status') == 'ok' and not version.startswith(plan['model_config']['model']):
            raise ValueError('Actual provider model identity mismatch: ' + version)
        return response

    try:
        result = run_self_check(original_prompt=request['user'], generate=generate,
                                validator=ScoringResponseValidator(request['expected_ids']),
                                policy=SelfCheckPolicy(max_repairs=plan['max_repairs']))
    except PromptBlocked:
        result = dict(status='provider_blocked', value=None, attempts=attempts)
    except Exception as error:
        result = dict(status='error', value=None, error_type=type(error).__name__,
                      error=str(error)[:500], attempts=attempts)
    result.update(request_hash=key, assay_id=request['assay_id'], condition=request['condition'],
                  expected_ids=request['expected_ids'], model_config=plan['model_config'],
                  fingerprint=plan['fingerprint'])
    if result['status'] == 'ok':
        validate_scores(result['value'], request['expected_ids'], bounded=True)
        version = str((result['attempts'][-1].get('response_metadata') or {}).get('model_version', ''))
        if not version.startswith(MODEL['model']):
            result.update(status='model_mismatch', value=None, error=version)
    save(target, result)
    print(json.dumps(dict(event='settled', assay=request['assay_id'], condition=request['condition'],
                          status=result['status'])), flush=True)
    return result


def run(out, credential_path):
    plan = load(out / 'plan.json')
    if digest({k: v for k, v in plan.items() if k != 'fingerprint'}) != plan['fingerprint']:
        raise ValueError('Plan changed')
    if plan['model_config'] != MODEL or plan['planned_new_requests'] != 28:
        raise ValueError('Run contract differs')
    os.environ[MODEL['api_key_env']] = credential_value(credential_path)
    gateway = ConcurrentGateway(model_config=MODEL, run_dir=out / 'provider',
                                fingerprint=plan['fingerprint'], token_budget=plan['token_budget'],
                                reserve_budget=True, max_concurrent_requests=plan['workers'])
    queue = load(out / 'queue.json')
    try:
        first = _one(out, plan, gateway, queue[0])
        if first['status'] != 'ok':
            print(json.dumps(dict(event='pilot_failed', status=first['status'],
                                  error=first.get('error'))), flush=True)
            return
        pending = [unit for unit in queue[1:] if not (out / 'results' / f"{unit['request_hash']}.json").exists()]
        with concurrent.futures.ThreadPoolExecutor(max_workers=plan['workers']) as pool:
            list(pool.map(lambda unit: _one(out, plan, gateway, unit), pending))
    finally:
        summary = gateway.summary()
        gateway.close()
        path = out / 'usage.json'
        path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
        print(json.dumps(dict(event='usage_written', provider_attempts=summary.get('provider_attempts'),
                              reported_tokens=summary.get('reported_tokens'),
                              paused=summary.get('paused'))), flush=True)


def _usage(result):
    attempts = [item for item in result.get('attempts') or [] if item.get('status') == 'ok']
    if not attempts:
        attempts = result.get('attempts') or []
    totals = dict(input_tokens=0, output_tokens=0, reasoning_tokens=0, total_tokens=0, elapsed_s=0.0, calls=0)
    for item in attempts:
        usage = item.get('usage_summary') or {}
        for key in ('input_tokens', 'output_tokens', 'reasoning_tokens', 'total_tokens'):
            totals[key] += int(usage.get(key) or 0)
        totals['elapsed_s'] += float(item.get('elapsed_s') or 0)
        totals['calls'] += 1
    return totals


def analyze(out):
    plan = load(out / 'plan.json')
    queue = load(out / 'queue.json')
    from scipy.stats import spearmanr
    rows = []
    usage_rows = []
    for assay in plan['assays']:
        eid = assay['assay_id']
        features = load(V3 / 'assays' / eid / 'features.json')
        ids = [f'M{i:03d}' for i in range(1, 11)]
        scores = {}
        for condition in CONDITIONS:
            unit = next(item for item in queue if item['assay_id'] == eid and item['condition'] == condition)
            result = load(out / 'results' / f"{unit['request_hash']}.json")
            if result['status'] != 'ok':
                raise ValueError(f'{eid} {condition} is {result["status"]}')
            validate_scores(result['value'], ids, bounded=True)
            scores[condition] = result['value']
            usage = _usage(result)
            usage_rows.append(dict(assay_id=eid, condition=condition, **usage))
        def origin_for(condition):
            unit = next(item for item in queue if item['assay_id'] == eid and item['condition'] == condition)
            path = out / 'results' / f"{unit['request_hash']}.json"
            return dict(source_response=str(path), source_response_sha256=file_hash(path))
        sequence_fusion = fusion_artifact(ids, scores['sequence'], features, origin_for('sequence'),
                                          condition='postprocessing_of_sequence_scores')
        structure_fusion = fusion_artifact(ids, scores['structure'], features, origin_for('structure'),
                                           condition='postprocessing_of_structure_scores')
        if sequence_fusion['status'] != 'complete' or structure_fusion['status'] != 'complete':
            raise ValueError('Same-7 fusion incomplete for ' + eid)
        save(out / 'assays' / eid / 'msa_gemini_rank5050.json', sequence_fusion)
        save(out / 'assays' / eid / 'msa_gemini_rank5050_structure.json', structure_fusion)
        predictions = dict(scores)
        predictions['msa_gemini_rank5050'] = {row['candidate_id']: row['fusion_value'] for row in sequence_fusion['rows']}
        predictions['msa_gemini_rank5050_structure'] = {row['candidate_id']: row['fusion_value'] for row in structure_fusion['rows']}
        labels = load(HISTORY / 'episodes' / eid / 'evaluation_labels.json')['labels']
        validate_scores(labels, ids)
        row = dict(assay_id=eid, assay_group_id=assay['group_id'],
                   reference_complex_id=assay['reference_complex_id'], N=10, CR=25)
        for method, values in predictions.items():
            validate_scores(values, ids)
            ordered = [values[i] for i in ids]
            gold = [labels[i] for i in ids]
            row[method] = float(spearmanr(ordered, gold).statistic)
            ranking = dict(method=method, rows=[
                dict(candidate_id=i, score=s, final_rank=r)
                for i, s, r in zip(ids, ordered, midranks(ordered, descending=True))])
            target = out / 'assays' / eid / 'rankings' / f'{method}.json'
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(ranking, indent=2, allow_nan=False) + '\n')
        rows.append(row)
        print(json.dumps(dict(event='analyzed', assay=eid, structure=row['structure'])), flush=True)
    _write_csv(out / 'per_assay.csv', rows)
    _write_csv(out / 'gemini_usage_by_assay.csv', usage_rows)
    means = []
    for method in METHODS:
        calls = [item for item in usage_rows if item['condition'] == method]
        reused = 'sequence' if method == 'msa_gemini_rank5050' else 'structure' if method == 'msa_gemini_rank5050_structure' else ''
        reused_rows = [item for item in usage_rows if item['condition'] == reused]
        means.append(dict(
            method=method, label=LABELS[method],
            mean_signed_spearman=sum(row[method] for row in rows) / len(rows),
            assays=len(rows), N=10, CR=25,
            gemini_requests=len(calls), new_gemini_calls=sum(item['calls'] for item in calls),
            input_tokens=sum(item['input_tokens'] for item in calls),
            output_tokens=sum(item['output_tokens'] for item in calls),
            reasoning_tokens=sum(item['reasoning_tokens'] for item in calls),
            total_tokens=sum(item['total_tokens'] for item in calls),
            sum_request_elapsed_s=sum(item['elapsed_s'] for item in calls),
            reused_from=reused,
            reused_total_tokens=sum(item['total_tokens'] for item in reused_rows),
            reused_sum_request_elapsed_s=sum(item['elapsed_s'] for item in reused_rows),
        ))
    _write_csv(out / 'mean_spearman.csv', means)
    print(json.dumps({row['method']: row['mean_signed_spearman'] for row in means}, ensure_ascii=False), flush=True)


def export(out, paper):
    if paper.exists() and any(paper.iterdir()):
        raise ValueError('Paper export requires a new directory')
    paper.mkdir(parents=True)
    for name in ('per_assay.csv', 'mean_spearman.csv', 'gemini_usage_by_assay.csv'):
        (paper / name).write_text((out / name).read_text())
    flash_mean = list(csv.DictReader((FLASH / 'mean_spearman.csv').open()))
    flash_assay = list(csv.DictReader((FLASH / 'per_assay.csv').open()))
    pro_mean = list(csv.DictReader((out / 'mean_spearman.csv').open()))
    pro_assay = list(csv.DictReader((out / 'per_assay.csv').open()))
    flash_by_label = {row['label']: row for row in flash_mean}
    comparisons = []
    for row in pro_mean:
        old = flash_by_label[row['label']]
        old_rho = float(old['mean_signed_spearman'])
        new_rho = float(row['mean_signed_spearman'])
        comparisons.append(dict(
            method=row['method'], label=row['label'],
            spearman_3_8_flash_medium=old_rho, spearman_3_1_pro_medium=new_rho,
            delta_pro_minus_flash=new_rho - old_rho,
            tokens_3_8=int(old['total_tokens']), tokens_3_1=int(float(row['total_tokens'])),
            elapsed_s_3_8=float(old['sum_request_elapsed_s']),
            elapsed_s_3_1=float(row['sum_request_elapsed_s']),
        ))
    _write_csv(paper / 'mean_comparison.csv', comparisons)
    method_alias = {'sequence_cached': 'sequence'}
    per = []
    for old in flash_assay:
        new = next(row for row in pro_assay if row['assay_id'] == old['assay_id'])
        item = dict(assay_id=old['assay_id'], assay_group_id=old['assay_group_id'],
                    reference_complex_id=old['reference_complex_id'])
        for column, method in (('sequence_cached', 'sequence'), ('msa_gemini_rank5050', 'msa_gemini_rank5050'),
                               ('msa_prompt', 'msa_prompt'), ('structure', 'structure'),
                               ('msa_prompt_structure', 'msa_prompt_structure'),
                               ('msa_gemini_rank5050_structure', 'msa_gemini_rank5050_structure')):
            old_rho = float(old[column])
            new_rho = float(new[method_alias.get(column, column)])
            item[LABELS[method] + ' 3.8'] = old_rho
            item[LABELS[method] + ' 3.1'] = new_rho
            item[LABELS[method] + ' Δ'] = new_rho - old_rho
        per.append(item)
    _write_csv(paper / 'per_assay_comparison.csv', per)
    lines = [
        'Protein–ligand Gemini 消融：同一批 7 个 assay，gemini-3.1-pro-preview，thinking medium。',
        '',
        '对照是 gemini-3.8-flash medium 的同 7 assay 表。',
        '提示文本与 3.8 的冻结请求一致。原序列这次重新提交，因为模型变了，不能沿用 3.8 分数。',
        '两路 50/50 仍是固定权重的事后融合，没有新的 Gemini 调用，也没有用标签拟合权重。',
        'N=10，CR=25，seed=42。有符号 Spearman，等权平均，负值保留。',
        '五个范围外 assay 没有进入队列。',
        '',
        '均分对照（3.1 Pro − 3.8 Flash）：',
    ]
    for row in comparisons:
        lines.append(
            f"{row['label']}: 3.8={row['spearman_3_8_flash_medium']:.4f}  "
            f"3.1={row['spearman_3_1_pro_medium']:.4f}  "
            f"Δ={row['delta_pro_minus_flash']:+.4f}")
    (paper / 'README.txt').write_text('\n'.join(lines) + '\n')
    print(json.dumps(dict(event='exported', paper=str(paper))), flush=True)


def _write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('prepare', 'run', 'analyze', 'export', 'all'))
    parser.add_argument('--output', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--paper', type=Path, default=PAPER)
    parser.add_argument('--credential', type=Path, default=CREDENTIAL)
    args = parser.parse_args()
    if args.command in ('prepare', 'all'):
        prepare(args.output)
    if args.command in ('run', 'all'):
        run(args.output, args.credential)
    if args.command in ('analyze', 'all'):
        analyze(args.output)
    if args.command in ('export', 'all'):
        export(args.output, args.paper)


if __name__ == '__main__':
    main()
