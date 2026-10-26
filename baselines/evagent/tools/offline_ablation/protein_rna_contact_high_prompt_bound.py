"""Rescore protein–RNA contact ablation at Gemini 3.8 Flash thinking HIGH.

The only prompt change relative to the frozen HIGH campaign is one sentence
telling the model to reason from the prompt and not from memory. Sequence
stays in every prompt. Only 2ERR, 4BS2 and 4CIO add the contact block.
Duplicate prompts are scored once. Labels are read only by evaluate().
"""
from __future__ import annotations

import argparse
import asyncio
import copy
import csv
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time
from urllib.parse import urlsplit

CODE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE / 'tools/offline_ablation'))
from protein_rna_v2 import credential_value, legacy_module  # noqa: E402
from evagent.offline_ablation import fuse_rankings  # noqa: E402

MODEL = 'gemini-3.8-flash'
ENDPOINT = 'https://generativelanguage.googleapis.com/v1beta/models/gemini-3.8-flash:generateContent'
LEVELS = ('HIGH',)
SENTENCE = 'Do not retrieve memory; reason only from the content of this prompt.'
ANCHOR = '\n\nTask:\n'
MAX_OUTPUT_TOKENS = 65536
SEQUENCE = 'sequence'
MSA = 'msa_prompt'
STRUCTURE = 'structure_contact'
JOINT = 'msa_prompt_structure_contact'
FUSION = 'msa_gemini_rank5050_structure_contact'
FUSION_SEQ = 'msa_gemini_rank5050'
METHODS = (SEQUENCE, FUSION_SEQ, MSA, STRUCTURE, JOINT, FUSION)
FLASH = Path('/data/huangyanjie/EVAgent/experiments/protein_rna_ablation_N10_CR25_20261005_structure_contact')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def apply_thinking(request, level):
    if level not in LEVELS:
        raise ValueError('This campaign only rescores thinking HIGH')
    request['generationConfig']['thinkingConfig']['thinkingLevel'] = level
    request['generationConfig']['thinkingConfig']['includeThoughts'] = True
    request['generationConfig']['maxOutputTokens'] = MAX_OUTPUT_TOKENS
    return request


def bind_prompt(text):
    if text.count(ANCHOR) != 1:
        raise ValueError('Prompt is missing the single Task anchor')
    if SENTENCE in text:
        raise ValueError('Prompt-bound sentence is already present')
    return text.replace(ANCHOR, ' ' + SENTENCE + ANCHOR, 1)


def unbind_prompt(text):
    needle = ' ' + SENTENCE + ANCHOR
    if text.count(SENTENCE) != 1 or text.count(needle) != 1:
        raise ValueError('Prompt-bound sentence is missing or repeated')
    return text.replace(needle, ANCHOR, 1)


def prepare(args):
    out = args.output
    flash = args.flash
    baseline = args.baseline
    level = args.thinking
    if out.exists() and any(out.iterdir()):
        raise ValueError('Output directory is not empty; refuse to mix campaigns')
    out.mkdir(parents=True, exist_ok=True)
    cohort = read(flash / 'cohort.json')
    if len(cohort) != 10:
        raise ValueError('Expected the frozen 10-assay cohort')
    index = []
    for assay in cohort:
        assay_id = assay['id']
        sequence_request = read(baseline / 'sequence_requests' / (assay_id + '.json'))
        sequence_text = sequence_request['contents'][0]['parts'][0]['text']
        msa_text = (baseline / 'prompts/msa_prompt' / (assay_id + '.txt')).read_text()
        structure_text = (flash / 'prompts/structure_contact' / (assay_id + '.txt')).read_text()
        joint_text = (flash / 'prompts/msa_prompt_structure_contact' / (assay_id + '.txt')).read_text()
        texts = {SEQUENCE: bind_prompt(sequence_text), MSA: bind_prompt(msa_text),
                 STRUCTURE: bind_prompt(structure_text), JOINT: bind_prompt(joint_text)}
        copies = {
            STRUCTURE: SEQUENCE if structure_text == sequence_text else None,
            JOINT: MSA if joint_text == msa_text else None,
        }
        if assay['reference_complex_id'] in ('2ERR', '4BS2', '4CIO'):
            if copies[STRUCTURE] or copies[JOINT]:
                raise ValueError('Assay-matched contact prompt collapsed to the base prompt')
        elif copies[STRUCTURE] != SEQUENCE or copies[JOINT] != MSA:
            raise ValueError('Unmatched contact prompt differs from the base prompt')
        for condition, text in texts.items():
            request = apply_thinking(copy.deepcopy(sequence_request), level)
            request['contents'][0]['parts'][0]['text'] = text
            request_path = out / 'requests' / condition / (assay_id + '.json')
            prompt_path = out / 'prompts' / condition / (assay_id + '.txt')
            request_path.parent.mkdir(parents=True, exist_ok=True)
            prompt_path.parent.mkdir(parents=True, exist_ok=True)
            request_path.write_text(json.dumps(request, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
            prompt_path.write_text(text)
            source = copies.get(condition)
            index.append(dict(assay, id=condition + '__' + assay_id, base_unit_id=assay_id, condition=condition,
                              request_path=str(request_path.relative_to(out)), request_sha256=sha(request_path),
                              reused_exact_prompt=source is not None,
                              score_source_unit=(source + '__' + assay_id) if source else None))
    protocol = dict(read(flash / 'config.json')['protocol'], requested_model=MODEL, endpoint=ENDPOINT,
                    thinking_level=level, max_output_tokens=MAX_OUTPUT_TOKENS, http_proxy=args.proxy,
                    prompt_bound_sentence=SENTENCE)
    new_count = sum(not row['reused_exact_prompt'] for row in index)
    reused = sum(row['reused_exact_prompt'] for row in index)
    if new_count != 26 or reused != 14:
        raise ValueError('Expected 26 new prompts and 14 exact copies')
    write(out / 'config.json', dict(schema='protein-rna-structure-contact-gemini38-high-prompt-bound-v1',
                                    flash_campaign=str(flash), baseline=str(baseline), N=10, CR=25,
                                    protocol=protocol, model=MODEL, thinking_level=level,
                                    prompt_bound_sentence=SENTENCE,
                                    new_request_count=new_count, reused_exact_prompts=reused,
                                    max_attempts=3, concurrency=3, total_deadline_seconds=600,
                                    created_unix=time.time(), runner_sha256=sha(Path(__file__))))
    write(out / 'cohort.json', cohort)
    write(out / 'index.json', index)
    print(json.dumps(dict(new_requests=new_count, reused=reused, model=MODEL, thinking=level)), flush=True)


def validate(args):
    out = args.output
    config = read(out / 'config.json')
    level = config['thinking_level']
    if sha(Path(__file__)) != config['runner_sha256']:
        raise ValueError('Runner differs from the frozen thinking campaign')
    if config['model'] != MODEL or level not in LEVELS:
        raise ValueError('Model or thinking level is not the frozen Flash setting')
    if config['protocol']['generation_seed'] != 42 or config['new_request_count'] != 26:
        raise ValueError('Seed or request count changed')
    flash = Path(config['flash_campaign'])
    baseline = Path(config['baseline'])
    if config.get('prompt_bound_sentence') != SENTENCE:
        raise ValueError('Prompt-bound sentence changed')
    for row in read(out / 'index.json'):
        text = (out / 'prompts' / row['condition'] / (row['base_unit_id'] + '.txt')).read_text()
        request = read(out / row['request_path'])
        if request['contents'][0]['parts'][0]['text'] != text:
            raise ValueError('Prompt and request diverged')
        if request['generationConfig']['seed'] != 42 or request['generationConfig']['temperature'] != 0:
            raise ValueError('Generation config changed')
        if request['generationConfig']['thinkingConfig']['thinkingLevel'] != level:
            raise ValueError('Thinking level changed')
        if request['generationConfig']['maxOutputTokens'] != MAX_OUTPUT_TOKENS:
            raise ValueError('HIGH output limit is not 65536')
        if request['generationConfig']['thinkingConfig'].get('includeThoughts') is not True:
            raise ValueError('includeThoughts is not set')
        original = unbind_prompt(text)
        if row['condition'] == SEQUENCE:
            source_text = read(baseline / 'sequence_requests' / (row['base_unit_id'] + '.json'))['contents'][0]['parts'][0]['text']
        elif row['condition'] == MSA:
            source_text = (baseline / 'prompts/msa_prompt' / (row['base_unit_id'] + '.txt')).read_text()
        else:
            source_text = (flash / 'prompts' / row['condition'] / (row['base_unit_id'] + '.txt')).read_text()
        if original != source_text:
            raise ValueError('Prompt differs from the frozen text by more than the bound sentence')
        if len(row['expected_ids']) != 10:
            raise ValueError('CR25 chunk changed')
    write(out / 'validation.json', dict(status='passed', new_prompts=26, reused_exact_prompts=14,
                                      model=MODEL, thinking_level=level, labels_read=False))
    print('Validation passed: 26 Gemini 3.8 Flash ' + level + ' prompts', flush=True)


def copy_reused(out):
    for unit in read(out / 'index.json'):
        if not unit['reused_exact_prompt']:
            continue
        source = read(out / 'responses' / (unit['score_source_unit'] + '.json'))
        if source.get('status') != 'ok':
            continue
        write(out / 'responses' / (unit['id'] + '.json'),
              dict(unit_id=unit['id'], request_sha256=unit['request_sha256'], status='ok',
                   scores=source['scores'], model=source['model'], reused_exact_prompt=True,
                   score_source_unit=unit['score_source_unit']))


async def run(args):
    import httpx
    out = args.output
    validate(args)
    config = read(out / 'config.json')
    proxy = config['protocol'].get('http_proxy')
    parsed = urlsplit(proxy)
    if parsed.hostname not in ('127.0.0.1', 'localhost') or parsed.username or parsed.password:
        raise ValueError('Only the repository default local proxy without embedded credentials is allowed')
    common = legacy_module(Path(read(Path(config['baseline']) / 'config.json')['source_base']))
    credential = credential_value(args.credential_file)
    stop = asyncio.Event()
    semaphore = asyncio.Semaphore(config['concurrency'])
    deadline = max(3600, config['total_deadline_seconds'])

    async def score(unit, client):
        if unit['reused_exact_prompt'] or stop.is_set():
            return
        response_path = out / 'responses' / (unit['id'] + '.json')
        if response_path.exists() and read(response_path).get('status') == 'ok':
            return
        async with semaphore:
            if stop.is_set():
                return
            result = {'status': 'transport_error'}
            http_status = None
            for number in range(1, config['max_attempts'] + 1):
                if stop.is_set():
                    return
                started = time.time()
                attempt_path = out / 'attempts' / unit['id'] / f'{number:03d}.json'
                attempt_path.parent.mkdir(parents=True, exist_ok=True)
                body = {}
                try:
                    async with asyncio.timeout(deadline):
                        response = await client.post(config['protocol']['endpoint'],
                                                     headers={'x-goog-api-key': credential},
                                                     json=read(out / unit['request_path']))
                        http_status = response.status_code
                        body = response.json()
                        if not isinstance(body, dict):
                            body = {'unexpected_body': body}
                        result = common.parse(body, unit['expected_ids']) if http_status == 200 else {'status': 'http_error'}
                        if result.get('status') == 'ok' and not str(body.get('modelVersion', '')).startswith(MODEL):
                            result = {'status': 'model_identity_mismatch'}
                except Exception as error:
                    result = {'status': 'transport_error', 'error_type': type(error).__name__}
                record = dict(unit_id=unit['id'], request_sha256=unit['request_sha256'], attempt=number,
                              status=result['status'], started=started, finished=time.time(), http_status=http_status,
                              model=body.get('modelVersion') if isinstance(body, dict) else None,
                              usage=body.get('usageMetadata') if isinstance(body, dict) else None,
                              body=body, parsed=result)
                record = json.loads(json.dumps(record, ensure_ascii=False).replace(credential, '[REDACTED]'))
                write(attempt_path, record)
                print(json.dumps(dict(unit_id=unit['id'], thinking=config['thinking_level'], attempt=number,
                                      status=result['status'], http_status=http_status)), flush=True)
                if result['status'] == 'ok':
                    write(response_path, dict(unit_id=unit['id'], request_sha256=unit['request_sha256'],
                                              status='ok', scores=result['scores'], model=body.get('modelVersion'),
                                              reused_exact_prompt=False, raw_attempt=str(attempt_path),
                                              raw_attempt_sha256=sha(attempt_path)))
                    return
                if http_status == 429:
                    await asyncio.sleep(min(30, 5 * number))
                    continue
                if result['status'] in ('explicit_refusal', 'provider_blocked', 'model_identity_mismatch') or http_status in (401, 403):
                    write(response_path, dict(unit_id=unit['id'], request_sha256=unit['request_sha256'],
                                              status=result['status'], http_status=http_status,
                                              raw_attempt=str(attempt_path)))
                    stop.set()
                    return
            write(response_path, dict(unit_id=unit['id'], request_sha256=unit['request_sha256'],
                                      status=result['status'], http_status=http_status))

    async with httpx.AsyncClient(timeout=httpx.Timeout(deadline, connect=30), trust_env=False, proxy=proxy) as client:
        await asyncio.gather(*(score(unit, client) for unit in read(out / 'index.json')))
    if stop.is_set():
        raise SystemExit('Scoring stopped after an explicit refusal, provider block, or authentication failure')
    copy_reused(out)


def evaluate(args):
    from scipy.stats import spearmanr
    out = args.output
    config = read(out / 'config.json')
    if sha(Path(__file__)) != config['runner_sha256']:
        raise ValueError('Runner differs from the frozen thinking campaign')
    baseline = Path(config['baseline'])
    flash = Path(config['flash_campaign'])
    labels = {(row['group_id'], row['variant_id']): row['experimental_score_mean']
              for row in read(Path(read(baseline / 'config.json')['source_base']) / 'offline_labels.json')}
    flash_rows = {(row['group_id'], row['method']): float(row['spearman'])
                  for row in csv.DictReader((flash / 'per_assay.csv').open())}
    by_assay = {}
    for unit in read(out / 'index.json'):
        bundle = by_assay.setdefault(unit['group_id'], dict(unit))
        path = out / 'responses' / (unit['id'] + '.json')
        bundle[unit['condition']] = read(path) if path.exists() else {'status': 'missing'}
    metrics = []
    for assay in read(out / 'cohort.json'):
        group = assay['group_id']
        ids = assay['expected_ids']
        bundle = by_assay[group]
        msa = read(baseline / 'features' / (group + '.json'))['independent_msa_scores']

        def scores_of(condition):
            response = bundle[condition]
            values = response.get('scores') if response.get('status') == 'ok' else None
            return values if values and set(values) == set(ids) else None

        sequence_scores = scores_of(SEQUENCE)
        structure_scores = scores_of(STRUCTURE)
        msa_scores = scores_of(MSA)
        joint_scores = scores_of(JOINT)
        seq_fusion = None
        structure_fusion = None
        if sequence_scores is not None:
            seq_fusion = {row['candidate_id']: row['fusion_value']
                          for row in fuse_rankings(ids, sequence_scores, {i: msa[i] for i in ids})}
        if structure_scores is not None:
            structure_fusion = {row['candidate_id']: row['fusion_value']
                                for row in fuse_rankings(ids, structure_scores, {i: msa[i] for i in ids})}
        local = {SEQUENCE: sequence_scores, FUSION_SEQ: seq_fusion, MSA: msa_scores,
                 STRUCTURE: structure_scores, JOINT: joint_scores, FUSION: structure_fusion}
        for method, values in local.items():
            rho = None
            if values is not None:
                xs = [values[i] for i in ids]
                ys = [labels[group, i] for i in ids]
                if len(set(xs)) > 1 and len(set(ys)) > 1:
                    rho = float(spearmanr(xs, ys).statistic)
            medium = flash_rows[group, method]
            metrics.append(dict(group_id=group, reference_complex_id=assay['reference_complex_id'],
                                method=method, thinking_level=config['thinking_level'], N=10, CR=25,
                                status='complete' if rho is not None else 'missing_score',
                                spearman=rho, medium_spearman=medium,
                                delta=None if rho is None else rho - medium))
    means = {}
    for method in METHODS:
        values = [row['spearman'] for row in metrics if row['method'] == method and row['spearman'] is not None]
        mediums = [row['medium_spearman'] for row in metrics if row['method'] == method]
        means[method] = dict(level=statistics.mean(values) if len(values) == 10 else None,
                             medium=statistics.mean(mediums))
    attempts = sorted((out / 'attempts').glob('*/*.json'))
    attempt_rows = [read(path) for path in attempts]
    summary = dict(model=MODEL, thinking_level=config['thinking_level'],
                   compared_with='gemini-3.8-flash thinking MEDIUM',
                   prompt_bound_sentence=config.get('prompt_bound_sentence'),
                   assays=10, N=10, CR=25,
                   mean_signed_spearman=means, new_api_attempts=len(attempt_rows),
                   successful_new_calls=sum(row.get('status') == 'ok' for row in attempt_rows),
                   unfinished=[row for row in metrics if row['status'] != 'complete'])
    write(out / 'summary.json', summary)
    with (out / 'per_assay.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics[0]))
        writer.writeheader()
        writer.writerows(metrics)
    print(json.dumps(means, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'validate', 'run', 'evaluate'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--thinking', choices=LEVELS)
    parser.add_argument('--flash', type=Path, default=FLASH)
    parser.add_argument('--baseline', type=Path,
                        default=Path('/data/huangyanjie/EVAgent/experiments/protein_rna_ablation_N10_CR25_20261003_msa_v2_native_auth'))
    parser.add_argument('--credential-file', type=Path)
    parser.add_argument('--proxy', default='http://127.0.0.1:17892')
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.flash = args.flash.resolve()
    args.baseline = args.baseline.resolve()
    if args.command == 'prepare' and args.thinking is None:
        parser.error('prepare requires --thinking')
    if args.command == 'run':
        if args.credential_file is None:
            parser.error('run requires --credential-file')
        asyncio.run(run(args))
    else:
        globals()[args.command](args)


if __name__ == '__main__':
    main()
