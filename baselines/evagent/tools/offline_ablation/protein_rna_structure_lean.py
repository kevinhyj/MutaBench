"""Lean protein–RNA structure prompt: omit unmatched interfaces entirely.

Eligible assays append four wild-type RNA-interface fields to the frozen MSA
prompt. Unmatched assays keep that MSA prompt byte-for-byte and reuse its score.
N=10, chunk ranking CR=25. Labels are read only by evaluate().
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

CODE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE / 'tools/offline_ablation'))
from protein_rna_v2 import credential_value, legacy_module  # noqa: E402
from evagent.offline_ablation import base_sites  # noqa: E402

CONDITION = 'msa_prompt_structure_lean'
ELIGIBLE = {'2ERR', '4BS2', '4CIO'}
MOIETIES = ('base', 'ribose', 'phosphate')
GUIDANCE = (
    'Assay-matched WT RNA interface:\n'
    'These values describe the frozen wild-type complex against the RNA used in this assay. '
    'min_partner_heavy_distance_A is angstroms. delta_sasa_partner_A2 is the site surface, '
    'in square angstroms, covered by that RNA. partner_contact_count_5A counts distinct RNA '
    'nucleotides with any heavy atom within 5 angstroms. Zero is a real count on this RNA. '
    'Absence of direct contact does not by itself mean a neutral mutation effect. '
    'No mutant model is supplied.\n')


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


def lean_row(site):
    """Four measured fields. Moieties appear only for a real positive contact."""
    if site['interface_geometry_status'] != 'computed' or site['sasa_status'] != 'computed':
        raise ValueError('Eligible site is missing computed RNA geometry')
    distance = site['min_partner_heavy_distance_A']
    count = site['partner_contact_count_5A']
    buried = site['delta_sasa_partner_A2']
    if distance is None or count is None or buried is None:
        raise ValueError('Computed RNA geometry contains null')
    if type(count) is not int or count < 0 or distance < 0 or buried < -1e-6:
        raise ValueError('Computed RNA geometry is not a real nonnegative measurement')
    row = dict(site_id=site['site_id'], min_partner_heavy_distance_A=distance,
               partner_contact_count_5A=count, delta_sasa_partner_A2=buried)
    if count > 0 and site.get('rna_moiety_status') == 'complete':
        moieties = site['rna_contact_count_5A_by_moiety']
        names = [name for name in MOIETIES if moieties[name] > 0]
        if names:
            row['rna_contact_moieties'] = names
    return row


def ordered_site_ids(task, ids):
    _, sites = base_sites(task)
    seen = []
    for cid in ids:
        for site in sites[cid]:
            if site['site_id'] not in seen:
                seen.append(site['site_id'])
    return seen


def structure_block(structure, site_ids):
    by_id = {row['site_id']: row for row in structure['sites']}
    missing = [sid for sid in site_ids if sid not in by_id]
    if missing:
        raise ValueError('Candidate site is absent from the structure table')
    return [lean_row(by_id[sid]) for sid in site_ids]


def prompt_text(baseline_text, structure, task, ids, eligible):
    if not eligible:
        return baseline_text
    block = structure_block(structure, ordered_site_ids(task, ids))
    body = json.dumps(block, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
    if 'null' in body:
        raise ValueError('Lean structure block contains null')
    return baseline_text + '\n\n' + GUIDANCE + body


def prepare(args):
    out = args.output
    baseline = args.baseline
    geometry = args.geometry
    out.mkdir(parents=True, exist_ok=True)
    cohort = read(baseline / 'cohort.json')
    if len(cohort) != 10:
        raise ValueError('Expected the frozen 10-assay protein–RNA cohort')
    units = {(unit['group_id'], unit['condition']): unit for unit in read(baseline / 'index.json')}
    index = []
    for assay in cohort:
        group = assay['group_id']
        eligible = assay['reference_complex_id'] in ELIGIBLE
        mapping = read(geometry / 'structure_mapping' / (group + '.json'))
        if mapping['scope_eligible'] != eligible:
            raise ValueError('Eligibility disagrees with the frozen structure review: ' + group)
        structure = read(geometry / 'structure_features' / (group + '.json'))
        task = read(baseline / assay['task_path'])
        source = units[group, 'msa_prompt']
        source_request = baseline / source['request_path']
        source_prompt = (baseline / 'prompts/msa_prompt' / (assay['id'] + '.txt')).read_text()
        request = read(source_request)
        if request['contents'][0]['parts'][0]['text'] != source_prompt:
            raise ValueError('Baseline MSA prompt and request text differ')
        text = prompt_text(source_prompt, structure, task, assay['expected_ids'], eligible)
        request = copy.deepcopy(request)
        request['contents'][0]['parts'][0]['text'] = text
        request_path = out / 'requests' / CONDITION / (assay['id'] + '.json')
        prompt_path = out / 'prompts' / CONDITION / (assay['id'] + '.txt')
        request_path.parent.mkdir(parents=True, exist_ok=True)
        prompt_path.parent.mkdir(parents=True, exist_ok=True)
        encoded = json.dumps(request, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
        if request_path.exists() and request_path.read_text() != encoded:
            raise ValueError('Frozen lean request changed')
        if not request_path.exists():
            request_path.write_text(encoded)
        if prompt_path.exists() and prompt_path.read_text() != text:
            raise ValueError('Frozen lean prompt changed')
        if not prompt_path.exists():
            prompt_path.write_text(text)
        if not eligible and text != source_prompt:
            raise ValueError('Unmatched assay prompt differs from the reused MSA prompt')
        record = dict(assay, id=CONDITION + '__' + assay['id'], base_unit_id=assay['id'],
                      condition=CONDITION, request_path=str(request_path.relative_to(out)),
                      request_sha256=sha(request_path), structure_included=eligible,
                      reused_exact_msa_prompt=not eligible,
                      source_response=str(baseline / 'responses' / (source['id'] + '.json')) if not eligible else None)
        index.append(record)
        if not eligible:
            source_response = read(record['source_response'])
            if source_response['status'] != 'ok' or source_response['request_sha256'] != source['request_sha256']:
                raise ValueError('Reusable MSA response does not match its prompt')
            reuse = dict(unit_id=record['id'], request_sha256=record['request_sha256'], status='ok',
                         scores=source_response['scores'], model=source_response['model'],
                         reused_exact_msa_prompt=True, source_response=record['source_response'],
                         source_response_sha256=sha(record['source_response']),
                         source_request_sha256=source['request_sha256'])
            write(out / 'responses' / (record['id'] + '.json'), reuse)
    config = read(baseline / 'config.json')
    save_config = dict(schema='protein-rna-structure-lean-v1', baseline=str(baseline), geometry=str(geometry),
                       N=10, CR=25, protocol=config['protocol'], eligible_complexes=sorted(ELIGIBLE),
                       fields=['min_partner_heavy_distance_A', 'partner_contact_count_5A',
                               'delta_sasa_partner_A2', 'rna_contact_moieties'],
                       omitted_when_unmatched=True, new_request_count=3, reused_exact_msa_prompts=7,
                       max_attempts=3, concurrency=1, created_unix=time.time(),
                       runner_sha256=sha(Path(__file__)))
    write(out / 'config.json', save_config)
    write(out / 'cohort.json', cohort)
    write(out / 'index.json', index)
    print(json.dumps(dict(eligible=3, reused=7, new_requests=3)), flush=True)


def validate(args):
    out = args.output
    config = read(out / 'config.json')
    if sha(Path(__file__)) != config['runner_sha256']:
        raise ValueError('Runner differs from the frozen lean campaign')
    if config['N'] != 10 or config['CR'] != 25 or config['new_request_count'] != 3:
        raise ValueError('Lean campaign N/CR/request count changed')
    index = read(out / 'index.json')
    if len(index) != 10 or sum(row['structure_included'] for row in index) != 3:
        raise ValueError('Expected structure text in exactly three assays')
    baseline = Path(config['baseline'])
    for row in index:
        text = (out / 'prompts' / CONDITION / (row['base_unit_id'] + '.txt')).read_text()
        request = read(out / row['request_path'])
        if request['contents'][0]['parts'][0]['text'] != text or sha(out / row['request_path']) != row['request_sha256']:
            raise ValueError('Prompt and request diverged')
        source_prompt = (baseline / 'prompts/msa_prompt' / (row['base_unit_id'] + '.txt')).read_text()
        if row['reused_exact_msa_prompt']:
            if text != source_prompt or 'Assay-matched WT RNA interface' in text:
                raise ValueError('Unmatched assay still contains a structure block')
        else:
            if not text.startswith(source_prompt + '\n\n' + GUIDANCE):
                raise ValueError('Eligible prompt does not preserve the MSA prompt')
            block = text.split(GUIDANCE, 1)[1]
            if 'null' in block or 'unavailable' in block:
                raise ValueError('Eligible structure block contains null or an unavailable marker')
            rows = json.loads(block)
            if not rows or any(set(item) - {'site_id', 'min_partner_heavy_distance_A', 'partner_contact_count_5A',
                                             'delta_sasa_partner_A2', 'rna_contact_moieties'} for item in rows):
                raise ValueError('Lean field set changed')
            for item in rows:
                if item['partner_contact_count_5A'] == 0 and 'rna_contact_moieties' in item:
                    raise ValueError('A real zero contact invented a moiety list')
        if len(row['expected_ids']) != 10 or row['batch'] != 0:
            raise ValueError('CR25 chunk changed')
    write(out / 'validation.json', dict(status='passed', assays=10, new_prompts=3, reused_exact_msa_prompts=7,
                                      CR=25, labels_read=False))
    print('Validation passed: 3 lean structure prompts, 7 exact MSA-prompt reuses', flush=True)


async def run(args):
    import httpx
    out = args.output
    validate(args)
    config = read(out / 'config.json')
    common = legacy_module(Path(read(Path(config['baseline']) / 'config.json')['source_base']))
    credential = credential_value(args.credential_file)
    proxy = config['protocol'].get('http_proxy')
    async with httpx.AsyncClient(timeout=httpx.Timeout(max(3600, config['protocol']['timeout_seconds']), connect=30),
                                 trust_env=False, proxy=proxy) as client:
        for unit in read(out / 'index.json'):
            if unit['reused_exact_msa_prompt']:
                continue
            response_path = out / 'responses' / (unit['id'] + '.json')
            if response_path.exists() and read(response_path).get('status') == 'ok':
                continue
            attempt_dir = out / 'attempts' / unit['id']
            for number in range(1, config['max_attempts'] + 1):
                started = time.time()
                attempt_path = attempt_dir / f'{number:03d}.json'
                attempt_path.parent.mkdir(parents=True, exist_ok=True)
                body, http_status, result = {}, None, {'status': 'transport_error'}
                try:
                    async with asyncio.timeout(max(3600, config['protocol']['total_deadline_seconds'])):
                        response = await client.post(config['protocol']['endpoint'],
                                                     headers={'x-goog-api-key': credential},
                                                     json=read(out / unit['request_path']))
                        http_status = response.status_code
                        body = response.json()
                        if not isinstance(body, dict):
                            body = {'unexpected_body': body}
                        result = common.parse(body, unit['expected_ids']) if http_status == 200 else {'status': 'http_error'}
                        if result.get('status') == 'ok' and not str(body.get('modelVersion', '')).startswith(config['protocol']['requested_model']):
                            result = {'status': 'model_identity_mismatch'}
                except Exception as error:
                    result = {'status': 'transport_error', 'error_type': type(error).__name__}
                record = dict(unit_id=unit['id'], request_sha256=unit['request_sha256'], attempt=number,
                              status=result['status'], started=started, finished=time.time(),
                              http_status=http_status, model=body.get('modelVersion') if isinstance(body, dict) else None,
                              usage=body.get('usageMetadata') if isinstance(body, dict) else None,
                              body=body, parsed=result)
                record = json.loads(json.dumps(record, ensure_ascii=False).replace(credential, '[REDACTED]'))
                write(attempt_path, record)
                print(json.dumps(dict(unit_id=unit['id'], attempt=number, status=result['status'],
                                      http_status=http_status)), flush=True)
                if result['status'] == 'ok':
                    write(response_path, dict(unit_id=unit['id'], request_sha256=unit['request_sha256'],
                                              status='ok', scores=result['scores'], model=body.get('modelVersion'),
                                              reused_exact_msa_prompt=False, raw_attempt=str(attempt_path),
                                              raw_attempt_sha256=sha(attempt_path)))
                    break
                if result['status'] in ('explicit_refusal', 'provider_blocked', 'model_identity_mismatch'):
                    write(response_path, dict(unit_id=unit['id'], request_sha256=unit['request_sha256'],
                                              status=result['status'], raw_attempt=str(attempt_path)))
                    return
    validate(args)


def evaluate(args):
    from scipy.stats import spearmanr
    validate(args)
    out = args.output
    config = read(out / 'config.json')
    baseline = Path(config['baseline'])
    labels = {(row['group_id'], row['variant_id']): row
              for row in read(Path(read(baseline / 'config.json')['source_base']) / 'offline_labels.json')}
    previous = {(row['group_id'], row['method']): row for row in csv.DictReader((baseline / 'per_assay.csv').open())}
    metrics = []
    for unit in read(out / 'index.json'):
        response = read(out / 'responses' / (unit['id'] + '.json'))
        values = response.get('scores') if response.get('status') == 'ok' else None
        ids = unit['expected_ids']
        rho = None
        if values and set(values) == set(ids):
            xs = [values[cid] for cid in ids]
            ys = [labels[unit['group_id'], cid]['experimental_score_mean'] for cid in ids]
            if len(set(xs)) > 1 and len(set(ys)) > 1:
                rho = float(spearmanr(xs, ys).statistic)
        metrics.append(dict(group_id=unit['group_id'], reference_complex_id=unit['reference_complex_id'],
                            method=CONDITION, structure_included=unit['structure_included'],
                            reused_exact_msa_prompt=unit['reused_exact_msa_prompt'], N=10, CR=25,
                            status='complete' if rho is not None or values else response.get('status'),
                            spearman=rho,
                            msa_prompt_spearman=float(previous[unit['group_id'], 'msa_prompt']['spearman']),
                            old_structure_spearman=float(previous[unit['group_id'], 'msa_prompt_structure']['spearman'])))
    mean = statistics.mean(row['spearman'] for row in metrics if row['spearman'] is not None)
    attempts = sorted((out / 'attempts').glob('*/*.json')) if (out / 'attempts').exists() else []
    attempt_rows = [read(path) for path in attempts]
    write(out / 'summary.json', dict(assays=10, mean_signed_spearman=mean, N=10, CR=25,
                                    new_api_attempts=len(attempt_rows),
                                    successful_new_calls=sum(row.get('status') == 'ok' for row in attempt_rows),
                                    reused_exact_msa_prompts=sum(row['reused_exact_msa_prompt'] for row in metrics)))
    with (out / 'per_assay.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics[0]))
        writer.writeheader()
        writer.writerows(metrics)
    print(json.dumps(dict(mean=mean, rows=len(metrics))), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'validate', 'run', 'evaluate'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--baseline', type=Path,
                        default=Path('/data/huangyanjie/EVAgent/experiments/protein_rna_ablation_N10_CR25_20261003_msa_v2_native_auth'))
    parser.add_argument('--geometry', type=Path,
                        default=Path('/data/huangyanjie/EVAgent/experiments/protein_rna_ablation_N10_CR25_20261004_structure_v3'))
    parser.add_argument('--credential-file', type=Path)
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.baseline = args.baseline.resolve()
    args.geometry = args.geometry.resolve()
    if args.command == 'run':
        if args.credential_file is None:
            parser.error('run requires --credential-file')
        asyncio.run(run(args))
    else:
        globals()[args.command](args)


if __name__ == '__main__':
    main()
