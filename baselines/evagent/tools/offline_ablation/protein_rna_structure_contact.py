"""Assay-matched RNA contact flags on top of the frozen sequence prompt.

Only 2ERR, 4BS2 and 4CIO append a wild-type contact flag and, when a contact
exists, the nucleotide moieties involved. Distances, contact counts and buried
areas stay in the selection audit. Other assays keep the sequence or MSA prompt
byte-for-byte. N=10, CR=25. Labels are read only by evaluate().
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
from protein_rna_structure_lean import ordered_site_ids  # noqa: E402
from protein_rna_v2 import credential_value, legacy_module  # noqa: E402
from evagent.offline_ablation import fuse_rankings  # noqa: E402

STRUCTURE = 'structure_contact'
JOINT = 'msa_prompt_structure_contact'
FUSION = 'msa_gemini_rank5050_structure_contact'
ELIGIBLE = {'2ERR', '4BS2', '4CIO'}
MOIETIES = ('base', 'ribose', 'phosphate')
GUIDANCE = (
    'Assay-matched WT RNA contact:\n'
    'direct_rna_contact is true when any heavy atom of the wild-type side chain lies within 5 angstroms '
    'of the RNA used in this assay, and false when none does. '
    'rna_contact_moieties lists only nucleotide parts that participate in such a contact: base, ribose, or phosphate. '
    'These fields describe the wild-type residue against that RNA. They are not the effect of the substitution. '
    'No mutant model is supplied.\n')
FORBIDDEN = ('null', 'unavailable', 'min_partner_heavy_distance_A', 'delta_sasa_partner_A2', 'partner_contact_count_5A')


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


def contact_row(site):
    """Boolean contact plus moieties. No distance, count or area."""
    if site.get('interface_geometry_status') != 'computed':
        raise ValueError('Assay-matched site is missing computed RNA geometry')
    count = site['partner_contact_count_5A']
    if type(count) is not int or count < 0 or site['min_partner_heavy_distance_A'] is None:
        raise ValueError('Assay-matched contact count is not a real measurement')
    row = dict(site_id=site['site_id'], direct_rna_contact=count > 0)
    if count > 0 and site.get('rna_moiety_status') == 'complete':
        names = [name for name in MOIETIES if site['rna_contact_count_5A_by_moiety'][name] > 0]
        if names:
            row['rna_contact_moieties'] = names
    return row


def contact_rows(structure_sites, site_ids):
    by_id = {row['site_id']: row for row in structure_sites}
    missing = [sid for sid in site_ids if sid not in by_id]
    if missing:
        raise ValueError('Candidate site is absent from the structure table')
    return [contact_row(by_id[sid]) for sid in site_ids]


def render(base_text, rows):
    if not rows:
        return base_text
    body = json.dumps(rows, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
    if any(token in body for token in FORBIDDEN):
        raise ValueError('Contact block contains a forbidden field or marker')
    return base_text + '\n\n' + GUIDANCE + body


def prepare(args):
    out = args.output
    baseline = args.baseline
    geometry = args.geometry
    if out.exists() and any(out.iterdir()):
        raise ValueError('Output directory is not empty; refuse to mix campaigns')
    out.mkdir(parents=True, exist_ok=True)
    cohort = read(baseline / 'cohort.json')
    if len(cohort) != 10 or {row['reference_complex_id'] for row in cohort} != ELIGIBLE | {
            '4I67', '5EN1', '5HO4', '2ZZN', '4YVI', '1B23'}:
        raise ValueError('Expected the frozen 10-assay protein–RNA cohort')
    units = {(unit['group_id'], unit['condition']): unit for unit in read(baseline / 'index.json')}
    index = []
    audits = []
    for assay in cohort:
        group = assay['group_id']
        complex_id = assay['reference_complex_id']
        eligible = complex_id in ELIGIBLE
        mapping = read(geometry / 'structure_mapping' / (group + '.json'))
        if bool(mapping['scope_eligible']) != eligible:
            raise ValueError('Eligibility disagrees with the frozen structure review: ' + group)
        task = read(baseline / assay['task_path'])
        site_ids = ordered_site_ids(task, assay['expected_ids'])
        structure_sites = read(geometry / 'structure_features' / (group + '.json'))['sites'] if eligible else []
        rows = contact_rows(structure_sites, site_ids) if eligible else []
        msa_unit = units[group, 'msa_prompt']
        msa_text = (baseline / 'prompts/msa_prompt' / (assay['id'] + '.txt')).read_text()
        msa_request = read(baseline / msa_unit['request_path'])
        sequence_request = read(baseline / 'sequence_requests' / (assay['id'] + '.json'))
        if msa_request['contents'][0]['parts'][0]['text'] != msa_text:
            raise ValueError('Baseline MSA prompt and request text differ')
        sequence_text = sequence_request['contents'][0]['parts'][0]['text']
        texts = {STRUCTURE: render(sequence_text, rows), JOINT: render(msa_text, rows)}
        if not eligible and (texts[STRUCTURE] != sequence_text or texts[JOINT] != msa_text):
            raise ValueError('Unmatched assay prompt changed')
        audits.append(dict(group_id=group, reference_complex_id=complex_id, eligible=eligible,
                           sites=rows, measurement_cache=[
                               dict(site_id=row['site_id'], partner_contact_count_5A=row['partner_contact_count_5A'],
                                    min_partner_heavy_distance_A=row['min_partner_heavy_distance_A'],
                                    delta_sasa_partner_A2=row['delta_sasa_partner_A2'])
                               for row in structure_sites] if eligible else []))
        sources = {STRUCTURE: sequence_request, JOINT: msa_request}
        for condition, text in texts.items():
            request = copy.deepcopy(sources[condition])
            request['contents'][0]['parts'][0]['text'] = text
            request_path = out / 'requests' / condition / (assay['id'] + '.json')
            prompt_path = out / 'prompts' / condition / (assay['id'] + '.txt')
            request_path.parent.mkdir(parents=True, exist_ok=True)
            prompt_path.parent.mkdir(parents=True, exist_ok=True)
            request_path.write_text(json.dumps(request, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
            prompt_path.write_text(text)
            record = dict(assay, id=condition + '__' + assay['id'], base_unit_id=assay['id'],
                          condition=condition, eligible=eligible,
                          request_path=str(request_path.relative_to(out)), request_sha256=sha(request_path),
                          reused_exact_prompt=not eligible)
            if not eligible:
                if condition == STRUCTURE:
                    source = baseline / 'sequence_scores' / (assay['id'] + '.json')
                    payload = read(source)
                    scores, model = payload['scores'], payload['provenance']['model']
                else:
                    source = baseline / 'responses' / (msa_unit['id'] + '.json')
                    payload = read(source)
                    if payload['status'] != 'ok':
                        raise ValueError('Reusable MSA response is not ok')
                    scores, model = payload['scores'], payload['model']
                record['source_response'] = str(source)
                write(out / 'responses' / (record['id'] + '.json'),
                      dict(unit_id=record['id'], request_sha256=record['request_sha256'], status='ok',
                           scores=scores, model=model, reused_exact_prompt=True, source_response=str(source),
                           source_response_sha256=sha(source)))
            index.append(record)
    protocol = read(baseline / 'config.json')['protocol']
    save_config = dict(schema='protein-rna-structure-contact-v1', baseline=str(baseline), geometry=str(geometry),
                       N=10, CR=25, protocol=protocol, eligible_complexes=sorted(ELIGIBLE),
                       prompt_fields=['direct_rna_contact', 'rna_contact_moieties'],
                       audit_only_fields=['min_partner_heavy_distance_A', 'partner_contact_count_5A',
                                          'delta_sasa_partner_A2'],
                       new_request_count=sum(not row['reused_exact_prompt'] for row in index),
                       reused_exact_prompts=sum(row['reused_exact_prompt'] for row in index),
                       max_attempts=3, concurrency=3, created_unix=time.time(), runner_sha256=sha(Path(__file__)))
    write(out / 'config.json', save_config)
    write(out / 'cohort.json', cohort)
    write(out / 'index.json', index)
    write(out / 'selection.json', audits)
    print(json.dumps(dict(new_requests=save_config['new_request_count'],
                          reused=save_config['reused_exact_prompts'])), flush=True)


def _body(text):
    marker = '\n\n' + GUIDANCE
    if marker not in text:
        raise ValueError('Eligible prompt is missing contact guidance')
    return text.split(marker, 1)[1]


def validate(args):
    out = args.output
    config = read(out / 'config.json')
    if sha(Path(__file__)) != config['runner_sha256']:
        raise ValueError('Runner differs from the frozen contact campaign')
    if config['N'] != 10 or config['CR'] != 25 or config['new_request_count'] != 6:
        raise ValueError('N, CR or new-request count changed')
    index = read(out / 'index.json')
    if len(index) != 20 or sum(row['eligible'] for row in index) != 6:
        raise ValueError('Expected structure text on exactly three assays and two arms')
    baseline = Path(config['baseline'])
    for row in index:
        text = (out / 'prompts' / row['condition'] / (row['base_unit_id'] + '.txt')).read_text()
        request = read(out / row['request_path'])
        if request['contents'][0]['parts'][0]['text'] != text or sha(out / row['request_path']) != row['request_sha256']:
            raise ValueError('Prompt and request diverged')
        if request['generationConfig']['seed'] != 42 or request['generationConfig']['temperature'] != 0:
            raise ValueError('Generation protocol changed')
        if row['reused_exact_prompt']:
            source_name = 'sequence_requests' if row['condition'] == STRUCTURE else 'prompts/msa_prompt'
            if row['condition'] == STRUCTURE:
                source_text = read(baseline / 'sequence_requests' / (row['base_unit_id'] + '.json'))['contents'][0]['parts'][0]['text']
            else:
                source_text = (baseline / 'prompts/msa_prompt' / (row['base_unit_id'] + '.txt')).read_text()
            if text != source_text or 'direct_rna_contact' in text:
                raise ValueError('Unmatched prompt still contains a contact block')
            del source_name
        else:
            body = _body(text)
            if any(token in body for token in FORBIDDEN):
                raise ValueError('Contact block contains a distance, area, count, null, or unavailable marker')
            parsed = json.loads(body)
            if not parsed or any(item['direct_rna_contact'] is False and set(item) != {'site_id', 'direct_rna_contact'}
                                 for item in parsed):
                raise ValueError('A negative contact invented extra fields')
            if any(item['direct_rna_contact'] is True and set(item) - {'site_id', 'direct_rna_contact', 'rna_contact_moieties'}
                   for item in parsed):
                raise ValueError('A positive contact contains an unexpected field')
        if len(row['expected_ids']) != 10 or row['batch'] != 0:
            raise ValueError('CR25 chunk changed')
    write(out / 'validation.json', dict(status='passed', assays=10, new_prompts=6, reused_exact_prompts=14,
                                      CR=25, labels_read=False))
    print('Validation passed: 6 contact prompts, 14 exact reuses', flush=True)


async def run(args):
    import httpx
    out = args.output
    validate(args)
    config = read(out / 'config.json')
    proxy = config['protocol'].get('http_proxy')
    if proxy:
        parsed = urlsplit(proxy)
        if parsed.hostname not in ('127.0.0.1', 'localhost') or parsed.username or parsed.password:
            raise ValueError('Only the repository default local proxy without embedded credentials is allowed')
    common = legacy_module(Path(read(Path(config['baseline']) / 'config.json')['source_base']))
    credential = credential_value(args.credential_file)
    stop = asyncio.Event()
    semaphore = asyncio.Semaphore(config['concurrency'])

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
                              status=result['status'], started=started, finished=time.time(), http_status=http_status,
                              model=body.get('modelVersion') if isinstance(body, dict) else None,
                              usage=body.get('usageMetadata') if isinstance(body, dict) else None,
                              body=body, parsed=result)
                record = json.loads(json.dumps(record, ensure_ascii=False).replace(credential, '[REDACTED]'))
                write(attempt_path, record)
                print(json.dumps(dict(unit_id=unit['id'], attempt=number, status=result['status'],
                                      http_status=http_status)), flush=True)
                if result['status'] == 'ok':
                    write(response_path, dict(unit_id=unit['id'], request_sha256=unit['request_sha256'],
                                              status='ok', scores=result['scores'], model=body.get('modelVersion'),
                                              reused_exact_prompt=False, raw_attempt=str(attempt_path),
                                              raw_attempt_sha256=sha(attempt_path)))
                    return
                if result['status'] in ('explicit_refusal', 'provider_blocked', 'model_identity_mismatch') or http_status in (401, 403):
                    write(response_path, dict(unit_id=unit['id'], request_sha256=unit['request_sha256'],
                                              status=result['status'], http_status=http_status,
                                              raw_attempt=str(attempt_path)))
                    stop.set()
                    return
            write(response_path, dict(unit_id=unit['id'], request_sha256=unit['request_sha256'],
                                      status=result['status'], http_status=http_status))

    async with httpx.AsyncClient(timeout=httpx.Timeout(max(3600, config['protocol']['timeout_seconds']), connect=30),
                                 trust_env=False, proxy=proxy) as client:
        await asyncio.gather(*(score(unit, client) for unit in read(out / 'index.json')))
    if stop.is_set():
        raise SystemExit('Scoring stopped after an explicit refusal, provider block, or authentication failure')


def evaluate(args):
    from scipy.stats import spearmanr
    out = args.output
    config = read(out / 'config.json')
    if sha(Path(__file__)) != config['runner_sha256']:
        raise ValueError('Runner differs from the frozen contact campaign')
    baseline = Path(config['baseline'])
    labels = {(row['group_id'], row['variant_id']): row
              for row in read(Path(read(baseline / 'config.json')['source_base']) / 'offline_labels.json')}
    previous = {(row['group_id'], row['method']): float(row['spearman'])
                for row in csv.DictReader((baseline / 'per_assay.csv').open())}
    by_assay = {}
    for unit in read(out / 'index.json'):
        bundle = by_assay.setdefault(unit['group_id'], dict(unit))
        response_path = out / 'responses' / (unit['id'] + '.json')
        bundle[unit['condition']] = read(response_path) if response_path.exists() else {'status': 'missing'}
    metrics = []
    for assay in read(out / 'cohort.json'):
        group = assay['group_id']
        ids = assay['expected_ids']
        bundle = by_assay[group]
        msa = read(baseline / 'features' / (group + '.json'))['independent_msa_scores']
        scores = {}
        for condition in (STRUCTURE, JOINT):
            response = bundle[condition]
            values = response.get('scores') if response.get('status') == 'ok' else None
            scores[condition] = values if values and set(values) == set(ids) else None
        if scores[STRUCTURE] is not None:
            fused = fuse_rankings(ids, scores[STRUCTURE], {cid: msa[cid] for cid in ids})
            scores[FUSION] = {row['candidate_id']: row['fusion_value'] for row in fused}
        else:
            scores[FUSION] = None
        local_methods = {STRUCTURE: scores[STRUCTURE], JOINT: scores[JOINT], FUSION: scores[FUSION]}
        reused_methods = {'sequence': previous[group, 'sequence'],
                          'msa_gemini_rank5050': previous[group, 'msa_gemini_rank5050'],
                          'msa_prompt': previous[group, 'msa_prompt']}
        for method, values in {**reused_methods, **local_methods}.items():
            rho = values if method in reused_methods else None
            if method not in reused_methods and values is not None:
                xs = [values[cid] for cid in ids]
                ys = [labels[group, cid]['experimental_score_mean'] for cid in ids]
                rho = float(spearmanr(xs, ys).statistic) if len(set(xs)) > 1 and len(set(ys)) > 1 else None
            metrics.append(dict(group_id=group, reference_complex_id=assay['reference_complex_id'],
                                method=method, eligible=bundle['eligible'], N=10, CR=25,
                                status='complete' if rho is not None else 'missing_score', spearman=rho))
    order = ('sequence', 'msa_gemini_rank5050', 'msa_prompt', STRUCTURE, JOINT, FUSION)
    means = {method: statistics.mean(row['spearman'] for row in metrics if row['method'] == method) for method in order}
    attempts = sorted((out / 'attempts').glob('*/*.json')) if (out / 'attempts').exists() else []
    attempt_rows = [read(path) for path in attempts]
    summary = dict(assays=10, N=10, CR=25, mean_signed_spearman=means,
                   new_api_attempts=len(attempt_rows),
                   successful_new_calls=sum(row.get('status') == 'ok' for row in attempt_rows),
                   reused_exact_prompts=14,
                   unfinished=[row for row in metrics if row['status'] != 'complete' and row['method'] in (STRUCTURE, JOINT, FUSION)])
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
