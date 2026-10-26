"""Local protein–RNA structure prompts for three frozen arms.

Assay-matched complexes keep real contact counts, including zero. Partial crystal
RNA and the corrected 2ZZN assembly contribute only sites with an observed
contact. Related structures contribute a boolean contact, never a distance.
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
from urllib.parse import urlsplit

CODE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE / 'tools/offline_ablation'))
from protein_rna_structure_lean import GUIDANCE as EXACT_GUIDANCE  # noqa: E402
from protein_rna_structure_lean import lean_row, prompt_text as exact_prompt_text, structure_block  # noqa: E402
from protein_rna_v2 import credential_value, legacy_module  # noqa: E402
from evagent.offline_ablation import fuse_rankings, midranks  # noqa: E402
from protein_rna_structure_lean import ordered_site_ids  # noqa: E402
STRUCTURE = 'structure_local'
JOINT = 'msa_prompt_structure_local'
FUSION = 'msa_gemini_rank5050_structure_local'
CONDITIONS = (STRUCTURE, JOINT)
EXACT = {'2ERR', '4BS2', '4CIO'}
OBSERVED = {'4I67', '5EN1', '5HO4'}
PAIRED = {'2ZZN'}
RELATED = {'4YVI', '1B23'}
MOIETIES = ('base', 'ribose', 'phosphate')
LEAN_ROOT = Path('/data/huangyanjie/EVAgent/experiments/protein_rna_ablation_N10_CR25_20261005_structure_lean')
LOCAL_GUIDANCE = (
    'Observed local WT RNA contact:\n'
    'Only mutation sites with a heavy-atom contact to RNA in the available coordinates are listed. '
    'Sites without such a contact are omitted. '
    'min_partner_heavy_distance_A is angstroms. delta_sasa_partner_A2 is square angstroms covered by that observed RNA. '
    'partner_contact_count_5A counts distinct observed RNA nucleotides within 5 angstroms. '
    'The coordinates show the crystallized RNA neighborhood, which may be a fragment or a partially reviewed assembly rather than the complete assay RNA. '
    'A missing site is not a measurement of no contact. '
    'No mutant model is supplied.\n')
RELATED_GUIDANCE = (
    'Related-structure local contact:\n'
    'The listed sites can be placed on a related structure, and that position contacts RNA in the related structure. '
    'related_structure_contact is true for that related structure only. '
    'Distances and buried areas are omitted because this is not the assay complex. '
    'Unlisted sites are not asserted to lack contact. '
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


def tier_of(complex_id):
    if complex_id in EXACT:
        return 'assay_matched'
    if complex_id in OBSERVED:
        return 'observed_contact'
    if complex_id in PAIRED:
        return 'paired_trna_c'
    if complex_id in RELATED:
        return 'related_structure'
    raise ValueError('No structure tier for ' + complex_id)


def guidance_for(tier):
    if tier == 'assay_matched':
        return EXACT_GUIDANCE
    if tier in ('observed_contact', 'paired_trna_c'):
        return LOCAL_GUIDANCE
    if tier == 'related_structure':
        return RELATED_GUIDANCE
    raise ValueError(tier)


def render(base_text, guidance, rows):
    if not rows:
        return base_text
    body = json.dumps(rows, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
    if 'null' in body or 'unavailable' in body:
        raise ValueError('Structure block contains null or an unavailable marker')
    return base_text + '\n\n' + guidance + body


def _positive_observed_row(sid, obs):
    distance = obs['min_partner_heavy_distance_A']
    count = obs['partner_contact_count_5A']
    buried = obs['delta_sasa_partner_A2']
    if distance is None or count is None or buried is None:
        raise ValueError('Observed geometry contains null')
    if count <= 0:
        return None
    site = dict(obs, site_id=sid, interface_geometry_status='computed', sasa_status='computed')
    return lean_row(site)


def _paired_row(sid, obs):
    hits = list(obs.get('rna_contacts') or [])
    paired = [hit for hit in hits if str(hit.get('rna_instance_id', '')).startswith('C@')]
    if not paired:
        return None
    foreign = [hit for hit in hits if not str(hit.get('rna_instance_id', '')).startswith('C@')]
    moieties = {name: 0 for name in MOIETIES}
    for hit in paired:
        for name in hit.get('moieties') or []:
            if name in moieties:
                moieties[name] += 1
    distance = min(float(hit['min_distance_A']) for hit in paired)
    count = len({hit['rna_site_id'] for hit in paired})
    buried = None if foreign else obs['delta_sasa_partner_A2']
    site = dict(site_id=sid, interface_geometry_status='computed', sasa_status='computed',
                min_partner_heavy_distance_A=distance, partner_contact_count_5A=count,
                delta_sasa_partner_A2=0.0 if buried is None else buried,
                rna_moiety_status=obs.get('rna_moiety_status'),
                rna_contact_count_5A_by_moiety=moieties)
    row = lean_row(site)
    if buried is None:
        del row['delta_sasa_partner_A2']
    return row


def _related_row(sid, complex_id, obs):
    count = obs['partner_contact_count_5A']
    if count is None:
        raise ValueError('Related-structure contact count is null')
    if count <= 0:
        return None
    return dict(site_id=sid, structure_id=complex_id, related_structure_contact=True)


def select_rows(complex_id, site_ids, structure_sites, observed):
    """Return prompt rows and the site ids that were left out."""
    tier = tier_of(complex_id)
    if tier == 'assay_matched':
        rows = structure_block({'sites': structure_sites}, site_ids)
        return tier, rows, []
    rows = []
    omitted = []
    for sid in site_ids:
        obs = observed.get(sid)
        if obs is None:
            omitted.append(sid)
            continue
        if tier == 'observed_contact':
            row = _positive_observed_row(sid, obs)
        elif tier == 'paired_trna_c':
            row = _paired_row(sid, obs)
        else:
            row = _related_row(sid, complex_id, obs)
        if row is None:
            omitted.append(sid)
        else:
            rows.append(row)
    return tier, rows, omitted


def prepare(args):
    out = args.output
    baseline = args.baseline
    geometry = args.geometry
    if out.exists() and any(out.iterdir()):
        raise ValueError('Output directory is not empty; refuse to mix campaigns')
    out.mkdir(parents=True, exist_ok=True)
    cohort = read(baseline / 'cohort.json')
    if len(cohort) != 10 or {row['reference_complex_id'] for row in cohort} != EXACT | OBSERVED | PAIRED | RELATED:
        raise ValueError('Expected the frozen 10-assay protein–RNA cohort')
    units = {(unit['group_id'], unit['condition']): unit for unit in read(baseline / 'index.json')}
    index = []
    audits = []
    for assay in cohort:
        group = assay['group_id']
        complex_id = assay['reference_complex_id']
        mapping = read(geometry / 'structure_mapping' / (group + '.json'))
        observed = mapping['observed_scope_geometry']
        if (complex_id in EXACT) != bool(mapping['scope_eligible']):
            raise ValueError('Eligibility disagrees with the frozen structure review: ' + group)
        if complex_id == '5HO4':
            instances = {hit.get('rna_instance_id') for row in observed.values() for hit in row.get('rna_contacts') or []}
            if 'B@SMTRY4_cell_-1_-1_0' not in instances:
                raise ValueError('5HO4 observed geometry does not include the literature symmetry RNA')
        structure_sites = read(geometry / 'structure_features' / (group + '.json'))['sites'] if complex_id in EXACT else []
        task = read(baseline / assay['task_path'])
        site_ids = ordered_site_ids(task, assay['expected_ids'])
        tier, rows, omitted = select_rows(complex_id, site_ids, structure_sites, observed)
        guidance = guidance_for(tier)
        msa_unit = units[group, 'msa_prompt']
        msa_text = (baseline / 'prompts/msa_prompt' / (assay['id'] + '.txt')).read_text()
        msa_request = read(baseline / msa_unit['request_path'])
        sequence_request = read(baseline / 'sequence_requests' / (assay['id'] + '.json'))
        if msa_request['contents'][0]['parts'][0]['text'] != msa_text:
            raise ValueError('Baseline MSA prompt and request text differ')
        sequence_text = sequence_request['contents'][0]['parts'][0]['text']
        if tier == 'assay_matched':
            joint_text = exact_prompt_text(msa_text, {'sites': structure_sites}, task, assay['expected_ids'], True)
            lean_prompt = (LEAN_ROOT / 'prompts/msa_prompt_structure_lean' / (assay['id'] + '.txt')).read_text()
            if joint_text != lean_prompt:
                raise ValueError('Assay-matched joint prompt is not the frozen lean prompt')
            suffix = lean_prompt[len(msa_text):]
            structure_text = sequence_text + suffix
        else:
            joint_text = render(msa_text, guidance, rows)
            structure_text = render(sequence_text, guidance, rows)
        texts = {STRUCTURE: structure_text, JOINT: joint_text}
        bases = {STRUCTURE: sequence_request, JOINT: msa_request}
        audit = dict(group_id=group, reference_complex_id=complex_id, tier=tier,
                     sites_in_prompt=[row['site_id'] for row in rows], omitted_site_ids=omitted,
                     symmetry_rna_included=complex_id == '5HO4',
                     paired_rna_chain='C' if complex_id == '2ZZN' else None)
        audits.append(audit)
        for condition, text in texts.items():
            request = copy.deepcopy(bases[condition])
            request['contents'][0]['parts'][0]['text'] = text
            request_path = out / 'requests' / condition / (assay['id'] + '.json')
            prompt_path = out / 'prompts' / condition / (assay['id'] + '.txt')
            request_path.parent.mkdir(parents=True, exist_ok=True)
            prompt_path.parent.mkdir(parents=True, exist_ok=True)
            request_path.write_text(json.dumps(request, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
            prompt_path.write_text(text)
            reused = condition == JOINT and tier == 'assay_matched'
            record = dict(assay, id=condition + '__' + assay['id'], base_unit_id=assay['id'],
                          condition=condition, tier=tier, request_path=str(request_path.relative_to(out)),
                          request_sha256=sha(request_path), sites_in_prompt=len(rows),
                          omitted_sites=len(omitted), reused_exact_prompt=reused)
            if reused:
                source = LEAN_ROOT / 'responses' / ('msa_prompt_structure_lean__' + assay['id'] + '.json')
                source_response = read(source)
                if source_response['status'] != 'ok':
                    raise ValueError('Lean joint response is not reusable')
                record['source_response'] = str(source)
                write(out / 'responses' / (record['id'] + '.json'),
                      dict(unit_id=record['id'], request_sha256=record['request_sha256'], status='ok',
                           scores=source_response['scores'], model=source_response['model'],
                           reused_exact_prompt=True, source_response=str(source),
                           source_response_sha256=sha(source)))
            index.append(record)
    protocol = read(baseline / 'config.json')['protocol']
    save_config = dict(schema='protein-rna-structure-local-v1', baseline=str(baseline), geometry=str(geometry),
                       lean_joint_cache=str(LEAN_ROOT), N=10, CR=25, protocol=protocol,
                       tiers=dict(assay_matched=sorted(EXACT), observed_contact=sorted(OBSERVED),
                                  paired_trna_c=sorted(PAIRED), related_structure=sorted(RELATED)),
                       reused_exact_joint_prompts=sum(row['reused_exact_prompt'] for row in index),
                       new_request_count=sum(not row['reused_exact_prompt'] for row in index),
                       max_attempts=3, concurrency=4, created_unix=time.time(), runner_sha256=sha(Path(__file__)))
    write(out / 'config.json', save_config)
    write(out / 'cohort.json', cohort)
    write(out / 'index.json', index)
    write(out / 'selection.json', audits)
    print(json.dumps(dict(new_requests=save_config['new_request_count'],
                          reused_joint=save_config['reused_exact_joint_prompts'],
                          tiers=[{k: audit[k] for k in ('reference_complex_id', 'tier', 'sites_in_prompt', 'omitted_site_ids')}
                                 for audit in audits]), ensure_ascii=False), flush=True)


def _appended(text, guidance):
    marker = '\n\n' + guidance
    if marker not in text:
        raise ValueError('Prompt is missing its structure guidance')
    return text.split(marker, 1)[1]


def validate(args):
    out = args.output
    config = read(out / 'config.json')
    if sha(Path(__file__)) != config['runner_sha256']:
        raise ValueError('Runner differs from the frozen local-structure campaign')
    if config['N'] != 10 or config['CR'] != 25:
        raise ValueError('N or CR changed')
    index = read(out / 'index.json')
    if len(index) != 20:
        raise ValueError('Expected two prompt arms for 10 assays')
    audits = {row['group_id']: row for row in read(out / 'selection.json')}
    baseline = Path(config['baseline'])
    for row in index:
        text = (out / 'prompts' / row['condition'] / (row['base_unit_id'] + '.txt')).read_text()
        request = read(out / row['request_path'])
        if request['contents'][0]['parts'][0]['text'] != text or sha(out / row['request_path']) != row['request_sha256']:
            raise ValueError('Prompt and request diverged')
        if request['generationConfig']['seed'] != 42 or request['generationConfig']['temperature'] != 0:
            raise ValueError('Generation protocol changed')
        audit = audits[row['group_id']]
        guidance = guidance_for(row['tier'])
        if row['tier'] == 'assay_matched' and row['condition'] == JOINT:
            lean_prompt = (LEAN_ROOT / 'prompts/msa_prompt_structure_lean' / (row['base_unit_id'] + '.txt')).read_text()
            if text != lean_prompt or not row['reused_exact_prompt']:
                raise ValueError('Assay-matched joint prompt was not reused')
        else:
            if row['reused_exact_prompt']:
                raise ValueError('Only assay-matched joint prompts may reuse a score')
        if row['tier'] == 'related_structure':
            body = _appended(text, guidance)
            if 'min_partner_heavy_distance_A' in body or 'delta_sasa_partner_A2' in body:
                raise ValueError('Related structure leaked a distance or area')
            parsed = json.loads(body)
            if not parsed or any(set(item) != {'site_id', 'structure_id', 'related_structure_contact'} or
                                 item['related_structure_contact'] is not True for item in parsed):
                raise ValueError('Related-structure row changed')
        elif row['tier'] != 'assay_matched':
            body = _appended(text, guidance)
            if '"partner_contact_count_5A":0' in body or 'null' in body or 'unavailable' in body:
                raise ValueError('Partial RNA prompt contains a zero, null, or unavailable marker')
            parsed = json.loads(body)
            if [item['site_id'] for item in parsed] != audit['sites_in_prompt']:
                raise ValueError('Prompt sites disagree with the selection audit')
            if any('rna_contact_moieties' in item and not item['rna_contact_moieties'] for item in parsed):
                raise ValueError('Empty moiety list entered the prompt')
        if len(row['expected_ids']) != 10 or row['batch'] != 0:
            raise ValueError('CR25 chunk changed')
    if sum(row['reused_exact_prompt'] for row in index) != 3 or config['new_request_count'] != 17:
        raise ValueError('Reuse count changed')
    write(out / 'validation.json', dict(status='passed', assays=10, prompt_arms=2, new_prompts=17,
                                      reused_exact_joint_prompts=3, CR=25, labels_read=False))
    print('Validation passed: 17 new local-structure prompts, 3 exact joint reuses', flush=True)


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
            for number in range(1, config['max_attempts'] + 1):
                if stop.is_set():
                    return
                started = time.time()
                attempt_path = out / 'attempts' / unit['id'] / f'{number:03d}.json'
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
        raise ValueError('Runner differs from the frozen local-structure campaign')
    baseline = Path(config['baseline'])
    labels = {(row['group_id'], row['variant_id']): row
              for row in read(Path(read(baseline / 'config.json')['source_base']) / 'offline_labels.json')}
    previous = {(row['group_id'], row['method']): row for row in csv.DictReader((baseline / 'per_assay.csv').open())}
    by_assay = {}
    for unit in read(out / 'index.json'):
        by_assay.setdefault(unit['group_id'], dict(unit))
        response = read(out / 'responses' / (unit['id'] + '.json')) if (out / 'responses' / (unit['id'] + '.json')).exists() else {'status': 'missing'}
        by_assay[unit['group_id']][unit['condition']] = response
    metrics = []
    rankings = []
    for assay in read(out / 'cohort.json'):
        group = assay['group_id']
        bundle = by_assay[group]
        ids = assay['expected_ids']
        msa = read(baseline / 'features' / (group + '.json'))['independent_msa_scores']
        scores = {}
        for condition in CONDITIONS:
            response = bundle[condition]
            values = response.get('scores') if response.get('status') == 'ok' else None
            scores[condition] = values if values and set(values) == set(ids) else None
        fusion_rows = None
        if scores[STRUCTURE] is not None and set(msa) >= set(ids):
            fusion_rows = fuse_rankings(ids, scores[STRUCTURE], {cid: msa[cid] for cid in ids})
            scores[FUSION] = {row['candidate_id']: row['fusion_value'] for row in fusion_rows}
        else:
            scores[FUSION] = None
        fused = {row['candidate_id']: row for row in fusion_rows or []}
        for method in (STRUCTURE, JOINT, FUSION):
            values = scores[method]
            rho = None
            if values is not None:
                xs = [values[cid] for cid in ids]
                ys = [labels[group, cid]['experimental_score_mean'] for cid in ids]
                if len(set(xs)) > 1 and len(set(ys)) > 1:
                    rho = float(spearmanr(xs, ys).statistic)
                if method == FUSION:
                    for cid in ids:
                        item = fused[cid]
                        rankings.append(dict(group_id=group, method=method, candidate_id=cid,
                                             score=item['fusion_value'], final_rank=item['final_rank'],
                                             gemini_score=item['gemini_score'], msa_score=item['msa_score']))
                else:
                    for cid, rank in zip(ids, midranks(xs, descending=True)):
                        rankings.append(dict(group_id=group, method=method, candidate_id=cid,
                                             score=values[cid], final_rank=rank,
                                             gemini_score=None, msa_score=None))
            metrics.append(dict(group_id=group, reference_complex_id=assay['reference_complex_id'],
                                method=method, tier=bundle['tier'], sites_in_prompt=bundle['sites_in_prompt'],
                                N=10, CR=25, status='complete' if values is not None else 'missing_score',
                                spearman=rho))
    means = {}
    for method in (STRUCTURE, JOINT, FUSION):
        values = [row['spearman'] for row in metrics if row['method'] == method and row['spearman'] is not None]
        means[method] = statistics.mean(values) if len(values) == 10 else None
    attempts = sorted((out / 'attempts').glob('*/*.json')) if (out / 'attempts').exists() else []
    attempt_rows = [read(path) for path in attempts]
    summary = dict(assays=10, N=10, CR=25, mean_signed_spearman=means,
                   new_api_attempts=len(attempt_rows),
                   successful_new_calls=sum(row.get('status') == 'ok' for row in attempt_rows),
                   reused_exact_joint_prompts=3,
                   unfinished=[row for row in metrics if row['status'] != 'complete'])
    write(out / 'summary.json', summary)
    with (out / 'per_assay.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics[0]))
        writer.writeheader()
        writer.writerows(metrics)
    if rankings:
        with (out / 'rankings.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rankings[0]))
            writer.writeheader()
            writer.writerows(rankings)
    lines = ['# Local structure arms (N=10, CR=25)', '',
             'Structure text is the local neighborhood the available coordinates actually show. '
             'Assay-matched RNA keeps real zeros. Partial RNA and corrected 2ZZN chain C keep positive contacts only. '
             '1B23 and 4YVI are related-structure contact flags, without distances. '
             'Fusion is fixed 50/50 rank fusion of the structure-only Gemini score with the frozen independent MSA score.', '']
    for method, value in means.items():
        lines.append(f'{method}: {value:.6f}' if value is not None else f'{method}: incomplete')
    lines += ['', f'New API attempts: {summary["new_api_attempts"]}. Successful new calls: {summary["successful_new_calls"]}. '
              'Reused exact joint prompts: 3 (2ERR, 4BS2, 4CIO).', '',
              '| Complex | Tier | Structure | MSA+structure | Fusion | Prior structure | Prior MSA+structure | Prior fusion |',
              '|---|---|---:|---:|---:|---:|---:|---:|']
    for assay in read(out / 'cohort.json'):
        local = {row['method']: row for row in metrics if row['group_id'] == assay['group_id']}
        display = []
        for method in (STRUCTURE, JOINT, FUSION):
            rho = local[method]['spearman']
            display.append(f'{rho:.6f}' if rho is not None else local[method]['status'])
        for old in ('structure', 'msa_prompt_structure', 'msa_gemini_rank5050_structure'):
            display.append(previous[assay['group_id'], old]['spearman'])
        lines.append('| ' + assay['reference_complex_id'] + ' | ' + local[STRUCTURE]['tier'] + ' | ' + ' | '.join(display) + ' |')
    lines.append('')
    (out / 'REPORT.txt').write_text('\n'.join(lines) + '\n')
    print(json.dumps(summary, ensure_ascii=False), flush=True)


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
