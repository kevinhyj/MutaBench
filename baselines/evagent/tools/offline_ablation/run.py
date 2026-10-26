"""Prepare, run and evaluate a new CR25 cohort without modifying old runs.

Only the existing reviewed PPI task intersection is executable. The complete
88-assay inventory is retained with explicit unavailable/excluded states.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import copy
import csv
import fcntl
import importlib.util
import json
import math
import os
import shlex
import sys
import time
import urllib.error
import urllib.request
import urllib.parse
from collections import Counter, defaultdict
from pathlib import Path

CODE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE))
from evagent.offline_ablation import (
    base_sites, digest, file_hash, finite, fuse_rankings, matching_cache,
    midranks, render_evidence, validate_scores,
)
from features import FeatureBuilder, save

ROOT = CODE.parent
SEQUENCE = ROOT / 'explanation_test/complex_sequence_all_assays_N250_N10_seed42_20260928'
BASELINE = ROOT / 'zeroshot_baseline_experiment/multichain/runs/n250_n20_n10_seed42_20260927'
ROSETTA88 = ROOT / 'zeroshot_baseline_experiment/multichain/runs/rosetta_flex_ddg_n250_88_20261002'
DEFAULT_OUTPUT = ROOT / 'experiments/ppi_cr25_from_new_serialization_v2_20261002'
CONDITIONS = ('msa_prompt', 'structure', 'msa_prompt_structure')
METHODS = ('gemini_sequence', 'msa_site_independent', 'msa_gemini_rank5050',
           'msa_prompt', 'gemini_structure', 'msa_gemini_rank5050_structure', 'msa_prompt_structure')
COHORTS = ('reviewed_21', 'previously_excluded_67', 'executable_88')
NATIVE_GEMINI = 'https://generativelanguage.googleapis.com/v1beta/models/'
SCORING_MODELS = {
    'gemini-3.8-flash': 'MEDIUM',
    'gemini-3.1-pro-preview': 'MEDIUM',
}
THINKING_LEVELS = ('LOW', 'MEDIUM', 'HIGH')


def normalize_thinking_level(level):
    if level is None:
        return 'MEDIUM'
    value = str(level).strip().upper()
    if value not in THINKING_LEVELS:
        raise ValueError('thinking_level must be LOW, MEDIUM, or HIGH')
    return value


def resolve_thinking_level(scoring_model, thinking_level=None):
    thinking = normalize_thinking_level(thinking_level) if thinking_level is not None else SCORING_MODELS[scoring_model]
    if scoring_model != 'gemini-3.8-flash' and thinking != 'MEDIUM':
        raise ValueError('Non-MEDIUM thinking is only enabled for gemini-3.8-flash PPI scoring')
    return thinking


def ppi_cohort(sequence_tasks, reviewed, name):
    """Keep the frozen 88/21 split; a later run may score the 67 previously held out."""
    if len(sequence_tasks) != 88 or len(reviewed) != 21:
        raise ValueError('Historical cohort changed; re-audit before preparation')
    if name == 'reviewed_21':
        return set(reviewed)
    if name == 'previously_excluded_67':
        return set(sequence_tasks) - set(reviewed)
    if name == 'executable_88':
        return set(sequence_tasks)
    raise ValueError('Unknown PPI ablation cohort')


def native_generate_endpoint(model):
    if model not in SCORING_MODELS:
        raise ValueError('Requested model is not on the PPI scoring allowlist')
    return NATIVE_GEMINI + model + ':generateContent'


def read(path):
    return json.loads(Path(path).read_text())


def atomic(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def csv_save(path: Path, rows: list[dict], fields=None):
    fields = fields or (list(rows[0]) if rows else [])
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def legacy_common():
    spec = importlib.util.spec_from_file_location('frozen_sequence_common', SEQUENCE / 'common.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def normalized(sites):
    return sorted([{key: site[key] for key in ('chain', 'position', 'wt', 'mutant')} |
                   {'insertion_code': site.get('insertion_code') or ''} for site in sites],
                  key=lambda site: (site['chain'], site['position'], site['insertion_code'], site['wt'], site['mutant']))


def snapshot_inputs(output, selected, sequence_tasks):
    """Read candidate identities from new/, excluding every measurement field."""
    expected = {}
    for group_id, selection in selected.items():
        for candidate in selection['variants']:
            for fact_id in candidate['fact_ids']:
                expected[fact_id] = (group_id, candidate)
    found, representatives, counts = {}, {}, Counter()
    inputs = {}
    for source in ('abagym', 'bindinggym'):
        path = ROOT / 'data/final_mutation_data/new' / f'{source}_facts_N250.jsonl'
        input_hash = file_hash(path)
        inputs[source] = dict(path=str(path), sha256=input_hash)
        with path.open() as handle:
            for line in handle:
                record = json.loads(line)
                group_id = source + '__' + record['assay_group_id']
                counts[group_id] += 1
                if record['fact_id'] not in expected:
                    continue
                wanted_group, variant = expected[record['fact_id']]
                current = normalized([dict(chain=m['chain'], position=m['position'],
                                           insertion_code=m.get('insertion_code') or '', wt=m['wt_aa'],
                                           mutant=m['mutant_aa']) for m in record['chain_mutations']])
                if group_id != wanted_group or current != normalized(variant['sites']):
                    raise ValueError('new/ candidate identity differs from the frozen selection')
                if record['fact_id'] in found:
                    raise ValueError('Duplicate fact ID in new/ input')
                found[record['fact_id']] = dict(group_id=group_id, candidate_id=variant['id'], sites=current)
                representatives.setdefault(group_id, dict(modality=record['modality'],
                    phenotype=record.get('phenotype'), temperature_K=record['context'].get('temperature_K'),
                    pH=record['context'].get('ph'), experimental_method=record['context'].get('experimental_method')))
    if set(found) != set(expected):
        raise ValueError(f"Frozen facts missing from new/: {len(set(expected) - set(found))}")
    if set(counts) != set(sequence_tasks) or len(counts) != 88:
        raise ValueError('new/ assay inventory differs from the original 88-assay cohort')
    save(output / 'input_snapshot.json', dict(inputs=inputs, counts=dict(counts),
                                             selected_fact_count=len(found), labels_read=False,
                                             selected_identity_sha256=digest(found),
                                             field_removal_report_sha256=file_hash(ROOT / 'data/final_mutation_data/new/field_removal_report.json')))
    save(output / 'selected_fact_mapping.json', found)
    return representatives


def payload(task, unit, features, condition, protocol, common):
    _, sites = base_sites(task)
    # new/ is the source of the context: discarded assay/complex/organism names
    # are not recovered into revised prompts from the old task metadata.
    context = dict(modality=task['modality'], assay_context=task['new_assay_context'],
                   reference_chains=[{key: chain[key] for key in ('chain', 'polymer_type', 'reference_sequence')}
                                     for chain in task['reference_chains']],
                   binding_partners={'groups': [dict(chains=group) for group in features['structure_definition']['partner_groups']]},
                   candidates=[dict(id=cid, sites=sites[cid]) for cid in unit['expected_ids']])
    context['evidence'] = render_evidence(task, unit['expected_ids'], features, condition)
    instruction = common.INSTRUCTION.replace(
        'No coordinates, structural features, MSA, or experimental variant measurements are supplied.',
        'Only the selected frozen evidence block is supplied; no experimental variant measurements are supplied.').replace(
        'Use only the supplied sequence and assay context.',
        'Use only the supplied sequence, assay context and selected evidence. Missing evidence is unknown, not zero.')
    instruction += ('MSA probabilities, if present, are weighted empirical amino-acid frequencies with a pseudocount, '
                    'not probabilities of homology. Chain homolog_count and global_neff apply to all corresponding '
                    'site rows. The 20-amino-acid probability vectors are not part of this prompt. '
                    'Do not perform any MSA/Gemini rank fusion in your response.\n')
    thinking_level = resolve_thinking_level(protocol.get('requested_model', 'gemini-3.8-flash'),
                                            protocol.get('thinking_level'))
    return dict(contents=[{'role': 'user', 'parts': [{'text': instruction + json.dumps(context, ensure_ascii=False, separators=(',', ':'), allow_nan=False)}]}],
                generationConfig=dict(temperature=0, seed=42, maxOutputTokens=protocol['max_output_tokens'],
                                      thinkingConfig=dict(includeThoughts=True, thinkingLevel=thinking_level)))


def verify_freeze(output):
    freeze = read(output / 'freeze.json')
    for relative, sha256 in freeze['artifact_hashes'].items():
        if file_hash(output / relative) != sha256:
            raise ValueError(f'Frozen artifact changed: {relative}')
    for path, sha256 in freeze['code_hashes'].items():
        if file_hash(Path(path)) != sha256:
            raise ValueError(f'Frozen code changed: {path}')
    for item in read(output / 'input_snapshot.json')['inputs'].values():
        if file_hash(Path(item['path'])) != item['sha256']:
            raise ValueError('new/ input changed after preparation')


def prepare(output, credential_file=None, endpoint=None, proxy=None, cohort='reviewed_21',
            requested_model=None, thinking_level=None):
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'freeze.json').exists():
        verify_freeze(output); print('already frozen', output); return
    sequence_tasks = {row['group_id']: row for row in read(SEQUENCE / 'task_index.json')
                      if row['source'] in ('abagym', 'bindinggym')}
    reviewed = {row['source'] + '__' + row['assay_group_id']: row for row in read(BASELINE / 'task_index.json')
                if row['source'] in ('abagym', 'bindinggym')}
    executable = ppi_cohort(sequence_tasks, reviewed, cohort)
    decisions = {row['source'] + '__' + row['assay_group_id']: row for row in read(BASELINE / 'assay_decisions.json')
                 if row['source'] + '__' + row['assay_group_id'] in sequence_tasks}
    selections = {row['group_id']: row for row in read(SEQUENCE / 'selection.json') if row['group_id'] in executable}
    representatives = snapshot_inputs(output, selections, sequence_tasks)
    protocol = read(SEQUENCE / 'protocol.json')
    scoring_model = requested_model or protocol['requested_model']
    if scoring_model not in SCORING_MODELS:
        raise ValueError('Requested model is not on the PPI scoring allowlist')
    thinking = resolve_thinking_level(scoring_model, thinking_level)
    native_endpoint = native_generate_endpoint(scoring_model)
    selected_endpoint = endpoint or (native_endpoint if scoring_model != protocol['requested_model'] else protocol['endpoint'])
    allowed_endpoints = {protocol['endpoint'], native_generate_endpoint(protocol['requested_model']), native_endpoint}
    if selected_endpoint not in allowed_endpoints:
        raise ValueError('Endpoint differs from the approved native Gemini routes')
    if scoring_model != protocol['requested_model'] and selected_endpoint != native_endpoint:
        raise ValueError('Alternate scoring model must use its native generateContent route')
    if proxy not in (None, 'http://127.0.0.1:7777'):
        raise ValueError('Proxy differs from the explicitly requested local 7777 route')
    pro_run = scoring_model == 'gemini-3.1-pro-preview'
    high_thinking = thinking == 'HIGH'
    config = dict(schema='ppi-cr25-ablation-v2', data_directory=str(ROOT / 'data/final_mutation_data/new'),
                  cohort=cohort, cohort_assays=88, executable_assays=len(executable),
                  executable_reviewed_assays=21, N=250, chunk_ranking_CR=25,
                  sampling_seed=42, generation_seed=42, temperature=0, thinking_level=thinking,
                  requested_model=scoring_model, endpoint=selected_endpoint, proxy=proxy,
                  authentication='x-goog-api-key' if urllib.parse.urlsplit(selected_endpoint).hostname == 'generativelanguage.googleapis.com' else 'gateway_api_key_and_bearer',
                  max_output_tokens=protocol['max_output_tokens'], concurrency=5 if (pro_run or high_thinking) else 10, max_attempts=3,
                  timeout_seconds=600 if (pro_run or high_thinking) else 300, credential_file=str(Path(credential_file).expanduser().resolve()) if credential_file else '/home/huangyanjie/.config/evagent/yunjintao.key',
                  credential_format='external raw token, GEMINI_API_KEY env file, or api_key JSON; never persisted',
                  fusion=dict(weights=[0.5, 0.5], direction='higher is better',
                              components='within-assay ascending average midrank percentile (rank-1)/(N-1)',
                              final_rank='descending average midrank; rank 1 is best; exact ties preserved',
                              missing='no imputation; no fusion metric without all N scores',
                              definition_source=str(ROOT / 'explanation_test/complex_rank_fusion_true_n250_seed42_20261002/recompute_rank_fusion.py')),
                  conditions=dict(msa_gemini_rank5050='cached exact original sequence prompt, then offline fusion',
                                  msa_gemini_rank5050_structure='new structure prompt, then offline fusion',
                                  msa_prompt='new reduced MSA prompt',
                                  msa_prompt_structure='new reduced MSA and structure prompt'),
                  base_prompt_policy='Revised prompts use new/ context allowlist and frozen reference sequence/partner mappings; discarded metadata names are not restored',
                  comparison_limit='Original sequence cache retains its original prompt; revised prompt arms also reflect the requested new/ metadata removal and shared site-ID representation',
                  original_predictions_reused_only_for_postprocessing=True,
                  labels_policy='Read native_benefit from new/ only in independent evaluation; check against the exact historical fact-ID label ledger; never fit weights',
                  provider_unknown_errors_not_classified_as_safety=True,
                  scoring_model_note=('Same CR25 prompts, seed=42 and temperature 0; thinking_level is frozen per directory. '
                                      'requested_model/endpoint identify the scorer. Sequence cache keeps its historical prompt. '
                                      '3.1 Pro and HIGH thinking use timeout 600s and concurrency 5.'))
    save(output / 'config.json', config)
    common = legacy_common(); builder = FeatureBuilder(ROOT, output)
    original_units = read(SEQUENCE / 'index.json')
    baseline_groups = read(BASELINE / 'partner_groups.json')
    baseline_freeze = read(BASELINE / 'freeze.json')
    rosetta_groups = read(ROSETTA88 / 'partner_groups.json')
    rosetta_freeze = read(ROSETTA88 / 'freeze.json')
    plan, inventory, cached, checks = [], [], [], []
    for group_id, row in sorted(sequence_tasks.items()):
        decision = decisions[group_id]
        status = dict(group_id=group_id, source=row['source'], assay_group_id=row['assay_group_id'], N=250,
                      input_records=read(output / 'input_snapshot.json')['counts'][group_id],
                      historical_status=decision['status'], historical_reason=decision['reason'],
                      inference_status='not_executed', original_selection_sha256=file_hash(SEQUENCE / 'selection.json'))
        if group_id not in executable:
            status['inference_status'] = ('held_out_reviewed_21_separate_run' if cohort == 'previously_excluded_67'
                                          else 'not_executed_outside_reviewed_scope')
            inventory.append(status); continue
        task_path = SEQUENCE / row['task_path']; task = read(task_path)
        in_reviewed = group_id in reviewed
        if in_reviewed:
            observed_row = reviewed[group_id]
            observed_path = BASELINE / observed_row['task_path']
            freeze_key = observed_row['task_path']
            group_key = observed_row['group_id']
            groups, observed_freeze = baseline_groups, baseline_freeze
        else:
            observed_path = ROSETTA88 / 'prepared' / f'{group_id}.json'
            freeze_key = f'prepared/{group_id}.json'
            group_key = group_id
            groups, observed_freeze = rosetta_groups, rosetta_freeze
        if file_hash(observed_path) != observed_freeze['task_hashes'][freeze_key]:
            raise ValueError('Observed-sequence frozen task changed')
        observed = read(observed_path)
        candidates = task['candidates']; ids = [candidate['id'] for candidate in candidates]
        if len(ids) != 250 or len(set(ids)) != 250:
            raise ValueError('Original N250 candidate set is incomplete/duplicate')
        if [candidate['id'] for candidate in observed['candidates']] != ids:
            raise ValueError('Sequence/MSA candidate order differs')
        if [normalized(candidate['sites']) for candidate in observed['candidates']] != [normalized(candidate['sites']) for candidate in candidates]:
            raise ValueError('Sequence/MSA candidate substitutions differ')
        task['new_assay_context'] = {key: value for key, value in representatives[group_id].items() if key != 'modality'}
        features, mapping = builder.build(task, observed, groups[group_key])
        old_msa = None
        if in_reviewed:
            old_msa_path = BASELINE / 'results/MSA_site_independent' / f"{observed_row['group_id']}.json"
            old_msa = read(old_msa_path)
            if old_msa['status'] == 'complete':
                previous = {prediction['candidate_id']: prediction['score'] for prediction in old_msa['predictions']}
                validate_scores(previous, ids)
                if not features['full_msa_coverage']:
                    raise ValueError('New MSA evidence lost previously complete coverage')
                if not all(math.isclose(previous[cid], features['independent_msa_scores'][cid], abs_tol=1e-10, rel_tol=1e-10) for cid in ids):
                    raise ValueError('Frozen MSA score protocol changed')
            elif features['full_msa_coverage']:
                raise ValueError('Unexpected MSA coverage change needs a separate protocol review')
            mapping.update(original_task_sha256=file_hash(task_path), observed_task_sha256=file_hash(observed_path),
                           original_msa_result_sha256=file_hash(old_msa_path),
                           original_msa_status=old_msa['status'], original_msa_reason=old_msa.get('reason'))
        else:
            mapping.update(original_task_sha256=file_hash(task_path), observed_task_sha256=file_hash(observed_path),
                           original_msa_result_sha256=None, original_msa_status='not_in_reviewed_baseline_queue',
                           original_msa_reason=decision['reason'], observed_source=str(observed_path))
        save(output / 'site_mapping' / f'{group_id}.json', mapping)
        save(output / 'features' / f'{group_id}.json', features)
        save(output / 'prepared' / f'{group_id}.json', task)
        units = [unit for unit in original_units if unit['group_id'] == group_id]
        if [cid for unit in units for cid in unit['expected_ids']] != ids or any(len(unit['expected_ids']) != 25 for unit in units):
            raise ValueError('Original CR25 partition/order differs')
        all_cached_scores = {}
        cached_chunks = 0
        for unit in units:
            request_path = SEQUENCE / unit['request_path']; response_path = SEQUENCE / 'responses' / f"{unit['id']}.json"
            expected = common.payload(read(task_path), unit, protocol)
            cache_unit = unit
            cache_error = None
            if not response_path.exists():
                cache_error = 'missing_sequence_response'
            else:
                response_metadata = read(response_path)
                actual_hash = response_metadata.get('actual_request_sha256')
                if actual_hash and actual_hash != unit['request_sha256']:
                    amendment = next((entry for entry in read(SEQUENCE / 'format_retry_protocol.json')['units']
                                      if entry['unit_id'] == unit['id']), None)
                    if amendment is None or amendment['frozen_request_sha256'] != unit['request_sha256']:
                        cache_error = 'unrecognized_amended_cache'
                    elif file_hash(request_path) != unit['request_sha256'] or read(request_path) != expected:
                        cache_error = 'amended_request_base_differs'
                    else:
                        request_path = SEQUENCE / amendment['actual_request_path']
                        amended = read(request_path)
                        reduced = copy.deepcopy(amended)
                        del reduced['generationConfig']['responseSchema']
                        del reduced['generationConfig']['responseMimeType']
                        if reduced != expected or actual_hash != amendment['actual_request_sha256']:
                            cache_error = 'amendment_changed_scientific_prompt'
                        else:
                            cache_unit = {**unit, 'request_sha256': actual_hash,
                                          'original_request_sha256': unit['request_sha256']}
                            expected = amended
            if cache_error is None:
                try:
                    scores, origin = matching_cache(cache_unit, task_path, request_path, response_path, expected)
                except ValueError as error:
                    cache_error = str(error)
            if cache_error is None:
                save(output / 'cached_sequence' / f"{unit['id']}.json", dict(scores=scores, origin=origin))
                save(output / 'cached_sequence_prompts' / f"{unit['id']}.json", read(request_path))
                all_cached_scores.update(scores); cached.append(dict(group_id=group_id, unit_id=unit['id'], **origin))
                cached_chunks += 1
            elif in_reviewed:
                raise ValueError(cache_error)
            for condition in CONDITIONS:
                new_unit = dict(id=unit['id'] + '__' + condition, group_id=group_id, source=row['source'],
                                condition=condition, batch=unit['batch'], expected_ids=unit['expected_ids'])
                request = payload(task, unit, features, condition, config, common)
                new_path = output / 'requests' / f"{new_unit['id']}.json"
                save(new_path, request)
                new_unit.update(request_path=str(new_path.relative_to(output)), request_sha256=file_hash(new_path),
                                semantic_request_sha256=digest(request))
                plan.append(new_unit)
        if cached_chunks == 10:
            validate_scores(all_cached_scores, ids, bounded=True)
        elif in_reviewed:
            raise ValueError('Original sequence cache is incomplete')
        reproduced = bool(old_msa and old_msa['status'] == 'complete')
        status.update(inference_status='prepared', candidates=250, original_Gemini_cache_chunks=cached_chunks,
                      full_msa_coverage=features['full_msa_coverage'],
                      msa_candidates_available=sum(value is not None for value in features['independent_msa_scores'].values()),
                      structure_sasa_available=sum(item['sasa_status'] == 'ok'
                          for rows in features['structure'].values() for item in rows.values()))
        inventory.append(status)
        checks.append(dict(group_id=group_id, candidates=250, CR=25, cache_match=cached_chunks == 10,
                           original_MSA_score_reproduced=reproduced,
                           full_msa_coverage=features['full_msa_coverage'], site_mapping_validated=True,
                           missing_values_explicit=True, labels_read_by_predictor=False))
        print('prepared', group_id, 'MSA_complete', features['full_msa_coverage'], 'sequence_cache', cached_chunks, flush=True)
    save(output / 'plan.json', plan)
    save(output / 'assay_inventory.json', inventory)
    save(output / 'cache_sources.json', cached)
    save(output / 'preflight_validation.json', dict(status='passed', unit_tests='code/tests/test_offline_ablation.py',
                                                  assays=checks, retained_assay_inventory=88,
                                                  reviewed_prepared_assays=sum(row['inference_status'] == 'prepared' for row in inventory),
                                                  planned_new_calls=len(plan),
                                                  matching_cached_calls=len(cached), cohort=cohort))
    files = sorted(path for path in output.rglob('*') if path.is_file() and path.name != 'freeze.json')
    code_paths = [Path(__file__), Path(__file__).with_name('features.py'), CODE / 'evagent/offline_ablation.py',
                  ROOT / 'baseline/singlechain/models.py', ROOT / 'baseline/singlechain/common.py', SEQUENCE / 'common.py']
    save(output / 'freeze.json', dict(artifact_hashes={str(path.relative_to(output)): file_hash(path) for path in files},
                                     code_hashes={str(path): file_hash(path) for path in code_paths}))
    print('FROZEN', output, 'cached', len(cached), 'new_calls', len(plan), flush=True)


def run_unit(output, unit, config, credential, common):
    response_path = output / 'responses' / f"{unit['id']}.json"
    if response_path.exists():
        response = read(response_path)
        if response.get('request_sha256') != unit['request_sha256']:
            raise ValueError('Saved new response hash mismatch')
        if response.get('status') == 'ok':
            validate_scores(response['scores'], unit['expected_ids'], bounded=True)
            return response
        if response.get('terminal'):
            return response
    # Started markers are audit records, not completed provider attempts.
    attempt_directory = output / 'attempts' / unit['id']
    prior = sorted(path for path in attempt_directory.glob('*.json')
                   if not path.name.endswith('.started.json'))
    for marker in attempt_directory.glob('*.started.json'):
        completed = marker.with_name(marker.name.replace('.started.json', '.json'))
        if not completed.exists():
            # Delivery may already have happened; never resend or overwrite it.
            response = dict(unit_id=unit['id'], group_id=unit['group_id'],
                            condition=unit['condition'], status='delivery_unknown', terminal=True,
                            request_sha256=unit['request_sha256'], raw_attempt=str(marker))
            atomic(response_path, response)
            return response
    previous_attempt = max((int(path.stem) for path in prior), default=0)
    if previous_attempt >= config['max_attempts']:
        return dict(unit_id=unit['id'], status='retry_exhausted', terminal=True)
    request_path = output / unit['request_path']
    if file_hash(request_path) != unit['request_sha256']:
        raise ValueError('New frozen prompt hash mismatch')
    proxy = config.get('proxy')
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({'http': proxy, 'https': proxy} if proxy else {}))
    for attempt in range(previous_attempt + 1, config['max_attempts'] + 1):
        body, http_status, result, started = {}, None, {}, time.time()
        ap = output / 'attempts' / unit['id'] / f'{attempt:03d}.json'
        ap.parent.mkdir(parents=True, exist_ok=True)
        atomic(ap.with_suffix('.started.json'), dict(unit_id=unit['id'], attempt=attempt,
                                                    request_sha256=unit['request_sha256'], started=started))
        try:
            headers = {'Content-Type': 'application/json', 'x-goog-api-key': credential}
            if config.get('authentication') != 'x-goog-api-key':
                headers['Authorization'] = 'Bearer ' + credential
            request = urllib.request.Request(config['endpoint'], data=json.dumps(read(request_path)).encode(), method='POST',
                                             headers=headers)
            try:
                with opener.open(request, timeout=max(3600, config['timeout_seconds'])) as remote:
                    http_status = remote.status; data = remote.read()
            except urllib.error.HTTPError as error:
                http_status = error.code; data = error.read()
            try:
                body = json.loads(data)
            except ValueError:
                body = dict(unparsed_text=data.decode(errors='replace')[:10000])
            result = common.parse(body, unit['expected_ids']) if http_status == 200 else dict(status='http_error')
            reported_model = str(body.get('modelVersion', ''))
            expected_model = config['requested_model']
            if result['status'] == 'ok' and not (reported_model == expected_model or
                                               reported_model.startswith(expected_model + '-')):
                result = dict(status='model_identity_mismatch')
        except Exception as error:
            result = dict(status='transport_error', error_type=type(error).__name__, message=str(error))
        record = dict(unit_id=unit['id'], attempt=attempt, request_sha256=unit['request_sha256'],
                      started=started, finished=time.time(), http_status=http_status,
                      status=result['status'], body=body, parsed=result)
        record = json.loads(json.dumps(record).replace(credential, '[REDACTED]'))
        atomic(ap, record)
        stop = result['status'] in ('permanent_refusal', 'model_identity_mismatch') or http_status in (400, 401, 402, 403, 404, 413, 422)
        if result['status'] == 'ok':
            response = dict(unit_id=unit['id'], group_id=unit['group_id'], condition=unit['condition'],
                            status='ok', scores=result['scores'], request_sha256=unit['request_sha256'],
                            model=body.get('modelVersion'), attempt=attempt, usage=body.get('usageMetadata'),
                            actual_prompt_path=str(request_path), raw_attempt=str(ap), finished=time.time())
            atomic(response_path, response); return response
        if stop or attempt == config['max_attempts']:
            response = dict(unit_id=unit['id'], group_id=unit['group_id'], condition=unit['condition'],
                            status=result['status'], terminal=True, http_status=http_status,
                            request_sha256=unit['request_sha256'], raw_attempt=str(ap))
            atomic(response_path, response); return response
        time.sleep(5 * attempt)


def load_credential(path):
    """Read an authorized local credential without sourcing shell code or saving it."""
    path = Path(path).expanduser().resolve()
    if path.stat().st_mode & 0o077:
        raise ValueError('Credential permissions are too broad')
    text = path.read_text().strip()
    if text.startswith('{'):
        try:
            credential = json.loads(text).get('api_key')
        except (ValueError, AttributeError):
            raise ValueError('Malformed credential JSON') from None
    elif path.suffix == '.env' or text.startswith(('GEMINI_API_KEY=', 'export GEMINI_API_KEY=')):
        credentials = []
        for line in text.splitlines():
            line = line.strip()
            if line.startswith('export '):
                line = line[7:].strip()
            if not line or line.startswith('#') or '=' not in line:
                continue
            name, value = line.split('=', 1)
            if name.strip() == 'GEMINI_API_KEY':
                try:
                    parsed = shlex.split(value, comments=True)
                except ValueError:
                    raise ValueError('Malformed credential env file') from None
                credentials.append(parsed[0] if len(parsed) == 1 else None)
        credential = credentials[0] if len(credentials) == 1 else None
    else:
        credential = text
    if not isinstance(credential, str) or not credential or any(character.isspace() for character in credential):
        raise ValueError('Credential unavailable')
    return credential


def run(output, credential_file=None):
    verify_freeze(output)
    with (output / 'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        _run_locked(output, credential_file)


def _run_locked(output, credential_file=None):
    config = read(output / 'config.json'); plan = read(output / 'plan.json')
    key_path = Path(credential_file or config['credential_file']).expanduser().resolve()
    credential = load_credential(key_path)
    atomic(output / 'runtime_authentication.json', dict(credential_file=str(key_path),
           credential_persisted=False, endpoint=config['endpoint'], proxy=config.get('proxy'), updated=time.time()))
    common = legacy_common()
    pending = []
    for unit in plan:
        path = output / 'responses' / f"{unit['id']}.json"
        if path.exists() and read(path).get('status') == 'ok':
            validate_scores(read(path)['scores'], unit['expected_ids'], bounded=True)
        else:
            pending.append(unit)
    started = time.time(); counts = Counter()
    atomic(output / 'run_state.json', dict(status='running', total=len(plan), pending=len(pending), started=started))
    # Submit incrementally so authentication/credit/model errors stop new calls.
    with concurrent.futures.ThreadPoolExecutor(max_workers=config['concurrency']) as pool:
        iterator = iter(pending); active = {}; stop = False
        while True:
            # Probe the first request before filling the concurrent queue.
            capacity = config['concurrency'] if counts else 1
            while not stop and len(active) < capacity:
                unit = next(iterator, None)
                if unit is None:
                    break
                active[pool.submit(run_unit, output, unit, config, credential, common)] = unit
            if not active:
                break
            finished, _ = concurrent.futures.wait(active, timeout=30, return_when=concurrent.futures.FIRST_COMPLETED)
            for future in finished:
                unit = active.pop(future); result = future.result(); counts[result['status']] += 1
                stop |= result.get('http_status') in (400, 401, 402, 403, 404, 413, 422) or result['status'] == 'model_identity_mismatch'
                # A sandbox connectivity failure stops further submissions for escalation.
                if result['status'] == 'transport_error':
                    raw = read(Path(result['raw_attempt']))
                    message = raw['parsed'].get('message', '').lower()
                    stop |= any(word in message for word in ('name resolution', 'name or service', 'operation not permitted', 'network is unreachable'))
                print(unit['id'], result['status'], flush=True)
            atomic(output / 'run_state.json', dict(status='stopping' if stop else 'running',
                  counts=dict(counts), total=len(plan), active=len(active), started=started, updated=time.time()))
    paths = list((output / 'responses').glob('*.json'))
    states = Counter(read(path)['status'] for path in paths)
    atomic(output / 'run_state.json', dict(status='complete' if states.get('ok', 0) == len(plan) else 'finished_with_uncompleted_units',
           counts=dict(states), total=len(plan), started=started, finished=time.time()))
    print('RUN_FINISHED', dict(states), flush=True)


def evaluate(output):
    from scipy.stats import kendalltau, spearmanr
    verify_freeze(output)
    inventory = read(output / 'assay_inventory.json'); plan = read(output / 'plan.json')
    source_map = read(output / 'selected_fact_mapping.json')
    # This evaluator is the only stage that reads new/ measurement fields.
    measurements = defaultdict(list)
    for item in read(output / 'input_snapshot.json')['inputs'].values():
        with Path(item['path']).open() as handle:
            for line in handle:
                record = json.loads(line)
                if record['fact_id'] in source_map:
                    key = source_map[record['fact_id']]
                    value = record['native_benefit']
                    if not finite(value):
                        raise ValueError('Selected evaluator measurement is unavailable')
                    measurements[(key['group_id'], key['candidate_id'])].append(value)
    gold = {key: sum(values) / len(values) for key, values in measurements.items()}
    old_gold = {(row['group_id'], row['variant_id']): row for row in read(SEQUENCE / 'offline_labels.json')
                if row['group_id'] in {key[0] for key in gold}}
    if set(gold) != set(old_gold) or any(not math.isclose(value, old_gold[key]['experimental_score_mean'],
                                                        abs_tol=1e-12, rel_tol=1e-12) for key, value in gold.items()):
        raise ValueError('new/ labels differ from the original frozen evaluation policy')
    atomic(output / 'evaluation_labels.json', [dict(group_id=key[0], candidate_id=key[1], value=value,
                                                   original_fact_ids=old_gold[key]['fact_ids']) for key, value in sorted(gold.items())])
    metrics, predictions = [], []
    for assay in inventory:
        gid = assay['group_id']
        if assay['inference_status'] != 'prepared':
            for method in METHODS:
                metrics.append(dict(group_id=gid, source=assay['source'], method=method, N=250,
                                    scored=0, status=assay['inference_status'], full_msa_coverage=None,
                                    spearman=None, kendall=None, reason=assay['historical_reason']))
            continue
        task = read(output / 'prepared' / f'{gid}.json'); ids = [candidate['id'] for candidate in task['candidates']]
        features = read(output / 'features' / f'{gid}.json')
        values = defaultdict(dict); origins = defaultdict(dict)
        units = [unit for unit in read(SEQUENCE / 'index.json') if unit['group_id'] == gid]
        assay_plan = [unit for unit in plan if unit['group_id'] == gid]
        live_sequence = any(unit['condition'] == 'sequence' for unit in assay_plan)
        if not live_sequence:
            for unit in units:
                cached_path = output / 'cached_sequence' / f"{unit['id']}.json"
                if not cached_path.exists():
                    continue
                cached = read(cached_path)
                values['gemini_sequence'].update(cached['scores'])
                origins['gemini_sequence'].update({cid: cached['origin']['source_response'] for cid in unit['expected_ids']})
        values['msa_site_independent'] = {cid: value for cid, value in features['independent_msa_scores'].items() if value is not None}
        method_name = {'structure': 'gemini_structure', 'sequence': 'gemini_sequence'}
        for unit in assay_plan:
            path = output / 'responses' / f"{unit['id']}.json"
            if not path.exists():
                continue
            response = read(path)
            if response.get('status') != 'ok' or response.get('request_sha256') != unit['request_sha256']:
                continue
            validate_scores(response['scores'], unit['expected_ids'], bounded=True)
            method = method_name.get(unit['condition'], unit['condition'])
            values[method].update(response['scores'])
            origins[method].update({cid: str(path) for cid in unit['expected_ids']})
        fusion_details = {}
        for method, component in (('msa_gemini_rank5050', 'gemini_sequence'),
                                  ('msa_gemini_rank5050_structure', 'gemini_structure')):
            if len(values[component]) == 250 and len(values['msa_site_independent']) == 250:
                rows = fuse_rankings(ids, values[component], values['msa_site_independent'])
                fusion_details[method] = {row['candidate_id']: row for row in rows}
                values[method] = {row['candidate_id']: row['fusion_value'] for row in rows}
                origins[method] = origins[component]
        for method in METHODS:
            scores = values[method]; complete = set(scores) == set(ids) and len(scores) == 250
            row = dict(group_id=gid, source=assay['source'], method=method, N=250, scored=len(scores),
                       status='complete' if complete else 'incomplete', full_msa_coverage=features['full_msa_coverage'],
                       spearman=None, kendall=None, reason=None if complete else 'missing_MSA_or_model_scores')
            if complete:
                x, y = [scores[cid] for cid in ids], [gold[(gid, cid)] for cid in ids]
                if len(set(x)) > 1 and len(set(y)) > 1:
                    row['spearman'] = float(spearmanr(x, y).statistic)
                    row['kendall'] = float(kendalltau(x, y).statistic)
                else:
                    row['reason'] = 'constant_predictions_or_labels'
                rank = dict(zip(ids, midranks(x, descending=True)))
                for cid in ids:
                    prediction = dict(group_id=gid, source=assay['source'], method=method, candidate_id=cid,
                                      score=scores[cid], final_rank=rank[cid], cache_or_response_source=origins[method].get(cid),
                                      gemini_score=None, msa_score=None, gemini_rank_percentile=None,
                                      msa_rank_percentile=None, fusion_value=None)
                    if method in fusion_details:
                        prediction.update(fusion_details[method][cid])
                    predictions.append(prediction)
            metrics.append(row)
    csv_save(output / 'per_assay_metrics.csv', metrics)
    csv_save(output / 'rankings_and_components.csv', predictions)
    summary = []
    for method in METHODS:
        for source in ('all', 'abagym', 'bindinggym'):
            rows = [row for row in metrics if row['method'] == method and (source == 'all' or row['source'] == source)]
            valid = [row for row in rows if row['status'] == 'complete' and row['spearman'] is not None]
            paired = [row for row in valid if row['full_msa_coverage']]
            summary.append(dict(method=method, source=source, inventory_assays=len(rows),
                                complete_assays=sum(row['status'] == 'complete' for row in rows),
                                defined_assays=len(valid), mean_spearman=sum(row['spearman'] for row in valid) / len(valid) if valid else None,
                                full_MSA_paired_assays=len(paired),
                                full_MSA_paired_mean_spearman=sum(row['spearman'] for row in paired) / len(paired) if paired else None))
    csv_save(output / 'summary.csv', summary)
    attempts = [read(path) for path in (output / 'attempts').glob('*/*.json') if not path.name.endswith('.started.json')]
    report = dict(cohort_assays=88, reviewed_executable_assays=read(output / 'config.json').get('executable_assays'),
                  cohort=read(output / 'config.json').get('cohort', 'reviewed_21'),
                  historical_outside_reviewed_scope=67,
                  historical_exclusion_reasons=dict(Counter(row['historical_reason'] for row in inventory if row['inference_status'] != 'prepared')),
                  CR=25, N=250, seed=42,
                  reused_Gemini_chunks=len(read(output / 'cache_sources.json')),
                  postprocessing_complete_assays=sum(row['method'] == 'msa_gemini_rank5050' and row['status'] == 'complete' for row in metrics),
                  planned_new_logical_calls=len(plan), new_attempt_calls=len(attempts),
                  new_successful_logical_calls=sum(read(path).get('status') == 'ok' for path in (output / 'responses').glob('*.json')),
                  provider_status_counts=dict(Counter(attempt['status'] for attempt in attempts)),
                  reported_total_tokens=sum((attempt.get('body', {}).get('usageMetadata') or {}).get('totalTokenCount', 0) for attempt in attempts),
                  labels_from_new_match_historical=True, label_count=len(gold), weights_fitted_on_labels=False,
                  original_results_modified=False, means=summary,
                  limitations=['This directory scores the 88-assay executable union; Gemini identity is the frozen requested_model.'
                               if read(output / 'config.json').get('cohort') == 'executable_88'
                               else 'The entire 88-assay inference/ranking request was not executed; excluded contexts remain unscored.',
                               'Nine reviewed assays lack complete MSA candidate coverage; no replacements and no missing scores imputed.',
                               'Prompt arms use explicit missing evidence and are reported separately from the full-MSA paired cohort.',
                               'The original sequence cache retains its original prompt; revised prompts also adopt new/ metadata removal.',
                               'Structure SASA uses observed WT atoms and excludes incomplete target residues; no missing atoms are generated.'])
    atomic(output / 'report.json', report)
    print(json.dumps({key: value for key, value in report.items() if key not in ('means', 'limitations')}, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('prepare', 'run', 'evaluate'))
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--credential-file', type=Path, help='Prepare/run: external raw/env/JSON credential path; token is never saved')
    parser.add_argument('--endpoint', help='Prepare only: exact approved native Gemini endpoint')
    parser.add_argument('--proxy', help='Prepare only: explicitly requested http://127.0.0.1:7777 proxy')
    parser.add_argument('--cohort', choices=COHORTS, default='reviewed_21',
                        help='Prepare only: reviewed 21, the 67 previously held out, or the 88-assay executable union')
    parser.add_argument('--requested-model', choices=tuple(SCORING_MODELS),
                        help='Prepare only: native scorer; default is the frozen sequence-protocol Flash model')
    parser.add_argument('--thinking-level', choices=THINKING_LEVELS,
                        help='Prepare only: LOW, MEDIUM or HIGH for gemini-3.8-flash; default MEDIUM')
    args = parser.parse_args()
    if args.stage == 'evaluate' and args.credential_file is not None:
        parser.error('--credential-file applies only to prepare/run')
    if args.stage != 'prepare' and (args.endpoint or args.proxy or args.cohort != 'reviewed_21' or args.requested_model or args.thinking_level):
        parser.error('--endpoint, --proxy, --cohort, --requested-model and --thinking-level are frozen during prepare')
    if args.stage == 'run':
        run(args.output.resolve(), args.credential_file)
    elif args.stage == 'prepare':
        prepare(args.output.resolve(), args.credential_file, args.endpoint, args.proxy, args.cohort,
                args.requested_model, args.thinking_level)
    else:
        evaluate(args.output.resolve())


if __name__ == '__main__':
    main()
