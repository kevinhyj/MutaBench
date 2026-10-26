"""Clone the completed 3.8 Flash MEDIUM PPI freeze; only thinkingLevel changes.

Does not rescore MEDIUM. Sequence, MSA, structure and MSA+structure are four
CR25 arms. Historical sequence cache is not used as LOW/HIGH scores.
"""
from __future__ import annotations

import argparse
import copy
import json
import shutil
import sys
from pathlib import Path

ROOT = Path('/data/huangyanjie/EVAgent')
CODE = ROOT / 'code'
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(CODE / 'tools/offline_ablation'))
from features import save
from evagent.offline_ablation import digest, file_hash

R4 = ROOT / 'experiments/ppi_cr25_from_new_serialization_v2_20261004_r4_completion'
R2 = ROOT / 'experiments/ppi_cr25_previously_excluded67_20261004_r2_completion'
SEQUENCE = ROOT / 'explanation_test/complex_sequence_all_assays_N250_N10_seed42_20260928'
ARMS = ('sequence', 'msa_prompt', 'structure', 'msa_prompt_structure')
COPY_DIRS = ('features', 'site_mapping', 'prepared', 'frozen_probability_tables',
             'cached_sequence', 'cached_sequence_prompts')


def load(path: Path):
    return json.loads(path.read_text())


def retarget_request(request: dict, level: str, max_output_tokens=None) -> dict:
    if level not in ('LOW', 'HIGH'):
        raise ValueError('Clone thinking_level must be LOW or HIGH; MEDIUM is reused in place')
    out = copy.deepcopy(request)
    gen = out['generationConfig']
    extra = set(gen) - {'temperature', 'seed', 'maxOutputTokens', 'thinkingConfig',
                        'responseMimeType', 'responseSchema'}
    if extra:
        raise ValueError(f'Unexpected generationConfig keys: {sorted(extra)}')
    if gen['temperature'] != 0 or gen['seed'] != 42:
        raise ValueError('Seed or temperature differs from the completed MEDIUM freeze')
    thinking = gen['thinkingConfig']
    if set(thinking) != {'includeThoughts', 'thinkingLevel'}:
        raise ValueError('thinkingConfig keys differ from the completed freeze')
    if thinking['includeThoughts'] is not True or thinking['thinkingLevel'] != 'MEDIUM':
        raise ValueError('Completed freeze is not MEDIUM includeThoughts')
    if gen['maxOutputTokens'] not in (16384, 65536):
        raise ValueError('maxOutputTokens is not the completed 16384/65536 protocol')
    thinking['thinkingLevel'] = level
    if max_output_tokens is not None:
        if max_output_tokens not in (16384, 65536):
            raise ValueError('max_output_tokens must be the completed 16384/65536 protocol')
        gen['maxOutputTokens'] = max_output_tokens
    if out['contents'] != request['contents']:
        raise ValueError('Prompt contents changed while retargeting')
    return out


def copy_tree(src: Path, dst: Path, name: str) -> None:
    source = src / name
    if not source.exists():
        return
    target = dst / name
    target.mkdir(parents=True, exist_ok=True)
    for path in source.iterdir():
        if path.is_file():
            shutil.copy2(path, target / path.name)


def sequence_request(chunk_id: str, sources: list[Path]) -> dict:
    for source in sources:
        cached = source / 'cached_sequence_prompts' / f'{chunk_id}.json'
        if cached.exists():
            return load(cached)
    path = SEQUENCE / 'requests' / f'{chunk_id}.json'
    if not path.exists():
        raise ValueError(f'Missing frozen sequence CR25 request: {chunk_id}')
    return load(path)


def merge_inventory(sources: list[Path]) -> list:
    by = {}
    for source in sources:
        for row in load(source / 'assay_inventory.json'):
            previous = by.get(row['group_id'])
            if previous is None or (previous.get('inference_status') != 'prepared' and row.get('inference_status') == 'prepared'):
                by[row['group_id']] = row
    if len(by) != 88:
        raise ValueError('Completed MEDIUM freeze is not the 88-assay N250 cohort')
    prepared = [row for row in by.values() if row.get('inference_status') == 'prepared']
    if len(prepared) != 88:
        raise ValueError('An inventory row is not prepared')
    if any(row.get('N') != 250 for row in prepared):
        raise ValueError('An assay is not N=250')
    return [by[key] for key in sorted(by)]


def clone(output: Path, level: str, proxy: str, concurrency: int, timeout: int,
          max_output_tokens=None) -> None:
    output = output.resolve()
    if output.exists() and (output / 'freeze.json').exists():
        raise ValueError(f'Already frozen: {output}')
    output.mkdir(parents=True, exist_ok=True)
    sources = [R4, R2]
    for source in sources:
        for name in COPY_DIRS:
            copy_tree(source, output, name)
    mapping = {}
    cache_sources = []
    for source in sources:
        mapping.update(load(source / 'selected_fact_mapping.json'))
        cache_sources.extend(load(source / 'cache_sources.json'))
    if len(mapping) != 22000:
        raise ValueError('Merged selected facts are not 88 x N250')
    snapshot = load(R4 / 'input_snapshot.json')
    snapshot['selected_fact_count'] = len(mapping)
    snapshot['selected_identity_sha256'] = digest(mapping)
    snapshot['labels_read'] = False
    save(output / 'input_snapshot.json', snapshot)
    save(output / 'selected_fact_mapping.json', mapping)
    save(output / 'cache_sources.json', cache_sources)
    inventory = merge_inventory(sources)
    index = {row['id']: row for row in load(SEQUENCE / 'index.json')
             if row['source'] in ('abagym', 'bindinggym')}
    plan = []
    for source in sources:
        for unit in load(source / 'plan.json'):
            if unit['condition'] not in ('msa_prompt', 'structure', 'msa_prompt_structure'):
                raise ValueError('Unexpected MEDIUM plan condition')
            request = retarget_request(load(source / unit['request_path']), level, max_output_tokens)
            path = output / 'requests' / Path(unit['request_path']).name
            save(path, request)
            cloned = dict(unit)
            cloned['request_path'] = str(path.relative_to(output))
            cloned['request_sha256'] = file_hash(path)
            cloned['semantic_request_sha256'] = digest(request)
            cloned['source_medium_request_sha256'] = unit['request_sha256']
            cloned['source_medium_dir'] = str(source)
            plan.append(cloned)
    assays = {row['group_id'] for row in inventory}
    sequence_chunks = [row for row in index.values() if row['group_id'] in assays]
    if len(sequence_chunks) != 880:
        raise ValueError('Frozen sequence CR25 partition is not 88 assays x 10 chunks')
    for chunk in sorted(sequence_chunks, key=lambda row: row['id']):
        request = retarget_request(sequence_request(chunk['id'], sources), level, max_output_tokens)
        unit_id = chunk['id'] + '__sequence'
        path = output / 'requests' / f'{unit_id}.json'
        save(path, request)
        plan.append(dict(id=unit_id, group_id=chunk['group_id'], source=chunk['source'],
                         condition='sequence', batch=chunk['batch'], expected_ids=chunk['expected_ids'],
                         request_path=str(path.relative_to(output)), request_sha256=file_hash(path),
                         semantic_request_sha256=digest(request), CR=25, N=250))
    if len(plan) != 3520:
        raise ValueError(f'Expected 88 assays x 10 CR x 4 arms = 3520, got {len(plan)}')
    config = load(R4 / 'config.json')
    config.update(cohort='executable_88', executable_assays=88, thinking_level=level,
                  proxy=proxy, concurrency=concurrency, timeout_seconds=timeout,
                  cloned_from=[str(R4), str(R2)],
                  clone_policy='Exact MEDIUM prompt bytes and CR25 partitions; only thinkingLevel changes; MEDIUM scores are not repeated',
                  sequence_arm='Frozen original sequence CR25 requests, rescored at this thinking level',
                  labels_policy=config['labels_policy'])
    save(output / 'config.json', config)
    save(output / 'assay_inventory.json', inventory)
    save(output / 'plan.json', plan)
    save(output / 'preflight_validation.json', dict(status='cloned_from_completed_medium',
                                                   thinking_level=level, planned_new_calls=len(plan),
                                                   arms=list(ARMS), assays=88, N=250, CR=25,
                                                   medium_not_rescored=True))
    files = sorted(path for path in output.rglob('*') if path.is_file() and path.name != 'freeze.json')
    code_paths = [CODE / 'tools/offline_ablation/run.py', CODE / 'tools/offline_ablation/features.py',
                  CODE / 'evagent/offline_ablation.py', ROOT / 'baseline/singlechain/models.py',
                  ROOT / 'baseline/singlechain/common.py', SEQUENCE / 'common.py']
    save(output / 'freeze.json', dict(artifact_hashes={str(path.relative_to(output)): file_hash(path) for path in files},
                                     code_hashes={str(path): file_hash(path) for path in code_paths}))
    print('CLONED', output, 'calls', len(plan), 'thinking', level, 'proxy', proxy, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--thinking-level', choices=('LOW', 'HIGH'), required=True)
    parser.add_argument('--proxy', required=True)
    parser.add_argument('--concurrency', type=int, required=True)
    parser.add_argument('--timeout-seconds', type=int, required=True)
    parser.add_argument('--max-output-tokens', type=int, choices=(16384, 65536),
                        help='Keep each MEDIUM request budget unless HIGH needs the completed 65536 cap')
    args = parser.parse_args()
    clone(args.output, args.thinking_level, args.proxy, args.concurrency, args.timeout_seconds,
          args.max_output_tokens)


if __name__ == '__main__':
    main()
