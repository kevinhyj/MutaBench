"""Strict, label-free helpers for frozen CR25 ablations.

Historical experiment scripts are left untouched. Rank fusion follows the
existing average-midrank percentile definition; no experimental labels enter
this module.
"""
from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from pathlib import Path
from typing import Any


MSA_FIELDS = (
    'p_wt', 'p_mutant', 'log_probability_ratio', 'weighted_valid_count',
    'weighted_gap_fraction', 'weighted_unknown_fraction', 'status',
)
STRUCTURE_FIELDS = (
    'min_partner_heavy_distance_A', 'partner_contact_count_5A',
    'sasa_bound_A2', 'delta_sasa_partner_A2', 'site_status',
    'partner_status', 'sasa_status', 'partner_chain_ids', 'source',
)


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def file_hash(path: Path) -> str:
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def finite(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def validate_scores(scores: dict, ids: list[str], *, bounded: bool = False) -> None:
    if len(ids) != len(set(ids)) or set(scores) != set(ids):
        raise ValueError('Scores must cover the exact unique frozen candidate set')
    if not all(finite(value) and (not bounded or -1 <= value <= 1)
               for value in scores.values()):
        raise ValueError('Missing, nonfinite, boolean or out-of-range score')


def midranks(values: list[float], *, descending: bool = False) -> list[float]:
    if not all(finite(value) for value in values):
        raise ValueError('Cannot rank missing/nonfinite values')
    order = sorted(range(len(values)), key=lambda index: values[index], reverse=descending)
    result = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        for index in order[start:end]:
            result[index] = (start + 1 + end) / 2
        start = end
    return result


def fuse_rankings(ids: list[str], gemini: dict, msa: dict) -> list[dict]:
    """Larger score/percentile is better; final rank 1 is best.

    Exact ties receive average midranks, including the final ranking. Missing
    values fail the complete-assay join. Weights are fixed, never label fitted.
    N=1 has undefined rank percentiles and is rejected.
    """
    if len(ids) < 2:
        raise ValueError('Rank percentiles require at least two candidates')
    validate_scores(gemini, ids)
    validate_scores(msa, ids)
    g_rank = midranks([gemini[i] for i in ids])
    m_rank = midranks([msa[i] for i in ids])
    g = [(rank - 1) / (len(ids) - 1) for rank in g_rank]
    m = [(rank - 1) / (len(ids) - 1) for rank in m_rank]
    # Half-integer midranks are exactly representable. Add them before the
    # common division so mathematically equal rank sums cannot split by ULPs.
    fusion = [(a + b - 2) / (2 * (len(ids) - 1)) for a, b in zip(g_rank, m_rank)]
    ranks = midranks(fusion, descending=True)
    return [dict(candidate_id=i, gemini_score=gemini[i], msa_score=msa[i],
                 gemini_rank_percentile=a, msa_rank_percentile=b,
                 fusion_value=f, final_rank=r)
            for i, a, b, f, r in zip(ids, g, m, fusion, ranks)]


def site_identity(group_id: str, site: dict) -> str:
    identity = dict(group_id=group_id, chain=site['chain'], position=site['position'],
                    insertion_code=site.get('insertion_code') or '', wt=site['wt'],
                    sequence_index_1based=site.get('sequence_index_1based'))
    return 'site_' + digest(identity)[:24]


def base_sites(task: dict) -> tuple[dict, dict]:
    """A WT locus has one ID across candidates, MSA and structure blocks."""
    mapping: dict[str, dict] = {}
    numbered: dict[int, set] = {}
    for candidate in task['candidates']:
        for site in candidate['sites']:
            sid = site_identity(task['group_id'], site)
            identity = {k: v for k, v in site.items() if k != 'mutant'}
            if sid in mapping and mapping[sid] != identity:
                raise ValueError('Conflicting WT identity for a shared site')
            mapping[sid] = identity
            numbered.setdefault(site['position'], set()).add(
                (site['chain'], site.get('insertion_code') or ''))
    ambiguous = {number for number, identities in numbered.items() if len(identities) > 1}
    rendered = {}
    for candidate in task['candidates']:
        rows = []
        for site in candidate['sites']:
            row = {key: site[key] for key in ('chain', 'position', 'wt', 'mutant')}
            row['site_id'] = site_identity(task['group_id'], site)
            if site.get('sequence_index_1based') is not None:
                row['sequence_index_1based'] = site['sequence_index_1based']
            if site['position'] in ambiguous or site.get('insertion_code'):
                row['insertion_code'] = site.get('insertion_code') or ''
                row['identity_status'] = 'full_chain_position_insertion_identity'
            rows.append(row)
        rendered[candidate['id']] = rows
    return mapping, rendered


def evidence_block(ids: list[str], base: dict, evidence: dict, fields: tuple) -> dict:
    if set(evidence) != set(base):
        raise ValueError('Evidence candidate identity differs from the base task')
    result = {}
    for candidate_id in ids:
        rows = []
        if set(evidence[candidate_id]) != {site['site_id'] for site in base[candidate_id]}:
            raise ValueError('Evidence site identity differs from the base candidate')
        for site in base[candidate_id]:
            sid = site['site_id']
            if sid not in evidence[candidate_id]:
                raise ValueError(f'Missing evidence identity: {candidate_id}/{sid}')
            item = evidence[candidate_id][sid]
            missing = set(fields) - set(item)
            if missing:
                raise ValueError(f'Missing evidence fields: {sorted(missing)}')
            if fields == MSA_FIELDS:
                for key in MSA_FIELDS[:-1]:
                    if item[key] is not None and not finite(item[key]):
                        raise ValueError(f'Nonfinite MSA evidence: {key}')
                for key in ('p_wt', 'p_mutant', 'weighted_gap_fraction', 'weighted_unknown_fraction'):
                    if item[key] is not None and not 0 <= item[key] <= 1:
                        raise ValueError(f'Invalid probability/fraction: {key}')
                if item['weighted_valid_count'] is not None and item['weighted_valid_count'] < 0:
                    raise ValueError('Negative weighted valid count')
                if item['status'] in ('ok', 'global_neff_below_frozen_20'):
                    if (any(item[key] is None for key in MSA_FIELDS[:-1]) or
                            item['p_wt'] <= 0 or item['p_mutant'] <= 0 or
                            item['weighted_valid_count'] <= 0):
                        raise ValueError('Available MSA evidence has missing/invalid values')
                    expected = math.log(item['p_mutant'] / item['p_wt'])
                    if not math.isclose(item['log_probability_ratio'], expected, abs_tol=1e-10):
                        raise ValueError('MSA log ratio differs from the frozen probabilities')
                    if item['weighted_gap_fraction'] + item['weighted_unknown_fraction'] > 1 + 1e-12:
                        raise ValueError('Gap and unknown fractions exceed one')
            elif fields == STRUCTURE_FIELDS:
                for key in STRUCTURE_FIELDS[:4]:
                    if item[key] is not None and (not finite(item[key]) or item[key] < -1e-6):
                        raise ValueError(f'Invalid structure evidence: {key}')
                count = item['partner_contact_count_5A']
                if count is not None and (type(count) is not int or count < 0):
                    raise ValueError('Partner contacts must be a nonnegative integer or null')
                if item['sasa_status'] == 'ok' and any(item[key] is None for key in ('sasa_bound_A2', 'delta_sasa_partner_A2')):
                    raise ValueError('Available SASA has missing values')
                if item['sasa_status'] != 'ok' and any(item[key] is not None for key in ('sasa_bound_A2', 'delta_sasa_partner_A2')):
                    raise ValueError('Unavailable SASA must be explicit null')
                geometry_ok = item['site_status'] == 'ok' and item['partner_status'] == 'ok'
                if geometry_ok != all(item[key] is not None for key in STRUCTURE_FIELDS[:2]):
                    raise ValueError('Geometry values and availability statuses disagree')
                if not geometry_ok and any(item[key] is not None for key in STRUCTURE_FIELDS[:4]):
                    raise ValueError('Unavailable geometry must be explicit null')
                if not isinstance(item['partner_chain_ids'], list) or not item['source']:
                    raise ValueError('Structure partner identity and source are required')
            row = dict(site)
            row.update({key: item[key] for key in fields})
            rows.append(row)
        result[candidate_id] = rows
    # Nonfinite numbers must be represented as explicit null before rendering.
    json.dumps(result, allow_nan=False)
    return result


def render_evidence(task: dict, ids: list[str], features: dict, condition: str) -> dict:
    if condition not in ('msa_prompt', 'structure', 'msa_prompt_structure'):
        raise ValueError('Ranking fusion is postprocessing, not a prompt condition')
    _, sites = base_sites(task)
    if not ids or len(ids) != len(set(ids)) or set(ids) - set(sites):
        raise ValueError('Evidence IDs must be a nonempty unique subset of the base task')
    if not ids or len(set(ids)) != len(ids) or not set(ids) <= set(sites):
        raise ValueError('Prompt IDs must be a unique nonempty subset of base candidates')
    output = {}
    if condition in ('msa_prompt', 'msa_prompt_structure'):
        required_chains = sorted({site['chain'] for cid in ids for site in sites[cid]})
        for chain in required_chains:
            metadata = features['msa_chains'][chain]
            count, neff = metadata['homolog_count'], metadata['global_neff']
            if count is not None and (type(count) is not int or count < 0):
                raise ValueError('Homolog count must be a nonnegative integer or null')
            if neff is not None and (not finite(neff) or neff < 0):
                raise ValueError('Global Neff must be finite nonnegative or null')
        output['msa_chains'] = [dict(chain=chain,
                                    homolog_count=features['msa_chains'][chain]['homolog_count'],
                                    global_neff=features['msa_chains'][chain]['global_neff'])
                                for chain in required_chains]
        for chain in output['msa_chains']:
            count, neff = chain['homolog_count'], chain['global_neff']
            if count is not None and (type(count) is not int or count < 0):
                raise ValueError('Invalid homolog count')
            if neff is not None and (not finite(neff) or neff < 0):
                raise ValueError('Invalid global Neff')
        output['msa_sites'] = evidence_block(ids, sites, features['msa'], MSA_FIELDS)
    if condition in ('structure', 'msa_prompt_structure'):
        output['structure_definition'] = features['structure_definition']
        output['structure_sites'] = evidence_block(ids, sites, features['structure'], STRUCTURE_FIELDS)
    return output


def matching_cache(unit: dict, task_path: Path, request_path: Path,
                   response_path: Path, expected_request: dict,
                   *, expected_model: str = 'gemini-3.8-flash') -> tuple[dict, dict]:
    """Reject altered candidates, assay identity, prompt, config or response."""
    if file_hash(task_path) != unit['task_sha256']:
        raise ValueError('Frozen base task hash mismatch')
    if file_hash(request_path) != unit['request_sha256']:
        raise ValueError('Frozen request file hash mismatch')
    actual_request = json.loads(request_path.read_text())
    if actual_request != expected_request:
        raise ValueError('Prompt or generation configuration mismatch')
    task = json.loads(task_path.read_text())
    if task['group_id'] != unit['group_id'] or task['source'] != unit['source']:
        raise ValueError('Assay/source mismatch')
    selected = [candidate['id'] for candidate in task['candidates']
                if candidate['id'] in set(unit['expected_ids'])]
    if selected != unit['expected_ids']:
        raise ValueError('Candidate set/order mismatch')
    response = json.loads(response_path.read_text())
    if response.get('unit_id') != unit['id'] or response.get('status') != 'ok':
        raise ValueError('Unsuccessful or mismatched cached response')
    original_hash = unit.get('original_request_sha256')
    if original_hash is not None:
        if (response.get('request_sha256') != original_hash or
                response.get('actual_request_sha256') != unit['request_sha256']):
            raise ValueError('Amended request lineage or actual configuration hash mismatch')
    else:
        if response.get('request_sha256') != unit['request_sha256']:
            raise ValueError('Cached response request hash mismatch')
        if response.get('actual_request_sha256') not in (None, unit['request_sha256']):
            raise ValueError('Amended configuration requires its own cache identity')
    validate_scores(response['scores'], unit['expected_ids'], bounded=True)
    if not str(response.get('model', '')).startswith(expected_model):
        raise ValueError('Cached model differs from the frozen experiment')
    origin = dict(
        source_response=str(response_path), source_response_sha256=file_hash(response_path),
        source_request=str(request_path), source_request_sha256=unit['request_sha256'],
        source_task=str(task_path), source_task_sha256=unit['task_sha256'],
        model=response['model'], match='exact_assay_candidates_prompt_configuration',
    )
    if original_hash is not None:
        origin['original_request_sha256'] = original_hash
    return response['scores'], origin


def matching_fewshot_cache(plan_path: Path, episode_dir: Path, request_path: Path,
                          result_path: Path, ledger_paths: list[Path],
                          expected_model: dict) -> tuple[dict, dict]:
    """Validate a historical CR25 result against its freeze and actual wire ledger.

    Candidate identity includes the frozen complete sequences and their order.
    A result JSON alone is insufficient evidence of the generation configuration.
    Ledgers are opened read-only, including for results carried across recoveries.
    """
    from .integrated.chunked_scoring import ScoringResponseValidator
    from .integrated.gateway import public_model_config

    plan = json.loads(plan_path.read_text())
    if digest({k: v for k, v in plan.items() if k != 'fingerprint'}) != plan['fingerprint']:
        raise ValueError('Historical plan fingerprint mismatch')
    if plan['query_chunk_size'] != 25 or public_model_config(expected_model) != plan['model_config']:
        raise ValueError('CR or model configuration mismatch')
    episode = json.loads((episode_dir / 'episode.json').read_text())
    eid = episode['episode_id']
    for name in ('episode.json', 'support.json'):
        if file_hash(episode_dir / name) != plan['episode_input_hashes'][eid][name]:
            raise ValueError('Frozen episode/support changed')
    request = json.loads(request_path.read_text())
    if request['request_hash'] != digest({k: v for k, v in request.items() if k != 'request_hash'}):
        raise ValueError('Historical prompt identity mismatch')
    record = next((r for r in plan['requests'] if r['request_hash'] == request['request_hash']), None)
    if record is None or any(record[k] != request[k] for k in
                             ('episode_id', 'condition', 'chunk_index', 'expected_ids', 'support_ids')):
        raise ValueError('Request not in the frozen plan')
    selected = episode['query'][25 * request['chunk_index']:25 * (request['chunk_index'] + 1)]
    if (request['episode_id'] != eid or request['expected_ids'] != [r['variant_id'] for r in selected]
            or request['condition']['name'] != 'numeric_0' or request['support_ids']):
        raise ValueError('Assay/candidate/order/zero-shot mismatch')
    result = json.loads(result_path.read_text())
    if result.get('status') != 'ok' or result.get('execution_mode') != 'live':
        raise ValueError('Cached result is not a successful live prediction')
    for key in ('request_hash', 'episode_id', 'condition', 'chunk_index', 'expected_ids'):
        if result.get(key) != request[key]:
            raise ValueError('Cached result identity mismatch')
    validate_scores(result['value'], request['expected_ids'], bounded=True)
    origins = []
    for index, attempt in enumerate(result['attempts']):
        found = None
        for ledger in ledger_paths:
            if not ledger.exists():
                continue
            with sqlite3.connect(ledger.resolve().as_uri() + '?mode=ro', uri=True) as db:
                row = db.execute('SELECT request_json,response_json,status FROM attempts WHERE request_sha256=?',
                                 (attempt['request_sha256'],)).fetchone()
            if row:
                found = (ledger, row)
                break
        if found is None:
            raise ValueError('Actual provider request ledger is missing')
        ledger, (wire_json, response_json, status) = found
        wire, response = json.loads(wire_json), json.loads(response_json)
        if digest(wire) != attempt['request_sha256'] or wire['model_config'] != plan['model_config']:
            raise ValueError('Actual generation configuration differs from frozen cache')
        if wire['fingerprint'] != plan['fingerprint']:
            raise ValueError('Provider request belongs to a different plan')
        messages = wire['messages']
        if messages[0] != dict(role='system', content=request['system']):
            raise ValueError('Actual system prompt mismatch')
        if index == 0 and messages[1] != dict(role='user', content=request['user']):
            raise ValueError('Actual initial prompt mismatch')
        if hashlib.sha256(messages[1]['content'].encode()).hexdigest() != attempt['attempt_prompt_sha256']:
            raise ValueError('Actual attempt prompt checksum mismatch')
        if response.get('final_text') != attempt.get('final_text'):
            raise ValueError('Cached response differs from the provider ledger')
        origins.append(dict(ledger=str(ledger), provider_request_sha256=attempt['request_sha256'],
                            provider_status=status, wire_request=wire))
    if not origins:
        raise ValueError('Missing provider attempts')
    validated = ScoringResponseValidator(request['expected_ids']).validate(result['attempts'][-1]['final_text'])
    if not validated.valid or validated.value != result['value']:
        raise ValueError('Cached scores differ from the validated raw response')
    return result['value'], dict(
        match='exact_frozen_assay_candidates_order_prompt_and_actual_generation_configuration',
        plan=str(plan_path), plan_sha256=file_hash(plan_path), model_config=plan['model_config'],
        source_episode=str(episode_dir / 'episode.json'),
        source_episode_sha256=file_hash(episode_dir / 'episode.json'),
        source_request=str(request_path), source_request_sha256=file_hash(request_path),
        source_response=str(result_path), source_response_sha256=file_hash(result_path),
        provider_attempts=origins,
    )
