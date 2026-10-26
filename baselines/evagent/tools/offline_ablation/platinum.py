"""Isolated CR25 rerun of the admitted existing Platinum N10 candidates.

Predictors never read evaluation_labels.json. The analyze command is separate.
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures
import csv
import hashlib
import json
import math
import os
import sqlite3
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE))
from evagent.offline_ablation import (base_sites, digest, file_hash, fuse_rankings, matching_fewshot_cache,
                                      midranks, render_evidence, validate_scores)
from evagent.integrated.chunked_scoring import ScoringResponseValidator
from evagent.integrated.research.concurrent_gateway import ConcurrentGateway
from evagent.integrated.self_check import SelfCheckPolicy, run_self_check
from ligand_features import LigandFeatureBuilder
from features import save

PROJECT = CODE.parent
HISTORY = PROJECT / 'work/final_mutation_gemini_cr25_seed42_20260928'
DEFAULT_OUT = PROJECT / 'experiments/platinum_n10_cr25_ablation_20261004_v3'
EXCLUDED = {
    'platinum_02137aa4e9133b0f': 'pathogen_antimalarial_inhibitor_binding',
    'platinum_aac36e36dbbd214d': 'pathogen_antimalarial_inhibitor_binding',
    'platinum_f2325d022ba9a6bd': 'pathogen_antimalarial_inhibitor_binding',
    'platinum_5e2dc789239e48b0': 'toxin_activation_ligand_binding',
    'platinum_677a123adc4248bd': 'pathogen_antifungal_inhibitor_binding',
}
CONDITIONS = ('msa_prompt', 'structure', 'msa_prompt_structure')


def load(path):
    return json.loads(Path(path).read_text())


def write_report(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    os.replace(temporary, path)


def source_files():
    return [Path(__file__), CODE / 'evagent/offline_ablation.py',
            Path(__file__).with_name('features.py'), Path(__file__).with_name('ligand_features.py'),
            PROJECT / 'baseline/singlechain/models.py',
            CODE / 'evagent/integrated/gateway.py',
            CODE / 'evagent/integrated/research/concurrent_gateway.py',
            CODE / 'evagent/integrated/chunked_scoring.py',
            CODE / 'evagent/integrated/self_check.py']


def credential_value(path):
    if path.stat().st_mode & 0o077:
        raise ValueError('Credential permissions are too broad')
    content = path.read_text().strip()
    entries = [line.strip().removeprefix('export ').split('=', 1)[1].strip()
               for line in content.splitlines()
               if line.strip().removeprefix('export ').startswith('GEMINI_API_KEY=')]
    value = entries[0].strip('\"').strip("'") if len(entries) == 1 else content
    if len(entries) > 1 or not value or any(c.isspace() for c in value):
        raise ValueError('Credential unavailable or ambiguous')
    return value


def prompt_user(base_request, evidence):
    user = base_request['user'] + '\n\nFrozen additional evidence:\n' + json.dumps(
        evidence, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
    if 'msa_sites' in evidence:
        user += '\nMSA probabilities are weighted empirical amino-acid frequencies with pseudocounts. They are statistical evidence from the supplied alignment, not probabilities of homology or measured binding outcomes.'
    return user + '\nJSON null means unavailable, never zero.'


def fusion_artifact(ids, scores, features, origin, *, condition='postprocessing_only'):
    msa = features['independent_msa_scores']
    if features['full_msa_coverage']:
        rows = fuse_rankings(ids, scores, msa)
        status = 'complete'
    else:
        # Keep every frozen candidate. Never rank the available MSA subset.
        ranks = midranks([scores[i] for i in ids])
        rows = [dict(candidate_id=i, gemini_score=scores[i], msa_score=msa[i],
                     gemini_rank_percentile=(r - 1) / (len(ids) - 1),
                     msa_rank_percentile=None, fusion_value=None, final_rank=None)
                for i, r in zip(ids, ranks)]
        status = 'incomplete_MSA'
    for row in rows:
        row['cache_source'] = origin['source_response']
        row['cache_source_sha256'] = origin['source_response_sha256']
    return dict(condition=condition, status=status, rows=rows, new_calls=0,
                missing_candidate_ids=[i for i in ids if msa[i] is None],
                weights=dict(gemini=0.5, msa=0.5))


def extract_task(folder, facts):
    from Bio.PDB import PDBParser
    from Bio.SeqUtils import seq1
    episode, manifest = load(folder / 'episode.json'), load(folder / 'input_manifest.json')
    path = PROJECT / manifest['structure_path']
    if file_hash(path) != manifest['structure_sha256']:
        raise ValueError('Structure migration checksum mismatch')
    selected = [facts[fid] for fid in manifest['sampled_fact_ids']]
    if len(selected) != 10 or len(episode['query']) != 10:
        raise ValueError('Original N10 freeze required')
    chains = []
    for chain in next(PDBParser(QUIET=True).get_structure('WT', path).get_models()):
        residues = [r for r in chain if r.id[0] == ' ' and seq1(r.resname, custom_map={'MSE': 'M', 'SEC': 'C', 'PYL': 'K'}, undef_code='X') != 'X']
        if residues:
            chains.append(dict(chain=chain.id, resolved_sequence=''.join(seq1(r.resname, custom_map={'MSE': 'M', 'SEC': 'C', 'PYL': 'K'}) for r in residues),
                               residue_ids=[str(r.id[1]) + r.id[2].strip() for r in residues]))
    chain_map = {c['chain']: c for c in chains}
    wt = ' | '.join(f"{ch}={chain_map[ch]['resolved_sequence']}" for ch in sorted(chain_map))
    if wt != episode['wt_sequence']:
        raise ValueError('Base WT chain input changed')
    candidates = []
    for candidate, fact in zip(episode['query'], selected, strict=True):
        if fact['assay_group_id'] != manifest['assay_group_id'] or fact['reference_complex_id'] != manifest['reference_complex_id']:
            raise ValueError('Source fact belongs to a different assay')
        sites, mutated = [], {ch: list(c['resolved_sequence']) for ch, c in chain_map.items()}
        for m in fact['chain_mutations']:
            ch, ins = str(m['chain']), m.get('insertion_code') or ''
            coordinate = str(m['position']) + ins
            index = chain_map[ch]['residue_ids'].index(coordinate)
            if mutated[ch][index] != m['wt_aa']:
                raise ValueError('Candidate WT identity mismatch')
            mutated[ch][index] = m['mutant_aa']
            sites.append(dict(chain=ch, position=m['position'], insertion_code=ins, wt=m['wt_aa'],
                              mutant=m['mutant_aa'], sequence_index_1based=index + 1))
        sequence = ' | '.join(f"{ch}={''.join(mutated[ch])}" for ch in sorted(mutated))
        if sequence != candidate['sequence']:
            raise ValueError('Frozen candidate sequence reconstruction differs')
        candidates.append(dict(id=candidate['variant_id'], fact_id=fact['fact_id'], sites=sites))
    ligand_ids = {r['context']['ligand_id'] for r in selected}
    if len(ligand_ids) != 1:
        raise ValueError('Ambiguous annotated ligand')
    return dict(group_id=manifest['assay_group_id'], episode_id=episode['episode_id'], source='platinum',
                reference_complex_id=manifest['reference_complex_id'], wt_chains=chains, candidates=candidates,
                structure_source=dict(path=str(path), sha256=file_hash(path)),
                binding_partner=dict(pdb_ligand_id=next(iter(ligand_ids))),
                source_episode_sha256=file_hash(folder / 'episode.json'),
                source_manifest_sha256=file_hash(folder / 'input_manifest.json'))


def find_cache(folder, model):
    ledgers = [HISTORY / 'provider/request_ledger.sqlite', *[p / 'provider/request_ledger.sqlite' for p in sorted(HISTORY.glob('execution_v*'))]]
    failures = []
    for root in [*reversed(sorted(HISTORY.glob('execution_v*'))), HISTORY]:
        if not (root / 'plan.json').exists():
            continue
        plan = load(root / 'plan.json')
        for record in plan['requests']:
            if record['episode_id'] != folder.name or record['condition']['name'] != 'numeric_0':
                continue
            request_path = root / 'predictions' / record['request_hash'] / 'request.json'
            result_path = request_path.with_name('result.json')
            if not request_path.exists() or not result_path.exists():
                continue
            try:
                scores, origin = matching_fewshot_cache(root / 'plan.json', folder, request_path, result_path, ledgers, model)
                origin['rejected_cache_candidates'] = failures
                return scores, origin, load(request_path)
            except (ValueError, KeyError) as error:
                failures.append(dict(path=str(result_path), reason=str(error)))
    raise ValueError(f'No exactly matched cache: {failures}')


def prepare(out):
    if out.exists() and any(out.iterdir()):
        raise ValueError('Preparation requires a new empty output directory; use validate/run to resume')
    model = load(HISTORY / 'execution_v6/plan.json')['model_config']
    source = PROJECT / 'data/final_mutation_data/platinum/canonical/platinum_facts_N10.jsonl'
    facts = {r['fact_id']: r for r in map(json.loads, source.read_text().splitlines())}
    builder = LigandFeatureBuilder(PROJECT, out)
    queue, assays = [], []
    for folder in sorted((HISTORY / 'episodes').glob('platinum_*')):
        manifest = load(folder / 'input_manifest.json')
        eid = folder.name
        record = dict(assay_id=eid, reference_complex_id=manifest['reference_complex_id'],
                      N=10, group_id=manifest['assay_group_id'])
        if eid in EXCLUDED:
            record.update(status='not_run_scope_boundary', reason=EXCLUDED[eid])
            assays.append(record)
            continue
        if file_hash(source) != manifest['source_sha256']:
            raise ValueError('N10 source dataset changed')
        task = extract_task(folder, facts)
        target = out / 'assays' / eid
        _, task['base_sites'] = base_sites(task)
        save(target / 'task.json', task)
        scores, origin, base_request = find_cache(folder, model)
        save(target / 'cache_origin.json', origin)
        save(target / 'base_prompt.json', base_request)
        save(target / 'base_scores.json', scores)
        features, mapping = builder.build_ligand(task)
        save(target / 'features.json', features)
        save(target / 'site_mapping.json', mapping)
        ids = [c['id'] for c in task['candidates']]
        save(target / 'msa_gemini_rank5050.json', fusion_artifact(ids, scores, features, origin))
        for condition in CONDITIONS:
            evidence = render_evidence(task, ids, features, condition)
            evidence['base_sites'] = task['base_sites']
            # The sequence/task/score-scale instruction remains byte-identical.
            # Identity rows associate evidence with original frozen mutations.
            user = prompt_user(base_request, evidence)
            request = dict(assay_id=eid, condition=condition, chunk_index=0, CR=25, expected_ids=ids,
                           system=base_request['system'], user=user, evidence=evidence)
            request['request_hash'] = digest(request)
            save(out / 'requests' / f"{request['request_hash']}.json", request)
            queue.append(dict(request_hash=request['request_hash'], assay_id=eid, condition=condition))
        record.update(status='prepared', msa_complete=features['full_msa_coverage'],
                      msa_errors=features['source_provenance']['profile_errors'],
                      cache_source=origin['source_response'])
        assays.append(record)
        print(json.dumps(dict(event='prepared', assay=eid, msa_complete=record['msa_complete'])), flush=True)
    save(out / 'queue.json', queue)
    for source_path in source_files():
        target = out / 'code_snapshot' / source_path.relative_to(PROJECT)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source_path.read_bytes())
    inputs = {str(p.relative_to(out)): file_hash(p) for p in sorted(out.rglob('*')) if p.is_file()
              if p.name not in ('environment_repair.json', 'plan.json')}
    plan = dict(schema='platinum-n10-cr25-ablation-v2', CR=25, N=10, sample_seed=42, generation_seed=42,
                model_config=model, conditions=list(CONDITIONS),
                postprocessing_conditions=['msa_gemini_rank5050', 'msa_gemini_rank5050_structure'],
                postprocessing_condition='msa_gemini_rank5050', weights=dict(gemini=0.5, msa=0.5),
                fusion_definition='Per-assay ascending average midrank percentile (rank-1)/(N-1); larger is better; final descending average midrank; missing component rejects complete fusion',
                fusion_reference=str(PROJECT / 'work/structure_msa_fusion_evaluate_20260916.py'),
                fusion_reference_sha256=file_hash(PROJECT / 'work/structure_msa_fusion_evaluate_20260916.py'),
                msa_definition='Existing singlechain MSA protocol: sorted unique non-WT unambiguous homologs <=2048; inverse 80% identity neighbors with >=50% overlap; pseudocount total 1; min global Neff 20; sum site log ratios',
                assays=assays, input_hashes=inputs, source_hashes={str(p): file_hash(p) for p in source_files()},
                original_dataset_sha256=file_hash(source), planned_new_requests=len(queue),
                reused_scoring_requests=sum(a['status'] == 'prepared' for a in assays),
                token_budget=5_000_000, workers=4, max_repairs=1,
                labels_read_by_predictor=False, originals_modified=False)
    plan['fingerprint'] = digest(plan)
    save(out / 'plan.json', plan)
    validate(out)


def validate(out):
    plan = load(out / 'plan.json')
    if digest({k: v for k, v in plan.items() if k != 'fingerprint'}) != plan['fingerprint']:
        raise ValueError('Plan changed')
    for path, expected in plan['source_hashes'].items():
        if file_hash(Path(path)) != expected:
            raise ValueError('Code changed after freeze: ' + path)
    for path, expected in plan['input_hashes'].items():
        if file_hash(out / path) != expected:
            raise ValueError('Frozen input changed: ' + path)
    if (plan['CR'] != 25 or plan['N'] != 10 or plan['generation_seed'] != 42 or
            plan['weights'] != dict(gemini=0.5, msa=0.5) or plan['conditions'] != list(CONDITIONS)):
        raise ValueError('Frozen scientific protocol differs')
    source = PROJECT / 'data/final_mutation_data/platinum/canonical/platinum_facts_N10.jsonl'
    if file_hash(source) != plan['original_dataset_sha256']:
        raise ValueError('Original dataset changed')
    facts = {r['fact_id']: r for r in map(json.loads, source.read_text().splitlines())}
    prepared = {a['assay_id'] for a in plan['assays'] if a['status'] == 'prepared'}
    if len(plan['assays']) != 15 or len({a['assay_id'] for a in plan['assays']}) != 15:
        raise ValueError('Expected all 15 distinct assay status records')
    for eid in sorted(prepared):
        target = out / 'assays' / eid
        task = extract_task(HISTORY / 'episodes' / eid, facts)
        _, task['base_sites'] = base_sites(task)
        if task != load(target / 'task.json'):
            raise ValueError('Original assay/candidate/site identity changed')
        origin = load(target / 'cache_origin.json')
        scores, fresh_origin = matching_fewshot_cache(
            Path(origin['plan']), HISTORY / 'episodes' / eid,
            Path(origin['source_request']), Path(origin['source_response']),
            list(dict.fromkeys(Path(a['ledger']) for a in origin['provider_attempts'])), plan['model_config'])
        if scores != load(target / 'base_scores.json') or any(
                fresh_origin[k] != origin[k] for k in fresh_origin):
            raise ValueError('Historical scoring cache provenance changed')
        features = load(target / 'features.json')
        for candidate in task['candidates']:
            rows = list(features['msa'][candidate['id']].values())
            expected_score = 0.0
            for r in rows:
                if r['status'] != 'ok':
                    expected_score = None
                    break
                expected_score += r['log_probability_ratio']
            if features['independent_msa_scores'][candidate['id']] != expected_score:
                raise ValueError('Independent MSA score differs from the frozen site log ratios')
        if fusion_artifact([c['id'] for c in task['candidates']], scores, features, origin) != load(target / 'msa_gemini_rank5050.json'):
            raise ValueError('Fusion components/replay differ')
        for metadata in features['source_provenance']['msa_chains'].values():
            for key, path in [('msa', metadata['msa_path']), ('mapping', metadata['mapping_path'])]:
                if file_hash(Path(path)) != metadata['source_hashes'][key]:
                    raise ValueError('Original MSA/mapping source changed')
            table = load(Path(metadata['probability_table']))
            if table['source_hashes'] != metadata['source_hashes']:
                raise ValueError('Frozen probability table source identity differs')
            weights_path = out / 'msa_weights' / (Path(metadata['probability_table']).stem + '.npz')
            if file_hash(weights_path) != table['weights_sha256']:
                raise ValueError('Frozen MSA weights changed')
            for column in table['rows']:
                probabilities = column['probabilities_20']
                if len(probabilities) != 20 or not all(0 < x <= 1 for x in probabilities) or not math.isclose(sum(probabilities), 1, abs_tol=1e-12):
                    raise ValueError('Invalid frozen 20-amino-acid probability table')
    queue = load(out / 'queue.json')
    if len(queue) != 3 * len(prepared) or collections.Counter(
            (u['assay_id'], u['condition']) for u in queue) != collections.Counter(
                (eid, condition) for eid in prepared for condition in CONDITIONS):
        raise ValueError('Queue must cover exactly the three changed-prompt conditions')
    for unit in queue:
        if unit['assay_id'] in EXCLUDED:
            raise ValueError('Excluded endpoint entered inference queue')
        request = load(out / 'requests' / f"{unit['request_hash']}.json")
        if digest({k: v for k, v in request.items() if k != 'request_hash'}) != unit['request_hash']:
            raise ValueError('Request identity changed')
        target = out / 'assays' / unit['assay_id']
        task, features = load(target / 'task.json'), load(target / 'features.json')
        expected_evidence = render_evidence(task, request['expected_ids'], features, unit['condition'])
        expected_evidence['base_sites'] = task['base_sites']
        if request['evidence'] != expected_evidence:
            raise ValueError('Evidence renderer replay differs')
        base_request = load(target / 'base_prompt.json')
        if (request['user'] != prompt_user(base_request, expected_evidence) or
                request['system'] != base_request['system'] or request['condition'] != unit['condition'] or
                request['assay_id'] != unit['assay_id'] or request['chunk_index'] != 0):
            raise ValueError('Actual prompt/condition differs from the frozen contract')
        text = json.dumps(request['evidence'])
        if any(key in text for key in ('probabilities_20', 'msa_column', 'msa_source_hashes')):
            raise ValueError('Audit-only MSA fields entered the main prompt')
        if request['CR'] != 25 or request['expected_ids'] != [c['id'] for c in task['candidates']]:
            raise ValueError('Candidate freeze changed')
    save(out / 'validation.json', dict(status='ok', assays=15, prepared_assays=plan['reused_scoring_requests'],
                                       new_prompt_units=len(queue), labels_read=False, CR=25))
    print(json.dumps(dict(event='validation_ok', units=len(queue))), flush=True)
    return plan, queue


def validated_result(out, unit, plan):
    request = load(out / 'requests' / f"{unit['request_hash']}.json")
    result_path = out / 'results' / f"{unit['request_hash']}.json"
    result = load(result_path)
    for key in ('request_hash', 'assay_id', 'condition', 'expected_ids'):
        if result.get(key) != request[key]:
            raise ValueError('Cached changed-prompt result identity mismatch')
    if result.get('model_config') != plan['model_config'] or result.get('fingerprint') != plan['fingerprint']:
        raise ValueError('Cached changed-prompt generation configuration mismatch')
    if result['status'] != 'ok':
        return result
    validate_scores(result['value'], request['expected_ids'], bounded=True)
    attempts = result.get('attempts') or []
    if not attempts:
        raise ValueError('Scoring result has no provider attempts')
    ledger = out / 'provider/request_ledger.sqlite'
    with sqlite3.connect(ledger.resolve().as_uri() + '?mode=ro', uri=True) as db:
        for index, attempt in enumerate(attempts):
            row = db.execute('SELECT request_json,response_json,status FROM attempts WHERE request_sha256=?',
                             (attempt['request_sha256'],)).fetchone()
            if not row or row[2] != 'ok':
                raise ValueError('Successful scoring provider ledger missing')
            wire, raw = map(json.loads, row[:2])
            if (digest(wire) != attempt['request_sha256'] or wire['model_config'] != plan['model_config'] or
                    wire['fingerprint'] != plan['fingerprint'] or
                    wire['messages'][0] != dict(role='system', content=request['system']) or
                    wire['metadata'].get('request_hash') != unit['request_hash'] or
                    wire['metadata'].get('assay_id') != unit['assay_id'] or
                    wire['metadata'].get('condition') != unit['condition']):
                raise ValueError('Actual changed-prompt provider request differs')
            if index == 0 and wire['messages'][1] != dict(role='user', content=request['user']):
                raise ValueError('Actual initial prompt mismatch')
            if hashlib.sha256(wire['messages'][1]['content'].encode()).hexdigest() != attempt['attempt_prompt_sha256']:
                raise ValueError('Actual attempt prompt checksum mismatch')
            if raw.get('final_text') != attempt.get('final_text'):
                raise ValueError('Scoring raw response differs from the ledger')
            if not str((raw.get('response_metadata') or {}).get('model_version', '')).startswith(plan['model_config']['model']):
                raise ValueError('Actual provider model identity mismatch')
    parsed = ScoringResponseValidator(request['expected_ids']).validate(attempts[-1]['final_text'])
    if not parsed.valid or parsed.value != result['value']:
        raise ValueError('Scores differ from the validated raw provider response')
    return result


def run(out, credential_path, *, pilot=False):
    plan, queue = validate(out)
    os.environ[plan['model_config']['api_key_env']] = credential_value(credential_path)
    gateway = ConcurrentGateway(model_config=plan['model_config'], run_dir=out / 'provider',
                                fingerprint=plan['fingerprint'], token_budget=plan['token_budget'],
                                reserve_budget=True, max_concurrent_requests=plan['workers'])
    def one(unit):
        key = unit['request_hash']
        target = out / 'results' / f'{key}.json'
        if target.exists():
            return validated_result(out, unit, plan)
        request = load(out / 'requests' / f'{key}.json')
        class PromptBlocked(Exception):
            pass
        attempts = []
        def generate(prompt, attempt):
            response = gateway.generate(system=request['system'], user=prompt, run_name='platinum_cr25_ablation',
                                        metadata=dict(assay_id=request['assay_id'], condition=request['condition'],
                                                      chunk_index=0, attempt=attempt, request_hash=key))
            attempts.append(response)
            if (response.get('response_metadata') or {}).get('prompt_feedback', {}).get('block_reason'):
                raise PromptBlocked()
            if response.get('status') == 'ok' and not str((response.get('response_metadata') or {}).get('model_version', '')).startswith(plan['model_config']['model']):
                raise ValueError('Actual provider model identity mismatch')
            return response
        try:
            result = run_self_check(original_prompt=request['user'], generate=generate,
                                    validator=ScoringResponseValidator(request['expected_ids']),
                                    policy=SelfCheckPolicy(max_repairs=plan['max_repairs']))
        except PromptBlocked:
            result = dict(status='provider_blocked', value=None, attempts=attempts)
        except Exception as error:
            result = dict(status='error', value=None, error_type=type(error).__name__, error=str(error), attempts=attempts)
        result.update(request_hash=key, assay_id=request['assay_id'], condition=request['condition'],
                      expected_ids=request['expected_ids'], model_config=plan['model_config'],
                      fingerprint=plan['fingerprint'])
        if result['status'] == 'ok':
            try:
                validate_scores(result['value'], request['expected_ids'], bounded=True)
            except ValueError as error:
                result.update(status='invalid_score_range', value=None, error=str(error))
        save(target, result)
        if result['status'] == 'ok':
            validated_result(out, unit, plan)
        print(json.dumps(dict(event='settled', assay=request['assay_id'], condition=request['condition'], status=result['status'])), flush=True)
        return result
    try:
        first = one(queue[0])
        if first['status'] == 'ok' and not pilot:
            with concurrent.futures.ThreadPoolExecutor(max_workers=plan['workers']) as pool:
                list(pool.map(one, queue[1:]))
    finally:
        write_report(out / 'usage.json', gateway.summary())
        gateway.close()


def analyze(out):
    plan, queue = validate(out)
    _evaluate(out, out, plan, queue)


def publish_fusion(source, dest):
    """Write the structure/MSA fusion into a new directory. Does not call a provider or edit source."""
    if dest.resolve() == source.resolve():
        raise ValueError('Structure fusion publish requires a different output directory')
    if dest.exists() and any(dest.iterdir()):
        raise ValueError('Publish requires a new empty output directory')
    plan = load(source / 'plan.json')
    for path, expected in plan['input_hashes'].items():
        if file_hash(source / path) != expected:
            raise ValueError('Frozen input changed: ' + path)
    _evaluate(source, dest, plan, load(source / 'queue.json'))


def _evaluate(source, dest, plan, queue):
    from scipy.stats import spearmanr, kendalltau
    historical_csv = HISTORY / 'execution_v6/spearman_by_assay.csv'
    with historical_csv.open() as handle:
        previous = {r['assay_id']: r for r in csv.DictReader(handle) if r['source'] == 'platinum'}
    rows = []
    for assay in plan['assays']:
        eid = assay['assay_id']
        row = dict(assay_id=eid, reference_complex_id=assay['reference_complex_id'], N=10,
                   historical_sequence_spearman=float(previous[eid]['spearman']), status=assay['status'])
        if assay['status'] != 'prepared':
            row['reason'] = assay['reason']; rows.append(row); continue
        stored = source / 'assays' / eid
        written = dest / 'assays' / eid
        task = load(stored / 'task.json'); ids = [c['id'] for c in task['candidates']]
        predictions = {'sequence_cached': load(stored / 'base_scores.json')}
        fusion_path = stored / 'msa_gemini_rank5050.json'
        fusion = load(fusion_path)
        row['msa_gemini_rank5050_status'] = fusion['status']
        if fusion['status'] == 'complete':
            predictions['msa_gemini_rank5050'] = {r['candidate_id']: r['fusion_value'] for r in fusion['rows']}
        else:
            row['msa_gemini_rank5050_status'] = 'incomplete_MSA'
        structure_unit = next((u for u in queue if u['assay_id'] == eid and u['condition'] == 'structure'), None)
        structure_path = source / 'results' / f"{structure_unit['request_hash']}.json" if structure_unit else None
        structure_result = validated_result(source, structure_unit, plan) if structure_path and structure_path.exists() else {'status': 'not_run'}
        if structure_result.get('status') == 'ok':
            origin = dict(source_response=str(structure_path), source_response_sha256=file_hash(structure_path))
            structure_fusion = fusion_artifact(
                ids, structure_result['value'], load(stored / 'features.json'), origin,
                condition='postprocessing_of_structure_scores')
            save(written / 'msa_gemini_rank5050_structure.json', structure_fusion)
            row['msa_gemini_rank5050_structure_status'] = structure_fusion['status']
            if structure_fusion['status'] == 'complete':
                predictions['msa_gemini_rank5050_structure'] = {
                    item['candidate_id']: item['fusion_value'] for item in structure_fusion['rows']}
            else:
                row['msa_gemini_rank5050_structure_status'] = 'incomplete_MSA'
        else:
            row['msa_gemini_rank5050_structure_status'] = structure_result.get('status', 'not_run')
        for unit in queue:
            if unit['assay_id'] != eid:
                continue
            p = source / 'results' / f"{unit['request_hash']}.json"
            result = validated_result(source, unit, plan) if p.exists() else {'status': 'not_run'}
            row[unit['condition'] + '_status'] = result['status']
            if result['status'] == 'ok':
                predictions[unit['condition']] = result['value']
        # Only this evaluation stage opens the exact historical labels.
        labels_path = HISTORY / 'episodes' / eid / 'evaluation_labels.json'
        labels = load(labels_path)['labels']; validate_scores(labels, ids)
        metric = dict(assay_id=eid, label_source=str(labels_path), label_sha256=file_hash(labels_path),
                      N=10, CR=25, methods={}, scoring_sources={})
        for method, scores in predictions.items():
            validate_scores(scores, ids)
            values = [scores[i] for i in ids]; gold = [labels[i] for i in ids]
            defined = len(set(values)) > 1 and len(set(gold)) > 1
            rho = float(spearmanr(values, gold).statistic) if defined else None
            tau = float(kendalltau(values, gold).statistic) if defined else None
            metric['methods'][method] = dict(spearman=rho, kendall=tau, status='complete' if defined else 'constant_undefined')
            row[method + '_spearman'] = rho
            if method == 'msa_gemini_rank5050_structure':
                metric['scoring_sources'][method] = str(written / 'msa_gemini_rank5050_structure.json')
            elif method in ('sequence_cached', 'msa_gemini_rank5050'):
                metric['scoring_sources'][method] = str(stored / 'cache_origin.json')
            else:
                unit = next(u for u in queue if u['assay_id'] == eid and u['condition'] == method)
                metric['scoring_sources'][method] = str(source / 'results' / f"{unit['request_hash']}.json")
            ranking = dict(method=method, rows=[dict(candidate_id=i, score=s, final_rank=r)
                                               for i, s, r in zip(ids, values, midranks(values, descending=True))])
            (written / 'rankings').mkdir(parents=True, exist_ok=True)
            (written / 'rankings' / f'{method}.json').write_text(json.dumps(ranking, indent=2, allow_nan=False) + '\n')
        written.mkdir(parents=True, exist_ok=True)
        (written / 'metrics.json').write_text(json.dumps(metric, indent=2, allow_nan=False) + '\n')
        if abs(row['sequence_cached_spearman'] - row['historical_sequence_spearman']) > 1e-10:
            raise ValueError('Cached baseline metric differs from historical same-set result')
        row['status'] = 'complete' if all(row[c + '_status'] == 'ok' for c in CONDITIONS) else 'incomplete_scoring'
        rows.append(row)
    fields = sorted(set().union(*(set(row) for row in rows)))
    dest.mkdir(parents=True, exist_ok=True)
    with (dest / 'per_assay.csv').open('w') as handle:
        writer = csv.DictWriter(handle, fields); writer.writeheader(); writer.writerows(rows)
    methods = ('sequence_cached', 'msa_gemini_rank5050', *CONDITIONS, 'msa_gemini_rank5050_structure')
    summary = dict(requested_assays=15, admitted_assays=plan['reused_scoring_requests'],
                   excluded_assays=sum(a['status'] != 'prepared' for a in plan['assays']),
                   reused_requests=plan['reused_scoring_requests'], N=10, CR=25,
                   planned_new_requests=plan['planned_new_requests'], methods={}, paired_comparisons={},
                   all_15_assays_completed=all(
                       r['status'] == 'complete' and r.get('msa_gemini_rank5050_status') == 'complete'
                       and r.get('msa_gemini_rank5050_structure_status') == 'complete' for r in rows))
    for method in methods:
        values = [r[method + '_spearman'] for r in rows if r.get(method + '_spearman') is not None]
        summary['methods'][method] = dict(complete_defined_assays=len(values),
                                          macro_spearman=sum(values) / len(values) if values else None)
        if method != 'sequence_cached':
            paired = [r for r in rows if r.get(method + '_spearman') is not None]
            deltas = [r[method + '_spearman'] - r['sequence_cached_spearman'] for r in paired]
            summary['paired_comparisons'][method] = dict(
                assays=[r['assay_id'] for r in paired], n_assays=len(paired),
                cached_sequence_macro_spearman=sum(r['sequence_cached_spearman'] for r in paired) / len(paired) if paired else None,
                delta_macro_spearman=sum(deltas) / len(deltas) if deltas else None)
    summary['usage'] = load(source / 'usage.json') if (source / 'usage.json').exists() else dict(provider_attempts=0)
    summary['new_provider_calls'] = 0 if dest.resolve() != source.resolve() else summary['usage']['provider_attempts']
    summary['source_result_dir'] = str(source)
    summary['changed_prompt_units_completed'] = sum(r.get(c + '_status') == 'ok' for r in rows for c in CONDITIONS)
    summary['reused_prompt_results'] = summary['changed_prompt_units_completed']
    summary['unfinished'] = [dict(assay_id=r['assay_id'], reason=r.get('reason') or 'incomplete_scoring')
                             for r in rows if r['status'] != 'complete']
    for condition in ('msa_gemini_rank5050', 'msa_gemini_rank5050_structure'):
        summary['unfinished'] += [dict(assay_id=r['assay_id'], condition=condition, reason='missing_MSA_component')
                                  for r in rows if r.get(condition + '_status') == 'incomplete_MSA']
    (dest / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    print(json.dumps(summary), flush=True)
    lines = ['Protein–ligand 离线消融评估（N=10，chunk ranking CR=25）', '',
             f"15 项状态均记录；新 prompt 完成 {summary['changed_prompt_units_completed']}/{plan['planned_new_requests']} 个评分单元。",
             f"复用 {summary['reused_requests']} 个精确匹配序列评分缓存；本次新增 provider 调用 {summary['new_provider_calls']} 次。",
             f"已有 prompt 评分来自 {source}，复用 {summary['reused_prompt_results']} 个成功结果。",
             '候选、顺序、划分、seed=42、评分参数及有符号 Spearman 沿用历史配置；标签仅由 analyze 读取。',
             '两路 50/50 都是后处理：原序列缓存或 Structure prompt 的 Gemini 分，各自与独立 MSA 分数融合。',
             '融合沿用平均并列名次；两路升序 midrank 各转为 (rank-1)/(N-1)，固定 50/50，最终降序平均名次。',
             '任一 MSA 分量缺失时整个 assay 的两路融合都缺失，保留所有逐候选分量，不对子集排名，也不重新调用 Gemini。',
             'MSA 统计概率与全 20 种概率表分离，基础位点 ID、映射、来源哈希与原始数据保留。',
             '指定伙伴沿用原注释 ligand residue name 的全部 HETATM 拷贝；接触数单位为伙伴重原子个数，SASA 单位 Å²。',
             'SASA 使用 Bio.PDB ShrakeRupley（探针 1.4 Å，100 点）；不可用值为 null。', '',
             '| assay | PDB | 原序列 | 原序列+MSA 50/50 | MSA prompt | Structure prompt | MSA+Structure prompt | Structure+MSA 50/50 |',
             '|---|---|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        values = [row.get(m + '_spearman') for m in methods]
        if row['status'] == 'not_run_scope_boundary':
            values[0] = row['historical_sequence_spearman']
        lines.append('| ' + ' | '.join([row['assay_id'], row['reference_complex_id'],
                     *['缺失' if v is None else f'{v:.4f}' for v in values]]) + ' |')
    lines += ['', '各方法覆盖率与同集合比较：']
    for method, item in summary['methods'].items():
        lines.append(f"{method}: {item['complete_defined_assays']} assay; macro Spearman={item['macro_spearman']}")
    lines += ['', '未完成项目：', *[json.dumps(r, ensure_ascii=False) for r in summary['unfinished']], '',
              '5 个敏感 assay 不继续逐变体评分/排名，仅列出历史汇总及排除原因。',
              '旧 20261002_v1 目录保持原位；新请求不能复用旧的不同 prompt 评分。',
              'code_snapshot/、plan.json、validation.json、requests/、results/、provider/request_ledger.sqlite、',
              'frozen_probability_tables/、msa_weights/、assays/*/site_mapping.json 与 metrics.json 提供完整审计记录。']
    (dest / 'REPORT.txt').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'validate', 'pilot', 'run', 'analyze', 'publish-fusion'])
    parser.add_argument('--output', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--source', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--credential-file', type=Path, default=PROJECT / 'private_config/evagent/credentials.env')
    args = parser.parse_args()
    if args.command in ('pilot', 'run'):
        run(args.output, args.credential_file, pilot=args.command == 'pilot')
    elif args.command == 'publish-fusion':
        publish_fusion(args.source, args.output)
    else:
        globals()[args.command](args.output)
