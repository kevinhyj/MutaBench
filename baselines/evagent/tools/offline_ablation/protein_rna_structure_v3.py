"""Versioned Structure v3 rescore of two frozen protein–RNA arms, N10/CR25.

Preserve v2 candidate order, MSA features, scores, split and fixed rank fusion.
Only evaluate() opens offline_labels.json. Changed prompts get new Gemini calls.
"""
from __future__ import annotations
import argparse
import asyncio
import copy
import csv
import importlib.util
import json
from pathlib import Path
import shutil
import statistics
import sys
import time

CODE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE))
CONDITIONS = ('structure_v3', 'msa_prompt_structure_v3')
TARGETS = ('msa_prompt_structure_v3', 'msa_gemini_rank5050_structure_v3')


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    if path.exists():
        if path.read_text() != text:
            raise ValueError('Frozen artifact differs: ' + str(path))
    else:
        path.write_text(text)


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def dependencies(args, create=False):
    out, old = args.output, args.baseline
    if create:
        sources = {**{p: out / 'code_snapshot/historical' / p.relative_to(old / 'code_snapshot')
                      for p in (old / 'code_snapshot').rglob('*.py')},
                   **{p: out / 'code_snapshot/current' / p.name for p in
                      (Path(__file__), Path(__file__).with_name('structure_v3.py'),
                       Path(__file__).with_name('structure_v3_reviews.json'))}}
        design = args.root / 'reports/protein_rna_structure_input_v3_design_20261004'
        sources.update({design / name: out / 'code_snapshot/design' / name for name in
                        ('structure_prompt.schema.json', 'validate_contract.py', 'prompt_interpretation.txt', 'SPECIFICATION.txt')})
        for source, target in sources.items():
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and target.read_bytes() != source.read_bytes():
                raise ValueError('Code/design snapshot mismatch')
            if not target.exists():
                shutil.copyfile(source, target)
    historical = out / 'code_snapshot/historical/code'
    sys.path.insert(0, str(historical))
    sys.path.insert(0, str(historical / 'tools/offline_ablation'))
    helper = load_module('evagent.offline_ablation', historical / 'evagent/offline_ablation.py')
    load_module('features', historical / 'tools/offline_ablation/features.py')
    load_module('rna_features', historical / 'tools/offline_ablation/rna_features.py')
    backend = load_module('structure_v3_historical_transport', historical / 'tools/offline_ablation/protein_rna_v2.py')
    geometry = load_module('structure_v3', out / 'code_snapshot/current/structure_v3.py')
    contract = load_module('structure_v3_contract', out / 'code_snapshot/design/validate_contract.py')
    return helper, backend, geometry, contract


def check_old(old):
    manifest = read(old / 'results_manifest.json')
    for name, expected in manifest['files'].items():
        if sha(old / name) != expected:
            raise ValueError('Original result changed: ' + name)
    return len(manifest['files'])


def prompt_request(helper, common, task, unit, msa, structure, condition, original, rules):
    _, sites = helper.base_sites(task)
    context = {k: task[k] for k in ('reference_complex_id', 'modality', 'assay_context', 'reference_chains', 'binding_partners')}
    context['candidates'] = [dict(id=cid, sites=sites[cid]) for cid in unit['expected_ids']]
    evidence = helper.render_evidence(task, unit['expected_ids'], msa, 'msa_prompt') if condition == 'msa_prompt_structure_v3' else {}
    evidence['structure_v3'] = structure
    instruction = common.INSTRUCTION.replace(
        'Use only the supplied sequence and assay context. No coordinates, structural features, MSA, or experimental variant measurements are supplied.',
        'Use only the supplied sequence, assay context and frozen evidence. No experimental variant measurements are supplied.')
    guidance = ('Chunk ranking CR=25: this request contains one consecutive frozen chunk of 10 candidates. '
                'Return a continuous score for every supplied ID under the independent [-1,1] score contract above; '
                'the evaluator merges chunk scores and sorts them offline. '
                'MSA probabilities are weighted empirical amino-acid frequencies with pseudocount smoothing, not probabilities of homology. '
                'log_probability_ratio = ln(p_mutant/p_wt). Independent-site evidence does not describe epistasis. '
                'Structure context identifies the physical RNA scope separately from the unchanged base sequence identities. '
                'When the structure scope or construct differs, its coordinates do not replace the supplied base assay reference. '
                'Ranking fusion is performed separately offline, with no fusion weights applied inside the prompt.\n' + rules)
    request = copy.deepcopy(original)
    request['contents'][0]['parts'][0]['text'] = (
        instruction + json.dumps(context, ensure_ascii=False, separators=(',', ':'), allow_nan=False) +
        '\n\nEvidence interpretation:\n' + guidance +
        '\n\nFrozen evidence:\n' + json.dumps(evidence, ensure_ascii=False, separators=(',', ':'), allow_nan=False))
    return request


def prepare(args):
    out, old = args.output, args.baseline
    preserved = check_old(old)
    helper, backend, geometry, contract = dependencies(args, create=True)
    inherited = read(old / 'config.json')
    common = backend.legacy_module(Path(inherited['source_base']))
    protocol = inherited['protocol']
    cohort = read(old / 'cohort.json')
    if len(cohort) != 10 or inherited['N'] != 10 or inherited['CR'] != 25:
        raise ValueError('Original N10/CR25 cohort differs')
    reviews = read(out / 'code_snapshot/current/structure_v3_reviews.json')
    rules = (out / 'code_snapshot/design/prompt_interpretation.txt').read_text()
    index, coverage = [], []
    baseline_units = {(u['group_id'], u['condition']): u for u in read(old / 'index.json')}
    for assay in cohort:
        group = assay['group_id']
        task = read(old / assay['task_path'])
        ids = assay['expected_ids']
        if [c['id'] for c in task['candidates']] != ids or len(ids) != 10 or assay['batch'] != 0:
            raise ValueError('Original candidate order/chunk differs')
        msa = read(old / 'features' / (group + '.json'))
        mapping = read(old / 'mapping_cache' / (group + '.json'))
        review = copy.deepcopy(reviews['structures'][assay['reference_complex_id']])
        first_site = next(iter(mapping['sites'].values()))
        review.update(structure_path=first_site['structure_path'], structure_sha256=first_site['structure_sha256'],
                      partner_definition_id=assay['reference_complex_id'] + '-explicit-reviewed-polymer-scope-v3')
        structure, cache = geometry.build(task, review, out)
        contract.validate(structure, mapping['sites'], require_full_site_coverage=True)
        for relpath in (assay['task_path'], 'features/' + group + '.json', 'mapping_cache/' + group + '.json'):
            target = out / ('msa_' + relpath if relpath.startswith('features/') else relpath)
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and target.read_bytes() != (old / relpath).read_bytes():
                raise ValueError('Copied original artifact changed')
            if not target.exists():
                shutil.copyfile(old / relpath, target)
        save(out / 'structure_features' / (group + '.json'), structure)
        save(out / 'structure_mapping' / (group + '.json'), cache)
        unit = baseline_units[group, 'structure']
        original = read(old / unit['request_path'])
        for condition in CONDITIONS:
            request = prompt_request(helper, common, task, assay, msa, structure, condition, original, rules)
            request_path = 'requests/' + condition + '/' + assay['id'] + '.json'
            save(out / request_path, request)
            prompt_path = out / 'prompts' / condition / (assay['id'] + '.txt')
            prompt_path.parent.mkdir(parents=True, exist_ok=True)
            text = request['contents'][0]['parts'][0]['text']
            if prompt_path.exists() and prompt_path.read_text() != text:
                raise ValueError('Frozen prompt differs')
            prompt_path.write_text(text)
            index.append(dict(assay, id=condition + '__' + assay['id'], base_unit_id=assay['id'],
                              condition=condition, request_path=request_path, request_sha256=sha(out / request_path)))
        coverage.append(dict(group_id=group, reference_complex_id=assay['reference_complex_id'],
            unique_sites=len(structure['sites']), eligible_scope=cache['scope_eligible'],
            computed_RNA_geometry_sites=sum(s['interface_geometry_status'] == 'computed' for s in structure['sites']),
            local_protein_environment_sites=sum(s['protein_environment_status'] == 'observed' for s in structure['sites']),
            context=structure['context']))
        print(json.dumps({k: v for k, v in coverage[-1].items() if k != 'context'}), flush=True)
    config = dict(schema='protein-rna-structure-v3-rescore', source_base=inherited['source_base'],
        baseline=str(old), baseline_manifest_sha256=sha(old / 'results_manifest.json'),
        protocol=protocol, CR=25, N=10, scoring_mode='chunk-ranking absolute independent scores, then merge/sort',
        chunks_per_assay=1, actual_chunk_size=10, prompt_conditions=list(CONDITIONS), target_arms=list(TARGETS),
        postprocessing_conditions={'msa_gemini_rank5050_structure_v3': 'structure_v3'},
        fusion=inherited['fusion'], msa_scoring_policy=inherited['msa_scoring_policy'],
        source_hashes=inherited['source_hashes'], new_request_count=20, max_attempts=3, concurrency=5,
        original_results_preserved=True, reused_Gemini_responses=0,
        optional_structure_interactions='not computed and omitted; pairing annotations unknown',
        status_policy='v3 full-interface admission/masks; no weakening of design eligibility for coverage',
        created_unix=time.time())
    save(out / 'config.json', config)
    save(out / 'cohort.json', cohort)
    save(out / 'index.json', index)
    save(out / 'preflight.json', dict(assays=coverage, original_result_files_checked=preserved,
                                    model_calls=0, offline_labels_opened=False, MSA_recomputed=False,
                                    candidate_order_unchanged=True))
    frozen = {str(p.relative_to(out)): sha(p) for p in sorted(out.rglob('*'))
              if p.is_file() and p.name not in ('freeze.json', 'validation.json', 'prepare.log')}
    sources = {str(old / 'results_manifest.json'): sha(old / 'results_manifest.json')}
    for source, expected in inherited['source_hashes'].items():
        if sha(source) != expected:
            raise ValueError('Historical source changed')
        sources[source] = expected
    for assay in cohort:
        feature = read(old / 'features' / (assay['group_id'] + '.json'))
        for data in feature['source_provenance']['msa_chains'].values():
            for key, source in (('msa', data['msa_path']), ('mapping', data['mapping_path'])):
                if sha(source) != data['source_hashes'][key]:
                    raise ValueError('MSA evidence source changed')
                sources[source] = data['source_hashes'][key]
        path = read(out / 'structure_mapping' / (assay['group_id'] + '.json'))['structure_path']
        sources[path] = sha(path)
    save(out / 'freeze.json', dict(files=frozen, external_sources=sources, runner_sha256=sha(Path(__file__))))
    validate(args)


def checked_response(out, unit, common, helper):
    path = out / 'responses' / (unit['id'] + '.json')
    if not path.exists():
        return None
    record = read(path)
    if record['unit_id'] != unit['id'] or record['request_sha256'] != unit['request_sha256']:
        raise ValueError('Response unit/request identity mismatch')
    if record['status'] != 'ok':
        return record
    attempt = Path(record['raw_attempt'])
    if not attempt.is_relative_to(out / 'attempts') or sha(attempt) != record['raw_attempt_sha256']:
        raise ValueError('Raw response attempt lineage/hash mismatch')
    raw = read(attempt)
    if raw['unit_id'] != unit['id'] or raw['request_sha256'] != unit['request_sha256'] or raw['http_status'] != 200:
        raise ValueError('Raw attempt identity/HTTP mismatch')
    parsed = common.parse(raw['body'], unit['expected_ids'])
    if parsed['status'] != 'ok' or parsed['scores'] != record['scores'] or parsed['scores'] != raw['parsed']['scores']:
        raise ValueError('Saved scores differ from independent raw response parse')
    helper.validate_scores(record['scores'], unit['expected_ids'], bounded=True)
    model = read(out / 'config.json')['protocol']['requested_model']
    if not str(record['model']).startswith(model) or raw['body'].get('modelVersion') != record['model']:
        raise ValueError('Model identity mismatch')
    return record


def validate(args):
    out = args.output
    freeze = read(out / 'freeze.json')
    if sha(Path(__file__)) != freeze['runner_sha256']:
        raise ValueError('Runner differs from frozen campaign')
    for relpath, expected in freeze['files'].items():
        if sha(out / relpath) != expected:
            raise ValueError('Frozen input/code changed: ' + relpath)
    for path, expected in freeze['external_sources'].items():
        if sha(path) != expected:
            raise ValueError('Frozen source changed: ' + path)
    preserved = check_old(args.baseline)
    helper, backend, _, contract = dependencies(args)
    config = read(out / 'config.json')
    if config['N'] != 10 or config['CR'] != 25 or config['fusion']['weights'] != [0.5, 0.5]:
        raise ValueError('Frozen N/CR/fusion changed')
    common = backend.legacy_module(Path(config['source_base']))
    cohort = read(out / 'cohort.json')
    if cohort != read(args.baseline / 'cohort.json'):
        raise ValueError('Assay cohort/order changed')
    rules = (out / 'code_snapshot/design/prompt_interpretation.txt').read_text()
    index = read(out / 'index.json')
    if len(index) != 20 or len({u['id'] for u in index}) != 20:
        raise ValueError('Expected exactly twenty unique requests')
    old_units = {(u['group_id'], u['condition']): u for u in read(args.baseline / 'index.json')}
    response_count = 0
    for unit in index:
        group = unit['group_id']
        task = read(out / unit['task_path'])
        if task != read(args.baseline / unit['task_path']):
            raise ValueError('Original task/candidates changed')
        mapping = read(out / 'mapping_cache' / (group + '.json'))['sites']
        msa_path = out / 'msa_features' / (group + '.json')
        if sha(msa_path) != sha(args.baseline / 'features' / (group + '.json')):
            raise ValueError('Original MSA protocol/evidence changed')
        structure = read(out / 'structure_features' / (group + '.json'))
        contract.validate(structure, mapping, require_full_site_coverage=True)
        old_unit = old_units[group, 'structure']
        expected = prompt_request(helper, common, task, unit, read(msa_path), structure, unit['condition'],
                                  read(args.baseline / old_unit['request_path']), rules)
        request = read(out / unit['request_path'])
        if request != expected or sha(out / unit['request_path']) != unit['request_sha256']:
            raise ValueError('Actual request does not match validated prompt/fields')
        prompt = (out / 'prompts' / unit['condition'] / (unit['base_unit_id'] + '.txt')).read_text()
        if prompt != request['contents'][0]['parts'][0]['text']:
            raise ValueError('Actual prompt text differs')
        if unit['batch'] != 0 or len(unit['expected_ids']) != 10 or len(unit['expected_ids']) > 25:
            raise ValueError('CR25 chunk changed')
        if request['generationConfig'] != read(args.baseline / old_unit['request_path'])['generationConfig']:
            raise ValueError('Generation configuration changed')
        if checked_response(out, unit, common, helper) is not None:
            response_count += 1
    write(out / 'validation.json', dict(status='passed', assays=10, candidates=100, requests=20,
        CR=25, actual_chunk_size=10, original_frozen_result_files_checked=preserved,
        saved_response_lineages_checked=response_count, offline_labels_opened=False,
        checks=['v3 schema and relational masks', 'full stable site-ID join', 'exact original candidates/order',
                'original MSA bytes', 'RNA instance/chemical classification', 'SASA arithmetic',
                'frozen code/prompt/source hashes', 'raw-response hash/reparse and model identity']))
    print('Validation passed: 10 assays, 20 new-prompt CR25 chunks; raw-response lineage checked', flush=True)


async def run(args):
    helper, backend, _, _ = dependencies(args)
    backend.validate = validate
    args.resume_auth_failures = False
    await backend.run(args)
    validate(args)


def csvwrite(path, rows):
    with Path(path).open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [])
        writer.writeheader(); writer.writerows(rows)


def evaluate(args):
    from scipy.stats import spearmanr
    validate(args)
    out = args.output
    helper, backend, _, _ = dependencies(args)
    config = read(out / 'config.json')
    common = backend.legacy_module(Path(config['source_base']))
    labels = {(r['group_id'], r['variant_id']): r for r in read(Path(config['source_base']) / 'offline_labels.json')}
    units = {(u['group_id'], u['condition']): u for u in read(out / 'index.json')}
    previous = {(r['group_id'], r['method']): r for r in csv.DictReader((args.baseline / 'per_assay.csv').open())}
    metrics, ranks, joined = [], [], []
    for assay in read(out / 'cohort.json'):
        group, ids = assay['group_id'], assay['expected_ids']
        msa = read(out / 'msa_features' / (group + '.json'))['independent_msa_scores']
        scores, origins, statuses, fused = {}, {}, {}, {}
        for condition in CONDITIONS:
            unit = units[group, condition]
            response = checked_response(out, unit, common, helper)
            origins[condition] = str(out / 'responses' / (unit['id'] + '.json'))
            statuses[condition] = response['status'] if response else 'not_scored'
            if response and response['status'] == 'ok':
                scores[condition] = response['scores']
        method = 'msa_gemini_rank5050_structure_v3'
        if 'structure_v3' in scores:
            fused[method] = helper.fuse_rankings(ids, scores['structure_v3'], msa)
            scores[method] = {r['candidate_id']: r['fusion_value'] for r in fused[method]}
            origins[method] = origins['structure_v3']
        for method in (*TARGETS, 'structure_v3'):
            values = scores.get(method)
            complete = values is not None and set(values) == set(ids)
            rho = record_rho = None
            if complete:
                xs = [values[cid] for cid in ids]
                ys = [labels[group, cid]['experimental_score_mean'] for cid in ids]
                if len(set(xs)) > 1 and len(set(ys)) > 1:
                    rho = float(spearmanr(xs, ys).statistic)
                    rawx = [values[cid] for cid in ids for _ in labels[group, cid]['experimental_scores']]
                    rawy = [y for cid in ids for y in labels[group, cid]['experimental_scores']]
                    record_rho = float(spearmanr(rawx, rawy).statistic)
                components = {r['candidate_id']: r for r in fused.get(method, [])}
                final = helper.midranks(xs, descending=True)
            else:
                components, final = {}, [None] * len(ids)
            for cid, rank in zip(ids, final):
                comp = components.get(cid, {})
                origin = origins.get(method, '')
                ranks.append(dict(group_id=group, reference_complex_id=assay['reference_complex_id'],
                    method=method, candidate_id=cid, score=values.get(cid) if values else None, final_rank=rank,
                    gemini_score=comp.get('gemini_score'), msa_score=comp.get('msa_score'),
                    gemini_rank_percentile=comp.get('gemini_rank_percentile'), msa_rank_percentile=comp.get('msa_rank_percentile'),
                    fusion_value=comp.get('fusion_value'), score_source=origin, score_source_sha256=sha(origin) if origin and Path(origin).is_file() else None,
                    msa_source=str(out / 'msa_features' / (group + '.json')) if method in config['postprocessing_conditions'] else None))
            metrics.append(dict(group_id=group, reference_complex_id=assay['reference_complex_id'], method=method,
                N=10, CR=25, scored=len(values) if values else 0, status='complete' if complete else statuses.get(method, 'missing_Gemini'),
                spearman=rho, record_level_spearman=record_rho, reason='constant_predictions_or_labels' if complete and rho is None else None))
        local = {r['method']: r for r in metrics if r['group_id'] == group}
        number = lambda key: float(previous[group, key]['spearman']) if previous[group, key]['spearman'] else None
        joined.append(dict(group_id=group, reference_complex_id=assay['reference_complex_id'],
            old_msa_prompt_structure=number('msa_prompt_structure'),
            new_msa_prompt_structure=local['msa_prompt_structure_v3']['spearman'],
            old_msa_rank_structure=number('msa_gemini_rank5050_structure'),
            new_msa_rank_structure=local['msa_gemini_rank5050_structure_v3']['spearman']))
    attempts = [read(p) for p in sorted((out / 'attempts').glob('*/*.json'))]
    means = {m: statistics.mean(v) if (v := [r['spearman'] for r in metrics if r['method'] == m and r['spearman'] is not None]) else None
             for m in (*TARGETS, 'structure_v3')}
    coverage = read(out / 'preflight.json')['assays']
    summary = dict(assays=10, candidates=100, N=10, CR=25, actual_chunk_size=10, scoring_mode=config['scoring_mode'],
        complete_assays={m: sum(r['method'] == m and r['status'] == 'complete' for r in metrics) for m in means},
        mean_signed_spearman=means, new_api_attempts=len(attempts), successful_new_calls=sum(a['status'] == 'ok' for a in attempts),
        reused_Gemini_responses=0, MSA_evidence_recomputed=False, fusion_weights=[0.5, 0.5],
        attempts_by_status={s: sum(a['status'] == s for a in attempts) for s in {a['status'] for a in attempts}},
        reported_total_tokens=sum((a.get('usage') or {}).get('totalTokenCount', 0) for a in attempts),
        scoring_start_unix=min((a['started'] for a in attempts), default=None), scoring_end_unix=max((a.get('finished', a['started']) for a in attempts), default=None),
        structure_evidence_coverage=dict(eligible_assays=sum(c['eligible_scope'] for c in coverage),
            unique_sites=sum(c['unique_sites'] for c in coverage),
            computed_RNA_geometry_sites=sum(c['computed_RNA_geometry_sites'] for c in coverage),
            local_protein_environment_sites=sum(c['local_protein_environment_sites'] for c in coverage)),
        unfinished_scoring=[r for r in metrics if r['status'] != 'complete'],
        unresolved_structure_assays=[c for c in coverage if not c['eligible_scope']],
        original_results_preserved=True, labels_rendered_to_predictor=False,
        config_sha256=sha(out / 'config.json'))
    csvwrite(out / 'per_assay.csv', metrics); csvwrite(out / 'rankings.csv', ranks)
    csvwrite(out / 'comparison.csv', joined); write(out / 'summary.json', summary)
    lines = ['Protein–RNA Structure v3 rescore: N=10, chunk ranking CR=25', '',
             'Two requested arms use twenty changed-prompt requests. Structure-only scores supply the Gemini component of fixed 50/50 rank-percentile fusion.',
             'Original candidate sets/order, MSA v2 evidence, metrics and model configuration are unchanged. Original files are read-only.', '',
             f'API attempts={len(attempts)}, successful calls={summary["successful_new_calls"]}, reused Gemini responses=0.',
             'Mean signed Spearman: ' + json.dumps(means), '',
             'Complex / assay | old MSA prompt+Structure | new | old MSA rank+Structure | new']
    for row in joined:
        lines.append(' | '.join([row['reference_complex_id'] + '/' + row['group_id'].split('__')[-1][:8]] +
                    [f'{row[k]:.6f}' if row[k] is not None else 'missing' for k in
                     ('old_msa_prompt_structure', 'new_msa_prompt_structure', 'old_msa_rank_structure', 'new_msa_rank_structure')]))
    lines.extend(['', 'Structure evidence coverage: ' + json.dumps(summary['structure_evidence_coverage']),
                  'Unreviewed, mismatched, incomplete RNA scopes are explicitly masked to null; local WT protein counts remain available.',
                  'Complete candidate scoring does not mean complete biological interface evidence. Exact review and missing-atom records are in structure_mapping/.',
                  'SASA uses the explicit polymer environment, 1.4A probe and original 100 points. Eligible scopes also have 960-point sensitivity audit, not used for scoring.',
                  'H-bond, stacking, water and base-pair annotations are not invented; optional interactions omitted, pairing_state unknown.',
                  '5HO4 literature-supported RNA image is in the observed-scope audit; incomplete RNA and assembly review prevent exact interface fields entering its prompt.',
                  f'Unfinished scored method/assay units: {len(summary["unfinished_scoring"])}. Raw-response hashes and independent parsing validated.'])
    (out / 'REPORT.txt').write_text('\n'.join(lines) + '\n')
    write(out / 'results_manifest.json', dict(algorithm='sha256', files={
        str(p.relative_to(out)): sha(p) for p in sorted(out.rglob('*'))
        if p.is_file() and p.name not in ('results_manifest.json', 'run.lock') and '__pycache__' not in p.parts}))
    print(json.dumps({k: summary[k] for k in ('complete_assays', 'mean_signed_spearman', 'new_api_attempts', 'successful_new_calls', 'structure_evidence_coverage')}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('prepare', 'validate', 'run', 'evaluate'))
    parser.add_argument('--root', type=Path, default=Path('/data/huangyanjie/EVAgent'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, default=Path('/data/huangyanjie/EVAgent/experiments/protein_rna_ablation_N10_CR25_20261003_msa_v2_native_auth'))
    parser.add_argument('--credential-file', type=Path)
    args = parser.parse_args()
    args.root, args.output, args.baseline = args.root.resolve(), args.output.resolve(), args.baseline.resolve()
    if args.output == args.baseline or args.baseline in args.output.parents:
        raise ValueError('New output must be outside original results')
    if args.action == 'run':
        if not args.credential_file:
            raise ValueError('Explicit credential file required')
        asyncio.run(run(args))
    else:
        globals()[args.action](args)


if __name__ == '__main__':
    main()
