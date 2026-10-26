"""Prepare, validate, score and evaluate frozen protein–RNA CR25 ablations.

Run with an explicit Python environment. Historical files are read-only;
predictors never open offline_labels.json. No sequence generation or resampling.
"""
from __future__ import annotations

import argparse
import asyncio
import copy
import csv
import importlib.util
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time
from urllib.parse import urlsplit

CODE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE))
from evagent.offline_ablation import (base_sites, digest, file_hash, fuse_rankings,
                                     matching_cache, midranks, render_evidence,
                                     validate_scores)
from features import FeatureBuilder, save
from rna_features import POLICY, ProteinRNAFeatureBuilder
FeatureBuilder = ProteinRNAFeatureBuilder

CONDITIONS = ('msa_prompt', 'structure', 'msa_prompt_structure')


def read(path):
    return json.loads(Path(path).read_text())


def credential_value(path):
    """Read either a protected plain token or the configured GEMINI .env entry."""
    if path.stat().st_mode & 0o077:
        raise ValueError('Credential permissions are too broad')
    content = path.read_text().strip()
    entries = [line.strip().removeprefix('export ').split('=', 1)[1].strip()
               for line in content.splitlines()
               if line.strip().removeprefix('export ').startswith('GEMINI_API_KEY=')]
    value = entries[0].strip('"').strip("'") if len(entries) == 1 else content
    if len(entries) > 1 or not value or any(c.isspace() for c in value):
        raise ValueError('Credential unavailable or ambiguous')
    return value


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    os.replace(temporary, path)


def legacy_module(base):
    spec = importlib.util.spec_from_file_location('frozen_sequence_common', base / 'common.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def bind_contract_snapshot(args, *, create=False):
    """Isolate this campaign from concurrent edits to shared evidence helpers."""
    sources = (Path(__file__).resolve(), Path(__file__).with_name('features.py'),
               Path(__file__).with_name('rna_features.py'),
               CODE / 'evagent/offline_ablation.py', args.root / 'baseline/singlechain/models.py')
    paths = {}
    for source in sources:
        target = args.output / 'code_snapshot' / source.relative_to(args.root)
        if create:
            target.parent.mkdir(parents=True, exist_ok=True)
            content = source.read_bytes()
            if target.exists() and target.read_bytes() != content:
                raise ValueError('Snapshot already exists with a different implementation')
            if not target.exists():
                target.write_bytes(content)
        paths[str(source)] = target
    for name, source in [('evagent.offline_ablation', CODE / 'evagent/offline_ablation.py'),
                         ('features', Path(__file__).with_name('features.py')),
                         ('rna_features', Path(__file__).with_name('rna_features.py'))]:
        spec = importlib.util.spec_from_file_location(name, paths[str(source)])
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        if name == 'evagent.offline_ablation':
            for symbol in ('base_sites', 'digest', 'file_hash', 'fuse_rankings', 'matching_cache',
                           'midranks', 'render_evidence', 'validate_scores'):
                globals()[symbol] = getattr(module, symbol)
        elif name == 'features':
            globals()['FeatureBuilder'] = module.FeatureBuilder
            globals()['save'] = module.save
        else:
            globals()['FeatureBuilder'] = module.ProteinRNAFeatureBuilder
    return paths


def observed_input(root, task, inventory):
    """Use frozen reference numbering; extract observed residues for MSA join."""
    from Bio.PDB import PDBParser
    from Bio.SeqUtils import seq1
    item = inventory[(task['source'], task['reference_complex_id'])]
    path = root / 'data/final_mutation_data' / item['path']
    if file_hash(path) != item['sha256']:
        raise ValueError('Structure inventory source hash mismatch')
    model = PDBParser(QUIET=True).get_structure('frozen', path)[0]
    wt_chains = []
    reference = {chain['chain']: chain for chain in task['reference_chains']}
    _, sites = base_sites(task)
    for rows in sites.values():
        for site in rows:
            chain = reference[site['chain']]
            index = site['sequence_index_1based'] - 1
            number = chain['numbering'][index]
            if (chain['reference_sequence'][index] != site['wt'] or
                number['reference_residue_number'] != site['position'] or
                (number.get('insertion_code') or '') != site.get('insertion_code', '')):
                raise ValueError('Base mutation does not match frozen reference numbering/WT')
    groups = [group['chains'] for group in task['binding_partners']['groups']]
    if len(groups) != 2 or {group['role'] for group in task['binding_partners']['groups']} != {'protein', 'RNA'}:
        raise ValueError('Expected the existing protein/RNA partner partition')
    for chain_id in sum(groups, []):
        chain = reference[chain_id]
        allowed = {(n['reference_residue_number'], n.get('insertion_code') or '')
                   for n in chain['numbering']}
        residues = [residue for residue in model[chain_id]
                    if (residue.id[1], residue.id[2].strip()) in allowed]
        coordinates = [(residue.id[1], residue.id[2].strip()) for residue in residues]
        if len(coordinates) != len(set(coordinates)):
            raise ValueError('Ambiguous PDB residue identity')
        wt_chains.append(dict(chain=chain_id,
                              residue_ids=[str(pos) + ins for pos, ins in coordinates],
                              resolved_sequence=''.join(seq1(r.resname, custom_map={'MSE': 'M'})
                                                       for r in residues) if chain['polymer_type'] == 'protein' else None))
    candidate_hash = digest(task['candidates'])
    observed = dict(source=task['source'], reference_complex_id=task['reference_complex_id'],
                    candidate_hash=candidate_hash, wt_chains=wt_chains,
                    structure_source=dict(path=str(path), sha256=item['sha256']))
    partners = dict(candidate_hash=candidate_hash, groups=groups,
                    evidence=dict(definition='Existing base-task binding_partners.groups, unchanged',
                                  binding_partners=task['binding_partners'],
                                  isolated_group='All own-group chains retained together'),
                    excluded_extra_chains=sorted(set(model.child_dict) - set(sum(groups, []))))
    return observed, partners


def prompt_request(common, task, unit, features, condition, original):
    _, base = base_sites(task)
    ids = unit['expected_ids']
    context = {k: task[k] for k in ('reference_complex_id', 'modality', 'assay_context',
                                  'reference_chains', 'binding_partners')}
    context['candidates'] = [dict(id=cid, sites=base[cid]) for cid in ids]
    evidence = render_evidence(task, ids, features, condition)
    instruction = common.INSTRUCTION.replace(
        'Use only the supplied sequence and assay context. No coordinates, structural features, MSA, or experimental variant measurements are supplied.',
        'Use only the supplied sequence, assay context and frozen evidence. No experimental variant measurements are supplied.')
    guidance = (
        'MSA probabilities are weighted empirical amino-acid frequencies with pseudocount smoothing; '
        'they are statistical evidence in the supplied alignment, not probabilities of homology or measured binding effects. '
        'log_probability_ratio = ln(p_mutant/p_wt). Independent-site evidence does not describe epistasis. '
        'Structure fields describe the frozen WT complex and the explicitly specified partner groups. '
        'Distances use angstroms (A); SASA and delta SASA use square angstroms (A2). '
        'null means missing/unavailable, never zero. Retain uncertainty and availability statuses. '
        'Return independent continuous scores on the original [-1,1] scale; ranking fusion is performed separately offline.')
    request = copy.deepcopy(original)
    request['contents'][0]['parts'][0]['text'] = (
        instruction + json.dumps(context, ensure_ascii=False, separators=(',', ':')) +
        '\n\nFrozen evidence interpretation:\n' + guidance + '\n\nFrozen site evidence:\n' +
        json.dumps(evidence, ensure_ascii=False, separators=(',', ':'), allow_nan=False))
    return request


def prepare(args):
    root, out = args.root, args.output
    snapshots = bind_contract_snapshot(args, create=True)
    base = root / 'explanation_test/complex_sequence_all_assays_N250_N10_seed42_20260928'
    common = legacy_module(base)
    protocol = read(base / 'protocol.json')
    if args.use_default_proxy:
        default_path = CODE / 'examples/gemini_flash_medium_seed42.json'
        default = read(default_path)
        if (default['model'] != protocol['requested_model'] or default['generation_seed'] != 42 or
                default['thinking_level'].lower() != protocol['thinking_level'].lower() or
                default['temperature'] != protocol['temperature']):
            raise ValueError('Default proxy model configuration differs from frozen scoring semantics')
        protocol['endpoint'] = 'https://generativelanguage.googleapis.com/v1beta/models/' + default['model'] + ':generateContent'
        protocol['http_proxy'] = default['proxy']
        protocol['transport_provenance'] = dict(definition='Existing repository default Google Gemini proxy configuration',
                                               config_path=str(default_path), config_sha256=file_hash(default_path),
                                               prior_direct_gateway_misconfiguration=True,
                                               wan_neng_api_used=False)
    inventory_path = root / 'data/final_mutation_data/reports/curation_dedup_structure_20260927/structure_inventory.json'
    inventory = {(row['source'], row['complex_id']): row for row in read(inventory_path)}
    builder = FeatureBuilder(root, out)
    selected, units, caches, checks = [], [], [], []
    for unit in read(base / 'index.json'):
        if unit['source'] != 'nabe':
            continue
        task = read(base / unit['task_path'])
        if not any(g['role'] == 'RNA' for g in task['binding_partners'].get('groups', [])):
            continue
        if task['N'] != 10 or len(unit['expected_ids']) != 10 or unit['batch'] != 0:
            raise ValueError('Frozen N10 cohort/batch differs')
        expected = common.payload(task, unit, protocol)
        scores, origin = matching_cache(unit, base / unit['task_path'], base / unit['request_path'],
                                        base / 'responses' / f'{unit["id"]}.json', expected)
        save(out / unit['task_path'], task)
        save(out / 'sequence_requests' / f'{unit["id"]}.json', expected)
        save(out / 'sequence_scores' / f'{unit["id"]}.json', dict(scores=scores, provenance=origin))
        observed, partners = observed_input(root, task, inventory)
        features, mapping = builder.build(task, observed, partners)
        save(out / 'features' / f'{task["group_id"]}.json', features)
        save(out / 'mapping_cache' / f'{task["group_id"]}.json', mapping)
        save(out / 'partner_mapping' / f'{task["group_id"]}.json', partners)
        old_conditions = {}
        for condition in ('msa_prompt', 'msa_prompt_structure', 'msa_gemini_rank5050', 'msa_gemini_rank5050_structure'):
            old = root / 'explanation_test/complex_ablation_n250_n10_seed42_20260928' / condition / unit['request_path']
            if old.exists():
                old_conditions[condition] = dict(path=str(old), sha256=file_hash(old),
                    reuse=False, reason='Changed prompt or legacy in-prompt fusion is not this scoring condition')
        for condition in CONDITIONS:
            request = prompt_request(common, task, unit, features, condition, expected)
            request_path = Path('requests') / condition / f'{unit["id"]}.json'
            save(out / request_path, request)
            prompt_path = out / 'prompts' / condition / f'{unit["id"]}.txt'
            prompt_path.parent.mkdir(parents=True, exist_ok=True)
            prompt = request['contents'][0]['parts'][0]['text']
            if prompt_path.exists() and prompt_path.read_text() != prompt:
                raise ValueError('Frozen prompt changed')
            if not prompt_path.exists():
                prompt_path.write_text(prompt)
            units.append(dict(unit, id=condition + '__' + unit['id'], base_unit_id=unit['id'],
                              condition=condition, request_path=str(request_path),
                              request_sha256=file_hash(out / request_path)))
        selected.append(dict(unit, reference_complex_id=task['reference_complex_id'],
                             assay_group_id=task['assay_group_id']))
        caches.append(dict(group_id=task['group_id'], matching_sequence_cache=origin,
                           legacy_condition_cache_audit=old_conditions))
        checks.append(dict(group_id=task['group_id'], N=10, CR=25,
                           full_msa_coverage=features['full_msa_coverage'],
                           msa_statuses=mapping['all_msa_site_statuses'],
                           unique_sites=len(mapping['sites']),
                           sasa_missing=sum(row['sasa_status'] != 'ok'
                                            for rows in features['structure'].values() for row in rows.values())))
        print(json.dumps(checks[-1]), flush=True)
    if len(selected) != 10:
        raise ValueError(f'Expected exactly 10 frozen protein–RNA assays, found {len(selected)}')
    config = dict(schema='protein-rna-offline-ablation-v2', source_base=str(base),
                  msa_scoring_policy=POLICY,
                  prior_strict_results=str(root / 'experiments/protein_rna_ablation_N10_CR25_20261002_v4'),
                  protocol=protocol, CR=25, N=10, sampling_seed=42, generation_seed=42,
                  prompt_conditions=list(CONDITIONS),
                  postprocessing_conditions={'msa_gemini_rank5050': 'sequence',
                                             'msa_gemini_rank5050_structure': 'structure'},
                  fusion=dict(weights=[0.5, 0.5], direction='higher scores and percentiles better; final rank 1 best',
                              scale='(ascending average midrank - 1)/(N - 1)',
                              ties='average midrank, including final rank',
                              missing='incomplete assay; no imputation or replacement',
                              sources=['README.md', 'explanation_test/complex_rank_fusion_true_n250_seed42_20261002/recompute_rank_fusion.py']),
                  label_policy=protocol['label_policy'], metrics=['signed Spearman', 'record-level signed Spearman'],
                  new_request_count=len(units), max_attempts=3, concurrency=5,
                  response_policy='Provider explicit refusals are final; no configuration amendments',
                  source_hashes={str(base / name): file_hash(base / name)
                                 for name in ('index.json', 'protocol.json', 'common.py', 'offline_labels.json')})
    # Credentials are configuration only and are never read/copied at preparation.
    config['protocol'] = {key: value for key, value in protocol.items() if key != 'credential_file'}
    save(out / 'config.json', config)
    save(out / 'cohort.json', selected)
    save(out / 'index.json', units)
    save(out / 'cache_audit.json', caches)
    save(out / 'preflight.json', dict(assays=checks, labels_read=False, model_calls=0))
    input_dirs = {'prepared', 'sequence_requests', 'sequence_scores', 'features', 'mapping_cache',
                  'partner_mapping', 'requests', 'prompts', 'frozen_probability_tables', 'msa_weights'}
    input_files = {'config.json', 'cohort.json', 'index.json', 'cache_audit.json', 'preflight.json'}
    frozen = {str(path.relative_to(out)): file_hash(path)
              for path in sorted(out.rglob('*')) if path.is_file() and
              (path.relative_to(out).parts[0] in input_dirs or path.name in input_files)}
    save(out / 'freeze.json', dict(files=frozen,
                                 code={str(path): file_hash(path) for path in snapshots.values()},
                                 original_code_paths={source: str(path) for source, path in snapshots.items()},
                                 runner_sha256=file_hash(Path(__file__).resolve())))
    validate(args)


def validate(args):
    out = args.output
    frozen = read(out / 'freeze.json')
    if file_hash(Path(__file__).resolve()) != frozen['runner_sha256']:
        raise ValueError('Runner implementation changed; use a matching runner or a new campaign')
    for relpath, expected in frozen['files'].items():
        if file_hash(out / relpath) != expected:
            raise ValueError('Frozen input changed: ' + relpath)
    for path, expected in frozen['code'].items():
        if file_hash(Path(path)) != expected:
            raise ValueError('Frozen code changed: ' + path)
    for cache in read(out / 'cache_audit.json'):
        origin = cache['matching_sequence_cache']
        for kind in ('response', 'request', 'task'):
            if file_hash(Path(origin[f'source_{kind}'])) != origin[f'source_{kind}_sha256']:
                raise ValueError('Original cache source changed')
    for unit in read(out / 'index.json'):
        task = read(out / unit['task_path'])
        features = read(out / 'features' / f'{unit["group_id"]}.json')
        mapping = read(out / 'mapping_cache' / f'{unit["group_id"]}.json')
        identity, base = base_sites(task)
        if set(mapping['sites']) != set(identity):
            raise ValueError('Mapping cache site set differs')
        for sid, site in identity.items():
            if any(mapping['sites'][sid][key] != value for key, value in site.items()):
                raise ValueError('Mapping cache base identity differs')
            if 'msa_column' not in mapping['sites'][sid] or 'msa_source_hashes' not in mapping['sites'][sid]:
                raise ValueError('Mapping lacks MSA column/source identity')
        evidence = render_evidence(task, unit['expected_ids'], features, unit['condition'])
        request = read(out / unit['request_path'])
        encoded = json.dumps(evidence, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
        if not request['contents'][0]['parts'][0]['text'].endswith(encoded):
            raise ValueError('Actual prompt evidence differs from validated fields')
        if request['generationConfig']['seed'] != 42 or len(unit['expected_ids']) > 25:
            raise ValueError('CR25/generation configuration differs')
        for chain in features['source_provenance']['msa_chains'].values():
            table = read(chain['probability_table'])
            for key, path in (('msa', chain['msa_path']), ('mapping', chain['mapping_path'])):
                if file_hash(Path(path)) != chain['source_hashes'][key]:
                    raise ValueError('MSA input changed')
            for row in table['rows']:
                probabilities = row['probabilities_20']
                if len(probabilities) != 20 or not all(0 < p <= 1 for p in probabilities) or not math.isclose(sum(probabilities), 1, abs_tol=1e-12):
                    raise ValueError('Frozen probability table invalid')
    write(out / 'validation.json', dict(status='passed', assays=10, candidates=100,
                                      prompt_requests=30, CR=25, labels_read=False,
                                      checks=['field completeness', 'base/MSA/structure site identity',
                                              'availability/null consistency', 'probability tables',
                                              'input/code/source hashes', 'exact cached prompt/configuration']))
    print('Validation passed: 10 assays, 100 frozen candidates, 30 changed-prompt CR25 requests', flush=True)


async def run(args):
    import fcntl
    import httpx
    out = args.output
    validate(args)
    lock = (out / 'run.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    config = read(out / 'config.json')
    common = legacy_module(Path(config['source_base']))
    credential_path = args.credential_file
    credential = credential_value(credential_path)
    semaphore = asyncio.Semaphore(config['concurrency'])
    stop = asyncio.Event()

    def rejected_authentication(record):
        error = (record.get('body') or {}).get('error') or {}
        return (record.get('status') == 'provider_blocked' and record.get('http_status') == 403 and
                error.get('type') == 'gateway_error' and
                str(error.get('message', '')).startswith('API token required.'))

    async def score(unit, client):
        async with semaphore:
            response_path = out / 'responses' / f'{unit["id"]}.json'
            if response_path.exists():
                record = read(response_path)
                if record['request_sha256'] != unit['request_sha256']:
                    raise ValueError('Response request hash mismatch')
                if record['status'] == 'ok':
                    validate_scores(record['scores'], unit['expected_ids'], bounded=True)
                if (args.resume_auth_failures and record['status'] == 'provider_blocked' and
                        rejected_authentication(read(record['raw_attempt']))):
                    history = out / 'auth_failure_responses' / f'{unit["id"]}.json'
                    history.parent.mkdir(parents=True, exist_ok=True)
                    if history.exists():
                        raise ValueError('Authentication retry history already exists')
                    os.rename(response_path, history)
                else:
                    return
            attempt_dir = out / 'attempts' / unit['id']
            prior = sorted(attempt_dir.glob('*.json')) if attempt_dir.exists() else []
            if (prior and read(prior[-1])['status'] in ('started', 'explicit_refusal', 'provider_blocked', 'model_identity_mismatch')
                    and not (args.resume_auth_failures and rejected_authentication(read(prior[-1])))):
                write(response_path, dict(unit_id=unit['id'], request_sha256=unit['request_sha256'],
                                          status='requires_review', reason=read(prior[-1])['status']))
                return
            for number in range(len(prior) + 1, config['max_attempts'] + 1):
                if stop.is_set():
                    return
                started = time.time()
                attempt_path = attempt_dir / f'{number:03d}.json'
                write(attempt_path, dict(unit_id=unit['id'], attempt=number, started=started,
                                         request_sha256=unit['request_sha256'], status='started'))
                body, http_status, result = {}, None, {'status': 'transport_error'}
                try:
                    async with asyncio.timeout(max(3600, config['protocol']['total_deadline_seconds'])):
                        headers = {'x-goog-api-key': credential}
                        if not config['protocol'].get('http_proxy'):
                            headers['Authorization'] = 'Bearer ' + credential
                        response = await client.post(config['protocol']['endpoint'],
                                                     headers=headers,
                                                     json=read(out / unit['request_path']))
                        http_status = response.status_code
                        try:
                            body = response.json()
                        except ValueError:
                            body = {'unparsed_text': response.text[:20000]}
                        if not isinstance(body, dict):
                            body = {'unexpected_body': body}
                        result = common.parse(body, unit['expected_ids']) if http_status == 200 else {'status': 'http_error'}
                        if (body.get('promptFeedback') or {}).get('blockReason') or result['status'] == 'permanent_refusal':
                            result = {'status': 'explicit_refusal'}
                        elif http_status in (401, 402, 403):
                            result = {'status': 'provider_blocked'}
                            stop.set()
                        elif result['status'] == 'ok' and not str(body.get('modelVersion', '')).startswith(config['protocol']['requested_model']):
                            result = {'status': 'model_identity_mismatch'}
                            stop.set()
                except Exception as error:
                    result = {'status': 'transport_error', 'error_type': type(error).__name__}
                record = dict(unit_id=unit['id'], request_sha256=unit['request_sha256'], attempt=number,
                              status=result['status'], started=started, finished=time.time(),
                              http_status=http_status, model=body.get('modelVersion'),
                              usage=body.get('usageMetadata'), body=body, parsed=result)
                record = json.loads(json.dumps(record, ensure_ascii=False).replace(credential, '[REDACTED]'))
                write(attempt_path, record)
                print(json.dumps(dict(unit_id=unit['id'], attempt=number, status=result['status'],
                                      http_status=http_status, seconds=round(time.time() - started, 1))), flush=True)
                terminal = result['status'] in ('ok', 'explicit_refusal', 'provider_blocked', 'model_identity_mismatch') or http_status in (400,404,405,413,422)
                if terminal or number == config['max_attempts']:
                    write(response_path, dict(unit_id=unit['id'], request_sha256=unit['request_sha256'],
                                              status=result['status'], scores=result.get('scores'),
                                              model=body.get('modelVersion'), raw_attempt=str(attempt_path),
                                              raw_attempt_sha256=file_hash(attempt_path)))
                    return
                await asyncio.sleep(min(30, 5 * 2 ** (number - 1)))
    proxy = config['protocol'].get('http_proxy')
    if proxy:
        parsed = urlsplit(proxy)
        if parsed.hostname not in ('127.0.0.1', 'localhost') or parsed.username or parsed.password:
            raise ValueError('Only the repository default local proxy without embedded credentials is allowed')
    async with httpx.AsyncClient(timeout=httpx.Timeout(max(3600, config['protocol']['timeout_seconds']), connect=30),
                                 trust_env=False, proxy=proxy) as client:
        units = read(out / 'index.json')
        # Verify the original transport, validator and model identity with one
        # real frozen unit before admitting any concurrent new requests.
        await score(units[0], client)
        if not stop.is_set():
            await asyncio.gather(*(score(unit, client) for unit in units[1:]))
    lock.close()


def csvwrite(path, rows):
    with Path(path).open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [])
        writer.writeheader()
        writer.writerows(rows)


def evaluate(args):
    from scipy.stats import spearmanr
    out = args.output
    validate(args)
    config = read(out / 'config.json')
    # Label access occurs only here, after prompt preparation/validation.
    label_path = Path(config['source_base']) / 'offline_labels.json'
    if file_hash(label_path) != config['source_hashes'][str(label_path)]:
        raise ValueError('Frozen labels changed')
    labels = {(row['group_id'], row['variant_id']): row for row in read(label_path)}
    units = {(unit['group_id'], unit['condition']): unit for unit in read(out / 'index.json')}
    methods = ('sequence', 'msa_only', 'msa_gemini_rank5050', 'msa_prompt',
               'structure', 'msa_prompt_structure', 'msa_gemini_rank5050_structure')
    metrics, predictions = [], []
    for assay in read(out / 'cohort.json'):
        group, ids = assay['group_id'], assay['expected_ids']
        features = read(out / 'features' / f'{group}.json')
        sequence = read(out / 'sequence_scores' / f'{assay["id"]}.json')
        scores = {'sequence': sequence['scores'], 'msa_only': features['independent_msa_scores']}
        origins = {'sequence': sequence['provenance']['source_response'],
                   'msa_only': str(out / 'features' / f'{group}.json')}
        statuses = {}
        for condition in CONDITIONS:
            unit = units[(group, condition)]
            response_path = out / 'responses' / f'{unit["id"]}.json'
            origins[condition] = str(response_path)
            if response_path.exists():
                response = read(response_path)
                if response['request_sha256'] != unit['request_sha256']:
                    raise ValueError('Response prompt/config identity differs')
                statuses[condition] = response['status']
                if response['status'] == 'ok':
                    validate_scores(response['scores'], ids, bounded=True)
                    if not str(response.get('model', '')).startswith(config['protocol']['requested_model']):
                        raise ValueError('Response model identity differs')
                    scores[condition] = response['scores']
            else:
                statuses[condition] = 'not_scored'
        fused_rows = {}
        for method, component in config['postprocessing_conditions'].items():
            if component in scores and features['full_msa_coverage']:
                fused_rows[method] = fuse_rankings(ids, scores[component], scores['msa_only'])
                scores[method] = {row['candidate_id']: row['fusion_value'] for row in fused_rows[method]}
                origins[method] = origins[component]
            else:
                statuses[method] = 'missing_MSA_or_Gemini_component'
        for method in methods:
            values = scores.get(method)
            complete = values is not None and set(values) == set(ids) and all(v is not None for v in values.values())
            rho = raw_rho = None
            reason = None
            if complete:
                validate_scores(values, ids)
                xs = [values[cid] for cid in ids]
                ys = [labels[(group, cid)]['experimental_score_mean'] for cid in ids]
                if len(set(xs)) > 1 and len(set(ys)) > 1:
                    rho = float(spearmanr(xs, ys).statistic)
                    rawx = [values[cid] for cid in ids for _ in labels[(group, cid)]['experimental_scores']]
                    rawy = [y for cid in ids for y in labels[(group, cid)]['experimental_scores']]
                    raw_rho = float(spearmanr(rawx, rawy).statistic)
                else:
                    reason = 'constant_predictions_or_labels'
                ranks = midranks(xs, descending=True)
                components = {row['candidate_id']: row for row in fused_rows.get(method, [])}
                for cid, rank in zip(ids, ranks):
                    row = components.get(cid, {})
                    predictions.append(dict(group_id=group, reference_complex_id=assay['reference_complex_id'],
                        method=method, candidate_id=cid, score=values[cid], final_rank=rank,
                        gemini_score=row.get('gemini_score'), msa_score=row.get('msa_score'),
                        gemini_rank_percentile=row.get('gemini_rank_percentile'),
                        msa_rank_percentile=row.get('msa_rank_percentile'), fusion_value=row.get('fusion_value'),
                        cache_or_score_source=origins[method], source_sha256=file_hash(Path(origins[method])),
                        status='ok', msa_source=str(out / 'features' / f'{group}.json') if method in config['postprocessing_conditions'] else None))
            else:
                # Preserve every ID and available component, but never rank a
                # smaller MSA subset or silently substitute missing evidence.
                component = config['postprocessing_conditions'].get(method)
                gemini = scores.get(component) if component else None
                g_percentiles = dict(zip(ids, [(r - 1) / (len(ids) - 1)
                    for r in midranks([gemini[cid] for cid in ids])])) if gemini else {}
                origin = origins.get(method, origins.get(component, ''))
                for cid in ids:
                    predictions.append(dict(group_id=group, reference_complex_id=assay['reference_complex_id'],
                        method=method, candidate_id=cid, score=values.get(cid) if values else None, final_rank=None,
                        gemini_score=gemini.get(cid) if gemini else None,
                        msa_score=scores['msa_only'].get(cid) if component else None,
                        gemini_rank_percentile=g_percentiles.get(cid), msa_rank_percentile=None, fusion_value=None,
                        cache_or_score_source=origin,
                        source_sha256=file_hash(Path(origin)) if origin and Path(origin).is_file() else None,
                        status=statuses.get(method, 'missing_MSA'),
                        msa_source=str(out / 'features' / f'{group}.json') if component else None))
            metrics.append(dict(group_id=group, reference_complex_id=assay['reference_complex_id'],
                                assay_group_id=assay['assay_group_id'], method=method, N=10, CR=25,
                                scored=sum(v is not None for v in values.values()) if values else 0,
                                status='complete' if complete else statuses.get(method, 'missing_MSA'),
                                spearman=rho, record_level_spearman=raw_rho, reason=reason))
    attempts = [read(path) for path in sorted((out / 'attempts').glob('*/*.json'))]
    counts = {method: sum(row['method'] == method and row['status'] == 'complete' for row in metrics)
              for method in methods}
    means = {method: statistics.mean(values) if (values := [row['spearman'] for row in metrics
                                                          if row['method'] == method and row['spearman'] is not None]) else None
             for method in methods}
    summary = dict(assays=10, candidates=100, N=10, CR=25, complete_assays=counts,
                   mean_signed_spearman=means, reused_sequence_calls=10,
                   new_api_calls=len(attempts), successful_new_calls=sum(a['status'] == 'ok' for a in attempts),
                   attempts_by_status={status: sum(a['status'] == status for a in attempts)
                                       for status in sorted({a['status'] for a in attempts})},
                   reported_tokens=sum((a.get('usage') or {}).get('totalTokenCount', 0) for a in attempts),
                   unfinished=[row for row in metrics if row['status'] != 'complete'],
                   original_results_preserved=True, labels_used_only_for_evaluation=True,
                   fusion_weights=[0.5, 0.5], config_sha256=file_hash(out / 'config.json'))
    summary['paired_comparisons'] = {}
    for method, comparator in [('msa_gemini_rank5050', 'sequence'),
                               ('msa_gemini_rank5050_structure', 'structure'),
                               ('msa_prompt', 'sequence'), ('structure', 'sequence'),
                               ('msa_prompt_structure', 'msa_prompt')]:
        pair = [(row['spearman'], other['spearman']) for row in metrics for other in metrics
                if row['method'] == method and other['method'] == comparator and
                row['group_id'] == other['group_id'] and row['spearman'] is not None and other['spearman'] is not None]
        summary['paired_comparisons'][method + '_vs_' + comparator] = dict(
            assays=len(pair), method_mean=statistics.mean(a for a, _ in pair) if pair else None,
            comparator_mean=statistics.mean(b for _, b in pair) if pair else None,
            mean_delta=statistics.mean(a - b for a, b in pair) if pair else None)
    csvwrite(out / 'per_assay.csv', metrics)
    csvwrite(out / 'rankings.csv', predictions)
    write(out / 'summary.json', summary)
    lines = ['# Protein–RNA offline ablation (N=10, CR=25)', '',
             'Original candidates, data split, seed and signed Spearman are unchanged. '
             'All old files are preserved. New prompt scores are separate from fixed 50:50 postprocessing.', '',
             'Fusion uses ascending average-midrank percentiles; higher is better and final rank 1 is best. '
             'Ties use average ranks, missing scores leave the full assay incomplete. No weight fitting.', '',
             f'Reused sequence calls: 10. New API attempts: {len(attempts)}. '
             f'Successful new calls: {summary["successful_new_calls"]}. Reported tokens: {summary["reported_tokens"]}.', '',
             '| Complex / assay suffix | Sequence | MSA fusion | MSA prompt | Structure | MSA+Structure | Structure fusion |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for assay in read(out / 'cohort.json'):
        local = {row['method']: row for row in metrics if row['group_id'] == assay['group_id']}
        display = []
        for method in ('sequence', 'msa_gemini_rank5050', 'msa_prompt', 'structure', 'msa_prompt_structure', 'msa_gemini_rank5050_structure'):
            row = local[method]
            display.append(f'{row["spearman"]:.4f}' if row['spearman'] is not None else row['status'])
        lines.append('| ' + assay['reference_complex_id'] + ' / ' + assay['group_id'].split('__')[-1][:8] + ' | ' + ' | '.join(display) + ' |')
    lines += ['', 'Partner definition: unchanged base-task protein/RNA chain groups; only designated partner chains contribute contacts and SASA removal. '
              'Contact counts are distinct partner residues with a heavy atom within 5 A. '
              'SASA uses Bio.PDB ShrakeRupley, 1.4 A probe and 100 points. Missing site heavy atoms produce null; no atom imputation.', '',
              'MSA: existing inverse 80% identity-neighbor weighting (50% overlap), sorted unique non-WT homologs, cap 2048, '
              'total pseudocount 1. Homologs are selected by deterministic SHA256 ordering, cap 2048, seed 42. Low Neff is an explicit quality flag; strict Neff>=20 scores are retained separately. Whole ambiguous homologs are excluded before reweighting; retained unknown fraction is zero. '
              'These are empirical amino-acid frequencies, not homology probabilities. Full 20-AA distributions are in frozen_probability_tables; '
              'msa_column, insertion codes and source hashes are in mapping_cache.', '',
              f'Unfinished assay/condition units: {len(summary["unfinished"])}. See summary.json for exact statuses.', '',
              'Artifacts: config.json; freeze.json; cache_audit.json; validation.json; prompts/; requests/; '
              'responses/; attempts/; sequence_scores/; features/; mapping_cache/; partner_mapping/; '
              'frozen_probability_tables/; msa_weights/; rankings.csv; per_assay.csv.']
    (out / 'README.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'validate', 'run', 'evaluate'))
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--credential-file', type=Path)
    parser.add_argument('--use-default-proxy', action='store_true',
                        help='Preparation only: freeze the repository existing default Gemini proxy transport')
    parser.add_argument('--resume-auth-failures', action='store_true',
                        help='After supplying valid authentication, retry only proven gateway token rejection; preserve failure records')
    args = parser.parse_args()
    args.root, args.output = args.root.resolve(), args.output.resolve()
    if args.command != 'prepare':
        bind_contract_snapshot(args)
    if args.command == 'run':
        if args.credential_file is None:
            parser.error('run requires --credential-file')
        asyncio.run(run(args))
    else:
        globals()[args.command](args)


if __name__ == '__main__':
    main()
