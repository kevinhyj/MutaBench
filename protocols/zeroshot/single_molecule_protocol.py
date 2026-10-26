"""Versioned single-molecule inputs; old shared/complex runs remain immutable.

Protein/RNA retain the shared feature validator. ADMET has an independent
absolute-property contract and three frozen group schedules, with no WT/MSA.
This module only handles label-free inputs and never submits model requests.
"""
import copy
import json
from pathlib import Path

import shared_protocol as shared
from prepare_shared_prompts import singlechain_record, safe_name
from prompt_semantics import definitions

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
VERSION = 'evagent-single-molecule-1'
SYSTEM = (HERE/'prompts/shared_fields_v1_draft/system_prompt.txt').read_text()
load, save, sha, encode, digest = shared.load, shared.save, shared.sha, shared.encode, shared.digest
require, finite = shared.require, shared.finite


def implementation_hashes():
    names = ['single_molecule_protocol.py', 'prepare_shared_prompts.py',
             'shared_protocol.py', 'prompt_semantics.py', 'msa_missingness.py',
             'rna_confidence.py']
    return {str(HERE/n): sha(HERE/n) for n in names} | {
        str(shared.SYSTEM_PATH): sha(shared.SYSTEM_PATH),
        str(HERE/'audits/feature_contract_20261014_v1/PROFILES.json'):
            sha(HERE/'audits/feature_contract_20261014_v1/PROFILES.json')}


def predictor_check(value):
    if isinstance(value, dict):
        forbidden = {'labels', 'DMS_score', 'experimental_score', 'experimental_score_mean',
                     'label_path', 'msa_only_scores', 'independent_msa_scores', 'fitness'}
        require(not forbidden.intersection(value), 'Measured/aggregate scores in predictor input')
        for v in value.values(): predictor_check(v)
    elif isinstance(value, list):
        for v in value: predictor_check(v)
    elif isinstance(value, str):
        require('/data/' not in value and '/home/' not in value, 'Private path in predictor input')


def sequence_record(domain, assay, sources):
    record = singlechain_record(domain, assay, sources['query'],
                               {k: sources[k] for k in ('msa', 'structure')})
    shared.validate_features(record)
    msa = record['evidence']['msa']
    packet = sources['msa']
    msa['source_status'] = packet.get('source_status', 'unavailable')
    msa['source_caveat'] = packet.get('caveat', packet.get('reason', ''))
    msa['source_status_definition'] = (
        'RNA primary and exploratory are source-review tiers, distinct from per-site '
        'support and coverage. Exploratory evidence is not equivalent to primary evidence. '
        'precomputed_ProteinGym_profile identifies the frozen protein profile source.')
    if domain == 'protein':
        definition = msa['definition']
        definition.pop('historical_exception', None)
        definition['computation']['unaligned'] = (
            'Unaligned WT positions have null ratio/depth/gap and an explicit missing reason. '
            'Aligned columns without valid observations retain observed depth/gap but null ratio.')
    else:
        definition = record['evidence']['structure']['definition']
        definition['computation'].pop('confidence_actual', None)
        definition['computation']['confidence_required'] = (
            'This packet uses original predicted CIF atom confidence after source, atom identity '
            'and coordinate verification; experimental/unverified confidence is null.')
    # Coverage qualifiers are visible even when the legacy producer did not supply flags.
    record['evidence']['structure']['coverage_caveat'] = (
        'Only observed WT atoms contribute. Missing-atom completeness is not established by '
        'these legacy summary features; neighbor counts and SASA are not guarantees of complete geometry.')
    predictor_check(record)
    return record


def molecule_record(assay, sources):
    task = sources['query']
    require(set(task) == {'endpoint', 'kind', 'description', 'candidates'}, 'Unexpected ADMET fields')
    require(task['endpoint'] == assay and task['kind'] in {'regression', 'classification'}, 'Endpoint mismatch')
    require(isinstance(task['description'], str) and task['description'].strip(), 'Missing endpoint definition')
    ids = []
    for c in task['candidates']:
        require(set(c) == {'id', 'smiles'} and isinstance(c['id'], str) and c['id'], 'Invalid molecule identity')
        require(isinstance(c['smiles'], str) and c['smiles'], 'Missing full SMILES')
        ids.append(c['id'])
    require(ids and len(ids) == len(set(ids)), 'Duplicate endpoint row identity')
    require(set(sources['structure']) == set(ids), 'ADMET structure identity mismatch')
    fields = {'NPR1', 'NPR2', 'RadiusOfGyration', 'Asphericity', 'SpherocityIndex', 'PBF'}
    for f in sources['structure'].values():
        require(f.get('status') in {'available', 'unavailable'}, 'Unknown shape availability')
        if f['status'] == 'unavailable':
            require(bool(f.get('reason')) and not f.get('shape'), 'Missing shape needs reason and no numeric placeholder')
        else:
            require(set(f.get('shape', {})) == fields, 'Incomplete shape descriptors')
            require(all(finite(v) and v >= -1e-10 for v in f['shape'].values()), 'Invalid shape values')
            require(type(f.get('conformers')) is int and 1 <= f['conformers'] <= 3, 'Invalid conformer count')
    schedule = sources['schedule']
    require(schedule and {g['round'] for g in schedule} == {0, 1, 2}, 'Three frozen ADMET rounds required')
    for rnd in range(3):
        groups = [g for g in schedule if g['round'] == rnd]
        require([g['group'] for g in groups] == list(range(len(groups))), 'Noncontiguous group order')
        flattened = [i for g in groups for i in g['ids']]
        require(len(flattened) == len(ids) and set(flattened) == set(ids), 'Round must cover every official row exactly once')
        require(all(1 <= len(g['ids']) <= 25 for g in groups), 'Invalid CR25 group')
    record = dict(domain='admet', assay=assay, task=task, structure=sources['structure'], schedule=schedule,
                  definition=definitions()['admet_molecule_structure_legacy'])
    predictor_check(record)
    return record


def record_from_sources(domain, assay, sources):
    require(domain in {'protein', 'rna', 'admet'}, 'Unsupported single-molecule domain')
    safe_name(assay)
    required = {'query', 'structure', 'schedule'} if domain == 'admet' else {'query', 'msa', 'structure'}
    require(set(sources) == required, 'Unexpected source channels')
    return molecule_record(assay, sources) if domain == 'admet' else sequence_record(domain, assay, sources)


def render(record):
    domain, assay = record['domain'], record['assay']
    units = []
    if domain == 'admet':
        task = record['task']; by_id = {c['id']: c for c in task['candidates']}
        for group in record['schedule']:
            ids = group['ids']
            for arm in ('smiles', 'smiles_structure'):
                contract = shared.output_contract(ids)
                contract['score'] = dict(finite=True, min=-1, max=1, unit='dimensionless',
                    interpretation='Absolute endpoint scale: -1 very low, 0 intermediate, +1 very high. '
                    'For classification higher means more likely the defined positive class. '
                    'There is no WT or neutral mutation reference; do not normalize within a batch.')
                user = dict(schema_version=VERSION,
                    task=dict(dataset_id='ADMET', assay_id=assay, modality='admet', kind=task['kind'],
                              assay_description=task['description'], metric='AUROC' if task['kind']=='classification' else 'signed Spearman'),
                    reference=None, candidates=[by_id[i] for i in ids],
                    evidence=dict(msa=dict(status='not_applicable'), structure=(
                        dict(definition=record['definition'], variants=[dict(variant_id=i, **record['structure'][i]) for i in ids])
                        if arm=='smiles_structure' else dict(status='not_requested'))), output_contract=contract)
                units.append(dict(domain=domain, assay=assay, arm=arm, round_index=group['round'],
                    chunk_index=group['group'], template=dict(system=SYSTEM, user=encode(user), expected_ids=ids)))
    else:
        candidates = record['base']['candidates']
        for start in range(0, len(candidates), 25):
            base = copy.deepcopy(record['base']); base['candidates'] = candidates[start:start+25]
            ids = [c['variant_id'] for c in base['candidates']]
            for arm in shared.ARMS:
                evidence = {}
                for kind in ('msa', 'structure'):
                    if kind in arm:
                        evidence[kind] = copy.deepcopy(record['evidence'][kind])
                        evidence[kind]['variants'] = evidence[kind]['variants'][start:start+25]
                    else:
                        evidence[kind] = dict(status='not_requested', profile_id=None, definition=None, variants=[])
                user = dict(schema_version=VERSION, **base, evidence=evidence, output_contract=shared.output_contract(ids))
                units.append(dict(domain=domain, assay=assay, arm=arm, chunk_index=start//25,
                    template=dict(system=SYSTEM, user=encode(user), expected_ids=ids)))
    for unit in units:
        predictor_check(json.loads(unit['template']['user']))
        unit['logical_id'] = digest(unit)
    return units


def add_assay(bundle, domain, assay, sources):
    record = record_from_sources(domain, assay, sources); units = render(record)
    descriptors = {}
    for kind, value in sources.items():
        path = bundle/'sources'/domain/assay/(kind+'.json'); save(path, value)
        descriptors[kind] = dict(path=str(path.relative_to(bundle)), sha256=sha(path))
    rp = bundle/'records'/domain/(assay+'.json'); save(rp, record)
    qp = bundle/'queries'/domain/assay/'query.json'; save(qp, sources['query'])
    return dict(domain=domain, assay=assay, record=str(rp.relative_to(bundle)), sources=descriptors), units


def acceptance(bundle):
    build, queue = load(bundle/'BUILD.json'), load(bundle/'QUEUE.json')
    require(build and queue, 'Empty input bundle')
    require((bundle/'SYSTEM.txt').read_text() == SYSTEM, 'SYSTEM mismatch')
    seen, offset, domains, sizes = set(), 0, {}, []
    for entry in build:
        key = (entry['domain'], entry['assay']); require(key not in seen, 'Duplicate assay'); seen.add(key)
        sources = {}
        for kind, desc in entry['sources'].items():
            path = (bundle/desc['path']).resolve()
            require(path.is_relative_to(bundle.resolve()) and sha(path) == desc['sha256'], 'Archived source mismatch')
            sources[kind] = load(path)
        record = record_from_sources(*key, sources)
        require(record == load(bundle/entry['record']), 'Record differs from source replay')
        require(load(bundle/'queries'/key[0]/key[1]/'query.json') == sources['query'], 'Evaluator query differs')
        expected = render(record)
        require(queue[offset:offset+len(expected)] == expected, 'Prompt differs from accepted sources')
        offset += len(expected); domains[key[0]] = domains.get(key[0], 0)+1
        sizes.extend(len(u['template']['system'])+len(u['template']['user']) for u in expected)
    require(offset == len(queue) and len({q['logical_id'] for q in queue}) == len(queue), 'Queue scope/identity mismatch')
    require({(r['domain'], r['assay']) for r in load(bundle/'SCOPE.json')['included']} == seen, 'Scope differs from reviewed list')
    return dict(schema=VERSION, status='PASS', domains=domains, assays=len(seen), requests=len(queue),
                max_prompt_characters=max(sizes), total_prompt_characters=sum(sizes),
                implementation_hashes=implementation_hashes(), labels_in_predictor=False,
                queue_sha256=sha(bundle/'QUEUE.json'), build_sha256=sha(bundle/'BUILD.json'))


def finalize(bundle, version):
    require(not (bundle/'MANIFEST.json').exists(), 'Cannot overwrite frozen bundle')
    report = acceptance(bundle); save(bundle/'ACCEPTANCE.json', report)
    files = {str(p.relative_to(bundle)): sha(p) for p in bundle.rglob('*') if p.is_file()}
    manifest = dict(schema=VERSION, protocol_version=VERSION, version=version, fully_ready=True,
                    files=files, acceptance=dict(path='ACCEPTANCE.json', sha256=sha(bundle/'ACCEPTANCE.json')),
                    queue_units=report['requests'], domains=report['domains'], candidate_resampling=False,
                    seed_sampling=42, generation_seed=42, admet_grouping='Frozen historical schedule; no reshuffle')
    save(bundle/'MANIFEST.json', manifest); (bundle/'MANIFEST.sha256').write_text(sha(bundle/'MANIFEST.json')+'\n')
    return manifest


def validate_bundle(bundle):
    bundle = Path(bundle).resolve(); m = load(bundle/'MANIFEST.json')
    require(m.get('fully_ready') is True and m.get('protocol_version') == m.get('schema') == VERSION, 'Unaccepted protocol')
    require(sha(bundle/'MANIFEST.json') == (bundle/'MANIFEST.sha256').read_text().strip(), 'Manifest changed')
    for name, h in m['files'].items():
        p = (bundle/name).resolve()
        require(p.is_relative_to(bundle) and sha(p) == h, 'Frozen file changed: '+name)
    require(m['acceptance'] == dict(path='ACCEPTANCE.json', sha256=sha(bundle/'ACCEPTANCE.json')), 'Acceptance not bound')
    report = acceptance(bundle)
    require(report == load(bundle/'ACCEPTANCE.json'), 'Stale acceptance')
    require(report['domains'] == m['domains'] and report['requests'] == m['queue_units'], 'Scope mismatch')
    return m, load(bundle/'QUEUE.json')


def requests_from_queue(queue):
    requests = []
    for q in queue:
        messages = [dict(role='system', content=q['template']['system']), dict(role='user', content=q['template']['user'])]
        requests.append(dict(request_id=q['logical_id'], domain=q['domain'], assay=q['assay'], arm=q['arm'],
            chunk_index=q['chunk_index'], round_index=q.get('round_index'), expected_ids=q['template']['expected_ids'],
            messages=messages, messages_sha256=digest(messages)))
    return requests
