"""One offline renderer and fail-closed acceptance gate for six sequence domains.

Records contain label-free, repaired features. Requests are always regenerated
from records, never from historical prompt prose. Acceptance is recomputed at
each execution entry; a ready flag or a rehashed queue alone is insufficient.
"""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
VERSION = 'evagent-zero-shot-shared-2'
DOMAINS = ('protein', 'rna', 'protein_protein', 'protein_rna', 'protein_dna', 'protein_ligand')
ARMS = ('sequence', 'msa_prompt', 'structure', 'msa_prompt_structure')
SYSTEM_PATH = HERE / 'prompts/shared_fields_v1_draft/system_prompt.txt'
PROFILES = {
    'protein': ('singlechain_msa_protein_legacy', 'singlechain_protein_structure_legacy'),
    'rna': ('singlechain_msa_rna_legacy', 'singlechain_rna_structure_legacy'),
    'protein_protein': ('complex_protein_site_msa_legacy', 'complex_protein_site_structure_legacy'),
    'protein_rna': ('protein_rna_msa_lexical2048_v3', 'protein_rna_structure_contact_legacy'),
    'protein_dna': ('complex_protein_site_msa_v1', 'protein_dna_partner_geometry_v1'),
    'protein_ligand': ('complex_protein_site_msa_v1', 'protein_ligand_heavy_atom_geometry_v1'),
}


def encode(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(encode(value).encode()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(encode(value) + '\n')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def finite(v):
    return isinstance(v, (float, int)) and not isinstance(v, bool) and math.isfinite(v)


def output_contract(ids):
    return dict(format='json', root_key='variants', item_fields=['variant_id', 'predicted_score'],
                expected_variant_ids=ids, score=dict(finite=True, min=-1, max=1, unit='dimensionless',
                reference_value=0, interpretation='Independent effect relative to the supplied reference in the stated assay direction. Do not normalize within a batch.'), extra_text=False)


def validate_features(record):
    """Check identity and profile-specific missingness before any arm is rendered."""
    from msa_missingness import COMPLEX_FIELDS, MISSING_STATUSES, OBSERVED_STATUSES, UNALIGNED
    from prepare_shared_prompts import feature_definition, block_status, safe_name
    domain = record['domain']; base = record['base']; candidates = base['candidates']
    safe_name(record['assay'])
    require(domain in DOMAINS and base['task']['modality'] == domain, 'Unknown/mismatched domain')
    ids = [c['variant_id'] for c in candidates]
    require(ids and len(ids) == len(set(ids)), 'Empty/duplicate candidates')
    require(set(base) == {'task', 'reference', 'candidates'}, 'Unexpected shared base fields')
    require(set(record['evidence']) == {'msa', 'structure'}, 'Both evidence channels required')
    require(base['task'].get('assay_id') and base['task'].get('score_target',{}).get('higher_score_means'), 'Explicit assay score direction required')
    for c in candidates:
        require(c['sites'] and len({s['site_id'] for s in c['sites']}) == len(c['sites']), 'Empty/duplicate mutation sites')
        require(all(s['wt'] != s['mutant'] for s in c['sites']), 'Unchanged mutation site')
    for kind, profile in zip(('msa', 'structure'), PROFILES[domain]):
        block = record['evidence'][kind]
        require(block['profile_id'] == profile, 'Unaccepted feature profile')
        require(block['status'] in {'available', 'partial', 'unavailable'}, 'Missing evidence availability')
        require(block['definition'].get('semantics_version') in {
            'prompt-semantics-20261015-v2', 'prompt-semantics-20261015-v3-msa'}, 'Missing shared feature semantics')
        require(block['definition'] == feature_definition(domain,kind), 'Feature definitions differ from accepted profile')
        require(block['status'] == block_status(block['variants'],kind,domain), 'Incorrect evidence coverage status')
        require([v['variant_id'] for v in block['variants']] == ids, 'Feature/candidate order mismatch')
        for c, variant in zip(candidates, block['variants']):
            sites = variant['sites']
            require(len(sites) == len(c['sites']), 'Missing feature site; use explicit unavailable rows')
            for identity, site in zip(c['sites'], sites):
                require({k:v for k,v in site.items() if k != 'features'} == identity, 'Feature site identity mismatch')
                f = site['features']
                if f.get('status') == 'unavailable' and set(f) <= {'status', 'reason'}:
                    require(bool(f.get('reason')), 'Unavailable feature needs a reason')
                    continue
                require(set(f) <= set(block['definition']['fields']), 'Undocumented feature fields')
                if kind == 'msa' and domain == 'protein':
                    require(f.get('status') in {'available', 'unavailable'}, 'Protein MSA was not repaired')
                    if f['status'] == 'unavailable':
                        require(f.get('site_log_frequency_ratio') is None, 'Missing MSA ratio must be null')
                        require(bool(f.get('missing_reason')), 'Missing MSA provenance')
                    if f.get('homolog_column') == UNALIGNED or f.get('missing_reason') in {'position_absent_from_profile', 'query_insertion_unaligned'}:
                        require(all(f.get(k) is None for k in ('site_log_frequency_ratio', 'non_gap_effective_depth', 'gap_fraction')), 'Unmapped MSA has numeric placeholders')
                    elif f['status'] == 'available':
                        require(finite(f.get('site_log_frequency_ratio')), 'Observed MSA needs a finite ratio')
                        require(finite(f.get('non_gap_effective_depth')) and f['non_gap_effective_depth'] > 0, 'Pseudocount-only MSA')
                elif kind == 'msa' and domain == 'rna':
                    require(finite(f.get('site_log_frequency_ratio')), 'RNA MSA ratio must be finite')
                    require(finite(f.get('non_gap_effective_depth')) and f['non_gap_effective_depth'] > 0, 'RNA MSA has no observations')
                elif kind == 'msa' and domain not in {'protein', 'rna'}:
                    status = f.get('status')
                    require(status in MISSING_STATUSES | OBSERVED_STATUSES, 'Unknown complex MSA status')
                    if status in MISSING_STATUSES:
                        require(all(f.get(k) is None for k in COMPLEX_FIELDS), 'Missing complex MSA has numeric placeholders')
                    else:
                        require(all(finite(f.get(k)) and 0 < f[k] <= 1 for k in ('p_wt','p_mutant')), 'Invalid observed frequencies')
                        require(finite(f.get('weighted_valid_count')) and f['weighted_valid_count'] > 0, 'Pseudocount-only MSA')
                        if 'log_probability_ratio' in f:
                            require(finite(f['log_probability_ratio']) and math.isclose(f['log_probability_ratio'], math.log(f['p_mutant']/f['p_wt']), abs_tol=1e-8), 'MSA log ratio mismatch')
                        require(all(finite(f.get(k)) and 0 <= f[k] <= 1 for k in ('weighted_gap_fraction','weighted_unknown_fraction')), 'Invalid MSA fractions')
                if kind == 'msa' and domain in {'protein','rna'} and f.get('gap_fraction') is not None:
                    require(finite(f['gap_fraction']) and 0 <= f['gap_fraction'] <= 1, 'Invalid gap fraction')
                if kind == 'structure' and domain == 'rna' and f.get('plddt') is not None:
                    require(block.get('confidence_status') == 'verified', 'Unverified RNA pLDDT')
                    require(finite(f['plddt']) and 0 <= f['plddt'] <= 100, 'Invalid RNA pLDDT')
                if kind == 'structure' and domain == 'protein_rna':
                    require(type(f.get('direct_rna_contact')) is bool, 'Contact unavailable must not be false')
                    require('including backbone' in block['definition']['fields']['direct_rna_contact'], 'Contact must use whole residue')
                if kind == 'structure':
                    for k in ('plddt','sasa_a2','nonlocal_ca_contacts_8a','nonlocal_neighbors_8a',
                              'min_partner_heavy_distance_A','partner_contact_count_5A','sasa_bound_A2'):
                        if f.get(k) is not None:
                            require(finite(f[k]) and f[k] >= 0, 'Invalid structure feature: '+k)
                    if f.get('plddt') is not None:
                        require(f['plddt'] <= 100, 'Invalid pLDDT')
                        if domain == 'protein':
                            require(block.get('source_kind') == 'predicted_WT_AlphaFold', 'Unverified protein pLDDT')
                    if domain not in {'protein','rna','protein_rna'}:
                        require(all(k in f for k in ('site_status','partner_status','sasa_status')), 'Missing geometry availability')
                        if f['site_status'] != 'ok' or f['partner_status'] != 'ok':
                            require(all(f.get(k) is None for k in ('min_partner_heavy_distance_A','partner_contact_count_5A','delta_sasa_partner_A2')), 'Unavailable geometry has numeric evidence')
                        if f['sasa_status'] != 'ok':
                            require(all(f.get(k) is None for k in ('sasa_bound_A2','delta_sasa_partner_A2')), 'Unavailable SASA has numeric evidence')
    rendered = encode(base) + encode(record['evidence'])
    require('/data/' not in rendered and 'source_fact_ids' not in rendered, 'Private paths or fact IDs in prompt')
    for forbidden in ('"labels":', '"experimental_score":', '"experimental_score_mean":', '"DMS_score":', '"fitness":', '"label_path":', '"msa_only_scores":', '"independent_msa_scores":'):
        require(forbidden not in rendered, 'Evaluator/aggregate score in predictor record')


def render(record, system=None):
    validate_features(record)
    system = SYSTEM_PATH.read_text() if system is None else system
    result = []
    for start in range(0, len(record['base']['candidates']), 25):
        base = copy.deepcopy(record['base']); base['candidates'] = base['candidates'][start:start+25]
        ids = [c['variant_id'] for c in base['candidates']]
        for arm in ARMS:
            evidence = {}
            for kind in ('msa', 'structure'):
                if kind not in arm:
                    evidence[kind] = dict(status='not_requested', profile_id=None, definition=None, variants=[])
                else:
                    evidence[kind] = copy.deepcopy(record['evidence'][kind])
                    evidence[kind]['variants'] = evidence[kind]['variants'][start:start+25]
            user = dict(schema_version=VERSION, **base, evidence=evidence, output_contract=output_contract(ids))
            unit = dict(domain=record['domain'], assay=record['assay'], arm=arm, chunk_index=start//25,
                        template=dict(system=system, user=encode(user), expected_ids=ids))
            unit['logical_id'] = digest(unit); result.append(unit)
    return result


def implementation_hashes():
    names = ('shared_protocol.py', 'prepare_shared_prompts.py', 'prompt_semantics.py', 'msa_missingness.py', 'rna_confidence.py', 'prepare_protein_rna_contact.py', 'zeroshot_msa.py')
    return {name:sha(HERE/name) for name in names} | {'SYSTEM':sha(SYSTEM_PATH), 'PROFILES':sha(HERE/'audits/feature_contract_20261014_v1/PROFILES.json'),
        'site_identity':sha(HERE.parents[1]/'code/evagent/offline_ablation.py')}


def acceptance(bundle):
    """Recompute full four-arm/query/feature checks, binding exact input bytes."""
    bundle = Path(bundle)
    from prepare_shared_prompts import replay_records
    replay_records(bundle)
    index = load(bundle/'RECORDS.json'); queue = load(bundle/'QUEUE.json')
    require(index and queue, 'Empty shared input')
    require((bundle/'SYSTEM.txt').read_bytes() == SYSTEM_PATH.read_bytes(), 'Shared SYSTEM mismatch')
    offset = 0; seen = set(); feature_hashes = {}; domains = {}; queries = {}
    for name in index:
        path = (bundle/name).resolve()
        require(path.is_relative_to(bundle.resolve()), 'Record outside bundle')
        record = load(path); key = (record['domain'], record['assay'])
        require(key not in seen, 'Duplicate assay'); seen.add(key)
        expected = render(record)
        require(queue[offset:offset+len(expected)] == expected, 'Requests differ from shared renderer/accepted features')
        offset += len(expected); feature_hashes[name] = sha(path)
        qp = bundle/'queries'/record['domain']/record['assay']/'query.json'
        ids = [c['variant_id'] for c in record['base']['candidates']]
        require([c['id'] for c in load(qp)['candidates']] == ids, 'Evaluation query identity mismatch')
        queries[str(qp.relative_to(bundle))] = sha(qp)
        domains[key[0]] = domains.get(key[0], 0) + 1
    require(offset == len(queue) and len({q['logical_id'] for q in queue}) == len(queue), 'Unexpected/duplicate requests')
    provenance = load(bundle/'SOURCES.json')
    require(provenance and all(isinstance(h,str) and len(h)==64 for h in provenance.values()), 'Missing source provenance')
    return dict(schema=VERSION, status='PASS', failures=[], feature_hashes=feature_hashes, query_hashes=queries,
                queue_sha256=sha(bundle/'QUEUE.json'), records_sha256=sha(bundle/'RECORDS.json'),
                sources_sha256=sha(bundle/'SOURCES.json'), system_sha256=sha(bundle/'SYSTEM.txt'),
                build_sha256=sha(bundle/'BUILD.json'),
                implementation_hashes=implementation_hashes(), assays=len(seen), domains=domains, requests=len(queue),
                four_arms=True, feature_validation=True, labels_read=False, model_calls=0)


def finalize(bundle, version):
    bundle = Path(bundle)
    require(not (bundle/'MANIFEST.json').exists(), 'Cannot refinalize frozen input')
    report = acceptance(bundle)
    save(bundle/'ACCEPTANCE.json', report)
    files = {str(p.relative_to(bundle)):sha(p) for p in bundle.rglob('*') if p.is_file()}
    manifest = dict(schema=VERSION, protocol_version=VERSION, version=version, fully_ready=True,
                    acceptance=dict(path='ACCEPTANCE.json', sha256=sha(bundle/'ACCEPTANCE.json')),
                    files=files, queue_units=report['requests'], domains=report['domains'], seed_sampling=42,
                    arms=list(ARMS), candidate_resampling=False, labels_read=False, model_calls=0)
    save(bundle/'MANIFEST.json', manifest)
    (bundle/'MANIFEST.sha256').write_text(sha(bundle/'MANIFEST.json')+'\n')
    return manifest


def validate_bundle(bundle):
    bundle = Path(bundle).resolve(); m = load(bundle/'MANIFEST.json')
    require(m.get('fully_ready') is True, 'Explicit fully_ready=true required')
    require(m.get('schema') == m.get('protocol_version') == VERSION, 'Current shared protocol required')
    require(sha(bundle/'MANIFEST.json') == (bundle/'MANIFEST.sha256').read_text().strip(), 'Manifest checksum changed')
    required = {'QUEUE.json','SYSTEM.txt','RECORDS.json','SOURCES.json','BUILD.json','ACCEPTANCE.json'}
    require(required <= set(m['files']), 'Missing acceptance/input files')
    for name, expected in m['files'].items():
        path = (bundle/name).resolve()
        require(path.is_relative_to(bundle), 'Frozen file outside input bundle')
        require(sha(path) == expected, 'Frozen input changed: '+name)
    require(m.get('acceptance') == dict(path='ACCEPTANCE.json', sha256=sha(bundle/'ACCEPTANCE.json')), 'Feature acceptance not bound to manifest')
    report = acceptance(bundle)
    require(report == load(bundle/'ACCEPTANCE.json'), 'Feature acceptance stale or mismatched')
    require(m['queue_units'] == report['requests'] and m['domains'] == report['domains'], 'Scope mismatch')
    require(set(load(bundle/'RECORDS.json')) <= set(m['files']), 'Unhashed feature records')
    require(set(report['query_hashes']) <= set(m['files']), 'Unhashed queries')
    require(m.get('arms') == list(ARMS) and m.get('seed_sampling') == 42 and m.get('candidate_resampling') is False, 'Sampling/arm contract mismatch')
    for entry in load(bundle/'BUILD.json'):
        require({v['path'] for v in entry['sources'].values()} <= set(m['files']), 'Unhashed adapter sources')
    return m, load(bundle/'QUEUE.json')


def validate_subset(queue, accepted):
    indexed = {q['logical_id']:q for q in accepted}
    require(queue and len({q['logical_id'] for q in queue}) == len(queue), 'Empty/duplicate run requests')
    require(all(indexed.get(q['logical_id']) == q for q in queue), 'Run request is not an accepted input request')


def load_requests(bundle):
    _, queue = validate_bundle(bundle)
    return [dict(request_id=q['logical_id'], domain=q['domain'], assay=q['assay'], arm=q['arm'],
                 chunk_index=q['chunk_index'], expected_ids=q['template']['expected_ids'],
                 messages=(messages:=[dict(role='system',content=q['template']['system']),dict(role='user',content=q['template']['user'])]),
                 messages_sha256=digest(messages)) for q in queue]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--inputs', type=Path, required=True)
    args = parser.parse_args(); manifest, queue = validate_bundle(args.inputs)
    print(encode(dict(status='PASS', requests=len(queue), domains=manifest['domains'], model_calls=0)))
