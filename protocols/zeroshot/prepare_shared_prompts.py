"""Offline shared input builder. Sources are structured data, never old prompt prose.

--spec supplies an explicit assay list with hashed query/msa/structure descriptors.
Complex assays also supply a hashed context object containing task and reference.
Every consumed input is copied into the new bundle and replayed during acceptance.
"""
import argparse
import copy
import re
from pathlib import Path

from shared_protocol import (HERE, DOMAINS, PROFILES, SYSTEM_PATH, finalize, load,
                             render, require, save, sha)
from prompt_semantics import definitions, repair_shared_user
from msa_missingness import repair_complex_row


def safe_name(value):
    require(isinstance(value, str) and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', value),
            'Unsafe version/assay name')
    return value


def feature_definition(domain, kind):
    profile = PROFILES[domain][kind == 'structure']
    if domain in {'protein_dna', 'protein_ligand'}:
        user = dict(task=dict(modality=domain), evidence={kind:dict(
            status='unavailable', definition={}, variants=[])})
        return repair_shared_user(user)['evidence'][kind]['definition']
    result = definitions()['protein_rna_msa_v2' if profile == 'protein_rna_msa_lexical2048_v3' else profile]
    if kind == 'msa' and domain in {'protein', 'protein_protein', 'protein_rna'}:
        result['semantics_version'] = 'prompt-semantics-20261015-v3-msa'
        result.pop('historical_exception', None)
        if domain == 'protein':
            result['computation']['unaligned'] = ('Unmapped or unaligned sites have null log ratio, depth and gap, '
                'with an explicit missing reason and homolog-column marker. At aligned sites with actual zero valid '
                'observations, retain the observed depth/gap but set the pseudocount-only log ratio to null.')
        elif domain == 'protein_protein':
            result['computation']['missing'] = ('In this packet, unavailable MSA, WT mismatch and unsupported columns '
                'have null probabilities, log ratio, weighted valid count, gap and unknown fractions, with an explicit '
                'status. No 0/1 placeholders represent unavailable evidence. Valid-column unknown fraction is zero '
                'because entire ambiguous homologs are excluded.')
        else:
            result['computation']['selection'] = ('Project homologs to WT match columns; deduplicate; exclude projected '
                'WT-identical and ambiguous homologs; sort sequences lexicographically and use the first 2048. '
                'No random or hash subsampling; fewer than 2048 valid homologs are all retained.')
    if kind == 'msa' and domain in {'protein', 'rna'}:
        result['fields'].update(status='available or unavailable; missing is unknown',
            missing_reason='Reason for missing observations or mapping',
            homolog_column='Source alignment column or explicit unaligned status')
    return result


def unavailable(reason):
    require(bool(reason), 'Missing evidence needs an explicit reason')
    return dict(status='unavailable', reason=reason)


def block_status(variants, kind, domain):
    states = []
    for variant in variants:
        for site in variant['sites']:
            f = site['features']
            if f.get('status') == 'unavailable' or f.get('status') in {
                    'no_valid_homolog_at_column', 'unavailable_MSA', 'WT_identity_mismatch'}:
                states.append('unavailable')
            elif kind == 'msa':
                key = 'site_log_frequency_ratio' if domain in {'protein','rna'} else 'p_mutant'
                states.append('available' if f.get(key) is not None else 'unavailable')
            elif domain == 'protein_rna':
                states.append('available' if type(f.get('direct_rna_contact')) is bool else 'unavailable')
            else:
                keys = (('plddt','sasa_a2','nonlocal_ca_contacts_8a') if domain == 'protein' else
                        ('plddt','nonlocal_neighbors_8a') if domain == 'rna' else
                        ('min_partner_heavy_distance_A','partner_contact_count_5A','sasa_bound_A2','delta_sasa_partner_A2'))
                count = sum(f.get(k) is not None for k in keys)
                states.append('available' if count == len(keys) else 'partial' if count else 'unavailable')
    return ('available' if states and all(s == 'available' for s in states) else
            'partial' if any(s != 'unavailable' for s in states) else 'unavailable')


def singlechain_record(domain, assay, query, packets, context=None):
    require(domain in {'protein','rna'}, 'Not a singlechain domain')
    require(query['domain'] == domain and query['assay'] == assay, 'Query identity mismatch')
    wt = query['wt_sequence']; candidates = []
    for c in query['candidates']:
        require(set(c) == {'id','mutation','sequence','changed_positions'}, 'Unexpected query fields')
        require(len(c['sequence']) == len(wt), 'Substitution query length mismatch')
        positions = [i for i,(a,b) in enumerate(zip(wt,c['sequence']),1) if a != b]
        require(positions and positions == c['changed_positions'], 'Mutation positions mismatch')
        sites = [dict(site_id='sequence:'+str(pos), chain_id='reference', sequence_index_1based=pos,
                      wt=wt[pos-1], mutant=c['sequence'][pos-1], identity_status='reference_sequence_verified')
                 for pos in positions]
        candidates.append(dict(variant_id=c['id'], kind='sequence_variant', sequence=c['sequence'], sites=sites))
    # Preserve assay metadata, while the acceptance gate rejects measured outcomes/paths.
    task = dict(dataset_id='ProteinGym' if domain == 'protein' else 'RNAGym', assay_id=assay,
                modality=domain, assay_context=copy.deepcopy(query.get('assay_context') or {}),
                score_target=dict(kind='relative_assay_effect', reference='supplied_reference',
                                  higher_score_means='higher assay readout in the frozen benchmark direction'))
    if context is not None:
        task = copy.deepcopy(context['task'])
    base = dict(task=task, reference=dict(kind='single_sequence', molecule_type=domain,
                                        chain_id='reference', sequence=wt), candidates=candidates)
    blocks = {}
    for kind, packet in packets.items():
        variants = []; rows = packet.get('site_evidence', {})
        require(not rows or set(rows) == {c['variant_id'] for c in candidates}, 'Evidence candidates mismatch')
        for c in candidates:
            source = rows.get(c['variant_id'], [])
            require(not source or [r['position'] for r in source] == [s['sequence_index_1based'] for s in c['sites']],
                    'Evidence positions mismatch')
            sites = []
            for i, identity in enumerate(c['sites']):
                if source:
                    row = source[i]
                    for key in ('wt','mutant'):
                        require(key not in row or row[key] == identity[key], 'Evidence residue mismatch')
                    f = {k:v for k,v in row.items() if k not in {'position','wt','mutant'}}
                else:
                    require(packet.get('status') == 'unavailable', 'Missing site evidence')
                    f = unavailable(packet.get('reason', 'source_packet_unavailable'))
                sites.append(dict(identity, features=copy.deepcopy(f)))
            variants.append(dict(variant_id=c['variant_id'], sites=sites))
        block = dict(profile_id=PROFILES[domain][kind == 'structure'],
                     definition=feature_definition(domain, kind), variants=variants,
                     status=block_status(variants, kind, domain))
        if kind == 'structure':
            block['source_kind'] = packet.get('kind', 'unavailable')
            if domain == 'rna':
                block['confidence_status'] = packet.get('confidence_provenance', {}).get('status','unverified')
                block['confidence_unavailable_reason'] = packet.get('confidence_provenance',{}).get('reason')
        blocks[kind] = block
    return dict(domain=domain, assay=assay, base=base, evidence=blocks)


def complex_record(domain, assay, query, packets, context):
    """Adapt prepared complex tasks and per-candidate/per-site feature maps.

PPI/DNA/ligand structure uses the geometry map. Protein-RNA structure uses the
whole-residue geometry packet (sites list), not the old sidechain prompt text.
Context explicitly supplies assay direction, partner groups and ligand identity.
"""
    from prepare_protein_rna_contact import base_sites, contact_row
    if domain == 'protein_rna':
        from zeroshot_msa import POLICY_VERSION
        require(packets['msa'].get('msa_scoring_policy', {}).get('version') == POLICY_VERSION,
                'Protein-RNA MSA must be recomputed with the lexical2048 policy')
    require(domain in set(DOMAINS)-{'protein','rna'}, 'Unsupported complex domain')
    task = copy.deepcopy(context['task']); reference = copy.deepcopy(context['reference'])
    require(task['modality'] == domain, 'Context modality mismatch')
    require(reference.get('partners'), 'Explicit assay partner definition required')
    chains = {c['chain']:c for c in query['wt_chains']}
    require(len(chains) == len(query['wt_chains']), 'Duplicate reference chain')
    canonical_chains = [dict(chain_id=c['chain'], molecule_type=c['polymer_type'],
                            sequence=c['resolved_sequence']) for c in query['wt_chains']]
    require(reference.get('chains') == canonical_chains, 'Reference chains differ from prepared task')
    if domain == 'protein_ligand':
        require(reference.get('ligand',{}).get('smiles') and reference['ligand'].get('ccd_id'), 'Ligand identity required')
    _, original_sites = base_sites(query); candidates = []
    for c in query['candidates']:
        sites = []
        for s in c['sites']:
            ch = chains[s['chain']]; pos = s['sequence_index_1based']
            require(type(pos) is int and 1 <= pos <= len(ch['resolved_sequence']), 'Invalid sequence index')
            require(ch['resolved_sequence'][pos-1] == s['wt'], 'Complex WT identity mismatch')
            require(ch['residue_ids'][pos-1] == str(s['position'])+(s.get('insertion_code') or ''), 'PDB/sequence mapping mismatch')
            sites.append(dict(site_id=f"{s['chain']}:pdb:{s['position']}{s.get('insertion_code') or ''}",
                chain_id=s['chain'], sequence_index_1based=pos, pdb_residue_number=s['position'],
                insertion_code=s.get('insertion_code') or '', wt=s['wt'], mutant=s['mutant'],
                identity_status='reference_sequence_verified'))
        candidates.append(dict(variant_id=c['id'], kind='sequence_variant', sites=sites))
    blocks = {}
    for kind, packet in packets.items():
        variants = []; definition = feature_definition(domain, kind)
        mapping = packet.get(kind, {})
        contact = {s['site_id']:s for s in packet.get('sites',[])} if domain == 'protein_rna' and kind == 'structure' else None
        if contact is not None:
            require(len(contact) == len(packet.get('sites',[])), 'Duplicate contact geometry')
        else:
            require(not mapping or set(mapping) == {c['variant_id'] for c in candidates}, 'Complex evidence candidates mismatch')
        for c in candidates:
            sites = []; cid = c['variant_id']
            if mapping:
                require(set(mapping[cid]) == {s['site_id'] for s in original_sites[cid]}, 'Complex evidence site identity mismatch')
            for identity, original in zip(c['sites'], original_sites[cid]):
                sid = original['site_id']
                if contact is not None and sid in contact:
                    row = contact[sid]
                    f = (contact_row(row) if row.get('interface_geometry_status') == 'computed'
                         else unavailable(row.get('interface_geometry_status') or 'geometry_unavailable'))
                    f.pop('site_id', None)
                elif mapping:
                    row = copy.deepcopy(mapping[cid][sid])
                    if kind == 'msa':
                        # DNA/ligand low-Neff gates can intentionally hide frequencies.
                        if row.get('status') == 'global_neff_below_frozen_20' and (domain == 'protein_dna' or row.get('p_wt') is None):
                            f = unavailable('global_neff_below_frozen_20')
                        else:
                            row = repair_complex_row(row)
                            f = {k:v for k,v in row.items() if k in definition['fields']}
                    else:
                        f = {k:v for k,v in row.items() if k in definition['fields']}
                else:
                    require(packet.get('status') == 'unavailable', 'Missing complex feature site')
                    f = unavailable(packet.get('reason','source_packet_unavailable'))
                sites.append(dict(identity, features=f))
            variants.append(dict(variant_id=cid, sites=sites))
        blocks[kind] = dict(profile_id=PROFILES[domain][kind == 'structure'], definition=definition,
                            variants=variants, status=block_status(variants, kind, domain))
        if kind == 'msa':
            blocks[kind]['chains'] = [dict(chain_id=chain, **{k:row.get(k) for k in ('homolog_count','global_neff')})
                                     for chain,row in sorted(packet.get('msa_chains',{}).items())]
    return dict(domain=domain, assay=assay, base=dict(task=task, reference=reference,
                                                   candidates=candidates), evidence=blocks)


def record_from_sources(domain, assay, sources):
    packets = {kind:sources[kind] for kind in ('msa','structure')}
    if domain in {'protein','rna'}:
        return singlechain_record(domain, assay, sources['query'], packets, sources.get('context'))
    query = sources['query']; task = sources.get('task',query)
    require(query.get('domain',domain) == domain and query.get('assay',assay) == assay, 'Complex query identity mismatch')
    def identities(value):
        return [dict(id=c['id'],sites=c['sites']) for c in value['candidates']]
    require(identities(query) == identities(task), 'Complex task differs from frozen evaluator query')
    return complex_record(domain, assay, task, packets, sources['context'])


def add_assay(bundle, domain, assay, sources):
    """Archive data objects and bind the adapter output to those exact bytes."""
    require(domain in DOMAINS, 'Unsupported domain (ADMET requires its own protocol)')
    safe_name(assay)
    required = {'query','msa','structure'}
    allowed = required | {'context'} | ({'task'} if domain not in {'protein','rna'} else set())
    require(required <= set(sources) <= allowed, 'Unexpected source channels')
    build = load(bundle/'BUILD.json') if (bundle/'BUILD.json').exists() else []
    require((domain,assay) not in {(s['domain'],s['assay']) for s in build}, 'Duplicate assay')
    record = record_from_sources(domain, assay, sources)
    units = render(record)  # All feature checks before writing.
    paths = {}
    for kind, value in sources.items():
        path = bundle/'sources'/domain/assay/(kind+'.json'); save(path,value)
        paths[kind] = dict(path=str(path.relative_to(bundle)), sha256=sha(path))
    name = f'records/{domain}/{assay}.json'; save(bundle/name,record)
    query_path = bundle/'queries'/domain/assay/'query.json'
    if query_path.exists():
        require(load(query_path) == sources['query'], 'Existing query changed')
    else:
        save(query_path, sources['query'])
    build.append(dict(domain=domain, assay=assay, record=name, sources=paths)); save(bundle/'BUILD.json',build)
    return name, units


def replay_records(bundle):
    rows = load(bundle/'BUILD.json'); require(bool(rows), 'Missing adapter provenance')
    for entry in rows:
        sources = {}
        for kind, desc in entry['sources'].items():
            path = (bundle/desc['path']).resolve()
            require(path.is_relative_to(bundle.resolve()), 'Source outside bundle')
            require(sha(path) == desc['sha256'], 'Archived source changed')
            sources[kind] = load(path)
        expected = record_from_sources(entry['domain'],entry['assay'],sources)
        require(load(bundle/entry['record']) == expected, 'Feature record differs from source adapter')
        require(load(bundle/'queries'/entry['domain']/entry['assay']/'query.json') == sources['query'],
                'Query differs from archived source')
    require([r['record'] for r in rows] == load(bundle/'RECORDS.json'), 'Adapter/record index mismatch')


def finish(bundle, version, names, queue, provenance):
    save(bundle/'RECORDS.json',names); save(bundle/'QUEUE.json',queue)
    save(bundle/'SOURCES.json',provenance)
    (bundle/'SYSTEM.txt').write_bytes(SYSTEM_PATH.read_bytes())
    return finalize(bundle,version)


def prepare(spec_path, version, here=HERE):
    safe_name(version); spec_path = Path(spec_path).resolve(); spec = load(spec_path)
    out = here/'inputs'/version; evaluator = here/'evaluator'/version
    require(not out.exists() and not evaluator.exists(), 'Use new input/evaluator version directories')
    require(bool(spec['assays']), 'Empty assay scope')
    provenance = {str(spec_path):sha(spec_path)}; names = []; queue = []; refs = []
    # Descriptors must bind source bytes explicitly; never infer readiness from paths.
    def read(desc):
        path = (spec_path.parent/desc['path']).resolve()
        require(sha(path) == desc['sha256'], 'Source checksum mismatch: '+str(path))
        provenance[str(path)] = desc['sha256']
        return load(path)
    for entry in spec['assays']:
        sources = {k:read(v) for k,v in entry['sources'].items()}
        name, units = add_assay(out,entry['domain'],entry['assay'],sources)
        names.append(name); queue.extend(units)
    # Only reference metadata is read here, never the referenced labels.
    original_refs = read(spec['evaluator_references'])
    indexed = {(r['domain'],r['assay']):r for r in original_refs}
    require(len(indexed) == len(original_refs), 'Duplicate evaluator reference')
    for entry in spec['assays']:
        ref = dict(indexed[(entry['domain'],entry['assay'])])
        require(ref['query_sha256'] == entry['sources']['query']['sha256'], 'Evaluator/query source mismatch')
        qp = out/'queries'/entry['domain']/entry['assay']/'query.json'
        ref.update(query_path=str(qp),query_sha256=sha(qp)); refs.append(ref)
    result = finish(out,version,names,queue,provenance)
    save(evaluator/'LABEL_REFERENCES.json',refs)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', type=Path, required=True); parser.add_argument('--version', required=True)
    args = parser.parse_args(); result = prepare(args.spec,args.version)
    print({k:result[k] for k in ('version','domains','queue_units','fully_ready','model_calls')})
