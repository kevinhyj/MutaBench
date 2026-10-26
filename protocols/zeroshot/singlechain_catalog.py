"""Current Protein/RNA data entry: frozen inventory plus registered extensions.

Cataloguing reads identities and hashes, never parses evaluation labels or renders
prompts. Preparation requires an explicit selection and delegates to the existing
sequence builder after resolving every assay through this same catalog.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REGISTRY = HERE / 'singlechain_sources.json'


def load(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def resolve(path, base, root):
    path = Path(path)
    if not path.is_absolute():
        path = base / path
    if path.is_file():
        return path.resolve()
    # A migrated path is accepted only after checked() verifies its frozen SHA.
    for prefix, replacement in (('/data/yanjie_huang/EVAgent/', root),
                                ('/data/yanjie_huang/', root.parent)):
        if str(path).startswith(prefix):
            return replacement / str(path)[len(prefix):]
    return path


def catalog(registry=REGISTRY, *, root=ROOT):
    registry, root = Path(registry).resolve(), Path(root).resolve()
    config = load(registry)
    require(config['schema'] == 'singlechain-sources-1', 'Unsupported source registry')
    checked_files = {str(registry): sha(registry)}

    def checked(ref, base=registry.parent):
        path = resolve(ref['path'], base, root)
        actual = checked_files.get(str(path))
        if actual is None:
            actual = sha(path)
        require(actual == ref['sha256'], 'Source hash mismatch: ' + str(path))
        checked_files[str(path)] = actual
        return path

    def references(ref):
        path = checked(ref)
        result = {}
        for row in load(path):
            key = (row['domain'], row['assay'])
            require(key not in result, 'Duplicate evaluator reference: ' + str(key))
            result[key] = row
        return result

    rows, evaluator, seen = [], [], set()

    def append(domain, aid, query_ref, source_csv, ref, origin, evidence, extension=None):
        key = (domain, aid)
        require(domain in {'protein', 'rna'}, 'Unsupported sequence domain')
        require(key not in seen, 'Duplicate assay across sources: ' + str(key))
        require(aid and Path(aid).name == aid and aid not in {'.', '..'}, 'Invalid assay name')
        qp = checked(query_ref)
        query = load(qp)
        require((query['domain'], query['assay']) == key, 'Query identity mismatch')
        require(set(query) == {'schema', 'domain', 'assay', 'n', 'seed', 'wt_sequence',
                               'assay_context', 'candidates'}, 'Unexpected query fields')
        require(query['seed'] == 42, 'Wrong query seed')
        candidates = query['candidates']
        require(0 < len(candidates) <= 250 and query['n'] == len(candidates), 'Query count mismatch')
        for index, candidate in enumerate(candidates, 1):
            require(set(candidate) == {'id', 'mutation', 'sequence', 'changed_positions'},
                    'Unexpected candidate fields')
            require(candidate['id'] == f'M{index:03d}', 'Candidate ID/order mismatch')
        require((ref['domain'], ref['assay']) == key, 'Evaluator identity mismatch')
        require(ref['query_sha256'] == query_ref['sha256'], 'Evaluator query mismatch')
        checked({'path': ref['query_path'], 'sha256': ref['query_sha256']})
        lp = checked({'path': ref['label_path'], 'sha256': ref['label_sha256']})
        sp = checked(source_csv)
        record = dict(domain=domain, assay=aid, origin=origin, n=len(candidates),
                      cohort='N250' if len(candidates) == 250 else 'short_query',
                      seed=42, query_path=str(qp), query_sha256=query_ref['sha256'],
                      candidate_ids=[c['id'] for c in candidates],
                      source_csv=dict(path=str(sp), sha256=source_csv['sha256']),
                      evidence=evidence)
        if extension:
            record['extension_manifest'] = extension
        rows.append(record)
        evaluator.append(dict(domain=domain, assay=aid, query_path=str(qp),
                              query_sha256=query_ref['sha256'], label_path=str(lp),
                              label_sha256=ref['label_sha256']))
        seen.add(key)

    base = config['historical']
    ip, pp = checked(base['inventory']), checked(base['plan'])
    cleaner = checked(base['cleaner'])
    inventory, plan, refs = load(ip), load(pp), references(base['evaluator_references'])
    expected_keys = {(d, item['assay']) for d in ('protein', 'rna') for item in inventory[d]}
    require(set(refs) == expected_keys, 'Historical evaluator coverage mismatch')
    for domain in ('protein', 'rna'):
        for item in inventory[domain]:
            aid = item['assay']
            hashes = plan['input_hashes'][domain + '/' + aid]
            evidence = {}
            for kind in ('msa', 'structure'):
                ep = checked(dict(path=str(ip.parent/'evidence'/domain/aid/(kind+'.json')),
                                  sha256=hashes[kind]))
                evidence[kind] = load(ep).get('status', 'unknown')
            append(domain, aid, dict(path=str(ip.parent/'queries'/domain/aid/'query.json'),
                                    sha256=hashes['query']), item['source_csv'], refs[(domain, aid)],
                   'historical_inventory', evidence)
            # Match the frozen cleaner's RNA representation, without rewriting query bytes.
            wt = item['wt_sequence'].strip().upper().replace('T', 'U') if domain == 'rna' else item['wt_sequence']
            require(load(rows[-1]['query_path'])['wt_sequence'] == wt, 'WT mismatch: ' + aid)

    for entry in config['extensions']:
        mp = checked(entry['manifest']); ext = load(mp)
        key = (ext['domain'], ext['assay'])
        require(ext['seed'] == 42 and ext['labels_in_query'] is False, 'Invalid extension contract')
        require(ext['cleaning_sha256'] == sha(cleaner), 'Extension cleaning version mismatch')
        checked(dict(path=ext['cleaning_source'], sha256=ext['cleaning_sha256']))
        checked(dict(path=ext['metadata_path'], sha256=ext['metadata_sha256']))
        refs = references(entry['evaluator_references'])
        require(set(refs) == {key}, 'Extension evaluator coverage mismatch')
        append(*key, dict(path=ext['query_path'], sha256=ext['query_sha256']),
               ext['source_csv'], refs[key], 'registered_extension',
               {k: ext[k+'_status'] for k in ('msa', 'structure')},
               dict(path=str(mp), sha256=entry['manifest']['sha256']))
        require(rows[-1]['n'] == ext['n'], 'Extension count mismatch')
    summary = {d: dict(assays=sum(r['domain'] == d for r in rows),
                       cohorts=dict(Counter(r['cohort'] for r in rows if r['domain'] == d)))
               for d in ('protein', 'rna')}
    # Label paths are deliberately confined to the independent evaluator output.
    public_sources = {p: h for p, h in checked_files.items()
                      if 'label' not in Path(p).name.lower()}
    return dict(schema='singlechain-catalog-1', registry_path=str(registry),
                registry_sha256=sha(registry), summary=summary, assays=rows,
                sources=public_sources, model_calls=0), evaluator


def selection_scope(data, selection):
    require(set(selection) == {'assays'} and selection['assays'], 'Explicit nonempty assays selection required')
    indexed = {(r['domain'], r['assay']): r for r in data['assays']}
    result, seen = [], set()
    for item in selection['assays']:
        require(set(item) == {'domain', 'assay'}, 'Selection accepts identity only; sources come from registry')
        key = (item['domain'], item['assay'])
        require(key in indexed and key not in seen, 'Unknown/duplicate selected assay: ' + str(key))
        row = indexed[key]
        selected = dict(item)
        if 'extension_manifest' in row:
            selected['extension_manifest'] = row['extension_manifest']
        result.append(selected); seen.add(key)
    return dict(schema='singlechain-selection-1', registry_path=data['registry_path'],
                registry_sha256=data['registry_sha256'], assays=result)


def write_new(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as handle:
        handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + '\n')


def prepare_selection(selection, version, *, registry=REGISTRY):
    # Lazy import keeps cataloguing independent of prompt/scientific dependencies.
    from prepare_monomer_sequences import prepare, safe_name
    safe_name(version)
    source_version = version + '_source_shared2'
    require(not (HERE/'inputs'/source_version).exists() and not (HERE/'evaluator'/source_version).exists(),
            'New source input version required')
    from singlechain_cohort import replay, inspect_bundle
    data, refs = catalog(registry)
    scope = selection_scope(data, load(selection))
    scope['selection_source'] = dict(path=str(Path(selection).resolve()), sha256=sha(selection))
    scope_path = HERE/'index'/'singlechain_selections'/(version + '.json')
    require(not (HERE/'inputs'/version).exists() and not (HERE/'evaluator'/version).exists(),
            'New version required')
    require(not scope_path.exists(), 'New selection version required')
    keys = {(r['domain'], r['assay']) for r in scope['assays']}
    audit = HERE/'audits'/('cohort_' + version)
    require(not audit.exists(), 'New cohort audit version required')
    proof = replay(data, entries=[r for r in data['assays'] if (r['domain'], r['assay']) in keys],
                   detail_path=audit/'CANDIDATES.jsonl')
    write_new(audit/'REPLAY.json', proof)
    require(proof['all_pass'], 'Cohort replay failed; no input prepared')
    scope['cohort_replay'] = dict(path=str(audit/'REPLAY.json'), sha256=sha(audit/'REPLAY.json'))
    write_new(scope_path, scope)
    prepare(scope_path, source_version)
    source = HERE/'inputs'/source_version
    check = inspect_bundle(source, data, refs, expected_keys=keys, evaluator_root=HERE/'evaluator')
    write_new(audit/'SOURCE_COHORT_CHECK.json', check)
    require(check['complete'], 'Prepared cohort differs from frozen selection; not accepted')
    from protein_msa_guard import check_bundle
    write_new(audit/'PROTEIN_MSA_CHECK.json', check_bundle(source))
    from rna_msa_guard import check_bundle as check_rna_msa
    write_new(audit/'RNA_MSA_CHECK.json', check_rna_msa(source))
    from rna_structure_guard import check_bundle as check_rna_structure
    write_new(audit/'RNA_STRUCTURE_CHECK.json', check_rna_structure(source, registry=registry))
    # Keep the legacy feature artifact and its guards immutable; only the new
    # monomer protocol can be returned as the current prepared prompt input.
    from monomer_sequence_semantics_v3 import prepare as prepare_semantics
    result = prepare_semantics(version, parent=source, here=HERE)
    check = inspect_bundle(HERE/'inputs'/version, data, refs, expected_keys=keys, evaluator_root=HERE/'evaluator')
    write_new(audit/'COHORT_CHECK.json', check)
    require(check['complete'], 'Revised prompt cohort differs from frozen selection')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registry', type=Path, default=REGISTRY)
    commands = parser.add_subparsers(dest='command', required=True)
    check = commands.add_parser('catalog', help='Verify all registered sources; no model calls')
    check.add_argument('--output', type=Path, help='New metadata-only catalog file')
    check.add_argument('--evaluator-output', type=Path, help='New independent label reference file')
    build = commands.add_parser('prepare', help='Prepare only an explicit assay selection')
    build.add_argument('--selection', type=Path, required=True)
    build.add_argument('--version', required=True)
    args = parser.parse_args()
    if args.command == 'catalog':
        data, refs = catalog(args.registry)
        for path in (args.output, args.evaluator_output):
            if path is not None and path.exists():
                raise FileExistsError(path)
        if args.output:
            write_new(args.output, data)
        if args.evaluator_output:
            write_new(args.evaluator_output, refs)
        print(json.dumps(data['summary'], ensure_ascii=False))
    else:
        result = prepare_selection(args.selection, args.version, registry=args.registry)
        print(json.dumps({k: result[k] for k in ('version', 'domains', 'queue_units', 'fully_ready')}))


if __name__ == '__main__':
    main()
