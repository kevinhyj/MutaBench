"""Replay frozen singlechain cohorts and gate same-cohort comparisons, offline only."""
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path

from singlechain_catalog import HERE, REGISTRY, catalog, load, require, sha, write_new
from sampling import singlechain_select

FIELDS = ('id', 'mutation', 'sequence', 'changed_positions')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False).encode()).hexdigest()


def identity(query):
    candidates = query['candidates']
    require(candidates and query['n'] == len(candidates), 'Invalid candidate count')
    require(len({c['id'] for c in candidates}) == len(candidates), 'Duplicate candidate IDs')
    require(all(set(c) == set(FIELDS) for c in candidates), 'Unexpected candidate fields')
    return dict(domain=query['domain'], assay=query['assay'], seed=query['seed'],
                wt_sequence=query['wt_sequence'],
                candidates=[{k: c[k] for k in FIELDS} for c in candidates])


def compare_queries(reference, other):
    left, right = identity(reference), identity(other)
    checks = {k: left[k] == right[k] for k in ('domain', 'assay', 'seed', 'wt_sequence')}
    checks['count'] = len(left['candidates']) == len(right['candidates'])
    checks.update({k: [c[k] for c in left['candidates']] == [c[k] for c in right['candidates']]
                   for k in FIELDS})
    return dict(match=all(checks.values()), checks=checks,
                reference_identity_sha256=digest(left), actual_identity_sha256=digest(right))


def replay(data, *, entries=None, detail_path=None):
    """Historical cleaner; independent original NumPy expression AND current sampler.

    Source score values are inspected by the frozen cleaner only for eligibility/
    duplicate handling. Neither labels nor scores are serialized into audit files.
    """
    import numpy as np
    config = load(data['registry_path'])
    cp = Path(config['historical']['cleaner']['path'])
    if not cp.is_absolute(): cp = Path(data['registry_path']).parent/cp
    require(sha(cp) == config['historical']['cleaner']['sha256'], 'Frozen cleaner changed')
    spec = importlib.util.spec_from_file_location('cohort_frozen_cleaner', cp)
    cleaner = importlib.util.module_from_spec(spec); spec.loader.exec_module(cleaner)
    selected_entries = data['assays'] if entries is None else entries
    handle = None
    if detail_path is not None:
        Path(detail_path).parent.mkdir(parents=True, exist_ok=True)
        handle = Path(detail_path).open('x')
    results = []
    try:
        for row in selected_entries:
            qp = Path(row['query_path']); require(sha(qp) == row['query_sha256'], 'Query changed before replay')
            query = load(qp)
            wt, pool, exclusions = cleaner.source_rows(
                dict(wt_sequence=query['wt_sequence'], source_csv=row['source_csv']), row['domain'])
            ordered = sorted(pool, key=lambda c: (c['mutation'], c['sequence']))
            indices = np.random.RandomState(42).permutation(len(ordered))[:min(250, len(ordered))]
            historical = [ordered[int(i)] for i in indices]
            current = singlechain_select(pool, n=250, seed=42)
            candidates = lambda rows: [dict(id=f'M{i:03d}', **{k: r[k] for k in FIELDS if k != 'id'})
                                      for i, r in enumerate(rows, 1)]
            original = dict(query, wt_sequence=wt, seed=42, n=len(historical), candidates=candidates(historical))
            rebuilt = dict(query, wt_sequence=wt, seed=42, n=len(current), candidates=candidates(current))
            checks = compare_queries(query, original)
            pipeline = compare_queries(original, rebuilt)
            result = dict(domain=row['domain'], assay=row['assay'], n=query['n'], pool_n=len(pool),
                          query_sha256=row['query_sha256'], cohort=row['cohort'],
                          identity_sha256=digest(identity(query)),
                          frozen_vs_original=checks, original_vs_pipeline=pipeline,
                          source_csv_sha256=row['source_csv']['sha256'], exclusions=exclusions,
                          pass_replay=checks['match'] and pipeline['match'])
            results.append(result)
            if handle:
                for i in range(max(len(query['candidates']), len(historical), len(current))):
                    variants = [q['candidates'][i] if i < len(q['candidates']) else None
                                for q in (query, original, rebuilt)]
                    handle.write(json.dumps(dict(domain=row['domain'], assay=row['assay'], ordinal=i+1,
                        candidate_id=variants[0]['id'] if variants[0] else None,
                        field_matches={k: all(c is not None for c in variants) and
                                       variants[0][k] == variants[1][k] == variants[2][k] for k in FIELDS},
                        frozen_sha256=digest(variants[0]), original_replay_sha256=digest(variants[1]),
                        pipeline_replay_sha256=digest(variants[2])), sort_keys=True)+'\n')
            print(f"replay {row['domain']}/{row['assay']}: {'PASS' if result['pass_replay'] else 'FAIL'}", flush=True)
    finally:
        if handle: handle.close()
    return dict(schema='singlechain-replay-1', registry_sha256=data['registry_sha256'],
                cleaner_sha256=sha(cp), sampler_sha256=sha(HERE/'sampling.py'),
                numpy_version=np.__version__, seed=42, assays=results,
                all_pass=bool(results) and all(r['pass_replay'] for r in results),
                candidates_checked=sum(r['n'] for r in results), model_calls=0)


def inspect_bundle(bundle, data, refs, *, expected_keys=None, evaluator_root=None):
    bundle = Path(bundle).resolve()
    canonical = {(r['domain'], r['assay']): r for r in data['assays']}
    gold_refs = {(r['domain'], r['assay']): r for r in refs}
    mp = bundle/'MANIFEST.json'; manifest = load(mp) if mp.exists() else {}
    sp = bundle/'SCOPE.json'; scope = load(sp) if sp.exists() else {}
    declared = scope.get('assays', scope.get('included', manifest.get('assays')))
    if declared is None and manifest.get('domain') in {'protein', 'rna'}:
        declared = [manifest]
    declared_keys = {(r['domain'], r['assay']) for r in declared
                     if r['domain'] in {'protein', 'rna'}} if declared is not None else None
    if expected_keys is not None:
        declared_keys = set(expected_keys)
    eroot = Path(evaluator_root) if evaluator_root is not None else HERE/'evaluator'
    ep = eroot/bundle.name/'LABEL_REFERENCES.json'
    new_refs = {}; errors = []
    if ep.exists():
        for r in load(ep):
            key = (r['domain'], r['assay'])
            if key in new_refs: errors.append('duplicate_evaluator_reference:'+str(key))
            new_refs[key] = r
    else: errors.append('missing_evaluator_references')
    rows = []; seen = set()
    for qp in sorted((bundle/'queries').glob('*/*/query.json')):
        if qp.parent.parent.name not in {'protein', 'rna'}: continue
        key = (qp.parent.parent.name, qp.parent.name); seen.add(key)
        qsha = sha(qp); frozen = canonical.get(key)
        result = dict(domain=key[0], assay=key[1], query_sha256=qsha, path=str(qp))
        if frozen is None:
            result.update(match=False, reason='unregistered_assay_requires_new_cohort_registration')
            rows.append(result); continue
        query = load(qp); check = compare_queries(load(frozen['query_path']), query)
        expected_hash = manifest.get('files', {}).get(str(qp.relative_to(bundle)))
        if not expected_hash and manifest.get('query_sha256') and declared_keys == {key}:
            expected_hash = manifest['query_sha256']
        ref, original_ref = new_refs.get(key, {}), gold_refs[key]
        label_match = ref.get('label_sha256') == original_ref['label_sha256']
        ref_match = ref.get('query_sha256') == qsha and ref.get('query_path') is not None
        if ref_match:
            rp = Path(ref['query_path'])
            ref_match = rp.is_file() and sha(rp) == qsha
        if label_match:
            lp = Path(ref.get('label_path', ''))
            label_match = lp.is_file() and sha(lp) == original_ref['label_sha256']
        result.update(comparison=check, n=query['n'], cohort=frozen['cohort'],
                      query_bytes_match=qsha == frozen['query_sha256'],
                      manifest_query_hash_match=qsha == expected_hash,
                      label_reference_match=label_match, evaluator_query_match=ref_match,
                      label_sha256=ref.get('label_sha256'),
                      match=check['match'] and qsha == frozen['query_sha256'] and
                            qsha == expected_hash and label_match and ref_match)
        rows.append(result)
    if declared_keys is None: errors.append('missing_declared_scope')
    if declared_keys is not None and seen != declared_keys: errors.append('declared_scope_mismatch')
    if {k for k in new_refs if k[0] in {'protein', 'rna'}} != seen: errors.append('evaluator_scope_mismatch')
    if not mp.exists(): errors.append('unfinished_bundle_no_manifest')
    complete = bool(rows) and not errors and all(r['match'] for r in rows)
    signature = digest(sorted((r['domain'], r['assay'], r['query_sha256'], r.get('label_sha256')) for r in rows))
    return dict(schema='singlechain-cohort-check-1', bundle=str(bundle), version=bundle.name,
                manifest_sha256=sha(mp) if mp.exists() else None, registry_sha256=data['registry_sha256'],
                complete=complete, errors=errors, assays=rows, assay_count=len(rows),
                cohort_signature=signature, same_as_full_registry=complete and seen == set(canonical),
                missing_from_registry_scope=[list(k) for k in sorted(set(canonical)-seen)],
                missing_declared_assays=[list(k) for k in sorted((declared_keys or set())-seen)],
                counts=dict(Counter(r['domain'] for r in rows)), model_calls=0)


def comparison_gate(left, right):
    """No overall means across different assay sets; common exact queries are explicit."""
    def eligible(report):
        if not report['complete']: return {}
        return {(r['domain'], r['assay']): (r['query_sha256'], r['label_sha256'])
                for r in report['assays'] if r['match']}
    a, b = eligible(left), eligible(right)
    common = sorted(k for k in a.keys() & b.keys() if a[k] == b[k])
    same = bool(a) and a == b
    return dict(same_full_cohort=same, direct_overall_comparison_allowed=same,
                reason='same_queries_and_labels' if same else 'different_or_unverified_cohorts',
                common_exact_assays=[list(k) for k in common],
                comparison_policy='Recompute on explicit common assays; short queries remain supplemental',
                cohort_change_requires_new_version=True)


def require_same_cohort(left, right):
    result = comparison_gate(left, right)
    require(result['same_full_cohort'], 'Cohort mismatch: separate versions and explicit common-assay comparison required')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registry', type=Path, default=REGISTRY)
    parser.add_argument('--output', type=Path, required=True, help='New audit directory')
    parser.add_argument('--bundles', nargs='*', type=Path, default=[])
    args = parser.parse_args()
    require(not args.output.exists(), 'New audit directory required')
    data, refs = catalog(args.registry)
    proof = replay(data, detail_path=args.output/'CANDIDATES.jsonl')
    write_new(args.output/'REPLAY.json', proof)
    bundles = [inspect_bundle(p, data, refs) for p in args.bundles]
    write_new(args.output/'BUNDLES.json', bundles)
    comparisons = [dict(left=a['version'], right=b['version'], **comparison_gate(a, b))
                   for i, a in enumerate(bundles) for b in bundles[i+1:]]
    write_new(args.output/'COMPARISONS.json', comparisons)
    write_new(args.output/'MANIFEST.json', dict(schema='singlechain-cohort-audit-1',
        registry_sha256=data['registry_sha256'], code_sha256=sha(__file__),
        files={p.name: sha(p) for p in args.output.iterdir() if p.is_file()},
        replay_pass=proof['all_pass'], candidates_checked=proof['candidates_checked'], model_calls=0))
    require(proof['all_pass'], 'Historical sampling replay failed; inspect audit')
    print(json.dumps(dict(assays=len(proof['assays']), candidates=proof['candidates_checked'],
                         replay_pass=proof['all_pass'], bundles=len(bundles))))


if __name__ == '__main__':
    main()
