"""Frozen molecule protocol: official rows, paired SMILES/3D, historical CR groups.

No labels, model calls, conformer generation or SMILES deduplication in this module.
Classification scores are ordered propensities, not calibrated probabilities.
"""
import argparse
import copy
from pathlib import Path

from shared_protocol import load, save, sha, digest, encode, require, finite, SYSTEM_PATH

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
VERSION = 'evagent-admet-molecules-1'
ARMS = ('smiles', 'smiles_structure')
FEATURES = ('NPR1', 'NPR2', 'RadiusOfGyration', 'Asphericity', 'SpherocityIndex', 'PBF')
DEFINITION = {
    'profile_id': 'admet_saved_rdkit_shape_v1',
    'computation': 'Median descriptors over up to three saved ETKDGv3 conformers, after MMFF94 or UFF optimization attempts. Conformer seed 1. No guarantee of convergence or minimum energy. No new conformers generated.',
    'interpretation': 'Isolated molecule geometry, not experimental structure, a bound pose, binding energy or a measured endpoint. Missing evidence is unknown.',
    'fields': dict(NPR1='smallest/largest principal moment of inertia, dimensionless',
        NPR2='middle/largest principal moment of inertia, dimensionless',
        RadiusOfGyration='mass-weighted radius of gyration, angstrom',
        Asphericity='mass-weighted gyration asphericity, dimensionless',
        SpherocityIndex='unweighted gyration spherocity, dimensionless',
        PBF='mean distance from plane of best fit, angstrom')}


def shape_valid(row):
    require(isinstance(row, dict), 'Shape must be an object')
    if row.get('status') == 'unavailable':
        require(set(row) == {'status', 'reason'} and bool(row['reason']), 'Missing shape needs a reason')
        return
    require(set(row) == {'status', 'conformers', 'optimization', 'shape'}, 'Unexpected shape fields')
    require(row['status'] == 'available' and type(row['conformers']) is int and 1 <= row['conformers'] <= 3,
            'Invalid conformer count')
    require(row['optimization'] in {'MMFF94', 'UFF'}, 'Unknown saved optimizer')
    require(set(row['shape']) == set(FEATURES), 'Missing/extra shape descriptors')
    require(all(finite(v) and v >= 0 for v in row['shape'].values()), 'Nonfinite/negative shape')
    require(all(row['shape'][k] <= 1.0001 for k in ('NPR1', 'NPR2', 'Asphericity', 'SpherocityIndex')),
            'Invalid dimensionless descriptor')
    require(row['shape']['NPR1'] <= row['shape']['NPR2'], 'Reversed principal moments')


def render(tasks, shapes, groups, system):
    indexed = {}; queue = []; seen = set(); per_round = {}
    for task in tasks:
        require(set(task) == {'endpoint', 'kind', 'description', 'candidates'}, 'Unexpected task fields')
        ep = task['endpoint']
        require(ep not in indexed and task['kind'] in {'classification', 'regression'} and task['description'], 'Invalid endpoint')
        candidates = task['candidates']; ids = [c['id'] for c in candidates]
        require(ids and len(ids) == len(set(ids)), 'Duplicate molecule row ID')
        require(all(set(c) == {'id', 'smiles'} and isinstance(c['smiles'], str) and c['smiles'] for c in candidates),
                'Unexpected candidate fields or missing SMILES')
        require(set(shapes[ep]) == set(ids), 'Shape/molecule identity mismatch')
        for c in candidates: shape_valid(shapes[ep][c['id']])
        indexed[ep] = task
    require(set(shapes) == set(indexed), 'Unexpected shape endpoints')
    for group in groups:
        require(set(group) == {'endpoint', 'round', 'group', 'ids'}, 'Unexpected grouping fields')
        ep, r, g, ids = (group[k] for k in ('endpoint', 'round', 'group', 'ids'))
        require(ep in indexed and type(r) is int and r in (0, 1, 2) and type(g) is int and g >= 0, 'Invalid group')
        require((ep,r,g) not in seen and 1 <= len(ids) <= 25 and len(ids) == len(set(ids)), 'Duplicate/invalid CR25 group')
        seen.add((ep,r,g)); per_round.setdefault((ep,r), []).extend(ids)
        task = indexed[ep]; candidates = {c['id']: c for c in task['candidates']}
        require(set(ids) <= set(candidates), 'Unknown molecule ID')
        for arm in ARMS:
            evidence = dict(status='not_requested', profile_id=None, definition=None, molecules=[])
            if arm == 'smiles_structure':
                values = [shapes[ep][i]['status'] for i in ids]
                evidence = dict(status='available' if all(v == 'available' for v in values) else
                    'unavailable' if all(v == 'unavailable' for v in values) else 'partial',
                    profile_id=DEFINITION['profile_id'], definition=DEFINITION,
                    molecules=[dict(variant_id=i, **copy.deepcopy(shapes[ep][i])) for i in ids])
            interpretation = ('Higher score means greater propensity for the positive class defined by the endpoint; '
                'this is not a calibrated probability.' if task['kind'] == 'classification' else
                'Higher score means a higher numerical endpoint readout in the supplied definition; '
                'this is an ordinal score, not a prediction in physical units.')
            user = dict(schema_version=VERSION,
                task=dict(dataset_id='ADMET_TDC22', assay_id=ep, modality='admet', kind=task['kind'],
                    assay_description=task['description'], score_target=dict(higher_score_means=interpretation)),
                reference=dict(kind='independent_molecules', shared_wild_type=False),
                candidates=[dict(variant_id=i, kind='molecule', smiles=candidates[i]['smiles']) for i in ids],
                evidence=dict(structure=evidence),
                output_contract=dict(format='json', root_key='variants', item_fields=['variant_id','predicted_score'],
                    expected_variant_ids=ids, score=dict(min=-1, max=1, finite=True, unit='dimensionless',
                    reference_value=None, interpretation=interpretation+' Use a consistent scale across batches; do not normalize within a batch.'), extra_text=False))
            unit = dict(domain='admet', assay=ep, arm=arm, round_index=r, chunk_index=g,
                template=dict(system=system, user=encode(user), expected_ids=ids))
            unit['logical_id'] = digest(unit); queue.append(unit)
    require(set(per_round) == {(ep,r) for ep in indexed for r in range(3)}, 'Missing endpoint/round')
    for (ep,r), ids in per_round.items():
        require(len(ids) == len(set(ids)) and set(ids) == {c['id'] for c in indexed[ep]['candidates']},
                'Round does not cover the entire official test')
        indices = sorted(g for e,rr,g in seen if (e,rr) == (ep,r))
        require(indices == list(range(len(indices))), 'Noncontiguous chunk indices')
    return queue


def implementation_hashes():
    return {str(Path(__file__)): sha(__file__), str(SYSTEM_PATH): sha(SYSTEM_PATH)}


def validate_bundle(bundle):
    bundle = Path(bundle).resolve(); m = load(bundle/'MANIFEST.json')
    require(m.get('schema') == VERSION and m.get('fully_ready') is True, 'ADMET protocol not ready')
    require(sha(bundle/'MANIFEST.json') == (bundle/'MANIFEST.sha256').read_text().strip(), 'Manifest changed')
    require(m['implementation_hashes'] == implementation_hashes(), 'ADMET implementation changed')
    required = {'tasks.json','shapes.json','groups.json','SYSTEM.txt','QUEUE.json','SOURCES.json','ACCEPTANCE.json'}
    require(required <= set(m['files']), 'Missing frozen files')
    for name, h in m['files'].items():
        f = (bundle/name).resolve(); require(f.is_relative_to(bundle), 'File outside bundle')
        require(sha(f) == h, 'Frozen ADMET file changed: '+name)
    # Reconnect the archived normalized objects to immutable historical sources.
    provenance = load(bundle/'SOURCES.json')
    for path, h in provenance.items():
        require(sha(path) == h, 'Historical ADMET source changed: '+path)
    roles = m['source_roles']
    require(load(bundle/'tasks.json') == load(roles['tasks']), 'Official task replay mismatch')
    replay_shapes = {}; replay_groups = []
    for source in roles['shape_runs']:
        values = load(Path(source)/'features.json')
        require(not set(values) & set(replay_shapes), 'Duplicate source endpoint')
        replay_shapes.update(values)
        replay_groups.extend({k:r[k] for k in ('endpoint','round','group','ids')}
                             for r in load(Path(source)/'queue.json'))
    require(replay_shapes == load(bundle/'shapes.json') and replay_groups == load(bundle/'groups.json'),
            'Shapes or groups differ from frozen source')
    expected = render(load(bundle/'tasks.json'), load(bundle/'shapes.json'), load(bundle/'groups.json'),
                      (bundle/'SYSTEM.txt').read_text())
    require((bundle/'SYSTEM.txt').read_bytes() == SYSTEM_PATH.read_bytes(), 'SYSTEM mismatch')
    require(expected == load(bundle/'QUEUE.json'), 'ADMET request replay mismatch')
    require(m['queue_units'] == len(expected) and m['domains'] == {'admet': len(load(bundle/'tasks.json'))}, 'Scope mismatch')
    return m, expected


def prepare(version, here=HERE):
    from prepare_shared_prompts import safe_name
    safe_name(version)
    out, ev = here/'inputs'/version, here/'evaluator'/version
    require(not out.exists() and not ev.exists(), 'New version required')
    data = ROOT/'data/admet_anewdde'; manifest = load(data/'manifest.json')
    tp = data/'frozen/tasks.public.json'
    require(sha(tp) == manifest['hashes']['frozen/tasks.public.json'], 'Official tasks changed')
    tasks = load(tp); shapes = {}; groups = []
    provenance = {str(tp): sha(tp), str(data/'manifest.json'): sha(data/'manifest.json')}
    for part in ('run', 'remaining'):
        source = ROOT/'experiments/admet_cr_structure_v1'/part
        plan = load(source/'plan.json')
        for name in ('features.json', 'queue.json'):
            require(sha(source/name) == plan['hashes'][name], 'Saved shape/group source changed')
            provenance[str(source/name)] = sha(source/name)
        values = load(source/'features.json')
        require(not set(shapes) & set(values), 'Duplicate shape endpoint')
        shapes.update(values)
        for row in load(source/'queue.json'):
            require(row['method'] == 'cr', 'Unexpected ADMET scoring method')
            groups.append({k: row[k] for k in ('endpoint','round','group','ids')})
        provenance[str(source/'plan.json')] = sha(source/'plan.json')
    # Explicitly verify molecules against the historical tasks from which groups/features were built.
    old_tasks = ROOT/'experiments/admet_cr_elo_v1/tasks.public.json'
    require({t['endpoint']:t for t in load(old_tasks)} == {t['endpoint']:t for t in tasks},
            'Official molecule rows differ from historical feature input')
    for part in ('run','remaining'):
        plan = load(ROOT/'experiments/admet_cr_structure_v1'/part/'plan.json')
        require(sha(old_tasks) == plan['baseline_hashes']['tasks.public.json'], 'Historical task provenance mismatch')
    provenance[str(old_tasks)] = sha(old_tasks)
    queue = render(tasks, shapes, groups, SYSTEM_PATH.read_text())
    for name, obj in [('tasks.json',tasks),('shapes.json',shapes),('groups.json',groups),
                      ('QUEUE.json',queue),('SOURCES.json',provenance)]: save(out/name,obj)
    (out/'SYSTEM.txt').write_bytes(SYSTEM_PATH.read_bytes())
    refs = []
    for task in tasks:
        ep = task['endpoint']; qp = out/'queries/admet'/ep/'query.json'
        save(qp,dict(domain='admet',assay=ep,candidates=task['candidates']))
        refs.append(dict(domain='admet',assay=ep,kind=task['kind'],
            metric='auroc' if task['kind'] == 'classification' else 'spearman',
            query_path=str(qp.resolve()),query_sha256=sha(qp),
            label_path=str(data/'labels/labels.local.json'),label_sha256=manifest['hashes']['labels/labels.local.json'],
            label_container='admet_endpoint_y',label_key=ep))
    save(out/'ACCEPTANCE.json',dict(status='PASS', endpoints=len(tasks), requests=len(queue),
         official_rows=sum(len(t['candidates']) for t in tasks),
         grouping='Exact historical three rounds, CR25; no feedback; no new sampling',
         paired_arms=list(ARMS), labels_read=False, model_calls=0))
    m = dict(schema=VERSION,protocol_version=VERSION,version=version,fully_ready=True,
        queue_units=len(queue),domains={'admet':len(tasks)},arms=list(ARMS),rounds=3,
        candidate_resampling=False,labels_read=False,model_calls=0,
        source_roles=dict(tasks=str(tp),shape_runs=[str(ROOT/'experiments/admet_cr_structure_v1'/s) for s in ('run','remaining')]),
        implementation_hashes=implementation_hashes(),
        files={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()})
    save(out/'MANIFEST.json',m); (out/'MANIFEST.sha256').write_text(sha(out/'MANIFEST.json')+'\n')
    save(ev/'LABEL_REFERENCES.json',refs)
    validate_bundle(out)
    return m


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--version'); p.add_argument('--validate',type=Path)
    a = p.parse_args()
    if a.validate: print(validate_bundle(a.validate)[0]['queue_units'])
    elif a.version: print(prepare(a.version)['queue_units'])
    else: p.error('--version or --validate required')
