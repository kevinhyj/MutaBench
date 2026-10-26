"""Migrate frozen PPI features/requests into a new pipeline version, offline.

This adapter repairs missingness only. It retains the historical request layout
and evidence-arm scope; a shared four-arm protocol still needs its own renderer.
"""
import argparse
import copy
import json
from pathlib import Path

from msa_missingness import VERSION, encode, rebuild_ppi_user, reidentify, repair_complex_features, sha, split_ppi_user

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
COHORTS = ('ppi_cr25_from_new_serialization_v2_20261004_r4_completion',
           'ppi_cr25_previously_excluded67_20261004_r2_completion')


def load(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        stream.write(encode(value) + '\n')


def prepare(output, sources=None):
    if output.exists():
        raise FileExistsError('New PPI input version required')
    sources = sources if sources is not None else [ROOT / 'experiments' / name for name in COHORTS]
    output.mkdir(parents=True)
    queue, assays, provenance, seen = [], [], [], set()
    for old in sources:
        freeze = load(old / 'freeze.json')
        hashes = freeze['artifact_hashes']

        def verified(relative):
            p = old / relative
            if relative not in hashes or sha(p) != hashes[relative]:
                raise ValueError('Frozen PPI source mismatch: ' + str(p))
            return load(p)

        plan = verified('plan.json')
        features = {}
        tasks = {}
        for group in sorted({unit['group_id'] for unit in plan}):
            if group in seen:
                raise ValueError('Duplicate PPI assay across cohorts')
            seen.add(group)
            task = verified(f'prepared/{group}.json')
            before = verified(f'features/{group}.json')
            if set(before['msa']) != {c['id'] for c in task['candidates']}:
                raise ValueError('PPI task/feature candidates differ')
            after = repair_complex_features(before)
            features[group], tasks[group] = after, task
            save(output / 'assays' / group / 'task.json', task)
            save(output / 'assays' / group / 'features.json', after)
            changes = sum(before['msa'][cid][sid] != row for cid, sites in after['msa'].items() for sid, row in sites.items())
            assays.append(dict(domain='protein_protein', assay=group, n=len(task['candidates']),
                               changed_site_occurrences=changes, source_directory=str(old)))
        for unit in plan:
            group, ids = unit['group_id'], unit['expected_ids']
            candidates = tasks[group]['candidates']
            if [c['id'] for c in candidates[unit['batch']*25:(unit['batch']+1)*25]] != ids:
                raise ValueError('Frozen PPI candidate/chunk order mismatch')
            request = verified(unit['request_path'])
            if sha(old / unit['request_path']) != unit['request_sha256']:
                raise ValueError('PPI request hash differs from plan')
            if len(request['contents']) != 1 or len(request['contents'][0]['parts']) != 1:
                raise ValueError('Unsupported PPI request layout')
            repaired = copy.deepcopy(request)
            user = request['contents'][0]['parts'][0]['text']
            repaired_user = rebuild_ppi_user(user, features[group], ids)
            if ('msa' in unit['condition']) != ('msa_sites' in split_ppi_user(user)[1]['evidence']):
                raise ValueError('PPI arm/evidence mismatch')
            repaired['contents'][0]['parts'][0]['text'] = repaired_user
            save(output / 'requests' / (unit['id'] + '.json'), repaired)
            instruction = request.get('systemInstruction')
            if instruction is not None:
                raise ValueError('Unexpected native PPI system instruction')
            queue.append(reidentify(dict(domain='protein_protein', assay=group, arm=unit['condition'],
                                         chunk_index=unit['batch'], template=dict(
                                             system='', user=repaired_user, expected_ids=ids))))
        provenance.append(dict(directory=str(old), freeze_sha256=sha(old / 'freeze.json'),
                               plan_sha256=sha(old / 'plan.json'), requests=len(plan)))
    if len({u['logical_id'] for u in queue}) != len(queue):
        raise ValueError('Duplicate PPI request identity')
    save(output / 'QUEUE.json', queue)
    files = {str(p.relative_to(output)): sha(p) for p in output.rglob('*.json')}
    manifest = dict(schema='ppi-missingness-migration-v1', version=output.name, files=files,
                    source_cohorts=provenance, assays=assays, queue_units=len(queue),
                    msa_missingness_version=VERSION,
                    code_sha256={name: sha(HERE / name) for name in ['msa_missingness.py', 'prepare_ppi_missingness.py']},
                    candidate_resampling=False, model_calls=0, labels_read=False, fully_ready=False,
                    reason='Missingness repaired; historical three evidence arms retained. Shared four-arm renderer and independent evaluator integration are pending.')
    save(output / 'MANIFEST.json', manifest)
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--version', required=True)
    args = parser.parse_args()
    if not args.version.replace('_', '').replace('-', '').isalnum():
        raise ValueError('Simple version name required')
    result = prepare(HERE / 'inputs' / args.version)
    print(json.dumps(dict(assays=len(result['assays']), queue_units=result['queue_units'],
                          fully_ready=result['fully_ready'], model_calls=0)))
