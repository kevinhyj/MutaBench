"""Repair whole-residue RNA-contact guidance in new pipeline inputs, offline."""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'code'))
from evagent.offline_ablation import base_sites

VERSION = 'protein-rna-whole-residue-contact-v2'
SOURCE = ROOT / 'experiments/protein_rna_ablation_N10_CR25_20261005_structure_contact'
AUDITED_SOURCES = HERE / 'audits/feature_contract_20261014_v1/INPUT_SHA256.json'
HEADER = 'Assay-matched WT RNA contact:\n'
LEGACY_GUIDANCE = (
    HEADER + 'direct_rna_contact is true when any heavy atom of the wild-type side chain lies within 5 angstroms '
    'of the RNA used in this assay, and false when none does. '
    'rna_contact_moieties lists only nucleotide parts that participate in such a contact: base, ribose, or phosphate. '
    'These fields describe the wild-type residue against that RNA. They are not the effect of the substitution. '
    'No mutant model is supplied.\n')
GUIDANCE = (
    HEADER + 'direct_rna_contact is true when any heavy atom of the whole wild-type residue, '
    'including backbone and side-chain atoms, is within 5 angstroms (distance <= 5 angstroms) '
    'of any heavy atom of the specified RNA used in this assay. It is false only when the '
    'assay-matched geometry is available and no such contact exists. '
    'rna_contact_moieties lists the RNA nucleotide parts contacted by those same whole-residue '
    'heavy atoms: base, ribose, or phosphate; it is omitted when classification is unavailable. '
    'These fields describe the wild-type residue against that RNA. They are not the effect of '
    'the substitution. No mutant model is supplied.\n')
MOIETIES = ('base', 'ribose', 'phosphate')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def encode(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        stream.write(encode(value) + '\n')


def contact_row(site):
    """Use whole-residue counts and moieties; never gate on sidechain distance."""
    if site.get('interface_geometry_status') != 'computed':
        raise ValueError('Assay-matched RNA geometry unavailable')
    count, distance = site['partner_contact_count_5A'], site['min_partner_heavy_distance_A']
    if (type(count) is not int or count < 0 or isinstance(distance, bool)
            or not isinstance(distance, (int, float)) or not math.isfinite(distance) or distance < 0):
        raise ValueError('Invalid whole-residue RNA contact geometry')
    if (count > 0) != (distance <= 5):
        raise ValueError('Contact count and whole-residue distance disagree')
    result = dict(site_id=site['site_id'], direct_rna_contact=count > 0)
    if count and site.get('rna_moiety_status') == 'complete':
        counts = site['rna_contact_count_5A_by_moiety']
        if any(type(counts[name]) is not int or counts[name] < 0 for name in MOIETIES):
            raise ValueError('Invalid whole-residue moiety counts')
        names = [name for name in MOIETIES if counts[name] > 0]
        if names:
            result['rna_contact_moieties'] = names
    return result


def render_contact(base_text, rows):
    """New renderers use the corrected definition by construction."""
    if HEADER in base_text:
        raise ValueError('Base prompt already contains contact guidance')
    return base_text if not rows else base_text + '\n\n' + GUIDANCE + encode(rows)


def repair_contact_user(user, expected_rows):
    if not expected_rows:
        if HEADER in user:
            raise ValueError('Unexpected contact block for unavailable geometry')
        return user
    marker = '\n\n' + LEGACY_GUIDANCE
    if user.count(HEADER) != 1 or user.count(marker) != 1:
        raise ValueError('Unknown or ambiguous legacy contact guidance')
    base, body = user.split(marker)
    if json.loads(body) != expected_rows:
        raise ValueError('Saved contact rows disagree with whole-residue geometry')
    # Preserve every byte outside the definition, including the contact values.
    return base + '\n\n' + GUIDANCE + body


def prepare(output, source=SOURCE, expected_sources=None):
    if output.exists():
        raise FileExistsError('New protein-RNA input version required')
    expected_sources = load(AUDITED_SOURCES) if expected_sources is None else expected_sources
    sources = {}

    def verified(path, expected=None):
        path = Path(path)
        expected = expected if expected is not None else expected_sources.get(str(path))
        actual = sha(path)
        if expected is None or actual != expected:
            raise ValueError('Source hash mismatch or missing provenance: ' + str(path))
        sources[str(path)] = actual
        return load(path)

    config = verified(source / 'config.json')
    index = verified(source / 'index.json')
    baseline, geometry = Path(config['baseline']), Path(config['geometry'])
    queue, records, features = [], [], {}
    output.mkdir(parents=True)
    for unit in index:
        request = verified(source / unit['request_path'], unit['request_sha256'])
        task = verified(baseline / unit['task_path'], unit['task_sha256'])
        ids = unit['expected_ids']
        if [c['id'] for c in task['candidates'][unit['batch']*25:(unit['batch']+1)*25]] != ids:
            raise ValueError('Candidate order or chunk differs from source')
        if unit['condition'] not in {'structure_contact', 'msa_prompt_structure_contact'}:
            raise ValueError('Unsupported contact condition')
        if unit['eligible'] != (unit['reference_complex_id'] in config['eligible_complexes']):
            raise ValueError('Assay eligibility changed')
        group = unit['group_id']; rows = []
        if unit['eligible']:
            if group not in features:
                features[group] = verified(geometry / 'structure_features' / (group + '.json'))
                save(output / 'geometry' / (group + '.json'), features[group])
            by_id = {row['site_id']: row for row in features[group]['sites']}
            if len(by_id) != len(features[group]['sites']):
                raise ValueError('Duplicate geometry site')
            _, sites = base_sites(task)
            site_ids = list(dict.fromkeys(s['site_id'] for cid in ids for s in sites[cid]))
            rows = [contact_row(by_id[sid]) for sid in site_ids]
            if not rows:
                raise ValueError('Eligible assay has no contact sites')
        if len(request['contents']) != 1 or len(request['contents'][0]['parts']) != 1:
            raise ValueError('Unsupported contact request layout')
        old_user = request['contents'][0]['parts'][0]['text']
        user = repair_contact_user(old_user, rows)
        new_request = copy.deepcopy(request)
        new_request['contents'][0]['parts'][0]['text'] = user
        save(output / 'requests' / (unit['id'] + '.json'), new_request)
        if request.get('systemInstruction') is not None:
            raise ValueError('Unexpected native system instruction')
        new_unit = dict(domain='protein_rna', assay=group, arm=unit['condition'], chunk_index=unit['batch'],
                        template=dict(system='', user=user, expected_ids=ids))
        new_unit['logical_id'] = hashlib.sha256((encode(new_unit)+'\n').encode()).hexdigest()
        queue.append(new_unit)
        records.append(dict(assay=group, condition=unit['condition'], contact_sites=len(rows),
                            guidance_changed=user != old_user, source_request_sha256=unit['request_sha256']))
    if len({u['logical_id'] for u in queue}) != len(queue):
        raise ValueError('Duplicate request identity')
    if any(sha(path) != digest for path, digest in sources.items()):
        raise ValueError('Historical source changed during preparation')
    save(output / 'QUEUE.json', queue)
    manifest = dict(schema=VERSION, version=output.name, files={str(p.relative_to(output)):sha(p) for p in output.rglob('*.json')},
                    code_sha256=sha(__file__), site_identity_code_sha256=sha(ROOT/'code/evagent/offline_ablation.py'),
                    source_hashes=sources, records=records, queue_units=len(queue),
                    guidance_changed_requests=sum(r['guidance_changed'] for r in records),
                    contact_values_unchanged=True, candidate_resampling=False, model_calls=0, labels_read=False,
                    fully_ready=False, reason='Contact guidance repaired; historical two contact arms retained. Shared four-arm renderer/evaluator integration remains pending.')
    save(output / 'MANIFEST.json', manifest)
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--version', required=True)
    args = parser.parse_args()
    if not args.version.replace('_', '').replace('-', '').isalnum():
        raise ValueError('Simple version name required')
    result = prepare(HERE/'inputs'/args.version)
    print(json.dumps({k:result[k] for k in ['queue_units','guidance_changed_requests','contact_values_unchanged','model_calls','fully_ready']}))
