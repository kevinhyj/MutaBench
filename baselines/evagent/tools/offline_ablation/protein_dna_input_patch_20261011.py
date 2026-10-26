"""Prepare and audit DNA-only prompt additions for the reviewed DNA cohort.

No model calls, evaluation labels, or new candidates are used by this tool.
The full 22-assay inventory is retained; executable inputs are limited to the
13 previously reviewed non-sensitive assays, pending user scope confirmation.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path
import re
import shutil

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / 'experiments/protein_dna22_n10_cr25_ablation_gemini38_medium_20261008'
REVIEW = ROOT / 'reports/complex_smallpool_biosafe_inventory_20261009/REVIEW.json'
DEFAULT_OUT = ROOT / 'experiments/protein_dna22_dna_input_gemini38_medium_seed42_cr25_20261011_v1'
ARMS = ('sequence', 'msa_prompt', 'structure', 'msa_prompt_structure')
SAFE_PDBS = frozenset(('2VWJ', '4IQR', '4CH1', '1CSP', '1KXL', '4QGU', '2JTM',
                      '1EFA', '3LWI', '3LWH', '4A15', '1J5N', '1XSN'))
RESTRICTED = {
    '1RTD': 'HIV-1 replication-related reverse-transcriptase variant scoring',
    '2HMI': 'HIV-1 replication-related reverse-transcriptase variant scoring',
    '1TN9': 'Tn916 integrase; antibiotic-resistance dissemination-related variant scoring',
    '2A0I': 'TraI conjugative DNA-transfer-related variant scoring',
}


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(',', ':'), ensure_ascii=False)


def digest(obj):
    return hashlib.sha256(canonical(obj).encode()).hexdigest()


def write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def dna_input(assay):
    partner_path = SOURCE / 'assays' / assay['assay_id'] / 'partner_definition.json'
    partner = read(partner_path)
    chains = partner['groups'][1]
    if not chains:
        return dict(status='missing_retained', chains={}, source=None, source_sha256=None)
    evidence = partner['evidence']
    path = Path(evidence['source'])
    if sha(path) != evidence['source_sha256']:
        raise ValueError('Prepared DNA source checksum mismatch')
    task = read(path)
    by_chain = {c['chain']: c for c in task['reference_chains']}
    seqs = {}
    for chain in chains:
        entry = by_chain[chain]
        sequence = entry['reference_sequence']
        if entry['polymer_type'] != 'nucleic_acid' or not sequence:
            raise ValueError('Missing or incorrectly typed DNA reference')
        if len(entry['numbering']) != len(sequence):
            raise ValueError('DNA sequence/numbering mismatch')
        seqs[chain] = sequence
    return dict(status='DNA_added', chains=seqs, source=str(path),
                source_sha256=sha(path), representation='unchanged upstream reference_sequence',
                direction=task['binding_partners'].get('nucleic_acid_direction'))


def patch_text(original, dna):
    text = original
    patches = []
    for chain, sequence in dna.items():
        # Only empty chain sequence fields; proteins and candidate mutations stay identical.
        pattern = re.compile(r'(?<![A-Za-z0-9_])' + re.escape(chain) +
                             r'=(?=(?: \| |\\?"|\n|$))')
        old, new = chain + '=', chain + '=' + sequence
        text, count = pattern.subn(lambda _: new, text)
        if count != 12:  # assay metadata, WT line, and ten candidate sequences
            raise ValueError(f'Unexpected number of empty DNA fields: {chain}: {count}')
        patches.append(dict(kind='chain_sequence', old=old, new=new, count=count))
    if dna and '"prompt_nucleic_sequences":' in text:
        old = '"prompt_nucleic_sequences":' + canonical({c: '' for c in dna})
        new = '"prompt_nucleic_sequences":' + canonical(dna)
        if text.count(old) != 1 or text.count('"prompt_nucleic_sequences_blank":true') != 1:
            raise ValueError('Unexpected structure sequence metadata')
        for a, b in [(old, new), ('"prompt_nucleic_sequences_blank":true',
                                '"prompt_nucleic_sequences_blank":false')]:
            text = text.replace(a, b)
            patches.append(dict(kind='sequence_availability_metadata', old=a, new=b, count=1))
    restored = text
    for p in reversed(patches):
        if restored.count(p['new']) != p['count']:
            raise ValueError('DNA patch is not uniquely reversible')
        restored = restored.replace(p['new'], p['old'])
    if restored != original:
        raise ValueError('Unexpected change beyond DNA addition')
    return text, patches


def patch_request(old, dna):
    request = copy.deepcopy(old)
    request.pop('request_hash')
    request['user'], patches = patch_text(old['user'], dna)
    if dna and 'structure_definition' in request.get('evidence', {}):
        info = request['evidence']['structure_definition']['partner_definition']
        if info['prompt_nucleic_sequences'] != {c: '' for c in dna}:
            raise ValueError('Unexpected structured DNA metadata')
        info['prompt_nucleic_sequences'] = dna
        info['prompt_nucleic_sequences_blank'] = False
    request['request_hash'] = digest(request)
    return request, patches


def prepare(out):
    if out.exists():
        raise ValueError('Output already exists; refusing to overwrite')
    old_plan = read(SOURCE / 'plan.json')
    review = read(REVIEW)
    allowed_groups = {r[1] for r in review['allowed'] if r[0] == 'nabe' and r[2] in SAFE_PDBS}
    if len(old_plan['assays']) != 22:
        raise ValueError('Expected unchanged DNA22 inventory')
    for rel, expected in old_plan['input_hashes'].items():
        if sha(SOURCE / rel) != expected:
            raise ValueError('Frozen source input changed: ' + rel)
    for filename, expected in old_plan['source_hashes'].items():
        if sha(filename) != expected:
            raise ValueError('Historical scoring implementation changed: ' + filename)
    out.mkdir(parents=True)
    old_queue = read(SOURCE / 'queue.json')
    rows, audit, queue = [], [], []
    source_hashes = {str(SOURCE / 'plan.json'): sha(SOURCE / 'plan.json'), str(REVIEW): sha(REVIEW)}
    for assay in old_plan['assays']:
        eid, pdb = assay['assay_id'], assay['reference_complex_id']
        row = dict(assay_id=eid, assay_group_id=assay['assay_group_id'], reference_complex_id=pdb,
                   retained_in_inventory=True, execution_status='not_submitted', N=10, CR=25)
        if pdb not in SAFE_PDBS or assay['assay_group_id'] not in allowed_groups:
            row.update(preparation_status='not_prepared',
                       scope_status='restricted' if pdb in RESTRICTED else 'outside_reviewed_execution_scope',
                       reason=RESTRICTED.get(pdb, 'Not included in the 13-assay reviewed subset; no new inputs prepared'))
            rows.append(row)
            continue
        addition = dna_input(assay)
        row.update(preparation_status=addition['status'], scope_status='awaiting_user_scope_confirmation',
                   DNA_chain_count=len(addition['chains']))
        folder = out / 'assays' / eid
        folder.mkdir(parents=True)
        for name in ['task.json', 'features.json', 'observed_input.json', 'site_mapping.json',
                     'partner_definition.json']:
            src = SOURCE / 'assays' / eid / name
            shutil.copyfile(src, folder / name)
            source_hashes[str(src)] = sha(src)
        write(folder / 'DNA_INPUT.json', addition)
        if addition['source']:
            source_hashes[addition['source']] = addition['source_sha256']
        for unit in old_queue:
            if unit['assay_id'] != eid:
                continue
            src = SOURCE / 'requests' / (unit['request_hash'] + '.json')
            old = read(src)
            if digest({k: v for k, v in old.items() if k != 'request_hash'}) != old['request_hash']:
                raise ValueError('Source request hash mismatch')
            new, patches = patch_request(old, addition['chains'])
            key = new['request_hash']
            write(out / 'requests' / (key + '.json'), new)
            prompt = out / 'prompts' / unit['condition'] / (eid + '.txt')
            prompt.parent.mkdir(parents=True, exist_ok=True)
            prompt.write_text(new['user'])
            queue.append(dict(assay_id=eid, condition=unit['condition'], request_hash=key))
            audit.append(dict(assay_id=eid, condition=unit['condition'], source_request=str(src),
                              source_sha256=sha(src), new_request_hash=key, patches=patches,
                              exact_original_restored_after_removing_DNA=True))
            source_hashes[str(src)] = sha(src)
        rows.append(row)
    if len(queue) != 52 or sum(r['preparation_status'] == 'missing_retained' for r in rows) != 3:
        raise ValueError('Unexpected reviewed cohort coverage')
    write(out / 'COHORT.json', rows)
    columns = sorted(set().union(*(r.keys() for r in rows)))
    with (out / 'COHORT.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, columns); w.writeheader(); w.writerows(rows)
    write(out / 'PATCH_AUDIT.json', audit)
    write(out / 'queue.json', queue)
    plan = {k: copy.deepcopy(old_plan[k]) for k in ['CR', 'N', 'sample_seed', 'generation_seed',
            'model_config', 'scoring_conditions', 'postprocessing_conditions', 'weights',
            'fusion_definition', 'workers', 'max_repairs']}
    plan.update(schema='protein-dna-dna-addition-review-v1', requested_assays=22,
                prepared_assays=13, prepared_scoring_requests=52, DNA_added_assays=10,
                missing_DNA_retained_assays=3, model_calls=0,
                execution_status='awaiting_user_scope_confirmation',
                source_experiment=str(SOURCE), scientific_change='DNA reference sequence addition only',
                historical_features_unchanged=True, labels_read=False,
                source_hashes=source_hashes, scoring_code_hashes=old_plan['source_hashes'],
                runtime_note='Historical proxy 7897 is unavailable; configured current transport uses 7777. No model submitted.')
    write(out / 'PLAN.json', plan)
    validate(out)


def validate(out):
    plan = read(out / 'PLAN.json')
    for p, expected in plan['source_hashes'].items():
        if sha(p) != expected:
            raise ValueError('Source changed: ' + p)
    for p, expected in plan['scoring_code_hashes'].items():
        if sha(p) != expected:
            raise ValueError('Scoring code changed: ' + p)
    counts = dict(DNA_added_requests=0, missing_DNA_retained_requests=0)
    for item in read(out / 'PATCH_AUDIT.json'):
        old = read(item['source_request'])
        addition = read(out / 'assays' / item['assay_id'] / 'DNA_INPUT.json')
        expected, patches = patch_request(old, addition['chains'])
        new = read(out / 'requests' / (item['new_request_hash'] + '.json'))
        if new != expected or patches != item['patches']:
            raise ValueError('Prepared input differs from DNA-only transformation')
        if len(new['expected_ids']) != 10 or new['CR'] != 25 or new['system'] != old['system']:
            raise ValueError('Original protocol or candidates changed')
        for name in ['task.json', 'features.json', 'observed_input.json', 'site_mapping.json']:
            if sha(out / 'assays' / item['assay_id'] / name) != sha(SOURCE / 'assays' / item['assay_id'] / name):
                raise ValueError('Frozen scientific evidence changed')
        text = (out / 'prompts' / item['condition'] / (item['assay_id'] + '.txt')).read_text()
        if text != new['user']:
            raise ValueError('Prompt and request differ')
        counts['DNA_added_requests' if addition['chains'] else 'missing_DNA_retained_requests'] += 1
    report = dict(status='PASS', **counts, total_inventory=22, prepared_assays=13,
                  restricted_or_unprepared_assays=9, model_calls=0, labels_read=False,
                  candidate_and_protein_inputs_unchanged=True, MSA_and_structure_numbers_unchanged=True,
                  exact_prompt_reversal_passed=True, scientific_parameters_unchanged=True,
                  all_three_missing_DNA_assays_retained=True, historical_outputs_unchanged=True)
    write(out / 'VALIDATION.json', report)
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare', 'validate'])
    parser.add_argument('--output', type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    (prepare if args.action == 'prepare' else validate)(args.output)
