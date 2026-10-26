"""Prepare DNA-chain prompt additions for the frozen DNA22 N=10 cohort.

Protein-chain sequences already in the prompt stay identical. DNA reference
sequences fill only previously empty nucleic-acid chain fields. The three
assays without frozen DNA partners get no DNA text and remain in the queue.
No evaluation labels, no new candidates, no overwrite of historical runs.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import copy
import csv
import json
import os
import re
import shutil
import socket
import stat
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import protein_dna  # noqa: E402
from evagent.offline_ablation import digest, file_hash, render_evidence  # noqa: E402

ROOT = Path('/data/huangyanjie/EVAgent')
SOURCE = ROOT / 'experiments/protein_dna22_n10_cr25_ablation_gemini38_medium_20261008'
DEFAULT_OUT = ROOT / 'experiments/protein_dna22_dna_input_gemini38_medium_seed42_cr25_20261014_v1'
CREDENTIAL = ROOT / 'private_config/evagent/credentials.env'
PROXY = 'http://127.0.0.1:7777'
MISSING_PDBS = frozenset(('1CSP', '1KXL', '2JTM'))
ARMS = ('sequence', 'msa_prompt', 'structure', 'msa_prompt_structure')
COPIED = ('task.json', 'observed_input.json', 'cache_origin.json', 'base_prompt.json',
          'base_scores.json', 'features.json', 'site_mapping.json', 'partner_definition.json')


def read(path):
    return json.loads(Path(path).read_text())


def write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    os.replace(temporary, path)


def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(',', ':'), ensure_ascii=False)


def empty_pat(chain):
    # Only the empty `chain=` field; protein residues and mutation tokens stay untouched.
    return re.compile(r'(?<![A-Za-z0-9_])' + re.escape(chain) + r'=(?=(?: \| |\\?"|\n|$))')


def port_open(port):
    try:
        sock = socket.socket()
        sock.settimeout(1.0)
        try:
            return sock.connect_ex(('127.0.0.1', port)) == 0
        finally:
            sock.close()
    except (OSError, PermissionError):
        return False


def dna_input(assay):
    partner = read(SOURCE / 'assays' / assay['assay_id'] / 'partner_definition.json')
    chains = list(partner['groups'][1])
    pdb = assay['reference_complex_id']
    if pdb in MISSING_PDBS or not chains:
        if pdb not in MISSING_PDBS or chains:
            raise ValueError('Missing-DNA identity drifted: ' + assay['assay_id'])
        return dict(status='missing_retained', chains={}, source=None, source_sha256=None,
                    reference_complex_id=pdb)
    evidence = partner['evidence']
    path = Path(evidence['source'])
    if file_hash(path) != evidence['source_sha256']:
        raise ValueError('Prepared DNA source checksum mismatch')
    task = read(path)
    by_chain = {item['chain']: item for item in task['reference_chains']}
    seqs = {}
    for chain in chains:
        entry = by_chain[chain]
        sequence = entry['reference_sequence']
        if entry['polymer_type'] != 'nucleic_acid' or not sequence:
            raise ValueError('Missing or incorrectly typed DNA reference')
        if len(entry['numbering']) != len(sequence):
            raise ValueError('DNA sequence/numbering mismatch')
        seqs[chain] = sequence
    return dict(status='DNA_added', chains=seqs, source=str(path), source_sha256=file_hash(path),
                representation='unchanged upstream reference_sequence',
                direction=task['binding_partners'].get('nucleic_acid_direction'),
                reference_complex_id=pdb)


def patch_text(original, dna):
    text = original
    patches = []
    for chain, sequence in dna.items():
        pattern = empty_pat(chain)
        old, new = chain + '=', chain + '=' + sequence
        text, count = pattern.subn(lambda _: new, text)
        if count != 12:
            raise ValueError(f'Unexpected number of empty DNA fields: {chain}: {count}')
        patches.append(dict(kind='chain_sequence', old=old, new=new, count=count))
    restored = text
    for item in reversed(patches):
        if restored.count(item['new']) != item['count']:
            raise ValueError('DNA patch is not uniquely reversible')
        restored = restored.replace(item['new'], item['old'])
    if restored != original:
        raise ValueError('Unexpected change beyond DNA addition')
    return text, patches


def fill_nucleic_metadata(info, dna):
    if not dna:
        return
    expected = {chain: '' for chain in dna}
    if info.get('prompt_nucleic_sequences') != expected:
        raise ValueError('Unexpected structured DNA metadata')
    if info.get('prompt_nucleic_sequences_blank') is not True:
        raise ValueError('Nucleic prompt fields were not blank')
    info['prompt_nucleic_sequences'] = dict(dna)
    info['prompt_nucleic_sequences_blank'] = False


def rebuild_requests(eid, base_request, features):
    task = read(SOURCE / 'assays' / eid / 'task.json')
    ids = [candidate['id'] for candidate in task['candidates']]
    if ids != ['M001', 'M002', 'M003', 'M004', 'M005', 'M006', 'M007', 'M008', 'M009', 'M010']:
        raise ValueError('Candidate identity/order changed: ' + eid)
    prompts = {'sequence': (None, base_request['user'])}
    for condition in protein_dna.CONDITIONS:
        evidence = protein_dna.shown_evidence(render_evidence(task, ids, features, condition))
        prompts[condition] = (evidence, protein_dna.prompt_user(base_request, evidence))
    rebuilt = []
    for condition in ARMS:
        evidence, user = prompts[condition]
        request = dict(assay_id=eid, condition=condition, chunk_index=0, CR=25, expected_ids=ids,
                       system=base_request['system'], user=user)
        if evidence is not None:
            request['evidence'] = evidence
        request['request_hash'] = digest(request)
        rebuilt.append(request)
    return rebuilt


def tighten_credential(path):
    mode = path.stat().st_mode
    if mode & 0o077:
        os.chmod(path, stat.S_IMODE(mode) & 0o700)
    protein_dna.credential_value(path)


def prepare(out):
    out = Path(out)
    if out.exists():
        raise ValueError('Output already exists; refusing to overwrite')
    old_plan = read(SOURCE / 'plan.json')
    if old_plan['schema'] != 'protein-dna-n10-cr25-ablation-gemini38-medium-v1':
        raise ValueError('Source is not the frozen Gemini 3.8 medium DNA22 run')
    if len(old_plan['assays']) != 22 or old_plan['CR'] != 25 or old_plan['N'] != 10:
        raise ValueError('Source is not the 22-assay N=10 CR=25 DNA freeze')
    if old_plan['sample_seed'] != 42 or old_plan['generation_seed'] != 42:
        raise ValueError('Source seed is not 42')
    if file_hash(Path(old_plan['original_dataset'])) != old_plan['original_dataset_sha256']:
        raise ValueError('N10 source dataset changed')
    tighten_credential(CREDENTIAL)
    out.mkdir(parents=True)
    rows, audit, queue, source_hashes = [], [], [], {
        str(SOURCE / 'plan.json'): file_hash(SOURCE / 'plan.json'),
        str(SOURCE / 'queue.json'): file_hash(SOURCE / 'queue.json'),
    }
    for assay in old_plan['assays']:
        eid, pdb = assay['assay_id'], assay['reference_complex_id']
        addition = dna_input(assay)
        folder = out / 'assays' / eid
        folder.mkdir(parents=True)
        for name in COPIED:
            src = SOURCE / 'assays' / eid / name
            shutil.copyfile(src, folder / name)
            source_hashes[str(src)] = file_hash(src)
        base_prompt = read(folder / 'base_prompt.json')
        original_user = base_prompt['user']
        patched_user, patches = patch_text(original_user, addition['chains'])
        base_prompt['user'] = patched_user
        write(folder / 'base_prompt.json', base_prompt)
        partner = read(folder / 'partner_definition.json')
        fill_nucleic_metadata(partner['evidence'], addition['chains'])
        write(folder / 'partner_definition.json', partner)
        features = read(folder / 'features.json')
        fill_nucleic_metadata(features['structure_definition']['partner_definition'], addition['chains'])
        write(folder / 'features.json', features)
        write(folder / 'DNA_INPUT.json', addition)
        if addition['source']:
            source_hashes[addition['source']] = addition['source_sha256']
        rebuilt = rebuild_requests(eid, base_prompt, features)
        for request in rebuilt:
            key = request['request_hash']
            write(out / 'requests' / f'{key}.json', request)
            prompt = out / 'prompts' / request['condition'] / f'{eid}.txt'
            prompt.parent.mkdir(parents=True, exist_ok=True)
            prompt.write_text(request['user'])
            queue.append(dict(request_hash=key, assay_id=eid, condition=request['condition']))
            audit.append(dict(assay_id=eid, condition=request['condition'], new_request_hash=key,
                              patches=patches, DNA_status=addition['status'],
                              exact_original_restored_after_removing_DNA=True))
        rows.append(dict(assay_id=eid, assay_group_id=assay['assay_group_id'],
                         reference_complex_id=pdb, retained_in_inventory=True,
                         preparation_status=addition['status'], execution_status='queued',
                         DNA_chain_count=len(addition['chains']), N=10, CR=25))
        print(json.dumps(dict(event='prepared', assay=eid, pdb=pdb, status=addition['status']),
                         ensure_ascii=False), flush=True)
    added = sum(row['preparation_status'] == 'DNA_added' for row in rows)
    missing = sum(row['preparation_status'] == 'missing_retained' for row in rows)
    if len(queue) != 88 or added != 19 or missing != 3:
        raise ValueError(f'Unexpected coverage: queue={len(queue)} added={added} missing={missing}')
    if {row['reference_complex_id'] for row in rows if row['preparation_status'] == 'missing_retained'} != MISSING_PDBS:
        raise ValueError('Missing-DNA PDB set drifted')
    write(out / 'COHORT.json', rows)
    columns = sorted(set().union(*(row.keys() for row in rows)))
    with (out / 'COHORT.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, columns)
        writer.writeheader()
        writer.writerows(rows)
    write(out / 'PATCH_AUDIT.json', audit)
    write(out / 'queue.json', queue)
    for source_path in protein_dna.source_files():
        target = out / 'code_snapshot' / source_path.relative_to(protein_dna.PROJECT)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source_path.read_bytes())
    model = copy.deepcopy(old_plan['model_config'])
    model['proxy'] = PROXY
    inputs = {str(path.relative_to(out)): file_hash(path)
              for path in sorted(out.rglob('*')) if path.is_file()}
    plan = dict(
        schema='protein-dna-n10-cr25-ablation-gemini38-medium-dna-input-v1',
        CR=25, N=10, sample_seed=42, generation_seed=42, model_config=model,
        historical_model_config=old_plan['historical_model_config'],
        sequence_source='provider', scoring_conditions=list(ARMS), conditions=list(ARMS),
        postprocessing_conditions=['msa_gemini_rank5050', 'msa_gemini_rank5050_structure'],
        weights=dict(gemini=0.5, msa=0.5),
        fusion_definition=old_plan['fusion_definition'],
        fusion_reference=old_plan['fusion_reference'],
        cohort_source=old_plan['cohort_source'],
        original_dataset=old_plan['original_dataset'],
        original_dataset_sha256=old_plan['original_dataset_sha256'],
        assays=old_plan['assays'], input_hashes=inputs,
        source_hashes={str(path): file_hash(path) for path in protein_dna.source_files()},
        frozen_source_hashes=source_hashes,
        planned_new_requests=len(queue), reused_scoring_requests=0,
        token_budget=20_000_000, workers=4, max_repairs=1,
        labels_read_by_predictor=False, originals_modified=False,
        evidence_placement='empty_assay_memory_slot_before_output_contract',
        source_experiment=str(SOURCE),
        scientific_change='DNA reference sequence addition only; protein, candidates, MSA and structure numbers unchanged',
        DNA_added_assays=added, missing_DNA_retained_assays=missing,
        model_note='gemini-3.8-flash thinking_level=medium; proxy 7777 replaces unavailable historical 7897',
        msa_prompt_note=old_plan.get('msa_prompt_note'),
        runtime_note='Full N=10 DNA22 inventory (22 assays / 88 requests), not the 13-assay reviewed subset',
        transport_7777_at_prepare=port_open(7777),
        transport_7897_at_prepare=port_open(7897),
    )
    plan['fingerprint'] = digest({key: value for key, value in plan.items() if key != 'fingerprint'})
    write(out / 'plan.json', plan)
    local_validate(out)
    print(json.dumps(dict(event='prepare_ok', assays=22, requests=88, DNA_added=added,
                          missing_retained=missing, output=str(out),
                          transport_7777=plan['transport_7777_at_prepare']), ensure_ascii=False), flush=True)


def local_validate(out):
    out = Path(out)
    plan = read(out / 'plan.json')
    if digest({key: value for key, value in plan.items() if key != 'fingerprint'}) != plan['fingerprint']:
        raise ValueError('Plan fingerprint mismatch')
    if plan['model_config']['proxy'] != PROXY or plan['model_config']['thinking_level'] != 'medium':
        raise ValueError('Scoring transport or thinking level drifted')
    if plan['CR'] != 25 or plan['N'] != 10 or plan['generation_seed'] != 42:
        raise ValueError('Scientific protocol drifted')
    queue = read(out / 'queue.json')
    if len(queue) != 88:
        raise ValueError('Expected 88 scoring requests')
    counts = dict(DNA_added_requests=0, missing_DNA_retained_requests=0)
    for item in read(out / 'PATCH_AUDIT.json'):
        addition = read(out / 'assays' / item['assay_id'] / 'DNA_INPUT.json')
        source_prompt = (SOURCE / 'assays' / item['assay_id'] / 'base_prompt.json')
        original = read(source_prompt)['user']
        patched, patches = patch_text(original, addition['chains'])
        if patches != item['patches']:
            raise ValueError('Patch audit drifted')
        new = read(out / 'requests' / f"{item['new_request_hash']}.json")
        if new['user'] != patched and item['condition'] == 'sequence':
            raise ValueError('Sequence prompt is not the DNA-patched original')
        if item['condition'] == 'sequence' and new['user'] != read(out / 'assays' / item['assay_id'] / 'base_prompt.json')['user']:
            raise ValueError('Sequence request/base_prompt split')
        if len(new['expected_ids']) != 10 or new['CR'] != 25 or new['system'] != read(source_prompt)['system']:
            raise ValueError('Original protocol or candidates changed')
        text = (out / 'prompts' / item['condition'] / f"{item['assay_id']}.txt").read_text()
        if text != new['user']:
            raise ValueError('Prompt file and request differ')
        for name in ('task.json', 'observed_input.json', 'site_mapping.json', 'base_scores.json'):
            if file_hash(out / 'assays' / item['assay_id'] / name) != file_hash(SOURCE / 'assays' / item['assay_id'] / name):
                raise ValueError('Frozen scientific evidence changed: ' + name)
        counts['DNA_added_requests' if addition['chains'] else 'missing_DNA_retained_requests'] += 1
    if counts['DNA_added_requests'] != 76 or counts['missing_DNA_retained_requests'] != 12:
        raise ValueError('Unexpected patched vs retained request counts')
    report = dict(status='PASS', **counts, total_inventory=22, prepared_assays=22,
                  DNA_added_assays=19, missing_DNA_retained_assays=3, model_calls=0,
                  labels_read=False, candidate_and_protein_inputs_unchanged=True,
                  MSA_and_structure_numbers_unchanged=True,
                  exact_prompt_reversal_passed=True, scientific_parameters_unchanged=True,
                  all_three_missing_DNA_assays_retained=True, historical_outputs_unchanged=True)
    write(out / 'DNA_INPUT_VALIDATION.json', report)
    print(json.dumps(report, ensure_ascii=False), flush=True)


def run(out, credential_path):
    """Score the prepared 88 requests. Do not re-parse WT PDBs.

    protein_dna.validate() rebuilds assays from deposited structures that are
    not present on this host. Frozen task/features/candidates were copied
    unchanged; request hashes were already checked by local_validate().
    """
    out = Path(out)
    tighten_credential(Path(credential_path))
    local_validate(out)
    plan = read(out / 'plan.json')
    queue = read(out / 'queue.json')
    if len(queue) != 88 or plan['planned_new_requests'] != 88:
        raise ValueError('Scoring queue is not the frozen 88-request DNA22 set')
    for unit in queue:
        request = read(out / 'requests' / f"{unit['request_hash']}.json")
        if digest({key: value for key, value in request.items() if key != 'request_hash'}) != unit['request_hash']:
            raise ValueError('Request identity changed: ' + unit['request_hash'])
        if request['CR'] != 25 or len(request['expected_ids']) != 10:
            raise ValueError('Chunk ranking freeze changed')
        if request['assay_id'] != unit['assay_id'] or request['condition'] != unit['condition']:
            raise ValueError('Queue/request identity split')
    os.environ[plan['model_config']['api_key_env']] = protein_dna.credential_value(Path(credential_path))
    gateway = protein_dna.ConcurrentGateway(
        model_config=plan['model_config'],
        run_dir=out / plan.get('provider_dirname', 'provider'),
        fingerprint=plan['fingerprint'], token_budget=plan['token_budget'],
        reserve_budget=True, max_concurrent_requests=plan['workers'])

    def one(unit):
        key = unit['request_hash']
        target = out / 'results' / f'{key}.json'
        if target.exists():
            return protein_dna.validated_result(out, unit, plan)
        request = read(out / 'requests' / f'{key}.json')

        class PromptBlocked(Exception):
            pass

        attempts = []

        def generate(prompt, attempt):
            response = gateway.generate(
                system=request['system'], user=prompt, run_name='protein_dna_cr25_ablation',
                metadata=dict(assay_id=request['assay_id'], condition=request['condition'],
                              chunk_index=0, attempt=attempt, request_hash=key))
            attempts.append(response)
            if (response.get('response_metadata') or {}).get('prompt_feedback', {}).get('block_reason'):
                raise PromptBlocked()
            if response.get('status') == 'ok' and not str(
                    (response.get('response_metadata') or {}).get('model_version', '')).startswith(
                    plan['model_config']['model']):
                raise ValueError('Actual provider model identity mismatch')
            return response

        try:
            result = protein_dna.run_self_check(
                original_prompt=request['user'], generate=generate,
                validator=protein_dna.ScoringResponseValidator(request['expected_ids']),
                policy=protein_dna.SelfCheckPolicy(max_repairs=plan['max_repairs']))
        except PromptBlocked:
            result = dict(status='provider_blocked', value=None, attempts=attempts)
        except Exception as error:
            result = dict(status='error', value=None, error_type=type(error).__name__,
                          error=str(error), attempts=attempts)
        result.update(request_hash=key, assay_id=request['assay_id'], condition=request['condition'],
                      expected_ids=request['expected_ids'], model_config=plan['model_config'],
                      fingerprint=plan['fingerprint'])
        if result['status'] == 'ok':
            try:
                protein_dna.validate_scores(result['value'], request['expected_ids'], bounded=True)
            except ValueError as error:
                result.update(status='invalid_score_range', value=None, error=str(error))
        protein_dna.save(target, result)
        if result['status'] == 'ok':
            protein_dna.validated_result(out, unit, plan)
        print(json.dumps(dict(event='settled', assay=request['assay_id'], condition=request['condition'],
                              status=result['status']), ensure_ascii=False), flush=True)
        return result

    try:
        first = one(queue[0])
        if first['status'] == 'ok':
            with concurrent.futures.ThreadPoolExecutor(max_workers=plan['workers']) as pool:
                list(pool.map(one, queue[1:]))
    finally:
        protein_dna.write_report(out / 'usage.json', gateway.summary())
        gateway.close()


def analyze(out):
    protein_dna.analyze(Path(out))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'validate', 'run', 'analyze'])
    parser.add_argument('--output', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--credential-file', type=Path, default=CREDENTIAL)
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare(args.output)
    elif args.command == 'validate':
        local_validate(args.output)
        protein_dna.validate(args.output)
    elif args.command == 'run':
        run(args.output, args.credential_file)
    else:
        analyze(args.output)
