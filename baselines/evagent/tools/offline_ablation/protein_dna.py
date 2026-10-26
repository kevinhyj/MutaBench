"""CR25 ablation for the frozen protein–DNA N=10 cohort.

Predictors never read evaluation labels. Candidate identity, order and the
original sequence prompt stay those of the DNA22 Gemini CR25 cache. Changed
prompts are separate requests. Original result directories are not modified.
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures
import csv
import hashlib
import json
import math
import os
import shutil
import sqlite3
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from evagent.offline_ablation import (base_sites, digest, file_hash, fuse_rankings, matching_fewshot_cache,
                                      midranks, render_evidence, validate_scores)
from evagent.integrated.chunked_scoring import ScoringResponseValidator
from evagent.integrated.research.concurrent_gateway import ConcurrentGateway
from evagent.integrated.self_check import SelfCheckPolicy, run_self_check
from features import DELTA_SASA_TEXT, FeatureBuilder, save

PROJECT = CODE.parent
HISTORY = PROJECT / 'work/final_mutation_gemini_cr25_seed42_20260928'
DEFAULT_OUT = PROJECT / 'experiments/protein_dna22_n10_cr25_ablation_20261005'
PREPARED = PROJECT / 'explanation_test/complex_sequence_all_assays_N250_N10_seed42_20260928/prepared'
ZEROSHOT = PROJECT / 'zeroshot_baseline_experiment/multichain/runs/uncovered133_esm2_20260928/tasks'
COHORT_CSV = HISTORY / 'nabe_spearman_by_type.csv'
CONDITIONS = ('msa_prompt', 'structure', 'msa_prompt_structure')


def load(path):
    return json.loads(Path(path).read_text())


def write_report(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    os.replace(temporary, path)


def source_files():
    return [Path(__file__), CODE / 'evagent/offline_ablation.py',
            Path(__file__).with_name('features.py'),
            PROJECT / 'baseline/singlechain/models.py',
            CODE / 'evagent/integrated/gateway.py',
            CODE / 'evagent/integrated/research/concurrent_gateway.py',
            CODE / 'evagent/integrated/chunked_scoring.py',
            CODE / 'evagent/integrated/self_check.py']


def credential_value(path):
    if path.stat().st_mode & 0o077:
        raise ValueError('Credential permissions are too broad')
    content = path.read_text().strip()
    entries = [line.strip().removeprefix('export ').split('=', 1)[1].strip()
               for line in content.splitlines()
               if line.strip().removeprefix('export ').startswith('GEMINI_API_KEY=')]
    value = entries[0].strip('"').strip("'") if len(entries) == 1 else content
    if len(entries) > 1 or not value or any(c.isspace() for c in value):
        raise ValueError('Credential unavailable or ambiguous')
    return value


EVIDENCE_SLOT = (
    'No frozen assay summary is supplied. Use the supplied assay metadata and sequences, '
    'together with any frozen additional evidence provided in this request. Respect the '
    'applicability and uncertainty stated for each source; absent evidence is not evidence of no effect:\n{}'
)
OUTPUT_CONTRACT = 'Output JSON only; do not include prose after it.'


def shown_evidence(evidence):
    """Keep the log ratio in the frozen table and rank fusion, not in the prompt."""
    if 'msa_sites' not in evidence:
        return evidence
    shown = json.loads(json.dumps(evidence))
    for rows in shown['msa_sites'].values():
        for row in rows:
            row.pop('log_probability_ratio', None)
            if row.get('status') != 'ok':
                for key in ('p_wt', 'p_mutant', 'weighted_valid_count',
                            'weighted_gap_fraction', 'weighted_unknown_fraction'):
                    row[key] = None
    return shown


def evidence_text(evidence):
    lines = ['Frozen additional evidence:',
             json.dumps(evidence, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False)]
    if 'msa_sites' in evidence:
        lines.append('MSA values are weighted empirical amino-acid frequencies with pseudocounts from the supplied alignment. '
                     'They describe column composition. They are not homology probabilities and not the binding measurement. '
                     'predicted_score remains an estimate of the binding measurement. '
                     'Frequencies are omitted where the column has no homolog support or global Neff is below 20.')
    if 'structure_sites' in evidence:
        lines.append('Structure distances are angstroms. SASA and delta SASA are square angstroms. '
                     'delta_sasa_partner_A2 is this residue only. '
                     'Partner chains are the frozen specified DNA group, not the geometrically nearest chains.')
    lines.append('JSON null means unavailable, never zero.')
    return '\n'.join(lines)


def prompt_user(base_request, evidence):
    user = base_request['user']
    if user.count(EVIDENCE_SLOT) != 1:
        raise ValueError('Base prompt lacks exactly one empty evidence slot')
    user = user.replace(EVIDENCE_SLOT, EVIDENCE_SLOT[:-2] + '\n' + evidence_text(evidence), 1)
    evidence_at = user.find('Frozen additional evidence:')
    output_at = user.find(OUTPUT_CONTRACT)
    if evidence_at < 0 or output_at < 0 or not evidence_at < output_at:
        raise ValueError('Evidence must occupy the assay slot before the output contract')
    if 'Frozen additional evidence:' in user[output_at:]:
        raise ValueError('Evidence follows the output contract')
    return user


def fusion_artifact(ids, scores, features, origin, *, condition):
    msa = features['independent_msa_scores']
    if features['full_msa_coverage']:
        rows = fuse_rankings(ids, scores, msa)
        status = 'complete'
    else:
        ranks = midranks([scores[i] for i in ids])
        rows = [dict(candidate_id=i, gemini_score=scores[i], msa_score=msa[i],
                     gemini_rank_percentile=(r - 1) / (len(ids) - 1),
                     msa_rank_percentile=None, fusion_value=None, final_rank=None)
                for i, r in zip(ids, ranks)]
        status = 'incomplete_MSA'
    for row in rows:
        row['cache_source'] = origin.get('source_response')
        row['cache_source_sha256'] = origin.get('source_response_sha256')
    return dict(condition=condition, status=status, rows=rows, new_calls=0,
                missing_candidate_ids=[i for i in ids if msa[i] is None],
                weights=dict(gemini=0.5, msa=0.5),
                definition='Ascending average midrank percentile (rank-1)/(N-1); equal 0.5 weights; final descending average midrank')


def parse_chain_sequences(text):
    parsed = {}
    for part in text.split(' | '):
        chain, sequence = part.split('=', 1)
        parsed[chain] = sequence
    return parsed


def parse_residue_token(token):
    if token[-1:].isalpha():
        return int(token[:-1]), token[-1]
    return int(token), ''


def dna_rows():
    with COHORT_CSV.open() as handle:
        rows = [row for row in csv.DictReader(handle) if row['nucleic_acid_type'] == 'DNA']
    if len(rows) != 22 or len({row['assay_group_id'] for row in rows}) != 22:
        raise ValueError('Frozen DNA22 cohort is not 22 distinct assays')
    return rows


def partner_specification(group_id):
    prepared = PREPARED / f'nabe__{group_id}.json'
    if prepared.exists():
        task = load(prepared)
        roles = {group['role']: group['chains'] for group in task['binding_partners']['groups']}
        if set(roles) != {'protein', 'DNA'}:
            raise ValueError('Prepared partner partition is not protein/DNA')
        numbering = {chain['chain']: [(item['reference_residue_number'], item.get('insertion_code') or '')
                                      for item in chain['numbering']]
                     for chain in task['reference_chains']}
        evidence = dict(definition='Existing base-task binding_partners.groups, unchanged',
                        source=str(prepared), source_sha256=file_hash(prepared),
                        binding_partners=task['binding_partners'])
        return roles['protein'], roles['DNA'], numbering, evidence
    zeroshot = ZEROSHOT / f'nabe__{group_id}.json'
    if zeroshot.exists():
        task = load(zeroshot)
        protein = [chain['chain'] for chain in task['wt_chains'] if chain.get('polymer_type') == 'protein']
        dna = [chain['chain'] for chain in task['wt_chains'] if chain.get('polymer_type') == 'nucleic_acid']
        numbering = {chain['chain']: [parse_residue_token(token) for token in chain['residue_ids']]
                     for chain in task['wt_chains']}
        evidence = dict(definition='Frozen zeroshot wt_chains polymer_type partition; nucleic_acid chains are the specified DNA partners for this DNA assay',
                        source=str(zeroshot), source_sha256=file_hash(zeroshot))
        return protein, dna, numbering, evidence
    return [], [], {}, dict(
        definition='No frozen protein/DNA chain partition is stored for this assay',
        source='DNA22 episode nucleic_acid_type=DNA; deposited structure has no frozen nucleic-acid chains',
        specified_partner_chains=[])


def observed_chains(model, chain_ids, numbering, *, protein):
    from Bio.SeqUtils import seq1
    records, missing = [], []
    for chain_id in chain_ids:
        allowed = numbering.get(chain_id)
        if chain_id not in model or not allowed:
            missing.append(chain_id)
            continue
        by_key = {}
        for residue in model[chain_id]:
            by_key.setdefault((residue.id[1], residue.id[2].strip()), residue)
        chosen = []
        for key in allowed:
            residue = by_key.get(key)
            if residue is None:
                missing.append(chain_id)
                chosen = None
                break
            chosen.append(residue)
        if chosen is None:
            continue
        record = dict(chain=chain_id, residue_ids=[str(pos) + ins for pos, ins in allowed])
        if protein:
            record['resolved_sequence'] = ''.join(seq1(residue.resname, custom_map={'MSE': 'M'}) for residue in chosen)
        else:
            record['resolved_sequence'] = None
        records.append(record)
    return records, missing


def extract_task(folder, facts):
    from Bio.PDB import PDBParser
    episode, manifest = load(folder / 'episode.json'), load(folder / 'input_manifest.json')
    description = json.loads(episode['assay_description'])
    if description.get('nucleic_acid_type') != 'DNA' or description['assay_group_id'] != manifest['assay_group_id']:
        raise ValueError('Episode is not the frozen DNA assay named by its manifest')
    path = PROJECT / manifest['structure_path']
    if file_hash(path) != manifest['structure_sha256']:
        raise ValueError('Structure migration checksum mismatch')
    selected = [facts[fid] for fid in manifest['sampled_fact_ids']]
    if len(selected) != 10 or len(episode['query']) != 10:
        raise ValueError('Original N10 freeze required')
    from Bio.SeqUtils import seq1
    protein_ids, dna_ids, numbering, evidence = partner_specification(manifest['assay_group_id'])
    if not protein_ids:
        protein_ids = sorted({str(item['chain']) for fact in selected for item in fact['chain_mutations']})
    model = next(PDBParser(QUIET=True).get_structure('WT', path).get_models())
    for chain_id in protein_ids:
        if numbering.get(chain_id):
            continue
        if chain_id not in model:
            continue
        numbering[chain_id] = [(residue.id[1], residue.id[2].strip()) for residue in model[chain_id]
                               if residue.id[0] == ' ' and seq1(residue.resname, custom_map={'MSE': 'M'}, undef_code='X') != 'X']
        evidence = dict(evidence, protein_numbering_fallback='PDB amino-acid residues because no frozen numbering was stored')
    protein_chains, missing_protein = observed_chains(model, protein_ids, numbering, protein=True)
    dna_chains, missing_dna = observed_chains(model, dna_ids, numbering, protein=False)
    if missing_protein:
        raise ValueError('Specified protein chains lack frozen coordinates: ' + ','.join(missing_protein))
    chain_map = {chain['chain']: chain for chain in protein_chains}
    prompt_chains = parse_chain_sequences(episode['wt_sequence'])
    for chain_id, chain in chain_map.items():
        if prompt_chains.get(chain_id) != chain['resolved_sequence']:
            raise ValueError('Base protein sequence differs from the cached prompt')
    evidence = dict(evidence)
    evidence['prompt_nucleic_sequences'] = {chain: prompt_chains.get(chain, '') for chain in dna_ids}
    evidence['prompt_nucleic_sequences_blank'] = all(not sequence for sequence in evidence['prompt_nucleic_sequences'].values())
    candidates = []
    for candidate, fact in zip(episode['query'], selected, strict=True):
        if fact['assay_group_id'] != manifest['assay_group_id'] or fact['reference_complex_id'] != manifest['reference_complex_id']:
            raise ValueError('Source fact belongs to a different assay')
        mutated = {chain_id: list(chain['resolved_sequence']) for chain_id, chain in chain_map.items()}
        sites = []
        for change in fact['chain_mutations']:
            chain_id, insertion = str(change['chain']), change.get('insertion_code') or ''
            coordinate = str(change['position']) + insertion
            index = chain_map[chain_id]['residue_ids'].index(coordinate)
            if mutated[chain_id][index] != change['wt_aa']:
                raise ValueError('Candidate WT identity mismatch')
            mutated[chain_id][index] = change['mutant_aa']
            sites.append(dict(chain=chain_id, position=change['position'], insertion_code=insertion,
                              wt=change['wt_aa'], mutant=change['mutant_aa'], sequence_index_1based=index + 1))
        observed = parse_chain_sequences(candidate['sequence'])
        for chain_id, residues in mutated.items():
            if observed.get(chain_id) != ''.join(residues):
                raise ValueError('Frozen candidate protein sequence reconstruction differs')
        if candidate['mutation'] != ';'.join(
                f"{site['chain']}:{site['wt']}{site['position']}{site['insertion_code']}{site['mutant']}" for site in sites):
            raise ValueError('Frozen mutation string differs from the N10 fact')
        candidates.append(dict(id=candidate['variant_id'], fact_id=fact['fact_id'], sites=sites))
    present = {chain['chain'] for chain in protein_chains + dna_chains}
    unavailable = None
    if missing_dna or not dna_ids:
        unavailable = ('Specified DNA partner coordinates are absent from the frozen structure'
                       if dna_ids else
                       'No frozen DNA partner chains are defined; structure measurements are unavailable')
        evidence['missing_specified_chains'] = missing_dna
    observed = dict(source='nabe', reference_complex_id=manifest['reference_complex_id'],
                    candidate_hash=digest(candidates), wt_chains=protein_chains + dna_chains,
                    structure_source=dict(path=str(path), sha256=file_hash(path)))
    partners = dict(candidate_hash=observed['candidate_hash'],
                    groups=[protein_ids, dna_ids], evidence=evidence,
                    excluded_extra_chains=sorted(set(model.child_dict) - present))
    task = dict(group_id=manifest['assay_group_id'], episode_id=episode['episode_id'], source='nabe',
                reference_complex_id=manifest['reference_complex_id'], modality='protein_DNA',
                candidates=candidates, structure_source=observed['structure_source'])
    return task, observed, partners, unavailable


def find_cache(folder, model):
    ledgers = [HISTORY / 'provider/request_ledger.sqlite',
               *[path / 'provider/request_ledger.sqlite' for path in sorted(HISTORY.glob('execution_v*'))]]
    failures = []
    for root in [*reversed(sorted(HISTORY.glob('execution_v*'))), HISTORY]:
        if not (root / 'plan.json').exists():
            continue
        plan = load(root / 'plan.json')
        for record in plan['requests']:
            if record['episode_id'] != folder.name or record['condition']['name'] != 'numeric_0':
                continue
            request_path = root / 'predictions' / record['request_hash'] / 'request.json'
            result_path = request_path.with_name('result.json')
            if not request_path.exists() or not result_path.exists():
                continue
            try:
                scores, origin = matching_fewshot_cache(root / 'plan.json', folder, request_path, result_path, ledgers, model)
                origin['rejected_cache_candidates'] = failures
                return scores, origin, load(request_path)
            except (ValueError, KeyError) as error:
                failures.append(dict(path=str(result_path), reason=str(error)))
    raise ValueError('No exactly matched Gemini sequence cache: ' + json.dumps(failures, ensure_ascii=False))


def prepare(out):
    if out.exists() and any(out.iterdir()):
        raise ValueError('Preparation requires a new empty output directory')
    model = load(HISTORY / 'execution_v6/plan.json')['model_config']
    source = PROJECT / 'data/final_mutation_data/nabe/canonical/nabe_facts_N10.jsonl'
    facts = {row['fact_id']: row for row in map(json.loads, source.read_text().splitlines())}
    builder = FeatureBuilder(PROJECT, out)
    queue, assays = [], []
    for row in dna_rows():
        folder = HISTORY / 'episodes' / row['assay_id']
        manifest = load(folder / 'input_manifest.json')
        if file_hash(source) != manifest['source_sha256']:
            raise ValueError('N10 source dataset changed')
        task, observed, partners, unavailable = extract_task(folder, facts)
        eid = row['assay_id']
        target = out / 'assays' / eid
        _, task['base_sites'] = base_sites(task)
        save(target / 'task.json', task)
        save(target / 'observed_input.json', {key: value for key, value in observed.items() if key != 'wt_chains'} | dict(
            wt_chains=observed['wt_chains']))
        scores, origin, base_request = find_cache(folder, model)
        save(target / 'cache_origin.json', origin)
        save(target / 'base_prompt.json', base_request)
        save(target / 'base_scores.json', scores)
        features, mapping = builder.build(task, observed, partners, unavailable_structure_reason=unavailable)
        save(target / 'features.json', features)
        save(target / 'site_mapping.json', mapping)
        save(target / 'partner_definition.json', partners)
        ids = [candidate['id'] for candidate in task['candidates']]
        save(target / 'msa_gemini_rank5050.json', fusion_artifact(
            ids, scores, features, origin, condition='postprocessing_only'))
        for condition in CONDITIONS:
            evidence = shown_evidence(render_evidence(task, ids, features, condition))
            user = prompt_user(base_request, evidence)
            request = dict(assay_id=eid, condition=condition, chunk_index=0, CR=25, expected_ids=ids,
                           system=base_request['system'], user=user, evidence=evidence)
            request['request_hash'] = digest(request)
            save(out / 'requests' / f"{request['request_hash']}.json", request)
            prompt_path = out / 'prompts' / condition / f'{eid}.txt'
            prompt_path.parent.mkdir(parents=True, exist_ok=True)
            prompt_path.write_text(user)
            queue.append(dict(request_hash=request['request_hash'], assay_id=eid, condition=condition))
        assays.append(dict(assay_id=eid, assay_group_id=row['assay_group_id'],
                           reference_complex_id=manifest['reference_complex_id'], N=10, status='prepared',
                           historical_sequence_spearman=float(row['spearman']),
                           msa_complete=features['full_msa_coverage'],
                           msa_errors=features['source_provenance']['profile_errors'],
                           structure_unavailable=unavailable,
                           partner_definition=partners['evidence']['definition'],
                           cache_source=origin['source_response']))
        print(json.dumps(dict(event='prepared', assay=eid, msa_complete=features['full_msa_coverage'],
                              structure_unavailable=bool(unavailable)), ensure_ascii=False), flush=True)
    if len(assays) != 22 or len(queue) != 66:
        raise ValueError('DNA22 prepare did not cover 22 assays and 66 changed-prompt blocks')
    save(out / 'queue.json', queue)
    for source_path in source_files():
        target = out / 'code_snapshot' / source_path.relative_to(PROJECT)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source_path.read_bytes())
    inputs = {str(path.relative_to(out)): file_hash(path) for path in sorted(out.rglob('*')) if path.is_file()}
    plan = dict(schema='protein-dna-n10-cr25-ablation-v1', CR=25, N=10, sample_seed=42, generation_seed=42,
                model_config=model, conditions=list(CONDITIONS),
                postprocessing_conditions=['msa_gemini_rank5050', 'msa_gemini_rank5050_structure'],
                weights=dict(gemini=0.5, msa=0.5),
                fusion_definition='Per-assay ascending average midrank percentile (rank-1)/(N-1); larger is better; final descending average midrank; missing component rejects complete fusion',
                fusion_reference='code/evagent/offline_ablation.py:fuse_rankings',
                cohort_source=str(COHORT_CSV), original_dataset=str(source),
                original_dataset_sha256=file_hash(source), assays=assays, input_hashes=inputs,
                source_hashes={str(path): file_hash(path) for path in source_files()},
                planned_new_requests=len(queue), reused_scoring_requests=len(assays),
                token_budget=5_000_000, workers=4, max_repairs=1,
                labels_read_by_predictor=False, originals_modified=False)
    plan['fingerprint'] = digest(plan)
    save(out / 'plan.json', plan)
    validate(out)


def reprompt(out):
    """Discard changed-prompt scores whose evidence followed the output contract."""
    previous = load(out / 'plan.json')
    if len(previous['assays']) != 22 or previous['CR'] != 25 or previous['N'] != 10:
        raise ValueError('Refusing to rebuild a directory that is not the DNA22 CR25 plan')
    built = []
    for assay in previous['assays']:
        target = out / 'assays' / assay['assay_id']
        feature_path = target / 'features.json'
        feature_text = feature_path.read_text()
        old_delta = 'SASA of the complete own partner group minus SASA in the specified two-group complex'
        if feature_text.count(old_delta) != 1:
            raise ValueError('Delta SASA definition is not the misplaced whole-group sentence')
        replaced = feature_text.replace(old_delta, DELTA_SASA_TEXT, 1)
        features = json.loads(replaced)
        task = load(target / 'task.json')
        base_request = load(target / 'base_prompt.json')
        ids = [candidate['id'] for candidate in task['candidates']]
        users = {}
        for condition in CONDITIONS:
            evidence = shown_evidence(render_evidence(task, ids, features, condition))
            users[condition] = (evidence, prompt_user(base_request, evidence))
        built.append((assay, replaced, base_request, ids, users))
    removed = []
    for name in ('prompts', 'requests', 'results', 'provider'):
        path = out / name
        if path.exists():
            shutil.rmtree(path)
            removed.append(name)
    for name in ('usage.json', 'per_assay.csv', 'rankings_and_components.csv', 'summary.json',
                 'REPORT.txt', 'validation.json', 'queue.json'):
        path = out / name
        if path.exists():
            path.unlink()
            removed.append(name)
    queue = []
    for assay, replaced, base_request, ids, users in built:
        eid = assay['assay_id']
        target = out / 'assays' / eid
        (target / 'features.json').write_text(replaced)
        for extra in ('metrics.json', 'msa_gemini_rank5050_structure.json'):
            path = target / extra
            if path.exists():
                path.unlink()
                removed.append(str(path.relative_to(out)))
        ranking_dir = target / 'rankings'
        if ranking_dir.exists():
            shutil.rmtree(ranking_dir)
            removed.append(str(ranking_dir.relative_to(out)))
        for condition in CONDITIONS:
            evidence, user = users[condition]
            request = dict(assay_id=eid, condition=condition, chunk_index=0, CR=25, expected_ids=ids,
                           system=base_request['system'], user=user, evidence=evidence)
            request['request_hash'] = digest(request)
            save(out / 'requests' / f"{request['request_hash']}.json", request)
            prompt_path = out / 'prompts' / condition / f'{eid}.txt'
            prompt_path.parent.mkdir(parents=True, exist_ok=True)
            prompt_path.write_text(user)
            queue.append(dict(request_hash=request['request_hash'], assay_id=eid, condition=condition))
    if len(queue) != 66:
        raise ValueError('Corrected prompt rebuild did not produce 66 requests')
    save(out / 'queue.json', queue)
    for source_path in source_files():
        target = out / 'code_snapshot' / source_path.relative_to(PROJECT)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source_path.read_bytes())
    inputs = {str(path.relative_to(out)): file_hash(path) for path in sorted(out.rglob('*'))
              if path.is_file() and path != out / 'plan.json'}
    source = Path(previous['original_dataset'])
    plan = dict(schema='protein-dna-n10-cr25-ablation-v1', CR=25, N=10, sample_seed=42, generation_seed=42,
                model_config=previous['model_config'], conditions=list(CONDITIONS),
                postprocessing_conditions=['msa_gemini_rank5050', 'msa_gemini_rank5050_structure'],
                weights=dict(gemini=0.5, msa=0.5),
                fusion_definition=previous['fusion_definition'], fusion_reference=previous['fusion_reference'],
                cohort_source=previous['cohort_source'], original_dataset=str(source),
                original_dataset_sha256=file_hash(source), assays=previous['assays'], input_hashes=inputs,
                source_hashes={str(path): file_hash(path) for path in source_files()},
                planned_new_requests=len(queue), reused_scoring_requests=len(previous['assays']),
                token_budget=previous['token_budget'], workers=previous['workers'],
                max_repairs=previous['max_repairs'], labels_read_by_predictor=False, originals_modified=False,
                evidence_placement='empty_assay_memory_slot_before_output_contract')
    plan['fingerprint'] = digest(plan)
    write_report(out / 'plan.json', plan)
    validate(out)
    print(json.dumps(dict(event='reprompt_ok', removed=removed, units=66), ensure_ascii=False), flush=True)


def drop_log_ratio(out):
    """Rewrite MSA prompts without the log ratio and rescore only those conditions."""
    previous = load(out / 'plan.json')
    if len(previous['assays']) != 22 or previous['CR'] != 25:
        raise ValueError('Refusing to rewrite a directory that is not the DNA22 CR25 plan')
    queue = load(out / 'queue.json')
    old_by = {(unit['assay_id'], unit['condition']): unit['request_hash'] for unit in queue}
    rebuilt = []
    removed = []
    for assay in previous['assays']:
        eid = assay['assay_id']
        target = out / 'assays' / eid
        task = load(target / 'task.json')
        features = load(target / 'features.json')
        base_request = load(target / 'base_prompt.json')
        ids = [candidate['id'] for candidate in task['candidates']]
        for condition in ('msa_prompt', 'msa_prompt_structure'):
            evidence = shown_evidence(render_evidence(task, ids, features, condition))
            if 'log_probability_ratio' in json.dumps(evidence):
                raise ValueError('Log ratio remained in the rewritten prompt')
            user = prompt_user(base_request, evidence)
            request = dict(assay_id=eid, condition=condition, chunk_index=0, CR=25, expected_ids=ids,
                           system=base_request['system'], user=user, evidence=evidence)
            request['request_hash'] = digest(request)
            old_hash = old_by[(eid, condition)]
            if old_hash != request['request_hash']:
                for relative in (f'requests/{old_hash}.json', f'results/{old_hash}.json',
                                 f'prompts/{condition}/{eid}.txt'):
                    path = out / relative
                    if path.exists():
                        path.unlink()
                        removed.append(relative)
            rebuilt.append(request)
    if len(rebuilt) != 44:
        raise ValueError('Expected 44 MSA prompt rewrites')
    for request in rebuilt:
        save(out / 'requests' / f"{request['request_hash']}.json", request)
        prompt_path = out / 'prompts' / request['condition'] / f"{request['assay_id']}.txt"
        prompt_path.parent.mkdir(parents=True, exist_ok=True)
        prompt_path.write_text(request['user'])
    new_queue = []
    rebuilt_by = {(item['assay_id'], item['condition']): item['request_hash'] for item in rebuilt}
    for unit in queue:
        key = (unit['assay_id'], unit['condition'])
        new_queue.append(dict(unit, request_hash=rebuilt_by.get(key, unit['request_hash'])))
    write_report(out / 'queue.json', new_queue)
    for name in ('per_assay.csv', 'rankings_and_components.csv', 'summary.json', 'REPORT.txt', 'validation.json'):
        path = out / name
        if path.exists():
            path.unlink()
    for assay in previous['assays']:
        target = out / 'assays' / assay['assay_id']
        for extra in ('metrics.json',):
            path = target / extra
            if path.exists():
                path.unlink()
        ranking_dir = target / 'rankings'
        if ranking_dir.exists():
            for path in ranking_dir.glob('msa_prompt*.json'):
                path.unlink()
    for source_path in source_files():
        target = out / 'code_snapshot' / source_path.relative_to(PROJECT)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source_path.read_bytes())
    inputs = dict(previous['input_hashes'])
    for relative in removed:
        inputs.pop(relative, None)
    for relative in ('validation.json', 'per_assay.csv', 'rankings_and_components.csv', 'summary.json', 'REPORT.txt'):
        inputs.pop(relative, None)
    for path in sorted(out.rglob('*')):
        if not path.is_file() or path == out / 'plan.json':
            continue
        relative = str(path.relative_to(out))
        if relative.startswith(('requests/', 'prompts/', 'code_snapshot/', 'queue.json')):
            inputs[relative] = file_hash(path)
    plan = dict(previous)
    plan.pop('fingerprint', None)
    plan['input_hashes'] = inputs
    plan['source_hashes'] = {str(path): file_hash(path) for path in source_files()}
    plan['accepted_result_fingerprints'] = [previous['fingerprint']]
    plan['provider_dirname'] = 'provider_msa_nolpr'
    plan['msa_prompt_note'] = (
        'Prompt omits log_probability_ratio. Frequencies are null unless MSA status is ok. '
        'Independent MSA scores and rank fusion still use the log ratio.'
    )
    plan['fingerprint'] = digest(plan)
    write_report(out / 'plan.json', plan)
    validate(out)
    print(json.dumps(dict(event='drop_log_ratio_ok', rewritten=44, removed=len(removed)), ensure_ascii=False), flush=True)


def prepare_rescored(out, source, *, model, schema, model_note, token_budget=8_000_000, event='prepared_rescored'):
    """Copy frozen DNA22 features and rescore sequence plus the three prompt arms."""
    source = Path(source) if source else DEFAULT_OUT
    if out.exists() and any(out.iterdir()):
        raise ValueError('Preparation requires a new empty output directory')
    if out.resolve() == source.resolve():
        raise ValueError('Refusing to overwrite the source experiment')
    previous = load(source / 'plan.json')
    if previous['model_config']['model'] != 'gemini-3.8-flash' or previous['model_config']['thinking_level'] != 'medium':
        raise ValueError('Source is not the frozen 3.8 medium DNA ablation')
    if len(previous['assays']) != 22:
        raise ValueError('Source is not the 22-assay DNA ablation')
    scoring = ('sequence', *CONDITIONS)
    queue = []
    for assay in previous['assays']:
        eid = assay['assay_id']
        origin = source / 'assays' / eid
        target = out / 'assays' / eid
        target.mkdir(parents=True)
        for name in ('task.json', 'observed_input.json', 'cache_origin.json', 'base_prompt.json',
                     'base_scores.json', 'features.json', 'site_mapping.json', 'partner_definition.json'):
            shutil.copyfile(origin / name, target / name)
        task = load(target / 'task.json')
        base_request = load(target / 'base_prompt.json')
        features = load(target / 'features.json')
        ids = [candidate['id'] for candidate in task['candidates']]
        prompts = {'sequence': (None, base_request['user'])}
        for condition in CONDITIONS:
            evidence = shown_evidence(render_evidence(task, ids, features, condition))
            prompts[condition] = (evidence, prompt_user(base_request, evidence))
        for condition in scoring:
            evidence, user = prompts[condition]
            request = dict(assay_id=eid, condition=condition, chunk_index=0, CR=25, expected_ids=ids,
                           system=base_request['system'], user=user)
            if evidence is not None:
                request['evidence'] = evidence
            request['request_hash'] = digest(request)
            save(out / 'requests' / f"{request['request_hash']}.json", request)
            prompt_path = out / 'prompts' / condition / f'{eid}.txt'
            prompt_path.parent.mkdir(parents=True, exist_ok=True)
            prompt_path.write_text(user)
            queue.append(dict(request_hash=request['request_hash'], assay_id=eid, condition=condition))
        print(json.dumps(dict(event=event, assay=eid), ensure_ascii=False), flush=True)
    if len(queue) != 88:
        raise ValueError('Rescored prepare did not produce 88 scoring blocks')
    save(out / 'queue.json', queue)
    for source_path in source_files():
        target = out / 'code_snapshot' / source_path.relative_to(PROJECT)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source_path.read_bytes())
    inputs = {str(path.relative_to(out)): file_hash(path) for path in sorted(out.rglob('*')) if path.is_file()}
    plan = dict(schema=schema, CR=25, N=10, sample_seed=42, generation_seed=42, model_config=model,
                historical_model_config=previous['model_config'], sequence_source='provider',
                scoring_conditions=list(scoring), conditions=list(scoring),
                postprocessing_conditions=['msa_gemini_rank5050', 'msa_gemini_rank5050_structure'],
                weights=dict(gemini=0.5, msa=0.5),
                fusion_definition=previous['fusion_definition'], fusion_reference=previous['fusion_reference'],
                cohort_source=previous['cohort_source'], original_dataset=previous['original_dataset'],
                original_dataset_sha256=previous['original_dataset_sha256'], assays=previous['assays'],
                input_hashes=inputs, source_hashes={str(path): file_hash(path) for path in source_files()},
                planned_new_requests=len(queue), reused_scoring_requests=0, token_budget=token_budget,
                workers=4, max_repairs=1, labels_read_by_predictor=False, originals_modified=False,
                evidence_placement='empty_assay_memory_slot_before_output_contract',
                source_experiment=str(source), model_note=model_note,
                msa_prompt_note=previous.get('msa_prompt_note'))
    plan['fingerprint'] = digest(plan)
    save(out / 'plan.json', plan)
    validate(out)


def prepare_pro(out, source=None):
    """Rescore the DNA22 ablation with Gemini 3.1 Pro. The 3.8 medium run stays untouched."""
    source = Path(source) if source else DEFAULT_OUT
    previous = load(source / 'plan.json')
    model = dict(previous['model_config'])
    model['model'] = 'gemini-3.1-pro-preview'
    prepare_rescored(
        out, source, model=model, schema='protein-dna-n10-cr25-ablation-gemini31pro-v1',
        model_note='API id gemini-3.1-pro-preview; no non-preview gemini-3.1-pro is listed. Thinking stays medium.',
        event='prepared_pro')


def prepare_thinking(out, source, level):
    """Rescore the DNA22 ablation with Gemini 3.8 Flash at one thinking level."""
    if level not in ('low', 'medium', 'high'):
        raise ValueError('thinking_level must be low, medium, or high')
    source = Path(source) if source else DEFAULT_OUT
    previous = load(source / 'plan.json')
    model = dict(previous['model_config'])
    model['model'] = 'gemini-3.8-flash'
    model['thinking_level'] = level
    prepare_rescored(
        out, source, model=model,
        schema=f'protein-dna-n10-cr25-ablation-gemini38-{level}-v1',
        model_note=f'gemini-3.8-flash thinking_level={level}; sequence is rescored at this level.',
        token_budget=20_000_000, event=f'prepared_thinking_{level}')


def validate(out):
    plan = load(out / 'plan.json')
    if digest({key: value for key, value in plan.items() if key != 'fingerprint'}) != plan['fingerprint']:
        raise ValueError('Plan changed')
    for path, expected in plan['source_hashes'].items():
        # Scoring used the frozen snapshot. Later analyze-only fixes stay outside that hash.
        snapshot = out / 'code_snapshot' / Path(path).relative_to(PROJECT)
        if not snapshot.is_file() or file_hash(snapshot) != expected:
            raise ValueError('Frozen code snapshot changed: ' + path)
    for path, expected in plan['input_hashes'].items():
        if path == 'plan.json':
            continue
        if file_hash(out / path) != expected:
            raise ValueError('Frozen input changed: ' + path)
    scoring = tuple(plan.get('scoring_conditions', CONDITIONS))
    if (plan['CR'] != 25 or plan['N'] != 10 or plan['generation_seed'] != 42 or
            plan['weights'] != dict(gemini=0.5, msa=0.5) or tuple(plan['conditions']) != scoring):
        raise ValueError('Frozen scientific protocol differs')
    source = PROJECT / 'data/final_mutation_data/nabe/canonical/nabe_facts_N10.jsonl'
    if file_hash(source) != plan['original_dataset_sha256']:
        raise ValueError('Original dataset changed')
    facts = {row['fact_id']: row for row in map(json.loads, source.read_text().splitlines())}
    if len(plan['assays']) != 22:
        raise ValueError('Expected 22 DNA assays')
    for assay in plan['assays']:
        eid = assay['assay_id']
        target = out / 'assays' / eid
        task, observed, partners, unavailable = extract_task(HISTORY / 'episodes' / eid, facts)
        _, task['base_sites'] = base_sites(task)
        if task != load(target / 'task.json'):
            raise ValueError('Original assay/candidate/site identity changed')
        origin = load(target / 'cache_origin.json')
        historical_model = load(Path(origin['plan']))['model_config']
        cache_model = plan.get('historical_model_config', plan['model_config'])
        if ({key: value for key, value in cache_model.items() if key != 'proxy'}
                != {key: value for key, value in historical_model.items() if key != 'proxy'}):
            raise ValueError('Historical generation configuration mismatch')
        scores, fresh = matching_fewshot_cache(
            Path(origin['plan']), HISTORY / 'episodes' / eid, Path(origin['source_request']),
            Path(origin['source_response']),
            list(dict.fromkeys(Path(item['ledger']) for item in origin['provider_attempts'])),
            historical_model)
        if scores != load(target / 'base_scores.json'):
            raise ValueError('Historical scoring cache changed')
        if fresh['source_response_sha256'] != origin['source_response_sha256']:
            raise ValueError('Historical cache provenance changed')
        features = load(target / 'features.json')
        ids = [candidate['id'] for candidate in task['candidates']]
        if plan.get('sequence_source') != 'provider' and fusion_artifact(
                ids, scores, features, origin, condition='postprocessing_only') != load(target / 'msa_gemini_rank5050.json'):
            raise ValueError('Fusion components differ on replay')
        for candidate_id, sites in features['msa'].items():
            for row in sites.values():
                if row['status'] == 'no_valid_homolog_at_column' and any(
                        row[key] is not None for key in ('p_wt', 'p_mutant', 'log_probability_ratio',
                                                         'weighted_valid_count', 'weighted_gap_fraction',
                                                         'weighted_unknown_fraction')):
                    raise ValueError('Absent MSA column was imputed')
        for rows in features['structure'].values():
            for row in rows.values():
                if row['sasa_status'] != 'ok' and (row['sasa_bound_A2'] is not None or row['delta_sasa_partner_A2'] is not None):
                    raise ValueError('Unavailable SASA was imputed')
                if row['partner_status'] != 'ok' and (
                        row['min_partner_heavy_distance_A'] is not None or row['partner_contact_count_5A'] not in (None,)):
                    raise ValueError('Unavailable partner geometry was imputed')
        if unavailable and features['structure_definition'].get('unavailable_reason') != unavailable:
            raise ValueError('Missing structure was not explicitly recorded')
        for metadata in features['source_provenance']['msa_chains'].values():
            table = load(Path(metadata['probability_table']))
            if any('probabilities_20' not in column or len(column['probabilities_20']) != 20 for column in table['rows']):
                raise ValueError('Frozen 20-amino-acid table is incomplete')
    queue = load(out / 'queue.json')
    if len(queue) != plan['planned_new_requests']:
        raise ValueError('Scoring queue does not match the plan')
    for unit in queue:
        request = load(out / 'requests' / f"{unit['request_hash']}.json")
        if digest({key: value for key, value in request.items() if key != 'request_hash'}) != unit['request_hash']:
            raise ValueError('Request identity changed')
        target = out / 'assays' / unit['assay_id']
        task, features = load(target / 'task.json'), load(target / 'features.json')
        base_user = load(target / 'base_prompt.json')['user']
        if unit['condition'] == 'sequence':
            if request['user'] != base_user or 'evidence' in request:
                raise ValueError('Rescored sequence prompt is not the original sequence prompt')
        else:
            expected = shown_evidence(render_evidence(task, request['expected_ids'], features, unit['condition']))
            text = json.dumps(expected, allow_nan=False)
            if 'log_probability_ratio' in text or any(
                    token in text for token in ('probabilities_20', 'msa_column', 'homology probability')):
                raise ValueError('Audit-only or mislabeled MSA fields entered the prompt')
            chains = expected.get('msa_chains', [])
            if len(chains) != len({item['chain'] for item in chains}):
                raise ValueError('MSA chain summary is duplicated')
            if request['evidence'] != expected or request['user'] != prompt_user(load(target / 'base_prompt.json'), expected):
                raise ValueError('Prompt replay differs')
        if request['CR'] != 25 or len(request['expected_ids']) != 10:
            raise ValueError('Chunk ranking freeze changed')
    write_report(out / 'validation.json', dict(status='ok', assays=22, new_prompt_units=len(queue),
                                             reused_sequence_caches=0 if plan.get('sequence_source') == 'provider' else 22,
                                             labels_read=False, CR=25, N=10, model=plan['model_config']['model']))
    print(json.dumps(dict(event='validation_ok', assays=22, units=len(queue), model=plan['model_config']['model'])), flush=True)
    return plan, queue


def validated_result(out, unit, plan):
    request = load(out / 'requests' / f"{unit['request_hash']}.json")
    result = load(out / 'results' / f"{unit['request_hash']}.json")
    for key in ('request_hash', 'assay_id', 'condition', 'expected_ids'):
        if result.get(key) != request[key]:
            raise ValueError('Changed-prompt result identity mismatch')
    accepted = {plan['fingerprint'], *plan.get('accepted_result_fingerprints', [])}
    if result.get('model_config') != plan['model_config'] or result.get('fingerprint') not in accepted:
        raise ValueError('Changed-prompt generation configuration mismatch')
    if result['status'] != 'ok':
        return result
    validate_scores(result['value'], request['expected_ids'], bounded=True)
    attempts = result.get('attempts') or []
    if not attempts:
        raise ValueError('Scoring result has no provider attempts')
    ledgers = [out / name / 'request_ledger.sqlite'
               for name in dict.fromkeys(('provider', plan.get('provider_dirname'))) if name]
    for index, attempt in enumerate(attempts):
        row = None
        for ledger in ledgers:
            if not ledger.exists():
                continue
            with sqlite3.connect(ledger.resolve().as_uri() + '?mode=ro', uri=True) as probe:
                row = probe.execute('SELECT request_json,response_json,status FROM attempts WHERE request_sha256=?',
                                    (attempt['request_sha256'],)).fetchone()
            if row:
                break
        if not row or row[2] != 'ok':
            raise ValueError('Successful scoring provider ledger missing')
        wire, raw = map(json.loads, row[:2])
        if (digest(wire) != attempt['request_sha256'] or wire['model_config'] != plan['model_config'] or
                wire['fingerprint'] != result['fingerprint'] or
                wire['messages'][0] != dict(role='system', content=request['system']) or
                wire['metadata'].get('request_hash') != unit['request_hash']):
            raise ValueError('Actual changed-prompt provider request differs')
        if index == 0 and wire['messages'][1] != dict(role='user', content=request['user']):
            raise ValueError('Actual initial prompt mismatch')
        if hashlib.sha256(wire['messages'][1]['content'].encode()).hexdigest() != attempt['attempt_prompt_sha256']:
            raise ValueError('Actual attempt prompt checksum mismatch')
        if raw.get('final_text') != attempt.get('final_text'):
            raise ValueError('Scoring raw response differs from the ledger')
    parsed = ScoringResponseValidator(request['expected_ids']).validate(attempts[-1]['final_text'])
    if not parsed.valid or parsed.value != result['value']:
        raise ValueError('Scores differ from the validated raw provider response')
    return result


def run(out, credential_path, *, pilot=False):
    plan, queue = validate(out)
    os.environ[plan['model_config']['api_key_env']] = credential_value(credential_path)
    gateway = ConcurrentGateway(model_config=plan['model_config'],
                                run_dir=out / plan.get('provider_dirname', 'provider'),
                                fingerprint=plan['fingerprint'], token_budget=plan['token_budget'],
                                reserve_budget=True, max_concurrent_requests=plan['workers'])

    def one(unit):
        key = unit['request_hash']
        target = out / 'results' / f'{key}.json'
        if target.exists():
            return validated_result(out, unit, plan)
        request = load(out / 'requests' / f'{key}.json')
        class PromptBlocked(Exception):
            pass
        attempts = []

        def generate(prompt, attempt):
            response = gateway.generate(system=request['system'], user=prompt, run_name='protein_dna_cr25_ablation',
                                        metadata=dict(assay_id=request['assay_id'], condition=request['condition'],
                                                      chunk_index=0, attempt=attempt, request_hash=key))
            attempts.append(response)
            if (response.get('response_metadata') or {}).get('prompt_feedback', {}).get('block_reason'):
                raise PromptBlocked()
            if response.get('status') == 'ok' and not str((response.get('response_metadata') or {}).get('model_version', '')).startswith(plan['model_config']['model']):
                raise ValueError('Actual provider model identity mismatch')
            return response
        try:
            result = run_self_check(original_prompt=request['user'], generate=generate,
                                    validator=ScoringResponseValidator(request['expected_ids']),
                                    policy=SelfCheckPolicy(max_repairs=plan['max_repairs']))
        except PromptBlocked:
            result = dict(status='provider_blocked', value=None, attempts=attempts)
        except Exception as error:
            result = dict(status='error', value=None, error_type=type(error).__name__, error=str(error), attempts=attempts)
        result.update(request_hash=key, assay_id=request['assay_id'], condition=request['condition'],
                      expected_ids=request['expected_ids'], model_config=plan['model_config'],
                      fingerprint=plan['fingerprint'])
        if result['status'] == 'ok':
            try:
                validate_scores(result['value'], request['expected_ids'], bounded=True)
            except ValueError as error:
                result.update(status='invalid_score_range', value=None, error=str(error))
        save(target, result)
        if result['status'] == 'ok':
            validated_result(out, unit, plan)
        print(json.dumps(dict(event='settled', assay=request['assay_id'], condition=request['condition'],
                              status=result['status']), ensure_ascii=False), flush=True)
        return result
    try:
        first = one(queue[0])
        if first['status'] == 'ok' and not pilot:
            with concurrent.futures.ThreadPoolExecutor(max_workers=plan['workers']) as pool:
                list(pool.map(one, queue[1:]))
    finally:
        write_report(out / 'usage.json', gateway.summary())
        gateway.close()


def correlation(scores, labels, ids):
    from scipy.stats import kendalltau, spearmanr
    values = [scores[i] for i in ids]
    gold = [labels[i] for i in ids]
    defined = len(set(values)) > 1 and len(set(gold)) > 1
    if not defined:
        return None, None, 'constant_undefined'
    return float(spearmanr(values, gold).statistic), float(kendalltau(values, gold).statistic), 'complete'


def analyze(out):
    plan, queue = validate(out)
    rows, components = [], []
    for assay in plan['assays']:
        eid = assay['assay_id']
        target = out / 'assays' / eid
        task = load(target / 'task.json')
        ids = [candidate['id'] for candidate in task['candidates']]
        features = load(target / 'features.json')
        origin = load(target / 'cache_origin.json')
        rescored = plan.get('sequence_source') == 'provider'
        predictions = {}
        if rescored:
            sequence_unit = next(unit for unit in queue if unit['assay_id'] == eid and unit['condition'] == 'sequence')
            sequence_path = out / 'results' / f"{sequence_unit['request_hash']}.json"
            sequence_result = validated_result(out, sequence_unit, plan) if sequence_path.exists() else dict(status='not_run')
            row_sequence_status = sequence_result['status']
            if sequence_result['status'] == 'ok':
                predictions['sequence_cached'] = sequence_result['value']
                fusion = fusion_artifact(ids, sequence_result['value'], features, origin, condition='rescored_sequence')
            else:
                fusion = dict(status='not_run', rows=[])
            save(target / 'msa_gemini_rank5050.json', fusion)
        else:
            row_sequence_status = 'cached'
            predictions['sequence_cached'] = load(target / 'base_scores.json')
            fusion = load(target / 'msa_gemini_rank5050.json')
        row = dict(assay_id=eid, assay_group_id=assay['assay_group_id'],
                   reference_complex_id=assay['reference_complex_id'], N=10, CR=25,
                   historical_sequence_spearman=assay['historical_sequence_spearman'],
                   msa_complete=assay['msa_complete'], structure_unavailable=assay['structure_unavailable'],
                   sequence_status=row_sequence_status,
                   msa_gemini_rank5050_status=fusion['status'], partner_definition=assay['partner_definition'])
        if fusion['status'] == 'complete':
            predictions['msa_gemini_rank5050'] = {item['candidate_id']: item['fusion_value'] for item in fusion['rows']}
        for item in fusion['rows']:
            components.append(dict(assay_id=eid, condition='msa_gemini_rank5050', **item))
        for unit in queue:
            if unit['assay_id'] != eid:
                continue
            path = out / 'results' / f"{unit['request_hash']}.json"
            result = validated_result(out, unit, plan) if path.exists() else dict(status='not_run')
            row[unit['condition'] + '_status'] = result['status']
            if result['status'] == 'ok':
                predictions[unit['condition']] = result['value']
        if 'structure' in predictions:
            structure_path = out / 'results' / (next(
                unit['request_hash'] for unit in queue
                if unit['assay_id'] == eid and unit['condition'] == 'structure') + '.json')
            structure_origin = dict(source_response=str(structure_path),
                                    source_response_sha256=file_hash(structure_path))
            structure_fusion = fusion_artifact(ids, predictions['structure'], features, structure_origin,
                                               condition='postprocessing_of_new_structure_scores')
            save(target / 'msa_gemini_rank5050_structure.json', structure_fusion)
            row['msa_gemini_rank5050_structure_status'] = structure_fusion['status']
            if structure_fusion['status'] == 'complete':
                predictions['msa_gemini_rank5050_structure'] = {
                    item['candidate_id']: item['fusion_value'] for item in structure_fusion['rows']}
            for item in structure_fusion['rows']:
                components.append(dict(assay_id=eid, condition='msa_gemini_rank5050_structure', **item))
        else:
            row['msa_gemini_rank5050_structure_status'] = 'not_run'
        labels = load(HISTORY / 'episodes' / eid / 'evaluation_labels.json')
        if set(labels['labels']) != set(ids):
            raise ValueError('Evaluation labels do not match the frozen candidates')
        metric = dict(assay_id=eid, N=10, CR=25, label_source=str(HISTORY / 'episodes' / eid / 'evaluation_labels.json'),
                      methods={}, cache_source=origin['source_response'])
        methods = ('sequence_cached', 'msa_gemini_rank5050', *CONDITIONS, 'msa_gemini_rank5050_structure')
        for method in methods:
            if method not in predictions:
                continue
            rho, tau, status = correlation(predictions[method], labels['labels'], ids)
            metric['methods'][method] = dict(spearman=rho, kendall=tau, status=status)
            row[method + '_spearman'] = rho
            ranking = dict(method=method, rows=[dict(candidate_id=i, score=predictions[method][i],
                                                     final_rank=rank)
                                               for i, rank in zip(ids, midranks([predictions[method][i] for i in ids], descending=True))])
            (target / 'rankings').mkdir(exist_ok=True)
            (target / 'rankings' / f'{method}.json').write_text(json.dumps(ranking, indent=2, allow_nan=False) + '\n')
        if rescored:
            cached_rho, _, cached_status = correlation(load(target / 'base_scores.json'), labels['labels'], ids)
            if cached_status != 'complete' or abs(cached_rho - row['historical_sequence_spearman']) > 1e-10:
                raise ValueError('Historical 3.8 sequence cache no longer matches')
        elif row.get('sequence_cached_spearman') is None or abs(row['sequence_cached_spearman'] - row['historical_sequence_spearman']) > 1e-10:
            raise ValueError('Cached baseline metric differs from the historical same-set result')
        (target / 'metrics.json').write_text(json.dumps(metric, indent=2, allow_nan=False) + '\n')
        required = tuple(plan.get('scoring_conditions', CONDITIONS))
        row['status'] = 'complete' if all(row.get(condition + '_status') == 'ok' for condition in required) else 'incomplete_scoring'
        rows.append(row)
    fields = sorted(set().union(*(row.keys() for row in rows)))
    with (out / 'per_assay.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        writer.writerows(rows)
    if components:
        with (out / 'rankings_and_components.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(components[0]))
            writer.writeheader()
            writer.writerows(components)
    summary = dict(requested_assays=22, N=10, CR=25,
                   reused_sequence_caches=0 if plan.get('sequence_source') == 'provider' else 22,
                   planned_new_requests=plan['planned_new_requests'], model=plan['model_config']['model'],
                   methods={}, paired_comparisons={})
    for method in ('sequence_cached', 'msa_gemini_rank5050', *CONDITIONS, 'msa_gemini_rank5050_structure'):
        values = [row[method + '_spearman'] for row in rows if row.get(method + '_spearman') is not None]
        summary['methods'][method] = dict(complete_defined_assays=len(values),
                                          macro_spearman=sum(values) / len(values) if values else None)
    usage_path = out / 'usage.json'
    summary['usage'] = load(usage_path) if usage_path.exists() else dict(provider_attempts=0)
    summary['new_provider_calls'] = summary['usage'].get('provider_attempts', 0)
    scored = tuple(plan.get('scoring_conditions', CONDITIONS))
    summary['changed_prompt_units_completed'] = sum(row.get(condition + '_status') == 'ok' for row in rows for condition in scored)
    summary['unfinished'] = [dict(assay_id=row['assay_id'], condition=condition, status=row.get(condition + '_status', 'not_run'))
                             for row in rows for condition in scored if row.get(condition + '_status') != 'ok']
    summary['unfinished'] += [dict(assay_id=row['assay_id'], condition='msa_gemini_rank5050', reason='missing_MSA_component')
                              for row in rows if row['msa_gemini_rank5050_status'] != 'complete']
    (out / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    lines = [
        'Protein–DNA 离线消融（22 个 NaBE assay，N=10，chunk ranking CR=25）',
        '',
        f"复用精确匹配的原序列评分 {summary['reused_sequence_caches']} 个；新增 provider 调用 {summary['new_provider_calls']} 次。",
        f"评分单元完成 {summary['changed_prompt_units_completed']}/{summary['planned_new_requests']}。模型 {summary['model']}。",
        '融合使用仓库 fuse_rankings：两路越大越好，升序平均并列名次转为 (rank-1)/(N-1)，固定各 0.5，最终降序平均名次，1 最好。',
        '不混合原始分数与名次。任一 MSA 分量缺失时该 assay 的完整融合不可用，保留已有分量，不补零、不对子集重排。',
        'MSA 与 Structure 证据写入原 prompt 的空 assay memory 槽，位于候选和输出指令之前。',
        'MSA 概率是加权氨基酸频率统计，不是同源概率。全 20 种概率只在冻结概率表。',
        'Structure 伙伴沿用既有 binding_partners 或冻结 polymer_type 分组；不可用值为 null。',
        '',
    ]
    for method, item in summary['methods'].items():
        value = item['macro_spearman']
        lines.append(f"{method}: {item['complete_defined_assays']}/22 defined; macro Spearman={value if value is None else round(value, 6)}")
    lines += ['', '未完成项目：']
    lines += [json.dumps(item, ensure_ascii=False) for item in summary['unfinished']] or ['无']
    (out / 'REPORT.txt').write_text('\n'.join(lines) + '\n')
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'prepare-pro', 'prepare-thinking', 'reprompt', 'drop_log_ratio', 'validate', 'pilot', 'run', 'analyze'])
    parser.add_argument('--output', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--source', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--thinking-level', choices=['low', 'medium', 'high'])
    parser.add_argument('--credential-file', type=Path)
    args = parser.parse_args()
    if args.command in ('pilot', 'run'):
        if args.credential_file is None:
            raise SystemExit('pilot/run requires --credential-file')
        run(args.output, args.credential_file, pilot=args.command == 'pilot')
    elif args.command == 'prepare-pro':
        prepare_pro(args.output, args.source)
    elif args.command == 'prepare-thinking':
        if args.thinking_level is None:
            raise SystemExit('prepare-thinking requires --thinking-level')
        prepare_thinking(args.output, args.source, args.thinking_level)
    else:
        globals()[args.command](args.output)
