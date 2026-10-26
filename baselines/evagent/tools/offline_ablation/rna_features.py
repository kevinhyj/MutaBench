"""Label-free protein–RNA MSA profiles with unbiased deterministic capping.

Only the existing aligned homolog pool is read. Candidates are unchanged.
Low Neff is retained as an explicit quality flag, alongside the legacy gate.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

from evagent.offline_ablation import digest, file_hash
from features import AA, FeatureBuilder, save

POLICY = dict(version='protein-rna-msa-sha256-cap-v2', cap=2048, seed=42,
              selection='Unique non-WT valid projected homologs ordered by SHA256(seed, sequence), first 2048',
              weighting='Inverse >=80% identity neighbors with >=50% pairwise column overlap',
              pseudocount_total=1.0, low_neff='quality flag, not missing evidence',
              legacy_strict_neff_gate=20, primary_neff_gate=0,
              candidate_selection_changed=False, evaluation_labels_used=False)


def selected_alignment(path: Path, sequence: str, *, cap=2048):
    raw = []
    for line in path.read_text().splitlines():
        if line.startswith('>'):
            raw.append('')
        elif line.strip():
            if not raw:
                raise ValueError('MSA must be FASTA')
            raw[-1] += line.strip()
    if not raw or ''.join(c.upper() for c in raw[0] if c.isalpha()) != sequence:
        raise ValueError('MSA query differs from the observed frozen reference')
    aligned = [''.join(c for c in row if c != '.' and not c.islower()) for row in raw]
    if len({len(row) for row in aligned}) != 1:
        raise ValueError('MSA alignment lengths differ')
    query = aligned[0]
    columns = [index for index, letter in enumerate(query) if letter != '-']
    positions, position = [], 0
    for letter in raw[0]:
        if letter.isalpha():
            if letter.isupper():
                positions.append(position)
            position += 1
    if ''.join(sequence[index] for index in positions) != query.replace('-', ''):
        raise ValueError('MSA query/observed coordinate mapping differs')
    unique = set(''.join(row[index] for index in columns) for row in aligned[1:])
    unique.discard(query.replace('-', ''))
    invalid = sum(bool(set(row) - set(AA + '-')) for row in unique)
    pool = [row for row in unique if not set(row) - set(AA + '-')]
    pool.sort(key=lambda row: (hashlib.sha256(('42\n' + row).encode()).digest(), row))
    chosen = pool[:cap]
    if len(chosen) < 2:
        raise ValueError('Insufficient valid unique non-WT homologs')
    full_support = [sum(row[index] != '-' for row in pool) for index in range(len(positions))]
    retained = [index for index in range(len(positions)) if any(row[index] != '-' for row in chosen)]
    array = np.asarray([[AA.find(row[index]) for index in retained] for row in chosen], dtype=np.int8)
    if not retained:
        raise ValueError('No valid retained homolog columns')
    return array, [positions[index] for index in retained], dict(
        raw_sequence_count=len(raw), unique_valid_pool_count=len(pool), selected_count=len(chosen),
        ambiguous_homologs_excluded=invalid, selected_homologs_sha256=digest(chosen),
        full_pool_support_by_observed_index={str(position + 1): count for position, count in zip(positions, full_support)},
        **POLICY)


def inverse_identity_weights(array):
    neighbors = np.zeros(len(array))
    for offset in range(0, len(array), 16):
        a, b = array[offset:offset + 16, None, :], array[None, :, :]
        valid = (a >= 0) & (b >= 0)
        overlap = valid.sum(2)
        matches = ((a == b) & valid).sum(2)
        neighbors[offset:offset + 16] = ((overlap >= .5 * array.shape[1]) & (matches >= .8 * overlap)).sum(1)
    return 1 / np.maximum(neighbors, 1)


class ProteinRNAFeatureBuilder(FeatureBuilder):
    def profile(self, task, chain):
        mapping_path = self.msa_root / 'mappings' / f"{task['source']}__{task['reference_complex_id']}__{chain['chain']}.json"
        mapping = json.loads(mapping_path.read_text())
        columns = mapping['columns']
        if [str(row['position']) + (row.get('insertion_code') or '') for row in columns] != chain['residue_ids']:
            raise ValueError('Frozen MSA/PDB mapping differs')
        msa_path = self.msa_root / mapping['msa_path']
        hashes = dict(msa=file_hash(msa_path), mapping=file_hash(mapping_path),
                      observed_reference=digest(chain['resolved_sequence']))
        key = digest(dict(source_hashes=hashes, policy=POLICY))
        table_path = self.output / 'frozen_probability_tables' / f'{key}.json'
        weights_path = self.output / 'msa_weights' / f'{key}.npz'
        if key not in self.profiles:
            if table_path.exists():
                self.profiles[key] = json.loads(table_path.read_text())
            else:
                array, positions, provenance = selected_alignment(msa_path, chain['resolved_sequence'])
                weights = inverse_identity_weights(array)
                neff = float(weights.sum())
                counts = np.stack([((array == index) * weights[:, None]).sum(0) for index in range(20)], axis=1)
                valid = counts.sum(1)
                probabilities = (counts + .05) / (valid[:, None] + 1)
                table = dict(AA_order=AA, source_hashes=hashes, homolog_count=len(weights), global_neff=neff,
                    pseudocount_total=1.0, min_global_neff=0, legacy_strict_min_global_neff=20,
                    rows=[dict(msa_column=position + 1, probabilities_20=[float(p) for p in probabilities[index]],
                               weighted_valid_count=float(valid[index]),
                               weighted_gap_fraction=float(((array[:, index] == -1) * weights).sum() / neff),
                               weighted_unknown_fraction=0.0) for index, position in enumerate(positions)],
                    provenance=provenance, msa_policy=POLICY,
                    probability_meaning='Weighted empirical amino-acid frequencies with total pseudocount 1; not probabilities of homology',
                    unknown_policy='Whole ambiguous homologs excluded before selection and reweighting')
                weights_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(weights_path, weights=weights, retained_msa_columns=np.asarray(positions) + 1)
                table['weights_sha256'] = file_hash(weights_path)
                save(table_path, table)
                self.profiles[key] = table
        table = self.profiles[key]
        if table['source_hashes'] != hashes or table['msa_policy'] != POLICY:
            raise ValueError('MSA profile source/policy differs')
        return table, dict(mapping_path=str(mapping_path), msa_path=str(msa_path),
                          probability_table=str(table_path), source_hashes=hashes,
                          coordinate_columns={(row['position'], row.get('insertion_code') or ''): row['msa_column'] for row in columns})

    def build(self, task, observed_task, partner_groups):
        features, mapping = super().build(task, observed_task, partner_groups)
        features['legacy_strict_independent_msa_scores'] = dict(features['independent_msa_scores'])
        scores = {}
        for candidate, sites in features['msa'].items():
            complete = all(row['status'] in ('ok', 'global_neff_below_frozen_20') and
                           row['weighted_valid_count'] > 0 and math.isfinite(row['log_probability_ratio'])
                           for row in sites.values())
            scores[candidate] = sum(row['log_probability_ratio'] for row in sites.values()) if complete else None
        features.update(independent_msa_scores=scores, full_msa_coverage=all(v is not None for v in scores.values()),
                        msa_scoring_policy=POLICY,
                        legacy_strict_full_msa_coverage=all(v is not None for v in features['legacy_strict_independent_msa_scores'].values()))
        return features, mapping
