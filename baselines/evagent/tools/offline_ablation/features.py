"""Frozen MSA and WT geometry for an explicit two-group partner partition.

Partner groups may contain protein or nucleic-acid chains. Membership comes
from the frozen task, never from nearest-chain geometry or experimental labels.
"""
from __future__ import annotations

import copy
import json
import math
import sys
from pathlib import Path

import numpy as np
from Bio.PDB import PDBParser, ShrakeRupley
from Bio.PDB.Atom import Atom
from Bio.PDB.Chain import Chain
from Bio.PDB.Model import Model
from Bio.PDB.Residue import Residue
from Bio.PDB.Structure import Structure
from Bio.SeqUtils import seq1
from scipy.spatial.distance import cdist

from evagent.offline_ablation import base_sites, digest, file_hash

AA = 'ACDEFGHIKLMNPQRSTVWY'
SIDECHAIN = dict(zip('ARNDCQEGHILKMFPSTWYV', [
    'CB', 'CB CG CD NE CZ NH1 NH2', 'CB CG OD1 ND2', 'CB CG OD1 OD2',
    'CB SG', 'CB CG CD1 CD2', 'CB CG CD OE1 OE2', '',
    'CB CG ND1 CD2 CE1 NE2', 'CB CG1 CG2 CD1', 'CB CG CD1 CD2',
    'CB CG CD CE NZ', 'CB CG SD CE', 'CB CG CD1 CD2 CE1 CE2 CZ',
    'CB CG CD', 'CB OG', 'CB OG1 CG2', 'CB CG CD1 CD2 NE1 CE2 CE3 CZ2 CH2',
    'CB CG CD1 CD2 CE1 CE2 CZ OH', 'CB CG1 CG2',
]))
# Q, E and W are explicit to avoid relying on library-specific residue tables.
SIDECHAIN['Q'] = 'CB CG CD OE1 NE2'
SIDECHAIN['E'] = 'CB CG CD OE1 OE2'
SIDECHAIN['W'] = 'CB CG CD1 CD2 NE1 CE2 CE3 CZ2 CZ3 CH2'
SIDECHAIN = {key: set(value.split()) for key, value in SIDECHAIN.items()}


DELTA_SASA_TEXT = (
    'SASA of this residue in the own partner group after the specified partner is removed, '
    'minus SASA of the same residue in the two-group complex'
)


def save(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    if path.exists():
        if path.read_text() != text:
            raise ValueError(f'Frozen feature changed: {path}')
        return
    with path.open('x') as handle:
        handle.write(text)


def migrated(path: str, root: Path) -> Path:
    old = '/data/yanjie_huang/EVAgent/'
    return root / path[len(old):] if path.startswith(old) else Path(path)


class FeatureBuilder:
    def __init__(self, root: Path, output: Path):
        self.root, self.output = root, output
        self.msa_root = root / 'data/final_mutation_data/msa_completion_20260927'
        self.profiles, self.structures = {}, {}
        sys.path.insert(0, str(root / 'baseline'))
        from singlechain import models
        self.models = models

    def profile(self, task: dict, chain: dict) -> tuple[dict, dict]:
        mapping_path = self.msa_root / 'mappings' / (
            f"{task['source']}__{task['reference_complex_id']}__{chain['chain']}.json")
        mapping = json.loads(mapping_path.read_text())
        columns = mapping['columns']
        coordinate_ids = [str(row['position']) + (row.get('insertion_code') or '') for row in columns]
        if coordinate_ids != chain['residue_ids']:
            raise ValueError('Frozen PDB/MSA residue map differs')
        msa_path = self.msa_root / mapping['msa_path']
        hashes = dict(msa=file_hash(msa_path), mapping=file_hash(mapping_path),
                      observed_reference=digest(chain['resolved_sequence']))
        key = digest(hashes)
        if key not in self.profiles:
            table_path = self.output / 'frozen_probability_tables' / f'{key}.json'
            weights_path = self.output / 'msa_weights' / f'{key}.npz'
            if table_path.exists():
                self.profiles[key] = json.loads(table_path.read_text())
            else:
                item = dict(base=chain['resolved_sequence'], modality='protein', candidates=[],
                            evidence={'msa': dict(path=str(msa_path), sha256=hashes['msa'], start=1, min_neff=0)})
                array, weights, positions, alphabet, provenance = self.models.alignment(item)
                counts = np.stack([((array == index) * weights[:, None]).sum(0)
                                   for index in range(len(alphabet))], axis=1)
                valid = counts.sum(1)
                probabilities = (counts + 1 / len(alphabet)) / (valid[:, None] + 1)
                neff = float(weights.sum())
                if not np.allclose(probabilities.sum(1), 1, atol=1e-12):
                    raise ValueError('Frozen amino-acid probabilities do not sum to one')
                table = dict(AA_order=alphabet, source_hashes=hashes,
                             homolog_count=len(weights), global_neff=neff,
                             pseudocount_total=1.0, min_global_neff=20,
                             rows=[dict(msa_column=position + 1,
                                        probabilities_20=[float(value) for value in row],
                                        weighted_valid_count=float(count),
                                        weighted_gap_fraction=float(((array[:, index] == -1) * weights).sum() / neff),
                                        weighted_unknown_fraction=float(((array[:, index] < -1) * weights).sum() / neff))
                                   for index, (position, row, count) in enumerate(zip(positions, probabilities, valid))],
                             probability_meaning='Weighted empirical MSA amino-acid frequency with total pseudocount 1; not a probability of homology',
                             unknown_policy='Legacy protocol excludes entire ambiguous homolog sequences before reweighting; retained unknown fraction is zero',
                             provenance={key: value for key, value in provenance.items()
                                         if key not in ('homolog_sequences', 'query')})
                weights_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(weights_path, weights=weights, retained_msa_columns=np.asarray(positions) + 1)
                table['weights_sha256'] = file_hash(weights_path)
                save(table_path, table)
                self.profiles[key] = table
        table = self.profiles[key]
        if table['source_hashes'] != hashes:
            raise ValueError('Cached MSA source identity changed')
        return table, dict(mapping_path=str(mapping_path), msa_path=str(msa_path),
                           probability_table=str(self.output / 'frozen_probability_tables' / f'{key}.json'),
                           source_hashes=hashes,
                           coordinate_columns={(row['position'], row.get('insertion_code') or ''): row['msa_column']
                                               for row in columns})

    def geometry(self, task: dict, groups: dict) -> dict:
        path = migrated(task['structure_source']['path'], self.root)
        if file_hash(path) != task['structure_source']['sha256']:
            raise ValueError('Migrated structure hash differs from the frozen source')
        partition = groups['groups']
        if len(partition) != 2 or set(partition[0]) & set(partition[1]):
            raise ValueError('Partner partition must be two disjoint groups')
        key = digest(dict(structure_sha256=file_hash(path), groups=partition))
        if key in self.structures:
            return self.structures[key]
        structure = PDBParser(QUIET=True).get_structure('source', path)
        model = next(structure.get_models())
        bound = Structure('frozen_complex'); bound.add(Model(0))
        modeled = {chain['chain']: set(chain['residue_ids']) for chain in task['wt_chains']}
        for chain_id in sum(partition, []):
            if chain_id not in model or chain_id not in modeled:
                raise ValueError('Specified partner chain lacks a frozen coordinate model')
            new_chain = Chain(chain_id); bound[0].add(new_chain)
            for residue in model[chain_id]:
                coordinate_id = str(residue.id[1]) + residue.id[2].strip()
                if coordinate_id not in modeled[chain_id]:
                    continue
                atoms = [atom for atom in residue.get_unpacked_list()
                         if atom.element not in ('H', 'D') and np.isfinite(atom.coord).all()]
                alternatives = {atom.altloc for atom in atoms if atom.altloc.strip()}
                chosen = max(alternatives, key=lambda label: (
                    sum((atom.occupancy or 0) for atom in atoms if atom.altloc == label),
                    sum(atom.altloc == label for atom in atoms), label == 'A', label)) if alternatives else None
                selected = [atom for atom in atoms if not atom.altloc.strip() or atom.altloc == chosen]
                new_residue = Residue(residue.id, residue.resname, residue.segid)
                names = {}
                for atom in selected:
                    if atom.name not in names or (atom.occupancy or 0) > (names[atom.name].occupancy or 0):
                        names[atom.name] = atom
                if not names:
                    continue
                new_chain.add(new_residue)
                for atom in names.values():
                    # Atom.copy() detaches the parent before copying coordinates;
                    # deepcopy(atom) would copy its entire parent structure.
                    copied = atom.copy(); copied.set_altloc(' ')
                    new_residue.add(copied)
        # SASA is evaluated on the specified two-group complex, then separately
        # on the complete own group (H/L stay together, for example).
        sasa = ShrakeRupley(probe_radius=1.4, n_points=100)
        sasa.compute(bound, level='R')
        isolated = []
        for own in partition:
            value = copy.deepcopy(bound)
            for chain_id in list(value[0].child_dict):
                if chain_id not in own:
                    value[0].detach_child(chain_id)
            sasa.compute(value, level='R')
            isolated.append(value)
        residue_map = {(chain.id, residue.id[1], residue.id[2].strip()): residue
                       for chain in bound[0] for residue in chain}
        result = dict(bound=bound, isolated=isolated, residue_map=residue_map,
                      groups=partition, source_path=str(path), source_sha256=file_hash(path),
                      all_heavy_occupancies_zero=all((atom.occupancy or 0) == 0 for atom in bound.get_atoms()))
        self.structures[key] = result
        return result

    def build(self, task: dict, observed_task: dict, partner_groups: dict, *,
              unavailable_structure_reason: str | None = None) -> tuple[dict, dict]:
        mapping, base = base_sites(task)
        if partner_groups['candidate_hash'] != observed_task['candidate_hash']:
            raise ValueError('Partner definition belongs to another candidate freeze')
        chains = {chain['chain']: chain for chain in observed_task['wt_chains']}
        required = sorted({site['chain'] for site in mapping.values()})
        profiles, metadata, profile_errors = {}, {}, {}
        for chain_id in required:
            try:
                profiles[chain_id], metadata[chain_id] = self.profile(observed_task, chains[chain_id])
            except (ValueError, FileNotFoundError) as error:
                profile_errors[chain_id] = str(error)
        if unavailable_structure_reason:
            geometry = dict(groups=partner_groups['groups'], residue_map={},
                            source_path=observed_task['structure_source']['path'],
                            source_sha256=observed_task['structure_source']['sha256'],
                            all_heavy_occupancies_zero=None,
                            unavailable_reason=unavailable_structure_reason)
        else:
            geometry = self.geometry(observed_task, partner_groups)
        structure_rows = {}
        for sid, site in mapping.items():
            chain_id, pos, insertion = site['chain'], site['position'], site.get('insertion_code') or ''
            own_index = next((index for index, group in enumerate(geometry['groups']) if chain_id in group), None)
            partners = list(geometry['groups'][1 - own_index]) if own_index is not None else []
            residue = geometry['residue_map'].get((chain_id, pos, insertion))
            row = dict(min_partner_heavy_distance_A=None, partner_contact_count_5A=None,
                       sasa_bound_A2=None, delta_sasa_partner_A2=None,
                       site_status='missing_coordinates', partner_status='missing_partner',
                       sasa_status='unavailable', partner_chain_ids=partners,
                       source='Frozen WT complex, first model, coherent heavy-atom conformer')
            if unavailable_structure_reason:
                row['source'] = unavailable_structure_reason
            partner_atoms = [(key, atom) for key, other in geometry['residue_map'].items()
                             if key[0] in partners for atom in other]
            if residue is not None:
                observed_aa = seq1(residue.resname, custom_map={'MSE': 'M'})
                row['site_status'] = 'ok' if observed_aa == site['wt'] else 'WT_identity_mismatch'
            if partner_atoms:
                row['partner_status'] = 'ok'
            if row['site_status'] == 'ok' and partner_atoms:
                distances = cdist(np.asarray([atom.coord for atom in residue]),
                                  np.asarray([atom.coord for _, atom in partner_atoms]))
                row['min_partner_heavy_distance_A'] = float(distances.min())
                row['partner_contact_count_5A'] = len({partner_atoms[index][0]
                                                     for index in np.flatnonzero((distances <= 5).any(axis=0))})
                needed = {'N', 'CA', 'C', 'O'} | SIDECHAIN.get(site['wt'], set())
                if needed <= set(residue.child_dict):
                    unbound = geometry['isolated'][own_index][0][chain_id][residue.id]
                    row['sasa_bound_A2'] = float(residue.sasa)
                    row['delta_sasa_partner_A2'] = float(unbound.sasa - residue.sasa)
                    row['sasa_status'] = 'ok'
                else:
                    row['sasa_status'] = 'missing_WT_heavy_atoms'
            structure_rows[sid] = row
            mapping[sid].update(structure_sha256=geometry['source_sha256'],
                                structure_path=geometry['source_path'])
        features = dict(msa_chains={}, msa={}, structure={}, structure_definition=dict(
            partner_groups=geometry['groups'], partner_definition=partner_groups['evidence'],
            excluded_extra_chains=partner_groups['excluded_extra_chains'],
            distance='Minimum distance between site heavy atoms and specified partner heavy atoms',
            contacts='Number of distinct specified partner residues with any heavy atom within 5 A',
            sasa='Bio.PDB ShrakeRupley, probe 1.4 A, 100 points; observed WT heavy atoms; missing site heavy atoms yield null',
            delta_sasa=DELTA_SASA_TEXT,
            units=dict(distance='angstrom', contacts='partner residues', sasa='angstrom^2'),
            source_sha256=geometry['source_sha256'],
            all_heavy_occupancies_zero=geometry['all_heavy_occupancies_zero'],
            unavailable_reason=geometry.get('unavailable_reason')))
        for chain_id in required:
            table = profiles.get(chain_id)
            features['msa_chains'][chain_id] = dict(
                homolog_count=table['homolog_count'] if table else None,
                global_neff=table['global_neff'] if table else None)
        msa_scores, statuses = {}, []
        for cid, sites in base.items():
            features['msa'][cid], features['structure'][cid] = {}, {}
            score = 0.0; complete = True
            for site in sites:
                sid = site['site_id']; identity = mapping[sid]; chain_id = identity['chain']
                table = profiles.get(chain_id)
                row = dict(p_wt=None, p_mutant=None, log_probability_ratio=None,
                           weighted_valid_count=None, weighted_gap_fraction=None,
                           weighted_unknown_fraction=None, status='unavailable_MSA')
                if table:
                    coordinate = (identity['position'], identity.get('insertion_code') or '')
                    column = metadata[chain_id]['coordinate_columns'].get(coordinate)
                    mapping[sid].update(msa_column=column,
                                        msa_source_hashes=table['source_hashes'],
                                        probability_table=metadata[chain_id]['probability_table'])
                    value = next((entry for entry in table['rows'] if entry['msa_column'] == column), None)
                    if value is None:
                        # A coordinate absent from the frozen table is missing evidence.
                        # Do not impute a zero count or an all-gap fraction.
                        row['status'] = 'no_valid_homolog_at_column'
                    else:
                        coordinate_id = str(identity['position']) + (identity.get('insertion_code') or '')
                        index = chains[chain_id]['residue_ids'].index(coordinate_id)
                        if chains[chain_id]['resolved_sequence'][index] != site['wt']:
                            row['status'] = 'WT_identity_mismatch'
                        else:
                            probs = value['probabilities_20']
                            pw, pm = probs[AA.index(site['wt'])], probs[AA.index(site['mutant'])]
                            row.update(p_wt=pw, p_mutant=pm, log_probability_ratio=math.log(pm / pw),
                                       weighted_valid_count=value['weighted_valid_count'],
                                       weighted_gap_fraction=value['weighted_gap_fraction'],
                                       weighted_unknown_fraction=value['weighted_unknown_fraction'],
                                       status='ok' if table['global_neff'] >= 20 else 'global_neff_below_frozen_20')
                else:
                    mapping[sid].update(msa_column=None, msa_source_hashes=None,
                                        msa_error=profile_errors.get(chain_id))
                statuses.append(row['status'])
                complete &= row['status'] == 'ok'
                if row['status'] == 'ok':
                    score += row['log_probability_ratio']
                features['msa'][cid][sid] = row
                features['structure'][cid][sid] = structure_rows[sid]
            msa_scores[cid] = score if complete else None
        features['independent_msa_scores'] = msa_scores
        features['full_msa_coverage'] = all(value is not None for value in msa_scores.values())
        features['source_provenance'] = dict(msa_chains={key: {k: v for k, v in value.items() if k != 'coordinate_columns'}
                                                      for key, value in metadata.items()},
                                            structure_path=geometry['source_path'],
                                            profile_errors=profile_errors)
        return features, dict(sites=mapping, all_msa_site_statuses=sorted(set(statuses)))
