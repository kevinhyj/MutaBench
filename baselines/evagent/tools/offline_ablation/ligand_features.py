"""WT-only features with the historical annotated-ligand partner definition."""
from __future__ import annotations

import copy
import math
from pathlib import Path

import numpy as np
from Bio.PDB import PDBParser, ShrakeRupley
from Bio.SeqUtils import seq1
from scipy.spatial.distance import cdist

from evagent.offline_ablation import base_sites, file_hash
from features import AA, SIDECHAIN, FeatureBuilder


class LigandFeatureBuilder(FeatureBuilder):
    def build_ligand(self, task: dict) -> tuple[dict, dict]:
        mapping, base = base_sites(task)
        chains = {chain['chain']: chain for chain in task['wt_chains']}
        required = sorted({site['chain'] for site in mapping.values()})
        profiles, metadata, errors = {}, {}, {}
        for ch in required:
            try:
                profiles[ch], metadata[ch] = self.profile(task, chains[ch])
            except (ValueError, FileNotFoundError, KeyError) as error:
                errors[ch] = f'{type(error).__name__}: {error}'
        path = Path(task['structure_source']['path'])
        if file_hash(path) != task['structure_source']['sha256']:
            raise ValueError('WT structure checksum changed')
        model = next(PDBParser(QUIET=True).get_structure('WT', path).get_models())
        ligand_id = task['binding_partner']['pdb_ligand_id']
        protein_ids = {(ch, value) for ch, chain in chains.items() for value in chain['residue_ids']}
        # Preserve every protein chain from the base input, and every occurrence
        # of the annotated ligand ID. Water and other cofactors are excluded.
        bound = copy.deepcopy(model)
        partner_keys = []
        for chain in list(bound):
            for res in list(chain):
                rid = str(res.id[1]) + res.id[2].strip()
                is_protein = res.id[0] == ' ' and (chain.id, rid) in protein_ids
                is_partner = res.id[0].startswith('H_') and res.resname == ligand_id
                if not (is_protein or is_partner):
                    chain.detach_child(res.id)
                    continue
                # Bio.PDB selects one highest-occupancy conformer per atom.
                for atom in list(res):
                    if atom.element in ('H', 'D') or not np.isfinite(atom.coord).all():
                        res.detach_child(atom.id)
                if is_partner and len(res):
                    partner_keys.append((chain.id, res.id))
            if not len(chain):
                bound.detach_child(chain.id)
        partner_atoms = [atom for ch, rid in partner_keys for atom in bound[ch][rid]]
        isolated = copy.deepcopy(bound)
        for ch, rid in partner_keys:
            isolated[ch].detach_child(rid)
        sasa_error = None
        try:
            sr = ShrakeRupley(probe_radius=1.4, n_points=100)
            sr.compute(bound, level='R')
            sr.compute(isolated, level='R')
        except (ValueError, KeyError) as error:
            sasa_error = f'{type(error).__name__}: {error}'
        definition = dict(
            partner_definition='All HETATM heavy atoms matching the source annotated ligand residue name, all copies in the first WT PDB model',
            pdb_ligand_id=ligand_id,
            partner_instances=[dict(chain=ch, position=rid[1], insertion_code=rid[2].strip())
                               for ch, rid in partner_keys],
            distance='Minimum site-heavy-atom to specified-ligand-heavy-atom distance',
            contacts='Number of specified partner heavy atoms within <=5 angstrom of any site heavy atom',
            sasa='Bio.PDB ShrakeRupley, probe 1.4 angstrom, 100 points; all base-input protein chains plus annotated ligand',
            delta_sasa='SASA after removing only the specified ligand minus SASA of the bound complex',
            excluded='Water, hydrogens, and other ligand/cofactor residue names',
            altloc='Bio.PDB highest-occupancy atom conformer selection',
            units=dict(distance='angstrom', contacts='partner heavy atoms', sasa='angstrom^2'),
            source=str(path), source_sha256=file_hash(path), sasa_error=sasa_error,
        )
        features = dict(msa_chains={}, msa={}, structure={}, structure_definition=definition)
        for ch in required:
            table = profiles.get(ch)
            features['msa_chains'][ch] = dict(homolog_count=table['homolog_count'] if table else None,
                                            global_neff=table['global_neff'] if table else None)
        structure_rows = {}
        for sid, site in mapping.items():
            ch, pos, ins = site['chain'], site['position'], site.get('insertion_code') or ''
            res = bound[ch].child_dict.get((' ', pos, ins or ' ')) if ch in bound else None
            row = dict(min_partner_heavy_distance_A=None, partner_contact_count_5A=None,
                       sasa_bound_A2=None, delta_sasa_partner_A2=None,
                       site_status='missing_coordinates', partner_status='ok' if partner_atoms else 'missing_annotated_ligand',
                       sasa_status='unavailable', partner_chain_ids=sorted({key[0] for key in partner_keys}),
                       source='Frozen first WT PDB model; angstrom distances and angstrom^2 SASA')
            if res is not None and len(res):
                row['site_status'] = 'ok' if seq1(res.resname, custom_map={'MSE': 'M'}) == site['wt'] else 'WT_identity_mismatch'
            if row['site_status'] == 'ok' and partner_atoms:
                distances = cdist(np.asarray([a.coord for a in res]), np.asarray([a.coord for a in partner_atoms]))
                row['min_partner_heavy_distance_A'] = float(distances.min())
                row['partner_contact_count_5A'] = int((distances <= 5).any(axis=0).sum())
                complete = ({'N', 'CA', 'C', 'O'} | SIDECHAIN[site['wt']]) <= set(res.child_dict)
                if complete and sasa_error is None:
                    row.update(sasa_bound_A2=float(res.sasa),
                               delta_sasa_partner_A2=float(isolated[ch][res.id].sasa - res.sasa), sasa_status='ok')
                else:
                    row['sasa_status'] = 'calculation_unavailable' if sasa_error else 'missing_WT_heavy_atoms'
            structure_rows[sid] = row
            mapping[sid].update(structure_path=str(path), structure_sha256=file_hash(path))
        scores = {}
        for cid, sites in base.items():
            features['msa'][cid], features['structure'][cid] = {}, {}
            total, complete = 0.0, True
            for site in sites:
                sid, ch = site['site_id'], site['chain']
                table = profiles.get(ch)
                row = dict(p_wt=None, p_mutant=None, log_probability_ratio=None,
                           weighted_valid_count=None, weighted_gap_fraction=None,
                           weighted_unknown_fraction=None, status='unavailable_MSA')
                if table:
                    column = metadata[ch]['coordinate_columns'].get((site['position'], site.get('insertion_code') or ''))
                    mapping[sid].update(msa_column=column, msa_source_hashes=table['source_hashes'],
                                        probability_table=metadata[ch]['probability_table'])
                    value = next((r for r in table['rows'] if r['msa_column'] == column), None)
                    if value:
                        probs = value['probabilities_20']
                        pw, pm = probs[AA.index(site['wt'])], probs[AA.index(site['mutant'])]
                        row.update(p_wt=pw, p_mutant=pm, log_probability_ratio=math.log(pm / pw),
                                   weighted_valid_count=value['weighted_valid_count'],
                                   weighted_gap_fraction=value['weighted_gap_fraction'],
                                   weighted_unknown_fraction=value['weighted_unknown_fraction'],
                                   status='ok' if table['global_neff'] >= 20 else 'global_neff_below_frozen_20')
                    else:
                        row['status'] = 'no_valid_homolog_at_column'
                else:
                    mapping[sid].update(msa_column=None, msa_source_hashes=None, msa_error=errors.get(ch))
                complete &= row['status'] == 'ok'
                if row['status'] == 'ok':
                    total += row['log_probability_ratio']
                features['msa'][cid][sid] = row
                features['structure'][cid][sid] = structure_rows[sid]
            scores[cid] = total if complete else None
        features.update(independent_msa_scores=scores, full_msa_coverage=all(v is not None for v in scores.values()),
                        source_provenance=dict(profile_errors=errors, msa_chains={
                            ch: {k: v for k, v in meta.items() if k != 'coordinate_columns'} for ch, meta in metadata.items()}))
        return features, dict(sites=mapping)
