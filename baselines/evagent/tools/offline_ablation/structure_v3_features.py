"""Label-free WT environment and substitution descriptors for protein/RNA.

Interactions below are explicit geometry/typing candidates, not measured bonds,
energies, protonation assignments, or modeled mutant structures.
"""
from __future__ import annotations

import copy
import math
import numpy as np
from evagent.offline_ablation import base_sites
from features import SIDECHAIN

AA_DONORS = {'R': {'NE', 'NH1', 'NH2'}, 'K': {'NZ'}, 'N': {'ND2'},
             'Q': {'NE2'}, 'S': {'OG'}, 'T': {'OG1'}, 'Y': {'OH'}, 'W': {'NE1'}}
AA_ACCEPTORS = {'D': {'OD1', 'OD2'}, 'E': {'OE1', 'OE2'}, 'N': {'OD1'},
                'Q': {'OE1'}, 'S': {'OG'}, 'T': {'OG1'}, 'Y': {'OH'}, 'M': {'SD'}}
RNA_DONORS = {'A': {'N6'}, 'G': {'N1', 'N2'}, 'C': {'N4'}, 'U': {'N3'}}
RNA_ACCEPTORS = {'A': {'N1', 'N3', 'N7'}, 'G': {'N3', 'N7', 'O6'},
                 'C': {'N3', 'O2'}, 'U': {'O2', 'O4'}}
PHOSPHATE = {'P', 'OP1', 'OP2', 'OP3', 'O1P', 'O2P', 'O3P'}
PHOSPHATE_O = PHOSPHATE - {'P'}
RNA_NAMES = {'A', 'G', 'C', 'U', 'RA', 'RG', 'RC', 'RU', 'ADE', 'GUA', 'CYT', 'URA',
             'PSU', '5MU', 'H2U', '1MA', '2MG', 'M2G', '7MG', 'OMG', 'OMC', 'YG'}
CANONICAL = {'A': 'A', 'G': 'G', 'C': 'C', 'U': 'U', 'RA': 'A', 'RG': 'G',
             'RC': 'C', 'RU': 'U', 'ADE': 'A', 'GUA': 'G', 'CYT': 'C', 'URA': 'U'}
AA_RINGS = {'F': [('CG', 'CD1', 'CE1', 'CZ', 'CE2', 'CD2')],
            'Y': [('CG', 'CD1', 'CE1', 'CZ', 'CE2', 'CD2')],
            'H': [('CG', 'ND1', 'CE1', 'NE2', 'CD2')],
            'W': [('CG', 'CD1', 'NE1', 'CE2', 'CD2'),
                  ('CD2', 'CE2', 'CZ2', 'CH2', 'CZ3', 'CE3')]}
PURINE_RINGS = [('N9', 'C8', 'N7', 'C5', 'C4'), ('C4', 'C5', 'C6', 'N1', 'C2', 'N3')]
PYRIMIDINE_RINGS = [('N1', 'C2', 'N3', 'C4', 'C5', 'C6')]
RNA_RINGS = {'A': PURINE_RINGS, 'G': PURINE_RINGS, 'C': PYRIMIDINE_RINGS,
             'U': PYRIMIDINE_RINGS, 'PSU': PYRIMIDINE_RINGS, '5MU': PYRIMIDINE_RINGS}

DEFINITION = {
    'version': 'protein-rna-wt-chemistry-v3',
    'substitution_properties': 'Fixed amino-acid sidechain descriptors; size is heavy-atom count, not volume. Nominal charges are not pH-specific protonation estimates. Histidine and cysteine donor/acceptor ambiguity remains null.',
    'rna_contact_profile': 'Distinct designated RNA residue counts and atom-pair counts within 5 A, separated into base, ribose, phosphate and unknown. One residue may contact multiple components.',
    'polar_proximity': 'Nominal donor/acceptor heavy-atom pairs within 3.5 A. No hydrogens or donor angles are inferred: these are proximity candidates, not confirmed hydrogen bonds. Canonical base typing only; modified-base chemistry remains untyped.',
    'ionic_proximity': 'WT Lys NZ or Arg NE/NH1/NH2 versus RNA phosphate oxygen within 4 A; nominal ionic-contact candidates, not electrostatic energies. Histidine protonation is not assumed.',
    'aromatic_geometry': 'Complete recognized WT aromatic rings and RNA base rings: centroid distance <=5.5 A, acute normal angle <=30 degrees, lateral offset relative to both ring planes <=2 A; count distinct RNA residues. No mutant rings are modeled. Nonplanar/incomplete rings unavailable.',
    'interpretation': 'WT accessibility, distance and contact counts describe environment and do not determine the sign of a substitution effect. Combine with substitution chemistry and sequence/MSA; no automatic preference for closer or more buried sites.',
}


def properties(aa):
    if aa not in SIDECHAIN:
        raise ValueError('Unknown amino-acid descriptor')
    charge = None if aa == 'H' else (1 if aa in 'KR' else -1 if aa in 'DE' else 0)
    return dict(nominal_charge=charge, sidechain_heavy_atom_count=len(SIDECHAIN[aa]),
                polarity_class=('protonation_dependent' if aa == 'H' else
                                'charged' if aa in 'KRDE' else 'polar' if aa in 'STNQCY' else 'nonpolar'),
                aromatic=aa in 'FYW H'.replace(' ', ''),
                nominal_sidechain_donors=None if aa in 'HC' else len(AA_DONORS.get(aa, ())),
                nominal_sidechain_acceptors=None if aa in 'HC' else len(AA_ACCEPTORS.get(aa, ())),
                glycine=aa == 'G', proline=aa == 'P')


def substitution(wt, mutant):
    a, b = properties(wt), properties(mutant)
    numeric = ('nominal_charge', 'sidechain_heavy_atom_count',
               'nominal_sidechain_donors', 'nominal_sidechain_acceptors')
    changes = {key: None if a[key] is None or b[key] is None else b[key] - a[key] for key in numeric}
    changes.update(aromaticity_changed=a['aromatic'] != b['aromatic'],
                   polarity_class_changed=a['polarity_class'] != b['polarity_class'])
    return dict(wt=a, mutant=b, mutant_minus_wt=changes)


def atom_name(atom):
    return atom.name.strip().replace('*', "'")


def component(residue, atom):
    name = atom_name(atom)
    if name in PHOSPHATE:
        return 'phosphate'
    if "'" in name:
        return 'ribose'
    return 'base' if residue.resname.strip() in RNA_NAMES else 'unknown'


def rna_roles(residue, atom):
    name = atom_name(atom)
    part = component(residue, atom)
    if part == 'phosphate':
        return False, name in PHOSPHATE_O, True
    if part == 'ribose':
        return name == "O2'", name in {"O2'", "O3'", "O4'", "O5'"}, True
    parent = CANONICAL.get(residue.resname.strip())
    if parent is None:
        return False, False, False
    return name in RNA_DONORS[parent], name in RNA_ACCEPTORS[parent], True


def protein_roles(wt, atom):
    name = atom_name(atom)
    if name in {'O', 'OXT'}:
        return False, True, True
    if name == 'N':
        return wt != 'P', False, True
    if wt in 'HC' and name in SIDECHAIN[wt]:
        return False, False, False
    return name in AA_DONORS.get(wt, ()), name in AA_ACCEPTORS.get(wt, ()), True


def ring_plane(residue, names):
    if not set(names) <= set(residue.child_dict):
        return None
    xyz = np.asarray([residue[name].coord for name in names], dtype=float)
    centre = xyz.mean(axis=0)
    _, singular, basis = np.linalg.svd(xyz - centre)
    if singular[1] < 1e-6:
        return None
    normal = basis[-1]
    if np.sqrt(np.mean(((xyz - centre) @ normal) ** 2)) > .15:
        return None
    return centre, normal


def stacking_candidate(first, second):
    if first is None or second is None:
        return False
    x, nx = first; y, ny = second
    displacement = y - x
    distance = float(np.linalg.norm(displacement))
    angle = math.degrees(math.acos(float(np.clip(abs(np.dot(nx, ny)), 0, 1))))
    offsets = [float(np.linalg.norm(displacement - np.dot(displacement, n) * n)) for n in (nx, ny)]
    return distance <= 5.5 and angle <= 30 and max(offsets) <= 2


def environment(residue, wt, partner_residues):
    if residue is None or not partner_residues:
        return dict(rna_contact_profile=None, wt_interaction_profile=None, status='unavailable')
    parts = ('base', 'ribose', 'phosphate', 'unknown')
    contacts = {part: set() for part in parts}
    pairs = {group: dict.fromkeys(parts, 0) for group in ('sidechain', 'backbone')}
    polar = {'sidechain': 0, 'backbone': 0}
    ionic, untyped = 0, 0
    pro_rings = [ring_plane(residue, names) for names in AA_RINGS.get(wt, ())]
    stacked = set(); recognized_rings = 0
    for key, partner in partner_residues:
        for pa in residue:
            group = 'sidechain' if pa.name in SIDECHAIN[wt] else 'backbone'
            pd, pac, ptyped = protein_roles(wt, pa)
            for ra in partner:
                distance = float(np.linalg.norm(pa.coord - ra.coord))
                part = component(partner, ra)
                if distance <= 5:
                    contacts[part].add(key); pairs[group][part] += 1
                rd, rac, rtyped = rna_roles(partner, ra)
                if distance <= 3.5:
                    if ptyped and rtyped and ((pd and rac) or (pac and rd)):
                        polar[group] += 1
                    elif not ptyped or not rtyped:
                        untyped += 1
                if distance <= 4 and ((wt == 'K' and pa.name == 'NZ') or
                                     (wt == 'R' and pa.name in {'NE', 'NH1', 'NH2'})) and atom_name(ra) in PHOSPHATE_O:
                    ionic += 1
        for names in RNA_RINGS.get(partner.resname.strip(), ()):
            ring = ring_plane(partner, names)
            recognized_rings += ring is not None
            if any(stacking_candidate(pro, ring) for pro in pro_rings):
                stacked.add(key)
    return dict(
        rna_contact_profile=dict(component_residue_counts_5A={k: len(v) for k, v in contacts.items()},
                                 sidechain_component_atom_pairs_5A=pairs['sidechain'],
                                 backbone_component_atom_pairs_5A=pairs['backbone']),
        wt_interaction_profile=dict(donor_acceptor_proximity_pairs_3_5A=polar,
                                    untyped_close_atom_pairs_3_5A=untyped,
                                    nominal_ionic_contact_pairs_4A=ionic,
                                    aromatic_stacking_candidate_RNA_residues=(len(stacked) if not AA_RINGS.get(wt) or
                                        (all(r is not None for r in pro_rings) and recognized_rings) else None),
                                    valid_protein_aromatic_rings=sum(r is not None for r in pro_rings),
                                    valid_partner_aromatic_rings=recognized_rings,
                                    status='WT_geometry_candidates_not_confirmed_bonds'), status='ok')


def build(task, original_features, geometry):
    """No labels, inference, structure relaxation, or candidate changes."""
    mapping, candidates = base_sites(task)
    sites = {}
    for sid, site in mapping.items():
        key = (site['chain'], site['position'], site.get('insertion_code') or '')
        own = next(i for i, group in enumerate(geometry['groups']) if site['chain'] in group)
        partner_ids = geometry['groups'][1 - own]
        partners = [(k, residue) for k, residue in geometry['residue_map'].items() if k[0] in partner_ids]
        source_row = next(rows[sid] for rows in original_features['structure'].values() if sid in rows)
        values = copy.deepcopy(source_row)
        values.update(environment(geometry['residue_map'].get(key), site['wt'], partners))
        values['site_identity'] = dict(site, site_id=sid)
        sites[sid] = values
    candidate_rows = {}
    for cid, rows in candidates.items():
        candidate_rows[cid] = [dict(site_id=row['site_id'], wt=row['wt'], mutant=row['mutant'],
                                    substitution_properties=substitution(row['wt'], row['mutant'])) for row in rows]
    return dict(definition=DEFINITION, wt_sites=sites, candidate_substitutions=candidate_rows)


def render(task, ids, features, condition, base_renderer):
    evidence = base_renderer(task, ids, features, condition)
    if condition == 'msa_prompt':
        return evidence
    extra = features['structure_v3']
    wanted = {row['site_id'] for cid in ids for row in extra['candidate_substitutions'][cid]}
    # WT fields appear only once per site; chemical changes remain per candidate.
    evidence.pop('structure_sites')
    evidence['structure_v3_definition'] = extra['definition']
    evidence['structure_wt_sites'] = {sid: extra['wt_sites'][sid] for sid in sorted(wanted)}
    evidence['candidate_substitutions'] = {cid: extra['candidate_substitutions'][cid] for cid in ids}
    return evidence
