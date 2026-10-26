"""WT-only protein–RNA input v3 from frozen coordinates and reviewed metadata.

Partial-scope numeric evidence remains in audit tables, never in the main prompt.
No candidates, sequence models, evaluation labels or model inference are created.
"""
from __future__ import annotations
import copy
from pathlib import Path
import numpy as np
from Bio.PDB import PDBParser, ShrakeRupley
from Bio.PDB.Chain import Chain
from Bio.PDB.Model import Model
from Bio.PDB.Residue import Residue
from Bio.PDB.Structure import Structure
from Bio.SeqUtils import seq1
from scipy.spatial.distance import cdist
from features import SIDECHAIN
from evagent.offline_ablation import base_sites, digest, file_hash

VERSION = 'protein-rna-structure-prompt-v3'
BACKBONE = {'N', 'CA', 'C', 'O', 'OXT'}
RIBOSE = set("C1' C2' C3' C4' C5' O2' O3' O4' O5'".split())
PHOSPHATE = {'P', 'OP1', 'OP2', 'OP3'}
BASE_ATOMS = {
    'A': set('N9 C8 N7 C5 C6 N6 N1 C2 N3 C4'.split()),
    'G': set('N9 C8 N7 C5 C6 O6 N1 C2 N2 N3 C4'.split()),
    'C': set('N1 C2 O2 N3 C4 N4 C5 C6'.split()),
    'U': set('N1 C2 O2 N3 C4 O4 C5 C6'.split()),
}
RNA_NULL = ('min_partner_heavy_distance_A', 'min_sidechain_rna_distance_A',
            'partner_contact_count_5A', 'rna_contact_count_5A_by_moiety', 'rna_contacts')
SASA_NULL = ('sasa_bound_A2', 'sasa_without_rna_A2', 'delta_sasa_partner_A2', 'rna_burial_fraction')


def canonical_atom(name):
    name = name.strip().replace('*', "'")
    return {'O1P': 'OP1', 'O2P': 'OP2', 'O3P': 'OP3'}.get(name, name)


def moiety(residue_name, atom_name):
    """Modified bases remain unknown without an audited chemical component map."""
    if residue_name not in BASE_ATOMS:
        return None
    name = canonical_atom(atom_name)
    if name in RIBOSE:
        return 'ribose'
    if name in PHOSPHATE:
        return 'phosphate'
    if name in BASE_ATOMS[residue_name]:
        return 'base'
    return None


def coherent_residue(residue):
    atoms = [a for a in residue.get_unpacked_list()
             if a.element not in ('H', 'D') and np.isfinite(a.coord).all()]
    alts = {a.altloc for a in atoms if a.altloc.strip()}
    chosen = max(alts, key=lambda label: (
        sum((a.occupancy or 0) for a in atoms if a.altloc == label),
        sum(a.altloc == label for a in atoms), label == 'A', label)) if alts else None
    selected = {}
    for atom in atoms:
        if atom.altloc.strip() and atom.altloc != chosen:
            continue
        if atom.name not in selected or (atom.occupancy or 0) > (selected[atom.name].occupancy or 0):
            selected[atom.name] = atom
    result = Residue(residue.id, residue.resname, residue.segid)
    for atom in selected.values():
        value = atom.copy(); value.set_altloc(' '); result.add(value)
    return result, chosen


def header(path):
    lines = path.read_text().splitlines()
    seqres = {}
    for line in lines:
        if line.startswith('SEQRES'):
            seqres.setdefault(line[11], []).extend(line[19:70].split())
    resolution = next((float(s.split()[3]) for s in lines
                       if s.startswith('REMARK   2 RESOLUTION.') and 'ANGSTROMS' in s), None)
    method = ' '.join(s[10:].strip() for s in lines if s.startswith('EXPDTA')).strip() or 'unknown'
    evidence = [s for s in lines if s.startswith(('JRNL', 'REMARK 350', 'REMARK 465', 'REMARK 470'))]
    return dict(seqres=seqres, resolution_A=resolution, structure_method=method, evidence=evidence)


def coordinate_audit(chain, expected_names):
    """Full polymer length and expected heavy atoms, including terminal chemistry."""
    missing, unknown = [], []
    if len(chain) != len(expected_names):
        missing.append(dict(reason='polymer_residue_count_mismatch', expected=len(expected_names), observed=len(chain)))
    if [r.resname for r in chain] != expected_names:
        missing.append(dict(reason='polymer_sequence_or_order_mismatch'))
    for index, residue in enumerate(chain):
        names = {canonical_atom(a.name) for a in residue}
        if residue.resname not in BASE_ATOMS:
            unknown.append(dict(position=residue.id[1], chem_comp_id=residue.resname,
                                reason='modified_component_heavy_atom_definition_unreviewed'))
            continue
        needed = BASE_ATOMS[residue.resname] | RIBOSE
        if index != 0 or 'P' in names:
            needed = needed | {'P', 'OP1', 'OP2'}
        absent = sorted(needed - names)
        if absent:
            missing.append(dict(position=residue.id[1], insertion_code=residue.id[2].strip(), absent_atoms=absent))
        if any(moiety(residue.resname, atom.name) is None for atom in residue):
            unknown.append(dict(position=residue.id[1], reason='unclassified_observed_heavy_atom'))
    return dict(status='partial' if missing else 'unknown' if unknown else 'complete',
                missing=missing, unknown=unknown)


def add_literature_rna_image(bound, source_chain, source_path):
    """The previously diagnosed 5HO4 B image, still explicitly partial-scope."""
    operators, scale = {}, {}
    for line in source_path.read_text().splitlines():
        if line.startswith('REMARK 290') and 'SMTRY' in line:
            values = line.split()
            operators.setdefault(int(values[3]), np.zeros((3, 4)))[int(values[2][-1]) - 1] = [float(v) for v in values[4:8]]
        elif line.startswith('SCALE'):
            values = line.split(); scale[int(values[0][-1]) - 1] = [float(v) for v in values[1:4]]
    matrix = operators[4]
    lattice = np.linalg.inv(np.array([scale[i] for i in range(3)]))
    shift = np.array([-1, -1, 0])
    image_id = source_chain + '@SMTRY4_cell_-1_-1_0'
    image = copy.deepcopy(bound[0][source_chain]); image.id = image_id
    bound[0].add(image)
    translation = matrix[:, 3] + lattice @ shift
    for atom in image.get_atoms():
        atom.set_coord(atom.coord @ matrix[:, :3].T + translation)
    return image_id, dict(rotation=matrix[:, :3].tolist(), translation_A=translation.tolist(),
                         symmetry_operator=4, unit_cell_shift=shift.tolist(),
                         evidence_url='https://www.nature.com/articles/s41467-017-02770-z',
                         status='Literature-supported contact image for observed-scope audit; not a completed functional assembly review')


def site_geometry(residue, site, protein_residues, rna_residues, instance_map, structure_hash):
    coordinates = np.asarray([a.coord for a in residue])
    sc_indices = [i for i, a in enumerate(residue) if a.name not in BACKBONE]
    contacts, minima, side_minima, missing_classification = [], [], [], False
    origins, counts = set(), {'base': 0, 'ribose': 0, 'phosphate': 0}
    for chain_id, other in rna_residues:
        distances = cdist(coordinates, np.asarray([a.coord for a in other]))
        minimum = float(distances.min()); minima.append(minimum)
        if sc_indices:
            side_minima.append(float(distances[sc_indices, :].min()))
        hits = np.argwhere(distances <= 5)
        if not len(hits):
            continue
        atom_rows = list(other)
        categories = {moiety(other.resname, atom_rows[j].name) for _, j in hits}
        missing_classification |= None in categories
        sc_hit = any(i in sc_indices for i, _ in hits)
        bb_hit = any(i not in sc_indices for i, _ in hits)
        origin = 'both' if sc_hit and bb_hit else 'sidechain' if sc_hit else 'backbone'
        origins.add(origin)
        known_categories = sorted(categories - {None})
        for category in known_categories:
            counts[category] += 1
        rna_id = 'rna_site_' + digest(dict(structure_sha256=structure_hash,
            source_chain=instance_map[chain_id]['source_chain'], position=other.id[1],
            insertion_code=other.id[2].strip(), chem_comp_id=other.resname))[:24]
        contacts.append(dict(rna_instance_id=instance_map[chain_id]['instance_id'], rna_site_id=rna_id,
            chem_comp_id=other.resname, base_identity=other.resname if other.resname in BASE_ATOMS else 'modified',
            moieties=None if None in categories else known_categories,
            protein_contact_origin=origin, min_distance_A=minimum, pairing_state='unknown'))
    contacts.sort(key=lambda c: (c['min_distance_A'], c['rna_instance_id'], c['rna_site_id']))
    neighbors = 0
    for chain_id, other in protein_residues:
        if chain_id == site['chain'] and other.id == residue.id:
            continue
        if cdist(coordinates, np.asarray([a.coord for a in other])).min() <= 5:
            neighbors += 1
    origin = 'none' if not contacts else 'both' if 'both' in origins or len(origins) > 1 else next(iter(origins))
    return dict(min_partner_heavy_distance_A=min(minima) if minima else None,
                min_sidechain_rna_distance_A=min(side_minima) if side_minima else None,
                partner_contact_count_5A=len(contacts) if minima else None,
                rna_contact_count_5A_by_moiety=None if missing_classification else counts,
                rna_moiety_status='partial' if missing_classification else 'complete',
                contact_origin_5A=origin, rna_contacts=contacts,
                protein_neighbor_residue_count_5A=neighbors)


def build(task, review, output, *, check_sampling=True):
    mapping, _ = base_sites(task)
    path = Path(review['structure_path'])
    if file_hash(path) != review['structure_sha256']:
        raise ValueError('Structure source hash mismatch')
    metadata = header(path)
    source = PDBParser(QUIET=True).get_structure('frozen', path)[0]
    groups = task['binding_partners']['groups']
    proteins = next(g['chains'] for g in groups if g['role'] == 'protein')
    rnas = next(g['chains'] for g in groups if g['role'] == 'RNA')
    bound = Structure('reviewed_scope'); bound.add(Model(0))
    conformers, excluded = {}, []
    for chain_id in proteins + rnas:
        chain = Chain(chain_id); bound[0].add(chain)
        for residue in source[chain_id]:
            allowed = (seq1(residue.resname, custom_map={'MSE': 'M'}) in SIDECHAIN if chain_id in proteins
                       else residue.resname in set(metadata['seqres'].get(chain_id, [])))
            if not allowed:
                excluded.append(dict(chain=chain_id, residue_id=list(residue.id), chem_comp_id=residue.resname))
                continue
            copied, alt = coherent_residue(residue)
            if len(copied):
                chain.add(copied)
                conformers[str((chain_id, residue.id))] = alt
    audits = {c: coordinate_audit(bound[0][c], metadata['seqres'].get(c, [])) for c in rnas}
    coordinate_status = ('partial' if any(a['status'] == 'partial' for a in audits.values()) else
                         'unknown' if any(a['status'] == 'unknown' for a in audits.values()) else 'complete')
    instance_map = {c: dict(instance_id=c + '@deposited', source_chain=c, origin='deposited',
                           polymer_length_nt=len(metadata['seqres'].get(c, [])) or None,
                           observed_nt_count=len(bound[0][c])) for c in rnas}
    transformations = {}
    if task['reference_complex_id'] == '5HO4':
        image_id, transform = add_literature_rna_image(bound, rnas[0], path)
        transformations[image_id] = transform
        instance_map[image_id] = dict(instance_map[rnas[0]], instance_id=image_id,
                                     origin='literature_supported_symmetry')
    context = dict(structure_id=task['reference_complex_id'], provenance_ref=task['group_id'],
        partner_definition_id=review['partner_definition_id'], interface_scope=review['interface_scope'],
        interface_review_status=review['interface_review_status'], interface_coverage=review['interface_coverage'],
        partner_coordinate_status=coordinate_status, assay_rna_match=review['assay_rna_match'],
        assay_protein_match=review['assay_protein_match'], rna_architecture=review['rna_architecture'],
        structure_method=metadata['structure_method'], resolution_A=metadata['resolution_A'],
        model_selection='First deposited model; no ensemble averaging or atom completion',
        source_description='Frozen WT source; coherent heavy-atom conformers. SASA environment: specified protein/RNA polymers, including only explicitly recorded RNA images; solvent, ions and other ligands excluded. ' + review['scope_description'],
        partner_instances=list(instance_map.values()))
    scope_ok = (context['interface_review_status'] == 'verified' and context['interface_coverage'] == 'complete'
                and coordinate_status == 'complete' and context['assay_rna_match'] == 'matched'
                and context['assay_protein_match'] in ('matched', 'compatible_domain'))
    rna_residues = [(c, r) for c in instance_map for r in bound[0][c]]
    protein_residues = [(c, r) for c in proteins for r in bound[0][c]]
    ShrakeRupley(probe_radius=1.4, n_points=100).compute(bound, level='R')
    without = copy.deepcopy(bound)
    for chain_id in instance_map:
        without[0].detach_child(chain_id)
    ShrakeRupley(probe_radius=1.4, n_points=100).compute(without, level='R')
    rows, observed, identities = [], {}, {}
    for sid, site in mapping.items():
        matches = [r for r in bound[0][site['chain']]
                   if r.id[1] == site['position'] and r.id[2].strip() == (site.get('insertion_code') or '')]
        if len(matches) > 1:
            raise ValueError('Ambiguous protein site identity')
        residue = matches[0] if matches else None
        names = set(residue.child_dict) if residue is not None else set()
        if residue is not None and residue.resname == 'MSE':
            names = (names - {'SE'}) | {'SD'}
        sc_status = ('not_applicable' if site['wt'] == 'G' else 'missing' if residue is None else
                     'complete' if SIDECHAIN[site['wt']] <= names else 'partial')
        needed = {'N', 'CA', 'C', 'O'} | SIDECHAIN[site['wt']]
        site_status = ('missing' if residue is None else 'WT_mismatch' if seq1(residue.resname, custom_map={'MSE': 'M'}) != site['wt'] else
                       'complete' if needed <= names else 'partial')
        geometry = site_geometry(residue, site, protein_residues, rna_residues, instance_map,
                                 review['structure_sha256']) if residue is not None else {}
        if residue is not None and site_status == 'complete':
            isolated = without[0][site['chain']][residue.id]
            a, b = float(residue.sasa), float(isolated.sasa)
            if b < a - 1e-6:
                raise ValueError('Removing RNA decreased SASA')
            geometry.update(sasa_bound_A2=a, sasa_without_rna_A2=b,
                            delta_sasa_partner_A2=max(0.0, b - a), rna_burial_fraction=(b - a) / b if b else None)
        observed[sid] = dict(geometry, coordinate_status=site_status, sidechain_status=sc_status)
        identities[sid] = dict(site, expected_WT_heavy_atoms=sorted(needed), missing_WT_heavy_atoms=sorted(needed - names))
        row = {k: None for k in RNA_NULL + SASA_NULL}
        status = 'unavailable_scope' if not scope_ok else 'unavailable_site' if site_status != 'complete' else 'computed'
        row.update(site_id=sid, coordinate_status=site_status, sidechain_status=sc_status,
                   interface_geometry_status=status, sasa_status=status, rna_moiety_status='unavailable',
                   contact_origin_5A='unknown', rna_contact_list_truncated=False,
                   protein_neighbor_residue_count_5A=geometry.get('protein_neighbor_residue_count_5A'),
                   protein_environment_status='observed' if residue is not None and site_status in ('complete', 'partial') else 'unavailable')
        if row['protein_environment_status'] != 'observed':
            row['protein_neighbor_residue_count_5A'] = None
        if status == 'computed':
            row.update({k: geometry[k] for k in RNA_NULL + SASA_NULL})
            row.update(rna_moiety_status=geometry['rna_moiety_status'], contact_origin_5A=geometry['contact_origin_5A'],
                       rna_contacts=geometry['rna_contacts'][:6], rna_contact_list_truncated=len(geometry['rna_contacts']) > 6)
        rows.append(row)
    sampling = {}
    if check_sampling and scope_ok:
        dense_bound, dense_without = copy.deepcopy(bound), copy.deepcopy(without)
        for value in (dense_bound, dense_without):
            ShrakeRupley(probe_radius=1.4, n_points=960).compute(value, level='R')
        for sid, site in mapping.items():
            if observed[sid]['coordinate_status'] != 'complete':
                continue
            residue = next(r for r in dense_bound[0][site['chain']] if r.id[1] == site['position'] and r.id[2].strip() == (site.get('insertion_code') or ''))
            unbound = dense_without[0][site['chain']][residue.id]
            dense_delta = float(unbound.sasa - residue.sasa)
            sampling[sid] = dict(delta_sasa_100_A2=observed[sid]['delta_sasa_partner_A2'],
                                 delta_sasa_960_A2=dense_delta,
                                 absolute_difference_A2=abs(dense_delta - observed[sid]['delta_sasa_partner_A2']))
    prompt = dict(schema_version=VERSION, units=dict(distance='angstrom', area='angstrom^2'), context=context, sites=rows)
    cache = dict(structure_path=str(path), structure_sha256=review['structure_sha256'], review=review,
        source_header_evidence=metadata['evidence'], partner_coordinate_audit=audits,
        instance_mapping=instance_map, transformations=transformations,
        base_site_mapping=identities, chosen_altloc_by_residue=conformers,
        excluded_components=excluded, excluded_extra_chains=sorted(set(source.child_dict) - set(proteins + rnas)),
        observed_scope_geometry=observed,
        full_rna_identity_table=[dict(instance_id=instance_map[c]['instance_id'], source_chain=instance_map[c]['source_chain'],
                                      rna_site_id='rna_site_' + digest(dict(structure_sha256=review['structure_sha256'],
                                          source_chain=instance_map[c]['source_chain'], position=r.id[1],
                                          insertion_code=r.id[2].strip(), chem_comp_id=r.resname))[:24],
                                      position=r.id[1], insertion_code=r.id[2].strip(), chem_comp_id=r.resname,
                                      heavy_atom_names=[a.name for a in r],
                                      heavy_atoms=[dict(name=a.name, element=a.element,
                                          coordinate_A=[float(v) for v in a.coord], moiety=moiety(r.resname, a.name)) for a in r])
                                  for c, r in rna_residues],
        numerical_protocol=dict(probe_radius_A=1.4, n_points=100, sensitivity_n_points=960,
                                selection='Preserve original 100-point SASA; denser values are audit only, no label-based parameter choice',
                                precision='Full finite Python float JSON; no prompt-only numerical rounding'),
        sasa_sensitivity=sampling, optional_interactions='Not computed; no H-bond, stacking or water counts invented',
        scope_eligible=scope_ok, candidate_generation=False, evaluation_labels_used=False)
    return prompt, cache
