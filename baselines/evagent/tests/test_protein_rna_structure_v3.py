"""Independent physical-identity and missing-coordinate regression cases."""
import importlib.util
from pathlib import Path
import sys

import numpy as np
from Bio.PDB.Atom import Atom
from Bio.PDB.Residue import Residue

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(CODE / 'tools/offline_ablation'))
import structure_v3 as v3


def residue(name, number, atoms):
    result = Residue((' ', number, ' '), name, '')
    for index, (atom_name, xyz) in enumerate(atoms):
        result.add(Atom(atom_name, np.asarray(xyz, dtype=float), 0.0, 1.0, ' ',
                        atom_name.ljust(4), index + 1, element=atom_name[0]))
    return result


def test_contacts_are_nucleotides_and_moiety_union_can_overlap():
    protein = residue('ALA', 1, [('CA', (0, 0, 0)), ('CB', (0.1, 0, 0))])
    rna = residue('A', 2, [('N6', (3, 0, 0)), ("O2'", (4, 0, 0)), ('OP1', (5, 0, 0))])
    result = v3.site_geometry(protein, {'chain': 'A'}, [('A', protein)], [('B', rna)],
                              {'B': {'instance_id': 'B@deposited', 'source_chain': 'B'}}, 'hash')
    assert result['partner_contact_count_5A'] == 1
    assert result['rna_contact_count_5A_by_moiety'] == {'base': 1, 'ribose': 1, 'phosphate': 1}
    assert result['contact_origin_5A'] == 'both'
    assert result['protein_neighbor_residue_count_5A'] == 0
    assert result['min_sidechain_rna_distance_A'] == 2.9


def test_physical_rna_images_not_deduplicated_by_source_residue():
    protein = residue('GLY', 1, [('CA', (0, 0, 0))])
    rna = residue('A', 2, [('N6', (3, 0, 0))])
    instances = {key: {'instance_id': key, 'source_chain': 'B'} for key in ('B', 'B_image')}
    result = v3.site_geometry(protein, {'chain': 'A'}, [('A', protein)],
                              [('B', rna), ('B_image', rna)], instances, 'hash')
    assert result['partner_contact_count_5A'] == 2
    assert result['rna_contacts'][0]['rna_site_id'] == result['rna_contacts'][1]['rna_site_id']
    assert result['rna_contacts'][0]['rna_instance_id'] != result['rna_contacts'][1]['rna_instance_id']
    assert result['min_sidechain_rna_distance_A'] is None
    assert result['contact_origin_5A'] == 'backbone'


def test_unknown_modified_rna_chemistry_not_assigned_canonical_counts():
    protein = residue('GLY', 1, [('CA', (0, 0, 0))])
    rna = residue('PSU', 2, [('N1', (3, 0, 0))])
    result = v3.site_geometry(protein, {'chain': 'A'}, [('A', protein)], [('B', rna)],
                              {'B': {'instance_id': 'B', 'source_chain': 'B'}}, 'hash')
    assert result['partner_contact_count_5A'] == 1
    assert result['rna_contact_count_5A_by_moiety'] is None
    assert result['rna_moiety_status'] == 'partial'
    assert result['rna_contacts'][0]['base_identity'] == 'modified'


def test_missing_base_atoms_and_polymer_residues_mark_partial():
    rna = residue('A', 1, [('N6', (0, 0, 0))])
    result = v3.coordinate_audit([rna], ['A', 'G'])
    assert result['status'] == 'partial'
    assert any(x.get('reason') == 'polymer_residue_count_mismatch' for x in result['missing'])
    assert any("O2'" in x.get('absent_atoms', []) for x in result['missing'])


def test_atom_name_aliases_and_ribose_2OH():
    assert v3.moiety('A', 'O1P') == 'phosphate'
    assert v3.moiety('A', 'O2*') == 'ribose'
    assert v3.moiety('A', 'N6') == 'base'
    assert v3.moiety('PSU', 'O2') is None


def test_real_5HO4_image_matches_frozen_independent_diagnostic():
    root = CODE.parent
    path = root / 'data/final_mutation_data/nabe/source/structures/nabe_pdbs/5HO4.pdb'
    source = v3.PDBParser(QUIET=True).get_structure('test', path)
    before = np.array([a.coord for a in source[0]['B'].get_atoms()])
    image, transform = v3.add_literature_rna_image(source, 'B', path)
    assert image != 'B'
    assert transform['symmetry_operator'] == 4
    assert transform['unit_cell_shift'] == [-1, -1, 0]
    after = np.array([a.coord for a in source[0][image].get_atoms()])
    expected = before @ np.array(transform['rotation']).T + np.array(transform['translation_A'])
    np.testing.assert_allclose(after, expected, atol=1e-8)
    residue113 = source[0]['A'][113]
    protein = np.array([a.coord for a in residue113.get_atoms() if a.element not in ('H', 'D')])
    rna = np.array([a.coord for r in source[0][image] if r.resname in v3.BASE_ATOMS
                    for a in r.get_atoms() if a.element not in ('H', 'D')])
    minimum = np.linalg.norm(protein[:, None, :] - rna[None, :, :], axis=2).min()
    assert abs(minimum - 2.548306694288226) < 1e-6
