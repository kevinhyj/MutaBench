"""Synthetic checks for mutation identity, RNA chemistry and WT geometry."""
import copy
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from Bio.PDB.Atom import Atom
from Bio.PDB.Residue import Residue

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools/offline_ablation'))
import structure_v3_features as v3
from evagent.offline_ablation import base_sites, render_evidence
from test_offline_ablation_features import FrozenFeatureTests


def residue(name, atoms):
    row = Residue((' ', 1, ' '), name, ' ')
    for i, (atom, xyz, element) in enumerate(atoms, 1):
        row.add(Atom(atom, np.array(xyz, dtype=float), 10., 1., ' ', atom, i, element=element))
    return row


class StructureV3Tests(unittest.TestCase):
    def test_same_site_different_substitutions_have_different_chemistry(self):
        self.assertNotEqual(v3.substitution('A', 'D'), v3.substitution('A', 'C'))
        self.assertEqual(v3.substitution('K', 'E')['mutant_minus_wt']['nominal_charge'], -2)
        self.assertIsNone(v3.substitution('H', 'K')['mutant_minus_wt']['nominal_charge'])
        self.assertEqual(v3.properties('G')['sidechain_heavy_atom_count'], 0)
        self.assertEqual(v3.properties('W')['sidechain_heavy_atom_count'], 10)
        with self.assertRaises(ValueError):
            v3.properties('X')

    def test_modified_base_is_contact_classified_but_not_chemical_typed(self):
        row = residue('PSU', [('N1', (0, 0, 0), 'N'), ("O2'", (1, 0, 0), 'O'), ('OP1', (2, 0, 0), 'O')])
        self.assertEqual(v3.component(row, row['N1']), 'base')
        self.assertFalse(v3.rna_roles(row, row['N1'])[2])
        self.assertEqual(v3.rna_roles(row, row["O2'"]), (True, True, True))
        self.assertEqual(v3.component(row, row['OP1']), 'phosphate')

    def test_contact_components_and_nominal_ionic_geometry(self):
        site = residue('LYS', [('NZ', (0, 0, 0), 'N')])
        partner = residue('A', [('OP1', (3, 0, 0), 'O'), ("O2'", (0, 3, 0), 'O'), ('N1', (0, 0, 3), 'N')])
        result = v3.environment(site, 'K', [(('B', 1, ''), partner)])
        profile = result['rna_contact_profile']
        self.assertEqual(profile['component_residue_counts_5A'], dict(base=1, ribose=1, phosphate=1, unknown=0))
        self.assertEqual(result['wt_interaction_profile']['nominal_ionic_contact_pairs_4A'], 1)
        self.assertEqual(result['wt_interaction_profile']['donor_acceptor_proximity_pairs_3_5A']['sidechain'], 3)
        self.assertIn('not_confirmed', result['wt_interaction_profile']['status'])

    def test_ring_geometry_rejects_missing_nonplanar_and_lateral_rings(self):
        names = ('a', 'b', 'c', 'd', 'e', 'f')
        def ring(z, offset=0):
            return residue('SYN', [(name, (np.cos(i*np.pi/3)+offset, np.sin(i*np.pi/3), z), 'C')
                                   for i, name in enumerate(names)])
        first = v3.ring_plane(ring(0), names)
        self.assertTrue(v3.stacking_candidate(first, v3.ring_plane(ring(3.4), names)))
        self.assertFalse(v3.stacking_candidate(first, v3.ring_plane(ring(3.4, 3), names)))
        self.assertIsNone(v3.ring_plane(ring(0), names + ('missing',)))
        bad = ring(0); bad['a'].coord[2] = 2
        self.assertIsNone(v3.ring_plane(bad, names))

    def test_translation_does_not_change_contact_descriptors(self):
        a = residue('LYS', [('NZ', (0, 0, 0), 'N')])
        b = residue('A', [('OP1', (3, 0, 0), 'O')])
        before = v3.environment(a, 'K', [(('B', 1, ''), b)])
        for row in (a, b):
            for atom in row:
                atom.coord += np.array([13, -4, 7])
        self.assertEqual(before, v3.environment(a, 'K', [(('B', 1, ''), b)]))

    def test_shared_sites_exact_msa_reuse_and_designated_partner_only(self):
        with tempfile.TemporaryDirectory() as directory:
            builder, task, observed, partners = FrozenFeatureTests().fixture(Path(directory))
            original_task = copy.deepcopy(task)
            features, _ = builder.build(task, observed, partners)
            old_msa = render_evidence(task, ['one', 'two'], features, 'msa_prompt')
            features['structure_v3'] = v3.build(task, features, builder.geometry(observed, partners))
            self.assertEqual(task, original_task)
            output = v3.render(task, ['one', 'two'], features, 'structure', render_evidence)
            self.assertEqual(len(output['structure_wt_sites']), 1)
            self.assertNotIn('structure_sites', output)
            self.assertEqual(output['candidate_substitutions']['one'][0]['site_id'],
                             output['candidate_substitutions']['two'][0]['site_id'])
            self.assertEqual(old_msa, v3.render(task, ['one', 'two'], features, 'msa_prompt', render_evidence))
            site = next(iter(output['structure_wt_sites'].values()))
            self.assertEqual(site['partner_chain_ids'], ['B'])
            self.assertEqual(site['rna_contact_profile']['component_residue_counts_5A']['phosphate'], 1)
            self.assertEqual(site['rna_contact_profile']['component_residue_counts_5A']['unknown'], 0)


if __name__ == '__main__':
    unittest.main()
