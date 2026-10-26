"""Regression tests for confidence loss, atom mapping, and actual request contents."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from rna_confidence import (STRUCTURE_MARKER, cif_confidence, pdb_atoms,
                            rebuild_structure_unit, repair_packet, sha)


class RNAConfidenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.pdb = self.root / 'mapped_WT.pdb'
        # PDB residue numbering deliberately differs from sequence positions.
        self.pdb.write_text(''.join(
            f"ATOM  {i:5d}  C4'   {base} A{pos:4d}{ins}   {float(i):8.3f}{0.:8.3f}{0.:8.3f}{1.:6.2f}{0.:6.2f}           C\n"
            for i, (base, pos, ins) in enumerate([('A', 10, ' '), ('C', 10, 'A')], 1)))
        self.cif = self.root / 'sample_model.cif'
        fields = ['group_PDB', 'id', 'label_asym_id', 'label_seq_id', 'label_comp_id',
                  'label_atom_id', 'pdbx_PDB_ins_code', 'label_alt_id', 'Cartn_x',
                  'Cartn_y', 'Cartn_z', 'B_iso_or_equiv', 'pdbx_PDB_model_num']
        self.cif.write_text('data_test\n_software.name AlphaFold\nloop_\n' +
                           '\n'.join('_atom_site.' + f for f in fields) + '\n' +
                           'ATOM 1 A 10 A "C4\'" ? . 1 0 0 82.25 1\n' +
                           'ATOM 2 A 10 C "C4\'" A . 2 0 0 63.75 1\n#\n')
        self.item = {'assay': 'synthetic', 'structure_kind': 'predicted_WT',
                     'structure_method': 'AF3 WT prediction, synthetic fixture',
                     'structure': {'path': str(self.pdb), 'sha256': sha(self.pdb)}}
        self.query = {'assay': 'synthetic', 'wt_sequence': 'AC',
                      'candidates': [{'id': 'M001', 'changed_positions': [1, 2]}]}
        self.packet = {'assay': 'synthetic', 'status': 'available', 'kind': 'predicted_WT',
                       'source': copy.deepcopy(self.item['structure']), 'definition': 'WT geometry.',
                       'site_evidence': {'M001': [
                           {'position': i, 'anchor_atom': "C4'", 'plddt': 0.,
                            'nonlocal_neighbors_8a': 0, 'nearest_nonlocal_positions': []}
                           for i in [1, 2]]}}

    def repair(self, **kwargs):
        return repair_packet(self.query, self.item, self.packet, **kwargs)

    def assert_unavailable(self, reason):
        result = self.repair()
        self.assertEqual(result['confidence_provenance']['status'], 'unavailable')
        self.assertIn(reason, result['confidence_provenance']['reason'])
        self.assertTrue(all(s['plddt'] is None for s in result['site_evidence']['M001']))

    def test_restores_anchor_confidence_and_pdb_without_changing_geometry(self):
        original = self.pdb.read_bytes()
        target = self.root / 'new/mapped_WT.pdb'
        result = self.repair(pdb_output=target)
        self.assertEqual([s['plddt'] for s in result['site_evidence']['M001']], [82.25, 63.75])
        self.assertEqual(self.pdb.read_bytes(), original)
        for before, after in zip(self.pdb.read_text().splitlines(), target.read_text().splitlines()):
            self.assertEqual(before[:60] + before[66:], after[:60] + after[66:])
        self.assertEqual(result['source']['sha256'], sha(target))
        self.assertEqual(self.packet['site_evidence']['M001'][0]['plddt'], 0.)
        with self.assertRaises(FileExistsError):
            self.repair(pdb_output=target)

    def test_missing_cif_does_not_fall_back_to_pdb_bfactor(self):
        self.cif.unlink()
        self.assert_unavailable('missing_or_ambiguous_source_cif')

    def test_coordinate_mismatch_is_unavailable(self):
        self.cif.write_text(self.cif.read_text().replace('1 0 0 82.25', '9 0 0 82.25'))
        self.assert_unavailable('coordinate_mismatch')

    def test_wrong_chain_or_insertion_code_is_unavailable(self):
        self.cif.write_text(self.cif.read_text().replace('ATOM 2 A 10', 'ATOM 2 B 10'))
        self.assert_unavailable('unmapped_pdb_atom')

    def test_invalid_confidence_and_all_zero_source_are_unavailable(self):
        original = self.cif.read_text()
        for value, reason in [('nan', 'invalid_cif'), ('101', 'invalid_cif'),
                              ('-1', 'invalid_cif'), ('0', 'all_zero_source')]:
            with self.subTest(value=value):
                self.cif.write_text(original.replace('82.25', value).replace('63.75', value))
                self.assert_unavailable(reason)

    def test_real_zero_confidence_is_retained_if_other_atoms_are_valid(self):
        self.cif.write_text(self.cif.read_text().replace('82.25', '0'))
        self.assertEqual(self.repair()['site_evidence']['M001'][0]['plddt'], 0.)

    def test_experimental_bfactor_is_never_confidence(self):
        self.item['structure_kind'] = 'experimental'
        self.packet['kind'] = 'experimental'
        self.assert_unavailable('experimental_b_factor_is_not_plddt')

    def test_snapshot_cif_hash_is_enforced(self):
        snapshot = {'sources': {'structure': {'provenance': [
            {'id': 'tertiary', 'tool': 'AlphaFold3', 'artifact': self.cif.name, 'sha256': 'bad'}]}}}
        (self.root / 'snapshot.json').write_text(json.dumps(snapshot))
        self.assert_unavailable('cif_provenance_hash_mismatch')

    def test_unverified_cif_software_is_unavailable(self):
        self.cif.write_text(self.cif.read_text().replace('AlphaFold', 'unknown'))
        self.assert_unavailable('cif_not_alphafold')

    def test_duplicate_atoms_and_multiple_models_are_rejected(self):
        original = self.cif.read_text()
        self.cif.write_text(original.replace('#\n', 'ATOM 3 A 10 A "C4\'" ? . 1 0 0 82.25 1\n#\n'))
        self.assert_unavailable('duplicate_cif_atom')
        self.cif.write_text(original.replace('63.75 1', '63.75 2'))
        self.assert_unavailable('ambiguous_cif_models')

    def test_bad_geometry_source_hash_is_fatal(self):
        self.pdb.write_text(self.pdb.read_text() + 'END\n')
        with self.assertRaisesRegex(ValueError, 'pdb_source_hash_mismatch'):
            self.repair()

    def test_request_uses_new_values_and_identity_preserving_other_sections(self):
        block = {k: self.packet[k] for k in ['kind', 'definition', 'site_evidence']}
        unit = {'domain': 'rna', 'assay': 'synthetic', 'arm': 'sequence_structure',
                'chunk_index': 0, 'logical_id': 'old',
                'template': {'system': 'system', 'expected_ids': ['M001'],
                             'user': 'unchanged context\n\n' + STRUCTURE_MARKER + json.dumps(block) + '\n\nunchanged output'}}
        result = rebuild_structure_unit(unit, self.repair())
        self.assertTrue(result['template']['user'].startswith('unchanged context\n\n'))
        self.assertTrue(result['template']['user'].endswith('\n\nunchanged output'))
        new_block = json.loads(result['template']['user'].split(STRUCTURE_MARKER)[1].split('\n\n')[0])
        self.assertEqual(new_block['site_evidence']['M001'][0]['plddt'], 82.25)
        self.assertEqual(result['template']['expected_ids'], ['M001'])
        identity = {k: v for k, v in result.items() if k != 'logical_id'}
        payload = json.dumps(identity, ensure_ascii=False, sort_keys=True, separators=(',', ':')) + '\n'
        self.assertEqual(result['logical_id'], hashlib.sha256(payload.encode()).hexdigest())
        self.assertEqual(unit['logical_id'], 'old')


if __name__ == '__main__':
    unittest.main()
