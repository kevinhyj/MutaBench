"""Reject stale or forged confidence at the current RNA preparation boundary."""
import copy
import json
import unittest

import test_rna_confidence as fixtures
from rna_confidence import repair_packet, sha
from rna_structure_guard import check_packet, check_prompt


class RNAEntryGuardTests(unittest.TestCase):
    def setUp(self):
        self.f=fixtures.RNAConfidenceTests()
        self.f.setUp()
        self.addCleanup(self.f.doCleanups)
        self.f.query['domain']='rna'

    def repaired(self):
        f=self.f
        return repair_packet(f.query,f.item,f.packet,pdb_output=f.root/'derived.pdb')

    def check(self,packet):
        f=self.f
        return check_packet(f.query,f.item,f.packet,packet)

    def test_verified_source_and_individual_true_zero(self):
        self.f.cif.write_text(self.f.cif.read_text().replace('82.25','0'))
        packet=self.repaired()
        self.assertEqual(packet['site_evidence']['M001'][0]['plddt'],0.)
        self.assertEqual(self.check(packet)['verified_atoms'],2)

    def test_stale_exported_zero_is_rejected(self):
        packet=self.repaired()
        packet['site_evidence']['M001'][0]['plddt']=0.
        with self.assertRaisesRegex(ValueError,'confidence/geometry'):
            self.check(packet)

    def test_experimental_bfactor_remains_separate_and_forgery_fails(self):
        f=self.f
        f.pdb.write_text('\n'.join(line[:60]+' 42.00'+line[66:]
                                  for line in f.pdb.read_text().splitlines())+'\n')
        f.item['structure']['sha256']=sha(f.pdb)
        f.packet['source']=copy.deepcopy(f.item['structure'])
        f.item['structure_kind']=f.packet['kind']='experimental_WT'
        packet=self.repaired()
        self.assertEqual(self.check(packet)['confidence_status'],'unavailable')
        self.assertIsNone(packet['site_evidence']['M001'][0]['plddt'])
        self.assertEqual(float(f.pdb.read_text().splitlines()[0][60:66]),42.)
        packet['confidence_provenance']['status']='verified'
        packet['site_evidence']['M001'][0]['plddt']=42.
        with self.assertRaisesRegex(ValueError,'provenance'):
            self.check(packet)

    def test_missing_cif_cannot_be_replaced_by_verified_flag(self):
        self.f.cif.unlink()
        packet=self.repaired()
        self.assertEqual(self.check(packet)['confidence_status'],'unavailable')
        packet['confidence_provenance']['status']='verified'
        with self.assertRaisesRegex(ValueError,'provenance'):
            self.check(packet)

    def test_tampered_derived_source_hash_is_rejected(self):
        packet=self.repaired()
        packet['source']['sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'hash mismatch'):
            self.check(packet)

    def test_rehashed_coordinate_change_is_rejected(self):
        packet=self.repaired();path=self.f.root/'derived.pdb'
        lines=path.read_text().splitlines()
        lines[0]=lines[0][:30]+f'{99.:8.3f}'+lines[0][38:]
        path.write_text('\n'.join(lines)+'\n');packet['source']['sha256']=sha(path)
        with self.assertRaisesRegex(ValueError,'geometry'):
            self.check(packet)

    def test_rehashed_derived_confidence_change_is_rejected(self):
        packet=self.repaired();path=self.f.root/'derived.pdb'
        lines=path.read_text().splitlines()
        lines[0]=lines[0][:60]+'  0.00'+lines[0][66:]
        path.write_text('\n'.join(lines)+'\n');packet['source']['sha256']=sha(path)
        with self.assertRaisesRegex(ValueError,'original CIF'):
            self.check(packet)

    def test_actual_prompt_checks_source_value_reason_and_order(self):
        packet=self.repaired();provenance=packet['confidence_provenance']
        block=dict(source_kind=packet['kind'],confidence_status=provenance['status'],
                   confidence_unavailable_reason=provenance['reason'],variants=[dict(variant_id='M001',sites=[
                       dict(sequence_index_1based=s['position'],features={k:v for k,v in s.items()
                            if k not in {'position','wt','mutant'}}) for s in packet['site_evidence']['M001']])])
        def unit(b):
            return dict(template=dict(user=json.dumps(dict(evidence=dict(structure=b))),expected_ids=['M001']))
        self.assertEqual(check_prompt(unit(block),packet)['site_occurrences'],2)
        for field,value in [('source_kind','experimental_WT'),('confidence_status','unverified'),
                            ('confidence_unavailable_reason','invented')]:
            with self.subTest(field=field):
                bad=copy.deepcopy(block);bad[field]=value
                with self.assertRaises(ValueError):check_prompt(unit(bad),packet)
        for mode in ('value','position','id'):
            with self.subTest(mode=mode):
                bad=copy.deepcopy(block)
                if mode=='value':bad['variants'][0]['sites'][0]['features']['plddt']=0.
                elif mode=='position':bad['variants'][0]['sites'].reverse()
                else:bad['variants'][0]['variant_id']='M002'
                with self.assertRaises(ValueError):check_prompt(unit(bad),packet)

    def test_unavailable_structure_cannot_contain_numeric_confidence(self):
        packet=dict(status='unavailable',reason='not_prepared')
        site=dict(features=dict(status='unavailable'))
        block=dict(source_kind='unavailable',confidence_status='unverified',
                   confidence_unavailable_reason=None,variants=[dict(variant_id='M001',sites=[site])])
        unit=dict(template=dict(expected_ids=['M001']))
        unit['template']['user']=json.dumps(dict(evidence=dict(structure=block)))
        self.assertEqual(check_prompt(unit,packet)['unavailable_structure_sites'],1)
        site['features']['plddt']=0.
        unit['template']['user']=json.dumps(dict(evidence=dict(structure=block)))
        with self.assertRaises(ValueError):check_prompt(unit,packet)


if __name__=='__main__':unittest.main()
