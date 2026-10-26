"""Offline contract checks with synthetic inputs, no model or real labels."""
import unittest
from sampling import complex_groups, digest, singlechain_select


class SamplingChecks(unittest.TestCase):
    def records(self):
        return [dict(source="demo",assay_group_id="assay",reference_complex_id="WT",fact_id=f"f{i}",
                     chain_mutations=[dict(chain="A",position=i,wt_aa="A",mutant_aa="V")]) for i in range(1,9)]

    def test_hash_matches_historical_formula_and_input_reordering(self):
        rows=self.records()
        a=complex_groups(rows,4);b=complex_groups(list(reversed(rows)),4)
        self.assertEqual(a,b)
        for g in a:self.assertEqual(g["priority"],digest([42,digest(g["sites"])]))

    def test_duplicate_measurements_do_not_increase_variant_count(self):
        rows=self.records(); extra=dict(rows[0],fact_id="replicate");rows.append(extra)
        a=complex_groups(rows,8)
        self.assertEqual(len(a),8)
        self.assertEqual(sum(g["measurement_resolution_required"] for g in a),1)
        with self.assertRaises(ValueError):complex_groups(rows,9)

    def test_scope_mixing_is_rejected(self):
        rows=self.records();rows[0]["assay_group_id"]="other"
        with self.assertRaises(ValueError):complex_groups(rows,4)

    def test_insertion_codes_distinguish_variants(self):
        rows=self.records()[:1];r=dict(rows[0],fact_id="insert",chain_mutations=[dict(chain="A",position=1,insertion_code="A",wt_aa="A",mutant_aa="V")]);rows.append(r)
        self.assertEqual(len(complex_groups(rows,2)),2)

    def test_singlechain_matches_randomstate_and_stable_sort(self):
        import numpy as np
        rows=[dict(mutation=f"A{i}V",sequence="A"*i+"V") for i in range(1,9)]
        expected=[rows[int(i)] for i in np.random.RandomState(42).permutation(8)[:4]]
        self.assertEqual(singlechain_select(rows,4),expected)
        self.assertEqual(singlechain_select(list(reversed(rows)),4),expected)
        self.assertEqual(len(singlechain_select(rows,250)),8)


if __name__=="__main__":unittest.main()
