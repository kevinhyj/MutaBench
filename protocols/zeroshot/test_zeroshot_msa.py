import copy, unittest
import numpy as np
from zeroshot_msa import alignment, weights_for, POLICY_VERSION
from prepare_shared_prompts import feature_definition, record_from_sources
from test_shared_protocol import complex_sources

class MSARegression(unittest.TestCase):
    def test_lexical_cap_not_file_order_and_exclude_wt_ambiguous(self):
        raw=['ACD','CCD','-CD','ACD','XCD','CCD','DCD']
        a,pos,meta=alignment(raw,cap=2)
        self.assertEqual(meta['unique_valid_pool_count'],3)
        self.assertEqual(a.tolist(),[[-1,1,2],[1,1,2]])
        self.assertEqual(pos.tolist(),[1,2,3])
        b,_,_=alignment([raw[0],*reversed(raw[1:])],cap=2)
        np.testing.assert_array_equal(a,b)
    def test_lowercase_query_insert_is_not_supported_column(self):
        a,pos,_=alignment(['AaCD','-CD','CCD'])
        self.assertEqual(pos.tolist(),[1,3,4])
    def test_weight_counts_and_overlap(self):
        a=np.array([[0,0,0,0],[0,0,0,0],[1,1,1,1],[-1,-1,-1,0]])
        np.testing.assert_array_equal(weights_for(a),[.5,.5,1,1])
    def test_new_definitions_describe_null_and_lexical(self):
        for domain in ['protein','protein_protein']:
            d=feature_definition(domain,'msa');self.assertNotIn('historical_exception',d)
            self.assertNotIn('This historical exception is present',str(d))
            self.assertIn('null',str(d))
        d=feature_definition('protein_rna','msa')
        self.assertIn('lexicographically',d['computation']['selection'])
        self.assertNotIn('SHA256',d['computation']['selection'])
    def test_old_rna_policy_cannot_be_relabelled(self):
        src=complex_sources('protein_rna');src['msa']['msa_scoring_policy']['version']='protein-rna-msa-sha256-cap-v2'
        with self.assertRaisesRegex(ValueError,'recomputed'):record_from_sources('protein_rna','synthetic',src)
if __name__=='__main__':unittest.main()
