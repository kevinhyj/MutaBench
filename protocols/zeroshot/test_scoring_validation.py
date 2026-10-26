import json
import unittest

from multillm_transport import parse
from multillm_campaign import configured_models
from scoring_validation import ScoringResponseValidator


class OpaqueIDTests(unittest.TestCase):
    def test_hash_ids_are_exact_and_scores_unchanged(self):
        ids=['v_0123456789abcdef0123','v_abcdef0123456789abcd']
        text=json.dumps({'variants':[{'variant_id':x,'predicted_score':v} for x,v in zip(ids,[-.2,.5])]})
        r=ScoringResponseValidator(ids).validate(text)
        self.assertTrue(r.valid);self.assertEqual(r.value,dict(zip(ids,[-.2,.5])))
        for invalid in [ids[0].upper(),'M001',' '+ids[0],ids[0][:-1]]:
            self.assertFalse(ScoringResponseValidator(ids).validate(text.replace(ids[0],invalid)).valid)

    def test_duplicate_missing_nonfinite_boolean(self):
        ids=['v_abcdef'];v=ScoringResponseValidator(ids)
        good={'variant_id':ids[0],'predicted_score':0}
        self.assertFalse(v.validate(json.dumps({'variants':[good,good]})).valid)
        for score in [True,float('nan'),float('inf'),1.01,'0']:
            self.assertFalse(v.validate(json.dumps({'variants':[dict(good,predicted_score=score)]})).valid)
        self.assertFalse(v.validate(json.dumps({'variants':[]})).valid)

    def test_cosmetic_keys_and_metadata_do_not_change_content(self):
        v=ScoringResponseValidator(['v_abcdef'])
        obj={' variants ': [{' variant_id':'v_abcdef','predicted_score ':0.125,'explanation':'optional'}], 'metadata': {'unused':True}}
        report=v.validate('```json\n'+json.dumps(obj)+'\n```')
        self.assertTrue(report.valid);self.assertEqual(report.value,{'v_abcdef':0.125})

    def test_ambiguous_aliases_and_duplicate_json_keys_are_rejected(self):
        v=ScoringResponseValidator(['v_abcdef'])
        good={'variant_id':'v_abcdef','predicted_score':0.1}
        for row in [dict(good,**{' variant_id':'other'}),dict(good,**{'predicted_score ':0.9})]:
            self.assertFalse(v.validate(json.dumps({'variants':[row]})).valid)
        self.assertTrue(v.validate(json.dumps({'variants':[dict(good,**{' variant_id':'v_abcdef'})]})).valid)
        self.assertFalse(v.validate('{"variants":[{"variant_id":"v_abcdef","predicted_score":0.1,"predicted_score":0.9}]}').valid)
        self.assertFalse(v.validate(json.dumps({'variants':[good],' variants':[]})).valid)

    def test_provider_streams_with_realistic_opaque_ids(self):
        ids=['v_0123456789abcdef0123'];text=json.dumps({'variants':[{'variant_id':ids[0],'predicted_score':-.25}]})
        for m in configured_models('ppi_rna_six').values():
            if m['provider']=='google':
                event=dict(modelVersion=m['model'],candidates=[dict(content={'parts':[{'text':text}]},finishReason='STOP')]);tail=''
            else:
                event=dict(model=m['model'],choices=[dict(delta={'content':text},finish_reason='stop')]);tail='data: [DONE]\n\n'
            raw=('data: '+json.dumps(event)+'\n\n'+tail).encode()
            r=parse(raw,m,ids,200,0)
            self.assertEqual(r['status'],'ok');self.assertEqual(r['value'],{ids[0]:-.25})


if __name__=='__main__':unittest.main()
