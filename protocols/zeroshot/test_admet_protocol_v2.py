"""Endpoint semantics, row preservation and dispatch/evaluator integration."""
import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import admet_protocol as legacy
import admet_protocol_v2 as protocol
import monomer_campaign as campaign
import test_monomers as fixtures
from evaluate_monomers import evaluate
from shared_protocol import load,save,sha


class EndpointTests(unittest.TestCase):
    def setUp(self):
        self.fixture=fixtures.IntegrationTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root=self.fixture.root;self.here=self.fixture.here
        self.tasks=fixtures.fixture_tasks()
        registry=load(protocol.REGISTRY)
        d=copy.deepcopy(registry['endpoints']['hia_hou'])
        d['frozen_description']=self.tasks[0]['description']
        source=self.root/'data/admet_anewdde/frozen/tasks.public.json'
        self.registry=self.root/'endpoints.json'
        save(self.registry,dict(schema=registry['schema'],tasks_source=dict(path=str(source),sha256=sha(source)),
                                endpoints=dict(synthetic=d)))
        self.addCleanup(patch.stopall)
        patch.object(protocol,'REGISTRY',self.registry).start()

    def prepare(self):
        # Builders can hash, but cannot parse, private label files.
        original=load
        def no_labels(p):
            if Path(p).name=='labels.local.json':raise AssertionError('labels leaked into preparation')
            return original(p)
        with patch.object(legacy,'load',side_effect=no_labels),patch.object(protocol,'load',side_effect=no_labels):
            legacy.prepare('old',here=self.here)
            self.old=self.here/'inputs/old';self.new=self.here/'inputs/new'
            protocol.prepare('new',self.old,here=self.here)
        return protocol.validate_bundle(self.new)

    def test_real_preparation_preserves_rows_groups_and_defines_scores(self):
        manifest,queue=self.prepare()
        self.assertEqual(len(queue),12)
        self.assertEqual(manifest['schema'],protocol.VERSION)
        for name in ('tasks.json','shapes.json','groups.json','queries/admet/synthetic/query.json'):
            self.assertEqual((self.new/name).read_bytes(),(self.old/name).read_bytes())
        for a,b in zip(queue[::2],queue[1::2]):
            x,y=(json.loads(u['template']['user']) for u in (a,b))
            self.assertEqual(x['task']['endpoint_definition']['classes']['positive']['value'],1)
            self.assertEqual(x['task']['metric'],'auroc')
            self.assertEqual(x['reference'],{'kind':'independent_molecules'})
            self.assertEqual(x['output_contract']['score']['semantics'],'ordinal_positive_class_propensity')
            self.assertNotIn('sources',x['task']['endpoint_definition'])
            x.pop('evidence');y.pop('evidence');self.assertEqual(x,y)

    def test_registry_unknown_kind_missing_unit_and_reversed_labels_rejected(self):
        original=load(self.registry)
        for mutation in ('kind','unit','labels','description'):
            with self.subTest(mutation=mutation):
                obj=copy.deepcopy(original);d=obj['endpoints']['synthetic']
                if mutation=='kind':d['kind']='regression'
                elif mutation=='unit':d['readout']['unit']=''
                elif mutation=='labels':d['classes']['positive']['value']=0
                else:d['frozen_description']='different task'
                save(self.registry,obj)
                with self.assertRaises(ValueError):protocol.endpoint_definitions(self.tasks)

    def test_registry_cannot_change_frozen_candidate_identity(self):
        tasks=copy.deepcopy(self.tasks);tasks[0]['candidates'].reverse()
        with self.assertRaisesRegex(ValueError,'identity'):protocol.endpoint_definitions(tasks)

    def rehash(self,name):
        m=load(self.new/'MANIFEST.json');m['files'][name]=sha(self.new/name)
        save(self.new/'MANIFEST.json',m)
        (self.new/'MANIFEST.sha256').write_text(sha(self.new/'MANIFEST.json')+'\n')

    def test_rehashed_endpoint_unit_forgery_rejected(self):
        self.prepare();d=load(self.new/'ENDPOINTS.json')
        d['synthetic']['readout']['unit']='invented_unit'
        save(self.new/'ENDPOINTS.json',d);self.rehash('ENDPOINTS.json')
        with self.assertRaisesRegex(ValueError,'semantics'):protocol.validate_bundle(self.new)

    def test_rehashed_actual_prompt_wt_or_probability_forgery_rejected(self):
        _,queue=self.prepare()
        for alteration in ('wt','probability','positive'):
            with self.subTest(alteration=alteration):
                q=copy.deepcopy(queue);u=json.loads(q[0]['template']['user'])
                if alteration=='wt':u['reference']['wt_sequence']='ACGU'
                elif alteration=='probability':u['output_contract']['score']['unit']='probability'
                else:u['task']['endpoint_definition']['classes']['positive']['value']=0
                q[0]['template']['user']=json.dumps(u);save(self.new/'QUEUE.json',q);self.rehash('QUEUE.json')
                with self.assertRaisesRegex(ValueError,'replay'):protocol.validate_bundle(self.new)

    def test_unknown_or_mislabeled_evaluator_endpoint_rejected_before_writes(self):
        legacy.prepare('old',here=self.here)
        refs=self.here/'evaluator/old/LABEL_REFERENCES.json';rows=load(refs)
        rows[0]['kind']='regression';rows[0]['metric']='spearman';save(refs,rows)
        with self.assertRaisesRegex(ValueError,'semantics'):
            protocol.prepare('new',self.here/'inputs/old',here=self.here)
        self.assertFalse((self.here/'inputs/new').exists())

    def test_campaign_dispatch_and_evaluator_use_auroc_without_any_api(self):
        self.prepare();run=self.here/'runs/synthetic'
        campaign.prepare(run,[self.new],{'model':self.fixture.model},2)
        plan,requests=campaign.validate(run)
        self.assertEqual(plan['planned_requests'],12)
        for request in requests:
            # Synthetic truth only: reversed ordering must yield AUROC 0, not abs/corrected 1.
            value={i:float(1-int(i[1:])%2) for i in request['expected_ids']}
            campaign.write_unit(run,'model',request,dict(status='ok',failure_category='ok',
                request_id=request['request_id'],messages_sha256=request['messages_sha256']),value)
        rows=evaluate(run)
        self.assertEqual(len(rows),4)
        self.assertTrue(all(r['metric']=='auroc' and r['value']==0 for r in rows))
        (run/'models/model/units'/(requests[-1]['request_id']+'.json')).unlink()
        rows=evaluate(run)
        self.assertTrue(any(r['status']=='incomplete' and r['value'] is None for r in rows))

    def test_existing_version_cannot_be_overwritten(self):
        self.prepare()
        with self.assertRaisesRegex(ValueError,'New version'):
            protocol.prepare('new',self.old,here=self.here)

    def test_campaign_cannot_switch_classification_to_regression(self):
        self.prepare();rp=self.here/'evaluator/new/LABEL_REFERENCES.json';refs=load(rp)
        refs[0].update(kind='regression',metric='spearman');save(rp,refs)
        with self.assertRaisesRegex(ValueError,'semantics'):
            campaign.prepare(self.here/'runs/bad',[self.new],{'model':self.fixture.model},2)
        self.assertFalse((self.here/'runs/bad').exists())


if __name__=='__main__':unittest.main()
