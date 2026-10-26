import copy
import json
import pytest
from test_framework_campaign import framework_fixture, FrameworkTransport, units
from evagent.integrated.research.campaign import run_campaign
from evagent.integrated.research.grounded_runner import DESIGN, check_background_query, compact_card, integration_payload


class GroundedTransport(FrameworkTransport):
    def __call__(self, request):
        # Native structured-report acknowledgement is not a scientific result.
        request=copy.deepcopy(request)
        request['messages']=[m for m in request['messages'] if m['role']!='tool' or
                             (m['content'].startswith('{') and '"evidence"' in m['content'])]
        return super().__call__(request)


def fixture(tmp_path):
    p = framework_fixture(tmp_path)
    c = json.loads(p.read_text())
    c.update(analysis_profile='grounded_agents_v5', analysis=DESIGN, evaluation_ns=[50])
    p.write_text(json.dumps(c))
    return p


def test_grounded_real_tools_four_conditions_and_cache(tmp_path):
    p = fixture(tmp_path)
    dry = run_campaign(p, dry_run=True)
    assert dry['initial_CR_units'] == 8 and dry['method_results'] == 4
    t = GroundedTransport()
    r = run_campaign(p, transport=t)
    assert r['status'] == 'complete' and r['completed_units'] == 8
    assert r['statuses'] == {'ok': 4}
    assert r['paired_four_condition'][0]['tasks'] == 1
    assert len(t.requests) == 20
    for req in t.requests:
        if req['metadata']['condition'] != 'reference':
            text = json.dumps(req['messages'])
            assert 'delta_ensemble_kcal_mol' not in text
            assert 'mfe_kcal_mol' not in text
        if req['metadata'].get('phase') == 'integration' and req['metadata']['condition'] in ['single','multi']:
            text = req['messages'][-1]['content']
            assert 'Independent single computation' not in text
            assert 'Independent structure computation' not in text
            assert 'PRIVATE_SINGLE_REPORT' not in text
    assert all(u['withheld_hypotheses'] for u in units(tmp_path) if u['condition'] in ['single','multi'])
    r2 = run_campaign(p, transport=lambda _:pytest.fail('cached replay called model'))
    assert r2['completed_units'] == 8


def test_query_allows_endpoint_not_effects():
    check_background_query('yeast tRNA aminoacylation nuclear export maturation fitness')
    for q in ['tRNA variant effects', 'tRNA DMS scores', 'tRNA mutational scanning', 'A12G ribozyme activity']:
        with pytest.raises(ValueError):check_background_query(q)


def test_uncertainty_cannot_bypass_gate_via_claims_or_limitations():
    e={'id':'s1','source':'structure','tool':'rna_secondary','variant_id':'M01','status':'available'}
    row={'variant_id':'M01','relative_to_wt':'uncertain','evidence_ids':['s1'],'interpretation':'UNSUPPORTED_DIRECTION'}
    claim={'candidate_ids':['M01'],'evidence_ids':['s1'],'text':'UNSUPPORTED_DIRECTION'}
    r={'role':'single','abstain':False,'claims':[claim], 'assessments':[row], 'pairwise_preferences':[],
       'limitations':['UNSUPPORTED_DIRECTION'], 'observed_evidence':[{'evidence':[e]}]}
    facts, conclusions, withheld=integration_payload({'single':r})
    assert len(withheld)==1 and 'UNSUPPORTED_DIRECTION' not in json.dumps(conclusions)
    r['assessments'][0]['relative_to_wt']='lower'
    assert integration_payload({'single':r})[1][0]['assessments']==[]
    e.update(tool='rna_contacts',changed_positions=[3],usable_positions=[3],contacts=[{'positions':[3,12]}])
    assert len(integration_payload({'single':r})[1][0]['assessments'])==1


def test_compact_card_retains_identity_missing_and_mapping():
    raw={'source':'structure','id':'s1','variant_id':'M01','status':'unavailable','reason':'No mapping',
         'changed_positions':[2], 'delta_ensemble_kcal_mol':42, 'wt':{'mfe_kcal_mol':-19}}
    frozen=copy.deepcopy(raw);v=compact_card(raw)
    assert raw==frozen and v['reason']=='No mapping' and v['changed_positions']==[2]
    assert 'delta_ensemble_kcal_mol' not in v and 'wt' not in v and v['raw_card_sha256']


class CorrectOnce(GroundedTransport):
    def __init__(self):super().__init__('citation');self.repaired=0
    def __call__(self, request):
        if any('report_validation_error' in str(m.get('content')) for m in request['messages']):
            self.fault=None;self.repaired+=1
        return super().__call__(request)


def test_one_bounded_citation_correction_is_ledgered(tmp_path):
    p=fixture(tmp_path);t=CorrectOnce();r=run_campaign(p, transport=t)
    assert r['status']=='complete' and t.repaired==1
    assert len(t.requests)==21 and r['ledger']['provider_attempts']==21
    assert len(list((tmp_path/'out/report_corrections').glob('*.json')))==1


def test_repeated_bad_report_stops_without_reference_substitution(tmp_path):
    p=fixture(tmp_path);t=GroundedTransport('citation');r=run_campaign(p,transport=t)
    assert r['status']=='paused' and r['reason']=='Calibration failed'
    bad=[u for u in units(tmp_path) if u['condition']=='single']
    assert all(u['status']=='analysis_invalid' and u['cr'] is None for u in bad)
    assert all(len(u['reports']['single']['model_calls'])==3 for u in bad)


def test_unknown_usage_does_not_enter_report_correction(tmp_path):
    p=fixture(tmp_path);t=GroundedTransport('unknown_usage');r=run_campaign(p,transport=t)
    assert r['status']=='paused' and r['completed_units']==2
    assert not (tmp_path/'out/report_corrections').exists()
