"""End-to-end framework campaigns with real agent loops/local RNA calculations.

Provider responses are deterministic offline fixtures. These tests verify the
scientific/control contracts, never evidence of predictive improvement.
"""

import asyncio
import copy
import json
import re
import sqlite3
from types import SimpleNamespace

import pytest
from test_research_agents import fixture as legacy_fixture
from test_research_agents import response

from evagent.integrated.gateway import RequestGateway, digest
from evagent.integrated.research.campaign import run_campaign
from evagent.integrated.research.contracts import fuse, save
from evagent.integrated.research.framework_model import LedgerAgentModel
from evagent.integrated.research.framework_runner import DESIGN, FrameworkTools


def framework_fixture(tmp_path, *, msa=False):
    path = legacy_fixture(tmp_path)
    config = json.loads(path.read_text())
    config.update(repeats=1, analysis_profile='framework_agents_v4', analysis=DESIGN)
    config['model_config']['research_tool_protocol'] = 'native_framework_v4'
    config['tasks'][0]['assay_memory_file'] = str(tmp_path / 'assay_memory.json')
    save(tmp_path / 'assay_memory.json', {
        'assay_id': 'synthetic_test',
        'assay_summary': 'Synthetic RNA folding assay. REVIEWED_ENDPOINT_CONTEXT.',
        'property_direction': 'Larger measured endpoint values are better.',
    })
    sample = json.loads((tmp_path / 'sample.json').read_text())
    for i, row in enumerate(sample['variants']):
        row['score'] = 1000000 + i
        row['private_evaluator_note'] = 'EVALUATOR_ONLY_SENTINEL'
    save(tmp_path / 'sample.json', sample)
    snapshot = json.loads((tmp_path / 'snapshot.json').read_text())
    snapshot.update(literature_policy='passage_review_v2', literature_documents=[],
                    denied_dois=[], denied_titles=[], msa_usage='external_fusion')
    if msa:
        snapshot['msa_values'] = {row['variant_id']: float(i % 7) for i, row in enumerate(sample['variants'])}
        snapshot['sources']['msa'].update(status='available', reason=None,
            task={'note': 'FROZEN_MSA_NOT_PROMPTED'}, provenance=[{'kind': 'offline statistical fixture'}])
    snapshot['sha256'] = digest({key: value for key, value in snapshot.items() if key != 'sha256'})
    save(tmp_path / 'snapshot.json', snapshot)
    save(path, config)
    return path


class FrameworkTransport:
    def __init__(self, fault=None, fault_role='single'):
        self.requests = []
        self.fault, self.fault_role = fault, fault_role

    def __call__(self, request):
        self.requests.append(request)
        meta = request['metadata']
        if meta.get('phase') != 'analysis':
            user = request['messages'][-1]['content']
            ids = re.search(r'Required IDs: ([^\n]+)', user).group(1).split(', ')
            base = 0.500001 if meta['condition'] == 'reference' else 0.100001
            return response(json.dumps({'variants': [
                {'variant_id': variant, 'predicted_score': base + int(variant[1:]) / 1000}
                for variant in ids]}))

        role = meta['role']
        task = json.loads(next(message['content'] for message in request['messages']
                               if message['role'] == 'user'))['task']
        observed = [json.loads(message['content']) for message in request['messages']
                    if message['role'] == 'tool']
        faulty = role == self.fault_role
        if not observed:
            calls = []
            if role in {'single', 'structure'}:
                args = {'mode': 'pairing'}
                if faulty and self.fault == 'tool_coverage':
                    args['candidate_ids'] = [task['variants'][0]['variant_id']]
                calls.append({'name': 'rna_secondary', 'args': args, 'id': role + '-structure'})
            if role in {'single', 'literature'}:
                calls.append({'name': 'literature_search', 'args': {'query': 'Synthetic RNA folding mechanism'},
                              'id': role + '-literature'})
            result = {**response(''), 'tool_calls': calls}
            if faulty and self.fault == 'unknown_usage':
                result['usage_summary'] = None
            return result

        evidence = {row['variant_id']: row for result in observed for row in result['evidence']
                    if row.get('source') == 'structure'}
        report = {
            'claims': [],
            'assessments': [
                {'variant_id': variant, 'relative_to_wt': 'uncertain',
                 'interpretation': 'Independent ' + role + ' computation; functional direction is unestablished.',
                 'evidence_ids': [row['id']], 'limitations': 'Folding is not measured function.'}
                for variant, row in evidence.items()
            ],
            'pairwise_preferences': [],
            'limitations': ['PRIVATE_' + role.upper() + '_REPORT', 'No reviewed literature is available.'],
            'abstain': role == 'literature',
        }
        if faulty and self.fault == 'report_coverage':
            report['assessments'] = report['assessments'][:-1]
        if faulty and self.fault == 'citation':
            report['assessments'][0]['evidence_ids'] = ['unobserved-evidence']
        if faulty and self.fault == 'schema':
            report = {'claims': 5}
        calls = [{'name': 'AnalystReport', 'args': report, 'id': role + '-report'}]
        if faulty and self.fault == 'multiple_reports':
            calls.append({'name': 'AnalystReport', 'args': report, 'id': role + '-extra-report'})
        return {**response(''), 'tool_calls': calls}


def ledger_rows(tmp_path):
    with sqlite3.connect(tmp_path / 'out/provider/request_ledger.sqlite') as db:
        return db.execute('SELECT status,tokens,request_json,response_json FROM attempts ORDER BY id').fetchall()


def units(tmp_path):
    return [json.loads(path.read_text()) for path in (tmp_path / 'out/units').glob('*.json')]


def test_framework_three_conditions_real_tools_accounting_and_resume(tmp_path):
    path = framework_fixture(tmp_path, msa=True)
    dry = run_campaign(path, dry_run=True)
    assert dry['initial_CR_units'] == 6 and dry['method_results'] == 3 and dry['api_calls'] == 0
    assert not (tmp_path / 'out').exists()
    transport = FrameworkTransport()
    partial = run_campaign(path, transport=transport, stop_after=3)
    assert partial['status'] == 'stopped_after_test_limit' and partial['completed_units'] == 3
    first_group_calls = len(transport.requests)
    report = run_campaign(path, transport=transport)
    assert report['status'] == 'complete' and report['completed_units'] == 6
    assert report['statuses'] == {'ok': 3}
    assert len(transport.requests) == 18 and first_group_calls == 9
    assert json.loads((tmp_path / 'out/calibration.json').read_text())['status'] == 'passed'
    assert report['ledger']['provider_attempts'] == 18
    assert report['ledger']['reported_tokens'] == 360
    rows = ledger_rows(tmp_path)
    assert len(rows) == 18 and all(row[0] == 'ok' and row[1] == 20 for row in rows)
    assert all(row['msa_available'] for row in report['rows'])

    output_units = units(tmp_path)
    assert len(output_units) == 6
    refs = [request for request in transport.requests if request['metadata']['condition'] == 'reference']
    assert len(refs) == 2  # Fresh once per group; dependency replays never rebill.
    analysts = [request for request in transport.requests if request['metadata'].get('phase') == 'analysis']
    assert len(analysts) == 12
    assert sum(len(analyst['model_calls']) for unit in output_units
               for analyst in unit['reports'].values()) == 12
    assert all(unit['draft_cr'] != unit['cr'] for unit in output_units if unit['condition'] != 'reference')

    sample = json.loads((tmp_path / 'sample.json').read_text())
    snapshot = json.loads((tmp_path / 'snapshot.json').read_text())
    ids = [row['variant_id'] for row in sample['variants']]
    for evaluated in report['rows']:
        cr = {variant: value for unit in output_units if unit['condition'] == evaluated['condition']
              for variant, value in unit['cr'].items()}
        assert evaluated['predictions'] == fuse(ids, cr, snapshot['msa_values'])

    for request in transport.requests:
        text = json.dumps(request['messages'])
        assert 'EVALUATOR_ONLY_SENTINEL' not in text and '1000025' not in text
        assert 'FROZEN_MSA_NOT_PROMPTED' not in text
        if request['metadata']['condition'] != 'reference':
            assert '0.501001' not in text and 'draft_cr' not in text and 'initial_predictions' not in text
    first_calls = {request['metadata']['role']: request for request in analysts
                   if request['metadata']['group'] == 0 and request['metadata']['model_turn'] == 0}
    assert set(first_calls) == {'single', 'literature', 'structure'}
    specs = {role: {tool['name']: tool for tool in request['tool_declarations']
                    if tool['name'] != 'AnalystReport'} for role, request in first_calls.items()}
    assert specs['single'] == specs['literature'] | specs['structure']
    for role, request in first_calls.items():
        text = json.dumps(request['messages'])
        assert 'REVIEWED_ENDPOINT_CONTEXT' in text
        assert 'PRIVATE_LITERATURE_REPORT' not in text and 'PRIVATE_STRUCTURE_REPORT' not in text
        public = json.loads(next(message['content'] for message in request['messages'] if message['role'] == 'user'))
        assert public['assigned_sources'] == (['literature', 'structure'] if role == 'single' else [role])
    for unit in output_units:
        if unit['condition'] == 'multi':
            assert unit['reports']['literature']['abstain'] is True
            assert len(unit['reports']['structure']['assessments']) == 25
    again = run_campaign(path, transport=lambda _: pytest.fail('Completed replay called provider'))
    assert again['ledger']['reported_tokens'] == 360 and again['completed_units'] == 6


def test_framework_unknown_usage_stops_before_repair_and_never_retries(tmp_path):
    path = framework_fixture(tmp_path)
    transport = FrameworkTransport('unknown_usage')
    report = run_campaign(path, transport=transport)
    assert report['status'] == 'paused' and report['completed_units'] == 1
    assert len(transport.requests) == 2
    assert report['ledger']['provider_attempts'] == 2
    assert report['ledger']['unknown_usage_attempts'] == 1
    assert report['ledger']['reported_tokens'] == 20
    assert not any(request['metadata'].get('attempt_kind') == 'repair' for request in transport.requests)
    assert {unit['condition'] for unit in units(tmp_path)} == {'reference'}
    states = [json.loads(path.read_text()) for path in (tmp_path / 'out/framework_agent_states').glob('*.json')]
    assert len(states) == 1 and states[0]['status'] == 'paused'
    assert states[0]['model_calls'][0]['status'] == 'unknown_cost'
    assert all(not row['msa_available'] for row in report['rows'])
    resumed = run_campaign(path, transport=lambda _: pytest.fail('Unknown request was resubmitted'))
    assert resumed['status'] == 'paused' and resumed['ledger']['provider_attempts'] == 2


@pytest.mark.parametrize('fault,reason', [
    ('tool_coverage', 'Complete candidate secondary analysis required'),
    ('report_coverage', 'Assess every candidate exactly once'),
    ('citation', 'Unknown or unobserved evidence citation'),
    ('schema', 'Failed to parse structured output'),
    ('multiple_reports', 'multiple structured responses'),
])
def test_framework_invalid_analysis_fails_without_reference_fallback(tmp_path, fault, reason):
    path = framework_fixture(tmp_path)
    transport = FrameworkTransport(fault)
    report = run_campaign(path, transport=transport)
    assert report['status'] == 'paused' and report['reason'] == 'Calibration failed'
    assert report['completed_units'] == 6 and report['statuses'] == {'ok': 2, 'failed': 1}
    single = [unit for unit in units(tmp_path) if unit['condition'] == 'single']
    assert len(single) == 2
    assert all(unit['status'] == 'analysis_invalid' and unit['cr'] is None for unit in single)
    assert all(reason.lower() in unit['reports']['single']['error'].lower() for unit in single)
    assert not any(request['metadata']['condition'] == 'single' and
                   request['metadata'].get('phase') == 'integration' for request in transport.requests)
    assert not any(request['metadata'].get('attempt_kind') == 'repair' for request in transport.requests)
    row = next(row for row in report['rows'] if row['condition'] == 'single')
    assert row['status'] == 'failed' and 'predictions' not in row
    assert report['ledger']['provider_attempts'] == len(transport.requests)


def test_framework_freeze_rejects_wrong_msa_usage_and_missing_task_context(tmp_path):
    path = framework_fixture(tmp_path)
    snapshot_path = tmp_path / 'snapshot.json'
    original = json.loads(snapshot_path.read_text())
    changed = {**original, 'msa_usage': 'prompt_only'}
    changed['sha256'] = digest({key: value for key, value in changed.items() if key != 'sha256'})
    save(snapshot_path, changed)
    with pytest.raises(ValueError, match='external-only MSA'):
        run_campaign(path, dry_run=True)
    save(snapshot_path, original)
    save(tmp_path / 'assay_memory.json', {})
    with pytest.raises(ValueError, match='reviewed task context'):
        run_campaign(path, dry_run=True)


def validation_tools(card, *, role='single', documents=True):
    """A returned-evidence fixture; execution is covered by campaign tests above."""
    tools = FrameworkTools.__new__(FrameworkTools)
    tools.task = {'wt_sequence': 'AGCU', 'variants': [{'variant_id': 'M01'}]}
    tools.role = role
    tools.sources = ['literature', 'structure'] if role == 'single' else ['literature']
    tools.snapshot = {'literature_documents': [{'id': 'paper'}] if documents else []}
    tools.literature_read = True
    tools.structure = SimpleNamespace(required_geometry=set(), invocations={
        'rna_secondary': {'M01'}, 'rna_contacts': set(), 'rna_annotations': set(),
    }) if role == 'single' else None
    tools.observed = {'S': {'id': 'S', 'source': 'structure', 'variant_id': 'M01', **card},
                      'L': {'id': 'L', 'source': 'literature', 'excerpt': 'Family mechanism.'}}
    tools.records = []
    tools.journal = lambda event: None
    return tools


def validation_report(*, literature_only=False):
    return {'claims': [], 'assessments': [] if literature_only else [
        {'variant_id': 'M01', 'relative_to_wt': 'uncertain', 'interpretation': 'Computed facts only.',
         'evidence_ids': ['S'], 'limitations': 'Function was not measured.'}],
        'pairwise_preferences': [], 'limitations': ['Evidence is limited.'], 'abstain': literature_only}


def test_empty_literature_requires_explicit_abstention_and_no_claims():
    tools = validation_tools({}, role='literature', documents=False)
    valid = validation_report(literature_only=True)
    tools.validate(valid)
    bad = {**valid, 'abstain': False}
    with pytest.raises(ValueError, match='Absent literature requires abstention'):
        tools.validate(bad)
    bad = {**valid, 'claims': [{'text': 'An invented mechanism.', 'evidence_ids': ['L'],
                             'candidate_ids': [], 'positions': []}]}
    with pytest.raises(ValueError, match='no invented claims'):
        tools.validate(bad)


@pytest.mark.parametrize('candidate_ids,positions', [(['M01'], []), ([], [1])])
def test_single_agent_cannot_invent_mapping_from_literature_only(candidate_ids, positions):
    tools = validation_tools({'selected_positions': [1]})
    report = validation_report()
    report['claims'] = [{'text': 'The family mechanism maps here.', 'evidence_ids': ['L'],
                         'candidate_ids': candidate_ids, 'positions': positions}]
    with pytest.raises(ValueError, match='Literature mapping is not independently verified'):
        tools.validate(report)
    report['claims'][0].update(candidate_ids=[], positions=[])
    tools.validate(report)


@pytest.mark.parametrize('card,observed', [
    ({'selected_positions': [1]}, [1]),
    ({'pair_changes': [{'positions': [1, 2], 'wt_probability': 0.2}]}, [1, 2]),
    ({'unpaired_probability': [{'position': 2, 'wt': 0.8}]}, [2]),
    ({'candidate': {'centroid': '....', 'mea': '....'}}, [1, 2, 3, 4]),
])
def test_computed_claim_coordinates_must_occur_in_returned_evidence(card, observed):
    tools = validation_tools(card)
    report = validation_report()
    report['claims'] = [{'text': 'The computed facts cover these coordinates.', 'evidence_ids': ['S'],
                         'candidate_ids': ['M01'], 'positions': observed}]
    tools.validate(report)
    missing = set(range(1, 5)) - set(observed)
    if missing:
        bad = copy.deepcopy(report)
        bad['claims'][0]['positions'] = [min(missing)]  # In WT range, absent from cited result.
        with pytest.raises(ValueError, match='absent from cited computed evidence'):
            tools.validate(bad)


def test_nonempty_paperqa_retrieval_is_literal_local_and_uses_original_offsets(tmp_path, monkeypatch):
    import lmi
    import paperqa.core
    from test_evidence_agents_v2 import paper

    from evagent.integrated.research.literature_review import screen_document

    async def forbidden_model(*args, **kwargs):
        pytest.fail('Frozen literal retrieval must not call a summary or ranking LLM')

    monkeypatch.setattr(lmi.LiteLLMModel, 'call_single', forbidden_model)
    monkeypatch.setattr(paperqa.core, '_map_fxn_summary', forbidden_model)
    source = paper()
    documents, decisions = screen_document(source)
    assert len(documents) == 1 and decisions
    config = json.loads(framework_fixture(tmp_path).read_text())
    ledger = RequestGateway(model_config=config['model_config'], run_dir=tmp_path / 'tool-provider',
        fingerprint='literal-evidence-only', budget_mode='unlimited', token_budget=None,
        transport=lambda _: pytest.fail('Literal retrieval submitted a model request'))
    model = LedgerAgentModel(ledger, {'role': 'literature'}, 6)
    tools = FrameworkTools({'wt_sequence': 'AGCU', 'variants': []},
                           {'literature_documents': documents, 'denied_dois': [], 'denied_titles': []},
                           tmp_path / 'snapshot.json', 'literature', model)
    result = asyncio.run(tools._literature('RNA mutation mechanism and magnesium'))
    assert result['status'] == 'ok' and result['evidence']
    assert 'literal' in result['retrieval'] and tools.literature_read
    for excerpt in result['evidence']:
        assert source['text'][excerpt['source_start']:excerpt['source_end']] == excerpt['excerpt']
        assert excerpt['source_sha256'] == digest(source['text'])
        assert excerpt['doi'] == source['doi'] and excerpt['source'] == 'literature'
        assert 'A12G' not in json.dumps(excerpt)
    assert ledger.summary()['provider_attempts'] == 0 and model.records == []
    assert type(tools.paperqa.embedding).__name__ == 'SparseEmbeddingModel'
    ledger.close()
