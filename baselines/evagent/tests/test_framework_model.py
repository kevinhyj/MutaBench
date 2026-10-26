"""Offline contract tests for the real LangChain loop and ledger adapter."""

import asyncio
import copy
import json

import pytest
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from evagent.integrated.gateway import ProviderStreamInterrupted, RequestGateway, google_stream_transport
from evagent.integrated.research.contracts import CallLimit, ResearchFailure
from evagent.integrated.research.framework_model import LedgerAgentModel, canonical_wire_schema, encode_messages
from evagent.runtime import ExecutionPaused


MODEL = {'model': 'gemini-test', 'thinking_level': 'medium', 'max_output_tokens': 100,
         'research_tool_protocol': 'native_framework_v4', 'transport_mode': 'stream'}
DECLARATION = {'name': 'inspect_evidence', 'description': 'Read frozen evidence.',
               'parameters': {'type': 'object', 'properties': {'candidate': {'type': 'string'}},
                              'required': ['candidate']}}


def response(**changes):
    return {'final_text': 'complete', 'finish_reason': 'STOP', 'tool_calls': [],
            'usage_summary': {'input_tokens': 8, 'output_tokens': 2, 'total_tokens': 12,
                              'reasoning_tokens': 2, 'model_calls': 1},
            'response_metadata': {}, **changes}


def gateway(tmp_path, transport, **kwargs):
    return RequestGateway(model_config=MODEL, run_dir=tmp_path, fingerprint='framework-fixture',
                          token_budget=None, budget_mode='unlimited', transport=transport, **kwargs)


def test_real_create_agent_tool_strategy_roundtrip_and_shared_ledger(tmp_path):
    inspected, requests = [], []

    @tool
    def inspect_evidence(candidate: str) -> str:
        """Read only the frozen evidence for a candidate."""
        inspected.append(candidate)
        return json.dumps({'evidence_id': 'E-' + candidate, 'limitation': 'test fixture'})

    class FinalReport(BaseModel):
        conclusion: str

    def transport(request):
        requests.append(request)
        if len(requests) == 1:
            calls = [{'name': 'inspect_evidence', 'args': {'candidate': 'A'}},
                     {'name': 'inspect_evidence', 'args': {'candidate': 'B'}, 'id': 'provider-B'}]
            return response(final_text='', tool_calls=calls, native_parts=[
                {'function_call': calls[0], 'thought_signature': '_wFzaWc='},
                {'function_call': calls[1]},
            ])
        call = {'name': 'FinalReport', 'args': {'conclusion': 'Evidence is limited.'}, 'id': 'report'}
        return response(final_text='', tool_calls=[call])

    ledger = gateway(tmp_path, transport)
    model = LedgerAgentModel(ledger, {'role': 'structure', 'unit': 'fixture'}, 6)
    agent = create_agent(model=model, tools=[inspect_evidence],
                         response_format=ToolStrategy(FinalReport, handle_errors=False))
    result = agent.invoke({'messages': [HumanMessage('Compare A and B.')]},
                          config={'recursion_limit': 12})
    assert result['structured_response'].conclusion == 'Evidence is limited.'
    assert sorted(inspected) == ['A', 'B']
    assert len(model.records) == 2
    assert len(model.tool_call_records) == 3
    assert model.tool_call_records[-1]['tool'] == 'FinalReport'
    assert all(receipt['admitted'] for receipt in model.tool_call_records)
    assert all(record['role'] == 'structure' for record in model.records)
    assert ledger.summary()['reported_tokens'] == 24
    second = requests[1]['messages']
    assert second[1]['native_parts'][0]['thought_signature'] == '_wFzaWc='
    assert 'id' not in second[1]['native_parts'][0]['function_call']
    assert 'id' not in second[2]['native_parts'][0]['function_response']
    assert second[3]['native_parts'][0]['function_response']['id'] == 'provider-B'
    assert {row['name'] for row in requests[0]['tool_declarations']} == {'inspect_evidence', 'FinalReport'}
    assert requests[0]['tool_choice'] == 'any'
    ledger.close()


def test_native_google_stream_preserves_signed_parts_and_function_responses(tmp_path, monkeypatch):
    from google import genai
    from google.genai import types

    captured = []
    signature = b'\xff\x01signed'
    first_parts = [types.Part(function_call=types.FunctionCall(name='inspect_evidence',
                    args={'candidate': 'A'}, id='A'), thought_signature=signature),
                   types.Part(function_call=types.FunctionCall(name='inspect_evidence',
                    args={'candidate': 'B'}))]

    class FakeClient:
        def __init__(self, **kwargs):
            captured.append({'client': kwargs})
            self.models = self

        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def generate_content_stream(self, **kwargs):
            captured[-1]['generation'] = kwargs
            parts = first_parts if len(captured) == 1 else [
                types.Part(text='final', thought_signature=b'text signature')]
            # A real stream can deliver usage/finish only in the final chunk.
            yield types.GenerateContentResponse(candidates=[types.Candidate(
                content=types.Content(role='model', parts=parts))])
            yield types.GenerateContentResponse(candidates=[types.Candidate(finish_reason='STOP')],
                usage_metadata=types.GenerateContentResponseUsageMetadata(
                    prompt_token_count=8, candidates_token_count=2, total_token_count=12,
                    thoughts_token_count=2))

    monkeypatch.setenv('GEMINI_API_KEY', 'offline-fake-key')
    monkeypatch.setattr(genai, 'Client', FakeClient)
    ledger = gateway(tmp_path, google_stream_transport)
    model = LedgerAgentModel(ledger, {'role': 'structure'}, 4)
    bound = model.bind_tools([DECLARATION])
    first = bound.invoke([HumanMessage('inspect')])
    final = bound.invoke([HumanMessage('inspect'), first,
                          ToolMessage('A evidence', tool_call_id=first.tool_calls[0]['id']),
                          ToolMessage('B evidence', tool_call_id=first.tool_calls[1]['id'])])
    assert final.content == 'final'
    assert final.usage_metadata['total_tokens'] == 12
    assert final.usage_metadata['output_token_details']['reasoning'] == 2
    contents = captured[1]['generation']['contents']
    assert len(contents) == 3
    assert len(contents[-1].parts) == 2
    assert contents[1].parts[0].thought_signature == signature
    assert contents[1].parts[1].function_call.id is None
    assert contents[-1].parts[1].function_response.id is None
    assert captured[0]['client']['http_options'].retry_options.attempts == 1
    config = captured[0]['generation']['config']
    assert config.automatic_function_calling.disable
    assert config.tools[0].function_declarations[0].parameters_json_schema == DECLARATION['parameters']
    assert final.additional_kwargs['google_native_parts'][0]['thought_signature']
    ledger.close()


def test_schema_and_role_changes_isolate_cache_without_secondary_cache(tmp_path):
    requests = []
    ledger = gateway(tmp_path, lambda request: requests.append(request) or response())
    prompt = [HumanMessage('complete input ' * 10000)]
    first = LedgerAgentModel(ledger, {'role': 'single'}, 3).bind_tools([DECLARATION]).invoke(prompt)
    replay_model = LedgerAgentModel(ledger, {'role': 'single'}, 3)
    replay = replay_model.bind_tools([DECLARATION]).invoke(prompt)
    assert first.response_metadata['request_sha256'] == replay.response_metadata['request_sha256']
    assert replay_model.records[0]['cache_hit']
    changed = {**DECLARATION, 'description': 'Revised frozen tool.'}
    LedgerAgentModel(ledger, {'role': 'single'}, 3).bind_tools([changed]).invoke(prompt)
    LedgerAgentModel(ledger, {'role': 'structure'}, 3).bind_tools([DECLARATION]).invoke(prompt)
    assert len(requests) == 3
    assert requests[0]['messages'][0]['content'] == prompt[0].content
    assert ledger.summary()['provider_attempts'] == 3
    ledger.close()


def test_unknown_usage_propagates_out_of_real_agent_and_records_failed_attempt(tmp_path):
    ledger = gateway(tmp_path, lambda request: response(usage_summary=None))
    model = LedgerAgentModel(ledger, {'role': 'single'}, 6)
    agent = create_agent(model=model, tools=[])
    with pytest.raises(ExecutionPaused):
        agent.invoke({'messages': [HumanMessage('task')]})
    assert len(model.records) == 1 and model.records[0]['status'] == 'unknown_cost'
    with pytest.raises(ExecutionPaused):
        agent.invoke({'messages': [HumanMessage('another task')]})
    assert ledger.summary()['provider_attempts'] == 1
    ledger.close()


def test_actual_connection_attempts_consume_role_allowance(tmp_path, monkeypatch):
    monkeypatch.setattr('evagent.integrated.gateway.time.sleep', lambda _: None)
    requests = []

    def transport(request):
        requests.append(request)
        if len(requests) == 1:
            raise ProviderStreamInterrupted(response(final_text='', finish_reason=None,
                response_metadata={'transport_error': {'exception_type': 'ReadError'}}))
        return response()

    policy = {'max_attempts': 3, 'backoff_seconds': [15, 30], 'authorization': 'offline fixture'}
    ledger = gateway(tmp_path, transport, connection_retry_policy=policy)
    model = LedgerAgentModel(ledger, {'role': 'structure'}, 2)
    model.invoke('task')
    assert len(model.records) == 2
    assert model.records[0]['status'] == 'failed_unknown_cost'
    assert requests[0]['metadata']['provider_attempt_limit'] == 2
    with pytest.raises(CallLimit):
        model.invoke('must not submit')
    assert len(requests) == 2
    ledger.close()


def test_exhausted_connection_retries_preserve_all_failing_records(tmp_path, monkeypatch):
    monkeypatch.setattr('evagent.integrated.gateway.time.sleep', lambda _: None)

    def transport(request):
        raise ProviderStreamInterrupted(response(final_text='', finish_reason=None,
            response_metadata={'transport_error': {'exception_type': 'ReadError'}}))

    policy = {'max_attempts': 3, 'backoff_seconds': [15, 30], 'authorization': 'offline fixture'}
    ledger = gateway(tmp_path, transport, connection_retry_policy=policy)
    model = LedgerAgentModel(ledger, {'role': 'structure'}, 2)
    with pytest.raises(ExecutionPaused):
        model.invoke('task')
    assert len(model.records) == 2
    assert ledger.summary()['provider_attempts'] == 2
    assert ledger.summary()['paused']
    ledger.close()


@pytest.mark.parametrize('calls', [
    [{'name': 'unknown', 'args': {}}],
    [{'name': 'inspect_evidence', 'args': {}, 'id': 'same'},
     {'name': 'inspect_evidence', 'args': {}, 'id': 'same'}],
])
def test_invalid_native_calls_are_terminal_and_never_executed(tmp_path, calls):
    ledger = gateway(tmp_path, lambda _: response(final_text='', tool_calls=calls))
    model = LedgerAgentModel(ledger, {'role': 'structure'}, 3)
    with pytest.raises(ResearchFailure):
        model.bind_tools([DECLARATION]).invoke('task')
    assert model.records[0]['status'] == 'terminal_failure'
    assert ledger.summary()['provider_attempts'] == 1
    ledger.close()


def test_tool_result_identity_checked_before_submission():
    ai = AIMessage(content='', tool_calls=[{'id': 'a', 'name': 'inspect_evidence', 'args': {}}],
                   additional_kwargs={'google_native_parts': [
                       {'function_call': {'name': 'inspect_evidence', 'args': {}, 'id': 'a'}}]})
    with pytest.raises(ValueError, match='missing'):
        encode_messages([HumanMessage('task'), ai])
    with pytest.raises(ValueError, match='outstanding'):
        encode_messages([HumanMessage('task'), ai, ToolMessage('result', tool_call_id='wrong')])
    with pytest.raises(ValueError, match='outstanding'):
        encode_messages([HumanMessage('task'), ai, ToolMessage('result', tool_call_id='a', name='wrong')])


def test_async_and_sync_share_same_allowance_and_gateway(tmp_path):
    ledger = gateway(tmp_path, lambda _: response())
    model = LedgerAgentModel(ledger, {'role': 'literature'}, 1)
    assert asyncio.run(model.ainvoke('task')).content == 'complete'
    with pytest.raises(CallLimit):
        model.invoke('task two')
    assert ledger.summary()['provider_attempts'] == 1
    ledger.close()


def test_legacy_protocol_cannot_bind_dynamic_tools(tmp_path):
    ledger = RequestGateway(model_config={key: value for key, value in MODEL.items()
                                          if key != 'research_tool_protocol'},
                            run_dir=tmp_path, fingerprint='legacy', token_budget=100,
                            transport=lambda _: pytest.fail('Do not submit'))
    with pytest.raises(ValueError, match='requires native_framework_v4'):
        LedgerAgentModel(ledger, {'role': 'single'}, 2)
    with pytest.raises(ValueError, match='require native_framework_v4'):
        ledger.invoke_messages(messages=[{'role': 'user', 'content': 'task'}],
                               run_name='test', tool_declarations=[DECLARATION])
    assert ledger.summary()['provider_attempts'] == 0
    ledger.close()


def test_invalid_tool_arguments_still_count_before_langchain_error_normalization(tmp_path):
    executed, requests = [], []

    @tool
    def inspect_evidence(candidate: str) -> str:
        """Read the frozen candidate evidence."""
        executed.append(candidate)
        return 'evidence'

    def transport(request):
        requests.append(request)
        if len(requests) == 1:
            return response(final_text='', tool_calls=[
                {'name': 'inspect_evidence', 'args': {'candidate': 123}, 'id': 'invalid-type'}])
        return response()

    ledger = gateway(tmp_path, transport)
    model = LedgerAgentModel(ledger, {'role': 'structure'}, 2)
    create_agent(model=model, tools=[inspect_evidence]).invoke({'messages': [HumanMessage('task')]})
    assert executed == []
    assert len(model.tool_call_records) == 1
    assert model.tool_call_records[0]['arguments'] == {'candidate': 123}
    assert any(message['role'] == 'tool' for message in requests[1]['messages'])
    files = list((tmp_path.parent / 'framework_tool_proposals').glob('*.jsonl'))
    assert files and any('invalid-type' in path.read_text() for path in files)
    ledger.close()


def test_oversized_native_batch_is_billed_but_nothing_executes(tmp_path):
    executed = []

    @tool
    def inspect_evidence(candidate: str) -> str:
        """Read the frozen candidate evidence."""
        executed.append(candidate)
        return 'evidence'

    calls = [{'name': 'inspect_evidence', 'args': {'candidate': str(i)}} for i in range(3)]
    ledger = gateway(tmp_path, lambda _: response(final_text='', tool_calls=calls))
    model = LedgerAgentModel(ledger, {'role': 'structure'}, 2, tool_maximum=2)
    with pytest.raises(CallLimit, match='before execution'):
        create_agent(model=model, tools=[inspect_evidence]).invoke({'messages': [HumanMessage('task')]})
    assert executed == []
    assert len(model.tool_call_records) == 3
    assert not any(receipt['admitted'] for receipt in model.tool_call_records)
    assert ledger.summary()['reported_tokens'] == 12
    assert len(model.records) == 1
    ledger.close()


def test_partial_unknown_usage_tool_proposals_are_preserved_without_admission(tmp_path):
    call = {'name': 'inspect_evidence', 'args': {'candidate': 'A'}, 'id': 'partial-A'}
    ledger = gateway(tmp_path, lambda _: response(final_text='', usage_summary=None, tool_calls=[call]))
    model = LedgerAgentModel(ledger, {'role': 'structure'}, 2)
    with pytest.raises(ExecutionPaused):
        model.bind_tools([DECLARATION]).invoke('task')
    assert len(model.tool_call_records) == 1
    assert not model.tool_call_records[0]['admitted']
    assert model.tool_call_records[0]['provider_status'] == 'unknown_cost'
    ledger.close()


def test_sdk_http400_retains_sanitized_diagnostic_without_retry_or_cost_assumption(tmp_path, monkeypatch):
    from google import genai
    from google.genai import errors

    credential = 'fake-configured-key+secret'
    calls = []
    detail = ('Schema has too many states at parameters.properties.assessments. '
              + credential + ' https://example.invalid/generate?key=' + credential
              + '\nAuthorization: Bearer private-header-token\n'
              + ' api_key=secondary-secret bearer inline-bearer-token ' + 'x' * 2000)

    class FakeClient:
        def __init__(self, **kwargs):
            self.models = self

        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def generate_content_stream(self, **kwargs):
            calls.append(kwargs)
            raise errors.ClientError(400, {'error': {'status': 'INVALID_ARGUMENT', 'message': detail},
                                          'headers': {'private': 'do-not-persist-headers'},
                                          'details': {'private': 'do-not-persist-details'}})

    monkeypatch.setenv('GEMINI_API_KEY', credential)
    monkeypatch.setattr(genai, 'Client', FakeClient)
    policy = {'max_attempts': 3, 'backoff_seconds': [15, 30], 'authorization': 'offline fixture'}
    ledger = gateway(tmp_path, google_stream_transport, connection_retry_policy=policy)
    model = LedgerAgentModel(ledger, {'role': 'single'}, 6)
    with pytest.raises(ExecutionPaused):
        model.bind_tools([DECLARATION]).invoke('offline schema request')
    row = ledger.db.execute('SELECT status,response_json,tokens FROM attempts').fetchone()
    raw = json.loads(row[1])
    diagnostic = raw['response_metadata']['transport_error']
    assert diagnostic['status_code'] == 400 and diagnostic['provider_status'] == 'INVALID_ARGUMENT'
    assert diagnostic['message'].startswith('Schema has too many states')
    assert len(diagnostic['message']) <= 1600
    for secret in (credential, 'secondary-secret', 'inline-bearer-token', 'private-header-token',
                   'example.invalid', 'do-not-persist-headers', 'do-not-persist-details'):
        assert secret not in row[1]
    assert row[0] == 'failed_unknown_cost' and row[2] is None
    assert raw['usage_summary'] is None and len(calls) == 1
    assert ledger.summary()['paused'] and ledger.summary()['unknown_usage_attempts'] == 1
    ledger.close()


def test_wire_schema_simplifies_only_length_bounds_and_keeps_local_validation(tmp_path):
    from langchain_core.utils.function_calling import convert_to_openai_tool

    class Item(BaseModel):
        model_config = ConfigDict(extra='forbid')
        reason: str = Field(min_length=1, max_length=8)
        position: int = Field(ge=1, le=25)

    class BoundedReport(BaseModel):
        model_config = ConfigDict(extra='forbid')
        entries: list[Item] = Field(min_length=1, max_length=2)

    original = convert_to_openai_tool(BoundedReport)['function']['parameters']
    before = copy.deepcopy(original)
    wire = canonical_wire_schema(original)
    assert original == before
    assert wire['required'] == original['required'] and wire['additionalProperties'] is False
    entries = wire['properties']['entries']
    assert 'minItems' not in entries and 'maxItems' not in entries
    assert 'minimum 1 items' in entries['description'] and 'maximum 2 items' in entries['description']
    reason = entries['items']['properties']['reason']
    assert 'minLength' not in reason and 'maxLength' not in reason
    assert 'maximum 8 characters' in reason['description']
    assert entries['items']['properties']['position'] == original['properties']['entries']['items']['properties']['position']
    assert canonical_wire_schema({'type': 'string', 'enum': ['a', 'b'], 'maxLength': 1})['enum'] == ['a', 'b']
    constant = {'type': 'array', 'maxItems': 17}
    assert canonical_wire_schema({'enum': [constant], 'default': constant}) == {'enum': [constant], 'default': constant}
    for value in ({'entries': []}, {'entries': [{'reason': 'too long here', 'position': 1}]},
                  {'entries': [{'reason': 'ok', 'position': 1}] * 3}):
        with pytest.raises(ValidationError):
            BoundedReport.model_validate(value)

    requests = []
    ledger = gateway(tmp_path, lambda request: requests.append(request) or response())
    LedgerAgentModel(ledger, {'role': 'single'}, 2).bind_tools([BoundedReport]).invoke('task')
    actual = requests[0]['tool_declarations'][0]['parameters']
    assert actual == wire
    stored = json.loads(ledger.db.execute('SELECT request_json FROM attempts').fetchone()[0])
    assert stored['tool_declarations'][0]['parameters'] == wire
    changed = copy.deepcopy(before)
    changed['properties']['entries']['maxItems'] = 3
    LedgerAgentModel(ledger, {'role': 'single'}, 2).bind_tools([
        {'name': 'BoundedReport', 'parameters': changed}]).invoke('task')
    assert ledger.summary()['provider_attempts'] == 2
    ledger.close()


def test_actual_tool_strategy_still_rejects_overlength_report_after_wire_compilation(tmp_path):
    from langchain.agents.structured_output import StructuredOutputValidationError

    class SmallReport(BaseModel):
        reason: str = Field(max_length=3)

    ledger = gateway(tmp_path, lambda _: response(final_text='', tool_calls=[
        {'name': 'SmallReport', 'args': {'reason': 'outside the local length limit'}}]))
    model = LedgerAgentModel(ledger, {'role': 'single'}, 2)
    agent = create_agent(model=model, tools=[], response_format=ToolStrategy(SmallReport, handle_errors=False))
    with pytest.raises(StructuredOutputValidationError):
        agent.invoke({'messages': [HumanMessage('task')]})
    assert len(model.records) == 1 and ledger.summary()['provider_attempts'] == 1
    assert len(model.tool_call_records) == 1
    ledger.close()
