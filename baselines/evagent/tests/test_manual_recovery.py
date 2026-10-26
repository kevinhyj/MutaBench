"""Scoped operator recovery retains unknown costs; future ambiguity still stops."""
import json
from types import SimpleNamespace as NS

import pytest

from evagent.integrated.campaign import read_prior_ledgers
from evagent.integrated.gateway import (
    ExecutionPaused,
    ProviderStreamInterrupted,
    RequestGateway,
    google_stream_transport,
    public_model_config,
)
from evagent.integrated.tasks import file_hash

MODEL = {'model': 'test', 'thinking_level': 'medium', 'generation_seed': 1}


def test_manual_recovery_requires_exact_hash_and_keeps_new_pause(tmp_path):
    old = RequestGateway(model_config=MODEL, run_dir=tmp_path/'old', fingerprint='old',
                         token_budget=None, budget_mode='unlimited', transport=lambda _: {})
    with pytest.raises(ExecutionPaused):
        old.generate(system='test', user='test', run_name='old')
    sha = old.db.execute('SELECT request_sha256 FROM attempts').fetchone()[0]
    path = old.path
    old.close()
    record = {'path': str(path), 'sha256': file_hash(path)}
    with pytest.raises(ValueError, match='Unknown prior'):
        read_prior_ledgers([record], tmp_path, allow_manual_recovery=True)
    record['manual_recovery'] = {
        'request_sha256s': [sha], 'authorization': 'Operator explicitly requests recovery',
        'recorded_at': '2026-09-12T00:00:00Z', 'usage_status': 'unresolved_lower_bound',
    }
    with pytest.raises(ValueError, match='Invalid scoped'):
        read_prior_ledgers([record], tmp_path)
    prior = read_prior_ledgers([record], tmp_path, allow_manual_recovery=True)
    assert prior['unknown_usage_attempts'] == 1
    assert prior['reported_tokens'] == 0
    assert file_hash(path) == record['sha256']
    with pytest.raises(ValueError, match='finite budget'):
        RequestGateway(model_config=MODEL, run_dir=tmp_path/'finite', fingerprint='f',
                       token_budget=100, prior_usage=prior)
    calls = []
    new = RequestGateway(model_config=MODEL, run_dir=tmp_path/'new', fingerprint='new',
                         token_budget=None, budget_mode='unlimited', prior_usage=prior,
                         transport=lambda q: calls.append(q) or {})
    assert new.summary()['usage_is_lower_bound']
    for _ in range(2):
        with pytest.raises(ExecutionPaused):
            new.generate(system='test', user='test', run_name='new')
    assert len(calls) == 1
    assert new.summary()['unknown_usage_attempts'] == 2
    new.close()
    record['manual_recovery']['request_sha256s'] = ['wrong']
    with pytest.raises(ValueError, match='Invalid scoped'):
        read_prior_ledgers([record], tmp_path, allow_manual_recovery=True)


@pytest.mark.parametrize('mode', ['stream', 'nonstream'])
@pytest.mark.parametrize('fail', [False, True])
def test_transport_single_submission_and_redacted_errors(monkeypatch, mode, fail):
    import google.genai
    captured = []
    class Client:
        def __init__(self, **kw):
            assert kw['http_options'].retry_options.attempts == 1
            self.models = self
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def response(self, **kw):
            captured.append(kw)
            assert kw['config'].seed == 1
            assert kw['config'].automatic_function_calling.disable
            assert kw['config'].tool_config.function_calling_config.mode.value == 'NONE'
            assert kw['config'].response_mime_type == 'application/json'
            if fail:
                raise ConnectionError('sensitive URL and secret must not be persisted')
            return NS(usage_metadata=NS(prompt_token_count=8, candidates_token_count=2,
                      total_token_count=10, thoughts_token_count=0),
                      candidates=[NS(finish_reason='STOP', content=NS(parts=[
                          NS(text='OK', thought=False, function_call=None)]))],
                      prompt_feedback=None, model_version='test', response_id='response-1')
        def generate_content(self, **kw):
            assert mode == 'nonstream'
            return self.response(**kw)
        def generate_content_stream(self, **kw):
            assert mode == 'stream'
            yield self.response(**kw)
    monkeypatch.setattr(google.genai, 'Client', Client)
    monkeypatch.setenv('GEMINI_API_KEY', 'fake-key')
    cfg = public_model_config({**MODEL, 'transport_mode': mode, 'function_calling_mode': 'NONE', 'response_mime_type': 'application/json'})
    request = {'model': 'test', 'model_config': cfg, 'messages': [{'role': 'user', 'content': 'OK'}]}
    if fail:
        with pytest.raises(ProviderStreamInterrupted) as err:
            google_stream_transport(request)
        metadata = err.value.partial_response['response_metadata']
        assert metadata['transport_error']['exception_type'] == 'ConnectionError'
        assert 'sensitive' not in json.dumps(metadata)
    else:
        assert google_stream_transport(request)['response_metadata']['response_id'] == 'response-1'
    assert len(captured) == 1


def test_transport_default_preserves_old_config():
    assert 'transport_mode' not in public_model_config(MODEL)
    with pytest.raises(ValueError, match='transport_mode'):
        public_model_config({**MODEL, 'transport_mode': 'auto_retry'})

POLICY = {'max_attempts': 3, 'backoff_seconds': [15, 30], 'authorization': 'User explicitly permits two connection retries'}


def connection_failure():
    raise ProviderStreamInterrupted({'response_metadata': {'transport_error': {'exception_type': 'RemoteProtocolError'}}, 'usage_summary': None})


def good_response():
    return {'final_text': 'OK', 'finish_reason': 'STOP', 'usage_summary': {
        'input_tokens': 8, 'output_tokens': 2, 'reasoning_tokens': 0, 'total_tokens': 10, 'model_calls': 1}}


def retry_gateway(path, transport):
    return RequestGateway(model_config=MODEL, run_dir=path, fingerprint='retry-v1', token_budget=None,
                          budget_mode='unlimited', transport=transport, connection_retry_policy=POLICY)


def test_two_connection_retries_then_global_pause(tmp_path, monkeypatch):
    monkeypatch.setattr('evagent.integrated.gateway.time.sleep', lambda _: None)
    calls = []
    def provider(q):
        calls.append(q)
        connection_failure()
    gateway = retry_gateway(tmp_path, provider)
    for user in ['first', 'next']:
        with pytest.raises(ExecutionPaused):
            gateway.generate(system='test', user=user, run_name='retry')
    assert len(calls) == 3
    assert [q['metadata']['transport_attempt'] for q in calls] == [1, 2, 3]
    assert gateway.summary()['unknown_usage_attempts'] == 3
    assert gateway.summary()['paused']
    gateway.close()
    gateway = retry_gateway(tmp_path, provider)
    with pytest.raises(ExecutionPaused):
        gateway.generate(system='test', user='first', run_name='retry')
    assert len(calls) == 3
    gateway.close()


def test_retries_count_toward_specialist_cap_and_replay_without_charge(tmp_path, monkeypatch):
    from evagent.integrated.research.contracts import BoundedModel, CallLimit
    monkeypatch.setattr('evagent.integrated.gateway.time.sleep', lambda _: None)
    calls = []
    def provider(q):
        calls.append(q)
        if len(calls) == 1:
            connection_failure()
        return good_response()
    gateway = retry_gateway(tmp_path, provider)
    for _ in range(2):
        model = BoundedModel(gateway, {}, 2)
        assert model.generate('test', 'test') == 'OK'
        assert len(model.records) == 2
        with pytest.raises(CallLimit):
            model.generate('test', 'next')
    assert len(calls) == 2
    assert gateway.summary()['unknown_usage_attempts'] == 1
    assert not gateway.summary()['paused']
    gateway.close()


def test_other_unknowns_never_retry_under_connection_policy(tmp_path, monkeypatch):
    monkeypatch.setattr('evagent.integrated.gateway.time.sleep', lambda _: None)
    calls = []
    gateway = retry_gateway(tmp_path, lambda q: calls.append(q) or {})
    with pytest.raises(ExecutionPaused):
        gateway.generate(system='test', user='test', run_name='test')
    assert len(calls) == 1
    gateway.close()


def test_connection_retry_cannot_exceed_remaining_expert_calls(tmp_path, monkeypatch):
    monkeypatch.setattr('evagent.integrated.gateway.time.sleep', lambda _: None)
    calls = []
    def provider(q):
        calls.append(q)
        connection_failure()
    gateway = retry_gateway(tmp_path, provider)
    with pytest.raises(ExecutionPaused):
        gateway.generate(system='test', user='test', run_name='test', metadata={'provider_attempt_limit': 1})
    assert len(calls) == 1
    gateway.close()


def test_cancellation_during_backoff_keeps_pause(tmp_path, monkeypatch):
    def cancel(_):
        raise KeyboardInterrupt()
    monkeypatch.setattr('evagent.integrated.gateway.time.sleep', cancel)
    gateway = retry_gateway(tmp_path, lambda _: connection_failure())
    with pytest.raises(ExecutionPaused, match='retry interrupted'):
        gateway.generate(system='test', user='test', run_name='test')
    assert gateway.summary()['provider_attempts'] == 1
    assert gateway.summary()['paused']
    gateway.close()


@pytest.mark.parametrize('protocol,role,name,finish,expected', [
    (None, 'msa', 'evidence', 'STOP', 'terminal_failure'),
    ('native_v1', 'msa', 'evidence', 'STOP', 'ok'),
    ('native_v1', 'msa', 'literature_search', 'STOP', 'terminal_failure'),
    ('native_v1', 'msa', 'evidence', 'MALFORMED_FUNCTION_CALL', 'terminal_failure'),
])
def test_native_calls_require_explicit_role_scope(tmp_path, protocol, role, name, finish, expected):
    model = {**MODEL, **({'research_tool_protocol': protocol} if protocol else {})}
    raw = {**good_response(), 'final_text': '', 'finish_reason': finish,
           'tool_calls': [{'name': name, 'args': {'source': 'msa'}, 'id': 'call1'}]}
    gateway = RequestGateway(model_config=model, run_dir=tmp_path, fingerprint='native',
                             token_budget=100, transport=lambda _: raw)
    result = gateway.generate(system='test', user='test', run_name='test', metadata={'phase':'analysis','role':role})
    assert result['status'] == expected
    if expected == 'ok':
        assert json.loads(result['final_text']) == {'tool':'evidence', 'arguments':{'source':'msa'}}
        assert result['tool_calls'] == raw['tool_calls']
    gateway.close()


def test_native_declarations_sent_only_to_research_analysis(monkeypatch):
    import google.genai
    calls = []
    class Client:
        def __init__(self, **kw):
            self.models = self
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def generate_content_stream(self, **kw):
            calls.append(kw['config'])
            assert kw['config'].automatic_function_calling.disable
            yield NS(usage_metadata=NS(prompt_token_count=8,candidates_token_count=2,total_token_count=10,thoughts_token_count=0),
                     candidates=[NS(finish_reason='STOP',content=NS(parts=[NS(text='OK',thought=False,function_call=None)]))],
                     prompt_feedback=None,model_version='test')
    monkeypatch.setattr(google.genai,'Client',Client)
    monkeypatch.setenv('GEMINI_API_KEY','fake')
    config=public_model_config({**MODEL,'research_tool_protocol':'native_v1'})
    for phase in ('analysis','integration','paperqa_evidence'):
        google_stream_transport({'model':'test','model_config':config,'messages':[{'role':'user','content':'test'}],
                                 'metadata':{'phase':phase,'role':'literature'}})
    assert {d.name for d in calls[0].tools[0].function_declarations} == {'evidence','literature_search'}
    for generation in calls[1:]:
        assert not generation.tools
        assert generation.tool_config.function_calling_config.mode.value == 'NONE'
