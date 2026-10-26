import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from evagent.integrated.gateway import ExecutionPaused, ProviderStreamInterrupted, digest
from evagent.integrated.research.concurrent_gateway import ConcurrentGateway
from evagent.integrated.research.campaign import prepare, run_campaign
from evagent.integrated.research.contracts import save
from evagent.integrated.research.parallel_resume import freeze_reference, load_resume
from evagent.integrated.tasks import file_hash
from test_research_agents import MODEL, fixture, response, transport


def gateway(tmp_path, fn, **kwargs):
    options = dict(model_config=MODEL, run_dir=tmp_path, fingerprint='parallel-test',
                   token_budget=None, budget_mode='unlimited', transport=fn)
    options.update(kwargs)
    return ConcurrentGateway(**options)


def call(g, text):
    return g.generate(system='test', user=text, run_name='test')


def test_real_overlap_and_same_request_dedup(tmp_path):
    barrier = threading.Barrier(4)
    def fn(r):
        barrier.wait(timeout=5)
        return response('ok')
    g = gateway(tmp_path, fn)
    with ThreadPoolExecutor(8) as pool:
        results = list(pool.map(lambda i: call(g, str(i)), [0, 1, 2, 3, 0, 1, 2, 3]))
    info = g.summary()
    assert info['provider_attempts'] == 4 and info['reported_tokens'] == 80
    assert info['peak_concurrent_requests'] == 4 and info['inflight_requests'] == 0
    assert sum(x['cache_hit'] for x in results) == 4
    g.close()


def test_pending_reservations_stop_finite_oversubscription(tmp_path):
    entered, release = threading.Event(), threading.Event()
    def fn(r):
        entered.set()
        assert release.wait(5)
        return response('ok')
    g = gateway(tmp_path, fn, model_config={**MODEL, 'max_output_tokens': 100},
                token_budget=1800, budget_mode='limited', reserve_budget=True)
    with ThreadPoolExecutor(2) as pool:
        first = pool.submit(call, g, 'one')
        assert entered.wait(5)
        with pytest.raises(ExecutionPaused):
            call(g, 'two')
        release.set()
        assert first.result()['status'] == 'ok'
    assert g.summary()['provider_attempts'] == 1 and g.summary()['paused']
    assert g.summary()['reported_tokens'] == 20
    g.close()


def interrupted():
    return ProviderStreamInterrupted({'response_metadata': {
        'transport_error': {'exception_type': 'RemoteProtocolError'}}})


POLICY = {'max_attempts': 3, 'backoff_seconds': [15, 30],
          'authorization': 'Explicit test authorization for bounded connection retries'}


def test_retry_cannot_clear_other_threads_hard_pause(tmp_path, monkeypatch):
    barrier = threading.Barrier(2)
    failed = threading.Event()
    def fn(r):
        barrier.wait(5)
        if r['messages'][-1]['content'] == 'connection':
            assert failed.wait(5)
            raise interrupted()
        failed.set()
        return {'final_text': 'missing usage and finish'}
    monkeypatch.setattr('evagent.integrated.research.concurrent_gateway.time.sleep', lambda _: None)
    g = gateway(tmp_path, fn, connection_retry_policy=POLICY)
    with ThreadPoolExecutor(2) as pool:
        jobs = [pool.submit(call, g, x) for x in ('connection', 'hard')]
        for job in jobs:
            with pytest.raises(ExecutionPaused):
                job.result()
    assert g.summary()['paused'] and g.summary()['provider_attempts'] == 2
    with pytest.raises(ExecutionPaused):
        call(g, 'never submitted')
    g.close()


def test_exactly_three_connection_attempts_then_global_pause(tmp_path, monkeypatch):
    monkeypatch.setattr('evagent.integrated.research.concurrent_gateway.time.sleep', lambda _: None)
    def fn(r):
        raise interrupted()
    g = gateway(tmp_path, fn, connection_retry_policy=POLICY)
    with pytest.raises(ExecutionPaused):
        call(g, 'three')
    assert g.summary()['provider_attempts'] == 3 and g.summary()['unknown_usage_attempts'] == 3
    assert g.summary()['paused']
    with pytest.raises(ExecutionPaused):
        call(g, 'four')
    assert g.summary()['provider_attempts'] == 3
    g.close()


def test_partial_cache_uses_exact_messages_and_does_not_charge_twice(tmp_path, monkeypatch):
    monkeypatch.setattr('evagent.integrated.research.concurrent_gateway.time.sleep', lambda _: None)
    g = gateway(tmp_path, lambda r: response('new'), connection_retry_policy=POLICY)
    metadata = {'transport_attempt': 1}
    request = g._request([{'role': 'system', 'content': 'test'}, {'role': 'user', 'content': 'cached'}], metadata)
    key = digest({k: v for k, v in request.items() if k != 'fingerprint'})
    g.previous_cache[key] = {'response': {**response('old'), 'status': 'ok', 'request_sha256': 'old-hash'},
                             'source': {'request_sha256': 'old-hash'}}
    assert call(g, 'cached')['final_text'] == 'old'
    assert g.summary()['provider_attempts'] == 0
    assert call(g, 'changed')['final_text'] == 'new'
    assert g.summary()['provider_attempts'] == 1
    g.close()


def test_inherited_failed_attempt_is_not_resubmitted(tmp_path, monkeypatch):
    monkeypatch.setattr('evagent.integrated.research.concurrent_gateway.time.sleep', lambda _: None)
    def forbidden(r):
        pytest.fail('Historical retry chain must not resubmit')
    g = gateway(tmp_path, forbidden, connection_retry_policy=POLICY)
    for ordinal in (1, 2):
        req = g._request([{'role': 'system', 'content': 'test'}, {'role': 'user', 'content': 'old-chain'}],
                         {'transport_attempt': ordinal})
        value = ({'response_metadata': {'transport_error': {'exception_type': 'RemoteProtocolError'}},
                  'status': 'failed_unknown_cost'} if ordinal == 1 else {**response('old'), 'status': 'ok'})
        g.previous_cache[digest({k: v for k, v in req.items() if k != 'fingerprint'})] = {
            'response': {**value, 'request_sha256': str(ordinal)}, 'retry_acknowledged': ordinal == 1,
            'source': {'request_sha256': str(ordinal)}}
    result = call(g, 'old-chain')
    assert result['final_text'] == 'old' and len(result['transport_attempt_records']) == 2
    assert g.summary()['provider_attempts'] == 0
    g.close()


def parallel_config(p):
    config = json.loads(p.read_text())
    config['execution'] = {'unit_workers': 2, 'max_concurrent_requests': 4, 'parallel_specialists': True}
    save(p, config)
    return config


def test_parallel_full_matrix_restart_and_unchanged_prompts(tmp_path):
    p = fixture(tmp_path)
    seen_serial, seen_parallel = [], []
    serial = run_campaign(p, transport=lambda r: (seen_serial.append(r), transport(r))[1])
    config = parallel_config(p)
    config['output_dir'] = str(tmp_path / 'parallel')
    save(p, config)
    def fn(r):
        seen_parallel.append(r)
        time.sleep(.002)
        return transport(r)
    parallel = run_campaign(p, transport=fn)
    assert parallel['status'] == 'complete' and parallel['completed_units'] == 18
    assert parallel['statuses'] == serial['statuses'] == {'ok': 9}
    assert parallel['ledger']['peak_concurrent_requests'] >= 2
    def signatures(requests):
        return sorted(digest({k: v for k, v in r.items() if k != 'fingerprint'}) for r in requests)
    assert signatures(seen_parallel) == signatures(seen_serial)
    again = run_campaign(p, transport=lambda _: pytest.fail('Already completed'))
    assert again['ledger']['reported_tokens'] == parallel['ledger']['reported_tokens']


def test_parallel_unknown_pauses_and_drains_inflight(tmp_path):
    p = fixture(tmp_path)
    parallel_config(p)
    def fn(r):
        time.sleep(.005)
        return {} if r['metadata']['condition'] == 'single' else transport(r)
    report = run_campaign(p, transport=fn)
    assert report['status'] == 'paused' and report['ledger']['inflight_requests'] == 0
    assert report['ledger']['provider_attempts'] <= 2
    assert report['completed_units'] <= 1


def test_audited_resume_preserves_units_costs_and_rejects_science_change(tmp_path):
    p = fixture(tmp_path)
    old = run_campaign(p, transport=transport, stop_after=7)
    root = tmp_path / 'out'
    reference = freeze_reference(root, Path(__file__).resolve().parents[1])
    config = parallel_config(p)
    config.update(output_dir=str(tmp_path / 'resumed'), resume_from=reference,
                  prior_ledgers=[{'path': str(root / 'provider/request_ledger.sqlite'),
                                  'sha256': reference['ledger_sha256']}])
    save(p, config)
    dry = run_campaign(p, dry_run=True)
    assert dry['inherited_units'] == 7 and dry['api_calls'] == 0
    resumed = run_campaign(p, transport=transport)
    assert resumed['status'] == 'complete' and resumed['inherited_units'] == 7
    assert resumed['ledger']['prior_reported_tokens'] == old['ledger']['reported_tokens']
    assert len([x for x in (tmp_path / 'resumed/units').glob('*.json')
                if json.loads(x.read_text()).get('resume_source')]) == 7
    manifest, _ = prepare(p)
    for field in ('model_config', 'tasks', 'analysis_limits'):
        changed = json.loads(json.dumps(manifest))
        changed[field] = {'changed': True}
        with pytest.raises(ValueError, match='scientific configuration'):
            load_resume(changed)
    checkpoint = next((root / 'units').glob('*.json'))
    checkpoint.write_text(checkpoint.read_text() + '\n')
    with pytest.raises(ValueError, match='artifacts changed'):
        load_resume(manifest)
