"""Bounded concurrent I/O with serialized durable ledger admission and settlement."""
from __future__ import annotations

import json
import threading
import time

from ..gateway import (
    AdmissionPaused,
    ExecutionPaused,
    ProviderStreamInterrupted,
    RequestGateway,
    _classify,
    assert_public,
    digest,
)

CONNECTION_ERRORS = {'RemoteProtocolError', 'ReadError', 'WriteError', 'ConnectError', 'ConnectionError'}


class ConcurrentGateway(RequestGateway):
    def __init__(self, *, max_concurrent_requests=4, previous_cache=None, **kwargs):
        if type(max_concurrent_requests) is not int or not 1 <= max_concurrent_requests <= 64:
            raise ValueError('Concurrency must be between 1 and 64')
        if kwargs.get('budget_mode', 'limited') != 'unlimited' and not kwargs.get('reserve_budget'):
            raise ValueError('Concurrent finite budgets require conservative reservations')
        super().__init__(**kwargs)
        self.capacity = max_concurrent_requests
        self.slots = threading.BoundedSemaphore(self.capacity)
        self.logical_locks = {}
        self.reservations = {}
        self.previous_cache = previous_cache or {}
        self.db.execute('CREATE TABLE IF NOT EXISTS inherited_cache (request_sha256 TEXT PRIMARY KEY, source_json TEXT NOT NULL)')
        state = dict(self.db.execute('SELECT key,value FROM run_state'))
        if 'max_concurrent_requests' in state and int(state['max_concurrent_requests']) != self.capacity:
            self.close()
            raise ValueError('Concurrency is immutable in a run directory')
        self.db.execute('INSERT OR IGNORE INTO run_state VALUES (?,?)', ('max_concurrent_requests', str(self.capacity)))
        self.db.commit()

    def summary(self):
        result = super().summary()
        with self.lock:
            row = self.db.execute("SELECT value FROM run_state WHERE key='peak_concurrent_requests'").fetchone()
            result.update(max_concurrent_requests=getattr(self, 'capacity', 1),
                          inflight_requests=self.db.execute("SELECT COUNT(*) FROM attempts WHERE status='inflight'").fetchone()[0],
                          peak_concurrent_requests=int(row[0]) if row else 0)
        return result

    def invoke_messages(self, *, messages, run_name, metadata=None):
        metadata = metadata or {}
        assert_public(metadata)
        cap = metadata.get('provider_attempt_limit', 3)
        if type(cap) is not int or cap < 1:
            raise ValueError('provider_attempt_limit must be a positive integer')
        logical = digest(self._request(messages, metadata))
        with self.lock:
            mutex = self.logical_locks.setdefault(logical, threading.Lock())
        # Same logical request is deduplicated; unrelated network I/O never holds the DB lock.
        with mutex, self.slots:
            attempts = min(3, cap) if self.connection_retry_policy else 1
            records = []
            for ordinal in range(attempts):
                scoped = {**metadata, 'transport_attempt': ordinal + 1} if self.connection_retry_policy else metadata
                key = digest(self._request(messages, scoped))
                try:
                    response = self._once(messages, run_name, scoped, ordinal + 1 < attempts)
                    records.append({k: response.get(k) for k in (
                        'request_sha256', 'status', 'cache_hit', 'usage_summary', 'elapsed_s')})
                    return {**response, 'transport_attempt_records': records} if self.connection_retry_policy else response
                except ExecutionPaused:
                    with self.lock:
                        row = self.db.execute('SELECT status,response_json,elapsed_s FROM attempts WHERE request_sha256=?', (key,)).fetchone()
                        inherited = None
                        if row is None:
                            payload = self._request(messages, scoped)
                            inherited = self.previous_cache.get(digest({k: v for k, v in payload.items() if k != 'fingerprint'}))
                            if inherited and inherited.get('retry_acknowledged'):
                                response = inherited['response']
                                row = (response['status'], json.dumps(response), response.get('elapsed_s'))
                        raw = json.loads(row[1] or '{}') if row else {}
                        error = raw.get('response_metadata', {}).get('transport_error', {}).get('exception_type')
                        if (ordinal + 1 >= attempts or not row or row[0] != 'failed_unknown_cost'
                                or error not in CONNECTION_ERRORS or self.summary()['paused']
                                or self._marker()):
                            raise
                        records.append({'request_sha256': inherited['source']['request_sha256'] if inherited else key,
                                        'status': row[0], 'cache_hit': bool(inherited),
                                        'usage_summary': raw.get('usage_summary'), 'elapsed_s': row[2]})
                        self.db.execute('INSERT OR IGNORE INTO run_state VALUES (?,?)',
                                        ('connection_retry_ack:' + key, json.dumps({
                                            'policy': self.connection_retry_policy, 'time': time.time(),
                                            'usage_status': 'unresolved_lower_bound', 'next_attempt': ordinal + 2})))
                        self.db.commit()
                    try:
                        if not inherited:
                            time.sleep(self.connection_retry_policy['backoff_seconds'][ordinal])
                    except BaseException as exc:
                        with self.lock:
                            self._pause('Connection retry interrupted before next submission')
                            self.db.commit()
                        raise ExecutionPaused('Connection retry interrupted') from exc
            raise AssertionError('Unreachable retry state')

    def _marker(self):
        return (self.run_dir / 'PAUSE').exists() or (self.run_dir.parent / 'PAUSE').exists()

    def _once(self, messages, run_name, metadata, can_retry):
        request = self._request(messages, metadata)
        key = digest(request)
        with self.lock:
            self.db.execute('BEGIN IMMEDIATE')
            old = self.db.execute('SELECT status,response_json FROM attempts WHERE request_sha256=?', (key,)).fetchone()
            if old and old[0] in {'ok', 'blocked', 'truncated', 'terminal_failure'}:
                self.db.commit()
                return {**json.loads(old[1]), 'cache_hit': True}
            cache_key = digest({k: v for k, v in request.items() if k != 'fingerprint'})
            cached = self.previous_cache.get(cache_key)
            if not old and cached:
                self.db.execute('INSERT OR IGNORE INTO inherited_cache VALUES (?,?)', (key, json.dumps(cached['source'])))
                self.db.commit()
                if cached['response']['status'] == 'failed_unknown_cost':
                    raise ExecutionPaused('Inherited acknowledged connection attempt')
                return {**cached['response'], 'cache_hit': True, 'cache_source': cached['source']}
            info = self.summary()
            if info['paused'] or self._marker():
                self.db.commit()
                raise AdmissionPaused(info['pause_reason'] or 'Campaign PAUSE marker')
            if old:
                self.db.commit()
                raise ExecutionPaused('Prior unresolved attempt')
            reserve = 0
            if self.token_budget is not None:
                maximum = self.model_config['max_output_tokens']
                if maximum is None:
                    self.db.rollback()
                    raise ValueError('Concurrent finite admission requires max_output_tokens')
                reserve = sum(len(m['content'].encode()) for m in messages) + 2 * maximum + 1024
                if info['reported_tokens'] + sum(self.reservations.values()) + reserve > self.token_budget:
                    self._pause('Insufficient remaining budget including inflight reservations')
                    self.db.commit()
                    raise AdmissionPaused('Concurrent budget reserve exhausted')
            started = time.time()
            cursor = self.db.execute(
                "INSERT INTO attempts(request_sha256,run_name,status,request_json,started) VALUES (?,?,'inflight',?,?)",
                (key, run_name, json.dumps(request), started))
            aid = cursor.lastrowid
            self.reservations[aid] = reserve
            inflight = self.db.execute("SELECT COUNT(*) FROM attempts WHERE status='inflight'").fetchone()[0]
            peak = max(inflight, info['peak_concurrent_requests'])
            self.db.execute('INSERT OR REPLACE INTO run_state VALUES (?,?)', ('peak_concurrent_requests', str(peak)))
            self.db.commit()
        try:
            raw = self.transport(request)
        except BaseException as exc:
            with self.lock:
                self.reservations.pop(aid, None)
                if isinstance(exc, AdmissionPaused):
                    self.db.execute("UPDATE attempts SET status='not_submitted',tokens=0,ended=? WHERE id=?", (time.time(), aid))
                    self._pause('Outer admission controller stopped submission')
                else:
                    partial = exc.partial_response if isinstance(exc, ProviderStreamInterrupted) else None
                    tokens = ((partial.get('usage_summary') or {}).get('total_tokens') if partial else None)
                    if type(tokens) is not int or tokens < 0:
                        tokens = None
                    self.db.execute("UPDATE attempts SET status='failed_unknown_cost',response_json=?,tokens=?,ended=?,elapsed_s=?,error_type=? WHERE id=?",
                                    (json.dumps(partial) if partial else None, tokens, time.time(), time.time()-started, type(exc).__name__, aid))
                    error = (partial or {}).get('response_metadata', {}).get('transport_error', {}).get('exception_type')
                    # Approved recoverable connection failures do not erase another request's hard pause.
                    if not (can_retry and error in CONNECTION_ERRORS and not self._marker()):
                        self._pause('Provider attempt failed with unknown response/cost')
                self.db.commit()
            raise ExecutionPaused('Provider attempt paused; inspect ledger') from exc
        with self.lock:
            self.reservations.pop(aid, None)
            permitted = False
            if (self.model_config.get('research_tool_protocol') == 'native_v1'
                    and metadata.get('phase') == 'analysis' and isinstance(raw, dict)
                    and isinstance(raw.get('tool_calls'), list) and len(raw['tool_calls']) == 1):
                from .native_tools import declarations
                call = raw['tool_calls'][0]
                if isinstance(call, dict) and call.get('name') in {d['name'] for d in declarations(metadata.get('role'))} and isinstance(call.get('args'), dict):
                    raw = {**raw, 'provider_final_text': raw.get('final_text', ''),
                           'final_text': json.dumps({'tool': call['name'], 'arguments': call['args']})}
                    permitted = True
            reason, tokens, status = _classify(raw, allow_tool_call=permitted)
            serializable = raw if isinstance(raw, dict) else {'response_type': type(raw).__name__}
            result = {**serializable, 'request_sha256': key, 'cache_hit': False,
                      'elapsed_s': time.time()-started, 'status': status}
            self.db.execute('UPDATE attempts SET status=?,response_json=?,tokens=?,ended=?,elapsed_s=? WHERE id=?',
                            (status, json.dumps(result, default=str), tokens, time.time(), result['elapsed_s'], aid))
            if reason:
                self._pause(reason)
            self.db.commit()
            if reason:
                raise ExecutionPaused(reason)
            return result
