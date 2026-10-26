"""Bounded unit scheduling with one durable, concurrent provider gateway."""
from __future__ import annotations

import fcntl
import json
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import contextmanager
from pathlib import Path

from ..experiment import _implementation
from ..gateway import ExecutionPaused, digest
from ..tasks import file_hash, load_task
from .concurrent_gateway import ConcurrentGateway
from .contracts import ResearchFailure, load_snapshot, save
from .parallel_resume import load_resume


def validate_execution(config):
    if (not isinstance(config, dict)
            or set(config) != {'unit_workers', 'max_concurrent_requests', 'parallel_specialists'}
            or type(config['unit_workers']) is not int or not 1 <= config['unit_workers'] <= 8
            or type(config['max_concurrent_requests']) is not int
            or not 1 <= config['max_concurrent_requests'] <= 16
            or type(config['parallel_specialists']) is not bool):
        raise ValueError('Invalid explicit parallel execution configuration')


@contextmanager
def draining_pool(gateway, workers):
    pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix='cr-unit')
    try:
        yield pool
    except BaseException:
        # Close admission before waiting on siblings after an operator interruption.
        with gateway.lock:
            gateway._pause('Parallel scheduler interrupted; settling outstanding requests')
            gateway.db.commit()
        raise
    finally:
        pool.shutdown(wait=True, cancel_futures=True)


def run_parallel_campaign(path, manifest, root, *, dry_run=False, transport=None,
                          stop_after=None, selection=None):
    from .campaign import all_units, evaluate, run_unit, unit_key
    from .reporting import audit_report

    if selection is not None:
        raise ValueError('Parallel execution currently requires the frozen whole campaign')
    settings = manifest['execution']
    validate_execution(settings)
    units = all_units(manifest)
    inherited, cache, inherited_calibration = load_resume(manifest)
    if dry_run:
        return {'status': 'dry_run', 'tasks': len(manifest['tasks']),
                'initial_CR_units': len(units), 'method_results': 9 * sum(
                    1 if r['maximum_n'] == 50 else 2 for r in manifest['tasks']),
                'fingerprint': manifest['fingerprint'], 'api_calls': 0, 'writes': 0,
                'execution': settings, 'inherited_units': len(inherited),
                'inherited_response_cache': len(cache)}
    root.mkdir(parents=True, exist_ok=True)
    with (root / 'campaign.lock').open('a+') as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        mpath = root / 'manifest.json'
        if mpath.exists() and json.loads(mpath.read_text())['fingerprint'] != manifest['fingerprint']:
            raise ValueError('Frozen campaign changed; select a new output directory')
        save(mpath, manifest)
        completed = {}
        expected = {unit_key(*u) for u in units}
        for p in sorted((root / 'units').glob('*.json')):
            item = json.loads(p.read_text())
            if (p.stem not in expected or item.get('fingerprint') != manifest['fingerprint']
                    or item.get('sha256') != digest({k: v for k, v in item.items() if k != 'sha256'})):
                raise ValueError('Checkpoint integrity failure')
            completed[p.stem] = item
        for key, item in inherited.items():
            if key in completed and completed[key] != item:
                raise ValueError('Inherited checkpoint changed')
            if key not in completed:
                save(root / 'units' / (key + '.json'), item)
                completed[key] = item
        if inherited_calibration:
            save(root / 'calibration.json', inherited_calibration)
        gateway = ConcurrentGateway(
            model_config=manifest['model_config'], run_dir=root / 'provider',
            fingerprint=manifest['fingerprint'], token_budget=manifest.get('token_budget'),
            budget_mode=manifest['budget_mode'], transport=transport,
            prior_usage=manifest['prior_usage'], reserve_budget=manifest['budget_mode'] == 'limited',
            connection_retry_policy=manifest.get('connection_retry_policy'),
            max_concurrent_requests=settings['max_concurrent_requests'], previous_cache=cache)
        gateway.parallel_specialists = settings['parallel_specialists']
        pending = {}
        status, reason = 'complete', None
        calibrated = inherited_calibration is not None
        calibration_keys = {unit_key(*u) for u in units[:6]}

        def validate_inputs(unit):
            if (root / 'PAUSE').exists() or gateway.summary()['paused']:
                raise ExecutionPaused(gateway.summary()['pause_reason'] or 'Campaign PAUSE marker')
            if file_hash(Path(path)) != manifest['config_sha256'] or _implementation() != manifest['implementation']:
                raise ExecutionPaused('Frozen configuration or implementation changed')
            if manifest.get('environment_file') and file_hash(Path(manifest['environment_file'])) != manifest['environment_sha256']:
                raise ExecutionPaused('Frozen environment manifest changed')
            r = unit[0]
            task = load_task(r)
            if task.input_hashes != r['input_hashes']:
                raise ExecutionPaused('Frozen task context or sample changed')
            snapshot = load_snapshot(Path(r['snapshot_file']), task)
            if snapshot['sha256'] != r['snapshot_sha256']:
                raise ExecutionPaused('Snapshot changed')
            return task, snapshot

        def worker(unit, task, snapshot):
            try:
                return run_unit(*unit, gateway, snapshot, task)
            except ResearchFailure as exc:
                return {'assay': unit[0]['assay'], 'repeat': unit[1], 'condition': unit[2],
                        'group': unit[3], 'status': 'research_failed', 'reason': str(exc)}

        def progress():
            save(root / 'progress.json', {
                'status': 'running' if status == 'complete' else status, 'reason': reason,
                'completed_units': len(completed), 'planned_units': len(units),
                'inherited_units': len(inherited), 'execution': settings,
                'active_units': [{'assay': u[0]['assay'], 'repeat': u[1],
                                  'condition': u[2], 'group': u[3]} for u in pending.values()],
                'unit_statuses': dict(Counter(x['status'] for x in completed.values())),
                'ledger': gateway.summary()})

        def stop(exc):
            nonlocal status, reason
            status, reason = 'paused', str(exc)
            with gateway.lock:
                gateway._pause(reason)
                gateway.db.commit()

        # Only a bounded number of units are submitted. The scheduler owns checkpoint writes.
        try:
            with draining_pool(gateway, settings['unit_workers']) as pool:
                remaining = [u for u in units if unit_key(*u) not in completed]
                while remaining or pending:
                    if not calibrated and calibration_keys <= completed.keys():
                        calibration = {'status': 'passed' if all(
                            completed[k]['status'] == 'ok' and all(
                                r['status'] in {'ok', 'abstained'} for r in completed[k].get('reports', {}).values())
                            for k in calibration_keys) else 'failed',
                            'units': sorted(calibration_keys), 'ledger': gateway.summary(),
                            'selection': 'fixed first task N50; format/tools/accounting only'}
                        save(root / 'calibration.json', calibration)
                        calibrated = calibration['status'] == 'passed'
                        if not calibrated:
                            stop(ExecutionPaused('Calibration failed'))
                    while remaining and status == 'complete' and len(pending) < settings['unit_workers']:
                        if stop_after and len(completed) + len(pending) >= stop_after:
                            break
                        unit = remaining[0]
                        if not calibrated and unit_key(*unit) not in calibration_keys:
                            break
                        try:
                            task, snapshot = validate_inputs(unit)
                        except Exception as exc:
                            stop(exc)
                            break
                        remaining.pop(0)
                        pending[pool.submit(worker, unit, task, snapshot)] = unit
                    progress()
                    if not pending:
                        if stop_after and len(completed) >= stop_after and status == 'complete':
                            status = 'stopped_after_test_limit'
                        break
                    done, _ = wait(pending, timeout=10, return_when=FIRST_COMPLETED)
                    for future in done:
                        unit = pending.pop(future)
                        try:
                            result = future.result()
                        except Exception as exc:
                            stop(exc)
                            continue
                        key = unit_key(*unit)
                        result['fingerprint'] = manifest['fingerprint']
                        result['sha256'] = digest(result)
                        save(root / 'units' / (key + '.json'), result)
                        completed[key] = result
                    # On a hard pause, outstanding calls settle and all successful units persist.
                    if status != 'complete' and not pending:
                        break
        except BaseException as exc:
            stop(exc)
            for future, unit in pending.items():
                if future.cancelled():
                    continue
                try:
                    result = future.result()
                except BaseException:
                    continue
                key = unit_key(*unit)
                result['fingerprint'] = manifest['fingerprint']
                result['sha256'] = digest(result)
                save(root / 'units' / (key + '.json'), result)
                completed[key] = result
        finally:
            ledger = gateway.summary()
            gateway.close()
        report = {'status': status, 'reason': reason, 'completed_units': len(completed),
                  'planned_units': len(units), 'inherited_units': len(inherited),
                  'execution': settings, 'resume_source': manifest.get('resume_from'),
                  'unit_statuses': dict(Counter(x['status'] for x in completed.values())),
                  'ledger': ledger, **evaluate(manifest, completed),
                  'audit': audit_report(root, manifest, completed)}
        save(root / 'report.json', report)
        save(root / 'progress.json', report)
        return report
