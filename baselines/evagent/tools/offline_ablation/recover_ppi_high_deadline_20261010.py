"""Deadline recovery for PPI Gemini 3.8 HIGH: retry only proxy/transport failures.

Copies the live HIGH freeze into a new directory, keeps ok and format-failure
responses, drops Connection-refused / 500 terminals, and scores remaining units
on unused proxies in parallel. Prompt bytes, seed, temperature and HIGH thinking
are unchanged. MEDIUM/LOW directories are not touched.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import shutil
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

ROOT = Path('/data/huangyanjie/EVAgent')
CODE = ROOT / 'code'
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(CODE / 'tools/offline_ablation'))

import run as ablation

SOURCE = ROOT / 'experiments/ppi_cr25_gemini38_high_clone_medium_cr65k_20261009'
OUTPUT = ROOT / 'experiments/ppi_cr25_gemini38_high_deadline_r2_20261010'
PROXIES = [
    'http://127.0.0.1:17890',
    'http://127.0.0.1:17891',
    'http://127.0.0.1:17892',
    'http://127.0.0.1:17893',
]
KEEP_STATUS = {'ok', 'invalid_prediction', 'invalid_response'}
RETRY_HTTP = {500, 502, 503, 504, 429}
CONCURRENCY = 10
SKIP_COPY = {
    'freeze.json', 'execution_snapshot_manifest.json', 'execution_snapshot_amendment.json',
    'run_state.json', 'dispatch_run.log', 'run.lock', 'pilot_status.json',
    'frozen_dispatch_records.jsonl', 'execution_code_snapshot',
}


def load(path: Path):
    return json.loads(path.read_text())


def save(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def should_retry(response: dict) -> bool:
    status = response.get('status')
    if status in KEEP_STATUS:
        return False
    if status == 'http_error' and response.get('http_status') not in RETRY_HTTP:
        return False
    return True


def hardlink_tree(source: Path, dest: Path) -> None:
    dest.mkdir(parents=True)
    for root, dirs, files in os.walk(source):
        rel = Path(root).relative_to(source)
        dirs[:] = [name for name in dirs if not (rel / name).as_posix() in SKIP_COPY and name not in SKIP_COPY]
        target_dir = dest / rel
        target_dir.mkdir(parents=True, exist_ok=True)
        for name in files:
            if name in SKIP_COPY:
                continue
            src = Path(root) / name
            dst = target_dir / name
            try:
                os.link(src, dst)
            except OSError:
                shutil.copy2(src, dst)


def unlink_retry_units(output: Path, plan: list) -> list:
    pending = []
    archived = 0
    archive = output / 'transport_failure_archive'
    for unit in plan:
        path = output / 'responses' / f"{unit['id']}.json"
        if not path.exists():
            pending.append(unit)
            continue
        response = load(path)
        if not should_retry(response):
            continue
        dest = archive / path.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            dest.unlink()
        path.rename(dest)
        archived += 1
        attempt_dir = output / 'attempts' / unit['id']
        if attempt_dir.exists():
            shutil.move(str(attempt_dir), str(archive / 'attempts' / unit['id']))
        pending.append(unit)
    save(output / 'recovery_pending.json', dict(pending_units=len(pending), archived_retryable=archived,
                                                keep_status=sorted(KEEP_STATUS)))
    print('PENDING', len(pending), 'ARCHIVED', archived, flush=True)
    return pending


def probe_proxy(proxy: str, credential: str, endpoint: str) -> dict:
    body = {
        'contents': [{'role': 'user', 'parts': [{'text': 'Reply with the single token ok.'}]}],
        'generationConfig': {'temperature': 0, 'maxOutputTokens': 16,
                             'thinkingConfig': {'thinkingLevel': 'LOW', 'includeThoughts': False}},
    }
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({'http': proxy, 'https': proxy}))
    request = urllib.request.Request(endpoint, data=json.dumps(body).encode(), method='POST',
                                     headers={'Content-Type': 'application/json', 'x-goog-api-key': credential})
    started = time.time()
    try:
        with opener.open(request, timeout=3600) as remote:
            payload = json.loads(remote.read())
        return dict(proxy=proxy, ok=True, http=200, seconds=time.time() - started,
                    model=payload.get('modelVersion'))
    except urllib.error.HTTPError as error:
        return dict(proxy=proxy, ok=False, http=error.code, seconds=time.time() - started,
                    error=error.read()[:200].decode(errors='replace'))
    except Exception as error:
        return dict(proxy=proxy, ok=False, http=None, seconds=time.time() - started, error=str(error))


def run_shard(output: Path, units: list, proxy: str, shard_id: int, concurrency: int) -> None:
    config = load(output / 'config.json')
    config = dict(config)
    config['proxy'] = proxy
    config['concurrency'] = concurrency
    credential = ablation.load_credential(config['credential_file'])
    common = ablation.legacy_common()
    started = time.time()
    state_path = output / f'shard_{shard_id}_state.json'
    log_path = output / f'shard_{shard_id}.log'
    counts = Counter()
    active_error = None

    def write_state(status, extra=None):
        payload = dict(status=status, shard=shard_id, proxy=proxy, counts=dict(counts),
                       total=len(units), started=started, updated=time.time())
        if extra:
            payload.update(extra)
        save(state_path, payload)

    write_state('running')
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        iterator = iter(units)
        active = {}
        while True:
            while active_error is None and len(active) < (concurrency if counts else 1):
                unit = next(iterator, None)
                if unit is None:
                    break
                active[pool.submit(ablation.run_unit, output, unit, config, credential, common)] = unit
            if not active:
                break
            finished, _ = concurrent.futures.wait(active, timeout=30, return_when=concurrent.futures.FIRST_COMPLETED)
            for future in finished:
                unit = active.pop(future)
                try:
                    result = future.result()
                except Exception as error:
                    active_error = str(error)
                    with log_path.open('a') as handle:
                        handle.write(f"{unit['id']} runner_exception {error}\n")
                    continue
                counts[result['status']] += 1
                if result.get('http_status') in (401, 403) or result['status'] == 'model_identity_mismatch':
                    active_error = result['status']
                with log_path.open('a') as handle:
                    handle.write(f"{unit['id']} {result['status']}\n")
            write_state('running', dict(active=len(active), halt=active_error))
            if active_error and not active:
                break
    write_state('finished_with_error' if active_error else 'finished', dict(halt=active_error, finished=time.time()))
    print('SHARD_DONE', shard_id, dict(counts), 'halt', active_error, flush=True)


def prepare(output: Path) -> list:
    if output.exists():
        raise ValueError('Output already exists: ' + str(output))
    print('COPY', SOURCE, '->', output, flush=True)
    hardlink_tree(SOURCE, output)
    for name in ('config.json', 'plan.json', 'README.txt'):
        path = output / name
        if path.exists():
            data = path.read_bytes()
            path.unlink()
            path.write_bytes(data)
    config = load(output / 'config.json')
    config['proxy'] = PROXIES[0]
    config['concurrency'] = CONCURRENCY
    config['timeout_seconds'] = 600
    config['recovery'] = dict(
        source=str(SOURCE),
        policy='retry transport_error and HTTP 5xx/429 only; keep format failures; do not rescore ok',
        proxies=PROXIES,
        deadline_hours=6,
    )
    save(output / 'config.json', config)
    plan = load(output / 'plan.json')
    return unlink_retry_units(output, plan)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('prepare', 'probe', 'run', 'status'))
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--shard-id', type=int)
    parser.add_argument('--proxy')
    parser.add_argument('--shards', type=int, default=3)
    parser.add_argument('--concurrency', type=int, default=CONCURRENCY)
    parser.add_argument('--pending-file', type=Path, help='JSON list of unit ids; default pending_units.json')
    args = parser.parse_args()
    output = args.output.resolve()
    if args.stage == 'prepare':
        pending = prepare(output)
        save(output / 'pending_units.json', [unit['id'] for unit in pending])
        return
    if args.stage == 'probe':
        config = load(SOURCE / 'config.json')
        credential = ablation.load_credential(config['credential_file'])
        rows = [probe_proxy(proxy, credential, config['endpoint']) for proxy in PROXIES]
        print(json.dumps(rows, ensure_ascii=False, indent=2))
        save(output.parent / 'ppi_high_deadline_proxy_probe.json' if not output.exists() else output / 'proxy_probe.json', rows)
        return
    if args.stage == 'status':
        plan = load(output / 'plan.json')
        counts = Counter()
        missing = 0
        for unit in plan:
            path = output / 'responses' / f"{unit['id']}.json"
            if not path.exists():
                missing += 1
                continue
            counts[load(path).get('status')] += 1
        print(json.dumps(dict(ok=counts.get('ok'), missing=missing, counts=dict(counts), total=len(plan)), indent=2))
        return
    pending_ids = set(load(args.pending_file or (output / 'pending_units.json')))
    # Skip units that became ok after prepare.
    units = []
    for unit in load(output / 'plan.json'):
        if unit['id'] not in pending_ids:
            continue
        path = output / 'responses' / f"{unit['id']}.json"
        if path.exists() and load(path).get('status') == 'ok':
            continue
        units.append(unit)
    if args.shard_id is None or args.proxy is None:
        raise SystemExit('run requires --shard-id and --proxy')
    mine = [unit for index, unit in enumerate(units) if index % args.shards == (args.shard_id % args.shards)]
    print('SHARD', args.shard_id, 'units', len(mine), 'proxy', args.proxy, flush=True)
    run_shard(output, mine, args.proxy, args.shard_id, args.concurrency)


if __name__ == '__main__':
    main()
