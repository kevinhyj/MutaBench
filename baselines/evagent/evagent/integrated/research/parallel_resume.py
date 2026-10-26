"""Audited serial-to-parallel continuation; scientific inputs must be identical."""
from __future__ import annotations

import ast
import json
import sqlite3
from contextlib import closing
from pathlib import Path

from ..gateway import digest
from ..tasks import file_hash

TERMINAL = {'ok', 'blocked', 'truncated', 'terminal_failure'}
ORCHESTRATION = {
    'integrated/research/campaign.py': {'prepare', 'run_campaign'},
    'integrated/research/agents.py': {'analyze'},
}
ADDITIONS = {'integrated/research/' + name + '.py' for name in (
    'concurrent_gateway', 'parallel_campaign', 'parallel_resume')}
OPERATIONAL = {'fingerprint', 'implementation', 'config_sha256', 'output_dir',
               'prior_ledgers', 'prior_usage', 'execution', 'resume_from'}


def scientific_view(manifest):
    return {k: v for k, v in manifest.items() if k not in OPERATIONAL}


def _tree_without(text, names):
    tree = ast.parse(text)
    tree.body = [node for node in tree.body
                 if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) or node.name not in names]
    return ast.dump(tree, include_attributes=False)


def verify_source(old, new, source_code):
    """Only reviewed orchestration functions may differ; prompts/tools remain exact."""
    source = Path(source_code) / 'evagent'
    current = Path(__file__).resolve().parents[2]
    before, after = old['source_files'], new['source_files']
    for name, sha in before.items():
        if file_hash(source / name) != sha:
            raise ValueError('Resume source code changed: ' + name)
        if name not in after:
            raise ValueError('Resume source file removed: ' + name)
        if after[name] != sha:
            if name not in ORCHESTRATION or _tree_without((source / name).read_text(), ORCHESTRATION[name]) != _tree_without((current / name).read_text(), ORCHESTRATION[name]):
                raise ValueError('Resume scientific source changed: ' + name)
    if set(after) - set(before) - ADDITIONS:
        raise ValueError('Unreviewed source addition during resume')
    for key in ('python', 'dependencies'):
        if old[key] != new[key]:
            raise ValueError('Resume runtime changed: ' + key)


def inventory(root):
    return {p.name: file_hash(p) for p in sorted((Path(root) / 'units').glob('*.json'))}


def freeze_reference(root, source_code):
    """Return pins for an already stopped source run; performs no writes."""
    root = Path(root).resolve()
    ledger = root / 'provider/request_ledger.sqlite'
    if Path(str(ledger) + '-wal').exists() and Path(str(ledger) + '-wal').stat().st_size:
        raise ValueError('Close the source campaign before freezing resume')
    with closing(sqlite3.connect(f'file:{ledger}?mode=ro&immutable=1', uri=True)) as db:
        if db.execute("SELECT COUNT(*) FROM attempts WHERE status='inflight'").fetchone()[0]:
            raise ValueError('Source has inflight provider requests')
    return {'root': str(root), 'source_code': str(Path(source_code).resolve()),
            'manifest_sha256': file_hash(root / 'manifest.json'),
            'checkpoint_inventory_sha256': digest(inventory(root)),
            'ledger_sha256': file_hash(ledger),
            'calibration_sha256': file_hash(root / 'calibration.json')}


def load_resume(manifest):
    reference = manifest.get('resume_from')
    if reference is None:
        return {}, {}, None
    root = Path(reference['root'])
    if freeze_reference(root, reference['source_code']) != reference:
        raise ValueError('Resume source artifacts changed')
    old = json.loads((root / 'manifest.json').read_text())
    if old['fingerprint'] != digest({k: v for k, v in old.items() if k != 'fingerprint'}):
        raise ValueError('Resume manifest integrity failure')
    if scientific_view(old) != scientific_view(manifest):
        raise ValueError('Resume scientific configuration changed')
    verify_source(old['implementation'], manifest['implementation'], reference['source_code'])
    ledger = root / 'provider/request_ledger.sqlite'
    if not any(r['path'] == str(ledger) and r['sha256'] == reference['ledger_sha256']
               for r in manifest['prior_usage']['ledgers']):
        raise ValueError('Resume ledger costs were not carried forward')
    calibration = json.loads((root / 'calibration.json').read_text())
    if calibration['status'] != 'passed':
        raise ValueError('Source calibration did not pass')
    completed = {}
    from .campaign import all_units, unit_key
    expected = {unit_key(*u): (u[0]['assay'], *u[1:]) for u in all_units(manifest)}
    for p in sorted((root / 'units').glob('*.json')):
        item = json.loads(p.read_text())
        if (item.get('sha256') != digest({k: v for k, v in item.items() if k != 'sha256'})
                or item.get('fingerprint') != old['fingerprint']
                or expected.get(p.stem) != tuple(item.get(k) for k in ('assay', 'repeat', 'condition', 'group'))):
            raise ValueError('Resume checkpoint integrity failure')
        source = {'path': str(p), 'sha256': file_hash(p), 'fingerprint': old['fingerprint'],
                  'ledger': str(ledger), 'ledger_sha256': reference['ledger_sha256']}
        item = {k: v for k, v in item.items() if k != 'sha256'}
        item.update(fingerprint=manifest['fingerprint'], resume_source=source)
        item['sha256'] = digest(item)
        completed[p.stem] = item
    cache = {}
    with closing(sqlite3.connect(f'file:{ledger}?mode=ro&immutable=1', uri=True)) as db:
        state = dict(db.execute('SELECT key,value FROM run_state'))
        if state['fingerprint'] != old['fingerprint']:
            raise ValueError('Source ledger fingerprint mismatch')
        for request_hash, status, request_json, response_json, elapsed in db.execute(
                'SELECT request_sha256,status,request_json,response_json,elapsed_s FROM attempts ORDER BY id'):
            request = json.loads(request_json)
            if digest(request) != request_hash or request['fingerprint'] != old['fingerprint']:
                raise ValueError('Source request integrity failure')
            if status == 'not_submitted':
                continue
            acknowledged = 'connection_retry_ack:' + request_hash in state
            if status not in TERMINAL and not (status == 'failed_unknown_cost' and acknowledged):
                raise ValueError('Source contains an unresolved attempt without an approved retry acknowledgment')
            response = json.loads(response_json)
            if status in TERMINAL and (response.get('request_sha256') != request_hash or response.get('status') != status):
                raise ValueError('Source response integrity failure')
            response = {**response, 'request_sha256': request_hash, 'status': status, 'elapsed_s': elapsed}
            key = digest({k: v for k, v in request.items() if k != 'fingerprint'})
            cache[key] = {'response': response, 'retry_acknowledged': acknowledged, 'source': {
                'ledger': str(ledger), 'ledger_sha256': reference['ledger_sha256'],
                'request_sha256': request_hash, 'fingerprint': old['fingerprint']}}
    return completed, cache, {**calibration, 'resume_source': reference}
