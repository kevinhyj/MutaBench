"""Freeze a closed serial run and create an audited parallel continuation config."""
import datetime
import fcntl
import json
import sqlite3
import sys
from contextlib import closing
from pathlib import Path

from evagent.integrated.research.campaign import run_campaign
from evagent.integrated.research.contracts import save
from evagent.integrated.research.parallel_resume import freeze_reference

source_config, source_code, destination = map(lambda x: Path(x).resolve(), sys.argv[1:])
raw = json.loads(source_config.read_text())
root = (source_config.parent / raw['output_dir']).resolve()
if destination.exists():
    raise ValueError('Destination config already exists')
with (root / 'campaign.lock').open('r+') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    reference = freeze_reference(root, source_code)
    ledger = root / 'provider/request_ledger.sqlite'
    with closing(sqlite3.connect(f'file:{ledger}?mode=ro&immutable=1', uri=True)) as db:
        state = dict(db.execute('SELECT key,value FROM run_state'))
        unknown = sorted(h for h, status, tokens in db.execute(
            "SELECT request_sha256,status,tokens FROM attempts WHERE status!='not_submitted'")
            if status not in {'ok', 'blocked', 'truncated', 'terminal_failure'} or type(tokens) is not int)
        if any('connection_retry_ack:' + h not in state for h in unknown):
            raise ValueError('Unknown attempt lacks existing bounded retry authorization')
    debit = {'path': str(ledger), 'sha256': reference['ledger_sha256']}
    if unknown:
        debit['manual_recovery'] = {
            'request_sha256s': unknown,
            'authorization': raw['connection_retry_policy']['authorization'] +
                ' User requested parallel restart; carry forward already acknowledged attempts without erasing unknown usage or replaying them.',
            'recorded_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'usage_status': 'unresolved_lower_bound'}
    # Keep all source paths meaningful even if the destination config lives elsewhere.
    for row in raw['tasks']:
        for name in ('sample_file', 'context_file', 'snapshot_file'):
            row[name] = str((source_config.parent / row[name]).resolve())
    for name in ('coverage_file', 'environment_file'):
        if name in raw:
            raw[name] = str((source_config.parent / raw[name]).resolve())
    for row in raw.get('prior_ledgers', []):
        row['path'] = str((source_config.parent / row['path']).resolve())
    raw.update(output_dir=str(destination.with_suffix('')),
               execution={'unit_workers': 2, 'max_concurrent_requests': 4, 'parallel_specialists': True},
               resume_from=reference, prior_ledgers=[*raw.get('prior_ledgers', []), debit])
    save(destination, raw)
    result = run_campaign(destination, dry_run=True)
    save(destination.with_suffix('') / 'restart_dry_run.json', result)
    print(json.dumps(result, indent=2))
