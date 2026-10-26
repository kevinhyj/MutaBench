"""Freeze the user-approved connection-only retry revision; preserve both incidents."""
import datetime
import json
import sqlite3
import subprocess
import sys
from pathlib import Path

from evagent.integrated.research.campaign import prepare, run_campaign
from evagent.integrated.tasks import file_hash

root = Path(sys.argv[1]).resolve()
source = root/'campaign_recovery_v1.json'
destination = root/'campaign_bounded_retry_v1.json'
if destination.exists():
    raise RuntimeError('Retry revision already frozen')
config = json.loads(source.read_text())
old = Path(config['output_dir'])/'provider/request_ledger.sqlite'
with sqlite3.connect(old.as_uri()+'?mode=ro&immutable=1', uri=True) as db:
    attempts = db.execute('SELECT request_sha256,status FROM attempts').fetchall()
expected = [('7a7fed8a72111bd3f72a5a2749db2f509decb85ba507d7ed535ceae83449d649', 'failed_unknown_cost')]
if attempts != expected:
    raise RuntimeError('Incident differs from reviewed failure')
authorization = 'User explicitly approved: 允许上述有界重试. Only connection interruptions: at most two retries, each independently accounted, retain unknown usage; three consecutive failed attempts pause the campaign. Other unknown responses do not retry.'
config['connection_retry_policy'] = {'max_attempts': 3, 'backoff_seconds': [15, 30], 'authorization': authorization}
config['output_dir'] = str(root/'campaign_bounded_retry_v1')
config['model_config']['transport_mode'] = 'stream'
config['prior_ledgers'].append({'path': str(old), 'sha256': file_hash(old), 'manual_recovery': {
    'request_sha256s': [expected[0][0]], 'authorization': authorization,
    'recorded_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'usage_status': 'unresolved_lower_bound'}})
destination.write_text(json.dumps(config, ensure_ascii=False, indent=2)+'\n')
manifest, _ = prepare(destination)
receipt = {'authorization': authorization, 'source_commit': subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip(),
           'source_config_sha256': file_hash(source), 'new_config_sha256': file_hash(destination),
           'fingerprint': manifest['fingerprint'], 'prior_usage': manifest['prior_usage'],
           'transport_reason': 'Long nonstream trial also disconnected; restore stream to preserve partial response and usage while applying explicitly authorized bounded connection retries.',
           'science_configuration_changes': [], 'dry_run': run_campaign(destination, dry_run=True)}
(root/'bounded_retry_authorization_v1.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2)+'\n')
print(json.dumps(receipt['dry_run']))
