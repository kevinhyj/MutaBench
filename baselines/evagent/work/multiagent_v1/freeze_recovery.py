"""Create a new frozen revision for one explicitly acknowledged interrupted attempt."""
import datetime
import json
import sqlite3
import subprocess
import sys
from pathlib import Path

from evagent.integrated.research.campaign import prepare, run_campaign
from evagent.integrated.tasks import file_hash

root = Path(sys.argv[1]).resolve()
source = root/'campaign_release.json'
destination = root/'campaign_recovery_v1.json'
receipt_path = root/'recovery_authorization_v1.json'
if destination.exists() or receipt_path.exists():
    raise RuntimeError('Recovery revision already frozen; do not regenerate')
original = json.loads(source.read_text())
old_root = Path(original['output_dir'])
old_ledger = old_root/'provider/request_ledger.sqlite'
smoke = root/'recovery_transport_check/request_ledger.sqlite'
if json.loads((root/'recovery_transport_check/check.json').read_text())['status'] != 'passed':
    raise RuntimeError('Transport check did not pass')
with sqlite3.connect(old_ledger.as_uri()+'?mode=ro&immutable=1', uri=True) as db:
    attempts = db.execute('SELECT request_sha256,status FROM attempts').fetchall()
expected = [('8d20523414da68492482ad767e2f7887156ae59170f46a4554089eb8ef6362fa', 'failed_unknown_cost')]
if attempts != expected:
    raise RuntimeError('Historical attempt differs from the operator-reviewed incident')
ack = {
    'request_sha256s': [expected[0][0]],
    'authorization': 'User: 好的，解决这些问题，然后跑上完整实验。 One-time operator-directed recovery after disclosure of the interrupted attempt; retain unresolved usage. This is not authorization for automatic retries of future unknown attempts.',
    'recorded_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'usage_status': 'unresolved_lower_bound',
}
config = {**original, 'output_dir': str(root/'campaign_recovery_v1'),
          'model_config': {**original['model_config'], 'transport_mode': 'nonstream'},
          'prior_ledgers': [*original['prior_ledgers'],
                            {'path': str(old_ledger), 'sha256': file_hash(old_ledger), 'manual_recovery': ack},
                            {'path': str(smoke), 'sha256': file_hash(smoke)}]}
destination.write_text(json.dumps(config, ensure_ascii=False, indent=2)+'\n')
manifest, _ = prepare(destination)
dry = run_campaign(destination, dry_run=True)
receipt = {'authorization': ack, 'old_config_sha256': file_hash(source),
           'new_config_sha256': file_hash(destination), 'old_campaign_preserved': str(old_root),
           'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
           'fingerprint': manifest['fingerprint'], 'prior_usage': manifest['prior_usage'],
           'changes': ['explicit nonstream transport', 'redacted network diagnostics', 'one-time acknowledged historical unresolved usage'],
           'unchanged': ['cohort', 'snapshots', 'samples', 'prompts', 'model', 'thinking', 'seeds', 'fusion', 'repairs', 'tool limits'],
           'future_unknown_policy': 'pause; no automatic retry', 'dry_run': dry}
receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2)+'\n')
print(json.dumps(dry))
