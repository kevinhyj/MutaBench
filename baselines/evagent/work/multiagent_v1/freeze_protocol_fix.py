"""Freeze a format-only calibration correction without reusing old predictions."""
import json
import subprocess
import sys
from pathlib import Path

from evagent.integrated.research.campaign import prepare, run_campaign
from evagent.integrated.tasks import file_hash

root = Path(sys.argv[1]).resolve()
source = root/'campaign_bounded_retry_v1.json'
destination = root/'campaign_tools_v2.json'
if destination.exists():
    raise RuntimeError('Protocol revision already frozen')
config = json.loads(source.read_text())
previous = Path(config['output_dir'])
report = json.loads((previous/'report.json').read_text())
if report['reason'] != 'Calibration failed' or report['completed_units'] != 6:
    raise RuntimeError('Expected completed format calibration failure')
ledger = previous/'provider/request_ledger.sqlite'
config['prior_ledgers'].append({'path': str(ledger), 'sha256': file_hash(ledger)})
config['output_dir'] = str(root/'campaign_tools_v2')
config['model_config']['function_calling_mode'] = 'NONE'
destination.write_text(json.dumps(config, ensure_ascii=False, indent=2)+'\n')
manifest, _ = prepare(destination)
receipt = {'source_commit': subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip(),
           'source_config_sha256': file_hash(source), 'new_config_sha256': file_hash(destination),
           'fingerprint': manifest['fingerprint'], 'prior_usage': manifest['prior_usage'],
           'reason': 'Calibration exposed MALFORMED_FUNCTION_CALL and source-list ambiguity; explicitly disable provider function calling and describe one source string per local JSON request.',
           'selection_on_predictive_performance': False, 'reused_predictions': 0,
           'dry_run': run_campaign(destination, dry_run=True)}
(root/'protocol_fix_v2.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2)+'\n')
print(json.dumps(receipt['dry_run']))
