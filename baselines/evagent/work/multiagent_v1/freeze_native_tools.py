"""Freeze native function dispatch after the real format-only routing check passes."""
import json
import subprocess
import sys
from pathlib import Path

from evagent.integrated.research.campaign import prepare, run_campaign
from evagent.integrated.tasks import file_hash

root = Path(sys.argv[1]).resolve()
source = root/'campaign_tools_v2.json'
destination = root/'campaign_native_v3.json'
if destination.exists():
    raise RuntimeError('Native tool revision already frozen')
check = json.loads((root/'native_tool_check_v1/check.json').read_text())
if check['status'] != 'passed' or set(check['roles']) != {'single','literature','structure','msa'}:
    raise RuntimeError('Native routing not validated')
config = json.loads(source.read_text())
previous = Path(config['output_dir'])
if not (previous/'report.json').exists():
    raise RuntimeError('Previous campaign still active')
for ledger in (previous/'provider/request_ledger.sqlite',
               root/'structured_tool_check_v1/provider/request_ledger.sqlite',
               root/'native_tool_check_v1/provider/request_ledger.sqlite'):
    config['prior_ledgers'].append({'path': str(ledger), 'sha256': file_hash(ledger)})
config['output_dir'] = str(root/'campaign_native_v3')
config['model_config'] = {k:v for k,v in config['model_config'].items()
                          if k not in {'response_mime_type','function_calling_mode'}}
config['model_config']['research_tool_protocol'] = 'native_v1'
destination.write_text(json.dumps(config, ensure_ascii=False, indent=2)+'\n')
manifest, _ = prepare(destination)
receipt = {'source_commit': subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip(),
           'new_config_sha256': file_hash(destination), 'fingerprint': manifest['fingerprint'],
           'prior_usage': manifest['prior_usage'], 'native_check': check,
           'reason': 'Real JSON-only routing repeatedly returned MALFORMED_FUNCTION_CALL. Use declared provider functions with AFC disabled, validate one role-scoped call, and dispatch only through FrozenTools.',
           'selection_on_predictive_performance': False, 'reused_predictions': 0,
           'dry_run': run_campaign(destination, dry_run=True)}
(root/'native_protocol_v3.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2)+'\n')
print(json.dumps(receipt['dry_run']))
