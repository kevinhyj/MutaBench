"""Run actual specialist routing on a format-only, evidence-empty synthetic fixture."""
import json
import os
import sys
from pathlib import Path

from dotenv import dotenv_values

from evagent.integrated.gateway import RequestGateway, digest
from evagent.integrated.research.agents import run_specialist
from evagent.integrated.research.contracts import ResearchFailure, save
from evagent.integrated.research.tools import FrozenTools
from evagent.runtime import ExecutionPaused

config_path, root = map(Path, sys.argv[1:])
config = json.loads(config_path.read_text())
old_root = Path(config['output_dir'])
if not (old_root/'report.json').exists():
    raise RuntimeError('Wait for the preceding campaign to close before diagnostics')
key = dotenv_values('/home/yanjie/.config/evagent/credentials.env').get('GEMINI_API_KEY')
if key:
    os.environ['GEMINI_API_KEY'] = key
model = {k: v for k,v in config['model_config'].items() if k not in {'function_calling_mode', 'response_mime_type'}}
model['research_tool_protocol'] = 'native_v1'
task = {'assay': 'format_only_synthetic_fixture', 'wt_sequence': 'AGCU',
        'variants': [{'variant_id': 'M01', 'sequence': 'AGCU'}],
        'metadata': {'description': 'A routing test with no biological evidence. Report missing evidence; no biological inference.'}}
empty = {'status': 'unavailable', 'reason': 'Intentionally empty format-only fixture',
         'task': {}, 'positions': {}, 'candidates': {}, 'provenance': [], 'limitations': []}
snapshot = {'sources': {s: empty for s in ('literature', 'structure', 'msa')},
            'geometry': {'status': 'unavailable'}, 'literature_documents': []}
snapshot['sha256'] = digest(snapshot)
gateway = RequestGateway(model_config=model, run_dir=root/'provider',
                         fingerprint=digest({'purpose': 'native_routing_v1', 'model': model}),
                         token_budget=None, budget_mode='unlimited',
                         connection_retry_policy=config['connection_retry_policy'])
reports = {}
status = 'passed'
try:
    for role in ('single', 'literature', 'structure', 'msa'):
        result = run_specialist(task, snapshot, role, gateway,
                               {'phase': 'format_check', 'assay': task['assay']}, FrozenTools)
        reports[role] = result
        save(root/(role+'.json'), result)
        if result['status'] not in {'ok', 'abstained'}:
            status = 'failed'
            break
except ExecutionPaused:
    status = 'paused'
except ResearchFailure:
    status = 'failed'
finally:
    summary = gateway.summary()
    gateway.close()
receipt = {'status': status, 'roles': {k: v['status'] for k,v in reports.items()}, 'ledger': summary}
save(root/'check.json', receipt)
print(json.dumps(receipt), flush=True)
raise SystemExit(0 if status == 'passed' else 2)
