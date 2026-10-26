"""One accounted, non-biological transport check; no retry or campaign admission."""
import json
import os
import sys
from pathlib import Path

from dotenv import dotenv_values

from evagent.integrated.gateway import RequestGateway, digest
from evagent.runtime import ExecutionPaused

config_path, output_path = map(Path, sys.argv[1:])
model = json.loads(config_path.read_text())['model_config']
model = {**model, 'transport_mode': 'nonstream'}
key = dotenv_values('/home/yanjie/.config/evagent/credentials.env').get('GEMINI_API_KEY')
if key:
    os.environ['GEMINI_API_KEY'] = key
prompt = 'Return only JSON: {"status":"OK","sum":5050}. The sum is 1+2+...+100.'
gateway = RequestGateway(model_config=model, run_dir=output_path,
                         fingerprint=digest({'purpose': 'nonstream_recovery_check_v1', 'prompt': prompt}),
                         token_budget=None, budget_mode='unlimited')
try:
    result = gateway.generate(system='Follow the requested JSON format.', user=prompt,
                              run_name='transport-check', metadata={'phase': 'transport_check'})
    content = result['final_text'].strip().removeprefix('```json').removesuffix('```').strip()
    if json.loads(content) != {'status': 'OK', 'sum': 5050}:
        raise ValueError('Transport check returned incorrect final JSON')
    receipt = {'status': 'passed', 'finish_reason': result['finish_reason'],
               'response_metadata': result['response_metadata'], 'ledger': gateway.summary()}
except ExecutionPaused:
    receipt = {'status': 'paused', 'ledger': gateway.summary()}
finally:
    gateway.close()
(output_path/'check.json').write_text(json.dumps(receipt, indent=2))
print(json.dumps(receipt), flush=True)
raise SystemExit(0 if receipt['status'] == 'passed' else 2)
