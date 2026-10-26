"""Single-request Gemini baseline using the project's native Google gateway."""
import json
import sys
from pathlib import Path
from .common import PROJECT, freeze, load, sha, validate_predictions


def run(item, input_path, request_path, config_path, out, token_budget):
    request, config = load(request_path), load(config_path)
    if (request['input_sha256'] != sha(input_path)
            or request.get('generation_seed') != 42
            or config.get('generation_seed') != 42):
        raise ValueError('Request cohort mismatch or generation seed !=42')
    if config.get('model') != 'gemini-3.8-flash':
        raise ValueError('This experiment requires gemini-3.8-flash')
    if (len(request['messages']) != 2
            or [m['role'] for m in request['messages']] != ['system', 'user']):
        raise ValueError('Expected exactly system and user messages')
    expected = {k:item[k] for k in ('assay','modality','base','endpoint','candidates')}
    if json.loads(request['messages'][1]['content']) != expected:
        raise ValueError('Request must contain only the exact label-free sequence/SMILES task')
    sys.path.insert(0, str(PROJECT / 'code'))
    from evagent.integrated.gateway import RequestGateway, public_model_config, digest
    config = public_model_config(config)
    if config.get('research_tool_protocol'):
        raise ValueError('Sequence/SMILES-only baseline must not enable research tools')
    if type(token_budget) is not int or token_budget <= 0:
        raise ValueError('A positive explicit --token-budget is required')
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    freeze(out / 'model_config.json', config)
    fingerprint = digest({'input_sha256':sha(input_path), 'request_sha256':sha(request_path),
                          'model_config':config, 'token_budget':token_budget})
    source = PROJECT / 'code/evagent/integrated/gateway.py'
    transport = RequestGateway(model_config=config, run_dir=out / 'provider',
        fingerprint=fingerprint, token_budget=token_budget, reserve_budget=True)
    try:
        response = transport.invoke_messages(messages=request['messages'], run_name=item['assay'])
        if response['status'] != 'ok':
            raise ValueError('Gemini response did not finish successfully; inspect provider ledger')
        values = json.loads(response['final_text'])
        validate_predictions(item, values)
        if any(r['base_fitness'] != 0 or not -1 <= r['pred_fitness'] <= 1 for r in values):
            raise ValueError('LLM did not follow frozen reference-zero [-1,1] scale')
    finally:
        freeze(out / 'provider_summary.json', transport.summary())
        transport.close()
    freeze(out / 'predictions.json', values)
    receipt = {'status':'complete', 'method':'llm:'+config['model']+':thinking='+config['thinking_level'],
        'assay':item['assay'], 'n':item['n'], 'seed':42,
        'thinking_level':config['thinking_level'], 'elapsed_s':response['elapsed_s'],
        'generation_seed':42, 'input_sha256':sha(input_path), 'request_sha256':sha(request_path),
        'predictions_sha256':sha(out/'predictions.json'), 'source_type':'fresh_api_inference',
        'comparison_group':'llm_sequence_smiles_only', 'independent_zero_shot_admitted':True,
        'smoke_only':item['smoke_only'], 'labels_read':False, 'msa_or_structure':False,
        'provider':'google_genai', 'provider_ledger':'provider/request_ledger.sqlite',
        'transport_sha256':sha(source), 'adapter_sha256':sha(__file__),
        'model_config_sha256':sha(out/'model_config.json'), 'token_budget':token_budget,
        'usage':response['usage_summary'],
        'returned_model':response.get('response_metadata',{}).get('model_version')}
    freeze(out / 'receipt.json',receipt)
    return {'status':'complete','method':receipt['method'],'n':item['n']}
