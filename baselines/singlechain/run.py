#!/usr/bin/env python3
"""Entry point. Run from any cwd using baseline/environments/singlechain/bin/python."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from singlechain.common import ROOT, freeze, load, sha, records, validate_input, validate_predictions


def device_preflight(args):
    if args.device == 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        return
    if not args.gpu_authorized or not args.gpu_uuid or not args.gpu_uuid.startswith('GPU-'):
        raise ValueError('GPU requires explicit resource authorization and --gpu-uuid; idle memory is not authorization')
    status = subprocess.check_output(['nvidia-smi', '-i', args.gpu_uuid,
        '--query-gpu=uuid,memory.used,utilization.gpu', '--format=csv,noheader,nounits'], text=True).strip()
    uuid, used, util = [s.strip() for s in status.split(',')]
    processes = subprocess.check_output(['nvidia-smi', '-i', args.gpu_uuid,
        '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True).strip()
    if uuid != args.gpu_uuid or processes or float(used) > 512 or float(util) > 5:
        raise RuntimeError('Authorized GPU is occupied; refusing to contend or auto-switch GPUs')
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_uuid
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'


def run_predict(args):
    device_preflight(args)
    item = validate_input(load(args.input), smoke=args.smoke)
    out = Path(args.out)
    if out.exists():
        raise FileExistsError('Use a new run directory; predictions and receipts are immutable')
    started = time.time()
    if args.method == 'gemme:native':
        if args.device != 'cpu':
            raise ValueError('GEMME container uses CPU only')
        from singlechain.gemme import predict
        scores, base, provenance = predict(item, args.input, args.out)
        group, admitted = 'msa_statistical', True
    elif args.method == 'potts':
        from singlechain.potts import predict
        scores, base, provenance = predict(item, args.model_dir)
        group, admitted = 'msa_statistical', True
    elif args.method in ('ecfp4:rf', 'ecfp4:xgboost'):
        from singlechain.external_ml import predict
        scores, base, provenance = predict(item, args.model_dir)
        if provenance['estimator'] != args.method.split(':')[1]:
            raise ValueError('External estimator does not match method name')
        group, admitted = 'external_supervised_transfer', True
    elif args.method == 'chemprop:external-esol':
        from singlechain.chemprop_external import predict
        scores, base, provenance = predict(item, args.model_dir)
        group, admitted = 'external_supervised_transfer', True
    else:
        from singlechain.models import predict
        scores, base, provenance, group, admitted = predict(item, args.method, args.device, args.batch_size)
    values = records(item, scores, base, rationale=f'{args.method}: native score, fixed reference; see receipt for endpoint/proxy meaning.')
    validate_predictions(item, values)
    freeze(out / 'predictions.json', values)
    receipt = {'status': 'complete', 'method': args.method, 'assay': item['assay'],
        'input_sha256': sha(args.input), 'n': item['n'], 'seed': 42,
        'predictions_sha256': sha(out / 'predictions.json'), 'source_type': 'fresh_local_inference',
        'comparison_group': group, 'independent_zero_shot_admitted': admitted,
        'smoke_only': item['smoke_only'], 'labels_read': False, 'device': args.device,
        'physical_gpu_uuid': args.gpu_uuid, 'elapsed_s': time.time()-started,
        'python': sys.version, 'provenance': provenance,
        'code_hashes': {str(p.relative_to(ROOT)): sha(p) for p in sorted((ROOT / 'singlechain').glob('*.py'))}}
    freeze(out / 'receipt.json', receipt)
    return {k: receipt[k] for k in ('status', 'method', 'assay', 'n', 'elapsed_s', 'comparison_group', 'independent_zero_shot_admitted')}


def export_prompt(args):
    item = validate_input(load(args.input))
    # Predictors receive a deliberate whitelist, not the manifest, labels or evidence paths.
    request = {k: item[k] for k in ('assay', 'modality', 'base', 'endpoint', 'candidates')}
    system = ('EVAgent-2 single-chain sequence/SMILES-only zero-shot baseline. N=250, data seed=42. '
        'Do not retrieve external information, use MSA/structures, train, or infer experimental labels. '
        'Use one fixed scale [-1,1] and base_fitness=0 across this assay. '
        'This is an anchored relative score, not a calibrated absolute fitness. '
        'For ligand the base is a reference molecule, not a wild type. '
        'Interpret the declared endpoint and higher_means exactly. '
        'Return only a JSON list with exactly one record per supplied ID: '
        '{"id":int,"base_fitness":0.0,"pred_fitness":float,"delta_fitness":float,"rationale":str}. '
        'delta_fitness=pred_fitness-base_fitness. Use at most one sentence for rationale.')
    # Model and transport budget are explicit run choices; this preparation command makes no API call.
    freeze(args.out, {'generation_seed': 42, 'data_seed': 42, 'input_sha256': sha(args.input),
                      'messages': [{'role': 'system', 'content': system},
                                   {'role': 'user', 'content': json.dumps(request, ensure_ascii=False)}]})
    return {'status': 'prompt_ready_no_api_call', 'out': args.out}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare'); p.add_argument('--spec', required=True); p.add_argument('--out', required=True)
    p = sub.add_parser('predict'); p.add_argument('--input', required=True); p.add_argument('--out', required=True)
    p.add_argument('--method', required=True); p.add_argument('--model-dir')
    p.add_argument('--device', choices=['cpu', 'cuda:0'], default='cpu')
    p.add_argument('--gpu-uuid'); p.add_argument('--gpu-authorized', action='store_true')
    p.add_argument('--batch-size', type=int, default=4); p.add_argument('--smoke', action='store_true')
    p = sub.add_parser('evaluate'); p.add_argument('--input', required=True); p.add_argument('--labels', required=True)
    p.add_argument('--run', required=True); p.add_argument('--out', required=True)
    p = sub.add_parser('fit-potts'); p.add_argument('--input', required=True); p.add_argument('--out', required=True)
    p = sub.add_parser('fit-external'); p.add_argument('--source', required=True); p.add_argument('--target-public', required=True)
    p.add_argument('--estimator', choices=['rf', 'xgboost'], default='rf'); p.add_argument('--out', required=True)
    p = sub.add_parser('fit-chemprop'); p.add_argument('--external-card', required=True); p.add_argument('--out', required=True)
    p = sub.add_parser('prompt'); p.add_argument('--input', required=True); p.add_argument('--out', required=True)
    p = sub.add_parser('import-archive'); p.add_argument('--input', required=True); p.add_argument('--spec', required=True); p.add_argument('--out', required=True)
    p = sub.add_parser('llm'); p.add_argument('--input', required=True); p.add_argument('--request', required=True)
    p.add_argument('--provider-config', default=str(ROOT / 'configs/gemini_flash_medium_seed42.json'))
    p.add_argument('--token-budget', type=int, required=True); p.add_argument('--out', required=True)
    p = sub.add_parser('validate'); p.add_argument('--input', required=True); p.add_argument('--predictions')
    args = parser.parse_args()
    if args.command == 'prepare':
        from singlechain.prepare import prepare
        result = prepare(args.spec, args.out)
    elif args.command == 'predict':
        result = run_predict(args)
    elif args.command == 'evaluate':
        from singlechain.evaluate import evaluate
        result = evaluate(args.input, args.labels, args.run, args.out)
    elif args.command == 'fit-potts':
        from singlechain.potts import fit
        result = fit(validate_input(load(args.input)), args.out)
    elif args.command == 'fit-external':
        from singlechain.external_ml import fit_esol
        result = fit_esol(args.source, args.target_public, args.out, args.estimator)
    elif args.command == 'fit-chemprop':
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        from singlechain.chemprop_external import fit
        result = fit(args.external_card, args.out)
    elif args.command == 'prompt':
        result = export_prompt(args)
    elif args.command == 'import-archive':
        from singlechain.archive import import_scores
        result = import_scores(validate_input(load(args.input)), args.input, args.spec, args.out)
    elif args.command == 'llm':
        from singlechain.llm import run
        result = run(validate_input(load(args.input)), args.input, args.request, args.provider_config, args.out, args.token_budget)
    elif args.command == 'validate':
        item = validate_input(load(args.input))
        if args.predictions:
            validate_predictions(item, load(args.predictions))
        result = {'status': 'valid', 'n': item['n'], 'seed': 42}
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == '__main__':
    main()
