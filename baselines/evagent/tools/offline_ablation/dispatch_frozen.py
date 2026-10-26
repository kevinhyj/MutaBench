"""Pilot or continue an immutable scoring run without rewriting its freeze."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def recover_quota_failures(module, output):
    """Explicit recovery only for confirmed quota errors, with original records intact."""
    units = {unit['id']: unit for unit in module.read(output / 'plan.json')}
    recovered = []
    for path in sorted((output / 'responses').glob('*.json')):
        response = module.read(path)
        unit = units.get(response.get('unit_id'))
        if unit is None or response.get('request_sha256') != unit['request_sha256']:
            raise ValueError('Response differs from the frozen plan')
        if response.get('status') != 'http_error' or response.get('http_status') != 402:
            continue
        attempt_path = Path(response['raw_attempt'])
        attempt = module.read(attempt_path)
        if attempt.get('http_status') != 402 or attempt.get('body', {}).get('error', {}).get('code') != 'insufficient_quota':
            continue
        if attempt.get('request_sha256') != unit['request_sha256']:
            raise ValueError('Quota attempt differs from the frozen request')
        archive = output / 'quota_failure_responses' / (path.stem + '__attempt_' + attempt_path.stem + '.json')
        archive.parent.mkdir(exist_ok=True)
        if archive.exists():
            raise ValueError('Quota failure archive already exists; never overwrite it')
        original_hash = sha(path)
        path.rename(archive)
        assert sha(archive) == original_hash
        recovered.append(dict(unit_id=unit['id'], archived_response=str(archive),
                              archived_response_sha256=original_hash, raw_attempt=str(attempt_path),
                              request_sha256=unit['request_sha256']))
    with (output / 'quota_recovery_records.jsonl').open('a') as handle:
        handle.write(json.dumps(dict(time=time.time(), confirmed_quota_errors=recovered,
                                    prompt_and_configuration_unchanged=True)) + '\n')
    return recovered


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('pilot', 'run', 'evaluate'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume-quota-failures', action='store_true',
                        help='Only after quota is restored: archive confirmed 402 failures and retry unchanged requests')
    args = parser.parse_args()
    if args.stage == 'evaluate' and args.resume_quota_failures:
        parser.error('--resume-quota-failures applies only to pilot/run')
    output = args.output.resolve()
    base_path = Path(__file__).with_name('execute_frozen.py')
    spec = importlib.util.spec_from_file_location('_ppi_frozen_dispatch_base', base_path)
    base = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(base)
    original_loader = base.load_snapshot

    def load_snapshot(name, snapshot, original):
        module = original_loader(name, snapshot, original)
        if name == '_frozen_ppi_ablation_runner' and args.stage == 'pilot':
            def pilot(directory):
                module.verify_freeze(directory)
                config = module.read(directory / 'config.json')
                plan = module.read(directory / 'plan.json')
                unit = next((row for row in plan if not (directory / 'responses' / (row['id'] + '.json')).exists()
                             or module.read(directory / 'responses' / (row['id'] + '.json')).get('status') != 'ok'), plan[0])
                with (directory / 'run.lock').open('a') as lock:
                    module.fcntl.flock(lock, module.fcntl.LOCK_EX | module.fcntl.LOCK_NB)
                    credential = module.load_credential(config['credential_file'])
                    response = module.run_unit(directory, unit, config, credential, module.legacy_common())
                report = dict(unit_id=unit['id'], request_sha256=unit['request_sha256'],
                              status=response['status'], model=response.get('model'),
                              http_status=response.get('http_status'),
                              frozen_scientific_configuration_unchanged=True,
                              credential_persisted=False, finished=time.time())
                module.atomic(directory / 'pilot_status.json', report)
                print(json.dumps(report, indent=2), flush=True)
            module.run = pilot
        if name == '_frozen_ppi_ablation_runner' and args.resume_quota_failures:
            original_run = module.run
            def run_with_quota_recovery(directory):
                module.verify_freeze(directory)
                with (directory / 'run.lock').open('a') as lock:
                    module.fcntl.flock(lock, module.fcntl.LOCK_EX | module.fcntl.LOCK_NB)
                    recover_quota_failures(module, directory)
                return original_run(directory)
            module.run = run_with_quota_recovery
        return module

    base.load_snapshot = load_snapshot
    output.mkdir(parents=True, exist_ok=True)
    dispatch_snapshot = output / 'execution_code_snapshot' / (sha(__file__) + '.py')
    dispatch_snapshot.parent.mkdir(exist_ok=True)
    if not dispatch_snapshot.exists():
        with dispatch_snapshot.open('xb') as handle:
            handle.write(Path(__file__).read_bytes())
    assert sha(dispatch_snapshot) == sha(__file__)
    provenance = dict(stage=args.stage, dispatch_path=str(Path(__file__).resolve()),
                      dispatch_snapshot=str(dispatch_snapshot),
                      dispatch_sha256=sha(__file__), base_wrapper_sha256=sha(base_path),
                      resume_quota_failures=args.resume_quota_failures,
                      scientific_prompts_configuration_and_scoring_unchanged=True,
                      started=time.time())
    log = output / 'frozen_dispatch_records.jsonl'
    with log.open('a') as handle:
        handle.write(json.dumps(provenance) + '\n')
    sys.argv = [str(base_path), 'run' if args.stage == 'pilot' else args.stage,
                '--output', str(output)]
    base.main()


if __name__ == '__main__':
    main()
