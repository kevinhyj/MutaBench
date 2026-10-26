"""Archive confirmed pre-connect local sandbox failures, then rerun frozen units.

Not a retry of provider refusals, HTTP failures, or requests of uncertain delivery.
The original prompts, config, frozen runner and all failure records are preserved.
"""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--credential-file', type=Path, required=True)
    parser.add_argument('--confirmed-local-socket-denial', action='store_true', required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    read = lambda p: json.loads(Path(p).read_text())
    archive = out / 'sandbox_failure_history'
    if archive.exists():
        raise ValueError('Local failure history already archived; use the ordinary frozen runner for further status/resume')
    index = read(out / 'index.json')
    attempts = sorted((out / 'attempts').glob('*/*.json'))
    responses = sorted((out / 'responses').glob('*.json'))
    if len(responses) != len(index) or len(attempts) != 3 * len(index):
        raise ValueError('Expected exactly the completed pre-connect local failure stage')
    for path in attempts:
        record = read(path)
        if (record['status'] != 'transport_error' or record.get('http_status') is not None
                or record.get('body') != {} or record['parsed'].get('error_type') != 'ConnectError'
                or record['finished'] - record['started'] > 0.1):
            raise ValueError('Failure may have reached provider or has uncertain delivery; cannot use this local-only recovery')
    for path in responses:
        if read(path)['status'] != 'transport_error':
            raise ValueError('Cannot recover a provider response with the local-sandbox path')
    archive.mkdir()
    shutil.move(str(out / 'attempts'), str(archive / 'attempts'))
    shutil.move(str(out / 'responses'), str(archive / 'responses'))
    (archive / 'recovery.json').write_text(json.dumps(dict(
        archived_unix=time.time(), local_client_attempts=len(attempts), Gemini_scoring_submissions=0,
        reason='Python socket.socket() in the original filesystem sandbox raised PermissionError errno=1 before any connection. All 60 HTTP client failures were immediate ConnectError with empty body/no HTTP status.',
        request_count=len(index), frozen_prompts_changed=False, generation_config_changed=False,
        provider_refusal_retried=False, recovery='Use explicitly approved network permissions with the same frozen requests'
    ), indent=2) + '\n')
    runner = Path(__file__).with_name('protein_rna_structure_v3.py')
    result = subprocess.run([sys.executable, '-B', str(runner), 'run', '--output', str(out),
                             '--credential-file', str(args.credential_file)])
    raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
