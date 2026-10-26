"""Resume a cloned PPI thinking run after native Google 400s on sequence schema.

Prompt text, seed, temperature and thinkingLevel stay. Only responseSchema /
responseMimeType are removed from sequence requests that native generateContent
rejected. Completed ok/invalid responses are copied; the 400 records are not.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path('/data/huangyanjie/EVAgent')
CODE = ROOT / 'code'
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(CODE / 'tools/offline_ablation'))
from features import save
from evagent.offline_ablation import digest, file_hash

SKIP = {'freeze.json', 'execution_snapshot_manifest.json', 'execution_snapshot_amendment.json',
        'run_state.json', 'dispatch_run.log', 'run.lock', 'pilot_status.json',
        'frozen_dispatch_records.jsonl'}


def load(path: Path):
    return json.loads(path.read_text())


def resume(source: Path, output: Path) -> None:
    source, output = source.resolve(), output.resolve()
    if (output / 'freeze.json').exists():
        raise ValueError('Already frozen: ' + str(output))
    shutil.copytree(source, output, ignore=lambda directory, names: [
        name for name in names
        if name in SKIP or name == 'execution_code_snapshot'
        or (Path(directory).name == 'responses' and name.endswith('.json') and _is_schema_400(Path(directory) / name))
    ])
    stripped = []
    for path in sorted((output / 'requests').glob('*__sequence.json')):
        request = load(path)
        gen = request.get('generationConfig', {})
        if 'responseSchema' not in gen and 'responseMimeType' not in gen:
            continue
        contents = request['contents']
        gen.pop('responseSchema', None)
        gen.pop('responseMimeType', None)
        if request['contents'] != contents:
            raise ValueError('Prompt contents changed')
        path.write_text(json.dumps(request, ensure_ascii=False, indent=2) + '\n')
        stripped.append(path.name)
    plan = load(output / 'plan.json')
    for unit in plan:
        path = output / unit['request_path']
        unit['request_sha256'] = file_hash(path)
        unit['semantic_request_sha256'] = digest(load(path))
    (output / 'plan.json').write_text(json.dumps(plan, ensure_ascii=False, indent=2) + '\n')
    config = load(output / 'config.json')
    config['native_sequence_schema_stripped'] = stripped
    config['resume_source'] = str(source)
    (output / 'config.json').write_text(json.dumps(config, ensure_ascii=False, indent=2) + '\n')
    files = sorted(path for path in output.rglob('*') if path.is_file() and path.name != 'freeze.json')
    code_paths = [CODE / 'tools/offline_ablation/run.py', CODE / 'tools/offline_ablation/features.py',
                  CODE / 'evagent/offline_ablation.py', ROOT / 'baseline/singlechain/models.py',
                  ROOT / 'baseline/singlechain/common.py',
                  ROOT / 'explanation_test/complex_sequence_all_assays_N250_N10_seed42_20260928/common.py']
    save(output / 'freeze.json', dict(artifact_hashes={str(path.relative_to(output)): file_hash(path) for path in files},
                                     code_hashes={str(path): file_hash(path) for path in code_paths}))
    print('RESUMED', output, 'stripped', len(stripped), flush=True)


def _is_schema_400(path: Path) -> bool:
    if not path.exists():
        return False
    row = load(path)
    return row.get('http_status') == 400 and row.get('condition') == 'sequence'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    resume(args.source, args.output)


if __name__ == '__main__':
    main()
