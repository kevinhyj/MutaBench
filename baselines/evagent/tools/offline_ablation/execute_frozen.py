"""Execute an immutable code snapshot while other workspace work continues.

The original preparation freeze remains unchanged. Snapshot SHA256s must match
that freeze exactly; changes to a working source file cannot alter this run.
"""
import argparse
import hashlib
import importlib
import json
import sys
import types
from pathlib import Path


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def load_snapshot(name, snapshot, original):
    module = types.ModuleType(name)
    module.__file__ = original
    module.__package__ = name.rpartition('.')[0]
    sys.modules[name] = module
    exec(compile(Path(snapshot).read_bytes(), original, 'exec'), module.__dict__)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('run', 'evaluate'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    freeze = json.loads((output / 'freeze.json').read_text())
    snapshot_directory = output / 'execution_code_snapshot'
    snapshot_directory.mkdir(exist_ok=True)
    manifest = {}
    for original, expected in freeze['code_hashes'].items():
        target = snapshot_directory / (expected + '.py')
        if not target.exists():
            if sha(original) != expected:
                raise ValueError(f'Original source changed before snapshot: {original}')
            with target.open('xb') as handle:
                handle.write(Path(original).read_bytes())
        if sha(target) != expected:
            raise ValueError('Execution code snapshot hash mismatch')
        manifest[original] = dict(snapshot=str(target), sha256=expected)
    manifest_path = output / 'execution_snapshot_manifest.json'
    manifest_text = json.dumps(manifest, indent=2) + '\n'
    if manifest_path.exists() and manifest_path.read_text() != manifest_text:
        raise ValueError('Execution snapshot manifest changed')
    if not manifest_path.exists():
        manifest_path.write_text(manifest_text)

    code = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(code))
    importlib.import_module('evagent')
    core_path = str(code / 'evagent/offline_ablation.py')
    feature_path = str(Path(__file__).with_name('features.py'))
    runner_path = str(Path(__file__).with_name('run.py'))
    common_path = next(path for path in manifest if path.endswith('/complex_sequence_all_assays_N250_N10_seed42_20260928/common.py'))
    load_snapshot('evagent.offline_ablation', manifest[core_path]['snapshot'], core_path)
    load_snapshot('features', manifest[feature_path]['snapshot'], feature_path)
    common = load_snapshot('_frozen_sequence_common', manifest[common_path]['snapshot'], common_path)
    runner = load_snapshot('_frozen_ppi_ablation_runner', manifest[runner_path]['snapshot'], runner_path)

    def verify_snapshot(directory):
        if Path(directory).resolve() != output:
            raise ValueError('Execution output differs from the frozen snapshot')
        for relative, expected in freeze['artifact_hashes'].items():
            if sha(output / relative) != expected:
                raise ValueError(f'Frozen data artifact changed: {relative}')
        for entry in manifest.values():
            if sha(entry['snapshot']) != entry['sha256']:
                raise ValueError('Frozen execution code changed')
        for entry in json.loads((output / 'input_snapshot.json').read_text())['inputs'].values():
            if sha(entry['path']) != entry['sha256']:
                raise ValueError('new/ input changed')

    runner.verify_freeze = verify_snapshot
    runner.legacy_common = lambda: common
    execution_amendment = dict(
        wrapper_path=str(Path(__file__).resolve()), wrapper_sha256=sha(__file__),
        source_code_sha256s_unchanged=True,
        verification='Validate immutable snapshots against original freeze hashes instead of mutable workspace files',
        scientific_prompts_configuration_and_scoring_unchanged=True)
    amendment_path = output / 'execution_snapshot_amendment.json'
    amendment_text = json.dumps(execution_amendment, indent=2) + '\n'
    if amendment_path.exists() and amendment_path.read_text() != amendment_text:
        raise ValueError('Execution wrapper changed')
    if not amendment_path.exists():
        amendment_path.write_text(amendment_text)
    verify_snapshot(output)
    {'run': runner.run, 'evaluate': runner.evaluate}[args.stage](output)


if __name__ == '__main__':
    main()
