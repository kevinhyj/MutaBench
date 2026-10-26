"""Compute secondary evidence for already admitted, label-free worker inputs."""
import concurrent.futures
import json
import os
from pathlib import Path

from evagent.integrated.evidence_tools import canonical, secondary_features
from evagent.integrated.gateway import assert_public, digest

ROOT = Path('/data/huangyanjie/EVAgent-stage-b')


def run(path):
    import RNA
    task = json.loads(path.read_text())
    assert_public(task)
    out = ROOT/'outputs/secondary'/path.name
    if out.exists():
        if json.loads(out.read_text())['provenance'][0]['input_sha256'] != digest(task):
            raise ValueError('Secondary evidence input changed')
        return {'assay': task['assay'], 'status': 'existing'}
    result = secondary_features(canonical(task['wt_sequence']), task['variants'])
    result['provenance'] = [{'id': 'secondary', 'tool': 'ViennaRNA', 'version': RNA.__version__,
                             'sha256': digest(result), 'input_sha256': digest(task),
                             'parameters': {'temperature_c': 37.0, 'sequence_normalization': 'T_to_U'}}]
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_suffix('.tmp')
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    temporary.replace(out)
    return {'assay': task['assay'], 'status': 'complete', 'variants': len(task['variants'])}


if __name__ == '__main__':
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    paths = sorted((ROOT/'assets/tasks').glob('*.json'))
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(run, p): p for p in paths}
        for future in concurrent.futures.as_completed(futures):
            p = futures[future]
            try:
                print(json.dumps(future.result()), flush=True)
            except Exception as e:
                print(json.dumps({'assay': p.stem, 'status': 'failed', 'error': str(e)}), flush=True)
