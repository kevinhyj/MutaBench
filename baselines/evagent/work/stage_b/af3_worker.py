"""Two single-GPU WT-only AF3 workers with immutable receipts and input hashes."""
import concurrent.futures
import hashlib
import json
import subprocess
import time
from pathlib import Path

ROOT = Path('/data/huangyanjie/EVAgent-stage-b')
IMAGE = 'sha256:3d32f89387a265de0e49fc1f9b823704c3f87c24bea1b2477409cd476d3bf7c4'
WEIGHTS = 'f03ba90fae974e518e2218fa485127dd1ac723fcb496515e0b519d99fd06e457'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def run(p, gpu):
    name = p.stem
    out = ROOT/'outputs/af3'/name
    out.mkdir(parents=True, exist_ok=True)
    receipt = out/'receipt.json'
    if receipt.exists():
        value = json.loads(receipt.read_text())
        if value['input_sha256'] != sha(p):
            raise ValueError('Frozen AF3 input changed')
        return value
    memory = subprocess.check_output(['nvidia-smi', '-i', str(gpu), '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True)
    if int(memory.strip()) > 100:
        raise RuntimeError(f'GPU {gpu} is no longer idle; no other process stopped')
    args = ['docker', 'run', '--rm', '--gpus', f'device={gpu}',
        '-e', 'XLA_PYTHON_CLIENT_PREALLOCATE=false',
        '-v', f'{p.parent}:/input:ro',
        '-v', '/data/huangyanjie/alphafold3_repo/models:/models:ro',
        '-v', f'{out}:/output', '--entrypoint', 'python', IMAGE,
        '/app/alphafold/run_alphafold.py', f'--json_path=/input/{p.name}',
        '--model_dir=/models', '--output_dir=/output', '--run_data_pipeline=false',
        '--run_inference=true', '--flash_attention_implementation=xla',
        '--num_diffusion_samples=5', '--num_recycles=10']
    before = time.time()
    with (out/'run.log').open('w') as log:
        result = subprocess.run(args, stdout=log, stderr=subprocess.STDOUT)
    value = {'input_sha256': sha(p), 'image': IMAGE, 'weights_sha256': WEIGHTS,
        'repository_commit': '3dfc5a91beee404474d614df1031fbdefaef3277',
        'gpu': gpu, 'seed': 1, 'samples': 5, 'recycles': 10, 'command': args,
        'elapsed_s': time.time()-before, 'returncode': result.returncode,
        'outputs': {str(f.relative_to(out)): sha(f) for f in out.rglob('*') if f.is_file()}}
    value['status'] = 'complete' if result.returncode == 0 and (out/name/f'{name}_model.cif').exists() else 'failed'
    receipt.write_text(json.dumps(value, indent=2)+'\n')
    print(json.dumps({'job': name, 'status': value['status'], 'elapsed_s': value['elapsed_s']}), flush=True)
    return value


def main():
    paths = sorted((ROOT/'assets/af3_inputs').glob('*.json'))
    # GPU indices were checked at startup and are rechecked before every task.
    def lane(items, gpu):
        for p in items:
            run(p, gpu)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        jobs = [pool.submit(lane, paths[i::2], gpu) for i, gpu in enumerate((1, 2))]
        for future in jobs:
            future.result()


if __name__ == '__main__':
    main()
