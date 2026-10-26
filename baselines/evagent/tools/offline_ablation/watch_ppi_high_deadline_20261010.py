"""Restart HIGH r4 shards until every retry unit has a response file."""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

OUT = Path('/data/huangyanjie/EVAgent/experiments/ppi_cr25_gemini38_high_deadline_r2_20261010')
PY = Path('/data/huangyanjie/af3_test/bin/python')
SRC = Path('/data/huangyanjie/EVAgent/code/tools/offline_ablation/recover_ppi_high_deadline_20261010.py')
PENDING = OUT / 'pending_r4.json'
SHARDS = [
    (20, 'http://127.0.0.1:17890', 16),
    (21, 'http://127.0.0.1:17892', 16),
    (22, 'http://127.0.0.1:17893', 16),
]
LOG = OUT / 'watchdog.log'


def log(message: str) -> None:
    line = time.strftime('%H:%M:%S') + ' ' + message
    with LOG.open('a') as handle:
        handle.write(line + '\n')
    print(line, flush=True)


def alive(shard: int) -> bool:
    try:
        out = subprocess.check_output(
            ['pgrep', '-af', f'recover_ppi_high_deadline_20261010.py run --output {OUT} --shard-id {shard} '],
            text=True,
        )
    except subprocess.CalledProcessError:
        return False
    return 'python' in out


def start(shard: int, proxy: str, concurrency: int) -> None:
    stdout = OUT / f'shard_{shard}_stdout.log'
    cmd = [str(PY), '-u', str(SRC), 'run', '--output', str(OUT),
           '--shard-id', str(shard), '--shards', '3', '--proxy', proxy,
           '--concurrency', str(concurrency), '--pending-file', str(PENDING)]
    with stdout.open('ab') as handle:
        subprocess.Popen(cmd, stdout=handle, stderr=handle, start_new_session=True)


def pending_units_in_plan_order() -> list[str]:
    pending = set(json.loads(PENDING.read_text()))
    plan = json.loads((OUT / 'plan.json').read_text())
    return [unit['id'] for unit in plan if unit['id'] in pending]


def remaining(shard: int | None = None) -> int:
    n = 0
    for index, unit_id in enumerate(pending_units_in_plan_order()):
        if shard is not None and index % 3 != shard % 3:
            continue
        if not (OUT / 'responses' / f'{unit_id}.json').exists():
            n += 1
    return n


def main() -> None:
    log('watchdog r4 start remaining=' + str(remaining()))
    for shard, proxy, conc in SHARDS:
        if not alive(shard):
            start(shard, proxy, conc)
            log(f'start shard {shard} {proxy} c={conc}')
            time.sleep(2)
    while True:
        left = remaining()
        if left == 0:
            log('complete remaining=0')
            return
        for shard, proxy, conc in SHARDS:
            if not alive(shard) and remaining(shard) > 0:
                start(shard, proxy, conc)
                log(f'restart shard {shard} remaining={left} shard_missing={remaining(shard)}')
        time.sleep(45)


if __name__ == '__main__':
    main()
