"""Label-free, resumable paired ADMET inference. Evaluation lives in evaluate.py."""
import argparse
import concurrent.futures
import fcntl
import hashlib
import json
import math
import os
import random
import shlex
import threading
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from evagent.integrated.gateway import RequestGateway, digest

SYSTEM = ('Predict the assayed molecular property using only the supplied complete SMILES and assay '
          'definition. No external tools or retrieval are available. Candidates are independent molecules; '
          'there is no shared wild type. Higher predicted values mean higher numerical assay readouts '
          'or greater probability of the specified positive class, not necessarily better drug properties. '
          'Return only the requested JSON.')


def load(path):
 return json.loads(Path(path).read_text())


def sha(path):
 return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
 path = Path(path)
 path.parent.mkdir(parents=True, exist_ok=True)
 temp = path.with_name(path.name + f'.{os.getpid()}.{threading.get_ident()}.tmp')
 temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
 temp.replace(path)


def connected(ids, rounds):
 adjacency = {i: set() for i in ids}
 for groups in rounds:
  for group in groups:
   for other in group[1:]:
    adjacency[group[0]].add(other)
    adjacency[other].add(group[0])
 seen, pending = set(), [ids[0]]
 while pending:
  i = pending.pop()
  if i not in seen:
   seen.add(i)
   pending.extend(adjacency[i] - seen)
 return len(seen) == len(ids)


def schedule(ids):
 # Equal exposure: each molecule exactly once per round, matched across methods.
 # Randomized positions, not exact position balancing from unavailable paper code.
 n_groups = math.ceil(len(ids) / 25)
 for attempt in range(1000):
  rounds = []
  for r in range(3):
   order = list(ids)
   random.Random(1 + 1000 * attempt + r).shuffle(order)
   q, rem = divmod(len(order), n_groups)
   sizes = [q + (j < rem) for j in range(n_groups)]
   offset, groups = 0, []
   for size in sizes:
    groups.append(order[offset:offset+size])
    offset += size
   rounds.append(groups)
  if connected(ids, rounds):
   return rounds
 raise ValueError('Disconnected comparison graph')


def prompt(task, ids, method):
 by_id = {c['id']: c for c in task['candidates']}
 user = 'Assay definition:\n' + task['description'] + '\n\nComplete candidates:\n'
 user += '\n'.join(json.dumps(by_id[i], sort_keys=True) for i in ids)
 if method == 'cr':
  instruction = ('Estimate an independent continuous property score for every candidate on one shared '
                 'scale across independent batches: -1 is very low, 0 is intermediate, and +1 is very high '
                 'for the measured property. Do not normalize scores to the current batch. Every score '
                 'must be finite and within [-1, 1]. Return ONLY {"variants":[{"variant_id":"ID",'
                 '"predicted_score":0.0},...]} with exactly these two fields per item.')
 else:
  instruction = ('Rank every candidate from highest to lowest predicted numerical assay readout '
                 '(or probability of the defined positive class). Return ONLY {"ranking":["ID",...]} '
                 'with no additional fields.')
 return user + '\n\n' + instruction + '\nInclude every required ID exactly once: ' + ', '.join(ids)


def parse(text, ids, method):
 text = text.strip()
 if text.startswith('```') and text.endswith('```'):
  text = text[text.index('\n')+1:-3].strip()
 data = json.loads(text)
 if method == 'rank':
  if not isinstance(data, dict) or set(data) != {'ranking'}:
   raise ValueError('Invalid ranking schema')
  v = data['ranking']
  if not isinstance(v, list) or any(type(i) is not str for i in v) or len(v) != len(ids) or set(v) != set(ids):
   raise ValueError('Missing, duplicate or unknown ranking IDs')
  return {'ranking': v}
 if not isinstance(data, dict) or set(data) != {'variants'} or not isinstance(data['variants'], list):
  raise ValueError('Invalid CR schema')
 scores = {}
 for row in data['variants']:
  if not isinstance(row, dict) or set(row) != {'variant_id', 'predicted_score'}:
   raise ValueError('Invalid CR row')
  i, v = row['variant_id'], row['predicted_score']
  if not isinstance(i, str) or i not in ids or i in scores or type(v) not in (int, float) or not math.isfinite(v) or not -1 <= v <= 1:
   raise ValueError('Invalid CR ID or value')
  scores[i] = float(v)
 if set(scores) != set(ids):
  raise ValueError('Missing CR IDs')
 return {'scores': scores}


def prepare(root, data_root, model_config):
 root = Path(root)
 if (root / 'plan.json').exists():
  return validate(root)
 tasks = load(data_root / 'frozen/tasks.public.json')
 # Strict input allowlist: measured labels and source names never enter inference.
 for task in tasks:
  if set(task) != {'endpoint', 'kind', 'description', 'candidates'}:
   raise ValueError('Unexpected task fields')
  for c in task['candidates']:
   if set(c) != {'id', 'smiles'} or not c['smiles']:
    raise ValueError('Unexpected molecular fields')
 tasks.sort(key=lambda t: (len(t['candidates']), t['endpoint']))
 units = []
 for ti, t in enumerate(tasks):
  ids = [c['id'] for c in t['candidates']]
  for r, groups in enumerate(schedule(ids)):
   for gi, group in enumerate(groups):
    # Interleave methods so elapsed-time drift does not favor one complete arm.
    for method in (('cr', 'rank') if (ti+r+gi) % 2 == 0 else ('rank', 'cr')):
     units.append({'unit_id': f'{t["endpoint"]}.r{r}.{method}.g{gi:03d}',
                   'endpoint': t['endpoint'], 'round': r, 'method': method,
                   'group': gi, 'ids': group, 'system': SYSTEM, 'user': prompt(t, group, method)})
 save(root / 'tasks.public.json', tasks)
 save(root / 'queue.json', units)
 # Snapshot code and dependencies that actually affect inference and evaluation.
 source = Path(__file__).parent
 for name in ('campaign.py', 'prepare.py', 'evaluate.py'):
  save_path = root / 'source' / name
  save_path.parent.mkdir(parents=True, exist_ok=True)
  save_path.write_bytes((source / name).read_bytes())
 gateway_path = source.parent / 'integrated/gateway.py'
 (root / 'source/gateway.py').write_bytes(gateway_path.read_bytes())
 plan = {'schema': 'admet-cr-elo-v1', 'created_utc': datetime.now(timezone.utc).isoformat(),
         'model_config': model_config, 'token_budget': None, 'budget_mode': 'unlimited',
         'authorization': 'User: 就是现在的配置，预算无上限', 'rounds': 3, 'max_group_size': 25,
         'concurrency': 4, 'max_semantic_repairs': 1, 'automatic_transport_retries': 0,
         'seed': 1, 'elo': {'initial': 1000, 'k': 16, 'scale': 400, 'shuffle_seeds': list(range(1000,1100))},
         'primary_comparison': 'CR3 mean continuous scores versus listwise-Elo3, same batches and three passes. CR1 separately reports single-pass efficiency.',
         'scope': {'endpoints': len(tasks), 'test_records': sum(len(t['candidates']) for t in tasks), 'base_requests': len(units)},
         'paper_replication': False,
         'limitations': ['Original prompt, scheduling details, Elo hyperparameters and per-row manifest unavailable.',
                        'Randomized candidate positions are not exact paper position balancing.',
                        'Equal exposure and output cap do not guarantee equal realized token cost.',
                        'Public historical test data; unknown LLM pretraining exposure.'],
         'data_manifest_sha256': sha(data_root / 'manifest.json'),
         'hashes': {str(p.relative_to(root)): sha(p) for p in sorted(root.rglob('*')) if p.is_file()}}
 save(root / 'plan.json', plan)
 return validate(root)


def validate(root):
 plan = load(root / 'plan.json')
 for name, h in plan['hashes'].items():
  if sha(root / name) != h:
   raise ValueError('Frozen artifact changed: ' + name)
 for name in ('campaign.py', 'prepare.py', 'evaluate.py'):
  if sha(Path(__file__).parent / name) != sha(root / 'source' / name):
   raise ValueError('Runtime differs from frozen source: ' + name)
 if sha(Path(__file__).parent.parent / 'integrated/gateway.py') != sha(root / 'source/gateway.py'):
  raise ValueError('Gateway differs from frozen source')
 return plan


def credentials(project):
 if not os.environ.get('GEMINI_API_KEY'):
  path = Path(load(project / 'experiments/proteingym_cr_msa_remainder_v1/plan.json')['credentials_file'])
  for line in path.read_text().splitlines():
   text = line.strip().removeprefix('export ')
   if text.startswith('GEMINI_API_KEY='):
    values = shlex.split(text.split('=', 1)[1])
    if len(values) == 1:
     os.environ['GEMINI_API_KEY'] = values[0]
     break
 if not os.environ.get('GEMINI_API_KEY'):
  raise ValueError('Missing provider credential')
 os.environ['LANGSMITH_TRACING'] = 'false'
 os.environ['LANGCHAIN_TRACING_V2'] = 'false'


def run(root, max_units=None):
 root = Path(root)
 plan = validate(root)
 credentials(root.parents[1])
 queue = load(root / 'queue.json')
 stop = threading.Event()
 lease = (root / 'worker.lock').open('a+')
 fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
 gateway = RequestGateway(model_config=plan['model_config'], run_dir=root / 'provider',
                          fingerprint=digest(plan), token_budget=None, budget_mode='unlimited')
 def execute(unit):
  if stop.is_set() or (root / 'PAUSE').exists():
   raise RuntimeError('Paused')
  path = root / 'units' / (unit['unit_id'] + '.json')
  if path.exists():
   result = load(path)
   if result['input_sha256'] != digest(unit) or path.with_suffix('.sha256').read_text().strip() != sha(path):
    raise ValueError('Cached unit identity/hash mismatch')
   return result
  attempts, user = [], unit['user']
  result, status, issue = {}, 'invalid', None
  for attempt in range(2):
   if stop.is_set() or (root / 'PAUSE').exists():
    raise RuntimeError('Paused')
   raw = gateway.generate(system=unit['system'], user=user,
        run_name=unit['unit_id']+f'.a{attempt}',
        metadata={k: unit[k] for k in ('unit_id','endpoint','round','method','group')} | {'attempt_index': attempt})
   attempts.append({'request_sha256': raw['request_sha256'], 'status': raw['status'], 'cache_hit': raw.get('cache_hit', False)})
   if raw['status'] != 'ok':
    status, issue = raw['status'], raw['status']
    break
   try:
    result = parse(raw['final_text'], unit['ids'], unit['method'])
    status = 'ok'
    break
   except (ValueError, TypeError, KeyError) as exc:
    issue = str(exc)
    user = unit['user'] + '\nCorrect your JSON once. Previous response:\n' + raw['final_text'] + '\nIssue: ' + issue
  record = {k: unit[k] for k in ('unit_id','endpoint','round','method','group','ids')}
  record.update(input_sha256=digest(unit), status=status, issue=issue, attempts=attempts, result=result)
  save(path, record)
  path.with_suffix('.sha256').write_text(sha(path)+'\n')
  return record
 def progress(state):
  outcomes = [load(p) for p in (root / 'units').glob('*.json')] if (root / 'units').exists() else []
  save(root / 'STATUS.json', {'state': state, 'updated_utc': datetime.now(timezone.utc).isoformat(),
       'planned_units': len(queue), 'finished_units': len(outcomes),
       'unit_statuses': dict(Counter(x['status'] for x in outcomes)), 'usage': gateway.summary()})
 try:
  # Validate cached records before skipping them; never silently retry an invalid result.
  pending = []
  for unit in queue:
   if (root / 'units' / (unit['unit_id']+'.json')).exists():
    execute(unit)
   else:
    pending.append(unit)
  if max_units is not None:
   pending = pending[:max_units]
  progress('running')
  with concurrent.futures.ThreadPoolExecutor(max_workers=plan['concurrency']) as pool:
   for offset in range(0, len(pending), plan['concurrency']):
    futures = [pool.submit(execute, u) for u in pending[offset:offset+plan['concurrency']]]
    try:
     for f in concurrent.futures.as_completed(futures):
      f.result()
      progress('running')
    except BaseException:
     stop.set()
     raise
  progress('queue_finished' if max_units is None else 'preflight_finished')
 except BaseException as exc:
  stop.set()
  save(root / 'failure.json', {'exception_type': type(exc).__name__, 'note': 'Inspect ledger and service log; no automatic retry of ambiguous cost.'})
  progress('paused_failure')
  raise
 finally:
  gateway.close()
  lease.close()


def main():
 p = argparse.ArgumentParser()
 p.add_argument('command', choices=['prepare','validate','run'])
 p.add_argument('--root', type=Path, required=True)
 p.add_argument('--data-root', type=Path)
 p.add_argument('--model-config', type=Path)
 p.add_argument('--max-units', type=int)
 args = p.parse_args()
 if args.command == 'prepare':
  print(json.dumps(prepare(args.root, args.data_root, load(args.model_config))['scope']))
 elif args.command == 'validate':
  print(json.dumps(validate(args.root)['scope']))
 else:
  run(args.root, args.max_units)


if __name__ == '__main__':
 main()
