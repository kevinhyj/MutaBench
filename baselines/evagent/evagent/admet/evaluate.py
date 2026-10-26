"""Offline evaluation; never import a model client or send test labels to inference."""
import argparse
import csv
import hashlib
import itertools
import json
import sqlite3
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, spearmanr


def load(path):
 return json.loads(Path(path).read_text())


def sha(path):
 return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def auroc(y, scores):
 y = np.asarray(y)
 pos, neg = int(sum(y == 1)), int(sum(y == 0))
 if not pos or not neg:
  return None
 ranks = rankdata(scores, method='average')
 return float((sum(ranks[y == 1]) - pos * (pos + 1) / 2) / (pos * neg))


def metric(kind, y, scores):
 if kind == 'classification':
  return auroc(y, scores)
 if len(set(y)) < 2 or len(set(scores)) < 2:
  return None
 return float(spearmanr(y, scores).statistic)


def elo(ids, groups, shuffles=100):
 index = {key: i for i, key in enumerate(ids)}
 edges = [(index[a], index[b]) for group in groups for a,b in itertools.combinations(group, 2)]
 accumulated = np.zeros(len(ids))
 for seed in range(1000, 1000+shuffles):
  ratings = [1000.0] * len(ids)
  for ei in np.random.default_rng(seed).permutation(len(edges)):
   a, b = edges[int(ei)]
   expected = 1 / (1 + 10 ** ((ratings[b] - ratings[a]) / 400))
   delta = 16 * (1 - expected)
   ratings[a] += delta
   ratings[b] -= delta
  accumulated += ratings
 return dict(zip(ids, (accumulated / shuffles).tolist()))


def evaluate(root, data_root):
 root, data_root = Path(root), Path(data_root)
 plan = load(root / 'plan.json')
 manifest = load(data_root / 'manifest.json')
 assert sha(data_root / 'manifest.json') == plan['data_manifest_sha256']
 for name, h in plan['hashes'].items():
  assert sha(root / name) == h, name
 for name in ('labels/labels.local.json', 'frozen/tasks.public.json'):
  assert sha(data_root / name) == manifest['hashes'][name], name
 tasks, labels = load(root / 'tasks.public.json'), load(data_root / 'labels/labels.local.json')
 queue = load(root / 'queue.json')
 by_ep = defaultdict(list)
 db = sqlite3.connect(f'file:{root / "provider/request_ledger.sqlite"}?mode=ro', uri=True)
 ledger = {r[0]: (r[1], json.loads(r[2])) for r in db.execute('SELECT request_sha256,status,request_json FROM attempts')}
 db.close()
 for u in queue:
  by_ep[u['endpoint']].append(u)
 rows, predictions = [], []
 for task in tasks:
  ep, ids = task['endpoint'], [c['id'] for c in task['candidates']]
  valid = defaultdict(dict)
  for u in by_ep[ep]:
   path = root / 'units' / (u['unit_id']+'.json')
   if not path.exists():
    continue
   assert sha(path) == path.with_suffix('.sha256').read_text().strip()
   out = load(path)
   expected = hashlib.sha256(json.dumps(u, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()
   assert out['input_sha256'] == expected
   assert all(out[k] == u[k] for k in ('unit_id','endpoint','round','method','group','ids'))
   for attempt in out['attempts']:
    status, req = ledger[attempt['request_sha256']]
    assert status != 'not_submitted'
    assert req['metadata']['unit_id'] == u['unit_id']
   if out['status'] == 'ok':
    valid[(u['method'],u['round'])][u['group']] = out['result']
  def complete(method, rounds):
   return all(len(valid[(method, r)]) == sum(u['method'] == method and u['round'] == r for u in by_ep[ep]) for r in rounds)
  methods = {}
  for name, rounds in [('CR1',[0]), ('CR3',[0,1,2])]:
   if complete('cr', rounds):
    observations = defaultdict(list)
    for r in rounds:
     for result in valid[('cr',r)].values():
      for i,v in result['scores'].items():
       observations[i].append(v)
    assert set(observations) == set(ids) and all(len(v) == len(rounds) for v in observations.values())
    methods[name] = {i: float(np.mean(v)) for i,v in observations.items()}
  if complete('rank',[0,1,2]):
   groups = [valid[('rank',r)][g]['ranking'] for r in range(3) for g in sorted(valid[('rank',r)])]
   methods['Elo3'] = elo(ids, groups)
  y = [labels[ep]['y'][i] for i in ids]
  row = {'endpoint': ep, 'kind': task['kind'], 'n': len(ids),
         'metric': 'AUROC' if task['kind'] == 'classification' else 'Spearman',
         'valid_units': sum(len(v) for v in valid.values()), 'planned_units': len(by_ep[ep]),
         'CR1': None, 'CR3': None, 'Elo3': None, 'paired_delta_CR3_minus_Elo3': None}
  for name, scores in methods.items():
   row[name] = metric(task['kind'], y, [scores[i] for i in ids])
   predictions.extend({'endpoint': ep, 'id': i, 'method': name, 'prediction': scores[i]} for i in ids)
  if row['CR3'] is not None and row['Elo3'] is not None:
   row['paired_delta_CR3_minus_Elo3'] = row['CR3']-row['Elo3']
  rows.append(row)
 summary = {'endpoints': rows, 'scope': plan['scope'], 'macro_common_endpoints': {}}
 for kind in ('classification','regression'):
  cohort = [r for r in rows if r['kind'] == kind and r['paired_delta_CR3_minus_Elo3'] is not None]
  summary['macro_common_endpoints'][kind] = {'n_endpoints': len(cohort),
     'expected_endpoints': 13 if kind == 'classification' else 9,
     'CR3': float(np.mean([r['CR3'] for r in cohort])) if cohort else None,
     'Elo3': float(np.mean([r['Elo3'] for r in cohort])) if cohort else None,
     'endpoint_ids': [r['endpoint'] for r in cohort]}
 summary['note'] = 'Partial coverage is not the full benchmark. No metric is computed on a silently reduced candidate cohort. Realized costs may differ between methods.'
 out = root / 'analysis'
 out.mkdir(exist_ok=True)
 (out / 'results.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
 for name, data in [('metrics.csv',rows),('predictions.csv',predictions)]:
  if data:
   with (out / name).open('w') as handle:
    writer=csv.DictWriter(handle,fieldnames=list(data[0])); writer.writeheader(); writer.writerows(data)
 return summary


if __name__ == '__main__':
 p = argparse.ArgumentParser()
 p.add_argument('--root', type=Path, required=True)
 p.add_argument('--data-root', type=Path, required=True)
 args = p.parse_args()
 print(json.dumps(evaluate(args.root,args.data_root)['macro_common_endpoints'],indent=2))
