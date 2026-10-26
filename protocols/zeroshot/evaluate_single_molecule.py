"""Independent evaluator: complete-query signed Spearman or endpoint AUROC.

ADMET CR1 uses round zero; CR3 averages all three complete rounds by row ID.
Only this process opens evaluation label contents, after predictions exist.
"""
import argparse
import collections
import csv
import json
import math
from pathlib import Path

from scipy.stats import rankdata, spearmanr
from single_molecule_protocol import load, save, sha, require, finite


def metric_value(metric, labels, predictions, ids):
    y, scores = [labels[i] for i in ids], [predictions[i] for i in ids]
    require(all(finite(v) for v in y+scores), 'Nonfinite labels/predictions')
    if metric == 'auroc':
        require(set(y) <= {0, 1}, 'Nonbinary classification labels')
        positives = sum(y); negatives = len(y)-positives
        if not positives or not negatives: return None
        ranks = rankdata(scores, method='average')
        return float((sum(r for r, label in zip(ranks, y) if label == 1)-positives*(positives+1)/2)/(positives*negatives))
    require(metric == 'spearman', 'Unknown evaluation metric')
    if len(set(y)) < 2 or len(set(scores)) < 2: return None
    value = float(spearmanr(y, scores).statistic)
    return value if math.isfinite(value) else None


def merge_round(run, units, ids):
    prediction, failed = {}, False
    for unit in units:
        p = run/'units'/(unit['logical_id']+'.json')
        if not p.exists(): continue
        require(sha(p) == p.with_suffix('.sha256').read_text().strip(), 'Prediction checksum mismatch')
        result = load(p)
        if result.get('status') != 'ok': failed = True; continue
        values = result['value']
        require(set(values) == set(unit['template']['expected_ids']), 'Prediction IDs differ')
        require(not set(prediction) & set(values), 'Duplicate predicted ID within a round')
        require(all(finite(v) and -1 <= v <= 1 for v in values.values()), 'Invalid score')
        prediction.update(values)
    status = 'complete' if set(prediction) == set(ids) and not failed else 'failed' if failed else 'incomplete'
    return status, prediction


def evaluate(run):
    plan = load(run/'PLAN.json'); inputs = Path(plan['inputs'])
    require(sha(inputs/'MANIFEST.json') == plan['input_manifest_sha256'], 'Input changed')
    refpath = Path(plan['evaluator_references'])
    require(sha(refpath) == plan['evaluator_references_sha256'], 'Evaluator references changed')
    require(sha(run/'QUEUE.json') == plan['queue_sha256'], 'Evaluation queue changed')
    refs = {(r['domain'], r['assay']): r for r in load(refpath)}
    cells = collections.defaultdict(list)
    for q in load(run/'QUEUE.json'): cells[q['domain'], q['assay'], q['arm']].append(q)
    rows = []
    label_cache = {}
    for (domain, assay, arm), units in cells.items():
        ref = refs[domain, assay]
        require(sha(ref['query_path']) == ref['query_sha256'], 'Query changed')
        query = load(ref['query_path']); ids = [c['id'] for c in query['candidates']]
        if domain == 'admet':
            rounds = [merge_round(run, [u for u in units if u['round_index']==r], ids) for r in range(3)]
            complete = all(s == 'complete' for s, _ in rounds)
            averaged = {i: sum(p[i] for _, p in rounds)/3 for i in ids} if complete else {}
            conditions = [(arm+'_cr1', *rounds[0]),
                          (arm+'_cr3', 'complete' if complete else 'failed' if any(s=='failed' for s,_ in rounds) else 'incomplete', averaged)]
        else:
            conditions = [(arm, *merge_round(run, units, ids))]
        for condition, status, pred in conditions:
            value = None
            if status == 'complete':
                path = ref['label_path']
                if path not in label_cache:
                    require(sha(path) == ref['label_sha256'], 'Labels changed')
                    label_cache[path] = load(path)
                raw = label_cache[path]
                labels = raw[assay]['y'] if ref['label_format']=='admet_endpoint_y' else raw['labels']
                require(set(labels) == set(ids), 'Label identities differ from exact query')
                value = metric_value(ref['metric'], labels, pred, ids)
            rows.append(dict(domain=domain, assay=assay, arm=condition, status=status,
                n_expected=len(ids), n_predicted=len(pred), metric=ref['metric'], metric_value=value,
                spearman=value if ref['metric']=='spearman' else None, auroc=value if ref['metric']=='auroc' else None,
                cohort='official_test' if domain=='admet' else 'N250' if len(ids)==250 else 'short_query'))
    out = run/'evaluation'; out.mkdir(exist_ok=True)
    save(out/'METRICS.json', dict(rows=rows, partial_not_scored=True, undefined_not_zero=True,
                                 signed_spearman=True, admet_cr3='mean of three complete rounds, by stable official row ID'))
    with (out/'per_assay.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    return rows


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--run', type=Path, required=True); a = p.parse_args()
    rows = evaluate(a.run.resolve())
    print(json.dumps(dict(cells=len(rows), complete=sum(r['status']=='complete' for r in rows))))
