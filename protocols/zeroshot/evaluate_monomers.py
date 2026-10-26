"""Evaluator-only metrics and matched-cohort comparisons for monomer campaigns."""
import argparse
import collections
import csv
import math
from pathlib import Path

from scipy.stats import rankdata, spearmanr
from shared_protocol import save, sha, require, finite
from monomer_integrity import (strict_load as load, task_metric, check_rounds, check_receipt,
                               check_result_inventory)


def metric(kind, labels, scores):
    require(len(labels) == len(scores) and len(labels) > 0, 'Unequal/empty metric vectors')
    require(all(finite(x) for x in labels + scores), 'Nonfinite metric input')
    if kind == 'classification':
        require(set(labels) <= {0, 1}, 'AUROC requires binary labels')
        n1 = sum(labels); n0 = len(labels)-n1
        if not n0 or not n1: return None, 'single_class_labels'
        # Mann-Whitney U, with average tied ranks. Constant predictions yield 0.5.
        ranks = rankdata(scores, method='average')
        return float((sum(r for r,y in zip(ranks, labels) if y == 1)-n1*(n1+1)/2)/(n1*n0)), None
    require(kind == 'regression', 'Unknown metric kind')
    if len(set(labels)) < 2 and len(set(scores)) < 2:return None,'constant_labels_and_predictions'
    if len(set(labels)) < 2:return None,'constant_labels'
    if len(set(scores)) < 2:return None,'constant_predictions'
    value = float(spearmanr(labels, scores).statistic)
    return (value, None) if math.isfinite(value) else (None, 'undefined_correlation')


def merge_predictions(requests, directory, model_key=None):
    """Never score a partial round; reject duplicate IDs and mismatched request receipts."""
    directory=Path(directory);model_key=model_key or directory.parent.name
    require(len({r['request_id'] for r in requests})==len(requests),'Duplicate planned requests')
    all_ids=[i for r in requests for i in r['expected_ids']]
    require(len(all_ids)==len(set(all_ids)),'Duplicate candidate within round')
    predictions = {}; failures = []
    for request in requests:
        rid = request['request_id']; path = directory/(rid+'.json')
        if not path.exists(): failures.append('not_completed'); continue
        require(sha(path) == path.with_suffix('.sha256').read_text().strip(), 'Prediction hash mismatch')
        row = load(path)
        check_receipt(row,request,model_key)
        if row['status'] != 'ok': failures.append(row.get('failure_category','failed')); continue
        scores = row['value']; require(set(scores) == set(request['expected_ids']), 'Prediction ID mismatch')
        require(not set(scores) & set(predictions), 'Duplicate score within round')
        require(all(finite(v) and -1 <= v <= 1 for v in scores.values()), 'Invalid prediction value')
        predictions.update(scores)
    return predictions, failures


def paired_summary(rows):
    groups = collections.defaultdict(dict)
    for r in rows:
        if r['status'] == 'complete' and r['value'] is not None and r['cohort'] != 'short_query':
            groups[(r['domain'],r['arm'],r['aggregation'],r['metric'])].setdefault(r['model_key'], {})[
                (r['assay'],r['query_sha256'])] = r['value']
    result = []
    # Require every planned model, including those with zero complete assays.
    all_models = sorted({r['model_key'] for r in rows})
    for (domain, arm, aggregation, name), by_model in sorted(groups.items()):
        common = set.intersection(*(set(by_model.get(m,{})) for m in all_models))
        result.append(dict(domain=domain,arm=arm,aggregation=aggregation,metric=name,
            common_assays=len(common),identities=[list(k) for k in sorted(common)],
            models={m:sum(by_model[m][k] for k in common)/len(common) if common else None for m in all_models}))
    return result


def evaluate(run):
    from monomer_campaign import validate
    run = Path(run).resolve(); plan, requests = validate(run)
    refs = {(r['domain'],r['assay']):r for r in load(run/'EVALUATOR_REFERENCES.json')}
    grouped = collections.defaultdict(list)
    for request in requests:
        grouped[(request['domain'],request['assay'],request['arm'])].append(request)
    rows = []; gold_cache = {}
    for model in plan['models']:check_result_inventory(run/'models'/model/'units',requests)
    for model in plan['models']:
        for (domain, assay, arm), units in grouped.items():
            ref = refs[(domain,assay)]; qp = Path(ref['query_path'])
            kind,metric_name=task_metric(domain,ref)
            require(sha(qp) == ref['query_sha256'], 'Evaluation query changed')
            ids = [c['id'] for c in load(qp)['candidates']]
            require(len(ids) == len(set(ids)), 'Duplicate evaluation ID')
            check_rounds(units,domain,ids)
            rounds = collections.defaultdict(list)
            for r in units: rounds[r.get('round_index',0)].append(r)
            aggregations = {'CR1': (0,), 'CR3': (0,1,2)} if domain == 'admet' else {'single_pass': (0,)}
            for aggregation, required_rounds in aggregations.items():
                observations = []; errors = []; counts = []
                for r in required_rounds:
                    require(r in rounds, 'Missing planned round')
                    pred, failures = merge_predictions(rounds[r], run/'models'/model/'units',model)
                    errors.extend(failures); counts.append(len(pred))
                    if set(pred) != set(ids): errors.append('incomplete_cohort')
                    observations.append(pred)
                value = reason = None
                if not errors:
                    key = (domain,assay)
                    if key not in gold_cache:
                        lp = Path(ref['label_path']); require(sha(lp) == ref['label_sha256'], 'Evaluation labels changed')
                        blob = load(lp)
                        labels = blob[ref['label_key']]['y'] if ref.get('label_container') == 'admet_endpoint_y' else blob['labels']
                        require(set(labels) == set(ids), 'Label identity mismatch')
                        gold_cache[key] = labels
                    labels = gold_cache[key]
                    scores = [sum(p[i] for p in observations)/len(observations) for i in ids]
                    value, reason = metric(kind, [labels[i] for i in ids], scores)
                rows.append(dict(model_key=model,domain=domain,assay=assay,arm=arm,aggregation=aggregation,
                    metric=metric_name,value=value,status='incomplete' if errors else
                        'undefined_metric' if value is None else 'complete',
                    coverage_status='incomplete' if errors else 'complete',
                    metric_status='not_computed' if errors else 'undefined' if value is None else 'defined',
                    reason=';'.join(sorted(set(errors))) if errors else reason,
                    n_expected=len(ids), n_predicted=min(counts),
                    cohort='official_test' if domain == 'admet' else 'N250' if len(ids) == 250 else 'short_query',
                    query_sha256=ref['query_sha256']))
    out = run/'evaluation'; save(out/'METRICS.json',dict(rows=rows,partial_not_scored=True,missing_not_zero=True))
    save(out/'PAIRED.json',paired_summary(rows))
    with (out/'per_assay.csv').open('w',newline='') as f:
        w = csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    return rows


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--run',required=True,type=Path)
    a = p.parse_args(); rows = evaluate(a.run)
    print(dict(cells=len(rows),complete=sum(r['status']=='complete' for r in rows)))
