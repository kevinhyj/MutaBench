"""Read-only cost and matched-cohort reporting, separate from inference."""
import argparse
import json
import sqlite3
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from .evaluate import evaluate


def report(root, data_root):
    result = evaluate(root, data_root)
    db = sqlite3.connect(f'file:{root / "provider/request_ledger.sqlite"}?mode=ro', uri=True)
    costs = defaultdict(lambda: {'attempts': 0, 'reported_tokens': 0, 'unknown_usage_attempts': 0, 'statuses': Counter()})
    for request, status, tokens in db.execute('SELECT request_json,status,tokens FROM attempts WHERE status != "not_submitted"'):
        meta = json.loads(request)['metadata']
        arm = meta['method']
        for key in (arm, meta['endpoint'] + '/' + arm):
            costs[key]['attempts'] += 1
            costs[key]['statuses'][status] += 1
            costs[key]['reported_tokens'] += tokens or 0
            costs[key]['unknown_usage_attempts'] += int(tokens is None or status == 'inflight' or 'unknown' in status)
    db.close()
    result['cost_by_method_and_endpoint'] = dict(costs)
    result['generated_utc'] = datetime.now(timezone.utc).isoformat()
    (root/'analysis/report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    lines = ['# ADMET CR vs AnewDDE-style Elo', '', f'Checked: {result["generated_utc"]}', '',
             'Only full candidate coverage yields an endpoint metric. Missing values are not zero.', '',
             '| Endpoint | Metric | N | CR1 | CR3 | Elo3 | CR3 - Elo3 | Valid requests |',
             '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    def fmt(v):
        return 'pending/invalid' if v is None else f'{v:.4f}'
    for r in result['endpoints']:
        lines.append(f'| {r["endpoint"]} | {r["metric"]} | {r["n"]} | {fmt(r["CR1"])} | {fmt(r["CR3"])} | {fmt(r["Elo3"])} | {fmt(r["paired_delta_CR3_minus_Elo3"])} | {r["valid_units"]}/{r["planned_units"]} |')
    lines.extend(['', '## Cost', '', '| Arm | Attempts | Reported tokens | Unknown / inflight |', '| --- | ---: | ---: | ---: |'])
    for arm in ('cr', 'rank'):
        c = costs[arm]
        lines.append(f'| {arm} | {c["attempts"]} | {c["reported_tokens"]} | {c["unknown_usage_attempts"]} |')
    lines.extend(['', 'CR3 and Elo3 use matched sample exposure and model settings; realized token costs can differ.',
                  'The original paper did not disclose all harness details. This is a reconstruction, not AnewMind model reproduction.',
                  'Partial coverage cannot establish the full-benchmark macro average. No confidence interval or superiority claim is inferred from point estimates alone.', ''])
    (root/'analysis/REPORT.md').write_text('\n'.join(lines))
    return result


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--data-root',type=Path,required=True)
    args=p.parse_args()
    r=report(args.root,args.data_root)
    print(json.dumps({'macro_common_endpoints':r['macro_common_endpoints'], 'cost':{k:v for k,v in r['cost_by_method_and_endpoint'].items() if '/' not in k}},indent=2))
