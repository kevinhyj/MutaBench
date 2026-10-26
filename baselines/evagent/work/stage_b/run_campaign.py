"""Run the frozen campaign using host-local credentials, then write its report."""
import argparse
import json
import os
from pathlib import Path

from evagent.integrated.campaign import run_campaign

ROOT = Path('/data/yanjie_huang/EVAgent/experiments/cr_ncrna_evidence_b_v1')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=ROOT/'campaign.json')
    config = parser.parse_args().config.resolve()
    # Read only the named credential; never log the file, value, or provider headers.
    for line in Path('/home/yanjie/.config/evagent/credentials.env').read_text().splitlines():
        line = line.strip().removeprefix('export ')
        if line.startswith('GEMINI_API_KEY='):
            os.environ['GEMINI_API_KEY'] = line.split('=', 1)[1].strip().strip('\"\'')
    if not os.environ.get('GEMINI_API_KEY'):
        raise RuntimeError('Required host-local credential is unavailable')
    result = run_campaign(config)
    lines = ['# RNAGym ncRNA Stage B', '', f"Campaign status: **{result['status']}**", '',
        f"Complete results: {result['complete_cells']} / {result['eligible_cells']} eligible; 310 matrix slots, 305 after sample-size feasibility.",
        f"Reported tokens: {result['ledger']['reported_tokens']:,} / 50,000,000.",
        f"Actual provider attempts: {result['ledger']['provider_attempts']}; provider elapsed seconds: {result['ledger']['provider_elapsed_s']:.1f}.",
        '', '| Status | Cells |', '|---|---:|']
    lines += [f'| {k} | {v} |' for k, v in sorted(result['statuses'].items())]
    lines += ['', '| N | Condition | Paired assays | Families | Family macro ΔSpearman | Family macro ΔKendall |', '|---:|---|---:|---:|---:|---:|']
    lines += [f"| {r['n']} | {r['condition']} | {r['paired_assays']} | {r['families']} | {r['family_macro_delta_spearman']:.5f} | {r['family_macro_delta_kendall']:.5f} |" for r in result['comparisons']]
    lines += ['', 'No failed/missing condition is imputed from another method. Paired comparisons use complete matched results only; per-assay results and family-bootstrap intervals are in report.json.',
        'Evidence is text/numeric multisource augmentation. Structure predictions may depend on MSA; literature uses reviewed task/family/target background; consult coverage and review-scope receipts. Source-unresolved and pathogen-derived tasks are coverage-only.',
        'A low number of independent RNA families limits uncertainty estimates. This controls external evidence leakage, not pretrained-model exposure.', '']
    (config.parent/'REPORT.md').write_text('\n'.join(lines))
    print(json.dumps({k: result[k] for k in ('status', 'statuses', 'complete_cells', 'eligible_cells', 'ledger')}, indent=2))
    return 0 if result['status'] == 'complete' else 2


if __name__ == '__main__':
    raise SystemExit(main())
