"""Snapshot existing aggregate metrics only; never read candidate predictions."""
import argparse
import csv
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    explanation = root / 'explanation_test'
    sequence = explanation / 'complex_sequence_all_assays_N250_N10_seed42_20260928'
    cohort = {row['source'] + '__' + row['assay_group_id']
              for row in json.loads((sequence / 'task_index.json').read_text())
              if row['source'] in ('abagym', 'bindinggym')}
    paths = {'sequence': sequence / 'assay_spearman.csv'}
    for arm in ('msa_prompt', 'msa_gemini_rank5050', 'msa_prompt_structure', 'msa_gemini_rank5050_structure'):
        paths[arm] = explanation / 'complex_ablation_n250_n10_seed42_20260928' / arm / 'assay_spearman.csv'
    rows, sources = [], {}
    for arm, path in paths.items():
        sources[arm] = dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        with path.open() as handle:
            for row in csv.DictReader(handle):
                group_id = row['source'] + '__' + row['assay_group_id']
                if group_id not in cohort:
                    continue
                rows.append(dict(group_id=group_id, historical_arm=arm,
                                 status=row['status'], N=row['N'], scored=row['scored'],
                                 spearman=row['spearman'], response_models=row['response_models'],
                                 definition='historical prompt-instructed fusion, not independently computed rank fusion'
                                            if 'rank5050' in arm else 'historical unchanged metric',
                                 origin=str(path), is_new_evaluation=False))
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / 'historical_aggregate_metrics_88.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    (args.output / 'historical_metrics_provenance.json').write_text(json.dumps(dict(
        assay_count=len(cohort), sources=sources, candidate_predictions_read=False,
        new_model_calls=0, new_rankings_computed=False), indent=2) + '\n')
    print('Existing aggregate metrics snapshotted:', len(rows), 'rows for', len(cohort), 'assays')


if __name__ == '__main__':
    main()
