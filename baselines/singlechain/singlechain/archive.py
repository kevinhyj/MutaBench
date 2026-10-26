"""Re-evaluate published per-variant scores; never label them new model inference."""
import csv
import math
from pathlib import Path
from .common import load, normalize, records, freeze, sha, validate_predictions


def import_scores(item, input_path, spec_path, out):
    spec = load(spec_path)
    if spec['assay'] != item['assay'] or spec['modality'] != item['modality']:
        raise ValueError('Archive assay/modality mismatch')
    source = Path(spec['source_csv'])
    if sha(source) != spec['source_sha256']:
        raise ValueError('Archive source changed')
    if spec['score_column'].lower() in ('dms_score', 'dms_score_bin', 'y', 'fitness', 'label', 'score'):
        raise ValueError('Ambiguous or experimental label column cannot be used as predictor')
    wanted = {c['sequence'] for c in item['candidates']}
    found = {}
    with source.open(newline='', encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            # Only read identity and the explicitly reviewed model score column.
            seq = normalize(row[spec['sequence_column']], item['modality'])
            if seq not in wanted:
                continue
            if row.get('sequence') and row.get('mutated_sequence'):
                if normalize(row['sequence'],item['modality']) != normalize(row['mutated_sequence'],item['modality']):
                    raise ValueError('Archive modeled sequence differs from full source sequence')
            value = float(row[spec['score_column']])
            if not math.isfinite(value):
                raise ValueError('Nonfinite published score')
            if seq in found and not math.isclose(found[seq], value, rel_tol=1e-6, abs_tol=1e-7):
                raise ValueError('Conflicting published scores for the same sequence')
            found[seq] = value
    if set(found) != wanted:
        raise ValueError(f'Archive incomplete on fixed cohort: {len(found)}/{len(wanted)}; no imputation')
    values = records(item, [found[c['sequence']] for c in item['candidates']],
                     rationale='Published model score re-evaluated on seed42 cohort; base=0 is an arbitrary additive origin, not measured WT fitness.')
    validate_predictions(item, values)
    out = Path(out)
    freeze(out / 'predictions.json', values)
    receipt = {'status': 'complete', 'method': spec['method'], 'n': item['n'], 'seed': 42,
        'input_sha256': sha(input_path), 'predictions_sha256': sha(out / 'predictions.json'),
        'source_type': 'published_archive_re_evaluation', 'comparison_group': spec['comparison_group'],
        'independent_zero_shot_admitted': True, 'smoke_only': False,
        'model_inference_performed_this_run': False, 'provenance': spec,
        'mapping': 'exact full normalized sequence; complete cohort; no sign selection',
        'model_training_exposure': 'unknown', 'checkpoint_scope': 'published archive snapshot; not newly verified model weights'}
    freeze(out / 'receipt.json', receipt)
    return {'method': spec['method'], 'source_type': receipt['source_type'], 'n': item['n']}
