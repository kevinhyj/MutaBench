"""One documented format-only calibration revision; no effect metrics are read."""
import json
import shutil
from pathlib import Path

from evagent.integrated.bundles import file_hash

ROOT = Path('/data/yanjie_huang/EVAgent/experiments')
OLD = ROOT/'cr_ncrna_evidence_b_v1'
NEW = ROOT/'cr_ncrna_evidence_b_v2'


def save(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(obj, indent=2)+'\n'
    if p.exists() and p.read_text() != text:
        raise ValueError('Refusing to overwrite changed frozen revision')
    p.write_text(text)


def main():
    old = json.loads((OLD/'campaign.json').read_text())
    report = json.loads((OLD/'campaign/report.json').read_text())
    if report['status'] != 'calibration_failed' or report['ledger']['requests_by_status'] != {'ok': 3, 'truncated': 1}:
        raise ValueError('This revision is only for the observed known-cost output truncation')
    ledger = OLD/'campaign/provider/request_ledger.sqlite'
    revised = {**old, 'output_dir': str(NEW/'campaign'),
        'model_config': {**old['model_config'], 'max_tokens': 32768},
        'prior_ledgers': [{'path': str(ledger), 'sha256': file_hash(ledger)}]}
    for row in revised['cells']:
        if row['status'] != 'eligible':
            continue
        p = Path(row['config'])
        cfg = json.loads(p.read_text())
        cfg['max_tokens'] = 32768
        cfg['output_dir'] = str(NEW/Path(cfg['output_dir']).relative_to(OLD))
        new_path = NEW/p.relative_to(OLD)
        save(new_path, cfg)
        row['config'] = str(new_path)
    save(NEW/'campaign.json', revised)
    shutil.copyfile(OLD/'coverage.json', NEW/'coverage.json')
    save(NEW/'revision.json', {'prior_campaign': str(OLD), 'reason': 'Known MAX_TOKENS termination during no-evidence calibration; final output incomplete',
        'change': {'max_tokens': [8192, 32768]}, 'unchanged': ['model', 'thinking', 'prompts', 'samples', 'evidence', 'repair limit', 'seed'],
        'prior_reported_tokens': report['ledger']['reported_tokens'], 'total_budget': 50_000_000,
        'prior_ledger': revised['prior_ledgers'][0], 'accuracy_used_for_decision': False,
        'frozen_assets': {'path': str(OLD/'assets_manifest.json'), 'sha256': file_hash(OLD/'assets_manifest.json')}})
    print(json.dumps({'config': str(NEW/'campaign.json'), 'prior_tokens': report['ledger']['reported_tokens']}))


if __name__ == '__main__':
    main()
