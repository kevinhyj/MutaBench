"""Freeze the official TDC archive without resplitting, subsampling or deduplicating."""
import argparse
import csv
import hashlib
import io
import json
import math
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

# Counts independently transcribed from AnewDDE Table S3, page 37.
# Positive classification semantics; regression direction is numerical assay value.
ENDPOINTS = {
 'hia_hou': (117, 'classification', 'Human intestinal absorption; higher score means more likely high absorption (positive class).'),
 'pgp_broccatelli': (245, 'classification', 'P-glycoprotein inhibition; higher score means more likely an inhibitor.'),
 'bioavailability_ma': (128, 'classification', 'Oral bioavailability; higher score means more likely orally bioavailable (positive class).'),
 'bbb_martins': (406, 'classification', 'Blood-brain barrier penetration; higher score means more likely BBB permeable.'),
 'herg': (132, 'classification', 'hERG potassium channel inhibition; higher score means more likely a hERG blocker.'),
 'ames': (1457, 'classification', 'Ames bacterial mutagenicity; higher score means more likely mutagenic (positive Ames test).'),
 'dili': (96, 'classification', 'Drug-induced liver injury; higher score means more likely to cause liver injury.'),
 'caco2_wang': (182, 'regression', 'Caco-2 cell permeability, log10(cm/s); higher numerical readout means greater permeability.'),
 'lipophilicity_astrazeneca': (840, 'regression', 'Lipophilicity, experimental octanol/water logD at pH 7.4; higher means more lipophilic.'),
 'solubility_aqsoldb': (1997, 'regression', 'Aqueous solubility, log10(mol/L); higher numerical readout means greater solubility.'),
 'ppbr_az': (559, 'regression', 'Plasma protein binding percentage; higher means more protein-bound, not more unbound.'),
 'vdss_lombardo': (226, 'regression', 'Steady-state volume of distribution (L/kg); higher means larger volume of distribution.'),
 'half_life_obach': (135, 'regression', 'Elimination half-life in hours; higher means longer half-life.'),
 'clearance_microsome_az': (221, 'regression', 'Intrinsic clearance measured in liver microsomes; higher means faster clearance, not greater metabolic stability.'),
 'clearance_hepatocyte_az': (243, 'regression', 'Intrinsic clearance measured in hepatocytes; higher means faster clearance, not greater metabolic stability.'),
 'ld50_zhu': (1478, 'regression', 'Rat acute oral toxicity LD50 encoded as log10(1/(mol/kg)); higher encoded value means lower molar lethal dose and greater toxicity.'),
}
for enzyme, count in [('2c9', 2419), ('2d6', 2626), ('3a4', 2467)]:
 ENDPOINTS[f'cyp{enzyme}_veith'] = (count, 'classification', f'CYP{enzyme.upper()} inhibition; higher score means more likely an inhibitor.')
 ENDPOINTS[f'cyp{enzyme}_substrate_carbonmangels'] = (135, 'classification', f'CYP{enzyme.upper()} substrate status; higher score means more likely metabolized as a substrate by this enzyme, not inhibition.')


def sha(path):
 return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
 path = Path(path)
 path.parent.mkdir(parents=True, exist_ok=True)
 payload = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n'
 if path.exists():
  if path.read_text() != payload:
   raise ValueError(f'Refusing to overwrite frozen artifact {path}')
 else:
  path.write_text(payload)


def prepare(root):
 root = Path(root)
 if (root / 'manifest.json').exists():
  m = json.loads((root / 'manifest.json').read_text())
  for name, digest in m['hashes'].items():
   if sha(root / name) != digest:
    raise ValueError(f'Frozen data changed: {name}')
  return m
 z = zipfile.ZipFile(root / 'raw/admet_group.zip')
 tasks, labels, audits = [], {}, []
 for ep, (expected, kind, description) in ENDPOINTS.items():
  rows_by_split = {}
  for split in ('train_val', 'test'):
   raw = z.read(f'admet_group/{ep}/{split}.csv')
   path = root / 'raw/official' / ep / f'{split}.csv'
   path.parent.mkdir(parents=True, exist_ok=True)
   if path.exists() and path.read_bytes() != raw:
    raise ValueError('Raw file changed')
   path.write_bytes(raw)
   rows_by_split[split] = list(csv.DictReader(io.StringIO(raw.decode('utf-8-sig'))))
  rows = rows_by_split['test']
  assert len(rows) == expected, (ep, len(rows), expected)
  candidates, target, source_ids = [], {}, {}
  for i, row in enumerate(rows, 1):
   key = f'M{i:05d}'
   y = float(row['Y'])
   assert math.isfinite(y) and row['Drug'].strip()
   if kind == 'classification':
    assert y in (0, 1)
   candidates.append({'id': key, 'smiles': row['Drug']})
   target[key] = y
   source_ids[key] = {'Drug_ID': row['Drug_ID'], 'test_csv_row_1based': i + 1}
  task = {'endpoint': ep, 'kind': kind, 'description': description, 'candidates': candidates}
  tasks.append(task)
  labels[ep] = {'y': target, 'source_ids': source_ids}
  test_smiles = Counter(r['Drug'] for r in rows)
  train_smiles = {r['Drug'] for r in rows_by_split['train_val']}
  audits.append({'endpoint': ep, 'kind': kind, 'test_count': len(rows), 'paper_count': expected,
                 'train_val_count': len(rows_by_split['train_val']), 'metric': 'auroc' if kind == 'classification' else 'spearman',
                 'duplicate_test_smiles_extra_rows': sum(n-1 for n in test_smiles.values()),
                 'exact_smiles_train_test_overlap_unique': len(set(test_smiles) & train_smiles),
                 'policy': 'Preserve official rows including duplicates; no train labels used in zero-shot inference.',
                 'class_counts': dict(Counter(target.values())) if kind == 'classification' else None})
  write(root / 'processed' / f'{ep}.public.json', task)
 write(root / 'frozen/tasks.public.json', tasks)
 write(root / 'labels/labels.local.json', labels)
 write(root / 'audit.json', audits)
 files = [p for p in root.rglob('*') if p.is_file() and p.name != 'manifest.json']
 m = {'schema': 'tdc-admet-anewdde-v1', 'created_utc': datetime.now(timezone.utc).isoformat(),
      'endpoint_count': len(tasks), 'test_records': sum(len(t['candidates']) for t in tasks),
      'sources': {'archive': 'https://dataverse.harvard.edu/api/access/datafile/4426004',
                  'tdc': 'https://tdcommons.ai/benchmark/admet_group/overview/',
                  'report': 'https://anewlabs-research.tos-cn-beijing.volces.com/anewdde-tech-report.pdf'},
      'paper_count_match': True, 'paper_row_hash_manifest_available': False,
      'license_note': 'Retain TDC/source attribution; per-dataset terms differ. Local research intake; no redistribution authorization asserted.',
      'hashes': {str(p.relative_to(root)): sha(p) for p in sorted(files)}}
 assert m['test_records'] == 16379 and m['endpoint_count'] == 22
 write(root / 'manifest.json', m)
 return m


if __name__ == '__main__':
 p = argparse.ArgumentParser()
 p.add_argument('--data-root', type=Path, required=True)
 args = p.parse_args()
 m = prepare(args.data_root)
 print(json.dumps({k: v for k, v in m.items() if k != 'hashes'}, indent=2))
