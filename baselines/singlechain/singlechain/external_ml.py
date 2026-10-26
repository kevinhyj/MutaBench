"""External-supervised ligand transfer; external labels only, with molecule-overlap audit."""
import csv
import math
from pathlib import Path
import numpy as np
from .common import load, freeze, sha


def identity(smiles):
    from rdkit import Chem
    from rdkit.Chem.MolStandardize import rdMolStandardize as std
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError('Invalid SMILES')
    mol = std.Cleanup(mol)
    mol = std.FragmentParent(mol)
    mol = std.Uncharger().uncharge(mol)
    Chem.RemoveStereochemistry(mol)
    # Standard InChI connectivity block collapses stereochemistry and common prototropic tautomers.
    # This avoids silently accepting a truncated tautomer-enumeration search.
    key = Chem.MolToInchiKey(mol)
    if not key:
        raise ValueError('Molecule identity normalization failed')
    return key.split('-')[0]


def features(smiles):
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048, includeChirality=False)
    return np.asarray([generator.GetFingerprintAsNumPy(Chem.MolFromSmiles(s)) for s in smiles], dtype=np.float32)


def fit_esol(source, target_public_json, out, estimator='rf'):
    import joblib
    import sklearn
    from rdkit import rdBase
    # Full test-pool public file contains only SMILES and IDs, no test labels.
    target_ids, invalid_targets = set(), []
    for row in load(target_public_json)['candidates']:
        try:
            target_ids.add(identity(row['smiles']))
        except ValueError:
            invalid_targets.append(row['id'])
    with Path(source).open() as f:
        raw = list(csv.DictReader(f))
    train, seen, overlap, invalid = [], set(), 0, 0
    for row in raw:
        try:
            smi = row['smiles']
            key = identity(smi)
            y = float(row['measured log solubility in mols per litre'])
            if not math.isfinite(y):
                raise ValueError('Nonfinite external label')
            if key in target_ids:
                overlap += 1
                continue
            if key in seen:
                continue
            seen.add(key); train.append((smi, y, key))
        except (ValueError, KeyError):
            invalid += 1
    train.sort(key=lambda x: x[2])
    if len(train) < 100:
        raise ValueError('Too few external training molecules')
    if estimator == 'rf':
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(n_estimators=500, random_state=42, n_jobs=2, max_features=1.0)
    elif estimator == 'xgboost':
        from xgboost import XGBRegressor
        model = XGBRegressor(n_estimators=500, max_depth=6, learning_rate=.05, random_state=42, n_jobs=2)
    else:
        raise ValueError('Unknown external estimator')
    model.fit(features([x[0] for x in train]), [x[1] for x in train])
    out = Path(out); out.mkdir(parents=True, exist_ok=False)
    joblib.dump(model, out / 'model.joblib')
    params = {k: (str(v) if isinstance(v,float) and not math.isfinite(v) else v)
              for k,v in model.get_params().items()}
    card = {'status': 'complete', 'estimator': estimator, 'source': str(Path(source).resolve()),
        'source_url': 'https://github.com/deepchem/deepchem/blob/master/datasets/delaney-processed.csv',
        'source_sha256': sha(source), 'endpoint': 'solubility_aqsoldb', 'training_endpoint': 'ESOL measured log molar aqueous solubility',
        'target_test_source': str(Path(target_public_json).resolve()), 'target_test_source_sha256': sha(target_public_json),
        'target_labels_read': False, 'source_n': len(raw), 'external_train_n': len(train),
        'overlap_rows_removed': overlap, 'invalid_external_rows': invalid,
        'invalid_target_ids_not_admitted': invalid_targets,
        'identity_rule': 'RDKit cleanup, fragment parent, uncharge, remove stereochemistry, standard InChIKey connectivity block',
        'training_identities': sorted(seen), 'audited_target_identities': sorted(target_ids),
        'overlap_after_filter': len(seen & target_ids), 'sklearn': sklearn.__version__, 'rdkit': rdBase.rdkitVersion,
        'model_sha256': sha(out / 'model.joblib'), 'seed': 42, 'features': 'ECFP4 Morgan radius=2, 2048 bits, no chirality',
        'hyperparameters': params, 'tuning': 'none',
        'comparison_group': 'external_supervised_transfer',
        'caveat': 'Molecule-disjoint external supervision, not label-free pretraining; close analog and measurement-source overlap not excluded'}
    freeze(out / 'manifest.json', card)
    return {k:v for k,v in card.items() if k not in ('training_identities', 'audited_target_identities', 'hyperparameters')}


def predict(item, model_dir):
    import joblib
    root = Path(model_dir)
    card = load(root / 'manifest.json')
    if item['modality'] != 'ligand' or item['endpoint']['name'] != card['endpoint']:
        raise ValueError('External model endpoint mismatch')
    if card['status'] != 'complete' or sha(root / 'model.joblib') != card['model_sha256']:
        raise ValueError('External model is incomplete or changed')
    candidates = [c['sequence'] for c in item['candidates']]
    keys = {identity(s) for s in candidates}
    if keys & set(card['training_identities']) or not keys <= set(card['audited_target_identities']):
        raise ValueError('Test overlap or target cohort not covered by external training audit')
    # Only load trusted locally created artifact after verifying its frozen manifest.
    model = joblib.load(root / 'model.joblib')
    values = model.predict(features([item['base']] + candidates)) * item['endpoint']['label_sign']
    return list(values[1:]), float(values[0]), {'model_manifest_sha256': sha(root / 'manifest.json'),
        'estimator': card['estimator'], 'external_train_n': card['external_train_n'],
        'overlap_after_filter': 0, 'labels_read': False, 'supervision': 'external ESOL only'}
