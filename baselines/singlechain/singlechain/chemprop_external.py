"""Chemprop trained only on the same filtered external ESOL pool as RF."""
import csv
from pathlib import Path
import numpy as np
from .common import freeze, load, sha
from .external_ml import identity


def fit(external_card, out):
    import torch
    import chemprop
    from chemprop import data, featurizers, models, nn
    from lightning import pytorch as pl
    torch.set_num_threads(2)
    pl.seed_everything(42, workers=True)
    card = load(external_card)
    if card['status'] != 'complete' or card['overlap_after_filter'] != 0:
        raise ValueError('Completed external-data overlap audit required')
    if sha(card['source']) != card['source_sha256']:
        raise ValueError('External training source changed')
    keys = set(card['training_identities'])
    seen, train = set(), []
    with Path(card['source']).open() as f:
        for row in csv.DictReader(f):
            key = identity(row['smiles'])
            if key in keys and key not in seen:
                seen.add(key)
                train.append((key, row['smiles'], float(row['measured log solubility in mols per litre'])))
    if seen != keys:
        raise ValueError('External pool differs from frozen audit')
    train.sort()
    points = [data.MoleculeDatapoint.from_smi(smi, np.asarray([y], dtype=np.float32)) for _, smi, y in train]
    dataset = data.MoleculeDataset(points, featurizers.SimpleMoleculeMolGraphFeaturizer())
    scaler = dataset.normalize_targets()
    loader = data.build_dataloader(dataset, batch_size=64, num_workers=0, shuffle=True, seed=42)
    mp = nn.BondMessagePassing(d_h=300, depth=3, dropout=0.0)
    ffn = nn.RegressionFFN(input_dim=mp.output_dim, hidden_dim=300, n_layers=1,
                         output_transform=nn.UnscaleTransform.from_standard_scaler(scaler))
    model = models.MPNN(mp, nn.MeanAggregation(), ffn, batch_norm=False)
    out = Path(out); out.mkdir(parents=True, exist_ok=False)
    trainer = pl.Trainer(logger=False, enable_checkpointing=False, enable_progress_bar=False,
        accelerator='cpu', devices=1, max_epochs=30, limit_val_batches=0, deterministic=True,
        default_root_dir=str(out))
    trainer.fit(model, train_dataloaders=loader)
    trainer.save_checkpoint(out / 'model.ckpt')
    card = dict(card)
    card.update(estimator='chemprop', model_sha256=sha(out / 'model.ckpt'), model_file='model.ckpt',
                chemprop_version=chemprop.__version__, external_card_sha256=sha(external_card),
                features='Chemprop atom/bond graph features; no 3D conformer',
                hyperparameters={'epochs':30,'batch_size':64,'hidden_dim':300,'depth':3,'dropout':0.,
                                 'batch_norm':False, 'seed':42},
                target_scaling={'mean':scaler.mean_.tolist(),'scale':scaler.scale_.tolist()},
                tuning='none; fixed 30 epochs; no target labels, no test/validation-based checkpoint selection')
    freeze(out / 'manifest.json', card)
    return {k:card[k] for k in ('status','estimator','external_train_n','overlap_after_filter','chemprop_version')}


def predict(item, model_dir):
    import torch
    from chemprop import data, featurizers, models
    from lightning import pytorch as pl
    torch.set_num_threads(2)
    root = Path(model_dir); card = load(root / 'manifest.json')
    if item['modality'] != 'ligand' or item['endpoint']['name'] != card['endpoint']:
        raise ValueError('Chemprop endpoint mismatch')
    if card['status'] != 'complete' or sha(root / 'model.ckpt') != card['model_sha256']:
        raise ValueError('Model checkpoint changed')
    keys = {identity(c['sequence']) for c in item['candidates']}
    if keys & set(card['training_identities']) or not keys <= set(card['audited_target_identities']):
        raise ValueError('Unverified cohort or external training overlap')
    # Chemprop 2.3.1 explicitly supplies weights_only=False to Lightning internally.
    model = models.MPNN.load_from_checkpoint(root / 'model.ckpt', map_location='cpu')
    points = [data.MoleculeDatapoint.from_smi(s) for s in [item['base']] + [c['sequence'] for c in item['candidates']]]
    dataset = data.MoleculeDataset(points, featurizers.SimpleMoleculeMolGraphFeaturizer())
    loader = data.build_dataloader(dataset, batch_size=64, num_workers=0, shuffle=False)
    trainer = pl.Trainer(logger=False, enable_checkpointing=False, enable_progress_bar=False, accelerator='cpu', devices=1)
    with torch.inference_mode():
        values = torch.cat(trainer.predict(model, loader)).flatten().cpu().numpy() * item['endpoint']['label_sign']
    return values[1:].tolist(), float(values[0]), {'model_manifest_sha256':sha(root/'manifest.json'),
        'external_train_n':card['external_train_n'], 'overlap_after_filter':0, 'supervision':'external ESOL only',
        'labels_read':False, 'chemprop_version':card['chemprop_version']}
