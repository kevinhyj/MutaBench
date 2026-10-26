"""plmc EVmutation/Potts using unlabeled homologs; no experimental-label training."""
from pathlib import Path
import subprocess
import numpy as np
from .common import ROOT, freeze, load, sha
from .models import alignment


def read_params(path):
    # plmc_v2 binary layout documented in upstream scripts/read_params.m and src/plm.c.
    with Path(path).open('rb') as f:
        header = np.fromfile(f, dtype='<i4', count=5)
        if len(header) != 5:
            raise ValueError('Truncated plmc header')
        length, q, n, invalid, iterations = map(int, header)
        if not 1 <= length <= 2048 or not 2 <= q <= 32 or n < 1 or invalid < 0:
            raise ValueError('Invalid plmc dimensions')
        scalars = np.fromfile(f, dtype='<f4', count=5)
        alphabet = f.read(q).decode('ascii')
        np.fromfile(f, dtype='<f4', count=n+invalid)
        target = f.read(length).decode('ascii')
        offsets = np.fromfile(f, dtype='<i4', count=length)
        np.fromfile(f, dtype='<f4', count=length*q)
        fields = np.fromfile(f, dtype='<f4', count=length*q).reshape(length, q)
        pairs = [(i, j) for i in range(length) for j in range(i+1, length)]
        count = len(pairs)*q*q
        f.seek(count*4, 1)
        couplings = np.fromfile(f, dtype='<f4', count=count).reshape(len(pairs), q, q)
        if f.read(1) or not np.isfinite(fields).all() or not np.isfinite(couplings).all():
            raise ValueError('Unexpected or nonfinite plmc parameters')
    return alphabet, target, offsets, fields, pairs, couplings, scalars, iterations


def fit(item, out):
    ar, weights, positions, aa, evidence = alignment(item)
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    homologs = evidence.pop('homolog_sequences')
    query = evidence.pop('query')
    msa = out / 'alignment.fasta'
    msa.write_text('>WT/1-%d\n%s\n' % (len(query), query) +
                   ''.join(f'>H{i}\n{s}\n' for i, s in enumerate(homologs)))
    binary = ROOT / 'vendor/plmc/bin/plmc'
    params = out / 'model.params'
    # Conventional regularization; frozen, never selected against target labels.
    lambda_j = .2 * (len(query)-1)
    cmd = [str(binary), '-o', str(params), '-f', 'WT', '-a', '-'+aa,
           '-t', '0.2', '-lh', '0.01', '-le', str(lambda_j), '-m', '100', '-n', '2', str(msa)]
    with (out / 'fit.log').open('w') as log:
        subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=3600)
    read_params(params)
    receipt = {'status': 'complete', 'method': 'EVmutation/Potts plmc', 'target': query,
        'positions_1based': [i+1 for i in positions], 'modality': item['modality'],
        'msa': evidence, 'alignment_sha256': sha(msa), 'binary_sha256': sha(binary),
        'params_sha256': sha(params), 'command': cmd,
        'labels_read': False, 'training': 'unlabeled MSA pseudolikelihood',
        'iterations': 100, 'lambda_h': .01, 'lambda_J': lambda_j,
        'weighting': 'plmc native 80% identity neighborhood, query included; differs from site-independent filter weights',
        'protocol': 'sum h_i + J_ij; mutant minus WT, higher evolutionary plausibility'}
    freeze(out / 'manifest.json', receipt)
    return receipt


def predict(item, model_dir):
    root = Path(model_dir)
    card = load(root / 'manifest.json')
    if card['status'] != 'complete' or card['params_sha256'] != sha(root / 'model.params'):
        raise ValueError('Potts model not complete or changed')
    positions = [i-1 for i in card['positions_1based']]
    if card['modality'] != item['modality'] or ''.join(item['base'][i] for i in positions) != card['target']:
        raise ValueError('Potts WT/modality mismatch')
    alphabet, target, offsets, fields, pairs, couplings, scalars, iterations = read_params(root / 'model.params')
    if target != card['target'] or list(offsets) != list(range(1, len(target)+1)):
        raise ValueError('Potts target coordinates do not match')
    def energy(seq):
        mapped = set(positions)
        if any(a != b and i not in mapped for i, (a,b) in enumerate(zip(item['base'], seq))):
            raise ValueError('Mutation outside modeled region')
        encoded = [alphabet.index(seq[i]) for i in positions]
        return float(sum(float(fields[i, x]) for i, x in enumerate(encoded)) +
                     sum(float(couplings[k, encoded[i], encoded[j]]) for k, (i, j) in enumerate(pairs)))
    wt_score = energy(item['base'])
    scores = [energy(c['sequence'])-wt_score for c in item['candidates']]
    return scores, 0.0, {'model_manifest': str(root / 'manifest.json'),
                        'model_manifest_sha256': sha(root / 'manifest.json'),
                        'params_sha256': sha(root / 'model.params'), 'labels_read': False}
