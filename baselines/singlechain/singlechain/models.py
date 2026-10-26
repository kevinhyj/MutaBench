"""Predictors consume only the public input. No predictor opens test labels."""
from __future__ import annotations
import importlib
import importlib.metadata
import os
from pathlib import Path
import sys
import types
import numpy as np
from .common import ALPHABETS, ROOT, PROJECT, digest, load, mutations, sha

ESM_WEIGHTS = Path(os.environ.get('EVAGENT_ESM2_WEIGHTS', str(ROOT / 'assets/esm2/esm2_t33_650M_UR50D.pt')))
RNA_WEIGHTS = PROJECT / 'experiments/rnagym31_n250_rnalm_zeroshot_20260923/weights/RNA-FM_pretrained.pth'
RNA_CODE = Path(os.environ.get('EVAGENT_RNAFM_CODE', str(ROOT / 'vendor/rnafm_h200/fm')))
# Evo2 GPU inference is currently hosted on america; this is only a local override,
# never an automatic download or implicit change to a remote scoring protocol.
EVO_WEIGHTS = Path(os.environ.get('EVAGENT_EVO2_WEIGHTS', str(ROOT / 'assets/evo2/evo2_7b.pt')))


def torch_setup(device):
    import torch
    torch.set_num_threads(2)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if device != 'cpu' and not torch.cuda.is_available():
        raise RuntimeError('Selected environment has no CUDA; do not silently fall back to CPU')
    return torch


def load_mlm(modality, device):
    torch = torch_setup(device)
    if modality == 'protein':
        import esm
        # Local checkpoint only; bypass hub/contact-weight fetching. The trusted asset hash is recorded.
        data = torch.load(ESM_WEIGHTS, map_location='cpu', weights_only=False)
        model, alphabet = esm.pretrained.load_model_and_alphabet_core(
            'esm2_t33_650M_UR50D', data, regression_data=None)
        weights = ESM_WEIGHTS
        code = {'fair-esm': importlib.metadata.version('fair-esm')}
    else:
        for name, path in [('model', RNA_CODE.parents[1]), ('model.RNAFM', RNA_CODE.parent),
                           ('model.RNAFM.fm', RNA_CODE)]:
            package = types.ModuleType(name); package.__path__ = [str(path)]
            sys.modules[name] = package
        package = sys.modules['model.RNAFM.fm']
        package.Alphabet = importlib.import_module('model.RNAFM.fm.data').Alphabet
        package.RNABertModel = importlib.import_module('model.RNAFM.fm.model').RNABertModel
        pt = importlib.import_module('model.RNAFM.fm.pretrained')
        from argparse import Namespace
        with torch.serialization.safe_globals([Namespace]):
            data = torch.load(RNA_WEIGHTS, map_location='cpu', weights_only=True)
        model, alphabet = pt.load_model_and_alphabet_core(data, theme='rna')
        weights = RNA_WEIGHTS
        code = {str(p): sha(p) for p in sorted(RNA_CODE.rglob('*.py'))}
    model.eval().to(device)
    return model, alphabet, {'checkpoint': str(weights), 'checkpoint_sha256': sha(weights),
                              'code': code, 'torch': torch.__version__, 'precision': 'FP32, TF32 disabled'}


def mlm(item, method, device, batch_size):
    import torch
    mode = method.split(':')[1]
    if mode not in ('wt-marginals', 'masked-marginals', 'joint-masked'):
        raise ValueError('Unknown MLM scoring protocol')
    wt = item['base']
    if len(wt) + 2 > 1024:
        raise ValueError('Sequence exceeds frozen 1024-token limit; no silent truncation')
    model, alphabet, provenance = load_mlm(item['modality'], device)
    _, _, wt_tokens = alphabet.get_batch_converter()([('WT', wt)])
    if wt_tokens.shape[1] != len(wt) + 2:
        raise ValueError('Unexpected tokenizer offsets')
    changes = [mutations(wt, c['sequence']) for c in item['candidates']]
    scores = np.zeros(item['n'], dtype=float)
    with torch.inference_mode():
        if mode == 'wt-marginals':
            logp = model(wt_tokens.to(device))['logits'].log_softmax(-1)[0]
            for j, edits in enumerate(changes):
                scores[j] = sum(float(logp[i+1, alphabet.get_idx(b)] - logp[i+1, alphabet.get_idx(a)])
                                for i, a, b in edits)
        elif mode == 'masked-marginals':
            positions = sorted({i for edits in changes for i, _, _ in edits})
            values = {}
            for start in range(0, len(positions), batch_size):
                group = positions[start:start+batch_size]
                batch = wt_tokens.repeat(len(group), 1)
                for j, i in enumerate(group):
                    batch[j, i+1] = alphabet.mask_idx
                logp = model(batch.to(device))['logits'].log_softmax(-1)
                for j, i in enumerate(group):
                    values[i] = logp[j, i+1].cpu()
            for j, edits in enumerate(changes):
                scores[j] = sum(float(values[i][alphabet.get_idx(b)] - values[i][alphabet.get_idx(a)])
                                for i, a, b in edits)
        else:
            for start in range(0, len(changes), batch_size):
                group = changes[start:start+batch_size]
                batch = wt_tokens.repeat(len(group), 1)
                for j, edits in enumerate(group):
                    for i, _, _ in edits:
                        batch[j, i+1] = alphabet.mask_idx
                logp = model(batch.to(device))['logits'].log_softmax(-1)
                for j, edits in enumerate(group):
                    scores[start+j] = sum(float(logp[j, i+1, alphabet.get_idx(b)] - logp[j, i+1, alphabet.get_idx(a)])
                                          for i, a, b in edits)
    provenance.update(protocol=mode, batch_size=batch_size, reference_zero='log-probability ratio gauge, not measured WT fitness')
    return list(scores), 0.0, provenance


def read_alignment(item):
    evidence = item['evidence'].get('msa')
    if not evidence:
        raise ValueError('No MSA supplied; no zero imputation')
    path = Path(evidence['path'])
    if sha(path) != evidence['sha256']:
        raise ValueError('MSA hash mismatch')
    names, sequences = [], []
    for line in path.read_text().splitlines():
        if line.startswith('>'):
            names.append(line[1:]); sequences.append('')
        elif line.strip():
            if not sequences:
                raise ValueError('Not FASTA')
            sequences[-1] += line.strip()
    raw_query = sequences[0] if sequences else ''
    # Query lowercase insertions count in WT coordinates but are NOT homolog-supported columns.
    start = evidence.get('start', 1) - 1
    full_query = ''.join(c.upper() for c in raw_query if c.isalpha())
    if item['modality'] == 'rna':
        full_query = full_query.replace('T', 'U')
    if item['base'][start:start+len(full_query)] != full_query:
        raise ValueError('MSA query does not match declared WT coordinates')
    positions, position = [], start
    for letter in raw_query:
        if letter.isalpha():
            if letter.isupper():
                positions.append(position)
            position += 1
    # A2M/A3M inserts are lower-case or dots; drop them consistently.
    sequences = [''.join(c for c in s if c != '.' and not c.islower()) for s in sequences]
    if item['modality'] == 'rna':
        sequences = [s.replace('T', 'U') for s in sequences]
    if not sequences or len({len(s) for s in sequences}) != 1:
        raise ValueError('Inconsistent alignment lengths')
    query = sequences[0]
    ungapped = query.replace('-', '')
    if ''.join(item['base'][i] for i in positions) != ungapped:
        raise ValueError('MSA aligned-column coordinate mismatch')
    columns = [i for i, aa in enumerate(query) if aa != '-']
    homologs = sorted(set(''.join(s[i] for i in columns) for s in sequences[1:]))
    homologs = [s for s in homologs if s != ungapped]
    aa = ALPHABETS[item['modality']]
    invalid_n = sum(bool(set(s) - set(aa + '-')) for s in homologs)
    homologs = [s for s in homologs if not set(s) - set(aa + '-')][:2048]
    evidence = dict(evidence, ambiguous_homologs_excluded=invalid_n)
    if len(homologs) < 2:
        raise ValueError('Insufficient unique non-WT homologs')
    keep = [i for i in range(len(positions)) if any(s[i] != '-' for s in homologs)]
    evidence['all_gap_match_columns_excluded'] = len(positions)-len(keep)
    positions = [positions[i] for i in keep]
    homologs = [''.join(s[i] for i in keep) for s in homologs]
    ungapped = ''.join(ungapped[i] for i in keep)
    if not positions:
        raise ValueError('No homolog-supported alignment columns')
    return homologs, positions, aa, path, evidence, ungapped


def alignment(item):
    homologs, positions, aa, path, evidence, ungapped = read_alignment(item)
    ar = np.array([[aa.find(x) for x in s] for s in homologs], dtype=np.int8)
    neighbors = np.zeros(len(ar))
    for offset in range(0, len(ar), 16):
        a, b = ar[offset:offset+16, None, :], ar[None, :, :]
        valid = (a >= 0) & (b >= 0)
        overlap = valid.sum(2)
        matches = ((a == b) & valid).sum(2)
        neighbors[offset:offset+16] = ((overlap >= .5 * ar.shape[1]) & (matches >= .8 * overlap)).sum(1)
    weights = 1 / np.maximum(neighbors, 1)
    neff = float(weights.sum())
    if neff < evidence.get('min_neff', 20):
        raise ValueError(f'MSA Neff={neff:.2f} below preregistered threshold')
    mapped = set(positions)
    for candidate in item['candidates']:
        if any(i not in mapped for i, _, _ in mutations(item['base'], candidate['sequence'])):
            raise ValueError('Mutation outside MSA coordinates; choose a common eligible cohort before sampling')
    return ar, weights, positions, aa, {
        'path': str(path), 'sha256': sha(path), 'positions_1based': [i+1 for i in positions],
        'homologs': len(homologs), 'neff': neff, 'homolog_sequences': homologs,
        'ambiguous_homologs_excluded': evidence['ambiguous_homologs_excluded'],
        'all_gap_match_columns_excluded': evidence['all_gap_match_columns_excluded'],
        'query': ungapped, 'weighting': 'inverse 80% identity neighbors, >=50% overlap',
        'selection': 'sorted unique non-WT homologs, first 2048'}


def msa_score(item):
    ar, weights, positions, aa, provenance = alignment(item)
    index = {position: i for i, position in enumerate(positions)}
    probabilities = np.stack([((ar == k) * weights[:, None]).sum(0) for k in range(len(aa))], axis=1)
    probabilities = (probabilities + 1 / len(aa)) / (probabilities.sum(1, keepdims=True) + 1)
    values = []
    for candidate in item['candidates']:
        values.append(sum(float(np.log(probabilities[index[i], aa.index(b)] / probabilities[index[i], aa.index(a)]))
                          for i, a, b in mutations(item['base'], candidate['sequence'])))
    provenance.pop('homolog_sequences'); provenance.pop('query')
    provenance['pseudocount_total'] = 1
    return values, 0.0, provenance


def vienna(item):
    import RNA
    md = RNA.md(); md.temperature = 37.0
    md.dangles = 2; md.noLP = 0
    reference = item['evidence'].get('reference_structure')
    if reference is None:
        reference, _ = RNA.fold_compound(item['base'], md).mfe()
        reference_source = 'WT MFE at 37 C, frozen independently of candidate labels'
    else:
        reference_source = 'prespecified dot-bracket input'
    if len(reference) != len(item['base']) or set(reference) - set('.()'):
        raise ValueError('Invalid reference dot-bracket structure')
    # ensemble_defect uses base-pair probabilities and returns a length-normalized defect.
    values = []
    for seq in [item['base']] + [c['sequence'] for c in item['candidates']]:
        fc = RNA.fold_compound(seq, md)
        fc.pf()
        values.append(-float(fc.ensemble_defect(reference)))
    return values[1:], values[0], {'viennarna_version': RNA.__version__, 'temperature_C': 37,
        'dangles': 2, 'reference_structure': reference, 'reference_source': reference_source,
        'score': 'negative normalized ensemble defect to fixed reference',
        'meaning': 'structure preservation proxy; not universal function or maximum stability'}


def crippen(item):
    from rdkit import Chem, rdBase
    from rdkit.Chem import Crippen
    if item['endpoint']['name'] not in ('lipophilicity_astrazeneca', 'logP'):
        raise ValueError('Crippen logP is a lipophilicity proxy, not a universal ADMET predictor')
    sign = item['endpoint']['label_sign']
    values = [sign * Crippen.MolLogP(Chem.MolFromSmiles(s)) for s in [item['base']] + [c['sequence'] for c in item['candidates']]]
    return values[1:], values[0], {'rdkit': rdBase.rdkitVersion, 'score': 'Crippen logP',
        'cross_endpoint_proxy': 'Crippen logP versus experimental lipophilicity/logD; distinguish endpoints'}


def admet(item):
    from admet_ai import ADMETModel
    import admet_ai
    import torch
    torch_setup('cpu')
    if torch.cuda.is_available():
        raise RuntimeError('ADMET CPU runner must hide GPUs before torch import')
    model = ADMETModel(include_physchem=False, drugbank_path=None, num_workers=0)
    sequences = [item['base']] + [c['sequence'] for c in item['candidates']]
    result = model.predict(smiles=sequences)
    # Match endpoint by exact case-insensitive name, never a vaguely similar endpoint.
    keys = [k for k in result.columns if k.lower() == item['endpoint']['name'].lower()]
    if len(keys) != 1 or len(result) != len(sequences) or list(result.index) != sequences:
        raise ValueError('ADMET endpoint missing, invalid molecule dropped, or input order changed')
    values = result[keys[0]].to_numpy(dtype=float) * item['endpoint']['label_sign']
    from admet_ai.constants import DEFAULT_MODELS_DIR
    weights = {str(p.relative_to(DEFAULT_MODELS_DIR)): sha(p) for p in sorted(Path(DEFAULT_MODELS_DIR).rglob('*.pt'))}
    return list(values[1:]), float(values[0]), {'admet_ai': admet_ai.__version__, 'model_weights': weights,
        'endpoint_column': keys[0], 'drugbank_reference': False, 'external_training': 'TDC ADMET datasets',
        'training_test_overlap': 'unknown; not admitted as independent zero-shot evidence'}


def evo2(item, device, batch_size):
    if device != 'cuda:0':
        raise ValueError('Evo2 BF16 requires an explicitly allocated GPU; no implicit CPU fallback')
    torch = torch_setup(device)
    free, _ = torch.cuda.mem_get_info()
    if free < 24 * 1024**3:
        raise RuntimeError('Evo2 preflight requires >=24 GiB free on the authorized GPU')
    from evo2 import Evo2
    model = Evo2('evo2_7b', local_path=str(EVO_WEIGHTS))
    model.model.eval()
    sequences = [s.replace('U', 'T') for s in [item['base']] + [c['sequence'] for c in item['candidates']]]
    with torch.inference_mode():
        scores = model.score_sequences(sequences, batch_size=batch_size, prepend_bos=False,
                                        reduce_method='mean', average_reverse_complement=True)
    return [float(s) for s in scores[1:]], float(scores[0]), {
        'checkpoint': str(EVO_WEIGHTS), 'checkpoint_sha256': sha(EVO_WEIGHTS),
        'precision': 'local BF16; not official FP8 reproduction', 'prepend_bos': False,
        'reduce_method': 'mean', 'average_reverse_complement': True, 'rna_u_to_t': True}


METHODS = {
    'random': ('any', 'simple_control', True),
    'esm2:masked-marginals': ('protein', 'pretrained_sequence', True),
    'esm2:wt-marginals': ('protein', 'pretrained_sequence', True),
    'rnafm:joint-masked': ('rna', 'pretrained_sequence', True),
    'rnafm:wt-marginals': ('rna', 'pretrained_sequence', True),
    'rnafm:masked-marginals': ('rna', 'pretrained_sequence', True),
    'evo2:bf16': ('rna', 'pretrained_sequence', True),
    'venusrem:native': ('protein', 'pretrained_structure_msa', True),
    'prosst:wt-marginals': ('protein', 'pretrained_structure', True),
    'aido:rag-zeroshot': ('protein', 'pretrained_structure_msa', True),
    'poet2:msa-wtstructure': ('protein', 'pretrained_structure_msa', True),
    'msa:site-independent': ('sequence', 'msa_statistical', True),
    'vienna:ensemble-defect': ('rna', 'physical_proxy', True),
    'crippen:logp': ('ligand', 'property_proxy', True),
    'admet-ai': ('ligand', 'external_supervised_overlap_unresolved', False),
}


def predict(item, method, device='cpu', batch_size=4):
    if batch_size < 1:
        raise ValueError('Positive batch size required')
    modality, group, independent = METHODS[method]
    if modality == 'sequence' and item['modality'] not in ALPHABETS:
        raise ValueError('MSA method requires protein/RNA')
    if modality not in ('any', 'sequence') and modality != item['modality']:
        raise ValueError('Method/modality mismatch')
    if method.startswith(('esm2:', 'rnafm:')):
        result = mlm(item, method, device, batch_size)
    elif method.startswith('msa:'):
        result = msa_score(item)
    elif method.startswith('vienna:'):
        result = vienna(item)
    elif method.startswith('crippen:'):
        result = crippen(item)
    elif method == 'admet-ai':
        result = admet(item)
    elif method.startswith('evo2:'):
        result = evo2(item, device, batch_size)
    elif method in ('venusrem:native', 'prosst:wt-marginals'):
        from .venusrem import predict as predict_structure
        result = predict_structure(item, method, device)
    elif method == 'aido:rag-zeroshot':
        from .aido import predict as predict_aido
        result = predict_aido(item, device)
    elif method == 'poet2:msa-wtstructure':
        from .poet2 import predict as predict_poet2
        result = predict_poet2(item, device, batch_size)
    elif method == 'random':
        import random
        # ID-stable, order-invariant, no test-label use.
        values = [random.Random(digest([42, item['assay'], c['id']])).uniform(-1, 1) for c in item['candidates']]
        result = values, 0.0, {'seed': 42, 'distribution': 'uniform(-1,1); SHA256 of seed, assay, ID'}
    else:
        raise ValueError('Unimplemented method')
    return (*result, group, independent)
