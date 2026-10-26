"""PoET-2 native scoring: 30 MSA/WT-structure prompts for frozen AMFR.

Uses the upstream supported structure_in_context=0 setting: homolog sequence
contexts, with/without a masked-sequence WT structure query. It does not claim
the default AFDB homolog-structure protocol or its published benchmark score.
"""
from __future__ import annotations
import importlib.util
import ast
import itertools
import json
import os
from pathlib import Path
import sys
import types
import numpy as np
from .common import ROOT, digest, freeze, load, sha, validate_input

SOURCE = ROOT / 'vendor/PoET-2'
WEIGHTS = ROOT / 'assets/poet2/poet-2.ckpt'
WEIGHT_SHA = '1f5627e08ab6dd45236a655fe676df5d17d769ce7109fa10dfebb27fdf152a79'
CACHE = ROOT / 'assets/poet2/amfr_prepared_v1'
ASSAY = 'AMFR_HUMAN_Tsuboyama_2023_4G3O'


def imports():
    sys.path.insert(0, str(SOURCE / 'src'))
    sys.path.insert(0, str(SOURCE / 'scripts'))
    if os.environ.get('CUDA_VISIBLE_DEVICES') == '':
        # Triton 3.2 initializes its GPU autotuner while importing norm.py even
        # when only CPU input is intended. Load the unchanged upstream RMSNorm
        # class alone; its explicit torch CPU branch is the computation used.
        # GPU execution never uses this import path or a fake Triton driver.
        name = 'poet_2.models.modules.norm'
        if name not in sys.modules:
            import torch
            source = SOURCE / 'src/poet_2/models/modules/norm.py'
            node = next(n for n in ast.parse(source.read_text()).body
                        if isinstance(n, ast.ClassDef) and n.name == 'RMSNorm')
            module = types.ModuleType(name)
            module.__file__ = str(source)
            module.torch = torch
            def gpu_unavailable(*args, **kwargs):
                raise RuntimeError('CPU-isolated norm module cannot execute CUDA kernels')
            module.rms_norm_fn = gpu_unavailable
            exec(compile(ast.Module(body=[node], type_ignores=[]),str(source),'exec'),module.__dict__)
            sys.modules[name] = module
    import poet_2.models.poet_2_helpers as helpers
    return helpers


def prepare(input_path):
    import torch
    import numba
    helpers = imports()
    from openprotein.protein import Protein
    from poet_2.alphabet.sparse_uniref_cluster2 import Alphabet
    from utils import get_names_and_seqs_from_fastalike, get_encoded_msa_from_a3m_seqs, get_numpy_seed
    item = validate_input(load(input_path))
    if item['assay'] != ASSAY or item['modality'] != 'protein':
        raise ValueError('Deployment restricted to reviewed AMFR cohort')
    if CACHE.exists():
        raise FileExistsError('Use a fresh preparation directory')
    torch.set_num_threads(16)
    numba.set_num_threads(32)
    msa_path = Path(item['evidence']['msa']['path'])
    if sha(msa_path) != item['evidence']['msa']['sha256']:
        raise ValueError('MSA checksum mismatch')
    names, sequences = get_names_and_seqs_from_fastalike(msa_path)
    # The frozen input is A2M: dots pad insertion columns. Upstream expects
    # A3M (lowercase inserts, no dot padding). Retain all actual residues.
    dot_padding_removed = sum(s.count(b'.') for s in sequences)
    sequences = [s.replace(b'.', b'') for s in sequences]
    if sequences[0].upper().replace(b'-', b'').decode() != item['base']:
        raise ValueError('MSA query does not match WT')
    msa = get_encoded_msa_from_a3m_seqs(sequences, Alphabet())
    spec = importlib.util.spec_from_file_location('poet_upstream_score', SOURCE / 'scripts/score.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    pdb = ROOT / 'assets/aido_amfr' / f'{ASSAY}.pdb'
    query = Protein.from_filepath(pdb, chain_id='A')
    if query.sequence.decode() != item['base']:
        raise ValueError('Structure query does not match WT')
    CACHE.mkdir(parents=True)
    with (CACHE / 'query_structure.npz').open('xb') as handle:
        np.savez_compressed(handle, coordinates=query.coordinates[:, :3], plddt=query.plddt)
    # Exactly the seed transformation used by upstream parse_args(seed=42).
    seed = int(np.random.RandomState(42).randint(2**31))
    prompts = []
    for context_length, similarity, inverse_query in itertools.product(
            [6144, 12288, 24576], [1.0, .95, .90, .70, .50], [False, True]):
        name = f'{context_length}_{similarity}_False_{inverse_query}_seed={seed}'
        prompt_seed = get_numpy_seed(name)
        context = module.sample_context(names, sequences, msa, .2, context_length,
            similarity, False, inverse_query, prompt_seed, CACHE / 'sampling_weights')
        if not context:
            raise ValueError('Empty native MSA context; no silent replacement')
        prompts.append(dict(context_length=context_length, max_similarity=similarity,
            inverse_folding_query=inverse_query, structure_in_context=False,
            prompt_seed=prompt_seed, sequences=[p.sequence.decode() for p in context]))
        print(f'PoET-2 prepared prompt {len(prompts)}/30 ({len(context)} homologs)', flush=True)
    freeze(CACHE / 'prompts.json', prompts)
    freeze(CACHE / 'receipt.json', dict(status='cpu_prompts_prepared', assay=ASSAY,
        input_digest=digest(item), input_sha256=sha(input_path), labels_read=False,
        data_seed=42, upstream_seed_argument=42, upstream_transformed_seed=seed,
        msa_rows=len(sequences), prompt_count=len(prompts), cpu_numba_threads=32,
        a2m_to_a3m_dot_padding_removed=dot_padding_removed,
        checkpoint_sha256=sha(WEIGHTS), upstream_commit='015d8f8b1d97dd1a0ebbb8f9324bda6d750a2469',
        protocol='Native 3 context lengths x 5 similarity ceilings x 2 WT-structure-query settings; homolog structures disabled',
        source_sha256={str(p): sha(p) for p in [msa_path, pdb, SOURCE / 'scripts/score.py',
            SOURCE / 'scripts/utils.py', CACHE / 'prompts.json', CACHE / 'query_structure.npz']}))
    return load(CACHE / 'receipt.json')


def load_model(device):
    import torch
    helpers = imports()
    if sha(WEIGHTS) != WEIGHT_SHA:
        raise ValueError('PoET-2 checkpoint checksum mismatch')
    if device != 'cpu':
        return helpers.load_model(WEIGHTS, torch.device(device)).eval()
    # Same constructor/state as upstream load_model, with its CPU FP32 choice.
    # Disable only the fused SwiGLU accelerator before upstream warmup.
    from poet_2.models.poet_2 import PoET2
    checkpoint = torch.load(WEIGHTS, map_location='cpu', weights_only=False)
    hparams = checkpoint['hyper_parameters']
    if 'init_args' in hparams:
        hparams = hparams['init_args']
    model = PoET2(**hparams['model_spec']['init_args'])
    model.load_state_dict({k.removeprefix('model.'): v for k, v in checkpoint['state_dict'].items()})
    model.mask_token_s3di = helpers.S3DiAlphabet().mask_token
    model.float().eval()
    for submodule in model.modules():
        if hasattr(submodule, 'use_fused_swiglu'):
            submodule.use_fused_swiglu = False
    helpers.jit_warmup(model)
    return model


def smoke():
    import torch
    torch.set_num_threads(16)
    torch.manual_seed(42)
    helpers = imports()
    model = load_model('cpu')
    with torch.inference_mode():
        memory, _ = helpers.compute_memory(model, [['ACDEFGHIKLMNPQRSTVWY']])
        scores = helpers.score_sequences(model, memory, ['ACDEFGHIK', 'ACDEYGHIK'], batch_size=2)
    if scores.shape != (2,) or not torch.isfinite(scores).all():
        raise ValueError('PoET-2 smoke failed')
    return dict(status='cpu_synthetic_forward_passed', checkpoint_sha256=WEIGHT_SHA,
        sequences='two synthetic sequences; not assay evaluation', n=2,
        scores=scores.tolist(), torch=torch.__version__, precision='FP32', cpu_threads=16,
        fused_swiglu=False, norm='unchanged native PyTorch CPU branch; GPU autotuner imports isolated',
        cuda_initialized=torch.cuda.is_initialized())


def predict(item, device, batch_size):
    import torch
    if device not in ('cpu', 'cuda:0'):
        raise ValueError('Use CPU or the explicitly allocated cuda:0 device')
    receipt = load(CACHE / 'receipt.json')
    if item['assay'] != ASSAY or digest(item) != receipt['input_digest']:
        raise ValueError('PoET-2 prepared cohort mismatch')
    for path, expected in receipt['source_sha256'].items():
        if sha(path) != expected:
            raise ValueError(f'PoET-2 input/source checksum mismatch: {path}')
    torch.set_num_threads(32 if device == 'cpu' else 16)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    helpers = imports()
    model = load_model(device)
    variants = [c['sequence'] for c in item['candidates']] + [item['base']]
    structure = np.load(CACHE / 'query_structure.npz', allow_pickle=False)
    prompts = load(CACHE / 'prompts.json')
    progress = Path(os.environ['POET2_PROGRESS_DIR']) if os.environ.get('POET2_PROGRESS_DIR') else None
    if progress:
        freeze(progress / 'identity.json', dict(input_digest=digest(item), device=device,
            batch_size=batch_size, preparation_sha256=sha(CACHE / 'receipt.json'),
            checkpoint_sha256=WEIGHT_SHA, adapter_sha256=sha(Path(__file__))))
    all_scores = []
    with torch.inference_mode():
        for i, prompt in enumerate(prompts):
            progress_file = progress / f'prompt_{i:02d}.json' if progress else None
            if progress_file and progress_file.exists():
                previous = load(progress_file)
                if previous['prompt_digest'] != digest(prompt):
                    raise ValueError('Resume prompt mismatch')
                scores = np.asarray(previous['scores'], dtype=np.float32)
                if scores.shape != (item['n']+1,) or not np.isfinite(scores).all():
                    raise ValueError('Invalid resumed scores')
                all_scores.append(scores)
                continue
            query = 'X' * len(item['base']) if prompt['inverse_folding_query'] else None
            context = [helpers.NamedInput(sequence=s).enforce() for s in prompt['sequences']]
            if query is not None:
                context.insert(0, helpers.NamedInput(sequence=query,
                    plddt=structure['plddt'].copy(), atomx=structure['coordinates'].copy()).enforce())
            memory, ys_ref = helpers.compute_memory(model, [context])
            scores = helpers.score_sequences(model, memory, variants,
                ys_ref_values=ys_ref if query is not None else None, self_prompt=query,
                batch_size=batch_size, verbose=True).cpu().float().numpy()
            if not np.isfinite(scores).all():
                raise ValueError('Nonfinite PoET-2 scores')
            all_scores.append(scores)
            if progress_file:
                freeze(progress_file, dict(prompt_digest=digest(prompt), scores=scores.tolist(), labels_read=False))
            print(f'PoET-2 scored prompt {i + 1}/30', flush=True)
    mean = np.asarray(all_scores).mean(axis=0)
    return (mean[:-1] - mean[-1]).tolist(), 0.0, dict(
        checkpoint_sha256=WEIGHT_SHA, preparation_receipt_sha256=sha(CACHE / 'receipt.json'),
        protocol=receipt['protocol'], precision='FP32 CPU' if device == 'cpu' else 'FP16 GPU', batch_size=batch_size,
        prompt_count=len(prompts), prompt_wt_log_likelihoods=[float(s[-1]) for s in all_scores],
        reference_zero='Mean ensemble log-likelihood difference to WT',
        torch=torch.__version__, cpu_threads=torch.get_num_threads(),
        max_gpu_memory_allocated=torch.cuda.max_memory_allocated() if device != 'cpu' else 0)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--prepare')
    group.add_argument('--cpu-smoke', action='store_true')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    result = prepare(args.prepare) if args.prepare else smoke()
    freeze(args.out, result)
    print(json.dumps(result, indent=2))
