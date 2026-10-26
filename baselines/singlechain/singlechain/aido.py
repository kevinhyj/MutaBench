"""Pinned AIDO RAG zero-shot adapter for the frozen benign AMFR cohort.

The official MSA selection/tokenization functions are executed unchanged.
Only the structure encoder is isolated from unused folding/decoder imports.
No evaluation labels are read by preparation or inference.
"""
from __future__ import annotations
import ast
import gzip
import importlib
import json
import random
import sys
import types
from pathlib import Path
from typing import List, Tuple
import numpy as np
from .common import ROOT, digest, freeze, load, mutations, sha, validate_input

ASSAY = 'AMFR_HUMAN_Tsuboyama_2023_4G3O'
SOURCE = ROOT / 'vendor/AIDO-RAG-zeroshot'
ASSETS = ROOT / 'assets/aido_amfr'
WEIGHTS = ROOT / 'assets/aido_rag16b_zeroshot'
CACHE = ASSETS / 'prepared_v2'


def upstream_functions():
    """Load the reviewed pure functions without the unused ESMFold imports."""
    from scipy.spatial.distance import cdist
    path = SOURCE / 'utils/misc.py'
    tree = ast.parse(path.read_text())
    wanted = {'load_msa_txt', 'greedy_select', 'tokenize'}
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in wanted]
    if len(nodes) != len(wanted):
        raise ValueError('Pinned upstream function set changed')
    namespace = dict(np=np, gzip=gzip, random=random, cdist=cdist, List=List, Tuple=Tuple)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace


def prepare(input_path):
    import torch
    from transformers import AutoTokenizer
    item = validate_input(load(input_path))
    if item['assay'] != ASSAY or item['modality'] != 'protein':
        raise ValueError('This deployment is restricted to the reviewed AMFR cohort')
    if CACHE.exists():
        raise FileExistsError('Frozen AIDO preparation already exists')
    torch.set_num_threads(16)
    torch.manual_seed(42)
    package = types.ModuleType('utils')
    package.__path__ = [str(SOURCE / 'utils')]
    sys.modules['utils'] = package
    protein = importlib.import_module('utils.protein')
    pdb = ASSETS / f'{ASSAY}.pdb'
    prot = protein.from_pdb_string(pdb.read_text(), molecular_type='protein')
    fn = upstream_functions()
    msa_path = ASSETS / f'{ASSAY}.txt.gz'
    msa = fn['load_msa_txt'](str(msa_path))
    if not (prot.seq(True) == msa[0] == item['base']):
        raise ValueError('PDB, MSA, frozen WT mismatch')
    if len(item['base']) > 768 or not np.all(prot.atom_mask[:, :3]):
        raise ValueError('Expected one complete backbone window')
    source_root = ROOT / 'vendor/modelgenerator_structure/modelgenerator'
    for name, path in [('modelgenerator', source_root),
                       ('modelgenerator.structure_tokenizer', source_root / 'structure_tokenizer'),
                       ('modelgenerator.structure_tokenizer.models', source_root / 'structure_tokenizer/models')]:
        package = types.ModuleType(name)
        package.__path__ = [str(path)]
        sys.modules[name] = package
    encoder_class = importlib.import_module(
        'modelgenerator.structure_tokenizer.models.equiformer_encoder').EquiformerEncoderLightning
    encoder_path = ROOT / 'assets/aido_structure_encoder'
    encoder = encoder_class(str(encoder_path)).eval()
    positions = prot.atom_positions.copy()
    positions[prot.atom_mask == 0] = np.nan
    length = len(item['base'])
    # Exactly the tensors produced by the upstream collate_fn for one full chain.
    with torch.inference_mode():
        tokens = encoder(torch.tensor(positions[None], dtype=torch.float32),
                         torch.ones((1, length), dtype=torch.bool),
                         torch.arange(1, length + 1)[None])['idx'][0].cpu()
    codebook_path = SOURCE / 'utils/codebook.pt'
    codebook = torch.load(codebook_path, map_location='cpu', weights_only=True)
    embeddings = torch.nn.functional.embedding(tokens, codebook).numpy()
    if tokens.shape != (length,) or not np.isfinite(embeddings).all():
        raise ValueError('Invalid structure encoding')
    tokenizer = AutoTokenizer.from_pretrained(str(WEIGHTS), trust_remote_code=True, local_files_only=True)
    selected = fn['greedy_select'](msa[1:], num_seqs=None, num_tokens=12800, seed=0)
    selected.sort(key=lambda x: x.count('-'))
    input_tokens, position_ids = fn['tokenize'](item['base'], selected, tokenizer, max_context=12800)
    if not np.array_equal(position_ids[:, :length], np.stack([np.arange(length), np.zeros(length)])):
        raise ValueError('Unexpected query offsets')
    positions_to_score = sorted({i for c in item['candidates'] for i, _, _ in mutations(item['base'], c['sequence'])})
    CACHE.mkdir(parents=True)
    with (CACHE / 'inputs.npz').open('xb') as handle:
        np.savez_compressed(handle, input_tokens=input_tokens, position_ids=position_ids,
                            structure_tokens=tokens.numpy(), structure_embeddings=embeddings,
                            positions_to_score=np.array(positions_to_score))
    sources = [pdb, msa_path, codebook_path, SOURCE / 'utils/misc.py', SOURCE / 'utils/protein.py',
               encoder_path / 'model.safetensors', encoder_path / 'config.json',
               WEIGHTS / 'tokenization.py', WEIGHTS / 'vocab.json']
    freeze(CACHE / 'receipt.json', dict(status='cpu_structure_and_msa_prepared', labels_read=False,
        input_sha256=sha(input_path), input_digest=digest(item), assay=ASSAY,
        cache_sha256=sha(CACHE / 'inputs.npz'), structure_length=length,
        msa_rows=len(msa), selected_msa_rows=len(selected), context_tokens=len(input_tokens),
        masked_positions=len(positions_to_score), data_seed=42, upstream_msa_seed=0,
        structure_device='cpu', structure_precision='FP32', cpu_threads=16,
        codebook_equal_to_encoder=torch.equal(codebook, encoder.encoder.codebook),
        source_sha256={str(p): sha(p) for p in sources}))
    return load(CACHE / 'receipt.json')


def scores_from_logits(item, table, positions, vocabulary):
    from scipy.special import log_softmax
    table = np.asarray(table, dtype=np.float64)
    mutant_lp, wildtype_lp = log_softmax(table, axis=-1), log_softmax(table / 1.5, axis=-1)
    row_of_position = {pos: index for index, pos in enumerate(positions)}
    return [round(sum(float(mutant_lp[row_of_position[i], vocabulary[b]] -
                            wildtype_lp[row_of_position[i], vocabulary[a]])
                      for i, a, b in mutations(item['base'], c['sequence'])), 5)
            for c in item['candidates']]


def predict(item, device):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if device != 'cuda:0':
        raise ValueError('AIDO 16B full scoring requires the explicitly allocated GPU')
    receipt = load(CACHE / 'receipt.json')
    if item['assay'] != ASSAY or digest(item) != receipt['input_digest']:
        raise ValueError('AIDO prepared input does not match the frozen cohort')
    if sha(CACHE / 'inputs.npz') != receipt['cache_sha256']:
        raise ValueError('AIDO cache checksum mismatch')
    for path, expected in receipt['source_sha256'].items():
        if sha(path) != expected:
            raise ValueError(f'AIDO source checksum mismatch: {path}')
    meta = load(WEIGHTS / 'hub_metadata.json')
    hashes = {}
    for f in meta['siblings']:
        if f['rfilename'].endswith('.bin'):
            path = WEIGHTS / f['rfilename']
            actual = sha(path)
            if actual != f['lfs']['sha256']:
                raise ValueError(f'AIDO weight checksum mismatch: {path.name}')
            hashes[path.name] = actual
    torch.set_num_threads(16)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    model = AutoModelForCausalLM.from_pretrained(str(WEIGHTS), trust_remote_code=True,
        local_files_only=True, torch_dtype=torch.bfloat16).eval().to(device)
    tokenizer = AutoTokenizer.from_pretrained(str(WEIGHTS), trust_remote_code=True, local_files_only=True)
    cache = np.load(CACHE / 'inputs.npz', allow_pickle=False)
    ids = torch.from_numpy(cache['input_tokens']).to(device)
    position_ids = torch.from_numpy(cache['position_ids']).to(device)
    embedding = torch.from_numpy(cache['structure_embeddings']).to(device, dtype=torch.bfloat16)
    positions = cache['positions_to_score'].tolist()
    table = []
    with torch.inference_mode():
        for index, pos in enumerate(positions):
            masked = ids.clone()
            masked[position_ids[0] == pos] = tokenizer.TokenToId('tMASK')
            output = model.transformer(input_ids=masked[None], position_ids=position_ids[None],
                full_attention_mask=None, inputs_str_embeds=embedding[None])
            logits = model.transformer.output_layer(output['last_hidden_state'])[:, 0]
            table.append(logits[pos].cpu().float().numpy())
            print(f'AIDO masked column {index + 1}/{len(positions)}', flush=True)
    scores = scores_from_logits(item, table, positions, tokenizer.get_vocab())
    return scores, 0.0, dict(checkpoint_repo=meta['id'], checkpoint_revision=meta['sha'],
        checkpoint_sha256=hashes, preparation_receipt_sha256=sha(CACHE / 'receipt.json'),
        protocol='Official column masking, MSA seed 0, context 12800, mutant T=1 / WT T=1.5, sum edits, round 5',
        precision='BF16 LM / FP32 CPU structure encoder', tf32=False,
        reference_zero='Empty-edit score zero; asymmetric temperatures are retained from upstream',
        torch=torch.__version__, max_gpu_memory_allocated=torch.cuda.max_memory_allocated())


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.prepare), indent=2))
