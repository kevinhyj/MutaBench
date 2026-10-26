"""Pinned upstream VenusREM/ProSST scoring on the validated human AMFR cohort."""
import importlib.util
import os
from pathlib import Path
from .common import ROOT, load, sha, mutations


def predict(item, method, device):
    if item['assay'] != 'AMFR_HUMAN_Tsuboyama_2023_4G3O' or item['modality'] != 'protein':
        raise ValueError('Only AMFR reference structure tokens have been validated; prepare other structures explicitly')
    from .models import torch_setup
    torch = torch_setup(device)
    assets=ROOT/'assets/prosst2048'
    manifest=load(assets/'assets_manifest.json')
    for name,checksum in manifest['files'].items():
        if sha(assets/name)!=checksum:
            raise ValueError('ProSST asset hash mismatch: '+name)
    structure=assets/(item['assay']+'.structure.fasta')
    reference=assets/(item['assay']+'.reference.fasta')
    read_fasta=lambda p: ''.join(x.strip() for x in p.read_text().splitlines() if not x.startswith('>'))
    if read_fasta(reference)!=item['base']:
        raise ValueError('Upstream structure reference sequence differs from cohort WT')
    ss=[int(x) for x in read_fasta(structure).split(',')]
    if len(ss)!=len(item['base']) or any(x<0 or x>=2048 for x in ss):
        raise ValueError('Structure token count or alphabet mismatch')
    alpha=0.8 if method=='venusrem:native' else 0.0
    alignment=None
    if alpha:
        evidence=item['evidence']['msa']
        alignment=Path(evidence['path'])
        if sha(alignment)!=evidence['sha256']:
            raise ValueError('MSA hash mismatch')
        # Upstream VenusREM uses unweighted columns, uppercases lower-case positions,
        # and retains pad/gap tokens. Do not substitute our separate PSSM protocol.
        sequences=[];names=[]
        for line in alignment.read_text().splitlines():
            if line.startswith('>'):
                names.append(line[1:]);sequences.append('')
            else:sequences[-1]+=line.strip()
        if any(len(s)!=len(item['base']) for s in sequences) or sequences[0].upper()!=item['base']:
            raise ValueError('Native VenusREM currently requires full-length aligned AMFR query coordinates')
        if len(names)!=len(set(names)):
            raise ValueError('Duplicate FASTA headers would be overwritten by upstream parser')
    os.environ['HF_HUB_OFFLINE']='1'
    os.environ['TRANSFORMERS_OFFLINE']='1'
    os.environ['HF_MODULES_CACHE']=str(ROOT/'assets/hf_modules')
    from transformers import AutoModelForMaskedLM, AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(str(assets),local_files_only=True,trust_remote_code=True)
    model=AutoModelForMaskedLM.from_pretrained(str(assets),local_files_only=True,trust_remote_code=True).to(device).eval()
    source=ROOT/'vendor/VenusREM/compute_fitness.py'
    spec=importlib.util.spec_from_file_location('baseline_pinned_venusrem',source)
    upstream=importlib.util.module_from_spec(spec);spec.loader.exec_module(upstream)
    upstream.device=device
    if alignment:
        # Upstream forgets to normalize its final record. Apply the documented
        # uppercase / pad transformation consistently, without changing row counts.
        aligned={'> '+name:s.upper().replace('-','<pad>').replace('.','<pad>')
                 for name,s in zip(names,sequences)}
        upstream.read_multi_fasta=lambda path: aligned
    import pandas as pd
    mutant_names=[]
    for row in item['candidates']:
        edits=mutations(item['base'],row['sequence'])
        mutant_names.append(':'.join(f'{a}{i+1}{b}' for i,a,b in edits) or f'{item["base"][0]}1{item["base"][0]}')
    # Only substitution strings enter the upstream scorer. No DMS values are provided.
    table=pd.DataFrame({'mutant':mutant_names})
    with torch.inference_mode():
        scores=upstream.score_protein(model,tokenizer,str(reference),str(structure),table,
            alpha=alpha,aa_seq_aln_file=str(alignment) if alignment else None,
            struc_seq_aln_file=None,sample_ratio=1.0,sample_times=1)
    return scores,0.0,{'model':'ProSST-2048','revision':manifest['revision'],
        'asset_manifest_sha256':sha(assets/'assets_manifest.json'), 'upstream_scorer_sha256':sha(source),
        'upstream_commit':'7c687ad72b8b1e16eae23e938852c8bd93012b75',
        'structure_tokens':'official VenusREM 2048-token WT structure sequence; verified WT and length',
        'msa_sha256':sha(alignment) if alignment else None,'msa_rows':len(sequences) if alignment else 0,
        'alpha':alpha,'sample_ratio':1.0,'sample_times':1,
        'upstream_parser_fix':'normalize the final FASTA record exactly like all other rows' if alignment else None,
        'protocol':'upstream unmasked WT log-probability ratios; native MSA mixing when alpha=0.8',
        'labels_read':False,'precision':'FP32','scope':'AMFR only; no inferred structure for other assays'}
