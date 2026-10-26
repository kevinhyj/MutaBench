"""Zero-shot-only lexical2048 protein-site MSA; never changes few-shot assets."""
import copy
import hashlib
import json
import math
from pathlib import Path
import numpy as np

AA = 'ACDEFGHIKLMNPQRSTVWY'
POLICY_VERSION = 'zeroshot-protein-site-lexical2048-v3'
POLICY = dict(version=POLICY_VERSION, cap=2048,
    selection='Lexicographically sorted unique valid non-WT projected homologs; first 2048',
    weighting='Inverse >=80% shared-valid identity neighbors with >=50% retained-column overlap',
    pseudocount_total=1., low_neff='Retain as quality flag; no new missingness gate',
    candidate_selection_changed=False, evaluation_labels_used=False)

def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def alignment(raw, cap=2048):
    aligned=[''.join(c for c in s if c!='.' and not c.islower()) for s in raw]
    if not aligned or len(set(map(len,aligned)))!=1: raise ValueError('Invalid alignment length')
    query=aligned[0];columns=[i for i,c in enumerate(query) if c!='-']
    observed=[];position=0
    for c in raw[0]:
        if c.isalpha():
            position+=1
            if c.isupper(): observed.append(position)
    pool=set(''.join(s[i] for i in columns) for s in aligned[1:]);pool.discard(query.replace('-',''))
    valid=sorted(s for s in pool if not set(s)-set(AA+'-'))
    selected=valid[:cap]
    if len(selected)<2: raise ValueError('Insufficient non-WT homologs')
    ar=np.asarray([[AA.find(c) for c in s] for s in selected],dtype=np.int8)
    keep=(ar>=0).any(axis=0)
    if not keep.any():raise ValueError('No supported columns')
    return ar[:,keep],np.asarray(observed)[keep],dict(raw_sequence_count=len(raw),unique_valid_pool_count=len(valid),
        selected_count=len(selected),selected_homologs_sha256=digest(selected),ambiguous_homologs_excluded=len(pool)-len(valid))

def weights_for(ar):
    w=np.empty(len(ar))
    for start in range(0,len(ar),16):
        a,b=ar[start:start+16,None,:],ar[None,:,:];valid=(a>=0)&(b>=0)
        overlap=valid.sum(2);matches=((a==b)&valid).sum(2)
        neighbors=((overlap>=.5*ar.shape[1])&(matches>=.8*overlap)).sum(1)
        w[start:start+16]=1/np.maximum(neighbors,1)
    return w

def profile(prov, out):
    for key in ('msa','mapping'):
        if sha(prov[key+'_path'])!=prov['source_hashes'][key]:raise ValueError('MSA source changed')
    raw=[]
    for line in Path(prov['msa_path']).read_text().splitlines():
        if line.startswith('>'):raw.append('')
        elif line.strip():raw[-1]+=line.strip()
    query=''.join(c.upper() for c in raw[0] if c.isalpha())
    if digest(query)!=prov['source_hashes']['observed_reference']:raise ValueError('Observed WT changed')
    key=digest(dict(source=prov['source_hashes'],policy=POLICY))
    path=out/'msa_profiles'/(key+'.json');wp=out/'msa_weights'/(key+'.npz')
    if path.exists():return json.loads(path.read_text()),path
    ar,columns,counts=alignment(raw);w=weights_for(ar);neff=float(w.sum())
    aa_counts=np.stack([((ar==i)*w[:,None]).sum(0) for i in range(20)],axis=1)
    depth=aa_counts.sum(1);prob=(aa_counts+.05)/(depth[:,None]+1)
    path.parent.mkdir(parents=True,exist_ok=True);wp.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(wp,weights=w,retained_msa_columns=columns)
    table=dict(AA_order=AA,source_hashes=prov['source_hashes'],homolog_count=len(w),global_neff=neff,
        pseudocount_total=1.,msa_policy=POLICY,provenance=counts,weights_sha256=sha(wp),
        rows=[dict(msa_column=int(col),probabilities_20=[float(x) for x in prob[j]],weighted_valid_count=float(depth[j]),
            weighted_gap_fraction=float(w[ar[:,j]<0].sum()/neff),weighted_unknown_fraction=0.) for j,col in enumerate(columns)])
    path.write_text(json.dumps(table,indent=2,allow_nan=False)+'\n')
    return table,path

def rebuild_rna(task, packet, out):
    result=copy.deepcopy(packet);tables={};columns={}
    for ch,prov in packet['source_provenance']['msa_chains'].items():
        table,path=profile(prov,out);tables[ch]=table
        mapping=json.loads(Path(prov['mapping_path']).read_text())
        columns[ch]={(r['position'],r.get('insertion_code') or ''):r['msa_column'] for r in mapping['columns']}
        result['source_provenance']['msa_chains'][ch]['probability_table']=str(path)
        result['msa_chains'][ch]=dict(global_neff=table['global_neff'],homolog_count=table['homolog_count'])
    result['msa']={};scores={};strict={}
    for candidate in task['candidates']:
        cid=candidate['id'];result['msa'][cid]={};total=0.;complete=True;strict_ok=True
        for site in candidate['sites']:
            identity=dict(group_id=task['group_id'],chain=site['chain'],position=site['position'],
                insertion_code=site.get('insertion_code') or '',wt=site['wt'],sequence_index_1based=site.get('sequence_index_1based'))
            sid='site_'+digest(identity)[:24];old=packet['msa'][cid][sid]
            row={k:None for k in ('p_wt','p_mutant','log_probability_ratio','weighted_valid_count','weighted_gap_fraction','weighted_unknown_fraction')}
            row['status']='unavailable_MSA';ch=site['chain']
            if ch in tables:
                col=columns[ch].get((site['position'],site.get('insertion_code') or ''))
                value=next((r for r in tables[ch]['rows'] if r['msa_column']==col),None)
                row['status']='no_valid_homolog_at_column'
                if old['status']=='WT_identity_mismatch':row['status']='WT_identity_mismatch'
                elif value is not None:
                    pw=value['probabilities_20'][AA.index(site['wt'])];pm=value['probabilities_20'][AA.index(site['mutant'])]
                    row.update(p_wt=pw,p_mutant=pm,log_probability_ratio=math.log(pm/pw),
                        status='ok' if tables[ch]['global_neff']>=20 else 'global_neff_below_frozen_20')
                    for k in ('weighted_valid_count','weighted_gap_fraction','weighted_unknown_fraction'):row[k]=value[k]
            result['msa'][cid][sid]=row
            complete &= row['log_probability_ratio'] is not None;strict_ok &= row['status']=='ok'
            if row['log_probability_ratio'] is not None:total+=row['log_probability_ratio']
        scores[cid]=total if complete else None;strict[cid]=total if strict_ok else None
    result.update(independent_msa_scores=scores,legacy_strict_independent_msa_scores=strict,
        full_msa_coverage=all(v is not None for v in scores.values()),legacy_strict_full_msa_coverage=all(v is not None for v in strict.values()),
        msa_scoring_policy=POLICY)
    return result
