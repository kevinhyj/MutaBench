"""Offline evaluator for repaired MSA-only scores; incomplete candidates stay null."""
import argparse
from pathlib import Path
from scipy.stats import rankdata

from shared_protocol import load, save, sha, require, finite, validate_bundle
from evaluate_monomers import metric
from protein_msa_guard import check_packet
from rna_msa_guard import check_packet as check_rna_packet


def rank_fusion(ids, llm, msa):
    """Frozen 50:50 average-rank fusion; no partial query or sign flipping."""
    require(ids and len(ids)==len(set(ids)), 'Duplicate/empty fusion cohort')
    require(set(ids)==set(llm)==set(msa), 'Fusion cohort mismatch')
    require(all(finite(v) for v in list(llm.values())+list(msa.values())), 'Missing fusion component')
    a=rankdata([llm[i] for i in ids],method='average')
    b=rankdata([msa[i] for i in ids],method='average')
    return {i:float((x+y)/2) for i,x,y in zip(ids,a,b)}


def evaluate(inputs, output):
    inputs=Path(inputs).resolve();output=Path(output).resolve()
    require(not output.exists(), 'Use a new result version')
    manifest,_=validate_bundle(inputs)
    refs=load(inputs.parents[1]/'evaluator'/manifest['version']/'LABEL_REFERENCES.json')
    rows=[];scores={}
    for ref in refs:
        domain,aid=ref['domain'],ref['assay']
        require(domain in {'protein','rna'}, 'Sequence assay required')
        qp=Path(ref['query_path']);require(sha(qp)==ref['query_sha256'], 'Query changed')
        query=load(qp);ids=[c['id'] for c in query['candidates']]
        packet=load(inputs/'evidence'/domain/aid/'msa.json')
        if domain == 'protein':
            check_packet(query, packet)  # Before labels: a finite partial sum is not a complete score.
        else:
            check_rna_packet(query, packet)
        values=packet.get('msa_only_scores',{})
        require(set(values)<=set(ids), 'Unexpected MSA-only IDs')
        predictions={i:values.get(i) for i in ids};usable=sum(finite(v) for v in predictions.values())
        rho=reason=None;status='complete' if usable==len(ids) else 'incomplete_evidence'
        if status=='complete':
            lp=Path(ref['label_path']);require(sha(lp)==ref['label_sha256'], 'Labels changed')
            labels=load(lp)['labels'];require(set(labels)==set(ids), 'Label mismatch')
            rho,reason=metric('regression',[labels[i] for i in ids],[predictions[i] for i in ids])
        rows.append(dict(domain=domain,assay=aid,status=status,reason=reason,n_expected=len(ids),n_finite=usable,
            cohort='N250' if len(ids)==250 else 'short_query',spearman=rho,query_sha256=ref['query_sha256']))
        scores[domain+'/'+aid]=predictions
    save(output/'METRICS.json',dict(rows=rows,source_manifest_sha256=sha(inputs/'MANIFEST.json'),
        missing_not_zero=True,partial_not_scored=True,model_calls=0))
    save(output/'PREDICTIONS.json',scores)
    return rows


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--inputs',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();rows=evaluate(a.inputs,a.output);print(dict(assays=len(rows),complete=sum(r['status']=='complete' for r in rows)))
