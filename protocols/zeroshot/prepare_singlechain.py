"""Replay the confirmed singlechain sampler; freeze label-free inputs and requests."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
from sampling import singlechain_select
from prepare_shared_prompts import add_assay, finish, save
from rna_confidence import VERSION as RNA_CONFIDENCE_VERSION, repair_packet
from msa_missingness import VERSION as MSA_MISSINGNESS_VERSION, load_and_repair_protein_packet

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
OLD=ROOT/'experiments/pg_rna_seed42_n250_v1'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):return json.loads(Path(p).read_text())
def resolve(path):
    p=Path(path)
    if p.is_file():return p
    return Path(str(path).replace('/data/yanjie_huang/EVAgent/',str(ROOT)+'/').replace('/data/yanjie_huang/','/data/huangyanjie/'))


def main():
    a=argparse.ArgumentParser();a.add_argument('--version',required=True);args=a.parse_args()
    if not args.version.replace('_','').replace('-','').isalnum():raise ValueError('Simple version name required')
    output=HERE/'inputs'/args.version
    if output.exists() or (HERE/'evaluator'/args.version).exists():raise FileExistsError('New version directory required')
    spec=importlib.util.spec_from_file_location('historical_singlechain_prepare',OLD/'prepare.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    inventory=load(OLD/'INVENTORY.json');replay=[];hashes={};evaluation=[];rna_packets={};protein_msa_packets={}
    oldplan=load(OLD/'PLAN.json')
    output.mkdir(parents=True)
    names=[];queue=[];provenance={str(p):sha(p) for p in (OLD/'PLAN.json',OLD/'INVENTORY.json',OLD/'prepare.py',Path(__file__),HERE/'sampling.py',HERE/'sampling_contract.json')}
    for domain in ['protein','rna']:
        for original in inventory[domain]:
            item=dict(original);item['source_csv']=dict(original['source_csv']);item['source_csv']['path']=str(resolve(item['source_csv']['path']))
            wt,rows,exclusions=mod.source_rows(item,domain)
            selected=singlechain_select(rows,n=250,seed=42)
            aid=item['assay'];qroot=OLD/'queries'/domain/aid;query=load(qroot/'query.json')
            if sha(qroot/'query.json')!=oldplan['input_hashes'][domain+'/'+aid]['query']:raise ValueError('Frozen query changed: '+aid)
            candidates=query['candidates']
            if len(selected)!=len(candidates) or wt!=query['wt_sequence']:raise ValueError('Frozen cohort mismatch: '+aid)
            for index,(row,candidate) in enumerate(zip(selected,candidates),1):
                if candidate['id']!=f'M{index:03d}' or any(candidate[k]!=row[k] for k in ['mutation','sequence','changed_positions']):
                    raise ValueError('Independent sampler mismatch: '+aid)
                if set(candidate)-{'id','mutation','sequence','changed_positions'}:raise ValueError('Unexpected candidate fields')
            dest=output/'queries'/domain/aid;dest.mkdir(parents=True)
            shutil.copyfile(qroot/'query.json',dest/'query.json')
            hashes[str((dest/'query.json').relative_to(output))]=sha(dest/'query.json')
            evidence={};packets={}
            for kind in ['msa','structure']:
                src=OLD/'evidence'/domain/aid/(kind+'.json');d=load(src)
                if sha(src)!=oldplan['input_hashes'][domain+'/'+aid][kind]:raise ValueError('Frozen evidence changed: '+aid)
                ep=output/'evidence'/domain/aid/(kind+'.json');ep.parent.mkdir(parents=True,exist_ok=True)
                if domain=='protein' and kind=='msa':
                    d=load_and_repair_protein_packet(query,item,d,resolve)
                    ep.write_text(json.dumps(d,ensure_ascii=False,sort_keys=True,allow_nan=False)+'\n')
                    protein_msa_packets[aid]=d
                elif domain=='rna' and kind=='structure':
                    pdb=output/'structures'/'rna'/aid/'mapped_WT.pdb'
                    d=repair_packet(query,item,d,pdb_output=pdb)
                    ep.write_text(json.dumps(d,ensure_ascii=False,sort_keys=True,allow_nan=False)+'\n')
                    rna_packets[aid]=d
                    if pdb.exists():hashes[str(pdb.relative_to(output))]=sha(pdb)
                else:shutil.copyfile(src,ep)
                hashes[str(ep.relative_to(output))]=sha(ep);evidence[kind]=d.get('status');packets[kind]=d
                provenance[str(src)]=sha(src)
            name,units=add_assay(output,domain,aid,dict(query=query,**packets))
            names.append(name);queue.extend(units)
            provenance[str(qroot/'query.json')]=sha(qroot/'query.json')
            label=resolve(qroot/'evaluation_labels.json')
            evaluation.append(dict(domain=domain,assay=aid,query_path=str(dest/'query.json'),query_sha256=sha(dest/'query.json'),
                                   label_path=str(label),label_sha256=sha(label)))
            replay.append(dict(domain=domain,assay=aid,n=len(selected),pool_n=len(rows),source_csv=item['source_csv'],
                               selected_order_match=True,evidence=evidence,exclusions=exclusions))
    save(output/'SAMPLING_REPLAY.json',dict(seed=42,assays=replay,
        rna_confidence_version=RNA_CONFIDENCE_VERSION,msa_missingness_version=MSA_MISSINGNESS_VERSION,
        candidate_resampling=False,model_calls=0))
    manifest=finish(output,args.version,names,queue,provenance)
    ev=HERE/'evaluator'/args.version;ev.mkdir(parents=True)
    (ev/'LABEL_REFERENCES.json').write_text(json.dumps(evaluation,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'assays':len(replay),'queue_units':len(queue),'sampling_replay':'PASS','output':str(output),'model_calls':0}))


if __name__=='__main__':main()
