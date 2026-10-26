"""Recheck RNA confidence against its original atoms, independently of status flags."""
from pathlib import Path
from collections import Counter
import json

from rna_confidence import repair_packet, pdb_atoms, cif_confidence, resolve
from singlechain_catalog import REGISTRY, load, sha, require, write_new


def check_packet(query, item, legacy, actual):
    require(query.get('domain')=='rna','RNA query required')
    expected=repair_packet(query,item,legacy)
    require(actual.get('assay')==query['assay'] and actual.get('status')==expected['status'],
            'RNA structure assay/status mismatch')
    require(actual.get('kind')==item['structure_kind'],'RNA structure kind mismatch')
    require(actual.get('confidence_provenance')==expected['confidence_provenance'],
            'RNA confidence provenance differs from verified source')
    require(actual.get('site_evidence')==expected['site_evidence'],
            'RNA confidence/geometry differs from original verified atoms')
    source=actual['source'];path=resolve(source['path'])
    require(sha(path)==source['sha256'],'Derived RNA structure hash mismatch')
    original_path=resolve(item['structure']['path'])
    old_atoms, new_atoms=pdb_atoms(original_path),pdb_atoms(path)
    require(old_atoms.keys()==new_atoms.keys(),'Derived RNA atom identities changed')
    for key,(xyz,line) in new_atoms.items():
        original_xyz,original_line=old_atoms[key]
        require(xyz==original_xyz and line[:60]+line[66:]==original_line[:60]+original_line[66:],
                'Derived RNA geometry or atom metadata changed')
    provenance=expected['confidence_provenance'];values={}
    if provenance['status']=='verified':
        require(item['structure_kind']=='predicted_WT','Experimental/unknown source cannot supply pLDDT')
        values=cif_confidence(Path(provenance['cif_path']),old_atoms)
        for key,(_,line) in new_atoms.items():
            require(float(line[60:66])==round(values[key],2),'Derived PDB confidence differs from original CIF')
    else:
        require(all(s.get('plddt') is None for sites in actual['site_evidence'].values() for s in sites),
                'Unavailable confidence must be null')
    return dict(assay=query['assay'],kind=item['structure_kind'],confidence_status=provenance['status'],
                reason=provenance.get('reason'),verified_atoms=len(values),
                checked_site_occurrences=sum(map(len,actual['site_evidence'].values())),
                status='PASS',source_hashes={str(path):sha(path),str(original_path):sha(original_path)} |
                {provenance[k+'_path']:provenance[k+'_sha256'] for k in ('cif','snapshot') if k+'_path' in provenance})


def check_prompt(unit, packet):
    shown=json.loads(unit['template']['user'])['evidence']['structure'];counts=Counter()
    provenance=packet.get('confidence_provenance',{})
    require(shown['source_kind']==packet.get('kind','unavailable'),'Rendered RNA structure kind mismatch')
    require(shown['confidence_status']==provenance.get('status','unverified'),
            'Rendered RNA confidence provenance mismatch')
    require(shown.get('confidence_unavailable_reason')==provenance.get('reason'),
            'Rendered RNA confidence reason mismatch')
    require([v['variant_id'] for v in shown['variants']]==unit['template']['expected_ids'],
            'RNA structure prompt candidate mismatch')
    for variant in shown['variants']:
        sites=packet.get('site_evidence',{}).get(variant['variant_id'])
        if sites is None:
            require(packet.get('status')=='unavailable' and bool(variant['sites']) and
                    all(s['features'].get('status')=='unavailable' and s['features'].get('plddt') is None
                        for s in variant['sites']),'Structure-free RNA prompt has numeric confidence')
            counts['unavailable_structure_sites']+=len(variant['sites'])
            continue
        require(len(sites)==len(variant['sites']),'Rendered RNA structure site count mismatch')
        for source,target in zip(sites,variant['sites']):
            require(target['sequence_index_1based']==source['position'] and
                    target['features']=={k:v for k,v in source.items() if k not in {'position','wt','mutant'}},
                    'Actual RNA structure prompt differs from verified evidence')
            counts['site_occurrences']+=1;counts['null_confidence_occurrences']+=source['plddt'] is None
    counts['structure_request_blocks']+=1
    return counts


def check_bundle(inputs, *, registry=REGISTRY):
    inputs=Path(inputs);config=load(registry)['historical']
    def checked(ref):
        p=Path(ref['path'])
        if not p.is_absolute():p=Path(registry).resolve().parent/p
        require(sha(p)==ref['sha256'],'Frozen RNA confidence registry source changed')
        return p
    ip,pp=checked(config['inventory']),checked(config['plan'])
    inventory={r['assay']:r for r in load(ip)['rna']};plan=load(pp)
    rows=[];packets={}
    for qp in sorted((inputs/'queries/rna').glob('*/query.json')):
        aid=qp.parent.name;query=load(qp)
        ep=inputs/'evidence/rna'/aid/'structure.json';packet=load(ep);packets[aid]=packet
        if aid not in inventory:
            # Registered extensions without prepared structure, e.g. Ke.
            require(packet.get('status')=='unavailable' and not packet.get('site_evidence') and
                    packet.get('reason') and packet.get('confidence_provenance',{}).get('status')!='verified',
                    'Unregistered RNA structure source')
            rows.append(dict(assay=aid,kind='unavailable',confidence_status='unavailable',status='PASS',
                             reason=packet['reason'],verified_atoms=0,checked_site_occurrences=0,source_hashes={}))
            continue
        require(sha(qp)==plan['input_hashes']['rna/'+aid]['query'],'RNA confidence query changed')
        old=ip.parent/'evidence/rna'/aid/'structure.json'
        require(sha(old)==plan['input_hashes']['rna/'+aid]['structure'],'Original RNA evidence changed')
        result=check_packet(query,inventory[aid],load(old),packet)
        result.update(query_sha256=sha(qp),packet_sha256=sha(ep));rows.append(result)
    counts=Counter(); queue=load(inputs/'QUEUE.json') if rows else []
    for unit in queue:
        if unit['domain']!='rna' or 'structure' not in unit['arm']:continue
        counts.update(check_prompt(unit,packets[unit['assay']]))
    return dict(schema='rna-structure-source-guard-1',status='PASS',assay_count=len(rows),assays=rows,
                confidence_counts=dict(Counter(r['kind']+'/'+r['confidence_status'] for r in rows)),
                prompt_checks=dict(counts),model_calls=0,labels_read=False)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    require(not args.output.exists(),'New RNA structure audit output required')
    result=check_bundle(args.inputs)
    write_new(args.output,result)
    print(json.dumps({k:v for k,v in result.items() if k!='assays'},ensure_ascii=False))
