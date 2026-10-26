"""Explicit endpoint semantics for independent molecules; frozen v1 stays readable.

Preparation is offline: unchanged official rows, groups and saved geometry.
No labels are parsed, no conformers generated and no model is called.
"""
import argparse
import copy
import json
from pathlib import Path
import shutil

import admet_protocol as legacy
from shared_protocol import load, save, sha, digest, encode, require

HERE = Path(__file__).resolve().parent
VERSION = 'evagent-admet-molecules-2'
REGISTRY = HERE/'admet_endpoints_v2.json'
SYSTEM_PATH = HERE/'prompts/admet_v2/system.txt'
DEFAULT_SOURCE = HERE/'inputs/admet22_monomer_cr3_20261015_v1'


def endpoint_definitions(tasks):
    registry = load(REGISTRY)
    require(registry['schema']=='admet-endpoint-semantics-2','Unknown endpoint registry')
    source = registry['tasks_source']
    require(sha(source['path'])==source['sha256'],'Endpoint metadata source hash mismatch')
    require(tasks==load(source['path']),'Endpoint task identity differs from frozen registry')
    definitions=registry['endpoints']
    require(set(definitions)=={t['endpoint'] for t in tasks},'Endpoint registry coverage mismatch')
    for task in tasks:
        d=definitions[task['endpoint']];cls=task['kind']=='classification'
        require(d['kind']==task['kind'] and d['frozen_description']==task['description'],
                'Endpoint kind/description mismatch')
        require(d['metric']==('auroc' if cls else 'spearman'),'Endpoint metric mismatch')
        require(all(isinstance(d['readout'][k],str) and d['readout'][k].strip()
                    for k in ('name','unit','encoding')),'Endpoint readout units/encoding missing')
        require(bool(d['sources']) and bool(d['higher_score_means']),'Missing endpoint provenance/direction')
        if cls:
            require(d['classes']['positive']['value']==1 and d['classes']['negative']['value']==0
                    and d['classes']['positive']['meaning'] and d['classes']['negative']['meaning'],
                    'Classification requires explicit label 1/0 meanings')
            require(d['readout']['unit']=='binary label','Classification is not a physical-unit readout')
        else:
            require(d['classes'] is None and d['decision_threshold'] is None
                    and d['readout']['higher_value_means'],'Regression cannot use class semantics')
    return definitions


def render(tasks, shapes, groups, definitions):
    require(definitions==endpoint_definitions(tasks),'Unverified endpoint semantics')
    queue=legacy.render(tasks,shapes,groups,SYSTEM_PATH.read_text())
    for unit in queue:
        user=json.loads(unit['template']['user']);definition=definitions[unit['assay']]
        user['schema_version']=VERSION
        user['task']['endpoint_definition']=copy.deepcopy({k:v for k,v in definition.items()
                                                         if k not in {'sources','frozen_description'}})
        user['task']['metric']=definition['metric']
        user['reference']=dict(kind='independent_molecules')
        user['output_contract']['score'].update(
            zero_meaning='No special biological reference or decision threshold.',
            comparable_across_endpoints=False,
            semantics='ordinal_positive_class_propensity' if definition['kind']=='classification' else 'ordinal_numerical_readout')
        unit['template']['user']=encode(user)
        unit.pop('logical_id');unit['logical_id']=digest(unit)
    return queue


def implementation_hashes():
    paths=[Path(__file__),REGISTRY,SYSTEM_PATH,HERE/'shared_protocol.py']
    return legacy.implementation_hashes() | {str(p):sha(p) for p in paths}


def validate_bundle(bundle):
    bundle=Path(bundle).resolve();m=load(bundle/'MANIFEST.json')
    require(m.get('schema')==VERSION and m.get('fully_ready') is True,'ADMET v2 not ready')
    require(sha(bundle/'MANIFEST.json')==(bundle/'MANIFEST.sha256').read_text().strip(),'Manifest changed')
    require(m['implementation_hashes']==implementation_hashes(),'ADMET v2 implementation changed')
    required={'tasks.json','shapes.json','groups.json','SYSTEM.txt','QUEUE.json','ENDPOINTS.json','ACCEPTANCE.json'}
    require(required<=set(m['files']),'Missing ADMET v2 files')
    for name,h in m['files'].items():
        p=(bundle/name).resolve()
        require(p.is_relative_to(bundle) and sha(p)==h,'ADMET v2 file changed: '+name)
    source=Path(m['parent']['path'])
    require(sha(source/'MANIFEST.json')==m['parent']['manifest_sha256'],'Parent manifest changed')
    parent,_=legacy.validate_bundle(source)
    for name in ('tasks.json','shapes.json','groups.json'):
        require((source/name).read_bytes()==(bundle/name).read_bytes(),'Frozen rows/features/groups changed')
    tasks=load(bundle/'tasks.json')
    require(load(bundle/'ENDPOINTS.json')==endpoint_definitions(tasks),'Endpoint semantics changed')
    require((bundle/'SYSTEM.txt').read_bytes()==SYSTEM_PATH.read_bytes(),'Molecule SYSTEM mismatch')
    expected=render(tasks,load(bundle/'shapes.json'),load(bundle/'groups.json'),load(bundle/'ENDPOINTS.json'))
    require(expected==load(bundle/'QUEUE.json'),'ADMET v2 request replay mismatch')
    for task in tasks:
        name=Path('queries/admet')/task['endpoint']/'query.json'
        require(str(name) in m['files'] and (source/name).read_bytes()==(bundle/name).read_bytes(),
                'Molecule IDs/SMILES/order changed')
    require(m['queue_units']==len(expected) and m['domains']==parent['domains'] and
            m['arms']==list(legacy.ARMS) and m['rounds']==3,'ADMET v2 scope mismatch')
    return m,expected


def prepare(version, source=DEFAULT_SOURCE, here=HERE):
    from prepare_shared_prompts import safe_name
    safe_name(version);source=Path(source).resolve();here=Path(here)
    out,ev=here/'inputs'/version,here/'evaluator'/version
    require(not out.exists() and not ev.exists(),'New version required')
    parent,_=legacy.validate_bundle(source)
    tasks=load(source/'tasks.json');definitions=endpoint_definitions(tasks)
    queue=render(tasks,load(source/'shapes.json'),load(source/'groups.json'),definitions)
    refs=load(source.parent.parent/'evaluator'/parent['version']/'LABEL_REFERENCES.json')
    by_endpoint={t['endpoint']:t for t in tasks}
    require(len(refs)==len(by_endpoint) and {r['assay'] for r in refs}==set(by_endpoint),
            'Missing/duplicate endpoint evaluator reference')
    for ref in refs:
        qp=source/'queries/admet'/ref['assay']/'query.json';d=definitions[ref['assay']]
        require(ref['domain']=='admet' and Path(ref['query_path']).resolve()==qp and
                sha(qp)==ref['query_sha256'],'Evaluator query mismatch')
        require(ref['kind']==d['kind'] and ref['metric']==d['metric'] and
                ref['label_key']==ref['assay'] and ref['label_container']=='admet_endpoint_y',
                'Evaluator endpoint semantics mismatch')
        require(sha(ref['label_path'])==ref['label_sha256'],'Evaluator label hash mismatch')
    out.mkdir(parents=True)
    for name in ('tasks.json','shapes.json','groups.json'):
        shutil.copyfile(source/name,out/name)
    shutil.copytree(source/'queries',out/'queries')
    save(out/'ENDPOINTS.json',definitions);save(out/'QUEUE.json',queue)
    (out/'SYSTEM.txt').write_bytes(SYSTEM_PATH.read_bytes())
    save(out/'ACCEPTANCE.json',dict(status='PASS',endpoints=len(tasks),requests=len(queue),
         official_rows=sum(len(t['candidates']) for t in tasks),
         endpoint_semantics='Explicit readout/units/encoding/direction/positive-class definitions',
         paired_arms=list(legacy.ARMS),candidate_resampling=False,labels_read=False,model_calls=0))
    m=dict(schema=VERSION,protocol_version=VERSION,version=version,fully_ready=True,
           parent=dict(path=str(source),manifest_sha256=sha(source/'MANIFEST.json')),
           queue_units=len(queue),domains=parent['domains'],arms=list(legacy.ARMS),rounds=3,
           candidate_resampling=False,labels_read=False,model_calls=0,
           implementation_hashes=implementation_hashes(),
           files={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()})
    save(out/'MANIFEST.json',m);(out/'MANIFEST.sha256').write_text(sha(out/'MANIFEST.json')+'\n')
    for ref in refs:ref['query_path']=str(out.resolve()/'queries/admet'/ref['assay']/'query.json')
    save(ev/'LABEL_REFERENCES.json',refs)
    validate_bundle(out)
    return m


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--version');p.add_argument('--source',type=Path,default=DEFAULT_SOURCE)
    p.add_argument('--validate',type=Path);a=p.parse_args()
    if a.validate:print(validate_bundle(a.validate)[0]['queue_units'])
    elif a.version:print(prepare(a.version,a.source)['queue_units'])
    else:p.error('--version or --validate required')
