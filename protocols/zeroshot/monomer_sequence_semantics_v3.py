"""Versioned monomer-only semantic repair; shared-2/complex files stay frozen.

Preserves every query, group, SYSTEM and site feature. No measured outcomes,
inference, sequence sampling or geometry computation are performed here.
"""
import argparse
import collections
import copy
import csv
import json
from pathlib import Path
import shutil

from shared_protocol import require, sha, save, encode, digest
from monomer_integrity import strict_load as load, strict_json, label_free
from prepare_shared_prompts import block_status

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SCHEMA = 'evagent-monomer-sequences-3'
DEFAULT_PARENT = HERE/'inputs/monomer_sequences_scoped_20261015_v2'
METADATA = ROOT/'work/proteingym_v1/source/DMS_substitutions.csv'


def task_definition(query, metadata=None):
    domain, assay = query['domain'], query['assay']
    context = copy.deepcopy(query['assay_context'])
    benchmark = 'ProteinGym' if domain == 'protein' else 'RNAGym'
    if domain == 'protein':
        require(metadata is not None and metadata['DMS_id'] == assay and
                metadata['target_seq'] == query['wt_sequence'], 'Protein task metadata identity mismatch')
        require(metadata['raw_DMS_directionality'] in ('1', '-1') and
                metadata['raw_DMS_phenotype_name'], 'Protein readout/direction missing')
        multiplier = int(metadata['raw_DMS_directionality'])
        name = metadata['raw_DMS_phenotype_name']
        orientation = dict(status='verified_reference_metadata', multiplier=multiplier,
            meaning='Orientation relative to the named raw readout; this does not specify the full numerical preprocessing formula.')
        meaning = ('Larger' if multiplier == 1 else 'Smaller') + ' values of the original readout "' + name + '" correspond to the higher ProteinGym benchmark direction.'
        readout = dict(benchmark_readout='Processed ProteinGym DMS_score', original_readout_name=name,
            assay_description=context['selection_assay'], measurement_method=context['selection_type'],
            raw_to_benchmark_orientation=orientation,
            conditions='Use the named readout column and supplied assay context. Do not invent unprovided concentrations, units or experimental conditions.')
    else:
        require(domain == 'rna' and context.get('ASSAY_DESCRIPTION'), 'RNA assay description missing')
        meaning = 'Larger processed RNAGym DMS_score for this assay is the prediction target; keep the supplied processed-score direction, with no additional sign inversion.'
        readout = dict(benchmark_readout='Processed RNAGym DMS_score',
            assay_description=context['ASSAY_DESCRIPTION'], original_readout_name=None,
            raw_to_benchmark_orientation=dict(status='not_supplied_in_frozen_metadata', multiplier=None,
                meaning='The frozen RNA context does not specify an original raw readout column or its conversion. Do not equate affinity with Kd, infer a sign, or invent a numerical transform.'),
            conditions='Use the supplied assay description. Unspecified conditions, physical units and raw-to-processed transforms remain unknown.')
    readout['prediction_scale'] = 'Dimensionless signed effect relative to the supplied reference in the processed benchmark direction; not an experimental-unit prediction. Zero denotes the predicted reference-relative effect, not a measured WT readout.'
    task = dict(dataset_id=benchmark, assay_id=assay, modality=domain, assay_context=context,
        readout_definition=readout, score_target=dict(kind='relative_assay_effect', reference='supplied_reference',
            higher_score_means=meaning + ' Higher does not imply a universal biological benefit.'))
    label_free(task)
    return task


def current_rna_confidence(definition):
    result = copy.deepcopy(definition)
    result['semantics_version'] = 'monomer-semantics-20261015-v3'
    result['computation'].pop('confidence_actual', None)
    result['computation'].pop('confidence_required', None)
    result.pop('new_input_gate', None)
    result['computation']['confidence'] = (
        'For predicted_WT with confidence_status=verified, plddt is the original predicted CIF confidence '
        'of the selected anchor atom after verified atom identity and coordinate mapping to the supplied WT. '
        'Experimental structure B-factors are not pLDDT. Unverified or unavailable confidence is null. '
        'A valid observed zero remains zero; absent or zero-filled confidence without provenance is not evidence. '
        'Confidence describes the reference structure, not the effect of a mutation.')
    return result


def repair_unit(unit, task):
    require(unit['domain'] in ('protein', 'rna'), 'Monomer sequence unit required')
    result = copy.deepcopy(unit)
    user = strict_json(result['template']['user'])
    require(task['assay_id'] == unit['assay'] and task['modality'] == unit['domain'], 'Task identity differs')
    user['schema_version'] = SCHEMA
    user['task'] = copy.deepcopy(task)
    for kind, block in user['evidence'].items():
        if block['status'] == 'not_requested':
            continue
        block['status'] = block_status(block['variants'], kind, unit['domain'])
        if kind == 'structure' and unit['domain'] == 'rna':
            block['definition'] = current_rna_confidence(block['definition'])
    label_free(user)
    result['template']['user'] = encode(user)
    result.pop('logical_id')
    result['logical_id'] = digest(result)
    return result


def verify_allowed_diff(before, after):
    """Independent allowlist: semantic fixes must never alter identity or numbers."""
    require({k:v for k,v in before.items() if k not in {'template','logical_id'}} ==
            {k:v for k,v in after.items() if k not in {'template','logical_id'}}, 'Request grouping changed')
    require(before['template']['system'] == after['template']['system'] and
            before['template']['expected_ids'] == after['template']['expected_ids'], 'SYSTEM/IDs changed')
    old = strict_json(before['template']['user']); new = strict_json(after['template']['user'])
    require(old['reference'] == new['reference'] and old['candidates'] == new['candidates'] and
            old['output_contract'] == new['output_contract'], 'Candidate/reference/output contract changed')
    old.pop('schema_version'); new.pop('schema_version')
    old_task = old.pop('task'); new_task = new.pop('task')
    require({k:v for k,v in old_task.items() if k != 'score_target'} ==
            {k:v for k,v in new_task.items() if k not in {'score_target','readout_definition'}}, 'Assay context changed')
    for kind in old['evidence']:
        a,b = old['evidence'][kind],new['evidence'][kind]
        if a['status'] == 'not_requested':
            require(a == b, 'Disabled evidence changed'); continue
        a.pop('status'); b.pop('status')
        if kind == 'structure' and before['domain'] == 'rna':
            require(current_rna_confidence(a['definition']) == b['definition'], 'RNA definition drift')
            a.pop('definition'); b.pop('definition')
        require(a == b, 'Evidence values/provenance changed')
    require(old == new, 'Unapproved prompt change')


def implementation_hashes():
    names = ['monomer_sequence_semantics_v3.py','prepare_shared_prompts.py','shared_protocol.py',
             'monomer_integrity.py','prompt_semantics.py']
    return {name:sha(HERE/name) for name in names}


def checked_files(path, manifest):
    require(sha(path/'MANIFEST.json') == (path/'MANIFEST.sha256').read_text().strip(), 'Manifest checksum changed')
    for name, expected in manifest['files'].items():
        file = (path/name).resolve()
        require(file.is_relative_to(path.resolve()) and sha(file) == expected, 'Frozen input changed: '+name)


def expected_tasks(bundle, parent):
    metadata = load(bundle/'TASK_METADATA.json')
    result = {}
    for qp in sorted((bundle/'queries').glob('*/*/query.json')):
        q = load(qp); key = q['domain']+'/'+q['assay']
        source_query = parent/'queries'/q['domain']/q['assay']/'query.json'
        require(qp.read_bytes() == source_query.read_bytes(), 'Query bytes changed')
        result[key] = task_definition(q, metadata.get(q['assay']) if q['domain']=='protein' else None)
    require(set(metadata) == {k.split('/',1)[1] for k in result if k.startswith('protein/')}, 'Task metadata scope changed')
    return result


def validate_bundle(bundle):
    bundle = Path(bundle).resolve(); m = load(bundle/'MANIFEST.json')
    require(m['schema'] == SCHEMA and m['fully_ready'] is True, 'Monomer sequence v3 required')
    checked_files(bundle,m)
    require(m['implementation_hashes'] == implementation_hashes(), 'Monomer v3 implementation changed')
    parent = Path(m['parent']['path'])
    require(sha(parent/'MANIFEST.json') == m['parent']['manifest_sha256'], 'Parent manifest changed')
    pm = load(parent/'MANIFEST.json'); checked_files(parent,pm)
    require(pm['fully_ready'] and pm['schema']=='evagent-zero-shot-shared-2', 'Accepted shared-2 parent required')
    tasks = expected_tasks(bundle,parent)
    require(tasks == load(bundle/'TASK_DEFINITIONS.json'), 'Task definitions differ from metadata')
    # Compare only allowlisted fields of the independent public reference table.
    meta = m['task_metadata_source']; require(sha(meta['path']) == meta['sha256'], 'Task reference changed')
    with Path(meta['path']).open() as f:
        reference = {r['DMS_id']:r for r in csv.DictReader(f)}
    for assay, row in load(bundle/'TASK_METADATA.json').items():
        require(row == {k:reference[assay][k] for k in row}, 'Task metadata differs from original reference')
    before,after = load(parent/'QUEUE.json'),load(bundle/'QUEUE.json')
    require(len(before) == len(after) == m['queue_units'], 'Queue coverage changed')
    for a,b in zip(before,after):
        require(b == repair_unit(a,tasks[a['domain']+'/'+a['assay']]), 'Request differs from monomer v3 renderer')
        verify_allowed_diff(a,b)
    require(m['domains']==pm['domains'] and m['arms']==pm['arms'] and
            m['candidate_resampling'] is False and m['seed_sampling']==42, 'Cohort contract changed')
    require((bundle/'SYSTEM.txt').read_bytes()==(parent/'SYSTEM.txt').read_bytes(), 'SYSTEM changed')
    return m,after


def prepare(version, parent=DEFAULT_PARENT, here=HERE):
    from prepare_shared_prompts import safe_name
    from shared_protocol import validate_bundle as validate_parent
    safe_name(version); parent=Path(parent).resolve(); here=Path(here)
    out=here/'inputs'/version; ev=here/'evaluator'/version
    require(not out.exists() and not ev.exists(), 'New monomer input version required')
    pm, queue = validate_parent(parent)
    require(set(pm['domains'])<= {'protein','rna'}, 'Sequence monomers only')
    out.mkdir(parents=True)
    with METADATA.open() as f: metadata={r['DMS_id']:r for r in csv.DictReader(f)}
    allowed = ['DMS_id','target_seq','raw_DMS_phenotype_name','raw_DMS_directionality']
    selected={q.parent.name:{k:metadata[q.parent.name][k] for k in allowed}
              for q in (parent/'queries/protein').glob('*/query.json')}
    save(out/'TASK_METADATA.json',selected)
    shutil.copytree(parent/'queries',out/'queries')
    shutil.copyfile(parent/'SYSTEM.txt',out/'SYSTEM.txt')
    tasks=expected_tasks(out,parent); save(out/'TASK_DEFINITIONS.json',tasks)
    repaired=[];diff=[];status_changes=collections.Counter();rna_changes=0
    for unit in queue:
        new=repair_unit(unit,tasks[unit['domain']+'/'+unit['assay']]);verify_allowed_diff(unit,new)
        old_user=strict_json(unit['template']['user']);new_user=strict_json(new['template']['user'])
        changed=[]
        for kind in old_user['evidence']:
            if old_user['evidence'][kind]['status'] != new_user['evidence'][kind]['status']:
                changed.append(kind);status_changes[unit['domain']+'/'+kind]+=1
        rna_updated=unit['domain']=='rna' and 'structure' in unit['arm'];rna_changes+=rna_updated
        diff.append(dict(domain=unit['domain'],assay=unit['assay'],arm=unit['arm'],chunk_index=unit['chunk_index'],
            old_request_id=unit['logical_id'],new_request_id=new['logical_id'],
            task_definition_added=True,status_channels_changed=changed,rna_confidence_definition_changed=rna_updated))
        repaired.append(new)
    save(out/'QUEUE.json',repaired);save(out/'REQUEST_CHANGES.json',diff)
    save(out/'VALIDATION.json',dict(status='PASS',original_parent_replayed=True,requests=len(repaired),
        status_changes=dict(status_changes),rna_confidence_definition_changes=rna_changes,
        negative_direction_protein_assays=sum(r['raw_DMS_directionality']=='-1' for r in selected.values()),
        query_bytes_unchanged=True,site_values_unchanged=True,system_unchanged=True,
        labels_read=False,model_calls=0,allowed_changes=['schema_version','task readout/direction','chunk status','RNA confidence definition']))
    m=dict(schema=SCHEMA,protocol_version=SCHEMA,version=version,fully_ready=True,
        parent=dict(path=str(parent),manifest_sha256=sha(parent/'MANIFEST.json')),
        task_metadata_source=dict(path=str(METADATA),sha256=sha(METADATA)),
        implementation_hashes=implementation_hashes(),domains=pm['domains'],arms=pm['arms'],
        queue_units=len(repaired),seed_sampling=42,candidate_resampling=False,
        files={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()})
    save(out/'MANIFEST.json',m);(out/'MANIFEST.sha256').write_text(sha(out/'MANIFEST.json')+'\n')
    refs=load(here/'evaluator'/pm['version']/'LABEL_REFERENCES.json')
    for ref in refs:
        qp=out/'queries'/ref['domain']/ref['assay']/'query.json'
        require(sha(qp)==ref['query_sha256'], 'Evaluator query changed')
        ref['query_path']=str(qp)
    save(ev/'LABEL_REFERENCES.json',refs)
    validate_bundle(out)
    return m


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--version');parser.add_argument('--parent',type=Path,default=DEFAULT_PARENT)
    parser.add_argument('--validate',type=Path);args=parser.parse_args()
    if args.validate:print(validate_bundle(args.validate)[0]['queue_units'])
    elif args.version:print(prepare(args.version,args.parent)['version'])
    else:parser.error('--version or --validate required')
