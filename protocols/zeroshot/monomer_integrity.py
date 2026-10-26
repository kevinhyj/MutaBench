"""Task routing, predictor-only messages, and exact result identities.

This module never loads label values. Evaluation alone may use strict_load on
the separately pinned label file after complete prediction coverage is proved.
"""
import collections
import json
from pathlib import Path
import re

from frozen_prompt_bundle import message_sha256
from shared_protocol import require, finite

VERSION='monomer-result-integrity-1'
IDENTITY=('domain','assay','arm','round_index','chunk_index')
_NORMALIZE=re.compile(r'[\s_-]')
_FORBIDDEN={_NORMALIZE.sub('',k).lower() for k in (
    'label','labels','labelpath','labelsha256','evaluationlabels','testlabels','labelvalues',
    'groundtruth','groundtruthscore','groundtruthscores','truescore','truescores','targetvalue',
    'experimental_score','experimental_scores','experimental_score_mean','dms_score','fitness',
    'fitnessvalue','fitnessvalues','msa_only_scores','independent_msa_scores','y','scores')}


def strict_json(text):
    def unique(pairs):
        result={}
        for key,value in pairs:
            require(key not in result,'Duplicate JSON key: '+key)
            result[key]=value
        return result
    def invalid(value):
        raise ValueError('Nonfinite JSON constant: '+value)
    return json.loads(text,object_pairs_hook=unique,parse_constant=invalid)


def strict_load(path):
    return strict_json(Path(path).read_text())


def task_metric(domain, ref):
    if domain in {'protein','rna'}:
        require(ref.get('kind','regression')=='regression' and ref.get('metric','spearman')=='spearman',
                'Protein/RNA require signed Spearman')
        require(ref.get('label_container') in {None,'labels'},'Invalid sequence label container')
        return 'regression','spearman'
    require(domain=='admet' and ref.get('kind') in {'classification','regression'},'Unknown task metric route')
    name='auroc' if ref['kind']=='classification' else 'spearman'
    require(ref.get('metric')==name,'ADMET metric mismatch')
    require(ref.get('label_container')=='admet_endpoint_y' and ref.get('label_key')==ref['assay'],
            'ADMET label endpoint mismatch')
    return ref['kind'],name


def label_free(value):
    if isinstance(value,dict):
        for key,item in value.items():
            require(_NORMALIZE.sub('',key).lower() not in _FORBIDDEN,'Evaluator field in predictor: '+key)
            label_free(item)
    elif isinstance(value,list):
        for item in value:label_free(item)


def check_request(request):
    require(set(request)=={'request_id','domain','assay','arm','chunk_index','round_index',
                          'expected_ids','messages','messages_sha256'},'Unexpected predictor request fields')
    for key in ('request_id','domain','assay','arm'):
        require(isinstance(request[key],str) and request[key] and '/' not in request[key] and
                '\\' not in request[key] and request[key] not in {'.','..'},'Invalid request identity')
    domain=request['domain'];require(domain in {'protein','rna','admet'},'Unsupported monomer task')
    require(request['arm'] in ({'smiles','smiles_structure'} if domain=='admet' else
            {'sequence','msa_prompt','structure','msa_prompt_structure'}),'Task arm mismatch')
    require(type(request['round_index']) is int and request['round_index'] in (range(3) if domain=='admet' else (0,)),
            'Wrong prediction round')
    require(type(request['chunk_index']) is int and request['chunk_index']>=0,'Invalid prediction chunk')
    ids=request['expected_ids']
    require(isinstance(ids,list) and ids and all(isinstance(i,str) and i for i in ids) and
            len(ids)==len(set(ids)),'Expected nonempty unique string IDs')
    messages=request['messages']
    require(isinstance(messages,list) and len(messages)==2 and
            [m.get('role') for m in messages]==['system','user'] and
            all(set(m)=={'role','content'} and isinstance(m['content'],str) for m in messages),
            'Unexpected predictor message envelope')
    require(message_sha256(messages)==request['messages_sha256'],'Request message hash mismatch')
    user=strict_json(messages[1]['content']);label_free(user)
    require(user['task']['modality']==domain and user['task']['assay_id']==request['assay'],
            'Prompt task identity mismatch')
    require([c['variant_id'] for c in user['candidates']]==ids and
            user['output_contract']['expected_variant_ids']==ids,'Prompt candidate identity/order mismatch')


def check_request_set(requests):
    require(requests and len({r['request_id'] for r in requests})==len(requests),'Duplicate/empty request IDs')
    groups=collections.defaultdict(list)
    for r in requests:
        check_request(r);groups[(r['domain'],r['assay'],r['arm'])].append(r)
    cohorts={};arms=collections.defaultdict(set)
    for (domain,assay,arm),units in groups.items():
        key=(domain,assay);arms[key].add(arm)
        ids=check_rounds(units,domain)
        require(key not in cohorts or cohorts[key]==ids,'Paired arms have different candidate sets')
        cohorts[key]=ids
    for (domain,assay),values in arms.items():
        require(values==({'smiles','smiles_structure'} if domain=='admet' else
                {'sequence','msa_prompt','structure','msa_prompt_structure'}),'Incomplete task arms')


def check_rounds(units, domain, expected_ids=None):
    grouped=collections.defaultdict(list)
    require(len({u['request_id'] for u in units})==len(units),'Duplicate planned request')
    for u in units:
        require(type(u['round_index']) is int,'Invalid round type')
        require(type(u['chunk_index']) is int and u['chunk_index']>=0,'Invalid chunk type')
        grouped[u['round_index']].append(u)
    require(set(grouped)==({0,1,2} if domain=='admet' else {0}),'Missing/extra prediction round')
    target=set(expected_ids) if expected_ids is not None else None
    for rs in grouped.values():
        require(sorted(r['chunk_index'] for r in rs)==list(range(len(rs))),'Duplicate/noncontiguous chunks')
        ids=[i for r in rs for i in r['expected_ids']]
        require(ids and len(ids)==len(set(ids)),'Duplicate candidate within round')
        if target is None:target=set(ids)
        require(set(ids)==target,'Round candidate coverage mismatch')
    return target


def check_scores(scores, ids):
    require(isinstance(scores,dict) and set(scores)==set(ids),'Missing/extra prediction ID')
    require(all(finite(v) and -1<=v<=1 for v in scores.values()),'Invalid prediction value')


def check_receipt(row, request, model_key):
    require(isinstance(row,dict) and set(row)=={'integrity_version','model_key','identity','request_id',
            'messages_sha256','status','failure_category','value'},'Unexpected result receipt fields')
    require(row.get('integrity_version')==VERSION,'Unverified legacy result receipt')
    require(row.get('model_key')==model_key and row.get('request_id')==request['request_id'] and
            row.get('messages_sha256')==request['messages_sha256'] and
            row.get('identity')=={k:request[k] for k in IDENTITY},'Prediction request/model/round mismatch')
    require(row.get('status') in {'ok','failed'},'Unknown receipt status')
    if row['status']=='ok':check_scores(row.get('value'),request['expected_ids'])
    else:
        require(isinstance(row.get('failure_category'),str) and row['failure_category'],
                'Failed result requires a failure category')
        require(row.get('value') is None,'Failed result must not carry accepted scores')


def check_result_inventory(directory, requests):
    directory=Path(directory);expected={r['request_id'] for r in requests}
    require(len(expected)==len(requests),'Duplicate planned requests')
    if not directory.exists():return
    files={p.stem for p in directory.glob('*.json')}
    require(files<=expected,'Unexpected result file (unknown task/round/request)')
    require({p.stem for p in directory.glob('*.sha256')}<=expected,'Unexpected result hash file')
