"""Independent RNA evidence/aggregate guard using the hashed source profile."""
from pathlib import Path
from collections import Counter
import math

from prepare_singlechain import resolve
from singlechain_catalog import load, sha, require
from shared_protocol import finite
from rna_msa_missingness import MISSING_REASON


def inspect_packet(query, packet, profile):
    require(query['domain']=='rna', 'RNA query required')
    ids = [c['id'] for c in query['candidates']]
    require(len(ids)==len(set(ids)), 'Duplicate RNA candidates')
    rows = packet.get('site_evidence',{}); scores=packet.get('msa_only_scores',{})
    require(set(scores)<=set(ids), 'Unexpected RNA aggregate IDs')
    counts=Counter();errors=Counter()
    if not rows:
        require(packet.get('status')=='unavailable', 'Missing RNA sites without unavailable status')
        errors['aggregate_without_evidence'] += sum(v is not None for v in scores.values())
    else:
        require(packet.get('assay')==query['assay'] and set(rows)==set(ids) and set(scores)==set(ids),
                'RNA evidence/aggregate identities differ')
        require(profile is not None and profile['status']=='available', 'RNA source profile missing')
        for c in query['candidates']:
            sites=rows[c['id']]; require([s['position'] for s in sites]==c['changed_positions'] and sites,
                                        'RNA site order/coverage mismatch')
            incomplete=False
            for site in sites:
                pos=site['position'];raw=profile['positions'].get(str(pos))
                require(raw is not None and raw['wt']==site['wt']==query['wt_sequence'][pos-1] and
                        site['mutant']==c['sequence'][pos-1], 'RNA site source identity mismatch')
                depth=raw['non_gap_effective_depth'];require(finite(depth) and depth>=0,'Invalid RNA source depth')
                if depth==0:
                    incomplete=True;counts['zero_source_depth_site_occurrences']+=1
                    if site.get('status')!='unavailable' or site.get('reason')!=MISSING_REASON:
                        errors['missing_status_or_uncertainty_reason']+=1
                    if any(site.get(k) is not None for k in ('site_log_frequency_ratio','non_gap_effective_depth','gap_fraction')):
                        errors['zero_source_depth_has_numeric_evidence']+=1
                else:
                    counts['positive_source_depth_site_occurrences']+=1
                    ratio=site.get('site_log_frequency_ratio')
                    expected=round(math.log(raw['frequency'][site['mutant']]/raw['frequency'][site['wt']]),5)
                    if not finite(ratio) or abs(ratio-expected)>1e-9:errors['observed_ratio_mismatch']+=1
                    if not finite(site.get('non_gap_effective_depth')) or site['non_gap_effective_depth']<=0 or \
                            site['non_gap_effective_depth'] not in {round(depth,3),depth}:
                        errors['positive_source_depth_lost']+=1
                    counts['observed_zero_ratios_retained']+=ratio==0
            if incomplete:
                counts['candidates_requiring_null']+=1
                if scores[c['id']] is not None: errors['incomplete_candidate_has_score']+=1
            elif not finite(scores[c['id']]):errors['observed_candidate_score_invalid']+=1
    return dict(valid=not any(errors.values()),counts=dict(counts),errors={k:v for k,v in errors.items() if v})


def check_packet(query, packet):
    profile=None
    if packet.get('site_evidence'):
        src=packet['source_profile'];p=resolve(src['path'])
        require(sha(p)==src['sha256'],'RNA source profile changed')
        profile=load(p);profile=profile.get('sources',{}).get('msa',profile)
    report=inspect_packet(query,packet,profile)
    require(report['valid'],'Invalid RNA MSA missingness/aggregate: '+str(report['errors']))
    return report


def check_bundle(inputs):
    inputs=Path(inputs);rows=[]
    for qp in sorted((inputs/'queries/rna').glob('*/query.json')):
        ep=inputs/'evidence/rna'/qp.parent.name/'msa.json'
        rows.append(dict(assay=qp.parent.name,query_sha256=sha(qp),packet_sha256=sha(ep),
                         **check_packet(load(qp),load(ep))))
    return dict(schema='rna-msa-guard-1',status='PASS',assay_count=len(rows),assays=rows,model_calls=0)
