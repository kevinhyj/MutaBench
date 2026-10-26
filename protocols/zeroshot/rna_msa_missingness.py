"""Do not expose zero-depth RNA pseudocounts as observed evolutionary evidence.

The legacy RNA profile already rounded depths. A stored zero cannot establish
positive observational support. Preserve that uncertainty explicitly rather than
claiming it proves a truly empty column. Positive depths and true zero ratios stay.
"""
import copy
import math

from shared_protocol import load, sha, require, finite

VERSION = 'rna-profile-observation-gate-2'
MISSING_REASON = 'upstream_rounded_depth_zero_positive_observation_not_established'


def repair(query, packet, profile):
    result=copy.deepcopy(packet)
    if not packet.get('site_evidence'):
        require(packet.get('status')=='unavailable', 'RNA MSA sites missing')
        require(not any(v is not None for v in packet.get('msa_only_scores',{}).values()),
                'RNA aggregate without evidence')
        return result
    require(query['domain']=='rna' and query['assay']==packet['assay'], 'RNA MSA identity mismatch')
    require(set(result['site_evidence'])=={c['id'] for c in query['candidates']}, 'RNA MSA candidate mismatch')
    changed=0;valid=0;unavailable_candidates=0
    for candidate in query['candidates']:
        rows=result['site_evidence'][candidate['id']]
        require([s['position'] for s in rows]==candidate['changed_positions'], 'RNA MSA site order mismatch')
        incomplete=False
        for i,site in enumerate(rows):
            pos=site['position'];raw=profile['positions'].get(str(pos))
            require(raw is not None and raw['wt']==site['wt']==query['wt_sequence'][pos-1], 'RNA MSA WT mismatch')
            require(site['mutant']==candidate['sequence'][pos-1], 'RNA MSA mutant mismatch')
            depth=raw['non_gap_effective_depth']
            require(finite(depth) and depth>=0, 'RNA profile depth invalid')
            if depth==0:
                if site.get('status') == 'unavailable':
                    require(site.get('reason') == MISSING_REASON, 'Unknown RNA missingness reason')
                else:
                    require(site.get('non_gap_effective_depth') == 0, 'RNA profile depth mismatch')
                rows[i]=dict(position=pos,wt=site['wt'],mutant=site['mutant'],status='unavailable',
                    reason=MISSING_REASON)
                incomplete=True;changed+=1
            else:
                require(finite(site.get('non_gap_effective_depth')) and
                        site['non_gap_effective_depth'] in {round(depth,3), depth}, 'RNA profile depth mismatch')
                ratio=math.log(raw['frequency'][site['mutant']]/raw['frequency'][site['wt']])
                require(abs(round(ratio,5)-site['site_log_frequency_ratio'])<1e-9, 'RNA profile ratio mismatch')
                if site['non_gap_effective_depth']==0:site['non_gap_effective_depth']=depth
                valid+=1
        if incomplete:
            result.setdefault('msa_only_scores',{})[candidate['id']]=None;unavailable_candidates+=1
    result['status']='partial' if changed and valid else 'unavailable' if changed else 'available'
    result['observation_gate']=dict(version=VERSION,changed_site_occurrences=changed,retained_site_occurrences=valid,
        candidates_with_incomplete_msa=unavailable_candidates,
        note='Null total for any unavailable site; no partial-site sum. Stored zero depth is not proof of an exact raw zero.')
    return result


def load_and_repair(query, packet, resolve):
    if not packet.get('site_evidence'):
        return repair(query,packet,None)
    source=packet['source_profile'];path=resolve(source['path'])
    require(sha(path)==source['sha256'], 'RNA profile source changed')
    profile=load(path)
    if 'sources' in profile:profile=profile['sources']['msa']
    require(profile['status']=='available', 'RNA source profile unavailable')
    return repair(query,packet,profile)
