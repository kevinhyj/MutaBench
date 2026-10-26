"""Reject missing Protein MSA evidence disguised as numbers; never remove candidates."""
from collections import Counter
from pathlib import Path

from msa_missingness import UNALIGNED, finite
from prepare_singlechain import resolve
from singlechain_catalog import load, sha, require


def inspect_packet(query, packet, profile):
    require(query['domain'] == 'protein', 'Protein query required')
    ids = [c['id'] for c in query['candidates']]
    require(len(ids) == len(set(ids)), 'Duplicate MSA query IDs')
    sites_by_id = packet.get('site_evidence', {})
    scores = packet.get('msa_only_scores', {})
    require(set(scores) <= set(ids), 'Unexpected MSA aggregate IDs')
    errors = Counter(); counts = Counter(); affected = set()
    if not sites_by_id:
        require(packet.get('status') == 'unavailable', 'MSA sites absent without unavailable status')
        for cid, value in scores.items():
            if value is not None: errors['score_without_evidence'] += 1; affected.add(cid)
        return dict(valid=not errors, counts=dict(counts), errors=dict(errors), affected_candidates=len(affected))
    require(profile is not None and set(sites_by_id) == set(ids), 'MSA profile/candidate coverage mismatch')
    require(set(scores) == set(ids), 'Missing MSA aggregate IDs')
    for candidate in query['candidates']:
        cid = candidate['id']; sites = sites_by_id[cid]
        require([s['position'] for s in sites] == candidate['changed_positions'] and sites,
                'Missing/reordered required MSA sites')
        missing, observed = [], []
        for site in sites:
            pos = site['position']; wt = query['wt_sequence'][pos-1]
            require((site['wt'], site['mutant']) == (wt, candidate['sequence'][pos-1]), 'MSA substitution mismatch')
            row = profile['positions'].get(str(pos))
            if row is None:
                reason = 'unmapped'
            else:
                require(row['wt'] == wt, 'MSA source WT mismatch')
                require(row.get('homolog_column') in {None, UNALIGNED}, 'Unknown MSA alignment provenance')
                depth = row['non_gap_effective_depth']
                require(finite(depth) and depth >= 0, 'Invalid raw observation depth')
                reason = 'unmapped' if row.get('homolog_column') == UNALIGNED else 'no_observations' if depth == 0 else None
            value = site.get('site_log_frequency_ratio')
            if reason:
                missing.append(site); counts[reason+'_site_occurrences'] += 1
                if value is not None: errors['missing_site_has_numeric_ratio'] += 1; affected.add(cid)
                if site.get('status') != 'unavailable' or not site.get('missing_reason'):
                    errors['missing_site_has_no_explicit_status'] += 1
                if reason == 'unmapped' and any(site.get(k) is not None for k in ('non_gap_effective_depth','gap_fraction')):
                    errors['unmapped_has_numeric_depth_or_gap'] += 1
            else:
                observed.append(site); counts['observed_site_occurrences'] += 1
                counts['observed_zero_ratios'] += value == 0
                if not finite(value) or site.get('status') != 'available':
                    errors['observed_site_invalid'] += 1; affected.add(cid)
        if missing:
            counts['candidates_requiring_null'] += 1
            counts['multisite_partially_observed_candidates'] += bool(observed)
            if scores[cid] is not None:
                errors['missing_required_site_has_candidate_score'] += 1; affected.add(cid)
                if observed: errors['partial_evidence_has_candidate_score'] += 1
        elif not finite(scores[cid]):
            errors['complete_evidence_has_invalid_score'] += 1; affected.add(cid)
    return dict(valid=not errors, counts=dict(counts), errors=dict(errors), affected_candidates=len(affected))


def check_packet(query, packet):
    profile = None; source = packet.get('source_profile')
    if packet.get('site_evidence'):
        require(source is not None, 'Protein MSA needs hashed upstream provenance')
        path = resolve(source['path'])
        require(sha(path) == source['sha256'], 'Protein MSA source changed')
        profile = load(path)
    report = inspect_packet(query, packet, profile)
    require(report['valid'], 'Invalid Protein MSA missingness/aggregate: ' + str(report['errors']))
    return report


def check_bundle(inputs):
    inputs = Path(inputs); rows = []
    for qp in sorted((inputs/'queries/protein').glob('*/query.json')):
        packet_path = inputs/'evidence/protein'/qp.parent.name/'msa.json'
        rows.append(dict(assay=qp.parent.name, query_sha256=sha(qp), packet_sha256=sha(packet_path),
                         **check_packet(load(qp), load(packet_path))))
    return dict(schema='protein-msa-guard-1', status='PASS', assays=rows,
                assay_count=len(rows), model_calls=0)
