"""Versioned missing-data adapters; observed zero and low support are retained."""
import copy
import hashlib
import json
import math
from pathlib import Path

VERSION = 'msa-missingness-v1'
UNALIGNED = 'query_insertion_unaligned_uniform_pseudocount_only'
MSA_MARKER = 'Natural-homolog MSA evidence; evolutionary preference is not assay fitness:\n'
COMPLEX_FIELDS = ('p_wt', 'p_mutant', 'log_probability_ratio', 'weighted_valid_count',
                  'weighted_gap_fraction', 'weighted_unknown_fraction')
MISSING_STATUSES = {'no_valid_homolog_at_column', 'unavailable_MSA', 'WT_identity_mismatch'}
OBSERVED_STATUSES = {'ok', 'global_neff_below_frozen_20'}
POLICY = ('Unmapped columns have null numerical evidence and an explicit reason. '
          'Observed zero counts or gap fractions remain observations; pseudocount-only '
          'log ratios with no valid observations are null. Low support with valid observations '
          'is retained. Missing is unknown, not a neutral effect. A multi-site score is null '
          'if any required site is unavailable; never sum only the available sites.')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def encode(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def reidentify(unit):
    unit.pop('logical_id', None)
    unit['logical_id'] = hashlib.sha256((encode(unit) + '\n').encode()).hexdigest()
    return unit


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def availability(usable):
    return 'available' if usable and all(usable) else 'partial' if any(usable) else 'unavailable'


def repair_protein_packet(query, packet, profile):
    """Read unrounded upstream column provenance, not rounded packet zeroes."""
    result = copy.deepcopy(packet)
    if packet.get('status') != 'available':
        return result
    if query['domain'] != 'protein' or packet['assay'] != query['assay']:
        raise ValueError('Protein assay identity mismatch')
    if set(packet['site_evidence']) != {c['id'] for c in query['candidates']}:
        raise ValueError('MSA candidate identity mismatch')
    all_usable, candidate_status = [], {}
    for candidate in query['candidates']:
        sites = result['site_evidence'][candidate['id']]
        if [s['position'] for s in sites] != candidate['changed_positions']:
            raise ValueError('MSA site positions mismatch')
        usable = []
        for site in sites:
            pos = site['position']
            if not 1 <= pos <= len(query['wt_sequence']):
                raise ValueError('Invalid MSA sequence position')
            wt, mutant = query['wt_sequence'][pos-1], candidate['sequence'][pos-1]
            if (site['wt'], site['mutant']) != (wt, mutant):
                raise ValueError('MSA substitution mismatch')
            row = profile['positions'].get(str(pos))
            reason = None
            if row is None:
                reason = 'position_absent_from_profile'
            elif row.get('wt') != wt:
                raise ValueError('MSA profile WT mismatch')
            elif row.get('homolog_column') == UNALIGNED:
                reason = 'query_insertion_unaligned'
            elif row.get('homolog_column') is not None:
                raise ValueError('Unknown homolog column provenance')
            if reason:
                for key in ('site_log_frequency_ratio', 'non_gap_effective_depth', 'gap_fraction'):
                    site[key] = None
                site.update(status='unavailable', missing_reason=reason, support='unavailable',
                            homolog_column=row.get('homolog_column') if row else None)
            else:
                depth, gap = row['non_gap_effective_depth'], row['gap_fraction']
                # Floating subtraction 1-depth/Neff can yield -4e-16.
                if not finite(depth) or depth < 0 or not finite(gap) or not -1e-12 <= gap <= 1+1e-12:
                    raise ValueError('Invalid upstream column statistics')
                # Rounded depth=0 is not sufficient evidence of missingness.
                if depth == 0:
                    site.update(site_log_frequency_ratio=None, status='unavailable',
                                missing_reason='no_valid_observations_at_aligned_column')
                else:
                    if not finite(site['site_log_frequency_ratio']):
                        raise ValueError('Invalid observed log frequency ratio')
                    site.update(status='available', missing_reason=None)
            usable.append(site['site_log_frequency_ratio'] is not None)
        if not all(usable):
            result['msa_only_scores'][candidate['id']] = None
        candidate_status[candidate['id']] = availability(usable)
        all_usable.extend(usable)
    result.update(schema='pg-rna-seed42-msa-evidence-v2', status=availability(all_usable),
                  missingness_policy_version=VERSION, candidate_msa_status=candidate_status,
                  missingness_coverage={'site_occurrences': len(all_usable),
                                        'available_site_occurrences': sum(all_usable),
                                        'unavailable_site_occurrences': len(all_usable)-sum(all_usable)},
                  caveat=packet['caveat'] + ' ' + POLICY)
    return result


def load_and_repair_protein_packet(query, item, packet, resolve):
    if packet.get('status') != 'available':
        return copy.deepcopy(packet)
    source = packet['source_profile']
    path = resolve(source['path'])
    if sha(path) != source['sha256'] or source['sha256'] != item['msa_profile']['sha256']:
        raise ValueError('MSA profile source hash mismatch')
    result = repair_protein_packet(query, packet, json.loads(path.read_text()))
    result['missingness_provenance'] = {'profile_path': str(path), 'profile_sha256': sha(path),
                                       'policy_version': VERSION}
    return result


def rebuild_protein_msa_unit(unit, packet):
    result = copy.deepcopy(unit)
    user = result['template']['user']
    if user.count(MSA_MARKER) != 1:
        raise ValueError('Expected one MSA evidence block')
    prefix, tail = user.split(MSA_MARKER)
    old_json, suffix = tail.split('\n\n', 1)
    ids = result['template']['expected_ids']
    if set(json.loads(old_json)['site_evidence']) != set(ids):
        raise ValueError('MSA request candidate mismatch')
    sites = {cid: packet['site_evidence'][cid] for cid in ids}
    status = availability([s['site_log_frequency_ratio'] is not None for rows in sites.values() for s in rows])
    block = dict(source_status=packet['source_status'], status=status, caveat=packet['caveat'],
                 missingness_policy_version=VERSION, site_evidence=sites)
    result['template']['user'] = prefix + MSA_MARKER + encode(block) + '\n\n' + suffix
    return reidentify(result)


def repair_complex_row(row):
    result = copy.deepcopy(row)
    status = result['status']
    if status in MISSING_STATUSES:
        for key in COMPLEX_FIELDS:
            result[key] = None
    elif status in OBSERVED_STATUSES:
        if not all(finite(result.get(k)) for k in COMPLEX_FIELDS):
            raise ValueError('Nonfinite observed complex MSA evidence')
        if result['weighted_valid_count'] <= 0:
            raise ValueError('Observed complex MSA row has no valid support')
        if not all(0 < result[k] <= 1 for k in ('p_wt', 'p_mutant')):
            raise ValueError('Invalid complex MSA probabilities')
        if not all(0 <= result[k] <= 1 for k in ('weighted_gap_fraction', 'weighted_unknown_fraction')):
            raise ValueError('Invalid complex MSA fractions')
    else:
        raise ValueError('Unknown complex MSA status: ' + str(status))
    return result


def repair_complex_features(features):
    result = copy.deepcopy(features)
    for cid, sites in result['msa'].items():
        for sid, row in sites.items():
            sites[sid] = repair_complex_row(row)
        # Preserve the profile's existing low-Neff scoring gate, but never
        # retain a partial sum if any site is genuinely unavailable.
        if any(row['status'] in MISSING_STATUSES for row in sites.values()):
            result['independent_msa_scores'][cid] = None
    result['full_msa_coverage'] = all(v is not None for v in result['independent_msa_scores'].values())
    result['missingness_policy'] = {'version': VERSION, 'definition': POLICY}
    return result


def split_ppi_user(user):
    decoder = json.JSONDecoder()
    for i, ch in enumerate(user):
        if ch != '{':
            continue
        try:
            value, end = decoder.raw_decode(user[i:])
        except ValueError:
            continue
        if isinstance(value, dict) and {'candidates', 'binding_partners', 'evidence'} <= value.keys():
            return user[:i], value, user[i+end:]
    raise ValueError('PPI task context missing')


def rebuild_ppi_user(user, features, expected_ids):
    prefix, context, suffix = split_ppi_user(user)
    if [c['id'] for c in context['candidates']] != expected_ids:
        raise ValueError('PPI candidate order mismatch')
    evidence = context['evidence']
    if 'msa_sites' not in evidence:
        return user
    if set(evidence['msa_sites']) != set(expected_ids):
        raise ValueError('PPI MSA candidate mismatch')
    for cid, rows in evidence['msa_sites'].items():
        if {row['site_id'] for row in rows} != set(features['msa'][cid]) or len(rows) != len(features['msa'][cid]):
            raise ValueError('PPI MSA site identity mismatch')
        for row in rows:
            repaired = repair_complex_row(row)
            source = features['msa'][cid][row['site_id']]
            for key in (*COMPLEX_FIELDS, 'status'):
                if repaired[key] != source[key]:
                    raise ValueError('PPI rendered MSA differs from repaired features')
                row[key] = source[key]
    evidence['msa_missingness'] = {'version': VERSION, 'definition': POLICY}
    return prefix + encode(context) + suffix
