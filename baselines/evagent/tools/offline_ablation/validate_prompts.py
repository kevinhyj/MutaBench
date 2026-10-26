"""Audit the actual frozen wire prompts, separately from the renderer."""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.output.resolve()
    plans = json.loads((root / 'plan.json').read_text())
    config = json.loads((root / 'config.json').read_text())
    forbidden = {'probabilities_20', 'msa_column', 'msa_source_hashes', 'p_aa20',
                 'native_benefit', 'raw_measurement', 'normalized_utility',
                 'canonical_effect', 'canonical_score', 'experimental_score_mean',
                 'experimental_scores', 'effect_label', 'labels', 'scores'}
    counts = Counter()
    for unit in plans:
        path = root / unit['request_path']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == unit['request_sha256']
        request = json.loads(path.read_text())
        context = json.loads(request['contents'][0]['parts'][0]['text'].rsplit('\n', 1)[1])
        generation = request['generationConfig']
        assert generation['seed'] == 42 and generation['temperature'] == 0
        assert generation['thinkingConfig']['thinkingLevel'] == config['thinking_level']
        candidates = context['candidates']
        assert [candidate['id'] for candidate in candidates] == unit['expected_ids']
        assert len(candidates) == len(set(unit['expected_ids'])) == 25

        def visit(obj):
            if isinstance(obj, dict):
                assert not set(obj) & forbidden, set(obj) & forbidden
                for value in obj.values():
                    visit(value)
            elif isinstance(obj, list):
                for value in obj:
                    visit(value)
        visit(context)
        evidence = context['evidence']
        if 'msa_chains' in evidence:
            chains = evidence['msa_chains']
            assert len(chains) == len({chain['chain'] for chain in chains})
            assert all(set(chain) == {'chain', 'homolog_count', 'global_neff'} for chain in chains)
            assert set(evidence['msa_sites']) == set(unit['expected_ids'])
        for candidate in candidates:
            expected = {site['site_id']: site for site in candidate['sites']}
            assert len(expected) == len(candidate['sites'])
            for category in ('msa_sites', 'structure_sites'):
                if category not in evidence:
                    continue
                rows = evidence[category][candidate['id']]
                assert {row['site_id'] for row in rows} == set(expected)
                for row in rows:
                    base = expected[row['site_id']]
                    for key in ('chain', 'position', 'wt', 'mutant'):
                        assert row[key] == base[key]
                    if category == 'msa_sites':
                        required = {'p_wt', 'p_mutant', 'log_probability_ratio',
                                    'weighted_valid_count', 'weighted_gap_fraction',
                                    'weighted_unknown_fraction', 'status'}
                        assert required <= set(row)
                        if row['status'] != 'ok':
                            assert row['p_wt'] is None and row['p_mutant'] is None
                            assert row['log_probability_ratio'] is None
                        counts['MSA_site_' + row['status']] += 1
                    else:
                        required = {'min_partner_heavy_distance_A', 'partner_contact_count_5A',
                                    'sasa_bound_A2', 'delta_sasa_partner_A2',
                                    'site_status', 'partner_status', 'sasa_status', 'partner_chain_ids', 'source'}
                        assert required <= set(row)
                        if row['sasa_status'] != 'ok':
                            assert row['sasa_bound_A2'] is None and row['delta_sasa_partner_A2'] is None
                        counts['structure_SASA_' + row['sasa_status']] += 1
        counts['condition_' + unit['condition']] += 1
    report = dict(status='passed', requests_verified=len(plans), CR=25,
                  evidence_identity_and_fields_complete=True,
                  no_labels_or_full_probability_vectors_or_MSA_columns=True,
                  chain_summary_once_per_chain_per_request=True,
                  explicit_missing_values=True, counts=dict(counts))
    (root / 'actual_prompt_validation.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
