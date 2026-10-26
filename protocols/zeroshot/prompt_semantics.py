"""Versioned, model-visible meanings; no labels or biological feature computation.

Historical packets keep their own algorithms. Definitions describe those algorithms,
including their limitations, rather than silently giving them a common meaning.
"""
import copy
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
VERSION = 'prompt-semantics-20261015-v2'
PROFILE_SOURCE = HERE / 'audits/feature_contract_20261014_v1/PROFILES.json'

MSA_LIMITATIONS = (
    'MSA frequencies describe amino-acid or nucleotide composition among the selected natural homologs. '
    'They are not probabilities of being homologous, measured fitness, measured binding affinity, '
    'or probabilities that a substitution improves the assay outcome. Evolutionary preference can '
    'reflect folding, expression, other functions, and selection conditions different from this assay. '
    'A positive frequency log ratio means the mutant symbol is more frequent than WT at that column; '
    'it does not establish a positive assay effect. A negative ratio does not establish a negative assay effect. '
    'Interpret this evidence in the supplied assay and partner context; do not mechanically replace '
    'the requested prediction with frequency ranking. Summing per-site log ratios is an additive '
    'evolutionary statistic, not a measurement or a model of epistasis. Missing evidence is unknown, '
    'not neutral, deleterious, or zero; weak support is uncertainty rather than a measured effect.'
)
STRUCTURE_LIMITATIONS = (
    'These features describe the supplied reference WT coordinates, not a relaxed or predicted mutant '
    'structure. Distance, contact counts, solvent exposure and burial are geometric context, not '
    'measured affinity, binding energy, or the effect of a substitution. A larger contact count or '
    'buried area does not by itself imply a more favorable mutation. Missing atoms or unavailable '
    'features mean unknown; an observed zero contact count is different from missing coordinates. '
    'Site, partner and SASA availability must be read separately.'
)
COMMON = {
    'task': 'Predict only the stated assay readout and its declared direction. Higher is not automatically healthier or biologically preferable.',
    'identity': 'Candidate IDs identify outputs within this assay. Sequence indices are one-based; PDB residue numbers require chain and insertion code and are not sequence indices.',
    'missing': 'not_requested means disabled by the arm; unavailable means requested but absent; partial means only some required sites or fields are usable. null is unknown, never numeric zero.',
    'score': 'Follow the task-specific output range and unit. Dimensionless relative effects are not experimental kcal/mol. Do not normalize or rank scores separately inside each batch.',
    'joint': 'MSA and structure describe different evidence and are not independent measurements of the assay outcome. Their effects need not agree or add linearly.',
}

COMPLEX_MSA_FIELDS = {
    'p_wt': 'Smoothed weighted frequency of the WT amino acid at the mapped protein MSA column; dimensionless in (0,1].',
    'p_mutant': 'Smoothed weighted frequency of the mutant amino acid at the same mapped column; same denominator as p_wt, dimensionless in (0,1].',
    'log_probability_ratio': 'Natural logarithm ln(p_mutant / p_wt), dimensionless; positive means mutant more frequent, not stronger binding.',
    'weighted_valid_count': 'Sum of homolog weights with a canonical amino acid at this column; weighted count, before pseudocount, not raw sequence count or chain-level Neff.',
    'weighted_gap_fraction': 'Sum of weights with a gap at this column divided by chain-level global_neff; dimensionless [0,1].',
    'weighted_unknown_fraction': 'Sum of weights with an unknown symbol divided by global_neff; dimensionless [0,1]. Entire ambiguous homologs are excluded in this profile, so valid-column values are zero.',
    'status': 'ok means the mapped site passes this frozen profile gate. Other values name a missing-data or quality reason, not an assay-effect direction.',
    'global_neff': 'Chain-level sum of inverse-neighbor homolog weights; not the number of valid observations at every site.',
}
SINGLE_MSA_FIELDS = {
    'site_log_frequency_ratio': 'ln(frequency[mutant]/frequency[wt]), natural log, dimensionless; evolutionary composition rather than assay effect.',
    'non_gap_effective_depth': 'Sum of weights with a canonical symbol at the column; weighted count, distinct from chain-level Neff.',
    'gap_fraction': 'Dimensionless [0,1]. Protein legacy includes gaps and invalid symbols; RNA legacy counts literal gaps only. Use the specified profile.',
    'support': 'low when valid depth <20 or gap fraction >0.5 before producer rounding, otherwise adequate; quality flag, not predicted function.',
}
GEOMETRY_FIELDS = {
    'min_partner_heavy_distance_A': 'Minimum WT site heavy-atom to specified partner heavy-atom distance, angstrom.',
    'partner_contact_count_5A': 'Number of distinct partner residues with any heavy atom within <=5 angstrom of a WT site heavy atom; not atom-pair count.',
    'sasa_bound_A2': 'Solvent-accessible surface area of this WT residue in the declared bound environment, angstrom squared.',
    'delta_sasa_partner_A2': 'SASA of this same WT residue after removing the specified opposite partner minus its SASA in the bound environment, angstrom squared; not total interface or group SASA.',
    'site_status': 'Whether WT site identity and required coordinates are available; does not guarantee partner or SASA completeness.',
    'partner_status': 'Availability of the specified partner coordinates, separate from site and SASA status.',
    'sasa_status': 'ok only when the required WT heavy atoms and SASA calculation are available; otherwise SASA values are unknown.',
    'partner_chain_ids': 'Frozen opposite partner chain group, not automatically the nearest chain.',
    'source': 'Provenance of the reference coordinates, not evidence of a mutant structure.',
}


def definitions():
    """All ten audited computation profiles, with explicit model-facing semantics."""
    profiles = json.loads(PROFILE_SOURCE.read_text())['profiles']
    result = {}
    for name, profile in profiles.items():
        details = {k: copy.deepcopy(v) for k, v in profile.items()
                   if k not in {'producer', 'findings', 'external_sources'}}
        item = {'semantics_version': VERSION, 'computation': details, 'common': COMMON.copy()}
        if 'msa' in name:
            item['interpretation'] = MSA_LIMITATIONS
            item['fields'] = copy.deepcopy(SINGLE_MSA_FIELDS if name.startswith('singlechain') else COMPLEX_MSA_FIELDS)
            if name == 'singlechain_msa_protein_legacy':
                item['historical_exception'] = 'Unsupported WT insertion columns have pseudocount-only uniform frequencies, depth=0, gap=1 and log ratio=0. This zero is absence of preference, not neutral assay effect. New packets must mark unsupported columns explicitly.'
            elif name == 'complex_protein_site_msa_legacy':
                item['historical_exception'] = 'Legacy PPI unsupported columns may contain count=0/gap=1 placeholders with a non-ok status and null probabilities. They are not observed data; new packets must preserve the missing reason and null unavailable measurements.'
            elif name == 'protein_rna_msa_v2':
                item['low_support'] = 'Low global Neff is a retained quality flag in this profile, not automatically missing. This is protein-site MSA, not RNA-partner MSA.'
        elif name == 'singlechain_protein_structure_legacy':
            item['interpretation'] = STRUCTURE_LIMITATIONS
            item['fields'] = {
                'plddt': 'Prediction confidence 0..100 only with verified AlphaFold provenance; experimental or unverified B-factor is not pLDDT. Not mutation effect.',
                'sasa_a2': 'Absolute SASA of this residue in the isolated matched chain, angstrom squared; not complex interface area. Missing-heavy-atom completeness is not guaranteed by this legacy producer.',
                'nonlocal_ca_contacts_8a': 'Observed same-chain C-alpha neighbors <=8 angstrom, excluding |sequence index difference|<=2; residue count, not partner contacts.',
                'nearest_nonlocal': 'Up to three nearest eligible same-chain C-alpha neighbors and distances in angstrom, without an 8-angstrom cutoff.',
            }
        elif name == 'singlechain_rna_structure_legacy':
            item['interpretation'] = STRUCTURE_LIMITATIONS
            item['fields'] = {
                'anchor_atom': "C4-prime if observed, else C1-prime, else P; not a protein C-alpha.",
                'nonlocal_neighbors_8a': 'Observed anchor neighbors <=8 angstrom, excluding |sequence index difference|<=3; anchor count.',
                'nearest_nonlocal_positions': 'Up to three nearest eligible anchor positions and distances in angstrom, without an 8-angstrom cutoff.',
                'plddt': 'Confidence 0..100 only if original predicted atom confidence and coordinate mapping are verified. Zero-filled exported B-factors are unavailable confidence, not low-confidence predictions; use null if provenance is unverified.',
            }
            item['new_input_gate'] = 'Reject unverified pLDDT, including the 20 historical mapped files with lost B-factors, or explicitly replace only that unavailable channel with null in a new version.'
        elif name == 'protein_rna_structure_contact_legacy':
            item['interpretation'] = STRUCTURE_LIMITATIONS
            item['fields'] = {
                'direct_rna_contact': 'True if any WT residue heavy atom, including backbone, is <=5 angstrom from the specified assay RNA; false only with available geometry and no such contact. Not a sidechain-only test.',
                'rna_contact_moieties': 'RNA base, ribose, phosphate classes contacted by the whole WT residue, including backbone; optional and absent when classification is unavailable.',
            }
            # Describe the corrected prompt, while keeping the historical defect in audit metadata.
            item['computation'].pop('actual_boolean', None)
            item['computation'].pop('required', None)
        elif name in {'complex_protein_site_structure_legacy', 'protein_ligand_structure_legacy'}:
            item['interpretation'] = STRUCTURE_LIMITATIONS
            item['fields'] = copy.deepcopy(GEOMETRY_FIELDS)
            if name == 'protein_ligand_structure_legacy':
                item['fields']['partner_contact_count_5A'] = 'Number of distinct annotated ligand heavy atoms within <=5 angstrom of any WT site heavy atom; not residues or atom pairs. All matching ligand copies in the first model are included.'
                item['fields']['delta_sasa_partner_A2'] = 'SASA of this same WT residue after removing only the annotated ligand copies minus its bound SASA, angstrom squared; all reference protein chains remain. Not total complex SASA.'
        elif name == 'admet_molecule_structure_legacy':
            item['interpretation'] = 'Whole-molecule shape descriptors of generated conformers, not measured ADMET outcomes, a bound pose, or guaranteed lowest-energy/converged conformers. No WT, mutation sites or MSA apply. Higher prediction follows the endpoint definition; higher toxicity is not beneficial.'
            item['fields'] = {k: details['descriptor_definitions'][k] + '; unit: ' + v for k, v in details['descriptor_units'].items()}
        else:
            raise ValueError('Unaudited profile: ' + name)
        result[name] = item
    return result


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def coverage(block, kind):
    """Count candidate-site occurrences, not unique physical sites."""
    full = partial = absent = 0
    for variant in block['variants']:
        for site in variant['sites']:
            f = site['features']
            if kind == 'msa':
                usable = f.get('status') == 'ok' and all(finite(f.get(k)) for k in ('p_wt', 'p_mutant'))
                full += usable
                absent += not usable
            else:
                keys = ('min_partner_heavy_distance_A', 'partner_contact_count_5A', 'sasa_bound_A2', 'delta_sasa_partner_A2')
                available = [finite(f.get(k)) for k in keys]
                full += all(available)
                partial += any(available) and not all(available)
                absent += not any(available)
    total = full + partial + absent
    status = 'available' if total and full == total else 'partial' if full or partial else 'unavailable'
    return status, dict(site_occurrences=total, fully_usable_site_occurrences=full,
                        partially_usable_site_occurrences=partial, unavailable_site_occurrences=absent,
                        counting='Each candidate-site occurrence; not unique physical sites. Field/site status still governs use.')


def repair_shared_user(user):
    """Only semantics/availability metadata change; numerical evidence is preserved."""
    out = copy.deepcopy(user)
    domain = out['task']['modality']
    if domain not in {'protein_dna', 'protein_ligand'}:
        raise ValueError('Shared-complex repair currently supports only DNA and ligand')
    for kind, block in out['evidence'].items():
        if block['status'] == 'not_requested':
            continue
        if kind == 'msa':
            fields = copy.deepcopy(COMPLEX_MSA_FIELDS)
            if domain == 'protein_dna':
                fields.pop('log_probability_ratio')
            block['definition'] = dict(
                semantics_version=VERSION,
                probability='(weighted amino-acid count + 1/20)/(weighted valid column count + 1); total pseudocount is 1 across 20 canonical amino acids',
                weighting='Inverse count of neighbors with >=80% identity and >=50% valid overlap on retained columns; at most 2048 lexically sorted unique non-WT homologs; entire ambiguous homologs excluded',
                fields=fields, min_global_neff=20,
                interpretation=MSA_LIMITATIONS,
                missing='Use the frozen per-site status and supplied values. Non-ok sites do not pass this profile gate; null values remain unknown. Some covered sites do not imply complete coverage of this candidate or assay.',
                log_ratio='ln(p_mutant / p_wt) when supplied; intentionally omitted for DNA; do not assume a hidden value was supplied',
                scope='MSA of the protein mutation sites; no DNA or small-molecule partner alignment is supplied',
            )
        elif kind == 'structure':
            fields = copy.deepcopy(GEOMETRY_FIELDS)
            profile = 'complex_protein_site_structure_legacy' if domain == 'protein_dna' else 'protein_ligand_structure_legacy'
            fields.update(definitions()[profile]['fields'])
            block['definition'].update(semantics_version=VERSION, fields=fields, interpretation=STRUCTURE_LIMITATIONS)
            if domain == 'protein_ligand':
                block['definition']['delta_sasa'] = fields['delta_sasa_partner_A2']
        else:
            raise ValueError('Unknown evidence kind: ' + kind)
        block['status'], block['coverage'] = coverage(block, kind)
    return out


def validate_repair(before, after):
    """Fail closed if a repair changes candidates, scores, features or base text."""
    assert set(before) == set(after)
    assert {k:v for k,v in before.items() if k != 'evidence'} == {k:v for k,v in after.items() if k != 'evidence'}
    for kind, old in before['evidence'].items():
        new = after['evidence'][kind]
        if old['status'] == 'not_requested':
            assert old == new
            continue
        assert {k:v for k,v in old.items() if k not in {'definition', 'status', 'coverage'}} == {k:v for k,v in new.items() if k not in {'definition', 'status', 'coverage'}}
        assert (new['status'], new['coverage']) == coverage(new, kind)
        assert new['definition']['semantics_version'] == VERSION
        assert new['definition']['interpretation'] == (MSA_LIMITATIONS if kind == 'msa' else STRUCTURE_LIMITATIONS)
        for variant in new['variants']:
            for site in variant['sites']:
                f = site['features']
                assert set(f) <= set(new['definition']['fields']), 'Undocumented feature: ' + str(set(f)-set(new['definition']['fields']))
                if kind == 'msa':
                    if f['status'] == 'ok':
                        assert all(finite(f.get(k)) and 0 < f[k] <= 1 for k in ('p_wt', 'p_mutant'))
                        if 'log_probability_ratio' in f:
                            assert math.isclose(f['log_probability_ratio'], math.log(f['p_mutant']/f['p_wt']), abs_tol=1e-9)
                    if after['task']['modality'] == 'protein_dna':
                        assert 'log_probability_ratio' not in f
                else:
                    if f.get('sasa_status') != 'ok':
                        assert f.get('sasa_bound_A2') is None and f.get('delta_sasa_partner_A2') is None
    json.dumps(after, allow_nan=False)
