"""Local structure rows keep assay-matched zeros and drop unmatched non-contacts."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools/offline_ablation'))
import protein_rna_structure_local as local


def measured(**overrides):
    row = dict(site_id='site_a', interface_geometry_status='computed', sasa_status='computed',
               min_partner_heavy_distance_A=12.0, partner_contact_count_5A=0,
               delta_sasa_partner_A2=0.0, rna_moiety_status='complete',
               rna_contact_count_5A_by_moiety={'base': 0, 'ribose': 0, 'phosphate': 0})
    row.update(overrides)
    return row


def test_assay_matched_zero_stays_in_the_prompt_row():
    tier, rows, omitted = local.select_rows('2ERR', ['site_a'], [measured()], {})
    assert tier == 'assay_matched'
    assert omitted == []
    assert rows[0]['partner_contact_count_5A'] == 0
    assert 'rna_contact_moieties' not in rows[0]


def test_fragment_drops_a_real_zero_and_lists_only_hit_moieties():
    far = measured(site_id='site_far')
    near = measured(site_id='site_near', min_partner_heavy_distance_A=3.2,
                    partner_contact_count_5A=1, delta_sasa_partner_A2=8.0,
                    rna_contact_count_5A_by_moiety={'base': 1, 'ribose': 0, 'phosphate': 0})
    tier, rows, omitted = local.select_rows('4I67', ['site_far', 'site_near'], [],
                                             {'site_far': far, 'site_near': near})
    assert tier == 'observed_contact'
    assert omitted == ['site_far']
    assert rows[0]['site_id'] == 'site_near'
    assert rows[0]['rna_contact_moieties'] == ['base']


def test_unpaired_trna_contact_is_omitted():
    wrong = dict(rna_contacts=[dict(rna_instance_id='D@deposited', rna_site_id='d1',
                                    min_distance_A=2.5, moieties=['base'])],
                 delta_sasa_partner_A2=9.0, rna_moiety_status='complete')
    paired = dict(rna_contacts=[dict(rna_instance_id='C@deposited', rna_site_id='c1',
                                     min_distance_A=3.1, moieties=['ribose'])],
                  delta_sasa_partner_A2=4.0, rna_moiety_status='complete')
    tier, rows, omitted = local.select_rows('2ZZN', ['site_wrong', 'site_paired'], [],
                                             {'site_wrong': wrong, 'site_paired': paired})
    assert tier == 'paired_trna_c'
    assert omitted == ['site_wrong']
    assert rows[0]['site_id'] == 'site_paired'
    assert rows[0]['partner_contact_count_5A'] == 1
    assert 'D@' not in json.dumps(rows)


def test_mixed_trna_contact_drops_the_contaminated_area():
    obs = dict(rna_contacts=[
        dict(rna_instance_id='C@deposited', rna_site_id='c1', min_distance_A=3.0, moieties=['base']),
        dict(rna_instance_id='D@deposited', rna_site_id='d1', min_distance_A=4.0, moieties=['phosphate'])],
        delta_sasa_partner_A2=20.0, rna_moiety_status='complete')
    row = local._paired_row('site_mixed', obs)
    assert 'delta_sasa_partner_A2' not in row
    assert row['rna_contact_moieties'] == ['base']
    assert row['partner_contact_count_5A'] == 1


def test_related_structure_is_a_boolean_without_distances():
    far = measured(site_id='site_far')
    near = measured(site_id='site_near', partner_contact_count_5A=2)
    tier, rows, omitted = local.select_rows('1B23', ['site_far', 'site_near'], [],
                                             {'site_far': far, 'site_near': near})
    assert tier == 'related_structure'
    assert omitted == ['site_far']
    assert rows == [dict(site_id='site_near', structure_id='1B23', related_structure_contact=True)]
    text = local.render('SEQUENCE', local.guidance_for(tier), rows)
    assert text.startswith('SEQUENCE\n\n')
    assert 'min_partner_heavy_distance_A' not in text
    assert 'unavailable' not in text


def test_no_rows_leaves_the_baseline_text_unchanged():
    assert local.render('MSA only', local.LOCAL_GUIDANCE, []) == 'MSA only'


def test_null_marker_is_rejected():
    try:
        local.render('BASE', local.LOCAL_GUIDANCE, [{'site_id': 'site_a', 'value': None}])
    except ValueError:
        return
    raise AssertionError('null entered a structure prompt')
