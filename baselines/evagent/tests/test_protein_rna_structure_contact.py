"""Contact rows keep a boolean and moieties, never a distance or a zero count."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools/offline_ablation'))
import protein_rna_structure_contact as contact


def site(**overrides):
    row = dict(site_id='site_a', interface_geometry_status='computed',
               min_partner_heavy_distance_A=6.8, partner_contact_count_5A=0,
               delta_sasa_partner_A2=0.0, rna_moiety_status='complete',
               rna_contact_count_5A_by_moiety={'base': 0, 'ribose': 0, 'phosphate': 0})
    row.update(overrides)
    return row


def test_negative_contact_is_a_boolean_without_measurements():
    row = contact.contact_row(site())
    assert row == dict(site_id='site_a', direct_rna_contact=False)


def test_positive_contact_lists_only_hit_moieties():
    row = contact.contact_row(site(partner_contact_count_5A=2, min_partner_heavy_distance_A=2.8,
                                   delta_sasa_partner_A2=29.0,
                                   rna_contact_count_5A_by_moiety={'base': 1, 'ribose': 1, 'phosphate': 0}))
    assert row['direct_rna_contact'] is True
    assert row['rna_contact_moieties'] == ['base', 'ribose']
    assert 'delta_sasa_partner_A2' not in row
    assert 'partner_contact_count_5A' not in row


def test_unmatched_prompt_is_unchanged():
    assert contact.render('SEQUENCE', []) == 'SEQUENCE'


def test_prompt_rejects_a_distance_field():
    try:
        contact.render('SEQUENCE', [dict(site_id='site_a', min_partner_heavy_distance_A=2.8)])
    except ValueError:
        return
    raise AssertionError('distance entered the contact prompt')


def test_rendered_negative_contact_has_no_zero():
    text = contact.render('SEQUENCE', [contact.contact_row(site())])
    body = text.split(contact.GUIDANCE, 1)[1]
    assert json.loads(body) == [dict(site_id='site_a', direct_rna_contact=False)]
    assert '0' not in body
