"""Lean structure rows keep real zeros and omit unmatched moiety lists."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools/offline_ablation'))
import protein_rna_structure_lean as lean


def site(**overrides):
    row = dict(site_id='site_a', interface_geometry_status='computed', sasa_status='computed',
               min_partner_heavy_distance_A=12.0, partner_contact_count_5A=0,
               delta_sasa_partner_A2=0.0, rna_moiety_status='complete',
               rna_contact_count_5A_by_moiety={'base': 0, 'ribose': 0, 'phosphate': 0})
    row.update(overrides)
    return row


def test_zero_contact_keeps_the_count_and_omits_moieties():
    row = lean.lean_row(site())
    assert row['partner_contact_count_5A'] == 0
    assert row['delta_sasa_partner_A2'] == 0.0
    assert 'rna_contact_moieties' not in row


def test_positive_contact_lists_only_hit_moieties():
    row = lean.lean_row(site(partner_contact_count_5A=1,
                             rna_contact_count_5A_by_moiety={'base': 1, 'ribose': 1, 'phosphate': 0}))
    assert row['rna_contact_moieties'] == ['base', 'ribose']


def test_unmatched_prompt_is_unchanged():
    assert lean.prompt_text('MSA only', {'sites': []}, {'candidates': []}, [], False) == 'MSA only'


def test_uncomputed_site_is_rejected():
    try:
        lean.lean_row(site(interface_geometry_status='unavailable_scope'))
    except ValueError:
        return
    raise AssertionError('uncomputed geometry was accepted')
