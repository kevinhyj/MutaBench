"""Pro rerun keeps Flash prompt text and copies exact duplicate prompts."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools/offline_ablation'))
import protein_rna_contact_pro as pro


def test_duplicate_prompt_matches_only_the_same_text():
    assert pro.same_prompt('SEQUENCE', 'SEQUENCE')
    assert not pro.same_prompt('SEQUENCE\n\ncontact', 'SEQUENCE')


def test_model_id_is_the_listed_pro_preview():
    assert pro.MODEL == 'gemini-3.1-pro-preview'
    assert pro.MODEL in pro.ENDPOINT
