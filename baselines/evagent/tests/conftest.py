"""Mark tests requiring deliberately unbundled research assets explicitly."""
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ASSETS = {
    "test_all_curated_passages_are_literal_and_audited": "work/multiagent_v4/curation/reviewed_passages.json",
    "test_denied_source_cannot_be_admitted": "work/multiagent_v4/curation/reviewed_passages.json",
    "test_bundled_structure_retains_source_hash_and_explicit_mapping":
        "evagent/integrated/knowledge/structures/fitness_assays_3D_structures/1ehz.pdb",
    "test_structure_without_public_construct_reports_missing_mapping":
        "evagent/integrated/knowledge/structures/fitness_assays_3D_structures/1ehz.pdb",
}


def pytest_collection_modifyitems(items):
    for item in items:
        asset = ASSETS.get(item.originalname or item.name)
        if asset and not (ROOT / asset).is_file():
            item.add_marker(pytest.mark.skip(reason="Separate research asset is not bundled in this code release: " + asset))
        if item.name == "test_real_5HO4_image_matches_frozen_independent_diagnostic" and not (
            ROOT.parent / "data/final_mutation_data/nabe/source/structures/nabe_pdbs/5HO4.pdb"
        ).is_file():
            item.add_marker(pytest.mark.skip(reason="Original 5HO4 structure is part of the separate data release"))
