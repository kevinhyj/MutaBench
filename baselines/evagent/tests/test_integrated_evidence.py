"""Self-contained evidence tests; no remote data or model calls."""

import csv
import json
import shutil
from pathlib import Path
from unittest.mock import patch

from evagent.integrated.evidence import (
    EVIDENCE_MODES,
    EvidenceBundle,
    build_evidence_bundle,
    load_assay_metadata,
    parse_pdb_summary,
    render_evidence_context,
)

ASSETS = Path(__file__).resolve().parents[1] / "evagent" / "integrated"


def _reference(root, *, pdb="1EHZ", sequence="ACGU", rna_type="tRNA"):
    path = root / "data/RNAGym/fitness/reference_sheet_final.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "DMS_ID": "synthetic_assay", "RNA_TYPE": rna_type,
        "RNA_TYPE_GRANULAR": "synthetic fixture", "ASSAY_DESCRIPTION": "toy phenotype",
        "PDB_STRUCTURE": pdb, "RFAM_LINK": "", "RAW_CONSTRUCT_SEQ": sequence,
        "WT_DMS_SCORE": "991337", "Notes": "HELD_OUT_NOTE_SENTINEL",
    }
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    return path


def test_disabled_evidence_reads_no_files(tmp_path):
    with patch.object(Path, "open", side_effect=AssertionError("file opened")), \
         patch.object(Path, "read_bytes", side_effect=AssertionError("file read")), \
         patch.object(Path, "is_file", side_effect=AssertionError("file inspected")):
        bundle = build_evidence_bundle(assay="synthetic", rg_llm_root=tmp_path, mode="none")
        assert render_evidence_context(bundle) == ""
        assert render_evidence_context(bundle, modalities=()) == ""
    assert bundle["source_files"] == {}
    assert list(tmp_path.iterdir()) == []


def test_metadata_uses_allowlist_and_bundle_excludes_labels(tmp_path):
    _reference(tmp_path)
    metadata = load_assay_metadata(tmp_path, "synthetic_assay")
    assert set(metadata) == {
        "DMS_ID", "RNA_TYPE", "RNA_TYPE_GRANULAR", "ASSAY_DESCRIPTION",
        "PDB_STRUCTURE", "RFAM_LINK",
    }
    bundle = build_evidence_bundle(assay="synthetic_assay", rg_llm_root=tmp_path)
    assert isinstance(bundle, EvidenceBundle)
    assert bundle["representation"] == "frozen_text"
    for forbidden in ("991337", "HELD_OUT_NOTE_SENTINEL", "WT_DMS_SCORE", "RAW_CONSTRUCT_SEQ"):
        assert forbidden not in json.dumps(bundle)
        assert forbidden not in render_evidence_context(bundle)
    exported = bundle.as_dict()
    exported["assay_context"]["rna_type"] = "changed"
    assert bundle["assay_context"]["rna_type"] == "tRNA"


def test_bundled_structure_retains_source_hash_and_explicit_mapping(tmp_path):
    _reference(tmp_path)
    first = build_evidence_bundle(assay="synthetic_assay", rg_llm_root=tmp_path)
    second = build_evidence_bundle(assay="synthetic_assay", rg_llm_root=tmp_path)
    assert first["bundle_sha256"] == second["bundle_sha256"]
    assert first["structure"]["pdb_id"] == "1EHZ"
    assert not first["structure"]["construct_mapping"]["position_transfer_allowed"]
    summary = parse_pdb_summary(ASSETS / "knowledge/structures/fitness_assays_3D_structures/1ehz.pdb")
    assert summary["sha256"] == "3a43aefed0b6d5d8b4be47c44470e61258d0e90b5ff608f132a2c9731c9df41f"
    _reference(tmp_path, pdb="")
    missing = build_evidence_bundle(assay="synthetic_assay", rg_llm_root=tmp_path)
    assert missing["structure"] is None
    assert "No explicit assay-to-PDB" in " ".join(missing["unavailable_evidence"])


def test_missing_assets_are_reported_and_changed_asset_invalidates_hash(tmp_path):
    data, assets = tmp_path / "rg", tmp_path / "assets"
    _reference(data, pdb="9ZZZ")
    missing = build_evidence_bundle(assay="synthetic_assay", rg_llm_root=data, project_root=assets)
    assert missing["structure"] is None
    assert missing["fitness_skill"] is None
    assert missing["literature"] == []
    for expected in ("literature pack", "fitness skill", "source manifest", "PDB 9ZZZ"):
        assert expected in render_evidence_context(missing)
    shutil.copytree(ASSETS / "knowledge", assets / "knowledge")
    first = build_evidence_bundle(assay="synthetic_assay", rg_llm_root=data, project_root=assets)
    pack_path = assets / "knowledge/literature/fitness_evidence_v1.json"
    pack = json.loads(pack_path.read_text())
    pack["items"][0]["agent_guidance"] = "Keep all outputs reproducible."
    pack["items"][0]["labels"] = "HELD_OUT_LABEL_SENTINEL"
    pack_path.write_text(json.dumps(pack))
    second = build_evidence_bundle(assay="synthetic_assay", rg_llm_root=data, project_root=assets)
    assert second["bundle_sha256"] != first["bundle_sha256"]
    assert "Keep all outputs reproducible." in render_evidence_context(second)
    assert "HELD_OUT_LABEL_SENTINEL" not in json.dumps(second)


def test_ablation_renderer_keeps_modalities_separate(tmp_path):
    _reference(tmp_path)
    bundle = build_evidence_bundle(assay="synthetic_assay", rg_llm_root=tmp_path)
    structure = render_evidence_context(bundle, modalities=EVIDENCE_MODES["structure"])
    literature = render_evidence_context(bundle, modalities=EVIDENCE_MODES["literature"])
    skill = render_evidence_context(bundle, modalities=EVIDENCE_MODES["skill"])
    assert "1EHZ" in structure
    assert "General literature guidance" not in structure
    assert "1EHZ" not in literature
    assert "dms-map-seq-2017" in literature
    assert "ViennaRNA ensemble features unavailable" not in literature
    assert "1EHZ" not in skill
    assert "dms-map-seq-2017" not in skill
    assert "Rfam family mapping" not in skill


def test_literature_only_does_not_load_structures_or_require_sequence(tmp_path):
    _reference(tmp_path, sequence="", rna_type="mRNA-coding")
    with patch("evagent.integrated.evidence.parse_pdb_summary", side_effect=AssertionError("structure read")):
        bundle = build_evidence_bundle(assay="synthetic_assay", rg_llm_root=tmp_path, mode="literature")
    ids = {item["id"] for item in bundle["literature"]}
    assert "dms-overview-2014" in ids
    assert "dms-map-seq-2017" not in ids
    assert bundle["structure"] is None
    assert bundle["fitness_skill"] is None
    assert "structure_manifest" not in bundle["source_files"]


def test_structure_without_public_construct_reports_missing_mapping(tmp_path):
    _reference(tmp_path, sequence="")
    bundle = build_evidence_bundle(assay="synthetic_assay", rg_llm_root=tmp_path, mode="structure")
    assert bundle["structure"]["construct_mapping"]["status"] == "construct_unavailable"
    assert not bundle["structure"]["construct_mapping"]["position_transfer_allowed"]
    assert "Public construct sequence unavailable" in render_evidence_context(bundle)
