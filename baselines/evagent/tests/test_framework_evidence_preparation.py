"""Admission and cohort preparation contracts independent of provider calls."""
import copy
import importlib.util
import json
from pathlib import Path

import pytest

from evagent.integrated.gateway import digest
from evagent.integrated.tasks import file_hash

SPEC = importlib.util.spec_from_file_location(
    "framework_preparation", Path(__file__).parents[1] / "work/multiagent_v4/prepare_evidence.py")
PREP = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PREP)


def test_all_curated_passages_are_literal_and_audited():
    root = Path(__file__).parents[1]
    value = json.loads((root / "work/multiagent_v4/curation/reviewed_passages.json").read_text())
    count = 0
    for family in value["families"]:
        admitted, raw, audit, reason = PREP.admit(family, value, [], [])
        count += len(admitted)
        if not admitted:
            assert reason
        for passage in admitted:
            original = next(r for r in raw if r["doi"] == passage["doi"])
            assert original["text"][passage["source_start"]:passage["source_end"]] == passage["text"]
            assert passage["source_sha256"] == digest(original["text"])
            assert passage["limitations"] and passage["review"]["reason"]
    assert count == 11


def test_denied_source_cannot_be_admitted():
    root = Path(__file__).parents[1]
    value = json.loads((root / "work/multiagent_v4/curation/reviewed_passages.json").read_text())
    family = "mammalian_CPEB3"
    doi = value["families"][family]["documents"][0]["record"]["doi"]
    with pytest.raises(ValueError, match="failed admission"):
        PREP.admit(family, value, [doi], [])


def test_alignment_resolution_requires_reviewed_origin_and_exact_hash(tmp_path):
    asset = tmp_path / "legacy/raw/msa.sto"
    asset.parent.mkdir(parents=True)
    asset.write_text("# STOCKHOLM 1.0\nWT_query ACGU\nh1 ACGA\nh2 ACUC\n//\n")
    snapshot = {"sources": {"msa": {"status": "available", "provenance": [
        {"origin": "natural_homologs", "reviewed": True, "source_urls": ["https://rfam.org/"]},
        {"artifact": "legacy/raw/msa.sto", "sha256": file_hash(asset)}]}}}
    result = PREP.find_alignment(snapshot, tmp_path / "snapshot.json", "task", tmp_path)
    assert result["sha256"] == file_hash(asset)
    altered = copy.deepcopy(snapshot)
    altered["sources"]["msa"]["provenance"][0]["origin"] = "candidate_library"
    with pytest.raises(ValueError, match="natural origin"):
        PREP.find_alignment(altered, tmp_path / "snapshot.json", "task", tmp_path)
    asset.write_text("changed")
    with pytest.raises(ValueError, match="hash changed"):
        PREP.find_alignment(snapshot, tmp_path / "snapshot.json", "task", tmp_path)


def test_task_context_does_not_need_or_accept_label_access():
    class PublicTask:
        assay = "safe_assay"
        metadata = {"ASSAY_DESCRIPTION": "Assayed RNA self-cleavage."}
        input_hashes = {"context": "frozen-context"}
    value = PREP.assay_context(PublicTask(), "synthetic_family", [])
    assert value["data"]["assay_summary"].startswith("Assayed RNA self-cleavage.")
    assert value["data"]["property_direction"]
    assert "Conditions not specified" in value["data"]["assay_summary"]
    assert "No independently reviewed" in value["data"]["functional_hypotheses"][0]
