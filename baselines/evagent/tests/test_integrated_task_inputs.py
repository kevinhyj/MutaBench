"""Frozen input/type boundaries using synthetic fixtures and offline scoring."""

import hashlib
import json
from collections import Counter
from pathlib import Path

import pytest

from evagent.integrated.tasks import evaluate_result, load_task


def input_files(tmp_path, *, with_summary=False):
    sample = {
        "assay": "synthetic_inputs", "size": 3,
        "variants": [
            {"variant_id": f"M{index:02d}", "id": f"synthetic-{index}", "seq": "ACGU", "score": index}
            for index in range(1, 4)
        ],
    }
    context = {
        "assay": sample["assay"], "wt_sequence": "ACGU",
        "metadata": {"DMS_ID": sample["assay"], "ASSAY_DESCRIPTION": "Synthetic fixture", "YEAR": 2026},
    }
    config = {"sample_file": str(tmp_path / "sample.json"), "context_file": str(tmp_path / "context.json")}
    Path(config["sample_file"]).write_text(json.dumps(sample))
    Path(config["context_file"]).write_text(json.dumps(context))
    if with_summary:
        summary = {"status": "complete", "data": {
            "assay_id": sample["assay"], "assay_summary": "Frozen summary sentinel",
            "property_direction": "higher", "functional_hypotheses": ["Synthetic hypothesis"],
            "sequence_features": [],
        }}
        config["assay_memory_file"] = str(tmp_path / "summary.json")
        Path(config["assay_memory_file"]).write_text(json.dumps(summary))
    return config, sample, context


def test_input_hashes_and_values_use_one_read_per_snapshot(tmp_path, monkeypatch):
    config, sample, context = input_files(tmp_path, with_summary=True)
    payloads = {Path(path): Path(path).read_bytes() for path in config.values()}
    counts = Counter()
    original_read = Path.read_bytes

    def read_bytes(path):
        if path not in payloads:
            return original_read(path)
        counts[path] += 1
        if counts[path] > 1:
            raise AssertionError(f"Snapshot was read twice: {path}")
        return payloads[path]

    monkeypatch.setattr(Path, "read_bytes", read_bytes)
    task = load_task(config)
    assert all(count == 1 for count in counts.values())
    assert len(counts) == 3
    for kind, name in (("sample", "sample_file"), ("context", "context_file"), ("assay_memory", "assay_memory_file")):
        assert task.input_hashes[kind] == hashlib.sha256(payloads[Path(config[name])]).hexdigest()
    assert task.assay_memory["assay_summary"] == "Frozen summary sentinel"


def test_changed_file_after_read_cannot_change_snapshot_hash(tmp_path, monkeypatch):
    config, sample, context = input_files(tmp_path)
    sample_path = Path(config["sample_file"])
    before = sample_path.read_bytes()
    original_read = Path.read_bytes

    def read_bytes(path):
        payload = original_read(path)
        if path == sample_path:
            changed = json.loads(payload)
            changed["variants"][0]["seq"] = "UUUU"
            path.write_text(json.dumps(changed))
        return payload

    monkeypatch.setattr(Path, "read_bytes", read_bytes)
    task = load_task(config)
    assert task.variants[0]["sequence"] == "ACGU"
    assert task.input_hashes["sample"] == hashlib.sha256(before).hexdigest()
    assert original_read(sample_path) != before


@pytest.mark.parametrize("field,value", [
    ("variant_id", 1), ("id", {"experimental_score": 9123}),
    ("seq", {"experimental_score": 9123}), ("seq", ["ACGU"]),
    ("seq", None), ("seq", "AC GU"),
])
def test_nonstring_or_broken_candidate_fields_rejected(tmp_path, field, value):
    config, sample, context = input_files(tmp_path)
    sample["variants"][0][field] = value
    Path(config["sample_file"]).write_text(json.dumps(sample))
    with pytest.raises(ValueError, match="Candidate"):
        load_task(config)


@pytest.mark.parametrize("value", [None, 123, {"score": 9123}, ["ACGU"], "AC GU"])
def test_nonstring_or_broken_wild_type_rejected(tmp_path, value):
    config, sample, context = input_files(tmp_path)
    context["wt_sequence"] = value
    Path(config["context_file"]).write_text(json.dumps(context))
    with pytest.raises(ValueError, match="WT sequence"):
        load_task(config)


@pytest.mark.parametrize("value", [{"experimental_score": 9123}, ["label"], float("nan"), float("inf")])
def test_metadata_fields_must_be_scalar_and_finite(tmp_path, value):
    config, sample, context = input_files(tmp_path)
    context["metadata"]["ASSAY_DESCRIPTION"] = value
    Path(config["context_file"]).write_text(json.dumps(context))
    with pytest.raises(ValueError, match="metadata"):
        load_task(config)


def test_unselected_labels_do_not_enter_model_task(tmp_path):
    config, sample, context = input_files(tmp_path)
    sample["variants"][0]["experimental_score"] = "LABEL_SENTINEL"
    context["metadata"]["ground_truth"] = {"label": "LABEL_SENTINEL"}
    Path(config["sample_file"]).write_text(json.dumps(sample))
    Path(config["context_file"]).write_text(json.dumps(context))
    task = load_task(config)
    assert "LABEL_SENTINEL" not in task.case.agent_prompt
    assert "experimental_score" not in task.variants[0]
    assert task.case.ground_truth_scores is None


def test_numeric_aliases_rejected_before_protocol_rendering(tmp_path):
    config, sample, context = input_files(tmp_path)
    sample["variants"][0]["variant_id"] = "M1"
    sample["variants"][1]["variant_id"] = "M01"
    Path(config["sample_file"]).write_text(json.dumps(sample))
    with pytest.raises(ValueError, match="numeric identities"):
        load_task(config)


@pytest.mark.parametrize("field,value", [
    ("assay_summary", {"label": 9123}), ("property_direction", 1),
    ("functional_hypotheses", [{"experimental_score": 9123}]),
    ("sequence_features", "not-a-list"),
])
def test_frozen_summary_fields_are_typed(tmp_path, field, value):
    config, sample, context = input_files(tmp_path, with_summary=True)
    path = Path(config["assay_memory_file"])
    summary = json.loads(path.read_text())
    summary["data"][field] = value
    path.write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="Assay summary"):
        load_task(config)


def test_context_assay_identity_must_match(tmp_path):
    config, sample, context = input_files(tmp_path)
    context["metadata"]["DMS_ID"] = "different_assay"
    Path(config["context_file"]).write_text(json.dumps(context))
    with pytest.raises(ValueError, match="DMS_ID"):
        load_task(config)


def test_evaluator_hashes_and_scores_the_same_single_read(tmp_path, monkeypatch):
    config, sample, context = input_files(tmp_path)
    sample_path = Path(config["sample_file"])
    before = sample_path.read_bytes()
    expected_hash = hashlib.sha256(before).hexdigest()
    count = 0
    original_read = Path.read_bytes

    def read_bytes(path):
        nonlocal count
        if path != sample_path:
            return original_read(path)
        count += 1
        if count > 1:
            raise AssertionError("Evaluator reread the sample after hashing")
        changed = json.loads(before)
        changed["variants"][0]["score"] = 1000
        path.write_text(json.dumps(changed))
        return before

    monkeypatch.setattr(Path, "read_bytes", read_bytes)
    result = evaluate_result(sample_path, expected_hash, ["M01", "M02", "M03"],
                             {"status": "ok", "ranking": ["M03", "M02", "M01"]})
    assert result["status"] == "ok"
    assert result["spearman"] == pytest.approx(1.0)
    assert count == 1


def test_evaluator_rejects_changed_snapshot(tmp_path):
    config, sample, context = input_files(tmp_path)
    sample_path = Path(config["sample_file"])
    expected_hash = hashlib.sha256(sample_path.read_bytes()).hexdigest()
    sample["variants"][0]["score"] = 1000
    sample_path.write_text(json.dumps(sample))
    with pytest.raises(ValueError, match="Sample changed"):
        evaluate_result(sample_path, expected_hash, ["M01", "M02", "M03"],
                        {"status": "ok", "ranking": ["M03", "M02", "M01"]})


def test_incomplete_prediction_does_not_open_label_file(monkeypatch):
    def unexpected_read(path):
        raise AssertionError("Incomplete outputs must not open the evaluator sample")
    monkeypatch.setattr(Path, "read_bytes", unexpected_read)
    assert evaluate_result(Path("unused.json"), "unused", ["M01", "M02"],
                           {"status": "invalid_output", "ranking": None}) == {
        "status": "not_evaluated_incomplete"
    }
