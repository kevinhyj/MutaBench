import itertools
import json
from pathlib import Path

import pytest

from evagent.fewshot.data import (
    NotEligibleError,
    freeze_episode,
    load_landscape,
    sha256_file,
    write_episode,
)


def metadata(**updates):
    result = {
        "assay_id": "test_rna", "domain": "rna", "wt_sequence": "AAAAAA",
        "assay_description": "Synthetic offline assay", "endpoint": "activity",
        "readout": "Measured activity", "higher_is_better": True,
    }
    result.update(updates)
    return result


def records():
    return [{"sequence": "".join(seq), "fitness": i / 17.0}
            for i, seq in enumerate(itertools.product("AC", repeat=6))]


def load(tmp_path, rows=None, meta=None, name="source.json"):
    source = tmp_path / name
    source.write_text(json.dumps(records() if rows is None else rows))
    return load_landscape(source, meta or metadata())


def test_nested_disjoint_queries_and_supports(tmp_path):
    landscape = load(tmp_path)
    full = freeze_episode(landscape, "random", 1, query_size=20)
    fewer = freeze_episode(landscape, "random", 1, query_size=20, shots=(0, 4, 8, 16))
    assert full["episode"]["query"] == fewer["episode"]["query"]
    assert full["support"]["examples"][:16] == fewer["support"]["examples"]
    assert len(full["support"]["examples"]) == 32
    query = full["episode"]["query"]
    support = full["support"]["examples"]
    assert {r["sequence"] for r in query}.isdisjoint(r["sequence"] for r in support)
    assert [r["variant_id"] for r in query] == [f"M{i:03d}" for i in range(1, 21)]
    assert [r["variant_id"] for r in support] == [f"S{i:03d}" for i in range(1, 33)]
    assert freeze_episode(landscape, "random", 2, query_size=20)["episode"]["query"] != query


def test_sampling_independent_of_fitness_and_input_row_order(tmp_path):
    original = records()
    a = freeze_episode(load(tmp_path, original), "random", 4, query_size=20)
    changed = [dict(row, fitness=1000 - row["fitness"] * 3) for row in reversed(original)]
    b = freeze_episode(load(tmp_path, changed, name="changed.json"), "random", 4, query_size=20)
    assert a["episode"] == b["episode"]
    assert [{k: v for k, v in row.items() if k != "fitness"} for row in a["support"]["examples"]] == [
        {k: v for k, v in row.items() if k != "fitness"} for row in b["support"]["examples"]
    ]
    assert a["evaluation_labels"]["labels"] != b["evaluation_labels"]["labels"]


def test_query_labels_and_source_paths_are_not_in_inference_artifacts(tmp_path):
    meta = metadata(label_path="/secret/labels.json", best_mutations=["A1C"], DMS_score=987654321)
    landscape = load(tmp_path, meta=meta)
    a = freeze_episode(landscape, "random", 1, query_size=20)
    query_sequences = {row["sequence"] for row in a["episode"]["query"]}
    changed = [dict(row, fitness=1234567) if row["sequence"] in query_sequences else row for row in records()]
    b = freeze_episode(load(tmp_path, changed, name="altered.json"), "random", 1, query_size=20)
    assert a["episode"] == b["episode"]
    assert a["support"] == b["support"]
    assert a["manifest"]["files"]["episode.json"] == b["manifest"]["files"]["episode.json"]
    assert a["manifest"]["files"]["support.json"] == b["manifest"]["files"]["support.json"]
    assert "fitness" not in json.dumps(a["episode"])
    for artifact in (a["episode"], a["support"]):
        encoded = json.dumps(artifact)
        assert "label_path" not in encoded and "best_mutations" not in encoded
        assert str(tmp_path) not in encoded and "source_sha256" not in encoded
    assert len(a["evaluation_labels"]["labels"]) == 20


def test_order_extrapolation_and_insufficient_pool(tmp_path):
    landscape = load(tmp_path)
    frozen = freeze_episode(landscape, "low_order_to_high_order", 1, query_size=20, shots=(0, 4, 8, 16))
    assert all(row["mutation_order"] in (1, 2) for row in frozen["support"]["examples"])
    assert all(row["mutation_order"] >= 3 for row in frozen["episode"]["query"])
    with pytest.raises(NotEligibleError, match="has 21 and 42"):
        freeze_episode(landscape, "low_order_to_high_order", 1, query_size=20)
    with pytest.raises(NotEligibleError, match="requires 282"):
        freeze_episode(landscape, "random", 1)


def test_no_wt_random_and_explicit_order_extrapolation(tmp_path):
    landscape = load(tmp_path, meta=metadata(wt_sequence=None, higher_is_better=False))
    frozen = freeze_episode(landscape, "random", 1, query_size=20)
    assert frozen["episode"]["wt_sequence"] is None
    assert frozen["evaluation_labels"]["higher_is_better"] is False
    assert all(row["mutation_order"] is None for row in frozen["episode"]["query"])
    with pytest.raises(NotEligibleError, match="has 0 and 0"):
        freeze_episode(landscape, "low_order_to_high_order", 1, query_size=20)
    explicit = [dict(row, mutation_order=row["sequence"].count("C")) for row in records()]
    known = load(tmp_path, explicit, metadata(wt_sequence=None), name="explicit.json")
    result = freeze_episode(known, "low_order_to_high_order", 2, query_size=20, shots=(0, 4, 8, 16))
    assert all(row["mutation_order"] <= 2 for row in result["support"]["examples"])


def test_deduplication_and_nonfinite_audit(tmp_path):
    rows = records() + [dict(records()[1]), {"sequence": "GGGGGG", "fitness": float("nan")},
                        {"sequence": "UUUUUU", "fitness": float("inf")},
                        {"sequence": "XXXXXX", "fitness": 1}]
    landscape = load(tmp_path, rows)
    assert len(landscape.records) == 63
    assert landscape.provenance["excluded_rows_by_reason"] == {
        "duplicate_sequence": 1, "reference_sequence": 1,
        "nonfinite_fitness": 2, "invalid_sequence_alphabet": 1,
    }
    assert sum(landscape.provenance["excluded_rows_by_reason"].values()) + len(landscape.records) == len(rows)
    with pytest.raises(ValueError, match="conflicting fitness"):
        load(tmp_path, records() + [dict(records()[1], fitness=500)])


def test_source_csv_mapping_and_raw_direction(tmp_path):
    source = tmp_path / "source.csv"
    source.write_text("mutated_sequence,DMS_score,mutant\nAC,5.5,A2C\nAG,-2,A2G\n")
    landscape = load_landscape(source, metadata(wt_sequence="AA", higher_is_better=False),
                               {"sequence": "mutated_sequence", "fitness": "DMS_score", "mutation": "mutant"})
    assert {r["fitness"] for r in landscape.records} == {5.5, -2}
    assert landscape.records[0]["mutation"] == "A2C"
    assert landscape.provenance["source_sha256"] == sha256_file(source)


def test_frozen_write_idempotent_and_immutable(tmp_path):
    frozen = freeze_episode(load(tmp_path), "random", 1, query_size=20)
    out = tmp_path / "episode"
    paths = write_episode(frozen, out)
    assert write_episode(frozen, out) == paths
    for filename, expected in frozen["manifest"]["files"].items():
        assert sha256_file(out / filename) == expected
    assert Path(paths["evaluation_labels"]).stat().st_mode & 0o777 == 0o600
    assert Path(paths["support"]).stat().st_mode & 0o777 == 0o600
    frozen["support"]["examples"][0]["fitness"] += 1
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_episode(frozen, out)


@pytest.mark.parametrize("shots", [(4, 8), (0, 8, 4), (0, 4, 4), (0, True), (0, -1)])
def test_reject_invalid_shot_design(tmp_path, shots):
    with pytest.raises(ValueError, match="shots"):
        freeze_episode(load(tmp_path), "random", 1, query_size=20, shots=shots)


def test_reference_order_mismatch_is_audited(tmp_path):
    landscape = load(tmp_path, [{"sequence": "AAAAAC", "fitness": 1, "mutation_order": 4},
                                {"sequence": "AAAACC", "fitness": 2, "mutation_order": 2}])
    assert len(landscape.records) == 1
    assert landscape.provenance["excluded_rows_by_reason"] == {"mutation_order_reference_mismatch": 1}


def test_mutation_alias_dedup_is_row_order_invariant_without_wt(tmp_path):
    rows = [{"sequence": "AAAACC", "fitness": 1, "mutation": "second", "mutation_order": 2},
            {"sequence": "AAAACC", "fitness": 1, "mutation": "first", "mutation_order": 2}]
    a = load(tmp_path, rows, metadata(wt_sequence=None))
    b = load(tmp_path, list(reversed(rows)), metadata(wt_sequence=None), name="reversed.json")
    assert a.records == b.records
