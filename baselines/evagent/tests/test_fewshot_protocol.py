"""Scientific controls for fixed CR few-shot prompt construction (no network)."""

from __future__ import annotations

import copy
import json
import re

import pytest

from evagent.fewshot.protocol import (
    CONDITIONS,
    Condition,
    all_requests,
    make_request,
    measured_examples,
    validate_inputs,
)


def _sequence(number):
    alphabet = "ACGU"
    return "".join(alphabet[(number // 4**position) % 4] for position in range(8))


@pytest.fixture
def inputs():
    episode = {
        "episode_id": "test_assay__random__seed1",
        "assay_id": "test_assay",
        "domain": "rna",
        "wt_sequence": "UUUUUUUU",
        "assay_description": "RNA cleavage experiment at a fixed temperature.",
        "endpoint": "cleavage activity",
        "readout": "raw cleaved fraction",
        "higher_is_better": True,
        "query": [
            {"variant_id": f"M{i:03d}", "mutation": f"U1A:{i}", "sequence": _sequence(i)}
            for i in range(1, 61)
        ],
    }
    support = {
        "episode_id": episode["episode_id"],
        "examples": [
            {"variant_id": f"S{i:03d}", "mutation": f"U1C:{i}",
             "sequence": _sequence(i + 100), "fitness": float(i) + 0.125}
            for i in range(1, 33)
        ],
    }
    return episode, support


def _condition(name):
    return next(item for item in CONDITIONS if item.name == name)


def _rendered_examples(request):
    marker = "Reference examples (S IDs are context, never prediction targets):\n"
    if marker not in request["user"]:
        return []
    tail = request["user"].split(marker, 1)[1]
    _, encoded = tail.split("\n", 1)
    return json.JSONDecoder().raw_decode(encoded)[0]


def _without_support(user):
    start = user.find("\n\nReference examples (")
    if start < 0:
        start = user.find("\n\nNo measured reference examples are supplied.")
    assert start >= 0
    candidate = re.search(r"\*\*\d+ candidate mutants to score\*\*", user[start:])
    assert candidate
    end = start + candidate.start()
    return user[:start] + user[end:]


@pytest.mark.parametrize("shots", [0, 4, 8, 16, 32])
def test_every_chunk_contains_exact_numeric_support_prefix(inputs, shots):
    episode, support = inputs
    original = copy.deepcopy(inputs)
    for chunk_index in range(3):
        request = make_request(episode, support, _condition(f"numeric_{shots}"), chunk_index)
        examples = _rendered_examples(request)
        assert len(examples) == shots
        assert request["support_ids"] == [item["variant_id"] for item in support["examples"][:shots]]
        for actual, expected in zip(examples, support["examples"][:shots], strict=True):
            assert actual == {"variant_id": expected["variant_id"], "mutation": expected["mutation"],
                              "sequence": expected["sequence"], "measured_value": expected["fitness"]}
        assert request["expected_ids"] == [
            item["variant_id"] for item in episode["query"][chunk_index * 25:(chunk_index + 1) * 25]
        ]
        assert set(request["support_ids"]).isdisjoint(request["expected_ids"])
    assert inputs == original


def test_all_conditions_share_system_candidates_and_fixed_cr_instructions(inputs):
    episode, support = inputs
    for chunk in range(3):
        arms = [make_request(episode, support, condition, chunk) for condition in CONDITIONS]
        assert len({item["system"] for item in arms}) == 1
        assert len({_without_support(item["user"]) for item in arms}) == 1
        assert all(item["expected_ids"] == arms[0]["expected_ids"] for item in arms)
        assert all("[-1.0, 1.0]" in item["user"] for item in arms)
        assert all("no missing, duplicate, or extra IDs" in item["user"] for item in arms)


def test_unlabeled_control_contains_same_sequences_without_measurements(inputs):
    episode, support = inputs
    request = make_request(episode, support, _condition("unlabeled_16"), 0)
    examples = _rendered_examples(request)
    assert len(examples) == 16
    assert "no measurements are supplied" in request["user"]
    assert all(set(item) == {"variant_id", "mutation", "sequence"} for item in examples)
    assert [item["sequence"] for item in examples] == [item["sequence"] for item in support["examples"][:16]]
    assert all(str(item["fitness"]) not in request["user"] for item in support["examples"][:16])


def test_shuffle_is_fixed_across_chunks_reproducible_and_episode_seeded(inputs):
    episode, support = inputs
    condition = _condition("shuffled_16")
    first = make_request(episode, support, condition, 0)
    examples = _rendered_examples(first)
    original = [item["fitness"] for item in support["examples"][:16]]
    shuffled = [item["measured_value"] for item in examples]
    assert sorted(shuffled) == original
    assert shuffled != original
    assert examples == _rendered_examples(make_request(episode, support, condition, 1))
    assert first == make_request(episode, support, condition, 0)
    assert [item["sequence"] for item in examples] == [item["sequence"] for item in support["examples"][:16]]
    changed_episode, changed_support = copy.deepcopy(inputs)
    changed_episode["episode_id"] = "test_assay__random__seed2"
    changed_support["episode_id"] = changed_episode["episode_id"]
    assert shuffled != [item["measured_value"] for item in measured_examples(
        changed_episode, changed_support, condition
    )]


@pytest.mark.parametrize("higher_is_better", [True, False])
def test_ordinal_control_uses_support_only_average_ranks_and_explicit_direction(inputs, higher_is_better):
    episode, support = inputs
    episode["higher_is_better"] = higher_is_better
    support["examples"][1]["fitness"] = support["examples"][0]["fitness"]
    request = make_request(episode, support, _condition("ordinal_16"), 0)
    examples = _rendered_examples(request)
    assert len(examples) == 16
    assert all("measured_value" not in item for item in examples)
    assert examples[0]["measured_rank"] == examples[1]["measured_rank"] == (15.5 if higher_is_better else 1.5)
    assert examples[-1]["measured_rank"] == (1 if higher_is_better else 16)
    assert "1 is best" in request["user"]
    support["examples"][-1]["fitness"] = -999_999
    assert _rendered_examples(make_request(episode, support, _condition("ordinal_16"), 0)) == examples


@pytest.mark.parametrize("condition", CONDITIONS, ids=lambda condition: condition.name)
def test_injected_query_labels_and_unknown_metadata_cannot_change_prompt(inputs, condition):
    episode, support = inputs
    expected = make_request(episode, support, condition, 1)
    # Poison values deliberately cannot be serialized by json.dumps. Safe
    # projection must exclude them rather than merely hide column names.
    for row in episode["query"]:
        row["fitness"] = object()
        row["labels"] = {"hidden": "secret held-out measurement"}
        row["private_score"] = 123456789
    episode["evaluation_labels"] = object()
    episode["hidden_labels_path"] = "/private/not-for-inference/labels.json"
    support["hidden_labels"] = object()
    for row in support["examples"]:
        row["query_fitness"] = object()
    assert make_request(episode, support, condition, 1) == expected


def test_missing_wt_and_mutation_annotations_do_not_invent_a_reference(inputs):
    episode, support = inputs
    episode["wt_sequence"] = None
    for row in episode["query"] + support["examples"]:
        row["mutation"] = ""
    request = make_request(episode, support, _condition("numeric_16"), 0)
    assert "Unavailable; do not invent a WT." in request["user"]
    assert "no WT is supplied" in request["user"]
    assert "0 is approximately neutral" not in request["user"]
    assert "relative to WT" not in request["user"]
    assert len(_rendered_examples(request)) == 16


@pytest.mark.parametrize("overlap", ["variant_id", "sequence"])
def test_cross_split_duplicate_ids_or_sequences_are_rejected(inputs, overlap):
    episode, support = inputs
    support["examples"][0][overlap] = episode["query"][0][overlap]
    with pytest.raises(ValueError, match="Duplicate|overlapping"):
        validate_inputs(episode, support)


@pytest.mark.parametrize("side", ["query", "support"])
@pytest.mark.parametrize("duplicate", ["variant_id", "sequence"])
def test_duplicates_within_either_split_are_rejected(inputs, side, duplicate):
    episode, support = inputs
    rows = episode["query"] if side == "query" else support["examples"]
    rows[1][duplicate] = rows[0][duplicate]
    with pytest.raises(ValueError, match="Duplicate|overlapping"):
        validate_inputs(episode, support)


@pytest.mark.parametrize("bad_value", [None, True, float("nan"), float("inf"), "0.5"])
def test_support_measurement_must_be_explicit_finite_number(inputs, bad_value):
    episode, support = inputs
    support["examples"][0]["fitness"] = bad_value
    with pytest.raises(ValueError, match="measurement"):
        validate_inputs(episode, support)


@pytest.mark.parametrize("side,bad_id", [("query", "Mbad"), ("query", "S001"), ("support", "Sbad"), ("support", "M999")])
def test_ids_must_be_numeric_namespaces_compatible_with_cr_parser(inputs, side, bad_id):
    episode, support = inputs
    rows = episode["query"] if side == "query" else support["examples"]
    rows[0]["variant_id"] = bad_id
    with pytest.raises(ValueError):
        validate_inputs(episode, support)


def test_episode_identity_insufficient_support_and_unknown_arm_are_rejected(inputs):
    episode, support = inputs
    wrong = dict(support, episode_id="another_episode")
    with pytest.raises(ValueError, match="mismatch"):
        make_request(episode, wrong, _condition("numeric_16"), 0)
    short = dict(support, examples=support["examples"][:15])
    with pytest.raises(ValueError, match="Insufficient"):
        make_request(episode, short, _condition("numeric_16"), 0)
    with pytest.raises(ValueError, match="Unknown"):
        make_request(episode, support, Condition("new", 16, "numeric"), 0)


def test_frozen_chunks_schedule_and_request_hash_are_deterministic(inputs):
    episode, support = inputs
    requests = all_requests(episode, support)
    assert len(requests) == 3 * len(CONDITIONS)
    assert requests == all_requests(episode, support)
    assert len({item["request_hash"] for item in requests}) == len(requests)
    for chunk_index in range(3):
        chunk = [item for item in requests if item["chunk_index"] == chunk_index]
        assert {item["condition"]["name"] for item in chunk} == {item.name for item in CONDITIONS}
    with pytest.raises(ValueError, match="B=25"):
        make_request(episode, support, _condition("numeric_0"), 0, chunk_size=10)
    for invalid_chunk in (-1, 3):
        with pytest.raises(ValueError, match="Invalid chunk"):
            make_request(episode, support, _condition("numeric_0"), invalid_chunk)
