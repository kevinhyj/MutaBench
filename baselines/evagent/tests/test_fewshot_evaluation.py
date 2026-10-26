from __future__ import annotations

import math

import numpy as np
import pytest

from evagent.fewshot.evaluation import (
    aggregate_metrics,
    evaluate_predictions,
    normalized_sequence_distance,
    ridge_predict,
    sequence_nearest_neighbor,
)


def support():
    return [
        {"variant_id": "s0", "sequence": "AA", "mutation": "", "fitness": 0.0},
        {"variant_id": "s1", "sequence": "CC", "mutation": "", "fitness": 2.0},
    ]


@pytest.mark.parametrize("predict", [sequence_nearest_neighbor, ridge_predict])
def test_baselines_ignore_query_labels_and_orient_scores(predict):
    query = [
        {"variant_id": "q0", "sequence": "AA", "fitness": object()},
        {"variant_id": "q1", "sequence": "CC", "fitness": float("nan")},
    ]
    positive = predict(support(), query)
    negative = predict(support(), query, higher_is_better=False)
    assert positive["q0"] < positive["q1"]
    assert negative == pytest.approx({key: -value for key, value in positive.items()})
    assert predict(support(), list(reversed(query))) == pytest.approx(positive)


def test_nearest_neighbor_averages_ties_without_using_order():
    query = [{"variant_id": "q", "sequence": "AC"}]
    assert sequence_nearest_neighbor(support(), query) == {"q": 1.0}
    assert sequence_nearest_neighbor(list(reversed(support())), query) == {"q": 1.0}


@pytest.mark.parametrize("predict", [sequence_nearest_neighbor, ridge_predict])
def test_duplicate_sequences_with_distinct_ids_do_not_break_baselines(predict):
    # Episode preparation owns biological deduplication. Standalone predictors
    # still handle a rank-deficient design without leaking labels or crashing.
    records = [dict(record, sequence="AA") for record in support()]
    result = predict(records, [{"variant_id": "q", "sequence": "AC"}])
    assert result == pytest.approx({"q": 1.0})


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [("AA", "AC", 0.5), ("AC", "CA", 1.0), ("AAA", "AA", 1 / 3), ("ACG", "AT", 2 / 3)],
)
def test_distance_uses_hamming_and_edit_distance(left, right, expected):
    assert normalized_sequence_distance(left, right) == pytest.approx(expected)
    assert normalized_sequence_distance(right, left) == pytest.approx(expected)


def test_unequal_length_nearest_neighbor_has_no_padding():
    records = [
        {"variant_id": "s0", "sequence": "AA", "fitness": 1.0},
        {"variant_id": "s1", "sequence": "CCCC", "fitness": 8.0},
    ]
    assert sequence_nearest_neighbor(records, [{"variant_id": "q", "sequence": "AAA"}]) == {
        "q": 1.0
    }


def test_ridge_matches_closed_form_penalized_regression():
    # Two opposite sequences differ in four positional one-hot coordinates.
    # With unpenalized intercept and alpha=1, predictions shrink toward 1.
    scores = ridge_predict(
        support(), [{"variant_id": "q0", "sequence": "AA"}, {"variant_id": "q1", "sequence": "CC"}]
    )
    assert scores == pytest.approx({"q0": 1 / 3, "q1": 5 / 3})


@pytest.mark.parametrize("predict", [sequence_nearest_neighbor, ridge_predict])
def test_single_label_and_constant_labels_are_supported(predict):
    query = [{"variant_id": "q", "sequence": "ZZZZZ"}]
    assert predict(support()[:1], query) == {"q": 0.0}
    constant = [dict(record, fitness=3.0) for record in support()]
    assert predict(constant, query) == pytest.approx({"q": 3.0})


def test_ridge_variable_length_encoding_does_not_depend_on_other_queries():
    records = [
        {"variant_id": "s0", "sequence": "A", "fitness": 1.0},
        {"variant_id": "s1", "sequence": "AA", "fitness": 2.0},
    ]
    query = [{"variant_id": "q", "sequence": "AAA"}]
    first = ridge_predict(records, query)
    expanded = ridge_predict(records, query + [{"variant_id": "extra", "sequence": "A" * 100}])
    assert first["q"] == expanded["q"]
    assert np.isfinite(list(expanded.values())).all()


@pytest.mark.parametrize("predict", [sequence_nearest_neighbor, ridge_predict])
def test_invalid_baseline_inputs_raise(predict):
    query = [{"variant_id": "q", "sequence": "AA"}]
    with pytest.raises(ValueError, match="at least one"):
        predict([], query)
    with pytest.raises(ValueError, match="duplicate"):
        predict(support() + support()[:1], query)
    with pytest.raises(ValueError, match="disjoint"):
        predict(support(), [{"variant_id": "s0", "sequence": "AA"}])
    with pytest.raises(ValueError, match="finite"):
        predict([dict(support()[0], fitness=math.inf)], query)
    with pytest.raises(ValueError, match="unsupported"):
        predict(support(), [{"variant_id": "q", "sequence": "A A"}])


@pytest.mark.parametrize("alpha", [0, -1, math.inf, math.nan, True])
def test_ridge_requires_finite_positive_alpha(alpha):
    with pytest.raises(ValueError):
        ridge_predict(support(), [], alpha=alpha)


def test_full_metrics_and_lower_is_better_labels():
    labels = {f"v{i:02}": i for i in range(50)}
    scores = {key: -value for key, value in labels.items()}
    result = evaluate_predictions(scores, labels, higher_is_better=False)
    assert result["spearman"] == pytest.approx(1.0)
    assert result["top_k_overlap_precision"] == 1.0
    assert result["coverage"] == 1.0
    assert result["top_k"] == 25
    opposite = evaluate_predictions(scores, labels)
    assert opposite["spearman"] == pytest.approx(-1.0)
    assert opposite["top_k_overlap_precision"] == 0.0


def test_metric_ties_use_average_ranks_and_deterministic_id_tiebreak():
    labels = {"b": 1, "a": 1, "c": 0}
    scores = {"c": 0, "b": 0.8, "a": 0.8}
    result = evaluate_predictions(scores, labels, top_k=1)
    assert result["spearman"] == pytest.approx(1.0)
    assert result["top_k_overlap_precision"] == 1.0
    reordered = evaluate_predictions(dict(reversed(list(scores.items()))), labels, top_k=1)
    assert reordered == result
    assert "variant_id_asc" in result["tie_policy"]


@pytest.mark.parametrize(
    ("predictions", "labels", "reason"),
    [({"a": 1}, {"a": 2}, "fewer_than_two_queries"),
     ({"a": 1, "b": 2}, {"a": 0, "b": 0}, "constant_measured_fitness"),
     ({"a": 1, "b": 1}, {"a": 0, "b": 2}, "constant_predictions")],
)
def test_undefined_correlations_are_not_zero(predictions, labels, reason):
    result = evaluate_predictions(predictions, labels)
    assert result["spearman"] is None
    assert result["spearman_undefined_reason"] == reason
    assert result["top_k"] == min(25, len(labels))


@pytest.mark.parametrize(
    "predictions",
    [
        {"a": 1},
        {"a": 1, "b": 2, "extra": 3},
        {"a": 1, "b": math.nan},
        {"a": 1, "b": math.inf},
        [{"variant_id": "a", "score": 1}, {"variant_id": "a", "score": 2}],
    ],
)
def test_incomplete_nonfinite_or_duplicate_predictions_rejected(predictions):
    with pytest.raises(ValueError):
        evaluate_predictions(predictions, {"a": 1, "b": 2})


def test_expected_ids_reject_wrong_label_cohort_and_duplicates():
    with pytest.raises(ValueError, match="labels"):
        evaluate_predictions({"a": 1}, {"a": 1, "b": 2}, query_ids=["a"])
    with pytest.raises(ValueError, match="duplicate query"):
        evaluate_predictions({"a": 1}, {"a": 1}, query_ids=["a", "a"])
    with pytest.raises(ValueError, match="duplicate fitness"):
        evaluate_predictions({"a": 1}, [{"variant_id": "a", "fitness": 1}] * 2)


def test_sequence_label_records_and_prediction_records_are_supported():
    result = evaluate_predictions(
        [{"variant_id": "a", "score": 1}, {"variant_id": "b", "score": 2}],
        [{"variant_id": "a", "fitness": 1}, {"variant_id": "b", "fitness": 2}],
    )
    assert result["spearman"] == pytest.approx(1.0)


def test_aggregation_counts_undefined_values_and_rejects_partial():
    good = evaluate_predictions({"a": 1, "b": 2}, {"a": 1, "b": 2})
    undefined = evaluate_predictions({"a": 1, "b": 1}, {"a": 1, "b": 2})
    summary = aggregate_metrics([good, undefined])
    assert summary["n_spearman_defined"] == 1
    assert summary["n_spearman_undefined"] == 1
    assert summary["mean_spearman"] == pytest.approx(1.0)
    with pytest.raises(ValueError, match="complete"):
        aggregate_metrics([dict(good, coverage=0.5)])
