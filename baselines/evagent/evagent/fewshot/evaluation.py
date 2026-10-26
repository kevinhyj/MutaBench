"""Label-budget-matched sequence baselines and isolated few-shot evaluation.

Baseline scores always have the ``higher is better`` orientation. Query labels,
if accidentally present in an input object, are never read. The evaluator is the
only entry point here that consumes held-out fitness labels.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from scipy.stats import spearmanr

# A fixed vocabulary, including ambiguous residues, RNA U, stops and gaps. No
# vocabulary is learned from the held-out labels or from the query sequences.
SEQUENCE_ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ-.*"
_ALPHABET_INDEX = {letter: index for index, letter in enumerate(SEQUENCE_ALPHABET)}


def _identifier(value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("variant_id must be a nonempty string")
    return value


def _finite(value: Any, field: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a finite number, not a boolean")
    try:
        result = float(value)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError(f"{field} must be a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _sequences(records: Sequence[Mapping[str, Any]], role: str) -> list[tuple[str, str]]:
    result: list[tuple[str, str]] = []
    seen: set[str] = set()
    for record in records:
        variant_id = _identifier(record["variant_id"])
        if variant_id in seen:
            raise ValueError(f"duplicate {role} variant_id: {variant_id}")
        sequence = record["sequence"]
        if not isinstance(sequence, str) or not sequence:
            raise ValueError(f"{role} sequence must be a nonempty string: {variant_id}")
        sequence = sequence.upper()
        if set(sequence) - set(SEQUENCE_ALPHABET):
            raise ValueError(f"unsupported character in {role} sequence: {variant_id}")
        seen.add(variant_id)
        result.append((variant_id, sequence))
    return result


def _baseline_inputs(
    support: Sequence[Mapping[str, Any]],
    query: Sequence[Mapping[str, Any]],
    higher_is_better: bool,
) -> tuple[list[tuple[str, str]], list[tuple[str, str]], np.ndarray]:
    if not isinstance(higher_is_better, bool):
        raise ValueError("higher_is_better must be a boolean")
    if not support:
        raise ValueError("a supervised baseline requires at least one support label")
    support_sequences = _sequences(support, "support")
    query_sequences = _sequences(query, "query")
    if {item[0] for item in support_sequences} & {item[0] for item in query_sequences}:
        raise ValueError("support and query variant IDs must be disjoint")
    direction = 1.0 if higher_is_better else -1.0
    targets = np.array([direction * _finite(item["fitness"], "fitness") for item in support])
    return support_sequences, query_sequences, targets


def normalized_sequence_distance(left: str, right: str) -> float:
    """Hamming / length for equal lengths; Levenshtein / max length otherwise.

    No padding, truncation, alignment inference or label-dependent weighting is
    applied. Empty sequences are invalid for both protein and RNA episodes.
    """
    if not left or not right:
        raise ValueError("sequence distance requires nonempty sequences")
    if len(left) == len(right):
        return sum(a != b for a, b in zip(left, right)) / len(left)
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for row_index, left_letter in enumerate(left, start=1):
        current = [row_index]
        for column_index, right_letter in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column_index] + 1,
                    previous[column_index - 1] + (left_letter != right_letter),
                )
            )
        previous = current
    return previous[-1] / len(left)


def sequence_nearest_neighbor(
    support: Sequence[Mapping[str, Any]],
    query: Sequence[Mapping[str, Any]],
    higher_is_better: bool = True,
) -> dict[str, float]:
    """Predict with the mean label of all equally nearest support sequences."""
    support_sequences, query_sequences, targets = _baseline_inputs(
        support, query, higher_is_better
    )
    result: dict[str, float] = {}
    for variant_id, sequence in query_sequences:
        distances = np.array(
            [normalized_sequence_distance(sequence, item[1]) for item in support_sequences]
        )
        nearest = np.isclose(distances, distances.min(), atol=1e-12, rtol=0)
        result[variant_id] = float(targets[nearest].mean())
    return result


def _position_features(sequences: list[tuple[str, str]], support_length: int) -> np.ndarray:
    # Absent positions are all zero, explicitly NOT a residue or a gap. A
    # separate normalized length feature distinguishes length effects. Query
    # positions beyond the longest support sequence have no trainable residue
    # coefficient and influence only length. The mutation string is not used:
    # this baseline is defined for datasets without WT or mutation annotations.
    width = len(SEQUENCE_ALPHABET)
    features = np.zeros((len(sequences), support_length * width + 1), dtype=float)
    for row, (_, sequence) in enumerate(sequences):
        for position, residue in enumerate(sequence[:support_length]):
            features[row, position * width + _ALPHABET_INDEX[residue]] = 1.0
        features[row, -1] = len(sequence) / support_length
    return features


def ridge_predict(
    support: Sequence[Mapping[str, Any]],
    query: Sequence[Mapping[str, Any]],
    higher_is_better: bool = True,
    *,
    alpha: float = 1.0,
) -> dict[str, float]:
    """Fixed-alpha positional one-hot ridge with an unpenalized intercept.

    Features and target centering use support samples only. The fixed residue
    alphabet needs neither a WT nor query-label statistics. See
    ``_position_features`` for the explicit variable-length policy. Alpha is
    fixed at 1.0 for the first experiment; this function does no tuning.
    """
    support_sequences, query_sequences, targets = _baseline_inputs(
        support, query, higher_is_better
    )
    alpha = _finite(alpha, "alpha")
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    support_length = max(len(item[1]) for item in support_sequences)
    train = _position_features(support_sequences, support_length)
    test = _position_features(query_sequences, support_length)
    feature_mean = train.mean(axis=0)
    target_mean = targets.mean()
    centered = train - feature_mean
    # Few-shot support is small; a dual solve avoids a large position x residue
    # matrix inverse, and regularization also handles rank-deficient designs.
    dual = np.linalg.solve(
        centered @ centered.T + alpha * np.eye(len(support)), targets - target_mean
    )
    scores = (test - feature_mean) @ centered.T @ dual + target_mean
    return {item[0]: float(score) for item, score in zip(query_sequences, scores)}


def _score_map(
    records: Mapping[str, Any] | Sequence[Mapping[str, Any]], field: str
) -> dict[str, float]:
    if isinstance(records, Mapping):
        items = records.items()
    else:
        items = ((record["variant_id"], record[field]) for record in records)
    result: dict[str, float] = {}
    for raw_id, value in items:
        variant_id = _identifier(raw_id)
        if variant_id in result:
            raise ValueError(f"duplicate {field} variant_id: {variant_id}")
        result[variant_id] = _finite(value, field)
    return result


def evaluate_predictions(
    predictions: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    labels: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    higher_is_better: bool = True,
    *,
    query_ids: Sequence[str] | None = None,
    top_k: int = 25,
) -> dict[str, Any]:
    """Evaluate a complete, unique, finite held-out prediction set.

    Predictions are already higher-is-better; only measured labels are oriented
    here. Duplicate, missing, unexpected and nonfinite predictions raise an
    error rather than silently changing the comparison cohort. Spearman uses
    average ranks for ties and is undefined (None) for a constant vector or N<2.
    Top-k ties are broken by ascending variant_id on both sides, independent of
    input ordering; this deterministic convention must accompany the metric.
    """
    if not isinstance(higher_is_better, bool):
        raise ValueError("higher_is_better must be a boolean")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
        raise ValueError("top_k must be a positive integer")
    predicted = _score_map(predictions, "score")
    measured = _score_map(labels, "fitness")
    expected = list(measured) if query_ids is None else [_identifier(item) for item in query_ids]
    if not expected:
        raise ValueError("evaluation requires at least one query")
    if len(expected) != len(set(expected)):
        raise ValueError("duplicate query variant_id")
    expected_set = set(expected)
    for role, values in (("predictions", predicted), ("labels", measured)):
        missing = sorted(expected_set - set(values))
        extra = sorted(set(values) - expected_set)
        if missing or extra:
            raise ValueError(f"incomplete or unexpected {role}: missing={missing}, extra={extra}")
    direction = 1.0 if higher_is_better else -1.0
    truth = {key: direction * value for key, value in measured.items()}
    truth_vector = np.array([truth[item] for item in expected])
    score_vector = np.array([predicted[item] for item in expected])
    spearman: float | None = None
    undefined_reason: str | None = None
    if len(expected) < 2:
        undefined_reason = "fewer_than_two_queries"
    elif np.all(truth_vector == truth_vector[0]):
        undefined_reason = "constant_measured_fitness"
    elif np.all(score_vector == score_vector[0]):
        undefined_reason = "constant_predictions"
    else:
        correlation = float(spearmanr(score_vector, truth_vector).statistic)
        if math.isfinite(correlation):
            spearman = correlation
        else:
            undefined_reason = "nonfinite_spearman"
    effective_k = min(top_k, len(expected))
    predicted_top = sorted(expected, key=lambda item: (-predicted[item], item))[:effective_k]
    measured_top = sorted(expected, key=lambda item: (-truth[item], item))[:effective_k]
    overlap = len(set(predicted_top) & set(measured_top))
    return {
        "n_expected": len(expected),
        "n_predicted": len(predicted),
        "n_valid": len(predicted),
        "coverage": 1.0,
        "spearman": spearman,
        "spearman_undefined_reason": undefined_reason,
        "top_k": effective_k,
        "top_k_requested": top_k,
        "top_k_overlap": overlap,
        "top_k_overlap_precision": overlap / effective_k,
        "tie_policy": "spearman_average_rank;top_k_score_desc_variant_id_asc",
    }


def aggregate_metrics(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Macro-average complete episodes; explicitly count undefined correlations.

    This is a descriptive seed summary, not a confidence interval and not an
    aggregation over different biological studies. Callers must group matching
    assays, methods, protocols and shot counts before calling this function.
    """
    if not results:
        raise ValueError("cannot aggregate an empty set of evaluations")
    for result in results:
        if result["coverage"] != 1.0 or result["n_valid"] != result["n_expected"]:
            raise ValueError("only complete evaluations can be aggregated")
    correlations = [
        _finite(result["spearman"], "spearman")
        for result in results
        if result["spearman"] is not None
    ]
    precision = [_finite(result["top_k_overlap_precision"], "precision") for result in results]
    return {
        "n_episodes": len(results),
        "n_spearman_defined": len(correlations),
        "n_spearman_undefined": len(results) - len(correlations),
        "mean_spearman": float(np.mean(correlations)) if correlations else None,
        "mean_top_k_overlap_precision": float(np.mean(precision)),
        "minimum_coverage": 1.0,
    }
