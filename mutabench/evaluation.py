"""Evaluate complete frozen tasks without exposing labels to a predictor.

Task identities are namespaced by the release manifest, not inferred from row
order. A prediction file is one or more explicit model/condition groups. Every
group is evaluated against every task in the manifest, including absent tasks.
"""
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

from scipy.stats import rankdata, spearmanr


def _identifier(value, field):
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{field} must be a nonempty string without surrounding whitespace")
    return value


def _number(value, field):
    if isinstance(value, bool):
        raise ValueError(f"{field} must be finite and numeric")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} must be finite and numeric") from error
    if not math.isfinite(number):
        raise ValueError(f"{field} must be finite and numeric")
    return number


def load_tasks(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("schema") != "mutabench-tasks-v1":
        raise ValueError("Expected schema mutabench-tasks-v1")
    tasks = data.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("A nonempty task list is required")
    seen = set()
    for task in tasks:
        if not isinstance(task, dict):
            raise ValueError("Every task must be an object")
        task_id = _identifier(task["task_id"], "task_id")
        if task_id in seen:
            raise ValueError(f"Duplicate task: {task_id}")
        seen.add(task_id)
        _identifier(task["domain"], "domain")
        if task["metric"] not in {"spearman", "auroc"}:
            raise ValueError(f"Unsupported metric for {task_id}")
        if not isinstance(task.get("higher_label_is_better"), bool):
            raise ValueError(f"Explicit higher_label_is_better is required for {task_id}")
        if task["metric"] == "auroc" and not task["higher_label_is_better"]:
            raise ValueError("AUROC requires label 1 as the positive class and larger predicted scores for it")
        ids = task.get("candidate_ids")
        if not isinstance(ids, list) or not ids:
            raise ValueError(f"Empty candidate set for {task_id}")
        for candidate in ids:
            _identifier(candidate, "variant_id")
        if len(set(ids)) != len(ids):
            raise ValueError(f"Duplicate candidate in {task_id}")
    return tasks


def _csv(path, required):
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        if len(set(fields)) != len(fields) or not required.issubset(fields):
            raise ValueError(f"CSV requires unique columns {sorted(required)}: {path}")
        rows = list(reader)
        if any(None in row or any(v is None for v in row.values()) for row in rows):
            raise ValueError(f"Malformed CSV row: {path}")
        return rows


def metric_value(metric, labels, scores):
    """Signed Spearman or tie-aware AUROC; undefined values remain None."""
    if metric == "spearman":
        if len(labels) < 2 or len(set(labels)) < 2 or len(set(scores)) < 2:
            return None
        return float(spearmanr(labels, scores).statistic)
    if metric != "auroc":
        raise ValueError(f"Unsupported metric: {metric}")
    if not set(labels) <= {0.0, 1.0}:
        raise ValueError("AUROC labels must be 0 or 1")
    positives = sum(labels)
    negatives = len(labels) - positives
    if not positives or not negatives:
        return None
    ranks = rankdata(scores, method="average")
    positive_ranks = math.fsum(float(rank) for rank, label in zip(ranks, labels) if label == 1)
    return (positive_ranks - positives * (positives + 1) / 2) / (positives * negatives)


def evaluate(tasks_path, labels_path, predictions_path):
    tasks = load_tasks(tasks_path)
    expected = {(t["task_id"], candidate) for t in tasks for candidate in t["candidate_ids"]}
    labels = {}
    for row in _csv(labels_path, {"task_id", "variant_id", "label"}):
        key = (row["task_id"], row["variant_id"])
        if key not in expected or key in labels:
            raise ValueError(f"Unexpected or duplicate label identity: {key}")
        labels[key] = _number(row["label"], "label")
    if set(labels) != expected:
        raise ValueError("Labels must cover exactly the frozen candidate identities")
    # Validate label domains even when predictions are absent or incomplete.
    for task in tasks:
        if task["metric"] == "auroc" and not {
            labels[task["task_id"], candidate] for candidate in task["candidate_ids"]
        } <= {0.0, 1.0}:
            raise ValueError(f"AUROC labels must be 0 or 1: {task['task_id']}")

    groups = defaultdict(dict)
    for row in _csv(predictions_path, {"task_id", "variant_id", "score"}):
        key = (row["task_id"], row["variant_id"])
        model = _identifier(row.get("model", "submitted"), "model")
        condition = _identifier(row.get("condition", "default"), "condition")
        predictions = groups[model, condition]
        if key not in expected or key in predictions:
            raise ValueError(f"Unexpected or duplicate prediction identity: {(model, condition, *key)}")
        predictions[key] = _number(row["score"], "score")
    if not groups:
        groups["submitted", "default"] = {}

    per_task = []
    for (model, condition), predictions in sorted(groups.items()):
        for task in tasks:
            keys = [(task["task_id"], candidate) for candidate in task["candidate_ids"]]
            count = sum(key in predictions for key in keys)
            complete = count == len(keys)
            value = None
            if complete:
                direction = 1 if task["higher_label_is_better"] else -1
                value = metric_value(task["metric"], [direction * labels[key] for key in keys],
                                     [predictions[key] for key in keys])
            per_task.append(dict(model=model, condition=condition, task_id=task["task_id"],
                domain=task["domain"], metric=task["metric"], n_expected=len(keys), n_predicted=count,
                status="complete" if complete else "incomplete",
                metric_status="defined" if value is not None else "undefined" if complete else "not_scored",
                value=value))
    summary_groups = defaultdict(list)
    for row in per_task:
        summary_groups[row["model"], row["condition"], row["domain"], row["metric"]].append(row)
    summary = []
    for (model, condition, domain, metric), rows in sorted(summary_groups.items()):
        values = [row["value"] for row in rows if row["value"] is not None]
        summary.append(dict(model=model, condition=condition, domain=domain, metric=metric,
            n_tasks=len(rows), n_complete=sum(row["status"] == "complete" for row in rows),
            n_defined=len(values), mean=math.fsum(values) / len(values) if values else None))
    return dict(schema="mutabench-evaluation-v1", per_task=per_task, summary=summary,
        policy=dict(signed_spearman=True, incomplete_not_scored=True, undefined_not_zero=True,
                    equal_task_weights=True, cross_model_means_require_matched_tasks=True))
