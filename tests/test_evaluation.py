"""Check identity isolation, missing predictions, ties and metric direction."""
import csv
import json
from pathlib import Path

import pytest

from mutabench.evaluation import evaluate, metric_value


def write_csv(path, fields, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def files(tmp_path):
    tasks = [{"task_id": task, "domain": "protein", "metric": "spearman",
              "higher_label_is_better": True, "candidate_ids": ["a", "b", "c"]}
             for task in ("source1/assay", "source2/assay")]
    tp = tmp_path / "tasks.json"
    tp.write_text(json.dumps({"schema": "mutabench-tasks-v1", "tasks": tasks}))
    labels = [{"task_id": task["task_id"], "variant_id": candidate, "label": i}
              for task in tasks for i, candidate in enumerate(task["candidate_ids"])]
    lp = tmp_path / "labels.csv"
    write_csv(lp, ["task_id", "variant_id", "label"], labels)
    predictions = [{"task_id": row["task_id"], "variant_id": row["variant_id"], "score": row["label"]}
                   for row in reversed(labels)]
    pp = tmp_path / "predictions.csv"
    write_csv(pp, ["task_id", "variant_id", "score"], predictions)
    return tp, lp, pp, tasks, predictions


def test_matches_namespaced_ids_instead_of_row_order(files):
    result = evaluate(*files[:3])
    assert [r["value"] for r in result["per_task"]] == [1, 1]


def test_signed_spearman_and_ties():
    assert metric_value("spearman", [1, 2, 3], [3, 2, 1]) == -1
    assert metric_value("spearman", [1, 1, 3], [2, 2, 9]) == pytest.approx(1)
    assert metric_value("auroc", [0, 1, 0, 1], [1, 1, 1, 1]) == .5
    assert metric_value("auroc", [0, 1, 0, 1], [0, 1, 0, 1]) == 1


@pytest.mark.parametrize("metric,labels,scores", [
    ("spearman", [1, 1], [1, 2]), ("spearman", [1, 2], [0, 0]),
    ("spearman", [1], [1]), ("auroc", [0, 0], [1, 2]),
])
def test_undefined_is_not_zero(metric, labels, scores):
    assert metric_value(metric, labels, scores) is None


def test_incomplete_task_is_not_scored(files):
    tp, lp, pp, _, rows = files
    write_csv(pp, ["task_id", "variant_id", "score"], rows[:-1])
    result = evaluate(tp, lp, pp)
    assert result["summary"][0]["n_complete"] == 1
    assert result["summary"][0]["n_tasks"] == 2
    assert result["per_task"][0]["value"] is None


def test_absent_task_is_reported(files):
    tp, lp, pp, _, rows = files
    write_csv(pp, ["task_id", "variant_id", "score"], [r for r in rows if r["task_id"] == "source1/assay"])
    result = evaluate(tp, lp, pp)
    assert result["per_task"][1]["n_predicted"] == 0
    assert result["per_task"][1]["status"] == "incomplete"


@pytest.mark.parametrize("change", ["duplicate", "unexpected", "nan", "inf"])
def test_invalid_predictions_fail(files, change):
    tp, lp, pp, _, rows = files
    if change == "duplicate":
        rows.append(rows[0])
    elif change == "unexpected":
        rows[0]["variant_id"] = "missing"
    else:
        rows[0]["score"] = change
    write_csv(pp, ["task_id", "variant_id", "score"], rows)
    with pytest.raises(ValueError):
        evaluate(tp, lp, pp)


def test_label_direction_is_explicit(files):
    tp, lp, pp, tasks, _ = files
    tasks[0]["higher_label_is_better"] = False
    tp.write_text(json.dumps({"schema": "mutabench-tasks-v1", "tasks": tasks}))
    assert evaluate(tp, lp, pp)["per_task"][0]["value"] == -1


def test_labels_must_cover_exact_query(files):
    tp, lp, pp, _, _ = files
    lp.write_text("task_id,variant_id,label\nsource1/assay,a,1\n")
    with pytest.raises(ValueError, match="exactly"):
        evaluate(tp, lp, pp)


def test_groups_do_not_share_predictions(files):
    tp, lp, pp, _, rows = files
    rows = [dict(r, model="complete", condition="sequence") for r in rows] + [
        dict(rows[0], model="partial", condition="sequence")]
    write_csv(pp, ["task_id", "variant_id", "score", "model", "condition"], rows)
    result = evaluate(tp, lp, pp)
    assert [r["n_complete"] for r in result["summary"]] == [2, 0]


def test_duplicate_manifest_ids_rejected(files):
    tp, lp, pp, tasks, _ = files
    tasks[0]["candidate_ids"].append("a")
    tp.write_text(json.dumps({"schema": "mutabench-tasks-v1", "tasks": tasks}))
    with pytest.raises(ValueError, match="Duplicate candidate"):
        evaluate(tp, lp, pp)


def test_synthetic_example():
    root = Path(__file__).resolve().parents[1]
    result = evaluate(root/"examples/tasks.json", root/"examples/labels.csv", root/"examples/predictions.csv")
    assert [r["value"] for r in result["per_task"]] == [1, 1]
