"""Validate a public development sample using the release evaluator's task contract."""
import hashlib
import json
import re
from pathlib import Path

from .evaluation import _csv, _identifier, _number, load_tasks

FILES = {"dataset.json", "tasks.json", "queries.csv", "labels.csv"}


def validate_dataset(directory: Path):
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("Expected a dataset directory")
    entries = list(directory.iterdir())
    if {p.name for p in entries} != FILES or any(p.is_symlink() or not p.is_file() for p in entries):
        raise ValueError("A public development package contains exactly: " + ", ".join(sorted(FILES)))

    card = json.loads((directory / "dataset.json").read_text(encoding="utf-8"))
    if not isinstance(card, dict) or card.get("schema") != "mutabench-community-v1":
        raise ValueError("Expected dataset.json schema mutabench-community-v1")
    for field in ("dataset_id", "version", "title", "source", "source_revision", "license", "readout", "split"):
        _identifier(card.get(field), field)
    dataset_id = card["dataset_id"]
    if dataset_id != directory.name or not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", dataset_id):
        raise ValueError("dataset_id must match its lowercase, hyphenated directory name")
    if not re.fullmatch(r"(?:0|[1-9][0-9]*)(?:\.(?:0|[1-9][0-9]*)){2}", card["version"]):
        raise ValueError("Expected a numeric X.Y.Z version")
    if card.get("test_access") not in ("development-only", "withheld"):
        raise ValueError("test_access must be development-only or withheld")
    contributors = card.get("contributors")
    if not isinstance(contributors, list) or not contributors:
        raise ValueError("List at least one contributor")
    for name in contributors:
        _identifier(name, "contributor")

    tasks = load_tasks(directory / "tasks.json")
    expected = set()
    for task in tasks:
        if not task["task_id"].startswith(dataset_id + "/") or task["task_id"] == dataset_id + "/":
            raise ValueError("Prefix each task_id with dataset_id/ and an assay identifier")
        expected.update((task["task_id"], variant) for variant in task["candidate_ids"])

    keys = {"task_id", "variant_id"}
    queries = _csv(directory / "queries.csv", keys)
    inputs = set(queries[0]) - keys if queries else set()
    if not inputs or inputs & {"label", "target", "measured_effect", "y_true"}:
        raise ValueError("Supply model-visible inputs only in queries.csv")
    query_ids = {(r["task_id"], r["variant_id"]) for r in queries}
    if query_ids != expected or len(queries) != len(expected):
        raise ValueError("Queries must cover each frozen candidate exactly once")
    if any(not any(row[col].strip() for col in inputs) for row in queries):
        raise ValueError("Each query needs at least one populated input")

    records = _csv(directory / "labels.csv", keys | {"label"})
    labels = {(r["task_id"], r["variant_id"]): _number(r["label"], "label") for r in records}
    if len(records) != len(expected) or labels.keys() != expected:
        raise ValueError("Development labels must cover each frozen candidate exactly once")
    stats = []
    for task in tasks:
        values = [labels[task["task_id"], variant] for variant in task["candidate_ids"]]
        if task["metric"] == "auroc" and not set(values) <= {0, 1}:
            raise ValueError("AUROC labels must be binary")
        stats.append(dict(task_id=task["task_id"], candidates=len(values), distinct_labels=len(set(values))))
    return dict(dataset_id=dataset_id, version=card["version"], development_tasks=stats,
                sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(entries)})
