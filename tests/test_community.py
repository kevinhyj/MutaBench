"""Run submission checks on public fixtures only."""
import json
import shutil
from pathlib import Path

import pytest

from mutabench.community import validate_dataset

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples/community/example-fluorescence"


def test_submission_catalog():
    result = validate_dataset(EXAMPLE)
    assert result["development_tasks"] == [
        dict(task_id="example-fluorescence/brightness", candidates=4, distinct_labels=3)
    ]
    for package in sorted((ROOT / "community").iterdir()):
        if package.name != "README.md":
            validate_dataset(package)


@pytest.fixture
def sample(tmp_path):
    return shutil.copytree(EXAMPLE, tmp_path / EXAMPLE.name)


def test_rejects_missing_development_label(sample):
    labels = sample / "labels.csv"
    labels.write_text("\n".join(labels.read_text().splitlines()[:-1]))
    with pytest.raises(ValueError, match="labels must cover"):
        validate_dataset(sample)


def test_rejects_repeated_query(sample):
    queries = sample / "queries.csv"
    with queries.open("a") as handle:
        handle.write(queries.read_text().splitlines()[-1] + "\n")
    with pytest.raises(ValueError, match="exactly once"):
        validate_dataset(sample)


def test_rejects_label_in_query(sample):
    queries = sample / "queries.csv"
    queries.write_text(queries.read_text().replace("reporter_sequence", "label"))
    with pytest.raises(ValueError, match="model-visible"):
        validate_dataset(sample)


def test_rejects_private_extra_file(sample):
    (sample / "test_labels.csv").touch()
    with pytest.raises(ValueError, match="exactly"):
        validate_dataset(sample)


def test_rejects_nonbinary_classification(sample):
    path = sample / "tasks.json"
    data = json.loads(path.read_text())
    data["tasks"][0]["metric"] = "auroc"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="binary"):
        validate_dataset(sample)
