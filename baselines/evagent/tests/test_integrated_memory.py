"""Offline tests for the adapted collaborator memory interface."""

import hashlib
import json
import sqlite3
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pytest

from evagent.integrated.learning_memory import KnowledgeCard, TaskContext
from evagent.integrated.learning_store import LearningMemoryStore
from evagent.integrated.memory import (
    MemoryProvider,
    checkpoint_path_for_mode,
    import_legacy_basic_memory,
    initialize_memory,
    inspect_memory,
    write_basic_memory,
)
from evagent.integrated.strategy_memory import load_strategy_memory, validate_strategy_memory


def _card() -> KnowledgeCard:
    return KnowledgeCard(
        card_id="mem_test", version=1, kind="procedural", status="active",
        condition="ranking output is required", guidance="validate identifier coverage",
        contraindication="do not infer complete coverage from valid JSON",
        tags=("all", "output_mode:ranking"), confidence=0.8,
        support_count=6, source_episode_ids=("exp_a",),
    )


def test_disabled_memory_does_not_open_or_inspect_files(tmp_path):
    provider = MemoryProvider(basic_db=tmp_path / "no-basic", learning_db=tmp_path / "no-learning")
    with patch("sqlite3.connect", side_effect=AssertionError("database opened")), \
         patch.object(Path, "read_bytes", side_effect=AssertionError("file read")), \
         patch.object(Path, "exists", side_effect=AssertionError("filesystem inspected")), \
         patch.object(Path, "resolve", side_effect=AssertionError("filesystem resolved")):
        first = provider.snapshot()
        assert first == provider.snapshot()
    assert first["text"] == ""
    assert first["sources"] == []
    assert first["mode"] == "none"
    assert list(tmp_path.iterdir()) == []


def test_basic_admin_write_is_persistent_and_inference_is_read_only(tmp_path):
    path = tmp_path / "basic.sqlite"
    write_basic_memory(path, "Validate identifier coverage before returning JSON.")
    before = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in tmp_path.iterdir()}
    calls = []
    original = sqlite3.connect
    def tracking_connect(*args, **kwargs):
        calls.append((args, kwargs))
        return original(*args, **kwargs)
    with patch("sqlite3.connect", side_effect=tracking_connect):
        snap = MemoryProvider(mode="basic", basic_db=path).snapshot()
        assert inspect_memory(basic_db=path)["basic"]["sha256"] == snap["sha256"]
    assert "Validate identifier coverage" in snap["text"]
    assert all("mode=ro&immutable=1" in str(args[0]) and kwargs["uri"] for args, kwargs in calls)
    assert {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in tmp_path.iterdir()} == before
    updated = write_basic_memory(path, "Validate identifier uniqueness before returning JSON.")
    assert snap["sha256"] != updated["sha256"]
    assert updated["sources"][0]["version"] == 2


def test_missing_active_database_is_not_created(tmp_path):
    path = tmp_path / "missing" / "learning.sqlite"
    with pytest.raises(FileNotFoundError):
        MemoryProvider(mode="learning", learning_db=path).snapshot()
    assert not path.parent.exists()


def test_learning_retrieves_only_active_latest_and_never_writes(tmp_path):
    basic, learning = tmp_path / "basic.sqlite", tmp_path / "learning.sqlite"
    initialize_memory(basic, learning)
    write_basic_memory(basic, "Check the final output schema.")
    connection = sqlite3.connect(learning)
    store = LearningMemoryStore(connection, learning)
    store.put_card(_card())
    store.put_card(replace(_card(), card_id="candidate", status="candidate"))
    store.put_card(replace(_card(), card_id="validated", status="validated"))
    connection.close()
    before = learning.read_bytes()
    provider = MemoryProvider(mode="hybrid", basic_db=basic, learning_db=learning)
    active = provider.snapshot(TaskContext())
    assert "final output schema" in active["text"]
    assert "validate identifier coverage" in active["text"]
    assert active["sources"][1]["cards"] == [{"card_id": "mem_test", "version": 1}]
    assert learning.read_bytes() == before
    connection = sqlite3.connect(learning)
    LearningMemoryStore(connection, learning).put_card(replace(_card(), version=2, status="retired"))
    connection.close()
    retired = provider.snapshot()
    assert "validate identifier coverage" not in retired["text"]
    assert retired["sha256"] != active["sha256"]


def test_modes_and_frozen_strategy_are_explicit(tmp_path):
    strategy = tmp_path / "strategy.md"
    strategy.write_text("Verify complete output coverage.")
    for mode in ("none", "basic"):
        with pytest.raises(ValueError, match="strategy_file requires"):
            MemoryProvider(mode=mode, strategy_file=strategy)
    frozen = MemoryProvider(mode="learning", strategy_file=strategy).snapshot()
    assert frozen["text"] == "Verify complete output coverage."
    strategy.write_text("Verify output uniqueness.")
    assert MemoryProvider(mode="learning", strategy_file=strategy).snapshot()["sha256"] != frozen["sha256"]
    assert not list(tmp_path.glob("*.sqlite"))
    assert checkpoint_path_for_mode(tmp_path / "checkpoints.sqlite", "none") != \
        checkpoint_path_for_mode(tmp_path / "checkpoints.sqlite", "hybrid")
    with pytest.raises(ValueError, match="physically separate"):
        initialize_memory(tmp_path / "same", tmp_path / "same")
    with pytest.raises(ValueError, match="physically separate"):
        MemoryProvider(mode="hybrid", basic_db=tmp_path / "same", learning_db=tmp_path / "same")


@pytest.mark.parametrize("text,reason", [
    ("Prefer M01.", "candidate IDs"),
    ("Use ACGUACGUACGUACGUACGU.", "RNA sequences"),
    ("Use heldout_assay examples.", "assay identifiers"),
])
def test_strategy_and_basic_inference_reject_direct_leakage(tmp_path, text, reason):
    with pytest.raises(ValueError, match=reason):
        validate_strategy_memory(text, forbidden_assays=["heldout_assay"])
    if reason == "assay identifiers":
        path = tmp_path / "basic.sqlite"
        write_basic_memory(path, text)
        with pytest.raises(ValueError, match=reason):
            MemoryProvider(mode="basic", basic_db=path).snapshot(forbidden_assays=["heldout_assay"])


def test_frozen_strategy_load_fingerprints_normalized_text(tmp_path):
    path = tmp_path / "strategy.md"
    path.write_bytes(b"  Verify output coverage.\r\n")
    strategy = load_strategy_memory(path, forbidden_assays=[])
    assert strategy.text == "Verify output coverage."
    assert strategy.sha256 == hashlib.sha256(strategy.text.encode()).hexdigest()


def test_legacy_import_is_explicit_and_preserves_source(tmp_path):
    legacy, target = tmp_path / "legacy.sqlite", tmp_path / "basic.sqlite"
    connection = sqlite3.connect(legacy)
    connection.execute("CREATE TABLE store(prefix TEXT,key TEXT,value TEXT)")
    connection.execute("INSERT INTO store VALUES(?,?,?)", (
        "rnagym-deep-agent", "/memories/AGENTS.md",
        json.dumps({"content": ["Validate output coverage.", "Keep prompts reproducible."]}),
    ))
    connection.commit()
    connection.close()
    original = legacy.read_bytes()
    with pytest.raises(ValueError, match="explicitly import"):
        MemoryProvider(mode="basic", basic_db=legacy).snapshot()
    with pytest.raises(ValueError, match="explicit import"):
        initialize_memory(basic_db=legacy)
    imported = import_legacy_basic_memory(legacy, target)
    assert "Keep prompts reproducible." in imported["snapshot"]["text"]
    assert legacy.read_bytes() == original


def test_nonempty_wal_is_rejected_without_creating_sidecars(tmp_path):
    path = tmp_path / "basic.sqlite"
    write_basic_memory(path, "Validate output coverage.")
    wal = Path(str(path) + "-wal")
    wal.write_bytes(b"uncheckpointed")
    before = sorted(p.name for p in tmp_path.iterdir())
    with pytest.raises(ValueError, match="uncheckpointed WAL"):
        MemoryProvider(mode="basic", basic_db=path).snapshot()
    assert sorted(p.name for p in tmp_path.iterdir()) == before


def test_snapshot_detects_concurrent_admin_edit(tmp_path):
    path = tmp_path / "basic.sqlite"
    write_basic_memory(path, "Validate output coverage.")
    raw = path.read_bytes()
    with patch.object(Path, "read_bytes", side_effect=[raw, raw + b"changed"]):
        with pytest.raises(RuntimeError, match="changed while"):
            MemoryProvider(mode="basic", basic_db=path).snapshot()
