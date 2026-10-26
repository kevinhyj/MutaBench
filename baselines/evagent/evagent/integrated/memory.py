"""Frozen inference memory and explicit administrative storage operations.

Adapted from ypxia/src/rnagym_deep_agent/memory.py. The source mode separation
is retained, while Deep Agents storage and checkpoint ownership are removed.
Inference never initializes, migrates, seeds, or writes a database. Legacy
LangGraph basic memory requires the explicit ``import_legacy_basic_memory``
administrative operation; the independent learning schema is preserved.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import contextmanager
from dataclasses import asdict
from enum import Enum
from pathlib import Path
from typing import Any, Iterator

from .learning_memory import TaskContext, compile_strategy_text, retrieve_cards
from .learning_store import LearningMemoryStore
from .strategy_memory import load_strategy_memory, validate_strategy_memory

MEMORY_VERSION = "evagent-frozen-memory-v1"
MEMORY_NAMESPACE = ("rnagym-deep-agent",)
MEMORY_KEY = "/AGENTS.md"
LEGACY_MEMORY_KEY = "/memories/AGENTS.md"


class MemoryMode(str, Enum):
    NONE = "none"
    BASIC = "basic"
    LEARNING = "learning"
    HYBRID = "hybrid"

    @property
    def uses_basic(self) -> bool:
        return self in {MemoryMode.BASIC, MemoryMode.HYBRID}

    @property
    def uses_learning(self) -> bool:
        return self in {MemoryMode.LEARNING, MemoryMode.HYBRID}


MEMORY_MODES = tuple(mode.value for mode in MemoryMode)


def parse_memory_mode(value: str | MemoryMode) -> MemoryMode:
    if isinstance(value, MemoryMode):
        return value
    try:
        return MemoryMode(value.strip().casefold())
    except (ValueError, AttributeError) as error:
        raise ValueError(f"unknown memory mode {value!r}; choose one of {MEMORY_MODES}") from error


def checkpoint_path_for_mode(path: Path, mode: str | MemoryMode) -> Path:
    selected = parse_memory_mode(mode)
    suffix = path.suffix or ".sqlite"
    stem = path.stem if path.suffix else path.name
    return path.with_name(f"{stem}.{selected.value}{suffix}")


def _hash(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


@contextmanager
def _read_database(path: Path) -> Iterator[tuple[sqlite3.Connection, dict[str, Any]]]:
    """Read a frozen DELETE-journal database without making SQLite sidecars.

    ``immutable`` prevents SQLite from writing a WAL shared-memory file. A
    nonempty WAL is rejected rather than silently ignoring uncheckpointed data.
    Source hashes are checked again on exit to detect concurrent admin edits.
    """
    path = Path(path).resolve(strict=True)
    wal = Path(str(path) + "-wal")
    if wal.exists() and wal.stat().st_size:
        raise ValueError("memory has uncheckpointed WAL data; freeze it administratively first")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    connection = sqlite3.connect(path.as_uri() + "?mode=ro&immutable=1", uri=True)
    try:
        connection.execute("PRAGMA query_only=ON")
        yield connection, {"path": str(path), "sha256": digest}
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise RuntimeError("memory database changed while taking its frozen snapshot")
        if wal.exists() and wal.stat().st_size:
            raise RuntimeError("memory WAL changed while taking its frozen snapshot")
    finally:
        connection.close()


class MemoryProvider:
    """The sole prediction-side memory interface; dependencies are standard library."""

    def __init__(
        self,
        *,
        mode: str | MemoryMode = "none",
        basic_db: Path | None = None,
        learning_db: Path | None = None,
        strategy_file: Path | None = None,
    ) -> None:
        self.mode = parse_memory_mode(mode)
        self.basic_db = basic_db
        self.learning_db = learning_db
        self.strategy_file = strategy_file
        if strategy_file is not None and not self.mode.uses_learning:
            raise ValueError("strategy_file requires learning or hybrid memory mode")
        if self.mode.uses_basic and basic_db is None:
            raise ValueError("basic_db is required for basic or hybrid memory mode")
        if self.mode.uses_learning and learning_db is None and strategy_file is None:
            raise ValueError("learning memory requires learning_db or a frozen strategy_file")
        if self.mode == MemoryMode.HYBRID and learning_db is not None:
            assert basic_db is not None
            if Path(basic_db).resolve() == Path(learning_db).resolve():
                raise ValueError("basic and learning memory require physically separate databases")

    def snapshot(
        self,
        context: TaskContext | None = None,
        forbidden_assays: list[str] | None = None,
    ) -> dict[str, Any]:
        forbidden = forbidden_assays or []
        sources: list[dict[str, Any]] = []
        parts: list[str] = []
        if self.mode.uses_basic:
            assert self.basic_db is not None
            with _read_database(self.basic_db) as (connection, source):
                try:
                    row = connection.execute(
                        "SELECT text, version FROM basic_memory WHERE singleton=1"
                    ).fetchone()
                except sqlite3.OperationalError as error:
                    raise ValueError(
                        "basic memory schema missing; initialize or explicitly import legacy memory"
                    ) from error
                text, version = (str(row[0]), int(row[1])) if row else ("", 0)
                if text.strip():
                    parts.append(validate_strategy_memory(text, forbidden_assays=forbidden))
                sources.append({"kind": "basic", "version": version, **source})
        if self.mode.uses_learning and self.learning_db is not None:
            with _read_database(self.learning_db) as (connection, source):
                store = LearningMemoryStore(connection, Path(self.learning_db))
                cards = retrieve_cards(store.list_cards(status="active"), context or TaskContext())
                if cards:
                    parts.append(compile_strategy_text(cards, forbidden_assays=forbidden))
                sources.append({
                    "kind": "learning", **source,
                    "cards": [{"card_id": card.card_id, "version": card.version} for card in cards],
                    "selected_cards_sha256": _hash([asdict(card) for card in cards]),
                    "context": asdict(context or TaskContext()),
                })
        if self.strategy_file is not None:
            strategy = load_strategy_memory(Path(self.strategy_file), forbidden_assays=forbidden)
            parts.append(strategy.text)
            sources.append({
                "kind": "frozen_strategy", "path": str(strategy.source_path),
                "sha256": strategy.sha256,
            })
        snapshot = {
            "text": "\n\n".join(parts), "version": MEMORY_VERSION,
            "mode": self.mode.value, "sources": sources,
        }
        snapshot["sha256"] = _hash(snapshot)
        return snapshot


def _admin_connection(path: Path) -> sqlite3.Connection:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(path)


def initialize_memory(
    basic_db: Path | None = None, learning_db: Path | None = None,
) -> dict[str, str]:
    """Explicit management operation. Does not seed memory or copy histories."""
    if basic_db is None and learning_db is None:
        raise ValueError("choose at least one memory database to initialize")
    if basic_db is not None and learning_db is not None:
        if Path(basic_db).resolve() == Path(learning_db).resolve():
            raise ValueError("basic and learning memory require physically separate databases")
    result: dict[str, str] = {}
    if basic_db is not None:
        connection = _admin_connection(basic_db)
        try:
            tables = {row[0] for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )}
            if tables and "basic_memory" not in tables:
                raise ValueError("existing database is not EVAgent basic memory; use explicit import")
            connection.execute(
                "CREATE TABLE IF NOT EXISTS basic_memory ("
                "singleton INTEGER PRIMARY KEY CHECK(singleton=1), "
                "text TEXT NOT NULL, version INTEGER NOT NULL)"
            )
            connection.commit()
            result["basic_db"] = str(Path(basic_db).resolve())
        finally:
            connection.close()
    if learning_db is not None:
        connection = _admin_connection(learning_db)
        try:
            tables = {row[0] for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )}
            if tables and not {"episode_lessons", "knowledge_cards"}.issubset(tables):
                raise ValueError("existing database is not learning memory")
            LearningMemoryStore(connection, Path(learning_db)).setup()
            result["learning_db"] = str(Path(learning_db).resolve())
        finally:
            connection.close()
    return result


def write_basic_memory(path: Path, text: str) -> dict[str, Any]:
    """Explicitly replace basic memory; inference still validates against assay IDs."""
    normalized = validate_strategy_memory(text, forbidden_assays=[])
    initialize_memory(basic_db=path)
    connection = _admin_connection(path)
    try:
        connection.execute(
            "INSERT INTO basic_memory(singleton,text,version) VALUES (1,?,1) "
            "ON CONFLICT(singleton) DO UPDATE SET text=excluded.text,version=basic_memory.version+1",
            (normalized,),
        )
        connection.commit()
    finally:
        connection.close()
    return MemoryProvider(mode="basic", basic_db=path).snapshot()


def import_legacy_basic_memory(source_db: Path, target_db: Path) -> dict[str, Any]:
    """Explicitly import one LangGraph SqliteStore AGENTS.md into separate storage.

    No Deep Agents or LangGraph dependency is loaded. Ambiguous stores are
    rejected rather than importing another namespace or overwriting the source.
    """
    if Path(source_db).resolve() == Path(target_db).resolve():
        raise ValueError("legacy source and target database must be different")
    with _read_database(source_db) as (connection, source):
        columns = {row[1] for row in connection.execute("PRAGMA table_info(store)")}
        if not {"prefix", "key", "value"}.issubset(columns):
            raise ValueError("unsupported legacy SqliteStore schema; expected store(prefix,key,value)")
        rows = connection.execute(
            'SELECT prefix,key,value FROM store WHERE key IN (?,?)',
            (MEMORY_KEY, LEGACY_MEMORY_KEY),
        ).fetchall()
        namespace = MEMORY_NAMESPACE[0]
        rows = [row for row in rows if str(row[0]) in {
            namespace, json.dumps([namespace]), json.dumps((namespace,)),
        }]
        current = [row for row in rows if row[1] == MEMORY_KEY]
        selected = current or rows
        if len(selected) != 1:
            raise ValueError("legacy store must contain exactly one matching AGENTS.md")
        value = json.loads(selected[0][2])
        content = value.get("content", "")
        text = "\n".join(content) if isinstance(content, list) else str(content)
    snapshot = write_basic_memory(target_db, text)
    return {"imported_from": source, "snapshot": snapshot}


def inspect_memory(
    basic_db: Path | None = None, learning_db: Path | None = None,
) -> dict[str, Any]:
    """Read-only administration summary; does not reveal memory text."""
    result: dict[str, Any] = {}
    if basic_db is not None:
        snap = MemoryProvider(mode="basic", basic_db=basic_db).snapshot()
        result["basic"] = {"sha256": snap["sha256"], "sources": snap["sources"]}
    if learning_db is not None:
        with _read_database(learning_db) as (connection, source):
            result["learning"] = {
                **LearningMemoryStore(connection, Path(learning_db)).summary(), **source,
            }
    return result
