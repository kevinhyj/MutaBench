"""Persistent storage for the evidence-gated learning-memory module.

This database is intentionally independent from Deep Agents' ``AGENTS.md``
store.  Raw conversational preferences can therefore never be mistaken for a
validated knowledge card, and disabling learning memory means this database is
not opened at all.
"""

# Adapted from ypxia/src/rnagym_deep_agent/learning_store.py; see provenance manifest.

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .learning_memory import EpisodeLesson, KnowledgeCard, TaskContext


class LearningMemoryStore:
    """Small, auditable SQLite registry for lessons and versioned cards."""

    def __init__(self, connection: sqlite3.Connection, path: Path) -> None:
        self.connection = connection
        self.path = path

    def setup(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS episode_lessons (
                episode_id TEXT PRIMARY KEY,
                payload_json TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS knowledge_cards (
                card_id TEXT NOT NULL,
                version INTEGER NOT NULL,
                status TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (card_id, version)
            );
            CREATE INDEX IF NOT EXISTS knowledge_cards_status
                ON knowledge_cards(status);
            """
        )
        self.connection.commit()

    def put_lesson(self, lesson: EpisodeLesson) -> None:
        payload = json.dumps(asdict(lesson), ensure_ascii=False, sort_keys=True)
        self.connection.execute(
            """
            INSERT INTO episode_lessons(episode_id, payload_json)
            VALUES (?, ?)
            ON CONFLICT(episode_id) DO UPDATE SET payload_json=excluded.payload_json
            """,
            (lesson.episode_id, payload),
        )
        self.connection.commit()

    def list_lessons(self) -> list[EpisodeLesson]:
        rows = self.connection.execute(
            "SELECT payload_json FROM episode_lessons ORDER BY episode_id"
        ).fetchall()
        return [_lesson_from_payload(json.loads(str(row[0]))) for row in rows]

    def put_card(self, card: KnowledgeCard) -> None:
        payload = json.dumps(asdict(card), ensure_ascii=False, sort_keys=True)
        self.connection.execute(
            """
            INSERT INTO knowledge_cards(card_id, version, status, payload_json)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(card_id, version) DO UPDATE SET
                status=excluded.status,
                payload_json=excluded.payload_json
            """,
            (card.card_id, card.version, card.status, payload),
        )
        self.connection.commit()

    def list_cards(
        self, *, status: str | None = None, latest_only: bool = True
    ) -> list[KnowledgeCard]:
        where = "WHERE c.status = ?" if status is not None else ""
        latest = (
            "JOIN (SELECT card_id, MAX(version) AS version FROM knowledge_cards "
            "GROUP BY card_id) latest "
            "ON c.card_id = latest.card_id AND c.version = latest.version"
            if latest_only
            else ""
        )
        parameters: tuple[str, ...] = (status,) if status is not None else ()
        rows = self.connection.execute(
            f"SELECT c.payload_json FROM knowledge_cards c {latest} {where} "
            "ORDER BY c.card_id, c.version",
            parameters,
        ).fetchall()
        return [_card_from_payload(json.loads(str(row[0]))) for row in rows]

    def summary(self) -> dict[str, Any]:
        lesson_count = int(
            self.connection.execute("SELECT COUNT(*) FROM episode_lessons").fetchone()[0]
        )
        rows = self.connection.execute(
            "SELECT status, COUNT(*) FROM knowledge_cards GROUP BY status ORDER BY status"
        ).fetchall()
        return {
            "path": str(self.path),
            "episode_lessons": lesson_count,
            "card_versions": {str(status): int(count) for status, count in rows},
            "active_latest": len(self.list_cards(status="active")),
        }


def _lesson_from_payload(value: dict[str, Any]) -> EpisodeLesson:
    context = TaskContext(**value.pop("context"))
    value["context"] = context
    value["tags"] = tuple(value["tags"])
    return EpisodeLesson(**value)


def _card_from_payload(value: dict[str, Any]) -> KnowledgeCard:
    value["tags"] = tuple(value["tags"])
    value["source_episode_ids"] = tuple(value["source_episode_ids"])
    return KnowledgeCard(**value)
