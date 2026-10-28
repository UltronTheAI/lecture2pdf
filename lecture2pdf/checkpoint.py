"""SQLite-backed crash-safe checkpoints for resumable processing."""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


class CheckpointMismatchError(RuntimeError):
    """Raised when a cache belongs to a different input or configuration."""


@dataclass(frozen=True)
class Checkpoint:
    sequence: int
    stage: str
    payload: dict[str, Any]
    created_at: float


class CheckpointStore:
    """A small SQLite store with transactionally consistent latest checkpoints."""

    def __init__(self, database: str | Path):
        self.database = Path(database)
        self.database.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.database, timeout=30)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=FULL")
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS job_meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS checkpoints (
                sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                stage TEXT NOT NULL,
                payload TEXT NOT NULL,
                created_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS completed_items (
                item_key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                completed_at REAL NOT NULL
            );
            """
        )
        self.connection.commit()

    def initialize(self, *, video_fingerprint: str, config_fingerprint: str, job_id: str) -> None:
        expected = {
            "video_fingerprint": video_fingerprint,
            "config_fingerprint": config_fingerprint,
            "job_id": job_id,
        }
        existing = dict(self.connection.execute("SELECT key, value FROM job_meta"))
        if existing and any(existing.get(key) not in (None, value) for key, value in expected.items()):
            raise CheckpointMismatchError(
                "Cache fingerprints do not match this video/configuration; use a new cache or clear it."
            )
        with self.connection:
            self.connection.executemany(
                "INSERT OR REPLACE INTO job_meta(key, value) VALUES (?, ?)", expected.items()
            )

    def metadata(self) -> dict[str, str]:
        return dict(self.connection.execute("SELECT key, value FROM job_meta"))

    def save(self, stage: str, payload: Mapping[str, Any]) -> Checkpoint:
        created_at = time.time()
        encoded = json.dumps(dict(payload), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        with self.connection:
            cursor = self.connection.execute(
                "INSERT INTO checkpoints(stage, payload, created_at) VALUES (?, ?, ?)",
                (stage, encoded, created_at),
            )
        return Checkpoint(cursor.lastrowid, stage, dict(payload), created_at)

    def latest(self, stage: str | None = None) -> Checkpoint | None:
        if stage is None:
            row = self.connection.execute(
                "SELECT sequence, stage, payload, created_at FROM checkpoints ORDER BY sequence DESC LIMIT 1"
            ).fetchone()
        else:
            row = self.connection.execute(
                "SELECT sequence, stage, payload, created_at FROM checkpoints WHERE stage = ? ORDER BY sequence DESC LIMIT 1",
                (stage,),
            ).fetchone()
        if row is None:
            return None
        return Checkpoint(row["sequence"], row["stage"], json.loads(row["payload"]), row["created_at"])

    def mark_completed(self, item_key: str, value: Any = True) -> None:
        encoded = json.dumps(value, ensure_ascii=False, sort_keys=True)
        with self.connection:
            self.connection.execute(
                "INSERT OR REPLACE INTO completed_items(item_key, value, completed_at) VALUES (?, ?, ?)",
                (item_key, encoded, time.time()),
            )

    def is_completed(self, item_key: str) -> bool:
        return self.connection.execute(
            "SELECT 1 FROM completed_items WHERE item_key = ?", (item_key,)
        ).fetchone() is not None

    def completed(self) -> dict[str, Any]:
        return {
            row["item_key"]: json.loads(row["value"])
            for row in self.connection.execute("SELECT item_key, value FROM completed_items")
        }

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "CheckpointStore":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
