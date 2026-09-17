"""SQLite-backed event and state stores.

``sqlite3`` is in the standard library, so this gives real durability with
**zero** extra dependencies — the right default for single-node deployments,
local development and integration tests.

All blocking calls are dispatched to a worker thread, so the event loop is never
blocked.  A single connection with WAL journaling and a serialising lock gives
correct behaviour for concurrent tasks in one process.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..core.types import ExecutionStatus
from ..events.event import Event
from ..events.store import EventStore
from ..state.checkpoint import Checkpoint, ExecutionRecord
from ..state.store import StateStore

__all__ = ["SqliteEventStore", "SqliteStateStore", "open_database"]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    event_id      TEXT PRIMARY KEY,
    execution_id  TEXT NOT NULL,
    sequence      INTEGER NOT NULL,
    type          TEXT NOT NULL,
    timestamp     TEXT NOT NULL,
    step          INTEGER NOT NULL DEFAULT 0,
    node_id       TEXT,
    trace_id      TEXT,
    payload       TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_events_execution
    ON events (execution_id, sequence);

CREATE TABLE IF NOT EXISTS executions (
    execution_id        TEXT PRIMARY KEY,
    status              TEXT NOT NULL,
    executable          TEXT,
    workflow_hash       TEXT,
    input               TEXT,
    output              TEXT,
    error               TEXT,
    policy              TEXT,
    metadata            TEXT,
    parent_execution_id TEXT,
    created_at          TEXT NOT NULL,
    updated_at          TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_executions_status
    ON executions (status, created_at DESC);

CREATE TABLE IF NOT EXISTS checkpoints (
    checkpoint_id  TEXT PRIMARY KEY,
    execution_id   TEXT NOT NULL,
    seq            INTEGER NOT NULL,
    step           INTEGER NOT NULL,
    node_id        TEXT,
    state          TEXT NOT NULL,
    model_calls    INTEGER NOT NULL DEFAULT 0,
    tool_calls     INTEGER NOT NULL DEFAULT 0,
    input_tokens   INTEGER NOT NULL DEFAULT 0,
    output_tokens  INTEGER NOT NULL DEFAULT 0,
    cost           REAL NOT NULL DEFAULT 0,
    status         TEXT NOT NULL,
    workflow_hash  TEXT,
    reason         TEXT,
    created_at     TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_checkpoints_execution
    ON checkpoints (execution_id, seq);

CREATE TABLE IF NOT EXISTS locks (
    execution_id TEXT PRIMARY KEY,
    owner        TEXT NOT NULL,
    expires_at   REAL NOT NULL
);
"""


def open_database(path: str | Path) -> sqlite3.Connection:
    """Open a WAL-mode connection usable from worker threads."""
    target = Path(path)
    if target.parent and str(target.parent) not in ("", "."):
        target.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(target), check_same_thread=False,
                                 isolation_level=None)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute("PRAGMA busy_timeout=5000")
    connection.executescript(_SCHEMA)
    return connection


class _SqliteBase:
    def __init__(self, path: str | Path = "coflowai.db",
                 connection: sqlite3.Connection | None = None) -> None:
        self._connection = connection or open_database(path)
        self._lock = asyncio.Lock()

    async def _run(self, func, *args: Any) -> Any:
        async with self._lock:
            return await asyncio.to_thread(func, *args)

    async def close(self) -> None:
        await asyncio.to_thread(self._connection.close)


class SqliteEventStore(_SqliteBase, EventStore):
    """Durable append-only event log."""

    async def append(self, event: Event) -> Event:
        def write() -> Event:
            cursor = self._connection.execute(
                "SELECT COALESCE(MAX(sequence), 0) AS seq FROM events "
                "WHERE execution_id = ?", (event.execution_id,))
            event.sequence = cursor.fetchone()["seq"] + 1
            self._connection.execute(
                "INSERT INTO events (event_id, execution_id, sequence, type, "
                "timestamp, step, node_id, trace_id, payload) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (event.event_id, event.execution_id, event.sequence, event.type,
                 event.timestamp.isoformat(), event.step, event.node_id,
                 event.trace_id, json.dumps(event.payload, default=str)))
            return event

        return await self._run(write)

    async def get_events(self, execution_id: str) -> list[Event]:
        def read() -> list[Event]:
            rows = self._connection.execute(
                "SELECT * FROM events WHERE execution_id = ? ORDER BY sequence",
                (execution_id,)).fetchall()
            return [_row_to_event(row) for row in rows]

        return await self._run(read)

    async def list_executions(self) -> list[str]:
        def read() -> list[str]:
            rows = self._connection.execute(
                "SELECT DISTINCT execution_id FROM events").fetchall()
            return [row["execution_id"] for row in rows]

        return await self._run(read)


class SqliteStateStore(_SqliteBase, StateStore):
    """Durable execution records, checkpoints and lock leases."""

    def __init__(self, path: str | Path = "coflowai.db",
                 connection: sqlite3.Connection | None = None,
                 owner: str = "worker") -> None:
        super().__init__(path, connection)
        self.owner = owner

    # ------------------------------------------------------------- executions
    async def save_execution(self, record: ExecutionRecord) -> None:
        def write() -> None:
            self._connection.execute(
                "INSERT INTO executions (execution_id, status, executable, "
                "workflow_hash, input, output, error, policy, metadata, "
                "parent_execution_id, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(execution_id) DO UPDATE SET "
                "status=excluded.status, output=excluded.output, "
                "error=excluded.error, metadata=excluded.metadata, "
                "updated_at=excluded.updated_at",
                (record.execution_id, record.status.value, record.executable,
                 record.workflow_hash, _dump(record.input), _dump(record.output),
                 record.error, _dump(record.policy), _dump(record.metadata),
                 record.parent_execution_id, record.created_at.isoformat(),
                 record.updated_at.isoformat()))

        await self._run(write)

    async def get_execution(self, execution_id: str) -> ExecutionRecord | None:
        def read() -> ExecutionRecord | None:
            row = self._connection.execute(
                "SELECT * FROM executions WHERE execution_id = ?",
                (execution_id,)).fetchone()
            return _row_to_record(row) if row else None

        return await self._run(read)

    async def list_executions(self, *, status: str | None = None,
                              limit: int = 100) -> list[ExecutionRecord]:
        def read() -> list[ExecutionRecord]:
            if status:
                rows = self._connection.execute(
                    "SELECT * FROM executions WHERE status = ? "
                    "ORDER BY created_at DESC LIMIT ?", (status, limit)).fetchall()
            else:
                rows = self._connection.execute(
                    "SELECT * FROM executions ORDER BY created_at DESC LIMIT ?",
                    (limit,)).fetchall()
            return [_row_to_record(row) for row in rows]

        return await self._run(read)

    # ------------------------------------------------------------ checkpoints
    async def save_checkpoint(self, checkpoint: Checkpoint) -> Checkpoint:
        def write() -> Checkpoint:
            cursor = self._connection.execute(
                "SELECT COALESCE(MAX(seq), 0) AS seq FROM checkpoints "
                "WHERE execution_id = ?", (checkpoint.execution_id,))
            sequence = cursor.fetchone()["seq"] + 1
            self._connection.execute(
                "INSERT OR REPLACE INTO checkpoints (checkpoint_id, execution_id, "
                "seq, step, node_id, state, model_calls, tool_calls, input_tokens, "
                "output_tokens, cost, status, workflow_hash, reason, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (checkpoint.checkpoint_id, checkpoint.execution_id, sequence,
                 checkpoint.step, checkpoint.node_id,
                 json.dumps(checkpoint.state, default=str), checkpoint.model_calls,
                 checkpoint.tool_calls, checkpoint.input_tokens,
                 checkpoint.output_tokens, checkpoint.cost,
                 checkpoint.status.value, checkpoint.workflow_hash,
                 checkpoint.reason, checkpoint.created_at.isoformat()))
            return checkpoint

        return await self._run(write)

    async def list_checkpoints(self, execution_id: str) -> list[Checkpoint]:
        def read() -> list[Checkpoint]:
            rows = self._connection.execute(
                "SELECT * FROM checkpoints WHERE execution_id = ? ORDER BY seq",
                (execution_id,)).fetchall()
            return [_row_to_checkpoint(row) for row in rows]

        return await self._run(read)

    # ------------------------------------------------------------------ locks
    async def acquire_lock(self, execution_id: str, *, ttl: float = 60.0) -> bool:
        """Best-effort lease so two workers never drive one execution."""
        def write() -> bool:
            now = time.time()
            self._connection.execute("DELETE FROM locks WHERE expires_at < ?", (now,))
            try:
                self._connection.execute(
                    "INSERT INTO locks (execution_id, owner, expires_at) "
                    "VALUES (?, ?, ?)", (execution_id, self.owner, now + ttl))
                return True
            except sqlite3.IntegrityError:
                return False

        return await self._run(write)

    async def release_lock(self, execution_id: str) -> None:
        def write() -> None:
            self._connection.execute(
                "DELETE FROM locks WHERE execution_id = ? AND owner = ?",
                (execution_id, self.owner))

        await self._run(write)


# --------------------------------------------------------------------------- #
# row mapping
# --------------------------------------------------------------------------- #
def _dump(value: Any) -> str | None:
    if value is None:
        return None
    try:
        return json.dumps(value, default=str)
    except TypeError:  # pragma: no cover - exotic objects
        return json.dumps(repr(value))


def _load(value: str | None, default: Any = None) -> Any:
    if value is None:
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError:  # pragma: no cover - corrupted row
        return default


def _row_to_event(row: sqlite3.Row) -> Event:
    return Event(
        execution_id=row["execution_id"], type=row["type"],
        payload=_load(row["payload"], {}), event_id=row["event_id"],
        timestamp=datetime.fromisoformat(row["timestamp"]),
        sequence=row["sequence"], step=row["step"], node_id=row["node_id"],
        trace_id=row["trace_id"],
    )


def _row_to_record(row: sqlite3.Row) -> ExecutionRecord:
    record = ExecutionRecord(
        execution_id=row["execution_id"], status=ExecutionStatus(row["status"]),
        input=_load(row["input"]), output=_load(row["output"]),
        error=row["error"], executable=row["executable"] or "",
        workflow_hash=row["workflow_hash"], policy=_load(row["policy"], {}),
        metadata=_load(row["metadata"], {}),
        parent_execution_id=row["parent_execution_id"],
    )
    record.created_at = datetime.fromisoformat(row["created_at"])
    record.updated_at = datetime.fromisoformat(row["updated_at"])
    return record


def _row_to_checkpoint(row: sqlite3.Row) -> Checkpoint:
    return Checkpoint(
        execution_id=row["execution_id"], step=row["step"], node_id=row["node_id"],
        state=_load(row["state"], {}), model_calls=row["model_calls"],
        tool_calls=row["tool_calls"], input_tokens=row["input_tokens"],
        output_tokens=row["output_tokens"], cost=row["cost"],
        status=ExecutionStatus(row["status"]), workflow_hash=row["workflow_hash"],
        reason=row["reason"] or "auto", checkpoint_id=row["checkpoint_id"],
        created_at=datetime.fromisoformat(row["created_at"]).replace(
            tzinfo=timezone.utc) if "+" not in row["created_at"]
        else datetime.fromisoformat(row["created_at"]),
    )
