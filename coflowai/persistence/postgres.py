"""PostgreSQL-backed persistence (requires ``asyncpg``).

Designed for multi-worker deployments:

* append-only ``coflowai_events`` with a per-execution sequence assigned inside
  a transaction, so concurrent appends can't collide
* ``coflowai_locks`` implements a TTL lease with atomic steal-on-expiry, so a
  crashed worker's executions become claimable without manual intervention
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from typing import Any

from ..core.exceptions import ConfigurationError
from ..core.types import ExecutionStatus
from ..events.event import Event
from ..events.store import EventStore
from ..state.checkpoint import Checkpoint, ExecutionRecord
from ..state.store import StateStore

__all__ = ["PostgresEventStore", "PostgresStateStore", "create_pool", "SCHEMA"]

SCHEMA = """
CREATE TABLE IF NOT EXISTS coflowai_events (
    event_id      TEXT PRIMARY KEY,
    execution_id  TEXT NOT NULL,
    sequence      BIGINT NOT NULL,
    type          TEXT NOT NULL,
    timestamp     TIMESTAMPTZ NOT NULL,
    step          INTEGER NOT NULL DEFAULT 0,
    node_id       TEXT,
    trace_id      TEXT,
    payload       JSONB NOT NULL,
    UNIQUE (execution_id, sequence)
);
CREATE INDEX IF NOT EXISTS idx_coflowai_events_exec
    ON coflowai_events (execution_id, sequence);

CREATE TABLE IF NOT EXISTS coflowai_executions (
    execution_id        TEXT PRIMARY KEY,
    status              TEXT NOT NULL,
    executable          TEXT,
    workflow_hash       TEXT,
    input               JSONB,
    output              JSONB,
    error               TEXT,
    policy              JSONB,
    metadata            JSONB,
    parent_execution_id TEXT,
    created_at          TIMESTAMPTZ NOT NULL,
    updated_at          TIMESTAMPTZ NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_coflowai_executions_status
    ON coflowai_executions (status, created_at DESC);

CREATE TABLE IF NOT EXISTS coflowai_checkpoints (
    checkpoint_id  TEXT PRIMARY KEY,
    execution_id   TEXT NOT NULL,
    seq            BIGINT NOT NULL,
    step           INTEGER NOT NULL,
    node_id        TEXT,
    state          JSONB NOT NULL,
    model_calls    INTEGER NOT NULL DEFAULT 0,
    tool_calls     INTEGER NOT NULL DEFAULT 0,
    input_tokens   INTEGER NOT NULL DEFAULT 0,
    output_tokens  INTEGER NOT NULL DEFAULT 0,
    cost           DOUBLE PRECISION NOT NULL DEFAULT 0,
    status         TEXT NOT NULL,
    workflow_hash  TEXT,
    reason         TEXT,
    created_at     TIMESTAMPTZ NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_coflowai_checkpoints_exec
    ON coflowai_checkpoints (execution_id, seq);

CREATE TABLE IF NOT EXISTS coflowai_locks (
    execution_id TEXT PRIMARY KEY,
    owner        TEXT NOT NULL,
    expires_at   DOUBLE PRECISION NOT NULL
);
"""


async def create_pool(dsn: str, **kwargs: Any) -> Any:
    """Create an asyncpg pool and ensure the schema exists."""
    try:
        import asyncpg
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ConfigurationError(
            'asyncpg is required for Postgres: pip install "coflowai[postgres]"',
            cause=exc,
        ) from exc
    pool = await asyncpg.create_pool(dsn, **kwargs)
    async with pool.acquire() as connection:
        await connection.execute(SCHEMA)
    return pool


class PostgresEventStore(EventStore):
    def __init__(self, pool: Any) -> None:
        self.pool = pool

    async def append(self, event: Event) -> Event:
        async with self.pool.acquire() as connection:
            async with connection.transaction():
                sequence = await connection.fetchval(
                    "SELECT COALESCE(MAX(sequence), 0) + 1 FROM coflowai_events "
                    "WHERE execution_id = $1", event.execution_id)
                event.sequence = sequence
                await connection.execute(
                    "INSERT INTO coflowai_events (event_id, execution_id, sequence, "
                    "type, timestamp, step, node_id, trace_id, payload) "
                    "VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9::jsonb)",
                    event.event_id, event.execution_id, sequence, event.type,
                    event.timestamp, event.step, event.node_id, event.trace_id,
                    json.dumps(event.payload, default=str))
        return event

    async def get_events(self, execution_id: str) -> list[Event]:
        async with self.pool.acquire() as connection:
            rows = await connection.fetch(
                "SELECT * FROM coflowai_events WHERE execution_id = $1 "
                "ORDER BY sequence", execution_id)
        return [
            Event(execution_id=row["execution_id"], type=row["type"],
                  payload=_load(row["payload"], {}), event_id=row["event_id"],
                  timestamp=row["timestamp"], sequence=row["sequence"],
                  step=row["step"], node_id=row["node_id"],
                  trace_id=row["trace_id"])
            for row in rows
        ]

    async def list_executions(self) -> list[str]:
        async with self.pool.acquire() as connection:
            rows = await connection.fetch(
                "SELECT DISTINCT execution_id FROM coflowai_events")
        return [row["execution_id"] for row in rows]


class PostgresStateStore(StateStore):
    def __init__(self, pool: Any, *, owner: str = "worker") -> None:
        self.pool = pool
        self.owner = owner

    # ------------------------------------------------------------- executions
    async def save_execution(self, record: ExecutionRecord) -> None:
        async with self.pool.acquire() as connection:
            await connection.execute(
                "INSERT INTO coflowai_executions (execution_id, status, executable, "
                "workflow_hash, input, output, error, policy, metadata, "
                "parent_execution_id, created_at, updated_at) VALUES "
                "($1,$2,$3,$4,$5::jsonb,$6::jsonb,$7,$8::jsonb,$9::jsonb,$10,$11,$12) "
                "ON CONFLICT (execution_id) DO UPDATE SET status = EXCLUDED.status, "
                "output = EXCLUDED.output, error = EXCLUDED.error, "
                "metadata = EXCLUDED.metadata, updated_at = EXCLUDED.updated_at",
                record.execution_id, record.status.value, record.executable,
                record.workflow_hash, _dump(record.input), _dump(record.output),
                record.error, _dump(record.policy), _dump(record.metadata),
                record.parent_execution_id, record.created_at, record.updated_at)

    async def get_execution(self, execution_id: str) -> ExecutionRecord | None:
        async with self.pool.acquire() as connection:
            row = await connection.fetchrow(
                "SELECT * FROM coflowai_executions WHERE execution_id = $1",
                execution_id)
        return _to_record(row) if row else None

    async def list_executions(self, *, status: str | None = None,
                              limit: int = 100) -> list[ExecutionRecord]:
        async with self.pool.acquire() as connection:
            if status:
                rows = await connection.fetch(
                    "SELECT * FROM coflowai_executions WHERE status = $1 "
                    "ORDER BY created_at DESC LIMIT $2", status, limit)
            else:
                rows = await connection.fetch(
                    "SELECT * FROM coflowai_executions ORDER BY created_at DESC "
                    "LIMIT $1", limit)
        return [_to_record(row) for row in rows]

    # ------------------------------------------------------------ checkpoints
    async def save_checkpoint(self, checkpoint: Checkpoint) -> Checkpoint:
        async with self.pool.acquire() as connection:
            async with connection.transaction():
                sequence = await connection.fetchval(
                    "SELECT COALESCE(MAX(seq), 0) + 1 FROM coflowai_checkpoints "
                    "WHERE execution_id = $1", checkpoint.execution_id)
                await connection.execute(
                    "INSERT INTO coflowai_checkpoints (checkpoint_id, execution_id, "
                    "seq, step, node_id, state, model_calls, tool_calls, "
                    "input_tokens, output_tokens, cost, status, workflow_hash, "
                    "reason, created_at) VALUES ($1,$2,$3,$4,$5,$6::jsonb,$7,$8,$9,"
                    "$10,$11,$12,$13,$14,$15) ON CONFLICT (checkpoint_id) DO NOTHING",
                    checkpoint.checkpoint_id, checkpoint.execution_id, sequence,
                    checkpoint.step, checkpoint.node_id,
                    json.dumps(checkpoint.state, default=str),
                    checkpoint.model_calls, checkpoint.tool_calls,
                    checkpoint.input_tokens, checkpoint.output_tokens,
                    checkpoint.cost, checkpoint.status.value,
                    checkpoint.workflow_hash, checkpoint.reason,
                    checkpoint.created_at)
        return checkpoint

    async def list_checkpoints(self, execution_id: str) -> list[Checkpoint]:
        async with self.pool.acquire() as connection:
            rows = await connection.fetch(
                "SELECT * FROM coflowai_checkpoints WHERE execution_id = $1 "
                "ORDER BY seq", execution_id)
        return [_to_checkpoint(row) for row in rows]

    # ------------------------------------------------------------------ locks
    async def acquire_lock(self, execution_id: str, *, ttl: float = 60.0) -> bool:
        """Atomic lease: claim if free, or steal if the previous lease expired."""
        now = time.time()
        async with self.pool.acquire() as connection:
            claimed = await connection.fetchval(
                "INSERT INTO coflowai_locks (execution_id, owner, expires_at) "
                "VALUES ($1, $2, $3) ON CONFLICT (execution_id) DO UPDATE SET "
                "owner = EXCLUDED.owner, expires_at = EXCLUDED.expires_at "
                "WHERE coflowai_locks.expires_at < $4 "
                "RETURNING execution_id",
                execution_id, self.owner, now + ttl, now)
        return claimed is not None

    async def release_lock(self, execution_id: str) -> None:
        async with self.pool.acquire() as connection:
            await connection.execute(
                "DELETE FROM coflowai_locks WHERE execution_id = $1 AND owner = $2",
                execution_id, self.owner)


# --------------------------------------------------------------------------- #
# mapping
# --------------------------------------------------------------------------- #
def _dump(value: Any) -> str | None:
    return None if value is None else json.dumps(value, default=str)


def _load(value: Any, default: Any = None) -> Any:
    if value is None:
        return default
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):  # pragma: no cover
        return default


def _to_record(row: Any) -> ExecutionRecord:
    record = ExecutionRecord(
        execution_id=row["execution_id"], status=ExecutionStatus(row["status"]),
        input=_load(row["input"]), output=_load(row["output"]), error=row["error"],
        executable=row["executable"] or "", workflow_hash=row["workflow_hash"],
        policy=_load(row["policy"], {}), metadata=_load(row["metadata"], {}),
        parent_execution_id=row["parent_execution_id"],
    )
    record.created_at = _as_datetime(row["created_at"])
    record.updated_at = _as_datetime(row["updated_at"])
    return record


def _to_checkpoint(row: Any) -> Checkpoint:
    return Checkpoint(
        execution_id=row["execution_id"], step=row["step"], node_id=row["node_id"],
        state=_load(row["state"], {}), model_calls=row["model_calls"],
        tool_calls=row["tool_calls"], input_tokens=row["input_tokens"],
        output_tokens=row["output_tokens"], cost=row["cost"],
        status=ExecutionStatus(row["status"]), workflow_hash=row["workflow_hash"],
        reason=row["reason"] or "auto", checkpoint_id=row["checkpoint_id"],
        created_at=_as_datetime(row["created_at"]),
    )


def _as_datetime(value: Any) -> datetime:
    return value if isinstance(value, datetime) else datetime.fromisoformat(str(value))
