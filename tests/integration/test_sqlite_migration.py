"""SQLite stores must read rows correctly and upgrade pre-1.0.1 databases."""
from __future__ import annotations

import sqlite3

from coflowai.core.types import ExecutionStatus
from coflowai.persistence.sqlite import SqliteStateStore
from coflowai.state.checkpoint import ExecutionRecord


async def test_record_roundtrip_includes_permissions(tmp_path):
    store = SqliteStateStore(tmp_path / "s.db")
    await store.save_execution(ExecutionRecord(execution_id="e1",
                                               permissions=["db.read"]))
    record = await store.get_execution("e1")
    assert record.permissions == ["db.read"]
    assert record.status is ExecutionStatus.CREATED


async def test_old_database_without_permissions_column_is_migrated(tmp_path):
    path = tmp_path / "old.db"
    legacy = sqlite3.connect(path)
    legacy.execute(
        "CREATE TABLE executions (execution_id TEXT PRIMARY KEY, status TEXT NOT NULL,"
        " executable TEXT, workflow_hash TEXT, input TEXT, output TEXT, error TEXT,"
        " policy TEXT, metadata TEXT, parent_execution_id TEXT,"
        " created_at TEXT NOT NULL, updated_at TEXT NOT NULL)")
    legacy.commit()
    legacy.close()
    store = SqliteStateStore(path)
    await store.save_execution(ExecutionRecord(execution_id="e1",
                                               permissions=["x.y"]))
    assert (await store.get_execution("e1")).permissions == ["x.y"]
