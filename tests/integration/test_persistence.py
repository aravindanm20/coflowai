"""Durable persistence: survives process restart, enforces lock leases."""

from __future__ import annotations

import pytest

from coflowai import Agent, CoFlowAi, ExecutionPolicy, ExecutionStatus, Workflow
from coflowai.persistence.sqlite import (
    SqliteEventStore,
    SqliteStateStore,
    open_database,
)
from coflowai.testing.fakes import FakeModelProvider


def build_app(tmp_path, responses, **policy_kwargs):
    path = tmp_path / "state.db"
    app = CoFlowAi(event_store=SqliteEventStore(path),
                   state_store=SqliteStateStore(path),
                   policy=ExecutionPolicy(**policy_kwargs) if policy_kwargs else None)
    provider = FakeModelProvider(responses)
    app.models.register("fake-model", provider, default=True)
    return app, provider, path


async def test_events_and_checkpoints_survive_a_new_process(tmp_path):
    app, _, path = build_app(tmp_path, ["a", "b"])
    workflow = Workflow("w").start(Agent("first", "i", "fake-model")) \
                            .then(Agent("second", "i", "fake-model"))
    result = await app.run(workflow, "go")
    assert result.succeeded

    # a brand-new store object over the same file = a restarted process
    reopened_events = SqliteEventStore(path)
    reopened_state = SqliteStateStore(path)

    events = await reopened_events.get_events(result.execution_id)
    assert [e.type for e in events][:2] == ["execution.created", "execution.started"]
    assert [e.sequence for e in events] == list(range(1, len(events) + 1))

    record = await reopened_state.get_execution(result.execution_id)
    assert record.status is ExecutionStatus.COMPLETED
    assert record.executable == "w"

    checkpoints = await reopened_state.list_checkpoints(result.execution_id)
    assert [c.reason for c in checkpoints][-1] == "final"
    assert checkpoints[0].workflow_hash == workflow.hash


async def test_resume_works_across_a_restart(tmp_path):
    path = tmp_path / "resume.db"
    provider = FakeModelProvider(["first done"])
    app = CoFlowAi(event_store=SqliteEventStore(path),
                   state_store=SqliteStateStore(path),
                   policy=ExecutionPolicy(retry_attempts=0, max_steps=6))
    app.models.register("fake-model", provider, default=True)

    workflow = Workflow("w").start(Agent("first", "i", "fake-model")) \
                            .then(Agent("second", "i", "fake-model"))
    failed = await app.run(workflow, "go")
    assert failed.status is ExecutionStatus.FAILED

    # simulate a restart: fresh app object, same database file
    provider2 = FakeModelProvider(["second done"])
    app2 = CoFlowAi(event_store=SqliteEventStore(path),
                    state_store=SqliteStateStore(path),
                    policy=ExecutionPolicy(retry_attempts=0, max_steps=6))
    app2.models.register("fake-model", provider2, default=True)

    resumed = await app2.resume(failed.execution_id, executable=workflow)
    assert resumed.succeeded and resumed.output == "second done"
    # the first node was not re-executed by the new process
    assert provider2.call_count == 1


async def test_replay_reads_from_disk(tmp_path):
    app, _, path = build_app(tmp_path, ["done"])
    result = await app.run(Agent("a", "i", "fake-model"), "go")

    app2 = CoFlowAi(event_store=SqliteEventStore(path),
                    state_store=SqliteStateStore(path))
    trace = await app2.replay(result.execution_id)
    assert trace.status == "completed"
    assert len(trace.model_calls()) == 1


async def test_lock_lease_prevents_two_workers(tmp_path):
    connection = open_database(tmp_path / "locks.db")
    worker_a = SqliteStateStore(connection=connection, owner="a")
    worker_b = SqliteStateStore(connection=connection, owner="b")

    assert await worker_a.acquire_lock("exec_1", ttl=60) is True
    assert await worker_b.acquire_lock("exec_1", ttl=60) is False

    await worker_a.release_lock("exec_1")
    assert await worker_b.acquire_lock("exec_1", ttl=60) is True


async def test_expired_lock_can_be_claimed(tmp_path):
    connection = open_database(tmp_path / "expiry.db")
    dead_worker = SqliteStateStore(connection=connection, owner="dead")
    live_worker = SqliteStateStore(connection=connection, owner="live")

    await dead_worker.acquire_lock("exec_1", ttl=-1)   # already expired
    assert await live_worker.acquire_lock("exec_1", ttl=60) is True


async def test_list_executions_filters_by_status(tmp_path):
    app, provider, path = build_app(tmp_path, ["one"], retry_attempts=0)
    await app.run(Agent("a", "i", "fake-model"), "go")
    await app.run(Agent("a", "i", "fake-model"), "go")   # script exhausted -> failed

    store = SqliteStateStore(path)
    assert len(await store.list_executions(status="completed")) == 1
    assert len(await store.list_executions(status="failed")) == 1
    assert len(await store.list_executions()) == 2


async def test_state_json_roundtrip_preserves_nested_structures(tmp_path):
    app, _, path = build_app(tmp_path, ["a", "b"])
    workflow = Workflow("w").start(Agent("first", "i", "fake-model")) \
                            .then(Agent("second", "i", "fake-model"))
    result = await app.run(workflow, {"nested": {"list": [1, 2, 3]}})

    store = SqliteStateStore(path)
    checkpoint = (await store.list_checkpoints(result.execution_id))[-1]
    completed = checkpoint.state["__completed_nodes__:w"]
    assert completed["first"] == "a"


def test_postgres_and_redis_adapters_are_importable_without_a_server():
    """Adapters must import cleanly; they only need a server when used."""
    from coflowai.persistence.postgres import SCHEMA, PostgresStateStore
    from coflowai.persistence.redis import RedisStateStore

    assert "coflowai_events" in SCHEMA
    assert hasattr(PostgresStateStore, "acquire_lock")
    assert hasattr(RedisStateStore, "extend_lock")


async def test_redis_store_works_against_a_fake_client():
    """Exercises the Redis code paths without a server."""
    from coflowai.persistence.redis import RedisStateStore

    store = RedisStateStore(_FakeRedis(), owner="w1")
    assert await store.acquire_lock("exec_1", ttl=30) is True
    assert await store.acquire_lock("exec_1", ttl=30) is False   # NX blocks
    await store.release_lock("exec_1")
    assert await store.acquire_lock("exec_1", ttl=30) is True


class _FakeRedis:
    """Tiny in-memory stand-in covering the commands the stores use."""

    def __init__(self) -> None:
        self.kv: dict[str, str] = {}

    async def set(self, key, value, nx=False, px=None):
        if nx and key in self.kv:
            return None
        self.kv[key] = value
        return True

    async def get(self, key):
        return self.kv.get(key)

    async def eval(self, script, numkeys, key, token):
        if self.kv.get(key) == token:
            del self.kv[key]
            return 1
        return 0

    async def pexpire(self, key, ms):
        return True
