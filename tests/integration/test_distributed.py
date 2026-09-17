"""Distributed execution: queues, workers, the execution service, at-least-once."""

from __future__ import annotations

import asyncio

import pytest

from coflowai import (
    Agent,
    CoFlowAi,
    ConfigurationError,
    ExecutionPolicy,
    ExecutionStatus,
    Workflow,
)
from coflowai.distributed import (
    ExecutableRegistry,
    ExecutionService,
    InMemoryJobQueue,
    Job,
    Worker,
    WorkerPool,
)
from coflowai.persistence.sqlite import SqliteEventStore, SqliteStateStore
from coflowai.testing.fakes import FakeModelProvider


def build(tmp_path, responses, *, repeat=True):
    path = tmp_path / "dist.db"
    app = CoFlowAi(event_store=SqliteEventStore(path),
                   state_store=SqliteStateStore(path),
                   policy=ExecutionPolicy(max_steps=6, retry_attempts=0))
    provider = FakeModelProvider(responses, repeat_last=repeat)
    app.models.register("fake-model", provider, default=True)
    return app, provider


async def test_service_submits_and_worker_executes(tmp_path):
    app, _ = build(tmp_path, ["worker did it"])
    queue = InMemoryJobQueue()
    service = ExecutionService(queue, app.runtime)
    agent = Agent("assistant", "Answer.", "fake-model")
    service.register(agent)

    execution_id = await service.submit(agent, "question")
    assert execution_id.startswith("exec_")
    assert await service.pending() == 1

    # the API caller already has an id before any work happened
    record = await service.status(execution_id)
    assert record.status is ExecutionStatus.CREATED

    worker = Worker(app.runtime, queue, service.executables)
    await worker.run_once()

    settled = await service.wait(execution_id, timeout=5)
    assert settled.status is ExecutionStatus.COMPLETED
    assert settled.output == "worker did it"
    assert worker.processed == 1


async def test_worker_pool_drains_many_jobs_concurrently(tmp_path):
    app, _ = build(tmp_path, ["done"])
    queue = InMemoryJobQueue()
    service = ExecutionService(queue, app.runtime)
    agent = Agent("assistant", "Answer.", "fake-model")
    service.register(agent)

    ids = [await service.submit(agent, f"question {i}") for i in range(12)]

    async with WorkerPool(app.runtime, queue, service.executables, size=4):
        for execution_id in ids:
            record = await service.wait(execution_id, timeout=10)
            assert record.status is ExecutionStatus.COMPLETED

    assert await queue.depth() == 0


async def test_redelivery_is_safe_because_execution_is_idempotent(tmp_path):
    """At-least-once delivery must not duplicate completed work."""
    app, provider = build(tmp_path, ["first", "second"], repeat=False)
    queue = InMemoryJobQueue()
    registry = ExecutableRegistry()
    workflow = Workflow("w").start(Agent("first", "i", "fake-model")) \
                            .then(Agent("second", "i", "fake-model"))
    registry.register(workflow)

    worker = Worker(app.runtime, queue, registry)
    await queue.enqueue(Job(kind="run", execution_id="exec_dup",
                            executable="w", input="go"))
    await worker.run_once()
    assert provider.call_count == 2

    # the same job is delivered again after an ack was lost
    await queue.enqueue(Job(kind="resume", execution_id="exec_dup",
                            executable="w"))
    await worker.run_once()
    # completed nodes were skipped, so no extra model calls were made
    assert provider.call_count == 2


async def test_failed_jobs_are_retried_then_dead_lettered(tmp_path):
    app, _ = build(tmp_path, ["x"])
    queue = InMemoryJobQueue(max_attempts=2)
    registry = ExecutableRegistry()          # deliberately empty -> resolution fails
    worker = Worker(app.runtime, queue, registry)

    await queue.enqueue(Job(kind="run", execution_id="exec_1",
                            executable="missing-agent", input="go"))
    await worker.run_once()
    # ConfigurationError is permanent: dead-lettered immediately, never retried
    assert len(queue.dead_letter) == 1
    assert worker.failed == 1


async def test_transient_job_failures_are_requeued(tmp_path):
    app, _ = build(tmp_path, ["x"])
    queue = InMemoryJobQueue(max_attempts=3)
    registry = ExecutableRegistry()
    registry.register(Agent("assistant", "Answer.", "fake-model"))
    worker = Worker(app.runtime, queue, registry)

    # 'resume' with no checkpoint raises a non-configuration error -> requeue
    await queue.enqueue(Job(kind="resume", execution_id="exec_unknown",
                            executable="assistant"))
    await worker.run_once()
    assert await queue.depth() == 1
    assert queue.dead_letter == []


async def test_human_approval_across_the_queue(tmp_path):
    app, _ = build(tmp_path, ["deployed"])
    queue = InMemoryJobQueue()
    service = ExecutionService(queue, app.runtime)
    workflow = (Workflow("deploy")
                .approve("Ship it?", name="gate")
                .then(Agent("deployer", "Deploy.", "fake-model")))
    service.register(workflow)

    execution_id = await service.submit(workflow, "release 2.0")
    worker = Worker(app.runtime, queue, service.executables)
    await worker.run_once()

    record = await service.status(execution_id)
    assert record.status is ExecutionStatus.WAITING_FOR_APPROVAL

    # an operator approves via the API; a worker applies the decision
    await service.submit_decision(execution_id, approved=True, by="aravindan")
    await worker.run_once()

    final = await service.wait(execution_id, timeout=5)
    assert final.status is ExecutionStatus.COMPLETED
    assert final.output == "deployed"


async def test_registry_rejects_conflicting_definitions():
    """Two workers must never disagree about what a name means."""
    registry = ExecutableRegistry()
    registry.register(Agent("assistant", "Version one.", "fake-model"))
    with pytest.raises(ConfigurationError):
        registry.register(Agent("assistant", "Version two.", "fake-model"))


async def test_worker_stops_cleanly(tmp_path):
    app, _ = build(tmp_path, ["x"])
    queue = InMemoryJobQueue()
    worker = Worker(app.runtime, queue, ExecutableRegistry(), poll_timeout=0.01)
    task = asyncio.create_task(worker.run_forever())
    await asyncio.sleep(0.05)
    worker.stop()
    await asyncio.wait_for(task, timeout=2)


async def test_queue_roundtrip_serialisation():
    queue = InMemoryJobQueue()
    original = Job(kind="run", execution_id="exec_1", executable="w",
                   input={"a": [1, 2]}, metadata={"by": "x"})
    await queue.enqueue(original)
    leased = await queue.dequeue(timeout=0.1)
    assert leased.input == {"a": [1, 2]}
    assert leased.attempts == 1
    assert Job.from_dict(leased.to_dict()).job_id == leased.job_id
    await queue.ack(leased)
    assert await queue.depth() == 0


async def test_empty_queue_returns_none_without_blocking_forever():
    queue = InMemoryJobQueue()
    assert await queue.dequeue(timeout=0.01) is None
