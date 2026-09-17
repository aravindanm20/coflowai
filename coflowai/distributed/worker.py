"""Workers and the execution service.

    API ──▶ ExecutionService ──▶ Queue ──▶ Worker(s) ──▶ Persistent State

The service submits work and returns immediately; workers lease jobs and drive
them through the ordinary :class:`~coflowai.runtime.runtime.Runtime`.  Because
the runtime is checkpointed and side effects are idempotent, at-least-once
delivery is safe: a redelivered job resumes rather than duplicating work.
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import Any, Callable, Iterable

from ..core.exceptions import ConfigurationError, ExecutionError
from ..core.executable import Executable
from ..core.ids import execution_id as new_execution_id
from ..core.result import ExecutionResult
from ..core.types import ExecutionStatus
from ..observability.logger import get_logger
from ..policies.policy import ExecutionPolicy
from ..runtime.runtime import Runtime
from ..state.checkpoint import ExecutionRecord
from .queue import Job, JobQueue

__all__ = ["ExecutableRegistry", "Worker", "WorkerPool", "ExecutionService"]

_log = get_logger("worker")


class ExecutableRegistry:
    """Maps logical names to executables.

    A queue can only carry a *name*; every worker must be able to resolve that
    name to the same agent or workflow.  Registering the same name with a
    different definition is rejected, because it would silently break resume.
    """

    def __init__(self) -> None:
        self._items: dict[str, Executable] = {}

    def register(self, executable: Executable, *, name: str | None = None
                 ) -> Executable:
        key = name or executable.name
        existing = self._items.get(key)
        if existing is not None and existing.signature() != executable.signature():
            raise ConfigurationError(
                f"'{key}' is already registered with a different definition; "
                "resuming across definitions is unsafe",
                name=key,
            )
        self._items[key] = executable
        return executable

    def get(self, name: str) -> Executable:
        try:
            return self._items[name]
        except KeyError as exc:
            raise ConfigurationError(f"executable '{name}' is not registered "
                                     "on this worker",
                                     available=sorted(self._items)) from exc

    def names(self) -> list[str]:
        return sorted(self._items)


class Worker:
    """Leases jobs and executes them on a shared runtime."""

    def __init__(self, runtime: Runtime, queue: JobQueue,
                 executables: ExecutableRegistry, *, name: str = "worker-1",
                 poll_timeout: float = 1.0,
                 on_result: Callable[[Job, ExecutionResult], Any] | None = None
                 ) -> None:
        self.runtime = runtime
        self.queue = queue
        self.executables = executables
        self.name = name
        self.poll_timeout = poll_timeout
        self.on_result = on_result
        self.processed = 0
        self.failed = 0
        self._stopping = asyncio.Event()

    # ------------------------------------------------------------------ loop
    async def run_forever(self) -> None:
        _log.info("worker.started", worker=self.name,
                  executables=self.executables.names())
        while not self._stopping.is_set():
            await self.run_once()
        _log.info("worker.stopped", worker=self.name, processed=self.processed)

    async def run_once(self) -> Job | None:
        """Lease and process a single job.  Returns ``None`` when idle."""
        job = await self.queue.dequeue(timeout=self.poll_timeout)
        if job is None:
            return None
        try:
            result = await self._execute(job)
            await self.queue.ack(job)
            self.processed += 1
            if self.on_result is not None:
                outcome = self.on_result(job, result)
                if hasattr(outcome, "__await__"):
                    await outcome
        except asyncio.CancelledError:
            await self.queue.nack(job, requeue=True)
            raise
        except Exception as exc:
            self.failed += 1
            _log.warning("worker.job_failed", worker=self.name, job_id=job.job_id,
                         kind=job.kind, execution_id=job.execution_id,
                         error=repr(exc))
            # Config errors are permanent — requeuing would just spin.
            await self.queue.nack(job,
                                  requeue=not isinstance(exc, ConfigurationError))
        return job

    def stop(self) -> None:
        self._stopping.set()

    # -------------------------------------------------------------- dispatch
    async def _execute(self, job: Job) -> ExecutionResult:
        executable = self.executables.get(job.executable) if job.executable else None
        policy = ExecutionPolicy(**job.policy) if job.policy else None

        if job.kind == "run":
            if executable is None:
                raise ConfigurationError("run job requires an executable name")
            return await self.runtime.run(executable, job.input, policy=policy,
                                          execution_id=job.execution_id,
                                          metadata=job.metadata)
        if job.kind == "resume":
            return await self.runtime.resume(job.execution_id,
                                             executable=executable, policy=policy)
        if job.kind == "approve":
            return await self.runtime.approve(
                job.execution_id, executable=executable,
                approval_id=job.metadata.get("approval_id"),
                by=job.metadata.get("by"), note=job.metadata.get("note"))
        if job.kind == "reject":
            return await self.runtime.reject(
                job.execution_id, executable=executable,
                approval_id=job.metadata.get("approval_id"),
                by=job.metadata.get("by"), note=job.metadata.get("note"))
        raise ExecutionError(f"unknown job kind '{job.kind}'", job_id=job.job_id)


class WorkerPool:
    """Runs N workers concurrently against one queue."""

    def __init__(self, runtime: Runtime, queue: JobQueue,
                 executables: ExecutableRegistry, *, size: int = 4,
                 name: str = "pool") -> None:
        self.workers = [
            Worker(runtime, queue, executables, name=f"{name}-{index + 1}")
            for index in range(size)
        ]
        self._tasks: list[asyncio.Task[None]] = []

    async def start(self) -> None:
        self._tasks = [asyncio.create_task(worker.run_forever())
                       for worker in self.workers]

    async def stop(self, *, drain_timeout: float = 5.0) -> None:
        for worker in self.workers:
            worker.stop()
        if not self._tasks:
            return
        done, pending = await asyncio.wait(self._tasks, timeout=drain_timeout)
        for task in pending:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        self._tasks.clear()

    async def __aenter__(self) -> "WorkerPool":
        await self.start()
        return self

    async def __aexit__(self, *exc_info: Any) -> None:
        await self.stop()

    @property
    def processed(self) -> int:
        return sum(worker.processed for worker in self.workers)


class ExecutionService:
    """Front door: accepts work, persists intent, returns an execution id.

    The caller gets an id immediately; workers do the work. This is what makes
    long-running and background agents practical.
    """

    def __init__(self, queue: JobQueue, runtime: Runtime,
                 executables: ExecutableRegistry | None = None) -> None:
        self.queue = queue
        self.runtime = runtime
        self.executables = executables or ExecutableRegistry()

    def register(self, executable: Executable, *, name: str | None = None
                 ) -> Executable:
        return self.executables.register(executable, name=name)

    async def submit(self, executable: Executable | str, input: Any = None, *,
                     policy: ExecutionPolicy | None = None,
                     metadata: dict[str, Any] | None = None) -> str:
        """Enqueue a new execution and return its id immediately."""
        name = executable if isinstance(executable, str) else executable.name
        if not isinstance(executable, str):
            self.executables.register(executable, name=name)

        execution_id = new_execution_id()
        # Persist intent before queueing, so a crash between the two is visible.
        await self.runtime.state_store.save_execution(ExecutionRecord(
            execution_id=execution_id, status=ExecutionStatus.CREATED,
            input=input, executable=name,
            policy=policy.to_dict() if policy else {},
            metadata=dict(metadata or {}),
        ))
        await self.queue.enqueue(Job(
            kind="run", execution_id=execution_id, executable=name, input=input,
            policy=policy.to_dict() if policy else {},
            metadata=dict(metadata or {}),
        ))
        return execution_id

    async def submit_resume(self, execution_id: str, *,
                            executable: str | None = None) -> str:
        record = await self.runtime.state_store.get_execution(execution_id)
        name = executable or (record.executable if record else "")
        await self.queue.enqueue(Job(kind="resume", execution_id=execution_id,
                                     executable=name))
        return execution_id

    async def submit_decision(self, execution_id: str, *, approved: bool,
                              approval_id: str | None = None,
                              by: str | None = None,
                              note: str | None = None) -> str:
        record = await self.runtime.state_store.get_execution(execution_id)
        await self.queue.enqueue(Job(
            kind="approve" if approved else "reject", execution_id=execution_id,
            executable=record.executable if record else "",
            metadata={"approval_id": approval_id, "by": by, "note": note},
        ))
        return execution_id

    async def status(self, execution_id: str) -> ExecutionRecord | None:
        return await self.runtime.state_store.get_execution(execution_id)

    async def wait(self, execution_id: str, *, timeout: float = 30.0,
                   poll_interval: float = 0.05) -> ExecutionRecord:
        """Poll until the execution reaches a terminal or paused state."""
        from ..core.types import TERMINAL_STATUSES

        deadline = asyncio.get_running_loop().time() + timeout
        settled = TERMINAL_STATUSES | {ExecutionStatus.WAITING_FOR_APPROVAL}
        while True:
            record = await self.runtime.state_store.get_execution(execution_id)
            if record is not None and record.status in settled:
                return record
            if asyncio.get_running_loop().time() > deadline:
                raise ExecutionError("timed out waiting for execution",
                                     execution_id=execution_id)
            await asyncio.sleep(poll_interval)

    async def pending(self) -> int:
        return await self.queue.depth()


def workers_for(runtime: Runtime, queue: JobQueue,
                executables: Iterable[Executable], *, size: int = 4) -> WorkerPool:
    """Convenience: build a pool that knows how to run ``executables``."""
    registry = ExecutableRegistry()
    for executable in executables:
        registry.register(executable)
    return WorkerPool(runtime, queue, registry, size=size)
