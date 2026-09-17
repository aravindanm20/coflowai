"""Distributed execution: queues, workers and the execution service.

    API ─▶ ExecutionService ─▶ Queue ─▶ Worker(s) ─▶ Persistent State

Nothing here is required for local execution; `Runtime.run()` works in-process.
"""

from .queue import InMemoryJobQueue, Job, JobQueue, RedisJobQueue
from .worker import (
    ExecutableRegistry,
    ExecutionService,
    Worker,
    WorkerPool,
    workers_for,
)

__all__ = ["Job", "JobQueue", "InMemoryJobQueue", "RedisJobQueue", "Worker",
           "WorkerPool", "ExecutionService", "ExecutableRegistry", "workers_for"]
