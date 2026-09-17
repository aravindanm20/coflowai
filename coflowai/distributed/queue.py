"""Work queue abstraction.

The runtime never assumes single-process execution: submitting work and doing
work are separate concerns joined by this interface.  Two implementations ship
in-tree — in-memory (development/tests) and Redis (production) — and the
interface is deliberately small enough to back with SQS, RabbitMQ or Kafka.

Delivery is at-least-once, which is safe because execution is idempotent:
checkpoints skip completed nodes and side-effect tools carry idempotency keys.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from ..core.exceptions import ConfigurationError

__all__ = ["Job", "JobQueue", "InMemoryJobQueue", "RedisJobQueue"]


@dataclass(slots=True)
class Job:
    """A unit of work: run, resume, or apply an approval decision."""

    kind: str                       # "run" | "resume" | "approve" | "reject"
    execution_id: str
    executable: str = ""            # logical name resolved by the worker
    input: Any = None
    policy: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    job_id: str = field(default_factory=lambda: f"job_{uuid.uuid4().hex[:12]}")
    attempts: int = 0
    enqueued_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id, "kind": self.kind,
            "execution_id": self.execution_id, "executable": self.executable,
            "input": self.input, "policy": self.policy, "metadata": self.metadata,
            "attempts": self.attempts, "enqueued_at": self.enqueued_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Job":
        return cls(**data)


class JobQueue(ABC):
    """Minimal queue contract: enqueue, lease, acknowledge, and dead-letter."""

    @abstractmethod
    async def enqueue(self, job: Job) -> Job:
        ...

    @abstractmethod
    async def dequeue(self, *, timeout: float = 1.0) -> Job | None:
        """Lease the next job, or return ``None`` when the queue stays empty."""

    @abstractmethod
    async def ack(self, job: Job) -> None:
        """Mark a leased job as finished so it is not redelivered."""

    @abstractmethod
    async def nack(self, job: Job, *, requeue: bool = True) -> None:
        """Return a job for retry, or send it to the dead-letter queue."""

    @abstractmethod
    async def depth(self) -> int:
        ...

    async def close(self) -> None:  # pragma: no cover - adapters may override
        return None


class InMemoryJobQueue(JobQueue):
    """Development/test queue.  Correct, but bounded to one process."""

    def __init__(self, *, max_attempts: int = 3) -> None:
        self._queue: asyncio.Queue[Job] = asyncio.Queue()
        self._in_flight: dict[str, Job] = {}
        self.dead_letter: list[Job] = []
        self.max_attempts = max_attempts

    async def enqueue(self, job: Job) -> Job:
        await self._queue.put(job)
        return job

    async def dequeue(self, *, timeout: float = 1.0) -> Job | None:
        try:
            job = await asyncio.wait_for(self._queue.get(), timeout=timeout)
        except TimeoutError:
            return None
        job.attempts += 1
        self._in_flight[job.job_id] = job
        return job

    async def ack(self, job: Job) -> None:
        self._in_flight.pop(job.job_id, None)

    async def nack(self, job: Job, *, requeue: bool = True) -> None:
        self._in_flight.pop(job.job_id, None)
        if requeue and job.attempts < self.max_attempts:
            await self._queue.put(job)
        else:
            self.dead_letter.append(job)

    async def depth(self) -> int:
        return self._queue.qsize()


class RedisJobQueue(JobQueue):
    """Production queue with visibility timeouts and a dead-letter list.

    Jobs are leased into a processing set with a deadline; :meth:`recover`
    returns jobs whose worker died before acknowledging them.
    """

    def __init__(self, client: Any, *, namespace: str = "coflowai",
                 queue: str = "jobs", visibility_timeout: float = 300.0,
                 max_attempts: int = 3) -> None:
        if client is None:  # pragma: no cover - defensive
            raise ConfigurationError("RedisJobQueue requires a redis client")
        self.client = client
        self.ns = namespace
        self.queue_name = queue
        self.visibility_timeout = visibility_timeout
        self.max_attempts = max_attempts

    def _key(self, *parts: str) -> str:
        return ":".join((self.ns, self.queue_name, *parts))

    async def enqueue(self, job: Job) -> Job:
        await self.client.lpush(self._key("pending"),
                                json.dumps(job.to_dict(), default=str))
        return job

    async def dequeue(self, *, timeout: float = 1.0) -> Job | None:
        raw = await self.client.brpop(self._key("pending"), timeout=int(timeout) or 1)
        if raw is None:
            return None
        _key, payload = raw
        job = Job.from_dict(json.loads(payload))
        job.attempts += 1
        await self.client.zadd(self._key("processing"),
                               {json.dumps(job.to_dict(), default=str):
                                time.time() + self.visibility_timeout})
        return job

    async def ack(self, job: Job) -> None:
        await self._remove_in_flight(job)

    async def nack(self, job: Job, *, requeue: bool = True) -> None:
        await self._remove_in_flight(job)
        payload = json.dumps(job.to_dict(), default=str)
        if requeue and job.attempts < self.max_attempts:
            await self.client.lpush(self._key("pending"), payload)
        else:
            await self.client.lpush(self._key("dead"), payload)

    async def _remove_in_flight(self, job: Job) -> None:
        members = await self.client.zrange(self._key("processing"), 0, -1)
        for member in members:
            if json.loads(member).get("job_id") == job.job_id:
                await self.client.zrem(self._key("processing"), member)
                return

    async def recover(self) -> int:
        """Requeue jobs whose visibility timeout expired (worker crashed)."""
        now = time.time()
        expired = await self.client.zrangebyscore(self._key("processing"), 0, now)
        for member in expired:
            await self.client.zrem(self._key("processing"), member)
            await self.client.lpush(self._key("pending"), member)
        return len(expired)

    async def depth(self) -> int:
        return int(await self.client.llen(self._key("pending")))
