"""State store interface + in-memory implementation.

The interface is deliberately "distributed-ready": every method is async and
takes an execution id, so a Postgres/Redis adapter is a drop-in replacement.
"""

from __future__ import annotations

import asyncio
import copy
from abc import ABC, abstractmethod
from ..core.exceptions import CheckpointError
from .checkpoint import Checkpoint, ExecutionRecord

__all__ = ["StateStore", "InMemoryStateStore"]


class StateStore(ABC):
    # --------------------------------------------------------- execution rows
    @abstractmethod
    async def save_execution(self, record: ExecutionRecord) -> None:
        ...

    @abstractmethod
    async def get_execution(self, execution_id: str) -> ExecutionRecord | None:
        ...

    @abstractmethod
    async def list_executions(self, *, status: str | None = None,
                              limit: int = 100) -> list[ExecutionRecord]:
        ...

    # ------------------------------------------------------------ checkpoints
    @abstractmethod
    async def save_checkpoint(self, checkpoint: Checkpoint) -> Checkpoint:
        ...

    @abstractmethod
    async def list_checkpoints(self, execution_id: str) -> list[Checkpoint]:
        ...

    async def latest_checkpoint(self, execution_id: str) -> Checkpoint | None:
        checkpoints = await self.list_checkpoints(execution_id)
        return checkpoints[-1] if checkpoints else None

    async def get_checkpoint(self, execution_id: str,
                             checkpoint_id: str) -> Checkpoint:
        for checkpoint in await self.list_checkpoints(execution_id):
            if checkpoint.checkpoint_id == checkpoint_id:
                return checkpoint
        raise CheckpointError("checkpoint not found", execution_id=execution_id,
                              checkpoint_id=checkpoint_id)

    async def checkpoint_at_step(self, execution_id: str,
                                 step: int) -> Checkpoint | None:
        candidates = [c for c in await self.list_checkpoints(execution_id)
                      if c.step <= step]
        return candidates[-1] if candidates else None

    # ------------------------------------------------------------- locking
    async def acquire_lock(self, execution_id: str, *, ttl: float = 60.0) -> bool:
        """Single-process default: always granted.  Distributed adapters
        implement a real lease so two workers never resume the same execution."""
        return True

    async def release_lock(self, execution_id: str) -> None:
        return None


class InMemoryStateStore(StateStore):
    def __init__(self) -> None:
        self._executions: dict[str, ExecutionRecord] = {}
        self._checkpoints: dict[str, list[Checkpoint]] = {}
        self._locks: set[str] = set()
        self._mutex = asyncio.Lock()

    async def save_execution(self, record: ExecutionRecord) -> None:
        async with self._mutex:
            self._executions[record.execution_id] = record

    async def get_execution(self, execution_id: str) -> ExecutionRecord | None:
        return self._executions.get(execution_id)

    async def list_executions(self, *, status: str | None = None,
                              limit: int = 100) -> list[ExecutionRecord]:
        records = sorted(self._executions.values(), key=lambda r: r.created_at,
                         reverse=True)
        if status:
            records = [r for r in records if r.status.value == status]
        return records[:limit]

    async def save_checkpoint(self, checkpoint: Checkpoint) -> Checkpoint:
        async with self._mutex:
            stored = copy.deepcopy(checkpoint)
            self._checkpoints.setdefault(checkpoint.execution_id, []).append(stored)
            return stored

    async def list_checkpoints(self, execution_id: str) -> list[Checkpoint]:
        return [copy.deepcopy(c) for c in self._checkpoints.get(execution_id, ())]

    async def acquire_lock(self, execution_id: str, *, ttl: float = 60.0) -> bool:
        async with self._mutex:
            if execution_id in self._locks:
                return False
            self._locks.add(execution_id)
            return True

    async def release_lock(self, execution_id: str) -> None:
        async with self._mutex:
            self._locks.discard(execution_id)

    def clear(self) -> None:
        self._executions.clear()
        self._checkpoints.clear()
        self._locks.clear()
