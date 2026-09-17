"""Event store interface + the zero-dependency in-memory implementation."""

from __future__ import annotations

import asyncio
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable

from .event import Event

__all__ = ["EventStore", "InMemoryEventStore", "JsonlEventStore"]


class EventStore(ABC):
    """Append-only log of execution events."""

    @abstractmethod
    async def append(self, event: Event) -> Event:
        ...

    @abstractmethod
    async def get_events(self, execution_id: str) -> list[Event]:
        ...

    @abstractmethod
    async def list_executions(self) -> list[str]:
        ...

    async def append_many(self, events: Iterable[Event]) -> None:
        for event in events:
            await self.append(event)


class InMemoryEventStore(EventStore):
    """Default store: no infrastructure required."""

    def __init__(self) -> None:
        self._events: dict[str, list[Event]] = {}
        self._lock = asyncio.Lock()

    async def append(self, event: Event) -> Event:
        async with self._lock:
            bucket = self._events.setdefault(event.execution_id, [])
            event.sequence = len(bucket) + 1
            bucket.append(event)
            return event

    async def get_events(self, execution_id: str) -> list[Event]:
        return list(self._events.get(execution_id, ()))

    async def list_executions(self) -> list[str]:
        return list(self._events.keys())

    def clear(self) -> None:
        self._events.clear()


class JsonlEventStore(EventStore):
    """File-backed store (one JSON object per line) for local durability.

    Still dependency-free: useful for CLI inspection between processes.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)
        self._lock = asyncio.Lock()

    async def append(self, event: Event) -> Event:
        async with self._lock:
            existing = sum(
                1 for e in self._read() if e.execution_id == event.execution_id
            )
            event.sequence = existing + 1
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event.to_dict(), default=str) + "\n")
            return event

    async def get_events(self, execution_id: str) -> list[Event]:
        return [e for e in self._read() if e.execution_id == execution_id]

    async def list_executions(self) -> list[str]:
        seen: dict[str, None] = {}
        for event in self._read():
            seen.setdefault(event.execution_id, None)
        return list(seen)

    def _read(self) -> list[Event]:
        if not self.path.exists():
            return []
        events: list[Event] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                events.append(Event.from_dict(json.loads(line)))
        return events
