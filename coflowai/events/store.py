"""Event store interface + the zero-dependency in-memory implementation."""

from __future__ import annotations

import asyncio
import json
from abc import ABC, abstractmethod
from collections.abc import Iterable
from pathlib import Path

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
        # execution_id -> last sequence, plus the byte offset indexed so far.
        # Appends are O(1) amortised: only bytes written since the last append
        # (by this or another process) are scanned -- the previous version
        # re-read and parsed the whole log on every event (O(n^2) overall).
        self._sequences: dict[str, int] = {}
        self._indexed_bytes = 0

    def _catch_up(self) -> None:
        """Index any complete lines appended since ``_indexed_bytes``."""
        size = self.path.stat().st_size
        if size < self._indexed_bytes:          # truncated/rotated -> rebuild
            self._sequences.clear()
            self._indexed_bytes = 0
        if size == self._indexed_bytes:
            return
        with self.path.open("rb") as handle:
            handle.seek(self._indexed_bytes)
            for raw in handle:
                if not raw.endswith(b"\n"):
                    break                       # partial line from a writer
                self._indexed_bytes += len(raw)
                line = raw.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except ValueError:
                    continue                    # skip a corrupt line
                exec_id = data.get("execution_id")
                seq = data.get("sequence") or self._sequences.get(exec_id, 0) + 1
                if exec_id is not None:
                    self._sequences[exec_id] = max(self._sequences.get(exec_id, 0),
                                                   int(seq))

    async def append(self, event: Event) -> Event:
        async with self._lock:
            self._catch_up()
            event.sequence = self._sequences.get(event.execution_id, 0) + 1
            payload = (json.dumps(event.to_dict(), default=str) + "\n").encode("utf-8")
            with self.path.open("ab") as handle:
                handle.write(payload)
            self._sequences[event.execution_id] = event.sequence
            self._indexed_bytes += len(payload)
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
        with self.path.open(encoding="utf-8") as handle:   # stream, don't slurp
            for line in handle:
                if line.strip():
                    try:
                        events.append(Event.from_dict(json.loads(line)))
                    except ValueError:
                        continue
        return events
