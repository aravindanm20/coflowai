"""Memory interfaces.  Memory is always optional (Rules 4 & 5)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

__all__ = ["MemoryKind", "MemoryRecord", "MemoryStore"]


class MemoryKind(str, Enum):
    WORKING = "working"
    CONVERSATION = "conversation"
    LONG_TERM = "long_term"
    SEMANTIC = "semantic"
    EPISODIC = "episodic"


@dataclass(slots=True)
class MemoryRecord:
    key: str
    value: Any
    kind: MemoryKind = MemoryKind.LONG_TERM
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    score: float = 0.0

    def as_text(self) -> str:
        return self.value if isinstance(self.value, str) else repr(self.value)


class MemoryStore(ABC):
    @abstractmethod
    async def put(self, key: str, value: Any, *,
                  kind: MemoryKind = MemoryKind.LONG_TERM,
                  metadata: dict[str, Any] | None = None) -> MemoryRecord:
        ...

    @abstractmethod
    async def get(self, key: str) -> MemoryRecord | None:
        ...

    @abstractmethod
    async def search(self, query: str, *, limit: int = 10,
                     kind: MemoryKind | None = None) -> list[MemoryRecord]:
        ...

    @abstractmethod
    async def delete(self, key: str) -> None:
        ...

    async def clear(self) -> None:  # pragma: no cover - adapters may override
        return None
