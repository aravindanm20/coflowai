"""Event publisher: writes to the store and fans out to subscribers.

Subscriber failures never break an execution — they are logged and swallowed.
"""

from __future__ import annotations

import asyncio
import inspect
from typing import Any, Awaitable, Callable

from ..observability.logger import get_logger
from .event import Event
from .store import EventStore, InMemoryEventStore

Subscriber = Callable[[Event], Awaitable[None] | None]

__all__ = ["EventPublisher", "Subscriber"]

_log = get_logger("events")


class EventPublisher:
    def __init__(self, store: EventStore | None = None) -> None:
        self.store: EventStore = store or InMemoryEventStore()
        self._subscribers: list[tuple[str | None, Subscriber]] = []

    def subscribe(self, subscriber: Subscriber, *, event_type: str | None = None) -> None:
        self._subscribers.append((event_type, subscriber))

    def unsubscribe(self, subscriber: Subscriber) -> None:
        self._subscribers = [s for s in self._subscribers if s[1] is not subscriber]

    async def publish(self, event: Event) -> Event:
        await self.store.append(event)
        _log.info(event.type, **_log_payload(event))
        for wanted, subscriber in list(self._subscribers):
            if wanted is not None and wanted != event.type:
                continue
            try:
                result = subscriber(event)
                if inspect.isawaitable(result):
                    await result
            except asyncio.CancelledError:  # pragma: no cover - propagate
                raise
            except Exception as exc:  # pragma: no cover - defensive
                _log.warning("event.subscriber_failed", error=repr(exc),
                             event_type=event.type)
        return event


def _log_payload(event: Event) -> dict[str, Any]:
    payload = {
        "execution_id": event.execution_id,
        "step": event.step,
        "sequence": event.sequence,
    }
    if event.node_id:
        payload["node_id"] = event.node_id
    for key in ("agent", "tool", "model", "duration_ms", "status", "error"):
        if key in event.payload:
            payload[key] = event.payload[key]
    return payload
