"""Bounded in-process caches for long-running runtimes/workers."""
from __future__ import annotations

import time
from collections import OrderedDict
from typing import Generic, Iterator, MutableMapping, TypeVar

K = TypeVar("K")
V = TypeVar("V")

__all__ = ["BoundedCache"]


class BoundedCache(MutableMapping[K, V], Generic[K, V]):
    """A dict with LRU eviction and optional time-to-live.

    Used for per-execution runtime caches (executables, policies, tracers,
    usage) so a worker that processes millions of executions has bounded
    memory.  Evicted entries are *caches*, not state: the durable record in
    the state store remains the source of truth.
    """

    def __init__(self, max_size: int | None = 1024,
                 ttl_seconds: float | None = None) -> None:
        if max_size is not None and max_size <= 0:
            raise ValueError("max_size must be positive (or None for unbounded)")
        if ttl_seconds is not None and ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive (or None)")
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._data: OrderedDict[K, tuple[float, V]] = OrderedDict()

    def _expired(self, stamp: float) -> bool:
        return self.ttl_seconds is not None and \
            time.monotonic() - stamp > self.ttl_seconds

    def __getitem__(self, key: K) -> V:
        stamp, value = self._data[key]
        if self._expired(stamp):
            del self._data[key]
            raise KeyError(key)
        self._data.move_to_end(key)
        return value

    def __setitem__(self, key: K, value: V) -> None:
        self._data[key] = (time.monotonic(), value)
        self._data.move_to_end(key)
        self._evict()

    def __delitem__(self, key: K) -> None:
        del self._data[key]

    def __iter__(self) -> Iterator[K]:
        self._purge_expired()
        return iter(list(self._data))

    def __len__(self) -> int:
        self._purge_expired()
        return len(self._data)

    def __contains__(self, key: object) -> bool:
        try:
            self[key]  # type: ignore[index]
        except KeyError:
            return False
        return True

    def _purge_expired(self) -> None:
        if self.ttl_seconds is None:
            return
        for key in [k for k, (stamp, _v) in self._data.items() if self._expired(stamp)]:
            del self._data[key]

    def _evict(self) -> None:
        self._purge_expired()
        if self.max_size is None:
            return
        while len(self._data) > self.max_size:
            self._data.popitem(last=False)
