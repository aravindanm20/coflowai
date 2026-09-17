"""Dependency-free memory store with lexical (BM25-ish) scoring.

Good enough for local development and tests; swap in a vector adapter when the
application actually needs embeddings.
"""

from __future__ import annotations

import asyncio
import math
import re
from collections import Counter
from typing import Any

from .base import MemoryKind, MemoryRecord, MemoryStore

__all__ = ["InMemoryMemoryStore"]

_TOKEN = re.compile(r"[a-z0-9]+")


def _tokenise(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class InMemoryMemoryStore(MemoryStore):
    def __init__(self, max_records: int | None = None) -> None:
        self._records: dict[str, MemoryRecord] = {}
        self._max_records = max_records
        self._lock = asyncio.Lock()

    async def put(self, key: str, value: Any, *,
                  kind: MemoryKind = MemoryKind.LONG_TERM,
                  metadata: dict[str, Any] | None = None) -> MemoryRecord:
        async with self._lock:
            record = MemoryRecord(key=key, value=value, kind=kind,
                                  metadata=metadata or {})
            self._records[key] = record
            if self._max_records and len(self._records) > self._max_records:
                oldest = sorted(self._records.values(), key=lambda r: r.created_at)
                for stale in oldest[: len(self._records) - self._max_records]:
                    self._records.pop(stale.key, None)
            return record

    async def get(self, key: str) -> MemoryRecord | None:
        return self._records.get(key)

    async def delete(self, key: str) -> None:
        self._records.pop(key, None)

    async def clear(self) -> None:
        self._records.clear()

    async def search(self, query: str, *, limit: int = 10,
                     kind: MemoryKind | None = None) -> list[MemoryRecord]:
        candidates = [r for r in self._records.values()
                      if kind is None or r.kind is kind]
        if not candidates:
            return []
        query_terms = Counter(_tokenise(query))
        if not query_terms:
            return candidates[:limit]

        doc_freq: Counter[str] = Counter()
        tokenised: dict[str, Counter[str]] = {}
        for record in candidates:
            tokens = Counter(_tokenise(record.as_text()))
            tokenised[record.key] = tokens
            for term in tokens:
                doc_freq[term] += 1

        total = len(candidates)
        scored: list[MemoryRecord] = []
        for record in candidates:
            tokens = tokenised[record.key]
            score = 0.0
            for term, weight in query_terms.items():
                if term not in tokens:
                    continue
                idf = math.log(1 + total / (1 + doc_freq[term]))
                score += weight * idf * (tokens[term] / (tokens[term] + 1.5))
            if score > 0:
                record.score = round(score, 6)
                scored.append(record)

        scored.sort(key=lambda r: (-r.score, r.key))
        return scored[:limit]

    def __len__(self) -> int:
        return len(self._records)
