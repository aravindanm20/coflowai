"""Metrics interface with an in-process default implementation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any

__all__ = ["MetricsSink", "InMemoryMetrics", "NullMetrics"]

Labels = tuple[tuple[str, str], ...]


class MetricsSink(ABC):
    @abstractmethod
    def increment(self, name: str, value: float = 1.0, **labels: Any) -> None:
        ...

    @abstractmethod
    def observe(self, name: str, value: float, **labels: Any) -> None:
        ...

    @abstractmethod
    def gauge(self, name: str, value: float, **labels: Any) -> None:
        ...


class NullMetrics(MetricsSink):
    def increment(self, name: str, value: float = 1.0, **labels: Any) -> None:
        return None

    def observe(self, name: str, value: float, **labels: Any) -> None:
        return None

    def gauge(self, name: str, value: float, **labels: Any) -> None:
        return None


class InMemoryMetrics(MetricsSink):
    """Default sink: cheap, inspectable, test-friendly."""

    def __init__(self) -> None:
        self.counters: dict[tuple[str, Labels], float] = defaultdict(float)
        self.histograms: dict[tuple[str, Labels], list[float]] = defaultdict(list)
        self.gauges: dict[tuple[str, Labels], float] = {}

    @staticmethod
    def _key(name: str, labels: dict[str, Any]) -> tuple[str, Labels]:
        return name, tuple(sorted((k, str(v)) for k, v in labels.items()))

    def increment(self, name: str, value: float = 1.0, **labels: Any) -> None:
        self.counters[self._key(name, labels)] += value

    def observe(self, name: str, value: float, **labels: Any) -> None:
        self.histograms[self._key(name, labels)].append(value)

    def gauge(self, name: str, value: float, **labels: Any) -> None:
        self.gauges[self._key(name, labels)] = value

    def snapshot(self) -> dict[str, Any]:
        def fmt(key: tuple[str, Labels]) -> str:
            name, labels = key
            if not labels:
                return name
            rendered = ",".join(f"{k}={v}" for k, v in labels)
            return f"{name}{{{rendered}}}"

        return {
            "counters": {fmt(k): v for k, v in self.counters.items()},
            "histograms": {
                fmt(k): {
                    "count": len(v),
                    "sum": round(sum(v), 3),
                    "avg": round(sum(v) / len(v), 3) if v else 0.0,
                    "max": round(max(v), 3) if v else 0.0,
                }
                for k, v in self.histograms.items()
            },
            "gauges": {fmt(k): v for k, v in self.gauges.items()},
        }
