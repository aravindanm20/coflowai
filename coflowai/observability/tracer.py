"""Execution tracing.

A built-in span recorder produces the nested execution trace shown in §52.  If
``opentelemetry-api`` is installed, spans are mirrored to OTel as well — but the
core never requires it.
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator

from .redaction import redact_mapping

__all__ = ["Span", "Tracer", "render_trace"]

try:  # pragma: no cover - optional dependency
    from opentelemetry import trace as _otel_trace

    _OTEL_TRACER = _otel_trace.get_tracer("coflowai")
except Exception:  # pragma: no cover
    _OTEL_TRACER = None


@dataclass(slots=True)
class Span:
    name: str
    kind: str = "internal"          # execution | workflow | agent | model | tool
    attributes: dict[str, Any] = field(default_factory=dict)
    children: list["Span"] = field(default_factory=list)
    start: float = field(default_factory=time.perf_counter)
    end: float | None = None
    error: str | None = None

    @property
    def duration_ms(self) -> float:
        return ((self.end or time.perf_counter()) - self.start) * 1000

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "duration_ms": round(self.duration_ms, 3),
            "attributes": redact_mapping(self.attributes),
            "error": self.error,
            "children": [c.to_dict() for c in self.children],
        }

    def walk(self) -> Iterator["Span"]:
        yield self
        for child in self.children:
            yield from child.walk()


class Tracer:
    """Per-execution span tree.  Not global, not shared: no global state."""

    def __init__(self, execution_id: str, trace_id: str) -> None:
        self.execution_id = execution_id
        self.trace_id = trace_id
        self.root: Span | None = None
        self._stack: list[Span] = []

    @asynccontextmanager
    async def span(self, name: str, kind: str = "internal", **attributes: Any):
        span = Span(name=name, kind=kind, attributes=dict(attributes))
        if self._stack:
            self._stack[-1].children.append(span)
        else:
            self.root = span
        self._stack.append(span)
        otel_cm = None
        if _OTEL_TRACER is not None:  # pragma: no cover - optional
            otel_cm = _OTEL_TRACER.start_as_current_span(name)
            otel_cm.__enter__()
        try:
            yield span
        except BaseException as exc:
            span.error = repr(exc)
            raise
        finally:
            span.end = time.perf_counter()
            self._stack.pop()
            if otel_cm is not None:  # pragma: no cover - optional
                otel_cm.__exit__(None, None, None)

    def to_dict(self) -> dict[str, Any]:
        return {
            "execution_id": self.execution_id,
            "trace_id": self.trace_id,
            "root": self.root.to_dict() if self.root else None,
        }

    def render(self) -> str:
        return render_trace(self.root) if self.root else "(no spans)"


def render_trace(span: Span, indent: str = "", last: bool = True,
                 root: bool = True) -> str:
    """Render a span tree like the execution trace in the design document."""
    label = f"{span.name}"
    duration = f"{span.duration_ms / 1000:.1f}s"
    if root:
        line = f"{label:<40}{duration:>8}"
    else:
        connector = "└── " if last else "├── "
        line = f"{indent}{connector}{label:<{max(4, 36 - len(indent))}}{duration:>8}"
    lines = [line]
    child_indent = indent + ("    " if last or root else "│   ") if not root else ""
    for i, child in enumerate(span.children):
        lines.append(
            render_trace(child, child_indent, i == len(span.children) - 1, root=False)
        )
    return "\n".join(lines)
