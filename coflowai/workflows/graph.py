"""Compiled execution graph."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..core.ids import stable_hash
from .node import Node

__all__ = ["Edge", "ExecutionGraph"]


@dataclass(slots=True, frozen=True)
class Edge:
    source: str
    target: str
    label: str | None = None


@dataclass(slots=True)
class ExecutionGraph:
    """Validated, immutable-by-convention representation of a workflow."""

    name: str
    version: str
    nodes: list[Node]
    edges: list[Edge] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def hash(self) -> str:
        return stable_hash(self.name, self.version,
                           [n.signature() for n in self.nodes],
                           [(e.source, e.target, e.label) for e in self.edges])

    def node(self, node_id: str) -> Node:
        for node in self.nodes:
            if node.id == node_id:
                return node
        raise KeyError(node_id)

    def index_of(self, node_id: str) -> int:
        for index, node in enumerate(self.nodes):
            if node.id == node_id:
                return index
        raise KeyError(node_id)

    def successors(self, node_id: str) -> list[str]:
        return [e.target for e in self.edges if e.source == node_id]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "hash": self.hash,
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": [{"source": e.source, "target": e.target, "label": e.label}
                      for e in self.edges],
        }

    def to_mermaid(self) -> str:
        lines = ["graph TD"]
        for node in self.nodes:
            lines.append(f'    {_safe(node.id)}["{node.name} ({node.kind.value})"]')
        for edge in self.edges:
            arrow = f"-->|{edge.label}|" if edge.label else "-->"
            lines.append(f"    {_safe(edge.source)} {arrow} {_safe(edge.target)}")
        return "\n".join(lines)

    def render(self) -> str:
        lines = [f"{self.name} v{self.version} ({self.hash})"]
        for index, node in enumerate(self.nodes, start=1):
            lines.append(f"  {index}. {node.name} [{node.kind.value}]")
        return "\n".join(lines)


def _safe(node_id: str) -> str:
    return "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in node_id)
