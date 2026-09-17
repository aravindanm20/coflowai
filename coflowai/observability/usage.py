"""Usage tracking per execution / workflow / agent / model / tool."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from ..core.types import Usage

__all__ = ["UsageTracker"]


class UsageTracker:
    """Accumulates usage across scopes.  One tracker per execution."""

    def __init__(self) -> None:
        self.total = Usage()
        self.by_agent: dict[str, Usage] = defaultdict(Usage)
        self.by_model: dict[str, Usage] = defaultdict(Usage)
        self.by_tool: dict[str, Usage] = defaultdict(Usage)
        self.by_node: dict[str, Usage] = defaultdict(Usage)

    def record_model(self, *, model: str, agent: str | None, node_id: str | None,
                     input_tokens: int, output_tokens: int, cached_tokens: int,
                     cost: float, latency_ms: float) -> None:
        usage = Usage(
            model_calls=1,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_tokens=cached_tokens,
            cost=cost,
            model_latency_ms=latency_ms,
        )
        self._fan_out(usage, agent=agent, node_id=node_id, model=model)

    def record_tool(self, *, tool: str, agent: str | None, node_id: str | None,
                    latency_ms: float) -> None:
        usage = Usage(tool_calls=1, tool_latency_ms=latency_ms)
        self._fan_out(usage, agent=agent, node_id=node_id, tool=tool)

    def _fan_out(self, usage: Usage, *, agent: str | None, node_id: str | None,
                 model: str | None = None, tool: str | None = None) -> None:
        self.total.add(_copy(usage))
        if agent:
            self.by_agent[agent].add(_copy(usage))
        if node_id:
            self.by_node[node_id].add(_copy(usage))
        if model:
            self.by_model[model].add(_copy(usage))
        if tool:
            self.by_tool[tool].add(_copy(usage))

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": self.total.to_dict(),
            "by_agent": {k: v.to_dict() for k, v in self.by_agent.items()},
            "by_model": {k: v.to_dict() for k, v in self.by_model.items()},
            "by_tool": {k: v.to_dict() for k, v in self.by_tool.items()},
            "by_node": {k: v.to_dict() for k, v in self.by_node.items()},
        }


def _copy(usage: Usage) -> Usage:
    return Usage(
        model_calls=usage.model_calls,
        tool_calls=usage.tool_calls,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        cached_tokens=usage.cached_tokens,
        cost=usage.cost,
        duration_ms=usage.duration_ms,
        model_latency_ms=usage.model_latency_ms,
        tool_latency_ms=usage.tool_latency_ms,
    )
