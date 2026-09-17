"""Tool registry with permission-aware exposure.

Prompt-injection defence: an agent is only ever *shown* the tools it is
authorised to call.
"""

from __future__ import annotations

from typing import Iterable

from ..core.exceptions import ToolNotFound
from ..policies.permissions import PermissionSet
from .tool import Tool

__all__ = ["ToolRegistry"]


class ToolRegistry:
    def __init__(self, tools: Iterable[Tool] | None = None) -> None:
        self._tools: dict[str, Tool] = {}
        for item in tools or ():
            self.register(item)

    def register(self, tool: Tool, *, replace: bool = True) -> Tool:
        if not isinstance(tool, Tool):
            raise TypeError(f"{tool!r} is not a Tool — decorate it with @tool")
        if not replace and tool.name in self._tools:
            raise ValueError(f"tool '{tool.name}' already registered")
        self._tools[tool.name] = tool
        return tool

    def register_many(self, tools: Iterable[Tool]) -> None:
        for item in tools:
            self.register(item)

    def unregister(self, name: str) -> None:
        self._tools.pop(name, None)

    def get(self, name: str) -> Tool:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise ToolNotFound(f"tool '{name}' is not registered",
                               tool=name, available=sorted(self._tools)) from exc

    def has(self, name: str) -> bool:
        return name in self._tools

    def all(self) -> list[Tool]:
        return list(self._tools.values())

    def names(self) -> list[str]:
        return sorted(self._tools)

    # ------------------------------------------------------------- filtering
    def visible_to(self, permissions: PermissionSet,
                   allowed_names: Iterable[str] | None = None) -> list[Tool]:
        """Tools an agent may see: intersection of allow-list and permissions."""
        allowed = set(allowed_names) if allowed_names is not None else None
        return [
            tool for tool in self._tools.values()
            if (allowed is None or tool.name in allowed)
            and permissions.allows(tool.permission)
        ]

    def scoped(self, names: Iterable[str]) -> "ToolRegistry":
        registry = ToolRegistry()
        for name in names:
            registry.register(self.get(name))
        return registry

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: object) -> bool:
        return name in self._tools
