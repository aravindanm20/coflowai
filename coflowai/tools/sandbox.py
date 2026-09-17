"""Sandbox abstraction for tool execution.

The core ships an in-process sandbox (no isolation beyond timeouts and the
permission runtime).  Stronger isolation — subprocess, container, remote worker
— is provided by adapters implementing this interface.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import Any

from .tool import Tool

__all__ = ["Sandbox", "InProcessSandbox", "ThreadSandbox"]


class Sandbox(ABC):
    name: str = "sandbox"

    @abstractmethod
    async def run(self, tool: Tool, arguments: dict[str, Any],
                  *, metadata: dict[str, Any] | None = None) -> Any:
        ...


class InProcessSandbox(Sandbox):
    """Default: run in the current event loop (sync tools go to a thread)."""

    name = "in_process"

    async def run(self, tool: Tool, arguments: dict[str, Any],
                  *, metadata: dict[str, Any] | None = None) -> Any:
        return await tool.invoke(arguments)


class ThreadSandbox(Sandbox):
    """Force every tool onto a worker thread — keeps the loop responsive for
    CPU-ish or accidentally blocking tools."""

    name = "thread"

    def __init__(self, max_workers: int = 8) -> None:
        self._semaphore = asyncio.Semaphore(max_workers)

    async def run(self, tool: Tool, arguments: dict[str, Any],
                  *, metadata: dict[str, Any] | None = None) -> Any:
        async with self._semaphore:
            if asyncio.iscoroutinefunction(tool.func):
                return await tool.func(**arguments)
            return await asyncio.to_thread(tool.func, **arguments)
