"""Sandbox abstraction for tool execution.

The core ships an in-process sandbox (no isolation beyond timeouts and the
permission runtime).  Stronger isolation — subprocess, container, remote worker
— is provided by adapters implementing this interface.
"""

from __future__ import annotations

import asyncio
import inspect
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
        # Inject idempotency_key if the tool declares it as a parameter
        final_args = _inject_metadata_params(tool, arguments, metadata)
        return await tool.invoke(final_args)


class ThreadSandbox(Sandbox):
    """Force every tool onto a worker thread — keeps the loop responsive for
    CPU-ish or accidentally blocking tools."""

    name = "thread"

    def __init__(self, max_workers: int = 8) -> None:
        self._semaphore = asyncio.Semaphore(max_workers)

    async def run(self, tool: Tool, arguments: dict[str, Any],
                  *, metadata: dict[str, Any] | None = None) -> Any:
        # Inject idempotency_key if the tool declares it
        final_args = _inject_metadata_params(tool, arguments, metadata)
        async with self._semaphore:
            if asyncio.iscoroutinefunction(tool.func):
                return await tool.func(**final_args)
            return await asyncio.to_thread(tool.func, **final_args)


def _inject_metadata_params(tool: Tool, arguments: dict[str, Any],
                             metadata: dict[str, Any] | None) -> dict[str, Any]:
    """Inject reserved parameters from metadata if the tool function declares them.

    Currently supports:
    - idempotency_key: stable key for deduplication across retries/resumes
    """
    if not metadata:
        return arguments

    sig = inspect.signature(tool.func)
    final_args = dict(arguments)

    # Inject idempotency_key if the tool declares it
    if "idempotency_key" in sig.parameters and "idempotency_key" in metadata:
        final_args["idempotency_key"] = metadata["idempotency_key"]

    return final_args
