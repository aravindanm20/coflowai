"""Module-level tools used to exercise the subprocess sandbox.

The sandbox requires importable, module-level functions (closures cannot cross a
process boundary), so these fixtures live in a real module rather than a test.
"""

from __future__ import annotations

import os

from ..tools.decorator import tool

__all__ = ["echo", "explode", "read_env", "burn_memory"]


@tool(description="Echo a value back", timeout=10)
async def echo(value: str) -> dict:
    return {"echoed": value}


@tool(description="Always fails", timeout=10)
async def explode() -> str:
    raise RuntimeError("tool exploded on purpose")


@tool(description="Read an environment variable", timeout=10)
async def read_env(name: str) -> str | None:
    return os.environ.get(name)


@tool(description="Allocate a large list to trip the memory limit", timeout=15)
async def burn_memory(megabytes: int = 2048) -> int:
    blob = bytearray(megabytes * 1024 * 1024)
    return len(blob)
