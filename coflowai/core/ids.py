"""Identifier helpers.

All identifiers in CoFlowAi are opaque, prefixed, URL-safe strings.  They are
deliberately *not* raw UUIDs so that a human reading a log line can tell what
kind of object an id refers to.
"""

from __future__ import annotations

import hashlib
from uuid import uuid4

__all__ = [
    "new_id",
    "execution_id",
    "event_id",
    "node_id",
    "tool_call_id",
    "model_call_id",
    "trace_id",
    "stable_hash",
    "idempotency_key",
]


def new_id(prefix: str, size: int = 12) -> str:
    """Return a new prefixed identifier, e.g. ``exec_9f2c1ab34d01``."""
    return f"{prefix}_{uuid4().hex[:size]}"


def execution_id() -> str:
    return new_id("exec")


def event_id() -> str:
    return new_id("evt")


def node_id(name: str) -> str:
    return f"node_{_slug(name)}_{uuid4().hex[:6]}"


def tool_call_id() -> str:
    return new_id("call")


def model_call_id() -> str:
    return new_id("mcall")


def trace_id() -> str:
    return new_id("trace", 16)


def stable_hash(*parts: object) -> str:
    """Deterministic short hash used for workflow hashes / idempotency keys."""
    digest = hashlib.sha256()
    for part in parts:
        digest.update(repr(part).encode("utf-8"))
        digest.update(b"\x1f")
    return digest.hexdigest()[:16]


def idempotency_key(execution: str, node: str | None, call: str) -> str:
    """Deterministic key so retries/resumes never duplicate side effects."""
    return stable_hash(execution, node or "-", call)


def _slug(value: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in value).strip("_").lower() or "node"
