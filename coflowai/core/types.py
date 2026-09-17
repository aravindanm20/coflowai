"""Provider-independent primitive types.

Nothing in this module may reference a provider SDK.  Adapters translate these
types to and from their own wire formats.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

__all__ = [
    "Role",
    "ToolCall",
    "ToolResult",
    "Message",
    "ExecutionStatus",
    "ExecutionMode",
    "Usage",
    "TERMINAL_STATUSES",
    "RESUMABLE_STATUSES",
    "VALID_TRANSITIONS",
]


class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass(slots=True)
class ToolCall:
    """A tool invocation *requested by a model*.  Arguments are untrusted."""

    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "name": self.name, "arguments": self.arguments}


@dataclass(slots=True)
class ToolResult:
    call_id: str
    name: str
    output: Any = None
    error: str | None = None
    duration_ms: float = 0.0

    @property
    def ok(self) -> bool:
        return self.error is None

    def to_dict(self) -> dict[str, Any]:
        return {
            "call_id": self.call_id,
            "name": self.name,
            "output": self.output,
            "error": self.error,
            "duration_ms": self.duration_ms,
        }


@dataclass(slots=True)
class Message:
    role: Role
    content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None
    name: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def system(cls, content: str) -> "Message":
        return cls(Role.SYSTEM, content)

    @classmethod
    def user(cls, content: str) -> "Message":
        return cls(Role.USER, content)

    @classmethod
    def assistant(cls, content: str | None = None,
                  tool_calls: list[ToolCall] | None = None) -> "Message":
        return cls(Role.ASSISTANT, content, tool_calls=list(tool_calls or []))

    @classmethod
    def tool(cls, call_id: str, name: str, content: str) -> "Message":
        return cls(Role.TOOL, content, tool_call_id=call_id, name=name)

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role.value,
            "content": self.content,
            "tool_calls": [c.to_dict() for c in self.tool_calls],
            "tool_call_id": self.tool_call_id,
            "name": self.name,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Message":
        return cls(
            role=Role(data["role"]),
            content=data.get("content"),
            tool_calls=[ToolCall(**c) for c in data.get("tool_calls", [])],
            tool_call_id=data.get("tool_call_id"),
            name=data.get("name"),
        )


class ExecutionStatus(str, Enum):
    CREATED = "created"
    RUNNING = "running"
    WAITING = "waiting"
    WAITING_FOR_APPROVAL = "waiting_for_approval"
    RETRYING = "retrying"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"


TERMINAL_STATUSES: frozenset[ExecutionStatus] = frozenset(
    {
        ExecutionStatus.COMPLETED,
        ExecutionStatus.FAILED,
        ExecutionStatus.CANCELLED,
        ExecutionStatus.TIMED_OUT,
    }
)

#: Explicitly enforced state machine (see §11 of the design).
VALID_TRANSITIONS: dict[ExecutionStatus, frozenset[ExecutionStatus]] = {
    ExecutionStatus.CREATED: frozenset({ExecutionStatus.RUNNING,
                                        ExecutionStatus.CANCELLED}),
    ExecutionStatus.RUNNING: frozenset({
        ExecutionStatus.WAITING,
        ExecutionStatus.WAITING_FOR_APPROVAL,
        ExecutionStatus.RETRYING,
        ExecutionStatus.COMPLETED,
        ExecutionStatus.FAILED,
        ExecutionStatus.CANCELLED,
        ExecutionStatus.TIMED_OUT,
    }),
    ExecutionStatus.WAITING: frozenset({ExecutionStatus.RUNNING,
                                        ExecutionStatus.CANCELLED,
                                        ExecutionStatus.TIMED_OUT}),
    ExecutionStatus.WAITING_FOR_APPROVAL: frozenset({ExecutionStatus.RUNNING,
                                                     ExecutionStatus.FAILED,
                                                     ExecutionStatus.CANCELLED,
                                                     ExecutionStatus.TIMED_OUT}),
    ExecutionStatus.RETRYING: frozenset({ExecutionStatus.RUNNING,
                                         ExecutionStatus.FAILED,
                                         ExecutionStatus.CANCELLED,
                                         ExecutionStatus.TIMED_OUT}),
    ExecutionStatus.COMPLETED: frozenset(),
    # FAILED is terminal for the *result*, but an operator may explicitly
    # recover a failed execution from its last checkpoint via runtime.resume().
    ExecutionStatus.FAILED: frozenset({ExecutionStatus.RUNNING}),
    ExecutionStatus.CANCELLED: frozenset(),
    ExecutionStatus.TIMED_OUT: frozenset(),
}


#: Statuses from which ``runtime.resume(...)`` is allowed.
RESUMABLE_STATUSES: frozenset[ExecutionStatus] = frozenset(
    {
        ExecutionStatus.CREATED,
        ExecutionStatus.RUNNING,
        ExecutionStatus.WAITING,
        ExecutionStatus.WAITING_FOR_APPROVAL,
        ExecutionStatus.RETRYING,
        ExecutionStatus.FAILED,
    }
)


class ExecutionMode(str, Enum):
    """Determinism level requested by the caller (see §68)."""

    STRICT = "strict"
    STANDARD = "standard"
    EXPLORATORY = "exploratory"


@dataclass(slots=True)
class Usage:
    """Aggregated usage for an execution (or any sub-scope of it)."""

    model_calls: int = 0
    tool_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    cost: float = 0.0
    duration_ms: float = 0.0
    model_latency_ms: float = 0.0
    tool_latency_ms: float = 0.0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def add(self, other: "Usage") -> "Usage":
        self.model_calls += other.model_calls
        self.tool_calls += other.tool_calls
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.cached_tokens += other.cached_tokens
        self.cost += other.cost
        self.model_latency_ms += other.model_latency_ms
        self.tool_latency_ms += other.tool_latency_ms
        return self

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_calls": self.model_calls,
            "tool_calls": self.tool_calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cached_tokens": self.cached_tokens,
            "total_tokens": self.total_tokens,
            "cost": round(self.cost, 6),
            "duration_ms": round(self.duration_ms, 3),
            "model_latency_ms": round(self.model_latency_ms, 3),
            "tool_latency_ms": round(self.tool_latency_ms, 3),
        }
