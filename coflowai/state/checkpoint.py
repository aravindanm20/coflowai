"""Checkpoint and execution records."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from ..core.ids import new_id
from ..core.types import ExecutionStatus

__all__ = ["Checkpoint", "ExecutionRecord"]


@dataclass(slots=True)
class Checkpoint:
    execution_id: str
    step: int = 0
    node_id: str | None = None
    state: dict[str, Any] = field(default_factory=dict)
    model_calls: int = 0
    tool_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost: float = 0.0
    status: ExecutionStatus = ExecutionStatus.RUNNING
    workflow_hash: str | None = None
    reason: str = "auto"
    checkpoint_id: str = field(default_factory=lambda: new_id("ckpt"))
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def tokens_used(self) -> int:
        return self.input_tokens + self.output_tokens

    def to_dict(self) -> dict[str, Any]:
        return {
            "checkpoint_id": self.checkpoint_id,
            "execution_id": self.execution_id,
            "step": self.step,
            "node_id": self.node_id,
            "state": self.state,
            "model_calls": self.model_calls,
            "tool_calls": self.tool_calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "tokens_used": self.tokens_used,
            "cost": self.cost,
            "status": self.status.value,
            "workflow_hash": self.workflow_hash,
            "reason": self.reason,
            "created_at": self.created_at.isoformat(),
        }


@dataclass(slots=True)
class ExecutionRecord:
    """Durable metadata about one execution (the 'header' row)."""

    execution_id: str
    status: ExecutionStatus = ExecutionStatus.CREATED
    input: Any = None
    output: Any = None
    error: str | None = None
    executable: str = ""
    workflow_hash: str | None = None
    policy: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    parent_execution_id: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def touch(self, status: ExecutionStatus | None = None) -> None:
        if status is not None:
            self.status = status
        self.updated_at = datetime.now(timezone.utc)

    def to_dict(self) -> dict[str, Any]:
        return {
            "execution_id": self.execution_id,
            "status": self.status.value,
            "executable": self.executable,
            "workflow_hash": self.workflow_hash,
            "error": self.error,
            "policy": self.policy,
            "metadata": self.metadata,
            "parent_execution_id": self.parent_execution_id,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }
