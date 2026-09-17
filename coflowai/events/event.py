"""Event record + the canonical event-type catalogue."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from ..core.ids import event_id

__all__ = ["Event", "EventType"]


class EventType:
    """String constants for every standard event (see §13)."""

    EXECUTION_CREATED = "execution.created"
    EXECUTION_STARTED = "execution.started"
    EXECUTION_COMPLETED = "execution.completed"
    EXECUTION_FAILED = "execution.failed"
    EXECUTION_CANCELLED = "execution.cancelled"
    EXECUTION_TIMEOUT = "execution.timeout"
    EXECUTION_RESUMED = "execution.resumed"
    EXECUTION_PAUSED = "execution.paused"
    EXECUTION_FORKED = "execution.forked"

    AGENT_STARTED = "agent.started"
    AGENT_COMPLETED = "agent.completed"
    AGENT_FAILED = "agent.failed"

    MODEL_REQUESTED = "model.requested"
    MODEL_COMPLETED = "model.completed"
    MODEL_FAILED = "model.failed"
    MODEL_RETRYING = "model.retrying"

    TOOL_REQUESTED = "tool.requested"
    TOOL_STARTED = "tool.started"
    TOOL_COMPLETED = "tool.completed"
    TOOL_FAILED = "tool.failed"
    TOOL_DENIED = "tool.denied"
    TOOL_RETRYING = "tool.retrying"

    STATE_UPDATED = "state.updated"
    CHECKPOINT_CREATED = "checkpoint.created"
    CHECKPOINT_RESTORED = "checkpoint.restored"

    APPROVAL_REQUESTED = "approval.requested"
    APPROVAL_APPROVED = "approval.approved"
    APPROVAL_REJECTED = "approval.rejected"

    BUDGET_WARNING = "budget.warning"
    BUDGET_EXCEEDED = "budget.exceeded"

    WORKFLOW_STARTED = "workflow.started"
    WORKFLOW_NODE_STARTED = "workflow.node.started"
    WORKFLOW_NODE_COMPLETED = "workflow.node.completed"
    WORKFLOW_NODE_FAILED = "workflow.node.failed"
    WORKFLOW_NODE_SKIPPED = "workflow.node.skipped"
    WORKFLOW_COMPLETED = "workflow.completed"

    ALL: frozenset[str] = frozenset()


EventType.ALL = frozenset(
    value
    for name, value in vars(EventType).items()
    if name.isupper() and isinstance(value, str)
)


@dataclass(slots=True)
class Event:
    """An immutable fact about an execution.  Events are the source of truth."""

    execution_id: str
    type: str
    payload: dict[str, Any] = field(default_factory=dict)
    event_id: str = field(default_factory=event_id)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    sequence: int = 0
    step: int = 0
    node_id: str | None = None
    trace_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "execution_id": self.execution_id,
            "type": self.type,
            "timestamp": self.timestamp.isoformat(),
            "sequence": self.sequence,
            "step": self.step,
            "node_id": self.node_id,
            "trace_id": self.trace_id,
            "payload": self.payload,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Event":
        return cls(
            execution_id=data["execution_id"],
            type=data["type"],
            payload=data.get("payload", {}),
            event_id=data.get("event_id", event_id()),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            sequence=data.get("sequence", 0),
            step=data.get("step", 0),
            node_id=data.get("node_id"),
            trace_id=data.get("trace_id"),
        )
