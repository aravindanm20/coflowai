"""Replay: turn the event log back into a human- and machine-readable story."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..events.event import Event, EventType

__all__ = ["ReplayStep", "ReplayTrace"]

_STEP_STARTERS = {
    EventType.MODEL_REQUESTED: "model",
    EventType.TOOL_REQUESTED: "tool",
    EventType.WORKFLOW_NODE_STARTED: "node",
    EventType.AGENT_STARTED: "agent",
    EventType.APPROVAL_REQUESTED: "approval",
}


@dataclass(slots=True)
class ReplayStep:
    index: int
    kind: str
    name: str
    step: int
    node_id: str | None
    events: list[Event] = field(default_factory=list)
    status: str = "completed"
    duration_ms: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "kind": self.kind,
            "name": self.name,
            "step": self.step,
            "node_id": self.node_id,
            "status": self.status,
            "duration_ms": self.duration_ms,
            "events": [e.to_dict() for e in self.events],
        }


@dataclass(slots=True)
class ReplayTrace:
    execution_id: str
    events: list[Event]
    steps: list[ReplayStep] = field(default_factory=list)

    @classmethod
    def from_events(cls, execution_id: str, events: list[Event]) -> "ReplayTrace":
        trace = cls(execution_id=execution_id, events=list(events))
        current: ReplayStep | None = None
        index = 0
        for event in trace.events:
            kind = _STEP_STARTERS.get(event.type)
            if kind is not None:
                index += 1
                current = ReplayStep(
                    index=index, kind=kind,
                    name=_name_for(event, kind), step=event.step,
                    node_id=event.node_id, events=[event],
                )
                trace.steps.append(current)
                continue
            if current is not None:
                current.events.append(event)
                if event.type.endswith(".completed"):
                    current.status = "completed"
                    current.duration_ms = event.payload.get("duration_ms")
                elif event.type.endswith(".failed"):
                    current.status = "failed"
                elif event.type.endswith(".denied"):
                    current.status = "denied"
        return trace

    # --------------------------------------------------------------- queries
    @property
    def status(self) -> str:
        for event in reversed(self.events):
            if event.type.startswith("execution.") and event.type.split(".")[1] in (
                    "completed", "failed", "cancelled", "timeout", "paused"):
                return event.type.split(".", 1)[1]
        return "unknown"

    def events_of(self, event_type: str) -> list[Event]:
        return [e for e in self.events if e.type == event_type]

    def model_calls(self) -> list[Event]:
        return self.events_of(EventType.MODEL_COMPLETED)

    def tool_calls(self) -> list[Event]:
        return self.events_of(EventType.TOOL_COMPLETED)

    def to_dict(self) -> dict[str, Any]:
        return {
            "execution_id": self.execution_id,
            "status": self.status,
            "steps": [s.to_dict() for s in self.steps],
            "events": [e.to_dict() for e in self.events],
        }

    def render(self) -> str:
        lines = [f"Execution: {self.execution_id} ({self.status})"]
        for step in self.steps:
            duration = (f"{step.duration_ms:.0f}ms"
                        if step.duration_ms is not None else "-")
            lines.append(
                f"  step {step.index:>2}  {step.kind:<8} {step.name:<28} "
                f"{step.status:<9} {duration:>8}"
            )
        return "\n".join(lines)


def _name_for(event: Event, kind: str) -> str:
    payload = event.payload
    for key in ("model", "tool", "agent", "node", "approval_id"):
        if key in payload and payload[key]:
            return str(payload[key])
    return event.node_id or kind
