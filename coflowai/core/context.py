"""Execution context: the only mutable execution state in the framework.

There is no global registry of "current execution" — everything an executable
needs is passed explicitly through this object, which makes concurrent and
distributed execution safe by construction.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ..events.event import Event, EventType
from ..policies.permissions import PermissionSet
from ..policies.policy import ExecutionPolicy
from .exceptions import ExecutionCancelled, ExecutionTimeout
from .ids import execution_id as new_execution_id
from .ids import trace_id as new_trace_id
from .types import ExecutionMode, ExecutionStatus, Usage

if TYPE_CHECKING:  # pragma: no cover
    from ..events.publisher import EventPublisher
    from ..models.registry import ModelRegistry
    from ..observability.metrics import MetricsSink
    from ..observability.tracer import Tracer
    from ..observability.usage import UsageTracker
    from ..state.store import StateStore
    from ..tools.registry import ToolRegistry

__all__ = ["ExecutionContext", "RuntimeServices"]


@dataclass(slots=True)
class RuntimeServices:
    """Infrastructure handles shared by every context in one execution."""

    publisher: "EventPublisher"
    state_store: "StateStore"
    models: "ModelRegistry"
    tools: "ToolRegistry"
    gateway: Any               # ModelGateway
    tool_executor: Any         # ToolExecutor
    tracer: "Tracer"
    metrics: "MetricsSink"
    usage: "UsageTracker"
    memory: Any = None
    runtime: Any = None
    approvals: Any = None


@dataclass
class ExecutionContext:
    """Isolated per-execution state, budgets and services."""

    execution_id: str = field(default_factory=new_execution_id)
    input: Any = None
    state: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    step: int = 0
    model_calls: int = 0
    tool_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    cost: float = 0.0

    policy: ExecutionPolicy = field(default_factory=ExecutionPolicy)
    permissions: PermissionSet = field(default_factory=PermissionSet.all)
    status: ExecutionStatus = ExecutionStatus.CREATED
    node_id: str | None = None
    agent_name: str | None = None
    parent_execution_id: str | None = None
    trace_id: str = field(default_factory=new_trace_id)

    services: RuntimeServices | None = None
    started_at: float = field(default_factory=time.monotonic)
    deadline: float | None = None
    _cancelled: bool = False
    _budget_warned: bool = False

    # ------------------------------------------------------------------ setup
    def __post_init__(self) -> None:
        if self.deadline is None and self.policy.timeout_seconds:
            self.deadline = self.started_at + self.policy.timeout_seconds

    # ------------------------------------------------------------- properties
    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def mode(self) -> ExecutionMode:
        return self.policy.mode

    @property
    def elapsed_seconds(self) -> float:
        return time.monotonic() - self.started_at

    @property
    def remaining_seconds(self) -> float | None:
        if self.deadline is None:
            return None
        return max(0.0, self.deadline - time.monotonic())

    def usage_snapshot(self) -> Usage:
        return Usage(
            model_calls=self.model_calls,
            tool_calls=self.tool_calls,
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            cached_tokens=self.cached_tokens,
            cost=self.cost,
            duration_ms=self.elapsed_seconds * 1000,
        )

    # ----------------------------------------------------------- child scopes
    def child(self, *, input: Any = None, node_id: str | None = None,
              agent_name: str | None = None,
              permissions: PermissionSet | None = None,
              state: dict[str, Any] | None = None) -> "ExecutionContext":
        """Create a scoped view that shares counters through ``merge_from``.

        Children get their own ``input``/``node_id`` but the *same* budget
        ledger semantics: usage accrued in a child is merged into the parent.
        """
        child = ExecutionContext(
            execution_id=self.execution_id,
            input=self.input if input is None else input,
            state=self.state if state is None else state,
            metadata=dict(self.metadata),
            step=self.step,
            model_calls=self.model_calls,
            tool_calls=self.tool_calls,
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            cached_tokens=self.cached_tokens,
            cost=self.cost,
            policy=self.policy,
            permissions=(self.permissions if permissions is None
                         else self.permissions.intersect(permissions)),
            status=self.status,
            node_id=node_id or self.node_id,
            agent_name=agent_name or self.agent_name,
            parent_execution_id=self.parent_execution_id,
            trace_id=self.trace_id,
            services=self.services,
            started_at=self.started_at,
            deadline=self.deadline,
        )
        return child

    def merge_from(self, child: "ExecutionContext") -> None:
        """Fold a child's consumption back into this context."""
        # counters are absolute in children, so the maximum is the true total
        self.step = max(self.step, child.step)
        self.model_calls = max(self.model_calls, child.model_calls)
        self.tool_calls = max(self.tool_calls, child.tool_calls)
        self.input_tokens = max(self.input_tokens, child.input_tokens)
        self.output_tokens = max(self.output_tokens, child.output_tokens)
        self.cached_tokens = max(self.cached_tokens, child.cached_tokens)
        self.cost = max(self.cost, child.cost)

    def absorb_parallel(self, children: list["ExecutionContext"],
                        baseline: "ExecutionContext") -> None:
        """Merge sibling contexts that ran concurrently (sum of deltas)."""
        for child in children:
            self.model_calls += child.model_calls - baseline.model_calls
            self.tool_calls += child.tool_calls - baseline.tool_calls
            self.input_tokens += child.input_tokens - baseline.input_tokens
            self.output_tokens += child.output_tokens - baseline.output_tokens
            self.cached_tokens += child.cached_tokens - baseline.cached_tokens
            self.cost += child.cost - baseline.cost
            self.step = max(self.step, child.step)

    # ------------------------------------------------------------ enforcement
    def cancel(self) -> None:
        self._cancelled = True

    @property
    def cancelled(self) -> bool:
        return self._cancelled

    def ensure_alive(self) -> None:
        """Called before every model call, tool call and workflow node."""
        if self._cancelled:
            raise ExecutionCancelled("execution cancelled",
                                     execution_id=self.execution_id)
        if self.deadline is not None and time.monotonic() > self.deadline:
            raise ExecutionTimeout(
                "execution timeout exceeded",
                execution_id=self.execution_id,
                timeout_seconds=self.policy.timeout_seconds,
            )

    def next_step(self) -> int:
        self.ensure_alive()
        self.step += 1
        self.policy.check_step(self.step)
        return self.step

    async def check_model_budget(self) -> None:
        self.ensure_alive()
        try:
            self.policy.check_model_call(model_calls=self.model_calls,
                                         tokens=self.total_tokens, cost=self.cost)
        except Exception as exc:
            await self.emit(EventType.BUDGET_EXCEEDED, reason=str(exc), scope="model")
            raise
        await self.maybe_warn_budget()

    async def check_tool_budget(self) -> None:
        self.ensure_alive()
        try:
            self.policy.check_tool_call(tool_calls=self.tool_calls,
                                        tokens=self.total_tokens, cost=self.cost)
        except Exception as exc:
            await self.emit(EventType.BUDGET_EXCEEDED, reason=str(exc), scope="tool")
            raise
        await self.maybe_warn_budget()

    async def maybe_warn_budget(self) -> None:
        """Emit ``budget.warning`` once consumption crosses the threshold."""
        if self._budget_warned:
            return
        budget = self.policy.budget
        utilisation = budget.utilisation(
            cost=self.cost, tokens=self.total_tokens,
            model_calls=self.model_calls, tool_calls=self.tool_calls,
        )
        if utilisation >= budget.warn_at:
            self._budget_warned = True
            await self.emit(
                EventType.BUDGET_WARNING,
                utilisation=round(utilisation, 3),
                remaining=budget.remaining(
                    cost=self.cost, tokens=self.total_tokens,
                    model_calls=self.model_calls, tool_calls=self.tool_calls,
                ),
            )

    # ---------------------------------------------------------------- events
    async def emit(self, event_type: str, **payload: Any) -> Event | None:
        if self.services is None:
            return None
        event = Event(
            execution_id=self.execution_id,
            type=event_type,
            payload=payload,
            step=self.step,
            node_id=self.node_id,
            trace_id=self.trace_id,
        )
        return await self.services.publisher.publish(event)

    # --------------------------------------------------------------- state IO
    def set_state(self, key: str, value: Any) -> None:
        self.state[key] = value

    def get_state(self, key: str, default: Any = None) -> Any:
        return self.state.get(key, default)

    async def update_state(self, key: str, value: Any) -> None:
        self.state[key] = value
        await self.emit(EventType.STATE_UPDATED, key=key)

    # ------------------------------------------------------------ persistence
    def to_dict(self) -> dict[str, Any]:
        return {
            "execution_id": self.execution_id,
            "step": self.step,
            "node_id": self.node_id,
            "agent_name": self.agent_name,
            "status": self.status.value,
            "model_calls": self.model_calls,
            "tool_calls": self.tool_calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cached_tokens": self.cached_tokens,
            "cost": self.cost,
            "state": self.state,
            "metadata": self.metadata,
            "trace_id": self.trace_id,
            "permissions": sorted(self.permissions.granted),
        }
