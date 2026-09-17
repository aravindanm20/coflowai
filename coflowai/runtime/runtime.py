"""The execution runtime — the part of the framework the LLM cannot influence."""

from __future__ import annotations

import asyncio
import copy
import time
from typing import Any, AsyncIterator, Iterable

from ..approvals.approval import APPROVALS_STATE_KEY, ApprovalRecord, ApprovalRequired
from ..core.context import ExecutionContext, RuntimeServices
from ..core.exceptions import (
    ApprovalRejected,
    BudgetExceeded,
    CheckpointError,
    ExecutionCancelled,
    ExecutionError,
    ExecutionTimeout,
    MaxStepsExceeded,
)
from ..core.executable import Executable, as_executable
from ..core.ids import execution_id as new_execution_id
from ..core.ids import new_id
from ..core.ids import trace_id as new_trace_id
from ..core.result import ExecutionResult
from ..core.types import (
    RESUMABLE_STATUSES,
    TERMINAL_STATUSES,
    VALID_TRANSITIONS,
    ExecutionStatus,
    Usage,
)
from ..events.event import EventType
from ..events.publisher import EventPublisher
from ..events.store import EventStore, InMemoryEventStore
from ..memory.base import MemoryStore
from ..models.gateway import MODEL_OVERRIDES_KEY, ModelGateway
from ..models.registry import ModelRegistry
from ..observability.logger import get_logger
from ..observability.metrics import InMemoryMetrics, MetricsSink
from ..observability.tracer import Tracer
from ..observability.usage import UsageTracker
from ..policies.permissions import PermissionSet
from ..policies.policy import ExecutionPolicy
from ..state.checkpoint import Checkpoint, ExecutionRecord
from ..state.store import InMemoryStateStore, StateStore
from ..tools.executor import TOOL_OVERRIDES_KEY, ToolExecutor
from ..tools.registry import ToolRegistry
from ..tools.sandbox import Sandbox
from .replay import ReplayTrace

__all__ = ["Runtime"]

_log = get_logger("runtime")


class Runtime:
    """Creates executions, enforces policy, persists everything important."""

    _default: "Runtime | None" = None

    def __init__(self, *,
                 event_store: EventStore | None = None,
                 state_store: StateStore | None = None,
                 model_registry: ModelRegistry | None = None,
                 tool_registry: ToolRegistry | None = None,
                 memory: MemoryStore | None = None,
                 metrics: MetricsSink | None = None,
                 sandbox: Sandbox | None = None,
                 default_policy: ExecutionPolicy | None = None,
                 permissions: PermissionSet | Iterable[str] | None = None) -> None:
        self.event_store = event_store or InMemoryEventStore()
        self.publisher = EventPublisher(self.event_store)
        self.state_store = state_store or InMemoryStateStore()
        self.models = model_registry or ModelRegistry()
        self.tools = tool_registry or ToolRegistry()
        self.memory = memory
        self.metrics = metrics or InMemoryMetrics()
        self.gateway = ModelGateway(self.models)
        self.tool_executor = ToolExecutor(self.tools, sandbox=sandbox)
        self.default_policy = default_policy or ExecutionPolicy()
        self.permissions = (permissions if isinstance(permissions, PermissionSet)
                            else PermissionSet(permissions)
                            if permissions is not None else PermissionSet.all())

        self._running: dict[str, ExecutionContext] = {}
        self._executables: dict[str, Executable] = {}
        self._policies: dict[str, ExecutionPolicy] = {}
        self._tracers: dict[str, Tracer] = {}
        self._usage: dict[str, UsageTracker] = {}

    # ------------------------------------------------------------- factories
    @classmethod
    def default(cls) -> "Runtime":
        if cls._default is None:
            cls._default = cls()
        return cls._default

    @classmethod
    def reset_default(cls) -> None:
        cls._default = None

    # ------------------------------------------------------------------- run
    async def run(self, executable: Any, input: Any = None, *,
                  policy: ExecutionPolicy | None = None,
                  execution_id: str | None = None,
                  permissions: PermissionSet | Iterable[str] | None = None,
                  metadata: dict[str, Any] | None = None,
                  state: dict[str, Any] | None = None) -> ExecutionResult:
        target = as_executable(executable)
        policy = policy or self.default_policy
        exec_id = execution_id or new_execution_id()

        granted = (permissions if isinstance(permissions, PermissionSet)
                   else PermissionSet(permissions) if permissions is not None
                   else self.permissions)

        record = ExecutionRecord(
            execution_id=exec_id,
            status=ExecutionStatus.CREATED,
            input=input,
            executable=target.name,
            workflow_hash=getattr(target, "hash", None),
            policy=policy.to_dict(),
            metadata=dict(metadata or {}),
        )
        await self.state_store.save_execution(record)

        context = ExecutionContext(
            execution_id=exec_id,
            input=input,
            state=dict(state or {}),
            metadata=dict(metadata or {}),
            policy=policy,
            permissions=granted,
            trace_id=new_trace_id(),
        )
        context.services = self._services(context)

        await context.emit(EventType.EXECUTION_CREATED, executable=target.name,
                           policy=policy.to_dict())
        return await self._drive(target, context, record)

    # ------------------------------------------------------------- streaming
    async def stream(self, executable: Any, input: Any = None, *,
                     policy: ExecutionPolicy | None = None,
                     execution_id: str | None = None,
                     permissions: PermissionSet | Iterable[str] | None = None,
                     metadata: dict[str, Any] | None = None) -> AsyncIterator[Any]:
        """Stream an agent execution chunk by chunk.

        The same policy engine, event log and usage accounting apply; the final
        chunk carries the completed :class:`ExecutionResult`.
        """
        target = as_executable(executable)
        if not hasattr(target, "execute_stream"):
            raise ExecutionError(f"{target.name} does not support streaming",
                                 executable=target.name)

        policy = policy or self.default_policy
        exec_id = execution_id or new_execution_id()
        granted = (permissions if isinstance(permissions, PermissionSet)
                   else PermissionSet(permissions) if permissions is not None
                   else self.permissions)

        record = ExecutionRecord(execution_id=exec_id,
                                 status=ExecutionStatus.CREATED, input=input,
                                 executable=target.name, policy=policy.to_dict(),
                                 metadata=dict(metadata or {}))
        await self.state_store.save_execution(record)

        context = ExecutionContext(execution_id=exec_id, input=input,
                                   metadata=dict(metadata or {}), policy=policy,
                                   permissions=granted, trace_id=new_trace_id())
        context.services = self._services(context)
        self._executables[exec_id] = target
        self._policies[exec_id] = policy
        self._running[exec_id] = context

        started = time.perf_counter()
        self._transition(record, ExecutionStatus.RUNNING)
        context.status = ExecutionStatus.RUNNING
        await self.state_store.save_execution(record)
        await context.emit(EventType.EXECUTION_CREATED, executable=target.name,
                           streaming=True)
        await context.emit(EventType.EXECUTION_STARTED, executable=target.name,
                           streaming=True)

        try:
            async for chunk in target.execute_stream(context):
                if chunk.is_final:
                    result = chunk.metadata.get("result")
                    if result is not None:
                        await self._complete(record, context, result, started)
                        chunk.metadata["result"] = result
                yield chunk
        except BaseException as exc:  # noqa: BLE001 - mapped to a status
            if isinstance(exc, asyncio.CancelledError):
                raise
            await self._fail(record, context, exc, started)
            raise
        finally:
            self._running.pop(exec_id, None)

    # ---------------------------------------------------------------- resume
    async def resume(self, execution_id: str, *, executable: Any = None,
                     policy: ExecutionPolicy | None = None) -> ExecutionResult:
        record = await self._require_record(execution_id)
        if record.status not in RESUMABLE_STATUSES:
            raise ExecutionError(
                f"execution is {record.status.value} and cannot be resumed",
                execution_id=execution_id)

        target = as_executable(executable) if executable is not None \
            else self._executables.get(execution_id)
        if target is None:
            raise ExecutionError(
                "cannot resume: the executable is unknown to this runtime "
                "(pass executable=...)",
                execution_id=execution_id,
            )

        checkpoint = await self.state_store.latest_checkpoint(execution_id)
        if checkpoint is None:
            raise CheckpointError("no checkpoint available for resume",
                                  execution_id=execution_id)

        # Rule 9: verify workflow compatibility before touching anything.
        current_hash = getattr(target, "hash", None)
        if checkpoint.workflow_hash and current_hash and \
                checkpoint.workflow_hash != current_hash:
            raise CheckpointError(
                "workflow definition changed; refusing to resume silently",
                execution_id=execution_id,
                checkpoint_hash=checkpoint.workflow_hash,
                current_hash=current_hash,
            )

        policy = policy or self._policies.get(execution_id) or self.default_policy
        context = ExecutionContext(
            execution_id=execution_id,
            input=record.input,
            state=copy.deepcopy(checkpoint.state),
            metadata=dict(record.metadata),
            policy=policy,
            permissions=self.permissions,
            step=checkpoint.step,
            model_calls=checkpoint.model_calls,
            tool_calls=checkpoint.tool_calls,
            input_tokens=checkpoint.input_tokens,
            output_tokens=checkpoint.output_tokens,
            cost=checkpoint.cost,
        )
        context.services = self._services(context)

        await context.emit(EventType.CHECKPOINT_RESTORED,
                           checkpoint_id=checkpoint.checkpoint_id,
                           step=checkpoint.step, node_id=checkpoint.node_id)
        await context.emit(EventType.EXECUTION_RESUMED,
                           from_step=checkpoint.step,
                           checkpoint_id=checkpoint.checkpoint_id)
        return await self._drive(target, context, record, resumed=True)

    # -------------------------------------------------------------- approvals
    async def approve(self, execution_id: str, *, approval_id: str | None = None,
                      by: str | None = None, note: str | None = None,
                      executable: Any = None) -> ExecutionResult:
        await self._decide(execution_id, approval_id=approval_id, decision="approved",
                           by=by, note=note)
        return await self.resume(execution_id, executable=executable)

    async def reject(self, execution_id: str, *, approval_id: str | None = None,
                     by: str | None = None, note: str | None = None,
                     executable: Any = None) -> ExecutionResult:
        await self._decide(execution_id, approval_id=approval_id, decision="rejected",
                           by=by, note=note)
        return await self.resume(execution_id, executable=executable)

    async def pending_approvals(self, execution_id: str) -> list[ApprovalRecord]:
        checkpoint = await self.state_store.latest_checkpoint(execution_id)
        if checkpoint is None:
            return []
        ledger = checkpoint.state.get(APPROVALS_STATE_KEY, {})
        return [ApprovalRecord.from_dict(v) for v in ledger.values()
                if v.get("status") == "pending"]

    async def _decide(self, execution_id: str, *, approval_id: str | None,
                      decision: str, by: str | None, note: str | None) -> None:
        record = await self._require_record(execution_id)
        if record.status is not ExecutionStatus.WAITING_FOR_APPROVAL:
            raise ExecutionError("execution is not waiting for approval",
                                 execution_id=execution_id,
                                 status=record.status.value)
        checkpoint = await self.state_store.latest_checkpoint(execution_id)
        if checkpoint is None:  # pragma: no cover - defensive
            raise CheckpointError("no checkpoint to apply the decision to",
                                  execution_id=execution_id)

        ledger = checkpoint.state.setdefault(APPROVALS_STATE_KEY, {})
        pending = [k for k, v in ledger.items() if v.get("status") == "pending"]
        target_id = approval_id or (pending[0] if pending else None)
        if target_id is None or target_id not in ledger:
            raise ExecutionError("no pending approval with that id",
                                 execution_id=execution_id, approval_id=approval_id)

        entry = ApprovalRecord.from_dict(ledger[target_id])
        entry.status = decision
        entry.decided_by = by
        entry.note = note
        entry.decided_at = _now_iso()
        ledger[target_id] = entry.to_dict()

        checkpoint.reason = f"approval_{decision}"
        checkpoint.checkpoint_id = new_id("ckpt")
        await self.state_store.save_checkpoint(checkpoint)

        event = (EventType.APPROVAL_APPROVED if decision == "approved"
                 else EventType.APPROVAL_REJECTED)
        await self.publisher.publish(_event(execution_id, event,
                                            approval_id=target_id, by=by, note=note,
                                            step=checkpoint.step))

    # ------------------------------------------------------------ cancellation
    async def cancel(self, execution_id: str, *, reason: str = "cancelled by operator"
                     ) -> bool:
        context = self._running.get(execution_id)
        if context is not None:
            context.cancel()
            await context.emit(EventType.EXECUTION_CANCELLED, reason=reason)
            return True
        record = await self.state_store.get_execution(execution_id)
        if record is None or record.status in TERMINAL_STATUSES:
            return False
        self._transition(record, ExecutionStatus.CANCELLED)
        record.error = reason
        await self.state_store.save_execution(record)
        await self.publisher.publish(
            _event(execution_id, EventType.EXECUTION_CANCELLED, reason=reason))
        return True

    # ---------------------------------------------------------------- replay
    async def replay(self, execution_id: str) -> ReplayTrace:
        events = await self.event_store.get_events(execution_id)
        if not events:
            raise ExecutionError("no events recorded for this execution",
                                 execution_id=execution_id)
        return ReplayTrace.from_events(execution_id, events)

    async def fork(self, execution_id: str, *, from_step: int,
                   overrides: dict[str, Any] | None = None,
                   executable: Any = None,
                   policy: ExecutionPolicy | None = None) -> ExecutionResult:
        """Re-run an execution from a checkpoint with modifications."""
        record = await self._require_record(execution_id)
        checkpoint = await self.state_store.checkpoint_at_step(execution_id,
                                                               from_step)
        if checkpoint is None and from_step > 0:
            raise CheckpointError("no checkpoint at or before that step",
                                  execution_id=execution_id, from_step=from_step)

        overrides = dict(overrides or {})
        target = as_executable(executable) if executable is not None \
            else self._executables.get(execution_id)
        if target is None:
            raise ExecutionError("cannot fork: unknown executable",
                                 execution_id=execution_id)

        if "model" in overrides:
            target = _rebind_model(target, overrides["model"])

        # from_step=0 means "re-run from the very beginning" — fresh state.
        state = copy.deepcopy(checkpoint.state) if checkpoint else {}
        state.update(overrides.get("state", {}))
        if "model_responses" in overrides:
            state[MODEL_OVERRIDES_KEY] = overrides["model_responses"]
        if "tool_results" in overrides:
            state[TOOL_OVERRIDES_KEY] = overrides["tool_results"]
        if "instructions" in overrides:
            target = _rebind_instructions(target, overrides["instructions"])
        # A fork intentionally changes the definition (model, prompt, ...), so the
        # inherited workflow hashes are dropped rather than silently mismatched.
        for key in [k for k in state if k.startswith("__workflow_hash__")]:
            state.pop(key)
        fork_input = overrides.get("input", record.input)
        fork_id = new_execution_id()

        await self.publisher.publish(_event(
            execution_id, EventType.EXECUTION_FORKED, fork_id=fork_id,
            from_step=from_step, overrides=sorted(overrides)))

        return await self.run(
            target, fork_input,
            policy=policy or overrides.get("policy")
            or self._policies.get(execution_id) or self.default_policy,
            execution_id=fork_id,
            metadata={**record.metadata, "forked_from": execution_id,
                      "forked_at_step": from_step},
            state=state,
        )

    # ----------------------------------------------------------- checkpoints
    async def create_checkpoint(self, context: ExecutionContext, *,
                                node_id: str | None = None,
                                reason: str = "auto") -> Checkpoint | None:
        if not context.policy.checkpoint_enabled:
            return None
        checkpoint = Checkpoint(
            execution_id=context.execution_id,
            step=context.step,
            node_id=node_id or context.node_id,
            state=copy.deepcopy(context.state),
            model_calls=context.model_calls,
            tool_calls=context.tool_calls,
            input_tokens=context.input_tokens,
            output_tokens=context.output_tokens,
            cost=context.cost,
            status=context.status,
            workflow_hash=context.state.get("__workflow_hash__"),
            reason=reason,
        )
        stored = await self.state_store.save_checkpoint(checkpoint)
        await context.emit(EventType.CHECKPOINT_CREATED,
                           checkpoint_id=stored.checkpoint_id,
                           node_id=stored.node_id, reason=reason)
        return stored

    # ------------------------------------------------------------ inspection
    async def get_execution(self, execution_id: str) -> ExecutionRecord | None:
        return await self.state_store.get_execution(execution_id)

    async def list_executions(self, *, status: str | None = None,
                              limit: int = 100) -> list[ExecutionRecord]:
        return await self.state_store.list_executions(status=status, limit=limit)

    def trace(self, execution_id: str) -> Tracer | None:
        return self._tracers.get(execution_id)

    def usage(self, execution_id: str) -> UsageTracker | None:
        return self._usage.get(execution_id)

    # ------------------------------------------------------------------ core
    def _services(self, context: ExecutionContext) -> RuntimeServices:
        tracer = Tracer(context.execution_id, context.trace_id)
        usage = UsageTracker()
        self._tracers[context.execution_id] = tracer
        self._usage[context.execution_id] = usage
        return RuntimeServices(
            publisher=self.publisher,
            state_store=self.state_store,
            models=self.models,
            tools=self.tools,
            gateway=self.gateway,
            tool_executor=self.tool_executor,
            tracer=tracer,
            metrics=self.metrics,
            usage=usage,
            memory=self.memory,
            runtime=self,
        )

    async def _drive(self, target: Executable, context: ExecutionContext,
                     record: ExecutionRecord, *, resumed: bool = False
                     ) -> ExecutionResult:
        execution_id = context.execution_id
        self._executables[execution_id] = target
        self._policies[execution_id] = context.policy
        self._running[execution_id] = context

        if not await self.state_store.acquire_lock(execution_id):
            raise ExecutionError("execution is already running elsewhere",
                                 execution_id=execution_id)

        started = time.perf_counter()
        self._transition(record, ExecutionStatus.RUNNING)
        context.status = ExecutionStatus.RUNNING
        await self.state_store.save_execution(record)
        await context.emit(EventType.EXECUTION_STARTED, executable=target.name,
                           resumed=resumed)
        self.metrics.increment("coflowai.executions.started")

        tracer = context.services.tracer  # type: ignore[union-attr]
        result: ExecutionResult
        try:
            async with tracer.span(f"execution:{target.name}", kind="execution",
                                   execution_id=execution_id):
                result = await self._with_deadline(target, context)
            result.status = ExecutionStatus.COMPLETED if result.status is None \
                else result.status
            await self._complete(record, context, result, started)

        except ApprovalRequired as pause:
            result = await self._pause_for_approval(record, context, pause, started)

        except asyncio.CancelledError:
            raise

        except BaseException as exc:  # noqa: BLE001 - mapped to statuses below
            result = await self._fail(record, context, exc, started)

        finally:
            self._running.pop(execution_id, None)
            await self.state_store.release_lock(execution_id)

        return result

    async def _with_deadline(self, target: Executable,
                             context: ExecutionContext) -> ExecutionResult:
        timeout = context.policy.timeout_seconds
        if timeout is None:
            return await target.execute(context)
        try:
            async with asyncio.timeout(timeout):
                return await target.execute(context)
        except TimeoutError as exc:
            raise ExecutionTimeout("execution timeout exceeded", cause=exc,
                                   execution_id=context.execution_id,
                                   timeout_seconds=timeout) from exc

    async def _complete(self, record: ExecutionRecord, context: ExecutionContext,
                        result: ExecutionResult, started: float) -> None:
        duration_ms = (time.perf_counter() - started) * 1000
        usage = self._final_usage(context, duration_ms)
        result.usage = usage
        result.execution_id = context.execution_id
        for key, value in context.metadata.items():
            result.metadata.setdefault(key, value)
        result.metadata.setdefault("duration_ms", round(duration_ms, 3))

        if result.status in TERMINAL_STATUSES or result.paused:
            status = result.status
        else:  # pragma: no cover - defensive
            status = ExecutionStatus.COMPLETED
        self._transition(record, status)
        record.output = result.output
        await self.state_store.save_execution(record)

        context.status = status
        if context.policy.checkpoint_enabled:
            await self.create_checkpoint(context, reason="final")
        await context.emit(EventType.EXECUTION_COMPLETED,
                           status=status.value,
                           duration_ms=round(duration_ms, 3),
                           usage=usage.to_dict())
        self.metrics.increment("coflowai.executions.completed")
        self.metrics.observe("coflowai.execution.duration_ms", duration_ms)

    async def _pause_for_approval(self, record: ExecutionRecord,
                                  context: ExecutionContext,
                                  pause: ApprovalRequired,
                                  started: float) -> ExecutionResult:
        context.status = ExecutionStatus.WAITING_FOR_APPROVAL
        checkpoint = await self.create_checkpoint(
            context, node_id=pause.node_id, reason="approval")
        self._transition(record, ExecutionStatus.WAITING_FOR_APPROVAL)
        await self.state_store.save_execution(record)

        await context.emit(EventType.APPROVAL_REQUESTED,
                           approval_id=pause.approval_id, message=pause.prompt,
                           node_id=pause.node_id, payload=pause.payload)
        await context.emit(EventType.EXECUTION_PAUSED, reason="approval",
                           approval_id=pause.approval_id)
        self.metrics.increment("coflowai.executions.paused", reason="approval")

        duration_ms = (time.perf_counter() - started) * 1000
        return ExecutionResult(
            execution_id=context.execution_id,
            status=ExecutionStatus.WAITING_FOR_APPROVAL,
            output=None,
            usage=self._final_usage(context, duration_ms),
            checkpoint_id=checkpoint.checkpoint_id if checkpoint else None,
            metadata={"approval_id": pause.approval_id,
                      "message": pause.prompt,
                      "node_id": pause.node_id},
        )

    async def _fail(self, record: ExecutionRecord, context: ExecutionContext,
                    error: BaseException, started: float) -> ExecutionResult:
        status = _status_for(error)
        duration_ms = (time.perf_counter() - started) * 1000
        usage = self._final_usage(context, duration_ms)

        context.status = status
        if context.policy.checkpoint_enabled:
            await self.create_checkpoint(context, reason=f"failure:{status.value}")

        self._transition(record, status)
        record.error = repr(error)
        await self.state_store.save_execution(record)

        event_type = {
            ExecutionStatus.TIMED_OUT: EventType.EXECUTION_TIMEOUT,
            ExecutionStatus.CANCELLED: EventType.EXECUTION_CANCELLED,
        }.get(status, EventType.EXECUTION_FAILED)
        await context.emit(event_type, error=repr(error), status=status.value,
                           duration_ms=round(duration_ms, 3))
        self.metrics.increment("coflowai.executions.failed",
                               status=status.value, error=type(error).__name__)
        _log.warning("execution.failed", execution_id=context.execution_id,
                     status=status.value, error=repr(error))

        return ExecutionResult(
            execution_id=context.execution_id,
            status=status,
            output=None,
            error=error if isinstance(error, Exception) else None,
            usage=usage,
            metadata={"duration_ms": round(duration_ms, 3)},
        )

    def _final_usage(self, context: ExecutionContext, duration_ms: float) -> Usage:
        tracker = self._usage.get(context.execution_id)
        usage = tracker.total if tracker else context.usage_snapshot()
        usage.duration_ms = duration_ms
        return usage

    @staticmethod
    def _transition(record: ExecutionRecord, status: ExecutionStatus) -> None:
        """Enforce the documented execution state machine (§11)."""
        if record.status is status:
            return
        allowed = VALID_TRANSITIONS.get(record.status, frozenset())
        if status not in allowed:
            raise ExecutionError(
                f"illegal status transition {record.status.value} -> {status.value}",
                execution_id=record.execution_id,
            )
        record.touch(status)

    async def _require_record(self, execution_id: str) -> ExecutionRecord:
        record = await self.state_store.get_execution(execution_id)
        if record is None:
            raise ExecutionError("unknown execution", execution_id=execution_id)
        return record


# ------------------------------------------------------------------ helpers
def _status_for(error: BaseException) -> ExecutionStatus:
    if isinstance(error, ExecutionTimeout):
        return ExecutionStatus.TIMED_OUT
    if isinstance(error, ExecutionCancelled):
        return ExecutionStatus.CANCELLED
    if isinstance(error, (BudgetExceeded, MaxStepsExceeded, ApprovalRejected)):
        return ExecutionStatus.FAILED
    return ExecutionStatus.FAILED


def _event(execution_id: str, event_type: str, *, step: int = 0, **payload: Any):
    from ..events.event import Event

    return Event(execution_id=execution_id, type=event_type, payload=payload,
                 step=step)


def _now_iso() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def _rebind_instructions(target: Executable, instructions: str) -> Executable:
    """Return a copy of ``target`` whose agents use a different prompt."""
    from ..agents.agent import Agent
    from ..workflows.workflow import Workflow

    if isinstance(target, Agent):
        clone = copy.copy(target)
        clone.instructions = instructions
        return clone
    if isinstance(target, Workflow):
        clone = copy.copy(target)
        clone._nodes = [copy.copy(node) for node in target._nodes]
        for node in clone._nodes:
            node.executable = _rebind_instructions(node.executable, instructions)
        clone._graph = None
        return clone
    return target


def _rebind_model(target: Executable, model: str) -> Executable:
    """Return a copy of ``target`` whose agents use a different model."""
    from ..agents.agent import Agent
    from ..workflows.workflow import Workflow

    if isinstance(target, Agent):
        clone = copy.copy(target)
        clone.model = model
        return clone
    if isinstance(target, Workflow):
        clone = copy.copy(target)
        clone._nodes = [copy.copy(node) for node in target._nodes]
        for node in clone._nodes:
            node.executable = _rebind_model(node.executable, model)
        clone._graph = None
        return clone
    return target
