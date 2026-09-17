"""Tool runtime.

Pipeline (§29) — no step is optional:

    request -> registry -> permission -> validation -> rate limit -> timeout
            -> sandbox -> execute -> output validation -> audit event
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from typing import Any

from ..core.context import ExecutionContext
from ..core.exceptions import (
    ExecutionTimeout,
    PermissionDenied,
    ToolError,
    ToolNotFound,
    ToolTimeout,
    ValidationError,
    is_retryable,
)
from ..core.ids import idempotency_key
from ..core.types import ToolCall, ToolResult
from ..events.event import EventType
from ..observability.redaction import redact
from ..policies.policy import RetryPolicy
from ..core.retry import run_with_retry, run_with_timeout
from .registry import ToolRegistry
from .sandbox import InProcessSandbox, Sandbox

__all__ = ["ToolExecutor", "TOOL_OVERRIDES_KEY"]

_IDEMPOTENCY_STATE_KEY = "__idempotency__"

#: ``state[TOOL_OVERRIDES_KEY]`` maps a tool name to a canned result, used by
#: ``runtime.fork(..., overrides={"tool_results": {...}})``.  The permission and
#: validation pipeline still runs; only the execution step is substituted.
TOOL_OVERRIDES_KEY = "__tool_overrides__"


class ToolExecutor:
    def __init__(self, registry: ToolRegistry, *, sandbox: Sandbox | None = None
                 ) -> None:
        self.registry = registry
        self.sandbox = sandbox or InProcessSandbox()
        self._rate_windows: dict[str, deque[float]] = defaultdict(deque)

    async def execute(self, context: ExecutionContext, call: ToolCall, *,
                      allowed_tools: set[str] | None = None) -> ToolResult:
        """Execute one model-requested tool call.  Never raises for tool
        failures — it returns a ``ToolResult`` the agent can feed back to the
        model.  Policy violations (budget/cancel) *do* propagate."""
        started = time.perf_counter()
        await context.emit(EventType.TOOL_REQUESTED, tool=call.name,
                           tool_call_id=call.id, agent=context.agent_name,
                           arguments=redact(call.arguments))
        try:
            await context.check_tool_budget()
            tool = self._resolve(call, allowed_tools)
            self._authorise(context, tool, call)
            arguments = tool.validate_arguments(call.arguments)
            self._rate_limit(tool.definition.name,
                             tool.definition.rate_limit_per_minute)

            overrides = context.state.get(TOOL_OVERRIDES_KEY) or {}
            if call.name in overrides:
                return await self._finish(context, call, overrides[call.name],
                                          started, replayed=True)

            cached = self._idempotent_hit(context, tool, call)
            if cached is not None:
                return await self._finish(context, call, cached, started,
                                          replayed=True)

            output = await self._run(context, tool, arguments, call)
            output = tool.validate_output(output)
            self._remember_idempotent(context, tool, call, output)
            return await self._finish(context, call, output, started)

        except (PermissionDenied,) as exc:
            await context.emit(EventType.TOOL_DENIED, tool=call.name,
                               tool_call_id=call.id, reason=str(exc))
            return self._failure(call, exc, started, context)
        except (ToolNotFound, ValidationError, ToolError) as exc:
            await context.emit(EventType.TOOL_FAILED, tool=call.name,
                               tool_call_id=call.id, error=repr(exc))
            return self._failure(call, exc, started, context)

    # ----------------------------------------------------------- pipeline bits
    def _resolve(self, call: ToolCall, allowed: set[str] | None):
        if allowed is not None and call.name not in allowed:
            raise ToolNotFound(f"tool '{call.name}' is not available to this agent",
                               tool=call.name, available=sorted(allowed))
        return self.registry.get(call.name)

    @staticmethod
    def _authorise(context: ExecutionContext, tool, call: ToolCall) -> None:
        required = tool.definition.permission
        if not context.permissions.allows(required):
            raise PermissionDenied(
                f"agent is not authorised to call '{tool.name}'",
                tool=tool.name, required=required,
                granted=sorted(context.permissions.granted),
            )

    def _rate_limit(self, name: str, limit: int | None) -> None:
        if not limit:
            return
        now = time.monotonic()
        window = self._rate_windows[name]
        while window and now - window[0] > 60:
            window.popleft()
        if len(window) >= limit:
            raise ToolError(f"rate limit exceeded for tool '{name}'",
                            tool=name, limit=limit)
        window.append(now)

    async def _run(self, context: ExecutionContext, tool,
                   arguments: dict[str, Any], call: ToolCall) -> Any:
        timeout = _tool_timeout(context, tool.definition.timeout)
        retry = self._retry_policy(context, tool)

        async def attempt(attempt_number: int) -> Any:
            context.ensure_alive()
            metadata = {
                "execution_id": context.execution_id,
                "node_id": context.node_id,
                "tool_call_id": call.id,
                "idempotency_key": idempotency_key(context.execution_id,
                                                   context.node_id, call.id),
                "attempt": attempt_number,
            }
            try:
                return await run_with_timeout(
                    lambda: self.sandbox.run(tool, arguments, metadata=metadata),
                    timeout, what=f"tool '{tool.name}'", tool=tool.name,
                )
            except ExecutionTimeout as exc:
                # a tool deadline is a tool failure, not an execution failure
                raise ToolTimeout(str(exc), cause=exc, tool=tool.name) from exc
            except Exception as exc:
                if type(exc).__module__.startswith("coflowai."):
                    raise
                raise ToolError(f"tool '{tool.name}' raised {type(exc).__name__}",
                                cause=exc, tool=tool.name) from exc

        async def on_retry(attempt_number: int, error: BaseException,
                           delay: float) -> None:
            await context.emit(EventType.TOOL_RETRYING, tool=tool.name,
                               tool_call_id=call.id, attempt=attempt_number,
                               delay=round(delay, 3), error=repr(error))

        await context.emit(EventType.TOOL_STARTED, tool=tool.name,
                           tool_call_id=call.id, sandbox=self.sandbox.name)

        tracer = context.services.tracer if context.services else None
        if tracer is not None:
            async with tracer.span(f"tool:{tool.name}", kind="tool",
                                   tool=tool.name, call_id=call.id):
                return await run_with_retry(attempt, retry, on_retry=on_retry,
                                            retryable=is_retryable)
        return await run_with_retry(attempt, retry, on_retry=on_retry,
                                    retryable=is_retryable)

    @staticmethod
    def _retry_policy(context: ExecutionContext, tool) -> RetryPolicy:
        """Rule 8: never silently retry non-idempotent side effects."""
        definition = tool.definition
        if definition.side_effect and not definition.idempotent:
            return RetryPolicy.none()
        attempts = definition.retries
        base = context.policy.retry
        return RetryPolicy(attempts=attempts, backoff=base.backoff,
                           base_delay=base.base_delay, max_delay=base.max_delay,
                           jitter=base.jitter)

    # --------------------------------------------------------- idempotency
    @staticmethod
    def _key(context: ExecutionContext, call: ToolCall) -> str:
        return idempotency_key(context.execution_id, context.node_id, call.id)

    def _idempotent_hit(self, context: ExecutionContext, tool,
                        call: ToolCall) -> Any:
        if not tool.definition.side_effect:
            return None
        ledger = context.state.get(_IDEMPOTENCY_STATE_KEY, {})
        return ledger.get(self._key(context, call))

    def _remember_idempotent(self, context: ExecutionContext, tool, call: ToolCall,
                             output: Any) -> None:
        if not tool.definition.side_effect:
            return
        ledger = context.state.setdefault(_IDEMPOTENCY_STATE_KEY, {})
        ledger[self._key(context, call)] = output

    # ------------------------------------------------------------- completion
    async def _finish(self, context: ExecutionContext, call: ToolCall, output: Any,
                      started: float, *, replayed: bool = False) -> ToolResult:
        duration_ms = (time.perf_counter() - started) * 1000
        context.tool_calls += 1
        if context.services is not None:
            context.services.usage.record_tool(
                tool=call.name, agent=context.agent_name,
                node_id=context.node_id, latency_ms=duration_ms,
            )
            context.services.metrics.increment("coflowai.tool.calls", tool=call.name)
            context.services.metrics.observe("coflowai.tool.latency_ms", duration_ms,
                                             tool=call.name)
        await context.maybe_warn_budget()
        await context.emit(EventType.TOOL_COMPLETED, tool=call.name,
                           tool_call_id=call.id, duration_ms=round(duration_ms, 3),
                           replayed=replayed, output=redact(output))
        return ToolResult(call_id=call.id, name=call.name, output=output,
                          duration_ms=duration_ms)

    @staticmethod
    def _failure(call: ToolCall, error: Exception, started: float,
                 context: ExecutionContext) -> ToolResult:
        duration_ms = (time.perf_counter() - started) * 1000
        context.tool_calls += 1
        if context.services is not None:
            context.services.metrics.increment("coflowai.tool.failures",
                                               tool=call.name,
                                               error=type(error).__name__)
        return ToolResult(call_id=call.id, name=call.name, error=str(error),
                          duration_ms=duration_ms)


def _tool_timeout(context: ExecutionContext, tool_timeout: float | None
                  ) -> float | None:
    candidates = [t for t in (tool_timeout, context.policy.tool_timeout_seconds)
                  if t]
    remaining = context.remaining_seconds
    if remaining is not None:
        candidates.append(max(0.01, remaining))
    return min(candidates) if candidates else None
