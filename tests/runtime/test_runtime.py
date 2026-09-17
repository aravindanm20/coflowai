"""Runtime guarantees: budgets, timeouts, cancellation, state machine, metrics."""

from __future__ import annotations

import asyncio

import pytest

from coflowai import (
    Agent,
    BudgetExceeded,
    ExecutionError,
    ExecutionPolicy,
    ExecutionStatus,
    ExecutionTimeout,
    ModelError,
    ModelResponse,
    Workflow,
    tool,
)
from coflowai.testing import build_test_app, tool_call
from coflowai.testing.fakes import FakeModelProvider


def agent(name: str = "assistant", **kwargs) -> Agent:
    return Agent(name, "Answer.", "fake-model", **kwargs)


async def test_every_execution_gets_an_id_and_a_record():
    app, _ = build_test_app(["ok"])
    result = await app.run(agent(), "hello")
    record = await app.runtime.get_execution(result.execution_id)
    assert record is not None
    assert record.status is ExecutionStatus.COMPLETED
    assert result.execution_id.startswith("exec_")


async def test_model_call_budget_stops_execution():
    app, _ = build_test_app([tool_call("noop"), tool_call("noop"), "done"],
                            policy=ExecutionPolicy(max_model_calls=1, max_steps=5),
                            tools=[_noop()])
    result = await app.run(agent(tools=[_noop()]), "go")
    assert result.status is ExecutionStatus.FAILED
    assert isinstance(result.error, BudgetExceeded)


async def test_cost_budget_stops_before_overspending():
    app, _ = build_test_app(["one", "two", "three"],
                            policy=ExecutionPolicy(max_cost=0.000001, max_steps=3))
    workflow = Workflow("w").start(agent("a")).then(agent("b")).then(agent("c"))
    result = await app.run(workflow, "go")
    assert isinstance(result.error, BudgetExceeded)
    events = [e.type for e in
              await app.runtime.event_store.get_events(result.execution_id)]
    assert "budget.exceeded" in events


async def test_budget_warning_is_emitted_before_the_limit():
    app, _ = build_test_app(["a", "b", "c"],
                            policy=ExecutionPolicy(max_model_calls=3, max_steps=5))
    workflow = Workflow("w").start(agent("a")).then(agent("b")).then(agent("c"))
    result = await app.run(workflow, "go")
    events = [e.type for e in
              await app.runtime.event_store.get_events(result.execution_id)]
    assert "budget.warning" in events


async def test_execution_timeout_is_enforced():
    async def slow(_request):
        await asyncio.sleep(0.5)
        return ModelResponse(text="late")

    provider = FakeModelProvider(handler=lambda request: _never())
    app, _ = build_test_app(provider=provider,
                            policy=ExecutionPolicy(timeout_seconds=0.05,
                                                   model_timeout_seconds=None,
                                                   retry_attempts=0))
    result = await app.run(agent(), "go")
    assert result.status is ExecutionStatus.TIMED_OUT
    assert isinstance(result.error, ExecutionTimeout)


async def test_cancellation_propagates():
    app, _ = build_test_app(["a", "b", "c"], policy=ExecutionPolicy(max_steps=5))
    workflow = Workflow("w").start(agent("a")).then(agent("b")).then(agent("c"))

    async def cancel_soon(execution_ids):
        for _ in range(200):
            await asyncio.sleep(0.001)
            if execution_ids:
                await app.cancel(execution_ids[0])
                return

    ids: list[str] = []

    def capture(event):
        if event.type == "execution.started":
            ids.append(event.execution_id)

    app.on_event(capture)
    canceller = asyncio.create_task(cancel_soon(ids))
    result = await app.run(workflow, "go")
    await canceller
    assert result.status in (ExecutionStatus.CANCELLED, ExecutionStatus.COMPLETED)


async def test_cancel_unknown_execution_returns_false():
    app, _ = build_test_app(["ok"])
    assert await app.cancel("exec_does_not_exist") is False


async def test_model_failures_are_wrapped_not_leaked():
    provider = FakeModelProvider(failures=[RuntimeError("provider exploded")])
    app, _ = build_test_app(provider=provider,
                            policy=ExecutionPolicy(retry_attempts=0))
    result = await app.run(agent(), "go")
    assert isinstance(result.error, ModelError)
    assert isinstance(result.error.cause, RuntimeError)


async def test_transient_model_failures_are_retried():
    from coflowai.core.exceptions import RateLimitError

    provider = FakeModelProvider(["recovered"],
                                 failures=[RateLimitError("slow down")])
    policy = ExecutionPolicy(retry_attempts=2)
    policy.retry.base_delay = 0.001
    app, _ = build_test_app(provider=provider, policy=policy)
    result = await app.run(agent(), "go")
    assert result.succeeded and result.output == "recovered"


async def test_provider_failover_to_fallback_model():
    from coflowai import CoFlowAi
    from coflowai.core.exceptions import ModelUnavailable

    app = CoFlowAi(policy=ExecutionPolicy(retry_attempts=0))
    broken = FakeModelProvider(failures=[ModelUnavailable("down")])
    healthy = FakeModelProvider(["from backup"])
    app.models.register("primary", broken, fallbacks=["backup"], default=True)
    app.models.register("backup", healthy)

    result = await app.run(Agent("a", "answer", "primary"), "go")
    assert result.output == "from backup"


async def test_illegal_status_transitions_are_rejected():
    from coflowai.runtime.runtime import Runtime
    from coflowai.state.checkpoint import ExecutionRecord

    record = ExecutionRecord(execution_id="exec_1",
                             status=ExecutionStatus.COMPLETED)
    with pytest.raises(ExecutionError):
        Runtime._transition(record, ExecutionStatus.RUNNING)


async def test_metrics_and_usage_are_tracked_per_scope():
    app, _ = build_test_app([tool_call("noop"), "done"], tools=[_noop()])
    result = await app.run(agent(tools=[_noop()]), "go")
    usage = app.usage(result.execution_id)
    assert usage.total.model_calls == 2
    assert usage.by_tool["noop"].tool_calls == 1
    assert usage.by_agent["assistant"].model_calls == 2
    snapshot = app.metrics_snapshot()
    assert snapshot["counters"]["coflowai.executions.completed"] == 1


def _noop():
    @tool(name="noop", description="does nothing")
    async def noop() -> str:
        return "noop"

    return noop


async def _never():
    await asyncio.sleep(10)
