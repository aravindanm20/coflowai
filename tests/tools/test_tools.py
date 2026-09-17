"""Tool runtime: schema derivation, validation, permissions, retries, idempotency."""

from __future__ import annotations

import asyncio

import pytest

from coflowai import (
    ExecutionContext,
    ExecutionPolicy,
    FakeTool,
    PermissionSet,
    ToolCall,
    ToolRegistry,
    tool,
)
from coflowai.core.context import RuntimeServices
from coflowai.events.publisher import EventPublisher
from coflowai.events.store import InMemoryEventStore
from coflowai.models.registry import ModelRegistry
from coflowai.observability.metrics import InMemoryMetrics
from coflowai.observability.tracer import Tracer
from coflowai.observability.usage import UsageTracker
from coflowai.state.store import InMemoryStateStore
from coflowai.tools.executor import ToolExecutor


@tool(description="Add two numbers", permission="math.compute")
async def add(a: int, b: int = 0) -> int:
    return a + b


def make_context(registry: ToolRegistry, permissions=("*",), **policy_kwargs):
    executor = ToolExecutor(registry)
    context = ExecutionContext(policy=ExecutionPolicy(**policy_kwargs),
                               permissions=PermissionSet(permissions))
    context.services = RuntimeServices(
        publisher=EventPublisher(InMemoryEventStore()),
        state_store=InMemoryStateStore(),
        models=ModelRegistry(),
        tools=registry,
        gateway=None,
        tool_executor=executor,
        tracer=Tracer(context.execution_id, context.trace_id),
        metrics=InMemoryMetrics(),
        usage=UsageTracker(),
    )
    return context, executor


def test_schema_is_derived_from_type_hints():
    schema = add.definition.input_schema
    assert schema["properties"]["a"]["type"] == "integer"
    assert schema["required"] == ["a"]
    assert add.definition.permission == "math.compute"


async def test_tool_executes_and_emits_audit_events():
    registry = ToolRegistry([add])
    context, executor = make_context(registry)
    result = await executor.execute(context, ToolCall("c1", "add", {"a": 2, "b": 3}))
    assert result.ok and result.output == 5
    assert context.tool_calls == 1

    events = [e.type for e in
              await context.services.publisher.store.get_events(context.execution_id)]
    assert events == ["tool.requested", "tool.started", "tool.completed"]


async def test_invalid_arguments_are_rejected_not_executed():
    registry = ToolRegistry([add])
    context, executor = make_context(registry)
    result = await executor.execute(context, ToolCall("c1", "add", {"a": "banana"}))
    assert not result.ok and "invalid arguments" in result.error


async def test_permission_denied_is_not_executed():
    registry = ToolRegistry([add])
    context, executor = make_context(registry, permissions=("weather.read",))
    result = await executor.execute(context, ToolCall("c1", "add", {"a": 1}))
    assert not result.ok and "not authorised" in result.error
    events = [e.type for e in
              await context.services.publisher.store.get_events(context.execution_id)]
    assert "tool.denied" in events


async def test_agent_cannot_call_tools_outside_its_allow_list():
    registry = ToolRegistry([add])
    context, executor = make_context(registry)
    result = await executor.execute(context, ToolCall("c1", "add", {"a": 1}),
                                    allowed_tools=set())
    assert not result.ok and "not available" in result.error


async def test_tool_timeout_is_enforced():
    @tool(timeout=0.05)
    async def slow() -> str:
        await asyncio.sleep(1)
        return "never"

    registry = ToolRegistry([slow])
    context, executor = make_context(registry)
    result = await executor.execute(context, ToolCall("c1", "slow", {}))
    assert not result.ok and "timed out" in result.error


async def test_transient_failures_are_retried():
    attempts = {"n": 0}

    @tool(retries=2)
    async def flaky() -> str:
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise ConnectionError("boom")
        return "ok"

    registry = ToolRegistry([flaky])
    context, executor = make_context(registry)
    context.policy.retry.base_delay = 0.001
    result = await executor.execute(context, ToolCall("c1", "flaky", {}))
    assert result.ok and attempts["n"] == 3


async def test_non_idempotent_side_effects_are_never_retried():
    attempts = {"n": 0}

    @tool(retries=3, side_effect=True, idempotent=False)
    async def charge_card() -> str:
        attempts["n"] += 1
        raise ConnectionError("network blip")

    registry = ToolRegistry([charge_card])
    context, executor = make_context(registry)
    result = await executor.execute(context, ToolCall("c1", "charge_card", {}))
    assert not result.ok and attempts["n"] == 1


async def test_idempotent_side_effect_is_deduplicated_on_replay():
    calls = {"n": 0}

    @tool(side_effect=True, idempotent=True)
    async def create_payment(amount: int) -> dict:
        calls["n"] += 1
        return {"payment_id": calls["n"], "amount": amount}

    registry = ToolRegistry([create_payment])
    context, executor = make_context(registry)
    call = ToolCall("c1", "create_payment", {"amount": 10})
    first = await executor.execute(context, call)
    second = await executor.execute(context, call)
    assert calls["n"] == 1
    assert first.output == second.output


async def test_rate_limit():
    @tool(rate_limit_per_minute=1)
    async def ping() -> str:
        return "pong"

    registry = ToolRegistry([ping])
    context, executor = make_context(registry)
    assert (await executor.execute(context, ToolCall("c1", "ping", {}))).ok
    second = await executor.execute(context, ToolCall("c2", "ping", {}))
    assert not second.ok and "rate limit" in second.error


async def test_tool_budget_is_enforced_by_the_runtime():
    from coflowai import BudgetExceeded

    registry = ToolRegistry([add])
    context, executor = make_context(registry, max_tool_calls=1)
    await executor.execute(context, ToolCall("c1", "add", {"a": 1}))
    with pytest.raises(BudgetExceeded):
        await executor.execute(context, ToolCall("c2", "add", {"a": 1}))


def test_registry_only_exposes_authorised_tools():
    registry = ToolRegistry([add, FakeTool("public")])
    visible = registry.visible_to(PermissionSet({"weather.read"}))
    assert [t.name for t in visible] == ["public"]


def test_registry_rejects_undecorated_functions():
    with pytest.raises(TypeError):
        ToolRegistry().register(lambda: None)  # type: ignore[arg-type]
