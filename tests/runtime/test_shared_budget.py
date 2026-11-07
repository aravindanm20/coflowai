"""Parallel branches share one execution-wide budget ledger."""
from __future__ import annotations

import asyncio

from coflowai import Agent, ExecutionPolicy, ExecutionStatus, Workflow
from coflowai.core.context import ExecutionContext
from coflowai.testing import build_test_app
from coflowai.testing.fakes import FakeModelProvider


async def test_parallel_branches_cannot_overspend_model_calls():
    provider = FakeModelProvider(["a", "b", "c"])
    app, _ = build_test_app(provider=provider,
                            policy=ExecutionPolicy(max_model_calls=2,
                                                   retry_attempts=0))
    workflow = Workflow("w").parallel(*(Agent(n, "i", "fake-model")
                                        for n in ("x", "y", "z")), fail_fast=False)
    result = await app.run(workflow, "go")
    assert result.status is ExecutionStatus.FAILED
    assert type(result.error).__name__ == "BudgetExceeded"
    assert len(provider.calls) == 2           # was 3 before the shared ledger


async def test_reservation_is_atomic_across_concurrent_siblings():
    root = ExecutionContext(policy=ExecutionPolicy(max_tool_calls=1))
    a, b = root.child(), root.child()
    outcomes = await asyncio.gather(a.check_tool_budget(), b.check_tool_budget(),
                                    return_exceptions=True)
    assert sum(isinstance(o, Exception) for o in outcomes) == 1


async def test_released_reservation_frees_the_slot():
    root = ExecutionContext(policy=ExecutionPolicy(max_model_calls=1))
    reservation = await root.child().check_model_budget()
    reservation.release()
    await root.child().check_model_budget()    # slot available again


async def test_sibling_cost_is_visible_immediately():
    root = ExecutionContext(policy=ExecutionPolicy(max_cost=1.0))
    a, b = root.child(), root.child()
    a.record_model_usage(cost=1.5)
    try:
        await b.check_model_budget()
    except Exception as exc:
        assert "max_cost" in str(exc)
    else:
        raise AssertionError("sibling spend was not visible")
