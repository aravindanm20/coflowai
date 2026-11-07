"""Regression: framework ledgers written in child scopes must reach checkpoints."""
from __future__ import annotations

from coflowai import (Agent, ExecutionPolicy, ExecutionStatus, HumanApproval,
                      Workflow, tool)
from coflowai.testing import build_test_app, tool_call
from coflowai.testing.fakes import FakeModelProvider


async def test_approval_inside_a_parallel_branch_can_be_approved():
    app, _ = build_test_app(["fast branch done"])
    workflow = Workflow("w").parallel(
        Agent("fast", "i", "fake-model"),
        HumanApproval("Approve the risky branch?", name="risky"),
        fail_fast=True)
    paused = await app.run(workflow, "go")
    assert paused.status is ExecutionStatus.WAITING_FOR_APPROVAL
    pending = await app.runtime.pending_approvals(paused.execution_id)
    assert [p.approval_id for p in pending] == ["parallel.risky"]   # node-scoped


async def test_side_effect_is_not_repeated_when_a_later_node_fails():
    charges = []

    @tool(side_effect=True, idempotent=True)
    async def charge(amount: int) -> str:
        charges.append(amount)
        return "charged"

    provider = FakeModelProvider([tool_call("charge", amount=5), "paid"])
    app, _ = build_test_app(provider=provider, tools=[charge],
                            policy=ExecutionPolicy(retry_attempts=0, max_steps=8))
    workflow = (Workflow("w").start(Agent("payer", "i", "fake-model", [charge]))
                .then(Agent("notifier", "i", "fake-model")))
    failed = await app.run(workflow, "go")         # notifier: script exhausted
    assert failed.status is ExecutionStatus.FAILED and charges == [5]
    checkpoint = await app.runtime.state_store.latest_checkpoint(failed.execution_id)
    assert checkpoint.state.get("__idempotency__"), "ledger lost on failure"
    provider.responses.append("notified")
    resumed = await app.resume(failed.execution_id, executable=workflow)
    assert resumed.succeeded and charges == [5]   # charged exactly once


async def test_completed_parallel_branch_is_not_rerun_after_approval():
    app, provider = build_test_app(["fast branch done"])
    workflow = Workflow("w").parallel(
        Agent("fast", "i", "fake-model"),
        HumanApproval("Approve?", name="risky"),
        fail_fast=False)
    paused = await app.run(workflow, "go")
    assert paused.status is ExecutionStatus.WAITING_FOR_APPROVAL
    resumed = await app.approve(paused.execution_id, approval_id="parallel.risky",
                                executable=workflow)
    assert resumed.succeeded, resumed.error
    assert resumed.output["fast"] == "fast branch done"
    assert len(provider.calls) == 1          # 'fast' ran exactly once


async def test_parallel_inside_a_loop_reruns_every_iteration():
    from coflowai.workflows.node import LoopNode, ParallelNode
    seen = []

    def branch(x):
        seen.append(x)
        return x

    counter = {"n": 0}

    def until(_out):
        counter["n"] += 1
        return counter["n"] >= 2

    body = ParallelNode(branch, name="p", merge=lambda o: next(iter(o.values())))
    app, _ = build_test_app([])
    result = await app.run(Workflow("w").start(LoopNode(body, until=until,
                                                        max_iterations=3)), "x")
    assert result.succeeded, result.error
    assert len(seen) == 2
