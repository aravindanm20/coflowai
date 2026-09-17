"""Event sourcing, checkpoints, resume, replay, fork and human approval."""

from __future__ import annotations

import pytest

from coflowai import (
    Agent,
    ApprovalRejected,
    CheckpointError,
    ExecutionError,
    ExecutionPolicy,
    ExecutionStatus,
    Workflow,
)
from coflowai.testing import build_test_app
from coflowai.testing.fakes import FakeModelProvider


def agent(name: str) -> Agent:
    return Agent(name, f"You are {name}.", "fake-model")


# ------------------------------------------------------------------ checkpoints
async def test_checkpoints_are_created_around_every_node():
    app, _ = build_test_app(["a", "b"])
    workflow = Workflow("w").start(agent("first")).then(agent("second"))
    result = await app.run(workflow, "go")

    checkpoints = await app.runtime.state_store.list_checkpoints(
        result.execution_id)
    reasons = [c.reason for c in checkpoints]
    assert reasons.count("before_node") == 2
    assert reasons.count("after_node") == 2
    assert reasons[-1] == "final"
    assert all(c.workflow_hash == workflow.hash for c in checkpoints[:-1])


async def test_resume_continues_and_never_reruns_completed_nodes():
    # The second agent fails (script exhausted), then we resume with a fresh script.
    provider = FakeModelProvider(["first-done"])
    app, _ = build_test_app(provider=provider,
                            policy=ExecutionPolicy(retry_attempts=0, max_steps=6))
    workflow = Workflow("w").start(agent("first")).then(agent("second"))

    failed = await app.run(workflow, "go")
    assert failed.status is ExecutionStatus.FAILED

    provider.responses.append("second-done")
    resumed = await app.resume(failed.execution_id, executable=workflow)

    assert resumed.succeeded and resumed.output == "second-done"
    # 'first' ran exactly once across both attempts
    events = await app.runtime.event_store.get_events(failed.execution_id)
    started = [e.payload.get("node_id") for e in events
               if e.type == "workflow.node.started"]
    skipped = [e.payload.get("node_id") for e in events
               if e.type == "workflow.node.skipped"]
    assert started.count("first") == 1
    assert skipped == ["first"]


async def test_resume_refuses_incompatible_workflow_versions():
    provider = FakeModelProvider(["first-done"])
    app, _ = build_test_app(provider=provider,
                            policy=ExecutionPolicy(retry_attempts=0))
    original = Workflow("w").start(agent("first")).then(agent("second"))
    failed = await app.run(original, "go")

    changed = (Workflow("w").start(agent("first")).then(agent("different"))
               .then(agent("second")))
    with pytest.raises(CheckpointError):
        await app.resume(failed.execution_id, executable=changed)


async def test_resume_rejects_terminal_executions():
    app, _ = build_test_app(["done"])
    result = await app.run(agent("only"), "go")
    with pytest.raises(ExecutionError):
        await app.resume(result.execution_id)


# ---------------------------------------------------------------------- replay
async def test_replay_reconstructs_the_execution_story():
    from coflowai import tool

    @tool
    async def lookup(city: str) -> dict:
        return {"city": city}

    from coflowai.testing import tool_call

    app, _ = build_test_app([tool_call("lookup", city="Chennai"), "done"],
                            tools=[lookup])
    result = await app.run(Agent("assistant", "Answer.", "fake-model", [lookup]),
                           "weather?")
    trace = await app.replay(result.execution_id)

    assert trace.status == "completed"
    kinds = [step.kind for step in trace.steps]
    assert kinds == ["agent", "model", "tool", "model"]
    assert len(trace.model_calls()) == 2
    assert "step" in trace.render()


async def test_replay_of_unknown_execution_fails_loudly():
    app, _ = build_test_app(["x"])
    with pytest.raises(ExecutionError):
        await app.replay("exec_nope")


async def test_fork_reruns_from_a_checkpoint_with_a_different_model():
    from coflowai import CoFlowAi

    app = CoFlowAi(policy=ExecutionPolicy(max_steps=6))
    app.models.register("model-a", FakeModelProvider(["a1", "a2"]), default=True)
    app.models.register("model-b", FakeModelProvider(["b2"]))

    workflow = (Workflow("w")
                .start(Agent("first", "You are first.", "model-a"))
                .then(Agent("second", "You are second.", "model-a")))
    original = await app.run(workflow, "go")
    assert original.output == "a2"

    forked = await app.fork(original.execution_id, from_step=1,
                            overrides={"model": "model-b"})
    assert forked.succeeded and forked.output == "b2"
    assert forked.execution_id != original.execution_id
    assert forked.metadata["forked_from"] == original.execution_id


# -------------------------------------------------------------------- approvals
async def test_approval_pauses_persists_and_resumes():
    app, model = build_test_app(["deployed"])
    workflow = (Workflow("deploy")
                .approve("Approve production deployment?", name="gate")
                .then(agent("deployer")))

    paused = await app.run(workflow, "release 1.2.3")
    assert paused.status is ExecutionStatus.WAITING_FOR_APPROVAL
    assert paused.metadata["approval_id"] == "gate"
    assert model.call_count == 0          # nothing ran, worker released

    pending = await app.runtime.pending_approvals(paused.execution_id)
    assert [p.approval_id for p in pending] == ["gate"]

    resumed = await app.approve(paused.execution_id, by="aravindan",
                                executable=workflow)
    assert resumed.succeeded and resumed.output == "deployed"

    events = [e.type for e in
              await app.runtime.event_store.get_events(paused.execution_id)]
    assert "approval.requested" in events
    assert "approval.approved" in events
    assert "execution.resumed" in events


async def test_rejected_approval_fails_the_execution():
    app, model = build_test_app(["deployed"])
    workflow = (Workflow("deploy")
                .approve("Approve?", name="gate")
                .then(agent("deployer")))
    paused = await app.run(workflow, "release")
    rejected = await app.reject(paused.execution_id, note="not now",
                                executable=workflow)

    assert rejected.status is ExecutionStatus.FAILED
    assert isinstance(rejected.error, ApprovalRejected)
    assert model.call_count == 0


async def test_approving_a_running_execution_is_rejected():
    app, _ = build_test_app(["done"])
    result = await app.run(agent("only"), "go")
    with pytest.raises(ExecutionError):
        await app.approve(result.execution_id)


async def test_approval_guards_a_specific_action():
    app, _ = build_test_app(["planned", "deployed"])
    workflow = (Workflow("deploy")
                .start(agent("planner"))
                .approve("Ship it?", action=agent("deployer"), name="gate"))
    paused = await app.run(workflow, "release")
    assert paused.status is ExecutionStatus.WAITING_FOR_APPROVAL

    done = await app.approve(paused.execution_id, executable=workflow)
    assert done.output == "deployed"
