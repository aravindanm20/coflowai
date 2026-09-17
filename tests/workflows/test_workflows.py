"""Workflows: sequencing, parallelism, routing, loops and compile-time checks."""

from __future__ import annotations

import pytest

from coflowai import (
    Agent,
    CompilationError,
    ExecutionPolicy,
    ExecutionStatus,
    Workflow,
    WorkflowError,
    tool,
)
from coflowai.testing import build_test_app


def agent(name: str) -> Agent:
    return Agent(name, f"You are {name}.", "fake-model")


async def test_sequential_workflow_pipes_output_forward():
    app, model = build_test_app(["plan", "research", "report"])
    workflow = (Workflow("research")
                .start(agent("planner"))
                .then(agent("researcher"))
                .then(agent("writer")))
    result = await app.run(workflow, "topic")
    assert result.output == "report"
    assert model.call_count == 3
    # each agent received the previous agent's output
    assert "plan" in model.calls[1].messages[-1].content


async def test_transform_nodes_are_deterministic_and_free():
    app, model = build_test_app(["hello"])
    workflow = (Workflow("shout")
                .start(agent("greeter"))
                .transform(lambda value: value.upper(), name="upper"))
    result = await app.run(workflow, "hi")
    assert result.output == "HELLO"
    assert model.call_count == 1


async def test_parallel_branches_run_concurrently_and_merge():
    app, _ = build_test_app(["plan", "web", "db", "docs", "final"])
    workflow = (Workflow("research")
                .start(agent("planner"))
                .parallel(agent("web"), agent("db"), agent("docs"))
                .then(agent("writer")))
    result = await app.run(workflow, "topic")
    assert result.succeeded and result.output == "final"

    events = await app.runtime.event_store.get_events(result.execution_id)
    nodes = [e.payload.get("node_id") for e in events
             if e.type == "workflow.node.completed"]
    assert nodes == ["planner", "parallel", "writer"]


async def test_parallel_merge_function():
    app, _ = build_test_app(["a", "b"])
    workflow = Workflow("merge").parallel(
        agent("first"), agent("second"),
        merge=lambda outputs: "+".join(sorted(outputs.values())),
    )
    result = await app.run(workflow, "x")
    assert result.output == "a+b"


async def test_deterministic_routing():
    app, _ = build_test_app(["technical answer"])
    workflow = Workflow("support").route(
        condition=lambda text: "technical" if "error" in text else "general",
        routes={"technical": agent("tech"), "general": agent("general")},
    )
    result = await app.run(workflow, "I get an error 500")
    assert result.output == "technical answer"
    assert result.metadata["route"] == "technical"


async def test_unknown_route_is_rejected():
    app, _ = build_test_app(["x"])
    workflow = Workflow("support").route(
        condition=lambda text: "unknown-branch",
        routes={"general": agent("general")},
    )
    result = await app.run(workflow, "hello")
    assert result.status is ExecutionStatus.FAILED
    assert isinstance(result.error, WorkflowError)


async def test_loop_stops_on_predicate():
    app, model = build_test_app(["draft-1", "approved"])
    workflow = Workflow("review").loop(
        agent("reviewer"), until=lambda output: output == "approved",
        max_iterations=5,
    )
    result = await app.run(workflow, "text")
    assert result.output == "approved" and model.call_count == 2


async def test_loop_is_capped():
    app, model = build_test_app(["nope"] * 10, policy=ExecutionPolicy(max_steps=20))
    workflow = Workflow("review").loop(
        agent("reviewer"), until=lambda output: False, max_iterations=3)
    result = await app.run(workflow, "text")
    assert result.succeeded and model.call_count == 3
    assert result.metadata["iterations"] == 3


def test_loop_requires_a_positive_bound():
    with pytest.raises(WorkflowError):
        Workflow("bad").loop(agent("x"), max_iterations=0)


async def test_nested_subworkflow():
    app, _ = build_test_app(["inner", "outer"])
    inner = Workflow("inner").start(agent("inner-agent"))
    workflow = Workflow("outer").subworkflow(inner).then(agent("outer-agent"))
    result = await app.run(workflow, "go")
    assert result.output == "outer"


# ------------------------------------------------------------------ compiler
def test_compiler_rejects_empty_workflow():
    with pytest.raises(CompilationError):
        Workflow("empty").compile()


def test_compiler_rejects_unknown_model():
    app, _ = build_test_app(["x"])
    workflow = Workflow("w").start(Agent("a", "i", "missing-model"))
    with pytest.raises(CompilationError):
        workflow.compile(model_registry=app.runtime.models, force=True)


def test_compiler_rejects_missing_capabilities():
    from coflowai import CoFlowAi, ModelCapabilities
    from coflowai.testing import FakeModelProvider

    app = CoFlowAi()
    app.models.register("weak-model", FakeModelProvider(["x"]),
                        capabilities=ModelCapabilities(tool_calling=False))

    @tool
    async def noop() -> str:
        return "ok"

    workflow = Workflow("w").start(Agent("a", "i", "weak-model", [noop]))
    with pytest.raises(CompilationError) as info:
        workflow.compile(model_registry=app.runtime.models, force=True)
    assert "tool_calling" in str(info.value)


def test_compiler_rejects_missing_permissions():
    @tool(permission="database.write")
    async def write_row(value: str) -> str:
        return value

    workflow = Workflow("w", permissions=["database.read"]).start(
        Agent("a", "i", "fake-model", [write_row]))
    with pytest.raises(CompilationError) as info:
        workflow.compile()
    assert "permissions" in str(info.value)


def test_workflow_hash_changes_with_definition():
    first = Workflow("w").start(agent("a")).hash
    second = Workflow("w").start(agent("a")).then(agent("b")).hash
    assert first != second


def test_workflow_renders_mermaid():
    workflow = Workflow("w").start(agent("a")).then(agent("b"))
    mermaid = workflow.to_mermaid()
    assert mermaid.startswith("graph TD")
    assert "a --> b" in mermaid
