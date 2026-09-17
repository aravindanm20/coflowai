"""Streaming, determinism modes, fork overrides, summarisation and sandboxing."""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from coflowai import (
    Agent,
    ExecutionMode,
    ExecutionPolicy,
    ExecutionStatus,
    Message,
    Workflow,
    tool,
)
from coflowai.models.streaming import ChunkType, StreamAccumulator, StreamChunk
from coflowai.testing import build_test_app, tool_call


class Report(BaseModel):
    summary: str


@tool(description="Ping")
async def ping(value: str = "x") -> str:
    return f"pong:{value}"


# ------------------------------------------------------------------ streaming
async def test_streaming_yields_text_then_a_final_result():
    app, _ = build_test_app(["Distributed systems are hard"])
    agent = Agent("assistant", "Answer.", "fake-model")

    chunks, result = [], None
    async for chunk in app.stream(agent, "explain"):
        if chunk.is_final:
            result = chunk.metadata["result"]
        else:
            chunks.append(chunk.text)

    assert len(chunks) > 1                      # genuinely incremental
    assert "".join(chunks) == "Distributed systems are hard"
    assert result.output == "Distributed systems are hard"


async def test_streaming_still_records_events_and_usage():
    app, _ = build_test_app(["hello world"])
    agent = Agent("assistant", "Answer.", "fake-model")

    execution_id = None
    async for chunk in app.stream(agent, "hi"):
        if chunk.is_final:
            execution_id = chunk.metadata["result"].execution_id

    events = [e.type for e in
              await app.runtime.event_store.get_events(execution_id)]
    assert "model.completed" in events and "execution.completed" in events
    usage = app.usage(execution_id)
    assert usage.total.model_calls == 1
    assert usage.total.input_tokens > 0         # accounted exactly once


async def test_streaming_executes_tools_between_iterations():
    app, _ = build_test_app([tool_call("ping", value="a"), "all done"],
                            tools=[ping])
    agent = Agent("assistant", "Answer.", "fake-model", [ping])

    text, result = "", None
    async for chunk in app.stream(agent, "go"):
        if chunk.is_final:
            result = chunk.metadata["result"]
        elif chunk.type is ChunkType.TEXT:
            text += chunk.text

    assert text == "all done"
    assert result.usage.tool_calls == 1


async def test_streaming_respects_budgets():
    from coflowai import BudgetExceeded

    app, _ = build_test_app(["a", "b"], policy=ExecutionPolicy(max_model_calls=0))
    agent = Agent("assistant", "Answer.", "fake-model")
    with pytest.raises(BudgetExceeded):
        async for _chunk in app.stream(agent, "go"):
            pass


async def test_structured_output_is_not_streamed_partially():
    """Half a JSON object is worse than no JSON: emit it only once validated."""
    app, _ = build_test_app([{"summary": "done"}])
    agent = Agent("analyst", "Analyse.", "fake-model", output_schema=Report)

    text_chunks, result = [], None
    async for chunk in app.stream(agent, "go"):
        if chunk.is_final:
            result = chunk.metadata["result"]
        else:
            text_chunks.append(chunk)

    assert text_chunks == []
    assert isinstance(result.output, Report) and result.output.summary == "done"


async def test_non_streaming_provider_is_transparently_emulated():
    from coflowai.models.base import ModelProvider, ModelResponse

    class PlainProvider(ModelProvider):
        name = "plain"

        async def generate(self, request):
            return ModelResponse(text="no native streaming", input_tokens=1,
                                 output_tokens=2, cost=0.0)

    app, _ = build_test_app([])
    app.models.register("plain", PlainProvider(), default=True)
    agent = Agent("assistant", "Answer.", "plain")

    received = [c async for c in app.stream(agent, "go")]
    assert any(c.type is ChunkType.TEXT for c in received)
    assert received[-1].is_final


def test_accumulator_merges_streamed_tool_arguments():
    from coflowai import ToolCall

    accumulator = StreamAccumulator()
    accumulator.add(StreamChunk(type=ChunkType.TEXT, text="par"))
    accumulator.add(StreamChunk(type=ChunkType.TEXT, text="tial"))
    accumulator.add(StreamChunk(type=ChunkType.TOOL_CALL, index=0,
                                tool_call=ToolCall("c1", "t", {"a": 1})))
    accumulator.add(StreamChunk(type=ChunkType.TOOL_CALL, index=0,
                                tool_call=ToolCall("c1", "t", {"b": 2})))
    response = accumulator.build(input_tokens=5)
    assert response.text == "partial"
    assert response.tool_calls[0].arguments == {"a": 1, "b": 2}


# ------------------------------------------------------------ execution modes
async def test_strict_mode_pins_temperature_to_zero():
    app, model = build_test_app(["ok"],
                                policy=ExecutionPolicy(mode=ExecutionMode.STRICT))
    await app.run(Agent("a", "i", "fake-model", temperature=0.9), "go")
    assert model.calls[0].temperature == 0.0


async def test_standard_mode_keeps_the_agents_temperature():
    app, model = build_test_app(["ok"])
    await app.run(Agent("a", "i", "fake-model", temperature=0.9), "go")
    assert model.calls[0].temperature == 0.9


async def test_strict_mode_serialises_tool_calls_for_reproducible_order():
    from coflowai import FakeTool

    recorder = FakeTool("record", result=lambda **kw: kw.get("value"))
    calls = [tool_call("record", value="a"), tool_call("record", value="b")]
    app, _ = build_test_app([calls, "done"], tools=[recorder],
                            policy=ExecutionPolicy(mode=ExecutionMode.STRICT,
                                                   max_steps=5))
    agent = Agent("a", "i", "fake-model", [recorder])
    result = await app.run(agent, "go")

    assert result.usage.tool_calls == 2
    # STRICT runs them one at a time, so ordering matches the model's request
    assert [i["value"] for i in recorder.invocations] == ["a", "b"]


def test_mode_iteration_allowances_differ():
    strict = ExecutionPolicy(max_steps=20, mode=ExecutionMode.STRICT)
    standard = ExecutionPolicy(max_steps=20)
    exploratory = ExecutionPolicy(max_steps=20, mode=ExecutionMode.EXPLORATORY)

    # an agent asking for 10 iterations
    assert strict.iteration_allowance(10) == 5        # clamped tight
    assert standard.iteration_allowance(10) == 10     # honoured
    assert exploratory.iteration_allowance(10) == 20  # given room to explore
    # never unbounded, in any mode
    assert exploratory.iteration_allowance(None) == 20


async def test_exploratory_mode_grants_more_reasoning_room():
    app, _ = build_test_app([tool_call("ping")] * 8 + ["finished"], tools=[ping],
                            policy=ExecutionPolicy(
                                max_steps=10, max_tool_calls=20,
                                mode=ExecutionMode.EXPLORATORY))
    # the agent asks for only 2 iterations; EXPLORATORY lets it go further
    agent = Agent("a", "i", "fake-model", [ping], max_iterations=2)
    result = await app.run(agent, "explore")
    assert result.succeeded
    assert result.usage.tool_calls == 8


# ------------------------------------------------------------ fork overrides
async def test_fork_can_replace_a_model_response():
    app, _ = build_test_app(["original answer"], policy=ExecutionPolicy(max_steps=4))
    agent = Agent("assistant", "Answer.", "fake-model")
    original = await app.run(agent, "question")
    assert original.output == "original answer"

    forked = await app.fork(original.execution_id, from_step=0,
                            overrides={"model_responses": {1: "edited answer"}})
    assert forked.output == "edited answer"
    assert forked.usage.cost == 0.0        # no provider was billed


async def test_fork_can_replace_a_tool_result():
    """Re-run an execution asking: what if the tool had returned X instead?"""
    from coflowai import FakeTool

    real_tool = FakeTool("lookup", result={"temperature": 30})
    app, model = build_test_app(
        [tool_call("lookup"), "It is 30 degrees."],
        tools=[real_tool], policy=ExecutionPolicy(max_steps=6))
    agent = Agent("assistant", "Answer.", "fake-model", [real_tool])
    original = await app.run(agent, "weather?")
    assert real_tool.call_count == 1

    model.responses = [tool_call("lookup"), "It is -40 degrees."]
    model.reset()
    model.responses = [tool_call("lookup"), "It is -40 degrees."]
    forked = await app.fork(
        original.execution_id, from_step=0,
        overrides={"tool_results": {"lookup": {"temperature": -40}}})

    assert forked.succeeded
    # the real tool was never invoked again — the override was substituted
    assert real_tool.call_count == 1
    events = await app.runtime.event_store.get_events(forked.execution_id)
    completed = [e for e in events if e.type == "tool.completed"]
    assert completed[0].payload["output"] == {"temperature": -40}
    assert completed[0].payload["replayed"] is True


async def test_fork_can_replace_instructions():
    app, model = build_test_app(["a", "b"], policy=ExecutionPolicy(max_steps=4))
    agent = Agent("assistant", "Be terse.", "fake-model")
    original = await app.run(agent, "question")

    await app.fork(original.execution_id, from_step=0,
                   overrides={"instructions": "Be extremely verbose."})
    prompts = [m.content for m in model.calls[-1].messages if m.content]
    assert any("extremely verbose" in p for p in prompts)


# ------------------------------------------------------------- summarisation
def test_default_summariser_keeps_decisions_over_pleasantries():
    from coflowai.context import ExtractiveSummariser

    messages = [
        Message.user("Hi there, hope you're well."),
        Message.assistant("We decided to use Postgres because it supports "
                          "transactional checkpoints."),
        Message.tool("c1", "deploy", "Deployment failed: disk full."),
        Message.user("Thanks!"),
    ]
    summary = ExtractiveSummariser()(messages, token_budget=100)
    assert "decided" in summary.lower()
    assert "failed" in summary.lower()
    assert "hope you're well" not in summary


def test_summariser_is_deterministic():
    from coflowai.context import ExtractiveSummariser

    messages = [Message.user(f"Decision {i}: we must ship.") for i in range(10)]
    first = ExtractiveSummariser()(messages, 60)
    second = ExtractiveSummariser()(messages, 60)
    assert first == second


async def test_builder_uses_the_default_summariser_without_configuration():
    from coflowai import ContextBudget, ContextBuilder

    builder = ContextBuilder(ContextBudget(max_tokens=2000, conversation=40,
                                           reserve_output=0))
    builder.conversation([Message.user("We decided to adopt CoFlowAi. " * 20),
                          Message.user("latest question")])
    messages = await builder.build()
    assert any("Summary of" in (m.content or "") for m in messages)


# ------------------------------------------------------------------ sandbox
async def test_subprocess_sandbox_runs_a_module_level_tool():
    from coflowai.testing import sandbox_fixtures
    from coflowai.tools.subprocess_sandbox import SubprocessSandbox

    sandbox = SubprocessSandbox(cpu_seconds=5, memory_mb=256)
    result = await sandbox.run(sandbox_fixtures.echo, {"value": "hello"})
    assert result == {"echoed": "hello"}


async def test_subprocess_sandbox_rejects_unsandboxable_closures():
    from coflowai import ConfigurationError
    from coflowai.tools.subprocess_sandbox import SubprocessSandbox

    @tool
    async def local_tool() -> str:          # defined inside a function
        return "nope"

    with pytest.raises(ConfigurationError):
        await SubprocessSandbox().run(local_tool, {})


async def test_subprocess_sandbox_surfaces_tool_errors():
    from coflowai import ToolError
    from coflowai.testing import sandbox_fixtures
    from coflowai.tools.subprocess_sandbox import SubprocessSandbox

    with pytest.raises(ToolError):
        await SubprocessSandbox().run(sandbox_fixtures.explode, {})


async def test_subprocess_sandbox_does_not_inherit_secrets(monkeypatch):
    from coflowai.testing import sandbox_fixtures
    from coflowai.tools.subprocess_sandbox import SubprocessSandbox

    monkeypatch.setenv("OPENAI_API_KEY", "sk-should-not-leak")
    result = await SubprocessSandbox().run(sandbox_fixtures.read_env,
                                           {"name": "OPENAI_API_KEY"})
    assert result is None
