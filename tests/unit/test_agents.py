"""Agent loop: tool use, structured output, repair, limits, tool exposure."""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from coflowai import (
    Agent,
    ExecutionPolicy,
    ExecutionStatus,
    MaxStepsExceeded,
    ValidationError,
    tool,
)
from coflowai.testing import build_test_app, tool_call


class Analysis(BaseModel):
    summary: str
    risk: str
    confidence: float


@tool(description="Look up the weather", permission="weather.read")
async def get_weather(city: str) -> dict:
    return {"city": city, "temperature": 30}


async def test_simple_agent_returns_text():
    app, model = build_test_app(["Distributed systems are..."])
    agent = Agent("assistant", "You are helpful.", "fake-model")
    result = await app.run(agent, "Explain distributed systems.")
    assert result.succeeded
    assert result.output == "Distributed systems are..."
    assert model.call_count == 1
    assert result.usage.model_calls == 1 and result.usage.cost > 0


async def test_agent_uses_tools_then_answers():
    app, model = build_test_app(
        [tool_call("get_weather", city="Chennai"), "It is 30 degrees."],
        tools=[get_weather],
    )
    agent = Agent("assistant", "Answer questions.", "fake-model", [get_weather])
    result = await app.run(agent, "Weather in Chennai?")
    assert result.output == "It is 30 degrees."
    assert result.usage.tool_calls == 1
    events = [e.type for e in
              await app.runtime.event_store.get_events(result.execution_id)]
    assert events == [
        "execution.created", "execution.started", "agent.started",
        "model.requested", "model.completed",
        "tool.requested", "tool.started", "tool.completed",
        "model.requested", "model.completed",
        "agent.completed", "checkpoint.created", "execution.completed",
    ]


async def test_parallel_tool_calls_are_bounded_and_all_executed():
    calls = [tool_call("get_weather", city="Chennai"),
             tool_call("get_weather", city="Bengaluru")]
    app, _ = build_test_app([calls, "Both cities are warm."], tools=[get_weather])
    agent = Agent("assistant", "Answer questions.", "fake-model", [get_weather])
    result = await app.run(agent, "Compare the weather.")
    assert result.usage.tool_calls == 2
    assert result.succeeded


async def test_structured_output_is_validated():
    payload = {"summary": "ok", "risk": "low", "confidence": 0.9}
    app, _ = build_test_app([payload])
    agent = Agent("analyst", "Analyse.", "fake-model", output_schema=Analysis)
    result = await app.run(agent, "analyse this")
    assert isinstance(result.output, Analysis)
    assert result.output.confidence == 0.9


async def test_structured_output_is_repaired_then_accepted():
    app, model = build_test_app([
        "not json at all",
        {"summary": "ok", "risk": "low", "confidence": 0.5},
    ])
    agent = Agent("analyst", "Analyse.", "fake-model", output_schema=Analysis)
    result = await app.run(agent, "analyse")
    assert result.succeeded and model.call_count == 2


async def test_invalid_structured_output_never_returned_silently():
    app, _ = build_test_app(["nope", "still nope", "nope again"],
                            policy=ExecutionPolicy(
                                structured_output_repair_attempts=1, max_steps=5))
    agent = Agent("analyst", "Analyse.", "fake-model", output_schema=Analysis)
    result = await app.run(agent, "analyse")
    assert result.status is ExecutionStatus.FAILED
    assert isinstance(result.error, ValidationError)


async def test_agent_only_sees_permitted_tools():
    app, model = build_test_app(["done"], tools=[get_weather],
                                permissions=["other.permission"])
    agent = Agent("assistant", "Answer.", "fake-model", [get_weather])
    await app.run(agent, "hi")
    assert model.calls[0].tools == []


async def test_denied_tool_call_is_reported_back_to_the_model():
    app, _ = build_test_app([tool_call("get_weather", city="X"), "I could not."],
                            tools=[get_weather], permissions=["other.permission"])
    agent = Agent("assistant", "Answer.", "fake-model", [get_weather])
    result = await app.run(agent, "weather?")
    assert result.succeeded
    events = [e.type for e in
              await app.runtime.event_store.get_events(result.execution_id)]
    assert "tool.denied" in events


async def test_agent_loop_is_bounded():
    app, _ = build_test_app([tool_call("get_weather", city="X")] * 10,
                            tools=[get_weather],
                            policy=ExecutionPolicy(max_steps=3, max_tool_calls=10))
    agent = Agent("assistant", "Answer.", "fake-model", [get_weather])
    result = await app.run(agent, "loop forever")
    assert result.status is ExecutionStatus.FAILED
    assert isinstance(result.error, MaxStepsExceeded)


async def test_agent_declares_required_permissions_and_capabilities():
    agent = Agent("assistant", "Answer.", "fake-model", [get_weather],
                  output_schema=Analysis)
    assert agent.required_permissions == ["weather.read"]
    assert agent.model_requirements == {"tool_calling": True,
                                        "structured_output": True}


async def test_agent_requires_a_runtime():
    from coflowai import ConfigurationError, ExecutionContext

    agent = Agent("assistant", "Answer.", "fake-model")
    with pytest.raises(ConfigurationError):
        await agent.execute(ExecutionContext(input="hi"))


async def test_memory_is_injected_into_context_when_configured():
    app, model = build_test_app(["ok"], memory=True)
    await app.memory.put("pref", "The user prefers concise answers.")
    agent = Agent("assistant", "Answer.", "fake-model")
    await app.run(agent, "user answers preference")
    system_prompts = [m.content for m in model.calls[0].messages if m.content]
    assert any("prefers concise" in text for text in system_prompts)
