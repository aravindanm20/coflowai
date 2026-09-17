"""Provider adapters, tested fully offline through MockTransport."""

from __future__ import annotations

import json

import pytest

from coflowai import ConfigurationError, Message, ModelError, ToolCall
from coflowai.core.exceptions import ModelUnavailable, RateLimitError
from coflowai.models.base import ModelRequest
from coflowai.providers.anthropic import AnthropicProvider
from coflowai.providers.gemini import GeminiProvider
from coflowai.providers.http import MockTransport, translate_status
from coflowai.providers.openai import (
    AzureOpenAIProvider,
    OpenAIProvider,
    _harden,
)


def request(**kwargs) -> ModelRequest:
    kwargs.setdefault("messages", [Message.system("rules"), Message.user("hello")])
    return ModelRequest(**kwargs)


# --------------------------------------------------------------------- OpenAI
def openai_body(content="hi", tool_calls=None):
    return {
        "model": "gpt-4o",
        "choices": [{"message": {"content": content, "tool_calls": tool_calls},
                     "finish_reason": "tool_calls" if tool_calls else "stop"}],
        "usage": {"prompt_tokens": 11, "completion_tokens": 5,
                  "prompt_tokens_details": {"cached_tokens": 4}},
    }


async def test_openai_translates_request_and_response():
    transport = MockTransport([openai_body("Hello there")])
    provider = OpenAIProvider(model="gpt-4o", transport=transport)
    response = await provider.generate(request())

    sent = transport.requests[0]
    assert sent["model"] == "gpt-4o"
    assert sent["messages"][0] == {"role": "system", "content": "rules"}
    assert response.text == "Hello there"
    assert response.input_tokens == 11 and response.output_tokens == 5
    assert response.cached_tokens == 4


async def test_openai_translates_tool_calls_and_results():
    body = openai_body(None, [{"id": "call_1", "type": "function",
                               "function": {"name": "get_weather",
                                            "arguments": '{"city": "Chennai"}'}}])
    transport = MockTransport([body])
    provider = OpenAIProvider(model="gpt-4o", transport=transport)
    response = await provider.generate(request(
        tools=[{"name": "get_weather", "description": "weather",
                "parameters": {"type": "object", "properties": {}}}]))

    assert transport.requests[0]["tools"][0]["function"]["name"] == "get_weather"
    assert response.tool_calls[0].arguments == {"city": "Chennai"}

    # a tool result round-trips back into the wire format
    transport2 = MockTransport([openai_body("done")])
    provider2 = OpenAIProvider(model="gpt-4o", transport=transport2)
    await provider2.generate(request(messages=[
        Message.user("q"),
        Message.assistant(None, [ToolCall("call_1", "get_weather", {"city": "X"})]),
        Message.tool("call_1", "get_weather", '{"temperature": 30}'),
    ]))
    wire = transport2.requests[0]["messages"]
    assert wire[-1] == {"role": "tool", "tool_call_id": "call_1",
                        "content": '{"temperature": 30}'}


async def test_openai_malformed_tool_arguments_do_not_crash():
    """Model output is untrusted — bad JSON must not raise."""
    body = openai_body(None, [{"id": "c1", "type": "function",
                               "function": {"name": "t", "arguments": "{not json"}}])
    provider = OpenAIProvider(model="gpt-4o", transport=MockTransport([body]))
    response = await provider.generate(request())
    assert response.tool_calls[0].arguments == {"__raw__": "{not json"}


async def test_openai_structured_output_uses_strict_json_schema():
    transport = MockTransport([openai_body('{"a": 1}')])
    provider = OpenAIProvider(model="gpt-4o", transport=transport)
    await provider.generate(request(response_schema={
        "type": "object", "properties": {"a": {"type": "integer"}}}))
    fmt = transport.requests[0]["response_format"]
    assert fmt["type"] == "json_schema"
    assert fmt["json_schema"]["schema"]["additionalProperties"] is False


async def test_openai_streaming_accumulates_text_and_tools():
    lines = [
        'data: {"model":"gpt-4o","choices":[{"delta":{"content":"Hel"}}]}',
        'data: {"choices":[{"delta":{"content":"lo"}}]}',
        'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c1",'
        '"function":{"name":"ping","arguments":"{\\"x\\": 1}"}}]}}]}',
        'data: {"choices":[{"finish_reason":"tool_calls"}],'
        '"usage":{"prompt_tokens":3,"completion_tokens":2}}',
        "data: [DONE]",
    ]
    provider = OpenAIProvider(model="gpt-4o",
                              transport=MockTransport(stream_lines=[lines]))
    text, final = "", None
    async for chunk in provider.stream(request()):
        if chunk.is_final:
            final = chunk.response
        else:
            text += chunk.text
    assert text == "Hello"
    assert final.tool_calls[0].name == "ping"
    assert final.tool_calls[0].arguments == {"x": 1}
    assert final.input_tokens == 3


@pytest.mark.parametrize(("status", "expected"), [
    (429, RateLimitError), (503, ModelUnavailable), (401, ConfigurationError),
    (400, ModelError),
])
def test_http_status_translation(status, expected):
    assert isinstance(translate_status(status, "body", provider="p"), expected)


def test_openai_requires_credentials():
    with pytest.raises(ConfigurationError):
        OpenAIProvider(model="gpt-4o", api_key=None, base_url="http://x")


def test_harden_is_recursive():
    hardened = _harden({"type": "object", "properties": {
        "nested": {"type": "object", "properties": {"a": {"type": "string"}}}}})
    assert hardened["properties"]["nested"]["additionalProperties"] is False


async def test_azure_uses_deployment_url_and_api_key_header():
    transport = MockTransport([openai_body("azure")])
    provider = AzureOpenAIProvider(endpoint="https://x.openai.azure.com",
                                   deployment="gpt4o", api_key="secret",
                                   transport=transport)
    await provider.generate(request())
    assert "deployments/gpt4o/chat/completions" in transport.urls[0]
    assert transport.headers[0]["api-key"] == "secret"
    assert "Authorization" not in transport.headers[0]


# ------------------------------------------------------------------ Anthropic
async def test_anthropic_hoists_system_and_uses_content_blocks():
    body = {"model": "claude-sonnet-4", "stop_reason": "end_turn",
            "content": [{"type": "text", "text": "Hi"}],
            "usage": {"input_tokens": 7, "output_tokens": 3,
                      "cache_read_input_tokens": 2}}
    transport = MockTransport([body])
    provider = AnthropicProvider(model="claude-sonnet-4", transport=transport,
                                 api_key="k")
    response = await provider.generate(request())

    sent = transport.requests[0]
    assert sent["system"] == "rules"
    assert sent["messages"][0]["content"][0]["text"] == "hello"
    assert response.text == "Hi" and response.cached_tokens == 2


async def test_anthropic_tool_results_become_user_blocks():
    body = {"content": [{"type": "text", "text": "ok"}], "usage": {}}
    transport = MockTransport([body])
    provider = AnthropicProvider(model="c", transport=transport, api_key="k")
    await provider.generate(request(messages=[
        Message.user("q"),
        Message.assistant("thinking", [ToolCall("tu_1", "weather", {"city": "X"})]),
        Message.tool("tu_1", "weather", "30C"),
    ]))
    messages = transport.requests[0]["messages"]
    assert messages[1]["content"][1]["type"] == "tool_use"
    assert messages[2]["role"] == "user"
    assert messages[2]["content"][0]["type"] == "tool_result"


async def test_anthropic_structured_output_is_a_forced_tool_call():
    """Anthropic has no json_schema mode; the adapter emulates it."""
    body = {"content": [{"type": "tool_use", "id": "t1",
                         "name": "emit_structured_output",
                         "input": {"summary": "s", "confidence": 0.9}}],
            "usage": {"input_tokens": 1, "output_tokens": 1}}
    transport = MockTransport([body])
    provider = AnthropicProvider(model="c", transport=transport, api_key="k")
    response = await provider.generate(request(response_schema={
        "type": "object", "properties": {"summary": {"type": "string"}}}))

    assert transport.requests[0]["tool_choice"]["name"] == "emit_structured_output"
    assert json.loads(response.text) == {"summary": "s", "confidence": 0.9}
    assert response.tool_calls == []      # the wrapper call is unwrapped, not leaked


async def test_anthropic_streaming():
    lines = [
        'data: {"type":"message_start","message":{"usage":{"input_tokens":5}}}',
        'data: {"type":"content_block_delta","delta":{"type":"text_delta",'
        '"text":"Hi "}}',
        'data: {"type":"content_block_delta","delta":{"type":"text_delta",'
        '"text":"there"}}',
        'data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},'
        '"usage":{"output_tokens":4}}',
    ]
    provider = AnthropicProvider(model="c", api_key="k",
                                 transport=MockTransport(stream_lines=[lines]))
    text, final = "", None
    async for chunk in provider.stream(request()):
        if chunk.is_final:
            final = chunk.response
        else:
            text += chunk.text
    assert text == "Hi there"
    assert final.input_tokens == 5 and final.output_tokens == 4


# --------------------------------------------------------------------- Gemini
async def test_gemini_translates_roles_and_schema():
    body = {"modelVersion": "gemini-2.0-flash",
            "candidates": [{"finishReason": "STOP", "content": {"parts": [
                {"text": "Hello"}]}}],
            "usageMetadata": {"promptTokenCount": 9, "candidatesTokenCount": 2}}
    transport = MockTransport([body])
    provider = GeminiProvider(model="gemini-2.0-flash", transport=transport,
                              api_key="k")
    response = await provider.generate(request(response_schema={
        "type": "object", "additionalProperties": False, "title": "X",
        "properties": {"a": {"type": "string"}}}))

    sent = transport.requests[0]
    assert sent["system_instruction"]["parts"][0]["text"] == "rules"
    assert sent["contents"][0]["role"] == "user"
    # Gemini rejects these JSON Schema keywords
    schema = sent["generationConfig"]["responseSchema"]
    assert "additionalProperties" not in schema and "title" not in schema
    assert response.text == "Hello" and response.input_tokens == 9


async def test_gemini_assistant_role_is_model():
    body = {"candidates": [{"content": {"parts": [{"text": "ok"}]}}],
            "usageMetadata": {}}
    transport = MockTransport([body])
    provider = GeminiProvider(model="g", transport=transport, api_key="k")
    await provider.generate(request(messages=[Message.assistant("prior"),
                                              Message.user("next")]))
    assert transport.requests[0]["contents"][0]["role"] == "model"


async def test_gemini_blocked_prompt_raises_model_error():
    transport = MockTransport([{"promptFeedback": {"blockReason": "SAFETY"}}])
    provider = GeminiProvider(model="g", transport=transport, api_key="k")
    with pytest.raises(ModelError):
        await provider.generate(request())
