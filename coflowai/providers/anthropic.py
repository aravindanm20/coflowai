"""Anthropic Messages API adapter.

Notable translation work: Anthropic keeps the system prompt out of ``messages``,
uses content blocks rather than a flat string, and returns tool calls as
``tool_use`` blocks with tool results sent back as ``tool_result`` blocks in a
*user* message.
"""

from __future__ import annotations

import json
import os
from typing import Any, AsyncIterator

from ..core.exceptions import ConfigurationError, ModelError
from ..core.ids import tool_call_id
from ..core.types import Message, Role, ToolCall
from ..models.base import (
    ModelCapabilities,
    ModelPricing,
    ModelProvider,
    ModelRequest,
    ModelResponse,
)
from ..models.streaming import ChunkType, StreamChunk
from .http import HttpTransport, HttpxTransport, sse_lines

__all__ = ["AnthropicProvider", "ANTHROPIC_CAPABILITIES", "ANTHROPIC_PRICING"]

ANTHROPIC_CAPABILITIES = ModelCapabilities(
    tool_calling=True, structured_output=True, json_schema=True, vision=True,
    streaming=True, parallel_tool_calls=True, reasoning=True,
    max_context_tokens=200_000,
)

#: USD per 1M tokens — verify against current provider pricing before relying on it.
ANTHROPIC_PRICING: dict[str, ModelPricing] = {
    "claude-sonnet-4": ModelPricing(3.0, 15.0, 0.3),
    "claude-opus-4": ModelPricing(15.0, 75.0, 1.5),
    "claude-haiku-3.5": ModelPricing(0.8, 4.0, 0.08),
}

_STRUCTURED_TOOL = "emit_structured_output"


class AnthropicProvider(ModelProvider):
    name = "anthropic"

    def __init__(self, *, model: str, api_key: str | None = None,
                 base_url: str = "https://api.anthropic.com/v1",
                 version: str = "2023-06-01",
                 max_output_tokens: int = 4096,
                 transport: HttpTransport | None = None,
                 extra_headers: dict[str, str] | None = None,
                 extra_body: dict[str, Any] | None = None) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.version = version
        self.default_max_tokens = max_output_tokens
        self.extra_headers = dict(extra_headers or {})
        self.extra_body = dict(extra_body or {})
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if self._api_key is None and transport is None:
            raise ConfigurationError(
                "Anthropic API key missing: pass api_key= or set ANTHROPIC_API_KEY")
        self.transport = transport or HttpxTransport(provider=self.name)

    @property
    def _url(self) -> str:
        return f"{self.base_url}/messages"

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json",
                   "anthropic-version": self.version, **self.extra_headers}
        if self._api_key:
            headers["x-api-key"] = self._api_key
        return headers

    def _payload(self, request: ModelRequest, *, stream: bool) -> dict[str, Any]:
        system, messages = _split_system(request.messages)
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": request.max_output_tokens or self.default_max_tokens,
            **self.extra_body,
        }
        if system:
            payload["system"] = system
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.stop:
            payload["stop_sequences"] = request.stop

        tools = [{"name": t["name"], "description": t.get("description", ""),
                  "input_schema": t.get("parameters", {"type": "object",
                                                       "properties": {}})}
                 for t in request.tools]

        # Anthropic has no json_schema response format: structured output is
        # implemented as a forced single-tool call, then unwrapped on the way back.
        if request.response_schema:
            tools.append({"name": _STRUCTURED_TOOL,
                          "description": "Return the final structured result.",
                          "input_schema": request.response_schema})
            payload["tool_choice"] = {"type": "tool", "name": _STRUCTURED_TOOL}
        if tools:
            payload["tools"] = tools
        if stream:
            payload["stream"] = True
        return payload

    # --------------------------------------------------------------- generate
    async def generate(self, request: ModelRequest) -> ModelResponse:
        raw = await self.transport.post_json(
            self._url, headers=self._headers(),
            payload=self._payload(request, stream=False),
            timeout=request.timeout_seconds,
        )
        return _parse_response(raw, fallback_model=self.model,
                               structured=bool(request.response_schema))

    # ----------------------------------------------------------------- stream
    async def stream(self, request: ModelRequest) -> AsyncIterator[StreamChunk]:
        structured = bool(request.response_schema)
        text_parts: list[str] = []
        blocks: dict[int, dict[str, Any]] = {}
        input_tokens = output_tokens = cached = 0
        stop_reason: str | None = None

        lines = self.transport.post_stream(
            self._url, headers=self._headers(),
            payload=self._payload(request, stream=True),
            timeout=request.timeout_seconds,
        )
        async for event in sse_lines(lines):
            kind = event.get("type")
            if kind == "message_start":
                usage = event.get("message", {}).get("usage", {})
                input_tokens = usage.get("input_tokens", 0)
                cached = usage.get("cache_read_input_tokens", 0)
            elif kind == "content_block_start":
                block = event.get("content_block", {})
                if block.get("type") == "tool_use":
                    blocks[event.get("index", 0)] = {
                        "id": block.get("id"), "name": block.get("name", ""),
                        "json": "",
                    }
            elif kind == "content_block_delta":
                delta = event.get("delta", {})
                if delta.get("type") == "text_delta":
                    text = delta.get("text", "")
                    text_parts.append(text)
                    yield StreamChunk(type=ChunkType.TEXT, text=text)
                elif delta.get("type") == "thinking_delta":
                    yield StreamChunk(type=ChunkType.REASONING,
                                      text=delta.get("thinking", ""))
                elif delta.get("type") == "input_json_delta":
                    slot = blocks.setdefault(event.get("index", 0),
                                             {"id": None, "name": "", "json": ""})
                    slot["json"] += delta.get("partial_json", "")
            elif kind == "message_delta":
                stop_reason = event.get("delta", {}).get("stop_reason", stop_reason)
                output_tokens = event.get("usage", {}).get("output_tokens",
                                                           output_tokens)

        calls = [ToolCall(id=blocks[k]["id"] or tool_call_id(),
                          name=blocks[k]["name"], arguments=_loads(blocks[k]["json"]))
                 for k in sorted(blocks)]
        text = "".join(text_parts) or None
        if structured:
            text, calls = _unwrap_structured(text, calls)
        for index, call in enumerate(calls):
            yield StreamChunk(type=ChunkType.TOOL_CALL, tool_call=call, index=index)

        yield StreamChunk(type=ChunkType.DONE, response=ModelResponse(
            text=text, tool_calls=calls, input_tokens=input_tokens,
            output_tokens=output_tokens, cached_tokens=cached,
            finish_reason=stop_reason, model=self.model,
        ))

    async def close(self) -> None:
        await self.transport.aclose()


# --------------------------------------------------------------------------- #
# translation helpers
# --------------------------------------------------------------------------- #
def _split_system(messages: list[Message]) -> tuple[str, list[dict[str, Any]]]:
    """Anthropic takes the system prompt as a top-level field."""
    system_parts: list[str] = []
    wire: list[dict[str, Any]] = []
    pending_results: list[dict[str, Any]] = []

    def flush() -> None:
        if pending_results:
            wire.append({"role": "user", "content": list(pending_results)})
            pending_results.clear()

    for message in messages:
        if message.role is Role.SYSTEM:
            if message.content:
                system_parts.append(message.content)
            continue
        if message.role is Role.TOOL:
            pending_results.append({"type": "tool_result",
                                    "tool_use_id": message.tool_call_id,
                                    "content": message.content or ""})
            continue
        flush()
        if message.role is Role.ASSISTANT:
            content: list[dict[str, Any]] = []
            if message.content:
                content.append({"type": "text", "text": message.content})
            for call in message.tool_calls:
                content.append({"type": "tool_use", "id": call.id,
                                "name": call.name, "input": call.arguments})
            wire.append({"role": "assistant",
                         "content": content or [{"type": "text", "text": ""}]})
        else:
            wire.append({"role": "user",
                         "content": [{"type": "text", "text": message.content or ""}]})
    flush()
    return "\n\n".join(system_parts), wire


def _parse_response(raw: dict[str, Any], *, fallback_model: str,
                    structured: bool) -> ModelResponse:
    if raw.get("type") == "error":
        raise ModelError("Anthropic returned an error",
                         body=str(raw.get("error"))[:400])
    text_parts: list[str] = []
    calls: list[ToolCall] = []
    for block in raw.get("content", []):
        if block.get("type") == "text":
            text_parts.append(block.get("text", ""))
        elif block.get("type") == "tool_use":
            calls.append(ToolCall(id=block.get("id") or tool_call_id(),
                                  name=block.get("name", ""),
                                  arguments=block.get("input") or {}))
    usage = raw.get("usage") or {}
    text = "".join(text_parts) or None
    if structured:
        text, calls = _unwrap_structured(text, calls)
    return ModelResponse(
        text=text,
        tool_calls=calls,
        input_tokens=usage.get("input_tokens", 0),
        output_tokens=usage.get("output_tokens", 0),
        cached_tokens=usage.get("cache_read_input_tokens", 0),
        finish_reason=raw.get("stop_reason"),
        model=raw.get("model", fallback_model),
        raw_response=raw,
    )


def _unwrap_structured(text: str | None,
                       calls: list[ToolCall]) -> tuple[str | None, list[ToolCall]]:
    """Turn the forced structured-output tool call back into JSON text."""
    remaining = [c for c in calls if c.name != _STRUCTURED_TOOL]
    for call in calls:
        if call.name == _STRUCTURED_TOOL:
            return json.dumps(call.arguments), remaining
    return text, remaining


def _loads(payload: str) -> dict[str, Any]:
    if not payload:
        return {}
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError:
        return {"__raw__": payload}
    return parsed if isinstance(parsed, dict) else {"__value__": parsed}
