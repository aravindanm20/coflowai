"""Reference adapter template.

Copy this file into a separate distribution (``coflowai-openai``,
``coflowai-anthropic``, ...).  The core never imports it, and it must never be
imported by agent code — registration happens once, at wiring time:

    registry.register("smart-model", MyProvider(client), capabilities=...)

Adapter responsibilities:

1. translate :class:`ModelRequest` into the provider's wire format
2. translate the provider response into :class:`ModelResponse`
3. report token usage (and cost when the provider exposes it)
4. raise ``ModelError`` / ``RateLimitError`` / ``ModelUnavailable`` — never leak
   provider SDK exceptions
"""

from __future__ import annotations

import json
from typing import Any

from ..core.exceptions import ModelError, ModelUnavailable, RateLimitError
from ..core.ids import tool_call_id
from ..core.types import Message, Role, ToolCall
from ..models.base import ModelCapabilities, ModelProvider, ModelRequest, ModelResponse

__all__ = ["TemplateProvider", "to_wire_messages", "to_wire_tools",
           "DEFAULT_CAPABILITIES"]

DEFAULT_CAPABILITIES = ModelCapabilities(
    tool_calling=True, structured_output=True, json_schema=True,
    parallel_tool_calls=True, streaming=True, reasoning=False,
)


def to_wire_messages(messages: list[Message]) -> list[dict[str, Any]]:
    """Chat-completions-style message translation (most providers accept this)."""
    wire: list[dict[str, Any]] = []
    for message in messages:
        if message.role is Role.TOOL:
            wire.append({"role": "tool", "tool_call_id": message.tool_call_id,
                         "content": message.content or ""})
            continue
        entry: dict[str, Any] = {"role": message.role.value,
                                 "content": message.content or ""}
        if message.tool_calls:
            entry["tool_calls"] = [
                {"id": call.id, "type": "function",
                 "function": {"name": call.name,
                              "arguments": json.dumps(call.arguments)}}
                for call in message.tool_calls
            ]
        wire.append(entry)
    return wire


def to_wire_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{"type": "function",
             "function": {"name": t["name"], "description": t.get("description", ""),
                          "parameters": t.get("parameters", {})}}
            for t in tools]


class TemplateProvider(ModelProvider):
    """Skeleton showing the required shape of a real adapter."""

    name = "template"

    def __init__(self, client: Any, model: str) -> None:
        self.client = client
        self.model = model

    async def generate(self, request: ModelRequest) -> ModelResponse:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": to_wire_messages(request.messages),
        }
        if request.tools:
            payload["tools"] = to_wire_tools(request.tools)
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.max_output_tokens:
            payload["max_tokens"] = request.max_output_tokens
        if request.response_schema:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "output", "schema": request.response_schema,
                                "strict": True},
            }

        try:
            raw = await self.client.chat_completions(**payload)
        except Exception as exc:  # translate, never leak
            status = getattr(exc, "status_code", None)
            if status == 429:
                raise RateLimitError("provider rate limited", cause=exc) from exc
            if status in (500, 502, 503, 504):
                raise ModelUnavailable("provider unavailable", cause=exc) from exc
            raise ModelError("provider request failed", cause=exc) from exc

        choice = raw["choices"][0]
        message = choice["message"]
        calls = [
            ToolCall(id=c.get("id") or tool_call_id(),
                     name=c["function"]["name"],
                     arguments=json.loads(c["function"].get("arguments") or "{}"))
            for c in message.get("tool_calls", []) or []
        ]
        usage = raw.get("usage", {})
        return ModelResponse(
            text=message.get("content"),
            tool_calls=calls,
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
            cached_tokens=usage.get("prompt_tokens_details", {}).get("cached_tokens", 0),
            finish_reason=choice.get("finish_reason"),
            model=raw.get("model", self.model),
            raw_response=raw,
        )
