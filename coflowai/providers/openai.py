"""OpenAI / Azure OpenAI / OpenAI-compatible adapter.

Covers OpenAI, Azure OpenAI, and any OpenAI-compatible endpoint (vLLM, Together,
Groq, Ollama, OpenRouter, ...) by pointing ``base_url`` at it.

No part of the framework core imports this module; register it at wiring time:

    app.models.register("smart-model", OpenAIProvider(api_key=..., model="gpt-4o"),
                        capabilities=OPENAI_CAPABILITIES,
                        pricing=ModelPricing(2.5, 10.0))
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

__all__ = ["OpenAIProvider", "AzureOpenAIProvider", "OPENAI_CAPABILITIES",
           "OPENAI_PRICING"]

OPENAI_CAPABILITIES = ModelCapabilities(
    tool_calling=True, structured_output=True, json_schema=True, vision=True,
    streaming=True, parallel_tool_calls=True, reasoning=False,
    max_context_tokens=128_000,
)

#: USD per 1M tokens — verify against current provider pricing before relying on it.
OPENAI_PRICING: dict[str, ModelPricing] = {
    "gpt-4o": ModelPricing(2.5, 10.0, 1.25),
    "gpt-4o-mini": ModelPricing(0.15, 0.6, 0.075),
    "gpt-4.1": ModelPricing(2.0, 8.0, 0.5),
    "gpt-4.1-mini": ModelPricing(0.4, 1.6, 0.1),
    "o3-mini": ModelPricing(1.1, 4.4, 0.55),
}


class OpenAIProvider(ModelProvider):
    """Chat Completions adapter."""

    name = "openai"

    def __init__(self, *, model: str, api_key: str | None = None,
                 base_url: str = "https://api.openai.com/v1",
                 organization: str | None = None,
                 transport: HttpTransport | None = None,
                 extra_headers: dict[str, str] | None = None,
                 extra_body: dict[str, Any] | None = None,
                 strict_schema: bool = True) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.organization = organization
        self.extra_headers = dict(extra_headers or {})
        self.extra_body = dict(extra_body or {})
        self.strict_schema = strict_schema
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if self._api_key is None and transport is None:
            raise ConfigurationError(
                "OpenAI API key missing: pass api_key= or set OPENAI_API_KEY")
        self.transport = transport or HttpxTransport(provider=self.name)

    # ------------------------------------------------------------------ wiring
    @property
    def _url(self) -> str:
        return f"{self.base_url}/chat/completions"

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json", **self.extra_headers}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        if self.organization:
            headers["OpenAI-Organization"] = self.organization
        return headers

    def _payload(self, request: ModelRequest, *, stream: bool) -> dict[str, Any]:
        payload: dict[str, Any] = {
            # request.model is the *logical* registry name; the wire needs the
            # provider's own model id.
            "model": self.model,
            "messages": _to_wire_messages(request.messages),
            **self.extra_body,
        }
        if request.tools:
            payload["tools"] = [
                {"type": "function",
                 "function": {"name": t["name"],
                              "description": t.get("description", ""),
                              "parameters": t.get("parameters", {})}}
                for t in request.tools
            ]
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.max_output_tokens:
            payload["max_completion_tokens"] = request.max_output_tokens
        if request.stop:
            payload["stop"] = request.stop
        if request.response_schema:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "structured_output",
                                "schema": _harden(request.response_schema)
                                if self.strict_schema else request.response_schema,
                                "strict": self.strict_schema},
            }
        if stream:
            payload["stream"] = True
            payload["stream_options"] = {"include_usage": True}
        return payload

    # --------------------------------------------------------------- generate
    async def generate(self, request: ModelRequest) -> ModelResponse:
        raw = await self.transport.post_json(
            self._url, headers=self._headers(),
            payload=self._payload(request, stream=False),
            timeout=request.timeout_seconds,
        )
        return _parse_response(raw, fallback_model=self.model)

    # ----------------------------------------------------------------- stream
    async def stream(self, request: ModelRequest) -> AsyncIterator[StreamChunk]:
        text_parts: list[str] = []
        calls: dict[int, dict[str, Any]] = {}
        usage: dict[str, Any] = {}
        finish_reason: str | None = None
        model_name = self.model

        lines = self.transport.post_stream(
            self._url, headers=self._headers(),
            payload=self._payload(request, stream=True),
            timeout=request.timeout_seconds,
        )
        async for event in sse_lines(lines):
            model_name = event.get("model", model_name)
            if event.get("usage"):
                usage = event["usage"]
            for choice in event.get("choices", []):
                finish_reason = choice.get("finish_reason") or finish_reason
                delta = choice.get("delta", {})
                content = delta.get("content")
                if content:
                    text_parts.append(content)
                    yield StreamChunk(type=ChunkType.TEXT, text=content)
                for fragment in delta.get("tool_calls", []) or []:
                    index = fragment.get("index", 0)
                    slot = calls.setdefault(index, {"id": fragment.get("id"),
                                                    "name": "", "arguments": ""})
                    slot["id"] = fragment.get("id") or slot["id"]
                    function = fragment.get("function", {})
                    slot["name"] += function.get("name") or ""
                    slot["arguments"] += function.get("arguments") or ""

        tool_calls = [
            ToolCall(id=calls[key]["id"] or tool_call_id(),
                     name=calls[key]["name"],
                     arguments=_loads(calls[key]["arguments"]))
            for key in sorted(calls)
        ]
        for index, call in enumerate(tool_calls):
            yield StreamChunk(type=ChunkType.TOOL_CALL, tool_call=call, index=index)

        details = usage.get("prompt_tokens_details") or {}
        yield StreamChunk(type=ChunkType.DONE, response=ModelResponse(
            text="".join(text_parts) or None,
            tool_calls=tool_calls,
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
            cached_tokens=details.get("cached_tokens", 0),
            finish_reason=finish_reason,
            model=model_name,
        ))

    async def close(self) -> None:
        await self.transport.aclose()


class AzureOpenAIProvider(OpenAIProvider):
    """Azure deployment-based routing (``api-key`` header, api-version query)."""

    name = "azure_openai"

    def __init__(self, *, endpoint: str, deployment: str,
                 api_version: str = "2024-10-21", api_key: str | None = None,
                 transport: HttpTransport | None = None, **kwargs: Any) -> None:
        key = api_key or os.environ.get("AZURE_OPENAI_API_KEY")
        if key is None and transport is None:
            raise ConfigurationError(
                "Azure OpenAI key missing: pass api_key= or set "
                "AZURE_OPENAI_API_KEY")
        self.endpoint = endpoint.rstrip("/")
        self.deployment = deployment
        self.api_version = api_version
        super().__init__(model=deployment, api_key=key or "unused",
                         base_url=self.endpoint, transport=transport, **kwargs)

    @property
    def _url(self) -> str:
        return (f"{self.endpoint}/openai/deployments/{self.deployment}"
                f"/chat/completions?api-version={self.api_version}")

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json", **self.extra_headers}
        if self._api_key:
            headers["api-key"] = self._api_key
        return headers


# --------------------------------------------------------------------------- #
# translation helpers
# --------------------------------------------------------------------------- #
def _to_wire_messages(messages: list[Message]) -> list[dict[str, Any]]:
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


def _parse_response(raw: dict[str, Any], *, fallback_model: str) -> ModelResponse:
    try:
        choice = raw["choices"][0]
    except (KeyError, IndexError) as exc:
        raise ModelError("OpenAI response contained no choices",
                         cause=exc, body=str(raw)[:400]) from exc
    message = choice.get("message", {})
    calls = [
        ToolCall(id=item.get("id") or tool_call_id(),
                 name=item["function"]["name"],
                 arguments=_loads(item["function"].get("arguments")))
        for item in message.get("tool_calls") or []
    ]
    usage = raw.get("usage") or {}
    details = usage.get("prompt_tokens_details") or {}
    return ModelResponse(
        text=message.get("content"),
        tool_calls=calls,
        input_tokens=usage.get("prompt_tokens", 0),
        output_tokens=usage.get("completion_tokens", 0),
        cached_tokens=details.get("cached_tokens", 0),
        finish_reason=choice.get("finish_reason"),
        model=raw.get("model", fallback_model),
        raw_response=raw,
    )


def _loads(payload: str | None) -> dict[str, Any]:
    """Model-produced arguments are untrusted: never explode on bad JSON."""
    if not payload:
        return {}
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError:
        return {"__raw__": payload}
    return parsed if isinstance(parsed, dict) else {"__value__": parsed}


def _harden(schema: dict[str, Any]) -> dict[str, Any]:
    """OpenAI strict mode requires ``additionalProperties: false`` everywhere."""
    if not isinstance(schema, dict):
        return schema
    hardened = dict(schema)
    if hardened.get("type") == "object":
        hardened.setdefault("additionalProperties", False)
        properties = hardened.get("properties") or {}
        hardened["properties"] = {k: _harden(v) for k, v in properties.items()}
        hardened.setdefault("required", sorted(properties))
    for key in ("items", "$defs", "definitions"):
        value = hardened.get(key)
        if isinstance(value, dict):
            hardened[key] = ({k: _harden(v) for k, v in value.items()}
                             if key != "items" else _harden(value))
    return hardened
