"""Google Gemini (Generative Language API) adapter.

Translation notes: Gemini uses ``contents`` with ``parts``, calls the assistant
role ``model``, carries the system prompt as ``system_instruction``, and returns
tool calls as ``functionCall`` parts.  It supports native JSON schema output via
``responseSchema``.
"""

from __future__ import annotations

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

__all__ = ["GeminiProvider", "GEMINI_CAPABILITIES", "GEMINI_PRICING"]

GEMINI_CAPABILITIES = ModelCapabilities(
    tool_calling=True, structured_output=True, json_schema=True, vision=True,
    streaming=True, parallel_tool_calls=True, reasoning=True,
    max_context_tokens=1_000_000,
)

#: USD per 1M tokens — verify against current provider pricing before relying on it.
GEMINI_PRICING: dict[str, ModelPricing] = {
    "gemini-2.0-flash": ModelPricing(0.1, 0.4, 0.025),
    "gemini-2.5-pro": ModelPricing(1.25, 10.0, 0.31),
    "gemini-2.5-flash": ModelPricing(0.3, 2.5, 0.075),
}

_UNSUPPORTED_SCHEMA_KEYS = {"additionalProperties", "$schema", "$defs",
                            "definitions", "title", "default", "examples"}


class GeminiProvider(ModelProvider):
    name = "gemini"

    def __init__(self, *, model: str, api_key: str | None = None,
                 base_url: str = "https://generativelanguage.googleapis.com/v1beta",
                 transport: HttpTransport | None = None,
                 extra_body: dict[str, Any] | None = None) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.extra_body = dict(extra_body or {})
        self._api_key = api_key or os.environ.get("GEMINI_API_KEY") \
            or os.environ.get("GOOGLE_API_KEY")
        if self._api_key is None and transport is None:
            raise ConfigurationError(
                "Gemini API key missing: pass api_key= or set GEMINI_API_KEY")
        self.transport = transport or HttpxTransport(provider=self.name)

    def _url(self, *, stream: bool) -> str:
        verb = "streamGenerateContent?alt=sse" if stream else "generateContent"
        return f"{self.base_url}/models/{self.model}:{verb}"

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["x-goog-api-key"] = self._api_key
        return headers

    def _payload(self, request: ModelRequest) -> dict[str, Any]:
        system, contents = _split_system(request.messages)
        payload: dict[str, Any] = {"contents": contents, **self.extra_body}
        if system:
            payload["system_instruction"] = {"parts": [{"text": system}]}

        config: dict[str, Any] = {}
        if request.temperature is not None:
            config["temperature"] = request.temperature
        if request.max_output_tokens:
            config["maxOutputTokens"] = request.max_output_tokens
        if request.stop:
            config["stopSequences"] = request.stop
        if request.response_schema:
            config["responseMimeType"] = "application/json"
            config["responseSchema"] = _clean_schema(request.response_schema)
        if config:
            payload["generationConfig"] = config

        if request.tools:
            payload["tools"] = [{"functionDeclarations": [
                {"name": t["name"], "description": t.get("description", ""),
                 "parameters": _clean_schema(t.get("parameters", {}))}
                for t in request.tools
            ]}]
        return payload

    # --------------------------------------------------------------- generate
    async def generate(self, request: ModelRequest) -> ModelResponse:
        raw = await self.transport.post_json(
            self._url(stream=False), headers=self._headers(),
            payload=self._payload(request), timeout=request.timeout_seconds,
        )
        return _parse_response(raw, fallback_model=self.model)

    # ----------------------------------------------------------------- stream
    async def stream(self, request: ModelRequest) -> AsyncIterator[StreamChunk]:
        text_parts: list[str] = []
        calls: list[ToolCall] = []
        usage: dict[str, Any] = {}
        finish_reason: str | None = None

        lines = self.transport.post_stream(
            self._url(stream=True), headers=self._headers(),
            payload=self._payload(request), timeout=request.timeout_seconds,
        )
        async for event in sse_lines(lines):
            usage = event.get("usageMetadata", usage)
            for candidate in event.get("candidates", []):
                finish_reason = candidate.get("finishReason") or finish_reason
                for part in candidate.get("content", {}).get("parts", []):
                    if "text" in part and part["text"]:
                        text_parts.append(part["text"])
                        yield StreamChunk(type=ChunkType.TEXT, text=part["text"])
                    elif "functionCall" in part:
                        function = part["functionCall"]
                        calls.append(ToolCall(id=tool_call_id(),
                                              name=function.get("name", ""),
                                              arguments=function.get("args") or {}))
        for index, call in enumerate(calls):
            yield StreamChunk(type=ChunkType.TOOL_CALL, tool_call=call, index=index)

        yield StreamChunk(type=ChunkType.DONE, response=ModelResponse(
            text="".join(text_parts) or None, tool_calls=calls,
            input_tokens=usage.get("promptTokenCount", 0),
            output_tokens=usage.get("candidatesTokenCount", 0),
            cached_tokens=usage.get("cachedContentTokenCount", 0),
            finish_reason=finish_reason, model=self.model,
        ))

    async def close(self) -> None:
        await self.transport.aclose()


# --------------------------------------------------------------------------- #
# translation helpers
# --------------------------------------------------------------------------- #
def _split_system(messages: list[Message]) -> tuple[str, list[dict[str, Any]]]:
    system_parts: list[str] = []
    contents: list[dict[str, Any]] = []
    for message in messages:
        if message.role is Role.SYSTEM:
            if message.content:
                system_parts.append(message.content)
            continue
        if message.role is Role.TOOL:
            contents.append({"role": "user", "parts": [{
                "functionResponse": {"name": message.name or "tool",
                                     "response": {"result": message.content or ""}}
            }]})
            continue
        parts: list[dict[str, Any]] = []
        if message.content:
            parts.append({"text": message.content})
        for call in message.tool_calls:
            parts.append({"functionCall": {"name": call.name,
                                           "args": call.arguments}})
        contents.append({"role": "model" if message.role is Role.ASSISTANT else "user",
                         "parts": parts or [{"text": ""}]})
    return "\n\n".join(system_parts), contents


def _parse_response(raw: dict[str, Any], *, fallback_model: str) -> ModelResponse:
    candidates = raw.get("candidates") or []
    if not candidates:
        if raw.get("promptFeedback", {}).get("blockReason"):
            raise ModelError("Gemini blocked the prompt",
                             reason=raw["promptFeedback"]["blockReason"])
        raise ModelError("Gemini response contained no candidates",
                         body=str(raw)[:400])
    candidate = candidates[0]
    text_parts: list[str] = []
    calls: list[ToolCall] = []
    for part in candidate.get("content", {}).get("parts", []):
        if "text" in part:
            text_parts.append(part["text"])
        elif "functionCall" in part:
            function = part["functionCall"]
            calls.append(ToolCall(id=tool_call_id(), name=function.get("name", ""),
                                  arguments=function.get("args") or {}))
    usage = raw.get("usageMetadata") or {}
    return ModelResponse(
        text="".join(text_parts) or None,
        tool_calls=calls,
        input_tokens=usage.get("promptTokenCount", 0),
        output_tokens=usage.get("candidatesTokenCount", 0),
        cached_tokens=usage.get("cachedContentTokenCount", 0),
        finish_reason=candidate.get("finishReason"),
        model=raw.get("modelVersion", fallback_model),
        raw_response=raw,
    )


def _clean_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Gemini rejects several standard JSON Schema keywords."""
    if not isinstance(schema, dict):
        return schema
    cleaned: dict[str, Any] = {}
    for key, value in schema.items():
        if key in _UNSUPPORTED_SCHEMA_KEYS:
            continue
        if key == "properties" and isinstance(value, dict):
            cleaned[key] = {k: _clean_schema(v) for k, v in value.items()}
        elif key == "items":
            cleaned[key] = _clean_schema(value)
        elif isinstance(value, dict):
            cleaned[key] = _clean_schema(value)
        else:
            cleaned[key] = value
    return cleaned
