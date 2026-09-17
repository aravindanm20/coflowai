"""AWS Bedrock Converse API adapter.

Bedrock's Converse API normalises message shape across model families, so this
adapter stays small.  Authentication uses SigV4, which is delegated to the
``boto3``/``botocore`` session the caller supplies — the framework core never
imports it.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, AsyncIterator

from ..core.exceptions import ConfigurationError, ModelError, ModelUnavailable
from ..core.exceptions import RateLimitError
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

__all__ = ["BedrockProvider", "BEDROCK_CAPABILITIES"]

BEDROCK_CAPABILITIES = ModelCapabilities(
    tool_calling=True, structured_output=True, json_schema=True, vision=True,
    streaming=True, parallel_tool_calls=True, reasoning=True,
    max_context_tokens=200_000,
)

_STRUCTURED_TOOL = "emit_structured_output"


class BedrockProvider(ModelProvider):
    """Adapter over a ``bedrock-runtime`` client.

    The boto3 client is synchronous, so calls are dispatched to a worker thread
    and never block the event loop.
    """

    name = "bedrock"

    def __init__(self, *, model_id: str, client: Any = None,
                 region_name: str | None = None,
                 pricing: ModelPricing | None = None,
                 inference_config: dict[str, Any] | None = None) -> None:
        self.model_id = model_id
        self.pricing = pricing or ModelPricing()
        self.inference_config = dict(inference_config or {})
        self._client = client or self._build_client(region_name)

    @staticmethod
    def _build_client(region_name: str | None) -> Any:
        try:
            import boto3  # noqa: PLC0415 - optional dependency, adapter-only
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ConfigurationError(
                'boto3 is required for Bedrock: pip install "coflowai[bedrock]"',
                cause=exc,
            ) from exc
        return boto3.client("bedrock-runtime", region_name=region_name)

    # ------------------------------------------------------------------ build
    def _payload(self, request: ModelRequest) -> dict[str, Any]:
        system, messages = _split_system(request.messages)
        config = dict(self.inference_config)
        if request.temperature is not None:
            config["temperature"] = request.temperature
        if request.max_output_tokens:
            config["maxTokens"] = request.max_output_tokens
        if request.stop:
            config["stopSequences"] = request.stop

        payload: dict[str, Any] = {"modelId": self.model_id, "messages": messages}
        if system:
            payload["system"] = [{"text": system}]
        if config:
            payload["inferenceConfig"] = config

        specs = [{"toolSpec": {"name": t["name"],
                               "description": t.get("description", ""),
                               "inputSchema": {"json": t.get("parameters", {})}}}
                 for t in request.tools]
        # Converse has no JSON-schema response format: force a single tool call.
        if request.response_schema:
            specs.append({"toolSpec": {
                "name": _STRUCTURED_TOOL,
                "description": "Return the final structured result.",
                "inputSchema": {"json": request.response_schema},
            }})
        if specs:
            payload["toolConfig"] = {"tools": specs}
            if request.response_schema:
                payload["toolConfig"]["toolChoice"] = {
                    "tool": {"name": _STRUCTURED_TOOL}}
        return payload

    # --------------------------------------------------------------- generate
    async def generate(self, request: ModelRequest) -> ModelResponse:
        payload = self._payload(request)
        try:
            raw = await asyncio.to_thread(self._client.converse, **payload)
        except Exception as exc:  # translate botocore errors
            raise _translate(exc) from exc
        return _parse_response(raw, fallback_model=self.model_id,
                               structured=bool(request.response_schema))

    # ----------------------------------------------------------------- stream
    async def stream(self, request: ModelRequest) -> AsyncIterator[StreamChunk]:
        payload = self._payload(request)
        structured = bool(request.response_schema)
        try:
            raw = await asyncio.to_thread(self._client.converse_stream, **payload)
        except Exception as exc:
            raise _translate(exc) from exc

        text_parts: list[str] = []
        blocks: dict[int, dict[str, Any]] = {}
        usage: dict[str, Any] = {}
        stop_reason: str | None = None

        for event in raw.get("stream", []):
            if "contentBlockStart" in event:
                start = event["contentBlockStart"]
                tool_use = start.get("start", {}).get("toolUse")
                if tool_use:
                    blocks[start.get("contentBlockIndex", 0)] = {
                        "id": tool_use.get("toolUseId"),
                        "name": tool_use.get("name", ""), "json": ""}
            elif "contentBlockDelta" in event:
                block = event["contentBlockDelta"]
                delta = block.get("delta", {})
                if "text" in delta:
                    text_parts.append(delta["text"])
                    yield StreamChunk(type=ChunkType.TEXT, text=delta["text"])
                elif "toolUse" in delta:
                    slot = blocks.setdefault(block.get("contentBlockIndex", 0),
                                             {"id": None, "name": "", "json": ""})
                    slot["json"] += delta["toolUse"].get("input", "")
            elif "messageStop" in event:
                stop_reason = event["messageStop"].get("stopReason")
            elif "metadata" in event:
                usage = event["metadata"].get("usage", usage)

        calls = [ToolCall(id=blocks[k]["id"] or tool_call_id(),
                          name=blocks[k]["name"], arguments=_loads(blocks[k]["json"]))
                 for k in sorted(blocks)]
        text = "".join(text_parts) or None
        if structured:
            text, calls = _unwrap_structured(text, calls)
        for index, call in enumerate(calls):
            yield StreamChunk(type=ChunkType.TOOL_CALL, tool_call=call, index=index)

        yield StreamChunk(type=ChunkType.DONE, response=ModelResponse(
            text=text, tool_calls=calls,
            input_tokens=usage.get("inputTokens", 0),
            output_tokens=usage.get("outputTokens", 0),
            cached_tokens=usage.get("cacheReadInputTokens", 0),
            finish_reason=stop_reason, model=self.model_id,
        ))


# --------------------------------------------------------------------------- #
# translation helpers
# --------------------------------------------------------------------------- #
def _split_system(messages: list[Message]) -> tuple[str, list[dict[str, Any]]]:
    system_parts: list[str] = []
    wire: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []

    def flush() -> None:
        if pending:
            wire.append({"role": "user", "content": list(pending)})
            pending.clear()

    for message in messages:
        if message.role is Role.SYSTEM:
            if message.content:
                system_parts.append(message.content)
            continue
        if message.role is Role.TOOL:
            pending.append({"toolResult": {
                "toolUseId": message.tool_call_id,
                "content": [{"text": message.content or ""}]}})
            continue
        flush()
        if message.role is Role.ASSISTANT:
            content: list[dict[str, Any]] = []
            if message.content:
                content.append({"text": message.content})
            for call in message.tool_calls:
                content.append({"toolUse": {"toolUseId": call.id,
                                            "name": call.name,
                                            "input": call.arguments}})
            wire.append({"role": "assistant", "content": content or [{"text": ""}]})
        else:
            wire.append({"role": "user",
                         "content": [{"text": message.content or ""}]})
    flush()
    return "\n\n".join(system_parts), wire


def _parse_response(raw: dict[str, Any], *, fallback_model: str,
                    structured: bool) -> ModelResponse:
    message = raw.get("output", {}).get("message")
    if message is None:
        raise ModelError("Bedrock response contained no message",
                         body=str(raw)[:400])
    text_parts: list[str] = []
    calls: list[ToolCall] = []
    for block in message.get("content", []):
        if "text" in block:
            text_parts.append(block["text"])
        elif "toolUse" in block:
            use = block["toolUse"]
            calls.append(ToolCall(id=use.get("toolUseId") or tool_call_id(),
                                  name=use.get("name", ""),
                                  arguments=use.get("input") or {}))
    usage = raw.get("usage") or {}
    text = "".join(text_parts) or None
    if structured:
        text, calls = _unwrap_structured(text, calls)
    return ModelResponse(
        text=text, tool_calls=calls,
        input_tokens=usage.get("inputTokens", 0),
        output_tokens=usage.get("outputTokens", 0),
        cached_tokens=usage.get("cacheReadInputTokens", 0),
        finish_reason=raw.get("stopReason"),
        model=fallback_model, raw_response=raw,
    )


def _unwrap_structured(text: str | None,
                       calls: list[ToolCall]) -> tuple[str | None, list[ToolCall]]:
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


def _translate(exc: BaseException) -> BaseException:
    """Map botocore client errors onto the framework error model."""
    name = type(exc).__name__
    code = ""
    response = getattr(exc, "response", None)
    if isinstance(response, dict):
        code = response.get("Error", {}).get("Code", "")
    if "Throttl" in name or "Throttl" in code or "TooManyRequests" in code:
        return RateLimitError("Bedrock throttled the request", cause=exc, code=code)
    if "ServiceUnavailable" in code or "InternalServer" in code or "Timeout" in name:
        return ModelUnavailable("Bedrock is unavailable", cause=exc, code=code)
    if "AccessDenied" in code or "UnrecognizedClient" in code:
        return ConfigurationError("Bedrock rejected the credentials", cause=exc,
                                  code=code)
    return ModelError("Bedrock request failed", cause=exc, code=code)
