"""Deterministic test doubles.

These make it possible to unit-test agents and workflows with no network, no
API keys, no token spend and no nondeterminism.
"""

from __future__ import annotations

import inspect
import json
from typing import Any, AsyncIterator, Callable, Iterable, Sequence

from ..core.exceptions import ModelError
from ..core.ids import tool_call_id
from ..core.types import ToolCall
from ..models.base import (
    ModelCapabilities,
    ModelPricing,
    ModelProvider,
    ModelRequest,
    ModelResponse,
)
from ..models.streaming import ChunkType, StreamChunk
from ..tools.tool import Tool, ToolDefinition

__all__ = ["FakeModelProvider", "FakeTool", "scripted", "text", "tool_call"]

Scripted = Any  # str | dict | ModelResponse | ToolCall | list[ToolCall] | callable


def text(content: str) -> ModelResponse:
    return ModelResponse(text=content, finish_reason="stop")


def tool_call(name: str, **arguments: Any) -> ToolCall:
    return ToolCall(id=tool_call_id(), name=name, arguments=arguments)


def scripted(*responses: Scripted) -> "FakeModelProvider":
    return FakeModelProvider(list(responses))


class FakeModelProvider(ModelProvider):
    """A model that returns a scripted sequence of responses.

    Each entry may be:

    * ``str``                      -> assistant text
    * ``dict``                     -> JSON encoded structured output
    * ``ToolCall`` / list thereof  -> tool call request
    * ``ModelResponse``            -> used verbatim
    * ``callable(request)``        -> computed at call time
    """

    name = "fake"

    def __init__(self, responses: Sequence[Scripted] | None = None, *,
                 handler: Callable[[ModelRequest], Scripted] | None = None,
                 capabilities: ModelCapabilities | None = None,
                 pricing: ModelPricing | None = None,
                 repeat_last: bool = False,
                 failures: Iterable[BaseException] | None = None) -> None:
        self.responses: list[Scripted] = list(responses or ())
        self.handler = handler
        self.capabilities = capabilities or ModelCapabilities(
            tool_calling=True, structured_output=True, json_schema=True,
            reasoning=True, parallel_tool_calls=True,
        )
        self.pricing = pricing or ModelPricing(input_per_million=1.0,
                                               output_per_million=3.0)
        self.repeat_last = repeat_last
        self.calls: list[ModelRequest] = []
        self._failures = list(failures or ())
        self._index = 0

    # ------------------------------------------------------------------ model
    async def generate(self, request: ModelRequest) -> ModelResponse:
        self.calls.append(request)

        if self._failures:
            raise self._failures.pop(0)

        if self.handler is not None:
            payload = self.handler(request)
        elif self._index < len(self.responses):
            payload = self.responses[self._index]
            self._index += 1
        elif self.repeat_last and self.responses:
            payload = self.responses[-1]
        else:
            raise ModelError("FakeModelProvider script exhausted",
                             calls=len(self.calls))

        if callable(payload) and not isinstance(payload, ModelResponse):
            payload = payload(request)
        if inspect.isawaitable(payload):
            payload = await payload

        response = _coerce(payload)
        response.input_tokens = response.input_tokens or _count_input(request)
        response.output_tokens = response.output_tokens or _count_output(response)
        response.cost = self.pricing.estimate(response.input_tokens,
                                              response.output_tokens)
        response.model = request.model or "fake-model"
        return response

    async def stream(self, request: ModelRequest) -> AsyncIterator[StreamChunk]:
        """Deterministic word-by-word streaming, for testing stream consumers."""
        response = await self.generate(request)
        if response.text:
            words = response.text.split(" ")
            for index, word in enumerate(words):
                suffix = "" if index == len(words) - 1 else " "
                yield StreamChunk(type=ChunkType.TEXT, text=word + suffix,
                                  index=index)
        for index, call in enumerate(response.tool_calls):
            yield StreamChunk(type=ChunkType.TOOL_CALL, tool_call=call, index=index)
        yield StreamChunk(type=ChunkType.DONE, response=response)

    @property
    def call_count(self) -> int:
        return len(self.calls)

    def reset(self) -> None:
        self._index = 0
        self.calls.clear()


def _coerce(payload: Scripted) -> ModelResponse:
    if isinstance(payload, ModelResponse):
        return payload
    if isinstance(payload, ToolCall):
        return ModelResponse(tool_calls=[payload], finish_reason="tool_calls")
    if isinstance(payload, list) and payload and isinstance(payload[0], ToolCall):
        return ModelResponse(tool_calls=list(payload), finish_reason="tool_calls")
    if isinstance(payload, (dict, list)):
        return ModelResponse(text=json.dumps(payload), finish_reason="stop")
    return ModelResponse(text=str(payload), finish_reason="stop")


def _count_input(request: ModelRequest) -> int:
    chars = sum(len(m.content or "") for m in request.messages)
    chars += sum(len(json.dumps(t, default=str)) for t in request.tools)
    return max(1, chars // 4)


def _count_output(response: ModelResponse) -> int:
    chars = len(response.text or "")
    chars += sum(len(json.dumps(c.arguments, default=str)) + len(c.name)
                 for c in response.tool_calls)
    return max(1, chars // 4)


class FakeTool(Tool):
    """A tool with a canned result — keeps workflow tests deterministic."""

    def __init__(self, name: str, result: Any = None, *,
                 description: str = "fake tool",
                 error: BaseException | None = None,
                 permission: str | None = None,
                 input_schema: dict[str, Any] | None = None,
                 side_effect: bool = False,
                 idempotent: bool = True) -> None:
        definition = ToolDefinition(
            name=name,
            description=description,
            input_schema=input_schema or {"type": "object", "properties": {},
                                          "additionalProperties": True},
            permission=permission,
            timeout=5.0,
            retries=0,
            side_effect=side_effect,
            idempotent=idempotent,
        )
        super().__init__(definition=definition, func=self._call, validate_input=False)
        self.result = result
        self.error = error
        self.invocations: list[dict[str, Any]] = []

    async def _call(self, **kwargs: Any) -> Any:
        self.invocations.append(kwargs)
        if self.error is not None:
            raise self.error
        if callable(self.result):
            return self.result(**kwargs)
        return self.result

    @property
    def call_count(self) -> int:
        return len(self.invocations)
