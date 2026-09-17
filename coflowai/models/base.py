"""Provider-independent model interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, AsyncIterator

from ..core.types import Message, ToolCall

if TYPE_CHECKING:  # pragma: no cover
    from .streaming import StreamChunk

__all__ = [
    "ModelRequest",
    "ModelResponse",
    "ModelCapabilities",
    "ModelPricing",
    "ModelProvider",
]


@dataclass(slots=True)
class ModelRequest:
    messages: list[Message]
    tools: list[dict[str, Any]] = field(default_factory=list)
    temperature: float | None = None
    max_output_tokens: int | None = None
    response_schema: dict[str, Any] | None = None
    stop: list[str] = field(default_factory=list)
    timeout_seconds: float | None = None
    model: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "messages": [m.to_dict() for m in self.messages],
            "tools": [t.get("name") for t in self.tools],
            "temperature": self.temperature,
            "max_output_tokens": self.max_output_tokens,
            "has_response_schema": self.response_schema is not None,
        }


@dataclass(slots=True)
class ModelResponse:
    text: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    cost: float | None = None
    finish_reason: str | None = None
    model: str | None = None
    raw_response: object | None = None

    @property
    def wants_tools(self) -> bool:
        return bool(self.tool_calls)

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "tool_calls": [c.to_dict() for c in self.tool_calls],
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cached_tokens": self.cached_tokens,
            "cost": self.cost,
            "finish_reason": self.finish_reason,
            "model": self.model,
        }


@dataclass(slots=True)
class ModelCapabilities:
    tool_calling: bool = False
    structured_output: bool = False
    vision: bool = False
    reasoning: bool = False
    streaming: bool = False
    parallel_tool_calls: bool = False
    json_schema: bool = False
    max_context_tokens: int = 128_000

    def satisfies(self, requirements: dict[str, bool]) -> bool:
        return all(getattr(self, key, False) for key, wanted in requirements.items()
                   if wanted)

    def missing(self, requirements: dict[str, bool]) -> list[str]:
        return [key for key, wanted in requirements.items()
                if wanted and not getattr(self, key, False)]

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_calling": self.tool_calling,
            "structured_output": self.structured_output,
            "vision": self.vision,
            "reasoning": self.reasoning,
            "streaming": self.streaming,
            "parallel_tool_calls": self.parallel_tool_calls,
            "json_schema": self.json_schema,
            "max_context_tokens": self.max_context_tokens,
        }


@dataclass(slots=True)
class ModelPricing:
    """Cost per 1M tokens.  Used when a provider does not report cost."""

    input_per_million: float = 0.0
    output_per_million: float = 0.0
    cached_input_per_million: float = 0.0

    def estimate(self, input_tokens: int, output_tokens: int,
                 cached_tokens: int = 0) -> float:
        billable_input = max(0, input_tokens - cached_tokens)
        return (
            billable_input * self.input_per_million / 1_000_000
            + output_tokens * self.output_per_million / 1_000_000
            + cached_tokens * self.cached_input_per_million / 1_000_000
        )


class ModelProvider(ABC):
    """Every provider adapter implements exactly this."""

    name: str = "provider"

    @abstractmethod
    async def generate(self, request: ModelRequest) -> ModelResponse:
        ...

    async def stream(self, request: ModelRequest) -> AsyncIterator["StreamChunk"]:
        """Incremental generation.

        The default implementation emulates a stream from :meth:`generate`, so
        every provider is streamable from the caller's point of view.  Adapters
        that support real server-sent events override this.
        """
        from .streaming import emulate_stream

        response = await self.generate(request)
        async for chunk in emulate_stream(response):
            yield chunk

    async def close(self) -> None:  # pragma: no cover - adapters may override
        return None
