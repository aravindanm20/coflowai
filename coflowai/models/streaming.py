"""Streaming support.

Streaming is *presentation*, not control flow: the runtime still accounts for
usage, budgets and events exactly once, when the stream completes.  A provider
that cannot stream is transparently emulated by emitting one final chunk, so
application code never branches on capability.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncIterator

from ..core.types import ToolCall
from .base import ModelResponse

__all__ = ["ChunkType", "StreamChunk", "StreamAccumulator", "emulate_stream"]


class ChunkType(str, Enum):
    TEXT = "text"
    TOOL_CALL = "tool_call"
    REASONING = "reasoning"
    DONE = "done"


@dataclass(slots=True)
class StreamChunk:
    """One incremental piece of a model response."""

    type: ChunkType = ChunkType.TEXT
    text: str = ""
    tool_call: ToolCall | None = None
    index: int = 0
    response: ModelResponse | None = None   # populated on the DONE chunk
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_final(self) -> bool:
        return self.type is ChunkType.DONE

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type.value,
            "text": self.text,
            "tool_call": self.tool_call.to_dict() if self.tool_call else None,
            "index": self.index,
            "final": self.is_final,
        }


class StreamAccumulator:
    """Rebuilds a complete :class:`ModelResponse` from a chunk stream.

    Adapters that emit deltas use this so the runtime always ends up with the
    same response object it would have received from a non-streaming call.
    """

    def __init__(self) -> None:
        self._text: list[str] = []
        self._reasoning: list[str] = []
        self._tool_calls: dict[int, ToolCall] = {}

    def add(self, chunk: StreamChunk) -> None:
        if chunk.type is ChunkType.TEXT and chunk.text:
            self._text.append(chunk.text)
        elif chunk.type is ChunkType.REASONING and chunk.text:
            self._reasoning.append(chunk.text)
        elif chunk.type is ChunkType.TOOL_CALL and chunk.tool_call is not None:
            existing = self._tool_calls.get(chunk.index)
            if existing is None:
                self._tool_calls[chunk.index] = chunk.tool_call
            else:
                # merge streamed argument fragments
                existing.name = chunk.tool_call.name or existing.name
                existing.arguments.update(chunk.tool_call.arguments)

    @property
    def text(self) -> str:
        return "".join(self._text)

    def build(self, *, input_tokens: int = 0, output_tokens: int = 0,
              cached_tokens: int = 0, cost: float | None = None,
              finish_reason: str | None = None, model: str | None = None,
              raw_response: object | None = None) -> ModelResponse:
        calls = [self._tool_calls[key] for key in sorted(self._tool_calls)]
        return ModelResponse(
            text=self.text or None,
            tool_calls=calls,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_tokens=cached_tokens,
            cost=cost,
            finish_reason=finish_reason or ("tool_calls" if calls else "stop"),
            model=model,
            raw_response=raw_response,
        )


async def emulate_stream(response: ModelResponse) -> AsyncIterator[StreamChunk]:
    """Turn a non-streaming response into a two-chunk stream.

    Lets callers use one code path regardless of provider capability.
    """
    if response.text:
        yield StreamChunk(type=ChunkType.TEXT, text=response.text)
    for index, call in enumerate(response.tool_calls):
        yield StreamChunk(type=ChunkType.TOOL_CALL, tool_call=call, index=index)
    yield StreamChunk(type=ChunkType.DONE, response=response)
