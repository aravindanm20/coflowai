"""The ``@tool`` decorator."""

from __future__ import annotations

import inspect
from typing import Any, Callable, TypeVar, overload

from pydantic import BaseModel

from .tool import Tool, ToolDefinition, build_input_model

__all__ = ["tool"]

F = TypeVar("F", bound=Callable[..., Any])


@overload
def tool(func: F) -> Tool: ...
@overload
def tool(*, name: str | None = ..., description: str | None = ...,
         timeout: float = ..., retries: int = ..., permission: str | None = ...,
         side_effect: bool = ..., idempotent: bool = ...,
         rate_limit_per_minute: int | None = ...,
         output_model: type[BaseModel] | None = ...,
         tags: set[str] | None = ...) -> Callable[[F], Tool]: ...


def tool(func: Callable[..., Any] | None = None, *,
         name: str | None = None,
         description: str | None = None,
         timeout: float = 30.0,
         retries: int = 0,
         permission: str | None = None,
         side_effect: bool = False,
         idempotent: bool = False,
         rate_limit_per_minute: int | None = None,
         output_model: type[BaseModel] | None = None,
         tags: set[str] | None = None) -> Tool | Callable[[Callable[..., Any]], Tool]:
    """Turn an (async) function into a :class:`Tool`.

    The input JSON schema is derived from type hints, so the model is told
    exactly what shape of arguments is acceptable — and the runtime validates
    them again before execution.
    """

    def decorate(target: Callable[..., Any]) -> Tool:
        tool_name = name or target.__name__
        input_model = build_input_model(target, tool_name)
        schema = input_model.model_json_schema() if input_model else {
            "type": "object", "properties": {}
        }
        schema.pop("title", None)

        definition = ToolDefinition(
            name=tool_name,
            description=(description
                         or inspect.cleandoc(target.__doc__ or "")
                         or f"Execute {tool_name}"),
            input_schema=schema,
            permission=permission,
            timeout=timeout,
            retries=retries,
            side_effect=side_effect,
            idempotent=idempotent,
            rate_limit_per_minute=rate_limit_per_minute,
            tags=set(tags or ()),
            output_schema=(output_model.model_json_schema() if output_model else None),
        )
        wrapper = Tool(definition=definition, func=target, input_model=input_model,
                       output_model=output_model)
        wrapper.__doc__ = target.__doc__
        return wrapper

    if func is not None:
        return decorate(func)
    return decorate
