"""Tool definition and the callable wrapper around a developer's function."""

from __future__ import annotations

import asyncio
import inspect
from dataclasses import dataclass, field
from typing import Any, Callable

from pydantic import BaseModel, ValidationError as PydanticValidationError
from pydantic import create_model

from ..core.exceptions import ValidationError

__all__ = ["ToolDefinition", "Tool", "build_input_model"]


@dataclass(slots=True)
class ToolDefinition:
    name: str
    description: str = ""
    input_schema: dict[str, Any] = field(default_factory=dict)
    permission: str | None = None
    timeout: float = 30.0
    retries: int = 0
    side_effect: bool = False
    idempotent: bool = False
    rate_limit_per_minute: int | None = None
    tags: set[str] = field(default_factory=set)
    output_schema: dict[str, Any] | None = None

    def to_model_format(self) -> dict[str, Any]:
        """Provider-neutral tool declaration handed to the model gateway."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.input_schema,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "permission": self.permission,
            "timeout": self.timeout,
            "retries": self.retries,
            "side_effect": self.side_effect,
            "idempotent": self.idempotent,
            "input_schema": self.input_schema,
        }


class Tool:
    """Executable unit of side effect.

    A ``Tool`` never runs itself in application code — the
    :class:`~coflowai.tools.executor.ToolExecutor` runs it after permission,
    validation, rate-limit, timeout and audit handling.
    """

    def __init__(self, definition: ToolDefinition,
                 func: Callable[..., Any],
                 *, input_model: type[BaseModel] | None = None,
                 output_model: type[BaseModel] | None = None,
                 validate_input: bool = True) -> None:
        self.definition = definition
        self.func = func
        self.input_model = input_model
        self.output_model = output_model
        self.validate_input = validate_input and input_model is not None
        self._is_async = asyncio.iscoroutinefunction(func)

    # ------------------------------------------------------------- properties
    @property
    def name(self) -> str:
        return self.definition.name

    @property
    def permission(self) -> str | None:
        return self.definition.permission

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"<Tool {self.name}>"

    # ------------------------------------------------------------- validation
    def validate_arguments(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Model output is untrusted: coerce/validate before execution."""
        if not isinstance(arguments, dict):
            raise ValidationError("tool arguments must be an object",
                                  tool=self.name, received=type(arguments).__name__)
        if not self.validate_input or self.input_model is None:
            return arguments
        try:
            model = self.input_model(**arguments)
        except PydanticValidationError as exc:
            raise ValidationError(
                f"invalid arguments for tool '{self.name}'",
                cause=exc, tool=self.name, errors=exc.errors(include_url=False),
            ) from exc
        return {k: getattr(model, k) for k in type(model).model_fields}

    def validate_output(self, output: Any) -> Any:
        if self.output_model is None:
            return output
        if isinstance(output, self.output_model):
            return output
        try:
            if isinstance(output, dict):
                return self.output_model(**output)
            return self.output_model.model_validate(output)
        except PydanticValidationError as exc:
            raise ValidationError(f"tool '{self.name}' returned invalid output",
                                  cause=exc, tool=self.name) from exc

    # ---------------------------------------------------------------- calling
    async def invoke(self, arguments: dict[str, Any]) -> Any:
        """Raw invocation.  Use ``ToolExecutor`` instead in application code."""
        if self._is_async:
            return await self.func(**arguments)
        return await asyncio.to_thread(self.func, **arguments)


def build_input_model(func: Callable[..., Any], name: str) -> type[BaseModel] | None:
    """Derive a Pydantic model (and therefore JSON schema) from type hints."""
    try:
        signature = inspect.signature(func)
        hints = inspect.get_annotations(func, eval_str=True)
    except Exception:  # pragma: no cover - exotic callables
        return None

    fields: dict[str, tuple[Any, Any]] = {}
    for param_name, param in signature.parameters.items():
        if param_name in ("self", "cls"):
            continue
        if param.kind in (inspect.Parameter.VAR_POSITIONAL,
                          inspect.Parameter.VAR_KEYWORD):
            continue
        annotation = hints.get(param_name, Any)
        default = ... if param.default is inspect.Parameter.empty else param.default
        fields[param_name] = (annotation, default)

    if not fields:
        return create_model(f"{name.title().replace('_', '')}Input")
    return create_model(f"{name.title().replace('_', '')}Input", **fields)
