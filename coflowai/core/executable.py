"""The single abstraction every runnable object implements."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Awaitable, Callable

from .context import ExecutionContext
from .ids import stable_hash
from .result import ExecutionResult
from .types import ExecutionStatus

__all__ = ["Executable", "FunctionExecutable", "as_executable"]


class Executable(ABC):
    """Agents, tools, workflows, routers and approvals are all Executables."""

    name: str = "executable"

    @abstractmethod
    async def execute(self, context: ExecutionContext) -> ExecutionResult:
        ...

    # -------------------------------------------------------------- metadata
    @property
    def required_permissions(self) -> list[str]:
        """Permissions the compiler must verify before execution starts."""
        return []

    @property
    def model_requirements(self) -> dict[str, bool]:
        """Model capabilities this executable needs (validated at compile time)."""
        return {}

    def signature(self) -> str:
        """Stable identity used for workflow hashing / version compatibility."""
        return stable_hash(type(self).__name__, self.name,
                           sorted(self.required_permissions),
                           sorted(self.model_requirements.items()))

    def describe(self) -> dict[str, Any]:
        return {
            "type": type(self).__name__,
            "name": self.name,
            "permissions": self.required_permissions,
            "model_requirements": self.model_requirements,
        }

    # ----------------------------------------------------------- convenience
    async def run(self, input: Any = None, *, policy=None, runtime=None,
                  **kwargs: Any) -> ExecutionResult:
        """Run through a runtime (a default in-memory one if not supplied)."""
        from ..runtime.runtime import Runtime  # local import avoids a cycle

        rt = runtime or Runtime.default()
        return await rt.run(self, input, policy=policy, **kwargs)


class FunctionExecutable(Executable):
    """Adapter turning a plain (async) callable into an Executable node."""

    def __init__(self, func: Callable[..., Any | Awaitable[Any]],
                 name: str | None = None) -> None:
        self.func = func
        self.name = name or getattr(func, "__name__", "function")

    async def execute(self, context: ExecutionContext) -> ExecutionResult:
        import inspect

        context.ensure_alive()
        result = self.func(context.input, context) if _takes_context(self.func) \
            else self.func(context.input)
        if inspect.isawaitable(result):
            result = await result
        return ExecutionResult(
            execution_id=context.execution_id,
            status=ExecutionStatus.COMPLETED,
            output=result,
            usage=context.usage_snapshot(),
            metadata={"node": self.name},
        )

    def signature(self) -> str:
        return stable_hash("function", self.name, getattr(self.func, "__qualname__", ""))


def _takes_context(func: Callable[..., Any]) -> bool:
    import inspect

    try:
        params = inspect.signature(func).parameters
    except (TypeError, ValueError):  # pragma: no cover - builtins
        return False
    return len(params) >= 2


def as_executable(obj: Any) -> Executable:
    """Coerce agents / workflows / callables into an Executable."""
    if isinstance(obj, Executable):
        return obj
    if callable(obj):
        return FunctionExecutable(obj)
    raise TypeError(f"{obj!r} is not executable")
