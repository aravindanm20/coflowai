"""Execution results returned by every ``Executable`` and by the runtime."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .exceptions import CoFlowAiError
from .types import ExecutionStatus, Usage

__all__ = ["ExecutionResult"]


@dataclass
class ExecutionResult:
    """Uniform result envelope.

    ``output`` is the business value (string, dict or a validated Pydantic
    model).  ``status`` tells the caller whether it is safe to use.
    """

    execution_id: str
    status: ExecutionStatus = ExecutionStatus.COMPLETED
    output: Any = None
    error: BaseException | None = None
    usage: Usage = field(default_factory=Usage)
    metadata: dict[str, Any] = field(default_factory=dict)
    messages: list[Any] = field(default_factory=list)
    checkpoint_id: str | None = None

    # ------------------------------------------------------------- predicates
    @property
    def succeeded(self) -> bool:
        return self.status is ExecutionStatus.COMPLETED

    @property
    def paused(self) -> bool:
        return self.status in (ExecutionStatus.WAITING,
                               ExecutionStatus.WAITING_FOR_APPROVAL)

    def raise_for_status(self) -> "ExecutionResult":
        """Convert a failed result into an exception (opt-in)."""
        if self.succeeded:
            return self
        if self.error is not None:
            raise self.error
        raise CoFlowAiError(f"execution {self.status.value}",
                            execution_id=self.execution_id)

    def __bool__(self) -> bool:
        return self.succeeded

    def to_dict(self) -> dict[str, Any]:
        return {
            "execution_id": self.execution_id,
            "status": self.status.value,
            "output": _plain(self.output),
            "error": (self.error.to_dict() if isinstance(self.error, CoFlowAiError)
                      else (repr(self.error) if self.error else None)),
            "usage": self.usage.to_dict(),
            "metadata": self.metadata,
            "checkpoint_id": self.checkpoint_id,
        }


def _plain(value: Any) -> Any:
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        try:
            return dump()
        except Exception:  # pragma: no cover - defensive
            return repr(value)
    return value
