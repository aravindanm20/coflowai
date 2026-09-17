"""Framework-level error model.

Rule: application code never sees provider SDK exceptions.  Provider errors are
always wrapped and exposed through ``__cause__`` / ``.cause``.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "CoFlowAiError",
    "ConfigurationError",
    "ExecutionError",
    "ExecutionCancelled",
    "ExecutionTimeout",
    "MaxStepsExceeded",
    "WorkflowError",
    "CompilationError",
    "AgentError",
    "ModelError",
    "ModelUnavailable",
    "RateLimitError",
    "ToolError",
    "ToolNotFound",
    "ToolTimeout",
    "PermissionDenied",
    "BudgetExceeded",
    "ValidationError",
    "ApprovalRejected",
    "CheckpointError",
    "PluginError",
    "is_retryable",
]


class CoFlowAiError(Exception):
    """Base class for every error raised by the framework."""

    retryable: bool = False

    def __init__(
        self,
        message: str,
        *,
        cause: BaseException | None = None,
        **details: Any,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.cause = cause
        self.details: dict[str, Any] = details
        if cause is not None:
            self.__cause__ = cause

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        if self.details:
            rendered = ", ".join(f"{k}={v!r}" for k, v in self.details.items())
            return f"{self.message} ({rendered})"
        return self.message

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": type(self).__name__,
            "message": self.message,
            "details": self.details,
            "cause": repr(self.cause) if self.cause else None,
        }


# --------------------------------------------------------------------------- #
# configuration / wiring
# --------------------------------------------------------------------------- #
class ConfigurationError(CoFlowAiError):
    """The framework was wired incorrectly (missing registry, model, ...)."""


class PluginError(CoFlowAiError):
    """A plugin failed to load or register."""


# --------------------------------------------------------------------------- #
# execution
# --------------------------------------------------------------------------- #
class ExecutionError(CoFlowAiError):
    """Generic failure inside the execution runtime."""


class ExecutionCancelled(ExecutionError):
    """Execution was cancelled by an operator or a parent execution."""


class ExecutionTimeout(ExecutionError):
    """A deadline (execution / agent / model / tool / node) expired."""

    retryable = False


class MaxStepsExceeded(ExecutionError):
    """Loop protection triggered: the step / iteration limit was reached."""


class BudgetExceeded(ExecutionError):
    """A token, cost or call budget would be exceeded by the next operation."""


# --------------------------------------------------------------------------- #
# workflow
# --------------------------------------------------------------------------- #
class WorkflowError(CoFlowAiError):
    """Failure while executing a workflow graph."""


class CompilationError(WorkflowError):
    """The workflow graph is invalid and was rejected before execution."""


# --------------------------------------------------------------------------- #
# agent / model
# --------------------------------------------------------------------------- #
class AgentError(CoFlowAiError):
    """Failure inside an agent's reasoning loop."""


class ModelError(CoFlowAiError):
    """A model provider failed."""


class ModelUnavailable(ModelError):
    retryable = True


class RateLimitError(ModelError):
    retryable = True


# --------------------------------------------------------------------------- #
# tools / security
# --------------------------------------------------------------------------- #
class ToolError(CoFlowAiError):
    """A tool failed to execute."""


class ToolNotFound(ToolError):
    """The model requested a tool that is not registered or not exposed."""


class ToolTimeout(ToolError):
    retryable = True


class PermissionDenied(CoFlowAiError):
    """The agent is not authorised to use the requested tool/resource."""


class ValidationError(CoFlowAiError):
    """Input/output schema validation failed."""

    retryable = True


class ApprovalRejected(CoFlowAiError):
    """A human rejected a pending approval."""


class CheckpointError(CoFlowAiError):
    """Checkpoint could not be created, loaded or is incompatible."""


def is_retryable(error: BaseException) -> bool:
    """Retry policy predicate: never retry permanent or policy errors."""
    if isinstance(error, (PermissionDenied, BudgetExceeded, MaxStepsExceeded,
                          ApprovalRejected, ExecutionCancelled, CompilationError,
                          ConfigurationError, ExecutionTimeout)):
        return False
    if isinstance(error, CoFlowAiError):
        if error.retryable:
            return True
        # a wrapped transport error stays retryable through the wrapper
        return error.cause is not None and is_retryable(error.cause)
    # Transport-ish errors from adapters are considered transient.
    return isinstance(error, (TimeoutError, ConnectionError, OSError))
