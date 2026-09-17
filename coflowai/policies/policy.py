"""Execution policies: the runtime's rulebook.

The LLM decides *what* to do; these objects decide whether it is *allowed to
happen*.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field, replace
from typing import Any

from ..core.exceptions import BudgetExceeded, MaxStepsExceeded
from ..core.types import ExecutionMode

__all__ = ["RetryPolicy", "Budget", "ExecutionPolicy"]


@dataclass(slots=True)
class RetryPolicy:
    """Backoff configuration.  Never retries permanent errors (see
    :func:`coflowai.core.exceptions.is_retryable`)."""

    attempts: int = 2
    backoff: str = "exponential"  # "exponential" | "linear" | "fixed"
    base_delay: float = 0.5
    max_delay: float = 30.0
    jitter: bool = True

    def delay_for(self, attempt: int) -> float:
        """Delay *before* retry number ``attempt`` (1-based)."""
        if self.backoff == "fixed":
            delay = self.base_delay
        elif self.backoff == "linear":
            delay = self.base_delay * attempt
        else:
            delay = self.base_delay * (2 ** (attempt - 1))
        delay = min(delay, self.max_delay)
        if self.jitter:
            delay *= 0.5 + random.random() / 2
        return delay

    @classmethod
    def none(cls) -> "RetryPolicy":
        return cls(attempts=0)


@dataclass(slots=True)
class Budget:
    """Hard limits tracked as part of execution state."""

    max_cost: float | None = None
    max_tokens: int | None = None
    max_model_calls: int | None = None
    max_tool_calls: int | None = None
    warn_at: float = 0.8  # emit budget.warning at 80% consumption

    def remaining(self, *, cost: float, tokens: int,
                  model_calls: int, tool_calls: int) -> dict[str, float | int | None]:
        return {
            "cost": None if self.max_cost is None else self.max_cost - cost,
            "tokens": None if self.max_tokens is None else self.max_tokens - tokens,
            "model_calls": (None if self.max_model_calls is None
                            else self.max_model_calls - model_calls),
            "tool_calls": (None if self.max_tool_calls is None
                           else self.max_tool_calls - tool_calls),
        }

    def utilisation(self, *, cost: float, tokens: int,
                    model_calls: int, tool_calls: int) -> float:
        ratios = []
        if self.max_cost:
            ratios.append(cost / self.max_cost)
        if self.max_tokens:
            ratios.append(tokens / self.max_tokens)
        if self.max_model_calls:
            ratios.append(model_calls / self.max_model_calls)
        if self.max_tool_calls:
            ratios.append(tool_calls / self.max_tool_calls)
        return max(ratios, default=0.0)


@dataclass(slots=True)
class ExecutionPolicy:
    """User-facing knob set for a single execution."""

    max_steps: int = 20
    timeout_seconds: float | None = 300.0
    max_model_calls: int = 20
    max_tool_calls: int = 30
    max_tokens: int | None = None
    max_cost: float | None = None
    retry_attempts: int = 2
    checkpoint_enabled: bool = True
    max_concurrency: int = 5
    mode: ExecutionMode = ExecutionMode.STANDARD
    #: how many times a structured-output failure may be repaired
    structured_output_repair_attempts: int = 2
    model_timeout_seconds: float | None = 60.0
    tool_timeout_seconds: float | None = 30.0
    retry: RetryPolicy = field(default_factory=RetryPolicy)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive; unbounded loops are forbidden")
        # keep `retry_attempts` and `retry.attempts` consistent
        if self.retry.attempts != self.retry_attempts:
            self.retry = replace(self.retry, attempts=self.retry_attempts)

    # ---------------------------------------------------------------- budgets
    @property
    def budget(self) -> Budget:
        return Budget(
            max_cost=self.max_cost,
            max_tokens=self.max_tokens,
            max_model_calls=self.max_model_calls,
            max_tool_calls=self.max_tool_calls,
        )

    def temperature_override(self) -> float | None:
        """STRICT pins sampling to 0; EXPLORATORY leaves the agent's own value."""
        if self.mode is ExecutionMode.STRICT:
            return 0.0
        return None

    def iteration_allowance(self, requested: int | None) -> int:
        """How many reasoning iterations an agent may use in this mode.

        STRICT       — tight: agents get the smaller of their own cap and 5
        STANDARD     — the agent's own cap, bounded by max_steps
        EXPLORATORY  — more autonomy: up to max_steps even if the agent asked
                       for fewer, since exploration needs room to backtrack
        """
        ceiling = self.max_steps
        if self.mode is ExecutionMode.EXPLORATORY:
            return max(1, min(ceiling, max(requested or 0, ceiling)))
        if self.mode is ExecutionMode.STRICT:
            return max(1, min(ceiling, requested or ceiling, 5))
        return max(1, min(ceiling, requested or ceiling))

    @property
    def allows_parallel_tool_calls(self) -> bool:
        """STRICT executes tools one at a time for reproducible ordering."""
        return self.mode is not ExecutionMode.STRICT

    # Checks are performed *before* each operation, never after the money is
    # already spent.
    def check_step(self, step: int) -> None:
        if step > self.max_steps:
            raise MaxStepsExceeded(
                "max_steps exceeded", max_steps=self.max_steps, step=step
            )

    def check_model_call(self, *, model_calls: int, tokens: int, cost: float) -> None:
        if model_calls >= self.max_model_calls:
            raise BudgetExceeded("max_model_calls exceeded",
                                 limit=self.max_model_calls, used=model_calls)
        self._check_shared(tokens=tokens, cost=cost)

    def check_tool_call(self, *, tool_calls: int, tokens: int, cost: float) -> None:
        if tool_calls >= self.max_tool_calls:
            raise BudgetExceeded("max_tool_calls exceeded",
                                 limit=self.max_tool_calls, used=tool_calls)
        self._check_shared(tokens=tokens, cost=cost)

    def _check_shared(self, *, tokens: int, cost: float) -> None:
        if self.max_tokens is not None and tokens >= self.max_tokens:
            raise BudgetExceeded("max_tokens exceeded",
                                 limit=self.max_tokens, used=tokens)
        if self.max_cost is not None and cost >= self.max_cost:
            raise BudgetExceeded("max_cost exceeded",
                                 limit=self.max_cost, used=round(cost, 6))

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_steps": self.max_steps,
            "timeout_seconds": self.timeout_seconds,
            "max_model_calls": self.max_model_calls,
            "max_tool_calls": self.max_tool_calls,
            "max_tokens": self.max_tokens,
            "max_cost": self.max_cost,
            "retry_attempts": self.retry_attempts,
            "checkpoint_enabled": self.checkpoint_enabled,
            "max_concurrency": self.max_concurrency,
            "mode": self.mode.value,
        }
