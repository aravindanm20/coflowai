"""Retry + timeout helpers used by the model gateway and tool runtime."""

from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable

from .exceptions import ExecutionTimeout, is_retryable
from ..policies.policy import RetryPolicy

__all__ = ["run_with_retry", "run_with_timeout"]

OnRetry = Callable[[int, BaseException, float], Awaitable[None] | None]


async def run_with_timeout(coro_factory: Callable[[], Awaitable[Any]],
                           timeout: float | None,
                           *, what: str = "operation",
                           **details: Any) -> Any:
    """Run a coroutine under a deadline, converting to ``ExecutionTimeout``."""
    if timeout is None:
        return await coro_factory()
    try:
        async with asyncio.timeout(timeout):
            return await coro_factory()
    except TimeoutError as exc:
        raise ExecutionTimeout(f"{what} timed out after {timeout}s",
                               cause=exc, timeout_seconds=timeout, **details) from exc


async def run_with_retry(operation: Callable[[int], Awaitable[Any]],
                         policy: RetryPolicy,
                         *,
                         retryable: Callable[[BaseException], bool] = is_retryable,
                         on_retry: OnRetry | None = None,
                         sleep: Callable[[float], Awaitable[None]] = asyncio.sleep
                         ) -> Any:
    """Execute ``operation(attempt)`` with backoff.

    Permanent errors (permissions, budgets, cancellation) are never retried.
    """
    attempt = 0
    last_error: BaseException
    while True:
        try:
            return await operation(attempt)
        except asyncio.CancelledError:
            raise
        except BaseException as exc:  # noqa: BLE001 - policy decides
            last_error = exc
            if attempt >= policy.attempts or not retryable(exc):
                raise
            attempt += 1
            delay = policy.delay_for(attempt)
            if on_retry is not None:
                result = on_retry(attempt, last_error, delay)
                if result is not None and hasattr(result, "__await__"):
                    await result
            await sleep(delay)
