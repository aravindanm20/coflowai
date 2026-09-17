"""Model gateway.

Every model call in the framework goes through here so that budgets, retries,
timeouts, events, usage and cost accounting are impossible to bypass.
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from ..core.context import ExecutionContext
from ..core.exceptions import ModelError, is_retryable
from ..core.ids import model_call_id
from ..events.event import EventType
from ..core.retry import run_with_retry, run_with_timeout
from .base import ModelRequest, ModelResponse
from .registry import ModelRegistry, RegisteredModel
from .streaming import ChunkType, StreamAccumulator, StreamChunk

__all__ = ["ModelGateway", "MODEL_OVERRIDES_KEY"]

#: ``state[MODEL_OVERRIDES_KEY]`` maps a 1-based model-call number to canned text.
#: Used by ``runtime.fork(..., overrides={"model_responses": {...}})`` so a
#: developer can answer "what if the model had said *this* instead?".
MODEL_OVERRIDES_KEY = "__model_overrides__"


class ModelGateway:
    def __init__(self, registry: ModelRegistry) -> None:
        self.registry = registry

    async def call(self, context: ExecutionContext, request: ModelRequest, *,
                   model: str | None = None,
                   requirements: dict[str, bool] | None = None,
                   prefer: str = "priority") -> ModelResponse:
        await context.check_model_budget()

        override = self._override_for(context)
        if override is not None:
            return await self._apply_override(context, override)

        registered = self.registry.resolve(model=model, requirements=requirements,
                                           prefer=prefer)
        call_id = model_call_id()

        # STRICT mode forces deterministic sampling.
        override = context.policy.temperature_override()
        if override is not None:
            request.temperature = override
        request.model = registered.name
        request.timeout_seconds = request.timeout_seconds or _model_timeout(context)

        await context.emit(
            EventType.MODEL_REQUESTED,
            model=registered.name,
            model_call_id=call_id,
            agent=context.agent_name,
            request=request.to_dict(),
        )

        chain = [registered] + [
            self.registry.get(name) for name in registered.fallbacks
            if self.registry.has(name)
        ]
        last_error: BaseException | None = None

        for candidate in chain:
            request.model = candidate.name
            try:
                return await self._call_one(context, candidate, request, call_id)
            except Exception as exc:  # try the next provider only for transient faults
                last_error = exc
                if not is_retryable(exc) or candidate is chain[-1]:
                    raise
                await context.emit(
                    EventType.MODEL_FAILED,
                    model=candidate.name,
                    model_call_id=call_id,
                    error=repr(exc),
                    failover_to=chain[chain.index(candidate) + 1].name,
                )
        raise ModelError("model call failed", cause=last_error)  # pragma: no cover

    # ---------------------------------------------------------------- override
    @staticmethod
    def _override_for(context: ExecutionContext) -> Any:
        overrides = context.state.get(MODEL_OVERRIDES_KEY)
        if not overrides:
            return None
        return overrides.get(str(context.model_calls + 1)) \
            or overrides.get(context.model_calls + 1)

    async def _apply_override(self, context: ExecutionContext,
                              override: Any) -> ModelResponse:
        """Substitute a recorded/edited response without calling a provider.

        Usage is still counted (the call *happened* from the execution's point
        of view) but cost is zero, since no provider was billed.
        """
        if isinstance(override, ModelResponse):
            response = override
        elif isinstance(override, dict) and "text" in override:
            response = ModelResponse(text=override.get("text"),
                                     finish_reason=override.get("finish_reason",
                                                                "stop"))
        else:
            response = ModelResponse(text=str(override), finish_reason="stop")
        response.cost = 0.0
        response.model = response.model or "override"
        context.model_calls += 1
        await context.emit(EventType.MODEL_COMPLETED, model="override",
                           agent=context.agent_name, overridden=True,
                           response=response.to_dict())
        return response

    # ---------------------------------------------------------------- streaming
    async def stream(self, context: ExecutionContext, request: ModelRequest, *,
                     model: str | None = None,
                     requirements: dict[str, bool] | None = None,
                     prefer: str = "priority") -> AsyncIterator[StreamChunk]:
        """Stream a model response.

        Budgets are checked up-front and usage is accounted exactly once, when
        the stream terminates — so streaming never becomes a way to bypass the
        policy engine.  Streaming is not retried mid-flight: once bytes have
        been delivered to the caller, a silent restart would duplicate output.
        """
        await context.check_model_budget()
        registered = self.registry.resolve(model=model, requirements=requirements,
                                           prefer=prefer)
        call_id = model_call_id()

        override = context.policy.temperature_override()
        if override is not None:
            request.temperature = override
        request.model = registered.name
        request.timeout_seconds = request.timeout_seconds or _model_timeout(context)

        await context.emit(
            EventType.MODEL_REQUESTED, model=registered.name, model_call_id=call_id,
            agent=context.agent_name, streaming=True, request=request.to_dict(),
        )

        accumulator = StreamAccumulator()
        final_response: ModelResponse | None = None
        started = time.perf_counter()
        tracer = context.services.tracer if context.services else None
        span = tracer.span(f"model:{registered.name}", kind="model",
                           model=registered.name, call_id=call_id,
                           streaming=True) if tracer else _null_span()

        try:
            async with span:
                async for chunk in registered.provider.stream(request):
                    context.ensure_alive()
                    if chunk.is_final:
                        # adapters report authoritative token usage here
                        final_response = chunk.response
                        break
                    accumulator.add(chunk)
                    yield chunk
        except Exception as exc:
            await context.emit(EventType.MODEL_FAILED, model=registered.name,
                               model_call_id=call_id, error=repr(exc),
                               streaming=True)
            if isinstance(exc, ModelError) or type(exc).__module__.startswith(
                    "coflowai."):
                raise
            raise ModelError(f"provider '{registered.name}' failed while streaming",
                             cause=exc, model=registered.name) from exc

        # Prefer the adapter's own response (it carries real token counts);
        # fall back to the accumulated content if the adapter omitted it.
        if final_response is not None:
            response = final_response
            if response.text is None and accumulator.text:
                response.text = accumulator.text
        else:
            response = accumulator.build(model=registered.name)
        latency_ms = (time.perf_counter() - started) * 1000
        self._account(context, registered, response, latency_ms)
        await context.maybe_warn_budget()
        await context.emit(
            EventType.MODEL_COMPLETED, model=registered.name, model_call_id=call_id,
            agent=context.agent_name, duration_ms=round(latency_ms, 3),
            streaming=True, response=response.to_dict(),
        )
        yield StreamChunk(type=ChunkType.DONE, response=response)

    # ------------------------------------------------------------------ inner
    async def _call_one(self, context: ExecutionContext, registered: RegisteredModel,
                        request: ModelRequest, call_id: str) -> ModelResponse:
        started = time.perf_counter()

        async def attempt(attempt_number: int) -> ModelResponse:
            context.ensure_alive()
            try:
                return await run_with_timeout(
                    lambda: registered.provider.generate(request),
                    request.timeout_seconds,
                    what=f"model '{registered.name}'",
                    model=registered.name,
                )
            except Exception as exc:
                if isinstance(exc, ModelError) or type(exc).__module__.startswith(
                        "coflowai."):
                    raise
                # never leak provider SDK exceptions into application code
                raise ModelError(f"provider '{registered.name}' failed",
                                 cause=exc, model=registered.name,
                                 attempt=attempt_number) from exc

        async def on_retry(attempt_number: int, error: BaseException,
                           delay: float) -> None:
            await context.emit(
                EventType.MODEL_RETRYING,
                model=registered.name, model_call_id=call_id,
                attempt=attempt_number, delay=round(delay, 3), error=repr(error),
            )

        tracer = context.services.tracer if context.services else None
        try:
            if tracer is not None:
                async with tracer.span(f"model:{registered.name}", kind="model",
                                       model=registered.name, call_id=call_id):
                    response = await run_with_retry(attempt, context.policy.retry,
                                                    on_retry=on_retry)
            else:  # pragma: no cover - tracer is always present in practice
                response = await run_with_retry(attempt, context.policy.retry,
                                                on_retry=on_retry)
        except Exception as exc:
            await context.emit(EventType.MODEL_FAILED, model=registered.name,
                               model_call_id=call_id, error=repr(exc))
            raise

        latency_ms = (time.perf_counter() - started) * 1000
        self._account(context, registered, response, latency_ms)
        await context.maybe_warn_budget()

        await context.emit(
            EventType.MODEL_COMPLETED,
            model=registered.name,
            model_call_id=call_id,
            agent=context.agent_name,
            duration_ms=round(latency_ms, 3),
            response=response.to_dict(),
        )
        return response

    def _account(self, context: ExecutionContext, registered: RegisteredModel,
                 response: ModelResponse, latency_ms: float) -> None:
        cost = response.cost
        if cost is None:
            cost = registered.pricing.estimate(
                response.input_tokens, response.output_tokens, response.cached_tokens
            )
            response.cost = cost
        response.model = response.model or registered.name

        context.model_calls += 1
        context.input_tokens += response.input_tokens
        context.output_tokens += response.output_tokens
        context.cached_tokens += response.cached_tokens
        context.cost += cost

        if context.services is None:
            return
        context.services.usage.record_model(
            model=registered.name,
            agent=context.agent_name,
            node_id=context.node_id,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            cached_tokens=response.cached_tokens,
            cost=cost,
            latency_ms=latency_ms,
        )
        metrics = context.services.metrics
        metrics.increment("coflowai.model.calls", model=registered.name)
        metrics.observe("coflowai.model.latency_ms", latency_ms, model=registered.name)
        metrics.observe("coflowai.model.tokens",
                        response.input_tokens + response.output_tokens,
                        model=registered.name)
        metrics.observe("coflowai.model.cost", cost, model=registered.name)


def _model_timeout(context: ExecutionContext) -> float | None:
    """Model timeout, always clamped by the remaining execution deadline."""
    candidates: list[float] = []
    if context.policy.model_timeout_seconds:
        candidates.append(context.policy.model_timeout_seconds)
    remaining = context.remaining_seconds
    if remaining is not None:
        candidates.append(max(0.01, remaining))
    return min(candidates) if candidates else None


@asynccontextmanager
async def _null_span():  # pragma: no cover - only when no tracer is attached
    yield None
