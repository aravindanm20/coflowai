"""The Agent: an LLM reasoning loop that the runtime keeps on a leash."""

from __future__ import annotations

import asyncio
import json
from typing import Any, AsyncIterator, Iterable, Sequence

from pydantic import BaseModel
from pydantic import ValidationError as PydanticValidationError

from ..context.builder import ContextBudget, ContextBuilder
from ..core.context import ExecutionContext
from ..core.exceptions import (
    ConfigurationError,
    MaxStepsExceeded,
    MaxStepsExceeded,
    ValidationError,
)
from ..core.executable import Executable
from ..core.ids import stable_hash
from ..core.result import ExecutionResult
from ..core.types import ExecutionStatus, Message, ToolCall, ToolResult
from ..events.event import EventType
from ..memory.base import MemoryKind
from ..models.base import ModelRequest
from ..models.streaming import ChunkType, StreamChunk
from ..policies.permissions import PermissionSet
from ..tools.tool import Tool

__all__ = ["Agent"]

_CONVERSATION_STATE_PREFIX = "__conversation__"


class Agent(Executable):
    """Reasoning unit.

    The agent decides *what* it wants (tools, answers); the runtime decides
    whether that is allowed, affordable and in time.
    """

    def __init__(
        self,
        name: str = "agent",
        instructions: str = "You are a helpful assistant.",
        model: str | None = None,
        tools: Sequence[Tool] | None = None,
        *,
        output_schema: type[BaseModel] | None = None,
        permissions: Iterable[str] | None = None,
        model_requirements: dict[str, bool] | None = None,
        model_preference: str = "priority",
        max_iterations: int | None = None,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
        context_budget: ContextBudget | None = None,
        summariser: Any = None,
        memory_query: str | None = None,
        memory_limit: int = 5,
        input_formatter: Any = None,
        description: str = "",
    ) -> None:
        if model is None and not model_requirements:
            model_requirements = {}
        self.name = name
        self.instructions = instructions
        self.model = model
        self.tools: list[Tool] = list(tools or ())
        self.output_schema = output_schema
        self.permissions = None if permissions is None else PermissionSet(permissions)
        self._model_requirements = dict(model_requirements or {})
        self.model_preference = model_preference
        self.max_iterations = max_iterations
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.context_budget = context_budget or ContextBudget()
        self.summariser = summariser
        self.memory_query = memory_query
        self.memory_limit = memory_limit
        self.input_formatter = input_formatter
        self.description = description or f"Agent '{name}'"

    # ------------------------------------------------------------- metadata
    @property
    def required_permissions(self) -> list[str]:
        required = {t.definition.permission for t in self.tools
                    if t.definition.permission}
        return sorted(required)

    @property
    def model_requirements(self) -> dict[str, bool]:
        requirements = dict(self._model_requirements)
        if self.tools:
            requirements.setdefault("tool_calling", True)
        if self.output_schema is not None:
            requirements.setdefault("structured_output", True)
        return requirements

    def signature(self) -> str:
        return stable_hash(
            "agent", self.name, self.instructions, self.model,
            sorted(t.name for t in self.tools),
            self.output_schema.__name__ if self.output_schema else None,
        )

    def describe(self) -> dict[str, Any]:
        return {
            **super().describe(),
            "model": self.model,
            "tools": [t.name for t in self.tools],
            "output_schema": (self.output_schema.__name__
                              if self.output_schema else None),
            "max_iterations": self.max_iterations,
        }

    # ------------------------------------------------------------- execution
    async def execute(self, context: ExecutionContext) -> ExecutionResult:
        services = context.services
        if services is None:
            raise ConfigurationError(
                "agents must be executed through a Runtime (services missing)")

        ctx = context.child(agent_name=self.name, permissions=self.permissions)
        ctx.ensure_alive()

        for tool in self.tools:
            if not services.tools.has(tool.name):
                services.tools.register(tool)

        # Exposure is permission-filtered (prompt-injection defence): the model
        # is only *shown* tools it may use.  Authorisation is still enforced
        # independently by the tool runtime, so a hallucinated name is denied.
        visible = services.tools.visible_to(
            ctx.permissions, [t.name for t in self.tools] if self.tools else []
        )
        allowed_names = {t.name for t in self.tools}

        await ctx.emit(EventType.AGENT_STARTED, agent=self.name,
                       tools=sorted(allowed_names),
                       model=self.model or "capability-routed")

        conversation = self._load_conversation(ctx)
        repairs_left = ctx.policy.structured_output_repair_attempts
        max_iterations = self._iteration_limit(ctx)

        try:
            async with services.tracer.span(f"agent:{self.name}", kind="agent",
                                            agent=self.name):
                for iteration in range(1, max_iterations + 1):
                    ctx.next_step()
                    context.step = ctx.step

                    messages = await self._build_context(ctx, conversation)
                    request = ModelRequest(
                        messages=messages,
                        tools=[t.definition.to_model_format() for t in visible],
                        temperature=self.temperature,
                        max_output_tokens=self.max_output_tokens,
                        response_schema=self._response_schema(),
                        metadata={"agent": self.name, "iteration": iteration},
                    )
                    response = await services.gateway.call(
                        ctx, request, model=self.model,
                        requirements=self.model_requirements or None,
                        prefer=self.model_preference,
                    )
                    conversation.append(
                        Message.assistant(response.text, response.tool_calls)
                    )
                    self._save_conversation(ctx, conversation)

                    if response.tool_calls:
                        results = await self._run_tools(ctx, response.tool_calls,
                                                        allowed_names)
                        for result in results:
                            conversation.append(Message.tool(
                                result.call_id, result.name,
                                _render_tool_result(result),
                            ))
                        self._save_conversation(ctx, conversation)
                        continue

                    try:
                        output = self._finalise_output(response.text)
                    except ValidationError as exc:
                        if repairs_left <= 0:
                            raise
                        repairs_left -= 1
                        conversation.append(Message.user(_repair_prompt(exc)))
                        self._save_conversation(ctx, conversation)
                        continue

                    context.merge_from(ctx)
                    await ctx.emit(EventType.AGENT_COMPLETED, agent=self.name,
                                   iterations=iteration,
                                   status=ExecutionStatus.COMPLETED.value)
                    return ExecutionResult(
                        execution_id=ctx.execution_id,
                        status=ExecutionStatus.COMPLETED,
                        output=output,
                        usage=ctx.usage_snapshot(),
                        messages=list(conversation),
                        metadata={"agent": self.name, "iterations": iteration,
                                  "node_id": ctx.node_id},
                    )

            raise MaxStepsExceeded(
                f"agent '{self.name}' did not converge",
                agent=self.name, max_iterations=max_iterations,
            )
        except Exception as exc:
            context.merge_from(ctx)
            await ctx.emit(EventType.AGENT_FAILED, agent=self.name, error=repr(exc))
            raise

    # ------------------------------------------------------------- streaming
    async def execute_stream(self, context: ExecutionContext
                             ) -> AsyncIterator[StreamChunk]:
        """Run the agent, streaming assistant text as it is produced.

        Tool calls still execute through the full tool runtime between
        iterations; only the model's textual output is streamed.  The terminal
        ``DONE`` chunk carries the finished :class:`ExecutionResult`, so callers
        get both the live experience and the validated result.
        """
        services = context.services
        if services is None:
            raise ConfigurationError(
                "agents must be executed through a Runtime (services missing)")

        ctx = context.child(agent_name=self.name, permissions=self.permissions)
        ctx.ensure_alive()
        for tool in self.tools:
            if not services.tools.has(tool.name):
                services.tools.register(tool)
        visible = services.tools.visible_to(
            ctx.permissions, [t.name for t in self.tools] if self.tools else [])
        allowed_names = {t.name for t in self.tools}

        await ctx.emit(EventType.AGENT_STARTED, agent=self.name, streaming=True,
                       tools=sorted(allowed_names))

        conversation = self._load_conversation(ctx)
        repairs_left = ctx.policy.structured_output_repair_attempts
        max_iterations = self._iteration_limit(ctx)

        for iteration in range(1, max_iterations + 1):
            ctx.next_step()
            context.step = ctx.step
            messages = await self._build_context(ctx, conversation)
            request = ModelRequest(
                messages=messages,
                tools=[t.definition.to_model_format() for t in visible],
                temperature=self.temperature,
                max_output_tokens=self.max_output_tokens,
                response_schema=self._response_schema(),
                metadata={"agent": self.name, "iteration": iteration},
            )

            response: Any = None
            async for chunk in services.gateway.stream(
                    ctx, request, model=self.model,
                    requirements=self.model_requirements or None,
                    prefer=self.model_preference):
                if chunk.is_final:
                    response = chunk.response
                    break
                # structured output is validated as a whole; don't stream partial JSON
                if self.output_schema is None:
                    yield chunk

            conversation.append(Message.assistant(response.text,
                                                  response.tool_calls))
            self._save_conversation(ctx, conversation)

            if response.tool_calls:
                results = await self._run_tools(ctx, response.tool_calls,
                                                allowed_names)
                for result in results:
                    conversation.append(Message.tool(result.call_id, result.name,
                                                     _render_tool_result(result)))
                self._save_conversation(ctx, conversation)
                continue

            try:
                output = self._finalise_output(response.text)
            except ValidationError:
                if repairs_left <= 0:
                    raise
                repairs_left -= 1
                conversation.append(Message.user(_repair_prompt(
                    ValidationError("schema validation failed"))))
                self._save_conversation(ctx, conversation)
                continue

            context.merge_from(ctx)
            await ctx.emit(EventType.AGENT_COMPLETED, agent=self.name,
                           iterations=iteration, streaming=True)
            result = ExecutionResult(
                execution_id=ctx.execution_id,
                status=ExecutionStatus.COMPLETED,
                output=output,
                usage=ctx.usage_snapshot(),
                messages=list(conversation),
                metadata={"agent": self.name, "iterations": iteration},
            )
            yield StreamChunk(type=ChunkType.DONE, metadata={"result": result})
            return

        context.merge_from(ctx)
        raise MaxStepsExceeded(f"agent '{self.name}' did not converge",
                               agent=self.name, max_iterations=max_iterations)

    # --------------------------------------------------------------- helpers
    def _iteration_limit(self, ctx: ExecutionContext) -> int:
        """Rule 1: there is always a bound — the mode only tunes how tight."""
        return ctx.policy.iteration_allowance(self.max_iterations)

    async def _build_context(self, ctx: ExecutionContext,
                             conversation: list[Message]) -> list[Message]:
        builder = ContextBuilder(self.context_budget,
                                 summariser=self.summariser)
        builder.instructions(self._system_prompt())
        builder.workflow_state(ctx.state)

        memory = ctx.services.memory if ctx.services else None
        if memory is not None:
            query = self.memory_query or _as_text(ctx.input)
            try:
                records = await memory.search(query, limit=self.memory_limit,
                                              kind=MemoryKind.LONG_TERM)
            except Exception:  # pragma: no cover - memory must never break a run
                records = []
            if records:
                builder.memory([r.as_text() for r in records])

        builder.conversation(conversation)
        return await builder.build()

    def _system_prompt(self) -> str:
        prompt = self.instructions
        if self.output_schema is not None:
            schema = json.dumps(self.output_schema.model_json_schema(), indent=2)
            prompt += (
                "\n\nRespond with a single JSON object that validates against "
                f"this JSON schema. Do not add commentary.\n{schema}"
            )
        return prompt

    def _response_schema(self) -> dict[str, Any] | None:
        if self.output_schema is None:
            return None
        return self.output_schema.model_json_schema()

    # ------------------------------------------------------------- tool loop
    async def _run_tools(self, ctx: ExecutionContext, calls: list[ToolCall],
                         allowed: set[str]) -> list[ToolResult]:
        executor = ctx.services.tool_executor
        limit = max(1, ctx.policy.max_concurrency)

        # STRICT mode serialises tool calls so ordering is reproducible.
        if len(calls) == 1 or not ctx.policy.allows_parallel_tool_calls:
            if len(calls) > 1:
                return [await executor.execute(ctx, call, allowed_tools=allowed)
                        for call in calls]
            return [await executor.execute(ctx, calls[0], allowed_tools=allowed)]


        semaphore = asyncio.Semaphore(limit)

        async def run_one(call: ToolCall) -> ToolResult:
            async with semaphore:
                return await executor.execute(ctx, call, allowed_tools=allowed)

        results: list[ToolResult] = []
        async with asyncio.TaskGroup() as group:
            tasks = [group.create_task(run_one(call)) for call in calls]
        for task in tasks:
            results.append(task.result())
        return results

    # -------------------------------------------------------- output handling
    def _finalise_output(self, text: str | None) -> Any:
        if self.output_schema is None:
            return text
        if not text:
            raise ValidationError("model returned no content for structured output",
                                  agent=self.name)
        payload = _extract_json(text)
        try:
            return self.output_schema.model_validate(payload)
        except PydanticValidationError as exc:
            raise ValidationError(
                f"structured output failed validation for '{self.name}'",
                cause=exc, agent=self.name,
                errors=exc.errors(include_url=False),
            ) from exc

    # --------------------------------------------------------- conversation IO
    def _state_key(self, ctx: ExecutionContext) -> str:
        return f"{_CONVERSATION_STATE_PREFIX}{ctx.node_id or self.name}"

    def _load_conversation(self, ctx: ExecutionContext) -> list[Message]:
        stored = ctx.state.get(self._state_key(ctx))
        if stored:
            return [Message.from_dict(m) for m in stored]
        return [Message.user(self._format_input(ctx.input))]

    def _save_conversation(self, ctx: ExecutionContext,
                           conversation: list[Message]) -> None:
        ctx.state[self._state_key(ctx)] = [m.to_dict() for m in conversation]

    def _format_input(self, value: Any) -> str:
        if self.input_formatter is not None:
            return self.input_formatter(value)
        return _as_text(value)


# ----------------------------------------------------------------- utilities
def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, BaseModel):
        return value.model_dump_json(indent=2)
    if isinstance(value, ExecutionResult):  # pragma: no cover - defensive
        return _as_text(value.output)
    try:
        return json.dumps(value, indent=2, default=str)
    except TypeError:  # pragma: no cover - exotic objects
        return str(value)


def _render_tool_result(result: ToolResult) -> str:
    if not result.ok:
        return json.dumps({"error": result.error})
    return _as_text(result.output)


def _repair_prompt(error: ValidationError) -> str:
    return (
        "Your previous response was rejected by schema validation: "
        f"{error}. Reply with corrected JSON only — no prose, no code fences."
    )


def _extract_json(text: str) -> Any:
    """Tolerate fenced/decorated JSON, then validate strictly."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```")[1]
        if cleaned.lstrip().lower().startswith("json"):
            cleaned = cleaned.lstrip()[4:]
        cleaned = cleaned.strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start != -1 and end > start:
            try:
                return json.loads(cleaned[start:end + 1])
            except json.JSONDecodeError as exc:
                raise ValidationError("model output is not valid JSON",
                                      cause=exc) from exc
        raise ValidationError("model output is not valid JSON")
