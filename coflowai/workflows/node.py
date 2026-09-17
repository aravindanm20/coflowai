"""Workflow node wrappers: Sequence, Parallel, Router, Loop, Transform."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Iterable, Sequence

from ..core.context import ExecutionContext
from ..core.exceptions import WorkflowError
from ..core.executable import Executable, as_executable
from ..core.ids import stable_hash
from ..core.result import ExecutionResult
from ..core.types import ExecutionStatus

__all__ = ["NodeKind", "Node", "SequenceNode", "ParallelNode", "RouterNode",
           "LoopNode", "TransformNode"]


class NodeKind(str, Enum):
    TASK = "task"
    PARALLEL = "parallel"
    ROUTER = "router"
    LOOP = "loop"
    APPROVAL = "approval"
    SUBWORKFLOW = "subworkflow"
    TRANSFORM = "transform"


@dataclass(slots=True)
class Node:
    """A compiled graph node."""

    id: str
    name: str
    executable: Executable
    kind: NodeKind = NodeKind.TASK
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def required_permissions(self) -> list[str]:
        return self.executable.required_permissions

    @property
    def model_requirements(self) -> dict[str, bool]:
        return self.executable.model_requirements

    def signature(self) -> str:
        return stable_hash(self.id, self.kind.value, self.executable.signature())

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "kind": self.kind.value,
            "executable": self.executable.describe(),
        }


class SequenceNode(Executable):
    """Run executables one after another, piping output into input."""

    def __init__(self, *steps: Any, name: str = "sequence") -> None:
        self.name = name
        self.steps = [as_executable(step) for step in steps]

    @property
    def required_permissions(self) -> list[str]:
        return sorted({p for s in self.steps for p in s.required_permissions})

    @property
    def model_requirements(self) -> dict[str, bool]:
        merged: dict[str, bool] = {}
        for step in self.steps:
            merged.update(step.model_requirements)
        return merged

    def signature(self) -> str:
        return stable_hash("sequence", self.name,
                           [s.signature() for s in self.steps])

    async def execute(self, context: ExecutionContext) -> ExecutionResult:
        current = context.input
        result = ExecutionResult(execution_id=context.execution_id, output=current)
        for step in self.steps:
            context.ensure_alive()
            child = context.child(input=current, node_id=step.name)
            result = await step.execute(child)
            context.merge_from(child)
            if not result.succeeded:
                return result
            current = result.output
        result.output = current
        return result


class ParallelNode(Executable):
    """Fan out to several branches with bounded concurrency (§37)."""

    def __init__(self, *branches: Any, name: str = "parallel",
                 merge: Callable[[dict[str, Any]], Any] | None = None,
                 max_concurrency: int | None = None,
                 fail_fast: bool = True) -> None:
        self.name = name
        self.branches = [as_executable(b) for b in branches]
        if not self.branches:
            raise WorkflowError("parallel node requires at least one branch")
        self.merge = merge
        self.max_concurrency = max_concurrency
        self.fail_fast = fail_fast

    @property
    def required_permissions(self) -> list[str]:
        return sorted({p for b in self.branches for p in b.required_permissions})

    @property
    def model_requirements(self) -> dict[str, bool]:
        merged: dict[str, bool] = {}
        for branch in self.branches:
            merged.update(branch.model_requirements)
        return merged

    def signature(self) -> str:
        return stable_hash("parallel", self.name,
                           sorted(b.signature() for b in self.branches))

    async def execute(self, context: ExecutionContext) -> ExecutionResult:
        context.ensure_alive()
        limit = self.max_concurrency or context.policy.max_concurrency
        semaphore = asyncio.Semaphore(max(1, limit))
        baseline = context.child()
        children: list[ExecutionContext] = []

        async def run(branch: Executable) -> tuple[str, ExecutionResult,
                                                   ExecutionContext]:
            async with semaphore:
                child = context.child(node_id=f"{self.name}.{branch.name}")
                children.append(child)
                return branch.name, await branch.execute(child), child

        outputs: dict[str, Any] = {}
        failures: dict[str, BaseException] = {}

        if self.fail_fast:
            async with asyncio.TaskGroup() as group:
                tasks = [group.create_task(run(b)) for b in self.branches]
            pairs = [t.result() for t in tasks]
        else:
            pairs = []
            gathered = await asyncio.gather(*(run(b) for b in self.branches),
                                            return_exceptions=True)
            for branch, item in zip(self.branches, gathered):
                if isinstance(item, BaseException):
                    failures[branch.name] = item
                else:
                    pairs.append(item)

        for name, result, _child in pairs:
            if result.succeeded:
                outputs[name] = result.output
            else:
                failures[name] = result.error or WorkflowError(
                    f"branch '{name}' ended as {result.status.value}")

        context.absorb_parallel(children, baseline)

        if failures and self.fail_fast:
            first = next(iter(failures.values()))
            raise first if isinstance(first, BaseException) else WorkflowError(
                "parallel branch failed")

        output = self.merge(outputs) if self.merge else outputs
        return ExecutionResult(
            execution_id=context.execution_id,
            status=ExecutionStatus.COMPLETED,
            output=output,
            usage=context.usage_snapshot(),
            metadata={"node": self.name, "branches": list(outputs),
                      "failed_branches": sorted(failures)},
        )


class RouterNode(Executable):
    """Deterministic (or agent-assisted) routing between branches (§34)."""

    def __init__(self, condition: Callable[..., Any],
                 routes: dict[str, Any], *, name: str = "router",
                 default: Any = None) -> None:
        self.name = name
        self.condition = condition
        self.routes = {key: as_executable(value) for key, value in routes.items()}
        self.default = as_executable(default) if default is not None else None

    @property
    def required_permissions(self) -> list[str]:
        targets = list(self.routes.values()) + ([self.default] if self.default else [])
        return sorted({p for t in targets for p in t.required_permissions})

    @property
    def model_requirements(self) -> dict[str, bool]:
        merged: dict[str, bool] = {}
        for target in list(self.routes.values()) + (
                [self.default] if self.default else []):
            merged.update(target.model_requirements)
        return merged

    def signature(self) -> str:
        return stable_hash("router", self.name, sorted(self.routes),
                           getattr(self.condition, "__qualname__", "condition"))

    async def execute(self, context: ExecutionContext) -> ExecutionResult:
        context.ensure_alive()
        decision = self.condition(context.input, context) \
            if _takes_two(self.condition) else self.condition(context.input)
        if hasattr(decision, "__await__"):
            decision = await decision

        # Rule 2: routing decisions are validated, never trusted blindly.
        key = decision.output if isinstance(decision, ExecutionResult) else decision
        key = str(key).strip() if key is not None else ""

        target = self.routes.get(key, self.default)
        if target is None:
            raise WorkflowError(
                f"router '{self.name}' produced an unknown route",
                route=key, known=sorted(self.routes),
            )
        child = context.child(node_id=f"{self.name}.{key or 'default'}")
        result = await target.execute(child)
        context.merge_from(child)
        result.metadata.setdefault("route", key)
        return result


class LoopNode(Executable):
    """Bounded loop (Rule 1: ``max_iterations`` is mandatory)."""

    def __init__(self, body: Any, *, until: Callable[..., Any] | None = None,
                 max_iterations: int = 3, name: str = "loop") -> None:
        if max_iterations <= 0:
            raise WorkflowError("loop requires a positive max_iterations")
        self.name = name
        self.body = as_executable(body)
        self.until = until
        self.max_iterations = max_iterations

    @property
    def required_permissions(self) -> list[str]:
        return self.body.required_permissions

    @property
    def model_requirements(self) -> dict[str, bool]:
        return self.body.model_requirements

    def signature(self) -> str:
        return stable_hash("loop", self.name, self.body.signature(),
                           self.max_iterations)

    async def execute(self, context: ExecutionContext) -> ExecutionResult:
        current = context.input
        result = ExecutionResult(execution_id=context.execution_id, output=current)
        iterations = 0
        for iteration in range(1, self.max_iterations + 1):
            context.ensure_alive()
            iterations = iteration
            child = context.child(input=current,
                                  node_id=f"{self.name}#{iteration}")
            result = await self.body.execute(child)
            context.merge_from(child)
            if not result.succeeded:
                return result
            current = result.output
            if self.until is not None:
                done = self.until(current, context) if _takes_two(self.until) \
                    else self.until(current)
                if hasattr(done, "__await__"):
                    done = await done
                if bool(done):
                    break
        result.output = current
        result.metadata["iterations"] = iterations
        result.metadata["max_iterations"] = self.max_iterations
        return result


class TransformNode(Executable):
    """Deterministic Python step — cheaper and safer than asking an LLM."""

    def __init__(self, func: Callable[..., Any], name: str | None = None) -> None:
        self.func = func
        self.name = name or getattr(func, "__name__", "transform")

    async def execute(self, context: ExecutionContext) -> ExecutionResult:
        context.ensure_alive()
        output = self.func(context.input, context) if _takes_two(self.func) \
            else self.func(context.input)
        if hasattr(output, "__await__"):
            output = await output
        return ExecutionResult(execution_id=context.execution_id,
                               status=ExecutionStatus.COMPLETED, output=output,
                               usage=context.usage_snapshot(),
                               metadata={"node": self.name})


def _takes_two(func: Callable[..., Any]) -> bool:
    import inspect

    try:
        return len(inspect.signature(func).parameters) >= 2
    except (TypeError, ValueError):  # pragma: no cover
        return False


def coerce_all(items: Iterable[Any]) -> Sequence[Executable]:  # pragma: no cover
    return [as_executable(item) for item in items]
