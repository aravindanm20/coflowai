"""Workflow: the developer-facing orchestration API."""

from __future__ import annotations

from typing import Any, Callable, Iterable

from ..approvals.approval import HumanApproval
from ..core.context import ExecutionContext
from ..core.exceptions import CheckpointError, WorkflowError
from ..core.executable import Executable, as_executable
from ..core.ids import stable_hash
from ..core.result import ExecutionResult
from ..core.types import ExecutionStatus
from ..events.event import EventType
from ..models.registry import ModelRegistry
from ..policies.permissions import PermissionSet
from .compiler import WorkflowCompiler
from .graph import ExecutionGraph
from .node import (
    LoopNode,
    Node,
    NodeKind,
    ParallelNode,
    RouterNode,
    TransformNode,
)

__all__ = ["Workflow", "COMPLETED_NODES_KEY", "WORKFLOW_HASH_KEY"]

COMPLETED_NODES_KEY = "__completed_nodes__"
LAST_OUTPUT_KEY = "__last_output__"
WORKFLOW_HASH_KEY = "__workflow_hash__"


class Workflow(Executable):
    """Fluent builder that compiles to a validated execution graph.

    ``Workflow`` is itself an :class:`Executable`, so workflows nest.
    """

    def __init__(self, name: str, *, version: str = "1",
                 permissions: Iterable[str] | None = None,
                 description: str = "") -> None:
        self.name = name
        self.version = version
        self.description = description
        self.permissions = None if permissions is None else PermissionSet(permissions)
        self._nodes: list[Node] = []
        self._graph: ExecutionGraph | None = None

    # ------------------------------------------------------------- building
    def _add(self, executable: Any, kind: NodeKind,
             name: str | None = None, **metadata: Any) -> "Workflow":
        node_executable = as_executable(executable)
        node_name = name or node_executable.name
        node_id = self._unique_id(node_name)
        self._nodes.append(Node(id=node_id, name=node_name,
                                executable=node_executable, kind=kind,
                                metadata=metadata))
        self._graph = None  # invalidate previous compilation
        return self

    def _unique_id(self, name: str) -> str:
        base = "".join(ch if ch.isalnum() else "_" for ch in name).strip("_").lower()
        base = base or "node"
        existing = {n.id for n in self._nodes}
        if base not in existing:
            return base
        index = 2
        while f"{base}_{index}" in existing:
            index += 1
        return f"{base}_{index}"

    def start(self, executable: Any, *, name: str | None = None) -> "Workflow":
        if self._nodes:
            raise WorkflowError(f"workflow '{self.name}' already has a start node")
        return self._add(executable, NodeKind.TASK, name)

    def then(self, executable: Any, *, name: str | None = None) -> "Workflow":
        if not self._nodes:
            return self.start(executable, name=name)
        return self._add(executable, NodeKind.TASK, name)

    def transform(self, func: Callable[..., Any], *,
                  name: str | None = None) -> "Workflow":
        return self._add(TransformNode(func, name=name), NodeKind.TRANSFORM, name)

    def parallel(self, *branches: Any, name: str = "parallel",
                 merge: Callable[[dict[str, Any]], Any] | None = None,
                 max_concurrency: int | None = None,
                 fail_fast: bool = True) -> "Workflow":
        node = ParallelNode(*branches, name=name, merge=merge,
                            max_concurrency=max_concurrency, fail_fast=fail_fast)
        return self._add(node, NodeKind.PARALLEL, name,
                         branches=[b.name for b in node.branches])

    def route(self, condition: Callable[..., Any], routes: dict[str, Any], *,
              name: str = "router", default: Any = None) -> "Workflow":
        node = RouterNode(condition, routes, name=name, default=default)
        return self._add(node, NodeKind.ROUTER, name, routes=sorted(routes))

    def loop(self, body: Any, *, until: Callable[..., Any] | None = None,
             max_iterations: int = 3, name: str = "loop") -> "Workflow":
        node = LoopNode(body, until=until, max_iterations=max_iterations, name=name)
        return self._add(node, NodeKind.LOOP, name, max_iterations=max_iterations)

    def approve(self, message: str, *, action: Any = None,
                name: str = "approval", on_reject: str = "fail") -> "Workflow":
        node = HumanApproval(message, action=action, name=name, on_reject=on_reject)
        return self._add(node, NodeKind.APPROVAL, name)

    def subworkflow(self, workflow: "Workflow", *,
                    name: str | None = None) -> "Workflow":
        return self._add(workflow, NodeKind.SUBWORKFLOW, name or workflow.name)

    # ------------------------------------------------------------ compilation
    def compile(self, *, model_registry: ModelRegistry | None = None,
                permissions: PermissionSet | None = None,
                strict_permissions: bool = True,
                force: bool = False) -> ExecutionGraph:
        if self._graph is not None and not force:
            return self._graph
        compiler = WorkflowCompiler(
            model_registry=model_registry,
            permissions=permissions if permissions is not None else self.permissions,
            strict_permissions=strict_permissions,
        )
        self._graph = compiler.compile(
            name=self.name, version=self.version, nodes=self._nodes,
            metadata={"description": self.description},
        )
        return self._graph

    @property
    def graph(self) -> ExecutionGraph:
        return self.compile()

    @property
    def hash(self) -> str:
        return self.graph.hash

    @property
    def nodes(self) -> list[Node]:
        return list(self._nodes)

    def signature(self) -> str:
        return stable_hash("workflow", self.name, self.version,
                           [n.signature() for n in self._nodes])

    @property
    def required_permissions(self) -> list[str]:
        return sorted({p for n in self._nodes for p in n.required_permissions})

    @property
    def model_requirements(self) -> dict[str, bool]:
        merged: dict[str, bool] = {}
        for node in self._nodes:
            merged.update(node.model_requirements)
        return merged

    def describe(self) -> dict[str, Any]:
        return {**super().describe(), "version": self.version,
                "hash": self.hash, "nodes": [n.to_dict() for n in self._nodes]}

    def to_mermaid(self) -> str:
        return self.graph.to_mermaid()

    # -------------------------------------------------------------- execution
    async def execute(self, context: ExecutionContext) -> ExecutionResult:
        services = context.services
        registry = services.models if services else None
        graph = self.compile(model_registry=registry,
                             permissions=self.permissions, force=True)

        # Namespaced state keys so nested workflows never collide.
        hash_key = f"{WORKFLOW_HASH_KEY}:{self.name}"
        completed_key = f"{COMPLETED_NODES_KEY}:{self.name}"
        output_key = f"{LAST_OUTPUT_KEY}:{self.name}"

        self._verify_resume_compatibility(context, graph, hash_key)
        context.state[hash_key] = graph.hash
        context.state.setdefault(WORKFLOW_HASH_KEY, graph.hash)  # root alias

        completed: dict[str, Any] = context.state.setdefault(completed_key, {})
        current = context.state.get(output_key, context.input) \
            if completed else context.input

        await context.emit(EventType.WORKFLOW_STARTED, workflow=self.name,
                           version=self.version, workflow_hash=graph.hash,
                           nodes=[n.id for n in graph.nodes],
                           resumed=bool(completed))

        result = ExecutionResult(execution_id=context.execution_id, output=current)
        tracer = services.tracer if services else None

        async def run_nodes() -> ExecutionResult:
            nonlocal current, result
            for node in graph.nodes:
                context.ensure_alive()

                if node.id in completed:
                    await context.emit(EventType.WORKFLOW_NODE_SKIPPED,
                                       workflow=self.name, node_id=node.id,
                                       reason="already completed")
                    current = completed[node.id]
                    continue

                await self._checkpoint(context, node.id, "before_node")
                await context.emit(EventType.WORKFLOW_NODE_STARTED,
                                   workflow=self.name, node_id=node.id,
                                   node=node.name, kind=node.kind.value)

                child = context.child(input=current, node_id=node.id)
                try:
                    node_result = await node.executable.execute(child)
                except Exception as exc:
                    context.merge_from(child)
                    await context.emit(EventType.WORKFLOW_NODE_FAILED,
                                       workflow=self.name, node_id=node.id,
                                       error=repr(exc))
                    raise

                context.merge_from(child)
                if not node_result.succeeded:
                    return node_result

                current = node_result.output
                completed[node.id] = current
                context.state[output_key] = current
                await context.emit(EventType.WORKFLOW_NODE_COMPLETED,
                                   workflow=self.name, node_id=node.id,
                                   node=node.name)
                await self._checkpoint(context, node.id, "after_node")
                result = node_result

            result.output = current
            result.status = ExecutionStatus.COMPLETED
            result.usage = context.usage_snapshot()
            result.metadata.update({"workflow": self.name,
                                    "workflow_hash": graph.hash,
                                    "version": self.version})
            return result

        if tracer is not None:
            async with tracer.span(f"workflow:{self.name}", kind="workflow",
                                   workflow=self.name, version=self.version):
                outcome = await run_nodes()
        else:  # pragma: no cover
            outcome = await run_nodes()

        await context.emit(EventType.WORKFLOW_COMPLETED, workflow=self.name,
                           status=outcome.status.value)
        return outcome

    # --------------------------------------------------------------- helpers
    @staticmethod
    def _verify_resume_compatibility(context: ExecutionContext,
                                     graph: ExecutionGraph,
                                     hash_key: str) -> None:
        """Rule 9: never resume against an incompatible workflow."""
        recorded = context.state.get(hash_key)
        if recorded and recorded != graph.hash:
            raise CheckpointError(
                "workflow definition changed since this execution was "
                "checkpointed; resume aborted",
                execution_id=context.execution_id,
                checkpoint_hash=recorded, current_hash=graph.hash,
            )

    @staticmethod
    async def _checkpoint(context: ExecutionContext, node_id: str,
                          reason: str) -> None:
        services = context.services
        if services is None or services.runtime is None:
            return
        if not context.policy.checkpoint_enabled:
            return
        await services.runtime.create_checkpoint(context, node_id=node_id,
                                                 reason=reason)
