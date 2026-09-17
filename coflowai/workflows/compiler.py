"""Workflow compiler.

Everything that can be proven wrong before spending a token is checked here:
missing nodes, invalid edges, unreachable nodes, illegal cycles, missing
permissions and unsupported model capabilities.
"""

from __future__ import annotations

from typing import Any, Iterable

from ..core.exceptions import CompilationError
from ..models.registry import ModelRegistry
from ..policies.permissions import PermissionSet
from .graph import Edge, ExecutionGraph
from .node import Node

__all__ = ["WorkflowCompiler", "compile_workflow"]


class WorkflowCompiler:
    def __init__(self, *, model_registry: ModelRegistry | None = None,
                 permissions: PermissionSet | None = None,
                 strict_permissions: bool = True) -> None:
        self.model_registry = model_registry
        self.permissions = permissions
        self.strict_permissions = strict_permissions

    def compile(self, *, name: str, version: str, nodes: list[Node],
                edges: Iterable[Edge] | None = None,
                metadata: dict[str, Any] | None = None) -> ExecutionGraph:
        nodes = list(nodes)
        self._validate_nodes(name, nodes)
        edge_list = list(edges) if edges is not None else _linear_edges(nodes)
        self._validate_edges(name, nodes, edge_list)
        self._validate_reachability(name, nodes, edge_list)
        self._validate_acyclic(name, edge_list)
        self._validate_permissions(name, nodes)
        self._validate_models(name, nodes)
        return ExecutionGraph(name=name, version=version, nodes=nodes,
                              edges=edge_list, metadata=dict(metadata or {}))

    # ----------------------------------------------------------- validations
    @staticmethod
    def _validate_nodes(name: str, nodes: list[Node]) -> None:
        if not nodes:
            raise CompilationError(f"workflow '{name}' has no nodes", workflow=name)
        seen: set[str] = set()
        for node in nodes:
            if node.id in seen:
                raise CompilationError(f"duplicate node id '{node.id}'",
                                       workflow=name)
            seen.add(node.id)
            if node.executable is None:
                raise CompilationError(f"node '{node.id}' has no executable",
                                       workflow=name)

    @staticmethod
    def _validate_edges(name: str, nodes: list[Node], edges: list[Edge]) -> None:
        ids = {n.id for n in nodes}
        for edge in edges:
            missing = [end for end in (edge.source, edge.target) if end not in ids]
            if missing:
                raise CompilationError(
                    f"edge {edge.source}->{edge.target} references unknown nodes",
                    workflow=name, missing=missing,
                )

    @staticmethod
    def _validate_reachability(name: str, nodes: list[Node],
                               edges: list[Edge]) -> None:
        if len(nodes) == 1:
            return
        start = nodes[0].id
        adjacency: dict[str, list[str]] = {}
        for edge in edges:
            adjacency.setdefault(edge.source, []).append(edge.target)
        seen = {start}
        stack = [start]
        while stack:
            current = stack.pop()
            for nxt in adjacency.get(current, ()):
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        unreachable = [n.id for n in nodes if n.id not in seen]
        if unreachable:
            raise CompilationError(f"workflow '{name}' has unreachable nodes",
                                   workflow=name, unreachable=unreachable)

    @staticmethod
    def _validate_acyclic(name: str, edges: list[Edge]) -> None:
        adjacency: dict[str, list[str]] = {}
        for edge in edges:
            adjacency.setdefault(edge.source, []).append(edge.target)
        WHITE, GREY, BLACK = 0, 1, 2
        colour: dict[str, int] = {}

        def visit(node: str, path: list[str]) -> None:
            colour[node] = GREY
            for nxt in adjacency.get(node, ()):
                state = colour.get(nxt, WHITE)
                if state == GREY:
                    raise CompilationError(
                        f"workflow '{name}' contains an illegal cycle "
                        "(use .loop(..., max_iterations=n) for repetition)",
                        workflow=name, cycle=path + [node, nxt],
                    )
                if state == WHITE:
                    visit(nxt, path + [node])
            colour[node] = BLACK

        for source in list(adjacency):
            if colour.get(source, WHITE) == WHITE:
                visit(source, [])

    def _validate_permissions(self, name: str, nodes: list[Node]) -> None:
        if self.permissions is None or not self.strict_permissions:
            return
        problems: dict[str, list[str]] = {}
        for node in nodes:
            missing = self.permissions.missing(node.required_permissions)
            if missing:
                problems[node.id] = missing
        if problems:
            raise CompilationError(
                f"workflow '{name}' requires permissions that were not granted",
                workflow=name, missing=problems,
            )

    def _validate_models(self, name: str, nodes: list[Node]) -> None:
        if self.model_registry is None:
            return
        for node in nodes:
            for executable in _leaves(node.executable):
                self._validate_one(name, node.id, executable)

    def _validate_one(self, name: str, node_id: str, executable: Any) -> None:
        requirements = executable.model_requirements
        model = getattr(executable, "model", None)
        if model:
            if not self.model_registry.has(model):
                raise CompilationError(
                    f"node '{node_id}' references unregistered model '{model}'",
                    workflow=name, model=model,
                    available=sorted(m.name for m in self.model_registry.all()),
                )
            missing = self.model_registry.capabilities(model).missing(requirements)
            if missing:
                raise CompilationError(
                    f"model '{model}' lacks capabilities needed by "
                    f"node '{node_id}'",
                    workflow=name, missing=missing,
                )
        elif requirements:
            try:
                self.model_registry.select(requirements)
            except Exception as exc:
                raise CompilationError(
                    f"no registered model satisfies node '{node_id}'",
                    workflow=name, requirements=requirements, cause=exc,
                ) from exc


def _leaves(executable: Any) -> list[Any]:
    """Flatten composite nodes (parallel/router/loop/sequence) into leaves."""
    for attribute in ("branches", "steps"):
        children = getattr(executable, attribute, None)
        if children:
            return [leaf for child in children for leaf in _leaves(child)]
    routes = getattr(executable, "routes", None)
    if routes:
        targets = list(routes.values())
        default = getattr(executable, "default", None)
        if default is not None:
            targets.append(default)
        return [leaf for target in targets for leaf in _leaves(target)]
    for attribute in ("body", "action"):
        child = getattr(executable, attribute, None)
        if child is not None:
            return _leaves(child)
    nodes = getattr(executable, "nodes", None)     # nested workflow
    if nodes:
        return [leaf for node in nodes for leaf in _leaves(node.executable)]
    return [executable]


def _linear_edges(nodes: list[Node]) -> list[Edge]:
    return [Edge(source=a.id, target=b.id) for a, b in zip(nodes, nodes[1:])]


def compile_workflow(workflow, **kwargs: Any) -> ExecutionGraph:
    """Convenience wrapper used by the CLI."""
    return workflow.compile(**kwargs)
