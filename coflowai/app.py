"""``CoFlowAi`` — the one convenient entry point (§58)."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterable
from typing import Any

from .core.result import ExecutionResult
from .events.store import EventStore
from .memory.base import MemoryStore
from .models.base import ModelCapabilities, ModelPricing, ModelProvider
from .models.registry import ModelRegistry
from .observability.logger import configure_logging
from .observability.metrics import MetricsSink
from .plugins.loader import discover_plugins
from .plugins.plugin import Plugin, PluginRegistry
from .policies.permissions import PermissionSet
from .policies.policy import ExecutionPolicy
from .runtime.replay import ReplayTrace
from .runtime.runtime import Runtime
from .state.store import StateStore
from .tools.registry import ToolRegistry
from .tools.sandbox import Sandbox
from .tools.tool import Tool

__all__ = ["CoFlowAi"]


class _ModelsFacade:
    def __init__(self, registry: ModelRegistry) -> None:
        self._registry = registry

    def register(self, name: str, provider: ModelProvider, *,
                 capabilities: ModelCapabilities | None = None,
                 pricing: ModelPricing | None = None, **kwargs: Any):
        return self._registry.register(name, provider, capabilities=capabilities,
                                       pricing=pricing, **kwargs)

    def list(self) -> list[dict[str, Any]]:
        return [m.to_dict() for m in self._registry.all()]

    def set_default(self, name: str) -> None:
        self._registry.set_default(name)

    def __getattr__(self, item: str) -> Any:  # delegate the rest
        return getattr(self._registry, item)


class _ToolsFacade:
    def __init__(self, registry: ToolRegistry) -> None:
        self._registry = registry

    def register(self, *tools: Tool) -> None:
        for tool in tools:
            self._registry.register(tool)

    def list(self) -> list[dict[str, Any]]:
        return [t.definition.to_dict() for t in self._registry.all()]

    def __getattr__(self, item: str) -> Any:
        return getattr(self._registry, item)


class CoFlowAi:
    """Application object wiring registries, runtime and plugins together."""

    def __init__(self, *,
                 event_store: EventStore | None = None,
                 state_store: StateStore | None = None,
                 memory: MemoryStore | None = None,
                 metrics: MetricsSink | None = None,
                 sandbox: Sandbox | None = None,
                 policy: ExecutionPolicy | None = None,
                 permissions: Iterable[str] | PermissionSet | None = None,
                 logging_level: str | int | None = None,
                 auto_discover_plugins: bool = False) -> None:
        if logging_level is not None:
            configure_logging(logging_level)

        self.runtime = Runtime(
            event_store=event_store,
            state_store=state_store,
            memory=memory,
            metrics=metrics,
            sandbox=sandbox,
            default_policy=policy,
            permissions=permissions,
        )
        self.models = _ModelsFacade(self.runtime.models)
        self.tools = _ToolsFacade(self.runtime.tools)
        self.plugins = PluginRegistry(self)
        self._memory = memory

        if auto_discover_plugins:
            for plugin in discover_plugins():
                self.plugins.register(plugin)

    # ------------------------------------------------------------- lifecycle
    async def setup(self) -> CoFlowAi:
        await self.plugins.setup_all()
        return self

    async def shutdown(self) -> None:
        await self.plugins.teardown_all()

    async def __aenter__(self) -> CoFlowAi:
        return await self.setup()

    async def __aexit__(self, *exc_info: Any) -> None:
        await self.shutdown()

    # -------------------------------------------------------------- wiring
    @property
    def memory(self) -> MemoryStore | None:
        return self.runtime.memory

    @property
    def event_store(self) -> EventStore:
        """The event log (``await app.event_store.get_events(execution_id)``)."""
        return self.runtime.event_store

    @property
    def state_store(self) -> StateStore:
        """Execution records and checkpoints
        (``await app.state_store.get_execution(execution_id)``)."""
        return self.runtime.state_store

    def use_memory(self, store: MemoryStore) -> None:
        self.runtime.memory = store

    def register_plugin(self, plugin: Plugin) -> Plugin:
        return self.plugins.register(plugin)

    def on_event(self, subscriber, *, event_type: str | None = None) -> None:
        self.runtime.publisher.subscribe(subscriber, event_type=event_type)

    # ------------------------------------------------------------ execution
    # Explicit signatures (instead of **kwargs) so IDEs, type checkers and
    # scripts/check_docs.py catch misspelt arguments before runtime.
    async def run(self, executable: Any, input: Any = None, *,
                  policy: ExecutionPolicy | None = None,
                  execution_id: str | None = None,
                  permissions: PermissionSet | Iterable[str] | None = None,
                  metadata: dict[str, Any] | None = None,
                  state: dict[str, Any] | None = None) -> ExecutionResult:
        return await self.runtime.run(executable, input, policy=policy,
                                      execution_id=execution_id,
                                      permissions=permissions,
                                      metadata=metadata, state=state)

    def stream(self, executable: Any, input: Any = None, *,
               policy: ExecutionPolicy | None = None,
               execution_id: str | None = None,
               permissions: PermissionSet | Iterable[str] | None = None,
               metadata: dict[str, Any] | None = None) -> AsyncIterator[Any]:
        """Stream an agent execution (async iterator of ``StreamChunk``)."""
        return self.runtime.stream(executable, input, policy=policy,
                                   execution_id=execution_id,
                                   permissions=permissions, metadata=metadata)

    async def resume(self, execution_id: str, *, executable: Any = None,
                     policy: ExecutionPolicy | None = None) -> ExecutionResult:
        return await self.runtime.resume(execution_id, executable=executable,
                                         policy=policy)

    async def approve(self, execution_id: str, *, approval_id: str | None = None,
                      by: str | None = None, note: str | None = None,
                      executable: Any = None) -> ExecutionResult:
        return await self.runtime.approve(execution_id, approval_id=approval_id,
                                          by=by, note=note, executable=executable)

    async def reject(self, execution_id: str, *, approval_id: str | None = None,
                     by: str | None = None, note: str | None = None,
                     executable: Any = None) -> ExecutionResult:
        return await self.runtime.reject(execution_id, approval_id=approval_id,
                                         by=by, note=note, executable=executable)

    async def cancel(self, execution_id: str, *,
                     reason: str = "cancelled by operator") -> bool:
        return await self.runtime.cancel(execution_id, reason=reason)

    async def replay(self, execution_id: str) -> ReplayTrace:
        return await self.runtime.replay(execution_id)

    async def fork(self, execution_id: str, *, from_step: int,
                   overrides: dict[str, Any] | None = None,
                   executable: Any = None,
                   policy: ExecutionPolicy | None = None) -> ExecutionResult:
        return await self.runtime.fork(execution_id, from_step=from_step,
                                       overrides=overrides, executable=executable,
                                       policy=policy)

    # ----------------------------------------------------------- inspection
    def trace(self, execution_id: str):
        return self.runtime.trace(execution_id)

    def usage(self, execution_id: str):
        return self.runtime.usage(execution_id)

    def metrics_snapshot(self) -> dict[str, Any]:
        snapshot = getattr(self.runtime.metrics, "snapshot", None)
        return snapshot() if callable(snapshot) else {}
