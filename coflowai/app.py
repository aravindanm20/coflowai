"""``CoFlowAi`` — the one convenient entry point (§58)."""

from __future__ import annotations

from typing import Any, Iterable

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
    async def setup(self) -> "CoFlowAi":
        await self.plugins.setup_all()
        return self

    async def shutdown(self) -> None:
        await self.plugins.teardown_all()

    async def __aenter__(self) -> "CoFlowAi":
        return await self.setup()

    async def __aexit__(self, *exc_info: Any) -> None:
        await self.shutdown()

    # -------------------------------------------------------------- wiring
    @property
    def memory(self) -> MemoryStore | None:
        return self.runtime.memory

    def use_memory(self, store: MemoryStore) -> None:
        self.runtime.memory = store

    def register_plugin(self, plugin: Plugin) -> Plugin:
        return self.plugins.register(plugin)

    def on_event(self, subscriber, *, event_type: str | None = None) -> None:
        self.runtime.publisher.subscribe(subscriber, event_type=event_type)

    # ------------------------------------------------------------ execution
    async def run(self, executable: Any, input: Any = None, *,
                  policy: ExecutionPolicy | None = None,
                  **kwargs: Any) -> ExecutionResult:
        return await self.runtime.run(executable, input, policy=policy, **kwargs)

    def stream(self, executable: Any, input: Any = None, **kwargs: Any):
        """Stream an agent execution (async iterator of ``StreamChunk``)."""
        return self.runtime.stream(executable, input, **kwargs)

    async def resume(self, execution_id: str, **kwargs: Any) -> ExecutionResult:
        return await self.runtime.resume(execution_id, **kwargs)

    async def approve(self, execution_id: str, **kwargs: Any) -> ExecutionResult:
        return await self.runtime.approve(execution_id, **kwargs)

    async def reject(self, execution_id: str, **kwargs: Any) -> ExecutionResult:
        return await self.runtime.reject(execution_id, **kwargs)

    async def cancel(self, execution_id: str, **kwargs: Any) -> bool:
        return await self.runtime.cancel(execution_id, **kwargs)

    async def replay(self, execution_id: str) -> ReplayTrace:
        return await self.runtime.replay(execution_id)

    async def fork(self, execution_id: str, **kwargs: Any) -> ExecutionResult:
        return await self.runtime.fork(execution_id, **kwargs)

    # ----------------------------------------------------------- inspection
    def trace(self, execution_id: str):
        return self.runtime.trace(execution_id)

    def usage(self, execution_id: str):
        return self.runtime.usage(execution_id)

    def metrics_snapshot(self) -> dict[str, Any]:
        snapshot = getattr(self.runtime.metrics, "snapshot", None)
        return snapshot() if callable(snapshot) else {}
