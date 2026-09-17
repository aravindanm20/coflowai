"""Plugin API.

Plugins register models, tools, memory/event/state stores, tracing exporters and
workflow nodes.  They are how the core stays dependency-free.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

__all__ = ["Plugin", "PluginRegistry"]


class Plugin(ABC):
    name: str = "plugin"
    version: str = "0.0.0"
    requires: tuple[str, ...] = ()

    @abstractmethod
    async def setup(self, app: Any) -> None:
        """Register components on the :class:`~coflowai.app.CoFlowAi` instance."""

    async def teardown(self, app: Any) -> None:  # pragma: no cover - optional
        return None

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "version": self.version,
                "requires": list(self.requires)}


class PluginRegistry:
    def __init__(self, app: Any) -> None:
        self._app = app
        self._plugins: dict[str, Plugin] = {}

    def register(self, plugin: Plugin) -> Plugin:
        from ..core.exceptions import PluginError

        if not isinstance(plugin, Plugin):
            raise PluginError(f"{plugin!r} does not implement the Plugin interface")
        missing = [dep for dep in plugin.requires if dep not in self._plugins]
        if missing:
            raise PluginError(f"plugin '{plugin.name}' has unmet dependencies",
                              missing=missing)
        self._plugins[plugin.name] = plugin
        return plugin

    async def setup_all(self) -> None:
        from ..core.exceptions import PluginError

        for plugin in list(self._plugins.values()):
            try:
                await plugin.setup(self._app)
            except Exception as exc:
                raise PluginError(f"plugin '{plugin.name}' failed during setup",
                                  cause=exc, plugin=plugin.name) from exc

    async def teardown_all(self) -> None:
        for plugin in reversed(list(self._plugins.values())):
            await plugin.teardown(self._app)

    def all(self) -> list[Plugin]:
        return list(self._plugins.values())

    def get(self, name: str) -> Plugin | None:
        return self._plugins.get(name)

    def __len__(self) -> int:
        return len(self._plugins)
