"""Discover plugins published under the ``coflowai.plugins`` entry point group."""

from __future__ import annotations

from importlib.metadata import entry_points
from typing import Iterable

from ..core.exceptions import PluginError
from ..observability.logger import get_logger
from .plugin import Plugin

__all__ = ["discover_plugins", "load_plugin"]

_log = get_logger("plugins")
ENTRY_POINT_GROUP = "coflowai.plugins"


def discover_plugins(group: str = ENTRY_POINT_GROUP) -> list[Plugin]:
    found: list[Plugin] = []
    try:
        points: Iterable = entry_points(group=group)
    except Exception:  # pragma: no cover - very old importlib
        return found
    for point in points:
        try:
            factory = point.load()
            plugin = factory() if callable(factory) else factory
            if isinstance(plugin, Plugin):
                found.append(plugin)
        except Exception as exc:  # pragma: no cover - broken third-party plugin
            _log.warning("plugin.load_failed", plugin=point.name, error=repr(exc))
    return found


def load_plugin(path: str) -> Plugin:
    """Load ``package.module:Factory`` and instantiate it."""
    module_name, _, attribute = path.partition(":")
    if not attribute:
        raise PluginError("plugin path must look like 'module:Factory'", path=path)
    import importlib

    try:
        module = importlib.import_module(module_name)
        factory = getattr(module, attribute)
    except Exception as exc:
        raise PluginError(f"cannot load plugin '{path}'", cause=exc) from exc
    plugin = factory() if callable(factory) else factory
    if not isinstance(plugin, Plugin):
        raise PluginError(f"'{path}' is not a Plugin")
    return plugin
