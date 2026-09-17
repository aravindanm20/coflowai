"""Model registry + capability-based router.

Agents refer to models by *logical name* or by *capability requirements*; they
never import a provider SDK.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from ..core.exceptions import ConfigurationError
from .base import ModelCapabilities, ModelPricing, ModelProvider

__all__ = ["RegisteredModel", "ModelRegistry", "ModelRouter"]


@dataclass(slots=True)
class RegisteredModel:
    name: str
    provider: ModelProvider
    capabilities: ModelCapabilities = field(default_factory=ModelCapabilities)
    pricing: ModelPricing = field(default_factory=ModelPricing)
    priority: int = 100          # lower wins when several models match
    latency_hint_ms: float = 0.0
    fallbacks: list[str] = field(default_factory=list)
    tags: set[str] = field(default_factory=set)

    def cost_hint(self) -> float:
        return self.pricing.input_per_million + self.pricing.output_per_million

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "provider": getattr(self.provider, "name", type(self.provider).__name__),
            "capabilities": self.capabilities.to_dict(),
            "pricing": {
                "input_per_million": self.pricing.input_per_million,
                "output_per_million": self.pricing.output_per_million,
            },
            "priority": self.priority,
            "fallbacks": self.fallbacks,
            "tags": sorted(self.tags),
        }


class ModelRegistry:
    def __init__(self) -> None:
        self._models: dict[str, RegisteredModel] = {}
        self._default: str | None = None

    # ----------------------------------------------------------- registration
    def register(self, name: str, provider: ModelProvider, *,
                 capabilities: ModelCapabilities | None = None,
                 pricing: ModelPricing | None = None,
                 priority: int = 100,
                 latency_hint_ms: float = 0.0,
                 fallbacks: Iterable[str] | None = None,
                 tags: Iterable[str] | None = None,
                 default: bool = False) -> RegisteredModel:
        model = RegisteredModel(
            name=name,
            provider=provider,
            # adapters may advertise their own capabilities/pricing
            capabilities=(capabilities
                          or getattr(provider, "capabilities", None)
                          or ModelCapabilities()),
            pricing=pricing or getattr(provider, "pricing", None) or ModelPricing(),
            priority=priority,
            latency_hint_ms=latency_hint_ms,
            fallbacks=list(fallbacks or ()),
            tags=set(tags or ()),
        )
        self._models[name] = model
        if default or self._default is None:
            self._default = name
        return model

    def unregister(self, name: str) -> None:
        self._models.pop(name, None)
        if self._default == name:
            self._default = next(iter(self._models), None)

    # ---------------------------------------------------------------- lookups
    def get(self, name: str | None = None) -> RegisteredModel:
        key = name or self._default
        if key is None:
            raise ConfigurationError("no models registered")
        try:
            return self._models[key]
        except KeyError as exc:
            raise ConfigurationError(
                f"model '{key}' is not registered",
                available=sorted(self._models),
            ) from exc

    def has(self, name: str) -> bool:
        return name in self._models

    def all(self) -> list[RegisteredModel]:
        return list(self._models.values())

    @property
    def default_model(self) -> str | None:
        return self._default

    def set_default(self, name: str) -> None:
        if name not in self._models:
            raise ConfigurationError(f"model '{name}' is not registered")
        self._default = name

    def capabilities(self, name: str) -> ModelCapabilities:
        return self.get(name).capabilities

    # ---------------------------------------------------------------- routing
    def select(self, requirements: dict[str, bool] | None = None, *,
               prefer: str = "priority", tags: Iterable[str] | None = None
               ) -> RegisteredModel:
        """Capability-based selection: filter, then rank by policy."""
        requirements = requirements or {}
        wanted_tags = set(tags or ())
        candidates = [
            m for m in self._models.values()
            if m.capabilities.satisfies(requirements)
            and wanted_tags.issubset(m.tags)
        ]
        if not candidates:
            raise ConfigurationError(
                "no registered model satisfies the requirements",
                requirements=requirements,
                available=sorted(self._models),
            )
        if prefer == "cost":
            key = lambda m: (m.cost_hint(), m.priority)          # noqa: E731
        elif prefer == "latency":
            key = lambda m: (m.latency_hint_ms, m.priority)      # noqa: E731
        else:
            key = lambda m: (m.priority, m.cost_hint())          # noqa: E731
        return sorted(candidates, key=key)[0]

    def resolve(self, *, model: str | None,
                requirements: dict[str, bool] | None = None,
                prefer: str = "priority") -> RegisteredModel:
        if model:
            registered = self.get(model)
            if requirements:
                missing = registered.capabilities.missing(requirements)
                if missing:
                    raise ConfigurationError(
                        f"model '{model}' lacks required capabilities",
                        missing=missing,
                    )
            return registered
        if requirements:
            return self.select(requirements, prefer=prefer)
        return self.get(None)


#: Backwards-compatible alias — routing lives in the registry itself.
ModelRouter = ModelRegistry
