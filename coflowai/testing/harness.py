"""Testing harness: spin up a fully wired, offline CoFlowAi app in one line."""

from __future__ import annotations

from typing import Any, Iterable, Sequence

from ..app import CoFlowAi
from ..core.result import ExecutionResult
from ..memory.in_memory import InMemoryMemoryStore
from ..models.base import ModelCapabilities
from ..policies.policy import ExecutionPolicy
from ..tools.tool import Tool
from .fakes import FakeModelProvider

__all__ = ["build_test_app", "AgentTestHarness"]


def build_test_app(responses: Sequence[Any] | None = None, *,
                   tools: Iterable[Tool] | None = None,
                   model_name: str = "fake-model",
                   policy: ExecutionPolicy | None = None,
                   permissions: Iterable[str] | None = None,
                   memory: bool = False,
                   provider: FakeModelProvider | None = None,
                   capabilities: ModelCapabilities | None = None
                   ) -> tuple[CoFlowAi, FakeModelProvider]:
    """Return an app wired to a :class:`FakeModelProvider` — no network at all."""
    app = CoFlowAi(policy=policy, permissions=permissions,
                   memory=InMemoryMemoryStore() if memory else None)
    fake = provider or FakeModelProvider(responses or [])
    app.models.register(model_name, fake,
                        capabilities=capabilities or fake.capabilities,
                        pricing=fake.pricing, default=True)
    for tool in tools or ():
        app.tools.register(tool)
    return app, fake


class AgentTestHarness:
    """Convenience wrapper for asserting on agent/workflow behaviour."""

    def __init__(self, executable: Any, responses: Sequence[Any] | None = None,
                 **kwargs: Any) -> None:
        self.executable = executable
        self.app, self.model = build_test_app(responses, **kwargs)

    async def run(self, input: Any = None, **kwargs: Any) -> ExecutionResult:
        self.result = await self.app.run(self.executable, input, **kwargs)
        return self.result

    async def events(self, execution_id: str | None = None) -> list[str]:
        target = execution_id or self.result.execution_id
        events = await self.app.runtime.event_store.get_events(target)
        return [e.type for e in events]

    async def replay(self, execution_id: str | None = None):
        return await self.app.replay(execution_id or self.result.execution_id)

    @property
    def model_calls(self) -> int:
        return self.model.call_count
