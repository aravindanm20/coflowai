from __future__ import annotations

import asyncio
import inspect

import pytest

from coflowai import Agent, ExecutionPolicy, tool
from coflowai.testing import build_test_app


@pytest.fixture
def weather_tool():
    @tool(description="Get the weather for a city", permission="weather.read")
    async def get_weather(city: str) -> dict:
        return {"city": city, "temperature": 30}

    return get_weather


@pytest.fixture
def make_app():
    def factory(responses=None, **kwargs):
        return build_test_app(responses, **kwargs)

    return factory


@pytest.fixture
def simple_agent():
    return Agent(name="assistant", model="fake-model",
                 instructions="Answer the user's question.")


@pytest.fixture
def fast_policy():
    return ExecutionPolicy(max_steps=5, timeout_seconds=10, max_model_calls=5,
                           max_tool_calls=5, retry_attempts=0)


# --------------------------------------------------------------------------- #
# Async test support without requiring a third-party plugin.
# If pytest-asyncio is installed (dev extra) it takes over; otherwise every
# `async def` test is executed on a fresh event loop by this hook.
# --------------------------------------------------------------------------- #
def pytest_pyfunc_call(pyfuncitem):
    if pyfuncitem.config.pluginmanager.hasplugin("asyncio"):
        return None
    test = pyfuncitem.obj
    if not inspect.iscoroutinefunction(test):
        return None
    kwargs = {name: pyfuncitem.funcargs[name]
              for name in pyfuncitem._fixtureinfo.argnames}
    asyncio.run(test(**kwargs))
    return True
