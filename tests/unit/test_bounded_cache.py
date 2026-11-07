import time

import pytest

from coflowai import Agent
from coflowai.core.cache import BoundedCache
from coflowai.runtime.runtime import Runtime
from coflowai.testing import build_test_app


def test_lru_eviction():
    c = BoundedCache(max_size=2)
    c["a"], c["b"] = 1, 2
    _ = c["a"]                 # a is now most recent
    c["c"] = 3
    assert "b" not in c and set(c) == {"a", "c"}


def test_ttl_expiry():
    c = BoundedCache(max_size=None, ttl_seconds=0.01)
    c["a"] = 1
    time.sleep(0.02)
    assert c.get("a") is None and len(c) == 0


def test_invalid_limits():
    with pytest.raises(ValueError):
        BoundedCache(max_size=0)


async def test_runtime_caches_stay_bounded():
    app, _ = build_test_app([f"r{i}" for i in range(20)])
    rt = app.runtime
    for cache in (rt._executables, rt._policies, rt._tracers, rt._usage):
        cache.max_size = 5
    agent = Agent("a", "i", "fake-model")
    for _ in range(20):
        assert (await app.run(agent, "go")).succeeded
    assert max(len(rt._executables), len(rt._policies),
               len(rt._tracers), len(rt._usage)) <= 5


def test_runtime_accepts_cache_and_lock_settings():
    rt = Runtime(cache_size=10, cache_ttl_seconds=60, lock_ttl_seconds=30)
    assert rt._policies.max_size == 10 and rt._policies.ttl_seconds == 60
    assert rt.lock_ttl_seconds == 30
    with pytest.raises(ValueError):
        Runtime(lock_ttl_seconds=0)
