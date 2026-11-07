"""RedisJobQueue against an in-memory fake client (no server needed)."""
from __future__ import annotations

import asyncio
import json
import time

from coflowai.distributed.queue import Job, RedisJobQueue


class FakeRedis:
    def __init__(self):
        self.lists, self.zsets, self.hashes = {}, {}, {}
        self.zrange_calls = 0

    async def lpush(self, k, v): self.lists.setdefault(k, []).insert(0, v)

    async def brpop(self, k, timeout=1):
        items = self.lists.get(k)
        return (k, items.pop()) if items else None

    async def llen(self, k): return len(self.lists.get(k, []))

    async def zadd(self, k, mapping, xx=False):
        z = self.zsets.setdefault(k, {})
        for m, score in mapping.items():
            if not xx or m in z:
                z[m] = score

    async def zrem(self, k, m):
        return 1 if self.zsets.get(k, {}).pop(m, None) is not None else 0

    async def zrange(self, k, a, b):
        self.zrange_calls += 1
        return list(self.zsets.get(k, {}))

    async def zrangebyscore(self, k, lo, hi):
        return [m for m, s in self.zsets.get(k, {}).items() if lo <= s <= hi]

    async def hset(self, k, f, v): self.hashes.setdefault(k, {})[f] = v

    async def hget(self, k, f): return self.hashes.get(k, {}).get(f)

    async def hdel(self, k, f): return 1 if self.hashes.get(k, {}).pop(f, None) else 0


def _job(i=0):
    return Job(kind="run", execution_id=f"e{i}")


async def test_ack_is_keyed_by_job_id():
    r = FakeRedis(); q = RedisJobQueue(r)
    for i in range(50):
        await q.enqueue(_job(i))
    jobs = [await q.dequeue() for _ in range(50)]
    for job in jobs:
        await q.ack(job)
    assert r.zrange_calls == 0                       # no full scans
    assert not r.zsets[q._key("processing")] and not r.hashes[q._key("inflight")]


async def test_recover_requeues_then_dead_letters():
    r = FakeRedis(); q = RedisJobQueue(r, visibility_timeout=0.0, max_attempts=2)
    await q.enqueue(_job())
    await q.dequeue(); time.sleep(0.001)
    assert await q.recover() == 1 and await q.depth() == 1
    await q.dequeue(); time.sleep(0.001)             # attempt 2 == max
    assert await q.recover() == 1
    assert await q.depth() == 0 and len(r.lists[q._key("dead")]) == 1


async def test_concurrent_recover_requeues_once():
    r = FakeRedis()
    q1 = RedisJobQueue(r, visibility_timeout=0.0)
    q2 = RedisJobQueue(r, visibility_timeout=0.0)
    await q1.enqueue(_job()); await q1.dequeue(); time.sleep(0.001)
    assert await q1.recover() + await q2.recover() == 1


async def test_legacy_json_members_are_still_recovered():
    r = FakeRedis(); q = RedisJobQueue(r)
    await r.zadd(q._key("processing"), {json.dumps(_job().to_dict()): 0})
    assert await q.recover() == 1 and await q.depth() == 1


async def test_worker_extends_lease_for_long_jobs():
    from coflowai.distributed.worker import Worker
    r = FakeRedis(); q = RedisJobQueue(r, visibility_timeout=0.15)
    extended = []
    original = q.extend

    async def spy(job, **kw):
        extended.append(job.job_id)
        await original(job, **kw)

    q.extend = spy
    worker = Worker.__new__(Worker)          # only the heartbeat is under test
    worker.queue, worker.name = q, "w"
    await q.enqueue(_job()); job = await q.dequeue()
    task = worker._start_lease_heartbeat(job)
    await asyncio.sleep(0.3)
    task.cancel()
    assert extended and await q.recover() == 0   # lease kept alive
