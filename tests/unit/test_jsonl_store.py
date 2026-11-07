import time

from coflowai.events.event import Event
from coflowai.events.store import JsonlEventStore


async def test_sequences_and_reopen(tmp_path):
    path = tmp_path / "events.jsonl"
    store = JsonlEventStore(path)
    for _ in range(3):
        await store.append(Event(execution_id="a", type="t"))
    await store.append(Event(execution_id="b", type="t"))
    reopened = JsonlEventStore(path)            # new process, same file
    e = await reopened.append(Event(execution_id="a", type="t"))
    assert e.sequence == 4
    assert [x.sequence for x in await reopened.get_events("a")] == [1, 2, 3, 4]


async def test_sees_writes_from_another_writer(tmp_path):
    path = tmp_path / "events.jsonl"
    one, two = JsonlEventStore(path), JsonlEventStore(path)
    await one.append(Event(execution_id="a", type="t"))
    assert (await two.append(Event(execution_id="a", type="t"))).sequence == 2
    assert (await one.append(Event(execution_id="a", type="t"))).sequence == 3


async def test_append_does_not_rescan_whole_file(tmp_path):
    store = JsonlEventStore(tmp_path / "e.jsonl")
    start = time.perf_counter()
    for i in range(3000):
        await store.append(Event(execution_id=f"x{i % 50}", type="t"))
    assert time.perf_counter() - start < 5     # quadratic version: far slower
