"""Context engine, memory store, tracer/metrics/replay rendering."""

from __future__ import annotations

from coflowai import ContextBudget, ContextBuilder, InMemoryMemoryStore, Message
from coflowai.context.builder import estimate_tokens
from coflowai.memory.base import MemoryKind


async def test_builder_orders_sections_by_priority():
    builder = ContextBuilder()
    builder.conversation([Message.user("question")])
    builder.instructions("system rules")
    builder.memory(["remembered fact"])
    messages = await builder.build()
    assert messages[0].content == "system rules"
    assert messages[-1].content == "question"


async def test_builder_enforces_section_budgets_keeping_recent_turns():
    budget = ContextBudget(max_tokens=400, instructions=100, conversation=60,
                           reserve_output=0)
    builder = ContextBuilder(budget)
    builder.instructions("rules")
    builder.conversation([Message.user("old " * 50), Message.user("recent question")])
    messages = await builder.build()
    contents = [m.content for m in messages]
    assert "recent question" in contents
    assert not any(c.startswith("old ") for c in contents)


async def test_builder_summarises_old_conversation_instead_of_dropping_it():
    def summarise(messages, limit):
        return f"{len(messages)} earlier turns"

    budget = ContextBudget(max_tokens=1000, conversation=30, reserve_output=0)
    builder = ContextBuilder(budget, summariser=summarise)
    builder.conversation([Message.user("old " * 40), Message.user("newest")])
    messages = await builder.build()
    assert any("[summary]" in (m.content or "") for m in messages)
    assert any("newest" == m.content for m in messages)


async def test_tool_results_are_compressed_not_truncated_from_the_end():
    budget = ContextBudget(max_tokens=5000, tool_results=100, reserve_output=0)
    builder = ContextBuilder(budget)
    builder.add("tool_results", [Message.tool("c1", "dump", "x" * 5000)])
    messages = await builder.build()
    assert "omitted" in messages[0].content


async def test_workflow_state_is_exposed_but_internals_are_hidden():
    builder = ContextBuilder()
    builder.workflow_state({"topic": "AI", "__completed_nodes__": {"a": 1}})
    messages = await builder.build()
    rendered = messages[0].content
    assert "topic" in rendered and "__completed_nodes__" not in rendered


def test_token_estimation_is_monotonic():
    assert estimate_tokens("") == 0
    assert estimate_tokens("a" * 400) > estimate_tokens("a" * 40)


# ---------------------------------------------------------------------- memory
async def test_memory_search_ranks_relevant_records_first():
    store = InMemoryMemoryStore()
    await store.put("a", "The user prefers dark mode and concise answers")
    await store.put("b", "Deployment runbook for the billing service")
    results = await store.search("deployment runbook", limit=1)
    assert [r.key for r in results] == ["b"]


async def test_memory_filters_by_kind_and_supports_delete():
    store = InMemoryMemoryStore()
    await store.put("a", "episodic note", kind=MemoryKind.EPISODIC)
    await store.put("b", "long term note", kind=MemoryKind.LONG_TERM)
    assert len(await store.search("note", kind=MemoryKind.EPISODIC)) == 1
    await store.delete("a")
    assert await store.get("a") is None


async def test_memory_is_bounded_when_configured():
    store = InMemoryMemoryStore(max_records=2)
    for index in range(5):
        await store.put(f"k{index}", f"value {index}")
    assert len(store) == 2
