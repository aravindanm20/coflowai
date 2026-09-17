"""Context engine.

Context != conversation history.  The builder assembles the model input from
labelled sections, enforces a per-section token budget, and degrades gracefully
(drop low relevance -> summarise old turns -> compress tool output) instead of
truncating blindly from the end.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Iterable

from ..core.types import Message, Role

__all__ = ["ContextBudget", "ContextSection", "ContextBuilder", "estimate_tokens"]

Summariser = Callable[[list[Message], int], Awaitable[str] | str]


def estimate_tokens(text: str | None) -> int:
    """Cheap, provider-independent estimate (~4 characters per token)."""
    if not text:
        return 0
    return max(1, (len(text) + 3) // 4)


def message_tokens(message: Message) -> int:
    total = estimate_tokens(message.content) + 4
    for call in message.tool_calls:
        total += estimate_tokens(call.name) + estimate_tokens(str(call.arguments))
    return total


@dataclass(slots=True)
class ContextBudget:
    max_tokens: int = 32_000
    instructions: int = 4_000
    conversation: int = 8_000
    memory: int = 6_000
    tool_results: int = 6_000
    retrieved_context: int = 8_000
    reserve_output: int = 2_000

    def limit_for(self, section: str) -> int:
        return int(getattr(self, section, self.max_tokens))


@dataclass(slots=True)
class ContextSection:
    """A labelled block of context.

    ``order``    -> position in the final prompt (ascending)
    ``priority`` -> importance when the budget is tight (ascending = kept longer)
    """

    name: str
    messages: list[Message] = field(default_factory=list)
    priority: int = 100
    order: int = 100
    relevance: float = 1.0

    def tokens(self) -> int:
        return sum(message_tokens(m) for m in self.messages)


class ContextBuilder:
    """Assembles the final model context for one model request."""

    def __init__(self, budget: ContextBudget | None = None, *,
                 summariser: Summariser | None | bool = None,
                 min_relevance: float = 0.0) -> None:
        self.budget = budget or ContextBudget()
        # A deterministic extractive summariser is the default, so oversized
        # conversations degrade intelligently with no configuration.
        # Pass ``summariser=False`` to fall back to recency truncation.
        if summariser is None:
            from .summarizer import default_summariser
            summariser = default_summariser()
        self.summariser = summariser or None
        self.min_relevance = min_relevance
        self._sections: dict[str, ContextSection] = {}
        self._order: list[str] = []

    # ---------------------------------------------------------------- authoring
    def add(self, name: str, messages: Iterable[Message], *, priority: int = 100,
            order: int | None = None, relevance: float = 1.0) -> "ContextBuilder":
        section = self._sections.get(name)
        if section is None:
            section = ContextSection(name=name, priority=priority,
                                     order=priority if order is None else order,
                                     relevance=relevance)
            self._sections[name] = section
            self._order.append(name)
        section.messages.extend(messages)
        section.priority = priority
        if order is not None:
            section.order = order
        section.relevance = relevance
        return self

    def add_text(self, name: str, text: str, *, role: Role = Role.SYSTEM,
                 priority: int = 100, order: int | None = None,
                 relevance: float = 1.0) -> "ContextBuilder":
        if text:
            self.add(name, [Message(role, text)], priority=priority, order=order,
                     relevance=relevance)
        return self

    def instructions(self, text: str) -> "ContextBuilder":
        return self.add_text("instructions", text, priority=0, order=0)

    def conversation(self, messages: Iterable[Message]) -> "ContextBuilder":
        # kept longest (priority 10) but always rendered last (order 90)
        return self.add("conversation", messages, priority=10, order=90)

    def memory(self, items: Iterable[str], *, relevance: float = 0.6
               ) -> "ContextBuilder":
        rendered = [Message(Role.SYSTEM, f"[memory] {item}") for item in items]
        return self.add("memory", rendered, priority=70, order=20,
                        relevance=relevance)

    def retrieved(self, items: Iterable[str], *, relevance: float = 0.7
                  ) -> "ContextBuilder":
        rendered = [Message(Role.SYSTEM, f"[knowledge] {item}") for item in items]
        return self.add("retrieved_context", rendered, priority=60, order=30,
                        relevance=relevance)

    def tool_results(self, messages: Iterable[Message]) -> "ContextBuilder":
        return self.add("tool_results", messages, priority=50, order=40)

    def workflow_state(self, state: dict[str, Any]) -> "ContextBuilder":
        visible = {k: v for k, v in state.items() if not k.startswith("__")}
        if visible:
            self.add_text("workflow_state",
                          "Workflow state:\n" + _render_state(visible),
                          priority=40, order=10)
        return self

    # ----------------------------------------------------------------- building
    async def build(self) -> list[Message]:
        sections = [self._sections[name] for name in self._order]

        # 1. drop low-relevance sections first
        sections = [s for s in sections
                    if s.relevance >= self.min_relevance or s.priority <= 10]

        # 2. per-section caps
        for section in sections:
            cap = self.budget.limit_for(section.name)
            if section.tokens() > cap:
                section.messages = await self._shrink(section, cap)

        # 3. global cap: compress the least important sections first
        overall = self.budget.max_tokens - self.budget.reserve_output
        while _total(sections) > overall:
            victim = max(sections, key=lambda s: (s.priority, s.tokens()))
            if not victim.messages or victim.priority == 0:
                break
            target = max(0, victim.tokens() - (_total(sections) - overall))
            victim.messages = await self._shrink(victim, target)
            if victim.tokens() == 0:
                sections.remove(victim)
            if not sections:
                break

        messages: list[Message] = []
        for section in sorted(sections, key=lambda s: (s.order, s.priority)):
            messages.extend(section.messages)
        return messages

    async def _shrink(self, section: ContextSection, cap: int) -> list[Message]:
        messages = section.messages
        if cap <= 0:
            return []

        # Tool results: compress the payloads before dropping anything.
        if section.name == "tool_results":
            budget_chars = (cap * 4) // max(1, len(messages))
            per_message = max(80, budget_chars - 80)
            messages = [_compress(m, per_message) for m in messages]

        # Conversation: summarise the oldest turns, keep the most recent verbatim.
        if section.name == "conversation" and self.summariser is not None:
            kept: list[Message] = []
            tokens = 0
            for message in reversed(messages):
                cost = message_tokens(message)
                if tokens + cost > cap * 0.7:
                    break
                kept.append(message)
                tokens += cost
            kept.reverse()
            old = messages[: len(messages) - len(kept)]
            if old:
                summary = self.summariser(old, int(cap * 0.3))
                if hasattr(summary, "__await__"):
                    summary = await summary  # type: ignore[assignment]
                return [Message(Role.SYSTEM, f"[summary] {summary}"), *kept]
            return kept

        # Default: keep the most recent messages that fit.
        kept: list[Message] = []
        tokens = 0
        for message in reversed(messages):
            cost = message_tokens(message)
            if tokens + cost > cap:
                continue
            kept.append(message)
            tokens += cost
        kept.reverse()
        return kept


def _total(sections: list[ContextSection]) -> int:
    return sum(s.tokens() for s in sections)


def _compress(message: Message, limit: int = 800) -> Message:
    if message.content and len(message.content) > limit:
        omitted = len(message.content) - limit
        return Message(
            role=message.role,
            content=message.content[:limit] + f"... [{omitted} chars omitted]",
            tool_call_id=message.tool_call_id,
            name=message.name,
        )
    return message


def _render_state(state: dict[str, Any], limit: int = 400) -> str:
    lines = []
    for key, value in state.items():
        rendered = repr(value)
        if len(rendered) > limit:
            rendered = rendered[:limit] + "..."
        lines.append(f"- {key}: {rendered}")
    return "\n".join(lines)
