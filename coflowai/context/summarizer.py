"""Conversation summarisers.

Two implementations ship in-tree:

* :class:`ExtractiveSummariser` — the **default**. Deterministic, free, offline.
  It keeps decisions, tool outcomes and questions rather than truncating.
* :class:`ModelSummariser` — higher quality, costs a model call. Opt in when
  fidelity matters more than latency and spend.

A default exists so oversized context degrades sensibly without configuration
(Rule 4 in spirit: capability, never obligation).
"""

from __future__ import annotations

import re
from typing import Any

from ..core.types import Message, Role

__all__ = ["ExtractiveSummariser", "ModelSummariser", "default_summariser"]

_SIGNAL = re.compile(
    r"\b(decid|chose|selected|because|therefore|error|failed|must|require|"
    r"conclusion|result|recommend|next step|blocked|risk)\w*\b",
    re.IGNORECASE,
)


class ExtractiveSummariser:
    """Deterministic, dependency-free conversation compaction.

    Ranks sentences by signal (decisions, failures, requirements) and recency,
    then emits a compact digest that fits the caller's token budget.  Because it
    is deterministic, executions stay reproducible and replayable.
    """

    def __init__(self, *, max_sentences: int = 12) -> None:
        self.max_sentences = max_sentences

    def __call__(self, messages: list[Message], token_budget: int) -> str:
        if not messages:
            return ""
        char_budget = max(120, token_budget * 4)
        scored: list[tuple[float, str]] = []

        for position, message in enumerate(messages):
            recency = (position + 1) / len(messages)
            label = _label(message)
            for sentence in _sentences(message.content or ""):
                if len(sentence) < 15:
                    continue
                signals = len(_SIGNAL.findall(sentence))
                score = recency + 1.5 * signals
                if message.role is Role.TOOL:
                    score += 0.5           # tool outcomes are usually load-bearing
                if sentence.rstrip().endswith("?"):
                    score += 0.4           # open questions matter
                # Drop filler: no signal, not a question, not recent.
                if signals == 0 and recency < 0.75 \
                        and not sentence.rstrip().endswith("?") \
                        and message.role is not Role.TOOL:
                    continue
                scored.append((score, f"{label}{sentence.strip()}"))

        if not scored:
            return f"{len(messages)} earlier messages omitted."

        scored.sort(key=lambda item: -item[0])
        chosen: list[str] = []
        used = 0
        for _score, line in scored[: self.max_sentences]:
            if used + len(line) > char_budget:
                break
            chosen.append(line)
            used += len(line)

        header = f"Summary of {len(messages)} earlier messages:"
        body = "\n".join(f"- {line}" for line in chosen)
        return f"{header}\n{body}" if body else header


class ModelSummariser:
    """Summarise with an LLM.  Costs a model call; use when fidelity matters.

    The summarising call is deliberately made outside the agent's own budget
    accounting path (it is infrastructure, not reasoning), but it still goes
    through the gateway, so it is traced and logged like everything else.
    """

    PROMPT = (
        "Compress the following conversation for an AI agent's working memory. "
        "Preserve decisions, constraints, tool results, open questions and "
        "unresolved errors. Drop pleasantries. Write terse bullet points."
    )

    def __init__(self, gateway: Any, context: Any, *, model: str | None = None,
                 fallback: ExtractiveSummariser | None = None) -> None:
        self.gateway = gateway
        self.context = context
        self.model = model
        self.fallback = fallback or ExtractiveSummariser()

    async def __call__(self, messages: list[Message], token_budget: int) -> str:
        from ..models.base import ModelRequest

        transcript = "\n".join(
            f"{_label(m)}{(m.content or '')[:1500]}" for m in messages)
        request = ModelRequest(
            messages=[Message.system(self.PROMPT), Message.user(transcript)],
            max_output_tokens=max(128, token_budget),
            temperature=0.0,
            metadata={"purpose": "context_summary"},
        )
        try:
            response = await self.gateway.call(self.context, request,
                                               model=self.model)
        except Exception:
            # Summarisation must never break an execution: degrade, don't fail.
            return self.fallback(messages, token_budget)
        return response.text or self.fallback(messages, token_budget)


def default_summariser() -> ExtractiveSummariser:
    """The summariser used when the developer configures nothing."""
    return ExtractiveSummariser()


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _label(message: Message) -> str:
    if message.role is Role.TOOL:
        return f"[tool:{message.name or 'result'}] "
    if message.role is Role.ASSISTANT:
        return "[assistant] "
    if message.role is Role.USER:
        return "[user] "
    return ""


def _sentences(text: str) -> list[str]:
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [part.strip() for part in parts if part.strip()]
