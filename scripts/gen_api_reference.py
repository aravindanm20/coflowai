#!/usr/bin/env python
"""Generate docs/API_REFERENCE.md from the code itself.

The previous hand-written reference drifted from the implementation
(``dsn=`` stores, ``RetryPolicy(max_retries=...)``, ``Worker(service=...)``,
event names that don't exist...).  Signatures and summaries here come straight
from ``inspect``, so they can't drift: CI runs ``--check`` and fails if the
committed file differs from what this script generates.

    python scripts/gen_api_reference.py          # rewrite the file
    python scripts/gen_api_reference.py --check  # exit 1 if out of date
"""
from __future__ import annotations

import importlib
import inspect
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "API_REFERENCE.md"
sys.path.insert(0, str(ROOT))

SECTIONS: list[tuple[str, str, list[str]]] = [
    ("Application", "coflowai", ["CoFlowAi", "Runtime"]),
    ("Agents", "coflowai", ["Agent"]),
    ("Workflows", "coflowai", ["Workflow", "SequenceNode", "ParallelNode",
                               "RouterNode", "LoopNode", "TransformNode",
                               "HumanApproval"]),
    ("Tools", "coflowai", ["tool", "Tool", "ToolDefinition", "ToolRegistry",
                           "InProcessSandbox", "ThreadSandbox", "SubprocessSandbox"]),
    ("Policies", "coflowai", ["ExecutionPolicy", "RetryPolicy", "Budget",
                              "PermissionSet", "ExecutionMode"]),
    ("Models", "coflowai", ["ModelProvider", "ModelRequest", "ModelResponse",
                            "ModelCapabilities", "ModelPricing", "ModelRegistry",
                            "StreamChunk", "ChunkType"]),
    ("Providers", "coflowai.providers.openai", ["OpenAIProvider", "AzureOpenAIProvider"]),
    ("Providers", "coflowai.providers.anthropic", ["AnthropicProvider"]),
    ("Providers", "coflowai.providers.gemini", ["GeminiProvider"]),
    ("Providers", "coflowai.providers.bedrock", ["BedrockProvider"]),
    ("Providers", "coflowai.providers.http", ["HttpxTransport", "MockTransport"]),
    ("Context & memory", "coflowai", ["ContextBuilder", "ContextBudget",
                                      "ExtractiveSummariser", "ModelSummariser",
                                      "MemoryStore", "InMemoryMemoryStore",
                                      "MemoryRecord", "MemoryKind"]),
    ("Events", "coflowai", ["Event", "EventStore", "InMemoryEventStore",
                            "JsonlEventStore"]),
    ("Persistence", "coflowai.persistence.sqlite",
     ["SqliteEventStore", "SqliteStateStore"]),
    ("Persistence", "coflowai.persistence.postgres", ["create_pool", "PostgresEventStore",
                                                      "PostgresStateStore"]),
    ("Persistence", "coflowai.persistence.redis", ["RedisEventStore", "RedisStateStore"]),
    ("Distributed", "coflowai.distributed", ["ExecutionService", "ExecutableRegistry",
                                             "Worker", "WorkerPool", "Job",
                                             "InMemoryJobQueue", "RedisJobQueue"]),
    ("Observability", "coflowai.observability", ["configure_logging", "get_logger",
                                                 "Tracer", "UsageTracker", "redact"]),
    ("Observability", "coflowai.observability.redaction", ["add_redaction_pattern",
                                                           "add_redaction_key"]),
    ("Results", "coflowai", ["ExecutionResult", "ExecutionStatus", "Usage",
                             "ExecutionContext"]),
    ("Plugins", "coflowai", ["Plugin"]),
    ("Testing", "coflowai.testing", ["build_test_app", "FakeModelProvider", "FakeTool",
                                     "tool_call", "AgentTestHarness"]),
]


def _summary(obj: Any) -> str:
    doc = inspect.getdoc(obj) or ""
    return doc.split("\n\n")[0].replace("\n", " ").strip()


def _ann(annotation: Any) -> str:
    if annotation is inspect.Parameter.empty:
        return ""
    text = (annotation if isinstance(annotation, str)
            else inspect.formatannotation(annotation))
    return text.strip("'\"")


def _default(value: Any) -> str:
    text = repr(value)
    if text.startswith("<"):           # <factory>, <Enum.MEMBER: 'x'>, objects
        member = getattr(value, "name", None)
        return f"{type(value).__name__}.{member}" if member else "..."
    return text


def _params(obj: Any, *, keep_self: bool) -> tuple[list[str], str]:
    sig = inspect.signature(obj)
    parts: list[str] = []
    star = False
    for p in sig.parameters.values():
        if p.name in ("self", "cls") and not keep_self:
            continue
        ann = _ann(p.annotation)
        if p.kind is p.VAR_POSITIONAL:
            star = True
            parts.append(f"*{p.name}" + (f": {ann}" if ann else ""))
            continue
        if p.kind is p.VAR_KEYWORD:
            parts.append(f"**{p.name}" + (f": {ann}" if ann else ""))
            continue
        if p.kind is p.KEYWORD_ONLY and not star:
            parts.append("*")
            star = True
        text = p.name + (f": {ann}" if ann else "")
        if p.default is not inspect.Parameter.empty:
            text += (" = " if ann else "=") + _default(p.default)
        parts.append(text)
    ret = _ann(sig.return_annotation)
    return parts, (f" -> {ret}" if ret else "")


def _sig(obj: Any, name: str, *, keep_self: bool = False) -> str:
    try:
        parts, ret = _params(obj, keep_self=keep_self)
    except (TypeError, ValueError):
        return f"{name}(*args, **kwargs)"
    one_line = f"{name}({', '.join(parts)}){ret}"
    if len(one_line) <= 84 or not parts:
        return one_line
    body = "".join(f"    {p},\n" for p in parts)
    return f"{name}(\n{body}){ret}"


def _public_members(cls: type) -> list[tuple[str, Any, Any]]:
    out = []
    for name, raw in cls.__dict__.items():
        if name.startswith("_"):
            continue
        func = raw.__func__ if isinstance(raw, (staticmethod, classmethod)) else raw
        if isinstance(raw, property) or inspect.isfunction(func):
            out.append((name, raw, func))
    return out


def _stub(name: str, obj: Any, *, method: bool = False) -> str:
    prefix = "async def " if (inspect.iscoroutinefunction(obj)
                              or inspect.isasyncgenfunction(obj)) else "def "
    return f"{prefix}{_sig(obj, name, keep_self=method)}: ..."


def render_object(name: str, obj: Any) -> list[str]:
    lines = [f"### `{name}`", ""]
    summary = _summary(obj)
    if summary:
        lines += [summary, ""]
    if not inspect.isclass(obj):
        return lines + ["```python", _stub(name, obj), "```", ""]
    if issubclass(obj, BaseException):
        return lines
    if hasattr(obj, "__members__"):  # Enum
        lines += ["| Member | Value |", "|---|---|"]
        lines += [f"| `{m}` | `{v.value}` |" for m, v in obj.__members__.items()]
        return lines + [""]
    code = [f"class {name}:"]
    init = obj.__dict__.get("__init__")
    if init is not None:
        sig = _sig(init, "__init__", keep_self=True)
        code.append(f"    def {sig}: ...".replace("\n", "\n    "))
    elif getattr(obj, "__dataclass_fields__", None):
        sig = _sig(obj, "__init__")
        sig = "__init__(self)" if sig.startswith("__init__()") else \
            sig.replace("__init__(\n", "__init__(\n    self,\n", 1) if "\n" in sig \
            else sig.replace("__init__(", "__init__(self, ", 1)
        code.append(f"    def {sig}: ...".replace("\n", "\n    "))
    props, methods, docs = [], [], []
    for mname, raw, func in _public_members(obj):
        if isinstance(raw, property):
            props.append(f"    @property\n    def {mname}(self): ...")
            doc = _summary(raw.fget)
        else:
            deco = ("    @staticmethod\n" if isinstance(raw, staticmethod) else
                    "    @classmethod\n" if isinstance(raw, classmethod) else "")
            methods.append(deco + "    " + _stub(
                mname, func, method=not isinstance(raw, staticmethod)
            ).replace("\n", "\n    "))
            doc = _summary(func)
        if doc:
            docs.append(f"- **`{mname}`** — {doc}")
    code += props + methods
    if len(code) == 1:
        code.append("    ...")
    lines += ["```python", *code, "```", ""]
    if docs:
        lines += docs + [""]
    return lines


def render() -> str:
    from coflowai import __version__
    from coflowai.core import exceptions as exc_mod
    from coflowai.events.event import EventType

    out = ["# API Reference", "",
           "<!-- GENERATED by scripts/gen_api_reference.py -- do not edit by hand. -->",
           "",
           f"Reference for CoFlowAi {__version__}, generated from the source code. "
           "For explanations and examples see [CORE_CONCEPTS.md](CORE_CONCEPTS.md) "
           "and [USER_GUIDE.md](USER_GUIDE.md).", "", "## Contents", ""]
    sections: list[str] = []
    for section, _mod, _names in SECTIONS:
        if section not in sections:
            sections.append(section)
    sections += ["Event types", "Exceptions"]
    out += [f"- [{s}](#{s.lower().replace(' ', '-').replace('&', '')})"
            for s in sections] + [""]
    current = None
    for section, module_name, names in SECTIONS:
        module = importlib.import_module(module_name)
        if section != current:
            out += ["---", "", f"## {section}", ""]
            current = section
        out += [f"`from {module_name} import {', '.join(names)}`", ""]
        for name in names:
            out += render_object(name, getattr(module, name))
    out += ["---", "", "## Event types", "",
            "Every event has `execution_id`, `type`, `sequence` (per execution), "
            "`step`, `node_id`, `trace_id`, `timestamp` and a `payload`. "
            "Subscribe with `app.on_event(callback, event_type=...)`.", "",
            "| Constant | Value |", "|---|---|"]
    for key, value in sorted(vars(EventType).items(), key=lambda kv: str(kv[1])):
        if not key.startswith("_") and isinstance(value, str):
            out.append(f"| `EventType.{key}` | `\"{value}\"` |")
    out += ["", "---", "", "## Exceptions", "",
            "All importable from `coflowai` and `coflowai.core`.", "", "```text"]
    classes = [c for c in vars(exc_mod).values()
               if inspect.isclass(c) and issubclass(c, exc_mod.CoFlowAiError)]

    def tree(base: type, depth: int) -> None:
        out.append("    " * depth + base.__name__)
        for sub in sorted((c for c in classes if c.__bases__[0] is base),
                          key=lambda c: c.__name__):
            tree(sub, depth + 1)

    tree(exc_mod.CoFlowAiError, 0)
    out += ["```", "", "`app.run()` does not raise for execution failures: it returns "
            "an `ExecutionResult` whose `status` and `error` describe what happened "
            "(call `result.raise_for_status()` to opt into exceptions). "
            "`is_retryable(error)` decides whether the runtime retries; permission, "
            "budget, step-limit, approval, cancellation, compilation, configuration "
            "and timeout errors are never retried.", ""]
    return "\n".join(out)


def main() -> int:
    text = render()
    if "--check" in sys.argv:
        current = OUT.read_text(encoding="utf-8") if OUT.exists() else ""
        if current != text:
            print("docs/API_REFERENCE.md is out of date; run "
                  "python scripts/gen_api_reference.py")
            return 1
        print("docs/API_REFERENCE.md is up to date")
        return 0
    OUT.write_text(text, encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)} ({len(text.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
