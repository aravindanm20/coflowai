#!/usr/bin/env python
"""Verify that Python snippets in the docs match the real CoFlowAi API.

For every ```python block in README.md and docs/*.md this checks that:
  * the block parses (top-level ``await`` is allowed),
  * every ``from coflowai... import X`` resolves,
  * every call to a CoFlowAi class/function passes parameters it accepts and
    supplies its required ones,
  * methods/attributes used on variables bound to CoFlowAi objects exist,
  * string literals used as event types are real ``EventType`` values.

Imports carry across the blocks of one page, as for a reader going top to bottom.
Prefix a block with ``<!-- docs-check: skip -->`` to exempt pseudo-code.
Exit status is non-zero when any problem is found, so this runs in CI.
"""
from __future__ import annotations

import ast
import importlib
import inspect
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
FILES = [ROOT / "README.md", *sorted((ROOT / "docs").glob("*.md"))]
BLOCK = re.compile(r"(<!--\s*docs-check:\s*skip\s*-->\s*\n)?```python\n(.*?)```", re.S)
FLAGS = ast.PyCF_ONLY_AST | ast.PyCF_ALLOW_TOP_LEVEL_AWAIT
EVENT_NAME = re.compile(
    r"^(execution|agent|model|tool|checkpoint|approval|budget|workflow|state)"
    r"\.[a-z_.]+$")
EVENT_SINKS = {"on_event", "subscribe", "get_events", "emit"}


def _accepts(fn: Any, kwargs: list[str], npos: int) -> str | None:
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return None
    params = list(sig.parameters.values())
    if params and params[0].name in ("self", "cls"):
        params = params[1:]
    var_kw = any(p.kind is p.VAR_KEYWORD for p in params)
    names = {p.name for p in params
             if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)}
    for kw in kwargs:
        if kw not in names and not var_kw:
            accepted = ", ".join(sorted(names)) or "none"
            return f"unexpected keyword '{kw}' (accepts: {accepted})"
    positional = [p for p in params if p.kind in (p.POSITIONAL_ONLY,
                                                   p.POSITIONAL_OR_KEYWORD)]
    if npos > len(positional) and not any(p.kind is p.VAR_POSITIONAL for p in params):
        return f"takes at most {len(positional)} positional argument(s), got {npos}"
    filled = {p.name for p in positional[:npos]} | set(kwargs)
    missing = [p.name for p in params
               if p.default is p.empty and p.name not in filled
               and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD,
                              p.KEYWORD_ONLY)]
    if missing:
        return f"missing required argument(s): {', '.join(missing)}"
    return None


def _callable_of(obj: Any) -> Any:
    return obj.__init__ if inspect.isclass(obj) else obj


_ATTR_CACHE: dict[type, set[str]] = {}


def _instance_attrs(cls: type) -> set[str]:
    """Attributes assigned as ``self.x = ...`` anywhere in the class MRO."""
    if cls not in _ATTR_CACHE:
        names: set[str] = set()
        for base in cls.__mro__:
            try:
                src = inspect.getsource(base)
            except (OSError, TypeError):
                continue
            names.update(re.findall(r"self\.(\w+)\s*(?::[^=\n]+)?=", src))
        _ATTR_CACHE[cls] = names
    return _ATTR_CACHE[cls]


def _is_dataclass_field(cls: type, name: str) -> bool:
    ann: dict[str, Any] = {}
    for base in cls.__mro__:
        ann.update(getattr(base, "__annotations__", {}))
    return name in (getattr(cls, "__dataclass_fields__", {}) or {}) or name in ann


def _has_attr(cls: type, name: str) -> bool:
    return (hasattr(cls, name) or _is_dataclass_field(cls, name)
            or name in _instance_attrs(cls) or "__getattr__" in vars(cls))


def _qual(obj: Any) -> str:
    return getattr(obj, "__qualname__", None) or getattr(obj, "__name__", repr(obj))


class Checker(ast.NodeVisitor):
    def __init__(self) -> None:
        self.names: dict[str, Any] = {}       # local name -> coflowai object
        self.instances: dict[str, type] = {}  # variable -> coflowai class
        self.errors: list[tuple[int, str]] = []

    def err(self, node: ast.AST, msg: str) -> None:
        self.errors.append((getattr(node, "lineno", 0), msg))

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            if alias.name.split(".")[0] != "coflowai":
                continue
            try:
                mod = importlib.import_module(alias.name)
            except Exception as exc:
                self.err(node, f"cannot import {alias.name}: {exc}")
                continue
            self.names[(alias.asname or alias.name).split(".")[0]] = (
                mod if alias.asname else importlib.import_module("coflowai"))

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if not node.module or not node.module.startswith("coflowai"):
            return
        try:
            mod = importlib.import_module(node.module)
        except Exception as exc:
            self.err(node, f"cannot import module {node.module}: {exc}")
            return
        for alias in node.names:
            if hasattr(mod, alias.name):
                obj = getattr(mod, alias.name)
            else:
                try:
                    obj = importlib.import_module(f"{node.module}.{alias.name}")
                except Exception:
                    self.err(node, f"{node.module} has no '{alias.name}'")
                    continue
            self.names[alias.asname or alias.name] = obj

    def resolve(self, node: ast.AST) -> Any:
        if isinstance(node, ast.Name):
            return self.names.get(node.id)
        if isinstance(node, ast.Attribute):
            base = self.resolve(node.value)
            if base is None or not (inspect.ismodule(base) or inspect.isclass(base)):
                return None
            if hasattr(base, node.attr):
                return getattr(base, node.attr)
            if inspect.ismodule(base):
                try:
                    return importlib.import_module(f"{base.__name__}.{node.attr}")
                except Exception:
                    pass
            if not (inspect.isclass(base) and _is_dataclass_field(base, node.attr)):
                self.err(node, f"{_qual(base)} has no attribute '{node.attr}'")
        return None

    def visit_Attribute(self, node: ast.Attribute) -> None:
        self.generic_visit(node)
        if isinstance(node.value, ast.Name) and node.value.id in self.instances \
                and isinstance(node.ctx, ast.Load):
            cls = self.instances[node.value.id]
            if not _has_attr(cls, node.attr):
                self.err(node, f"{cls.__name__} has no attribute '{node.attr}'")

    def visit_Assign(self, node: ast.Assign) -> None:
        self.generic_visit(node)
        value = node.value.value if isinstance(node.value, ast.Await) else node.value
        cls = None
        if isinstance(value, ast.Call):
            target = self.resolve(value.func)
            if inspect.isclass(target):
                cls = target
        for t in node.targets:
            if isinstance(t, ast.Name):
                if cls is not None:
                    self.instances[t.id] = cls
                else:
                    self.instances.pop(t.id, None)

    def visit_Call(self, node: ast.Call) -> None:
        self.generic_visit(node)
        if any(isinstance(a, ast.Starred) for a in node.args) or \
                any(k.arg is None for k in node.keywords):
            return
        kwargs = [k.arg for k in node.keywords if k.arg]
        target = self.resolve(node.func)
        if target is not None and callable(target):
            problem = _accepts(_callable_of(target), kwargs, len(node.args))
            if problem:
                self.err(node, f"{_qual(target)}(): {problem}")
            return
        func = node.func
        if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name) \
                and func.value.id in self.instances:
            cls = self.instances[func.value.id]
            if not hasattr(cls, func.attr):
                return          # reported by visit_Attribute
            attr = inspect.getattr_static(cls, func.attr)
            if isinstance(attr, (staticmethod, classmethod)):
                attr = attr.__func__
            if callable(attr):
                problem = _accepts(attr, kwargs, len(node.args))
                if problem:
                    self.err(node, f"{cls.__name__}.{func.attr}(): {problem}")


def _event_types() -> set[str]:
    from coflowai.events.event import EventType
    return {v for k, v in vars(EventType).items()
            if not k.startswith("_") and isinstance(v, str)}


def _event_name_literals(tree: ast.AST) -> list[ast.Constant]:
    """String literals used *as event types*: arguments of on_event/emit/...,
    or compared against something named ``type``/``event_type``."""
    found: list[ast.Constant] = []

    def strings(node: ast.AST) -> None:
        found.extend(n for n in ast.walk(node)
                     if isinstance(n, ast.Constant) and isinstance(n.value, str))

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = getattr(node.func, "attr", getattr(node.func, "id", ""))
            if name in EVENT_SINKS:
                for arg in node.args:
                    strings(arg)
        elif isinstance(node, ast.Compare):
            parts = [node.left, *node.comparators]
            if any(getattr(p, "attr", getattr(p, "id", "")) in ("type", "event_type")
                   for p in parts):
                for p in parts:
                    strings(p)
    return found


def check_file(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    problems = []
    known_events = _event_types()
    checker = Checker()
    for match in BLOCK.finditer(text):
        if match.group(1):
            continue
        code = match.group(2)
        line0 = text[: match.start(2)].count("\n") + 1
        try:
            tree = compile(code, str(path), "exec", flags=FLAGS)
        except SyntaxError as exc:
            problems.append(f"{path.relative_to(ROOT)}:{line0 + (exc.lineno or 1) - 1}: "
                            f"syntax error: {exc.msg}")
            continue
        checker.errors = []
        checker.visit(tree)
        for node in _event_name_literals(tree):
            if EVENT_NAME.match(node.value) and node.value not in known_events \
                    and not node.value.endswith(".*"):
                checker.err(node, f"unknown event type '{node.value}'")
        for lineno, msg in checker.errors:
            problems.append(f"{path.relative_to(ROOT)}:{line0 + lineno - 1}: {msg}")
    return problems


def main() -> int:
    sys.path.insert(0, str(ROOT))
    problems = [p for f in FILES if f.exists() for p in check_file(f)]
    for p in problems:
        print(p)
    print(f"\n{len(problems)} problem(s) in {len(FILES)} file(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
