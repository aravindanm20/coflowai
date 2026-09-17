"""Subprocess sandbox: real OS-level isolation for untrusted tools.

`InProcessSandbox` gives you timeouts and the permission runtime, but a tool
still runs with the host process's privileges.  This sandbox executes the tool
in a separate interpreter with CPU, memory, and file-descriptor limits applied
via ``resource``, an optional scrubbed environment, and hard process kill on
timeout.

Requirements: the tool must be **importable by module path** and its arguments
and return value must be JSON-serialisable, because they cross a process
boundary. Closures and local functions cannot be sandboxed this way.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from typing import Any

from ..core.exceptions import ConfigurationError, ToolError, ToolTimeout
from .sandbox import Sandbox
from .tool import Tool

__all__ = ["SubprocessSandbox"]

#: Child-side bootstrap: import the tool, run it, print one JSON line.
_RUNNER = r"""
import asyncio, importlib, json, sys

def _limit(cpu, memory, files):
    try:
        import resource
    except ImportError:
        return
    if cpu:
        resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))
    if memory:
        resource.setrlimit(resource.RLIMIT_AS, (memory, memory))
    if files:
        resource.setrlimit(resource.RLIMIT_NOFILE, (files, files))

def main():
    request = json.loads(sys.stdin.read())
    _limit(request.get("cpu_seconds"), request.get("memory_bytes"),
           request.get("max_files"))
    module = importlib.import_module(request["module"])
    target = getattr(module, request["qualname"])
    func = getattr(target, "func", target)          # unwrap a Tool object
    result = func(**request["arguments"])
    if asyncio.iscoroutine(result):
        result = asyncio.run(result)
    sys.stdout.write("\x00RESULT\x00" + json.dumps({"ok": True, "value": result},
                                                   default=str))

try:
    main()
except BaseException as exc:
    sys.stdout.write("\x00RESULT\x00" + json.dumps(
        {"ok": False, "error": f"{type(exc).__name__}: {exc}"}))
"""

_MARKER = "\x00RESULT\x00"


class SubprocessSandbox(Sandbox):
    """Run tools in an isolated child interpreter with resource limits."""

    name = "subprocess"

    def __init__(self, *, cpu_seconds: int = 10, memory_mb: int = 512,
                 max_files: int = 64, max_concurrency: int = 4,
                 env_allowlist: tuple[str, ...] = ("PATH", "PYTHONPATH", "LANG",
                                                   "HOME", "TMPDIR"),
                 python_executable: str | None = None,
                 working_dir: str | None = None) -> None:
        self.cpu_seconds = cpu_seconds
        self.memory_bytes = memory_mb * 1024 * 1024
        self.max_files = max_files
        self.env_allowlist = env_allowlist
        self.python = python_executable or sys.executable
        self.working_dir = working_dir
        self._semaphore = asyncio.Semaphore(max_concurrency)

    # ------------------------------------------------------------------- run
    async def run(self, tool: Tool, arguments: dict[str, Any], *,
                  metadata: dict[str, Any] | None = None) -> Any:
        module, qualname = _locate(tool)
        payload = json.dumps({
            "module": module, "qualname": qualname, "arguments": arguments,
            "cpu_seconds": self.cpu_seconds, "memory_bytes": self.memory_bytes,
            "max_files": self.max_files,
        }, default=str)

        async with self._semaphore:
            process = await asyncio.create_subprocess_exec(
                self.python, "-c", _RUNNER,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=self._environment(),
                cwd=self.working_dir,
            )
            timeout = tool.definition.timeout or 30.0
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(payload.encode()), timeout=timeout)
            except TimeoutError as exc:
                await _terminate(process)
                raise ToolTimeout(
                    f"sandboxed tool '{tool.name}' exceeded {timeout}s",
                    cause=exc, tool=tool.name, sandbox=self.name) from exc

        return _parse(tool, process.returncode, stdout, stderr, self.name)

    def _environment(self) -> dict[str, str]:
        """Scrubbed environment — secrets are not inherited by tool processes."""
        env = {key: os.environ[key] for key in self.env_allowlist
               if key in os.environ}
        env.setdefault("PYTHONPATH", os.getcwd())
        env["COFLOWAI_SANDBOX"] = "1"
        return env


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _locate(tool: Tool) -> tuple[str, str]:
    func = tool.func
    module = getattr(func, "__module__", None)
    qualname = getattr(func, "__qualname__", None)
    if not module or not qualname or "<locals>" in qualname:
        raise ConfigurationError(
            f"tool '{tool.name}' cannot be sandboxed: it must be a module-level "
            "function (closures and locals cannot cross a process boundary)",
            tool=tool.name,
        )
    if module == "__main__":
        raise ConfigurationError(
            f"tool '{tool.name}' is defined in __main__; move it into an "
            "importable module to use the subprocess sandbox",
            tool=tool.name,
        )
    return module, qualname


def _parse(tool: Tool, returncode: int | None, stdout: bytes, stderr: bytes,
           sandbox: str) -> Any:
    text = stdout.decode("utf-8", "replace")
    marker = text.rfind(_MARKER)
    if marker == -1:
        detail = stderr.decode("utf-8", "replace")[-400:]
        if returncode and returncode < 0:
            raise ToolError(
                f"sandboxed tool '{tool.name}' was killed (signal {-returncode}) — "
                "it likely exceeded its CPU or memory limit",
                tool=tool.name, sandbox=sandbox, returncode=returncode)
        raise ToolError(f"sandboxed tool '{tool.name}' produced no result",
                        tool=tool.name, sandbox=sandbox, stderr=detail,
                        returncode=returncode)
    try:
        outcome = json.loads(text[marker + len(_MARKER):])
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive
        raise ToolError(f"sandboxed tool '{tool.name}' returned invalid JSON",
                        cause=exc, tool=tool.name) from exc
    if not outcome.get("ok"):
        raise ToolError(f"sandboxed tool '{tool.name}' failed: "
                        f"{outcome.get('error')}", tool=tool.name, sandbox=sandbox)
    return outcome.get("value")


async def _terminate(process: Any) -> None:
    """Kill the child and reap it so no zombie is left behind."""
    if process.returncode is not None:
        return
    process.kill()
    try:
        await asyncio.wait_for(process.wait(), timeout=5)
    except TimeoutError:  # pragma: no cover - extremely unlikely
        pass
