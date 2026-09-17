"""``coflowai`` command line interface.

The CLI loads a Python module that exposes an ``app`` (``CoFlowAi``) and
optionally ``workflow`` / ``agent`` objects, then runs, inspects, replays or
resumes executions against it.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

from . import __version__
from .app import CoFlowAi
from .observability.logger import configure_logging
from .workflows.workflow import Workflow

__all__ = ["main", "build_parser"]


def _load_module(path: str) -> Any:
    file = Path(path).resolve()
    if not file.exists():
        raise SystemExit(f"file not found: {file}")
    spec = importlib.util.spec_from_file_location(file.stem, file)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise SystemExit(f"cannot import {file}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[file.stem] = module
    spec.loader.exec_module(module)
    return module


def _find_app(module: Any) -> CoFlowAi:
    app = getattr(module, "app", None)
    if isinstance(app, CoFlowAi):
        return app
    for value in vars(module).values():
        if isinstance(value, CoFlowAi):
            return value
    raise SystemExit("module does not expose a CoFlowAi instance named 'app'")


def _find_target(module: Any, name: str | None) -> Any:
    if name:
        target = getattr(module, name, None)
        if target is None:
            raise SystemExit(f"'{name}' not found in module")
        return target
    for attribute in ("workflow", "agent", "main_workflow", "target"):
        target = getattr(module, attribute, None)
        if target is not None:
            return target
    raise SystemExit("no runnable target found (expected 'workflow' or 'agent')")


def _print(data: Any) -> None:
    print(json.dumps(data, indent=2, default=str))


async def cmd_run(args: argparse.Namespace) -> int:
    module = _load_module(args.file)
    app = _find_app(module)
    target = _find_target(module, args.target)
    await app.setup()
    result = await app.run(target, args.input)
    _print(result.to_dict())
    if args.trace:
        tracer = app.trace(result.execution_id)
        if tracer:
            print(tracer.render(), file=sys.stderr)
    return 0 if result.succeeded else 1


async def cmd_executions_list(args: argparse.Namespace) -> int:
    app = _find_app(_load_module(args.file))
    records = await app.runtime.list_executions(status=args.status, limit=args.limit)
    _print([r.to_dict() for r in records])
    return 0


async def cmd_executions_inspect(args: argparse.Namespace) -> int:
    app = _find_app(_load_module(args.file))
    record = await app.runtime.get_execution(args.execution_id)
    if record is None:
        raise SystemExit(f"unknown execution {args.execution_id}")
    checkpoints = await app.runtime.state_store.list_checkpoints(args.execution_id)
    events = await app.runtime.event_store.get_events(args.execution_id)
    _print({
        "execution": record.to_dict(),
        "checkpoints": [c.to_dict() for c in checkpoints],
        "events": [e.to_dict() for e in events],
    })
    return 0


async def cmd_executions_replay(args: argparse.Namespace) -> int:
    app = _find_app(_load_module(args.file))
    trace = await app.replay(args.execution_id)
    if args.json:
        _print(trace.to_dict())
    else:
        print(trace.render())
    return 0


async def cmd_executions_resume(args: argparse.Namespace) -> int:
    module = _load_module(args.file)
    app = _find_app(module)
    target = _find_target(module, args.target)
    result = await app.resume(args.execution_id, executable=target)
    _print(result.to_dict())
    return 0 if result.succeeded else 1


async def cmd_executions_approve(args: argparse.Namespace) -> int:
    module = _load_module(args.file)
    app = _find_app(module)
    target = _find_target(module, args.target)
    action = app.reject if args.reject else app.approve
    result = await action(args.execution_id, approval_id=args.approval_id,
                          by=args.by, note=args.note, executable=target)
    _print(result.to_dict())
    return 0 if result.succeeded else 1


async def cmd_worker(args: argparse.Namespace) -> int:
    """Run a worker against a module's queue (module must expose `service`)."""
    from .distributed import Worker, WorkerPool

    module = _load_module(args.file)
    app = _find_app(module)
    service = getattr(module, "service", None)
    if service is None:
        raise SystemExit("module does not expose an ExecutionService named "
                         "'service'")
    await app.setup()
    if args.concurrency > 1:
        pool = WorkerPool(app.runtime, service.queue, service.executables,
                          size=args.concurrency)
        await pool.start()
        try:
            await asyncio.Event().wait()
        except (KeyboardInterrupt, asyncio.CancelledError):  # pragma: no cover
            pass
        finally:
            await pool.stop()
        return 0

    worker = Worker(app.runtime, service.queue, service.executables,
                    name=args.name)
    if args.once:
        job = await worker.run_once()
        _print({"processed": worker.processed, "job": job.to_dict() if job else None})
        return 0
    try:  # pragma: no cover - long-running
        await worker.run_forever()
    except KeyboardInterrupt:
        worker.stop()
    return 0


async def cmd_workflows_validate(args: argparse.Namespace) -> int:
    module = _load_module(args.file)
    app = _find_app(module)
    problems = 0
    for name, value in vars(module).items():
        if not isinstance(value, Workflow):
            continue
        try:
            graph = value.compile(model_registry=app.runtime.models, force=True)
            print(f"OK   {name}: {graph.render()}")
        except Exception as exc:
            problems += 1
            print(f"FAIL {name}: {exc}", file=sys.stderr)
    return 1 if problems else 0


async def cmd_models_list(args: argparse.Namespace) -> int:
    app = _find_app(_load_module(args.file))
    _print(app.models.list())
    return 0


async def cmd_tools_list(args: argparse.Namespace) -> int:
    app = _find_app(_load_module(args.file))
    _print(app.tools.list())
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="coflowai",
                                     description="CoFlowAi execution CLI")
    parser.add_argument("--version", action="version",
                        version=f"coflowai {__version__}")
    parser.add_argument("--log-level", default="INFO")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run an agent or workflow")
    run.add_argument("file")
    run.add_argument("--input", default=None)
    run.add_argument("--target", default=None)
    run.add_argument("--trace", action="store_true")
    run.set_defaults(func=cmd_run)

    executions = sub.add_parser("executions", help="inspect executions")
    ex_sub = executions.add_subparsers(dest="subcommand", required=True)

    listing = ex_sub.add_parser("list")
    listing.add_argument("file")
    listing.add_argument("--status", default=None)
    listing.add_argument("--limit", type=int, default=50)
    listing.set_defaults(func=cmd_executions_list)

    inspect = ex_sub.add_parser("inspect")
    inspect.add_argument("file")
    inspect.add_argument("execution_id")
    inspect.set_defaults(func=cmd_executions_inspect)

    replay = ex_sub.add_parser("replay")
    replay.add_argument("file")
    replay.add_argument("execution_id")
    replay.add_argument("--json", action="store_true")
    replay.set_defaults(func=cmd_executions_replay)

    resume = ex_sub.add_parser("resume")
    resume.add_argument("file")
    resume.add_argument("execution_id")
    resume.add_argument("--target", default=None)
    resume.set_defaults(func=cmd_executions_resume)

    approve = ex_sub.add_parser("approve")
    approve.add_argument("file")
    approve.add_argument("execution_id")
    approve.add_argument("--approval-id", default=None)
    approve.add_argument("--by", default=None)
    approve.add_argument("--note", default=None)
    approve.add_argument("--reject", action="store_true")
    approve.add_argument("--target", default=None)
    approve.set_defaults(func=cmd_executions_approve)

    worker = sub.add_parser("worker", help="run a queue worker")
    worker.add_argument("file")
    worker.add_argument("--name", default="worker-1")
    worker.add_argument("--concurrency", type=int, default=1)
    worker.add_argument("--once", action="store_true",
                        help="process a single job then exit")
    worker.set_defaults(func=cmd_worker)

    workflows = sub.add_parser("workflows", help="workflow tooling")
    wf_sub = workflows.add_subparsers(dest="subcommand", required=True)
    validate = wf_sub.add_parser("validate")
    validate.add_argument("file")
    validate.set_defaults(func=cmd_workflows_validate)

    models = sub.add_parser("models")
    models_sub = models.add_subparsers(dest="subcommand", required=True)
    models_list = models_sub.add_parser("list")
    models_list.add_argument("file")
    models_list.set_defaults(func=cmd_models_list)

    tools = sub.add_parser("tools")
    tools_sub = tools.add_subparsers(dest="subcommand", required=True)
    tools_list = tools_sub.add_parser("list")
    tools_list.add_argument("file")
    tools_list.set_defaults(func=cmd_tools_list)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging(args.log_level)
    return asyncio.run(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
