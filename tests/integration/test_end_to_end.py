"""End-to-end scenarios, plugin wiring, CLI and the V0.1 definition of done."""

from __future__ import annotations

import json

import pytest
from pydantic import BaseModel

from coflowai import (
    Agent,
    CoFlowAi,
    ExecutionPolicy,
    ExecutionStatus,
    InMemoryMemoryStore,
    Plugin,
    Workflow,
    tool,
)
from coflowai.testing import build_test_app, tool_call
from coflowai.testing.fakes import FakeModelProvider


class Report(BaseModel):
    summary: str
    confidence: float


@tool(description="Search the web", permission="network.external")
async def search_web(query: str) -> list[str]:
    return [f"result for {query}"]


@tool(description="Query the warehouse", permission="database.read")
async def query_database(sql: str) -> dict:
    return {"rows": 2}


async def test_full_research_workflow_with_tools_parallelism_and_schema():
    app = CoFlowAi(policy=ExecutionPolicy(max_steps=20, max_model_calls=15,
                                          max_tool_calls=25, max_cost=3.0,
                                          timeout_seconds=30),
                   permissions=["network.external", "database.read"],
                   memory=InMemoryMemoryStore())
    provider = FakeModelProvider([
        "plan: investigate",                                   # planner
        tool_call("search_web", query="agent frameworks"),     # web researcher
        "web findings",
        tool_call("query_database", sql="select 1"),           # db researcher
        "db findings",
        {"summary": "agent frameworks are converging", "confidence": 0.82},
    ])
    app.models.register("smart-model", provider, default=True)
    app.tools.register(search_web, query_database)

    planner = Agent("planner", "Plan the research.", "smart-model")
    web = Agent("web", "Research on the web.", "smart-model", [search_web])
    database = Agent("database", "Research in the warehouse.", "smart-model",
                     [query_database])
    analyst = Agent("analyst", "Summarise findings.", "smart-model",
                    output_schema=Report)

    workflow = (Workflow("research", version="2")
                .start(planner)
                .parallel(web, database, max_concurrency=2)
                .then(analyst))

    result = await app.run(workflow, "Research AI agent frameworks")

    assert result.succeeded
    assert isinstance(result.output, Report)
    assert result.output.confidence == 0.82
    assert result.usage.model_calls == 6 and result.usage.tool_calls == 2
    assert result.usage.cost > 0
    assert result.metadata["workflow"] == "research"

    trace = app.trace(result.execution_id)
    rendered = trace.render()
    assert "workflow:research" in rendered and "tool:search_web" in rendered

    replay = await app.replay(result.execution_id)
    assert replay.status == "completed"
    assert len(replay.tool_calls()) == 2


async def test_plugin_registers_models_and_tools():
    class DemoPlugin(Plugin):
        name = "demo"
        version = "1.0.0"

        async def setup(self, app: CoFlowAi) -> None:
            app.models.register("plugin-model", FakeModelProvider(["from plugin"]),
                                default=True)
            app.tools.register(search_web)

    app = CoFlowAi()
    app.register_plugin(DemoPlugin())
    await app.setup()

    result = await app.run(Agent("a", "answer", "plugin-model"), "hi")
    assert result.output == "from plugin"
    assert "search_web" in app.tools.names()
    await app.shutdown()


async def test_failing_plugin_is_reported_clearly():
    from coflowai import PluginError

    class BrokenPlugin(Plugin):
        name = "broken"

        async def setup(self, app: CoFlowAi) -> None:
            raise RuntimeError("no credentials")

    app = CoFlowAi()
    app.register_plugin(BrokenPlugin())
    with pytest.raises(PluginError):
        await app.setup()


async def test_event_subscribers_receive_the_stream():
    seen: list[str] = []
    app, _ = build_test_app(["ok"])
    app.on_event(lambda event: seen.append(event.type))
    await app.run(Agent("a", "answer", "fake-model"), "hi")
    assert "execution.completed" in seen


async def test_subscriber_failure_never_breaks_execution():
    def boom(event):
        raise RuntimeError("subscriber is broken")

    app, _ = build_test_app(["ok"])
    app.on_event(boom)
    result = await app.run(Agent("a", "answer", "fake-model"), "hi")
    assert result.succeeded


def test_cli_runs_a_module(tmp_path, capsys):
    from coflowai.cli import main

    script = tmp_path / "app_module.py"
    script.write_text(
        "from coflowai import Agent\n"
        "from coflowai.testing import build_test_app\n"
        "app, _model = build_test_app(['cli output'])\n"
        "agent = Agent('assistant', 'Answer.', 'fake-model')\n"
    )
    assert main(["run", str(script), "--input", "hello"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "completed"
    assert payload["output"] == "cli output"


def test_cli_validates_workflows(tmp_path, capsys):
    from coflowai.cli import main

    script = tmp_path / "wf_module.py"
    script.write_text(
        "from coflowai import Agent, Workflow\n"
        "from coflowai.testing import build_test_app\n"
        "app, _model = build_test_app(['x'])\n"
        "workflow = Workflow('demo').start(Agent('a', 'i', 'fake-model'))\n"
    )
    assert main(["workflows", "validate", str(script)]) == 0
    assert "OK" in capsys.readouterr().out


def test_cli_lists_models_and_tools(tmp_path, capsys):
    from coflowai.cli import main

    script = tmp_path / "list_module.py"
    script.write_text(
        "from coflowai import Agent, tool\n"
        "from coflowai.testing import build_test_app\n"
        "@tool\n"
        "async def ping() -> str:\n"
        "    return 'pong'\n"
        "app, _model = build_test_app(['x'], tools=[ping])\n"
        "agent = Agent('a', 'i', 'fake-model')\n"
    )
    assert main(["models", "list", str(script)]) == 0
    assert "fake-model" in capsys.readouterr().out
    assert main(["tools", "list", str(script)]) == 0
    assert "ping" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# §96 — definition of done for V0.1
# --------------------------------------------------------------------------- #
async def test_definition_of_done_v01():
    app, model = build_test_app([tool_call("search_web", query="x"), "final answer"],
                                tools=[search_web], permissions=["network.external"])
    agent = Agent("assistant", "Answer.", "fake-model", [search_web])

    first = await app.run(agent, "question")
    model.reset()
    model.responses = [tool_call("search_web", query="x"), "final answer"]
    second = await app.run(agent, "question")

    # deterministic with the fake provider
    assert first.output == second.output == "final answer"
    assert first.execution_id != second.execution_id
    assert first.usage.to_dict()["total_tokens"] == second.usage.to_dict()[
        "total_tokens"]

    events = [e.type for e in
              await app.runtime.event_store.get_events(first.execution_id)]
    for required in ("execution.started", "agent.started", "model.requested",
                     "tool.completed", "execution.completed"):
        assert required in events

    assert first.status is ExecutionStatus.COMPLETED


def test_core_package_has_no_provider_dependencies():
    """Rule 3 / §3: the *core* must not import any provider or infra SDK.

    Adapters under ``providers/`` and ``persistence/`` are explicitly allowed to
    — that is their entire job — but nothing else may.
    """
    import pathlib

    banned = ("import openai", "import anthropic", "import boto3", "import redis",
              "import asyncpg", "import qdrant_client", "from openai",
              "from anthropic", "import pinecone")
    root = pathlib.Path(__file__).resolve().parents[2] / "coflowai"
    adapter_dirs = {"providers", "persistence"}

    offenders = []
    for path in root.rglob("*.py"):
        if adapter_dirs & set(path.relative_to(root).parts):
            continue
        text = path.read_text(encoding="utf-8")
        if any(token in text for token in banned):
            offenders.append(str(path.relative_to(root)))
    assert offenders == []


def test_importing_coflowai_loads_no_provider_sdk():
    """The stronger guarantee: `import coflowai` pulls in zero provider SDKs."""
    import subprocess
    import sys

    probe = (
        "import sys, coflowai\n"
        "leaked = [m for m in ('openai', 'anthropic', 'boto3', 'redis',\n"
        "                      'asyncpg', 'httpx', 'qdrant_client')\n"
        "          if m in sys.modules]\n"
        "print(','.join(leaked))"
    )
    result = subprocess.run([sys.executable, "-c", probe], capture_output=True,
                            text=True, check=True)
    assert result.stdout.strip() == ""
