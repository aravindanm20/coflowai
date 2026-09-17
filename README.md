# CoFlowAi

**A production-grade agentic AI framework for Python 3.12+**

> **The AI can be nondeterministic. The execution platform cannot be uncontrolled.**

CoFlowAi separates *agent reasoning* from *execution control*. The LLM decides which
tool to use and what to say; the runtime decides whether that is permitted,
affordable, in time, and recoverable.

```python
from coflowai import Agent, tool
from coflowai.testing import build_test_app

@tool(description="Get the weather", permission="weather.read")
async def get_weather(city: str) -> dict:
    return {"city": city, "temperature": 30}

app, _ = build_test_app(tools=[get_weather], permissions=["weather.read"])
agent = Agent("assistant", "Answer the user's question.", "fake-model", [get_weather])

result = await app.run(agent, "What's the weather in Chennai?")
print(result.output)
```

---

## Why

Most agent frameworks let nondeterministic model output influence too much of the
execution lifecycle. CoFlowAi draws a hard boundary:

| The LLM decides | The runtime decides |
| --- | --- |
| Which tool to call | Whether the tool *may* execute |
| What arguments to pass | Whether those arguments are valid |
| What the response contains | Whether the budget, deadline or step limit allows it |
| Whether to reason again | Whether to retry, pause, resume, checkpoint or stop |

## Core guarantees

- **Deterministic runtime** — explicit state machine, enforced transitions, no global mutable state.
- **Provider independence** — the core imports no LLM/database SDK. Adapters are optional packages.
- **Event sourced** — every meaningful action emits an event; executions are replayable.
- **Resumable** — checkpoints before/after every node; resume never reruns completed work.
- **Forkable** — re-run a production execution from step *n* with a different model or prompt.
- **Secure by default** — namespaced permissions, validated tool arguments, secret redaction, human approval gates.
- **Bounded** — step, iteration, token, cost, model-call and tool-call limits are mandatory, not advisory.
- **Observable** — structured logs, nested execution traces, metrics and per-scope usage, built in.
- **Testable offline** — `FakeModelProvider` / `FakeTool` make agent tests deterministic with no network, keys or spend.
- **Streaming** — token-by-token output that still obeys budgets, events and usage accounting.
- **Durable** — SQLite (stdlib, zero deps), Postgres or Redis persistence; executions survive restarts.
- **Distributed** — queue + worker pool + lock leases, with at-least-once delivery made safe by idempotent execution.

## Install

```bash
pip install coflowai                   # core: pydantic + typing-extensions only
pip install "coflowai[openai]"         # + httpx for OpenAI/Azure/compatible
pip install "coflowai[anthropic]"      # + httpx for Anthropic
pip install "coflowai[bedrock]"        # + boto3 for AWS Bedrock
pip install "coflowai[postgres,redis]" # durable multi-worker persistence
pip install "coflowai[all]"            # everything except cloud SDKs
```

SQLite persistence needs **no** extra dependency — `sqlite3` is in the standard
library, so durable single-node deployments work out of the box.

Requires Python 3.12+ (uses `asyncio.TaskGroup` and `asyncio.timeout`).

---

## Architecture

```
                    Developer SDK
                 Agent / Workflow / Tool
                          │
                  Workflow Compiler          ← validates before a token is spent
                          │
                  Execution Runtime          ← policy, state machine, checkpoints
          ┌───────────────┼───────────────┐
    Policy Engine    State Engine     Event Engine
          └───────────────┼───────────────┘
          ┌───────────────┼───────────────┐
    Model Gateway    Tool Runtime    Context Engine
          │               │               │
      Providers       Permissions       Memory
                          │
                    Observability
                 Logs · Traces · Metrics
```

Dependency direction is strictly one-way: **API → Runtime → Interfaces → Adapters.**
The core knows interfaces, never implementations.

---

## Concepts

### Everything is an `Executable`

```python
class Executable(ABC):
    async def execute(self, context: ExecutionContext) -> ExecutionResult: ...
```

`Agent`, `Workflow`, `ParallelNode`, `RouterNode`, `LoopNode`, `TransformNode` and
`HumanApproval` all implement it, so they compose freely and nest arbitrarily.

### Policies

```python
policy = ExecutionPolicy(
    max_steps=20,
    timeout_seconds=300,
    max_model_calls=15,
    max_tool_calls=25,
    max_tokens=100_000,
    max_cost=3.00,
    max_concurrency=5,
    mode=ExecutionMode.STRICT,   # forces temperature=0
    checkpoint_enabled=True,
)
result = await app.run(workflow, request, policy=policy)
```

Budgets are checked **before** each operation and emit `budget.warning` at 80%
utilisation and `budget.exceeded` when a limit is hit.

### Tools

```python
@tool(
    name="update_customer",
    description="Update a customer record",
    permission="database.write",
    timeout=10,
    retries=2,
    side_effect=True,
    idempotent=True,
)
async def update_customer(customer_id: str, email: str) -> dict: ...
```

The input JSON schema is derived from your type hints. Every call goes through the
full pipeline — registry → permission → validation → rate limit → timeout → sandbox
→ execute → output validation → audit event. Agents never call tools directly.

Non-idempotent side-effect tools are **never** retried. Idempotent ones get a
deterministic key from `execution_id + node_id + tool_call_id`, so a retry or resume
can't duplicate a payment.

### Workflows

```python
workflow = (
    Workflow("software-development", version="2")
    .start(requirement_agent)
    .then(architect_agent)
    .parallel(backend_agent, frontend_agent, database_agent, max_concurrency=3)
    .route(condition=classify, routes={"hotfix": hotfix_agent, "normal": review_agent})
    .loop(reviewer, until=is_approved, max_iterations=3)
    .approve("Ship to production?", action=deploy_agent)
)
```

Compilation rejects, **before execution**: missing nodes, invalid edges, unreachable
nodes, illegal cycles, ungranted permissions, unregistered models and unsupported
model capabilities. Loops always require `max_iterations`.

### Structured output

```python
class Analysis(BaseModel):
    summary: str
    risk: str
    confidence: float

agent = Agent("analyst", "Analyse.", "smart-model", output_schema=Analysis)
result.output.confidence   # validated Pydantic model, never raw JSON
```

Invalid output is repaired-and-retried, then fails loudly. It is never silently returned.

### Human approval

```python
paused = await app.run(workflow, "release 1.4.0")
paused.status              # ExecutionStatus.WAITING_FOR_APPROVAL

# hours later, from an API, a Slack button or the CLI:
await app.approve(paused.execution_id, by="aravindan", executable=workflow)
```

The execution is checkpointed and the worker is released — nothing is blocked in memory.

### Resume, replay, fork

```python
await runtime.resume(execution_id)                 # continues from the last checkpoint
trace = await runtime.replay(execution_id)         # full step-by-step story
await runtime.fork(execution_id, from_step=4, overrides={"model": "new-model"})
```

Resume verifies the workflow hash first and refuses to continue against an
incompatible definition. Completed nodes are skipped, never rerun.

### Context engine

Context is not conversation history. The builder assembles labelled sections
(instructions, workflow state, memory, retrieved knowledge, tool results,
conversation), applies per-section token budgets, and degrades gracefully:
drop low-relevance → summarise old turns → compress tool output. It never blindly
truncates from the end.

### Streaming

```python
async for chunk in app.stream(agent, "Explain distributed systems."):
    if chunk.is_final:
        result = chunk.metadata["result"]      # validated ExecutionResult
    else:
        print(chunk.text, end="", flush=True)
```

Budgets, events and usage accounting apply identically — streaming is
presentation, never an escape hatch from the policy engine. Providers without
native streaming are transparently emulated, so callers never branch on
capability. Structured output is never streamed partially: half a JSON object is
worse than none.

### Durable persistence

```python
from coflowai.persistence.sqlite import SqliteEventStore, SqliteStateStore

app = CoFlowAi(event_store=SqliteEventStore("state.db"),
               state_store=SqliteStateStore("state.db"))
```

Swap in `PostgresEventStore`/`PostgresStateStore` (asyncpg) or the Redis
equivalents for multi-worker deployments. Both implement atomic TTL lock leases,
so a crashed worker's executions become claimable without manual intervention —
and two workers can never drive the same execution.

### Distributed execution

```python
service = ExecutionService(RedisJobQueue(client), runtime)
service.register(workflow)

execution_id = await service.submit(workflow, request)   # returns immediately

# elsewhere, on N machines:
#   coflowai worker app.py --concurrency 4
```

Delivery is at-least-once, which is safe *because* execution is idempotent:
checkpoints skip completed nodes and side-effect tools carry idempotency keys. A
redelivered job resumes rather than duplicating work.

### Sandboxed tools

```python
app = CoFlowAi(sandbox=SubprocessSandbox(cpu_seconds=5, memory_mb=256))
```

Untrusted tools run in a separate interpreter with CPU, memory and
file-descriptor limits, a scrubbed environment (secrets are **not** inherited),
and hard kill on timeout.

### Determinism modes

| Mode | Temperature | Tool calls | Iteration budget |
| --- | --- | --- | --- |
| `STRICT` | pinned to 0 | serialised for reproducible ordering | tightest (≤5) |
| `STANDARD` | agent's own | parallel | the agent's own cap |
| `EXPLORATORY` | agent's own | parallel | up to `max_steps` — room to backtrack |

No mode is ever unbounded.

### Provider independence

```python
registry.register(
    "smart-model",
    provider=MyProvider(client),
    capabilities=ModelCapabilities(tool_calling=True, structured_output=True, reasoning=True),
    pricing=ModelPricing(input_per_million=3.0, output_per_million=15.0),
    fallbacks=["backup-model"],
)

agent = Agent("researcher", model_requirements={"reasoning": True, "tool_calling": True})
```

Agents refer to models by logical name or by capability. Four adapters ship in-tree — `openai` (also Azure and any OpenAI-compatible
endpoint), `anthropic`, `gemini`, `bedrock` — each with request/response
translation, streaming, structured-output emulation where the provider lacks it,
and error mapping onto the framework error model. Write your own by copying
`coflowai/providers/template.py`.

Every adapter is tested offline via `MockTransport`, so the suite needs no
network or API keys.

---

## Testing

```python
from coflowai.testing import build_test_app, tool_call

app, model = build_test_app([tool_call("get_weather", city="Chennai"), "It is 30°C."],
                            tools=[get_weather])
result = await app.run(agent, "weather?")

assert result.succeeded
assert model.call_count == 2
```

No network, no API keys, no token cost, fully deterministic.

## CLI

```bash
coflowai run app.py --input "Research AI frameworks" --trace
coflowai executions list app.py
coflowai executions inspect app.py exec_123
coflowai executions replay app.py exec_123
coflowai executions resume app.py exec_123
coflowai executions approve app.py exec_123 --by aravindan
coflowai worker app.py --concurrency 4
coflowai workflows validate workflow.py
coflowai models list app.py
coflowai tools list app.py
```

---

## Documentation

📚 **[Complete Documentation](./docs/README.md)** - Start here for comprehensive guides

Quick links:
- **[Getting Started](./docs/GETTING_STARTED.md)** - Installation and quickstart
- **[Core Concepts](./docs/CORE_CONCEPTS.md)** - Architecture deep dive
- **[API Reference](./docs/API_REFERENCE.md)** - Complete API documentation
- **[User Guide](./docs/USER_GUIDE.md)** - Common patterns and recipes
- **[Advanced Topics](./docs/ADVANCED.md)** - Distributed execution, security, optimization
- **[Examples](./docs/EXAMPLES.md)** - Working code examples
- **[Quick Reference](./docs/QUICK_REFERENCE.md)** - Fast lookup
- **[Troubleshooting](./docs/TROUBLESHOOTING.md)** - Common issues and solutions
- **[Contributing](./docs/CONTRIBUTING.md)** - Contribute to the project

---

## Repository layout

```
coflowai/
├── core/           Executable, ExecutionContext, ExecutionResult, errors, ids, retry
├── agents/         Agent reasoning loop
├── runtime/        Runtime, replay/fork, retry & timeout helpers
├── workflows/      Workflow builder, compiler, graph, nodes
├── models/         Provider interface, registry, capability router, gateway
├── tools/          @tool, registry, permission-checked executor, sandboxes
├── events/         Event, event types, store, publisher
├── state/          Checkpoints, execution records, state store
├── policies/       ExecutionPolicy, RetryPolicy, Budget, PermissionSet
├── context/        Context builder, token budget, summarisation
├── memory/         Memory interface + in-memory (BM25-ish) store
├── approvals/      Human-in-the-loop
├── observability/  Structured logging, tracer, metrics, usage, redaction
├── plugins/        Plugin API + entry-point loader
├── testing/        FakeModelProvider, FakeTool, harness, sandbox fixtures
├── providers/      OpenAI/Azure, Anthropic, Gemini, Bedrock + shared transport
├── persistence/    SQLite (stdlib), Postgres, Redis stores + lock leases
└── distributed/    Job queue, workers, execution service
```

## Dependency boundaries

The core installs only `pydantic` and `typing-extensions`. Adapters live under
`providers/` and `persistence/` and are imported *explicitly*, never by
`import coflowai` — a test asserts that importing the package pulls in zero
provider SDKs. Install only the extras you use.

## Engineering rules

1. Never allow unlimited agent loops — every loop has `max_iterations`.
2. Never trust model output — validate tool arguments, structured responses, routing decisions.
3. Never let provider details into agent logic.
4. Never make memory mandatory.
5. Never make a vector database mandatory.
6. Never make an external database mandatory for local execution.
7. Never let tools bypass the permission runtime.
8. Never silently retry side-effect operations.
9. Never resume workflows without checking compatibility.
10. Every production execution is traceable by `execution_id`.

## Development

```bash
pip install -e ".[dev]"
python -m pytest tests -q      # 167 tests, no network or API keys required
ruff check coflowai
mypy coflowai
```

