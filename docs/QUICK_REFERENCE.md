# Quick Reference

Fast reference for common CoFlowAi operations.

## Installation

```bash
pip install coflowai                   # Core only
pip install "coflowai[openai]"         # + OpenAI
pip install "coflowai[anthropic]"      # + Anthropic
pip install "coflowai[all]"            # Everything
```

## Import

```python
from coflowai import (
    Agent,
    Workflow,
    tool,
    CoFlowAi,
)
from coflowai.policies import ExecutionPolicy, PermissionSet, RetryPolicy
from coflowai.testing import build_test_app, tool_call
```

## Agent

```python
# Basic agent
agent = Agent("name", "instructions", "model")

# With tools
agent = Agent("name", "instructions", "model", tools=[tool1, tool2])

# With structured output
agent = Agent("name", "instructions", "model", output_schema=MyModel)

# Full options
agent = Agent(
    name="researcher",
    instructions="You are a research assistant.",
    model="gpt-4o",
    tools=[search, analyze],
    output_schema=Report,
    max_iterations=10,
    temperature=0.7,
    max_output_tokens=2000,
    model_requirements={"tool_calling": True},
)
```

## Tool

```python
# Basic tool
@tool(description="Do something", permission="action.do")
async def my_tool(arg: str) -> dict:
    return {"result": arg}

# Full options
@tool(
    name="custom_name",
    description="What this tool does",
    permission="namespace.action",
    timeout=10,
    retries=2,
    side_effect=True,
    idempotent=True,
)
async def my_tool(arg: str) -> dict:
    return {"result": arg}
```

## Workflow

```python
# Sequential
workflow = (
    Workflow("name")
    .start(agent_a)
    .then(agent_b)
    .then(agent_c)
)

# Parallel
workflow = (
    Workflow("name")
    .start(agent_a)
    .parallel(agent_b, agent_c, agent_d, max_concurrency=2)
)

# Conditional
workflow = (
    Workflow("name")
    .start(agent_a)
    .route(
        condition=lambda ctx: ctx.state["type"],
        routes={"a": agent_b, "b": agent_c}
    )
)

# Loop
workflow = (
    Workflow("name")
    .start(agent_a)
    .loop(
        agent_b,
        until=lambda ctx: ctx.state["done"],
        max_iterations=3
    )
)

# Human approval
workflow = (
    Workflow("name")
    .start(agent_a)
    .approve("Continue?")
    .then(agent_b)
)
```

## Execution Policy

```python
policy = ExecutionPolicy(
    max_steps=20,
    timeout_seconds=300,
    max_model_calls=15,
    max_tool_calls=25,
    max_tokens=100_000,
    max_cost=3.00,
    max_concurrency=5,
    mode=ExecutionMode.STANDARD,
    checkpoint_enabled=True,
)
```

## Application Setup

```python
from coflowai import CoFlowAi
from coflowai.providers.openai import OpenAIProvider

# Create app
app = CoFlowAi()

# Register provider
provider = OpenAIProvider(model="gpt-4o", api_key="sk-...")
app.models.register("gpt-4o", provider=provider)  # optional: capabilities=..., pricing=..., fallbacks=[...]

# Run
result = await app.run(agent, "request", policy=policy)
```

## Execution

```python
# Run agent
result = await app.run(agent, "What's 2+2?")

# Run workflow
result = await app.run(workflow, "Process data", policy=policy)

# Stream
async for chunk in app.stream(agent, "Explain"):
    print(chunk.text, end="")

# Resume
await app.resume(execution_id, executable=workflow)

# Approve
await app.approve(execution_id, by="alice", executable=workflow)

# Replay
trace = await app.replay(execution_id)

# Fork
await app.fork(execution_id, from_step=4, executable=workflow,
               overrides={"model_responses": {"3": "a different answer"}})
```

## Result

```python
result = await app.run(agent, "request")

# Status
result.succeeded                    # bool
result.status == ExecutionStatus.FAILED                       # bool
result.status                       # ExecutionStatus

# Output
result.output                       # Agent/workflow output

# Usage
result.usage.cost            # float
result.usage.total_tokens          # int
result.usage.duration_ms / 1000 # float
result.usage.model_calls           # int
result.usage.tool_calls            # int

# Metadata
result.execution_id                # str
result.metadata                    # dict
```

## Testing

```python
from coflowai.testing import build_test_app, tool_call

# Create test app
app, model = build_test_app(
    responses=["Hello!"],
    tools=[my_tool],
    permissions=["my.permission"]
)

# With tool calls
app, model = build_test_app(
    responses=[
        tool_call("my_tool", arg="value"),
        "Final response"
    ],
    tools=[my_tool]
)

# Run test
result = await app.run(agent, "test")
assert result.succeeded
assert model.call_count == 2
```

## Providers

```python
# OpenAI
from coflowai.providers.openai import OpenAIProvider
provider = OpenAIProvider(model="gpt-4o", api_key="sk-...")

# Anthropic
from coflowai.providers.anthropic import AnthropicProvider
provider = AnthropicProvider(model="claude-sonnet-4", api_key="sk-ant-...")

# Azure OpenAI
from coflowai.providers.openai import AzureOpenAIProvider
provider = AzureOpenAIProvider(
    endpoint="https://your-resource.openai.azure.com",
    deployment="gpt-4o",          # your deployment name
    api_key="...",
    api_version="2024-10-21",
)

# Gemini
from coflowai.providers.gemini import GeminiProvider
provider = GeminiProvider(model="gemini-2.0-flash", api_key="...")

# Bedrock
from coflowai.providers.bedrock import BedrockProvider
provider = BedrockProvider(model_id="anthropic.claude-3-5-sonnet-20240620-v1:0",
                           region_name="us-east-1")
```

## Persistence

```python
# SQLite (zero dependencies)
from coflowai.persistence.sqlite import SqliteEventStore, SqliteStateStore
event_store = SqliteEventStore("state.db")
state_store = SqliteStateStore("state.db")

# Postgres
from coflowai.persistence.postgres import (PostgresEventStore,
                                           PostgresStateStore, create_pool)
pool = await create_pool("postgresql://...")  # asyncpg pool, shared by both stores
event_store = PostgresEventStore(pool)
state_store = PostgresStateStore(pool, owner="worker-1")  # owner = lock holder id

# Redis
from coflowai.persistence.redis import RedisEventStore, RedisStateStore
event_store = RedisEventStore(client=redis_client)
state_store = RedisStateStore(client=redis_client)

# Use in app
app = CoFlowAi(event_store=event_store, state_store=state_store)
```

## Distributed Execution

```python
from coflowai.distributed import ExecutionService, RedisJobQueue, WorkerPool

# Setup service
queue = RedisJobQueue(redis_client)
service = ExecutionService(queue=queue, runtime=app.runtime)
service.register(workflow)

# Submit job
execution_id = await service.submit(workflow, "request")

# Start workers (in the worker process)
pool = WorkerPool(app.runtime, queue, service.executables, size=4)
await pool.start()
...
await pool.stop()          # graceful: drains in-flight jobs
```

## Events

```python
# Get events
events = await app.event_store.get_events(execution_id)

for event in events:
    print(f"{event.timestamp}: {event.type}")
    print(f"  {event.payload}")
```

## Tracing

```python
# Get trace
trace = app.trace(execution_id)
print(trace.render())

# Inspect spans
for span in trace.spans:
    print(f"{span.name}: {span.duration}s")
```

## CLI

```bash
# Run
coflowai run app.py --input "task" --trace

# Executions
coflowai executions list app.py
coflowai executions inspect app.py exec_123
coflowai executions replay app.py exec_123
coflowai executions resume app.py exec_123
coflowai executions approve app.py exec_123 --by alice

# Workflows
coflowai workflows validate workflow.py

# Models & Tools
coflowai models list app.py
coflowai tools list app.py

# Worker
coflowai worker app.py --concurrency 4
```

## Error Handling

```python
from coflowai.core import (
    ExecutionError,
    ExecutionTimeout,
    BudgetExceeded,
    ModelError,
    ToolError,
)

# app.run() does not raise for execution failures: it returns a result
# whose .status / .error tell you what happened.
result = await app.run(agent, "task")
if not result.succeeded:
    if isinstance(result.error, BudgetExceeded):
        print(f"Budget exceeded: {result.error}")
    elif isinstance(result.error, ExecutionTimeout):
        print("Timed out")
    else:
        print(f"Execution failed ({result.status.value}): {result.error}")
```

## Common Patterns

### Retry on Failure
```python
retry_policy = RetryPolicy(
    attempts=3,              # total tries per model/tool call
    backoff="exponential",   # or "linear" / "fixed"
    base_delay=1.0,          # seconds
    max_delay=30.0,
    jitter=True,
)
policy = ExecutionPolicy(retry=retry_policy)
```

### Budget Limits
```python
policy = ExecutionPolicy(
    max_cost=1.00,
    max_tokens=50_000,
    max_steps=20,
)
```

### Access Workflow State
```python
def my_transform(context, output):
    context.state["result"] = output
    return output

workflow.then(agent).transform(my_transform)
```

### Conditional Routing
```python
def route_by_type(context):
    return context.state.get("type", "default")

workflow.route(
    condition=route_by_type,
    routes={"a": agent_a, "b": agent_b},
    default=agent_default
)
```

### Structured Output
```python
from pydantic import BaseModel

class Output(BaseModel):
    summary: str
    score: float

agent = Agent("name", "instructions", "model", output_schema=Output)
result = await app.run(agent, "task")
print(result.output.score)  # Validated Pydantic model
```

---

For complete documentation, see:
- **[Getting Started](./GETTING_STARTED.md)**
- **[API Reference](./API_REFERENCE.md)**
- **[User Guide](./USER_GUIDE.md)**
- **[Examples](./EXAMPLES.md)**
