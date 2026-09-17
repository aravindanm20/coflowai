# Getting Started with CoFlowAi

## Overview

CoFlowAi is a production-grade agentic AI framework for Python 3.12+ that separates agent reasoning from execution control. The LLM decides which tool to use and what to say; the runtime decides whether that is permitted, affordable, in time, and recoverable.

## Installation

### Core Installation

```bash
pip install coflowai
```

The core package includes only `pydantic` and `typing-extensions` dependencies.

### With Provider Support

```bash
# OpenAI/Azure/Compatible endpoints
pip install "coflowai[openai]"

# Anthropic
pip install "coflowai[anthropic]"

# AWS Bedrock
pip install "coflowai[bedrock]"

# Google Gemini
pip install "coflowai[gemini]"

# PostgreSQL and Redis persistence
pip install "coflowai[postgres,redis]"

# Everything (except cloud SDKs)
pip install "coflowai[all]"
```

### Requirements

- **Python 3.12+** (uses `asyncio.TaskGroup` and `asyncio.timeout`)
- SQLite support is built-in (no extra dependencies needed for durable single-node deployments)

## Quick Start

### Your First Agent

```python
import asyncio
from coflowai import Agent, tool
from coflowai.testing import build_test_app

@tool(description="Get the weather", permission="weather.read")
async def get_weather(city: str) -> dict:
    return {"city": city, "temperature": 30}

app, _ = build_test_app(tools=[get_weather], permissions=["weather.read"])
agent = Agent("assistant", "Answer the user's question.", "fake-model", [get_weather])

async def main():
    result = await app.run(agent, "What's the weather in Chennai?")
    print(result.output)

if __name__ == "__main__":
    asyncio.run(main())
```

### Understanding the Basic Flow

1. **Define Tools** - Decorate functions with `@tool` to make them available to agents
2. **Create an Agent** - Specify the agent's role, instructions, and available tools
3. **Build an Application** - Set up the runtime with tools and permissions
4. **Execute** - Run the agent with a user request

## Core Concepts

### Agents

Agents are the reasoning units that interact with LLMs:

```python
agent = Agent(
    name="assistant",
    instructions="You are a helpful assistant.",
    model="gpt-4o",
    tools=[get_weather, send_email],
    max_iterations=10,
    temperature=0.7
)
```

### Tools

Tools are functions that agents can call:

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
async def update_customer(customer_id: str, email: str) -> dict:
    # Implementation
    return {"status": "updated"}
```

**Key Properties:**
- `permission`: Required permission namespace
- `timeout`: Maximum execution time in seconds
- `side_effect`: Marks operations that modify state
- `idempotent`: Safe to retry (generates deterministic idempotency keys)

### Workflows

Workflows orchestrate multiple agents and control flow:

```python
from coflowai import Workflow

workflow = (
    Workflow("software-development", version="2")
    .start(requirement_agent)
    .then(architect_agent)
    .parallel(backend_agent, frontend_agent, database_agent, max_concurrency=3)
    .route(
        condition=classify,
        routes={"hotfix": hotfix_agent, "normal": review_agent}
    )
    .loop(reviewer, until=is_approved, max_iterations=3)
    .approve("Ship to production?", action=deploy_agent)
)
```

### Execution Policies

Control execution boundaries and costs:

```python
from coflowai.policies import ExecutionPolicy, ExecutionMode

policy = ExecutionPolicy(
    max_steps=20,              # Maximum execution steps
    timeout_seconds=300,       # 5 minute timeout
    max_model_calls=15,        # Limit LLM calls
    max_tool_calls=25,         # Limit tool invocations
    max_tokens=100_000,        # Token budget
    max_cost=3.00,             # Cost limit in USD
    max_concurrency=5,         # Parallel execution limit
    mode=ExecutionMode.STRICT, # Temperature=0, serialized tool calls
    checkpoint_enabled=True,   # Enable resume capability
)

result = await app.run(workflow, request, policy=policy)
```

## Testing Without LLMs

CoFlowAi provides a fake model provider for deterministic testing:

```python
from coflowai.testing import build_test_app, tool_call

# Define expected responses
app, model = build_test_app(
    responses=[
        tool_call("get_weather", city="Chennai"),  # Agent calls tool
        "It is 30°C in Chennai."                    # Agent's final response
    ],
    tools=[get_weather],
    permissions=["weather.read"]
)

result = await app.run(agent, "What's the weather?")

assert result.succeeded
assert model.call_count == 2
```

**Benefits:**
- No network calls
- No API keys required
- No token costs
- Fully deterministic
- Fast test execution

## Working with Real Providers

### OpenAI

```python
from coflowai import CoFlowAi
from coflowai.providers.openai import OpenAIProvider

provider = OpenAIProvider(api_key="sk-...")
app = CoFlowAi()
app.models.register(
    "gpt-4o",
    provider=provider,
    capabilities={"tool_calling": True, "structured_output": True},
    pricing={"input_per_million": 2.50, "output_per_million": 10.00}
)

agent = Agent("assistant", "Help the user.", "gpt-4o", tools=[...])
result = await app.run(agent, "Analyze this data...")
```

### Anthropic

```python
from coflowai.providers.anthropic import AnthropicProvider

provider = AnthropicProvider(api_key="sk-ant-...")
app.models.register(
    "claude-sonnet-4",
    provider=provider,
    capabilities={"tool_calling": True, "reasoning": True},
    pricing={"input_per_million": 3.00, "output_per_million": 15.00}
)
```

### Azure OpenAI

```python
from coflowai.providers.openai import AzureOpenAIProvider

provider = AzureOpenAIProvider(
    api_key="...",
    endpoint="https://your-resource.openai.azure.com",
    api_version="2024-02-15-preview"
)
```

## Structured Output

Get validated Pydantic models instead of raw JSON:

```python
from pydantic import BaseModel

class Analysis(BaseModel):
    summary: str
    risk: str
    confidence: float

agent = Agent(
    "analyst",
    "Analyse the report.",
    "gpt-4o",
    output_schema=Analysis
)

result = await app.run(agent, "Analyze Q3 numbers")
print(result.output.confidence)  # Validated Pydantic model
```

Invalid output is automatically repaired and retried, then fails loudly - never silently returned.

## Persistence and Resumability

### SQLite (Zero Dependencies)

```python
from coflowai.persistence.sqlite import SqliteEventStore, SqliteStateStore

app = CoFlowAi(
    event_store=SqliteEventStore("state.db"),
    state_store=SqliteStateStore("state.db")
)
```

### Resume Execution

```python
# Start execution
result = await app.run(workflow, "process data")
execution_id = result.execution_id

# Later, after crash or restart
await app.resume(execution_id, workflow)
```

Completed nodes are skipped, never rerun. Workflow hash is verified for compatibility.

## Observability

### Execution Traces

```python
result = await app.run(agent, "task")
trace = app.trace(result.execution_id)
print(trace.render())
```

### Usage Tracking

```python
print(f"Tokens: {result.usage.total_tokens}")
print(f"Cost: ${result.usage.total_cost:.4f}")
print(f"Duration: {result.usage.execution_time_seconds}s")
print(f"Model calls: {result.usage.model_calls}")
print(f"Tool calls: {result.usage.tool_calls}")
```

### Events

```python
# Listen to events
for event in app.events.get_events(execution_id):
    print(f"{event.type}: {event.data}")
```

## CLI Commands

```bash
# Run an agent/workflow
coflowai run app.py --input "Research AI frameworks" --trace

# List executions
coflowai executions list app.py

# Inspect execution details
coflowai executions inspect app.py exec_123

# Replay execution step-by-step
coflowai executions replay app.py exec_123

# Resume paused/failed execution
coflowai executions resume app.py exec_123

# Approve waiting execution
coflowai executions approve app.py exec_123 --by aravindan

# Validate workflow definition
coflowai workflows validate workflow.py

# List registered models
coflowai models list app.py

# List registered tools
coflowai tools list app.py

# Start distributed worker
coflowai worker app.py --concurrency 4
```

## Next Steps

- **[Core Concepts](./CORE_CONCEPTS.md)** - Deep dive into architecture and design
- **[API Reference](./API_REFERENCE.md)** - Detailed API documentation
- **[User Guide](./USER_GUIDE.md)** - Common patterns and recipes
- **[Advanced Topics](./ADVANCED.md)** - Distributed execution, security, optimization
- **[Examples](../examples/)** - Working code examples

## Philosophy

CoFlowAi is built on these principles:

1. **The AI can be nondeterministic. The execution platform cannot be uncontrolled.**
2. Never allow unlimited agent loops
3. Never trust model output
4. Never let provider details leak into agent logic
5. Every production execution is traceable
6. External databases should not be mandatory for local execution

## Getting Help

- **Documentation**: [docs/](../docs/)
- **Examples**: [examples/](../examples/)
- **Issues**: [GitHub Issues](https://github.com/aravindanm20/coflowai/issues)
- **Discussions**: [GitHub Discussions](https://github.com/aravindanm20/coflowai/discussions)
