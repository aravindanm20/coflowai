# Examples Guide

Comprehensive examples demonstrating CoFlowAi features and patterns.

## Table of Contents

- [Basic Examples](#basic-examples)
- [Agent Examples](#agent-examples)
- [Workflow Examples](#workflow-examples)
- [Tool Examples](#tool-examples)
- [Provider Examples](#provider-examples)
- [Testing Examples](#testing-examples)
- [Production Examples](#production-examples)

---

## Basic Examples

### Hello World

```python
"""Simplest possible agent."""
import asyncio
from coflowai import Agent
from coflowai.testing import build_test_app

app, _ = build_test_app(["Hello, World!"])
agent = Agent("assistant", "You are helpful.", "fake-model")

async def main():
    result = await app.run(agent, "Say hello")
    print(result.output)  # "Hello, World!"

if __name__ == "__main__":
    asyncio.run(main())
```

### Agent with Single Tool

```python
"""Agent with weather tool."""
import asyncio
from coflowai import Agent, tool
from coflowai.testing import build_test_app, tool_call

@tool(description="Get weather", permission="weather.read")
async def get_weather(city: str) -> dict:
    return {"city": city, "temp": 72, "condition": "sunny"}

app, _ = build_test_app(
    responses=[
        tool_call("get_weather", city="San Francisco"),
        "It's 72°F and sunny in San Francisco."
    ],
    tools=[get_weather],
    permissions=["weather.read"]
)

agent = Agent(
    "weather-bot",
    "Help users check weather.",
    "fake-model",
    tools=[get_weather]
)

async def main():
    result = await app.run(agent, "What's the weather in San Francisco?")
    print(result.output)

if __name__ == "__main__":
    asyncio.run(main())
```

---

## Agent Examples

### Multi-Tool Agent

```python
"""Agent with multiple tools."""
from coflowai import Agent, tool

@tool(description="Search the web", permission="web.search")
async def search_web(query: str) -> list[str]:
    return [f"Result for {query}", "Another result"]

@tool(description="Read a URL", permission="web.read")
async def read_url(url: str) -> str:
    return f"Content from {url}"

@tool(description="Summarize text", permission="text.process")
async def summarize(text: str) -> str:
    return f"Summary of: {text}"

research_agent = Agent(
    "researcher",
    """You are a research assistant.
    1. Search for relevant information
    2. Read detailed sources
    3. Summarize findings""",
    "gpt-4o",
    tools=[search_web, read_url, summarize],
    max_iterations=10
)
```

### Agent with Structured Output

```python
"""Agent returning validated structured data."""
from pydantic import BaseModel, Field

class ResearchReport(BaseModel):
    """Research report structure."""
    title: str = Field(..., description="Report title")
    summary: str = Field(..., description="Executive summary")
    key_findings: list[str] = Field(..., description="List of findings")
    confidence: float = Field(..., ge=0.0, le=1.0)
    sources: list[str] = Field(..., description="Source URLs")

analyst = Agent(
    "analyst",
    "Analyze the topic and produce a structured report.",
    "gpt-4o",
    output_schema=ResearchReport,
    tools=[search_web]
)

result = await app.run(analyst, "Research quantum computing")
report: ResearchReport = result.output

print(f"Title: {report.title}")
print(f"Confidence: {report.confidence}")
for finding in report.key_findings:
    print(f"  - {finding}")
```

### Conversational Agent with Memory

```python
"""Agent with conversation memory."""
from coflowai import Agent
from coflowai.memory import InMemoryStore, MemoryItem

# Setup memory
memory = InMemoryStore()
await memory.store([
    MemoryItem(
        content="User's name is Alice",
        metadata={"type": "user_info"}
    ),
    MemoryItem(
        content="User prefers Python over JavaScript",
        metadata={"type": "preference"}
    ),
    MemoryItem(
        content="User is working on a web scraping project",
        metadata={"type": "context"}
    )
])

app = CoFlowAi(memory_store=memory)

agent = Agent(
    "assistant",
    "You are a helpful programming assistant. Use memory to personalize responses.",
    "gpt-4o"
)

# Memory is automatically retrieved and included
result = await app.run(
    agent,
    "What language should I use for my project?"
)
# Agent has context about user's preference and project
```

### Agent with Retry Logic

```python
"""Agent with custom retry policy."""
from coflowai.policies import RetryPolicy, ExecutionPolicy
from coflowai.core import ModelRateLimitError, ModelTemporaryError

retry_policy = RetryPolicy(
    max_retries=3,
    backoff_seconds=1.0,
    backoff_multiplier=2.0,
    jitter=True,
    retry_on=[ModelRateLimitError, ModelTemporaryError]
)

policy = ExecutionPolicy(
    max_steps=20,
    max_cost=1.00,
    retry_policy=retry_policy
)

agent = Agent("assistant", "Help the user", "gpt-4o")

result = await app.run(agent, "Analyze this data", policy=policy)
```

---

## Workflow Examples

### Sequential Pipeline

```python
"""Sequential workflow with state passing."""
from coflowai import Workflow, Agent

def set_plan(context, output):
    """Store plan in workflow state."""
    context.state["plan"] = output
    return output

def set_implementation(context, output):
    """Store implementation in workflow state."""
    context.state["implementation"] = output
    return output

pipeline = (
    Workflow("software-dev", version="1.0")
    .start(Agent("planner", "Create implementation plan", "gpt-4o"))
    .transform(set_plan)
    .then(Agent("coder", "Implement based on plan", "gpt-4o"))
    .transform(set_implementation)
    .then(Agent("tester", "Write tests", "gpt-4o"))
    .then(Agent("reviewer", "Review code", "gpt-4o"))
)

result = await app.run(pipeline, "Build a REST API")
```

### Parallel Processing

```python
"""Fan-out/fan-in pattern with merging."""
from coflowai import Workflow, Agent

def merge_research(outputs: dict[str, str]) -> str:
    """Merge outputs from parallel agents."""
    sections = []
    for agent_name, output in sorted(outputs.items()):
        sections.append(f"## {agent_name}\n{output}")
    return "\n\n".join(sections)

workflow = (
    Workflow("parallel-research")
    .start(Agent("coordinator", "Plan research areas", "gpt-4o"))
    .parallel(
        Agent("web-researcher", "Search web sources", "gpt-4o", tools=[search_web]),
        Agent("paper-researcher", "Search academic papers", "gpt-4o", tools=[search_papers]),
        Agent("news-researcher", "Search news articles", "gpt-4o", tools=[search_news]),
        max_concurrency=3,
        merge=merge_research
    )
    .then(Agent("synthesizer", "Synthesize all findings", "gpt-4o"))
)

result = await app.run(
    workflow,
    "Research AI safety",
    policy=ExecutionPolicy(max_cost=2.00)
)
```

### Conditional Routing

```python
"""Route based on classification."""
from coflowai import Workflow, Agent, ExecutionContext

def classify_severity(context: ExecutionContext) -> str:
    """Classify issue severity."""
    analysis = context.state.get("analysis", {})
    severity = analysis.get("severity", "medium")
    
    if severity == "critical":
        return "urgent"
    elif severity in ["high", "medium"]:
        return "standard"
    else:
        return "backlog"

workflow = (
    Workflow("ticket-triage")
    .start(Agent("analyzer", "Analyze the issue", "gpt-4o"))
    .route(
        condition=classify_severity,
        routes={
            "urgent": Agent("escalator", "Escalate to team lead", "gpt-4o"),
            "standard": Agent("assigner", "Assign to available dev", "gpt-4o"),
            "backlog": Agent("planner", "Add to backlog", "gpt-4o"),
        }
    )
)

result = await app.run(workflow, "Bug: Production database connection failing")
```

### Iterative Refinement Loop

```python
"""Loop until quality threshold met."""
from coflowai import Workflow, Agent, ExecutionContext

def quality_check(context: ExecutionContext) -> bool:
    """Check if output meets quality standards."""
    score = context.state.get("quality_score", 0.0)
    return score >= 0.8

def extract_score(context, output):
    """Extract quality score from review."""
    # Assuming output is structured with score
    context.state["quality_score"] = output.get("score", 0.0)
    return output

workflow = (
    Workflow("content-refinement")
    .start(Agent("writer", "Write initial draft", "gpt-4o"))
    .loop(
        executable=Agent(
            "editor",
            "Review and improve. Return quality score.",
            "gpt-4o",
            output_schema=ReviewOutput
        ),
        until=quality_check,
        max_iterations=3
    )
    .then(Agent("finalizer", "Finalize for publication", "gpt-4o"))
)

result = await app.run(workflow, "Write blog post about Python async")
```

### Human Approval Workflow

```python
"""Workflow with human-in-the-loop."""
from coflowai import Workflow, Agent

deployment_workflow = (
    Workflow("deploy-pipeline")
    .start(Agent("builder", "Build application", "gpt-4o"))
    .then(Agent("tester", "Run test suite", "gpt-4o"))
    .approve(
        message="Tests passed. Deploy to staging?",
        metadata={"environment": "staging"}
    )
    .then(Agent("deployer", "Deploy to staging", "gpt-4o", tools=[deploy]))
    .approve(
        message="Staging validation complete. Deploy to production?",
        metadata={"environment": "production"},
        timeout_hours=24
    )
    .then(Agent("deployer", "Deploy to production", "gpt-4o", tools=[deploy]))
)

# Start execution
result = await app.run(deployment_workflow, "Deploy v2.0.0")
# Execution pauses at first approval

# Later, approve via API or CLI
await app.approve(result.execution_id, by="alice", executable=deployment_workflow)
```

### Nested Workflows

```python
"""Compose workflows from sub-workflows."""
from coflowai import Workflow, Agent

# Sub-workflow for testing
test_workflow = (
    Workflow("testing")
    .start(Agent("unit-tester", "Run unit tests", "gpt-4o"))
    .then(Agent("integration-tester", "Run integration tests", "gpt-4o"))
    .then(Agent("e2e-tester", "Run E2E tests", "gpt-4o"))
)

# Sub-workflow for deployment
deploy_workflow = (
    Workflow("deployment")
    .start(Agent("builder", "Build artifacts", "gpt-4o"))
    .then(Agent("deployer", "Deploy to environment", "gpt-4o"))
    .then(Agent("validator", "Validate deployment", "gpt-4o"))
)

# Main workflow
main_workflow = (
    Workflow("ci-cd-pipeline", version="1.0")
    .start(Agent("linter", "Run linting", "gpt-4o"))
    .then(test_workflow)      # Nested workflow
    .approve("Deploy to staging?")
    .then(deploy_workflow)    # Nested workflow
    .approve("Deploy to production?")
    .then(deploy_workflow)    # Reuse same workflow
)
```

---

## Tool Examples

### Async HTTP Tool

```python
"""Tool making external API calls."""
import httpx
from coflowai import tool

@tool(
    description="Fetch data from API",
    permission="api.external.data",
    timeout=10,
    retries=2
)
async def fetch_api_data(endpoint: str, params: dict | None = None) -> dict:
    """Fetch data from external API."""
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"https://api.example.com/{endpoint}",
            params=params or {},
            timeout=10.0
        )
        response.raise_for_status()
        return response.json()
```

### Database Tool with Validation

```python
"""Tool with input validation."""
from pydantic import BaseModel, validator, Field
from coflowai import tool

class CustomerUpdate(BaseModel):
    customer_id: str = Field(..., regex=r"^cust_[a-zA-Z0-9]+$")
    email: str = Field(..., regex=r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$")
    name: str = Field(..., min_length=1, max_length=100)
    
    @validator("email")
    def validate_email_domain(cls, v):
        if v.endswith("@tempmail.com"):
            raise ValueError("Temporary email addresses not allowed")
        return v

@tool(
    description="Update customer information",
    permission="database.customers.write",
    timeout=5,
    side_effect=True,
    idempotent=True
)
async def update_customer(data: CustomerUpdate) -> dict:
    """Update customer with validation."""
    # Input is validated by Pydantic before reaching here
    async with database.transaction() as tx:
        await tx.execute(
            "UPDATE customers SET email = $1, name = $2 WHERE id = $3",
            data.email, data.name, data.customer_id
        )
    return {"status": "updated", "customer_id": data.customer_id}
```

### Rate-Limited Tool

```python
"""Tool with rate limiting."""
import asyncio
from datetime import datetime, timedelta
from collections import defaultdict
from coflowai import tool

class RateLimiter:
    """Simple token bucket rate limiter."""
    
    def __init__(self, rate: int, per_seconds: int):
        self.rate = rate
        self.per_seconds = per_seconds
        self.buckets = defaultdict(list)
    
    async def acquire(self, key: str):
        """Acquire rate limit token."""
        now = datetime.now()
        bucket = self.buckets[key]
        
        # Remove old tokens
        bucket[:] = [t for t in bucket if now - t < timedelta(seconds=self.per_seconds)]
        
        if len(bucket) >= self.rate:
            wait_time = (bucket[0] + timedelta(seconds=self.per_seconds) - now).total_seconds()
            await asyncio.sleep(wait_time)
            bucket.pop(0)
        
        bucket.append(now)

limiter = RateLimiter(rate=10, per_seconds=60)  # 10 calls per minute

@tool(description="Search API", permission="api.search")
async def search_api(query: str) -> list[dict]:
    """Rate-limited API search."""
    await limiter.acquire("search_api")
    # Make API call
    return await api_client.search(query)
```

### Tool with Context Access

```python
"""Tool that accesses execution context."""
from coflowai import tool
from coflowai.core import get_current_context

@tool(description="Save data to workflow state", permission="state.write")
async def save_to_state(key: str, value: Any) -> dict:
    """Save data to execution state."""
    context = get_current_context()
    context.state[key] = value
    
    # Emit event
    context.events.emit("data.saved", {
        "key": key,
        "execution_id": context.execution_id
    })
    
    return {"status": "saved", "key": key}

@tool(description="Load data from workflow state", permission="state.read")
async def load_from_state(key: str) -> Any:
    """Load data from execution state."""
    context = get_current_context()
    return context.state.get(key)
```

---

## Provider Examples

### OpenAI Provider

```python
"""Using OpenAI provider."""
from coflowai import CoFlowAi, Agent
from coflowai.providers.openai import OpenAIProvider
from coflowai.models import ModelCapabilities, ModelPricing

# Setup provider
provider = OpenAIProvider(
    api_key="sk-...",
    organization="org-...",  # Optional
)

app = CoFlowAi()

# Register model
app.models.register(
    "gpt-4o",
    provider=provider,
    capabilities=ModelCapabilities(
        tool_calling=True,
        structured_output=True,
        streaming=True,
        vision=True,
    ),
    pricing=ModelPricing(
        input_per_million=2.50,
        output_per_million=10.00,
    ),
    fallbacks=["gpt-4o-mini"]
)

# Use agent
agent = Agent("assistant", "Help the user", "gpt-4o")
result = await app.run(agent, "Explain asyncio")
```

### Anthropic Provider

```python
"""Using Anthropic provider."""
from coflowai.providers.anthropic import AnthropicProvider

provider = AnthropicProvider(api_key="sk-ant-...")

app.models.register(
    "claude-sonnet-4",
    provider=provider,
    capabilities=ModelCapabilities(
        tool_calling=True,
        reasoning=True,
        streaming=True,
    ),
    pricing=ModelPricing(
        input_per_million=3.00,
        output_per_million=15.00,
    )
)
```

### Multi-Provider Setup

```python
"""Using multiple providers with fallbacks."""
from coflowai.providers.openai import OpenAIProvider
from coflowai.providers.anthropic import AnthropicProvider

openai = OpenAIProvider(api_key="...")
anthropic = AnthropicProvider(api_key="...")

app = CoFlowAi()

# Primary: OpenAI
app.models.register("gpt-4o", provider=openai, ...)

# Fallback: Anthropic
app.models.register("claude-sonnet-4", provider=anthropic, ...)

# Agent with automatic fallback
agent = Agent(
    "assistant",
    "Help user",
    "gpt-4o",  # Primary
    # If gpt-4o fails with transient error, falls back to claude-sonnet-4
)
```

---

## Testing Examples

### Basic Unit Test

```python
"""Unit test with fake provider."""
import pytest
from coflowai.testing import build_test_app, tool_call

@pytest.mark.asyncio
async def test_weather_agent():
    """Test weather agent with mocked responses."""
    app, model = build_test_app(
        responses=[
            tool_call("get_weather", city="Tokyo"),
            "It's 25°C in Tokyo."
        ],
        tools=[get_weather],
        permissions=["weather.read"]
    )
    
    agent = Agent("weather", "Help with weather", "fake-model", tools=[get_weather])
    result = await app.run(agent, "Weather in Tokyo?")
    
    assert result.succeeded
    assert "25" in result.output
    assert "Tokyo" in result.output
    assert model.call_count == 2
```

### Testing Structured Output

```python
"""Test agent with structured output."""
from pydantic import BaseModel

class Analysis(BaseModel):
    summary: str
    score: float
    recommendation: str

@pytest.mark.asyncio
async def test_structured_output():
    """Test structured output validation."""
    app, model = build_test_app([
        {"summary": "Good performance", "score": 0.85, "recommendation": "Approve"}
    ])
    
    agent = Agent("analyst", "Analyze", "fake-model", output_schema=Analysis)
    result = await app.run(agent, "Analyze Q3")
    
    assert isinstance(result.output, Analysis)
    assert result.output.score == 0.85
    assert result.output.recommendation == "Approve"
```

### Integration Test with Real Provider

```python
"""Integration test (requires API key)."""
@pytest.mark.integration
@pytest.mark.asyncio
async def test_real_openai():
    """Test with real OpenAI API."""
    import os
    
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")
    
    provider = OpenAIProvider(api_key=os.getenv("OPENAI_API_KEY"))
    app = CoFlowAi()
    app.models.register("gpt-4o-mini", provider=provider, ...)
    
    agent = Agent("math", "Solve math problems", "gpt-4o-mini")
    result = await app.run(
        agent,
        "What is 15 * 23?",
        policy=ExecutionPolicy(max_cost=0.01)
    )
    
    assert result.succeeded
    assert "345" in result.output
```

---

## Production Examples

### FastAPI Integration

```python
"""Production API server with CoFlowAi."""
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from coflowai import CoFlowAi, Workflow
from coflowai.persistence.postgres import PostgresEventStore, PostgresStateStore
from coflowai.distributed import ExecutionService, RedisJobQueue

# Setup
app_api = FastAPI()

event_store = PostgresEventStore(dsn="postgresql://...")
state_store = PostgresStateStore(dsn="postgresql://...")
coflowai_app = CoFlowAi(event_store=event_store, state_store=state_store)

queue = RedisJobQueue(redis_client)
service = ExecutionService(queue=queue, runtime=coflowai_app.runtime)
service.register(my_workflow)

class ExecutionRequest(BaseModel):
    workflow_name: str
    input_data: str
    max_cost: float = 1.0

@app_api.post("/execute")
async def create_execution(request: ExecutionRequest):
    """Submit workflow for execution."""
    try:
        execution_id = await service.submit(
            executable=my_workflow,
            request=request.input_data,
            policy=ExecutionPolicy(max_cost=request.max_cost)
        )
        return {"execution_id": execution_id, "status": "queued"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app_api.get("/executions/{execution_id}")
async def get_execution(execution_id: str):
    """Get execution status."""
    try:
        state = await coflowai_app.state_store.get(execution_id)
        return {
            "execution_id": execution_id,
            "status": state.status,
            "output": state.output if state.status == "completed" else None,
            "error": str(state.error) if state.error else None,
        }
    except KeyError:
        raise HTTPException(status_code=404, detail="Execution not found")

@app_api.get("/health")
async def health():
    """Health check."""
    return {"status": "healthy"}
```

### Worker Process

```python
"""Production worker with monitoring."""
import signal
import asyncio
from coflowai.distributed import Worker
import structlog

logger = structlog.get_logger()
shutdown_event = asyncio.Event()

def handle_shutdown(signum, frame):
    """Handle shutdown gracefully."""
    logger.info("Shutdown signal received")
    shutdown_event.set()

signal.signal(signal.SIGTERM, handle_shutdown)
signal.signal(signal.SIGINT, handle_shutdown)

async def run_worker():
    """Run worker with graceful shutdown."""
    worker = Worker(
        service=service,
        concurrency=4,
        poll_interval_seconds=1.0,
    )
    
    worker_task = asyncio.create_task(worker.run())
    
    # Wait for shutdown signal
    await shutdown_event.wait()
    
    # Graceful shutdown
    logger.info("Stopping worker...")
    worker.stop()
    await worker_task
    logger.info("Worker stopped")

if __name__ == "__main__":
    asyncio.run(run_worker())
```

### With Monitoring and Metrics

```python
"""Production setup with full observability."""
from prometheus_client import Counter, Histogram, start_http_server
from opentelemetry import trace
import structlog

# Configure logging
structlog.configure(processors=[...])

# Configure metrics
executions_total = Counter("executions_total", "Total executions", ["status"])
execution_duration = Histogram("execution_duration_seconds", "Duration")
execution_cost = Histogram("execution_cost_dollars", "Cost")

# Configure tracing
from coflowai.observability import OpenTelemetryTracer
tracer = OpenTelemetryTracer(trace.get_tracer("coflowai"))

# Setup app
app = CoFlowAi(
    event_store=event_store,
    state_store=state_store,
    tracer=tracer
)

# Wrapper with observability
async def run_with_observability(workflow, request):
    """Run with full observability."""
    start = time.time()
    
    try:
        result = await app.run(workflow, request)
        
        # Success metrics
        executions_total.labels(status="success").inc()
        execution_duration.observe(time.time() - start)
        execution_cost.observe(result.usage.total_cost)
        
        # Structured logging
        logger.info(
            "execution_completed",
            execution_id=result.execution_id,
            duration=time.time() - start,
            cost=result.usage.total_cost,
        )
        
        return result
        
    except Exception as e:
        executions_total.labels(status="failed").inc()
        logger.error("execution_failed", error=str(e))
        raise

# Start metrics server
start_http_server(9090)
```

---

For more examples, see the [examples/](../examples/) directory in the repository.
