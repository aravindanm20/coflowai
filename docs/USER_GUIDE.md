# User Guide

Common patterns, recipes, and best practices for CoFlowAi.

## Table of Contents

- [Agent Patterns](#agent-patterns)
- [Workflow Patterns](#workflow-patterns)
- [Tool Design](#tool-design)
- [Error Handling](#error-handling)
- [Testing Strategies](#testing-strategies)
- [Performance Optimization](#performance-optimization)
- [Production Deployment](#production-deployment)

---

## Agent Patterns

### Simple Q&A Agent

```python
from coflowai import Agent, CoFlowAi
from coflowai.providers.openai import OpenAIProvider

provider = OpenAIProvider(api_key="...")
app = CoFlowAi()
app.models.register("gpt-4o", provider=provider, ...)

agent = Agent(
    "assistant",
    "You are a helpful assistant.",
    "gpt-4o"
)

result = await app.run(agent, "What is Python?")
print(result.output)
```

### Agent with Tools

```python
from coflowai import Agent, tool

@tool(description="Search the web", permission="web.search")
async def search(query: str) -> list[str]:
    # Implementation
    return ["result1", "result2"]

@tool(description="Fetch URL content", permission="web.fetch")
async def fetch_url(url: str) -> str:
    # Implementation
    return "content"

researcher = Agent(
    "researcher",
    "You are a research assistant. Search and analyze information.",
    "gpt-4o",
    tools=[search, fetch_url],
    max_iterations=5
)

result = await app.run(
    researcher,
    "Research the latest AI frameworks",
    policy=ExecutionPolicy(max_cost=1.00)
)
```

### Structured Output Agent

```python
from pydantic import BaseModel, Field

class CodeReview(BaseModel):
    summary: str = Field(description="Brief summary")
    issues: list[str] = Field(description="List of issues found")
    severity: str = Field(description="high/medium/low")
    approved: bool

reviewer = Agent(
    "code-reviewer",
    "Review code for security and quality issues.",
    "gpt-4o",
    output_schema=CodeReview
)

result = await app.run(reviewer, f"Review: {code}")
review: CodeReview = result.output

if review.severity == "high":
    # Handle critical issues
    pass
```

### Conversational Agent with Memory

```python
from coflowai.memory import InMemoryStore, MemoryItem

memory = InMemoryStore()

# Store conversation context
await memory.store([
    MemoryItem(
        content="User prefers concise responses",
        metadata={"type": "preference"}
    ),
    MemoryItem(
        content="Last discussed: Python asyncio",
        metadata={"type": "context"}
    )
])

app = CoFlowAi(memory_store=memory)

agent = Agent(
    "assistant",
    "You are a helpful assistant. Use memory to personalize responses.",
    "gpt-4o"
)

# Memory is automatically retrieved and included in context
result = await app.run(agent, "Continue our discussion")
```

### Multi-Model Agent

```python
# Register multiple models
app.models.register("gpt-4o", provider=openai_provider, ...)
app.models.register("claude-sonnet-4", provider=anthropic_provider, ...)

# Agent with capability requirements
agent = Agent(
    "analyst",
    "Analyze data and provide insights.",
    model_requirements={
        "tool_calling": True,
        "reasoning": True,
    },
    tools=[analyze_data]
)

# Runtime selects compatible model based on requirements
result = await app.run(agent, "Analyze Q3 revenue")
```

---

## Workflow Patterns

### Sequential Pipeline

```python
from coflowai import Workflow

workflow = (
    Workflow("data-pipeline", version="1")
    .start(Agent("extractor", "Extract data", "gpt-4o"))
    .then(Agent("validator", "Validate data", "gpt-4o"))
    .then(Agent("transformer", "Transform data", "gpt-4o"))
    .then(Agent("loader", "Load data", "gpt-4o"))
)

result = await app.run(workflow, "Process customer data")
```

### Fan-Out/Fan-In (Parallel)

```python
def merge_research(outputs: dict[str, str]) -> str:
    """Merge parallel research outputs."""
    return "\n\n".join(f"## {k}\n{v}" for k, v in outputs.items())

workflow = (
    Workflow("research-pipeline")
    .start(Agent("planner", "Plan research", "gpt-4o"))
    .parallel(
        Agent("web-researcher", "Search web", "gpt-4o", tools=[search]),
        Agent("db-researcher", "Query database", "gpt-4o", tools=[query_db]),
        Agent("doc-researcher", "Read documents", "gpt-4o", tools=[read_docs]),
        max_concurrency=3,
        merge=merge_research
    )
    .then(Agent("synthesizer", "Synthesize findings", "gpt-4o"))
)
```

### Conditional Routing

```python
def classify_urgency(context: ExecutionContext) -> str:
    """Classify based on context state."""
    data = context.state.get("analysis", {})
    if data.get("severity") == "critical":
        return "urgent"
    return "normal"

workflow = (
    Workflow("ticket-processor")
    .start(Agent("analyzer", "Analyze ticket", "gpt-4o"))
    .route(
        condition=classify_urgency,
        routes={
            "urgent": Agent("escalator", "Escalate immediately", "gpt-4o"),
            "normal": Agent("responder", "Send standard response", "gpt-4o"),
        },
        default=Agent("reviewer", "Needs review", "gpt-4o")
    )
)
```

### Iterative Refinement

```python
def is_approved(context: ExecutionContext) -> bool:
    """Check if output is approved."""
    return context.state.get("approved", False)

def score_quality(context: ExecutionContext) -> bool:
    """Check if quality threshold met."""
    return context.state.get("quality_score", 0) >= 0.8

workflow = (
    Workflow("content-creation")
    .start(Agent("writer", "Write content", "gpt-4o"))
    .loop(
        Agent("reviewer", "Review and suggest improvements", "gpt-4o"),
        until=score_quality,
        max_iterations=3
    )
    .approve(
        "Content ready for publication?",
        action=Agent("publisher", "Publish content", "gpt-4o")
    )
)
```

### Error Recovery Pattern

```python
def has_error(context: ExecutionContext) -> str:
    """Check for errors and route appropriately."""
    if context.state.get("error"):
        return "retry"
    return "success"

workflow = (
    Workflow("robust-pipeline")
    .start(Agent("processor", "Process data", "gpt-4o"))
    .route(
        condition=has_error,
        routes={
            "retry": (
                Workflow("retry-branch")
                .start(Agent("fixer", "Fix errors", "gpt-4o"))
                .then(Agent("processor", "Re-process", "gpt-4o"))
            ),
            "success": Agent("finalizer", "Finalize", "gpt-4o")
        }
    )
)
```

### Nested Workflows

```python
# Sub-workflow for code generation
code_gen_workflow = (
    Workflow("code-gen")
    .start(Agent("designer", "Design solution", "gpt-4o"))
    .then(Agent("coder", "Write code", "gpt-4o"))
    .then(Agent("tester", "Write tests", "gpt-4o"))
)

# Main workflow
main_workflow = (
    Workflow("software-project")
    .start(Agent("pm", "Create requirements", "gpt-4o"))
    .then(code_gen_workflow)  # Nested workflow
    .then(Agent("reviewer", "Review code", "gpt-4o"))
    .approve("Deploy to production?")
)
```

---

## Tool Design

### Basic Tool

```python
@tool(description="Calculate sum", permission="math.basic")
def add(a: float, b: float) -> float:
    """Tools can be sync or async."""
    return a + b
```

### Async Tool with External API

```python
import httpx

@tool(
    description="Fetch weather data",
    permission="api.weather",
    timeout=10,
    retries=2
)
async def get_weather(city: str) -> dict:
    """Always use async for I/O operations."""
    async with httpx.AsyncClient() as client:
        response = await client.get(
            "https://api.weather.com/v1/current",
            params={"city": city}
        )
        response.raise_for_status()
        return response.json()
```

### Tool with Validation

```python
from pydantic import BaseModel, validator

class UpdateCustomerInput(BaseModel):
    customer_id: str
    email: str
    
    @validator("email")
    def validate_email(cls, v):
        if "@" not in v:
            raise ValueError("Invalid email")
        return v

@tool(
    description="Update customer email",
    permission="database.write",
    side_effect=True,
    idempotent=True
)
async def update_customer(input: UpdateCustomerInput) -> dict:
    """Type hints drive JSON schema generation."""
    # Implementation with idempotency key available via context
    return {"status": "updated"}
```

### Side-Effect Tool with Idempotency

```python
@tool(
    description="Charge payment",
    permission="payments.write",
    timeout=30,
    retries=3,
    side_effect=True,
    idempotent=True,  # Safe to retry
)
async def charge_payment(
    customer_id: str,
    amount: float,
    idempotency_key: str | None = None
) -> dict:
    """
    Idempotency key is automatically provided by runtime:
    execution_id + node_id + tool_call_id
    
    Safe to retry even on ambiguous errors.
    """
    # Pass idempotency key to payment provider
    result = await payment_provider.charge(
        customer_id=customer_id,
        amount=amount,
        idempotency_key=idempotency_key
    )
    return result
```

### Tool with Context Access

```python
from coflowai.core import get_current_context

@tool(description="Save to execution state", permission="state.write")
async def save_analysis(key: str, value: Any) -> dict:
    """Access execution context within tool."""
    context = get_current_context()
    context.state[key] = value
    return {"status": "saved", "key": key}
```

### Rate-Limited Tool

```python
import asyncio
from collections import defaultdict
from datetime import datetime, timedelta

# Simple rate limiter
rate_limits = defaultdict(list)

@tool(
    description="Search external API",
    permission="api.search",
    timeout=10
)
async def search_api(query: str) -> list[str]:
    """Rate limit at tool level."""
    now = datetime.now()
    calls = rate_limits["search_api"]
    
    # Remove old calls
    calls[:] = [t for t in calls if now - t < timedelta(minutes=1)]
    
    if len(calls) >= 10:  # 10 calls per minute
        wait = 60 - (now - calls[0]).seconds
        await asyncio.sleep(wait)
    
    calls.append(now)
    
    # Make API call
    return ["result1", "result2"]
```

---

## Error Handling

### Graceful Degradation

```python
from coflowai.policies import RetryPolicy, ExecutionPolicy
from coflowai.core import ModelTemporaryError

retry_policy = RetryPolicy(
    max_retries=3,
    backoff_seconds=2.0,
    backoff_multiplier=2.0,
    jitter=True,
    retry_on=[ModelTemporaryError],
)

policy = ExecutionPolicy(
    max_cost=5.00,
    retry_policy=retry_policy
)

try:
    result = await app.run(agent, request, policy=policy)
except BudgetExceededError as e:
    # Budget exhausted
    logger.warning(f"Budget exceeded: {e}")
    # Fall back to cached response or simpler model
except ModelRateLimitError as e:
    # Rate limited
    logger.warning("Rate limited, queueing for later")
    await queue_for_retry(request)
except ExecutionTimeoutError as e:
    # Timeout
    logger.error(f"Timeout after {e.elapsed}s")
    # Check if partial results available
    if e.execution_id:
        checkpoint = await app.state_store.get(e.execution_id)
        # Use partial results
```

### Fallback Models

```python
# Register with fallback chain
app.models.register(
    "gpt-4o",
    provider=openai_provider,
    fallbacks=["gpt-4o-mini", "claude-sonnet-4"]
)

# Automatic fallback on transient errors
agent = Agent("assistant", "Help the user", "gpt-4o")
result = await app.run(agent, request)
# If gpt-4o fails with transient error, automatically tries fallbacks
```

### Custom Error Recovery

```python
from coflowai.core import ExecutionError

async def run_with_recovery(agent, request):
    """Execute with custom recovery logic."""
    try:
        return await app.run(agent, request)
    except ExecutionError as e:
        # Log to monitoring
        await log_to_datadog(e)
        
        # Attempt recovery based on error type
        if isinstance(e, ModelRateLimitError):
            # Wait and retry
            await asyncio.sleep(60)
            return await app.run(agent, request)
        
        elif isinstance(e, BudgetExceededError):
            # Use cheaper model
            agent.model = "gpt-4o-mini"
            return await app.run(agent, request)
        
        else:
            # Can't recover, propagate
            raise
```

---

## Testing Strategies

### Unit Testing Agents

```python
import pytest
from coflowai.testing import build_test_app, tool_call

@pytest.mark.asyncio
async def test_weather_agent():
    """Test agent with mocked tool responses."""
    app, model = build_test_app(
        responses=[
            tool_call("get_weather", city="Chennai"),
            "The temperature is 30°C in Chennai."
        ],
        tools=[get_weather],
        permissions=["weather.read"]
    )
    
    agent = Agent(
        "assistant",
        "Help with weather",
        "fake-model",
        tools=[get_weather]
    )
    
    result = await app.run(agent, "What's the weather in Chennai?")
    
    assert result.succeeded
    assert "30" in result.output
    assert model.call_count == 2
```

### Testing Workflows

```python
@pytest.mark.asyncio
async def test_research_workflow():
    """Test workflow with multiple agents."""
    app, model = build_test_app(
        responses=[
            "Research plan",       # Planner
            "Web findings",        # Web researcher
            "DB findings",         # DB researcher
            "Final report",        # Synthesizer
        ],
        tools=[search, query_db],
        permissions=["web.search", "database.read"]
    )
    
    workflow = (
        Workflow("research")
        .start(Agent("planner", "Plan", "fake-model"))
        .parallel(
            Agent("web", "Search", "fake-model", tools=[search]),
            Agent("db", "Query", "fake-model", tools=[query_db]),
        )
        .then(Agent("synthesizer", "Synthesize", "fake-model"))
    )
    
    result = await app.run(workflow, "Research AI")
    
    assert result.succeeded
    assert model.call_count == 4
```

### Testing with Structured Output

```python
from pydantic import BaseModel

class Analysis(BaseModel):
    summary: str
    score: float

@pytest.mark.asyncio
async def test_structured_output():
    """Test agent with structured output validation."""
    app, model = build_test_app(
        responses=[
            {"summary": "Good", "score": 0.85}  # Valid
        ]
    )
    
    agent = Agent(
        "analyst",
        "Analyze",
        "fake-model",
        output_schema=Analysis
    )
    
    result = await app.run(agent, "Analyze this")
    
    assert isinstance(result.output, Analysis)
    assert result.output.score == 0.85
```

### Integration Testing with Real Providers

```python
@pytest.mark.integration
@pytest.mark.asyncio
async def test_real_openai():
    """Integration test with real provider."""
    provider = OpenAIProvider(api_key=os.environ["OPENAI_API_KEY"])
    app = CoFlowAi()
    app.models.register("gpt-4o-mini", provider=provider, ...)
    
    agent = Agent(
        "assistant",
        "Answer concisely",
        "gpt-4o-mini"
    )
    
    result = await app.run(
        agent,
        "What is 2+2?",
        policy=ExecutionPolicy(max_cost=0.01)
    )
    
    assert result.succeeded
    assert "4" in result.output.lower()
```

---

## Performance Optimization

### Parallel Execution

```python
# Use parallel node for independent operations
workflow = (
    Workflow("optimized")
    .start(Agent("coordinator", "Plan", "gpt-4o"))
    .parallel(
        Agent("task1", "Process part 1", "gpt-4o"),
        Agent("task2", "Process part 2", "gpt-4o"),
        Agent("task3", "Process part 3", "gpt-4o"),
        max_concurrency=3  # All run in parallel
    )
)
```

### Caching

```python
from functools import lru_cache

@lru_cache(maxsize=1000)
@tool(description="Expensive computation", permission="compute")
def expensive_calculation(input_data: str) -> dict:
    """Cache repeated calculations."""
    # Expensive operation
    return {"result": compute(input_data)}
```

### Streaming for Long Responses

```python
# Stream for better UX on long outputs
async for chunk in app.stream(agent, "Write a long essay"):
    if chunk.is_final:
        # Stream complete
        result = chunk.metadata["result"]
        print(f"\n\nCost: ${result.usage.total_cost:.4f}")
    else:
        # Incremental output
        print(chunk.text, end="", flush=True)
```

### Model Selection by Cost

```python
# Use cheaper models for simple tasks
simple_agent = Agent("helper", "Help", "gpt-4o-mini")
complex_agent = Agent("analyst", "Analyze", "gpt-4o")

# Route based on task complexity
if is_simple_task(request):
    result = await app.run(simple_agent, request)
else:
    result = await app.run(complex_agent, request)
```

### Budget Management

```python
# Strict budgets prevent runaway costs
policy = ExecutionPolicy(
    max_cost=0.50,           # Hard limit
    max_tokens=50_000,       # Token budget
    max_model_calls=10,      # Limit iterations
)

try:
    result = await app.run(agent, request, policy=policy)
except BudgetExceededError:
    # Handle budget exhaustion
    logger.warning("Budget exceeded, using cached response")
```

---

## Production Deployment

### Durable Persistence

```python
from coflowai.persistence.postgres import PostgresEventStore, PostgresStateStore

# Production-ready persistence
event_store = PostgresEventStore(
    dsn="postgresql://user:pass@host/db"
)
state_store = PostgresStateStore(
    dsn="postgresql://user:pass@host/db",
    lock_lease_seconds=300
)

app = CoFlowAi(
    event_store=event_store,
    state_store=state_store
)
```

### Distributed Workers

```python
from coflowai.distributed import ExecutionService, RedisJobQueue
import redis.asyncio as redis

# Setup service
redis_client = redis.Redis(host="redis", port=6379)
queue = RedisJobQueue(redis_client)
service = ExecutionService(queue=queue, runtime=app.runtime)

# Register workflows
service.register(workflow)

# In web server: submit jobs
execution_id = await service.submit(workflow, request)
return {"execution_id": execution_id}

# In worker process: process jobs
worker = Worker(service=service, concurrency=4)
await worker.run()
```

### Observability

```python
from coflowai.observability import configure_logging
import structlog

# Structured logging
configure_logging(
    level="INFO",
    format="json",
    redact_secrets=True,
    redact_patterns=[
        r"sk-[a-zA-Z0-9]+",
        r"Bearer [a-zA-Z0-9]+",
    ]
)

logger = structlog.get_logger()

# Execution
result = await app.run(agent, request)

# Log with context
logger.info(
    "execution_completed",
    execution_id=result.execution_id,
    cost=result.usage.total_cost,
    duration=result.usage.execution_time_seconds,
)
```

### Metrics Collection

```python
from datadog import statsd

async def run_with_metrics(agent, request):
    """Execute with metrics tracking."""
    start = time.time()
    
    try:
        result = await app.run(agent, request)
        
        # Success metrics
        statsd.increment("agent.execution.success")
        statsd.timing("agent.execution.duration", time.time() - start)
        statsd.gauge("agent.execution.cost", result.usage.total_cost)
        statsd.gauge("agent.execution.tokens", result.usage.total_tokens)
        
        return result
        
    except Exception as e:
        statsd.increment(f"agent.execution.error.{type(e).__name__}")
        raise
```

### Health Checks

```python
from fastapi import FastAPI, HTTPException

app_api = FastAPI()

@app_api.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        # Check database connectivity
        await app.state_store.health_check()
        
        # Check model provider
        await app.models.get("gpt-4o").provider.health_check()
        
        return {"status": "healthy"}
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))
```

### Graceful Shutdown

```python
import signal
import asyncio

shutdown_event = asyncio.Event()

def handle_shutdown(signum, frame):
    """Handle shutdown signal."""
    shutdown_event.set()

signal.signal(signal.SIGTERM, handle_shutdown)
signal.signal(signal.SIGINT, handle_shutdown)

async def worker_loop():
    """Worker with graceful shutdown."""
    worker = Worker(service=service, concurrency=4)
    
    # Run until shutdown
    worker_task = asyncio.create_task(worker.run())
    await shutdown_event.wait()
    
    # Graceful shutdown
    logger.info("Shutdown signal received, finishing current executions...")
    worker.stop()
    await worker_task
    logger.info("Worker stopped")
```

---

## Next Steps

- **[Advanced Topics](./ADVANCED.md)** - Security, optimization, advanced patterns
- **[API Reference](./API_REFERENCE.md)** - Complete API documentation
- **[Examples](../examples/)** - More working examples
