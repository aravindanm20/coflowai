# Advanced Topics

Advanced patterns, optimization strategies, and production considerations for CoFlowAi.

## Table of Contents

- [Distributed Execution](#distributed-execution)
- [Security](#security)
- [Advanced Workflows](#advanced-workflows)
- [Performance Tuning](#performance-tuning)
- [Monitoring and Debugging](#monitoring-and-debugging)
- [Custom Providers](#custom-providers)
- [Plugin Development](#plugin-development)

---

## Distributed Execution

### Architecture

CoFlowAi's distributed architecture separates job submission from execution:

```
┌─────────────┐         ┌──────────────┐         ┌─────────────┐
│  API Server │────────>│  Job Queue   │<────────│   Worker    │
│   (submit)  │         │  (Redis)     │         │ Pool (N×4)  │
└─────────────┘         └──────────────┘         └─────────────┘
                                │
                                │
                        ┌───────v────────┐
                        │  State Store   │
                        │  (Postgres)    │
                        └────────────────┘
```

### Setup

#### 1. Configure Persistence

```python
from coflowai import CoFlowAi
from coflowai.persistence.postgres import PostgresEventStore, PostgresStateStore

# Shared persistence for all workers
event_store = PostgresEventStore(
    dsn="postgresql://user:pass@postgres:5432/coflowai"
)

state_store = PostgresStateStore(
    dsn="postgresql://user:pass@postgres:5432/coflowai",
    lock_lease_seconds=300  # 5-minute lock lease
)

app = CoFlowAi(
    event_store=event_store,
    state_store=state_store
)
```

#### 2. Setup Job Queue

```python
from coflowai.distributed import ExecutionService, RedisJobQueue
import redis.asyncio as redis

# Create job queue
redis_client = redis.Redis(
    host="redis",
    port=6379,
    decode_responses=False
)

queue = RedisJobQueue(
    client=redis_client,
    queue_name="coflowai:jobs",
    visibility_timeout=600,  # 10 minutes
    max_retries=3
)

# Create execution service
service = ExecutionService(
    queue=queue,
    runtime=app.runtime
)

# Register workflows
service.register(my_workflow)
service.register(another_workflow)
```

#### 3. Submit Jobs (API Server)

```python
from fastapi import FastAPI

api = FastAPI()

@api.post("/execute")
async def execute_workflow(request: dict):
    """Submit workflow for execution."""
    execution_id = await service.submit(
        executable=my_workflow,
        request=request["input"],
        policy=ExecutionPolicy(max_cost=1.00)
    )
    
    return {
        "execution_id": execution_id,
        "status": "queued"
    }

@api.get("/status/{execution_id}")
async def get_status(execution_id: str):
    """Check execution status."""
    state = await app.state_store.get(execution_id)
    return {
        "execution_id": execution_id,
        "status": state.status,
        "progress": state.metadata.get("progress")
    }
```

#### 4. Run Workers

```python
from coflowai.distributed import Worker

async def run_worker():
    """Worker process."""
    worker = Worker(
        service=service,
        concurrency=4,  # Process 4 jobs concurrently
        poll_interval_seconds=1.0,
        max_jobs=1000  # Restart after 1000 jobs (optional)
    )
    
    await worker.run()

if __name__ == "__main__":
    import asyncio
    asyncio.run(run_worker())
```

**Or via CLI:**

```bash
coflowai worker app.py --concurrency 4 --poll-interval 1.0
```

### Lock Leasing

Workers acquire locks on executions to prevent duplicate processing:

```python
# Worker A acquires lock
lock = await state_store.acquire_lock(execution_id, lease_seconds=300)

# Worker A processes execution
await runtime.execute(...)

# If Worker A crashes, lock expires after 300s
# Worker B can then acquire the lock and resume

# Worker A releases lock on completion
await state_store.release_lock(execution_id, lock.token)
```

**Key Properties:**
- Fenced tokens prevent ABA problem
- Atomic test-and-set operations
- Configurable lease duration
- Automatic expiry on crash

### At-Least-Once Delivery

Jobs may be delivered multiple times, but executions are idempotent:

1. **Checkpointing** - Completed nodes are skipped on resume
2. **Tool Idempotency Keys** - Side-effect tools get deterministic keys
3. **State Versioning** - Optimistic concurrency control prevents conflicts

```python
# Job redelivered after timeout
# Worker resumes from last checkpoint
result = await app.resume(execution_id, workflow)

# Completed nodes are skipped
# Only incomplete work is re-executed
```

### Scalability

Scale workers horizontally:

```bash
# Machine 1
coflowai worker app.py --concurrency 4

# Machine 2
coflowai worker app.py --concurrency 4

# Machine 3
coflowai worker app.py --concurrency 8

# Total: 16 concurrent executions across 3 machines
```

**Best Practices:**
- Set `concurrency` based on CPU cores
- Monitor queue depth
- Scale workers based on queue latency
- Use separate queues for priority tiers

---

## Security

### Permission Model

#### Namespaced Permissions

```python
from coflowai.policies import PermissionSet

# Define permission hierarchy
permissions = PermissionSet([
    "database.users.read",
    "database.users.write",
    "database.orders.read",
    "api.external.payment.*",
    "api.internal.*",
])

# Check permissions
permissions.check("database.users.read")       # ✓
permissions.check("database.users.delete")     # ✗
permissions.check("api.external.payment.charge") # ✓
permissions.check("api.internal.analytics")     # ✓
```

#### Least Privilege

```python
# Agent A: Read-only
agent_a = Agent("reader", "Read data", "gpt-4o", tools=[read_db])
permissions_a = PermissionSet(["database.*.read"])

# Agent B: Can write
agent_b = Agent("writer", "Write data", "gpt-4o", tools=[write_db])
permissions_b = PermissionSet(["database.*.write"])

# Workflow inherits intersection of permissions
workflow = (
    Workflow("pipeline")
    .start(agent_a)  # Can only read
    .then(agent_b)   # Can only write (no read!)
)
```

#### Dynamic Permission Checks

```python
@tool(description="Update record", permission="database.write")
async def update_record(table: str, record_id: str, data: dict) -> dict:
    """Permission checked at runtime."""
    context = get_current_context()
    
    # Additional runtime checks
    if not context.state.get("user_authorized"):
        raise ToolPermissionDeniedError("User not authorized")
    
    # Proceed
    return await db.update(table, record_id, data)
```

### Secret Management

#### Redaction

```python
from coflowai.observability import configure_logging

configure_logging(
    redact_secrets=True,
    redact_patterns=[
        r"sk-[a-zA-Z0-9]{20,}",        # OpenAI keys
        r"sk-ant-[a-zA-Z0-9-]{20,}",   # Anthropic keys
        r"Bearer [a-zA-Z0-9\-._~+/]+", # Bearer tokens
        r"\bAKIA[A-Z0-9]{16}\b",       # AWS access keys
        r"password['\"]?\s*[:=]\s*['\"]?([^'\"]+)", # Passwords
    ]
)

# Secrets automatically redacted from:
# - Logs
# - Events
# - Traces
# - Error messages
```

#### Environment-Based Configuration

```python
import os

# Never hardcode secrets
provider = OpenAIProvider(
    api_key=os.environ["OPENAI_API_KEY"]
)

# Use secret managers in production
from azure.keyvault.secrets import SecretClient

secret_client = SecretClient(vault_url="...", credential=...)
api_key = secret_client.get_secret("openai-api-key").value
```

### Sandboxing

#### Subprocess Sandbox

```python
from coflowai.tools.sandbox import SubprocessSandbox

# Isolate untrusted tools
sandbox = SubprocessSandbox(
    cpu_seconds=5,              # CPU time limit
    memory_mb=256,              # Memory limit
    allowed_paths=["/tmp"],     # Filesystem access
    blocked_syscalls=["socket"], # Block network
    env_inherit=False,          # Scrubbed environment
)

app = CoFlowAi(sandbox=sandbox)

@tool(description="Run user code", permission="code.execute")
async def execute_code(code: str) -> str:
    """Executed in isolated subprocess."""
    # CPU, memory, and syscall limits enforced by OS
    return eval(code)
```

#### Container Sandbox

```python
# For maximum isolation, run workers in containers
# docker-compose.yml
services:
  worker:
    image: coflowai-worker
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 2G
    security_opt:
      - no-new-privileges:true
    cap_drop:
      - ALL
```

### Input Validation

#### Tool Input Validation

```python
from pydantic import BaseModel, validator, Field

class ChargeInput(BaseModel):
    customer_id: str = Field(..., regex=r'^cus_[a-zA-Z0-9]{14,}$')
    amount: float = Field(..., gt=0, le=10000)
    currency: str = Field(..., regex=r'^[A-Z]{3}$')
    
    @validator("amount")
    def validate_amount(cls, v):
        if v > 1000:
            # Require additional validation for large amounts
            raise ValueError("Amount over $1000 requires approval")
        return v

@tool(
    description="Charge payment",
    permission="payments.write",
    side_effect=True,
    idempotent=True
)
async def charge_payment(input: ChargeInput) -> dict:
    """Pydantic validation runs before execution."""
    return await payment_provider.charge(**input.dict())
```

#### Output Validation

```python
from pydantic import BaseModel

class SafeOutput(BaseModel):
    """Ensure output doesn't contain sensitive data."""
    result: str
    
    @validator("result")
    def no_secrets(cls, v):
        if "password" in v.lower() or "secret" in v.lower():
            raise ValueError("Output contains sensitive keywords")
        return v

agent = Agent(
    "assistant",
    "Help user",
    "gpt-4o",
    output_schema=SafeOutput
)
```

### Audit Logging

```python
# All tool calls automatically logged
for event in app.events.get_events(execution_id):
    if event.type == "tool.call_started":
        audit_log.info(
            "tool_called",
            execution_id=execution_id,
            tool_name=event.data["tool_name"],
            arguments=event.data["arguments"],
            user=event.metadata.get("user_id"),
            timestamp=event.timestamp
        )
```

---

## Advanced Workflows

### Dynamic Workflow Generation

```python
def create_workflow(num_agents: int) -> Workflow:
    """Generate workflow dynamically."""
    workflow = Workflow(f"dynamic-{num_agents}")
    workflow.start(Agent("coordinator", "Coordinate", "gpt-4o"))
    
    # Add N parallel agents
    agents = [
        Agent(f"worker-{i}", f"Process part {i}", "gpt-4o")
        for i in range(num_agents)
    ]
    workflow.parallel(*agents, max_concurrency=5)
    
    workflow.then(Agent("aggregator", "Aggregate", "gpt-4o"))
    return workflow

# Create workflow based on data size
data_size = len(input_data)
num_workers = min(data_size // 1000, 10)
workflow = create_workflow(num_workers)
```

### Conditional Sub-Workflows

```python
def route_by_complexity(context: ExecutionContext) -> str:
    """Route based on task complexity."""
    complexity = analyze_complexity(context.state["task"])
    if complexity > 0.8:
        return "complex"
    elif complexity > 0.5:
        return "medium"
    return "simple"

# Different workflows for different complexity
simple_workflow = Workflow("simple").start(Agent("fast", "", "gpt-4o-mini"))
medium_workflow = Workflow("medium").start(Agent("standard", "", "gpt-4o"))
complex_workflow = (
    Workflow("complex")
    .start(Agent("planner", "", "gpt-4o"))
    .parallel(Agent("a", "", "gpt-4o"), Agent("b", "", "gpt-4o"))
    .then(Agent("synthesizer", "", "gpt-4o"))
)

main_workflow = (
    Workflow("adaptive")
    .start(Agent("analyzer", "Analyze task", "gpt-4o"))
    .route(
        condition=route_by_complexity,
        routes={
            "simple": simple_workflow,
            "medium": medium_workflow,
            "complex": complex_workflow,
        }
    )
)
```

### Workflow Composition

```python
# Reusable workflow components

def create_review_workflow(reviewer_count: int) -> Workflow:
    """Multi-stage review workflow."""
    return (
        Workflow("review")
        .parallel(*[
            Agent(f"reviewer-{i}", "Review", "gpt-4o")
            for i in range(reviewer_count)
        ])
        .then(Agent("aggregator", "Aggregate reviews", "gpt-4o"))
    )

def create_approval_workflow(approvers: list[str]) -> Workflow:
    """Sequential approval workflow."""
    workflow = Workflow("approval")
    for approver in approvers:
        workflow.approve(
            f"Approval required from {approver}",
            metadata={"approver": approver}
        )
    return workflow

# Compose into main workflow
main = (
    Workflow("content-pipeline")
    .start(Agent("creator", "Create content", "gpt-4o"))
    .then(create_review_workflow(reviewer_count=3))
    .then(create_approval_workflow(approvers=["manager", "director"]))
    .then(Agent("publisher", "Publish", "gpt-4o"))
)
```

### Workflow Versioning

```python
# Version 1
workflow_v1 = Workflow("pipeline", version="1.0.0")

# Version 2 with breaking changes
workflow_v2 = Workflow("pipeline", version="2.0.0")

# Execution references specific version
execution_id = await app.run(workflow_v1, request)

# Resume validates version compatibility
try:
    await app.resume(execution_id, workflow_v2)
except WorkflowVersionMismatchError:
    # Version incompatible, cannot resume
    logger.error("Workflow version mismatch")
```

---

## Performance Tuning

### Model Selection Strategy

```python
class ModelSelector:
    """Intelligent model selection."""
    
    def select_model(self, task_complexity: float, budget: float) -> str:
        """Select model based on task and budget."""
        if budget < 0.01:
            return "gpt-4o-mini"
        
        if task_complexity > 0.8:
            return "gpt-4o"
        elif task_complexity > 0.5:
            return "gpt-4o"
        else:
            return "gpt-4o-mini"

selector = ModelSelector()
model = selector.select_model(complexity=0.7, budget=0.50)
agent = Agent("assistant", "Help", model)
```

### Caching Strategies

#### Response Caching

```python
from functools import lru_cache
import hashlib

class CachedExecutor:
    """Cache execution results."""
    
    def __init__(self, app: CoFlowAi):
        self.app = app
        self.cache = {}
    
    def cache_key(self, agent: Agent, request: str) -> str:
        """Generate cache key."""
        key_input = f"{agent.name}:{agent.model}:{request}"
        return hashlib.sha256(key_input.encode()).hexdigest()
    
    async def run(self, agent: Agent, request: str, ttl: int = 3600):
        """Run with caching."""
        key = self.cache_key(agent, request)
        
        if key in self.cache:
            result, timestamp = self.cache[key]
            if time.time() - timestamp < ttl:
                return result
        
        result = await self.app.run(agent, request)
        self.cache[key] = (result, time.time())
        return result
```

#### Tool Result Caching

```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def cached_api_call(endpoint: str, params: str) -> dict:
    """Cache API responses."""
    # params must be hashable (string, not dict)
    return requests.get(endpoint, json.loads(params)).json()

@tool(description="Fetch data", permission="api.read")
async def fetch_data(endpoint: str, **params) -> dict:
    """Tool with caching."""
    params_str = json.dumps(params, sort_keys=True)
    return cached_api_call(endpoint, params_str)
```

### Batch Processing

```python
async def batch_process(requests: list[str], batch_size: int = 10):
    """Process requests in batches."""
    results = []
    
    for i in range(0, len(requests), batch_size):
        batch = requests[i:i + batch_size]
        
        # Process batch concurrently
        tasks = [app.run(agent, req) for req in batch]
        batch_results = await asyncio.gather(*tasks)
        results.extend(batch_results)
    
    return results
```

### Connection Pooling

```python
from httpx import AsyncClient, Limits

# Configure connection pooling for providers
limits = Limits(
    max_keepalive_connections=20,
    max_connections=100,
    keepalive_expiry=30.0
)

client = AsyncClient(limits=limits)
provider = OpenAIProvider(api_key="...", client=client)
```

### Token Budgeting

```python
from coflowai.context import ContextBuilder

# Configure context builder with strict budgets
context_builder = ContextBuilder(
    max_tokens=4000,
    section_budgets={
        "instructions": 500,
        "workflow_state": 300,
        "memory": 500,
        "tools": 500,
        "conversation": 2000,
    },
    summarization_strategy="extractive"
)

app = CoFlowAi(context_builder=context_builder)
```

---

## Monitoring and Debugging

### Structured Logging

```python
import structlog

# Configure structured logging
structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer()
    ]
)

logger = structlog.get_logger()

# Log with context
logger.info(
    "agent_execution_started",
    execution_id=execution_id,
    agent_name=agent.name,
    model=agent.model
)
```

### Distributed Tracing

```python
from opentelemetry import trace
from opentelemetry.exporter.jaeger import JaegerExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

# Setup OpenTelemetry
jaeger_exporter = JaegerExporter(
    agent_host_name="localhost",
    agent_port=6831,
)

provider = TracerProvider()
provider.add_span_processor(BatchSpanProcessor(jaeger_exporter))
trace.set_tracer_provider(provider)

# Configure CoFlowAi tracer
from coflowai.observability import OpenTelemetryTracer

tracer = OpenTelemetryTracer(trace.get_tracer("coflowai"))
app = CoFlowAi(tracer=tracer)

# Traces automatically sent to Jaeger
result = await app.run(agent, request)
```

### Metrics Export

```python
from prometheus_client import Counter, Histogram, Gauge, start_http_server

# Define metrics
executions_total = Counter(
    "coflowai_executions_total",
    "Total executions",
    ["status", "agent"]
)

execution_duration = Histogram(
    "coflowai_execution_duration_seconds",
    "Execution duration",
    ["agent"]
)

execution_cost = Histogram(
    "coflowai_execution_cost_dollars",
    "Execution cost",
    ["agent"]
)

# Record metrics
async def run_with_metrics(agent, request):
    start = time.time()
    
    try:
        result = await app.run(agent, request)
        
        executions_total.labels(status="success", agent=agent.name).inc()
        execution_duration.labels(agent=agent.name).observe(time.time() - start)
        execution_cost.labels(agent=agent.name).observe(result.usage.total_cost)
        
        return result
    except Exception as e:
        executions_total.labels(status="failed", agent=agent.name).inc()
        raise

# Start metrics server
start_http_server(9090)
```

### Debugging Failed Executions

```python
async def debug_execution(execution_id: str):
    """Debug failed execution."""
    # Get execution state
    state = await app.state_store.get(execution_id)
    print(f"Status: {state.status}")
    print(f"Error: {state.error}")
    
    # Get events
    events = await app.events.get_events(execution_id)
    for event in events:
        if event.type in ["execution.failed", "model.call_failed", "tool.call_failed"]:
            print(f"{event.timestamp}: {event.type}")
            print(f"  Data: {event.data}")
    
    # Get trace
    trace = app.trace(execution_id)
    print(trace.render())
    
    # Replay execution
    replay = await app.replay(execution_id)
    for step in replay.steps:
        print(f"Step {step.sequence}: {step.node_id} - {step.status}")
```

---

## Custom Providers

### Implementing a Provider

```python
from coflowai.models import ModelProvider, ModelRequest, ModelResponse
from coflowai.models import StreamChunk
from typing import AsyncIterator

class CustomProvider(ModelProvider):
    """Custom model provider implementation."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.client = CustomAPIClient(api_key)
    
    async def call(self, request: ModelRequest) -> ModelResponse:
        """Synchronous model call."""
        response = await self.client.complete(
            model=request.model,
            messages=request.messages,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            tools=request.tools,
        )
        
        return ModelResponse(
            content=response["content"],
            tool_calls=self._parse_tool_calls(response.get("tool_calls", [])),
            usage={
                "input_tokens": response["usage"]["prompt_tokens"],
                "output_tokens": response["usage"]["completion_tokens"],
            },
            raw_response=response,
        )
    
    async def stream(self, request: ModelRequest) -> AsyncIterator[StreamChunk]:
        """Streaming model call."""
        async for chunk in self.client.stream(
            model=request.model,
            messages=request.messages,
        ):
            if chunk["type"] == "content":
                yield StreamChunk(
                    text=chunk["text"],
                    is_final=False,
                )
            elif chunk["type"] == "done":
                yield StreamChunk(
                    text="",
                    is_final=True,
                    metadata={"usage": chunk["usage"]},
                )
```

### Provider Testing

```python
from coflowai.providers.testing import MockTransport

class TestCustomProvider:
    """Test custom provider with mock transport."""
    
    @pytest.mark.asyncio
    async def test_call(self):
        mock = MockTransport(responses=[
            {"content": "Hello!", "usage": {"prompt_tokens": 10, "completion_tokens": 5}}
        ])
        
        provider = CustomProvider(api_key="test", transport=mock)
        
        response = await provider.call(ModelRequest(
            model="custom-model",
            messages=[{"role": "user", "content": "Hi"}]
        ))
        
        assert response.content == "Hello!"
        assert response.usage["input_tokens"] == 10
```

---

## Plugin Development

### Creating a Plugin

```python
from coflowai.plugins import Plugin

class MyPlugin(Plugin):
    """Custom plugin implementation."""
    
    def __init__(self, config: dict):
        self.config = config
    
    async def on_startup(self, app: CoFlowAi):
        """Called when application starts."""
        print(f"Plugin {self.name} starting")
        # Register tools, models, etc.
    
    async def on_shutdown(self, app: CoFlowAi):
        """Called when application shuts down."""
        print(f"Plugin {self.name} shutting down")
    
    async def on_execution_started(self, execution_id: str, context: ExecutionContext):
        """Called when execution starts."""
        pass
    
    async def on_execution_completed(self, execution_id: str, result: ExecutionResult):
        """Called when execution completes."""
        pass
```

### Registering a Plugin

```python
# Via code
app = CoFlowAi(plugins=[MyPlugin(config={"key": "value"})])

# Via entry points (pyproject.toml)
[project.entry-points."coflowai.plugins"]
myplugin = "myplugin:MyPlugin"
```

### Plugin Examples

#### Logging Plugin

```python
class LoggingPlugin(Plugin):
    """Log all executions."""
    
    async def on_execution_completed(self, execution_id: str, result: ExecutionResult):
        logger.info(
            "execution_completed",
            execution_id=execution_id,
            status=result.status,
            cost=result.usage.total_cost,
            duration=result.usage.execution_time_seconds
        )
```

#### Metrics Plugin

```python
class MetricsPlugin(Plugin):
    """Collect metrics."""
    
    async def on_execution_completed(self, execution_id: str, result: ExecutionResult):
        statsd.increment("executions.completed")
        statsd.gauge("executions.cost", result.usage.total_cost)
        statsd.timing("executions.duration", result.usage.execution_time_seconds)
```

---

## Next Steps

- **[API Reference](./API_REFERENCE.md)** - Complete API documentation
- **[User Guide](./USER_GUIDE.md)** - Common patterns and recipes
- **[Examples](../examples/)** - Working code examples
- **[Contributing](./CONTRIBUTING.md)** - Contribute to CoFlowAi
