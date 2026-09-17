# Troubleshooting Guide

Common issues and solutions for CoFlowAi.

## Table of Contents

- [Installation Issues](#installation-issues)
- [Execution Errors](#execution-errors)
- [Provider Issues](#provider-issues)
- [Performance Issues](#performance-issues)
- [Testing Issues](#testing-issues)
- [Distributed Execution Issues](#distributed-execution-issues)

---

## Installation Issues

### Python Version Error

**Problem:**
```
ERROR: Package 'coflowai' requires a different Python: 3.11.0 not in '>=3.12'
```

**Solution:**
CoFlowAi requires Python 3.12+. Upgrade your Python installation:

```bash
# Check version
python --version

# Upgrade using pyenv
pyenv install 3.12.0
pyenv global 3.12.0

# Or use conda
conda install python=3.12
```

### Missing Dependencies

**Problem:**
```
ModuleNotFoundError: No module named 'httpx'
```

**Solution:**
Install provider-specific dependencies:

```bash
# For OpenAI
pip install "coflowai[openai]"

# For all providers
pip install "coflowai[all]"
```

### Import Errors

**Problem:**
```python
from coflowai.providers.openai import OpenAIProvider
ImportError: cannot import name 'OpenAIProvider'
```

**Solution:**
Provider modules are not imported by default. Install the provider extra:

```bash
pip install "coflowai[openai]"
```

---

## Execution Errors

### BudgetExceededError

**Problem:**
```
BudgetExceededError: Maximum cost of $1.00 exceeded (spent: $1.23)
```

**Solution:**
Increase budget or optimize execution:

```python
# Increase budget
policy = ExecutionPolicy(max_cost=2.00)

# Or use cheaper model
agent.model = "gpt-4o-mini"

# Or reduce max_iterations
agent.max_iterations = 5
```

### ExecutionTimeoutError

**Problem:**
```
ExecutionTimeoutError: Execution timed out after 120 seconds
```

**Solution:**
Increase timeout or optimize workflow:

```python
# Increase timeout
policy = ExecutionPolicy(timeout_seconds=600)

# Or add agent-level timeout
agent = Agent(..., timeout_seconds=60)

# Or optimize expensive operations
```

### Max Iterations Reached

**Problem:**
```
Agent reached maximum iterations (10) without completing
```

**Solution:**

```python
# Increase iterations
agent = Agent(..., max_iterations=20)

# Or simplify task
# Or add explicit termination condition in instructions
agent = Agent(
    "assistant",
    """Complete the task. 
    IMPORTANT: Provide final answer when ready, don't keep iterating.""",
    "gpt-4o"
)
```

### Tool Permission Denied

**Problem:**
```
ToolPermissionDeniedError: Tool 'write_file' requires permission 'filesystem.write'
```

**Solution:**
Grant required permissions:

```python
from coflowai.policies import PermissionSet

permissions = PermissionSet([
    "filesystem.read",
    "filesystem.write",  # Add missing permission
])

app = CoFlowAi(permissions=permissions)
```

### Tool Validation Error

**Problem:**
```
ToolValidationError: Invalid arguments for tool 'update_customer'
```

**Solution:**
Check tool input schema:

```python
# Add better validation error messages
from pydantic import BaseModel, validator

class UpdateInput(BaseModel):
    customer_id: str
    email: str
    
    @validator("email")
    def validate_email(cls, v):
        if "@" not in v:
            raise ValueError(f"Invalid email format: {v}")
        return v

@tool(...)
async def update_customer(input: UpdateInput):
    ...
```

### Workflow Compilation Error

**Problem:**
```
WorkflowCompilationError: Node 'reviewer' requires permission 'code.review' but workflow only grants ['code.read']
```

**Solution:**
Grant all required permissions:

```python
# Check what permissions are needed
workflow.compile()  # This will list all required permissions

# Grant them
permissions = PermissionSet([
    "code.read",
    "code.review",  # Add missing
    "code.write",
])
```

---

## Provider Issues

### Authentication Error

**Problem:**
```
ModelAuthenticationError: Invalid API key
```

**Solution:**
Verify API key is correct:

```python
import os

# Use environment variables
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not set")

provider = OpenAIProvider(api_key=api_key)
```

### Rate Limit Error

**Problem:**
```
ModelRateLimitError: Rate limit exceeded, retry after 20 seconds
```

**Solution:**
Configure retry policy:

```python
from coflowai.policies import RetryPolicy

retry_policy = RetryPolicy(
    max_retries=5,
    backoff_seconds=2.0,
    backoff_multiplier=2.0,
    jitter=True,
)

policy = ExecutionPolicy(retry_policy=retry_policy)
```

Or implement rate limiting in application:

```python
import asyncio
from collections import deque
from datetime import datetime, timedelta

class RateLimiter:
    def __init__(self, calls_per_minute: int):
        self.calls_per_minute = calls_per_minute
        self.calls = deque()
    
    async def acquire(self):
        now = datetime.now()
        # Remove old calls
        while self.calls and now - self.calls[0] > timedelta(minutes=1):
            self.calls.popleft()
        
        if len(self.calls) >= self.calls_per_minute:
            wait = (self.calls[0] + timedelta(minutes=1) - now).total_seconds()
            await asyncio.sleep(wait)
        
        self.calls.append(now)

limiter = RateLimiter(calls_per_minute=50)

async def run_with_limit(agent, request):
    await limiter.acquire()
    return await app.run(agent, request)
```

### Model Not Found

**Problem:**
```
ModelNotFoundError: Model 'gpt-5' not found in registry
```

**Solution:**
Register the model first:

```python
# List registered models
print(app.models.list())

# Register missing model
app.models.register("gpt-4o", provider=provider, ...)

# Or use existing model
agent.model = "gpt-4o"
```

### Structured Output Not Supported

**Problem:**
```
Model 'custom-model' does not support structured output
```

**Solution:**
Use a model with structured output support, or structured output will be emulated:

```python
# Check capabilities
capabilities = app.models.get("custom-model").capabilities
print(capabilities.structured_output)

# Use compatible model
agent.model = "gpt-4o"  # Supports structured output

# Or let framework emulate (via tool calls)
# This happens automatically if the model doesn't support it natively
```

### Connection Timeout

**Problem:**
```
httpx.ConnectTimeout: Connection timeout after 30 seconds
```

**Solution:**
Increase provider timeout:

```python
provider = OpenAIProvider(
    api_key="...",
    timeout=120  # Increase from default 60
)
```

---

## Performance Issues

### Slow Execution

**Problem:**
Execution takes too long.

**Diagnosis:**
Check execution trace:

```python
trace = app.trace(execution_id)
print(trace.render())

# Look for slow operations
for span in trace.spans:
    if span.duration > 10.0:  # > 10 seconds
        print(f"Slow: {span.name} took {span.duration}s")
```

**Solutions:**

1. **Use parallel execution:**
```python
# Instead of sequential
workflow.then(agent_a).then(agent_b).then(agent_c)

# Use parallel
workflow.parallel(agent_a, agent_b, agent_c)
```

2. **Use faster models:**
```python
# Replace expensive model
agent.model = "gpt-4o-mini"  # Faster and cheaper
```

3. **Reduce max_iterations:**
```python
agent.max_iterations = 5  # Fewer iterations
```

4. **Cache repeated operations:**
```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def expensive_operation(input_data):
    ...
```

### High Costs

**Problem:**
Execution costs too much.

**Solution:**

1. **Set cost budgets:**
```python
policy = ExecutionPolicy(max_cost=0.50)
```

2. **Use cheaper models:**
```python
agent.model = "gpt-4o-mini"  # 60% cheaper
```

3. **Reduce token usage:**
```python
# Shorter instructions
agent.instructions = "Summarize concisely."

# Limit conversation history
context_builder = ContextBuilder(
    max_tokens=2000,
    section_budgets={"conversation": 500}
)
```

4. **Monitor and alert:**
```python
result = await app.run(agent, request)
if result.usage.total_cost > 1.0:
    logger.warning(f"High cost: ${result.usage.total_cost:.2f}")
```

### Memory Usage

**Problem:**
High memory consumption.

**Solutions:**

1. **Use streaming:**
```python
async for chunk in app.stream(agent, request):
    process(chunk)
# Lower memory footprint
```

2. **Limit context size:**
```python
context_builder = ContextBuilder(max_tokens=4000)
```

3. **Clear old executions:**
```python
# Cleanup old state
await state_store.cleanup(older_than_days=7)
```

---

## Testing Issues

### Tests Require API Keys

**Problem:**
Tests fail without API keys.

**Solution:**
Use fake model provider:

```python
from coflowai.testing import build_test_app

# No API key needed
app, model = build_test_app(["response"])
```

### Non-Deterministic Tests

**Problem:**
Tests fail intermittently.

**Solution:**
Use FakeModelProvider for deterministic tests:

```python
from coflowai.testing import build_test_app, tool_call

app, model = build_test_app(
    responses=[
        tool_call("get_data", id="123"),
        {"result": "data"}
    ],
    tools=[get_data]
)

# Always returns same responses
result = await app.run(agent, "test")
```

### Slow Tests

**Problem:**
Test suite takes too long.

**Solutions:**

1. **Use fake provider:**
```python
# Fast (no network)
app, model = build_test_app([...])
```

2. **Run in parallel:**
```bash
pytest tests -n auto
```

3. **Skip integration tests:**
```bash
pytest tests -m "not integration"
```

---

## Distributed Execution Issues

### Lock Contention

**Problem:**
```
StateLockError: Could not acquire lock for execution_123
```

**Solution:**
Increase lock lease duration or reduce concurrency:

```python
state_store = PostgresStateStore(
    dsn="...",
    lock_lease_seconds=600  # Increase from 300
)

# Or reduce worker concurrency
worker = Worker(service=service, concurrency=2)
```

### Job Redelivery

**Problem:**
Jobs executed multiple times.

**Diagnosis:**
Check visibility timeout:

```python
queue = RedisJobQueue(
    client=redis_client,
    visibility_timeout=600  # Should be > expected execution time
)
```

**Solution:**
This is expected (at-least-once delivery). Ensure idempotency:

```python
@tool(side_effect=True, idempotent=True)  # Safe to retry
async def charge_payment(...):
    # Idempotency key automatically provided
    ...
```

### Worker Crashes

**Problem:**
Workers crash unexpectedly.

**Diagnosis:**
Check logs:

```python
import structlog
logger = structlog.get_logger()

# Worker logs errors before crash
logger.error("worker_crashed", error=str(e))
```

**Solutions:**

1. **Handle errors gracefully:**
```python
async def safe_worker():
    try:
        worker = Worker(service=service)
        await worker.run()
    except Exception as e:
        logger.error("worker_error", error=str(e))
        # Optionally restart
```

2. **Monitor worker health:**
```python
# Health check endpoint
@app.get("/worker/health")
async def worker_health():
    return {"status": "healthy", "concurrency": worker.concurrency}
```

3. **Use process supervisor:**
```bash
# supervisord
[program:coflowai_worker]
command=coflowai worker app.py --concurrency 4
autostart=true
autorestart=true
```

### Database Connection Pool Exhausted

**Problem:**
```
asyncpg.exceptions.TooManyConnectionsError: Cannot acquire connection
```

**Solution:**
Increase connection pool size:

```python
state_store = PostgresStateStore(
    dsn="...",
    min_pool_size=5,
    max_pool_size=20  # Increase
)
```

Or reduce worker concurrency:

```python
worker = Worker(service=service, concurrency=2)  # Fewer concurrent jobs
```

---

## Debugging Tips

### Enable Debug Logging

```python
import logging

logging.basicConfig(level=logging.DEBUG)

# Or structured logging
import structlog

structlog.configure(
    processors=[
        structlog.processors.add_log_level,
        structlog.dev.ConsoleRenderer()
    ],
    logger_factory=structlog.PrintLoggerFactory(),
)
```

### Inspect Execution Events

```python
# Get all events
events = await app.events.get_events(execution_id)

for event in events:
    print(f"{event.timestamp}: {event.type}")
    if event.type.endswith("failed"):
        print(f"  Error: {event.data}")
```

### Replay Execution

```python
# Replay step-by-step
replay = await app.replay(execution_id)

for step in replay.steps:
    print(f"Step {step.sequence}: {step.node_id}")
    print(f"  Status: {step.status}")
    print(f"  Output: {step.output}")
```

### Use Execution Trace

```python
trace = app.trace(execution_id)
print(trace.render())

# Find slow operations
for span in trace.spans:
    if span.duration > 5.0:
        print(f"Slow: {span.name} ({span.duration}s)")
```

---

## Getting Help

If you're still stuck:

1. **Search existing issues**: [GitHub Issues](https://github.com/aravindanm20/coflowai/issues)
2. **Ask in discussions**: [GitHub Discussions](https://github.com/aravindanm20/coflowai/discussions)
3. **Check documentation**: [Documentation](./README.md)
4. **Open an issue**: Include:
   - CoFlowAi version
   - Python version
   - Minimal reproducible example
   - Error messages and stack traces
   - Execution trace (if applicable)

## Reporting Bugs

When reporting bugs, include:

```python
import coflowai
import sys

print(f"CoFlowAi version: {coflowai.__version__}")
print(f"Python version: {sys.version}")
print(f"Platform: {sys.platform}")

# Minimal reproducible example
from coflowai import Agent
from coflowai.testing import build_test_app

app, _ = build_test_app(["Hello"])
agent = Agent("test", "Test", "fake-model")

result = await app.run(agent, "Hi")
# Include error output
```
