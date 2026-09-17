# Core Concepts

## Architecture Overview

CoFlowAi's architecture enforces a strict separation between nondeterministic AI decisions and deterministic execution control.

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

### Dependency Direction

Dependencies flow strictly one way: **API → Runtime → Interfaces → Adapters**

The core never imports provider SDKs. This design ensures:
- Provider independence
- Testability without network or API keys
- Minimal core dependencies (only `pydantic` and `typing-extensions`)

## The Executable Abstraction

Everything in CoFlowAi is an `Executable`:

```python
from abc import ABC, abstractmethod

class Executable(ABC):
    @abstractmethod
    async def execute(self, context: ExecutionContext) -> ExecutionResult:
        """Execute this unit of work."""
```

This single abstraction enables composition:
- `Agent` - AI reasoning loop
- `Workflow` - Orchestration graph
- `ParallelNode` - Concurrent execution
- `RouterNode` - Conditional branching
- `LoopNode` - Iteration with bounds
- `TransformNode` - Data transformation
- `HumanApproval` - Human-in-the-loop gates

All implement `Executable`, so they compose freely and nest arbitrarily.

## Execution Context

The `ExecutionContext` carries all state through execution:

```python
@dataclass
class ExecutionContext:
    execution_id: str           # Unique execution identifier
    request: str                # User's original request
    policy: ExecutionPolicy     # Budgets and limits
    state: dict[str, Any]       # Execution state (persisted)
    events: EventPublisher      # Event emission
    tracer: ExecutionTracer     # Nested trace spans
    usage: UsageTracker         # Cost and token tracking
    memory: MemoryStore         # Agent memory
    parent_context: Optional[ExecutionContext]  # For nested executables
```

**Key Properties:**
- **Immutable per execution** - No global mutable state
- **Hierarchical** - Child executables get derived contexts
- **Traceable** - Every operation is traced
- **Observable** - Events flow to event stores

## Execution Flow

### 1. Submission

```python
result = await app.run(workflow, request, policy=policy)
```

The runtime:
1. Generates a unique `execution_id`
2. Validates the workflow (if applicable)
3. Creates an `ExecutionContext`
4. Transitions state to `RUNNING`
5. Invokes `workflow.execute(context)`

### 2. State Machine

Executions follow a strict state machine:

```
PENDING → RUNNING → [COMPLETED | FAILED | CANCELLED]
            ↓
     WAITING_FOR_APPROVAL
            ↓
         RUNNING
```

Illegal transitions are rejected. Operators can force resume of `FAILED` executions.

### 3. Checkpointing

Checkpoints are created:
- Before each workflow node
- After each workflow node
- Before human approval gates
- On failure
- On explicit `checkpoint()` calls

```python
# Resume from last checkpoint
await runtime.resume(execution_id, workflow)
```

Completed nodes are skipped using checkpoint state. Workflow hash is verified for compatibility.

### 4. Completion

On completion, the runtime:
1. Emits `execution.completed` or `execution.failed` event
2. Records final usage
3. Persists the result
4. Releases any locks

## Agents

### Agent Reasoning Loop

```python
agent = Agent(
    name="researcher",
    instructions="You are a research assistant.",
    model="gpt-4o",
    tools=[search, analyze],
    max_iterations=10,
    temperature=0.7,
    output_schema=ResearchReport
)
```

The agent loop:

```
1. Build context (instructions + memory + conversation)
2. Call model
3. Process response:
   - If tool calls → execute tools → loop (max_iterations)
   - If text response → validate against schema (if any) → return
4. Repeat until:
   - Agent produces final output
   - max_iterations reached
   - Budget exceeded
   - Timeout
```

### Agent Context Building

The context builder assembles:

```
┌─────────────────────────────────────┐
│ System Instructions                 │ ← Always included
├─────────────────────────────────────┤
│ Workflow State (if in workflow)     │ ← Conditional
├─────────────────────────────────────┤
│ Memory (relevant items)             │ ← Conditional, BM25-scored
├─────────────────────────────────────┤
│ Retrieved Knowledge                 │ ← Conditional
├─────────────────────────────────────┤
│ Tool Results (recent)               │ ← Recent tool outputs
├─────────────────────────────────────┤
│ Conversation History                │ ← Sliding window
└─────────────────────────────────────┘
```

Per-section token budgets prevent any section from dominating. When over budget:
1. Drop low-relevance items
2. Summarize old turns
3. Compress tool output
4. Never blindly truncate

## Tools

### Tool Pipeline

Every tool call goes through a complete pipeline:

```
Registry → Permission Check → Input Validation → Rate Limit 
  → Timeout → Sandbox → Execute → Output Validation → Audit Event
```

```python
@tool(
    name="charge_payment",
    description="Charge a customer's payment method",
    permission="payments.write",
    timeout=10,
    retries=2,
    side_effect=True,
    idempotent=True,
)
async def charge_payment(customer_id: str, amount: float) -> dict:
    # Idempotency key: execution_id + node_id + tool_call_id
    # Safe to retry even with side effects
    return {"status": "charged", "amount": amount}
```

### Tool Properties

**Permissions:**
```python
permission="database.write"      # Exact match
permission="database.*"          # Wildcard
permission=["read.data", "write.logs"]  # Multiple
```

**Side Effects:**
- `side_effect=False` (default) - Safe to retry on any failure
- `side_effect=True, idempotent=False` - **Never** retried
- `side_effect=True, idempotent=True` - Retried with deterministic idempotency key

**Timeouts:**
```python
# Tool-specific timeout
@tool(timeout=5)

# Execution-wide timeout
policy = ExecutionPolicy(timeout_seconds=300)

# Agent-specific timeout
agent = Agent(..., timeout_seconds=60)
```

Timeouts propagate down: execution → workflow node → agent → tool.

### Sandboxed Execution

```python
from coflowai.tools.sandbox import SubprocessSandbox

app = CoFlowAi(
    sandbox=SubprocessSandbox(
        cpu_seconds=5,
        memory_mb=256,
        allowed_paths=["/tmp"],
    )
)
```

Untrusted tools run in isolated subprocesses:
- CPU and memory limits enforced by OS
- Scrubbed environment (secrets not inherited)
- Hard kill on timeout
- File descriptor limits

## Workflows

### Workflow Compiler

Workflows are validated **before execution**:

```python
workflow = (
    Workflow("pipeline", version="1")
    .start(agent_a)
    .then(agent_b)
    .parallel(agent_c, agent_d)
)

# Compilation checks:
# ✓ No missing nodes
# ✓ No duplicate node IDs
# ✓ No unreachable nodes
# ✓ No illegal cycles
# ✓ All permissions granted
# ✓ All models registered
# ✓ All models have required capabilities
# ✓ All loops have max_iterations
```

Compilation errors are raised immediately, before any tokens are spent.

### Workflow Nodes

**Sequence:**
```python
.start(agent_a)
.then(agent_b)
.then(agent_c)
```

**Parallel (fan-out/fan-in):**
```python
.parallel(
    agent_a,
    agent_b,
    agent_c,
    max_concurrency=2,
    merge=lambda outputs: combine(outputs)
)
```

**Router (conditional):**
```python
.route(
    condition=lambda ctx: classify(ctx.state["result"]),
    routes={
        "urgent": urgent_path,
        "normal": normal_path,
    },
    default=fallback_path
)
```

**Loop:**
```python
.loop(
    reviewer,
    until=lambda ctx: ctx.state.get("approved", False),
    max_iterations=3
)
```

**Human Approval:**
```python
.approve(
    message="Deploy to production?",
    action=deploy_agent,
    timeout_hours=24
)
```

### Workflow State

Nodes communicate through workflow state:

```python
# Write to state
context.state["analysis"] = result

# Read from state
previous_result = context.state.get("analysis")
```

State is:
- **Checkpointed** after every node
- **Validated** - must be JSON-serializable
- **Isolated** - nested workflows get derived state

## Policies and Budgets

### Execution Policy

```python
policy = ExecutionPolicy(
    max_steps=20,              # Total operations
    max_model_calls=15,        # LLM calls
    max_tool_calls=25,         # Tool invocations
    max_tokens=100_000,        # Combined input+output
    max_cost=3.00,             # USD
    timeout_seconds=300,
    max_concurrency=5,
    mode=ExecutionMode.STRICT,
    checkpoint_enabled=True,
)
```

**Budget Events:**
- At 80% utilization → `budget.warning` event
- At 100% utilization → `budget.exceeded` event, execution stops

### Execution Modes

| Mode | Temperature | Tool Calls | Iteration Budget | Use Case |
|------|------------|------------|------------------|----------|
| `STRICT` | 0 (forced) | Serialized | Tightest (≤5) | Reproducible results |
| `STANDARD` | Agent's own | Parallel | Agent's cap | Normal operation |
| `EXPLORATORY` | Agent's own | Parallel | Up to max_steps | Experimentation |

### Retry Policies

```python
from coflowai.policies import RetryPolicy

policy = RetryPolicy(
    max_retries=3,
    backoff_seconds=1.0,
    backoff_multiplier=2.0,
    jitter=True,
    retry_on=[ModelRateLimitError, ModelTemporaryError],
    never_retry_on=[ModelInvalidRequestError],
)
```

**Error Classification:**
- **Transient** - Safe to retry (rate limits, timeouts, connection errors)
- **Permanent** - Never retry (invalid request, authentication, model not found)
- **Ambiguous** - Retry with caution (depends on idempotency)

## Model Providers

### Provider Interface

```python
class ModelProvider(ABC):
    @abstractmethod
    async def call(self, request: ModelRequest) -> ModelResponse:
        """Synchronous model call."""
    
    @abstractmethod
    async def stream(self, request: ModelRequest) -> AsyncIterator[StreamChunk]:
        """Streaming model call."""
```

### Model Registry

```python
from coflowai.models import ModelRegistry, ModelCapabilities, ModelPricing

registry = ModelRegistry()

registry.register(
    name="gpt-4o",
    provider=openai_provider,
    capabilities=ModelCapabilities(
        tool_calling=True,
        structured_output=True,
        streaming=True,
        reasoning=False,
        vision=True,
    ),
    pricing=ModelPricing(
        input_per_million=2.50,
        output_per_million=10.00,
        cache_write_per_million=1.25,
        cache_read_per_million=0.25,
    ),
    fallbacks=["gpt-4o-mini"],
    metadata={"max_tokens": 128_000},
)
```

### Capability-Based Routing

```python
# Agent requires specific capabilities
agent = Agent(
    "analyst",
    model_requirements={
        "structured_output": True,
        "tool_calling": True,
    }
)

# Registry routes to compatible model
# If primary model lacks capability, fallback is used
```

### Provider Adapters

Built-in adapters handle provider differences:

**OpenAI:**
- Native tool calling
- Native structured output (JSON schema mode)
- SSE streaming
- Azure variant with different authentication

**Anthropic:**
- System prompt hoisting
- Content blocks translation
- Structured output emulated via tool calls
- Automatic unwrapping

**Gemini:**
- Contents/parts translation
- Native `responseSchema`
- JSON Schema keyword scrubbing

**Bedrock:**
- Converse API
- Thread-dispatched boto3 client
- Provider-specific error mapping

## Events

### Event Types

```python
# Execution lifecycle
"execution.started"
"execution.completed"
"execution.failed"
"execution.cancelled"
"execution.checkpointed"
"execution.resumed"

# Agent events
"agent.iteration_started"
"agent.iteration_completed"
"agent.output_invalid"
"agent.output_repaired"

# Model events
"model.call_started"
"model.call_completed"
"model.call_failed"
"model.call_retrying"
"model.streaming_started"
"model.streaming_completed"

# Tool events
"tool.call_started"
"tool.call_completed"
"tool.call_failed"
"tool.permission_denied"
"tool.validation_failed"

# Budget events
"budget.warning"        # 80% utilization
"budget.exceeded"       # 100% utilization

# Approval events
"approval.requested"
"approval.granted"
"approval.denied"
"approval.timeout"
```

### Event Store

```python
# Get events for execution
events = app.events.get_events(execution_id)

for event in events:
    print(f"{event.timestamp}: {event.type}")
    print(f"  Data: {event.data}")
```

Events enable:
- Audit trails
- Debugging
- Metrics
- Replay
- Real-time monitoring

## Memory

### Memory Store Interface

```python
class MemoryStore(ABC):
    @abstractmethod
    async def store(self, items: list[MemoryItem]) -> None:
        """Store memory items."""
    
    @abstractmethod
    async def retrieve(
        self,
        query: str,
        limit: int = 10,
        filters: dict[str, Any] | None = None
    ) -> list[MemoryItem]:
        """Retrieve relevant memories."""
```

### In-Memory Implementation

```python
from coflowai.memory import InMemoryStore

memory = InMemoryStore()

# Store conversation
await memory.store([
    MemoryItem(
        content="User prefers concise responses",
        metadata={"type": "preference", "user_id": "123"}
    )
])

# Retrieve relevant
items = await memory.retrieve("communication style", limit=5)
```

Built-in implementation uses BM25-like lexical matching. No vector database required for local development.

### Memory in Context

Memory items are automatically included in agent context with relevance scoring:

```
System Instructions
Relevant Memories          ← BM25-scored by query
  - [0.92] User prefers...
  - [0.81] Previous decision...
Conversation History
```

## Observability

### Structured Logging

```python
from coflowai.observability import configure_logging

configure_logging(
    level="INFO",
    format="json",
    redact_secrets=True,
    redact_patterns=[r"sk-[a-zA-Z0-9]+", r"Bearer [a-zA-Z0-9]+"]
)
```

### Execution Tracing

```python
trace = app.trace(execution_id)

print(trace.render())
# Output:
# ├─ workflow:pipeline [200ms]
# │  ├─ agent:planner [50ms] - 1 model call, 2 tool calls
# │  ├─ parallel:researchers [150ms]
# │  │  ├─ agent:web [100ms]
# │  │  ├─ agent:database [80ms]
# │  │  └─ agent:documents [120ms]
# │  └─ agent:writer [50ms]
```

### OpenTelemetry Integration

```python
from opentelemetry import trace
from coflowai.observability import OpenTelemetryTracer

tracer = OpenTelemetryTracer(trace.get_tracer("coflowai"))
app = CoFlowAi(tracer=tracer)
```

Traces mirror to OpenTelemetry for integration with:
- Jaeger
- Zipkin
- Honeycomb
- Datadog
- New Relic

### Metrics

```python
usage = result.usage

print(f"Execution time: {usage.execution_time_seconds}s")
print(f"Model calls: {usage.model_calls}")
print(f"Tool calls: {usage.tool_calls}")
print(f"Input tokens: {usage.input_tokens}")
print(f"Output tokens: {usage.output_tokens}")
print(f"Total cost: ${usage.total_cost:.4f}")

# Per-node breakdown
for node_id, node_usage in usage.by_node.items():
    print(f"{node_id}: ${node_usage.cost:.4f}")
```

## Security

### Permission Model

```python
# Namespaced permissions
permissions = PermissionSet([
    "database.read",
    "database.write",
    "api.external.*",      # Wildcard
])

# Tool requires permission
@tool(permission="database.write")
async def update_record(): ...

# Permission denied → event emitted, execution continues or fails based on policy
```

### Secret Redaction

```python
# Automatic pattern-based redaction
configure_logging(
    redact_secrets=True,
    redact_patterns=[
        r"sk-[a-zA-Z0-9]+",           # API keys
        r"Bearer [a-zA-Z0-9]+",        # Bearer tokens
        r"\b[A-Z0-9]{20,}\b",          # Long alphanumeric (AWS keys)
    ]
)
```

Secrets are redacted from:
- Logs
- Events
- Traces
- Error messages

### Human Approval Gates

```python
workflow.approve(
    message="Deploy to production?",
    action=deploy_agent,
    metadata={"environment": "prod", "version": "1.2.0"},
    timeout_hours=24
)

# Execution pauses, checkpoint created
# Worker released (no blocking)
# Later, via API or CLI:
await app.approve(execution_id, by="aravindan")
```

## Next Steps

- **[API Reference](./API_REFERENCE.md)** - Detailed API documentation
- **[User Guide](./USER_GUIDE.md)** - Common patterns and recipes
- **[Advanced Topics](./ADVANCED.md)** - Distributed execution, optimization
