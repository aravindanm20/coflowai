# API Reference

Complete API documentation for CoFlowAi.

## Table of Contents

- [Core](#core)
- [Agents](#agents)
- [Workflows](#workflows)
- [Tools](#tools)
- [Policies](#policies)
- [Models](#models)
- [Events](#events)
- [Persistence](#persistence)
- [Distributed](#distributed)
- [Testing](#testing)

---

## Core

### CoFlowAi

The main application object.

```python
from coflowai import CoFlowAi

app = CoFlowAi(
    event_store: EventStore | None = None,
    state_store: StateStore | None = None,
    memory_store: MemoryStore | None = None,
    sandbox: Sandbox | None = None,
    tracer: ExecutionTracer | None = None,
    plugins: list[Plugin] | None = None,
)
```

**Parameters:**
- `event_store` - Event persistence (default: `InMemoryEventStore`)
- `state_store` - State persistence (default: `InMemoryStateStore`)
- `memory_store` - Memory storage (default: `InMemoryStore`)
- `sandbox` - Tool execution sandbox (default: `InProcessSandbox`)
- `tracer` - Execution tracing (default: `DefaultTracer`)
- `plugins` - Additional plugins to load

**Methods:**

#### run()

Execute an agent or workflow.

```python
async def run(
    executable: Executable,
    request: str | dict[str, Any],
    policy: ExecutionPolicy | None = None,
    execution_id: str | None = None,
) -> ExecutionResult
```

**Parameters:**
- `executable` - Agent or Workflow to execute
- `request` - User request (string or dict)
- `policy` - Execution policy (budgets, timeouts)
- `execution_id` - Optional execution ID (for resume)

**Returns:** `ExecutionResult`

**Raises:**
- `ExecutionError` - On execution failure
- `BudgetExceededError` - When budget limits hit
- `TimeoutError` - On timeout

#### stream()

Execute with streaming output.

```python
async def stream(
    executable: Executable,
    request: str | dict[str, Any],
    policy: ExecutionPolicy | None = None,
) -> AsyncIterator[StreamChunk]
```

**Yields:** `StreamChunk` instances

**Example:**
```python
async for chunk in app.stream(agent, "Explain"):
    if chunk.is_final:
        result = chunk.metadata["result"]
    else:
        print(chunk.text, end="", flush=True)
```

#### resume()

Resume a paused or failed execution.

```python
async def resume(
    execution_id: str,
    executable: Executable,
    policy: ExecutionPolicy | None = None,
) -> ExecutionResult
```

**Parameters:**
- `execution_id` - Execution to resume
- `executable` - Workflow definition (must match hash)
- `policy` - Optional policy override

#### replay()

Replay an execution step-by-step.

```python
async def replay(execution_id: str) -> ExecutionTrace
```

**Returns:** Full execution trace

#### fork()

Fork an execution from a specific step.

```python
async def fork(
    execution_id: str,
    executable: Executable,
    from_step: int = 0,
    overrides: dict[str, Any] | None = None,
) -> ExecutionResult
```

**Parameters:**
- `from_step` - Step to fork from (0 = beginning)
- `overrides` - Model, state, or instruction overrides

#### approve()

Approve a waiting execution.

```python
async def approve(
    execution_id: str,
    by: str,
    executable: Executable,
    metadata: dict[str, Any] | None = None,
) -> ExecutionResult
```

#### trace()

Get execution trace.

```python
def trace(execution_id: str) -> ExecutionTrace
```

---

## Agents

### Agent

AI reasoning agent.

```python
from coflowai import Agent

agent = Agent(
    name: str,
    instructions: str,
    model: str,
    tools: list[Callable] | None = None,
    output_schema: type[BaseModel] | None = None,
    max_iterations: int = 10,
    temperature: float | None = None,
    timeout_seconds: int | None = None,
    model_requirements: dict[str, bool] | None = None,
    metadata: dict[str, Any] | None = None,
)
```

**Parameters:**
- `name` - Agent identifier (used in traces)
- `instructions` - System instructions / role definition
- `model` - Model name from registry
- `tools` - Available tools (functions decorated with `@tool`)
- `output_schema` - Pydantic model for structured output
- `max_iterations` - Maximum reasoning iterations
- `temperature` - Sampling temperature (None = model default)
- `timeout_seconds` - Agent-level timeout
- `model_requirements` - Required capabilities dict
- `metadata` - Additional metadata

**Example:**
```python
agent = Agent(
    "researcher",
    "You are a research assistant.",
    "gpt-4o",
    tools=[search, analyze],
    output_schema=ResearchReport,
    max_iterations=10,
    model_requirements={"tool_calling": True}
)
```

### execute()

Execute the agent (implements `Executable`).

```python
async def execute(self, context: ExecutionContext) -> ExecutionResult
```

---

## Workflows

### Workflow

Orchestration workflow.

```python
from coflowai import Workflow

workflow = Workflow(
    name: str,
    version: str | None = None,
    description: str | None = None,
)
```

**Parameters:**
- `name` - Workflow identifier
- `version` - Version string (used for hash compatibility)
- `description` - Human-readable description

### Builder Methods

#### start()

Set the entry node.

```python
def start(self, executable: Executable) -> Workflow
```

#### then()

Add sequential node.

```python
def then(self, executable: Executable) -> Workflow
```

#### parallel()

Add parallel execution node.

```python
def parallel(
    self,
    *executables: Executable,
    max_concurrency: int | None = None,
    merge: Callable[[dict[str, Any]], Any] | None = None,
) -> Workflow
```

**Parameters:**
- `executables` - Executables to run in parallel
- `max_concurrency` - Maximum concurrent executions
- `merge` - Function to merge outputs: `dict[node_id, output] -> merged`

**Example:**
```python
.parallel(
    agent_a,
    agent_b,
    agent_c,
    max_concurrency=2,
    merge=lambda outputs: "\n".join(outputs.values())
)
```

#### route()

Add conditional routing.

```python
def route(
    self,
    condition: Callable[[ExecutionContext], str],
    routes: dict[str, Executable],
    default: Executable | None = None,
) -> Workflow
```

**Parameters:**
- `condition` - Function returning route key
- `routes` - Map of keys to executables
- `default` - Fallback executable

**Example:**
```python
.route(
    condition=lambda ctx: classify(ctx.state["data"]),
    routes={
        "urgent": urgent_path,
        "normal": normal_path,
    },
    default=fallback_path
)
```

#### loop()

Add iteration loop.

```python
def loop(
    self,
    executable: Executable,
    until: Callable[[ExecutionContext], bool],
    max_iterations: int,
) -> Workflow
```

**Parameters:**
- `executable` - Executable to loop
- `until` - Exit condition
- `max_iterations` - Maximum iterations (required)

**Example:**
```python
.loop(
    reviewer,
    until=lambda ctx: ctx.state.get("approved", False),
    max_iterations=3
)
```

#### approve()

Add human approval gate.

```python
def approve(
    self,
    message: str,
    action: Executable | None = None,
    timeout_hours: int | None = None,
    metadata: dict[str, Any] | None = None,
) -> Workflow
```

**Parameters:**
- `message` - Approval request message
- `action` - Executable to run after approval (optional)
- `timeout_hours` - Approval timeout
- `metadata` - Additional context

#### transform()

Add data transformation node.

```python
def transform(
    self,
    func: Callable[[ExecutionContext], Any],
    output_key: str | None = None,
) -> Workflow
```

**Parameters:**
- `func` - Transformation function
- `output_key` - State key to store result

### compile()

Compile and validate workflow.

```python
def compile(self, context: ExecutionContext | None = None) -> CompiledWorkflow
```

Validates:
- No missing nodes
- No unreachable nodes
- No illegal cycles
- All permissions granted
- All models registered
- All loops have `max_iterations`

**Raises:** `WorkflowCompilationError` on validation failure

---

## Tools

### @tool

Decorator to register a tool.

```python
from coflowai import tool

@tool(
    name: str | None = None,
    description: str | None = None,
    permission: str | list[str] | None = None,
    timeout: int | None = None,
    retries: int = 0,
    side_effect: bool = False,
    idempotent: bool = False,
    metadata: dict[str, Any] | None = None,
)
def my_tool(...) -> ...:
    ...
```

**Parameters:**
- `name` - Tool name (default: function name)
- `description` - Tool description (shown to LLM)
- `permission` - Required permission(s)
- `timeout` - Tool-specific timeout in seconds
- `retries` - Retry attempts on failure
- `side_effect` - Whether tool modifies state
- `idempotent` - Whether safe to retry (requires side_effect=True)
- `metadata` - Additional metadata

**Example:**
```python
@tool(
    description="Update customer email",
    permission="database.write",
    timeout=5,
    retries=2,
    side_effect=True,
    idempotent=True,
)
async def update_customer(customer_id: str, email: str) -> dict:
    # Execution gets idempotency key:
    # execution_id + node_id + tool_call_id
    return {"status": "updated"}
```

### ToolRegistry

Tool registration and lookup.

```python
from coflowai.tools import ToolRegistry

registry = ToolRegistry()
registry.register(my_tool)

tool_def = registry.get("my_tool")
tools = registry.list(permissions=["database.*"])
```

---

## Policies

### ExecutionPolicy

Execution budgets and limits.

```python
from coflowai.policies import ExecutionPolicy, ExecutionMode

policy = ExecutionPolicy(
    max_steps: int | None = None,
    timeout_seconds: int | None = None,
    max_model_calls: int | None = None,
    max_tool_calls: int | None = None,
    max_tokens: int | None = None,
    max_cost: float | None = None,
    max_concurrency: int | None = None,
    mode: ExecutionMode = ExecutionMode.STANDARD,
    checkpoint_enabled: bool = True,
    retry_policy: RetryPolicy | None = None,
)
```

**Parameters:**
- `max_steps` - Maximum total operations
- `timeout_seconds` - Overall timeout
- `max_model_calls` - Maximum LLM calls
- `max_tool_calls` - Maximum tool invocations
- `max_tokens` - Total token budget (input + output)
- `max_cost` - Maximum cost in USD
- `max_concurrency` - Parallel execution limit
- `mode` - Execution mode (STRICT/STANDARD/EXPLORATORY)
- `checkpoint_enabled` - Enable checkpointing
- `retry_policy` - Retry behavior

### ExecutionMode

```python
class ExecutionMode(Enum):
    STRICT = "strict"           # Temperature=0, serialized tools, tight iterations
    STANDARD = "standard"       # Agent settings, parallel tools, normal iterations
    EXPLORATORY = "exploratory" # Agent settings, parallel tools, room to backtrack
```

### RetryPolicy

```python
from coflowai.policies import RetryPolicy

policy = RetryPolicy(
    max_retries: int = 3,
    backoff_seconds: float = 1.0,
    backoff_multiplier: float = 2.0,
    max_backoff_seconds: float = 60.0,
    jitter: bool = True,
    retry_on: list[type[Exception]] | None = None,
    never_retry_on: list[type[Exception]] | None = None,
)
```

### PermissionSet

```python
from coflowai.policies import PermissionSet

permissions = PermissionSet([
    "database.read",
    "database.write",
    "api.external.*",
])

permissions.check("database.read")   # True
permissions.check("api.external.foo") # True
permissions.check("files.write")     # False
```

**Wildcard Support:**
- `database.*` - Matches `database.read`, `database.write`, etc.
- `*` - Matches everything

---

## Models

### ModelRegistry

Model registration and management.

```python
from coflowai.models import ModelRegistry

registry = ModelRegistry()

registry.register(
    name: str,
    provider: ModelProvider,
    capabilities: ModelCapabilities,
    pricing: ModelPricing,
    fallbacks: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
)
```

**Example:**
```python
from coflowai.providers.openai import OpenAIProvider

provider = OpenAIProvider(api_key="sk-...")

registry.register(
    "gpt-4o",
    provider=provider,
    capabilities=ModelCapabilities(
        tool_calling=True,
        structured_output=True,
        streaming=True,
    ),
    pricing=ModelPricing(
        input_per_million=2.50,
        output_per_million=10.00,
    ),
    fallbacks=["gpt-4o-mini"],
)
```

### ModelCapabilities

```python
from coflowai.models import ModelCapabilities

capabilities = ModelCapabilities(
    tool_calling: bool = False,
    structured_output: bool = False,
    streaming: bool = False,
    reasoning: bool = False,
    vision: bool = False,
    audio: bool = False,
)
```

### ModelPricing

```python
from coflowai.models import ModelPricing

pricing = ModelPricing(
    input_per_million: float,
    output_per_million: float,
    cache_write_per_million: float | None = None,
    cache_read_per_million: float | None = None,
)
```

---

## Providers

### OpenAIProvider

```python
from coflowai.providers.openai import OpenAIProvider

provider = OpenAIProvider(
    api_key: str,
    organization: str | None = None,
    base_url: str | None = None,  # For compatible endpoints
    timeout: int = 60,
)
```

### AzureOpenAIProvider

```python
from coflowai.providers.openai import AzureOpenAIProvider

provider = AzureOpenAIProvider(
    api_key: str,
    endpoint: str,
    api_version: str = "2024-02-15-preview",
    timeout: int = 60,
)
```

### AnthropicProvider

```python
from coflowai.providers.anthropic import AnthropicProvider

provider = AnthropicProvider(
    api_key: str,
    base_url: str | None = None,
    timeout: int = 60,
)
```

### GeminiProvider

```python
from coflowai.providers.gemini import GeminiProvider

provider = GeminiProvider(
    api_key: str,
    timeout: int = 60,
)
```

### BedrockProvider

```python
from coflowai.providers.bedrock import BedrockProvider

provider = BedrockProvider(
    region_name: str = "us-east-1",
    profile_name: str | None = None,
    aws_access_key_id: str | None = None,
    aws_secret_access_key: str | None = None,
)
```

---

## Events

### EventStore

```python
from coflowai.events import EventStore

class EventStore(ABC):
    async def append(self, event: Event) -> None: ...
    async def get_events(
        self,
        execution_id: str,
        from_sequence: int = 0,
    ) -> list[Event]: ...
```

### Event

```python
@dataclass
class Event:
    id: str
    execution_id: str
    type: str
    timestamp: datetime
    sequence: int
    data: dict[str, Any]
    metadata: dict[str, Any]
```

### Event Types

```python
# Execution
"execution.started"
"execution.completed"
"execution.failed"
"execution.checkpointed"

# Agent
"agent.iteration_started"
"agent.iteration_completed"

# Model
"model.call_started"
"model.call_completed"

# Tool
"tool.call_started"
"tool.call_completed"
"tool.permission_denied"

# Budget
"budget.warning"
"budget.exceeded"
```

---

## Persistence

### SqliteEventStore

```python
from coflowai.persistence.sqlite import SqliteEventStore

store = SqliteEventStore(
    database_path: str,
    wal_mode: bool = True,
)
```

### SqliteStateStore

```python
from coflowai.persistence.sqlite import SqliteStateStore

store = SqliteStateStore(
    database_path: str,
    wal_mode: bool = True,
    lock_lease_seconds: int = 300,
)
```

### PostgresEventStore

```python
from coflowai.persistence.postgres import PostgresEventStore

store = PostgresEventStore(
    dsn: str,  # "postgresql://user:pass@host/db"
)
```

### PostgresStateStore

```python
from coflowai.persistence.postgres import PostgresStateStore

store = PostgresStateStore(
    dsn: str,
    lock_lease_seconds: int = 300,
)
```

### RedisEventStore

```python
from coflowai.persistence.redis import RedisEventStore
import redis.asyncio as redis

client = redis.Redis(host="localhost", port=6379)
store = RedisEventStore(client=client)
```

### RedisStateStore

```python
from coflowai.persistence.redis import RedisStateStore

store = RedisStateStore(
    client=client,
    lock_lease_seconds: int = 300,
)
```

---

## Distributed

### ExecutionService

```python
from coflowai.distributed import ExecutionService, RedisJobQueue

queue = RedisJobQueue(redis_client)
service = ExecutionService(queue=queue, runtime=app.runtime)

# Register workflows
service.register(workflow)

# Submit for execution
execution_id = await service.submit(workflow, request="task")

# Worker processes this elsewhere
```

### Worker

```python
from coflowai.distributed import Worker

worker = Worker(
    service=service,
    concurrency=4,
    poll_interval_seconds=1.0,
)

await worker.run()
```

### CLI Worker

```bash
coflowai worker app.py --concurrency 4
```

---

## Testing

### build_test_app()

Create a deterministic test application.

```python
from coflowai.testing import build_test_app

app, model = build_test_app(
    responses: list[str | dict | ToolCall],
    tools: list[Callable] | None = None,
    permissions: list[str] | None = None,
)
```

**Parameters:**
- `responses` - Sequence of model responses
- `tools` - Available tools
- `permissions` - Granted permissions

**Returns:** `(CoFlowAi, FakeModelProvider)`

**Example:**
```python
from coflowai.testing import tool_call

app, model = build_test_app(
    responses=[
        tool_call("get_weather", city="Chennai"),
        "The temperature is 30°C.",
    ],
    tools=[get_weather],
    permissions=["weather.read"]
)

result = await app.run(agent, "weather?")
assert result.succeeded
assert model.call_count == 2
```

### FakeModelProvider

```python
from coflowai.testing import FakeModelProvider

model = FakeModelProvider(responses=[...])

# Properties
model.call_count: int
model.calls: list[ModelRequest]
model.responses_given: list[ModelResponse]
```

### tool_call()

Create a tool call response.

```python
from coflowai.testing import tool_call

response = tool_call(
    tool_name: str,
    **kwargs: Any
) -> ToolCall
```

---

## Result Types

### ExecutionResult

```python
@dataclass
class ExecutionResult:
    execution_id: str
    status: ExecutionStatus
    output: Any
    error: Exception | None
    usage: UsageTracker
    metadata: dict[str, Any]
    
    @property
    def succeeded(self) -> bool: ...
    
    @property
    def failed(self) -> bool: ...
```

### ExecutionStatus

```python
class ExecutionStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    WAITING_FOR_APPROVAL = "waiting_for_approval"
```

### UsageTracker

```python
@dataclass
class UsageTracker:
    execution_time_seconds: float
    model_calls: int
    tool_calls: int
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cached_tokens: int
    total_cost: float
    
    by_model: dict[str, ModelUsage]
    by_node: dict[str, NodeUsage]
    by_tool: dict[str, ToolUsage]
```

---

## Exception Hierarchy

```
CoFlowAiError (base)
├── ExecutionError
│   ├── ExecutionTimeoutError
│   ├── BudgetExceededError
│   ├── CheckpointNotFoundError
│   └── WorkflowCompilationError
├── ModelError
│   ├── ModelNotFoundError
│   ├── ModelInvalidRequestError
│   ├── ModelRateLimitError
│   ├── ModelTemporaryError
│   └── ModelAuthenticationError
├── ToolError
│   ├── ToolNotFoundError
│   ├── ToolPermissionDeniedError
│   ├── ToolValidationError
│   └── ToolTimeoutError
└── StateError
    ├── StateConflictError
    └── StateLockError
```

---

## Next Steps

- **[User Guide](./USER_GUIDE.md)** - Common patterns and recipes
- **[Advanced Topics](./ADVANCED.md)** - Distributed execution, optimization
- **[Examples](../examples/)** - Working code examples
