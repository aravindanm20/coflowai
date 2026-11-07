# Changelog

All notable changes to CoFlowAi are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/); versioning is semantic.

## [1.0.1] — Correctness and operability fixes

### Fixed
- **Approvals and exactly-once side effects**: state written inside a workflow
  node (approval ledger, idempotency ledger, agent conversations, nested
  progress) was lost because child scopes are deep copies; it is now merged back
  (`ExecutionContext.merge_durable_state`), including when a node fails.
- **Parallel nodes**: a failing branch raised an opaque `ExceptionGroup`; the
  real error (e.g. `ApprovalRequired`, `BudgetExceeded`) is now surfaced.
  With `fail_fast=False`, control-flow signals (approval, cancellation,
  timeout, budget/step limits) are no longer swallowed as "branch failures" —
  previously a pending approval could be silently skipped.
- **Parallel budgets**: concurrent branches each checked a private snapshot and
  could jointly overspend `max_model_calls`/`max_tool_calls`/`max_tokens`/
  `max_cost`. Enforcement now uses one execution-wide `UsageLedger` with atomic
  call-slot reservations.
- Custom executables returning a plain value no longer crash the runtime.

### Added
- Per-branch progress for `ParallelNode`: completed branches are not re-run on
  resume (e.g. after a sibling paused for approval).
- `Runtime(cache_size=, cache_ttl_seconds=, lock_ttl_seconds=)`; per-execution
  runtime caches are now bounded (LRU/TTL) instead of growing forever.
- `RedisJobQueue.extend()` and a worker lease heartbeat, so jobs longer than the
  visibility timeout are not redelivered while still running.
- `CoFlowAi.event_store` / `CoFlowAi.state_store` properties; explicit
  signatures on `CoFlowAi.run/stream/resume/approve/reject/cancel/fork`.
- Exceptions re-exported from `coflowai.core`; `ExecutionMode` from
  `coflowai.policies`; `StreamChunk`/`ChunkType` from `coflowai.models`.
- `scripts/check_docs.py` (validates every docs snippet against the real API)
  and `scripts/gen_api_reference.py` (generates `docs/API_REFERENCE.md`); both
  run in CI (`.github/workflows/ci.yml`).

### Changed
- `JsonlEventStore.append` is O(1) amortised (was a full-file re-read per
  event: ~100x faster at 1,000 events).
- `RedisJobQueue` keys in-flight jobs by `job_id` (O(log n) ack instead of a
  full scan); `recover()` is safe to run concurrently and dead-letters jobs
  that exhausted `max_attempts`. Legacy in-flight entries are still recovered.
- Documentation rewritten where it described APIs that don't exist (121
  mismatches fixed); `app.run()` failure handling is documented correctly
  (failures are returned on the result, not raised).

## [1.0.0] — Boundaries filled

Closes every interface that 0.9.0 left scaffolded. The framework now runs against
real models, persists durably, and scales across workers.

### Added — providers (V0.8 delivered)
- `OpenAIProvider` — Chat Completions, strict JSON-schema structured output, SSE
  streaming; also covers Azure (`AzureOpenAIProvider`) and any OpenAI-compatible
  endpoint via `base_url`
- `AnthropicProvider` — Messages API, system-prompt hoisting, content blocks,
  structured output emulated as a forced tool call and unwrapped transparently
- `GeminiProvider` — `contents`/`parts` translation, native `responseSchema`,
  JSON Schema keyword scrubbing
- `BedrockProvider` — Converse API over a thread-dispatched boto3 client
- Shared `HttpTransport` with status→error mapping; `MockTransport` makes every
  adapter testable with no network or API keys

### Added — streaming
- `StreamChunk` / `StreamAccumulator`; `ModelProvider.stream()` with automatic
  emulation for non-streaming providers
- `Runtime.stream()` and `Agent.execute_stream()`; budgets, events and usage are
  accounted exactly once at stream completion
- Structured output is never streamed partially

### Added — persistence (V0.9 delivered)
- `SqliteEventStore` / `SqliteStateStore` — durable with **zero** extra
  dependencies (WAL mode, thread-dispatched, lock leases)
- `PostgresEventStore` / `PostgresStateStore` — atomic steal-on-expiry lock leases
- `RedisEventStore` / `RedisStateStore` — fenced locks released by a Lua script
  that cannot delete a successor's lock

### Added — distributed execution
- `JobQueue` abstraction with `InMemoryJobQueue` and `RedisJobQueue`
  (visibility timeouts, dead-letter, crash recovery)
- `Worker`, `WorkerPool`, `ExecutionService` and `ExecutableRegistry`
- At-least-once delivery proven safe by existing checkpoint + idempotency
  guarantees; `coflowai worker` CLI command

### Added — security
- `SubprocessSandbox` — real OS isolation with CPU/memory/FD limits, scrubbed
  environment (secrets are not inherited) and hard kill on timeout

### Added — context
- `ExtractiveSummariser` is now the **default**: deterministic, offline, and
  keeps decisions/failures over pleasantries instead of truncating
- `ModelSummariser` for higher fidelity, degrading to extractive on failure

### Changed
- `ExecutionMode.EXPLORATORY` and `STRICT` now genuinely alter behaviour
  (iteration allowance and tool-call serialisation), not just temperature
- `fork()` accepts `model_responses`, `tool_results` and `instructions`
  overrides; `from_step=0` means "re-run from the beginning"

### Fixed
- Streaming discarded provider-reported token usage, under-counting spend on
  every streamed call
- `Runtime._transition` now permits operator-driven resume of failed executions
- Nested workflows could collide on shared state keys

## [0.9.0] — Kernel complete

The framework kernel through the V0.9 milestone. The public API is stable in shape
but not yet frozen; breaking changes remain possible until 1.0.

### V0.1 — Execution kernel
- `Executable` abstraction; `ExecutionContext` with no global mutable state
- `Runtime` with an explicitly enforced execution state machine
- `Agent` reasoning loop with mandatory iteration bounds
- `ModelProvider` interface, `ModelRegistry`, `FakeModelProvider`
- `@tool`, `ToolRegistry`, permission-checked `ToolExecutor`
- `ExecutionPolicy`, `InMemoryEventStore`, `InMemoryStateStore`
- Usage tracking; framework-level error model

### V0.2 — Orchestration
- `Workflow` fluent builder; sequence, parallel, router, loop and transform nodes
- Workflow compiler: duplicate/missing nodes, invalid edges, unreachable nodes,
  illegal cycles, permission and model-capability validation

### V0.3 — Reliability
- Pydantic structured output with repair-and-retry; never returns invalid output
- Retry policies distinguishing transient from permanent errors
- Execution, agent, model, tool and node timeouts that propagate down the hierarchy
- Token, cost, model-call and tool-call budgets with warning and exceeded events

### V0.4 — Durability
- Event sourcing over the full standard event catalogue
- Checkpoints before/after every node, before approvals and on failure
- `resume()` with workflow-hash compatibility checking; completed nodes are skipped
- `replay()` reconstructing the step-by-step execution story
- `fork()` with model/prompt/state overrides

### V0.5 — Security
- Namespaced permissions with wildcard support and least-privilege intersection
- Human approval gates that pause durably and release the worker
- Idempotency keys derived from `execution_id + node_id + tool_call_id`
- Sandbox abstraction (`InProcessSandbox`, `ThreadSandbox`)
- Tool exposure filtered by permission as prompt-injection defence

### V0.6 — Context and memory
- Context engine with labelled sections and per-section token budgets
- Graceful degradation: drop low relevance → summarise → compress tool output
- `MemoryStore` interface with a dependency-free lexical in-memory implementation

### V0.7 — Observability
- Structured JSON logging with configurable secret redaction
- Nested execution tracer with optional OpenTelemetry mirroring
- Metrics sink and per-execution/agent/model/tool/node usage tracking

### V0.8 — Providers
- Provider adapter template documenting the full contract
- Capability-based model routing, cost/latency/priority preference
- Automatic failover to fallback models on transient provider faults

### V0.9 — Operations
- `CoFlowAi` application object and plugin architecture with entry-point discovery
- Distributed-ready state store interface including lock leasing
- `coflowai` CLI: run, executions list/inspect/replay/resume/approve, workflows
  validate, models list, tools list
- `JsonlEventStore` for cross-process local durability

## [Unreleased] — toward 1.0
- Provider adapter distributions (OpenAI, Anthropic, Azure, Bedrock, Gemini)
- Postgres/Redis persistence, distributed locks, queue-backed workers
- Execution migration, streaming, benchmarks, load/recovery/chaos testing
