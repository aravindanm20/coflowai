# CoFlowAi Documentation

Complete documentation for CoFlowAi - A production-grade agentic AI framework for Python.

## Quick Links

- **[Getting Started](./GETTING_STARTED.md)** - Installation, quickstart, and basic concepts
- **[Core Concepts](./CORE_CONCEPTS.md)** - Architecture and design principles
- **[API Reference](./API_REFERENCE.md)** - Complete API documentation
- **[User Guide](./USER_GUIDE.md)** - Common patterns and recipes
- **[Advanced Topics](./ADVANCED.md)** - Distributed execution, security, optimization
- **[Examples](./EXAMPLES.md)** - Working code examples
- **[Troubleshooting](./TROUBLESHOOTING.md)** - Common issues and solutions
- **[Contributing](./CONTRIBUTING.md)** - Contribute to CoFlowAi

---

## What is CoFlowAi?

CoFlowAi is a production-grade agentic AI framework for Python 3.12+ that enforces a strict separation between:

- **What the AI decides** (nondeterministic reasoning)
- **What the platform controls** (deterministic execution)

The LLM decides which tool to call and what to say. The runtime decides whether that is permitted, affordable, in time, and recoverable.

### Key Features

✅ **Deterministic Runtime** - Explicit state machine, no global mutable state  
✅ **Provider Independent** - Works with OpenAI, Anthropic, Gemini, Bedrock, and custom providers  
✅ **Event Sourced** - Every action emits an event; executions are replayable  
✅ **Resumable** - Checkpoints before/after every node; resume never reruns completed work  
✅ **Secure by Default** - Namespaced permissions, validated tool arguments, secret redaction  
✅ **Bounded** - Step, token, cost, and call limits are mandatory  
✅ **Observable** - Structured logs, nested traces, metrics built-in  
✅ **Testable Offline** - FakeModelProvider makes tests deterministic with no API keys  
✅ **Streaming** - Token-by-token output with full budget and policy enforcement  
✅ **Durable** - SQLite, Postgres, or Redis persistence  
✅ **Distributed** - Queue + worker pool for scalable execution  

---

## Documentation Structure

### 📚 For New Users

Start here if you're new to CoFlowAi:

1. **[Getting Started](./GETTING_STARTED.md)**
   - Installation
   - Quick start example
   - Core concepts overview
   - Testing without LLMs
   - Working with real providers

2. **[Core Concepts](./CORE_CONCEPTS.md)**
   - Architecture overview
   - Execution flow
   - Agents, tools, and workflows
   - Policies and budgets
   - Model providers
   - Events and observability

### 🛠️ For Building Applications

Use these guides when building with CoFlowAi:

3. **[API Reference](./API_REFERENCE.md)**
   - Complete API documentation
   - All classes and methods
   - Parameters and return types
   - Exception hierarchy

4. **[User Guide](./USER_GUIDE.md)**
   - Agent patterns
   - Workflow patterns
   - Tool design
   - Error handling
   - Testing strategies
   - Performance optimization
   - Production deployment

5. **[Examples](./EXAMPLES.md)**
   - Basic examples
   - Agent examples
   - Workflow examples
   - Tool examples
   - Provider examples
   - Testing examples
   - Production examples

### 🚀 For Advanced Use Cases

Deep dives into advanced topics:

6. **[Advanced Topics](./ADVANCED.md)**
   - Distributed execution
   - Security (permissions, sandboxing, secrets)
   - Advanced workflows
   - Performance tuning
   - Monitoring and debugging
   - Custom providers
   - Plugin development

### 🔧 For Troubleshooting

7. **[Troubleshooting](./TROUBLESHOOTING.md)**
   - Installation issues
   - Execution errors
   - Provider issues
   - Performance issues
   - Testing issues
   - Distributed execution issues
   - Debugging tips

### 🤝 For Contributors

8. **[Contributing](./CONTRIBUTING.md)**
   - Code of conduct
   - Development setup
   - Project structure
   - Development workflow
   - Testing guidelines
   - Code style
   - Pull request process

---

## Quick Example

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

---

## Common Tasks

### Installation

```bash
# Core only
pip install coflowai

# With OpenAI
pip install "coflowai[openai]"

# Everything
pip install "coflowai[all]"
```

### Create an Agent

```python
from coflowai import Agent

agent = Agent(
    name="assistant",
    instructions="You are a helpful assistant.",
    model="gpt-4o",
    tools=[search, analyze],
    max_iterations=10
)
```

### Create a Workflow

```python
from coflowai import Workflow

workflow = (
    Workflow("pipeline")
    .start(agent_a)
    .then(agent_b)
    .parallel(agent_c, agent_d, agent_e)
    .route(condition=classify, routes={"urgent": agent_f, "normal": agent_g})
    .loop(agent_h, until=is_done, max_iterations=3)
)
```

### Execute

```python
from coflowai.policies import ExecutionPolicy

policy = ExecutionPolicy(
    max_cost=1.00,
    timeout_seconds=300,
    max_steps=20
)

result = await app.run(workflow, "Process this data", policy=policy)
```

### Test

```python
from coflowai.testing import build_test_app, tool_call

app, model = build_test_app(
    responses=[
        tool_call("search", query="AI"),
        "Here are the results..."
    ],
    tools=[search],
    permissions=["web.search"]
)

result = await app.run(agent, "Search for AI")
assert result.succeeded
```

---

## Architecture Diagram

```
                    Developer SDK
                 Agent / Workflow / Tool
                          │
                  Workflow Compiler          ← validates before execution
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

---

## Core Principles

CoFlowAi is built on these foundational principles:

1. **The AI can be nondeterministic. The execution platform cannot be uncontrolled.**
2. Never allow unlimited agent loops - every loop has `max_iterations`
3. Never trust model output - validate tool arguments and structured responses
4. Never let provider details leak into agent logic
5. Never make memory or vector databases mandatory
6. Never make external databases mandatory for local execution
7. Never let tools bypass the permission runtime
8. Never silently retry side-effect operations
9. Never resume workflows without checking compatibility
10. Every production execution is traceable by `execution_id`

---

## Project Layout

```
coflowai/
├── coflowai/           # Main package
│   ├── core/           # Executable, ExecutionContext, ExecutionResult
│   ├── agents/         # Agent reasoning loop
│   ├── runtime/        # Execution runtime
│   ├── workflows/      # Workflow builder, compiler, nodes
│   ├── models/         # Provider interface and registry
│   ├── tools/          # @tool decorator and executor
│   ├── events/         # Event system
│   ├── state/          # Checkpoints and state management
│   ├── policies/       # ExecutionPolicy, budgets, permissions
│   ├── context/        # Context builder
│   ├── memory/         # Memory stores
│   ├── observability/  # Logging, tracing, metrics
│   ├── persistence/    # SQLite, Postgres, Redis
│   ├── distributed/    # Job queue and workers
│   ├── providers/      # OpenAI, Anthropic, Gemini, Bedrock
│   └── testing/        # FakeModelProvider and test utilities
├── tests/              # Test suite
├── examples/           # Example scripts
├── docs/               # Documentation (you are here)
└── pyproject.toml      # Project configuration
```

---

## CLI Commands

```bash
# Run an agent/workflow
coflowai run app.py --input "Research AI" --trace

# List executions
coflowai executions list app.py

# Inspect execution
coflowai executions inspect app.py exec_123

# Replay execution
coflowai executions replay app.py exec_123

# Resume execution
coflowai executions resume app.py exec_123

# Approve waiting execution
coflowai executions approve app.py exec_123 --by alice

# Validate workflow
coflowai workflows validate workflow.py

# List models
coflowai models list app.py

# List tools
coflowai tools list app.py

# Start worker
coflowai worker app.py --concurrency 4
```

---

## Resources

### Documentation
- **[Getting Started](./GETTING_STARTED.md)** - Start here
- **[API Reference](./API_REFERENCE.md)** - Complete API
- **[Examples](./EXAMPLES.md)** - Working code

### Community
- **GitHub**: [github.com/aravindanm20/coflowai](https://github.com/aravindanm20/coflowai)
- **Issues**: [Report bugs](https://github.com/aravindanm20/coflowai/issues)
- **Discussions**: [Ask questions](https://github.com/aravindanm20/coflowai/discussions)

### Support
- **Documentation**: You're reading it!
- **Examples**: See [examples/](../examples/)
- **Troubleshooting**: [TROUBLESHOOTING.md](./TROUBLESHOOTING.md)

---

## Version Information

- **Current Version**: 1.0.0
- **Python Requirement**: 3.12+
- **License**: Apache 2.0

---

## What's Next?

- **New to CoFlowAi?** Start with [Getting Started](./GETTING_STARTED.md)
- **Building an application?** Check the [User Guide](./USER_GUIDE.md)
- **Need API details?** See [API Reference](./API_REFERENCE.md)
- **Having issues?** Check [Troubleshooting](./TROUBLESHOOTING.md)
- **Want to contribute?** Read [Contributing](./CONTRIBUTING.md)

---

**Happy building with CoFlowAi! 🚀**
