# Contributing to CoFlowAi

Thank you for your interest in contributing to CoFlowAi! This guide will help you get started.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Project Structure](#project-structure)
- [Development Workflow](#development-workflow)
- [Testing](#testing)
- [Code Style](#code-style)
- [Documentation](#documentation)
- [Pull Request Process](#pull-request-process)

---

## Code of Conduct

### Our Pledge

We are committed to providing a welcoming and inspiring community for all. Please be respectful and constructive in all interactions.

### Our Standards

- Be respectful and inclusive
- Accept constructive criticism gracefully
- Focus on what is best for the community
- Show empathy towards other community members

---

## Getting Started

### Ways to Contribute

- **Report Bugs** - Found a bug? Open an issue
- **Suggest Features** - Have an idea? Start a discussion
- **Fix Issues** - Check the issues labeled `good-first-issue`
- **Improve Documentation** - Documentation can always be better
- **Write Tests** - Increase test coverage
- **Add Examples** - Show others how to use CoFlowAi

### Before You Start

1. Check existing issues and PRs to avoid duplication
2. For major changes, open an issue first to discuss
3. Read this contributing guide completely
4. Set up your development environment

---

## Development Setup

### Prerequisites

- Python 3.12 or higher
- Git
- Virtual environment tool (venv, conda, etc.)

### Setup Steps

```bash
# 1. Fork the repository on GitHub

# 2. Clone your fork
git clone https://github.com/YOUR_USERNAME/coflowai.git
cd coflowai

# 3. Add upstream remote
git remote add upstream https://github.com/aravindanm20/coflowai.git

# 4. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 5. Install in development mode with all extras
pip install -e ".[dev,all]"

# 6. Install pre-commit hooks (optional)
pre-commit install
```

### Verify Installation

```bash
# Run tests
pytest tests -v

# Run type checking
mypy coflowai

# Run linting
ruff check coflowai
```

---

## Project Structure

```
coflowai/
├── coflowai/                 # Main package
│   ├── core/                 # Core abstractions
│   │   ├── executable.py     # Executable interface
│   │   ├── context.py        # ExecutionContext
│   │   ├── result.py         # ExecutionResult
│   │   └── exceptions.py     # Error hierarchy
│   ├── agents/               # Agent implementation
│   ├── workflows/            # Workflow orchestration
│   ├── tools/                # Tool system
│   ├── models/               # Model providers
│   ├── runtime/              # Execution runtime
│   ├── policies/             # Policies and budgets
│   ├── events/               # Event system
│   ├── state/                # State management
│   ├── context/              # Context building
│   ├── memory/               # Memory stores
│   ├── observability/        # Logging, tracing, metrics
│   ├── persistence/          # Durable storage
│   ├── distributed/          # Distributed execution
│   ├── providers/            # Provider adapters
│   │   ├── openai.py
│   │   ├── anthropic.py
│   │   ├── gemini.py
│   │   └── bedrock.py
│   ├── testing/              # Test utilities
│   └── plugins/              # Plugin system
├── tests/                    # Test suite
│   ├── unit/
│   ├── integration/
│   └── fixtures/
├── examples/                 # Example scripts
├── docs/                     # Documentation
└── pyproject.toml           # Project configuration
```

### Key Modules

- **core/** - Fundamental abstractions, no dependencies on other modules
- **runtime/** - Execution engine, orchestrates everything
- **providers/** - External provider adapters, optional dependencies
- **persistence/** - Storage backends, optional dependencies
- **testing/** - Test utilities for users and internal tests

---

## Development Workflow

### Branching Strategy

```bash
# Create feature branch
git checkout -b feature/your-feature-name

# Create bugfix branch
git checkout -b fix/issue-123

# Keep your branch updated
git fetch upstream
git rebase upstream/main
```

### Commit Messages

Use clear, descriptive commit messages:

```
<type>: <subject>

<body>

<footer>
```

**Types:**
- `feat` - New feature
- `fix` - Bug fix
- `docs` - Documentation changes
- `test` - Test additions or changes
- `refactor` - Code refactoring
- `perf` - Performance improvements
- `chore` - Maintenance tasks

**Examples:**

```
feat: add support for Gemini structured output

Implement native responseSchema support for Gemini provider.
Falls back to tool-based emulation if not supported.

Closes #123
```

```
fix: prevent race condition in lock acquisition

Add fenced tokens to state store lock operations to prevent
ABA problem in distributed execution.

Fixes #456
```

### Making Changes

1. **Write Tests First** (TDD approach recommended)

```python
# tests/test_my_feature.py
def test_my_new_feature():
    """Test the new feature."""
    result = my_new_feature()
    assert result == expected
```

2. **Implement Feature**

```python
# coflowai/module/feature.py
def my_new_feature():
    """Implementation."""
    return result
```

3. **Run Tests**

```bash
pytest tests/test_my_feature.py -v
```

4. **Run Full Test Suite**

```bash
pytest tests -v
```

5. **Check Code Quality**

```bash
ruff check coflowai
mypy coflowai
```

---

## Testing

### Test Structure

```
tests/
├── unit/                    # Unit tests (fast, isolated)
│   ├── test_agents.py
│   ├── test_workflows.py
│   └── test_tools.py
├── integration/             # Integration tests (slower)
│   ├── test_providers.py
│   └── test_persistence.py
└── fixtures/               # Test fixtures and utilities
    └── conftest.py
```

### Writing Tests

#### Unit Tests

```python
import pytest
from coflowai import Agent
from coflowai.testing import build_test_app

@pytest.mark.asyncio
async def test_agent_execution():
    """Test basic agent execution."""
    app, model = build_test_app(["Hello!"])
    agent = Agent("test", "Say hello", "fake-model")
    
    result = await app.run(agent, "Hi")
    
    assert result.succeeded
    assert result.output == "Hello!"
    assert model.call_count == 1
```

#### Integration Tests

```python
@pytest.mark.integration
@pytest.mark.asyncio
async def test_real_provider():
    """Test with real provider (requires API key)."""
    import os
    
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")
    
    provider = OpenAIProvider(api_key=os.getenv("OPENAI_API_KEY"))
    # ... test with real provider
```

#### Test Fixtures

```python
# tests/fixtures/conftest.py
import pytest

@pytest.fixture
def test_app():
    """Reusable test application."""
    from coflowai.testing import build_test_app
    app, model = build_test_app(["response"])
    return app, model

@pytest.fixture
def sample_agent():
    """Reusable test agent."""
    return Agent("test", "Test agent", "fake-model")
```

### Running Tests

```bash
# Run all tests
pytest tests -v

# Run specific test file
pytest tests/unit/test_agents.py -v

# Run specific test
pytest tests/unit/test_agents.py::test_agent_execution -v

# Run with coverage
pytest tests --cov=coflowai --cov-report=html

# Run only unit tests (fast)
pytest tests/unit -v

# Run integration tests
pytest tests/integration -v -m integration

# Run in parallel
pytest tests -n auto
```

### Test Guidelines

1. **Test Isolation** - Tests should not depend on each other
2. **No Network** - Unit tests should not make network calls
3. **No API Keys** - Unit tests should not require API keys
4. **Fast Execution** - Unit tests should run quickly (<1s each)
5. **Clear Assertions** - Make expectations explicit
6. **Descriptive Names** - Test names should describe what they test

---

## Code Style

### Python Style

Follow [PEP 8](https://pep8.org/) with these specifics:

- Line length: 90 characters
- Use type hints
- Use async/await for I/O operations
- Prefer composition over inheritance

### Type Hints

```python
from typing import Any, Optional
from collections.abc import Callable

async def my_function(
    arg1: str,
    arg2: int | None = None,
    callback: Callable[[str], Any] | None = None,
) -> dict[str, Any]:
    """Function with proper type hints."""
    return {"result": arg1}
```

### Docstrings

Use Google-style docstrings:

```python
def complex_function(param1: str, param2: int) -> dict:
    """Brief description.
    
    Longer description if needed. Explain the purpose,
    behavior, and any important details.
    
    Args:
        param1: Description of param1
        param2: Description of param2
    
    Returns:
        Description of return value
    
    Raises:
        ValueError: When param2 is negative
        RuntimeError: When operation fails
    
    Example:
        >>> result = complex_function("test", 42)
        >>> print(result)
        {'status': 'success'}
    """
    if param2 < 0:
        raise ValueError("param2 must be non-negative")
    return {"status": "success"}
```

### Linting

```bash
# Run ruff
ruff check coflowai

# Auto-fix issues
ruff check coflowai --fix

# Check specific file
ruff check coflowai/agents/agent.py
```

### Type Checking

```bash
# Run mypy
mypy coflowai

# Check specific module
mypy coflowai/agents
```

### Code Quality Rules

1. **No Global Mutable State** - All state in ExecutionContext
2. **Explicit Error Handling** - Never silent failures
3. **Interface over Implementation** - Depend on abstractions
4. **Minimal Dependencies** - Core has minimal deps
5. **Provider Independence** - Core never imports provider SDKs

---

## Documentation

### Documentation Structure

- **Getting Started** - Installation, quickstart
- **Core Concepts** - Architecture, design principles
- **API Reference** - Complete API documentation
- **User Guide** - Common patterns, recipes
- **Advanced Topics** - Complex use cases
- **Examples** - Working code examples

### Writing Documentation

#### Inline Documentation

```python
def my_function(arg: str) -> dict:
    """Brief one-line description.
    
    More detailed explanation of what the function does,
    when to use it, and any important considerations.
    """
    pass
```

#### Markdown Documentation

```markdown
# Title

Brief introduction.

## Section

Content with examples:

\`\`\`python
# Code example
result = my_function("test")
\`\`\`

**Key Points:**
- Point 1
- Point 2
```

### Examples

Add examples to `examples/`:

```python
"""
Brief description of what this example demonstrates.

Key concepts:
- Concept 1
- Concept 2
"""

import asyncio
from coflowai import Agent

# Clear, commented code
agent = Agent("demo", "Demonstrate feature", "fake-model")

async def main():
    # Explanation of what happens
    result = await app.run(agent, "request")
    print(result.output)

if __name__ == "__main__":
    asyncio.run(main())
```

---

## Pull Request Process

### Before Submitting

1. ✅ Tests pass: `pytest tests -v`
2. ✅ Type checking passes: `mypy coflowai`
3. ✅ Linting passes: `ruff check coflowai`
4. ✅ Documentation updated
5. ✅ Examples added (if applicable)
6. ✅ CHANGELOG.md updated

### Submitting PR

1. **Push to Your Fork**

```bash
git push origin feature/your-feature
```

2. **Create Pull Request**

- Go to GitHub and create PR from your fork
- Fill in the PR template
- Link related issues

3. **PR Template**

```markdown
## Description

Brief description of changes.

## Related Issues

Closes #123
Fixes #456

## Changes

- Change 1
- Change 2

## Testing

- [ ] Unit tests added/updated
- [ ] Integration tests added (if applicable)
- [ ] All tests pass
- [ ] Manual testing performed

## Documentation

- [ ] Code comments added
- [ ] API documentation updated
- [ ] Examples added/updated

## Checklist

- [ ] Code follows project style guidelines
- [ ] Tests pass locally
- [ ] Type checking passes
- [ ] Linting passes
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
```

### Review Process

1. Maintainers review your PR
2. Address feedback and make requested changes
3. Keep PR updated with main branch
4. Once approved, maintainers will merge

### After Merge

```bash
# Update your fork
git checkout main
git fetch upstream
git merge upstream/main
git push origin main

# Delete feature branch
git branch -d feature/your-feature
git push origin --delete feature/your-feature
```

---

## Release Process

(For maintainers)

### Version Bumping

```bash
# Update version in pyproject.toml
# Update CHANGELOG.md

# Create tag
git tag -a v1.1.0 -m "Release v1.1.0"
git push origin v1.1.0
```

### Publishing to PyPI

```bash
# Build distribution
python -m build

# Upload to PyPI
python -m twine upload dist/*
```

---

## Getting Help

- **Questions** - Open a [Discussion](https://github.com/aravindanm20/coflowai/discussions)
- **Bugs** - Open an [Issue](https://github.com/aravindanm20/coflowai/issues)

---

## Recognition

Contributors are recognized in:
- CONTRIBUTORS.md
- Release notes
- Project README

Thank you for contributing to CoFlowAi! 🎉
