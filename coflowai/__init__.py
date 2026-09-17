"""CoFlowAi — a production-grade agentic AI framework for Python.

    The AI can be nondeterministic. The execution platform cannot be uncontrolled.
"""

from __future__ import annotations

__version__ = "1.0.0"

from .agents.agent import Agent
from .app import CoFlowAi
from .approvals.approval import ApprovalRecord, HumanApproval
from .context.builder import ContextBudget, ContextBuilder
from .context.summarizer import ExtractiveSummariser, ModelSummariser
from .core.context import ExecutionContext
from .core.exceptions import (
    AgentError,
    ApprovalRejected,
    BudgetExceeded,
    CheckpointError,
    CoFlowAiError,
    CompilationError,
    ConfigurationError,
    ExecutionCancelled,
    ExecutionError,
    ExecutionTimeout,
    MaxStepsExceeded,
    ModelError,
    PermissionDenied,
    PluginError,
    ToolError,
    ToolNotFound,
    ValidationError,
    WorkflowError,
)
from .core.executable import Executable
from .core.result import ExecutionResult
from .core.types import (
    ExecutionMode,
    ExecutionStatus,
    Message,
    Role,
    ToolCall,
    ToolResult,
    Usage,
)
from .events.event import Event, EventType
from .events.store import EventStore, InMemoryEventStore, JsonlEventStore
from .memory.base import MemoryKind, MemoryRecord, MemoryStore
from .memory.in_memory import InMemoryMemoryStore
from .models.base import (
    ModelCapabilities,
    ModelPricing,
    ModelProvider,
    ModelRequest,
    ModelResponse,
)
from .models.registry import ModelRegistry
from .models.streaming import ChunkType, StreamChunk
from .observability.logger import configure_logging, get_logger
from .observability.metrics import InMemoryMetrics, MetricsSink
from .observability.redaction import redact
from .plugins.plugin import Plugin
from .policies.permissions import PermissionSet
from .policies.policy import Budget, ExecutionPolicy, RetryPolicy
from .runtime.replay import ReplayTrace
from .runtime.runtime import Runtime
from .state.checkpoint import Checkpoint, ExecutionRecord
from .state.store import InMemoryStateStore, StateStore
from .testing.fakes import FakeModelProvider, FakeTool
from .tools.decorator import tool
from .tools.registry import ToolRegistry
from .tools.sandbox import InProcessSandbox, Sandbox, ThreadSandbox
from .tools.subprocess_sandbox import SubprocessSandbox
from .tools.tool import Tool, ToolDefinition
from .workflows.node import (
    LoopNode,
    ParallelNode,
    RouterNode,
    SequenceNode,
    TransformNode,
)
from .workflows.workflow import Workflow

__all__ = [
    "__version__",
    # developer API
    "CoFlowAi", "Agent", "Workflow", "Runtime", "tool", "Tool", "ToolDefinition",
    "HumanApproval", "ApprovalRecord",
    # core
    "Executable", "ExecutionContext", "ExecutionResult", "ExecutionStatus",
    "ExecutionMode", "Message", "Role", "ToolCall", "ToolResult", "Usage",
    # policy & security
    "ExecutionPolicy", "RetryPolicy", "Budget", "PermissionSet",
    # models
    "ModelProvider", "ModelRequest", "ModelResponse", "ModelCapabilities",
    "ModelPricing", "ModelRegistry",
    # workflows
    "SequenceNode", "ParallelNode", "RouterNode", "LoopNode", "TransformNode",
    # events & state
    "Event", "EventType", "EventStore", "InMemoryEventStore", "JsonlEventStore",
    "StateStore", "InMemoryStateStore", "Checkpoint", "ExecutionRecord",
    "ReplayTrace",
    # context & memory
    "ContextBuilder", "ContextBudget", "MemoryStore", "MemoryRecord", "MemoryKind",
    "InMemoryMemoryStore",
    # tools runtime
    "ToolRegistry", "Sandbox", "InProcessSandbox", "ThreadSandbox",
    "SubprocessSandbox",
    # streaming
    "StreamChunk", "ChunkType",
    # summarisation
    "ExtractiveSummariser", "ModelSummariser",
    # observability
    "configure_logging", "get_logger", "MetricsSink", "InMemoryMetrics", "redact",
    # plugins
    "Plugin",
    # testing
    "FakeModelProvider", "FakeTool",
    # errors
    "CoFlowAiError", "ExecutionError", "ExecutionTimeout", "ExecutionCancelled",
    "MaxStepsExceeded", "BudgetExceeded", "WorkflowError", "CompilationError",
    "AgentError", "ModelError", "ToolError", "ToolNotFound", "PermissionDenied",
    "ValidationError", "ApprovalRejected", "CheckpointError", "ConfigurationError",
    "PluginError",
]
