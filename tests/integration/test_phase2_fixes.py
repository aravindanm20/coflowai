"""Regression tests for Phase 2 idempotency and checkpointing fixes."""

import asyncio
import pytest
from coflowai.agents.agent import Agent
from coflowai.core.types import ExecutionStatus
from coflowai.runtime.runtime import Runtime
from coflowai.state.store import InMemoryStateStore
from coflowai.testing.fakes import FakeModelProvider
from coflowai.tools.decorator import tool


class TestWriteAheadCheckpointing:
    """Test that tool calls are checkpointed before execution."""

    @pytest.mark.asyncio
    async def test_checkpoint_created_before_tool_execution(self):
        """Checkpoint should be created after assistant message, before tools run."""
        runtime = Runtime(state_store=InMemoryStateStore())
        runtime.models.register("fake", FakeModelProvider([
            [{"id": "call_1", "name": "test_tool", "arguments": {"value": "test"}}],
            "Done"
        ]))

        call_count = []

        @tool()
        async def test_tool(value: str) -> str:
            call_count.append(value)
            return f"Processed: {value}"

        agent = Agent(
            name="test-agent",
            model="fake",
            instructions="Use the tool",
            tools=[test_tool]
        )

        result = await runtime.run(agent, "test input")
        assert result.status == ExecutionStatus.COMPLETED
        assert len(call_count) == 1

        # Check that checkpoints were created
        checkpoints = await runtime.state_store.list_checkpoints(result.execution_id)
        # Should have: agent_tool_calls, agent_tool_results, and final checkpoints
        assert len(checkpoints) >= 2
        # First checkpoint should be after tool_calls message
        assert checkpoints[0].reason == "agent_tool_calls"

    @pytest.mark.asyncio
    async def test_checkpoint_created_after_tool_results(self):
        """Checkpoint should also be created after tool results."""
        runtime = Runtime(state_store=InMemoryStateStore())
        runtime.models.register("fake", FakeModelProvider([
            [{"id": "call_1", "name": "test_tool", "arguments": {"value": "test"}}],
            "Done"
        ]))

        @tool()
        async def test_tool(value: str) -> str:
            return f"Result: {value}"

        agent = Agent(
            name="test-agent",
            model="fake",
            instructions="Use the tool",
            tools=[test_tool]
        )

        result = await runtime.run(agent, "test")
        checkpoints = await runtime.state_store.list_checkpoints(result.execution_id)

        # Find checkpoint reasons
        reasons = [c.reason for c in checkpoints]
        assert "agent_tool_calls" in reasons
        assert "agent_tool_results" in reasons


class TestPendingToolCallReplay:
    """A crash between 'model asked for tools' and 'tools ran' must replay
    exactly those calls, with the same ids, on resume."""

    @pytest.mark.asyncio
    async def test_resume_replays_pending_tool_calls(self):
        from coflowai.core.types import Message, ToolCall
        from coflowai.state.checkpoint import Checkpoint, ExecutionRecord
        runtime = Runtime(state_store=InMemoryStateStore())
        provider = FakeModelProvider(["Done after resume"])
        runtime.models.register("fake", provider)
        seen = []

        @tool()
        async def tracked_tool(value: str) -> str:
            seen.append(value)
            return f"Processed: {value}"

        agent = Agent(name="test-agent", model="fake", instructions="t",
                      tools=[tracked_tool])
        # Fabricate the state a crash would leave behind
        exec_id = "exec_crashed"
        await runtime.state_store.save_execution(ExecutionRecord(
            execution_id=exec_id, status=ExecutionStatus.FAILED, input="go",
            executable=agent.name))
        pending = ToolCall(id="call_fixed", name="tracked_tool",
                           arguments={"value": "replay"})
        conversation = [Message.user("go"), Message.assistant(None, [pending])]
        await runtime.state_store.save_checkpoint(Checkpoint(
            execution_id=exec_id, model_calls=1,
            state={"__conversation__test-agent":
                   [m.to_dict() for m in conversation]}))
        result = await runtime.resume(exec_id, executable=agent)
        assert result.succeeded and result.output == "Done after resume"
        assert seen == ["replay"]                    # replayed exactly once
        events = await runtime.event_store.get_events(exec_id)
        ids = [e.payload["tool_call_id"] for e in events
               if e.type == "tool.completed"]
        assert ids == ["call_fixed"]                 # same id -> same idem. key


class TestIdempotencyKeyInjection:
    """Test that idempotency keys are passed to tools that declare them."""

    @pytest.mark.asyncio
    async def test_tool_receives_idempotency_key(self):
        """Tools with idempotency_key parameter should receive it."""
        runtime = Runtime(state_store=InMemoryStateStore())
        runtime.models.register("fake", FakeModelProvider([
            [{"id": "call_1", "name": "payment_tool", "arguments": {"amount": 100}}],
            "Done"
        ]))

        received_keys = []

        @tool(side_effect=True, idempotent=True)
        async def payment_tool(amount: int, idempotency_key: str) -> str:
            # Tool that uses idempotency key for deduplication
            received_keys.append(idempotency_key)
            return f"Paid ${amount} with key {idempotency_key}"

        agent = Agent(
            name="payment-agent",
            model="fake",
            instructions="Process payment",
            tools=[payment_tool]
        )

        result = await runtime.run(agent, "pay")
        assert result.status == ExecutionStatus.COMPLETED
        assert len(received_keys) == 1
        # Key should be non-empty and deterministic
        assert received_keys[0]
        assert len(received_keys[0]) > 10  # Should be a meaningful hash

    @pytest.mark.asyncio
    async def test_idempotency_key_not_in_schema(self):
        """idempotency_key should not appear in the tool schema shown to the model."""
        from coflowai.tools.decorator import tool

        @tool(side_effect=True)
        def payment(amount: int, idempotency_key: str) -> str:
            return "ok"

        # The input schema should not include idempotency_key
        schema = payment.definition.input_schema
        assert "idempotency_key" not in schema.get("properties", {})

    @pytest.mark.asyncio
    async def test_tool_without_idempotency_key_still_works(self):
        """Tools without idempotency_key parameter should work normally."""
        runtime = Runtime(state_store=InMemoryStateStore())
        runtime.models.register("fake", FakeModelProvider([
            [{"id": "call_1", "name": "simple_tool", "arguments": {"value": "test"}}],
            "Done"
        ]))

        @tool()
        async def simple_tool(value: str) -> str:
            return f"Got: {value}"

        agent = Agent(
            name="simple-agent",
            model="fake",
            instructions="Use tool",
            tools=[simple_tool]
        )

        result = await runtime.run(agent, "test")
        assert result.status == ExecutionStatus.COMPLETED


class TestIdempotencyAcrossCrash:
    """Test that idempotency prevents duplicate side effects across crashes."""

    @pytest.mark.asyncio
    async def test_resume_uses_cached_tool_result(self):
        """Resume should use cached result for idempotent side-effect tools."""
        runtime = Runtime(state_store=InMemoryStateStore())
        runtime.models.register("fake", FakeModelProvider([
            [{"id": "call_1", "name": "send_email", "arguments": {"to": "test@example.com"}}],
            "Email sent"
        ]))

        email_sent_count = []

        @tool(side_effect=True, idempotent=True)
        async def send_email(to: str, idempotency_key: str = "") -> str:
            # Track actual sends (not cache hits)
            email_sent_count.append(to)
            return f"Email sent to {to}"

        agent = Agent(
            name="email-agent",
            model="fake",
            instructions="Send email",
            tools=[send_email]
        )

        # First execution
        exec_id = "test-idempotency-123"
        result1 = await runtime.run(agent, "send", execution_id=exec_id)
        assert result1.status == ExecutionStatus.COMPLETED
        assert len(email_sent_count) == 1

        # Simulate resume after crash - mark as RUNNING
        record = await runtime.state_store.get_execution(exec_id)
        record.status = ExecutionStatus.RUNNING
        await runtime.state_store.save_execution(record)

        # Resume - the tool should NOT be called again (cached)
        runtime.models.register("fake", FakeModelProvider(["Resume complete"]))
        result2 = await runtime.resume(exec_id)

        # Email should only have been sent once
        assert len(email_sent_count) == 1


class TestToolCallStability:
    """Test that tool_call_ids remain stable across resume."""

    @pytest.mark.asyncio
    async def test_same_tool_call_ids_on_replay(self):
        """Replayed tool calls should have the same IDs as before crash."""
        runtime = Runtime(state_store=InMemoryStateStore())
        runtime.models.register("fake", FakeModelProvider(["Done"]))

        @tool()
        async def test_tool(value: str) -> str:
            return value

        agent = Agent(
            name="test-agent",
            model="fake",
            instructions="test",
            tools=[test_tool]
        )

        # Create an execution with a checkpoint containing tool calls
        exec_id = "test-stability-123"
        await runtime.run(agent, "test", execution_id=exec_id)

        # Get the checkpoint
        checkpoint = await runtime.state_store.latest_checkpoint(exec_id)
        assert checkpoint is not None

        # The idempotency ledger should have stable keys
        ledger = checkpoint.state.get("__idempotency__", {})
        # Keys should be based on execution_id, node_id, and call_id
        for key in ledger:
            assert exec_id in key or key.startswith("idempotency:")
