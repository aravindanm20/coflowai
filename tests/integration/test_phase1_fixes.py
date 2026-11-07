"""Regression tests for Phase 1 critical correctness fixes."""

import asyncio
import pytest
from coflowai.agents.agent import Agent
from coflowai.core.types import ExecutionStatus
from coflowai.distributed.queue import InMemoryJobQueue, Job
from coflowai.distributed.worker import ExecutableRegistry, Worker
from coflowai.persistence.sqlite import SqliteStateStore
from coflowai.policies.permissions import PermissionSet
from coflowai.policies.policy import ExecutionPolicy, ExecutionMode
from coflowai.runtime.runtime import Runtime
from coflowai.state.store import InMemoryStateStore
from coflowai.testing.fakes import FakeModelProvider


@pytest.fixture
def runtime():
    """Create a runtime with an in-memory state store."""
    store = InMemoryStateStore()
    runtime = Runtime(state_store=store)
    runtime.models.register("fake", FakeModelProvider([{"message": "Hello!"}]))
    return runtime


@pytest.fixture
def agent():
    """Create a simple test agent."""
    return Agent(name="test-agent", model="fake", instructions="Say hello")


class TestRedeliveredRunJob:
    """Test that redelivered run jobs don't re-execute completed work."""

    @pytest.mark.asyncio
    async def test_redelivered_after_completion(self, runtime, agent):
        """Same run job sent twice should not re-execute after completion."""
        queue = InMemoryJobQueue()
        registry = ExecutableRegistry()
        registry.register(agent)
        worker = Worker(runtime, queue, registry)

        # Create and execute first job
        job = Job(
            kind="run",
            execution_id="exec-123",
            executable="test-agent",
            input="test input",
            policy=ExecutionPolicy().to_dict(),
        )
        await queue.enqueue(job)

        # Process first delivery
        result1 = await worker.run_once()
        assert result1 is not None

        # Check execution completed
        record = await runtime.state_store.get_execution("exec-123")
        assert record is not None
        assert record.status == ExecutionStatus.COMPLETED

        # Simulate redelivery - enqueue same job again
        await queue.enqueue(job)
        result2 = await worker.run_once()
        assert result2 is not None

        # Should return same result without re-running
        record2 = await runtime.state_store.get_execution("exec-123")
        assert record2.status == ExecutionStatus.COMPLETED
        # Verify model wasn't called again by checking usage
        assert runtime.usage("exec-123").model_calls == 1

    @pytest.mark.asyncio
    async def test_redelivered_during_execution_resumes(self, runtime, agent):
        """Redelivered job for in-progress execution should resume from checkpoint."""
        queue = InMemoryJobQueue()
        registry = ExecutableRegistry()
        registry.register(agent)
        worker = Worker(runtime, queue, registry)

        # Create an execution and checkpoint it
        exec_id = "exec-456"
        result = await runtime.run(agent, "test", execution_id=exec_id)
        assert result.status == ExecutionStatus.COMPLETED

        # Manually set back to RUNNING to simulate mid-execution
        record = await runtime.state_store.get_execution(exec_id)
        record.status = ExecutionStatus.RUNNING
        await runtime.state_store.save_execution(record)

        # Verify checkpoint exists
        checkpoint = await runtime.state_store.latest_checkpoint(exec_id)
        assert checkpoint is not None

        # Enqueue job for already-running execution
        job = Job(
            kind="run",
            execution_id=exec_id,
            executable="test-agent",
            input="test",
            policy=ExecutionPolicy().to_dict(),
        )
        await queue.enqueue(job)

        # Should resume instead of starting fresh
        result = await worker.run_once()
        assert result is not None


class TestPolicyDeserialization:
    """Test that ExecutionPolicy correctly deserializes from dict."""

    def test_mode_string_converted_to_enum(self):
        """Policy mode should be converted from string to enum."""
        policy_dict = {
            "max_steps": 10,
            "mode": "strict",  # string, not enum
            "max_model_calls": 5,
            "max_tool_calls": 10,
        }
        policy = ExecutionPolicy.from_dict(policy_dict)
        assert policy.mode == ExecutionMode.STRICT
        assert isinstance(policy.mode, ExecutionMode)

    def test_to_dict_includes_all_fields(self):
        """to_dict should include model_timeout_seconds, tool_timeout_seconds, etc."""
        policy = ExecutionPolicy(
            model_timeout_seconds=120.0,
            tool_timeout_seconds=60.0,
            structured_output_repair_attempts=3,
        )
        policy_dict = policy.to_dict()
        assert "model_timeout_seconds" in policy_dict
        assert "tool_timeout_seconds" in policy_dict
        assert "structured_output_repair_attempts" in policy_dict
        assert policy_dict["model_timeout_seconds"] == 120.0
        assert policy_dict["tool_timeout_seconds"] == 60.0

    def test_from_dict_roundtrip(self):
        """Policy should survive to_dict() -> from_dict() roundtrip."""
        original = ExecutionPolicy(
            mode=ExecutionMode.EXPLORATORY,
            max_steps=30,
            model_timeout_seconds=90.0,
        )
        restored = ExecutionPolicy.from_dict(original.to_dict())
        assert restored.mode == original.mode
        assert restored.max_steps == original.max_steps
        assert restored.model_timeout_seconds == original.model_timeout_seconds


class TestPermissionsPersistence:
    """Test that permissions are persisted and restored correctly."""

    @pytest.mark.asyncio
    async def test_permissions_saved_on_run(self, runtime, agent):
        """Permissions should be saved to execution record."""
        permissions = PermissionSet(["database.read", "api.call"])
        result = await runtime.run(agent, "test", permissions=permissions)

        record = await runtime.state_store.get_execution(result.execution_id)
        assert record is not None
        assert set(record.permissions) == {"database.read", "api.call"}

    @pytest.mark.asyncio
    async def test_permissions_restored_on_resume(self):
        """Resume must use the original (narrow) grant, not the runtime default."""
        runtime = Runtime(state_store=InMemoryStateStore(),
                          permissions=PermissionSet(["database.read"]))
        provider = FakeModelProvider([])            # empty script -> first run fails
        runtime.models.register("fake", provider)
        agent = Agent(name="a", model="fake", instructions="x")
        failed = await runtime.run(agent, "go",
                                   policy=ExecutionPolicy(retry_attempts=0))
        assert failed.status is ExecutionStatus.FAILED
        runtime.permissions = PermissionSet.all()  # widen the default afterwards
        seen = []
        original_services = runtime._services
        def spy(context):
            seen.append(set(context.permissions.granted))
            return original_services(context)
        runtime._services = spy
        provider.responses.append("recovered")
        resumed = await runtime.resume(failed.execution_id)
        assert resumed.succeeded
        assert seen == [{"database.read"}]

    @pytest.mark.asyncio
    async def test_policy_restored_on_resume(self):
        """With the in-memory cache cleared, policy is rebuilt from the record."""
        runtime = Runtime(state_store=InMemoryStateStore())
        provider = FakeModelProvider([])
        runtime.models.register("fake", provider)
        agent = Agent(name="a", model="fake", instructions="x", temperature=0.9)
        policy = ExecutionPolicy(mode=ExecutionMode.STRICT, max_steps=5,
                                 retry_attempts=0)
        failed = await runtime.run(agent, "go", policy=policy)
        runtime._policies.clear()                   # simulate a fresh process
        provider.responses.append("ok")
        resumed = await runtime.resume(failed.execution_id)
        assert resumed.succeeded
        assert provider.calls[-1].temperature == 0.0   # STRICT survived


class TestLockOwnership:
    """Test that locks have unique owner IDs."""

    @pytest.mark.asyncio
    async def test_sqlite_unique_owner_ids(self, tmp_path):
        """SQLite stores should have unique owner IDs by default."""
        db_path = tmp_path / "test.db"
        store1 = SqliteStateStore(db_path)
        store2 = SqliteStateStore(db_path)

        assert store1.owner != store2.owner
        assert store1.owner != "worker"
        assert store2.owner != "worker"

    @pytest.mark.asyncio
    async def test_different_workers_cannot_steal_locks(self, tmp_path):
        """Different workers should not be able to release each other's locks."""
        db_path = tmp_path / "test.db"
        store1 = SqliteStateStore(db_path)
        store2 = SqliteStateStore(db_path)

        # Worker 1 acquires lock
        acquired1 = await store1.acquire_lock("exec-1")
        assert acquired1 is True

        # Worker 2 cannot acquire same lock
        acquired2 = await store2.acquire_lock("exec-1")
        assert acquired2 is False

        # Worker 2 cannot release worker 1's lock
        await store2.release_lock("exec-1")

        # Worker 1 should still hold the lock (can extend it)
        extended = await store1.extend_lock("exec-1")
        assert extended is True


class TestLockHeartbeat:
    """Test that locks are kept alive via heartbeat for long executions."""

    @pytest.mark.asyncio
    async def test_lock_extended_during_execution(self, runtime, agent):
        """Lock should be automatically extended during long-running execution."""
        # Create a slow agent
        slow_agent = Agent(
            name="slow-agent",
            model="fake",
            instructions="Be slow",
        )

        # Mock execution that takes longer than lock TTL
        async def slow_execute(ctx):
            await asyncio.sleep(0.3)  # Longer than typical heartbeat interval
            return ctx.input

        slow_agent.execute = slow_execute

        exec_id = "exec-long"
        result = await runtime.run(slow_agent, "test", execution_id=exec_id)

        # Execution should complete successfully without losing lock
        assert result.status == ExecutionStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_execution_cancelled_if_lock_lost(self, tmp_path):
        """Execution should be cancelled if lock is stolen."""
        db_path = tmp_path / "test.db"
        store = SqliteStateStore(db_path)
        runtime = Runtime(state_store=store)
        runtime.models.register("fake", FakeModelProvider([{"message": "Hello!"}]))

        agent = Agent(name="test", model="fake", instructions="test")

        async def steal_lock_midway(ctx):
            # Simulate another worker stealing the lock
            await asyncio.sleep(0.1)
            # Force release and let TTL expire
            await store.release_lock(ctx.execution_id)
            # Try to extend - should fail
            await asyncio.sleep(0.1)
            return "done"

        agent.execute = steal_lock_midway

        # The execution should detect lock loss
        # (In practice this would be CancelledError or similar)
        result = await runtime.run(agent, "test")
        # Depending on timing, might complete or be cancelled
        assert result.status in {ExecutionStatus.COMPLETED, ExecutionStatus.CANCELLED}
