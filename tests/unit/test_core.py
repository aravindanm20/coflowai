"""Core primitives: ids, types, policies, permissions, redaction, context."""

from __future__ import annotations

import pytest

from coflowai import (
    Budget,
    BudgetExceeded,
    ExecutionContext,
    ExecutionMode,
    ExecutionPolicy,
    ExecutionStatus,
    MaxStepsExceeded,
    Message,
    PermissionSet,
    RetryPolicy,
    Role,
    Usage,
    redact,
)
from coflowai.core.ids import idempotency_key, new_id, stable_hash
from coflowai.core.types import VALID_TRANSITIONS


def test_ids_are_prefixed_and_unique():
    a, b = new_id("exec"), new_id("exec")
    assert a.startswith("exec_") and a != b


def test_stable_hash_is_deterministic():
    assert stable_hash("a", 1) == stable_hash("a", 1)
    assert stable_hash("a", 1) != stable_hash("a", 2)


def test_idempotency_key_depends_on_execution_node_and_call():
    base = idempotency_key("exec_1", "node_1", "call_1")
    assert base == idempotency_key("exec_1", "node_1", "call_1")
    assert base != idempotency_key("exec_1", "node_1", "call_2")
    assert base != idempotency_key("exec_2", "node_1", "call_1")


def test_message_roundtrip():
    message = Message.assistant("hi")
    assert Message.from_dict(message.to_dict()).content == "hi"
    assert Message.tool("call_1", "weather", "{}").role is Role.TOOL


def test_usage_addition():
    total = Usage(input_tokens=10, cost=0.5)
    total.add(Usage(input_tokens=5, output_tokens=2, cost=0.25))
    assert total.input_tokens == 15
    assert total.total_tokens == 17
    assert total.cost == 0.75


def test_state_machine_forbids_resurrection():
    assert VALID_TRANSITIONS[ExecutionStatus.COMPLETED] == frozenset()
    assert ExecutionStatus.RUNNING in VALID_TRANSITIONS[
        ExecutionStatus.WAITING_FOR_APPROVAL]


# --------------------------------------------------------------------- policy
def test_policy_rejects_unbounded_loops():
    with pytest.raises(ValueError):
        ExecutionPolicy(max_steps=0)


def test_policy_enforces_step_and_call_limits():
    policy = ExecutionPolicy(max_steps=2, max_model_calls=1, max_tool_calls=1)
    policy.check_step(2)
    with pytest.raises(MaxStepsExceeded):
        policy.check_step(3)
    with pytest.raises(BudgetExceeded):
        policy.check_model_call(model_calls=1, tokens=0, cost=0)
    with pytest.raises(BudgetExceeded):
        policy.check_tool_call(tool_calls=1, tokens=0, cost=0)


def test_policy_enforces_token_and_cost_budgets():
    policy = ExecutionPolicy(max_tokens=100, max_cost=1.0)
    with pytest.raises(BudgetExceeded):
        policy.check_model_call(model_calls=0, tokens=101, cost=0)
    with pytest.raises(BudgetExceeded):
        policy.check_model_call(model_calls=0, tokens=0, cost=1.5)


def test_strict_mode_forces_zero_temperature():
    assert ExecutionPolicy(mode=ExecutionMode.STRICT).temperature_override() == 0.0
    assert ExecutionPolicy().temperature_override() is None


def test_retry_backoff_is_bounded():
    policy = RetryPolicy(attempts=5, base_delay=1, max_delay=4, jitter=False)
    assert policy.delay_for(1) == 1
    assert policy.delay_for(2) == 2
    assert policy.delay_for(10) == 4


def test_budget_utilisation():
    budget = Budget(max_tokens=100, max_cost=10)
    assert budget.utilisation(cost=5, tokens=90, model_calls=0, tool_calls=0) == 0.9
    assert budget.remaining(cost=5, tokens=90, model_calls=0,
                            tool_calls=0)["tokens"] == 10


# ---------------------------------------------------------------- permissions
@pytest.mark.parametrize(
    ("granted", "required", "expected"),
    [
        ({"database.read"}, "database.read", True),
        ({"database.read"}, "database.write", False),
        ({"database.*"}, "database.write", True),
        ({"*"}, "anything.at.all", True),
        (set(), None, True),
        (set(), "network.external", False),
    ],
)
def test_permission_matching(granted, required, expected):
    assert PermissionSet(granted).allows(required) is expected


def test_permission_intersection_is_least_privilege():
    parent = PermissionSet({"database.read", "email.send"})
    child = parent.intersect(PermissionSet({"database.read", "deployment.execute"}))
    assert child.granted == {"database.read"}


# ------------------------------------------------------------------ redaction
def test_redaction_hides_secrets():
    payload = {"api_key": "sk-abcdef1234567890abcd", "nested": {"password": "hunter2"},
               "note": "connect via postgres://user:pw@host/db"}
    cleaned = redact(payload)
    assert cleaned["api_key"] == "***redacted***"
    assert cleaned["nested"]["password"] == "***redacted***"
    assert "postgres://" not in cleaned["note"]


# -------------------------------------------------------------------- context
def test_context_counters_and_children():
    context = ExecutionContext(input="x", policy=ExecutionPolicy(max_steps=3))
    child = context.child(input="y", node_id="n1")
    child.model_calls += 2
    child.cost += 1.5
    context.merge_from(child)
    assert context.model_calls == 2 and context.cost == 1.5
    assert child.input == "y" and child.node_id == "n1"


def test_context_step_limit():
    context = ExecutionContext(policy=ExecutionPolicy(max_steps=1))
    context.next_step()
    with pytest.raises(MaxStepsExceeded):
        context.next_step()


def test_context_cancellation():
    from coflowai import ExecutionCancelled

    context = ExecutionContext()
    context.cancel()
    with pytest.raises(ExecutionCancelled):
        context.ensure_alive()
