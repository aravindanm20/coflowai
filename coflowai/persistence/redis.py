"""Redis-backed persistence and distributed locking (requires ``redis>=5``).

Locks use the standard ``SET NX PX`` pattern with a fencing token, and are
released by a Lua script that only deletes the key if the caller still owns it —
so a slow worker whose lease expired can never delete a successor's lock.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from typing import Any

from ..core.exceptions import ConfigurationError
from ..core.types import ExecutionStatus
from ..events.event import Event
from ..events.store import EventStore
from ..state.checkpoint import Checkpoint, ExecutionRecord
from ..state.store import StateStore

__all__ = ["RedisEventStore", "RedisStateStore", "create_client"]

#: Delete the key only if we still own it (prevents releasing someone else's lock).
_RELEASE_SCRIPT = """
if redis.call('get', KEYS[1]) == ARGV[1] then
    return redis.call('del', KEYS[1])
else
    return 0
end
"""


def create_client(url: str = "redis://localhost:6379/0", **kwargs: Any) -> Any:
    try:
        import redis.asyncio as redis
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ConfigurationError(
            'redis is required: pip install "coflowai[redis]"', cause=exc,
        ) from exc
    return redis.from_url(url, decode_responses=True, **kwargs)


class _RedisBase:
    def __init__(self, client: Any, *, namespace: str = "coflowai") -> None:
        self.client = client
        self.ns = namespace

    def _key(self, *parts: str) -> str:
        return ":".join((self.ns, *parts))


class RedisEventStore(_RedisBase, EventStore):
    """Events in a per-execution list; sequence from an atomic counter."""

    async def append(self, event: Event) -> Event:
        sequence_key = self._key("seq", event.execution_id)
        event.sequence = int(await self.client.incr(sequence_key))
        await self.client.rpush(self._key("events", event.execution_id),
                                json.dumps(event.to_dict(), default=str))
        await self.client.sadd(self._key("executions"), event.execution_id)
        return event

    async def get_events(self, execution_id: str) -> list[Event]:
        raw = await self.client.lrange(self._key("events", execution_id), 0, -1)
        return [Event.from_dict(json.loads(item)) for item in raw]

    async def list_executions(self) -> list[str]:
        return sorted(await self.client.smembers(self._key("executions")))


class RedisStateStore(_RedisBase, StateStore):
    def __init__(self, client: Any, *, namespace: str = "coflowai",
                 owner: str | None = None) -> None:
        super().__init__(client, namespace=namespace)
        self.owner = owner or uuid.uuid4().hex
        self._tokens: dict[str, str] = {}

    # ------------------------------------------------------------- executions
    async def save_execution(self, record: ExecutionRecord) -> None:
        payload = record.to_dict()
        payload["input"] = record.input
        payload["output"] = record.output
        await self.client.set(self._key("execution", record.execution_id),
                              json.dumps(payload, default=str))
        await self.client.zadd(self._key("executions_by_time"),
                               {record.execution_id: record.created_at.timestamp()})

    async def get_execution(self, execution_id: str) -> ExecutionRecord | None:
        raw = await self.client.get(self._key("execution", execution_id))
        return _to_record(json.loads(raw)) if raw else None

    async def list_executions(self, *, status: str | None = None,
                              limit: int = 100) -> list[ExecutionRecord]:
        ids = await self.client.zrevrange(self._key("executions_by_time"), 0,
                                          max(0, limit * 4 - 1))
        records: list[ExecutionRecord] = []
        for execution_id in ids:
            record = await self.get_execution(execution_id)
            if record is None:
                continue
            if status and record.status.value != status:
                continue
            records.append(record)
            if len(records) >= limit:
                break
        return records

    # ------------------------------------------------------------ checkpoints
    async def save_checkpoint(self, checkpoint: Checkpoint) -> Checkpoint:
        await self.client.rpush(
            self._key("checkpoints", checkpoint.execution_id),
            json.dumps(checkpoint.to_dict(), default=str))
        return checkpoint

    async def list_checkpoints(self, execution_id: str) -> list[Checkpoint]:
        raw = await self.client.lrange(self._key("checkpoints", execution_id), 0, -1)
        return [_to_checkpoint(json.loads(item)) for item in raw]

    # ------------------------------------------------------------------ locks
    async def acquire_lock(self, execution_id: str, *, ttl: float = 60.0) -> bool:
        token = f"{self.owner}:{uuid.uuid4().hex}"
        acquired = await self.client.set(self._key("lock", execution_id), token,
                                         nx=True, px=int(ttl * 1000))
        if acquired:
            self._tokens[execution_id] = token
            return True
        return False

    async def release_lock(self, execution_id: str) -> None:
        token = self._tokens.pop(execution_id, None)
        if token is None:
            return
        await self.client.eval(_RELEASE_SCRIPT, 1,
                               self._key("lock", execution_id), token)

    async def extend_lock(self, execution_id: str, *, ttl: float = 60.0) -> bool:
        """Heartbeat for long-running executions."""
        token = self._tokens.get(execution_id)
        if token is None:
            return False
        current = await self.client.get(self._key("lock", execution_id))
        if current != token:
            return False
        await self.client.pexpire(self._key("lock", execution_id), int(ttl * 1000))
        return True


# --------------------------------------------------------------------------- #
# mapping
# --------------------------------------------------------------------------- #
def _to_record(data: dict[str, Any]) -> ExecutionRecord:
    record = ExecutionRecord(
        execution_id=data["execution_id"],
        status=ExecutionStatus(data["status"]),
        input=data.get("input"), output=data.get("output"),
        error=data.get("error"), executable=data.get("executable", ""),
        workflow_hash=data.get("workflow_hash"), policy=data.get("policy", {}),
        metadata=data.get("metadata", {}),
        parent_execution_id=data.get("parent_execution_id"),
    )
    record.created_at = datetime.fromisoformat(data["created_at"])
    record.updated_at = datetime.fromisoformat(data["updated_at"])
    return record


def _to_checkpoint(data: dict[str, Any]) -> Checkpoint:
    return Checkpoint(
        execution_id=data["execution_id"], step=data["step"],
        node_id=data.get("node_id"), state=data.get("state", {}),
        model_calls=data.get("model_calls", 0), tool_calls=data.get("tool_calls", 0),
        input_tokens=data.get("input_tokens", 0),
        output_tokens=data.get("output_tokens", 0), cost=data.get("cost", 0.0),
        status=ExecutionStatus(data.get("status", "running")),
        workflow_hash=data.get("workflow_hash"), reason=data.get("reason", "auto"),
        checkpoint_id=data["checkpoint_id"],
        created_at=datetime.fromisoformat(data["created_at"]),
    )
