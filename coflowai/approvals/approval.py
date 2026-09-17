"""Human-in-the-loop approvals.

An approval pauses the execution *durably*: state is checkpointed, the worker is
released, and the execution is resumed later by ``runtime.approve(...)`` or
failed by ``runtime.reject(...)``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from ..core.context import ExecutionContext
from ..core.exceptions import ApprovalRejected, CoFlowAiError
from ..core.executable import Executable, as_executable
from ..core.ids import stable_hash
from ..core.result import ExecutionResult
from ..core.types import ExecutionStatus

__all__ = ["ApprovalRequired", "ApprovalRecord", "HumanApproval",
           "APPROVALS_STATE_KEY"]

APPROVALS_STATE_KEY = "__approvals__"


class ApprovalRequired(CoFlowAiError):
    """Control-flow signal: pause and wait for a human decision."""

    def __init__(self, approval_id: str, message: str, *,
                 node_id: str | None = None, payload: Any = None) -> None:
        super().__init__(message, approval_id=approval_id, node_id=node_id)
        self.approval_id = approval_id
        self.prompt = message
        self.node_id = node_id
        self.payload = payload


@dataclass(slots=True)
class ApprovalRecord:
    approval_id: str
    message: str
    status: str = "pending"          # pending | approved | rejected
    decided_by: str | None = None
    note: str | None = None
    payload: Any = None
    requested_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat())
    decided_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "approval_id": self.approval_id,
            "message": self.message,
            "status": self.status,
            "decided_by": self.decided_by,
            "note": self.note,
            "payload": self.payload,
            "requested_at": self.requested_at,
            "decided_at": self.decided_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ApprovalRecord":
        return cls(**data)


class HumanApproval(Executable):
    """Gate an action behind an explicit human decision."""

    def __init__(self, message: str, *, action: Any = None,
                 name: str = "approval", approval_id: str | None = None,
                 on_reject: str = "fail") -> None:
        self.name = name
        self.message = message
        self.action = as_executable(action) if action is not None else None
        self.approval_id = approval_id
        self.on_reject = on_reject          # "fail" | "skip"

    @property
    def required_permissions(self) -> list[str]:
        return self.action.required_permissions if self.action else []

    @property
    def model_requirements(self) -> dict[str, bool]:
        return self.action.model_requirements if self.action else {}

    def signature(self) -> str:
        return stable_hash("approval", self.name, self.message,
                           self.action.signature() if self.action else None)

    def _id(self, context: ExecutionContext) -> str:
        return self.approval_id or context.node_id or self.name

    async def execute(self, context: ExecutionContext) -> ExecutionResult:
        context.ensure_alive()
        approval_id = self._id(context)
        ledger: dict[str, Any] = context.state.setdefault(APPROVALS_STATE_KEY, {})
        raw = ledger.get(approval_id)
        record = ApprovalRecord.from_dict(raw) if raw else None

        if record is None or record.status == "pending":
            if record is None:
                record = ApprovalRecord(approval_id=approval_id,
                                        message=self.message,
                                        payload=_summarise(context.input))
                ledger[approval_id] = record.to_dict()
            raise ApprovalRequired(approval_id, self.message,
                                   node_id=context.node_id,
                                   payload=record.payload)

        if record.status == "rejected":
            if self.on_reject == "skip":
                return ExecutionResult(
                    execution_id=context.execution_id,
                    status=ExecutionStatus.COMPLETED,
                    output=context.input,
                    usage=context.usage_snapshot(),
                    metadata={"approval": approval_id, "decision": "rejected",
                              "skipped": True},
                )
            raise ApprovalRejected(f"approval '{approval_id}' was rejected",
                                   approval_id=approval_id, note=record.note)

        # approved -> run the guarded action (if any)
        if self.action is None:
            return ExecutionResult(
                execution_id=context.execution_id,
                status=ExecutionStatus.COMPLETED,
                output=context.input,
                usage=context.usage_snapshot(),
                metadata={"approval": approval_id, "decision": "approved"},
            )
        result = await self.action.execute(context)
        result.metadata.setdefault("approval", approval_id)
        result.metadata.setdefault("decision", "approved")
        return result


def _summarise(value: Any, limit: int = 500) -> Any:
    rendered = value if isinstance(value, (str, int, float, bool, type(None))) \
        else repr(value)
    if isinstance(rendered, str) and len(rendered) > limit:
        return rendered[:limit] + "..."
    return rendered
