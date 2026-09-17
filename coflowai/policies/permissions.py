"""Namespaced permission model.

Permissions are dotted strings (``database.read``).  Wildcards are supported at
segment level: ``database.*`` grants every ``database.<x>`` permission, ``*``
grants everything (intended for local development only).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

__all__ = ["PermissionSet", "Permission"]

Permission = str


@dataclass(frozen=True, slots=True)
class PermissionSet:
    granted: frozenset[str]

    def __init__(self, granted: Iterable[str] | None = None) -> None:
        object.__setattr__(self, "granted", frozenset(granted or ()))

    # ------------------------------------------------------------------ query
    def allows(self, required: str | None) -> bool:
        """A tool without a declared permission is unrestricted."""
        if not required:
            return True
        if "*" in self.granted or required in self.granted:
            return True
        segments = required.split(".")
        for i in range(len(segments), 0, -1):
            prefix = ".".join(segments[:i - 1] + ["*"]) if i > 1 else "*"
            if prefix in self.granted:
                return True
        return False

    def missing(self, required: Iterable[str | None]) -> list[str]:
        return [r for r in required if r and not self.allows(r)]

    # ---------------------------------------------------------------- algebra
    def union(self, other: "PermissionSet | Iterable[str]") -> "PermissionSet":
        extra = other.granted if isinstance(other, PermissionSet) else frozenset(other)
        return PermissionSet(self.granted | extra)

    def intersect(self, other: "PermissionSet") -> "PermissionSet":
        """Least privilege: a child never gets more than its parent."""
        if "*" in self.granted:
            return other
        if "*" in other.granted:
            return self
        return PermissionSet(self.granted & other.granted)

    def __iter__(self):
        return iter(sorted(self.granted))

    def __bool__(self) -> bool:
        return bool(self.granted)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"PermissionSet({sorted(self.granted)!r})"

    @classmethod
    def all(cls) -> "PermissionSet":
        return cls({"*"})
