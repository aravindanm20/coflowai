"""Secret redaction.

Nothing sensitive is ever logged by default.  Redaction is applied to log
payloads, event payloads written by the runtime, and trace attributes.
"""

from __future__ import annotations

import re
from typing import Any, Iterable

__all__ = ["redact", "redact_mapping", "add_redaction_key", "add_redaction_pattern",
           "REDACTED"]

REDACTED = "***redacted***"

_SENSITIVE_KEYS: set[str] = {
    "api_key", "apikey", "authorization", "auth", "password", "passwd", "secret",
    "access_token", "refresh_token", "token", "client_secret", "private_key",
    "session_key", "credentials", "connection_string", "dsn", "cookie",
    "x-api-key", "bearer",
}

_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"sk-[A-Za-z0-9]{16,}"),                       # OpenAI-style keys
    re.compile(r"Bearer\s+[A-Za-z0-9._\-]{12,}"),             # bearer tokens
    re.compile(r"[A-Za-z0-9+/]{40,}={0,2}"),                  # long base64 blobs
    re.compile(r"(?i)(postgres|mysql|redis|mongodb)://[^\s\"']+"),  # DSNs
]


def add_redaction_key(*keys: str) -> None:
    """Register additional field names that must never be logged."""
    _SENSITIVE_KEYS.update(k.lower() for k in keys)


def add_redaction_pattern(*patterns: str | re.Pattern[str]) -> None:
    for pattern in patterns:
        _PATTERNS.append(re.compile(pattern) if isinstance(pattern, str) else pattern)


def _is_sensitive(key: str) -> bool:
    lowered = key.lower()
    return lowered in _SENSITIVE_KEYS or any(s in lowered for s in _SENSITIVE_KEYS)


def redact(value: Any, *, max_len: int = 2000) -> Any:
    """Recursively redact secrets from an arbitrary value."""
    if isinstance(value, str):
        redacted = value
        for pattern in _PATTERNS:
            redacted = pattern.sub(REDACTED, redacted)
        if len(redacted) > max_len:
            redacted = redacted[:max_len] + f"...<truncated {len(redacted) - max_len}>"
        return redacted
    if isinstance(value, dict):
        return redact_mapping(value, max_len=max_len)
    if isinstance(value, (list, tuple, set)):
        return [redact(item, max_len=max_len) for item in value]
    return value


def redact_mapping(mapping: dict[Any, Any], *, max_len: int = 2000) -> dict[Any, Any]:
    out: dict[Any, Any] = {}
    for key, value in mapping.items():
        if isinstance(key, str) and _is_sensitive(key):
            out[key] = REDACTED
        else:
            out[key] = redact(value, max_len=max_len)
    return out


def sensitive_keys() -> Iterable[str]:  # pragma: no cover - introspection helper
    return sorted(_SENSITIVE_KEYS)
