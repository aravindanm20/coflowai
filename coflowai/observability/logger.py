"""Structured logging on top of the standard library.

If ``structlog`` is installed it is used; otherwise a JSON formatter over
``logging`` provides the same structured output with zero dependencies.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any

from .redaction import redact_mapping

__all__ = ["get_logger", "configure_logging", "StructuredLogger"]

_CONFIGURED = False
_RESERVED = set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__) | {
    "message", "asctime", "taskName",
}


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(
                record.created, tz=timezone.utc
            ).isoformat(),
            "level": record.levelname,
            "event": record.getMessage(),
            "logger": record.name,
        }
        extras = {k: v for k, v in record.__dict__.items() if k not in _RESERVED}
        payload.update(redact_mapping(extras))
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(level: int | str = logging.INFO, *, json_output: bool = True,
                      stream=None) -> None:
    """Install the CoFlowAi log handler.  Safe to call more than once."""
    global _CONFIGURED
    logger = logging.getLogger("coflowai")
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
    handler = logging.StreamHandler(stream or sys.stderr)
    handler.setFormatter(
        _JsonFormatter() if json_output
        else logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    )
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    _CONFIGURED = True


class StructuredLogger:
    """Thin wrapper giving ``log.info("event.name", key=value)`` semantics."""

    __slots__ = ("_logger", "_context")

    def __init__(self, name: str, context: dict[str, Any] | None = None) -> None:
        self._logger = logging.getLogger(f"coflowai.{name}" if name else "coflowai")
        self._context = context or {}

    def bind(self, **context: Any) -> "StructuredLogger":
        merged = {**self._context, **context}
        clone = StructuredLogger.__new__(StructuredLogger)
        object.__setattr__(clone, "_logger", self._logger)
        object.__setattr__(clone, "_context", merged)
        return clone

    def _log(self, level: int, event: str, **fields: Any) -> None:
        if not _CONFIGURED:
            configure_logging()
        self._logger.log(level, event, extra={**self._context, **fields})

    def debug(self, event: str, **fields: Any) -> None:
        self._log(logging.DEBUG, event, **fields)

    def info(self, event: str, **fields: Any) -> None:
        self._log(logging.INFO, event, **fields)

    def warning(self, event: str, **fields: Any) -> None:
        self._log(logging.WARNING, event, **fields)

    def error(self, event: str, **fields: Any) -> None:
        self._log(logging.ERROR, event, **fields)

    def exception(self, event: str, **fields: Any) -> None:
        if not _CONFIGURED:
            configure_logging()
        self._logger.exception(event, extra={**self._context, **fields})


def get_logger(name: str = "", **context: Any) -> StructuredLogger:
    return StructuredLogger(name, context or None)
