"""Shared HTTP transport for provider adapters.

Adapters do not import ``httpx`` directly — they depend on this thin transport
so they can be unit-tested offline with :class:`MockTransport` and so retry,
timeout and error translation behave identically across providers.
"""

from __future__ import annotations

import json
from typing import Any, AsyncIterator, Callable

from ..core.exceptions import (
    ConfigurationError,
    ModelError,
    ModelUnavailable,
    RateLimitError,
)

__all__ = ["HttpTransport", "HttpxTransport", "MockTransport", "sse_lines",
           "translate_status"]


def translate_status(status: int, body: str, *, provider: str) -> Exception:
    """Map an HTTP status onto the framework error model."""
    detail = body[:400]
    if status == 429:
        return RateLimitError(f"{provider} rate limited", status=status,
                              body=detail)
    if status in (408, 500, 502, 503, 504, 529):
        return ModelUnavailable(f"{provider} is unavailable", status=status,
                                body=detail)
    if status in (401, 403):
        return ConfigurationError(f"{provider} rejected the credentials",
                                  status=status, body=detail)
    return ModelError(f"{provider} request failed", status=status, body=detail)


class HttpTransport:
    """Minimal async HTTP interface an adapter needs."""

    async def post_json(self, url: str, *, headers: dict[str, str],
                        payload: dict[str, Any],
                        timeout: float | None = None) -> dict[str, Any]:
        raise NotImplementedError

    async def post_stream(self, url: str, *, headers: dict[str, str],
                          payload: dict[str, Any],
                          timeout: float | None = None) -> AsyncIterator[str]:
        raise NotImplementedError
        yield ""  # pragma: no cover - typing only

    async def aclose(self) -> None:
        return None


class HttpxTransport(HttpTransport):
    """Production transport backed by ``httpx`` (optional dependency)."""

    def __init__(self, *, provider: str = "provider", client: Any = None,
                 timeout: float = 60.0) -> None:
        self.provider = provider
        self._timeout = timeout
        if client is not None:
            self._client = client
            return
        try:
            import httpx
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise ConfigurationError(
                "httpx is required for HTTP providers: "
                'pip install "coflowai[http]"', cause=exc,
            ) from exc
        self._client = httpx.AsyncClient(timeout=timeout)

    async def post_json(self, url: str, *, headers: dict[str, str],
                        payload: dict[str, Any],
                        timeout: float | None = None) -> dict[str, Any]:
        response = await self._client.post(url, headers=headers, json=payload,
                                           timeout=timeout or self._timeout)
        if response.status_code >= 400:
            raise translate_status(response.status_code, response.text,
                                   provider=self.provider)
        return response.json()

    async def post_stream(self, url: str, *, headers: dict[str, str],
                          payload: dict[str, Any],
                          timeout: float | None = None) -> AsyncIterator[str]:
        async with self._client.stream("POST", url, headers=headers, json=payload,
                                       timeout=timeout or self._timeout) as response:
            if response.status_code >= 400:
                body = await response.aread()
                raise translate_status(response.status_code, body.decode("utf-8",
                                                                         "replace"),
                                       provider=self.provider)
            async for line in response.aiter_lines():
                yield line

    async def aclose(self) -> None:
        close = getattr(self._client, "aclose", None)
        if close is not None:
            await close()


class MockTransport(HttpTransport):
    """Offline transport for adapter unit tests.

    ``responses`` may be dicts (returned verbatim), exceptions (raised), or
    callables receiving the request payload.
    """

    def __init__(self, responses: list[Any] | None = None, *,
                 stream_lines: list[list[str]] | None = None,
                 handler: Callable[[dict[str, Any]], Any] | None = None) -> None:
        self.responses = list(responses or ())
        self.stream_lines = list(stream_lines or ())
        self.handler = handler
        self.requests: list[dict[str, Any]] = []
        self.headers: list[dict[str, str]] = []
        self.urls: list[str] = []

    async def post_json(self, url: str, *, headers: dict[str, str],
                        payload: dict[str, Any],
                        timeout: float | None = None) -> dict[str, Any]:
        self.requests.append(payload)
        self.headers.append(headers)
        self.urls.append(url)
        if self.handler is not None:
            result = self.handler(payload)
        elif self.responses:
            result = self.responses.pop(0)
        else:
            raise ModelError("MockTransport has no queued response")
        if isinstance(result, BaseException):
            raise result
        return result

    async def post_stream(self, url: str, *, headers: dict[str, str],
                          payload: dict[str, Any],
                          timeout: float | None = None) -> AsyncIterator[str]:
        self.requests.append(payload)
        self.headers.append(headers)
        self.urls.append(url)
        lines = self.stream_lines.pop(0) if self.stream_lines else []
        for line in lines:
            yield line


async def sse_lines(lines: AsyncIterator[str]) -> AsyncIterator[dict[str, Any]]:
    """Parse ``data:`` server-sent events into JSON payloads."""
    async for line in lines:
        if not line or not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if not data or data == "[DONE]":
            continue
        try:
            yield json.loads(data)
        except json.JSONDecodeError:  # pragma: no cover - provider noise
            continue
