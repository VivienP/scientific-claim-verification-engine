"""Async HTTP client wrapping the lite API.

This client is the only network seam in the MCP wrapper. The MCP tool
functions in ``server.py`` consume it; tests mock its underlying httpx
calls with ``pytest-httpx``.

Error mapping (single source of truth — keep in sync with tests):
    - ConnectError / TransportError  -> LiteApiClientError("Cannot reach API ...")
    - 401 / 403                       -> LiteApiClientError("Auth failed ...")
    - 404                             -> ValueError("... not found")
    - 422                             -> ValueError("Invalid input: ...")
    - 503                             -> LiteApiClientError("API unavailable ...")
    - any other 4xx/5xx               -> LiteApiClientError("HTTP {code}: ...")
    - polling exceeds timeout         -> LiteApiTimeoutError(...)

We deliberately surface ``ValueError`` for caller-actionable issues
(bad input, missing run) and ``LiteApiClientError`` for infrastructure
issues. This separation lets the MCP layer translate cleanly into
agent-facing errors.
"""

from __future__ import annotations

import asyncio
import re
import time
from types import TracebackType
from typing import Any, Literal, Self

import httpx
import structlog

from src.mcp_server.config import McpConfig

logger: structlog.BoundLogger = structlog.get_logger(__name__)


class LiteApiClientError(RuntimeError):
    """Raised for infrastructure errors talking to the lite API."""


class LiteApiTimeoutError(LiteApiClientError):
    """Raised when ``wait_for_job`` exceeds the configured timeout."""


_TerminalStatus = Literal["completed", "failed"]

# Identifiers received from MCP agents (job_id, run_id) are interpolated
# into outbound URL paths. The API enforces path-traversal defence on
# /runs/{run_id}/copilot_report.html (see src/api/app.py:172-185) and
# rejects malformed ids, but defending here as well keeps the MCP layer
# robust against an API regression. Pattern allows base64/hex/uuid/dash
# id shapes and explicitly excludes /, ., %, \ — anything a path traversal
# payload would need.
_SAFE_ID_PATTERN: Literal[r"^[A-Za-z0-9_\-]+$"] = r"^[A-Za-z0-9_\-]+$"


def _validate_id(name: str, value: str) -> None:
    """Raise ValueError if an id contains anything beyond [A-Za-z0-9_-]."""
    if not re.match(_SAFE_ID_PATTERN, value):
        raise ValueError(f"Invalid {name}: must match {_SAFE_ID_PATTERN} (got {value!r})")


# NOTE on `dict[str, Any]` returns: the lite API responses are typed
# server-side via Pydantic models in `src/api/models.py`. Re-deriving
# matching TypedDicts here would couple the MCP client to that schema and
# require updates on every API field addition. The wrapper deliberately
# stays loose on shape so that adding a non-breaking field (e.g. extra
# diagnostic counters) does not require a parallel edit. Callers that need
# typed access import the Pydantic models directly.


class LiteApiClient:
    """Async wrapper over the lite API.

    Designed to be long-lived — one client per MCP server process. The
    underlying ``httpx.AsyncClient`` keeps a connection pool alive across
    calls, which matters when an agent submits N jobs in quick succession.
    """

    def __init__(self, config: McpConfig) -> None:
        self._config = config
        # Single shared client. Auth header is added per-request rather than
        # baked in so /health (no auth) doesn't leak the key to log scrapers.
        self._http = httpx.AsyncClient(
            base_url=config.api_base_url,
            timeout=httpx.Timeout(config.request_timeout_seconds),
        )
        self._closed = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        """Idempotent close. Safe to call twice."""
        if self._closed:
            return
        self._closed = True
        await self._http.aclose()

    # ------------------------------------------------------------------
    # Endpoints
    # ------------------------------------------------------------------

    async def health(self) -> dict[str, Any]:
        """GET /health — no auth, used to confirm reachability."""
        try:
            r = await self._http.get("/health")
        except httpx.RequestError as exc:
            raise LiteApiClientError(
                f"Cannot reach API at {self._config.api_base_url}: {exc!s}"
            ) from exc
        self._raise_for_status(r)
        body: dict[str, Any] = r.json()
        return body

    async def submit_verification(
        self,
        *,
        text: str,
        mode: Literal["v1", "copilot"] = "copilot",
        copilot_mode: Literal["pharma", "academic", "general"] = "pharma",
        enable_primary_lookup: bool = True,
        enable_recommended_fix: bool = True,
    ) -> dict[str, Any]:
        """POST /verify — returns ``{job_id, status, poll_url}``.

        Mirrors ``VerifyRequest`` in ``src/api/models.py``. Any future
        request fields go here first, then in ``server.verify_text``.
        """
        payload = {
            "text": text,
            "mode": mode,
            "copilot_mode": copilot_mode,
            "enable_primary_lookup": enable_primary_lookup,
            "enable_recommended_fix": enable_recommended_fix,
        }
        try:
            r = await self._http.post("/verify", json=payload, headers=self._auth_headers())
        except httpx.RequestError as exc:
            raise LiteApiClientError(
                f"Cannot reach API at {self._config.api_base_url}: {exc!s}"
            ) from exc
        self._raise_for_status(r)
        body: dict[str, Any] = r.json()
        return body

    async def get_job(self, job_id: str) -> dict[str, Any]:
        """GET /jobs/{job_id} — single poll."""
        _validate_id("job_id", job_id)
        try:
            r = await self._http.get(f"/jobs/{job_id}", headers=self._auth_headers())
        except httpx.RequestError as exc:
            raise LiteApiClientError(
                f"Cannot reach API at {self._config.api_base_url}: {exc!s}"
            ) from exc
        self._raise_for_status(r)
        body: dict[str, Any] = r.json()
        return body

    async def wait_for_job(
        self,
        job_id: str,
        *,
        timeout_seconds: float | None = None,
        poll_interval_seconds: float | None = None,
    ) -> dict[str, Any]:
        """Poll ``/jobs/{job_id}`` until terminal or timeout.

        ``timeout_seconds`` and ``poll_interval_seconds`` override the
        config defaults so a per-call short-circuit (e.g. an agent with
        an interactive deadline) does not require a new client.
        """
        deadline = time.monotonic() + (timeout_seconds or self._config.timeout_seconds)
        interval = poll_interval_seconds or self._config.poll_interval_seconds
        while True:
            envelope = await self.get_job(job_id)
            status: str = envelope["status"]
            if status in ("completed", "failed"):
                return envelope
            if time.monotonic() >= deadline:
                limit = timeout_seconds or self._config.timeout_seconds
                raise LiteApiTimeoutError(f"Job {job_id} did not complete within {limit:.0f}s")
            await asyncio.sleep(interval)

    async def get_report_html(self, run_id: str) -> str:
        """GET /runs/{run_id}/copilot_report.html — returns HTML body."""
        _validate_id("run_id", run_id)
        try:
            r = await self._http.get(
                f"/runs/{run_id}/copilot_report.html",
                headers=self._auth_headers(),
            )
        except httpx.RequestError as exc:
            raise LiteApiClientError(
                f"Cannot reach API at {self._config.api_base_url}: {exc!s}"
            ) from exc
        self._raise_for_status(r)
        return r.text

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _auth_headers(self) -> dict[str, str]:
        return {"X-API-Key": self._config.api_key}

    @staticmethod
    def _raise_for_status(response: httpx.Response) -> None:
        """Translate an httpx response into our typed exception hierarchy.

        We deliberately do NOT call ``response.raise_for_status()`` — its
        ``HTTPStatusError`` is not specific enough. Distinct error classes
        let MCP tool wrappers map to MCP-friendly messages.
        """
        if response.is_success:
            return
        # Best-effort detail extraction; pharma-grade APIs occasionally
        # return non-JSON bodies (e.g. 502 from a misbehaving proxy).
        detail = LiteApiClient._extract_detail(response)
        code = response.status_code
        if code in (401, 403):
            raise LiteApiClientError(f"Auth failed ({code}): {detail}")
        if code == 404:
            raise ValueError(f"Resource not found: {detail}")
        if code in (400, 422):
            # 400 (e.g. path-traversal rejection) and 422 (Pydantic validation)
            # are both "the caller passed something bad" — distinct from
            # infrastructure failures.
            raise ValueError(f"Invalid input: {detail}")
        if code == 503:
            raise LiteApiClientError(f"API unavailable (503): {detail}")
        raise LiteApiClientError(f"HTTP {code}: {detail}")

    @staticmethod
    def _extract_detail(response: httpx.Response) -> str:
        """Pull a human-readable message out of an error response."""
        try:
            payload = response.json()
        except ValueError:
            # Non-JSON error body (e.g. a misbehaving proxy returning HTML).
            # Log so we notice when an upstream stops speaking JSON; surface
            # a truncated body so callers still get something actionable.
            logger.warning(
                "lite_api_non_json_error_body",
                status_code=response.status_code,
                body_prefix=response.text[:80],
            )
            return response.text[:200] or "no detail"
        if isinstance(payload, dict):
            for key in ("detail", "error", "message"):
                value = payload.get(key)
                if isinstance(value, str):
                    return value
                if isinstance(value, list) and value:
                    first = value[0]
                    if isinstance(first, dict) and isinstance(first.get("msg"), str):
                        return str(first["msg"])
        return str(payload)[:200]


__all__ = [
    "LiteApiClient",
    "LiteApiClientError",
    "LiteApiTimeoutError",
]
