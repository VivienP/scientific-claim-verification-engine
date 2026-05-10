"""Unit tests for src/mcp_server/client.py — fully offline (pytest-httpx mocks).

The MCP client is a thin async wrapper around the lite API. We test its
behaviour against the four endpoints the server exposes: /health, /verify,
/jobs/{id}, /runs/{id}/copilot_report.html — plus the polling loop with
explicit timeout/interval semantics.
"""

from __future__ import annotations

import asyncio

import httpx
import pytest
from pytest_httpx import HTTPXMock

from src.mcp_server.client import (
    LiteApiClient,
    LiteApiClientError,
    LiteApiTimeoutError,
)
from src.mcp_server.config import McpConfig

_BASE = "http://api.test"
_KEY = "test-key"


def _config(**overrides: object) -> McpConfig:
    """Build a McpConfig with sane test defaults; overrides win."""
    base = {
        "api_base_url": _BASE,
        "api_key": _KEY,
        "poll_interval_seconds": 0.001,  # near-instant for tests
        "timeout_seconds": 5.0,
        "request_timeout_seconds": 2.0,
    }
    base.update(overrides)
    return McpConfig(**base)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# health
# ---------------------------------------------------------------------------


class TestHealth:
    @pytest.mark.asyncio
    async def test_returns_payload_on_200(self, httpx_mock: HTTPXMock) -> None:
        httpx_mock.add_response(
            url=f"{_BASE}/health",
            json={"status": "ok", "version": "0.1.0", "n_jobs_active": 0},
        )
        async with LiteApiClient(_config()) as client:
            r = await client.health()
        assert r["status"] == "ok"
        assert r["version"] == "0.1.0"

    @pytest.mark.asyncio
    async def test_does_not_send_api_key(self, httpx_mock: HTTPXMock) -> None:
        httpx_mock.add_response(
            url=f"{_BASE}/health",
            json={"status": "ok", "version": "0.1.0", "n_jobs_active": 0},
        )
        async with LiteApiClient(_config()) as client:
            await client.health()
        request = httpx_mock.get_request()
        assert request is not None
        # /health is unauthenticated; sending the key is wasteful + fingerprints us.
        assert "x-api-key" not in {k.lower() for k in request.headers}

    @pytest.mark.asyncio
    async def test_raises_on_unreachable(self, httpx_mock: HTTPXMock) -> None:
        httpx_mock.add_exception(httpx.ConnectError("boom"))
        async with LiteApiClient(_config()) as client:
            with pytest.raises(LiteApiClientError, match="reach API"):
                await client.health()


# ---------------------------------------------------------------------------
# submit_verification
# ---------------------------------------------------------------------------


class TestSubmit:
    @pytest.mark.asyncio
    async def test_posts_payload_with_api_key(self, httpx_mock: HTTPXMock) -> None:
        httpx_mock.add_response(
            url=f"{_BASE}/verify",
            method="POST",
            status_code=202,
            json={"job_id": "abc", "status": "pending", "poll_url": f"{_BASE}/jobs/abc"},
        )
        async with LiteApiClient(_config()) as client:
            r = await client.submit_verification(
                text="hello", mode="copilot", copilot_mode="pharma"
            )
        assert r["job_id"] == "abc"
        request = httpx_mock.get_request()
        assert request is not None
        assert request.headers["x-api-key"] == _KEY
        assert request.headers["content-type"].startswith("application/json")

    @pytest.mark.asyncio
    async def test_validation_error_raises_value_error(self, httpx_mock: HTTPXMock) -> None:
        httpx_mock.add_response(
            url=f"{_BASE}/verify",
            method="POST",
            status_code=422,
            json={"detail": [{"msg": "text must not be empty"}]},
        )
        async with LiteApiClient(_config()) as client:
            with pytest.raises(ValueError, match="Invalid input"):
                await client.submit_verification(text="", mode="v1")

    @pytest.mark.asyncio
    async def test_auth_error_raises_dedicated_message(self, httpx_mock: HTTPXMock) -> None:
        httpx_mock.add_response(
            url=f"{_BASE}/verify",
            method="POST",
            status_code=401,
            json={"detail": "Missing X-API-Key header."},
        )
        async with LiteApiClient(_config()) as client:
            with pytest.raises(LiteApiClientError, match="Auth failed"):
                await client.submit_verification(text="x", mode="v1")

    @pytest.mark.asyncio
    async def test_503_raises_unavailable(self, httpx_mock: HTTPXMock) -> None:
        httpx_mock.add_response(
            url=f"{_BASE}/verify",
            method="POST",
            status_code=503,
            json={"detail": "Server is not configured"},
        )
        async with LiteApiClient(_config()) as client:
            with pytest.raises(LiteApiClientError, match="API unavailable"):
                await client.submit_verification(text="x", mode="v1")


# ---------------------------------------------------------------------------
# get_job
# ---------------------------------------------------------------------------


class TestGetJob:
    @pytest.mark.asyncio
    async def test_returns_status_envelope(self, httpx_mock: HTTPXMock) -> None:
        httpx_mock.add_response(
            url=f"{_BASE}/jobs/abc",
            json={
                "job_id": "abc",
                "status": "completed",
                "created_at": 1.0,
                "updated_at": 2.0,
                "run_id": "api-abc",
                "error": None,
                "result": {"n_claims": 3, "verdict_counts": {"supported": 3}},
            },
        )
        async with LiteApiClient(_config()) as client:
            r = await client.get_job("abc")
        assert r["status"] == "completed"
        assert r["result"]["n_claims"] == 3

    @pytest.mark.asyncio
    async def test_404_raises_value_error(self, httpx_mock: HTTPXMock) -> None:
        httpx_mock.add_response(
            url=f"{_BASE}/jobs/missing",
            status_code=404,
            json={"error": "not_found", "detail": "Job missing not found."},
        )
        async with LiteApiClient(_config()) as client:
            with pytest.raises(ValueError, match="not found"):
                await client.get_job("missing")


# ---------------------------------------------------------------------------
# wait_for_job
# ---------------------------------------------------------------------------


class TestWaitForJob:
    @pytest.mark.asyncio
    async def test_returns_when_completed(self, httpx_mock: HTTPXMock) -> None:
        # Two pending polls, then completed.
        for status in ("pending", "running", "completed"):
            httpx_mock.add_response(
                url=f"{_BASE}/jobs/abc",
                json={
                    "job_id": "abc",
                    "status": status,
                    "created_at": 1.0,
                    "updated_at": 2.0,
                    "run_id": "api-abc" if status == "completed" else None,
                    "error": None,
                    "result": {"n_claims": 1} if status == "completed" else None,
                },
            )
        async with LiteApiClient(_config()) as client:
            r = await client.wait_for_job("abc")
        assert r["status"] == "completed"
        assert r["run_id"] == "api-abc"

    @pytest.mark.asyncio
    async def test_returns_when_failed(self, httpx_mock: HTTPXMock) -> None:
        httpx_mock.add_response(
            url=f"{_BASE}/jobs/abc",
            json={
                "job_id": "abc",
                "status": "failed",
                "created_at": 1.0,
                "updated_at": 2.0,
                "run_id": "api-abc",
                "error": "boom",
                "result": None,
            },
        )
        async with LiteApiClient(_config()) as client:
            r = await client.wait_for_job("abc")
        assert r["status"] == "failed"
        assert r["error"] == "boom"

    @pytest.mark.asyncio
    async def test_raises_on_timeout(self, httpx_mock: HTTPXMock) -> None:
        # Always return pending — wait_for_job should give up.
        httpx_mock.add_response(
            url=f"{_BASE}/jobs/abc",
            is_reusable=True,
            json={
                "job_id": "abc",
                "status": "pending",
                "created_at": 1.0,
                "updated_at": 2.0,
                "run_id": None,
                "error": None,
                "result": None,
            },
        )
        cfg = _config(timeout_seconds=0.05, poll_interval_seconds=0.01)
        async with LiteApiClient(cfg) as client:
            with pytest.raises(LiteApiTimeoutError):
                await client.wait_for_job("abc")


# ---------------------------------------------------------------------------
# get_report_html
# ---------------------------------------------------------------------------


class TestGetReportHtml:
    @pytest.mark.asyncio
    async def test_returns_html_text(self, httpx_mock: HTTPXMock) -> None:
        httpx_mock.add_response(
            url=f"{_BASE}/runs/api-xyz/copilot_report.html",
            text="<html>report</html>",
            headers={"content-type": "text/html"},
        )
        async with LiteApiClient(_config()) as client:
            html = await client.get_report_html("api-xyz")
        assert "<html>" in html

    @pytest.mark.asyncio
    async def test_404_raises_value_error(self, httpx_mock: HTTPXMock) -> None:
        httpx_mock.add_response(
            url=f"{_BASE}/runs/missing/copilot_report.html",
            status_code=404,
            json={"error": "not_found", "detail": "Report not found"},
        )
        async with LiteApiClient(_config()) as client:
            with pytest.raises(ValueError, match="not found"):
                await client.get_report_html("missing")

    @pytest.mark.asyncio
    async def test_path_traversal_rejected_pre_send(self) -> None:
        # Pre-send validation rejects ids containing /, ., %, \, etc.
        # The httpx layer is never reached, so no mock is registered.
        async with LiteApiClient(_config()) as client:
            with pytest.raises(ValueError, match="Invalid run_id"):
                await client.get_report_html("..%2Fsecret")

    @pytest.mark.asyncio
    async def test_invalid_job_id_rejected_pre_send(self) -> None:
        async with LiteApiClient(_config()) as client:
            with pytest.raises(ValueError, match="Invalid job_id"):
                await client.get_job("../../etc/passwd")


# ---------------------------------------------------------------------------
# Context manager hygiene
# ---------------------------------------------------------------------------


class TestLifecycle:
    @pytest.mark.asyncio
    async def test_can_be_used_without_async_with(self, httpx_mock: HTTPXMock) -> None:
        """Allow callers to manage the lifecycle manually for long-lived MCP servers."""
        httpx_mock.add_response(
            url=f"{_BASE}/health",
            json={"status": "ok", "version": "0.1.0", "n_jobs_active": 0},
        )
        client = LiteApiClient(_config())
        try:
            r = await client.health()
            assert r["status"] == "ok"
        finally:
            await client.aclose()

    @pytest.mark.asyncio
    async def test_double_aclose_is_safe(self, httpx_mock: HTTPXMock) -> None:
        client = LiteApiClient(_config())
        await client.aclose()
        await client.aclose()  # second close must not raise

    @pytest.mark.asyncio
    async def test_concurrent_requests_share_one_client(self, httpx_mock: HTTPXMock) -> None:
        """One underlying httpx.AsyncClient — connection pool reuse."""
        httpx_mock.add_response(
            url=f"{_BASE}/health",
            is_reusable=True,
            json={"status": "ok", "version": "0.1.0", "n_jobs_active": 0},
        )
        async with LiteApiClient(_config()) as client:
            results = await asyncio.gather(*(client.health() for _ in range(5)))
        assert len(results) == 5
        assert all(r["status"] == "ok" for r in results)
