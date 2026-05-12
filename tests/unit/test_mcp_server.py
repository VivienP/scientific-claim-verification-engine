"""Unit tests for src/mcp_server/server.py — MCP tool functions, offline.

These tests call the underlying tool callables directly. The FastMCP
decorator wraps them but exposes the original function via the registry,
so we can invoke them as plain async functions without spinning up the
JSON-RPC transport.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import pytest
from pytest_httpx import HTTPXMock

from src.mcp_server.client import LiteApiClient, LiteApiClientError
from src.mcp_server.config import McpConfig
from src.mcp_server.server import (
    build_server,
    get_health,
    get_job_status,
    get_report_html,
    verify_text,
)

_BASE = "http://api.test"
_KEY = "test-key"


def _config(**overrides: object) -> McpConfig:
    base = {
        "api_base_url": _BASE,
        "api_key": _KEY,
        "poll_interval_seconds": 0.001,
        "timeout_seconds": 5.0,
        "request_timeout_seconds": 2.0,
    }
    base.update(overrides)
    return McpConfig(**base)  # type: ignore[arg-type]


@pytest.fixture
async def client() -> AsyncIterator[LiteApiClient]:
    """Yield a freshly-created LiteApiClient bound to the test config."""
    c = LiteApiClient(_config())
    try:
        yield c
    finally:
        await c.aclose()


# ---------------------------------------------------------------------------
# build_server — registry assertions
# ---------------------------------------------------------------------------


class TestBuildServer:
    def test_returns_fastmcp_with_expected_tools(self) -> None:
        server = build_server(_config())
        # FastMCP exposes registered tools via the underlying tool manager.
        names = {tool.name for tool in server._tool_manager.list_tools()}
        assert {"verify_text", "get_job_status", "get_health"} <= names

    def test_resource_template_registered(self) -> None:
        server = build_server(_config())
        templates = list(server._resource_manager.list_templates())
        # report://{run_id} should be exposed as a resource template.
        uri_templates = [t.uri_template for t in templates]
        assert any("report://" in u and "{run_id}" in u for u in uri_templates)


# ---------------------------------------------------------------------------
# verify_text — happy paths
# ---------------------------------------------------------------------------


class TestVerifyText:
    @pytest.mark.asyncio
    async def test_wait_true_polls_until_completed(
        self, client: LiteApiClient, httpx_mock: HTTPXMock
    ) -> None:
        httpx_mock.add_response(
            url=f"{_BASE}/verify",
            method="POST",
            status_code=202,
            json={"job_id": "abc", "status": "pending", "poll_url": f"{_BASE}/jobs/abc"},
        )
        httpx_mock.add_response(
            url=f"{_BASE}/jobs/abc",
            json={
                "job_id": "abc",
                "status": "completed",
                "created_at": 1.0,
                "updated_at": 2.0,
                "run_id": "api-abc",
                "error": None,
                "result": {
                    "n_claims": 2,
                    "verdict_counts": {"supported": 2},
                    "report_html_url": "/runs/api-abc/copilot_report.html",
                },
            },
        )
        out = await verify_text(
            text="claim text [1]",
            mode="copilot",
            copilot_mode="pharma",
            wait=True,
            client=client,
        )
        assert out["status"] == "completed"
        assert out["job_id"] == "abc"
        assert out["run_id"] == "api-abc"
        assert out["result"]["n_claims"] == 2

    @pytest.mark.asyncio
    async def test_wait_false_returns_immediately(
        self, client: LiteApiClient, httpx_mock: HTTPXMock
    ) -> None:
        httpx_mock.add_response(
            url=f"{_BASE}/verify",
            method="POST",
            status_code=202,
            json={"job_id": "abc", "status": "pending", "poll_url": f"{_BASE}/jobs/abc"},
        )
        out = await verify_text(
            text="x", mode="v1", copilot_mode="pharma", wait=False, client=client
        )
        assert out["status"] == "pending"
        assert out["job_id"] == "abc"
        # No polling should have happened — only the POST.
        requests = httpx_mock.get_requests()
        assert len(requests) == 1
        assert requests[0].method == "POST"

    @pytest.mark.asyncio
    async def test_failed_job_surfaces_error_in_envelope(
        self, client: LiteApiClient, httpx_mock: HTTPXMock
    ) -> None:
        httpx_mock.add_response(
            url=f"{_BASE}/verify",
            method="POST",
            status_code=202,
            json={"job_id": "abc", "status": "pending", "poll_url": f"{_BASE}/jobs/abc"},
        )
        httpx_mock.add_response(
            url=f"{_BASE}/jobs/abc",
            json={
                "job_id": "abc",
                "status": "failed",
                "created_at": 1.0,
                "updated_at": 2.0,
                "run_id": "api-abc",
                "error": "RuntimeError: pipeline boom",
                "result": None,
            },
        )
        out = await verify_text(
            text="x", mode="v1", copilot_mode="pharma", wait=True, client=client
        )
        # Failure is returned as data, not raised — the agent decides what to do.
        assert out["status"] == "failed"
        assert "boom" in out["error"]


# ---------------------------------------------------------------------------
# verify_text — error paths
# ---------------------------------------------------------------------------


class TestVerifyTextErrors:
    @pytest.mark.asyncio
    async def test_validation_error_raises_value_error(
        self, client: LiteApiClient, httpx_mock: HTTPXMock
    ) -> None:
        httpx_mock.add_response(
            url=f"{_BASE}/verify",
            method="POST",
            status_code=422,
            json={"detail": "text required"},
        )
        with pytest.raises(ValueError):
            await verify_text(text="", mode="v1", copilot_mode="pharma", wait=False, client=client)

    @pytest.mark.asyncio
    async def test_auth_error_propagates_typed_exception(
        self, client: LiteApiClient, httpx_mock: HTTPXMock
    ) -> None:
        httpx_mock.add_response(
            url=f"{_BASE}/verify",
            method="POST",
            status_code=403,
            json={"detail": "Invalid API key."},
        )
        with pytest.raises(LiteApiClientError, match="Auth failed"):
            await verify_text(text="x", mode="v1", copilot_mode="pharma", wait=False, client=client)


# ---------------------------------------------------------------------------
# get_job_status / get_health
# ---------------------------------------------------------------------------


class TestStatusAndHealth:
    @pytest.mark.asyncio
    async def test_get_job_status(self, client: LiteApiClient, httpx_mock: HTTPXMock) -> None:
        httpx_mock.add_response(
            url=f"{_BASE}/jobs/abc",
            json={
                "job_id": "abc",
                "status": "running",
                "created_at": 1.0,
                "updated_at": 2.0,
                "run_id": None,
                "error": None,
                "result": None,
            },
        )
        out = await get_job_status(job_id="abc", client=client)
        assert out["status"] == "running"

    @pytest.mark.asyncio
    async def test_get_health(self, client: LiteApiClient, httpx_mock: HTTPXMock) -> None:
        httpx_mock.add_response(
            url=f"{_BASE}/health",
            json={"status": "ok", "version": "0.1.0", "n_jobs_active": 7},
        )
        out = await get_health(client=client)
        assert out["status"] == "ok"
        assert out["n_jobs_active"] == 7


# ---------------------------------------------------------------------------
# Resource: report://{run_id}
# ---------------------------------------------------------------------------


class TestReportResource:
    @pytest.mark.asyncio
    async def test_returns_html_text(self, client: LiteApiClient, httpx_mock: HTTPXMock) -> None:
        httpx_mock.add_response(
            url=f"{_BASE}/runs/api-xyz/copilot_report.html",
            text="<html>r</html>",
            headers={"content-type": "text/html"},
        )
        html = await get_report_html(run_id="api-xyz", client=client)
        assert html.startswith("<html>")

    @pytest.mark.asyncio
    async def test_missing_run_raises(self, client: LiteApiClient, httpx_mock: HTTPXMock) -> None:
        httpx_mock.add_response(
            url=f"{_BASE}/runs/missing/copilot_report.html",
            status_code=404,
            json={"error": "not_found", "detail": "Report not found"},
        )
        with pytest.raises(ValueError, match="not found"):
            await get_report_html(run_id="missing", client=client)


# ---------------------------------------------------------------------------
# main() entrypoint smoke test
# ---------------------------------------------------------------------------


class TestMain:
    def test_main_does_not_raise_when_called_with_dry_run(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The dry-run path builds the server and exits — no stdio loop entered.

        We don't want to spawn a real stdio server in unit tests; the dry-run
        flag (env var) lets us assert that build_server() + main() wiring is
        correct without invoking ``mcp.run()``.
        """
        monkeypatch.setenv("VERIFIER_API_BASE_URL", _BASE)
        monkeypatch.setenv("VERIFIER_API_KEY", _KEY)
        monkeypatch.setenv("VERIFIER_MCP_DRY_RUN", "1")
        from src.mcp_server.server import main

        # Should return cleanly (rc=0) without entering stdio.
        rc = main()
        assert rc == 0

    def test_main_fails_fast_when_api_key_missing(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.delenv("VERIFIER_API_KEY", raising=False)
        monkeypatch.setenv("VERIFIER_API_BASE_URL", _BASE)
        monkeypatch.setenv("VERIFIER_MCP_DRY_RUN", "1")
        from src.mcp_server.server import main

        rc = main()
        assert rc == 2
        captured = capsys.readouterr()
        assert "VERIFIER_API_KEY" in captured.err


# ---------------------------------------------------------------------------
# Schema sanity
# ---------------------------------------------------------------------------


class TestToolSchemas:
    def test_verify_text_schema_has_required_fields(self) -> None:
        server = build_server(_config())
        tools = {t.name: t for t in server._tool_manager.list_tools()}
        schema: dict[str, Any] = tools["verify_text"].parameters
        props = schema["properties"]
        # Surface contract — these fields must be advertised to the agent.
        for field in ("text", "mode", "copilot_mode", "wait"):
            assert field in props, f"verify_text schema missing {field}"
        # text is the only truly required input.
        assert "text" in schema.get("required", [])
