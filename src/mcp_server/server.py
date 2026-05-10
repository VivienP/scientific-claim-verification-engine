"""FastMCP server exposing the lite verification API to AI agents.

Tools:
    - verify_text       Submit + (optionally) wait for a verification job.
    - get_job_status    Poll a single time. Use after verify_text(wait=False).
    - get_health        Cheap reachability check (no auth path on API side).

Resources:
    - report://{run_id} Copilot HTML report. The agent can render it inline
                        or save it to disk.

Transport:
    Default ``stdio`` for Claude Desktop / Claude Agent SDK. The optional
    ``MCP_TRANSPORT`` env var lets you switch to ``streamable-http`` for
    remote agents (Phase D); see README for the wiring.

Why module-level tool functions:
    The tools are defined at module level so tests can call them directly
    without instantiating FastMCP. ``build_server`` registers them via the
    decorator API. This double-registration is intentional — FastMCP tool
    introspection (parameter schemas, names) requires the decorator pass,
    while unit tests want plain async functions.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Literal

import structlog
from mcp.server.fastmcp import FastMCP

from src.mcp_server.client import LiteApiClient
from src.mcp_server.config import McpConfig

logger: structlog.BoundLogger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Tool implementations — plain async functions, no MCP coupling.
# ---------------------------------------------------------------------------


async def verify_text(
    text: str,
    mode: Literal["v1", "copilot"] = "copilot",
    copilot_mode: Literal["pharma", "academic", "general"] = "pharma",
    wait: bool = True,
    timeout_seconds: float | None = None,
    *,
    client: LiteApiClient,
) -> dict[str, Any]:
    """Submit text for scientific claim verification.

    Returns a dict matching the lite API's ``JobStatusResponse`` envelope when
    ``wait=True`` (the job has reached a terminal state), or the
    ``JobCreated`` envelope ({job_id, status, poll_url}) when ``wait=False``.

    Args:
        text: Scientific text to verify (paper draft, regulatory paragraph,
            AI-generated summary). 1 to 2,000,000 chars.
        mode: ``"v1"`` for the plain claim list; ``"copilot"`` (default)
            adds the Phase B enrichment + HTML report.
        copilot_mode: Schema profile applied in Copilot mode.
        wait: When True (default), block until the job is completed/failed.
            When False, return immediately with the job id.
        timeout_seconds: Optional override on the wait deadline. Falls back
            to ``COPILOT_MCP_TIMEOUT`` (default 600s).
    """
    submitted = await client.submit_verification(
        text=text,
        mode=mode,
        copilot_mode=copilot_mode,
    )
    if not wait:
        return submitted
    job_id = submitted["job_id"]
    return await client.wait_for_job(job_id, timeout_seconds=timeout_seconds)


async def get_job_status(
    job_id: str,
    *,
    client: LiteApiClient,
) -> dict[str, Any]:
    """Return the current status envelope for a previously submitted job."""
    return await client.get_job(job_id)


async def get_health(
    *,
    client: LiteApiClient,
) -> dict[str, Any]:
    """Return the API health envelope. Useful before a long verify_text call."""
    return await client.health()


async def get_report_html(
    run_id: str,
    *,
    client: LiteApiClient,
) -> str:
    """Return the self-contained Copilot HTML report for a completed run."""
    return await client.get_report_html(run_id)


# ---------------------------------------------------------------------------
# Server factory — wires the FastMCP decorators against a shared client.
# ---------------------------------------------------------------------------


def build_server(config: McpConfig) -> FastMCP:
    """Construct a FastMCP instance bound to a fresh LiteApiClient.

    The client is owned by the server's lifecycle. Tests that need to
    inspect the registry (parameters, descriptions) should use this
    factory; tests that exercise the tool logic should call the module-
    level coroutines directly with their own client fixture.
    """
    mcp = FastMCP(
        name="scve-copilot",
        instructions=(
            "Scientific Claim Verification Engine. "
            "Submit scientific text via verify_text; poll via get_job_status; "
            "check reachability via get_health. Reports are accessible at "
            "report://{run_id} once a Copilot job completes."
        ),
    )
    client = LiteApiClient(config)

    @mcp.tool(
        name="verify_text",
        description=(
            "Verify scientific claims in a block of text against their cited "
            "sources. Returns a per-claim verdict (supported / unsupported / "
            "not_addressed / no_passage_found) with retrieval evidence and, "
            "in copilot mode, a recommended fix and HTML report."
        ),
    )
    async def _verify_text_tool(
        text: str,
        mode: Literal["v1", "copilot"] = "copilot",
        copilot_mode: Literal["pharma", "academic", "general"] = "pharma",
        wait: bool = True,
        timeout_seconds: float | None = None,
    ) -> dict[str, Any]:
        return await verify_text(
            text=text,
            mode=mode,
            copilot_mode=copilot_mode,
            wait=wait,
            timeout_seconds=timeout_seconds,
            client=client,
        )

    @mcp.tool(
        name="get_job_status",
        description="Poll a previously submitted verification job by id.",
    )
    async def _get_job_status_tool(job_id: str) -> dict[str, Any]:
        return await get_job_status(job_id=job_id, client=client)

    @mcp.tool(
        name="get_health",
        description="Check that the underlying verification API is reachable.",
    )
    async def _get_health_tool() -> dict[str, Any]:
        return await get_health(client=client)

    @mcp.resource(
        uri="report://{run_id}",
        name="copilot_report",
        description="The HTML report generated by a completed Copilot-mode run.",
        mime_type="text/html",
    )
    async def _report_resource(run_id: str) -> str:
        return await get_report_html(run_id=run_id, client=client)

    return mcp


# ---------------------------------------------------------------------------
# Entrypoint — installed as the `copilot-mcp` console script.
# ---------------------------------------------------------------------------


def _configure_logging_for_stdio() -> None:
    """Route structlog output to stderr — only call on the real stdio path.

    Critical for stdio transport: stdout is the JSON-RPC channel, and any
    log line landing on stdout corrupts the MCP frame stream. We
    deliberately do NOT call this from the dry-run path or from tests,
    because ``structlog.configure`` is process-wide and would leak into
    sibling tests in the same pytest session.
    """
    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.add_log_level,
            structlog.processors.JSONRenderer(),
        ],
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
    )


def _bootstrap_error(message: str) -> None:
    """Write a CLI bootstrap error to stderr.

    Used for config / transport errors that occur before structlog has
    been configured for stdio. Direct ``sys.stderr.write`` rather than a
    logger call because configuring structlog would pollute the global
    state used by sibling tests / the rest of the process.
    """
    sys.stderr.write(f"copilot-mcp: {message}\n")


def main() -> int:
    """Build the server and run on stdio.

    Returns:
        0 on a clean dry-run / shutdown, 2 on configuration error. Real
        stdio loop never returns under normal operation.
    """
    try:
        config = McpConfig.from_env()
    except ValueError as exc:
        _bootstrap_error(f"configuration error: {exc}")
        return 2

    server = build_server(config)

    # Dry-run path for unit tests / CI smoke checks. Avoids spawning the
    # stdio loop, which would block forever waiting for a JSON-RPC peer.
    if os.environ.get("COPILOT_MCP_DRY_RUN"):
        logger.info("mcp_dry_run", api_base_url=config.api_base_url)
        return 0

    transport_env = os.environ.get("MCP_TRANSPORT", "stdio")
    if transport_env not in ("stdio", "sse", "streamable-http"):
        _bootstrap_error(f"unknown MCP_TRANSPORT={transport_env!r}")
        return 2

    transport: Literal["stdio", "sse", "streamable-http"] = transport_env  # type: ignore[assignment]
    _configure_logging_for_stdio()
    logger.info("mcp_starting", transport=transport, api_base_url=config.api_base_url)
    server.run(transport=transport)
    return 0


__all__ = [
    "build_server",
    "get_health",
    "get_job_status",
    "get_report_html",
    "main",
    "verify_text",
]
