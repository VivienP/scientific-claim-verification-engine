"""MCP wrapper exposing the lite verification API to AI agents.

The MCP server is a thin adapter: it forwards `verify_text`, `get_job_status`,
`get_health` calls and the `report://{run_id}` resource to the FastAPI lite
API documented in `src/api/`. This means agents (Claude Desktop, Claude
Agent SDK, any MCP client) can consume the engine without speaking HTTP +
managing async-job polling themselves.

Public re-exports kept minimal — most callers only need `main` (the stdio
entrypoint installed as the `copilot-mcp` console script) or `build_server`
(for tests / custom transports).
"""

from src.mcp_server.client import (
    LiteApiClient,
    LiteApiClientError,
    LiteApiTimeoutError,
)
from src.mcp_server.config import McpConfig
from src.mcp_server.server import build_server, main

__all__ = [
    "LiteApiClient",
    "LiteApiClientError",
    "LiteApiTimeoutError",
    "McpConfig",
    "build_server",
    "main",
]
