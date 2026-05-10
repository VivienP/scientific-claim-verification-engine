"""Environment-driven configuration for the MCP wrapper.

Frozen dataclass — same shape as the rest of the codebase (no pydantic-settings
to keep startup time low for the stdio subprocess). Loaded once via
``McpConfig.from_env`` in ``main``; tests construct it directly.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class McpConfig:
    """Runtime configuration for the MCP wrapper.

    Attributes:
        api_base_url: Base URL of the lite API (no trailing slash).
        api_key: Value forwarded as ``X-API-Key`` on protected endpoints.
        poll_interval_seconds: Sleep between polls in ``wait_for_job``.
            Tuned for sub-second feedback on short docs without thrashing
            the API on long ones.
        timeout_seconds: Hard ceiling on ``wait_for_job`` duration. The
            default (600s = 10 min) covers the worst-case Copilot run on
            a 30-claim pharma doc; agents that want shorter waits should
            pass ``timeout_seconds`` explicitly to ``verify_text``.
        request_timeout_seconds: Per-HTTP-request timeout passed to httpx.
            This is independent of the polling timeout — a single GET
            should never block the event loop indefinitely.
    """

    api_base_url: str
    api_key: str
    poll_interval_seconds: float = 2.0
    timeout_seconds: float = 600.0
    request_timeout_seconds: float = 30.0

    @classmethod
    def from_env(cls) -> McpConfig:
        """Build a config from process environment.

        Required:
            COPILOT_API_KEY    — forwarded to the API as X-API-Key.

        Optional:
            COPILOT_API_BASE_URL          (default http://127.0.0.1:8000)
            COPILOT_MCP_TIMEOUT           (default 600.0 seconds)
            COPILOT_MCP_POLL_INTERVAL     (default 2.0 seconds)
            COPILOT_MCP_REQUEST_TIMEOUT   (default 30.0 seconds)

        Raises:
            ValueError: when COPILOT_API_KEY is missing or empty. The MCP
                stdio subprocess must fail fast — Claude Desktop logs a
                useful error rather than a silent retry loop.
        """
        api_key = os.environ.get("COPILOT_API_KEY", "").strip()
        if not api_key:
            raise ValueError("COPILOT_API_KEY env var is required (must match the lite API's key).")
        base_url = os.environ.get("COPILOT_API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
        return cls(
            api_base_url=base_url,
            api_key=api_key,
            timeout_seconds=float(os.environ.get("COPILOT_MCP_TIMEOUT", "600.0")),
            poll_interval_seconds=float(os.environ.get("COPILOT_MCP_POLL_INTERVAL", "2.0")),
            request_timeout_seconds=float(os.environ.get("COPILOT_MCP_REQUEST_TIMEOUT", "30.0")),
        )


__all__ = ["McpConfig"]
