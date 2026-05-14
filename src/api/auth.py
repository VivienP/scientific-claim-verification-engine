"""Single shared API key auth middleware.

Validates the ``X-API-Key`` header against the ``VERIFIER_API_KEY``
environment variable. Per-tenant keys + rate limiting are a future-phase
concern.

Security notes:
- The expected key MUST be loaded from env, never hard-coded.
- Comparison uses ``hmac.compare_digest`` to defend against timing attacks
  even though the lite-API threat model is on-prem.
- Empty / unset ``VERIFIER_API_KEY`` raises 503 — fail closed, never open.
"""

from __future__ import annotations

import hmac
import os
from typing import Annotated

from fastapi import Header, HTTPException, status

# Header name kept lowercase for Pydantic/FastAPI; HTTP is case-insensitive.
_API_KEY_HEADER = "x-api-key"
_API_KEY_ENV = "VERIFIER_API_KEY"


def _expected_key() -> str:
    key = os.environ.get(_API_KEY_ENV)
    if not key:
        # Fail closed. We do not allow an unset key to mean "open".
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Server is not configured: {_API_KEY_ENV} is not set.",
        )
    return key


async def require_api_key(
    x_api_key: Annotated[str | None, Header(alias=_API_KEY_HEADER)] = None,
) -> None:
    """FastAPI dependency that gates a route on a valid X-API-Key header.

    Use as ``Depends(require_api_key)`` on any protected route. /health does
    NOT use it — health checks must work for the load balancer / on-prem
    monitoring without secrets.
    """
    expected = _expected_key()
    if x_api_key is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing X-API-Key header.",
        )
    if not hmac.compare_digest(x_api_key.encode("utf-8"), expected.encode("utf-8")):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key.",
        )


__all__ = ["_API_KEY_ENV", "_API_KEY_HEADER", "require_api_key"]
