"""Pydantic v2 request/response models for the lite API.

Mirrors the public-facing contracts. Internal dataclasses
(``ClaimVerification``, ``EnrichedVerification``) are converted to/from
these models via ``examples/copilot_run.py`` serialisation helpers and the
field-level mapping in ``app.py``.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------


class VerifyRequest(BaseModel):
    """POST /verify body."""

    model_config = ConfigDict(extra="forbid")

    text: str = Field(
        ...,
        min_length=1,
        max_length=2_000_000,
        description="The scientific text to verify (e.g. a paper draft, "
        "regulatory paragraph, or AI-generated summary).",
    )
    mode: Literal["v1", "copilot"] = Field(
        "copilot",
        description='"v1" returns plain ClaimVerification list. '
        '"copilot" runs the Phase B enrichment layer and produces a '
        "copilot_report.html.",
    )
    copilot_mode: Literal["pharma", "academic", "general"] = Field(
        "pharma",
        description="Which Copilot schema to apply (only used when mode=copilot).",
    )
    enable_primary_lookup: bool = Field(
        True,
        description="Enable Semantic Scholar primary-source lookup (pharma mode only).",
    )
    enable_recommended_fix: bool = Field(
        True,
        description="Enable LLM recommended_fix generation.",
    )


# ---------------------------------------------------------------------------
# Job tracking models
# ---------------------------------------------------------------------------


JobStatus = Literal["pending", "running", "completed", "failed"]


class JobCreated(BaseModel):
    """Response for POST /verify — immediately returned, work runs in background."""

    job_id: str
    status: JobStatus = "pending"
    poll_url: str = Field(
        ..., description="GET this URL to retrieve job status and result when ready."
    )


class JobStatusResponse(BaseModel):
    """Response for GET /jobs/{job_id}."""

    job_id: str
    status: JobStatus
    created_at: float
    updated_at: float
    run_id: str | None = Field(
        None,
        description="Run directory name on disk (under reports/runs/). "
        "Use it to fetch the HTML report.",
    )
    error: str | None = Field(None, description="Error message if status=failed.")
    result: dict[str, Any] | None = Field(
        None,
        description="Pipeline result payload when status=completed. Contains "
        "claims (list), n_claims, summary stats. Full enriched JSON at "
        "/jobs/{job_id}/enriched when mode=copilot.",
    )


class HealthResponse(BaseModel):
    """Response for GET /health."""

    status: Literal["ok"] = "ok"
    version: str
    n_jobs_active: int


class ErrorResponse(BaseModel):
    """4xx / 5xx error envelope."""

    error: str
    detail: str | None = None
