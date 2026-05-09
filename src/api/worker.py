"""Background worker — runs the V1 pipeline + optional Copilot enrichment.

Invoked by FastAPI's BackgroundTasks. Wraps the synchronous pipeline so
exceptions are captured into the JobStore rather than crashing the process.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import structlog

from src.api.jobs import JobStore
from src.api.models import VerifyRequest

logger: structlog.BoundLogger = structlog.get_logger(__name__)


def run_verification_job(
    job_id: str,
    req: VerifyRequest,
    store: JobStore,
    *,
    runs_root: Path = Path("reports/runs"),
) -> None:
    """Execute one verification job to completion. Never raises.

    On exception the job is marked ``failed`` with the exception text in
    ``error``. The traceback is logged via structlog but NOT returned to
    the API consumer — pharma deployments don't want internals leaking.
    """
    started_at = time.time()
    store.update(job_id, status="running")
    run_dir = runs_root / f"api-{job_id[:8]}"
    run_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Lazy import — keeps app startup fast and avoids loading heavy
        # pipeline dependencies in test environments that mock this worker.
        from src.copilot.enricher import CopilotConfig, CopilotEnricher
        from src.copilot.models import CopilotMode
        from src.copilot.report_html import build_copilot_report
        from src.pipeline import PipelineConfig, run_pipeline

        # Step 1 — V1 pipeline.
        pipeline_config = PipelineConfig()
        cvs, _pipeline_steps = run_pipeline(req.text, config=pipeline_config)
        logger.info("api_pipeline_done", job_id=job_id, n_claims=len(cvs))

        # Step 2 — optional Copilot enrichment.
        if req.mode == "copilot":
            copilot_cfg = CopilotConfig(
                mode=CopilotMode(req.copilot_mode),
                enable_primary_lookup=req.enable_primary_lookup,
                enable_recommended_fix=req.enable_recommended_fix,
                db_path=run_dir / "_cache.db",
            )
            enricher = CopilotEnricher(copilot_cfg)
            enriched = enricher.enrich_all(cvs)

            # Persist the HTML report.
            html_path = build_copilot_report(
                run_dir,
                enriched,
                run_id=run_dir.name,
                runtime_seconds=time.time() - started_at,
                total_cost_usd=_estimate_cost(enriched),
            )
            logger.info("api_copilot_done", job_id=job_id, html=str(html_path))

            result = _build_copilot_result_summary(enriched, run_dir)
        else:
            result = _build_v1_result_summary(cvs, run_dir)

        store.update(
            job_id,
            status="completed",
            run_id=run_dir.name,
            result=result,
        )
        logger.info(
            "api_job_completed",
            job_id=job_id,
            run_id=run_dir.name,
            elapsed_s=round(time.time() - started_at, 2),
        )

    except Exception as exc:
        # Log the full traceback for ops; surface a redacted message to API.
        logger.exception("api_job_failed", job_id=job_id, run_id=run_dir.name)
        store.update(
            job_id,
            status="failed",
            run_id=run_dir.name,
            error=_redact_exception_for_api(exc),
        )


def _estimate_cost(enriched: list[Any]) -> float:
    """Sum copilot_steps token usage, convert to USD using Sonnet-4 pricing."""
    total_in = 0
    total_out = 0
    for ev in enriched:
        for step in ev.copilot_steps:
            if step.tokens_in:
                total_in += step.tokens_in
            if step.tokens_out:
                total_out += step.tokens_out
    return round((total_in * 3.0 + total_out * 15.0) / 1_000_000, 4)


def _build_v1_result_summary(cvs: list[Any], run_dir: Path) -> dict[str, Any]:
    """Compact V1 result envelope returned by GET /jobs/{id}."""
    counts: dict[str, int] = {}
    for cv in cvs:
        counts[cv.result.status] = counts.get(cv.result.status, 0) + 1
    return {
        "n_claims": len(cvs),
        "verdict_counts": counts,
        "run_dir": str(run_dir),
        "report_html_url": None,  # V1 mode does not produce the copilot HTML.
    }


def _build_copilot_result_summary(enriched: list[Any], run_dir: Path) -> dict[str, Any]:
    """Compact Copilot result envelope. The full enriched JSON is on disk."""
    counts: dict[str, int] = {}
    n_with_fix = 0
    for ev in enriched:
        counts[ev.base.result.status] = counts.get(ev.base.result.status, 0) + 1
        if ev.copilot.recommended_fix is not None:
            n_with_fix += 1
    return {
        "n_claims": len(enriched),
        "verdict_counts": counts,
        "n_with_fix": n_with_fix,
        "run_dir": str(run_dir),
        "report_html_url": f"/runs/{run_dir.name}/copilot_report.html",
    }


def _redact_exception_for_api(exc: BaseException) -> str:
    """Convert an exception into a single safe message for the API response.

    Strips file paths and tracebacks. The full information is in the logs.
    """
    name = type(exc).__name__
    msg = str(exc).splitlines()[0] if str(exc) else "no message"
    # Cap at 200 chars to avoid unbounded payloads.
    return f"{name}: {msg[:200]}"
