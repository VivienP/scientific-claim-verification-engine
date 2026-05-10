"""FastAPI app factory for the lite verification API.

Routes:
    GET  /health                     liveness + active-job count (no auth)
    POST /verify                     async; returns {job_id, poll_url}
    GET  /jobs/{job_id}              poll job status; returns result when done
    GET  /runs/{run_id}/copilot_report.html   serves the copilot HTML

Worker pluggability:
    ``create_app`` accepts an optional ``worker`` callable so tests can
    swap in a fake that completes synchronously without invoking the real
    pipeline. This is the only pluggable seam — every other concern (auth,
    job storage, models) is concrete in the lite app.

Background execution model:
    FastAPI's ``BackgroundTasks`` schedules the worker on the same event
    loop. For long-running synchronous work we'd prefer a thread or
    process pool; the worker uses ``starlette.concurrency.run_in_threadpool``
    to avoid blocking the loop. This is sufficient for single-tenant
    Phase C; Phase D moves to Celery/RQ.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import structlog
from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    HTTPException,
    Request,
    status,
)
from fastapi.responses import FileResponse, JSONResponse
from starlette.concurrency import run_in_threadpool

from src.api.auth import require_api_key
from src.api.jobs import JobStore
from src.api.models import (
    HealthResponse,
    JobCreated,
    JobStatusResponse,
    VerifyRequest,
)
from src.api.worker import run_verification_job

logger: structlog.BoundLogger = structlog.get_logger(__name__)

API_VERSION = "0.1.0"

WorkerFn = Callable[[str, VerifyRequest, JobStore], None]
"""Type alias: the worker signature consumed by BackgroundTasks."""


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def create_app(
    *,
    job_store: JobStore | None = None,
    worker: WorkerFn | None = None,
    runs_root: Path | None = None,
) -> FastAPI:
    """Build a configured FastAPI app.

    Args:
        job_store: Override the default in-memory store. Tests pass a fresh
            instance so cases are isolated.
        worker: Override the worker. Tests pass a fake that records calls
            and completes synchronously to avoid invoking the real pipeline.
        runs_root: Where ``api-{job_id}`` directories are created. Tests
            use ``tmp_path`` to avoid littering the workspace.
    """
    app = FastAPI(
        title="Scientific Claim Verification Engine",
        version=API_VERSION,
        description="Lite API for V1 pipeline + Phase B Copilot enrichment.",
    )
    app.state.job_store = job_store or JobStore()
    app.state.runs_root = runs_root or Path("reports/runs")
    app.state.worker = worker or run_verification_job

    # ----- Routes -----

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        return HealthResponse(
            version=API_VERSION,
            n_jobs_active=len(app.state.job_store.list_active()),
        )

    @app.post(
        "/verify",
        response_model=JobCreated,
        status_code=status.HTTP_202_ACCEPTED,
        dependencies=[Depends(require_api_key)],
    )
    async def submit_verify(
        req: VerifyRequest,
        background: BackgroundTasks,
        request: Request,
    ) -> JobCreated:
        """Create a new verification job and run it in the background.

        Returns 202 Accepted with a poll URL — the caller polls
        ``/jobs/{job_id}`` until status is ``completed`` or ``failed``.
        """
        job = app.state.job_store.create()
        worker_fn = app.state.worker
        runs_root = app.state.runs_root
        store = app.state.job_store

        # Run synchronous worker on a thread to avoid blocking the loop.
        async def _runner() -> None:
            await run_in_threadpool(
                worker_fn,
                job.job_id,
                req,
                store,
                runs_root=runs_root,
            )

        background.add_task(_runner)

        logger.info("api_job_submitted", job_id=job.job_id, mode=req.mode)
        return JobCreated(
            job_id=job.job_id,
            status="pending",
            poll_url=str(request.url_for("get_job", job_id=job.job_id)),
        )

    @app.get(
        "/jobs/{job_id}",
        response_model=JobStatusResponse,
        name="get_job",
        dependencies=[Depends(require_api_key)],
    )
    async def get_job(job_id: str) -> JobStatusResponse:
        job = app.state.job_store.get(job_id)
        if job is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Job {job_id} not found.",
            )
        return JobStatusResponse(
            job_id=job.job_id,
            status=job.status,
            created_at=job.created_at,
            updated_at=job.updated_at,
            run_id=job.run_id,
            error=job.error,
            result=job.result,
        )

    @app.get(
        "/runs/{run_id}/copilot_report.html",
        dependencies=[Depends(require_api_key)],
    )
    async def get_report(run_id: str) -> FileResponse:
        """Serve the Copilot HTML report for a completed run.

        Path is resolved + confined to ``runs_root`` to defend against
        traversal payloads in the ``run_id`` URL segment. Requires a
        valid X-API-Key header. Returns 400 on traversal, 404 when the
        report has not been generated for this run.
        """
        runs_root: Path = app.state.runs_root
        candidate = (runs_root / run_id / "copilot_report.html").resolve()
        try:
            candidate.relative_to(runs_root.resolve())
        except ValueError as exc:
            logger.warning(
                "api_path_traversal_attempt",
                run_id=run_id,
                runs_root=str(runs_root),
            )
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid run_id (path traversal detected).",
            ) from exc
        if not candidate.is_file():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Report not found for run {run_id}.",
            )
        return FileResponse(candidate, media_type="text/html")

    @app.exception_handler(404)
    async def _not_found_handler(_: Request, exc: HTTPException) -> JSONResponse:
        return JSONResponse(
            status_code=404,
            content={"error": "not_found", "detail": exc.detail if exc.detail else "not found"},
        )

    return app


# ---------------------------------------------------------------------------
# ASGI entrypoint for `uvicorn src.api.app:app`
# ---------------------------------------------------------------------------

app = create_app()
