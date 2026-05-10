"""FastAPI lite API for the Scientific Claim Verification Engine.

Phase C deliverable: a single-tenant on-prem-deployable HTTP service that
wraps ``run_pipeline()`` (V1) and ``CopilotEnricher.enrich_all()`` (Phase B).

Design constraints:
- **Async jobs + polling**: pipeline runs take 2-8 minutes per document, so
  POST /verify returns a ``job_id`` immediately and the work happens in a
  background task. GET /jobs/{id} polls for status and result.
- **Single API key**: validated via ``X-API-Key`` header against the
  ``COPILOT_API_KEY`` environment variable. Phase D adds multi-tenant.
- **In-memory job store**: sufficient for one-process on-prem deployments.
  A single-tenant biotech does not need Redis or Postgres for Phase C.
- **No persistent state in the API itself**: all run artifacts
  (``report.json``, ``provenance.jsonl``, ``copilot_report.html``) live on
  disk under ``reports/runs/{run_id}/``. The API just orchestrates.

Public exports:
- :func:`create_app` — factory; tests pass a JobStore stub for isolation.
"""

from src.api.app import create_app

__all__ = ["create_app"]
