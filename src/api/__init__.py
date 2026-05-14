"""FastAPI lite API for the Scientific Claim Verification Engine.

A single-tenant on-prem-deployable HTTP service that wraps
``run_pipeline()`` (V1) and ``CopilotEnricher.enrich_all()``.

Design constraints:
- **Async jobs + polling**: pipeline runs take 2-8 minutes per document, so
  POST /verify returns a ``job_id`` immediately and the work happens in a
  background task. GET /jobs/{id} polls for status and result.
- **Single API key**: validated via ``X-API-Key`` header against the
  ``VERIFIER_API_KEY`` environment variable. Multi-tenant auth is a
  future-phase concern.
- **In-memory job store**: sufficient for one-process on-prem deployments.
  A single-tenant biotech does not need Redis or Postgres.
- **No persistent state in the API itself**: all run artifacts
  (``report.json``, ``provenance.jsonl``, ``copilot_report.html``) live on
  disk under ``reports/runs/{run_id}/``. The API just orchestrates.

Public exports:
- :func:`create_app` — factory; tests pass a JobStore stub for isolation.
"""

from src.api.app import create_app

__all__ = ["create_app"]
