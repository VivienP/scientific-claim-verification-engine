# Phase C lite API — single-tenant on-prem container.
# Build:  docker build -t locuslab/copilot:0.1.0 .
# Run:    docker compose up
#
# Multi-stage to keep the runtime image small (no build toolchain, no .pyc cache).
# Non-root user, read-only root FS at runtime, healthcheck on /health.

# -----------------------------------------------------------------------------
# Stage 1 — build dependencies
# -----------------------------------------------------------------------------
FROM python:3.12-slim-bookworm AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

# System deps for pymupdf and sqlite (TTL cache).
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libsqlite3-dev \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
COPY src/ ./src/

# Exact-pinned dependencies (no `>=` ranges) to make the image reproducible
# and to block silent supply-chain upgrades within unbounded ranges. Versions
# match the local development environment as of 2026-05-09. Upgrade by editing
# this list explicitly; CI-driven SBOM diff reviews will catch unintended
# changes. (Hash-pinning via `pip-compile --generate-hashes` is the next step
# once the dependency cadence stabilises.)
RUN pip install --upgrade "pip==24.2" \
 && pip install --prefix=/install \
        "fastapi==0.135.1" "uvicorn[standard]==0.41.0" \
        "anthropic==0.97.0" "httpx==0.28.1" "structlog==25.5.0" \
        "rank-bm25==0.2.2" "pymupdf==1.27.2.3" "tiktoken==0.7.0" \
        "jinja2==3.1.4" "pydantic==2.12.5" "pydantic-settings==2.13.1"

# -----------------------------------------------------------------------------
# Stage 2 — runtime image
# -----------------------------------------------------------------------------
FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    UVICORN_WORKERS=2

# Run as non-root.
RUN groupadd -r app && useradd -r -g app -u 10001 app

WORKDIR /app

COPY --from=builder /install /usr/local
COPY --chown=app:app src/ ./src/

# Reports directory is mounted as a volume in compose; predeclare so the
# image is usable standalone too.
RUN mkdir -p /app/reports/runs && chown -R app:app /app

USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import httpx,sys; sys.exit(0 if httpx.get('http://127.0.0.1:8000/health').status_code==200 else 1)"

# Run uvicorn directly. Workers > 1 means each worker has its own in-memory
# JobStore — for single-tenant Phase C this is acceptable because a single
# user's polling will hit the same worker only with sticky sessions; for
# strict correctness, run with --workers 1 until Phase D adds shared state.
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
