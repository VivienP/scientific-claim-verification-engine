"""Shared constants and helpers for ``src/clients/*`` modules.

Each client module previously redeclared the same TTL constants, retry
budget, and cache-key helper. Centralising them here:

* keeps the cached values byte-identical across the migration (no cache
  invalidation), because :func:`make_cache_key` joins parts with ``":"``
  exactly the same way the per-client helpers did;
* documents the convention that **the first part is a namespace** (e.g.
  ``"crossref:doi"``, ``"pubmed_record_v1"``) so multiple clients can
  share one SQLite database without key collisions;
* gives one place to bump TTLs or retry behaviour in future sprints.
"""

from __future__ import annotations

import hashlib

# 30 days — covers a full sprint cycle without re-hitting CrossRef /
# OpenAlex / PubMed / Europe PMC / Unpaywall / PMC. The benchmark harness
# relies on this for repeatable cost numbers across re-runs.
CACHE_TTL_DEFAULT_SECONDS = 30 * 24 * 3600

# 7 days — retraction notices propagate slowly but are stable; we want
# faster invalidation than the regular metadata cache so audits surface
# new retractions within a week.
RETRACTION_CACHE_TTL_SECONDS = 7 * 24 * 3600

# Retry budget for HTTP calls. Three attempts with exponential backoff
# (2s, 4s, 8s) is empirically enough to ride out CrossRef and OpenAlex
# rate-limit windows without making the verifier feel hung.
RETRY_MAX = 3
RETRY_BACKOFF_BASE = 2.0


def make_cache_key(*parts: str) -> str:
    """Build a sha256 cache key from a colon-joined sequence of parts.

    The result is byte-identical to the per-client ``hashlib.sha256(
    f"{prefix}:{value}".encode()).hexdigest()`` calls that this helper
    replaces. This is a hard requirement: changing the key format would
    invalidate every entry in ``.cache/api_cache.db`` and force the
    benchmark harness to re-pay every CrossRef / PubMed / Europe PMC
    request.
    """
    return hashlib.sha256(":".join(parts).encode()).hexdigest()
