"""Shared constants and helpers for ``src/clients/*`` modules.

Each client module imports its TTL constants, retry budget, and cache-key
helper from here. The module:

* exposes :func:`make_cache_key`, which joins parts with ``":"`` so all
  clients produce byte-identical cache keys for the same logical request;
* documents the convention that **the first part is a namespace** (e.g.
  ``"crossref:doi"``, ``"pubmed_record_v1"``) so multiple clients can
  share one SQLite database without key collisions;
* gives one place to tune TTLs or retry behaviour.
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
