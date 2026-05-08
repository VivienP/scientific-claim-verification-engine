"""Unit tests for src/clients/_common.py.

The intent here is to *lock the cache-key bytes*: the migration from
per-client ``hashlib.sha256(f"{prefix}:{value}".encode()).hexdigest()``
calls to :func:`make_cache_key` must not invalidate any existing cache
entry in ``.cache/api_cache.db``. These tests will fail loudly if a
future refactor changes the format.
"""

from __future__ import annotations

import hashlib

from src.clients._common import (
    CACHE_TTL_DEFAULT_SECONDS,
    RETRACTION_CACHE_TTL_SECONDS,
    RETRY_BACKOFF_BASE,
    RETRY_MAX,
    make_cache_key,
)


class TestCacheTTLs:
    def test_default_ttl_is_30_days(self) -> None:
        assert CACHE_TTL_DEFAULT_SECONDS == 30 * 24 * 3600

    def test_retraction_ttl_is_7_days(self) -> None:
        assert RETRACTION_CACHE_TTL_SECONDS == 7 * 24 * 3600

    def test_retraction_ttl_strictly_shorter_than_default(self) -> None:
        assert RETRACTION_CACHE_TTL_SECONDS < CACHE_TTL_DEFAULT_SECONDS


class TestRetryBudget:
    def test_retry_max_is_three(self) -> None:
        assert RETRY_MAX == 3

    def test_backoff_base_is_two_seconds(self) -> None:
        # Yields wait=2, 4, 8 seconds — fits inside the verifier's per-claim
        # budget without making interactive runs feel hung.
        assert RETRY_BACKOFF_BASE == 2.0


class TestMakeCacheKeyByteEquivalence:
    """Migration constraint: keys must equal pre-migration hashes."""

    def test_two_part_key_matches_legacy_format(self) -> None:
        # Legacy: hashlib.sha256(f"{prefix}:{value}".encode()).hexdigest()
        legacy = hashlib.sha256(b"crossref:doi:10.1/abc").hexdigest()
        assert make_cache_key("crossref:doi", "10.1/abc") == legacy

    def test_single_part_key_matches_legacy_unpaywall_format(self) -> None:
        legacy = hashlib.sha256(b"unpaywall:10.5/x").hexdigest()
        assert make_cache_key("unpaywall", "10.5/x") == legacy

    def test_three_part_key_matches_legacy_pubmed_title_format(self) -> None:
        legacy = hashlib.sha256(b"pubmed_title_to_pmid_v3:some title:2020").hexdigest()
        assert make_cache_key("pubmed_title_to_pmid_v3", "some title", "2020") == legacy

    def test_three_part_key_with_empty_year_matches_legacy(self) -> None:
        # Original used `f"...:{year or ''}"` — empty string for None year.
        legacy = hashlib.sha256(b"pubmed_title_to_pmid_v3:some title:").hexdigest()
        assert make_cache_key("pubmed_title_to_pmid_v3", "some title", "") == legacy

    def test_unicode_value_is_utf8_encoded(self) -> None:
        # CrossRef titles can contain non-ASCII. The legacy code used
        # ``.encode()`` which defaults to UTF-8.
        legacy = hashlib.sha256("crossref:Pösö 1986".encode()).hexdigest()
        assert make_cache_key("crossref", "Pösö 1986") == legacy

    def test_keys_differ_when_namespace_differs(self) -> None:
        a = make_cache_key("crossref", "10.1/x")
        b = make_cache_key("openalex", "10.1/x")
        assert a != b
