"""Batch citation resolution via OpenAlex with CrossRef fallback and retraction check."""

from __future__ import annotations

import dataclasses
import hashlib
import time
import uuid
from pathlib import Path

import structlog

from src.clients import crossref as _crossref
from src.clients.openalex import search_paper
from src.models import Claim, ProvenanceStep, ResolvedSource

logger: structlog.BoundLogger = structlog.get_logger(__name__)

_NOT_FOUND = ResolvedSource(found=False, doi=None, title=None, abstract=None, similarity_score=None)


def _hash(data: str) -> str:
    return hashlib.sha256(data.encode()).hexdigest()


def _build_query(claim: Claim) -> str:
    parts: list[str] = (
        claim.cited_authors[:3]
        + ([str(claim.cited_year)] if claim.cited_year else [])
        + claim.claim_text.split()[:5]
    )
    return " ".join(parts)


def resolve_citations(
    claims: list[Claim],
    *,
    api_key: str | None = None,
    db_path: Path | None = None,
) -> tuple[dict[str, ResolvedSource], list[ProvenanceStep]]:
    """Resolve each claim's cited source via OpenAlex.

    Returns a dict keyed by claim_id (entry present for EVERY claim, even unresolved).
    Returns one ProvenanceStep per claim (operation="resolve", model_id=None).
    Claims with cited_authors=[] or cited_year=None → ResolvedSource(found=False) without HTTP call.
    """
    sources: dict[str, ResolvedSource] = {}
    steps: list[ProvenanceStep] = []

    for claim in claims:
        ts = time.time()
        if not claim.cited_authors or claim.cited_year is None:
            source = _NOT_FOUND
            logger.debug("resolve_skipped_no_citation", claim_id=claim.claim_id)
        else:
            query = _build_query(claim)
            source = search_paper(query, api_key=api_key, db_path=db_path)
            if not source.found:
                cf_source = _crossref.search_paper(query, db_path=db_path)
                if cf_source.found:
                    source = cf_source
                    logger.info("crossref_fallback_success", claim_id=claim.claim_id)

        if source.doi is not None:
            retracted = _crossref.check_retraction(source.doi, db_path=db_path)
            if retracted:
                logger.warning("retraction_detected", claim_id=claim.claim_id, doi=source.doi)
            source = dataclasses.replace(source, retraction_status=retracted)

        sources[claim.claim_id] = source
        steps.append(
            ProvenanceStep(
                step_id=str(uuid.uuid4()),
                claim_id=claim.claim_id,
                operation="resolve",
                input_hash=_hash(repr(claim)),
                output_hash=_hash(repr(source)),
                model_id=None,
                timestamp=ts,
                tokens_in=None,
                tokens_out=None,
                cache_hit=None,
                confidence=source.similarity_score,
            )
        )

    return sources, steps
