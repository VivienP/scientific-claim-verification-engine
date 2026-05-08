"""Post-resolution enrichment helpers: fill abstract / pmcid / oa_url gaps from secondary APIs."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import structlog

from src.clients import europepmc as _europepmc
from src.clients import pubmed as _pubmed
from src.models import ResolvedSource
from src.resolve_utils import _normalize_pmcid

logger: structlog.BoundLogger = structlog.get_logger(__name__)


def _pick_longer_abstract(existing: str | None, candidate: str | None) -> str | None:
    """Return whichever abstract is more informative (longer, prefers existing on ties)."""
    if not existing:
        return candidate
    if not candidate:
        return existing
    return existing if len(existing) >= len(candidate) else candidate


def _enrich_via_pubmed(
    source: ResolvedSource,
    *,
    db_path: Path | None,
) -> ResolvedSource:
    """Populate pmcid and (if missing) abstract from PubMed via DOI->PMID->record.

    Bug A fix (S1-P1-A): the previous `_enrich_abstract_via_pubmed` short-circuited
    whenever CrossRef had any abstract, silently dropping the pmcid. PubMed often
    has the pmcid even when CrossRef does not (NIH-deposit OA mirrors), and
    propagating it unblocks the PMC fulltext path in `fetch_fulltext`. This
    function:

    * fires for any found source with a DOI that lacks either abstract OR pmcid;
    * uses `find_pmid_by_doi` -> `fetch_record` (preserving the full record);
    * preserves the existing abstract when CrossRef's is longer than PubMed's,
      and always backfills pmcid when PubMed exposes one.

    Returns source unchanged on any miss (caching layer keeps repeat lookups cheap).
    """
    if not source.found or source.doi is None:
        return source
    if source.abstract and source.pmcid:
        return source
    pmid = _pubmed.find_pmid_by_doi(source.doi, db_path=db_path)
    if pmid is None:
        return source
    record = _pubmed.fetch_record(pmid, db_path=db_path)
    if record is None:
        return source
    new_abstract = _pick_longer_abstract(source.abstract, record.abstract)
    new_pmcid = source.pmcid or _normalize_pmcid(record.pmcid)
    if new_abstract == source.abstract and new_pmcid == source.pmcid:
        return source
    logger.info(
        "pubmed_record_enriched",
        doi=source.doi,
        pmid=pmid,
        abstract_added=source.abstract is None and record.abstract is not None,
        pmcid_added=source.pmcid is None and new_pmcid is not None,
    )
    return dataclasses.replace(source, abstract=new_abstract, pmcid=new_pmcid)


def _enrich_via_europepmc(
    source: ResolvedSource,
    *,
    db_path: Path | None,
) -> ResolvedSource:
    """Fill abstract / pmcid / oa_url gaps from Europe PMC.

    S2-P1: Europe PMC indexes biomedical OA mirrors that NCBI sometimes misses
    and exposes abstracts through `abstractText` even when CrossRef returns
    null. The S2-P0 OA discovery probe confirmed 100 % abstract coverage and
    50 % OA URL coverage on a paywall-heavy benchmark sample.

    Fires only when at least one of (abstract, pmcid, oa_url) is still missing
    after `_enrich_via_pubmed`. Existing values are preserved on conflict
    (longer abstract wins; pmcid / oa_url are filled only when None).

    Returns source unchanged on any miss.
    """
    if not source.found or source.doi is None:
        return source
    if source.abstract and source.pmcid and source.oa_url:
        return source
    record = _europepmc.fetch_record(source.doi, db_path=db_path)
    if record is None:
        return source

    new_abstract = _pick_longer_abstract(source.abstract, record.abstract)
    new_pmcid = source.pmcid or record.pmcid
    new_oa_url = source.oa_url or record.pdf_url or record.html_url

    if (
        new_abstract == source.abstract
        and new_pmcid == source.pmcid
        and new_oa_url == source.oa_url
    ):
        return source

    logger.info(
        "europepmc_enriched",
        doi=source.doi,
        abstract_added=source.abstract is None and record.abstract is not None,
        pmcid_added=source.pmcid is None and new_pmcid is not None,
        oa_url_added=source.oa_url is None and new_oa_url is not None,
    )
    return dataclasses.replace(
        source,
        abstract=new_abstract,
        pmcid=new_pmcid,
        oa_url=new_oa_url,
    )
