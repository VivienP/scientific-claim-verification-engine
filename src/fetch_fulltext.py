"""Full-text retrieval orchestrator: oa_url -> PMC -> Europe PMC -> Unpaywall -> fallback.

I1 (2026-05-12): fetch_fulltext now returns a structured FetchOutcome instead
of a (text, method) tuple. The chain order is unchanged; each attempt
(success or failure) is recorded with a reason code so report.json can
surface coverage-by-publisher diagnostics without re-running the pipeline.
"""

from __future__ import annotations

import time
from pathlib import Path

import structlog

from src.clients import europepmc, pdf, pmc, publisher_html, unpaywall
from src.models import (
    FetchAttempt,
    FetchFailureReason,
    FetchOutcome,
    FulltextMethod,
    ResolvedSource,
)

logger: structlog.BoundLogger = structlog.get_logger(__name__)

# Re-export FulltextMethod from models for backward-compatible imports.
__all__ = ["FulltextMethod", "fetch_fulltext"]


def _ms_since(t0: float) -> int:
    return int((time.perf_counter() - t0) * 1000)


def _attempt(
    method: FulltextMethod,
    success: bool,
    reason: FetchFailureReason | None,
    started_at: float,
) -> FetchAttempt:
    return FetchAttempt(
        method=method, success=success, reason=reason, elapsed_ms=_ms_since(started_at)
    )


def fetch_fulltext(
    source: ResolvedSource,
    *,
    db_path: Path | None = None,
) -> FetchOutcome:
    """Try retrieval chain and return a structured FetchOutcome.

    Order, stopping at first success:
        1. source.oa_url    -> pdf.download_and_extract
        2. source.pmcid     -> pmc.fetch_fulltext
        3. source.doi       -> publisher_html.fetch_via_doi  (known publishers, e.g. NEJM)
        4. source.doi       -> europepmc.fetch_oa_url -> pdf.download_and_extract
        5. source.doi       -> unpaywall.get_oa_url   -> pdf.download_and_extract
        6. abstract_fallback (text=None)

    Short-circuits without HTTP when all of doi, pmcid, oa_url are None,
    recording a single attempt with reason="no_identifiers".

    Never raises.
    """
    t0 = time.perf_counter()
    attempts: list[FetchAttempt] = []

    if source.doi is None and source.pmcid is None and source.oa_url is None:
        logger.debug("fulltext_no_identifiers")
        attempts.append(
            FetchAttempt(
                method="abstract_fallback",
                success=False,
                reason="no_identifiers",
                elapsed_ms=0,
            )
        )
        return FetchOutcome(
            text=None,
            method="abstract_fallback",
            attempts=tuple(attempts),
            elapsed_ms_total=_ms_since(t0),
        )

    if source.oa_url:
        ts = time.perf_counter()
        text = pdf.download_and_extract(source.oa_url, db_path=db_path)
        if text is not None:
            attempts.append(_attempt("oa_url_pdf", True, None, ts))
            logger.info("fulltext_method", method="oa_url_pdf", doi=source.doi)
            return FetchOutcome(
                text=text,
                method="oa_url_pdf",
                attempts=tuple(attempts),
                elapsed_ms_total=_ms_since(t0),
            )
        # Heuristic: the most common reason an oa_url PDF fetch returns None
        # is that the publisher served an HTML paywall page (Content-Type
        # mismatch). We attribute generically as `oa_url_pdf_failed`; the
        # finer-grained `oa_url_not_pdf` is reserved for the case where the
        # PDF client surfaces a Content-Type signal (deferred until pdf.py
        # is structurally upgraded per Track D2).
        attempts.append(_attempt("oa_url_pdf", False, "oa_url_pdf_failed", ts))

    if source.pmcid:
        ts = time.perf_counter()
        text = pmc.fetch_fulltext(source.pmcid, db_path=db_path)
        if text is not None:
            attempts.append(_attempt("pmc", True, None, ts))
            logger.info("fulltext_method", method="pmc", pmcid=source.pmcid)
            return FetchOutcome(
                text=text,
                method="pmc",
                attempts=tuple(attempts),
                elapsed_ms_total=_ms_since(t0),
            )
        attempts.append(_attempt("pmc", False, "pmc_no_fulltext", ts))

    if source.doi:
        ts = time.perf_counter()
        html_text = publisher_html.fetch_via_doi(source.doi, db_path=db_path)
        if html_text is not None:
            attempts.append(_attempt("publisher_html", True, None, ts))
            logger.info("fulltext_method", method="publisher_html", doi=source.doi)
            return FetchOutcome(
                text=html_text,
                method="publisher_html",
                attempts=tuple(attempts),
                elapsed_ms_total=_ms_since(t0),
            )
        # publisher_html.fetch_via_doi short-circuits to None for any DOI
        # whose prefix isn't in the known-publisher map. Distinguish that
        # cheap no-op from an actual failed fetch: if the DOI is in the
        # known-publisher set, the call attempted HTTP and got blocked/
        # paywalled. Otherwise it's a free skip.
        # The known-prefix set is small (NEJM only at present); checking it
        # here avoids a public-API change in publisher_html.
        known_prefix = source.doi.startswith("10.1056/")  # NEJM; extend as map grows
        reason_pubhtml: FetchFailureReason = (
            "publisher_html_blocked" if known_prefix else "publisher_html_unknown"
        )
        attempts.append(_attempt("publisher_html", False, reason_pubhtml, ts))

        ts = time.perf_counter()
        epmc_url = europepmc.fetch_oa_url(source.doi, db_path=db_path)
        if epmc_url:
            text = pdf.download_and_extract(epmc_url, db_path=db_path)
            if text is not None:
                attempts.append(_attempt("europepmc_pdf", True, None, ts))
                logger.info("fulltext_method", method="europepmc_pdf", doi=source.doi)
                return FetchOutcome(
                    text=text,
                    method="europepmc_pdf",
                    attempts=tuple(attempts),
                    elapsed_ms_total=_ms_since(t0),
                )
            attempts.append(_attempt("europepmc_pdf", False, "europepmc_pdf_failed", ts))
        else:
            attempts.append(_attempt("europepmc_pdf", False, "europepmc_no_oa", ts))

        ts = time.perf_counter()
        oa_url = unpaywall.get_oa_url(source.doi, db_path=db_path)
        if oa_url:
            text = pdf.download_and_extract(oa_url, db_path=db_path)
            if text is not None:
                attempts.append(_attempt("unpaywall_pdf", True, None, ts))
                logger.info("fulltext_method", method="unpaywall_pdf", doi=source.doi)
                return FetchOutcome(
                    text=text,
                    method="unpaywall_pdf",
                    attempts=tuple(attempts),
                    elapsed_ms_total=_ms_since(t0),
                )
            attempts.append(_attempt("unpaywall_pdf", False, "unpaywall_pdf_failed", ts))
        else:
            attempts.append(_attempt("unpaywall_pdf", False, "unpaywall_no_oa", ts))

    logger.info("fulltext_method", method="abstract_fallback", doi=source.doi)
    return FetchOutcome(
        text=None,
        method="abstract_fallback",
        attempts=tuple(attempts),
        elapsed_ms_total=_ms_since(t0),
    )
