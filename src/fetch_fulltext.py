"""Full-text retrieval orchestrator: oa_url -> PMC -> Europe PMC -> Unpaywall -> fallback."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import structlog

from src.clients import europepmc, pdf, pmc, unpaywall
from src.models import ResolvedSource

logger: structlog.BoundLogger = structlog.get_logger(__name__)

FulltextMethod = Literal[
    "oa_url_pdf",
    "pmc",
    "europepmc_pdf",
    "unpaywall_pdf",
    "abstract_fallback",
]


def fetch_fulltext(
    source: ResolvedSource,
    *,
    db_path: Path | None = None,
) -> tuple[str | None, FulltextMethod]:
    """Try retrieval chain and return (fulltext_or_none, method).

    Order, stopping at first success:
        1. source.oa_url    -> pdf.download_and_extract
        2. source.pmcid     -> pmc.fetch_fulltext
        3. source.doi       -> europepmc.fetch_oa_url -> pdf.download_and_extract
        4. source.doi       -> unpaywall.get_oa_url   -> pdf.download_and_extract
        5. abstract_fallback (None)

    S2-P1 (Europe PMC) sits between PMC and Unpaywall because Europe PMC's
    full-text URLs are mostly Europe-PMC-mirror PDFs (already JATS-extracted)
    while Unpaywall returns publisher PDFs which may be paywalled despite a
    DOI match.

    Short-circuits without HTTP if all of doi, pmcid, oa_url are None.
    Never raises.
    """
    if source.doi is None and source.pmcid is None and source.oa_url is None:
        logger.debug("fulltext_no_identifiers")
        return None, "abstract_fallback"

    if source.oa_url:
        text = pdf.download_and_extract(source.oa_url, db_path=db_path)
        if text is not None:
            logger.info("fulltext_method", method="oa_url_pdf", doi=source.doi)
            return text, "oa_url_pdf"

    if source.pmcid:
        text = pmc.fetch_fulltext(source.pmcid, db_path=db_path)
        if text is not None:
            logger.info("fulltext_method", method="pmc", pmcid=source.pmcid)
            return text, "pmc"

    if source.doi:
        epmc_url = europepmc.fetch_oa_url(source.doi, db_path=db_path)
        if epmc_url:
            text = pdf.download_and_extract(epmc_url, db_path=db_path)
            if text is not None:
                logger.info("fulltext_method", method="europepmc_pdf", doi=source.doi)
                return text, "europepmc_pdf"

        oa_url = unpaywall.get_oa_url(source.doi, db_path=db_path)
        if oa_url:
            text = pdf.download_and_extract(oa_url, db_path=db_path)
            if text is not None:
                logger.info("fulltext_method", method="unpaywall_pdf", doi=source.doi)
                return text, "unpaywall_pdf"

    logger.info("fulltext_method", method="abstract_fallback", doi=source.doi)
    return None, "abstract_fallback"
