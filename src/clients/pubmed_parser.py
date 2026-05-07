"""Pure text-parsing and title-ranking helpers for the PubMed client.

All functions here are deterministic (no I/O, no side effects, no module-level
mutable state). They are extracted from pubmed.py to keep the HTTP/cache logic
separate from the text-processing logic.
"""

from __future__ import annotations

import re

_PMID_LINE = re.compile(r"^\s*PMID:\s*\d+.*$", re.MULTILINE)
_DOI_LINE = re.compile(r"^\s*DOI:\s*(\S+).*$", re.MULTILINE)
_PMCID_LINE = re.compile(r"^\s*PMCID:\s*(PMC\d+).*$", re.MULTILINE)
_AUTHOR_INFO = re.compile(r"^Author information:.*?(?=\n\n)", re.MULTILINE | re.DOTALL)
_COPYRIGHT_LINE = re.compile(r"^©.*$|^Copyright .*$", re.MULTILINE)
_WHITESPACE = re.compile(r"\s+")
_TOKEN = re.compile(r"[a-z0-9]+")
_TITLE_STOPWORDS = {
    "a",
    "an",
    "and",
    "for",
    "in",
    "of",
    "on",
    "the",
    "to",
    "with",
}


def _content_tokens(text: str) -> set[str]:
    return {token for token in _TOKEN.findall(text.lower()) if token not in _TITLE_STOPWORDS}


def _title_overlap_score(target_title: str, candidate_title: str) -> tuple[int, int]:
    target_tokens = _content_tokens(target_title)
    candidate_tokens = _content_tokens(candidate_title)
    if not target_tokens or not candidate_tokens:
        return (0, 0)
    overlap = target_tokens & candidate_tokens
    return (len(overlap), len(overlap) - len(candidate_tokens - target_tokens))


def _strip_metadata(text: str) -> str:
    cleaned = _AUTHOR_INFO.sub("", text)
    cleaned = _PMID_LINE.sub("", cleaned)
    cleaned = _DOI_LINE.sub("", cleaned)
    cleaned = _PMCID_LINE.sub("", cleaned)
    cleaned = _COPYRIGHT_LINE.sub("", cleaned)
    return cleaned


def _extract_abstract_body(raw: str) -> str | None:
    """Pull the abstract body out of the efetch text response.

    The efetch text response has a header (citation, authors, affiliations),
    then a blank line, then the abstract body, then trailing metadata
    (DOI, PMID, copyright). Concatenate all non-metadata text and normalize.
    """
    cleaned = _strip_metadata(raw)
    blocks = [b.strip() for b in cleaned.split("\n\n") if b.strip()]
    if not blocks:
        return None
    # The first block is typically the citation/header. The abstract body is
    # usually the longest remaining block. Concatenate everything after the
    # first block to keep multi-paragraph abstracts together.
    body_parts = blocks[1:] if len(blocks) > 1 else blocks
    body = " ".join(body_parts)
    body = _WHITESPACE.sub(" ", body).strip()
    return body or None


def _extract_record_fields(raw: str) -> tuple[str | None, str | None, str | None]:
    """Return (abstract, doi, pmcid) parsed from an efetch text response."""
    doi_match = _DOI_LINE.search(raw)
    pmcid_match = _PMCID_LINE.search(raw)
    abstract = _extract_abstract_body(raw)
    doi = doi_match.group(1).rstrip(".,;)") if doi_match else None
    pmcid = pmcid_match.group(1) if pmcid_match else None
    return abstract, doi, pmcid
