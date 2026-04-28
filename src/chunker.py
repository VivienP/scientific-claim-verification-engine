"""Section-aware IMRAD chunking — deterministic, regex-based, zero LLM."""

from __future__ import annotations

import re

from src.models import PaperChunk, SectionLabel

_MIN_CHUNK_LENGTH = 50
_MAX_HEADER_LINE_LENGTH = 80

# Order matters: most-specific patterns first (e.g. "Materials and Methods" before "Methods").
# Each pattern matches at the start of a line and may be preceded by a numbering token
# like "1.", "1)", "II.", "III.".
_HEADER_PATTERNS: list[tuple[re.Pattern[str], SectionLabel]] = [
    (re.compile(r"^(?:\d+[\.\)]?\s+|[IVX]+\.\s+)?materials?\s+and\s+methods?\b", re.I), "methods"),
    (re.compile(r"^(?:\d+[\.\)]?\s+|[IVX]+\.\s+)?methods?\b", re.I), "methods"),
    (re.compile(r"^(?:\d+[\.\)]?\s+|[IVX]+\.\s+)?results?\b", re.I), "results"),
    (re.compile(r"^(?:\d+[\.\)]?\s+|[IVX]+\.\s+)?discussion\b", re.I), "discussion"),
    (re.compile(r"^(?:\d+[\.\)]?\s+|[IVX]+\.\s+)?conclusions?\b", re.I), "discussion"),
    (re.compile(r"^(?:\d+[\.\)]?\s+|[IVX]+\.\s+)?introduction\b", re.I), "introduction"),
    (re.compile(r"^(?:\d+[\.\)]?\s+|[IVX]+\.\s+)?abstract\b", re.I), "introduction"),
]


def _find_section_boundaries(text: str) -> list[tuple[int, SectionLabel]]:
    """Find (line_start_offset, label) for every detected section header line.

    A header is a line whose trimmed length is <= _MAX_HEADER_LINE_LENGTH and
    whose start matches one of the IMRAD patterns. Returned list is sorted by
    char offset, with duplicate offsets dropped (first label wins).
    """
    boundaries: list[tuple[int, SectionLabel]] = []
    seen_offsets: set[int] = set()

    pos = 0
    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        if 0 < len(stripped) <= _MAX_HEADER_LINE_LENGTH:
            for pattern, label in _HEADER_PATTERNS:
                if pattern.match(stripped):
                    if pos not in seen_offsets:
                        boundaries.append((pos, label))
                        seen_offsets.add(pos)
                    break
        pos += len(line)

    return boundaries


def chunk_paper(doi: str, text: str) -> list[PaperChunk]:
    """Split a paper's text into section-aware chunks.

    Algorithm:
        1. Find header lines matching IMRAD patterns.
        2. Slice text between consecutive headers; assign label.
        3. Text before the first header → section="other".
        4. If no headers detected → single chunk with section="other".
        5. Filter out chunks whose text is < _MIN_CHUNK_LENGTH chars.

    Pure function. Deterministic. Never calls any external API or LLM.
    """
    if not text or not text.strip():
        return []

    boundaries = _find_section_boundaries(text)

    if not boundaries:
        return _maybe_chunk(doi, "other", text, 0, len(text))

    chunks: list[PaperChunk] = []

    first_offset = boundaries[0][0]
    if first_offset > 0:
        chunks.extend(_maybe_chunk(doi, "other", text, 0, first_offset))

    for i, (start, label) in enumerate(boundaries):
        end = boundaries[i + 1][0] if i + 1 < len(boundaries) else len(text)
        chunks.extend(_maybe_chunk(doi, label, text, start, end))

    return chunks


def _maybe_chunk(
    doi: str,
    section: SectionLabel,
    text: str,
    raw_start: int,
    raw_end: int,
) -> list[PaperChunk]:
    """Build a chunk for text[raw_start:raw_end] with offsets adjusted to the
    trimmed slice, so that text[char_start:char_end] == chunk.text exactly.
    Returns [] if the trimmed slice is < _MIN_CHUNK_LENGTH.
    """
    raw = text[raw_start:raw_end]
    leading = len(raw) - len(raw.lstrip())
    trailing = len(raw) - len(raw.rstrip())
    body = raw[leading : len(raw) - trailing]
    if len(body) < _MIN_CHUNK_LENGTH:
        return []
    return [
        PaperChunk(
            doi=doi,
            section=section,
            text=body,
            char_start=raw_start + leading,
            char_end=raw_end - trailing,
        )
    ]
