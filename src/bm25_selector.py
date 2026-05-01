"""BM25 passage selection — deterministic, pure Python, zero LLM.

Selection is BM25-ranked, then trimmed to fit a token budget so the verifier
prompt stays under Claude's rate-limit-friendly window. tiktoken (cl100k_base)
underestimates Claude Sonnet by ~10-15% on dense scientific text — defaults
account for this; do not raise without re-measuring against the SDK counter.
"""

from __future__ import annotations

import dataclasses
import re

import structlog
import tiktoken
from rank_bm25 import BM25Okapi  # type: ignore[import-untyped]

from src.models import PaperChunk

logger = structlog.get_logger(__name__)

_TOKEN_RE = re.compile(r"\w+")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_ENCODER = tiktoken.get_encoding("cl100k_base")
# Each chunk is wrapped as `<passage section="...">\n{text}\n</passage>` in
# verify._build_passages_block — reserve a small allowance per chunk for those tags.
_WRAPPER_TOKEN_OVERHEAD = 12


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _count_tokens(text: str) -> int:
    return len(_ENCODER.encode(text))


def _hard_cut_at_whitespace(text: str, max_tokens: int) -> str:
    """Last-resort cut when no full sentence fits the budget — never mid-word."""
    tokens = _ENCODER.encode(text)
    if len(tokens) <= max_tokens:
        return text
    candidate = _ENCODER.decode(tokens[:max_tokens])
    last_ws = max(candidate.rfind(" "), candidate.rfind("\n"), candidate.rfind("\t"))
    if last_ws == -1:
        return candidate
    return candidate[:last_ws]


def _truncate_to_budget(chunk: PaperChunk, max_tokens: int) -> PaperChunk:
    """Sentence-boundary truncate so encoded text fits under max_tokens.

    char_start/char_end are deliberately preserved — they reference the chunk's
    location in the source paper, not len(text). Truncation does not move the
    chunk in the original document.
    """
    sentences = _SENTENCE_SPLIT_RE.split(chunk.text)
    kept: list[str] = []
    running = 0
    for sentence in sentences:
        candidate = sentence + " "
        cost = _count_tokens(candidate)
        if running + cost > max_tokens:
            break
        kept.append(sentence)
        running += cost

    truncated_text = " ".join(kept) if kept else _hard_cut_at_whitespace(chunk.text, max_tokens)
    return dataclasses.replace(chunk, text=truncated_text)


def select_passages(
    claim_text: str,
    chunks: list[PaperChunk],
    *,
    top_k: int = 3,
    # 6000 tokens x 3 passages = 18k per verify call. Combined with system prompt
    # + claim + structured output (~3.5k), total ~21k per call. With 25s throttle
    # = ~50k tokens/min headroom under the 30k limit. Reduce if rate-limit errors return.
    max_total_tokens: int = 6000,
) -> list[PaperChunk]:
    """Select up to top_k chunks most relevant to a claim, ranked by BM25Okapi.

    Tokenization (BM25): lowercase + word characters only (\\w+).
    Token budget: enforced via tiktoken cl100k_base. Adds chunks (highest
    BM25 score first) until the next would exceed max_total_tokens. If the
    top-ranked chunk alone exceeds the budget, it is truncated at sentence
    boundary and a `chunk_truncated_to_fit` warning is logged.

    Returns [] when chunks is empty, the query has no token overlap with any
    chunk, or all corpus tokens are empty. Pure function, deterministic.
    """
    if not chunks:
        return []

    corpus = [_tokenize(c.text) for c in chunks]
    query = _tokenize(claim_text)

    if not query or all(len(doc) == 0 for doc in corpus):
        return []

    query_tokens = set(query)
    if not any(query_tokens.intersection(doc) for doc in corpus):
        return []

    if len(chunks) > 1:
        bm25 = BM25Okapi(corpus)
        scores = bm25.get_scores(query)
        if max(scores) == 0:
            return []
        ranked_indices = sorted(range(len(chunks)), key=lambda i: (-scores[i], i))
    else:
        ranked_indices = [0]

    candidates = [chunks[i] for i in ranked_indices[:top_k]]

    out: list[PaperChunk] = []
    running = 0
    for chunk in candidates:
        cost = _count_tokens(chunk.text) + _WRAPPER_TOKEN_OVERHEAD
        if not out and cost > max_total_tokens:
            truncated = _truncate_to_budget(chunk, max_total_tokens - _WRAPPER_TOKEN_OVERHEAD)
            logger.warning(
                "chunk_truncated_to_fit",
                doi=chunk.doi,
                section=chunk.section,
                char_start=chunk.char_start,
                char_end=chunk.char_end,
                original_chars=len(chunk.text),
                truncated_chars=len(truncated.text),
                original_tokens=_count_tokens(chunk.text),
                truncated_tokens=_count_tokens(truncated.text),
                budget=max_total_tokens,
            )
            out.append(truncated)
            return out
        if running + cost > max_total_tokens:
            break
        running += cost
        out.append(chunk)

    return out
