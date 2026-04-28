"""BM25 passage selection — deterministic, pure Python, zero LLM."""

from __future__ import annotations

import re

from rank_bm25 import BM25Okapi  # type: ignore[import-untyped]

from src.models import PaperChunk

_TOKEN_RE = re.compile(r"\w+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def select_passages(
    claim_text: str,
    chunks: list[PaperChunk],
    *,
    top_k: int = 3,
) -> list[PaperChunk]:
    """Select the top-k chunks most relevant to a claim, ranked by BM25Okapi.

    Tokenization: lowercase + word characters only (\\w+).
    If len(chunks) <= top_k, returns chunks unchanged.
    If all BM25 scores are zero (no token overlap), returns the first top_k
    chunks by position — never an empty list when chunks are provided.
    Pure function, deterministic.
    """
    if not chunks:
        return []
    if len(chunks) <= top_k:
        return list(chunks)

    corpus = [_tokenize(c.text) for c in chunks]
    query = _tokenize(claim_text)

    if not query or all(len(doc) == 0 for doc in corpus):
        return list(chunks[:top_k])

    bm25 = BM25Okapi(corpus)
    scores = bm25.get_scores(query)

    if max(scores) == 0:
        return list(chunks[:top_k])

    # Sort indices by score descending, then by position ascending for tie-break
    ranked = sorted(range(len(chunks)), key=lambda i: (-scores[i], i))
    return [chunks[i] for i in ranked[:top_k]]
