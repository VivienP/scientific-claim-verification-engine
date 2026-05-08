"""End-to-end orchestration for the verification pipeline.

This module is the single canonical composer of extraction, bibliography
parsing, multi-source resolution, full-text fetching, passage selection,
and verifier-mode dispatch. It is consumed by:

    examples/sample_run.py                       — interactive demo
    eval/e2e/_run_pipeline_on_verdicts.py        — benchmark harness
    scripts/measure_e2e_recall.py                — recall-only run
    tests/integration/test_full_pipeline.py      — end-to-end contract test

Design contract:

    1. **Pure orchestration.** No I/O configuration is decided in this
       module: callers pass a ``PipelineConfig`` carrying explicit values
       (db_path, api_key, top_k_passages, …).

    2. **No cost-cap or partial-save behaviour.** Callers that need
       per-claim cost gating (the benchmark harness) iterate over
       :func:`verify_one_claim` directly and sum costs themselves, then
       persist whatever shape they need. Keeping that policy out of the
       library is what lets the same orchestrator power the demo, the
       benchmark, the recall script, and the integration tests.

    3. **Side-effecting tasks live in their own modules.** ``src/extract.py``
       owns LLM-based extraction; ``src/resolve.py`` and ``src/clients/``
       own HTTP; ``src/verify.py`` owns the verifier modes. This module
       composes them but does not implement them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import structlog

from src.bibliography import BibEntry, parse_bibliography
from src.bm25_selector import select_passages
from src.chunker import chunk_paper
from src.extract import extract_claims
from src.fetch_fulltext import FulltextMethod, fetch_fulltext
from src.models import (
    Claim,
    PaperChunk,
    ProvenanceStep,
    ResolvedSource,
    ResolvedSourceSet,
    VerificationResult,
)
from src.resolve import resolve_citations_multi
from src.verify import (
    verify_claim,
    verify_claim_citing_context,
    verify_claim_fulltext_with_numeric,
    verify_claim_multi_source,
    verify_claim_title_only,
)

logger: structlog.BoundLogger = structlog.get_logger(__name__)


@dataclass(frozen=True)
class PipelineConfig:
    """Explicit knobs passed by callers — no defaults read from environment.

    ``api_key`` is required for any LLM-calling phase; pass ``None`` only
    when every claim is expected to short-circuit (e.g. resolver hit-rate
    measurement on cached data).

    ``db_path`` enables the per-client SQLite cache; ``None`` runs without
    a cache and is appropriate for one-shot integration tests.

    ``top_k_passages`` is the BM25 selection size for fulltext mode.

    ``enable_multi_source`` and ``enable_citing_context_fallback`` exist
    so demos and the recall script can run a slimmer path.
    """

    api_key: str | None = None
    db_path: Path | None = None
    top_k_passages: int = 3
    title_only_min_title_length: int = 20
    enable_multi_source: bool = True
    enable_citing_context_fallback: bool = True


@dataclass(frozen=True)
class ClaimVerification:
    """All per-claim outputs returned by :func:`verify_one_claim`.

    Holds the full audit trail: the canonical ``source`` used for the
    primary verdict, the full ``source_set`` for multi-citation claims,
    the verifier output, the retrieval method that produced (or failed
    to produce) full-text, and the BM25-selected passages. ``steps``
    contains every ProvenanceStep emitted during this claim's verification
    (resolve steps live on the parent run, not here).
    """

    claim: Claim
    source: ResolvedSource
    source_set: ResolvedSourceSet
    result: VerificationResult
    fetch_method: FulltextMethod | str
    passages: tuple[PaperChunk, ...] = field(default_factory=tuple)
    steps: tuple[ProvenanceStep, ...] = field(default_factory=tuple)


def verify_one_claim(
    claim: Claim,
    source_set: ResolvedSourceSet,
    *,
    citing_paper_text: str | None = None,
    config: PipelineConfig,
) -> ClaimVerification:
    """Run fetch → select → verify on a single resolved claim.

    Routing decision tree (kept here, in one place, so the benchmark and
    demo cannot drift apart):

        1. ``len(source_set) > 1`` and at least one source resolved
           → :func:`verify_claim_multi_source` over per-source passages.
        2. Otherwise, fetch full-text on the primary source.
           a. Full-text retrieved → :func:`verify_claim_fulltext_with_numeric`.
           b. Abstract present, no full-text → :func:`verify_claim`.
           c. Title-only, no abstract → :func:`verify_claim_title_only`
              (hard-capped to ``partially_supported``).
        3. If the resulting verdict has ``evidence_quality == 'no_evidence'``
           and ``citing_paper_text`` is provided, attempt
           :func:`verify_claim_citing_context` as a last-resort
           internal-consistency check (capped to ``partially_supported``).

    The function never raises on missing data: a fully unresolvable claim
    returns a ``ClaimVerification`` whose ``result.status`` is
    ``not_addressed`` and whose ``fetch_method`` is ``"abstract_fallback"``.
    """
    source = source_set.primary()
    steps: list[ProvenanceStep] = []
    passages: tuple[PaperChunk, ...] = ()
    fetch_method: FulltextMethod | str = "abstract_fallback"

    if config.enable_multi_source and len(source_set) > 1 and len(source_set.found_sources()) > 0:
        passages_per_source: dict[str, list[PaperChunk]] = {}
        for sub_source in source_set:
            if not sub_source.found:
                continue
            ft, _ = fetch_fulltext(sub_source, db_path=config.db_path)
            if ft is not None:
                sub_chunks = chunk_paper(sub_source.doi or claim.claim_id, ft)
                passages_per_source[sub_source.doi or ""] = list(
                    select_passages(claim.claim_text, sub_chunks, top_k=config.top_k_passages)
                )
        result, verify_steps = verify_claim_multi_source(
            claim,
            source_set,
            passages_per_source=passages_per_source,
            api_key=config.api_key,
        )
        steps.extend(verify_steps)
        # Backward-compat: report the primary source's fetch method.
        _, fetch_method = fetch_fulltext(source, db_path=config.db_path)
        passages = tuple(passages_per_source.get(source.doi or "", []))
    else:
        fulltext, fetch_method = fetch_fulltext(source, db_path=config.db_path)
        if fulltext is not None:
            chunks = chunk_paper(source.doi or claim.claim_id, fulltext)
            passages = tuple(select_passages(claim.claim_text, chunks, top_k=config.top_k_passages))
            result, verify_steps = verify_claim_fulltext_with_numeric(
                claim,
                source,
                list(passages),
                api_key=config.api_key,
            )
            steps.extend(verify_steps)
        elif source.found and source.abstract:
            result, verify_step = verify_claim(claim, source, api_key=config.api_key)
            steps.append(verify_step)
        elif (
            source.found
            and source.title is not None
            and len(source.title) >= config.title_only_min_title_length
        ):
            result, verify_step = verify_claim_title_only(claim, source, api_key=config.api_key)
            steps.append(verify_step)
        else:
            result, verify_step = verify_claim(claim, source, api_key=config.api_key)
            steps.append(verify_step)

    if (
        config.enable_citing_context_fallback
        and result.evidence_quality == "no_evidence"
        and citing_paper_text is not None
        and len(claim.claim_text) >= 20
    ):
        cc_result, cc_step = verify_claim_citing_context(
            claim,
            source,
            citing_paper_text,
            api_key=config.api_key,
        )
        steps.append(cc_step)
        if cc_result.status in ("partially_supported", "unsupported"):
            result = cc_result
            fetch_method = "citing_paper_context"

    return ClaimVerification(
        claim=claim,
        source=source,
        source_set=source_set,
        result=result,
        fetch_method=fetch_method,
        passages=passages,
        steps=tuple(steps),
    )


def run_pipeline(
    text: str,
    *,
    config: PipelineConfig,
    pre_extracted_claims: list[Claim] | None = None,
    pre_parsed_bibliography: dict[int, BibEntry] | None = None,
) -> tuple[list[ClaimVerification], list[ProvenanceStep]]:
    """Run the full pipeline on a free-form scientific text.

    Phases:

        1. Extract claims via :func:`extract_claims` (skipped when
           ``pre_extracted_claims`` is supplied; the benchmark harness
           uses this to evaluate verifiers against fixed claim sets).
        2. Parse bibliography (skipped when ``pre_parsed_bibliography``
           is supplied).
        3. Resolve all claims via :func:`resolve_citations_multi`.
        4. For each claim, dispatch to :func:`verify_one_claim` with the
           citing paper text passed through for the citing-context
           fallback.

    Returns ``(claim_verifications, all_provenance_steps)``. The steps
    list contains the extract step (if any), every resolve step, and
    every verify step in the order they were emitted.
    """
    all_steps: list[ProvenanceStep] = []

    if pre_extracted_claims is None:
        claims, extract_step = extract_claims(text, api_key=config.api_key)
        all_steps.append(extract_step)
    else:
        claims = list(pre_extracted_claims)

    bibliography = (
        pre_parsed_bibliography if pre_parsed_bibliography is not None else parse_bibliography(text)
    )

    source_sets, resolve_steps = resolve_citations_multi(
        claims, bibliography=bibliography, api_key=config.api_key, db_path=config.db_path
    )
    all_steps.extend(resolve_steps)

    verifications: list[ClaimVerification] = []
    for claim in claims:
        cv = verify_one_claim(
            claim,
            source_sets[claim.claim_id],
            citing_paper_text=text,
            config=config,
        )
        verifications.append(cv)
        all_steps.extend(cv.steps)

    logger.info(
        "pipeline_complete",
        n_claims=len(claims),
        n_steps=len(all_steps),
        n_multi_source=sum(1 for v in verifications if len(v.source_set) > 1),
    )
    return verifications, all_steps


__all__ = [
    "ClaimVerification",
    "PipelineConfig",
    "run_pipeline",
    "verify_one_claim",
]
