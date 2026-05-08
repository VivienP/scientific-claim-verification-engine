# Scientific Claim Verification Engine

![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue)
![License](https://img.shields.io/badge/license-Apache%202.0-green)
![Tests](https://img.shields.io/badge/tests-295-brightgreen)

Auditable claim-by-claim evidence review for scientific text.

The pipeline turns free-form scientific text — paper drafts, literature
reviews, AI-generated summaries — into a source-grounded verification report,
one verdict per cited claim, with full provenance. The current scope is
deliberately conservative: this is an honest evidence-grounding verifier, not
a broad automatic inconsistency detector.

**Hand-labeled full-pipeline benchmark** ([`eval/e2e/`](eval/e2e/reference_paper_v1_results.md)): **16/25 (64%) verdict agreement** on 25 cited claims from a domain-expert-annotated lactate-ISF literature review (Perrelle 2023). Verifier-component F1 on SciFact dev maintained at 0.94 across all sprints.

| Sprint | Verdict agreement | Key changes |
|---|---|---|
| Baseline (Phase 0) | 1/25 (4%) | Initial pipeline, no bibliography awareness |
| Interim | 4/25 (16%) | Resolver hardening |
| End of S1 | 12/25 (48%) | Rubric v2 + title-only verifier + PMCID enrichment + Jaccard scoring |
| End of S2 | 16/25 (64%) | Europe PMC client + multi-source aggregation |
| End of S3 | 16/25 (64%) | Citing-context fallback + fulltext token bump + test cull |

The remaining 9 disagreements split into three structural categories: (a) paywalled / identifier-less sources unreachable without institutional access (claims 008, 009, 011, 015) — out of scope for an OA-only verifier, (b) abstract-on-topic-but-silent on the specific assertion (008, 009 again, 022, 024) — the verifier is correctly conservative, (c) annotator-borderline calls (017, 022, 024). See [`eval/e2e/reference_paper_v1_results.md`](eval/e2e/reference_paper_v1_results.md) for the per-claim diagnostic.

## Related Work

The space of LLM-grounded scientific claim verification is now actively populated. This project's positioning rests on architecture and use case rather than problem novelty.

**Closest neighbors:**

- [Valsci](https://github.com/bricee98/Valsci) ([Brice et al. 2025, BMC Bioinformatics](https://link.springer.com/article/10.1186/s12859-025-06159-4)) is an open-source self-hostable verifier grounded in Semantic Scholar, optimized for large-batch literature verification with bibliometric scoring. This project differs in three structural ways: (i) cited-source auditing of AI-agent outputs rather than literature corpora, (ii) multi-source resolution chain (CrossRef → OpenAlex → PubMed → Europe PMC → Unpaywall) instead of single-source S2, (iii) deterministic retrieval and numeric checks separated from probabilistic LLM verification per the `no-llm-in-deterministic` rule.
- **CiteAudit** (arXiv 2602.23452) is a 5-agent pipeline (Extractor → Memory → Web → Scholar → Judge) reporting F1 = 0.838 on real citations. CiteAudit and this engine are complementary: CiteAudit covers metadata consistency (does the citation exist? does it match what is claimed about it?); this engine covers semantic entailment (does the cited source actually support the claim's specific assertion?).

**Benchmarks and evaluation standards:**

- **SciClaimEval** ([sciclaimeval.github.io](https://sciclaimeval.github.io/)) — cross-modal claim ↔ table/figure benchmark (1,664 samples, 180 papers). SOTA is o4-mini at 68.2% pair-accuracy. Not a direct competitor (cross-modal, not text→text), but its perturbation methodology is informative.
- **AAR standard** (arXiv 2602.13855) defines four metrics — Provenance Coverage (PCov), Provenance Soundness (PSnd), Claim Transparency (CTran), Audit Efficiency (AEff) — emerging as the consensus scorecard for evaluating audit tools over research agents. Adopting this scorecard is on the S4 roadmap.
- **SciClaimHunt_Num** (arXiv 2502.10003) is the closest public asset to the lactate-ISF benchmark — numeric scientific claim verification with structured ground truth.
- **MuSciClaims** confirms the 3-class SUPPORT / NEUTRAL / CONTRADICT rubric used here is the public formalization of the absence-of-support-vs-not-addressed distinction.
- **AFEV** (arXiv 2506.07446) introduces adaptive atomic decomposition for compound claims — relevant to the multi-source aggregation ceiling we hit at S2.

**Where this engine is uniquely positioned today:**

1. **Multi-source aggregation** — when a claim cites `[81-83]`, the pipeline resolves all three references and aggregates verdicts with explicit precedence rules, instead of averaging or picking the strongest.
2. **Paywall recovery beyond Unpaywall + Europe PMC** — bibliography-aware routing recovers references that would otherwise hit a not-found dead-end.
3. **Partial-support reasoning over numeric ranges** — the verifier rubric (Clauses A/B/C/D) explicitly handles range/uncertainty inclusion, trajectory-vs-snapshot, and numeric-verbatim-absence cases that flatten in 3-class systems.

## What It Does

The pipeline takes free-form scientific text and writes:

- `report.json`: one record per extracted claim, with verdict, cited source,
  retrieval status, evidence quality, source quotes when found, and numeric
  checks when applicable.
- `provenance.jsonl`: append-only provenance for each extraction, resolution,
  verification, numeric, and aggregation step, including token and cache data.

Supported verdicts are `supported`, `unsupported`, `partially_supported`, and
`not_addressed`. The verifier keeps abstention explicit instead of forcing weak
evidence into a contradiction.

## Real-Output Benchmark

Committed reports under `benchmarks/real_outputs/` currently cover three
AI-for-science outputs:

| tool | claims | supported | partially_supported | unsupported | not_addressed | citation_found_rate | fulltext_verified | retracted_sources | numeric_checks_run | numeric_inconsistencies_flagged | cost |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Edison Scientific Literature (TREM2) | 20 | 3 | 2 | 1 | 14 | 85.0% | 12 | 0 | 1 | 0 | $0.38 |
| Sakana AI Scientist v2 (CompReg) | 14 | 0 | 0 | 0 | 14 | 71.4% | 3 | 0 | 0 | 0 | $0.16 |
| AnswerThis (lactate ISF PK) | 25 | 1 | 1 | 1 | 22 | 64.0% | 13 | 0 | 0 | 0 | $0.47 |
| **Total** | **59** | **4** | **3** | **2** | **50** | **72.9%** | **28** | **0** | **1** | **0** | **$1.01** |

Numbers regenerated 2026-05-06 from `benchmarks/real_outputs/SUMMARY.md`. Re-run via `python scripts/check_summary_alignment.py`.

## Canary Controls

`benchmarks/canary/` contains a clearly labeled seeded input for demo and
regression testing. It is separate from the real-output benchmark and should not
be merged into public aggregate counts.

Currently verified: contradiction detection (deliberately inverted AlphaFold
claim → `unsupported`). Weak resolution, numeric inconsistency, and retraction
paths are declared in `benchmarks/canary/README.md` under **Not yet
implemented**.

## Quick Start

```bash
pip install -e ".[dev]"
export ANTHROPIC_API_KEY=sk-...
python examples/sample_run.py
python scripts/show_report.py
```

Default input: `benchmarks/real_outputs/edison_trem2/input.txt`.

Expected shape (cache-cold, ~3-4 min, ~$0.25):

```text
Extracted 21 claims.
Report written to: reports/runs/{report_id}/
Full-text retrieval methods: abstract_fallback=N, oa_url_pdf=N, unpaywall_pdf=N
```

The exact retrieval-method counts vary run-to-run with extraction
non-determinism and OA availability. See
`benchmarks/real_outputs/edison_trem2/report.json` for the most recent
committed run (12 passage-found, 9 fulltext-unavailable, $0.24).

Run the canary controls explicitly:

```bash
python examples/sample_run.py benchmarks/canary/input.txt
python scripts/show_report.py
```

## Pipeline

```text
input text
  -> extract_claims()              # LLM: citation-anchored claims
  -> parse_bibliography()          # numbered references (if present in source)
  -> resolve_citations_multi()     # bib DOI -> CrossRef direct; else richer query
                                   # fallback chain: OpenAlex -> CrossRef -> PubMed
                                   # multi-marker citations resolve all sources,
                                   # returns ResolvedSourceSet for aggregation
  -> _enrich_via_pubmed()          # PMID-via-DOI for abstract / PMCID propagation
  -> _enrich_via_europepmc()       # OA discovery + abstract when CrossRef is null
  -> fetch_fulltext()              # OA URL -> PMC -> Europe PMC -> Unpaywall PDF
  -> chunk_paper()                 # deterministic IMRAD section chunks
  -> select_passages()             # BM25 + token-budget truncation
  -> route to verifier mode:
        verify_claim_fulltext()         # passages available -> primary path
        verify_claim()                  # abstract-only fallback
        verify_claim_title_only()       # title-only, hard-capped to partially_*
        verify_claim_multi_source()     # ResolvedSourceSet -> aggregated verdict
        verify_claim_citing_context()   # last-resort: internal-consistency check
                                        #   against the citing paper's own text,
                                        #   capped to partially_supported, never
                                        #   labelled as independent verification
  -> run_numeric_check()           # deterministic OR/CI and p-value/CI checks
  -> build_report()                # report.json + provenance.jsonl
```

Resolution and retrieval diagnostics are first-class fields:

- `source.title_match_score`: lexical title/abstract overlap with the query.
- `source.resolution_low_confidence`: true when source matching is weak.
- `verification.retrieval_status`: `passage_found`, `no_passage_found`, or
  `fulltext_unavailable`.
- `verification.evidence_quality`: `quoted_passage`, `abstract_only`, or
  `no_evidence`.

## Public API

Core calls:

```python
claims, extract_step = extract_claims(text)
bibliography = parse_bibliography(text)

# Multi-source resolution returns ResolvedSourceSet per claim;
# .primary() preserves backward compatibility with single-source callers.
source_sets, resolve_steps = resolve_citations_multi(claims, bibliography=bibliography)
source = source_sets[claim.claim_id].primary()

fulltext, method = fetch_fulltext(source)
chunks = chunk_paper(source.doi or claim.claim_id, fulltext)
passages = select_passages(claim.claim_text, chunks, top_k=3)

# Single-source verifier (most common path)
result, steps = verify_claim_fulltext_with_numeric(claim, source, passages)

# Multi-source aggregation (when len(source_sets[id].sources) > 1)
result, steps = verify_claim_multi_source(claim, source_sets[claim.claim_id])

run_dir = build_report(report_id, text, claims, sources, results, steps)
```

Data models are frozen dataclasses in `src/models.py`: `Claim`,
`ResolvedSource`, `ResolvedSourceSet`, `VerificationResult`, `ProvenanceStep`,
and `PaperChunk`.

## Evaluation

Verifier-component F1 = 0.94 on SciFact dev (oracle inputs). Note: the SciFact eval at [`scripts/eval_scifact.py:184`](scripts/eval_scifact.py) collapses `partially_supported` into `supported` to match SciFact's 3-class label set, so this F1 measures binary support/contradict on oracle abstracts and does not measure how the verifier handles the partial class.

*Full-pipeline verdict agreement on the lactate-ISF hand-labeled benchmark moved from 1/25 (Phase 0) → 12/25 (S1) → 16/25 (S2-S3) over three sprints. The 9 remaining disagreements break down as: 4 paywalled or identifier-less sources unreachable without institutional access (out of scope for an OA-only verifier), 3 abstracts on-topic but silent on the specific assertion (the verifier is correctly conservative on these), and 2 annotator-borderline calls. The 25-claim sample is a fixed evaluation harness, not a tuning target — generalization across documents is on the S4+ roadmap. Confusion matrix, per-claim breakdown, and recovery analysis are in [`eval/e2e/reference_paper_v1_results.md`](eval/e2e/reference_paper_v1_results.md).*

`scripts/eval_scifact.py` builds oracle claims and oracle abstract sources,
bypassing extraction, resolution, retrieval, and BM25. It does not measure the
full extract → resolve → retrieve → verify pipeline.

```bash
python scripts/eval_scifact.py --split dev
```

The SciFact test split is locked and must not be used for prompt selection or
tuning.

## Development

```bash
python -m pytest -v
python -m mypy --strict src
python -m ruff check src tests scripts examples
```

## Known Limitations

- Extraction currently requires explicit author/year citation anchors. Numeric bracket citations work only when the references are present in the input.
- `similarity_score` is still year-proximity; use `title_match_score` and `resolution_low_confidence` for source-match diagnostics.
- BM25 is lexical. If no chunk shares claim tokens, the pipeline now reports `no_passage_found` instead of sending arbitrary first chunks to the verifier.
- LLM `confidence` is self-reported and not calibrated. Treat
  `retrieval_status`, `evidence_quality`, and source quotes as higher-signal evidence diagnostics.
- Numeric coverage is intentionally narrow: OR/CI consistency and p-value/CI null-crossing checks only.
- Each claim is checked against its cited source, not against the whole literature.

## Contact

Questions, feedback, or interesting failure modes:
[@PerrelleVivien](https://x.com/PerrelleVivien) on X.

## License

Apache 2.0. See [LICENSE](LICENSE).
