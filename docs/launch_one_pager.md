# Scientific Claim Verification Engine

**Auditable claim-by-claim verification of any scientific text — built for the AI-for-science era where research output velocity outruns peer review.**

## Problem

Generative tools (literature reviews, AI-generated drafts, autonomous-agent papers) emit confident scientific claims faster than humans can fact-check. Existing options are unfit for the job: hallucination detectors don't ground in cited sources, plagiarism tools don't check truth, and human peer review doesn't scale. The cost of an unverified false claim entering a regulatory or clinical pipeline is measured in years and millions.

## What it does

Takes any scientific text (paper draft, AI summary, literature review) and produces a per-claim verification report grounded in the cited source's full text — not just abstracts. Surfaces three failure modes peer review misses: claim-source contradictions, internally inconsistent statistics, and citations to retracted papers.

## Demo numbers (real outputs from three AI-for-science tools)

| tool | claims | full-text verified | supported | partially | unsupported | numeric checks | cost |
|---|---|---|---|---|---|---|---|
| Edison Scientific (TREM2) | 21 | 14 | 3 | 1 | 0 | 2 | $1.25 |
| Sakana AI Scientist v2 | 13 | 2 | 0 | 0 | 0 | 0 | $0.28 |
| AnswerThis (lactate) | 19 | 8 | 1 | 1 | 0 | 0 | $0.88 |
| **Total** | **53** | **24** | **4** | **2** | **0** | **2** | **$2.41** |

`<!-- TODO: replace numeric_inconsistencies_flagged column once a clear inconsistency is captured -->`

## How it works

1. **Extract** — LLM pulls verifiable claims with author/year anchors from free-form text
2. **Resolve** — OpenAlex + CrossRef → DOI for each citation
3. **Fetch full text** — chain: OA URL → PMC → Unpaywall → abstract fallback (zero-cost public APIs)
4. **Chunk + select** — IMRAD section-aware chunking; BM25 picks the 3 most relevant passages
5. **Verify** — full-text passage comparison against the claim, with cited-source provenance
6. **Numeric check** — deterministic OR/CI consistency engine (Python, no LLM in the comparison step)
7. **Report** — claim-by-claim verdict with full provenance trail (every step hashed, cached, costed)

Engineering choices that matter: prompt-cached system prompts, structured logging, mypy --strict, 173+ unit tests, F1=0.94 on SciFact dev split (locked test split untouched).

## Worked example: numeric consistency

```
Claim: "ARM were 77.5% in A+T− vs 7.8% in A−T− (OR 40.53, 95% CI 23.58–73.71)"

LLM verdict (full-text):  partially_supported  (confidence 0.85)
Numeric engine verdict:   consistent           (or_ci_consistency: 23.58 ≤ 40.53 ≤ 73.71, CI ratio 3.13)
```

The numeric engine is deterministic — same input, same output, every time. No prompt-engineering moat to lose.

## What we want from a design partner

- **A regulated workflow with skin in the game.** Clinical-stage biotech, regulatory submissions, or a lab that publishes high-stakes preprints.
- **Two real input streams per week.** AI-generated drafts, literature reviews, or competitor papers — anything where the cost of a hallucinated citation is non-trivial.
- **30-min weekly feedback call** for 4 weeks. Trade: free use of the engine, on-prem deployable if needed, your reports kept private.

## Contact

**Vivien Perrelle** — Founder, Locus Lab
[vivienperrelle@gmail.com](mailto:vivienperrelle@gmail.com)
`<!-- TODO: add Calendly link if available at sprint end -->`

`<!-- numbers above sourced from benchmarks/real_outputs/SUMMARY.md, generated 2026-04-28 -->`
