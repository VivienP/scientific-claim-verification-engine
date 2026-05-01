# Scientific Claim Verification Engine

**Auditable claim-by-claim verification of any scientific text — built for the AI-for-science era where research output velocity outruns peer review.**

## Problem

Generative tools (literature reviews, AI-generated drafts, autonomous-agent papers) emit confident scientific claims faster than humans can fact-check. Existing options are unfit for the job: hallucination detectors don't ground in cited sources, plagiarism tools don't check truth, and human peer review doesn't scale. The cost of an unverified false claim entering a regulatory or clinical pipeline is measured in years and millions.

## What it does

Takes any scientific text (paper draft, AI summary, literature review) and produces a per-claim verification report grounded in the cited source's full text — not just abstracts. Current positioning: honest evidence grounding with explicit abstention diagnostics, plus narrow deterministic numeric checks where applicable.

## Demo numbers (real outputs from three AI-for-science tools)

| tool | claims | passage found | supported | partially | unsupported | not addressed | numeric checks | cost |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Edison Scientific (TREM2) | 21 | 14 | 3 | 1 | 0 | 17 | 2 | $1.25 |
| Sakana AI Scientist v2 | 13 | 2 | 0 | 0 | 0 | 13 | 0 | $0.28 |
| AnswerThis (lactate) | 19 | 8 | 1 | 1 | 0 | 17 | 0 | $0.88 |
| **Total** | **53** | **24** | **4** | **2** | **0** | **47** | **2** | **$2.41** |

Use these as real-output evidence, not as a contradiction-catching claim: the current benchmark found 6 supported/partial claims and abstained on 47.

## How it works

1. **Extract** — LLM pulls verifiable claims with author/year anchors from free-form text
2. **Resolve** — OpenAlex + CrossRef → DOI for each citation
3. **Fetch full text** — chain: OA URL → PMC → Unpaywall → abstract fallback (zero-cost public APIs)
4. **Chunk + select** — IMRAD section-aware chunking; BM25 picks relevant passages or reports `no_passage_found`
5. **Verify** — full-text passage comparison against the claim, with cited-source provenance
6. **Numeric check** — deterministic OR/CI and p-value/CI checks (Python, no LLM in the comparison step)
7. **Report** — claim-by-claim verdict with full provenance trail (every step hashed, cached, costed)

Engineering choices that matter: prompt-cached system prompts, structured logging, mypy --strict, 200+ unit tests, F1=0.94 on SciFact dev split (verifier-only; locked test split untouched).

Controlled canary cases live under `benchmarks/canary/` for demoing weak resolution, contradictions, narrow numeric inconsistencies, and retraction checks without mixing seeded controls into real-output metrics.

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
