# Real-Tool Benchmark Summary

Aggregate verification outcomes across three distinct AI-for-science tools.
Numbers below are programmatically extracted from each `report.json`'s `summary` block — no hand-typing.
Generated 2026-04-29.

| tool | claims | supported | partially | unsupported | not_addressed | fulltext_verified | retracted | numeric_run | numeric_inconsistent | cost |
|---|---|---|---|---|---|---|---|---|---|---|
| Edison Scientific Literature (TREM2) | 21 | 3 | 1 | 0 | 17 | 14 | 0 | 2 | 0 | $1.25 |
| Sakana AI Scientist v2 (CompReg) | 13 | 0 | 0 | 0 | 13 | 2 | 0 | 0 | 0 | $0.28 |
| AnswerThis (lactate ISF PK) | 19 | 1 | 1 | 0 | 17 | 8 | 0 | 0 | 0 | $0.88 |
| **Total** | **53** | **4** | **2** | **0** | **47** | **24** | **0** | **2** | **0** | **$2.41** |

## Per-tool detail

### Edison Scientific Literature (TREM2)

- **Tool:** Edison Scientific Literature
- **Agent / model:** Edison Literature agent (production)
- **Prompt:** What is the role of TREM2 in Alzheimer's microglia, and what are the key quantitative findings?
- **Fetch date:** 2026-04-13
- **Pipeline run:** Phase 1 (full-text) + Phase 2 (numeric OR/CI consistency)
- **Verifiability status:** verifiable
- **Citation found rate:** 71.4%

### Sakana AI Scientist v2 (CompReg)

- **Tool:** Sakana AI Scientist v2
- **Agent / model:** AI Scientist v2 (BFTS, agentic tree search; ICLR 2025 workshop submission)
- **Prompt:** broad topic: compositional regularization for neural network generalization
- **Fetch date:** 2026-04-28
- **Pipeline run:** Phase 1 (full-text) + Phase 2 (numeric OR/CI consistency)
- **Verifiability status:** verifiable
- **Citation found rate:** 76.9%

### AnswerThis (lactate ISF PK)

- **Tool:** AnswerThis
- **Agent / model:** AnswerThis (web UI literature review, manual paste)
- **Prompt:** Lactate pharmacokinetics in interstitial fluid during exercise
- **Fetch date:** 2026-04-28
- **Pipeline run:** Phase 1 (full-text) + Phase 2 (numeric OR/CI consistency)
- **Verifiability status:** verifiable
- **Citation found rate:** 52.6%

## Reading the table

- **claims**: total verifiable claims extracted from the input
- **supported / partially / unsupported / not_addressed**: full-text verification verdicts
- **fulltext_verified**: how many claims had full-text passages available (vs. abstract-only fallback)
- **retracted**: cited papers flagged as retracted via CrossRef `update-to`
- **numeric_run**: claims whose extracted assertions formed an OR/CI triple, triggering the deterministic engine
- **numeric_inconsistent**: subset of `numeric_run` where the OR/CI consistency check failed
- **cost**: total Anthropic API spend (input + cached + output, claude-sonnet-4-6 pricing)

## Honest reading of the aggregate

Across 53 claims, the pipeline surfaced 4 supported, 2 partially supported, **0 unsupported**, and 47 not_addressed. The numeric engine fired on 2 claims, flagged 0 inconsistencies. See `docs/quality_audit_2026-04-29.md` for an analysis of why this aggregate is structurally honest but launch-flat, and which pipeline concerns most plausibly drive the 47/53 not_addressed rate.
