# Real-Tool Benchmark Summary

Aggregate verification outcomes across three distinct AI-for-science tools, run against the post-audit pipeline (title-match resolution gate, BM25 honest-fallback, retrieval_status diagnostics, p-value/CI numeric check). Numbers below are programmatically extracted from each `report.json`'s `summary` block — no hand-typing.

Generated 2026-05-01 (Edison TREM2 re-run after Caveat 1 fix: the runner-side 4000-char-per-passage cap was replaced with token-budget-aware selection inside `bm25_selector.py` (default 6000 tokens/passage, BM25-ranked, sentence-boundary truncation when a single chunk overshoots). The previous cap had stripped supporting evidence on long-passage claims, masking real `supported` verdicts as `partially_supported`. Sakana and AnswerThis rows reflect the prior run; their inputs were not re-verified.

| tool | claims | supported | partially | unsupported | not_addressed | passage_found | no_passage | fulltext_unavail | low_conf_resolution | retracted | numeric_run | numeric_inconsistent | cost |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Edison Scientific Literature (TREM2) | 20 | 3 | 2 | 1 | 14 | 12 | 0 | 8 | 3 | 0 | 1 | 0 | $0.38 |
| Sakana AI Scientist v2 (CompReg) | 17 | 1 | 0 | 0 | 16 | 3 | 0 | 14 | 0 | 0 | 0 | 0 | $0.32 |
| AnswerThis (lactate ISF PK) | 23 | 2 | 0 | 1 | 20 | 11 | 0 | 12 | 0 | 0 | 0 | 0 | $0.23 |
| **Total** | **60** | **6** | **2** | **2** | **50** | **26** | **0** | **34** | **3** | **0** | **1** | **0** | **$0.93** |

### Canary (seeded controls — not real-tool evidence)

| tool | claims | supported | partially | unsupported | not_addressed | passage_found | no_passage | fulltext_unavail | low_conf | numeric_run | numeric_inconsistent | cost |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Canary suite | 4 | 0 | 0 | 1 | 3 | 2 | 0 | 2 | 0 | 0 | 0 | $0.03 |

The canary is a controlled four-claim input designed to exercise contradiction detection, retraction flagging, numeric inconsistency, and weak-resolution diagnostics. Its results are not pooled with the real-tool aggregate above.

## Per-tool detail

### Edison Scientific Literature (TREM2)

- **Tool:** Edison Scientific Literature
- **Agent / model:** Edison Literature agent (production)
- **Prompt:** What is the role of TREM2 in Alzheimer's microglia, and what are the key quantitative findings?
- **Fetch date:** 2026-04-13
- **Pipeline run:** Phase 1 (full-text) + Phase 2 (numeric checks: OR/CI + p-value/CI), token-aware passage selection
- **Verifiability status:** verifiable
- **Citation found rate:** 85.0%

### Sakana AI Scientist v2 (CompReg)

- **Tool:** Sakana AI Scientist v2
- **Agent / model:** AI Scientist v2 (BFTS, agentic tree search; ICLR 2025 workshop submission)
- **Prompt:** broad topic: compositional regularization for neural network generalization
- **Fetch date:** 2026-04-28
- **Pipeline run:** Phase 1 (full-text) + Phase 2 (numeric checks: OR/CI + p-value/CI)
- **Verifiability status:** verifiable
- **Citation found rate:** 52.9%

### AnswerThis (lactate ISF PK)

- **Tool:** AnswerThis
- **Agent / model:** AnswerThis (web UI literature review, manual paste)
- **Prompt:** Lactate pharmacokinetics in interstitial fluid during exercise
- **Fetch date:** 2026-04-28
- **Pipeline run:** Phase 1 (full-text) + Phase 2 (numeric checks: OR/CI + p-value/CI)
- **Verifiability status:** verifiable
- **Citation found rate:** 60.9%

## Reading the table

- **claims**: total verifiable claims extracted from the input
- **supported / partially / unsupported / not_addressed**: full-text verification verdicts
- **passage_found**: BM25 returned a relevant passage (full-text verification)
- **no_passage**: full text fetched but BM25 found zero token overlap with the claim
- **fulltext_unavail**: fell back to abstract-only or no source available
- **low_conf_resolution**: title-match score < 15% — likely wrong-paper resolution
- **retracted**: cited papers flagged as retracted via CrossRef `update-to`
- **numeric_run**: claims whose extracted assertions formed an OR/CI or p-value/CI tuple
- **numeric_inconsistent**: subset of `numeric_run` where the deterministic check failed
- **cost**: total Anthropic API spend (claude-sonnet-4-6, prompt-cached system prompts)

## Honest reading of the new aggregate

After the Caveat 1 fix (token-aware passage selection), Edison TREM2 now surfaces **3 supported / 2 partially_supported / 1 unsupported / 14 not_addressed**, vs. **0 / 4 / 0 / 17** under the 4000-char cap. The recovered `supported=3` matches the pre-cap baseline documented in the prior SUMMARY — confirming the cap was masking real positive verdicts on long-passage claims. Across the three tools, the post-fix aggregate stands at **6 supported / 2 partially / 2 unsupported / 50 not_addressed**. The new diagnostics still distinguish `passage_found` from `fulltext_unavail` from `no_passage_found`, eliminating the prior misleading `verification_depth=fulltext` label on claims where BM25 had silently fallen back to irrelevant chunks.
