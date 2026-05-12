# Real-Tool Benchmark Summary

> **Numbers being recomputed (2026-05-12).** The pre-fix aggregate
> table below the banner conflated three epistemically distinct
> verdicts — `unsupported` (source contradicts), `not_addressed`
> (source is silent), and what is now `unverifiable` (pipeline could
> not access full text). After the 2026-05-12 verifier fix
> (commits [`992303e`](../../README.md) through
> [`3211b23`](../../README.md)), 2 of the 6 inputs have been re-run on
> the post-fix pipeline. The remaining 4 are in the Track C queue.
>
> Cite numbers ONLY from the post-fix table directly below.

## Post-fix runs (2026-05-12, Tracks A+D+F+G+I)

| tool | claims | supported | partially | unsupported | not_addressed | unverifiable | citation_found_rate | fulltext_verified | cost |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Elicit Report mode (psilocybin / TRD) | 43 | 10 | 17 | 0 | 7 | 9 | 100.0% | 9 | $0.75 |
| AnswerThis (lactate ISF PK) | 19 | 6 | 8 | 1 | 4 | 0 | 100.0% | 12 | $0.25 |
| _(Edison TREM2, Elicit GLP-1 MACE, Elicit PD-1 NSCLC, Sakana AI Scientist — re-run pending)_ | — | — | — | — | — | — | — | — | — |

### Two distinct kinds of post-fix win

**Psilocybin** is the silent-failure-elimination story. On the pre-fix
run, 15 of 57 confident verdicts (26%) were silent failures: numeric
claims on paywalled NEJM / Nature papers where the verifier emitted
confident `supported` / `unsupported` from the abstract alone. Post-fix:
- 0 silent failures — the helper downgrades all 9 numeric-on-abstract
  claims to `unverifiable` with `unverifiable_reason="numeric_claim_abstract_only"`.
- 17 false-`unsupported` verdicts → 0 (Track G's prompt rewrite reclassifies
  silence as `not_addressed`; A2+F1 helper handles the numeric subset).
- 34 of 43 claims fall back to abstract because NEJM / Nature / Lancet /
  AJP serve paywall HTML on the PDF endpoint; `fetch_traces.jsonl`
  (Track I1) shows the exact per-step failure reasons.
- Replay infrastructure: [`scripts/replay_psilocybin_kpi.py`](../../scripts/replay_psilocybin_kpi.py).

**AnswerThis** is the claim-coverage story. The pre-fix pipeline punted
on 22 of 25 claims (88% `not_addressed`); the post-fix pipeline actually
verifies 15 of 19 claims (79% confidently classified as
`supported` / `partially_supported` / `unsupported`). Cost dropped from
$0.47 to $0.25. Two compounding effects:
- Better fetch chain (Tracks D1 + I1) raised fulltext success rate to 63%,
  giving the verifier something to ground on.
- Track G's prompt rewrite means the verifier no longer dumps anything-it-
  cannot-find-in-the-abstract into `not_addressed` — when the fulltext
  contains the assertion, the verifier emits `supported`.

## Pre-fix runs (archived — do not cite)

The original aggregate table previously appeared here. It is preserved
below for diff/replay purposes but should not be used as evidence of
current pipeline behavior. The per-run pre-fix outputs are under each
benchmark's `_archive_pre_fix/` directory; see
[`_archive_README.md`](_archive_README.md) for context.

<details>
<summary>Click to expand pre-fix aggregate table (DEPRECATED)</summary>

| tool | claims | supported | partially_supported | unsupported | not_addressed | citation_found_rate | fulltext_verified | retracted_sources | numeric_checks_run | numeric_inconsistencies_flagged | cost |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Edison Scientific Literature (TREM2) | 20 | 3 | 2 | 1 | 14 | 85.0% | 12 | 0 | 1 | 0 | $0.38 |
| Sakana AI Scientist v2 (CompReg) | 14 | 0 | 0 | 0 | 14 | 71.4% | 3 | 0 | 0 | 0 | $0.16 |
| AnswerThis (lactate ISF PK) | 25 | 1 | 1 | 1 | 22 | 64.0% | 13 | 0 | 0 | 0 | $0.47 |
| Elicit Report mode (psilocybin / TRD) | 57 | 25 | 15 | 17 | 0 | 100.0% | 18 | 0 | 2 | 2 | $1.16 |
| Elicit Systematic Review (GLP-1 / MACE) | 46 | 30 | 5 | 10 | 1 | 93.5% | 26 | 0 | 13 | 0 | $1.46 |
| Elicit Systematic Review (PD-1 NSCLC gaps) | 25 | 8 | 6 | 2 | 9 | 60.0% | 13 | 0 | 8 | 2 | $0.64 |
| **Total** | **187** | **67** | **29** | **31** | **60** | **84.5%** | **85** | **0** | **24** | **4** | **$4.27** |

</details>

## Canary (seeded controls — not real-tool evidence)

| tool | claims | supported | partially_supported | unsupported | not_addressed | citation_found_rate | fulltext_verified | retracted_sources | numeric_checks_run | numeric_inconsistencies_flagged | cost |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Canary suite | 4 | 0 | 0 | 1 | 3 | 50.0% | 2 | 0 | 0 | 0 | $0.03 |

The canary is a controlled four-claim input designed to exercise contradiction detection, retraction flagging, numeric inconsistency, and weak-resolution diagnostics. Its results are not pooled with the real-tool aggregate above. The canary `report.json` has not yet been re-run on the post-fix pipeline.

## Reading the table

- **claims**: total verifiable claims extracted from the input
- **supported / partially_supported / unsupported / not_addressed / unverifiable**: full-text verification verdicts (the post-fix pipeline distinguishes `unverifiable` — pipeline could not access full text — from `not_addressed` — source is silent — and `unsupported` — source explicitly contradicts)
- **citation_found_rate**: fraction of claims where the cited source was resolved (title-match ≥ 15%)
- **fulltext_verified**: claims that received full-text BM25 passage selection and verification
- **retracted_sources**: cited papers flagged as retracted via CrossRef `update-to`
- **numeric_checks_run**: claims whose extracted assertions formed an OR/CI or p-value/CI tuple
- **numeric_inconsistencies_flagged**: subset of `numeric_checks_run` where the deterministic check failed
- **cost**: total Anthropic API spend (claude-sonnet-4-6, prompt-cached system prompts)

## Claim Transparency (CTran)

Independent of verdict correctness, CTran measures whether each claim's `report.json` entry contains enough evidence for a human auditor to trace the verdict. A claim is *transparent* when `source_passages` is non-empty OR `evidence_quality` is in `{abstract_only, quoted_passage, title_only, passages_searched_no_quote}`.

The pre-Phase-A.2 CTran numbers (135 claims, 65.9% aggregate, +17pp vs 48.9% baseline) were measured on pre-fix runs. Post-fix CTran has not yet been recomputed; expected to improve further because the new `unverifiable` verdict carries explicit `unverifiable_reason` provenance, and `fetch_traces.jsonl` adds per-attempt fetch reasoning that any auditor can read.
