# Real-Tool Benchmark Summary

> **Cite numbers ONLY from the current-runs table directly below.** The
> archived aggregate table further down conflates three epistemically
> distinct verdicts — `unsupported` (source contradicts), `not_addressed`
> (source is silent), and `unverifiable` (pipeline could not access full
> text) — and must not be cited as evidence of current pipeline behaviour.
> 3 of the 6 inputs have been re-run on the current pipeline; the
> remaining 3 are pending.

## Current runs

| tool | claims | supported | partially | unsupported | not_addressed | unverifiable | citation_found_rate | fulltext_verified | cost |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Elicit Report mode (psilocybin / TRD) | 43 | 10 | 17 | 0 | 7 | 9 | 100.0% | 8 | $0.75 |
| Elicit Systematic Review (GLP-1 / MACE) | 36 | 19 | 2 | 0 | 8 | 7 | 91.7% | 20 | $0.63 |
| AnswerThis (lactate ISF PK) | 19 | 6 | 8 | 1 | 4 | 0 | 100.0% | 12 | $0.25 |
| _(Edison TREM2, Elicit PD-1 NSCLC, Sakana AI Scientist — re-run pending)_ | — | — | — | — | — | — | — | — | — |

### Three highlighted current-run effects

**Psilocybin** — silent-failure elimination. 0 silent failures across 43
claims · prior baseline 15/57 = 26% (numeric claims on paywalled NEJM /
Nature papers where the verifier emitted confident `supported` /
`unsupported` from the abstract alone).

- 0 silent failures: the helper downgrades all 9 numeric-on-abstract
  claims to `unverifiable` with `unverifiable_reason="numeric_claim_abstract_only"`.
- 0 `unsupported` verdicts · prior baseline 17. The prompt's
  contradicts-vs-silent split reclassifies silence as `not_addressed`;
  the helper handles the numeric subset.
- 34 of 43 claims fall back to abstract because NEJM / Nature / Lancet /
  AJP serve paywall HTML on the PDF endpoint; `fetch_traces.jsonl`
  shows the exact per-step failure reasons.
- Replay infrastructure: [`scripts/replay_psilocybin_kpi.py`](../../scripts/replay_psilocybin_kpi.py).

**AnswerThis** — claim-coverage. 15 of 19 claims confidently classified
(79%) at $0.25 · prior baseline 3/25 (12%) at $0.47, with 22/25 routed
to `not_addressed`. Two compounding effects:

- Better fetch chain raises fulltext success rate to 63%, giving the
  verifier something to ground on.
- The prompt's contradicts-vs-silent split means the verifier no longer
  dumps anything-it-cannot-find-in-the-abstract into `not_addressed` —
  when the fulltext contains the assertion, the verifier emits `supported`.

**GLP-1 MACE** — Premium systematic-review auditability. 36 claims at
$0.63 with 0 silent failures, 91.7% citation resolution, and 20 full-text
verifications. The 7 `unverifiable` verdicts are all explicit
`numeric_claim_abstract_only` cases rather than confident abstract-only
claims.

## Archived runs (do not cite)

The original aggregate table is preserved below for diff/replay purposes
and is not evidence of current pipeline behaviour. The per-run archived
outputs sit under each benchmark's `_archive_pre_fix/` directory; see
[`_archive_README.md`](_archive_README.md) for context.

<details>
<summary>Click to expand archived aggregate table (DO NOT CITE)</summary>

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

The canary is a controlled four-claim input designed to exercise contradiction detection, retraction flagging, numeric inconsistency, and weak-resolution diagnostics. Its results are not pooled with the real-tool aggregate above. The canary `report.json` has not yet been re-run on the current pipeline.

## Reading the table

- **claims**: total verifiable claims extracted from the input
- **supported / partially_supported / unsupported / not_addressed / unverifiable**: full-text verification verdicts (the pipeline distinguishes `unverifiable` — pipeline could not access full text — from `not_addressed` — source is silent — and `unsupported` — source explicitly contradicts)
- **citation_found_rate**: fraction of claims where the cited source was resolved (title-match ≥ 15%)
- **fulltext_verified**: claims that received full-text BM25 passage selection and verification
- **retracted_sources**: cited papers flagged as retracted via CrossRef `update-to`
- **numeric_checks_run**: claims whose extracted assertions formed an OR/CI or p-value/CI tuple
- **numeric_inconsistencies_flagged**: subset of `numeric_checks_run` where the deterministic check failed
- **cost**: total Anthropic API spend (claude-sonnet-4-6, prompt-cached system prompts)

## Claim Transparency (CTran)

Independent of verdict correctness, CTran measures whether each claim's `report.json` entry contains enough evidence for a human auditor to trace the verdict. A claim is *transparent* when `source_passages` is non-empty OR `evidence_quality` is in `{abstract_only, quoted_passage, title_only, passages_searched_no_quote}`.

CTran aggregate: 65.9% transparent across 135 claims · prior baseline 48.9% (+17pp). Recomputation on the current pipeline is pending; the `unverifiable` verdict's explicit `unverifiable_reason` provenance and the per-attempt `fetch_traces.jsonl` give auditors strictly more signal than the baseline measurement included.
