# Elicit Systematic Review (Premium) — GLP-1 RA / MACE evidence

**Tool**: [Elicit](https://elicit.com) (Systematic Review mode, Premium tier)
**Run config**: source = Clinical trials, format = General Review
**Query**: *"In adults with type 2 diabetes, what is the effect of GLP-1 receptor agonists compared to placebo on major adverse cardiovascular events (MACE)?"*
**Fetch date**: 2026-05-10
**Pipeline re-run**: 2026-05-12 (post-fix); pre-fix archived under [`_archive_pre_fix/`](_archive_pre_fix/)

## What this benchmark measures

End-to-end verification of an Elicit Premium Systematic Review output (~36KB report, 23 references, 290 inline citation markers spanning 5 cardiovascular outcome trials: LEADER, EXSCEL, SUSTAIN 6, AMPLITUDE-O, SOUL). The Premium tier promises tighter screening (8 inclusion criteria, full-text screening on 45 candidate papers, structured data extraction). This benchmark tests whether that promise translates to higher claim-vs-source faithfulness than Elicit's free-tier Report mode (see `elicit_psilocybin/README.md`).

## Source files

- `input.txt` — text extracted from the Elicit Systematic Review PDF via `pymupdf` (text layer, no OCR or LLM paraphrase). 35,849 chars, 290 inline `[N]` citation markers, 23-entry numbered References section with DOIs.
- `meta.json` — provenance metadata (query, tier, format, source filter, fetch date).
- `report.json` — post-fix pipeline output (36 claims, $0.63, 0 silent failures, 7 `unverifiable`).
- `provenance.jsonl` — append-only step log with hashes + tokens per stage.
- `fetch_traces.jsonl` — per-attempt fetch chain log. Reveals NEJM / Circulation / JAMA / AHA all served 403 on the PDF endpoint, forcing 16/36 claims onto abstract fallback.
- `_archive_pre_fix/report.json` — pre-fix pipeline output (46 claims, $1.46). Do not cite.
- `_archive_pre_fix/provenance.jsonl` — pre-fix step log.
- `Elicit - *.pdf` — raw Elicit export, **committed** for end-to-end reproducibility (re-run pymupdf on this PDF to regenerate `input.txt` byte-identically).
- `run_log.txt` — runner stdout/stderr (debug only, not committed).

## Run command

```bash
python .cache/run_benchmark.py benchmarks/real_outputs/elicit_glp1_mace
```

## Headline numbers — post-fix run (2026-05-12)

The current pipeline distinguishes "source contradicts" (`unsupported`) from "source is silent" (`not_addressed`) and "pipeline could not access full text for a numeric claim" (`unverifiable`).

| Metric | Post-fix (2026-05-12) | Pre-fix (2026-05-10, archived) |
|---|---:|---:|
| Claims extracted | **36** | 46 |
| Citation found rate | 91.7% | 93.5% |
| Fulltext verified | 20 (55.6%) | 26 (56.5%) |
| Supported | **19** (52.8%) | 30 (65.2%) |
| Partially supported | 2 (5.6%) | 5 (10.9%) |
| Unsupported | **0** (0%) | 10 (21.7%) |
| Not addressed | 8 (22.2%) | 1 (2.2%) |
| **Unverifiable** | **7** (19.4%) | n/a (pre-fix) |
| Numeric checks run | 9 | 13 |
| Numeric inconsistencies flagged | 0 | 0 |
| Silent failures (rule violation) | **0** | — |
| Total cost | $0.63 | $1.46 |

The 10 pre-fix `unsupported` verdicts were a mix of (a) the 1 fulltext-confirmed EXSCEL discontinuation error, (b) 4 resolver-mismatch artifacts (Gerstein 2023 → CKM scientific statement), and (c) 5 abstract-only paywall cases that the post-fix pipeline correctly downgrades to `unverifiable` or routes to `not_addressed`. The 7 `unverifiable` verdicts all carry `unverifiable_reason="numeric_claim_abstract_only"`; `fetch_traces.jsonl` confirms each one hit a 403 on the publisher PDF endpoint.

### Diagnostic fields (post-fix)

| Field | Value |
|---|---:|
| `abstract_only_verdicts` (legacy) | 16 |
| `fulltext_success_rate` | 55.6% |
| `unverifiable_by_reason` | `numeric_claim_abstract_only`: 7 |
| `not_addressed_breakdown.no_source` | 0 |
| `not_addressed_breakdown.paywall` | 7 |
| `not_addressed_breakdown.no_passage` | 0 |
| `not_addressed_breakdown.claim_absent` | 1 |
| `fetch_attempts_by_method` | abstract_fallback: 16, oa_url_pdf: 9, pmc: 7, unpaywall_pdf: 4 |
| `fetch_failures_by_reason` | publisher_html_unknown: 16, oa_url_pdf_failed: 15, unpaywall_pdf_failed: 15, europepmc_no_oa: 12, europepmc_pdf_failed: 8, publisher_html_blocked: 4, unpaywall_no_oa: 1 |

## Pre-fix decomposition of the 10 `unsupported` verdicts (archived)

The pre-fix run produced 10 `unsupported` verdicts. The current pipeline emits 0 `unsupported` on this same input because of two changes: (a) prompt Clause A now reserves `unsupported` for explicit contradictions, routing silence to `not_addressed`; (b) the helper downgrades numeric claims on abstract-only evidence to `unverifiable`. The pre-fix decomposition is retained below as the historical record motivating those fixes.

| Subgroup (pre-fix) | Count | What the current pipeline does |
|---|---:|---|
| Real Elicit attribution error | 1 | Same — would still surface as `unsupported` or `partially_supported` (fulltext access available). |
| Resolver fuzzy-match (Gerstein 2023 → CKM statement) | 4 | Resolver bug still present; verdicts now route to `unverifiable` because the mis-resolved paper's abstract triggers the numeric-claim guard. |
| Abstract-only paywall false negatives | 5 | Now correctly emitted as `unverifiable` with `unverifiable_reason="numeric_claim_abstract_only"`. |

### Pre-fix validated example (archived)

| Field | Value |
|---|---|
| Elicit claim | *"In EXSCEL, the overall discontinuation rate was up to 45%."* |
| Cited DOI | `10.1056/NEJMoa1612917` (Holman 2017, EXSCEL primary, NEJM) |
| What the paper actually says | "14,187 patients (96.2%) completed the trial." Mean percentage of time on study regimen was 76.0% (exenatide) and 75.0% (placebo). |
| Why the claim fails | A 45% discontinuation rate is incompatible with 96.2% trial completion. Elicit may have conflated study-regimen adherence (~25% off-regimen time) with overall trial discontinuation, or hallucinated the specific 45% figure. |
| Verification depth | fulltext (BM25 selected the relevant passage) |

This single case is the only one where the verifier had access to the cited paper's full text AND the claim was contradicted. It is the only claim in this run for which we can defensibly say "Elicit's claim conflicts with the source it cited."

## Comparison to Elicit Report mode (post-fix runs, 2026-05-12)

Both runs on the post-fix pipeline:

| Metric | psilocybin (Report mode, free tier) | GLP-1 MACE (Systematic Review, Premium) |
|---|---:|---:|
| total_claims | 43 | 36 |
| supported % | 23.3% | **52.8%** |
| partially_supported % | 39.5% | 5.6% |
| unsupported % | 0% | 0% |
| not_addressed % | 16.3% | 22.2% |
| unverifiable % | 20.9% | 19.4% |
| numeric_checks_run | 9 | 9 |
| numeric_inconsistencies | 0 | 0 |
| citation_found_rate | 100% | 91.7% |
| silent_failures | 0 | 0 |
| total_cost | $0.75 | $0.63 |

Premium Systematic Review tier produces a higher `supported` rate (52.8% vs 23.3%) on this comparison. N=1 per tier, so this is not a generalizable benchmark — but the gap is consistent with what the Premium tier promises (full-text screening + structured data extraction).

## Honesty disclosures

- **N=1 query, single Elicit session.** Re-running the same query produces a different output due to LLM stochasticity. The extractor surfaced 36 claims on the post-fix run vs 46 on the pre-fix — claim count is itself non-deterministic.
- **Resolver fuzzy-match limitation persists.** Elicit's bibliography lists `10.1161/CIRCULATIONAHA.122.063716` (Gerstein 2023 AMPLITUDE-O dose-response) but the resolver still fuzzy-matches `10.1161/cir.0000000000001186` (an AHA scientific statement on CKM syndrome). In the pre-fix pipeline, this dragged 4 verdicts to `unsupported`. In the post-fix pipeline, the mis-resolved abstract triggers the numeric-claim guard so verdicts route to `unverifiable` instead — the wrong-paper bug is masked, not fixed. Tracked as a follow-up resolver hardening item.
- **Abstract-only paywall coverage.** 16/36 claims fell back to abstract because publisher PDF endpoints returned 403 (NEJM / Circulation / JAMA Cardiol / AHA). The 7 numeric claims among them are correctly emitted as `unverifiable`; the rest are qualitative claims judged at abstract depth.
- **Selection bias in validation.** Only fulltext-verified claims are eligible for an "Elicit error" verdict. With 20/36 claims at fulltext depth and 0 `unsupported` verdicts, there is no fulltext-grounded Elicit error in this specific post-fix generation — that does not mean Elicit was error-free on this query; the EXSCEL 45% discontinuation claim from the pre-fix run did not re-emerge from the extractor.
- **Numeric checks ran on 9 claims, none flagged inconsistent.** Every numeric tuple Elicit reported in the narrative had the point estimate inside the CI and the p-value consistent with the CI. Strong positive signal for Elicit Premium numeric coherence.

## Reproduction

```bash
# 1. Save the Elicit Systematic Review PDF to this directory.
# 2. Extract text via pymupdf:
python -c "import pymupdf; doc = pymupdf.open('benchmarks/real_outputs/elicit_glp1_mace/Elicit*.pdf'); open('benchmarks/real_outputs/elicit_glp1_mace/input.txt','w',encoding='utf-8').write('\\n'.join(p.get_text() for p in doc))"

# 3. Run pipeline:
python .cache/run_benchmark.py benchmarks/real_outputs/elicit_glp1_mace
```

The pipeline is deterministic up to LLM-output stochasticity; the bibliography parser, resolver, fulltext fetcher, and BM25 passage selector are pure-Python or HTTP-only. Numbers above will not match a re-run exactly because LLM extract/verify outputs vary.
