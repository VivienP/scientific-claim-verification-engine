# Elicit Report mode — psilocybin / TRD evidence

**Tool**: [Elicit](https://elicit.com) (web UI, Report mode export)
**Queries**:
1. *"What is the evidence for psilocybin in treatment-resistant depression?"*
2. *"What are the differences in symptom reduction for depression, anxiety, or treatment-resistant conditions across Phase II trials of psilocybin-assisted therapy?"*

**Fetch date**: 2026-05-10
**Free-tier limitation**: the user could not select "Clinical Trials" as the source filter; default semantic search across 138M papers (Semantic Scholar + OpenAlex) was used.

## What this benchmark measures

Verification depth on a single Elicit Report-mode systematic-review-style answer. Elicit positions itself as evidence-based with structured citations. We extract the answer text, parse its numbered References section into DOIs, and verify each extracted claim against the cited source's abstract or full text via the standard pipeline (extract → resolve → fetch_fulltext → BM25 passage select → verify).

Only the Report-mode export from query 1 is committed as `input.txt` (32,687 chars). Both raw Elicit PDF exports (query 1 and query 2) are now committed alongside `input.txt` for end-to-end reproducibility — anyone cloning the repo can re-run pymupdf on the same PDFs and regenerate `input.txt` byte-identically.

## Source files

- `input.txt` — text extracted from `Elicit - Psilocybin and Treatment-Resistant Depression - Report.pdf` via `pymupdf` (text layer, no OCR or LLM paraphrase). Includes the full Abstract, Flow Diagram, Screening criteria, Data extraction, Results tables, Synthesis discussion, and 10 numbered References with DOIs.
- `meta.json` — provenance metadata (queries, fetch date, source URL, license note).
- `report.json` — pipeline output (2026-05-12): 43 claims, $0.75, **0 silent failures** · prior baseline 15/57 = 26%.
- `provenance.jsonl` — append-only step log with hashes and tokens per stage.
- `fetch_traces.jsonl` — per-attempt fetch chain log. NEJM / Nature / Lancet / AJP all serve paywall HTML on the PDF endpoint, forcing 34/43 claims onto abstract fallback; the helper downgrades 9 of those numeric claims to `unverifiable` with `unverifiable_reason="numeric_claim_abstract_only"`.
- `_archive_pre_fix/` — archived prior verifier output (57 claims, $1.16). Do not cite; see [`../_archive_README.md`](../_archive_README.md).
- `Elicit - *.pdf` — raw Elicit exports, **committed** for end-to-end reproducibility. `*Sources.txt` (Elicit's auxiliary citation list) remains gitignored as it is not used by the verification pipeline.

## Run command

```bash
python .cache/run_benchmark.py benchmarks/real_outputs/elicit_psilocybin
```

## Headline numbers (2026-05-12)

The verifier distinguishes "source contradicts" (`unsupported`) from "source is silent" (`not_addressed`) and "pipeline could not access full text for a numeric claim" (`unverifiable`).

| Metric | Current (2026-05-12) | Prior baseline (2026-05-10) |
|---|---:|---:|
| Claims extracted | **43** | 57 |
| Citation found rate | 100.0% | 100.0% |
| Fulltext verified | 9 (21%) | 18 (32%) |
| Supported | **10** | 25 |
| Partially supported | **17** | 15 |
| Unsupported | **0** | 17 |
| Not addressed | **7** | 0 |
| **Unverifiable** | **9** | n/a |
| Numeric checks run | 9 | 2 |
| Numeric inconsistencies flagged | 0 | 2 |
| Silent failures (rule violation) | **0** | **15** (26%) |
| Total cost | $0.75 | $1.16 |

The 17 baseline `unsupported` verdicts decompose into: 0 genuine contradictions, 7 silences (`not_addressed`), and 9 abstract-only numeric claims downgraded to `unverifiable` on the current pipeline. The 9 `unverifiable` claims all hit NEJM / Lancet / AJP paywall; `fetch_traces.jsonl` confirms each returned paywall HTML on the PDF endpoint.

### Resolved DOI distribution

All 57 claims in the archived baseline resolved to one of the 9 distinct papers Elicit cited (10 references in the bibliography; ref [3] Meikle 2025 was cited fewer times):

| DOI | Paper | Claims |
|---|---|---:|
| 10.1056/NEJMoa2206443 | Goodwin 2022 (NEJM) | 16 |
| 10.1016/S2215-0366(16)30065-7 | Carhart-Harris 2016 (Lancet Psychiatry) | 12 |
| 10.1007/s00213-017-4771-x | Carhart-Harris 2017 (Psychopharmacology) | 7 |
| 10.1176/appi.ajp.20231063 | Aaronson 2025 (Am J Psychiatry) | 6 |
| 10.1038/s41386-023-01648-7 | Goodwin 2023 (Neuropsychopharmacology) | 4 |
| 10.1038/s41591-022-01744-z | Daws 2022 (Nature Medicine) | 4 |
| 10.1038/s41598-017-13282-7 | Carhart-Harris 2017 (Sci Reports) | 4 |
| 10.1177/20451253251377187 | Meikle 2025 (Therapeutic Adv Psych) | 3 |
| 10.1016/j.medj.2024.01.005 | Rosenblat 2024 (i Medicina) | 1 |

## Pipeline behaviour exercised by this input

This fixture exercises two PDF-derived bibliography paths and one
extractor path:

- **Bibliography preprocessing** ([src/bibliography.py](../../../src/bibliography.py)).
  PDF text from pymupdf interleaves page-number-only lines and wraps
  long URLs across line boundaries. `_clean_pdf_artefacts` strips
  digit-only lines and collapses URL line-wraps with a negative
  lookahead protecting next-entry boundaries. Coverage on this
  fixture: 10/10 correct DOIs.
- **Bibliography → resolver wiring** ([.cache/run_benchmark.py](../../../.cache/run_benchmark.py)).
  `resolve_citations` accepts a `bibliography` kwarg; without one the
  resolver falls back to fuzzy CrossRef search on each claim's text and
  lands on unrelated papers (e.g. Goodwin 2022 NEJM → *"Growth Hormone
  is Useless in IVF: The Largest Randomized Controlled Trial"*). The
  benchmark runner passes the parsed bibliography to keep resolution
  pinned.
- **Streaming extractor** ([src/extract.py](../../../src/extract.py)).
  Report-mode density (~57 claims for a 32K-char input) overflows a
  hardcoded 4096 output-token ceiling, so the extractor uses
  `messages.stream()` and exposes a `max_output_tokens` kwarg.

## Manually validated examples

Three claims where the verifier and the cited paper genuinely disagree (selected from the 17 `unsupported` and 15 `partially_supported` verdicts; only fulltext-verified cases are eligible — abstract-only verdicts cannot distinguish a missing detail from a contradicted detail).

### Example 1 — timing misattribution (UNSUPPORTED)

| Field | Value |
|---|---|
| Elicit claim | *"QIDS-SR16 scores decreased from baseline 18.9±3 to 8.8±6.2 at 1 week post-treatment (p<0.001)"* |
| Cited DOI | `10.1038/s41598-017-13282-7` (Carhart-Harris 2017, *Scientific Reports*) |
| What the paper actually says | Two distinct score comparisons: (a) baseline 18.9±3 → 5-weeks post-treatment 10.9±4.8, p<0.001; (b) week prior to pre-treatment scan 16.9±5.1 → day of post-treatment scan 8.8±6.2, p<0.001. |
| Why the claim fails | Elicit pairs the baseline value (18.9±3) with the post-scan value (8.8±6.2) and labels the timepoint as "1 week post-treatment." Neither pair exists in the paper at that timepoint. |
| Verification depth | fulltext (2 BM25-selected passages quoted in `report.json`) |

### Example 2 — outcome fabrication (UNSUPPORTED)

| Field | Value |
|---|---|
| Elicit claim | *"Suicidal ideation was significantly reduced at 1 and 2 weeks post-treatment in one study"* |
| Cited DOI | `10.1007/s00213-017-4771-x` (Carhart-Harris 2017, *Psychopharmacology*) |
| What the paper actually says | Reports significant reductions in depressive symptoms, anxiety (STAI), and anhedonia (SHAPS) at 1 week post-treatment. Suicidal ideation is not reported as an outcome measure at 1 or 2 weeks. |
| Why the claim fails | The specific outcome (suicidal ideation reduction) at the specific timepoints (1 and 2 weeks) is not assessed in the cited paper. |
| Verification depth | fulltext |

### Example 3 — fabricated mechanism (PARTIALLY SUPPORTED)

| Field | Value |
|---|---|
| Elicit claim | *"Reductions in depressive symptoms at 5 weeks were predicted by the quality of the acute psychedelic experience, with significant relationships between the 'USB' factor and symptom changes"* |
| Cited DOI | `10.1007/s00213-017-4771-x` (Carhart-Harris 2017, *Psychopharmacology*) |
| What the paper actually says | First half supported: the paper does state reductions at 5 weeks were predicted by acute psychedelic experience quality. The "USB" factor and its specific relationships are not mentioned. |
| Why the claim is partial | Elicit fabricates a specific mechanism ("USB factor") that does not appear in the cited paper, while the surrounding claim is supported. |
| Verification depth | fulltext |

## Honesty disclosures

- **Selection bias in validation examples**: only fulltext-verified cases are shown above. The 3 validated examples all use open-access sources where the verifier saw the full text. The 9 abstract-only numeric claims from NEJM / Lancet / AJP are emitted as `unverifiable` (not `unsupported`) and so are not eligible for the validation set — the pipeline itself flags the evidence gap rather than issuing a confident verdict.
- **N=1 query, single user session**: this benchmark reflects one Elicit Report-mode generation. Results are not generalizable beyond this run. Re-running the same prompt would produce a different answer due to the underlying LLM stochasticity.
- **Pipeline still has limitations**: our own e2e benchmark on a hand-annotated lactate-ISF review currently sits at 16/25 verdict agreement (64%) — the verifier and resolver have known weaknesses outside the bibliography path. The validated examples above are robust because they isolate single-paper verification with fulltext access; broader claims about Elicit's overall accuracy are not warranted from this single benchmark.
