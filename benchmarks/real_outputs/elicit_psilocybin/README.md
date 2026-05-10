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
- `report.json` — pipeline output for the current run.
- `provenance.jsonl` — append-only step log with hashes and tokens per stage.
- `Elicit - *.pdf` — raw Elicit exports, **committed** for end-to-end reproducibility. `*Sources.txt` (Elicit's auxiliary citation list) remains gitignored as it is not used by the verification pipeline.

## Run command

```bash
python .cache/run_benchmark.py benchmarks/real_outputs/elicit_psilocybin
```

## Headline numbers — 2026-05-10 (run #2, after bibliography fixes)

| Metric | Run #1 (broken) | Run #2 (fixed) |
|---|---:|---:|
| Claims extracted | 65 | 57 |
| Citation found rate | 66.2% | **100.0%** |
| Resolution low-confidence | 4 | **0** |
| Fulltext verified | 33 | 18 |
| Supported | 5 | **25** |
| Partially supported | 2 | **15** |
| Unsupported | 35 | 17 |
| Not addressed | 23 | **0** |
| Numeric checks run | 6 | 2 |
| Numeric inconsistencies flagged | 6 | 2 |
| Total cost | $1.38 | $1.16 |

### Verdict distribution (run #2)

| Verdict | Count | % of resolved |
|---|---:|---:|
| supported | 25 | 43.9% |
| partially_supported | 15 | 26.3% |
| unsupported | 17 | 29.8% |
| not_addressed | 0 | 0.0% |

### Resolved DOI distribution (run #2)

All 57 claims resolved to one of the 9 distinct papers Elicit cited (10 references in the bibliography, ref [3] Meikle 2025 was cited fewer times):

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

## What changed between run #1 and run #2

Run #1 surfaced two bugs that systematically broke citation resolution on PDF-derived inputs:

**Bug R1 — runner did not pass bibliography to resolver** ([.cache/run_benchmark.py](../../../.cache/run_benchmark.py)). `resolve_citations` accepts an optional `bibliography` kwarg, but the benchmark runner called it without one. Without a parsed bibliography, the resolver fell back to fuzzy CrossRef search on each claim's text, landing on unrelated papers (e.g. claim about Goodwin 2022 NEJM resolved to *"Growth Hormone is Useless in IVF: The Largest Randomized Controlled Trial"* via the phrase "the largest randomized controlled trial"). Only 4/65 (6%) of run #1 resolutions matched one of the actual Elicit-cited papers.

**Bug R2 — pymupdf artefacts corrupted DOI extraction** ([src/bibliography.py](../../../src/bibliography.py)). PDF text extraction interleaves page-number-only lines and wraps long URLs across line boundaries. The original `_DOI_FIELD_RE` lookahead missed both:
- *Page numbers fused with DOIs*: `https://doi.org/10.1038/s41591-022-01744-z\n12\n` extracted as `10.1038/s41591-022-01744-z12` (corrupted).
- *URL-wrap drops*: `https://doi.or\ng/10.1177/20451253251377187` extracted as `None`.

Fix added a `_clean_pdf_artefacts` preprocessing step that strips digit-only lines and collapses URL line-wraps with a negative lookahead protecting next-entry boundaries. Coverage: 6/10 → **10/10** correct DOIs on this fixture.

Tests added:
- [`tests/unit/test_bibliography.py::TestParseBibliographyRobustness::test_pymupdf_page_numbers_stripped`](../../../tests/unit/test_bibliography.py)
- `test_url_line_wrap_collapsed`
- `test_url_wrap_does_not_eat_next_entry`
- `test_real_elicit_input_parses` (end-to-end on this fixture)

All 638 unit tests pass after the fix.

A separate fix migrated [`src/extract.py`](../../../src/extract.py) to streaming (`messages.stream()`) and exposed a `max_output_tokens` kwarg, because Report-mode density (~57 claims for a 32K-char input) overflows the prior hardcoded 4096 ceiling and triggers connection drops on the resulting long generations.

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

- **Selection bias in validation examples**: only fulltext-verified cases are shown above. The remaining 15 `unsupported` verdicts are abstract-only — when NEJM, Lancet Psychiatry, and other paywalled journals are cited, the verifier sees only the abstract. Specific numbers in the methods/results often correctly derive from the paper's full text but cannot be confirmed from the abstract alone, biasing those verdicts toward `unsupported`. These are pipeline limitations, not Elicit failures, and are excluded from the validation set.
- **N=1 query, single user session**: this benchmark reflects one Elicit Report-mode generation. Results are not generalizable beyond this run. Re-running the same prompt would produce a different answer due to the underlying LLM stochasticity.
- **Pipeline still has limitations**: our own e2e benchmark on a hand-annotated lactate-ISF review currently sits at 7/25 verdict agreement — the verifier and resolver have known weaknesses outside the bibliography path. The validated examples above are robust because they isolate single-paper verification with fulltext access; broader claims about Elicit's overall accuracy are not warranted from this single benchmark.
