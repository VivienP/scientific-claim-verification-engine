# Elicit Systematic Review (Premium) — PD-1 inhibitors in PD-L1-high metastatic NSCLC

**Tool**: [Elicit](https://elicit.com) (Systematic Review mode, Premium tier)
**Run config**: source = Clinical trials, format = Research Gap Analysis
**Query**: *"What is the effectiveness of PD-1 inhibitor monotherapy compared to combination therapy as first-line treatment for PD-L1-high metastatic non-small cell lung cancer in terms of overall survival and progression-free survival?"*
**Fetch date**: 2026-05-10

## What this benchmark measures

Companion run to `elicit_glp1_mace/` — same Elicit Premium tier and same source filter (Clinical trials), but the synthesis format is changed from General Review to **Research Gap Analysis**. Goal: isolate the effect of synthesis format on claim-vs-source faithfulness while holding tier and source constant.

The Research Gap Analysis format produces a structurally different output: more meta-level summary statements ("the NMAs consistently converge on..."), more references to absent evidence ("only two studies explicitly discuss liver metastases"), and an explicit "Recommendations for Future Research" section. These structural differences directly affect what claims the extractor can produce and how the resolver can ground them.

## Source files

- `input.txt` — text reconstructed from the Elicit Systematic Review PDF as visible inline during the working session. The original PDF (`Elicit - PD-1 Inhibitors in NSCLC Treatment - Report.pdf`) is now committed alongside; re-running pymupdf on it should produce a byte-similar input.txt (formatting may differ slightly from the inline-reconstructed version). 32,066 chars, 101 inline `[N]` citation markers, 18-entry numbered References section with DOIs.
- `Elicit - *.pdf` — raw Elicit Systematic Review export, **committed** for end-to-end reproducibility.
- `meta.json` — provenance metadata.
- `report.json` — pipeline output for the current run (25 claims, $0.64 cost).
- `provenance.jsonl` — append-only step log.
- `run_log.txt` — runner stdout/stderr (debug only, not committed).

## Run command

```bash
python .cache/run_benchmark.py benchmarks/real_outputs/elicit_io_nsclc_gaps
```

## Headline numbers — 2026-05-10

| Metric | Value | Comment |
|---|---:|---|
| Claims extracted | 25 | Less dense than GLP-1 (46) — Research Gap Analysis = more meta-level claims, fewer numeric ones |
| Citation found rate | 60.0% | 15/25 resolved — the 10 unresolved are a mix of meta-claims with no citation + 1 resolver mismatch |
| Fulltext verified | 13 (52%) | NMAs and reviews are typically open-access in Frontiers / BMC / BJC → PMC pulls them |
| Supported | 8 (32%) | |
| Partially supported | 6 (24%) | |
| Unsupported | 2 (8%) | 1 real Elicit attribution issue + 1 resolver mismatch (see below) |
| **Not addressed** | **9 (36%)** | Dominated by `no_source` (7 meta-claims without citation) |
| Numeric checks run | 8 | |
| **Numeric inconsistencies flagged** | **2** | **Both are false positives — see disclosure below** |
| Total cost | $0.64 | |

### Diagnostic fields (commit `e38150f`)

| Field | Value | Interpretation |
|---|---:|---|
| `abstract_only_verdicts` | 4 | Mostly fulltext access via PMC (NMAs are typically open-access) |
| `fulltext_success_rate` | **72.2%** | Higher than GLP-1 (56.5%) because oncology NMAs are often in open-access journals |
| `not_addressed_breakdown.no_source` | **7** | **The dominant signal** — see analysis below |
| `not_addressed_breakdown.paywall` | 1 | One paywalled abstract did not address the claim |
| `not_addressed_breakdown.no_passage` | 0 | BM25 always found relevant passages when fulltext was available |
| `not_addressed_breakdown.claim_absent` | 1 | Passage found but verifier judged claim absent from source |

The breakdown is the most informative diagnostic on this run. Without it, we'd see "9 not_addressed" and infer engine paywall problems. With it, we see **7/9 not_addressed = `no_source`**, which is a structural feature of the Research Gap Analysis format (meta-level synthesis claims with no specific citation), not an engine failure.

## What the 7 `no_source` claims actually are

These are not Elicit attribution errors — they're meta-level synthesis statements from the Research Gap Analysis output that the extractor surfaced as standalone claims. Examples:

| # | Claim text (truncated) |
|--:|---|
| 1 | "Both PD-1 inhibitor monotherapy and PD-1 inhibitor–chemotherapy combination therapy both significantly improve OS and PFS relative to chemotherapy alone in PD-L1-high NSCLC..." |
| 2 | "Combination therapy consistently demonstrates superior PFS, with HRs ranging from 0.55 to 0.81 across analyses." |
| 3 | "No statistically significant OS difference between combination therapy and monotherapy has been identified in the PD-L1-high subgroup..." |
| 4 | "Combination therapy is associated with higher rates of grade 3–5 treatment-related adverse events." |
| 5 | "The NMAs consistently converge on the finding that combination therapy improves PFS but does not significantly improve OS in PD-L1-high disease." |
| 6 | "Only two studies explicitly discuss liver metastases as a clinical modifier." |
| 7 | "Brain metastases subgroups are reported in only a few analyses." |

Each of these summarises a pattern across multiple cited papers without attributing the synthesis to a single source. The extractor (which expects per-claim citations) treats them as orphan claims with `cited_authors=[]` and `cited_year=None`, and the resolver correctly returns "no source found" rather than guessing. **This is the right behavior — these claims are not falsifiable against a single paper.** A future improvement is to mark them as `claim_type="meta_synthesis"` upstream and skip resolver attempts.

## Decomposition of the 2 `unsupported` verdicts

| Subgroup | Count | Verification depth | Interpretation |
|---|---:|---|---|
| Real Elicit attribution issue (likely) | 1 | fulltext | Bachurski 2026 cited for a specific PFS HR (0.59, 95% CI 0.43–0.81) that the verifier could not locate in the paper's full text |
| Resolver mismatch (wrong paper) | 1 | abstract | Liang 2020 oncology paper resolved instead to a **geophysics paper on full-waveform inversion** (DOI `10.1190/segam2020-3427858.1`). This is a CrossRef fuzzy-match failure — the actual Liang 2020 paper has DOI `10.21037/tlcr.2020.02.14`, but my resolver picked a similarly-titled seismic-imaging paper. |

So the **defensible Elicit-attributable error rate is 1/25 = 4%**, with that 1 case requiring further manual verification.

### Manually validated example — possible Elicit attribution drift

| Field | Value |
|---|---|
| Elicit claim | *"Bachurski et al. (2026) found that combination therapy improves PFS (HR 0.59, 95% CI 0.43–0.81) but does not clearly improve OS compared with monotherapy in PD-L1-high disease."* |
| Cited DOI | `10.12775/qs.2026.51.68252` (Bachurski 2026, *Quality in Sport*) |
| Verifier verdict | unsupported |
| Verifier explanation | "The passages do not contain the specific HR of 0.59 (95% CI 0.43–0.81) for PFS cited in the claim. The passages do confirm that combination therapy does not clearly improve OS compared with monotherapy in PD-L1-high disease, but the specific PFS hazard ratio reported in the paper (from the NMA by Hu...)" |
| Verification depth | fulltext |
| Defensibility | The HR 0.59 likely originates from a different NMA that Bachurski cites (probably Pathak 2020 or Wang 2022, both of which report HR 0.59 for PFS in chemo-ICI vs ICI). Elicit may have synthesized the HR by attributing it to Bachurski's review of those NMAs rather than directly to the source NMA. This is a citation-chain drift, not an outright fabrication. |

This is a **borderline case** — the HR 0.59 exists in the literature Bachurski reviewed, but the direct attribution to Bachurski rather than to the upstream NMA is technically incorrect.

## Critical disclosure: the 2 numeric inconsistencies are FALSE POSITIVES

This is the most important honesty section in this README.

`numeric_inconsistencies_flagged: 2` in the summary, but **both are caused by a known limitation of the engine's numeric pairing logic, not by Elicit reporting wrong numbers.**

### False positive #1 — Zhou 2019 sentence

| Field | Value |
|---|---|
| Elicit claim | *"Zhou et al. (2019) found that combination therapy was superior for ORR (RR 1.62) and PFS (HR 0.55, 95% CI 0.32–0.97); trend toward improved OS but not significant (HR 0.76, P=0.184)."* |
| Source paper | Zhou 2019, J Immunother Cancer (`10.1186/s40425-019-0600-6`) |
| Verifier text verdict | **supported** — "exactly matching the claim" |
| Numeric check verdict | **inconsistent** — "OR/CI inconsistent: OR=1.62 outside CI [0.32, 0.97]" |
| What actually happened | The CI [0.32, 0.97] belongs to the **PFS HR (0.55)**, not the **ORR RR (1.62)**. Elicit correctly reports "ORR (RR 1.62)" without giving a CI for the RR, and separately "PFS (HR 0.55, 95% CI 0.32–0.97)". The engine's numeric extractor mis-paired RR=1.62 with the next-available CI in the same sentence. |
| Source paper's actual numbers | RR 1.62 (95% CI 1.18–2.23) for ORR; HR 0.55 (95% CI 0.32–0.97) for PFS — both correctly reported by Elicit. |

### False positive #2 — Wang 2022 sentence

| Field | Value |
|---|---|
| Elicit claim | *"Wang et al. (2022) found that chemo-ICI significantly improved ORR (OR 1.7) and PFS (HR 0.59, 95% CI 0.48–0.74) vs ICI alone; no significant OS difference (HR 0.82, 95% CI 0.6–1.1)."* |
| Source paper | Wang 2022, Br J Cancer (`10.1038/s41416-022-01832-4`) |
| Verifier text verdict | **supported** — "exactly matching all three values and directions stated in the claim" |
| Numeric check verdict | **inconsistent** — "OR/CI inconsistent: OR=1.7 outside CI [0.48, 0.74]" |
| What actually happened | Same pattern. CI [0.48, 0.74] belongs to PFS HR (0.59), not ORR OR (1.7). The numeric extractor pairs the OR with the next CI in the sentence regardless of which point estimate the CI semantically belongs to. |

### Root cause and remediation

The engine's `numeric_extract` LLM call returns a list of `(metric_type, point_estimate, ci_low, ci_high)` tuples per sentence, and the deterministic `numeric_check` pure-python comparator validates each tuple in isolation. The pairing is correct when each tuple stays internal to one metric, but in compact narrative sentences with multiple metrics — common in systematic-review summaries — the LLM occasionally projects the CI from metric B onto the point estimate from metric A.

**This is a documented engine limitation, not an Elicit failure.** Reporting "Elicit had 2 numeric inconsistencies" in a public-facing summary of this benchmark would be intellectually dishonest. The accurate framing is: **on 8 numeric checks Elicit ran on this output, 0 of Elicit's reported numbers were arithmetically wrong; 2 false-positive flags from the engine reflect its multi-metric pairing limitation.**

A future fix would either (a) prompt the numeric extractor with explicit positional grounding (require it to point each tuple at a specific phrase span), or (b) downgrade `inconsistent` flags to `low_confidence` when multiple metrics share a sentence. Tracked as a follow-up.

## Comparison to GLP-1 MACE (same tier, different format)

| Metric | GLP-1 MACE (General Review) | PD-1 NSCLC (Research Gap Analysis) |
|---|---:|---:|
| Claims extracted | 46 | 25 |
| supported % | 65.2% | 32.0% |
| not_addressed % | 2.2% | 36.0% |
| `not_addressed_breakdown.no_source` | 0 | **7** |
| citation_found_rate | 93.5% | 60.0% |
| Numeric checks | 13 | 8 |
| Numeric inconsistencies (raw) | 0 | 2 |
| Numeric inconsistencies (true Elicit errors) | 0 | **0** |
| Resolver mismatches | 1 (4 claims affected) | 1 (1 claim affected) |
| Defensible Elicit attribution errors | 1/46 (2.2%) | 1/25 (4.0%) |

The Research Gap Analysis format produces ~half the claims of General Review on the same tier, and a much higher fraction of those claims are meta-level summaries that don't attribute to a single source (28% of all extracted claims = `no_source`). After accounting for engine limitations (resolver mismatches, false-positive numeric flags, meta-claim no_source), **the Elicit-attributable error rate is broadly similar across both formats (~2-4%)** — Elicit Premium does not appear to be substantially less accurate on Research Gap Analysis than General Review on this single comparison.

## Honesty disclosures

- **N=1 query, single Elicit session.** Re-running would produce a different output.
- **2 numeric inconsistencies are FALSE POSITIVES.** See the dedicated section above. The headline number `numeric_inconsistencies_flagged: 2` in `report.json` would be misleading without that context.
- **Resolver fuzzy-match limitation surfaces here too.** The Liang 2020 → geophysics paper mismatch is the same class of CrossRef-fallback failure as the Gerstein 2023 → CKM scientific statement mismatch in the GLP-1 run. Same fix applies.
- **Bachurski 2026 attribution case is borderline.** The HR 0.59 exists in the broader NSCLC NMA literature; Elicit's attribution to Bachurski's review may be a synthesis-vs-direct-citation distinction rather than a fabrication. Verdict reported as "unsupported" but reasonable people could differ.
- **PD-1 PDF was originally inline-reconstructed.** The `input.txt` for this run was reconstructed from the inline PDF text content provided in the working session, before the original PDF was saved to disk. The PDF is now committed alongside (`Elicit - PD-1 Inhibitors in NSCLC Treatment - Report.pdf`), but the committed `input.txt` reflects the inline-reconstructed text — re-running pymupdf on the saved PDF may produce minor formatting differences (whitespace, page-break handling) without changing the substantive content. The `report.json` numbers in this directory derive from the committed `input.txt`, not from a fresh pymupdf extraction.
- **Selection bias in validation.** The 1 fulltext-verified Elicit error candidate (Bachurski 2026) is the only `unsupported` verdict that could be defensibly attributed to Elicit on this run after subtracting the resolver mismatch. The 6 `partially_supported` verdicts and the 4 abstract-only `partially_supported` cases were not manually inspected for this README.

## Reproduction

```bash
# 1. Save the Elicit Systematic Review PDF to this directory.
# 2. Extract text via pymupdf (replaces the reconstructed input.txt with authentic extraction):
python -c "import pymupdf; doc = pymupdf.open('benchmarks/real_outputs/elicit_io_nsclc_gaps/Elicit*.pdf'); open('benchmarks/real_outputs/elicit_io_nsclc_gaps/input.txt','w',encoding='utf-8').write('\\n'.join(p.get_text() for p in doc))"

# 3. Run pipeline:
python .cache/run_benchmark.py benchmarks/real_outputs/elicit_io_nsclc_gaps
```
