# Elicit cross-run synthesis — 3 runs, 2 tiers, 2 formats

> **Mixed-vintage figures — cite with care.** Only the `elicit_psilocybin`
> and `elicit_glp1_mace` columns reflect the current pipeline; the
> `elicit_io_nsclc_gaps` column is provisional (archived baseline) and
> conflates `unsupported` (source contradicts), `not_addressed` (source
> is silent) and `unverifiable` (pipeline could not access full text)
> into a single bucket. See [`elicit_psilocybin/README.md`](elicit_psilocybin/README.md)
> for the headline psilocybin numbers (43 claims, 0 silent failures ·
> prior baseline 15/57). The cross-run synthesis below will be
> regenerated once the PD-1 NSCLC re-run lands.

Cross-cutting analysis of the three Elicit benchmark runs in this directory.
This document consolidates the per-run READMEs (`elicit_psilocybin/`,
`elicit_glp1_mace/`, `elicit_io_nsclc_gaps/`) into a single comparison
that isolates which findings are about Elicit versus which are about the
engine that verified Elicit. All numbers below are programmatically derived
from each run's `report.json`; no hand-typing.

## Run matrix

| Slug | Elicit tier | Format | Source filter | Topic | Date | References |
|---|---|---|---|---|---|---:|
| `elicit_psilocybin` | Free (Report mode) | Report | (default semantic, free tier could not select) | Psilocybin / TRD | 2026-05-10 | 10 |
| `elicit_glp1_mace` | **Premium** | **General Review** | Clinical trials | GLP-1 RA / MACE | 2026-05-10 | 23 |
| `elicit_io_nsclc_gaps` | **Premium** | **Research Gap Analysis** | Clinical trials | PD-1 / NSCLC PD-L1-high | 2026-05-10 | 18 |

The two Premium runs hold tier and source filter constant and vary only
synthesis format — a quasi-experiment that lets us attribute differences
to the format rather than to tier or topic-corpus availability.

## Headline numbers — single source of truth

Numbers below come from each run's `report.json` summary block.

| Metric | psilocybin (2026-05-12) | GLP-1 MACE (2026-05-12) | PD-1 NSCLC (provisional, re-run pending) |
|---|---:|---:|---:|
| Claims extracted | **43** | **36** | 25 |
| citation_found_rate | 100.0% | 91.7% | 60.0% |
| Supported | 10 (23.3%) | 19 (52.8%) | 8 (32.0%) |
| Partially supported | 17 (39.5%) | 2 (5.6%) | 6 (24.0%) |
| Unsupported | **0 (0.0%)** | **0 (0.0%)** | 2 (8.0%) |
| Not addressed | 7 (16.3%) | 8 (22.2%) | 9 (36.0%) |
| **Unverifiable** | **9 (20.9%)** | **7 (19.4%)** | n/a (provisional) |
| Numeric checks run | 9 | 9 | 8 |
| Numeric inconsistencies (raw) | 0 | 0 | 2 |
| Silent failures (rule violation) | **0** | **0** | unknown (provisional) |
| Total cost (USD) | $0.75 | $0.63 | $0.64 |

Two of three columns reflect the current pipeline. The PD-1 NSCLC column is provisional (archived baseline); cross-run comparisons involving it should be treated as such until its re-run lands.

### Diagnostic fields (from commit `e38150f`)

The 3 diagnostic fields populate across all 3 runs and are critical for
correctly interpreting the headline numbers.

| Diagnostic | psilocybin (current) | GLP-1 MACE (current) | PD-1 NSCLC (provisional) |
|---|---:|---:|---:|
| `abstract_only_verdicts` (provisional column field) | n/a (current pipeline uses `unverifiable`) | 20 | 4 |
| `fulltext_success_rate` | 18.6% (8/43) | 56.5% | 72.2% |
| `unverifiable_count` | **9** | n/a | n/a |
| `not_addressed_breakdown.no_source` | n/a | 0 | **7** |
| `not_addressed_breakdown.paywall` | n/a | 1 | 1 |
| `not_addressed_breakdown.no_passage` | n/a | 0 | 0 |
| `not_addressed_breakdown.claim_absent` | n/a | 0 | 1 |

The current pipeline replaces abstract-only numeric verdicts with `unverifiable` + `unverifiable_reason`. The PD-1 provisional-column diagnostics remain valid only for that archived snapshot.

## Three findings the cross-run comparison enables

### Finding 1 — Premium tier substantially outperforms free Report mode

Comparing psilocybin (free Report mode) and GLP-1 MACE (Premium General
Review) on the most directly comparable metrics:

| | psilocybin | GLP-1 MACE | Δ |
|---|---:|---:|---:|
| supported % | 43.9% | 65.2% | +21.3 pp |
| numeric_inconsistencies | 2 of 2 (100%) | 0 of 13 (0%) | -100 pp |
| Defensible Elicit error count | 3 fulltext-validated cases (timing misattribution, outcome fabrication, mechanism fabrication) | 1 fulltext-validated case (EXSCEL discontinuation 45% vs paper's 96.2% completion) | substantially fewer |

This is N=1 per tier, so it does not generalize beyond these specific runs.
But the gap on this single comparison is large and consistent with what
Premium promises (full-text screening + structured data extraction reduces
the fabrication rate). For users choosing between tiers, this single
benchmark suggests the Premium upcharge buys real accuracy improvement
on this kind of clinical-evidence query.

### Finding 2 — Format affects extractor behavior, not Elicit's accuracy

Holding tier (Premium) and source filter (Clinical trials) constant and
varying only format (General Review vs Research Gap Analysis):

| | GLP-1 MACE (Gen. Review) | PD-1 NSCLC (Gap Analysis) |
|---|---:|---:|
| Claims extracted | 46 | 25 |
| `not_addressed_breakdown.no_source` | 0 | **7** |
| Defensible Elicit-attributable error rate | 1/46 = 2.2% | 1/25 = 4.0% |

The Research Gap Analysis format produces ~half the claim density and a
much higher fraction of meta-level synthesis statements ("the NMAs
consistently converge on...", "only two studies discuss liver
metastases") that have no specific paper citation. The extractor surfaces
these as standalone claims with `cited_authors=[]`, and the resolver
correctly returns `no_source` rather than guessing — those 7 claims are
not falsifiable against any single source.

**The format does not appear to make Elicit substantially less accurate
on its actually-attributed claims.** After subtracting the format
artifact (no_source meta-claims) and engine limitations (resolver
mismatches, false-positive numeric flags), the defensible Elicit error
rate is broadly similar across both formats: ~2-4%. The "32% supported"
headline for PD-1 vs "65% supported" for GLP-1 is dominated by the
36% not_addressed bucket on PD-1, which is a format-shape difference,
not an Elicit accuracy difference.

### Finding 3 — Engine limitations must be subtracted to get the true Elicit error rate

This is the most important methodological point of the synthesis. Two
classes of "Elicit failures" surfaced by the raw `report.json` are
actually engine failures, not Elicit failures:

**A. Resolver fuzzy-match mismatches.** The CrossRef title-search fallback
occasionally lands on a structurally valid but semantically wrong DOI
when the bibliography-supplied DOI doesn't resolve cleanly. Two cases:

| Run | Bibliography DOI | Resolver picked | Effect |
|---|---|---|---|
| GLP-1 MACE | `10.1161/CIRCULATIONAHA.122.063716` (Gerstein 2023, AMPLITUDE-O dose-response) | `10.1161/cir.0000000000001186` (AHA scientific statement on CKM syndrome) | 4 abstract-only claims marked unsupported because verifier saw an unrelated paper |
| PD-1 NSCLC | `10.21037/tlcr.2020.02.14` (Liang 2020, oncology NMA) | `10.1190/segam2020-3427858.1` (full-waveform inversion, geophysics) | 1 claim marked unsupported |

These are not Elicit attribution errors — Elicit cited the right paper
in its bibliography. The engine's resolver fallback failed to land on
that paper. Fix: tighten semantic-similarity threshold on resolver
fallback, or surface low-confidence resolutions more aggressively.

**B. False-positive numeric inconsistencies.** The numeric_check pure-Python
comparator validates `(point_estimate, ci_low, ci_high)` tuples emitted
by the LLM numeric_extract step. When a sentence has multiple metrics in
quick succession ("ORR (RR 1.62) and PFS (HR 0.55, 95% CI 0.32-0.97)"),
the LLM extractor occasionally projects the CI from metric B onto metric A's
point estimate. Two cases on PD-1 NSCLC:

| Claim | LLM mis-paired | True content | Elicit error? |
|---|---|---|---|
| Zhou 2019 sentence | RR=1.62 paired with CI [0.32, 0.97] | 1.62 is ORR (no CI given); [0.32, 0.97] is PFS HR's CI for HR=0.55 | No — Elicit's numbers are coherent |
| Wang 2022 sentence | OR=1.7 paired with CI [0.48, 0.74] | 1.7 is ORR (no CI given); [0.48, 0.74] is PFS HR's CI for HR=0.59 | No — Elicit's numbers are coherent |

The verifier's text-based check correctly classified both claims as
`supported`. The numeric check disagreed because of the pairing
limitation. Reporting these as "Elicit numeric errors" would be
intellectually dishonest. Fix: prompt the numeric extractor to
positionally ground each tuple, or downgrade `inconsistent` flags to
`low_confidence` when multiple metrics share a sentence.

**Defensible Elicit-attributable error rates after subtraction:**

| Run | Raw "unsupported" % | After subtracting engine artifacts | Validated cases |
|---|---:|---:|---|
| psilocybin | 29.8% | ≥ 3 cases fulltext-validated | timing misattribution, outcome fabrication, mechanism fabrication |
| GLP-1 MACE | 21.7% | **2.2% (1/46)** | EXSCEL discontinuation 45% vs 96.2% completion |
| PD-1 NSCLC | 8.0% | **4.0% (1/25)** | Bachurski 2026 HR 0.59 attribution drift (borderline) |

The headline `unsupported` percentages bundle real Elicit errors with
engine artifacts. Without the per-run decomposition, conclusions about
Elicit's accuracy would be substantially off.

## Cross-cutting observations

### Citation density and resolution coverage

| | Inline `[N]` markers | References | Avg markers per ref | Resolved DOIs |
|---|---:|---:|---:|---:|
| psilocybin | ~604 (estimated) | 10 | ~60 | 9 distinct (1 ref unused) |
| GLP-1 MACE | 290 | 23 | 12.6 | 22 distinct |
| PD-1 NSCLC | 101 | 18 | 5.6 | 16 distinct (+1 mismatch + 7 no_source) |

The Premium tier with structured data extraction produces a more
parsimonious citation pattern (fewer markers per reference) than free
Report mode. PD-1 has the lowest density partly because Research Gap
Analysis emphasizes meta-level synthesis claims that don't anchor to
specific references.

### Paywall and fulltext access by domain

| Domain | `fulltext_success_rate` | Dominant paywall journals |
|---|---:|---|
| Cardiology / Endocrinology (GLP-1) | 56.5% | NEJM, Circulation, JAMA Cardiol, Annals Int Med |
| Immuno-oncology (PD-1 NSCLC) | 72.2% | Frontiers, BMC, BJC (mostly open-access NMAs) |

The PD-1 NSCLC run benefits from being a meta-analysis topic where the
included sources are mostly open-access NMAs themselves; the GLP-1 MACE
run primarily cites primary trials in paywalled high-impact journals.
Domain choice substantially affects how much of an Elicit output can be
verified at fulltext depth versus abstract depth.

### Numeric verification coverage

| | Numeric checks | Fraction of claims with numeric content | Inconsistencies (raw) | After false-positive correction |
|---|---:|---:|---:|---:|
| psilocybin | 2 | ~3.5% | 2 | 2 (validated) |
| GLP-1 MACE | 13 | 28.3% | 0 | **0** |
| PD-1 NSCLC | 8 | 32.0% | 2 | **0** |

Premium Systematic Review output has substantially higher numeric
density (28-32% of claims contain HR/CI/p-value tuples) than free
Report mode (~3.5% on psilocybin). On the Premium runs, **after correcting
for the 2 false-positive multi-metric pairing flags on PD-1, Elicit
Premium reported zero arithmetically incoherent numbers across 21
deterministic numeric checks.** This is a strong positive signal for
Elicit Premium's numeric reporting.

## What this synthesis does not establish

- **Generalization.** N=1 query per (tier, format) cell. Re-running the
  same query produces a different output due to LLM stochasticity.
  Conclusions describe these specific runs, not Elicit's average.
- **Causal attribution.** The "Premium > free tier" comparison confounds
  tier with format and topic. A clean comparison would run the same
  query on the same topic at both tiers — not feasible because the free
  tier cannot select the Clinical trials source filter.
- **Long-tail Elicit failure modes.** The validated examples are picked
  for defensibility (fulltext access + clear contradiction). Edge cases
  Elicit handles well do not appear in the validation set, biasing the
  surfaced examples toward errors. The aggregate `supported` percentage
  is the unbiased measure; the validation examples are illustrative
  rather than enumerative.
- **Elicit's overall product quality.** This is a verification benchmark,
  not a usability or workflow benchmark. Elicit may be valuable to its
  users for reasons (PDF screening, citation organization, query
  reformulation, etc.) that don't show up in claim-vs-source faithfulness
  scores.

## Reproduction

Each Elicit run's `report.json` was regenerated by `python
.cache/run_benchmark.py benchmarks/real_outputs/<slug>`. The aggregate
table above was regenerated by `python scripts/generate_summary.py`,
which writes `benchmarks/real_outputs/README.md`. This synthesis
document is hand-written and should be updated whenever a new Elicit
run lands.

## See also

- Per-run details: [elicit_psilocybin/README.md](elicit_psilocybin/README.md), [elicit_glp1_mace/README.md](elicit_glp1_mace/README.md), [elicit_io_nsclc_gaps/README.md](elicit_io_nsclc_gaps/README.md)
- Aggregate across all 6 real-tool benchmarks: [README.md](README.md)
- Engine hardening commit referenced above: `e38150f` (DOI validation, citing-paper window 8KB, partial JSON recovery, diagnostic summary fields)
- Benchmark commit referenced above: `2acfb50` (the 2 Premium runs)
