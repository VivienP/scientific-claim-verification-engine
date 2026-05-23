# Dogfood run: Valsci paper (Edelman & Skolnick 2025)

**Source paper**: Edelman B, Skolnick J. *Valsci: an open-source, self-hostable literature review utility for automated large-batch scientific claim verification using large language models.* BMC Bioinformatics. 2025 May 28;26:140.

**DOI**: `10.1186/s12859-025-06159-4`
**PMC ID**: `PMC12121171`
**PMID**: `40437377`
**License**: CC-BY-NC-ND 4.0 (BMC Open Access)

## Why this paper

Generalization test for the verification pipeline. The lactate-ISF benchmark has a structural ceiling at 16/25 (paywall-bound). This run validates pipeline behaviour on:

- A **different domain** (bioinformatics/ML rather than physiology/PK)
- A **different citation style** (numbered `[1]`, `[2]` rather than author-year)
- A **competitor's own paper** — Valsci is one of the closest related works (cited in [README.md](../../../README.md) related work section). Running our pipeline on Valsci's paper enables a direct comparison narrative.

## Source files

- `source.pdf` — canonical PDF (**not committed**; gitignored). Fetch from BMC: [`https://bmcbioinformatics.biomedcentral.com/counter/pdf/10.1186/s12859-025-06159-4.pdf`](https://bmcbioinformatics.biomedcentral.com/counter/pdf/10.1186/s12859-025-06159-4.pdf). License: CC-BY-NC-ND 4.0 — non-commercial redistribution permitted, no modifications. Save locally as `benchmarks/real_papers/valsci_brice_2025/source.pdf` to reproduce the benchmark run.
- `input.txt` — text transcription of `source.pdf` body sections (no Appendix A/B). What our pipeline ingests.
- `report.json` — pipeline output.
- `provenance.jsonl` — append-only step log.
- `aar_scorecard.md` — AAR metrics.
- `oracle.json` — manual ground-truth DOIs per externally-cited claim; scored by `scripts/score_against_oracle.py`.

## input.txt provenance

Raw text extracted from the canonical PDF (PDF text layer, not LLM-paraphrased). Sections included:

- Title, authors, affiliation
- Abstract (Background / Results / Conclusions)
- Background (full)
- Related work (full)
- Implementation (full)
- Results (Hallucination rate / Processing speed / Precision-recall-F1 / Cochrane validation)
- Discussion (full)
- Limitations (full)
- Future directions (full)
- Conclusion (full)
- Abbreviations
- References (15 numbered entries with DOIs)

**Excluded from input.txt** (deliberately):

- **Appendix A — LLM prompts** (e.g. "You are an expert at converting scientific claims into strategic literature search queries..."). These are text templates, not scientific claims, and would correctly trigger our prompt-injection guard. Test prompt-injection defense in a separate dedicated benchmark.
- **Appendix B — screenshot captions** (no claims).
- **Acknowledgements / Funding / Declarations** (no scientific claims).

The user-uploaded PDF is the canonical source. The text was transcribed page-by-page preserving inline citation markers `[N]` exactly as printed.

## Run command

```python
from src.pipeline import PipelineConfig, run_pipeline
from src.report import build_report
import os, uuid, pathlib

text = pathlib.Path("benchmarks/real_papers/valsci_brice_2025/input.txt").read_text()
config = PipelineConfig(api_key=os.environ["ANTHROPIC_API_KEY"])
verifications, steps = run_pipeline(text, config=config)
run_dir = build_report(
    str(uuid.uuid4()), text,
    [v.claim for v in verifications],
    {v.claim.claim_id: v.source for v in verifications},
    {v.claim.claim_id: v.result for v in verifications},
    steps,
)
print(f"Report: {run_dir}")
```

```bash
python scripts/aar_scorecard.py reports/runs/<report_id>
```

## Expected metrics

- Claims extracted: target 10-25. Body text contains 14 distinct citation markers (`[1]`, `[2]`, `[3, 11]`, `[4]`, `[5]`, `[6]`, `[7, 9]`, `[8]`, `[10]`, `[12]`, `[13]`, `[14]`, `[15]`) — multi-marker citations like `[3, 11]` and `[7, 9]` may yield separate claims via multi-source resolution.
- Citation found rate: target ≥ 80% (most refs are arXiv preprints with DOIs — should be highly resolvable via CrossRef/Europe PMC).
- AAR scorecard: PCov, PSnd, CTran, AEff reported below.

## Hard floor

If citation found rate < 30% on a run, treat as a generalization regression and investigate before publishing.

## Headline numbers

**Report ID**: `5ebdee2a-c5a3-406e-b6ab-5bf4e3c80134`.

| Metric | Value |
|---|---|
| Claims extracted | 19 |
| External-citation found rate | 11/11 = **100%** |
| External-citation correct rate | 11/11 = **100%** (scored against `oracle.json`) |
| Multi-source aggregations fired | 2 (`[3, 11]` and similar multi-marker citations) |
| Total cost | $0.250 |

### Verdict distribution

| Verdict | Count |
|---|---:|
| supported | 0 |
| partially_supported | 13 |
| unsupported | 0 |
| not_addressed | 6 |

0 `supported` is honest, not regression: most resolved sources are arXiv preprints whose full text is not in our PMC/Unpaywall paths, so the verifier sees abstracts only and emits `partially_supported`. The citing-paper guard blocks self-recursion verdicts (a claim cannot cite the paper that contains it), so no false-positive `supported` from comparing the claim against its own paper.

### AAR scorecard

| Metric | Value | Detail |
|---|---|---|
| **PCov** | 100.00% | 19/19 claims with provenance |
| **PSnd** | 100.00% | 84/84 steps with valid hashes |
| **CTran** | 47.37% | 9/19 claims with concrete evidence |
| **AEff** | 76.45 | claims per USD |

## Pipeline behaviour exercised by this input

This fixture exercises four code paths that earlier benchmarks did not:

- **Numbered-citation bibliography parsing** ([src/bibliography.py](../../../src/bibliography.py)). The parser handles three numbered formats: `[N]` alone on a line (LaTeX), `[N] Author` inline (preprint), and `N. Author` inline (BMC/journal). This fixture uses the third.
- **Citing-paper DOI guard** ([src/resolve.py](../../../src/resolve.py)). The resolver auto-detects the citing paper's DOI via `detect_citing_paper_doi` and rejects any candidate matching it. Self-references in the claim text cannot resolve to the paper that contains them.
- **arXiv direct-search fallback** ([src/clients/arxiv.py](../../../src/clients/arxiv.py)). Inserted between `bib_pmid` and `crossref_title` in the resolver chain. Scores candidates with the 50/30/15 title/author/year blend. ML preprints often share titles with later journal extensions; arXiv-direct lookup pins the correct preprint instead of CrossRef's title-match.
- **Multi-source aggregation on marker-order primary**. `ResolvedSourceSet.primary()` walks `sources` in citation-marker order and returns the first `found=True`. When an author writes `[7, 9]`, ref [7] is the primary citation by textual intent — not the highest title-match score.

## Comparison to other benchmarks

| Tool | Claims | Cit-found | Fulltext-verified | Cost | PCov | CTran |
|---|---:|---:|---:|---:|---:|---:|
| Edison TREM2 | 20 | 85.0% | 12 (60%) | $0.38 | n/a | n/a |
| Sakana CompReg | 14 | 71.4% | 3 (21%) | $0.16 | n/a | n/a |
| AnswerThis lactate-ISF | 25 | 64.0% | 13 (52%) | $0.47 | n/a | n/a |
| **Valsci 2025 (this run)** | **19** | **100%** | **8 (42%)** | **$0.25** | **100%** | **47.4%** |

## Comparison to Valsci's own reported numbers

Valsci reports F1 = 0.761 on SciFact with GPT-4o, claim-by-claim verdicts on 500 SciFact claims. We are not comparing on the same task — Valsci verifies oracle SciFact claims against retrieved Semantic Scholar abstracts; we verify free-form claims-with-citations against the cited source. The two systems are complementary, not competing on this benchmark. The meaningful comparison is **what AAR metrics our pipeline produces on real audit work** (a metric Valsci does not publish): PCov 100%, PSnd 100%, CTran 47%, AEff 76.

## Lesson recorded for future validation runs

The first attempt at this run used a WebFetch-extracted text (LLM-paraphrased HTML→text). That was the wrong standard: WebFetch inserts an LLM intermediary that paraphrases prose, which corrupts:

- Claim density (compresses multiple sentences into one)
- Verbatim quoted statements (alters wording our verifier needs to match against sources)
- Real positioning ("we ran our pipeline on the Valsci paper" requires *the paper*, not a summary)

**Standard for benchmark inputs**: raw PDF text layer or user-uploaded text. Never an LLM-mediated extraction.
