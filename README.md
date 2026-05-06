# Scientific Claim Verification Engine

![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue)
![License](https://img.shields.io/badge/license-Apache%202.0-green)
![Tests](https://img.shields.io/badge/tests-203-brightgreen)

Auditable claim-by-claim evidence review for scientific text.

The pipeline turns free-form scientific text — paper drafts, literature
reviews, AI-generated summaries — into a source-grounded verification report,
one verdict per cited claim, with full provenance. The current scope is
deliberately conservative: this is an honest evidence-grounding verifier, not
a broad automatic inconsistency detector.

## What It Does

The pipeline takes free-form scientific text and writes:

- `report.json`: one record per extracted claim, with verdict, cited source,
  retrieval status, evidence quality, source quotes when found, and numeric
  checks when applicable.
- `provenance.jsonl`: append-only provenance for each extraction, resolution,
  verification, numeric, and aggregation step, including token and cache data.

Supported verdicts are `supported`, `unsupported`, `partially_supported`, and
`not_addressed`. The verifier keeps abstention explicit instead of forcing weak
evidence into a contradiction.

## Real-Output Benchmark

Committed reports under `benchmarks/real_outputs/` currently cover three
AI-for-science outputs:

| tool | claims | supported | partially_supported | unsupported | not_addressed | citation_found_rate | fulltext_verified | retracted_sources | numeric_checks_run | numeric_inconsistencies_flagged | cost |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Edison Scientific Literature (TREM2) | 20 | 3 | 2 | 1 | 14 | 85.0% | 12 | 0 | 1 | 0 | $0.38 |
| Sakana AI Scientist v2 (CompReg) | 14 | 0 | 0 | 0 | 14 | 71.4% | 3 | 0 | 0 | 0 | $0.16 |
| AnswerThis (lactate ISF PK) | 25 | 1 | 1 | 1 | 22 | 64.0% | 13 | 0 | 0 | 0 | $0.47 |
| **Total** | **59** | **4** | **3** | **2** | **50** | **72.9%** | **28** | **0** | **1** | **0** | **$1.01** |

Numbers regenerated 2026-05-06 from `benchmarks/real_outputs/SUMMARY.md`. Re-run via `python scripts/check_summary_alignment.py`.

## Canary Controls

`benchmarks/canary/` contains a clearly labeled seeded input for demo and
regression testing. It is separate from the real-output benchmark and should not
be merged into public aggregate counts. The canary exercises:

- weak source resolution
- contradicted claim
- deterministic p-value/CI inconsistency
- retraction check path

## Quick Start

```bash
pip install -e ".[dev]"
export ANTHROPIC_API_KEY=sk-...
python examples/sample_run.py
python scripts/show_report.py
```

Default input: `benchmarks/real_outputs/edison_trem2/input.txt`.

Expected shape (cache-cold, ~3-4 min, ~$0.25):

```text
Extracted 21 claims.
Report written to: reports/runs/{report_id}/
Full-text retrieval methods: abstract_fallback=N, oa_url_pdf=N, unpaywall_pdf=N
```

The exact retrieval-method counts vary run-to-run with extraction
non-determinism and OA availability. See
`benchmarks/real_outputs/edison_trem2/report.json` for the most recent
committed run (12 passage-found, 9 fulltext-unavailable, $0.24).

Run the canary controls explicitly:

```bash
python examples/sample_run.py benchmarks/canary/input.txt
python scripts/show_report.py
```

## Pipeline

```text
input text
  -> extract_claims()              # LLM: citation-anchored claims
  -> resolve_citations()           # OpenAlex + CrossRef fallback
  -> fetch_fulltext()              # OA URL -> PMC -> Unpaywall -> abstract fallback
  -> chunk_paper()                 # deterministic IMRAD section chunks
  -> select_passages()             # BM25; empty when no lexical passage match
  -> verify_claim_fulltext()       # LLM verifier over selected passages or abstract
  -> run_numeric_check()           # deterministic OR/CI and p-value/CI checks
  -> build_report()                # report.json + provenance.jsonl
```

Resolution and retrieval diagnostics are first-class fields:

- `source.title_match_score`: lexical title/abstract overlap with the query.
- `source.resolution_low_confidence`: true when source matching is weak.
- `verification.retrieval_status`: `passage_found`, `no_passage_found`, or
  `fulltext_unavailable`.
- `verification.evidence_quality`: `quoted_passage`, `abstract_only`, or
  `no_evidence`.

## Public API

Core calls:

```python
claims, extract_step = extract_claims(text)
sources, resolve_steps = resolve_citations(claims)
fulltext, method = fetch_fulltext(source)
chunks = chunk_paper(source.doi or claim.claim_id, fulltext)
passages = select_passages(claim.claim_text, chunks, top_k=3)
result, steps = verify_claim_fulltext_with_numeric(claim, source, passages)
run_dir = build_report(report_id, text, claims, sources, results, steps)
```

Data models are frozen dataclasses in `src/models.py`: `Claim`,
`ResolvedSource`, `VerificationResult`, `ProvenanceStep`, and `PaperChunk`.

## Evaluation

```bash
python scripts/eval_scifact.py --split dev
```

SciFact is used as a verifier regression baseline. The current F1 = 0.94 number
is verifier-only: `scripts/eval_scifact.py` builds oracle claims and oracle
abstract sources, then calls `verify_claim()`. It does not measure the full
extract -> resolve -> retrieve -> verify pipeline.

The SciFact test split is locked and must not be used for prompt selection or
tuning.

## Development

```bash
python -m pytest -v
python -m mypy --strict src
python -m ruff check src tests scripts examples
```

## Known Limitations

- Extraction currently requires explicit author/year citation anchors. Numeric bracket citations work only when the references are present in the input.
- `similarity_score` is still year-proximity; use `title_match_score` and `resolution_low_confidence` for source-match diagnostics.
- BM25 is lexical. If no chunk shares claim tokens, the pipeline now reports `no_passage_found` instead of sending arbitrary first chunks to the verifier.
- LLM `confidence` is self-reported and not calibrated. Treat
  `retrieval_status`, `evidence_quality`, and source quotes as higher-signal evidence diagnostics.
- Numeric coverage is intentionally narrow: OR/CI consistency and p-value/CI null-crossing checks only.
- Each claim is checked against its cited source, not against the whole literature.

## Contact

Questions, feedback, or interesting failure modes:
[@PerrelleVivien](https://x.com/PerrelleVivien) on X.

## License

Apache 2.0. See [LICENSE](LICENSE).
