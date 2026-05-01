# Scientific Claim Verification Engine

![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue)
![License](https://img.shields.io/badge/license-Apache%202.0-green)
![Tests](https://img.shields.io/badge/tests-203-brightgreen)

Auditable claim-by-claim evidence review for scientific text.

This project is a design-partner alpha for teams that need to turn AI-generated scientific drafts, literature reviews, and agent outputs into a source-grounded verification report. The current launch story is deliberately conservative: this is an honest evidence-grounding verifier, not yet a broad automatic
inconsistency detector.

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

| tool | claims | passage found | supported | partially | unsupported | not addressed | numeric checks | cost |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Edison Scientific, TREM2 | 21 | 12 | 0 | 4 | 0 | 17 | 1 | $0.24 |
| Sakana AI Scientist v2 | 17 | 3 | 1 | 0 | 0 | 16 | 0 | $0.32 |
| AnswerThis, lactate | 23 | 11 | 2 | 0 | 1 | 20 | 0 | $0.23 |
| **Total** | **61** | **26** | **3** | **4** | **1** | **53** | **1** | **$0.79** |

Interpretation: across 61 real-output claims, the system surfaced 7
supported/partially-supported claims, 1 unsupported (an AnswerThis claim
asserting MCT1, MCT2, MCT4 haplotype effects that the cited source addresses
only for MCT1), and abstained on 53 where evidence was not located or was
insufficient. That is useful for review triage, but the single unsupported
verdict on a real input is preliminary evidence rather than a track record —
treat it as such. Numbers are extracted programmatically from the per-tool
`report.json` files; see `benchmarks/real_outputs/SUMMARY.md` for the full
table including new diagnostic columns (`no_passage_found`,
`fulltext_unavailable`, `resolution_low_confidence`).

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

## Design Partners

Looking for medical writing, regulatory/scientific review, and AI-for-science tool teams with two real documents per week to verify privately. The useful feedback is where the system abstains, misses contradictions, or over-flags
weak evidence.

Contact: [vivienperrelle@gmail.com](mailto:vivienperrelle@gmail.com)

## License

Apache 2.0. See [LICENSE](LICENSE).
