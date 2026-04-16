# Scientific Claim Verification Engine

![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue)
![License](https://img.shields.io/badge/license-Apache%202.0-green)
![Tests](https://img.shields.io/badge/tests-82%20passing-brightgreen)

Auditable, claim-by-claim verification of scientific text against cited sources.

## What it does

Takes free-form scientific text (paper excerpt, literature review, etc.) and outputs:

- A `report.json` with a verdict for each extracted claim (`supported` / `unsupported` / `partially_supported` / `not_addressed`)
- A `provenance.jsonl` with a full audit trail of every LLM call, token count, and cache hit

## Quick start

```bash
pip install -e ".[dev]"
export ANTHROPIC_API_KEY=sk-...
python examples/sample_run.py
```

Report files are written to `reports/runs/{report_id}/`.

## Pipeline

```text
input text
    → extract_claims()     # LLM: extract verifiable claims with citations
    → resolve_citations()  # Semantic Scholar: find abstracts for each cited source
    → verify_claim()       # LLM: compare claim against abstract
    → build_report()       # Aggregate: write report.json + provenance.jsonl
```

## Public API

### `src.extract.extract_claims`

```python
def extract_claims(
    text: str,
    *,
    model_id: str = "claude-sonnet-4-6",
    api_key: str | None = None,
) -> tuple[list[Claim], ProvenanceStep]: ...
```

### `src.resolve.resolve_citations`

```python
def resolve_citations(
    claims: list[Claim],
    *,
    db_path: Path | None = None,
) -> tuple[dict[str, ResolvedSource], list[ProvenanceStep]]: ...
```

### `src.verify.verify_claim`

```python
def verify_claim(
    claim: Claim,
    source: ResolvedSource,
    *,
    model_id: str = "claude-sonnet-4-6",
    api_key: str | None = None,
) -> tuple[VerificationResult, ProvenanceStep]: ...
```

### `src.report.build_report`

```python
def build_report(
    report_id: str,
    input_text: str,
    claims: list[Claim],
    sources: dict[str, ResolvedSource],
    results: dict[str, VerificationResult],
    provenance_steps: list[ProvenanceStep],
    *,
    output_dir: Path | None = None,
) -> Path: ...
```

Returns the path to the run directory (`reports/runs/{report_id}/`).

## Data models

All models are frozen dataclasses defined in `src/models.py`:

- `Claim` — a single extracted claim with citation metadata
- `ResolvedSource` — result of a Semantic Scholar lookup (abstract, DOI, similarity score)
- `VerificationResult` — LLM verdict with confidence and explanation
- `ProvenanceStep` — audit record for one pipeline step (tokens, cache hit, model ID)

## Evaluation

```bash
python scripts/eval_scifact.py --split dev
```

Uses the SciFact dataset in `eval/scifact/`. The `test` split is locked — never use it during development.

### SciFact dev results (Phase 0 baseline)

| Approach | F1 | Macro-F1 | Cost (50 claims) |
| --- | --- | --- | --- |
| **This pipeline** (structured system prompt + provenance) | **0.94** | **0.94** | $0.17 |
| Direct LLM (naive single prompt, same model) | 0.62 | 0.60 | $0.10 |

Full dev set (300 claims): F1 = **0.923**, Macro-F1 = **0.919**, total cost = $0.84.

Per-class F1 (300 claims): supported = 0.905 · unsupported = 0.889 · not\_addressed = 0.961.

Baseline locked at `eval/results/baseline_phase0.json`.

## Development

```bash
python -m pytest tests/unit/ -v          # 82 tests, all offline
python -m mypy --strict src/             # zero errors
python -m ruff check src/ tests/ scripts/
python -m ruff format src/ tests/ scripts/
```

## Cost estimate

Targeting < $0.10 per 2-page document.

- Input: $3.00 / M tokens
- Cached input: $0.30 / M tokens (prompt caching on all system prompts)
- Output: $15.00 / M tokens

Token costs are logged per call and summed in `report.json` under `summary.total_cost_usd`.

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for setup, code style, and PR guidelines.

## License

Apache 2.0 — see [LICENSE](LICENSE).
