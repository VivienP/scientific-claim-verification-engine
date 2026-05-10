# Scientific Claim Verification Engine

![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue)
![License](https://img.shields.io/badge/license-Apache%202.0-green)

> Run any scientific text — paper drafts, AI summaries, literature reviews — through the pipeline; get back one verdict per cited claim, grounded in the actual cited source, with full provenance.

## Example

**Input:** a claim from an AI-generated literature review — *"Smith et al. (2023) showed that treatment X reduced biomarker Y by 40%."*

**Output:** verdict `partially_supported` — the cited paper reports a 40% reduction only in the high-dose cohort; the claim omits the qualifier. Source quote, retrieval status, and provenance hash included.

See [`benchmarks/real_outputs/edison_trem2/report.json`](benchmarks/real_outputs/edison_trem2/report.json) for the full committed run on Edison TREM2 (20 claims), showing all verdict types in a real report.

## Quick Start

```bash
pip install -e ".[dev]"
export ANTHROPIC_API_KEY=sk-...
```

```python
from src.pipeline import PipelineConfig, run_pipeline
from src.report import build_report
import os, uuid, pathlib

text = pathlib.Path("benchmarks/real_outputs/edison_trem2/input.txt").read_text()
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

~3 min, ~$0.40 on the default input ([Edison TREM2, 20 claims](benchmarks/real_outputs/edison_trem2/)).

## What You Get

- **`report.json`** — one record per claim: verdict, cited source, retrieval status, evidence quality, source quotes, numeric check result.
- **`provenance.jsonl`** — append-only audit trail: every extraction, resolution, verification, and aggregation step, with token and cache usage.

Verdicts: `supported` · `partially_supported` · `unsupported` · `not_addressed`. Abstention is explicit — the pipeline never forces weak evidence into `unsupported`.

## Track Record

| Benchmark | Scope | Result | Detail |
| --- | --- | --- | --- |
| Lactate-ISF, 25 expert-annotated claims | **full pipeline** | 16/25 verdict agreement (64%) | [eval/e2e/](eval/e2e/reference_paper_v1_results.md) |
| Valsci paper (bioinformatics), 11 external claims | resolver | 10/11 correct source (91%) | [benchmarks/real_papers/valsci_brice_2025/](benchmarks/real_papers/valsci_brice_2025/README.md) |
| SciFact dev | verifier, oracle inputs | F1 = 0.94 | binary, [scripts/eval_scifact.py](scripts/eval_scifact.py) |
| Real AI-for-science tools, 59 claims | resolver | 72.9% citation found rate | [benchmarks/real_outputs/](benchmarks/real_outputs/README.md) |

## Pipeline

```text
input text
  -> extract claims          (LLM, citation-anchored)
  -> resolve citations       (multi-source: bib DOI → CrossRef → OpenAlex → PubMed)
  -> enrich metadata         (PubMed PMID, Europe PMC OA discovery)
  -> fetch full text         (OA URL → PMC → Europe PMC → Unpaywall PDF)
  -> chunk and select        (deterministic IMRAD sections + BM25 token-budget)
  -> verify                  (route by retrieval depth: full-text / abstract / title-only / multi-source)
  -> deterministic numeric   (OR/CI consistency, p-value/CI null-crossing)
  -> report.json + provenance.jsonl
```

## Public API

```python
from src.pipeline import PipelineConfig, run_pipeline
from src.report import build_report
import os, uuid

config = PipelineConfig(api_key=os.environ["ANTHROPIC_API_KEY"])
verifications, steps = run_pipeline(text, config=config)

run_dir = build_report(
    str(uuid.uuid4()), text,
    claims=[v.claim for v in verifications],
    sources={v.claim.claim_id: v.source for v in verifications},
    results={v.claim.claim_id: v.result for v in verifications},
    provenance_steps=steps,
)
# report.json and provenance.jsonl written to run_dir
```

Data models (`Claim`, `ResolvedSource`, `ResolvedSourceSet`, `VerificationResult`, `ProvenanceStep`, `PaperChunk`) are frozen dataclasses in [`src/models.py`](src/models.py). Low-level step-by-step API: [`src/pipeline.py`](src/pipeline.py).

## HTTP API (on-prem deployment)

A FastAPI wrapper around `run_pipeline` and the Copilot enrichment layer is available for biotech ops teams that need to deploy behind a corporate firewall. It exposes async jobs (POST /verify → 202 + job_id; poll GET /jobs/{id}) so requests don't time out at any reverse proxy.

```bash
# Run locally
export COPILOT_API_KEY="$(openssl rand -hex 32)"
export ANTHROPIC_API_KEY="sk-ant-..."
uvicorn src.api.app:app --host 127.0.0.1 --port 8000

# Or via Docker (read-only rootfs, cap_drop ALL, bound to 127.0.0.1)
docker compose up
```

Endpoints (all require `X-API-Key: $COPILOT_API_KEY` except `/health`):

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Load-balancer probe (no auth) |
| `POST` | `/verify` | Submit a job; returns `202` + `{job_id, poll_url}` |
| `GET` | `/jobs/{job_id}` | Status + result envelope |
| `GET` | `/runs/{run_id}/copilot_report.html` | Self-contained Copilot HTML (path-confined) |

Programmatic factory: `from src.api import create_app; app = create_app()`. Client example: [`examples/api_run.py`](examples/api_run.py). Container hardening: non-root uid 10001, `read_only: true`, `cap_drop: ALL`, exact-pinned Python deps. Single-tenant Phase C; multi-tenant Postgres-backed JobStore is deferred to Phase D.

## Limitations

- Requires explicit author/year or numbered bracket citation anchors.
- Open-access only — paywalled sources without an OA copy return `not_addressed`.
- BM25 is lexical; claims with no token overlap with any retrieved chunk report `no_passage_found`.
- Numeric coverage: OR/CI consistency and p-value/CI null-crossing checks only.
- Each claim is checked against its cited source, not the full literature.
- The LLM-reported `confidence` field is unreliable (self-reported by the model). Use `retrieval_status` and `evidence_quality` as the trust signals — these are deterministically computed from what was actually retrieved.

## Related Work

Closest neighbours: [Valsci](https://github.com/bricee98/Valsci) (single-source, batch literature corpus) and [CiteAudit](https://arxiv.org/abs/2602.23452) (metadata consistency, 5-agent). This engine targets AI-agent output auditing with multi-source resolution and deterministic numeric checks separated from LLM verification. Full survey: [docs/related-work.md](docs/related-work.md).

## Development

```bash
python -m pytest -v
python -m mypy --strict src
python -m ruff check src tests scripts
```

Canary controls (contradiction detection, weak resolution, numeric inconsistency): use `benchmarks/canary/input.txt` as the input text in the Quick Start snippet above.

Claude Code workflows: `/eval` (SciFact dev metrics), `/dogfood` (run pipeline on real AI-tool output), `/skillify-failure` (convert dogfood failures into draft regression tests + rules + prompt patches). Agents and rules under [`.claude/`](.claude/).

## License & Contact

Apache 2.0 — see [LICENSE](LICENSE). Dataset under `eval/e2e/` is CC BY-NC.

**Looking for design partners.** If you're building an AI-for-science tool, working in pharma medical affairs, or evaluating AI-generated scientific text — reach out [@PerrelleVivien](https://x.com/PerrelleVivien).