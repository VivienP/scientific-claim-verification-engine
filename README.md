# Scientific Claim Verification Engine

![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue)
![License](https://img.shields.io/badge/license-Apache%202.0-green)

> Run any scientific text — paper drafts, AI summaries, literature reviews — through the pipeline; get back one verdict per cited claim, grounded in the actual cited source, with full provenance.

**Three ways to consume the engine**: as a Python library ([Quick Start](#quick-start) below), as an HTTP service for on-prem deployment ([HTTP API](#http-api-on-prem-deployment)), or as an [MCP server](#mcp-server-agent-callable) — Claude Desktop, Claude Agent SDK, and any MCP-compatible agent can call `verify_text` directly.

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

Each run writes `report.json` (one entry per claim) and `provenance.jsonl` (append-only audit log; sum `tokens_in + tokens_out` across lines for exact $ cost) under `reports/runs/{run_id}/`.

```yaml
# report.json — one entry per claim in claims[]
claim_id: uuid
claim_text: string
source:
  found: bool
  doi: string | null
  similarity_score: 0.0–1.0     # title/author cosine, used as resolution gate
  oa_url: url | null
  retraction_status: bool
verification:
  status: supported | partially_supported | unsupported | not_addressed
  evidence_quality: quoted_passage | passages_searched_no_quote
                  | abstract_only | title_only
                  | citing_paper_context | no_evidence
  verification_depth: fulltext | abstract | title_only | citing_paper_context
  source_passages: [string]      # always populated when evidence_quality != no_evidence
  numeric_check: object | null   # OR/CI or p-value/CI consistency, deterministic
  confidence: 0.0–1.0            # LLM self-report — UNRELIABLE; trust evidence_quality
```

Full schema (nested fields, Copilot enrichment, worked example): [docs/output-schema.md](docs/output-schema.md).

## Track Record

| Benchmark | Scope | Result | Detail |
| --- | --- | --- | --- |
| Lactate-ISF, 25 expert-annotated claims | **full pipeline** | 16/25 verdict agreement (64%) | [eval/e2e/](eval/e2e/reference_paper_v1_results.md) |
| Valsci paper (bioinformatics), 11 external claims | resolver | 10/11 correct source (91%) | [benchmarks/real_papers/valsci_brice_2025/](benchmarks/real_papers/valsci_brice_2025/README.md) |
| SciFact dev | verifier, oracle inputs | F1 = 0.94 | binary, [scripts/eval_scifact.py](scripts/eval_scifact.py) |
| Real AI-for-science tools, 187 claims across 6 outputs | full pipeline | 84.5% citation found rate; 67 supported / 29 partial / 31 unsupported / 60 not_addressed; 24 numeric checks (4 flagged) | [benchmarks/real_outputs/](benchmarks/real_outputs/README.md) |
| Claim Transparency (CTran) across 135 claims, 5 benchmarks | audit trail | 65.9% transparent (+17pp vs pre-fix baseline) | [reports/phase_a2/](reports/phase_a2/ctran_failure_matrix.md) |

## Pipeline

```text
input text
  -> extract claims          (LLM, citation-anchored; partial-recovery on truncated responses)
  -> resolve citations       (multi-source: bib DOI → CrossRef → OpenAlex → PubMed)
  -> enrich metadata         (PubMed PMID, Europe PMC OA discovery)
  -> fetch full text         (OA URL → PMC → Europe PMC → Unpaywall PDF)
  -> chunk and select        (deterministic IMRAD sections + BM25 token-budget)
  -> verify                  (route by retrieval depth: full-text / abstract / title-only / multi-source)
  -> audit-trail fallback    (surface BM25 passages as `passages_searched_no_quote` when LLM didn't quote)
  -> deterministic numeric   (OR/CI consistency, p-value/CI null-crossing)
  -> report.json + provenance.jsonl
```

End-to-end diagrams (verifier routing, audit-trail logic, multi-source aggregation): [docs/architecture.md](docs/architecture.md).

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
| `GET` | `/runs/{run_id}/copilot_report.html` | Self-contained Copilot HTML (path-confined) — see [docs/output-schema.md](docs/output-schema.md#6-copilot-enrichment) for the per-claim Copilot fields surfaced in the HTML |

Programmatic factory: `from src.api import create_app; app = create_app()`. Client example: [`examples/api_run.py`](examples/api_run.py). Container hardening: non-root uid 10001, `read_only: true`, `cap_drop: ALL`, exact-pinned Python deps. Single-tenant Phase C; multi-tenant Postgres-backed JobStore is deferred to Phase D.

## MCP server (agent-callable)

For AI agents (Claude Desktop, Claude Agent SDK, any MCP client), the engine ships a thin MCP wrapper at [`src/mcp_server/`](src/mcp_server/). It speaks JSON-RPC over stdio (default) or streamable HTTP and forwards tool calls to the lite API above — agents never see the polling loop.

**Tools exposed**: `verify_text(text, mode, copilot_mode, wait, timeout_seconds)`, `get_job_status(job_id)`, `get_health()`. **Resource**: `report://{run_id}` returns the Copilot HTML.

```bash
# Install with the mcp extra
pip install -e '.[mcp]'

# Required env (matches the lite API)
export COPILOT_API_KEY="$(openssl rand -hex 32)"
export COPILOT_API_BASE_URL="http://127.0.0.1:8000"  # or your VPC URL

# Stdio transport — what Claude Desktop spawns
copilot-mcp
```

Claude Desktop config (`claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "scve": {
      "command": "copilot-mcp",
      "env": {
        "COPILOT_API_BASE_URL": "http://127.0.0.1:8000",
        "COPILOT_API_KEY": "<same key as the lite API>"
      }
    }
  }
}
```

The MCP server requires the lite API to be running (locally or remotely). It is a stateless adapter — provenance is captured by the underlying API in `reports/runs/api-{job_id[:8]}/provenance.jsonl`, not by the MCP layer (consistent with the [`provenance-first`](.claude/rules/provenance-first.md) rule).

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
