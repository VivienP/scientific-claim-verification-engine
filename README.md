# Scientific Claim Verification Engine

![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue)
![License](https://img.shields.io/badge/license-Apache%202.0-green)

> Run any scientific text — paper drafts, AI summaries, literature reviews — through the pipeline; get back one verdict per cited claim, grounded in the actual cited source, with full provenance.

**Three ways to consume the engine**: as a Python library ([Quick Start](#quick-start) below), as an HTTP service for on-prem deployment ([HTTP API](#http-api-on-prem-deployment)), or as an [MCP server](#mcp-server-agent-callable) — Claude Desktop, Claude Agent SDK, and any MCP-compatible agent can call `verify_text` directly.

## Example

**Input:** a claim from an AI-generated literature review — *"Smith et al. (2023) showed that treatment X reduced biomarker Y by 40%."*

**Output:** verdict `partially_supported` — the cited paper reports a 40% reduction only in the high-dose cohort; the claim omits the qualifier. Source quote, retrieval status, and provenance hash included.

See [`benchmarks/real_outputs/elicit_psilocybin/report.json`](benchmarks/real_outputs/elicit_psilocybin/report.json) for a committed run (43 claims on Elicit's psilocybin / TRD report) — exercises `supported`, `partially_supported`, `not_addressed`, and `unverifiable` verdicts plus the `fetch_traces.jsonl` coverage diagnostic.

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

Each run writes four artifacts under `reports/runs/{run_id}/`: `report.json` (one entry per claim, canonical), `provenance.jsonl` (append-only step trace; sum `tokens_in + tokens_out` for exact $ cost), `report.md` (human-readable rendering of the same verdicts), and `fetch_traces.jsonl` (per-attempt fulltext fetch log for publisher-level access diagnostics).

For an external reviewer or buyer audit, export the canonical `report.json` into a lightweight audit package:

```bash
python -m scripts.export_audit_package benchmarks/real_outputs/elicit_psilocybin/report.json --output-dir reports/audit_packages/elicit_psilocybin
```

The package contains `claims.csv` for human adjudication, `audit_summary.md`, `limitations.md`, and `manifest.json`. It is post-processing only: it does not rerun the verifier or change verdict semantics.

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
  status: supported | partially_supported | unsupported | not_addressed | unverifiable
  evidence_quality: quoted_passage | passages_searched_no_quote
                  | abstract_only | title_only
                  | citing_paper_context | no_evidence
  verification_depth: fulltext | abstract | title_only | citing_paper_context
  source_passages: [string]      # always populated when evidence_quality != no_evidence
  numeric_check: object | null   # OR/CI or p-value/CI consistency, deterministic
  confidence: 0.0–1.0 | null     # null only when status=unverifiable; LLM self-report — UNRELIABLE
  unverifiable_reason: insufficient_evidence_depth | fulltext_unavailable
                     | numeric_claim_abstract_only | parse_error | null
```

Full schema (nested fields, Copilot enrichment, worked example): [docs/output-schema.md](docs/output-schema.md).

## Track Record

| Benchmark | Scope | Result | Detail |
| --- | --- | --- | --- |
| Lactate-ISF, 25 expert-annotated claims | **full pipeline** | 16/25 verdict agreement (64%) | [eval/e2e/](eval/e2e/reference_paper_v1_results.md) |
| Valsci paper (bioinformatics), 11 external claims | resolver | 11/11 correct source (100%) | [benchmarks/real_papers/valsci_brice_2025/](benchmarks/real_papers/valsci_brice_2025/README.md) |
| SciFact dev | verifier, oracle inputs | F1 = 0.94 | binary, [scripts/eval_scifact.py](scripts/eval_scifact.py) |
| Elicit psilocybin / TRD (2026-05-12) | full pipeline | 43 claims; **0 silent failures** · prior baseline 15/57 = 26%; 10 supported / 17 partial / 0 unsupported / 7 not_addressed / 9 unverifiable; $0.75 | [benchmarks/real_outputs/elicit_psilocybin/](benchmarks/real_outputs/elicit_psilocybin/README.md) |
| Elicit GLP-1 MACE (2026-05-12) | full pipeline | 36 claims; **0 silent failures**; 19 supported / 2 partial / 0 unsupported / 8 not_addressed / 7 unverifiable; $0.63 | [benchmarks/real_outputs/elicit_glp1_mace/](benchmarks/real_outputs/elicit_glp1_mace/README.md) |
| AnswerThis lactate (2026-05-12) | full pipeline | 19 claims, **79% confidently classified** · prior baseline 3/25 = 12% (22/25 routed to `not_addressed`); 6 supported / 8 partial / 1 unsupported / 4 not_addressed; $0.25 | [benchmarks/real_outputs/answerthis_lactate/](benchmarks/real_outputs/answerthis_lactate/report.json) |
| Real AI-for-science tools, 6 outputs | full pipeline | 3 of 6 confirmed on the current pipeline (psilocybin + GLP-1 + AnswerThis); 3 pending (Edison TREM2, Elicit PD-1 NSCLC, Sakana AI Scientist). Archived prior aggregates under each benchmark's `_archive_pre_fix/` (do not cite). | [benchmarks/real_outputs/](benchmarks/real_outputs/README.md) |
| Claim Transparency (CTran) across 135 claims, 5 benchmarks | audit trail | 65.9% transparent · prior baseline 48.9% (+17pp) | [benchmarks/real_outputs/](benchmarks/real_outputs/README.md#claim-transparency-ctran) |

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

> `VERIFIER_API_KEY` is the pre-shared secret that gates this HTTP API and the MCP server below — required only for those two paths. The Python library Quick Start above needs only `ANTHROPIC_API_KEY`.

```bash
# Run locally
export VERIFIER_API_KEY="$(openssl rand -hex 32)"
export ANTHROPIC_API_KEY="sk-ant-..."
uvicorn src.api.app:app --host 127.0.0.1 --port 8000

# Or via Docker (read-only rootfs, cap_drop ALL, bound to 127.0.0.1)
docker compose up
```

Endpoints (all require `X-API-Key: $VERIFIER_API_KEY` except `/health`):

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
export VERIFIER_API_KEY="$(openssl rand -hex 32)"
export VERIFIER_API_BASE_URL="http://127.0.0.1:8000"  # or your VPC URL

# Stdio transport — what Claude Desktop spawns
verifier-mcp
```

Claude Desktop config (`claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "scve": {
      "command": "verifier-mcp",
      "env": {
        "VERIFIER_API_BASE_URL": "http://127.0.0.1:8000",
        "VERIFIER_API_KEY": "<same key as the lite API>"
      }
    }
  }
}
```

The MCP server requires the lite API to be running (locally or remotely). It is a stateless adapter — provenance is captured by the underlying API in `reports/runs/api-{job_id[:8]}/provenance.jsonl`, not by the MCP layer.

## Limitations

- Requires explicit author/year or numbered bracket citation anchors.
- Open-access only — when no public full-text is retrievable, numeric claims (%, p-values, ratios) return `unverifiable` with `unverifiable_reason`; qualitative claims fall back to abstract and return `not_addressed` if the abstract is silent.
- BM25 is lexical; claims with no token overlap with any retrieved chunk report `no_passage_found`.
- Numeric coverage: OR/CI consistency and p-value/CI null-crossing checks only.
- Each claim is checked against its cited source, not the full literature.
- The LLM-reported `confidence` field is unreliable (self-reported by the model). Use `retrieval_status` and `evidence_quality` as the trust signals — these are deterministically computed from what was actually retrieved.

## Acknowledged Work

See [docs/related-work.md](docs/related-work.md).

## Development

```bash
python -m pytest -v
python -m mypy --strict src
python -m ruff check src tests scripts
```

Canary controls (contradiction detection, weak resolution, numeric inconsistency): use `benchmarks/canary/input.txt` as the input text in the Quick Start snippet above.

Internal Claude Code workflows used during development: `/eval` (SciFact dev metrics), `/dogfood` (run pipeline on real AI-tool output), `/skillify-failure` (convert dogfood failures into draft regression tests + rules + prompt patches).

## License & Contact

Apache 2.0 — see [LICENSE](LICENSE). Dataset under `eval/e2e/` is CC BY-NC.

**Looking for pilot users.** If you're building an AI-for-science tool, working in pharma medical affairs, or evaluating AI-generated scientific text — reach out [@PerrelleVivien](https://x.com/PerrelleVivien).
