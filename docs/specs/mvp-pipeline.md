# Spec: mvp-pipeline

**Phase:** Phase 0 — MVP, Day 1
**Complexity:** L
**Generated:** 2026-04-15

---

## 1. Goal

Build the end-to-end verification pipeline that takes scientific text, extracts verifiable claims, resolves each claim's cited source via Semantic Scholar, verifies each claim against the source abstract, and produces a structured JSON report with a full ProvenanceStep audit log.

This is the core Phase 0 deliverable: a working pipeline that catches hallucinated citations and unsupported claims in real scientific agent outputs.

---

## 2. Scope

**In scope:**
- `pyproject.toml` with all declared dependencies and tool config
- `src/models.py` — shared frozen dataclasses and type aliases
- `src/clients/_cache.py` — SQLite WAL cache shared by all API clients
- `src/clients/semantic_scholar.py` — httpx client for Semantic Scholar paper search
- `src/extract.py` — LLM-based claim extraction from free-form scientific text
- `src/resolve.py` — batch citation resolution using the Semantic Scholar client
- `src/verify.py` — single-claim LLM verification against source abstract
- `src/report.py` — aggregates results, writes `report.json` + `provenance.jsonl`
- `scripts/eval_scifact.py` — runs pipeline on SciFact dev split, reports F1 metrics
- `examples/sample_run.py` — end-to-end demo using `examples/inputs/crow_sample.txt`
- All corresponding unit tests in `tests/unit/`

**Out of scope:** see Section 11.

---

## 3. Files Touched

```
pyproject.toml                              ← create; project config + deps + tool settings
src/__init__.py                             ← create; empty package marker
src/models.py                               ← create; all shared dataclasses and Literal aliases
src/clients/__init__.py                     ← create; empty package marker
src/clients/_cache.py                       ← create; SQLite WAL cache (get/put/init_db)
src/clients/semantic_scholar.py             ← create; httpx + retry + cache client
src/extract.py                              ← create; extract_claims() via Claude API
src/resolve.py                              ← create; resolve_citations() via Semantic Scholar
src/verify.py                               ← create; verify_claim() via Claude API
src/report.py                               ← create; build_report() writes report.json + provenance.jsonl
scripts/eval_scifact.py                     ← create; SciFact dev-split eval + F1 reporting
examples/sample_run.py                      ← create; end-to-end pipeline demo
examples/inputs/crow_sample.txt             ← create; real Crow/Falcon output (committed Day 1 PM)
tests/__init__.py                           ← create; empty package marker
tests/unit/__init__.py                      ← create; empty package marker
tests/unit/test_models.py                   ← create; dataclass field and type tests
tests/unit/test_semantic_scholar.py         ← create; pytest-httpx mocked client tests
tests/unit/test_extract.py                  ← create; mocked Anthropic SDK tests
tests/unit/test_resolve.py                  ← create; mocked search_paper() tests
tests/unit/test_verify.py                   ← create; mocked Anthropic SDK tests
tests/unit/test_report.py                   ← create; tmp_path-based report/provenance tests
```

---

## 4. Public API

### `src/models.py`

```python
from __future__ import annotations
from dataclasses import dataclass
from typing import Literal

ClaimType = Literal["factual_numeric", "factual_qualitative", "methodological", "causal"]
VerificationStatus = Literal["supported", "unsupported", "not_addressed", "partially_supported"]
OperationType = Literal["extract", "resolve", "verify", "aggregate"]

@dataclass(frozen=True)
class Claim:
    claim_id: str            # uuid4 string
    claim_text: str
    cited_authors: list[str]
    cited_year: int | None
    claim_type: ClaimType

@dataclass(frozen=True)
class ResolvedSource:
    found: bool
    doi: str | None
    title: str | None
    abstract: str | None
    similarity_score: float | None  # 0.0–1.0; None when found=False

@dataclass(frozen=True)
class VerificationResult:
    status: VerificationStatus
    explanation: str
    confidence: float           # 0.0–1.0

@dataclass(frozen=True)
class ProvenanceStep:
    step_id: str                # uuid4 string
    claim_id: str               # Claim.claim_id, or "__extract__:{hash8}" for extraction
    operation: OperationType
    input_hash: str             # sha256(repr(input).encode()).hexdigest()
    output_hash: str            # sha256(repr(output).encode()).hexdigest()
    model_id: str | None        # "claude-sonnet-4-6" or None for deterministic steps
    timestamp: float            # time.time() at operation start
    tokens_in: int | None
    tokens_out: int | None
    cache_hit: bool | None      # from Anthropic response headers; None for non-LLM steps
    confidence: float | None    # mirrors VerificationResult.confidence; None where not applicable
```

### `src/clients/_cache.py`

```python
import sqlite3
from pathlib import Path

def default_db_path() -> Path:
    """Return Path to default cache DB: <project_root>/.cache/api_cache.db"""

def init_db(db_path: Path) -> sqlite3.Connection:
    """Open or create SQLite WAL database; create cache table if absent. Returns connection."""

def get(db_path: Path, cache_key: str) -> str | None:
    """Return cached JSON string for key if present and not expired. None otherwise."""

def put(db_path: Path, cache_key: str, value: str, ttl_seconds: int) -> None:
    """Upsert value with expiry = now + ttl_seconds. No return value."""
```

### `src/clients/semantic_scholar.py`

```python
from pathlib import Path
from src.models import ResolvedSource

_BASE_URL = "https://api.semanticscholar.org/graph/v1"
_FIELDS = "paperId,title,abstract,year,authors"
_RETRY_MAX = 3
_RETRY_BACKOFF_BASE = 2.0   # seconds; attempts: 2s, 4s, 8s
_CACHE_TTL_SECONDS = 30 * 24 * 3600  # 30 days

def search_paper(
    query: str,
    *,
    api_key: str | None = None,
    timeout: float = 10.0,
    db_path: Path | None = None,
) -> ResolvedSource:
    """Search Semantic Scholar for a paper matching the query string.

    Never raises. Returns ResolvedSource(found=False, ...) on all errors.
    Retries on 429 and connection errors with exponential backoff.
    Results cached in SQLite for 30 days.
    API key read from SEMANTIC_SCHOLAR_API_KEY env var if api_key is None.
    """
```

### `src/extract.py`

```python
from src.models import Claim, ProvenanceStep

MODEL_ID = "claude-sonnet-4-6"

def extract_claims(
    text: str,
    *,
    model_id: str = MODEL_ID,
    api_key: str | None = None,
) -> tuple[list[Claim], ProvenanceStep]:
    """Extract verifiable scientific claims from free-form scientific text.

    Uses Claude API with structured XML-tagged prompt.
    System prompt uses cache_control={"type": "ephemeral"} (>1024 tokens).
    Input text wrapped in <text>...</text> to prevent prompt injection.
    On malformed LLM response: returns ([], provenance_step), logs structlog.error.
    ProvenanceStep.claim_id = "__extract__:{sha256(text)[:8]}".
    """
```

### `src/resolve.py`

```python
from pathlib import Path
from src.models import Claim, ResolvedSource, ProvenanceStep

def resolve_citations(
    claims: list[Claim],
    *,
    api_key: str | None = None,
    db_path: Path | None = None,
) -> tuple[dict[str, ResolvedSource], list[ProvenanceStep]]:
    """Resolve each claim's cited source via Semantic Scholar.

    Returns a dict keyed by claim_id (entry present for EVERY claim, even unresolved).
    Returns one ProvenanceStep per claim (operation="resolve", model_id=None).
    Claims with cited_authors=[] or cited_year=None → ResolvedSource(found=False) without HTTP.
    Year ±1 tolerance: accepts results with year in {cited_year-1, cited_year, cited_year+1}.
    Rate-limited or errored claims → found=False; rest of batch unaffected.
    """
```

### `src/verify.py`

```python
from src.models import Claim, ResolvedSource, VerificationResult, ProvenanceStep

MODEL_ID = "claude-sonnet-4-6"

def verify_claim(
    claim: Claim,
    source: ResolvedSource,
    *,
    model_id: str = MODEL_ID,
    api_key: str | None = None,
) -> tuple[VerificationResult, ProvenanceStep]:
    """Verify a single claim against its resolved source abstract via Claude API.

    Short-circuits (no LLM call) when source.found=False or source.abstract is None:
      → returns (VerificationResult(status="not_addressed", confidence=1.0, ...), step)
    System prompt >1024 tokens → cache_control={"type": "ephemeral"}.
    Claim wrapped in <claim>...</claim>; abstract in <source>...</source>.
    Logs tokens_in, tokens_out, cache_hit, model_id via structlog on every LLM call.
    """
```

### `src/report.py`

```python
from pathlib import Path
from src.models import Claim, ResolvedSource, VerificationResult, ProvenanceStep

def build_report(
    report_id: str,
    input_text: str,
    claims: list[Claim],
    sources: dict[str, ResolvedSource],
    results: dict[str, VerificationResult],
    provenance_steps: list[ProvenanceStep],
    *,
    output_dir: Path | None = None,
) -> Path:
    """Aggregate pipeline results and write report.json + provenance.jsonl.

    report_id must be generated by the caller (pipeline entry point), not here.
    Creates reports/runs/{report_id}/ with mkdir(parents=True, exist_ok=True).
    output_dir defaults to project_root/reports/.
    Empty claims list produces valid report with total_claims=0.
    Returns path to the run directory (reports/runs/{report_id}/).
    Side effects: creates two files. No other side effects.
    """

def _compute_summary_stats(
    claims: list[Claim],
    results: dict[str, VerificationResult],
    sources: dict[str, ResolvedSource],
) -> dict[str, int | float]:
    """Pure helper — compute summary statistics. No I/O."""
```

### `scripts/eval_scifact.py`

```python
from pathlib import Path
from typing import Literal

def run_eval(
    split: Literal["dev"],
    corpus_path: Path,
    claims_path: Path,
    output_path: Path,
    *,
    limit: int | None = 50,
    max_cost_usd: float = 5.0,
    model_id: str = "claude-sonnet-4-6",
) -> dict[str, float]:
    """Run pipeline on a SciFact split and return metrics dict.

    Refuses split="test" with structlog.error + sys.exit(1).
    Tracks cumulative cost via ProvenanceStep tokens; aborts with sys.exit(1) if exceeded.
    Default limit=50 (smoke test ~$0.75-1.50). Pass limit=None for full ~300-claim run (~$7-10).
    Label mapping: SUPPORT→supported, CONTRADICT→unsupported, {}→not_addressed.
    partially_supported counted as supported for F1.
    Returns: {"precision", "recall", "f1", "macro_f1", "n_claims", "total_cost_usd", ...}
    """
```

---

## 5. Data Flow

```
INPUT: str  (free-form scientific text)
  │
  ▼
extract_claims(text)
  │  Claude API call (system prompt cached)
  │  Returns: list[Claim], ProvenanceStep (operation="extract")
  │
  ▼
resolve_citations(claims)
  │  For each claim: search_paper(author + year + keywords)
  │    → SQLite cache lookup (30-day TTL)
  │    → httpx GET Semantic Scholar (if cache miss)
  │    → year ±1 tolerance filter
  │  Returns: dict[claim_id → ResolvedSource], list[ProvenanceStep] (operation="resolve")
  │
  ▼
[for each claim] verify_claim(claim, source)
  │  Short-circuit if source.found=False or source.abstract=None
  │  Claude API call (system prompt cached; <claim>/<source> XML tags)
  │  Returns: VerificationResult, ProvenanceStep (operation="verify")
  │
  ▼
build_report(report_id, input_text, claims, sources, results, all_provenance_steps)
  │  Aggregates all results + computes summary stats
  │  ProvenanceStep (operation="aggregate") appended
  │  Writes: reports/runs/{report_id}/report.json
  │  Writes: reports/runs/{report_id}/provenance.jsonl  (one JSON line per step)
  │
  ▼
OUTPUT: Path  (reports/runs/{report_id}/)

report.json shape:
{
  "report_id": str,
  "timestamp": float,
  "input_text": str,
  "summary": {
    "total_claims": int,
    "supported": int,
    "unsupported": int,
    "not_addressed": int,
    "partially_supported": int,
    "citation_found_rate": float,
    "total_cost_usd": float
  },
  "claims": [
    {
      "claim_id": str,
      "claim_text": str,
      "claim_type": str,
      "cited_authors": list[str],
      "cited_year": int | null,
      "source": { "found": bool, "doi": str|null, "title": str|null, ... },
      "verification": { "status": str, "explanation": str, "confidence": float }
    }
  ]
}

provenance.jsonl: one JSON object per line, each a serialized ProvenanceStep
```

---

## 6. External Dependencies

| Dependency | Version | Purpose | Env var |
|---|---|---|---|
| `anthropic` | >=0.40.0 | Claude API (extract + verify) | `ANTHROPIC_API_KEY` |
| `httpx` | >=0.27.0 | Semantic Scholar HTTP client | — |
| `structlog` | >=24.1.0 | Structured JSON logging | — |
| Semantic Scholar API | v1 | Paper search | `SEMANTIC_SCHOLAR_API_KEY` (optional; higher rate limit) |
| SQLite | stdlib | API response cache | — |
| `pytest` | >=8.0.0 | Test runner | — |
| `pytest-asyncio` | >=0.23.0 | Async test support | — |
| `pytest-httpx` | >=0.30.0 | Mock httpx calls in unit tests | — |
| `mypy` | >=1.10.0 | Static type checking | — |
| `ruff` | >=0.4.0 | Formatter + linter | — |

**Do NOT add:** `pint`, `scipy`, `statsmodels` — Phase 2 only.

---

## 7. Edge Cases

All must be covered by unit tests:

**EC-1 — Claim with no citation**
`Claim(cited_authors=[], cited_year=None)`. `resolve.py` must return `ResolvedSource(found=False)` without making any HTTP call. Test: assert no HTTP requests fired when `cited_authors` is empty.

**EC-2 — Semantic Scholar returns null abstract**
Paper found but `abstract` field is `null` or `""`. `search_paper()` returns `ResolvedSource(found=True, abstract=None)`. `verify.py` short-circuits to `status="not_addressed"` without calling Claude. Test: mock response with `"abstract": null`; assert no Anthropic SDK call made.

**EC-3 — Claude returns malformed extraction response**
LLM returns non-JSON text or JSON missing the `claims` key. `extract_claims()` catches the parse failure, logs `structlog.error("extract_parse_error", raw=...)`, and returns `([], provenance_step)`. Must never raise. Test: mock content `"I cannot help with that"` → empty list returned, no exception.

**EC-4 — Paper indexed under year ±1 (preprint/published lag)**
Claim cites year=2019; Semantic Scholar returns `year=2020`. `resolve.py` must accept this result and set `similarity_score` reflecting the year mismatch (e.g., reduce score by 0.1). Test: mock response with `year=2020`, claim cites `2019` → `found=True`.

**EC-5 — Input text has no verifiable claims**
`extract_claims()` returns `([], step)`. `build_report()` called with empty `claims` list must produce valid `report.json` with `total_claims: 0`. No exception, no silent failure. Test: assert report.json parseable with correct structure.

**EC-6 — Rate limit hit mid-batch**
Third claim in a 5-claim batch hits 429 three consecutive times. `search_paper()` retries with backoff; after 3 failures returns `found=False`. Batch continues — claims 4 and 5 are still processed. Test: `httpx_mock` returns 3×429 for claim-3 query, success for others; assert claims 1, 2, 4, 5 return `found=True`, claim 3 returns `found=False`.

**EC-7 — Output directory does not exist**
`reports/runs/` absent (fresh clone). `build_report()` calls `Path.mkdir(parents=True, exist_ok=True)` before any write. Test: use `tmp_path` pointing to non-existent nested path; assert both files created.

---

## 8. Test Plan

### Unit tests (offline, mocked — `tests/unit/`)

| File | Scenarios |
|---|---|
| `test_models.py` | Frozen dataclass mutation rejection; Literal field values; `claim_id` / `step_id` are strings |
| `test_semantic_scholar.py` | Found (happy path); not found (empty results); 429 retry succeeds on attempt 2; network error returns `found=False`; cache hit skips HTTP |
| `test_extract.py` | Valid extraction → list of Claims; empty text → empty list (EC-5); malformed LLM response → empty list (EC-3); ProvenanceStep fields populated |
| `test_resolve.py` | Batch resolution; EC-1 (no citation → no HTTP); EC-4 (year ±1); EC-6 (429 mid-batch) |
| `test_verify.py` | All 4 status values; EC-2 (null abstract → short-circuit); `source.found=False` → short-circuit; ProvenanceStep tokens logged |
| `test_report.py` | Happy path writes both files; EC-5 (empty claims → valid JSON); EC-7 (missing directory); provenance.jsonl has one line per step |

All unit tests: no real HTTP, no real Anthropic calls. Use `pytest-httpx.HTTPXMock` for httpx; `unittest.mock.patch` for Anthropic SDK.

### Integration tests (real APIs — `tests/integration/`, NOT run in pre-commit)

| File | What it tests |
|---|---|
| `test_semantic_scholar.py` | Real search against Semantic Scholar; verifies ≥1 known SciFact paper resolves |
| `test_pipeline_e2e.py` | Full pipeline on `examples/inputs/crow_sample.txt`; verifies report.json + provenance.jsonl written |

Run integration tests manually: `pytest tests/integration/ -v --run-integration`

---

## 9. ProvenanceStep

ProvenanceStep is emitted by every module and collected by the caller. **Modules never write to disk — only `report.py` writes.**

| Module | When emitted | `operation` | `model_id` | `tokens_*` / `cache_hit` |
|---|---|---|---|---|
| `extract.py` | Once per `extract_claims()` call | `"extract"` | `"claude-sonnet-4-6"` | From `response.usage`; `cache_hit` from response headers |
| `resolve.py` | Once per claim in `resolve_citations()` | `"resolve"` | `None` | `None` / `None` / `None` |
| `verify.py` | Once per `verify_claim()` call | `"verify"` | `"claude-sonnet-4-6"` | From `response.usage`; `cache_hit` from response headers |
| `report.py` | Once per `build_report()` call | `"aggregate"` | `None` | `None` |

**`input_hash`**: `hashlib.sha256(repr(input_data).encode()).hexdigest()` where `input_data` is the function's primary input (text for extract, claim for resolve/verify, all results for aggregate).

**`output_hash`**: same pattern applied to the function's return value.

**`claim_id` for extraction**: `"__extract__:" + hashlib.sha256(text.encode()).hexdigest()[:8]` — extraction produces claims rather than operating on one, so no Claim.claim_id exists yet.

**Storage**: `report.py` writes all collected ProvenanceStep objects (serialised via `dataclasses.asdict()`) as newline-delimited JSON to `reports/runs/{report_id}/provenance.jsonl`.

**Phase 4+**: migration to provenance graph DB is backward-compatible with the JSONL format.

---

## 10. Risks

**R1 — Semantic Scholar coverage gap**
SciFact papers are 2016-2019; some may not be indexed or may have changed paper IDs. Mitigation: cache all responses so re-runs are free; log `citation_found_rate` per eval run; if <70% found rate, investigate specific failures before Day 6.

**R2 — F1 below 0.50 on Day 4 smoke test**
Root cause is almost always the extraction prompt (under-extracting claims) or the verification prompt (too many `not_addressed` returns when abstract is actually relevant). Mitigation: log the distribution of `status` values per eval run; if `not_addressed` is >50% of results and `citation_found_rate` is >70%, the verify prompt is the culprit. Invoke `@prompt-smith` immediately.

**R3 — Anthropic API key not set**
Pipeline will raise an `AuthenticationError` on first LLM call, not on import. Mitigation: `examples/sample_run.py` should check `os.environ.get("ANTHROPIC_API_KEY")` at startup and print a clear message before calling any pipeline function.

**R4 — SQLite cache growing unbounded**
Eval runs many queries; cache has no eviction beyond TTL. Mitigation: add a `prune_expired()` function to `_cache.py` that deletes rows past their TTL; call it once per `eval_scifact.py` run.

**R5 — Cost overrun on eval loops**
Multiple prompt iteration cycles on Day 4-5 could accumulate $50+ in API costs if running full dev split each time. Mitigation: `max_cost_usd=5.0` hard ceiling in `run_eval()`; default `limit=50` until F1 is stable.

---

## 11. Out of Scope

| Feature | Reason deferred |
|---|---|
| Full-text retrieval (PMC/Unpaywall/PDF) | Phase 1 — requires section-aware chunking not needed for abstracts |
| CrossRef / PMC / Unpaywall API clients | Phase 1 — Semantic Scholar alone sufficient for abstract-level verification |
| Numerical verification engine | Phase 2 — requires pure-Python comparison engine (pint, scipy) |
| Retraction checks | Phase 1 — Retraction Watch API integration deferred |
| Ontology integration (GO/MeSH/UniProt) | Phase 3 |
| FastAPI / REST endpoints | Phase 5 |
| PostgreSQL / Redis | Phase 5 — SQLite WAL sufficient through Phase 3 |
| Causal claim detector | Phase 3 — requires semantic precision layer |
| PDF parsing | Phase 1 |
| Any web UI | Phase 5+ |
| Confidence calibration against ground truth | Phase 4 |
| `pint` / `scipy` / `statsmodels` | Phase 2 — adding now adds import surface with zero benefit |

---

## Assumptions to validate

1. SciFact `claims.jsonl` uses paper IDs that map 1:1 to `corpus.jsonl` — verify after download.
2. Semantic Scholar free tier (100 req/5min) is sufficient for 50-claim smoke tests — if the eval run takes >5 min, cache warm-up is the bottleneck, not rate limiting.
3. `cache_hit` is detectable from Anthropic response headers — confirm by logging the raw response object headers in a test call on Day 3.

---

## Top 3 Risks (pre-implementation)

1. **F1 target** — extraction prompt quality is the biggest unknown. Plan mitigation: log status distribution from first eval run; if >50% `not_addressed` with high citation resolution, fix the verify prompt first.
2. **Semantic Scholar coverage** — if <70% of SciFact citations resolve, the eval F1 will be artificially low regardless of prompt quality. Measure `citation_found_rate` explicitly on Day 2.
3. **Cost control** — without the `max_cost_usd` ceiling, repeated eval loops during prompt iteration could burn significant budget. This must be wired into `eval_scifact.py` on Day 4, not added later.
