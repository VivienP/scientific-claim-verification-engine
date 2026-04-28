# AGENTS.md

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Reference:** $HOME\refs\cc-best-practice

When answering Codex best practice questions, always search `$HOME\refs\cc-best-practice` first (`best-practice/`, `reports/`, `tips/`, `implementation/`, `README.md`) before relying on training knowledge or web search. That repo is the authoritative source.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.

---

## Project: Scientific Claim Verification Engine

## Mission

Verification pipeline that takes scientific text (any source) and outputs auditable claim-by-claim verification reports. Source-agnostic, on-prem-capable.

## Non-goals

- Not an agent (no autonomous decision-making in production)
- Not a generator (no content creation, only verification)
- Not a wrapper (engineering > prompt)

## Current phase

CURRENT_PHASE: Phase 1 — MVP

## Architecture decisions

- Python 3.12+, no framework Phase 0-1
- FastAPI Phase 5+
- SQLite (WAL mode) local Phase 0-3, PostgreSQL Phase 5+
- No ORM until Phase 4
- Anthropic SDK for all LLM calls — prompt caching on all stable prefixes
- httpx for all HTTP (Semantic Scholar, CrossRef, PMC, Unpaywall)
- structlog for all logging (structured JSON output)
- Chunking strategy: section-aware (Introduction/Methods/Results/Discussion)
  for all full-paper processing. Never sliding window in Phase 0-3.
- Output directory: `reports/runs/{report_id}/` — contains `report.json` and `provenance.jsonl`

## Provenance schema (active from Phase 0)

Every verification step must emit a ProvenanceStep:

```python
step_id: str        # uuid4
claim_id: str       # links to the Claim this step belongs to
operation: str      # "extract" | "resolve" | "verify" | "aggregate"
input_hash: str     # sha256 of the input data
output_hash: str    # sha256 of the output data
model_id: str | None
timestamp: float    # time.time()
tokens_in: int | None
tokens_out: int | None
cache_hit: bool | None
confidence: float | None
```

Phase 0-3: append to `reports/runs/{report_id}/provenance.jsonl`.
Phase 4+: write to provenance graph DB.

## Eval baseline

- SciFact dataset (Allen AI): `eval/scifact/` — ground truth for Phase 0-3
- Train/eval/test split: 60/20/20, fixed seed, never reshuffled
- Test set is LOCKED. Never used for prompt selection or tuning.

## Cost constraints

- Log tokens_in/tokens_out/cache_hit for every LLM call
- Prompt caching required on all system prompts > 1024 tokens
- Target: < $0.10 per 2-page document at Phase 0

## Code style

- Type hints everywhere, mypy --strict
- ruff format + ruff check
- Pure functions by default
- Classes only for modules > 200 lines
- No try/except without logging (structlog)
- logger over print, always

## Definition of done

- Tests pass (`pytest -v`)
- `mypy --strict` passes
- `ruff check` passes
- Example in `examples/`
- README updated if public API changed
- Runs end-to-end on sample input
- @reviewer approval

## Forbidden patterns

- `time.sleep()` without exponential backoff
- `print()` in production code
- Hardcoded credentials or API keys
- Catch-all bare `except:`
- LLM calls in deterministic modules (Phase 2+)
- Sliding window chunking for scientific papers

## How to work with me

- Always invoke @scope-guard before non-trivial tasks
- Always invoke @architect before writing implementation code
- Always invoke @reviewer before commits
- Use `/eval` to validate prompt changes
- Use `/dogfood` weekly minimum
