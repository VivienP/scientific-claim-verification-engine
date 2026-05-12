# Spec: `unverifiable` verdict emission gates & prompt updates (Track A2 + A3)

**Date:** 2026-05-11
**Author:** @architect
**Phase:** Phase 1 (MVP)
**Parent plan:** `reports/that-s-very-bad-the-stateful-sloth.md`
**Tracks:** A2 (verifier emission) + A3 (prompt updates). Bundled because A3's prompts must match A2's emission gates exactly.
**Depends on:** Track A1 spec `docs/specs/unverifiable-verdict-schema-invariant.md` -- the `VerificationStatus` type alias, `VerificationResult.__post_init__` validator, and `safe_verification_result()` migration helper must be landed before this spec ships.

---

## Decision log

**2026-05-11 — A3 simplified.** The A3 prompt-clause additions (instructing the LLM to return `unverifiable` for numeric/Results-section claims on abstract-only sources) are **CANCELLED**. The deterministic `safe_verification_result` helper from A1 enforces the same invariant from Python without requiring LLM cooperation, so the prompt clause is redundant. See `memory/feedback_prefer_deterministic_gates.md`. A3 now consists only of adding `"unverifiable"` to `_VALID_STATUSES` in `src/verify_prompts.py` for forward-compatibility (so the helper accepts the value if the LLM ever emits it). Sections 4.7, Risk 1, Risk 2, and the "A3: Prompt changes (detailed)" appendix are removed accordingly. Edge cases 7.4 / 7.5 / 7.6 / 7.11 are retained — the helper still must handle `unverifiable` from any source, including future prompt-tuning experiments.

**2026-05-11 — `claim_text` parameter + Invariant 2 dropped.** Two coupled changes to the A1 spec propagate to this spec:

1. **`safe_verification_result` now takes a `claim_text: str | None` parameter** and applies `_claim_has_specific_numeric()` (from `src/numeric/heuristics.py`, new) to discriminate numeric vs qualitative claims. Only numeric/Results-section claims on insufficient evidence are downgraded; qualitative claims pass through. **Every `safe_verification_result(...)` call in this spec MUST also include `claim_text=claim.claim_text`** (or whichever local variable holds the claim text). For `verify_multi.py`, pass `claim_text=claim.claim_text` from the surrounding `verify_claim_multi(claim, ...)` scope.

2. **Invariant 2 dropped from `VerificationResult.__post_init__`.** The schema no longer raises on `(supported|unsupported, abstract_only|title_only|citing_paper_context|no_evidence)`. The helper is the sole enforcement point. References to "Invariant 2" elsewhere in this spec body (Sections 4.5, 7.x, etc.) are STALE — read them as "the helper's downgrade rule" rather than "schema invariant." The bug-fix correctness is preserved because every production emission site routes through `safe_verification_result`, which still downgrades numeric-claim cases.

   Consequence: the ~20 test fixture sites listed in A1 Section 11 Risk 1 NO LONGER need fixing. They construct `VerificationResult` directly with combinations that the old Invariant 2 would have forbidden — and the new schema accepts those constructions. Skip the fixture-update work entirely. If a test was specifically asserting that `__post_init__` raised on `(supported, abstract_only)`, that test must be DELETED or rewritten to assert the helper's downgrade behavior instead.

## 1. Goal

Wire every LLM-response parse boundary in the verification layer through the `safe_verification_result()` helper from A1, so that `(supported|unsupported)` verdicts on insufficient evidence (`abstract_only`, `title_only`, `citing_paper_context`, `no_evidence`) are automatically downgraded to `(unverifiable, confidence=None)`. The gate is purely deterministic — it inspects `evidence_quality` (a structured field, not LLM output) and applies the downgrade in Python. The LLM remains naive about the constraint; the gate catches it post-hoc.

## 2. Scope

**In:**
- Route `verify.py:184-194` through `safe_verification_result()`.
- Fix the `verify_fulltext.py:78-97` empty-passages fallback: `fulltext_available=False`, `retrieval_status="fulltext_unavailable"`.
- Fix `verify_multi.py:127-134` to inherit `verification_depth` and `evidence_quality` from the chosen primary verdict.
- Route `verify_citing_context.py:107-131` through `safe_verification_result()` (fixes unbounded `unsupported`).
- Confirm `verify_title_only.py` is already compliant under A1 (it caps to `partially_supported`).
- Add `unverifiable_reason: Literal[...] | None` as an optional field on `ProvenanceStep`.
- Add `"unverifiable"` to `_VALID_STATUSES` in `src/verify_prompts.py` (forward compatibility — accept the value if the LLM ever returns it; do not instruct it to).
- Update all affected unit tests.

**Out:**
- `report.py` summary changes for `unverifiable` counts (A4).
- `pipeline.py` status branching updates (A4).
- Fulltext fetcher improvements (Track D).
- Workflow defense / agent rule changes (Track B).
- Recomputing contaminated benchmarks (Track C).

## 3. Files touched

| Path | Lines | What changes |
|---|---|---|
| `src/verify.py` | 184-194 | Replace direct `VerificationResult(...)` with `safe_verification_result(...)` at the LLM parse boundary |
| `src/verify_fulltext.py` | 78-97 | Fix empty-passages fallback: `fulltext_available=False`, `retrieval_status="fulltext_unavailable"` |
| `src/verify_multi.py` | 122-134 | Derive `verification_depth`, `evidence_quality` from chosen primary verdict; route through `safe_verification_result()` |
| `src/verify_citing_context.py` | 107-131 | Replace direct `VerificationResult(...)` with `safe_verification_result(...)` |
| `src/verify_title_only.py` | (no change) | Already compliant: caps to `partially_supported` which is exempt from Invariant 2 |
| `src/models.py` | 159-171 | Add `unverifiable_reason: UnverifiableReason` optional field to `ProvenanceStep` |
| `src/verify_prompts.py` | 45 | Add `"unverifiable"` to `_VALID_STATUSES` (if not already done by A1). No prompt-content edits in this track. |
| `tests/unit/test_verify_abstract.py` | new tests | 5 new test methods for emission gate |
| `tests/unit/test_verify_fulltext.py` | new + update | 2 new tests for empty-passages fix; update existing `test_empty_passages_marks_no_passage_found` |
| `tests/unit/test_verify_multi.py` | new | 1 new test for depth aggregation |
| `tests/unit/test_verify_citing_context.py` | new | 1 new test for unbounded unsupported fix |
| `tests/unit/test_regressions.py` | new | 1 Goodwin acceptance test |

## 4. Public API

No new public functions. All changes are internal to existing functions. The only new public symbol is the `UnverifiableReason` type alias and the `ProvenanceStep.unverifiable_reason` field.

### 4.1 `src/verify.py:184-194` -- abstract-only emission

**Current code (lines 184-194):**

```python
    try:
        parsed: dict[str, Any] = json.loads(_strip_fences(response_text))
        status_raw = str(parsed["status"])
        if status_raw not in _VALID_STATUSES:
            raise ValueError(f"Invalid status: {status_raw}")
        status: VerificationStatus = status_raw  # type: ignore[assignment]
        result = VerificationResult(
            status=status,
            explanation=str(parsed["explanation"]),
            confidence=float(parsed["confidence"]),
        )
```

**Replacement:**

```python
    try:
        parsed: dict[str, Any] = json.loads(_strip_fences(response_text))
        status_raw = str(parsed["status"])
        if status_raw not in _VALID_STATUSES:
            raise ValueError(f"Invalid status: {status_raw}")
        raw_confidence = parsed.get("confidence")
        confidence_val: float | None = (
            None if raw_confidence is None else float(raw_confidence)
        )
        result = safe_verification_result(
            status=status_raw,
            confidence=confidence_val,
            explanation=str(parsed["explanation"]),
            evidence_quality="abstract_only",
        )
```

Notes:
- `safe_verification_result` (imported from `src.models`) handles the downgrade: if `status_raw in ("supported", "unsupported")` and `evidence_quality == "abstract_only"`, it returns `VerificationResult(status="unverifiable", confidence=None, ...)`.
- `evidence_quality="abstract_only"` is the default on `VerificationResult`, but passing it explicitly documents the intent.
- The LLM may now return `"unverifiable"` directly (after A3 prompt update); `safe_verification_result` passes it through with `confidence=None`.
- When the LLM returns `"unverifiable"` with a non-null confidence, `safe_verification_result` coerces confidence to `None`.

### 4.2 `src/verify_fulltext.py:78-97` -- empty-passages fallback

**Current code (lines 78-97):**

```python
    if not passages:
        from src.verify import verify_claim

        abstract_result, step = verify_claim(claim, source, model_id=model_id, api_key=api_key)
        result = dataclasses.replace(
            abstract_result,
            fulltext_available=True,            # BUG: lie
            verification_depth="abstract",
            retrieval_status="no_passage_found", # misleading
            retraction_status=source.retraction_status,
        )
        return (
            result,
            dataclasses.replace(
                step,
                input_hash=_hash(repr((claim, source, passages))),
                output_hash=_hash(repr(result)),
                confidence=result.confidence,
            ),
        )
```

**Replacement:**

```python
    if not passages:
        from src.verify import verify_claim

        abstract_result, step = verify_claim(claim, source, model_id=model_id, api_key=api_key)
        # After A2/A1: verify_claim now routes through safe_verification_result,
        # so abstract_result already has the correct status (unverifiable if the
        # LLM tried to emit supported/unsupported on abstract_only evidence).
        # We only override metadata fields here -- NOT status/confidence.
        result = dataclasses.replace(
            abstract_result,
            fulltext_available=False,                  # was: True (BUG)
            verification_depth="abstract",
            retrieval_status="fulltext_unavailable",   # was: "no_passage_found"
            retraction_status=source.retraction_status,
        )
        return (
            result,
            dataclasses.replace(
                step,
                input_hash=_hash(repr((claim, source, passages))),
                output_hash=_hash(repr(result)),
                confidence=result.confidence,
            ),
        )
```

Key behavior change: `fulltext_available=False` (truth), `retrieval_status="fulltext_unavailable"` (accurate).

No double-gating risk: `verify_claim` has already applied `safe_verification_result` in step 4.1 above. The `dataclasses.replace` here only changes metadata fields (`fulltext_available`, `verification_depth`, `retrieval_status`, `retraction_status`) -- it does NOT change `status`, `confidence`, or `evidence_quality`. Since the `abstract_result` from `verify_claim` is already valid under the A1 invariant (either `unverifiable` with `None` confidence, or a permitted combination), the replacement cannot violate the invariant.

### 4.3 `src/verify_multi.py:122-134` -- multi-source aggregation

**Current code (lines 122-134):**

```python
    aggregated_status = _aggregate_multi_source_verdicts(per_source_results)
    # Exclude confidence=0.0: these are parse-error results, not meaningful low-confidence verdicts.
    confidences = [r.confidence for r in per_source_results if r.confidence > 0]
    aggregated_confidence = sum(confidences) / len(confidences) if confidences else 0.0

    aggregated = VerificationResult(
        status=aggregated_status,
        explanation=" || ".join(explanations) if explanations else "Empty source set.",
        confidence=aggregated_confidence,
        verification_depth="abstract",
        evidence_quality="abstract_only" if per_source_results else "no_evidence",
        retraction_status=any(s.retraction_status for s in source_set),
    )
```

**Replacement:**

```python
    aggregated_status = _aggregate_multi_source_verdicts(per_source_results)
    # Exclude confidence=None (unverifiable) and confidence=0.0 (parse errors).
    confidences = [
        r.confidence for r in per_source_results
        if r.confidence is not None and r.confidence > 0
    ]
    aggregated_confidence: float | None = (
        sum(confidences) / len(confidences) if confidences else None
    )

    # Derive depth and evidence quality from the primary (best) per-source result.
    # Priority: fulltext > citing_paper_context > abstract > title_only.
    _DEPTH_PRIORITY = {"fulltext": 0, "citing_paper_context": 1, "abstract": 2, "title_only": 3}
    primary_result = min(
        per_source_results,
        key=lambda r: _DEPTH_PRIORITY.get(r.verification_depth, 99),
    ) if per_source_results else None
    agg_depth = primary_result.verification_depth if primary_result else "abstract"
    agg_evidence = primary_result.evidence_quality if primary_result else "no_evidence"

    aggregated = safe_verification_result(
        status=aggregated_status,
        confidence=aggregated_confidence,
        explanation=" || ".join(explanations) if explanations else "Empty source set.",
        verification_depth=agg_depth,
        evidence_quality=agg_evidence,
        retraction_status=any(s.retraction_status for s in source_set),
    )
```

Key changes:
1. `verification_depth` and `evidence_quality` are derived from the best per-source result, not hardcoded to `"abstract"`.
2. `aggregated_confidence` is `None` when all per-source results are either `unverifiable` (confidence=None) or parse errors (confidence=0.0).
3. Routed through `safe_verification_result()` to enforce the invariant on the aggregated combination.

### 4.4 `src/verify_citing_context.py:107-131` -- citing-context emission

**Current code (lines 107-131):**

```python
    try:
        parsed: dict[str, Any] = json.loads(_strip_fences(response_text))
        status_raw = str(parsed["status"])
        if status_raw not in _VALID_STATUSES:
            raise ValueError(f"Invalid status: {status_raw}")
        confidence = float(parsed["confidence"])
        # Hard cap: internal consistency cannot establish supported.
        if status_raw == "supported":
            status_raw = "partially_supported"
        confidence = min(confidence, _CITING_CONTEXT_MAX_CONFIDENCE)
        status: VerificationStatus = status_raw  # type: ignore[assignment]
        raw_explanation = str(parsed["explanation"])
        explanation = (
            raw_explanation
            if "internal-consistency" in raw_explanation.lower()
            else f"[Internal-consistency only] {raw_explanation}"
        )
        result = VerificationResult(
            status=status,
            explanation=explanation,
            confidence=confidence,
            verification_depth="citing_paper_context",
            evidence_quality="citing_paper_context",
            retraction_status=source.retraction_status,
        )
```

**Replacement:**

```python
    try:
        parsed: dict[str, Any] = json.loads(_strip_fences(response_text))
        status_raw = str(parsed["status"])
        if status_raw not in _VALID_STATUSES:
            raise ValueError(f"Invalid status: {status_raw}")
        confidence = float(parsed["confidence"])
        # Hard cap: internal consistency cannot establish supported.
        if status_raw == "supported":
            status_raw = "partially_supported"
        confidence = min(confidence, _CITING_CONTEXT_MAX_CONFIDENCE)
        raw_explanation = str(parsed["explanation"])
        explanation = (
            raw_explanation
            if "internal-consistency" in raw_explanation.lower()
            else f"[Internal-consistency only] {raw_explanation}"
        )
        # Route through safe_verification_result: after the supported->partially_supported
        # cap above, this catches the remaining gap: unsupported on citing_paper_context
        # evidence is downgraded to unverifiable by the A1 invariant.
        result = safe_verification_result(
            status=status_raw,
            confidence=confidence,
            explanation=explanation,
            verification_depth="citing_paper_context",
            evidence_quality="citing_paper_context",
            retraction_status=source.retraction_status,
        )
```

Key change: `unsupported` with `evidence_quality="citing_paper_context"` is now caught and downgraded to `unverifiable`. Previously only `supported` was capped (to `partially_supported`); `unsupported` passed through with unbounded confidence. After A1, that combination raises `ValueError` at `__post_init__`. The `safe_verification_result` helper prevents the crash by downgrading.

### 4.5 `src/verify_title_only.py` -- no change needed

Already compliant. The title-only verifier caps `supported` to `partially_supported` (line 102-103) and clamps confidence to `<= 0.7` (line 104). Under A1:
- `partially_supported` is exempt from Invariant 2.
- `unsupported` with `evidence_quality="title_only"` would violate Invariant 2 -- but `safe_verification_result` is NOT needed here because `unsupported` can legitimately come from title-only evidence (the title is clearly off-topic). **Wait -- this IS a violation under A1.** The A1 `__post_init__` rejects `unsupported` + `title_only`. The title-only verifier currently allows `unsupported` to pass through.

**Correction:** `verify_title_only.py` DOES need to route through `safe_verification_result()` to handle the `unsupported` + `title_only` case. Add to scope.

**Current code (lines 95-113):**

```python
    try:
        parsed: dict[str, Any] = json.loads(_strip_fences(response_text))
        status_raw = str(parsed["status"])
        if status_raw not in _VALID_STATUSES:
            raise ValueError(f"Invalid status: {status_raw}")
        confidence = float(parsed["confidence"])
        # Hard cap: title-only evidence cannot establish supported.
        if status_raw == "supported":
            status_raw = "partially_supported"
        confidence = min(confidence, _TITLE_ONLY_MAX_CONFIDENCE)
        status: VerificationStatus = status_raw  # type: ignore[assignment]
        result = VerificationResult(
            status=status,
            explanation=str(parsed["explanation"]),
            confidence=confidence,
            verification_depth="title_only",
            evidence_quality="title_only",
            retraction_status=source.retraction_status,
        )
```

**Replacement:**

```python
    try:
        parsed: dict[str, Any] = json.loads(_strip_fences(response_text))
        status_raw = str(parsed["status"])
        if status_raw not in _VALID_STATUSES:
            raise ValueError(f"Invalid status: {status_raw}")
        confidence = float(parsed["confidence"])
        # Hard cap: title-only evidence cannot establish supported.
        if status_raw == "supported":
            status_raw = "partially_supported"
        confidence = min(confidence, _TITLE_ONLY_MAX_CONFIDENCE)
        result = safe_verification_result(
            status=status_raw,
            confidence=confidence,
            explanation=str(parsed["explanation"]),
            verification_depth="title_only",
            evidence_quality="title_only",
            retraction_status=source.retraction_status,
        )
```

### 4.6 New `ProvenanceStep.unverifiable_reason` field

**New type alias in `src/models.py`:**

```python
UnverifiableReason = Literal[
    "insufficient_evidence_depth",
    "fulltext_unavailable",
    "numeric_claim_abstract_only",
] 
```

**New field on `ProvenanceStep`:**

```python
@dataclass(frozen=True)
class ProvenanceStep:
    step_id: str
    claim_id: str
    operation: OperationType
    input_hash: str
    output_hash: str
    model_id: str | None
    timestamp: float
    tokens_in: int | None
    tokens_out: int | None
    cache_hit: bool | None
    confidence: float | None
    unverifiable_reason: UnverifiableReason | None = None  # NEW
```

Default is `None` (backward compatible -- existing JSONL provenance files missing this field will deserialize correctly since the field has a default). The field is populated at each emission site where `safe_verification_result` returns `status="unverifiable"`.

**Decision rationale (add to ProvenanceStep, not only report.json):** The reason is audit-trail data, not display data. It belongs in provenance because (1) it enables programmatic querying of why verdicts were downgraded, (2) it is available even if `report.json` format changes, and (3) it costs one optional field on an already-optional schema. A4 will additionally surface it in the `report.json` per-claim record for human-readable output.

## 5. Data flow

```
LLM response JSON text
  |
  v
json.loads() + _strip_fences()
  |
  v
parsed["status"], parsed["confidence"]
  |
  v
_VALID_STATUSES check (reject unknown statuses)
  |
  v
safe_verification_result(
    status=parsed_status,        # str
    confidence=parsed_confidence,# float | None
    evidence_quality=<context>,  # EvidenceQuality from caller context
    **other_kwargs               # explanation, verification_depth, etc.
)
  |
  +-- status in ("supported","unsupported") AND evidence_quality in
  |   ("abstract_only","title_only","citing_paper_context","no_evidence")
  |     => VerificationResult(status="unverifiable", confidence=None, ...)
  |
  +-- status == "unverifiable" (LLM returned it directly, after A3 prompt)
  |     => confidence forced to None
  |     => VerificationResult(status="unverifiable", confidence=None, ...)
  |
  +-- otherwise (permitted combination)
  |     => VerificationResult(status=status, confidence=confidence, ...)
  |
  v
ProvenanceStep(
    confidence=result.confidence,  # None for unverifiable
    output_hash=_hash(repr(result)),  # hash taken AFTER downgrade
    unverifiable_reason=<reason>,  # NEW: populated when downgraded
)
```

**Determining `unverifiable_reason` at each site:**

| Emission site | `unverifiable_reason` value |
|---|---|
| `verify.py` (abstract-only) | `"insufficient_evidence_depth"` |
| `verify_fulltext.py` (empty-passages fallback) | `"fulltext_unavailable"` (set on the ProvenanceStep replacement, not the inner step from `verify_claim`) |
| `verify_multi.py` (aggregation) | `"insufficient_evidence_depth"` |
| `verify_citing_context.py` | `"insufficient_evidence_depth"` |
| `verify_title_only.py` | `"insufficient_evidence_depth"` |

The reason is determined by the caller context, not by `safe_verification_result`. The helper only decides IF to downgrade; the caller knows WHY.

## 6. External dependencies

None beyond what already exists:
- Anthropic SDK (existing) -- no prompt content changes in this track, so cache keys are preserved.
- No new libraries, APIs, env vars, or ontologies.

## 7. Edge cases

The implementer must handle each of these explicitly:

### 7.1 Empty-passages fallback double-gating

The call chain for empty passages is: `verify_claim_fulltext(passages=[])` -> `verify_claim(...)` -> `safe_verification_result(...)` -> VerificationResult. Then `verify_claim_fulltext` applies `dataclasses.replace(abstract_result, fulltext_available=False, ...)`.

**No double-gating occurs** because:
1. `verify_claim` applies `safe_verification_result` once (step 4.1).
2. `verify_claim_fulltext` only replaces metadata fields (`fulltext_available`, `verification_depth`, `retrieval_status`, `retraction_status`) -- it does NOT change `status`, `confidence`, or `evidence_quality`.
3. `dataclasses.replace` calls `__init__` then `__post_init__`. Since `status` and `evidence_quality` are unchanged from a valid combination, the invariant holds.

The implementer should add a comment at the `dataclasses.replace` site documenting this reasoning.

### 7.2 `dataclasses.replace()` in `verify_fulltext.py:82-88` after A2 fix

After A2, `abstract_result` from `verify_claim` may have `status="unverifiable"` with `confidence=None`. The `dataclasses.replace` only touches metadata fields. The resulting object will be `VerificationResult(status="unverifiable", confidence=None, fulltext_available=False, ...)` -- valid under A1's invariant. No violation.

The second `dataclasses.replace` on the step (lines 90-96) sets `confidence=result.confidence`. When result is `unverifiable`, `result.confidence` is `None`. `ProvenanceStep.confidence` already accepts `None`. Valid.

### 7.3 Parse-error fallback at `verify.py:202`

`_PARSE_ERROR_RESULT` at `src/verify_prompts.py:64-69` is `VerificationResult(status="not_addressed", confidence=0.0, evidence_quality="no_evidence")`. Under A1: `not_addressed` is exempt from Invariant 2, and `confidence=0.0` (non-None float) satisfies Invariant 1. No change needed. The catch block at `verify.py:195-202` falls through to `result = _PARSE_ERROR_RESULT` unchanged.

### 7.4 LLM returning `status="unverifiable"` directly (after A3 prompt)

After A3, the LLM may return `{"status": "unverifiable", "confidence": null, ...}`. The parse flow:
1. `status_raw = "unverifiable"` -- now in `_VALID_STATUSES` (added by A1/A2).
2. `confidence_val = None` (parsed from JSON null).
3. `safe_verification_result(status="unverifiable", confidence=None, ...)` -- passes through, returns `VerificationResult(status="unverifiable", confidence=None, ...)`.

Valid. No gating needed; the LLM is cooperating with the intent.

### 7.5 LLM returning `status="unverifiable"` with non-null confidence

Example: `{"status": "unverifiable", "confidence": 0.6}`. The parse flow:
1. `confidence_val = 0.6`.
2. `safe_verification_result(status="unverifiable", confidence=0.6, ...)` -- the helper coerces confidence to `None` (per A1 spec, line 132-133: `if status == "unverifiable": confidence = None`).
3. Returns `VerificationResult(status="unverifiable", confidence=None, ...)`.

Valid. The LLM's incorrect confidence is silently corrected.

### 7.6 LLM returning an invalid status (e.g., `"undetermined"`)

Handled by the existing `_VALID_STATUSES` check at line 187-188: `raise ValueError(f"Invalid status: {status_raw}")`. Falls through to the `except` block, which returns `_PARSE_ERROR_RESULT`. No change needed.

### 7.7 LLM returning `status="not_addressed"` on abstract-only

`not_addressed` is exempt from Invariant 2 (A1 spec, Edge case 2). `safe_verification_result` passes it through unchanged with its original confidence. No downgrade occurs. This is correct: "we checked the abstract and the source doesn't address the claim at all" is a valid finding.

### 7.8 `verify_multi.py` aggregation with mixed depths

When one source is fulltext-verified and another is abstract-only, the aggregation now uses the depth of the best source (fulltext). If the aggregated status is `supported` and the best source has `evidence_quality="quoted_passage"`, `safe_verification_result` passes it through -- correct. If all sources are abstract-only and the aggregated status is `supported`, `safe_verification_result` downgrades -- correct.

### 7.9 `verify_multi.py` with all unverifiable per-source results

When every per-source result is `unverifiable` (confidence=None), the `confidences` list after filtering is empty, so `aggregated_confidence = None`. `_aggregate_multi_source_verdicts` will produce some status from the per-source statuses -- if all are `unverifiable`, the aggregation function needs to handle this. The implementer must verify that `_aggregate_multi_source_verdicts` does not crash on all-`unverifiable` inputs and returns a sensible status. If `_aggregate_multi_source_verdicts` does not handle `unverifiable`, it should be updated to treat it like `not_addressed` for aggregation purposes. Document the decision.

### 7.10 `verify_fulltext.py:166-177` primary fulltext path -- no change

The primary fulltext path (passages non-empty) constructs `VerificationResult` with `evidence_quality="quoted_passage"` or `"passages_searched_no_quote"`. Both are fulltext-grade evidence. Under A1, `supported` and `unsupported` with these qualities are valid. No `safe_verification_result` routing needed on this path -- the direct constructor is correct.

### 7.11 Confidence parsing when LLM returns JSON `null` for confidence

After A3, the LLM may return `{"confidence": null}` for `unverifiable` verdicts. The current code at `verify.py:193` does `float(parsed["confidence"])` which would raise `TypeError` on `None`. The replacement code (Section 4.1) uses `parsed.get("confidence")` with a None check. All emission sites must handle this case. The `safe_verification_result` helper accepts `confidence: float | None`.

## 8. Behavior when input is insufficient

**(a) What counts as insufficient input:**

Any LLM response that returns `status in {"supported", "unsupported"}` when the evidence available to the verifier is `evidence_quality in {"abstract_only", "title_only", "citing_paper_context", "no_evidence"}`. The insufficiency is not about the LLM's input per se (it received the abstract), but about the epistemic gap: the abstract systematically omits Results-section data (exact percentages, p-values, CIs, sample sizes, hazard ratios), so a confident binary verdict on a claim about such data is structurally unsound.

**(b) Which schema fields signal the insufficiency:**

- `evidence_quality: EvidenceQuality` -- the primary signal. Passed as an explicit argument to `safe_verification_result()` at each emission site.
- The caller determines the evidence quality from context: `verify.py` always passes `"abstract_only"` (it only has the abstract); `verify_fulltext.py` empty-passages fallback inherits `"abstract_only"` from the inner `verify_claim` call; `verify_citing_context.py` always passes `"citing_paper_context"`.

**(c) What the module returns:**

`VerificationResult(status="unverifiable", confidence=None, evidence_quality=<original>, ...)`. The `evidence_quality` field retains the original value (e.g., `"abstract_only"`) -- it records what evidence was available, not what verdict was reached. The `status` and `confidence` are the adjusted values.

**(d) Which ProvenanceStep fields capture the uncertainty:**

- `ProvenanceStep.confidence = None` -- signals that no reliable confidence could be established.
- `ProvenanceStep.output_hash` -- hash of the `VerificationResult` including `status="unverifiable"`, so provenance diffs catch the downgrade.
- `ProvenanceStep.unverifiable_reason` (NEW) -- one of `"insufficient_evidence_depth"`, `"fulltext_unavailable"`, `"numeric_claim_abstract_only"`. Populated by the caller at each emission site.

## 9. Test plan

All tests in `tests/unit/`. All external APIs mocked per `.claude/rules/offline-tests.md`.

### Unit tests for `verify.py` emission gate

**Test 1: `test_verify_abstract_downgrades_confident_supported_to_unverifiable`**

Mock LLM returning `{"status": "supported", "confidence": 0.9, "explanation": "..."}`. Assert:
- `result.status == "unverifiable"`
- `result.confidence is None`
- `result.evidence_quality == "abstract_only"`
- ProvenanceStep `confidence is None`
- ProvenanceStep `unverifiable_reason == "insufficient_evidence_depth"`

**Test 2: `test_verify_abstract_downgrades_confident_unsupported_to_unverifiable`**

Mock LLM returning `{"status": "unsupported", "confidence": 0.75, "explanation": "..."}`. This is the exact Goodwin NEJM 2022 / 20% sustained response case. Assert:
- `result.status == "unverifiable"`
- `result.confidence is None`
- `result.evidence_quality == "abstract_only"`

**Test 3: `test_verify_abstract_preserves_not_addressed`**

Mock LLM returning `{"status": "not_addressed", "confidence": 0.9, "explanation": "..."}`. Assert:
- `result.status == "not_addressed"`
- `result.confidence == 0.9`

**Test 4: `test_verify_abstract_preserves_partially_supported`**

Mock LLM returning `{"status": "partially_supported", "confidence": 0.65, "explanation": "..."}`. Assert:
- `result.status == "partially_supported"`
- `result.confidence == 0.65`

**Test 5: `test_verify_abstract_accepts_explicit_unverifiable_from_llm`**

Mock LLM returning `{"status": "unverifiable", "confidence": null, "explanation": "..."}`. Assert:
- `result.status == "unverifiable"`
- `result.confidence is None`

### Unit tests for `verify_fulltext.py` empty-passages fix

**Test 6: `test_verify_fulltext_empty_passages_emits_fulltext_unavailable`**

Call `verify_claim_fulltext(claim, source, passages=[])` with mocked inner `verify_claim`. Assert:
- `result.fulltext_available is False`
- `result.retrieval_status == "fulltext_unavailable"`
- `result.verification_depth == "abstract"`

**Test 7: `test_verify_fulltext_empty_passages_with_abstract_supported_downgrades`**

Mock inner `verify_claim` returning `(supported, 0.9)` -- but since `verify_claim` now routes through `safe_verification_result`, it will actually return `(unverifiable, None)`. Assert:
- `result.status == "unverifiable"`
- `result.confidence is None`
- `result.fulltext_available is False`

### Unit test for `verify_multi.py` depth aggregation

**Test 8: `test_verify_multi_aggregated_depth_inherits_from_primary_source`**

Create two per-source results: one with `verification_depth="fulltext"` and `evidence_quality="quoted_passage"`, another with `verification_depth="abstract"` and `evidence_quality="abstract_only"`. Mock so the fulltext result is selected as primary. Assert:
- `aggregated.verification_depth == "fulltext"`
- `aggregated.evidence_quality == "quoted_passage"`
- `aggregated.status` is NOT hardcoded to abstract-level.

### Unit test for `verify_citing_context.py` unsupported fix

**Test 9: `test_verify_citing_context_downgrades_confident_unsupported`**

Mock LLM returning `{"status": "unsupported", "confidence": 0.55, "explanation": "..."}`. Assert:
- `result.status == "unverifiable"`
- `result.confidence is None`
- `result.evidence_quality == "citing_paper_context"`

### Unit test for `verify_title_only.py` unsupported fix

**Test 10: `test_verify_title_only_downgrades_unsupported_to_unverifiable`**

Mock LLM returning `{"status": "unsupported", "confidence": 0.6, "explanation": "..."}`. Assert:
- `result.status == "unverifiable"`
- `result.confidence is None`
- `result.evidence_quality == "title_only"`

### Goodwin acceptance test (regression pin)

**Test 11: `test_goodwin_nejm_2022_abstract_only_returns_unverifiable`**

Load regression entry `elicit_psilocybin__ae1ff864` from `eval/regressions/2026-05-11/abstract_only_unsupported/regression.jsonl`. Construct a `Claim` with `claim_text="Sustained response rates at 12 weeks were only 20% in the largest randomized trial"` and a `ResolvedSource` with `doi="10.1056/nejmoa2206443"`, `found=True`, abstract set to a representative NEJM abstract. Mock the LLM to return the original buggy response: `{"status": "unsupported", "confidence": 0.75, "explanation": "The abstract does not report a specific 20% sustained response rate..."}`. Assert:
- `result.status == "unverifiable"`
- `result.confidence is None`
- `result.evidence_quality == "abstract_only"`

This pins the specific user-caught bug. If it ever regresses, this test fails.

### Integration tests (not run in pre-commit)

None required for A2+A3. The changes are at the LLM-response parse boundary. Mocked unit tests fully cover the logic. Real LLM behavior with the new prompts should be validated via `/eval` on the dev split before declaring done.

## 10. ProvenanceStep

### Schema change: new `unverifiable_reason` field

```python
UnverifiableReason = Literal[
    "insufficient_evidence_depth",
    "fulltext_unavailable",
    "numeric_claim_abstract_only",
]

@dataclass(frozen=True)
class ProvenanceStep:
    # ... existing fields ...
    confidence: float | None
    unverifiable_reason: UnverifiableReason | None = None  # NEW, additive, optional
```

**Backward compatibility:** The field has a default value of `None`. Existing `provenance.jsonl` files that lack this field will deserialize correctly. New provenance entries include it. Additive optional field -- no migration needed.

**Where the field is populated:**

At each emission site, after `safe_verification_result` returns:

```python
reason: UnverifiableReason | None = None
if result.status == "unverifiable":
    reason = "insufficient_evidence_depth"  # or "fulltext_unavailable" in verify_fulltext

step = ProvenanceStep(
    ...,
    confidence=result.confidence,  # None for unverifiable
    unverifiable_reason=reason,
)
```

In `verify_fulltext.py` empty-passages fallback, the reason on the replacement step is `"fulltext_unavailable"` (distinct from `"insufficient_evidence_depth"` because the proximate cause is that fulltext retrieval failed, not just that the evidence depth is abstract).

**`output_hash` correctness:** The `output_hash` is computed as `_hash(repr(result))` AFTER `safe_verification_result` has returned the (possibly downgraded) result. This means the hash reflects the final `unverifiable` status, not the original LLM output. Correct for provenance diffing.

## 11. Risks

> Note: Risks 1 (prompt cache thrash) and 2 (LLM over-applies `unverifiable`) were removed in the 2026-05-11 simplification. They concerned the dropped A3 prompt clause and no longer apply. Subsequent risks retain their original numbering for stable cross-references.

### Risk 3: `verify_multi.py` depth aggregation choice

The spec chooses to derive aggregated depth from the *best available* per-source result (highest evidence quality). An alternative is to use the depth of the *chosen verdict* (the source whose status was selected by `_aggregate_multi_source_verdicts`).

**Decision:** Use best available depth. Rationale: the aggregation function picks the most informative verdict across sources. If one source provided fulltext evidence and another only abstract, the aggregated result should reflect that fulltext evidence was available and used. Using "chosen verdict depth" would require tracking which source was selected inside `_aggregate_multi_source_verdicts`, which currently returns only the aggregated status.

**Risk:** If the best-depth source has `status="not_addressed"` but another abstract-only source has `status="supported"`, the aggregated depth would be `"fulltext"` while the aggregated status came from the abstract-only source. This is misleading. The implementer should verify this edge case and document it. If it occurs frequently, switch to tracking the chosen source explicitly.

### Risk 4: `_aggregate_multi_source_verdicts` does not handle `unverifiable`

The current aggregation function was written before `unverifiable` existed. It operates on `per_source_results[i].status` and branches on `supported`, `unsupported`, `partially_supported`, `not_addressed`. If all per-source results are `unverifiable`, the function may return an unexpected status.

**Mitigation:** The implementer must read `_aggregate_multi_source_verdicts` and add handling for `unverifiable`. Recommended behavior: treat `unverifiable` like `not_addressed` for aggregation purposes (it contributes no evidence signal). If ALL per-source results are `unverifiable`, the aggregated status should be `unverifiable`.

### Risk 5: `verify_title_only.py` `unsupported` downgrade may be too aggressive

Title-only evidence can legitimately produce `unsupported` verdicts when the title is clearly off-topic (e.g., a geophysics paper cited for a medical claim). Under A1, `unsupported` + `title_only` is invalid, so all such verdicts become `unverifiable`. This loses information: "wrong paper entirely" is a stronger signal than "cannot determine."

**Mitigation:** Accept this for Phase 1. The title-only path is already a last resort (used only when no abstract is available). The information loss is minor. The correct long-term fix (Phase 3+) is to introduce a separate `"off_topic"` status or refine the evidence_quality taxonomy to distinguish "title confirms off-topic" from "title insufficient."

## 12. Out of scope

| Feature | Why refused |
|---|---|
| `report.py` summary changes for `unverifiable` count | A4 scope. A2 delivers the correct verdicts; A4 surfaces them in the summary. |
| `pipeline.py` status branching for `unverifiable` | A4 scope. The pipeline does not branch on individual statuses in Phase 1; it passes `VerificationResult` through to `report.py`. |
| Fulltext fetcher improvements (HTML fallback, PDF paywall detection) | Track D scope. A2 makes the abstract-only path safe; Track D reduces how often it is needed. |
| Workflow defense (rule files, reviewer/dogfooder/skillifier updates) | Track B scope. Independent of A2/A3. |
| Recomputing contaminated benchmark numbers | Track C scope. Blocked on A+D landing. |
| `"off_topic"` status for clear-cut wrong-paper verdicts on title-only | Phase 3+ taxonomy refinement. Premature now. |
| Numeric heuristic regex (`_claim_has_specific_numeric`) | The plan mentions this as a belt-and-braces alongside the Python gate. The `safe_verification_result` helper already catches ALL `supported`/`unsupported` on `abstract_only` regardless of claim content. The regex would provide a more targeted gate (only numeric claims), but the universal gate is both simpler and stricter. If the universal gate proves too aggressive (measured via `/eval`), a future spec can add the regex as a narrow-gate alternative. Do not implement the regex in A2. |
| `report.json` deserialization migration layer | A4 scope (see A1 Risk 3). |

---

## A3: Prompt changes (REMOVED — see Decision log at top)

The original A3 plan included appending an `unverifiable` instruction clause to both `src/prompts/verify_v1.md` and `src/prompts/verify_fulltext_v1.md`, plus updating the in-prompt JSON schema. **All of this is cancelled per the 2026-05-11 decision** documented in the Decision log at the top of this spec. Rationale: the deterministic `safe_verification_result` helper (Track A1) enforces the same invariant from Python without LLM cooperation. The prompt clause would be redundant, would cost extra tokens per request, would invalidate prompt-cache keys on deploy, and would introduce a recall-risk surface (the LLM might over-apply `unverifiable` to qualitative claims). Drop it; rely on the Python gate.

The single surviving piece of A3: add `"unverifiable"` to `_VALID_STATUSES` in `src/verify_prompts.py` so that if the LLM ever returns it (e.g. from a future prompt-tuning experiment, or via instruction-following bleed from the type alias name), the parser accepts the value and the helper handles the rest. The LLM is NOT instructed to use it.
