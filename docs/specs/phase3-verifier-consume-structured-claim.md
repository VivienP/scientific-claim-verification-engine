# Spec: Phase 3 -- Verifier consumes structured Claim fields

**Date:** 2026-05-23
**Author:** @architect
**Phase:** Phase 1 (MVP)
**Branch:** `feat/extract-negative-controls` (merged to `main` as `587626b`)
**Depends on:** Phase 2 / 2.5 structured Claim fields (merged), `docs/specs/unverifiable-verdict-emission-gates.md` (Tracks A1 + A2 + A3, implemented)
**Deliverables:** 4 commits (3.1, 3.2, 3.3, 3.4)

---

## 1. Goal

Wire the verifier layer to consume the three new structured Claim fields (`source_quote`, `extraction_confidence`, `extraction_confidence` threshold gate) and extend the existing `safe_verification_result` downgrade gate to qualitative claims on insufficient evidence, closing the P1-2 incident class of silent confident false-positive verdicts.

## 2. Scope

**In:**
- 3.1: Consume `source_quote` as a focal-text anchor in verifier user messages (when non-null); no-op when absent.
- 3.2: Surface `extracted_source_quote` in per-claim evidence records in `report.json`.
- 3.3: Deterministic extraction-confidence gate in `safe_verification_result` -- cap verdict to `partially_supported` when `extraction_confidence < threshold`.
- 3.4: Remove claim-type discrimination from the insufficient-evidence downgrade gate in `safe_verification_result`, applying it to qualitative claims as well (Option A -- see Section 3.4 recommendation).
- New `UnverifiableReason` literal for 3.3.
- Unit tests for all new behaviors.

**Out:**
- `src/extract.py` changes -- Phase 2.5 is done.
- `pipeline.py` status branching -- Track A4.
- Numeric heuristic refinement (`_claim_has_specific_numeric`).
- New regression test infrastructure beyond the specific assertions needed here.
- The two P2 WARNINGs from @reviewer (print() in audit script; truncated flag on ProvenanceStep).
- Prompt file content changes (`src/prompts/*.md`) -- 3.1 modifies user messages in Python code only.

## 3. Files touched

| Path | Role |
|---|---|
| `src/verify.py` | 3.1: Inject `source_quote` focal anchor into user_message. 3.3: Pass `extraction_confidence` to `safe_verification_result`. |
| `src/verify_fulltext.py` | 3.1: Inject `source_quote` focal anchor into user_message (fulltext path). |
| `src/verify_title_only.py` | 3.3: Pass `extraction_confidence` to `safe_verification_result`. |
| `src/verify_citing_context.py` | 3.3: Pass `extraction_confidence` to `safe_verification_result`. |
| `src/verify_multi.py` | 3.3: Pass `extraction_confidence` to `safe_verification_result` at aggregation site. |
| `src/models.py` | 3.3: Extend `safe_verification_result` signature with `extraction_confidence: float | None = None`; add verdict-cap logic. Add `"low_extraction_confidence"` to `UnverifiableReason`. 3.4: Remove `_claim_has_specific_numeric` gate from `safe_verification_result`. |
| `src/report.py` | 3.2: Include `extracted_source_quote` in per-claim record. |
| `tests/unit/test_models.py` | 3.3 + 3.4: New tests for extraction_confidence gate and qualitative-claim downgrade. |
| `tests/unit/test_verify_abstract.py` | 3.1: Test source_quote anchor injection. 3.3: Test extraction_confidence cap. 3.4: Test qualitative claim downgrade. |
| `tests/unit/test_verify_fulltext.py` | 3.1: Test source_quote anchor injection on fulltext path. |

## 4. Public API

### 4.1 `safe_verification_result` (modified signature)

```python
def safe_verification_result(
    *,
    status: str,
    confidence: float | None,
    evidence_quality: EvidenceQuality = "abstract_only",
    claim_text: str | None = None,
    extraction_confidence: float | None = None,  # NEW (3.3)
    unverifiable_reason: UnverifiableReason | None = None,
    **kwargs: Any,
) -> VerificationResult:
```

New parameter `extraction_confidence` has default `None` for backward compatibility. All existing call sites continue to work unchanged until explicitly updated.

### 4.2 `UnverifiableReason` (extended literal)

```python
UnverifiableReason = Literal[
    "insufficient_evidence_depth",
    "fulltext_unavailable",
    "numeric_claim_abstract_only",
    "parse_error",
    "resolution_low_confidence",
    "resolution_source_disagreement",
    "low_extraction_confidence",  # NEW (3.3)
]
```

### 4.3 No new public functions

All changes are internal to existing functions. The `source_quote` anchor is a user-message change, not a new API surface.

## 5. Data flow

### 5.1 source_quote focal anchor (3.1)

```
Claim.source_quote (str | None)
  |
  +-- None (~90% of claims): user_message unchanged
  |     "<claim>{claim_text}</claim>\n<source>{abstract}</source>"
  |
  +-- non-None (~10% of claims): prepend focal anchor to user_message
        "<claim>{claim_text}</claim>\n"
        "<source_quote>{source_quote}</source_quote>\n"
        "<source>{abstract}</source>"
```

The anchor goes in the **user message** (not system prompt) because it is per-call content and would break prompt cache if placed in the system prompt. The `<source_quote>` XML tag is distinct from the verifier's `source_passages` output field. The LLM sees it as additional context but the system prompt is unchanged.

Same pattern in `verify_fulltext.py`: when `source_quote` is non-null, prepend the `<source_quote>` block before the `<passages>` block.

Not injected into `verify_title_only.py` or `verify_citing_context.py` -- these are last-resort paths where the source_quote (tied to the input document, not the cited paper) would not add diagnostic value and could confuse the verifier about what constitutes "source evidence."

### 5.2 report.json extracted_source_quote (3.2)

```
Per-claim record in report.json:
{
    "claim_id": "...",
    "claim_text": "...",
    "claim_type": "...",
    "cited_authors": [...],
    "cited_year": ...,
    "extracted_source_quote": "verbatim text or null",  // NEW
    "source": { ... },
    "verification": { ... }
}
```

Field `extracted_source_quote` is set to `claim.source_quote` (which is `str | None`). Defaults to `null` in JSON when the extractor did not populate it. Named `extracted_source_quote` to avoid collision with the verifier's `source_passages` (which are quotes from the SOURCE paper, not the input text).

### 5.3 extraction_confidence gate (3.3)

```
safe_verification_result(
    status=<LLM verdict>,
    confidence=<LLM confidence>,
    extraction_confidence=claim.extraction_confidence,  # float | None
    ...
)
  |
  +-- extraction_confidence is None: no cap applied (backward compat)
  |
  +-- extraction_confidence >= _EXTRACTION_CONFIDENCE_THRESHOLD (0.5):
  |     no cap applied
  |
  +-- extraction_confidence < _EXTRACTION_CONFIDENCE_THRESHOLD (0.5):
  |     AND status in ("supported", "unsupported"):
  |       status -> "partially_supported"
  |       confidence -> min(confidence, extraction_confidence)
  |       unverifiable_reason field NOT set (this is a cap, not an unverifiable)
  |
  |     AND status == "partially_supported":
  |       confidence -> min(confidence, extraction_confidence)
  |       (no status change)
  |
  |     AND status in ("not_addressed", "unverifiable"):
  |       no change (these are already non-confident)
```

**Design decision: cap to `partially_supported`, not `unverifiable`.** Low extraction confidence means the *extractor* was uncertain about what the claim says, not that the *evidence* is insufficient. The claim may be poorly extracted but the source paper may genuinely address it. Downgrading to `unverifiable` would be epistemically wrong -- the evidence might be fine; the question is garbled. Capping to `partially_supported` with reduced confidence correctly signals "we are less sure about this verdict because the claim extraction itself was uncertain." The `unverifiable_reason` is NOT set in this case because the claim is not unverifiable -- it is hedged.

**Exception:** When the evidence-depth gate (existing or extended by 3.4) ALSO fires, the more aggressive gate wins (unverifiable trumps partially_supported). Order of operations: extraction_confidence cap runs FIRST, then the evidence-depth gate runs on the (possibly capped) status. This means a low-extraction-confidence claim on abstract-only evidence still becomes `unverifiable`, not `partially_supported`.

**Threshold justification (0.5):** The extraction_confidence field is LLM self-reported confidence in the extraction (not calibrated). Empirical finding from Phase 2.5.4: it populates at 100% of claims. The distribution is heavily right-skewed (most claims >0.8). A threshold of 0.5 catches only clearly uncertain extractions while leaving the vast majority of claims untouched. This is conservative by design -- we can tighten to 0.6 or 0.7 after observing the distribution on more diverse inputs. The threshold is a module-level constant (`_EXTRACTION_CONFIDENCE_THRESHOLD = 0.5`) for easy adjustment.

**Placement decision: inside `safe_verification_result`, not in `verify.py`.** The extraction_confidence cap is a deterministic post-parse gate on a structured field -- the same contract as the evidence-depth gate. Placing it in the helper keeps all deterministic verdict-modification logic in one function, makes it testable in isolation (via `test_models.py`), and ensures every call site that routes through the helper gets the cap automatically. The alternative (handling in each `verify_*.py` file after the parse boundary) would scatter the logic across 5 files and risk inconsistent application. The helper's signature widens by one `Optional` parameter with a `None` default -- fully backward compatible.

### 5.4 Qualitative claim downgrade (3.4)

```
BEFORE (current):
safe_verification_result gate fires only when:
  1. status in ("supported", "unsupported")
  2. evidence_quality in _INSUFFICIENT_EVIDENCE_SET
  3. claim_text is None OR _claim_has_specific_numeric(claim_text)

AFTER (3.4, Option A):
safe_verification_result gate fires when:
  1. status in ("supported", "unsupported")
  2. evidence_quality in _INSUFFICIENT_EVIDENCE_SET
  (condition 3 removed entirely)
```

The `_claim_has_specific_numeric` call and the `claim_text` conditional are removed from `safe_verification_result`. The gate becomes claim-type-agnostic: ANY `(supported|unsupported)` verdict on insufficient evidence is downgraded to `(unverifiable, confidence=None)`, regardless of whether the claim contains numeric patterns.

**Scope recommendation: Option A (MIRROR -- all 4 evidence modes, all claim types).**

See Section 3.4 Recommendation below.

### 5.4.1 Scope recommendation for 3.4 -- Option A (MIRROR)

The three options, with operational analysis of each verifier path:

**Option A -- MIRROR (all 4 modes, remove claim-type discrimination entirely):**

Removes the `_claim_has_specific_numeric` check entirely from `safe_verification_result`. Any `(supported|unsupported)` on `{abstract_only, title_only, citing_paper_context, no_evidence}` is downgraded regardless of claim type.

Impact on each verifier path:
- `verify.py` (abstract-only, line 199): Currently downgrades numeric claims only. After 3.4: also downgrades qualitative claims like "psilocybin reduces depression" when the abstract is the only evidence. The P1-2 incident class (Goodwin NEJM) was exactly this pattern -- a qualitative directional claim on abstract-only evidence produced a false positive.
- `verify_title_only.py` (line 109): Already routes through `safe_verification_result` with `claim_text`. Currently, `supported` is capped to `partially_supported` (which is exempt from the gate), so only `unsupported` on title-only is caught. After 3.4: same behavior -- `unsupported` on title-only is still downgraded to `unverifiable`. No new behavioral change for this path. Risk 5 from the existing spec (over-aggressive downgrade on clearly off-topic titles) remains accepted for Phase 1.
- `verify_citing_context.py` (line 126): `supported` is already capped to `partially_supported`. Only `unsupported` on citing-paper-context is caught. After 3.4: same behavior -- qualitative `unsupported` on citing-context is also downgraded. This is correct: citing-paper internal consistency is insufficient to conclude the cited paper contradicts the claim, regardless of claim type.
- `verify_multi.py` (line 292): Aggregation routes through `safe_verification_result`. After 3.4: aggregated qualitative verdicts on insufficient evidence are also downgraded. This is correct: if all sources are abstract-only, the aggregated verdict should reflect the same insufficiency.
- `verify_fulltext.py` (line 197): Does NOT route through `safe_verification_result` (direct `VerificationResult` construction). Evidence quality is `quoted_passage` or `passages_searched_no_quote` -- both are fulltext-grade and NOT in `_INSUFFICIENT_EVIDENCE_SET`. After 3.4: no change on this path. Fulltext verdicts are unaffected.

**Option B -- MINIMAL (abstract_only only for qualitative):**
Would add `abstract_only` specifically for qualitative claims but keep `title_only`, `citing_paper_context`, `no_evidence` as numeric-only gates. This creates a confusing asymmetry: why would citing-paper-context evidence be sufficient for qualitative `unsupported` but not for numeric `unsupported`? Both are epistemically insufficient. This also requires adding a *second* conditional branch in `safe_verification_result` (one for numeric, one for qualitative-abstract-only), making the function harder to reason about.

**Option C -- Hybrid (abstract_only + no_evidence for qualitative):**
A middle ground that still leaves `title_only` and `citing_paper_context` asymmetric. Same complexity problem as Option B, with less justification for the split.

**Recommendation: Option A.**

Justification:
1. **Symmetry is simpler.** Removing condition 3 entirely is a one-line deletion. Options B and C add new conditional branches. Simplicity first (CLAUDE.md section 2).
2. **The original justification was wrong.** The Phase 2 decision log (emission-gates spec line 14) said "the abstract is sufficient for 'X reduces Y'-style verdicts when it directly addresses the topic." The P1-2 incident proved otherwise: Elicit's "sustained response at 12 weeks was 20%" was qualitative-adjacent (the 20% is a detail, not the core direction claim), and the verifier emitted a false positive. The asymmetry was always a latent bug.
3. **Risk 5 (verify_title_only over-aggressive downgrade) is accepted.** The existing spec already accepted this risk for Phase 1. Making the gate symmetric does not increase the risk surface -- `verify_title_only` already routes through `safe_verification_result` and already downgrades numeric `unsupported`.
4. **`partially_supported` and `not_addressed` are unaffected.** The gate only fires on `(supported|unsupported)`. `partially_supported` is a hedge and passes through. `not_addressed` is a valid finding and passes through. The information loss is limited to "the abstract directly addresses the qualitative claim" verdicts, which are now `unverifiable` -- a conservative but correct outcome when we cannot access the full text.

**What is lost:** ~20-40% of currently-passing qualitative verdicts on abstract-only evidence will become `unverifiable`. This is a precision trade for safety. The /eval run will quantify the actual regression. If F1 drops >2%, the regression gate blocks the commit and we revisit.

## 6. External dependencies

None new. All changes use existing imports:
- `anthropic` SDK (existing)
- `structlog` (existing)
- No new libraries, APIs, env vars, or ontologies

## 7. Edge cases

### 7.1 source_quote is a paraphrase (90% of the time)

Phase 2.5.4 empirical finding: the `_validate_source_quote` check in the extractor rejects ~90% of LLM "quotes" as paraphrases. When rejected, `source_quote` is `None`. The verifier must handle `None` as the common case, not the exception. No focal anchor is prepended; user message format is unchanged.

### 7.2 source_quote contains XML-like content

The source_quote is wrapped in `<source_quote>...</source_quote>` XML tags in the user message. If the quote itself contains `<` or `>` characters (common in scientific text: `p < 0.05`), the Anthropic API handles raw text within content blocks without XML parsing. No escaping needed -- the XML tags are structural hints for the LLM, not parsed XML.

### 7.3 extraction_confidence is None (legacy claims / v1 extractor)

Claims constructed by v1 extractor or pre-existing fixtures have `extraction_confidence=None`. The gate in `safe_verification_result` must skip the cap when `extraction_confidence is None`. The default parameter value is `None`, so all existing call sites are unaffected.

### 7.4 extraction_confidence exactly at threshold (0.5)

The gate fires on `< 0.5`, not `<= 0.5`. At exactly 0.5, no cap is applied. This is a convention choice (half-open interval on the lower side); the difference is negligible given the field is not calibrated.

### 7.5 extraction_confidence cap + evidence-depth gate interaction

A claim with `extraction_confidence=0.3` (below threshold) AND `evidence_quality="abstract_only"` hits both gates. Order of operations in `safe_verification_result`:
1. Extraction confidence gate: status "supported" -> "partially_supported", confidence capped.
2. Evidence-depth gate: status "partially_supported" is NOT in `("supported", "unsupported")`, so the depth gate does NOT fire.

This is the correct outcome: the claim is hedged (partially_supported) rather than fully unverifiable. The extraction uncertainty is already expressed in the reduced confidence. If the LLM returned `"unsupported"` instead:
1. Extraction confidence gate: "unsupported" -> "partially_supported".
2. Evidence-depth gate: "partially_supported" is exempt. Pass through.

If we want the depth gate to still fire after extraction cap, we would need to check the *original* status, not the capped status. **Decision: do NOT do this.** The extraction cap already expresses uncertainty; stacking it with `unverifiable` would double-penalize. A claim with low extraction confidence that the LLM called "unsupported" on abstract-only evidence should be `partially_supported` (uncertain extraction + uncertain evidence = hedge, not full unverifiability).

### 7.6 extraction_confidence cap on fulltext-verified claims

Fulltext-verified claims have `evidence_quality` in `{"quoted_passage", "passages_searched_no_quote"}`. These are NOT in `_INSUFFICIENT_EVIDENCE_SET`, so the evidence-depth gate never fires. But the extraction_confidence cap is independent of evidence quality -- it applies to all claim types at all evidence depths. A fulltext-verified claim with `extraction_confidence=0.3` is capped to `partially_supported`. This is correct: even with great evidence, if the claim itself was poorly extracted, the verdict should be hedged.

### 7.7 safe_verification_result called without extraction_confidence from verify_multi aggregation

The `verify_multi.py` aggregation site calls `safe_verification_result` for the aggregated verdict. The aggregated claim is the same `Claim` object, so `claim.extraction_confidence` is available. The call site should pass it. If the aggregated status is already `unverifiable` (all sources unverifiable), the extraction_confidence cap is a no-op (status not in `("supported", "unsupported", "partially_supported")`).

### 7.8 3.4 removes _claim_has_specific_numeric from safe_verification_result but not from codebase

The function `_claim_has_specific_numeric` in `src/numeric/heuristics.py` is still used by `src/policy/evidence_sufficiency.py` (the pre-LLM policy gate). It must NOT be deleted from the codebase -- only the import and usage in `safe_verification_result` are removed. The `claim_text` parameter on `safe_verification_result` becomes unused by the evidence-depth gate but is still accepted (for backward compat and for potential future use). Its removal from the signature is out of scope.

## 8. Behavior when input is insufficient

### 8a. What counts as insufficient input

Three independent insufficiency conditions, each handled by a separate gate:

1. **Insufficient evidence depth** (existing gate, extended by 3.4): `evidence_quality in {"abstract_only", "title_only", "citing_paper_context", "no_evidence"}` with verdict in `{"supported", "unsupported"}`. After 3.4: applies to ALL claim types, not just numeric.

2. **Low extraction confidence** (new gate, 3.3): `extraction_confidence is not None AND extraction_confidence < 0.5` with verdict in `{"supported", "unsupported", "partially_supported"}`.

3. **Missing source_quote** (3.1): NOT an insufficiency condition. `source_quote=None` is the expected common case (~90%). The verifier falls back to current behavior. No verdict modification.

### 8b. Which schema fields signal the insufficiency

1. Evidence depth: `evidence_quality: EvidenceQuality` on `VerificationResult`.
2. Extraction confidence: `extraction_confidence: float | None` on `Claim` (input to `safe_verification_result` as a parameter).
3. Source quote: `source_quote: str | None` on `Claim` (consumed in user message construction, not a verdict gate).

### 8c. What the module returns/raises

1. Evidence depth gate fires: `VerificationResult(status="unverifiable", confidence=None, unverifiable_reason=<caller-specified>)`.
2. Extraction confidence gate fires: `VerificationResult(status="partially_supported", confidence=min(original_confidence, extraction_confidence))`. This is a cap, NOT an unverifiable -- the claim is hedged, not abandoned. `unverifiable_reason` is NOT set.

   NOTE: When both gates fire (low extraction_confidence AND insufficient evidence), the extraction cap runs first (status becomes `partially_supported`), which then exempts the claim from the evidence-depth gate (which only fires on `supported|unsupported`). See edge case 7.5.

3. Source quote absent: no change to output.

### 8d. Which ProvenanceStep fields capture the uncertainty

1. Evidence depth downgrade: `ProvenanceStep.confidence=None`, `ProvenanceStep.unverifiable_reason=<reason>` (existing behavior, unchanged).
2. Extraction confidence cap: `ProvenanceStep.confidence` reflects the capped confidence value. No `unverifiable_reason` is set (the claim is not unverifiable). The cap is visible in the provenance as a confidence lower than the LLM's raw output -- the `output_hash` captures the final result. A structlog `logger.info("extraction_confidence_cap", ...)` at the cap site provides auditable telemetry.

## 9. Test plan

All tests in `tests/unit/`. All Anthropic SDK calls mocked. No real API calls.

### 3.1 Tests (source_quote focal anchor)

**Test A1: `test_verify_abstract_includes_source_quote_anchor`**
File: `tests/unit/test_verify_abstract.py`
Claim with `source_quote="The incidence of sustained response at week 12 was 20%"`. Mock Anthropic client. Assert that the user message passed to `client.messages.create` contains `<source_quote>The incidence of sustained response at week 12 was 20%</source_quote>`.

**Test A2: `test_verify_abstract_omits_anchor_when_source_quote_none`**
File: `tests/unit/test_verify_abstract.py`
Claim with `source_quote=None`. Assert user message does NOT contain `<source_quote>`.

**Test A3: `test_verify_fulltext_includes_source_quote_anchor`**
File: `tests/unit/test_verify_fulltext.py`
Claim with non-null `source_quote`, non-empty passages. Assert user message contains `<source_quote>` block before `<passages>`.

### 3.2 Tests (report.json extracted_source_quote)

**Test B1: `test_report_claim_record_includes_extracted_source_quote`**
File: `tests/unit/test_models.py` or a new `tests/unit/test_report.py` if one exists.
Call `build_report` with a claim that has `source_quote="some text"`. Load the written `report.json`. Assert `claims[0]["extracted_source_quote"] == "some text"`.

**Test B2: `test_report_claim_record_null_when_source_quote_absent`**
Call `build_report` with a claim that has `source_quote=None`. Assert `claims[0]["extracted_source_quote"]` is `None` (JSON `null`).

### 3.3 Tests (extraction_confidence gate)

**Test C1: `test_safe_verification_result_caps_supported_on_low_extraction_confidence`**
File: `tests/unit/test_models.py`
```python
result = safe_verification_result(
    status="supported",
    confidence=0.9,
    evidence_quality="quoted_passage",  # fulltext-grade
    extraction_confidence=0.3,
    explanation="...",
)
assert result.status == "partially_supported"
assert result.confidence == 0.3  # min(0.9, 0.3)
```

**Test C2: `test_safe_verification_result_caps_unsupported_on_low_extraction_confidence`**
```python
result = safe_verification_result(
    status="unsupported",
    confidence=0.8,
    evidence_quality="quoted_passage",
    extraction_confidence=0.4,
    explanation="...",
)
assert result.status == "partially_supported"
assert result.confidence == 0.4
```

**Test C3: `test_safe_verification_result_no_cap_when_extraction_confidence_above_threshold`**
```python
result = safe_verification_result(
    status="supported",
    confidence=0.9,
    evidence_quality="quoted_passage",
    extraction_confidence=0.8,
    explanation="...",
)
assert result.status == "supported"
assert result.confidence == 0.9
```

**Test C4: `test_safe_verification_result_no_cap_when_extraction_confidence_none`**
```python
result = safe_verification_result(
    status="supported",
    confidence=0.9,
    evidence_quality="quoted_passage",
    extraction_confidence=None,
    explanation="...",
)
assert result.status == "supported"
assert result.confidence == 0.9
```

**Test C5: `test_safe_verification_result_extraction_cap_then_evidence_gate_interaction`**
Low extraction_confidence + abstract_only evidence: extraction cap fires first (supported -> partially_supported), then evidence gate does NOT fire (partially_supported is exempt).
```python
result = safe_verification_result(
    status="supported",
    confidence=0.9,
    evidence_quality="abstract_only",
    extraction_confidence=0.3,
    explanation="...",
)
assert result.status == "partially_supported"
assert result.confidence == 0.3
# NOT unverifiable: extraction cap preempts the evidence gate
```

**Test C6: `test_safe_verification_result_not_addressed_unaffected_by_low_extraction_confidence`**
```python
result = safe_verification_result(
    status="not_addressed",
    confidence=0.9,
    evidence_quality="abstract_only",
    extraction_confidence=0.2,
    explanation="...",
)
assert result.status == "not_addressed"
assert result.confidence == 0.9
```

### 3.4 Tests (qualitative claim downgrade)

**Test D1: `test_safe_verification_result_downgrades_qualitative_supported_on_abstract`**
File: `tests/unit/test_models.py`
```python
result = safe_verification_result(
    status="supported",
    confidence=0.9,
    evidence_quality="abstract_only",
    claim_text="psilocybin reduces depression symptoms",  # qualitative, no numeric
    explanation="abstract directly addresses",
)
assert result.status == "unverifiable"
assert result.confidence is None
```
This test REPLACES the existing `test_helper_passes_qualitative_supported_on_abstract_through` (which asserts `status=="supported"` and will now fail).

**Test D2: `test_safe_verification_result_downgrades_qualitative_unsupported_on_abstract`**
```python
result = safe_verification_result(
    status="unsupported",
    confidence=0.75,
    evidence_quality="abstract_only",
    claim_text="Protein folding rates increase with temperature",
    explanation="abstract contradicts",
)
assert result.status == "unverifiable"
assert result.confidence is None
```

**Test D3: `test_safe_verification_result_qualitative_not_addressed_passes_through`**
```python
result = safe_verification_result(
    status="not_addressed",
    confidence=0.9,
    evidence_quality="abstract_only",
    claim_text="psilocybin reduces depression symptoms",
    explanation="abstract silent",
)
assert result.status == "not_addressed"
assert result.confidence == 0.9
```

**Test D4: `test_safe_verification_result_qualitative_partially_supported_passes_through`**
```python
result = safe_verification_result(
    status="partially_supported",
    confidence=0.6,
    evidence_quality="abstract_only",
    claim_text="psilocybin reduces depression symptoms",
    explanation="partial match",
)
assert result.status == "partially_supported"
assert result.confidence == 0.6
```

### Integration tests

None required for Phase 3. All changes are at LLM-response parse boundaries or deterministic gates. Mocked unit tests fully cover the logic. Real LLM behavior validated via `/eval` before merge.

## 10. ProvenanceStep

### Existing emission points (unchanged)

All existing ProvenanceStep emission points in `verify.py`, `verify_fulltext.py`, `verify_title_only.py`, `verify_citing_context.py`, `verify_multi.py` continue to emit as-is. The `unverifiable_reason` field is already populated when `safe_verification_result` returns `status="unverifiable"`.

### New gate: extraction_confidence cap (3.3)

When the extraction_confidence cap fires (status capped to `partially_supported`), the ProvenanceStep at each emission site records:
- `confidence = result.confidence` (the capped value)
- `unverifiable_reason = None` (NOT set -- the claim is not unverifiable)

The cap is detectable from provenance by comparing `output_hash` against the raw LLM response hash (which would be different). Additionally, a structlog event `extraction_confidence_cap` is emitted at the `safe_verification_result` call site for real-time observability.

### New gate: qualitative claim downgrade (3.4)

Operationally identical to the existing numeric-claim downgrade. When a qualitative claim on insufficient evidence is downgraded:
- `ProvenanceStep.confidence = None`
- `ProvenanceStep.unverifiable_reason = <caller-specified reason>` (e.g., `"insufficient_evidence_depth"`)

No new ProvenanceStep fields needed. The existing `unverifiable_reason` literal `"insufficient_evidence_depth"` covers this case. The `"numeric_claim_abstract_only"` reason becomes vestigial (no longer the default) but remains in the `UnverifiableReason` literal for backward compatibility with existing provenance.jsonl files.

### Caller-side ProvenanceStep population (3.3)

At each emission site (e.g., `verify.py` line 218-231), the existing pattern:
```python
unverifiable_reason=result.unverifiable_reason,
```
continues to work correctly. The `VerificationResult` already carries `unverifiable_reason` when the evidence-depth gate fires. For the extraction_confidence cap, `unverifiable_reason` is `None` on the result, so `None` is propagated to the step -- correct.

## 11. Risks

### Risk 1: Qualitative claim downgrade (3.4) causes F1 regression on SciFact

Removing the `_claim_has_specific_numeric` exemption will cause currently-passing qualitative claims on abstract-only evidence to become `unverifiable`. SciFact claims are predominantly abstract-verified (the dataset includes abstracts, not full text). Many claims are qualitative. A significant fraction of the current "supported" verdicts may flip to "unverifiable."

**Mitigation:** Run `/eval` on the dev split immediately after implementation. If F1 drops >2%, the regression gate blocks the commit. Possible recovery: revert to Option B (abstract_only only) or introduce a partial-support path for qualitative claims where the abstract directly discusses the topic.

**Likelihood:** Medium-high. SciFact is designed around abstract-level verification, so the dataset may be adversarial to this change.

### Risk 2: extraction_confidence threshold too low (0.5) or too high

The field is LLM self-reported and uncalibrated. At 0.5, the gate may fire too rarely to be useful (if the distribution is heavily right-skewed above 0.5) or too often (if certain claim types systematically get low confidence).

**Mitigation:** The threshold is a named constant (`_EXTRACTION_CONFIDENCE_THRESHOLD`). Log all extraction_confidence values via structlog for offline analysis. Adjust after observing the distribution on 3+ diverse inputs.

### Risk 3: source_quote focal anchor confuses the verifier LLM

Adding `<source_quote>...</source_quote>` to the user message is a new input shape the verifier prompt was not designed for. The LLM might (a) over-anchor on the quote and ignore relevant abstract content, or (b) treat the quote as authoritative evidence rather than a hint.

**Mitigation:** The anchor is intentionally minimal (one XML-tagged block prepended to the existing user message). The system prompt is unchanged, so the LLM's task framing is preserved. Since source_quote is present in only ~10% of claims, the impact surface is small. Monitor via `/dogfood` and `/eval` for any verdict-quality changes on claims with vs. without the anchor.

## 12. Out of scope

| Feature | Why refused |
|---|---|
| Prompt changes (`src/prompts/*.md`) to reference source_quote | The anchor lives in the user message (per-call), not the system prompt (cached). Modifying prompts for this would break cache keys and add token cost for no benefit. |
| `extraction_confidence` calibration or bucketing | Phase 3+ concern. The field is LLM self-reported; proper calibration requires a ground-truth dataset of "correct extractions" which does not exist yet. |
| Adjusting `_claim_has_specific_numeric` patterns | Out of scope per constraint. The heuristic is used by the policy module and is unaffected by 3.4's change to `safe_verification_result`. |
| `pipeline.py` status branching for `partially_supported` from extraction cap | Track A4 scope. |
| Structured PICO field consumption by the verifier (subject, population, intervention, etc.) | Phase 4+ scope. These fields are available on the Claim but the verifier prompts would need substantial redesign to use them effectively. |
| `report.json` changes to surface extraction_confidence per claim | Not requested. The field is on the Claim dataclass which is already serialized via `dataclasses.asdict(claim)` in the source record -- it is already visible in report.json under `source.extraction_confidence` (implicit from the Claim serialization). Wait, Claim is NOT serialized via asdict in report.py -- only specific fields are picked. Adding it would be a scope expansion. Leave for a future spec. |
