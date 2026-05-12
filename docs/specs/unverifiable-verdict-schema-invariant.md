# Spec: `unverifiable` verdict schema invariant (Track A1)

## Decision log

**2026-05-11 — Numeric-claim discrimination added to helper.** Per
`memory/feedback_prefer_deterministic_gates.md`, the A3 prompt-clause
plan was dropped: the helper enforces the invariant from Python without
LLM cooperation. But the helper as originally specified downgrades
every `(supported|unsupported, abstract_only)` combination, which
over-fires on legitimate qualitative-claim verdicts (SciFact-style
"X reduces Y" claims where the abstract directly addresses it). The
helper now takes an optional `claim_text` parameter and applies a
deterministic regex (`_claim_has_specific_numeric`) to decide
whether the claim is structurally unverifiable from abstract-level
evidence. Numeric/Results-section claims (percentages, p-values, CIs,
hazard ratios, exact n=, response rates with timepoints) on
insufficient evidence are downgraded; qualitative claims pass through.
This preserves recall on qualitative claims while still catching the
Goodwin-style false-positive class.


**Date:** 2026-05-11
**Author:** @architect
**Phase:** Phase 1 (MVP)
**Parent plan:** `reports/that-s-very-bad-the-stateful-sloth.md`
**Track:** A1 only (schema change). Verifier emission = A2, prompts = A3, consumers = A4, fulltext fetcher = D.

---

## 1. Goal

Add an `"unverifiable"` status to the verification schema and enforce at construction time that confident positive/negative verdicts (`supported`, `unsupported`) cannot be emitted on insufficient evidence (`abstract_only`, `title_only`, `citing_paper_context`, `no_evidence`). This makes the silent-failure class described in CLAUDE.md line 167 structurally impossible.

## 2. Scope

**In:**
- Add `"unverifiable"` to the `VerificationStatus` Literal type alias.
- Change `VerificationResult.confidence` from `float` to `float | None`.
- Add a `__post_init__` validator on `VerificationResult` enforcing the confidence-evidence coupling invariant.
- Add `"unverifiable"` to `_VALID_STATUSES` in `src/verify_prompts.py`.
- Fix every callsite in `src/`, `tests/`, `eval/`, `examples/`, and `scripts/` that constructs a `VerificationResult` with a combination that now violates the invariant. The fix is to make the test fixture or default value comply, not to weaken the invariant.
- Provide a migration helper function for code that parses LLM responses into `VerificationResult`.

**Out:**
- Verifier emission logic changes (A2).
- Prompt changes (A3).
- Downstream consumer changes beyond fixing construction-time violations (A4).
- Fulltext fetcher improvements (D).
- `report.json` summary schema changes for `unverifiable` counts (A4).
- `ProvenanceStep` emission semantics for `unverifiable` (A2).

## 3. Files touched

| Path | Role |
|---|---|
| `src/models.py` | Add `"unverifiable"` to `VerificationStatus`; change `confidence: float \| None`; add `__post_init__` validator |
| `src/verify_prompts.py` | Add `"unverifiable"` to `_VALID_STATUSES` set |
| `src/report.py:106,243` | Fix default `VerificationResult` constructions (both use `status="not_addressed"` with `confidence=0.0` and `evidence_quality="abstract_only"` default -- these are valid under the new invariant, but verify) |
| `tests/unit/test_models.py` | New test class `TestVerificationResultInvariant` with 8 test methods |
| `tests/unit/test_report.py:35,216-237,299-317,464-472,532-563,608` | Fix test fixtures constructing `VerificationResult(status="supported", confidence=0.9)` with default `evidence_quality="abstract_only"` |
| `tests/unit/test_verify_abstract.py:376` | Fix `_result()` helper constructing `VerificationResult(status=..., confidence=...)` with default evidence_quality |
| `tests/unit/test_verify_fulltext.py:112,337,418` | Fix fixtures (some are fulltext-depth -- likely already valid; verify) |
| `tests/unit/test_verify_cross_modal.py:33` | Fix fixture |
| `tests/unit/test_enricher.py:41,344,462` | Fix fixtures |
| `tests/unit/test_fix_generator.py:44` | Fix fixture |
| `tests/unit/test_fix_generator_adversarial.py:56` | Fix fixture |
| `tests/unit/test_measure_e2e_recall.py:81` | Fix fixture |
| `tests/unit/test_pipeline.py:70` | Fix fixture |
| `tests/unit/test_primary_source.py:44` | Fix fixture |
| `tests/unit/test_rationale.py:48` | Fix fixture |
| `tests/unit/test_copilot_auto_eval.py:51` | Fix fixture |
| `tests/unit/test_copilot_run_example.py:50` | Fix fixture |
| `tests/unit/test_report_html.py:50` | Fix fixture |
| `examples/copilot_run.py:266` | Deserialization path; needs migration helper or invariant-aware construction |

## 4. Public API

### Changed type alias

```python
VerificationStatus = Literal[
    "supported", "unsupported", "not_addressed",
    "partially_supported", "unverifiable",
]
```

### Changed dataclass field

```python
@dataclass(frozen=True)
class VerificationResult:
    status: VerificationStatus
    explanation: str
    confidence: float | None  # was: float
    # ... remaining fields unchanged
```

### New `__post_init__` method

**Single invariant** (confidence ↔ status="unverifiable" coupling).
The evidence-quality coupling lives in the helper, not in the schema —
because it depends on `claim_text`, which `VerificationResult` does not
carry. See Decision log at the top of this spec.

```python
def __post_init__(self) -> None:
    # The only invariant the schema enforces: confidence is None
    # if and only if status == "unverifiable".
    if self.status == "unverifiable" and self.confidence is not None:
        raise ValueError(
            "unverifiable status requires confidence=None"
        )
    if self.status != "unverifiable" and self.confidence is None:
        raise ValueError(
            f"{self.status!r} status requires non-null confidence"
        )
```

The evidence-quality coupling (don't emit confident verdicts on
insufficient evidence for numeric claims) is enforced by
`safe_verification_result()` at LLM-response parse boundaries. The
helper is the canonical enforcement point. Direct `VerificationResult`
constructions in test fixtures and other call sites remain valid even
when they would have violated the dropped Invariant 2.

### Migration helper (new free function in `src/models.py`)

**Updated 2026-05-11:** takes `claim_text` to discriminate numeric vs
qualitative claims. Only numeric/Results-section claims on insufficient
evidence are downgraded.

```python
def safe_verification_result(
    *,
    status: str,
    confidence: float | None,
    evidence_quality: EvidenceQuality = "abstract_only",
    claim_text: str | None = None,
    **kwargs: Any,
) -> VerificationResult:
    """Construct a VerificationResult, downgrading to unverifiable when needed.

    Downgrade rule (all must hold):
      1. status in {"supported", "unsupported"}
      2. evidence_quality in {abstract_only, title_only,
                              citing_paper_context, no_evidence}
      3. claim_text is None (legacy callers — fail safe) OR
         _claim_has_specific_numeric(claim_text) is True

    Qualitative claims (no specific numeric content) on abstract-only
    evidence pass through unchanged — the abstract is sufficient for
    "X reduces Y"-style verdicts when it directly addresses the topic.

    Use at LLM-response parse boundaries where the raw parsed
    status/confidence may violate the invariant. Pure callers that know
    their inputs satisfy the invariant should construct VerificationResult
    directly.
    """
    from src.numeric.heuristics import _claim_has_specific_numeric

    _INSUFFICIENT = {
        "abstract_only", "title_only", "citing_paper_context", "no_evidence",
    }
    if (
        status in ("supported", "unsupported")
        and evidence_quality in _INSUFFICIENT
        and (claim_text is None or _claim_has_specific_numeric(claim_text))
    ):
        return VerificationResult(
            status="unverifiable",
            confidence=None,
            evidence_quality=evidence_quality,
            **kwargs,
        )
    if status == "unverifiable":
        confidence = None
    return VerificationResult(
        status=status,  # type: ignore[arg-type]
        confidence=confidence,
        evidence_quality=evidence_quality,
        **kwargs,
    )
```

### Numeric heuristic (new file `src/numeric/heuristics.py`)

Pure-Python deterministic regex check. No LLM, no I/O, no dependencies
beyond `re`. Lives in `src/numeric/` because it complements the
existing numeric-comparison engine and shares the same domain
vocabulary.

```python
"""Deterministic heuristics over claim text. Pure Python, no LLM."""
from __future__ import annotations

import re

# Compiled patterns that match specific numeric assertions typical of
# Results-section content. These are the claim shapes that cannot be
# reliably verified from an abstract alone.
_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\d+(?:\.\d+)?\s*%"),                          # 20%, 14.5%
    re.compile(r"\bp\s*[<>=]\s*0?\.\d+", re.IGNORECASE),       # p < 0.001, p=0.02
    re.compile(r"\bn\s*=\s*\d+", re.IGNORECASE),               # n=233
    re.compile(r"95\s*%\s*CI", re.IGNORECASE),                 # 95% CI
    re.compile(r"\b(?:HR|OR|RR)\s*[=:]?\s*\d", re.IGNORECASE), # HR 0.55, OR=1.7
    re.compile(r"hazard\s+ratio", re.IGNORECASE),
    re.compile(r"odds\s+ratio", re.IGNORECASE),
    re.compile(r"\b(?:Cohen'?s?\s*d|Hedges'?\s*g)\b", re.IGNORECASE),
    re.compile(r"\bweek\s*\d+", re.IGNORECASE),                # at week 12 (timepoint)
    re.compile(r"\b\d+\s*(?:mg|mcg|µg|ml|kg|points?)", re.IGNORECASE),
)


def _claim_has_specific_numeric(claim_text: str) -> bool:
    """True if the claim contains a specific numeric/Results-section assertion.

    These patterns mark claims that cannot be reliably verified from an
    abstract alone, because the abstract systematically omits exact
    figures (percentages, p-values, CIs, effect sizes, exact n=,
    timepoint-specific response rates).

    Pure deterministic. Same input -> same output, every run.
    """
    if not claim_text:
        return False
    return any(p.search(claim_text) for p in _PATTERNS)
```

**Helper behavior summary:**

| status                  | evidence_quality       | claim_text          | result                                         |
| ----------------------- | ---------------------- | ------------------- | ---------------------------------------------- |
| supported / unsupported | abstract_only / etc.   | has numeric pattern | DOWNGRADE → unverifiable, confidence=None      |
| supported / unsupported | abstract_only / etc.   | qualitative         | pass-through (abstract suffices to address it) |
| supported / unsupported | abstract_only / etc.   | None (legacy)       | DOWNGRADE (fail-safe)                          |
| supported / unsupported | quoted_passage / etc.  | any                 | pass-through                                   |
| partially_supported     | any                    | any                 | pass-through                                   |
| not_addressed           | any                    | any                 | pass-through                                   |
| unverifiable            | any                    | any                 | coerce confidence to None, then pass-through   |

The helper's Test 3–5 fixtures in Section 9 are updated to pass
`claim_text` with a known numeric pattern (e.g. `"20% sustained response"`).
A new positive test covers the qualitative pass-through:

```python
def test_helper_passes_qualitative_supported_on_abstract_through(self) -> None:
    """Qualitative claim with confident verdict on abstract: helper does NOT downgrade."""
    result = safe_verification_result(
        status="supported",
        confidence=0.9,
        evidence_quality="abstract_only",
        claim_text="psilocybin reduces depression symptoms",  # no numeric pattern
        explanation="abstract directly addresses the qualitative claim",
    )
    assert result.status == "supported"
    assert result.confidence == 0.9
    assert result.evidence_quality == "abstract_only"
```

**Note:** `safe_verification_result` is a convenience for A2 callers. Track A1 only defines it; Track A2 wires it into `verify.py`, `verify_fulltext.py`, etc. After A2, all LLM-response parse boundaries route through the helper, so the helper's discrimination rule is the *de facto* enforcement of the no-confident-verdict-on-insufficient-evidence invariant for the bug-class we're fixing.

### `_VALID_STATUSES` update (in `src/verify_prompts.py`)

```python
_VALID_STATUSES: set[str] = {
    "supported", "unsupported", "not_addressed",
    "partially_supported", "unverifiable",
}
```

## 5. Data flow

```
Path 1 — LLM response parse boundary (verify.py, verify_fulltext.py, ...):

Parsed (status, confidence) + caller-known (evidence_quality, claim_text)
  |
  v
safe_verification_result(...)
  |
  +-- numeric-claim AND insufficient evidence AND confident verdict?
  |     yes -> construct VerificationResult(unverifiable, None, ...)
  |     no  -> construct VerificationResult(status, confidence, ...)
  v
VerificationResult.__init__()
  |
  v
__post_init__() validates:
  1. status == "unverifiable" <=> confidence is None
  |
  +-- passes => frozen dataclass returned
  +-- fails  => ValueError raised


Path 2 — direct construction (test fixtures, internal helpers):

Caller passes (status, confidence, evidence_quality, ...) directly.
  |
  v
VerificationResult.__init__() -> __post_init__() validates Invariant 1 only.
```

The schema enforces only the confidence-None coupling. Discrimination
between "confident verdict on insufficient evidence is OK because the
claim is qualitative" vs "...not OK because the claim is numeric"
depends on `claim_text`, which is not a `VerificationResult` field.
That discrimination therefore lives in the helper, not the schema.

**Input shape:** the unchanged set of positional/keyword arguments to `VerificationResult`.

**Output shape:** a valid `VerificationResult` instance, or `ValueError`.

No data flows to external systems. This is a pure schema enforcement change.

## 6. External dependencies

None. This change is pure Python dataclass logic. No APIs, no libraries, no env vars.

## 7. Edge cases

The implementer must handle each of these explicitly:

1. **`partially_supported` on `abstract_only` -- allowed.**
   Rationale: `partially_supported` is itself a hedge. The verifier is saying "some elements of the claim appear in the abstract, but not all." This is a legitimate abstract-level observation. The invariant only blocks `supported` and `unsupported` -- the binary confident verdicts -- on insufficient evidence. Document this decision in a code comment inside `__post_init__`.

2. **`not_addressed` with `evidence_quality="abstract_only"` -- allowed.**
   Rationale: "we checked the abstract and the source doesn't address the claim at all" is a valid finding. The abstract is sufficient to determine that a topic is absent. The invariant targets false confident verdicts, not absence findings.

3. **`dataclasses.replace()` mutations that produce invariant violations.**
   Python's `dataclasses.replace()` calls `__init__` on the new instance, which triggers `__post_init__`. Any `dataclasses.replace(result, status="supported")` on a result with `evidence_quality="abstract_only"` will raise `ValueError`. This is correct behavior. The implementer must verify this with a dedicated test case (see test plan, item 8). Known `dataclasses.replace` sites that may break under A2/A4 but should NOT be fixed in A1:
   - `src/verify_fulltext.py:82-88` (abstract fallback on empty passages)
   - `src/verify.py:121` (attaching numeric_check -- safe, does not change status/evidence)

4. **`not_addressed` with `confidence=0.0` and default `evidence_quality="abstract_only"` -- allowed.**
   This is the default fallback in `src/report.py:106` and `src/report.py:243`. `not_addressed` is exempt from Invariant 2, and `confidence=0.0` (non-None float) satisfies Invariant 1. These callsites remain valid.

5. **`unverifiable` with any `evidence_quality` value -- allowed.**
   An `unverifiable` verdict can arise from any evidence depth. The claim might be unverifiable because fulltext was unavailable (abstract_only) or because even the fulltext passages were ambiguous (quoted_passage). The `__post_init__` only enforces that `confidence` is `None` when `status == "unverifiable"`. It does not constrain `evidence_quality` for `unverifiable`.

6. **Test fixtures constructing `VerificationResult(status="supported", confidence=0.9)` with implicit defaults.**
   The default `evidence_quality` is `"abstract_only"`. Every test fixture that constructs a `supported` or `unsupported` result without explicitly setting `evidence_quality` to a fulltext-grade value will crash. There are approximately 20 such callsites across `tests/unit/`. Each must be updated to include `evidence_quality="quoted_passage"` (or another fulltext-grade value). See the blast-radius grep in Section 10.

7. **`examples/copilot_run.py:266` -- dynamic deserialization from dict.**
   This path constructs `VerificationResult(**filtered_dict)` from stored data. If the stored data contains a violating combination, construction will raise. This is intentional for new data, but existing stored copilot runs may contain violating combinations. The implementer should NOT try to fix this in A1 -- it falls under backward-compatibility (Section 10, Risk 3). Add a code comment at the site documenting that old data may fail deserialization.

8. **`_SHORT_CIRCUIT_RESULT` and `_PARSE_ERROR_RESULT` in `src/verify_prompts.py`.**
   Both use `status="not_addressed"` with `evidence_quality="no_evidence"`. Under the new invariant, `not_addressed` is exempt from Invariant 2, so these remain valid. `confidence=1.0` on the short-circuit and `confidence=0.0` on the parse-error are both non-None floats, satisfying Invariant 1. No change needed.

## 8. Behavior when input is insufficient

**(a) What counts as insufficient input for this module:**

A `VerificationResult` is considered to have been constructed on insufficient input when:
- `status in {"supported", "unsupported"}` (a confident binary verdict), AND
- `evidence_quality in {"abstract_only", "title_only", "citing_paper_context", "no_evidence"}` (the verifier did not have fulltext-grade evidence).

This is the structural definition of the silent-failure class that triggered the plan. The abstract systematically omits Results-section data (exact percentages, p-values, CIs, sample sizes). A confident verdict based on abstract-only evidence for a claim about such data is epistemically unsound.

**(b) Which schema fields signal the insufficiency:**

- `evidence_quality: EvidenceQuality` -- the primary signal. Values `"quoted_passage"` and `"passages_searched_no_quote"` indicate fulltext-grade evidence. All other values indicate insufficient evidence for a confident verdict.
- `verification_depth: Literal["fulltext", "abstract", "title_only", "citing_paper_context"]` -- a corroborating signal (not used in the invariant directly, because `evidence_quality` is more precise).
- `fulltext_available: bool` -- contextual (a source can have fulltext available but still produce abstract-level evidence if BM25 found no relevant passages).

**(c) What the module returns/raises in that case:**

`ValueError` at construction time. The `__post_init__` validator raises immediately. There is no graceful fallback inside the schema layer -- the caller is responsible for not constructing violating combinations. The `safe_verification_result()` helper (Section 4) provides the migration path: callers at LLM-response boundaries should use it to auto-downgrade to `"unverifiable"`.

**(d) Which ProvenanceStep fields capture the uncertainty:**

Track A1 does not modify `ProvenanceStep`. However, the schema already supports the required fields:
- `ProvenanceStep.confidence: float | None` -- already accepts `None` (see `src/models.py:171`). When A2 wires the verifier to emit `unverifiable`, the ProvenanceStep will carry `confidence=None`.
- `ProvenanceStep.output_hash: str` -- the hash of the `VerificationResult`, which now includes `status="unverifiable"`. Sufficient for audit trail diffing.

A2/A4 will need to add an `unverifiable_reason` field (one of `"insufficient_evidence_depth"`, `"fulltext_unavailable"`, `"numeric_claim_abstract_only"`) to either the ProvenanceStep or the report.json per-claim record. A1 leaves this door open by not constraining the ProvenanceStep schema.

## 9. Test plan

All tests in `tests/unit/test_models.py`. No mocking required -- these are pure dataclass construction tests.

### New test class: `TestVerificationResultInvariant`

**Test 1: `test_post_init_rejects_unverifiable_with_confidence`**

```python
def test_post_init_rejects_unverifiable_with_confidence(self) -> None:
    with pytest.raises(ValueError, match="unverifiable status requires confidence=None"):
        VerificationResult(
            status="unverifiable",
            explanation="cannot determine",
            confidence=0.75,
            evidence_quality="abstract_only",
        )
```

**Test 2: `test_post_init_rejects_non_unverifiable_with_none_confidence`**

```python
def test_post_init_rejects_non_unverifiable_with_none_confidence(self) -> None:
    with pytest.raises(ValueError, match="requires non-null confidence"):
        VerificationResult(
            status="supported",
            explanation="ok",
            confidence=None,
            evidence_quality="quoted_passage",
        )
```

**Test 3: `test_post_init_rejects_confident_supported_on_abstract_only`**

```python
def test_post_init_rejects_confident_supported_on_abstract_only(self) -> None:
    with pytest.raises(ValueError, match="no-confident-verdict-without-evidence"):
        VerificationResult(
            status="supported",
            explanation="ok",
            confidence=0.9,
            evidence_quality="abstract_only",
        )
```

**Test 4: `test_post_init_rejects_confident_supported_on_title_only`**

```python
def test_post_init_rejects_confident_supported_on_title_only(self) -> None:
    with pytest.raises(ValueError, match="no-confident-verdict-without-evidence"):
        VerificationResult(
            status="supported",
            explanation="ok",
            confidence=0.9,
            evidence_quality="title_only",
        )
```

**Test 5: `test_post_init_rejects_confident_unsupported_on_citing_paper_context`**

```python
def test_post_init_rejects_confident_unsupported_on_citing_paper_context(self) -> None:
    with pytest.raises(ValueError, match="no-confident-verdict-without-evidence"):
        VerificationResult(
            status="unsupported",
            explanation="contradicted",
            confidence=0.8,
            evidence_quality="citing_paper_context",
        )
```

**Test 6: `test_post_init_accepts_supported_with_fulltext_quoted_passage`**

```python
def test_post_init_accepts_supported_with_fulltext_quoted_passage(self) -> None:
    result = VerificationResult(
        status="supported",
        explanation="ok",
        confidence=0.9,
        evidence_quality="quoted_passage",
        verification_depth="fulltext",
        fulltext_available=True,
        retrieval_status="passage_found",
    )
    assert result.status == "supported"
    assert result.confidence == 0.9
```

**Test 7: `test_post_init_accepts_unverifiable_with_none_confidence_and_any_evidence_quality`**

```python
@pytest.mark.parametrize("eq", [
    "abstract_only", "title_only", "citing_paper_context",
    "no_evidence", "quoted_passage", "passages_searched_no_quote",
])
def test_post_init_accepts_unverifiable_with_none_confidence_and_any_evidence_quality(
    self, eq: str,
) -> None:
    result = VerificationResult(
        status="unverifiable",
        explanation="cannot determine",
        confidence=None,
        evidence_quality=eq,  # type: ignore[arg-type]
    )
    assert result.status == "unverifiable"
    assert result.confidence is None
```

**Test 8: `test_existing_unverifiable_construction_via_dataclasses_replace_preserves_invariant`**

```python
def test_existing_unverifiable_construction_via_dataclasses_replace_preserves_invariant(
    self,
) -> None:
    """dataclasses.replace() calls __init__ -> __post_init__. Verify it
    catches invariant violations on the replacement, not just direct construction."""
    valid = VerificationResult(
        status="not_addressed",
        explanation="ok",
        confidence=0.5,
        evidence_quality="abstract_only",
    )
    # Replacing status to "supported" while evidence_quality stays "abstract_only" must fail
    with pytest.raises(ValueError, match="no-confident-verdict-without-evidence"):
        dataclasses.replace(valid, status="supported", confidence=0.9)
```

### Additional tests to add (non-invariant, but needed for completeness)

**Test 9: `test_partially_supported_allowed_on_abstract_only`**

```python
def test_partially_supported_allowed_on_abstract_only(self) -> None:
    """partially_supported is a hedge, not a confident verdict. Allowed on any evidence."""
    result = VerificationResult(
        status="partially_supported",
        explanation="some elements match",
        confidence=0.6,
        evidence_quality="abstract_only",
    )
    assert result.status == "partially_supported"
```

**Test 10: `test_not_addressed_allowed_on_abstract_only`**

```python
def test_not_addressed_allowed_on_abstract_only(self) -> None:
    """not_addressed = the source doesn't discuss the claim. Valid from abstract."""
    result = VerificationResult(
        status="not_addressed",
        explanation="source does not discuss this",
        confidence=0.9,
        evidence_quality="abstract_only",
    )
    assert result.status == "not_addressed"
```

### Callsite fix verification

After fixing all callsites per Section 3 "Files touched", the implementer must run:

```bash
pytest tests/unit/ -v
```

All existing tests must pass. The only test failures allowed are in tests that were intentionally testing the old schema behavior (there are none -- the old schema had no `__post_init__`).

### Integration tests

None required for A1. The schema change is pure Python. Integration testing of the end-to-end behavior with `unverifiable` verdicts belongs to A2/A4.

## 10. ProvenanceStep

**No ProvenanceStep schema change in A1.**

`ProvenanceStep.confidence: float | None` at `src/models.py:171` already accepts `None`. This was a deliberate design choice from Phase 0, and it means A1 does not need to touch the provenance schema.

**Dependency on A2:** The verifier emission paths (A2) will need to:
- Emit `ProvenanceStep(confidence=None)` when the verdict is `unverifiable`.
- Decide whether `unverifiable_reason` belongs in `ProvenanceStep.output_hash` semantics (by hashing the reason into the output) or as a new field on ProvenanceStep. A1's spec takes no position on this -- it is A2's decision.

**What A1 guarantees for A2:** After A1 lands, any `VerificationResult` with `status="unverifiable"` will have `confidence=None`. A2 can rely on this invariant when propagating confidence into ProvenanceStep -- it does not need to check for the impossible case of `status="unverifiable"` with a non-None confidence.

## 11. Risks

### Risk 1: Blast radius of the invariant

Every callsite that constructs `VerificationResult` with `status in {"supported", "unsupported"}` and default `evidence_quality` (which is `"abstract_only"`) will raise `ValueError` after A1 lands. This is by design -- but the blast radius is large.

**Discovery grep:**

```bash
# Find all VerificationResult constructions
grep -rn "VerificationResult(" src/ tests/ eval/ examples/ scripts/

# Filter for likely violators: supported/unsupported without explicit evidence_quality
# (default evidence_quality is "abstract_only")
grep -rn "VerificationResult(" src/ tests/ eval/ examples/ scripts/ \
  | grep -E 'status="(supported|unsupported)"' \
  | grep -v "evidence_quality"
```

**Known violating callsites (from reading the codebase):**

| File | Line | Status | Evidence quality | Fix |
|---|---|---|---|---|
| `tests/unit/test_report.py` | 35 | `supported` (param) | default `abstract_only` | Add `evidence_quality="quoted_passage"` |
| `tests/unit/test_report.py` | 216-237 | `supported`, `unsupported` | default `abstract_only` on c2 | c0/c1 already set `verification_depth="fulltext"` but lack `evidence_quality`; add it |
| `tests/unit/test_report.py` | 299-317 | `supported`, `unsupported` | default on some | Add explicit `evidence_quality` |
| `tests/unit/test_report.py` | 464-472 | `supported` | has `verification_depth="fulltext"` but no `evidence_quality` | Add `evidence_quality="quoted_passage"` |
| `tests/unit/test_report.py` | 556 | `supported` | `abstract` depth, no evidence_quality | Must change to `partially_supported` or set fulltext evidence |
| `tests/unit/test_verify_abstract.py` | 376 | parameterized | default `abstract_only` | Add `evidence_quality="quoted_passage"` in helper |
| `tests/unit/test_verify_fulltext.py` | 112 | `supported` | default | Add `evidence_quality="quoted_passage"` |
| `tests/unit/test_verify_cross_modal.py` | 33 | `supported` | default | Add `evidence_quality="quoted_passage"` |
| `tests/unit/test_enricher.py` | 41, 344, 462 | various | default | Add explicit `evidence_quality` |
| `tests/unit/test_models.py` | 148 | `supported` | default | Add `evidence_quality="quoted_passage"` |
| `tests/unit/test_pipeline.py` | 70 | likely `supported` | default | Add explicit `evidence_quality` |
| `tests/unit/test_copilot_*.py` | various | likely `supported` | default | Add explicit `evidence_quality` |
| `tests/unit/test_fix_generator*.py` | various | various | default | Add explicit `evidence_quality` |
| `tests/unit/test_primary_source.py` | 44 | various | default | Add explicit `evidence_quality` |
| `tests/unit/test_rationale.py` | 48 | various | default | Add explicit `evidence_quality` |
| `tests/unit/test_measure_e2e_recall.py` | 81 | various | default | Add explicit `evidence_quality` |
| `tests/unit/test_report_html.py` | 50 | various | default | Add explicit `evidence_quality` |
| `src/verify.py` | 190 | `supported`/`unsupported` (parsed) | default `abstract_only` | **This is the triggering bug.** A2 fixes this. A1 does NOT fix it -- the `__post_init__` will crash at runtime, which is the intended forcing function for A2. |
| `src/verify_multi.py` | 127 | aggregated status | explicit `abstract_only` | A2 fixes this. |
| `src/verify_title_only.py` | 106 | caps to `partially_supported` | `title_only` | `partially_supported` is exempt from Invariant 2; valid. |
| `src/verify_citing_context.py` | 124 | caps to `partially_supported` | `citing_paper_context` | `partially_supported` is exempt; valid. But `unsupported` can pass through (line 115 only caps `supported`). A2 must fix this. |

**Instruction to @implementer:** Fix every test fixture callsite. Do NOT fix `src/verify.py:190`, `src/verify_multi.py:127`, or `src/verify_citing_context.py:124` -- those are A2 scope. The invariant will cause those runtime paths to crash, which is the intended behavior until A2 lands. If tests that exercise those code paths through mocked LLM responses now fail, that is correct and expected -- update those tests to mock the LLM returning a non-violating combination (e.g., `"not_addressed"`) or to expect the `ValueError`.

### Risk 2: `dataclasses.replace` paths in verify_fulltext.py and verify_multi.py

Two known `dataclasses.replace` sites will produce invariant violations after A1:

- **`src/verify_fulltext.py:82-88`**: When `passages` is empty, this code calls `verify_claim` (abstract-only), gets a result with default `evidence_quality="abstract_only"`, then replaces fields. If the abstract verifier returned `status="supported"`, the replace preserves that status with `evidence_quality` still at `abstract_only` -- invariant violation. **This is A2 scope.** A1 should not fix it. The existing test `test_empty_passages_marks_no_passage_found` will need updating in A2.

- **`src/verify.py:121`**: Attaches `numeric_check` via replace. This only changes `numeric_check`, not `status` or `evidence_quality`. Safe under A1 -- no invariant violation.

**Instruction to @implementer:** Do not attempt to fix `verify_fulltext.py:82-88` in A1. Document in a code comment at `__post_init__` that A2 must address the verify_fulltext empty-passages path.

### Risk 3: Backward compatibility of `report.json`

Existing `reports/runs/*/report.json` files contain verdicts that violate the new invariant (e.g., `status: "unsupported", confidence: 0.75, evidence_quality: "abstract_only"`). These files are historical artifacts and must remain readable.

**Impact points:**
- `examples/copilot_run.py:266` -- constructs `VerificationResult` from stored JSON via `VerificationResult(**filtered_dict)`. Old data will crash.
- Any script or notebook that reads `report.json` and deserializes into `VerificationResult` will crash on old reports.

**Instruction to @implementer:** Do NOT fix old report.json files. Do NOT add backward-compat shims to `VerificationResult.__post_init__`. The invariant must be strict. Instead:
1. Add a comment at `examples/copilot_run.py:266` noting that old copilot data predating the `unverifiable` schema will fail deserialization.
2. Flag to A4 that `report.json` deserialization needs a migration path (read-only compat layer or versioned schema).

### Risk 4: `test_report.py:556` -- the `c4` supported-on-abstract fixture

This test fixture (`c4: VerificationResult(status="supported", confidence=0.9, retrieval_status="fulltext_unavailable", verification_depth="abstract")`) was intentionally testing the "abstract_only_verdicts" diagnostic counter. After A1, this construction is illegal. The implementer must decide:
- Change `status` to `"partially_supported"` (still counts for abstract_only_verdicts?), or
- Change `evidence_quality` to `"quoted_passage"` (but then it's not testing abstract-only anymore), or
- Change `status` to `"unverifiable"` with `confidence=None` and update the test assertion.

Recommended: change to `"partially_supported"` (allowed on abstract_only) and update the test to verify that `abstract_only_verdicts` counts `partially_supported` verdicts at abstract depth. This preserves the diagnostic intent while complying with the invariant.

## 12. Out of scope

| Feature | Why refused |
|---|---|
| `unverifiable_reason` field on VerificationResult | Belongs to A4. The reason ("insufficient_evidence_depth", "fulltext_unavailable", "numeric_claim_abstract_only") is verifier-emission context, not schema-level. A1 only enforces the structural invariant. |
| Verifier emission changes in `src/verify.py` | A2 scope. A1 provides the schema and the `safe_verification_result` helper; A2 wires them. |
| Prompt changes to teach the LLM about `unverifiable` | A3 scope. |
| `report.json` summary changes for `unverifiable` counts | A4 scope. |
| Report deserialization migration layer | A4 scope. A1 documents the risk (Section 10, Risk 3). |
| `__post_init__` confidence-range validation (e.g., 0.0 <= confidence <= 1.0) | Tempting but out of scope. The invariant is about confidence-evidence coupling, not confidence magnitude. Could be added later as a separate spec if needed. |
| Making `evidence_quality` derived from `verification_depth` | The two fields have different semantics (`verification_depth` = what depth was attempted; `evidence_quality` = what the verifier actually used). Collapsing them would lose information. Refuse. |
