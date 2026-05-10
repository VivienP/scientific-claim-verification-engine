# Spec: numeric-precision-fix

## 1. Goal

Fix two deterministic bugs in `src/numeric/engine.py` that cause 100% false positive rate on numeric inconsistency flags (4/4 across 3 Elicit benchmark runs), and add permanent regression infrastructure for all 4 captured cases.

## 2. Scope

**In scope:**

- Bug A fix: context-aware CI pairing in `_find_or_ci_triple` (engine.py only, no schema/prompt change)
- Bug B fix: remove `unit=None` fallback in `_find_or_ci_triple` -- require explicit ratio keyword
- Regression JSONL for 4 false positives from Elicit runs
- Unit tests for both bugs (TDD: tests written first, then fix)
- Confirmation that 19 currently-correct `consistent: true` checks still pass

**Out of scope:**

- Changes to `NumericAssertion` schema (no `primary_ref` field -- that is future Option A2)
- Changes to `src/numeric/extract.py` or the extraction prompt
- Changes to `src/numeric/checks.py` (the check functions are correct; the bug is in tuple selection)
- New check types (e.g., additive mean-diff/CI check)
- Changes to `_find_p_value_ci_tuple` (not affected by either bug)

## 3. Files touched

| Path | Role | Change type |
|---|---|---|
| `src/numeric/engine.py` | Triple-finder logic | **Modified** -- Bug A + Bug B fixes in `_find_or_ci_triple` |
| `tests/unit/test_numeric_engine.py` | Unit tests for engine | **Modified** -- add test cases for Bug A and Bug B |
| `tests/unit/test_numeric_regressions.py` | Regression tests wired to engine | **New** -- loads regression JSONL, runs `_find_or_ci_triple` on captured assertion lists |
| `eval/regressions/2026-05-10/elicit_numeric_workflow/regression.jsonl` | 4 regression entries | **New** |

## 4. Public API

No public API changes. `run_numeric_check` signature is unchanged. `_find_or_ci_triple` is a private function; its signature is unchanged but its behavior narrows.

### Changed internal function

```python
# src/numeric/engine.py -- unchanged signature, changed behavior
def _find_or_ci_triple(
    assertions: list[NumericAssertion],
) -> tuple[float, float, float] | None:
    """Find the first ratio-measure/CI triple in the assertion list.

    Returns (ratio_value, ci_low, ci_high) or None.

    Bug A fix: after selecting the primary, score each ci_low/ci_high by
    context similarity to the primary. Pick the CI pair whose context
    references the same statistic.

    Bug B fix: require the primary's context or raw_text to contain an
    explicit ratio keyword (OR, HR, RR, RRR, AHR, aHR, odds ratio,
    hazard ratio, risk ratio, relative risk). No unit=None fallback.
    """
```

### New helper (private, in engine.py)

```python
def _ci_matches_primary(
    primary: NumericAssertion,
    ci: NumericAssertion,
) -> bool:
    """Return True if the CI assertion's context references the primary's statistic.

    Heuristic: check if the primary's raw_text (e.g. "OR 1.7", "HR 0.59")
    appears as a substring in the CI's context. Falls back to checking
    whether any distinctive token from the primary's context (the measure
    type keyword + value) appears in the CI's context.
    """
```

## 5. Data flow

### Current flow (buggy)

```
assertions[] --> _find_or_ci_triple:
  1. Scan for role=primary with "odds ratio"/"OR" in context/raw_text
  2. FALLBACK: pick first role=primary with unit=None          <-- Bug B
  3. Scan forward from primary_idx for FIRST ci_low, ci_high   <-- Bug A
  4. Return (primary.value, ci_low.value, ci_high.value)

--> check_or_ci_consistency(primary, ci_low, ci_high)
--> NumericCheckResult
```

### Fixed flow

```
assertions[] --> _find_or_ci_triple:
  1. Scan for role=primary with RATIO KEYWORD in context or raw_text
     Keywords: "odds ratio", "hazard ratio", "risk ratio", "relative risk",
               "OR ", "HR ", "RR ", "RRR ", "AHR ", "aHR ",
               plus case-insensitive startswith("or", "hr", "rr", ...)
  2. NO FALLBACK. If no ratio keyword found, return None.       <-- Bug B fix
  3. For the selected primary, collect ALL ci_low and ci_high assertions.
  4. Score each CI by _ci_matches_primary(primary, ci).
     - If exactly one ci_low and one ci_high match: use them.
     - If multiple match or none match: fall back to positional
       (first ci_low/ci_high after primary_idx, same as before).  <-- Bug A fix
  5. Return (primary.value, ci_low.value, ci_high.value) or None.

--> check_or_ci_consistency(primary, ci_low, ci_high)
--> NumericCheckResult
```

### Data shapes

Input to `_find_or_ci_triple`:
```python
assertions: list[NumericAssertion]
# where NumericAssertion = (raw_text: str, value: float, unit: str|None,
#                           role: NumericRole, context: str)
```

Output:
```python
tuple[float, float, float] | None  # (ratio_value, ci_low, ci_high)
```

### Bug A example (claim_id 23a27499)

Assertions in order:
1. `primary, raw="OR 1.7", ctx="odds ratio for ORR..."`
2. `primary, raw="HR 0.59", ctx="hazard ratio for PFS..."`
3. `ci_low, raw="0.48", ctx="95% CI lower bound for PFS HR 0.59"`
4. `ci_high, raw="0.74", ctx="95% CI upper bound for PFS HR 0.59"`
5. `primary, raw="HR 0.82", ctx="hazard ratio for OS..."`
6. `ci_low, raw="0.6", ctx="95% CI lower bound for OS HR 0.82"`
7. `ci_high, raw="1.1", ctx="95% CI upper bound for OS HR 0.82"`

Current: picks primary=OR 1.7 (step 1), then ci_low=0.48, ci_high=0.74 (step 3, positional). These CIs belong to HR 0.59. Result: false positive.

Fixed: picks primary=OR 1.7 (step 1). Scores CIs:
- ci_low 0.48 context contains "PFS HR 0.59" -- does NOT contain "OR 1.7" --> no match
- ci_low 0.6 context contains "OS HR 0.82" -- does NOT contain "OR 1.7" --> no match
- Neither ci_low matches primary. Context-match finds zero. Fall back to positional scan from primary_idx=0. ci_low=0.48, ci_high=0.74. Still wrong?

**Wait.** The positional fallback would still pick the wrong CI. The correct behavior for this claim is: OR 1.7 has NO CI in the claim text. The claim says `ORR (OR 1.7)` with no CI, then `PFS (HR 0.59, 95% CI 0.48-0.74)`. So OR 1.7 genuinely has no CI. The fix should return None when no CI matches the selected primary.

**Revised step 4:** If no CI pair matches the primary by context, and the claim contains multiple primaries, return None (no check applies). Only fall back to positional when there is exactly one primary in the assertion list (unambiguous case).

### Bug B example (claim_id 944726cb)

Assertions:
1. `primary, raw="14.9", unit=None, ctx="mean MADRS reduction..."`
2. `ci_low, raw="-20.7", unit=None, ctx="95% CI lower bound for mean MADRS reduction"`
3. `ci_high, raw="-9.2", unit=None, ctx="95% CI upper bound for mean MADRS reduction"`

Current: step 1 finds no OR keyword. Step 2 fallback picks primary=14.9 (unit=None). Passes to check_or_ci_consistency which fails on `ci_low <= 0` (multiplicative scale check). False positive.

Fixed: step 1 finds no ratio keyword. Step 2 is removed. Returns None. No numeric_check attached. Correct.

## 6. External dependencies

None. Both bugs are in pure Python deterministic code. No new libraries, no API calls, no env vars.

## 7. Edge cases

The implementer must handle all of the following explicitly:

### E1: OR/HR/RR primary with no CI in claim

Example: `"ORR (OR 1.7)"` with no CI. `_find_or_ci_triple` selects OR 1.7 as primary. No CI assertions exist (or none match). Must return None.

### E2: Two primaries, each with its own CI, context-matched

Example: `"OR 1.7 (95% CI 1.1-2.5) and HR 0.59 (95% CI 0.48-0.74)"`. The first ratio keyword match (OR 1.7) is selected. Its CI pair (1.1, 2.5) matches by context ("OR 1.7" appears in CI context). Return (1.7, 1.1, 2.5). The HR is ignored (a single call checks one triple).

### E3: Single primary with unit=None but no ratio keyword (mean diff, Hedges' g, Cohen's d)

Must return None. This is Bug B. No fallback to unit=None.

### E4: Ratio keyword in raw_text but not in context (e.g. raw="OR 2.3", context="adjusted measure between groups")

Must still match. The keyword scan checks both `raw_text` and `context`.

### E5: Ambiguous CI context (context does not mention any primary's raw_text)

When multiple primaries exist and no CI context-matches any of them, return None. Do not guess. This is the "better no check than wrong check" principle from CLAUDE.md forbidden patterns.

### E6: Single primary with ratio keyword + CI pair, no ambiguity (the happy path)

Example: `"OR 40.53 (95% CI 23.58-73.71)"`. One primary, one ci_low, one ci_high. Context matching is irrelevant (unambiguous). Positional fallback applies cleanly. Return (40.53, 23.58, 73.71). This is the GLP-1 MACE case -- must not break.

### E7: Case sensitivity in ratio keywords

`"or 1.7"`, `"Hr 0.59"`, `"aHR 0.82"` must all match. Keyword matching is case-insensitive.

### E8: Ratio keyword as substring of unrelated word

`"OR" in "PRIOR" or "MONITOR". Mitigated by requiring word boundary or prefix: `raw_lower.startswith("or")` or `" or " in ctx_lower` or `"odds ratio" in ctx_lower`. The existing code already handles this partially; preserve that logic.

## 8. Test plan

### TDD order

Write tests first. Each test must fail before the fix, pass after.

#### Phase 1: Bug A regression tests (in test_numeric_engine.py)

```python
class TestFindOrCiTripleBugA:
    def test_multi_ratio_claim_or_without_ci_returns_none(self) -> None:
        """Bug A: OR 1.7 has no CI; CI [0.48, 0.74] belongs to HR 0.59.
        Must return None, not (1.7, 0.48, 0.74).
        Reproduces claim_id 23a27499."""
        assertions = [
            NumericAssertion(raw_text="OR 1.7", value=1.7, unit=None, role="primary",
                             context="odds ratio for ORR with chemo-ICI vs ICI alone"),
            NumericAssertion(raw_text="HR 0.59", value=0.59, unit=None, role="primary",
                             context="hazard ratio for PFS with chemo-ICI vs ICI alone"),
            NumericAssertion(raw_text="0.48", value=0.48, unit=None, role="ci_low",
                             context="95% CI lower bound for PFS HR 0.59"),
            NumericAssertion(raw_text="0.74", value=0.74, unit=None, role="ci_high",
                             context="95% CI upper bound for PFS HR 0.59"),
        ]
        result = _find_or_ci_triple(assertions)
        assert result is None

    def test_multi_ratio_claim_rr_without_ci_returns_none(self) -> None:
        """Bug A variant: RR 1.62 has no CI; CI [0.32, 0.97] belongs to HR 0.55.
        Reproduces claim_id 37ddafbb."""
        assertions = [
            NumericAssertion(raw_text="RR 1.62", value=1.62, unit=None, role="primary",
                             context="relative risk for objective response rate (ORR) with combination therapy"),
            NumericAssertion(raw_text="HR 0.55", value=0.55, unit=None, role="primary",
                             context="hazard ratio for progression-free survival (PFS) with combination therapy"),
            NumericAssertion(raw_text="0.32", value=0.32, unit=None, role="ci_low",
                             context="95% CI lower bound for HR 0.55 (PFS)"),
            NumericAssertion(raw_text="0.97", value=0.97, unit=None, role="ci_high",
                             context="95% CI upper bound for HR 0.55 (PFS)"),
        ]
        result = _find_or_ci_triple(assertions)
        assert result is None

    def test_or_with_own_ci_in_multi_ratio_claim(self) -> None:
        """OR has its own CI. Even with multiple primaries, context-match
        correctly pairs OR 1.7 with CI [1.1, 2.5]."""
        assertions = [
            NumericAssertion(raw_text="OR 1.7", value=1.7, unit=None, role="primary",
                             context="odds ratio for ORR"),
            NumericAssertion(raw_text="1.1", value=1.1, unit=None, role="ci_low",
                             context="95% CI lower bound for OR 1.7"),
            NumericAssertion(raw_text="2.5", value=2.5, unit=None, role="ci_high",
                             context="95% CI upper bound for OR 1.7"),
            NumericAssertion(raw_text="HR 0.59", value=0.59, unit=None, role="primary",
                             context="hazard ratio for PFS"),
            NumericAssertion(raw_text="0.48", value=0.48, unit=None, role="ci_low",
                             context="95% CI lower bound for HR 0.59"),
            NumericAssertion(raw_text="0.74", value=0.74, unit=None, role="ci_high",
                             context="95% CI upper bound for HR 0.59"),
        ]
        result = _find_or_ci_triple(assertions)
        assert result == (1.7, 1.1, 2.5)
```

#### Phase 2: Bug B regression tests (in test_numeric_engine.py)

```python
class TestFindOrCiTripleBugB:
    def test_mean_diff_with_ci_returns_none(self) -> None:
        """Bug B: mean MADRS reduction (unit=None, no ratio keyword) must not
        be treated as OR. Reproduces claim_id 944726cb."""
        assertions = [
            NumericAssertion(raw_text="14.9", value=14.9, unit=None, role="primary",
                             context="mean MADRS reduction at week 3 for psilocybin adjunct to SSRIs"),
            NumericAssertion(raw_text="-20.7", value=-20.7, unit=None, role="ci_low",
                             context="95% CI lower bound for mean MADRS reduction at week 3"),
            NumericAssertion(raw_text="-9.2", value=-9.2, unit=None, role="ci_high",
                             context="95% CI upper bound for mean MADRS reduction at week 3"),
        ]
        result = _find_or_ci_triple(assertions)
        assert result is None

    def test_hedges_g_with_ci_returns_none(self) -> None:
        """Bug B: Hedges' g is an effect size, not a ratio measure.
        Reproduces claim_id c110e5f5."""
        assertions = [
            NumericAssertion(raw_text="-7.14", value=-7.14, unit=None, role="primary",
                             context="mean QIDS change at 3 weeks"),
            NumericAssertion(raw_text="-1.27", value=-1.27, unit=None, role="primary",
                             context="Hedges' g effect size for QIDS change at 3 weeks"),
            NumericAssertion(raw_text="-2.40", value=-2.4, unit=None, role="ci_low",
                             context="95% CI lower bound for Hedges' g"),
            NumericAssertion(raw_text="-0.37", value=-0.37, unit=None, role="ci_high",
                             context="95% CI upper bound for Hedges' g"),
        ]
        result = _find_or_ci_triple(assertions)
        assert result is None
```

#### Phase 3: Non-regression tests (confirm happy path still works)

```python
class TestFindOrCiTripleHappyPath:
    def test_single_or_with_ci(self) -> None:
        """GLP-1 MACE happy path: single OR with CI, no ambiguity."""
        assertions = [
            NumericAssertion(raw_text="OR 40.53", value=40.53, unit=None, role="primary",
                             context="odds ratio ARM A+T- vs A-T-"),
            NumericAssertion(raw_text="23.58", value=23.58, unit=None, role="ci_low",
                             context="95% CI lower bound for OR 40.53"),
            NumericAssertion(raw_text="73.71", value=73.71, unit=None, role="ci_high",
                             context="95% CI upper bound for OR 40.53"),
        ]
        result = _find_or_ci_triple(assertions)
        assert result == (40.53, 23.58, 73.71)

    def test_single_hr_with_ci(self) -> None:
        """Single HR with CI, unambiguous."""
        assertions = [
            NumericAssertion(raw_text="HR 0.59", value=0.59, unit=None, role="primary",
                             context="hazard ratio for PFS"),
            NumericAssertion(raw_text="0.48", value=0.48, unit=None, role="ci_low",
                             context="95% CI lower bound"),
            NumericAssertion(raw_text="0.74", value=0.74, unit=None, role="ci_high",
                             context="95% CI upper bound"),
        ]
        result = _find_or_ci_triple(assertions)
        assert result == (0.59, 0.48, 0.74)

    def test_no_assertions_returns_none(self) -> None:
        assert _find_or_ci_triple([]) is None

    def test_only_p_value_returns_none(self) -> None:
        assertions = [
            NumericAssertion(raw_text="p=0.02", value=0.02, unit=None, role="p_value",
                             context="p-value for treatment effect"),
        ]
        assert _find_or_ci_triple(assertions) is None
```

#### Phase 4: Wired regression tests (in test_numeric_regressions.py)

Loads `eval/regressions/2026-05-10/elicit_numeric_workflow/regression.jsonl`. For each entry:
- Deserializes `extracted_assertions` into `list[NumericAssertion]`
- Calls `_find_or_ci_triple(assertions)`
- Asserts the expected behavior:
  - `"no_check_applies"` -> result is None
  - `"consistent"` -> result is not None, and `check_or_ci_consistency(*result)` returns `consistent=True`

### Integration tests

None required. Both bugs are in pure deterministic code. No API calls involved.

## 9. ProvenanceStep

### No changes to provenance emission points

The existing provenance emission in `run_numeric_check` (engine.py lines 150-162) is correct and unchanged:

- `extract_numeric_assertions` emits one ProvenanceStep with `operation="numeric_extract"` (LLM call).
- If a check runs, `run_numeric_check` emits one ProvenanceStep with `operation="numeric_check"`, `model_id=None` (deterministic).
- If no check applies (the new behavior for Bug A/B cases), only the extract step is emitted. This is the existing behavior for `triple is None and p_ci_tuple is None`.

The fix **increases** the set of claims that return `(None, [extract_step])` -- claims that previously got a false `numeric_check` step now get no check step. This is correct: no check is better than a wrong check.

No new ProvenanceStep emission points are needed.

## 10. Risks

### R1: Regression on the 19 currently-correct checks (HIGH)

The 13 GLP-1 and 6 NSCLC `consistent: true` checks must not break. Mitigation: the happy-path tests in Phase 3 cover the single-primary-with-CI pattern (the GLP-1 case). Additionally, the implementer must run `_find_or_ci_triple` on the full assertion list from each of the 19 correct checks and verify the same triple is returned. This can be a parametrized non-regression test using data extracted from `benchmarks/real_outputs/`.

### R2: Context-matching heuristic is fragile (MEDIUM)

The `_ci_matches_primary` heuristic relies on the LLM's `context` field containing the primary's `raw_text` as a substring (e.g., context "95% CI lower bound for OR 1.7" contains "OR 1.7"). If a future LLM response uses different phrasing (e.g., "95% CI lower bound for odds ratio 1.7" instead of "OR 1.7"), the match may fail. Mitigation: the fallback for single-primary claims (positional scan) handles this gracefully. Only multi-primary claims require context matching.

### R3: Ratio keyword list may be incomplete (LOW)

The list `[OR, HR, RR, RRR, AHR, aHR, odds ratio, hazard ratio, risk ratio, relative risk]` may miss niche ratio measures (e.g., "IRR" for incidence rate ratio, "SHR" for subdistribution hazard ratio). Mitigation: accept this as a known limitation. When a new ratio keyword is encountered in dogfood, add it to the list. The failure mode (returning None) is safe -- it means no check runs, not a false positive.

### R4: Bug A fix may reduce true positive catch rate (LOW)

Some legitimate inconsistencies in multi-ratio claims will now be missed (returning None instead of checking). This is acceptable: precision > recall for numeric checks at this stage. A false positive damages credibility more than a missed inconsistency.

### R5: Existing test `test_p_value_ci_check_runs_when_no_or_ci_triple` may break (LOW)

This test in `test_numeric_engine.py` constructs assertions with `role="p_value"` and CIs but no primary with ratio keyword. Currently `_find_or_ci_triple` returns None for this (no primary at all), so the p-value path runs. Bug B fix does not affect this -- p_value role is not "primary". Verify during implementation.

## 11. Out of scope

| Tempting feature | Why refuse now |
|---|---|
| `primary_ref` field on NumericAssertion (Option A2) | Schema + prompt change. Higher risk, higher cost. Context-matching (Option A1) is sufficient for Phase 1. Revisit if context-matching fails on >5% of multi-ratio claims in future dogfood. |
| Additive mean-diff/CI consistency check | New check type. Would need its own validation. Bug B fix correctly excludes additive quantities from OR/CI check; adding a new check for them is Phase 2+ work. |
| Auto-detection of ratio vs. additive scale from context | Requires NLP or LLM -- forbidden in deterministic module. The keyword list is the Phase 1 approach. |
| Fixing `_find_p_value_ci_tuple` CI pairing | Same bug pattern (positional scan) but no false positives observed yet. Fix when a false positive appears. |
| Running the full benchmark suite as part of this PR | Cost-prohibitive. The unit tests + captured assertion lists are sufficient. |
| Expanding ratio keyword scan to cover `_find_p_value_ci_tuple` too | No evidence of bugs there. Surgical change only. |
