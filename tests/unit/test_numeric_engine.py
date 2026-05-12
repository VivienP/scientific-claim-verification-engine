"""Unit tests for src/numeric/engine.py — covers _find_or_ci_triple and run_numeric_check.

Tests are organized in four classes:
  TestFindOrCiTripleBugA  -- Bug A: multi-ratio claim, primary has no CI
  TestFindOrCiTripleBugB  -- Bug B: non-ratio primary (mean diff, Hedges' g)
  TestFindOrCiTripleHappyPath -- non-regression: single primary, window match
  TestRunNumericCheck -- integration with run_numeric_check entry point
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from anthropic.types import TextBlock

from src.numeric.checks import NumericAssertion
from src.numeric.engine import (
    MAX_CHARS_BETWEEN_PRIMARY_AND_CI,
    _detect_or_ci_pairing_ambiguity,
    _find_or_ci_triple,
    _find_span_anchored_triple,
    run_numeric_check,
)


def _text_block(text: str) -> TextBlock:
    return TextBlock(type="text", text=text)


# ---------------------------------------------------------------------------
# Bug A: multi-ratio claim where the selected primary has no CI
# ---------------------------------------------------------------------------


class TestFindOrCiTripleBugA:
    def test_or_without_ci_in_multi_ratio_returns_none(self) -> None:
        """Bug A: OR 1.7 has no CI; CI [0.48, 0.74] belongs to HR 0.59.
        Must return None, not (1.7, 0.48, 0.74).
        Reproduces claim_id 23a27499."""
        assertions = [
            NumericAssertion(
                raw_text="OR 1.7",
                value=1.7,
                unit=None,
                role="primary",
                context="odds ratio for ORR with chemo-ICI vs ICI alone",
            ),
            NumericAssertion(
                raw_text="HR 0.59",
                value=0.59,
                unit=None,
                role="primary",
                context="hazard ratio for PFS with chemo-ICI vs ICI alone",
            ),
            NumericAssertion(
                raw_text="0.48",
                value=0.48,
                unit=None,
                role="ci_low",
                context="95% CI lower bound for PFS HR 0.59",
            ),
            NumericAssertion(
                raw_text="0.74",
                value=0.74,
                unit=None,
                role="ci_high",
                context="95% CI upper bound for PFS HR 0.59",
            ),
        ]
        assert _find_or_ci_triple(assertions) is None

    def test_rr_without_ci_in_multi_ratio_returns_none(self) -> None:
        """Bug A variant: RR 1.62 has no CI; CI [0.32, 0.97] belongs to HR 0.55.
        Reproduces claim_id 37ddafbb."""
        assertions = [
            NumericAssertion(
                raw_text="RR 1.62",
                value=1.62,
                unit=None,
                role="primary",
                context="relative risk for objective response rate (ORR) with combination therapy",
            ),
            NumericAssertion(
                raw_text="HR 0.55",
                value=0.55,
                unit=None,
                role="primary",
                context="hazard ratio for progression-free survival (PFS) with combination therapy",
            ),
            NumericAssertion(
                raw_text="0.32",
                value=0.32,
                unit=None,
                role="ci_low",
                context="95% CI lower bound for HR 0.55 (PFS)",
            ),
            NumericAssertion(
                raw_text="0.97",
                value=0.97,
                unit=None,
                role="ci_high",
                context="95% CI upper bound for HR 0.55 (PFS)",
            ),
        ]
        assert _find_or_ci_triple(assertions) is None


# ---------------------------------------------------------------------------
# Bug B: non-ratio primary (mean difference, Hedges' g) must not match
# ---------------------------------------------------------------------------


class TestFindOrCiTripleBugB:
    def test_mean_diff_returns_none(self) -> None:
        """Bug B: mean MADRS reduction (unit=None, no ratio keyword).
        Reproduces claim_id 944726cb."""
        assertions = [
            NumericAssertion(
                raw_text="14.9",
                value=14.9,
                unit=None,
                role="primary",
                context="mean MADRS reduction at week 3 for psilocybin adjunct to SSRIs",
            ),
            NumericAssertion(
                raw_text="-20.7",
                value=-20.7,
                unit=None,
                role="ci_low",
                context="95% CI lower bound for mean MADRS reduction at week 3",
            ),
            NumericAssertion(
                raw_text="-9.2",
                value=-9.2,
                unit=None,
                role="ci_high",
                context="95% CI upper bound for mean MADRS reduction at week 3",
            ),
        ]
        assert _find_or_ci_triple(assertions) is None

    def test_hedges_g_returns_none(self) -> None:
        """Bug B: Hedges' g is an effect size, not a ratio measure.
        Reproduces claim_id c110e5f5."""
        assertions = [
            NumericAssertion(
                raw_text="-7.14",
                value=-7.14,
                unit=None,
                role="primary",
                context="mean QIDS change at 3 weeks",
            ),
            NumericAssertion(
                raw_text="p=0.02",
                value=0.02,
                unit=None,
                role="p_value",
                context="p-value for mean QIDS change at 3 weeks",
            ),
            NumericAssertion(
                raw_text="-1.27",
                value=-1.27,
                unit=None,
                role="primary",
                context="Hedges' g effect size for QIDS change at 3 weeks",
            ),
            NumericAssertion(
                raw_text="-2.40",
                value=-2.4,
                unit=None,
                role="ci_low",
                context="95% CI lower bound for Hedges' g",
            ),
            NumericAssertion(
                raw_text="-0.37",
                value=-0.37,
                unit=None,
                role="ci_high",
                context="95% CI upper bound for Hedges' g",
            ),
        ]
        assert _find_or_ci_triple(assertions) is None


# ---------------------------------------------------------------------------
# Happy path: single primary, window match, strong match
# ---------------------------------------------------------------------------


class TestFindOrCiTripleHappyPath:
    def test_single_or_with_explicit_ci_in_context(self) -> None:
        """Single OR with CI context containing primary's raw_text — strong match."""
        assertions = [
            NumericAssertion(
                raw_text="OR 40.53",
                value=40.53,
                unit=None,
                role="primary",
                context="odds ratio ARM A+T- vs A-T-",
            ),
            NumericAssertion(
                raw_text="23.58",
                value=23.58,
                unit=None,
                role="ci_low",
                context="95% CI lower bound for OR 40.53",
            ),
            NumericAssertion(
                raw_text="73.71",
                value=73.71,
                unit=None,
                role="ci_high",
                context="95% CI upper bound for OR 40.53",
            ),
        ]
        assert _find_or_ci_triple(assertions) == (40.53, 23.58, 73.71)

    def test_or_with_own_ci_in_multi_ratio_claim(self) -> None:
        """Strong match in a multi-primary claim: OR has its own CI repeated in
        CI context. Spec section 8 Phase 1 case, ensures strong match wins
        before window fallback when both could apply."""
        assertions = [
            NumericAssertion(
                raw_text="OR 1.7",
                value=1.7,
                unit=None,
                role="primary",
                context="odds ratio for ORR",
            ),
            NumericAssertion(
                raw_text="1.1",
                value=1.1,
                unit=None,
                role="ci_low",
                context="95% CI lower bound for OR 1.7",
            ),
            NumericAssertion(
                raw_text="2.5",
                value=2.5,
                unit=None,
                role="ci_high",
                context="95% CI upper bound for OR 1.7",
            ),
            NumericAssertion(
                raw_text="HR 0.59",
                value=0.59,
                unit=None,
                role="primary",
                context="hazard ratio for PFS",
            ),
            NumericAssertion(
                raw_text="0.48",
                value=0.48,
                unit=None,
                role="ci_low",
                context="95% CI lower bound for HR 0.59",
            ),
            NumericAssertion(
                raw_text="0.74",
                value=0.74,
                unit=None,
                role="ci_high",
                context="95% CI upper bound for HR 0.59",
            ),
        ]
        assert _find_or_ci_triple(assertions) == (1.7, 1.1, 2.5)

    def test_two_hr_each_with_own_window_ci(self) -> None:
        """claim 56eec845: two HRs each with own CI in sequence.
        Window match returns first HR's triple (0.74, 0.58, 0.95)."""
        assertions = [
            NumericAssertion(
                raw_text="HR 0.74",
                value=0.74,
                unit=None,
                role="primary",
                context="hazard ratio for MACE with subcutaneous semaglutide (pooled SUSTAIN 6 and PIONEER 6)",
            ),
            NumericAssertion(
                raw_text="0.58",
                value=0.58,
                unit=None,
                role="ci_low",
                context="95% CI lower bound for HR of subcutaneous semaglutide for MACE",
            ),
            NumericAssertion(
                raw_text="0.95",
                value=0.95,
                unit=None,
                role="ci_high",
                context="95% CI upper bound for HR of subcutaneous semaglutide for MACE",
            ),
            NumericAssertion(
                raw_text="HR 0.79",
                value=0.79,
                unit=None,
                role="primary",
                context="hazard ratio for MACE with oral semaglutide (PIONEER 6)",
            ),
            NumericAssertion(
                raw_text="0.57",
                value=0.57,
                unit=None,
                role="ci_low",
                context="95% CI lower bound for HR of oral semaglutide for MACE",
            ),
            NumericAssertion(
                raw_text="1.11",
                value=1.11,
                unit=None,
                role="ci_high",
                context="95% CI upper bound for HR of oral semaglutide for MACE",
            ),
        ]
        assert _find_or_ci_triple(assertions) == (0.74, 0.58, 0.95)

    def test_hr_primary_then_comparator_with_ci(self) -> None:
        """claim 40ab3510: primary '0.96' (MACE HR, no HR prefix in raw_text),
        context says 'MACE HR'. Second value is role=comparator (not primary),
        so window correctly includes the CIs after index 0.
        Must return (0.96, 0.70, 1.31)."""
        assertions = [
            NumericAssertion(
                raw_text="0.96",
                value=0.96,
                unit=None,
                role="primary",
                context="MACE HR for patients with LVEF <40% in EXSCEL",
            ),
            NumericAssertion(
                raw_text="0.70",
                value=0.70,
                unit=None,
                role="ci_low",
                context="95% CI lower bound for MACE HR in LVEF <40%",
            ),
            NumericAssertion(
                raw_text="1.31",
                value=1.31,
                unit=None,
                role="ci_high",
                context="95% CI upper bound for MACE HR in LVEF <40%",
            ),
            NumericAssertion(
                raw_text="0.84",
                value=0.84,
                unit=None,
                role="comparator",
                context="MACE HR for patients with LVEF >=40% in EXSCEL",
            ),
            NumericAssertion(
                raw_text="0.71",
                value=0.71,
                unit=None,
                role="ci_low",
                context="95% CI lower bound for MACE HR in LVEF >=40%",
            ),
            NumericAssertion(
                raw_text="0.98",
                value=0.98,
                unit=None,
                role="ci_high",
                context="95% CI upper bound for MACE HR in LVEF >=40%",
            ),
        ]
        assert _find_or_ci_triple(assertions) == (0.96, 0.70, 1.31)

    def test_glp1_13_pct_relative_risk_reduction_does_not_match(self) -> None:
        """claim cefcd22f: '13%' has context 'relative risk reduction' (exclusion).
        Must skip '13%' and pick 'HR 0.87' -> return (0.87, 0.78, 0.97)."""
        assertions = [
            NumericAssertion(
                raw_text="13%",
                value=13.0,
                unit="%",
                role="primary",
                context="relative risk reduction with liraglutide in LEADER trial",
            ),
            NumericAssertion(
                raw_text="HR 0.87",
                value=0.87,
                unit=None,
                role="primary",
                context="hazard ratio for liraglutide vs comparator in LEADER trial",
            ),
            NumericAssertion(
                raw_text="0.78",
                value=0.78,
                unit=None,
                role="ci_low",
                context="95% CI lower bound for HR 0.87",
            ),
            NumericAssertion(
                raw_text="0.97",
                value=0.97,
                unit=None,
                role="ci_high",
                context="95% CI upper bound for HR 0.87",
            ),
        ]
        assert _find_or_ci_triple(assertions) == (0.87, 0.78, 0.97)

    def test_empty_assertions_returns_none(self) -> None:
        assert _find_or_ci_triple([]) is None

    def test_only_p_value_returns_none(self) -> None:
        assertions = [
            NumericAssertion(
                raw_text="p=0.02",
                value=0.02,
                unit=None,
                role="p_value",
                context="p-value for treatment effect",
            )
        ]
        assert _find_or_ci_triple(assertions) is None

    def test_ratio_keyword_in_word_boundary_only(self) -> None:
        """'PRIOR' and 'MONITOR' must NOT trigger a match on 'OR'.
        Context contains 'PRIOR therapy' — not a ratio measure."""
        assertions = [
            NumericAssertion(
                raw_text="5.0",
                value=5.0,
                unit=None,
                role="primary",
                context="PRIOR therapy count in MONITOR study",
            ),
            NumericAssertion(
                raw_text="2.0",
                value=2.0,
                unit=None,
                role="ci_low",
                context="95% CI lower bound",
            ),
            NumericAssertion(
                raw_text="8.0",
                value=8.0,
                unit=None,
                role="ci_high",
                context="95% CI upper bound",
            ),
        ]
        assert _find_or_ci_triple(assertions) is None


class TestSpanAnchoredPairing:
    """Lane A tier 1: span+sentence anchoring picks the right CIs deterministically."""

    def test_span_anchored_resolves_compact_two_primary_sentence(self) -> None:
        """Compact 'HR 0.74 (95% CI 0.58-0.95) ... HR 0.79 (95% CI 0.57-1.11)'.

        Both HRs in sentence 0 with same-sentence CIs that the window heuristic
        could mis-pair. Span anchoring picks the CIs closest to HR 0.74.
        """
        assertions = [
            NumericAssertion(
                raw_text="HR 0.74",
                value=0.74,
                unit=None,
                role="primary",
                context="subcut semaglutide pooled HR for MACE",
                span_start=0,
                span_end=7,
                sentence_id=0,
            ),
            NumericAssertion(
                raw_text="0.58",
                value=0.58,
                unit=None,
                role="ci_low",
                context="95% CI lower bound for HR 0.74",
                span_start=20,
                span_end=24,
                sentence_id=0,
            ),
            NumericAssertion(
                raw_text="0.95",
                value=0.95,
                unit=None,
                role="ci_high",
                context="95% CI upper bound for HR 0.74",
                span_start=25,
                span_end=29,
                sentence_id=0,
            ),
            NumericAssertion(
                raw_text="HR 0.79",
                value=0.79,
                unit=None,
                role="primary",
                context="oral semaglutide PIONEER 6 HR for MACE",
                span_start=120,
                span_end=127,
                sentence_id=1,
            ),
            NumericAssertion(
                raw_text="0.57",
                value=0.57,
                unit=None,
                role="ci_low",
                context="95% CI lower bound for HR 0.79",
                span_start=140,
                span_end=144,
                sentence_id=1,
            ),
            NumericAssertion(
                raw_text="1.11",
                value=1.11,
                unit=None,
                role="ci_high",
                context="95% CI upper bound for HR 0.79",
                span_start=145,
                span_end=149,
                sentence_id=1,
            ),
        ]
        triple = _find_span_anchored_triple(assertions, 0)
        assert triple == (0.74, 0.58, 0.95)

    def test_span_anchored_returns_none_when_primary_lacks_span(self) -> None:
        """Missing spans -> tier 1 returns None; caller falls through."""
        assertions = [
            NumericAssertion(
                raw_text="HR 0.74",
                value=0.74,
                unit=None,
                role="primary",
                context="HR for MACE",
            ),
            NumericAssertion(
                raw_text="0.58",
                value=0.58,
                unit=None,
                role="ci_low",
                context="ci low",
                span_start=10,
                span_end=14,
                sentence_id=0,
            ),
            NumericAssertion(
                raw_text="0.95",
                value=0.95,
                unit=None,
                role="ci_high",
                context="ci high",
                span_start=15,
                span_end=19,
                sentence_id=0,
            ),
        ]
        assert _find_span_anchored_triple(assertions, 0) is None

    def test_span_anchored_rejects_far_ci_across_window(self) -> None:
        """CI gap exceeds MAX_CHARS_BETWEEN_PRIMARY_AND_CI -> tier 1 returns None."""
        far = MAX_CHARS_BETWEEN_PRIMARY_AND_CI + 5
        assertions = [
            NumericAssertion(
                raw_text="HR 0.74",
                value=0.74,
                unit=None,
                role="primary",
                context="HR for MACE",
                span_start=0,
                span_end=7,
                sentence_id=0,
            ),
            NumericAssertion(
                raw_text="0.58",
                value=0.58,
                unit=None,
                role="ci_low",
                context="ci low",
                span_start=7 + far,
                span_end=11 + far,
                sentence_id=0,
            ),
            NumericAssertion(
                raw_text="0.95",
                value=0.95,
                unit=None,
                role="ci_high",
                context="ci high",
                span_start=12 + far,
                span_end=16 + far,
                sentence_id=0,
            ),
        ]
        assert _find_span_anchored_triple(assertions, 0) is None

    def test_span_anchored_rejects_cross_sentence_cis(self) -> None:
        """CIs in a different sentence than the primary -> tier 1 returns None."""
        assertions = [
            NumericAssertion(
                raw_text="HR 0.74",
                value=0.74,
                unit=None,
                role="primary",
                context="HR for MACE",
                span_start=0,
                span_end=7,
                sentence_id=0,
            ),
            NumericAssertion(
                raw_text="0.58",
                value=0.58,
                unit=None,
                role="ci_low",
                context="ci low",
                span_start=20,
                span_end=24,
                sentence_id=1,
            ),
            NumericAssertion(
                raw_text="0.95",
                value=0.95,
                unit=None,
                role="ci_high",
                context="ci high",
                span_start=25,
                span_end=29,
                sentence_id=1,
            ),
        ]
        assert _find_span_anchored_triple(assertions, 0) is None

    def test_find_or_ci_triple_prefers_span_anchored_over_window(self) -> None:
        """Tier 1 wins when both tier 1 and tier 3 (window) would match."""
        assertions = [
            NumericAssertion(
                raw_text="HR 0.74",
                value=0.74,
                unit=None,
                role="primary",
                context="MACE",
                span_start=0,
                span_end=7,
                sentence_id=0,
            ),
            NumericAssertion(
                raw_text="0.58",
                value=0.58,
                unit=None,
                role="ci_low",
                context="ci low",
                span_start=20,
                span_end=24,
                sentence_id=0,
            ),
            NumericAssertion(
                raw_text="0.95",
                value=0.95,
                unit=None,
                role="ci_high",
                context="ci high",
                span_start=25,
                span_end=29,
                sentence_id=0,
            ),
        ]
        assert _find_or_ci_triple(assertions) == (0.74, 0.58, 0.95)


class TestAmbiguityDetection:
    """Lane A ambiguity gate: refuse to flag inconsistencies when pairing is unclear."""

    def test_ambiguity_detected_on_cross_referenced_ci_context(self) -> None:
        """Two HRs without spans + window CI whose context names the OTHER HR.

        Reproduces the Zhou/Wang pattern: compact multi-metric sentence where
        the LLM's emitted CI context binds the CI to a non-selected primary.
        The window-match would still pick (HR_A, CI_x_for_HR_B) — ambiguous.
        """
        assertions = [
            NumericAssertion(
                raw_text="HR 0.74",
                value=0.74,
                unit=None,
                role="primary",
                context="MACE",
            ),
            NumericAssertion(
                raw_text="HR 0.79",
                value=0.79,
                unit=None,
                role="primary",
                context="oral MACE",
            ),
            NumericAssertion(
                raw_text="0.57",
                value=0.57,
                unit=None,
                role="ci_low",
                context="95% CI lower bound for HR 0.79 (oral)",
            ),
            NumericAssertion(
                raw_text="1.11",
                value=1.11,
                unit=None,
                role="ci_high",
                context="95% CI upper bound for HR 0.79 (oral)",
            ),
        ]
        assert _detect_or_ci_pairing_ambiguity(assertions) is True

    def test_ambiguity_not_detected_with_single_primary(self) -> None:
        assertions = [
            NumericAssertion(
                raw_text="HR 0.74",
                value=0.74,
                unit=None,
                role="primary",
                context="MACE",
            ),
            NumericAssertion(
                raw_text="0.58",
                value=0.58,
                unit=None,
                role="ci_low",
                context="ci",
            ),
            NumericAssertion(
                raw_text="0.95",
                value=0.95,
                unit=None,
                role="ci_high",
                context="ci",
            ),
        ]
        assert _detect_or_ci_pairing_ambiguity(assertions) is False

    def test_ambiguity_not_detected_when_span_anchored_resolves(self) -> None:
        """Span anchoring resolves the first primary -> no ambiguity."""
        assertions = [
            NumericAssertion(
                raw_text="HR 0.74",
                value=0.74,
                unit=None,
                role="primary",
                context="MACE",
                span_start=0,
                span_end=7,
                sentence_id=0,
            ),
            NumericAssertion(
                raw_text="0.58",
                value=0.58,
                unit=None,
                role="ci_low",
                context="ci low for HR 0.74",
                span_start=20,
                span_end=24,
                sentence_id=0,
            ),
            NumericAssertion(
                raw_text="0.95",
                value=0.95,
                unit=None,
                role="ci_high",
                context="ci high for HR 0.74",
                span_start=25,
                span_end=29,
                sentence_id=0,
            ),
            NumericAssertion(
                raw_text="HR 0.79",
                value=0.79,
                unit=None,
                role="primary",
                context="oral",
                span_start=120,
                span_end=127,
                sentence_id=1,
            ),
        ]
        assert _detect_or_ci_pairing_ambiguity(assertions) is False


class TestRunNumericCheck:
    @patch("src.numeric.extract.anthropic.Anthropic")
    def test_malformed_llm_response_falls_back_to_none(self, mock_anthropic_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.content = [_text_block("this is not valid json at all")]
        mock_response.usage.input_tokens = 100
        mock_response.usage.output_tokens = 10
        mock_response.usage.cache_read_input_tokens = 0
        mock_response.usage.cache_creation_input_tokens = 0
        mock_client.messages.create.return_value = mock_response

        result, steps = run_numeric_check("Some claim with OR 5 (95% CI 2-8)")

        assert result is None
        assert len(steps) == 1
        assert steps[0].operation == "numeric_extract"

    @patch("src.numeric.engine.extract_numeric_assertions")
    def test_ambiguous_pairing_emits_ambiguous_skipped_check(self, mock_extract: MagicMock) -> None:
        """End-to-end: ambiguous assertions -> NumericCheckResult(ambiguous=True)."""
        from src.models import ProvenanceStep
        from src.numeric.checks import NumericAssertion

        mock_extract.return_value = (
            [
                NumericAssertion(
                    raw_text="HR 0.74",
                    value=0.74,
                    unit=None,
                    role="primary",
                    context="MACE",
                ),
                NumericAssertion(
                    raw_text="HR 0.79",
                    value=0.79,
                    unit=None,
                    role="primary",
                    context="oral MACE",
                ),
                NumericAssertion(
                    raw_text="0.57",
                    value=0.57,
                    unit=None,
                    role="ci_low",
                    context="95% CI lower bound for HR 0.79",
                ),
                NumericAssertion(
                    raw_text="1.11",
                    value=1.11,
                    unit=None,
                    role="ci_high",
                    context="95% CI upper bound for HR 0.79",
                ),
            ],
            ProvenanceStep(
                step_id="ne",
                claim_id="claim-1",
                operation="numeric_extract",
                input_hash="i",
                output_hash="o",
                model_id="m",
                timestamp=0.0,
                tokens_in=10,
                tokens_out=5,
                cache_hit=False,
                confidence=None,
            ),
        )

        result, steps = run_numeric_check("two HRs in one sentence")
        assert result is not None
        assert result.ambiguous is True
        assert result.consistent is True  # we did not detect an inconsistency
        assert "ambiguous primary-CI pairing" in result.explanation
        assert len(steps) == 2
        # The check step is deterministic — no model_id, no tokens.
        check_step = steps[1]
        assert check_step.operation == "numeric_check"
        assert check_step.model_id is None
        assert check_step.tokens_in is None

    @patch("src.numeric.engine.extract_numeric_assertions")
    def test_p_value_ci_check_runs_when_no_or_ci_triple(self, mock_extract: MagicMock) -> None:
        from src.models import ProvenanceStep
        from src.numeric.checks import NumericAssertion

        mock_extract.return_value = (
            [
                NumericAssertion(
                    raw_text="p = 0.001",
                    value=0.001,
                    unit=None,
                    role="p_value",
                    context="odds ratio difference between groups",
                ),
                NumericAssertion(
                    raw_text="0.8",
                    value=0.8,
                    unit=None,
                    role="ci_low",
                    context="95% CI lower bound for odds ratio",
                ),
                NumericAssertion(
                    raw_text="1.2",
                    value=1.2,
                    unit=None,
                    role="ci_high",
                    context="95% CI upper bound for odds ratio",
                ),
            ],
            ProvenanceStep(
                step_id="ne",
                claim_id="claim-1",
                operation="numeric_extract",
                input_hash="i",
                output_hash="o",
                model_id="m",
                timestamp=0.0,
                tokens_in=10,
                tokens_out=5,
                cache_hit=False,
                confidence=None,
            ),
        )

        result, steps = run_numeric_check("Claim reports p = 0.001, 95% CI 0.8-1.2")

        assert result is not None
        assert result.check_type == "p_value_ci_consistency"
        assert result.consistent is False
        assert len(steps) == 2
        assert steps[1].operation == "numeric_check"
