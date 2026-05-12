"""Unit tests for src/numeric/checks.py — pure deterministic checks."""

from __future__ import annotations

from src.numeric.checks import (
    NumericAssertion,
    NumericCheckResult,
    check_or_ci_consistency,
    check_p_value_ci_consistency,
)


class TestCheckOrCiConsistency:
    def test_nguyen_2020_arm_happy_path(self) -> None:
        r = check_or_ci_consistency(40.53, 23.58, 73.71)
        assert r.consistent is True
        assert r.check_type == "or_ci_consistency"
        assert "consistent" in r.explanation.lower()

    def test_or_below_one_consistent(self) -> None:
        r = check_or_ci_consistency(0.101, 0.082, 0.124)
        assert r.consistent is True

    def test_or_outside_ci_high(self) -> None:
        r = check_or_ci_consistency(100.0, 23.58, 73.71)
        assert r.consistent is False
        assert "outside CI" in r.explanation

    def test_ci_inverted(self) -> None:
        r = check_or_ci_consistency(2.0, 3.0, 1.0)
        assert r.consistent is False
        assert "inverted" in r.explanation.lower()

    def test_ci_low_zero(self) -> None:
        r = check_or_ci_consistency(2.0, 0.0, 4.0)
        assert r.consistent is False
        assert "multiplicative" in r.explanation.lower()

    def test_ci_low_negative(self) -> None:
        r = check_or_ci_consistency(2.0, -0.5, 4.0)
        assert r.consistent is False

    def test_absurd_ci_ratio_or_above_one(self) -> None:
        # OR=2 with CI [0.01, 100] → ratio 100/0.01 = 10000, fails plausibility
        r = check_or_ci_consistency(2.0, 0.01, 100.0)
        assert r.consistent is False
        assert "plausibility" in r.explanation.lower()

    def test_absurd_ci_ratio_or_below_one(self) -> None:
        # OR=0.1 with CI [0.001, 50] → ratio 0.001/50 = 2e-5, fails plausibility
        r = check_or_ci_consistency(0.1, 0.001, 50.0)
        assert r.consistent is False
        assert "plausibility" in r.explanation.lower()

    def test_at_ci_boundary_low(self) -> None:
        r = check_or_ci_consistency(23.58, 23.58, 73.71)
        assert r.consistent is True

    def test_at_ci_boundary_high(self) -> None:
        r = check_or_ci_consistency(73.71, 23.58, 73.71)
        assert r.consistent is True

    def test_extracted_round_trips(self) -> None:
        ext = [
            NumericAssertion(
                raw_text="OR 40.53", value=40.53, unit=None, role="primary", context="ctx"
            ),
            NumericAssertion(
                raw_text="23.58", value=23.58, unit=None, role="ci_low", context="ctx"
            ),
            NumericAssertion(
                raw_text="73.71", value=73.71, unit=None, role="ci_high", context="ctx"
            ),
        ]
        r = check_or_ci_consistency(40.53, 23.58, 73.71, extracted=ext)
        assert isinstance(r, NumericCheckResult)
        assert r.extracted == ext


class TestCheckPValueCiConsistency:
    def test_significant_p_value_with_ci_excluding_null_is_consistent(self) -> None:
        r = check_p_value_ci_consistency(0.01, 1.2, 2.4, null_value=1.0)
        assert r.consistent is True
        assert r.check_type == "p_value_ci_consistency"

    def test_significant_p_value_with_ci_crossing_null_is_inconsistent(self) -> None:
        r = check_p_value_ci_consistency(0.001, 0.8, 1.2, null_value=1.0)
        assert r.consistent is False
        assert "crosses null" in r.explanation

    def test_non_significant_p_value_with_ci_excluding_null_is_inconsistent(self) -> None:
        r = check_p_value_ci_consistency(0.20, 1.2, 2.4, null_value=1.0)
        assert r.consistent is False
        assert "excludes null" in r.explanation

    def test_additive_null_defaults_to_zero(self) -> None:
        r = check_p_value_ci_consistency(0.03, -0.1, 0.4)
        assert r.consistent is False
        assert "crosses null=0.0" in r.explanation

    def test_invalid_p_value_is_inconsistent(self) -> None:
        r = check_p_value_ci_consistency(1.5, 1.2, 2.4, null_value=1.0)
        assert r.consistent is False
        assert "p-value" in r.explanation


class TestNumericCheckResultAmbiguous:
    """Lane A: ``ambiguous`` field surfaces compact multi-metric pairing skips."""

    def test_ambiguous_defaults_to_false(self) -> None:
        r = check_or_ci_consistency(0.74, 0.58, 0.95)
        assert r.ambiguous is False

    def test_ambiguous_can_be_constructed_directly(self) -> None:
        # Direct construction with ambiguous=True + consistent=True is the
        # contract the engine uses for the skip-the-check case.
        r = NumericCheckResult(
            check_type="or_ci_consistency",
            consistent=True,
            extracted=[],
            explanation="ambiguous pairing; check skipped",
            ambiguous=True,
        )
        assert r.ambiguous is True
        assert r.consistent is True


class TestNumericAssertionSpanFields:
    """Lane A: NumericAssertion gains span_start/end + sentence_id (optional, default None)."""

    def test_default_span_fields_are_none(self) -> None:
        a = NumericAssertion(
            raw_text="0.74",
            value=0.74,
            unit=None,
            role="primary",
            context="HR for MACE",
        )
        assert a.span_start is None
        assert a.span_end is None
        assert a.sentence_id is None

    def test_span_fields_round_trip(self) -> None:
        a = NumericAssertion(
            raw_text="0.74",
            value=0.74,
            unit=None,
            role="primary",
            context="HR for MACE",
            span_start=10,
            span_end=14,
            sentence_id=0,
        )
        assert a.span_start == 10
        assert a.span_end == 14
        assert a.sentence_id == 0
