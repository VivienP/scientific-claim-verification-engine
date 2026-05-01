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

    def test_or_outside_ci_low(self) -> None:
        r = check_or_ci_consistency(10.0, 23.58, 73.71)
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

    def test_default_extracted_is_empty_list(self) -> None:
        r = check_or_ci_consistency(2.0, 1.0, 3.0)
        assert r.extracted == []


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
