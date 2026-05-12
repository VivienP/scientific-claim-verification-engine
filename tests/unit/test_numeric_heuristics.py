"""Unit tests for src/numeric/heuristics.py — pure deterministic regex check."""

from __future__ import annotations


class TestClaimHasSpecificNumeric:
    """Tests for _claim_has_specific_numeric: true positives (numeric claims)
    and true negatives (qualitative claims)."""

    def test_percentage_triggers(self) -> None:
        from src.numeric.heuristics import _claim_has_specific_numeric

        assert _claim_has_specific_numeric("Sustained response at 12 weeks was 20%") is True

    def test_percentage_decimal_triggers(self) -> None:
        from src.numeric.heuristics import _claim_has_specific_numeric

        assert _claim_has_specific_numeric("Response rate was 14.5% in the treatment arm") is True

    def test_p_value_less_than_triggers(self) -> None:
        from src.numeric.heuristics import _claim_has_specific_numeric

        assert _claim_has_specific_numeric("The result was significant (p < 0.001)") is True

    def test_p_value_equals_triggers(self) -> None:
        from src.numeric.heuristics import _claim_has_specific_numeric

        assert _claim_has_specific_numeric("p=0.02 for the primary endpoint") is True

    def test_n_equals_triggers(self) -> None:
        from src.numeric.heuristics import _claim_has_specific_numeric

        assert _claim_has_specific_numeric("The trial enrolled n=233 patients") is True

    def test_n_equals_uppercase_triggers(self) -> None:
        from src.numeric.heuristics import _claim_has_specific_numeric

        assert _claim_has_specific_numeric("N=180 participants completed the study") is True

    def test_95_ci_triggers(self) -> None:
        from src.numeric.heuristics import _claim_has_specific_numeric

        assert _claim_has_specific_numeric("Effect size was 0.45 (95% CI 0.31-0.59)") is True

    def test_hr_abbreviation_triggers(self) -> None:
        from src.numeric.heuristics import _claim_has_specific_numeric

        assert _claim_has_specific_numeric("HR 0.55 favoring the treatment group") is True

    def test_or_abbreviation_triggers(self) -> None:
        from src.numeric.heuristics import _claim_has_specific_numeric

        assert _claim_has_specific_numeric("OR=1.7 for adverse events") is True

    def test_rr_abbreviation_triggers(self) -> None:
        from src.numeric.heuristics import _claim_has_specific_numeric

        assert _claim_has_specific_numeric("RR 2.3 in the intervention group") is True

    def test_hazard_ratio_phrase_triggers(self) -> None:
        from src.numeric.heuristics import _claim_has_specific_numeric

        assert _claim_has_specific_numeric("The hazard ratio was 0.72") is True

    def test_odds_ratio_phrase_triggers(self) -> None:
        from src.numeric.heuristics import _claim_has_specific_numeric

        assert _claim_has_specific_numeric("odds ratio of 2.1 was observed") is True

    def test_cohens_d_triggers(self) -> None:
        from src.numeric.heuristics import _claim_has_specific_numeric

        assert _claim_has_specific_numeric("effect size Cohen's d = 0.6") is True

    def test_hedges_g_triggers(self) -> None:
        from src.numeric.heuristics import _claim_has_specific_numeric

        assert _claim_has_specific_numeric("Hedges' g = 0.48 across studies") is True

    def test_week_timepoint_triggers(self) -> None:
        from src.numeric.heuristics import _claim_has_specific_numeric

        assert _claim_has_specific_numeric("response rate at week 12 was assessed") is True

    def test_dose_mg_triggers(self) -> None:
        from src.numeric.heuristics import _claim_has_specific_numeric

        assert _claim_has_specific_numeric("25 mg psilocybin was administered") is True

    def test_dose_mcg_triggers(self) -> None:
        from src.numeric.heuristics import _claim_has_specific_numeric

        assert _claim_has_specific_numeric("100 mcg dose was used") is True

    def test_goodwin_claim_triggers(self) -> None:
        """The triggering Goodwin NEJM 2022 claim must be detected as numeric."""
        from src.numeric.heuristics import _claim_has_specific_numeric

        assert (
            _claim_has_specific_numeric(
                "Sustained response rates at 12 weeks were only 20% in the largest randomized trial"
            )
            is True
        )

    # --- Negative cases (qualitative claims) ---

    def test_qualitative_claim_does_not_trigger(self) -> None:
        from src.numeric.heuristics import _claim_has_specific_numeric

        assert _claim_has_specific_numeric("psilocybin reduces depression symptoms") is False

    def test_qualitative_causal_claim_does_not_trigger(self) -> None:
        from src.numeric.heuristics import _claim_has_specific_numeric

        assert _claim_has_specific_numeric("Exercise improves cardiovascular health") is False

    def test_empty_string_does_not_trigger(self) -> None:
        from src.numeric.heuristics import _claim_has_specific_numeric

        assert _claim_has_specific_numeric("") is False

    def test_generic_text_without_numbers_does_not_trigger(self) -> None:
        from src.numeric.heuristics import _claim_has_specific_numeric

        assert (
            _claim_has_specific_numeric(
                "The treatment was associated with improved outcomes in this population"
            )
            is False
        )

    def test_plain_number_without_unit_does_not_trigger(self) -> None:
        """A bare number without a recognized unit/pattern is NOT a specific numeric claim."""
        from src.numeric.heuristics import _claim_has_specific_numeric

        # '3 studies' is not a numeric assertion in the Results-section sense
        assert _claim_has_specific_numeric("Three studies examined this") is False
