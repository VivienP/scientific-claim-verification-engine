"""Unit tests for eval/copilot/auto_eval.py — fully offline."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval.copilot.auto_eval import (
    evaluate,
    load_gold,
    report_to_dict,
)
from src.copilot.models import (
    CopilotFields,
    CopilotMode,
    EnrichedVerification,
    RecommendedFix,
)
from src.models import (
    Claim,
    ProvenanceStep,
    ResolvedSource,
    ResolvedSourceSet,
    VerificationResult,
)
from src.pipeline import ClaimVerification

# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _cv(claim_id: str, verdict: str, claim_text: str = "Some claim.") -> ClaimVerification:
    claim = Claim(
        claim_id=claim_id,
        claim_text=claim_text,
        cited_authors=["X"],
        cited_year=2022,
        claim_type="factual_qualitative",
    )
    source = ResolvedSource(
        found=True,
        doi="10.1234/source",
        title="Source",
        abstract="abs",
        similarity_score=0.6,
    )
    source_set = ResolvedSourceSet(sources=(source,), citation_markers=(1,))
    # A1: supported/unsupported require fulltext-grade evidence
    eq = "quoted_passage" if verdict in ("supported", "unsupported") else "abstract_only"
    actual_confidence: float | None = None if verdict == "unverifiable" else 0.4
    result = VerificationResult(
        status=verdict,  # type: ignore[arg-type]
        explanation="x",
        confidence=actual_confidence,  # type: ignore[arg-type]
        evidence_quality=eq,  # type: ignore[arg-type]
    )
    return ClaimVerification(
        claim=claim,
        source=source,
        source_set=source_set,
        result=result,
        fetch_method="abstract",
    )


def _ev(
    claim_id: str,
    verdict: str,
    *,
    is_primary_source: bool | None = False,
    primary_source_doi: str | None = "10.9999/primary",
    fix_doi: str | None = "10.9999/primary",
    fix_present: bool = True,
) -> EnrichedVerification:
    cv = _cv(claim_id, verdict)
    fix = None
    if fix_present and verdict in {"unsupported", "partially_supported", "not_addressed"}:
        fix = RecommendedFix(
            action="swap_doi",
            regulatory_risk_level="high",
            suggested_doi=fix_doi,
            suggested_doi_title="t",
            reworded_claim=None,
            confidence=0.85,
            provenance_step_id="fs",
        )
    copilot = CopilotFields(
        verdict_rationale="r",
        recommended_fix=fix,
        is_primary_source=is_primary_source,
        study_design="systematic_review",
        risk_of_bias="high",
        conflicting_evidence_flag=None,
        primary_source_doi=primary_source_doi,
        novelty_claim=None,
    )
    step = ProvenanceStep(
        step_id="s",
        claim_id=claim_id,
        operation="copilot_rationale",
        input_hash="a" * 64,
        output_hash="b" * 64,
        model_id=None,
        timestamp=0.0,
        tokens_in=None,
        tokens_out=None,
        cache_hit=None,
        confidence=None,
    )
    return EnrichedVerification(
        base=cv,
        copilot=copilot,
        copilot_steps=(step,),
        mode=CopilotMode.PHARMA,
    )


def _gold_file(tmp_path: Path, claims: list[dict]) -> Path:
    p = tmp_path / "gold.json"
    p.write_text(json.dumps({"metadata": {"annotator": "test"}, "claims": claims}))
    return p


# ---------------------------------------------------------------------------
# load_gold
# ---------------------------------------------------------------------------


class TestLoadGold:
    def test_loads_dict_with_claims_key(self, tmp_path: Path) -> None:
        gold = _gold_file(tmp_path, [{"claim_id": "c1", "claim_text": "x"}])
        assert len(load_gold(gold)) == 1

    def test_loads_bare_list(self, tmp_path: Path) -> None:
        p = tmp_path / "gold.json"
        p.write_text(json.dumps([{"claim_id": "c1"}]))
        assert len(load_gold(p)) == 1

    def test_rejects_unknown_schema(self, tmp_path: Path) -> None:
        p = tmp_path / "gold.json"
        p.write_text(json.dumps("not a list or dict"))
        with pytest.raises(ValueError):
            load_gold(p)


# ---------------------------------------------------------------------------
# evaluate — happy path
# ---------------------------------------------------------------------------


class TestEvaluateHappyPath:
    def test_perfect_match(self, tmp_path: Path) -> None:
        evs = [
            _ev("c1", "unsupported", is_primary_source=False, primary_source_doi="10.9/x"),
        ]
        gold = _gold_file(
            tmp_path,
            [
                {
                    "claim_id": "c1",
                    "claim_text": "Some claim.",
                    "expected_verdict": "unsupported",
                    "is_primary_source": False,
                    "primary_source_doi": "10.9/x",
                }
            ],
        )
        report = evaluate(evs, gold)
        assert report.n_claims_evaluated == 1
        assert report.verdict_metric.precision == 1.0
        assert report.is_primary_source_metric.precision == 1.0
        assert report.primary_source_doi_metric.precision == 1.0
        assert report.doi_hallucination_rate == 0.0

    def test_verdict_mismatch_drops_precision(self, tmp_path: Path) -> None:
        evs = [
            _ev(
                "c1",
                "supported",
                is_primary_source=True,
                primary_source_doi=None,
                fix_present=False,
            ),
        ]
        gold = _gold_file(
            tmp_path,
            [
                {
                    "claim_id": "c1",
                    "claim_text": "Some claim.",
                    "expected_verdict": "unsupported",
                    "is_primary_source": False,
                }
            ],
        )
        report = evaluate(evs, gold)
        assert report.verdict_metric.precision == 0.0
        assert report.is_primary_source_metric.precision == 0.0

    def test_n_a_in_gold_treated_as_none(self, tmp_path: Path) -> None:
        evs = [_ev("c1", "unsupported", primary_source_doi=None)]
        gold = _gold_file(
            tmp_path,
            [
                {
                    "claim_id": "c1",
                    "claim_text": "Some claim.",
                    "expected_verdict": "unsupported",
                    "primary_source_doi": "N/A",
                }
            ],
        )
        report = evaluate(evs, gold)
        assert report.primary_source_doi_metric.n_gold == 0
        assert report.primary_source_doi_metric.n_predicted == 0


# ---------------------------------------------------------------------------
# evaluate — DOI hallucination gate
# ---------------------------------------------------------------------------


class TestDoiHallucinationGate:
    def test_no_verified_set_means_zero_hallucination_rate(self, tmp_path: Path) -> None:
        evs = [_ev("c1", "unsupported", fix_doi="10.fake/123")]
        gold = _gold_file(
            tmp_path,
            [
                {
                    "claim_id": "c1",
                    "claim_text": "Some claim.",
                    "expected_verdict": "unsupported",
                }
            ],
        )
        report = evaluate(evs, gold, crossref_verified_dois=None)
        # Trust fix_generator's gate by default.
        assert report.doi_hallucination_rate == 0.0

    def test_unverified_doi_flagged_when_set_provided(self, tmp_path: Path) -> None:
        evs = [_ev("c1", "unsupported", fix_doi="10.fake/123")]
        gold = _gold_file(
            tmp_path,
            [
                {
                    "claim_id": "c1",
                    "claim_text": "Some claim.",
                    "expected_verdict": "unsupported",
                }
            ],
        )
        verified: frozenset[str] = frozenset()
        report = evaluate(evs, gold, crossref_verified_dois=verified)
        assert report.n_doi_hallucinations == 1
        assert report.doi_hallucination_rate == 1.0
        assert report.passes_phase_b_gate is False  # gate must fail

    def test_verified_doi_not_flagged(self, tmp_path: Path) -> None:
        evs = [_ev("c1", "unsupported", fix_doi="10.real/123")]
        gold = _gold_file(
            tmp_path,
            [
                {
                    "claim_id": "c1",
                    "claim_text": "Some claim.",
                    "expected_verdict": "unsupported",
                }
            ],
        )
        verified = frozenset({"10.real/123"})
        report = evaluate(evs, gold, crossref_verified_dois=verified)
        assert report.n_doi_hallucinations == 0


# ---------------------------------------------------------------------------
# evaluate — fix-present rate
# ---------------------------------------------------------------------------


class TestFixPresentRate:
    def test_all_unsupported_have_fix(self, tmp_path: Path) -> None:
        evs = [
            _ev("c1", "unsupported"),
            _ev("c2", "unsupported"),
            _ev("c3", "supported", fix_present=False),
        ]
        gold = _gold_file(
            tmp_path,
            [
                {"claim_id": "c1", "claim_text": "x", "expected_verdict": "unsupported"},
                {"claim_id": "c2", "claim_text": "x", "expected_verdict": "unsupported"},
                {"claim_id": "c3", "claim_text": "x", "expected_verdict": "supported"},
            ],
        )
        report = evaluate(evs, gold)
        assert report.n_unsupported_in_gold == 2
        assert report.n_unsupported_with_fix == 2
        assert report.fix_present_rate_unsupported == 1.0

    def test_partial_fix_coverage(self, tmp_path: Path) -> None:
        evs = [
            _ev("c1", "unsupported"),
            _ev("c2", "unsupported", fix_present=False),
        ]
        gold = _gold_file(
            tmp_path,
            [
                {"claim_id": "c1", "claim_text": "x", "expected_verdict": "unsupported"},
                {"claim_id": "c2", "claim_text": "x", "expected_verdict": "unsupported"},
            ],
        )
        report = evaluate(evs, gold)
        assert report.fix_present_rate_unsupported == 0.5


# ---------------------------------------------------------------------------
# evaluate — Phase B gate
# ---------------------------------------------------------------------------


class TestPhaseBGate:
    def test_gate_passes_when_all_metrics_meet_thresholds(self, tmp_path: Path) -> None:
        evs = [_ev(f"c{i}", "unsupported", is_primary_source=False) for i in range(5)]
        gold = _gold_file(
            tmp_path,
            [
                {
                    "claim_id": f"c{i}",
                    "claim_text": "x",
                    "expected_verdict": "unsupported",
                    "is_primary_source": False,
                }
                for i in range(5)
            ],
        )
        report = evaluate(evs, gold)
        assert report.passes_phase_b_gate is True

    def test_gate_fails_when_primary_precision_below_80(self, tmp_path: Path) -> None:
        evs = [
            _ev("c1", "unsupported", is_primary_source=True),
            _ev("c2", "unsupported", is_primary_source=True),
        ]
        gold = _gold_file(
            tmp_path,
            [
                {
                    "claim_id": "c1",
                    "claim_text": "x",
                    "expected_verdict": "unsupported",
                    "is_primary_source": False,
                },
                {
                    "claim_id": "c2",
                    "claim_text": "x",
                    "expected_verdict": "unsupported",
                    "is_primary_source": False,
                },
            ],
        )
        report = evaluate(evs, gold)
        assert report.is_primary_source_metric.precision == 0.0
        assert report.passes_phase_b_gate is False


# ---------------------------------------------------------------------------
# Matching: by claim_id + by claim_text fallback
# ---------------------------------------------------------------------------


class TestMatching:
    def test_text_fallback_when_id_differs(self, tmp_path: Path) -> None:
        ev = _ev("our-id-001", "unsupported")
        # Force matching by text.
        ev_with_text = EnrichedVerification(
            base=ClaimVerification(
                claim=Claim(
                    claim_id="our-id-001",
                    claim_text="Aspirin reduces mortality.",
                    cited_authors=["X"],
                    cited_year=2020,
                    claim_type="factual_qualitative",
                ),
                source=ev.base.source,
                source_set=ev.base.source_set,
                result=ev.base.result,
                fetch_method="abstract",
            ),
            copilot=ev.copilot,
            copilot_steps=ev.copilot_steps,
            mode=ev.mode,
        )
        gold = _gold_file(
            tmp_path,
            [
                {
                    "claim_id": "different-id",
                    "claim_text": "Aspirin reduces mortality.",
                    "expected_verdict": "unsupported",
                }
            ],
        )
        report = evaluate([ev_with_text], gold)
        assert report.n_claims_evaluated == 1


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------


class TestSerialisation:
    def test_report_to_dict_round_trip_safe(self, tmp_path: Path) -> None:
        evs = [_ev("c1", "unsupported")]
        gold = _gold_file(
            tmp_path,
            [
                {
                    "claim_id": "c1",
                    "claim_text": "x",
                    "expected_verdict": "unsupported",
                }
            ],
        )
        report = evaluate(evs, gold)
        payload = report_to_dict(report)
        # Must be JSON-serialisable.
        json_text = json.dumps(payload)
        loaded = json.loads(json_text)
        assert loaded["n_claims_evaluated"] == 1
        assert "verdict" in loaded
        assert "doi_hallucination_rate" in loaded
