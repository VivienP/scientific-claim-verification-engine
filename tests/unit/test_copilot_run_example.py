"""Unit tests for examples/copilot_run.py — serialise/deserialise round trip."""

from __future__ import annotations

import json
from pathlib import Path

from examples.copilot_run import (
    deserialize_enriched,
    serialize_enriched,
    write_enriched,
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


def _make_one(
    *,
    claim_id: str = "cl-1",
    verdict: str = "unsupported",
    with_fix: bool = True,
) -> EnrichedVerification:
    claim = Claim(
        claim_id=claim_id,
        claim_text="Drug X reduces biomarker Y.",
        cited_authors=["Jones"],
        cited_year=2022,
        claim_type="factual_qualitative",
    )
    source = ResolvedSource(
        found=True,
        doi="10.1234/source",
        title="Review",
        abstract="abstract",
        similarity_score=0.7,
    )
    source_set = ResolvedSourceSet(sources=(source,), citation_markers=(1,))
    result = VerificationResult(
        status=verdict,  # type: ignore[arg-type]
        explanation="Source does not support claim.",
        confidence=0.3,
    )
    cv = ClaimVerification(
        claim=claim,
        source=source,
        source_set=source_set,
        result=result,
        fetch_method="abstract",
    )
    fix: RecommendedFix | None = None
    if with_fix:
        fix = RecommendedFix(
            action="swap_doi",
            regulatory_risk_level="high",
            suggested_doi="10.9999/primary",
            suggested_doi_title="Primary RCT",
            reworded_claim=None,
            confidence=0.85,
            provenance_step_id="fs-1",
        )
    copilot = CopilotFields(
        verdict_rationale="The cited source is a review.",
        recommended_fix=fix,
        is_primary_source=False,
        study_design="systematic_review",
        risk_of_bias="medium",
        conflicting_evidence_flag=None,
        primary_source_doi="10.9999/primary",
        novelty_claim=None,
    )
    step = ProvenanceStep(
        step_id="s1",
        claim_id=claim_id,
        operation="copilot_rationale",
        input_hash="a" * 64,
        output_hash="b" * 64,
        model_id="claude-sonnet-4-6",
        timestamp=1234.5,
        tokens_in=100,
        tokens_out=10,
        cache_hit=False,
        confidence=None,
    )
    return EnrichedVerification(
        base=cv,
        copilot=copilot,
        copilot_steps=(step,),
        mode=CopilotMode.PHARMA,
    )


# ---------------------------------------------------------------------------
# Round-trip
# ---------------------------------------------------------------------------


class TestRoundTrip:
    def test_single_enriched_round_trip(self) -> None:
        original = [_make_one()]
        payload = serialize_enriched(original)
        # Must be JSON-serialisable.
        json_text = json.dumps(payload)
        loaded = deserialize_enriched(json.loads(json_text))
        assert len(loaded) == 1
        assert loaded[0].base.claim.claim_id == "cl-1"
        assert loaded[0].copilot.verdict_rationale == "The cited source is a review."
        assert loaded[0].copilot.recommended_fix is not None
        assert loaded[0].copilot.recommended_fix.suggested_doi == "10.9999/primary"
        assert loaded[0].mode == CopilotMode.PHARMA

    def test_round_trip_preserves_provenance_steps(self) -> None:
        ev = _make_one()
        loaded = deserialize_enriched(serialize_enriched([ev]))[0]
        assert len(loaded.copilot_steps) == 1
        step = loaded.copilot_steps[0]
        assert step.operation == "copilot_rationale"
        assert step.model_id == "claude-sonnet-4-6"
        assert step.tokens_in == 100

    def test_round_trip_with_no_fix(self) -> None:
        ev = _make_one(verdict="supported", with_fix=False)
        loaded = deserialize_enriched(serialize_enriched([ev]))[0]
        assert loaded.copilot.recommended_fix is None

    def test_multiple_modes(self) -> None:
        evs = [
            _make_one(claim_id="c1"),
            EnrichedVerification(
                base=_make_one(claim_id="c2").base,
                copilot=CopilotFields(
                    verdict_rationale="r2",
                    recommended_fix=None,
                    is_primary_source=None,
                    study_design=None,
                    risk_of_bias=None,
                    conflicting_evidence_flag=None,
                    primary_source_doi=None,
                    novelty_claim=None,
                ),
                copilot_steps=(),
                mode=CopilotMode.GENERAL,
            ),
        ]
        loaded = deserialize_enriched(serialize_enriched(evs))
        assert loaded[0].mode == CopilotMode.PHARMA
        assert loaded[1].mode == CopilotMode.GENERAL


# ---------------------------------------------------------------------------
# write_enriched
# ---------------------------------------------------------------------------


class TestWriteEnriched:
    def test_writes_valid_json(self, tmp_path: Path) -> None:
        path = tmp_path / "enriched.json"
        write_enriched(path, [_make_one()])
        assert path.exists()
        loaded = json.loads(path.read_text(encoding="utf-8"))
        assert isinstance(loaded, list)
        assert len(loaded) == 1

    def test_creates_parents(self, tmp_path: Path) -> None:
        path = tmp_path / "deep" / "dir" / "enriched.json"
        write_enriched(path, [_make_one()])
        assert path.exists()

    def test_deserialises_after_write(self, tmp_path: Path) -> None:
        path = tmp_path / "enriched.json"
        original = [_make_one()]
        write_enriched(path, original)
        loaded = deserialize_enriched(json.loads(path.read_text(encoding="utf-8")))
        assert loaded[0].base.claim.claim_id == original[0].base.claim.claim_id
