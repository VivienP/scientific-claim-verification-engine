"""Unit tests for scripts/diagnose_ctran.py — fully offline.

The diagnoser is the gate that decides which Phase A.2 fix to build first;
its categorisation must stay in sync with src.aar._claim_is_transparent or
the diagnosis will silently mis-classify failures.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

# Load scripts/diagnose_ctran.py as a module (the scripts/ folder is not a
# package, so a normal import will not work).
_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "diagnose_ctran.py"
_spec = importlib.util.spec_from_file_location("_diagnose_ctran", _SCRIPT)
assert _spec and _spec.loader
_diag = importlib.util.module_from_spec(_spec)
sys.modules["_diagnose_ctran"] = _diag
_spec.loader.exec_module(_diag)


def _claim(
    *,
    claim_id: str = "c1",
    text: str = "claim text",
    source_found: bool = True,
    doi: str | None = "10.1/x",
    evidence_quality: str | None = None,
    retrieval_status: str | None = None,
    source_passages: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "claim_id": claim_id,
        "claim_text": text,
        "source": {"found": source_found, "doi": doi},
        "verification": {
            "evidence_quality": evidence_quality,
            "retrieval_status": retrieval_status,
            "source_passages": source_passages or [],
        },
    }


# ---------------------------------------------------------------------------
# Categorisation contract — failure modes
# ---------------------------------------------------------------------------


class TestCategoriseFailures:
    def test_doi_unresolved(self) -> None:
        c = _claim(
            source_found=False,
            doi=None,
            evidence_quality="no_evidence",
            retrieval_status="fulltext_unavailable",
        )
        transparent, cat = _diag._categorise(c)
        assert transparent is False
        assert cat == "A1_doi_unresolved"

    def test_retrieval_failed(self) -> None:
        c = _claim(evidence_quality="no_evidence", retrieval_status="fulltext_unavailable")
        transparent, cat = _diag._categorise(c)
        assert transparent is False
        assert cat == "A2a_retrieval_failed"

    def test_verifier_did_not_quote(self) -> None:
        # Retrieval succeeded (passages were shown to the LLM) but evidence
        # quality is no_evidence with empty source_passages — the bucket
        # Phase A.2 was built to drain.
        c = _claim(evidence_quality="no_evidence", retrieval_status="passage_found")
        transparent, cat = _diag._categorise(c)
        assert transparent is False
        assert cat == "A2b_verifier_did_not_quote"

    def test_citing_context_only(self) -> None:
        c = _claim(evidence_quality="citing_paper_context", retrieval_status="fulltext_unavailable")
        transparent, cat = _diag._categorise(c)
        assert transparent is False
        assert cat == "A3_citing_context_only"


# ---------------------------------------------------------------------------
# Categorisation contract — passes
# ---------------------------------------------------------------------------


class TestCategorisePasses:
    def test_quoted_passages_pass_regardless_of_evidence_quality(self) -> None:
        # Non-empty source_passages take precedence (this is the AAR rule).
        c = _claim(
            source_passages=["a quoted sentence"],
            evidence_quality="no_evidence",  # contradictory, but passages win
        )
        transparent, cat = _diag._categorise(c)
        assert transparent is True
        assert cat == "PASS_quoted_passage"

    def test_passages_searched_no_quote_is_now_transparent(self) -> None:
        """The Phase A.2 fix introduced this evidence_quality. The diagnoser
        must agree with src.aar._claim_is_transparent that it counts as
        transparent — otherwise the diagnoser would keep flagging fixed
        claims as failures and we'd think the fix did nothing."""
        c = _claim(
            evidence_quality="passages_searched_no_quote",
            retrieval_status="passage_found",
            source_passages=[],  # empty — relies on the evidence_quality alone
        )
        transparent, cat = _diag._categorise(c)
        assert transparent is True
        assert cat == "PASS_passages_searched_no_quote"

    def test_abstract_only_passes(self) -> None:
        c = _claim(evidence_quality="abstract_only")
        transparent, _ = _diag._categorise(c)
        assert transparent is True

    def test_title_only_passes(self) -> None:
        c = _claim(evidence_quality="title_only")
        transparent, _ = _diag._categorise(c)
        assert transparent is True


# ---------------------------------------------------------------------------
# Sync check with src.aar — the most important invariant in this file
# ---------------------------------------------------------------------------


class TestDiagnoserMatchesAAR:
    """If these diverge, the failure matrix lies about CTran totals."""

    @pytest.mark.parametrize(
        "verification",
        [
            {"source_passages": ["q"], "evidence_quality": "quoted_passage"},
            {"source_passages": [], "evidence_quality": "abstract_only"},
            {"source_passages": [], "evidence_quality": "title_only"},
            {"source_passages": [], "evidence_quality": "quoted_passage"},
            {"source_passages": [], "evidence_quality": "passages_searched_no_quote"},
            {"source_passages": [], "evidence_quality": "no_evidence"},
            {"source_passages": [], "evidence_quality": "citing_paper_context"},
            {"source_passages": [], "evidence_quality": None},
        ],
    )
    def test_transparency_agrees_with_aar(self, verification: dict[str, Any]) -> None:
        from src.aar import _claim_is_transparent

        claim_dict = {
            "claim_id": "c1",
            "claim_text": "x",
            "source": {"found": True},
            "verification": verification,
        }
        diag_transparent, _ = _diag._categorise(claim_dict)
        aar_transparent = _claim_is_transparent(verification)
        assert diag_transparent == aar_transparent, (
            f"Mismatch on {verification!r}: diagnoser={diag_transparent}, AAR={aar_transparent}"
        )


# ---------------------------------------------------------------------------
# diagnose_run end-to-end on a tmp_path report.json
# ---------------------------------------------------------------------------


class TestDiagnoseRun:
    def test_reads_report_and_returns_one_diagnosis_per_claim(self, tmp_path: Path) -> None:
        import json

        run_dir = tmp_path / "run1"
        run_dir.mkdir()
        report = {
            "claims": [
                _claim(claim_id="c1", source_passages=["yes"]),  # PASS
                _claim(
                    claim_id="c2",
                    evidence_quality="no_evidence",
                    retrieval_status="fulltext_unavailable",
                ),  # A2a
                _claim(
                    claim_id="c3", evidence_quality="no_evidence", retrieval_status="passage_found"
                ),  # A2b
            ]
        }
        (run_dir / "report.json").write_text(json.dumps(report), encoding="utf-8")

        diagnoses = _diag.diagnose_run(run_dir)
        assert [d.claim_id for d in diagnoses] == ["c1", "c2", "c3"]
        assert [d.transparent for d in diagnoses] == [True, False, False]
        assert [d.category for d in diagnoses] == [
            "PASS_quoted_passage",
            "A2a_retrieval_failed",
            "A2b_verifier_did_not_quote",
        ]

    def test_missing_report_raises_clearly(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match=r"report\.json not found"):
            _diag.diagnose_run(tmp_path / "missing")
