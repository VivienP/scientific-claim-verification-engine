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


# ---------------------------------------------------------------------------
# render_markdown — smoke test on a known mix of pass/fail diagnoses
# ---------------------------------------------------------------------------


class TestRenderMarkdown:
    """Smoke-level coverage of the markdown output path. The diagnoser's
    primary user-facing artifact (``reports/phase_a2/ctran_failure_matrix.md``)
    flows through this function, so a regression that breaks the table layout
    or the dominant-failure call-out would silently corrupt the report."""

    @staticmethod
    def _diag_obj(
        *,
        claim_id: str,
        transparent: bool,
        category: str,
        text: str = "claim text",
        evidence_quality: str | None = None,
        retrieval_status: str | None = None,
    ) -> Any:
        return _diag.ClaimDiagnosis(
            claim_id=claim_id,
            claim_text=text,
            transparent=transparent,
            category=category,
            evidence_quality=evidence_quality,
            retrieval_status=retrieval_status,
            source_doi="10.1/x",
            source_found=True,
            source_passages_count=1 if transparent else 0,
        )

    def test_renders_all_required_sections(self) -> None:
        diagnoses = {
            "run_alpha": [
                self._diag_obj(claim_id="c1", transparent=True, category="PASS_quoted_passage"),
                self._diag_obj(
                    claim_id="c2",
                    transparent=False,
                    category="A2b_verifier_did_not_quote",
                    evidence_quality="no_evidence",
                    retrieval_status="passage_found",
                ),
                self._diag_obj(
                    claim_id="c3",
                    transparent=False,
                    category="A2b_verifier_did_not_quote",
                    evidence_quality="no_evidence",
                    retrieval_status="passage_found",
                ),
            ],
            "run_beta": [
                self._diag_obj(
                    claim_id="c4",
                    transparent=False,
                    category="A1_doi_unresolved",
                    evidence_quality=None,
                    retrieval_status=None,
                ),
            ],
        }
        out = _diag.render_markdown(diagnoses)

        # Header + four major sections must all be present.
        assert "# CTran failure diagnostic" in out
        assert "## Per-run summary" in out
        assert "## Rolled-up category counts" in out
        assert "## Dominant failure mode" in out
        assert "## Per-claim failure detail" in out

        # Per-run summary table includes both run names with their CTran %.
        assert "`run_alpha`" in out
        assert "`run_beta`" in out
        assert "33.33%" in out  # 1/3 transparent in run_alpha
        assert "0.00%" in out  # 0/1 transparent in run_beta

        # Dominant failure: A2b appears 2/3 of the time and wins over A1 (1).
        assert "`A2b_verifier_did_not_quote`" in out
        # The A2b recommendation block from _recommendation_for() is included.
        assert "Fix: verifier behaviour, not retrieval." in out

        # Per-claim table surfaces the failing claim_ids (truncated to 12 chars).
        assert "| `c2`" in out
        assert "| `c3`" in out
        assert "| `c4`" in out
        # Passing claims must NOT appear in the failure table.
        assert "| `c1`" not in out

    def test_handles_all_passing_no_dominant_callout(self) -> None:
        # All-pass run: no failures means no "Dominant failure mode" section.
        diagnoses = {
            "perfect_run": [
                self._diag_obj(claim_id="c1", transparent=True, category="PASS_quoted_passage"),
                self._diag_obj(claim_id="c2", transparent=True, category="PASS_abstract_only"),
            ],
        }
        out = _diag.render_markdown(diagnoses)
        assert "## Dominant failure mode" not in out
        assert "100.00%" in out  # CTran rate

    def test_empty_input_does_not_crash(self) -> None:
        # No runs → still produces a valid header so the output isn't blank.
        out = _diag.render_markdown({})
        assert "# CTran failure diagnostic" in out
        assert "## Per-run summary" in out
