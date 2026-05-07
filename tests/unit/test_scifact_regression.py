"""Tests for the SciFact baseline regression check (S1-P0)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.check_scifact_regression import check_regression, main


def _write_metrics(tmp: Path, *, f1: float, macro_f1: float) -> Path:
    payload = {
        "split": "dev",
        "model_id": "claude-sonnet-4-6",
        "metrics": {
            "f1": f1,
            "macro_f1": macro_f1,
            "precision": f1,
            "recall": f1,
            "per_class": {},
        },
        "n_claims": 50,
    }
    tmp.write_text(json.dumps(payload), encoding="utf-8")
    return tmp


def test_passes_when_candidate_within_tolerance(tmp_path: Path) -> None:
    baseline = _write_metrics(tmp_path / "baseline.json", f1=0.92, macro_f1=0.91)
    candidate = _write_metrics(tmp_path / "candidate.json", f1=0.915, macro_f1=0.905)

    passes, reason = check_regression(baseline, candidate)

    assert passes is True
    assert "within tolerance" in reason


def test_passes_when_candidate_better_than_baseline(tmp_path: Path) -> None:
    baseline = _write_metrics(tmp_path / "baseline.json", f1=0.92, macro_f1=0.91)
    candidate = _write_metrics(tmp_path / "candidate.json", f1=0.95, macro_f1=0.94)

    passes, _reason = check_regression(baseline, candidate)

    assert passes is True


def test_fails_when_f1_drops_beyond_tolerance(tmp_path: Path) -> None:
    baseline = _write_metrics(tmp_path / "baseline.json", f1=0.92, macro_f1=0.91)
    candidate = _write_metrics(tmp_path / "candidate.json", f1=0.89, macro_f1=0.905)

    passes, reason = check_regression(baseline, candidate)

    assert passes is False
    assert "f1" in reason.lower()
    assert "0.89" in reason or "0.890" in reason


def test_fails_when_macro_f1_drops_beyond_tolerance(tmp_path: Path) -> None:
    baseline = _write_metrics(tmp_path / "baseline.json", f1=0.92, macro_f1=0.91)
    candidate = _write_metrics(tmp_path / "candidate.json", f1=0.918, macro_f1=0.88)

    passes, reason = check_regression(baseline, candidate)

    assert passes is False
    assert "macro_f1" in reason.lower()


def test_custom_tolerance_can_be_strict(tmp_path: Path) -> None:
    baseline = _write_metrics(tmp_path / "baseline.json", f1=0.92, macro_f1=0.91)
    candidate = _write_metrics(tmp_path / "candidate.json", f1=0.915, macro_f1=0.905)

    passes, _reason = check_regression(
        baseline, candidate, f1_tolerance=0.001, macro_f1_tolerance=0.001
    )

    assert passes is False


def test_main_returns_exit_code_zero_on_pass(tmp_path: Path) -> None:
    baseline = _write_metrics(tmp_path / "baseline.json", f1=0.92, macro_f1=0.91)
    candidate = _write_metrics(tmp_path / "candidate.json", f1=0.918, macro_f1=0.908)

    exit_code = main([str(baseline), str(candidate)])

    assert exit_code == 0


def test_main_returns_exit_code_one_on_fail(tmp_path: Path) -> None:
    baseline = _write_metrics(tmp_path / "baseline.json", f1=0.92, macro_f1=0.91)
    candidate = _write_metrics(tmp_path / "candidate.json", f1=0.85, macro_f1=0.84)

    exit_code = main([str(baseline), str(candidate)])

    assert exit_code == 1


def test_check_regression_raises_on_missing_metric_field(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps({"metrics": {}}), encoding="utf-8")
    candidate = _write_metrics(tmp_path / "candidate.json", f1=0.9, macro_f1=0.9)

    with pytest.raises(KeyError):
        check_regression(baseline, candidate)
