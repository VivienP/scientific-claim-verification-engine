"""AAR scorecard — Provenance Coverage / Soundness, Claim Transparency, Audit Efficiency.

Implements the four-metric standard introduced in the Audit Agent Report
literature (arXiv 2602.13855) for evaluating audit tools that operate
over research-agent outputs. The metrics are computed *post-hoc* from a
run directory (``reports/runs/{report_id}/``) so:

  * any pipeline run is immediately scoreable without re-execution;
  * the metrics double as regression alarms (a refactor that drops a
    ProvenanceStep emission will visibly lower PCov on the next run);
  * the same metric definitions can score competitor tools whose output
    can be normalised to the report.json + provenance.jsonl shape.

Metrics:

  PCov  — Provenance Coverage. Fraction of claims that have at least
          one ProvenanceStep linked to them. Targets the audit
          accountability question: *can every verdict be traced?*
  PSnd  — Provenance Soundness. Fraction of ProvenanceStep records
          whose input_hash and output_hash are non-empty hex digests.
          Targets *can the trace be cryptographically verified?*
  CTran — Claim Transparency. Fraction of claims whose verdict is
          accompanied by *quoted source evidence* (source_passages
          non-empty OR evidence_quality in a transparent class).
          Targets *can the human auditor see the source content?*
  AEff  — Audit Efficiency. Claims verified per US dollar of LLM cost.
          Targets *is the audit affordable at scale?*

The metrics are deliberately complementary: PCov + PSnd capture the
plumbing, CTran captures the user-facing transparency, AEff captures
the cost trade-off. A tool can score 1.0 on PCov / PSnd by emitting
empty steps; it cannot score 1.0 on CTran without exposing source
quotes, nor on AEff without controlling cost.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

_HEX_RE = re.compile(r"^[0-9a-f]{16,}$")


@dataclass(frozen=True)
class AARScorecard:
    """The four AAR metrics plus the raw counts used to compute them.

    Frozen so the artifact can be checked into eval/results/ as a
    snapshot. Field order matches the AAR paper for downstream tooling.
    """

    pcov: float
    psnd: float
    ctran: float
    aeff: float
    n_claims: int
    n_steps: int
    n_claims_with_provenance: int
    n_steps_with_valid_hashes: int
    n_claims_with_quoted_evidence: int
    total_cost_usd: float


def _is_valid_hash(value: object) -> bool:
    """A non-empty hex digest of length >= 16 (sha256 hex is 64)."""
    return isinstance(value, str) and bool(_HEX_RE.match(value))


def _claim_is_transparent(verification: dict[str, Any]) -> bool:
    """A claim's verdict is transparent when the auditor can see source evidence.

    Two paths qualify:
      1. ``source_passages`` is a non-empty list of verbatim quotes —
         the strongest signal.
      2. ``evidence_quality`` is in {abstract_only, quoted_passage,
         title_only} — the source was at least retrieved and the
         verifier saw something concrete.

    citing_paper_context is *not* counted as transparent because the
    cited source itself was not seen — the verdict is internal-
    consistency only, which the rule explicitly caps at
    partially_supported.
    """
    passages = verification.get("source_passages", [])
    if isinstance(passages, list) and passages:
        return True
    evidence_quality = verification.get("evidence_quality")
    return evidence_quality in {"abstract_only", "quoted_passage", "title_only"}


def compute_aar(
    report: dict[str, Any],
    provenance_steps: list[dict[str, Any]],
) -> AARScorecard:
    """Compute the four AAR metrics from a report + provenance steps.

    ``report`` follows the schema written by :func:`src.report.build_report`
    (a dict with ``claims`` and ``summary`` keys). ``provenance_steps`` is
    a list of dicts (one per JSON line in ``provenance.jsonl``).

    Returns an :class:`AARScorecard`. Raises no exceptions — degenerate
    cases (zero claims, zero cost) yield 0.0 / inf metrics rather than
    raising, so the function is safe to call on partial / failed runs.
    """
    claims = report.get("claims", [])
    n_claims = len(claims)
    n_steps = len(provenance_steps)

    claim_ids = {str(c.get("claim_id")) for c in claims if c.get("claim_id")}
    claim_ids_with_steps = {
        str(s.get("claim_id")) for s in provenance_steps if s.get("claim_id") in claim_ids
    }
    n_claims_with_provenance = len(claim_ids_with_steps)

    n_steps_with_valid_hashes = sum(
        1
        for s in provenance_steps
        if _is_valid_hash(s.get("input_hash")) and _is_valid_hash(s.get("output_hash"))
    )

    n_claims_with_quoted_evidence = sum(
        1 for c in claims if _claim_is_transparent(c.get("verification", {}))
    )

    total_cost = float(report.get("summary", {}).get("total_cost_usd", 0.0))

    pcov = n_claims_with_provenance / n_claims if n_claims else 0.0
    psnd = n_steps_with_valid_hashes / n_steps if n_steps else 0.0
    ctran = n_claims_with_quoted_evidence / n_claims if n_claims else 0.0
    aeff = (n_claims / total_cost) if total_cost > 0 else float("inf")

    return AARScorecard(
        pcov=pcov,
        psnd=psnd,
        ctran=ctran,
        aeff=aeff,
        n_claims=n_claims,
        n_steps=n_steps,
        n_claims_with_provenance=n_claims_with_provenance,
        n_steps_with_valid_hashes=n_steps_with_valid_hashes,
        n_claims_with_quoted_evidence=n_claims_with_quoted_evidence,
        total_cost_usd=total_cost,
    )


def load_run(run_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Read ``report.json`` and ``provenance.jsonl`` from a run directory.

    Used by :func:`compute_aar_for_run` and the CLI scorecard script.
    """
    report_path = run_dir / "report.json"
    provenance_path = run_dir / "provenance.jsonl"
    if not report_path.exists():
        raise FileNotFoundError(f"report.json not found in {run_dir}")
    if not provenance_path.exists():
        raise FileNotFoundError(f"provenance.jsonl not found in {run_dir}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    steps: list[dict[str, Any]] = []
    for line in provenance_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            steps.append(json.loads(line))
    return report, steps


def compute_aar_for_run(run_dir: Path) -> AARScorecard:
    """Convenience wrapper: load + compute_aar for a run directory."""
    report, steps = load_run(run_dir)
    return compute_aar(report, steps)


def render_scorecard_markdown(card: AARScorecard) -> str:
    """Render a single scorecard as a one-liner-per-metric markdown block."""
    aeff_repr = f"{card.aeff:.2f}" if card.aeff != float("inf") else "+inf"
    return (
        f"| Metric | Value | Detail |\n"
        f"|---|---|---|\n"
        f"| PCov  | {card.pcov:.2%} | "
        f"{card.n_claims_with_provenance}/{card.n_claims} claims with provenance |\n"
        f"| PSnd  | {card.psnd:.2%} | "
        f"{card.n_steps_with_valid_hashes}/{card.n_steps} steps with valid hashes |\n"
        f"| CTran | {card.ctran:.2%} | "
        f"{card.n_claims_with_quoted_evidence}/{card.n_claims} claims with quoted evidence |\n"
        f"| AEff  | {aeff_repr} | claims/USD over ${card.total_cost_usd:.4f} total |\n"
    )


__all__ = [
    "AARScorecard",
    "compute_aar",
    "compute_aar_for_run",
    "load_run",
    "render_scorecard_markdown",
]


# Re-export asdict for downstream callers that want to JSON-serialize.
def scorecard_to_dict(card: AARScorecard) -> dict[str, Any]:
    return asdict(card)
