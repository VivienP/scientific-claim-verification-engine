"""Schema for end-to-end benchmark ground truth annotations.

Annotations live in JSON files like `eval/e2e/reference_paper_v1.json`. The
schema is loaded and validated at the start of `scripts/measure_e2e_recall.py`
so structural errors fail fast with a clear message rather than mid-run.

The schema deliberately keeps document-level metadata separate from per-claim
annotations so future references (v2, v3) can be added without breaking
existing files.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, get_args

ClaimType = Literal["factual_numeric", "factual_qualitative", "methodological", "causal"]
ClaimOrigin = Literal["primary", "secondary"]
SectionLabel = Literal["introduction", "methods", "results", "discussion", "other"]

SCHEMA_VERSION = "1.0"


@dataclass(frozen=True)
class GroundTruthClaim:
    """One annotated claim from the reference paper.

    `ground_truth_doi` is the DOI Vivien intended to cite when writing the
    claim. It is the comparator for `resolution_accuracy`.

    `claim_origin == "primary"` means the claim is a finding of the paper
    itself (no external citation expected) — these are skipped from
    resolution scoring but still counted in extraction recall.
    """

    gt_claim_id: str
    claim_text: str
    section: SectionLabel
    claim_type: ClaimType
    claim_origin: ClaimOrigin
    cited_authors: list[str]
    cited_year: int | None
    ground_truth_doi: str | None
    ground_truth_title: str | None


@dataclass(frozen=True)
class ReferencePaper:
    """A reference paper with all its annotated claims."""

    schema_version: str
    paper_title: str
    paper_authors: list[str]
    paper_year: int
    source_text_path: str
    claims: list[GroundTruthClaim]


def _validate_literal(value: str, literal_type: object, field_name: str) -> None:
    allowed = get_args(literal_type)
    if value not in allowed:
        raise ValueError(f"Invalid {field_name}: {value!r} (allowed: {sorted(allowed)})")


def _parse_claim(raw: dict[str, object], index: int) -> GroundTruthClaim:
    required = {
        "gt_claim_id",
        "claim_text",
        "section",
        "claim_type",
        "claim_origin",
        "cited_authors",
        "cited_year",
        "ground_truth_doi",
        "ground_truth_title",
    }
    missing = required - raw.keys()
    if missing:
        raise ValueError(f"claims[{index}]: missing fields {sorted(missing)}")

    section = str(raw["section"])
    claim_type = str(raw["claim_type"])
    claim_origin = str(raw["claim_origin"])
    _validate_literal(section, SectionLabel, f"claims[{index}].section")
    _validate_literal(claim_type, ClaimType, f"claims[{index}].claim_type")
    _validate_literal(claim_origin, ClaimOrigin, f"claims[{index}].claim_origin")

    cited_authors_raw = raw["cited_authors"]
    if not isinstance(cited_authors_raw, list):
        raise ValueError(f"claims[{index}].cited_authors must be a list")
    cited_authors = [str(a) for a in cited_authors_raw]

    cited_year = raw["cited_year"]
    if cited_year is not None and not isinstance(cited_year, int):
        raise ValueError(f"claims[{index}].cited_year must be int or null")

    return GroundTruthClaim(
        gt_claim_id=str(raw["gt_claim_id"]),
        claim_text=str(raw["claim_text"]),
        section=section,  # type: ignore[arg-type]
        claim_type=claim_type,  # type: ignore[arg-type]
        claim_origin=claim_origin,  # type: ignore[arg-type]
        cited_authors=cited_authors,
        cited_year=cited_year,
        ground_truth_doi=None if raw["ground_truth_doi"] is None else str(raw["ground_truth_doi"]),
        ground_truth_title=(
            None if raw["ground_truth_title"] is None else str(raw["ground_truth_title"])
        ),
    )


def load_reference_paper(path: Path) -> ReferencePaper:
    """Load and validate a reference paper annotation file.

    Raises ValueError on any schema violation.
    """
    raw = json.loads(path.read_text(encoding="utf-8"))

    schema_version = str(raw.get("schema_version", ""))
    if schema_version != SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported schema_version {schema_version!r} (expected {SCHEMA_VERSION!r})"
        )

    paper = raw.get("paper")
    if not isinstance(paper, dict):
        raise ValueError("Missing or invalid 'paper' object")
    for required_field in ("title", "authors", "year", "source_text_path"):
        if required_field not in paper:
            raise ValueError(f"paper.{required_field} is required")

    claims_raw = raw.get("claims")
    if not isinstance(claims_raw, list):
        raise ValueError("'claims' must be a list")

    claims = [_parse_claim(c, i) for i, c in enumerate(claims_raw)]

    seen_ids = set()
    for c in claims:
        if c.gt_claim_id in seen_ids:
            raise ValueError(f"Duplicate gt_claim_id: {c.gt_claim_id}")
        seen_ids.add(c.gt_claim_id)

    paper_authors_raw = paper["authors"]
    if not isinstance(paper_authors_raw, list):
        raise ValueError("paper.authors must be a list")
    paper_authors = [str(a) for a in paper_authors_raw]

    return ReferencePaper(
        schema_version=schema_version,
        paper_title=str(paper["title"]),
        paper_authors=paper_authors,
        paper_year=int(paper["year"]),
        source_text_path=str(paper["source_text_path"]),
        claims=claims,
    )
