# Negative-control fixtures for `extract_claims()`

These fixtures pin the extractor's rejection behavior on non-claim text that legitimately appears inside scientific publications. Every fixture except `results_paragraph_positive_control.txt` must extract **zero** claims; the positive control must extract **exactly two**.

The suite exists because the extractor is LLM-driven with no deterministic pre-filter. A prompt revision, a model swap, or a chunking change can silently regress rejection behavior without any other test catching it. These fixtures lock the behavior in CI.

## Files

| File | Source domain | Expected claims |
|---|---|---|
| `references_section_only.txt` | Bibliography from a real SciFact dev paper (~15 numbered refs with author/year/title) | 0 |
| `methods_only.txt` | Methods section of a Cochrane systematic review (no Results) | 0 |
| `figure_table_captions.txt` | 8 captions from real published papers (e.g. "Figure 1. Study flow diagram (n=120)") | 0 |
| `headings_only.txt` | 20 section headings from real IMRAD papers, including numeric ones like "3.2 PFS at 12 months in the treatment arm" | 0 |
| `acknowledgments_funding.txt` | Acknowledgments + Funding sections from 3 real papers | 0 |
| `results_paragraph_positive_control.txt` | Single Results paragraph from a SciFact paper | 2 (calibration) |

## Scope

All fixture content is sourced from real scientific publications (SciFact dev set, Cochrane reviews, open-access journals). Regulatory guidance, SSCPs, PSURs, IFUs, and other meta-documents are **out of scope** — the extractor's input contract is scientific publications with citation anchors.

## Mocking convention

Unit tests in `tests/unit/test_extract_negative_controls.py` mock the Anthropic client via `unittest.mock.patch` per the [offline-tests skill](../../../../.claude/skills/offline-tests/SKILL.md). Live API calls are never made from unit tests. Canned LLM responses are real one-off captures stored under `anthropic_responses/`.
