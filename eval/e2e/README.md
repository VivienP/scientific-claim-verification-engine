# End-to-end benchmark — Phase A artifact

> Status: scaffold complete; verdicts track populated (`reference_paper_v1_verdicts.json`); recall track (`reference_paper_v1.json`) pending.

This directory holds the hand-annotated ground truth used to measure the full
pipeline (extract → resolve → verify) on real content. It complements the
SciFact oracle eval, which only measures the verifier in isolation.

## Two distinct measurement tracks

Both tracks annotate the same source paper but target different metrics:

- **Verdicts track** (`reference_paper_v1_verdicts.json`): verdict agreement against domain-expert annotations. Each entry carries an `expected_verdict` (`supported | partially_supported | unsupported | not_addressed`), a one-sentence rationale, plus `is_primary_source`, `primary_source_doi`, `expected_full_text_available`, `annotation_notes`, and a `recommended_fix` for unsupported / partially_supported claims. Used to score the end-to-end pipeline on the verdict produced for each claim. Populated 2026-05-07 with 25 claims from the 2023 lactate ISF review (Vivien Perrelle, PKvitality).
- **Recall track** (`reference_paper_v1.json`, scaffolded but not yet populated): extraction recall + DOI resolution accuracy, scored against `gt_claim_id` / `ground_truth_doi` ground truth via `scripts/measure_e2e_recall.py`. Schema documented in `schema.py` and `reference_paper_v1.template.json`.

## Files

- `reference_paper_v1_verdicts.json` — **POPULATED.** 25 hand-labeled verdict annotations from Vivien (verdicts track)
- `schema.py` — typed dataclasses + JSON validator for the recall track
- `reference_paper_v1.template.json` — annotation template for the recall track
- `reference_paper_v1.json` — **NOT YET CREATED.** Recall-track annotation goes here
- `source_texts/` — plain-text exports of source manuscripts (kept out of git for size)
- `results/` — output JSONs from `scripts/measure_e2e_recall.py` (`baseline_pre_fixes.json`, etc.) and from the verdicts track measurement

## How to annotate (step-by-step)

1. **Export the manuscript to plain text.** From Word: File → Save As → Plain Text (.txt) UTF-8. From LaTeX: `pandoc paper.tex -o paper.txt`. Save under `eval/e2e/source_texts/review_2023_lactate_isf.txt`.

2. **Copy the template.** `cp reference_paper_v1.template.json reference_paper_v1.json` then edit.

3. **Fill the `paper` block** with your manuscript metadata.

4. **Annotate each verifiable claim.** Read the manuscript top-to-bottom. For each factual assertion that could in principle be checked (numeric values, mechanism statements, methodological claims, causal claims):
   - Add a JSON entry to `claims`
   - Use sequential `gt_claim_id` ("c001", "c002", ...)
   - Verbatim `claim_text` — full sentence or sub-sentence
   - Mark `claim_origin = "primary"` if the claim is a finding of the review itself (no citation), else `"secondary"`
   - For `secondary` claims, fill `cited_authors`, `cited_year`, and `ground_truth_doi` from your bibliography

5. **Decompose composite claims.** If a sentence asserts three distinct facts ("A, B, and C"), split it into three entries — that matches the FActScore atomic decomposition pattern that Phase C targets.

6. **Delete the `_annotation_guide` block** before saving the final file.

7. **Validate the file.** `python -c "from eval.e2e.schema import load_reference_paper; from pathlib import Path; load_reference_paper(Path('eval/e2e/reference_paper_v1.json'))"` — should print nothing if valid.

## Attribution

The 25 hand-labeled verdict annotations in `reference_paper_v1_verdicts.json` are derived from:

> Perrelle, V. (2023). *Exploring Activity-Induced Lactate Pharmacokinetics: Implications for Minimally-Invasive Monitoring*. De Vinci Innovation Center.

The source paper and its annotations are the original work of Vivien Perrelle. The benchmark dataset under `eval/e2e/` is released under CC BY-NC (the rest of the repository is Apache 2.0; see top-level [LICENSE](../../LICENSE)). If you use this benchmark in published work, please cite the source paper above.

## Time budget

4-6 hours for 60-80 claims on a 15-20 page review. The author of the paper (Vivien) is the epistemic authority on its claims, so this should be ~3x faster than a domain-naive annotator.

## How to run the measurement

Once `reference_paper_v1.json` exists:

```bash
python scripts/measure_e2e_recall.py --paper eval/e2e/reference_paper_v1.json --output eval/e2e/results/baseline_pre_fixes.json
```

The script runs the existing pipeline against the source text and outputs the 5 metrics (extraction_recall, extraction_precision, resolution_accuracy, e2e_coverage_useful, not_addressed_unknown_cause).

This is a **baseline measurement** — no fixes applied. It establishes the starting point against which Phase B+C improvements will be compared.
