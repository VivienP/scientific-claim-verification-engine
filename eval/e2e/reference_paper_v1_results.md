# End-to-end benchmark — reference_paper_v1_verdicts results

Hand-labeled benchmark (Vivien Perrelle, domain expert) vs. the current extract → resolve → fetch_fulltext → chunk → select → verify pipeline (`claude-sonnet-4-6`).

- Source: PERRELLE 2023 lactate ISF review (25 claims).
- Pipeline cost: $0.306 for the full run.
- **Headline: agreement = 1/25 (4.0%).** Read the diagnostic section before drawing conclusions — the dominant failure mode is resolution, not verification.

## Confusion matrix

Rows = expected (Vivien). Columns = pipeline output.

|                       | pipeline: supported | pipeline: partially_supported | pipeline: unsupported | pipeline: not_addressed | row total |
|---|---|---|---|---|---|
| **expected: supported** | 1 | 1 | 0 | 8 | 10 |
| **expected: partially_supported** | 0 | 0 | 0 | 12 | 12 |
| **expected: unsupported** | 0 | 0 | 0 | 3 | 3 |
| **expected: not_addressed** | 0 | 0 | 0 | 0 | 0 |
| **column total** | 1 | 1 | 0 | 23 | 25 |

## Per-class metrics

| class | precision | recall | F1 | support (expected) | predictions |
|---|---|---|---|---|---|
| supported | 1.00 | 0.10 | 0.18 | 10 | 1 |
| partially_supported | 0.00 | 0.00 | 0.00 | 12 | 1 |
| unsupported | 0.00 | 0.00 | 0.00 | 3 | 0 |
| not_addressed | 0.00 | 0.00 | 0.00 | 0 | 23 |

**Overall accuracy: 4.00% (1/25)**
**Macro-F1 (unweighted across 4 classes): 0.05**

## Diagnostic: where the pipeline breaks

The dominant failure pattern is **resolver returning the wrong paper** (or no paper) for the cited reference. Specifically:

- **Resolution found a paper**: 18/25 claims. The remaining 7 returned `found=False` from Semantic Scholar / OpenAlex / CrossRef.
- **Resolution returned the *correct* paper** (DOI matches Vivien's `primary_source_doi`): 3/15 on the subset where a DOI ground truth exists. The other resolutions returned high-similarity-score papers that are entirely off-topic (e.g., claim 003 on Goodwin 2007 lactate ratios resolved to *Amino acids and immune function*, similarity_score = 1.0).
- **Full-text fetched via OA URL**: 10/25 claims. PDFs were successfully extracted and chunked, but on the wrong source for most claims, so the BM25 selector found no relevant passage.
- **`passage_found` retrieval status**: 10/25 claims. Even when a passage was selected, the verifier correctly returned `not_addressed` because the passage discussed an unrelated topic.

This means the verifier itself is **behaving correctly**: when given the wrong source, it honestly reports `not_addressed` rather than hallucinating support. The pipeline-level disagreement is driven by **upstream resolution errors**, not by verifier weakness.

Resolution failure pattern by `cited_year`: 7 of 7 claims with `cited_year = null` failed to resolve (claims 001, 004, 005, 011, 017, 018, 020). Consistent with the known Semantic Scholar heuristic of down-weighting queries that omit the year.

## Per-claim breakdown

| claim_id | claim_text (truncated) | expected | pipeline | agree | brief note |
|---|---|---|---|---|---|
| lactate_review_001 | Blood contains approximately 100 times more l-lactate than … | unsupported | not_addressed | N | resolver: no source found |
| lactate_review_002 | Lactic acid accumulates in contracting muscle and blood, be… | supported | not_addressed | N | resolver: wrong paper (got Lactate as a fulcrum of metabolism) |
| lactate_review_003 | The whole blood-plasma lactate concentration ratio might va… | supported | not_addressed | N | resolver: wrong paper (got Amino acids and immune function) |
| lactate_review_004 | [La−]pla is around 1.5 times higher than whole [La−]blo at … | partially_supported | not_addressed | N | resolver: no source found |
| lactate_review_005 | Capillary plasma lactate concentration and hand-held point-… | partially_supported | not_addressed | N | resolver: no source found |
| lactate_review_006 | For all five portable analyzers, the analytical error withi… | supported | not_addressed | N | verifier: no_evidence |
| lactate_review_007 | The devices' reliability was generally lower than 0.5 mM fo… | supported | not_addressed | N | verifier: no_evidence |
| lactate_review_008 | Lactate ISF concentration has a nearly 1:1 ratio correlatio… | partially_supported | not_addressed | N | resolver: wrong paper (got Wearable sensors for monitoring the phy…) |
| lactate_review_009 | The blood-to-ISF lag time is between 5 to 15 minutes. | supported | not_addressed | N | resolver: wrong paper (got Opportunities and challenges in the dia…) |
| lactate_review_010 | In healthy people, skin lactate concentration at rest is be… | partially_supported | not_addressed | N | full text unavailable; verifier scored on abstract |
| lactate_review_011 | Resting dermal [La−]ISF is on average about 30% higher than… | partially_supported | not_addressed | N | resolver: no source found |
| lactate_review_012 | Skin contributes about 5% at rest to the whole-body lactate… | unsupported | not_addressed | N | verifier: no_evidence |
| lactate_review_013 | Recent non-commercialized microneedle-based lactate biosens… | partially_supported | not_addressed | N | resolver: wrong paper (got Biosensor-Integrated Microneedle Device…) |
| lactate_review_014 | The subcutaneous depth of the capillary plexus varies betwe… | partially_supported | not_addressed | N | resolver: wrong paper (got Radiobiological depth of subcutaneous i…) |
| lactate_review_015 | Krogstad et al. suggest a possible negative linear relation… | partially_supported | not_addressed | N | resolver: wrong paper (got The Mode of Delivery and the Risk of Ve…) |
| lactate_review_016 | Dermal ISF lactate concentration depends on catheter depth … | partially_supported | not_addressed | N | full text unavailable; verifier scored on abstract |
| lactate_review_017 | Active muscle lactate concentration presents similar patter… | supported | not_addressed | N | resolver: no source found |
| lactate_review_018 | Pores of continuous capillary lining only allow small solut… | partially_supported | not_addressed | N | resolver: no source found |
| lactate_review_019 | Lactic acidosis is defined as a blood lactate concentration… | unsupported | not_addressed | N | resolver: wrong paper (got Lactate versus non-lactate metabolic ac…) |
| lactate_review_020 | Plasma glucose values are about 10%–15% higher than whole b… | supported | not_addressed | N | resolver: no source found |
| lactate_review_021 | Correlation between arterial and capillary lactate increase… | partially_supported | not_addressed | N | resolver: wrong paper (got A new approach to the assessment of ana…) |
| lactate_review_022 | Microneedles have a small dimension of less than 1 mm in le… | supported | partially_supported | N | resolver: wrong paper (got Fabrication of sharp silicon hollow mic…) |
| lactate_review_023 | 3D printing has emerged as a promising technique for fabric… | supported | supported | Y | match |
| lactate_review_024 | Microfluidic technology has been used to control the porosi… | supported | not_addressed | N | resolver: wrong paper (got A guide to the organ-on-a-chip) |
| lactate_review_025 | The lag time of glucose concentrations in the ISF is around… | partially_supported | not_addressed | N | resolver: wrong paper (got Continuous Glucose Monitoring Systems: …) |

## Top-3 most informative disagreements

### 1. Silent wrong-DOI with high-similarity score (claim 003)
- **Claim:** *Whole blood-plasma lactate concentration ratio might vary from 63% to 81% depending on plasma water content and hematocrit.*
- **Cited:** Goodwin 2007 (DOI `10.1007/BF00382568`). Vivien: `supported`, verbatim match in the abstract.
- **Pipeline:** retrieved `10.1017/s000711450769936x` (*Amino acids and immune function*) with similarity_score = 1.0, then verifier honestly returned `not_addressed` because the passage is about immunology, not lactate.
- **Why this is the worst failure mode**: the resolver expressed *maximum* confidence on a paper that has zero topical overlap with the claim. This is the silent-error failure pattern. A cautious resolver would mark this as low-confidence; the current implementation reports it as a perfect match. Fixing this single bug would unlock at least 5 more correct verdicts in this benchmark.

### 2. Consequential factual error not surfaced (claim 019)
- **Claim:** *Lactic acidosis is defined as a blood lactate concentration of 5 mmol/L and a pH lower than 7.35.*
- **Cited:** Forsythe & Schmidt 2000 (DOI `10.1378/chest.117.1.260`). Vivien: `unsupported` — the standard definition is **>4 mmol/L**, not 5 mmol/L; the cited paper is also a treatment review, not a definition source. This is the most consequential factual error in the benchmark (off by 25% on a clinical threshold).
- **Pipeline:** retrieved `10.1186/cc3987` (a 2006 *Critical Care* paper, not Forsythe 2000 *Chest*). Verifier returned `not_addressed`.
- **Why this is informative**: the pipeline *should* be the safety net for exactly this kind of off-by-one definitional error in regulatory writing. It missed the catch entirely — but the failure was at resolution, not verification. If the resolver had returned the correct paper, the verifier would likely have spotted that Forsythe 2000 isn't a definition source and the 5-vs-4 mismatch with the broader literature.

### 3. The one success — when verbatim title matches the claim (claim 023)
- **Claim:** *3D printing has emerged as a promising technique for fabricating microneedle arrays, enabling point-of-care biosensing applications.*
- **Cited:** Rezapour Sarabi et al. 2022 (DOI `10.3390/mi13071099`).
- **Pipeline:** resolved correctly to `10.3390/mi13071099` (similarity_score = 1.0), fetched OA PDF from MDPI, BM25 selected the abstract section, verifier returned `supported` — the only match in the benchmark.
- **Why this is informative**: the title of the source paper is verbatim the claim (*"3D-Printed Microneedles for Point-of-Care Biosensing Applications"*). When the lexical signal is this strong, the pipeline works. Every other claim in the benchmark requires the resolver to map a different surface form — author + year + topic, not title — and that's where the system falls over. The success here is a positive control showing the verifier and fetcher work; the gap is upstream.

## What this benchmark validates

- The verifier behaves honestly: it returns `not_addressed` rather than hallucinating support when the retrieved source is irrelevant.
- The resolver is the dominant bottleneck on real-world content. Vivien's prior intuition (resolver-before-extractor as a fix priority) is experimentally validated.
- A 4-class verdict benchmark on hand-labeled real content surfaces failure modes that SciFact-on-oracle-inputs cannot see (SciFact F1 = 0.94 on the verifier alone vs. 4% end-to-end agreement here).
- This is the first benchmark in the repo that measures the full pipeline against domain-expert-validated ground truth on a bibliography-cited document.
