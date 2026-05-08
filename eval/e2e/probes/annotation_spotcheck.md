# Annotation Integrity Spot-Check

Sample seed: `20260507`. Selected claims: `lactate_review_001`, `lactate_review_022`, `lactate_review_018`, `lactate_review_011`, `lactate_review_016`.

| claim_id | annotation | spot-check result | basis |
|---|---|---|---|
| lactate_review_001 | unsupported | Agree with the narrow annotation that the cited sources do not establish the L:D lactate ratio; the factual statement itself may be directionally plausible. | The manuscript cites `[3,39]`; Goodwin 2007 is a blood-lactate review and the source export does not show D-lactate evidence. |
| lactate_review_022 | supported | Partially agree. The cited microneedle review supports sub-millimeter/hundreds-of-micrometers dimensions, but the claim also says they avoid deep subcutaneous tissue, which is a stronger anatomical assertion. | Web spot-check for [DOI `10.1002/adhm.201500450`](https://doi.org/10.1002/adhm.201500450); related PubMed review says typical microneedles are `0.1-1 mm`. |
| lactate_review_018 | partially_supported | Agree. Ono 2017 is open access and supports vessel permeability/size-barrier framing, but the manuscript claim simplifies it into a small-solute rule. | [DOI `10.1186/s41232-017-0042-9`](https://doi.org/10.1186/s41232-017-0042-9), BMC/Springer full-text snippet. |
| lactate_review_011 | partially_supported | Agree. Birklein 2000 supports elevated skin lactate in controls/patients and gives a control value, but it does not by itself establish the full 30% dermal-ISF-over-arterial-plasma statement. | [DOI `10.1212/WNL.55.8.1213`](https://doi.org/10.1212/WNL.55.8.1213), PubMed abstract. |
| lactate_review_016 | partially_supported | Agree with caution. Jansson 1996 supports dermal skin microdialysis and lactate release; the depth-dependence claim appears to require full-text/figure context not available to the current pipeline. | [DOI `10.1152/ajpendo.1996.271.1.E138`](https://doi.org/10.1152/ajpendo.1996.271.1.E138), PubMed abstract plus manuscript Figure 0.19 caption. |

No annotation was clearly invalid. The main integrity issue is not label quality; it is identifier quality. The most concrete defect is `lactate_review_003`, where the benchmark DOI field points to [`10.1007/BF00382568`](https://doi.org/10.1007/BF00382568), a geology paper, while the cited Goodwin 2007 article is [`10.1177/193229680700100414`](https://doi.org/10.1177/193229680700100414).
