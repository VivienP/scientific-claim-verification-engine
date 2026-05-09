# Related Work

Full survey of related systems, benchmarks, and evaluation standards. Summary in [README.md](../README.md#related-work).

## Closest neighbours

### Valsci (Edelman & Skolnick 2025)

[GitHub](https://github.com/bricee98/Valsci) · [BMC Bioinformatics](https://link.springer.com/article/10.1186/s12859-025-06159-4) · arXiv 2504.xxxxx

Open-source self-hostable verifier grounded in Semantic Scholar, optimized for large-batch literature corpus verification with bibliometric scoring (F1 = 0.761 on SciFact with GPT-4o).

Structural differences vs. this engine:

1. **Use case** — Valsci verifies oracle claims against a literature corpus; this engine audits free-form AI-agent outputs claim-by-claim against the specific cited source.
2. **Resolution chain** — Valsci uses single-source Semantic Scholar; this engine uses CrossRef → OpenAlex → PubMed → Europe PMC → Unpaywall with bibliography-aware DOI routing.
3. **Determinism boundary** — numeric comparison steps in this engine are pure Python (no LLM); Valsci does not separate deterministic from probabilistic verification.
4. **AAR metrics** — Valsci does not publish AAR (PCov/PSnd/CTran/AEff) scores. This engine adopts AAR as the primary audit scorecard; see [`src/aar.py`](../src/aar.py).

The two systems are complementary: Valsci excels at batch corpus auditing; this engine targets per-claim, multi-source, provenance-tracked auditing of AI-generated content.

### CiteAudit (arXiv 2602.23452)

5-agent pipeline (Extractor → Memory → Web → Scholar → Judge), F1 = 0.838 on real citations.

CiteAudit and this engine are complementary by design:

- **CiteAudit** — metadata consistency: does the cited paper exist? does its title/year/authors match what is claimed?
- **This engine** — semantic entailment: does the cited source's actual content support the specific assertion made?

A production audit pipeline would run both.

## Benchmarks and evaluation standards

### SciClaimEval

[sciclaimeval.github.io](https://sciclaimeval.github.io/) — cross-modal claim ↔ table/figure benchmark (1,664 samples, 180 papers). SOTA: o4-mini at 68.2% pair-accuracy. Not a direct competitor (cross-modal, not text→text), but its perturbation methodology is informative for designing adversarial test cases.

### AAR standard (arXiv 2602.13855)

Defines four metrics for evaluating audit tools over research agents:

- **PCov** (Provenance Coverage) — fraction of claims with a traceable evidence source
- **PSnd** (Provenance Soundness) — fraction of sourced claims where the source actually supports the claim
- **CTran** (Claim Transparency) — fraction of claims with explicit verdict rationale
- **AEff** (Audit Efficiency) — claims audited per USD

Adopted in this engine from S4 onward. Current Valsci-paper run: PCov 100% / PSnd 100% / CTran 36% / AEff 45.4. See `python scripts/aar_scorecard.py reports/runs/<id>`.

### SciClaimHunt_Num (arXiv 2502.10003)

Closest public asset to the lactate-ISF benchmark — numeric scientific claim verification with structured ground truth. Relevant for future numeric-check evaluation.

### MuSciClaims

Confirms the 3-class SUPPORT / NEUTRAL / CONTRADICT rubric used here is the public formalization of the absence-of-support-vs-not-addressed distinction. The `not_addressed` verdict in this engine maps to NEUTRAL in MuSciClaims notation.

### AFEV (arXiv 2506.07446)

Introduces adaptive atomic decomposition for compound claims — relevant to the multi-source aggregation ceiling hit at S2 (16/25 verdict agreement on lactate-ISF). A future direction for decomposing multi-assertion claims before per-source verification.

## Where this engine is uniquely positioned

1. **Multi-source aggregation** — when a claim cites `[81-83]`, the pipeline resolves all three references and aggregates verdicts with explicit precedence rules, instead of averaging or picking the strongest.
2. **Paywall recovery beyond Unpaywall + Europe PMC** — bibliography-aware DOI routing recovers references that would otherwise return not-found.
3. **Partial-support reasoning over numeric ranges** — the verifier rubric (Clauses A/B/C/D) explicitly handles range/uncertainty inclusion, trajectory-vs-snapshot, and numeric-verbatim-absence cases that flatten in 3-class systems.
4. **Provenance-first architecture** — every step emits a `ProvenanceStep` with input/output hash, model ID, token counts, and cache hit status, enabling full audit trail replay.
