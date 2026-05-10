# Output Schema — per-claim verification record

Every run produces two artifacts under `reports/runs/{run_id}/`:

- **`report.json`** — one entry per claim in `claims[]`, plus a `summary` block.
- **`provenance.jsonl`** — append-only audit log; one JSON object per pipeline step.

This document describes the on-disk shape of one `report.json["claims"][i]` entry, the `provenance.jsonl` record, and the **Copilot enrichment** that wraps a base verification when the run is launched in Copilot mode.

For the dataclass definitions (frozen, type-checked), see [src/models.py](../src/models.py) and [src/copilot/models.py](../src/copilot/models.py).

---

## 1. Base claim entry (V1 mode)

```jsonc
{
  "claim_id": "uuid",                   // stable id for cross-referencing provenance
  "claim_text": "string",               // verbatim or minimally paraphrased from input
  "claim_type": "factual_numeric"
              | "factual_qualitative"
              | "methodological"
              | "causal",
  "cited_authors": ["string"],          // last names; empty when only [N] markers
  "cited_year": 1234,                   // int or null
  "citation_markers": [81, 82, 83],     // numbered references, range-expanded; [] when absent

  "source": {
    "found": true,                      // CrossRef/OpenAlex/PubMed match passed similarity gate
    "doi": "10.xxxx/yyyy",              // null when unresolved
    "title": "string",                  // null when found=false
    "abstract": "string",               // null when paywalled & no OA copy
    "similarity_score": 0.92,           // title/author cosine match used for gating
    "title_match_score": 0.97,          // optional, used in title-only mode
    "resolution_low_confidence": false, // true => verifier downgrades verdict cap
    "oa_url": "https://...",            // PDF URL from OpenAlex / Unpaywall (or null)
    "pmcid": "PMC1234567",              // when Europe PMC has fulltext (or null)
    "retraction_status": false          // CrossRef "update-to" => true
  },

  "verification": {
    "status": "supported"
            | "partially_supported"
            | "unsupported"
            | "not_addressed",

    "explanation": "string",            // LLM-generated rationale, audited by humans
    "confidence": 0.85,                 // LLM self-report — UNRELIABLE; use the deterministic flags below

    "source_passages": ["string"],      // verbatim quotes the LLM cited; or BM25-selected
                                        // passages when audit-trail fallback fired (see §3)

    "source_section": "introduction"
                    | "methods"
                    | "results"
                    | "discussion"
                    | "other"
                    | null,

    "fulltext_available": true,
    "verification_depth": "fulltext"
                        | "abstract"
                        | "title_only"
                        | "citing_paper_context",

    "retrieval_status": "passage_found"
                      | "no_passage_found"
                      | "fulltext_unavailable",

    "evidence_quality": "quoted_passage"
                      | "passages_searched_no_quote"
                      | "abstract_only"
                      | "title_only"
                      | "citing_paper_context"
                      | "no_evidence",

    "retraction_status": false,         // mirrors source.retraction_status
    "numeric_check": null               // see §4
  }
}
```

### Trust signals — what to read

| Field | What it tells you | Why trust it |
|---|---|---|
| `verification.status` | The verdict | LLM judgment; cross-checked by `evidence_quality` |
| `evidence_quality` | What the verifier actually saw | Deterministic — derived from retrieval, not from LLM output |
| `retrieval_status` | Did the pipeline reach the source body? | Deterministic — set by `fetch_fulltext` chain |
| `verification_depth` | Which verifier mode produced the verdict | Deterministic — set by routing logic |
| `source_passages` | Quotes (or audit-trail BM25 chunks) | Always populated when `evidence_quality != no_evidence` after Phase A.2 |
| `confidence` | LLM-reported number 0.0–1.0 | **Unreliable** — model self-report; do not gate decisions on it |

---

## 2. Verdict × evidence_quality interaction

The verdict by itself is not enough — pair it with `evidence_quality` to know how much weight to assign.

| `status` × `evidence_quality` | What it means |
|---|---|
| `supported` × `quoted_passage` | Strongest: LLM quoted a passage that backs the claim |
| `supported` × `abstract_only` | Good: backed by abstract; fulltext was unavailable |
| `supported` × `title_only` | **Capped to `partially_supported`** by deterministic post-LLM rule (S1-P1-B) |
| `partially_supported` × `quoted_passage` | Claim is mostly right but missing a qualifier |
| `unsupported` × `quoted_passage` | Strong contradiction signal |
| `unsupported` × `passages_searched_no_quote` | Verifier saw passages but found none that supported the claim — the BM25 chunks are surfaced for audit |
| `not_addressed` × `passages_searched_no_quote` | Same retrieval coverage, but the verifier abstains rather than asserting absence |
| `not_addressed` × `no_evidence` | Pipeline could not reach the source body at all |
| anything × `citing_paper_context` | **Capped to `partially_supported`** — verdict is internal-consistency only (S3-P1) |

The `passages_searched_no_quote` value was added in Phase A.2 to distinguish *"fulltext was retrieved and the LLM saw passages but didn't quote any"* from *"no passages were ever shown to the LLM"* (`no_evidence`). The auditor still sees what was searched in the former case, so it counts as **transparent** in the CTran metric.

---

## 3. Audit-trail fallback (Phase A.2)

When the LLM returns a verdict but `source_passages = []` — typically because the verdict is `unsupported` or `not_addressed` and the model chose not to quote — the verifier falls back to the BM25-selected passages it had shown the LLM:

```
LLM returns []           BM25 had selected 3 passages
       │                          │
       └──────────► fallback ◄────┘
                       │
                       ▼
   source_passages = [bm25_chunk_1, ..., bm25_chunk_3]   (each truncated to 800 chars)
   evidence_quality = "passages_searched_no_quote"
```

Pre-fix this combination produced `evidence_quality = "no_evidence"` with empty `source_passages` — the auditor saw nothing despite the pipeline having seen everything. The fallback was the dominant CTran failure mode (50% of failures across 5 benchmarks).

The fallback also fires on JSON parse errors so a corrupted LLM response doesn't erase the audit trail.

See: [src/verify.py](../src/verify.py) `_truncate_passage` and the fallback in `verify_claim_fulltext`.

---

## 4. Numeric check (`verification.numeric_check`)

When the claim contains a quantitative assertion (OR/CI triple, p-value/CI), the **deterministic** numeric engine runs after LLM verification and attaches:

```jsonc
{
  "check_type": "or_ci_consistency"            // OR + lower CI + upper CI internally consistent
              | "p_value_ci_consistency",      // p-value < 0.05 ↔ CI does not cross null

  "consistent": true,                          // pass/fail of the check

  "extracted": [                               // structured numbers pulled by an LLM extractor
    { "label": "OR", "value": 1.42, "ci_low": 1.05, "ci_high": 1.94, "p_value": 0.024 }
  ],

  "explanation": "OR/CI internally consistent."
}
```

The comparison step itself contains **zero LLM calls** — same input always yields same verdict. The LLM is only used to *extract* the numbers from text.

---

## 5. Provenance record (`provenance.jsonl`)

Every step that touches a claim emits one line:

```jsonc
{
  "step_id": "uuid",
  "claim_id": "uuid",                           // foreign key to claims[].claim_id
  "operation": "extract"
             | "resolve"
             | "verify"
             | "verify_cross_modal"
             | "aggregate"
             | "numeric_extract"
             | "numeric_check"
             | "copilot_enrich"
             | "copilot_recommended_fix",
  "input_hash": "sha256(...)",                  // for replay/diff
  "output_hash": "sha256(...)",
  "model_id": "claude-sonnet-4-6",              // null for deterministic steps
  "timestamp": 1715347200.0,
  "tokens_in": 1234,                            // null for deterministic
  "tokens_out": 56,
  "cache_hit": true,                            // null when not applicable
  "confidence": 0.85                            // null when not applicable
}
```

Sum `tokens_in + tokens_out` across a run to compute exact cost. The `cache_hit` flag tracks Anthropic prompt-cache effectiveness on the cached system prompt.

---

## 6. Copilot enrichment

When the API is called with `mode: "copilot"`, the base claim entry above is wrapped in an `EnrichedVerification`:

```jsonc
{
  "base": { /* the entire V1 claim entry from §1 */ },

  "copilot": {
    "verdict_rationale": "string",              // present in all modes; user-facing rephrase

    "recommended_fix": {                        // null when no fix is needed
      "action": "swap_doi"
              | "reword"
              | "swap_and_reword"
              | "add_citation"
              | "remove",
      "regulatory_risk_level": "high"           // PHARMA mode only
                              | "medium"
                              | "low"
                              | null,
      "suggested_doi": "10.xxxx/yyyy",          // CrossRef-verified — never hallucinated
      "suggested_doi_title": "string",
      "reworded_claim": "string",               // when action ∈ {reword, swap_and_reword}
      "confidence": 0.90,
      "provenance_step_id": "uuid"              // links back to copilot_steps[]
    },

    // PHARMA mode fields (null in academic / general)
    "is_primary_source": true,
    "study_design": "rct" | "observational" | "case_control" | "animal_model"
                  | "in_vitro" | "meta_analysis" | "systematic_review"
                  | "narrative_review" | "guidelines" | "unknown" | null,
    "risk_of_bias": "low" | "medium" | "high" | "unknown" | null,
    "conflicting_evidence_flag": false,
    "primary_source_doi": "10.xxxx/yyyy",       // semantic-scholar lookup (or null)

    // ACADEMIC mode field (null in pharma / general)
    "novelty_claim": false
  },

  "copilot_steps": [ /* ProvenanceStep[] for the enrichment */ ],
  "mode": "pharma" | "academic" | "general"
}
```

### Fields surfaced in the Copilot HTML report

The `/runs/{run_id}/copilot_report.html` endpoint renders one card per claim with:

| Header | Source field |
|---|---|
| Verdict badge | `base.verification.status` |
| Rationale | `copilot.verdict_rationale` |
| Source link + retrieval badge | `base.source.doi` + `base.verification.verification_depth` |
| Source type / study design / risk of bias *(PHARMA only)* | `copilot.is_primary_source`, `copilot.study_design`, `copilot.risk_of_bias` |
| Recommended fix card *(when present)* | `copilot.recommended_fix.{action, suggested_doi, reworded_claim, confidence, regulatory_risk_level}` |
| Provenance accordion | `base` provenance + `copilot_steps` |

All user-facing strings (`claim_text`, `verdict_rationale`, `reworded_claim`) are HTML-escaped via Jinja2 `autoescape=True` before rendering.

See: [src/copilot/report_html.py](../src/copilot/report_html.py) and [src/copilot/templates/copilot_report.html.j2](../src/copilot/templates/copilot_report.html.j2).

---

## 7. Worked example

A minimal real entry (excerpted from [benchmarks/real_outputs/elicit_psilocybin/report.json](../benchmarks/real_outputs/elicit_psilocybin/report.json)):

```json
{
  "claim_id": "a3f1...",
  "claim_text": "Psilocybin produced rapid and sustained antidepressant effects in treatment-resistant depression (Carhart-Harris et al., 2016).",
  "claim_type": "causal",
  "cited_authors": ["Carhart-Harris"],
  "cited_year": 2016,
  "citation_markers": [],
  "source": {
    "found": true,
    "doi": "10.1016/s2215-0366(16)30065-7",
    "title": "Psilocybin with psychological support for treatment-resistant depression: ...",
    "similarity_score": 0.94,
    "oa_url": "https://www.thelancet.com/...",
    "retraction_status": false
  },
  "verification": {
    "status": "supported",
    "confidence": 0.92,
    "source_passages": [
      "All 12 patients showed reductions in depressive symptoms at 1 week post-treatment, with maximal effects at 5 weeks."
    ],
    "source_section": "results",
    "verification_depth": "fulltext",
    "retrieval_status": "passage_found",
    "evidence_quality": "quoted_passage",
    "numeric_check": null
  }
}
```

---

## See also

- [docs/architecture.md](architecture.md) — pipeline + verifier-routing diagrams
- [src/models.py](../src/models.py) — frozen dataclasses for `Claim`, `ResolvedSource`, `VerificationResult`, `ProvenanceStep`
