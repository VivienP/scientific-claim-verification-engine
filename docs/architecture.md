# Architecture

End-to-end diagrams for the verification pipeline. Use this doc to understand
*how* a claim flows from input text to a `report.json` entry. For the on-disk
shape of that entry, see [output-schema.md](output-schema.md).

All diagrams are GitHub-flavored mermaid; they render directly on github.com
and on the Copilot-served HTML view of this file.

---

## 1. End-to-end pipeline

```mermaid
flowchart TD
    A[input text] --> B[extract claims<br/>LLM + partial-recovery]
    B --> C[resolve citations<br/>bib DOI → CrossRef → OpenAlex → PubMed]
    C --> D[enrich metadata<br/>PubMed PMID, Europe PMC OA]
    D --> E[fetch fulltext<br/>OA URL → PMC → Europe PMC → Unpaywall]
    E --> F[chunk + BM25 select<br/>IMRAD-aware, token-budgeted]
    F --> G{verifier router}
    G -->|fulltext + passages| H1[verify_claim_fulltext]
    G -->|abstract only| H2[verify_claim]
    G -->|long title, no abstract| H3[verify_claim_title_only]
    G -->|multi-citation set| H4[verify_claim_multi_source]
    G -->|source unreachable| H5[verify_claim_citing_context]
    H1 --> I[audit-trail fallback<br/>surface BM25 passages on no-quote]
    H2 --> J
    H3 --> J
    H4 --> J
    H5 --> J
    I --> J{numeric assertion?}
    J -->|yes| K[deterministic numeric check<br/>OR/CI · p-value/CI]
    J -->|no| L[aggregate]
    K --> L
    L --> M[(report.json<br/>provenance.jsonl)]

    classDef llm fill:#fff3cd,stroke:#856404,color:#856404
    classDef det fill:#d4edda,stroke:#155724,color:#155724
    classDef io fill:#d1ecf1,stroke:#0c5460,color:#0c5460
    class B,H1,H2,H3,H4,H5 llm
    class F,I,K,L det
    class A,M io
```

**Legend** — yellow = LLM call · green = deterministic Python · blue = I/O.

Deterministic steps stay LLM-free by design: no LLM may be added to
chunk-selection, audit-trail fallback, numeric checks, or aggregation.
Same input always produces the same verdict.

---

## 2. Verifier routing

The router in [src/pipeline.py](../src/pipeline.py) picks a verifier mode
based on what was retrieved for the cited source. Each mode has different
correctness and cost characteristics.

```mermaid
flowchart TD
    Start([claim + ResolvedSourceSet]) --> Q1{markers cite<br/>multiple sources?}
    Q1 -->|yes| MS[verify_claim_multi_source<br/>fan-out → aggregate]
    Q1 -->|no| Q2{source.found?}
    Q2 -->|no| Q3{citing-paper text<br/>available?}
    Q3 -->|yes| CC[verify_claim_citing_context<br/>capped at partially_supported]
    Q3 -->|no| NA1[not_addressed<br/>no LLM call]
    Q2 -->|yes| Q4{fulltext fetched<br/>+ BM25 passages?}
    Q4 -->|yes| FT[verify_claim_fulltext<br/>+ audit-trail fallback]
    Q4 -->|no| Q5{abstract present?}
    Q5 -->|yes| AB[verify_claim<br/>abstract mode]
    Q5 -->|no| Q6{title length ≥ 20?}
    Q6 -->|yes| TO[verify_claim_title_only<br/>capped at partially_supported]
    Q6 -->|no| NA2[not_addressed<br/>no LLM call]
```

Hard caps (deterministic post-LLM):

- `title_only` mode → verdict capped at `partially_supported`, confidence ≤ 0.7.
- `citing_paper_context` mode → verdict capped at `partially_supported`, confidence ≤ 0.6.
- `multi_source` mode → aggregator returns `partially_supported` whenever any
  one source disagrees with another (cross-modal disagreement philosophy
  applied to multi-source).

---

## 3. Audit-trail fallback

When the verifier sees BM25 passages but doesn't quote any (e.g. parse error or low confidence), the fallback below preserves the audit trail so a reviewer can still inspect what was shown to the LLM.

```mermaid
flowchart LR
    P[BM25 selected<br/>passages 1..N] --> V[verify_claim_fulltext<br/>LLM call]
    V --> Q{LLM returned<br/>quoted passages?}
    Q -->|yes| QP["source_passages = LLM quotes<br/>evidence_quality = quoted_passage"]
    Q -->|no| FB["source_passages = BM25 chunks<br/>truncated to 800 chars each<br/>evidence_quality = passages_searched_no_quote"]
    Q -->|parse error| FB
    QP --> R[(verification record)]
    FB --> R

    classDef new fill:#cfe2ff,stroke:#084298,color:#084298
    class FB new
```

The blue path surfaces the BM25 chunks as `passages_searched_no_quote` instead of dropping them as `no_evidence`, so the auditor sees what the pipeline saw.

CTran impact across 135 claims on 5 benchmarks: **48.9% → 65.9% (+17pp)**.
Per-benchmark breakdown: see the CTran table in
[benchmarks/real_outputs/README.md](../benchmarks/real_outputs/README.md#claim-transparency-ctran).

---

## 4. Provenance graph for one claim

Every step that touches a claim emits a `ProvenanceStep` with input/output
hashes, model id, tokens, and cache state. The `claim_id` field links every
step back to the `claims[]` entry in `report.json`.

```mermaid
flowchart TD
    C0([input text]) -->|extract| S1[ProvenanceStep<br/>operation = extract]
    S1 --> C1([Claim])
    C1 -->|resolve| S2[ProvenanceStep<br/>operation = resolve]
    S2 --> C2([ResolvedSource])
    C2 -->|fetch + chunk + select| C3(["PaperChunk[]"])
    C2 -->|verify| S3[ProvenanceStep<br/>operation = verify]
    C3 --> S3
    S3 --> R0([VerificationResult])
    R0 -->|cross-modal check<br/>opt-in| S3b[ProvenanceStep<br/>operation = verify_cross_modal]
    R0 -->|numeric assertion| S4[ProvenanceStep<br/>operation = numeric_extract]
    S4 --> S5[ProvenanceStep<br/>operation = numeric_check]
    S5 --> R1([VerificationResult<br/>+ numeric_check])
    R1 -->|copilot mode| S6[ProvenanceStep<br/>operation = copilot_enrich]
    S6 --> S7[ProvenanceStep<br/>operation = copilot_recommended_fix]
    S7 --> R2([EnrichedVerification])
```

Replay invariant: re-running the pipeline on the same input text with the
same model should produce identical `input_hash` / `output_hash` chains for
the deterministic steps. LLM steps drift, but the chain shape is preserved.

Provenance schema (Phase 0–3): JSONL append at
`reports/runs/{run_id}/provenance.jsonl`; graph DB deferred to Phase 4+.

---

## 5. HTTP API request lifecycle

Phase C deployment exposes the pipeline behind FastAPI with async jobs so
calls don't time out at any reverse proxy.

```mermaid
sequenceDiagram
    autonumber
    participant Client
    participant API as FastAPI<br/>src/api/app.py
    participant Worker as Job worker<br/>src/api/jobs.py
    participant Pipe as run_pipeline
    participant FS as reports/runs/

    Client->>API: POST /verify  {text, mode, copilot_mode}
    API-->>Client: 202 Accepted  {job_id, poll_url}
    API->>Worker: enqueue(job_id)

    Worker->>Pipe: run_pipeline(text, config)
    Pipe-->>Worker: verifications, steps
    Worker->>FS: write report.json + provenance.jsonl
    Worker->>API: status = completed

    loop until completed
        Client->>API: GET /jobs/{job_id}
        API-->>Client: {status, run_id?, result?}
    end

    Client->>API: GET /runs/{run_id}/copilot_report.html
    API-->>Client: 200  self-contained HTML
```

The `/runs/{run_id}/copilot_report.html` endpoint reads the `report.json`
under `reports/runs/{run_id}/` and renders the Copilot HTML view documented
in [output-schema.md §6](output-schema.md#6-copilot-enrichment).

Hardening: container runs as non-root uid 10001, `read_only: true`, `cap_drop: ALL`,
exact-pinned Python deps, all endpoints (except `/health`) require
`X-API-Key`. Single-tenant Phase C; Postgres-backed multi-tenant `JobStore`
is deferred to Phase D.

---

## See also

- [output-schema.md](output-schema.md) — on-disk shape of `report.json` entries
- [src/models.py](../src/models.py) — frozen dataclasses, type-checked
- [src/pipeline.py](../src/pipeline.py) — verifier routing logic
- [src/verify.py](../src/verify.py) — verifier modes + audit-trail fallback
- [benchmarks/real_outputs/README.md](../benchmarks/real_outputs/README.md#claim-transparency-ctran) — CTran baseline + lift
