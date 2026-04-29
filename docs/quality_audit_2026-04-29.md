# Quality & Limitations Audit — Scientific Claim Verification Engine

**Date:** 2026-04-29
**Scope:** What is actually implemented, where the pipeline silently degrades, and which findings to challenge before launch. Grounded in code at specific file:line references and the three real-tool benchmark reports under `benchmarks/real_outputs/`.
**Use this doc to:** push back on what should be redesigned, expanded, or de-scoped before public launch.

---

## 1. What is implemented today

The pipeline takes free-form scientific text and produces a per-claim verification report. The data flow is:

```text
text
  └─ extract_claims (LLM)             src/extract.py
      └─ resolve_citations             src/resolve.py
          ├─ OpenAlex search           src/clients/openalex.py
          └─ CrossRef fallback         src/clients/crossref.py
          └─ retraction check          src/clients/crossref.py::check_retraction
              └─ fetch_fulltext        src/fetch_fulltext.py
                  ├─ oa_url PDF
                  ├─ PMC XML
                  ├─ Unpaywall PDF
                  └─ abstract fallback
                  └─ chunk_paper       src/chunker.py     (IMRAD section-aware)
                      └─ select_passages  src/bm25_selector.py  (BM25, top_k=3 hard default)
                          └─ verify_claim_fulltext_with_numeric  src/verify.py
                              ├─ verify (LLM, full-text passages)
                              └─ run_numeric_check   src/numeric/engine.py
                                  ├─ extract_numeric_assertions (LLM)
                                  └─ check_or_ci_consistency (pure Python)
                              └─ build_report         src/report.py
                                  ├─ report.json
                                  └─ provenance.jsonl
```

**Concrete coverage today:**

- 194 unit tests pass; mypy --strict clean; ruff clean
- F1 = 0.94 on SciFact dev split (oracle-mode), `verify_claim` byte-identical since baseline
- Three real-tool benchmark reports committed: Edison TREM2 (21 claims), Sakana AI Scientist v2 (13 claims), AnswerThis lactate (19 claims) — 53 total
- Numeric engine: **one** deterministic check (`or_ci_consistency`)
- Provenance: every step writes a hashed `ProvenanceStep` to `reports/runs/{report_id}/provenance.jsonl` with token/cost/cache metadata

**Aggregate signal across the three benchmarks:**

| metric | total |
|---|---|
| total claims | 53 |
| supported | 4 |
| partially_supported | 2 |
| **unsupported** | **0** |
| not_addressed | 47 |
| fulltext_verified | 24 |
| numeric_checks_run | 2 |
| **numeric_inconsistencies_flagged** | **0** |
| total cost | $2.41 |

**That zero/zero is the central concern of this doc.**

---

## 2. Quality concerns — 7 places the pipeline can silently degrade

### Concern 1 — Extraction recall ceiling: claims without explicit author-year anchors are dropped

**Where:** `src/extract.py:55-56` (system prompt)

```text
Do not include claims that have no citation anchor.
Do not hallucinate citations.
```

**What this means in practice:** any claim phrased in collective voice ("studies have shown…", "it has been demonstrated…") or referenced via numeric brackets without a parseable author/year is silently discarded. AnswerThis emits 60 `[N]` brackets — they only worked because the *References section* was pasted in, letting Claude resolve `[N]` to author/year on the fly. Without that, all 60 inline citations would be dropped at extraction.

**To challenge:** is the right answer to extend extraction to handle bracket+references, OR to widen the "verifiable claim" definition to include claims-with-implicit-citation? The first preserves precision; the second risks recall-driven false positives.

---

### Concern 2 — Resolution false positives: NO title/abstract similarity gate, only year-match

**Where:** `src/clients/openalex.py:52-65`

```python
def _pick_best_result(data, query_year):
    if not data: return None
    if query_year is None: return data[0]
    for result in data:
        if result.get("publication_year") == query_year:
            return result            # ← first year-match wins, regardless of title
    for result in data:
        year = result.get("publication_year")
        if year is not None and abs(year - query_year) <= 1:
            return result
    return data[0]
```

And `_compute_similarity` (lines 39–49) returns 1.0 / 0.9 / 0.8 based **purely on year proximity**. The `similarity_score` field surfaced in `report.json` is misleading — it is not a measure of how well the resolved paper matches the claim.

**Failure mode:** if OpenAlex returns five papers for a "Smith 2020 protein folding" query, and the first one published in 2020 is on Smith's *unrelated* topic, that wrong paper gets returned with `similarity_score=1.0`. Downstream the verifier fetches its full text and (correctly) returns `not_addressed` — which looks like the *claim* failed verification, when actually *resolution* failed.

**To challenge:** should we add a title/abstract cosine check (cheap, deterministic) before accepting a hit? Or compute `similarity_score` from title overlap with the claim's authors+text? The 47/53 not_addressed rate may be partly *wrong-paper-resolved* rather than *paper-doesnt-discuss-this*.

---

### Concern 3 — BM25 silent fallback returns IRRELEVANT chunks pretending to be relevant

**Where:** `src/bm25_selector.py:46-47`

```python
if max(scores) == 0:
    return list(chunks[:top_k])    # ← returns first 3 chunks of the paper
```

**What this means:** when the claim and the chunks share zero keyword overlap (paraphrase, synonym, entity rename), `select_passages` returns the first 3 chunks of the paper *as if they were the most relevant*. The downstream verifier sees `verification_depth="fulltext"` and gets handed the abstract, intro, and methods front-matter — even when the actual evidence is buried in Discussion or a sub-result.

**Why this matters:** the verifier's prompt says "err toward not_addressed rather than guessing." So when BM25 hands it three off-topic chunks, the verifier *correctly* says `not_addressed`. But the report claims `verification_depth="fulltext"` — which falsely suggests we tried hard. This is a calibration honesty problem: we report having full-text-verified a claim that we couldn't keyword-locate.

**To challenge:** add semantic embeddings (one MiniLM call per chunk, cached) so paraphrase-resilient retrieval is possible. Alternative: when `max(scores) == 0`, return empty and propagate `verification_depth="fulltext_no_passage"` rather than silently downgrading.

---

### Concern 4 — Verifier abstention bias: "err toward not_addressed" is a calibration choice baked into the prompt

**Where:** `src/verify.py:140` (system prompt)

```text
- If the passages are insufficient or off-topic, err toward not_addressed rather than guessing.
```

**What this means:** when the verifier is uncertain whether the passages contradict, support, or simply don't address the claim, the prompt explicitly tells it to abstain. The result is a systematic bias against `unsupported`.

This is conservative — abstention beats false alarms in a verification setting — but in the launch context it produces the implausible 0/53 unsupported count. The pipeline is structurally biased away from the very signal a Medical Writer or compliance reviewer needs.

**To challenge:** is "abstain when uncertain" the right default for a quality-control tool, or should we output `unsupported_low_confidence` as a fourth verdict so reviewers see *which* claims look mismatched even when evidence is thin? F1=0.94 was measured on SciFact, where abstention works because every example has a verifiable label — but real-world inputs are messier.

---

### Concern 5 — Confidence is LLM self-report, not calibrated against evidence

**Where:** `src/verify.py:252`

```python
confidence=float(parsed["confidence"])
```

The confidence value comes directly from the LLM's JSON output. There is no:

- post-hoc calibration against ground truth
- consistency check (high-confidence-supported with empty `source_passages` should be impossible but isn't enforced)
- cross-reference between `confidence` and BM25 score of the passages used

The system prompt (`verify.py:143`) gives a verbal anchor: "0.9-1.0 for clear-cut cases" — but this is the LLM grading itself.

**To challenge:** is uncalibrated self-reported confidence useful in a report shown to a Medical Writer? Should we replace it with a composite score (BM25 max × verdict-stability across N samples × passage coverage of claim entities)?

---

### Concern 6 — Numeric engine narrowness: the OR/CI check fires on a tiny slice of papers

**Where:** `src/numeric/checks.py` — only `check_or_ci_consistency` is implemented.

**Pre-conditions for the engine to fire (all must hold):**

1. Extractor (LLM) finds an `or_value`, `ci_low`, AND `ci_high` triple in the claim text
2. `ci_low > 0` (multiplicative scale check)
3. The extracted values parse as floats

In Edison TREM2 (a paper with multiple statistical tests), only **2 of 21** claims triggered the engine. Across the three benchmarks: **2/53 ran, 0/53 flagged**.

**What the engine cannot catch:**

- Percentages summing to >100% ("67% A, 45% B, 32% C, no overlap mentioned")
- p-value vs CI mismatch ("p=0.001" but "95% CI [-0.05, 0.30]")
- Sample size contradictions ("n=400 total" but "n=250 + n=200 in subgroups")
- Unit conversion errors ("dose 50 mg/kg, infused 500 mL")
- Effect-size vs CI mismatch
- Mean ± SD with SD wider than the plausible range
- Hazard ratio vs Kaplan-Meier curve description mismatch
- Any descriptive-stat consistency (mean inside [min, max], median between Q1 and Q3, etc.)

**To challenge:** the MVP scope deliberately froze at one check — is that still right at launch, or do we need 3-4 checks (percentages-sum-to-100, p-vs-CI, n-totals) so the engine fires on >50% of statistical claims? Each check is ~30 LOC of pure Python. The risk of over-scoping is real, but right now the engine is too narrow to surface *any* signal in real benchmarks.

---

### Concern 7 — The benchmark zero/zero result: pipeline is structurally honest but launch-flat

**Where:** `benchmarks/real_outputs/*/report.json`

| tool | sup | part | **unsup** | n/a | numeric_run | **inconsist** |
|---|---|---|---|---|---|---|
| Edison TREM2 | 3 | 1 | **0** | 17 | 2 | **0** |
| Sakana CompReg | 0 | 0 | **0** | 13 | 0 | **0** |
| AnswerThis lactate | 1 | 1 | **0** | 17 | 0 | **0** |
| **Total** | **4** | **2** | **0** | **47** | **2** | **0** |

**Two readings of this result:**

- **Charitable:** the pipeline is correctly conservative. It doesn't manufacture findings. The four supported + two partially_supported claims are real. The AnswerThis MCT haplotype claim ("MCT1, MCT2, MCT4" supported only for MCT1) is a genuine partial-overclaim catch — exactly the kind of finding a Medical Writer would value.
- **Critical:** the failure modes 1–6 above each push the pipeline toward `not_addressed`. Stack them and you get an engine that demonstrates "I am robust to false positives" but not "I catch the errors that matter." A launch demo needs at least one *unsupported* verdict on a real input where the cited paper actually contradicts the claim.

**To challenge:** is the right next step to (a) hand-curate one benchmark input where we *know* the source contradicts the AI-generated claim (ground-truth trap), (b) widen the failure modes 1–6 to surface latent unsupported, or (c) re-frame the launch story around "honest verifier + partial-overclaim catcher" rather than "inconsistency detector"?

---

## 3. Engineering hygiene that is solid

For symmetry — what's *not* concerning:

- **Provenance**: every step is hashed and persisted. Cost-per-claim is reconstructible from `provenance.jsonl`. Reproducibility is genuinely good.
- **`verify_claim` regression-safety**: byte-identical to the SciFact baseline. F1=0.94 preserved.
- **Caching**: OpenAlex (30 days), CrossRef, PDFs — all SQLite-cached with TTL. A re-run of the three benchmarks is now $0 of API spend.
- **Retraction check**: CrossRef `update-to` field is wired in (`src/clients/crossref.py::check_retraction`). It just hasn't fired on these inputs because none of the cited papers were retracted.
- **Type safety + tests**: 194/194 pass, mypy strict clean, no LLM in deterministic modules (rule enforced).

---

## 4. Open questions to challenge

In rough priority order. Each is a real fork in the roadmap, not a rhetorical question.

1. **Is the launch story "honest verifier" or "inconsistency catcher"?** They imply different prompt calibration (Concern 4) and different numeric engine scope (Concern 6).
2. **Should we add a title/abstract similarity gate to resolution?** ~50 LOC, deterministic, would directly attack Concern 2 — and may flip some `not_addressed` verdicts to `unsupported` (revealed wrong-paper resolutions).
3. **Should BM25 fallback to "no_passage_found" instead of returning random chunks?** ~5 LOC fix to Concern 3, with a corresponding `verification_depth` value. Costs zero, prevents misleading "fulltext_verified" labels.
4. **Should the numeric engine be widened from 1 to 3–4 checks before launch?** Concern 6. Each new check is small but each adds extraction-prompt complexity. Risk: re-scoping the MVP after Stage 2 was committed.
5. **Should we add semantic embeddings for passage selection?** Concern 3 root-cause fix. ~1-day spike, adds a dependency, but moves recall meaningfully.
6. **Should `confidence` be replaced or augmented with a composite score?** Concern 5. Cosmetic for now, but matters at launch.
7. **Should we hand-curate one ground-truth-contradiction input?** Concern 7. Risk: looks like cherry-picking. Reward: a real demo artifact.
8. **Should we re-extract AnswerThis with a `[N] → (Author, Year)` reference-rewriter** baked into extraction so future bracket-style inputs work without manual paste? Concern 1.

---

## 5. Recommendation for the next conversation

Rather than greenlight Stage 1E (SUMMARY.md) and Stage 3A (README rewrite) on top of the launch-flat data, **fix Concerns 2 and 3 first** — they are cheap, surgical, and they target the most plausible cause of the 47/53 not_addressed count. If after fixing them we still see 0 unsupported across 53 claims, the pipeline genuinely *is* a "honest verifier, not an inconsistency catcher" — and the launch story should reflect that. If unsupported counts rise, we have a real demo without manufacturing one.

Concern 6 (widening numeric checks) is a separate decision that does not block launch — but it should be made before the README claims "deterministic numeric verification" without scoping what that means.
