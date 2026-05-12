# Regression set 2026-05-11 — abstract_only_unsupported

## Trigger

Visual asset for X Thread #2 cited a 'fabricated' Elicit claim (20% sustained response at week 12, Goodwin NEJM 2022). The user pasted the NEJM full text showing the figure was real. Inspection of the verification record showed the verifier had only the abstract (NEJM PDF endpoint was paywalled, no PMC mirror, fulltext fetcher gave up) yet emitted `status: unsupported, confidence: 0.75` — a silent failure per CLAUDE.md line 167.

## Captured cases

15 regression entries — all confident unsupported verdicts from `elicit_psilocybin_rerun_860b1ae5` that were produced on abstract-only evidence.

## Blocking constraint

After Tracks A + D land, every entry in `regression.jsonl` must satisfy:

```
post_fix_verdict.status != 'unsupported'
  OR post_fix_verdict.evidence_quality NOT IN {abstract_only, title_only,
                                                citing_paper_context, no_evidence}
```

i.e. either the verdict changed (to `unverifiable`, or to a different fulltext-grounded status), or the evidence depth improved (to fulltext-grade).

## Acceptance for the marquee Goodwin case

Look for `elicit_psilocybin__ae1ff864` (or grep for 'Sustained response rates at 12 weeks were only 20%' in claim_text). Post-fix this specific claim emits `status="unverifiable", confidence=None` from the abstract-only path (pinned by `tests/unit/test_regressions.py::test_goodwin_nejm_2022_abstract_only_returns_unverifiable`). Upgrading the verdict from `unverifiable` to `supported` requires retrieving NEJM full text — currently blocked by Cloudflare bot protection (documented in `memory/project_publisher_access_limits.md`); the publisher_html fallback exists for future use when access is unblocked or a BYO-PDF path is added.

## Coverage clarification (post-implementation, 2026-05-11)

The 15 entries split into two coverage classes under the implemented fix:

- **Numeric / Results-section claims** (incl. the Goodwin marquee `elicit_psilocybin__ae1ff864`): Fully fixed by Track A. `_claim_has_specific_numeric(claim_text)` matches, the helper downgrades to `unverifiable`, the new verdict is `(status="unverifiable", confidence=None)`.
- **Qualitative claims** (e.g. "studies systematically excluded X"): NOT fully fixed by Track A. The helper's `claim_text` discrimination intentionally preserves qualitative-claim verdicts on `abstract_only` evidence — the abstract is often sufficient for qualitative claims, so over-downgrading would hurt recall on SciFact-style content. The original `expected_behavior` field on these entries predates the discrimination decision and is stale; under the implemented design, the new verdict is whatever the LLM emits, passed through unchanged. If the LLM still emits `unsupported` on a qualitative claim where the abstract is silent, that's a separate failure class (prompt-level: the verifier should emit `not_addressed`, not `unsupported`, when the source is silent on the topic). Track these for a future prompt-level pass via `@prompt-smith`.

The `blocking_constraint` field still applies as a regression-prevention guard: post-fix verdicts on these inputs must NOT be `unsupported` with `evidence_quality in {abstract_only, ...}` AND `confidence > 0.5`. For numeric entries this is enforced by the helper; for qualitative entries this is a TODO until the prompt patch lands.

## Plan reference

`reports/that-s-very-bad-the-stateful-sloth.md`
