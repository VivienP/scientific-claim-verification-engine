# Archived pre-fix benchmark outputs

The `_archive_pre_fix/` directories inside each benchmark folder hold the
`report.json` and `provenance.jsonl` files produced by the pre-2026-05-12
verifier — the one with the silent-failure bug fixed by commits
[`992303e`](../../../README.md) through [`3211b23`](../../../README.md).

**Do not cite these numbers.** Specifically, they conflate three
epistemically distinct verdicts into one:

- `unsupported` (the abstract directly contradicts the claim)
- `not_addressed` (the abstract is silent on the claim)
- pipeline could not access full text (now reported as `unverifiable`)

The pre-fix pipeline emitted confident `supported` / `unsupported`
verdicts on abstract-only evidence whenever the LLM was asked to,
regardless of whether the abstract actually contained the assertion.
On the captured Elicit psilocybin run, 15 of 57 confident verdicts
(26%) matched this silent-failure pattern. The post-fix pipeline
catches every numeric-claim case via the
`safe_verification_result` helper and routes silence to `not_addressed`
via the Track G prompt rewrite.

These archives stay on disk for two reasons:

1. **Pre-/post-fix diff replay.** `scripts/replay_psilocybin_kpi.py`
   reads `_archive_pre_fix/report.json` and routes the captured LLM
   outputs through the new helper to measure the KPI movement that
   is attributable to the helper alone (vs Track G's LLM-side fix).
2. **Reproducibility.** Anyone reproducing the May 2026 incident can
   compare the verdicts as-emitted at the time against the post-fix
   verdicts.

For current numbers, see the parent benchmark folder's `report.json`
(when present) or `benchmarks/real_outputs/README.md` for the aggregate
table.
