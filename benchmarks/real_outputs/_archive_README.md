# Archived benchmark outputs

The `_archive_pre_fix/` directories inside each benchmark folder hold
archived `report.json` and `provenance.jsonl` snapshots emitted under a
verifier configuration that conflates three epistemically distinct
verdicts into one:

- `unsupported` (the abstract directly contradicts the claim)
- `not_addressed` (the abstract is silent on the claim)
- pipeline could not access full text (now reported as `unverifiable`)

**Do not cite these numbers.** The archived configuration emits
confident `supported` / `unsupported` verdicts on abstract-only evidence
whenever the LLM is prompted to, regardless of whether the abstract
actually contains the assertion. On the captured Elicit psilocybin run,
15 of 57 confident verdicts (26%) match this silent-failure pattern.
The current pipeline catches every numeric-claim case via the
`safe_verification_result` helper and routes silence to `not_addressed`
via the contradicts-vs-silent prompt split.

The archives are retained for two reasons:

1. **Diff replay.** `scripts/replay_psilocybin_kpi.py` reads
   `_archive_pre_fix/report.json` and routes the captured LLM outputs
   through the current helper to measure the KPI movement attributable
   to the helper alone (vs the LLM-side prompt change).
2. **Reproducibility.** Anyone reproducing the May 2026 incident can
   compare the verdicts as-emitted at the time against current verdicts.

For current numbers, see the parent benchmark folder's `report.json`
(when present) or `benchmarks/real_outputs/README.md` for the aggregate
table.
