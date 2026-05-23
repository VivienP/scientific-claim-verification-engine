# Spec: Phase 4 — Close P1-2 with eval-aware scoring (MIRROR redux)

**Status:** **QUEUED** for Phase 4. Not active work. Do not implement without explicit phase promotion via @scope-guard.
**Date drafted:** 2026-05-23
**Author:** post-mortem of Phase 3 attempt
**Parent context:** [docs/specs/phase3-verifier-consume-structured-claim.md](phase3-verifier-consume-structured-claim.md), [docs/specs/unverifiable-verdict-emission-gates.md](unverifiable-verdict-emission-gates.md)
**Related memory:** `feedback_abstract_only_verdicts.md` (P1-2 incident class), `feedback_e2e_benchmark.md` (eval rigor)

## Problem statement

Phase 3 commit 3.4 (`9712761`) attempted the MIRROR scope — remove claim-type discrimination from `safe_verification_result` so qualitative claims on insufficient evidence are also downgraded to `unverifiable`. This closes the P1-2 silent-failure incident class.

The implementation was correct. The /eval gate caught a catastrophic regression: F1 dropped 0.92 → 0.38 on SciFact dev (50 claims), with `unsupported`-class F1 falling from 0.93 to 0.00. The commit was reverted in `d71f383`.

**Root cause:** SciFact dev split is 100% abstract-only evidence, and the SciFact class schema does not include an `unverifiable` class. After MIRROR, all `supported`/`unsupported` verdicts on abstract evidence flip to `unverifiable` — which the eval pipeline counts as misses for their true class.

The product is doing the right thing (honest abstention). The eval is mismeasuring it.

## What MUST land before MIRROR can be re-attempted

### Track 1: Eval scoring augmentation

The SciFact eval pipeline must learn to treat `unverifiable` as a valid abstention, not a misclassification.

Concrete options to spec when this lands:

- **Option A — Abstention-credited scoring**: count `unverifiable` as a partial credit (e.g. 0.5 of the true-class score, rewarding honesty without rewarding evasion). Risk: tunable parameter, sensitive to scoring tuning.
- **Option B — Two-tier eval**: report two F1 scores: "strict" (current) and "abstention-aware" (which excludes `unverifiable` predictions from precision/recall calculations entirely). Lets the founder track both axes.
- **Option C — Add `unverifiable` to gold standard**: programmatically reclassify SciFact gold-truth claims into `(supported, unsupported, not_addressed, unverifiable)` based on evidence depth available. Heavy lift, risks data contamination, but the most accurate.

Pick one (likely B, lightest touch).

### Track 2: Re-land MIRROR scope

After Track 1 lands, re-apply the same change reverted from commit `9712761`:

- `src/models.py::safe_verification_result` — remove the `claim_text is None or _claim_has_specific_numeric(claim_text)` discrimination
- Rewrite `tests/unit/test_models.py::test_helper_passes_qualitative_supported_on_abstract_through` to assert new behavior
- Rewrite `tests/unit/test_verify_abstract.py::test_supported_status_qualitative_claim_passes_through` similarly
- Add the 6 qualitative-downgrade tests originally in 3.4

The original commit body and diff are preserved in git history at `9712761` (revert at `d71f383`) — use as reference.

### Track 3: Re-run /eval gate with new scoring

Confirm:

- "Strict" F1 still shows the expected drop (~0.38) — proves MIRROR is active
- "Abstention-aware" F1 stays at baseline level (~0.92) — proves honest abstentions are credited
- Per-claim audit: every `unverifiable` emission has a corresponding `unverifiable_reason` populated

## Why this is queued, not active

- Phase 1 (current) goal is MVP shippability. The current behavior (numeric-only downgrade) is the empirical equilibrium between two correctness models.
- The eval scoring change is non-trivial — it requires understanding how SciFact eval is computed and whether the dataset itself needs annotation extension.
- Track 1 has its own design ambiguity (3 options) that needs founder input before architect can spec.
- Phase 4 in the project plan is for "post-MVP correctness work" per [CLAUDE.md](../../CLAUDE.md) phase boundaries — exactly the right home.

## Acceptance criteria (when picked up in Phase 4)

- [ ] Eval scoring augmented per Track 1 (Option A, B, or C)
- [ ] MIRROR re-landed per Track 2
- [ ] /eval shows both metrics: strict F1 reflects MIRROR's strictness, abstention-aware F1 stays at baseline
- [ ] @reviewer APPROVE
- [ ] All P1-2 silent-failure regression tests pass with explicit `unverifiable` verdicts

## Out of scope for this ticket

- Closing P1-2 via prompt-side instructions (would re-introduce the `prefer-deterministic-gates` violation A3 already dropped — see `docs/specs/unverifiable-verdict-emission-gates.md` Decision log line 14)
- Fulltext fetcher improvements (separate Track D)
- Re-architecting `safe_verification_result` beyond removing one condition
