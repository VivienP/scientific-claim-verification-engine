# Canary Benchmark

This directory contains a seeded demo input, not a real-tool benchmark result.
Use it to exercise launch-critical failure modes that may not appear in a small
sample of real AI-for-science outputs.

Do not merge the canary counts with `benchmarks/real_outputs/`. In public
materials, label this as a controlled canary suite.

## Active tests (verified working)

Each path below is exercised by `input.txt` and confirmed in `report.json`.

- **Contradiction detection** — AlphaFold claim is deliberately inverted
  ("less accurate than random baselines"); verifier returns `unsupported`.

## Not yet implemented

The following paths appear in the original spec but have no verified pipeline
coverage. Corresponding fixture inputs must be added before these can be
claimed.

- [ ] Weak-resolution / wrong-citation diagnostic (pipeline surfaces
  `resolution_low_confidence` but the current `input.txt` produces
  `resolution_low_confidence=0`)
- [ ] Deterministic numeric inconsistency (OR/CI null-crossing check) — the
  Smith/Nguyen claim in `input.txt` is not picked up by `numeric_checks_run`
- [ ] Retraction check — Wakefield (1998) is in `input.txt` but
  `retracted_sources=0`; CrossRef retraction lookup is not returning a hit
