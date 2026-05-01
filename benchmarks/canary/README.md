# Canary Benchmark

This directory contains a seeded demo input, not a real-tool benchmark result.
Use it to exercise launch-critical failure modes that may not appear in a small
sample of real AI-for-science outputs.

Expected signals:

- wrong-citation / weak-resolution diagnostic
- contradicted claim against a cited source
- deterministic numeric inconsistency
- retraction check on a known retracted biomedical paper

Do not merge the canary counts with `benchmarks/real_outputs/`. In public
materials, label this as a controlled canary suite.

