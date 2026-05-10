# Prompt evaluation sets

Per-prompt eval sets for `@prompt-smith`. Structure:

```
eval/prompts/{name}/
├── examples.jsonl       — labeled examples covering target failure modes
├── notes.md             — iteration history (date, hypothesis, F1 delta, deploy decision)
└── failure_analysis.md  — current failure modes driving next iteration
```

`{name}` matches the prompt file under [`src/prompts/`](../../src/prompts/) without
the `.md` extension. Example: prompt `verify_v1.md` → eval set
`eval/prompts/verify/`.

## Seeding a new eval set

1. Create the directory `eval/prompts/{name}/`.
2. Write a one-paragraph `failure_analysis.md` describing what is failing and why.
3. Sample 20–40 examples from a *labeled* source (SciFact dev split, prior dogfood
   regressions in [`eval/regressions/`](../regressions/)) covering the failure
   pattern + at least 5 happy-path examples to detect over-correction.
4. Write each example as one JSON object per line:
   ```json
   {"input": "...", "expected_output": "...", "label": "supported", "source": "scifact:dev:42"}
   ```
5. Leave `notes.md` empty — `@prompt-smith` will populate it.

## Train / dev / test isolation

- Eval set examples may come from `eval/scifact/train/` or `eval/scifact/dev/`.
- **Never** seed from `eval/scifact/test/`.
- Regressions in [`eval/regressions/`](../regressions/) are always safe to use.

## Why no automatic seeding

This directory is intentionally empty until the first prompt iteration. Auto-seeding
30 random claims is over-engineering: the value of an eval set comes from being
*targeted at a specific failure mode* observed in the wild. Wait until `/dogfood` or
a real bug surfaces, then seed deliberately.
