# Regression catalog

Permanent record of failures caught by `/dogfood` (or by hand) that the SciFact
dev split did not catch on its own. Each entry compounds: once added, it is a
test case forever.

## Directory layout

```
eval/regressions/
├── README.md
├── 2026-05-11/
│   └── 9c7f3a-elicit/
│       └── regression.jsonl
├── 2026-05-18/
│   └── 4ab2-consensus/
│       └── regression.jsonl
└── ...
```

- One sub-directory per `/dogfood` run that produced regressions
- One `regression.jsonl` per run (one JSON object per line, one per failed claim)
- Date-keyed for chronology, run-id-keyed for traceability back to the source run

## Schema (per JSON line)

```json
{
  "regression_id": "9c7f3a-elicit-claim-12",
  "captured_at": "2026-05-11T14:32:00Z",
  "source_run": "reports/dogfood/2026-05-11/elicit/",
  "claim_id": "claim-12-from-report.json",
  "claim_text": "Lactate concentration in interstitial fluid is approximately 1.7 mmol/L",
  "expected_verdict": "partially_supported",
  "actual_verdict": "supported",
  "expected_doi": "10.1234/example",
  "actual_doi": "10.1234/example",
  "failure_category": "verifier_overconfident_wrong",
  "notes": "Claim asserts a range; source reports a single point. Should be partially_supported per Clause B.1."
}
```

Required fields: `regression_id`, `captured_at`, `source_run`, `claim_id`,
`claim_text`, `actual_verdict`, `failure_category`.

Conditionally required: `expected_verdict` (set to `"TBD"` if unknown — must
be filled before the test is meaningful), `expected_doi` / `actual_doi`
(only for `resolver_*` categories).

## Why separate from `eval/scifact/dev.jsonl`

Per [`.claude/rules/benchmark-isolation.md`](../../.claude/rules/benchmark-isolation.md):
SciFact is a public benchmark used for external comparability. Mixing internal
regressions with SciFact dev would break that. Future re-evaluation of SciFact
on different splits or with different metrics must remain possible.

Also: regressions follow a different schema (failure_category, source_run, etc.)
that doesn't fit the SciFact label format.

## How `/eval` consumes this

[`tests/unit/test_regressions.py`](../../tests/unit/test_regressions.py)
auto-discovers every `regression.jsonl` via glob and parametrizes one test
case per JSON line. Tests fail (red) until the underlying pipeline behavior
is fixed.

`scripts/eval_scifact.py` does NOT consume regressions — they are tracked
separately in pytest, not in F1 metrics. The decoupling is intentional.

## Failure category taxonomy

| Category | Patch target |
|---|---|
| `extractor_missed_claim` | `src/prompts/extract_v1.md` |
| `extractor_false_positive` | `src/prompts/extract_v1.md` |
| `resolver_wrong_doi` | `src/resolve.py` (deterministic, no prompt patch) |
| `resolver_not_found` | `src/resolve.py` |
| `verifier_wrong_verdict` | `src/prompts/verify_*_v1.md` |
| `verifier_overconfident_wrong` | Cross-modal verify rule + prompt patch |
| `prompt_injection` | `src/prompt_guard.py` |
| `unknown` | Manual review — no auto-patch |

## Retention

Regression entries are permanent. Even after the underlying bug is fixed,
the regression test stays — it's the only thing preventing the same bug
from recurring under a different code path.

If a category is fully retired (e.g. all `prompt_injection` entries pre-date
the latest guard rewrite), tag the entries as `obsolete: true` rather than
deleting. They serve as historical record.
