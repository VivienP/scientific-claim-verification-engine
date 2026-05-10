# Current failure modes — verify_v1

(Empty placeholder. Populate before invoking `@prompt-smith improve src/prompts/verify_v1.md`.)

Required content:

1. **What is failing?** Specific examples (claim text, expected verdict, actual verdict, source run path).
2. **Category:** extraction-miss / false-positive / wrong-status / wrong-confidence / hallucinated-evidence.
3. **Hypothesis:** prompt ambiguity / missing examples / wrong framing / structural issue.

Reference rules:
- Never use the test split for prompt selection.
- Eval sets must run offline (no real API calls).
