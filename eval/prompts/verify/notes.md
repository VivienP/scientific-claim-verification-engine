# Iteration history — verify_v1

Format per entry:

```
## {YYYY-MM-DD} — Iteration {N}
- Failure addressed: {description}
- Winner: v{N}, hypothesis: {hypothesis}
- Before: F1={old}, P={old_p}, R={old_r}
- After:  F1={new}, P={new_p}, R={new_r}
- Deployed: yes/no
- Notes: {anything surprising}
```

(No iterations yet. First entry will be written by `@prompt-smith` after the
first `/dogfood` run that surfaces a verify-prompt-related failure.)
