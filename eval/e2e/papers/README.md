# Benchmark Source Papers

Tracked PDF copies of the papers used as benchmark inputs for the `reference_paper_v1` evaluation set.

| File | Paper | Author | Year | Notes |
|------|-------|--------|------|-------|
| `perrelle_2023_lactate_isf.pdf` | Exploring Activity-Induced Lactate Pharmacokinetics: Implications for Minimally-Invasive Monitoring | Vivien Perrelle | 2023 | Ground truth benchmark input. 25 annotated claims in `reference_paper_v1_verdicts.json`. |

## Pipeline text inputs

The pipeline reads plain-text versions of these papers from `eval/e2e/source_texts/` (gitignored).
To regenerate the text input from a PDF:

```bash
python -c "
import fitz  # pymupdf
doc = fitz.open('eval/e2e/papers/perrelle_2023_lactate_isf.pdf')
text = '\n'.join(page.get_text() for page in doc)
open('eval/e2e/source_texts/review_2023_lactate_isf.txt', 'w').write(text)
"
```
