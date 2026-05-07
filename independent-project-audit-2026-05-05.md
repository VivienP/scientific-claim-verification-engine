# Independent Project Audit - 2026-05-05

## Verdict

You are directionally right on the technical problem and still not heading in the right direction as a company-building project.

You built a real verification pipeline, not just a prompt wrapper. The repo has frozen dataclasses, provenance steps, citation resolution, full-text retrieval, BM25 passage selection, verifier calls, narrow numeric checks, benchmark artifacts, and explicit abstention fields. That is source-backed by the public README pipeline (`README.md:90-104`), the model definitions (`src/models.py`), the report summary/provenance code (`src/report.py:54-110`, `src/report.py:176-192`), and the committed benchmark outputs (`benchmarks/real_outputs/SUMMARY.md:7-12`).

But the project is currently much more credible as a careful engineering prototype than as a validated product. Your strongest evidence is still mostly self-produced. Your benchmark is tiny, stale in public-facing docs, and not yet end-to-end against hand-labeled real documents. Your public story says "design partners," but the repo contains no customer interview notes, pilot records, buyer workflow evidence, pricing evidence, or competitive landscape document. That absence is a finding, not an administrative gap.

The hard truth: you are treating verification credibility as if it can be earned primarily by better code and better benchmark hygiene. For this market, that is necessary and insufficient. The buyer risk is workflow trust, liability, procurement, and whether anyone with painful scientific review work will adopt this before an incumbent or a generic research assistant makes the output good enough.

## What This Project Is Today

You built a local Python 3.12 scientific claim verification engine. It takes text with citation anchors, extracts cited claims with an LLM, resolves citations through OpenAlex plus CrossRef fallback, tries to fetch full text through OA URL, PMC, and Unpaywall, selects passages using BM25, verifies each claim against selected passages or abstracts, optionally runs narrow numeric consistency checks, and writes `report.json` plus `provenance.jsonl`.

This is not yet an autonomous product, not yet an API, not yet a dataset company, and not yet a customer-validated workflow. Your own project instructions say `CURRENT_PHASE: Phase 1 - MVP`, FastAPI is deferred to Phase 5, and no ORM appears until Phase 4 (`AGENTS.md:87`, `AGENTS.md:92-94`). The roadmap's later phases include API, billing, benchmark datasets, domain models, and publisher integrations, but the current checkout is still a command-line verification prototype with examples and committed reports (`development-roadmap.md:47`, `development-roadmap.md:837-944`).

The actual current product surface is:

- CLI/dev usage through `examples/sample_run.py` and report display scripts.
- Public README plus benchmark directories.
- Three real-output benchmark inputs: Edison TREM2, Sakana AI Scientist, and AnswerThis lactate (`benchmarks/real_outputs/*/meta.json`).
- A seeded canary suite that is explicitly not real-tool evidence (`benchmarks/canary/README.md:3`, `benchmarks/canary/README.md:14-15`).
- A verifier-only SciFact baseline.
- An untracked `eval/e2e` scaffold for future full-pipeline measurement.

## Benchmark Credibility Is The Main Problem

Your README and benchmark summary disagree.

The README says the real-output benchmark covers 61 claims, with 3 supported, 4 partially supported, 1 unsupported, 53 not addressed, and total cost $0.79 (`README.md:36-41`). The current benchmark summary says 60 claims, with 6 supported, 2 partially supported, 2 unsupported, 50 not addressed, and total cost $0.93 (`benchmarks/real_outputs/SUMMARY.md:9-12`). The summary also says only Edison was re-run after the May 1 BM25 token-budget fix, while Sakana and AnswerThis were not re-verified (`benchmarks/real_outputs/SUMMARY.md:5`).

This matters because your public credibility depends on exactness. A tool that audits scientific claims cannot have its own headline benchmark numbers out of sync with its benchmark artifact.

The honest aggregate from `SUMMARY.md` is:

- 60 real-output claims.
- 8 non-abstention positive/partial verdicts: 6 supported plus 2 partially supported.
- 2 unsupported verdicts.
- 50 not addressed.
- 26 passage-found claims.
- 34 full-text-unavailable claims.
- 1 numeric check run.
- 0 numeric inconsistencies flagged.
- $0.93 total cost.

That means 83.3% of claims are `not_addressed` and 56.7% are `fulltext_unavailable`. Numeric coverage is 1.7% of claims. Unsupported detection is 3.3% of claims. These are not bad numbers for an honest early verifier, but they are weak numbers for a launch claim that the system catches scientific hallucinations in the wild.

Your benchmark is also only three tools and three prompts. Edison is one topic, AnswerThis is one topic where you have domain knowledge, and Sakana is one AI-generated paper. The metadata is useful, but two inputs have no source URL and one is manual paste (`benchmarks/real_outputs/edison_trem2/meta.json:2-6`, `benchmarks/real_outputs/answerthis_lactate/meta.json:2-9`). This is acceptable for a private dogfood artifact. It is not enough to support broad market claims.

The canary is worse than the README implies. The canary README says it should exercise weak resolution, contradiction, numeric inconsistency, and retraction check paths (`benchmarks/canary/README.md:7-12`). The actual canary report has 4 claims, 1 unsupported, 3 not addressed, 0 low-confidence resolutions, 0 retracted sources, 0 numeric checks, and 0 numeric inconsistencies (`benchmarks/canary/report.json:6-20`). The canary currently proves one contradiction path works. It does not prove the demo-critical numeric or retraction paths.

SciFact should not be used as a product proof. Your README correctly says the current F1 number is verifier-only and does not measure extract -> resolve -> retrieve -> verify (`README.md:135-142`). The script confirms this: it builds oracle claims and abstracts and explicitly does not call `extract_claims()` or `resolve_citations()` (`scripts/eval_scifact.py:101-103`, `scripts/eval_scifact.py:200-201`). That is a legitimate regression check for the verifier. It is not evidence that the system works on messy scientific documents.

You started fixing the right problem with the `eval/e2e` scaffold, but it is not done. The README in that directory says the annotated `reference_paper_v1.json` is not yet created (`eval/e2e/README.md:10-11`). The measurement script is therefore infrastructure without the ground truth that would make it matter. Until that file exists and has been run, the project has no hand-labeled full-pipeline recall number.

## Technical Reality

You made several sound technical choices. Source-backed:

- You avoided a framework and kept a small Python package surface, consistent with Phase 1 instructions (`AGENTS.md:87-94`).
- You emit structured provenance and aggregate hashes. The aggregate input and output hashes now include the full `claim_records`, not just summary stats (`src/report.py:176-183`).
- You made retrieval status explicit: `passage_found`, `no_passage_found`, and `fulltext_unavailable` are model fields (`src/models.py:66-70`) and report summary fields (`src/report.py:82-110`).
- You corrected an earlier BM25 truncation issue with token-aware passage selection. The May 1 commit explicitly says it replaced 4000-character truncation (`git log`, commit `7f58fd7`), and the selector enforces a token budget with BM25-ranked chunks (`src/bm25_selector.py:80-90`).

The structural limits are still large.

First, extraction only sees cited claims. The extractor system prompt requires a specific cited source and excludes claims without citation anchors (`src/extract.py:25-27`, `src/extract.py:55-56`). That is consistent with the current product promise, but it means the tool is not a broad scientific inconsistency detector. It mostly audits whether a cited source says what the surrounding text claims. If a user expects "find false claims in my draft," this will miss uncited falsehoods by design.

Second, citation resolution is fragile. The query is built from up to three cited authors, the year, and the first five words of the claim (`src/resolve.py:26-30`). Claims without authors or year are skipped without an HTTP call (`src/resolve.py:45`, `src/resolve.py:52-56`). That is a reasonable MVP heuristic, but it creates a predictable failure mode: messy citation styles, bracket-only references without parsed bibliography, review sentences with multiple sources, or author/year ambiguity can produce wrong or missing sources.

Third, full-text coverage is your dominant practical bottleneck. The retrieval chain is OA URL -> PMC -> Unpaywall -> abstract fallback (`src/fetch_fulltext.py:25-29`). In the current real-output benchmark, 34 of 60 claims are full-text unavailable (`benchmarks/real_outputs/SUMMARY.md:12`). This is not a small edge case. It is the majority path.

Fourth, BM25 is honest but shallow. The code returns no passages when there is no token overlap (`src/bm25_selector.py:90`). That is better than verifying irrelevant chunks, but it will miss semantic paraphrases, table evidence, figure evidence, and claims whose support depends on methods or supplementary material. SciClaimEval is a useful warning here: the emerging benchmark task focuses on claims supported or refuted by tables and figures across domains, not just plain text passages ([SciClaimEval](https://sciclaimeval.github.io/), lines 20-22 and 40-45). Your current engine is not built for that evidence type.

Fifth, the numeric engine is narrower than its public salience. README says numeric coverage is intentionally limited to OR/CI and p-value/CI null-crossing checks (`README.md:161`). The implementation confirms that OR/CI is tried first and p-value/CI is a fallback (`src/numeric/engine.py:109-113`). Numeric extraction is still LLM-driven (`src/numeric/extract.py:1`, `src/numeric/extract.py:32`); only the comparison step is deterministic (`src/numeric/engine.py:153-156`, `src/numeric/checks.py:1`). The current real benchmark ran 1 numeric check across 60 claims and flagged 0 inconsistencies (`benchmarks/real_outputs/SUMMARY.md:12`). So the numeric engine is a useful seed, not a moat yet.

Sixth, the cost claim is not yet where your own phase target points. The project instruction says target cost is under $0.10 per 2-page document at Phase 0 (`AGENTS.md:133`). Current committed runs cost $0.23 to $0.38 each for three inputs, with total $0.93 (`benchmarks/real_outputs/SUMMARY.md:9-12`). That may be acceptable for high-stakes workflows, but it is not the target and not yet tied to a buyer's willingness to pay.

Seventh, reproducibility is not clean in this environment. I attempted:

- `python -m pytest -q`
- `python -m ruff check src tests scripts examples eval`
- `python -m mypy --strict src scripts examples eval`

All three failed before reaching project code because `python` is not on PATH. I then tried the bundled Codex Python at `C:\Users\Lenovo L14\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe`; it exists, but lacks `pytest`, `ruff`, and `mypy`. That makes the current runtime status source-inspected but not locally verified. This is runtime-unverified, not a test failure.

There is also a public reproducibility claim gap. `CONTRIBUTING.md` says the CI pipeline enforces pytest, ruff, and mypy (`CONTRIBUTING.md:35-40`), but `rg --files .github` fails because `.github` does not exist in this checkout. If CI exists outside this checkout, it is not in the repo. If it does not exist, the contribution doc is false.

## Competitive Context

The problem is real. Scientific claim verification is not just intellectually interesting. Valsci, a 2025 BMC Bioinformatics paper, frames scientific claim verification as a way to reduce hallucinations and unreliable citations in LLM-assisted literature work, and presents an open-source, self-hostable system using retrieval, bibliometric scoring, and structured reports ([Valsci paper](https://link.springer.com/article/10.1186/s12859-025-06159-4), lines 57-67, 110-116). Valsci also claims high-throughput processing, Semantic Scholar grounding, and local/open model compatibility ([Valsci GitHub](https://github.com/bricee98/Valsci), lines 313-319 and 331-343).

That is bad for a vague moat and good for market validation. The category exists. But it means "open-source scientific claim verifier" is not enough. You need a sharper wedge than "auditable reports." Valsci already occupies the large-batch literature verification lane. SciClaimEval points toward multimodal evidence as an active frontier. Your current differentiator is not scale, not UI, not multimodal evidence, and not a proprietary benchmark. Your plausible wedge is cited-source auditing for AI-generated scientific outputs with provenance and explicit abstention. That wedge is narrower than your roadmap language, but more defensible.

The strategic question is whether a narrow cited-source verifier is painful enough for a buyer. The repo does not answer that.

## Market And Adoption Reality

Source-backed absence: I found no customer interview notes, no pilot logs, no design-partner call summaries, no buyer workflow map, no pricing tests, and no competitive landscape document in the tracked repo or the local ignored materials inspected. The local launch one-pager asks for a regulated workflow and weekly feedback calls (`docs/launch_one_pager.local.md:49-55`), but it is an ask, not evidence that anyone wants it.

This matters because the product is not obviously a self-serve developer tool. The likely users have different pain profiles:

- AI-for-science tool teams need QA on generated outputs, but they may prefer internal evals and may not want to surface failures.
- Biotech or clinical teams care about evidence quality, but they need privacy, auditability, compliance posture, and workflow integration.
- Publishers and institutions care about scale and integration, but your own roadmap puts those integrations at Phase 7 (`development-roadmap.md:940-944`).
- Researchers writing literature reviews may want a UI and broad corpus support, where Valsci-style systems already look more complete.

You have not yet proved which buyer has the urgent pain. The current artifacts prove you can build. They do not prove someone will switch workflow, upload sensitive drafts, pay, or tolerate abstention-heavy output.

Judgment: your highest-leverage risk is not another retrieval module. It is discovering whether the narrow cited-source audit is a must-have for one workflow. If the answer is no, more engineering will make the repo more impressive and the company no more real.

## Operator And Roadmap Risk

The git log shows high personal velocity and high reactivity. From April 15 to May 2, the repo went from initial scaffold to Phase 0 MVP, OpenAlex migration, full-text retrieval, numeric engine, real-output benchmarks, launch-prep, post-audit reruns, documentation cleanup, and ignored planning docs. The key sequence:

- `d18aa0a` on 2026-04-15: Phase 0 MVP pipeline.
- `24b3e1f` on 2026-04-16: OpenAlex migration and Edison integration.
- `5d6c695` on 2026-04-28: Phase 1 full-text verification.
- `46a8363` on 2026-04-28: Phase 2 numeric engine MVP.
- `ea888d3` and `d84fc0c` on 2026-04-28/29: real-tool benchmarks.
- `c8aad06` on 2026-05-01: post-audit rerun and fixes.
- `7f58fd7` on 2026-05-01: BM25 token-aware fix.

That pace is impressive as coding output. It is also a warning. In roughly 17 days, you added multiple pipeline phases, benchmarked them, patched audit findings, refreshed docs, then had docs drift again. The May 1 commit `936bab5` removed or decluttered outdated specs with 750 deletions, and `.gitignore` now excludes `development-roadmap.md`, `launch-readiness-audit.md`, `AGENTS.md`, `CLAUDE.md`, `.claude/`, and `decisions/` (`.gitignore:45-50`). Some of that is correct for public hygiene. It also means the public repo loses the reasoning trail that makes the engineering choices intelligible.

The pattern is that you respond to audit findings by rapidly generating artifacts. That closes visible gaps, but it can create new unvalidated claims. Example: the earlier launch audit said the canary should exercise numeric inconsistency and retraction. The canary artifact now exists, but the actual report does not exercise those signals. This is a classic founder failure mode: converting a criticism into a file rather than into a verified fact.

Judgment: you are currently allocating too much energy to publication readiness and too little to falsifying demand. The engineering work is comfortable because it has crisp tests and artifacts. The uncomfortable work is talking to buyers, watching them reject the workflow, and learning which part they actually need.

## What You Probably Do Not Want To Hear

1. You may be building the right evaluation primitive for the wrong initial user.

The engine is strongest when checking cited claims against cited sources. That is not the same as "verify this scientific document" in a buyer's mind. If users expect broad fact checking, uncited claim detection, table/figure verification, whole-literature contradiction search, or regulatory-grade evidence review, your current system will look evasive because it abstains often and by design.

2. Your benchmark does not yet prove usefulness.

The benchmark proves the system can produce honest reports on three hand-collected real outputs. It does not prove recall, precision, buyer value, or operational reliability. The most important number is not 0.94 SciFact F1. It is currently missing: "On a hand-labeled real document, what fraction of claims does the full system correctly extract, resolve, retrieve, and classify?"

3. Your moat is still mostly aspiration.

The roadmap says Phase 6 creates the defensible benchmark dataset and domain models (`development-roadmap.md:892-916`). Today, the moat is discipline: provenance, honest abstention, and careful engineering. Discipline is valuable, but it is not a moat unless it compounds into proprietary data, workflow trust, integrations, or a reputation for catching high-value failures competitors miss.

## Missing Artifacts

- Customer interview notes: needed to prove the pain is real and urgent outside your own reasoning.
- Design-partner commitments: needed because the launch one-pager asks for design partners, but the repo has no evidence of any.
- Buyer workflow map: needed to decide whether this should be a CLI, API, report generator, GitHub-style review bot, manuscript plugin, or batch QA layer.
- Willingness-to-pay or pricing evidence: needed because cost and buyer value are not tied together.
- Competitive landscape document: needed because Valsci and SciClaimEval show this space is active and your wedge must be specific.
- Annotated e2e ground truth: needed because SciFact is verifier-only and the current `eval/e2e` scaffold lacks `reference_paper_v1.json`.
- Current CI workflow: needed because CONTRIBUTING claims CI enforcement, but `.github` is absent in this checkout.
- Canary validation report that actually hits numeric and retraction paths: needed because the canary currently misses two expected demo-critical signals.
- Reproducible benchmark regeneration command and latest generated README numbers: needed because README and SUMMARY already drifted.

## High-Leverage Recommendations

1. Spend one day creating the first real e2e ground truth and run it before any more core features.

Effort: 4-8 hours if you annotate the lactate review yourself, then run `scripts/measure_e2e_recall.py`.

Why this matters more than alternatives: it gives the missing product truth: extraction recall, extraction precision, resolution accuracy, useful e2e coverage, and unknown-cause not-addressed rate (`eval/e2e/README.md:48-50`). Without it, every downstream benchmark discussion is weaker.

Success: a committed or intentionally local `reference_paper_v1.json`, one result JSON, and a README section that says plainly what the full pipeline catches and misses.

2. Freeze public benchmark claims until README, SUMMARY, report JSONs, and canary results are generated from one command.

Effort: 0.5-1 day.

Why this matters: a verification engine cannot have inconsistent public verification numbers. This matters more than adding another source client.

Success: README aggregate equals `benchmarks/real_outputs/SUMMARY.md`; canary expected signals equal actual canary report; the benchmark generator refuses to publish stale mixed-run summaries unless labeled as mixed-run.

3. Run ten buyer conversations before adding Phase 3 semantics.

Effort: 2 weeks calendar time, maybe 10-15 hours.

Why this matters: Phase 3 semantic/ontology work is technically interesting, but it will not answer whether anyone wants this workflow. The binary unknown is market pull, not whether more evidence types can be added.

Success: ten notes with role, current workflow, last painful verification failure, what they would upload, privacy constraints, required output format, budget owner, and a yes/no on a 4-week pilot.

4. Narrow the public wedge.

Effort: 2-3 hours after the e2e baseline.

Why this matters: broad "scientific claim verification" invites comparison to Valsci, literature review tools, fact-checkers, and multimodal benchmarks. Your current system is better described as "cited-source audit for AI-generated scientific text with explicit abstention and provenance."

Success: README and launch one-pager stop implying broad verification and lead with the narrow job, the known limits, and the exact benchmark evidence.

5. Add the missing CI or remove the CI enforcement claim.

Effort: 1-2 hours.

Why this matters: this is a trust hygiene issue. The contribution doc says CI enforces checks, but the repo does not contain `.github` workflows.

Success: either a visible workflow runs pytest/ruff/mypy or CONTRIBUTING says checks are expected locally but not yet enforced in CI.

## Binary Decisions This Month

1. Are you building a product or an open-source research artifact?

If product: buyer interviews outrank Phase 3 engineering. If open-source research artifact: benchmark rigor, reproducibility, and comparison to Valsci/SciClaimEval outrank outreach copy.

2. Is the wedge cited-source auditing or broad scientific verification?

If cited-source auditing: own abstention, citation dependency, and provenance. If broad verification: you need uncited claim detection, cross-source retrieval, tables/figures, and much larger ground truth.

3. Are real-output benchmarks marketing evidence or engineering diagnostics?

If marketing evidence: stale numbers and mixed reruns are unacceptable. If diagnostics: label them as diagnostics and stop using them as launch proof.

4. Will you pause feature expansion until the first hand-labeled e2e baseline exists?

If yes: the next month gets truth. If no: the next month likely gets more impressive code and the same core uncertainty.

## Pre-Mortems

30 days: The repo looks cleaner, the README is updated, and Phase 3 has started, but no buyer has committed to a pilot. The benchmark still has fewer than 100 real claims and no hand-labeled e2e recall. You can explain the system well, but you cannot say who urgently needs it.

6 months: You have a richer verifier, maybe an API, and more benchmark cases. Valsci-like tools and general research assistants improve. Your system remains more careful, but buyers do not switch because the workflow is not embedded where they work. The project becomes a respected technical demo rather than a business.

2 years: The ceiling is a niche open-source verifier used by technical researchers unless you have proprietary evaluation data, institutional integrations, or a high-stakes workflow where audit provenance is mandatory. If you do get those, the project can matter. Without them, "scientific claim verification" becomes a feature inside larger research platforms.

## Falsification Path

The assessment changes if, in the next 30 days, you produce evidence like this:

- A hand-labeled e2e benchmark with at least 60 real claims, plus full-pipeline extraction recall, resolution accuracy, useful coverage, and failure-cause breakdown.
- At least 5 design-partner conversations where 3 or more agree to send real documents weekly for a 4-week pilot.
- One buyer says they would pay or gives a procurement path, not just "interesting."
- A regenerated benchmark suite with 5+ tools, 150+ claims, current README/SUMMARY parity, and canary controls that actually exercise numeric inconsistency and retraction checks.
- One real case where the engine catches a materially important cited-source error that the originating tool or human workflow missed, and the user agrees it would have changed their process.

If those happen, the verdict shifts from "technically promising but product-unvalidated" to "narrow wedge with early market pull." If they do not, the project should stop expanding scope and confront whether it is a research artifact rather than a company.

