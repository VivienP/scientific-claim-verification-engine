# Contributing

## Prerequisites

- Python 3.12+
- `ANTHROPIC_API_KEY` — required for integration tests and the example script
- `SEMANTIC_SCHOLAR_API_KEY` — optional; increases rate limits on citation resolution

## Dev setup

```bash
git clone https://github.com/VivienP/scientific-claim-verification-engine
cd scientific-claim-verification-engine
pip install -e ".[dev]"
cp .env.example .env   # then fill in your API key
```

## Running tests

Unit tests run fully offline (all external APIs are mocked):

```bash
python -m pytest tests/unit/ -v
```

Integration tests hit real APIs and require credentials in `.env`:

```bash
python -m pytest tests/integration/ -v --run-integration
```

## Code style

```bash
python -m ruff format src/ tests/ scripts/
python -m ruff check src/ tests/ scripts/
python -m mypy --strict src/
```

All three must pass clean before a PR is accepted. The CI pipeline enforces this.

## Submitting a PR

1. Fork the repo and create a branch from `main`.
2. Write or update tests for any changed behaviour.
3. Run the full unit test suite (`pytest tests/unit/ -v`) — must pass.
4. Run `ruff` and `mypy` — must be clean.
5. Open a PR with a clear description of what changed and why.

If you're adding a new pipeline module, read [docs/specs/mvp-pipeline.md](docs/specs/mvp-pipeline.md) first — it documents the data model and provenance logging contract that all modules must respect.

## Reporting bugs

Open a [GitHub Issue](https://github.com/VivienP/scientific-claim-verification-engine/issues) with:

- A minimal reproducible example (input text + command)
- The full error output or unexpected `report.json` content
- Your Python version and OS

Thank you for helping.
