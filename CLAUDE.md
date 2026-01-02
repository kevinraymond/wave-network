# Wave Network Project Guidelines

## Test Coverage

**Always add tests when writing new code.** The project maintains a minimum 70% test coverage threshold.

When adding new features or modifying existing code:
1. Add unit tests in `tests/unit/` for new functionality
2. Run `uv run pytest --cov=models --cov=benchmarks` to check coverage
3. Ensure coverage doesn't drop below 70%

## Development Setup

```bash
uv sync --dev
pre-commit install
```

## Pre-commit Hooks

The project uses pre-commit hooks for `ruff` and `black`. These run automatically on commit.

To run manually:
```bash
uv run ruff check .
uv run black .
```
