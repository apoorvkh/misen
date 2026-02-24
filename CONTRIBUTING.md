# Contributing

## Setup

```bash
uv sync --frozen --all-groups --all-extras
```

## Local Checks

```bash
uv run ruff check .
uv run pytest
```

## Design Notes

- Keep the public API small and explicit (`Task`, `Workspace`, `Executor`, `Experiment`, `@task`).
- Preserve cache/locking semantics when changing internals.
- Update docs when changing behavior that affects execution, hashing, or caching.
