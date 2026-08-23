# Contributing

## Setup

```bash
uv sync --frozen --all-groups --all-extras
```

## Local Checks

```bash
uv run ruff check src
uv run ruff format --check .
uv run ty check src
uv run pytest
```

## Design Notes

- Keep the public API small and explicit (`Task`, `Workspace`, `Executor`, `Experiment`, `@meta`).
- Preserve cache/locking semantics when changing internals.
- Update docs when changing behavior that affects execution, hashing, or caching.

## Exception Policy

- Use Python's built-in exceptions for ordinary API and protocol contracts:
  `TypeError`, `ValueError`, `KeyError`, and `FileNotFoundError` should keep
  their conventional meanings.
- Use a specific `MisenError` subclass for expected failures owned by a Misen
  subsystem and exposed across its public boundary.
- Translate external-library errors narrowly with
  `raise DomainError(...) from exc`; do not blanket-wrap unexpected bugs or
  exceptions raised by user task functions.
- Render errors only at a CLI or worker boundary. Library code should preserve
  the original traceback instead of printing it.
- Document exceptions propagated by public APIs in a Google-style `Raises:`
  section, and add a contract test for the exception type and cause chain.
- Every broad exception handler must re-raise, record a durable failure, or be
  explicitly identified as best-effort cleanup.
