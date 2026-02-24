# misen

`misen` is a cache-aware DAG execution framework for Python workloads.

It keeps the user-facing API intentionally small:

- `@task(...)` declares task identity, caching, and resources.
- `Task(fn, ...)` builds a lazy dependency graph.
- `Workspace` stores results/hashes and coordinates locks.
- `Executor` runs cache-bounded work units.
- `Experiment` groups named tasks and exposes a CLI.

## Installation

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
source .venv/bin/activate
```

## Quick Example

```python
from misen import Experiment, Task, task


@task(id="load", cache=True)
def load_data(n: int) -> list[int]:
    return list(range(n))


@task(id="sum", cache=True)
def sum_data(values: list[int]) -> int:
    return sum(values)


class Demo(Experiment):
    n: int = 100

    def tasks(self) -> dict[str, Task[int]]:
        values = Task(load_data, self.n)
        total = Task(sum_data, values.T)
        return {"total": total}


if __name__ == "__main__":
    Demo(n=1000).run()
```

## Design Invariants

- Task identity is deterministic (`task_hash`, `resolved_hash`, `result_hash`).
- Cache boundaries define scheduling boundaries (`WorkUnit` graph).
- Cacheable task runtimes are serialized per resolved hash in a workspace.
- Execution backends only schedule work; storage/locking stays in `Workspace`.

## More Docs

- [Architecture and design](docs/index.md)
- [Getting started guide](docs/getting_started.md)
- [API reference](docs/api.md)
