---
icon: lucide/rocket
---

# Getting Started

## Project Setup

Initialize your project as a Python package (recommended):

```bash
uv init my-project --package --python 3.13
cd my-project
uv sync
uv add misen
```

## Define Tasks and an Experiment

```python
from misen import Experiment, Task, task


@task(id="load", cache=True)
def load(n: int) -> list[int]:
    return list(range(n))


@task(id="sum", cache=True)
def sum_values(values: list[int]) -> int:
    return sum(values)


class MyExperiment(Experiment):
    n: int = 100

    def tasks(self) -> dict[str, Task[int]]:
        values = Task(load, self.n)
        total = Task(sum_values, values.T)
        return {"total": total}


if __name__ == "__main__":
    MyExperiment.cli()
```

## Run It

```bash
uv run -m my_project.experiment
```

or:

```bash
misen experiment my_project.experiment:MyExperiment run
```

or:

```python
MyExperiment(n=1000).run()
```

## Optional Configuration (`misen.toml`)

`"auto"` workspace/executor resolution uses `misen.toml` in the working dir.

```toml
executor_type = "local"
executor_kwargs = { num_cpus = "all", max_memory = "all" }

workspace_type = "disk"
workspace_kwargs = { directory = ".misen" }
```

## Fill Missing Task IDs

If you have legacy `@task` decorators missing `id`, you can auto-fill them:

```bash
misen fill
```

Pass one or more positional paths (Python files and/or project directories):

```bash
misen fill src/my_project scripts/fixup.py
```

## Mental Model

- Build tasks lazily with `Task(...)`.
- Cache behavior is controlled on `@task(...)`.
- `Workspace` stores hashes/results and locks.
- `Executor` schedules work units derived from cache boundaries.
