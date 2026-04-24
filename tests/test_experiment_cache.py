from __future__ import annotations

import gc
import weakref

import misen.experiment as experiment_module
from misen import Experiment, Task, meta


@meta(id="weak_cache_task", cache=False)
def weak_cache_task(value: int) -> int:
    return value


_calls: list[int] = []


class WeakCacheExperiment(Experiment):
    value: int = 1

    def tasks(self) -> dict[str, Task[int]]:
        _calls.append(self.value)
        return {"task": Task(weak_cache_task, value=self.value)}


def test_experiment_tasks_cache_uses_weak_keys() -> None:
    experiment_module._TASKS_CACHE.clear()  # noqa: SLF001
    _calls.clear()

    experiment = WeakCacheExperiment(value=7)
    experiment_ref = weakref.ref(experiment)

    first = experiment.tasks()
    second = experiment.tasks()

    assert first is second
    assert _calls == [7]
    assert experiment in experiment_module._TASKS_CACHE  # noqa: SLF001

    del experiment, first, second
    gc.collect()

    assert experiment_ref() is None
