from __future__ import annotations

import gc
import weakref

import misen.experiment as experiment_module
from misen import Experiment, Task, meta


@meta(id="weak_cache_task", cache=False)
def weak_cache_task(value: int) -> int:
    return value


_calls: list[int] = []


class CacheExperiment(Experiment):
    value: int = 1

    def tasks(self) -> dict[str, Task[int]]:
        _calls.append(self.value)
        return {"task": Task(weak_cache_task, value=self.value)}


class ListFieldExperiment(Experiment):
    values: list[int] = []  # noqa: RUF012

    def tasks(self) -> dict[str, Task[int]]:
        _calls.append(sum(self.values))
        return {str(v): Task(weak_cache_task, value=v) for v in self.values}


def _clear_cache() -> None:
    experiment_module._HASH_BY_ID.clear()  # noqa: SLF001
    experiment_module._TASKS_BY_HASH.clear()  # noqa: SLF001


def test_experiment_tasks_cache_same_instance() -> None:
    _clear_cache()
    _calls.clear()

    experiment = CacheExperiment(value=7)
    first = experiment.tasks()
    second = experiment.tasks()

    assert first is second
    assert _calls == [7]


def test_experiment_tasks_cache_shared_across_equal_instances() -> None:
    _clear_cache()
    _calls.clear()

    first = CacheExperiment(value=7).tasks()
    second = CacheExperiment(value=7).tasks()
    third = CacheExperiment(value=8).tasks()

    assert first is second
    assert third is not first
    assert _calls == [7, 8]


def test_experiment_tasks_cache_drops_id_entry_on_gc() -> None:
    _clear_cache()

    experiment = CacheExperiment(value=7)
    experiment_ref = weakref.ref(experiment)
    experiment.tasks()

    assert id(experiment) in experiment_module._HASH_BY_ID  # noqa: SLF001
    experiment_id = id(experiment)

    del experiment
    gc.collect()

    assert experiment_ref() is None
    assert experiment_id not in experiment_module._HASH_BY_ID  # noqa: SLF001


def test_experiment_tasks_cache_handles_unhashable_fields() -> None:
    _clear_cache()
    _calls.clear()

    first = ListFieldExperiment(values=[1, 2, 3]).tasks()
    second = ListFieldExperiment(values=[1, 2, 3]).tasks()

    assert first is second
    assert _calls == [6]
