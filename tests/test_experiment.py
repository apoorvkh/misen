from collections.abc import Mapping

from misen import Experiment, Task, meta


@meta(id="test_experiment_noop", cache=False)
def _noop(x: int) -> int:
    return x


class _KeyedExperiment(Experiment):
    n: int = 3

    def tasks(self) -> dict[str, Task[int]]:
        return {str(i): Task(_noop, x=i) for i in range(self.n)}


class _UnkeyedSetExperiment(Experiment):
    n: int = 3

    def tasks(self) -> set[Task[int]]:
        return {Task(_noop, x=i) for i in range(self.n)}


class _UnkeyedListExperiment(Experiment):
    n: int = 3

    def tasks(self) -> list[Task[int]]:
        return [Task(_noop, x=i) for i in range(self.n)]


def test_keyed_tasks_returned_verbatim() -> None:
    exp = _KeyedExperiment(n=3)
    result = exp.tasks()
    assert isinstance(result, Mapping)
    assert set(result) == {"0", "1", "2"}


def test_unkeyed_set_tasks_returned_verbatim() -> None:
    exp = _UnkeyedSetExperiment(n=3)
    result = exp.tasks()
    assert isinstance(result, set)
    assert len(result) == 3


def test_unkeyed_list_tasks_returned_verbatim() -> None:
    exp = _UnkeyedListExperiment(n=3)
    result = exp.tasks()
    assert isinstance(result, list)
    assert len(result) == 3


def test_unkeyed_tasks_normalized_to_mapping() -> None:
    exp = _UnkeyedSetExperiment(n=3)
    normalized = exp.normalized_tasks()
    assert isinstance(normalized, Mapping)
    assert len(normalized) == 3
    for key, task in normalized.items():
        assert isinstance(task, Task)
        assert task.task_hash().b32() == key


def test_keyed_normalized_tasks_matches_raw() -> None:
    exp = _KeyedExperiment(n=3)
    raw = exp.tasks()
    normalized = exp.normalized_tasks()
    assert normalized is raw


def test_tasks_caches_across_calls() -> None:
    exp = _UnkeyedSetExperiment(n=3)
    first = exp.tasks()
    second = exp.tasks()
    assert first is second


def test_unkeyed_experiment_subscript_by_synthesized_key() -> None:
    exp = _UnkeyedSetExperiment(n=2)
    normalized = exp.normalized_tasks()
    for key, task in normalized.items():
        assert exp[key] is task
