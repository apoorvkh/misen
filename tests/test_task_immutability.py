import pytest

from misen.task import Task, task

cloudpickle = pytest.importorskip("cloudpickle")


@task(id="identity", cache=False)
def identity(x: int) -> int:
    return x


def test_task_is_immutable() -> None:
    t = Task(identity, x=1)

    with pytest.raises(AttributeError, match="immutable"):
        t.func = identity

    with pytest.raises(TypeError):
        t.kwargs["x"] = 2


def test_task_cloudpickle_roundtrip_preserves_immutability() -> None:
    t = Task(identity, x=1)
    restored = cloudpickle.loads(cloudpickle.dumps(t))

    assert restored == t
    assert restored.kwargs["x"] == 1

    with pytest.raises(AttributeError, match="immutable"):
        restored.func = identity
