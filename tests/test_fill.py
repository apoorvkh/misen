"""Tests for placeholder ``@meta(id=...)`` handling in ``misen fill`` and ``meta``."""
# ruff: noqa: D103, S101

import pytest

from misen import Task, meta
from misen.utils.cli.fill import fill_task_ids_in_source


def _fixed_uuid() -> str:
    return "FILLEDID00"


@pytest.mark.parametrize("placeholder", ['id=""', "id=''"])
def test_fill_rewrites_placeholder_ids(placeholder: str) -> None:
    source = f"@meta({placeholder}, cache=True)\ndef f():\n    pass\n"
    updated, replacements = fill_task_ids_in_source(source, uuid_factory=_fixed_uuid)
    assert replacements == 1
    assert updated == '@meta(id="FILLEDID00", cache=True)\ndef f():\n    pass\n'


def test_fill_inserts_missing_id() -> None:
    source = "@meta(cache=True)\ndef f():\n    pass\n"
    updated, replacements = fill_task_ids_in_source(source, uuid_factory=_fixed_uuid)
    assert replacements == 1
    assert updated == '@meta(id="FILLEDID00", cache=True)\ndef f():\n    pass\n'


def test_fill_preserves_existing_id() -> None:
    source = '@meta(id="KEEPME", cache=True)\ndef f():\n    pass\n'
    updated, replacements = fill_task_ids_in_source(source, uuid_factory=_fixed_uuid)
    assert replacements == 0
    assert updated == source


def test_meta_defers_empty_id_validation_until_task() -> None:
    @meta(id="")
    def placeholder_task() -> int:
        return 1

    assert placeholder_task() == 1
    with pytest.raises(ValueError, match=r"placeholder_task has no task id.*misen fill"):
        Task(placeholder_task)


def test_fill_does_not_treat_none_as_a_placeholder() -> None:
    source = "@meta(id=None, cache=True)\ndef f():\n    pass\n"
    updated, replacements = fill_task_ids_in_source(source, uuid_factory=_fixed_uuid)
    assert replacements == 0
    assert updated == source


def test_meta_defers_omitted_id_validation_until_task() -> None:
    @meta(cache=True)
    def placeholder_task() -> int:
        return 1

    assert placeholder_task() == 1
    with pytest.raises(ValueError, match=r"placeholder_task has no task id.*misen fill"):
        Task(placeholder_task)
