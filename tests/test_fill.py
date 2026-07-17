"""Tests for placeholder ``@meta(id=...)`` handling in ``misen fill`` and ``meta``."""
# ruff: noqa: D103, S101

import pytest

from misen import meta
from misen.utils.cli.fill import fill_task_ids_in_source


def _fixed_uuid() -> str:
    return "FILLEDID00"


@pytest.mark.parametrize("placeholder", ["id=None", 'id=""', "id=''"])
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


@pytest.mark.parametrize("id_value", [None, ""])
def test_meta_rejects_placeholder_ids(id_value: str | None) -> None:
    # id=None is type-invalid since the signature narrowed to `id: str`, but untyped
    # legacy callers must still get the clean "id must be provided" error at runtime.
    with pytest.raises(ValueError, match="id must be provided"):
        meta(id=id_value)  # ty:ignore[invalid-argument-type]
