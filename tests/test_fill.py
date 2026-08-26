"""Tests for placeholder ``@meta(id=...)`` handling in ``misen fill`` and ``meta``."""
# ruff: noqa: D103, S101

from pathlib import Path

import pytest

import misen.utils.cli.fill as fill_module
from misen import Task, meta
from misen.utils.cli.fill import fill_paths_task_ids, fill_task_ids_in_source


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


def test_fill_generates_twelve_character_mixed_case_crockford_style_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected_alphabet = "0123456789ABCDEFGHJKMNPQRSTVWXYZabcdefghjkmnpqrstvwxyz"
    generated_characters = iter("0AaZz9KkPpXx")

    def deterministic_choice(alphabet: str) -> str:
        assert alphabet == expected_alphabet
        return next(generated_characters)

    monkeypatch.setattr(fill_module.secrets, "choice", deterministic_choice)
    updated, replacements = fill_task_ids_in_source("@meta(cache=True)\ndef f():\n    pass\n")

    assert replacements == 1
    assert updated == '@meta(id="0AaZz9KkPpXx", cache=True)\ndef f():\n    pass\n'


def test_fill_reports_write_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source_path = tmp_path / "task.py"
    source_path.write_text("@meta(cache=True)\ndef f():\n    pass\n", encoding="utf-8")
    real_write_text = Path.write_text

    def write_text(path: Path, *args: object, **kwargs: object) -> int:
        if path == source_path:
            msg = "read-only source"
            raise PermissionError(msg)
        return real_write_text(path, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(Path, "write_text", write_text)
    report = fill_paths_task_ids([source_path], uuid_factory=_fixed_uuid)

    assert report.changed_files == 0
    assert report.failed_files == [(source_path, "read-only source")]


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
