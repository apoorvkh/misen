"""Tests for single-file (zip) save/load in :mod:`serde`."""
# ruff: noqa: D103, S101

from __future__ import annotations

import zipfile
from collections import OrderedDict
from typing import TYPE_CHECKING

import pytest

from misen.utils import serde

if TYPE_CHECKING:
    import pathlib

torch = pytest.importorskip("torch")


def test_save_zip_produces_one_file(tmp_path: pathlib.Path) -> None:
    """save_zip leaves exactly one file on disk — no stray temp dirs."""
    zip_path = tmp_path / "checkpoint.zip"
    serde.save_zip({"w": torch.ones(4)}, zip_path)

    # Only the zip exists in tmp_path.
    contents = list(tmp_path.iterdir())
    assert contents == [zip_path]
    assert zip_path.stat().st_size > 0


def test_save_zip_roundtrip_tensor(tmp_path: pathlib.Path) -> None:
    original = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    zip_path = tmp_path / "t.zip"
    serde.save_zip(original, zip_path)

    loaded = serde.load_zip(zip_path)
    assert torch.equal(loaded, original)


def test_save_zip_roundtrip_nested_dict_of_tensors(tmp_path: pathlib.Path) -> None:
    """State_dict-shaped save through the zip wrapper."""
    model = torch.nn.Linear(4, 2)
    sd = model.state_dict()
    zip_path = tmp_path / "sd.zip"
    serde.save_zip(sd, zip_path)

    loaded = serde.load_zip(zip_path)
    assert type(loaded) is OrderedDict
    assert set(loaded) == set(sd)
    for k, v in sd.items():
        assert torch.equal(loaded[k], v)


def test_zip_contents_mirror_directory_layout(tmp_path: pathlib.Path) -> None:
    """Opening the zip should show the same structure ``save`` writes to disk."""
    original = {"weights": torch.ones(4), "lr": 0.001}
    zip_path = tmp_path / "c.zip"
    serde.save_zip(original, zip_path)

    with zipfile.ZipFile(zip_path, "r") as zf:
        names = set(zf.namelist())

    assert "manifest.json" in names
    assert "leaves/torch_tensor/tensors.pt" in names
    # "lr" collapses to a msgpack leaf.
    assert "leaves/msgpack/data.msgpack" in names


def test_save_zip_stored_is_the_default(tmp_path: pathlib.Path) -> None:
    """Default compression is ZIP_STORED — tensor bytes don't compress."""
    zip_path = tmp_path / "stored.zip"
    serde.save_zip({"w": torch.ones(100)}, zip_path)

    with zipfile.ZipFile(zip_path, "r") as zf:
        for info in zf.infolist():
            assert info.compress_type == zipfile.ZIP_STORED


def test_save_zip_deflated_reduces_manifest_size(tmp_path: pathlib.Path) -> None:
    """ZIP_DEFLATED is opt-in and visibly compresses the text bits."""
    original = {"config": {"lr": 0.001, "name": "adam", "layers": [64, 32, 16, 8]}}
    stored_path = tmp_path / "stored.zip"
    deflated_path = tmp_path / "deflated.zip"
    serde.save_zip(original, stored_path)
    serde.save_zip(original, deflated_path, compression=zipfile.ZIP_DEFLATED)

    # Manifest.json repeats keys/whitespace — deflate should shrink it.
    with zipfile.ZipFile(deflated_path, "r") as zf:
        for info in zf.infolist():
            if info.filename == "manifest.json":
                assert info.compress_type == zipfile.ZIP_DEFLATED
                assert info.compress_size < info.file_size
                break
        else:
            pytest.fail("manifest.json not found in deflated zip")


def test_save_zip_overwrites_existing_file(tmp_path: pathlib.Path) -> None:
    """Re-saving to an existing path replaces the file cleanly."""
    zip_path = tmp_path / "x.zip"
    serde.save_zip(torch.ones(2), zip_path)
    serde.save_zip(torch.zeros(5), zip_path)

    loaded = serde.load_zip(zip_path)
    assert torch.equal(loaded, torch.zeros(5))


def test_load_zip_works_on_externally_zipped_save(tmp_path: pathlib.Path) -> None:
    """A zip produced by zipping a ``save`` directory should load too."""
    directory = tmp_path / "out"
    directory.mkdir()
    original = {"w": torch.ones(3), "step": 42}
    serde.save(original, directory)

    zip_path = tmp_path / "manual.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        for p in directory.rglob("*"):
            if p.is_file():
                zf.write(p, arcname=p.relative_to(directory).as_posix())

    loaded = serde.load_zip(zip_path)
    assert torch.equal(loaded["w"], original["w"])
    assert loaded["step"] == 42
