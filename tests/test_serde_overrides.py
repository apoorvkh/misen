"""Tests for :func:`serde.save` / :func:`serde.load` override + manifest integrity.

Covers the ``ser_cls`` override used by the workspace's per-task
serializer contract, plus the manifest-version sanity check.
"""
# ruff: noqa: D103, S101

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from misen.exceptions import SerializationError
from misen.utils import serde


# ---------------------------------------------------------------------------
# A custom Serializer that's NOT registered in the default Registry.
# ---------------------------------------------------------------------------
#
# It has to be module-level so ``type(obj).__qualname__`` resolves cleanly
# when the manifest records the serializer name.


class CustomTextSerializer(serde.Serializer[Any]):
    """Minimal text-file serializer used only via explicit ``ser_cls=``."""

    @staticmethod
    def match(obj: Any) -> bool:  # noqa: ARG004
        # Never match via auto-dispatch — callers must pass ser_cls=
        return False

    @staticmethod
    def write(obj: Any, directory: Path) -> Mapping[str, Any] | None:
        (directory / "data.txt").write_text(str(obj), encoding="utf-8")
        return {"original_type": type(obj).__name__}

    @staticmethod
    def read(directory: Path, *, meta: Mapping[str, Any]) -> Any:
        text = (directory / "data.txt").read_text(encoding="utf-8")
        return {"text": text, "original_type": meta["original_type"]}


# ---------------------------------------------------------------------------
# ser_cls override — the workspace.py contract
# ---------------------------------------------------------------------------


def test_save_ser_cls_override_uses_custom_serializer(tmp_path: Path) -> None:
    """Explicit ``ser_cls=`` bypasses registry dispatch at the root."""
    serde.save(12345, tmp_path, ser_cls=CustomTextSerializer)

    # Should produce the custom serializer's on-disk layout, not the default msgpack leaf.
    manifest = json.loads((tmp_path / serde.MANIFEST_FILENAME).read_text())
    assert manifest["root"]["_t"] == "dir"
    assert manifest["root"]["serializer"].endswith(".CustomTextSerializer")


def test_load_ser_cls_override_uses_custom_serializer(tmp_path: Path) -> None:
    """Explicit ``ser_cls=`` on load overrides the manifest's recorded serializer."""
    serde.save(12345, tmp_path, ser_cls=CustomTextSerializer)

    loaded = serde.load(tmp_path, ser_cls=CustomTextSerializer)
    assert loaded == {"text": "12345", "original_type": "int"}


def test_save_without_ser_cls_uses_registry_dispatch(tmp_path: Path) -> None:
    """Sanity check: the same value without ``ser_cls`` routes via the registry."""
    serde.save(12345, tmp_path)

    manifest = json.loads((tmp_path / serde.MANIFEST_FILENAME).read_text())
    assert manifest["root"]["_t"] == "leaf"
    assert manifest["root"]["serializer"].endswith(".MsgpackLeafSerializer")


def test_load_ser_cls_override_unregistered_class_still_works(tmp_path: Path) -> None:
    """A Serializer that isn't in the registry can still drive load via ser_cls.

    Regression guard on the ``test_task_hashing.py`` workflow where a
    user-defined ``DillSerializer`` is never added to
    ``all_serializers``; only passed as ``ser_cls``.
    """
    serde.save([1, 2, 3], tmp_path, ser_cls=CustomTextSerializer)
    # Replace the class name in the manifest with one that isn't registered,
    # to prove the load path doesn't depend on by_name lookup when ser_cls is given.
    manifest = json.loads((tmp_path / serde.MANIFEST_FILENAME).read_text())
    manifest["root"]["serializer"] = "nonexistent.module.SomeSerializer"
    (tmp_path / serde.MANIFEST_FILENAME).write_text(json.dumps(manifest))

    loaded = serde.load(tmp_path, ser_cls=CustomTextSerializer)
    assert loaded["text"] == "[1, 2, 3]"


# ---------------------------------------------------------------------------
# Manifest version check
# ---------------------------------------------------------------------------


def test_load_rejects_unknown_manifest_version(tmp_path: Path) -> None:
    """An unknown manifest version raises a clean ``SerializationError``."""
    serde.save({"hi": 1}, tmp_path)

    manifest_path = tmp_path / serde.MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text())
    manifest["version"] = 999
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(SerializationError, match="Unsupported .* version 999"):
        serde.load(tmp_path)


def test_load_rejects_manifest_with_missing_version(tmp_path: Path) -> None:
    """A manifest that predates versioning also fails cleanly."""
    serde.save({"hi": 1}, tmp_path)

    manifest_path = tmp_path / serde.MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text())
    del manifest["version"]
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(SerializationError, match="Unsupported"):
        serde.load(tmp_path)


def test_load_accepts_v1_manifest(tmp_path: Path) -> None:
    """Version 1 manifests predate graph refs but remain readable."""
    serde.save({"hi": 1}, tmp_path)

    manifest_path = tmp_path / serde.MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text())
    manifest["version"] = 1
    manifest["root"].pop("node_id", None)
    manifest_path.write_text(json.dumps(manifest))

    assert serde.load(tmp_path) == {"hi": 1}


def test_load_rejects_missing_manifest_file(tmp_path: Path) -> None:
    """No manifest → clean :class:`SerializationError`, not bare FileNotFoundError."""
    with pytest.raises(SerializationError, match="No manifest.json"):
        serde.load(tmp_path)


def test_duplicate_leaf_kind_owners_raise_clean_error() -> None:
    class OwnerA(serde.LeafSerializer[Any]):
        leaf_kind = "duplicate"

    class OwnerB(serde.LeafSerializer[Any]):
        leaf_kind = "duplicate"

    ctx = serde.EncodeCtx(registry=None)
    ctx.add_leaf(OwnerA, "duplicate", "a")
    with pytest.raises(SerializationError, match="already owned"):
        ctx.add_leaf(OwnerB, "duplicate", "b")


def test_load_rejects_wrong_node_shape_with_serialization_error(tmp_path: Path) -> None:
    serde.save([1, 2, 3], tmp_path)

    manifest_path = tmp_path / serde.MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text())
    manifest["root"]["serializer"] = "misen.utils.serde.libs.stdlib.ListSerializer"
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(SerializationError, match="expected a Container node"):
        serde.load(tmp_path)


def test_load_rejects_dangling_ref_with_serialization_error(tmp_path: Path) -> None:
    serde.save([1, 2, 3], tmp_path)

    manifest_path = tmp_path / serde.MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text())
    manifest["root"] = {"_t": "ref", "target": "missing"}
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(SerializationError, match="forward reference"):
        serde.load(tmp_path)
