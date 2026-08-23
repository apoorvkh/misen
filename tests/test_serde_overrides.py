"""Tests for :func:`serde.save` / :func:`serde.load` override + manifest integrity.

Covers the ``ser_cls`` override used by the workspace's per-task
serializer contract, plus the manifest-version sanity check.
"""
# ruff: noqa: D103, S101

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any
from unittest.mock import Mock

import msgspec
import pytest

from misen.exceptions import SerializationError
from misen.utils import serde

if TYPE_CHECKING:
    from collections.abc import Mapping


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


def test_save_wraps_manifest_write_error_with_cause(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_write_text = Path.write_text

    def write_text(
        path: Path,
        data: str,
        encoding: str | None = None,
        errors: str | None = None,
        newline: str | None = None,
    ) -> int:
        if path.name == serde.MANIFEST_FILENAME:
            msg = "disk unavailable"
            raise OSError(msg)
        return real_write_text(path, data, encoding=encoding, errors=errors, newline=newline)

    monkeypatch.setattr(Path, "write_text", write_text)

    with pytest.raises(SerializationError, match=r"Could not write manifest\.json") as exc_info:
        serde.save(12345, tmp_path)

    assert isinstance(exc_info.value.__cause__, OSError)


def test_save_wraps_manifest_json_encoding_error_with_cause(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def write_with_invalid_meta(obj: Any, directory: Path) -> Mapping[str, Any]:  # noqa: ARG001
        return {"invalid": object()}

    monkeypatch.setattr(CustomTextSerializer, "write", staticmethod(write_with_invalid_meta))

    with pytest.raises(SerializationError, match=r"Could not encode manifest\.json") as exc_info:
        serde.save("payload", tmp_path, ser_cls=CustomTextSerializer)

    assert isinstance(exc_info.value.__cause__, TypeError)


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


def test_save_writes_manifest_version_1(tmp_path: Path) -> None:
    serde.save({"hi": 1}, tmp_path)

    manifest_path = tmp_path / serde.MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text())
    assert manifest["version"] == 1


def test_load_rejects_manifest_node_missing_node_id(tmp_path: Path) -> None:
    serde.save({"hi": 1}, tmp_path)

    manifest_path = tmp_path / serde.MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text())
    del manifest["root"]["node_id"]
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(SerializationError, match="node_id"):
        serde.load(tmp_path)


def test_load_rejects_missing_manifest_file(tmp_path: Path) -> None:
    """No manifest → clean :class:`SerializationError`, not bare FileNotFoundError."""
    with pytest.raises(SerializationError, match=r"No manifest\.json") as exc_info:
        serde.load(tmp_path)

    assert isinstance(exc_info.value.__cause__, FileNotFoundError)


def test_load_rejects_malformed_manifest_json_with_cause(tmp_path: Path) -> None:
    (tmp_path / serde.MANIFEST_FILENAME).write_text("{not-json", encoding="utf-8")

    with pytest.raises(SerializationError, match=r"Could not read manifest\.json") as exc_info:
        serde.load(tmp_path)

    assert isinstance(exc_info.value.__cause__, json.JSONDecodeError)


def test_load_rejects_non_object_manifest(tmp_path: Path) -> None:
    (tmp_path / serde.MANIFEST_FILENAME).write_text("[]", encoding="utf-8")

    with pytest.raises(SerializationError, match="must contain a JSON object"):
        serde.load(tmp_path)


def test_load_rejects_missing_manifest_root_with_cause(tmp_path: Path) -> None:
    manifest = {"version": 1, "leaves": {}, "dirs": {}}
    (tmp_path / serde.MANIFEST_FILENAME).write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(SerializationError, match="missing required 'root'") as exc_info:
        serde.load(tmp_path)

    assert isinstance(exc_info.value.__cause__, KeyError)


def test_load_wraps_corrupt_msgpack_data(tmp_path: Path) -> None:
    serde.save({"value": 1}, tmp_path)
    (tmp_path / "leaves" / "msgpack" / "data.msgpack").write_bytes(b"\xc1")

    with pytest.raises(SerializationError, match="persisted data") as exc_info:
        serde.load(tmp_path)

    assert isinstance(exc_info.value.__cause__, msgspec.DecodeError)


def test_load_wraps_missing_msgpack_leaf(tmp_path: Path) -> None:
    serde.save(123, tmp_path)
    (tmp_path / "leaves" / "msgpack" / "data.msgpack").write_bytes(msgspec.msgpack.encode({}))

    with pytest.raises(SerializationError, match="does not contain leaf") as exc_info:
        serde.load(tmp_path)

    assert isinstance(exc_info.value.__cause__, KeyError)


def test_load_wraps_corrupt_numpy_bundle(tmp_path: Path) -> None:
    import numpy as np

    serde.save(np.arange(3), tmp_path)
    (tmp_path / "leaves" / "ndarray" / "arrays.npz").write_bytes(b"not-a-zip")

    with pytest.raises(SerializationError, match="NumPy array bundle") as exc_info:
        serde.load(tmp_path)

    assert isinstance(exc_info.value.__cause__, ValueError)


def test_numpy_scalar_binary_payload_roundtrips_non_msgpack_values(tmp_path: Path) -> None:
    import numpy as np

    structured_array = np.zeros((), dtype=[("count", "<i4"), ("coords", "<f8", (2,))])
    structured_array["count"] = 7
    structured_array["coords"] = (1.5, 2.5)
    values = [
        np.complex64(1 + 2j),
        np.clongdouble(3 - 4j),
        np.datetime64("2025-01-02T03:04:05", "ms"),
        np.timedelta64(123, "us"),
        np.str_("misen"),
        structured_array[()],
    ]

    for index, original in enumerate(values):
        directory = tmp_path / str(index)
        serde.save(original, directory)
        loaded = serde.load(directory)

        assert type(loaded) is type(original)
        assert loaded.dtype == original.dtype
        np.testing.assert_equal(loaded, original)


def test_numpy_scalar_reader_accepts_legacy_value_payload(tmp_path: Path) -> None:
    import numpy as np

    serde.save(np.float32(1.5), tmp_path)
    blob_path = tmp_path / "leaves" / "numpy_scalar" / "scalars.msgpack"
    blob = msgspec.msgpack.decode(blob_path.read_bytes())
    leaf_id = next(iter(blob))
    blob[leaf_id] = {"dtype": "<f4", "value": 1.5}
    blob_path.write_bytes(msgspec.msgpack.encode(blob))

    loaded = serde.load(tmp_path)
    assert type(loaded) is np.float32
    assert loaded == np.float32(1.5)


def test_load_wraps_corrupt_numpy_scalar_bundle(tmp_path: Path) -> None:
    import numpy as np

    serde.save(np.complex64(1 + 2j), tmp_path)
    (tmp_path / "leaves" / "numpy_scalar" / "scalars.msgpack").write_bytes(b"\xc1")

    with pytest.raises(SerializationError, match="NumPy scalar bundle") as exc_info:
        serde.load(tmp_path)

    assert isinstance(exc_info.value.__cause__, msgspec.DecodeError)


def test_save_wraps_numpy_array_write_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import numpy as np

    def fail_savez(*_args: object, **_kwargs: object) -> None:
        msg = "array storage unavailable"
        raise OSError(msg)

    monkeypatch.setattr(np, "savez", fail_savez)

    with pytest.raises(SerializationError, match="NumPy array bundle") as exc_info:
        serde.save(np.arange(3), tmp_path)

    assert isinstance(exc_info.value.__cause__, OSError)


def test_save_rejects_numpy_object_array_before_writing_bundle(tmp_path: Path) -> None:
    import numpy as np

    with pytest.raises(SerializationError, match="object-containing dtype"):
        serde.save(np.array([object()], dtype=object), tmp_path)

    assert not (tmp_path / "leaves" / "ndarray" / "arrays.npz").exists()


def test_save_wraps_numpy_masked_array_write_failure(tmp_path: Path) -> None:
    import numpy as np

    original = np.ma.MaskedArray(np.array([object()], dtype=object), mask=[False])

    with pytest.raises(SerializationError, match="NumPy masked array") as exc_info:
        serde.save(original, tmp_path)

    assert isinstance(exc_info.value.__cause__, ValueError)


def test_jax_array_roundtrip_uses_real_array_implementation(tmp_path: Path) -> None:
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    np = pytest.importorskip("numpy")
    original = jnp.arange(6).reshape(2, 3)

    serde.save(original, tmp_path)
    loaded = serde.load(tmp_path)

    assert isinstance(loaded, jax.Array)
    np.testing.assert_array_equal(np.asarray(loaded), np.asarray(original))


def test_tensorflow_symbolic_tensor_conversion_error_is_wrapped(tmp_path: Path) -> None:
    tf = pytest.importorskip("tensorflow")
    with tf.Graph().as_default():
        symbolic = tf.constant([1, 2, 3])

    with pytest.raises(SerializationError, match="Could not convert TensorFlow tensor") as exc_info:
        serde.save(symbolic, tmp_path)

    assert isinstance(exc_info.value.__cause__, AttributeError)


def test_tensorflow_op_error_is_wrapped() -> None:
    tf = pytest.importorskip("tensorflow")
    from misen.utils.serde.libs.tensorflow import TensorFlowTensorSerializer

    backend_error = tf.errors.InvalidArgumentError(None, None, "device transfer failed")
    tensor = Mock()
    tensor.numpy.side_effect = backend_error

    with pytest.raises(SerializationError, match="Could not convert TensorFlow tensor") as exc_info:
        TensorFlowTensorSerializer.to_payload(tensor)

    assert exc_info.value.__cause__ is backend_error


def test_load_does_not_double_wrap_serialization_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    serde.save("payload", tmp_path, ser_cls=CustomTextSerializer)
    expected = SerializationError("custom serializer failure")

    def fail_read(directory: Path, *, meta: Mapping[str, Any]) -> Any:  # noqa: ARG001
        raise expected

    monkeypatch.setattr(CustomTextSerializer, "read", staticmethod(fail_read))

    with pytest.raises(SerializationError) as exc_info:
        serde.load(tmp_path, ser_cls=CustomTextSerializer)

    assert exc_info.value is expected


def test_load_does_not_wrap_programmer_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    serde.save("payload", tmp_path, ser_cls=CustomTextSerializer)

    def fail_read(directory: Path, *, meta: Mapping[str, Any]) -> Any:  # noqa: ARG001
        msg = "serializer bug"
        raise AssertionError(msg)

    monkeypatch.setattr(CustomTextSerializer, "read", staticmethod(fail_read))

    with pytest.raises(AssertionError, match="serializer bug"):
        serde.load(tmp_path, ser_cls=CustomTextSerializer)


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


def test_load_rejects_invalid_dict_container_type_metadata(tmp_path: Path) -> None:
    import numpy as np

    serde.save({"values": np.arange(3)}, tmp_path)
    manifest_path = tmp_path / serde.MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text())
    manifest["root"]["meta"]["type"] = "SortedDict"
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(SerializationError, match="invalid 'type' metadata"):
        serde.load(tmp_path)


def test_load_rejects_unresolvable_defaultdict_factory_metadata(tmp_path: Path) -> None:
    import numpy as np

    serde.save(defaultdict(list, {"values": np.arange(3)}), tmp_path)
    manifest_path = tmp_path / serde.MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text())
    manifest["root"]["meta"]["factory"] = "missing.module.factory"
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(SerializationError, match="default_factory") as exc_info:
        serde.load(tmp_path)

    assert isinstance(exc_info.value.__cause__, ImportError)


def test_load_rejects_dangling_ref_with_serialization_error(tmp_path: Path) -> None:
    serde.save([1, 2, 3], tmp_path)

    manifest_path = tmp_path / serde.MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text())
    manifest["root"] = {"_t": "ref", "target": "missing"}
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(SerializationError, match="forward reference"):
        serde.load(tmp_path)
