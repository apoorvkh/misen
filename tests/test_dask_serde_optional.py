"""Regression tests for Dask serializers with optional collection dependencies."""
# ruff: noqa: S101

from __future__ import annotations

import builtins
import subprocess
import sys
from typing import TYPE_CHECKING, Any

import pytest

from misen.utils import serde
from misen.utils.serde.libs.dask import dask_serializers
from misen.utils.serde.libs.stdlib import MsgpackLeafSerializer
from misen.utils.serde.registry import Registry

if TYPE_CHECKING:
    import pathlib


def test_unrelated_roundtrip_does_not_import_dask_collection_submodules(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Dask match predicates remain safe when array/dataframe/bag extras are absent."""
    pytest.importorskip("dask")
    real_import = builtins.__import__

    def guarded_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name.startswith(("dask.array", "dask.bag", "dask.dataframe")):
            msg = f"Registry lookup unexpectedly imported optional submodule {name!r}"
            raise AssertionError(msg)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    # Put the Dask serializers first so every match predicate is exercised
    # before this unrelated value reaches its actual serializer.
    registry = Registry([*dask_serializers, MsgpackLeafSerializer])
    original = {"answer": 42}
    serde.save(original, tmp_path, registry=registry)
    assert serde.load(tmp_path, registry=registry) == original


def test_dask_collection_subclasses_match() -> None:
    """Public collection subclasses retain their base serializer dispatch."""
    da = pytest.importorskip("dask.array")
    db = pytest.importorskip("dask.bag")

    array = da.arange(8, chunks=4)

    class ArraySubclass(type(array)):
        pass

    array_subclass = ArraySubclass(
        array.dask,
        array.name,
        array.chunks,
        array.dtype,
        array._meta,  # noqa: SLF001
        shape=array.shape,
    )
    bag = db.from_sequence([1, 2], npartitions=2)

    class BagSubclass(type(bag)):
        pass

    bag_subclass = BagSubclass(bag.dask, bag.name, bag.npartitions)
    registry = Registry(dask_serializers)
    array_serializer = registry.lookup(array_subclass)
    bag_serializer = registry.lookup(bag_subclass)

    assert array_serializer is not None
    assert array_serializer.__name__ == "DaskArraySerializer"
    assert bag_serializer is not None
    assert bag_serializer.__name__ == "DaskBagSerializer"


def test_dask_internal_expression_does_not_match_collection() -> None:
    """An expression obtained through the public Array API is not a collection."""
    source = """
import dask
dask.config.set({"array.query-planning": True})
import dask.array as da
from misen.utils.serde.libs.dask import dask_serializers
from misen.utils.serde.registry import Registry

array = da.arange(8, chunks=4)
expression = getattr(array, "expr", None)
if expression is None:
    print("unsupported")
else:
    assert Registry(dask_serializers).lookup(expression) is None
    print("ok")
"""
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", source],
        check=True,
        capture_output=True,
        text=True,
    )
    if result.stdout.strip() == "unsupported":
        pytest.skip("This Dask version does not expose Array expressions through its public collection API.")
    assert result.stdout.strip() == "ok"
