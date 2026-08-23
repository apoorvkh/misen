"""Serializers for dask collections (DataFrame, Array, Bag).

A dask collection is a *lazy task graph* over chunked data, so we
persist the *materialized* result and the chunking hints.  On load,
the collection is rebuilt from the stored material.

User-facing behavior (data returned by ``.compute()``, ``.shape`` /
``.dtypes`` / ``.npartitions`` / ``.columns``, every standard
operation) round-trips identically.  The internal task graph is
fresh — that's storage-implementation detail rather than a property
the data-analysis API exposes.

DataFrame routes through Parquet, Array routes through ``.npy``,
and Bag routes its computed items through the framework recursively
(so each item dispatches to its own serializer — primitives collapse
into the msgpack leaf, ndarrays land in the numpy leaf, etc.).
"""

import importlib.util
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from misen.exceptions import SerializationError
from misen.utils.serde.base import BaseSerializer, Container, DecodeCtx, EncodeCtx, Node, Serializer, translate_errors
from misen.utils.type_registry import qualified_type_name

__all__ = ["dask_serializers", "dask_serializers_by_type"]

dask_serializers: list[type[BaseSerializer]] = []
dask_serializers_by_type: dict[str, type[BaseSerializer]] = {}


def _is_collection(obj: Any, module_name: str, type_name: str) -> bool:
    """Lazily match a public Dask collection type and its subclasses."""
    module = sys.modules.get(module_name)
    return module is not None and isinstance(obj, getattr(module, type_name))


def _array_chunks_from_meta(meta: Mapping[str, Any]) -> Any:
    """Validate and rebuild the nested Dask array chunk tuple."""
    chunks_meta = meta.get("chunks")
    if chunks_meta is None or chunks_meta == []:
        return "auto"
    if not isinstance(chunks_meta, (list, tuple)) or not all(
        isinstance(chunk_sizes, (list, tuple)) for chunk_sizes in chunks_meta
    ):
        msg = "Dask array 'chunks' metadata must be a nested sequence."
        raise TypeError(msg)
    return tuple(tuple(chunk_sizes) for chunk_sizes in chunks_meta)


if importlib.util.find_spec("dask") is not None:

    class DaskDataFrameSerializer(Serializer[Any]):
        """Serialize ``dask.dataframe.DataFrame`` — compute → parquet → from_pandas."""

        @staticmethod
        def match(obj: Any) -> bool:
            return _is_collection(obj, "dask.dataframe", "DataFrame")

        @staticmethod
        def write(obj: Any, directory: Path) -> Mapping[str, Any]:
            import dask

            df = obj.compute()
            df.to_parquet(directory / "data.parquet")
            return {
                "dask_version": dask.__version__,
                "npartitions": int(obj.npartitions),
            }

        @staticmethod
        def read(directory: Path, *, meta: Mapping[str, Any]) -> Any:
            import dask.dataframe as dd
            import pandas as pd

            df = pd.read_parquet(directory / "data.parquet")
            npartitions = int(meta.get("npartitions", 1)) or 1
            return dd.from_pandas(df, npartitions=npartitions)

    class DaskArraySerializer(Serializer[Any]):
        """Serialize ``dask.array.Array`` — compute → npy → from_array."""

        @staticmethod
        def match(obj: Any) -> bool:
            return _is_collection(obj, "dask.array", "Array")

        @staticmethod
        def write(obj: Any, directory: Path) -> Mapping[str, Any]:
            import dask
            import numpy as np

            # Materializing a lazy graph can execute user code, whose exception
            # type and traceback must remain intact.  Only translate failures
            # owned by this persistence adapter below.
            arr = obj.compute()
            with translate_errors(f"Could not encode Dask array in {directory}", (OSError, TypeError, ValueError)):
                np.save(directory / "data.npy", arr, allow_pickle=False)
                chunks = [list(chunk_sizes) for chunk_sizes in obj.chunks]
            return {
                "dask_version": dask.__version__,
                # ``chunks`` is a tuple of tuples; convert it into nested lists
                # for the JSON manifest.
                "chunks": chunks,
            }

        @staticmethod
        def read(directory: Path, *, meta: Mapping[str, Any]) -> Any:
            import dask.array as da
            import numpy as np

            with translate_errors(
                f"Could not decode Dask array in {directory}", (OSError, EOFError, TypeError, ValueError)
            ):
                arr = np.load(directory / "data.npy", allow_pickle=False)
                chunks = _array_chunks_from_meta(meta)
                return da.from_array(arr, chunks=chunks)

    class DaskBagSerializer(BaseSerializer[Any]):
        """Serialize ``dask.bag.Bag`` by recursively encoding each computed item.

        Each element dispatches independently through :func:`ctx.encode`,
        so primitive items collapse into the shared msgpack leaf and
        ndarrays land in the numpy leaf.
        """

        @staticmethod
        def match(obj: Any) -> bool:
            return _is_collection(obj, "dask.bag", "Bag")

        @classmethod
        def encode(cls, obj: Any, ctx: EncodeCtx) -> Node:
            items = list(obj.compute())
            return Container(
                serializer=qualified_type_name(cls),
                children=[ctx.encode(item) for item in items],
                meta={"npartitions": int(obj.npartitions)},
            )

        @classmethod
        def decode(cls, node: Node, ctx: DecodeCtx) -> Any:
            import dask.bag as db

            if not isinstance(node, Container):
                msg = f"{qualified_type_name(cls)} expected a Container node, got {type(node).__name__}."
                raise SerializationError(msg)
            items = [ctx.decode(c) for c in node.children]
            npartitions = int(node.meta.get("npartitions", 1)) or 1
            return db.from_sequence(items, npartitions=npartitions)

    dask_serializers = [DaskDataFrameSerializer, DaskArraySerializer, DaskBagSerializer]
    dask_serializers_by_type = {
        "dask.dataframe.core.DataFrame": DaskDataFrameSerializer,
        "dask.array.core.Array": DaskArraySerializer,
        "dask.bag.core.Bag": DaskBagSerializer,
    }
