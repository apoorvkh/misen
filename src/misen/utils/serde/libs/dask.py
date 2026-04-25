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
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from misen.exceptions import SerializationError
from misen.utils.serde.base import BaseSerializer, Container, DecodeCtx, EncodeCtx, Node, Serializer
from misen.utils.type_registry import qualified_type_name

__all__ = ["dask_serializers", "dask_serializers_by_type"]

dask_serializers: list[type[Serializer]] = []
dask_serializers_by_type: dict[str, type[Serializer]] = {}


if importlib.util.find_spec("dask") is not None:

    class DaskDataFrameSerializer(Serializer[Any]):
        """Serialize ``dask.dataframe.DataFrame`` — compute → parquet → from_pandas."""

        @staticmethod
        def match(obj: Any) -> bool:
            import dask.dataframe as dd

            return isinstance(obj, dd.DataFrame)

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
            import dask.array as da

            return isinstance(obj, da.Array)

        @staticmethod
        def write(obj: Any, directory: Path) -> Mapping[str, Any]:
            import dask
            import numpy as np

            arr = obj.compute()
            np.save(directory / "data.npy", arr, allow_pickle=False)
            return {
                "dask_version": dask.__version__,
                # ``chunks`` is a tuple of tuples — JSON-safe once stringified
                # into a nested list of ints.
                "chunks": [list(c) for c in obj.chunks],
            }

        @staticmethod
        def read(directory: Path, *, meta: Mapping[str, Any]) -> Any:
            import dask.array as da
            import numpy as np

            arr = np.load(directory / "data.npy", allow_pickle=False)
            chunks_meta = meta.get("chunks")
            chunks: Any = tuple(tuple(c) for c in chunks_meta) if chunks_meta else "auto"
            return da.from_array(arr, chunks=chunks)

    class DaskBagSerializer(BaseSerializer[Any]):
        """Serialize ``dask.bag.Bag`` by recursively encoding each computed item.

        Each element dispatches independently through :func:`ctx.encode`,
        so primitive items collapse into the shared msgpack leaf and
        ndarrays land in the numpy leaf.
        """

        @staticmethod
        def match(obj: Any) -> bool:
            import dask.bag as db

            return isinstance(obj, db.Bag)

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
