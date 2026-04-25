"""Serializers for pandas DataFrame, Series, index, and scalar types.

All paths use version-stable formats:

- DataFrame / Series / Index — Apache Parquet (an open columnar
  standard with a published schema).
- Scalars (Timestamp / Timedelta / Period / Interval) — JSON-encoded
  structured representation extracting each scalar's stable surface
  attributes (ISO timestamps, integer nanoseconds, period freq strings,
  interval bounds + closed side).
- CategoricalDtype — JSON-encoded categories list + ordered flag, with
  the categories' dtype recorded so int-vs-string categories don't
  collapse on round-trip.

Pickle is deliberately avoided so saves survive pandas / Python /
NumPy version bumps.
"""

import importlib.util
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from misen.exceptions import SerializationError
from misen.utils.serde.base import LeafSerializer, Serializer

__all__ = ["pandas_serializers", "pandas_serializers_by_type"]

pandas_serializers: list[type[Serializer]] = []
pandas_serializers_by_type: dict[str, type[Serializer]] = {}

if importlib.util.find_spec("pandas") is not None:

    class PandasDataFrameSerializer(Serializer[Any]):
        """Serialize ``pandas.DataFrame`` via Parquet."""

        @staticmethod
        def match(obj: Any) -> bool:
            import pandas as pd

            return isinstance(obj, pd.DataFrame)

        @staticmethod
        def write(obj: Any, directory: Path) -> Mapping[str, Any]:
            import pandas as pd

            obj.to_parquet(directory / "data.parquet")
            return {"pandas_version": pd.__version__}

        @staticmethod
        def read(directory: Path, *, meta: Mapping[str, Any]) -> Any:  # noqa: ARG004
            import pandas as pd

            return pd.read_parquet(directory / "data.parquet")

    class PandasSeriesSerializer(Serializer[Any]):
        """Serialize ``pandas.Series`` via Parquet (single-column DataFrame)."""

        @staticmethod
        def match(obj: Any) -> bool:
            import pandas as pd

            return isinstance(obj, pd.Series)

        @staticmethod
        def write(obj: Any, directory: Path) -> Mapping[str, Any]:
            import pandas as pd

            df = obj.to_frame(name="__series__")
            df.to_parquet(directory / "data.parquet")
            return {
                "pandas_version": pd.__version__,
                "series_name": obj.name,
            }

        @staticmethod
        def read(directory: Path, *, meta: Mapping[str, Any]) -> Any:
            import pandas as pd

            df = pd.read_parquet(directory / "data.parquet")
            series = df["__series__"]
            series.name = meta.get("series_name")
            return series

    class PandasIndexSerializer(Serializer[Any]):
        """Serialize ``pandas.Index`` (including DatetimeIndex, MultiIndex, ...).

        Routes through Parquet by wrapping in a single-column DataFrame.
        MultiIndex preserves all levels.
        """

        @staticmethod
        def match(obj: Any) -> bool:
            import pandas as pd

            return isinstance(obj, pd.Index)

        @staticmethod
        def write(obj: Any, directory: Path) -> Mapping[str, Any]:
            import pandas as pd

            df = obj.to_frame(index=False)
            df.to_parquet(directory / "data.parquet")
            return {
                "pandas_version": pd.__version__,
                "index_name": obj.name,
                "is_multi": isinstance(obj, pd.MultiIndex),
            }

        @staticmethod
        def read(directory: Path, *, meta: Mapping[str, Any]) -> Any:
            import pandas as pd

            df = pd.read_parquet(directory / "data.parquet")
            if meta.get("is_multi"):
                idx = pd.MultiIndex.from_frame(df)
            else:
                idx = pd.Index(df.iloc[:, 0])
                idx.name = meta.get("index_name")
            return idx

    def _encode_pandas_scalar(obj: Any) -> dict[str, Any]:
        """Decompose a pandas scalar into a JSON-friendly dict.

        Each branch picks the most stable surface representation:
        ``isoformat`` for Timestamps (preserves tz), integer nanoseconds
        for Timedeltas, ``str(period)`` + freq alias for Periods, and
        the literal endpoints + closed side for Intervals.
        """
        import pandas as pd

        if isinstance(obj, pd.Timestamp):
            tz = obj.tz
            return {
                "type": "Timestamp",
                "iso": obj.isoformat(),
                "tz": str(tz) if tz is not None else None,
            }
        if isinstance(obj, pd.Timedelta):
            return {"type": "Timedelta", "ns": int(obj.value)}
        if isinstance(obj, pd.Period):
            return {"type": "Period", "value": str(obj), "freq": obj.freqstr}
        if isinstance(obj, pd.Interval):
            return {
                "type": "Interval",
                "left": _encode_interval_endpoint(obj.left),
                "right": _encode_interval_endpoint(obj.right),
                "closed": obj.closed,
            }
        msg = f"Unexpected pandas scalar type: {type(obj).__name__}"
        raise SerializationError(msg)

    def _encode_interval_endpoint(v: Any) -> Any:
        """Endpoint of a pandas Interval — pandas accepts ints, floats, and Timestamps."""
        import pandas as pd

        if isinstance(v, pd.Timestamp):
            return {"_pd_ts": v.isoformat(), "tz": str(v.tz) if v.tz is not None else None}
        if isinstance(v, (int, float)):
            return v
        msg = f"Unsupported Interval endpoint type: {type(v).__name__}"
        raise SerializationError(msg)

    def _decode_interval_endpoint(v: Any) -> Any:
        import pandas as pd

        if isinstance(v, dict) and "_pd_ts" in v:
            return pd.Timestamp(v["_pd_ts"], tz=v.get("tz"))
        return v

    def _decode_pandas_scalar(payload: dict[str, Any]) -> Any:
        import pandas as pd

        kind = payload["type"]
        if kind == "Timestamp":
            return pd.Timestamp(payload["iso"], tz=payload.get("tz"))
        if kind == "Timedelta":
            # ``pd.Timedelta(int, unit='ns')`` is the canonical inverse of ``Timedelta.value``.
            return pd.Timedelta(payload["ns"], unit="ns")
        if kind == "Period":
            return pd.Period(payload["value"], freq=payload["freq"])
        if kind == "Interval":
            return pd.Interval(
                _decode_interval_endpoint(payload["left"]),
                _decode_interval_endpoint(payload["right"]),
                closed=payload["closed"],
            )
        msg = f"Unknown pandas scalar kind: {kind!r}"
        raise SerializationError(msg)

    class PandasScalarSerializer(LeafSerializer[Any]):
        """Batched leaf for pandas scalar types — Timestamp, Timedelta, Period, Interval.

        Each scalar is decomposed into its stable surface attributes
        (ISO string, integer ns, freq alias, endpoints + closed) and
        bundled into a single JSON file per save.  No pickle.
        """

        leaf_kind = "pandas_scalar"

        @staticmethod
        def match(obj: Any) -> bool:
            import pandas as pd

            return isinstance(obj, (pd.Timestamp, pd.Timedelta, pd.Period, pd.Interval))

        @classmethod
        def to_payload(cls, obj: Any) -> Any:
            return _encode_pandas_scalar(obj)

        @staticmethod
        def write_batch(
            entries: list[tuple[str, Any, Mapping[str, Any]]],
            directory: Path,
        ) -> Mapping[str, Any]:
            import pandas as pd

            bundle = {leaf_id: payload for leaf_id, payload, _ in entries}
            (directory / "scalars.json").write_text(json.dumps(bundle), encoding="utf-8")
            return {"pandas_version": pd.__version__}

        @staticmethod
        def read_batch(directory: Path, kind_meta: Mapping[str, Any]) -> Any:  # noqa: ARG004
            bundle = json.loads((directory / "scalars.json").read_text(encoding="utf-8"))

            def reader(leaf_id: str) -> Any:
                return _decode_pandas_scalar(bundle[leaf_id])

            return reader

    class PandasCategoricalDtypeSerializer(Serializer[Any]):
        """Serialize ``pandas.CategoricalDtype`` as JSON.

        Saves the categories list, the underlying numpy dtype string
        (so int-vs-str categories survive round-trip), and the ordered
        flag.  Reconstructed via ``pd.CategoricalDtype(categories=...,
        ordered=...)`` with the categories rehydrated to their original
        dtype.
        """

        @staticmethod
        def match(obj: Any) -> bool:
            import pandas as pd

            return isinstance(obj, pd.CategoricalDtype)

        @staticmethod
        def write(obj: Any, directory: Path) -> Mapping[str, Any]:
            import pandas as pd

            cats = obj.categories
            payload = {
                "categories": cats.tolist(),
                "categories_dtype": str(cats.dtype),
                "ordered": bool(obj.ordered),
            }
            (directory / "dtype.json").write_text(json.dumps(payload), encoding="utf-8")
            return {"pandas_version": pd.__version__}

        @staticmethod
        def read(directory: Path, *, meta: Mapping[str, Any]) -> Any:  # noqa: ARG004
            import numpy as np
            import pandas as pd

            payload = json.loads((directory / "dtype.json").read_text(encoding="utf-8"))
            # Rehydrate ``categories`` to the original dtype before constructing the dtype —
            # ``CategoricalDtype([1, 2])`` and ``CategoricalDtype(['1', '2'])`` are different
            # types, so the on-disk dtype string is what disambiguates.
            cats = np.asarray(payload["categories"], dtype=np.dtype(payload["categories_dtype"]))
            return pd.CategoricalDtype(categories=cats, ordered=payload["ordered"])

    pandas_serializers = [
        PandasDataFrameSerializer,
        PandasSeriesSerializer,
        PandasIndexSerializer,
        PandasScalarSerializer,
        PandasCategoricalDtypeSerializer,
    ]

    # Build by-type entries at import time so the registry's MRO fast path
    # reaches ``PandasScalarSerializer`` before ``datetime.datetime``
    # (``pd.Timestamp`` subclasses ``datetime.datetime`` and otherwise
    # routes to :class:`MsgpackLeafSerializer` via the MRO walk).
    from misen.utils.type_registry import qualified_type_name as _qname

    import pandas as _pd

    pandas_serializers_by_type = {
        _qname(_pd.DataFrame): PandasDataFrameSerializer,
        _qname(_pd.Series): PandasSeriesSerializer,
        # Index subclasses (DatetimeIndex, PeriodIndex, MultiIndex, ...)
        # dispatch here via the MRO walk.
        _qname(_pd.Index): PandasIndexSerializer,
        _qname(_pd.CategoricalDtype): PandasCategoricalDtypeSerializer,
        _qname(_pd.Timestamp): PandasScalarSerializer,
        _qname(_pd.Timedelta): PandasScalarSerializer,
        _qname(_pd.Period): PandasScalarSerializer,
        _qname(_pd.Interval): PandasScalarSerializer,
    }
