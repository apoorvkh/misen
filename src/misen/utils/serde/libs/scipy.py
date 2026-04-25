"""Serializers for scipy sparse matrices, stats distributions, interpolators, and optimize results.

All serializers route through stable scipy public APIs — no pickle:

- Sparse matrices: :func:`scipy.sparse.save_npz`/`load_npz` (npz of indptr,
  indices, data).
- Frozen ``scipy.stats`` distributions: save the distribution name +
  args + kwds, reconstruct via ``getattr(scipy.stats, name)(*args,
  **kwds)``.
- Splines (``BSpline``, ``PPoly``, ``BPoly``, and PPoly subclasses such
  as ``CubicSpline`` / ``Akima1DInterpolator`` / ``PchipInterpolator``):
  save knots / coefficients / breakpoints as ndarrays plus the small
  scalar attrs (degree, axis, extrapolate flag), reconstruct via the
  class's own ``construct_fast`` factory.
- ``OptimizeResult``: recursive Container — every field dispatches
  through :func:`ctx.encode`, so embedded ndarrays land in the numpy
  batch instead of being inlined.

Frozen distributions and OptimizeResult are :class:`BaseSerializer`
subclasses (recursion-aware containers) because their natural payload
contains ndarrays that should batch through the numpy path.
"""

import importlib.util
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from misen.exceptions import SerializationError
from misen.utils.serde.base import BaseSerializer, Container, DecodeCtx, EncodeCtx, Node, Serializer
from misen.utils.type_registry import import_by_qualified_name, qualified_type_name

__all__ = ["scipy_serializers", "scipy_serializers_by_type"]

scipy_serializers: list[type[Serializer]] = []
scipy_serializers_by_type: dict[str, type[Serializer]] = {}

if importlib.util.find_spec("scipy") is not None:

    class ScipySparseSerializer(Serializer[Any]):
        """Serialize scipy sparse matrices via ``save_npz``/``load_npz``."""

        @staticmethod
        def match(obj: Any) -> bool:
            import scipy.sparse

            return scipy.sparse.issparse(obj)

        @staticmethod
        def write(obj: Any, directory: Path) -> Mapping[str, Any]:
            import scipy
            import scipy.sparse

            scipy.sparse.save_npz(directory / "data.npz", obj)
            return {"scipy_version": scipy.__version__}

        @staticmethod
        def read(directory: Path, *, meta: Mapping[str, Any]) -> Any:  # noqa: ARG004
            import scipy.sparse

            return scipy.sparse.load_npz(directory / "data.npz")

    class ScipyStatsFrozenSerializer(BaseSerializer[Any]):
        """Serialize frozen ``scipy.stats`` distributions by re-applying their constructor.

        ``rv_frozen`` records the parent distribution (a singleton) plus
        the args/kwds the user passed.  We persist the registered
        distribution name and route the args/kwds through
        :func:`ctx.encode` so any embedded ndarrays land in the numpy
        leaf batch.  On decode we look the singleton back up by name
        and re-apply: ``getattr(scipy.stats, name)(*args, **kwds)``.
        """

        @staticmethod
        def match(obj: Any) -> bool:
            import scipy.stats._distn_infrastructure as _d

            candidates: tuple[type, ...] = ()
            for name in ("rv_frozen", "rv_continuous_frozen", "rv_discrete_frozen"):
                cls = getattr(_d, name, None)
                if cls is not None:
                    candidates = (*candidates, cls)
            return isinstance(obj, candidates) if candidates else False

        @classmethod
        def encode(cls, obj: Any, ctx: EncodeCtx) -> Node:
            return Container(
                serializer=qualified_type_name(cls),
                children={
                    "args": ctx.encode(list(obj.args)),
                    "kwds": ctx.encode(dict(obj.kwds)),
                },
                meta={"dist_name": obj.dist.name},
            )

        @classmethod
        def decode(cls, node: Node, ctx: DecodeCtx) -> Any:
            import scipy.stats

            if not isinstance(node, Container):
                msg = f"{qualified_type_name(cls)} expected a Container node, got {type(node).__name__}."
                raise SerializationError(msg)
            dist_name = node.meta.get("dist_name")
            if not dist_name or not hasattr(scipy.stats, dist_name):
                msg = f"Unknown scipy.stats distribution {dist_name!r} on decode."
                raise SerializationError(msg)
            dist = getattr(scipy.stats, dist_name)
            args = ctx.decode(node.children["args"])
            kwds = ctx.decode(node.children["kwds"])
            return dist(*args, **kwds)

    class ScipyInterpolatorSerializer(BaseSerializer[Any]):
        """Serialize PPoly/BPoly/BSpline (and PPoly subclasses) by their canonical attrs.

        Supported classes:

        - :class:`scipy.interpolate.BSpline` — knots ``t``, coefficients
          ``c``, degree ``k``, plus ``axis`` and ``extrapolate``.
        - :class:`scipy.interpolate.PPoly` and subclasses
          (``CubicSpline``, ``Akima1DInterpolator``, ``PchipInterpolator``,
          ``CubicHermiteSpline``) — coefficients ``c``, breakpoints
          ``x``, plus ``axis`` and ``extrapolate``.
        - :class:`scipy.interpolate.BPoly` — same as PPoly.

        Each is reconstructed via the class's ``construct_fast``
        factory, which scipy provides specifically to bypass the
        forward constructor's data-driven validation.  Other
        interpolator types (``RegularGridInterpolator``,
        ``RBFInterpolator``, ``interp1d``, ...) carry richer state and
        are intentionally unsupported.
        """

        @staticmethod
        def match(obj: Any) -> bool:
            import scipy.interpolate as si

            base_classes: tuple[type, ...] = ()
            for name in ("BSpline", "PPoly", "BPoly"):
                cls = getattr(si, name, None)
                if cls is not None:
                    base_classes = (*base_classes, cls)
            return isinstance(obj, base_classes) if base_classes else False

        @classmethod
        def encode(cls, obj: Any, ctx: EncodeCtx) -> Node:
            import scipy.interpolate as si

            qname = qualified_type_name(type(obj))
            if isinstance(obj, si.BSpline):
                children = {
                    "t": ctx.encode(obj.t),
                    "c": ctx.encode(obj.c),
                }
                meta = {
                    "kind": "BSpline",
                    "cls": qname,
                    "k": int(obj.k),
                    "axis": int(getattr(obj, "axis", 0)),
                    "extrapolate": _encode_extrapolate(getattr(obj, "extrapolate", True)),
                }
            else:  # PPoly / BPoly and their subclasses
                children = {
                    "c": ctx.encode(obj.c),
                    "x": ctx.encode(obj.x),
                }
                kind = "BPoly" if isinstance(obj, si.BPoly) else "PPoly"
                meta = {
                    "kind": kind,
                    "cls": qname,
                    "axis": int(getattr(obj, "axis", 0)),
                    "extrapolate": _encode_extrapolate(getattr(obj, "extrapolate", True)),
                }
            return Container(serializer=qualified_type_name(cls), children=children, meta=meta)

        @classmethod
        def decode(cls, node: Node, ctx: DecodeCtx) -> Any:
            if not isinstance(node, Container):
                msg = f"{qualified_type_name(cls)} expected a Container node, got {type(node).__name__}."
                raise SerializationError(msg)
            try:
                target_cls = import_by_qualified_name(node.meta["cls"])
            except (ImportError, KeyError) as exc:
                msg = f"Cannot import interpolator class {node.meta.get('cls')!r}: {exc}"
                raise SerializationError(msg) from exc
            extrapolate = _decode_extrapolate(node.meta.get("extrapolate", True))
            axis = int(node.meta.get("axis", 0))
            kind = node.meta.get("kind")
            if kind == "BSpline":
                t = ctx.decode(node.children["t"])
                c = ctx.decode(node.children["c"])
                k = int(node.meta["k"])
                return target_cls.construct_fast(t, c, k, extrapolate, axis)
            if kind in ("PPoly", "BPoly"):
                c = ctx.decode(node.children["c"])
                x = ctx.decode(node.children["x"])
                return target_cls.construct_fast(c, x, extrapolate, axis)
            msg = f"Unknown interpolator kind {kind!r}"
            raise SerializationError(msg)

    class ScipyOptimizeResultSerializer(BaseSerializer[Any]):
        """Serialize ``scipy.optimize.OptimizeResult`` by recursing over its fields.

        ``OptimizeResult`` is a ``dict`` subclass with optimizer-specific
        attributes (``jac``, ``hess_inv``, ``message``, ``njev``, ...).
        Each entry dispatches independently through :func:`ctx.encode`,
        so embedded ndarrays land in the numpy leaf batch instead of
        being inlined.  No pickle.
        """

        @staticmethod
        def match(obj: Any) -> bool:
            import scipy.optimize

            return isinstance(obj, scipy.optimize.OptimizeResult)

        @classmethod
        def encode(cls, obj: Any, ctx: EncodeCtx) -> Node:
            return Container(
                serializer=qualified_type_name(cls),
                children={k: ctx.encode(v) for k, v in obj.items()},
                meta={},
            )

        @classmethod
        def decode(cls, node: Node, ctx: DecodeCtx) -> Any:
            import scipy.optimize

            if not isinstance(node, Container):
                msg = f"{qualified_type_name(cls)} expected a Container node, got {type(node).__name__}."
                raise SerializationError(msg)
            out = scipy.optimize.OptimizeResult()
            ctx.remember_node(node, out)
            for k, v in node.children.items():
                out[k] = ctx.decode(v)
            return out

    def _encode_extrapolate(value: Any) -> Any:
        """``extrapolate`` is bool, ``None``, or the string ``"periodic"`` — JSON-safe as-is."""
        if value is None or isinstance(value, (bool, str)):
            return value
        # Some scipy versions return numpy.bool_ — coerce to Python bool.
        return bool(value)

    def _decode_extrapolate(value: Any) -> Any:
        # JSON manifest may have stored ``true``/``false``/``null``/``"periodic"`` directly.
        return value

    scipy_serializers = [
        ScipySparseSerializer,
        ScipyStatsFrozenSerializer,
        ScipyInterpolatorSerializer,
        ScipyOptimizeResultSerializer,
    ]
    # scipy sparse types vary (csr / csc / coo / bsr / dia / lil), so the
    # sparse serializer relies on ``match()``.  Frozen distributions and
    # interpolators have many subclasses dispatched via the linear scan
    # too.  ``OptimizeResult`` has a stable concrete type — register it.
    import scipy.optimize as _opt
    from misen.utils.type_registry import qualified_type_name as _qname

    scipy_serializers_by_type = {
        _qname(_opt.OptimizeResult): ScipyOptimizeResultSerializer,
    }
