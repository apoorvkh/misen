"""Serializer for matplotlib figures via PNG export.

The figure is saved as a high-quality PNG. On load, a ``PIL.Image.Image``
is returned (not a ``matplotlib.figure.Figure``) since reconstructing the
interactive figure object from pixels is not possible.  This is the
version-stable choice: the visual artifact is the meaningful result.
"""

import importlib.util
from typing import Any

from misen.utils.serde.serializer_base import (
    Serializer,
    SerializerTypeList,
    SerializerTypeRegistry,
    write_meta,
)

__all__ = ["matplotlib_serializers", "matplotlib_serializers_by_type"]

matplotlib_serializers: SerializerTypeList = []
matplotlib_serializers_by_type: SerializerTypeRegistry = {}

# Require both matplotlib and PIL for round-trip.
if importlib.util.find_spec("matplotlib") is not None and importlib.util.find_spec("PIL") is not None:
    from pathlib import Path

    class MatplotlibFigureSerializer(Serializer[Any]):
        """Serialize ``matplotlib.figure.Figure`` as high-quality PNG.

        Loads back as ``PIL.Image.Image``.
        """

        version = 1

        @staticmethod
        def match(obj: Any) -> bool:
            from matplotlib.figure import Figure

            return isinstance(obj, Figure)

        @staticmethod
        def save(obj: Any, directory: Path) -> None:
            import matplotlib as mpl

            obj.savefig(directory / "figure.png", format="png", dpi=150, bbox_inches="tight")
            write_meta(directory, MatplotlibFigureSerializer, matplotlib_version=mpl.__version__)

        @staticmethod
        def load(directory: Path) -> Any:
            from PIL import Image

            return Image.open(directory / "figure.png").copy()

    matplotlib_serializers = [MatplotlibFigureSerializer]
    matplotlib_serializers_by_type = {"matplotlib.figure.Figure": MatplotlibFigureSerializer}
