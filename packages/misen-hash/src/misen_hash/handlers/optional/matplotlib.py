"""Handlers for matplotlib figure and axes objects."""

import importlib.util
from typing import Any

from misen_hash.handler_base import CollectionHandler, HandlerTypeList, HandlerTypeRegistry

__all__ = ["matplotlib_handlers", "matplotlib_handlers_by_type"]


matplotlib_handlers: HandlerTypeList = []
matplotlib_handlers_by_type: HandlerTypeRegistry = {}

if importlib.util.find_spec("matplotlib") is not None:

    def _line_payload(line: Any) -> list[Any]:
        return [
            line.get_label(),
            line.get_linestyle(),
            float(line.get_linewidth()),
            line.get_marker(),
            line.get_color(),
            line.get_xdata(orig=False),
            line.get_ydata(orig=False),
        ]

    def _image_payload(image: Any) -> list[Any]:
        cmap = image.get_cmap()
        return [
            image.__class__.__module__,
            image.__class__.__qualname__,
            None if cmap is None else cmap.name,
            tuple(float(v) for v in image.get_extent()),
            image.get_array(),
        ]

    def _collection_payload(collection: Any) -> list[Any]:
        offsets = collection.get_offsets() if hasattr(collection, "get_offsets") else None
        values = collection.get_array() if hasattr(collection, "get_array") else None
        sizes = collection.get_sizes() if hasattr(collection, "get_sizes") else None
        facecolor = collection.get_facecolor() if hasattr(collection, "get_facecolor") else None
        edgecolor = collection.get_edgecolor() if hasattr(collection, "get_edgecolor") else None
        return [
            collection.__class__.__module__,
            collection.__class__.__qualname__,
            offsets,
            values,
            sizes,
            facecolor,
            edgecolor,
        ]

    def _patch_payload(patch: Any) -> list[Any]:
        path = patch.get_path() if hasattr(patch, "get_path") else None
        vertices = None if path is None else path.vertices
        codes = None if path is None else path.codes
        return [
            patch.__class__.__module__,
            patch.__class__.__qualname__,
            vertices,
            codes,
            patch.get_facecolor() if hasattr(patch, "get_facecolor") else None,
            patch.get_edgecolor() if hasattr(patch, "get_edgecolor") else None,
            patch.get_linewidth() if hasattr(patch, "get_linewidth") else None,
        ]

    def _text_payload(text: Any) -> list[Any]:
        return [
            text.get_text(),
            tuple(float(v) for v in text.get_position()),
            float(text.get_rotation()),
            text.get_fontsize(),
            text.get_color(),
        ]

    def _axis_payload(axis: Any) -> list[Any]:
        return [
            axis.__class__.__module__,
            axis.__class__.__qualname__,
            axis.get_title(),
            axis.get_xlabel(),
            axis.get_ylabel(),
            tuple(float(v) for v in axis.get_xlim()),
            tuple(float(v) for v in axis.get_ylim()),
            axis.get_xscale(),
            axis.get_yscale(),
            [_line_payload(line) for line in axis.get_lines()],
            [_image_payload(image) for image in axis.images],
            [_collection_payload(collection) for collection in axis.collections],
            [_patch_payload(patch) for patch in axis.patches],
            [_text_payload(text) for text in axis.texts],
        ]

    def _is_axes(obj: Any) -> bool:
        if type(obj).__module__.split(".")[0] != "matplotlib":
            return False

        import matplotlib.axes

        return isinstance(obj, matplotlib.axes.Axes)

    def _is_figure(obj: Any) -> bool:
        if type(obj).__module__.split(".")[0] != "matplotlib":
            return False

        import matplotlib.figure

        return isinstance(obj, matplotlib.figure.Figure)

    class MatplotlibAxesHandler(CollectionHandler):
        """Hash matplotlib Axes by structural payload."""

        @staticmethod
        def match(obj: Any) -> bool:
            return _is_axes(obj)

        @staticmethod
        def elements(obj: Any) -> list[Any]:
            # Drop top-level class metadata because stable_hash already includes runtime type.
            return _axis_payload(obj)[2:]

    class MatplotlibFigureHandler(CollectionHandler):
        """Hash matplotlib Figure objects by figure-level and axes payload."""

        @staticmethod
        def match(obj: Any) -> bool:
            return _is_figure(obj)

        @staticmethod
        def elements(obj: Any) -> list[Any]:
            return [
                tuple(float(v) for v in obj.get_size_inches()),
                float(obj.get_dpi()),
                [_axis_payload(axis) for axis in obj.axes],
                [_text_payload(text) for text in obj.texts],
            ]

    matplotlib_handlers = [MatplotlibAxesHandler, MatplotlibFigureHandler]
    matplotlib_handlers_by_type = {
        "matplotlib.axes._axes.Axes": MatplotlibAxesHandler,
        "matplotlib.figure.Figure": MatplotlibFigureHandler,
    }
