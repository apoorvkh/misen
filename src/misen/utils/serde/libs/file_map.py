"""Directory-owning serializer for :class:`misen.FileMap`.

Each file is placed into the result's own subdirectory inside the cache
at its relative layout path, so the subdirectory mirrors the structure
the files were collected with — handing :attr:`misen.FileMap.root` to a
directory-consuming tool (TensorBoard, MLflow, ...) just works.  On load,
the returned :class:`FileMap` exposes paths that resolve into whatever
the local workspace decides the cache directory is, so the round-tripped
value is portable across machines and workspaces.

Freshly-built FileMaps reference files in a task's scratch directory (or
elsewhere the user pointed at), so the serializer **moves** them — the
intended role is to extract files before the runtime cleans up scratch.
A *reloaded* FileMap (``_frozen``) references files already living in a
result cache; re-serializing one (e.g. a task that passes an upstream
result through) **hardlinks or copies** instead, so the upstream result
is never cannibalized.

The manifest (``entries.json`` inside the subdirectory) records
``(key, path)`` pairs, where ``path`` is the relative layout.  Keys are
JSON-native primitives (``str | int | float | bool | None``), enforced by
:class:`misen.FileMap` at construction time.
"""

from __future__ import annotations

import json
import os
import shutil
from typing import TYPE_CHECKING, Any

from misen.exceptions import SerializationError
from misen.file_map import FileMap
from misen.utils.serde.base import Serializer

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

__all__ = ["FileMapSerializer", "file_map_serializers", "file_map_serializers_by_type"]

_ENTRIES_FILENAME = "entries.json"


def _link_or_copy(src: Path, dst: Path) -> None:
    """Hardlink *src* to *dst*; fall back to a metadata-preserving copy.

    Used when persisting a reloaded FileMap whose sources live in a result
    cache: a hardlink is free on the same filesystem and avoids duplicating
    bytes, while the copy fallback covers cross-filesystem and link-hostile
    mounts.
    """
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


class FileMapSerializer(Serializer[FileMap[Any]]):
    """Persist :class:`FileMap` values as a directory of keyed files.

    The serializer never loads file *contents* into memory — it only
    moves/links/copies file handles — so RAM use during save is bounded
    independently of file sizes.
    """

    @staticmethod
    def match(obj: Any) -> bool:
        """Return ``True`` for :class:`FileMap` instances."""
        return isinstance(obj, FileMap)

    @staticmethod
    def write(obj: FileMap[Any], directory: Path) -> Mapping[str, Any]:
        """Place each entry's file into *directory* at its layout path; record the manifest.

        Freshly-built maps are moved; reloaded (frozen) maps are
        hardlinked/copied so the originating cache survives.
        """
        reuse_sources = obj._frozen  # noqa: SLF001  -- ownership signal, by design
        entries: list[dict[str, Any]] = []
        for key, entry in obj._entries.items():  # noqa: SLF001  -- serializer is the owning module
            dst = directory / entry.layout
            dst.parent.mkdir(parents=True, exist_ok=True)
            if reuse_sources:
                _link_or_copy(entry.path, dst)
            else:
                # ``shutil.move`` handles cross-filesystem fallback (copy +
                # delete) internally — rename when possible, correct otherwise.
                shutil.move(os.fspath(entry.path), os.fspath(dst))
            entries.append({"key": key, "path": entry.layout})
        (directory / _ENTRIES_FILENAME).write_text(
            json.dumps(entries, ensure_ascii=False),
            encoding="utf-8",
        )
        return {}

    @staticmethod
    def read(directory: Path, *, meta: Mapping[str, Any]) -> FileMap[Any]:  # noqa: ARG004
        """Reconstruct a frozen :class:`FileMap` whose paths point inside *directory*."""
        try:
            entries_blob = (directory / _ENTRIES_FILENAME).read_text(encoding="utf-8")
        except FileNotFoundError as exc:
            msg = f"FileMap is missing {_ENTRIES_FILENAME} in {directory}."
            raise SerializationError(msg) from exc
        try:
            entries = json.loads(entries_blob)
        except json.JSONDecodeError as exc:
            msg = f"FileMap {_ENTRIES_FILENAME} is not valid JSON in {directory}: {exc}"
            raise SerializationError(msg) from exc
        if not isinstance(entries, list):
            msg = f"FileMap {_ENTRIES_FILENAME} in {directory} must be a JSON array."
            raise SerializationError(msg)

        # Booleans round-trip cleanly through JSON, but ``True == 1`` and
        # ``False == 0`` collapse in dict keys.  That's already the case at
        # construction time too, so no special handling is needed here.
        records: list[tuple[Any, str, Path]] = []
        for entry in entries:
            if not isinstance(entry, dict) or "key" not in entry or "path" not in entry:
                msg = f"FileMap entry in {directory} is malformed: {entry!r}"
                raise SerializationError(msg)
            layout = entry["path"]
            records.append((entry["key"], layout, directory / layout))

        return FileMap._from_records(records, root=directory)  # noqa: SLF001  -- private accessor by design


file_map_serializers: list[type[Serializer]] = [FileMapSerializer]
file_map_serializers_by_type: dict[str, type[Serializer]] = {
    "misen.file_map.FileMap": FileMapSerializer,
}
