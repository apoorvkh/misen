"""Keyed collection of files persistable as a misen task result.

A :class:`FileMap` lets a task return "here are some files" instead of
materializing their contents in memory.  It is built incrementally from
sources on disk, and when persisted via misen's serializer the files are
moved into the result's cache directory, preserving their relative
layout.  On reload, the values are :class:`Path` objects that resolve
into the *current* workspace's cache directory, and :attr:`FileMap.root`
gives the single directory that holds them — suitable for handing to
external tools that consume a directory (TensorBoard, MLflow, ...).

Use this when a task produces many or large files (model checkpoints,
generated images, training logs, ...) and you want them to flow into
downstream tasks without round-tripping their contents through RAM.  The
intended role is to extract files out of a task's scratch directory
before the runtime cleans it up after the task completes.

Build it with the chainable ``include_*`` / ``exclude_*`` methods (or the
``from_glob`` / ``from_tree`` one-liners that wrap them)::

    from misen import FileMap, SCRATCH_DIR, meta

    @meta(cache=True)
    def train(scratch_dir: Path = SCRATCH_DIR) -> FileMap:
        return (FileMap()
                .include_glob(scratch_dir, "ckpt_*.pt",
                              key=lambda p: int(p.stem.split("_")[1]))
                .include_tree(scratch_dir / "tb_logs")
                .exclude_glob("*.tmp"))

    @meta(cache=True)
    def analyze(files: FileMap, step: int) -> dict[str, float]:
        state = torch.load(files[step], weights_only=True)  # by key
        ...

    # Hand the preserved directory to an external tool:
    logs = train_task.result()
    subprocess.run(["tensorboard", "--logdir", str(logs.root)])

Keys must be one of ``str``, ``int``, ``float``, ``bool`` or ``None``
(types that round-trip cleanly through JSON).  Exclusions are applied
eagerly: each ``exclude_*`` call filters whatever has been included so
far.  A FileMap reloaded from a workspace is read-only — build a fresh
one to stage different files.
"""

from __future__ import annotations

import fnmatch
from collections.abc import Hashable, Iterator, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Generic, NamedTuple, Self, TypeVar

if TYPE_CHECKING:
    from collections.abc import Callable
    from os import PathLike

__all__ = ["FileMap"]

K = TypeVar("K")

# Key types that round-trip cleanly through ``json.dumps``/``json.loads``
# *and* are hashable.  Tuples are intentionally excluded for v1 — JSON
# decodes them as lists, so they'd need a tagged encoding to preserve
# identity.  Add later if demand justifies it.
_ALLOWED_KEY_TYPES: tuple[type, ...] = (str, int, float, bool, type(None))


class _Entry(NamedTuple):
    """One file in a :class:`FileMap`.

    ``path`` is the file's current location (a source path while building,
    a cache path after reload).  ``layout`` is the file's relative path
    within the map — where it lands in the cache directory, preserving
    hierarchy for directory-consuming tools.
    """

    path: Path
    layout: str


class FileMap(Mapping[K, Path], Generic[K]):
    """A keyed collection of files persisted with a task result.

    Build incrementally with :meth:`include_glob`, :meth:`include_tree`,
    and :meth:`include`; trim with :meth:`exclude_glob` and
    :meth:`exclude`.  All builder methods mutate in place and return
    ``self`` for chaining.

    The map exposes a read-only ``Mapping[K, Path]`` for keyed access and
    :attr:`root` for the single directory that holds every file (valid
    after the map has been persisted and reloaded).

    A reloaded FileMap is frozen: its files live in the workspace cache,
    so further ``include_*`` / ``exclude_*`` calls raise.

    Args:
        paths: Optional initial explicit ``key -> source path`` entries,
            equivalent to calling :meth:`include` for each pair.

    Raises:
        TypeError: If any key is not an allowed type.
        ValueError: If a source is not a regular file, or a key/location
            collides with an existing entry.
    """

    def __init__(self, paths: Mapping[K, str | PathLike[str]] | None = None) -> None:
        """Initialize a (possibly empty) FileMap from explicit entries."""
        self._entries: dict[K, _Entry] = {}
        self._layouts: set[str] = set()
        self._counter: int = 0
        self._frozen: bool = False
        self._root: Path | None = None
        if paths is not None:
            for key, raw_path in paths.items():
                self.include(key, raw_path)

    # ------------------------------------------------------------------
    # Builders
    # ------------------------------------------------------------------

    def include(self, key: K, path: str | PathLike[str]) -> Self:
        """Add one explicit ``key -> file`` entry.

        The file is stored under a positional name (preserving its
        extension), so explicit entries never collide on basename.

        Args:
            key: Lookup key.  Must be ``str``/``int``/``float``/``bool``/``None``.
            path: Path to an existing regular file.

        Returns:
            ``self``, for chaining.
        """
        self._check_mutable()
        p = Path(path)
        layout = f"{self._counter}{p.suffix}"
        self._counter += 1
        self._add_entry(key, p, layout)
        return self

    def include_glob(
        self,
        root: str | PathLike[str],
        pattern: str,
        *,
        key: Callable[[Path], Hashable] = lambda p: p.stem,
    ) -> Self:
        """Add files matching ``pattern`` under ``root``.

        Uses :meth:`pathlib.Path.glob` semantics (``*``, ``?``, ``[seq]``,
        recursive ``**``).  Matches are added in sorted order for
        determinism; directories matched by the pattern are skipped.
        Each file's on-disk layout mirrors its path relative to ``root``.

        Args:
            root: Directory to search under.
            pattern: Glob pattern relative to ``root``.
            key: Maps each matched path to its key.  Defaults to the file
                stem (basename without extension).

        Returns:
            ``self``, for chaining.
        """
        self._check_mutable()
        root_path = Path(root)
        for match in sorted(root_path.glob(pattern)):
            if not match.is_file():
                continue
            self._add_entry(key(match), match, match.relative_to(root_path).as_posix())
        return self

    def include_tree(self, root: str | PathLike[str]) -> Self:
        """Add every file under ``root``, keyed by relative path.

        Preserves the directory structure verbatim in the cache, so
        :attr:`root` can be handed to tools that expect a directory tree.

        Args:
            root: Directory whose files (recursively) are added.

        Returns:
            ``self``, for chaining.
        """
        self._check_mutable()
        root_path = Path(root)
        for match in sorted(root_path.rglob("*")):
            if not match.is_file():
                continue
            rel = match.relative_to(root_path).as_posix()
            self._add_entry(rel, match, rel)
        return self

    # ------------------------------------------------------------------
    # Exclusions (eager: filter what has been included so far)
    # ------------------------------------------------------------------

    def exclude(self, predicate: Callable[[K, Path], bool]) -> Self:
        """Drop currently-included entries for which ``predicate`` is true.

        Args:
            predicate: Called with ``(key, source_path)``; return ``True``
                to drop the entry.

        Returns:
            ``self``, for chaining.
        """
        self._check_mutable()
        self._drop(key for key, entry in self._entries.items() if predicate(key, entry.path))
        return self

    def exclude_glob(self, pattern: str) -> Self:
        """Drop entries whose path within the map matches ``pattern``.

        Matching is shell-style (:func:`fnmatch.fnmatch`) against the
        relative layout path, so ``"*.tmp"`` and ``"tb_logs/*"`` both
        work.  For finer control use :meth:`exclude`.

        Args:
            pattern: Shell-style wildcard pattern.

        Returns:
            ``self``, for chaining.
        """
        self._check_mutable()
        self._drop(key for key, entry in self._entries.items() if fnmatch.fnmatch(entry.layout, pattern))
        return self

    # ------------------------------------------------------------------
    # Convenience constructors (sugar over the builder)
    # ------------------------------------------------------------------

    @classmethod
    def from_glob(
        cls,
        root: str | PathLike[str],
        pattern: str,
        *,
        key: Callable[[Path], Hashable] = lambda p: p.stem,
    ) -> FileMap:
        """Return a new FileMap of files matching ``pattern`` under ``root``."""
        return cls().include_glob(root, pattern, key=key)

    @classmethod
    def from_tree(cls, root: str | PathLike[str]) -> FileMap:
        """Return a new FileMap of every file under ``root``, keyed by relative path."""
        return cls().include_tree(root)

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def root(self) -> Path:
        """Directory containing every file, with relative layout preserved.

        Available only after the FileMap has been persisted and reloaded
        (while building, the files still live at their scattered sources).

        Raises:
            RuntimeError: If accessed before the map has been persisted.
        """
        if self._root is None:
            msg = (
                "FileMap.root is available only after the map has been persisted and reloaded "
                "(e.g. fetched from a task result). While building, files still live at their sources."
            )
            raise RuntimeError(msg)
        return self._root

    def __getitem__(self, key: K) -> Path:
        """Return the current path of the file stored under ``key``."""
        return self._entries[key].path

    def __iter__(self) -> Iterator[K]:
        """Iterate over keys in insertion order."""
        return iter(self._entries)

    def __len__(self) -> int:
        """Return the number of entries."""
        return len(self._entries)

    def __contains__(self, key: object) -> bool:
        """Return whether ``key`` is present."""
        return key in self._entries

    def __repr__(self) -> str:
        """Return a short summary; full contents are intentionally elided."""
        n = len(self._entries)
        suffix = "" if n == 1 else "s"
        return f"FileMap({n} file{suffix})"

    def __eq__(self, other: object) -> bool:
        """Return equality based on the underlying key→path mapping."""
        if not isinstance(other, FileMap):
            return NotImplemented
        return {k: e.path for k, e in self._entries.items()} == {k: e.path for k, e in other._entries.items()}

    # FileMap is conceptually mutable storage (the underlying files can be
    # rewritten out-of-band) so it isn't hashable.  ``Mapping`` already sets
    # ``__hash__ = None`` — we re-state it here so PLW1641 doesn't flag the
    # explicit ``__eq__`` definition as an oversight.
    __hash__ = None  # type: ignore[assignment]

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _check_mutable(self) -> None:
        if self._frozen:
            msg = (
                "This FileMap was loaded from a workspace and is read-only. "
                "Build a fresh FileMap to stage different files."
            )
            raise RuntimeError(msg)

    def _add_entry(self, key: Hashable, path: Path, layout: str) -> None:
        if not isinstance(key, _ALLOWED_KEY_TYPES):
            msg = f"FileMap keys must be str/int/float/bool/None; got {type(key).__name__} for key {key!r}."
            raise TypeError(msg)
        if not path.is_file():
            msg = f"FileMap[{key!r}]: not a regular file: {path}"
            raise ValueError(msg)
        if key in self._entries:
            msg = f"FileMap already contains key {key!r}."
            raise ValueError(msg)
        if layout in self._layouts:
            msg = f"FileMap already contains a file at location {layout!r}."
            raise ValueError(msg)
        self._entries[key] = _Entry(path=path, layout=layout)  # ty:ignore[invalid-assignment]
        self._layouts.add(layout)

    def _drop(self, keys: Iterator[K]) -> None:
        for key in list(keys):
            entry = self._entries.pop(key)
            self._layouts.discard(entry.layout)

    @classmethod
    def _from_records(cls, records: list[tuple[K, str, Path]], root: Path) -> FileMap[K]:
        """Construct a frozen FileMap from serializer records (no re-validation)."""
        obj: FileMap[K] = cls.__new__(cls)
        obj._entries = {key: _Entry(path=path, layout=layout) for key, layout, path in records}
        obj._layouts = {layout for _, layout, _ in records}
        obj._counter = 0
        obj._frozen = True
        obj._root = root
        return obj
