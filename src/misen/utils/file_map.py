"""Keyed files that can be persisted as a task result without loading them.

Build a map with the chainable ``include_*`` and ``exclude_*`` methods. Files
retain their relative layout in the result cache, and a reloaded map exposes
that directory through :attr:`FileMap.root`. Reloaded maps are read-only.
"""

from __future__ import annotations

import fnmatch
from collections.abc import Hashable, Iterator, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Generic, NamedTuple, Self, TypeVar

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable
    from os import PathLike

__all__ = ["FileMap"]

K = TypeVar("K", bound=Hashable)

# Key types that round-trip cleanly through ``json.dumps``/``json.loads``
# *and* are hashable.  Tuples are intentionally excluded for v1 — JSON
# decodes them as lists, so they'd need a tagged encoding to preserve
# identity.  Add later if demand justifies it.
_ALLOWED_KEY_TYPES: tuple[type, ...] = (str, int, float, bool, type(None))


class _Entry(NamedTuple):
    """A file's current path and relative layout within its map."""

    path: Path
    layout: str


class FileMap(Mapping[K, Path], Generic[K]):
    """A chainable, keyed collection of files persisted with a task result.

    Keys must be ``str``, ``int``, ``float``, ``bool``, or ``None``. Sources
    must be regular files with unique keys and cache layouts.
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
        return self._include_matches(root_path, root_path.glob(pattern), key)

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
        return self._include_matches(root_path, root_path.rglob("*"))

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
        return f"FileMap({n} file{'' if n == 1 else 's'})"

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

    def _include_matches(
        self,
        root: Path,
        matches: Iterable[Path],
        key: Callable[[Path], Hashable] | None = None,
    ) -> Self:
        for match in sorted(matches):
            if match.is_file():
                layout = match.relative_to(root).as_posix()
                self._add_entry(layout if key is None else key(match), match, layout)
        return self

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
