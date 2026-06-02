"""Tests for :class:`misen.FileMap` and its serializer.

Covers construction-time validation, the Mapping protocol, the builder
API (include_/exclude_), serializer round-trip (key-type fidelity,
hierarchical layout, source-move lifecycle), the frozen-reload contract
and copy-on-reserialize, ``.root``, portability (loaded paths resolve
into the load directory), and composition inside dataclasses + dicts.
"""
# ruff: noqa: D103, S101, PLR2004

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from misen import FileMap
from misen.exceptions import SerializationError
from misen.utils import serde
from misen.utils.serde.libs.file_map import FileMapSerializer

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write(p: Path, content: bytes = b"x") -> Path:
    p.write_bytes(content)
    return p


def _manifest(directory: Path) -> dict:
    return json.loads((directory / serde.MANIFEST_FILENAME).read_text())


# ---------------------------------------------------------------------------
# Construction + Mapping protocol
# ---------------------------------------------------------------------------


def test_construct_empty(tmp_path: Path) -> None:
    _ = tmp_path
    store: FileMap[str] = FileMap({})
    assert len(store) == 0
    assert list(store) == []
    assert dict(store) == {}


def test_construct_with_paths(tmp_path: Path) -> None:
    a = _write(tmp_path / "a.txt", b"a")
    b = _write(tmp_path / "b.txt", b"b")
    store: FileMap[str] = FileMap({"a": a, "b": b})

    assert len(store) == 2
    assert store["a"] == a
    assert store["b"] == b
    assert set(store) == {"a", "b"}
    assert "a" in store
    assert "missing" not in store
    assert dict(store.items()) == {"a": a, "b": b}


def test_str_paths_are_coerced(tmp_path: Path) -> None:
    a = _write(tmp_path / "a.txt", b"a")
    store: FileMap[str] = FileMap({"a": str(a)})
    assert isinstance(store["a"], Path)
    assert store["a"] == a


@pytest.mark.parametrize("key", ["s", 0, 1, 3.14, True, False, None])
def test_accepts_primitive_key_types(tmp_path: Path, key: object) -> None:
    f = _write(tmp_path / "x.txt", b"x")
    store: FileMap = FileMap({key: f})
    assert store[key] == f


@pytest.mark.parametrize("bad_key", [("a",), frozenset({1}), b"bytes", object()])
def test_rejects_non_primitive_key(tmp_path: Path, bad_key: object) -> None:
    """Hashable-but-unsupported key types should be rejected at construction.

    Unhashable types (list, dict, set) can't appear as ``Mapping`` keys at all,
    so they fail before reaching the FileMap — they're not tested here.
    """
    f = _write(tmp_path / "x.txt", b"x")
    with pytest.raises(TypeError, match="FileMap keys must be"):
        FileMap({bad_key: f})


def test_rejects_missing_path(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not a regular file"):
        FileMap({"a": tmp_path / "does-not-exist.txt"})


def test_rejects_directory_as_value(tmp_path: Path) -> None:
    d = tmp_path / "subdir"
    d.mkdir()
    with pytest.raises(ValueError, match="not a regular file"):
        FileMap({"a": d})


def test_repr_and_eq(tmp_path: Path) -> None:
    f = _write(tmp_path / "x.txt", b"x")
    a: FileMap[str] = FileMap({"k": f})
    b: FileMap[str] = FileMap({"k": f})
    assert a == b
    assert repr(a) == "FileMap(1 file)"
    assert repr(FileMap({})) == "FileMap(0 files)"


def test_mapping_unhashable(tmp_path: Path) -> None:
    f = _write(tmp_path / "x.txt", b"x")
    store: FileMap[str] = FileMap({"k": f})
    with pytest.raises(TypeError):
        hash(store)


# ---------------------------------------------------------------------------
# Dispatch — make sure the registry routes FileMap to FileMapSerializer
# ---------------------------------------------------------------------------


def test_registry_dispatches_to_filestore_serializer(tmp_path: Path) -> None:
    f = _write(tmp_path / "x.txt", b"x")
    store: FileMap[str] = FileMap({"k": f})
    registry = serde.Registry([FileMapSerializer])
    assert registry.lookup(store) is FileMapSerializer


def test_default_registry_dispatches_filestore() -> None:
    from misen.utils.serde.libs import default_registry

    f = Path(__file__)  # any existing file
    store: FileMap[str] = FileMap({"self": f})
    assert default_registry().lookup(store) is FileMapSerializer


# ---------------------------------------------------------------------------
# Round-trip
# ---------------------------------------------------------------------------


def test_roundtrip_basic(tmp_path: Path) -> None:
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    save_dir = tmp_path / "save"
    save_dir.mkdir()

    a = _write(src_dir / "a.txt", b"hello")
    b = _write(src_dir / "b.txt", b"world")

    original: FileMap[str] = FileMap({"alpha": a, "beta": b})
    serde.save(original, save_dir)

    # The root node in the manifest should be a directory-leaf claimed
    # by FileMapSerializer.
    root = _manifest(save_dir)["root"]
    assert root["_t"] == "dir"
    assert root["serializer"].endswith(".FileMapSerializer")

    loaded: FileMap[str] = serde.load(save_dir)
    assert isinstance(loaded, FileMap)
    assert set(loaded) == {"alpha", "beta"}
    assert loaded["alpha"].read_bytes() == b"hello"
    assert loaded["beta"].read_bytes() == b"world"


def test_roundtrip_preserves_extensions(tmp_path: Path) -> None:
    save_dir = tmp_path / "save"
    save_dir.mkdir()

    a = _write(tmp_path / "ckpt.pt", b"weights")
    b = _write(tmp_path / "metadata.json", b'{"v": 1}')

    serde.save(FileMap({"weights": a, "meta": b}), save_dir)
    loaded: FileMap[str] = serde.load(save_dir)

    # On-disk filenames inside the FileMap subdir use index + original suffix.
    cached_paths = list(loaded.values())
    suffixes = {p.suffix for p in cached_paths}
    assert suffixes == {".pt", ".json"}


@pytest.mark.parametrize(
    "key",
    ["alpha", 0, 42, -7, 0.0, 3.14, True, False, None],
)
def test_roundtrip_preserves_key_types(tmp_path: Path, key: object) -> None:
    src = _write(tmp_path / "src.bin", b"data")
    save_dir = tmp_path / "save"
    save_dir.mkdir()

    serde.save(FileMap({key: src}), save_dir)
    loaded: FileMap = serde.load(save_dir)

    (loaded_key,) = loaded.keys()
    assert loaded_key == key
    assert type(loaded_key) is type(key)


def test_roundtrip_loaded_paths_are_inside_load_dir(tmp_path: Path) -> None:
    """Portability: loaded paths resolve relative to *load*, not the source."""
    src = _write(tmp_path / "src.bin", b"data")
    save_dir = tmp_path / "save"
    save_dir.mkdir()

    serde.save(FileMap({"k": src}), save_dir)
    loaded: FileMap[str] = serde.load(save_dir)

    loaded_path = loaded["k"]
    assert loaded_path.is_file()
    assert save_dir in loaded_path.parents


def test_roundtrip_empty(tmp_path: Path) -> None:
    save_dir = tmp_path / "save"
    save_dir.mkdir()

    serde.save(FileMap({}), save_dir)
    loaded: FileMap = serde.load(save_dir)
    assert isinstance(loaded, FileMap)
    assert len(loaded) == 0


def test_roundtrip_same_basename_different_keys(tmp_path: Path) -> None:
    """Two keys can point at files with the same basename without collision."""
    a_dir = tmp_path / "a"
    b_dir = tmp_path / "b"
    a_dir.mkdir()
    b_dir.mkdir()
    a = _write(a_dir / "ckpt.pt", b"A")
    b = _write(b_dir / "ckpt.pt", b"B")
    save_dir = tmp_path / "save"
    save_dir.mkdir()

    serde.save(FileMap({0: a, 1: b}), save_dir)
    loaded: FileMap[int] = serde.load(save_dir)
    assert loaded[0].read_bytes() == b"A"
    assert loaded[1].read_bytes() == b"B"


# ---------------------------------------------------------------------------
# Source lifecycle: serializer always moves
# ---------------------------------------------------------------------------


def test_save_always_moves_source(tmp_path: Path) -> None:
    """Persistence transfers ownership: source paths must be gone after save."""
    src = _write(tmp_path / "src.bin", b"data")
    save_dir = tmp_path / "save"
    save_dir.mkdir()

    serde.save(FileMap({"k": src}), save_dir)
    assert not src.exists(), "FileMap serializer should move (remove) the source"

    loaded: FileMap[str] = serde.load(save_dir)
    assert loaded["k"].read_bytes() == b"data"


def test_save_moves_multiple_sources(tmp_path: Path) -> None:
    """Each entry's source is moved into the cache; none remain at origin."""
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    save_dir = tmp_path / "save"
    save_dir.mkdir()
    a = _write(src_dir / "a.txt", b"alpha")
    b = _write(src_dir / "b.txt", b"beta")

    serde.save(FileMap({0: a, 1: b}), save_dir)
    assert not a.exists()
    assert not b.exists()

    loaded: FileMap[int] = serde.load(save_dir)
    assert loaded[0].read_bytes() == b"alpha"
    assert loaded[1].read_bytes() == b"beta"


# ---------------------------------------------------------------------------
# Composition: FileMap inside containers
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class _Result:
    files: FileMap[int]
    config: dict[str, object]


def test_roundtrip_inside_dataclass(tmp_path: Path) -> None:
    a = _write(tmp_path / "a.pt", b"A")
    b = _write(tmp_path / "b.pt", b"B")
    save_dir = tmp_path / "save"
    save_dir.mkdir()

    original = _Result(
        files=FileMap({100: a, 200: b}),
        config={"lr": 0.001, "dim": 256},
    )
    serde.save(original, save_dir)

    loaded: _Result = serde.load(save_dir)
    assert isinstance(loaded, _Result)
    assert isinstance(loaded.files, FileMap)
    assert loaded.files[100].read_bytes() == b"A"
    assert loaded.files[200].read_bytes() == b"B"
    assert loaded.config == {"lr": 0.001, "dim": 256}


def test_roundtrip_inside_dict(tmp_path: Path) -> None:
    a = _write(tmp_path / "a.pt", b"A")
    save_dir = tmp_path / "save"
    save_dir.mkdir()

    original = {"ckpts": FileMap({"x": a}), "note": "hello"}
    serde.save(original, save_dir)

    loaded = serde.load(save_dir)
    assert isinstance(loaded["ckpts"], FileMap)
    assert loaded["ckpts"]["x"].read_bytes() == b"A"
    assert loaded["note"] == "hello"


# ---------------------------------------------------------------------------
# Error paths on load
# ---------------------------------------------------------------------------


def test_load_missing_entries_raises(tmp_path: Path) -> None:
    src = _write(tmp_path / "src.bin", b"data")
    save_dir = tmp_path / "save"
    save_dir.mkdir()
    serde.save(FileMap({"k": src}), save_dir)

    # Locate the FileMap subdirectory and remove its entries.json.
    root = _manifest(save_dir)["root"]
    subdir = save_dir / "dirs" / root["subdir"]
    (subdir / "entries.json").unlink()

    with pytest.raises(SerializationError, match=r"entries\.json"):
        serde.load(save_dir)


def test_load_malformed_entries_raises(tmp_path: Path) -> None:
    src = _write(tmp_path / "src.bin", b"data")
    save_dir = tmp_path / "save"
    save_dir.mkdir()
    serde.save(FileMap({"k": src}), save_dir)

    root = _manifest(save_dir)["root"]
    subdir = save_dir / "dirs" / root["subdir"]
    (subdir / "entries.json").write_text("not a json array")

    with pytest.raises(SerializationError, match=r"entries\.json"):
        serde.load(save_dir)


# ---------------------------------------------------------------------------
# RAM-bounded write: writer must not load file *contents* into memory
# ---------------------------------------------------------------------------


def test_write_does_not_read_file_contents(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Sanity check: persistence must use link/copy/move, never ``read_bytes``.

    We patch :class:`pathlib.Path.read_bytes` to fail loudly, then save.
    If the implementation accidentally pulls contents through memory, the
    test fails.
    """
    src = _write(tmp_path / "src.bin", b"data")
    save_dir = tmp_path / "save"
    save_dir.mkdir()

    def _no_read_bytes(self: Path) -> bytes:
        msg = f"FileMapSerializer.write must not call Path.read_bytes (called on {self})"
        raise AssertionError(msg)

    monkeypatch.setattr(Path, "read_bytes", _no_read_bytes)
    serde.save(FileMap({"k": src}), save_dir)


# ---------------------------------------------------------------------------
# from_glob convenience constructor
# ---------------------------------------------------------------------------


def test_from_glob_default_key_is_stem(tmp_path: Path) -> None:
    _write(tmp_path / "ckpt_100.pt", b"100")
    _write(tmp_path / "ckpt_500.pt", b"500")
    _write(tmp_path / "unrelated.txt", b"skip")

    fm: FileMap = FileMap.from_glob(tmp_path, "ckpt_*.pt")
    assert set(fm.keys()) == {"ckpt_100", "ckpt_500"}
    assert fm["ckpt_100"].read_bytes() == b"100"


def test_from_glob_with_typed_key_extractor(tmp_path: Path) -> None:
    _write(tmp_path / "ckpt_100.pt", b"100")
    _write(tmp_path / "ckpt_500.pt", b"500")

    fm: FileMap[int] = FileMap.from_glob(
        tmp_path,
        "ckpt_*.pt",
        key=lambda p: int(p.stem.split("_")[1]),
    )
    assert set(fm.keys()) == {100, 500}
    assert fm[100].read_bytes() == b"100"


def test_from_glob_iteration_is_sorted(tmp_path: Path) -> None:
    """Path.glob is filesystem-order-dependent; from_glob must be deterministic."""
    for name in ("c.pt", "a.pt", "b.pt"):
        _write(tmp_path / name, name.encode())

    fm: FileMap = FileMap.from_glob(tmp_path, "*.pt")
    assert list(fm.keys()) == ["a", "b", "c"]


def test_from_glob_supports_recursive_pattern(tmp_path: Path) -> None:
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    _write(tmp_path / "a" / "x.pt", b"ax")
    _write(tmp_path / "b" / "x.pt", b"bx")

    fm: FileMap = FileMap.from_glob(
        tmp_path,
        "**/*.pt",
        key=lambda p: str(p.relative_to(tmp_path)),
    )
    assert set(fm.keys()) == {"a/x.pt", "b/x.pt"}


def test_from_glob_no_matches_returns_empty_map(tmp_path: Path) -> None:
    fm: FileMap = FileMap.from_glob(tmp_path, "missing_*.pt")
    assert len(fm) == 0


def test_from_glob_rejects_duplicate_keys(tmp_path: Path) -> None:
    """Two matches that produce the same key should fail loudly, not silently overwrite."""
    _write(tmp_path / "a.pt", b"A")
    _write(tmp_path / "b.pt", b"B")

    with pytest.raises(ValueError, match="already contains key"):
        FileMap.from_glob(tmp_path, "*.pt", key=lambda _: "same")


def test_from_glob_round_trips(tmp_path: Path) -> None:
    _write(tmp_path / "ckpt_1.pt", b"one")
    _write(tmp_path / "ckpt_2.pt", b"two")
    save_dir = tmp_path / "save"
    save_dir.mkdir()

    fm: FileMap[int] = FileMap.from_glob(
        tmp_path,
        "ckpt_*.pt",
        key=lambda p: int(p.stem.split("_")[1]),
    )
    serde.save(fm, save_dir)
    loaded: FileMap[int] = serde.load(save_dir)
    assert loaded[1].read_bytes() == b"one"
    assert loaded[2].read_bytes() == b"two"


# ---------------------------------------------------------------------------
# from_tree + include_tree
# ---------------------------------------------------------------------------


def test_from_tree_keys_are_relative_paths(tmp_path: Path) -> None:
    root = tmp_path / "logs"
    (root / "run-1").mkdir(parents=True)
    _write(root / "run-1" / "events.0", b"e0")
    _write(root / "config.json", b"{}")

    fm: FileMap[str] = FileMap.from_tree(root)
    assert set(fm.keys()) == {"run-1/events.0", "config.json"}
    assert fm["run-1/events.0"].read_bytes() == b"e0"


def test_from_tree_preserves_layout_on_roundtrip(tmp_path: Path) -> None:
    root = tmp_path / "logs"
    (root / "run-1").mkdir(parents=True)
    _write(root / "run-1" / "events.0", b"e0")
    _write(root / "config.json", b"{}")
    save_dir = tmp_path / "save"
    save_dir.mkdir()

    serde.save(FileMap.from_tree(root), save_dir)
    loaded: FileMap[str] = serde.load(save_dir)

    # Relative structure survives: the loaded paths sit under .root at the
    # same relative layout the tree had.
    assert loaded["run-1/events.0"].read_bytes() == b"e0"
    assert loaded["run-1/events.0"].relative_to(loaded.root).as_posix() == "run-1/events.0"
    assert loaded["config.json"].relative_to(loaded.root).as_posix() == "config.json"


# ---------------------------------------------------------------------------
# Builder: chaining, combining sources, exclusions (eager)
# ---------------------------------------------------------------------------


def test_builder_chains_and_combines_sources(tmp_path: Path) -> None:
    (tmp_path / "tb").mkdir()
    _write(tmp_path / "ckpt_1.pt", b"one")
    _write(tmp_path / "tb" / "events.0", b"ev")

    fm = (
        FileMap()
        .include_glob(tmp_path, "ckpt_*.pt", key=lambda p: int(p.stem.split("_")[1]))
        .include_tree(tmp_path / "tb")
    )
    assert set(fm.keys()) == {1, "events.0"}


def test_exclude_glob_is_eager(tmp_path: Path) -> None:
    _write(tmp_path / "keep.pt", b"k")
    _write(tmp_path / "scratch.tmp", b"t")

    fm = FileMap().include_glob(tmp_path, "*", key=lambda p: p.name).exclude_glob("*.tmp")
    assert set(fm.keys()) == {"keep.pt"}


def test_exclude_glob_before_include_is_noop(tmp_path: Path) -> None:
    """Eager semantics: excluding before the matching include drops nothing."""
    _write(tmp_path / "x.tmp", b"t")
    fm = FileMap().exclude_glob("*.tmp").include_glob(tmp_path, "*", key=lambda p: p.name)
    assert set(fm.keys()) == {"x.tmp"}


def test_exclude_predicate(tmp_path: Path) -> None:
    _write(tmp_path / "big.pt", b"xxxxxxxx")
    _write(tmp_path / "small.pt", b"x")

    fm = FileMap().include_glob(tmp_path, "*.pt", key=lambda p: p.name)
    fm.exclude(lambda _key, path: path.stat().st_size > 4)
    assert set(fm.keys()) == {"small.pt"}


def test_builder_roundtrip_after_exclude(tmp_path: Path) -> None:
    _write(tmp_path / "keep.pt", b"k")
    _write(tmp_path / "drop.tmp", b"d")
    save_dir = tmp_path / "save"
    save_dir.mkdir()

    fm = FileMap().include_glob(tmp_path, "*", key=lambda p: p.name).exclude_glob("*.tmp")
    serde.save(fm, save_dir)
    # Excluded file was never staged, so it stays at the source untouched.
    assert (tmp_path / "drop.tmp").exists()
    assert not (tmp_path / "keep.pt").exists()  # moved into cache

    loaded: FileMap[str] = serde.load(save_dir)
    assert set(loaded.keys()) == {"keep.pt"}


# ---------------------------------------------------------------------------
# .root accessor
# ---------------------------------------------------------------------------


def test_root_raises_before_persist(tmp_path: Path) -> None:
    _write(tmp_path / "a.pt", b"a")
    fm = FileMap().include_glob(tmp_path, "*.pt")
    with pytest.raises(RuntimeError, match="only after the map has been persisted"):
        _ = fm.root


def test_root_after_reload_holds_all_files(tmp_path: Path) -> None:
    root = tmp_path / "logs"
    (root / "run-1").mkdir(parents=True)
    _write(root / "run-1" / "events.0", b"e0")
    save_dir = tmp_path / "save"
    save_dir.mkdir()

    serde.save(FileMap.from_tree(root), save_dir)
    loaded: FileMap[str] = serde.load(save_dir)

    assert loaded.root.is_dir()
    for path in loaded.values():
        assert loaded.root in path.parents


# ---------------------------------------------------------------------------
# Freeze-on-reload + copy-on-reserialize
# ---------------------------------------------------------------------------


def test_reloaded_map_is_frozen(tmp_path: Path) -> None:
    src = _write(tmp_path / "src.bin", b"data")
    save_dir = tmp_path / "save"
    save_dir.mkdir()
    serde.save(FileMap({"k": src}), save_dir)
    loaded: FileMap[str] = serde.load(save_dir)

    with pytest.raises(RuntimeError, match="read-only"):
        loaded.include(0, loaded["k"])
    with pytest.raises(RuntimeError, match="read-only"):
        loaded.exclude_glob("*")


def test_reserializing_reloaded_map_preserves_upstream(tmp_path: Path) -> None:
    """A reloaded FileMap re-persisted (passthrough task) must not move the upstream cache."""
    src = _write(tmp_path / "src.bin", b"data")
    save_a = tmp_path / "a"
    save_b = tmp_path / "b"
    save_a.mkdir()
    save_b.mkdir()

    serde.save(FileMap({"k": src}), save_a)
    loaded_a: FileMap[str] = serde.load(save_a)
    upstream_path = loaded_a["k"]
    assert upstream_path.is_file()

    # Re-persist the loaded map into a second result directory.
    serde.save(loaded_a, save_b)

    # Upstream A's file must still be present (copied/linked, not moved away).
    assert upstream_path.is_file()
    loaded_b: FileMap[str] = serde.load(save_b)
    assert loaded_b["k"].read_bytes() == b"data"
