from pathlib import Path
from typing import Any

import dill
import msgspec.msgpack

__all__ = ["to_files", "from_files"]


def to_files(obj: Any, dir: Path) -> None:
    try:
        (dir / "data.msgpack").write_bytes(msgspec.msgpack.encode(obj))
    except NotImplementedError:
        (dir / "data.dill").write_bytes(dill.dumps(obj))


def from_files(dir: Path) -> Any:
    if (dir / "data.msgpack").exists():
        return msgspec.msgpack.decode((dir / "data.msgpack").read_bytes())
    return dill.loads((dir / "data.dill").read_bytes())
