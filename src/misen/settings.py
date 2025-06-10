import os
import sys
from functools import cached_property
from pathlib import Path
from typing import Any

from msgspec import Struct

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib


class Settings(Struct, dict=True):
    file: Path = Path(os.environ.get("MISEN_SETTINGS_FILE") or (Path.cwd() / "misen_settings.toml"))

    @cached_property
    def toml_data(self) -> dict[str, Any]:
        try:
            return tomllib.loads(self.file.read_bytes().decode())
        except FileNotFoundError:
            return {}
