"""Serializer for BeautifulSoup objects via HTML string."""

import importlib.util
from typing import Any

from misen.utils.serde.serializer_base import (
    Serializer,
    SerializerTypeList,
    SerializerTypeRegistry,
    write_meta,
)

__all__ = ["beautifulsoup_serializers", "beautifulsoup_serializers_by_type"]

beautifulsoup_serializers: SerializerTypeList = []
beautifulsoup_serializers_by_type: SerializerTypeRegistry = {}

if importlib.util.find_spec("bs4") is not None:
    from pathlib import Path

    class BeautifulSoupSerializer(Serializer[Any]):
        """Serialize ``BeautifulSoup`` via HTML markup string."""

        version = 1

        @staticmethod
        def match(obj: Any) -> bool:
            from bs4 import BeautifulSoup

            return isinstance(obj, BeautifulSoup)

        @staticmethod
        def save(obj: Any, directory: Path) -> None:
            markup = str(obj)
            (directory / "data.html").write_text(markup, encoding="utf-8")
            write_meta(
                directory,
                BeautifulSoupSerializer,
                parser=obj.builder.NAME if obj.builder else "html.parser",
            )

        @staticmethod
        def load(directory: Path) -> Any:
            from bs4 import BeautifulSoup

            from misen.utils.serde.serializer_base import read_meta

            markup = (directory / "data.html").read_text(encoding="utf-8")
            meta = read_meta(directory)
            parser = meta.get("parser", "html.parser") if meta else "html.parser"
            return BeautifulSoup(markup, parser)

    beautifulsoup_serializers = [BeautifulSoupSerializer]
    beautifulsoup_serializers_by_type = {"bs4.BeautifulSoup": BeautifulSoupSerializer}
