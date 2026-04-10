"""Serializer for lxml Element and ElementTree via XML bytes."""

import importlib.util
from typing import Any

from misen.utils.serde.serializer_base import (
    Serializer,
    SerializerTypeList,
    SerializerTypeRegistry,
    write_meta,
)

__all__ = ["lxml_serializers", "lxml_serializers_by_type"]

lxml_serializers: SerializerTypeList = []
lxml_serializers_by_type: SerializerTypeRegistry = {}

if importlib.util.find_spec("lxml") is not None:
    from pathlib import Path

    class LxmlElementSerializer(Serializer[Any]):
        """Serialize lxml ``_Element`` via XML bytes."""

        version = 1

        @staticmethod
        def match(obj: Any) -> bool:
            from lxml import etree

            return isinstance(obj, etree._Element)  # noqa: SLF001

        @staticmethod
        def save(obj: Any, directory: Path) -> None:
            from lxml import etree

            xml_bytes = etree.tostring(obj, xml_declaration=True, encoding="utf-8")
            (directory / "data.xml").write_bytes(xml_bytes)
            write_meta(directory, LxmlElementSerializer)

        @staticmethod
        def load(directory: Path) -> Any:
            from lxml import etree

            xml_bytes = (directory / "data.xml").read_bytes()
            return etree.fromstring(xml_bytes)

    class LxmlElementTreeSerializer(Serializer[Any]):
        """Serialize lxml ``_ElementTree`` via XML bytes."""

        version = 1

        @staticmethod
        def match(obj: Any) -> bool:
            from lxml import etree

            return isinstance(obj, etree._ElementTree)  # noqa: SLF001

        @staticmethod
        def save(obj: Any, directory: Path) -> None:
            obj.write(str(directory / "data.xml"), xml_declaration=True, encoding="utf-8")
            write_meta(directory, LxmlElementTreeSerializer)

        @staticmethod
        def load(directory: Path) -> Any:
            from lxml import etree

            return etree.parse(str(directory / "data.xml"))

    lxml_serializers = [LxmlElementTreeSerializer, LxmlElementSerializer]
    lxml_serializers_by_type = {
        "lxml.etree._ElementTree": LxmlElementTreeSerializer,
        "lxml.etree._Element": LxmlElementSerializer,
    }
