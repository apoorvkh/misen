"""Serializers for HuggingFace transformers models and tokenizers."""

import importlib.util
from typing import Any

from misen.utils.serde.serializer_base import (
    Serializer,
    SerializerTypeList,
    SerializerTypeRegistry,
    write_meta,
)

__all__ = ["transformers_serializers", "transformers_serializers_by_type"]

transformers_serializers: SerializerTypeList = []
transformers_serializers_by_type: SerializerTypeRegistry = {}

if importlib.util.find_spec("transformers") is not None:
    from pathlib import Path

    class TransformersModelSerializer(Serializer[Any]):
        """Serialize HuggingFace ``PreTrainedModel`` via ``save_pretrained``.

        Uses ``AutoModel.from_pretrained`` on load, which reads the model
        architecture from the saved ``config.json``.
        """

        version = 1

        @staticmethod
        def match(obj: Any) -> bool:
            from transformers import PreTrainedModel

            return isinstance(obj, PreTrainedModel)

        @staticmethod
        def save(obj: Any, directory: Path) -> None:
            import transformers

            obj.save_pretrained(str(directory))
            write_meta(directory, TransformersModelSerializer, transformers_version=transformers.__version__)

        @staticmethod
        def load(directory: Path) -> Any:
            from transformers import AutoModel

            return AutoModel.from_pretrained(str(directory))

    class TransformersTokenizerSerializer(Serializer[Any]):
        """Serialize HuggingFace ``PreTrainedTokenizerBase`` via ``save_pretrained``.

        Uses ``AutoTokenizer.from_pretrained`` on load.
        """

        version = 1

        @staticmethod
        def match(obj: Any) -> bool:
            from transformers import PreTrainedTokenizerBase

            return isinstance(obj, PreTrainedTokenizerBase)

        @staticmethod
        def save(obj: Any, directory: Path) -> None:
            import transformers

            obj.save_pretrained(str(directory))
            write_meta(directory, TransformersTokenizerSerializer, transformers_version=transformers.__version__)

        @staticmethod
        def load(directory: Path) -> Any:
            from transformers import AutoTokenizer

            return AutoTokenizer.from_pretrained(str(directory))

    transformers_serializers = [TransformersModelSerializer, TransformersTokenizerSerializer]
    transformers_serializers_by_type = {
        "transformers.modeling_utils.PreTrainedModel": TransformersModelSerializer,
        "transformers.tokenization_utils_base.PreTrainedTokenizerBase": TransformersTokenizerSerializer,
    }
