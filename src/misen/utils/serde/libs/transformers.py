"""Serializers for HuggingFace transformers models, tokenizers, and processors."""

import importlib.util
from collections.abc import Mapping
from typing import Any

from misen.utils.serde import (
    Serializer,
    SerializerTypeRegistry,
)

__all__ = ["transformers_serializers", "transformers_serializers_by_type"]

transformers_serializers: list[type[Serializer]] = []
transformers_serializers_by_type: SerializerTypeRegistry = {}

if importlib.util.find_spec("transformers") is not None:
    from pathlib import Path

    class TransformersModelSerializer(Serializer[Any]):
        """Serialize HuggingFace ``PreTrainedModel`` via ``save_pretrained``.

        Uses ``AutoModel.from_pretrained`` on load, which reads the model
        architecture from the saved ``config.json``.
        """

        @staticmethod
        def match(obj: Any) -> bool:
            from transformers import PreTrainedModel

            return isinstance(obj, PreTrainedModel)

        @staticmethod
        def write(obj: Any, directory: Path) -> Mapping[str, Any]:
            import transformers

            obj.save_pretrained(str(directory))
            return {"transformers_version": transformers.__version__}

        @staticmethod
        def read(directory: Path, *, meta: Mapping[str, Any]) -> Any:  # noqa: ARG004
            from transformers import AutoModel

            return AutoModel.from_pretrained(str(directory))

    class TransformersTokenizerSerializer(Serializer[Any]):
        """Serialize HuggingFace ``PreTrainedTokenizerBase`` via ``save_pretrained``.

        Uses ``AutoTokenizer.from_pretrained`` on load.
        """

        @staticmethod
        def match(obj: Any) -> bool:
            from transformers import PreTrainedTokenizerBase

            return isinstance(obj, PreTrainedTokenizerBase)

        @staticmethod
        def write(obj: Any, directory: Path) -> Mapping[str, Any]:
            import transformers

            obj.save_pretrained(str(directory))
            return {"transformers_version": transformers.__version__}

        @staticmethod
        def read(directory: Path, *, meta: Mapping[str, Any]) -> Any:  # noqa: ARG004
            from transformers import AutoTokenizer

            return AutoTokenizer.from_pretrained(str(directory))

    class TransformersProcessorSerializer(Serializer[Any]):
        """Serialize HuggingFace ``ProcessorMixin`` (multimodal processors, etc.).

        Uses ``AutoProcessor.from_pretrained`` on load.
        """

        @staticmethod
        def match(obj: Any) -> bool:
            from transformers import ProcessorMixin

            return isinstance(obj, ProcessorMixin)

        @staticmethod
        def write(obj: Any, directory: Path) -> Mapping[str, Any]:
            import transformers

            obj.save_pretrained(str(directory))
            return {"transformers_version": transformers.__version__}

        @staticmethod
        def read(directory: Path, *, meta: Mapping[str, Any]) -> Any:  # noqa: ARG004
            from transformers import AutoProcessor

            return AutoProcessor.from_pretrained(str(directory))

    class TransformersImageProcessorSerializer(Serializer[Any]):
        """Serialize HuggingFace ``BaseImageProcessor`` via ``save_pretrained``.

        Uses ``AutoImageProcessor.from_pretrained`` on load.
        """

        @staticmethod
        def match(obj: Any) -> bool:
            from transformers import BaseImageProcessor

            return isinstance(obj, BaseImageProcessor)

        @staticmethod
        def write(obj: Any, directory: Path) -> Mapping[str, Any]:
            import transformers

            obj.save_pretrained(str(directory))
            return {"transformers_version": transformers.__version__}

        @staticmethod
        def read(directory: Path, *, meta: Mapping[str, Any]) -> Any:  # noqa: ARG004
            from transformers import AutoImageProcessor

            return AutoImageProcessor.from_pretrained(str(directory))

    transformers_serializers = [
        TransformersModelSerializer,
        TransformersTokenizerSerializer,
        TransformersImageProcessorSerializer,
        TransformersProcessorSerializer,
    ]
    transformers_serializers_by_type = {
        "transformers.modeling_utils.PreTrainedModel": TransformersModelSerializer,
        "transformers.tokenization_utils_base.PreTrainedTokenizerBase": TransformersTokenizerSerializer,
        "transformers.image_processing_utils.BaseImageProcessor": TransformersImageProcessorSerializer,
        "transformers.processing_utils.ProcessorMixin": TransformersProcessorSerializer,
    }
