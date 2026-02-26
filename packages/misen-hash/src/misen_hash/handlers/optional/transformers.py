"""Handlers for Hugging Face transformers objects."""

import importlib.util
from typing import Any

from misen_hash.handler_base import CollectionHandler, HandlerTypeList, HandlerTypeRegistry

__all__ = ["transformers_handlers", "transformers_handlers_by_type"]


transformers_handlers: HandlerTypeList = []
transformers_handlers_by_type: HandlerTypeRegistry = {}

if importlib.util.find_spec("transformers") is not None:

    def _has_transformers_base(obj: Any, type_name: str) -> bool:
        return any(
            base.__name__ == type_name and base.__module__.split(".")[0] == "transformers"
            for base in type(obj).__mro__
        )

    class TransformersConfigHandler(CollectionHandler):
        """Hash transformers configs by class identity and serialized payload."""

        @staticmethod
        def match(obj: Any) -> bool:
            return _has_transformers_base(obj, "PretrainedConfig")

        @staticmethod
        def elements(obj: Any) -> list[Any]:
            payload = obj.to_dict() if hasattr(obj, "to_dict") else vars(obj)
            return [payload]

    class TransformersTokenizerHandler(CollectionHandler):
        """Hash transformers tokenizers by vocab/special-token payload."""

        @staticmethod
        def match(obj: Any) -> bool:
            return _has_transformers_base(obj, "PreTrainedTokenizerBase")

        @staticmethod
        def elements(obj: Any) -> list[Any]:
            tokenizer_json = None
            backend = getattr(obj, "backend_tokenizer", None)
            if backend is not None and hasattr(backend, "to_str"):
                tokenizer_json = backend.to_str()

            vocab = obj.get_vocab() if hasattr(obj, "get_vocab") else None

            return [
                getattr(obj, "name_or_path", None),
                getattr(obj, "init_kwargs", None),
                getattr(obj, "special_tokens_map", None),
                getattr(obj, "model_max_length", None),
                vocab,
                tokenizer_json,
            ]

    class TransformersModelHandler(CollectionHandler):
        """Hash transformers models by config and state dict."""

        @staticmethod
        def match(obj: Any) -> bool:
            return _has_transformers_base(obj, "PreTrainedModel")

        @staticmethod
        def elements(obj: Any) -> list[Any]:
            config_payload = None
            if hasattr(obj, "config") and hasattr(obj.config, "to_dict"):
                config_payload = obj.config.to_dict()

            state_entries = None
            if hasattr(obj, "state_dict"):
                state_dict = obj.state_dict()
                sorted_keys = sorted(state_dict)
                state_entries = [(key, state_dict[key]) for key in sorted_keys]

            return [
                config_payload,
                state_entries,
            ]

    transformers_handlers = [
        TransformersConfigHandler,
        TransformersTokenizerHandler,
        TransformersModelHandler,
    ]
    transformers_handlers_by_type = {
        "transformers.configuration_utils.PretrainedConfig": TransformersConfigHandler,
        "transformers.tokenization_utils_base.PreTrainedTokenizerBase": TransformersTokenizerHandler,
        "transformers.modeling_utils.PreTrainedModel": TransformersModelHandler,
    }
