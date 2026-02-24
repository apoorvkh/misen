from misen_hash import CollectionHandler, canonical_hash
from misen_hash._dill import DillHandler
from misen_hash._torch import TorchModuleHandler


class _Custom:
    pass


def test_dill_handler_matches_any_object() -> None:
    assert DillHandler.match(object()) is True


def test_torch_module_handler_uses_collection_contract() -> None:
    assert issubclass(TorchModuleHandler, CollectionHandler)


def test_canonical_hash_falls_back_to_dill_handler() -> None:
    assert isinstance(canonical_hash(_Custom()), int)
