from typing import Callable, TypeVar

T = TypeVar("T")
R = TypeVar("R")


def cached_property(self: T, func: Callable[[T], R], key: str) -> Callable[[], R]:
    def wrapper() -> R:
        if not hasattr(self, key):
            setattr(self, key, func(self))
        return getattr(self, key)

    return wrapper
