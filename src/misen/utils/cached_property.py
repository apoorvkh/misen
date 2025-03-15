from collections.abc import Callable
from typing import ParamSpec, TypeVar

P = ParamSpec("P")
R = TypeVar("R")

# TODO: test or fix this
# supposed to be a cached property decorator without locking

def cached_property(
    key: str,
    condition: Callable[[], bool] = lambda: True,
) -> Callable[[Callable[P, R]], Callable[P, R]]:

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        if condition() is False:
            return func

        def cache_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            self = args[0]
            if not hasattr(self, key):
                setattr(self, key, func(*args, **kwargs))
            return getattr(self, key)

        return cache_wrapper

    return decorator
