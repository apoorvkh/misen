from typing import Callable, ParamSpec, TypeVar, Generic

P = ParamSpec("P")
R = TypeVar("R")


def my_function(a: int) -> int:
    return a * 2


class Task(Generic[R]):
    def __init__(self, func: Callable[P, R], *args: P.args, **kwargs: P.kwargs):
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def as_argument(self) -> R:
        return self  # type: ignore


a: int = Task(my_function, a=3).as_argument()
x: Task[int] = Task(my_function, a=a)
