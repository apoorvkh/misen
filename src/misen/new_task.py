# TODO: integrate this with task.py

from typing import Callable, ParamSpec, TypeVar

P = ParamSpec("P")
R = TypeVar("R")


def my_function(a: int) -> int:
    return a * 2


class Task:
    def __new__(cls, func: Callable[P, R], *args: P.args, **kwargs: P.kwargs) -> R:
        return super().__new__(cls)  # type: ignore

    def __init__(self, func: Callable[P, R], *args: P.args, **kwargs: P.kwargs):
        self.func = func
        self.args = args
        self.kwargs = kwargs

x = Task(my_function, a = 3)
