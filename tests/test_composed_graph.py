import logging
import os
import sys

from misen import ASSIGNED_RESOURCES, Experiment, Task, task
from misen.utils.assigned_resources import AssignedResources

logging.basicConfig(level=logging.DEBUG, format="%(levelname)s %(name)s %(message)s")
logging.getLogger("misen").setLevel(logging.DEBUG)


@task(id="add", cache=False, exclude={"x"})
def add(a: float, b: float, x: AssignedResources | None = None) -> float:
    print(f"Running add with {a}, {b}")
    print(f"{os.environ['MY_ENV_VAR']}")
    print(sys.executable)
    print("Assigned Resources:", x)
    return a + b


@task(id="multiply", cache=True)
def multiply(a: float, b: float) -> float:
    print(f"[CACHE] Running multiply with {a}, {b}")
    return a * b


@task(id="square", cache=False)
def square(x: float) -> float:
    print(f"Running square with {x}")
    return x * x


@task(id="sum_list", cache=True)
def sum_list(numbers: list[float]) -> float:
    print(f"[CACHE] Running sum_list with {numbers}")
    return sum(numbers)


@task(id="mean", cache=False)
def mean(numbers: list[float]) -> float:
    print(f"Running mean with {numbers}")
    total = sum_list(numbers)
    return total / len(numbers)


@task(id="variance", cache=True)
def variance(numbers: list[float]) -> float:
    print(f"[CACHE] Running variance with {numbers}")
    mean_val = mean(numbers)
    squared_diffs = [square(x - mean_val) for x in numbers]
    return sum_list(squared_diffs) / len(numbers)


@task(id="generate_numbers", cache=False)
def generate_numbers(n: int) -> list[float]:
    print(f"Running generate_numbers with {n}")
    return [float(i) for i in range(1, n + 1)]


class MyExperiment(Experiment):
    n: int = 5

    def tasks(self):
        gen_task = Task(generate_numbers, n=self.n)
        sum_task = Task(sum_list, numbers=gen_task.T)
        mean_task = Task(mean, numbers=gen_task.T)
        var_task = Task(variance, numbers=gen_task.T)
        mult_task = Task(multiply, a=sum_task.T, b=mean_task.T)

        add_task = Task(add, a=mult_task.T, b=var_task.T, x=ASSIGNED_RESOURCES)
        add_task_dup = Task(add, a=mult_task.T, b=var_task.T, x=ASSIGNED_RESOURCES)

        final_multiply_task = Task(multiply, a=add_task.T, b=add_task_dup.T)

        other_task = Task(add, a=5, b=10)

        # Expected calculations:
        # numbers = [1,2,3,4,5]
        # sum = 15
        # mean = 3.0
        # variance = (sum of (x-3)^2 for x in [1,2,3,4,5]) / 5 = (4+1+0+1+4)/5 = 10/5 = 2.0
        # mult = 15 * 3.0 = 45.0
        # final = 45.0 + 2.0 = 47.0

        return {
            "numbers": gen_task.T,
            "sum": sum_task.T,
            "mean": mean_task.T,
            "variance": var_task.T,
            "mult": mult_task.T,
            "final": final_multiply_task.T,
            "other": other_task.T,
        }


if __name__ == "__main__":
    MyExperiment.cli()
