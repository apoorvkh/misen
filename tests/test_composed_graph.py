from misen import Task, task
from misen.executors.slurm import SlurmExecutor
from misen.workspaces.disk import DiskWorkspace


@task(id="add", cache=False)
def add(a: float, b: float) -> float:
    print(f"Running add with {a}, {b}")
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


def graph() -> Task:
    gen_task = Task(generate_numbers, n=5)
    sum_task = Task(sum_list, numbers=gen_task.T)
    mean_task = Task(mean, numbers=gen_task.T)
    var_task = Task(variance, numbers=gen_task.T)
    mult_task = Task(multiply, a=sum_task.T, b=mean_task.T)

    add_task = Task(add, a=mult_task.T, b=var_task.T)
    add_task_dup = Task(add, a=mult_task.T, b=var_task.T)

    final_multiply_task = Task(multiply, a=add_task.T, b=add_task_dup.T)

    # Expected calculations:
    # numbers = [1,2,3,4,5]
    # sum = 15
    # mean = 3.0
    # variance = (sum of (x-3)^2 for x in [1,2,3,4,5]) / 5 = (4+1+0+1+4)/5 = 10/5 = 2.0
    # mult = 15 * 3.0 = 45.0
    # final = 45.0 + 2.0 = 47.0

    return final_multiply_task


if __name__ == "__main__":
    workspace = DiskWorkspace()
    executor = SlurmExecutor()
    t = graph()
    executor.submit(t, workspace=workspace)
