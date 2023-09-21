import random
from functools import wraps
from typing import Callable, ParamSpec, TypeVar

T = TypeVar("T")
P = ParamSpec("P")


def partition_randomly(xs: list[int]) -> list[list[int]]:
    xs = list(xs)
    random.shuffle(xs)

    out = []
    i = 0
    while i < len(xs):
        size = random.randint(1, len(xs) - i)
        out.append(xs[i : i + size])
        i += size

    return out


def seed_rng(seed: int) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Use a function (a test case, for example) with this decorator to seed the random number generator with the given seed."""

    def seeding_decorator(func: Callable[P, T]) -> Callable[P, T]:
        @wraps(func)
        def wrapped_function(*args: P.args, **kwargs: P.kwargs) -> T:
            random.seed(seed)
            return func(*args, **kwargs)

        return wrapped_function

    return seeding_decorator


def search_seed(start: int = 0, stop: int = 2000) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Search a seed for the RNG so that a given test passes."""

    def seeding_decorator(func: Callable[P, T]) -> Callable[P, T]:
        @wraps(func)
        def wrapped_function(*args: P.args, **kwargs: P.kwargs) -> T:
            for seed in range(start, stop + 1):
                random.seed(seed)
                try:
                    func(*args, **kwargs)
                    # No assertion error
                    raise ValueError(f"Found the seed {seed} for function {func}")
                except AssertionError:
                    continue
            raise AssertionError(f"No valid seed found in range {start}..{stop} for function `{func.__name__}`!")

        return wrapped_function

    return seeding_decorator
