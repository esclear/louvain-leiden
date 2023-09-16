import random


def partition_randomly(xs: list[int]) -> list[list[int]]:
    xs = list(xs)
    random.shuffle(xs)

    out = []
    i = 0
    while i < len(xs):
        size = random.randint(1, len(xs) - i)
        out.append(xs[i:i+size])
        i += size

    return out
