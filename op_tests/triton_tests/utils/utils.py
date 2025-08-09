import functools
import random
from typing import Optional


def minimal_x_vals(x_vals, num_vals: int = 20, sample: Optional[int] = None):
    """
    Returns the num_vals smallest test cases from a list of x. Useful for generating a subset to quickly test on.
    """
    num_ops = [(i, functools.reduce(lambda x, y: x * y, i)) for i in x_vals]
    sorted_x_vals = sorted(num_ops, key=lambda x: x[1])
    min_x_vals = [i[0] for i in sorted_x_vals[: min(num_vals, len(sorted_x_vals))]]
    if sample is not None:
        min_x_vals = random.sample(min_x_vals, min(sample, len(min_x_vals)))
    return min_x_vals
