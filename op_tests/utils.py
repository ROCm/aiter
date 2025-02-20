from functools import partial
import numpy as np
import os
import torch
import torch.profiler as tpf
from typing import Callable, Generic, TypeVar

T = TypeVar("T")

MAX_RAND_INT = 20
MIN_RAND_INT = -20

def rand_tensor(
    size: torch.Size,
    dtype: torch.dtype
) -> torch.tensor:
    """
    Generate a random PyTorch tensor with specified shape and data type.

    - For integer types: Uses torch.randint to generate random integers within a fixed range.
    - For float types: Uses torch.rand to generate random floats between 0 and 1.

    Parameters
    ----------
    size: torch.Size
        The size of the generated tensor.
    dtype: torch.dtype
        The data type of the generated tensor.

    Returns
    -------
    torch.Tensor : A random tensor of the specified shape and data type.

    Raises
    ------
    ValueError
        If an unsupported data type is provided.
    """
    if dtype in [torch.int8, torch.int16, torch.int32, torch.int64]:
        # For integer types, use randint
        return torch.randint(MIN_RAND_INT, MAX_RAND_INT, size=size, dtype=dtype)
    elif dtype in [torch.float16, torch.float32, torch.float64, torch.bfloat16]:
        # For float types, use rand
        return torch.rand(size=size, dtype=dtype)
    elif dtype == torch.float8_e4m3fnuz:
        # Special case for float8_e4m3fnuz
        return torch.rand(size=size, dtype=torch.float16).to(torch.float8_e4m3fnuz)

    raise ValueError(f"Unsupported dtype: {dtype}")

def check_all_close(
    a: torch.tensor,
    b: torch.tensor,
    rtol: float,
    atol: float,
) -> None:
    """
    Check if all elements in two tensors are close within specified tolerances.
    
    Parameters
    ----------
    a: torch.tensor
        First input tensor.
    b: torch.tensor
        Second input tensor.
    rtol: float
        Relative tolerence.
    atol: float
        Absolute tolerence.

    Raises
    -------
    AssertionError
        If any elements in 'a' and 'b' are not close within the specified tolerances.
        The error message includes details about the maximum and average delta,
        and the percentage of elements that are not close.
    """
    is_close = torch.isclose(a, b, rtol=rtol, atol=atol)
    is_not_close = ~is_close
    num_not_close = is_not_close.sum()
    delta = (a-b)[is_not_close]
    percent = num_not_close/a.numel()
    message = "" if num_not_close == 0 else f"""
check_all_close failed!
max delta:{delta.max()}
average delta:{delta.mean()}
delta details: {percent:.1%} ({num_not_close} of {a.numel()}) elements
        """
    assert is_close.all(), message
