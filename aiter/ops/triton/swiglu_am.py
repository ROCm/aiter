# ripped from softmax

import torch
import triton
import triton.language as tl
from aiter.ops.triton._triton_kernels.swiglu_am import _swiglu
from aiter.ops.triton.utils.logger import AiterTritonLogger

_LOGGER = AiterTritonLogger()


def swiglu(x, weights):
    """
    ripped from softmax, tests my implementation of swiglu according to how it would be in the FC1 layer
    """
    _LOGGER.info(f"swiglu: x={tuple(x.shape)}")
    n_rows, n_cols = x.shape
    w_rows, w_cols = weights.shape

    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(n_cols))
    y = torch.empty_like(x)

    waves_per_eu = 2
    num_warps = 8
    num_stages = 2

    num_programs = n_rows

    grid = lambda meta: (num_programs,)  # noqa: E731
    _swiglu[grid](
        y,
        x,
        weights,
        x.stride(0),
        weights.stride(0),
        weights.stride(0),
        y.stride(0),
        y.stride(0),
        y.stride(0),
        n_rows,
        n_cols,
        w_cols,
        BLOCK_SIZE,
        BLOCK_SIZE,
        BLOCK_SIZE, # for now
        waves_per_eu=waves_per_eu,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    return y
