import torch
import triton
# import triton.language as tl
from aiter.ops.triton._triton_kernels.softmax import _softmax_kernel_online
from aiter.ops.triton.utils.logger import AiterTritonLogger

_LOGGER = AiterTritonLogger()


def softmax(x):
    """
    Computes row-wise softmax of a 2D input tensor.

    Args:
        x (torch.Tensor): Input tensor with shape (n_rows, n_cols). Must be on GPU.

    Returns:
        torch.Tensor: Output with same shape as x, softmax applied along last dimension.
    """
    _LOGGER.info(f"SOFTMAX: x={tuple(x.shape)}")
    n_rows, n_cols = x.shape

    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(n_cols))
    y = torch.empty_like(x)

    waves_per_eu = 2
    num_warps = 8
    num_stages = 2

    num_programs = n_rows

    grid = lambda meta: (num_programs,)  # noqa: E731
    _softmax_kernel_online[grid](
        y,
        x,
        x.stride(0),
        y.stride(0),
        n_rows,  # it's not being used in the kernel
        n_cols,
        BLOCK_SIZE,
        waves_per_eu=waves_per_eu,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    return y
