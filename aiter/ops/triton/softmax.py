import torch
import triton
import triton.language as tl
from aiter.ops.triton._triton_kernels.softmax import _softmax_kernel_online
from aiter.ops.triton.utils.logger import AiterTritonLogger

_LOGGER = AiterTritonLogger()


def softmax(x):
    """
    Computes the row-wise softmax of a 2D input tensor.

    Key parameters:
        x (torch.Tensor): A 2D input tensor.

    Returns:
        torch.Tensor: A tensor of the same shape as 'x', where softmax has been
        applied along the last dimension (row-wise).

    Note:
        - The input tensor 'x' must reside on the GPU.
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
        n_rows,
        n_cols,
        BLOCK_SIZE,
        waves_per_eu=waves_per_eu,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    return y
