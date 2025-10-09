import torch
import triton
import triton.language as tl
from aiter.ops.triton._triton_kernels.swiglu import _swiglu_kernel
from aiter.ops.triton.utils.logger import AiterTritonLogger

_LOGGER = AiterTritonLogger()


def swiglu(x, w):
    """
    Computes the SwiGLU of a 2D input tensor. It will first compute
    a GEMM of the input 'x' with the weights 'w'. The output is then split
    into two halves. The first half will be fed into the Swish activation
    function, then compute the Hadamard product with the second half.

    Key parameters:
        x (torch.Tensor): A 2D input tensor.

    Returns:
        torch.Tensor: A tensor of the shape of half the number of
        columns as x@w.

    Note:
        - The input tensors 'x' and 'w' must reside on the GPU.
    """
    _LOGGER.info(f"SWIGLU: x={tuple(x.shape)} w={tuple(w.shape)}")
    n_rows, _ = x.shape
    _, n_cols = w.shape

    assert n_cols % 2 == 0, "Weight tensor 'w' must be a multiple of 2."

    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(n_cols // 2))
    y = torch.empty((n_rows, n_cols // 2)).cuda()

    waves_per_eu = 2
    num_warps = 8
    num_stages = 2

    grid = lambda meta: (
        triton.cdiv(n_rows, BLOCK_SIZE) * triton.cdiv(n_cols // 2, BLOCK_SIZE),
    )
    xw = x @ w
    _swiglu_kernel[grid](
        xw,
        y,
        xw.stride(0),
        xw.stride(1),
        y.stride(0),
        y.stride(1),
        n_rows,
        n_cols,
        BLOCK_SIZE,
        waves_per_eu=waves_per_eu,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    return y
