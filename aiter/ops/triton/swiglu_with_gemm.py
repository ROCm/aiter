import torch
import triton
import triton.language as tl
from aiter.ops.triton._triton_kernels.swiglu_with_gemm import _swiglu_with_gemm_kernel
from aiter.ops.triton.utils.logger import AiterTritonLogger

_LOGGER = AiterTritonLogger()


def swiglu_with_gemm(x, w):
    """
    Computes the SwiGLU of a 2D input tensor. It will split the input 'x'
    into two halves. The first half will be fed into a Swish activation
    function, then compute the Hadamard product with the second half.

    Key parameters:
        x (torch.Tensor): A 2D input tensor.
        w (torch.Tensor): A 2D weights tensor.

    Returns:
        torch.Tensor: A tensor of the shape of half the number of
        columns as 'xw'.

    Note:
        - The input tensor 'x' must reside on the GPU.
        - The input tensor 'w' must reside on the GPU.
    """
    _LOGGER.info(f"SWIGLU: x={tuple(x.shape)} w={tuple(w.shape)}")
    M, N = x.shape
    N, K2 = w.shape

    assert K2 % 2 == 0, "Weight tensor 'w' columns must be a multiple of 2."

    K = K2 // 2

    MAX_BLOCK_SIZE = int(tl.TRITON_MAX_TENSOR_NUMEL**0.5)
    MAX_FUSED_SIZE = int(65536**0.5 / max(x.element_size(), 4))
    BLOCK_SIZE = min(
        triton.next_power_of_2(MAX_BLOCK_SIZE),
        triton.next_power_of_2(MAX_FUSED_SIZE),
        triton.next_power_of_2(K),
    )
    y = torch.empty((M, K), dtype=x.dtype).cuda()

    waves_per_eu = 2
    num_warps = 8
    num_stages = 2

    grid = lambda meta: (triton.cdiv(M, BLOCK_SIZE) * triton.cdiv(K, BLOCK_SIZE),)
    _swiglu_with_gemm_kernel[grid](
        x,
        w,
        y,
        x.stride(0),
        x.stride(1),
        w.stride(0),
        w.stride(1),
        y.stride(0),
        y.stride(1),
        M,
        N,
        K,
        BLOCK_SIZE,
        waves_per_eu=waves_per_eu,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    return y
