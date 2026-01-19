import torch
import triton
from typing import Optional
from aiter.ops.triton._triton_kernels.fusions.fused_mul_add import _fused_mul_add_kernel
from aiter.ops.triton.utils.logger import AiterTritonLogger

_LOGGER = AiterTritonLogger()


def fused_mul_add(
    x: torch.Tensor,
    a: torch.Tensor | float | int,
    b: torch.Tensor | float | int,
    out: Optional[torch.Tensor] = None,
):
    """
    Computes elementwise multiplicated and addtion: out = x * a + b

    Key parameters:
    - x: must be a torch.Tensor, but with arbitrary shape,
    - a: can be float, int, or torch.Tensor with shape (1, ) or the same shape as x
    - b: can be float, int, or torch.Tensor with shape (1, ) or the same shape as x

    all tensors must be contiguous

    if out is None, the kernel will perform inplace computation on x instead of creating a new tensor

    Returns:
    - out: same shape as x
    """
    _LOGGER.info(
        f"FUSED_MUL_ADD: x={tuple(x.shape)} a={tuple(a.shape) if isinstance(a, torch.Tensor) else a} b={tuple(b.shape) if isinstance(b, torch.Tensor) else b}"
    )

    N = x.numel()
    assert x.is_contiguous(), "x should be contiguous"
    assert (
        isinstance(a, float)
        or isinstance(a, int)
        or (isinstance(a, torch.Tensor) and a.is_contiguous() and a.numel() in [1, N])
    ), "a should be a scalar or contiguous tensor with the same number of elements as x"
    assert (
        isinstance(b, float)
        or isinstance(b, int)
        or (isinstance(b, torch.Tensor) and b.is_contiguous() and b.numel() in [1, N])
    ), "b should be a scalar or contiguous tensor with the same number of elements as x"

    if out is None:
        out = x
    else:
        assert (
            out.is_contiguous() and out.numel() == N
        ), "out should be contiguous with the same number of elements as x"

    if isinstance(a, float) or isinstance(a, int):
        IS_A_SCALAR = True
        IS_A_TENSOR = False
    elif isinstance(a, torch.Tensor) and a.is_contiguous():
        IS_A_TENSOR = True
        if a.numel() == 1:
            IS_A_SCALAR = True
        else:
            IS_A_SCALAR = False
    if isinstance(b, float) or isinstance(b, int):
        IS_B_SCALAR = True
        IS_B_TENSOR = False
    elif isinstance(b, torch.Tensor) and b.is_contiguous():
        IS_B_TENSOR = True
        if b.numel() == 1:
            IS_B_SCALAR = True
        else:
            IS_B_SCALAR = False

    BLOCK_SIZE_N = max(min(triton.next_power_of_2(N), 32), 1024)
    grid = (triton.cdiv(N, BLOCK_SIZE_N),)
    _fused_mul_add_kernel[grid](
        x,
        a,
        b,
        out,
        N,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        NEED_MASK=N % BLOCK_SIZE_N != 0,
        IS_A_SCALAR=IS_A_SCALAR,
        IS_B_SCALAR=IS_B_SCALAR,
        IS_A_TENSOR=IS_A_TENSOR,
        IS_B_TENSOR=IS_B_TENSOR,
        num_warps=4,
        waves_per_eu=0,
    )

    return out
