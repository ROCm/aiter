import torch
import triton
import triton.language as tl
from typing import Optional


@triton.jit
def _fused_mul_add_kernel(
    x_ptr,
    a_ptr,
    b_ptr,
    out_ptr,
    N,
    BLOCK_SIZE_N: tl.constexpr,
    NEED_MASK: tl.constexpr,
    IS_A_SCALAR: tl.constexpr,
    IS_B_SCALAR: tl.constexpr,
    IS_A_TENSOR: tl.constexpr,
    IS_B_TENSOR: tl.constexpr,
):
    pid = tl.program_id(0)

    x_offs = pid * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    x_mask = None
    if NEED_MASK:
        x_mask = x_offs < N

    x = tl.load(x_ptr + x_offs, mask=x_mask).to(tl.float32)

    if IS_A_SCALAR and IS_A_TENSOR:
        a = tl.load(a_ptr)
    elif IS_A_SCALAR:
        a = a_ptr
    else:
        a = tl.load(a_ptr + x_offs, mask=x_mask)
    a = a.to(tl.float32)

    if IS_B_SCALAR and IS_B_TENSOR:
        b = tl.load(b_ptr)
    elif IS_B_SCALAR:
        b = b_ptr
    else:
        b = tl.load(b_ptr + x_offs, mask=x_mask)
    b = b.to(tl.float32)

    out = a * x + b
    out = out.to(out_ptr.dtype.element_ty)
    out = tl.store(out_ptr + x_offs, out, mask=x_mask)


def fused_mul_add(
    x: torch.Tensor,
    a: torch.Tensor | float | int,
    b: torch.Tensor | float | int,
    out: Optional[torch.Tensor] = None,
):
    """
    Computes elementwise multiplicated and addtion: out = x * a + b

    x must be a torch.Tensor, but with arbitrary shape,
    a can be float, int, or torch.Tensor with shape (1, ) or the same shape as x
    b can be float, int, or torch.Tensor with shape (1, ) or the same shape as x

    if out is None, the kernel will perform inplace computation on x

    """

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
