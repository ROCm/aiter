import triton
import triton.language as tl


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
