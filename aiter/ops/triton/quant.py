import triton
import triton.language as tl
import torch


@triton.jit
def _static_per_tensor_fp8_quant_kernel(
    qx_ptr: torch.Tensor,
    x_in_ptr: torch.Tensor,
    scale_in_ptr: torch.Tensor,
    cols: int,
    x_in_stride_r: int,
    NUM_COL_POW2: tl.constexpr,
):
    """
    #TODO: Add Doc
    """

    pid = tl.program_id(axis=0)
    tl.assume(pid > 0)
    tl.assume(x_in_stride_r > 0)

    offs = pid * x_in_stride_r + tl.arange(0, NUM_COL_POW2)
    mask = tl.arange(0, NUM_COL_POW2) < cols
    x = tl.load(x_in_ptr + offs, mask=mask, cache_modifier=".cg")

    scale = tl.load(scale_in_ptr)
    scale_recip = 1 / scale

    qx = (x * scale_recip).to(qx_ptr.dtype.element_ty)

    tl.store(qx_ptr + offs, qx, mask=mask)


def static_per_tensor_fp8_quant(
    qx: torch.Tensor, x_in: torch.Tensor, scale_in: torch.Tensor
):
    """
    #TODO: Add Doc
    """
    assert scale_in.numel() == 1  # only single scale value
    rows = x_in.shape[0]
    cols = x_in.shape[1]
    NUM_COL_POW2 = triton.next_power_of_2(cols)
    grid = lambda meta: (rows,)
    _static_per_tensor_fp8_quant_kernel[grid](
        qx, x_in, scale_in, cols, x_in.stride(0), NUM_COL_POW2=NUM_COL_POW2
    )

    return qx


@triton.jit
def _dynamic_per_tensor_fp8_quant_kernel(
    x_in_ptr: torch.Tensor,
    scale_out_ptr: torch.Tensor,
    cols: int,
    x_in_stride_r: int,
    NUM_COL_POW2: tl.constexpr,
    FP8_MAX: tl.constexpr,
):
    """
    #TODO: Add Doc
    """

    pid = tl.program_id(axis=0)
    tl.assume(pid > 0)
    tl.assume(x_in_stride_r > 0)

    offs = pid * x_in_stride_r + tl.arange(0, NUM_COL_POW2)
    mask = tl.arange(0, NUM_COL_POW2) < cols
    x = tl.load(x_in_ptr + offs, mask=mask, cache_modifier=".cg")

    m = tl.max(tl.abs(x))
    tl.atomic_max(scale_out_ptr, m / FP8_MAX, sem="relaxed")


def dynamic_per_tensor_fp8_quant(
    qx: torch.Tensor, x_in: torch.Tensor, scale_out: torch.Tensor
):
    """
    #TODO: Add Doc
    """
    rows = x_in.shape[0]
    cols = x_in.shape[1]
    NUM_COL_POW2 = triton.next_power_of_2(cols)
    grid = lambda meta: (rows,)
    _dynamic_per_tensor_fp8_quant_kernel[grid](
        x_in,
        scale_out,
        cols,
        x_in.stride(0),
        NUM_COL_POW2=NUM_COL_POW2,
        FP8_MAX=torch.finfo(qx.dtype).max,
    )

    _static_per_tensor_fp8_quant_kernel[grid](
        qx, x_in, scale_out, cols, x_in.stride(0), NUM_COL_POW2=NUM_COL_POW2
    )

    return qx, scale_out


@triton.jit
def _dynamic_per_token_fp8_quant_kernel(
    qx_ptr: torch.Tensor,
    scale_out_ptr: torch.Tensor,
    x_in_ptr: torch.Tensor,
    cols: int,
    x_in_stride_r: int,
    NUM_COL_POW2: tl.constexpr,
    FP8_MAX: tl.constexpr,
):
    """
    #TODO: Add Doc
    """

    pid = tl.program_id(axis=0)
    tl.assume(pid > 0)
    tl.assume(x_in_stride_r > 0)

    offs = pid * x_in_stride_r + tl.arange(0, NUM_COL_POW2)
    mask = tl.arange(0, NUM_COL_POW2) < cols
    x = tl.load(x_in_ptr + offs, mask=mask, cache_modifier=".cg")

    m = tl.max(tl.abs(x), axis=-1)
    scale_out = m / FP8_MAX
    scale_recip = 1 / scale_out

    qx = x * scale_recip
    qx = qx.to(qx_ptr.dtype.element_ty)

    scale_offs = pid
    tl.store(scale_out_ptr + scale_offs, scale_out)

    tl.store(qx_ptr + offs, qx, mask=mask, cache_modifier=".cs")


def dynamic_per_token_fp8_quant(
    qx: torch.Tensor,
    x_in: torch.Tensor,
    scale_out: torch.Tensor,
    quant_dtype=torch.float8_e4m3fnuz,
    dtypeMax: torch.Tensor = torch.finfo(torch.float8_e4m3fnuz).max,
):
    """
    #TODO: Add doc
    """
    rows = x_in.shape[0]
    cols = x_in.shape[1]
    NUM_COL_POW2 = triton.next_power_of_2(cols)
    grid = lambda meta: (rows,)
    _dynamic_per_token_fp8_quant_kernel[grid](
        qx,
        scale_out,
        x_in,
        cols,
        x_in.stride(0),
        NUM_COL_POW2=NUM_COL_POW2,
        FP8_MAX=dtypeMax,
    )

    return qx, scale_out
