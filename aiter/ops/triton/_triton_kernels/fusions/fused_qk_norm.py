import torch
import triton
import triton.language as tl

from aiter import rmsnorm2d_fwd

@triton.jit
def _rmsmorm_op(row, weight, n_cols, epsilon):
    row_norm = row * row
    row_norm = tl.sum(row_norm, axis=-1)
    norm_factor = tl.math.rsqrt((row_norm / n_cols) + epsilon)

    rms_norm = row * norm_factor * weight
    return rms_norm

@triton.jit
def _fused_qk_rmsnorm_kernel(
    inp1_ptr,
    out1_ptr,
    out1_row_stride,
    out1_col_stride,
    weight1_ptr,
    inp2_ptr,
    out2_ptr,
    out2_row_stride,
    out2_col_stride,
    weight2_ptr,
    eps1,
    eps2,
    inp1_n_cols,
    inp2_n_cols,
    inp1_row_stride,
    inp2_row_stride,
    inp1_col_stride,
    inp2_col_stride,
    BLOCK_SIZE_N1: tl.constexpr,
    BLOCK_SIZE_N2: tl.constexpr,
):
    m_pid = tl.program_id(0)

    n_offs1 = tl.arange(0, BLOCK_SIZE_N1)
    mask1 = n_offs1 < inp1_n_cols
    inp1 = tl.load(
        inp1_ptr + m_pid * inp1_row_stride + n_offs1 * inp1_col_stride,
        mask=mask1,
        other=0.0,
    ).to(tl.float32)
    w1 = tl.load(weight1_ptr + n_offs1, mask=mask1, other=0.0).to(tl.float32)
    norm1 = _rmsmorm_op(inp1, w1, inp1_n_cols, eps1)
    tl.store(
        out1_ptr + m_pid * out1_row_stride + n_offs1 * out1_col_stride,
        norm1,
        mask=mask1,
    )

    n_offs2 = tl.arange(0, BLOCK_SIZE_N2)
    mask2 = n_offs2 < inp2_n_cols
    inp2 = tl.load(
        inp2_ptr + m_pid * inp2_row_stride + n_offs2 * inp2_col_stride,
        mask=mask2,
        other=0.0,
    ).to(tl.float32)
    w2 = tl.load(weight2_ptr + n_offs2, mask=mask2, other=0.0).to(tl.float32)
    norm2 = _rmsmorm_op(inp2, w2, inp2_n_cols, eps2)
    tl.store(
        out2_ptr + m_pid * out2_row_stride + n_offs2 * out2_col_stride,
        norm2,
        mask=mask2,
    )
