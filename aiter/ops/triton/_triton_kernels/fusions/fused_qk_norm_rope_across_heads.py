# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import triton
import triton.language as tl

from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr
from aiter.ops.triton.utils.config_utils import (
    load_config_json,
    resolve_config_dir,
    select_leq_config,
)

_fused_qk_norm_rope_across_heads_repr = make_kernel_repr(
    "_fused_qk_norm_rope_across_heads_kernel",
    ["N", "HEAD_DIM", "BLOCK_M", "BLOCK_N", "USE_ROPE", "num_warps"],
)


def _get_config(N: int) -> dict:
    """Launch config for rows of N = heads * head_dim elements."""
    config_dir = resolve_config_dir("fusions", "FUSED_QK_NORM_ROPE_ACROSS_HEADS")
    configs = load_config_json(f"{config_dir}/DEFAULT.json", required=True)
    return select_leq_config(configs, N, prefix="N_LEQ_")


@triton.jit(repr=_fused_qk_norm_rope_across_heads_repr)
def _fused_qk_norm_rope_across_heads_kernel(
    q_ptr,
    k_ptr,
    wq_ptr,
    wk_ptr,
    cos_ptr,
    sin_ptr,
    qo_ptr,
    ko_ptr,
    S_q,
    S_k,
    M_q,
    M_k,
    num_blocks_q,
    stride_q_b,
    stride_q_s,
    stride_k_b,
    stride_k_s,
    stride_cos_s,
    eps,
    N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    USE_ROPE: tl.constexpr,
):
    """RMS norm over each row of N values, times the weight, then RoPE on each
    head. The first num_blocks_q programs do q, the others do k."""
    pid = tl.program_id(0)
    if pid < num_blocks_q:
        x_ptr = q_ptr
        w_ptr = wq_ptr
        out_ptr = qo_ptr
        S = S_q
        M = M_q
        stride_b = stride_q_b
        stride_s = stride_q_s
        block = pid
    else:
        x_ptr = k_ptr
        w_ptr = wk_ptr
        out_ptr = ko_ptr
        S = S_k
        M = M_k
        stride_b = stride_k_b
        stride_s = stride_k_s
        block = pid - num_blocks_q

    rows = (block * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)
    mask_m = rows < M
    s = rows % S
    in_offs = (rows // S) * stride_b + s * stride_s
    out_offs = rows * N
    out_dtype = out_ptr.dtype.element_ty

    # Pass 1: sum of squares over the row, in chunks of BLOCK_N.
    sumsq = tl.zeros((BLOCK_M,), dtype=tl.float32)
    for n0 in tl.static_range(0, N, BLOCK_N):
        cols = n0 + tl.arange(0, BLOCK_N)
        mask = mask_m[:, None] & (cols < N)[None, :]
        x = tl.load(x_ptr + in_offs[:, None] + cols[None, :], mask=mask, other=0.0)
        x = x.to(tl.float32)
        sumsq += tl.sum(x * x, axis=1)
    rstd = tl.rsqrt(sumsq / N + eps)

    # Pass 2: normalize, weight and RoPE each chunk, then store it.
    for n0 in tl.static_range(0, N, BLOCK_N):
        cols = n0 + tl.arange(0, BLOCK_N)
        mask_n = cols < N
        mask = mask_m[:, None] & mask_n[None, :]
        x = tl.load(x_ptr + in_offs[:, None] + cols[None, :], mask=mask, other=0.0)
        w = tl.load(w_ptr + cols, mask=mask_n, other=0.0).to(tl.float32)
        # Same as F.rms_norm: the weight is applied in fp32, the result rounds once.
        y = (x.to(tl.float32) * rstd[:, None] * w[None, :]).to(out_dtype)
        y = y.to(tl.float32)
        if USE_ROPE:
            # Pairs (2i, 2i + 1) of each head: cos from the even and sin from
            # the odd column of the table, as in the diffusers Wan RoPE.
            y_even, y_odd = tl.split(tl.reshape(y, (BLOCK_M, BLOCK_N // 2, 2)))
            col_even = n0 + 2 * tl.arange(0, BLOCK_N // 2)
            mask_half = mask_m[:, None] & (col_even < N)[None, :]
            tab = s[:, None] * stride_cos_s + (col_even % HEAD_DIM)[None, :]
            cos = tl.load(cos_ptr + tab, mask=mask_half, other=0.0).to(tl.float32)
            sin = tl.load(sin_ptr + tab + 1, mask=mask_half, other=0.0).to(tl.float32)
            out_even = y_even * cos - y_odd * sin
            out_odd = y_even * sin + y_odd * cos
            y = tl.reshape(tl.join(out_even, out_odd), (BLOCK_M, BLOCK_N))
        tl.store(
            out_ptr + out_offs[:, None] + cols[None, :], y.to(out_dtype), mask=mask
        )
