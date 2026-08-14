# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Fused clamp + sigmoid(alpha*gate) + mul + (up+1) on separated gate|up rows.

Each program tile loads gate and up from ``[M, 2 * N]`` with the same layout as
``torch.chunk(2, dim=-1)`` on gate-up GEMM output (MiniMax-M3 / GPT-OSS swigluoai).
"""

import triton
import triton.language as tl

from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr


def _fallback_config(n_rows: int, n_cols: int) -> dict:
    """Heuristic config used when no tuned JSON exists for this arch.

    Mirrors the tuned tables: small BLOCK_M at low concurrency so the grid
    spreads across CUs, growing BLOCK_M as the op becomes bandwidth-bound.
    """
    block_n = min(128, triton.next_power_of_2(max(n_cols, 1)))
    if n_rows <= 16:
        block_m, num_warps = 8, 2
    elif n_rows <= 512:
        block_m, num_warps = 8, 4
    elif n_rows <= 8192:
        block_m, num_warps = 16, 4
    else:
        block_m, num_warps = 32, 4
    return {
        "BLOCK_M": block_m,
        "BLOCK_N": block_n,
        "num_warps": num_warps,
        "waves_per_eu": 0,
    }


def _get_config(n_rows: int, n_cols: int) -> dict:
    """Select a launch config, mirroring the GEMM config-selection logic.

    Uses ``get_gemm_config`` (per-arch JSON, ``M_LEQ_x``/``M_GEQ_x`` buckets)
    with a per-``n_cols`` specialized file ``{arch}-FUSED-SWIGLU-GATE-D={n_cols}
    .json`` and a shared ``{arch}-FUSED-SWIGLU-GATE.json`` default. Falls back to
    a heuristic on arches with no tuned tables.
    """
    from aiter.ops.triton.utils.gemm_config_utils import get_gemm_config

    try:
        config, _ = get_gemm_config(
            "FUSED-SWIGLU-GATE", M=n_rows, specialized_filename=f"D={n_cols}"
        )
    except (AssertionError, KeyError):
        config = _fallback_config(n_rows, n_cols)
    return config


_fused_swiglu_gate_repr = make_kernel_repr(
    "_fused_swiglu_gate_kernel",
    [
        "BLOCK_M",
        "BLOCK_N",
        "HAVE_SWIGLU_CLAMP",
        "ADD_RESIDUAL",
    ],
)


@triton.jit(repr=_fused_swiglu_gate_repr)
def _fused_swiglu_gate_kernel(
    inp_ptr,
    out_ptr,
    n_rows,
    n_cols,
    row_stride_in,
    col_stride_in,
    row_stride_out,
    col_stride_out,
    neg_log2e_alpha,
    swiglu_limit,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HAVE_SWIGLU_CLAMP: tl.constexpr,
    ADD_RESIDUAL: tl.constexpr,
):
    m_pid = tl.program_id(0)
    n_pid = tl.program_id(1)
    m_offs = tl.arange(0, BLOCK_M)
    n_offs = tl.arange(0, BLOCK_N)
    row_idx = m_pid * BLOCK_M + m_offs
    col_idx = n_pid * BLOCK_N + n_offs

    row_in = row_idx * row_stride_in
    row_out = row_idx * row_stride_out

    gate_ptrs = inp_ptr + row_in[:, None] + col_idx[None, :] * col_stride_in
    up_ptrs = inp_ptr + row_in[:, None] + (n_cols + col_idx)[None, :] * col_stride_in
    out_ptrs = out_ptr + row_out[:, None] + col_idx[None, :] * col_stride_out

    mask = (row_idx < n_rows)[:, None] & (col_idx < n_cols)[None, :]
    # Default (cached) loads outperform ".cg" here: this is a pure streaming op
    # with no data reuse, and bypassing L1 via ".cg" adds coherence traffic that
    # regresses large-M bandwidth. The output is write-once, so ".wt"
    # (write-through) avoids polluting the cache with lines nobody reads back.
    gate = tl.load(gate_ptrs, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(up_ptrs, mask=mask, other=0.0).to(tl.float32)

    if HAVE_SWIGLU_CLAMP:
        gate = tl.minimum(gate, swiglu_limit)
        up = tl.clamp(up, -swiglu_limit, swiglu_limit)

    # sigmoid(alpha*g) = 1 / (1 + exp2(-log2(e) * alpha * g)); the -log2(e)*alpha
    # scalar is folded into neg_log2e_alpha by the caller.
    s = gate / (1 + tl.exp2(neg_log2e_alpha * gate))
    if ADD_RESIDUAL:
        out = tl.fma(s, up, s)
    else:
        out = s * up

    tl.store(
        out_ptrs, out.to(out_ptr.dtype.element_ty), mask=mask, cache_modifier=".wt"
    )
