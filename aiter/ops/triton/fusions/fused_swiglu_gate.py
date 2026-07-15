# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

from typing import Optional

import torch
import triton

from aiter.ops.triton._triton_kernels.fusions.fused_swiglu_gate import (
    _fused_swiglu_gate_kernel,
)
from aiter.ops.triton.utils.logger import AiterTritonLogger

_LOGGER = AiterTritonLogger()

# MiniMax-M3 / GPT-OSS defaults
_DEFAULT_SWIGLU_ALPHA = 1.702
_DEFAULT_SWIGLU_LIMIT = 7.0


def _pick_block_n(d: int, n_rows: int) -> int:
    """Tile size along the reduced last dim (cap 1024); at least 32 for vectorization."""
    n = max(d, 1)
    if n == 512:
        return 512 if n_rows > 4096 else 256
    if n == 384:
        return 256 if n_rows <= 128 else 128
    upper = min(n, 1024)
    p = 1
    while p * 2 <= upper:
        p *= 2
    return max(32, p)


def _pick_block_m(n_rows: int, block_n: int, d: int) -> int:
    if n_rows <= 64:
        return min(32, max(4, triton.next_power_of_2(n_rows)))
    if d == 384 and n_rows > 128:
        return 32 if n_rows > 8192 else 8
    if d == 512 and n_rows > 4096:
        return 8
    if d == 512 and 128 < n_rows <= 4096:
        return 8
    if block_n >= 1024:
        return 8
    if block_n >= 512:
        return 8
    return 16


def _pick_num_warps(n_rows: int, block_m: int, block_n: int) -> int:
    if n_rows <= 128 and block_m >= 16 and block_n >= 128:
        return 8
    return 2


def fused_swiglu_gate(
    inp: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    swiglu_alpha: float = _DEFAULT_SWIGLU_ALPHA,
    swiglu_limit: float = _DEFAULT_SWIGLU_LIMIT,
    add_residual: bool = True,
):
    """
    Fused MiniMax / GPT-OSS gate activation on separated gate|up GEMM output.

    Reference numerics (``swiglu_no_interleaved_with_alpha_and_limit``)::

        gate, up = inp.chunk(2, dim=-1)
        gate = gate.clamp(max=limit)
        up = up.clamp(min=-limit, max=limit)
        out = gate * sigmoid(gate * alpha) * (up + 1)

    Args:
        inp: ``[..., 2 * N]`` contiguous tensor; first ``N`` columns are gate,
            second ``N`` are up.
        out: optional pre-allocated output with shape ``[..., N]``.
        swiglu_alpha: sigmoid scale on the gate half (default 1.702).
        swiglu_limit: clamp limit; pass ``<= 0`` to skip clamping.
        add_residual: when True, multiply by ``(up + 1)``; else ``up`` only.
    """
    assert inp.is_cuda, "fused_swiglu_gate requires a CUDA tensor"
    assert inp.is_contiguous(), "inp must be contiguous"
    last = inp.size(-1)
    assert last % 2 == 0, "last dimension must be even (2 * N)"
    n_cols = last // 2
    leading = inp.shape[:-1]
    n_rows = inp.numel() // (2 * n_cols)

    _LOGGER.info(
        f"fused_swiglu_gate: inp={tuple(inp.shape)} n_cols={n_cols} rows={n_rows} "
        f"alpha={swiglu_alpha} limit={swiglu_limit}"
    )

    if n_rows == 0:
        if out is None:
            return torch.empty(*leading, n_cols, dtype=inp.dtype, device=inp.device)
        return out

    if out is None:
        out = torch.empty(*leading, n_cols, dtype=inp.dtype, device=inp.device)
    else:
        assert out.is_contiguous(), "out must be contiguous"
        assert out.shape == (*leading, n_cols)
        assert out.dtype == inp.dtype and out.device == inp.device

    row_stride_in = 2 * n_cols
    col_stride_in = 1
    row_stride_out = n_cols
    col_stride_out = 1

    block_n = _pick_block_n(n_cols, n_rows)
    block_m = _pick_block_m(n_rows, block_n, n_cols)
    grid_m = triton.cdiv(n_rows, block_m)
    grid_n = triton.cdiv(n_cols, block_n)
    num_warps = _pick_num_warps(n_rows, block_m, block_n)

    HAVE_SWIGLU_CLAMP = swiglu_limit > 0

    _fused_swiglu_gate_kernel[(grid_m, grid_n)](
        inp,
        out,
        n_rows,
        n_cols,
        row_stride_in,
        col_stride_in,
        row_stride_out,
        col_stride_out,
        swiglu_alpha,
        swiglu_limit,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        HAVE_SWIGLU_CLAMP=HAVE_SWIGLU_CLAMP,
        ADD_RESIDUAL=add_residual,
        num_warps=num_warps,
        waves_per_eu=0,
    )
    return out
