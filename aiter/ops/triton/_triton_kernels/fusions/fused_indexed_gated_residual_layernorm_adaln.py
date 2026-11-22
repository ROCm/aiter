# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import triton
import triton.language as tl

from aiter.ops.triton._triton_kernels.fusions.indexed_table import (
    _gather_indexed_rows,
    _load_block_indices,
)
from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr
from aiter.ops.triton.utils.config_utils import (
    load_config_json,
    resolve_config_dir,
    select_leq_config,
)

_fused_indexed_gated_residual_layernorm_adaln_repr = make_kernel_repr(
    "_fused_indexed_gated_residual_layernorm_adaln_kernel",
    [
        "BLOCK_M",
        "BLOCK_N",
        "SINGLE_TILE",
        "HAS_GATE",
        "ROUND_INTERMEDIATE",
        "HAS_WEIGHT",
        "HAS_BIAS",
    ],
)

# Used where an arch has no tuned file.
_DEFAULT_CONFIG = {"BLOCK_M": 1, "BLOCK_N": 2048, "num_warps": 8}


def _get_config(N: int) -> dict:
    """Launch config for rows of N elements: the first ``N_LEQ_<n>`` bucket that
    holds N, else ``any``."""
    config_dir = resolve_config_dir(
        "fusions", "FUSED_INDEXED_GATED_RESIDUAL_LAYERNORM_ADALN"
    )
    table = load_config_json(f"{config_dir}/DEFAULT.json", required=False)
    if table is None:
        return dict(_DEFAULT_CONFIG)
    return select_leq_config(table, N)


@triton.jit(repr=_fused_indexed_gated_residual_layernorm_adaln_repr)
def _fused_indexed_gated_residual_layernorm_adaln_kernel(
    out_ptr,
    h_ptr,
    residual_ptr,
    x_ptr,
    gate_ptr,
    shift_ptr,
    scale_ptr,
    w_ptr,
    b_ptr,
    idx_ptr,
    M,
    N,
    stride_residual_row,
    stride_x_row,
    stride_h_row,
    stride_out_row,
    stride_gate_row,
    stride_shift_row,
    stride_scale_row,
    eps,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SINGLE_TILE: tl.constexpr,
    HAS_GATE: tl.constexpr,
    ROUND_INTERMEDIATE: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    """h[m]   = residual[m] + gate[idx[m]] * x[m]          (gate = 1 without HAS_GATE)
    out[m] = layernorm(h[m]) * (1 + scale[idx[m]]) + shift[idx[m]]

    One program owns BLOCK_M rows. The first pass computes h tile by tile,
    stores it (rounded to the output dtype, as the unfused residual stream is)
    and accumulates the LayerNorm statistics of the stored values. The second
    pass normalises and modulates h: from registers when the row is a single
    tile (SINGLE_TILE), otherwise re-read from h, which this program has just
    written and is still in cache.
    """
    row_block = tl.program_id(0)
    rows = row_block * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = rows < M
    # M * N can exceed the int32 range for long sequences.
    rows64 = rows.to(tl.int64)

    idx, first, uniform = _load_block_indices(
        idx_ptr, row_block, rows, row_mask, BLOCK_M
    )

    # Mean and variance per column tile, merged with Chan's formula. The
    # statistics are of h - ref, the mean of the first tile, so a large mean
    # does not cost precision (as in fused_layernorm_indexed_adaln).
    cols = tl.arange(0, BLOCK_N)
    col_mask = cols < N
    mask = row_mask[:, None] & col_mask[None, :]
    residual = tl.load(
        residual_ptr + rows64[:, None] * stride_residual_row + cols[None, :],
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    x = tl.load(
        x_ptr + rows64[:, None] * stride_x_row + cols[None, :],
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    if HAS_GATE:
        gate = _gather_indexed_rows(
            gate_ptr,
            stride_gate_row,
            idx,
            first,
            uniform,
            cols,
            row_mask,
            col_mask,
            BLOCK_M,
            BLOCK_N,
        )
        x = gate * x
        if ROUND_INTERMEDIATE:
            # The eager `residual + gate[idx] * x` stores the product before
            # adding it. Reproduce that when bit-comparability matters more.
            x = x.to(h_ptr.dtype.element_ty).to(tl.float32)
    h = (residual + x).to(h_ptr.dtype.element_ty)
    tl.store(h_ptr + rows64[:, None] * stride_h_row + cols[None, :], h, mask=mask)
    h0 = h.to(tl.float32)
    count = tl.minimum(N, BLOCK_N).to(tl.float32)
    ref = tl.sum(tl.where(col_mask[None, :], h0, 0.0) / count, axis=1)
    d = tl.where(col_mask[None, :], h0 - ref[:, None], 0.0)
    mean = tl.sum(d, axis=1) / count
    centered = tl.where(col_mask[None, :], d - mean[:, None], 0.0)
    m2 = tl.sum(centered * centered, axis=1)

    if not SINGLE_TILE:
        for start in tl.range(BLOCK_N, N, BLOCK_N):
            cols = start + tl.arange(0, BLOCK_N)
            col_mask = cols < N
            mask = row_mask[:, None] & col_mask[None, :]
            residual = tl.load(
                residual_ptr + rows64[:, None] * stride_residual_row + cols[None, :],
                mask=mask,
                other=0.0,
            ).to(tl.float32)
            x = tl.load(
                x_ptr + rows64[:, None] * stride_x_row + cols[None, :],
                mask=mask,
                other=0.0,
            ).to(tl.float32)
            if HAS_GATE:
                gate = _gather_indexed_rows(
                    gate_ptr,
                    stride_gate_row,
                    idx,
                    first,
                    uniform,
                    cols,
                    row_mask,
                    col_mask,
                    BLOCK_M,
                    BLOCK_N,
                )
                x = gate * x
                if ROUND_INTERMEDIATE:
                    x = x.to(h_ptr.dtype.element_ty).to(tl.float32)
            h = (residual + x).to(h_ptr.dtype.element_ty)
            tl.store(
                h_ptr + rows64[:, None] * stride_h_row + cols[None, :], h, mask=mask
            )
            d = tl.where(col_mask[None, :], h.to(tl.float32) - ref[:, None], 0.0)
            n_tile = tl.minimum(N - start, BLOCK_N).to(tl.float32)
            tile_mean = tl.sum(d, axis=1) / n_tile
            centered = tl.where(col_mask[None, :], d - tile_mean[:, None], 0.0)
            tile_m2 = tl.sum(centered * centered, axis=1)
            total = count + n_tile
            delta = tile_mean - mean
            mean += delta * (n_tile / total)
            m2 += tile_m2 + delta * delta * (count * n_tile / total)
            count = total
    rstd = tl.rsqrt(m2 / N + eps)

    for start in tl.range(0, N, BLOCK_N):
        cols = start + tl.arange(0, BLOCK_N)
        col_mask = cols < N
        mask = row_mask[:, None] & col_mask[None, :]
        if SINGLE_TILE:
            hf = h0
        else:
            hf = tl.load(
                h_ptr + rows64[:, None] * stride_h_row + cols[None, :],
                mask=mask,
                other=0.0,
            ).to(tl.float32)
        normed = ((hf - ref[:, None]) - mean[:, None]) * rstd[:, None]
        if HAS_WEIGHT:
            w = tl.load(w_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
            normed = normed * w[None, :]
        if HAS_BIAS:
            b = tl.load(b_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
            normed = normed + b[None, :]
        scale = _gather_indexed_rows(
            scale_ptr,
            stride_scale_row,
            idx,
            first,
            uniform,
            cols,
            row_mask,
            col_mask,
            BLOCK_M,
            BLOCK_N,
        )
        shift = _gather_indexed_rows(
            shift_ptr,
            stride_shift_row,
            idx,
            first,
            uniform,
            cols,
            row_mask,
            col_mask,
            BLOCK_M,
            BLOCK_N,
        )
        out = normed * (1.0 + scale) + shift
        tl.store(
            out_ptr + rows64[:, None] * stride_out_row + cols[None, :],
            out.to(out_ptr.dtype.element_ty),
            mask=mask,
        )
