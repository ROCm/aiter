# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Triton kernels for re-basing a packed V4 NoPE cache and folding RoPE."""

import triton
import triton.language as tl


@triton.jit
def _blockwise_amax_kernel(
    data_ptr,
    byte_ptr,
    amax_ptr,
    n_rows,
    stride_row,
    BLOCK_ROWS: tl.constexpr,
    NBLK: tl.constexpr,
    BLK: tl.constexpr,
    D_NOPE: tl.constexpr,
):
    """Per-D-block max of |dequantized value| over every row."""
    pid = tl.program_id(0).to(tl.int64)
    rows = pid * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS).to(tl.int64)
    row_ok = rows < n_rows

    for b in tl.range(0, NBLK):
        cols = b * BLK + tl.arange(0, BLK)
        off = rows[:, None] * stride_row + cols[None, :]
        q = tl.load(data_ptr + off, mask=row_ok[:, None], other=0.0).to(tl.float32)
        # The E8M0 bytes share storage with the fp8 payload, so they have to be
        # read through a uint8 view -- an fp8 load would decode them as floats.
        e = tl.load(byte_ptr + rows * stride_row + D_NOPE + b, mask=row_ok, other=0)
        scale = tl.exp2(e.to(tl.float32) - 127.0)
        tl.atomic_max(amax_ptr + b, tl.max(tl.abs(q * scale[:, None])))


@triton.jit
def _rebase_kernel(
    data_ptr,
    byte_ptr,
    exp_ptr,
    n_rows,
    stride_row,
    BLOCK_ROWS: tl.constexpr,
    NBLK: tl.constexpr,
    BLK: tl.constexpr,
    D_NOPE: tl.constexpr,
):
    """Requantize each row onto the shared per-D-block exponents."""
    pid = tl.program_id(0).to(tl.int64)
    rows = pid * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS).to(tl.int64)
    row_ok = rows < n_rows

    for b in tl.range(0, NBLK):
        cols = b * BLK + tl.arange(0, BLK)
        off = rows[:, None] * stride_row + cols[None, :]
        q = tl.load(data_ptr + off, mask=row_ok[:, None], other=0.0).to(tl.float32)
        e_old = tl.load(byte_ptr + rows * stride_row + D_NOPE + b, mask=row_ok, other=0)
        e_new = tl.load(exp_ptr + b)
        # Both scales are powers of two, so re-basing is an exponent difference.
        ratio = tl.exp2(e_old.to(tl.float32) - e_new.to(tl.float32))
        tl.store(
            data_ptr + off, (q * ratio[:, None]).to(tl.float8e4nv), mask=row_ok[:, None]
        )
        # The trailer is not rewritten here: warps of this program read a row's
        # old exponent on their own schedule, so writing it in the same pass
        # races. The caller writes the new exponents afterwards.


@triton.jit
def _fold_rope_kernel(
    data_ptr,
    rope_ptr,
    n_rows,
    stride_row,
    stride_rope,
    BLOCK_ROWS: tl.constexpr,
    D_NOPE: tl.constexpr,
    D_ROPE: tl.constexpr,
):
    """Quantize the bf16 RoPE tail to fp8 and park it in slots [448, 512)."""
    pid = tl.program_id(0).to(tl.int64)
    rows = pid * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS).to(tl.int64)
    row_ok = rows < n_rows
    cols = tl.arange(0, D_ROPE)
    r = tl.load(
        rope_ptr + rows[:, None] * stride_rope + cols[None, :],
        mask=row_ok[:, None],
        other=0.0,
    ).to(tl.float32)
    tl.store(
        data_ptr + rows[:, None] * stride_row + D_NOPE + cols[None, :],
        r.to(tl.float8e4nv),
        mask=row_ok[:, None],
    )


@triton.jit
def _to_dma_layout_kernel(
    src_ptr,
    src_byte_ptr,
    rope_ptr,
    dst_ptr,
    n_rows,
    stride_src,
    stride_rope,
    BLOCK_ROWS: tl.constexpr,
    NBLK: tl.constexpr,
    BLK: tl.constexpr,
    D_NOPE: tl.constexpr,
    D_PADDED: tl.constexpr,
    D_ROPE: tl.constexpr,
):
    """Re-base each row to unit scale and fold its fp8 RoPE, into a new tensor."""
    pid = tl.program_id(0).to(tl.int64)
    rows = pid * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS).to(tl.int64)
    row_ok = rows < n_rows
    for b in tl.range(0, NBLK):
        cols = b * BLK + tl.arange(0, BLK)
        q = tl.load(
            src_ptr + rows[:, None] * stride_src + cols[None, :],
            mask=row_ok[:, None],
            other=0.0,
        ).to(tl.float32)
        e = tl.load(
            src_byte_ptr + rows * stride_src + D_NOPE + b, mask=row_ok, other=127
        )
        v = q * tl.exp2(e.to(tl.float32) - 127.0)[:, None]
        tl.store(
            dst_ptr + rows[:, None] * D_PADDED + cols[None, :],
            tl.clamp(v, -448.0, 448.0).to(tl.float8e4nv),
            mask=row_ok[:, None],
        )
    rc = tl.arange(0, D_ROPE)
    r = tl.load(
        rope_ptr + rows[:, None] * stride_rope + rc[None, :],
        mask=row_ok[:, None],
        other=0.0,
    ).to(tl.float32)
    tl.store(
        dst_ptr + rows[:, None] * D_PADDED + D_NOPE + rc[None, :],
        tl.clamp(r, -448.0, 448.0).to(tl.float8e4nv),
        mask=row_ok[:, None],
    )
