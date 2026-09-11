# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import triton
import triton.language as tl

from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr

_engram_embedding_lookup_repr = make_kernel_repr(
    "_engram_embedding_lookup_kernel",
    ["BLOCK_H", "BLOCK_D", "SCALE_BLOCK"],
)

_engram_gate_apply_repr = make_kernel_repr(
    "_engram_gate_apply_kernel",
    ["BLOCK_C", "BLOCK_D", "HAS_MASK"],
)


@triton.jit(repr=_engram_embedding_lookup_repr)
def _engram_embedding_lookup_kernel(
    out_ptr,
    ids_ptr,
    table_ptr,
    scale_ptr,
    H,
    D,
    row_offset,
    num_rows,
    stride_ids_token,
    stride_out_token,
    stride_table_row,
    stride_scale_row,
    SCALE_BLOCK: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """out[t, h * D : (h + 1) * D] = dequant(table[ids[t, h] - row_offset])

    One program owns one token and BLOCK_H of its hash columns. Only the
    gathered rows are addressed, so table size never enters the cost. Ids
    outside the shard window store 0.0 and issue no table read at all.
    """
    tok = tl.program_id(0).to(tl.int64)
    heads = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
    head_mask = heads < H

    # ids are int64, so every row and byte offset below is int64 too: R and
    # R * D both exceed int32 on a production table.
    ids = tl.load(ids_ptr + tok * stride_ids_token + heads, mask=head_mask, other=-1)
    rows = ids - row_offset
    owned = head_mask & (rows >= 0) & (rows < num_rows)
    rows = tl.where(owned, rows, 0)

    NS: tl.constexpr = BLOCK_D // SCALE_BLOCK
    for d0 in tl.range(0, D, BLOCK_D):
        cols = d0 + tl.arange(0, BLOCK_D)
        col_mask = cols < D
        gather = owned[:, None] & col_mask[None, :]
        vals = tl.load(
            table_ptr + rows[:, None] * stride_table_row + cols[None, :],
            mask=gather,
            other=0.0,
        ).to(tl.float32)

        s_cols = d0 // SCALE_BLOCK + tl.arange(0, NS)
        raw = tl.load(
            scale_ptr + rows[:, None] * stride_scale_row + s_cols[None, :],
            mask=owned[:, None] & (s_cols * SCALE_BLOCK < D)[None, :],
            other=0,
        )
        # e8m0: the stored byte is the exponent, biased by 127.
        dequant = tl.exp2(raw.to(tl.float32) - 127.0)

        vals = tl.reshape(vals, (BLOCK_H, NS, SCALE_BLOCK)) * dequant[:, :, None]
        vals = tl.reshape(vals, (BLOCK_H, BLOCK_D))

        tl.store(
            out_ptr + tok * stride_out_token + heads[:, None] * D + cols[None, :],
            vals.to(out_ptr.dtype.element_ty),
            mask=head_mask[:, None] & col_mask[None, :],
        )


@triton.jit(repr=_engram_gate_apply_repr)
def _engram_gate_apply_kernel(
    out_ptr,
    h_ptr,
    key_ptr,
    value_ptr,
    weight_ptr,
    token_mask_ptr,
    C,
    dim,
    stride_h_token,
    stride_h_c,
    stride_key_token,
    stride_key_c,
    stride_value_token,
    stride_weight_c,
    stride_out_token,
    stride_out_c,
    eps,
    clamp_value,
    inv_sqrt_dim,
    HAS_MASK: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """out = h + sigmoid(signed_sqrt(normalized <h * weight, key>)) * value

    One program owns one token and all C mHC copies, so `value` -- shared by
    the copies -- is read once. Pass one reduces h.h, key.key and
    <h * weight, key> over `dim`; pass two re-reads h and injects.
    """
    tok = tl.program_id(0).to(tl.int64)
    cs = tl.arange(0, BLOCK_C)
    c_mask = cs < C

    hh = tl.zeros((BLOCK_C,), dtype=tl.float32)
    kk = tl.zeros((BLOCK_C,), dtype=tl.float32)
    hk = tl.zeros((BLOCK_C,), dtype=tl.float32)
    for d0 in tl.range(0, dim, BLOCK_D):
        cols = d0 + tl.arange(0, BLOCK_D)
        tile = c_mask[:, None] & (cols < dim)[None, :]
        h = tl.load(
            h_ptr + tok * stride_h_token + cs[:, None] * stride_h_c + cols[None, :],
            mask=tile,
            other=0.0,
        ).to(tl.float32)
        key = tl.load(
            key_ptr
            + tok * stride_key_token
            + cs[:, None] * stride_key_c
            + cols[None, :],
            mask=tile,
            other=0.0,
        ).to(tl.float32)
        weight = tl.load(
            weight_ptr + cs[:, None] * stride_weight_c + cols[None, :],
            mask=tile,
            other=0.0,
        ).to(tl.float32)
        hh += tl.sum(h * h, axis=1)
        kk += tl.sum(key * key, axis=1)
        hk += tl.sum(h * weight * key, axis=1)

    # normalized per (token, copy) over dim, not jointly over the copies
    rstd = tl.rsqrt(hh / dim + eps) * tl.rsqrt(kk / dim + eps)
    dot = hk * rstd * inv_sqrt_dim
    magnitude = tl.sqrt(tl.maximum(tl.abs(dot), clamp_value))
    negative = (dot.to(tl.uint32, bitcast=True) & 0x80000000) != 0
    gate = tl.sigmoid(tl.where(negative, -magnitude, magnitude))
    if HAS_MASK:
        gate = tl.where(tl.load(token_mask_ptr + tok).to(tl.int1), gate, 0.0)

    for d0 in tl.range(0, dim, BLOCK_D):
        cols = d0 + tl.arange(0, BLOCK_D)
        col_mask = cols < dim
        tile = c_mask[:, None] & col_mask[None, :]
        h = tl.load(
            h_ptr + tok * stride_h_token + cs[:, None] * stride_h_c + cols[None, :],
            mask=tile,
            other=0.0,
        ).to(tl.float32)
        value = tl.load(
            value_ptr + tok * stride_value_token + cols, mask=col_mask, other=0.0
        ).to(tl.float32)
        tl.store(
            out_ptr
            + tok * stride_out_token
            + cs[:, None] * stride_out_c
            + cols[None, :],
            (h + gate[:, None] * value[None, :]).to(out_ptr.dtype.element_ty),
            mask=tile,
        )
