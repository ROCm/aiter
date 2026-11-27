# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

# Exponential-race sampling with a greedy override, tiled over the vocabulary.
# Stage 1 runs one program per (row, vocab tile). Stage 2 runs one program per
# row and combines the tiles.
import triton
import triton.language as tl

from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr

_tile_kernel_repr = make_kernel_repr("_fused_mixed_sample_tile_kernel", ["BLOCK_V"])
_pick_kernel_repr = make_kernel_repr("_fused_mixed_sample_pick_kernel", ["BLOCK_T"])


@triton.jit(repr=_tile_kernel_repr)
def _fused_mixed_sample_tile_kernel(
    input_ptr,
    stride_in,
    exp_ptr,
    stride_exp,
    temps_ptr,
    eps,
    greedy_eps,
    part_m_ptr,  # [n_tiles, rows] fp32
    part_key_ptr,  # [n_tiles, rows] fp32
    part_idx_ptr,  # [n_tiles, rows] int32
    rows,
    vocab,
    BLOCK_V: tl.constexpr,
):
    r = tl.program_id(0)
    t = tl.program_id(1)
    offs = t * BLOCK_V + tl.arange(0, BLOCK_V)
    mask = offs < vocab
    x = tl.load(
        input_ptr + r.to(tl.int64) * stride_in + offs, mask=mask, other=-float("inf")
    ).to(tl.float32)
    temp = tl.load(temps_ptr + r).to(tl.float32)
    o = t * rows + r
    # The raw-logit max of the tile. Both row kinds store it.
    m = tl.max(x, axis=0)
    tl.store(part_m_ptr + o, m)
    if temp <= greedy_eps:
        # Greedy row: the lowest id at the max.
        idx = tl.min(tl.where((x == m) & mask, offs, vocab), axis=0)
        tl.store(part_idx_ptr + o, idx)
    else:
        # Subtract the max before the division by T. Then a small T does not
        # lose the low bits of the logits. A tile that is -inf throughout has
        # m = -inf: shift it by 0, so that -inf - -inf does not give NaN.
        m_safe = tl.where(m == -float("inf"), 0.0, m)
        inv_t = 1.0 / tl.maximum(temp, 1e-5)
        e = (
            tl.load(exp_ptr + r.to(tl.int64) * stride_exp + offs, mask=mask, other=1.0)
            + eps
        )
        key = tl.where(mask, tl.exp((x - m_safe) * inv_t) / e, -1.0)
        best = tl.max(key, axis=0)
        idx = tl.min(tl.where((key == best) & mask, offs, vocab), axis=0)
        tl.store(part_key_ptr + o, best)
        tl.store(part_idx_ptr + o, idx)


@triton.jit(repr=_pick_kernel_repr)
def _fused_mixed_sample_pick_kernel(
    out_ptr,  # [rows] int32
    temps_ptr,
    greedy_eps,
    part_m_ptr,
    part_key_ptr,
    part_idx_ptr,
    rows,
    n_tiles,
    vocab,
    BLOCK_T: tl.constexpr,
):
    r = tl.program_id(0)
    ti = tl.arange(0, BLOCK_T)
    tm = ti < n_tiles
    o = ti * rows + r
    temp = tl.load(temps_ptr + r).to(tl.float32)
    m = tl.load(part_m_ptr + o, mask=tm, other=-float("inf"))
    idx = tl.load(part_idx_ptr + o, mask=tm, other=vocab)
    if temp <= greedy_eps:
        mm = tl.max(m, axis=0)
        tok = tl.min(tl.where((m == mm) & tm, idx, vocab), axis=0)
    else:
        k = tl.load(part_key_ptr + o, mask=tm, other=-1.0)
        mm = tl.max(m, axis=0)
        mm = tl.where(mm == -float("inf"), 0.0, mm)
        inv_t = 1.0 / tl.maximum(temp, 1e-5)
        # Rescale each tile key to the global max of the row.
        ks = tl.where(tm, k * tl.exp((m - mm) * inv_t), -1.0)
        best = tl.max(ks, axis=0)
        tok = tl.min(tl.where((ks == best) & tm, idx, vocab), axis=0)
    tl.store(out_ptr + r, tok.to(tl.int32))
