# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import triton
import triton.language as tl


@triton.jit
def _fused_qk_norm_rope_cached_kernel(
    q_ptr,
    k_ptr,
    qw_ptr,
    kw_ptr,
    cache_ptr,
    H,
    D,
    ROT,
    stride_q_tok,
    stride_k_tok,
    stride_cache_tok,
    eps,
    BLOCK_H: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Per-head RMSNorm then partial NeoX RoPE, on q and k, in place.

    One program owns a token: a token's heads are contiguous, so the [H, D]
    tile is one coalesced run, and the token's cos/sin row is read once and
    reused across every head instead of being broadcast into a [T, H, D]
    temporary.
    """
    tok = tl.program_id(0)
    heads = tl.arange(0, BLOCK_H)
    dims = tl.arange(0, BLOCK_D)
    head_mask = heads < H
    dim_mask = dims < D
    mask = head_mask[:, None] & dim_mask[None, :]
    offs = heads[:, None] * D + dims[None, :]

    half = ROT // 2
    # The cache holds cos and sin for the front half of the rotated subspace;
    # both halves of that subspace reuse it, which is what "front part reuse"
    # means for a partial rotation.
    front = tl.arange(0, BLOCK_D) % half
    rotate = dims < ROT
    upper = dims >= half
    cos = tl.load(
        cache_ptr + tok * stride_cache_tok + front, mask=rotate, other=1.0
    ).to(tl.float32)
    sin = tl.load(
        cache_ptr + tok * stride_cache_tok + half + front, mask=rotate, other=0.0
    ).to(tl.float32)
    # Partner lane of the rotation: j -> j+half for the lower half, j-half for
    # the upper, with the upper half negated. That is rotate_half without the
    # concatenate.
    partner = tl.where(upper, dims - half, dims + half)
    # Lanes outside the rotated subspace have no partner; clamp them in range
    # and drop the result below rather than masking every use.
    partner = tl.where(rotate, partner, 0)
    sign = tl.where(upper, 1.0, -1.0)

    for which in tl.static_range(2):
        base = q_ptr + tok * stride_q_tok if which == 0 else k_ptr + tok * stride_k_tok
        w_ptr = qw_ptr if which == 0 else kw_ptr

        x = tl.load(base + offs, mask=mask, other=0.0).to(tl.float32)
        # RMSNorm is per head over D, reduced in fp32 like torch.nn.RMSNorm.
        rstd = 1.0 / tl.sqrt(tl.sum(x * x, axis=1) / D + eps)
        w = tl.load(w_ptr + dims, mask=dim_mask, other=0.0).to(tl.float32)
        normed = x * rstd[:, None] * w[None, :]

        # The partner lane, normalised the same way. Reading it from memory is
        # cheaper than permuting within the tile: it is the same 128-wide row,
        # already in cache from the load above.
        xp = tl.load(
            base + heads[:, None] * D + partner[None, :], mask=mask, other=0.0
        ).to(tl.float32)
        wp = tl.load(w_ptr + partner, mask=dim_mask, other=0.0).to(tl.float32)
        normed_p = xp * rstd[:, None] * wp[None, :]

        rotated = normed * cos[None, :] + sign[None, :] * normed_p * sin[None, :]
        out = tl.where(rotate[None, :], rotated, normed)
        tl.store(base + offs, out.to(base.dtype.element_ty), mask=mask)
