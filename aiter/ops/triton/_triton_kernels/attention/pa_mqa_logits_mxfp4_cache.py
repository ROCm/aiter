# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#
# Cache-prep kernels for the paged MXFP4 MQA-logits kernel, as DeepSeek-V4.1's
# indexer uses it: the key goes k_norm -> RoPE -> MXFP4 -> the paged cache in
# the order the logits kernel reads, the query RoPE -> MXFP4 plus its head
# weights. MXFP4 here is e2m1 values, two per byte (low nibble first), with one
# e8m0 scale per 32 values: 2^ceil(log2(amax / 6)).

import triton
import triton.language as tl

from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr


@triton.jit
def _fp32_to_e2m1(x):
    # Round to nearest even on {0, .5, 1, 1.5, 2, 3, 4, 6} and saturate, as
    # cvt.rn.satfinite does. The > / >= split is what sends each tie to the
    # even code.
    a = tl.abs(x)
    code = (
        (a > 0.25).to(tl.int32)
        + (a >= 0.75).to(tl.int32)
        + (a > 1.25).to(tl.int32)
        + (a >= 1.75).to(tl.int32)
        + (a > 2.5).to(tl.int32)
        + (a >= 3.5).to(tl.int32)
        + (a > 5.0).to(tl.int32)
    )
    return tl.where(x < 0, code | 8, code)


@triton.jit
def _fp32x2_to_e2m1x2(x_lo, x_hi):
    """One byte per pair: x_lo in the low nibble, x_hi in the high one."""
    return (_fp32_to_e2m1(x_lo) | (_fp32_to_e2m1(x_hi) << 4)).to(tl.uint8)


@triton.jit
def _quantize_mxfp4_pair(x_lo, x_hi):
    """One 32-value block given as its even (x_lo) and odd (x_hi) halves ->
    (16 packed bytes, its e8m0 scale byte)."""
    amax = tl.maximum(tl.max(tl.abs(x_lo)), tl.max(tl.abs(x_hi)))
    amax = tl.maximum(amax, 6.0 * (2**-126))
    log2_ratio = tl.math.ceil(tl.math.log2(amax * (1.0 / 6.0)))
    log2_ratio = tl.minimum(tl.maximum(log2_ratio, -127.0), 127.0)
    scale = tl.math.exp2(log2_ratio)
    e8m0 = (log2_ratio + 127.0).to(tl.uint8)
    inv_scale = 1.0 / scale
    return _fp32x2_to_e2m1x2(x_lo * inv_scale, x_hi * inv_scale), e8m0


_indexer_k_cache_repr = make_kernel_repr(
    "_indexer_k_norm_rope_mxfp4_cache_kernel",
    ["HEAD_SIZE", "COMPRESS_RATIO", "PRESHUFFLE", "N_PER_TILE"],
)


@triton.jit(repr=_indexer_k_cache_repr)
def _indexer_k_norm_rope_mxfp4_cache_kernel(
    k_ptr,  # bf16 [T, HEAD_SIZE]
    k_stride,
    positions_ptr,  # [T]
    norm_weight_ptr,  # [HEAD_SIZE]
    norm_eps,
    cos_sin_ptr,  # [max_pos, ROPE_DIM]: cos half, then sin half
    cos_sin_stride,
    cache_ptr,  # u8 pages: values [page, HEAD_SIZE // 2], then e8m0 [page, HEAD_SIZE // 32]
    slot_mapping_ptr,  # [T]; -1 skips the token
    page_size,
    page_stride: tl.int64,
    HEAD_SIZE: tl.constexpr,
    ROPE_DIM: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    PRESHUFFLE: tl.constexpr,
    N_PER_TILE: tl.constexpr,
    D_PER_TILE: tl.constexpr,
    SCALE_LANES: tl.constexpr,
):
    """One program per token: RMSNorm, GPT-J RoPE on the last ROPE_DIM dims at
    the token's group position, MXFP4, and the store."""
    token_idx = tl.program_id(0)

    slot = tl.load(slot_mapping_ptr + token_idx)
    if slot < 0:
        return
    position = tl.load(positions_ptr + token_idx)
    # With compression, only the last token of a group publishes its key.
    if (position + 1) % COMPRESS_RATIO != 0:
        return

    offs = tl.arange(0, HEAD_SIZE)

    # k_norm in fp32, rounded through bf16 like the reference.
    k = tl.load(k_ptr + token_idx * k_stride + offs).to(tl.float32)
    w = tl.load(norm_weight_ptr + offs).to(tl.float32)
    variance = tl.sum(k * k, axis=0) / HEAD_SIZE
    k = (k * tl.rsqrt(variance + norm_eps) * w).to(tl.bfloat16).to(tl.float32)

    # GPT-J RoPE in fp32 on the last ROPE_DIM dims. A compressed key stands for
    # its group's first token, so group j is rotated at position j * ratio.
    NUM_PAIRS: tl.constexpr = HEAD_SIZE // 2
    NOPE_PAIRS: tl.constexpr = (HEAD_SIZE - ROPE_DIM) // 2
    HALF_ROPE: tl.constexpr = ROPE_DIM // 2
    even, odd = tl.split(tl.reshape(k, (NUM_PAIRS, 2)))
    rope_pair = tl.arange(0, NUM_PAIRS) - NOPE_PAIRS
    is_rope = rope_pair >= 0
    cs_idx = tl.maximum(rope_pair, 0)
    group_pos = (position // COMPRESS_RATIO) * COMPRESS_RATIO
    cs_base = cos_sin_ptr + group_pos * cos_sin_stride
    cos = tl.load(cs_base + cs_idx, mask=is_rope, other=1.0)
    sin = tl.load(cs_base + HALF_ROPE + cs_idx, mask=is_rope, other=0.0)
    new_even = (even * cos - odd * sin).to(tl.bfloat16).to(tl.float32)
    new_odd = (odd * cos + even * sin).to(tl.bfloat16).to(tl.float32)

    # MXFP4: a 32-value block is 16 consecutive (even, odd) pairs, so tiling the
    # halves as (blocks, 16) lands one block per row.
    N_BLOCKS: tl.constexpr = HEAD_SIZE // 32
    TOKEN_BYTES: tl.constexpr = HEAD_SIZE // 2
    even_2d = tl.reshape(new_even, (N_BLOCKS, 16))
    odd_2d = tl.reshape(new_odd, (N_BLOCKS, 16))
    amax = tl.maximum(tl.max(tl.abs(even_2d), axis=1), tl.max(tl.abs(odd_2d), axis=1))
    amax = tl.maximum(amax, 6.0 * (2**-126))
    log2_ratio = tl.ceil(tl.log2(amax * (1.0 / 6.0)))
    log2_ratio = tl.minimum(tl.maximum(log2_ratio, -127.0), 127.0)
    inv_scale = tl.reshape(tl.exp2(-log2_ratio), (N_BLOCKS, 1))
    e8m0 = (log2_ratio + 127.0).to(tl.uint8)
    packed = tl.reshape(
        _fp32x2_to_e2m1x2(even_2d * inv_scale, odd_2d * inv_scale), (TOKEN_BYTES,)
    )

    page = cache_ptr + (slot // page_size).to(tl.int64) * page_stride
    pos = slot % page_size
    byte = tl.arange(0, TOKEN_BYTES)
    scale_idx = tl.arange(0, N_BLOCKS)
    if PRESHUFFLE:
        # The caller's pattern: each run of N_PER_TILE tokens stores its values
        # as [D_PER_TILE-byte chunk, token, byte] and its scales as
        # [scale % SCALE_LANES, token, scale // SCALE_LANES]. SCALE_LANES =
        # 64 // N_PER_TILE is preshuffle_scales' mode 1, N_BLOCKS its mode 0.
        S_HI: tl.constexpr = N_BLOCKS // SCALE_LANES
        tl.static_assert(SCALE_LANES * S_HI == N_BLOCKS)
        tl.static_assert(TOKEN_BYTES % D_PER_TILE == 0)
        group = pos // N_PER_TILE
        lane = pos % N_PER_TILE
        value_off = (
            group * (N_PER_TILE * TOKEN_BYTES)
            + byte // D_PER_TILE * (N_PER_TILE * D_PER_TILE)
            + lane * D_PER_TILE
            + byte % D_PER_TILE
        )
        scale_off = (
            group * (N_PER_TILE * N_BLOCKS)
            + scale_idx % SCALE_LANES * (N_PER_TILE * S_HI)
            + lane * S_HI
            + scale_idx // SCALE_LANES
        )
    else:
        value_off = pos * TOKEN_BYTES + byte
        scale_off = pos * N_BLOCKS + scale_idx
    tl.store(page + value_off, packed)
    tl.store(page + page_size * TOKEN_BYTES + scale_off, e8m0)


_indexer_q_quant_repr = make_kernel_repr(
    "_indexer_q_rope_mxfp4_quant_kernel",
    ["HEAD_SIZE", "HALF_ROPE"],
)


@triton.jit(repr=_indexer_q_quant_repr)
def _indexer_q_rope_mxfp4_quant_kernel(
    positions_ptr,  # [T]
    q_ptr,  # [T, H, HEAD_SIZE] bf16/fp16
    q_stride0,
    q_stride1,
    cos_sin_ptr,  # [max_pos, 2 * HALF_ROPE]: cos half, then sin half
    cos_sin_stride,
    q_packed_ptr,  # u8 [T, H, HEAD_SIZE // 2]
    q_packed_stride0,
    q_packed_stride1,
    q_scale_ptr,  # u8 e8m0 [T, H, HEAD_SIZE // 32]
    q_scale_stride0,
    q_scale_stride1,
    weights_ptr,  # [T, H]
    weights_stride,
    softmax_scale,
    head_scale,
    weights_out_ptr,  # f32 [T, H]
    weights_out_stride,
    HEAD_SIZE: tl.constexpr,
    HALF_ROPE: tl.constexpr,
):
    """One program per (token, head): GPT-J RoPE on the last 2 * HALF_ROPE dims,
    MXFP4 per 32 values, and weights * softmax_scale * head_scale. Unlike the
    fp8 query, no scale folds into the weights: the per-block scales travel
    with the values."""
    ROPE_DIM: tl.constexpr = 2 * HALF_ROPE
    NOPE_DIM: tl.constexpr = HEAD_SIZE - ROPE_DIM
    NOPE_BLOCKS: tl.constexpr = NOPE_DIM // 32
    ROPE_BLOCKS: tl.constexpr = ROPE_DIM // 32
    tl.static_assert(NOPE_DIM >= 0)
    tl.static_assert(NOPE_DIM % 32 == 0)
    tl.static_assert(ROPE_DIM % 32 == 0)

    tok = tl.program_id(0)
    head = tl.program_id(1)
    pos = tl.load(positions_ptr + tok)
    q_base = q_ptr + tok * q_stride0 + head * q_stride1
    out_base = q_packed_ptr + tok * q_packed_stride0 + head * q_packed_stride1
    scale_base = q_scale_ptr + tok * q_scale_stride0 + head * q_scale_stride1
    half = tl.arange(0, 16)

    for b in tl.static_range(NOPE_BLOCKS):
        x_lo = tl.load(q_base + b * 32 + half * 2).to(tl.float32)
        x_hi = tl.load(q_base + b * 32 + half * 2 + 1).to(tl.float32)
        packed, e8m0 = _quantize_mxfp4_pair(x_lo, x_hi)
        tl.store(out_base + b * 16 + half, packed)
        tl.store(scale_base + b, e8m0)

    rope_base = q_base + NOPE_DIM
    for b in tl.static_range(ROPE_BLOCKS):
        pair = b * 16 + half
        cos = tl.load(cos_sin_ptr + pos * cos_sin_stride + pair).to(tl.float32)
        sin = tl.load(cos_sin_ptr + pos * cos_sin_stride + pair + HALF_ROPE).to(
            tl.float32
        )
        x_even = tl.load(rope_base + pair * 2).to(tl.float32)
        x_odd = tl.load(rope_base + pair * 2 + 1).to(tl.float32)
        # Rounded through bf16 like the reference.
        r_even = (x_even * cos - x_odd * sin).to(tl.bfloat16).to(tl.float32)
        r_odd = (x_odd * cos + x_even * sin).to(tl.bfloat16).to(tl.float32)
        packed, e8m0 = _quantize_mxfp4_pair(r_even, r_odd)
        tl.store(out_base + (NOPE_DIM + b * 32) // 2 + half, packed)
        tl.store(scale_base + NOPE_BLOCKS + b, e8m0)

    w = tl.load(weights_ptr + tok * weights_stride + head).to(tl.float32)
    w *= softmax_scale
    w *= head_scale
    tl.store(weights_out_ptr + tok * weights_out_stride + head, w)
