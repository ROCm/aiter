# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

# Ported from
# https://github.com/vllm-project/vllm/blob/main/vllm/models/deepseek_v41/common/ops/indexer_k_store.py

import triton
import triton.language as tl

from aiter.ops.triton._triton_kernels.quant.quant import _mxfp4_quant_op
from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr

_k_norm_rope_mxfp4_cache_repr = make_kernel_repr(
    "_k_norm_rope_mxfp4_cache_kernel",
    ["HEAD_SIZE", "ROPE_DIM", "COMPRESS_RATIO", "PRESHUFFLE", "N_PER_TILE"],
)


@triton.jit(repr=_k_norm_rope_mxfp4_cache_repr)
def _k_norm_rope_mxfp4_cache_kernel(
    k_ptr,  # bf16 [T, HEAD_SIZE]
    k_stride,
    positions_ptr,  # [T]
    norm_weight_ptr,  # [HEAD_SIZE]
    norm_eps,
    cos_sin_ptr,  # [max_pos, ROPE_DIM]: cos half, then sin half
    cos_sin_stride,
    cache_ptr,  # u8 pages: a page's values, then its e8m0 scales
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
    """One program per token."""
    token_idx = tl.program_id(0)

    slot = tl.load(slot_mapping_ptr + token_idx)
    if slot < 0:
        return
    position = tl.load(positions_ptr + token_idx)
    # With compression, only the last token of a group publishes its key.
    if (position + 1) % COMPRESS_RATIO != 0:
        return

    offs = tl.arange(0, HEAD_SIZE)

    # Rounded through bf16 to match the reference.
    k = tl.load(k_ptr + token_idx * k_stride + offs).to(tl.float32)
    w = tl.load(norm_weight_ptr + offs).to(tl.float32)
    variance = tl.sum(k * k, axis=0) / HEAD_SIZE
    k = (k * tl.rsqrt(variance + norm_eps) * w).to(tl.bfloat16).to(tl.float32)

    # A compressed key stands for its group's first token and rotates there.
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

    N_BLOCKS: tl.constexpr = HEAD_SIZE // 32
    TOKEN_BYTES: tl.constexpr = HEAD_SIZE // 2
    k_rot = tl.reshape(tl.join(new_even, new_odd), (1, HEAD_SIZE))
    packed, e8m0 = _mxfp4_quant_op(
        k_rot, HEAD_SIZE, 1, 32, SCALING_MODE=1, USE_ASM=True
    )
    packed = tl.reshape(packed, (TOKEN_BYTES,))
    e8m0 = tl.reshape(e8m0, (N_BLOCKS,))

    page = cache_ptr + (slot // page_size).to(tl.int64) * page_stride
    pos = slot % page_size
    byte = tl.arange(0, TOKEN_BYTES)
    scale_idx = tl.arange(0, N_BLOCKS)
    if PRESHUFFLE:
        # The pattern the wrapper's shuffle argument describes.
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
