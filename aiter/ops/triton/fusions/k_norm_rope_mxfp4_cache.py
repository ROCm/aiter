# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import torch

from aiter.ops.triton._triton_kernels.fusions.k_norm_rope_mxfp4_cache import (
    _k_norm_rope_mxfp4_cache_kernel,
)
from aiter.ops.triton.utils.logger import AiterTritonLogger

_LOGGER = AiterTritonLogger()

_SCALE_GROUP = 32


def k_norm_rope_mxfp4_cache(
    k: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_eps: float,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    compress_ratio: int = 1,
    shuffle: tuple[int, int, int] | None = None,
) -> None:
    """RMSNorm -> GPT-J RoPE -> MXFP4 -> paged cache store of one key per token.

    MXFP4 is e2m1 values, two per byte (low nibble first), with one e8m0 scale
    per 32 values chosen as 2^ceil(log2(amax / 6)), so no value saturates.

    Args:
        k: [T, head_size] bf16, the key before its norm. Rows past
            slot_mapping.numel() are not read.
        positions: [T] token positions.
        cos_sin_cache: [max_pos, rope_dim] f32 or bf16, the cos half then the
            sin half, for GPT-J (interleaved) pairs on the last rope_dim dims.
        norm_weight: [head_size] RMSNorm weight.
        norm_eps: RMSNorm epsilon.
        kv_cache: u8 [num_pages, page_size, (1,) head_size // 2 + head_size // 32].
            A page holds its tokens' values, then their scales. Pages may sit
            any stride apart but must be contiguous inside.
        slot_mapping: [T] global slot per token (page * page_size + offset);
            a negative slot skips the token.
        compress_ratio: with c > 1 only tokens with (position + 1) % c == 0
            write a key, rotated at their group's first position.
        shuffle: None stores a page's tokens and scales in natural order.
            (n_per_tile, d_per_tile, scale_lanes) stores each run of
            n_per_tile tokens with its values as [d_per_tile-byte chunk,
            token, byte] and its e8m0 scales as [scale % scale_lanes, token,
            scale // scale_lanes]. paged_mxfp4_mqa_logits reads
            preshuffle_cache()'s order: cache_format()'s n_per_tile and
            d_per_tile, and 64 // n_per_tile scale lanes.
    """
    _LOGGER.info(
        "K_NORM_ROPE_MXFP4_CACHE: k=%s kv_cache=%s compress_ratio=%d shuffle=%s",
        tuple(k.shape),
        tuple(kv_cache.shape),
        compress_ratio,
        shuffle,
    )
    head_size = k.shape[-1]
    num_tokens = slot_mapping.numel()
    rope_dim = cos_sin_cache.shape[-1]
    if k.ndim != 2 or k.dtype != torch.bfloat16 or k.stride(1) != 1:
        raise ValueError(
            f"k must be [T, D] bf16 with unit last stride, got "
            f"{tuple(k.shape)} {k.dtype} {k.stride()}"
        )
    if num_tokens > k.shape[0] or num_tokens > positions.numel():
        raise ValueError(
            f"slot_mapping covers {num_tokens} tokens but k has "
            f"{k.shape[0]} and positions {positions.numel()}"
        )
    if head_size % _SCALE_GROUP or rope_dim % 2 or rope_dim > head_size:
        raise ValueError(
            f"head_size {head_size} must be a multiple of "
            f"{_SCALE_GROUP} and hold the {rope_dim}-dim rope"
        )
    if compress_ratio < 1:
        raise ValueError(f"compress_ratio must be >= 1, got {compress_ratio}")
    if kv_cache.dtype != torch.uint8:
        raise ValueError(f"kv_cache must be uint8, got {kv_cache.dtype}")
    if num_tokens == 0:
        return

    # view() so a cache that is not contiguous within a page raises instead of
    # being copied on every call.
    pages = kv_cache.view(kv_cache.shape[0], -1)
    page_size = pages.shape[1] // (head_size // 2 + head_size // _SCALE_GROUP)
    # The natural order takes any page size.
    n_per_tile = d_per_tile = scale_lanes = 1
    if shuffle is not None:
        n_per_tile, d_per_tile, scale_lanes = (int(v) for v in shuffle)
        if (
            min(n_per_tile, d_per_tile, scale_lanes) <= 0
            or page_size % n_per_tile
            or (head_size // 2) % d_per_tile
            or (head_size // _SCALE_GROUP) % scale_lanes
        ):
            raise ValueError(
                f"shuffle {tuple(shuffle)} does not tile {page_size}-token "
                f"pages of {head_size // 2} value bytes and "
                f"{head_size // _SCALE_GROUP} scales per token"
            )

    _k_norm_rope_mxfp4_cache_kernel[(num_tokens,)](
        k,
        k.stride(0),
        positions,
        norm_weight,
        norm_eps,
        cos_sin_cache,
        cos_sin_cache.stride(0),
        kv_cache,
        slot_mapping,
        page_size,
        pages.stride(0),
        HEAD_SIZE=head_size,
        ROPE_DIM=rope_dim,
        COMPRESS_RATIO=compress_ratio,
        PRESHUFFLE=shuffle is not None,
        N_PER_TILE=n_per_tile,
        D_PER_TILE=d_per_tile,
        SCALE_LANES=scale_lanes,
        num_warps=1,
    )
