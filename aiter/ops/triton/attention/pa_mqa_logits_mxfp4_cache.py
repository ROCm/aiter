# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#
# Launchers for the paged MXFP4 MQA-logits kernel's cache-prep ops: write the
# indexer key into the paged cache, in natural order or in a shuffle pattern
# the caller passes, and quantize the query to the MXFP4 it takes.

import torch

from aiter.ops.triton._triton_kernels.attention.pa_mqa_logits_mxfp4_cache import (
    _indexer_k_norm_rope_mxfp4_cache_kernel,
    _indexer_q_rope_mxfp4_quant_kernel,
)
from aiter.ops.triton.attention.pa_mqa_logits_mxfp4 import (
    SCALE_GROUP,
    _split_cache,
)


def indexer_k_norm_rope_mxfp4_cache(
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
    """RMSNorm -> GPT-J RoPE -> MXFP4 -> paged store of indexer keys.

    Args:
        k: [T, head_size] bf16, the key before its norm. Rows past
            slot_mapping.numel() are not read.
        positions: [T] token positions.
        cos_sin_cache: [max_pos, rope_dim] f32 or bf16, the cos half then the
            sin half, for GPT-J (interleaved) pairs on the last rope_dim dims.
        norm_weight: [head_size] RMSNorm weight.
        norm_eps: RMSNorm epsilon.
        kv_cache: u8 [num_pages, page_size, (1,) head_size // 2 + head_size // 32]
            as pack_cache lays it out. Pages may sit any stride apart but must
            be contiguous inside.
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
    if head_size % SCALE_GROUP or rope_dim % 2 or rope_dim > head_size:
        raise ValueError(
            f"head_size {head_size} must be a multiple of "
            f"{SCALE_GROUP} and hold the {rope_dim}-dim rope"
        )
    if compress_ratio < 1:
        raise ValueError(f"compress_ratio must be >= 1, got {compress_ratio}")
    if kv_cache.dtype != torch.uint8:
        raise ValueError(f"kv_cache must be uint8, got {kv_cache.dtype}")
    if num_tokens == 0:
        return

    _, _, page_size, page_stride = _split_cache(kv_cache, head_size)
    # The natural order takes any page size.
    n_per_tile = d_per_tile = scale_lanes = 1
    if shuffle is not None:
        n_per_tile, d_per_tile, scale_lanes = (int(v) for v in shuffle)
        if (
            min(n_per_tile, d_per_tile, scale_lanes) <= 0
            or page_size % n_per_tile
            or (head_size // 2) % d_per_tile
            or (head_size // SCALE_GROUP) % scale_lanes
        ):
            raise ValueError(
                f"shuffle {tuple(shuffle)} does not tile {page_size}-token "
                f"pages of {head_size // 2} value bytes and "
                f"{head_size // SCALE_GROUP} scales per token"
            )

    _indexer_k_norm_rope_mxfp4_cache_kernel[(num_tokens,)](
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
        page_stride,
        HEAD_SIZE=head_size,
        ROPE_DIM=rope_dim,
        COMPRESS_RATIO=compress_ratio,
        PRESHUFFLE=shuffle is not None,
        N_PER_TILE=n_per_tile,
        D_PER_TILE=d_per_tile,
        SCALE_LANES=scale_lanes,
        num_warps=1,
    )


def indexer_q_rope_mxfp4_quant(
    positions: torch.Tensor,
    q: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    weights: torch.Tensor,
    softmax_scale: float,
    head_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """GPT-J RoPE on the last rope_dim dims -> MXFP4 query, plus its weights.

    Args:
        positions: [T] token positions.
        q: [T, H, head_size] bf16/fp16 indexer query (strided is fine, the
            last dim must be unit-stride).
        cos_sin_cache: [max_pos, rope_dim], the cos half then the sin half.
        weights: [T, H] per-head weights.
        softmax_scale, head_scale: multiplied into the weights.

    Returns:
        (q_packed u8 [T, H, head_size // 2], q_scale u8 e8m0
        [T, H, head_size // 32], weights f32 [T, H]). No query scale folds
        into the weights: the per-block scales travel with the values.
    """
    if q.ndim != 3 or q.stride(2) != 1:
        raise ValueError(
            f"q must be [T, H, D] with unit last stride, got "
            f"{tuple(q.shape)} {q.stride()}"
        )
    num_tokens, num_heads, head_size = q.shape
    rope_dim = cos_sin_cache.shape[-1]
    if head_size % SCALE_GROUP or rope_dim % SCALE_GROUP or rope_dim > head_size:
        raise ValueError(
            f"head_size {head_size} and rope_dim {rope_dim} must be "
            f"multiples of {SCALE_GROUP}, rope_dim <= head_size"
        )
    if (
        positions.numel() != num_tokens
        or tuple(weights.shape) != (num_tokens, num_heads)
        or weights.stride(1) != 1
    ):
        raise ValueError(
            f"positions {tuple(positions.shape)} and weights "
            f"{tuple(weights.shape)} (unit head stride) must cover "
            f"q's [T={num_tokens}, H={num_heads}]"
        )
    q_packed = torch.empty(
        (num_tokens, num_heads, head_size // 2), dtype=torch.uint8, device=q.device
    )
    q_scale = torch.empty(
        (num_tokens, num_heads, head_size // SCALE_GROUP),
        dtype=torch.uint8,
        device=q.device,
    )
    weights_out = torch.empty(
        (num_tokens, num_heads), dtype=torch.float32, device=q.device
    )
    if num_tokens == 0:
        return q_packed, q_scale, weights_out
    _indexer_q_rope_mxfp4_quant_kernel[(num_tokens, num_heads)](
        positions,
        q,
        q.stride(0),
        q.stride(1),
        cos_sin_cache,
        cos_sin_cache.stride(0),
        q_packed,
        q_packed.stride(0),
        q_packed.stride(1),
        q_scale,
        q_scale.stride(0),
        q_scale.stride(1),
        weights,
        weights.stride(0),
        softmax_scale,
        head_scale,
        weights_out,
        weights_out.stride(0),
        HEAD_SIZE=head_size,
        HALF_ROPE=rope_dim // 2,
        num_warps=1,
    )
    return q_packed, q_scale, weights_out
