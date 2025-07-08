# SPDX-License-Identifier: MIT
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

from typing import Tuple, Optional
from torch import Tensor
from ..jit.core import compile_ops


MD_NAME = "module_fmla_fwd"


@compile_ops("module_fmla_fwd")
def get_mla_metadata(
    cache_seqlens: Tensor,
    num_heads_per_head_k: int,
    num_heads_k: int,
) -> Tuple[Tensor, Tensor]:
    """
    Arguments:
        cache_seqlens: (batch_size), dtype torch.int32.
        num_heads_per_head_k: Equals to seq_len_q * num_heads_q // num_heads_k.
        num_heads_k: num_heads_k.

    Returns:
        tile_scheduler_metadata: (num_cu_parts, TileSchedulerMetaDataSize), dtype torch.int32.
        num_splits: (batch_size + 1), dtype torch.int32.
    """
    ...

@compile_ops("module_fmla_fwd")
def flash_mla_fwd_with_kvcache_impl(
    q: Tensor,
    k_cache: Tensor,
    v_cache: Tensor,
    head_dim_v: int,
    cache_seqlens: Tensor,
    block_table: Tensor,
    softmax_scale: float,
    causal: bool,
    tile_scheduler_metadata: Tensor,
    num_splits: Tensor
) -> Tuple[Tensor, Tensor]:
    ...

@compile_ops("module_fmla_fwd")
def flash_mla_fwd_prefill_with_kvcache_impl(
    q: Tensor,
    k_cache: Tensor,
    v_cache: Tensor,
    head_dim_v: int,
    cache_seqlens: Tensor,
    block_table: Tensor,
    softmax_scale: float,
    causal: bool,
) -> Tuple[Tensor, Tensor]:
    ...


def flash_mla_fwd_with_kvcache(
    q: Tensor,
    k_cache: Tensor,
    block_table: Tensor,
    cache_seqlens: Tensor,
    head_dim_v: int,
    tile_scheduler_metadata: Tensor,
    num_splits: Tensor,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
) -> Tuple[Tensor, Tensor]:
    """
    Arguments:
        q: (batch_size, seq_len_q, num_heads_q, head_dim).
        k_cache: (num_blocks, page_block_size, num_heads_k, head_dim).
        block_table: (batch_size, max_num_blocks_per_seq), torch.int32.
        cache_seqlens: (batch_size), torch.int32.
        head_dim_v: Head dimension of v.
        tile_scheduler_metadata: (num_cu_parts, TileSchedulerMetaDataSize), torch.int32, returned by get_mla_metadata.
        num_splits: (batch_size + 1), torch.int32, returned by get_mla_metadata.
        softmax_scale: float. The scale of QK^T before applying softmax. Default to 1 / sqrt(head_dim).
        causal: bool. Whether to apply causal attention mask.

    Returns:
        out: (batch_size, seq_len_q, num_heads_q, head_dim_v).
        softmax_lse: (batch_size, num_heads_q, seq_len_q), torch.float32.
    """
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)
    out, softmax_lse = flash_mla_fwd_with_kvcache_impl(
        q,
        k_cache,
        k_cache,
        head_dim_v,
        cache_seqlens,
        block_table,
        softmax_scale,
        causal,
        tile_scheduler_metadata,
        num_splits,
    )
    return out, softmax_lse


def flash_mla_fwd_prefill_with_kvcache(
    q: Tensor,
    k_cache: Tensor,
    block_table: Tensor,
    cache_seqlens: Tensor,
    head_dim_v: int,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
) -> Tuple[Tensor, Tensor]:
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)
    out, softmax_lse = flash_mla_fwd_prefill_with_kvcache_impl(
        q,
        k_cache,
        k_cache,
        head_dim_v,
        cache_seqlens,
        block_table,
        softmax_scale,
        causal)
    return out, softmax_lse
