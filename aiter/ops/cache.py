# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import torch
from torch import Tensor
from typing import Optional
from ..jit.core import compile_ops

MD_NAME = "module_cache"


# KV cache memory layout selectors for reshape_and_cache_flash. Must stay in
# sync with the ck_tile::BlockAttentionKVCacheMemoryLayoutEnum values in
# 3rdparty/composable_kernel/include/ck_tile/ops/fmha/block/block_attention_kvcache_layout_enum.hpp
# and with the mirror in aiter/ops/mha.py. Negative value triggers the
# legacy / packed-layout fast path inside the C++ wrapper.
KV_LAYOUT_AUTO = -1
KV_LAYOUT_VECTORIZED = 0
KV_LAYOUT_LINEAR = 1
# Tencent cross-layer 5D KV cache: per-layer non-contiguous view of a 6D
# physical buffer (NumBlocks, NumHeads, NumLayers, 2, PageSize, HeadDim).
# key_cache and value_cache are each 4D [NumBlocks, NumHeads, PageSize, HeadDim],
# sliced out of the 5D (2, ...) view by the framework.
KV_LAYOUT_LINEAR_HEADS_FIRST = 2


@compile_ops("module_cache", develop=True)
def swap_blocks(src: Tensor, dst: Tensor, block_mapping: Tensor) -> None: ...


@compile_ops("module_cache", develop=True)
def copy_blocks(
    key_caches: Tensor, value_caches: Tensor, block_mapping: Tensor
) -> None: ...


@compile_ops("module_cache", develop=True)
def reshape_and_cache(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    kv_cache_dtype: str,
    k_scale: Optional[torch.Tensor] = None,
    v_scale: Optional[torch.Tensor] = None,
    asm_layout: bool = False,
) -> None: ...


@compile_ops("module_cache", develop=True)
def reshape_and_cache_flash(
    key: Tensor,
    value: Tensor,
    key_cache: Tensor,
    value_cache: Tensor,
    slot_mapping: Tensor,
    kv_cache_dtype: str,
    k_scale: Tensor,
    v_scale: Tensor,
    # See KV_LAYOUT_* constants. KV_LAYOUT_AUTO (-1) preserves the legacy
    # packed [N, B, H, D] fast path. Set to KV_LAYOUT_LINEAR_HEADS_FIRST (2)
    # for Tencent cross-layer 5D KV cache writes against a [N, H, B, D] view.
    kv_layout: int = KV_LAYOUT_AUTO,
) -> None: ...


def reshape_and_cache_flash_func(
    key: Tensor,
    value: Tensor,
    key_cache: Optional[Tensor] = None,
    value_cache: Optional[Tensor] = None,
    slot_mapping: Optional[Tensor] = None,
    kv_cache_dtype: str = "auto",
    k_scale: Optional[Tensor] = None,
    v_scale: Optional[Tensor] = None,
    # Tencent cross-layer 5D KV cache entry point: 5D per-layer view
    # `[2, NumBlocks, NumKVHeads, PageSize, HeadDim]`, typically non-contiguous.
    # When supplied, the wrapper auto-slices into key_cache/value_cache and
    # forces kv_layout=KV_LAYOUT_LINEAR_HEADS_FIRST; explicit
    # key_cache/value_cache must be None.
    kv_cache: Optional[Tensor] = None,
    kv_layout: int = KV_LAYOUT_AUTO,
) -> None:
    """High-level wrapper around `reshape_and_cache_flash` that mirrors the
    ergonomics of `aiter.mha_batch_prefill_func`.

    Two call patterns are supported:

    1. Legacy packed K/V caches: pass `key_cache` and `value_cache` directly,
       both 4D `[NumBlocks, PageSize, NumKVHeads, HeadDim]`.

    2. Tencent cross-layer 5D KV cache: pass `kv_cache` instead, a 5D
       per-layer view `[2, NumBlocks, NumKVHeads, PageSize, HeadDim]`
       (typically non-contiguous, sliced from a 6D physical buffer of shape
       `(NumBlocks, NumKVHeads, NumLayers, 2, PageSize, HeadDim)`). The
       wrapper extracts `key_cache = kv_cache[0]` and
       `value_cache = kv_cache[1]` and dispatches with
       `kv_layout=KV_LAYOUT_LINEAR_HEADS_FIRST`.
    """
    if kv_cache is not None:
        if key_cache is not None or value_cache is not None:
            raise ValueError(
                "reshape_and_cache_flash_func: pass either kv_cache "
                "(5D [2, N, H, B, D]) or separate key_cache/value_cache, not both"
            )
        if kv_cache.dim() != 5:
            raise ValueError(
                "kv_cache must be 5D [2, NumBlocks, NumKVHeads, PageSize, HeadDim], "
                f"got dim {kv_cache.dim()}"
            )
        if kv_cache.size(0) != 2:
            raise ValueError(
                "kv_cache outer dim must be 2 (K, V), got " f"{kv_cache.size(0)}"
            )
        if kv_layout == KV_LAYOUT_AUTO:
            kv_layout = KV_LAYOUT_LINEAR_HEADS_FIRST
        elif kv_layout != KV_LAYOUT_LINEAR_HEADS_FIRST:
            raise ValueError(
                "kv_cache implies kv_layout=KV_LAYOUT_LINEAR_HEADS_FIRST, got "
                f"kv_layout={kv_layout}"
            )
        key_cache = kv_cache[0]
        value_cache = kv_cache[1]

    if key_cache is None or value_cache is None:
        raise ValueError(
            "reshape_and_cache_flash_func: must pass key_cache/value_cache or kv_cache"
        )
    if slot_mapping is None:
        raise ValueError("reshape_and_cache_flash_func: slot_mapping is required")

    # k_scale / v_scale are mandatory tensors in the underlying pybind binding
    # even when the dtype is "auto"; provide a sensible default for that case
    # so callers don't have to construct dummy tensors themselves.
    if k_scale is None:
        k_scale = torch.tensor([1.0], dtype=torch.float32, device=key.device)
    if v_scale is None:
        v_scale = torch.tensor([1.0], dtype=torch.float32, device=key.device)

    reshape_and_cache_flash(
        key,
        value,
        key_cache,
        value_cache,
        slot_mapping,
        kv_cache_dtype,
        k_scale,
        v_scale,
        kv_layout,
    )


@compile_ops("module_cache", develop=True)
def reshape_and_cache_with_pertoken_quant(
    key: Tensor,
    value: Tensor,
    key_cache: Tensor,
    value_cache: Tensor,
    k_dequant_scales: Tensor,
    v_dequant_scales: Tensor,
    slot_mapping: Tensor,
    asm_layout: bool,
) -> None: ...


@compile_ops("module_cache", develop=True)
def reshape_and_cache_with_block_quant(
    key: Tensor,
    value: Tensor,
    key_cache: Tensor,
    value_cache: Tensor,
    k_dequant_scales: Tensor,
    v_dequant_scales: Tensor,
    slot_mapping: Tensor,
    asm_layout: bool,
) -> None: ...


@compile_ops("module_cache", develop=True)
def reshape_and_cache_with_block_quant_for_asm_pa(
    key: Tensor,  # [batch_size, seq_len, num_heads, head_size]
    value: Tensor,  # [batch_size, seq_len, num_heads, head_size]
    key_cache: Tensor,  # [num_blocks, num_heads, head_size/x, block_size:16, x]
    value_cache: Tensor,  # [num_blocks, num_heads, head_size, block_size:16] / [num_blocks, kvhead, block_size/x, head_size, x]
    k_dequant_scales: Tensor,  # [num_heads, num_blocks/(ori_block_size/block_size:16)]
    v_dequant_scales: Tensor,  # [num_heads, num_blocks/(ori_block_size/block_size:16)]
    slot_mapping: Tensor,
    asm_layout: bool,
    ori_block_size: int = 128,  # [128/256]
) -> None: ...


@compile_ops("module_cache", develop=True)
def concat_and_cache_mla(
    kv_c: Tensor,
    k_pe: Tensor,
    kv_cache: Tensor,
    slot_mapping: Tensor,
    kv_cache_dtype: str,
    scale: Tensor,
) -> None: ...


@compile_ops("module_cache", develop=True)
def indexer_k_quant_and_cache(
    k: Tensor,
    kv_cache: Tensor,
    slot_mapping: Tensor,
    quant_block_size: int,
    scale_fmt: str,
    preshuffle: bool = False,
) -> None: ...


@compile_ops("module_cache", develop=True)
def indexer_qk_rope_quant_and_cache(
    q: Tensor,
    q_out: Tensor,
    weights: Tensor,
    weights_out: Tensor,
    k: Tensor,
    kv_cache: Tensor,
    slot_mapping: Tensor,
    norm_weight: Tensor,
    norm_bias: Tensor,
    positions: Tensor,
    cos_cache: Tensor,
    sin_cache: Tensor,
    epsilon: float,
    quant_block_size: int,
    scale_fmt: str,
    weights_scale: float,
    preshuffle: bool = False,
    is_neox: bool = True,
) -> None: ...


@compile_ops("module_cache", develop=True)
def cp_gather_indexer_k_quant_cache(
    kv_cache: Tensor,
    dst_k: Tensor,
    dst_scale: Tensor,
    block_table: Tensor,
    cu_seq_lens: Tensor,
    preshuffle: bool = False,
) -> None: ...


@compile_ops("module_cache", develop=True)
def fused_qk_rope_concat_and_cache_mla(
    q_nope: Tensor,
    q_pe: Tensor,  # [num_tokens, num_heads, pe_dim]
    kv_c: Tensor,  # [num_tokens, kv_lora_rank] or [num_tokens, k_num_heads, kv_lora_rank]
    k_pe: Tensor,  # [num_tokens, pe_dim] or [num_tokens, k_num_heads, pe_dim]
    kv_cache: Tensor,  # [num_blocks, block_size, (kv_lora_rank + pe_dim)] or [num_blocks, block_size, k_num_heads, kv_lora_rank + pe_dim)]
    q_out: Tensor,  # [num_tokens, num_heads, qk_lora_rank+pe_dim]
    slot_mapping: Tensor,
    k_scale: Tensor,
    q_scale: Tensor,
    positions: Tensor,  # [num_tokens]
    cos_cache: Tensor,  # [max_position, rot_dim//2]
    sin_cache: Tensor,  # [max_position, rot_dim//2]
    is_neox: bool,
    is_nope_first: bool,
) -> None: ...
