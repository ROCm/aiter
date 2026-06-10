# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import torch
from torch import Tensor
from typing import Optional
from ..jit.core import compile_ops

MD_NAME = "module_cache"


# KV cache memory layout selectors for reshape_and_cache_flash. Must stay in
# sync with ck_tile::BlockAttentionKVCacheMemoryLayoutEnum and aiter/ops/mha.py.
KV_LAYOUT_AUTO = -1
KV_LAYOUT_VECTORIZED = 0
KV_LAYOUT_LINEAR = 1
KV_LAYOUT_VEC_K_COL_V = 2
KV_LAYOUT_LINEAR_HEADS_FIRST = 3


@compile_ops("module_cache")
def swap_blocks(src: Tensor, dst: Tensor, block_mapping: Tensor) -> None: ...


@compile_ops("module_cache")
def copy_blocks(
    key_caches: Tensor, value_caches: Tensor, block_mapping: Tensor
) -> None: ...


@compile_ops("module_cache")
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


@compile_ops("module_cache")
def reshape_and_cache_flash(
    key: Tensor,
    value: Tensor,
    key_cache: Tensor,
    value_cache: Tensor,
    slot_mapping: Tensor,
    kv_cache_dtype: str,
    k_scale: Tensor,
    v_scale: Tensor,
    kv_layout: int = -1,
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
    kv_cache: Optional[Tensor] = None,
    kv_layout: int = KV_LAYOUT_AUTO,
) -> None:
    """High-level wrapper around `reshape_and_cache_flash`.

    Supports either legacy packed caches `[NumBlocks, PageSize, NumKVHeads,
    HeadDim]` via `key_cache`/`value_cache`, or a cross-layer per-layer
    `kv_cache` view `[2, NumBlocks, NumKVHeads, PageSize, HeadDim]`.
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


@compile_ops("module_cache")
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


@compile_ops("module_cache")
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


@compile_ops("module_cache")
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


@compile_ops("module_cache")
def concat_and_cache_mla(
    kv_c: Tensor,
    k_pe: Tensor,
    kv_cache: Tensor,
    slot_mapping: Tensor,
    kv_cache_dtype: str,
    scale: Tensor,
) -> None: ...


@compile_ops("module_cache")
def indexer_k_quant_and_cache(
    k: Tensor,
    kv_cache: Tensor,
    slot_mapping: Tensor,
    quant_block_size: int,
    scale_fmt: str,
) -> None: ...


@compile_ops("module_cache")
def cp_gather_indexer_k_quant_cache(
    kv_cache: Tensor,
    dst_k: Tensor,
    dst_scale: Tensor,
    block_table: Tensor,
    cu_seq_lens: Tensor,
) -> None: ...


@compile_ops("module_cache")
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
