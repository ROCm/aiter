# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import torch
from torch import Tensor
from typing import Optional
from ..jit.core import compile_ops

MD_NAME = "module_cache"


@compile_ops("module_cache", develop=True)
def swap_blocks(src: Tensor, dst: Tensor, block_mapping: Tensor) -> None: ...


@compile_ops("module_cache", develop=True)
def copy_blocks(
    key_caches: Tensor, value_caches: Tensor, block_mapping: Tensor
) -> None: ...


# Arches where the prebuilt HIP CK module 'module_cache' ships matching
# code objects. On any other arch (e.g. gfx1201 RDNA4 in
# rocm/atom-dev:latest) the HIP kernel launch SIGSEGVs with no catchable
# exception, so we must allowlist arches BEFORE calling the HIP path.
# Update when the prebuilt distribution adds support. Same set + same
# rationale as aiter.ops.gemm_op_a8w8._BLOCKSCALE_HIP_PREBUILT_ARCHES.
_CACHE_HIP_PREBUILT_ARCHES = frozenset({"gfx940", "gfx941", "gfx942", "gfx950"})


def _hip_cache_supported() -> bool:
    """True when the prebuilt 'module_cache' has a code object for the
    running device. False -> fall back to the triton variant."""
    try:
        from ..jit.utils.chip_info import get_gfx_runtime

        return get_gfx_runtime() in _CACHE_HIP_PREBUILT_ARCHES
    except Exception:
        return False


@compile_ops("module_cache", develop=True)
def _reshape_and_cache_ck(
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
) -> None:
    """Scatter (key, value) tokens into a paged KV cache.

    On arches with a prebuilt HIP CK code object for module_cache
    (gfx94x / gfx95x), dispatches to the CK kernel. On other arches
    (e.g. gfx1201 RDNA4) falls back to
    aiter.ops.triton.kv_cache.reshape_and_cache, which JIT-compiles for
    any arch.

    HND layout is used in both paths
    ([num_blocks, num_kv_heads, block_size, head_dim]).

    Triton-fallback constraints:
      - k_scale / v_scale must be None (FP8 KV scales not yet supported
        by the triton path)
      - asm_layout must be False (CDNA-specific layout)
    """
    if _hip_cache_supported():
        return _reshape_and_cache_ck(
            key,
            value,
            key_cache,
            value_cache,
            slot_mapping,
            kv_cache_dtype,
            k_scale,
            v_scale,
            asm_layout,
        )
    # Triton fallback for non-CDNA archs.
    assert k_scale is None and v_scale is None, (
        "reshape_and_cache triton fallback does not yet support FP8 KV "
        "cache scales (k_scale / v_scale must be None)"
    )
    assert not asm_layout, (
        "reshape_and_cache triton fallback does not support asm_layout "
        "(CDNA-specific cache layout)"
    )
    from .triton.kv_cache import reshape_and_cache as _reshape_and_cache_triton

    return _reshape_and_cache_triton(
        key, value, key_cache, value_cache, slot_mapping
    )


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
) -> None: ...


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
