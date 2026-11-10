# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import torch
from torch import Tensor

from ..jit.core import compile_ops

MD_NAME = "module_cache"


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
    k_scale: torch.Tensor | None = None,
    v_scale: torch.Tensor | None = None,
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
def concat_and_cache_mla_seg(
    kv_c: Tensor,  # [num_tokens, kv_lora_rank]
    k_pe: Tensor,  # [num_tokens, pe_dim]
    kv_cache: Tensor,  # [num_blocks, page_size*(kv_lora_rank + pe_dim)] flat (seg layout)
    slot_mapping: Tensor,  # [num_tokens]
    kv_cache_dtype: str,
    scale: Tensor,  # [1] fp32 static scale
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


@compile_ops("module_cache", fc_name="indexer_qk_rope_quant_and_cache", develop=True)
def _indexer_qk_rope_quant_and_cache_hip(
    q: Tensor,
    q_out: Tensor,  # fp8 [T, H, head_dim] | fp4 u8 [T, H, head_dim // 2]
    weights: Tensor,
    weights_out: Tensor,  # fp8: fp32 w * q_scale * weights_scale; fp4: q.dtype plain w
    k: Tensor,
    kv_cache: Tensor,  # fp8 [num_blocks, block_size, cache_stride]
    # fp4 u8 [num_blocks, k_tiles, 4, kv_block_size, 16]
    slot_mapping: Tensor,
    norm_weight: Tensor,
    norm_bias: Tensor,
    positions: Tensor,
    cos_cache: Tensor,
    sin_cache: Tensor,
    epsilon: float,
    quant_block_size: int,  # fp8: head_dim; fp4: 32
    scale_fmt: str,
    weights_scale: float,
    preshuffle: bool = False,
    is_neox: bool = True,
    # False (default): slot<0 rows skip the whole fused op.
    # True (DCP): compute Q/weights for every row; only valid slots write K cache.
    compute_all_q_rope: bool = False,
    # Supplying both scale buffers switches the op to the packed e2m1 + e8m0
    # layout that flydsl_pa_mqa_logits_fp4 consumes directly.
    q_scale_out: Tensor | None = None,  # u8 [T, k_tiles, 4, 16, round_up(H // 16, 4)]
    kv_cache_scale: Tensor | None = None,  # u8 [num_blocks, k_tiles, 4, kv_block_size]
) -> None: ...


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
    positions: Tensor | None,
    cos_cache: Tensor | None,
    sin_cache: Tensor | None,
    epsilon: float,
    quant_block_size: int,
    scale_fmt: str,
    weights_scale: float,
    preshuffle: bool = False,
    is_neox: bool = True,
    compute_all_q_rope: bool = False,
    q_scale_out: Tensor | None = None,
    kv_cache_scale: Tensor | None = None,
    *,
    scale_layout: str = "flydsl",
    rope_dim: int = 64,
) -> None:
    """Normalize K, apply RoPE, quantize Q/K, and write the index cache.

    The default preserves the HIP FP8/FlyDSL FP4 contract. For OPUS FP4,
    use scale_layout='opus', D=128, H=32/64 and rope_dim=0 or 64.
    Both scale outputs are required. The caller owns all output buffers.
    Negative slots skip K. With compute_all_q_rope=True, all Q rows are written.
    FP4 head weights stay unscaled. Return None.
    """
    if scale_layout == "opus":
        if q_scale_out is None or kv_cache_scale is None:
            raise ValueError("OPUS FP4 requires Q and K scale buffers")
        if quant_block_size != 32 or scale_fmt != "ue8m0" or preshuffle:
            raise ValueError("OPUS FP4 requires group32, ue8m0 and preshuffle=False")
        from aiter.ops.triton.fusions.indexer_qk_quant_and_cache_fp4 import (
            indexer_qk_quant_and_cache_fp4,
        )

        return indexer_qk_quant_and_cache_fp4(
            q,
            q_out,
            weights,
            weights_out,
            k,
            kv_cache,
            slot_mapping,
            norm_weight,
            norm_bias,
            positions,
            cos_cache,
            sin_cache,
            epsilon,
            q_scale_out,
            kv_cache_scale,
            rope_dim=rope_dim,
            is_neox=is_neox,
            compute_all_q_rope=compute_all_q_rope,
        )
    if scale_layout != "flydsl" or rope_dim != 64:
        raise ValueError(
            "The legacy indexer writer requires scale_layout='flydsl', RoPE64"
        )
    return _indexer_qk_rope_quant_and_cache_hip(
        q,
        q_out,
        weights,
        weights_out,
        k,
        kv_cache,
        slot_mapping,
        norm_weight,
        norm_bias,
        positions,
        cos_cache,
        sin_cache,
        epsilon,
        quant_block_size,
        scale_fmt,
        weights_scale,
        preshuffle,
        is_neox,
        compute_all_q_rope,
        q_scale_out,
        kv_cache_scale,
    )


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
    # False (default, non-DCP): slot<0 tokens early-return (skip Q RoPE + q_out).
    # True (DCP): compute Q RoPE for every token (needed after head all-gather).
    compute_all_q_rope: bool = False,
) -> None: ...


@compile_ops("module_cache", develop=True)
def fused_qk_rope_concat_and_cache_mla_seg(
    q_nope: Tensor,  # [num_tokens, num_heads, kv_lora_rank=512]
    q_pe: Tensor,  # [num_tokens, num_heads, pe_dim=64]
    kv_c: Tensor,  # [num_tokens, kv_lora_rank=512]
    k_pe: Tensor,  # [num_tokens, pe_dim=64]
    kv_cache: Tensor,  # [num_blocks, page_size*kv_lora + page_size*pe] flat fp8
    q_out: Tensor,  # [num_tokens, num_heads, q_out_dim>=576] fp8 (tail untouched)
    slot_mapping: Tensor,  # [num_tokens]
    k_scale: Tensor,  # [1] fp32 static scale
    q_scale: Tensor,  # [1] fp32 static scale
    positions: Tensor,  # [num_tokens]
    cos_cache: Tensor,  # [max_position, pe_dim//2=32]
    sin_cache: Tensor,  # [max_position, pe_dim//2=32]
    is_neox: bool,
    is_nope_first: bool = True,
) -> None: ...


@compile_ops("module_dsv4_dequant_gather_k", develop=True)
def dsv4_dequantize_and_gather_k(
    out: Tensor,
    k_cache: Tensor,
    seq_lens: Tensor,
    gather_lens: Tensor | None,
    block_table: Tensor,
    block_size: int,
    offset: int = 0,
    use_fnuz: bool = False,
) -> None:
    """Gather and dequantize the DeepSeek V4 paged K record into bf16.

    ``k_cache`` is ``[num_blocks, block_size, 584]`` uint8: per token 448 fp8
    e4m3 NoPE dims then 64 bf16 RoPE dims, with the block's 8-byte-per-token
    UE8M0 scale region (7 scales of 64 dims + pad) after all of its token data.

    Writes ``out[r, offset + i, :512]`` for ``i`` in ``[0, gather_lens[r])``,
    reading sequence positions ``seq_lens[r] - gather_lens[r]`` onward.
    ``gather_lens=None`` gathers the whole sequence. ``use_fnuz`` must match
    the encoder of this cache: e4m3fnuz when True, OCP e4m3 when False.
    """
