# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import torch
from torch import Tensor
from typing import Optional
from ..jit.core import compile_ops

MD_NAME = "module_cache"


@compile_ops("module_cache")
def swap_blocks(src: Tensor, dst: Tensor, block_mapping: Tensor): ...


@compile_ops("module_cache")
def copy_blocks(key_caches: Tensor, value_caches: Tensor, block_mapping: Tensor): ...


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
): ...


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
): ...


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
): ...


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
): ...
