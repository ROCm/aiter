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
def convert_fp8(
    dst_cache: Tensor, src_cache: Tensor, scale: float, kv_cache_dtype: str
): ...
