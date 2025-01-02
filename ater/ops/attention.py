# Copyright (c) 2024 Advanced Micro Devices, Inc.  All rights reserved.
#
# -*- coding:utf-8 -*-
# @Script: attention.py
# @Author: valarLip
# @Email: lingpeng.jin@amd.com
# @Create At: 2024-12-04 21:35:47
# @Last Modified By: valarLip
# @Last Modified At: 2024-12-13 20:56:07
# @Description: This is description.

import os
import torch
from typing import List, Optional
from ..jit.core import compile_ops, CK_DIR, ATER_CSRC_DIR, ATER_ROOT_DIR, get_argsOfBuild
MD_NAME = 'module_attention'

compile_ops_ = get_argsOfBuild("module_attention")

@compile_ops(**compile_ops_)
def pa_fwd_naive(
    # [num_seqs, num_heads, head_size]
    query: torch.Tensor,
    # [num_blocks, num_kv_heads, head_size/x, block_size, x]
    key_cache: torch.Tensor,
    # [num_blocks, num_kv_heads, head_size, block_size]
    value_cache: torch.Tensor,
    # [num_seqs, max_num_blocks_per_seq]
    block_tables: torch.Tensor,
    # [num_seqs]
    context_lens: torch.Tensor,
    k_dequant_scales: torch.Tensor,
    v_dequant_scales: torch.Tensor,
    max_seq_len: int,
    num_kv_heads: int,
    scale_s: float,
    scale_k: float,
    scale_v: float,
    block_size: int,
    quant_algo: int
) -> torch.Tensor: ...

compile_ops_ = get_argsOfBuild("module_attention_asm")
@compile_ops(**compile_ops_)
def pa_fwd_asm(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor) -> torch.Tensor: ...


MD_NAME = "module_pa"

compile_ops_ = get_argsOfBuild("module_pa")
@compile_ops(**compile_ops_)
def paged_attention_rocm(
    out: torch.Tensor,
    exp_sums: torch.Tensor,
    block_mapping: torch.Tensor,
    max_logits: torch.Tensor,
    tmp_out: torch.Tensor,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    num_kv_heads: int,
    scale: float,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    block_size: int,
    max_context_len: int,
    alibi_slopes: Optional[torch.Tensor],
    kv_cache_dtype: str,
    k_scale: float,
    v_scale: float,
    fp8_out_scale: Optional[torch.Tensor],
    partition_size: int,
): ...
