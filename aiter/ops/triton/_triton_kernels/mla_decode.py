# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

# Copyright (C) 2023-2025 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
Memory-efficient attention for decoding.
It supports page size = 1.
"""

# Adapted from
# https://github.com/ModelTC/lightllm/blob/96353e868a840db4d103138caf15ed9dbea8c186/lightllm/models/deepseek2/triton_kernel/gqa_flash_decoding_stage1.py
# https://github.com/ModelTC/lightllm/blob/96353e868a840db4d103138caf15ed9dbea8c186/lightllm/models/deepseek2/triton_kernel/gqa_flash_decoding_stage2.py

from typing import Optional
import functools
import json
import triton
import triton.language as tl
import torch
import aiter.ops.triton.utils._triton.arch_info as arch_info
from aiter.ops.triton.utils.core import AITER_TRITON_CONFIGS_PATH
from aiter.ops.triton.utils._triton.pid_preprocessing import remap_xcd
from aiter import dtypes

@triton.jit
def _fwd_grouped_kernel_stage1_n16x4_prefetch_k_paged_64(
    Q,  # Holds [Q_NOPE; Q_PE], b x h x (d+r)
    K_Buffer,  # Holds [KV; K_PE], b*s x (c+r)
    V_buffer,  # Holds [KV], b*s x (c)
    sm_scale,
    kv_indptr,
    kv_indices,
    Att_Out,  # b x h x NUM_KV_SPLITS x (kv_lora_rank)
    Att_Lse,  # b x h x NUM_KV_SPLITS x (1)
    stride_qb,
    stride_qh,
    stride_buf_kbs,
    stride_buf_kh,
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    stride_mid_lse_b,
    stride_mid_lse_h,
    stride_mid_lse_s,
    stride_b_block_table,
    dummyPointerArg,
    kv_lora_rank: tl.constexpr,
    qk_rope_head_dim: tl.constexpr,
    kv_group_num: tl.constexpr,
    q_head_num: tl.constexpr,
    batch: tl.constexpr,
    logit_cap: tl.constexpr,
    max_qo_len: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_H: tl.constexpr,
    NUM_KV_SPLITS: tl.constexpr,
    PAGE_BLOCK_SIZE: tl.constexpr,
):
    pass


@triton.jit
def _fwd_grouped_kernel_stage1_n16x2_prefetch_k_paged_64(
    Q,  # Holds [Q_NOPE; Q_PE], b x h x (d+r)
    K_Buffer,  # Holds [KV; K_PE], b*s x (c+r)
    V_buffer,  # Holds [KV], b*s x (c)
    sm_scale,
    kv_indptr,
    kv_indices,
    Att_Out,  # b x h x NUM_KV_SPLITS x (kv_lora_rank)
    Att_Lse,  # b x h x NUM_KV_SPLITS x (1)
    stride_qb,
    stride_qh,
    stride_buf_kbs,
    stride_buf_kh,
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    stride_mid_lse_b,
    stride_mid_lse_h,
    stride_mid_lse_s,
    stride_b_block_table,
    dummyPointerArg,
    kv_lora_rank: tl.constexpr,
    qk_rope_head_dim: tl.constexpr,
    kv_group_num: tl.constexpr,
    q_head_num: tl.constexpr,
    batch: tl.constexpr,
    logit_cap: tl.constexpr,
    max_qo_len: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_H: tl.constexpr,
    NUM_KV_SPLITS: tl.constexpr,
    PAGE_BLOCK_SIZE: tl.constexpr,
):
    pass
