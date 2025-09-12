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
import triton
import triton.language as tl
import torch
from aiter.ops.triton.utils.logger import AiterTritonLogger
from aiter.ops.triton._triton_kernels.mla_decode_rope import (
    _fwd_grouped_kernel_stage1_rope,
    _fwd_kernel_stage2,
    _get_config,
)

_LOGGER = AiterTritonLogger()


# TODO rope offset
def _decode_grouped_att_m_fwd_rope(
    q,
    k_buffer,
    v_buffer,
    att_out,
    k_pe_tokens_out,
    kv_lora_rank,  # c
    cos_sin_cache,
    positions,
    rotary_dim,
    kv_indptr,
    kv_indices,
    num_kv_splits,
    sm_scale,
    logit_cap,
    use_rope,
    is_neox_style,
    config,
):
    if use_rope:
        assert (
            k_pe_tokens_out is not None
        ), "We must output the k_pe tokens with rope applied if rope fusion enabled."

    qk_rope_head_dim = k_buffer.shape[-1] - kv_lora_rank
    batch, head_num = kv_indptr.shape[0] - 1, q.shape[1]
    kv_group_num = q.shape[1] // k_buffer.shape[1]

    config["BLOCK_C"] = triton.next_power_of_2(kv_lora_rank)
    config["BLOCK_R"] = triton.next_power_of_2(qk_rope_head_dim)

    config["NUM_KV_SPLITS"] = num_kv_splits
    grid = (
        triton.cdiv(head_num, min(config["BLOCK_H"], kv_group_num))
        * batch
        * config["NUM_KV_SPLITS"],
    )

    _fwd_grouped_kernel_stage1_rope[grid](
        q,
        k_buffer,
        v_buffer,
        cos_sin_cache,
        positions,
        sm_scale,
        kv_indptr,
        kv_indices,
        att_out,
        k_pe_tokens_out,
        q.stride(0),
        q.stride(1),
        k_buffer.stride(0),
        v_buffer.stride(0),
        att_out.stride(0),
        att_out.stride(1),
        att_out.stride(2),
        k_pe_tokens_out.stride(0) if use_rope else 0,
        cos_sin_cache.stride(0) if use_rope else 0,
        positions.stride(0) if use_rope else 0,
        rotary_dim,
        kv_lora_rank,
        qk_rope_head_dim,
        kv_group_num=kv_group_num,
        q_head_num=head_num,
        batch=batch,
        logit_cap=logit_cap,
        USE_ROPE=use_rope,
        IS_NEOX_STYLE=is_neox_style,
        **config,
    )


def _decode_softmax_reducev_fwd(
    logits,
    q,
    o,
    v_buffer,
    kv_indptr,
    num_kv_splits,
    config,
):
    batch, head_num = q.shape[0], q.shape[1]
    Lv = v_buffer.shape[-1]
    config["BLOCK_DV"] = triton.next_power_of_2(Lv)

    config["NUM_KV_SPLITS"] = num_kv_splits

    grid = (batch * head_num,)
    _fwd_kernel_stage2[grid](
        logits,
        o,
        kv_indptr,
        logits.stride(0),
        logits.stride(1),
        logits.stride(2),
        o.stride(0),
        o.stride(1),
        Lv=Lv,
        head_num=head_num,
        batch=batch,
        **config,
    )


def decode_attention_fwd_grouped_rope(
    q: torch.Tensor,
    k_buffer: torch.Tensor,
    v_buffer: torch.Tensor,
    o: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    k_pe_tokens: torch.Tensor,
    kv_lora_rank: int,
    rotary_dim: int,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    attn_logits: torch.Tensor,
    num_kv_splits: int,
    sm_scale: float,
    logit_cap: Optional[float] = 0.0,
    use_rope: Optional[bool] = False,
    is_neox_style: Optional[bool] = False,
    config: Optional[dict[str, any]] = None,
):
    """
    Implements deepseek decode attention with grouped query attention and rotary positional encoding

    parameters:
    q: Query Tensor
    k_buffer: Key Cache Tensor
    v_buffer: Value Cache Tensor
    o: Output tensor containing the result of decode. Allocated by the caller
    kv_indptr:
    kv_indices:
    k_pe_tokens:
    kv_lora_rank:
    rotary_dim
    cos_sin_cache:
    positions:
    attn_logits:
    num_kv_splits:
    sm_scale
    logit_cap:
    use_rope
    is_neox_style

    Returns:
    o: output Tensor

    """
    _LOGGER.info(
        f"DECODE_ATTENTION_FWD_GROUPED_ROPE:  q={tuple(q.shape)}  k_buffer={tuple(k_buffer.shape)}  v_buffer={tuple(v_buffer.shape)} "
        + f"k_pe_tokens={tuple(k_pe_tokens.shape) if k_pe_tokens is not None else None} cos_sin_cache={tuple(cos_sin_cache.shape) if cos_sin_cache is not None else None}"
    )
    if config is None:
        config = _get_config()

    _decode_grouped_att_m_fwd_rope(
        q,
        k_buffer,
        v_buffer,
        attn_logits,
        k_pe_tokens,
        kv_lora_rank,
        cos_sin_cache,
        positions,
        rotary_dim,
        kv_indptr,
        kv_indices,
        num_kv_splits,
        sm_scale,
        logit_cap,
        use_rope,
        is_neox_style,
        config["fwd_grouped_kernel_stage1_rope"],
    )
    _decode_softmax_reducev_fwd(
        attn_logits,
        q,
        o,
        v_buffer,
        kv_indptr,
        num_kv_splits,
        config["fwd_kernel_stage2"],
    )
