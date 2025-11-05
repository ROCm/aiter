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
Memory-efficient attention for prefill.
It supports page size = 1 and prefill with KV cache (i.e. extend).
"""

from typing import Optional
import torch
import triton
# import triton.language as tl


from aiter.ops.triton.prefill_attention import context_attention_fwd
from aiter.ops.triton.utils.logger import AiterTritonLogger
from aiter.ops.triton.utils.device_info import get_num_xcds
from aiter.ops.triton._triton_kernels.extend_attention import _fwd_kernel, _get_config

_LOGGER = AiterTritonLogger()


def extend_attention_fwd(
    q_extend,
    k_extend,
    v_extend,
    o_extend,
    k_buffer,
    v_buffer,
    qo_indptr,
    kv_indptr,
    kv_indices,
    custom_mask,
    is_causal,
    mask_indptr,
    max_len_extend,
    sm_scale=None,
    logit_cap=0.0,
    skip_prefix_custom_mask=True,
    config: Optional[dict[str, any]] = None,
):
    """
    Attention for prefill with KV cache (extend phase).
    Supports page size = 1 and variable-length sequences with prefix caching.

    Args:
        q_extend (torch.Tensor): Query tensor for extend tokens with shape (total_extend_tokens, num_q_heads, head_dim).
        k_extend (torch.Tensor): Key tensor for extend tokens with shape (total_extend_tokens, num_kv_heads, head_dim).
        v_extend (torch.Tensor): Value tensor for extend tokens with shape (total_extend_tokens, num_kv_heads, head_dim).
        o_extend (torch.Tensor): Output tensor for extend tokens with shape (total_extend_tokens, num_q_heads, head_dim).
        k_buffer (torch.Tensor): KV cache buffer containing prefix + extend keys with shape (total_tokens, num_kv_heads, head_dim).
        v_buffer (torch.Tensor): KV cache buffer containing prefix + extend values with shape (total_tokens, num_kv_heads, head_dim).
        qo_indptr (torch.Tensor): Index pointer for query/output sequences with shape (batch_size + 1,).
        kv_indptr (torch.Tensor): Index pointer for KV cache sequences with shape (batch_size + 1,).
        kv_indices (torch.Tensor): Indices mapping into KV cache buffer.
        custom_mask (Optional[torch.Tensor]): Custom attention mask tensor.
        is_causal (bool): Apply causal masking.
        mask_indptr (torch.Tensor): Index pointer for custom mask.
        max_len_extend (int): Maximum extend sequence length in batch.
        sm_scale (Optional[float]): Softmax scale, defaults to 1/sqrt(head_dim).
        logit_cap (float): Cap logits to prevent overflow.
        skip_prefix_custom_mask (bool): Skip custom mask for prefix portion.
        config (Optional[dict]): Kernel tuning parameters (BLOCK_M, BLOCK_N).

    Returns:
        None. Results written in-place to o_extend.
    """
    _LOGGER.info(
        f"EXTEND_ATTENTION_FWD: q_extend={tuple(q_extend.shape)} k_extend={tuple(k_extend.shape)} v_extend={tuple(v_extend.shape)} "
        + f"k_buffer={tuple(k_buffer.shape)} v_buffer={tuple(v_buffer.shape)}"
    )

    Lq, Lv = (
        q_extend.shape[-1],
        v_extend.shape[-1],
    )

    if Lq == 576:
        BLOCK_DMODEL = 512
        BLOCK_DPE = 64
    elif Lq == 288:
        BLOCK_DMODEL = 256
        BLOCK_DPE = 32
    elif Lq == 192:
        BLOCK_DMODEL = 128
        BLOCK_DPE = 64
    else:
        BLOCK_DMODEL = triton.next_power_of_2(Lq)
        BLOCK_DPE = 0
    BLOCK_DV = triton.next_power_of_2(Lv)

    sm_scale = sm_scale or 1.0 / (Lq**0.5)
    batch_size, head_num = qo_indptr.shape[0] - 1, q_extend.shape[1]
    kv_group_num = q_extend.shape[1] // k_extend.shape[1]

    USE_CUSTOM_MASK = custom_mask is not None
    # Skip custom mask for prefix part
    SKIP_PREFIX_CUSTOM_MASK = skip_prefix_custom_mask

    if config is None:
        config = _get_config(HEAD_SIZE=Lq, dtype=q_extend.dtype)

    num_blocks = triton.cdiv(max_len_extend, config["BLOCK_M"])
    grid = (head_num * num_blocks * batch_size,)

    _fwd_kernel[grid](
        q_extend,
        k_extend,
        v_extend,
        o_extend,
        k_buffer,
        v_buffer,
        qo_indptr,
        kv_indptr,
        kv_indices,
        custom_mask,
        mask_indptr,
        sm_scale,
        kv_group_num,
        q_extend.stride(0),
        q_extend.stride(1),
        k_extend.stride(0),
        k_extend.stride(1),
        v_extend.stride(0),
        v_extend.stride(1),
        o_extend.stride(0),
        o_extend.stride(1),
        k_buffer.stride(0),
        k_buffer.stride(1),
        v_buffer.stride(0),
        v_buffer.stride(1),
        logit_cap=logit_cap,
        BLOCK_DMODEL=BLOCK_DMODEL,
        BLOCK_DPE=BLOCK_DPE,
        BLOCK_DV=BLOCK_DV,
        # BLOCK_M=BLOCK_M,
        # BLOCK_N=BLOCK_N,
        Lq=Lq,
        Lv=Lv,
        USE_CUSTOM_MASK=USE_CUSTOM_MASK,
        IS_CAUSAL=is_causal,
        SKIP_PREFIX_CUSTOM_MASK=SKIP_PREFIX_CUSTOM_MASK,
        STORE_TRANSPOSE=True,
        NUM_Q_HEADS=head_num,
        NUM_BLOCKS=num_blocks,
        # BATCH=batch_size,
        NUM_XCDS=get_num_xcds(),
        # num_warps=num_warps,
        # num_stages=num_stages,
        **config,
    )


def redundant_attention(
    q_extend,
    o_extend,
    k_buffer,
    v_buffer,
    b_req_idx,
    b_start_loc,
    b_seq_len,
    b_seq_len_prefix,
    max_len_in_batch,
):
    """
    Alternative attention computation for extend tokens using full buffer reconstruction.

    Args:
        q_extend (torch.Tensor): Query tensor for extend tokens with shape (total_extend_tokens, num_q_heads, head_dim).
        o_extend (torch.Tensor): Output tensor for extend tokens with shape (total_extend_tokens, num_q_heads, head_dim).
        k_buffer (torch.Tensor): KV cache buffer for keys with shape (total_tokens, num_kv_heads, head_dim).
        v_buffer (torch.Tensor): KV cache buffer for values with shape (total_tokens, num_kv_heads, head_dim).
        b_req_idx (torch.Tensor): Batch request indices with shape (batch_size,).
        b_start_loc (torch.Tensor): Start locations for each sequence with shape (batch_size,).
        b_seq_len (torch.Tensor): Total sequence lengths (prefix + extend) with shape (batch_size,).
        b_seq_len_prefix (torch.Tensor): Prefix sequence lengths with shape (batch_size,).
        max_len_in_batch (int): Maximum sequence length in the batch.

    Returns:
        None. Results written in-place to o_extend.
    """
    _LOGGER.info(
        f"REDUNDANT_ATTENTION: q_extend={tuple(q_extend.shape)} o_extend={tuple(o_extend.shape)} \
        k_buffer={tuple(k_buffer.shape)} v_buffer={tuple(v_buffer.shape)}"
    )
    total_token_num = k_buffer.shape[0]
    B, H_Q, D = b_req_idx.shape[0], q_extend.shape[-2], q_extend.shape[-1]
    q_buffer = torch.empty(
        (total_token_num, H_Q, D), dtype=q_extend.dtype, device=q_extend.device
    )

    pt = 0
    for i in range(B):
        cur_seq_len_extend = b_seq_len[i] - b_seq_len_prefix[i]
        pl, pr = b_start_loc[i] + b_seq_len_prefix[i], b_start_loc[i] + b_seq_len[i]
        q_buffer[pl:pr] = q_extend[pt : pt + cur_seq_len_extend]
        pt += cur_seq_len_extend

    o_buffer = torch.empty_like(q_buffer)
    context_attention_fwd(
        q_buffer, k_buffer, v_buffer, o_buffer, b_start_loc, b_seq_len, max_len_in_batch
    )

    pt = 0
    for i in range(B):
        cur_seq_len_extend = b_seq_len[i] - b_seq_len_prefix[i]
        pl, pr = b_start_loc[i] + b_seq_len_prefix[i], b_start_loc[i] + b_seq_len[i]
        o_extend[pt : pt + cur_seq_len_extend] = o_buffer[pl:pr]
        pt += cur_seq_len_extend
