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
Memory-efficient attention for prefill.
It supporst page size = 1.
"""

# Adapted from
# https://github.com/ModelTC/lightllm/blob/f2a54f0912293f683bf1d1695fd12c4098a5bf82/lightllm/models/llama/triton_kernel/context_flashattention_nopad.py#L1
import triton
# import triton.language as tl
from aiter.ops.triton.utils.logger import AiterTritonLogger
from aiter.ops.triton._triton_kernels.prefill_attention import _fwd_kernel

_LOGGER = AiterTritonLogger()


def context_attention_fwd(
    q, k, v, o, b_start_loc, b_seq_len, max_input_len, is_causal=True
):
    """
    Memory-efficient attention for prefill with page size = 1.

    Args:
        q (torch.Tensor): Query tensor with shape (total_tokens, num_q_heads, head_dim).
        k (torch.Tensor): Key tensor with shape (total_tokens, num_kv_heads, head_dim).
        v (torch.Tensor): Value tensor with shape (total_tokens, num_kv_heads, head_dim).
        o (torch.Tensor): Output tensor with shape (total_tokens, num_q_heads, head_dim).
        b_start_loc (torch.Tensor): Start location for each sequence with shape (batch_size,).
        b_seq_len (torch.Tensor): Sequence length for each batch with shape (batch_size,).
        max_input_len (int): Maximum sequence length in the batch.
        is_causal (bool): Apply causal masking.

    Returns:
        None. Results written in-place to o.
    """
    _LOGGER.info(
        f"PREFILL_ATTENTION: q={tuple(q.shape)} k={tuple(k.shape)} v={tuple(v.shape)}"
    )

    BLOCK = 128
    Lq, Lk = q.shape[-1], k.shape[-1]

    sm_scale = 1.0 / (Lq**0.5)
    batch, head = b_seq_len.shape[0], q.shape[1]
    kv_group_num = q.shape[1] // k.shape[1]

    grid = (batch, head, triton.cdiv(max_input_len, BLOCK))
    num_warps = 4 if Lk <= 64 else 8

    _fwd_kernel[grid](
        q,
        k,
        v,
        sm_scale,
        b_start_loc,
        b_seq_len,
        o,
        q.stride(0),
        q.stride(1),
        k.stride(0),
        k.stride(1),
        v.stride(0),
        v.stride(1),
        o.stride(0),
        o.stride(1),
        kv_group_num=kv_group_num,
        BLOCK_M=BLOCK,
        BLOCK_DMODEL=triton.next_power_of_2(Lk),
        BLOCK_N=BLOCK,
        IS_CAUSAL=is_causal,
        num_warps=num_warps,
        num_stages=1,
        Lk=Lk,
    )
