# SPDX-License-Identifier: MIT
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

import torch

from aiter.ops.triton._triton_kernels.gather_kv_b_proj import (
    _triton_gather_kv_b_proj,
)


def gather_kv_b_proj(
    k_buffer: torch.Tensor,  # [num_block, block_size, hidden_dim]
    k_scale: torch.Tensor,  # [1]
    kv_indptr: torch.Tensor,  # [batch_size + 1]
    kv_indices: torch.Tensor,  # len(kv_indices) = kv_indptr[-1]
    kv_prefix_sum_context_lens: torch.Tensor,  # [batch_size + 1]
    kv_proj_weight: torch.Tensor,  # [2 * 128 // TP * 128, 512]
    kv_proj_scale: torch.Tensor,  # [2 * 128 // TP, 4], blockscale=128 x 128
    k_prefix: torch.Tensor,  # [total_kv, tp_k_head_num, qk_nope_head_dim + kv_pe_dim]
    v_prefix: torch.Tensor,  # [total_kv, tp_k_head_num, qk_nope_head_dim]
):
    num_block, block_size, hidden_dim = k_buffer.shape
    batch_size = kv_indptr.shape[0] - 1
    weight_n, weight_k = kv_proj_weight.shape
    scale_n, scale_k = kv_proj_scale.shape
    total_kv_k, tp_k_head_num_k, qk_nope_pe_dim = k_prefix.shape
    total_kv_v, tp_k_head_num_v, qk_nope_dim = v_prefix.shape

    scale_k_granularity = weight_k // scale_k
    scale_n_granularity = weight_n // scale_n

    ChunkK = 16 if k_buffer.dtype in [torch.float16, torch.bfloat16] else 32

    assert total_kv_k == total_kv_v
    assert tp_k_head_num_k == tp_k_head_num_v
    assert scale_k_granularity == 128
    assert scale_n_granularity == 128
    assert ChunkK % block_size == 0

    grid = (batch_size * tp_k_head_num_k,)
    kernel = _triton_gather_kv_b_proj[grid](
        batch_size,
        k_buffer,
        k_scale,
        kv_indptr,
        kv_indices,
        kv_prefix_sum_context_lens,
        kv_proj_weight,
        kv_proj_scale,
        k_prefix,
        v_prefix,
        KBlockSize=block_size,
        TpNumHeads=tp_k_head_num_k,
        QkNopeHeadDim=qk_nope_dim,
        KV_CDim=weight_k,
        KV_PeDim=qk_nope_pe_dim - qk_nope_dim,
        ChunkK=ChunkK,
        num_stages=3,
    )
