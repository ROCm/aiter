# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import torch

from aiter.ops.triton._triton_kernels.pa_mqa_logits import (
    _deepgemm_fp8_paged_mqa_logits_stage1,
    _deepgemm_fp8_paged_mqa_logits_stage1_ragged_k,
    _deepgemm_fp8_paged_mqa_logits,
    _deepgemm_fp8_paged_mqa_logits_ragged_k,
)
from aiter import dtypes


def deepgemm_fp8_paged_mqa_logits_ragged_k(
    q_fp8: torch.Tensor,  # dtype = float8
    kv_cache_fp8: torch.Tensor,  # dtype = float8
    weights: torch.Tensor,  # dtype = float32
    out_logits: torch.Tensor,  # dtype = float32
    prefix_sum_context_lens: torch.Tensor,
    kv_indices: torch.Tensor,
    max_model_len: int,
    ChunkK: int = 64,
    SplitKV: int = 5,
):
    batch_size, next_n, heads, hidden_dim = q_fp8.size()
    kv_cache_fp8, kv_cache_scale = (
        kv_cache_fp8[..., :hidden_dim],
        kv_cache_fp8[..., hidden_dim:],
    )
    # Since triton doesn't have have the reinterpret_cast, we slice the scale out and view it as float
    kv_cache_scale = kv_cache_scale.view(torch.float32)
    kv_cache_fp8 = kv_cache_fp8.view(dtypes.fp8)

    config = {
        "ChunkQ": heads,
        "ChunkK": ChunkK,
        "HiddenDim": hidden_dim,
        "SplitKV": SplitKV,
    }

    grid = (batch_size * next_n * config["SplitKV"],)
    _deepgemm_fp8_paged_mqa_logits_ragged_k[grid](
        batch_size,
        next_n,
        heads,
        q_fp8,
        q_fp8.stride(0),
        q_fp8.stride(1),
        q_fp8.stride(2),
        kv_cache_fp8,
        kv_cache_fp8.stride(0),
        kv_cache_scale,
        kv_cache_scale.stride(0),
        prefix_sum_context_lens,
        kv_indices,
        weights,
        weights.stride(0),
        out_logits,
        out_logits.stride(0),
        max_model_len,
        **config,
    )


def deepgemm_fp8_paged_mqa_logits_stage1_ragged_k(
    q_fp8: torch.Tensor,  # dtype = float8
    kv_cache_fp8: torch.Tensor,  # dtype = float8
    weights: torch.Tensor,  # dtype = float32
    out_qk: torch.Tensor,  # dtype = float32
    prefix_sum_context_lens: torch.Tensor,
    kv_indices: torch.Tensor,
    max_model_len: int,
):
    batch_size, next_n, heads, hidden_dim = q_fp8.size()
    kv_cache_fp8, kv_cache_scale = (
        kv_cache_fp8[..., :hidden_dim],
        kv_cache_fp8[..., hidden_dim:],
    )
    # Since triton doesn't have the reinterpret_cast, we slice the scale out and view it as float
    kv_cache_scale = kv_cache_scale.view(torch.float32)
    kv_cache_fp8 = kv_cache_fp8.view(dtypes.fp8)

    config = {
        "ChunkQ": 32,
        "ChunkK": 64,
        "HiddenDim": hidden_dim,
        "SplitKV": 5,
    }
    assert heads % config["ChunkQ"] == 0

    grid = (batch_size * next_n * (heads // config["ChunkQ"] * config["SplitKV"]),)
    _deepgemm_fp8_paged_mqa_logits_stage1_ragged_k[grid](
        batch_size,
        next_n,
        heads,
        q_fp8,
        q_fp8.stride(0),
        q_fp8.stride(1),
        q_fp8.stride(2),
        kv_cache_fp8,
        kv_cache_fp8.stride(0),
        kv_cache_scale,
        kv_cache_scale.stride(0),
        prefix_sum_context_lens,
        kv_indices,
        weights,
        weights.stride(0),
        out_qk,
        out_qk.stride(0),
        out_qk.stride(1),
        max_model_len,
        **config,
    )


def deepgemm_fp8_paged_mqa_logits_stage1(
    q_fp8: torch.Tensor,  # dtype = float8
    kv_cache_fp8: torch.Tensor,  # dtype = float8 [num_blocks, 1, 1, D+4]
    weights: torch.Tensor,  # dtype = float32
    out_qk: torch.Tensor,  # dtype = float32
    context_lens: torch.Tensor,
    kv_indices: torch.Tensor,
    max_model_len: int,
):
    batch_size, next_n, heads, hidden_dim = q_fp8.size()
    _, max_blk_len = kv_indices.size()
    kv_cache_fp8, kv_cache_scale = (
        kv_cache_fp8[..., :hidden_dim],
        kv_cache_fp8[..., hidden_dim:],
    )
    # Since triton doesn't have the reinterpret_cast, we slice the scale out and view it as float
    kv_cache_scale = kv_cache_scale.view(torch.float32)
    kv_cache_fp8 = kv_cache_fp8.view(dtypes.fp8)

    config = {
        "ChunkQ": 32,
        "ChunkK": 64,
        "HiddenDim": hidden_dim,
        "SplitKV": 5,
    }
    assert heads % config["ChunkQ"] == 0

    grid = (batch_size * next_n * (heads // config["ChunkQ"] * config["SplitKV"]),)
    _deepgemm_fp8_paged_mqa_logits_stage1[grid](
        batch_size,
        next_n,
        heads,
        q_fp8,
        q_fp8.stride(0),
        q_fp8.stride(1),
        q_fp8.stride(2),
        kv_cache_fp8,
        kv_cache_fp8.stride(0),
        kv_cache_scale,
        kv_cache_scale.stride(0),
        context_lens,
        kv_indices,
        weights,
        weights.stride(0),
        out_qk,
        out_qk.stride(0),
        out_qk.stride(1),
        max_model_len,
        max_blk_len,
        **config,
    )


def deepgemm_fp8_paged_mqa_logits(
    q_fp8: torch.Tensor,  # dtype = float8
    kv_cache_fp8: torch.Tensor,  # dtype = float8 [num_blocks, 1, 1, D+4]
    weights: torch.Tensor,  # dtype = float32
    out_logits: torch.Tensor,  # dtype = float32
    context_lens: torch.Tensor,
    kv_indices: torch.Tensor,
    max_model_len: int,
    ChunkK: int = 64,
    SplitKV: int = 5,
):
    batch_size, next_n, heads, hidden_dim = q_fp8.size()
    _, max_blk_len = kv_indices.size()
    kv_cache_fp8, kv_cache_scale = (
        kv_cache_fp8[..., :hidden_dim],
        kv_cache_fp8[..., hidden_dim:],
    )
    # Since triton doesn't have the reinterpret_cast, we slice the scale out and view it as float
    kv_cache_scale = kv_cache_scale.view(torch.float32)
    kv_cache_fp8 = kv_cache_fp8.view(dtypes.fp8)

    config = {
        "ChunkQ": heads,
        "ChunkK": ChunkK,
        "HiddenDim": hidden_dim,
        "SplitKV": SplitKV,
    }

    grid = (batch_size * next_n * config["SplitKV"],)
    _deepgemm_fp8_paged_mqa_logits[grid](
        batch_size,
        next_n,
        heads,
        q_fp8,
        q_fp8.stride(0),
        q_fp8.stride(1),
        q_fp8.stride(2),
        kv_cache_fp8,
        kv_cache_fp8.stride(0),
        kv_cache_scale,
        kv_cache_scale.stride(0),
        context_lens,
        kv_indices,
        weights,
        weights.stride(0),
        out_logits,
        out_logits.stride(0),
        max_model_len,
        max_blk_len,
        **config,
    )
