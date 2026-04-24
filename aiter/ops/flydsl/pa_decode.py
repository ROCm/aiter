# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Paged-attention decode wrapper for FlyDSL on gfx1250.

Computes: out = Softmax(Q @ K^T * scale) @ V
with Q, K, V stored in a paged KV-cache and bf16/f16 precision.

Usage:
    from aiter.ops.flydsl.pa_decode import flydsl_paged_attention_decode

    flydsl_paged_attention_decode(
        output,       # [num_seqs, num_q_heads, head_size]
        query,        # [num_seqs, num_q_heads, head_size]
        key_cache,    # [num_blocks, num_kv_heads, kv_block_size, head_size]
        value_cache,  # [num_blocks, num_kv_heads, kv_block_size, head_size]
        block_tables, # [num_seqs, max_num_blocks_per_seq]
        seq_lens,     # [num_seqs]
        attn_scale,   # float
    )
"""

from __future__ import annotations

import struct
from typing import Optional

import torch

from .kernels.pa_decode_gfx1250 import (
    compile_pa_decode_main,
    compile_pa_decode_reduce,
)


_DEFAULT_PARTITION_SIZE = 256


def _dtype_to_str(dt: torch.dtype) -> str:
    if dt == torch.bfloat16:
        return "bf16"
    if dt == torch.float16:
        return "f16"
    raise ValueError(f"Unsupported dtype for pa_decode: {dt}")


def flydsl_paged_attention_decode(
    output: torch.Tensor,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    attn_scale: float,
    *,
    partition_size: int = _DEFAULT_PARTITION_SIZE,
) -> torch.Tensor:
    """Paged-attention decode for gfx1250.

    Args:
        output: Output tensor [num_seqs, num_q_heads, head_size], bf16/f16.
        query:  Query tensor [num_seqs, num_q_heads, head_size], bf16/f16.
        key_cache: Flat KV cache [num_blocks, num_kv_heads, kv_block_size, head_size].
        value_cache: Same shape as key_cache.
        block_tables: [num_seqs, max_num_blocks_per_seq], int32.
        seq_lens: [num_seqs], int32.
        attn_scale: Softmax scale (usually 1/sqrt(head_size)).
        partition_size: Number of KV tokens per partition. Must be a multiple of
            ``kv_block_size``.

    Returns:
        The ``output`` tensor (for chaining).
    """
    assert output.dim() == 3, f"output must be 3D, got {output.shape}"
    assert query.dim() == 3, f"query must be 3D, got {query.shape}"
    assert key_cache.dim() == 4, f"key_cache must be 4D, got {key_cache.shape}"
    assert value_cache.dim() == 4, f"value_cache must be 4D, got {value_cache.shape}"
    assert block_tables.dim() == 2, f"block_tables must be 2D"
    assert seq_lens.dim() == 1, f"seq_lens must be 1D"
    assert output.dtype == query.dtype == key_cache.dtype == value_cache.dtype, \
        "Q / KV / output dtypes must match"

    num_seqs, num_q_heads, head_size = query.shape
    num_blocks, num_kv_heads, kv_block_size, head_size_kv = key_cache.shape
    assert head_size == head_size_kv, f"Q head_size {head_size} != KV head_size {head_size_kv}"
    assert num_q_heads % num_kv_heads == 0, f"num_q_heads {num_q_heads} not divisible by num_kv_heads {num_kv_heads}"
    query_group_size = num_q_heads // num_kv_heads

    assert block_tables.shape[0] == num_seqs
    assert seq_lens.shape[0] == num_seqs
    assert block_tables.dtype == torch.int32
    assert seq_lens.dtype == torch.int32

    # Contiguity: we assume canonical strides.
    assert query.is_contiguous(), "query must be contiguous"
    assert key_cache.is_contiguous(), "key_cache must be contiguous"
    assert value_cache.is_contiguous(), "value_cache must be contiguous"
    assert output.is_contiguous(), "output must be contiguous"
    assert block_tables.is_contiguous(), "block_tables must be contiguous"
    assert seq_lens.is_contiguous(), "seq_lens must be contiguous"

    max_seq_len = int(seq_lens.max().item()) if num_seqs > 0 else 0
    num_partitions = max(1, (max_seq_len + partition_size - 1) // partition_size)
    max_blocks_per_seq = block_tables.shape[1]

    device = query.device
    dtype_str = _dtype_to_str(query.dtype)

    # Allocate tmp buffers for split-KV outputs.
    tmp_out = torch.empty(
        (num_seqs, num_kv_heads, num_partitions, query_group_size, head_size),
        dtype=torch.float32,
        device=device,
    )
    max_logits = torch.full(
        (num_seqs, num_kv_heads, num_partitions, query_group_size),
        float("-inf"),
        dtype=torch.float32,
        device=device,
    )
    exp_sums = torch.zeros(
        (num_seqs, num_kv_heads, num_partitions, query_group_size),
        dtype=torch.float32,
        device=device,
    )

    # Compile kernels (cached)
    main_launch = compile_pa_decode_main(
        HEAD_SIZE=head_size,
        KV_BLOCK_SIZE=kv_block_size,
        QUERY_GROUP_SIZE=query_group_size,
        PARTITION_SIZE=partition_size,
        dtype=dtype_str,
    )
    reduce_launch = compile_pa_decode_reduce(
        HEAD_SIZE=head_size,
        QUERY_GROUP_SIZE=query_group_size,
        PARTITION_SIZE=partition_size,
        dtype=dtype_str,
    )

    # Pack float32 attn_scale as i32 for kernel arg (kernel bitcasts back).
    scale_i32 = struct.unpack("<i", struct.pack("<f", float(attn_scale)))[0]

    stream = torch.cuda.current_stream()

    # Launch main kernel. Flatten tensors to 1D views the kernels can index.
    main_launch(
        tmp_out.view(-1),
        max_logits.view(-1),
        exp_sums.view(-1),
        query.view(-1),
        key_cache.view(-1),
        value_cache.view(-1),
        block_tables.view(-1),
        seq_lens.view(-1),
        scale_i32,
        num_seqs,
        num_kv_heads,
        num_partitions,
        max_blocks_per_seq,
        stream,
    )

    reduce_launch(
        output.view(-1),
        tmp_out.view(-1),
        max_logits.view(-1),
        exp_sums.view(-1),
        seq_lens.view(-1),
        num_seqs,
        num_kv_heads,
        num_partitions,
        stream,
    )
    torch.cuda.synchronize()
    return output


__all__ = ["flydsl_paged_attention_decode"]
