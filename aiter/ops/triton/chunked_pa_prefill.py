# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

# The kernel in this file is adapted from the VLLM project:
# https://github.com/ROCm/vllm/blob/aiter_integration_final/vllm/attention/ops/chunked_prefill_paged_decode.py

# Authors:
#  - Burkhard Ringlein
#  - Jan van Lunteren
#  - Thomas Parnell


import triton
import triton.language as tl

from aiter.ops.triton.pa_prefill import context_attention_fwd
from aiter.ops.triton._triton_kernels.chunked_pa_prefill import (
    _kernel_paged_attention_2d,
)


def chunked_prefill_paged_decode(
    query,
    key,
    value,
    output,
    kv_cache_dtype,
    key_cache,
    value_cache,
    block_table,
    query_start_loc,
    seq_lens,
    max_query_len,
    k_scale,
    v_scale,
    alibi_slopes=None,
    sliding_window=None,
    sm_scale=None,
):
    """
    #TODO: Add Doc
    """
    if sm_scale is None:
        sm_scale = 1.0 / (query.shape[1] ** 0.5)

    use_alibi_slopes = alibi_slopes is not None

    if sliding_window is None or sliding_window <= 0:
        sliding_window = 0

    if max_query_len > 1:
        context_attention_fwd(
            q=query,
            k=key,
            v=value,
            o=output,
            kv_cache_dtype=kv_cache_dtype,
            k_cache=key_cache,
            v_cache=value_cache,
            b_loc=block_table,
            b_start_loc=query_start_loc,
            b_seq_len=seq_lens,
            max_input_len=max_query_len,
            k_scale=k_scale,
            v_scale=v_scale,
            alibi_slopes=alibi_slopes,
            sliding_window=sliding_window,
            sm_scale=sm_scale,
            skip_decode=True,
        )

    block_size = value_cache.shape[3]
    num_seqs = len(seq_lens)
    num_query_heads = query.shape[1]
    num_queries_per_kv = query.shape[1] // key.shape[1]
    head_size = query.shape[2]

    _kernel_paged_attention_2d[
        (
            num_seqs,
            num_query_heads,
        )
    ](
        output_ptr=output,
        query_ptr=query,
        key_cache_ptr=key_cache,
        value_cache_ptr=value_cache,
        block_tables_ptr=block_table,
        seq_lens_ptr=seq_lens,
        alibi_slopes_ptr=alibi_slopes,
        scale=sm_scale,
        k_scale=k_scale,
        v_scale=v_scale,
        num_query_heads=num_query_heads,
        num_queries_per_kv=num_queries_per_kv,
        block_table_stride=block_table.stride(0),
        query_stride_0=query.stride(0),
        query_stride_1=query.stride(1),
        output_stride_0=output.stride(0),
        output_stride_1=output.stride(1),
        BLOCK_SIZE=block_size,
        HEAD_SIZE=head_size,
        HEAD_SIZE_PADDED=triton.next_power_of_2(head_size),
        USE_ALIBI_SLOPES=use_alibi_slopes,
        SLIDING_WINDOW=sliding_window,
        x=key_cache.shape[4],
        stride_k_cache_0=key_cache.stride(0),
        stride_k_cache_1=key_cache.stride(1),
        stride_k_cache_2=key_cache.stride(2),
        stride_k_cache_3=key_cache.stride(3),
        stride_k_cache_4=key_cache.stride(4),
        stride_v_cache_0=value_cache.stride(0),
        stride_v_cache_1=value_cache.stride(1),
        stride_v_cache_2=value_cache.stride(2),
        stride_v_cache_3=value_cache.stride(3),
        filter_by_query_len=True,
        query_start_len_ptr=query_start_loc,
    )
