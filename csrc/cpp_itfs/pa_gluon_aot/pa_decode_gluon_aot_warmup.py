# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import torch
import triton
import triton.language as tl

import aiter
from aiter.ops.triton.utils.types import torch_to_triton_dtype
from aiter.ops.triton.gluon.pa_decode_gluon import (
    paged_attention_decode_v2_gluon_dot_kernel,
    paged_attention_decode_v2_gluon_large_block_dot_kernel,
    paged_attention_decode_v2_reduce_kernel,
    get_cdna_version,
)

TORCH_TO_TL_DTYPE_SIG = {
    torch.float8_e4m3fnuz: "fp8e4b8",
    torch.float8_e4m3fn: "fp8e4nv",
}

_DTYPE_MAP = {
    "torch.bfloat16": torch.bfloat16,
    "torch.float16": torch.float16,
    "torch.float8_e4m3fnuz": torch.float8_e4m3fnuz,
    "torch.float8_e4m3fn": torch.float8_e4m3fn,
}


def _get_hsaco(compiled_kernel):
    try:
        return compiled_kernel.asm["hsaco"]
    except (AttributeError, KeyError, TypeError):
        pass
    try:
        return compiled_kernel.kernel
    except AttributeError:
        pass
    raise RuntimeError("Cannot extract HSACO binary from compiled kernel")


def _get_kernel_name(compiled_kernel):
    try:
        return compiled_kernel.name
    except AttributeError:
        pass
    try:
        return compiled_kernel.metadata.name
    except AttributeError:
        pass
    raise RuntimeError("Cannot extract kernel name from compiled kernel")


def _get_shared_mem(compiled_kernel):
    return compiled_kernel.metadata.shared


def _get_num_warps(compiled_kernel):
    return compiled_kernel.metadata.num_warps


def warmup_pa_decode(
    compute_type_str: str,
    query_seq_len: int,
    one_query_group_size: int,
    head_size: int,
    kv_block_size: int,
    context_partition_size: int,
    query_quant_mode: int,
    kv_quant_mode: int,
    fp8_max_value: float,
    value_transposed: int,
    is_causal: int,
    use_sinks: int,
    cdna_version: int,
):
    """
    Warmup compile both attention and reduce kernels.

    Returns a dict with:
        attention_hsaco       : bytes  - HSACO binary for attention kernel
        attention_name        : str    - kernel function name in the HSACO
        attention_shared_mem  : int    - shared memory bytes
        attention_num_warps   : int    - number of warps
        reduce_hsaco          : bytes  - HSACO binary for reduce kernel
        reduce_name           : str    - kernel function name in the HSACO
        reduce_shared_mem     : int    - shared memory bytes
        reduce_num_warps      : int    - number of warps
    """
    device = "cuda"

    compute_type = _DTYPE_MAP[compute_type_str]
    compute_type_tl = torch_to_triton_dtype[compute_type]

    data_type = compute_type
    if compute_type in (torch.float8_e4m3fnuz, torch.float8_e4m3fn):
        data_type = torch.bfloat16

    query_dtype = aiter.dtypes.fp8 if query_quant_mode >= 0 else data_type
    kv_dtype = aiter.dtypes.fp8 if kv_quant_mode >= 0 else data_type
    kv_elements_per_vector = 16 // kv_dtype.itemsize

    head_size_pow2 = triton.next_power_of_2(head_size)
    query_group_size_pow2 = triton.next_power_of_2(
        query_seq_len
    ) * triton.next_power_of_2(one_query_group_size)
    equivalent_query_group_size = query_seq_len * one_query_group_size

    kv_compute_block_size = context_partition_size
    waves_per_eu = 1
    if kv_block_size > context_partition_size:
        if value_transposed:
            kv_compute_block_size = 128
    else:
        if query_group_size_pow2 == 64:
            waves_per_eu = 3
        else:
            waves_per_eu = 4

    num_seqs = 16
    num_kv_heads = 16
    max_context_partition_num = 16
    max_num_blocks_per_seq = 16
    num_blocks = num_seqs * max_num_blocks_per_seq

    exp_sums = torch.empty(
        num_seqs, num_kv_heads, max_context_partition_num,
        equivalent_query_group_size,
        dtype=torch.float32, device=device,
    )
    max_logits = torch.empty_like(exp_sums)
    temporary_output = torch.empty(
        num_seqs, num_kv_heads, max_context_partition_num,
        equivalent_query_group_size, head_size,
        dtype=data_type, device=device,
    )
    query_5d = torch.empty(
        num_seqs, query_seq_len, num_kv_heads, one_query_group_size, head_size,
        dtype=query_dtype, device=device,
    )
    key_cache = torch.empty(
        num_blocks, num_kv_heads,
        head_size // kv_elements_per_vector, kv_block_size, kv_elements_per_vector,
        dtype=kv_dtype, device=device,
    )
    if value_transposed:
        value_cache = torch.empty(
            num_blocks, num_kv_heads,
            kv_block_size // kv_elements_per_vector, head_size, kv_elements_per_vector,
            dtype=kv_dtype, device=device,
        )
    else:
        value_cache = torch.empty(
            num_blocks, num_kv_heads, head_size, kv_block_size,
            dtype=kv_dtype, device=device,
        )
    block_tables = torch.empty(
        num_seqs, max_num_blocks_per_seq,
        dtype=torch.int32, device=device,
    )
    context_lengths = torch.empty(num_seqs, dtype=torch.int32, device=device)

    if query_quant_mode == 0:
        query_scale = torch.empty(1, dtype=torch.float32, device=device)
    elif query_quant_mode == 1:
        query_scale = torch.empty(
            num_seqs, query_seq_len, num_kv_heads, one_query_group_size, 1,
            dtype=torch.float32, device=device,
        )
    else:
        query_scale = torch.empty(1, dtype=torch.float32, device=device)

    if kv_quant_mode == 0:
        key_scale = torch.empty(1, dtype=torch.float32, device=device)
        value_scale = torch.empty(1, dtype=torch.float32, device=device)
    elif kv_quant_mode == 1:
        key_scale = torch.empty(
            num_blocks, num_kv_heads, kv_block_size, 1,
            dtype=torch.float32, device=device,
        )
        value_scale = torch.empty_like(key_scale)
    else:
        key_scale = torch.empty(1, dtype=torch.float32, device=device)
        value_scale = torch.empty(1, dtype=torch.float32, device=device)

    if use_sinks:
        sinks = torch.empty(
            num_seqs, num_kv_heads, equivalent_query_group_size,
            dtype=torch.float32, device=device,
        )
    else:
        sinks = torch.empty(1, dtype=torch.int32, device=device)

    softmax_scale = 1.0 / (head_size ** 0.5)

    stride_query_scale_bs = 0
    stride_query_scale_qlen = 0
    stride_query_scale_kv_head = 0
    if query_scale is not None and query_scale.numel() > 1:
        stride_query_scale_bs = query_scale.stride(0)
        stride_query_scale_qlen = query_scale.stride(1)
        stride_query_scale_kv_head = query_scale.stride(2)

    kv_scale_stride_0 = 0
    kv_scale_stride_1 = 0
    if key_scale is not None and key_scale.numel() > 1:
        kv_scale_stride_0 = key_scale.stride(0)
        kv_scale_stride_1 = key_scale.stride(1)

    if kv_block_size > context_partition_size:
        attention_kernel = paged_attention_decode_v2_gluon_large_block_dot_kernel
    else:
        attention_kernel = paged_attention_decode_v2_gluon_dot_kernel

    attn_grid = (num_seqs, num_kv_heads, max_context_partition_num)

    attn_compiled = attention_kernel.warmup(
        exp_sums, max_logits, temporary_output, query_5d,
        key_cache, value_cache, block_tables, context_lengths,
        softmax_scale, query_scale, key_scale, value_scale,
        exp_sums.stride(0), exp_sums.stride(1), exp_sums.stride(2),
        temporary_output.stride(0), temporary_output.stride(1),
        temporary_output.stride(2), temporary_output.stride(3),
        query_5d.stride(0), query_5d.stride(1),
        query_5d.stride(2), query_5d.stride(3),
        key_cache.stride(0), key_cache.stride(1),
        key_cache.stride(2), key_cache.stride(3),
        value_cache.stride(0), value_cache.stride(1), value_cache.stride(2),
        block_tables.stride(0),
        stride_query_scale_bs, stride_query_scale_qlen, stride_query_scale_kv_head,
        kv_scale_stride_0, kv_scale_stride_1,
        head_size, num_seqs, num_kv_heads, max_context_partition_num,
        COMPUTE_TYPE=compute_type_tl,
        QUERY_SEQ_LEN=query_seq_len,
        ONE_QUERY_GROUP_SIZE=one_query_group_size,
        HEAD_SIZE_POW2=head_size_pow2,
        KV_BLOCK_SIZE=kv_block_size,
        CONTEXT_PARTITION_SIZE=context_partition_size,
        KV_COMPUTE_BLOCK_SIZE=kv_compute_block_size,
        QUERY_QUANT_MODE=query_quant_mode,
        KV_QUANT_MODE=kv_quant_mode,
        FP8_MAX_VALUE=fp8_max_value,
        VALUE_TRANSPOSED=value_transposed,
        IS_CAUSAL=is_causal,
        CDNA_VERSION=cdna_version,
        grid=attn_grid,
        num_warps=4,
        waves_per_eu=waves_per_eu,
        num_stages=1,
    )

    output_5d = torch.empty(
        num_seqs, query_seq_len, num_kv_heads, one_query_group_size, head_size,
        dtype=data_type, device=device,
    )
    reduce_grid = (num_seqs, num_kv_heads, 1)

    reduce_compiled = paged_attention_decode_v2_reduce_kernel.warmup(
        output_5d, exp_sums, max_logits, temporary_output,
        context_lengths, sinks,
        output_5d.stride(0), output_5d.stride(1),
        output_5d.stride(2), output_5d.stride(3),
        exp_sums.stride(0), exp_sums.stride(1), exp_sums.stride(2),
        temporary_output.stride(0), temporary_output.stride(1),
        temporary_output.stride(2), temporary_output.stride(3),
        head_size, num_seqs, num_kv_heads,
        OUTPUT_SEQ_LEN=query_seq_len,
        ONE_OUTPUT_GROUP_SIZE=one_query_group_size,
        HEAD_SIZE_POW2=head_size_pow2,
        CONTEXT_PARTITION_SIZE=context_partition_size,
        USE_SINKS=int(use_sinks),
        grid=reduce_grid,
        num_warps=4,
        num_stages=2,
    )

    return {
        "attention_hsaco": _get_hsaco(attn_compiled),
        "attention_name": _get_kernel_name(attn_compiled),
        "attention_shared_mem": _get_shared_mem(attn_compiled),
        "attention_num_warps": _get_num_warps(attn_compiled),
        "reduce_hsaco": _get_hsaco(reduce_compiled),
        "reduce_name": _get_kernel_name(reduce_compiled),
        "reduce_shared_mem": _get_shared_mem(reduce_compiled),
        "reduce_num_warps": _get_num_warps(reduce_compiled),
    }
