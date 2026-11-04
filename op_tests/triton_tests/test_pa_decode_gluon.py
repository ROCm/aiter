# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import argparse
import itertools
import os
import random
from types import SimpleNamespace

import pandas as pd
import pytest
import torch
import triton

from aiter import dtypes, logger, per_tensor_quant, pertoken_quant
from aiter.ops.triton.gluon.pa_decode_gluon import (
    get_recommended_splits,
    pa_decode_gluon,
)
from aiter.ops.triton.utils._triton import arch_info
from aiter.test_common import benchmark, checkAllclose, run_perftest

# op_tests/cpp/test_pa_decode_gluon_aot.py replaces TEST_NAME and pa_decode_gluon
# to run normal_accuracy_test() and normal_performance_test() on the AOT backend.
TEST_NAME = "main.normal_accuracy_performance.jit"
UNIFORM_RANGE = (-1, 1)
# test_head_192_full_context_regression needs this seed: see its docstring.
SEED = 123

# (compute_type, quant_q, quant_kv) settings of the case sets.
FP8 = (dtypes.fp8, True, True)
BF16 = (dtypes.bf16, False, False)
BF16_FP8_KV = (dtypes.bf16, False, True)

# Axes of a case-set grid in loop order: the last axis changes fastest.
AXES = (
    "num_heads",
    "compute_quant",
    "trans_v",
    "kv_varlen",
    "quant_mode",
    "block_size",
    "head_size",
    "query_length",
    "batch_size",
    "context_length",
    "use_sinks",
    "sliding_window",
    "ps",
)

# Each case set has one grid for each phase. A later grid changes only the axes
# that it names. Keep normal_accuracy and normal_performance equal to the AOT
# prebuild lists in csrc/cpp_itfs/pa_gluon_aot/pa_decode_gluon_aot_prebuild.py.
CASE_SETS = {
    "normal_accuracy": [
        {
            "num_heads": [(5, 1), (8, 1), (10, 1), (16, 1)],
            "compute_quant": [FP8, BF16],
            "trans_v": [False],
            "kv_varlen": [False, True],
            "quant_mode": ["per_token", "per_tensor"],
            "block_size": [16, 64, 1024],
            "head_size": [128],
            "query_length": [1, 2, 3, 4],
            "batch_size": [3, 81],
            "context_length": [1027],
            "use_sinks": [False],
            "sliding_window": [0],
            "ps": [False],
        },
        # Other head sizes.
        {
            "num_heads": [(8, 1)],
            "kv_varlen": [True],
            "quant_mode": ["per_token"],
            "head_size": [64, 192, 256],
            "query_length": [1, 3],
            "batch_size": [81],
        },
    ],
    "normal_performance": [
        {
            "num_heads": [(64, 4), (64, 8)],
            "compute_quant": [FP8, BF16],
            "trans_v": [False],
            "kv_varlen": [False],
            "quant_mode": ["per_tensor"],
            "block_size": [16],
            "head_size": [128],
            "query_length": [1, 2, 3, 4],
            "batch_size": [1, 2, 4, 8, 16, 32, 64, 128],
            "context_length": [2048, 4096, 8192],
            "use_sinks": [False],
            "sliding_window": [0],
            "ps": [False],
        },
        # The same cases with 64-token blocks.
        {"block_size": [64]},
    ],
    "ps_accuracy": [
        {
            "num_heads": [(8, 1), (16, 1), (64, 8)],
            "compute_quant": [BF16],
            "trans_v": [False],
            "kv_varlen": [False],
            "quant_mode": ["per_tensor"],
            "block_size": [16],
            "head_size": [128],
            "query_length": [1, 4],
            "batch_size": [16],
            "context_length": [2048, 8192],
            "use_sinks": [False],
            "sliding_window": [0],
            "ps": [True],
        },
    ],
    "sliding_window_accuracy": [
        {
            "num_heads": [(64, 8), (16, 1)],
            "compute_quant": [BF16_FP8_KV],
            "trans_v": [False],
            "kv_varlen": [True],
            "quant_mode": ["per_tensor"],
            "block_size": [16],
            "head_size": [128],
            "query_length": [1, 2, 3, 4],
            "batch_size": [1, 128],
            "context_length": [1024, 8192],
            "use_sinks": [True, False],
            "sliding_window": [0, 128],
            "ps": [True],
        },
        # The 5D shuffled V layout with 1024-token blocks.
        {"trans_v": [True], "block_size": [1024]},
    ],
    "sliding_window_performance": [
        {
            "num_heads": [(64, 8)],
            "compute_quant": [FP8, BF16],
            "trans_v": [False],
            "kv_varlen": [True],
            "quant_mode": ["per_tensor"],
            "block_size": [16],
            "head_size": [64],
            "query_length": [1],
            "batch_size": [4, 128],
            "context_length": [1024],
            "use_sinks": [False, True],
            "sliding_window": [0, 128],
            "ps": [True],
        },
    ],
}


def create_kv_cache(
    num_blocks: int,
    block_size: int,
    num_layers: int,
    num_heads: int,
    head_size: int,
    cache_dtype: str | torch.dtype | None,
    model_dtype: str | torch.dtype | None = None,
    seed: int = 0,
    device: str | None = "cuda",
    itemsize: int = 1,
    value_head_size: int | None = None,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Create uniform random key and value caches for num_layers layers.

    Keys are [num_blocks, num_heads, head_size // x, block_size, x] with
    x = 16 // itemsize and values are [num_blocks, num_heads, value_head_size,
    block_size]. cache_dtype is "half", "bfloat16", "float" or "auto" (use
    model_dtype). seed is not used.
    csrc/cpp_itfs/pa_gluon_aot/pa_attention_kernel_test.py imports this function.
    """
    dtype_map = {"half": torch.half, "bfloat16": torch.bfloat16, "float": torch.float}
    if cache_dtype == "auto" and isinstance(model_dtype, torch.dtype):
        dtype = model_dtype
    else:
        dtype = dtype_map.get(model_dtype if cache_dtype == "auto" else cache_dtype)
    if dtype is None:
        raise ValueError(f"Unsupported KV cache dtype {cache_dtype} ({model_dtype})")
    if value_head_size is None:
        value_head_size = head_size

    x = 16 // itemsize
    key_shape = (num_blocks, num_heads, head_size // x, block_size, x)
    value_shape = (num_blocks, num_heads, value_head_size, block_size)
    # Draw all key caches first, then all value caches.
    key_caches = [
        torch.empty(key_shape, dtype=dtype, device=device).uniform_(*UNIFORM_RANGE)
        for _ in range(num_layers)
    ]
    value_caches = [
        torch.empty(value_shape, dtype=dtype, device=device).uniform_(*UNIFORM_RANGE)
        for _ in range(num_layers)
    ]
    return key_caches, value_caches


def quantize_kv_cache_symmetric(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    quant_dtype: torch.dtype,
) -> tuple[torch.Tensor, ...]:
    """Quantize a KV cache in the create_kv_cache layouts per token.

    Returns (keys, key_scales_flat, values, value_scales_flat, key_scales,
    value_scales). The scales are [num_blocks, num_kv_heads, block_size, 1] and
    the flat scales are [num_kv_heads, num_blocks * block_size].
    csrc/cpp_itfs/pa_gluon_aot/pa_attention_kernel_test.py imports this function.
    """
    num_blocks, num_heads, value_head_size, block_size = value_cache.shape
    key_head_size = key_cache.shape[2] * key_cache.shape[4]
    total_tokens = num_blocks * block_size

    key_cache_reshaped = (
        key_cache.permute(0, 1, 3, 2, 4)
        .reshape(num_blocks, num_heads, block_size, -1)
        .contiguous()
    )
    value_cache_reshaped = (
        value_cache.permute(0, 1, 3, 2)
        .reshape(num_blocks, num_heads, block_size, -1)
        .contiguous()
    )
    quantized_keys, key_scales_original = pertoken_quant(
        key_cache_reshaped, quant_dtype=quant_dtype
    )
    quantized_values, value_scales_original = pertoken_quant(
        value_cache_reshaped, quant_dtype=quant_dtype
    )

    x = 16 // quant_dtype.itemsize
    quantized_keys = (
        quantized_keys.view(num_blocks, num_heads, block_size, key_head_size // x, x)
        .permute(0, 1, 3, 2, 4)
        .contiguous()
    )
    key_scales_flat = (
        key_scales_original.permute(1, 0, 2, 3)
        .contiguous()
        .view(num_heads, total_tokens)
    )
    quantized_values = (
        quantized_values.view(num_blocks, num_heads, block_size, value_head_size)
        .permute(0, 1, 3, 2)
        .contiguous()
    )
    value_scales_flat = (
        value_scales_original.permute(1, 0, 2, 3)
        .contiguous()
        .view(num_heads, total_tokens)
    )
    return (
        quantized_keys,
        key_scales_flat,
        quantized_values,
        value_scales_flat,
        key_scales_original,
        value_scales_original,
    )


def quantize_kv_cache_per_tensor(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    quant_dtype: torch.dtype,
) -> tuple[torch.Tensor, ...]:
    """Quantize a KV cache with one scale for the keys and one for the values.

    Returns the same tuple as quantize_kv_cache_symmetric. The scales are [1]
    and the flat scales expand them to [num_kv_heads, num_blocks * block_size].
    csrc/cpp_itfs/pa_gluon_aot/pa_attention_kernel_test.py imports this function.
    """
    num_blocks, num_heads, _, block_size = value_cache.shape
    key_head_size = key_cache.shape[2] * key_cache.shape[4]
    x = 16 // quant_dtype.itemsize

    # Change the key vector width to the one of quant_dtype.
    key_cache_reshaped = (
        key_cache.permute(0, 1, 3, 2, 4)
        .reshape(num_blocks, num_heads, block_size, -1)
        .contiguous()
        .view(num_blocks, num_heads, block_size, key_head_size // x, x)
        .permute(0, 1, 3, 2, 4)
        .contiguous()
    )
    quantized_keys, key_scales_original = per_tensor_quant(
        key_cache_reshaped, quant_dtype=quant_dtype
    )
    quantized_values, value_scales_original = per_tensor_quant(
        value_cache, quant_dtype=quant_dtype
    )
    key_scales_flat = key_scales_original.expand(num_heads, num_blocks * block_size)
    value_scales_flat = value_scales_original.expand(num_heads, num_blocks * block_size)
    return (
        quantized_keys,
        key_scales_flat,
        quantized_values,
        value_scales_flat,
        key_scales_original,
        value_scales_original,
    )


def shuffle_value_cache_layout(value_cache: torch.Tensor) -> torch.Tensor:
    """Shuffle V into the vectorized 5D layout of the kernel.

    [num_blocks, num_kv_heads, head_size, block_size] becomes [num_blocks,
    num_kv_heads, block_size // x, head_size, x] with x = 16 // element size.
    csrc/cpp_itfs/pa_gluon_aot/pa_attention_kernel_test.py imports this function.
    """
    x = 16 // value_cache.element_size()
    num_blocks, num_kv_heads, head_size, block_size = value_cache.shape
    value_cache = value_cache.view(
        num_blocks, num_kv_heads, head_size, block_size // x, x
    )
    return value_cache.permute(0, 1, 3, 2, 4).contiguous()


def make_inputs(
    *,
    batch_size: int,
    num_heads: tuple[int, int],
    head_size: int,
    value_head_size: int,
    block_size: int,
    context_length: int,
    query_length: int,
    data_type: torch.dtype,
    kv_varlen: bool,
    use_sinks: bool,
    quant_kv: bool,
    device: str,
) -> SimpleNamespace:
    """Create the seeded inputs and the paged KV cache of one configuration.

    Do not change the order of the CUDA random draws: it sets the input values.
    """
    torch.manual_seed(SEED)
    num_query_heads, num_kv_heads = num_heads
    # The cache has space for at least 16384 tokens for each sequence.
    total_blocks = triton.cdiv(max(16384, context_length), block_size) * batch_size

    query_output_indptr = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    # Each sequence has query_length queries. The randint draw only moves the RNG.
    query_lengths = torch.randint(
        1, 5, (batch_size,), dtype=torch.int32, device=device
    ).fill_(query_length)
    query_output_indptr[1:] = torch.cumsum(query_lengths, dim=0)
    qkv = torch.randn(
        batch_size * query_length,
        num_query_heads + 2 * num_kv_heads,
        head_size,
        dtype=data_type,
        device=device,
    )
    # The query is a strided view of the first num_query_heads heads.
    query = qkv[:, :num_query_heads].uniform_(*UNIFORM_RANGE)

    # The KV lengths and the block tables each use a new random.Random(SEED).
    rng = random.Random(SEED)
    kv_lens = [
        rng.randint(query_length, context_length) if kv_varlen else context_length
        for _ in range(batch_size)
    ]
    context_lengths = torch.tensor(kv_lens, dtype=torch.int32, device=device)
    sinks = None
    if use_sinks:
        sinks = torch.randn(num_query_heads, dtype=data_type, device=device)
    rng = random.Random(SEED)
    blocks_per_sequence = triton.cdiv(context_length, block_size)
    block_tables = torch.tensor(
        [
            [rng.randint(0, total_blocks - 1) for _ in range(blocks_per_sequence)]
            for _ in range(batch_size)
        ],
        dtype=torch.int32,
        device=device,
    )
    # An FP8 KV cache uses 16 key elements in each vector, a BF16 cache uses 8.
    (key_cache,), (value_cache,) = create_kv_cache(
        total_blocks,
        block_size,
        num_layers=1,
        num_heads=num_kv_heads,
        head_size=head_size,
        cache_dtype="auto",
        model_dtype=data_type,
        device=device,
        itemsize=1 if quant_kv else 2,
        value_head_size=value_head_size,
    )
    return SimpleNamespace(
        query=query,
        query_output_indptr=query_output_indptr,
        kv_lens=kv_lens,
        context_lengths=context_lengths,
        sinks=sinks,
        block_tables=block_tables,
        key_cache=key_cache,
        value_cache=value_cache,
    )


def quantize_kv(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    quant_mode: str,
    quant_kv: bool,
    compute_type: torch.dtype,
) -> tuple[torch.Tensor | None, ...]:
    """Quantize the KV cache to FP8 for one test configuration.

    Returns the 6-tuple layout of quantize_kv_cache_symmetric. The reference
    uses the flat scales and the kernel uses the other scales. Without
    quant_kv, the caches pass through with no scales.
    """
    if not quant_kv:
        return key_cache, None, value_cache, None, None, None
    if quant_mode != "per_token":
        return quantize_kv_cache_per_tensor(
            key_cache, value_cache, quant_dtype=dtypes.fp8
        )
    if compute_type == dtypes.fp8:
        return quantize_kv_cache_symmetric(
            key_cache, value_cache, quant_dtype=dtypes.fp8
        )
    # Without FP8 compute, cast to FP8 and give the kernel unit scales.
    key_scale = torch.tensor(1, dtype=torch.float32, device=key_cache.device)
    value_scale = torch.tensor(1, dtype=torch.float32, device=key_cache.device)
    keys, values = key_cache.to(dtypes.fp8), value_cache.to(dtypes.fp8)
    return keys, None, values, None, key_scale, value_scale


def reference_masked_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    softmax_scale: float,
    output_dtype: torch.dtype,
    sinks: torch.Tensor | None = None,
    sliding_window: int = 0,
) -> torch.Tensor:
    """Causal float32 attention of one sequence.

    query is [query_len, num_heads, head_size], key is [key_len, num_kv_heads,
    head_size] and value is [key_len, num_kv_heads, value_head_size]. A sliding
    window and sinks are optional.
    """
    device = query.device
    query = query.to(torch.float32)
    key = key.to(torch.float32)
    value = value.to(torch.float32)
    query_len, key_len = query.shape[0], key.shape[0]
    group_size = query.shape[1] // key.shape[1]
    key = key.repeat_interleave(group_size, dim=1)
    value = value.repeat_interleave(group_size, dim=1)

    attention_weights = torch.einsum("qhd,khd->hqk", query, key) * softmax_scale
    # Causal mask: query i attends to keys 0 to key_len - query_len + i.
    attention_bias = torch.zeros(query_len, key_len, dtype=torch.float32, device=device)
    causal_mask = torch.ones(query_len, key_len, dtype=torch.bool, device=device)
    causal_mask = causal_mask.tril(diagonal=key_len - query_len)
    attention_bias.masked_fill_(causal_mask.logical_not(), -3.4e38)
    attention_weights += attention_bias

    if sliding_window > 0:
        # In decode, the queries come after the key_len tokens in the cache.
        start = 0 if query_len == key_len else key_len
        query_positions = torch.arange(start, start + query_len, device=device)
        key_positions = torch.arange(key_len, device=device)
        # Attend only where 0 <= query position - key position < sliding_window.
        pos_diff = query_positions.unsqueeze(1) - key_positions.unsqueeze(0)
        outside_window = (pos_diff < 0) | (pos_diff >= sliding_window)
        attention_weights.masked_fill_(outside_window.unsqueeze(0), float("-inf"))

    if sinks is not None:
        logits_max = torch.max(attention_weights, dim=-1, keepdim=True).values
        sink_weights = torch.exp(sinks[:, None, None] - logits_max)
        unnormalized_scores = torch.exp(attention_weights - logits_max)
        normalizer = unnormalized_scores.sum(dim=-1, keepdim=True) + sink_weights
        attention_weights = unnormalized_scores / normalizer
    else:
        attention_weights = torch.softmax(attention_weights, dim=-1)
    output = torch.einsum("hqk,khd->qhd", attention_weights, value)
    return output.to(output_dtype)


def torch_mha_extend(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lengths: torch.Tensor,
    query_output_indptr: torch.Tensor,
    key_scale: torch.Tensor | None = None,
    value_scale: torch.Tensor | None = None,
    sinks: torch.Tensor | None = None,
    sliding_window: int = 0,
) -> torch.Tensor:
    """Reference paged attention decode.

    Gathers the tokens of each sequence from the paged cache, applies the flat
    [num_kv_heads, num_tokens] KV scales and calls reference_masked_attention.
    """
    _, num_heads, value_head_size, block_size = value_cache.shape
    key_head_size = key_cache.shape[2] * key_cache.shape[4]
    softmax_scale = 1.0 / (key_head_size**0.5)
    kv_dtype = key_cache.dtype

    queries = torch.tensor_split(query, query_output_indptr.tolist()[1:])
    # [num_blocks * block_size, num_heads, head_size]
    key_cache_flat = (
        key_cache.permute(0, 3, 1, 2, 4).contiguous().view(-1, num_heads, key_head_size)
    )
    value_cache_flat = (
        value_cache.permute(0, 3, 1, 2)
        .contiguous()
        .view(-1, num_heads, value_head_size)
    )

    num_seqs = query_output_indptr.shape[0] - 1
    outputs = []
    for seq_idx in range(num_seqs):
        context_length = context_lengths[seq_idx].item()
        token_indices = (
            block_tables[seq_idx].repeat_interleave(block_size)[:context_length]
            * block_size
            + torch.arange(context_length, device=block_tables.device) % block_size
        )
        keys = key_cache_flat.view(torch.int8)[token_indices].view(kv_dtype)
        keys = keys.to(torch.float)
        if key_scale is not None:
            keys *= key_scale[:, token_indices].t().unsqueeze(-1)
        values = value_cache_flat.view(torch.int8)[token_indices].view(kv_dtype)
        values = values.to(torch.float)
        if value_scale is not None:
            values *= value_scale[:, token_indices].t().unsqueeze(-1)
        outputs.append(
            reference_masked_attention(
                queries[seq_idx],
                keys,
                values,
                softmax_scale,
                query.dtype,
                sinks=sinks,
                sliding_window=sliding_window,
            )
        )
    return torch.cat(outputs)


def torch_attention_compute(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lengths: torch.Tensor,
    softmax_scale: float,
    q_seq_len: int,
    query_scale: torch.Tensor | None = None,
    key_scale: torch.Tensor | None = None,
    value_scale: torch.Tensor | None = None,
    alibi_slopes: torch.Tensor | None = None,
    compute_type: torch.dtype = torch.bfloat16,
    output_dtype: torch.dtype = torch.bfloat16,
    kv_block_size: int = 16,
    context_partition_size: int = 256,
    is_causal: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reference for the partition stage of the two-stage Gluon decode kernel.

    query is [num_seqs, num_q_heads, head_size] with num_q_heads = num_kv_heads
    * q_seq_len * group_size, in (KV head, query token, group head) order. The
    causal mask uses this order. The caches use the pa_decode_gluon layouts.
    A scale is [1] (per tensor) or per token:
    query_scale [num_seqs, num_q_heads, 1], key_scale and value_scale
    [num_blocks, num_kv_heads, kv_block_size, 1]. Returns exp_sums, max_logits
    and partial_output of each context partition.
    csrc/cpp_itfs/pa_gluon_aot/pa_attention_kernel_test.py imports this function.
    """
    supported_dtypes = [dtypes.fp8, dtypes.bf16, dtypes.fp16]
    for dtype in (query.dtype, key_cache.dtype, value_cache.dtype, compute_type):
        assert dtype in supported_dtypes, f"Unsupported dtype {dtype}"

    device = query.device
    num_seqs, num_q_heads_total, head_size = query.shape
    _, num_kv_heads, _, _, _ = key_cache.shape
    query_group_size = num_q_heads_total // num_kv_heads
    query_group_size_ori = query_group_size // q_seq_len
    assert num_q_heads_total % num_kv_heads == 0

    value_transposed = len(value_cache.shape) == 5
    compute_block_size = 256
    if kv_block_size > context_partition_size and value_transposed:
        compute_block_size = 128

    max_seq_len = context_lengths.max().item()
    num_partitions = triton.cdiv(max_seq_len, context_partition_size)
    intermediate_shape = (num_seqs, num_kv_heads, num_partitions, query_group_size)
    max_logits = torch.full(
        intermediate_shape, -float("inf"), dtype=torch.float32, device=device
    )
    exp_sums = torch.zeros(intermediate_shape, dtype=torch.float32, device=device)
    partial_output = torch.zeros(
        (*intermediate_shape, head_size), dtype=output_dtype, device=device
    )

    FP8_MAX = torch.finfo(dtypes.fp8).max

    # Quant modes: -1 none, 0 per tensor, 1 per token.
    query_quant_mode = -1
    if query_scale is not None:
        query_quant_mode = 0 if query_scale.numel() == 1 else 1
    kv_quant_mode = -1
    if key_scale is not None and value_scale is not None:
        kv_quant_mode = 0 if key_scale.numel() == 1 else 1

    # [num_blocks * block_size, num_kv_heads, head_size]
    key_cache_flat = (
        key_cache.permute(0, 3, 1, 2, 4).contiguous().view(-1, num_kv_heads, head_size)
    )
    value_order = (0, 2, 4, 1, 3) if value_transposed else (0, 3, 1, 2)
    value_cache_flat = (
        value_cache.permute(*value_order).contiguous().view(-1, num_kv_heads, head_size)
    )

    for seq_idx in range(num_seqs):
        seq_len = context_lengths[seq_idx].item()
        block_table = block_tables[seq_idx]

        # Physical cache index of each token of the sequence.
        block_indices = torch.arange(seq_len, device=device) // kv_block_size
        token_offsets = torch.arange(seq_len, device=device) % kv_block_size
        physical_block_ids = block_table[block_indices]
        physical_token_indices = physical_block_ids * kv_block_size + token_offsets

        # [num_kv_heads, query_group_size, head_size]
        q = query[seq_idx].view(num_kv_heads, query_group_size, head_size)
        if query_quant_mode == 0:
            q_scale = query_scale.item()
        elif query_quant_mode == 1:
            # [num_kv_heads, query_group_size, 1]
            q_scale = query_scale[seq_idx].view(num_kv_heads, query_group_size, 1)
        q_fp32 = q.to(compute_type).to(torch.float32)

        for part_idx in range(num_partitions):
            part_start = part_idx * context_partition_size
            part_end = min(part_start + context_partition_size, seq_len)
            if part_start >= seq_len:
                continue

            num_compute_blocks = triton.cdiv(part_end - part_start, compute_block_size)
            part_shape = (num_kv_heads, query_group_size)
            max_logits_part = torch.full(
                (*part_shape, 1), -float("inf"), dtype=torch.float32, device=device
            )
            exp_sums_part = torch.zeros(
                *part_shape, 1, dtype=torch.float32, device=device
            )
            output_part = torch.zeros(
                (*part_shape, head_size), dtype=torch.float32, device=device
            )
            for cb in range(num_compute_blocks):
                cb_start = part_start + cb * compute_block_size
                cb_end = min(cb_start + compute_block_size, part_end)
                cb_len = cb_end - cb_start

                token_range = torch.arange(cb_start, cb_end, device=device)
                indices = physical_token_indices[token_range]
                # [cb_len, num_kv_heads, head_size]
                k = key_cache_flat.view(torch.int8)[indices].view(key_cache.dtype)
                v = value_cache_flat.view(torch.int8)[indices].view(value_cache.dtype)

                if kv_quant_mode == 0:
                    k_scale_vals = key_scale.item()
                    v_scale_vals = value_scale.item()
                elif kv_quant_mode == 1:
                    # [num_kv_heads, 1, cb_len]
                    block_ids = physical_block_ids[token_range]
                    offsets = token_offsets[token_range]
                    k_scale_vals = key_scale[block_ids, :, offsets, 0]
                    v_scale_vals = value_scale[block_ids, :, offsets, 0]
                    k_scale_vals = k_scale_vals.reshape(cb_len, num_kv_heads, 1)
                    k_scale_vals = k_scale_vals.permute(1, 2, 0)
                    v_scale_vals = v_scale_vals.reshape(cb_len, num_kv_heads, 1)
                    v_scale_vals = v_scale_vals.permute(1, 2, 0)

                qk_scale = softmax_scale
                if query_quant_mode >= 0:
                    qk_scale = qk_scale * q_scale
                if kv_quant_mode >= 0:
                    qk_scale = qk_scale * k_scale_vals

                k_fp32 = k.to(compute_type).to(torch.float32)
                v_fp32 = v.to(compute_type).to(torch.float32)

                # [num_kv_heads, query_group_size, cb_len]
                qk = torch.einsum("hqd,khd->hqk", q_fp32, k_fp32)
                qk = qk * qk_scale

                if alibi_slopes is not None:
                    # As in the kernel: alibi_bias = slope * (col - kv_len + 1).
                    slopes = alibi_slopes[:, :query_group_size]
                    positions = token_range.unsqueeze(0).unsqueeze(0)
                    alibi_bias = slopes.unsqueeze(-1) * (positions - seq_len + 1)
                    qk += alibi_bias

                if is_causal:
                    q_positions = torch.arange(query_group_size, device=device)
                    # [query_group_size, cb_len]
                    valid_mask = (
                        q_seq_len
                        - 1
                        - q_positions[:, None] // query_group_size_ori
                        + token_range[None, :]
                        < seq_len
                    )
                    valid_mask = valid_mask[None, :, :]
                    qk = qk.masked_fill(~valid_mask, -3.4e38)

                # Online softmax. [num_kv_heads, query_group_size, 1]
                current_max = qk.max(dim=-1, keepdim=True).values
                new_max = torch.maximum(max_logits_part, current_max)
                acc_scale = torch.exp(max_logits_part - new_max)
                probs = torch.exp(qk - new_max)
                exp_sums_part = acc_scale * exp_sums_part + probs.sum(
                    dim=-1, keepdim=True
                )

                # Scale the probabilities into the FP8 range, as the kernel does.
                if kv_quant_mode == 1:
                    # [num_kv_heads, 1, 1]
                    v_scale_max = v_scale_vals.max(dim=-1, keepdim=True).values
                    v_scale_vals = v_scale_vals * FP8_MAX / (v_scale_max + 1e-8)
                    probs_scaled = v_scale_vals * probs
                    prob_scale = v_scale_max / FP8_MAX
                elif kv_quant_mode == 0:
                    probs_scaled = FP8_MAX * probs
                    prob_scale = v_scale_vals / FP8_MAX
                else:
                    probs_scaled = probs
                probs_scaled = probs_scaled.to(compute_type).to(torch.float32)

                # [num_kv_heads, query_group_size, head_size]
                pv = torch.einsum("hqk,khd->hqd", probs_scaled, v_fp32)
                output_part = acc_scale * output_part
                if kv_quant_mode >= 0:
                    output_part += prob_scale * pv
                else:
                    output_part += pv
                max_logits_part = new_max

            output_part = output_part * (1.0 / exp_sums_part)
            max_logits[seq_idx, :, part_idx, :] = max_logits_part.squeeze(-1)
            exp_sums[seq_idx, :, part_idx, :] = exp_sums_part.squeeze(-1)
            partial_output[seq_idx, :, part_idx, :, :] = output_part.to(output_dtype)

    return exp_sums, max_logits, partial_output


def torch_reduce_compute(
    output: torch.Tensor,
    exp_sums: torch.Tensor,
    max_logits: torch.Tensor,
    temporary_output: torch.Tensor,
    context_lengths: torch.Tensor,
    context_partition_size: int = 256,
) -> torch.Tensor:
    """Reference for the reduce stage: merge the outputs of the partitions.

    output is [num_seqs, num_q_heads, head_size]. exp_sums and max_logits are
    [num_seqs, num_kv_heads, num_partitions, query_group_size] and
    temporary_output adds head_size. Returns a new tensor like output.
    csrc/cpp_itfs/pa_gluon_aot/pa_reduce_kernel_test.py imports this function.
    """
    num_seqs, num_q_heads_total, head_size = output.shape
    final_output = torch.empty_like(output)

    for seq_idx in range(num_seqs):
        seq_len = context_lengths[seq_idx].item()
        num_parts = triton.cdiv(seq_len, context_partition_size)

        # [num_kv_heads, query_group_size]
        global_max = max_logits[seq_idx, :, :num_parts, :].max(dim=1).values
        # [num_kv_heads, num_parts, query_group_size]
        exp_sums_local = exp_sums[seq_idx, :, :num_parts, :]
        max_local = max_logits[seq_idx, :, :num_parts, :]
        exp_sums_rescaled = exp_sums_local * torch.exp(
            max_local - global_max.unsqueeze(1)
        )
        # Avoid a division by zero.
        global_exp_sum = torch.clamp(exp_sums_rescaled.sum(dim=1), min=1e-12)
        weights = exp_sums_rescaled / global_exp_sum.unsqueeze(1)

        # [num_kv_heads, num_parts, query_group_size, head_size]
        partial_seq = temporary_output[seq_idx, :, :num_parts, :, :]
        weighted = (partial_seq * weights.unsqueeze(-1)).sum(dim=1)
        final_output[seq_idx] = weighted.view(num_q_heads_total, head_size)

    return final_output


@benchmark()
def run_pa_gluon_test(
    context_length: int,
    batch_size: int,
    num_heads: tuple[int, int],
    head_size: int,
    block_size: int,
    compute_type: torch.dtype,
    query_length: int,
    quant_mode: str,
    context_partition_size: int,
    trans_v: bool,
    kv_varlen: bool,
    quant_q: bool,
    quant_kv: bool,
    use_sinks: bool,
    sliding_window: int,
    ps: bool,
    value_head_size: int | None = None,
) -> dict[str, float | str]:
    """Run pa_decode_gluon on one configuration and compare it with torch_mha_extend.

    Returns the kernel time in us, err_gluon (1 if an output element is out of
    tolerance, else 0) and the bandwidth in TB/s. @benchmark adds the arguments.
    """
    num_query_heads, num_kv_heads = num_heads
    assert num_query_heads % num_kv_heads == 0, "Q heads must be divisible by KV heads"
    query_group_size = num_query_heads // num_kv_heads
    if value_head_size is None:
        value_head_size = head_size
    # FP8 compute quantizes BF16 inputs.
    data_type = torch.bfloat16 if compute_type == dtypes.fp8 else compute_type
    device = "cuda:0"
    inputs = make_inputs(
        batch_size=batch_size,
        num_heads=num_heads,
        head_size=head_size,
        value_head_size=value_head_size,
        block_size=block_size,
        context_length=context_length,
        query_length=query_length,
        data_type=data_type,
        kv_varlen=kv_varlen,
        use_sinks=use_sinks,
        quant_kv=quant_kv,
        device=device,
    )

    quantized_query, query_scale = inputs.query, None
    if quant_q:
        quantize = pertoken_quant if quant_mode == "per_token" else per_tensor_quant
        quantized_query, query_scale = quantize(inputs.query, quant_dtype=dtypes.fp8)
    key_cache, key_scale_flat, value_cache, value_scale_flat, key_scale, value_scale = (
        quantize_kv(
            inputs.key_cache, inputs.value_cache, quant_mode, quant_kv, compute_type
        )
    )

    # The reference uses the unquantized query and the dequantized KV cache.
    reference = torch_mha_extend(
        inputs.query,
        key_cache,
        value_cache,
        inputs.block_tables,
        inputs.context_lengths,
        inputs.query_output_indptr,
        key_scale_flat,
        value_scale_flat,
        sinks=inputs.sinks,
        sliding_window=sliding_window,
    ).to(data_type)
    # Bytes that the kernel reads and writes. A sliding window limits the tokens.
    kv_tokens = sum(
        min(kv_len, sliding_window) if sliding_window > 0 else kv_len
        for kv_len in inputs.kv_lens
    )
    rw_bytes = kv_tokens * num_kv_heads * (
        head_size * key_cache.dtype.itemsize
        + value_head_size * value_cache.dtype.itemsize
    ) + batch_size * query_length * num_query_heads * (
        head_size * quantized_query.dtype.itemsize
        + value_head_size * reference.dtype.itemsize
    )
    if trans_v:
        value_cache = shuffle_value_cache_layout(value_cache)

    unquantized = compute_type != dtypes.fp8 and not quant_q and not quant_kv
    if sliding_window > 0:
        tolerance = 8e-2
    elif kv_varlen:
        tolerance = 5e-3 if unquantized else 5e-2
    else:
        tolerance = 5e-4 if unquantized else 5e-3

    max_kv_len = max(inputs.kv_lens)
    if sliding_window > 0:
        max_kv_len = min(max_kv_len, sliding_window)
    if sliding_window > 0 and block_size == 1024:
        # The window does not start at a partition boundary: it can touch one
        # more partition.
        num_partitions = triton.cdiv(sliding_window, context_partition_size) + 1
    elif ps:
        num_partitions = get_recommended_splits(
            batch_size,
            num_kv_heads,
            split_kv_blocks=triton.cdiv(block_size, context_partition_size),
            query_length=query_length,
            max_context_length=max_kv_len,
        )
    else:
        num_partitions = triton.cdiv(max_kv_len, context_partition_size)

    # Kernel workspace for the partial results of each context partition.
    shape = (batch_size, num_kv_heads, num_partitions, query_length * query_group_size)
    exp_sums = torch.empty(shape, dtype=torch.float32, device=device)
    max_logits = torch.empty(shape, dtype=torch.float32, device=device)
    temporary_output = torch.empty(
        *shape, value_head_size, dtype=data_type, device=device
    )
    output = torch.empty_like(reference)
    softmax_scale = 1.0 / (head_size**0.5)

    # Read the pa_decode_gluon global at call time: the AOT test replaces it.
    _, us = run_perftest(
        pa_decode_gluon,
        output,
        quantized_query,
        key_cache,
        value_cache,
        inputs.context_lengths,
        inputs.block_tables,
        softmax_scale,
        query_length,
        num_partitions,
        context_partition_size,
        compute_type,
        query_scale=query_scale,
        key_scale=key_scale,
        value_scale=value_scale,
        exp_sums=exp_sums,
        max_logits=max_logits,
        temporary_output=temporary_output,
        sinks=inputs.sinks,
        sliding_window=sliding_window,
        ps=ps,
    )
    err_ratio = checkAllclose(
        reference,
        output,
        atol=tolerance,
        rtol=tolerance,
        msg=f"[pa_decode_gluon vs torch][{quant_mode}] {us:.2f} us ",
    )
    if err_ratio > 0:
        logger.warning(
            "pa_decode_gluon: %.2f%% of the output is out of tolerance (atol=rtol=%g)",
            100 * err_ratio,
            tolerance,
        )
    return {
        "us_gluon": us,
        "err_gluon": int(err_ratio > 0),
        "gluon_bandwith(TB/s)": rw_bytes / (us * 1e6 * 1.024**4),
    }


def _expand(grid):
    """Yield the run_pa_gluon_test arguments of each grid point, in loop order."""
    for values in itertools.product(*(grid[axis] for axis in AXES)):
        config = dict(zip(AXES, values))
        compute_type, quant_q, quant_kv = config.pop("compute_quant")
        config.update(compute_type=compute_type, quant_q=quant_q, quant_kv=quant_kv)
        # Without quantization all quant modes are equal: run only the first one.
        if quant_q or quant_kv or config["quant_mode"] == grid["quant_mode"][0]:
            yield config


def run_case_set(name, overrides=None, sample_rate=1.0):
    """Run each phase of CASE_SETS[name] and assert that all cases pass.

    overrides replaces axes in each phase. With sample_rate < 1, each case runs
    with that probability. Outside pytest, each phase writes a CSV report.
    """
    overrides = overrides or {}
    rng = random.Random()
    grid = {}
    for phase, update in enumerate(CASE_SETS[name], 1):
        grid = {**grid, **update, **overrides}
        configs = list(_expand(grid))
        if sample_rate < 1.0:
            configs = [config for config in configs if rng.random() < sample_rate]
        results = []
        for i, config in enumerate(configs, 1):
            logger.info("%s phase %d: case %d of %d", name, phase, i, len(configs))
            # The kernel supports only 256-token context partitions.
            results.append(run_pa_gluon_test(**config, context_partition_size=256))

        df = pd.DataFrame(results)
        if "PYTEST_CURRENT_TEST" not in os.environ:
            csv_path = (
                f"run_pa_gluon_test.{TEST_NAME}.block_size_{grid['block_size'][0]}"
                f".triton.{triton.__version__}.csv"
            )
            df.to_csv(csv_path, index=False)
            logger.info("Results saved to %s", csv_path)
        if df.empty:
            continue
        logger.info("%s phase %d results:\n%s", name, phase, df)
        means = df.groupby(df["compute_type"].astype(str), sort=False)[
            ["us_gluon", "gluon_bandwith(TB/s)"]
        ].mean()
        logger.info("Mean by compute_type:\n%s", means)
        failed = df[df["err_gluon"] > 0]
        assert failed.empty, (
            f"{name} phase {phase}: {len(failed)} of {len(df)} cases are out of "
            f"tolerance:\n{failed.head(10).to_string()}"
        )


def normal_accuracy_test():
    """Run the normal_accuracy case set. The AOT accuracy test calls this."""
    run_case_set("normal_accuracy")


def normal_performance_test():
    """Run the normal_performance case set. The AOT performance run calls this."""
    run_case_set("normal_performance")


@pytest.mark.parametrize("quant_kv", [False, True], ids=["bf16-kv", "fp8-kv"])
@pytest.mark.parametrize(
    ("query_length", "num_query_heads", "context_length", "ps"),
    [
        pytest.param(1, 16, 512, True, id="decode"),
        pytest.param(4, 16, 512, True, id="qlen4"),
        pytest.param(1, 16, 256, False, id="one-shot"),
        pytest.param(4, 4, 512, True, id="packed-mtp"),
        pytest.param(4, 4, 256, False, id="packed-mtp-one-shot"),
    ],
)
@pytest.mark.parametrize(
    ("sliding_window", "use_sinks"),
    [(0, False), (128, True)],
    ids=["full-attention", "swa-with-sinks"],
)
def test_pa_decode_qk192_v128_vectorized_5d(
    quant_kv: bool,
    query_length: int,
    num_query_heads: int,
    context_length: int,
    ps: bool,
    sliding_window: int,
    use_sinks: bool,
):
    """Cover split and one-shot output paths, including packed qlen-4 MTP.

    ps=False with context_length=256 forces a single partition. Four Q heads
    keep QUERY_SEQ_LEN_POW2=4 instead of dispatching each query separately.
    """
    if arch_info.get_arch() not in ("gfx942", "gfx950"):
        pytest.skip("asymmetric Gluon paged decode supports gfx942/gfx950")

    result = run_pa_gluon_test(
        context_length=context_length,
        batch_size=2,
        num_heads=(num_query_heads, 1),
        head_size=192,
        value_head_size=128,
        block_size=64,
        compute_type=torch.bfloat16,
        query_length=query_length,
        quant_mode="per_tensor",
        context_partition_size=256,
        trans_v=True,
        kv_varlen=False,
        quant_q=False,
        quant_kv=quant_kv,
        use_sinks=use_sinks,
        sliding_window=sliding_window,
        ps=ps,
    )
    assert result["err_gluon"] == 0, "pa_decode_gluon output is out of tolerance"


def test_pa_decode_qk192_v128_temporary_output():
    """Reject unsafe workspaces and retain support for reusable padded storage."""
    if arch_info.get_arch() not in ("gfx942", "gfx950"):
        pytest.skip("asymmetric Gluon paged decode supports gfx942/gfx950")

    query = torch.zeros((1, 16, 192), dtype=torch.bfloat16, device="cuda")
    key_cache = torch.zeros((8, 1, 24, 64, 8), dtype=query.dtype, device=query.device)
    value_cache = torch.ones((8, 1, 8, 128, 8), dtype=query.dtype, device=query.device)
    output = torch.empty((1, 16, 128), dtype=query.dtype, device=query.device)
    context_lengths = torch.tensor([512], dtype=torch.int32, device=query.device)
    block_tables = torch.arange(8, dtype=torch.int32, device=query.device).view(1, 8)
    storage = torch.full(
        (1, 1, 2, 16, 256), -77, dtype=query.dtype, device=query.device
    )
    workspace = storage[..., :128]

    def launch(temporary_output):
        pa_decode_gluon(
            output,
            query,
            key_cache,
            value_cache,
            context_lengths,
            block_tables,
            softmax_scale=192**-0.5,
            query_length=1,
            max_context_partition_num=2,
            temporary_output=temporary_output,
            ps=True,
        )

    for invalid in (
        workspace[..., :64],
        workspace[:, :, :1],
        storage.flatten(),
        storage[..., ::2],
    ):
        with pytest.raises(ValueError, match="temporary_output"):
            launch(invalid)

    # Zero Q/K and constant-one V have an independent, exact output of one.
    launch(workspace)
    torch.testing.assert_close(
        output.float(), torch.ones_like(output, dtype=torch.float32), rtol=0, atol=0
    )
    padding = storage[..., 128:]
    assert torch.all(padding == -77), "pa_decode_gluon wrote outside temporary_output"


@pytest.mark.parametrize(
    ("context_length", "ps"),
    [(2048, True), (4097, True), (4097, False)],
    ids=["causal-split", "persistent-oob", "partitioned-oob"],
)
def test_head_192_full_context_regression(context_length, ps):
    """Cover head size 192 (padded to 256 in the kernel) with qlen 4 and FP8 KV.

    The 2048-token persistent case checks that the qlen-4 causal frontier
    applies at the end of the context, not at each split. With SEED = 123, the
    block tables of the 4097-token cases include the last block of the KV cache
    (block 8191). Before the padded K/V and reduction loads were masked, they
    read past that block and caused a GPU memory access fault. Other seeds can
    miss that block.
    """
    result = run_pa_gluon_test(
        context_length=context_length,
        batch_size=32,
        num_heads=(16, 1),
        head_size=192,
        block_size=64,
        compute_type=torch.bfloat16,
        query_length=4,
        quant_mode="per_tensor",
        context_partition_size=256,
        trans_v=True,
        kv_varlen=False,
        quant_q=False,
        quant_kv=True,
        use_sinks=False,
        sliding_window=0,
        ps=ps,
    )
    assert result["err_gluon"] == 0, "pa_decode_gluon output is out of tolerance"


@pytest.mark.parametrize(
    "num_seqs, num_kv_heads, kwargs, expected",
    [
        (1, 8, {}, 8),
        (1, 8, {"max_context_length": 32768}, 32),
        (1, 8, {"max_context_length": 12000}, 32),
        (1, 2, {"max_context_length": 4096}, 16),
        (1, 8, {"max_context_length": 1024}, 8),
        (2, 8, {"max_context_length": 32768}, 32),
        (1, 2, {"max_context_length": 8192}, 32),
        (4, 8, {"max_context_length": 32768}, 8),
        (1, 8, {"max_context_length": 32768, "query_length": 4}, 8),
        (1, 8, {"max_context_length": 32768, "split_kv_blocks": 4}, 8),
    ],
)
def test_get_recommended_splits(monkeypatch, num_seqs, num_kv_heads, kwargs, expected):
    """Check the recommended split counts on a GPU with 256 compute units."""
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda *a, **k: SimpleNamespace(multi_processor_count=256),
    )
    assert get_recommended_splits(num_seqs, num_kv_heads, **kwargs) == expected


@pytest.mark.parametrize(
    "kv_heads, lengths",
    [
        (8, [8192]),
        (8, [12000]),
        (8, [32768]),
        (4, [32768, 300]),
        (2, [32768, 1000]),
    ],
)
def test_ps_recommended_splits_match_reference(kv_heads, lengths):
    """Check ps=True with the recommended split count against an FP32 reference."""
    torch.manual_seed(0)
    group, head_size, page, x = 8, 64, 16, 8
    batch = len(lengths)
    pages_per_seq = [triton.cdiv(n, page) for n in lengths]
    max_pages = max(pages_per_seq)
    total_pages = sum(pages_per_seq)
    kwargs = {"device": "cuda", "dtype": torch.bfloat16}
    query = torch.randn(batch, kv_heads * group, head_size, **kwargs)
    key_cache = torch.randn(total_pages, kv_heads, head_size // x, page, x, **kwargs)
    value_cache = torch.randn(total_pages, kv_heads, page // x, head_size, x, **kwargs)
    context_lengths = torch.tensor(lengths, dtype=torch.int32, device="cuda")
    block_tables = torch.zeros(batch, max_pages, dtype=torch.int32, device="cuda")
    first_page = 0
    for i, n in enumerate(pages_per_seq):
        block_tables[i, :n] = torch.arange(first_page, first_page + n, device="cuda")
        first_page += n
    scale = head_size**-0.5

    splits = get_recommended_splits(batch, kv_heads, max_context_length=max(lengths))
    shape = (batch, kv_heads, splits, group)
    exp_sums = torch.empty(shape, dtype=torch.float32, device="cuda")
    max_logits = torch.empty_like(exp_sums)
    temporary_output = torch.empty((*shape, head_size), **kwargs)
    output = torch.zeros_like(query)
    pa_decode_gluon(
        output=output,
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        context_lengths=context_lengths,
        block_tables=block_tables,
        softmax_scale=scale,
        query_length=1,
        max_context_partition_num=splits,
        context_partition_size=256,
        compute_type=query.dtype,
        exp_sums=exp_sums,
        max_logits=max_logits,
        temporary_output=temporary_output,
        ps=True,
    )

    # Per sequence: the short ones of a ragged batch also run with the batch-wide split count.
    for i, n in enumerate(lengths):
        table = block_tables[i, : pages_per_seq[i]].long()
        k = key_cache[table].permute(1, 0, 3, 2, 4).reshape(kv_heads, -1, head_size)
        v = value_cache[table].permute(1, 0, 2, 4, 3).reshape(kv_heads, -1, head_size)
        k, v = k[:, :n].float(), v[:, :n].float()
        q = query[i].view(kv_heads, group, head_size).float()
        probs = torch.softmax(q @ k.transpose(1, 2) * scale, dim=-1)
        reference = (probs @ v).reshape(kv_heads * group, head_size)
        torch.testing.assert_close(output[i].float(), reference, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize(
    "case_set_name",
    [
        "normal_accuracy",
        "ps_accuracy",
        "sliding_window_accuracy",
        "sliding_window_performance",
    ],
)
def test_multi_case_set(case_set_name):
    """Run all configurations of one case set.

    normal_performance runs only from the command line and from the AOT test.
    """
    run_case_set(case_set_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run one case set of the Gluon paged attention decode test."
    )
    parser.add_argument(
        "--case_set",
        choices=list(CASE_SETS),
        default="sliding_window_accuracy",
        help="Case set to run",
    )
    parser.add_argument(
        "-n", "--num_heads", type=dtypes.str2tuple, help="Query and KV heads: 8,1"
    )
    parser.add_argument(
        "-q", "--query_length", type=int, choices=[1, 2, 3, 4], help="Query length"
    )
    parser.add_argument("-c", "--context_length", type=int, help="Context length")
    parser.add_argument("-b", "--batch_size", type=int, help="Batch size")
    parser.add_argument(
        "-d", "--head_dim", dest="head_size", type=int, help="Head size"
    )
    parser.add_argument("--block_size", type=int, help="KV cache block size")
    parser.add_argument(
        "--quant_mode",
        choices=["per_token", "per_tensor"],
        help="Quantization mode of Q and KV",
    )
    parser.add_argument(
        "--trans_v",
        type=dtypes.str2bool,
        help="Use the 5D shuffled V cache layout (true/false)",
    )
    parser.add_argument(
        "--kv_varlen",
        type=dtypes.str2bool,
        help="Use random KV lengths up to context_length (true/false)",
    )
    parser.add_argument(
        "--sample_rate", type=float, default=1.0, help="Probability to run each case"
    )
    args = vars(parser.parse_args())
    case_set, sample_rate = args.pop("case_set"), args.pop("sample_rate")
    # Each flag that is set replaces its axis in all phases of the case set.
    overrides = {axis: [value] for axis, value in args.items() if value is not None}
    logger.info("Triton version: %s", triton.__version__)
    run_case_set(case_set, overrides, sample_rate)
