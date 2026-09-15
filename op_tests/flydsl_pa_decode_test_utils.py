# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Shared variable-context FP32 references for PA scheduling tests and benchmarks."""

import torch

from aiter import dtypes, per_tensor_quant, pertoken_quant
from op_tests.test_flydsl_pa_decode import (
    KV_COMPUTE_BLOCK,
    _quant_dtype,
    _require_gpu,
    run_torch,
)


def _run_accuracy_case(
    lengths,
    *,
    num_query_heads,
    num_kv_heads,
    head_dim,
    block_size,
    query_dtype,
    num_partitions,
    tolerance,
    trans_v=False,
):
    """Run PA against a dequantized-FP8 torch reference without benchmarking."""
    _require_gpu()
    if num_query_heads % num_kv_heads:
        raise ValueError("num_query_heads must be divisible by num_kv_heads")

    torch.manual_seed(0)
    batch_size = len(lengths)
    pages_per_sequence = [
        (context_length + block_size - 1) // block_size for context_length in lengths
    ]
    max_pages = max(pages_per_sequence)
    num_blocks = sum(pages_per_sequence)

    query = torch.empty(
        batch_size, num_query_heads, head_dim, dtype=query_dtype
    ).uniform_(-0.5, 0.5)
    key = torch.empty(
        num_blocks, num_kv_heads, block_size, head_dim, dtype=query_dtype
    ).uniform_(-0.5, 0.5)
    value = torch.empty(
        num_blocks, num_kv_heads, head_dim, block_size, dtype=query_dtype
    ).uniform_(-0.5, 0.5)

    quant_dtype = _quant_dtype()
    key_quant, key_scale = per_tensor_quant(key, quant_dtype=quant_dtype)
    value_quant, value_scale = per_tensor_quant(value, quant_dtype=quant_dtype)
    key_cache = (
        key_quant.view(num_blocks, num_kv_heads, block_size, head_dim // 16, 16)
        .permute(0, 1, 3, 2, 4)
        .contiguous()
    )
    if trans_v:
        value_cache = (
            value_quant.permute(0, 1, 3, 2)
            .contiguous()
            .view(num_blocks, num_kv_heads, block_size // 16, 16, head_dim)
            .permute(0, 1, 2, 4, 3)
            .contiguous()
        )
    else:
        value_cache = value_quant.contiguous()

    block_tables = torch.zeros(
        batch_size, max_pages, dtype=dtypes.i32, device=query.device
    )
    next_page = 0
    for seq_idx, page_count in enumerate(pages_per_sequence):
        block_tables[seq_idx, :page_count] = torch.arange(
            next_page,
            next_page + page_count,
            dtype=dtypes.i32,
            device=query.device,
        )
        next_page += page_count
    context_lengths = torch.tensor(lengths, dtype=dtypes.i32, device=query.device)

    reference = run_torch(
        query,
        key_quant,
        value_quant,
        block_tables,
        context_lengths,
        key_scale,
        value_scale,
    )
    output = torch.empty_like(query)
    torch.ops.aiter.pa_decode_flydsl(
        output,
        query,
        key_cache,
        value_cache,
        context_lengths,
        block_tables,
        head_dim**-0.5,
        1,
        num_partitions,
        KV_COMPUTE_BLOCK,
        quant_dtype,
        None,
        key_scale,
        value_scale,
    )

    torch.testing.assert_close(
        output.float(),
        reference.float(),
        atol=tolerance,
        rtol=tolerance,
    )


def _run_mtp4_fused_reference_case(
    lengths, num_kv_heads, block_size, trans_v, num_partitions
):
    """Check MTP4 with sparse pages, varying scales, and a causal FP32 reference."""
    torch.manual_seed(37)
    query_length, query_group_size, head_dim = 4, 16, 128
    batch_size = len(lengths)
    num_query_heads = num_kv_heads * query_group_size
    pages_per_sequence = [(length + block_size - 1) // block_size for length in lengths]
    max_pages = max(pages_per_sequence)
    num_pages = sum(pages_per_sequence)
    quant_dtype = _quant_dtype()

    query = torch.empty(
        batch_size * query_length,
        num_query_heads,
        head_dim,
        dtype=dtypes.bf16,
    ).uniform_(-0.5, 0.5)
    key = torch.empty(
        num_pages, num_kv_heads, block_size, head_dim, dtype=dtypes.bf16
    ).uniform_(-0.5, 0.5)
    value = torch.empty(
        num_pages, num_kv_heads, head_dim, block_size, dtype=dtypes.bf16
    ).uniform_(-0.5, 0.5)
    key_quant, key_scale = pertoken_quant(key, quant_dtype=quant_dtype)
    value_token_quant, value_scale = pertoken_quant(
        value.permute(0, 1, 3, 2).contiguous(), quant_dtype=quant_dtype
    )
    value_quant = value_token_quant.permute(0, 1, 3, 2).contiguous()

    token = torch.arange(num_pages * block_size).reshape(num_pages, 1, block_size, 1)
    kv_head = torch.arange(num_kv_heads).reshape(1, num_kv_heads, 1, 1)
    key_scale *= torch.exp2(((2 * token + kv_head) % 4 - 2).float())
    value_scale *= torch.exp2(((token + 2 * kv_head) % 5 - 3).float())

    # Make C1--C4 analytically exact while still detecting causal off-by-one:
    # K=0 gives uniform attention, V progresses as [.25, .5, .75, 1], and
    # scale=1. A C4 row therefore produces [.25, .375, .5, .625].
    data_page = 0
    for length, page_count in zip(lengths, pages_per_sequence):
        if 0 < length <= query_length:
            capacity = page_count * block_size
            token_values = (
                torch.arange(capacity, dtype=dtypes.fp32) % query_length + 1
            ) * 0.25
            key_quant[data_page : data_page + page_count].zero_()
            value_quant[data_page : data_page + page_count] = (
                token_values.reshape(page_count, 1, 1, block_size)
                .expand(page_count, num_kv_heads, head_dim, block_size)
                .to(quant_dtype)
            )
            key_scale[data_page : data_page + page_count].fill_(1.0)
            value_scale[data_page : data_page + page_count].fill_(1.0)
        data_page += page_count

    # Scatter logical pages into a permuted set of odd physical pages. Leave
    # holes and page zero finite so masked/padded token reads remain valid.
    selected_pages = 2 * torch.randperm(num_pages) + 1
    physical_page_count = 2 * num_pages + 1

    def scatter_pages(tensor):
        sparse = torch.zeros(
            (physical_page_count, *tensor.shape[1:]), dtype=tensor.dtype
        )
        sparse[selected_pages] = tensor
        return sparse

    key_quant = scatter_pages(key_quant)
    value_quant = scatter_pages(value_quant)
    key_scale = scatter_pages(key_scale)
    value_scale = scatter_pages(value_scale)
    num_pages = physical_page_count

    key_cache = (
        key_quant.view(
            num_pages,
            num_kv_heads,
            block_size,
            head_dim // 16,
            16,
        )
        .permute(0, 1, 3, 2, 4)
        .contiguous()
    )
    if trans_v:
        value_cache = (
            value_quant.permute(0, 1, 3, 2)
            .contiguous()
            .view(
                num_pages,
                num_kv_heads,
                block_size // 16,
                16,
                head_dim,
            )
            .permute(0, 1, 2, 4, 3)
            .contiguous()
        )
    else:
        value_cache = value_quant

    block_tables = torch.zeros(batch_size, max_pages, dtype=dtypes.i32)
    next_page = 0
    for seq, page_count in enumerate(pages_per_sequence):
        block_tables[seq, :page_count] = selected_pages[
            next_page : next_page + page_count
        ]
        next_page += page_count
    context_lengths = torch.tensor(lengths, dtype=dtypes.i32)

    reference = torch.zeros_like(query, dtype=dtypes.fp32)
    for seq, length in enumerate(lengths):
        if length == 0:
            continue
        token_ids = torch.arange(length)
        logical_pages = token_ids // block_size
        token_offsets = token_ids % block_size
        physical_pages = block_tables[seq, logical_pages].long()
        keys = key_quant[physical_pages, :, token_offsets, :].float()
        values = value_quant[physical_pages, :, :, token_offsets].float()
        keys *= key_scale[physical_pages, :, token_offsets, 0].float().unsqueeze(-1)
        values *= value_scale[physical_pages, :, token_offsets, 0].float().unsqueeze(-1)
        keys = keys.repeat_interleave(query_group_size, dim=1)
        values = values.repeat_interleave(query_group_size, dim=1)
        for position in range(query_length):
            visible = max(0, length - (query_length - 1) + position)
            if visible == 0:
                continue
            row = seq * query_length + position
            scores = (
                torch.einsum("hd,khd->hk", query[row].float(), keys[:visible])
                * head_dim**-0.5
            )
            probs = torch.softmax(scores, dim=-1)
            reference[row] = torch.einsum("hk,khd->hd", probs, values[:visible])

    output = torch.full_like(query, float("nan"))
    torch.ops.aiter.pa_decode_flydsl(
        output,
        query,
        key_cache,
        value_cache,
        context_lengths,
        block_tables,
        head_dim**-0.5,
        query_length,
        num_partitions,
        KV_COMPUTE_BLOCK,
        quant_dtype,
        None,
        key_scale,
        value_scale,
    )
    assert torch.isfinite(output).all()
    torch.testing.assert_close(output.float(), reference, rtol=0.005, atol=0.005)
