# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch

import aiter
from aiter import dtypes

# CK requires the paged KV cache block size to be a multiple of 128.
PAGE_SIZE = 128


def build_paged_kv(k, v, seqlens_k, page_size):
    """Scatter a packed varlen KV buffer into a paged cache.

    Physical block ids are shuffled and every slot the block table does not
    reference is filled with garbage, so a kernel that ignores the block table
    (or over-reads past the last valid token of a page) produces wrong numbers
    instead of accidentally passing.
    """
    batch_size = len(seqlens_k)
    blocks_per_seq = [(s + page_size - 1) // page_size for s in seqlens_k]
    max_blocks_per_seq = max(blocks_per_seq)
    # Over-allocate so that some physical blocks stay unreferenced.
    num_blocks = sum(blocks_per_seq) + batch_size

    _, num_heads_k, head_size_q = k.shape
    head_size_v = v.shape[-1]
    k_cache = torch.full(
        (num_blocks, page_size, num_heads_k, head_size_q),
        100.0,
        dtype=k.dtype,
        device=k.device,
    )
    v_cache = torch.full(
        (num_blocks, page_size, num_heads_k, head_size_v),
        -100.0,
        dtype=v.dtype,
        device=v.device,
    )
    block_table = torch.zeros(
        (batch_size, max_blocks_per_seq), dtype=torch.int32, device=k.device
    )

    physical = torch.randperm(num_blocks, device=k.device)
    next_block = 0
    offset = 0
    for b, seqlen_k in enumerate(seqlens_k):
        for i in range(blocks_per_seq[b]):
            block_id = physical[next_block].item()
            next_block += 1
            block_table[b, i] = block_id
            begin = i * page_size
            end = min(begin + page_size, seqlen_k)
            k_cache[block_id, : end - begin] = k[offset + begin : offset + end]
            v_cache[block_id, : end - begin] = v[offset + begin : offset + end]
        offset += seqlen_k

    return k_cache, v_cache, block_table


@pytest.mark.parametrize("dtype", [dtypes.fp16, dtypes.bf16])
@pytest.mark.parametrize("head_dim", [64, 96])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("return_lse", [False, True])
def test_flash_attn_varlen_func_paged(dtype, head_dim, causal, return_lse):
    """flash_attn_varlen_func(block_table=...) must match the non-paged path.

    head_dim is kept below 128 so that the CK split-kv kernels are exercised;
    bf16 with head_dim=128 dispatches to the ASM v3 kernel instead.
    """
    torch.manual_seed(0)
    device = "cuda"

    num_heads, num_heads_k = 8, 2
    # Deliberately not aligned to PAGE_SIZE, so the last page of every sequence
    # is partially filled.
    seqlens_q = [7, 1, 64, 130]
    seqlens_k = [200, 1, 129, 383]

    def cu_seqlens(seqlens):
        out = [0]
        for s in seqlens:
            out.append(out[-1] + s)
        return torch.tensor(out, dtype=torch.int32, device=device)

    cu_seqlens_q = cu_seqlens(seqlens_q)
    cu_seqlens_k = cu_seqlens(seqlens_k)

    q = torch.randn(sum(seqlens_q), num_heads, head_dim, dtype=dtype, device=device)
    k = torch.randn(sum(seqlens_k), num_heads_k, head_dim, dtype=dtype, device=device)
    v = torch.randn(sum(seqlens_k), num_heads_k, head_dim, dtype=dtype, device=device)

    k_cache, v_cache, block_table = build_paged_kv(k, v, seqlens_k, PAGE_SIZE)

    common = {
        "cu_seqlens_q": cu_seqlens_q,
        "cu_seqlens_k": cu_seqlens_k,
        "max_seqlen_q": max(seqlens_q),
        "max_seqlen_k": max(seqlens_k),
        "causal": causal,
        "return_lse": return_lse,
    }
    dense = aiter.flash_attn_varlen_func(q, k, v, **common)
    paged = aiter.flash_attn_varlen_func(
        q, k_cache, v_cache, block_table=block_table, **common
    )

    if return_lse:
        dense, dense_lse = dense[0], dense[1]
        paged, paged_lse = paged[0], paged[1]
        torch.testing.assert_close(paged_lse, dense_lse, atol=1e-3, rtol=1e-3)

    torch.testing.assert_close(paged, dense, atol=1e-2, rtol=1e-2)
