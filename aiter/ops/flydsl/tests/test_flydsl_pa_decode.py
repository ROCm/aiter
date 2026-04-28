# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Correctness tests for FlyDSL gfx1250 paged-attention decode kernel.

Reference: a simple torch SDPA-style implementation over the paged KV cache.
"""

import math

import pytest
import torch

from aiter.ops.flydsl.pa_decode import flydsl_paged_attention_decode


def _torch_reference(
    query: torch.Tensor,       # [num_seqs, num_q_heads, head_size]
    key_cache: torch.Tensor,   # [num_blocks, num_kv_heads, kv_block_size, head_size]
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    attn_scale: float,
) -> torch.Tensor:
    """Compute reference paged attention decode output via torch ops.

    For each sequence, gather its K/V from the paged cache according to
    block_tables[:seq_lens[i]], then do standard scaled dot-product
    attention with one query token per sequence.
    """
    num_seqs, num_q_heads, head_size = query.shape
    _, num_kv_heads, kv_block_size, _ = key_cache.shape
    query_group_size = num_q_heads // num_kv_heads
    device = query.device
    dtype = query.dtype

    out = torch.empty_like(query)
    for s in range(num_seqs):
        ctx_len = int(seq_lens[s].item())
        if ctx_len <= 0:
            out[s].zero_()
            continue
        num_blocks_this_seq = (ctx_len + kv_block_size - 1) // kv_block_size
        blks = block_tables[s, :num_blocks_this_seq].tolist()
        # Assemble K/V for this sequence: [ctx_len, num_kv_heads, head_size]
        k_list, v_list = [], []
        for i, b in enumerate(blks):
            take = min(kv_block_size, ctx_len - i * kv_block_size)
            k_list.append(key_cache[b, :, :take, :].permute(1, 0, 2))   # [take, kv_heads, head]
            v_list.append(value_cache[b, :, :take, :].permute(1, 0, 2))
        K = torch.cat(k_list, dim=0).to(torch.float32)  # [ctx, kv_heads, head]
        V = torch.cat(v_list, dim=0).to(torch.float32)  # [ctx, kv_heads, head]

        # Expand to match num_q_heads via GQA repeat
        K_full = K.repeat_interleave(query_group_size, dim=1)  # [ctx, q_heads, head]
        V_full = V.repeat_interleave(query_group_size, dim=1)

        q = query[s].to(torch.float32)  # [q_heads, head]

        # attention scores: [q_heads, ctx]
        scores = torch.einsum("hd,chd->hc", q, K_full) * attn_scale
        probs = torch.softmax(scores, dim=-1)          # [q_heads, ctx]
        o = torch.einsum("hc,chd->hd", probs, V_full)  # [q_heads, head]
        out[s] = o.to(dtype)

    return out


def _generate_inputs(
    num_seqs: int,
    num_kv_heads: int,
    query_group_size: int,
    head_size: int,
    kv_block_size: int,
    seq_len: int,
    dtype: torch.dtype,
    seed: int = 42,
):
    torch.manual_seed(seed)
    device = "cuda"

    num_q_heads = num_kv_heads * query_group_size
    # Allocate enough blocks for every sequence plus some padding.
    num_blocks_per_seq = (seq_len + kv_block_size - 1) // kv_block_size
    total_blocks = num_seqs * num_blocks_per_seq + 2  # a couple of extras

    query = torch.randn((num_seqs, num_q_heads, head_size), dtype=dtype, device=device) * 0.5
    key_cache = torch.randn(
        (total_blocks, num_kv_heads, kv_block_size, head_size), dtype=dtype, device=device
    ) * 0.5
    value_cache = torch.randn(
        (total_blocks, num_kv_heads, kv_block_size, head_size), dtype=dtype, device=device
    ) * 0.5

    # Build block_tables: scramble block assignments to each sequence.
    all_blocks = torch.randperm(total_blocks, dtype=torch.int32, device=device)
    block_tables = torch.zeros(
        (num_seqs, num_blocks_per_seq), dtype=torch.int32, device=device
    )
    idx = 0
    for s in range(num_seqs):
        block_tables[s] = all_blocks[idx : idx + num_blocks_per_seq]
        idx += num_blocks_per_seq

    # seq_lens: use slightly varying but bounded by seq_len.
    # For simplicity use the same value for all sequences.
    seq_lens = torch.full((num_seqs,), seq_len, dtype=torch.int32, device=device)

    return query, key_cache, value_cache, block_tables, seq_lens


# Small set of cases — FFM is slow, so keep them tiny.
# WMMA kernel requires:
#   HEAD_SIZE >= NUM_WARPS*WAVE_SIZE = 128 and HEAD_SIZE % 32 == 0
#   KV_BLOCK_SIZE % 16 == 0
#   KV_COMPUTE_BLOCK_SIZE % 64 == 0 (NUM_WARPS * WMMA_N) and % KV_BLOCK_SIZE == 0
#   BLOCKS_PER_COMPUTE = KV_COMPUTE_BLOCK_SIZE/KV_BLOCK_SIZE: any positive int
#   PARTITION_SIZE % KV_COMPUTE_BLOCK_SIZE == 0
#   QUERY_GROUP_SIZE == 16 (WMMA M=16)
#   (head, kvb, qg, nkv, nseqs, seq_len, kv_compute, partition)
CASES = [
    # ---- Single-iter path: KV_COMPUTE_BLOCK_SIZE == PARTITION_SIZE ----
    (128, 32, 16, 1, 1, 256, 64, 64),     # BPC=2, 4 partitions
    (128, 16, 16, 1, 1, 256, 64, 64),     # BPC=4, 4 partitions
    (128, 64, 16, 1, 1, 256, 64, 64),     # BPC=1, 4 partitions
    (128, 32, 16, 2, 2, 256, 64, 64),     # multi-seq, multi-kv-head
    (128, 16, 16, 1, 1, 512, 128, 128),   # BPC=8 (lifted constraint), 4 partitions
    (128, 16, 16, 1, 1, 512, 256, 256),   # BPC=16 (gluon-style), 2 partitions

    # ---- Multi-iter path: KV_COMPUTE_BLOCK_SIZE < PARTITION_SIZE ----
    (128, 32, 16, 1, 1, 256, 64, 128),    # 2 compute iters per partition
    (128, 64, 16, 1, 1, 512, 128, 256),   # 2 compute iters per partition
    (128, 32, 16, 1, 1, 384, 64, 192),    # 3 compute iters per partition
    # (64, 16, 8, 1, 1, 1024, 256, 256), # gpt-oss shape to test later after fixing Q masks
]


@pytest.mark.parametrize(
    "head_size,kv_block_size,query_group_size,num_kv_heads,num_seqs,seq_len,kv_compute_block_size,partition_size",
    CASES,
)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_flydsl_pa_decode(
    head_size,
    kv_block_size,
    query_group_size,
    num_kv_heads,
    num_seqs,
    seq_len,
    kv_compute_block_size,
    partition_size,
    dtype,
):
    query, key_cache, value_cache, block_tables, seq_lens = _generate_inputs(
        num_seqs=num_seqs,
        num_kv_heads=num_kv_heads,
        query_group_size=query_group_size,
        head_size=head_size,
        kv_block_size=kv_block_size,
        seq_len=seq_len,
        dtype=dtype,
    )

    attn_scale = 1.0 / math.sqrt(head_size)

    ref = _torch_reference(query, key_cache, value_cache, block_tables, seq_lens, attn_scale)

    output = torch.zeros_like(query)
    flydsl_paged_attention_decode(
        output, query, key_cache, value_cache, block_tables, seq_lens, attn_scale,
        partition_size=partition_size,
        kv_compute_block_size=kv_compute_block_size,
    )

    torch.testing.assert_close(output, ref, atol=0.05, rtol=0.05)
