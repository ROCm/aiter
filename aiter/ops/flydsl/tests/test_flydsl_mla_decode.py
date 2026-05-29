# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Correctness tests for the FlyDSL gfx1250 MLA (Multi-head Latent Attention)
decode kernel.
"""

import math
import random

import pytest
import torch

from aiter.ops.flydsl.mla_decode import flydsl_mla_decode


# ---------------------------------------------------------------------------
# KV-cache pre-shuffle (from test_mla.py)
#
# Reorders each 16-row x (2*num_elements_per_thread)-col tile into lane-major
# order so the kernel can feed WMMA fragments with straight ds_read_b128 (no
# LDS padding for bank-conflict avoidance). The reference always runs on the
# UNSHUFFLED logical cache; only the kernel receives the shuffled copy.
# ---------------------------------------------------------------------------
def shuffle_kv_buffer(
    kv_buffer: torch.Tensor,  # [num_blocks, block_size, num_kv_heads, head_size]
    kv_lora_rank: int,
) -> torch.Tensor:
    """Shuffle KV cache layout for optimized WMMA-fragment loads.

    layout: (num_lanes, num_elements_per_thread) = (16, 8) for bf16/fp16 on gfx1250.
    WMMA instruction shape (bf16/fp16): 16x16x32.

    Returns a contiguous tensor shaped
    ``[num_blocks, num_kv_heads, block_size, head_size]`` with the bytes
    reordered within each tile.
    """
    dtype = kv_buffer.dtype
    assert dtype in (torch.bfloat16, torch.float16), f"unsupported dtype {dtype}"

    # 16-bit dtypes use a (16, 8) lane layout on gfx1250.
    num_lanes, num_elements_per_thread = (16, 8)

    num_blocks, block_size, num_kv_heads, head_size = kv_buffer.shape
    assert block_size >= 16
    assert block_size % num_lanes == 0

    def shuffle(kvb, h):
        kvb = kvb.view(
            -1,
            num_kv_heads,
            block_size // num_lanes,
            num_lanes,
            h // (2 * num_elements_per_thread),
            2,  # 2 thread groups: t0..t15 and t16..t31
            num_elements_per_thread,
        )
        kvb = kvb.permute(0, 1, 2, 4, 5, 3, 6).contiguous()
        kvb = kvb.view(-1, num_kv_heads, block_size // 16, h * 16)
        return kvb

    kv_shuffled = kv_buffer.view(-1, block_size, num_kv_heads, head_size).permute(
        0, 2, 1, 3
    )
    lora = shuffle(kv_shuffled[..., :kv_lora_rank], kv_lora_rank)
    rope = shuffle(kv_shuffled[..., kv_lora_rank:], head_size - kv_lora_rank)
    lora = lora.view(-1, num_kv_heads, block_size * kv_lora_rank)
    rope = rope.view(-1, num_kv_heads, block_size * (head_size - kv_lora_rank))
    kv_shuffled = torch.cat([lora, rope], dim=-1).contiguous()
    kv_shuffled = kv_shuffled.view(-1, num_kv_heads, block_size, head_size)
    return kv_shuffled


# ---------------------------------------------------------------------------
# Reference (golden): lifted from test_mla.py, trimmed to decode-only bf16/fp16.
# ---------------------------------------------------------------------------
def _ref_masked_attention(
    q: torch.Tensor,  # [1, num_q_heads, head_size]   (query_len == 1: decode)
    k: torch.Tensor,  # [kv_len, num_kv_heads, head_size]
    v: torch.Tensor,  # [kv_len, num_kv_heads, kv_lora_rank]
    scale: float,
) -> torch.Tensor:

    if q.shape[1] != k.shape[1]:  # GQA / MQA expand kv heads up to query heads
        k = torch.repeat_interleave(k, q.shape[1] // k.shape[1], dim=1)
        v = torch.repeat_interleave(v, q.shape[1] // v.shape[1], dim=1)
    k = k.to(q.dtype)
    attn = torch.einsum("qhd,khd->hqk", q, k).float()  # [num_q_heads, 1, kv_len]
    attn = attn * scale
    attn = torch.softmax(attn, dim=-1).to(q.dtype)
    v = v.to(q.dtype)
    out = torch.einsum("hqk,khd->qhd", attn, v)  # [1, num_q_heads, kv_lora_rank]
    return out


def _torch_mla_decode_ref(
    query: torch.Tensor,  # [num_seqs, num_q_heads, head_size]
    kv_cache: torch.Tensor,  # [num_blocks, block_size, num_kv_heads, head_size]
    block_tables: torch.Tensor,  # [num_seqs, max_num_blocks_per_seq]
    seq_lens: torch.Tensor,  # [num_seqs]
    scale: float,
    kv_lora_rank: int,
) -> torch.Tensor:
    """MLA decode golden. Returns [num_seqs, num_q_heads, kv_lora_rank].
    """
    num_seqs, num_q_heads, head_size = query.shape
    _, block_size, num_kv_heads, qk_head_dim = kv_cache.shape
    assert head_size == qk_head_dim
    device = query.device

    outputs = []
    for i in range(num_seqs):
        kv_len = int(seq_lens[i].item())
        if kv_len <= 0:
            outputs.append(
                torch.zeros(
                    (1, num_q_heads, kv_lora_rank), dtype=query.dtype, device=device
                )
            )
            continue

        num_kv_blocks = (kv_len + block_size - 1) // block_size
        block_indices = block_tables[i, :num_kv_blocks]
        k = kv_cache[block_indices].view(-1, num_kv_heads, qk_head_dim)[:kv_len]
        v = k[..., :kv_lora_rank]

        q = query[i : i + 1]  # [1, num_q_heads, head_size]
        outputs.append(_ref_masked_attention(q, k, v, scale))

    return torch.cat(outputs, dim=0)  # [num_seqs, num_q_heads, kv_lora_rank]


def _generate_inputs(
    num_seqs: int,
    num_query_heads: int,
    num_kv_heads: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    block_size: int,
    ctx_len: int,
    dtype: torch.dtype,
    varlen: bool = False,
    num_blocks: int | None = None,
    seed: int = 42,
    device: str = "cuda",
):
    torch.manual_seed(seed)
    random.seed(seed)

    qk_head_dim = kv_lora_rank + qk_rope_head_dim

    if varlen:
        lens = [
            int(max(random.normalvariate(ctx_len, ctx_len / 2), ctx_len))
            for _ in range(num_seqs)
        ]
        seq_lens = torch.tensor(lens, dtype=torch.int32, device=device)
    else:
        seq_lens = torch.full((num_seqs,), ctx_len, dtype=torch.int32, device=device)

    # Block-table width derived from the realized max
    max_seqlen = int(seq_lens.max().item())
    max_num_blocks_per_seq = (max_seqlen + block_size - 1) // block_size

    if num_blocks is None:
        num_blocks = max_num_blocks_per_seq * num_seqs + 16

    block_tables = torch.randint(
        0,
        num_blocks,
        (num_seqs, max_num_blocks_per_seq),
        dtype=torch.int32,
        device=device,
    )

    kv_cache = torch.randn(
        (num_blocks, block_size, num_kv_heads, qk_head_dim),
        dtype=torch.bfloat16,
        device=device,
    ).to(dtype)
    query = torch.randn(
        (num_seqs, num_query_heads, qk_head_dim),
        dtype=torch.bfloat16,
        device=device,
    ).to(dtype)

    return query, kv_cache, block_tables, seq_lens


_KV_LORA_RANK = 512
_QK_ROPE_HEAD_DIM = 64
_NUM_HEADS = (16, 1)  # (num_query_heads, num_kv_heads)

# (num_seqs, ctx_len)
_CASES = [
    (1, 200, 1),
    (1, 200, 4),
    (1, 200, 8),
    (1, 600, 2),
    (1, 256, 2),
    (2, 400, 1),
]

_BLOCK_SIZES = [
    (16, 32),
    (16, 64),
    (64, 64),
    (64, 128),
    (128, 128),    
]


# ---------------------------------------------------------------------------
# Kernel correctness
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("num_seqs,ctx_len,num_segs", _CASES)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("varlen", [True, False])
@pytest.mark.parametrize("num_warps", [1, 2])
@pytest.mark.parametrize("block_size, compute_block_size", _BLOCK_SIZES)
def test_flydsl_mla_decode(num_seqs, ctx_len, num_segs, dtype, varlen, block_size, num_warps, compute_block_size):
    num_query_heads, num_kv_heads = _NUM_HEADS
    query, kv_cache, block_tables, seq_lens = _generate_inputs(
        num_seqs=num_seqs,
        num_query_heads=num_query_heads,
        num_kv_heads=num_kv_heads,
        kv_lora_rank=_KV_LORA_RANK,
        qk_rope_head_dim=_QK_ROPE_HEAD_DIM,
        block_size=block_size,
        ctx_len=ctx_len,
        dtype=dtype,
        varlen=varlen,
    )
    head_size = _KV_LORA_RANK + _QK_ROPE_HEAD_DIM
    attn_scale = 1.0 / math.sqrt(head_size)

    # Reference runs on the logical (unshuffled) cache.
    ref = _torch_mla_decode_ref(
        query, kv_cache, block_tables, seq_lens, attn_scale, _KV_LORA_RANK
    )

    # Kernel only consumes the pre-shuffled cache.
    kernel_kv_cache = shuffle_kv_buffer(kv_cache, _KV_LORA_RANK)

    output = torch.zeros(
        (num_seqs, num_query_heads, _KV_LORA_RANK), dtype=dtype, device=query.device
    )
    flydsl_mla_decode(
        output,
        query,
        kernel_kv_cache,
        block_tables,
        seq_lens,
        attn_scale,
        kv_lora_rank=_KV_LORA_RANK,
        qk_rope_head_dim=_QK_ROPE_HEAD_DIM,
        num_segs=num_segs,
        num_warps=num_warps,
        kv_compute_block_size=compute_block_size,
    )

    assert not torch.isnan(output).any(), "output contains NaN"
    torch.testing.assert_close(output, ref, atol=1.5e-2, rtol=1e-2)
