# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL and SGLang-native Triton QSA sparse attention (prefill) vs a torch reference.

Row ``t`` of request ``r`` attends only to its selected tokens::

    out[t, h] = softmax(scale * q[t, h] @ K_r[sel_t].T) @ V_r[sel_t]

``sel_t`` is the indexer's selection: up to T complete four-token blocks of the
row's causal window, then its 0..3 causal tail tokens, then -1 padding, in an
int32 [rows, 4T + 3] tensor (T = 512 unless the case name starts with
``topk``). Q is BF16 [rows, H, 256] and K/V are the requests' packed BF16
[prefix + new tokens, HK, 256].

Groups (Q heads per KV head) follow Qwen3.8-Flash-Next per TP rank (12/6/3 for
TP2/4/8, where HK = 1 and H = group); HK = 2/16 cases repeat those groups over
more KV heads. Functional cases cover one row, a short extend and chunked
prefill against long cached contexts (one whose packed KV exceeds the direct
kernel's 64 MiB budget), ragged batches with an empty request, HK > 1 and
T = 256/768/1024 selections. Performance cases are single-request full
prefills of 1k-32k tokens plus the 12000-token serving prompt (T = 512); they
are checked as well and are the only cases using the full-prefill BQ21/BQ42
union tiles. ``shared`` selections let 32-row windows share block priorities;
``random`` rows choose independently; ``correlated`` (performance cases)
simulates the real indexer's selection distribution, captured Qwen3.8 layers
whose neighbouring rows share about half of their blocks (see
correlated_prefill).

``triton`` is SGLang's native QSA extend path, copied below and called as its
backend does: sparse_gqa_fwd_interface_triton when no request has a prefix,
else sparse_gqa_fwd_interface_triton_ck over the packed full-context K/V.

Usage:
    python op_tests/test_flydsl_qsa_attention.py
    python op_tests/test_flydsl_qsa_attention.py -c prefill_12000 -g 12 6 3
"""

import argparse
import itertools
import math

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import triton
import triton.language as tl

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl.kernels.qsa.attention import qsa_sparse_attention_flydsl
from aiter.test_common import benchmark, checkAllclose, run_perftest

torch.set_default_device("cuda")

SUPPORTED_GFX = ["gfx942"]
HEAD_DIM = 256
SCALE = HEAD_DIM**-0.5

# name -> (query_lens, prefix_lens, kv_heads, selection, blocks per row T)
# Functional coverage: request layouts, selections, KV heads, T, and the packed
# direct, raw direct and union paths.
FUNCTIONAL_CASES = {
    "one_row": ((1,), (0,), 1, "shared", 512),
    "extend_17_ctx_30017": ((17,), (30000,), 1, "random", 512),
    "chunk_1024_ctx_31024_random": ((1024,), (30000,), 1, "random", 512),
    "ragged_4": ((300, 129, 1001, 63), (0, 517, 4093, 12345), 1, "shared", 512),
    "chunk_1024_ctx_132096": ((1024,), (131072,), 1, "shared", 512),
    "hk2_ragged": ((7, 0, 9), (0, 5, 3000), 2, "shared", 512),
    "hk16_chunk_1024_ctx_9216": ((1024,), (8192,), 16, "random", 512),
    "topk256_chunk_1024_ctx_31024": ((1024,), (30000,), 1, "shared", 256),
    "topk768_ragged_4": ((300, 129, 1001, 63), (0, 517, 4093, 12345), 1, "shared", 768),
    "topk1024_chunk_1024_ctx_132096": ((1024,), (131072,), 1, "shared", 1024),
}
# Performance: one full prefill of 1k-32k tokens, plus the 12000-token serving prompt.
PERF_CASES = {
    f"prefill_{tokens}": ((tokens,), (0,), 1, "correlated", 512)
    for tokens in (1024, 2048, 4096, 8192, 12000, 16384, 32768)
}
CASES = {**FUNCTIONAL_CASES, **PERF_CASES}

# Sharing levels of real Qwen3.8 indexer selections at layers 3 and 47.
CORRELATED_PROFILES = (
    {
        "sharing_fraction": 0.25,
        "high_shared_blocks": 480,
        "low_shared_blocks": 64,
        "high_window_blocks": 768,
        "low_window_blocks": 4096,
    },
    {
        "sharing_fraction": 0.29,
        "high_shared_blocks": 496,
        "low_shared_blocks": 96,
        "high_window_blocks": 640,
        "low_window_blocks": 4096,
    },
)


def correlated_prefill(rows, seed=20260929):
    """Selections simulating the real indexer distribution, one prefill, T = 512.

    Mimics captured Qwen3.8 prefill selections: the first 2051 rows select their
    whole causal prefix. Later rows form 128-row windows, alternating the two
    profiles; a seeded, balanced ~27% of windows share a core of 480/496 blocks
    drawn from the last 768/640 visible blocks, the others a core of 64/96 from
    the last 4096. Each row adds private blocks from that span up to 512,
    shuffled (unsorted, as the indexer emits them), then its causal tail.
    """
    rng = np.random.default_rng(seed)
    fraction = sum(p["sharing_fraction"] for p in CORRELATED_PROFILES) / len(
        CORRELATED_PROFILES
    )
    phase = float(rng.random())
    indices = np.full((rows, 2051), -1, dtype=np.int32)
    dense = min(rows, 2051)
    for row in range(dense):
        indices[row, : row + 1] = np.arange(row + 1)
    for window, first in enumerate(range(dense, rows, 128)):
        high = math.floor((window + 1) * fraction + phase) > math.floor(
            window * fraction + phase
        )
        settings = CORRELATED_PROFILES[window % len(CORRELATED_PROFILES)]
        sharing = "high" if high else "low"
        shared = settings[f"{sharing}_shared_blocks"]
        span = settings[f"{sharing}_window_blocks"]
        available = (first + 1) // 4
        start = max(0, available - span)
        common = rng.choice(np.arange(start, available), shared, replace=False)
        for row in range(first, min(first + 128, rows)):
            visible = row + 1
            blocks = visible // 4
            pool = np.arange(start, blocks)
            private = pool[~np.isin(pool, common)]
            chosen = np.concatenate(
                (common, rng.choice(private, 512 - shared, replace=False))
            )
            rng.shuffle(chosen)
            indices[row, :2048] = (chosen[:, None] * 4 + np.arange(4)).reshape(-1)
            indices[row, 2048 : 2048 + visible % 4] = np.arange(blocks * 4, visible)
    return torch.from_numpy(indices).cuda()


def make_indices(query_lens, prefix_lens, selection, generator, topk):
    """Indexer-like selections of the packed requests.

    Inputs: per-request query rows and cached prefix tokens (host ints), the
    selection mode ("shared": 32-row windows share block priorities, "random":
    per row, "correlated": correlated_prefill, one prefill with T = 512 and its own
    seed), a CUDA generator, and T = topk blocks per row.

    Returns contiguous int32 [sum(query_lens), 4T + 3], one row per query in packed
    order; requests without queries add no rows. Row values are request-local token
    IDs. For a row whose causal window holds v tokens (0 .. v - 1, the last being
    the row's own token) and b = v // 4 complete four-token blocks:
      - columns [0, 4 * min(b, T)) hold min(b, T) distinct complete blocks, each as
        its tokens 4j .. 4j + 3: every block when b <= T, else T chosen blocks;
        block IDs ascend, except that "correlated" keeps them shuffled;
      - the next v % 4 columns hold the causal tail tokens 4b .. v - 1;
      - all remaining columns are -1.
    """
    if selection == "correlated":
        if len(query_lens) != 1 or prefix_lens[0] or topk != 512:
            raise ValueError("correlated selections model one prefill with T = 512")
        return correlated_prefill(query_lens[0])
    rows = []
    for count, prefix in zip(query_lens, prefix_lens):
        if count == 0:
            continue
        visible = torch.arange(prefix + 1, prefix + count + 1)
        blocks = visible // 4
        width = max(int(blocks.max()), 1)
        if selection == "shared":
            priority = torch.rand((count + 31) // 32, width, generator=generator)
            priority = priority.repeat_interleave(32, 0)[:count]
        else:
            priority = torch.rand(count, width, generator=generator)
        priority[torch.arange(width) >= blocks[:, None]] = -1
        kept = min(topk, width)
        score, chosen = priority.topk(kept, dim=1)
        chosen = torch.where(score >= 0, chosen, 1 << 30).sort(dim=1).values
        complete = blocks.clamp(max=topk)
        index = torch.full((count, 4 * topk + 3), -1, dtype=dtypes.i32)
        tokens = (chosen[:, :, None] * 4 + torch.arange(4)).reshape(count, -1)
        valid = torch.arange(4 * kept) < 4 * complete[:, None]
        index[:, : 4 * kept] = torch.where(valid, tokens, -1)
        tail = torch.arange(3)
        rows_with_tail = tail < (visible % 4)[:, None]
        position = 4 * complete[:, None] + tail
        value = (blocks * 4)[:, None] + tail
        row_ids = torch.arange(count)[:, None].expand(-1, 3)
        index[row_ids[rows_with_tail], position[rows_with_tail]] = value[
            rows_with_tail
        ].to(dtypes.i32)
        rows.append(index)
    return torch.cat(rows).contiguous()


def run_torch(q, k, v, indices, query_lens, prefix_lens, scale):
    """FP32 attention over each row's selected tokens; padding entries (-1) are masked."""
    rows, heads, dim = q.shape
    kv_heads = k.shape[1]
    # About 1 GiB of gathered FP32 keys per chunk of rows.
    chunk = max(1, 2**28 // (indices.shape[1] * kv_heads * dim))
    q_lens = torch.tensor(query_lens)
    kv_lens = q_lens + torch.tensor(prefix_lens)
    kv_start = torch.repeat_interleave(torch.cumsum(kv_lens, 0) - kv_lens, q_lens)
    out = torch.empty(rows, heads, dim, dtype=dtypes.fp32)
    for start in range(0, rows, chunk):
        end = min(start + chunk, rows)
        selected = indices[start:end].long()
        slots = selected.clamp(min=0) + kv_start[start:end, None]
        keys, values = k[slots].float(), v[slots].float()
        query = q[start:end].float().view(end - start, kv_heads, -1, dim)
        scores = torch.einsum("rhgd,rthd->rhgt", query, keys) * scale
        scores.masked_fill_((selected < 0)[:, None, None, :], float("-inf"))
        out[start:end] = torch.einsum(
            "rhgt,rthd->rhgd", scores.softmax(-1), values
        ).reshape(end - start, heads, dim)
    return out


# Copied unchanged from SGLang:
# https://github.com/sgl-project/sglang/blob/f2b73d4f9b4e64415b8d775b0f811c7801f1728b/python/sglang/srt/layers/attention/qsa/sparse_attn.py#L9-L267
_H20_CONFIGS = [
    (32, (32, 8, 2)),
    (64, (64, 8, 2)),
    (1024, (32, 4, 2)),
    (float("inf"), (16, 1, 2)),
]
_L20_CONFIGS = [
    (32, (32, 8, 2)),
    (64, (64, 8, 2)),
    (128, (64, 4, 2)),
    (512, (32, 4, 2)),
    (float("inf"), (16, 1, 2)),
]


def _get_best_config(total_q: int):
    table = _H20_CONFIGS if "H20" in torch.cuda.get_device_name(0) else _L20_CONFIGS
    return next(cfg for limit, cfg in table if total_q <= limit)


@triton.jit
def _sparse_gqa_prefill(
    q,
    k,
    v,
    out,
    indices,
    cu_seqlens,
    scale,
    topk,
    sq_m: tl.constexpr,
    sq_h: tl.constexpr,
    sq_d: tl.constexpr,
    sk_n: tl.constexpr,
    sk_h: tl.constexpr,
    sk_d: tl.constexpr,
    sv_n: tl.constexpr,
    sv_h: tl.constexpr,
    sv_d: tl.constexpr,
    so_m: tl.constexpr,
    so_h: tl.constexpr,
    so_d: tl.constexpr,
    si_m: tl.constexpr,
    si_g: tl.constexpr,
    si_n: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    batch_group = tl.program_id(1)
    group = batch_group % NUM_KV_HEADS
    batch = batch_group // NUM_KV_HEADS
    seq_start = tl.load(cu_seqlens + batch).to(tl.int64)
    seq_end = tl.load(cu_seqlens + batch + 1).to(tl.int64)
    query_relative = tl.program_id(0).to(tl.int64)
    query = seq_start + query_relative
    if query >= seq_end:
        return

    row_topk = tl.minimum(topk, query_relative + 1)
    row_limit = tl.minimum(topk, ((row_topk + BLOCK_N - 1) // BLOCK_N) * BLOCK_N)
    offs_h = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    head_start = group * GROUP_SIZE
    q_values = tl.load(
        q
        + query * sq_m
        + (head_start + offs_h[:, None]) * sq_h
        + offs_d[None, :] * sq_d,
        mask=(offs_h < GROUP_SIZE)[:, None],
        other=0.0,
    )
    q_values = (q_values * scale * 1.4426950408).to(q_values.dtype)
    k_base = k + seq_start * sk_n + group * sk_h
    v_base = v + seq_start * sv_n + group * sv_h
    idx_row = indices + query * si_m + group * si_g
    max_value = tl.full([BLOCK_M], -float("inf"), tl.float32)
    normalizer = tl.zeros([BLOCK_M], tl.float32)
    accumulator = tl.zeros([BLOCK_M, HEAD_DIM], tl.float32)
    offs_n = tl.arange(0, BLOCK_N)
    for start in range(0, row_limit, BLOCK_N):
        current = start + offs_n
        token = tl.load(idx_row + current * si_n, mask=current < topk, other=-1)
        valid = token >= 0
        keys = tl.load(
            k_base + token[None, :] * sk_n + offs_d[:, None] * sk_d,
            mask=valid[None, :],
            other=0.0,
        )
        values = tl.load(
            v_base + token[:, None] * sv_n + offs_d[None, :] * sv_d,
            mask=valid[:, None],
            other=0.0,
        )
        scores = tl.where(valid[None, :], tl.dot(q_values, keys), -float("inf"))
        next_max = tl.maximum(max_value, tl.max(scores, 1))
        alpha = tl.math.exp2(max_value - next_max)
        probabilities = tl.math.exp2(scores - next_max[:, None])
        accumulator = tl.dot(
            probabilities.to(values.dtype), values, accumulator * alpha[:, None]
        )
        normalizer = normalizer * alpha + tl.sum(probabilities, 1)
        max_value = next_max
    output = accumulator / normalizer[:, None]
    tl.store(
        out
        + query * so_m
        + (head_start + offs_h[:, None]) * so_h
        + offs_d[None, :] * so_d,
        output,
        mask=(offs_h < GROUP_SIZE)[:, None],
    )


def sparse_gqa_fwd_interface_triton(q, k, v, max_seqlen_k, indices, cu_seqlens, scale):
    total_q, num_q_heads, head_dim = q.shape
    num_kv_heads = k.shape[1]
    group_size = num_q_heads // num_kv_heads
    block_m = max(16, triton.next_power_of_2(group_size))
    block_n, warps, stages = _get_best_config(total_q)
    out = torch.empty_like(q)
    _sparse_gqa_prefill[(max_seqlen_k, (cu_seqlens.shape[0] - 1) * num_kv_heads)](
        q,
        k,
        v,
        out,
        indices,
        cu_seqlens,
        scale,
        indices.shape[-1],
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        indices.stride(0),
        indices.stride(1) if indices.ndim == 3 else 0,
        indices.stride(2) if indices.ndim == 3 else indices.stride(1),
        NUM_KV_HEADS=num_kv_heads,
        GROUP_SIZE=group_size,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        HEAD_DIM=head_dim,
        num_warps=warps,
        num_stages=stages,
    )
    return out


@triton.jit
def _sparse_gqa_chunk_prefill(
    q,
    k,
    v,
    out,
    indices,
    cu_q,
    cu_k,
    kv_lens,
    scale,
    topk,
    sq_m: tl.constexpr,
    sq_h: tl.constexpr,
    sq_d: tl.constexpr,
    sk_n: tl.constexpr,
    sk_h: tl.constexpr,
    sk_d: tl.constexpr,
    sv_n: tl.constexpr,
    sv_h: tl.constexpr,
    sv_d: tl.constexpr,
    so_m: tl.constexpr,
    so_h: tl.constexpr,
    so_d: tl.constexpr,
    si_m: tl.constexpr,
    si_g: tl.constexpr,
    si_n: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    query_relative = tl.program_id(0).to(tl.int64)
    batch_group = tl.program_id(1)
    group = batch_group % NUM_KV_HEADS
    batch = batch_group // NUM_KV_HEADS
    q_start = tl.load(cu_q + batch)
    q_end = tl.load(cu_q + batch + 1)
    query = (q_start + query_relative).to(tl.int64)
    if query >= q_end:
        return
    k_start = tl.load(cu_k + batch).to(tl.int64)
    kv_len = tl.load(kv_lens + batch).to(tl.int64)
    visible = query_relative + kv_len - (q_end - q_start) + 1
    row_topk = tl.minimum(topk, visible)
    row_limit = tl.minimum(topk, ((row_topk + BLOCK_N - 1) // BLOCK_N) * BLOCK_N)
    offs_h = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    q_values = tl.load(
        q
        + query * sq_m
        + (group * GROUP_SIZE + offs_h[:, None]) * sq_h
        + offs_d[None, :] * sq_d,
        mask=(offs_h < GROUP_SIZE)[:, None],
        other=0.0,
    )
    q_values = (q_values * scale * 1.4426950408).to(q_values.dtype)
    k_base = k + k_start * sk_n + group * sk_h
    v_base = v + k_start * sv_n + group * sv_h
    idx_row = indices + query * si_m + group * si_g
    max_value = tl.full([BLOCK_M], -float("inf"), tl.float32)
    normalizer = tl.zeros([BLOCK_M], tl.float32)
    accumulator = tl.zeros([BLOCK_M, HEAD_DIM], tl.float32)
    offs_n = tl.arange(0, BLOCK_N)
    for start in range(0, row_limit, BLOCK_N):
        current = start + offs_n
        token = tl.load(idx_row + current * si_n, mask=current < topk, other=-1)
        valid = token >= 0
        keys = tl.load(
            k_base + token[None, :] * sk_n + offs_d[:, None] * sk_d,
            mask=valid[None, :],
            other=0.0,
        )
        values = tl.load(
            v_base + token[:, None] * sv_n + offs_d[None, :] * sv_d,
            mask=valid[:, None],
            other=0.0,
        )
        scores = tl.where(valid[None, :], tl.dot(q_values, keys), -float("inf"))
        next_max = tl.maximum(max_value, tl.max(scores, 1))
        alpha = tl.math.exp2(max_value - next_max)
        probabilities = tl.math.exp2(scores - next_max[:, None])
        accumulator = tl.dot(
            probabilities.to(values.dtype), values, accumulator * alpha[:, None]
        )
        normalizer = normalizer * alpha + tl.sum(probabilities, 1)
        max_value = next_max
    output = accumulator / normalizer[:, None]
    tl.store(
        out
        + query * so_m
        + (group * GROUP_SIZE + offs_h[:, None]) * so_h
        + offs_d[None, :] * so_d,
        output,
        mask=(offs_h < GROUP_SIZE)[:, None],
    )


def sparse_gqa_fwd_interface_triton_ck(q, k, v, indices, cu_q, cu_k, kv_lens, scale):
    k, v = k.contiguous(), v.contiguous()
    total_q, num_q_heads, head_dim = q.shape
    num_kv_heads = k.shape[1]
    group_size = num_q_heads // num_kv_heads
    max_q = int((cu_q[1:] - cu_q[:-1]).max().item())
    block_m = max(16, triton.next_power_of_2(group_size))
    block_n, warps, stages = _get_best_config(total_q)
    out = torch.empty_like(q)
    _sparse_gqa_chunk_prefill[(max_q, (cu_q.shape[0] - 1) * num_kv_heads)](
        q,
        k,
        v,
        out,
        indices,
        cu_q,
        cu_k,
        kv_lens,
        scale,
        indices.shape[-1],
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        indices.stride(0),
        indices.stride(1) if indices.ndim == 3 else 0,
        indices.stride(2) if indices.ndim == 3 else indices.stride(1),
        NUM_KV_HEADS=num_kv_heads,
        GROUP_SIZE=group_size,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        HEAD_DIM=head_dim,
        num_warps=warps,
        num_stages=stages,
    )
    return out


def sglang_triton(q, k, v, indices, query_lens, prefix_lens):
    """(interface, args) as SGLang's qwen_sparse_attn_backend.forward_extend calls them."""
    lengths = [n + p for n, p in zip(query_lens, prefix_lens)]
    cu_q = F.pad(torch.tensor(query_lens, dtype=dtypes.i32).cumsum(0), (1, 0))
    if not any(prefix_lens):
        args = (q, k, v, max(lengths, default=1), indices, cu_q, SCALE)
        return sparse_gqa_fwd_interface_triton, args
    kv_lens = torch.tensor(lengths, dtype=dtypes.i32)
    cu_k = F.pad(kv_lens.cumsum(0), (1, 0))
    args = (q, k, v, indices, cu_q, cu_k, kv_lens, SCALE)
    return sparse_gqa_fwd_interface_triton_ck, args


@benchmark()
def test_qsa_sparse_attention(case, heads):
    query_lens, prefix_lens, kv_heads, selection, topk = CASES[case]
    generator = torch.Generator(device="cuda").manual_seed(
        list(CASES).index(case) * 16 + heads
    )
    rows, kv_tokens = sum(query_lens), sum(query_lens) + sum(prefix_lens)
    q = torch.randn(rows, heads, HEAD_DIM, generator=generator, dtype=dtypes.bf16)
    k = torch.randn(
        kv_tokens, kv_heads, HEAD_DIM, generator=generator, dtype=dtypes.bf16
    )
    v = torch.randn(
        kv_tokens, kv_heads, HEAD_DIM, generator=generator, dtype=dtypes.bf16
    )
    indices = make_indices(query_lens, prefix_lens, selection, generator, topk)

    # Both run as the SGLang QSA backend calls them; each allocates its output.
    candidates = {
        "flydsl": (
            qsa_sparse_attention_flydsl,
            (q, k, v, indices),
            {
                "query_lens": query_lens,
                "prefix_lens": prefix_lens,
                "softmax_scale": SCALE,
            },
        ),
        "triton": (*sglang_triton(q, k, v, indices, query_lens, prefix_lens), {}),
    }
    ref = run_torch(q, k, v, indices, query_lens, prefix_lens, SCALE)
    # QK^T and PV over the selected tokens; compulsory traffic reads Q/K/V/indices
    # and writes O once.
    flops = 4 * heads * HEAD_DIM * int((indices >= 0).sum().item())
    nbytes = 2 * q.nbytes + k.nbytes + v.nbytes + indices.nbytes
    ret = {
        "gfx": get_gfx(),
        "rows": rows,
        "kv_tokens": kv_tokens,
        "kv_heads": kv_heads,
        "topk": topk,
    }
    for name, (fn, args, kwargs) in candidates.items():
        out, us = run_perftest(fn, *args, **kwargs, num_rotate_args=10)
        err = checkAllclose(
            ref,
            out.to(dtypes.fp32),
            rtol=2e-2,
            atol=2e-2,
            msg=f"{case} H{heads} {name}: out",
        )
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = err
    # The SGLang kernels are only the baseline: FlyDSL alone must pass the check.
    assert ret["flydsl err"] == 0, f"{case} H{heads}: mismatch {ret['flydsl err']}"
    return ret


def main():
    if get_gfx() not in SUPPORTED_GFX:
        aiter.logger.warning(
            "qsa_sparse_attention_flydsl unsupported on %s; skipping", get_gfx()
        )
        return

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="config input of test",
    )
    parser.add_argument(
        "-c",
        "--case",
        type=str,
        nargs="*",
        choices=list(CASES),
        default=list(CASES),
        help="input layouts to run (functional coverage and performance cases)",
    )
    parser.add_argument(
        "-g",
        "--group",
        type=int,
        nargs="*",
        default=[12, 6, 3],
        help="Q heads per KV head (12/6/3 = Qwen3.8 flash next TP2/4/8, where HK = 1)",
    )
    args = parser.parse_args()

    for suite, cases in (("functional", FUNCTIONAL_CASES), ("performance", PERF_CASES)):
        selected = [case for case in args.case if case in cases]
        if not selected:
            continue
        df = [
            test_qsa_sparse_attention(case, group * CASES[case][2])
            for case, group in itertools.product(selected, args.group)
        ]
        aiter.logger.info(
            "qsa_sparse_attention_flydsl %s summary (markdown):\n%s",
            suite,
            pd.DataFrame(df).to_markdown(index=False),
        )


if __name__ == "__main__":
    main()
