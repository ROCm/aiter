# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL and SGLang-native Triton QSA sparse attention (prefill) vs a torch reference.

Row ``t`` of request ``r`` attends only to its selected tokens::

    out[t, h] = softmax(scale * q[t, h] @ K_r[sel_t].T) @ V_r[sel_t]

``sel_t`` is the indexer's selection: exactly min(v // 4, T) complete four-token
blocks of the row's v-token causal window, then its v % 4 causal tail tokens,
then -1 padding, in an int32 [rows, 4T + 3] tensor (T = 512 unless the case name
starts with ``topk``). Q is BF16 [rows, H, 256] and K/V are the requests' packed
BF16 [prefix + new tokens, HK, 256].

Groups (Q heads per KV head) follow Qwen3.8-Flash-Next per TP rank (12/6/3 for
TP2/4/8, where HK = 1 and H = group); HK = 2/16 cases repeat those groups over
more KV heads. Functional cases cover one row, a short extend and chunked
prefill against long cached contexts (one whose packed KV exceeds the direct
kernel's 64 MiB budget, one packed by the per-row rule, one only raw direct
reads), nine 1M-token requests whose packed K/V exceed 4 GiB (the last, a
1024-row chunk, starts at 4 GiB), requests whose K and V each span the
4 GiB - 1 MiB limit (raw direct, and union with gated raw direct at 768 rows on
gfx942 and 1536 rows on gfx950), ragged batches with an empty request, HK > 1
(HK = 16 with over 4096 tasks to route), T = 256/768/1024 selections and constant
Q/K whose logits sit near +1e11 on the union path (``large_logits``).
Each graph case captures a layout on its own stream, replays it after rewriting
its indices in place, then again after another layout grows the stream's scratch
arena, which must keep the captured scratch allocated. The
kv_slots cases scatter a layout's K/V into random 64-token pages of a NaN-filled
token pool (past 4 GiB for the ``pool_4g`` cases) and read it in place through
the slot table; the result must equal the packed call on the gathered rows bit
for bit. The span limit cases must be rejected: one request whose K/V spans one
token past 4 GiB - 1 MiB, or whose int32 slot table spans one entry past it.
On gfx950 the strided cases store the same layouts' K/V as interleaved K|V
records (one per token, as vLLM's QSA cache does) or as padded rows and read them
in place, packed and through kv_slots (past 4 GiB for the ``pool_4g`` cases); the
result must equal the call on the rows gathered into contiguous K/V bit for bit.
Elsewhere strided rows must be rejected. The UNION_LAYOUTS and RAW_DIRECT_LAYOUTS
cases (1536 rows or more) clear gfx950's 1536-row union gate (and gfx942's). The
summaries report the union tiles of each call (``union tiles``) and how its direct
side read K/V (``direct``); on gfx950, whose entry proposes union for every tile,
the UNION_LAYOUTS cases must run some, and the 1536-row request at the
4 GiB - 1 MiB limit must build a union plan whose direct side reads the raw K/V.
The route cases run that request with each route forced (AITER_QSA_PLATFORM):
every proposed tile on union (gfx950 proposes every tile), then none proposed (every
tile on gated raw direct). The max stride cases read a request through K/V rows
2**28 - 8 elements apart, the largest stride accepted: packed and through kv_slots
(past 4 GiB for ``wide``), on packed direct and on raw direct, and a 1536-row
request on union tiles (its blocks all read the same four pool rows). K/V rows
that are not 16-byte aligned, overlap, differ in stride between K and V, or sit
2**28 elements or more apart, K/V heads or dims that are not dense, a
non-contiguous Q, and an interleaved request whose rows span past 4 GiB - 1 MiB,
must be rejected.
Cases needing more device memory than is free are skipped with a warning.
Performance cases are single-request full prefills of 1k-32k tokens plus the
12000-token serving prompt (T = 512); they are checked as well and are the only
cases using the full-prefill BQ21/BQ42 union tiles. ``shared`` selections let
32-row windows share block priorities; ``random`` rows choose independently;
``correlated`` (performance cases) simulates the real indexer's selection
distribution, captured Qwen3.8 layers whose neighbouring rows share about half
of their blocks (see correlated_prefill).

``triton`` is SGLang's native QSA extend path, copied below and called as its
backend does: sparse_gqa_fwd_interface_triton when no request has a prefix,
else sparse_gqa_fwd_interface_triton_ck over the packed full-context K/V.

Usage:
    python op_tests/test_flydsl_qsa_sparse_prefill.py
    python op_tests/test_flydsl_qsa_sparse_prefill.py -c prefill_12000 -g 12 6 3
"""

import argparse
import contextlib
import itertools
import math
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import triton
import triton.language as tl

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx, get_gfx_runtime
from aiter.ops.flydsl import qsa_sparse_prefill as flydsl_prefill
from aiter.ops.flydsl.kernels.qsa import sparse_prefill_common
from aiter.ops.qsa import qsa_sparse_prefill
from aiter.test_common import benchmark, checkAllclose, run_perftest

torch.set_default_device("cuda")

SUPPORTED_GFX = ["gfx942", "gfx950"]
HEAD_DIM = 256
SCALE = HEAD_DIM**-0.5

# name -> (query_lens, prefix_lens, kv_heads, selection, blocks per row T)
FUNCTIONAL_CASES = {
    # Below both archs' union row thresholds: direct only (packed, raw, HK2 with an
    # empty request).
    "one_row": ((1,), (0,), 1, "shared", 512),
    "extend_17_ctx_30017": ((17,), (30000,), 1, "random", 512),
    "extend_64_ctx_70065": ((64,), (70001,), 1, "random", 512),
    "hk2_ragged": ((7, 0, 9), (0, 5, 3000), 2, "shared", 512),
    # On gfx942 (union from 384 rows): union + packed direct; random rows route every
    # tile direct, 4096 rows pack per row. Below gfx950's 1536 rows: direct only.
    "ragged_4": ((300, 129, 1001, 63), (0, 517, 4093, 12345), 1, "shared", 512),
    "chunk_1024_ctx_31024_random": ((1024,), (30000,), 1, "random", 512),
    "chunk_4096_ctx_131072": ((4096,), (126976,), 1, "shared", 512),
    # gfx942: union + gated raw direct over a windowed K2 (gfx950:
    # chunk_1536_ctx_8386560).
    "chunk_1024_ctx_132096": ((1024,), (131072,), 1, "shared", 512),
    # Packed K/V over 4 GiB: the 1024-row request starts at 4 GiB (union on gfx942, 4 K2
    # windows).
    "batch_9_ctx_1048576": (
        (2,) * 8 + (1024,),
        (2**20 - 2,) * 8 + (2**20 - 1024,),
        1,
        "shared",
        512,
    ),
    # One request at the 4 GiB - 1 MiB K/V limit, by raw direct and (gfx942) by union.
    "extend_64_ctx_8386560": (
        (64,),
        ((2**32 - 2**20) // (HEAD_DIM * 2) - 64,),
        1,
        "random",
        512,
    ),
    "chunk_768_ctx_8386560": (
        (768,),
        ((2**32 - 2**20) // (HEAD_DIM * 2) - 768,),
        1,
        "shared",
        512,
    ),
    # HK16 (over 4096 K3 tasks at H192/H96: two order CTAs) and T = 256/768/1024.
    "hk16_chunk_4096_ctx_12288": ((4096,), (8192,), 16, "random", 512),
    "topk256_chunk_1024_ctx_31024": ((1024,), (30000,), 1, "shared", 256),
    "topk768_ragged_4": ((300, 129, 1001, 63), (0, 517, 4093, 12345), 1, "shared", 768),
    "topk1024_chunk_1024_ctx_132096": ((1024,), (131072,), 1, "shared", 1024),
}
# name -> (shape as above, (Q, K)): constant Q/K and V = 1, so O = 1. Logits near
# +1e11 check union centering, which an FMA against the rounded row max overflowed.
CONSTANT_CASES = {
    "large_logits_prefill_768": (((768,), (0,), 1, "shared", 512), (65536.0, 66048.0)),
}
# Performance: one full prefill of 1k-32k tokens, plus the 12000-token serving prompt.
PERF_CASES = {
    f"prefill_{tokens}": ((tokens,), (0,), 1, "correlated", 512)
    for tokens in (1024, 2048, 4096, 8192, 12000, 16384, 32768)
}
# Layouts of the strided, kv_slots and graph cases only (last, so earlier cases keep
# their seeds): the 1024-row chunk takes union tiles on gfx942 only, the 2048-row chunks
# also clear gfx950's 1536-row union gate.
STRIDE_LAYOUTS = {
    "hk2_chunk_1024_ctx_8192": ((1024,), (7168,), 2, "shared", 512),
    "chunk_2048_ctx_30720": ((2048,), (28672,), 1, "shared", 512),
    "hk2_chunk_2048_ctx_8192": ((2048,), (6144,), 2, "shared", 512),
}
# Functional layouts over gfx950's 1536-row union gate (last, as above): ragged
# requests, and one request at the 4 GiB - 1 MiB K/V limit whose direct side reads the
# raw K/V.
UNION_CASES = {
    "ragged_4_rows_2986": (
        (600, 258, 2002, 126),
        (0, 517, 4093, 12345),
        1,
        "shared",
        512,
    ),
    "chunk_1536_ctx_8386560": (
        (1536,),
        ((2**32 - 2**20) // (HEAD_DIM * 2) - 1536,),
        1,
        "shared",
        512,
    ),
}
CASES = {
    **FUNCTIONAL_CASES,
    **PERF_CASES,
    **{name: shape for name, (shape, _) in CONSTANT_CASES.items()},
    **STRIDE_LAYOUTS,
    **UNION_CASES,
}
# Layouts whose calls must keep union tiles on gfx950, and those that must build a union
# plan there whose direct side reads the raw K/V (gated raw direct); gfx942's route is
# reported only.
UNION_LAYOUTS = {
    "chunk_2048_ctx_30720",
    "hk2_chunk_2048_ctx_8192",
    "ragged_4_rows_2986",
}
RAW_DIRECT_LAYOUTS = {"chunk_1536_ctx_8386560"}
# name -> (RAW_DIRECT_LAYOUTS layout, route forced through AITER_QSA_PLATFORM): every
# proposed tile kept on union (no union cost), or none proposed (every tile on gated raw
# direct).
FORCED_ROUTES = {"union": "union_fixed=0,union_step=0", "direct": "union_ratio=0"}
ROUTE_CASES = {
    f"route_{route}_{layout}": (layout, route)
    for layout in sorted(RAW_DIRECT_LAYOUTS)
    for route in FORCED_ROUTES
}
# name -> (captured layout, layout whose first call then grows the scratch arena: its
# packed KV outgrows the captured layout's)
GRAPH_CASES = {
    "graph_ragged_4": ("ragged_4", "chunk_1024_ctx_31024_random"),
    "graph_chunk_2048_ctx_30720": ("chunk_2048_ctx_30720", "chunk_4096_ctx_131072"),
}
# name -> (layout, pool rows or None for twice the pages needed, first pool page)
PAGE = 64
POOL_4G_ROWS = 9_000_000 // PAGE * PAGE
SLOT_CASES = {
    "kv_slots_ragged_4": ("ragged_4", None, 1),
    "kv_slots_hk2_ragged": ("hk2_ragged", None, 1),
    "kv_slots_extend_64_ctx_70065": ("extend_64_ctx_70065", None, 1),
    "kv_slots_topk1024_chunk_1024_ctx_132096": (
        "topk1024_chunk_1024_ctx_132096",
        None,
        1,
    ),
    "kv_slots_pool_4g_ragged_4": ("ragged_4", POOL_4G_ROWS, 8_000_000 // PAGE),
    "kv_slots_pool_4g_extend_64_ctx_70065": (
        "extend_64_ctx_70065",
        POOL_4G_ROWS,
        8_000_000 // PAGE,
    ),
    "kv_slots_pool_4g_chunk_1024_ctx_132096": (
        "chunk_1024_ctx_132096",
        POOL_4G_ROWS,
        8_000_000 // PAGE,
    ),
    "kv_slots_chunk_2048_ctx_30720": ("chunk_2048_ctx_30720", None, 1),
    "kv_slots_pool_4g_chunk_2048_ctx_30720": (
        "chunk_2048_ctx_30720",
        POOL_4G_ROWS,
        8_000_000 // PAGE,
    ),
}
# name -> (K/V tokens of one request, read through kv_slots): one past the
# 4 GiB - 1 MiB byte span of the request's K/V, or of the int32 slot table
SPAN_LIMIT_CASES = {
    "span_over_limit": ((2**32 - 2**20) // (HEAD_DIM * 2) + 1, False),
    "kv_slots_span_over_limit": ((2**32 - 2**20) // 4 + 1, True),
}
# name -> (layout, padding elements after each K/V row (None: one interleaved K|V record
# per token, as vLLM's QSA cache stores them), pool rows (None: packed K/V; 0: twice the
# pages needed), first pool page)
POOL_4G_RECORDS = 4_500_000 // PAGE * PAGE  # 1 KiB HK1 records: a 4.29 GiB pool
FIRST_PAGE_4G = 4_200_000 // PAGE  # pages from here on start past 4 GiB
STRIDE_CASES = {
    "interleaved_ragged_4": ("ragged_4", None, None, 0),
    "interleaved_extend_64_ctx_70065": ("extend_64_ctx_70065", None, None, 0),
    "interleaved_hk2_ragged": ("hk2_ragged", None, None, 0),
    "interleaved_hk2_chunk_1024_ctx_8192": ("hk2_chunk_1024_ctx_8192", None, None, 0),
    "padded_ragged_4": ("ragged_4", 8, None, 0),
    "kv_slots_interleaved_ragged_4": ("ragged_4", None, 0, 1),
    "kv_slots_interleaved_hk2_ragged": ("hk2_ragged", None, 0, 1),
    "kv_slots_interleaved_hk2_chunk_1024_ctx_8192": (
        "hk2_chunk_1024_ctx_8192",
        None,
        0,
        1,
    ),
    "kv_slots_padded_extend_64_ctx_70065": ("extend_64_ctx_70065", 8, 0, 1),
    "kv_slots_interleaved_pool_4g_extend_64_ctx_70065": (
        "extend_64_ctx_70065",
        None,
        POOL_4G_RECORDS,
        FIRST_PAGE_4G,
    ),
    "kv_slots_interleaved_pool_4g_chunk_1024_ctx_132096": (
        "chunk_1024_ctx_132096",
        None,
        POOL_4G_RECORDS,
        FIRST_PAGE_4G,
    ),
    "interleaved_chunk_2048_ctx_30720": ("chunk_2048_ctx_30720", None, None, 0),
    "padded_chunk_2048_ctx_30720": ("chunk_2048_ctx_30720", 8, None, 0),
    "interleaved_hk2_chunk_2048_ctx_8192": ("hk2_chunk_2048_ctx_8192", None, None, 0),
    "kv_slots_interleaved_chunk_2048_ctx_30720": ("chunk_2048_ctx_30720", None, 0, 1),
    "kv_slots_interleaved_hk2_chunk_2048_ctx_8192": (
        "hk2_chunk_2048_ctx_8192",
        None,
        0,
        1,
    ),
    "kv_slots_interleaved_pool_4g_chunk_2048_ctx_30720": (
        "chunk_2048_ctx_30720",
        None,
        POOL_4G_RECORDS,
        FIRST_PAGE_4G,
    ),
    "interleaved_ragged_4_rows_2986": ("ragged_4_rows_2986", None, None, 0),
    "kv_slots_interleaved_ragged_4_rows_2986": ("ragged_4_rows_2986", None, 0, 1),
}
# The wrapper's bound on the K/V row stride (elements): strides below it are accepted.
KV_STRIDE_LIMIT = 2**28
# name -> (tokens of one request, KV heads, K and V (row, head, dim) strides in
# elements, Q's element step along dims (2: not contiguous), the rejection's message):
# K/V or Q the kernels cannot read, and interleaved rows whose span passes
# 4 GiB - 1 MiB although the request's K values alone would fit. The K/V messages are
# the row check's own, so that off gfx950 the strided-row rejection cannot stand in.
DENSE = (HEAD_DIM, HEAD_DIM, 1)
ROWS = "dense [heads, 256]"
STRIDE_REJECT_CASES = {
    "stride_over_limit": (
        2,
        1,
        (KV_STRIDE_LIMIT, HEAD_DIM, 1),
        (KV_STRIDE_LIMIT, HEAD_DIM, 1),
        1,
        ROWS,
    ),
    "stride_not_16_byte_aligned": (
        4 * 512 + 3,
        1,
        (HEAD_DIM + 4, HEAD_DIM, 1),
        (HEAD_DIM + 4, HEAD_DIM, 1),
        1,
        ROWS,
    ),
    "stride_below_row": (
        4 * 512 + 3,
        1,
        (HEAD_DIM - 8, HEAD_DIM, 1),
        (HEAD_DIM - 8, HEAD_DIM, 1),
        1,
        ROWS,
    ),
    "k_v_strides_differ": (
        4 * 512 + 3,
        1,
        (2 * HEAD_DIM, HEAD_DIM, 1),
        (3 * HEAD_DIM, HEAD_DIM, 1),
        1,
        ROWS,
    ),
    # Per-head K|V records: each head's values 512 elements apart.
    "heads_not_dense": (
        4 * 512 + 3,
        2,
        (4 * HEAD_DIM, 2 * HEAD_DIM, 1),
        (4 * HEAD_DIM, 2 * HEAD_DIM, 1),
        1,
        ROWS,
    ),
    "dims_strided": (
        4 * 512 + 3,
        1,
        (2 * HEAD_DIM, HEAD_DIM, 2),
        (2 * HEAD_DIM, HEAD_DIM, 2),
        1,
        ROWS,
    ),
    # Dense K, strided V; and two heads whose values overlap, rows HK x 256 apart.
    "v_strided_k_dense": (
        4 * 512 + 3,
        1,
        DENSE,
        (2 * HEAD_DIM, HEAD_DIM, 1),
        1,
        ROWS,
    ),
    "heads_overlap": (
        4 * 512 + 3,
        2,
        (2 * HEAD_DIM, HEAD_DIM // 2, 1),
        (2 * HEAD_DIM, HEAD_DIM // 2, 1),
        1,
        ROWS,
    ),
    "q_not_contiguous": (4 * 512 + 3, 1, DENSE, DENSE, 2, "Q must be contiguous"),
    "interleaved_span_over_limit": (
        (2**32 - 2**20) // (4 * HEAD_DIM) + 1,
        1,
        (2 * HEAD_DIM, HEAD_DIM, 1),
        (2 * HEAD_DIM, HEAD_DIM, 1),
        1,
        "byte span",
    ),
}
# name -> (K/V view rows KV_STRIDE_LIMIT - 8 elements apart, the first of the four view
# rows holding K/V, kv_slots, wide, query rows of the one request): a 2-row request over
# 4 tokens in the packed request's 4 rows (1.5 GiB); in pool rows 4-7 of a 3.5 GiB view;
# in pool rows 8-11 of a 5.5 GiB view, which raw direct reads with 64-bit (wide) loads;
# and a 1536-row prefill (gfx950's union gate) whose four-token blocks all read pool
# rows 4-7. These requests pack; RAW_DIRECT makes direct read the raw K/V instead.
MAX_STRIDE_CASES = {
    "max_stride": (4, 0, False, False, 2),
    "kv_slots_max_stride": (8, 4, True, False, 2),
    "kv_slots_wide_max_stride": (12, 8, True, True, 2),
    "kv_slots_union_max_stride": (8, 4, True, False, 1536),
}
RAW_DIRECT = "max_packed_kv_bytes=0,pack_tokens_per_row=0"

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


def make_inputs(case, heads, generator):
    """Q/K/V and indices of a case: random BF16, or the CONSTANT_CASES fills."""
    query_lens, prefix_lens, kv_heads, selection, topk = CASES[case]
    rows, kv_tokens = sum(query_lens), sum(query_lens) + sum(prefix_lens)
    q = torch.randn(rows, heads, HEAD_DIM, generator=generator, dtype=dtypes.bf16)
    k = torch.randn(
        kv_tokens, kv_heads, HEAD_DIM, generator=generator, dtype=dtypes.bf16
    )
    v = torch.randn(
        kv_tokens, kv_heads, HEAD_DIM, generator=generator, dtype=dtypes.bf16
    )
    if case in CONSTANT_CASES:
        q_value, k_value = CONSTANT_CASES[case][1]
        q.fill_(q_value)
        k.fill_(k_value)
        v.fill_(1.0)
    indices = make_indices(query_lens, prefix_lens, selection, generator, topk)
    return q, k, v, indices


@benchmark()
def test_qsa_sparse_attention(case, heads):
    query_lens, prefix_lens, kv_heads, _, topk = CASES[case]
    generator = torch.Generator(device="cuda").manual_seed(
        list(CASES).index(case) * 16 + heads
    )
    q, k, v, indices = make_inputs(case, heads, generator)
    rows, kv_tokens = q.shape[0], k.shape[0]

    # Both run as the SGLang QSA backend calls them; each allocates its output.
    candidates = {
        "flydsl": (
            qsa_sparse_prefill,
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
    # Rotated copies of the 1M-token contexts' K/V would take tens of GiB.
    rotate = 10 if k.nbytes < 2**30 else 1
    for name, (fn, args, kwargs) in candidates.items():
        out, us = run_perftest(fn, *args, **kwargs, num_rotate_args=rotate)
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
    check_union(ret, case, f"{case} H{heads}")
    return ret


def latest_workspace():
    """The plans of the latest call (the workspace cache keeps it last)."""
    return next(reversed(flydsl_prefill._workspaces.values()))


def direct_path(plan):
    """How the direct side of a call reads K/V: packed by the pack kernel, or the raw
    K/V (64-bit loads past 4 GiB: wide)."""
    if plan.packed_key is not None:
        return "packed"
    return "raw wide" if plan.wide else "raw"


def check_union(ret, layout, label):
    """Report the latest call's union tiles (the route of sparse_prefill_order_masks
    writes 1 per union tile to the union plan's active array) and direct path in ret. On
    gfx950 a UNION_LAYOUTS call must have run union tiles, and a RAW_DIRECT_LAYOUTS call
    must have had a union plan whose direct side reads the raw K/V."""
    workspace = latest_workspace()
    plan = workspace.union
    ret["union tiles"] = 0 if plan is None else int((plan.active != 0).sum().item())
    ret["direct"] = direct_path(workspace.direct)
    if get_gfx_runtime() != "gfx950":
        return
    if layout in UNION_LAYOUTS:
        assert ret["union tiles"] > 0, f"{label}: no union tile ran"
    if layout in RAW_DIRECT_LAYOUTS:
        assert workspace.direct.gated and ret["direct"].startswith(
            "raw"
        ), f"{label}: direct {ret['direct']}, gated {workspace.direct.gated}"


@contextlib.contextmanager
def platform_entry(override):
    """Calls inside run with AITER_QSA_PLATFORM = override. The entry is cached, and so
    are the plans built from it: both are dropped on entry and on exit."""
    previous = os.environ.get(sparse_prefill_common.OVERRIDE)

    def reset():
        sparse_prefill_common.platform.cache_clear()
        flydsl_prefill._workspaces.clear()

    os.environ[sparse_prefill_common.OVERRIDE] = override
    reset()
    try:
        yield
    finally:
        if previous is None:
            del os.environ[sparse_prefill_common.OVERRIDE]
        else:
            os.environ[sparse_prefill_common.OVERRIDE] = previous
        reset()


def rejected_off_gfx950(label, *args, **kwargs):
    """Strided K/V rows are validated on gfx950 only: elsewhere the call must raise."""
    try:
        qsa_sparse_prefill(*args, **kwargs)
        message = ""
    except ValueError as error:
        message = str(error)
    assert "gfx950 only" in message, f"{label}: strided K/V gave {message!r}"
    return {"gfx": get_gfx(), "rejected off gfx950": True}


def released(pointer):
    """Whether device address pointer is outside every active caching-allocator block
    (a segment returned to HIP no longer appears in the snapshot)."""
    for segment in torch.cuda.memory_snapshot():
        address = segment["address"]
        for block in segment["blocks"]:
            if address <= pointer < address + block["size"]:
                return block["state"] != "active_allocated"
            address += block["size"]
    return True


@benchmark()
def test_qsa_graph_replay(case, heads):
    """A graph-captured layout replayed with indices rewritten in place, before
    and after another layout's first call grows the stream's scratch arena."""
    captured, growth = GRAPH_CASES[case]
    query_lens, prefix_lens, _, selection, topk = CASES[captured]
    generator = torch.Generator(device="cuda").manual_seed(heads)
    q, k, v, first = make_inputs(captured, heads, generator)
    second = make_indices(query_lens, prefix_lens, selection, generator, topk)
    kwargs = {
        "query_lens": query_lens,
        "prefix_lens": prefix_lens,
        "softmax_scale": SCALE,
    }
    indices, out = first.clone(), torch.empty_like(q)
    stream, graph = torch.cuda.Stream(), torch.cuda.CUDAGraph()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        qsa_sparse_prefill(q, k, v, indices, out=out, **kwargs)
        with torch.cuda.graph(graph, stream=stream):
            qsa_sparse_prefill(q, k, v, indices, out=out, **kwargs)

    def replay(selected, label):
        with torch.cuda.stream(stream):
            indices.copy_(selected)
            out.fill_(float("nan"))
            graph.replay()
        stream.synchronize()
        return checkAllclose(
            run_torch(q, k, v, selected, query_lens, prefix_lens, SCALE),
            out.to(dtypes.fp32),
            rtol=2e-2,
            atol=2e-2,
            msg=f"{case} H{heads} {label}: out",
        )

    errors = [replay(second, "replay")]
    ret = {"gfx": get_gfx(), "rows": q.shape[0], "growth": growth}
    check_union(ret, captured, f"{case} H{heads} replay")
    arena = flydsl_prefill._arenas[(q.device, stream.cuda_stream)]
    generation, scratch = arena.generation, [b.data_ptr() for b in arena.buffers]
    growth_lens, growth_prefix = CASES[growth][:2]
    with torch.cuda.stream(stream):
        qsa_sparse_prefill(
            *make_inputs(growth, heads, generator),
            query_lens=growth_lens,
            prefix_lens=growth_prefix,
            softmax_scale=SCALE,
        )
    stream.synchronize()
    assert arena.generation > generation, f"{growth} did not grow the {case} arena"
    # A freed buffer may still hold the right bytes, so check the allocator directly.
    assert not any(map(released, scratch)), f"{growth} freed the captured scratch"
    errors.append(replay(first, "replay after arena growth"))
    assert max(errors) == 0, f"{case} H{heads}: mismatch {errors}"
    ret["flydsl graph err"] = max(errors)
    return ret


def make_slots(query_lens, prefix_lens, pool_rows, first_page, generator):
    """int32 slots placing each request's tokens on random PAGE-token pages from
    first_page on, and the pool's rows (pool_rows, or twice the pages needed if 0 or
    None)."""
    lengths = [n + p for n, p in zip(query_lens, prefix_lens)]
    counts = [-(-n // PAGE) for n in lengths]
    pool_rows = pool_rows or (2 * sum(counts) + first_page) * PAGE
    pages = first_page + torch.randperm(
        pool_rows // PAGE - first_page, generator=generator, device="cuda"
    )
    slots, first = [], 0
    for n, count in zip(lengths, counts):
        tokens = torch.arange(n)
        slots.append(
            pages[first : first + count][tokens // PAGE] * PAGE + tokens % PAGE
        )
        first += count
    return torch.cat(slots).to(dtypes.i32), pool_rows


def make_pool(k, v, query_lens, prefix_lens, pool_rows, first_page, generator):
    """K/V token pools and int32 slots: each request's packed K/V rows scattered to
    random PAGE-token pages from first_page on; every other pool row is NaN."""
    slots, pool_rows = make_slots(
        query_lens, prefix_lens, pool_rows, first_page, generator
    )
    pools = []
    for packed in (k, v):
        pool = torch.full(
            (pool_rows, *packed.shape[1:]), float("nan"), dtype=packed.dtype
        )
        pool[slots.long()] = packed
        pools.append(pool)
    return *pools, slots


@benchmark()
def test_qsa_kv_slots(case, heads):
    """K/V read in place from a token pool through kv_slots, against the packed call on
    the gathered rows (what a caller without kv_slots runs); both match bit for bit."""
    layout, pool_rows, first_page = SLOT_CASES[case]
    query_lens, prefix_lens, kv_heads, *_ = CASES[layout]
    generator = torch.Generator(device="cuda").manual_seed(
        list(CASES).index(layout) * 16 + heads
    )
    q, k, v, indices = make_inputs(layout, heads, generator)
    ref = run_torch(q, k, v, indices, query_lens, prefix_lens, SCALE)
    pool_k, pool_v, slots = make_pool(
        k, v, query_lens, prefix_lens, pool_rows, first_page, generator
    )
    del k, v
    kwargs = {
        "query_lens": query_lens,
        "prefix_lens": prefix_lens,
        "softmax_scale": SCALE,
    }

    def gather(q, pool_k, pool_v, indices, slots):
        rows = slots.long()
        return qsa_sparse_prefill(q, pool_k[rows], pool_v[rows], indices, **kwargs)

    candidates = {
        "flydsl_gather": (gather, (q, pool_k, pool_v, indices, slots), {}),
        "flydsl_kv_slots": (
            qsa_sparse_prefill,
            (q, pool_k, pool_v, indices),
            {**kwargs, "kv_slots": slots},
        ),
    }
    flops = 4 * heads * HEAD_DIM * int((indices >= 0).sum().item())
    kv_bytes = slots.numel() * pool_k.stride(0) * pool_k.element_size()
    nbytes = 2 * q.nbytes + 2 * kv_bytes + indices.nbytes + slots.nbytes
    ret = {
        "gfx": get_gfx(),
        "rows": q.shape[0],
        "kv_tokens": slots.numel(),
        "kv_heads": kv_heads,
        "pool GiB": pool_k.nbytes / 2**30,
    }
    outputs = {}
    for name, (fn, args, kw) in candidates.items():
        out, us = run_perftest(
            fn, *args, **kw, num_rotate_args=10 if pool_k.nbytes < 2**30 else 1
        )
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = checkAllclose(
            ref,
            out.to(dtypes.fp32),
            rtol=2e-2,
            atol=2e-2,
            msg=f"{case} H{heads} {name}: out",
        )
        outputs[name] = out
    assert ret["flydsl_kv_slots err"] == 0, f"{case} H{heads}: kv_slots mismatch"
    check_union(ret, layout, f"{case} H{heads}")
    bitwise = checkAllclose(
        outputs["flydsl_gather"],
        outputs["flydsl_kv_slots"],
        rtol=0,
        atol=0,
        msg=f"{case} H{heads} flydsl_kv_slots vs flydsl_gather: out",
    )
    assert bitwise == 0, f"{case} H{heads}: kv_slots differs from the gathered call"
    return ret


def make_rows(rows, kv_heads, padding):
    """NaN-filled K and V views [rows, kv_heads, 256] with strided rows: the halves of
    one interleaved K|V record per row (padding None), or rows of two buffers with
    padding elements after their values."""
    if padding is None:
        records = torch.full(
            (rows, 2, kv_heads, HEAD_DIM), float("nan"), dtype=dtypes.bf16
        )
        return records[:, 0], records[:, 1]
    width = kv_heads * HEAD_DIM
    return tuple(
        torch.full((rows, width + padding), float("nan"), dtype=dtypes.bf16)[
            :, :width
        ].unflatten(1, (kv_heads, HEAD_DIM))
        for _ in range(2)
    )


@benchmark()
def test_qsa_strided_kv(case, heads):
    """K/V read in place through strided row views (interleaved K|V records or padded
    rows), packed or through kv_slots, against the call on the same rows gathered into
    contiguous K/V; both match bit for bit."""
    layout, padding, pool_rows, first_page = STRIDE_CASES[case]
    query_lens, prefix_lens, kv_heads, *_ = CASES[layout]
    generator = torch.Generator(device="cuda").manual_seed(
        list(CASES).index(layout) * 16 + heads
    )
    q, k, v, indices = make_inputs(layout, heads, generator)
    ref = None
    if get_gfx_runtime() == "gfx950":
        ref = run_torch(q, k, v, indices, query_lens, prefix_lens, SCALE)
    if pool_rows is None:
        slots, pool_rows = None, k.shape[0]
        rows = torch.arange(pool_rows)
    else:
        slots, pool_rows = make_slots(
            query_lens, prefix_lens, pool_rows, first_page, generator
        )
        rows = slots.long()
    strided_k, strided_v = make_rows(pool_rows, kv_heads, padding)
    strided_k[rows], strided_v[rows] = k, v
    del k, v
    kwargs = {
        "query_lens": query_lens,
        "prefix_lens": prefix_lens,
        "softmax_scale": SCALE,
    }
    if get_gfx_runtime() != "gfx950":
        return rejected_off_gfx950(
            f"{case} H{heads}",
            q,
            strided_k,
            strided_v,
            indices,
            **kwargs,
            kv_slots=slots,
        )

    def gather(q, k, v, indices):
        return qsa_sparse_prefill(q, k[rows], v[rows], indices, **kwargs)

    candidates = {
        "flydsl_gather": (gather, (q, strided_k, strided_v, indices), {}),
        "flydsl_strided": (
            qsa_sparse_prefill,
            (q, strided_k, strided_v, indices),
            {**kwargs, "kv_slots": slots},
        ),
    }
    flops = 4 * heads * HEAD_DIM * int((indices >= 0).sum().item())
    # Compulsory traffic, the same for both candidates: Q in, O out, the requests' K and
    # V values once (not padding, the interleaved other half, or the gather's copy),
    # indices and slots.
    nbytes = 2 * q.nbytes + 2 * rows.numel() * kv_heads * HEAD_DIM * 2
    nbytes += indices.nbytes + (0 if slots is None else slots.nbytes)
    span = ((pool_rows - 1) * strided_k.stride(0) + kv_heads * HEAD_DIM) * 2
    ret = {
        "gfx": get_gfx(),
        "rows": q.shape[0],
        "kv_tokens": rows.numel(),
        "kv_heads": kv_heads,
        "row stride": strided_k.stride(0),
        "K span GiB": span / 2**30,
    }
    outputs = {}
    for name, (fn, args, kw) in candidates.items():
        out, us = run_perftest(
            fn, *args, **kw, num_rotate_args=10 if span < 2**30 else 1
        )
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = checkAllclose(
            ref,
            out.to(dtypes.fp32),
            rtol=2e-2,
            atol=2e-2,
            msg=f"{case} H{heads} {name}: out",
        )
        outputs[name] = out
    assert ret["flydsl_strided err"] == 0, f"{case} H{heads}: strided K/V mismatch"
    check_union(ret, layout, f"{case} H{heads}")
    bitwise = checkAllclose(
        outputs["flydsl_gather"],
        outputs["flydsl_strided"],
        rtol=0,
        atol=0,
        msg=f"{case} H{heads} flydsl_strided vs flydsl_gather: out",
    )
    assert bitwise == 0, f"{case} H{heads}: strided K/V differ from the gathered call"
    return ret


def last_row_selection(tokens):
    """A valid T = 512 selection of the last row of a tokens-token request (its first
    min(tokens // 4, 512) blocks, then the causal tail), so that a missing check fails
    the rejection's assertion rather than faulting in a kernel."""
    blocks = min(tokens // 4, 512)
    indices = torch.full((1, 4 * 512 + 3), -1, dtype=dtypes.i32)
    indices[0, : 4 * blocks] = torch.arange(4 * blocks)
    indices[0, 4 * blocks : 4 * blocks + tokens % 4] = torch.arange(
        tokens // 4 * 4, tokens
    )
    return indices


@benchmark()
def test_qsa_span_limit(case, heads):
    """A request whose K/V or slot table spans more than 4 GiB - 1 MiB is rejected
    before any launch."""
    tokens, slotted = SPAN_LIMIT_CASES[case]
    q = torch.zeros(1, heads, HEAD_DIM, dtype=dtypes.bf16)
    kv = torch.empty(PAGE if slotted else tokens, 1, HEAD_DIM, dtype=dtypes.bf16)
    slots = torch.zeros(tokens, dtype=dtypes.i32) if slotted else None
    span = (kv if slots is None else slots).nbytes
    indices = last_row_selection(tokens)
    try:
        qsa_sparse_prefill(
            q,
            kv,
            kv,
            indices,
            query_lens=(1,),
            prefix_lens=(tokens - 1,),
            kv_slots=slots,
        )
        rejected = False
    except ValueError as error:
        message = str(error)
        rejected = "byte span" in message and ("kv_slots" in message) == slotted
    assert rejected, f"{case} H{heads}: a {span}-byte span was accepted"
    return {
        "gfx": get_gfx(),
        "kv_tokens": tokens,
        "kv_slots": slotted,
        "span bytes": span,
        "limit bytes": 2**32 - 2**20,
        "rejected": rejected,
    }


@benchmark()
def test_qsa_stride_rejected(case, heads):
    """K/V the kernels cannot read (rows not 16-byte aligned, overlapping, with K and V
    strides differing or KV_STRIDE_LIMIT elements or more apart; heads not dense; a
    strided last dim), a non-contiguous Q, or interleaved rows spanning past
    4 GiB - 1 MiB, are rejected before any launch."""
    tokens, kv_heads, k_strides, v_strides, q_step, expected = STRIDE_REJECT_CASES[case]
    q = torch.zeros(1, heads * kv_heads, HEAD_DIM * q_step, dtype=dtypes.bf16)
    q = q[..., ::q_step]
    k, v = (
        torch.empty(
            sum(
                (n - 1) * step for n, step in zip((tokens, kv_heads, HEAD_DIM), strides)
            )
            + 1,
            dtype=dtypes.bf16,
        ).as_strided((tokens, kv_heads, HEAD_DIM), strides)
        for strides in (k_strides, v_strides)
    )
    indices = last_row_selection(tokens)
    try:
        qsa_sparse_prefill(q, k, v, indices, query_lens=(1,), prefix_lens=(tokens - 1,))
        message = ""
    except ValueError as error:
        message = str(error)
    rejected = expected in message
    strides = f"K/V {k.stride()}/{v.stride()}, Q {q.stride()}"
    assert rejected, f"{case} H{heads}: {strides} gave {message!r}"
    return {
        "gfx": get_gfx(),
        "kv_tokens": tokens,
        "k strides": k_strides,
        "v strides": v_strides,
        "q contiguous": q.is_contiguous(),
        "rejected": rejected,
    }


@benchmark()
def test_qsa_max_stride(case, heads):
    """A request read in place through K/V rows KV_STRIDE_LIMIT - 8 elements apart, the
    largest stride accepted (packed rows, or pool rows through kv_slots), by packed
    direct and by raw direct (RAW_DIRECT), against the call on the same rows in
    contiguous K/V; both match bit for bit. Token t reads view row first + t % 4; the
    other view rows are NaN. The 1536-row request runs with every tile forced onto union
    (FORCED_ROUTES) and must run union tiles."""
    view_rows, first, slotted, wide, rows = MAX_STRIDE_CASES[case]
    tokens = max(rows, 4)
    query_lens, prefix_lens = (rows,), (tokens - rows,)
    generator = torch.Generator(device="cuda").manual_seed(heads)
    q = torch.randn(rows, heads, HEAD_DIM, generator=generator, dtype=dtypes.bf16)
    k4, v4 = (
        torch.randn(4, 1, HEAD_DIM, generator=generator, dtype=dtypes.bf16)
        for _ in range(2)
    )
    indices = make_indices(query_lens, prefix_lens, "shared", generator, 512)
    stride = KV_STRIDE_LIMIT - 8
    strided_k, strided_v = (
        torch.full(
            ((view_rows - 1) * stride + HEAD_DIM,), float("nan"), dtype=dtypes.bf16
        ).as_strided((view_rows, 1, HEAD_DIM), (stride, HEAD_DIM, 1))
        for _ in range(2)
    )
    strided_k[first : first + 4], strided_v[first : first + 4] = k4, v4
    k, v = k4[torch.arange(tokens) % 4], v4[torch.arange(tokens) % 4]
    kwargs = {
        "query_lens": query_lens,
        "prefix_lens": prefix_lens,
        "softmax_scale": SCALE,
    }
    slots = (first + torch.arange(tokens) % 4).to(dtypes.i32) if slotted else None
    if get_gfx_runtime() != "gfx950":
        return rejected_off_gfx950(
            f"{case} H{heads}",
            q,
            strided_k,
            strided_v,
            indices,
            kv_slots=slots,
            **kwargs,
        )
    ref = run_torch(q, k, v, indices, query_lens, prefix_lens, SCALE)
    ret = {
        "gfx": get_gfx(),
        "rows": rows,
        "row stride": stride,
        "K span GiB": strided_k.untyped_storage().nbytes() / 2**30,
        "kv_slots": slotted,
    }
    union = [FORCED_ROUTES["union"]] if rows >= 1536 else []
    for override, expected in (
        ([], "packed"),
        ([RAW_DIRECT], "raw wide" if wide else "raw"),
    ):
        with platform_entry(",".join(override + union)):
            out = qsa_sparse_prefill(
                q, strided_k, strided_v, indices, kv_slots=slots, **kwargs
            )
            check_union(ret, case, f"{case} H{heads}")
            contiguous = qsa_sparse_prefill(q, k, v, indices, **kwargs)
        label = f"{case} H{heads} flydsl_strided {ret['direct']}"
        err = checkAllclose(
            ref, out.to(dtypes.fp32), rtol=2e-2, atol=2e-2, msg=f"{label}: out"
        )
        bitwise = checkAllclose(
            contiguous, out, rtol=0, atol=0, msg=f"{label} vs contiguous: out"
        )
        assert ret["direct"] == expected, f"{label}: expected direct {expected}"
        if union:
            assert ret["union tiles"] > 0, f"{label}: no union tile ran"
        assert err == 0 and bitwise == 0, f"{label}: mismatch {err}, {bitwise}"
        ret[f"{expected} err"] = err
        ret[f"{expected} union tiles"] = ret.pop("union tiles")
    del ret["direct"]
    return ret


@benchmark()
def test_qsa_route(case, heads):
    """A RAW_DIRECT_LAYOUTS layout with its route forced (FORCED_ROUTES), against the
    reference: on gfx950 the union route must run union tiles, and the direct route must
    run every tile on gated raw direct (gfx942 reports the route only)."""
    layout, route = ROUTE_CASES[case]
    query_lens, prefix_lens, *_ = CASES[layout]
    generator = torch.Generator(device="cuda").manual_seed(
        list(CASES).index(layout) * 16 + heads
    )
    q, k, v, indices = make_inputs(layout, heads, generator)
    ref = run_torch(q, k, v, indices, query_lens, prefix_lens, SCALE)
    label = f"{case} H{heads}"
    ret = {"gfx": get_gfx(), "rows": q.shape[0], "route": route}
    with platform_entry(FORCED_ROUTES[route]):
        out = qsa_sparse_prefill(
            q,
            k,
            v,
            indices,
            query_lens=query_lens,
            prefix_lens=prefix_lens,
            softmax_scale=SCALE,
        )
        check_union(ret, layout, label)
        tiles = latest_workspace().union.num_tiles
    ret["tiles"] = tiles
    ret["flydsl err"] = checkAllclose(
        ref, out.to(dtypes.fp32), rtol=2e-2, atol=2e-2, msg=f"{label}: out"
    )
    assert ret["flydsl err"] == 0, f"{label}: mismatch {ret['flydsl err']}"
    if get_gfx_runtime() == "gfx950":
        if route == "union":
            assert ret["union tiles"] > 0, f"{label}: no union tile ran"
        else:
            assert ret["union tiles"] == 0, f"{label}: {ret['union tiles']} union tiles"
    return ret


def kv_bytes(case):
    """K/V bytes a case allocates, pools included (graph cases stay small)."""
    if case in MAX_STRIDE_CASES:
        return 2 * MAX_STRIDE_CASES[case][0] * KV_STRIDE_LIMIT * 2
    if case in ROUTE_CASES:
        case = ROUTE_CASES[case][0]
    if case in SPAN_LIMIT_CASES:
        tokens, slotted = SPAN_LIMIT_CASES[case]
        return tokens * (4 if slotted else HEAD_DIM * 2)
    if case in STRIDE_REJECT_CASES:
        tokens, _, k_strides, v_strides, *_ = STRIDE_REJECT_CASES[case]
        return tokens * (k_strides[0] + v_strides[0]) * 2
    if case in STRIDE_CASES:
        layout, _, pool_rows, _ = STRIDE_CASES[case]
    else:
        layout, pool_rows, _ = SLOT_CASES.get(case, (case, None, None))
    if layout not in CASES:
        return 0
    query_lens, prefix_lens, hk = CASES[layout][:3]
    rows = max(pool_rows or 0, sum(query_lens) + sum(prefix_lens))
    return 2 * rows * hk * HEAD_DIM * 2


def main():
    if get_gfx() not in SUPPORTED_GFX:
        aiter.logger.warning(
            "qsa_sparse_prefill unsupported on %s; skipping", get_gfx()
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
        choices=[
            *CASES,
            *ROUTE_CASES,
            *GRAPH_CASES,
            *SLOT_CASES,
            *STRIDE_CASES,
            *SPAN_LIMIT_CASES,
            *STRIDE_REJECT_CASES,
            *MAX_STRIDE_CASES,
        ],
        default=[
            *CASES,
            *ROUTE_CASES,
            *GRAPH_CASES,
            *SLOT_CASES,
            *STRIDE_CASES,
            *SPAN_LIMIT_CASES,
            *STRIDE_REJECT_CASES,
            *MAX_STRIDE_CASES,
        ],
        help="input layouts to run (functional, route, graph, kv_slots, strided, span "
        "limit, stride rejection, max stride and performance cases)",
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

    layouts = {**ROUTE_CASES, **GRAPH_CASES, **SLOT_CASES, **STRIDE_CASES}
    for suite, cases, test in (
        (
            "functional",
            [*FUNCTIONAL_CASES, *CONSTANT_CASES, *UNION_CASES],
            test_qsa_sparse_attention,
        ),
        ("route", list(ROUTE_CASES), test_qsa_route),
        ("graph", list(GRAPH_CASES), test_qsa_graph_replay),
        ("kv_slots", list(SLOT_CASES), test_qsa_kv_slots),
        ("strided", list(STRIDE_CASES), test_qsa_strided_kv),
        ("span limit", list(SPAN_LIMIT_CASES), test_qsa_span_limit),
        ("stride rejected", list(STRIDE_REJECT_CASES), test_qsa_stride_rejected),
        ("max stride", list(MAX_STRIDE_CASES), test_qsa_max_stride),
        ("performance", list(PERF_CASES), test_qsa_sparse_attention),
    ):
        df = []
        selected = [case for case in args.case if case in cases]
        for case, group in itertools.product(selected, args.group):
            # Rough peak: twice the K/V bytes (pools, gathered copies) plus 4 GiB of
            # reference temporaries.
            needed = 2 * kv_bytes(case) + 2**32
            torch.cuda.empty_cache()
            if needed > torch.cuda.mem_get_info()[0]:
                aiter.logger.warning(
                    "%s needs about %.0f GiB of free device memory; skipping",
                    case,
                    needed / 2**30,
                )
                continue
            layout = layouts.get(case, (case,))[0]
            hk = CASES[layout][2] if layout in CASES else 1
            df.append(test(case, group * hk))
        if not df:
            continue
        aiter.logger.info(
            "qsa_sparse_prefill %s summary (markdown):\n%s",
            suite,
            pd.DataFrame(df).to_markdown(index=False),
        )


if __name__ == "__main__":
    main()
