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
more KV heads. Cases listed in CASE_GROUPS run once, at their own group, whatever
-g selects. Functional cases cover one row, a short extend and chunked
prefill against long cached contexts (one whose packed KV exceeds the direct
kernel's 64 MiB budget, one packed by the per-row rule, one only raw direct
reads), nine 1M-token requests whose packed K/V exceed 4 GiB (the last, a
1024-row chunk, starts at 4 GiB), one request whose K and V each span the
4 GiB - 1 MiB limit (raw direct and union), ragged batches with an empty request,
HK > 1 (HK = 16 with over 4096 tasks to route), T = 256/768/1024 selections and
constant Q/K whose logits sit near +1e11 on the union path (``large_logits``).
The graph case captures a union layout on its own stream, replays it after
rewriting its indices in place, then again after another layout grows the
stream's scratch arena, which must keep the captured scratch allocated. The
kv_slots cases scatter a layout's K/V into random 64-token pages of a NaN-filled
token pool (past 4 GiB for the ``pool_4g`` cases) and read it in place through
the slot table; the result must equal the packed call on the gathered rows bit
for bit. The span limit cases must be rejected: one request whose K/V spans one
token past 4 GiB - 1 MiB, or whose int32 slot table spans one entry past it.
Cases needing more device memory than is free are skipped with a warning.
Performance cases are single-request full prefills of 1k-32k tokens plus the
12000-token serving prompt (T = 512); they are checked as well and are the only
cases using the full-prefill BQ21/BQ42 union tiles. ``shared`` selections let
32-row windows share block priorities; ``random`` rows choose independently;
``correlated`` (performance cases) simulates the real indexer's selection
distribution, captured Qwen3.8 layers whose neighbouring rows share about half
of their blocks (see correlated_prefill).

Group 16 is the most Q heads per KV head the interface accepts (BQ8 union
tiles, a direct wave of 16 heads): ``mixed_union_raw`` batches a union request
with direct rows reading the raw K/V (a tuple gives each request its own
selection), and ``kv_slots_g16_hk2_ragged`` reads two KV heads through a slot
table by packed direct. The new-scale graph cases warm a layout at the default
scale on their own stream, then capture graphs at softmax scales no other case
uses: the scale is a kernel argument, so capturing compiles nothing and builds
no workspace. They replay after rewriting the indices in place (and the slot
table, with the K/V moved to other pages), and an eager call at yet another
scale must add no FlyDSL specialization. The batch invariance cases run each
request alone (one of them a full H6 prefill, which also takes the BQ21 tiles)
and all of them in one call with batch_invariant=True, and both must give the
same bits: union planning (the row threshold) and the union tile rows are
decided per request and K3 routes no tile by the others (packed and raw direct
agree bit for bit). Each batch also runs in the default mode, whose decisions
span the call, on a workspace of its own.

FP8 K/V (dtypes.fp8, E4M3FNUZ on gfx942) come with FP32 scales, per tensor [1] or per
token and KV head [N, HK] ([P, HK] for a pool), and run with FP8 MFMAs (compute="fp8")
or dequantized to BF16 (compute="bf16"). Their data:
random K/V spread by 2**U(-1, 1) per (token, KV head), quantized with scale amax / 240.
compute="fp8" must stay finite, within FP8_TOL (per (row, head) norm-relative error,
mean and worst) of the FP32 reference on the dequantized K/V over sample rows (each
request's first and last rows plus 64 evenly spaced), or where noted within the case's bounds
of the reference on the FP8 Q the kernels compute with, and give the same bits when
repeated; compute="bf16" must give the bits of the BF16 path on the K/V the caller
dequantizes, and through a pool both give the packed call's bits (FP8_RAW_CHECKED: also
with every direct row on the raw K/V). The FP8 cases pair compute, scale layout, union /
packed direct / raw direct, packed or pooled K/V and groups 12/6/3, and add: group 16 and
HK2, T = 256/768/1024; an attention sink over a flat tail
whose FP8 probabilities must not underflow; K = 0 with power-of-two V scales, whose output
is exact, and zero Q rows; the reference on the FP8 Q the kernels compute with, also for
sharp scores; large logits; K/V past 4 GiB, packed (a request crossing it) and as a pool
whose pages lie past it; Q and out past 2 GiB (H192), whose rows past 2 GiB merge residual
blocks by their out offset; graph replay with every input rewritten in place, scratch growth
and new softmax scales; a workspace key that must hold the K/V dtype, scale layout, pool
and batch mode; input errors and the entry's signature; batch invariance with
batch_invariant=True. The 12000-token prefill is timed against the BF16 path on the
dequantized K/V and aiter's Triton FP8 GQA (qsa_sparse_paged_gqa). The FP8 cases assert
their paths too, including the FP8 split, core/residual union tiles and the skipped pack.

``triton`` is SGLang's native QSA extend path, copied below under SGLang's
Apache-2.0 license (see the notice there) and called as its backend does:
sparse_gqa_fwd_interface_triton when no request has a prefix, else
sparse_gqa_fwd_interface_triton_ck over the packed full-context K/V.

Usage:
    python op_tests/test_flydsl_qsa_sparse_prefill.py
    python op_tests/test_flydsl_qsa_sparse_prefill.py -c prefill_12000 -g 12 6 3
    python op_tests/test_flydsl_qsa_sparse_prefill.py -c batch_invariance_h12
    python op_tests/test_flydsl_qsa_sparse_prefill.py -c fp8_ragged_4_token
"""

import argparse
import contextlib
import gc
import importlib
import inspect
import math
import os
import pkgutil
import weakref
import zlib

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import triton
import triton.language as tl

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_cu_num, get_gfx
from aiter.ops.flydsl import qsa_sparse_prefill as flydsl_prefill
from aiter.ops.flydsl.kernels import qsa as qsa_kernels
from aiter.ops.qsa import qsa_sparse_paged_gqa, qsa_sparse_prefill
from aiter.test_common import benchmark, checkAllclose, run_perftest

torch.set_default_device("cuda")

SUPPORTED_GFX = ["gfx942"]
HEAD_DIM = 256
SCALE = HEAD_DIM**-0.5
FP8 = dtypes.fp8
FP8_MAX = 240.0  # E4M3FNUZ
# Per-(row, head) norm-relative error bounds (mean, worst) of compute="fp8" on random data:
# FP8 Q dominates (scores err by ~2**-5 of |q||k|), FP8 probabilities add ~2%.
FP8_TOL = (0.06, 0.16)

# name -> (query_lens, prefix_lens, kv_heads, selection, blocks per row T)
FUNCTIONAL_CASES = {
    # Below the union row threshold: direct only (packed, raw, HK2 with an empty request).
    "one_row": ((1,), (0,), 1, "shared", 512),
    "extend_17_ctx_30017": ((17,), (30000,), 1, "random", 512),
    "extend_64_ctx_70065": ((64,), (70001,), 1, "random", 512),
    "hk2_ragged": ((7, 0, 9), (0, 5, 3000), 2, "shared", 512),
    # Union + packed direct; random rows route every tile direct, 4096 rows pack per row.
    "ragged_4": ((300, 129, 1001, 63), (0, 517, 4093, 12345), 1, "shared", 512),
    "chunk_1024_ctx_31024_random": ((1024,), (30000,), 1, "random", 512),
    "chunk_4096_ctx_131072": ((4096,), (126976,), 1, "shared", 512),
    # Union + gated raw direct over a windowed K2.
    "chunk_1024_ctx_132096": ((1024,), (131072,), 1, "shared", 512),
    # Packed K/V over 4 GiB: the 1024-row request starts at 4 GiB (union, 4 K2 windows).
    "batch_9_ctx_1048576": (
        (2,) * 8 + (1024,),
        (2**20 - 2,) * 8 + (2**20 - 1024,),
        1,
        "shared",
        512,
    ),
    # One request at the 4 GiB - 1 MiB K/V limit, by raw direct and by union.
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
# Two requests with their own selections: the shared one runs union, the random one direct.
MIXED_CASES = {
    # The direct rows read the raw K/V: the batch's packed KV outgrows the pack budget.
    "mixed_union_raw": ((1024, 64), (3000, 70001), 1, ("shared", "random"), 512),
    # The direct rows read packed K/V.
    "mixed_union_packed": ((1001, 300), (4093, 30000), 1, ("shared", "random"), 512),
}
# Layouts only the FP8 cases use.
FP8_LAYOUTS = {
    # Union tiles and direct rows at HK2 and T = 768.
    "hk2_mixed_t768": ((1024, 64), (3000, 70001), 2, ("random", "random"), 768),
    # Union tiles over a 132096-token context, and direct rows; T = 1024.
    "topk1024_mixed_ctx_132096": (
        (1024, 64),
        (131072, 70001),
        1,
        ("shared", "random"),
        1024,
    ),
    "hk2_extend_64_ctx_70065": ((64,), (70001,), 2, "random", 512),
    "topk256_hk2_ragged": ((7, 0, 9), (0, 5, 3000), 2, "shared", 256),
    "chunk_1024_ctx_4024_random": ((1024,), (3000,), 1, "random", 512),
    "ragged_3_random": ((480, 1000, 301), (11392, 3, 4001), 1, "random", 512),
    "extend_9_ctx_139": ((9,), (130,), 1, "random", 512),
    # FP8 K/V past 4 GiB (16843185 tokens at HK1): the first two requests' K/V are the
    # longest a request may have (their BF16 span would be 4 GiB - 1 MiB), the last
    # request's crosses 4 GiB.
    "fp8_4g_packed": (
        (2, 768, 64),
        (8386558, 8385792, 70001),
        1,
        ("random", "shared", "random"),
        512,
    ),
    # Q and out of 2.25 GiB at group 12 (H192): rows from 21845 on reach past 2 GiB.
    "hk16_prefill_24576": ((24576,), (0,), 16, "shared", 512),
    # 262 tiles x 16 KV heads = 4192 union tasks, more than one K3 order CTA sorts (4096):
    # the random request's 6 tiles route direct, so the first chunk of the task order ends
    # with idle entries, and the second chunk (KV head 15 of the last 96 tiles) must run.
    "hk16_k3_chunks": ((60, 2560), (126976, 0), 16, ("random", "shared"), 512),
}
CASES = {
    **FUNCTIONAL_CASES,
    **PERF_CASES,
    **{name: shape for name, (shape, _) in CONSTANT_CASES.items()},
    **MIXED_CASES,
    **FP8_LAYOUTS,
}
# name -> (captured union layout, layout whose first call then grows the scratch arena:
# its packed KV outgrows the captured layout's)
GRAPH_CASES = {"graph_ragged_4": ("ragged_4", "chunk_1024_ctx_31024_random")}
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
    # At group 16 (see CASE_GROUPS): a 16-head direct wave per KV head, over two KV heads.
    "kv_slots_g16_hk2_ragged": ("hk2_ragged", None, 1),
}
# name -> (K/V tokens of one request, read through kv_slots): one past the
# 4 GiB - 1 MiB byte span of the request's K/V, or of the int32 slot table
SPAN_LIMIT_CASES = {
    "span_over_limit": ((2**32 - 2**20) // (HEAD_DIM * 2) + 1, False),
    "kv_slots_span_over_limit": ((2**32 - 2**20) // 4 + 1, True),
}
# name -> (layout, Q heads per KV head, read through kv_slots, softmax scales no other
# case uses: one captured, one called eagerly). Union with packed direct, and through a
# slot table union with raw direct, so that every kernel taking the scale runs.
SCALE_CASES = {
    "graph_new_scale_mixed_union_packed": (
        "mixed_union_packed",
        12,
        False,
        (2**-4.5, 0.0517),
    ),
    "graph_new_scale_kv_slots_mixed_union_raw": (
        "mixed_union_raw",
        3,
        True,
        (0.11, 0.0389),
    ),
}
# name -> (Q heads per KV head, requests as (query rows, cached prefix tokens, selection)):
# with batch_invariant=True each request alone must give the bits it gives batched.
BATCH_CASES = {
    # Alone, 300 rows are below the union planning rows and run direct; batched with a
    # union request they must stay direct (by default the batch's union plan covers them).
    # The batch reads raw K/V where the requests alone pack it, and holds an empty request.
    "batch_invariance_h12": (
        12,
        (
            (300, 517, "shared"),
            (1024, 30000, "random"),
            (0, 5, "shared"),
            (64, 70001, "random"),
            (1001, 4093, "shared"),
        ),
    ),
    # Alone, the full prefill takes the BQ21 union tiles of single-request H6 prefill.
    "batch_invariance_full_prefill_h6": (
        6,
        ((4096, 0, "correlated"), (17, 30000, "random")),
    ),
}
# FP8 pool rows of 4.5 GiB at HK1.
FP8_POOL_4G_ROWS = (2**32 + 2**29) // HEAD_DIM
# FP8 K/V cases: name -> (layout, Q heads per KV head, scales ("tensor" [1] or "token"
# [N, HK]), compute, None for packed K/V or a pool as (pool rows, None for twice the pages
# needed; first pool page; whether the last request's last page is the pool's last page)).
# The first eight pair compute, scales, paths, pools and groups 12/6/3; then group 16
# (direct through a pool at HK2 and T = 256, union with BQ8 tiles), large logits, the BQ42
# tile of H3 full prefill (FP8 union with 64-bit membership), the 12000-token serving
# prompt (see FP8_PERF_CASES), K/V past 4 GiB: packed (the last request crosses it) and a
# 4.5 GiB pool whose pages start at 4 GiB and end at its last page, Q and out past 2 GiB
# (see FP8_LARGE_Q_CASES), and union tasks past K3's first order chunk.
FP8_CASES = {
    "fp8_ragged_4_token": ("ragged_4", 12, "token", "fp8", None),
    "fp8_kv_slots_ragged_4_tensor": ("ragged_4", 12, "tensor", "fp8", (None, 1, False)),
    "fp8_hk2_mixed_t768_tensor": ("hk2_mixed_t768", 6, "tensor", "fp8", None),
    "fp8_kv_slots_topk1024_mixed_ctx_132096_token": (
        "topk1024_mixed_ctx_132096",
        3,
        "token",
        "fp8",
        (None, 1, False),
    ),
    "bf16c_mixed_union_raw_tensor": ("mixed_union_raw", 3, "tensor", "bf16", None),
    "bf16c_kv_slots_extend_64_ctx_70065_token": (
        "extend_64_ctx_70065",
        6,
        "token",
        "bf16",
        (None, 1, False),
    ),
    "bf16c_kv_slots_hk2_extend_64_ctx_70065_tensor": (
        "hk2_extend_64_ctx_70065",
        12,
        "tensor",
        "bf16",
        (None, 1, False),
    ),
    "bf16c_mixed_union_packed_token": ("mixed_union_packed", 12, "token", "bf16", None),
    "fp8_kv_slots_g16_topk256_hk2_ragged_token": (
        "topk256_hk2_ragged",
        16,
        "token",
        "fp8",
        (None, 1, False),
    ),
    "fp8_g16_mixed_union_raw_tensor": ("mixed_union_raw", 16, "tensor", "fp8", None),
    "fp8_large_logits_prefill_768_token": (
        "large_logits_prefill_768",
        3,
        "token",
        "fp8",
        None,
    ),
    "fp8_prefill_4096_h3_tensor": ("prefill_4096", 3, "tensor", "fp8", None),
    "fp8_prefill_12000_tensor": ("prefill_12000", 12, "tensor", "fp8", None),
    "fp8_4g_packed_token": ("fp8_4g_packed", 12, "token", "fp8", None),
    "fp8_pool_4g_ragged_4_tensor": (
        "ragged_4",
        12,
        "tensor",
        "fp8",
        (FP8_POOL_4G_ROWS, 2**32 // HEAD_DIM // PAGE, True),
    ),
    "fp8_hk16_prefill_24576_token": ("hk16_prefill_24576", 12, "token", "fp8", None),
    "fp8_hk16_k3_chunks_token": ("hk16_k3_chunks", 12, "token", "fp8", None),
}
# Also timed against the BF16 path on the dequantized K/V and aiter's Triton FP8 GQA, and
# checked against the reference on the kernels' FP8 Q (see test_qsa_fp8_query).
FP8_PERF_CASES = ("fp8_prefill_12000_tensor",)
# Checked against the reference on the kernels' FP8 Q instead of within FP8_TOL: FP8 Q
# rounding alone takes a few (row, head) vectors of its 4M-token rows (or of its 192 heads)
# near FP8_TOL's worst bound, while what its K/V or out addressing can break shows on
# either check.
FP8_QUERY_CHECKED = (
    "fp8_4g_packed_token",
    "fp8_hk16_prefill_24576_token",
    "fp8_hk16_k3_chunks_token",
)
# Also run with every direct row reading the raw K/V (raw_direct), which must give the
# same bits: packed and pooled K/V (the 4.5 GiB pool by 64-bit loads), both scale layouts,
# HK2 and group 16, direct-only and union batches.
FP8_RAW_CHECKED = (
    "fp8_hk2_mixed_t768_tensor",
    "fp8_kv_slots_g16_topk256_hk2_ragged_token",
    "fp8_g16_mixed_union_raw_tensor",
    "fp8_pool_4g_ragged_4_tensor",
)
# Q and out past 2 GiB: core/residual union tiles leave rows past 2 GiB of out to the
# direct kernel's residual merge, which finds their union log-sum-exp from the row's out
# offset. Checked (fp8_query_error) on every row past 2 GiB, some of which must have
# merged residual blocks (residual_rows).
FP8_LARGE_Q_CASES = ("fp8_hk16_prefill_24576_token",)
# name -> (layout, Q heads per KV head, scales): an attention sink over a flat tail.
FP8_SINK_CASES = {
    "fp8_sink_chunk_1024_ctx_4024_random_token": (
        "chunk_1024_ctx_4024_random",
        12,
        "token",
    ),
    "fp8_sink_extend_64_ctx_70065_tensor": ("extend_64_ctx_70065", 3, "tensor"),
}
# name -> (layout, Q heads per KV head); per token scales.
FP8_UNIFORM_CASES = {"fp8_uniform_ragged_3_random_token": ("ragged_3_random", 3)}
FP8_QUERY_CASES = {"fp8_query_reference_prefill_4096_token": ("prefill_4096", 12)}
# name -> (layout, Q heads per KV head, scales, read through kv_slots, layout whose first
# call then grows the stream's scratch arena or None, softmax scales no other case uses:
# one captured, one called eagerly)
FP8_GRAPH_CASES = {
    "fp8_graph_kv_slots_ragged_4_tensor": (
        "ragged_4",
        12,
        "tensor",
        True,
        "chunk_1024_ctx_31024_random",
        (2**-4.25, 0.0461),
    ),
    "fp8_graph_mixed_union_raw_token": (
        "mixed_union_raw",
        3,
        "token",
        False,
        None,
        (0.0723, 0.0297),
    ),
}
# name -> (layout, Q heads per KV head)
FP8_WORKSPACE_CASES = {"fp8_workspace_key_ragged_4": ("ragged_4", 12)}
FP8_ERROR_CASES = {"fp8_input_errors": ("extend_9_ctx_139", 3)}
# As BATCH_CASES, with FP8 K/V (per token scales) and batch_invariant=True. By default
# the first two requests alone have few enough union tasks to split their key ranges and
# batched they do not: batch_invariant=True must not split.
FP8_BATCH_CASES = {
    "fp8_batch_invariance_token": (
        3,
        ((768, 0, "shared"), (2048, 30000, "shared"), (64, 70001, "random")),
    ),
}
FP8_LAYOUT_CASES = {
    **FP8_CASES,
    **FP8_SINK_CASES,
    **FP8_UNIFORM_CASES,
    **FP8_QUERY_CASES,
    **FP8_GRAPH_CASES,
    **FP8_WORKSPACE_CASES,
    **FP8_ERROR_CASES,
}
# name -> Q heads per KV head: these cases run once, at this group, whatever -g selects.
# Group 16 is the most the interface accepts (BQ8 union tiles, a 16-head direct wave).
CASE_GROUPS = {
    "mixed_union_raw": 16,
    "kv_slots_g16_hk2_ragged": 16,
    **{name: group for name, (_, group, *_) in SCALE_CASES.items()},
    **{name: group for name, (group, *_) in BATCH_CASES.items()},
    **{name: group for name, (_, group, *_) in FP8_LAYOUT_CASES.items()},
    **{name: group for name, (group, _) in FP8_BATCH_CASES.items()},
}
# FP8 workspaces' paths (see routes): union tiles ran, direct rows ("packed", "raw" or None),
# union tile rows, key-range parts of the union tasks, core/residual tiles ran residual
# blocks direct, the context was packed for the direct kernel.
FP8_ROUTE_FIELDS = ("union", "direct", "query_tile", "split", "residual", "pack_k")
FP8_ROUTES = {
    "fp8_ragged_4_token": (True, None, 10, 1, True, True),
    "fp8_kv_slots_ragged_4_tensor": (True, None, 10, 1, True, True),
    "fp8_hk2_mixed_t768_tensor": (True, "packed", 16, 1, False, True),
    "fp8_kv_slots_topk1024_mixed_ctx_132096_token": (
        True,
        "packed",
        32,
        2,
        False,
        True,
    ),
    "fp8_kv_slots_g16_topk256_hk2_ragged_token": (
        False,
        "packed",
        None,
        None,
        False,
        True,
    ),
    "fp8_g16_mixed_union_raw_tensor": (True, "packed", 8, 1, False, True),
    # These two run every tile union whole: no pack runs.
    "fp8_large_logits_prefill_768_token": (True, None, 32, 2, False, False),
    "fp8_prefill_4096_h3_tensor": (True, None, 42, 1, False, False),
    "fp8_prefill_12000_tensor": (True, None, 10, 1, True, True),
    # Past the pack budget: direct reads the raw K/V, the union the raw V.
    "fp8_4g_packed_token": (True, "raw", 10, 1, False, False),
    "fp8_pool_4g_ragged_4_tensor": (True, None, 10, 1, True, True),
    "fp8_hk16_prefill_24576_token": (True, None, 10, 1, True, True),
    "fp8_hk16_k3_chunks_token": (True, "raw", 10, 1, False, False),
    "fp8_sink_chunk_1024_ctx_4024_random_token": (True, None, 10, 1, True, True),
    "fp8_sink_extend_64_ctx_70065_tensor": (False, "packed", None, None, False, True),
    "fp8_uniform_ragged_3_random_token": (True, "packed", 32, 2, False, True),
    "fp8_query_reference_prefill_4096_token": (True, None, 10, 1, True, True),
    "fp8_graph_kv_slots_ragged_4_tensor": (True, None, 10, 1, True, True),
    "fp8_graph_mixed_union_raw_token": (True, "packed", 32, 2, False, True),
    # The first request alone: no split with batch_invariant=True.
    "fp8_batch_invariance_token": (True, None, 32, 1, False, False),
}
# name -> paths its call under test takes (see routes); for batch invariance, the first
# request alone: whether union tiles ran, how direct rows read K/V ("packed", "raw", or
# None when every row ran union) and the union tile's query rows. compute="bf16" runs
# BF16 workspaces.
EXPECTED_ROUTES = {
    "bf16c_mixed_union_raw_tensor": {"union": True, "direct": "raw", "query_tile": 32},
    "bf16c_kv_slots_extend_64_ctx_70065_token": {
        "union": False,
        "direct": "raw",
        "query_tile": None,
    },
    "bf16c_kv_slots_hk2_extend_64_ctx_70065_tensor": {
        "union": False,
        "direct": "raw",
        "query_tile": None,
    },
    "bf16c_mixed_union_packed_token": {
        "union": True,
        "direct": "packed",
        "query_tile": 10,
    },
    **{
        name: dict(zip(FP8_ROUTE_FIELDS, paths, strict=True))
        for name, paths in FP8_ROUTES.items()
    },
}

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
    seed) or a tuple with one mode per request, a CUDA generator, and T = topk
    blocks per row.

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
    modes = (selection,) * len(query_lens) if isinstance(selection, str) else selection
    if len(modes) != len(query_lens):
        raise ValueError("a selection tuple needs one mode per request")
    if "correlated" in modes:
        if len(query_lens) != 1 or prefix_lens[0] or topk != 512:
            raise ValueError("correlated selections model one prefill with T = 512")
        return correlated_prefill(query_lens[0])
    rows = []
    for count, prefix, mode in zip(query_lens, prefix_lens, modes):
        if count == 0:
            continue
        visible = torch.arange(prefix + 1, prefix + count + 1)
        blocks = visible // 4
        width = max(int(blocks.max()), 1)
        if mode == "shared":
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


def run_torch(
    q, k, v, indices, query_lens, prefix_lens, scale, kv_scales=None, rows=None
):
    """FP32 attention over each row's selected tokens; padding entries (-1) are masked.

    FP8 K/V are dequantized after the gather by kv_scales = (k_scale, v_scale), FP32 [1]
    or [N, HK]. rows: the query rows to compute, in that order (default all of them).
    """
    heads, dim = q.shape[1:]
    kv_heads = k.shape[1]
    # About 1 GiB of gathered FP32 keys per chunk of rows.
    chunk = max(1, 2**28 // (indices.shape[1] * kv_heads * dim))
    q_lens = torch.tensor(query_lens)
    kv_lens = q_lens + torch.tensor(prefix_lens)
    kv_start = torch.repeat_interleave(torch.cumsum(kv_lens, 0) - kv_lens, q_lens)
    if rows is not None:
        rows = torch.tensor(rows, dtype=torch.long)
        q, indices, kv_start = q[rows], indices[rows], kv_start[rows]

    def gather(x, slots, x_scale):
        x = x[slots].float()
        if x_scale is None:
            return x
        return x * (x_scale[slots][..., None] if x_scale.ndim == 2 else x_scale)

    k_scale, v_scale = (None, None) if kv_scales is None else kv_scales
    out = torch.empty(q.shape[0], heads, dim, dtype=dtypes.fp32)
    for start in range(0, q.shape[0], chunk):
        end = min(start + chunk, q.shape[0])
        selected = indices[start:end].long()
        slots = selected.clamp(min=0) + kv_start[start:end, None]
        keys, values = gather(k, slots, k_scale), gather(v, slots, v_scale)
        query = q[start:end].float().view(end - start, kv_heads, -1, dim)
        scores = torch.einsum("rhgd,rthd->rhgt", query, keys) * scale
        scores.masked_fill_((selected < 0)[:, None, None, :], float("-inf"))
        out[start:end] = torch.einsum(
            "rhgt,rthd->rhgd", scores.softmax(-1), values
        ).reshape(end - start, heads, dim)
    return out


# Copied unchanged from SGLang, from here through sparse_gqa_fwd_interface_triton_ck: lines
# 9-309 of
# https://github.com/sgl-project/sglang/blob/52fecfdf0908dca24f4c6799ff5967125cc4110e/python/sglang/srt/layers/attention/qsa/sparse_attn.py
# That file has no license header; this is the notice of SGLang's LICENSE:
#
#   Copyright 2023-2024 SGLang Team
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
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


# End of the code copied from SGLang.


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


def latest_workspace():
    """The prefill's most recently used cached workspace (flydsl_prefill._workspaces keeps
    it last)."""
    torch.cuda.synchronize()
    return next(reversed(flydsl_prefill._workspaces.values()))


def routes():
    """Paths the latest qsa_sparse_prefill call took, read from its cached workspace.

    Returns whether any union tile ran ("union"), how the rows routed direct read K/V
    ("direct": "packed", "raw", or None when every row ran union) and the union tile's
    query rows ("query_tile", None without a union plan). FP8 workspaces (FP8 K/V with
    compute="fp8"; compute="bf16" runs BF16 ones) choose packed or raw direct the same way
    from the FP8 pack's size, and add the key-range parts of the union tasks ("split",
    None without a union plan), whether rows of core/residual union tiles ran their blocks
    outside the tile's core in the direct kernel ("residual") and whether the call packed
    the context for the direct kernel ("pack_k": never for raw direct, and no pack runs
    when every tile runs union whole).

    The helpers from latest_workspace to scratch_arena (and test_qsa_graph_replay's arena)
    are the only code reading prefill internals:
      - flydsl_prefill._workspaces, the most recently used workspace last;
      - a workspace's fp8 (None for BF16 K/V, else the scale layout), its BF16
        direct.pack_shape (empty when direct reads the raw K/V) or FP8 direct.raw, and
        its union plan (None
        without one): .active (the tiles the latest call ran union), .query_tile, and
        .union_ratio / .route_model (the calibration it routes with); FP8 plans add
        .split, .core_ratio, .residual (each tile's residual direct steps, written when
        core_ratio > 0), .residual_rows (each row's residual blocks, written per call
        when core_ratio > 0) and .direct_flag (entry 0 set when the pack must run);
      - flydsl_prefill.platform(None / "tensor" / "token"), the route calibrations;
      - a workspace's q8 / qs scratch, the FP8 Q of its latest call (fp8_query);
      - flydsl_prefill._arenas[(device, stream)].generation and .buffers.
    """
    workspace = latest_workspace()
    plan, fp8 = workspace.union, workspace.fp8 is not None
    raw = workspace.direct.raw if fp8 else not workspace.direct.pack_shape
    read = "raw" if raw else "packed"
    if plan is None:
        taken = {"union": False, "direct": read, "query_tile": None}
    else:
        active = plan.active.bool()
        taken = {
            "union": bool(active.any()),
            "direct": None if bool(active.all()) else read,
            "query_tile": plan.query_tile,
        }
    if fp8:
        taken["split"] = None if plan is None else plan.split
        taken["residual"] = (
            plan is not None
            and plan.core_ratio > 0
            and bool((active & (plan.residual > 0)).any())
        )
        # Without a union plan every row runs direct.
        taken["pack_k"] = not raw and (plan is None or bool(plan.direct_flag[0]))
    return taken


@contextlib.contextmanager
def raw_direct():
    """Direct rows read the raw K/V whatever its size: the platform's pack budget is zeroed
    (AITER_QSA_PLATFORM, read when the calibration is first used; a user's overrides stay,
    the later fields win) and workspaces planned with it are dropped, on entry and on
    exit."""

    def replan():
        flydsl_prefill.platform.cache_clear()
        workspaces = flydsl_prefill._workspaces
        for key in [key for key, value in workspaces.items() if not value.captured]:
            del workspaces[key]

    previous = os.environ.get("AITER_QSA_PLATFORM", "").strip()
    raw = "max_packed_kv_bytes=0,pack_tokens_per_row=0"
    os.environ["AITER_QSA_PLATFORM"] = f"{previous},{raw}" if previous else raw
    replan()
    try:
        yield
    finally:
        if previous:
            os.environ["AITER_QSA_PLATFORM"] = previous
        else:
            del os.environ["AITER_QSA_PLATFORM"]
        replan()


def check_routes(case, heads):
    """routes() of the latest call, which must be the paths EXPECTED_ROUTES lists for case
    on MI308X's 80 CUs (plans and routes follow the CU count)."""
    taken = routes()
    expected = EXPECTED_ROUTES[case]
    assert get_cu_num() != 80 or taken == expected, f"{case} H{heads}: took {taken}"
    return taken


def plan_terms():
    """(union_ratio, route_model) the latest call's union plan routes with."""
    plan = latest_workspace().union
    return plan.union_ratio, plan.route_model


def residual_rows():
    """Rows the latest call's direct kernel merged residual blocks into (rows of
    core/residual union tiles with blocks outside their tile's core); complete when every
    tile ran union (routes()["direct"] is None: the residual launch writes every row).
    """
    return (latest_workspace().union.residual_rows > 0).nonzero().flatten().tolist()


def calibrated_terms(scales):
    """(union_ratio, route_model) of the platform calibration of K/V with scales: None for
    BF16, else the FP8 kernels' "tensor" or "token" terms (flydsl_prefill.platform)."""
    config = flydsl_prefill.platform(scales)
    route_model = (
        config.union_fixed,
        config.union_step,
        config.row_step,
        config.route_margin,
    )
    return config.union_ratio, route_model


def fp8_query():
    """FP32 Q as the latest call quantized it for compute="fp8": its workspace's q8
    scratch (E4M3FNUZ per (row, head), written by the FP8 pack launch) times its scales
    qs (amax / 240). Call it before another call reuses the stream's scratch."""
    torch.cuda.synchronize()
    workspace = latest_workspace()
    return workspace.q8.view(FP8).float() * workspace.qs[..., None]


def scratch_arena(stream):
    """(generation, buffer addresses) of a stream's scratch arena (flydsl_prefill._arenas,
    keyed by device and stream); each growth replaces the buffers and adds a generation.
    """
    device = torch.device("cuda", torch.cuda.current_device())
    arena = flydsl_prefill._arenas[(device, stream.cuda_stream)]
    return arena.generation, [buffer.data_ptr() for buffer in arena.buffers]


def flydsl_variants():
    """FlyDSL specializations the QSA kernel modules hold so far, by FlyDSL's public
    JitFunction.cache_info(); None when FlyDSL's kernel cache is disabled."""
    total, seen = None, set()
    for info in pkgutil.iter_modules(qsa_kernels.__path__):
        module = importlib.import_module(f"{qsa_kernels.__name__}.{info.name}")
        for value in vars(module).values():
            if (
                id(value) in seen
                or not type(value).__module__.startswith("flydsl")
                or not callable(getattr(value, "cache_info", None))
            ):
                continue
            seen.add(id(value))
            stats = value.cache_info()
            if stats is not None:
                total = (total or 0) + stats.currsize
    return total


def mismatched_bits(a, b, msg):
    """Elements of BF16 a and b whose bits differ (checkAllclose logs them)."""
    checkAllclose(a, b, rtol=0, atol=0, msg=msg)
    return int((a.view(torch.int16) != b.view(torch.int16)).sum().item())


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
    return ret


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
    """A graph-captured union layout replayed with indices rewritten in place, before
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
    return {
        "gfx": get_gfx(),
        "rows": q.shape[0],
        "growth": growth,
        "flydsl graph err": max(errors),
    }


def make_pool(
    k, v, query_lens, prefix_lens, pool_rows, first_page, generator, last_page=False
):
    """K/V token pools and int32 slots: each request's packed K/V rows scattered to
    random PAGE-token pages from first_page on (with last_page, the last request's last
    page is the pool's last page); every other pool row is NaN (FP8: the E4M3FNUZ NaN
    byte 0x80)."""
    lengths = [n + p for n, p in zip(query_lens, prefix_lens)]
    counts = [-(-n // PAGE) for n in lengths]
    pool_rows = pool_rows or (2 * sum(counts) + first_page) * PAGE
    pages = first_page + torch.randperm(
        pool_rows // PAGE - first_page, generator=generator, device="cuda"
    )
    if last_page:
        used, end = sum(counts) - 1, pool_rows // PAGE - 1
        swap = [used, int((pages == end).nonzero())]
        pages[swap] = pages[swap[::-1]]
    slots, first = [], 0
    for n, count in zip(lengths, counts):
        tokens = torch.arange(n)
        slots.append(
            pages[first : first + count][tokens // PAGE] * PAGE + tokens % PAGE
        )
        first += count
    slots = torch.cat(slots).to(dtypes.i32)
    pools = []
    for packed in (k, v):
        if packed.dtype == FP8:
            pool = torch.full(
                (pool_rows, *packed.shape[1:]), 0x80, dtype=torch.uint8
            ).view(FP8)
        else:
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
    bitwise = checkAllclose(
        outputs["flydsl_gather"],
        outputs["flydsl_kv_slots"],
        rtol=0,
        atol=0,
        msg=f"{case} H{heads} flydsl_kv_slots vs flydsl_gather: out",
    )
    assert bitwise == 0, f"{case} H{heads}: kv_slots differs from the gathered call"
    return ret


@benchmark()
def test_qsa_span_limit(case, heads):
    """A request whose K/V or slot table spans more than 4 GiB - 1 MiB is rejected
    before any launch."""
    tokens, slotted = SPAN_LIMIT_CASES[case]
    q = torch.zeros(1, heads, HEAD_DIM, dtype=dtypes.bf16)
    kv = torch.empty(PAGE if slotted else tokens, 1, HEAD_DIM, dtype=dtypes.bf16)
    slots = torch.zeros(tokens, dtype=dtypes.i32) if slotted else None
    span = (kv if slots is None else slots).nbytes
    # A valid selection (blocks 0-511, then the causal tail), so that a missing check
    # fails the assertion rather than faulting in a kernel.
    indices = torch.full((1, 4 * 512 + 3), -1, dtype=dtypes.i32)
    indices[0, : 4 * 512] = torch.arange(4 * 512)
    indices[0, 4 * 512 : 4 * 512 + tokens % 4] = torch.arange(tokens // 4 * 4, tokens)
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
def test_qsa_graph_new_scale(case, heads):
    """softmax_scale is a kernel argument, not a compile-time constant. After one eager
    call at SCALE warms the layout on a stream, graphs captured there at SCALE and at a
    scale no other case uses replay correctly: capture fails on any kernel variant or
    workspace it would have to build. They replay after the indices are rewritten in
    place and, through a slot table, after the K/V move to other pages and the slots are
    rewritten in place. An eager call at yet another new scale adds no FlyDSL
    specialization."""
    layout, _, slotted, (new_scale, eager_scale) = SCALE_CASES[case]
    query_lens, prefix_lens, _, selection, topk = CASES[layout]
    generator = torch.Generator(device="cuda").manual_seed(
        list(CASES).index(layout) * 16 + heads
    )
    q, k, v, first = make_inputs(layout, heads, generator)
    second = make_indices(query_lens, prefix_lens, selection, generator, topk)
    kwargs = {"query_lens": query_lens, "prefix_lens": prefix_lens}
    kv = (k, v)
    if slotted:
        *kv, kwargs["kv_slots"] = make_pool(
            k, v, query_lens, prefix_lens, None, 1, generator
        )
    indices, scales = first.clone(), (SCALE, new_scale)
    outs = [torch.empty_like(q) for _ in scales]
    graphs = [torch.cuda.CUDAGraph() for _ in scales]
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        qsa_sparse_prefill(q, *kv, indices, out=outs[0], softmax_scale=SCALE, **kwargs)
    with torch.cuda.stream(stream):
        for graph, out, scale in zip(graphs, outs, scales):
            with torch.cuda.graph(graph, stream=stream):
                qsa_sparse_prefill(
                    q, *kv, indices, out=out, softmax_scale=scale, **kwargs
                )

    def replay(selected, label):
        # The graph's stream must see this stream's writes (the moved pages) first.
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            indices.copy_(selected)
            for graph, out in zip(graphs, outs):
                out.fill_(float("nan"))
                graph.replay()
        stream.synchronize()
        errors = []
        for out, scale in zip(outs, scales):
            msg = f"{case} H{heads} {label} at scale {scale:.4g}"
            eager = qsa_sparse_prefill(q, *kv, selected, softmax_scale=scale, **kwargs)
            errors.append(mismatched_bits(eager, out, f"{msg}: graph vs eager"))
            errors.append(
                checkAllclose(
                    run_torch(q, k, v, selected, query_lens, prefix_lens, scale),
                    out.to(dtypes.fp32),
                    rtol=2e-2,
                    atol=2e-2,
                    msg=f"{msg}: out",
                )
            )
        return max(errors)

    errors = [replay(second, "replay")]
    if slotted:
        # Other pages: the graph reads the slot table and the pool on every replay.
        moved = make_pool(k, v, query_lens, prefix_lens, kv[0].shape[0], 1, generator)
        for live, new in zip((*kv, kwargs["kv_slots"]), moved):
            live.copy_(new)
    errors.append(
        replay(first, "replay after moving the pages" if slotted else "replay")
    )
    assert max(errors) == 0, f"{case} H{heads}: mismatch {errors}"

    before = flydsl_variants()
    qsa_sparse_prefill(q, *kv, first, softmax_scale=eager_scale, **kwargs)
    after = flydsl_variants()
    assert after == before, (
        f"{case} H{heads}: softmax_scale {eager_scale} took new FlyDSL "
        f"specializations ({before} -> {after})"
    )
    return {
        "gfx": get_gfx(),
        "rows": q.shape[0],
        "kv_slots": slotted,
        "captured scales": f"{SCALE:.4g}, {new_scale:.4g}",
        "eager scale": f"{eager_scale:.4g}",
        "flydsl graph err": max(errors),
        "new specializations": None if before is None else after - before,
    }


@benchmark()
def test_qsa_batch_invariance(case, heads):
    """With batch_invariant=True every request alone gives the bits it gives in one call
    with all of the batch's requests: union planning (the row threshold) and the union
    tile rows (BQ21/BQ42 of a single full prefill) are decided per request, K3 routes no
    tile by the others, and packed and raw direct agree bit for bit. The one exception, a
    batch whose union scratch exceeds its budget running direct only, takes millions of
    rows and is not tested. The batch first runs in the default mode, whose decisions span
    the call: it must stay within tolerance, and batch_invariant=True must not reuse its
    workspace (its plan would change the requests' bits)."""
    _, requests = BATCH_CASES[case]
    generator = torch.Generator(device="cuda").manual_seed(
        len(CASES) * 16 + list(BATCH_CASES).index(case) * 16 + heads
    )
    parts = []
    for rows, prefix, selection in requests:
        q = torch.randn(rows, heads, HEAD_DIM, generator=generator, dtype=dtypes.bf16)
        k = torch.randn(
            rows + prefix, 1, HEAD_DIM, generator=generator, dtype=dtypes.bf16
        )
        v = torch.randn(
            rows + prefix, 1, HEAD_DIM, generator=generator, dtype=dtypes.bf16
        )
        if rows:
            indices = make_indices((rows,), (prefix,), selection, generator, 512)
        else:
            indices = torch.empty(0, 4 * 512 + 3, dtype=dtypes.i32)
        parts.append((q, k, v, indices))
    query_lens = tuple(rows for rows, _, _ in requests)
    prefix_lens = tuple(prefix for _, prefix, _ in requests)
    q, k, v, indices = (torch.cat(tensors) for tensors in zip(*parts))
    kwargs = {
        "query_lens": query_lens,
        "prefix_lens": prefix_lens,
        "softmax_scale": SCALE,
    }
    reference = run_torch(q, k, v, indices, query_lens, prefix_lens, SCALE)
    default_err = checkAllclose(
        reference,
        qsa_sparse_prefill(q, k, v, indices, **kwargs).to(dtypes.fp32),
        rtol=2e-2,
        atol=2e-2,
        msg=f"{case} H{heads} batched, default mode: out",
    )
    batched = qsa_sparse_prefill(q, k, v, indices, batch_invariant=True, **kwargs)
    err = checkAllclose(
        reference,
        batched.to(dtypes.fp32),
        rtol=2e-2,
        atol=2e-2,
        msg=f"{case} H{heads} batched: out",
    )
    differing, first = [], 0
    for part, (rows, prefix, _) in zip(parts, requests):
        if rows:
            out = qsa_sparse_prefill(
                *part,
                query_lens=(rows,),
                prefix_lens=(prefix,),
                softmax_scale=SCALE,
                batch_invariant=True,
            )
            differing.append(
                mismatched_bits(
                    out,
                    batched[first : first + rows],
                    f"{case} H{heads} request ({rows}, {prefix}) alone vs batched: out",
                )
            )
        first += rows
    assert err == 0, f"{case} H{heads}: mismatch {err}"
    assert not default_err, f"{case} H{heads}: default mode mismatch {default_err}"
    assert not any(differing), (
        f"{case} H{heads}: requests alone differ from the batch in {differing} "
        "elements"
    )
    return {
        "gfx": get_gfx(),
        "rows": q.shape[0],
        "requests": len(requests),
        "flydsl err": err,
        "default mode err": default_err,
        "differing elements": max(differing),
    }


def fp8_seed(case, heads):
    """Generator seed of an FP8 case."""
    return zlib.crc32(f"{case} H{heads}".encode())


def quantize_fp8(x, per_token):
    """E4M3FNUZ x and its FP32 scales amax / 240, per (token, KV head) [N, HK] or per
    tensor [1]."""
    x = x.float()
    amax = x.abs().amax(-1) if per_token else x.abs().amax().reshape(1)
    scale = (amax / FP8_MAX).clamp_min(1e-12)
    x8 = (x / (scale[..., None] if per_token else scale)).clamp(-FP8_MAX, FP8_MAX)
    return x8.to(FP8), scale.contiguous()


def dequantize_fp8(x8, scale):
    """BF16 K/V as a caller dequantizes FP8 x8: the FP32 product x8 * scale rounded once."""
    return (x8.float() * (scale[..., None] if scale.ndim == 2 else scale)).to(
        dtypes.bf16
    )


def make_fp8_kv(tokens, kv_heads, per_token, generator):
    """FP8 K or V and its scales: random BF16 values times 2**U(-1, 1) per (token, KV
    head), quantized; per token in chunks of 2**20 tokens."""
    x8 = torch.empty(tokens, kv_heads, HEAD_DIM, dtype=FP8)
    scales = []
    chunk = 2**20 if per_token else tokens
    for start in range(0, tokens, chunk):
        end = min(start + chunk, tokens)
        x = torch.randn(
            end - start, kv_heads, HEAD_DIM, generator=generator, dtype=dtypes.bf16
        )
        spread = torch.rand(end - start, kv_heads, generator=generator) * 2 - 1
        x8[start:end], scale = quantize_fp8(
            x.float() * torch.exp2(spread)[..., None], per_token
        )
        scales.append(scale)
    return x8, torch.cat(scales)


def make_fp8_inputs(layout, heads, per_token, generator):
    """(Q, FP8 K, its scales, FP8 V, its scales, indices) of a layout: make_fp8_kv K/V,
    or for CONSTANT_CASES their constant Q/K and V = 1, quantized as they are."""
    query_lens, prefix_lens, kv_heads, selection, topk = CASES[layout]
    rows, tokens = sum(query_lens), sum(query_lens) + sum(prefix_lens)
    q = torch.randn(rows, heads, HEAD_DIM, generator=generator, dtype=dtypes.bf16)
    if layout in CONSTANT_CASES:
        q_value, k_value = CONSTANT_CASES[layout][1]
        q.fill_(q_value)
        shape = (tokens, kv_heads, HEAD_DIM)
        k8, ks = quantize_fp8(torch.full(shape, k_value), per_token)
        v8, vs = quantize_fp8(torch.ones(shape), per_token)
    else:
        k8, ks = make_fp8_kv(tokens, kv_heads, per_token, generator)
        v8, vs = make_fp8_kv(tokens, kv_heads, per_token, generator)
    indices = make_indices(query_lens, prefix_lens, selection, generator, topk)
    return q, k8, ks, v8, vs, indices


def pool_scale(scale, slots, pool_rows):
    """A pool's scales: per tensor the same, per token a NaN [pool rows, HK] table holding
    each packed token's scales at its slot."""
    if scale.ndim == 1:
        return scale
    table = torch.full((pool_rows, scale.shape[1]), float("nan"), dtype=dtypes.fp32)
    table[slots.long()] = scale
    return table


def sample_rows(query_lens, count=64):
    """Each request's first and last query rows plus count evenly spaced rows."""
    total = sum(query_lens)
    edges = np.cumsum((0, *query_lens))
    rows = {*np.linspace(0, total - 1, count).astype(int).tolist()}
    rows |= {int(x) for x in (*edges[:-1], *(edges[1:] - 1)) if 0 <= x < total}
    return sorted(rows)


def fp8_error(out, expected):
    """(mean, worst) norm-relative error of out over its (row, head) vectors."""
    diff = (out.float() - expected).norm(dim=-1)
    error = diff / expected.norm(dim=-1).clamp_min(1e-6)
    return float(error.mean()), float(error.max())


def fp8_query_error(
    out, q, k8, ks, v8, vs, indices, query_lens, prefix_lens, rows=None
):
    """(mean, worst) error of compute="fp8" output against the reference on the FP8 Q it
    computes with (q: fp8_query()), which leaves it only its FP8 probability error: per
    (row, head), norms floored at their median (also for sharp scores);
    over rows (all rows by default).
    """
    expected = run_torch(
        q, k8, v8, indices, query_lens, prefix_lens, SCALE, (ks, vs), rows
    )
    if rows is not None:
        out = out[rows]
    norms = expected.norm(dim=-1)
    floor = torch.maximum(norms, norms.median())
    error = (out.float() - expected).norm(dim=-1) / floor
    return float(error.mean()), float(error.max())


def triton_fp8(q, k8, v8, k_scale, v_scale, indices, query_lens, prefix_lens):
    """aiter's Triton QSA GQA over FP8 K/V (qsa_sparse_paged_gqa, aiter #7008) on the
    packed K/V as one-token pages: (function, args, kwargs) for run_perftest."""
    lengths = [n + p for n, p in zip(query_lens, prefix_lens)]
    starts = np.cumsum((0, *lengths))
    table = torch.full((len(lengths), max(lengths)), -1, dtype=dtypes.i32)
    for request, (start, n) in enumerate(zip(starts.tolist(), lengths)):
        table[request, :n] = torch.arange(start, start + n, dtype=dtypes.i32)
    owners = torch.repeat_interleave(
        torch.arange(len(query_lens), dtype=dtypes.i32), torch.tensor(query_lens)
    )
    pages = [x.view(x.shape[0], 1, *x.shape[1:]) for x in (k8, v8)]
    planes = [
        (
            s.view(s.shape[0], 1, s.shape[1])
            if s.ndim == 2
            else s.view(1, 1, 1).expand(k8.shape[0], 1, k8.shape[1])
        )
        for s in (k_scale, v_scale)
    ]
    args = (q, *pages, indices, table, owners)
    return (
        qsa_sparse_paged_gqa,
        args,
        {
            "softmax_scale": SCALE,
            "k_scale": planes[0],
            "v_scale": planes[1],
        },
    )


@benchmark()
def test_qsa_fp8(case, heads):
    """FP8 K/V. compute="fp8": finite and within FP8_TOL of the FP32 reference on the
    dequantized K/V over sample_rows (constant cases: also O = 1 within 2e-2;
    FP8_QUERY_CHECKED: fp8_query_error instead), its union plans routing with the FP8
    kernels' calibration rather than BF16's. compute="bf16": the bits of the BF16 path
    on the K/V the caller dequantizes, within 2e-2 of the reference. A repeated call
    gives the same bits, and through a pool (unused rows and their per token scales NaN)
    the bits of the packed call on the gathered K/V and scales. FP8_PERF_CASES also time
    the BF16 path on the dequantized K/V and aiter's Triton FP8 GQA, and meet the
    fp8_query check (see test_qsa_fp8_query)."""
    layout, _, scales, compute, pool = FP8_CASES[case]
    query_lens, prefix_lens, kv_heads, _, topk = CASES[layout]
    generator = torch.Generator(device="cuda").manual_seed(fp8_seed(case, heads))
    q, k8, ks, v8, vs, indices = make_fp8_inputs(
        layout, heads, scales == "token", generator
    )
    kwargs = {
        "query_lens": query_lens,
        "prefix_lens": prefix_lens,
        "softmax_scale": SCALE,
    }
    packed = (q, k8, v8, indices), {
        **kwargs,
        "k_scale": ks,
        "v_scale": vs,
        "compute": compute,
    }
    args, kw = packed
    if pool is not None:
        pool_rows, first_page, last_page = pool
        pool_k, pool_v, slots = make_pool(
            k8, v8, query_lens, prefix_lens, pool_rows, first_page, generator, last_page
        )
        size = pool_k.shape[0]
        args = (q, pool_k, pool_v, indices)
        kw = {
            **packed[1],
            "kv_slots": slots,
            "k_scale": pool_scale(ks, slots, size),
            "v_scale": pool_scale(vs, slots, size),
        }
    msg = f"{case} H{heads}"
    flops = 4 * heads * HEAD_DIM * int((indices >= 0).sum().item())
    nbytes = 2 * q.nbytes + k8.nbytes + v8.nbytes + indices.nbytes
    ret = {
        "gfx": get_gfx(),
        "rows": q.shape[0],
        "kv_tokens": k8.shape[0],
        "kv_heads": kv_heads,
        "topk": topk,
        "scales": scales,
        "compute": compute,
        "pool GiB": None if pool is None else args[1].nbytes / 2**30,
    }
    rotate = 10 if max(q.nbytes, args[1].nbytes) < 2**30 else 1
    out, us = run_perftest(qsa_sparse_prefill, *args, **kw, num_rotate_args=rotate)
    query = fp8_query() if case in (*FP8_QUERY_CHECKED, *FP8_PERF_CASES) else None
    ret["routes"] = check_routes(case, heads)
    if compute == "fp8" and ret["routes"]["query_tile"] is not None:
        terms = plan_terms()
        assert terms == calibrated_terms(scales) != calibrated_terms(None), (
            f"{msg}: the union plan routes with {terms}, not the FP8 {scales} "
            "calibration"
        )
    ret["flydsl us"] = us
    ret["flydsl TFLOPS"] = flops / us / 1e6
    ret["flydsl TB/s"] = nbytes / us / 1e6
    differing = {
        "repeated": mismatched_bits(
            qsa_sparse_prefill(*args, **kw), out, f"{msg} repeated vs first: out"
        )
    }
    if pool is not None:
        differing["packed"] = mismatched_bits(
            qsa_sparse_prefill(*packed[0], **packed[1]),
            out,
            f"{msg} kv_slots vs packed: out",
        )
    if case in FP8_RAW_CHECKED:
        with raw_direct():
            differing["raw direct"] = mismatched_bits(
                qsa_sparse_prefill(*args, **kw), out, f"{msg} raw direct: out"
            )
    if compute == "bf16":
        kb, vb = dequantize_fp8(k8, ks), dequantize_fp8(v8, vs)
        differing["bf16 path"] = mismatched_bits(
            qsa_sparse_prefill(q, kb, vb, indices, **kwargs),
            out,
            f"{msg} vs the BF16 path on the dequantized K/V: out",
        )
        ret["flydsl err"] = checkAllclose(
            run_torch(q, kb, vb, indices, query_lens, prefix_lens, SCALE),
            out.to(dtypes.fp32),
            rtol=2e-2,
            atol=2e-2,
            msg=f"{msg}: out",
        )
    elif case in FP8_QUERY_CHECKED:
        rows = None
        if case in FP8_LARGE_Q_CASES:
            # Every row whose out bytes reach past 2 GiB (the full reference would take
            # tens of GiB), and how many of them merged residual blocks.
            first = 2**31 // (heads * HEAD_DIM * 2)
            rows = list(range(first, q.shape[0]))
            ret["merged rows past 2 GiB"] = sum(r >= first for r in residual_rows())
        ret["fp8_query mean err"], ret["fp8_query worst err"] = fp8_query_error(
            out, query, k8, ks, v8, vs, indices, query_lens, prefix_lens, rows
        )
    else:
        rows = sample_rows(query_lens)
        expected = run_torch(
            q, k8, v8, indices, query_lens, prefix_lens, SCALE, (ks, vs), rows
        )
        ret["fp8 mean err"], ret["fp8 worst err"] = fp8_error(out[rows], expected)
        if layout in CONSTANT_CASES:
            ret["flydsl err"] = checkAllclose(
                torch.ones_like(out, dtype=dtypes.fp32),
                out.to(dtypes.fp32),
                rtol=2e-2,
                atol=2e-2,
                msg=f"{msg}: out",
            )
    if case in FP8_PERF_CASES:
        kb, vb = dequantize_fp8(k8, ks), dequantize_fp8(v8, vs)
        _, ret["flydsl bf16 us"] = run_perftest(
            qsa_sparse_prefill, q, kb, vb, indices, **kwargs, num_rotate_args=rotate
        )
        fn, triton_args, triton_kwargs = triton_fp8(
            q, k8, v8, ks, vs, indices, query_lens, prefix_lens
        )
        triton_out, ret["triton fp8 us"] = run_perftest(
            fn, *triton_args, **triton_kwargs, num_rotate_args=rotate
        )
        # Only the baseline: its error is reported, not checked.
        ret["triton fp8 mean err"], ret["triton fp8 worst err"] = fp8_error(
            triton_out[rows], expected
        )
        ret["fp8_query mean err"], ret["fp8_query worst err"] = fp8_query_error(
            out, query, k8, ks, v8, vs, indices, query_lens, prefix_lens
        )
    assert bool(torch.isfinite(out).all()), f"{msg}: non-finite output"
    assert not any(differing.values()), f"{msg}: differing elements {differing}"
    if "fp8 mean err" in ret:
        mean, worst = ret["fp8 mean err"], ret["fp8 worst err"]
        assert (
            mean < FP8_TOL[0] and worst < FP8_TOL[1]
        ), f"{msg}: FP8 error mean {mean:.4f}, worst {worst:.4f}, bounds {FP8_TOL}"
    if "flydsl err" in ret:
        assert ret["flydsl err"] == 0, f"{msg}: mismatch {ret['flydsl err']}"
    if "fp8_query mean err" in ret:
        mean, worst = ret["fp8_query mean err"], ret["fp8_query worst err"]
        assert mean < 0.05 and worst < 0.2, (
            f"{msg}: error against the FP8-Q reference mean {mean:.4f}, "
            f"worst {worst:.4f}"
        )
    if "merged rows past 2 GiB" in ret:
        assert ret["merged rows past 2 GiB"], f"{msg}: no row past 2 GiB merged"
    return ret


@benchmark()
def test_qsa_fp8_sink(case, heads):
    """Every row selects an attention sink (token 0) over a flat tail whose 2047 keys
    weigh 2**-12 of it each (a third of the mass) with similar values: the tail's FP8
    probabilities must stay above the E4M3 underflow (the kernels place the row max at
    P_BIAS, not 1). Within bounds of the reference: mean 1%, worst 3%."""
    layout, _, scales = FP8_SINK_CASES[case]
    query_lens, prefix_lens, kv_heads, selection, topk = CASES[layout]
    rows, tokens = sum(query_lens), sum(query_lens) + sum(prefix_lens)
    generator = torch.Generator(device="cuda").manual_seed(fp8_seed(case, heads))
    direction = torch.randn(HEAD_DIM, generator=generator)
    direction = direction / direction.norm()
    q = (16 * direction).expand(rows, heads, HEAD_DIM).to(dtypes.bf16).contiguous()
    k = 0.1 * torch.randn(tokens, kv_heads, HEAD_DIM, generator=generator)
    # Scores in log2 units: q . k * scale * log2(e) = 12 for the sink, ~0 for the tail.
    k[0] = direction * (12 / (16 * SCALE * math.log2(math.e)))
    v = torch.randn(HEAD_DIM, generator=generator).expand(tokens, kv_heads, HEAD_DIM)
    v = v + 0.3 * torch.randn(tokens, kv_heads, HEAD_DIM, generator=generator)
    (k8, ks), (v8, vs) = (quantize_fp8(x, scales == "token") for x in (k, v))
    indices = make_indices(query_lens, prefix_lens, selection, generator, topk)
    # Rows selecting topk complete blocks without block 0 swap their first one for it.
    blocks = indices[:, : 4 * topk : 4] // 4
    missing = ~(blocks == 0).any(1) & (indices[:, 4 * topk - 1] >= 0)
    indices[missing, :4] = torch.arange(4, dtype=dtypes.i32)
    out = qsa_sparse_prefill(
        q,
        k8,
        v8,
        indices,
        query_lens=query_lens,
        prefix_lens=prefix_lens,
        softmax_scale=SCALE,
        k_scale=ks,
        v_scale=vs,
    )
    taken = check_routes(case, heads)
    sampled = sample_rows(query_lens)
    expected = run_torch(
        q, k8, v8, indices, query_lens, prefix_lens, SCALE, (ks, vs), sampled
    )
    mean, worst = fp8_error(out[sampled], expected)
    assert (
        mean < 0.01 and worst < 0.03
    ), f"{case} H{heads}: error mean {mean:.4f}, worst {worst:.4f}"
    return {
        "gfx": get_gfx(),
        "rows": rows,
        "scales": scales,
        "routes": taken,
        "mean err": mean,
        "worst err": worst,
    }


@benchmark()
def test_qsa_fp8_uniform(case, heads):
    """K = 0 weighs every selected token the same, and power-of-two V scales per token
    keep the scaled probabilities exact: the output is the selected tokens' mean V up to
    rounding (rtol 0.01 and atol 2e-3, on 96 sample rows). The V scales of the first and
    second requests reach the bounds of compute="fp8"'s per-token V scale clamp (2**56 to
    2**60, 2**-60 to 2**-56). An eighth of the (row, head) Q vectors are zero, whose FP8
    scale is 0, and an eighth tiny (amax about 2**-121, where 240 / amax overflows FP32),
    which quantize as if their amax were Q_MIN."""
    layout, _ = FP8_UNIFORM_CASES[case]
    query_lens, prefix_lens, kv_heads, selection, topk = CASES[layout]
    rows, tokens = sum(query_lens), sum(query_lens) + sum(prefix_lens)
    generator = torch.Generator(device="cuda").manual_seed(fp8_seed(case, heads))
    q = torch.randn(rows, heads, HEAD_DIM, generator=generator, dtype=dtypes.bf16)
    pick = torch.rand(rows, heads, generator=generator)
    q[pick < 1 / 8] = 0
    q[pick >= 7 / 8] *= 2.0**-123
    k8 = torch.zeros(tokens, kv_heads, HEAD_DIM, dtype=FP8)
    ks = torch.ones(tokens, kv_heads)
    v = torch.randn(tokens, kv_heads, HEAD_DIM, generator=generator) * 60
    v8 = v.clamp(-FP8_MAX, FP8_MAX).to(FP8)
    exponent = torch.randint(-3, 2, (tokens, kv_heads), generator=generator)
    lengths = [n + p for n, p in zip(query_lens, prefix_lens)]
    shifts = torch.tensor((59, -57, 0) * len(lengths))[: len(lengths)]
    exponent += shifts.repeat_interleave(torch.tensor(lengths))[:, None]
    vs = torch.exp2(exponent.float())
    indices = make_indices(query_lens, prefix_lens, selection, generator, topk)
    out = qsa_sparse_prefill(
        q,
        k8,
        v8,
        indices,
        query_lens=query_lens,
        prefix_lens=prefix_lens,
        softmax_scale=SCALE,
        k_scale=ks,
        v_scale=vs,
    )
    taken = check_routes(case, heads)
    sampled = sample_rows(query_lens, 96)
    err = checkAllclose(
        out[sampled].to(dtypes.fp32),
        run_torch(
            q, k8, v8, indices, query_lens, prefix_lens, SCALE, (ks, vs), sampled
        ),
        rtol=0.01,
        atol=2e-3,
        msg=f"{case} H{heads}: out",
    )
    assert err == 0, f"{case} H{heads}: mismatch {err}"
    return {"gfx": get_gfx(), "rows": rows, "routes": taken, "flydsl err": err}


@benchmark()
def test_qsa_fp8_query(case, heads):
    """Against the reference on the FP8 Q the kernels compute with (fp8_query) the output
    keeps only its FP8 probability error, also for sharp scores (Q x 8): fp8_query_error
    mean below 5% and worst below 20%, on core/residual union tiles. Tiny Q under K scales
    that restore its scores: Q x 2**-104 with K scales x 2**104 gives the bits of Q, and
    Q x 2**-124 with K scales x 2**124 (amax below Q_MIN, S * k_scale past FP32's range)
    stays within FP8_TOL of the reference on sample_rows."""
    layout, _ = FP8_QUERY_CASES[case]
    query_lens, prefix_lens = CASES[layout][:2]
    generator = torch.Generator(device="cuda").manual_seed(fp8_seed(case, heads))
    q, k8, ks, v8, vs, indices = make_fp8_inputs(layout, heads, True, generator)
    ret = {"gfx": get_gfx(), "rows": q.shape[0]}

    def call(query, k_scale):
        return qsa_sparse_prefill(
            query,
            k8,
            v8,
            indices,
            query_lens=query_lens,
            prefix_lens=prefix_lens,
            softmax_scale=SCALE,
            k_scale=k_scale,
            v_scale=vs,
        )

    outs = {}
    for gain in (1, 8):
        outs[gain] = call((q.float() * gain).to(dtypes.bf16), ks)
        ret["routes"] = check_routes(case, heads)
        ret[f"Q x{gain} mean err"], ret[f"Q x{gain} worst err"] = fp8_query_error(
            outs[gain], fp8_query(), k8, ks, v8, vs, indices, query_lens, prefix_lens
        )
    rows = sample_rows(query_lens)
    expected = run_torch(
        q, k8, v8, indices, query_lens, prefix_lens, SCALE, (ks, vs), rows
    )
    for exponent in (104, 124):
        name = f"Q x2**-{exponent}"
        tiny = call((q.float() * 2.0**-exponent).to(dtypes.bf16), ks * 2.0**exponent)
        ret[f"{name} mean err"], ret[f"{name} worst err"] = fp8_error(
            tiny[rows], expected
        )
        if exponent == 104:
            ret[f"{name} differing"] = mismatched_bits(
                tiny, outs[1], f"{case} H{heads} {name} vs Q: out"
            )
    for gain in (1, 8):
        mean, worst = ret[f"Q x{gain} mean err"], ret[f"Q x{gain} worst err"]
        assert (
            mean < 0.05 and worst < 0.2
        ), f"{case} H{heads} Q x{gain}: error mean {mean:.4f}, worst {worst:.4f}"
    assert ret["Q x2**-104 differing"] == 0, f"{case} H{heads}: Q x 2**-104 differs"
    mean, worst = ret["Q x2**-124 mean err"], ret["Q x2**-124 worst err"]
    assert mean < FP8_TOL[0] and worst < FP8_TOL[1], (
        f"{case} H{heads} Q x 2**-124: FP8 error mean {mean:.4f}, worst {worst:.4f}, "
        f"bounds {FP8_TOL}"
    )
    return ret


@benchmark()
def test_qsa_fp8_graph(case, heads):
    """FP8 graph capture. After an eager call at SCALE warms the layout on a stream that
    first waits for the inputs' stream, graphs captured there at SCALE and at a softmax
    scale no other case uses replay after Q, the K/V (moved to other pool pages), their
    scales, the indices and the slot table are rewritten in place: each replay gives the
    bits of the eager call on the new inputs and stays within FP8_TOL of the reference.
    With a growth layout, its first call on the stream then grows the scratch arena,
    which must keep the captured scratch, and the graphs replay again. An eager call at
    yet another new scale adds no FlyDSL specialization."""
    layout, _, scales, slotted, growth, (new_scale, eager_scale) = FP8_GRAPH_CASES[case]
    query_lens, prefix_lens = CASES[layout][:2]
    per_token = scales == "token"
    generator = torch.Generator(device="cuda").manual_seed(fp8_seed(case, heads))
    sources = [make_fp8_inputs(layout, heads, per_token, generator) for _ in range(2)]
    pool_rows = None

    def place(q, k8, ks, v8, vs, indices):
        """[Q, K, V, indices, k_scale, v_scale(, kv_slots)] of a call on these inputs:
        packed, or through a pool of pool_rows rows (on new pages each time)."""
        if not slotted:
            return [q, k8, v8, indices, ks, vs]
        pool_k, pool_v, slots = make_pool(
            k8, v8, query_lens, prefix_lens, pool_rows, 1, generator
        )
        rows = pool_k.shape[0]
        pool_scales = [pool_scale(s, slots, rows) for s in (ks, vs)]
        return [q, pool_k, pool_v, indices, *pool_scales, slots]

    def call(tensors, **kwargs):
        q, k, v, indices, k_scale, v_scale, *slots = tensors
        return qsa_sparse_prefill(
            q,
            k,
            v,
            indices,
            query_lens=query_lens,
            prefix_lens=prefix_lens,
            kv_slots=slots[0] if slots else None,
            k_scale=k_scale,
            v_scale=v_scale,
            **kwargs,
        )

    live = [tensor.clone() for tensor in place(*sources[0])]
    if slotted:
        pool_rows = live[1].shape[0]
    scales_captured = (SCALE, new_scale)
    outs = [torch.empty_like(live[0]) for _ in scales_captured]
    graphs = [torch.cuda.CUDAGraph() for _ in scales_captured]
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        call(live, out=outs[0], softmax_scale=SCALE)
    taken = check_routes(case, heads)
    with torch.cuda.stream(stream):
        for graph, out, scale in zip(graphs, outs, scales_captured):
            with torch.cuda.graph(graph, stream=stream):
                call(live, out=out, softmax_scale=scale)

    def replay(source, label):
        for tensor, new in zip(live, place(*source), strict=True):
            tensor.copy_(new)
        # The graph's stream must see this stream's rewrites first.
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for graph, out in zip(graphs, outs):
                out.fill_(float("nan"))
                graph.replay()
        stream.synchronize()
        q, k8, ks, v8, vs, indices = source
        sampled = sample_rows(query_lens)
        differing, errors = [], []
        for out, scale in zip(outs, scales_captured):
            text = f"{case} H{heads} {label} at scale {scale:.4g}"
            eager = call(live, softmax_scale=scale)
            differing.append(mismatched_bits(eager, out, f"{text}: graph vs eager"))
            expected = run_torch(
                q, k8, v8, indices, query_lens, prefix_lens, scale, (ks, vs), sampled
            )
            errors.append(fp8_error(out[sampled], expected))
        return differing, errors

    results = [replay(sources[1], "replay")]
    if growth is not None:
        generation, scratch = scratch_arena(stream)
        growth_lens, growth_prefix = CASES[growth][:2]
        q, k8, ks, v8, vs, indices = make_fp8_inputs(
            growth, heads, per_token, generator
        )
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            qsa_sparse_prefill(
                q,
                k8,
                v8,
                indices,
                query_lens=growth_lens,
                prefix_lens=growth_prefix,
                softmax_scale=SCALE,
                k_scale=ks,
                v_scale=vs,
            )
        stream.synchronize()
        assert (
            scratch_arena(stream)[0] > generation
        ), f"{growth} did not grow the {case} arena"
        # A freed buffer may still hold the right bytes, so check the allocator directly.
        assert not any(map(released, scratch)), f"{growth} freed the captured scratch"
    label = "replay after arena growth" if growth is not None else "replay"
    results.append(replay(sources[0], label))
    differing = [count for counts, _ in results for count in counts]
    errors = [error for _, pairs in results for error in pairs]
    assert not any(differing), f"{case} H{heads}: graph vs eager differ {differing}"
    for mean, worst in errors:
        assert (
            mean < FP8_TOL[0] and worst < FP8_TOL[1]
        ), f"{case} H{heads}: FP8 errors {errors}, bounds {FP8_TOL}"

    before = flydsl_variants()
    call(live, softmax_scale=eager_scale)
    after = flydsl_variants()
    assert after == before, (
        f"{case} H{heads}: softmax_scale {eager_scale} took new FlyDSL "
        f"specializations ({before} -> {after})"
    )
    return {
        "gfx": get_gfx(),
        "rows": live[0].shape[0],
        "scales": scales,
        "kv_slots": slotted,
        "growth": growth,
        "captured scales": f"{SCALE:.4g}, {new_scale:.4g}",
        "routes": taken,
        "differing elements": max(differing),
        "worst err": max(worst for _, worst in errors),
        "new specializations": None if before is None else after - before,
    }


@benchmark()
def test_qsa_fp8_workspace(case, heads):
    """Workspaces are cached per stream and layout, so their key must hold the K/V dtype,
    the FP8 scale layout, the pool and the batch mode, and every call must bind its
    tensors anew. One stream runs the layout with BF16 K/V, then FP8 K/V per tensor, per
    token, per tensor with new scale tensors, per tensor through a pool, per token with
    compute="bf16", per token with batch_invariant=True and per token again; each call
    gives the bits of the same call on a new stream. A workspace keeps none of the tensors
    of the call that built it alive (per token, and per tensor through a pool)."""
    layout, _ = FP8_WORKSPACE_CASES[case]
    query_lens, prefix_lens = CASES[layout][:2]
    generator = torch.Generator(device="cuda").manual_seed(fp8_seed(case, heads))
    q, k, v, indices = make_inputs(layout, heads, generator)
    (k8t, kst), (v8t, vst) = (quantize_fp8(x, False) for x in (k, v))
    (k8, ks), (v8, vs) = (quantize_fp8(x, True) for x in (k, v))
    pool_k, pool_v, slots = make_pool(
        k8t, v8t, query_lens, prefix_lens, None, 1, generator
    )
    tensor = {"k_scale": kst, "v_scale": vst}
    token = {"k_scale": ks, "v_scale": vs}
    calls = (
        ("BF16 K/V", (k, v), {}),
        ("FP8 per tensor", (k8t, v8t), tensor),
        ("FP8 per token", (k8, v8), token),
        (
            "FP8 per tensor, new scale tensors",
            (k8t, v8t),
            {"k_scale": kst * 2, "v_scale": vst / 2},
        ),
        ("FP8 per tensor, kv_slots", (pool_k, pool_v), {**tensor, "kv_slots": slots}),
        ('FP8 per token, compute="bf16"', (k8, v8), {**token, "compute": "bf16"}),
        (
            "FP8 per token, batch_invariant",
            (k8, v8),
            {**token, "batch_invariant": True},
        ),
        ("FP8 per token again", (k8, v8), token),
    )
    kwargs = {
        "query_lens": query_lens,
        "prefix_lens": prefix_lens,
        "softmax_scale": SCALE,
    }
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    differing = {}
    for label, kv, extra in calls:
        with torch.cuda.stream(stream):
            out = qsa_sparse_prefill(q, *kv, indices, **kwargs, **extra)
        fresh = torch.cuda.Stream()
        fresh.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(fresh):
            expected = qsa_sparse_prefill(q, *kv, indices, **kwargs, **extra)
        torch.cuda.synchronize()
        differing[label] = mismatched_bits(
            expected, out, f"{case} H{heads} {label} vs a new stream: out"
        )
    assert not any(
        differing.values()
    ), f"{case} H{heads}: calls differ from new streams {differing}"
    alive = {}
    for label, kv, extra in (calls[2], calls[4]):
        built = {"q": q, "k": kv[0], "v": kv[1], **extra}
        built = {name: x.clone() for name, x in built.items()}
        built["out"] = torch.empty_like(q)
        refs = {name: weakref.ref(x) for name, x in built.items()}
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            qsa_sparse_prefill(
                built.pop("q"),
                built.pop("k"),
                built.pop("v"),
                indices,
                **kwargs,
                **built,
            )
        torch.cuda.synchronize()
        del built
        gc.collect()
        alive[label] = [name for name, ref in refs.items() if ref() is not None]
    assert not any(
        alive.values()
    ), f"{case} H{heads}: workspaces keep caller tensors alive {alive}"
    return {"gfx": get_gfx(), "rows": q.shape[0], "calls": len(calls)}


# API.md section 1: qsa_sparse_prefill's parameters as (name, kind, default).
ENTRY_PARAMETERS = (
    *((name, inspect.Parameter.POSITIONAL_OR_KEYWORD) for name in "qkv"),
    ("indices", inspect.Parameter.POSITIONAL_OR_KEYWORD),
    ("query_lens", inspect.Parameter.KEYWORD_ONLY, None),
    ("prefix_lens", inspect.Parameter.KEYWORD_ONLY, None),
    ("softmax_scale", inspect.Parameter.KEYWORD_ONLY, None),
    ("out", inspect.Parameter.KEYWORD_ONLY, None),
    ("kv_slots", inspect.Parameter.KEYWORD_ONLY, None),
    ("batch_invariant", inspect.Parameter.KEYWORD_ONLY, False),
    ("k_scale", inspect.Parameter.KEYWORD_ONLY, None),
    ("v_scale", inspect.Parameter.KEYWORD_ONLY, None),
    ("compute", inspect.Parameter.KEYWORD_ONLY, "fp8"),
)


@benchmark()
def test_qsa_fp8_input_errors(case, heads):
    """Malformed FP8 calls raise ValueError before any launch (no FlyDSL specialization
    is added), each differing from a valid call in one input, and Q passed again as K
    with FP8 V; the entry's parameters are API.md's: names, order, keyword-only and
    defaults."""
    layout, _ = FP8_ERROR_CASES[case]
    query_lens, prefix_lens = CASES[layout][:2]
    generator = torch.Generator(device="cuda").manual_seed(fp8_seed(case, heads))
    q, k8, ks, v8, vs, indices = make_fp8_inputs(layout, heads, True, generator)
    kb, vb = dequantize_fp8(k8, ks), dequantize_fp8(v8, vs)
    pool_k, pool_v, slots = make_pool(
        k8, v8, query_lens, prefix_lens, None, 1, generator
    )
    one = torch.ones(1)
    scales = {"k_scale": ks, "v_scale": vs}
    buffer = torch.zeros(q.numel(), dtype=dtypes.bf16)
    alias = buffer.view(dtypes.fp32)[: ks.numel()].view(ks.shape)
    e4m3fn = [x.view(torch.uint8).view(torch.float8_e4m3fn) for x in (k8, v8)]
    bad = {
        "BF16 K/V with scales": ((kb, vb), {"k_scale": one, "v_scale": one}),
        "FP8 K/V without scales": ((k8, v8), {}),
        "a float k_scale": ((k8, v8), {**scales, "k_scale": 1.0}),
        "a BF16 k_scale": ((k8, v8), {**scales, "k_scale": ks.to(dtypes.bf16)}),
        "a [N - 1, HK] k_scale": ((k8, v8), {**scales, "k_scale": ks[:-1]}),
        "a [N, 2 HK] k_scale": ((k8, v8), {**scales, "k_scale": ks.repeat(1, 2)}),
        "a non-contiguous k_scale": (
            (k8, v8),
            {**scales, "k_scale": ks.repeat(1, 2)[:, :1]},
        ),
        "a host k_scale": ((k8, v8), {**scales, "k_scale": ks.cpu()}),
        "per tensor k_scale, per token v_scale": ((k8, v8), {**scales, "k_scale": one}),
        "[N, HK] scales of a [P, HK] pool": (
            (pool_k, pool_v),
            {**scales, "kv_slots": slots},
        ),
        'compute="fp16"': ((k8, v8), {**scales, "compute": "fp16"}),
        "FP8 K, BF16 V": ((k8, vb), scales),
        "float8_e4m3fn K/V": (e4m3fn, scales),
        "out overlapping k_scale": (
            (k8, v8),
            {**scales, "k_scale": alias, "out": buffer.view(q.shape)},
        ),
        "softmax_scale 1e-46": ((k8, v8), {**scales, "softmax_scale": 1e-46}),
    }
    kwargs = {"query_lens": query_lens, "prefix_lens": prefix_lens}
    # The valid call: every rejected one differs from it in one input.
    qsa_sparse_prefill(q, k8, v8, indices, **kwargs, **scales)
    before = flydsl_variants()
    accepted = []
    for label, (kv, extra) in bad.items():
        try:
            qsa_sparse_prefill(q, *kv, indices, **{**kwargs, **extra})
            accepted.append(label)
        except ValueError:
            pass
    # Q passed again as K (one BF16 tensor) with FP8 V: K must be FP8 as well.
    alias = q[:4].clone()
    try:
        qsa_sparse_prefill(
            alias,
            alias,
            torch.zeros_like(alias).to(FP8),
            make_indices((4,), (0,), "random", generator, (indices.shape[1] - 3) // 4),
            query_lens=(4,),
            prefix_lens=(0,),
            k_scale=one,
            v_scale=one,
        )
        accepted.append("Q as BF16 K with FP8 V")
    except ValueError:
        pass
    after = flydsl_variants()
    parameters = tuple(
        (
            (p.name, p.kind)
            if p.default is inspect.Parameter.empty
            else (p.name, p.kind, p.default)
        )
        for p in inspect.signature(qsa_sparse_prefill).parameters.values()
    )
    assert not accepted, f"{case} H{heads}: accepted {accepted}"
    assert after == before, (
        f"{case} H{heads}: rejected calls took FlyDSL specializations "
        f"({before} -> {after})"
    )
    assert parameters == ENTRY_PARAMETERS, (
        f"qsa_sparse_prefill's parameters {parameters} are not API.md's "
        f"{ENTRY_PARAMETERS}"
    )
    return {"gfx": get_gfx(), "rejected": len(bad), "parameters": len(parameters)}


@benchmark()
def test_qsa_fp8_batch_invariance(case, heads):
    """With batch_invariant=True every request alone gives the bits it gives batched
    with the others (as test_qsa_batch_invariance), FP8 K/V with per token scales: union
    planning and tile rows per request, no K3 routing by the others and no FP8 split of
    the union tasks (by default the first two requests alone split their key ranges). The
    batch is within FP8_TOL of the reference."""
    _, requests = FP8_BATCH_CASES[case]
    generator = torch.Generator(device="cuda").manual_seed(fp8_seed(case, heads))
    parts = []
    for rows, prefix, selection in requests:
        q = torch.randn(rows, heads, HEAD_DIM, generator=generator, dtype=dtypes.bf16)
        k8, ks = make_fp8_kv(rows + prefix, 1, True, generator)
        v8, vs = make_fp8_kv(rows + prefix, 1, True, generator)
        indices = make_indices((rows,), (prefix,), selection, generator, 512)
        parts.append((q, k8, v8, indices, ks, vs))
    query_lens = tuple(rows for rows, _, _ in requests)
    prefix_lens = tuple(prefix for _, prefix, _ in requests)
    q, k8, v8, indices, ks, vs = (torch.cat(tensors) for tensors in zip(*parts))
    kwargs = {"softmax_scale": SCALE, "batch_invariant": True}
    batched = qsa_sparse_prefill(
        q,
        k8,
        v8,
        indices,
        query_lens=query_lens,
        prefix_lens=prefix_lens,
        k_scale=ks,
        v_scale=vs,
        **kwargs,
    )
    taken = routes()
    sampled = sample_rows(query_lens)
    mean, worst = fp8_error(
        batched[sampled],
        run_torch(
            q, k8, v8, indices, query_lens, prefix_lens, SCALE, (ks, vs), sampled
        ),
    )
    differing, first, alone = [], 0, None
    for (*tensors, k_scale, v_scale), (rows, prefix, _) in zip(parts, requests):
        out = qsa_sparse_prefill(
            *tensors,
            query_lens=(rows,),
            prefix_lens=(prefix,),
            k_scale=k_scale,
            v_scale=v_scale,
            **kwargs,
        )
        if alone is None:
            # The first request alone takes the paths the batch must not change.
            alone = check_routes(case, heads)
        differing.append(
            mismatched_bits(
                out,
                batched[first : first + rows],
                f"{case} H{heads} request ({rows}, {prefix}) alone vs batched: out",
            )
        )
        first += rows
    assert taken["union"] and taken["direct"], (
        f"{case} H{heads}: the batch took {taken}; it must mix union tiles and "
        "direct rows"
    )
    assert bool(torch.isfinite(batched).all()), f"{case} H{heads}: non-finite output"
    assert (
        mean < FP8_TOL[0] and worst < FP8_TOL[1]
    ), f"{case} H{heads}: FP8 error mean {mean:.4f}, worst {worst:.4f}"
    assert not any(differing), (
        f"{case} H{heads}: requests alone differ from the batch in {differing} "
        "elements"
    )
    return {
        "gfx": get_gfx(),
        "rows": q.shape[0],
        "requests": len(requests),
        "batch routes": taken,
        "first request alone": alone,
        "mean err": mean,
        "worst err": worst,
        "differing elements": max(differing),
    }


def kv_bytes(case):
    """K/V bytes a case allocates, pools included (graph and batch invariance cases
    stay small). FP8 K/V count twice their bytes: the FP8 pack holds another copy; FP8
    cases add three times their Q bytes (Q, out, the FP8 Q and run_perftest's outputs:
    FP8_LARGE_Q_CASES pass 2 GiB)."""
    if case in SPAN_LIMIT_CASES:
        tokens, slotted = SPAN_LIMIT_CASES[case]
        return tokens * (4 if slotted else HEAD_DIM * 2)
    group = 0
    if case in FP8_CASES:
        layout, group, *_, pool = FP8_CASES[case]
        pool_rows = None if pool is None else pool[0]
    else:
        layout, pool_rows, _ = SLOT_CASES.get(case, (case, None, None))
    if layout not in CASES:
        return 0
    query_lens, prefix_lens, hk = CASES[layout][:3]
    rows = max(pool_rows or 0, sum(query_lens) + sum(prefix_lens))
    q_bytes = sum(query_lens) * group * hk * HEAD_DIM * 2
    return 2 * rows * hk * HEAD_DIM * 2 + 3 * q_bytes


def main():
    if get_gfx() not in SUPPORTED_GFX:
        aiter.logger.warning(
            "qsa_sparse_prefill unsupported on %s; skipping", get_gfx()
        )
        return

    suites = (
        (
            "functional",
            [*FUNCTIONAL_CASES, *CONSTANT_CASES, "mixed_union_raw"],
            test_qsa_sparse_attention,
        ),
        ("graph", list(GRAPH_CASES), test_qsa_graph_replay),
        ("kv_slots", list(SLOT_CASES), test_qsa_kv_slots),
        ("span limit", list(SPAN_LIMIT_CASES), test_qsa_span_limit),
        ("performance", list(PERF_CASES), test_qsa_sparse_attention),
        ("new-scale graph", list(SCALE_CASES), test_qsa_graph_new_scale),
        ("batch invariance", list(BATCH_CASES), test_qsa_batch_invariance),
        ("fp8", list(FP8_CASES), test_qsa_fp8),
        ("fp8 attention sink", list(FP8_SINK_CASES), test_qsa_fp8_sink),
        ("fp8 uniform", list(FP8_UNIFORM_CASES), test_qsa_fp8_uniform),
        ("fp8 query reference", list(FP8_QUERY_CASES), test_qsa_fp8_query),
        ("fp8 graph", list(FP8_GRAPH_CASES), test_qsa_fp8_graph),
        ("fp8 workspace", list(FP8_WORKSPACE_CASES), test_qsa_fp8_workspace),
        ("fp8 input errors", list(FP8_ERROR_CASES), test_qsa_fp8_input_errors),
        ("fp8 batch invariance", list(FP8_BATCH_CASES), test_qsa_fp8_batch_invariance),
    )
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="config input of test",
    )
    # Only names some suite runs: a layout only other rows use would run nothing.
    runnable = {case for _, cases, _ in suites for case in cases}
    all_cases = [
        case
        for case in (
            *CASES,
            *GRAPH_CASES,
            *SLOT_CASES,
            *SPAN_LIMIT_CASES,
            *SCALE_CASES,
            *BATCH_CASES,
            *FP8_LAYOUT_CASES,
            *FP8_BATCH_CASES,
        )
        if case in runnable
    ]
    parser.add_argument(
        "-c",
        "--case",
        type=str,
        nargs="*",
        choices=all_cases,
        default=all_cases,
        help="input layouts to run (functional, graph, kv_slots, span limit, "
        "performance, new-scale graph, batch invariance and FP8 cases)",
    )
    parser.add_argument(
        "-g",
        "--group",
        type=int,
        nargs="*",
        default=[12, 6, 3],
        help="Q heads per KV head (12/6/3 = Qwen3.8 flash next TP2/4/8, where HK = 1);\n"
        "cases in CASE_GROUPS run once, at their own group",
    )
    args = parser.parse_args()

    layouts = {**GRAPH_CASES, **SLOT_CASES, **SCALE_CASES, **FP8_LAYOUT_CASES}
    for suite, cases, test in suites:
        df = []
        selected = [case for case in args.case if case in cases]
        for case, group in [
            (c, g)
            for c in selected
            for g in ((CASE_GROUPS[c],) if c in CASE_GROUPS else args.group)
        ]:
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
