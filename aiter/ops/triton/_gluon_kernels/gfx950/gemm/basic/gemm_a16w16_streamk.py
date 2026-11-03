# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Gluon BF16/FP16 GEMM for gfx950: persistent data-parallel tiles and a Stream-K tail.

``y = x @ w.T (+ bias)`` with x (M, K) and w (N, K) both K-contiguous, y (M, N) row-major.

The kernel. One program per CU with four waves, one per SIMD. A BLOCK_M x BLOCK_N
output tile (sides 64, 128 or 256; BLOCK_K 64 or 128) is split into four quadrant
accumulators pinned to AGPRs (``cd_regclass="a"``), fed by a 2-deep async-copy LDS
pipeline with local prefetch and the K loop unrolled by 2; every layout is derived
from the tile. Tiles are assigned through an XCD-aware remap and a GROUP_SIZE_M
swizzle. The instruction schedule comes from Triton's MFMA scheduler, enabled per
launch with ``schedule_hint="mfma-schedule"``; ``AITER_MFMA_SCHED=0`` turns it off for
A/B runs, which costs about 15%. It needs an upstream Triton newer than
triton-lang/triton#12209 (``3b0c7f081``), which brought the scheduler;
``mfma(..., cd_regclass=)`` (triton-lang/triton#11792) was already there.

Shapes. Ragged shapes are specialized at compile time: a ragged M or N clamps the
loads of the last tile to the matrix edge and masks its stores; a ragged K (EVEN_K
false) is virtually padded to a whole number of K-step pairs (K_PAD). Each tile runs
its last pair first (on a ragged K the masked physical tail), then all remaining
pairs from aligned column zero, so the hot loop stays unmasked; no tensor is
physically padded. Leading strides must be multiples of 8 (16-byte loads and stores),
N a multiple of 8, K a multiple of 16 and K_PAD at least 4 BLOCK_K
(``unsupported_reason``).

Configuration. The caller supplies the tile (DEFAULT_TILE, 256x256x64, when it does
not); the plan is either given too (a tuned config) or left to the default plan
below. The launcher derives only shape- and topology-dependent constants::

    {"BLOCK_M": 256, "BLOCK_N": 256, "BLOCK_K": 64, "GROUP_SIZE_M": 4,
     "STREAMK_POLICY": "one_tile", "STREAMK_FIXUP": "rs", "SPLIT_N": 8,
     "STREAMK_EXCHANGE": "wt"}
    {"BLOCK_M": 256, "BLOCK_N": 256, "BLOCK_K": 64}       # default plan, GROUP_SIZE_M 4

P is the resident program count (one per CU), T the output tile count, S = T % P the
tail tiles left after the last full wave, and k the K-pairs per tile (a pair is two
BLOCK_K steps; Stream-K splits tiles in whole pairs). The pictures use P = 8 and
k = 8: [..] is one program's work, each letter one K-pair of a tile, and | a tile
boundary inside one program's work.

STREAMK_POLICY, how the tail tiles run. Full waves always run whole tiles in a
persistent loop, each tile's epilogue prefetching the program's next tile.

    "dp"        every tile whole on one program (T = 10, S = 2):
                    wave 0  [t0][t1][t2][t3][t4][t5][t6][t7]
                    wave 1  [t8][t9] .   .   .   .   .   .     6 programs idle
    "one_tile"  each tail tile's pairs cut evenly over SPLIT_N programs
                (T = 10, S = 2, SPLIT_N = 4, S * SPLIT_N = 8 programs):
                    wave 0  as dp
                    tail    t8: [AA][AA][AA][AA]   t9: [BB][BB][BB][BB]
    "spread"    S > P / 2 only: the tail's S * k pairs laid end to end and cut
                evenly over all P programs (S = 5):
                    [AAAAA][AAA|BB][BBBBB][B|CCCC][CCCC|D][DDDDD][DD|EEE][EEEEE]
    "two_tile"  as spread, over the last P + S tiles (needs a full wave to borrow).

    spread and two_tile run each program's pieces last to first, so an XCD's
    programs all start at pair 0 and share A and B in L2, and each publishes its
    partial before it waits for the tile it owns.

STREAMK_FIXUP, how one_tile's SPLIT_N partial sums are combined (spread and
two_tile always use the owner).

    "owner"  the program with a tile's first pairs owns it; the others store fp32
             partials and raise a flag; the owner adds them one after another:
                 p1, p2, p3 --partial--> p0 --> C tile
             Fewest bytes, but n - 1 serial reads: suits small SPLIT_N.
    "rs"     reduce-scatter: every program stores its partial, all meet at the
             tile's barrier, then program i sums slice i of every partial:
                 p0 p1 p2 p3 store | barrier | C tile: [ p0 | p1 | p2 | p3 ]
             Programs the tail leaves idle read slices too: R = max(min(blocks,
             pow2_floor(P / S)), own readers) per tile, with each tile's barrier
             in its own cache line. The ending costs about the same at any
             SPLIT_N: suits large SPLIT_N.

STREAMK_EXCHANGE, where partials and flags travel.

    "wt"  written through to memory; no placement constraint.
    "l2"  one_tile with rs only: all of a tile's programs on one XCD, partials and
          flags kept in its L2. Needs SPLIT_N <= (P / NUM_XCDS) / ceil(S / NUM_XCDS).

SPLIT_N, programs per tail tile: an integer 2 <= SPLIT_N <= min(P // S, k) for
one_tile, None for the other policies.

GROUP_SIZE_M, rows of tiles walked together by the persistent loop's swizzle.

Default plan. A config may name only the tile (BLOCK_M, BLOCK_N, BLOCK_K, and
optionally GROUP_SIZE_M, default 4); streamk_plan then supplies STREAMK_POLICY,
STREAMK_FIXUP and SPLIT_N, with the "wt" exchange. With m = P // S programs per
tail tile, pairs = ceil(K / (2 BLOCK_K)) K-pairs per tile and cap = min(m, pairs):

    dp              if S == 0 or cap < 2
    owner           if m <= 3:  n = sqrt(pairs / 1.5),    dp if pairs < 10
    reduce-scatter  otherwise:  n = sqrt(pairs * m / 6),  dp if pairs * (1 - 1/n) < 6

n is clamped to cap and rounded to the nearest of OWNER_SPLITS / RS_SPLITS. The
reduce-scatter's n balances each program's MAC (pairs / n) against the partials all
S n programs store and read at once; the owner reads its n - 1 partials one after
another, so its n grows only with sqrt(pairs). The same constants (STREAMK_PLAN)
hold for every tile spec, below one wave and after it. They, OWNER_SPLITS,
RS_SPLITS, DEFAULT_TILE and the default GROUP_SIZE_M live in
configs/gfx950/gluon/gemm/gemm_a16w16_streamk/DEFAULT.json (streamk_defaults).

Tile order. The persistent tiles are the first T - S tile ids, with the XCD remap
and GROUP_SIZE_M swizzle applied over those ids alone, and the Stream-K tiles are the
last S ids, which keeps the tail's mapping separate from the persistent one. The
last persistent trip prefetches the program's first Stream-K pair, which is never
slower than discarding it.

Every program of the Stream-K phase must be resident at once, because the endings
wait on each other: the grid is at most one program per CU. The flags live in one
buffer per (device, stream) and every launch leaves them re-armed, so every path is
safe under CUDA graph replay.

This module holds the kernel, its configuration and plan, and imports no torch; the
workspace, tensor checks and launch are in aiter/ops/triton/gemm/basic/gemm_a16w16.py
(``kernel_type="streamk"``).
"""

import functools
import math
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass

import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.amd import cdna4 as ttgl_cdna4
from triton.runtime.jit import constexpr_function

from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr
from aiter.ops.triton.utils.config_utils import load_config_json, resolve_config_dir
from aiter.ops.triton.utils.device_info import get_num_sms, get_num_xcds

NUM_WARPS = 4


@constexpr_function
def compute_gload_layout(shared_layout, load_contig, num_warps, threads_per_warp=64):
    """The coalesced global-load linear layout that matches a PaddedSharedLayout.

    Partitions the shared layout's offset bases into register, lane and warp bases so
    that each warp writes contiguous LDS offsets (what CoalesceAsyncCopy derives for a
    padded encoding), following BLOCK_M, BLOCK_N and BLOCK_K.
    """
    bases = [list(b) for b in shared_layout.offset_bases]
    rank = len(shared_layout.shape)
    n_reg = int(math.log2(load_contig))
    n_lane = int(math.log2(threads_per_warp))
    n_warp = int(math.log2(num_warps))
    reg = bases[:n_reg]
    lane = bases[n_reg : n_reg + n_lane]
    warp = bases[n_reg + n_lane : n_reg + n_lane + n_warp]
    while len(warp) < n_warp:  # zero-pad (broadcast) if we ran out of bases
        warp.append([0] * rank)
    reg += bases[n_reg + n_lane + n_warp :]
    return gl.DistributedLinearLayout(
        reg_bases=reg,
        lane_bases=lane,
        warp_bases=warp,
        block_bases=[],
        shape=list(shared_layout.shape),
    )


@constexpr_function
def coalesced_store_layout(rows, cols, elem_bytes, num_warps, threads_per_warp=64):
    """The row-major store layout of a rows x cols block that Triton's coalescing
    derives: up to 16 contiguous bytes per thread (no more than an even share of the
    block, so no thread repeats another's elements), the lanes across the columns
    and then down the rows, the warps down the rows."""
    per_thread = max(rows * cols // (num_warps * threads_per_warp), 1)
    contig = min(16 // elem_bytes, cols, per_thread)
    lanes_n = min(cols // contig, threads_per_warp)
    return gl.BlockedLayout(
        [1, contig], [threads_per_warp // lanes_n, lanes_n], [num_warps, 1], [1, 0]
    )


@gluon.jit
def even_split_start(i, TOTAL, PARTS: gl.constexpr):
    """First item of part i when TOTAL items are split over PARTS parts as evenly as
    possible: every part gets TOTAL // PARTS items, the first TOTAL % PARTS one more.
    Part i covers [even_split_start(i), even_split_start(i + 1)).

    TOTAL = 10, PARTS = 4 (base 2, extra 2; b = base item, + = extra item):

        item:   0  1  2 | 3  4  5 | 6  7 | 8  9
                b  b  + | b  b  + | b  b | b  b
        part:   0         1         2      3
        start:  0         3         6      8     = i * 2 + min(i, 2)
    """
    return i * (TOTAL // PARTS) + gl.minimum(i, TOTAL % PARTS)


@gluon.jit
def even_split_size(i, TOTAL, PARTS: gl.constexpr):
    """Items in part i of even_split_start's split."""
    return TOTAL // PARTS + (i < TOTAL % PARTS).to(gl.int32)


@gluon.jit
def ragged_transpose(i, count, PARTS: gl.constexpr):
    """Id i of count, read row-major from a (local, part) grid of PARTS columns
    (part i % PARTS, local i // PARTS), renumbered column-major so that each part's
    ids form one contiguous block of even_split_start's split. The grid is ragged
    when PARTS does not divide count; the result is a permutation of [0, count).

    The main use is XCD placement: hardware runs program p on XCD p % NUM_XCDS, so
    PARTS = NUM_XCDS gives each XCD one contiguous block of ids.

    count = 10, PARTS = NUM_XCDS = 4 (xcd = i % 4, local = i // 4):

                  local 0  local 1  local 2
        xcd 0:    i 0      i 4      i 8      -> 0 1 2   (start 0)
        xcd 1:    i 1      i 5      i 9      -> 3 4 5   (start 3)
        xcd 2:    i 2      i 6               -> 6 7     (start 6)
        xcd 3:    i 3      i 7               -> 8 9     (start 8)
    """
    return even_split_start(i % PARTS, count, PARTS) + i // PARTS


@gluon.jit
def group_m_tile_coords(
    tile,
    M,
    N,
    BM: gl.constexpr,
    BN: gl.constexpr,
    GROUP_SIZE_M: gl.constexpr,
):
    """(m, n) coordinates in the tile grid of linear tile id tile, under the
    GROUP_SIZE_M swizzle (GROUP_SIZE_M rows of tiles walked column by column)."""
    num_pid_m = gl.cdiv(M, BM)
    num_pid_n = gl.cdiv(N, BN)

    if GROUP_SIZE_M == 1:
        pid_m = tile // num_pid_n
        pid_n = tile % num_pid_n
    else:
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = tile // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((tile % num_pid_in_group) % group_size_m)
        pid_n = (tile % num_pid_in_group) // group_size_m

    return pid_m, pid_n


@gluon.jit
def vpid_tile_coords(
    vpid,
    M,
    N,
    BM: gl.constexpr,
    BN: gl.constexpr,
    GRID_MN,
    NUM_XCDS: gl.constexpr,
    GROUP_SIZE_M: gl.constexpr,
):
    """(m, n) coordinates of the tile that persistent-loop virtual pid vpid
    (program id + i NUM_PROGRAMS, < GRID_MN) computes. Its tile id is its XCD-blocked
    rank among the GRID_MN virtual pids (ragged_transpose; vpid stays on its
    program's XCD since NUM_XCDS divides NUM_PROGRAMS), then group_m_tile_coords."""
    tile = vpid
    if NUM_XCDS != 1:
        tile = ragged_transpose(vpid, GRID_MN, NUM_XCDS)
    return group_m_tile_coords(tile, M, N, BM, BN, GROUP_SIZE_M)


_gemm_a16w16_streamk_repr = make_kernel_repr(
    "_gemm_a16w16_streamk_kernel",
    [
        "BLOCK_M",
        "BLOCK_N",
        "BLOCK_K",
        "GROUP_SIZE_M",
        "STREAMK_TILES",
        "SPLIT_N",
        "REDUCE_SCATTER",
        "END_TO_END",
        "L2",
    ],
)


@gluon.jit(repr=_gemm_a16w16_streamk_repr)
def _gemm_a16w16_streamk_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    bias_ptr,
    p_ptr,
    locks_ptr,
    sync_ptr,
    M,
    N,
    K: gl.constexpr,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_K: gl.constexpr,
    NUM_PROGRAMS: gl.constexpr,
    NUM_XCDS: gl.constexpr,
    GROUP_SIZE_M: gl.constexpr,
    ADD_BIAS: gl.constexpr,
    EVEN_M: gl.constexpr,
    EVEN_N: gl.constexpr,
    EVEN_K: gl.constexpr,
    STREAMK_TILES: gl.constexpr,
    SPLIT_N: gl.constexpr,
    REDUCE_SCATTER: gl.constexpr,
    RS_NB: gl.constexpr,
    RS_U: gl.constexpr,
    RS_R: gl.constexpr,
    END_TO_END: gl.constexpr,
    L2: gl.constexpr = False,
    HAS_FULL: gl.constexpr = True,
    GRID: gl.constexpr = 0,
    OWNER_QPB: gl.constexpr = 1,
    RS_E: gl.constexpr = 0,
    BAR_STRIDE: gl.constexpr = 0,
):
    """
    Persistent full tiles followed by a uniform or end-to-end Stream-K tail, on a
    tile sliced in both M and N (4 quadrant accumulators, K loop unrolled by 2,
    pre-computed load offsets, XCD-aware PID remapping + GROUP_SIZE_M swizzling):

    1. A persistent loop over the first total_full_tiles = total_tiles - STREAMK_TILES
       tile ids, XCD-remapped over those ids alone: each tile's epilogue prefetches
       the first K-step pair of the program's next tile. The last trip's epilogue
       prefetches the first pair of the program's first Stream-K segment, which then
       skips its prologue (on a ragged K masked off when the program has no segment;
       on an even K the idle program's target is in bounds and never read). Without a
       Stream-K tail it loads the last pair of tile total_full_tiles - 1, which is
       discarded.
    2. A Stream-K tail over the last STREAMK_TILES tile ids, split in units of a
       K-step pair, whose partials are combined by an owner or a reduce-scatter
       ending. Every program of the Stream-K phase must be resident at once: the
       endings wait on each other.
    """
    start = gl.program_id(axis=0)
    num_tiles_m = gl.cdiv(M, BLOCK_M)
    num_tiles_n = gl.cdiv(N, BLOCK_N)
    total_tiles = num_tiles_m * num_tiles_n
    total_full_tiles = total_tiles - STREAMK_TILES
    strides = (stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn)

    # Every layout follows the tile: the shared layouts come from the dot-operand
    # layouts and the global-load layouts from the shared layouts.
    mfmaLayout: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4, instr_shape=[16, 16, 32], transposed=True, warps_per_cta=[2, 2]
    )
    dotOpLayoutA: gl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=mfmaLayout, k_width=8
    )
    dotOpLayoutB: gl.constexpr = gl.DotOperandLayout(
        operand_index=1, parent=mfmaLayout, k_width=8
    )
    sharedLayoutA: gl.constexpr = ttgl_cdna4.compute_efficient_padded_shared_layout(
        dotOpLayoutA, [BLOCK_M // 2, BLOCK_K], a_ptr.dtype.element_ty
    )
    sharedLayoutB: gl.constexpr = ttgl_cdna4.compute_efficient_padded_shared_layout(
        dotOpLayoutB, [BLOCK_K, BLOCK_N // 2], b_ptr.dtype.element_ty
    )
    gLoadLayoutA: gl.constexpr = compute_gload_layout(sharedLayoutA, 8, gl.num_warps())
    gLoadLayoutB: gl.constexpr = compute_gload_layout(sharedLayoutB, 8, gl.num_warps())

    nBuffers: gl.constexpr = 2
    smemA_top = gl.allocate_shared_memory(
        a_ptr.dtype.element_ty, [nBuffers, BLOCK_M // 2, BLOCK_K], sharedLayoutA
    )
    smemA_bot = gl.allocate_shared_memory(
        a_ptr.dtype.element_ty, [nBuffers, BLOCK_M // 2, BLOCK_K], sharedLayoutA
    )
    smemB_left = gl.allocate_shared_memory(
        b_ptr.dtype.element_ty, [nBuffers, BLOCK_K, BLOCK_N // 2], sharedLayoutB
    )
    smemB_right = gl.allocate_shared_memory(
        b_ptr.dtype.element_ty, [nBuffers, BLOCK_K, BLOCK_N // 2], sharedLayoutB
    )
    smem = (smemA_top, smemA_bot, smemB_left, smemB_right)

    offs_am = gl.arange(0, BLOCK_M // 2, gl.SliceLayout(1, gLoadLayoutA))
    offs_ak = gl.arange(0, BLOCK_K, gl.SliceLayout(0, gLoadLayoutA))

    offs_bn = gl.arange(0, BLOCK_N // 2, gl.SliceLayout(0, gLoadLayoutB))
    offs_bk = gl.arange(0, BLOCK_K, gl.SliceLayout(1, gLoadLayoutB))

    # Row and column terms of the offsets. The row strides are multiples of 8
    # elements (the wrapper checks), which keeps every 8-element group of a row
    # 16-byte aligned; the hint says so, because Triton's argument specialization
    # only records strides that are multiples of 16.
    a_rows_top = gl.multiple_of(offs_am * stride_am, 8)
    b_cols_left = gl.multiple_of(offs_bn * stride_bn, 8)

    # One half-tile offset tensor per operand, shared by both halves: the bottom /
    # right half (a_bot_shift / b_right_shift) and the odd K-step (BLOCK_K *
    # stride_ak / stride_bk) are shifts in the scalar base; a ragged tile replaces
    # them with its clamped offsets (a_load_addressing). Offsets are relative to the
    # tile, whose base is 64-bit.
    a_offsets = a_rows_top[:, None] + offs_ak[None, :] * stride_ak
    b_offsets = offs_bk[:, None] * stride_bk + b_cols_left[None, :]

    a_bot_shift = BLOCK_M // 2 * stride_am
    b_right_shift = BLOCK_N // 2 * stride_bn
    kofs = (a_offsets, a_bot_shift, b_offsets, b_right_shift)

    # A ragged K is virtually padded to whole K-step pairs; every tile runs its last
    # pair first (on a ragged K partial and masked), so the other pairs stay aligned to
    # the row's start.
    if EVEN_K:
        gl.static_assert(K % (2 * BLOCK_K) == 0)
        K_PAD: gl.constexpr = K
    else:
        K_PAD: gl.constexpr = gl.cdiv(K, 2 * BLOCK_K) * (2 * BLOCK_K)
    K_RES: gl.constexpr = K_PAD - K
    iterMax = gl.cdiv(K_PAD, BLOCK_K)

    gl.assume(iterMax > 3)
    PAIRS_PER_TILE: gl.constexpr = K_PAD // (2 * BLOCK_K)
    SK_PREFETCH: gl.constexpr = HAS_FULL and STREAMK_TILES > 0

    # Persistent phase. Pair zero is already in flight on entry to each tile;
    # its epilogue prefetches pair zero of the next virtual tile.
    if HAS_FULL:
        # The last trip's prefetch target: the first pair of the program's first
        # Stream-K segment, as quadrant_segment's prologue loads it.
        sk_target = None
        if SK_PREFETCH:
            sk_has, sk_tile, sk_k0 = streamk_first_segment(
                total_full_tiles,
                STREAMK_TILES,
                PAIRS_PER_TILE,
                SPLIT_N,
                END_TO_END,
                L2,
                NUM_PROGRAMS,
                NUM_XCDS,
            )
            sk_m, sk_n = group_m_tile_coords(
                sk_tile, M, N, BLOCK_M, BLOCK_N, GROUP_SIZE_M
            )
            sk_col = gl.where(
                sk_k0 == 0, K_PAD - 2 * BLOCK_K, (sk_k0 - 2) * BLOCK_K
            ).to(gl.int64)
            sk_khi = gl.where(sk_k0 == 0, 2 * BLOCK_K - K_RES, 2 * BLOCK_K) * sk_has
            sk_target = (sk_m, sk_n, sk_col, sk_khi)
        first_m, first_n = vpid_tile_coords(
            start,
            M,
            N,
            BLOCK_M,
            BLOCK_N,
            total_full_tiles,
            NUM_XCDS,
            GROUP_SIZE_M,
        )
        a_first = a_ptr + first_m.to(gl.int64) * BLOCK_M * stride_am
        b_first = b_ptr + first_n.to(gl.int64) * BLOCK_N * stride_bn
        a_addr = a_load_addressing(
            first_m,
            a_offsets,
            a_bot_shift,
            M,
            stride_am,
            stride_ak,
            BLOCK_M,
            BLOCK_K,
            gLoadLayoutA,
            EVEN_M,
        )
        b_addr = b_load_addressing(
            first_n,
            b_offsets,
            b_right_shift,
            N,
            stride_bk,
            stride_bn,
            BLOCK_N,
            BLOCK_K,
            gLoadLayoutB,
            EVEN_N,
        )
        if EVEN_K:
            mask_a0, mask_b0, mask_a1, mask_b1 = None, None, None, None
        else:
            mask_a0, mask_b0, mask_a1, mask_b1 = tail_pair_masks(
                2 * BLOCK_K - K_RES,
                K_RES,
                BLOCK_M,
                BLOCK_N,
                BLOCK_K,
                gLoadLayoutA,
                gLoadLayoutB,
                True,
            )
        a_top, b_left = load_first_pair(
            a_first + (K_PAD - 2 * BLOCK_K) * stride_ak,
            b_first + (K_PAD - 2 * BLOCK_K) * stride_bk,
            smem,
            a_addr,
            b_addr,
            dotOpLayoutA,
            dotOpLayoutB,
            mask_a0,
            mask_b0,
            mask_a1,
            mask_b1,
        )
        tile_addr = (a_addr, b_addr)
        for vpid in range(start, total_full_tiles, NUM_PROGRAMS):
            a_top, b_left, tile_addr = persistent_tile(
                vpid,
                a_top,
                b_left,
                tile_addr,
                a_ptr,
                b_ptr,
                c_ptr,
                bias_ptr,
                smem,
                kofs,
                strides,
                M,
                N,
                K_PAD,
                BLOCK_M,
                BLOCK_N,
                BLOCK_K,
                NUM_PROGRAMS,
                NUM_XCDS,
                GROUP_SIZE_M,
                ADD_BIAS,
                mfmaLayout,
                gLoadLayoutA,
                gLoadLayoutB,
                EVEN_M,
                EVEN_N,
                EVEN_K,
                K_RES,
                total_full_tiles,
                sk_target,
                SK_PREFETCH,
            )

    # The Stream-K phase picks up the last trip's prefetch.
    if not SK_PREFETCH:
        gl.amd.cdna4.async_copy.wait_group(0)
    if not HAS_FULL:
        a_top = 0
        b_left = 0

    if END_TO_END and STREAMK_TILES > 0:
        # Two-tile and spread always use the full resident grid.
        rank, sk_pair_start, seg_end = e2e_pair_range(
            STREAMK_TILES, PAIRS_PER_TILE, NUM_PROGRAMS, NUM_XCDS
        )
        # Segments run last to first: each program starts at k-step 0 of its last
        # tile, so an XCD's programs stay in step and share A / B in L2 (forward
        # order drops two-tile's hits from 68% to 9%), and it publishes its partial
        # before the owned tile's wait.
        if SK_PREFETCH:
            # The first segment, whose first pair the last persistent trip loaded.
            sk_t, seg_start, k0, is_owner, n_peers = e2e_segment_ending_at(
                seg_end,
                sk_pair_start,
                rank,
                STREAMK_TILES,
                PAIRS_PER_TILE,
                NUM_PROGRAMS,
            )
            quadrant_segment(
                total_full_tiles + sk_t,
                k0,
                seg_end - seg_start,
                1 - is_owner,
                n_peers,
                rank,
                0,
                0,
                0,
                a_ptr,
                b_ptr,
                c_ptr,
                bias_ptr,
                p_ptr,
                locks_ptr,
                sync_ptr,
                smem,
                kofs,
                strides,
                M,
                N,
                BLOCK_M,
                BLOCK_N,
                BLOCK_K,
                GROUP_SIZE_M,
                ADD_BIAS,
                -1,
                False,
                1,
                1,
                False,
                NUM_PROGRAMS,
                NUM_XCDS,
                mfmaLayout,
                gLoadLayoutA,
                gLoadLayoutB,
                EVEN_M,
                EVEN_N,
                EVEN_K,
                K_PAD,
                K_RES,
                True,
                OWNER_QPB,
                0,
                True,
                a_top,
                b_left,
            )
            seg_end = seg_start
        while seg_end > sk_pair_start:
            sk_t, seg_start, k0, is_owner, n_peers = e2e_segment_ending_at(
                seg_end,
                sk_pair_start,
                rank,
                STREAMK_TILES,
                PAIRS_PER_TILE,
                NUM_PROGRAMS,
            )
            quadrant_segment(
                total_full_tiles + sk_t,
                k0,
                seg_end - seg_start,
                1 - is_owner,
                n_peers,
                rank,
                0,
                0,
                0,
                a_ptr,
                b_ptr,
                c_ptr,
                bias_ptr,
                p_ptr,
                locks_ptr,
                sync_ptr,
                smem,
                kofs,
                strides,
                M,
                N,
                BLOCK_M,
                BLOCK_N,
                BLOCK_K,
                GROUP_SIZE_M,
                ADD_BIAS,
                -1,
                False,
                1,
                1,
                False,
                NUM_PROGRAMS,
                NUM_XCDS,
                mfmaLayout,
                gLoadLayoutA,
                gLoadLayoutB,
                EVEN_M,
                EVEN_N,
                EVEN_K,
                K_PAD,
                K_RES,
                True,
                OWNER_QPB,
                0,
            )
            seg_end = seg_start
    elif STREAMK_TILES > 0:
        # Uniform tile-aligned split: every tile has exactly SPLIT_N programs.
        WORKING: gl.constexpr = STREAMK_TILES * SPLIT_N
        NBLK: gl.constexpr = BLOCK_M * BLOCK_N // 1024
        gl.static_assert(SPLIT_N <= PAIRS_PER_TILE)
        # The wt reduce-scatter reads with RS_R = SPLIT_N + RS_E programs when RS_E
        # idle programs join each tile (rs_readers); otherwise with its own.
        RS_ALL: gl.constexpr = REDUCE_SCATTER and not L2
        if RS_ALL and RS_E > 0:
            gl.static_assert(RS_R == SPLIT_N + RS_E and RS_R <= NBLK)
        else:
            gl.static_assert(RS_R > 0 and RS_R <= SPLIT_N)
            gl.static_assert(
                2 * RS_R > SPLIT_N or RS_R == NBLK,
                "RS_R does not match SPLIT_N",
            )
        gl.static_assert((RS_R & (RS_R - 1)) == 0)
        gl.static_assert(RS_NB == 1 or RS_NB * RS_R <= NBLK)
        if L2:
            L2_GRID: gl.constexpr = NUM_PROGRAMS if GRID == 0 else GRID
            gl.static_assert(L2_GRID % NUM_XCDS == 0)
            rank = xcd_blocked_rank(L2_GRID, NUM_XCDS)
            t, j, n_tile = xcd_local_chunk_of(
                rank,
                STREAMK_TILES,
                L2_GRID // NUM_XCDS,
                NUM_XCDS,
                SPLIT_N,
            )
            active = n_tile > 0
        else:
            # A compact launch with extra readers ranks its producers as a full
            # grid does: pids from WORKING are the readers.
            if GRID > 0 and RS_E == 0:
                rank = xcd_blocked_rank(GRID, NUM_XCDS)
            else:
                rank = xcd_blocked_rank(WORKING, NUM_XCDS, IDLE=True)
            active = rank < WORKING
            t = rank // SPLIT_N
            j = rank % SPLIT_N
            n_tile = SPLIT_N
        if active:
            is_partial = (j != 0).to(gl.int32)
            n_peers = (SPLIT_N - 1) * (1 - is_partial)
            OWNER_MAX_PEERS: gl.constexpr = SPLIT_N - 1 if SPLIT_N <= 8 else 0
            quadrant_segment(
                total_full_tiles + t,
                even_split_start(j, PAIRS_PER_TILE, SPLIT_N) * 2,
                even_split_size(j, PAIRS_PER_TILE, SPLIT_N),
                is_partial,
                n_peers,
                rank,
                j,
                n_tile,
                RS_R,
                a_ptr,
                b_ptr,
                c_ptr,
                bias_ptr,
                p_ptr,
                locks_ptr,
                sync_ptr,
                smem,
                kofs,
                strides,
                M,
                N,
                BLOCK_M,
                BLOCK_N,
                BLOCK_K,
                GROUP_SIZE_M,
                ADD_BIAS,
                1,
                REDUCE_SCATTER,
                RS_NB,
                RS_U,
                L2,
                NUM_PROGRAMS,
                NUM_XCDS,
                mfmaLayout,
                gLoadLayoutA,
                gLoadLayoutB,
                EVEN_M,
                EVEN_N,
                EVEN_K,
                K_PAD,
                K_RES,
                False,
                OWNER_QPB,
                OWNER_MAX_PEERS,
                SK_PREFETCH,
                a_top,
                b_left,
                RS_E,
                BAR_STRIDE,
            )
        if RS_ALL and RS_E > 0:
            # Programs without a segment (hardware pids from WORKING) read too.
            e = gl.program_id(axis=0) - WORKING
            if (e >= 0) & (e < STREAMK_TILES * RS_E):
                # The last trip's prefetch (unmasked on an even K) may still be in
                # flight into the LDS the slice's layout conversions use.
                gl.amd.cdna4.async_copy.wait_group(0)
                rs_extra_reader(
                    e // RS_E,
                    SPLIT_N + e % RS_E,
                    total_full_tiles,
                    p_ptr,
                    bias_ptr,
                    c_ptr,
                    sync_ptr,
                    M,
                    N,
                    stride_cm,
                    stride_cn,
                    BLOCK_M,
                    BLOCK_N,
                    GROUP_SIZE_M,
                    SPLIT_N,
                    RS_R,
                    RS_E,
                    RS_NB,
                    RS_U,
                    ADD_BIAS,
                    EVEN_M,
                    EVEN_N,
                    BAR_STRIDE,
                )


# ---------------------------------------------------------------------------------
# The pipeline as functions, shared by the persistent tiles and the Stream-K
# segments.
#
# accs is (acc_tl, acc_bl, acc_tr, acc_br) and smem is (smemA_top, smemA_bot,
# smemB_left, smemB_right). An operand's load addressing is a_addr =
# (a_top_offsets, a_bot_offsets, a_bot_shift, a_odd_shift) and b_addr =
# (b_left_offsets, b_right_offsets, b_right_shift, b_odd_shift) (a_load_addressing /
# b_load_addressing): each half loads from base + shift + offsets, the shifts living
# in the scalar base.
# ---------------------------------------------------------------------------------


@gluon.jit
def clamped_a_offsets(
    m_last,
    stride_am,
    stride_ak,
    BLOCK_M: gl.constexpr,
    BLOCK_K: gl.constexpr,
    gLoadLayoutA: gl.constexpr,
):
    """A's top and bottom load offsets on a ragged tile, relative to the tile. Rows past
    m_last (the tile's last row inside the matrix) are clamped to it: the clamped rows
    load valid data (duplicates), compute rows nobody keeps, and the masked store drops
    them. Clamping keeps the buffer offsets inside one tile's range."""
    offs_am = gl.arange(0, BLOCK_M // 2, gl.SliceLayout(1, gLoadLayoutA))
    offs_ak = gl.arange(0, BLOCK_K, gl.SliceLayout(0, gLoadLayoutA))
    rows_top = gl.minimum(offs_am, m_last)
    rows_bot = gl.minimum(offs_am + BLOCK_M // 2, m_last)
    a_rows_top = gl.multiple_of(rows_top * stride_am, 8)
    a_rows_bot = gl.multiple_of(rows_bot * stride_am, 8)
    a_top_offsets = a_rows_top[:, None] + offs_ak[None, :] * stride_ak
    a_bot_offsets = a_rows_bot[:, None] + offs_ak[None, :] * stride_ak
    return a_top_offsets, a_bot_offsets


@gluon.jit
def clamped_b_offsets(
    n_last,
    stride_bk,
    stride_bn,
    BLOCK_N: gl.constexpr,
    BLOCK_K: gl.constexpr,
    gLoadLayoutB: gl.constexpr,
):
    """B's left and right load offsets on a ragged tile, columns past n_last clamped
    (clamped_a_offsets)."""
    offs_bn = gl.arange(0, BLOCK_N // 2, gl.SliceLayout(0, gLoadLayoutB))
    offs_bk = gl.arange(0, BLOCK_K, gl.SliceLayout(1, gLoadLayoutB))
    cols_left = gl.minimum(offs_bn, n_last)
    cols_right = gl.minimum(offs_bn + BLOCK_N // 2, n_last)
    b_cols_left = gl.multiple_of(cols_left * stride_bn, 8)
    b_cols_right = gl.multiple_of(cols_right * stride_bn, 8)
    b_left_offsets = offs_bk[:, None] * stride_bk + b_cols_left[None, :]
    b_right_offsets = offs_bk[:, None] * stride_bk + b_cols_right[None, :]
    return b_left_offsets, b_right_offsets


@gluon.jit
def a_load_addressing(
    m_idx,
    a_offsets,
    a_bot_shift,
    M,
    stride_am,
    stride_ak,
    BLOCK_M: gl.constexpr,
    BLOCK_K: gl.constexpr,
    gLoadLayoutA: gl.constexpr,
    EVEN_M: gl.constexpr,
):
    """Tile m_idx's a_addr = (top_offs, bot_offs, bot_shift, odd_shift): the bottom
    half loads from base + bot_shift + bot_offs and the odd K-step adds odd_shift.
    EVEN_M: both halves share the half-tile offsets a_offsets. Otherwise the tile's
    clamped offsets, which differ per lane, so bot_shift is 0."""
    if EVEN_M:
        top, bot, bot_shift = a_offsets, a_offsets, a_bot_shift
    else:
        top, bot = clamped_a_offsets(
            M - 1 - m_idx * BLOCK_M,
            stride_am,
            stride_ak,
            BLOCK_M,
            BLOCK_K,
            gLoadLayoutA,
        )
        bot_shift = 0
    return top, bot, bot_shift, BLOCK_K * stride_ak


@gluon.jit
def b_load_addressing(
    n_idx,
    b_offsets,
    b_right_shift,
    N,
    stride_bk,
    stride_bn,
    BLOCK_N: gl.constexpr,
    BLOCK_K: gl.constexpr,
    gLoadLayoutB: gl.constexpr,
    EVEN_N: gl.constexpr,
):
    """Tile n_idx's b_addr = (left_offs, right_offs, right_shift, odd_shift)
    (a_load_addressing)."""
    if EVEN_N:
        left, right, right_shift = b_offsets, b_offsets, b_right_shift
    else:
        left, right = clamped_b_offsets(
            N - 1 - n_idx * BLOCK_N,
            stride_bk,
            stride_bn,
            BLOCK_N,
            BLOCK_K,
            gLoadLayoutB,
        )
        right_shift = 0
    return left, right, right_shift, BLOCK_K * stride_bk


@gluon.jit
def k_tail_mask(
    k_hi,
    STEP: gl.constexpr,
    BLOCK_K: gl.constexpr,
    layout: gl.constexpr,
    K_FIRST: gl.constexpr,
):
    """Load mask of K-step STEP (0 or 1) of a half-tile in the last pair, whose first
    k_hi of 2 BLOCK_K K values are inside the matrix. K_FIRST: B's [BLOCK_K, 1], else
    A's [1, BLOCK_K]; the load broadcasts it over the half-tile."""
    if K_FIRST:
        k = gl.arange(0, BLOCK_K, gl.SliceLayout(1, layout))[:, None]
    else:
        k = gl.arange(0, BLOCK_K, gl.SliceLayout(0, layout))[None, :]
    return k + STEP * BLOCK_K < k_hi


@gluon.jit
def tail_pair_masks(
    k_hi,
    K_RES: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_K: gl.constexpr,
    gLoadLayoutA: gl.constexpr,
    gLoadLayoutB: gl.constexpr,
    TAIL: gl.constexpr = False,
):
    """(mask_a0, mask_b0, mask_a1, mask_b1) for load_first_pair of a pair whose first
    k_hi K values are in the matrix. K-step 1 of the last pair always reaches past K,
    K-step 0 only when K_RES > BLOCK_K; a mask that can never drop a lane is None.
    TAIL: k_hi is the last pair's 2 BLOCK_K - K_RES, so with K_RES >= BLOCK_K K-step 1
    drops every lane and its masks are constant."""
    mask_a0, mask_b0 = None, None
    if K_RES > BLOCK_K:
        mask_a0 = k_tail_mask(k_hi, 0, BLOCK_K, gLoadLayoutA, False)
        mask_b0 = k_tail_mask(k_hi, 0, BLOCK_K, gLoadLayoutB, True)
    if TAIL and K_RES >= BLOCK_K:
        mask_a1 = gl.full([BLOCK_M // 2, BLOCK_K], 0, gl.int1, gLoadLayoutA)
        mask_b1 = gl.full([BLOCK_K, BLOCK_N // 2], 0, gl.int1, gLoadLayoutB)
    else:
        mask_a1 = k_tail_mask(k_hi, 1, BLOCK_K, gLoadLayoutA, False)
        mask_b1 = k_tail_mask(k_hi, 1, BLOCK_K, gLoadLayoutB, True)
    return mask_a0, mask_b0, mask_a1, mask_b1


@gluon.jit
def load_first_pair(
    a_base,
    b_base,
    smem,
    a_addr,
    b_addr,
    dotOpLayoutA: gl.constexpr,
    dotOpLayoutB: gl.constexpr,
    mask_a0=None,
    mask_b0=None,
    mask_a1=None,
    mask_b1=None,
):
    """Prologue: the two K-steps at a_base / b_base into buffers 0 and 1,
    and the first one's a_top / b_left. mask_*: load masks of A's and B's K-step 0 and
    1 (ragged K), None for unmasked."""
    smemA_top, smemA_bot, smemB_left, smemB_right = smem
    a_top_offsets, a_bot_offsets, a_bot_shift, a_odd_shift = a_addr
    b_left_offsets, b_right_offsets, b_right_shift, b_odd_shift = b_addr

    g_idx = 0
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        smemB_left.index(g_idx), b_base, b_left_offsets, mask=mask_b0
    )
    gl.amd.cdna4.async_copy.commit_group()
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        smemA_top.index(g_idx), a_base, a_top_offsets, mask=mask_a0
    )
    gl.amd.cdna4.async_copy.commit_group()
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        smemA_bot.index(g_idx), a_base + a_bot_shift, a_bot_offsets, mask=mask_a0
    )
    gl.amd.cdna4.async_copy.commit_group()
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        smemB_right.index(g_idx), b_base + b_right_shift, b_right_offsets, mask=mask_b0
    )
    gl.amd.cdna4.async_copy.commit_group()

    g_idx = 1
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        smemB_left.index(g_idx), b_base + b_odd_shift, b_left_offsets, mask=mask_b1
    )
    gl.amd.cdna4.async_copy.commit_group()
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        smemA_top.index(g_idx), a_base + a_odd_shift, a_top_offsets, mask=mask_a1
    )
    gl.amd.cdna4.async_copy.commit_group()
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        smemA_bot.index(g_idx),
        a_base + a_bot_shift + a_odd_shift,
        a_bot_offsets,
        mask=mask_a1,
    )
    gl.amd.cdna4.async_copy.commit_group()
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        smemB_right.index(g_idx),
        b_base + b_right_shift + b_odd_shift,
        b_right_offsets,
        mask=mask_b1,
    )
    gl.amd.cdna4.async_copy.commit_group()

    gl.amd.cdna4.async_copy.wait_group(6)
    b_left = smemB_left.index(0).load(dotOpLayoutB)
    a_top = smemA_top.index(0).load(dotOpLayoutA)
    return a_top, b_left


@gluon.jit
def k_step_pair(
    accs,
    a_top,
    b_left,
    smem,
    a_base,
    b_base,
    a_addr,
    b_addr,
    dotOpLayoutA: gl.constexpr,
    dotOpLayoutB: gl.constexpr,
    SKIP_ODD: gl.constexpr = False,
):
    """One iteration of the main loop, two K-steps: consumes a_top / b_left (in
    registers) and buffers 0 and 1, prefetches the two K-steps at a_base / b_base into
    the same buffers, and returns the accumulators and the next a_top / b_left. Every
    wait keeps 5 groups of async copies in flight. Constant-zero accumulators make the
    first four MFMAs zero-C MFMAs. SKIP_ODD: buffer 1 holds only K padding, so its
    MFMAs are dropped; its waits and the prefetches stay."""
    acc_tl, acc_bl, acc_tr, acc_br = accs
    smemA_top, smemA_bot, smemB_left, smemB_right = smem
    a_top_offsets, a_bot_offsets, a_bot_shift, a_odd_shift = a_addr
    b_left_offsets, b_right_offsets, b_right_shift, b_odd_shift = b_addr

    ## =============================================================
    ## Sub-iteration 0: consume buffer 0, prefetch into buffer 0
    ## =============================================================

    ########################################
    ## Region 0: C_tl = DOT(a_top, b_left)
    ########################################
    acc_tl = gl.amd.cdna3.mfma(a_top, b_left, acc_tl, cd_regclass="a")

    gl.amd.cdna4.async_copy.wait_group(5)
    a_bot = smemA_bot.index(0).load(dotOpLayoutA)

    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        smemB_left.index(0), b_base, b_left_offsets
    )
    gl.amd.cdna4.async_copy.commit_group()

    ########################################
    ## Region 1: C_bl = DOT(a_bot, b_left)
    ########################################
    acc_bl = gl.amd.cdna3.mfma(a_bot, b_left, acc_bl, cd_regclass="a")

    gl.amd.cdna4.async_copy.wait_group(5)
    b_right = smemB_right.index(0).load(dotOpLayoutB)

    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        smemA_top.index(0), a_base, a_top_offsets
    )
    gl.amd.cdna4.async_copy.commit_group()

    ########################################
    ## Region 2: C_tr = DOT(a_top, b_right)
    ########################################
    acc_tr = gl.amd.cdna3.mfma(a_top, b_right, acc_tr, cd_regclass="a")

    gl.amd.cdna4.async_copy.wait_group(5)
    b_left = smemB_left.index(1).load(dotOpLayoutB)

    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        smemA_bot.index(0), a_base + a_bot_shift, a_bot_offsets
    )
    gl.amd.cdna4.async_copy.commit_group()

    ########################################
    ## Region 3: C_br = DOT(a_bot, b_right)
    ########################################
    acc_br = gl.amd.cdna3.mfma(a_bot, b_right, acc_br, cd_regclass="a")

    gl.amd.cdna4.async_copy.wait_group(5)
    a_top = smemA_top.index(1).load(dotOpLayoutA)

    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        smemB_right.index(0), b_base + b_right_shift, b_right_offsets
    )
    gl.amd.cdna4.async_copy.commit_group()

    ## =============================================================
    ## Loop unroll: Sub-iteration 1: consume buffer 1, prefetch
    ## into buffer 1. The odd K-step's loads add a_odd_shift / b_odd_shift to the base.
    ## =============================================================

    ########################################
    ## Region 0: C_tl = DOT(a_top, b_left)
    ########################################
    if not SKIP_ODD:
        acc_tl = gl.amd.cdna3.mfma(a_top, b_left, acc_tl, cd_regclass="a")

    gl.amd.cdna4.async_copy.wait_group(5)
    a_bot = smemA_bot.index(1).load(dotOpLayoutA)

    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        smemB_left.index(1), b_base + b_odd_shift, b_left_offsets
    )
    gl.amd.cdna4.async_copy.commit_group()

    ########################################
    ## Region 1: C_bl = DOT(a_bot, b_left)
    ########################################
    if not SKIP_ODD:
        acc_bl = gl.amd.cdna3.mfma(a_bot, b_left, acc_bl, cd_regclass="a")

    gl.amd.cdna4.async_copy.wait_group(5)
    b_right = smemB_right.index(1).load(dotOpLayoutB)

    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        smemA_top.index(1), a_base + a_odd_shift, a_top_offsets
    )
    gl.amd.cdna4.async_copy.commit_group()

    ########################################
    ## Region 2: C_tr = DOT(a_top, b_right)
    ########################################
    if not SKIP_ODD:
        acc_tr = gl.amd.cdna3.mfma(a_top, b_right, acc_tr, cd_regclass="a")

    gl.amd.cdna4.async_copy.wait_group(5)
    b_left = smemB_left.index(0).load(dotOpLayoutB)

    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        smemA_bot.index(1), a_base + a_bot_shift + a_odd_shift, a_bot_offsets
    )
    gl.amd.cdna4.async_copy.commit_group()

    ########################################
    ## Region 3: C_br = DOT(a_bot, b_right)
    ########################################
    if not SKIP_ODD:
        acc_br = gl.amd.cdna3.mfma(a_bot, b_right, acc_br, cd_regclass="a")

    gl.amd.cdna4.async_copy.wait_group(5)
    a_top = smemA_top.index(0).load(dotOpLayoutA)

    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        smemB_right.index(1), b_base + b_right_shift + b_odd_shift, b_right_offsets
    )
    gl.amd.cdna4.async_copy.commit_group()

    return (acc_tl, acc_bl, acc_tr, acc_br), a_top, b_left


@gluon.jit
def init_acc(
    bias_ptr,
    pid_n,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    mfmaLayout: gl.constexpr,
    ADD_BIAS: gl.constexpr,
):
    """The four fp32 accumulators (tl, bl, tr, br) of a tile, starting from the tile's
    bias[N] (or zero).

    Adding the bias here rather than before the C stores keeps it out of the epilogue,
    where the stores already hold the register peak; the epilogue add spills. Without
    bias the accumulators are a constant zero, which a peeled first MFMA folds into an
    inline-0 C operand.
    """
    acc_l = gl.zeros((BLOCK_M // 2, BLOCK_N // 2), gl.float32, mfmaLayout)
    acc_r = acc_l
    if ADD_BIAS:
        # Scalar base + 32-bit lane offsets: no per-lane 64-bit addresses.
        offs_bias = gl.arange(0, BLOCK_N // 2, gl.SliceLayout(0, mfmaLayout))
        bias_base = bias_ptr + pid_n * BLOCK_N
        bias_l = gl.amd.cdna3.buffer_load(ptr=bias_base, offsets=offs_bias)
        bias_r = gl.amd.cdna3.buffer_load(
            ptr=bias_base + BLOCK_N // 2, offsets=offs_bias
        )
        acc_l = acc_l + bias_l.to(gl.float32)[None, :]
        acc_r = acc_r + bias_r.to(gl.float32)[None, :]
    return acc_l, acc_l, acc_r, acc_r


@gluon.jit
def init_acc_edge(
    bias_ptr,
    pid_n,
    n_left,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    mfmaLayout: gl.constexpr,
    ADD_BIAS: gl.constexpr,
    EVEN_N: gl.constexpr,
):
    """init_acc, with the bias loaded only for the tile's first n_left columns on a
    ragged N."""
    if ADD_BIAS and not EVEN_N:
        acc_l = gl.zeros((BLOCK_M // 2, BLOCK_N // 2), gl.float32, mfmaLayout)
        offs_bias = gl.arange(0, BLOCK_N // 2, gl.SliceLayout(0, mfmaLayout))
        bias_base = bias_ptr + pid_n * BLOCK_N
        bias_l = gl.amd.cdna3.buffer_load(
            ptr=bias_base, offsets=offs_bias, mask=offs_bias < n_left, other=0.0
        )
        bias_r = gl.amd.cdna3.buffer_load(
            ptr=bias_base + BLOCK_N // 2,
            offsets=offs_bias,
            mask=offs_bias < n_left - BLOCK_N // 2,
            other=0.0,
        )
        acc_r = acc_l + bias_r.to(gl.float32)[None, :]
        acc_l = acc_l + bias_l.to(gl.float32)[None, :]
        tl, bl, tr, br = acc_l, acc_l, acc_r, acc_r
    else:
        tl, bl, tr, br = init_acc(
            bias_ptr, pid_n, BLOCK_M, BLOCK_N, mfmaLayout, ADD_BIAS
        )
    return tl, bl, tr, br


@gluon.jit
def persistent_tile(
    vpid,
    a_top,
    b_left,
    tile_addr,
    a_ptr,
    b_ptr,
    c_ptr,
    bias_ptr,
    smem,
    kofs,
    strides,
    M,
    N,
    K: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_K: gl.constexpr,
    NUM_PROGRAMS: gl.constexpr,
    NUM_XCDS: gl.constexpr,
    GROUP_SIZE_M: gl.constexpr,
    ADD_BIAS: gl.constexpr,
    mfmaLayout: gl.constexpr,
    gLoadLayoutA: gl.constexpr,
    gLoadLayoutB: gl.constexpr,
    EVEN_M: gl.constexpr,
    EVEN_N: gl.constexpr,
    EVEN_K: gl.constexpr,
    K_RES: gl.constexpr,
    total_full_tiles,
    sk_target,
    SK_PREFETCH: gl.constexpr,
):
    """One full-K tile of the persistent loop, for virtual pid vpid: the main loop and
    epilogue, the epilogue prefetching the next tile.

    SK_PREFETCH: on the program's last trip the prefetch loads sk_target = (m, n,
    column, k_hi), the first pair of the program's first Stream-K segment. On a ragged
    K it keeps the first k_hi K values (0: masked off). On an even K it is unmasked:
    k_hi is 2 BLOCK_K, or 0 on an idle program, whose target (Stream-K tile 0's last
    pair) is in bounds and never read. Without SK_PREFETCH the last trip's prefetch
    stays unmasked and is discarded. Either way masks cost more than the loads.

    Expects the tile's first pair in flight in buffers 0 and 1 and its first operands in
    a_top / b_left; prefetches the same for vpid + NUM_PROGRAMS (clamped to the last
    persistent tile, so every trip issues the same instructions and wait counts),
    or on the last trip the Stream-K target (SK_PREFETCH), and returns that
    tile's a_top / b_left and load addressing. tile_addr is this tile's (a_addr,
    b_addr) (a_load_addressing / b_load_addressing).

    The pair in flight is the tile's last pair (K is K_PAD here; on a ragged K the
    pair is partial), so the main loop starts at column 0, and the next tile's
    prefetch loads that tile's last pair, masked on a ragged K."""
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn = strides
    a_offsets, a_bot_shift, b_offsets, b_right_shift = kofs
    a_addr, b_addr = tile_addr
    smemA_top, smemA_bot, smemB_left, smemB_right = smem
    dotOpLayoutA: gl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=mfmaLayout, k_width=8
    )
    dotOpLayoutB: gl.constexpr = gl.DotOperandLayout(
        operand_index=1, parent=mfmaLayout, k_width=8
    )
    iterMax = gl.cdiv(K, BLOCK_K)

    pid_m, pid_n = vpid_tile_coords(
        vpid, M, N, BLOCK_M, BLOCK_N, total_full_tiles, NUM_XCDS, GROUP_SIZE_M
    )
    next_m, next_n = vpid_tile_coords(
        min(vpid + NUM_PROGRAMS, total_full_tiles - 1),
        M,
        N,
        BLOCK_M,
        BLOCK_N,
        total_full_tiles,
        NUM_XCDS,
        GROUP_SIZE_M,
    )
    has_next = vpid + NUM_PROGRAMS < total_full_tiles
    if SK_PREFETCH:
        sk_m, sk_n, sk_col, sk_khi = sk_target
        next_m = gl.where(has_next, next_m, sk_m)
        next_n = gl.where(has_next, next_n, sk_n)
        next_col = gl.where(has_next, K - 2 * BLOCK_K, sk_col)
    a_next = a_ptr + next_m.to(gl.int64) * BLOCK_M * stride_am
    b_next = b_ptr + next_n.to(gl.int64) * BLOCK_N * stride_bn

    # The last pair is in flight, so the first prefetch is column 0.
    a_base = a_ptr + pid_m.to(gl.int64) * BLOCK_M * stride_am
    b_base = b_ptr + pid_n.to(gl.int64) * BLOCK_N * stride_bn
    if SK_PREFETCH:
        a_next += next_col * stride_ak
        b_next += next_col * stride_bk
    else:
        a_next += (K - 2 * BLOCK_K) * stride_ak
        b_next += (K - 2 * BLOCK_K) * stride_bk
    m_left = M - pid_m * BLOCK_M
    n_left = gl.multiple_of(N - pid_n * BLOCK_N, 8)

    ## Main loop, its first iteration peeled: without bias, the
    ## constant-zero accumulator becomes the MFMA's inline-0 C operand, so no AGPRs
    ## are zeroed per tile (a loop-carried zero would be materialised).
    accs = init_acc_edge(
        bias_ptr, pid_n, n_left, BLOCK_M, BLOCK_N, mfmaLayout, ADD_BIAS, EVEN_N
    )
    accs, a_top, b_left = k_step_pair(
        accs,
        a_top,
        b_left,
        smem,
        a_base,
        b_base,
        a_addr,
        b_addr,
        dotOpLayoutA,
        dotOpLayoutB,
        (not EVEN_K) and K_RES >= BLOCK_K,
    )
    a_base += BLOCK_K * stride_ak * 2
    b_base += BLOCK_K * stride_bk * 2
    for k in range(2, iterMax - 2, 2):
        accs, a_top, b_left = k_step_pair(
            accs,
            a_top,
            b_left,
            smem,
            a_base,
            b_base,
            a_addr,
            b_addr,
            dotOpLayoutA,
            dotOpLayoutB,
        )
        a_base += BLOCK_K * stride_ak * 2
        b_base += BLOCK_K * stride_bk * 2
    acc_tl, acc_bl, acc_tr, acc_br = accs

    # The next tile's load offsets, for its prefetch below and for its own trip.
    na_addr = a_load_addressing(
        next_m,
        a_offsets,
        a_bot_shift,
        M,
        stride_am,
        stride_ak,
        BLOCK_M,
        BLOCK_K,
        gLoadLayoutA,
        EVEN_M,
    )
    nb_addr = b_load_addressing(
        next_n,
        b_offsets,
        b_right_shift,
        N,
        stride_bk,
        stride_bn,
        BLOCK_N,
        BLOCK_K,
        gLoadLayoutB,
        EVEN_N,
    )
    na_top, na_bot, na_bot_shift, na_odd_shift = na_addr
    nb_left, nb_right, nb_right_shift, nb_odd_shift = nb_addr
    # Computed here rather than with a_next, so they are not live across the K loop.
    if SK_PREFETCH and not EVEN_K:
        k_hi = gl.where(has_next, 2 * BLOCK_K - K_RES, sk_khi)
        mask_a0 = k_tail_mask(k_hi, 0, BLOCK_K, gLoadLayoutA, False)
        mask_b0 = k_tail_mask(k_hi, 0, BLOCK_K, gLoadLayoutB, True)
        mask_a1 = k_tail_mask(k_hi, 1, BLOCK_K, gLoadLayoutA, False)
        mask_b1 = k_tail_mask(k_hi, 1, BLOCK_K, gLoadLayoutB, True)
    elif EVEN_K:
        # Also with SK_PREFETCH: an idle program's target is in bounds and never
        # read. Dropping the masks frees registers, but at the 512-VGPR limit
        # (256x256x64, ragged M with bias) the earlier loads add spills; accepted.
        mask_a0, mask_b0, mask_a1, mask_b1 = None, None, None, None
    else:
        mask_a0, mask_b0, mask_a1, mask_b1 = tail_pair_masks(
            2 * BLOCK_K - K_RES,
            K_RES,
            BLOCK_M,
            BLOCK_N,
            BLOCK_K,
            gLoadLayoutA,
            gLoadLayoutB,
            True,
        )

    ## Epilogue: 4-quadrant stores with natural-pipeline ordering, each async copy
    ## prefetching the next tile's first pair.

    # Store layout sized to the half tile: 16 contiguous bytes per thread where the
    # quadrant has enough elements, and one warp spans BLOCK_N // 2 columns ([4, 16]
    # lanes at BLOCK_N = 256).
    gStoreLayoutC: gl.constexpr = coalesced_store_layout(
        BLOCK_M // 2,
        BLOCK_N // 2,
        c_ptr.dtype.element_ty.primitive_bitwidth // 8,
        gl.num_warps(),
    )

    offs_cm = gl.arange(0, BLOCK_M // 2, gl.SliceLayout(1, gStoreLayoutC))
    offs_cn = gl.arange(0, BLOCK_N // 2, gl.SliceLayout(0, gStoreLayoutC))
    # The tile's row goes in the 64-bit scalar base, so the 32-bit offsets only span
    # BLOCK_M rows. Its column stays in the vector offsets, which keeps them
    # tile-dependent: the compiler can't hoist them above the tile loop and keep them
    # live (and spilled) through the K loop. The quadrant shifts go in the scalar base
    # too, so all four stores share one offset tensor. On a ragged M the rows are
    # clamped to the tile's last one for the same reason (only rows the store masks
    # drop change). The row term carries the loads' multiple_of hint and n_left is
    # hinted a multiple of 8 (the wrapper checks N), so each thread's 8-column group
    # keeps one predicate and one 16-byte store when stride_cm is not a multiple of 16.
    c_tl_base = c_ptr + pid_m.to(gl.int64) * BLOCK_M * stride_cm
    col = pid_n * BLOCK_N + offs_cn
    if EVEN_M:
        c_rows = gl.multiple_of(stride_cm * offs_cm, 8)
    else:
        c_rows = gl.multiple_of(stride_cm * gl.minimum(offs_cm, m_left - 1), 8)
    c_offsets = c_rows[:, None] + stride_cn * col[None, :]
    c_tr_base = c_tl_base + BLOCK_N * stride_cn // 2
    c_bl_base = c_tl_base + BLOCK_M * stride_cm // 2
    c_br_base = c_bl_base + BLOCK_N * stride_cn // 2
    if EVEN_M and EVEN_N:
        mask_tl, mask_tr, mask_bl, mask_br = None, None, None, None
    else:
        # The ragged last tile stores only the rows and columns inside C.
        in_top = offs_cm[:, None] < m_left
        in_bot = offs_cm[:, None] + BLOCK_M // 2 < m_left
        if EVEN_N:
            mask_tl, mask_tr = in_top, in_top
            mask_bl, mask_br = in_bot, in_bot
        else:
            in_left = offs_cn[None, :] < n_left
            in_right = offs_cn[None, :] + BLOCK_N // 2 < n_left
            mask_tl = in_top & in_left
            mask_tr = in_top & in_right
            mask_bl = in_bot & in_left
            mask_br = in_bot & in_right

    ## Iter iterMax - 2: same 4-region pattern as main loop, each AC prefetching the
    ## next tile's K-step 0 into the buffer the main loop would refill here, so every
    ## wait keeps 5 groups in flight.
    acc_tl = gl.amd.cdna3.mfma(a_top, b_left, acc_tl, cd_regclass="a")
    gl.amd.cdna4.async_copy.wait_group(5)
    l_idx = (iterMax - 2) % 2
    a_bot = smemA_bot.index(l_idx).load(dotOpLayoutA)
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        smemB_left.index(0), b_next, nb_left, mask=mask_b0
    )
    gl.amd.cdna4.async_copy.commit_group()

    acc_bl = gl.amd.cdna3.mfma(a_bot, b_left, acc_bl, cd_regclass="a")
    gl.amd.cdna4.async_copy.wait_group(5)
    b_right = smemB_right.index(l_idx).load(dotOpLayoutB)
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        smemA_top.index(0), a_next, na_top, mask=mask_a0
    )
    gl.amd.cdna4.async_copy.commit_group()

    acc_tr = gl.amd.cdna3.mfma(a_top, b_right, acc_tr, cd_regclass="a")
    gl.amd.cdna4.async_copy.wait_group(5)
    g_idx = 1 - l_idx
    b_left = smemB_left.index(g_idx).load(dotOpLayoutB)
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        smemA_bot.index(0), a_next + na_bot_shift, na_bot, mask=mask_a0
    )
    gl.amd.cdna4.async_copy.commit_group()

    acc_br = gl.amd.cdna3.mfma(a_bot, b_right, acc_br, cd_regclass="a")
    gl.amd.cdna4.async_copy.wait_group(5)
    a_top = smemA_top.index(g_idx).load(dotOpLayoutA)
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        smemB_right.index(0), b_next + nb_right_shift, nb_right, mask=mask_b0
    )
    gl.amd.cdna4.async_copy.commit_group()

    ## Iter iterMax - 1
    ## Natural-pipeline epilogue: each store follows its MFMA with one
    ## MFMA cycle of gap, yielding uniform MFMA-store interleaving. The ACs
    ## prefetch the next tile's K-step 1.
    acc_tl = gl.amd.cdna3.mfma(a_top, b_left, acc_tl, cd_regclass="a")
    gl.amd.cdna4.async_copy.wait_group(5)
    a_bot = smemA_bot.index(g_idx).load(dotOpLayoutA)
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        smemB_left.index(1), b_next + nb_odd_shift, nb_left, mask=mask_b1
    )
    gl.amd.cdna4.async_copy.commit_group()

    acc_bl = gl.amd.cdna3.mfma(a_bot, b_left, acc_bl, cd_regclass="a")
    gl.amd.cdna4.async_copy.wait_group(5)
    b_right = smemB_right.index(g_idx).load(dotOpLayoutB)
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        smemA_top.index(1), a_next + na_odd_shift, na_top, mask=mask_a1
    )
    gl.amd.cdna4.async_copy.commit_group()

    c_tl = acc_tl.to(c_ptr.dtype.element_ty)
    c_tl = gl.convert_layout(c_tl, layout=gStoreLayoutC)
    gl.amd.cdna3.buffer_store(
        ptr=c_tl_base, offsets=c_offsets, stored_value=c_tl, mask=mask_tl
    )

    acc_tr = gl.amd.cdna3.mfma(a_top, b_right, acc_tr, cd_regclass="a")
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        smemA_bot.index(1), a_next + na_bot_shift + na_odd_shift, na_bot, mask=mask_a1
    )
    gl.amd.cdna4.async_copy.commit_group()

    c_bl = acc_bl.to(c_ptr.dtype.element_ty)
    c_bl = gl.convert_layout(c_bl, layout=gStoreLayoutC)
    gl.amd.cdna3.buffer_store(
        ptr=c_bl_base, offsets=c_offsets, stored_value=c_bl, mask=mask_bl
    )

    acc_br = gl.amd.cdna3.mfma(a_bot, b_right, acc_br, cd_regclass="a")

    # The next tile's K-step 0 operands (prefetched at iterMax - 2), read while the
    # last MFMAs and stores drain; this tile's operand registers are all dead after
    # acc_br. Both reads are deferred to here rather than interleaved as in the main
    # loop: reading b_left after acc_tr keeps it live across the c_bl store, which
    # costs 24-36 AGPR moves per tile and, on a ragged M and N, an extra spill.
    gl.amd.cdna4.async_copy.wait_group(5)
    b_left = smemB_left.index(0).load(dotOpLayoutB)
    a_top = smemA_top.index(0).load(dotOpLayoutA)
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        smemB_right.index(1),
        b_next + nb_right_shift + nb_odd_shift,
        nb_right,
        mask=mask_b1,
    )
    gl.amd.cdna4.async_copy.commit_group()

    c_tr = acc_tr.to(c_ptr.dtype.element_ty)
    c_tr = gl.convert_layout(c_tr, layout=gStoreLayoutC)
    gl.amd.cdna3.buffer_store(
        ptr=c_tr_base, offsets=c_offsets, stored_value=c_tr, mask=mask_tr
    )

    c_br = acc_br.to(c_ptr.dtype.element_ty)
    c_br = gl.convert_layout(c_br, layout=gStoreLayoutC)
    gl.amd.cdna3.buffer_store(
        ptr=c_br_base, offsets=c_offsets, stored_value=c_br, mask=mask_br
    )

    return a_top, b_left, (na_addr, nb_addr)


# ---------------------------------------------------------------------------------
# Stream-K: programs and tiles
# ---------------------------------------------------------------------------------
# The persistent loop covers the XCD-remapped tile ids [0, total_full_tiles)
# (vpid_tile_coords with GRID_MN = total_full_tiles), so Stream-K tile t is tile
# total_full_tiles + t.


@gluon.jit
def xcd_blocked_rank(
    N: gl.constexpr, NUM_XCDS: gl.constexpr, IDLE: gl.constexpr = False
):
    """Rank of this program among the N participating programs: an XCD-blocked id,
    so each XCD's ranks are contiguous (hardware pid p runs on XCD p % NUM_XCDS; the
    first N % NUM_XCDS XCDs hold one rank more). The participants are hardware pids
    0 .. N-1, so when fewer than the launched programs take part the idle ones are
    spread over the XCDs instead of packed onto the last XCDs. IDLE says the launch
    has programs beyond N; they get rank N. With N <= NUM_XCDS the transpose is the
    identity on pid < N, so it is skipped."""
    pid = gl.program_id(axis=0)
    rank = pid
    if NUM_XCDS != 1 and N > NUM_XCDS:
        rank = ragged_transpose(pid, N, NUM_XCDS)
    if IDLE:
        rank = gl.where(pid < N, rank, N)
    return rank


@gluon.jit
def streamk_first_segment(
    total_full_tiles,
    STREAMK_TILES: gl.constexpr,
    PAIRS_PER_TILE: gl.constexpr,
    SPLIT_N: gl.constexpr,
    END_TO_END: gl.constexpr,
    L2: gl.constexpr,
    NUM_PROGRAMS: gl.constexpr,
    NUM_XCDS: gl.constexpr,
):
    """(has, tile id, k0) of the first segment the Stream-K phase runs on this program
    (the kernel's segment assignment, full-grid launches only); has is 0 for an idle
    program, whose tile and k0 are then tile 0's last pair. k0 is a logical K-step:
    pair 0 is loaded from the physical last pair (see quadrant_segment)."""
    if END_TO_END:
        rank, sk_pair_start, sk_pair_end = e2e_pair_range(
            STREAMK_TILES, PAIRS_PER_TILE, NUM_PROGRAMS, NUM_XCDS
        )
        sk_t, _, k0, _, _ = e2e_segment_ending_at(
            sk_pair_end,
            sk_pair_start,
            rank,
            STREAMK_TILES,
            PAIRS_PER_TILE,
            NUM_PROGRAMS,
        )
        has = 1
    else:
        if L2:
            rank = xcd_blocked_rank(NUM_PROGRAMS, NUM_XCDS)
            sk_t, j, n_tile = xcd_local_chunk_of(
                rank, STREAMK_TILES, NUM_PROGRAMS // NUM_XCDS, NUM_XCDS, SPLIT_N
            )
            has = (n_tile > 0).to(gl.int32)
        else:
            rank = xcd_blocked_rank(STREAMK_TILES * SPLIT_N, NUM_XCDS, IDLE=True)
            has = (rank < STREAMK_TILES * SPLIT_N).to(gl.int32)
            sk_t = rank // SPLIT_N
            j = rank % SPLIT_N
        k0 = even_split_start(j, PAIRS_PER_TILE, SPLIT_N) * 2 * has
        sk_t = sk_t * has
    return has, total_full_tiles + sk_t, k0


@gluon.jit
def xcd_local_chunk_of(
    rank,
    S: gl.constexpr,
    PROGS_PER_XCD: gl.constexpr,
    NUM_XCDS: gl.constexpr,
    SPLIT_N: gl.constexpr,
):
    """(tile, chunk, programs on the tile) of the program with this rank in the
    XCD-local split (l2), with 0 programs if it idles. XCD x = rank // PROGS_PER_XCD
    owns part x of an even split of the S Stream-K tiles over the XCDs; its local
    program l works on its tile l // SPLIT_N as chunk l % SPLIT_N, so a tile's SPLIT_N
    programs are on one XCD. The host checks
    ceil(S / NUM_XCDS) * SPLIT_N <= PROGS_PER_XCD."""
    x = rank // PROGS_PER_XCD
    local = rank % PROGS_PER_XCD
    tl = local // SPLIT_N
    t = even_split_start(x, S, NUM_XCDS) + tl
    n_tile = SPLIT_N * (tl < even_split_size(x, S, NUM_XCDS)).to(gl.int32)
    return t, local % SPLIT_N, n_tile


@gluon.jit
def e2e_rank_of_pair(i, L: gl.constexpr, R: gl.constexpr):
    """End-to-end split: rank of the program whose range holds pair i, when the first
    R programs hold L + 1 pairs and the rest L."""
    head: gl.constexpr = R * (L + 1)
    if L == 0:
        rank = i // (L + 1)
    else:
        rank = i // (L + 1)
        if i >= head:
            rank = R + (i - head) // L
    return rank


@gluon.jit
def e2e_pair_range(
    STREAMK_TILES: gl.constexpr,
    PAIRS_PER_TILE: gl.constexpr,
    NUM_PROGRAMS: gl.constexpr,
    NUM_XCDS: gl.constexpr,
):
    """End-to-end split: (rank, sk_pair_start, sk_pair_end) of this program, whose range
    [sk_pair_start, sk_pair_end) of the Stream-K tiles' pairs is part rank of an even
    split over the full grid (XCD-blocked ranks)."""
    SK_PAIRS: gl.constexpr = STREAMK_TILES * PAIRS_PER_TILE
    gl.static_assert(NUM_PROGRAMS <= SK_PAIRS)
    rank = xcd_blocked_rank(NUM_PROGRAMS, NUM_XCDS)
    sk_pair_start = even_split_start(rank, SK_PAIRS, NUM_PROGRAMS)
    sk_pair_end = even_split_start(rank + 1, SK_PAIRS, NUM_PROGRAMS)
    return rank, sk_pair_start, sk_pair_end


@gluon.jit
def e2e_segment_ending_at(
    seg_end,
    sk_pair_start,
    rank,
    STREAMK_TILES: gl.constexpr,
    PAIRS_PER_TILE: gl.constexpr,
    NUM_PROGRAMS: gl.constexpr,
):
    """End-to-end split, segments run last to first (see the module docstring):
    (tile t, seg_start, k0, is_owner, n_peers) of this program's segment that ends at
    pair seg_end. It covers pairs [seg_start, seg_end) of Stream-K tile t, from logical
    K-step k0. is_owner: it holds the tile's last pair and collects the tile. n_peers:
    the programs holding the tile's earlier pairs (0 unless is_owner)."""
    SK_PAIRS: gl.constexpr = STREAMK_TILES * PAIRS_PER_TILE
    L: gl.constexpr = SK_PAIRS // NUM_PROGRAMS
    R: gl.constexpr = SK_PAIRS % NUM_PROGRAMS
    t = (seg_end - 1) // PAIRS_PER_TILE
    tile_pair_start = t * PAIRS_PER_TILE
    seg_start = gl.maximum(tile_pair_start, sk_pair_start)
    is_owner = (seg_end == tile_pair_start + PAIRS_PER_TILE).to(gl.int32)
    n_peers = (rank - e2e_rank_of_pair(tile_pair_start, L, R)) * is_owner
    return t, seg_start, (seg_start - tile_pair_start) * 2, is_owner, n_peers


# ---------------------------------------------------------------------------------
# Stream-K: synchronisation
#
# Partials are stored write-through (.wt) and drained in every wave before the flag
# or barrier arrival that publishes them, and read with .cv loads, which miss L1; so
# every atomic is relaxed, with compiler fences where an acquire would be. Flags and
# counts are re-armed inside the launch (zero between launches).
# ---------------------------------------------------------------------------------


@gluon.jit
def _one():
    return gl.zeros([1], gl.int32, gl.BlockedLayout([1], [64], [gl.num_warps()], [0]))


@gluon.jit
def drain_and_barrier():
    """Every wave waits for all its vector memory ops (stores and async copies: vmcnt
    is one counter), then a workgroup barrier. A flag set after this is ordered after
    all the waves' partial stores."""
    gl.inline_asm_elementwise(
        "s_waitcnt vmcnt(0)\ns_barrier\nv_mov_b32 $0, $1",
        "=v,v,~{memory}",
        [_one()],
        dtype=gl.int32,
        is_pure=False,
        pack=1,
    )


@gluon.jit
def asm_fence(code: gl.constexpr):
    """Inline asm with a memory clobber: the compiler moves no memory access across
    it."""
    gl.inline_asm_elementwise(
        code + "\nv_mov_b32 $0, $1",
        "=v,v,~{memory}",
        [_one()],
        dtype=gl.int32,
        is_pure=False,
        pack=1,
    )


@gluon.jit
def tile_barrier(sync_ptr, b, n, wait):
    """Arrive on the reduce-scatter barrier of the tile whose first slot is b and, if
    wait, wait until all n of its programs have published their partials.

    sync[2b] counts arrivals and sync[2b + 1] is a generation. The last arriver resets
    the count and bumps the generation, which re-arms the barrier for the next launch.
    Scalar atomics run once per workgroup and broadcast their result."""
    cnt_ptr = sync_ptr + 2 * b
    gen_ptr = cnt_ptr + 1
    g0 = gl.atomic_add(gen_ptr, 0, sem="relaxed", scope="gpu")
    # If the last arrival and the bump overtook this read, the program would wait for
    # a generation that never comes.
    asm_fence("s_waitcnt vmcnt(0)")
    arrived = gl.atomic_add(cnt_ptr, 1, sem="relaxed", scope="gpu")
    if arrived == n - 1:
        gl.atomic_xchg(cnt_ptr, 0, sem="relaxed", scope="gpu")
        gl.atomic_xchg(gen_ptr, g0 + 1, sem="relaxed", scope="gpu")
    elif wait:
        while gl.atomic_add(gen_ptr, 0, sem="relaxed", scope="gpu") == g0:
            pass
    asm_fence("s_nop 0")  # no partial read moves above the wait


@gluon.jit
def l2_tile_barrier(
    sync_ptr,
    b,
    n,
    readers,
    j,
    is_reader,
    NUM_PROGRAMS: gl.constexpr,
    NUM_XCDS: gl.constexpr,
):
    """Publish to each reader's private row, then let readers reset their row.

    Layout. P = NUM_PROGRAMS, PER = P / NUM_XCDS (32 words, one 128-byte
    line). A split tile's programs are slots b .. b+n-1 (chunk j = slot - b);
    chunks j < r (readers) also reduce. Each slot reserves PER flag words,
    so the tile owns the n*PER words of its own slots:

      tile_sync (P = 256, PER = 32), word offsets:

      0           512         544               704            896
      | wt count, | slot 0    | slot 1 ...      | slots 6..11  | slot 12 ...
      | gen (2P)  | 32 words  |                 | tile b = 6   |
                  ^ flag region at 2P           ^ base = 2P + b*PER

    Within the tile, word i*n + j is writer j's flag for reader i. Example,
    n = 6, r = 4, b = 6, P = 256 (base = 512 + 6*32 = 704):

      writer j:             0   1   2   3   4   5
      reader 0  row 0*n:  [ 0][ 1][ 2][ 3][ 4][ 5]
      reader 1  row 1*n:  [ 6][ 7][ 8][ 9][10][11]
      reader 2  row 2*n:  [12][13][14][15][16][17]
      reader 3  row 3*n:  [18][19][20][21][22][23]

      writer 4 stores 1 down column 4: offsets 4, 10, 16, 22 (words 708 ..)
      reader 2 polls, then zeroes, row 2: offsets 12 .. 17 (words 716 .. 721)

    In the code j is always this program's chunk, and a lane index covers
    the other side in one instruction:
      store:  flags + m*n + j     lane m = reader i, j = this writer
      poll:   row + lane          row = flags + j*n (this reader), lane = writer
      reset:  row + m             lane m = writer

    Each word has one writer and one reader, so plain stores set the flags
    and the reader alone resets its row: the region is zero between launches.
    r*n <= n*PER keeps a tile inside its own region (the next tile starts at
    (b+n)*PER); base is line-aligned, so the grid fits one line while
    r*n <= 32 (n <= 7).

    Every wave polls the row on its own, but only wave 0 stores and resets:
    a wave that reset the row while another wave of the same program still
    polled it would leave that wave spinning, and a lagging wave's duplicate
    1 could land after the reset. The workgroup barrier after the poll keeps
    the reset behind every wave's last poll."""
    PER: gl.constexpr = NUM_PROGRAMS // NUM_XCDS
    flags = sync_ptr + 2 * NUM_PROGRAMS + b * PER
    FL: gl.constexpr = gl.BlockedLayout([1], [64], [gl.num_warps()], [0])
    # Wave w holds elements 64 w .. 64 w + 63, so only wave 0 has m < 64.
    m = gl.arange(0, 64 * gl.num_warps(), FL)
    # Every wave holds a copy of elements 0 .. 63.
    lane = gl.arange(0, 64, FL)

    gl.store(
        flags + m * n + j,
        m * 0 + 1,
        mask=m < readers,
    )
    if is_reader:
        row = flags + j * n
        done = 0
        while done == 0:
            asm_fence("s_nop 0")  # keeps the poll inside the loop
            f = gl.amd.cdna3.buffer_load(
                ptr=row,
                offsets=lane,
                mask=lane < n,
                other=1,
                cache=".cg",
            )
            done = (gl.min(f, axis=0) == 1).to(gl.int32)
        gl.barrier()  # every wave has seen the row before the reset
        gl.store(row + m, m * 0, mask=m < n)
    asm_fence("s_nop 0")  # no partial read moves above the wait


# ---------------------------------------------------------------------------------
# Stream-K: partials and endings
#
# A program's slot in P is one fp32 tile, four quadrants of QUAD = HALF_M HALF_N
# elements, each stored straight from the MFMA layout (lane_contiguous_offsets) and
# read back by the owner in the same layout. Reduce-scatter reads it as NBLK =
# BLOCK_M BLOCK_N / 1024 blocks of 1024 elements, one per (quadrant, register
# vector), each a 32x32 square of the tile.
# ---------------------------------------------------------------------------------

# One 32x32 MFMA accumulator block (a register vector over 4 warps): each warp holds
# one 16x16 MFMA output, in which lane l holds row l % 16, columns 4 (l // 16) .. + 3.
# Its elements are stored [warp][lane][4 registers].
MFMA_DIM = gl.constexpr(16)  # rows and columns of one warp's MFMA output
MFMA_BLOCK_DIM = gl.constexpr(2 * MFMA_DIM.value)  # 32, the 2 x 2 warps' outputs
MFMA_LANE_ELEMS = gl.constexpr(4)  # registers 0-3 of a lane, 4 adjacent columns
# 64, the 4 columns of 16 lanes
MFMA_GROUP_ELEMS = gl.constexpr(MFMA_DIM.value * MFMA_LANE_ELEMS.value)
MFMA_WARP_ELEMS = gl.constexpr(64 * MFMA_LANE_ELEMS.value)  # 256, a 16x16 output
MFMA_BLOCK_ELEMS = gl.constexpr(4 * MFMA_WARP_ELEMS.value)  # 1024, the 32x32 block


@gluon.jit
def edge_mask(
    m_left,
    n_left,
    ROWS: gl.constexpr,
    COLS: gl.constexpr,
    layout: gl.constexpr,
    EVEN_M: gl.constexpr,
    EVEN_N: gl.constexpr,
):
    """In-bounds mask of a ROWS x COLS block of C with m_left rows and n_left columns
    before the matrix edge (either may exceed the block). On a single ragged
    dimension it is [ROWS, 1] or [1, COLS]; the store broadcasts it over the block."""
    if not EVEN_M:
        r = gl.arange(0, ROWS, gl.SliceLayout(1, layout))[:, None]
    if not EVEN_N:
        c = gl.arange(0, COLS, gl.SliceLayout(0, layout))[None, :]
    if not EVEN_M and not EVEN_N:
        mask = (r < m_left) & (c < n_left)
    elif not EVEN_M:
        mask = r < m_left
    else:
        mask = c < n_left
    return mask


@gluon.jit
def store_c(
    acc,
    c_base,
    c_offsets,
    out_dtype: gl.constexpr,
    gStoreLayoutC: gl.constexpr,
    m_left=0,
    n_left=0,
    EVEN_M: gl.constexpr = True,
    EVEN_N: gl.constexpr = True,
):
    """Convert a quadrant to C's type and store layout and store it, masked to its first
    m_left rows / n_left columns on a ragged M / N."""
    c = acc.to(out_dtype)
    c = gl.convert_layout(c, layout=gStoreLayoutC)
    if EVEN_M and EVEN_N:
        gl.amd.cdna3.buffer_store(ptr=c_base, offsets=c_offsets, stored_value=c)
    else:
        mask = edge_mask(
            m_left, n_left, c.shape[0], c.shape[1], gStoreLayoutC, EVEN_M, EVEN_N
        )
        gl.amd.cdna3.buffer_store(
            ptr=c_base, offsets=c_offsets, stored_value=c, mask=mask
        )


@constexpr_function
def lane_map_matches(ll, half_m, half_n):
    """True if a HALF_M x HALF_N quadrant's MFMA layout (as a linear layout) has the
    bases lane_contiguous_offsets and the reduce-scatter readers assume: registers
    col 1, 2, then the column blocks 32 .. HALF_N / 2, then the row blocks 32 ..
    HALF_M / 2; lanes row 1, 2, 4, 8 and col 4, 8; warps col 16 and row 16."""
    reg = (
        [[0, 1], [0, 2]]
        + [[0, c] for c in (32, 64) if c < half_n]
        + [[r, 0] for r in (32, 64) if r < half_m]
    )
    return (
        [list(b) for b in ll.reg_bases] == reg
        and [list(b) for b in ll.lane_bases]
        == [[1, 0], [2, 0], [4, 0], [8, 0], [0, 4], [0, 8]]
        and [list(b) for b in ll.warp_bases] == [[0, 16], [16, 0]]
    )


@gluon.jit
def in_block_row(r):
    """Offset of element (r, 0) in its 32x32 MFMA block: row r is lane r % 16 of the
    top warps (0, 1), or from row 16 of the bottom warps (2, 3). r is a row of the
    quadrant; only its row inside the block, r % MFMA_BLOCK_DIM, counts, and
    lane_contiguous_offsets adds the block's offset."""
    return (r % MFMA_DIM) * MFMA_LANE_ELEMS + (r // MFMA_DIM % 2) * (
        2 * MFMA_WARP_ELEMS
    )


@gluon.jit
def in_block_col(c):
    """Offset of element (0, c) in its 32x32 MFMA block: column c is register c % 4 of
    lane group c // 4 % 4 (16 lanes each) in the left warps (0, 2), or from column 16
    the right warps (1, 3). Written as c plus terms constant over each 4 columns, not
    with c % 4, so axis analysis sees 4 contiguous columns (lane_contiguous_offsets).
    c already counts each step it crosses, so each term adds the rest: a lane group is
    MFMA_GROUP_ELEMS on, of which c counted MFMA_LANE_ELEMS, and the right warps are
    MFMA_WARP_ELEMS on, of which c counted MFMA_DIM."""
    GROUPS: gl.constexpr = MFMA_DIM // MFMA_LANE_ELEMS
    return (
        c
        + (c // MFMA_LANE_ELEMS % GROUPS) * (MFMA_GROUP_ELEMS - MFMA_LANE_ELEMS)
        + (c // MFMA_DIM % 2) * (MFMA_WARP_ELEMS - MFMA_DIM)
    )


@gluon.jit
def lane_contiguous_offsets(
    HALF_M: gl.constexpr, HALF_N: gl.constexpr, mfmaLayout: gl.constexpr
):
    """Element offsets of a partial quadrant in its slot, in the MFMA layout: [VQ
    register vectors][4 warps][64 lanes][4 fp32], VQ = (HALF_M / 32) (HALF_N / 32), the
    column blocks before the row blocks.

    Lane l of a wave writes elements 4l .. 4l + 3 of its 1 KiB block, so one
    16-byte-per-lane access covers 8 whole cache lines (row-major addressing would
    spread it over 16 rows). The map is a bit permutation of (row, col) that follows
    the layout's bases, checked at compile time."""
    gl.static_assert(
        lane_map_matches(
            gl.to_linear_layout(mfmaLayout, [HALF_M, HALF_N]), HALF_M, HALF_N
        ),
        "the MFMA layout's bases are not the order the partial map assumes",
    )
    VN: gl.constexpr = HALF_N // MFMA_BLOCK_DIM
    r = gl.arange(0, HALF_M, gl.SliceLayout(1, mfmaLayout))
    c = gl.arange(0, HALF_N, gl.SliceLayout(0, mfmaLayout))
    # The row and column parts add because every basis of the layout steps only the
    # row or only the column (lane_map_matches). Vector v = column block + VN row block.
    f = in_block_row(r) + (r // MFMA_BLOCK_DIM) * (VN * MFMA_BLOCK_ELEMS)
    g = in_block_col(c) + (c // MFMA_BLOCK_DIM) * (MFMA_BLOCK_ELEMS - MFMA_BLOCK_DIM)
    return f[:, None] + g[None, :]


@gluon.jit
def store_partial(
    acc,
    q: gl.constexpr,
    p_ptr,
    rank,
    p_offsets,
    HALF_M: gl.constexpr,
    HALF_N: gl.constexpr,
    CACHE: gl.constexpr = ".wt",
):
    """Quadrant q of a partial to slot P[rank]: write-through, or the default policy
    for the l2 exchange."""
    QUAD: gl.constexpr = HALF_M * HALF_N
    SLOT: gl.constexpr = 4 * QUAD
    gl.amd.cdna3.buffer_store(
        stored_value=acc,
        ptr=p_ptr + rank.to(gl.int64) * SLOT + q * QUAD,
        offsets=p_offsets,
        cache=CACHE,
    )


@gluon.jit
def owner_store(
    acc,
    psum,
    q: gl.constexpr,
    bias_ptr,
    pid_n,
    c_base,
    c_offsets,
    BLOCK_N: gl.constexpr,
    HALF_N: gl.constexpr,
    ADD_BIAS: gl.constexpr,
    out_dtype: gl.constexpr,
    gStoreLayoutC: gl.constexpr,
    mfmaLayout: gl.constexpr,
    m_left,
    n_left,
    EVEN_M: gl.constexpr,
    EVEN_N: gl.constexpr,
):
    """Quadrant q of C: the accumulator plus psum (the peers' partials) and the bias.
    Contributors' partials exclude the bias, so it is added once, here."""
    HALF_M: gl.constexpr = acc.shape[0]
    if ADD_BIAS:
        offs_pn = gl.arange(0, HALF_N, gl.SliceLayout(0, mfmaLayout))
        if EVEN_N:
            bias = gl.amd.cdna3.buffer_load(
                ptr=bias_ptr + pid_n * BLOCK_N + (q // 2) * HALF_N, offsets=offs_pn
            )
        else:
            bias = gl.amd.cdna3.buffer_load(
                ptr=bias_ptr + pid_n * BLOCK_N + (q // 2) * HALF_N,
                offsets=offs_pn,
                mask=offs_pn < n_left - (q // 2) * HALF_N,
                other=0.0,
            )
        psum = psum + bias.to(gl.float32)[None, :]
    store_c(
        acc + psum,
        c_base,
        c_offsets,
        out_dtype,
        gStoreLayoutC,
        m_left - (q % 2) * HALF_M,
        n_left - (q // 2) * HALF_N,
        EVEN_M,
        EVEN_N,
    )


@gluon.jit
def peer_sums(
    p_ptr,
    rank,
    n_peers,
    p_offsets,
    z,
    q0: gl.constexpr,
    NQ: gl.constexpr,
    PEER_STEP: gl.constexpr,
    HALF_M: gl.constexpr,
    HALF_N: gl.constexpr,
    MAX_PEERS: gl.constexpr,
):
    """The sums of quadrants q0 .. q0 + NQ - 1 (NQ = 1, 2 or 4) over the partials of
    peers rank + PEER_STEP i, i = 1 .. n_peers, as four tensors (the last 4 - NQ stay z,
    zeros); a peer's NQ quadrants are one batch of loads.
    - MAX_PEERS > 0: a compile-time loop over MAX_PEERS peers, the loads of peers past
      n_peers masked off, so that several peers' batches can be in flight at once.
    - Otherwise a run-time loop, each batch's loads written before its adds: written
      load, add, load, add, the scheduler reuses one quadrant's registers for the next
      and waits for each load in turn."""
    QUAD: gl.constexpr = HALF_M * HALF_N
    SLOT: gl.constexpr = 4 * QUAD
    s0, s1, s2, s3 = z, z, z, z
    if MAX_PEERS > 0:
        for j in gl.static_range(MAX_PEERS):
            peer = (rank + PEER_STEP * (1 + j)).to(gl.int64)
            live = j < n_peers
            s0 += gl.amd.cdna3.buffer_load(
                ptr=p_ptr + peer * SLOT + q0 * QUAD,
                offsets=p_offsets,
                mask=live,
                other=0.0,
                cache=".cv",
            ).to(gl.float32)
            if NQ > 1:
                s1 += gl.amd.cdna3.buffer_load(
                    ptr=p_ptr + peer * SLOT + (q0 + 1) * QUAD,
                    offsets=p_offsets,
                    mask=live,
                    other=0.0,
                    cache=".cv",
                ).to(gl.float32)
            if NQ > 2:
                s2 += gl.amd.cdna3.buffer_load(
                    ptr=p_ptr + peer * SLOT + (q0 + 2) * QUAD,
                    offsets=p_offsets,
                    mask=live,
                    other=0.0,
                    cache=".cv",
                ).to(gl.float32)
                s3 += gl.amd.cdna3.buffer_load(
                    ptr=p_ptr + peer * SLOT + (q0 + 3) * QUAD,
                    offsets=p_offsets,
                    mask=live,
                    other=0.0,
                    cache=".cv",
                ).to(gl.float32)
    else:
        for j in range(n_peers):
            peer = (rank + PEER_STEP * (1 + j)).to(gl.int64)
            x0 = gl.amd.cdna3.buffer_load(
                ptr=p_ptr + peer * SLOT + q0 * QUAD, offsets=p_offsets, cache=".cv"
            ).to(gl.float32)
            if NQ == 1:
                s0 += x0
            elif NQ == 2:
                x1 = gl.amd.cdna3.buffer_load(
                    ptr=p_ptr + peer * SLOT + (q0 + 1) * QUAD,
                    offsets=p_offsets,
                    cache=".cv",
                ).to(gl.float32)
                s0 += x0
                s1 += x1
            else:
                x1 = gl.amd.cdna3.buffer_load(
                    ptr=p_ptr + peer * SLOT + (q0 + 1) * QUAD,
                    offsets=p_offsets,
                    cache=".cv",
                ).to(gl.float32)
                x2 = gl.amd.cdna3.buffer_load(
                    ptr=p_ptr + peer * SLOT + (q0 + 2) * QUAD,
                    offsets=p_offsets,
                    cache=".cv",
                ).to(gl.float32)
                x3 = gl.amd.cdna3.buffer_load(
                    ptr=p_ptr + peer * SLOT + (q0 + 3) * QUAD,
                    offsets=p_offsets,
                    cache=".cv",
                ).to(gl.float32)
                s0 += x0
                s1 += x1
                s2 += x2
                s3 += x3
    return s0, s1, s2, s3


@gluon.jit
def owner_ending(
    accs,
    is_partial,
    n_peers,
    rank,
    pid_n,
    bias_ptr,
    p_ptr,
    locks_ptr,
    c_bases,
    c_offsets,
    m_left,
    n_left,
    BLOCK_N: gl.constexpr,
    PEER_STEP: gl.constexpr,
    ADD_BIAS: gl.constexpr,
    out_dtype: gl.constexpr,
    gStoreLayoutC: gl.constexpr,
    mfmaLayout: gl.constexpr,
    EVEN_M: gl.constexpr,
    EVEN_N: gl.constexpr,
    QPB: gl.constexpr,
    MAX_PEERS: gl.constexpr,
):
    """The owner ending of a segment. A contributor (is_partial) stores its four
    quadrants to P[rank], drains and raises locks[rank]. The owner waits for its
    n_peers peers' flags (rank + PEER_STEP i), reads their partials QPB quadrants at a
    time (16 fp32 loads per wave: 4 at 128x128, 2 at 256x128 / 128x256, 1 at 256x256;
    peer_sums), adds each batch's sums into the accumulator once, outside the peer loop
    (a loop or branch that carries the AGPR accumulator spills), and stores that batch
    of C before the next batch's reads, so only one batch's sums are live. Each partial
    has one reader, so the owner lowers its peers' flags for the next launch; the
    kernel boundary orders that against it."""
    HALF_M: gl.constexpr = accs[0].shape[0]
    HALF_N: gl.constexpr = accs[0].shape[1]
    for pj in range(n_peers):
        while (
            gl.atomic_add(
                locks_ptr + rank + PEER_STEP * (1 + pj), 0, sem="relaxed", scope="gpu"
            )
            == 0
        ):
            pass
    if n_peers > 0:
        asm_fence("s_nop 0")

    # With a run-time peer loop over one quadrant per batch (256x256: end to end, or
    # above 8 programs per tile) a contributor stores each quadrant just before its
    # batch: storing all four first is about 1% slower on two-tile there.
    STORE_EACH: gl.constexpr = QPB == 1 and MAX_PEERS == 0
    p_offsets = lane_contiguous_offsets(HALF_M, HALF_N, mfmaLayout)
    z = gl.zeros((HALF_M, HALF_N), gl.float32, mfmaLayout)
    if not STORE_EACH:
        if is_partial != 0:
            for q in gl.static_range(4):
                store_partial(accs[q], q, p_ptr, rank, p_offsets, HALF_M, HALF_N)
    for q0 in gl.static_range(0, 4, QPB):
        if STORE_EACH:
            if is_partial != 0:
                store_partial(accs[q0], q0, p_ptr, rank, p_offsets, HALF_M, HALF_N)
        sums = peer_sums(
            p_ptr,
            rank,
            n_peers,
            p_offsets,
            z,
            q0,
            QPB,
            PEER_STEP,
            HALF_M,
            HALF_N,
            MAX_PEERS,
        )
        if is_partial == 0:
            for i in gl.static_range(QPB):
                owner_store(
                    accs[q0 + i],
                    sums[i],
                    q0 + i,
                    bias_ptr,
                    pid_n,
                    c_bases[q0 + i],
                    c_offsets,
                    BLOCK_N,
                    HALF_N,
                    ADD_BIAS,
                    out_dtype,
                    gStoreLayoutC,
                    mfmaLayout,
                    m_left,
                    n_left,
                    EVEN_M,
                    EVEN_N,
                )

    if is_partial != 0:
        # The flag may only become visible after all four waves' partial stores have
        # completed.
        drain_and_barrier()
        gl.atomic_xchg(locks_ptr + rank, 1, sem="relaxed", scope="gpu")
    for pj in range(n_peers):
        gl.atomic_xchg(
            locks_ptr + rank + PEER_STEP * (1 + pj), 0, sem="relaxed", scope="gpu"
        )


@gluon.jit
def block_col(v, VN: gl.constexpr):
    """Column of register vector (32x32 block) v in its quadrant, VN blocks across."""
    return (v % VN) * 32


@gluon.jit
def to_band(x, G: gl.constexpr, VN: gl.constexpr):
    """(G, 32, 32) aligned blocks -> their 2-D band, VN blocks across a quadrant. Block
    index bits from the lowest: the column blocks, then the row blocks, so the band is
    RB = G / CB row blocks of CB = min(G, VN) blocks (VN = 4: G = 1, 2, 4: 32 x 32 G;
    8, 16: 8 G x 128)."""
    CB: gl.constexpr = min(G, VN)
    RB: gl.constexpr = G // CB
    return gl.reshape(
        gl.permute(gl.reshape(x, [RB, CB, 32, 32]), [0, 2, 1, 3]),
        [RB * 32, CB * 32],
    )


@gluon.jit
def band_pos(
    g0,
    layout: gl.constexpr,
    ROWS: gl.constexpr,
    COLS: gl.constexpr,
    HALF_M: gl.constexpr = 128,
    HALF_N: gl.constexpr = 128,
):
    """Tile rows and columns of the band of the aligned blocks g0 .. g0 + G - 1."""
    VN: gl.constexpr = HALF_N // 32
    VQ: gl.constexpr = HALF_M * HALF_N // 1024
    r = gl.arange(0, ROWS, gl.SliceLayout(1, layout))[:, None]
    c = gl.arange(0, COLS, gl.SliceLayout(0, layout))[None, :]
    q = g0 // VQ
    v0 = g0 % VQ
    row0 = (v0 // VN) * 32
    col0 = block_col(v0, VN)
    return (q % 2) * HALF_M + row0 + r, (q // 2) * HALF_N + col0 + c


@gluon.jit
def reduce_scatter_slice(
    p_ptr,
    bias_ptr,
    c_tile,
    b,
    j,
    n,
    readers,
    pid_n,
    stride_cm,
    stride_cn,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    ADD_BIAS: gl.constexpr,
    NB: gl.constexpr,
    U: gl.constexpr,
    out_dtype: gl.constexpr,
    LOAD_CACHE: gl.constexpr,
    m_left,
    n_left,
    EVEN_M: gl.constexpr,
    EVEN_N: gl.constexpr,
):
    """Reader j of `readers` (a power of two) of a tile split n ways: sums blocks
    [NBLK j / readers, NBLK (j + 1) / readers) over the n partials in slots P[b .. b + n
    - 1], adds the bias and stores them to C. NB blocks (NB <= VQ, so a group stays in
    one quadrant) at a time, U partials of a group per trip, so NB U loads per lane are
    in flight. Each group is reshaped to its 2-D band and stored along C's rows, 16
    bytes per lane (8 for a single 32x32 block)."""
    HALF_M: gl.constexpr = BLOCK_M // 2
    HALF_N: gl.constexpr = BLOCK_N // 2
    VN: gl.constexpr = HALF_N // 32
    VQ: gl.constexpr = VN * (HALF_M // 32)
    NBLK: gl.constexpr = 4 * VQ
    gl.static_assert(NB <= VQ, "a reduce-scatter group must stay inside one quadrant")
    SLOT: gl.constexpr = BLOCK_M * BLOCK_N
    blk_lo = (NBLK * j) // readers
    blk_hi = (NBLK * (j + 1)) // readers
    # Rows before columns in lane order: lane l gets row l % 16 and columns 4 (l // 16)
    # .. + 3, which in_blk puts at 4 l .. 4 l + 3, so each 4-lane group reads 64
    # contiguous bytes. Columns first, lanes 0-3 hit 4 lines 256 bytes apart and the L1
    # stalls on tag conflicts (reads take twice as long).
    L3: gl.constexpr = gl.BlockedLayout([1, 1, 4], [1, 16, 4], [1, 2, 2], [1, 2, 0])
    i = gl.arange(0, NB, gl.SliceLayout(1, gl.SliceLayout(2, L3)))[:, None, None]
    r = gl.arange(0, 32, gl.SliceLayout(0, gl.SliceLayout(2, L3)))[None, :, None]
    c = gl.arange(0, 32, gl.SliceLayout(0, gl.SliceLayout(1, L3)))[None, None, :]
    # in_block_row(r) + in_block_col(c) (the two add: lane_contiguous_offsets) for rows
    # and columns 0-31, written without the modulo on bit 16 (the helpers' form costs
    # instructions here).
    GROUPS: gl.constexpr = MFMA_DIM // MFMA_LANE_ELEMS
    in_blk = (
        (r % MFMA_DIM) * MFMA_LANE_ELEMS
        + (r // MFMA_DIM) * (2 * MFMA_WARP_ELEMS)
        + c
        + (c // MFMA_LANE_ELEMS % GROUPS) * (MFMA_GROUP_ELEMS - MFMA_LANE_ELEMS)
        + (c // MFMA_DIM) * (MFMA_WARP_ELEMS - MFMA_DIM)
    )
    cb = gl.arange(0, 32, gl.SliceLayout(0, gl.SliceLayout(1, L3)))[None, :]
    ib = gl.arange(0, NB, gl.SliceLayout(1, gl.SliceLayout(1, L3)))[:, None]
    # The band: min(NB, VN) blocks across, the rest down.
    ROWS: gl.constexpr = 32 * (NB // min(NB, VN))
    COLS: gl.constexpr = 32 * min(NB, VN)
    SL: gl.constexpr = coalesced_store_layout(
        ROWS, COLS, out_dtype.primitive_bitwidth // 8, gl.num_warps()
    )
    for g in range(blk_lo, blk_hi, NB):
        blk = g + i
        valid = blk < blk_hi
        p_off = blk * MFMA_BLOCK_ELEMS + in_blk
        s = gl.zeros((NB, 32, 32), gl.float32, L3)
        for p0 in range(0, n, U):
            for u in gl.static_range(U):
                slot = p_ptr + (b + p0 + u).to(gl.int64) * SLOT
                s += gl.amd.cdna3.buffer_load(
                    ptr=slot,
                    offsets=p_off,
                    mask=valid & (p0 + u < n),
                    other=0.0,
                    cache=LOAD_CACHE,
                ).to(gl.float32)
        if ADD_BIAS:
            # Block q VQ + v: quadrant q (tl, bl, tr, br); register vector v's bits are
            # the column blocks, then the row blocks.
            qb = (g + ib) // VQ
            vb = (g + ib) % VQ
            col_b = (qb // 2) * HALF_N + block_col(vb, VN) + cb
            bias_mask = (g + ib) < blk_hi
            if not EVEN_N:
                bias_mask = bias_mask & (col_b < n_left)
            bias = gl.amd.cdna3.buffer_load(
                ptr=bias_ptr + pid_n * BLOCK_N, offsets=col_b, mask=bias_mask, other=0.0
            )
            s = s + bias.to(gl.float32)[:, None, :]
        cs = gl.convert_layout(to_band(s.to(out_dtype), NB, VN), SL)
        trow, tcol = band_pos(g, SL, ROWS, COLS, HALF_M, HALF_N)
        if EVEN_M and EVEN_N:
            gl.amd.cdna3.buffer_store(
                stored_value=cs, ptr=c_tile, offsets=trow * stride_cm + tcol * stride_cn
            )
        else:
            gl.amd.cdna3.buffer_store(
                stored_value=cs,
                ptr=c_tile,
                offsets=trow * stride_cm + tcol * stride_cn,
                mask=(trow < m_left) & (tcol < n_left),
            )


@gluon.jit
def rs_ending(
    accs,
    rank,
    j,
    n_tile,
    r_tile,
    pid_n,
    bias_ptr,
    p_ptr,
    sync_ptr,
    c_bases,
    c_offsets,
    stride_cm,
    stride_cn,
    m_left,
    n_left,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    ADD_BIAS: gl.constexpr,
    RS_NB: gl.constexpr,
    RS_U: gl.constexpr,
    out_dtype: gl.constexpr,
    gStoreLayoutC: gl.constexpr,
    mfmaLayout: gl.constexpr,
    EVEN_M: gl.constexpr,
    EVEN_N: gl.constexpr,
    RS_E: gl.constexpr = 0,
    BAR_STRIDE: gl.constexpr = 0,
):
    """The reduce-scatter ending of tile-aligned segment j of n_tile (its slots are
    rank - j .. + n_tile - 1). A whole tile stores C directly. A split tile's programs
    all store their partial and arrive on the tile's barrier; the first
    min(n_tile, r_tile) wait on it and each reduces slice j of r_tile
    (reduce_scatter_slice).

    With RS_E > 0, r_tile = n_tile + RS_E: RS_E idle programs read slices n_tile ..
    r_tile - 1 (rs_extra_reader) and arrive too, so the barrier counts n_tile + RS_E.
    Tile t's barrier is slot t BAR_STRIDE (0: rank - j = t n_tile), which gives each
    tile its own cache line (rs_readers)."""
    HALF_M: gl.constexpr = BLOCK_M // 2
    HALF_N: gl.constexpr = BLOCK_N // 2
    is_split = gl.to_tensor(n_tile > 1).to(gl.int32)
    p_offsets = lane_contiguous_offsets(HALF_M, HALF_N, mfmaLayout)
    z = gl.zeros((HALF_M, HALF_N), gl.float32, mfmaLayout)
    for q in gl.static_range(4):
        if is_split != 0:
            store_partial(accs[q], q, p_ptr, rank, p_offsets, HALF_M, HALF_N)
        if is_split == 0:
            owner_store(
                accs[q],
                z,
                q,
                bias_ptr,
                pid_n,
                c_bases[q],
                c_offsets,
                BLOCK_N,
                HALF_N,
                ADD_BIAS,
                out_dtype,
                gStoreLayoutC,
                mfmaLayout,
                m_left,
                n_left,
                EVEN_M,
                EVEN_N,
            )
    if n_tile > 1:
        # The arrival may only become visible after all four waves' partial stores have
        # completed.
        drain_and_barrier()
        is_reader = j < r_tile
        slot = rank - j
        if BAR_STRIDE > 0:
            slot = slot // n_tile * BAR_STRIDE
        tile_barrier(sync_ptr, slot, n_tile + RS_E, is_reader)
        if is_reader:
            reduce_scatter_slice(
                p_ptr,
                bias_ptr,
                c_bases[0] + pid_n * BLOCK_N * stride_cn,
                rank - j,
                j,
                n_tile,
                r_tile,
                pid_n,
                stride_cm,
                stride_cn,
                BLOCK_M,
                BLOCK_N,
                ADD_BIAS,
                RS_NB,
                RS_U,
                out_dtype,
                ".cv",
                m_left,
                n_left,
                EVEN_M,
                EVEN_N,
            )


@gluon.jit
def rs_extra_reader(
    t,
    jr,
    total_full_tiles,
    p_ptr,
    bias_ptr,
    c_ptr,
    sync_ptr,
    M,
    N,
    stride_cm,
    stride_cn,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    GROUP_SIZE_M: gl.constexpr,
    SPLIT_N: gl.constexpr,
    R: gl.constexpr,
    E: gl.constexpr,
    NB: gl.constexpr,
    U: gl.constexpr,
    ADD_BIAS: gl.constexpr,
    EVEN_M: gl.constexpr,
    EVEN_N: gl.constexpr,
    BAR_STRIDE: gl.constexpr,
):
    """An idle program as reader jr (SPLIT_N <= jr < R) of Stream-K tile t (rs_ending
    with RS_E > 0). It arrives on the tile's barrier at once: a waiter that only polled
    could read the generation after the last arrival had bumped it and spin forever.
    Then it waits for the producers and reduces its slice."""
    tile_barrier(sync_ptr, t * BAR_STRIDE, SPLIT_N + E, True)
    pid_m, pid_n = group_m_tile_coords(
        total_full_tiles + t, M, N, BLOCK_M, BLOCK_N, GROUP_SIZE_M
    )
    m_left = M - pid_m * BLOCK_M
    n_left = gl.multiple_of(N - pid_n * BLOCK_N, 8)
    c_tile = (
        c_ptr
        + pid_m.to(gl.int64) * BLOCK_M * stride_cm
        + pid_n.to(gl.int64) * BLOCK_N * stride_cn
    )
    reduce_scatter_slice(
        p_ptr,
        bias_ptr,
        c_tile,
        t * SPLIT_N,
        jr,
        SPLIT_N,
        R,
        pid_n,
        stride_cm,
        stride_cn,
        BLOCK_M,
        BLOCK_N,
        ADD_BIAS,
        NB,
        U,
        c_ptr.dtype.element_ty,
        ".cv",
        m_left,
        n_left,
        EVEN_M,
        EVEN_N,
    )


@gluon.jit
def l2_rs_ending(
    accs,
    rank,
    j,
    n_tile,
    r_tile,
    pid_n,
    bias_ptr,
    p_ptr,
    sync_ptr,
    c_bases,
    c_offsets,
    stride_cm,
    stride_cn,
    m_left,
    n_left,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    ADD_BIAS: gl.constexpr,
    RS_NB: gl.constexpr,
    RS_U: gl.constexpr,
    NUM_PROGRAMS: gl.constexpr,
    NUM_XCDS: gl.constexpr,
    out_dtype: gl.constexpr,
    gStoreLayoutC: gl.constexpr,
    mfmaLayout: gl.constexpr,
    EVEN_M: gl.constexpr,
    EVEN_N: gl.constexpr,
):
    """XCD-local reduce-scatter ending with graph-replay-safe flag rows: rs_ending
    with the partials kept in the XCD's L2 (default-policy stores, .cg loads)."""
    HALF_M: gl.constexpr = BLOCK_M // 2
    HALF_N: gl.constexpr = BLOCK_N // 2
    is_split = gl.to_tensor(n_tile > 1).to(gl.int32)
    p_offsets = lane_contiguous_offsets(HALF_M, HALF_N, mfmaLayout)
    z = gl.zeros((HALF_M, HALF_N), gl.float32, mfmaLayout)
    for q in gl.static_range(4):
        if is_split != 0:
            store_partial(accs[q], q, p_ptr, rank, p_offsets, HALF_M, HALF_N, "")
        if is_split == 0:
            owner_store(
                accs[q],
                z,
                q,
                bias_ptr,
                pid_n,
                c_bases[q],
                c_offsets,
                BLOCK_N,
                HALF_N,
                ADD_BIAS,
                out_dtype,
                gStoreLayoutC,
                mfmaLayout,
                m_left,
                n_left,
                EVEN_M,
                EVEN_N,
            )
    if n_tile > 1:
        drain_and_barrier()
        is_reader = j < r_tile
        l2_tile_barrier(
            sync_ptr, rank - j, n_tile, r_tile, j, is_reader, NUM_PROGRAMS, NUM_XCDS
        )
        if is_reader:
            reduce_scatter_slice(
                p_ptr,
                bias_ptr,
                c_bases[0] + pid_n * BLOCK_N * stride_cn,
                rank - j,
                j,
                n_tile,
                r_tile,
                pid_n,
                stride_cm,
                stride_cn,
                BLOCK_M,
                BLOCK_N,
                ADD_BIAS,
                RS_NB,
                RS_U,
                out_dtype,
                ".cg",
                m_left,
                n_left,
                EVEN_M,
                EVEN_N,
            )


@gluon.jit
def quadrant_segment(
    tile_id,
    k0,
    n_pairs,
    is_partial,
    n_peers,
    rank,
    j,
    n_tile,
    r_tile,
    a_ptr,
    b_ptr,
    c_ptr,
    bias_ptr,
    p_ptr,
    locks_ptr,
    sync_ptr,
    smem,
    kofs,
    strides,
    M,
    N,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_K: gl.constexpr,
    GROUP_SIZE_M: gl.constexpr,
    ADD_BIAS: gl.constexpr,
    PEER_STEP: gl.constexpr,
    REDUCE_SCATTER: gl.constexpr,
    RS_NB: gl.constexpr,
    RS_U: gl.constexpr,
    L2: gl.constexpr,
    NUM_PROGRAMS: gl.constexpr,
    NUM_XCDS: gl.constexpr,
    mfmaLayout: gl.constexpr,
    gLoadLayoutA: gl.constexpr,
    gLoadLayoutB: gl.constexpr,
    EVEN_M: gl.constexpr,
    EVEN_N: gl.constexpr,
    EVEN_K: gl.constexpr,
    K_PAD: gl.constexpr,
    K_RES: gl.constexpr,
    SEG_LOOP: gl.constexpr,
    OWNER_QPB: gl.constexpr,
    OWNER_MAX_PEERS: gl.constexpr,
    PREFETCHED: gl.constexpr = False,
    a_top_in=None,
    b_left_in=None,
    RS_E: gl.constexpr = 0,
    BAR_STRIDE: gl.constexpr = 0,
):
    """K-steps [k0, k0 + 2 n_pairs) of tile tile_id (n_pairs >= 1), from its own
    prologue, then the segment's ending: XCD-local reduce-scatter (l2_rs_ending, L2),
    reduce-scatter (rs_ending) or the owner ending (owner_ending). rank is the
    program's workspace slot, j its chunk (role 0 owns the tile), n_tile / r_tile the
    tile's programs and readers.

    Pair p of a tile (K-steps 2p, 2p + 1) is at columns 2 BLOCK_K (p - 1) for p >= 1,
    and pair 0 is the last pair, at K_PAD - 2 BLOCK_K, loaded on a ragged K with the
    lanes past K masked. So only the segment
    that starts at k0 = 0 masks, in its prologue. Every pair runs through the
    main-loop body; the last one's prefetch reloads the last pair (clamped), which the
    drain discards.

    SEG_LOOP: the caller runs segments in a loop (the end-to-end path), so the C row
    offsets are made tile-dependent (clamped to the tile's last row); otherwise the
    compiler hoists them above the loop and holds them, spilled, across it.

    PREFETCHED: the first pair is already in flight in the state load_first_pair
    leaves (the last persistent trip's epilogue loaded it), with its first operands in
    a_top_in / b_left_in."""
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn = strides
    a_offsets, a_bot_shift, b_offsets, b_right_shift = kofs
    dotOpLayoutA: gl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=mfmaLayout, k_width=8
    )
    dotOpLayoutB: gl.constexpr = gl.DotOperandLayout(
        operand_index=1, parent=mfmaLayout, k_width=8
    )
    pid_m, pid_n = group_m_tile_coords(tile_id, M, N, BLOCK_M, BLOCK_N, GROUP_SIZE_M)
    m_left = M - pid_m * BLOCK_M
    n_left = gl.multiple_of(N - pid_n * BLOCK_N, 8)

    a_addr = a_load_addressing(
        pid_m,
        a_offsets,
        a_bot_shift,
        M,
        stride_am,
        stride_ak,
        BLOCK_M,
        BLOCK_K,
        gLoadLayoutA,
        EVEN_M,
    )
    b_addr = b_load_addressing(
        pid_n,
        b_offsets,
        b_right_shift,
        N,
        stride_bk,
        stride_bn,
        BLOCK_N,
        BLOCK_K,
        gLoadLayoutB,
        EVEN_N,
    )

    # Pair i of the segment (i >= 1) at a_tile + 2 BLOCK_K i; its first pair at a_tile,
    # or for k0 = 0 the last pair (partial and masked at K on a ragged K).
    a_row = a_ptr + pid_m.to(gl.int64) * BLOCK_M * stride_am
    b_row = b_ptr + pid_n.to(gl.int64) * BLOCK_N * stride_bn
    a_tile = a_row + (k0 - 2).to(gl.int64) * BLOCK_K * stride_ak
    b_tile = b_row + (k0 - 2).to(gl.int64) * BLOCK_K * stride_bk
    if PREFETCHED:
        a_top, b_left = a_top_in, b_left_in
    else:
        first_col = gl.where(k0 == 0, K_PAD - 2 * BLOCK_K, (k0 - 2) * BLOCK_K).to(
            gl.int64
        )
        if EVEN_K:
            mask_a0, mask_b0, mask_a1, mask_b1 = None, None, None, None
        else:
            k_hi = gl.where(k0 == 0, 2 * BLOCK_K - K_RES, 2 * BLOCK_K)
            mask_a0, mask_b0, mask_a1, mask_b1 = tail_pair_masks(
                k_hi, K_RES, BLOCK_M, BLOCK_N, BLOCK_K, gLoadLayoutA, gLoadLayoutB
            )
        a_top, b_left = load_first_pair(
            a_row + first_col * stride_ak,
            b_row + first_col * stride_bk,
            smem,
            a_addr,
            b_addr,
            dotOpLayoutA,
            dotOpLayoutB,
            mask_a0,
            mask_b0,
            mask_a1,
            mask_b1,
        )
    # The last pair's clamped prefetch must stay inside the row: for k0 = 0, a_tile
    # itself is a pair before it.
    nxt_lo = (k0 == 0).to(gl.int64) * 2

    # The first pair is peeled: the constant-zero accumulator becomes the MFMA's
    # inline-0 C operand, so no AGPRs are zeroed per segment (a loop-carried zero
    # would be materialised).
    accs = init_acc(bias_ptr, pid_n, BLOCK_M, BLOCK_N, mfmaLayout, False)
    nxt = gl.maximum(gl.minimum(1, n_pairs - 1).to(gl.int64) * 2, nxt_lo)
    accs, a_top, b_left = k_step_pair(
        accs,
        a_top,
        b_left,
        smem,
        a_tile + nxt * BLOCK_K * stride_ak,
        b_tile + nxt * BLOCK_K * stride_bk,
        a_addr,
        b_addr,
        dotOpLayoutA,
        dotOpLayoutB,
    )
    for i in range(1, n_pairs):
        nxt = gl.maximum(gl.minimum(i + 1, n_pairs - 1).to(gl.int64) * 2, nxt_lo)
        accs, a_top, b_left = k_step_pair(
            accs,
            a_top,
            b_left,
            smem,
            a_tile + nxt * BLOCK_K * stride_ak,
            b_tile + nxt * BLOCK_K * stride_bk,
            a_addr,
            b_addr,
            dotOpLayoutA,
            dotOpLayoutB,
        )
    # The clamped prefetch is still in flight, and the next segment's prologue
    # overwrites the buffers other waves may still be reading.
    drain_and_barrier()

    out_dtype: gl.constexpr = c_ptr.dtype.element_ty
    gStoreLayoutC: gl.constexpr = coalesced_store_layout(
        BLOCK_M // 2, BLOCK_N // 2, out_dtype.primitive_bitwidth // 8, gl.num_warps()
    )
    offs_cm = gl.arange(0, BLOCK_M // 2, gl.SliceLayout(1, gStoreLayoutC))
    offs_cn = gl.arange(0, BLOCK_N // 2, gl.SliceLayout(0, gStoreLayoutC))
    c_tl_base = c_ptr + pid_m.to(gl.int64) * BLOCK_M * stride_cm
    col = pid_n * BLOCK_N + offs_cn
    if SEG_LOOP:
        c_rows = gl.multiple_of(stride_cm * gl.minimum(offs_cm, m_left - 1), 8)
    else:
        c_rows = gl.multiple_of(stride_cm * offs_cm, 8)
    c_offsets = c_rows[:, None] + stride_cn * col[None, :]
    c_tr_base = c_tl_base + BLOCK_N * stride_cn // 2
    c_bl_base = c_tl_base + BLOCK_M * stride_cm // 2
    c_br_base = c_bl_base + BLOCK_N * stride_cn // 2
    c_bases = (c_tl_base, c_bl_base, c_tr_base, c_br_base)

    if L2:
        l2_rs_ending(
            accs,
            rank,
            j,
            n_tile,
            r_tile,
            pid_n,
            bias_ptr,
            p_ptr,
            sync_ptr,
            c_bases,
            c_offsets,
            stride_cm,
            stride_cn,
            m_left,
            n_left,
            BLOCK_M,
            BLOCK_N,
            ADD_BIAS,
            RS_NB,
            RS_U,
            NUM_PROGRAMS,
            NUM_XCDS,
            out_dtype,
            gStoreLayoutC,
            mfmaLayout,
            EVEN_M,
            EVEN_N,
        )
    elif REDUCE_SCATTER:
        rs_ending(
            accs,
            rank,
            j,
            n_tile,
            r_tile,
            pid_n,
            bias_ptr,
            p_ptr,
            sync_ptr,
            c_bases,
            c_offsets,
            stride_cm,
            stride_cn,
            m_left,
            n_left,
            BLOCK_M,
            BLOCK_N,
            ADD_BIAS,
            RS_NB,
            RS_U,
            out_dtype,
            gStoreLayoutC,
            mfmaLayout,
            EVEN_M,
            EVEN_N,
            RS_E,
            BAR_STRIDE,
        )
    else:
        owner_ending(
            accs,
            is_partial,
            n_peers,
            rank,
            pid_n,
            bias_ptr,
            p_ptr,
            locks_ptr,
            c_bases,
            c_offsets,
            m_left,
            n_left,
            BLOCK_N,
            PEER_STEP,
            ADD_BIAS,
            out_dtype,
            gStoreLayoutC,
            mfmaLayout,
            EVEN_M,
            EVEN_N,
            OWNER_QPB,
            OWNER_MAX_PEERS,
        )


MFMA_SCHEDULE_HINT = "mfma-schedule"


def schedule_hint():
    """The schedule_hint the kernel launches with: the MFMA scheduler, unless
    AITER_MFMA_SCHED=0 (an unscheduled A/B run)."""
    return "" if os.environ.get("AITER_MFMA_SCHED", "1") == "0" else MFMA_SCHEDULE_HINT


# ---------------------------------------------------------------------------------
# Host side: configuration, topology and plan (no tensors; the launch is the wrapper's)
# ---------------------------------------------------------------------------------

TILE_SIDES = (64, 128, 256)
BLOCK_K_SIDES = (64, 128)
STREAMK_POLICIES = ("dp", "one_tile", "two_tile", "spread")
STREAMK_FIXUPS = ("owner", "rs")
STREAMK_EXCHANGES = ("wt", "l2")
CONFIG_KEYS = {
    "BLOCK_M": "block_m",
    "BLOCK_N": "block_n",
    "BLOCK_K": "block_k",
    "GROUP_SIZE_M": "group_size_m",
    "STREAMK_POLICY": "policy",
    "STREAMK_FIXUP": "fixup",
    "SPLIT_N": "split_n",
    "STREAMK_EXCHANGE": "exchange",
}
MAX_TILE_ELEMS = max(TILE_SIDES) ** 2


@dataclass(frozen=True)
class TileSpec:
    """One supported tile: sides in TILE_SIDES, BLOCK_K in BLOCK_K_SIDES."""

    block_m: int
    block_n: int
    block_k: int

    def __post_init__(self):
        if any(type(v) is not int for v in (self.block_m, self.block_n, self.block_k)):
            raise TypeError("tile dimensions must be integers")
        if self.block_m not in TILE_SIDES or self.block_n not in TILE_SIDES:
            raise ValueError(
                f"BLOCK_M and BLOCK_N must be in {TILE_SIDES}, "
                f"got {self.block_m}x{self.block_n}"
            )
        if self.block_k not in BLOCK_K_SIDES:
            raise ValueError(f"BLOCK_K must be in {BLOCK_K_SIDES}, got {self.block_k}")

    @property
    def blocks(self):
        """32x32 reduce-scatter blocks in the tile."""
        return self.block_m * self.block_n // (32 * 32)

    @property
    def quadrant_blocks(self):
        return self.blocks // 4


@dataclass(frozen=True)
class StreamKConfig:
    """A complete config (CONFIG_KEYS), validated on its own; whether it fits a
    shape is checked by _launch_constants."""

    block_m: int
    block_n: int
    block_k: int
    group_size_m: int
    policy: str
    fixup: str
    split_n: int | None
    exchange: str

    def __post_init__(self):
        TileSpec(self.block_m, self.block_n, self.block_k)
        if type(self.group_size_m) is not int or self.group_size_m <= 0:
            raise ValueError("GROUP_SIZE_M must be a positive integer")
        if self.policy not in STREAMK_POLICIES:
            raise ValueError(f"invalid STREAMK_POLICY {self.policy!r}")
        if self.fixup not in STREAMK_FIXUPS:
            raise ValueError(f"invalid STREAMK_FIXUP {self.fixup!r}")
        if self.exchange not in STREAMK_EXCHANGES:
            raise ValueError(f"invalid STREAMK_EXCHANGE {self.exchange!r}")
        if self.policy == "one_tile":
            if type(self.split_n) is not int or self.split_n < 2:
                raise ValueError("one_tile requires an integer SPLIT_N of at least 2")
        elif self.split_n is not None:
            raise ValueError(f"{self.policy} takes SPLIT_N=None")
        if self.policy != "one_tile" and self.fixup != "owner":
            raise ValueError(f"{self.policy} requires the owner fixup")
        if self.exchange == "l2" and (self.policy, self.fixup) != ("one_tile", "rs"):
            raise ValueError("the l2 exchange requires one_tile with rs")

    @property
    def tile(self):
        return TileSpec(self.block_m, self.block_n, self.block_k)

    @classmethod
    def from_dict(cls, values):
        """Parse a config dict with exactly the CONFIG_KEYS."""
        if isinstance(values, cls):
            return values
        if not isinstance(values, Mapping):
            raise TypeError("the Stream-K config must be a mapping")
        unknown = sorted(set(values) - set(CONFIG_KEYS))
        if unknown:
            raise ValueError(f"unknown Stream-K config keys {unknown}")
        missing = sorted(set(CONFIG_KEYS) - set(values))
        if missing:
            raise ValueError(f"incomplete Stream-K config; missing {missing}")
        return cls(**{CONFIG_KEYS[key]: value for key, value in values.items()})


@dataclass(frozen=True)
class ReduceScatterConfig:
    readers: int
    blocks_per_group: int
    partials_per_trip: int
    # Idle programs per tile that read too, and the tile barrier's slot stride
    # (0: slot t split_n).
    extra: int = 0
    barrier_stride: int = 0


def pow2_floor(x):
    return 1 << (x.bit_length() - 1)


def reduce_scatter_config(tile, split_n, loads_in_flight=16, readers=None):
    """The tile's own power-of-two readers (at most its 32x32 blocks), or
    ``readers``, and the slice's blocks per group and partials per trip."""
    if readers is None:
        readers = min(pow2_floor(split_n), tile.blocks)
    nb = max(1, min(tile.blocks // readers, loads_in_flight, tile.quadrant_blocks))
    return ReduceScatterConfig(
        readers=readers,
        blocks_per_group=nb,
        partials_per_trip=max(loads_in_flight // nb, 1),
    )


def rs_readers(tile, streamk_tiles, split_n, num_programs, loads_in_flight=16):
    """The one-tile write-through reduce-scatter with every resident program reading:
    R = max(min(blocks, pow2_floor(P / S)), own readers) readers per tile, so
    R S <= P; when R > split_n the extra = R - split_n idle programs per tile join.
    Each tile's barrier gets its own 128-byte line (16 slots of 2 words) where it
    fits: packed, the idle readers' polling slows the producers' arrivals on the
    shared line. With extra = 0 this is the own-readers rule with only the barrier
    slots moved."""
    own = reduce_scatter_config(tile, split_n, loads_in_flight).readers
    per_tile = num_programs // streamk_tiles
    readers = max(min(tile.blocks, pow2_floor(per_tile)), own)
    rs = reduce_scatter_config(tile, split_n, loads_in_flight, readers)
    return ReduceScatterConfig(
        readers=readers,
        blocks_per_group=rs.blocks_per_group,
        partials_per_trip=rs.partials_per_trip,
        extra=readers - min(split_n, readers),
        barrier_stride=max(split_n, min(16, per_tile)),
    )


@functools.lru_cache(maxsize=None)
def streamk_topology():
    """The validated (resident programs, XCDs) of the current device: one program
    per CU."""
    num_programs, num_xcds = int(get_num_sms()), int(get_num_xcds())
    if num_programs <= 0 or num_xcds <= 0 or num_programs % num_xcds:
        raise RuntimeError(
            f"invalid runtime topology: {num_programs} CUs, {num_xcds} XCDs"
        )
    return num_programs, num_xcds


def sync_words(num_programs, num_xcds):
    """int32 words of the flag buffer before the owner locks: the reduce-scatter
    (count, generation) pairs, then one programs-per-XCD L2 flag row per program."""
    return 2 * num_programs + num_programs * (num_programs // num_xcds)


def k_pad(K, block_k):
    """K virtually padded to a whole number of K-step pairs."""
    return triton.cdiv(K, 2 * block_k) * 2 * block_k


TILE_KEYS = ("BLOCK_M", "BLOCK_N", "BLOCK_K", "GROUP_SIZE_M")
PLAN_KEYS = ("STREAMK_POLICY", "STREAMK_FIXUP", "SPLIT_N", "STREAMK_EXCHANGE")


@functools.cache
def streamk_defaults():
    """The tuned defaults, from configs/gfx950/gluon/gemm/gemm_a16w16_streamk/
    DEFAULT.json (read-only, shared):

    DEFAULT_TILE   the tile of a config that names none
    GROUP_SIZE_M   the GROUP_SIZE_M of a config that gives none, and of every
                   tuning candidate
    STREAMK_PLAN   the default plan's constants (fitted on gfx950 over all 13 tile
                   specs, below one wave and after it): m_ow, the most programs per
                   tail tile that take the owner ending; c_ow and b, the owner's and
                   the reduce-scatter's split scales; pairs_ow, the fewest K-pairs per
                   tile worth an owner split; gain, the fewest K-pairs of the tail's
                   time a reduce-scatter split must save
    OWNER_SPLITS,  the SPLIT_N values the default plan rounds to and the tuner
    RS_SPLITS      times, for the owner and reduce-scatter endings
    """
    cfg_dir = resolve_config_dir("gemm", "GEMM-A16W16-STREAMK", backend="gluon")
    return load_config_json(f"{cfg_dir}/DEFAULT.json")


def _nearest_split(n, splits, cap):
    ok = [s for s in splits if s <= cap]
    return min(ok, key=lambda s: (abs(math.log(min(n, cap) / s)), s)) if ok else None


def streamk_plan(M, N, K, tile, num_programs, plan=None):
    """(STREAMK_POLICY, STREAMK_FIXUP, SPLIT_N) of the default plan for a shape at a
    tile (BLOCK_M, BLOCK_N, BLOCK_K); see the module docstring. ``plan`` overrides
    STREAMK_PLAN from streamk_defaults()."""
    defaults = streamk_defaults()
    if plan is None:
        plan = defaults["STREAMK_PLAN"]
    bm, bn, bk = tile
    total_tiles = triton.cdiv(M, bm) * triton.cdiv(N, bn)
    pairs = triton.cdiv(K, 2 * bk)
    leftover = total_tiles % num_programs
    if leftover == 0:
        return "dp", "owner", None
    per_tile = num_programs // leftover
    cap = min(per_tile, pairs)
    if cap < 2:
        return "dp", "owner", None
    if per_tile <= plan["m_ow"]:
        if pairs < plan["pairs_ow"]:
            return "dp", "owner", None
        n = _nearest_split(
            max(2.0, math.sqrt(pairs / plan["c_ow"])), defaults["OWNER_SPLITS"], cap
        )
        return ("one_tile", "owner", n) if n else ("dp", "owner", None)
    n = _nearest_split(
        max(2.0, math.sqrt(pairs * per_tile / plan["b"])), defaults["RS_SPLITS"], cap
    )
    if not n or pairs * (1 - 1 / n) < plan["gain"]:
        return "dp", "owner", None
    return "one_tile", "rs", n


def complete_config(config, M, N, K, num_programs):
    """The complete config dict: one that names only the tile (BLOCK_M, BLOCK_N,
    BLOCK_K, optionally GROUP_SIZE_M) gets GROUP_SIZE_M 4 and the default plan;
    the plan keys are all given (and kept) or all absent."""
    if isinstance(config, StreamKConfig):
        return config
    if not isinstance(config, Mapping):
        raise TypeError("the Stream-K config must be a mapping")
    given = [key for key in PLAN_KEYS if key in config]
    if given and len(given) != len(PLAN_KEYS):
        missing = [key for key in PLAN_KEYS if key not in config]
        raise ValueError(
            f"give all or none of the Stream-K plan keys; missing {missing}"
        )
    full = dict(config)
    full.setdefault("GROUP_SIZE_M", streamk_defaults()["GROUP_SIZE_M"])
    if not given:
        tile = tuple(full.get(key) for key in TILE_KEYS[:3])
        if any(v is None for v in tile):
            raise ValueError(
                f"the config must name the tile ({', '.join(TILE_KEYS[:3])})"
            )
        TileSpec(*tile)
        policy, fixup, split_n = streamk_plan(M, N, K, tile, num_programs)
        full.update(
            STREAMK_POLICY=policy,
            STREAMK_FIXUP=fixup,
            SPLIT_N=split_n,
            STREAMK_EXCHANGE="wt",
        )
    return full


def _launch_constants(M, N, K, config, num_programs, num_xcds):
    """Parse a config (a tile-only one gets the default plan, complete_config) and
    derive the shape- and topology-dependent launch constants; raises ValueError
    when the config cannot run the shape."""
    config = StreamKConfig.from_dict(complete_config(config, M, N, K, num_programs))
    tile = config.tile
    if k_pad(K, tile.block_k) < 4 * tile.block_k:
        raise ValueError(f"K={K} needs at least {4 * tile.block_k} after pair padding")
    total_tiles = triton.cdiv(M, tile.block_m) * triton.cdiv(N, tile.block_n)
    pairs = k_pad(K, tile.block_k) // (2 * tile.block_k)
    leftover = total_tiles % num_programs

    streamk_tiles = 0 if config.policy == "dp" else leftover
    if config.policy == "two_tile":
        if leftover == 0 or total_tiles < num_programs:
            raise ValueError(
                "two_tile requires a partial wave after at least one full wave"
            )
        streamk_tiles += num_programs
    if config.policy in ("one_tile", "spread") and leftover == 0:
        raise ValueError(f"{config.policy} requires leftover tiles")
    if config.policy == "spread" and 2 * leftover <= num_programs:
        raise ValueError(
            "spread requires more than half of the resident programs "
            "as leftover tiles"
        )

    end_to_end = config.policy in ("two_tile", "spread")
    l2 = config.exchange == "l2"
    per_xcd_tiles = 0
    if config.policy == "one_tile":
        split_n = config.split_n
        split_cap = min(num_programs // streamk_tiles, pairs)
        if split_cap < 2:
            raise ValueError(
                "shape leaves fewer than two programs/pairs per Stream-K tile"
            )
        if split_n > split_cap:
            raise ValueError(
                f"split_n={split_n} exceeds this shape's maximum {split_cap}"
            )
        streamk_programs = streamk_tiles * split_n
        if l2:
            per_xcd_tiles = triton.cdiv(streamk_tiles, num_xcds)
            xcd_max = max(
                1,
                min(
                    split_n,
                    pairs,
                    (num_programs // num_xcds) // per_xcd_tiles,
                ),
            )
            if xcd_max != split_n:
                raise ValueError(
                    f"split_n={split_n} cannot be placed within one XCD; "
                    f"maximum is {xcd_max}"
                )
    elif end_to_end:
        split_n = 1
        streamk_programs = num_programs
        if streamk_tiles * pairs < streamk_programs:
            raise ValueError(
                f"{config.policy} has fewer pair units than "
                f"the {num_programs} programs"
            )
    else:
        split_n = 1
        streamk_programs = 0

    if config.policy == "one_tile" and config.fixup == "rs" and not l2:
        rs = rs_readers(tile, streamk_tiles, split_n, num_programs)
    else:
        rs = reduce_scatter_config(tile, split_n if config.policy == "one_tile" else 1)
    # The L2 split gives every XCD ceil(S / NUM_XCDS) tiles' programs (pid p runs
    # on XCD p % NUM_XCDS), so its grid is a whole number of those per XCD. The wt
    # reduce-scatter also launches its S extra idle readers per tile.
    working_grid = (
        num_xcds * per_xcd_tiles * split_n
        if per_xcd_tiles
        else streamk_programs + streamk_tiles * rs.extra
    )
    compact = (
        streamk_tiles > 0
        and streamk_tiles == total_tiles
        and working_grid < num_programs
    )
    # Data-parallel launches have no Stream-K phase to keep resident, so a grid of
    # fewer tiles than programs launches only the programs with a tile.
    if compact:
        grid = working_grid
    elif streamk_tiles == 0:
        grid = min(num_programs, total_tiles)
    else:
        grid = num_programs
    return config, {
        "TOTAL_TILES": total_tiles,
        "STREAMK_TILES": streamk_tiles,
        "STREAMK_PROGRAMS": streamk_programs,
        "RESOLVED_SPLIT_N": split_n,
        "END_TO_END": end_to_end,
        "L2": l2,
        "COMPACT": compact,
        "GRID": grid,
        "RS_NB": rs.blocks_per_group,
        "RS_U": rs.partials_per_trip,
        "RS_R": rs.readers,
        "RS_E": rs.extra,
        "BAR_STRIDE": rs.barrier_stride,
    }


def unsupported_reason(M, N, K, activation=None, config=None):
    """Why this kernel cannot run the shape, or None if it can; tensors (dtype,
    strides, bias) are the launcher's to check. Without a config, whether some tile
    fits; with one (complete, or naming only the tile, which gets the default plan),
    whether it fits the shape."""
    if activation:
        return "no activation support"
    if M <= 0 or N <= 0 or K <= 0:
        return f"M, N and K must be positive (got {M}, {N}, {K})"
    if N % 8:
        return f"N must be a multiple of 8 (16-byte stores; got N={N})"
    if K % 16:
        return f"K must be a multiple of 16 (got K={K})"
    if config is None:
        if all(k_pad(K, bk) < 4 * bk for bk in BLOCK_K_SIDES):
            return f"K={K} needs at least {4 * min(BLOCK_K_SIDES)} after pair padding"
        return None
    try:
        _launch_constants(M, N, K, config, *streamk_topology())
    except (TypeError, ValueError) as error:
        return str(error)
    return None


def with_default_tile(config):
    """``config`` with DEFAULT_TILE (streamk_defaults) filled in when it names no
    tile (None included). DEFAULT_TILE is the same tile whatever the shape: a
    placeholder until there is a shape-aware rule (memory-bound shapes in
    particular)."""
    default_tile = streamk_defaults()["DEFAULT_TILE"]
    if config is None:
        return dict(default_tile)
    if isinstance(config, StreamKConfig) or any(key in config for key in TILE_KEYS[:3]):
        return config
    return {**default_tile, **config}


def fit_tuned_config(config, M, N, K):
    """``config`` (a complete tuned config) if its plan fits the shape, else its
    tile alone, which gets the default plan (a row reached through the padded-M
    lookup may have been tuned for another tile count)."""
    if unsupported_reason(M, N, K, config=config) is None:
        return config
    return {key: config[key] for key in TILE_KEYS if key in config}


# ---------------------------------------------------------------------------------
# Tuning: the candidate configs of a shape and their tuned-row kernelName
# ---------------------------------------------------------------------------------

# Every (BLOCK_M, BLOCK_N, BLOCK_K) whose double-buffered A and B tiles fit in LDS:
# BLOCK_K 128 only up to 128x128.
TUNING_TILES = (
    (256, 256, 64),
    (128, 256, 64),
    (256, 128, 64),
    (128, 128, 128),
    (128, 128, 64),
    (64, 256, 64),
    (256, 64, 64),
    (64, 128, 128),
    (128, 64, 128),
    (64, 128, 64),
    (128, 64, 64),
    (64, 64, 128),
    (64, 64, 64),
)


def _plans():
    """Every (STREAMK_POLICY, STREAMK_FIXUP, SPLIT_N, STREAMK_EXCHANGE) worth timing;
    which of them fit a shape is left to _launch_constants."""
    yield "dp", "owner", None, "wt"
    yield "two_tile", "owner", None, "wt"
    yield "spread", "owner", None, "wt"
    defaults = streamk_defaults()
    for n in defaults["OWNER_SPLITS"]:
        yield "one_tile", "owner", n, "wt"
    for n in defaults["RS_SPLITS"]:
        yield "one_tile", "rs", n, "wt"
        yield "one_tile", "rs", n, "l2"


def tuning_candidates(M, N, K):
    """The complete configs the tuner times for a shape: every TUNING_TILES tile at
    the default GROUP_SIZE_M (streamk_defaults) with every plan that fits it (and the default
    plan's pick), one per distinct launch. M, N and K may be numpy integers (a
    tuner's pandas rows)."""
    M, N, K = int(M), int(N), int(K)
    num_programs, num_xcds = streamk_topology()
    candidates, seen = [], set()
    for bm, bn, bk in TUNING_TILES:
        tile = {
            "BLOCK_M": bm,
            "BLOCK_N": bn,
            "BLOCK_K": bk,
            "GROUP_SIZE_M": streamk_defaults()["GROUP_SIZE_M"],
        }
        configs = [complete_config(tile, M, N, K, num_programs)]
        configs += [dict(tile, **dict(zip(PLAN_KEYS, plan))) for plan in _plans()]
        for config in configs:
            if unsupported_reason(M, N, K, config=config) is not None:
                continue
            parsed, launch = _launch_constants(M, N, K, config, num_programs, num_xcds)
            key = (
                parsed.tile,
                parsed.group_size_m,
                parsed.fixup,
                tuple(launch.items()),
            )
            if key not in seen:
                seen.add(key)
                candidates.append(config)
    return candidates


KERNEL_NAME_PREFIX = "gluon_streamk_"
_KERNEL_NAME = re.compile(
    r"^gluon_streamk_(?P<bm>\d+)x(?P<bn>\d+)x(?P<bk>\d+)_g(?P<g>\d+)_"
    r"(?:(?P<whole>dp|two_tile|spread)"
    r"|one_tile_(?P<fixup>owner|rs)_n(?P<n>\d+)_(?P<exchange>wt|l2))$"
)


def kernel_name(config):
    """The kernelName of a tuned row running a complete ``config``, e.g.
    gluon_streamk_256x256x64_g4_one_tile_rs_n8_wt or gluon_streamk_256x256x64_g4_dp
    (policies other than one_tile take the owner fixup and the wt exchange)."""
    c = StreamKConfig.from_dict(config)
    name = f"{KERNEL_NAME_PREFIX}{c.block_m}x{c.block_n}x{c.block_k}_g{c.group_size_m}_"
    if c.policy != "one_tile":
        return name + c.policy
    return name + f"one_tile_{c.fixup}_n{c.split_n}_{c.exchange}"


def parse_kernel_name(name):
    """The complete config a kernel_name() names."""
    m = _KERNEL_NAME.match(str(name))
    if m is None:
        raise ValueError(f"unrecognized Stream-K kernelName {name!r}")
    config = {
        "BLOCK_M": int(m["bm"]),
        "BLOCK_N": int(m["bn"]),
        "BLOCK_K": int(m["bk"]),
        "GROUP_SIZE_M": int(m["g"]),
    }
    if m["whole"]:
        plan = (m["whole"], "owner", None, "wt")
    else:
        plan = ("one_tile", m["fixup"], int(m["n"]), m["exchange"])
    config.update(zip(PLAN_KEYS, plan))
    return config


_KERNEL_MAP = {"streamk": _gemm_a16w16_streamk_kernel}
