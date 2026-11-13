# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Gluon BF16/FP16 GEMM for gfx950: persistent data-parallel tiles and a Stream-K or
sub-tiled tail.

``y = x @ w.T (+ bias)`` with x (M, K) and w (N, K) both K-contiguous, y (M, N) row-major.

The kernel. One program per CU with four waves, one per SIMD. A BLOCK_M x BLOCK_N
output tile (sides any multiple of 32 from 64 to 320 with at most 256x256
accumulators; BLOCK_K 64, or 128 when BLOCK_M + BLOCK_N <= 256) is split into four
quadrant accumulators pinned to AGPRs (``cd_regclass="a"``), fed by a 2-deep
async-copy LDS pipeline with local prefetch and the K loop unrolled by 2; every
layout is derived from the tile. A side that is not a power of two has uneven halves
(TOP_M / LEFT_N, from the tutorial's v9_uneven2): each half is a power-of-two block
plus an optional extra block of 32 or 64 (224 = 128 + (64 + 32)), so a quadrant is a
tuple of blocks; power-of-two tiles build the same code as before. Tiles are
assigned through an XCD-aware remap and a GROUP_SIZE_M
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
     "STREAMK_POLICY": "one_tile", "STREAMK_FIXUP": "rs", "TAIL_SPLIT_K": 8,
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
    "dp_sub"    after at least one full wave, each tail tile cut into Q sub-tiles,
                each whole on one program, no partials (subtile_tail picks them;
                T = 10, S = 2, 128x256 tiles, 128x128 sub-tiles, Q = 2):
                    wave 0  as dp
                    tail    t8: [t8.0][t8.1]   t9: [t9.0][t9.1]      4 programs
    "one_tile"  each tail tile's pairs cut evenly over TAIL_SPLIT_K programs
                (T = 10, S = 2, TAIL_SPLIT_K = 4, S * TAIL_SPLIT_K = 8 programs):
                    wave 0  as dp
                    tail    t8: [AA][AA][AA][AA]   t9: [BB][BB][BB][BB]
    "spread"    S > P / 2 only: the tail's S * k pairs laid end to end and cut
                evenly over all P programs (S = 5):
                    [AAAAA][AAA|BB][BBBBB][B|CCCC][CCCC|D][DDDDD][DD|EEE][EEEEE]
    "two_tile"  as spread, over the last P + S tiles (needs a full wave to borrow).

    spread and two_tile run each program's pieces last to first, so an XCD's
    programs all start at pair 0 and share A and B in L2, and each publishes its
    partial before it waits for the tile it owns.

STREAMK_FIXUP, how one_tile's TAIL_SPLIT_K partial sums are combined (spread and
two_tile always use the owner).

    "owner"  the program with a tile's first pairs owns it; the others store their
             partials and raise a flag; the owner adds them one after another:
                 p1, p2, p3 --partial--> p0 --> C tile
             Fewest bytes, but n - 1 serial reads: suits small TAIL_SPLIT_K.
    "rs"     reduce-scatter: every program stores its partial, all meet at the
             tile's barrier, then program i sums slice i of every partial:
                 p0 p1 p2 p3 store | barrier | C tile: [ p0 | p1 | p2 | p3 ]
             Programs the tail leaves idle read slices too: R = max(min(blocks,
             pow2_floor(P / S)), own readers) per tile, with each tile's barrier
             in its own cache line. The ending costs about the same at any
             TAIL_SPLIT_K: suits large TAIL_SPLIT_K.

STREAMK_EXCHANGE, where partials and flags travel.

    "wt"  written through to memory; no placement constraint.
    "l2"  one_tile with rs only: all of a tile's programs on one XCD, partials and
          flags kept in its L2. Needs
          TAIL_SPLIT_K <= (P / NUM_XCDS) / ceil(S / NUM_XCDS).

TAIL_SPLIT_K, programs per tail tile, each running an even share of its K-pairs: an
integer 2 <= TAIL_SPLIT_K <= min(P // S, k) for one_tile, None for the other policies.

STREAMK_PARTIALS (optional, default "fp32"): "16bit" stores the partials in C's type
(bf16 or fp16), half the bytes; they are still summed in fp32. The extra rounding
raises the RMS error 1.2-1.5 times over fp32 partials (the largest error up to about
2 times). Only the policies that store partials (one_tile, two_tile, spread) take it;
the default plan stays fp32.

dp_sub's sub-tile (TAIL_BLOCK_M x TAIL_BLOCK_N, sides from the tile menu dividing the
tile's) is derived from the shape at launch, so a tuned dp_sub follows M: the largest
Q = (BLOCK_M / TAIL_BLOCK_M) (BLOCK_N / TAIL_BLOCK_N) with S Q <= P, then the most
square sub-tile (subtile_tail). The full waves keep BLOCK_M x BLOCK_N; the tail runs
its own sub-tile pipeline in the same LDS. It trades the tile's MFMA efficiency for
no fixup: suits a tail of at most half a wave and short K.

GROUP_SIZE_M, rows of tiles walked together by the persistent loop's swizzle.

Default plan. A config may name only the tile (BLOCK_M, BLOCK_N, BLOCK_K, and
optionally GROUP_SIZE_M, default 4); streamk_plan then supplies STREAMK_POLICY,
STREAMK_FIXUP and TAIL_SPLIT_K, with the "wt" exchange. With m = P // S programs per
tail tile, pairs = ceil(K / (2 BLOCK_K)) K-pairs per tile and cap = min(m, pairs):

    dp              if S == 0 or cap < 2
    dp_sub          if T > P, S <= P / 2 and K <= SUBTILE_MAX_K (4096)
    owner           if m <= 3:  n = sqrt(pairs / 1.5),    dp if pairs < 10
    reduce-scatter  otherwise:  n = sqrt(pairs * m / 6),  dp if pairs * (1 - 1/n) < 6

n is clamped to cap and rounded to the nearest of OWNER_SPLITS / RS_SPLITS. The
reduce-scatter's n balances each program's MAC (pairs / n) against the partials all
S n programs store and read at once; the owner reads its n - 1 partials one after
another, so its n grows only with sqrt(pairs). The same constants (STREAMK_PLAN)
hold for every tile spec, below one wave and after it. They, SUBTILE_MAX_K,
OWNER_SPLITS, RS_SPLITS, DEFAULT_TILE and the default GROUP_SIZE_M live in
configs/gfx950/gluon/gemm/gemm_a16w16_streamk/DEFAULT.json (streamk_defaults).

Tile order. The persistent tiles are the first T - S tile ids, with the XCD remap
and GROUP_SIZE_M swizzle applied over those ids alone, and the tail tiles are the
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


# ---------------------------------------------------------------------------------
# Tile geometry: uneven halves
#
# A tile side splits into a first half (top / left) of FIRST rows (0: side / 2) and a
# second half of the rest. Each half is a power-of-two main block plus an optional
# extra block of 32 or 64 (224 = 128 + (64 + 32), 160 = (64 + 32) + 64), all on the
# same MFMA layout. A power-of-two tile has equal halves and no extras. A half is
# carried as a tuple of blocks (main[, extra]): its LDS buffers, load offsets and
# register operands; a quadrant's accumulators are a tuple in quad_mma's order.
# Each extra is copied in the same commit group as its main block, so the pipeline's
# wait counts are the power-of-two tile's.
# ---------------------------------------------------------------------------------

# The MFMA layout's warps (rows, columns). The partials' 32x32 block (MFMA_BLOCK_DIM)
# is the 2 x 2 warps' 16x16 outputs, so every partial map assumes [2, 2].
MFMA_WARPS_MN = gl.constexpr([2, 2])
assert MFMA_WARPS_MN.value == [2, 2], "the partial maps assume 2 x 2 MFMA warps"

MFMA_LAYOUT = gl.amd.AMDMFMALayout(
    version=4,
    instr_shape=[16, 16, 32],
    transposed=True,
    warps_per_cta=MFMA_WARPS_MN.value,
)


@constexpr_function
def half_size(side, first):
    """Rows (columns) of a side's first half."""
    return first if first > 0 else side // 2


@constexpr_function
def half_blocks(side, first, second):
    """Rows (columns) of the blocks of a side's first half, or its second: the
    power-of-two main block, then the extra if any."""
    top = half_size(side, first)
    n = side - top if second else top
    main = 1 << (n.bit_length() - 1)
    return (main,) if n == main else (main, n - main)


@constexpr_function
def block_starts(blocks, base):
    """First row (column) of each block of a half that starts at base."""
    out, start = [], base
    for b in blocks:
        out.append(start)
        start += b
    return tuple(out)


@constexpr_function
def is_uneven(block_m, top_m, block_n, left_n):
    """True unless both sides have two equal power-of-two halves."""
    return (
        len(half_blocks(block_m, top_m, False)) > 1
        or half_blocks(block_m, top_m, False) != half_blocks(block_m, top_m, True)
        or len(half_blocks(block_n, left_n, False)) > 1
        or half_blocks(block_n, left_n, False) != half_blocks(block_n, left_n, True)
    )


@constexpr_function
def operand_shared_layout(operand, rows, block_k, dtype):
    """The padded shared layout of one operand block: A's [rows, BLOCK_K] (operand 0)
    or B's [BLOCK_K, rows] (operand 1)."""
    dot = gl.DotOperandLayout(operand_index=operand, parent=MFMA_LAYOUT, k_width=8)
    shape = [rows, block_k] if operand == 0 else [block_k, rows]
    return ttgl_cdna4.compute_efficient_padded_shared_layout(dot, shape, dtype)


@constexpr_function
def operand_load_layout(operand, rows, block_k, dtype, num_warps):
    """The global-load layout of one operand block (compute_gload_layout)."""
    return compute_gload_layout(
        operand_shared_layout(operand, rows, block_k, dtype), 8, num_warps
    )


@constexpr_function
def tile_blocks(block_m, top_m, block_n, left_n, field):
    """One field of every accumulator block of a tile, in slot order (the order the
    partials are stored in): quadrants tl, bl, tr, br, each in quad_mma's order, with
    the one-block (32x32) blocks moved to the end so that 16-bit partials can pair them. Fields:
    "q" quadrant, "k" index in the quadrant's tuple, "r0" / "c0" first row / column
    in the tile, "rows" / "cols", "off" first element in the slot, "vq" 32x32 blocks."""
    halves_m = (
        (half_blocks(block_m, top_m, False), 0),
        (half_blocks(block_m, top_m, True), half_size(block_m, top_m)),
    )
    halves_n = (
        (half_blocks(block_n, left_n, False), 0),
        (half_blocks(block_n, left_n, True), half_size(block_n, left_n)),
    )
    blocks = []
    for q in range(4):
        rows, r_base = halves_m[q % 2]
        cols, c_base = halves_n[q // 2]
        r0s, c0s = block_starts(rows, r_base), block_starts(cols, c_base)
        for j in range(len(cols)):
            for i in range(len(rows)):
                blocks.append(
                    {
                        "q": q,
                        "k": i + len(rows) * j,
                        "r0": r0s[i],
                        "c0": c0s[j],
                        "rows": rows[i],
                        "cols": cols[j],
                        "vq": rows[i] * cols[j] // 1024,
                    }
                )
    blocks = [b for b in blocks if b["vq"] > 1] + [b for b in blocks if b["vq"] == 1]
    off = 0
    for b in blocks:
        b["off"] = off
        off += b["rows"] * b["cols"]
    return tuple(b[field] for b in blocks)


@gluon.jit
def quad_mma(a, b, acc):
    """A quadrant's MFMAs. a, b: its A and B halves' blocks (main[, extra]); block
    (i, j) accumulates in acc[i + len(a) * j]."""
    out = ()
    for j in gl.static_range(len(b)):
        for i in gl.static_range(len(a)):
            out = out + (
                gl.amd.cdna3.mfma(a[i], b[j], acc[i + len(a) * j], cd_regclass="a"),
            )
    return out


@gluon.jit
def read_half(smem, idx, layout: gl.constexpr):
    """A half's blocks from LDS buffer idx, main then extra."""
    out = ()
    for j in gl.static_range(len(smem)):
        out = out + (smem[j].index(idx).load(layout),)
    return out


@gluon.jit
def copy_half(smem, idx, base, offs, masks=None):
    """A half's async copies into LDS buffer idx; masks: one per block, or None."""
    for j in gl.static_range(len(smem)):
        if masks is None:
            gl.amd.cdna4.async_copy.buffer_load_to_shared(
                smem[j].index(idx), base, offs[j]
            )
        else:
            gl.amd.cdna4.async_copy.buffer_load_to_shared(
                smem[j].index(idx), base, offs[j], mask=masks[j]
            )


@gluon.jit
def operand_smem(
    a_dtype: gl.constexpr,
    b_dtype: gl.constexpr,
    BLOCK_M: gl.constexpr,
    TOP_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    LEFT_N: gl.constexpr,
    BLOCK_K: gl.constexpr,
    NBUF: gl.constexpr,
):
    """The tile's LDS buffers, (A top, A bottom, B left, B right), each a tuple of
    blocks."""
    return (
        half_smem(a_dtype, half_blocks(BLOCK_M, TOP_M, False), BLOCK_K, NBUF, 0),
        half_smem(a_dtype, half_blocks(BLOCK_M, TOP_M, True), BLOCK_K, NBUF, 0),
        half_smem(b_dtype, half_blocks(BLOCK_N, LEFT_N, False), BLOCK_K, NBUF, 1),
        half_smem(b_dtype, half_blocks(BLOCK_N, LEFT_N, True), BLOCK_K, NBUF, 1),
    )


@gluon.jit
def half_smem(
    dtype: gl.constexpr,
    BLOCKS: gl.constexpr,
    BLOCK_K: gl.constexpr,
    NBUF: gl.constexpr,
    OPERAND: gl.constexpr,
):
    """A half's LDS buffers, one per block: A's [NBUF, rows, BLOCK_K] or B's
    [NBUF, BLOCK_K, cols]."""
    out = ()
    for i in gl.static_range(len(BLOCKS)):
        if OPERAND == 0:
            out = out + (
                gl.allocate_shared_memory(
                    dtype,
                    [NBUF, BLOCKS[i], BLOCK_K],
                    operand_shared_layout(0, BLOCKS[i], BLOCK_K, dtype),
                ),
            )
        else:
            out = out + (
                gl.allocate_shared_memory(
                    dtype,
                    [NBUF, BLOCK_K, BLOCKS[i]],
                    operand_shared_layout(1, BLOCKS[i], BLOCK_K, dtype),
                ),
            )
    return out


@gluon.jit
def half_offsets(
    ROWS: gl.constexpr,
    R0S: gl.constexpr,
    stride_r,
    stride_k,
    lim,
    BLOCK_K: gl.constexpr,
    dtype: gl.constexpr,
    OPERAND: gl.constexpr,
    CLAMP: gl.constexpr,
):
    """Load offsets of a half's blocks: A's [rows, BLOCK_K] (OPERAND 0) or B's
    [BLOCK_K, cols] (OPERAND 1), the block's rows (columns) R0S[i] .. + ROWS[i]
    relative to the half's base. CLAMP: rows past lim (the tile's last row inside the
    matrix, relative to the same base) are clamped to it (clamped_a_offsets)."""
    out = ()
    for i in gl.static_range(len(ROWS)):
        out = out + (
            block_offsets(
                ROWS[i], R0S[i], stride_r, stride_k, lim, BLOCK_K, dtype, OPERAND, CLAMP
            ),
        )
    return out


@gluon.jit
def block_offsets(
    ROWS: gl.constexpr,
    R0: gl.constexpr,
    stride_r,
    stride_k,
    lim,
    BLOCK_K: gl.constexpr,
    dtype: gl.constexpr,
    OPERAND: gl.constexpr,
    CLAMP: gl.constexpr,
):
    """half_offsets of one block."""
    layout: gl.constexpr = operand_load_layout(
        OPERAND, ROWS, BLOCK_K, dtype, gl.num_warps()
    )
    if OPERAND == 0:
        r = gl.arange(0, ROWS, gl.SliceLayout(1, layout))
        k = gl.arange(0, BLOCK_K, gl.SliceLayout(0, layout))
    else:
        r = gl.arange(0, ROWS, gl.SliceLayout(0, layout))
        k = gl.arange(0, BLOCK_K, gl.SliceLayout(1, layout))
    if R0 > 0:
        r = r + R0
    if CLAMP:
        r = gl.minimum(r, lim)
    rt = gl.multiple_of(r * stride_r, 8)
    if OPERAND == 0:
        offs = rt[:, None] + k[None, :] * stride_k
    else:
        offs = k[:, None] * stride_k + rt[None, :]
    return offs


@gluon.jit
def operand_offsets(
    BLOCK_M: gl.constexpr,
    TOP_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    LEFT_N: gl.constexpr,
    BLOCK_K: gl.constexpr,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    a_dtype: gl.constexpr,
    b_dtype: gl.constexpr,
):
    """kofs of an uneven tile on an even M and N: (A top offsets, A bottom offsets,
    A bottom shift, B left offsets, B right offsets, B right shift). The second half
    loads from base + shift + its offsets, the shift (the first half's rows) in the
    scalar base; equal halves share one offset tuple, as a power-of-two tile does."""
    TOPS: gl.constexpr = half_blocks(BLOCK_M, TOP_M, False)
    BOTS: gl.constexpr = half_blocks(BLOCK_M, TOP_M, True)
    LEFTS: gl.constexpr = half_blocks(BLOCK_N, LEFT_N, False)
    RIGHTS: gl.constexpr = half_blocks(BLOCK_N, LEFT_N, True)
    a_top = half_offsets(
        TOPS, block_starts(TOPS, 0), stride_am, stride_ak, 0, BLOCK_K, a_dtype, 0, False
    )
    a_bot = a_top
    if BOTS != TOPS:
        a_bot = half_offsets(
            BOTS,
            block_starts(BOTS, 0),
            stride_am,
            stride_ak,
            0,
            BLOCK_K,
            a_dtype,
            0,
            False,
        )
    b_left = half_offsets(
        LEFTS,
        block_starts(LEFTS, 0),
        stride_bn,
        stride_bk,
        0,
        BLOCK_K,
        b_dtype,
        1,
        False,
    )
    b_right = b_left
    if RIGHTS != LEFTS:
        b_right = half_offsets(
            RIGHTS,
            block_starts(RIGHTS, 0),
            stride_bn,
            stride_bk,
            0,
            BLOCK_K,
            b_dtype,
            1,
            False,
        )
    return (
        a_top,
        a_bot,
        half_size(BLOCK_M, TOP_M) * stride_am,
        b_left,
        b_right,
        half_size(BLOCK_N, LEFT_N) * stride_bn,
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


@gluon.jit
def tail_unit_coords(
    v,
    M,
    N,
    BM: gl.constexpr,
    BN: gl.constexpr,
    TBM: gl.constexpr,
    TBN: gl.constexpr,
    GROUP_SIZE_M: gl.constexpr,
):
    """(m, n) in the TBM x TBN grid of unit v, sub-tile v % Q of tile v // Q, whose
    Q = (BM / TBM) (BN / TBN) sub-tiles are numbered row-major. Tail unit u is
    v = total_full_tiles Q + u; with TBM, TBN = BM, BN a unit is the whole tile.

    BM x BN = 256x256, TBM x TBN = 128x64 (Q = 8), v = 8 tile + q:

        +----+----+----+----+
        | q0 | q1 | q2 | q3 |      m = 2 pid_m + q // 4
        +----+----+----+----+      n = 4 pid_n + q % 4
        | q4 | q5 | q6 | q7 |
        +----+----+----+----+
    """
    if TBM == BM and TBN == BN:
        m, n = group_m_tile_coords(v, M, N, BM, BN, GROUP_SIZE_M)
    else:
        QM: gl.constexpr = BM // TBM
        QN: gl.constexpr = BN // TBN
        q = v % (QM * QN)
        pid_m, pid_n = group_m_tile_coords(v // (QM * QN), M, N, BM, BN, GROUP_SIZE_M)
        m = pid_m * QM + q // QN
        n = pid_n * QN + q % QN
    return m, n


_gemm_a16w16_streamk_repr = make_kernel_repr(
    "_gemm_a16w16_streamk_kernel",
    [
        "BLOCK_M",
        "BLOCK_N",
        "BLOCK_K",
        "GROUP_SIZE_M",
        "TAIL_TILES",
        "TAIL_SPLIT_K",
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
    TAIL_TILES: gl.constexpr,
    TAIL_SPLIT_K: gl.constexpr,
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
    TAIL_BLOCK_M: gl.constexpr = 0,
    TAIL_BLOCK_N: gl.constexpr = 0,
    TAIL_EVEN_M: gl.constexpr = True,
    TAIL_EVEN_N: gl.constexpr = True,
    TOP_M: gl.constexpr = 0,
    LEFT_N: gl.constexpr = 0,
    TAIL_TOP_M: gl.constexpr = 0,
    TAIL_LEFT_N: gl.constexpr = 0,
):
    """
    Persistent full tiles followed by a uniform or end-to-end tail, on a
    tile sliced in both M and N (4 quadrant accumulators, K loop unrolled by 2,
    pre-computed load offsets, XCD-aware PID remapping + GROUP_SIZE_M swizzling):

    1. A persistent loop over the first total_full_tiles = total_tiles - TAIL_TILES
       tile ids, XCD-remapped over those ids alone: each tile's epilogue prefetches
       the first K-step pair of the program's next tile. The last trip's epilogue
       prefetches the first pair of the program's first Stream-K segment, which then
       skips its prologue (on a ragged K masked off when the program has no segment;
       on an even K the idle program's target is in bounds and never read). Without a
       Stream-K tail it loads the last pair of tile total_full_tiles - 1, which is
       discarded.
    2. A tail over the last TAIL_TILES tile ids. Stream-K splits each tile in units
       of a K-step pair, whose partials are combined by an owner or a reduce-scatter
       ending; every program of the Stream-K phase must be resident at once, as the
       endings wait on each other. dp_sub instead runs each tile's sub-tiles whole,
       one per program (TAIL_BLOCK_M / TAIL_BLOCK_N).

    TAIL_SPLIT_K: the uniform tail's programs per tile, each running an even share of
    its K-pairs (1 on the end-to-end tail, whose split is not per tile).

    TAIL_BLOCK_M / TAIL_BLOCK_N (0: BLOCK_M / BLOCK_N): dp_sub's sub-tile. The
    uniform tail cuts each tail tile into Q = (BLOCK_M / TAIL_BLOCK_M)
    (BLOCK_N / TAIL_BLOCK_N) sub-tiles (tail_unit_coords), and one program runs each
    sub-tile over its whole K, with no partials. TAIL_EVEN_M / TAIL_EVEN_N are
    EVEN_M / EVEN_N at the sub-tile's size.

    P is fp32, or typed like C for 16-bit partials (is_16bit_partial), stored in pairs
    of blocks (pair_layout); their sums stay fp32.

    TOP_M / LEFT_N (0: BLOCK_M / 2, BLOCK_N / 2): the rows of the tile's top half and
    the columns of its left half. A half that is not a power of two is a power-of-two
    block plus an extra block of 32 or 64 (uneven halves; see "Tile geometry").
    TAIL_TOP_M / TAIL_LEFT_N are the dp_sub sub-tile's.
    """
    gl.static_assert(
        p_ptr.dtype.element_ty == gl.float32
        or p_ptr.dtype.element_ty == c_ptr.dtype.element_ty
    )
    start = gl.program_id(axis=0)
    num_tiles_m = gl.cdiv(M, BLOCK_M)
    num_tiles_n = gl.cdiv(N, BLOCK_N)
    total_tiles = num_tiles_m * num_tiles_n
    total_full_tiles = total_tiles - TAIL_TILES
    strides = (stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn)

    # Every layout follows the tile: the shared layouts come from the dot-operand
    # layouts and the global-load layouts from the shared layouts.
    mfmaLayout: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4,
        instr_shape=[16, 16, 32],
        transposed=True,
        warps_per_cta=MFMA_WARPS_MN,
    )
    dotOpLayoutA: gl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=mfmaLayout, k_width=8
    )
    dotOpLayoutB: gl.constexpr = gl.DotOperandLayout(
        operand_index=1, parent=mfmaLayout, k_width=8
    )
    UNEVEN: gl.constexpr = is_uneven(BLOCK_M, TOP_M, BLOCK_N, LEFT_N)
    a_dtype: gl.constexpr = a_ptr.dtype.element_ty
    b_dtype: gl.constexpr = b_ptr.dtype.element_ty
    nBuffers: gl.constexpr = 2
    if UNEVEN:
        smem = operand_smem(
            a_dtype, b_dtype, BLOCK_M, TOP_M, BLOCK_N, LEFT_N, BLOCK_K, nBuffers
        )
        kofs6 = operand_offsets(
            BLOCK_M,
            TOP_M,
            BLOCK_N,
            LEFT_N,
            BLOCK_K,
            stride_am,
            stride_ak,
            stride_bk,
            stride_bn,
            a_dtype,
            b_dtype,
        )
        kofs = ((kofs6[0], kofs6[1], kofs6[2]), (kofs6[3], kofs6[4], kofs6[5]))
    else:
        sharedLayoutA: gl.constexpr = ttgl_cdna4.compute_efficient_padded_shared_layout(
            dotOpLayoutA, [BLOCK_M // 2, BLOCK_K], a_ptr.dtype.element_ty
        )
        sharedLayoutB: gl.constexpr = ttgl_cdna4.compute_efficient_padded_shared_layout(
            dotOpLayoutB, [BLOCK_K, BLOCK_N // 2], b_ptr.dtype.element_ty
        )
        gLoadLayoutA: gl.constexpr = compute_gload_layout(
            sharedLayoutA, 8, gl.num_warps()
        )
        gLoadLayoutB: gl.constexpr = compute_gload_layout(
            sharedLayoutB, 8, gl.num_warps()
        )

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
        smem = ((smemA_top,), (smemA_bot,), (smemB_left,), (smemB_right,))

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
        kofs = (
            ((a_offsets,), (a_offsets,), a_bot_shift),
            ((b_offsets,), (b_offsets,), b_right_shift),
        )

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
    TBM: gl.constexpr = BLOCK_M if TAIL_BLOCK_M == 0 else TAIL_BLOCK_M
    TBN: gl.constexpr = BLOCK_N if TAIL_BLOCK_N == 0 else TAIL_BLOCK_N
    SUBTILE: gl.constexpr = TBM != BLOCK_M or TBN != BLOCK_N
    gl.static_assert(BLOCK_M % TBM == 0 and BLOCK_N % TBN == 0)
    gl.static_assert(not SUBTILE or not (END_TO_END or L2 or REDUCE_SCATTER))
    # The last persistent trip's prefetch fills the full tile's buffers and layouts.
    # Prefetching the sub-tile instead measured +0.4-0.65% (1-2% at short K) for
    # about 80% more instructions and 15% more compile time: left as a future option.
    SK_PREFETCH: gl.constexpr = HAS_FULL and TAIL_TILES > 0 and not SUBTILE

    # Persistent phase. Pair zero is already in flight on entry to each tile;
    # its epilogue prefetches pair zero of the next virtual tile.
    if HAS_FULL:
        # The last trip's prefetch target: the first pair of the program's first
        # Stream-K segment, as quadrant_segment's prologue loads it.
        sk_target = None
        if SK_PREFETCH:
            sk_has, sk_tile, sk_k0 = streamk_first_segment(
                total_full_tiles,
                TAIL_TILES,
                PAIRS_PER_TILE,
                TAIL_SPLIT_K,
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
            kofs[0],
            M,
            stride_am,
            stride_ak,
            BLOCK_M,
            BLOCK_K,
            EVEN_M,
            a_dtype,
            TOP_M,
        )
        b_addr = b_load_addressing(
            first_n,
            kofs[1],
            N,
            stride_bk,
            stride_bn,
            BLOCK_N,
            BLOCK_K,
            EVEN_N,
            b_dtype,
            LEFT_N,
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
                a_dtype,
                b_dtype,
                True,
                TOP_M,
                LEFT_N,
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
                EVEN_M,
                EVEN_N,
                EVEN_K,
                K_RES,
                total_full_tiles,
                sk_target,
                SK_PREFETCH,
                TOP_M,
                LEFT_N,
            )

    # The Stream-K phase picks up the last trip's prefetch.
    if not SK_PREFETCH:
        gl.amd.cdna4.async_copy.wait_group(0)
    if not HAS_FULL:
        a_top = 0
        b_left = 0

    if END_TO_END and TAIL_TILES > 0:
        # Two-tile and spread always use the full resident grid.
        rank, sk_pair_start, seg_end = e2e_pair_range(
            TAIL_TILES, PAIRS_PER_TILE, NUM_PROGRAMS, NUM_XCDS
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
                TAIL_TILES,
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
                TOP_M,
                LEFT_N,
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
                TAIL_TILES,
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
                TOP_M,
                LEFT_N,
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
    elif TAIL_TILES > 0:
        # Uniform tile-aligned split: every unit (a tile, or each of its Q sub-tiles)
        # has exactly TAIL_SPLIT_K programs.
        Q: gl.constexpr = (BLOCK_M // TBM) * (BLOCK_N // TBN)
        UNITS: gl.constexpr = TAIL_TILES * Q
        WORKING: gl.constexpr = UNITS * TAIL_SPLIT_K
        NBLK: gl.constexpr = BLOCK_M * BLOCK_N // MFMA_BLOCK_ELEMS
        gl.static_assert(TAIL_SPLIT_K <= PAIRS_PER_TILE)
        # A ragged edge can leave sub-tiles wholly outside C: their programs skip
        # them, and no other unit waits on them.
        SKIP_OUT: gl.constexpr = SUBTILE and not (EVEN_M and EVEN_N)
        if SUBTILE:
            # The tail's own pipeline at TBM x TBN. Its buffers reuse the persistent
            # phase's LDS, so every wave's last reads and copies there must be done.
            if HAS_FULL:
                drain_and_barrier()
        TAIL_UNEVEN: gl.constexpr = SUBTILE and is_uneven(
            TBM, TAIL_TOP_M, TBN, TAIL_LEFT_N
        )
        if TAIL_UNEVEN:
            tail_smem = operand_smem(
                a_dtype, b_dtype, TBM, TAIL_TOP_M, TBN, TAIL_LEFT_N, BLOCK_K, nBuffers
            )
            tail_kofs6 = operand_offsets(
                TBM,
                TAIL_TOP_M,
                TBN,
                TAIL_LEFT_N,
                BLOCK_K,
                stride_am,
                stride_ak,
                stride_bk,
                stride_bn,
                a_dtype,
                b_dtype,
            )
            tail_kofs = (
                (tail_kofs6[0], tail_kofs6[1], tail_kofs6[2]),
                (tail_kofs6[3], tail_kofs6[4], tail_kofs6[5]),
            )
            UNIT_EVEN_M: gl.constexpr = TAIL_EVEN_M
            UNIT_EVEN_N: gl.constexpr = TAIL_EVEN_N
        elif SUBTILE:
            tailSharedA: gl.constexpr = (
                ttgl_cdna4.compute_efficient_padded_shared_layout(
                    dotOpLayoutA, [TBM // 2, BLOCK_K], a_ptr.dtype.element_ty
                )
            )
            tailSharedB: gl.constexpr = (
                ttgl_cdna4.compute_efficient_padded_shared_layout(
                    dotOpLayoutB, [BLOCK_K, TBN // 2], b_ptr.dtype.element_ty
                )
            )
            tailLoadA: gl.constexpr = compute_gload_layout(
                tailSharedA, 8, gl.num_warps()
            )
            tailLoadB: gl.constexpr = compute_gload_layout(
                tailSharedB, 8, gl.num_warps()
            )
            tail_smem = (
                (
                    gl.allocate_shared_memory(
                        a_ptr.dtype.element_ty,
                        [nBuffers, TBM // 2, BLOCK_K],
                        tailSharedA,
                    ),
                ),
                (
                    gl.allocate_shared_memory(
                        a_ptr.dtype.element_ty,
                        [nBuffers, TBM // 2, BLOCK_K],
                        tailSharedA,
                    ),
                ),
                (
                    gl.allocate_shared_memory(
                        b_ptr.dtype.element_ty,
                        [nBuffers, BLOCK_K, TBN // 2],
                        tailSharedB,
                    ),
                ),
                (
                    gl.allocate_shared_memory(
                        b_ptr.dtype.element_ty,
                        [nBuffers, BLOCK_K, TBN // 2],
                        tailSharedB,
                    ),
                ),
            )
            tail_am = gl.arange(0, TBM // 2, gl.SliceLayout(1, tailLoadA))
            tail_ak = gl.arange(0, BLOCK_K, gl.SliceLayout(0, tailLoadA))
            tail_bn = gl.arange(0, TBN // 2, gl.SliceLayout(0, tailLoadB))
            tail_bk = gl.arange(0, BLOCK_K, gl.SliceLayout(1, tailLoadB))
            tail_a = (
                gl.multiple_of(tail_am * stride_am, 8)[:, None]
                + tail_ak[None, :] * stride_ak
            )
            tail_a_shift = TBM // 2 * stride_am
            tail_b = (
                tail_bk[:, None] * stride_bk
                + gl.multiple_of(tail_bn * stride_bn, 8)[None, :]
            )
            tail_kofs = (
                ((tail_a,), (tail_a,), tail_a_shift),
                ((tail_b,), (tail_b,), TBN // 2 * stride_bn),
            )
            UNIT_EVEN_M: gl.constexpr = TAIL_EVEN_M
            UNIT_EVEN_N: gl.constexpr = TAIL_EVEN_N
        else:
            tail_smem = smem
            tail_kofs = kofs
            UNIT_EVEN_M: gl.constexpr = EVEN_M
            UNIT_EVEN_N: gl.constexpr = EVEN_N
        # The wt reduce-scatter reads with RS_R = TAIL_SPLIT_K + RS_E programs when RS_E
        # idle programs join each tile (rs_readers); otherwise with its own.
        RS_ALL: gl.constexpr = REDUCE_SCATTER and not L2
        # 16-bit readers read whole pairs of blocks.
        RS_MAX: gl.constexpr = pow2_floor(
            NBLK // 2 if is_16bit_partial(p_ptr) else NBLK
        )
        if RS_ALL and RS_E > 0:
            gl.static_assert(RS_R == TAIL_SPLIT_K + RS_E and RS_R <= RS_MAX)
        else:
            RS_OWN: gl.constexpr = min(pow2_floor(TAIL_SPLIT_K), RS_MAX)
            gl.static_assert(
                RS_R == RS_OWN,
                "RS_R is not the tile's own readers (reduce_scatter_config)",
            )
        gl.static_assert((RS_R & (RS_R - 1)) == 0)
        gl.static_assert(RS_NB == 1 or RS_NB * RS_R <= NBLK)
        if L2:
            L2_GRID: gl.constexpr = NUM_PROGRAMS if GRID == 0 else GRID
            gl.static_assert(L2_GRID % NUM_XCDS == 0)
            rank = xcd_blocked_rank(L2_GRID, NUM_XCDS)
            t, j, n_tile = xcd_local_chunk_of(
                rank,
                TAIL_TILES,
                L2_GRID // NUM_XCDS,
                NUM_XCDS,
                TAIL_SPLIT_K,
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
            t = rank // TAIL_SPLIT_K
            j = rank % TAIL_SPLIT_K
            n_tile = TAIL_SPLIT_K
        if SKIP_OUT:
            um, un = tail_unit_coords(
                total_full_tiles * Q + t, M, N, BLOCK_M, BLOCK_N, TBM, TBN, GROUP_SIZE_M
            )
            active = active & (um * TBM < M) & (un * TBN < N)
        if active:
            is_partial = (j != 0).to(gl.int32)
            n_peers = (TAIL_SPLIT_K - 1) * (1 - is_partial)
            OWNER_MAX_PEERS: gl.constexpr = (
                TAIL_SPLIT_K - 1
                if TAIL_SPLIT_K <= owner_unroll_max(BLOCK_M, BLOCK_N)
                else 0
            )
            quadrant_segment(
                total_full_tiles * Q + t,
                even_split_start(j, PAIRS_PER_TILE, TAIL_SPLIT_K) * 2,
                even_split_size(j, PAIRS_PER_TILE, TAIL_SPLIT_K),
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
                tail_smem,
                tail_kofs,
                strides,
                M,
                N,
                TBM,
                TBN,
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
                TAIL_TOP_M if SUBTILE else TOP_M,
                TAIL_LEFT_N if SUBTILE else LEFT_N,
                UNIT_EVEN_M,
                UNIT_EVEN_N,
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
                TILE_M=BLOCK_M,
                TILE_N=BLOCK_N,
            )
        if RS_ALL and RS_E > 0:
            # Programs without a segment (hardware pids from WORKING) read too.
            e = gl.program_id(axis=0) - WORKING
            if (e >= 0) & (e < TAIL_TILES * RS_E):
                # The last trip's prefetch (unmasked on an even K) may still be in
                # flight into the LDS the slice's layout conversions use.
                gl.amd.cdna4.async_copy.wait_group(0)
                rs_extra_reader(
                    e // RS_E,
                    TAIL_SPLIT_K + e % RS_E,
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
                    TAIL_SPLIT_K,
                    RS_R,
                    RS_E,
                    RS_NB,
                    RS_U,
                    ADD_BIAS,
                    EVEN_M,
                    EVEN_N,
                    BAR_STRIDE,
                    TOP_M,
                    LEFT_N,
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
    a_kofs,
    M,
    stride_am,
    stride_ak,
    BLOCK_M: gl.constexpr,
    BLOCK_K: gl.constexpr,
    EVEN_M: gl.constexpr,
    a_dtype: gl.constexpr,
    TOP_M: gl.constexpr = 0,
):
    """Tile m_idx's a_addr = (top_offs, bot_offs, bot_shift, odd_shift), the offsets a
    tuple per half (one per block): the bottom half loads from base + bot_shift +
    bot_offs and the odd K-step adds odd_shift. EVEN_M: the tile-independent a_kofs =
    (top_offs, bot_offs, bot_shift) (equal halves share their offsets). Otherwise
    the tile's clamped offsets, which differ per lane, so bot_shift is 0."""
    a_top, a_bot, a_bot_shift = a_kofs
    if EVEN_M:
        top, bot, bot_shift = a_top, a_bot, a_bot_shift
    else:
        TOPS: gl.constexpr = half_blocks(BLOCK_M, TOP_M, False)
        BOTS: gl.constexpr = half_blocks(BLOCK_M, TOP_M, True)
        m_last = M - 1 - m_idx * BLOCK_M
        if len(TOPS) == 1 and TOPS == BOTS:
            t, b = clamped_a_offsets(
                m_last,
                stride_am,
                stride_ak,
                BLOCK_M,
                BLOCK_K,
                operand_load_layout(0, BLOCK_M // 2, BLOCK_K, a_dtype, gl.num_warps()),
            )
            top, bot = (t,), (b,)
        else:
            top = half_offsets(
                TOPS,
                block_starts(TOPS, 0),
                stride_am,
                stride_ak,
                m_last,
                BLOCK_K,
                a_dtype,
                0,
                True,
            )
            bot = half_offsets(
                BOTS,
                block_starts(BOTS, half_size(BLOCK_M, TOP_M)),
                stride_am,
                stride_ak,
                m_last,
                BLOCK_K,
                a_dtype,
                0,
                True,
            )
        bot_shift = 0
    return top, bot, bot_shift, BLOCK_K * stride_ak


@gluon.jit
def b_load_addressing(
    n_idx,
    b_kofs,
    N,
    stride_bk,
    stride_bn,
    BLOCK_N: gl.constexpr,
    BLOCK_K: gl.constexpr,
    EVEN_N: gl.constexpr,
    b_dtype: gl.constexpr,
    LEFT_N: gl.constexpr = 0,
):
    """Tile n_idx's b_addr = (left_offs, right_offs, right_shift, odd_shift)
    (a_load_addressing)."""
    b_left, b_right, b_right_shift = b_kofs
    if EVEN_N:
        left, right, right_shift = b_left, b_right, b_right_shift
    else:
        LEFTS: gl.constexpr = half_blocks(BLOCK_N, LEFT_N, False)
        RIGHTS: gl.constexpr = half_blocks(BLOCK_N, LEFT_N, True)
        n_last = N - 1 - n_idx * BLOCK_N
        if len(LEFTS) == 1 and LEFTS == RIGHTS:
            lt, rt = clamped_b_offsets(
                n_last,
                stride_bk,
                stride_bn,
                BLOCK_N,
                BLOCK_K,
                operand_load_layout(1, BLOCK_N // 2, BLOCK_K, b_dtype, gl.num_warps()),
            )
            left, right = (lt,), (rt,)
        else:
            left = half_offsets(
                LEFTS,
                block_starts(LEFTS, 0),
                stride_bn,
                stride_bk,
                n_last,
                BLOCK_K,
                b_dtype,
                1,
                True,
            )
            right = half_offsets(
                RIGHTS,
                block_starts(RIGHTS, half_size(BLOCK_N, LEFT_N)),
                stride_bn,
                stride_bk,
                n_last,
                BLOCK_K,
                b_dtype,
                1,
                True,
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
def operand_k_masks(
    k_hi,
    STEP: gl.constexpr,
    BLOCK_K: gl.constexpr,
    SIDE: gl.constexpr,
    FIRST: gl.constexpr,
    dtype: gl.constexpr,
    OPERAND: gl.constexpr,
    NONE: gl.constexpr = False,
):
    """One operand's K-step STEP masks (k_tail_mask) as (first half, second half),
    one per block; equal halves share theirs. NONE: constant all-false masks of the
    blocks' full shapes."""
    FIRSTS: gl.constexpr = half_blocks(SIDE, FIRST, False)
    SECONDS: gl.constexpr = half_blocks(SIDE, FIRST, True)
    first = half_k_masks(k_hi, STEP, BLOCK_K, FIRSTS, dtype, OPERAND, NONE)
    if SECONDS == FIRSTS:
        second = first
    else:
        second = half_k_masks(k_hi, STEP, BLOCK_K, SECONDS, dtype, OPERAND, NONE)
    return first, second


@gluon.jit
def half_k_masks(
    k_hi,
    STEP: gl.constexpr,
    BLOCK_K: gl.constexpr,
    BLOCKS: gl.constexpr,
    dtype: gl.constexpr,
    OPERAND: gl.constexpr,
    NONE: gl.constexpr,
):
    """operand_k_masks of one half."""
    out = ()
    for i in gl.static_range(len(BLOCKS)):
        out = out + (
            block_k_mask(k_hi, STEP, BLOCK_K, BLOCKS[i], dtype, OPERAND, NONE),
        )
    return out


@gluon.jit
def block_k_mask(
    k_hi,
    STEP: gl.constexpr,
    BLOCK_K: gl.constexpr,
    ROWS: gl.constexpr,
    dtype: gl.constexpr,
    OPERAND: gl.constexpr,
    NONE: gl.constexpr,
):
    layout: gl.constexpr = operand_load_layout(
        OPERAND, ROWS, BLOCK_K, dtype, gl.num_warps()
    )
    if NONE and OPERAND == 0:
        m = gl.full([ROWS, BLOCK_K], 0, gl.int1, layout)
    elif NONE:
        m = gl.full([BLOCK_K, ROWS], 0, gl.int1, layout)
    else:
        m = k_tail_mask(k_hi, STEP, BLOCK_K, layout, OPERAND == 1)
    return m


@gluon.jit
def tail_pair_masks(
    k_hi,
    K_RES: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_K: gl.constexpr,
    a_dtype: gl.constexpr,
    b_dtype: gl.constexpr,
    TAIL: gl.constexpr = False,
    TOP_M: gl.constexpr = 0,
    LEFT_N: gl.constexpr = 0,
):
    """(mask_a0, mask_b0, mask_a1, mask_b1) for load_first_pair of a pair whose first
    k_hi K values are in the matrix, each (first half, second half) of per-block
    masks (operand_k_masks). K-step 1 of the last pair always reaches past K,
    K-step 0 only when K_RES > BLOCK_K; a mask that can never drop a lane is None.
    TAIL: k_hi is the last pair's 2 BLOCK_K - K_RES, so with K_RES >= BLOCK_K K-step 1
    drops every lane and its masks are constant."""
    mask_a0, mask_b0 = None, None
    if K_RES > BLOCK_K:
        mask_a0 = operand_k_masks(k_hi, 0, BLOCK_K, BLOCK_M, TOP_M, a_dtype, 0)
        mask_b0 = operand_k_masks(k_hi, 0, BLOCK_K, BLOCK_N, LEFT_N, b_dtype, 1)
    CONST: gl.constexpr = TAIL and K_RES >= BLOCK_K
    mask_a1 = operand_k_masks(k_hi, 1, BLOCK_K, BLOCK_M, TOP_M, a_dtype, 0, CONST)
    mask_b1 = operand_k_masks(k_hi, 1, BLOCK_K, BLOCK_N, LEFT_N, b_dtype, 1, CONST)
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
    1 (ragged K), (first half, second half) of per-block masks (tail_pair_masks), None
    for unmasked."""
    smemA_top, smemA_bot, smemB_left, smemB_right = smem
    a_top_offsets, a_bot_offsets, a_bot_shift, a_odd_shift = a_addr
    b_left_offsets, b_right_offsets, b_right_shift, b_odd_shift = b_addr
    ma0_t, ma0_b, mb0_l, mb0_r = None, None, None, None
    ma1_t, ma1_b, mb1_l, mb1_r = None, None, None, None
    if mask_a0 is not None:
        ma0_t, ma0_b = mask_a0
        mb0_l, mb0_r = mask_b0
    if mask_a1 is not None:
        ma1_t, ma1_b = mask_a1
        mb1_l, mb1_r = mask_b1

    g_idx = 0
    copy_half(smemB_left, g_idx, b_base, b_left_offsets, mb0_l)
    gl.amd.cdna4.async_copy.commit_group()
    copy_half(smemA_top, g_idx, a_base, a_top_offsets, ma0_t)
    gl.amd.cdna4.async_copy.commit_group()
    copy_half(smemA_bot, g_idx, a_base + a_bot_shift, a_bot_offsets, ma0_b)
    gl.amd.cdna4.async_copy.commit_group()
    copy_half(smemB_right, g_idx, b_base + b_right_shift, b_right_offsets, mb0_r)
    gl.amd.cdna4.async_copy.commit_group()

    g_idx = 1
    copy_half(smemB_left, g_idx, b_base + b_odd_shift, b_left_offsets, mb1_l)
    gl.amd.cdna4.async_copy.commit_group()
    copy_half(smemA_top, g_idx, a_base + a_odd_shift, a_top_offsets, ma1_t)
    gl.amd.cdna4.async_copy.commit_group()
    copy_half(
        smemA_bot, g_idx, a_base + a_bot_shift + a_odd_shift, a_bot_offsets, ma1_b
    )
    gl.amd.cdna4.async_copy.commit_group()
    copy_half(
        smemB_right,
        g_idx,
        b_base + b_right_shift + b_odd_shift,
        b_right_offsets,
        mb1_r,
    )
    gl.amd.cdna4.async_copy.commit_group()

    gl.amd.cdna4.async_copy.wait_group(6)
    b_left = read_half(smemB_left, 0, dotOpLayoutB)
    a_top = read_half(smemA_top, 0, dotOpLayoutA)
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
    acc_tl = quad_mma(a_top, b_left, acc_tl)

    gl.amd.cdna4.async_copy.wait_group(5)
    a_bot = read_half(smemA_bot, 0, dotOpLayoutA)

    copy_half(smemB_left, 0, b_base, b_left_offsets)
    gl.amd.cdna4.async_copy.commit_group()

    ########################################
    ## Region 1: C_bl = DOT(a_bot, b_left)
    ########################################
    acc_bl = quad_mma(a_bot, b_left, acc_bl)

    gl.amd.cdna4.async_copy.wait_group(5)
    b_right = read_half(smemB_right, 0, dotOpLayoutB)

    copy_half(smemA_top, 0, a_base, a_top_offsets)
    gl.amd.cdna4.async_copy.commit_group()

    ########################################
    ## Region 2: C_tr = DOT(a_top, b_right)
    ########################################
    acc_tr = quad_mma(a_top, b_right, acc_tr)

    gl.amd.cdna4.async_copy.wait_group(5)
    b_left = read_half(smemB_left, 1, dotOpLayoutB)

    copy_half(smemA_bot, 0, a_base + a_bot_shift, a_bot_offsets)
    gl.amd.cdna4.async_copy.commit_group()

    ########################################
    ## Region 3: C_br = DOT(a_bot, b_right)
    ########################################
    acc_br = quad_mma(a_bot, b_right, acc_br)

    gl.amd.cdna4.async_copy.wait_group(5)
    a_top = read_half(smemA_top, 1, dotOpLayoutA)

    copy_half(smemB_right, 0, b_base + b_right_shift, b_right_offsets)
    gl.amd.cdna4.async_copy.commit_group()

    ## =============================================================
    ## Loop unroll: Sub-iteration 1: consume buffer 1, prefetch
    ## into buffer 1. The odd K-step's loads add a_odd_shift / b_odd_shift to the base.
    ## =============================================================

    ########################################
    ## Region 0: C_tl = DOT(a_top, b_left)
    ########################################
    if not SKIP_ODD:
        acc_tl = quad_mma(a_top, b_left, acc_tl)

    gl.amd.cdna4.async_copy.wait_group(5)
    a_bot = read_half(smemA_bot, 1, dotOpLayoutA)

    copy_half(smemB_left, 1, b_base + b_odd_shift, b_left_offsets)
    gl.amd.cdna4.async_copy.commit_group()

    ########################################
    ## Region 1: C_bl = DOT(a_bot, b_left)
    ########################################
    if not SKIP_ODD:
        acc_bl = quad_mma(a_bot, b_left, acc_bl)

    gl.amd.cdna4.async_copy.wait_group(5)
    b_right = read_half(smemB_right, 1, dotOpLayoutB)

    copy_half(smemA_top, 1, a_base + a_odd_shift, a_top_offsets)
    gl.amd.cdna4.async_copy.commit_group()

    ########################################
    ## Region 2: C_tr = DOT(a_top, b_right)
    ########################################
    if not SKIP_ODD:
        acc_tr = quad_mma(a_top, b_right, acc_tr)

    gl.amd.cdna4.async_copy.wait_group(5)
    b_left = read_half(smemB_left, 0, dotOpLayoutB)

    copy_half(smemA_bot, 1, a_base + a_bot_shift + a_odd_shift, a_bot_offsets)
    gl.amd.cdna4.async_copy.commit_group()

    ########################################
    ## Region 3: C_br = DOT(a_bot, b_right)
    ########################################
    if not SKIP_ODD:
        acc_br = quad_mma(a_bot, b_right, acc_br)

    gl.amd.cdna4.async_copy.wait_group(5)
    a_top = read_half(smemA_top, 0, dotOpLayoutA)

    copy_half(smemB_right, 1, b_base + b_right_shift + b_odd_shift, b_right_offsets)
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
    TOP_M: gl.constexpr = 0,
    LEFT_N: gl.constexpr = 0,
):
    """The four fp32 accumulators (tl, bl, tr, br) of a tile, each a tuple of blocks
    in quad_mma's order, starting from the tile's bias[N] (or zero).

    Adding the bias here rather than before the C stores keeps it out of the epilogue,
    where the stores already hold the register peak; the epilogue add spills. Without
    bias the accumulators are a constant zero, which a peeled first MFMA folds into an
    inline-0 C operand.
    """
    if is_uneven(BLOCK_M, TOP_M, BLOCK_N, LEFT_N):
        tl, bl, tr, br = init_acc_blocks(
            bias_ptr,
            pid_n,
            0,
            BLOCK_M,
            BLOCK_N,
            TOP_M,
            LEFT_N,
            mfmaLayout,
            ADD_BIAS,
            True,
        )
    else:
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
        tl, bl, tr, br = (acc_l,), (acc_l,), (acc_r,), (acc_r,)
    return tl, bl, tr, br


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
    TOP_M: gl.constexpr = 0,
    LEFT_N: gl.constexpr = 0,
):
    """init_acc, with the bias loaded only for the tile's first n_left columns on a
    ragged N."""
    if is_uneven(BLOCK_M, TOP_M, BLOCK_N, LEFT_N):
        tl, bl, tr, br = init_acc_blocks(
            bias_ptr,
            pid_n,
            n_left,
            BLOCK_M,
            BLOCK_N,
            TOP_M,
            LEFT_N,
            mfmaLayout,
            ADD_BIAS,
            EVEN_N,
        )
    elif ADD_BIAS and not EVEN_N:
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
        tl, bl, tr, br = (acc_l,), (acc_l,), (acc_r,), (acc_r,)
    else:
        tl, bl, tr, br = init_acc(
            bias_ptr, pid_n, BLOCK_M, BLOCK_N, mfmaLayout, ADD_BIAS
        )
    return tl, bl, tr, br


@gluon.jit
def init_acc_blocks(
    bias_ptr,
    pid_n,
    n_left,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    TOP_M: gl.constexpr,
    LEFT_N: gl.constexpr,
    mfmaLayout: gl.constexpr,
    ADD_BIAS: gl.constexpr,
    EVEN_N: gl.constexpr,
):
    """init_acc(_edge) of an uneven tile: each block from its column block's bias
    (masked to the first n_left columns on a ragged N), or zero."""
    TOPS: gl.constexpr = half_blocks(BLOCK_M, TOP_M, False)
    BOTS: gl.constexpr = half_blocks(BLOCK_M, TOP_M, True)
    LEFTS: gl.constexpr = half_blocks(BLOCK_N, LEFT_N, False)
    RIGHTS: gl.constexpr = half_blocks(BLOCK_N, LEFT_N, True)
    bias_l = half_bias(
        bias_ptr,
        pid_n,
        n_left,
        LEFTS,
        block_starts(LEFTS, 0),
        BLOCK_N,
        mfmaLayout,
        ADD_BIAS,
        EVEN_N,
    )
    bias_r = half_bias(
        bias_ptr,
        pid_n,
        n_left,
        RIGHTS,
        block_starts(RIGHTS, half_size(BLOCK_N, LEFT_N)),
        BLOCK_N,
        mfmaLayout,
        ADD_BIAS,
        EVEN_N,
    )
    tl = quad_init(TOPS, LEFTS, bias_l, mfmaLayout, ADD_BIAS)
    bl = quad_init(BOTS, LEFTS, bias_l, mfmaLayout, ADD_BIAS)
    tr = quad_init(TOPS, RIGHTS, bias_r, mfmaLayout, ADD_BIAS)
    br = quad_init(BOTS, RIGHTS, bias_r, mfmaLayout, ADD_BIAS)
    return tl, bl, tr, br


@gluon.jit
def half_bias(
    bias_ptr,
    pid_n,
    n_left,
    COLS: gl.constexpr,
    C0S: gl.constexpr,
    BLOCK_N: gl.constexpr,
    mfmaLayout: gl.constexpr,
    ADD_BIAS: gl.constexpr,
    EVEN_N: gl.constexpr,
):
    """bias[N] of a column half's blocks in fp32 (an empty tuple without bias)."""
    out = ()
    if ADD_BIAS:
        bias_base = bias_ptr + pid_n * BLOCK_N
        for j in gl.static_range(len(COLS)):
            offs = gl.arange(0, COLS[j], gl.SliceLayout(0, mfmaLayout))
            if EVEN_N:
                b = gl.amd.cdna3.buffer_load(ptr=bias_base + C0S[j], offsets=offs)
            else:
                b = gl.amd.cdna3.buffer_load(
                    ptr=bias_base + C0S[j],
                    offsets=offs,
                    mask=offs < n_left - C0S[j],
                    other=0.0,
                )
            out = out + (b.to(gl.float32),)
    return out


@gluon.jit
def quad_init(
    ROWS: gl.constexpr,
    COLS: gl.constexpr,
    bias,
    mfmaLayout: gl.constexpr,
    ADD_BIAS: gl.constexpr,
):
    """A quadrant's fp32 accumulator blocks in quad_mma's order, each its column
    block's bias or zero."""
    out = ()
    for j in gl.static_range(len(COLS)):
        for i in gl.static_range(len(ROWS)):
            acc = gl.zeros((ROWS[i], COLS[j]), gl.float32, mfmaLayout)
            if ADD_BIAS:
                acc = acc + bias[j][None, :]
            out = out + (acc,)
    return out


@gluon.jit
def c_store_addressing(
    c_ptr,
    pid_m,
    pid_n,
    m_left,
    n_left,
    stride_cm,
    stride_cn,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    TOP_M: gl.constexpr,
    LEFT_N: gl.constexpr,
    CLAMP_ROWS: gl.constexpr,
    MASKS: gl.constexpr,
    EVEN_M: gl.constexpr,
    EVEN_N: gl.constexpr,
):
    """C store addressing of a tile: (bases, offsets, masks), each (tl, bl, tr, br) of
    per-block tuples in quad_mma's order. A block stores at bases[q][b] + offsets[q][b]
    with the offsets' layout; masks (MASKS, on a ragged M or N; else empty tuples)
    keep the rows and columns inside C.

    The tile's row goes in the 64-bit scalar base, so the 32-bit offsets only span
    BLOCK_M rows. Its column stays in the vector offsets, which keeps them
    tile-dependent: the compiler can't hoist them above the tile loop and keep them
    live (and spilled) through the K loop. The block shifts go in the scalar base too,
    so blocks of one shape share one offset tensor (all four quadrants of a
    power-of-two tile). CLAMP_ROWS: the rows are clamped to the tile's last one for
    the same reason (only rows the store masks drop change). The row term carries the
    loads' multiple_of hint and n_left is hinted a multiple of 8 (the wrapper checks
    N), so each thread's 8-column group keeps one predicate and one 16-byte store when
    stride_cm is not a multiple of 16."""
    out_bytes: gl.constexpr = c_ptr.dtype.element_ty.primitive_bitwidth // 8
    if not is_uneven(BLOCK_M, TOP_M, BLOCK_N, LEFT_N):
        # Store layout sized to the half tile: 16 contiguous bytes per thread where
        # the quadrant has enough elements, and one warp spans BLOCK_N // 2 columns
        # ([4, 16] lanes at BLOCK_N = 256).
        gStoreLayoutC: gl.constexpr = coalesced_store_layout(
            BLOCK_M // 2, BLOCK_N // 2, out_bytes, gl.num_warps()
        )
        offs_cm = gl.arange(0, BLOCK_M // 2, gl.SliceLayout(1, gStoreLayoutC))
        offs_cn = gl.arange(0, BLOCK_N // 2, gl.SliceLayout(0, gStoreLayoutC))
        c_tl_base = c_ptr + pid_m.to(gl.int64) * BLOCK_M * stride_cm
        col = pid_n * BLOCK_N + offs_cn
        if CLAMP_ROWS:
            c_rows = gl.multiple_of(stride_cm * gl.minimum(offs_cm, m_left - 1), 8)
        else:
            c_rows = gl.multiple_of(stride_cm * offs_cm, 8)
        c_offsets = c_rows[:, None] + stride_cn * col[None, :]
        c_tr_base = c_tl_base + BLOCK_N * stride_cn // 2
        c_bl_base = c_tl_base + BLOCK_M * stride_cm // 2
        c_br_base = c_bl_base + BLOCK_N * stride_cn // 2
        masks = (), (), (), ()
        if MASKS and not (EVEN_M and EVEN_N):
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
            masks = (mask_tl,), (mask_bl,), (mask_tr,), (mask_br,)
        bases = (c_tl_base,), (c_bl_base,), (c_tr_base,), (c_br_base,)
        offsets = (c_offsets,), (c_offsets,), (c_offsets,), (c_offsets,)
    else:
        c_tl_base = c_ptr + pid_m.to(gl.int64) * BLOCK_M * stride_cm
        TOPS: gl.constexpr = half_blocks(BLOCK_M, TOP_M, False)
        BOTS: gl.constexpr = half_blocks(BLOCK_M, TOP_M, True)
        LEFTS: gl.constexpr = half_blocks(BLOCK_N, LEFT_N, False)
        RIGHTS: gl.constexpr = half_blocks(BLOCK_N, LEFT_N, True)
        T0S: gl.constexpr = block_starts(TOPS, 0)
        B0S: gl.constexpr = block_starts(BOTS, half_size(BLOCK_M, TOP_M))
        L0S: gl.constexpr = block_starts(LEFTS, 0)
        R0S: gl.constexpr = block_starts(RIGHTS, half_size(BLOCK_N, LEFT_N))
        tl = quad_c_addressing(
            c_tl_base,
            pid_n,
            m_left,
            n_left,
            stride_cm,
            stride_cn,
            TOPS,
            T0S,
            LEFTS,
            L0S,
            BLOCK_N,
            out_bytes,
            CLAMP_ROWS,
            MASKS,
            EVEN_M,
            EVEN_N,
        )
        bl = quad_c_addressing(
            c_tl_base,
            pid_n,
            m_left,
            n_left,
            stride_cm,
            stride_cn,
            BOTS,
            B0S,
            LEFTS,
            L0S,
            BLOCK_N,
            out_bytes,
            CLAMP_ROWS,
            MASKS,
            EVEN_M,
            EVEN_N,
        )
        tr = quad_c_addressing(
            c_tl_base,
            pid_n,
            m_left,
            n_left,
            stride_cm,
            stride_cn,
            TOPS,
            T0S,
            RIGHTS,
            R0S,
            BLOCK_N,
            out_bytes,
            CLAMP_ROWS,
            MASKS,
            EVEN_M,
            EVEN_N,
        )
        br = quad_c_addressing(
            c_tl_base,
            pid_n,
            m_left,
            n_left,
            stride_cm,
            stride_cn,
            BOTS,
            B0S,
            RIGHTS,
            R0S,
            BLOCK_N,
            out_bytes,
            CLAMP_ROWS,
            MASKS,
            EVEN_M,
            EVEN_N,
        )
        bases = tl[0], bl[0], tr[0], br[0]
        offsets = tl[1], bl[1], tr[1], br[1]
        masks = tl[2], bl[2], tr[2], br[2]
    return bases, offsets, masks


@gluon.jit
def quad_c_addressing(
    c_tl_base,
    pid_n,
    m_left,
    n_left,
    stride_cm,
    stride_cn,
    ROWS: gl.constexpr,
    R0S: gl.constexpr,
    COLS: gl.constexpr,
    C0S: gl.constexpr,
    BLOCK_N: gl.constexpr,
    OUT_BYTES: gl.constexpr,
    CLAMP_ROWS: gl.constexpr,
    MASKS: gl.constexpr,
    EVEN_M: gl.constexpr,
    EVEN_N: gl.constexpr,
):
    """c_store_addressing of an uneven tile's quadrant: (bases, offsets, masks) of
    its blocks in quad_mma's order."""
    qb = ()
    qo = ()
    qm = ()
    for j in gl.static_range(len(COLS)):
        for i in gl.static_range(len(ROWS)):
            b, o, m = block_c_addressing(
                c_tl_base,
                pid_n,
                m_left,
                n_left,
                stride_cm,
                stride_cn,
                ROWS[i],
                R0S[i],
                COLS[j],
                C0S[j],
                BLOCK_N,
                OUT_BYTES,
                CLAMP_ROWS,
                MASKS,
                EVEN_M,
                EVEN_N,
            )
            qb = qb + (b,)
            qo = qo + (o,)
            if MASKS and not (EVEN_M and EVEN_N):
                qm = qm + (m,)
    return qb, qo, qm


@gluon.jit
def block_c_addressing(
    c_tl_base,
    pid_n,
    m_left,
    n_left,
    stride_cm,
    stride_cn,
    ROWS: gl.constexpr,
    R0: gl.constexpr,
    COLS: gl.constexpr,
    C0: gl.constexpr,
    BLOCK_N: gl.constexpr,
    OUT_BYTES: gl.constexpr,
    CLAMP_ROWS: gl.constexpr,
    MASKS: gl.constexpr,
    EVEN_M: gl.constexpr,
    EVEN_N: gl.constexpr,
):
    """One block's (base, offsets, mask) (c_store_addressing)."""
    SL: gl.constexpr = coalesced_store_layout(ROWS, COLS, OUT_BYTES, gl.num_warps())
    r = gl.arange(0, ROWS, gl.SliceLayout(1, SL))
    c = gl.arange(0, COLS, gl.SliceLayout(0, SL))
    if CLAMP_ROWS:
        rr = gl.multiple_of(stride_cm * gl.minimum(r, m_left - 1), 8)
    else:
        rr = gl.multiple_of(stride_cm * r, 8)
    offs = rr[:, None] + stride_cn * (pid_n * BLOCK_N + c)[None, :]
    base = c_tl_base + R0 * stride_cm + C0 * stride_cn
    m = ()
    if MASKS and not EVEN_M and not EVEN_N:
        m = (r[:, None] + R0 < m_left) & (c[None, :] + C0 < n_left)
    elif MASKS and not EVEN_M:
        m = r[:, None] + R0 < m_left
    elif MASKS and not EVEN_N:
        m = c[None, :] + C0 < n_left
    return base, offs, m


@gluon.jit
def store_quad(acc, bases, offsets, masks):
    """C stores of a quadrant's blocks (c_store_addressing), converted to C's type
    and to each block's store layout; masks is empty when nothing is masked."""
    for b in gl.static_range(len(acc)):
        c = acc[b].to(bases[b].dtype.element_ty)
        c = gl.convert_layout(c, layout=offsets[b].type.layout)
        if len(masks) == 0:
            gl.amd.cdna3.buffer_store(ptr=bases[b], offsets=offsets[b], stored_value=c)
        else:
            gl.amd.cdna3.buffer_store(
                ptr=bases[b], offsets=offsets[b], stored_value=c, mask=masks[b]
            )


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
    EVEN_M: gl.constexpr,
    EVEN_N: gl.constexpr,
    EVEN_K: gl.constexpr,
    K_RES: gl.constexpr,
    total_full_tiles,
    sk_target,
    SK_PREFETCH: gl.constexpr,
    TOP_M: gl.constexpr = 0,
    LEFT_N: gl.constexpr = 0,
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
    a_kofs, b_kofs = kofs
    a_dtype: gl.constexpr = a_ptr.dtype.element_ty
    b_dtype: gl.constexpr = b_ptr.dtype.element_ty
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
        bias_ptr,
        pid_n,
        n_left,
        BLOCK_M,
        BLOCK_N,
        mfmaLayout,
        ADD_BIAS,
        EVEN_N,
        TOP_M,
        LEFT_N,
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
        a_kofs,
        M,
        stride_am,
        stride_ak,
        BLOCK_M,
        BLOCK_K,
        EVEN_M,
        a_dtype,
        TOP_M,
    )
    nb_addr = b_load_addressing(
        next_n,
        b_kofs,
        N,
        stride_bk,
        stride_bn,
        BLOCK_N,
        BLOCK_K,
        EVEN_N,
        b_dtype,
        LEFT_N,
    )
    na_top, na_bot, na_bot_shift, na_odd_shift = na_addr
    nb_left, nb_right, nb_right_shift, nb_odd_shift = nb_addr
    # Computed here rather than with a_next, so they are not live across the K loop.
    if SK_PREFETCH and not EVEN_K:
        k_hi = gl.where(has_next, 2 * BLOCK_K - K_RES, sk_khi)
        mask_a0 = operand_k_masks(k_hi, 0, BLOCK_K, BLOCK_M, TOP_M, a_dtype, 0)
        mask_b0 = operand_k_masks(k_hi, 0, BLOCK_K, BLOCK_N, LEFT_N, b_dtype, 1)
        mask_a1 = operand_k_masks(k_hi, 1, BLOCK_K, BLOCK_M, TOP_M, a_dtype, 0)
        mask_b1 = operand_k_masks(k_hi, 1, BLOCK_K, BLOCK_N, LEFT_N, b_dtype, 1)
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
            a_dtype,
            b_dtype,
            True,
            TOP_M,
            LEFT_N,
        )
    # Per half: (first, second) of per-block masks, or None.
    ma0_t, ma0_b, mb0_l, mb0_r = None, None, None, None
    ma1_t, ma1_b, mb1_l, mb1_r = None, None, None, None
    if mask_a0 is not None:
        ma0_t, ma0_b = mask_a0
        mb0_l, mb0_r = mask_b0
    if mask_a1 is not None:
        ma1_t, ma1_b = mask_a1
        mb1_l, mb1_r = mask_b1

    ## Epilogue: 4-quadrant stores with natural-pipeline ordering, each async copy
    ## prefetching the next tile's first pair.

    c_bases, c_offs, c_masks = c_store_addressing(
        c_ptr,
        pid_m,
        pid_n,
        m_left,
        n_left,
        stride_cm,
        stride_cn,
        BLOCK_M,
        BLOCK_N,
        TOP_M,
        LEFT_N,
        not EVEN_M,
        True,
        EVEN_M,
        EVEN_N,
    )

    ## Iter iterMax - 2: same 4-region pattern as main loop, each AC prefetching the
    ## next tile's K-step 0 into the buffer the main loop would refill here, so every
    ## wait keeps 5 groups in flight.
    acc_tl = quad_mma(a_top, b_left, acc_tl)
    gl.amd.cdna4.async_copy.wait_group(5)
    l_idx = (iterMax - 2) % 2
    a_bot = read_half(smemA_bot, l_idx, dotOpLayoutA)
    copy_half(smemB_left, 0, b_next, nb_left, mb0_l)
    gl.amd.cdna4.async_copy.commit_group()

    acc_bl = quad_mma(a_bot, b_left, acc_bl)
    gl.amd.cdna4.async_copy.wait_group(5)
    b_right = read_half(smemB_right, l_idx, dotOpLayoutB)
    copy_half(smemA_top, 0, a_next, na_top, ma0_t)
    gl.amd.cdna4.async_copy.commit_group()

    acc_tr = quad_mma(a_top, b_right, acc_tr)
    gl.amd.cdna4.async_copy.wait_group(5)
    g_idx = 1 - l_idx
    b_left = read_half(smemB_left, g_idx, dotOpLayoutB)
    copy_half(smemA_bot, 0, a_next + na_bot_shift, na_bot, ma0_b)
    gl.amd.cdna4.async_copy.commit_group()

    acc_br = quad_mma(a_bot, b_right, acc_br)
    gl.amd.cdna4.async_copy.wait_group(5)
    a_top = read_half(smemA_top, g_idx, dotOpLayoutA)
    copy_half(smemB_right, 0, b_next + nb_right_shift, nb_right, mb0_r)
    gl.amd.cdna4.async_copy.commit_group()

    ## Iter iterMax - 1
    ## Natural-pipeline epilogue: each store follows its MFMA with one
    ## MFMA cycle of gap, yielding uniform MFMA-store interleaving. The ACs
    ## prefetch the next tile's K-step 1.
    acc_tl = quad_mma(a_top, b_left, acc_tl)
    gl.amd.cdna4.async_copy.wait_group(5)
    a_bot = read_half(smemA_bot, g_idx, dotOpLayoutA)
    copy_half(smemB_left, 1, b_next + nb_odd_shift, nb_left, mb1_l)
    gl.amd.cdna4.async_copy.commit_group()

    acc_bl = quad_mma(a_bot, b_left, acc_bl)
    gl.amd.cdna4.async_copy.wait_group(5)
    b_right = read_half(smemB_right, g_idx, dotOpLayoutB)
    copy_half(smemA_top, 1, a_next + na_odd_shift, na_top, ma1_t)
    gl.amd.cdna4.async_copy.commit_group()

    store_quad(acc_tl, c_bases[0], c_offs[0], c_masks[0])

    acc_tr = quad_mma(a_top, b_right, acc_tr)
    copy_half(smemA_bot, 1, a_next + na_bot_shift + na_odd_shift, na_bot, ma1_b)
    gl.amd.cdna4.async_copy.commit_group()

    store_quad(acc_bl, c_bases[1], c_offs[1], c_masks[1])

    acc_br = quad_mma(a_bot, b_right, acc_br)

    # The next tile's K-step 0 operands (prefetched at iterMax - 2), read while the
    # last MFMAs and stores drain; this tile's operand registers are all dead after
    # acc_br. Both reads are deferred to here rather than interleaved as in the main
    # loop: reading b_left after acc_tr keeps it live across the c_bl store, which
    # costs 24-36 AGPR moves per tile and, on a ragged M and N, an extra spill.
    gl.amd.cdna4.async_copy.wait_group(5)
    b_left = read_half(smemB_left, 0, dotOpLayoutB)
    a_top = read_half(smemA_top, 0, dotOpLayoutA)
    copy_half(smemB_right, 1, b_next + nb_right_shift + nb_odd_shift, nb_right, mb1_r)
    gl.amd.cdna4.async_copy.commit_group()

    store_quad(acc_tr, c_bases[2], c_offs[2], c_masks[2])

    store_quad(acc_br, c_bases[3], c_offs[3], c_masks[3])

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
    TAIL_TILES: gl.constexpr,
    PAIRS_PER_TILE: gl.constexpr,
    TAIL_SPLIT_K: gl.constexpr,
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
            TAIL_TILES, PAIRS_PER_TILE, NUM_PROGRAMS, NUM_XCDS
        )
        sk_t, _, k0, _, _ = e2e_segment_ending_at(
            sk_pair_end,
            sk_pair_start,
            rank,
            TAIL_TILES,
            PAIRS_PER_TILE,
            NUM_PROGRAMS,
        )
        has = 1
    else:
        if L2:
            rank = xcd_blocked_rank(NUM_PROGRAMS, NUM_XCDS)
            sk_t, j, n_tile = xcd_local_chunk_of(
                rank, TAIL_TILES, NUM_PROGRAMS // NUM_XCDS, NUM_XCDS, TAIL_SPLIT_K
            )
            has = (n_tile > 0).to(gl.int32)
        else:
            rank = xcd_blocked_rank(TAIL_TILES * TAIL_SPLIT_K, NUM_XCDS, IDLE=True)
            has = (rank < TAIL_TILES * TAIL_SPLIT_K).to(gl.int32)
            sk_t = rank // TAIL_SPLIT_K
            j = rank % TAIL_SPLIT_K
        k0 = even_split_start(j, PAIRS_PER_TILE, TAIL_SPLIT_K) * 2 * has
        sk_t = sk_t * has
    return has, total_full_tiles + sk_t, k0


@gluon.jit
def xcd_local_chunk_of(
    rank,
    S: gl.constexpr,
    PROGS_PER_XCD: gl.constexpr,
    NUM_XCDS: gl.constexpr,
    TAIL_SPLIT_K: gl.constexpr,
):
    """(tile, chunk, programs on the tile) of the program with this rank in the
    XCD-local split (l2), with 0 programs if it idles. XCD x = rank // PROGS_PER_XCD
    owns part x of an even split of the S Stream-K tiles over the XCDs; its local
    program l works on its tile l // TAIL_SPLIT_K as chunk l % TAIL_SPLIT_K, so a
    tile's TAIL_SPLIT_K programs are on one XCD. The host checks
    ceil(S / NUM_XCDS) * TAIL_SPLIT_K <= PROGS_PER_XCD."""
    x = rank // PROGS_PER_XCD
    local = rank % PROGS_PER_XCD
    tl = local // TAIL_SPLIT_K
    t = even_split_start(x, S, NUM_XCDS) + tl
    n_tile = TAIL_SPLIT_K * (tl < even_split_size(x, S, NUM_XCDS)).to(gl.int32)
    return t, local % TAIL_SPLIT_K, n_tile


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
    TAIL_TILES: gl.constexpr,
    PAIRS_PER_TILE: gl.constexpr,
    NUM_PROGRAMS: gl.constexpr,
    NUM_XCDS: gl.constexpr,
):
    """End-to-end split: (rank, sk_pair_start, sk_pair_end) of this program, whose range
    [sk_pair_start, sk_pair_end) of the Stream-K tiles' pairs is part rank of an even
    split over the full grid (XCD-blocked ranks)."""
    SK_PAIRS: gl.constexpr = TAIL_TILES * PAIRS_PER_TILE
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
    TAIL_TILES: gl.constexpr,
    PAIRS_PER_TILE: gl.constexpr,
    NUM_PROGRAMS: gl.constexpr,
):
    """End-to-end split, segments run last to first (see the module docstring):
    (tile t, seg_start, k0, is_owner, n_peers) of this program's segment that ends at
    pair seg_end. It covers pairs [seg_start, seg_end) of Stream-K tile t, from logical
    K-step k0. is_owner: it holds the tile's last pair and collects the tile. n_peers:
    the programs holding the tile's earlier pairs (0 unless is_owner)."""
    SK_PAIRS: gl.constexpr = TAIL_TILES * PAIRS_PER_TILE
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
WAVE_LANES = gl.constexpr(64)  # lanes of a warp (wave64)
# 64, the 4 columns of 16 lanes
MFMA_GROUP_ELEMS = gl.constexpr(MFMA_DIM.value * MFMA_LANE_ELEMS.value)
# 256, a 16x16 output
MFMA_WARP_ELEMS = gl.constexpr(WAVE_LANES.value * MFMA_LANE_ELEMS.value)
MFMA_BLOCK_ELEMS = gl.constexpr(4 * MFMA_WARP_ELEMS.value)  # 1024, the 32x32 block
# 8, the 4-column groups across a block
BLOCK_GROUPS = gl.constexpr(MFMA_BLOCK_DIM.value // MFMA_LANE_ELEMS.value)
# A pair of 32x32 blocks of 16-bit partials (pair_layout): 2048 elements, a lane's
# 4 registers of each block, 8 elements.
PAIR_ELEMS = gl.constexpr(2 * MFMA_BLOCK_ELEMS.value)
PAIR_LANE_ELEMS = gl.constexpr(2 * MFMA_LANE_ELEMS.value)


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


# 16-bit partials (P typed like C, bf16 or fp16) are stored in pairs of 32x32 blocks,
# so a lane still writes 16 bytes: its 4 registers of each block, [pair][4 warps][64
# lanes][8 elements]. A quadrant of two or more blocks pairs its register vectors 2p
# and 2p + 1 (32 columns apart, or 32 rows when it is one block wide); the 64x64
# tile's one-block quadrants pair across the tile, tl with tr and bl with br. Every
# view below only renames registers: the partials stay in the MFMA layout's registers.


@constexpr_function
def is_16bit_partial(p_ptr):
    """True if the partials P points to are 16-bit (stored in pairs of blocks)."""
    return p_ptr.dtype.element_ty.primitive_bitwidth == 16


@constexpr_function
def pair_layout(pairs):
    """The MFMA layout of a 16-bit partial viewed as (pairs, 32, 8, 8) (to_pairs): a
    block row, its 4-column group and a lane's 8 registers, 4 columns of each block
    of the pair; the pairs in registers."""
    reg = [[0, 0, 0, 1], [0, 0, 0, 2], [0, 0, 0, 4]]
    reg += [[1 << i, 0, 0, 0] for i in range(pairs.bit_length() - 1)]
    return gl.DistributedLinearLayout(
        reg_bases=reg,
        lane_bases=[
            [0, 1, 0, 0],
            [0, 2, 0, 0],
            [0, 4, 0, 0],
            [0, 8, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 2, 0],
        ],
        warp_bases=[[0, 0, 4, 0], [0, 16, 0, 0]],
        block_bases=[],
        shape=[
            pairs,
            MFMA_BLOCK_DIM.value,
            BLOCK_GROUPS.value,
            PAIR_LANE_ELEMS.value,
        ],
    )


@gluon.jit
def pair_offsets(PAIRS: gl.constexpr, layout: gl.constexpr):
    """Element offsets of PAIRS pairs of the 16-bit map, (PAIRS, 32, 8, 8) in layout:
    element (p, r, g, e) is at 2048 p + 8 (64 warp + lane) + e, with lane = r % 16 +
    16 (g % 4) and warp = g // 4 + 2 (r // 16), the MFMA layout's lane and warp."""
    p = gl.arange(
        0, PAIRS, gl.SliceLayout(1, gl.SliceLayout(2, gl.SliceLayout(3, layout)))
    )
    r = gl.arange(
        0,
        MFMA_BLOCK_DIM,
        gl.SliceLayout(0, gl.SliceLayout(2, gl.SliceLayout(3, layout))),
    )
    g = gl.arange(
        0, BLOCK_GROUPS, gl.SliceLayout(0, gl.SliceLayout(1, gl.SliceLayout(3, layout)))
    )
    e = gl.arange(
        0,
        PAIR_LANE_ELEMS,
        gl.SliceLayout(0, gl.SliceLayout(1, gl.SliceLayout(2, layout))),
    )
    p = p[:, None, None, None]
    r = r[None, :, None, None]
    g = g[None, None, :, None]
    e = e[None, None, None, :]
    GROUPS: gl.constexpr = MFMA_DIM // MFMA_LANE_ELEMS
    lane = (r % MFMA_DIM) + MFMA_DIM * (g % GROUPS)
    warp = g // GROUPS + MFMA_WARPS_MN[1] * (r // MFMA_DIM)
    return p * PAIR_ELEMS + (warp * WAVE_LANES + lane) * PAIR_LANE_ELEMS + e


@gluon.jit
def to_pairs(x, HALF_M: gl.constexpr, HALF_N: gl.constexpr):
    """A quadrant of two or more blocks, (HALF_M, HALF_N) in the MFMA layout ->
    (VQ / 2, 32, 8, 8) in pair_layout."""
    VM: gl.constexpr = HALF_M // MFMA_BLOCK_DIM
    VN: gl.constexpr = HALF_N // MFMA_BLOCK_DIM
    VP: gl.constexpr = VM * VN // 2
    y = gl.permute(
        gl.reshape(x, [VM, MFMA_BLOCK_DIM, VN, BLOCK_GROUPS, MFMA_LANE_ELEMS]),
        [0, 2, 1, 3, 4],
    )
    y = gl.permute(
        gl.reshape(y, [VP, 2, MFMA_BLOCK_DIM, BLOCK_GROUPS, MFMA_LANE_ELEMS]),
        [0, 2, 3, 1, 4],
    )
    return gl.convert_layout(
        gl.reshape(y, [VP, MFMA_BLOCK_DIM, BLOCK_GROUPS, PAIR_LANE_ELEMS]),
        pair_layout(VP),
        assert_trivial=True,
    )


@gluon.jit
def from_pairs(y, HALF_M: gl.constexpr, HALF_N: gl.constexpr, mfmaLayout: gl.constexpr):
    """The inverse of to_pairs."""
    VM: gl.constexpr = HALF_M // MFMA_BLOCK_DIM
    VN: gl.constexpr = HALF_N // MFMA_BLOCK_DIM
    VP: gl.constexpr = VM * VN // 2
    x = gl.permute(
        gl.reshape(y, [VP, MFMA_BLOCK_DIM, BLOCK_GROUPS, 2, MFMA_LANE_ELEMS]),
        [0, 3, 1, 2, 4],
    )
    x = gl.permute(
        gl.reshape(x, [VM, VN, MFMA_BLOCK_DIM, BLOCK_GROUPS, MFMA_LANE_ELEMS]),
        [0, 2, 1, 3, 4],
    )
    return gl.convert_layout(
        gl.reshape(x, [HALF_M, HALF_N]), mfmaLayout, assert_trivial=True
    )


@gluon.jit
def join_pair(a, b):
    """Two one-block quadrants (32x32, MFMA layout) -> one pair, (1, 32, 8, 8)."""
    y = gl.permute(
        gl.reshape(gl.join(a, b), [MFMA_BLOCK_DIM, BLOCK_GROUPS, MFMA_LANE_ELEMS, 2]),
        [0, 1, 3, 2],
    )
    return gl.convert_layout(
        gl.reshape(y, [1, MFMA_BLOCK_DIM, BLOCK_GROUPS, PAIR_LANE_ELEMS]),
        pair_layout(1),
        assert_trivial=True,
    )


@gluon.jit
def split_pair(y, mfmaLayout: gl.constexpr):
    """The inverse of join_pair."""
    x = gl.reshape(
        gl.permute(
            gl.reshape(y, [MFMA_BLOCK_DIM, BLOCK_GROUPS, 2, MFMA_LANE_ELEMS]),
            [0, 1, 3, 2],
        ),
        [MFMA_BLOCK_DIM, MFMA_BLOCK_DIM, 2],
    )
    a, b = gl.split(x)
    return (
        gl.convert_layout(a, mfmaLayout, assert_trivial=True),
        gl.convert_layout(b, mfmaLayout, assert_trivial=True),
    )


@gluon.jit
def partial_map(
    p_ptr, HALF_M: gl.constexpr, HALF_N: gl.constexpr, mfmaLayout: gl.constexpr
):
    """The offsets of a part of a slot, a quadrant (a pair of quadrants at 64x64 with
    16-bit partials), and zeros in the layout the owner sums its peers' parts in."""
    if is_16bit_partial(p_ptr):
        VP: gl.constexpr = max(HALF_M * HALF_N // PAIR_ELEMS, 1)
        offsets = pair_offsets(VP, pair_layout(VP))
        zeros = gl.zeros(
            (VP, MFMA_BLOCK_DIM, BLOCK_GROUPS, PAIR_LANE_ELEMS),
            gl.float32,
            pair_layout(VP),
        )
    else:
        offsets = lane_contiguous_offsets(HALF_M, HALF_N, mfmaLayout)
        zeros = gl.zeros((HALF_M, HALF_N), gl.float32, mfmaLayout)
    return offsets, zeros


@gluon.jit
def from_partials(
    sums, p_ptr, HALF_M: gl.constexpr, HALF_N: gl.constexpr, mfmaLayout: gl.constexpr
):
    """peer_sums' parts as quadrants in the MFMA layout (fp32: unchanged)."""
    s0, s1, s2, s3 = sums
    if is_16bit_partial(p_ptr):
        if HALF_M * HALF_N == MFMA_BLOCK_ELEMS:
            tl, tr = split_pair(s0, mfmaLayout)
            bl, br = split_pair(s1, mfmaLayout)
            s0, s1, s2, s3 = tl, bl, tr, br
        else:
            s0 = from_pairs(s0, HALF_M, HALF_N, mfmaLayout)
            s1 = from_pairs(s1, HALF_M, HALF_N, mfmaLayout)
            s2 = from_pairs(s2, HALF_M, HALF_N, mfmaLayout)
            s3 = from_pairs(s3, HALF_M, HALF_N, mfmaLayout)
    return s0, s1, s2, s3


@gluon.jit
def store_partial(
    accs,
    q: gl.constexpr,
    p_ptr,
    rank,
    p_offsets,
    HALF_M: gl.constexpr,
    HALF_N: gl.constexpr,
    CACHE: gl.constexpr = ".wt",
):
    """Quadrant q of a partial to slot P[rank]: write-through, or the default policy
    for the l2 exchange. 16-bit at 64x64: quadrant q < 2 (tl, bl) stores one pair with
    the quadrant to its right (q + 2: tr, br); the calls for q = 2 and 3 store
    nothing."""
    QUAD: gl.constexpr = HALF_M * HALF_N
    SLOT: gl.constexpr = 4 * QUAD
    P_TYPE: gl.constexpr = p_ptr.dtype.element_ty
    slot = p_ptr + rank.to(gl.int64) * SLOT
    if not is_16bit_partial(p_ptr):
        gl.amd.cdna3.buffer_store(
            stored_value=accs[q], ptr=slot + q * QUAD, offsets=p_offsets, cache=CACHE
        )
    elif QUAD > MFMA_BLOCK_ELEMS:
        part = to_pairs(accs[q].to(P_TYPE), HALF_M, HALF_N)
        gl.amd.cdna3.buffer_store(
            stored_value=part, ptr=slot + q * QUAD, offsets=p_offsets, cache=CACHE
        )
    elif q < 2:
        part = join_pair(accs[q].to(P_TYPE), accs[q + 2].to(P_TYPE))
        gl.amd.cdna3.buffer_store(
            stored_value=part, ptr=slot + q * (2 * QUAD), offsets=p_offsets, cache=CACHE
        )


@gluon.jit
def owner_store(
    acc,
    psum,
    R0: gl.constexpr,
    C0: gl.constexpr,
    bias_ptr,
    pid_n,
    c_base,
    c_offsets,
    BLOCK_N: gl.constexpr,
    ADD_BIAS: gl.constexpr,
    out_dtype: gl.constexpr,
    mfmaLayout: gl.constexpr,
    m_left,
    n_left,
    EVEN_M: gl.constexpr,
    EVEN_N: gl.constexpr,
):
    """The block of C at tile row R0, column C0 (a quadrant of a power-of-two tile):
    the accumulator plus psum (the peers' partials) and the bias, stored in
    c_offsets' layout. Contributors' partials exclude the bias, so it is added once,
    here."""
    COLS: gl.constexpr = acc.shape[1]
    if ADD_BIAS:
        offs_pn = gl.arange(0, COLS, gl.SliceLayout(0, mfmaLayout))
        if EVEN_N:
            bias = gl.amd.cdna3.buffer_load(
                ptr=bias_ptr + pid_n * BLOCK_N + C0, offsets=offs_pn
            )
        else:
            bias = gl.amd.cdna3.buffer_load(
                ptr=bias_ptr + pid_n * BLOCK_N + C0,
                offsets=offs_pn,
                mask=offs_pn < n_left - C0,
                other=0.0,
            )
        psum = psum + bias.to(gl.float32)[None, :]
    store_c(
        acc + psum,
        c_base,
        c_offsets,
        out_dtype,
        c_offsets.type.layout,
        m_left - R0,
        n_left - C0,
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
      and waits for each load in turn.
    16-bit at 64x64 reads parts of two quadrants (partial_map): NQ = 4 is two parts,
    s0 and s1 (from_partials splits them)."""
    QUAD: gl.constexpr = HALF_M * HALF_N
    SLOT: gl.constexpr = 4 * QUAD
    PAIR_Q: gl.constexpr = is_16bit_partial(p_ptr) and QUAD == MFMA_BLOCK_ELEMS
    gl.static_assert(not PAIR_Q or (q0 == 0 and NQ == 4))
    PART: gl.constexpr = 2 * QUAD if PAIR_Q else QUAD
    NP: gl.constexpr = 2 if PAIR_Q else NQ
    s0, s1, s2, s3 = z, z, z, z
    if MAX_PEERS > 0:
        for j in gl.static_range(MAX_PEERS):
            peer = (rank + PEER_STEP * (1 + j)).to(gl.int64)
            live = j < n_peers
            s0 += gl.amd.cdna3.buffer_load(
                ptr=p_ptr + peer * SLOT + q0 * PART,
                offsets=p_offsets,
                mask=live,
                other=0.0,
                cache=".cv",
            ).to(gl.float32)
            if NP > 1:
                s1 += gl.amd.cdna3.buffer_load(
                    ptr=p_ptr + peer * SLOT + (q0 + 1) * PART,
                    offsets=p_offsets,
                    mask=live,
                    other=0.0,
                    cache=".cv",
                ).to(gl.float32)
            if NP > 2:
                s2 += gl.amd.cdna3.buffer_load(
                    ptr=p_ptr + peer * SLOT + (q0 + 2) * PART,
                    offsets=p_offsets,
                    mask=live,
                    other=0.0,
                    cache=".cv",
                ).to(gl.float32)
                s3 += gl.amd.cdna3.buffer_load(
                    ptr=p_ptr + peer * SLOT + (q0 + 3) * PART,
                    offsets=p_offsets,
                    mask=live,
                    other=0.0,
                    cache=".cv",
                ).to(gl.float32)
    else:
        for j in range(n_peers):
            peer = (rank + PEER_STEP * (1 + j)).to(gl.int64)
            x0 = gl.amd.cdna3.buffer_load(
                ptr=p_ptr + peer * SLOT + q0 * PART, offsets=p_offsets, cache=".cv"
            ).to(gl.float32)
            if NP == 1:
                s0 += x0
            elif NP == 2:
                x1 = gl.amd.cdna3.buffer_load(
                    ptr=p_ptr + peer * SLOT + (q0 + 1) * PART,
                    offsets=p_offsets,
                    cache=".cv",
                ).to(gl.float32)
                s0 += x0
                s1 += x1
            else:
                x1 = gl.amd.cdna3.buffer_load(
                    ptr=p_ptr + peer * SLOT + (q0 + 1) * PART,
                    offsets=p_offsets,
                    cache=".cv",
                ).to(gl.float32)
                x2 = gl.amd.cdna3.buffer_load(
                    ptr=p_ptr + peer * SLOT + (q0 + 2) * PART,
                    offsets=p_offsets,
                    cache=".cv",
                ).to(gl.float32)
                x3 = gl.amd.cdna3.buffer_load(
                    ptr=p_ptr + peer * SLOT + (q0 + 3) * PART,
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
    c_offs,
    m_left,
    n_left,
    BLOCK_N: gl.constexpr,
    PEER_STEP: gl.constexpr,
    ADD_BIAS: gl.constexpr,
    out_dtype: gl.constexpr,
    mfmaLayout: gl.constexpr,
    EVEN_M: gl.constexpr,
    EVEN_N: gl.constexpr,
    QPB: gl.constexpr,
    MAX_PEERS: gl.constexpr,
    BLOCK_M: gl.constexpr = 0,
    TOP_M: gl.constexpr = 0,
    LEFT_N: gl.constexpr = 0,
):
    """The owner ending of a segment. A contributor (is_partial) stores its four
    quadrants to P[rank], drains and raises locks[rank]. The owner waits for its
    n_peers peers' flags (rank + PEER_STEP i), reads their partials QPB quadrants at a
    time (16 fp32 loads per wave: 4 at 128x128, 2 at 256x128 / 128x256, 1 at 256x256;
    peer_sums), adds each batch's sums into the accumulator once, outside the peer loop
    (a loop or branch that carries the AGPR accumulator spills), and stores that batch
    of C before the next batch's reads, so only one batch's sums are live. Each partial
    has one reader, so the owner lowers its peers' flags for the next launch; the
    kernel boundary orders that against it.

    accs, c_bases and c_offs hold a tuple of blocks per quadrant (c_store_addressing):
    an uneven tile runs owner_ending_blocks."""
    if is_uneven(BLOCK_M, TOP_M, BLOCK_N, LEFT_N):
        owner_ending_blocks(
            accs,
            is_partial,
            n_peers,
            rank,
            pid_n,
            bias_ptr,
            p_ptr,
            locks_ptr,
            c_bases,
            c_offs,
            m_left,
            n_left,
            BLOCK_M,
            BLOCK_N,
            TOP_M,
            LEFT_N,
            PEER_STEP,
            ADD_BIAS,
            out_dtype,
            mfmaLayout,
            EVEN_M,
            EVEN_N,
            MAX_PEERS,
        )
    else:
        owner_ending_quads(
            (accs[0][0], accs[1][0], accs[2][0], accs[3][0]),
            is_partial,
            n_peers,
            rank,
            pid_n,
            bias_ptr,
            p_ptr,
            locks_ptr,
            (c_bases[0][0], c_bases[1][0], c_bases[2][0], c_bases[3][0]),
            c_offs[0][0],
            m_left,
            n_left,
            BLOCK_N,
            PEER_STEP,
            ADD_BIAS,
            out_dtype,
            mfmaLayout,
            EVEN_M,
            EVEN_N,
            QPB,
            MAX_PEERS,
        )


@gluon.jit
def owner_ending_quads(
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
    mfmaLayout: gl.constexpr,
    EVEN_M: gl.constexpr,
    EVEN_N: gl.constexpr,
    QPB: gl.constexpr,
    MAX_PEERS: gl.constexpr,
):
    """owner_ending of a power-of-two tile: accs and c_bases are its four quadrants,
    which share c_offsets."""
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
    p_offsets, z = partial_map(p_ptr, HALF_M, HALF_N, mfmaLayout)
    if not STORE_EACH:
        if is_partial != 0:
            for q in gl.static_range(4):
                store_partial(accs, q, p_ptr, rank, p_offsets, HALF_M, HALF_N)
    for q0 in gl.static_range(0, 4, QPB):
        if STORE_EACH:
            if is_partial != 0:
                store_partial(accs, q0, p_ptr, rank, p_offsets, HALF_M, HALF_N)
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
        sums = from_partials(sums, p_ptr, HALF_M, HALF_N, mfmaLayout)
        if is_partial == 0:
            for i in gl.static_range(QPB):
                owner_store(
                    accs[q0 + i],
                    sums[i],
                    ((q0 + i) % 2) * HALF_M,
                    ((q0 + i) // 2) * HALF_N,
                    bias_ptr,
                    pid_n,
                    c_bases[q0 + i],
                    c_offsets,
                    BLOCK_N,
                    ADD_BIAS,
                    out_dtype,
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


# ---------------------------------------------------------------------------------
# Partials of an uneven tile, block by block
#
# The slot holds the tile's blocks (tile_blocks) back to back, each straight from the
# MFMA layout as a power-of-two quadrant's is (lane_contiguous_offsets). A part is
# what one access per lane covers in order: a block, or with 16-bit partials two
# one-block (32x32) blocks stored as one pair (join_pair), which tile_blocks keeps
# adjacent at the slot's end, 16 bytes per lane. A tile with an odd number of
# one-block blocks stores the last one alone in 16 bits straight from the MFMA layout
# ("lone"), 8 bytes per lane.
# ---------------------------------------------------------------------------------


@constexpr_function
def tile_parts(block_m, top_m, block_n, left_n, p16, field):
    """One field of every part of an uneven tile's slot, in slot order: "b0" its
    first block (tile_blocks' index), "nb" its blocks (1, or 2 for a 16-bit pair of
    one-block blocks), "off" its first element, "vq" its 32x32 blocks, "rows" /
    "cols" its first block's, "loads" its loads per lane, "lone" a one-block block
    stored alone in 16 bits (8-byte accesses)."""
    vq = tile_blocks(block_m, top_m, block_n, left_n, "vq")
    rows = tile_blocks(block_m, top_m, block_n, left_n, "rows")
    cols = tile_blocks(block_m, top_m, block_n, left_n, "cols")
    off = tile_blocks(block_m, top_m, block_n, left_n, "off")
    parts, b = [], 0
    while b < len(vq):
        if p16 and vq[b] == 1 and b + 1 < len(vq):
            parts.append(
                dict(
                    b0=b, nb=2, off=off[b], vq=2, rows=32, cols=32, loads=1, lone=False
                )
            )
            b += 2
        elif p16 and vq[b] == 1:
            parts.append(
                dict(b0=b, nb=1, off=off[b], vq=1, rows=32, cols=32, loads=1, lone=True)
            )
            b += 1
        else:
            loads = vq[b] // 2 if p16 else vq[b]
            parts.append(
                dict(
                    b0=b,
                    nb=1,
                    off=off[b],
                    vq=vq[b],
                    rows=rows[b],
                    cols=cols[b],
                    loads=loads,
                    lone=False,
                )
            )
            b += 1
    return tuple(p[field] for p in parts)


@constexpr_function
def owner_unroll_max(block_m, block_n):
    """The largest TAIL_SPLIT_K whose owner unrolls its peer loop (MAX_PEERS): 8, or 4
    when the accumulators fill all 256 AGPRs (256x256), where unrolling 5-7 peers
    spills (up to 25 VGPRs) and is up to 8% slower than the run-time loop."""
    return 4 if block_m * block_n >= 256 * 256 else 8


@constexpr_function
def part_batches(block_m, top_m, block_n, left_n, p16, budget):
    """Boundaries of the owner's batches: runs of consecutive parts of at most
    `budget` 16-byte loads per lane together (at least one part each)."""
    loads = tile_parts(block_m, top_m, block_n, left_n, p16, "loads")
    out, total = [0], 0
    for i, n in enumerate(loads):
        if total > 0 and total + n > budget:
            out.append(i)
            total = 0
        total += n
    out.append(len(loads))
    return tuple(out)


@constexpr_function
def rs_part_groups(block_m, top_m, block_n, left_n, p16, nb, field):
    """The reduce-scatter's groups over an uneven tile's parts: "nb" blocks per group
    (at most nb, so a group stays in one block; a 16-bit pair of one-block blocks is one
    group of 2), "gs" a part's first group, "gr" its groups, "total" all groups."""
    vq = tile_parts(block_m, top_m, block_n, left_n, p16, "vq")
    nbp = tile_parts(block_m, top_m, block_n, left_n, p16, "nb")
    out, g = [], 0
    for v, n in zip(vq, nbp):
        k = 2 if n == 2 else max(pow2_floor(min(v, nb)), 2 if p16 and v > 1 else 1)
        out.append(dict(nb=k, gs=g, gr=v // k))
        g += v // k
    if field == "total":
        return g
    return tuple(o[field] for o in out)


@gluon.jit
def part_map(p_ptr, P: gl.constexpr, GEOM: gl.constexpr, mfmaLayout: gl.constexpr):
    """partial_map of part P of an uneven tile: a lone 16-bit block keeps the MFMA
    layout's lane-contiguous offsets (4 elements, 8 bytes, per lane)."""
    P16: gl.constexpr = is_16bit_partial(p_ptr)
    BM: gl.constexpr = GEOM[0]
    TM: gl.constexpr = GEOM[1]
    BN: gl.constexpr = GEOM[2]
    LN: gl.constexpr = GEOM[3]
    ROWS: gl.constexpr = tile_parts(BM, TM, BN, LN, P16, "rows")[P]
    COLS: gl.constexpr = tile_parts(BM, TM, BN, LN, P16, "cols")[P]
    if tile_parts(BM, TM, BN, LN, P16, "lone")[P]:
        offsets = lane_contiguous_offsets(ROWS, COLS, mfmaLayout)
        zeros = gl.zeros((ROWS, COLS), gl.float32, mfmaLayout)
    else:
        offsets, zeros = partial_map(p_ptr, ROWS, COLS, mfmaLayout)
    return offsets, zeros


@gluon.jit
def part_values(accs, P: gl.constexpr, GEOM: gl.constexpr, P_TYPE: gl.constexpr):
    """Part P of an uneven tile's partial, as stored: fp32 blocks as they are, 16-bit
    in pairs (to_pairs, or join_pair for two one-block blocks), a lone one-block
    block as it is."""
    P16: gl.constexpr = P_TYPE.primitive_bitwidth == 16
    BM: gl.constexpr = GEOM[0]
    TM: gl.constexpr = GEOM[1]
    BN: gl.constexpr = GEOM[2]
    LN: gl.constexpr = GEOM[3]
    B0: gl.constexpr = tile_parts(BM, TM, BN, LN, P16, "b0")[P]
    NB: gl.constexpr = tile_parts(BM, TM, BN, LN, P16, "nb")[P]
    QS: gl.constexpr = tile_blocks(BM, TM, BN, LN, "q")
    KS: gl.constexpr = tile_blocks(BM, TM, BN, LN, "k")
    LONE: gl.constexpr = tile_parts(BM, TM, BN, LN, P16, "lone")[P]
    a = accs[QS[B0]][KS[B0]]
    if not P16:
        part = a
    elif LONE:
        part = a.to(P_TYPE)
    elif NB == 1:
        part = to_pairs(a.to(P_TYPE), a.shape[0], a.shape[1])
    else:
        b = accs[QS[B0 + 1]][KS[B0 + 1]]
        part = join_pair(a.to(P_TYPE), b.to(P_TYPE))
    return part


@gluon.jit
def store_partial_blocks(
    accs,
    p_ptr,
    rank,
    GEOM: gl.constexpr,
    mfmaLayout: gl.constexpr,
    CACHE: gl.constexpr = ".wt",
):
    """An uneven tile's partial to slot P[rank], part by part (store_partial)."""
    P16: gl.constexpr = is_16bit_partial(p_ptr)
    BM: gl.constexpr = GEOM[0]
    TM: gl.constexpr = GEOM[1]
    BN: gl.constexpr = GEOM[2]
    LN: gl.constexpr = GEOM[3]
    OFFS: gl.constexpr = tile_parts(BM, TM, BN, LN, P16, "off")
    slot = p_ptr + rank.to(gl.int64) * (BM * BN)
    for P in gl.static_range(len(OFFS)):
        p_offsets = part_map(p_ptr, P, GEOM, mfmaLayout)[0]
        gl.amd.cdna3.buffer_store(
            stored_value=part_values(accs, P, GEOM, p_ptr.dtype.element_ty),
            ptr=slot + OFFS[P],
            offsets=p_offsets,
            cache=CACHE,
        )


@gluon.jit
def peer_sums_parts(
    p_ptr,
    rank,
    n_peers,
    P0: gl.constexpr,
    P1: gl.constexpr,
    PEER_STEP: gl.constexpr,
    GEOM: gl.constexpr,
    MAX_PEERS: gl.constexpr,
    mfmaLayout: gl.constexpr,
):
    """peer_sums over parts P0 .. P1 - 1 of an uneven tile: their fp32 sums over the
    partials of peers rank + PEER_STEP i, i = 1 .. n_peers, in the parts' layouts
    (partial_map); a peer's parts are one batch of loads."""
    P16: gl.constexpr = is_16bit_partial(p_ptr)
    BM: gl.constexpr = GEOM[0]
    TM: gl.constexpr = GEOM[1]
    BN: gl.constexpr = GEOM[2]
    LN: gl.constexpr = GEOM[3]
    OFFS: gl.constexpr = tile_parts(BM, TM, BN, LN, P16, "off")
    SLOT: gl.constexpr = BM * BN
    offs = ()
    sums = ()
    for P in gl.static_range(P0, P1):
        o, z = part_map(p_ptr, P, GEOM, mfmaLayout)
        offs = offs + (o,)
        sums = sums + (z,)
    if MAX_PEERS > 0:
        for jj in gl.static_range(MAX_PEERS):
            peer = (rank + PEER_STEP * (1 + jj)).to(gl.int64)
            live = jj < n_peers
            new = ()
            for p in gl.static_range(P1 - P0):
                new = new + (
                    sums[p]
                    + gl.amd.cdna3.buffer_load(
                        ptr=p_ptr + peer * SLOT + OFFS[P0 + p],
                        offsets=offs[p],
                        mask=live,
                        other=0.0,
                        cache=".cv",
                    ).to(gl.float32),
                )
            sums = new
    else:
        for jj in range(n_peers):
            peer = (rank + PEER_STEP * (1 + jj)).to(gl.int64)
            xs = ()
            for p in gl.static_range(P1 - P0):
                xs = xs + (
                    gl.amd.cdna3.buffer_load(
                        ptr=p_ptr + peer * SLOT + OFFS[P0 + p],
                        offsets=offs[p],
                        cache=".cv",
                    ).to(gl.float32),
                )
            new = ()
            for p in gl.static_range(P1 - P0):
                new = new + (sums[p] + xs[p],)
            sums = new
    return sums


@gluon.jit
def part_blocks(
    s, P: gl.constexpr, GEOM: gl.constexpr, P16: gl.constexpr, mfmaLayout: gl.constexpr
):
    """A part's fp32 sum as its blocks in the MFMA layout (from_partials)."""
    BM: gl.constexpr = GEOM[0]
    TM: gl.constexpr = GEOM[1]
    BN: gl.constexpr = GEOM[2]
    LN: gl.constexpr = GEOM[3]
    NB: gl.constexpr = tile_parts(BM, TM, BN, LN, P16, "nb")[P]
    LONE: gl.constexpr = tile_parts(BM, TM, BN, LN, P16, "lone")[P]
    if not P16 or LONE:
        out = (s,)
    elif NB == 1:
        ROWS: gl.constexpr = tile_parts(BM, TM, BN, LN, P16, "rows")[P]
        COLS: gl.constexpr = tile_parts(BM, TM, BN, LN, P16, "cols")[P]
        out = (from_pairs(s, ROWS, COLS, mfmaLayout),)
    else:
        a, b = split_pair(s, mfmaLayout)
        out = (a, b)
    return out


@gluon.jit
def owner_ending_blocks(
    accs,
    is_partial,
    n_peers,
    rank,
    pid_n,
    bias_ptr,
    p_ptr,
    locks_ptr,
    c_bases,
    c_offs,
    m_left,
    n_left,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    TOP_M: gl.constexpr,
    LEFT_N: gl.constexpr,
    PEER_STEP: gl.constexpr,
    ADD_BIAS: gl.constexpr,
    out_dtype: gl.constexpr,
    mfmaLayout: gl.constexpr,
    EVEN_M: gl.constexpr,
    EVEN_N: gl.constexpr,
    MAX_PEERS: gl.constexpr,
):
    """owner_ending of an uneven tile: the contributors store every part, the owner
    reads its peers' partials in batches of parts (part_batches: 16 16-byte loads per
    lane, as the power-of-two owner's quadrants per batch) and stores each batch's
    blocks of C before the next batch's reads. With 16-bit partials and four or more
    peers unrolled (MAX_PEERS), the batch is 8 loads: a 16-bit load holds two blocks,
    so 16 would hold twice fp32's blocks in fp32 sums, which spills (up to 441
    VGPRs at 192x320, split 8)."""
    P16: gl.constexpr = is_16bit_partial(p_ptr)
    GEOM: gl.constexpr = (BLOCK_M, TOP_M, BLOCK_N, LEFT_N)
    BATCH: gl.constexpr = part_batches(
        BLOCK_M, TOP_M, BLOCK_N, LEFT_N, P16, 8 if P16 and MAX_PEERS >= 4 else 16
    )
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

    if is_partial != 0:
        store_partial_blocks(accs, p_ptr, rank, GEOM, mfmaLayout)
    for bi in gl.static_range(len(BATCH) - 1):
        sums = peer_sums_parts(
            p_ptr,
            rank,
            n_peers,
            BATCH[bi],
            BATCH[bi + 1],
            PEER_STEP,
            GEOM,
            MAX_PEERS,
            mfmaLayout,
        )
        if is_partial == 0:
            for p in gl.static_range(BATCH[bi + 1] - BATCH[bi]):
                owner_store_part(
                    accs,
                    sums[p],
                    BATCH[bi] + p,
                    GEOM,
                    P16,
                    bias_ptr,
                    pid_n,
                    c_bases,
                    c_offs,
                    m_left,
                    n_left,
                    ADD_BIAS,
                    out_dtype,
                    mfmaLayout,
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
def owner_store_part(
    accs,
    s,
    P: gl.constexpr,
    GEOM: gl.constexpr,
    P16: gl.constexpr,
    bias_ptr,
    pid_n,
    c_bases,
    c_offs,
    m_left,
    n_left,
    ADD_BIAS: gl.constexpr,
    out_dtype: gl.constexpr,
    mfmaLayout: gl.constexpr,
    EVEN_M: gl.constexpr,
    EVEN_N: gl.constexpr,
):
    """The owner's C stores of part P's blocks: each accumulator plus its peers' sum
    s (part_blocks) and the bias."""
    BM: gl.constexpr = GEOM[0]
    TM: gl.constexpr = GEOM[1]
    BN: gl.constexpr = GEOM[2]
    LN: gl.constexpr = GEOM[3]
    B0: gl.constexpr = tile_parts(BM, TM, BN, LN, P16, "b0")[P]
    blocks = part_blocks(s, P, GEOM, P16, mfmaLayout)
    for e in gl.static_range(len(blocks)):
        owner_store_block(
            accs,
            blocks[e],
            B0 + e,
            GEOM,
            bias_ptr,
            pid_n,
            c_bases,
            c_offs,
            m_left,
            n_left,
            ADD_BIAS,
            out_dtype,
            mfmaLayout,
            EVEN_M,
            EVEN_N,
        )


@gluon.jit
def owner_store_block(
    accs,
    psum,
    B: gl.constexpr,
    GEOM: gl.constexpr,
    bias_ptr,
    pid_n,
    c_bases,
    c_offs,
    m_left,
    n_left,
    ADD_BIAS: gl.constexpr,
    out_dtype: gl.constexpr,
    mfmaLayout: gl.constexpr,
    EVEN_M: gl.constexpr,
    EVEN_N: gl.constexpr,
):
    """owner_store of block B (tile_blocks' index) of an uneven tile."""
    BM: gl.constexpr = GEOM[0]
    TM: gl.constexpr = GEOM[1]
    BN: gl.constexpr = GEOM[2]
    LN: gl.constexpr = GEOM[3]
    Q: gl.constexpr = tile_blocks(BM, TM, BN, LN, "q")[B]
    KK: gl.constexpr = tile_blocks(BM, TM, BN, LN, "k")[B]
    owner_store(
        accs[Q][KK],
        psum,
        tile_blocks(BM, TM, BN, LN, "r0")[B],
        tile_blocks(BM, TM, BN, LN, "c0")[B],
        bias_ptr,
        pid_n,
        c_bases[Q][KK],
        c_offs[Q][KK],
        BN,
        ADD_BIAS,
        out_dtype,
        mfmaLayout,
        m_left,
        n_left,
        EVEN_M,
        EVEN_N,
    )


@gluon.jit
def whole_tile_blocks(
    accs,
    pid_n,
    bias_ptr,
    c_bases,
    c_offs,
    m_left,
    n_left,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    TOP_M: gl.constexpr,
    LEFT_N: gl.constexpr,
    ADD_BIAS: gl.constexpr,
    out_dtype: gl.constexpr,
    mfmaLayout: gl.constexpr,
    EVEN_M: gl.constexpr,
    EVEN_N: gl.constexpr,
):
    """An uneven tile that no other program shares: its blocks straight to C."""
    GEOM: gl.constexpr = (BLOCK_M, TOP_M, BLOCK_N, LEFT_N)
    QS: gl.constexpr = tile_blocks(BLOCK_M, TOP_M, BLOCK_N, LEFT_N, "q")
    KS: gl.constexpr = tile_blocks(BLOCK_M, TOP_M, BLOCK_N, LEFT_N, "k")
    for B in gl.static_range(len(QS)):
        owner_store_block(
            accs,
            gl.zeros(accs[QS[B]][KS[B]].shape, gl.float32, mfmaLayout),
            B,
            GEOM,
            bias_ptr,
            pid_n,
            c_bases,
            c_offs,
            m_left,
            n_left,
            ADD_BIAS,
            out_dtype,
            mfmaLayout,
            EVEN_M,
            EVEN_N,
        )


@gluon.jit
def reduce_scatter_blocks(
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
    TOP_M: gl.constexpr,
    LEFT_N: gl.constexpr,
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
    """reduce_scatter_slice of an uneven tile. The slot's parts are cut into groups
    of up to NB blocks that stay inside one part (rs_part_groups), and reader j of
    `readers` reduces groups [G j / readers, G (j + 1) / readers) of the G in all, a
    part at a time: each group's blocks are loaded and summed as in the power-of-two
    slice, with as many loads in flight (NB U), and stored as their band of C. A 16-bit
    pair of one-block blocks is one group whose two blocks are stored where they lie."""
    P16: gl.constexpr = is_16bit_partial(p_ptr)
    G: gl.constexpr = rs_part_groups(BLOCK_M, TOP_M, BLOCK_N, LEFT_N, P16, NB, "total")
    NPARTS: gl.constexpr = len(tile_parts(BLOCK_M, TOP_M, BLOCK_N, LEFT_N, P16, "nb"))
    glo = (G * j) // readers
    ghi = (G * (j + 1)) // readers
    # reduce_scatter_slice's lane order and in-block offsets.
    L3: gl.constexpr = gl.BlockedLayout([1, 1, 4], [1, 16, 4], [1, 2, 2], [1, 2, 0])
    r = gl.arange(0, 32, gl.SliceLayout(0, gl.SliceLayout(2, L3)))[None, :, None]
    c = gl.arange(0, 32, gl.SliceLayout(0, gl.SliceLayout(1, L3)))[None, None, :]
    GROUPS: gl.constexpr = MFMA_DIM // MFMA_LANE_ELEMS
    in_blk = (
        (r % MFMA_DIM) * MFMA_LANE_ELEMS
        + (r // MFMA_DIM) * (2 * MFMA_WARP_ELEMS)
        + c
        + (c // MFMA_LANE_ELEMS % GROUPS) * (MFMA_GROUP_ELEMS - MFMA_LANE_ELEMS)
        + (c // MFMA_DIM) * (MFMA_WARP_ELEMS - MFMA_DIM)
    )
    for P in gl.static_range(NPARTS):
        rs_part(
            p_ptr,
            bias_ptr,
            c_tile,
            b,
            n,
            glo,
            ghi,
            pid_n,
            stride_cm,
            stride_cn,
            r,
            c,
            in_blk,
            P,
            (BLOCK_M, TOP_M, BLOCK_N, LEFT_N),
            P16,
            ADD_BIAS,
            NB,
            U,
            out_dtype,
            LOAD_CACHE,
            m_left,
            n_left,
            EVEN_M,
            EVEN_N,
        )


@gluon.jit
def rs_part(
    p_ptr,
    bias_ptr,
    c_tile,
    b,
    n,
    glo,
    ghi,
    pid_n,
    stride_cm,
    stride_cn,
    r,
    c,
    in_blk,
    P: gl.constexpr,
    GEOM: gl.constexpr,
    P16: gl.constexpr,
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
    """reduce_scatter_blocks' groups [glo, ghi) that lie in part P."""
    BM: gl.constexpr = GEOM[0]
    TM: gl.constexpr = GEOM[1]
    BN: gl.constexpr = GEOM[2]
    LN: gl.constexpr = GEOM[3]
    SLOT: gl.constexpr = BM * BN
    B0: gl.constexpr = tile_parts(BM, TM, BN, LN, P16, "b0")[P]
    PAIR: gl.constexpr = tile_parts(BM, TM, BN, LN, P16, "nb")[P] == 2
    OFF: gl.constexpr = tile_parts(BM, TM, BN, LN, P16, "off")[P]
    VN: gl.constexpr = tile_parts(BM, TM, BN, LN, P16, "cols")[P] // 32
    R0: gl.constexpr = tile_blocks(BM, TM, BN, LN, "r0")[B0]
    C0: gl.constexpr = tile_blocks(BM, TM, BN, LN, "c0")[B0]
    NBR: gl.constexpr = rs_part_groups(BM, TM, BN, LN, P16, NB, "nb")[P]
    GS: gl.constexpr = rs_part_groups(BM, TM, BN, LN, P16, NB, "gs")[P]
    GR: gl.constexpr = rs_part_groups(BM, TM, BN, LN, P16, NB, "gr")[P]
    # As many loads in flight as the power-of-two slice's NB U.
    UR: gl.constexpr = max(U * NB // NBR, 1)
    L3: gl.constexpr = gl.BlockedLayout([1, 1, 4], [1, 16, 4], [1, 2, 2], [1, 2, 0])
    i = gl.arange(0, NBR, gl.SliceLayout(1, gl.SliceLayout(2, L3)))[:, None, None]
    ib = gl.arange(0, NBR, gl.SliceLayout(1, gl.SliceLayout(1, L3)))[:, None]
    cb = gl.arange(0, 32, gl.SliceLayout(0, gl.SliceLayout(1, L3)))[None, :]
    # A lone 16-bit block is read as an fp32 block is, 4 elements (8 bytes) per lane.
    PAIRS: gl.constexpr = P16 and not tile_parts(BM, TM, BN, LN, P16, "lone")[P]
    if PAIRS:
        L4: gl.constexpr = gl.BlockedLayout(
            [1, 1, 1, 8], [1, 16, 4, 1], [1, 2, 2, 1], [3, 1, 2, 0]
        )
        in_pair = pair_offsets(NBR // 2, L4)
    for gg in range(gl.maximum(glo, GS), gl.minimum(ghi, GS + GR)):
        g0 = gg - GS
        if PAIRS:
            p_off = OFF + g0 * (NBR // 2) * (2 * MFMA_BLOCK_ELEMS) + in_pair
            s4 = gl.zeros((NBR // 2, 32, 8, 8), gl.float32, L4)
            for p0 in range(0, n, UR):
                for u in gl.static_range(UR):
                    slot = p_ptr + (b + p0 + u).to(gl.int64) * SLOT
                    s4 += gl.amd.cdna3.buffer_load(
                        ptr=slot,
                        offsets=p_off,
                        mask=p0 + u < n,
                        other=0.0,
                        cache=LOAD_CACHE,
                    ).to(gl.float32)
            s = gl.permute(gl.reshape(s4, [NBR // 2, 32, 8, 2, 4]), [0, 3, 1, 2, 4])
            s = gl.convert_layout(gl.reshape(s, [NBR, 32, 32]), L3, assert_trivial=True)
        else:
            p_off = OFF + (g0 * NBR + i) * MFMA_BLOCK_ELEMS + in_blk
            s = gl.zeros((NBR, 32, 32), gl.float32, L3)
            for p0 in range(0, n, UR):
                for u in gl.static_range(UR):
                    slot = p_ptr + (b + p0 + u).to(gl.int64) * SLOT
                    s += gl.amd.cdna3.buffer_load(
                        ptr=slot,
                        offsets=p_off,
                        mask=p0 + u < n,
                        other=0.0,
                        cache=LOAD_CACHE,
                    ).to(gl.float32)
        if PAIR:
            # Two one-block blocks, each where it lies in the tile.
            R1: gl.constexpr = tile_blocks(BM, TM, BN, LN, "r0")[B0 + 1]
            C1: gl.constexpr = tile_blocks(BM, TM, BN, LN, "c0")[B0 + 1]
            col_b = gl.where(ib == 0, C0, C1) + cb
            trow = gl.where(i == 0, R0, R1) + r
            tcol = gl.where(i == 0, C0, C1) + c
        else:
            v0 = g0 * NBR
            col_b = C0 + ((v0 + ib) % VN) * 32 + cb
        if ADD_BIAS:
            if EVEN_N:
                bias = gl.amd.cdna3.buffer_load(
                    ptr=bias_ptr + pid_n * BN, offsets=col_b
                )
            else:
                bias = gl.amd.cdna3.buffer_load(
                    ptr=bias_ptr + pid_n * BN,
                    offsets=col_b,
                    mask=col_b < n_left,
                    other=0.0,
                )
            s = s + bias.to(gl.float32)[:, None, :]
        if PAIR:
            cs = s.to(out_dtype)
        else:
            CB: gl.constexpr = min(NBR, VN)
            ROWS: gl.constexpr = 32 * (NBR // CB)
            COLS: gl.constexpr = 32 * CB
            SL: gl.constexpr = coalesced_store_layout(
                ROWS, COLS, out_dtype.primitive_bitwidth // 8, gl.num_warps()
            )
            cs = gl.convert_layout(to_band(s.to(out_dtype), NBR, VN), SL)
            rr = gl.arange(0, ROWS, gl.SliceLayout(1, SL))[:, None]
            cc = gl.arange(0, COLS, gl.SliceLayout(0, SL))[None, :]
            trow = R0 + (v0 // VN) * 32 + rr
            tcol = C0 + (v0 % VN) * 32 + cc
        # C's row stride is a multiple of 8 (reduce_scatter_slice).
        if PAIR:
            c_rows = gl.multiple_of(trow * stride_cm, [8, 8, 1])
        else:
            c_rows = gl.multiple_of(trow * stride_cm, [8, 1])
        c_off = c_rows + tcol * stride_cn
        if EVEN_M and EVEN_N:
            gl.amd.cdna3.buffer_store(stored_value=cs, ptr=c_tile, offsets=c_off)
        else:
            gl.amd.cdna3.buffer_store(
                stored_value=cs,
                ptr=c_tile,
                offsets=c_off,
                mask=(trow < m_left) & (tcol < n_left),
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
    VQ: gl.constexpr = HALF_M * HALF_N // MFMA_BLOCK_ELEMS
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
    TOP_M: gl.constexpr = 0,
    LEFT_N: gl.constexpr = 0,
):
    """Reader j of `readers` (a power of two) of a tile split n ways: sums blocks
    [NBLK j / readers, NBLK (j + 1) / readers) over the n partials in slots P[b .. b + n
    - 1], adds the bias and stores them to C. NB blocks (NB <= VQ, so a group stays in
    one quadrant) at a time, U partials of a group per trip, so NB U loads per lane are
    in flight. Each group is reshaped to its 2-D band and stored along C's rows, 16
    bytes per lane (8 for a single 32x32 block).

    16-bit partials are read in whole pairs (NB even, readers <= NBLK / 2), NB / 2
    16-byte loads per lane per partial.

    Each reader's NBLK / readers blocks are a whole number of groups
    (reduce_scatter_config), so every group lies inside [blk_lo, blk_hi)."""
    if is_uneven(BLOCK_M, TOP_M, BLOCK_N, LEFT_N):
        reduce_scatter_blocks(
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
            BLOCK_M,
            BLOCK_N,
            TOP_M,
            LEFT_N,
            ADD_BIAS,
            NB,
            U,
            out_dtype,
            LOAD_CACHE,
            m_left,
            n_left,
            EVEN_M,
            EVEN_N,
        )
    else:
        HALF_M: gl.constexpr = BLOCK_M // 2
        HALF_N: gl.constexpr = BLOCK_N // 2
        VN: gl.constexpr = HALF_N // 32
        VQ: gl.constexpr = VN * (HALF_M // 32)
        NBLK: gl.constexpr = 4 * VQ
        P16: gl.constexpr = is_16bit_partial(p_ptr)
        PAIR_Q: gl.constexpr = P16 and VQ == 1
        gl.static_assert(
            NB <= VQ or (PAIR_Q and NB == 2),
            "a reduce-scatter group must stay inside one quadrant",
        )
        gl.static_assert(not P16 or NB % 2 == 0)
        # The band a group covers: PAIR_Q's is tile row p, two quadrants across.
        VNB: gl.constexpr = 2 if PAIR_Q else VN
        SLOT: gl.constexpr = BLOCK_M * BLOCK_N
        blk_lo = (NBLK * j) // readers
        blk_hi = (NBLK * (j + 1)) // readers
        # Rows before columns in lane order: lane l gets row l % 16 and columns 4 (l // 16)
        # .. + 3, which in_blk puts at 4 l .. 4 l + 3, so each 4-lane group reads 64
        # contiguous bytes. Columns first, lanes 0-3 hit 4 lines 256 bytes apart and the L1
        # stalls on tag conflicts (reads take twice as long).
        L3: gl.constexpr = gl.BlockedLayout([1, 1, 4], [1, 16, 4], [1, 2, 2], [1, 2, 0])
        cb = gl.arange(0, 32, gl.SliceLayout(0, gl.SliceLayout(1, L3)))[None, :]
        ib = gl.arange(0, NB, gl.SliceLayout(1, gl.SliceLayout(1, L3)))[:, None]
        if P16:
            # The pairs in the same lane order, a lane's 8 elements the last dimension.
            L4: gl.constexpr = gl.BlockedLayout(
                [1, 1, 1, PAIR_LANE_ELEMS], [1, 16, 4, 1], [1, 2, 2, 1], [3, 1, 2, 0]
            )
            ip = gl.arange(
                0, NB // 2, gl.SliceLayout(1, gl.SliceLayout(2, gl.SliceLayout(3, L4)))
            )
            ip = ip[:, None, None, None]
            in_pair = pair_offsets(NB // 2, L4)
        else:
            i = gl.arange(0, NB, gl.SliceLayout(1, gl.SliceLayout(2, L3)))[
                :, None, None
            ]
            r = gl.arange(0, 32, gl.SliceLayout(0, gl.SliceLayout(2, L3)))[
                None, :, None
            ]
            c = gl.arange(0, 32, gl.SliceLayout(0, gl.SliceLayout(1, L3)))[
                None, None, :
            ]
            # in_block_row(r) + in_block_col(c) (the two add: lane_contiguous_offsets)
            # for rows and columns 0-31; the helpers' form costs instructions here.
            GROUPS: gl.constexpr = MFMA_DIM // MFMA_LANE_ELEMS
            in_blk = (
                (r % MFMA_DIM) * MFMA_LANE_ELEMS
                + (r // MFMA_DIM) * (2 * MFMA_WARP_ELEMS)
                + c
                + (c // MFMA_LANE_ELEMS % GROUPS) * (MFMA_GROUP_ELEMS - MFMA_LANE_ELEMS)
                + (c // MFMA_DIM) * (MFMA_WARP_ELEMS - MFMA_DIM)
            )
        # The band: min(NB, VNB) blocks across, the rest down.
        ROWS: gl.constexpr = 32 * (NB // min(NB, VNB))
        COLS: gl.constexpr = 32 * min(NB, VNB)
        SL: gl.constexpr = coalesced_store_layout(
            ROWS, COLS, out_dtype.primitive_bitwidth // 8, gl.num_warps()
        )
        for g in range(blk_lo, blk_hi, NB):
            if P16:
                valid = g + 2 * ip < blk_hi
                p_off = (g // 2) * PAIR_ELEMS + in_pair
                s4 = gl.zeros(
                    (NB // 2, MFMA_BLOCK_DIM, BLOCK_GROUPS, PAIR_LANE_ELEMS),
                    gl.float32,
                    L4,
                )
                for p0 in range(0, n, U):
                    for u in gl.static_range(U):
                        slot = p_ptr + (b + p0 + u).to(gl.int64) * SLOT
                        s4 += gl.amd.cdna3.buffer_load(
                            ptr=slot,
                            offsets=p_off,
                            mask=valid & (p0 + u < n),
                            other=0.0,
                            cache=LOAD_CACHE,
                        ).to(gl.float32)
                s = gl.permute(
                    gl.reshape(
                        s4, [NB // 2, MFMA_BLOCK_DIM, BLOCK_GROUPS, 2, MFMA_LANE_ELEMS]
                    ),
                    [0, 3, 1, 2, 4],
                )
                s = gl.convert_layout(
                    gl.reshape(s, [NB, MFMA_BLOCK_DIM, MFMA_BLOCK_DIM]),
                    L3,
                    assert_trivial=True,
                )
            else:
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
            if PAIR_Q:
                g_band = g // 2
                blk_b = g_band + 2 * ib
            else:
                g_band = g
                blk_b = g + ib
            if ADD_BIAS:
                # Block q VQ + v: quadrant q (tl, bl, tr, br); register vector v's bits are
                # the column blocks, then the row blocks.
                qb = blk_b // VQ
                vb = blk_b % VQ
                col_b = (qb // 2) * HALF_N + block_col(vb, VN) + cb
                if EVEN_N:
                    bias = gl.amd.cdna3.buffer_load(
                        ptr=bias_ptr + pid_n * BLOCK_N, offsets=col_b
                    )
                else:
                    bias = gl.amd.cdna3.buffer_load(
                        ptr=bias_ptr + pid_n * BLOCK_N,
                        offsets=col_b,
                        mask=col_b < n_left,
                        other=0.0,
                    )
                s = s + bias.to(gl.float32)[:, None, :]
            cs = gl.convert_layout(to_band(s.to(out_dtype), NB, VNB), SL)
            trow, tcol = band_pos(g_band, SL, ROWS, COLS, HALF_M, HALF_N)
            # C's row stride is a multiple of 8 (the wrapper checks), which the launch
            # only proves when it is one of 16: without the hint an N of 8 mod 16
            # stores 2 bytes at a time.
            c_off = gl.multiple_of(trow * stride_cm, [8, 1]) + tcol * stride_cn
            if EVEN_M and EVEN_N:
                gl.amd.cdna3.buffer_store(stored_value=cs, ptr=c_tile, offsets=c_off)
            else:
                gl.amd.cdna3.buffer_store(
                    stored_value=cs,
                    ptr=c_tile,
                    offsets=c_off,
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
    c_offs,
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
    mfmaLayout: gl.constexpr,
    EVEN_M: gl.constexpr,
    EVEN_N: gl.constexpr,
    RS_E: gl.constexpr = 0,
    BAR_STRIDE: gl.constexpr = 0,
    TOP_M: gl.constexpr = 0,
    LEFT_N: gl.constexpr = 0,
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
    is_split = gl.to_tensor(n_tile > 1).to(gl.int32)
    if is_uneven(BLOCK_M, TOP_M, BLOCK_N, LEFT_N):
        if is_split != 0:
            store_partial_blocks(
                accs, p_ptr, rank, (BLOCK_M, TOP_M, BLOCK_N, LEFT_N), mfmaLayout, ".wt"
            )
        if is_split == 0:
            whole_tile_blocks(
                accs,
                pid_n,
                bias_ptr,
                c_bases,
                c_offs,
                m_left,
                n_left,
                BLOCK_M,
                BLOCK_N,
                TOP_M,
                LEFT_N,
                ADD_BIAS,
                out_dtype,
                mfmaLayout,
                EVEN_M,
                EVEN_N,
            )
    else:
        HALF_M: gl.constexpr = BLOCK_M // 2
        HALF_N: gl.constexpr = BLOCK_N // 2
        quads = (accs[0][0], accs[1][0], accs[2][0], accs[3][0])
        p_offsets = partial_map(p_ptr, HALF_M, HALF_N, mfmaLayout)[0]
        z = gl.zeros((HALF_M, HALF_N), gl.float32, mfmaLayout)
        for q in gl.static_range(4):
            if is_split != 0:
                store_partial(quads, q, p_ptr, rank, p_offsets, HALF_M, HALF_N)
            if is_split == 0:
                owner_store(
                    quads[q],
                    z,
                    (q % 2) * HALF_M,
                    (q // 2) * HALF_N,
                    bias_ptr,
                    pid_n,
                    c_bases[q][0],
                    c_offs[q][0],
                    BLOCK_N,
                    ADD_BIAS,
                    out_dtype,
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
                c_bases[0][0] + pid_n * BLOCK_N * stride_cn,
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
                TOP_M,
                LEFT_N,
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
    TAIL_SPLIT_K: gl.constexpr,
    R: gl.constexpr,
    E: gl.constexpr,
    NB: gl.constexpr,
    U: gl.constexpr,
    ADD_BIAS: gl.constexpr,
    EVEN_M: gl.constexpr,
    EVEN_N: gl.constexpr,
    BAR_STRIDE: gl.constexpr,
    TOP_M: gl.constexpr = 0,
    LEFT_N: gl.constexpr = 0,
):
    """An idle program as reader jr (TAIL_SPLIT_K <= jr < R) of Stream-K tile t
    (rs_ending with RS_E > 0). It arrives on the tile's barrier at once: a waiter that
    only polled could read the generation after the last arrival had bumped it and spin
    forever. Then it waits for the producers and reduces its slice."""
    tile_barrier(sync_ptr, t * BAR_STRIDE, TAIL_SPLIT_K + E, True)
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
        t * TAIL_SPLIT_K,
        jr,
        TAIL_SPLIT_K,
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
        TOP_M,
        LEFT_N,
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
    c_offs,
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
    mfmaLayout: gl.constexpr,
    EVEN_M: gl.constexpr,
    EVEN_N: gl.constexpr,
    TOP_M: gl.constexpr = 0,
    LEFT_N: gl.constexpr = 0,
):
    """XCD-local reduce-scatter ending with graph-replay-safe flag rows: rs_ending
    with the partials kept in the XCD's L2 (default-policy stores, .cg loads)."""
    is_split = gl.to_tensor(n_tile > 1).to(gl.int32)
    if is_uneven(BLOCK_M, TOP_M, BLOCK_N, LEFT_N):
        if is_split != 0:
            store_partial_blocks(
                accs, p_ptr, rank, (BLOCK_M, TOP_M, BLOCK_N, LEFT_N), mfmaLayout, ""
            )
        if is_split == 0:
            whole_tile_blocks(
                accs,
                pid_n,
                bias_ptr,
                c_bases,
                c_offs,
                m_left,
                n_left,
                BLOCK_M,
                BLOCK_N,
                TOP_M,
                LEFT_N,
                ADD_BIAS,
                out_dtype,
                mfmaLayout,
                EVEN_M,
                EVEN_N,
            )
    else:
        HALF_M: gl.constexpr = BLOCK_M // 2
        HALF_N: gl.constexpr = BLOCK_N // 2
        quads = (accs[0][0], accs[1][0], accs[2][0], accs[3][0])
        p_offsets = partial_map(p_ptr, HALF_M, HALF_N, mfmaLayout)[0]
        z = gl.zeros((HALF_M, HALF_N), gl.float32, mfmaLayout)
        for q in gl.static_range(4):
            if is_split != 0:
                store_partial(quads, q, p_ptr, rank, p_offsets, HALF_M, HALF_N, "")
            if is_split == 0:
                owner_store(
                    quads[q],
                    z,
                    (q % 2) * HALF_M,
                    (q // 2) * HALF_N,
                    bias_ptr,
                    pid_n,
                    c_bases[q][0],
                    c_offs[q][0],
                    BLOCK_N,
                    ADD_BIAS,
                    out_dtype,
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
                c_bases[0][0] + pid_n * BLOCK_N * stride_cn,
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
                TOP_M,
                LEFT_N,
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
    TOP_M: gl.constexpr,
    LEFT_N: gl.constexpr,
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
    TILE_M: gl.constexpr = 0,
    TILE_N: gl.constexpr = 0,
):
    """K-steps [k0, k0 + 2 n_pairs) of tile tile_id (n_pairs >= 1), from its own
    prologue, then the segment's ending: XCD-local reduce-scatter (l2_rs_ending, L2),
    reduce-scatter (rs_ending) or the owner ending (owner_ending). rank is the
    program's workspace slot, j its chunk (role 0 owns the tile), n_tile / r_tile the
    tile's programs and readers. With TILE_M, TILE_N given, tile_id is a tail unit,
    a BLOCK_M x BLOCK_N (sub-)tile of a TILE_M x TILE_N tile (tail_unit_coords).

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
    a_top_in / b_left_in.

    TOP_M / LEFT_N: the (sub-)tile's first halves (uneven halves; 0: equal)."""
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn = strides
    a_kofs, b_kofs = kofs
    a_dtype: gl.constexpr = a_ptr.dtype.element_ty
    b_dtype: gl.constexpr = b_ptr.dtype.element_ty
    dotOpLayoutA: gl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=mfmaLayout, k_width=8
    )
    dotOpLayoutB: gl.constexpr = gl.DotOperandLayout(
        operand_index=1, parent=mfmaLayout, k_width=8
    )
    if TILE_M == 0:
        pid_m, pid_n = group_m_tile_coords(
            tile_id, M, N, BLOCK_M, BLOCK_N, GROUP_SIZE_M
        )
    else:
        pid_m, pid_n = tail_unit_coords(
            tile_id, M, N, TILE_M, TILE_N, BLOCK_M, BLOCK_N, GROUP_SIZE_M
        )
    m_left = M - pid_m * BLOCK_M
    n_left = gl.multiple_of(N - pid_n * BLOCK_N, 8)

    a_addr = a_load_addressing(
        pid_m, a_kofs, M, stride_am, stride_ak, BLOCK_M, BLOCK_K, EVEN_M, a_dtype, TOP_M
    )
    b_addr = b_load_addressing(
        pid_n,
        b_kofs,
        N,
        stride_bk,
        stride_bn,
        BLOCK_N,
        BLOCK_K,
        EVEN_N,
        b_dtype,
        LEFT_N,
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
                k_hi,
                K_RES,
                BLOCK_M,
                BLOCK_N,
                BLOCK_K,
                a_dtype,
                b_dtype,
                False,
                TOP_M,
                LEFT_N,
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
    accs = init_acc(bias_ptr, pid_n, BLOCK_M, BLOCK_N, mfmaLayout, False, TOP_M, LEFT_N)
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
    c_bases, c_offs, _ = c_store_addressing(
        c_ptr,
        pid_m,
        pid_n,
        m_left,
        n_left,
        stride_cm,
        stride_cn,
        BLOCK_M,
        BLOCK_N,
        TOP_M,
        LEFT_N,
        SEG_LOOP,
        False,
        EVEN_M,
        EVEN_N,
    )

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
            c_offs,
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
            mfmaLayout,
            EVEN_M,
            EVEN_N,
            TOP_M,
            LEFT_N,
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
            c_offs,
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
            mfmaLayout,
            EVEN_M,
            EVEN_N,
            RS_E,
            BAR_STRIDE,
            TOP_M,
            LEFT_N,
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
            c_offs,
            m_left,
            n_left,
            BLOCK_N,
            PEER_STEP,
            ADD_BIAS,
            out_dtype,
            mfmaLayout,
            EVEN_M,
            EVEN_N,
            OWNER_QPB,
            OWNER_MAX_PEERS,
            BLOCK_M,
            TOP_M,
            LEFT_N,
        )


MFMA_SCHEDULE_HINT = "mfma-schedule"


def schedule_hint():
    """The schedule_hint the kernel launches with: the MFMA scheduler, unless
    AITER_MFMA_SCHED=0 (an unscheduled A/B run)."""
    return "" if os.environ.get("AITER_MFMA_SCHED", "1") == "0" else MFMA_SCHEDULE_HINT


# ---------------------------------------------------------------------------------
# Host side: configuration, topology and plan (no tensors; the launch is the wrapper's)
# ---------------------------------------------------------------------------------

# Tile sides: any multiple of 32 from 64 to 320 (uneven_first splits each into two
# valid halves), with at most MAX_ACC accumulators (256 fp32 AGPRs per lane).
TILE_SIDES = tuple(range(64, 321, 32))
MAX_ACC = 256 * 256
BLOCK_K_SIDES = (64, 128)
STREAMK_POLICIES = ("dp", "dp_sub", "one_tile", "two_tile", "spread")
STREAMK_FIXUPS = ("owner", "rs")
STREAMK_EXCHANGES = ("wt", "l2")
STREAMK_PARTIALS = ("fp32", "16bit")
# Policies that store partials, so take STREAMK_PARTIALS "16bit".
PARTIAL_POLICIES = ("one_tile", "two_tile", "spread")
CONFIG_KEYS = {
    "BLOCK_M": "block_m",
    "BLOCK_N": "block_n",
    "BLOCK_K": "block_k",
    "GROUP_SIZE_M": "group_size_m",
    "STREAMK_POLICY": "policy",
    "STREAMK_FIXUP": "fixup",
    "TAIL_SPLIT_K": "tail_split_k",
    "STREAMK_EXCHANGE": "exchange",
    "STREAMK_PARTIALS": "partials",
}
# Keys a complete config may leave out (StreamKConfig's defaults).
OPTIONAL_KEYS = ("STREAMK_PARTIALS",)
MAX_TILE_ELEMS = MAX_ACC


def _valid_half(rows):
    """A half of `rows` rows: a power of two, plus nothing or a power of two of at
    least 32."""
    extra = rows - (1 << (rows.bit_length() - 1))
    return extra == 0 or (extra >= 32 and extra & (extra - 1) == 0)


def uneven_first(side):
    """The first half (top or left) of a tile side, the kernel's TOP_M / LEFT_N: 0
    (side / 2) for a power of two, else the side's 32-wide pieces split as evenly as
    two valid halves allow, the larger half first (224 = 128 + 96, 160 = 96 + 64)."""
    if side & (side - 1) == 0:
        return 0
    pieces = side // 32
    for big in range((pieces + 1) // 2, pieces):
        if _valid_half(32 * big) and _valid_half(32 * (pieces - big)):
            return 32 * big
    raise ValueError(
        f"side {side}: no split into two halves of a power of two + 32 or 64"
    )


@dataclass(frozen=True)
class TileSpec:
    """One supported tile: sides in TILE_SIDES with at most MAX_ACC accumulators,
    BLOCK_K in BLOCK_K_SIDES (128 only when BLOCK_M + BLOCK_N <= 256, so the
    double-buffered A and B fit in LDS)."""

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
        if self.block_m * self.block_n > MAX_ACC:
            raise ValueError(
                f"tile {self.block_m}x{self.block_n} has more than {MAX_ACC} "
                "accumulators"
            )
        if self.block_k not in BLOCK_K_SIDES:
            raise ValueError(f"BLOCK_K must be in {BLOCK_K_SIDES}, got {self.block_k}")
        if self.block_k == 128 and self.block_m + self.block_n > 256:
            raise ValueError(
                f"BLOCK_K 128 needs BLOCK_M + BLOCK_N <= 256 (LDS), got "
                f"{self.block_m}x{self.block_n}"
            )

    @property
    def top_m(self):
        """The kernel's TOP_M (uneven_first)."""
        return uneven_first(self.block_m)

    @property
    def left_n(self):
        return uneven_first(self.block_n)

    @property
    def uneven(self):
        """True unless both sides are powers of two (two equal halves)."""
        return is_uneven(self.block_m, self.top_m, self.block_n, self.left_n)

    @property
    def blocks(self):
        """32x32 reduce-scatter blocks in the tile."""
        return self.block_m * self.block_n // (32 * 32)

    @property
    def quadrant_blocks(self):
        """The most 32x32 blocks a reduce-scatter group can hold: a quadrant's, or an
        uneven tile's largest block's (tile_blocks)."""
        if not self.uneven:
            return self.blocks // 4
        return max(
            tile_blocks(self.block_m, self.top_m, self.block_n, self.left_n, "vq")
        )


def menu_tiles():
    """Every supported (BLOCK_M, BLOCK_N, BLOCK_K) (TileSpec)."""
    out = []
    for bm in TILE_SIDES:
        for bn in TILE_SIDES:
            for bk in BLOCK_K_SIDES:
                try:
                    TileSpec(bm, bn, bk)
                except ValueError:
                    continue
                out.append((bm, bn, bk))
    return out


def tuning_tiles(spec):
    """The tiles tuning_candidates(tiles=) takes for a tuner's tile option: None for
    None or "default" (TUNING_TILES), menu_tiles() for "menu", or a comma-separated
    list of BLOCK_MxBLOCK_NxBLOCK_K, each checked by TileSpec (ValueError)."""
    if spec is None or spec == "default":
        return None
    if spec == "menu":
        return menu_tiles()
    tiles = []
    for item in spec.split(","):
        sides = item.strip().lower().split("x")
        if len(sides) != 3 or not all(s.isdigit() for s in sides):
            raise ValueError(f"a tile is BLOCK_MxBLOCK_NxBLOCK_K, got {item!r}")
        TileSpec(*map(int, sides))
        tiles.append(tuple(map(int, sides)))
    return tiles


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
    tail_split_k: int | None
    exchange: str
    partials: str = "fp32"

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
            if type(self.tail_split_k) is not int or self.tail_split_k < 2:
                raise ValueError(
                    "one_tile requires an integer TAIL_SPLIT_K of at least 2"
                )
        elif self.tail_split_k is not None:
            raise ValueError(f"{self.policy} takes TAIL_SPLIT_K=None")
        if self.policy != "one_tile" and self.fixup != "owner":
            raise ValueError(f"{self.policy} requires the owner fixup")
        if self.exchange == "l2" and (self.policy, self.fixup) != ("one_tile", "rs"):
            raise ValueError("the l2 exchange requires one_tile with rs")
        if self.partials not in STREAMK_PARTIALS:
            raise ValueError(f"invalid STREAMK_PARTIALS {self.partials!r}")
        if self.partials == "16bit" and self.policy not in PARTIAL_POLICIES:
            raise ValueError(f"{self.policy} stores no partials; it takes fp32")

    @property
    def tile(self):
        return TileSpec(self.block_m, self.block_n, self.block_k)

    @classmethod
    def from_dict(cls, values):
        """Parse a config dict with the CONFIG_KEYS (OPTIONAL_KEYS may be absent)."""
        if isinstance(values, cls):
            return values
        if not isinstance(values, Mapping):
            raise TypeError("the Stream-K config must be a mapping")
        unknown = sorted(set(values) - set(CONFIG_KEYS))
        if unknown:
            raise ValueError(f"unknown Stream-K config keys {unknown}")
        missing = sorted(set(CONFIG_KEYS) - set(values) - set(OPTIONAL_KEYS))
        if missing:
            raise ValueError(f"incomplete Stream-K config; missing {missing}")
        return cls(**{CONFIG_KEYS[key]: value for key, value in values.items()})


@dataclass(frozen=True)
class ReduceScatterConfig:
    readers: int
    blocks_per_group: int
    partials_per_trip: int
    # Idle programs per tile that read too, and the tile barrier's slot stride
    # (0: slot t tail_split_k).
    extra: int = 0
    barrier_stride: int = 0


@constexpr_function
def pow2_floor(x):
    return 1 << (x.bit_length() - 1)


def max_readers(tile, p16=False):
    """The most reduce-scatter readers of a tile, a power of two: up to one per 32x32
    block, or per pair of blocks with 16-bit partials (a reader reads whole pairs)."""
    return pow2_floor(tile.blocks // 2 if p16 else tile.blocks)


def reduce_scatter_config(
    tile, tail_split_k, loads_in_flight=16, readers=None, p16=False
):
    """The tile's own power-of-two readers (at most max_readers), or ``readers``, and
    the slice's blocks per group and partials per trip, so that up to loads_in_flight
    16-byte loads per lane are in flight: a block of fp32 is one, a pair of 16-bit
    blocks one. A trip takes at most the tile's tail_split_k partials, as more would
    only be masked-off loads."""
    if readers is None:
        readers = min(pow2_floor(tail_split_k), max_readers(tile, p16))
    nb = pow2_floor(
        max(1, min(tile.blocks // readers, loads_in_flight, tile.quadrant_blocks))
    )
    if p16:
        nb = max(nb, 2)
    # reduce_scatter_slice's groups never cross a reader's slice (reduce_scatter_blocks
    # masks an uneven tile's).
    assert tile.uneven or (tile.blocks // readers) % nb == 0, (tile, readers, nb)
    loads_per_partial = nb // 2 if p16 else nb
    return ReduceScatterConfig(
        readers=readers,
        blocks_per_group=nb,
        partials_per_trip=min(
            max(loads_in_flight // loads_per_partial, 1), tail_split_k
        ),
    )


def rs_readers(
    tile, tail_tiles, tail_split_k, num_programs, loads_in_flight=16, p16=False
):
    """The one-tile write-through reduce-scatter with every resident program reading:
    R = max(min(blocks, pow2_floor(P / S)), own readers) readers per tile, so
    R S <= P; when R > tail_split_k the extra = R - tail_split_k idle programs per tile
    join. Each tile's barrier gets its own 128-byte line (16 slots of 2 words) where it
    fits: packed, the idle readers' polling slows the producers' arrivals on the
    shared line. With extra = 0 this is the own-readers rule with only the barrier
    slots moved."""
    own = reduce_scatter_config(tile, tail_split_k, loads_in_flight, p16=p16).readers
    per_tile = num_programs // tail_tiles
    readers = max(min(max_readers(tile, p16), pow2_floor(per_tile)), own)
    rs = reduce_scatter_config(tile, tail_split_k, loads_in_flight, readers, p16)
    return ReduceScatterConfig(
        readers=readers,
        blocks_per_group=rs.blocks_per_group,
        partials_per_trip=rs.partials_per_trip,
        extra=readers - min(tail_split_k, readers),
        barrier_stride=max(tail_split_k, min(16, per_tile)),
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
PLAN_KEYS = ("STREAMK_POLICY", "STREAMK_FIXUP", "TAIL_SPLIT_K", "STREAMK_EXCHANGE")


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
    SUBTILE_MAX_K  the longest K at which the default plan takes dp_sub
    OWNER_SPLITS,  the TAIL_SPLIT_K values the default plan rounds to and the tuner
    RS_SPLITS      times, for the owner and reduce-scatter endings
    """
    cfg_dir = resolve_config_dir("gemm", "GEMM-A16W16-STREAMK", backend="gluon")
    return load_config_json(f"{cfg_dir}/DEFAULT.json")


def _nearest_split(n, splits, cap):
    ok = [s for s in splits if s <= cap]
    return min(ok, key=lambda s: (abs(math.log(min(n, cap) / s)), s)) if ok else None


def subtile_tail(M, N, tile, num_programs):
    """dp_sub's (TAIL_BLOCK_M, TAIL_BLOCK_N) for a shape at a tile (BLOCK_M, BLOCK_N,
    ...), or None when it has none (no full wave, no leftover, or S > P / 2): the
    largest Q with S Q <= P, then the most square sub-tile, then the least padding,
    then the smaller TAIL_BLOCK_M."""
    bm, bn = tile[0], tile[1]
    total_tiles = triton.cdiv(M, bm) * triton.cdiv(N, bn)
    leftover = total_tiles % num_programs
    if total_tiles < num_programs or leftover == 0:
        return None
    # Any menu side that divides the tile's: for a power-of-two tile only 64, 128
    # and 256 do.
    subtiles = [
        (tm, tn)
        for tm in TILE_SIDES
        for tn in TILE_SIDES
        if bm % tm == 0 and bn % tn == 0 and (tm, tn) != (bm, bn)
    ]
    fits = [
        s for s in subtiles if leftover * (bm // s[0]) * (bn // s[1]) <= num_programs
    ]
    if not fits:
        return None
    return min(
        fits,
        key=lambda s: (
            # The most sub-tiles per tail tile (fits keeps S Q <= P).
            -(bm // s[0]) * (bn // s[1]),
            # The most square sub-tile.
            abs(math.log2(s[0] / s[1])),
            # The least padded area of C, the work in sub-tiles past its edge.
            triton.cdiv(M, s[0]) * s[0] * triton.cdiv(N, s[1]) * s[1],
            # The smaller TAIL_BLOCK_M.
            s[0],
        ),
    )


def streamk_plan(M, N, K, tile, num_programs, plan=None):
    """(STREAMK_POLICY, STREAMK_FIXUP, TAIL_SPLIT_K) of the default plan for a shape at
    a tile (BLOCK_M, BLOCK_N, BLOCK_K); see the module docstring. ``plan`` overrides
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
    if K <= defaults["SUBTILE_MAX_K"] and subtile_tail(M, N, tile, num_programs):
        return "dp_sub", "owner", None
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
    the plan keys are all given (and kept) or all absent. STREAMK_PARTIALS "16bit"
    with the default plan applies only when that plan stores partials."""
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
        policy, fixup, tail_split_k = streamk_plan(M, N, K, tile, num_programs)
        full.update(
            STREAMK_POLICY=policy,
            STREAMK_FIXUP=fixup,
            TAIL_SPLIT_K=tail_split_k,
            STREAMK_EXCHANGE="wt",
        )
        if policy not in PARTIAL_POLICIES and "STREAMK_PARTIALS" in full:
            full["STREAMK_PARTIALS"] = "fp32"
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

    tail = (0, 0)
    if config.policy == "dp_sub":
        tail = subtile_tail(M, N, (tile.block_m, tile.block_n), num_programs)
        if tail is None:
            raise ValueError(
                "dp_sub requires a full wave and at most half a wave of leftover tiles"
            )
    tail_tiles = 0 if config.policy == "dp" else leftover
    if config.policy == "two_tile":
        if leftover == 0 or total_tiles < num_programs:
            raise ValueError(
                "two_tile requires a partial wave after at least one full wave"
            )
        tail_tiles += num_programs
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
        tail_split_k = config.tail_split_k
        tail_split_cap = min(num_programs // tail_tiles, pairs)
        if tail_split_cap < 2:
            raise ValueError(
                "shape leaves fewer than two programs/pairs per Stream-K tile"
            )
        if tail_split_k > tail_split_cap:
            raise ValueError(
                f"tail_split_k={tail_split_k} exceeds this shape's maximum "
                f"{tail_split_cap}"
            )
        streamk_programs = tail_tiles * tail_split_k
        if l2:
            per_xcd_tiles = triton.cdiv(tail_tiles, num_xcds)
            xcd_max = max(
                1,
                min(
                    tail_split_k,
                    pairs,
                    (num_programs // num_xcds) // per_xcd_tiles,
                ),
            )
            if xcd_max != tail_split_k:
                raise ValueError(
                    f"tail_split_k={tail_split_k} cannot be placed within one XCD; "
                    f"maximum is {xcd_max}"
                )
    elif end_to_end:
        tail_split_k = 1
        streamk_programs = num_programs
        if tail_tiles * pairs < streamk_programs:
            raise ValueError(
                f"{config.policy} has fewer pair units than "
                f"the {num_programs} programs"
            )
    else:
        # dp_sub's sub-tiles (S Q <= P, subtile_tail) store no partials.
        tail_split_k = 1
        streamk_programs = 0

    p16 = config.partials == "16bit"
    if config.policy == "one_tile" and config.fixup == "rs" and not l2:
        rs = rs_readers(tile, tail_tiles, tail_split_k, num_programs, p16=p16)
    else:
        rs = reduce_scatter_config(
            tile, tail_split_k if config.policy == "one_tile" else 1, p16=p16
        )
    # The L2 split gives every XCD ceil(S / NUM_XCDS) tiles' programs (pid p runs
    # on XCD p % NUM_XCDS), so its grid is a whole number of those per XCD. The wt
    # reduce-scatter also launches its S extra idle readers per tile.
    working_grid = (
        num_xcds * per_xcd_tiles * tail_split_k
        if per_xcd_tiles
        else streamk_programs + tail_tiles * rs.extra
    )
    compact = (
        tail_tiles > 0 and tail_tiles == total_tiles and working_grid < num_programs
    )
    # Data-parallel launches have no Stream-K phase to keep resident, so a grid of
    # fewer tiles than programs launches only the programs with a tile.
    if compact:
        grid = working_grid
    elif tail_tiles == 0:
        grid = min(num_programs, total_tiles)
    else:
        grid = num_programs
    return config, {
        "TOTAL_TILES": total_tiles,
        "TAIL_TILES": tail_tiles,
        "TOP_M": tile.top_m,
        "LEFT_N": tile.left_n,
        "TAIL_BLOCK_M": tail[0],
        "TAIL_BLOCK_N": tail[1],
        "TAIL_TOP_M": uneven_first(tail[0]) if tail[0] else 0,
        "TAIL_LEFT_N": uneven_first(tail[1]) if tail[1] else 0,
        "STREAMK_PROGRAMS": streamk_programs,
        "RESOLVED_TAIL_SPLIT_K": tail_split_k,
        "END_TO_END": end_to_end,
        "L2": l2,
        "COMPACT": compact,
        "GRID": grid,
        "RS_NB": rs.blocks_per_group,
        "RS_U": rs.partials_per_trip,
        "RS_R": rs.readers,
        "RS_E": rs.extra,
        "BAR_STRIDE": rs.barrier_stride,
        "PARTIALS_16BIT": p16,
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
    return {key: config[key] for key in TILE_KEYS + OPTIONAL_KEYS if key in config}


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
    """Every (STREAMK_POLICY, STREAMK_FIXUP, TAIL_SPLIT_K, STREAMK_EXCHANGE) worth
    timing; which of them fit a shape is left to _launch_constants."""
    yield "dp", "owner", None, "wt"
    yield "dp_sub", "owner", None, "wt"
    yield "two_tile", "owner", None, "wt"
    yield "spread", "owner", None, "wt"
    defaults = streamk_defaults()
    for n in defaults["OWNER_SPLITS"]:
        yield "one_tile", "owner", n, "wt"
    for n in defaults["RS_SPLITS"]:
        yield "one_tile", "rs", n, "wt"
        yield "one_tile", "rs", n, "l2"


def tuning_candidates(M, N, K, partials_16bit=False, tiles=None):
    """The complete configs the tuner times for a shape: every TUNING_TILES tile at
    the default GROUP_SIZE_M (streamk_defaults) with every plan that fits it (and the default
    plan's pick), one per distinct launch; with ``partials_16bit``, also each plan
    that stores partials with STREAMK_PARTIALS "16bit". ``tiles`` replaces
    TUNING_TILES (any supported (BLOCK_M, BLOCK_N, BLOCK_K), e.g. menu_tiles()). M, N
    and K may be numpy integers (a tuner's pandas rows)."""
    M, N, K = int(M), int(N), int(K)
    num_programs, num_xcds = streamk_topology()
    candidates, seen = [], set()
    for bm, bn, bk in TUNING_TILES if tiles is None else tiles:
        tile = {
            "BLOCK_M": bm,
            "BLOCK_N": bn,
            "BLOCK_K": bk,
            "GROUP_SIZE_M": streamk_defaults()["GROUP_SIZE_M"],
        }
        configs = [complete_config(tile, M, N, K, num_programs)]
        configs += [dict(tile, **dict(zip(PLAN_KEYS, plan))) for plan in _plans()]
        if partials_16bit:
            configs += [
                dict(c, STREAMK_PARTIALS="16bit")
                for c in configs
                if c["STREAMK_POLICY"] in PARTIAL_POLICIES
            ]
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
    r"(?:(?P<whole>dp|dp_sub|two_tile|spread)"
    r"|one_tile_(?P<fixup>owner|rs)_k(?P<k>\d+)_(?P<exchange>wt|l2))"
    r"(?P<p16>_p16)?$"
)


def kernel_name(config):
    """The kernelName of a tuned row running a complete ``config``, e.g.
    gluon_streamk_256x256x64_g4_one_tile_rs_k8_wt or gluon_streamk_256x256x64_g4_dp
    (policies other than one_tile take the owner fixup and the wt exchange); 16-bit
    partials add _p16."""
    c = StreamKConfig.from_dict(config)
    name = f"{KERNEL_NAME_PREFIX}{c.block_m}x{c.block_n}x{c.block_k}_g{c.group_size_m}_"
    if c.policy != "one_tile":
        name += c.policy
    else:
        name += f"one_tile_{c.fixup}_k{c.tail_split_k}_{c.exchange}"
    return name + ("_p16" if c.partials == "16bit" else "")


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
        plan = ("one_tile", m["fixup"], int(m["k"]), m["exchange"])
    config.update(zip(PLAN_KEYS, plan))
    if m["p16"]:
        config["STREAMK_PARTIALS"] = "16bit"
    return config


_KERNEL_MAP = {"streamk": _gemm_a16w16_streamk_kernel}
