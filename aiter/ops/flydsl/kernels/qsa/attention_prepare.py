# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Exact QSA attention selection, private union scratch and fused preparation launches.

These three FlyDSL kernels (K1-K3) are the private preparation step of the
FlyDSL QSA sparse attention; they are not a public op.

Recovery does not validate the selection layout. Dense membership is zero when a plan is
allocated and after every completed build. Recovery scatters into it; its owning compact CTA
consumes and clears the same causal prefix. The next launch sorts tasks and builds disjoint
mask partitions. Plans use one stream's scratch and are pinned for graphs.
"""

import math
import operator
from dataclasses import dataclass
from functools import partial

import flydsl.compiler as flyc
import flydsl.expr as fx
import numpy as np
import torch
from flydsl.expr import rocdl

from aiter.jit.utils.chip_info import get_cu_num

from .attention_common import _buffer, _buffer_words, _min, _uniform, platform

THREADS = 256
INT_MAX = 2**31 - 1
LIMITS = 8  # union step limits the route compares: max * 2**(-j/2) for j < 7, then none
# K3 holds every task-sort width (keys per thread) and picks one per launch, so task counts
# never compile a new variant.
SORT_WIDTHS = (1, 2, 4, 8, 16)
EXCHANGE_WORDS = 1024  # K3 LDS: one cross-wave sort exchange round, then route partials
ORDER_WORDS = EXCHANGE_WORDS + 4 * (2 + 4 * LIMITS)


@dataclass(kw_only=True)
class SparsePlan:
    metadata: torch.Tensor
    dense_membership: torch.Tensor
    blocks: torch.Tensor
    membership: torch.Tensor
    score_masks: torch.Tensor
    counts: torch.Tensor
    costs: torch.Tensor
    active: torch.Tensor
    task_order: torch.Tensor
    query_tiles: torch.Tensor
    direct_flag: torch.Tensor
    query_tile: int
    group_padded: int
    block_capacity: int
    max_blocks: int
    num_tiles: int
    grid: int
    route_costs: tuple
    route_model: tuple  # platform union_fixed, union_step, row_step, route_margin
    union_ratio: float
    routing: bool = True


def _ballot(pred):
    return fx.Int64(rocdl.ballot(fx.Int64.ir_type, pred))


def _lanes_below(mask, lane):
    return fx.Int32(fx.ctpop(mask & ((fx.Int64(1) << fx.Int64(lane)) - 1)))


# gfx9 DPP controls: quad_perm xor 1 / 2 / 3, row_half_mirror (xor 7), row_mirror (xor 15).
_DPP_XOR = {1: (0xB1,), 2: (0x4E,), 4: (0x141, 0x1B), 8: (0x140, 0x141)}


def _lane_xor(value, mask):
    """The value of lane ^ mask; every lane of the wave must be active."""
    if isinstance(value, fx.Int64):
        halves = fx.Vector.from_elements([value], fx.Int64).bitcast(fx.Int32)
        return fx.Vector.from_elements(
            [_lane_xor(halves[0], mask), _lane_xor(halves[1], mask)], fx.Int32
        ).bitcast(fx.Int64)[0]
    if mask not in _DPP_XOR:
        return value.shuffle_xor(mask, 64)
    # Raw DPP (no fx wrapper): row-local moves instead of LDS-crossbar shuffles cut the
    # latency-bound K1 by 10-14% on small calls.
    raw = value.ir_value()
    for control in _DPP_XOR[mask]:
        raw = rocdl.update_dpp(fx.Int32.ir_type, raw, raw, control, 0xF, 0xF, True)
    return fx.Int32(raw)


def _wave_reduce(value, op):
    for mask in (32, 16, 8, 4, 2, 1):
        value = op(value, _lane_xor(value, mask))
    return value


def _exchange(keys, scratch, tid, partner, wide):
    """The keys of thread partner, swapped through LDS a round at a time."""
    words = 2 if wide else 1
    chunk = min(len(keys), EXCHANGE_WORDS // (THREADS * words))
    peers = []
    for first in range(0, len(keys), chunk):
        for e in range(first, first + chunk):
            slot = (tid * chunk + e - first) * words
            if wide:
                pair = fx.Vector.from_elements([keys[e]], fx.Int64).bitcast(fx.Int32)
                scratch[slot] = pair[0]
                scratch[slot + 1] = pair[1]
            else:
                scratch[slot] = keys[e]
        fx.gpu.barrier()
        for e in range(first, first + chunk):
            slot = (partner * chunk + e - first) * words
            if wide:
                pair = [scratch[slot], scratch[slot + 1]]
                pair = fx.Vector.from_elements(pair, fx.Int32).bitcast(fx.Int64)
                peers.append(pair[0])
            else:
                peers.append(scratch[slot])
        fx.gpu.barrier()
    return peers


def _bitonic(keys, lane, descending, cta=None):
    """Bitonic sort of one wave's keys, or with cta = (scratch, tid, wave, wide) of the CTA's.

    Thread t holds the keys t * w .. t * w + w - 1, w = len(keys) a power of two.
    """
    keys = list(keys)
    width = len(keys)
    log_width = width.bit_length() - 1
    log_size = log_width + (8 if cta else 6)

    def index_bit(bit):
        if bit < log_width + 6:
            return ((lane >> (bit - log_width)) & 1) != 0
        return ((cta[2] >> (bit - log_width - 6)) & 1) != 0

    for stage in range(1, log_size + 1):
        # Bit `stage` of the key index reverses its run's order; the last merge has none.
        flip = index_bit(stage) if log_width <= stage < log_size else None
        for bit in range(stage - 1, -1, -1):
            if bit < log_width:
                step = 1 << bit
                for e in range(width):
                    if e & step:
                        continue
                    first = fx.min(keys[e], keys[e | step])
                    second = fx.max(keys[e], keys[e | step])
                    if descending != (stage < log_width and ((e >> stage) & 1) == 1):
                        first, second = second, first
                    if flip is None:
                        keys[e], keys[e | step] = first, second
                    else:
                        keys[e] = flip.select(second, first)
                        keys[e | step] = flip.select(first, second)
                continue
            upper = index_bit(bit) if flip is None else index_bit(bit) ^ flip
            if bit < log_width + 6:
                peers = [_lane_xor(key, 1 << (bit - log_width)) for key in keys]
            else:
                scratch, tid, _, wide = cta
                partner = tid ^ (64 << (bit - log_width - 6))
                peers = _exchange(keys, scratch, tid, partner, wide)
            for e, peer in enumerate(peers):
                low, high = fx.min(keys[e], peer), fx.max(keys[e], peer)
                if descending:
                    low, high = high, low
                # The upper key of a pair takes the high one (the low one descending).
                keys[e] = upper.select(high, low)
    return keys


def _sort_columns(topk):
    """K1 lanes hold topk / 64 columns; the sort pads them to a power of two."""
    return 1 << (topk // 64 - 1).bit_length()


@flyc.jit
def _recover_row(
    INDICES: fx.Tensor,
    POSITIONS: fx.Tensor,
    BLOCKS: fx.Tensor,
    DENSE: fx.Tensor,
    QUERY_TILES: fx.Tensor,
    META: fx.Tensor,
    ROWS: fx.Int32,
    MAX_BLOCKS: fx.Int32,
    row,
    lane,
    scratch,
    slot,
    MEMBER_BITS: fx.Constexpr[int],
    TOPK: fx.Constexpr[int],
):
    # An indices row holds TOPK four-token blocks, then 3 tail tokens.
    row_bytes = (4 * TOPK + 3) * 4
    columns, width = TOPK // 64, _sort_columns(TOPK)
    visible = _uniform(POSITIONS[row]) + 1
    complete = _min(visible >> 2, fx.Int32(TOPK))
    indices = _buffer(INDICES, ROWS * row_bytes)
    unsorted = fx.Boolean(False)
    # Lane holds columns 64 * k + lane: each load, store and scatter spans 64 adjacent ones.
    keys = []
    for k in fx.range_constexpr(columns):
        column = k * 64 + lane
        full = column < complete
        block = _buffer_words(indices, row * row_bytes + column * 16, 1)[0] >> 2
        unsorted = unsorted | (full & (block != column))
        keys.append(full.select(block, fx.Int32(INT_MAX)))
    # Rows that select their first blocks in order (complete causal prefixes) skip the sort.
    if _ballot(unsorted) != 0:
        # The sort holds keys width * lane .. width * lane + width - 1: transpose through the
        # wave's LDS, INT_MAX past the last column.
        padded = keys + [fx.Int32(INT_MAX)] * (width - columns)
        for k in fx.range_constexpr(width):
            scratch[slot + k * 64 + lane] = padded[k]
        ordered = [scratch[slot + lane * width + r] for r in range(width)]
        ordered = _bitonic(ordered, lane, False)
        for r in fx.range_constexpr(width):
            scratch[slot + lane * width + r] = ordered[r]
        keys = [scratch[slot + k * 64 + lane] for k in range(columns)]
    for k in fx.range_constexpr(columns):
        column = k * 64 + lane
        BLOCKS[row * TOPK + column] = (column < complete).select(keys[k], fx.Int32(-1))
    if fx.const_expr(MEMBER_BITS != 0):
        tile = _uniform(QUERY_TILES[row])
        if tile >= 0:
            member = fx.Int64 if MEMBER_BITS == 64 else fx.Int32
            bit = member(1) << member(row - _uniform(META[tile * 5]))
            dense = fx.get_iter(DENSE) + fx.Int64(tile) * fx.Int64(MAX_BLOCKS)
            for k in fx.range_constexpr(columns):
                block = keys[k]
                if (k * 64 + lane < complete) & (block >= 0) & (block < MAX_BLOCKS):
                    fx.atomic_or(dense + block, bit, syncscope="agent")
            if (lane == 0) & ((visible & 3) != 0):
                fx.atomic_or(dense + (visible >> 2), bit, syncscope="agent")


@flyc.kernel(name="qsa_recover_scatter", known_block_size=[THREADS, 1, 1])
def _recover_kernel(
    INDICES: fx.Tensor,
    POSITIONS: fx.Tensor,
    BLOCKS: fx.Tensor,
    DENSE: fx.Tensor,
    QUERY_TILES: fx.Tensor,
    META: fx.Tensor,
    ROWS: fx.Int32,
    MAX_BLOCKS: fx.Int32,
    MEMBER_BITS: fx.Constexpr[int],
    TOPK: fx.Constexpr[int],
):
    words = 64 * _sort_columns(TOPK)  # one wave's sort transpose
    scratch = (
        fx.SharedAllocator()
        .allocate(fx.Array[fx.Int32, 4 * words, 16])
        .peek()
        .view(fx.make_layout(4 * words, 1))
    )
    tid = fx.thread_idx.x
    wave = _uniform(tid >> 6)
    # One wave per row, last rows first: in a causal request they reach the most blocks.
    row = ROWS - 1 - (fx.block_idx.x * 4 + wave)
    if row >= 0:
        _recover_row(
            INDICES,
            POSITIONS,
            BLOCKS,
            DENSE,
            QUERY_TILES,
            META,
            ROWS,
            MAX_BLOCKS,
            row,
            tid & 63,
            scratch,
            wave * words,
            MEMBER_BITS,
            TOPK,
        )


@flyc.jit
def _recover_launch(
    INDICES: fx.Tensor,
    POSITIONS: fx.Tensor,
    BLOCKS: fx.Tensor,
    DENSE: fx.Tensor,
    QUERY_TILES: fx.Tensor,
    META: fx.Tensor,
    ROWS: fx.Int32,
    MAX_BLOCKS: fx.Int32,
    MEMBER_BITS: fx.Constexpr[int],
    TOPK: fx.Constexpr[int],
    stream: fx.Stream,
):
    if ROWS > 0:
        _recover_kernel(
            INDICES,
            POSITIONS,
            BLOCKS,
            DENSE,
            QUERY_TILES,
            META,
            ROWS,
            MAX_BLOCKS,
            MEMBER_BITS,
            TOPK,
        ).launch(grid=((ROWS + 3) // 4, 1, 1), block=(THREADS, 1, 1), stream=stream)


@flyc.kernel(name="qsa_compact", known_block_size=[THREADS, 1, 1])
def _compact_kernel(
    DENSE: fx.Tensor,
    BLOCKS: fx.Tensor,
    MEMBERSHIP: fx.Tensor,
    COUNTS: fx.Tensor,
    COSTS: fx.Tensor,
    META: fx.Tensor,
    DIRECT_FLAG: fx.Tensor,
    MAX_BLOCKS: fx.Int32,
    CAPACITY: fx.Int32,
    RATIO: fx.Float32,
    TILES: fx.Int32,
    MEMBERS64: fx.Constexpr[bool],
    TOPK: fx.Constexpr[int],
):
    totals = (
        fx.SharedAllocator()
        .allocate(fx.Array[fx.Int32, 32, 16])
        .peek()
        .view(fx.make_layout(32, 1))
    )
    # Membership words hold one bit per query row (64-bit for BQ > 32).
    member, member_bytes = (fx.Int64, 8) if MEMBERS64 else (fx.Int32, 4)
    tid = fx.thread_idx.x
    lane, wave = tid & 63, _uniform(tid >> 6)
    if (fx.block_idx.x == 0) & (tid == 0):
        DIRECT_FLAG[0] = 0
    # Last tile first: in a causal request it holds the most blocks.
    tile = TILES - 1 - fx.block_idx.x
    rows = _uniform(META[tile * 5 + 1])
    position = _uniform(META[tile * 5 + 4])
    # Union pads blocks to N64 steps; one step equals sixteen BN32 direct steps.
    visible = position + lane + 1
    tokens = _min(visible >> 2, fx.Int32(TOPK)) * 4 + (visible & 3)
    in_tile = (lane < rows) & (lane < (64 if MEMBERS64 else 32))
    steps = in_tile.select((tokens + 31) >> 5, fx.Int32(0))
    direct_steps = _wave_reduce(steps, operator.add)
    row_steps = _wave_reduce(steps, fx.max)
    budget = RATIO * fx.Float32(direct_steps)
    end = _min(MAX_BLOCKS, (position + rows + 3) >> 2)
    # Recovery only scatters within the tile's causal range. Copy the union out in block order
    # and zero what was set; past the budget the tile is direct, its count partial, and the
    # rest is only zeroed. Chunks of 256/512/1024 blocks (by MAX_BLOCKS) set where it stops.
    chunks = (MAX_BLOCKS <= 512).select(fx.Int32(2), fx.Int32(4))
    chunks = (MAX_BLOCKS <= 256).select(fx.Int32(1), chunks)
    row_bytes = MAX_BLOCKS * member_bytes
    row_start = fx.get_iter(DENSE) + fx.Int64(tile) * fx.Int64(MAX_BLOCKS)
    dense = rocdl.make_buffer_ptr(
        fx.recast_iter(fx.Int8, row_start), num_records_bytes=row_bytes
    )
    count = fx.Int32(0)
    start = fx.Int32(0)
    parity = fx.Int32(0)
    while (start < end) & (fx.Float32(((count + 15) >> 4) * 16) <= budget):
        ids, bits, masks = [], [], []
        for k in fx.range_constexpr(4):
            index = start + k * THREADS + tid
            offset = ((index < end) & (k < chunks)).select(
                index * member_bytes, row_bytes
            )
            words = _buffer_words(dense, offset, member_bytes // 4)
            ids.append(index)
            bits.append(words.bitcast(fx.Int64)[0] if MEMBERS64 else words[0])
            masks.append(_ballot(bits[k] != 0))
            # Every lane stores its wave's (equal) count; parity double-buffers the chunks.
            totals[parity * 16 + k * 4 + wave] = fx.Int32(fx.ctpop(masks[k]))
        fx.gpu.barrier()
        for k in fx.range_constexpr(4):
            waves = [totals[parity * 16 + k * 4 + w] for w in range(4)]
            before = _lanes_below(masks[k], lane)
            for w in fx.range_constexpr(3):
                before = before + (wave > w).select(waves[w], fx.Int32(0))
            if bits[k] != 0:
                BLOCKS[tile * CAPACITY + count + before] = ids[k]
                MEMBERSHIP[tile * CAPACITY + count + before] = bits[k]
                row_start[ids[k]] = member(0)
            count = count + waves[0] + waves[1] + waves[2] + waves[3]
        start = start + chunks * THREADS
        parity = parity ^ 1
    rest = start
    while rest < end:
        for k in fx.range_constexpr(4):
            index = rest + k * THREADS + tid
            if index < end:
                row_start[index] = member(0)
        rest = rest + 4 * THREADS
    proposal = (start >= end) & (fx.Float32(((count + 15) >> 4) * 16) <= budget)
    if tid == 0:
        COUNTS[tile] = count
        COSTS[tile * 3] = direct_steps
        COSTS[tile * 3 + 1] = row_steps
        COSTS[tile * 3 + 2] = proposal.select(fx.Int32(1), fx.Int32(0))


@flyc.jit
def _compact_launch(
    DENSE: fx.Tensor,
    BLOCKS: fx.Tensor,
    MEMBERSHIP: fx.Tensor,
    COUNTS: fx.Tensor,
    COSTS: fx.Tensor,
    META: fx.Tensor,
    DIRECT_FLAG: fx.Tensor,
    MAX_BLOCKS: fx.Int32,
    CAPACITY: fx.Int32,
    RATIO: fx.Float32,
    TILES: fx.Int32,
    MEMBERS64: fx.Constexpr[bool],
    TOPK: fx.Constexpr[int],
    stream: fx.Stream,
):
    if TILES > 0:
        _compact_kernel(
            DENSE,
            BLOCKS,
            MEMBERSHIP,
            COUNTS,
            COSTS,
            META,
            DIRECT_FLAG,
            MAX_BLOCKS,
            CAPACITY,
            RATIO,
            TILES,
            MEMBERS64,
            TOPK,
        ).launch(grid=(TILES, 1, 1), block=(THREADS, 1, 1), stream=stream)


def _block_reduce(values, ops, scratch, offset, wave):
    """CTA-wide reductions; every lane stores its wave's (equal) partial."""
    for i, (value, op) in enumerate(zip(values, ops)):
        scratch[offset + i * 4 + wave] = _wave_reduce(value, op)
    fx.gpu.barrier()
    totals = []
    for i, op in enumerate(ops):
        total = scratch[offset + i * 4]
        for w in range(1, 4):
            total = op(total, scratch[offset + i * 4 + w])
        totals.append(total)
    return totals


def _route_cost(
    union_sum, union_max, direct_sum, direct_max, task_share, step, fixed, model
):
    # Separate multiplies and adds, as in the fitted model.
    union_fixed, union_step, row_step, _ = model
    union = fx.Float32(union_fixed) + fx.Float32(union_step) * fx.max(
        fx.Float32(union_max), fx.Float32(union_sum) * task_share
    )
    direct = fixed + fx.max(
        fx.Float32(row_step) * fx.Float32(direct_max), step * fx.Float32(direct_sum)
    )
    zero = fx.Float32(0.0)
    return (union_sum > 0).select(union, zero) + (direct_sum > 0).select(direct, zero)


def _route_choice(totals, limits, task_share, step, fixed, model):
    """The limit of the cheapest candidate; totals hold 4 sums per candidate."""
    for j in range(LIMITS):
        cost = _route_cost(*totals[4 * j : 4 * j + 4], task_share, step, fixed, model)
        if j == 0:
            # Keeping every union tile wins ties and must be route_margin slower to lose.
            best_cost, best = cost * fx.Float32(1.0 - model[3]), limits[0]
        else:
            better = cost < best_cost
            best_cost = better.select(cost, best_cost)
            best = better.select(limits[j], best)
    return best


@flyc.jit
def _route(
    COUNTS: fx.Tensor,
    COSTS: fx.Tensor,
    scratch,
    TILES: fx.Int32,
    TASK_SHARE: fx.Float32,
    DIRECT_STEP: fx.Float32,
    DIRECT_FIXED: fx.Float32,
    ROUTE_MODEL: fx.Constexpr[tuple],
):
    # Union and direct run back to back, so one long union task or a few direct rows set a
    # latency floor the per-tile rule cannot see. Return the most N64 steps a union tile keeps:
    # every proposal-1 tile, those within max * 2**(-j/2) for j = 1..6, or none (-1).
    tid = fx.thread_idx.x
    wave = _uniform(tid >> 6)
    zero = fx.Int32(0)
    union_sum, union_max = zero, zero
    for tile in range(tid, TILES, THREADS):
        union = COSTS[tile * 3 + 2] == 1
        steps = union.select((COUNTS[tile] + 15) >> 4, zero)
        union_sum = union_sum + steps
        union_max = fx.max(union_max, steps)
    union_sum, union_max = _block_reduce(
        [union_sum, union_max], [operator.add, fx.max], scratch, EXCHANGE_WORDS, wave
    )
    limits = [
        fx.Int32(fx.Float32(union_max) * fx.Float32(2.0 ** (-0.5 * j)))
        for j in range(LIMITS - 1)
    ] + [fx.Int32(-1)]
    threshold = union_max
    # Only a union phase set by its longest task, not by throughput, gains from a shorter one.
    if fx.Float32(union_max) > fx.Float32(union_sum) * TASK_SHARE:
        kept_sum, kept_max = [zero] * LIMITS, [zero] * LIMITS
        direct_sum, direct_max = [zero] * LIMITS, [zero] * LIMITS
        for tile in range(tid, TILES, THREADS):
            steps = (COUNTS[tile] + 15) >> 4
            direct = COSTS[tile * 3]
            row = COSTS[tile * 3 + 1]
            union = COSTS[tile * 3 + 2] == 1
            for j in fx.range_constexpr(LIMITS):
                keep = union & (steps <= limits[j])
                kept_sum[j] = kept_sum[j] + keep.select(steps, zero)
                kept_max[j] = fx.max(kept_max[j], keep.select(steps, zero))
                direct_sum[j] = direct_sum[j] + keep.select(zero, direct)
                direct_max[j] = fx.max(direct_max[j], keep.select(zero, row))
        totals = _block_reduce(
            [
                value
                for j in range(LIMITS)
                for value in (kept_sum[j], kept_max[j], direct_sum[j], direct_max[j])
            ],
            [operator.add, fx.max, operator.add, fx.max] * LIMITS,
            scratch,
            EXCHANGE_WORDS + 8,
            wave,
        )
        threshold = _route_choice(
            totals, limits, TASK_SHARE, DIRECT_STEP, DIRECT_FIXED, ROUTE_MODEL
        )
    return threshold


@flyc.jit
def _order(
    COUNTS: fx.Tensor,
    COSTS: fx.Tensor,
    ACTIVE: fx.Tensor,
    ORDER: fx.Tensor,
    DIRECT_FLAG: fx.Tensor,
    scratch,
    program,
    threshold,
    TASKS: fx.Int32,
    GRID: fx.Int32,
    SIZE: fx.Int32,
    SHIFT: fx.Int32,
    HK: fx.Constexpr[int],
    SLICES: fx.Constexpr[int],
    WIDE: fx.Constexpr[bool],
    WIDTH: fx.Constexpr[int],
):
    key = fx.Int64 if WIDE else fx.Int32
    tid = fx.thread_idx.x
    per_head = TASKS // HK
    keys = []
    for e in fx.range_constexpr(WIDTH):
        position = tid * WIDTH + e
        rank = program * SIZE + position
        valid = (position < SIZE) & (rank < TASKS)
        tile = (rank % per_head) // SLICES
        steps = (valid.select(COUNTS[tile], fx.Int32(0)) + 15) >> 4
        union = valid.select(COSTS[tile * 3 + 2], fx.Int32(0)) == 1
        active = union & (steps <= threshold)
        # The first head's first slice of each tile publishes the route; any direct tile
        # raises the flag compact cleared, which gated pack/direct launches read.
        if valid & (rank < per_head) & (rank % SLICES == 0):
            ACTIVE[tile] = active.select(fx.Int32(1), fx.Int32(0))
            if ~active:
                DIRECT_FLAG[0] = 1
        cost = key(active.select(steps, fx.Int32(0)))
        rank_key = (cost << key(SHIFT)) + key(TASKS - 1 - rank)
        keys.append(valid.select(rank_key, key(-1)))
    cta = (scratch, tid, _uniform(tid >> 6), WIDE)
    keys = _bitonic(keys, tid & 63, True, cta)
    # Sort hkv-major (tie order), publish the tile-major task the union kernel runs.
    mask = (key(1) << key(SHIFT)) - 1
    rounded = (TASKS + GRID - 1) // GRID * GRID
    for e in fx.range_constexpr(WIDTH):
        position = tid * WIDTH + e
        rank = program * SIZE + position
        task = TASKS - 1 - fx.Int32(keys[e] & mask)
        task = (keys[e] >= 0).select(
            (task % per_head) * HK + task // per_head, fx.Int32(-1)
        )
        row, column = rank // GRID, rank % GRID
        destination = row * GRID + ((row & 1) == 0).select(column, GRID - 1 - column)
        if (position < SIZE) & (destination < rounded):
            ORDER[destination] = task


def _slot(MEMBERSHIP, BLOCKS, slot, live):
    slot = live.select(slot, fx.Int32(0))
    member = MEMBERSHIP[slot]
    zero = type(member)(0)
    return live.select(member, zero), live.select(BLOCKS[slot] * 4, fx.Int32(0))


@flyc.jit
def _masks(
    COUNTS: fx.Tensor,
    COSTS: fx.Tensor,
    META: fx.Tensor,
    BLOCKS: fx.Tensor,
    MEMBERSHIP: fx.Tensor,
    MASKS: fx.Tensor,
    task,
    QB: fx.Int32,
    CAP: fx.Int32,
    TILES: fx.Int32,
):
    # Entry (nt, query, quarter) holds 4 bits for each of the slots nt*16 + quarter*2 +
    # (group//2)*8 + group%2. One lane owns an (nt, quarter) pair: its four slots are
    # loaded once and shared by every query. Each wave owns one (tile, part); a part
    # takes 16 of every 64 slot groups.
    lane = fx.thread_idx.x & 63
    tile = TILES - 1 - (task >> 2)
    if _uniform(COSTS[tile * 3 + 2]) != 0:
        count = _uniform(COUNTS[tile])
        rows = _uniform(META[tile * 5 + 1])
        length = _uniform(META[tile * 5 + 3])
        position = _uniform(META[tile * 5 + 4])
        groups = (count + 15) >> 4
        first = (task & 3) * 16 + (lane >> 2)
        quarter = lane & 3
        steps = (fx.max(groups - (task & 3) * 16, fx.Int32(0)) + 63) >> 6
        for step in range(fx.Int32(0), steps, fx.Int32(1)):
            nt = first + step * 64
            valid = nt < groups
            slot = nt * 16 + quarter * 2
            base = tile * CAP + slot
            pairs = [
                _slot(MEMBERSHIP, BLOCKS, base + o, valid & (slot + o < count))
                for o in (0, 1, 8, 9)
            ]
            masks = tile * (CAP >> 4) * QB * 4 + nt * QB * 4 + quarter
            for query in range(fx.Int32(0), QB, fx.Int32(1)):
                visible = fx.min(position + query + 1, length)
                ok = query < rows
                bits = fx.Int32(0)
                for i in fx.range_constexpr(4):
                    member, start = pairs[i]
                    hit = (((member >> type(member)(query)) & 1) != 0) & ok
                    width = fx.min(fx.max(visible - start, fx.Int32(0)), fx.Int32(4))
                    nibble = hit.select((fx.Int32(1) << width) - 1, fx.Int32(0))
                    bits = bits | (nibble << (4 * i))
                if valid:
                    MASKS[masks + query * 4] = fx.Int16(bits)


@flyc.kernel(name="qsa_order_masks", known_block_size=[THREADS, 1, 1])
def _order_masks_kernel(
    COUNTS: fx.Tensor,
    COSTS: fx.Tensor,
    ACTIVE: fx.Tensor,
    ORDER: fx.Tensor,
    DIRECT_FLAG: fx.Tensor,
    BLOCKS: fx.Tensor,
    MEMBERSHIP: fx.Tensor,
    META: fx.Tensor,
    MASKS: fx.Tensor,
    TASKS: fx.Int32,
    GRID: fx.Int32,
    SIZE: fx.Int32,
    SHIFT: fx.Int32,
    ORDER_CTAS: fx.Int32,
    QB: fx.Int32,
    CAP: fx.Int32,
    TILES: fx.Int32,
    TASK_SHARE: fx.Float32,
    DIRECT_STEP: fx.Float32,
    DIRECT_FIXED: fx.Float32,
    HK: fx.Constexpr[int],
    SLICES: fx.Constexpr[int],
    WIDE: fx.Constexpr[bool],
    ROUTE: fx.Constexpr[bool],
    ROUTE_MODEL: fx.Constexpr[tuple],
):
    scratch = (
        fx.SharedAllocator()
        .allocate(fx.Array[fx.Int32, ORDER_WORDS, 16])
        .peek()
        .view(fx.make_layout(ORDER_WORDS, 1))
    )
    program = fx.block_idx.x
    if program < ORDER_CTAS:
        threshold = fx.Int32(INT_MAX)
        if fx.const_expr(ROUTE):
            threshold = _route(
                COUNTS,
                COSTS,
                scratch,
                TILES,
                TASK_SHARE,
                DIRECT_STEP,
                DIRECT_FIXED,
                ROUTE_MODEL,
            )
        for width in fx.range_constexpr(len(SORT_WIDTHS)):
            width = SORT_WIDTHS[width]
            lower = 0 if width == 1 else THREADS * width // 2
            if (SIZE > lower) & (SIZE <= THREADS * width):
                _order(
                    COUNTS,
                    COSTS,
                    ACTIVE,
                    ORDER,
                    DIRECT_FLAG,
                    scratch,
                    program,
                    threshold,
                    TASKS,
                    GRID,
                    SIZE,
                    SHIFT,
                    HK,
                    SLICES,
                    WIDE,
                    width,
                )
    else:
        # Mask CTAs pack one (tile, part) per wave.
        task = (program - ORDER_CTAS) * 4 + _uniform(fx.thread_idx.x >> 6)
        _masks(COUNTS, COSTS, META, BLOCKS, MEMBERSHIP, MASKS, task, QB, CAP, TILES)


@flyc.jit
def _order_masks_launch(
    COUNTS: fx.Tensor,
    COSTS: fx.Tensor,
    ACTIVE: fx.Tensor,
    ORDER: fx.Tensor,
    DIRECT_FLAG: fx.Tensor,
    BLOCKS: fx.Tensor,
    MEMBERSHIP: fx.Tensor,
    META: fx.Tensor,
    MASKS: fx.Tensor,
    TASKS: fx.Int32,
    GRID: fx.Int32,
    SIZE: fx.Int32,
    SHIFT: fx.Int32,
    ORDER_CTAS: fx.Int32,
    QB: fx.Int32,
    CAP: fx.Int32,
    TILES: fx.Int32,
    TASK_SHARE: fx.Float32,
    DIRECT_STEP: fx.Float32,
    DIRECT_FIXED: fx.Float32,
    HK: fx.Constexpr[int],
    SLICES: fx.Constexpr[int],
    WIDE: fx.Constexpr[bool],
    ROUTE: fx.Constexpr[bool],
    ROUTE_MODEL: fx.Constexpr[tuple],
    stream: fx.Stream,
):
    if TILES > 0:
        _order_masks_kernel(
            COUNTS,
            COSTS,
            ACTIVE,
            ORDER,
            DIRECT_FLAG,
            BLOCKS,
            MEMBERSHIP,
            META,
            MASKS,
            TASKS,
            GRID,
            SIZE,
            SHIFT,
            ORDER_CTAS,
            QB,
            CAP,
            TILES,
            TASK_SHARE,
            DIRECT_STEP,
            DIRECT_FIXED,
            HK,
            SLICES,
            WIDE,
            ROUTE,
            ROUTE_MODEL,
        ).launch(grid=(ORDER_CTAS + TILES, 1, 1), block=(THREADS, 1, 1), stream=stream)


def _cdiv(a, b):
    return -(-a // b)


def upload(array, device) -> torch.Tensor:
    # Pinned and asynchronous: building a new layout never drains the stream.
    host = torch.from_numpy(np.ascontiguousarray(array, dtype=np.int32))
    return host.pin_memory().to(device, non_blocking=True)


def scratch_specs(plan: SparsePlan) -> list:
    """(field, shape, dtype, zeroed) of the per-call scratch; every launch rewrites what it reads."""
    tiles, capacity = plan.num_tiles, plan.block_capacity
    members = torch.int64 if plan.query_tile > 32 else torch.int32
    return [
        ("dense_membership", (tiles, plan.max_blocks), members, True),
        ("blocks", (tiles, capacity), torch.int32, False),
        ("membership", (tiles, capacity), members, False),
        (
            "score_masks",
            (tiles, capacity // 16, plan.query_tile, 4),
            torch.int16,
            False,
        ),
    ]


def allocate_scratch(owner, specs, device) -> None:
    for name, shape, dtype, zeroed in specs:
        setattr(
            owner,
            name,
            (torch.zeros if zeroed else torch.empty)(shape, dtype=dtype, device=device),
        )


def wide_tile_rows(query_tile: int) -> int:
    """Fewest full-prefill rows that take the BQ21/BQ42 tile."""
    return math.ceil(query_tile * get_cu_num() * platform().wide_tiles_per_cu)


def allocate_plan(*, inputs, query_tile: int, scratch: bool = True) -> SparsePlan:
    group = inputs.q.shape[1] // inputs.k.shape[1]
    if query_tile < 1 or query_tile > 32:
        raise ValueError("Require 1<=BQ<=32")
    config = platform()
    num_cus = get_cu_num()
    # Full H6/H3 prefill fills 126/128 MFMA rows (BQ21 / BQ42 with 64-bit membership). Keep the
    # established tiles for prefix/ragged work and batches too small to fill the persistent grid.
    full = (
        inputs.k.shape[1] == 1
        and len(inputs.query_lens) == 1
        and inputs.prefix_lens[0] == 0
    )
    if (
        group == 6
        and full
        and query_tile >= 21
        and inputs.query_lens[0] >= wide_tile_rows(21)
    ):
        query_tile = 21
    elif (
        group == 3
        and full
        and query_tile == 32
        and inputs.query_lens[0] >= wide_tile_rows(42)
    ):
        query_tile = 42
    else:
        query_tile = min(
            query_tile,
            128 // group if group == 12 else 1 << ((128 // group).bit_length() - 1),
        )
    q_lens = np.asarray(inputs.query_lens, dtype=np.int64)
    lengths = q_lens + np.asarray(inputs.prefix_lens, dtype=np.int64)
    # Equal tiles per request, larger first: a tiny last tile would route its few rows direct, after union.
    counts = -(-q_lens // query_tile)
    request = np.repeat(np.arange(len(q_lens)), counts)
    index = np.arange(len(request)) - np.repeat(np.cumsum(counts) - counts, counts)
    base = (q_lens // np.maximum(counts, 1))[request]
    extra = (q_lens % np.maximum(counts, 1))[request]
    local = index * base + np.minimum(index, extra)
    sizes = base + (index < extra)
    metadata = np.stack(
        (
            (np.cumsum(q_lens) - q_lens)[request] + local,
            sizes,
            (np.cumsum(lengths) - lengths)[request],
            lengths[request],
            (lengths - q_lens)[request] + local,
        ),
        axis=1,
    )
    max_blocks = _cdiv(inputs.max_seqlen_k, 4)
    # A query adds at most topk blocks and its tail block.
    capacity = _cdiv(min(max_blocks, query_tile * (inputs.topk + 1)), 16) * 16
    tiles = len(request)
    hk = inputs.k.shape[1]
    slices = _cdiv(query_tile * group, 128)
    tasks = tiles * hk * slices
    # A union CTA (8 waves, 64 KiB LDS) fills a CU: a larger grid only adds dispatch rounds.
    grid = min(tasks, num_cus)
    packed = int((-(-lengths // 4))[q_lens > 0].sum())
    # Per-layout terms of the route model: union step share per tile, direct cost per step, fixed direct cost.
    route_costs = (
        hk * slices / grid if grid else 0.0,
        config.direct_cu_step * hk / num_cus,
        config.direct_fixed + config.pack_block * packed * hk,
    )
    kwargs = {"dtype": torch.int32, "device": inputs.q.device}
    plan = SparsePlan(
        metadata=upload(metadata, inputs.q.device),
        dense_membership=None,
        blocks=None,
        membership=None,
        score_masks=None,
        counts=torch.empty(tiles, **kwargs),
        costs=torch.empty((tiles, 3), **kwargs),
        active=torch.empty(tiles, **kwargs),
        task_order=torch.empty(_cdiv(tasks, grid) * grid if grid else 0, **kwargs),
        direct_flag=torch.empty(1, **kwargs),
        query_tiles=upload(np.repeat(np.arange(tiles), sizes), inputs.q.device),
        query_tile=query_tile,
        group_padded=group,
        block_capacity=capacity,
        max_blocks=max_blocks,
        num_tiles=tiles,
        grid=grid,
        route_costs=route_costs,
        route_model=(
            config.union_fixed,
            config.union_step,
            config.row_step,
            config.route_margin,
        ),
        union_ratio=config.union_ratio,
    )
    if scratch:
        allocate_scratch(plan, scratch_specs(plan), inputs.q.device)
    return plan


_COMPILED = {}


def _compiled(launch, args, grid, constants):
    """Compile once per device and constexpr values. Compiling runs the launcher, so it gets
    an empty grid (args[grid] = 0) and launches nothing."""
    device = args[0].device
    key = (launch, device, *constants)
    compiled = _COMPILED.get(key)
    if compiled is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Warm the QSA specialization before graph capture")
        empty = list(args)
        empty[grid] = 0
        with torch.cuda.device(device):
            compiled = _COMPILED[key] = flyc.compile(launch, *empty)
    return compiled


def _launches(inputs, plan, stream):
    """(compiled, args) of K1-K3 in launch order; K1's first argument is the indices."""
    standin = inputs.query_positions  # tensor slots K1 does not read without a plan
    bits = 0 if plan is None else 64 if plan.query_tile > 32 else 32
    args = (
        inputs.indices,
        inputs.query_positions,
        inputs.block_indices.view(-1),
        # 2-D: FlyDSL passes int32 dims, and tiles * max_blocks may exceed 2**31.
        standin if plan is None else plan.dense_membership,
        standin if plan is None else plan.query_tiles,
        standin if plan is None else plan.metadata.view(-1),
        inputs.q.shape[0],
        0 if plan is None else plan.max_blocks,
        bits,
        inputs.topk,
        stream,
    )
    yield _compiled(_recover_launch, args, 6, (bits, inputs.topk)), args
    if plan is None or plan.num_tiles == 0:
        return
    members64 = plan.query_tile > 32
    args = (
        plan.dense_membership,
        plan.blocks.view(-1),
        plan.membership.view(-1),
        plan.counts,
        plan.costs.view(-1),
        plan.metadata.view(-1),
        plan.direct_flag,
        plan.max_blocks,
        plan.block_capacity,
        plan.union_ratio,
        plan.num_tiles,
        members64,
        inputs.topk,
        stream,
    )
    yield _compiled(_compact_launch, args, 10, (members64, inputs.topk)), args
    hk = inputs.k.shape[1]
    slices = _cdiv(plan.query_tile * plan.group_padded, 128)
    tasks = plan.num_tiles * hk * slices
    numel = plan.task_order.numel()
    # One order CTA sorts the whole task order up to the widest variant, else chunks of it.
    size = min(THREADS * SORT_WIDTHS[-1], numel)
    shift = max(1, (tasks - 1).bit_length())
    wide = ((plan.block_capacity // 16) << shift) + tasks - 1 >= 2**31
    args = (
        plan.counts,
        plan.costs.view(-1),
        plan.active,
        plan.task_order,
        plan.direct_flag,
        plan.blocks.view(-1),
        plan.membership.view(-1),
        plan.metadata.view(-1),
        plan.score_masks.view(-1),
        tasks,
        plan.grid,
        size,
        shift,
        _cdiv(numel, size),
        plan.query_tile,
        plan.block_capacity,
        plan.num_tiles,
        *plan.route_costs,
        hk,
        slices,
        wide,
        plan.routing,
        plan.route_model,
        stream,
    )
    constants = (hk, slices, wide, plan.routing, plan.route_model, members64)
    yield _compiled(_order_masks_launch, args, 16, constants), args


def run(*, inputs, plan):
    stream = torch.cuda.current_stream(inputs.q.device)
    for compiled, args in _launches(inputs, plan, stream):
        compiled(*args)


def launchers(*, inputs, plan):
    """K1-K3 launches with this binding's other arguments prebuilt; K1 takes the indices.

    Valid while the plan's scratch bindings and the current stream stay the same.
    """
    stream = torch.cuda.current_stream(inputs.q.device)
    (recover, args), *rest = _launches(inputs, plan, stream)
    tail = args[1:]
    return [lambda indices: recover(indices, *tail)] + [
        partial(compiled, *args) for compiled, args in rest
    ]
