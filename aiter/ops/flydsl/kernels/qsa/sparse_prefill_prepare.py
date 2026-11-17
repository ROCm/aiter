# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Exact QSA attention selection, private union scratch and fused preparation launches.

These three FlyDSL kernels (K1-K3) are the private preparation step of the
FlyDSL QSA sparse attention; they are not a public op.

Recovery does not validate the selection layout; it writes each row's sorted complete blocks.
Compaction builds each union candidate tile's block union and membership from those rows, in LDS.
The next launch sorts tasks and builds disjoint mask partitions; with a slot table it also maps
the union blocks to pool blocks. Plans use one stream's scratch and are pinned for graphs.
"""

import math
import operator
from dataclasses import dataclass
from types import SimpleNamespace

import flydsl.compiler as flyc
import flydsl.expr as fx
import numpy as np
import torch
from flydsl.expr import rocdl

from aiter.jit.utils.chip_info import get_cu_num

from .sparse_prefill_common import (
    _buffer,
    _buffer_store_words,
    _buffer_words,
    _lds_load,
    _lds_pointer,
    _lds_store,
    _min,
    _uniform,
    platform,
)

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
    # Slot-table launches (K/V read from a token pool through kv_slots): K3 writes each union block's
    # pool block id and each tile's final partial block pool row (-1: none) for the union kernel.
    slotted: bool = False
    pool_blocks: torch.Tensor | None = None
    tails: torch.Tensor | None = None
    # Gated raw direct (set by its plan): K3 lists the row groups holding a direct row, so the
    # direct grid runs them first. direct_flag[1] counts them.
    direct_meta: torch.Tensor | None = None
    direct_list: torch.Tensor | None = None
    direct_groups: int = 0


def _ballot(pred):
    return fx.Int64(rocdl.ballot(fx.Int64.ir_type, pred))


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
    ROWS: fx.Int32,
    row,
    lane,
    scratch,
    slot,
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


@flyc.kernel(name="sparse_prefill_recover_scatter", known_block_size=[THREADS, 1, 1])
def _recover_kernel(
    INDICES: fx.Tensor,
    POSITIONS: fx.Tensor,
    BLOCKS: fx.Tensor,
    ROWS: fx.Int32,
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
            ROWS,
            row,
            tid & 63,
            scratch,
            wave * words,
            TOPK,
        )


@flyc.jit
def _recover_launch(
    INDICES: fx.Tensor,
    POSITIONS: fx.Tensor,
    BLOCKS: fx.Tensor,
    ROWS: fx.Int32,
    TOPK: fx.Constexpr[int],
    stream: fx.Stream,
):
    if ROWS > 0:
        _recover_kernel(
            INDICES,
            POSITIONS,
            BLOCKS,
            ROWS,
            TOPK,
        ).launch(grid=((ROWS + 3) // 4, 1, 1), block=(THREADS, 1, 1), stream=stream)


# K2: each union candidate tile's exact block union, from the rows' sorted complete blocks
# (K1's block_indices) and their causal tail blocks. One CTA per tile, last tile first; LDS is
# sized per launch. A tile whose causal range fits the LDS ORs each row's bit into a per-block
# membership word and compacts the nonzero words in block order. Longer ranges go a
# WINDOW_BLOCKS-block window at a time: blocks set bits of a presence bitmap, a CTA-wide prefix
# over its words ranks every present block, and membership accumulates in LDS by union index,
# MEMBER_BYTES at a time. A tile whose padded union exceeds `ratio` times its direct work is
# routed direct (its count is then partial). Work grows with the rows' selections, plus one
# bitmap word per 32 blocks of a long range; no scratch grows with tiles x KV blocks.

# CTA sizes: tiles of more rows take more waves.
COMPACT_THREADS = (256, 512)
# Block ids loaded in flight per thread in a pass over the rows' blocks.
UNROLL = 16
WINDOW_BLOCKS = 65536
WORDS = WINDOW_BLOCKS // 32
# LDS of a launch whose longest range does not fit densely: two CTAs per CU.
LDS_BUDGET = 32768
# Windowed LDS: presence words, their exclusive prefix counts in the window, then a membership
# chunk up to the waves' scan totals, which always take the last SCAN_BYTES.
SCAN_BYTES = max(COMPACT_THREADS) // 64 * 4
PRESENCE, PREFIX, MEMBER = 0, WORDS * 4, 2 * WORDS * 4
MEMBER_BYTES = LDS_BUDGET - SCAN_BYTES - MEMBER


def _resource(tensor, size_bytes, offset=None):
    """Byte-addressed buffer over tensor from element offset on."""
    pointer = fx.get_iter(tensor)
    if offset is not None:
        pointer = pointer + offset
    return _buffer(fx.make_view(pointer, fx.make_layout(1, 1)), size_bytes)


def _word(buffer, offset):
    return _buffer_words(buffer, offset, 1)[0]


def _put(buffer, index, value):
    words = fx.Vector.from_elements([fx.Int32(value)], fx.Int32)
    _buffer_store_words(words, buffer, fx.Int32(index) * 4)


def _cta_sync():
    fx.memory_fence(syncscope="workgroup")
    rocdl.s_barrier()
    fx.memory_fence(syncscope="workgroup")


def _inclusive_scan(value, lane):
    for offset in (1, 2, 4, 8, 16, 32):
        source = ((lane - offset) & 63) * 4
        other = fx.Int32(rocdl.ds_bpermute(fx.Int32.ir_type, source, value))
        value = (lane >= offset).select(value + other, value)
    return value


def _popcount(value):
    return fx.Int32(fx.ctpop(value))


def _entry_bytes(wide):
    return 8 if wide else 4


def _entry_type(wide):
    return fx.Int64 if wide else fx.Int32


def _or(address, value, wide):
    fx.atomic_or(_lds_pointer(address, _entry_type(wide)), value, syncscope="workgroup")


def _member_bit(row, wide):
    return (fx.Int64(1) << fx.Int64(row)) if wide else (fx.Int32(1) << row)


def _load_entry(address, wide):
    return _lds_load(address, _entry_type(wide))


def _store_entry(address, value, wide):
    _lds_store(address, value, _entry_type(wide))


def _put_entry(members, index, value, wide):
    words = fx.Vector.from_elements([value], _entry_type(wide)).bitcast(fx.Int32)
    _buffer_store_words(words, members, index * _entry_bytes(wide))


def _zero(wide):
    return fx.Int64(0) if wide else fx.Int32(0)


def _rank(shared, offset):
    """Union index within the window of the present block `offset` blocks into it."""
    word = offset >> 5
    below = (fx.Int32(1) << (offset & 31)) - 1
    return _lds_load(shared + PREFIX + word * 4) + _popcount(
        _lds_load(shared + PRESENCE + word * 4) & below
    )


@flyc.jit
def _dense_block(shared, block, row, end, wide):
    if (block >= 0) & (block < end):
        _or(shared + block * _entry_bytes(wide), _member_bit(row, wide), wide)


@flyc.jit
def _mark_block(shared, block, w0, w1):
    if (block >= w0) & (block < w1):
        offset = block - w0
        _or(shared + PRESENCE + (offset >> 5) * 4, fx.Int32(1) << (offset & 31), False)


@flyc.jit
def _member_block(shared, block, row, w0, w1, c0, entries, wide):
    if (block >= w0) & (block < w1):
        index = _rank(shared, block - w0) - c0
        if (index >= 0) & (index < entries):
            address = shared + MEMBER + index * _entry_bytes(wide)
            _or(address, _member_bit(row, wide), wide)


def _visit(cta, block, row, mode, w0, w1, c0, entries):
    """Mode 0: OR the row's bit into the block's dense word; 1: mark the block present;
    2: add the row's bit to the block's union entry in the membership chunk."""
    if mode == 0:
        _dense_block(cta.shared, block, row, w1, cta.wide)
    elif mode == 1:
        _mark_block(cta.shared, block, w0, w1)
    else:
        _member_block(cta.shared, block, row, w0, w1, c0, entries, cta.wide)


def _chunks(cta, first, total):
    """UNROLL block ids at elements first + k * threads (all loads issued before any use;
    a wave's lanes hold consecutive elements), as (element, id), id -1 past the tile's rows.
    """
    loads = []
    for k in range(UNROLL):
        element = first + k * cta.threads
        loads.append((element, _word(cta.rows, element * 4)))
    return [
        (element, (element < total).select(block, fx.Int32(-1)))
        for element, block in loads
    ]


def _visit_chunks(cta, chunks, mode, w0, w1, c0, entries):
    topk = cta.topk
    for element, block in chunks:
        # Row of the element: TOPK block ids per row.
        if topk & (topk - 1) == 0:
            row = element >> (topk.bit_length() - 1)
        else:
            row = element // topk
        _visit(cta, block, row, mode, w0, w1, c0, entries)


@flyc.jit
def _blocks(cta, mode, w0, w1, c0, entries):
    """Visit every complete block of the tile's rows, then each row's causal tail block
    (visible // 4 holds its last visible % 4 tokens)."""
    total = cta.nrows * cta.topk
    for start in range(cta.tid, total, fx.Int32(cta.threads * UNROLL)):
        chunks = _chunks(cta, fx.Int32(start), total)
        _visit_chunks(cta, chunks, mode, w0, w1, c0, entries)
    visible = cta.position + cta.tid + 1
    if (cta.tid < cta.nrows) & ((visible & 3) != 0) & ((visible >> 2) < cta.max_blocks):
        _visit(cta, visible >> 2, cta.tid, mode, w0, w1, c0, entries)


def _prefix_base(cta, inclusive, local):
    """(this thread's exclusive prefix over the CTA, CTA total) from the waves' inclusive scans."""
    totals = [_lds_load(cta.scan + w * 4) for w in range(cta.threads // 64)]
    running = inclusive - local
    for w, total in enumerate(totals):
        running = running + (fx.Int32(w) < cta.wave).select(total, fx.Int32(0))
    present = totals[0]
    for total in totals[1:]:
        present = present + total
    return running, present


@flyc.jit
def _cta_prefix(cta, local):
    inclusive = _inclusive_scan(local, cta.lane)
    if cta.lane == 63:
        _lds_store(cta.scan + cta.wave * 4, inclusive)
    _cta_sync()
    running, present = _prefix_base(cta, inclusive, local)
    return running, present


def _over(count, budget):
    padded = (16 * ((count + 15) >> 4)).to(fx.Float32)
    return (padded > budget).select(fx.Int32(1), fx.Int32(0))


def _owned(cta, count):
    """This thread's contiguous share [lo, hi) of `count` items."""
    per = (count + cta.threads - 1) // cta.threads
    lo = _min(cta.tid * per, count)
    return lo, _min(lo + per, count)


def _below(mask):
    low = rocdl.mbcnt_lo(
        fx.Int32.ir_type, fx.Int32(mask).ir_value(), fx.Int32(0).ir_value()
    )
    return fx.Int32(
        rocdl.mbcnt_hi(fx.Int32.ir_type, fx.Int32(mask >> 32).ir_value(), low)
    )


def _count(mask):
    return fx.Int32(fx.ctpop(mask))


@flyc.jit
def _wave_bases(cta, total):
    """(this wave's exclusive prefix over the CTA, CTA total) of the waves' uniform totals."""
    if cta.lane == 0:
        _lds_store(cta.scan + cta.wave * 4, total)
    _cta_sync()
    base, present = _prefix_base(cta, fx.Int32(0), fx.Int32(0))
    return base, present


@flyc.jit
def _put_word(cta, index, word, base, mask):
    if word != _zero(cta.wide):
        destination = base + _below(mask)
        _put(cta.output, destination, index)
        _put_entry(cta.members, destination, word, cta.wide)


@flyc.jit
def _dense(cta, end, budget):
    """(union size, whether over budget) of a tile whose causal range fits the LDS, one word
    per block. Each wave compacts its contiguous share 64 words at a time, so the outputs
    coalesce."""
    size = _entry_bytes(cta.wide)
    zero = _zero(cta.wide)
    for index in range(cta.tid, end, fx.Int32(cta.threads)):
        _store_entry(cta.shared + fx.Int32(index) * size, zero, cta.wide)
    _cta_sync()
    _blocks(cta, 0, fx.Int32(0), end, fx.Int32(0), fx.Int32(0))
    _cta_sync()
    waves = cta.threads // 64
    per = ((end + waves - 1) // waves + 63) & ~63
    lo = _min(cta.wave * per, end)
    hi = _min(lo + per, end)
    total = fx.Int32(0)
    for start in range(lo, hi, fx.Int32(64)):
        index = fx.Int32(start) + cta.lane
        address = cta.shared + _min(index, end - 1) * size
        word = (index < hi).select(_load_entry(address, cta.wide), zero)
        total = total + _count(_ballot(word != zero))
    base, present = _wave_bases(cta, total)
    over = _over(present, budget)
    if over == 0:
        for start in range(lo, hi, fx.Int32(64)):
            index = fx.Int32(start) + cta.lane
            address = cta.shared + _min(index, end - 1) * size
            word = (index < hi).select(_load_entry(address, cta.wide), zero)
            mask = _ballot(word != zero)
            _put_word(cta, index, word, base, mask)
            base = base + _count(mask)
    return present, over


@flyc.jit
def _emit_blocks(cta, base, w0, lo, hi):
    # Present blocks of this thread's words in order, from union index base + prefix.
    for index in range(lo, hi, fx.Int32(1)):
        word = fx.Int32(index)
        bits = _lds_load(cta.shared + PRESENCE + word * 4)
        union = base + _lds_load(cta.shared + PREFIX + word * 4)
        while bits != 0:
            bit = fx.cttz(bits)
            _put(cta.output, union, w0 + word * 32 + bit)
            union = union + 1
            bits = bits & (bits - 1)


@flyc.jit
def _window(cta, w0, w1, count, budget):
    """Union of blocks [w0, w1) appended at `count`; returns (new count, whether over budget)."""
    words = (w1 - w0 + 31) >> 5
    for index in range(cta.tid, words, fx.Int32(cta.threads)):
        _lds_store(cta.shared + PRESENCE + fx.Int32(index) * 4, 0)
    _cta_sync()
    zero = fx.Int32(0)
    _blocks(cta, 1, w0, w1, zero, zero)
    _cta_sync()
    # Thread t owns words [lo, hi); a CTA-wide exclusive scan of their set bits gives each
    # word's prefix.
    lo, hi = _owned(cta, words)
    local = fx.Int32(0)
    for index in range(lo, hi, fx.Int32(1)):
        local = local + _popcount(
            _lds_load(cta.shared + PRESENCE + fx.Int32(index) * 4)
        )
    running, present = _cta_prefix(cta, local)
    for index in range(lo, hi, fx.Int32(1)):
        word = fx.Int32(index)
        _lds_store(cta.shared + PREFIX + word * 4, running)
        running = running + _popcount(_lds_load(cta.shared + PRESENCE + word * 4))
    over = _over(count + present, budget)
    if over == 0:
        _emit_blocks(cta, count, w0, lo, hi)
        size = _entry_bytes(cta.wide)
        entries = MEMBER_BYTES // size
        for start in range(fx.Int32(0), present, fx.Int32(entries)):
            c0 = fx.Int32(start)
            # Union indices c0 .. c0 + filled - 1 of this chunk.
            filled = _min(present - c0, fx.Int32(entries))
            for index in range(cta.tid, filled, fx.Int32(cta.threads)):
                address = cta.shared + MEMBER + fx.Int32(index) * size
                _store_entry(address, _zero(cta.wide), cta.wide)
            _cta_sync()
            _blocks(cta, 2, w0, w1, c0, fx.Int32(entries))
            _cta_sync()
            for index in range(cta.tid, filled, fx.Int32(cta.threads)):
                slot = fx.Int32(index)
                entry = _load_entry(cta.shared + MEMBER + slot * size, cta.wide)
                _put_entry(cta.members, count + c0 + slot, entry, cta.wide)
            _cta_sync()
    _cta_sync()
    return count + over.select(fx.Int32(0), present), over


@flyc.jit
def _tile(
    ROWS,
    META,
    BLOCKS,
    MEMBERSHIP,
    COUNTS,
    COSTS,
    DIRECT_FLAG,
    max_blocks,
    capacity,
    ratio,
    dense_limit,
    lds_bytes,
    wide,
    threads,
    topk,
):
    storage = (
        fx.SharedAllocator(static=False).allocate(fx.Array[fx.Int8, 16, 16]).peek()
    )
    shared = fx.Int32(fx.ptrtoint(fx.get_iter(storage.view(fx.make_layout(16, 1)))))
    tid = fx.thread_idx.x
    wave, lane = _uniform(tid >> 6), tid & 63
    program = fx.block_idx.x
    # Last tile first: in a causal request it holds the most blocks.
    tile = fx.grid_dim.x - 1 - program
    first_row, rows = _uniform(META[tile * 5]), _uniform(META[tile * 5 + 1])
    position = _uniform(META[tile * 5 + 4])
    # Union pads blocks to N64 steps; one step equals sixteen BN32 direct steps of a row.
    visible = position + lane + 1
    tokens = _min(visible >> 2, fx.Int32(topk)) * 4 + (visible & 3)
    steps = (lane < rows).select((tokens + 31) >> 5, fx.Int32(0))
    direct_steps = _wave_reduce(steps, operator.add)
    row_steps = _wave_reduce(steps, fx.max)
    budget = ratio * direct_steps.to(fx.Float32)
    end = _min(max_blocks, (position + rows + 3) >> 2)
    size = _entry_bytes(wide)
    # MEMBERSHIP is an int32 view; 64-bit entries take two words.
    cta = SimpleNamespace(
        shared=shared,
        scan=shared + lds_bytes - SCAN_BYTES,
        tid=tid,
        wave=wave,
        lane=lane,
        threads=threads,
        topk=topk,
        wide=wide,
        nrows=rows,
        position=position,
        max_blocks=max_blocks,
        rows=_resource(ROWS, rows * (topk * 4), fx.Int64(first_row) * topk),
        output=_resource(BLOCKS, capacity * 4, fx.Int64(tile) * fx.Int64(capacity)),
        members=_resource(
            MEMBERSHIP,
            capacity * size,
            fx.Int64(tile) * fx.Int64(capacity * (size // 4)),
        ),
    )
    count, over = fx.Int32(0), fx.Int32(0)
    if end <= dense_limit:
        count, over = _dense(cta, end, budget)
    else:
        w0 = fx.Int32(0)
        while (w0 < end) & (over == 0):
            w1 = _min(w0 + WINDOW_BLOCKS, end)
            count, over = _window(cta, w0, w1, count, budget)
            w0 = w1
    if tid == 0:
        _put(_resource(COUNTS, 4, fx.Int64(tile)), fx.Int32(0), count)
        costs = _resource(COSTS, 12, fx.Int64(tile) * 3)
        _put(costs, fx.Int32(0), direct_steps)
        _put(costs, fx.Int32(1), row_steps)
        _put(costs, fx.Int32(2), (over == 0).select(fx.Int32(1), fx.Int32(0)))
        if program == 0:
            flag = _resource(DIRECT_FLAG, 8)
            _put(flag, fx.Int32(0), fx.Int32(0))
            _put(flag, fx.Int32(1), fx.Int32(0))


@flyc.kernel(name="sparse_prefill_compact", known_block_size=[COMPACT_THREADS[0], 1, 1])
def _compact_kernel(
    ROWS: fx.Tensor,
    META: fx.Tensor,
    BLOCKS: fx.Tensor,
    MEMBERSHIP: fx.Tensor,
    COUNTS: fx.Tensor,
    COSTS: fx.Tensor,
    DIRECT_FLAG: fx.Tensor,
    MAX_BLOCKS: fx.Int32,
    CAPACITY: fx.Int32,
    RATIO: fx.Float32,
    DENSE_LIMIT: fx.Int32,
    LDS_BYTES: fx.Int32,
    MEMBERS64: fx.Constexpr[bool],
    TOPK: fx.Constexpr[int],
):
    _tile(
        ROWS,
        META,
        BLOCKS,
        MEMBERSHIP,
        COUNTS,
        COSTS,
        DIRECT_FLAG,
        MAX_BLOCKS,
        CAPACITY,
        RATIO,
        DENSE_LIMIT,
        LDS_BYTES,
        MEMBERS64,
        COMPACT_THREADS[0],
        TOPK,
    )


@flyc.kernel(name="sparse_prefill_compact", known_block_size=[COMPACT_THREADS[1], 1, 1])
def _compact_kernel_large(
    ROWS: fx.Tensor,
    META: fx.Tensor,
    BLOCKS: fx.Tensor,
    MEMBERSHIP: fx.Tensor,
    COUNTS: fx.Tensor,
    COSTS: fx.Tensor,
    DIRECT_FLAG: fx.Tensor,
    MAX_BLOCKS: fx.Int32,
    CAPACITY: fx.Int32,
    RATIO: fx.Float32,
    DENSE_LIMIT: fx.Int32,
    LDS_BYTES: fx.Int32,
    MEMBERS64: fx.Constexpr[bool],
    TOPK: fx.Constexpr[int],
):
    _tile(
        ROWS,
        META,
        BLOCKS,
        MEMBERSHIP,
        COUNTS,
        COSTS,
        DIRECT_FLAG,
        MAX_BLOCKS,
        CAPACITY,
        RATIO,
        DENSE_LIMIT,
        LDS_BYTES,
        MEMBERS64,
        COMPACT_THREADS[1],
        TOPK,
    )


@flyc.jit
def _compact_launch(
    ROWS: fx.Tensor,
    META: fx.Tensor,
    BLOCKS: fx.Tensor,
    MEMBERSHIP: fx.Tensor,
    COUNTS: fx.Tensor,
    COSTS: fx.Tensor,
    DIRECT_FLAG: fx.Tensor,
    MAX_BLOCKS: fx.Int32,
    CAPACITY: fx.Int32,
    RATIO: fx.Float32,
    DENSE_LIMIT: fx.Int32,
    LDS_BYTES: fx.Int32,
    TILES: fx.Int32,
    MEMBERS64: fx.Constexpr[bool],
    LARGE: fx.Constexpr[bool],
    TOPK: fx.Constexpr[int],
    stream: fx.Stream,
):
    if TILES > 0:
        args = (
            ROWS,
            META,
            BLOCKS,
            MEMBERSHIP,
            COUNTS,
            COSTS,
            DIRECT_FLAG,
            MAX_BLOCKS,
            CAPACITY,
            RATIO,
            DENSE_LIMIT,
            LDS_BYTES,
            MEMBERS64,
            TOPK,
        )
        if fx.const_expr(LARGE):
            _compact_kernel_large(*args).launch(
                grid=(TILES, 1, 1),
                block=(COMPACT_THREADS[1], 1, 1),
                smem=LDS_BYTES,
                stream=stream,
            )
        else:
            _compact_kernel(*args).launch(
                grid=(TILES, 1, 1),
                block=(COMPACT_THREADS[0], 1, 1),
                smem=LDS_BYTES,
                stream=stream,
            )


def lds_layout(max_blocks, wide):
    """(LDS bytes, longest causal range in blocks compacted densely) of a K2 launch: one
    word per block when the plan's longest range fits LDS_BUDGET, else LDS_BUDGET (ranges
    up to it still dense, longer ones windowed).
    """
    dense = (-(-max_blocks // 64) * 64) * _entry_bytes(wide) + SCAN_BYTES
    if dense <= LDS_BUDGET:
        return dense, max_blocks
    return LDS_BUDGET, (LDS_BUDGET - SCAN_BYTES) // _entry_bytes(wide)


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


def _claim(flag, count):
    """Atomically add count to flag[1]; returns the previous value."""
    return fx.atomic_add(fx.get_iter(flag) + 1, fx.Int32(count), syncscope="agent")


@flyc.jit
def _direct_groups(
    COUNTS: fx.Tensor,
    COSTS: fx.Tensor,
    QUERY_TILES: fx.Tensor,
    GROUP_META: fx.Tensor,
    GROUP_LIST: fx.Tensor,
    DIRECT_FLAG: fx.Tensor,
    scratch,
    chunk,
    threshold,
    GROUPS: fx.Int32,
):
    """Append this CTA's raw direct row groups that hold a row of a direct tile to GROUP_LIST
    (count in DIRECT_FLAG[1]), in group order within the CTA."""
    tid = fx.thread_idx.x
    wave = _uniform(tid >> 6)
    group = chunk * THREADS + tid
    index = _min(group, GROUPS - 1)
    first, rows = GROUP_META[index * 5], GROUP_META[index * 5 + 1]
    direct = fx.Int32(0)
    # A raw group holds up to four rows; the route matches _order's.
    for local in fx.range_constexpr(4):
        tile = QUERY_TILES[first + _min(fx.Int32(local), rows - 1)]
        steps = (COUNTS[tile] + 15) >> 4
        active = (COSTS[tile * 3 + 2] == 1) & (steps <= threshold)
        direct = direct | active.select(fx.Int32(0), fx.Int32(1))
    listed = (group < GROUPS) & (direct != 0)
    mask = _ballot(listed)
    scratch[wave] = _count(mask)
    fx.gpu.barrier()
    totals = [scratch[w] for w in range(THREADS // 64)]
    total = totals[0]
    for w in fx.range_constexpr(1, THREADS // 64):
        total = total + totals[w]
    if tid == 0:
        scratch[THREADS // 64] = _claim(DIRECT_FLAG, total)
    fx.gpu.barrier()
    base = scratch[THREADS // 64]
    for w in fx.range_constexpr(THREADS // 64 - 1):
        base = base + (fx.Int32(w) < wave).select(totals[w], fx.Int32(0))
    if listed:
        GROUP_LIST[base + _below(mask)] = group


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
    SLOTS: fx.Tensor,
    POOL_BLOCKS: fx.Tensor,
    TAILS: fx.Tensor,
    task,
    QB: fx.Int32,
    CAP: fx.Int32,
    TILES: fx.Int32,
    SLOTTED: fx.Constexpr[bool],
):
    # Entry (nt, query, quarter) holds 4 bits for each of the slots nt*16 + quarter*2 +
    # (group//2)*8 + group%2. One lane owns an (nt, quarter) pair: its four slots are
    # loaded once and shared by every query. Each wave owns one (tile, part); a part
    # takes 16 of every 64 slot groups. With a slot table the same lanes map their
    # blocks to pool blocks (a four-token block occupies four aligned pool rows).
    lane = fx.thread_idx.x & 63
    tile = TILES - 1 - (task >> 2)
    if _uniform(COSTS[tile * 3 + 2]) != 0:
        count = _uniform(COUNTS[tile])
        rows = _uniform(META[tile * 5 + 1])
        length = _uniform(META[tile * 5 + 3])
        position = _uniform(META[tile * 5 + 4])
        kv_start = fx.Int32(0)
        if fx.const_expr(SLOTTED):
            kv_start = _uniform(META[tile * 5 + 2])
            if ((task & 3) == 0) & (lane == 0):
                # Pool row of the final partial block's first token, -1 without one.
                tail = fx.Int32(SLOTS[kv_start + (((length - 1) >> 2) << 2)])
                TAILS[tile] = ((length & 3) != 0).select(tail, fx.Int32(-1))
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
            if fx.const_expr(SLOTTED):
                for i in fx.range_constexpr(4):
                    o = (0, 1, 8, 9)[i]
                    if valid & (slot + o < count):
                        row = fx.Int32(SLOTS[kv_start + pairs[i][1]])
                        POOL_BLOCKS[base + o] = row >> 2
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


@flyc.kernel(name="sparse_prefill_order_masks", known_block_size=[THREADS, 1, 1])
def _order_masks_kernel(
    SLOTS: fx.Tensor,
    COUNTS: fx.Tensor,
    COSTS: fx.Tensor,
    ACTIVE: fx.Tensor,
    ORDER: fx.Tensor,
    DIRECT_FLAG: fx.Tensor,
    BLOCKS: fx.Tensor,
    MEMBERSHIP: fx.Tensor,
    META: fx.Tensor,
    MASKS: fx.Tensor,
    POOL_BLOCKS: fx.Tensor,
    TAILS: fx.Tensor,
    QUERY_TILES: fx.Tensor,
    GROUP_META: fx.Tensor,
    GROUP_LIST: fx.Tensor,
    TASKS: fx.Int32,
    GRID: fx.Int32,
    SIZE: fx.Int32,
    SHIFT: fx.Int32,
    ORDER_CTAS: fx.Int32,
    QB: fx.Int32,
    CAP: fx.Int32,
    TILES: fx.Int32,
    GROUPS: fx.Int32,
    TASK_SHARE: fx.Float32,
    DIRECT_STEP: fx.Float32,
    DIRECT_FIXED: fx.Float32,
    HK: fx.Constexpr[int],
    SLICES: fx.Constexpr[int],
    WIDE: fx.Constexpr[bool],
    ROUTE: fx.Constexpr[bool],
    ROUTE_MODEL: fx.Constexpr[tuple],
    SLOTTED: fx.Constexpr[bool],
):
    scratch = (
        fx.SharedAllocator()
        .allocate(fx.Array[fx.Int32, ORDER_WORDS, 16])
        .peek()
        .view(fx.make_layout(ORDER_WORDS, 1))
    )
    program = fx.block_idx.x
    # Order CTAs, then one mask CTA per tile, then the raw direct group-list CTAs.
    if (program < ORDER_CTAS) | (program >= ORDER_CTAS + TILES):
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
        if program < ORDER_CTAS:
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
            _direct_groups(
                COUNTS,
                COSTS,
                QUERY_TILES,
                GROUP_META,
                GROUP_LIST,
                DIRECT_FLAG,
                scratch,
                program - ORDER_CTAS - TILES,
                threshold,
                GROUPS,
            )
    else:
        # Mask CTAs pack one (tile, part) per wave.
        task = (program - ORDER_CTAS) * 4 + _uniform(fx.thread_idx.x >> 6)
        _masks(
            COUNTS,
            COSTS,
            META,
            BLOCKS,
            MEMBERSHIP,
            MASKS,
            SLOTS,
            POOL_BLOCKS,
            TAILS,
            task,
            QB,
            CAP,
            TILES,
            SLOTTED,
        )


@flyc.jit
def _order_masks_launch(
    SLOTS: fx.Tensor,
    COUNTS: fx.Tensor,
    COSTS: fx.Tensor,
    ACTIVE: fx.Tensor,
    ORDER: fx.Tensor,
    DIRECT_FLAG: fx.Tensor,
    BLOCKS: fx.Tensor,
    MEMBERSHIP: fx.Tensor,
    META: fx.Tensor,
    MASKS: fx.Tensor,
    POOL_BLOCKS: fx.Tensor,
    TAILS: fx.Tensor,
    QUERY_TILES: fx.Tensor,
    GROUP_META: fx.Tensor,
    GROUP_LIST: fx.Tensor,
    TASKS: fx.Int32,
    GRID: fx.Int32,
    SIZE: fx.Int32,
    SHIFT: fx.Int32,
    ORDER_CTAS: fx.Int32,
    QB: fx.Int32,
    CAP: fx.Int32,
    TILES: fx.Int32,
    GROUPS: fx.Int32,
    TASK_SHARE: fx.Float32,
    DIRECT_STEP: fx.Float32,
    DIRECT_FIXED: fx.Float32,
    HK: fx.Constexpr[int],
    SLICES: fx.Constexpr[int],
    WIDE: fx.Constexpr[bool],
    ROUTE: fx.Constexpr[bool],
    ROUTE_MODEL: fx.Constexpr[tuple],
    SLOTTED: fx.Constexpr[bool],
    stream: fx.Stream,
):
    if TILES > 0:
        _order_masks_kernel(
            SLOTS,
            COUNTS,
            COSTS,
            ACTIVE,
            ORDER,
            DIRECT_FLAG,
            BLOCKS,
            MEMBERSHIP,
            META,
            MASKS,
            POOL_BLOCKS,
            TAILS,
            QUERY_TILES,
            GROUP_META,
            GROUP_LIST,
            TASKS,
            GRID,
            SIZE,
            SHIFT,
            ORDER_CTAS,
            QB,
            CAP,
            TILES,
            GROUPS,
            TASK_SHARE,
            DIRECT_STEP,
            DIRECT_FIXED,
            HK,
            SLICES,
            WIDE,
            ROUTE,
            ROUTE_MODEL,
            SLOTTED,
        ).launch(
            grid=(ORDER_CTAS + TILES + (GROUPS + THREADS - 1) // THREADS, 1, 1),
            block=(THREADS, 1, 1),
            stream=stream,
        )


def _cdiv(a, b):
    return -(-a // b)


def upload(array, device) -> torch.Tensor:
    # Pinned and asynchronous: building a new layout never drains the stream.
    host = torch.from_numpy(np.ascontiguousarray(array, dtype=np.int32))
    return host.pin_memory().to(device, non_blocking=True)


def scratch_specs(plan: SparsePlan) -> list:
    """(field, shape, dtype) of the per-call scratch; every launch rewrites what it reads."""
    tiles, capacity = plan.num_tiles, plan.block_capacity
    members = torch.int64 if plan.query_tile > 32 else torch.int32
    specs = [
        ("blocks", (tiles, capacity), torch.int32),
        ("membership", (tiles, capacity), members),
        ("score_masks", (tiles, capacity // 16, plan.query_tile, 4), torch.int16),
    ]
    if plan.slotted:
        specs += [
            ("pool_blocks", (tiles, capacity), torch.int32),
            ("tails", (tiles,), torch.int32),
        ]
    return specs


def allocate_scratch(owner, specs, device) -> None:
    for name, shape, dtype in specs:
        setattr(owner, name, torch.empty(shape, dtype=dtype, device=device))


def wide_tile_rows(query_tile: int) -> int:
    """Fewest full-prefill rows that take the BQ21/BQ42 tile."""
    return math.ceil(query_tile * get_cu_num() * platform().wide_tiles_per_cu)


def allocate_plan(
    *, inputs, query_tile: int, scratch: bool = True, slotted: bool = False
) -> SparsePlan:
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
        blocks=None,
        membership=None,
        score_masks=None,
        counts=torch.empty(tiles, **kwargs),
        costs=torch.empty((tiles, 3), **kwargs),
        active=torch.empty(tiles, **kwargs),
        task_order=torch.empty(_cdiv(tasks, grid) * grid if grid else 0, **kwargs),
        direct_flag=torch.empty(2, **kwargs),
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
        slotted=slotted,
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
    """(compiled, args) of K1-K3 in launch order; K1's first argument is the indices, K3's
    the slot table (a stand-in without one)."""
    args = (
        inputs.indices,
        inputs.query_positions,
        inputs.block_indices.view(-1),
        inputs.q.shape[0],
        inputs.topk,
        stream,
    )
    yield _compiled(_recover_launch, args, 3, (inputs.topk,)), args
    if plan is None or plan.num_tiles == 0:
        return
    members64 = plan.query_tile > 32
    # BQ > 21 tiles take the larger compact CTA.
    large = plan.query_tile > 21
    lds_bytes, dense_limit = lds_layout(plan.max_blocks, members64)
    args = (
        inputs.block_indices.view(-1),
        plan.metadata.view(-1),
        plan.blocks.view(-1),
        plan.membership.view(torch.int32).view(-1),
        plan.counts,
        plan.costs.view(-1),
        plan.direct_flag,
        plan.max_blocks,
        plan.block_capacity,
        plan.union_ratio,
        dense_limit,
        lds_bytes,
        plan.num_tiles,
        members64,
        large,
        inputs.topk,
        stream,
    )
    yield _compiled(_compact_launch, args, 12, (members64, large, inputs.topk)), args
    hk = inputs.k.shape[1]
    slices = _cdiv(plan.query_tile * plan.group_padded, 128)
    tasks = plan.num_tiles * hk * slices
    numel = plan.task_order.numel()
    # One order CTA sorts the whole task order up to the widest variant, else chunks of it.
    size = min(THREADS * SORT_WIDTHS[-1], numel)
    shift = max(1, (tasks - 1).bit_length())
    wide = ((plan.block_capacity // 16) << shift) + tasks - 1 >= 2**31
    standin = plan.counts  # slot-table tensors K3 does not read without a slot table
    args = (
        inputs.kv_slots if plan.slotted else standin,
        plan.counts,
        plan.costs.view(-1),
        plan.active,
        plan.task_order,
        plan.direct_flag,
        plan.blocks.view(-1),
        plan.membership.view(-1),
        plan.metadata.view(-1),
        plan.score_masks.view(-1),
        plan.pool_blocks.view(-1) if plan.slotted else standin,
        plan.tails if plan.slotted else standin,
        plan.query_tiles,
        plan.direct_meta.view(-1) if plan.direct_groups else standin,
        plan.direct_list if plan.direct_groups else standin,
        tasks,
        plan.grid,
        size,
        shift,
        _cdiv(numel, size),
        plan.query_tile,
        plan.block_capacity,
        plan.num_tiles,
        plan.direct_groups,
        *plan.route_costs,
        hk,
        slices,
        wide,
        plan.routing,
        plan.route_model,
        plan.slotted,
        stream,
    )
    constants = (
        hk,
        slices,
        wide,
        plan.routing,
        plan.route_model,
        members64,
        plan.slotted,
    )
    yield _compiled(_order_masks_launch, args, 22, constants), args


def run(*, inputs, plan):
    stream = torch.cuda.current_stream(inputs.q.device)
    for compiled, args in _launches(inputs, plan, stream):
        compiled(*args)


def launchers(*, inputs, plan):
    """K1-K3 launches with this binding's other arguments prebuilt. K1 takes the indices,
    K2 and K3 the slot table (read by K3 of a slot-table plan, ignored otherwise).

    Valid while the plan's scratch bindings and the current stream stay the same.
    """
    stream = torch.cuda.current_stream(inputs.q.device)
    (recover, args), *rest = _launches(inputs, plan, stream)
    tail = args[1:]
    result = [lambda indices: recover(indices, *tail)]
    if rest:
        (compact, compact_args), (masks, mask_args) = rest
        mask_tail = mask_args[1:]
        result.append(lambda slots: compact(*compact_args))
        if plan.slotted:
            result.append(lambda slots: masks(slots, *mask_tail))
        else:
            result.append(lambda slots: masks(*mask_args))
    return result
