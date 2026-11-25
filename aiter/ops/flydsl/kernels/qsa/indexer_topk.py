# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Exact per-row top-512 of QSA indexer logits for gfx942 (FlyDSL), expanded to request-local token ids.

Prefill: one wave owns one query row (four rows per 256-thread CTA, no CTA barriers), and the logits
kernel supplies the row's (min, max) bits. Decode: one 8-wave CTA owns a row; waves take contiguous
slot-group ranges and per-wave histogram counts place their outputs, so results match the one-wave
selection. Block j is lane j % 64 of slot j / 64; every pass sweeps wave-uniform slot groups with
UNROLL independent coalesced loads in flight. Fast path: one 256-bin histogram of a monotone linear
digit between the row min and max; blocks above the threshold bin are kept and the <= 64
threshold-bin keys are ranked exactly. Otherwise an ordered-key radix select finds the exact 512th
key. Ties keep the lowest block ids. The two ascending runs (kept, then resolving picks) are merged:
rows hold ascending blocks, 4 tokens each, then the causal tail, then -1. Logits rows wider than 65536
keys select with int32 instead of uint16 chosen block ids (the WIDE variants, compiled with the narrow
ones on first use).
"""

from types import SimpleNamespace

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import llvm
from flydsl.expr import gpu, rocdl

from aiter.ops.flydsl.kernels.qsa.sparse_prefill_common import (
    _buffer,
    _buffer_store_words,
    _buffer_words,
    _lds_load,
    _lds_pointer,
    _lds_store,
    _min,
    _uniform,
)

WAVES, TOPK, RATIO = 4, 512, 4
WIDTH = TOPK * RATIO + RATIO - 1
BINS, CANDIDATES, UNROLL = 256, 64, 8
# Keys per slot group; prefill logits are stored in key blocks of this size.
GROUP_KEYS = 64 * UNROLL
# Rows of more keys need the WIDE variants: int32 instead of uint16 chosen block ids.
NARROW_KEYS = 1 << 16
# Prefill rows at least this wide are SPLIT over a CTA's four waves instead of one wave each.
SPLIT_KEYS = 16384
# Per wave: 257 histogram ints (sink bin last; later 64 candidate keys, then their ids, then the
# merged ids), and the 512 chosen block ids.
SCRATCH = 1040
HIST = (BINS + 1) * 4
KEYS, IDS = 0, 64 * 4
SIGN = -(1 << 31)


def _lds(address):
    return _lds_pointer(address, fx.Int32)


def _load(address):
    return _lds_load(address)


def _store(address, value):
    _lds_store(address, value)


def _load16(address):
    # generic_load wraps signless i16 as Int16; block IDs need zero extension.
    return fx.Int32(_lds_load(address, fx.Uint16)) & 0xFFFF


def _store16(address, value):
    _lds_store(address, value, fx.Uint16)


def _wave_bytes(wide):
    """(scratch, chosen) LDS bytes of a wave-private row; the merged ids reuse the scratch."""
    chosen = TOPK * (4 if wide else 2)
    return max(SCRATCH, chosen), chosen


def _shared_row(waves):
    """LDS offsets (reduction words, candidates, chosen ids) of a row shared by `waves` waves, after their
    histograms."""
    red = waves * HIST
    cand = red + waves * 16
    return red, cand, cand + 2 * CANDIDATES * 4


def _kernel_bytes(wide, split):
    scratch, chosen = _wave_bytes(wide)
    return _shared_row(WAVES)[2] + chosen if split else WAVES * (scratch + chosen)


def _store_id(ids, index, value):
    if ids.wide:
        _store(ids.base + index * 4, value)
    else:
        _store16(ids.base + index * 2, value)


def _load_id(ids, index):
    return _load(ids.base + index * 4) if ids.wide else _load16(ids.base + index * 2)


def _add(address):
    fx.atomic_add(_lds(address), fx.Int32(1), syncscope="wavefront")


def _put(output, index, value):
    _buffer_store_words(
        fx.Vector.from_elements([fx.Int32(value)], fx.Int32),
        output,
        fx.Int32(index) * 4,
    )


def _sync():
    fx.memory_fence(syncscope="wavefront")
    # A workgroup barrier cannot replace this wave-only rendezvous.
    rocdl.wave_barrier()


def _intrinsic(name, dtype, *args):
    return dtype(
        llvm.call_intrinsic(dtype.ir_type, name, [a.ir_value() for a in args], [], [])
    )


def _ballot(predicate):
    return fx.Int64(rocdl.ballot(fx.Int64.ir_type, predicate.ir_value()))


def _below(mask):
    low = rocdl.mbcnt_lo(
        fx.Int32.ir_type, fx.Int32(mask).ir_value(), fx.Int32(0).ir_value()
    )
    return fx.Int32(
        rocdl.mbcnt_hi(fx.Int32.ir_type, fx.Int32(mask >> 32).ir_value(), low)
    )


def _popcount(mask):
    return fx.Int32(_intrinsic("llvm.ctpop.i64", fx.Int64, mask))


def _readlane(value, lane):
    return fx.Int32(
        rocdl.readlane(
            fx.Int32.ir_type, fx.Int32(value).ir_value(), fx.Int32(lane).ir_value()
        )
    )


def _minnum(a, b):
    return _intrinsic("llvm.minnum.f32", fx.Float32, a, b)


def _maxnum(a, b):
    return _intrinsic("llvm.maxnum.f32", fx.Float32, a, b)


def _umin(a, b):
    return _intrinsic("llvm.umin.i32", fx.Int32, a, b)


def _umax(a, b):
    return _intrinsic("llvm.umax.i32", fx.Int32, a, b)


def _ugt(a, b):
    return (a ^ SIGN) > (b ^ SIGN)


def _uge(a, b):
    return (a ^ SIGN) >= (b ^ SIGN)


def _order_key(x):
    bits = fx.Float32(x).bitcast(fx.Int32)
    return (bits < 0).select(~bits, bits | SIGN)


def _finite(x):
    return (fx.Float32(x).bitcast(fx.Int32) & 0x7F800000) != 0x7F800000


def _reduce(value, op):
    for offset in (32, 16, 8, 4, 2, 1):
        value = op(value, value.shuffle_xor(offset, 64))
    return value


def _group(logits, base, group, lane):
    # One coalesced scalar load per slot; buffer OOB reads return zero.
    offset = base + group * logits.step + lane * 4
    return fx.Vector.from_elements(
        [
            _buffer_words(logits.resource, offset + i * 256, words=1)[0].bitcast(
                fx.Float32
            )
            for i in range(UNROLL)
        ],
        fx.Float32,
    )


def _groups(count):
    return (count + GROUP_KEYS - 1) // GROUP_KEYS


def _slots(value, group, lane, count):
    for i in range(UNROLL):
        j = (group * UNROLL + i) * 64 + lane
        yield j, value[i], j < count


def _digit(x, low, scale):
    digit = ((x - low) * scale).to(fx.Int32)
    return (digit < BINS - 1).select(digit, fx.Int32(BINS - 1))


def _clear(bins, lane):
    for i in range(0, BINS + 1, 64):
        _store(bins + (i + lane if i < BINS else i) * 4, 0)


def _inclusive_scan(value, lane):
    for offset in (1, 2, 4, 8, 16, 32):
        source = ((lane - offset) & 63) * 4
        other = fx.Int32(
            rocdl.ds_bpermute(fx.Int32.ir_type, source.ir_value(), value.ir_value())
        )
        value = (lane >= offset).select(value + other, value)
    return value


def _threshold(bins, need, lane):
    """(bin, count above it, its size) for the bin holding the need-th largest element."""
    return _threshold_owned(
        [_load(bins + (lane * (BINS // 64) + i) * 4) for i in range(BINS // 64)],
        need,
        lane,
    )


def _threshold_owned(owned, need, lane):
    inclusive = _inclusive_scan(owned[0] + owned[1] + owned[2] + owned[3], lane)
    higher = _readlane(inclusive, 63) - inclusive
    found, found_above, found_size = fx.Int32(-1), fx.Int32(0), fx.Int32(0)
    for i in reversed(range(BINS // 64)):
        hit = (higher < need) & (higher + owned[i] >= need)
        found = hit.select(lane * (BINS // 64) + i, found)
        found_above = hit.select(higher, found_above)
        found_size = hit.select(owned[i], found_size)
        higher = higher + owned[i]
    source = fx.Int32(
        _intrinsic("llvm.cttz.i64", fx.Int64, _ballot(found >= 0), fx.Boolean(False))
    )
    return (
        _readlane(found, source),
        _readlane(found_above, source),
        _readlane(found_size, source),
    )


def _minmax_group(value, group, lane, count, low, high, nan):
    for _, x, live in _slots(value, group, lane, count):
        low = _minnum(low, live.select(x, fx.Float32(float("inf"))))
        high = _maxnum(high, live.select(x, fx.Float32(float("-inf"))))
        is_nan = x != x  # noqa: PLR0124 -- IEEE NaN predicate on GPU SSA.
        nan = nan | (live & is_nan).select(fx.Int32(1), fx.Int32(0))
    return low, high, nan


@flyc.jit
def _sweep_minmax(logits, base, count, lane, first, last):
    low, high, nan = fx.Float32(float("inf")), fx.Float32(float("-inf")), fx.Int32(0)
    following = _group(logits, base, _min(first, last - 1), lane)
    for index in range(first, last, fx.Int32(1)):
        group = fx.Int32(index)
        value = following
        if group + 1 < last:
            following = _group(logits, base, group + 1, lane)
        low, high, nan = _minmax_group(value, group, lane, count, low, high, nan)
    return low, high, nan


def _hist_group(value, group, lane, count, bins, low, scale):
    for _, x, live in _slots(value, group, lane, count):
        _add(bins + live.select(_digit(x, low, scale), fx.Int32(BINS)) * 4)


@flyc.jit
def _sweep_hist(logits, base, count, lane, bins, low, scale, first, last):
    following = _group(logits, base, _min(first, last - 1), lane)
    for index in range(first, last, fx.Int32(1)):
        group = fx.Int32(index)
        value = following
        if group + 1 < last:
            following = _group(logits, base, group + 1, lane)
        _hist_group(value, group, lane, count, bins, low, scale)


def _gather_group(
    value, group, lane, count, bins, chosen, low, scale, bin, kept, gathered
):
    for j, x, live in _slots(value, group, lane, count):
        digit = live.select(_digit(x, low, scale), fx.Int32(-1))
        over, inside = digit > bin, digit == bin
        over_mask, inside_mask = _ballot(over), _ballot(inside)
        _keep(over, chosen, kept, over_mask, j)
        _collect(inside, bins, gathered, inside_mask, x, j)
        kept = kept + _popcount(over_mask)
        gathered = gathered + _popcount(inside_mask)
    return kept, gathered


@flyc.jit
def _keep(condition, chosen, base, mask, j):
    if condition:
        _store_id(chosen, base + _below(mask), j)


@flyc.jit
def _collect(condition, bins, base, mask, x, j):
    if condition:
        slot = base + _below(mask)
        _store(bins + KEYS + slot * 4, _order_key(x))
        _store(bins + IDS + slot * 4, j)


@flyc.jit
def _keep_tie(condition, chosen, tied_base, ties, mask, need, j):
    if condition:
        rank = ties + _below(mask)
        if rank < need:
            _store_id(chosen, tied_base + rank, j)


@flyc.jit
def _sweep_gather(
    logits,
    base,
    count,
    lane,
    bins,
    chosen,
    low,
    scale,
    bin,
    first,
    last,
    kept,
    gathered,
):
    following = _group(logits, base, _min(first, last - 1), lane)
    for index in range(first, last, fx.Int32(1)):
        group = fx.Int32(index)
        value = following
        if group + 1 < last:
            following = _group(logits, base, group + 1, lane)
        kept, gathered = _gather_group(
            value, group, lane, count, bins, chosen, low, scale, bin, kept, gathered
        )
    return kept, gathered


@flyc.jit
def _rank(bins, chosen, lane, size, above):
    # Candidates are in ascending id order: rank by key, ties by lower id.
    key = _load(bins + KEYS + lane * 4)
    rank = fx.Int32(0)
    for index in range(fx.Int32(0), size, fx.Int32(1)):
        m = fx.Int32(index)
        other = _load(bins + KEYS + m * 4)
        rank = rank + (_ugt(other, key) | ((other == key) & (m < lane))).select(
            fx.Int32(1), fx.Int32(0)
        )
    rank = (lane < size).select(rank, fx.Int32(CANDIDATES))
    picked = rank < TOPK - above
    _keep_id(picked, chosen, above, _ballot(picked), bins + IDS + lane * 4)


@flyc.jit
def _keep_id(condition, chosen, base, mask, source):
    if condition:
        _store_id(chosen, base + _below(mask), _load(source))


def _keys_group(value, group, lane, count, high_key, low_key):
    for _, x, live in _slots(value, group, lane, count):
        key = _order_key(x)
        high_key = _umax(high_key, live.select(key, fx.Int32(0)))
        low_key = _umin(low_key, live.select(key, fx.Int32(-1)))
    return high_key, low_key


@flyc.jit
def _sweep_keys(logits, base, count, lane, first, last):
    high_key, low_key = fx.Int32(0), fx.Int32(-1)
    following = _group(logits, base, _min(first, last - 1), lane)
    for index in range(first, last, fx.Int32(1)):
        group = fx.Int32(index)
        value = following
        if group + 1 < last:
            following = _group(logits, base, group + 1, lane)
        high_key, low_key = _keys_group(value, group, lane, count, high_key, low_key)
    return high_key, low_key


def _radix_group(value, group, lane, count, bins, prefix, prefix_mask, shift, width):
    for _, x, live in _slots(value, group, lane, count):
        key = _order_key(x)
        hit = live & ((key & prefix_mask) == prefix)
        _add(
            bins
            + hit.select((key >> shift) & ((fx.Int32(1) << width) - 1), fx.Int32(BINS))
            * 4
        )


@flyc.jit
def _sweep_radix(
    logits, base, count, lane, bins, prefix, prefix_mask, shift, width, first, last
):
    following = _group(logits, base, _min(first, last - 1), lane)
    for index in range(first, last, fx.Int32(1)):
        group = fx.Int32(index)
        value = following
        if group + 1 < last:
            following = _group(logits, base, group + 1, lane)
        _radix_group(value, group, lane, count, bins, prefix, prefix_mask, shift, width)


def _select_group(
    value, group, lane, count, chosen, prefix, prefix_mask, whole, need, kept, ties
):
    tied_base = TOPK - need
    for j, x, live in _slots(value, group, lane, count):
        key = _order_key(x)
        over = live & (whole != 0).select(
            _uge(key & prefix_mask, prefix), _ugt(key, prefix)
        )
        tie = (whole == 0) & live & (key == prefix)
        over_mask, tie_mask = _ballot(over), _ballot(tie)
        _keep(over, chosen, kept, over_mask, j)
        _keep_tie(tie, chosen, tied_base, ties, tie_mask, need, j)
        kept = kept + _popcount(over_mask)
        ties = ties + _popcount(tie_mask)
    return kept, ties


@flyc.jit
def _sweep_select(
    logits,
    base,
    count,
    lane,
    chosen,
    prefix,
    prefix_mask,
    whole,
    need,
    first,
    last,
    kept,
    ties,
):
    following = _group(logits, base, _min(first, last - 1), lane)
    for index in range(first, last, fx.Int32(1)):
        group = fx.Int32(index)
        value = following
        if group + 1 < last:
            following = _group(logits, base, group + 1, lane)
        kept, ties = _select_group(
            value,
            group,
            lane,
            count,
            chosen,
            prefix,
            prefix_mask,
            whole,
            need,
            kept,
            ties,
        )


def _key_range(high_key, low_key):
    """(top differing bit or -1, prefix mask, prefix) shared by all live ordered keys."""
    differ = high_key ^ low_key
    top = (differ == 0).select(
        fx.Int32(-1),
        31 - _intrinsic("llvm.ctlz.i32", fx.Int32, differ, fx.Boolean(False)),
    )
    shifted = (fx.Int32(2) << (top & 31)) - 1
    prefix_mask = (top < 0).select(
        fx.Int32(-1), (top == 31).select(fx.Int32(0), ~shifted)
    )
    return top, prefix_mask, high_key & prefix_mask


def _space(
    waves, wave, lane, hist, cand, chosen, merged, wide, hist_base=None, red=None
):
    """LDS of one row's selection, shared by `waves` waves (a trace-time int; 1 = wave-private row).

    hist: this wave's 257-int histogram (wave w at hist_base + w * HIST when waves > 1); cand: 64
    candidate keys, then their ids (may alias hist); chosen: 512 block ids (uint16, int32 if wide);
    merged: finished scratch of the same size for the ascending merge; red: 4 reduction words per
    wave (waves > 1).
    """
    return SimpleNamespace(
        waves=waves,
        wave=wave,
        lane=lane,
        thread=lane if waves == 1 else wave * 64 + lane,
        threads=64 * waves,
        hist=hist,
        cand=cand,
        chosen=SimpleNamespace(base=chosen, wide=wide),
        merged=SimpleNamespace(base=merged, wide=wide),
        hist_base=hist_base,
        red=red,
    )


def _sync_of(row):
    return _sync if row.waves == 1 else _cta_sync


def _row_sync(row):
    _sync_of(row)()


def _span(row, count):
    """This wave's slot groups [first, last): all of them, or a contiguous 1/waves share."""
    groups = _groups(count)
    if row.waves == 1:
        return fx.Int32(0), groups
    per = (groups + row.waves - 1) // row.waves
    first = _min(row.wave * per, groups)
    return first, _min(first + per, groups)


def _exchange(row, values):
    """Per-wave values of every wave of the row (after one CTA barrier)."""
    for slot, value in enumerate(values):
        _store(row.red + row.wave * 16 + slot * 4, value)
    _cta_sync()
    return [
        [_load(row.red + w * 16 + slot * 4) for w in range(row.waves)]
        for slot in range(len(values))
    ]


def _below_wave(values, wave):
    total = fx.Int32(0)
    for w, value in enumerate(values):
        total = total + (fx.Int32(w) < wave).select(value, fx.Int32(0))
    return total


def _bases(row, above, inside):
    """Where this wave's outputs start: exclusive prefix over the row's waves."""
    if row.waves == 1:
        return fx.Int32(0), fx.Int32(0)
    aboves, insides = _exchange(row, [above, inside])
    return _below_wave(aboves, row.wave), _below_wave(insides, row.wave)


def _row_minmax(row, low, high, nan):
    """Row (min, max, any NaN) from this wave's partial sweep."""
    low, high = _reduce(low, _minnum), _reduce(high, _maxnum)
    nan = (_ballot(nan != 0) != 0).select(fx.Int32(1), fx.Int32(0))
    if row.waves == 1:
        return low, high, nan
    lows, highs, nans = _exchange(
        row, [low.bitcast(fx.Int32), high.bitcast(fx.Int32), nan]
    )
    low, high, nan = fx.Float32(float("inf")), fx.Float32(float("-inf")), fx.Int32(0)
    for value in lows:
        low = _minnum(low, value.bitcast(fx.Float32))
    for value in highs:
        high = _maxnum(high, value.bitcast(fx.Float32))
    for value in nans:
        nan = nan | value
    return low, high, nan


def _key_bounds(row, high_key, low_key):
    high_key, low_key = _reduce(high_key, _umax), _reduce(low_key, _umin)
    if row.waves == 1:
        return high_key, low_key
    highs, lows = _exchange(row, [high_key, low_key])
    high_key, low_key = fx.Int32(0), fx.Int32(-1)
    for value in highs:
        high_key = _umax(high_key, value)
    for value in lows:
        low_key = _umin(low_key, value)
    return high_key, low_key


def _row_threshold(row, need):
    """(bin, count above it, its size) of the need-th largest element over the row's waves."""
    if row.waves == 1:
        return _threshold(row.hist, need, row.lane)
    owned = []
    for i in range(BINS // 64):
        total = fx.Int32(0)
        for w in range(row.waves):
            total = total + _load(
                row.hist_base + w * HIST + (row.lane * (BINS // 64) + i) * 4
            )
        owned.append(total)
    return _threshold_owned(owned, need, row.lane)


def _split(row, bin):
    """This wave's (count above bin, count in bin)."""
    above = fx.Int32(0)
    for i in range(BINS // 64):
        index = row.lane * (BINS // 64) + i
        above = above + (index > bin).select(_load(row.hist + index * 4), fx.Int32(0))
    return _reduce(above, lambda a, b: a + b), _load(row.hist + bin * 4)


def _gather_bases(row, bin):
    if row.waves == 1:
        _sync()
        return fx.Int32(0), fx.Int32(0)
    return _bases(row, *_split(row, bin))


@flyc.jit
def _radix(logits, base, count, row):
    # Ordered-key radix: live keys share bits above `top`; <= 8-bit digits refine the 512th key.
    first, last = _span(row, count)
    high_key, low_key = _sweep_keys(logits, base, count, row.lane, first, last)
    top, prefix_mask, prefix = _key_range(*_key_bounds(row, high_key, low_key))
    need, whole = fx.Int32(TOPK), fx.Int32(0)
    # This wave's keys above the resolved prefix, and in its current bin (all live keys at first).
    mine = fx.Int32(0)
    inside = _min(last * (64 * UNROLL), count) - first * (64 * UNROLL)
    inside = (inside > 0).select(inside, fx.Int32(0))
    while (top >= 0) & (whole == 0):
        width = (top + 1 < 8).select(top + 1, fx.Int32(8))
        shift = top - width + 1
        _row_sync(row)
        _clear(row.hist, row.lane)
        _sync()
        _sweep_radix(
            logits,
            base,
            count,
            row.lane,
            row.hist,
            prefix,
            prefix_mask,
            shift,
            width,
            first,
            last,
        )
        _row_sync(row)
        bin, above, size = _row_threshold(row, need)
        higher, inside = _split(row, bin)
        mine = mine + higher
        need = need - above
        prefix = prefix | (bin << shift)
        prefix_mask = prefix_mask | (((fx.Int32(1) << width) - 1) << shift)
        whole = (size == need).select(fx.Int32(1), fx.Int32(0))
        top = (size == need).select(top, shift - 1)
    # Keys above the 512th key ascending, then the kept ties ascending.
    _row_sync(row)
    kept, ties = _bases(
        row,
        mine + (whole != 0).select(inside, fx.Int32(0)),
        (whole != 0).select(fx.Int32(0), inside),
    )
    _sweep_select(
        logits,
        base,
        count,
        row.lane,
        row.chosen,
        prefix,
        prefix_mask,
        whole,
        need,
        first,
        last,
        kept,
        ties,
    )
    return TOPK - need


def _max(a, b):
    return (a > b).select(a, b)


def _emit_ascending(output, chosen, merged, split, lane, threads, sync):
    # chosen[:split] (A) and chosen[split:] (B) are ascending. Merge path: each thread binary
    # searches where its output diagonal crosses the runs, merges its outputs into LDS `merged`
    # (finished scratch), and coalesced 16-byte stores then write four tokens per block.
    per = TOPK // threads
    diagonal = lane * per
    count = fx.Int32(TOPK) - split
    low = _max(diagonal - count, fx.Int32(0))
    high = _min(diagonal, split)
    for _ in range(10):
        active = low < high
        mid = (low + high) >> 1
        a = _load_id(chosen, _min(mid, fx.Int32(TOPK - 1)))
        b = _load_id(
            chosen,
            _min(_max(split + diagonal - mid - 1, fx.Int32(0)), fx.Int32(TOPK - 1)),
        )
        right = a < b
        low = (active & right).select(mid + 1, low)
        high = active.select(right.select(high, mid), high)
    i, j = low, diagonal - low
    # Above every block id of the variant.
    end = fx.Int32(1 << (30 if chosen.wide else 16))
    for k in range(per):
        a = (i < split).select(_load_id(chosen, _min(i, fx.Int32(TOPK - 1))), end)
        b = (j < count).select(
            _load_id(chosen, _min(split + j, fx.Int32(TOPK - 1))), end
        )
        first = a < b
        _store_id(merged, diagonal + k, first.select(a, b))
        i = first.select(i + 1, i)
        j = first.select(j, j + 1)
    sync()
    for k in range(TOPK // threads):
        index = lane + k * threads
        value = _load_id(merged, index)
        tokens = fx.Vector.from_elements(
            [value * RATIO + r for r in range(RATIO)], fx.Int32
        )
        _buffer_store_words(tokens, output, index * (RATIO * 4))


@flyc.jit
def _rank_row(row, size, above):
    if fx.const_expr(row.waves == 1):
        _rank(row.cand, row.chosen, row.lane, size, above)
    else:
        if row.wave == 0:
            _rank(row.cand, row.chosen, row.lane, size, above)


@flyc.jit
def _select(logits, base, count, row, low, high, nan):
    """Top-512 block ids of a row whose logits lie in [low, high] (nan: any NaN) into row.chosen, as
    two ascending runs; returns where the second run starts."""
    first, last = _span(row, count)
    fast, scale = _fast(low, high, nan)
    done = fx.Int32(0)
    split = fx.Int32(0)
    if fast:
        # x <= y implies digit(x) <= digit(y): bins partition blocks by value.
        _clear(row.hist, row.lane)
        _sync()
        _sweep_hist(logits, base, count, row.lane, row.hist, low, scale, first, last)
        _row_sync(row)
        bin, above, size = _row_threshold(row, fx.Int32(TOPK))
        if size <= CANDIDATES:
            kept, gathered = _gather_bases(row, bin)
            _sweep_gather(
                logits,
                base,
                count,
                row.lane,
                row.cand,
                row.chosen,
                low,
                scale,
                bin,
                first,
                last,
                kept,
                gathered,
            )
            _row_sync(row)
            _rank_row(row, size, above)
            done = fx.Int32(1)
            split = above
    if done == 0:
        split = _radix(logits, base, count, row)
    return split


def _fast(low, high, nan):
    """(whether the 256-bin fast path applies to logits in [low, high], its digit scale)."""
    scale = fx.Float32(float(BINS)) / (high - low)
    return (nan == 0) & _finite(low) & _finite(high) & (high > low) & _finite(
        scale
    ), scale


def _row_stats(STATS, index):
    """(low, high, any NaN) from the logits kernel's (min bits, ~max bits) of logits >= 0 over a superset
    of the row's causal keys: any NaN orders above +inf. Wider bounds keep the digit monotone, so
    selection stays exact."""
    low = fx.Int32(_uniform(STATS[2 * index]))
    high = ~fx.Int32(_uniform(STATS[2 * index + 1]))
    return (
        low.bitcast(fx.Float32),
        high.bitcast(fx.Float32),
        _ugt(high, fx.Int32(0x7F800000)).select(fx.Int32(1), fx.Int32(0)),
    )


def _bounds(logits, base, count, row, STATS, index):
    """The row's (low, high, any NaN): from the logits kernel's stats if given, else swept here."""
    if STATS is None:
        first, last = _span(row, count)
        return _row_minmax(
            row, *_sweep_minmax(logits, base, count, row.lane, first, last)
        )
    return _row_stats(STATS, index)


@flyc.jit
def _write_row(
    LOGITS,
    row_bytes,
    step,
    local,
    out_row,
    count,
    visible,
    length,
    OUT,
    row,
    STATS,
    long=None,
):
    # Logits row `local` (starting local * row_bytes bytes in, slot groups `step` bytes apart) selects
    # output row `out_row`; `long` holds the part CTAs' results of a multi-CTA decode row.
    visible = fx.max(_min(visible, length), fx.Int32(0))
    output = _buffer(
        fx.make_view(
            fx.get_iter(OUT) + fx.Int64(out_row) * WIDTH, fx.make_layout(1, 1)
        ),
        WIDTH * 4,
    )
    blocks = count
    if count <= TOPK:
        for token in range(row.thread, count * RATIO, fx.Int32(row.threads)):
            _put(output, token, token)
    else:
        blocks = fx.Int32(TOPK)
        logits = SimpleNamespace(
            resource=_buffer(LOGITS, fx.Int32(0x7FFFFFF0)), step=step
        )
        base = local * row_bytes
        if fx.const_expr(long is None):
            low, high, nan = _bounds(logits, base, count, row, STATS, out_row)
            split = _select(logits, base, count, row, low, high, nan)
        else:
            low, high, nan = _slot_stats(STATS, out_row, row.lane)
            split = _select_long(
                logits, base, count, row, low, high, nan, long, out_row
            )
        _row_sync(row)
        _emit_ascending(
            output,
            row.chosen,
            row.merged,
            split,
            row.thread,
            row.threads,
            _sync_of(row),
        )
    tail_start = visible // RATIO * RATIO
    tail_count = visible - tail_start
    for index in range(
        blocks * RATIO + row.thread, fx.Int32(WIDTH), fx.Int32(row.threads)
    ):
        column = fx.Int32(index)
        offset = column - blocks * RATIO
        token = tail_start + offset
        value = (
            (offset < RATIO - 1) & (offset < tail_count) & (token < length)
        ).select(token, fx.Int32(-1))
        _put(output, column, value)


@flyc.jit
def _row(
    LOGITS, block_stride, pitch, row0, POSITIONS, ROW_INFO, OUT, STATS, local, row
):
    index = row0 + local
    expected, length = _uniform(ROW_INFO[2 * index]), _uniform(ROW_INFO[2 * index + 1])
    if fx.Int64(POSITIONS[index]) != fx.Int64(expected):
        # s_trap 2: the queue enters the error state and the process aborts.
        llvm.intr_trap()
    visible = expected + 1
    count = visible // RATIO
    count = (count < length // RATIO).select(count, length // RATIO)
    _write_row(
        LOGITS,
        pitch * 4,
        block_stride * 4,
        local,
        index,
        count,
        visible,
        length,
        OUT,
        row,
        STATS,
    )


@flyc.kernel(name="qsa_indexer_topk", known_block_size=[256, 1, 1])
def _kernel(
    LOGITS: fx.Tensor,
    block_stride: fx.Int32,
    pitch: fx.Int32,
    row0: fx.Int32,
    rows: fx.Int32,
    POSITIONS: fx.Tensor,
    ROW_INFO: fx.Tensor,
    OUT: fx.Tensor,
    STATS: fx.Tensor,
    WIDE: fx.Constexpr[bool],
    SPLIT: fx.Constexpr[bool],
):
    scratch, chosen = _wave_bytes(WIDE)
    size = _kernel_bytes(WIDE, SPLIT)
    storage = fx.SharedAllocator().allocate(fx.Array[fx.Int8, size, 16]).peek()
    shared = fx.Int32(fx.ptrtoint(fx.get_iter(storage.view(fx.make_layout(size, 1)))))
    tid = fx.thread_idx.x
    wave, lane = _uniform(tid >> 6), tid & 63
    if fx.const_expr(SPLIT):
        # The four waves share one row, like decode; the merged ids reuse the histograms.
        local = fx.block_idx.x
        red, cand, ids = _shared_row(WAVES)
        row = _space(
            WAVES,
            wave,
            lane,
            shared + wave * HIST,
            shared + cand,
            shared + ids,
            shared,
            WIDE,
            hist_base=shared,
            red=shared + red,
        )
    else:
        local = _uniform(fx.block_idx.x * WAVES + wave)
        # One wave per row: the histogram scratch also holds the candidates and the merge.
        bins = shared + wave * scratch
        row = _space(
            1,
            wave,
            lane,
            bins,
            bins,
            shared + WAVES * scratch + wave * chosen,
            bins,
            WIDE,
        )
    if local < rows:
        _row(
            LOGITS,
            block_stride,
            pitch,
            row0,
            POSITIONS,
            ROW_INFO,
            OUT,
            STATS,
            local,
            row,
        )


@flyc.jit
def _launch(
    LOGITS: fx.Tensor,
    block_stride: fx.Int32,
    pitch: fx.Int32,
    row0: fx.Int32,
    rows: fx.Int32,
    POSITIONS: fx.Tensor,
    ROW_INFO: fx.Tensor,
    OUT: fx.Tensor,
    STATS: fx.Tensor,
    ctas: fx.Int32,
    WIDE: fx.Constexpr[bool],
    SPLIT: fx.Constexpr[bool],
    stream: fx.Stream,
):
    if ctas > 0:
        _kernel(
            LOGITS,
            block_stride,
            pitch,
            row0,
            rows,
            POSITIONS,
            ROW_INFO,
            OUT,
            STATS,
            WIDE,
            SPLIT,
        ).launch(grid=(ctas, 1, 1), block=(256, 1, 1), stream=stream)


# Decode: one CTA of DECODE_WAVES waves per row. Waves own contiguous slot-group ranges, so each
# wave's histogram counts give its output offsets: row order and bits match the one-wave selection.
# LDS: per-wave histograms, per-wave reduction words, 64 candidates (keys, ids), chosen ids.
DECODE_WAVES = 8
D_RED, D_CAND, D_CHOSEN = _shared_row(DECODE_WAVES)
# Decode logits waves fold (min bits, ~max bits) into one of STAT_SLOTS pairs per row: same-address
# atomics from a few thousand waves of one row cost tens of microseconds.
STAT_SLOTS = 256
# Rows of more than LONG_KEYS keys on the 256-bin fast path spread over up to MAX_PARTS CTAs: one
# launch counts each part's keys per bin, the next finds the bin of the 512th key from the parts' sum
# and writes each part's kept blocks and threshold-bin candidates at prefix offsets, and the row CTA
# ranks the candidates and emits the row. Other rows select in the row CTA alone.
LONG_KEYS = 32768
MAX_PARTS = 32
# Gather LDS: per wave 512 kept ids, then 64 candidate keys and their ids; reduction words; four
# broadcast words.
G_WAVE = TOPK * 4 + 2 * CANDIDATES * 4
G_RED = DECODE_WAVES * G_WAVE
G_WORDS = G_RED + DECODE_WAVES * 16
# Byte offset that fails every buffer range check.
_DROP = 0x40000000


def _cta_sync():
    fx.memory_fence(syncscope="workgroup")
    gpu.barrier()
    fx.memory_fence(syncscope="workgroup")


def _word(resource, offset):
    return _buffer_words(resource, fx.Int32(offset), words=1)[0]


def _row_view(TENSOR, row, size):
    """Buffer over the `size` int32 values of TENSOR from row * size on."""
    return _buffer(
        fx.make_view(
            fx.get_iter(TENSOR) + fx.Int64(row) * fx.Int64(size), fx.make_layout(1, 1)
        ),
        size * 4,
    )


def _part_range(count, part, parts):
    """Slot groups [first, last) of part `part` of `parts` of a row of `count` keys, and how many parts
    hold groups."""
    groups = _groups(count)
    per = (groups + parts - 1) // parts
    first = _min(part * per, groups)
    return first, _min(first + per, groups), (groups + per - 1) // per


def _wave_range(first, last, wave):
    per = (last - first + DECODE_WAVES - 1) // DECODE_WAVES
    begin = _min(first + wave * per, last)
    return begin, _min(begin + per, last)


def _slot_stats(STATS, row, lane):
    """_row_stats of a decode row whose stats are STAT_SLOTS (min bits, ~max bits) pairs."""
    words = _row_view(STATS, row, 2 * STAT_SLOTS)
    # Lane l reads pairs [STAT_SLOTS / 64 * l, STAT_SLOTS / 64 * (l + 1)), two per 16-byte load.
    pairs = [
        _buffer_words(words, lane * (STAT_SLOTS // 8) + k * 16)
        for k in range(STAT_SLOTS // 128)
    ]
    low, high = fx.Int32(-1), fx.Int32(-1)
    for pair in pairs:
        low = _umin(_umin(low, pair[0]), pair[2])
        high = _umin(_umin(high, pair[1]), pair[3])
    low, high = _reduce(low, _umin), ~_reduce(high, _umin)
    return (
        low.bitcast(fx.Float32),
        high.bitcast(fx.Float32),
        _ugt(high, fx.Int32(0x7F800000)).select(fx.Int32(1), fx.Int32(0)),
    )


def _long_row(LENGTHS, STATS, row, lane):
    """(keys, low, digit scale, whether part CTAs select the row)."""
    count = _uniform(LENGTHS[row])
    low, high, nan = _slot_stats(STATS, row, lane)
    fast, scale = _fast(low, high, nan)
    return count, low, scale, fast & (count > LONG_KEYS)


def _suffix_counts(owned, lane):
    """Keys with digit >= b for the bins b = 4 * lane + i, from their counts."""
    inclusive = _inclusive_scan(owned[0] + owned[1] + owned[2] + owned[3], lane)
    running = _readlane(inclusive, 63) - inclusive
    suffix = []
    for value in reversed(owned):
        running = running + value
        suffix.insert(0, running)
    return suffix


def _cta_counts(shared, lane):
    """Counts of the bins 4 * lane + i summed over the CTA's per-wave histograms."""
    owned = []
    for i in range(BINS // 64):
        total = fx.Int32(0)
        for w in range(DECODE_WAVES):
            total = total + _load(shared + w * HIST + (lane * (BINS // 64) + i) * 4)
        owned.append(total)
    return owned


def _pick(values, index):
    picked = values[0]
    for i in range(1, len(values)):
        picked = (index == i).select(values[i], picked)
    return picked


def _sum(a, b):
    return a + b


def _suffix_threshold(suffix, need, lane):
    """(bin, count above it, its size) of the need-th largest key, from the suffix counts of bins
    4 * lane + i (non-increasing in the bin)."""
    hits = fx.Int32(0)
    for value in suffix:
        hits = hits + (value >= need).select(fx.Int32(1), fx.Int32(0))
    bin = _uniform(_reduce(hits, _sum) - 1)
    at = _readlane(_pick(suffix, bin & 3), bin >> 2)
    after = bin + 1
    beyond = (after < BINS).select(
        _readlane(_pick(suffix, after & 3), _min(after, fx.Int32(BINS - 1)) >> 2),
        fx.Int32(0),
    )
    return bin, beyond, at - beyond


@flyc.jit
def _row_suffix(SUFFIX, row, parts, used, lane):
    """Suffix counts of bins 4 * lane + i summed over the row's first `used` parts, eight loads per round."""
    source = _row_view(SUFFIX, row, parts * BINS)
    t0, t1, t2, t3 = fx.Int32(0), fx.Int32(0), fx.Int32(0), fx.Int32(0)
    for index in range(fx.Int32(0), used, fx.Int32(8)):
        first = fx.Int32(index)
        loads = [
            _buffer_words(
                source,
                (
                    (first + k < used).select(
                        (first + k) * BINS + lane * 4, fx.Int32(_DROP // 4)
                    )
                )
                * 4,
            )
            for k in range(8)
        ]
        for words in loads:
            t0, t1, t2, t3 = t0 + words[0], t1 + words[1], t2 + words[2], t3 + words[3]
    return t0, t1, t2, t3


def _part_bases(SUFFIX, row, parts, part, bin, lane):
    """(kept, candidates) of the parts before `part`: their keys above `bin` and inside it."""
    source = _row_view(SUFFIX, row, parts * BINS)
    offset = (lane * BINS + bin) * 4
    mine = lane < part
    at = _word(source, mine.select(offset, fx.Int32(_DROP)))
    beyond = _word(
        source, (mine & (bin + 1 < BINS)).select(offset + 4, fx.Int32(_DROP))
    )
    return _reduce(beyond, _sum), _reduce(at - beyond, _sum)


@flyc.jit
def _part_hist(
    LOGITS,
    stride,
    SUFFIX,
    parts,
    row,
    part,
    count,
    low,
    scale,
    first,
    last,
    shared,
    wave,
    lane,
):
    logits = SimpleNamespace(
        resource=_buffer(LOGITS, fx.Int32(0x7FFFFFF0)), step=fx.Int32(GROUP_KEYS * 4)
    )
    begin, end = _wave_range(first, last, wave)
    hist = shared + wave * HIST
    _clear(hist, lane)
    _sync()
    _sweep_hist(logits, row * stride * 4, count, lane, hist, low, scale, begin, end)
    _cta_sync()
    if wave == 0:
        output = _row_view(SUFFIX, row * parts + part, BINS)
        _buffer_store_words(
            fx.Vector.from_elements(
                _suffix_counts(_cta_counts(shared, lane), lane), fx.Int32
            ),
            output,
            lane * 16,
        )


@flyc.kernel(
    name="qsa_indexer_decode_topk_hist", known_block_size=[64 * DECODE_WAVES, 1, 1]
)
def _hist_kernel(
    LOGITS: fx.Tensor,
    stride: fx.Int32,
    LENGTHS: fx.Tensor,
    STATS: fx.Tensor,
    SUFFIX: fx.Tensor,
    parts: fx.Int32,
):
    storage = (
        fx.SharedAllocator().allocate(fx.Array[fx.Int8, DECODE_WAVES * HIST, 16]).peek()
    )
    shared = fx.Int32(
        fx.ptrtoint(fx.get_iter(storage.view(fx.make_layout(DECODE_WAVES * HIST, 1))))
    )
    tid = fx.thread_idx.x
    wave, lane = _uniform(tid >> 6), tid & 63
    part, row = fx.block_idx.x, fx.block_idx.y
    count, low, scale, long = _long_row(LENGTHS, STATS, row, lane)
    first, last, _ = _part_range(count, part, parts)
    if long & (first < last):
        _part_hist(
            LOGITS,
            stride,
            SUFFIX,
            parts,
            row,
            part,
            count,
            low,
            scale,
            first,
            last,
            shared,
            wave,
            lane,
        )


@flyc.jit
def _part_gather(
    LOGITS,
    stride,
    SUFFIX,
    KEPT,
    CAND,
    parts,
    row,
    part,
    used,
    count,
    low,
    scale,
    first,
    last,
    shared,
    wave,
    lane,
):
    words = shared + G_WORDS
    if wave == 0:
        found, _, found_size = _suffix_threshold(
            _row_suffix(SUFFIX, row, parts, used, lane), fx.Int32(TOPK), lane
        )
        kept_before, cand_before = _part_bases(SUFFIX, row, parts, part, found, lane)
        if lane == 0:
            _store(words, found)
            _store(words + 4, found_size)
            _store(words + 8, kept_before)
            _store(words + 12, cand_before)
    _cta_sync()
    bin, size = _load(words), _load(words + 4)
    if size <= CANDIDATES:
        logits = SimpleNamespace(
            resource=_buffer(LOGITS, fx.Int32(0x7FFFFFF0)),
            step=fx.Int32(GROUP_KEYS * 4),
        )
        begin, end = _wave_range(first, last, wave)
        own = shared + wave * G_WAVE
        cand = own + TOPK * 4
        kept, gathered = _sweep_gather(
            logits,
            row * stride * 4,
            count,
            lane,
            cand,
            SimpleNamespace(base=own, wide=True),
            low,
            scale,
            bin,
            begin,
            end,
            fx.Int32(0),
            fx.Int32(0),
        )
        keeps, gathers = _exchange(
            SimpleNamespace(red=shared + G_RED, wave=wave, waves=DECODE_WAVES),
            [kept, gathered],
        )
        kept_base = _load(words + 8) + _below_wave(keeps, wave)
        cand_base = _load(words + 12) + _below_wave(gathers, wave)
        kept_out = _row_view(KEPT, row, TOPK)
        for index in range(lane, kept, fx.Int32(64)):
            i = fx.Int32(index)
            _put(kept_out, kept_base + i, _load(own + i * 4))
        if lane < gathered:
            cand_out = _row_view(CAND, row, 2 * CANDIDATES)
            _put(cand_out, cand_base + lane, _load(cand + KEYS + lane * 4))
            _put(cand_out, CANDIDATES + cand_base + lane, _load(cand + IDS + lane * 4))


@flyc.kernel(
    name="qsa_indexer_decode_topk_gather", known_block_size=[64 * DECODE_WAVES, 1, 1]
)
def _gather_kernel(
    LOGITS: fx.Tensor,
    stride: fx.Int32,
    LENGTHS: fx.Tensor,
    STATS: fx.Tensor,
    SUFFIX: fx.Tensor,
    KEPT: fx.Tensor,
    CAND: fx.Tensor,
    parts: fx.Int32,
):
    storage = fx.SharedAllocator().allocate(fx.Array[fx.Int8, G_WORDS + 16, 16]).peek()
    shared = fx.Int32(
        fx.ptrtoint(fx.get_iter(storage.view(fx.make_layout(G_WORDS + 16, 1))))
    )
    tid = fx.thread_idx.x
    wave, lane = _uniform(tid >> 6), tid & 63
    part, row = fx.block_idx.x, fx.block_idx.y
    count, low, scale, long = _long_row(LENGTHS, STATS, row, lane)
    first, last, used = _part_range(count, part, parts)
    if long & (first < last):
        _part_gather(
            LOGITS,
            stride,
            SUFFIX,
            KEPT,
            CAND,
            parts,
            row,
            part,
            used,
            count,
            low,
            scale,
            first,
            last,
            shared,
            wave,
            lane,
        )


@flyc.jit
def _select_long(logits, base, count, row, low, high, nan, long, index):
    """_select, except that rows the part CTAs selected only rank the gathered threshold-bin candidates."""
    fast, _ = _fast(low, high, nan)
    split = fx.Int32(0)
    done = fx.Int32(0)
    if fast & (count > LONG_KEYS) & (long.parts > 0):
        if row.wave == 0:
            _, _, used = _part_range(count, fx.Int32(0), long.parts)
            _, found_above, found_size = _suffix_threshold(
                _row_suffix(long.SUFFIX, index, long.parts, used, row.lane),
                fx.Int32(TOPK),
                row.lane,
            )
            if row.lane == 0:
                _store(row.red, found_above)
                _store(row.red + 4, found_size)
        _cta_sync()
        above, size = _load(row.red), _load(row.red + 4)
        if size <= CANDIDATES:
            kept = _row_view(long.KEPT, index, TOPK)
            for i in range(row.thread, above, fx.Int32(row.threads)):
                _store_id(row.chosen, fx.Int32(i), _word(kept, fx.Int32(i) * 4))
            if row.thread < size:
                cand = _row_view(long.CAND, index, 2 * CANDIDATES)
                _store(row.cand + KEYS + row.thread * 4, _word(cand, row.thread * 4))
                _store(
                    row.cand + IDS + row.thread * 4,
                    _word(cand, (CANDIDATES + row.thread) * 4),
                )
            _cta_sync()
            _rank_row(row, size, above)
            done = fx.Int32(1)
            split = above
    if done == 0:
        split = _select(logits, base, count, row, low, high, nan)
    return split


@flyc.kernel(name="qsa_indexer_decode_topk", known_block_size=[64 * DECODE_WAVES, 1, 1])
def _decode_kernel(
    LOGITS: fx.Tensor,
    stride: fx.Int32,
    rows: fx.Int32,
    LENGTHS: fx.Tensor,
    POSITIONS: fx.Tensor,
    SEQUENCES: fx.Tensor,
    OUT: fx.Tensor,
    STATS: fx.Tensor,
    SUFFIX: fx.Tensor,
    KEPT: fx.Tensor,
    CAND: fx.Tensor,
    parts: fx.Int32,
    WIDE: fx.Constexpr[bool],
    LONG: fx.Constexpr[bool],
):
    size = D_CHOSEN + _wave_bytes(WIDE)[1]
    storage = fx.SharedAllocator().allocate(fx.Array[fx.Int8, size, 16]).peek()
    shared = fx.Int32(fx.ptrtoint(fx.get_iter(storage.view(fx.make_layout(size, 1)))))
    tid = fx.thread_idx.x
    wave, lane = _uniform(tid >> 6), tid & 63
    row = fx.block_idx.x
    count = _uniform(LENGTHS[row])
    visible = _uniform(POSITIONS[row]) + 1
    length = _uniform(SEQUENCES[row])
    # The merged ids reuse the histograms.
    space = _space(
        DECODE_WAVES,
        wave,
        lane,
        shared + wave * HIST,
        shared + D_CAND,
        shared + D_CHOSEN,
        shared,
        WIDE,
        hist_base=shared,
        red=shared + D_RED,
    )
    if fx.const_expr(LONG):
        long = SimpleNamespace(SUFFIX=SUFFIX, KEPT=KEPT, CAND=CAND, parts=parts)
        _write_row(
            LOGITS,
            stride * 4,
            fx.Int32(GROUP_KEYS * 4),
            row,
            row,
            count,
            visible,
            length,
            OUT,
            space,
            STATS,
            long,
        )
    else:
        _write_row(
            LOGITS,
            stride * 4,
            fx.Int32(GROUP_KEYS * 4),
            row,
            row,
            count,
            visible,
            length,
            OUT,
            space,
            None,
        )


@flyc.jit
def _launch_parts(
    LOGITS: fx.Tensor,
    stride: fx.Int32,
    LENGTHS: fx.Tensor,
    STATS: fx.Tensor,
    SUFFIX: fx.Tensor,
    KEPT: fx.Tensor,
    CAND: fx.Tensor,
    parts: fx.Int32,
    rows: fx.Int32,
    stream: fx.Stream,
):
    if rows > 0:
        _hist_kernel(LOGITS, stride, LENGTHS, STATS, SUFFIX, parts).launch(
            grid=(parts, rows, 1), block=(64 * DECODE_WAVES, 1, 1), stream=stream
        )
        _gather_kernel(
            LOGITS, stride, LENGTHS, STATS, SUFFIX, KEPT, CAND, parts
        ).launch(grid=(parts, rows, 1), block=(64 * DECODE_WAVES, 1, 1), stream=stream)


@flyc.jit
def _launch_decode(
    LOGITS: fx.Tensor,
    stride: fx.Int32,
    rows: fx.Int32,
    LENGTHS: fx.Tensor,
    POSITIONS: fx.Tensor,
    SEQUENCES: fx.Tensor,
    OUT: fx.Tensor,
    STATS: fx.Tensor,
    SUFFIX: fx.Tensor,
    KEPT: fx.Tensor,
    CAND: fx.Tensor,
    parts: fx.Int32,
    ctas: fx.Int32,
    WIDE: fx.Constexpr[bool],
    LONG: fx.Constexpr[bool],
    stream: fx.Stream,
):
    if ctas > 0:
        _decode_kernel(
            LOGITS,
            stride,
            rows,
            LENGTHS,
            POSITIONS,
            SEQUENCES,
            OUT,
            STATS,
            SUFFIX,
            KEPT,
            CAND,
            parts,
            WIDE,
            LONG,
        ).launch(grid=(ctas, 1, 1), block=(64 * DECODE_WAVES, 1, 1), stream=stream)
