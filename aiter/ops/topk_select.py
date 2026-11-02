# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Per-row top-k with a DeepSelect-shaped interface, over five backends.

aiter carries five per-row selectors, each fastest in a different corner and
none able to cover the whole domain:

    argmax   k=1 only, and a reduction rather than a selection: the row split
             across as many workgroups as it takes to fill the part. Owns k=1,
             and is the only one to take bf16/fp16 as well as fp32.
    small_k  one chunk per lane, so k <= wave_size, and a survivor buffer that
             grows with the row bound. Unbeatable on short rows and tiny k.
    plain    the C++/ASM radix selector. Bounded at k=2048. Scales with rows
             better than anything else, so it owns the wide-M middle.
    decode   a grid-wide radix select: histogram, reduce, gather. Owns the very
             wide rows at low row counts, and all of k=4096.
    stream   one workgroup per row, reading the row once. Owns the narrow rows
             and the large-M end of the wide ones.

`topk_select` picks between them. The parameter names, order and return shape
follow `deep_select.topk` so code written against DeepSelect ports across; where
this cannot honour DeepSelect's behaviour it raises rather than diverging
silently. The differences are listed on `topk_select` itself.

Ties: as in DeepSelect, no prefix index may be assumed by default. `plain` has
no tie order at all (measured: 40 equal scores, 8 places, a set that is neither
the lowest nor the highest); `small_k` resolves toward the larger column and
`stream` toward the smaller, whether or not they were asked to. `decode` gives
the smaller column only when asked, since that is a mode of its kernel and the
mode costs up to 20%; `tie='low'` is the request. `tie=` also narrows the
backend set to those that can promise a direction, which costs speed.
"""

import builtins
import os
from functools import lru_cache

import torch
import triton
import triton.language as tl

from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl.kernels.tensor_shim import wave_size_of
from aiter.ops.flydsl.kernels.topk.topk_per_row_radix_stream import (
    _lds_budgets,
    topk_per_row_radix_stream_serves,
)
from aiter.ops.flydsl.topk.row_lens import no_row_lens
from aiter.ops.flydsl.topk.topk_per_row import flydsl_top_k_per_row_decode
from aiter.ops.flydsl.topk.topk_per_row_argmax import (
    ARGMAX_DTYPES,
    topk_per_row_argmax,
    topk_per_row_argmax_serves,
)
from aiter.ops.flydsl.topk.topk_per_row_radix_stream import topk_per_row_radix_stream
from aiter.ops.flydsl.topk.topk_per_row_small_k import (
    topk_per_row_small_k,
    topk_per_row_small_k_serves,
)
from aiter.ops.topk import (
    _device_arch,
    _sampled_min_width,
    _sampled_on_device,
    _sampled_run,
    _sampled_supports_cached,
    _sampled_takes,
    top_k_per_row_prefill_sampled,
)
from aiter.ops.topk_plain import (
    topk_plain,
    topk_plain_batches_ragged_rows,
    topk_plain_reads_pitch,
    topk_plain_use_mulblocks,
    topk_plain_width,
)

__all__ = ["topk_select", "topk_select_backend"]

_SUPPORTED_GFX = ("gfx942", "gfx950", "gfx1250")

_PLAIN_MAX_K = 2048
# Where plain's module builds. Its kernels are wave64 code -- for gfx1250 the
# build stops at topk_plain_kernels.cu's `>= 2 * opus::get_warp_size()`
# static_asserts -- and the questions `_available` and the shape rules ask of it
# live in the same module, so elsewhere plain is neither offered nor asked.
_PLAIN_ARCHES = ("gfx942", "gfx950")
# Which backends can promise a column order among equal scores. `plain` appears
# in no list: its tie order falls out of its internal geometry.
#
# `small_k` is absent from "low" for more than a flipped comparison: once more
# chunks tie at the cut than there are places, step 2 has already dropped chunks
# by lane id, and lane is `(col // 4) % wave`, not monotone in the column.
# Measured on 64 tied chunks at k=16, the flipped build returns [208, 256, 260,
# ...] against a canonical [128, 132, 136, ...]. Serving "low" needs the chunk
# tie-break made column-aware first.
_BACKENDS_BY_TIE = {
    # `sampled` is listed under no promise only. Its tie direction has not been
    # measured, and `topk_select_backend` below records what happens when a
    # backend is declared for a tie mode it does not honour: `tie='high'` once
    # fell through to `decode`, which ties the opposite way, silently. Declaring
    # it for `low` or `high` on a guess would repeat that.
    #
    # `stream_reject` and `whole_row` select the set `stream` does wherever they
    # serve, with its tie order, so they sit where it does.
    None: (
        "argmax",
        "small_k",
        "plain",
        "decode",
        "stream",
        "sampled",
        "stream_reject",
        "whole_row",
    ),
    "low": ("argmax", "decode", "stream", "stream_reject", "whole_row"),
    "high": ("small_k",),
}
# `plain` selects a different set of tied columns from one call to the next:
# measured, 2 of 512 slots differed on a repeat, 18 for one row across a batch of
# 100. Every other backend is a pure function of the row -- the property a
# tensor-parallel caller needs, and weaker than promising a direction.
# `sampled` is in here because its repeat-stability has NOT been measured, not
# because it was found unstable. The conservative direction: a caller asking
# for `deterministic` never reaches it. Measure it and remove it from this set
# -- `op_tests/test_topk_per_row_stable.py` is the model, and `decode` is the
# cautionary tale, since a radix select being a pure function of the row is
# the obvious guess and it is wrong.
_NONDETERMINISTIC = frozenset({"plain", "sampled"})
# Order to fall back in when the shape rules name nothing that is available.
# Streaming first because it takes a row length natively and scales with rows;
# `plain` last for the reasons below.
# `sampled` last: this order only decides who serves when every shape rule
# declined, and putting it anywhere else would change a fallback that is
# already fitted. Last means it is picked only when nothing else can serve.
_PREFERENCE = (
    "argmax",
    "stream",
    "decode",
    "small_k",
    "plain",
    "sampled",
    "stream_reject",
    "whole_row",
)
# Fitted to a 565-cell sweep -- rows 1..16384, widths 2048..1M, k 16..4096, to
# 8 GiB -- by `topk_backend_fit.py` over `topk_backend_sweep.py`'s table. Re-run
# both rather than nudging a number: the function is piecewise constant, and
# three of these constants are pinned by ties rather than by a measurement (see
# below). Every backend is checked against the others' selected VALUES before
# being timed, in interleaved rounds, because one reading of this domain runs
# 1.42x off another.
#
# Fitted to the stopwatch alone. The rules this replaces were fitted to a
# preference instead -- among backends within 1.4x of the fastest they took the
# most preferred, and `_PREFERENCE` lists `stream` ahead of `decode` -- which
# was defensible while both were deterministic and close. They are not close:
# decode is the fastest backend on 257 of 565 cells and the old rules reached it
# on 65, for a mean 1.295x and a worst 5.916x against the per-cell oracle.
#
# Constants the sweep cannot separate, held at their shipped values rather than
# moved on a tie: `_SMALL_K_MAX_K` (no k=32 cell) and `_PLAIN_MANY_ROWS_BAND`'s
# floor (no width between 4096 and 8192). Widen the sweep before touching either.
#
# One rule was dropped, not retuned: `plain` used to take a middling-width band
# at <= 8 rows, and with decode's gate widened that band is now decode's on
# every cell of it. Every value of the old bound scored identically, which is
# what a dead rule looks like.

# The block widths `topk_per_row_radix_stream_block_threads` chooses between.
# `_available` has to ask about both without knowing which the row count picks.
_STREAM_BLOCK_WIDTHS = (512, 1024)

# `plain` is the only backend that scales WITH rows, so past enough of them on a
# middling width it wins outright -- ahead of small_k, hence tested first. It is
# never the fastest below k=1024: of the 64 cells it wins, 29 are k=1024 and 35
# are k=2048.
_PLAIN_MANY_ROWS = 256
_PLAIN_MANY_ROWS_BAND = (8192, 65536)
_PLAIN_MIN_K = 1024
# gfx942 at k=2048, where the generic many-row rule leaves plain regions it
# wins. Measured against decode, which the rules pick there otherwise (device
# time of replayed graphs, randn; uniform rows and rows ending uniformly in
# [k, width] or [width/2, width]; 1..4096 rows of 2048..32770 columns). Below
# 128 CUs plain ran 0.44x-0.80x of decode from 128 rows on every width past k
# through 20000 columns, ragged rows included, and lost up to 1.68x on fewer
# rows. With 128 CUs or more it ran 0.46x-0.72x on uniform rows of the two
# triplets inside the register window plain keeps off gfx950, at every row
# count, and lost up to 1.42x on uniform rows from 24576 columns and up to
# 1.46x on ragged rows. Should the window (`reg_window_min`..
# `reg_window_max`, asked of the kernels) stop covering a row of a triplet,
# that band lapses for it rather than send it to the generic kernel.
_PLAIN_K2048_SMALL_CU = 128
_PLAIN_K2048_SMALL_CU_MIN_ROWS = 128
_PLAIN_K2048_SMALL_CU_MAX_WIDTH = 20000
_PLAIN_K2048_REG_WINDOW_BANDS = ((4096, 4098), (8192, 8194))
_PLAIN_K2048_REG_WINDOW_MAX_ROWS = 4096
# gfx950's k=2048 plain band (`_plain_gfx950_k2048_takes`). There plain's
# dispatch serves every row length up to 80 * 1024 columns with a register or
# LDS-tail kernel, and on uniform rows it beat what the rules pick otherwise,
# 0.31x-0.94x: stream from 2053 to 8191 columns at any row count, decode from
# 8192 columns to the kernels' own `any_rows_max` through 128 rows, and stream
# past 128 rows to 131071 columns. Rows of one or two rejects stay with
# `stream_reject`'s unmasked branch (`_SMALL_REJECT_UNMASKED_MAX`), 1.09x-1.60x
# faster; from three rejects plain beat that kernel.
_PLAIN_K2048_NARROW_WIDTH = 8192
_PLAIN_K2048_FEW_ROWS = 128
_PLAIN_K2048_MAX_WIDTH = 131071
# How that band and `sampled` share the widths (`_sampled_gfx950_k2048_takes`,
# `_plain_k2048_lds_tail_takes`). Below sampled's width door uniform rows stay
# in plain's band, which beat the work door's `sampled` there (0.41x-0.97x);
# rows plain's own dispatch hands to its multi-block kernel (`_plain_multiblock`)
# and the few-row band's rows past `any_rows_max` go to `sampled`, which beat
# that kernel by 1.7x-2.8x and decode by 1.47x-2.56x -- bar a row whose sample
# undershoots k and takes the exact fallback, which follows the values rather
# than the shape (up to 1.32x decode's time). Past the door, uniform rows from
# 3000 rows go to plain's LDS tail up to its own `lds_tail_max`, 0.80x-0.97x of
# `sampled`.
_PLAIN_K2048_LDS_TAIL_MIN_ROWS = 3000
# Ragged rows at k=2048 on gfx950, which plain's band above does not take (see
# `_plain_takes`). Every backend measured over 1..4096 rows of 2100..131071
# columns with three row-end patterns -- uniform in [width/2, width], causal
# (row r ends at width - rows + r + 1) and full -- on MI355X, randn, two
# rounds, 288 cells. Below 8192 columns stream was the fastest on 47 of 48
# cells (within 1.02x on the other).
#
# Past that the router cannot see how far rows reach, so the bands were
# re-measured under those three patterns and a fourth, rows ending uniformly
# in [k, width], in device time (replayed graphs, 3 seeds, MI355X). From 8192
# to 65535 columns decode takes the rows below plain's floor
# (`_PLAIN_MANY_ROWS`): from 129 rows it ran 0.53x-0.98x of stream, and
# `sampled` lost to it by up to 2.13x on short rows and 1.28x on full ones.
# From 256 rows plain keeps them: decode ran up to 1.06x slower there, and
# 1.59x at 288..320 rows, over 16385..20000 columns. From 65536 columns the
# general rules take them; `sampled` serves uniform rows only (`_sampled_takes`).
_RAGGED_K2048_STREAM_MAX_WIDTH = 8191
_RAGGED_K2048_DECODE_MAX_WIDTH = 65535
# Rows that select all of their columns go to `whole_row` (`_stream_specialised`).
# With the values wanted, one launch writing both replaces a selection and a
# gather: 0.47x-0.78x of the fastest backend at every measured row count from
# 1 to 16384 (seed-0 randn, MI355X, 4 processes, worst process). Indices alone
# are one launch either way, and there the write beat stream only from 640 to
# 12288 rows in device time (replayed graphs, MI355X): 0.86x-0.97x through
# 8192 rows and 0.99x at 12288, against 1.02x-1.08x slower at 5..512 rows and
# at 16384.
_WHOLE_ROW_INDEX_ROWS = (640, 8192)

# small_k narrows by dropping chunks below the cut, and a chunk is a lane: at k
# equal to the wave width it drops none. Survivors at 8192 columns run 18 at
# k=16, 44 at k=32, then 300 at k=64, and the time steps 1.7x-1.8x between k=63
# and k=64 alone. Below this bound it wins over the whole row range.
_SMALL_K_MAX_K = 32

# decode is grid-wide: it needs a row wide enough to spread a grid over, and few
# enough rows that the grid is not already full. BOTH bounds relax at the k that
# amortises its launch -- its time is flat at 30..36us across widths
# 32768..65536, k 16..1024 and 1..64 rows, which is fixed cost rather than work,
# while `stream` over those same cells runs 18.6..51.4us.
#
# `_DECODE_AMORTISING_K` sits in an unsampled gap (256..1024) and the narrow
# gate's width in another (4096..8192). Widen the sweep before moving either.
_DECODE_AMORTISING_K = 1024
# (narrowest row, most rows) decode will take, below and at that k.
_DECODE_GATE = (65536, 64)
_DECODE_GATE_AMORTISED = (8192, 128)
# At the largest k served, decode wins at ANY row count over a band of widths --
# wide enough to spread a grid across, narrow enough that `stream`'s split
# cannot outrun it. Measured at k=4096 over rows 1..16384, decode against
# stream: 0.28x..0.96x inside the band, 0.90x..1.26x at 8192 and 1.02x..1.73x at
# 262144, which is why it has both ends.
_DECODE_ANY_ROWS_K = 4096
_DECODE_ANY_ROWS_BAND = (16384, 131072)


@lru_cache(maxsize=1)
def _unsupported_arch() -> str | None:
    """The arch, if this is one the selectors are not built for. Not evaluated
    at import: aiter is imported for codegen on hosts with no GPU."""
    gfx = get_gfx()
    return None if gfx in _SUPPORTED_GFX else gfx


@lru_cache(maxsize=8)
def _cached_bounds(rows: int, width: int, device: int, stream: int):
    return (
        torch.zeros(rows, dtype=torch.int32, device=device),
        torch.full((rows,), width, dtype=torch.int32, device=device),
    )


def _default_bounds(rows: int, width: int, device: int):
    """`(starts, ends)` of a call that gave no `end`: every row from column 0
    to the full width.

    Only the C++ backends that take a [start, end) pair ask for it: `sampled`,
    whose entry checks both have `rows` entries and reads them for a row-strided
    call, and plain's ranged form for one. The FlyDSL backends read no length
    tensor without an `end`; they build full-width rows (`no_row_lens`).

    Cached because they are constants the kernels only read. Building the pair
    per call is two allocations and two fill launches, worth 8.1us at 4096 rows
    and a fifth of the whole call at 64 for `sampled`: calling its kernel
    directly rather than through this entry measured 1.20x at m=64 n=524288 and
    1.13x at m=64 n=1048576, and this pair was that gap.

    Not cached while a CUDA graph is capturing: there the fills are only
    recorded, so a pair cached then would stay unfilled for the eager calls
    after it, and a graph must not hold tensors the cache can later free.

    Cached per stream as well: a pair is filled on the stream of the call that
    builds it and read only by calls on that stream, after the fill, and one
    the cache drops goes back to that stream's pool, which no other stream's
    kernels read from.
    """
    if torch.cuda.is_current_stream_capturing():
        return _cached_bounds.__wrapped__(rows, width, device, 0)
    return _cached_bounds(
        rows, width, device, torch._C._cuda_getCurrentRawStream(device)
    )


class _GrowDuringCapture(Exception):
    """The zero buffer would have to grow while a CUDA graph captures."""


# Per GPU, every zero buffer `_zero_rows` has cut views from, the last the
# largest. Never freed: a captured graph may still read any of them.
_ZERO_BUFFERS: dict[int, list[torch.Tensor]] = {}


@lru_cache(maxsize=8)
def _cached_zero_rows(rows: int, device: int) -> torch.Tensor:
    buffers = _ZERO_BUFFERS.setdefault(device, [])
    if not buffers or buffers[-1].numel() < rows:
        if torch.cuda.is_current_stream_capturing():
            raise _GrowDuringCapture
        size = max(rows, 2 * buffers[-1].numel()) if buffers else rows
        grown = torch.zeros(size, dtype=torch.int32, device=device)
        # Every stream reads these, so the fill has to be done before any can:
        # a wait on the growth, which doubles, rather than a fence per call.
        torch.cuda.current_stream(device).synchronize()
        buffers.append(grown)
    return buffers[-1][:rows]


def _zero_rows(rows: int, device: int) -> torch.Tensor:
    """The starts of a call that gave its own `end`: every row from column 0.

    A view of a zero buffer that is never freed, so the cache may drop it while
    a captured graph still reads it, and no capture check is paid per call. A
    buffer that would have to grow during a capture, where its fill would only
    be recorded, is replaced by a tensor of the graph's own.
    """
    try:
        return _cached_zero_rows(rows, device)
    except _GrowDuringCapture:
        return torch.zeros(rows, dtype=torch.int32, device=device)


@triton.jit
def _gather_selected_kernel(
    scores_ptr,
    idx_ptr,
    out_ptr,
    scores_stride0,
    idx_stride0,
    out_stride0,
    topk,
    fill,
    BLOCK_K: tl.constexpr,
):
    """`out[r, j] = scores[r, idx[r, j]]`, or `fill` where `idx` is negative."""
    row = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, BLOCK_K)
    live = offs < topk
    idx = tl.load(idx_ptr + row * idx_stride0 + offs, mask=live, other=-1)
    # A padded slot holds -1, which would address backwards; the mask below is
    # what keeps `fill` in that lane, so the clamp only has to be in bounds.
    kept = live & (idx >= 0)
    val = tl.load(
        # int64 throughout: `rows * stride0` passes 2^31 at 16384 rows of a
        # 1M-wide fp32 tensor, and a 32-bit offset wraps there silently.
        scores_ptr + row * scores_stride0 + tl.where(kept, idx, 0).to(tl.int64),
        mask=kept,
        other=fill,
    )
    tl.store(out_ptr + row * out_stride0 + offs, val, mask=live)


def _gather_selected(scores, idx, fill):
    """The selected scores, padded slots filled, in one launch.

    Replaces `idx.long().clamp_min_(0)`, a `gather`, an `idx < 0` and a
    `masked_fill_` -- five launches whose cost is almost all fixed. Measured
    across `[rows, topk]` from 640 to 524288 elements, the element count grows
    800x while those five grow 17.1us to 38.1us, so what they cost is being five
    rather than what they touch. One launch over the same data measured 5.9-11.4us.
    """
    rows, topk = idx.shape
    out = torch.empty((rows, topk), dtype=scores.dtype, device=scores.device)
    _gather_selected_kernel[(rows,)](
        scores,
        idx,
        out,
        scores.stride(0),
        idx.stride(0),
        out.stride(0),
        topk,
        fill,
        BLOCK_K=triton.next_power_of_2(topk),
    )
    return out


@triton.jit
def _whole_row_kernel(
    scores_ptr,
    idx_ptr,
    out_ptr,
    scores_stride0,
    idx_stride0,
    out_stride0,
    topk,
    VALUES: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """`idx[r, j] = j` and, with VALUES, `out[r, j] = scores[r, j]`."""
    row = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, BLOCK_K)
    live = offs < topk
    tl.store(idx_ptr + row * idx_stride0 + offs, offs, mask=live)
    if VALUES:
        val = tl.load(scores_ptr + row * scores_stride0 + offs, mask=live)
        tl.store(out_ptr + row * out_stride0 + offs, val, mask=live)


def _select_whole_rows(input, idx, with_values):
    """The `whole_row` backend: uniform rows that select every column.

    `stream` writes 0..k-1 in column order on such a row whatever the scores --
    NaN, -inf, ties and constant rows included -- so the answer is known before
    reading them, and with the values wanted this one launch also stands in for
    `_gather_selected`, whose values it returns.
    """
    rows, topk = idx.shape
    out = (
        torch.empty((rows, topk), dtype=input.dtype, device=input.device)
        if with_values
        else None
    )
    _whole_row_kernel[(rows,)](
        input,
        idx,
        idx if out is None else out,
        input.stride(0),
        idx.stride(0),
        0 if out is None else out.stride(0),
        topk,
        VALUES=with_values,
        BLOCK_K=triton.next_power_of_2(topk),
    )
    return out


# The rejects `_stream_small_reject_kernel` takes through its unmasked branch,
# which is written out for one and two; plain's gfx950 k=2048 band leaves those
# rows to it.
_SMALL_REJECT_UNMASKED_MAX: tl.constexpr = tl.constexpr(2)
# The most rejects `stream_reject` serves: its block holds a row of k=2048 plus
# them, and `_stream_small_reject` routes it no further.
_STREAM_REJECT_MAX = 8


@triton.jit
def _stream_small_reject_kernel(
    scores_ptr,
    idx_ptr,
    scores_stride0,
    idx_stride0,
    WIDTH: tl.constexpr,
    REJECTS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Select ``BLOCK_K`` columns by rejecting a small bottom tail.

    The generic stream selector orders fp32 keys descending and columns
    ascending.  Here the same order is expressed as a signed key so repeated
    reductions can find the complement: smaller key is worse and, among equal
    keys, the larger column is worse.  NaNs collapse above +inf and signed zero
    collapses to +0, matching the stream key transform exactly.
    """
    row = tl.program_id(0).to(tl.int64)
    col = tl.arange(0, BLOCK_N)
    live = col < WIDTH
    score = tl.load(
        scores_ptr + row * scores_stride0 + col,
        mask=live,
        other=0.0,
    )
    bits = score.to(tl.int32, bitcast=True)
    bits = tl.where(bits == -2147483648, 0, bits)
    key = bits ^ ((bits >> 31) & 0x7FFFFFFF)
    is_nan = (bits & 0x7FFFFFFF) > 0x7F800000
    key = tl.where(is_nan | ~live, 0x7FFFFFFF, key)

    if REJECTS <= _SMALL_REJECT_UNMASKED_MAX:
        # Keep the two hottest shapes byte-for-byte equivalent to the original
        # specialization: no rejection mask and no prefix scan.
        worst_key0 = tl.min(key, axis=0)
        reject0 = tl.max(tl.where(live & (key == worst_key0), col, -1), axis=0)
        reject_lo = reject0
        reject_hi = reject0
        if REJECTS == 2:
            key1 = tl.where(col == reject0, 0x7FFFFFFF, key)
            worst_key1 = tl.min(key1, axis=0)
            reject1 = tl.max(
                tl.where(live & (col != reject0) & (key1 == worst_key1), col, -1),
                axis=0,
            )
            reject_lo = tl.minimum(reject0, reject1)
            reject_hi = tl.maximum(reject0, reject1)
        slot = tl.arange(0, BLOCK_K)
        out_col = tl.where(slot >= reject_lo, slot + 1, slot)
        if REJECTS == 2:
            out_col = tl.where(out_col >= reject_hi, out_col + 1, out_col)
        tl.store(idx_ptr + row * idx_stride0 + slot, out_col)
    else:
        rejected = col < 0
        for _ in tl.static_range(0, REJECTS):
            remaining_key = tl.where(rejected, 0x7FFFFFFF, key)
            worst_key = tl.min(remaining_key, axis=0)
            reject = tl.max(
                tl.where(live & ~rejected & (remaining_key == worst_key), col, -1),
                axis=0,
            )
            rejected |= col == reject

        # Compact the complement in ascending-column order.  Exactly BLOCK_K
        # live columns remain, so every output slot is written once.
        keep = live & ~rejected
        slot = tl.cumsum(keep.to(tl.int32), axis=0) - 1
        tl.store(
            idx_ptr + row * idx_stride0 + slot,
            col,
            mask=keep,
        )


@lru_cache(maxsize=256)
def _available(
    width: int,
    k: int,
    wave_size: int,
    ragged: bool,
    fp32: bool,
    sampled_ok: bool,
    arch: str,
) -> frozenset:
    """Backends that can serve this geometry at all.

    Each is asked through its own predicate rather than through a copy of its
    limits kept here: a second copy drifts, and drift reads as declining a shape
    that works or, worse, accepting one that does not. Asked once per geometry,
    because asking costs 12us of Python against 7-10us of device time for the
    selection -- every predicate otherwise re-derives the arch from scratch.

    The tensor half of each contract (inner stride 1, int32 indices) is already
    enforced by `_reject_unsupported`, so what is left is geometry -- and the
    one dtype fact the geometry predicates do not carry: only the reduction has
    a half-format build, so a non-fp32 input leaves it alone in the set. That
    shape is already unreachable via `_reject_unsupported`, which refuses a half
    format past k=1; keeping it here means the two cannot disagree about which
    backend would have been asked.
    """
    out = set()
    if topk_per_row_argmax_serves(k) is None:
        out.add("argmax")
    if not fp32:
        return frozenset(out)
    if topk_per_row_small_k_serves(k, width, wave_size) is None:
        out.add("small_k")
    if (
        arch in _PLAIN_ARCHES
        and k <= _PLAIN_MAX_K
        and topk_plain_batches_ragged_rows(width, k)
    ):
        # plain's radix path only: it is the build that writes no values and
        # ranks a NaN of either sign above +inf, as the other backends do. The
        # block-sort fallback does neither, and for ranged rows it is a
        # host-side loop of one call per row: 294918 launches at k=16, N=32768
        # where the batched form is one.
        out.add("plain")
    # decode was refused at 4 GiB when it built descriptors over the whole
    # tensor; it slices the row first now, so there is nothing left to ask.
    out.add("decode")
    # Both block widths, not the one `_dispatch` will pick: this set is memoized
    # without the row count, and the width `_dispatch` chooses depends on it. So
    # admit the streaming selector only where EITHER width could be asked for,
    # which keeps `_available` from promising a build that then refuses.
    #
    # Not free by construction -- a wider block has a wider tile and so wants
    # more LDS for the same prefetch depth (k=2048 resolves to 72 KiB at 512 and
    # 120 KiB at 1024) -- but `_resolve_lds` trades depth for room, and on gfx950
    # both widths resolve over the whole k range.
    #
    # On gfx942 they do not: a 64 KiB CU holds the half-width build up to k=1024
    # and the full-width one only to k=256, so `all` withdraws the streaming
    # selector for a k the narrow block could still have served. That is the
    # conservative direction and the one this set can express -- it is memoized
    # without the row count, so it cannot know which width will be asked for --
    # and it is a decline the router can act on rather than a launch failure.
    if all(
        topk_per_row_radix_stream_serves(
            k, wave_size, block_threads=bt, lds_budgets=_lds_budgets(arch)
        )
        is None
        for bt in (_STREAM_BLOCK_WIDTHS)
    ):
        out.add("stream")
    # gfx950 k=2048 uniform rows that select every column, or all but a few:
    # one Triton launch each, `_select_whole_rows` and
    # `_stream_small_reject_kernel`.
    if not ragged and k == 2048 and arch == "gfx950":
        if width == k:
            out.add("whole_row")
        elif 0 < width - k <= _STREAM_REJECT_MAX:
            out.add("stream_reject")
    # fp32 only -- the kernel has no half-format build at all. The row-count half
    # of its predicate cannot be asked here, because this set is memoized without
    # the row count; `topk_select_backend` asks `topk_sampled_supports` with the
    # rows it has. Admitting it here and refusing there is the same shape as the
    # streaming selector's two-block-width case above.
    # `sampled_ok` is whether this call may use it at all: the input's own GPU
    # can run it (`_sampled_on_device`: the kernels are traps off gfx950, and
    # one process can hold both kinds) and the caller has not withdrawn it.
    if fp32 and sampled_ok:
        out.add("sampled")
    return frozenset(out)


@lru_cache(maxsize=1024)
def _choose(
    rows: int,
    width: int,
    stride0: int,
    k: int,
    wave_size: int,
    ragged: bool,
    tie: str | None,
    deterministic: bool,
    fp32: bool,
    device: int,
    allow_sampled: bool,
    with_values: bool = False,
) -> str:
    """The backend for one call shape on one GPU, resolved once.

    Every input is a scalar the caller varies rarely, and the whole decision --
    which backends can serve, which the promises leave, which the shape rules
    name -- is a pure function of them. Memoized as one step so the serving path
    is a dict lookup rather than a set build, an intersection and a rule chain.
    """
    allowed = frozenset(_BACKENDS_BY_TIE[tie])
    if deterministic:
        allowed -= _NONDETERMINISTIC
    sampled_ok = allow_sampled and _sampled_on_device(device)
    arch = _device_arch(device)
    available = (
        _available(width, k, wave_size, ragged, fp32, sampled_ok, arch) & allowed
    )
    if not available:
        raise RuntimeError(
            f"no backend serves rows={rows} width={width} topk={k} "
            f"tie={tie!r} deterministic={deterministic} fp32={fp32}"
        )
    return topk_select_backend(
        rows,
        width,
        k,
        available,
        ragged=ragged,
        device=device,
        stride0=stride0,
        with_values=with_values,
    )


@lru_cache(maxsize=1024)
def _plain_multiblock(
    rows: int, stride0: int, k: int, ragged: bool, device: int
) -> bool:
    """Whether plain's launcher takes its multi-block kernel for this call.

    The launcher answers (`topk_plain_use_mulblocks`), through the predicate
    its dispatch runs -- the reach of the gfx950 one-block kernels, the CU
    count and the `TOPK_FORCE_PATH` / `TOPK_DISPATCH_FACTOR` overrides
    included -- so the answer cannot drift from the kernel that runs.
    `stride0` is the row length the launcher is handed: the width for uniform
    rows, the row pitch for ragged ones. Asked only while a shape is being
    routed, never per call, so the overrides are read once per shape.
    `device` keys the memo per GPU, like `_sampled_supports_cached`; the
    launcher itself reads the CU count and architecture once per process.
    """
    return bool(topk_plain_use_mulblocks(rows, stride0, k, ragged))


@lru_cache(maxsize=1024)
def _plain_reads_pitch(rows: int, width: int, k: int, device: int) -> bool:
    """Whether plain reads uniform rows in place at a pitch past their width
    (`topk_plain_reads_pitch`); where it does not, `topk_plain` copies them
    first. `device` keys the memo per GPU, as in `_plain_multiblock`."""
    return bool(topk_plain_reads_pitch(rows, width, k))


@lru_cache(maxsize=1024)
def _plain_reads_pitch_ranged(
    rows: int, width: int, pitch: int, k: int, device: int
) -> bool:
    """Whether plain reads uniform rows at `pitch` in place through its ranged
    form -- every row from column 0 to `width` -- where its uniform kernels do
    not read the pitch (`_plain_reads_pitch`) and `topk_plain` would copy them.

    The ranged form hands the launcher the pitch as the row length. Where both
    forms take the one-block kernel (`_plain_multiblock`) it runs the kernel
    the copy would feed, minus the copy. Wherever the multi-block kernel takes
    either, the pitch decides its speed, not monotonically, and it can run
    slower than the copy, so those keep the copy.
    """
    return not (
        _plain_reads_pitch(rows, width, k, device)
        or _plain_multiblock(rows, width, k, False, device)
        or _plain_multiblock(rows, pitch, k, True, device)
    )


@lru_cache(maxsize=8)
def _plain_width(name: str) -> int:
    """A row length bounding plain's one-block kernels (`topk_plain_width`),
    asked of them rather than kept here, so the bands cannot drift from them."""
    width = topk_plain_width(name)
    if width < 0:
        raise ValueError(f"topk_plain_width does not know {name!r}")
    return width


def _plain_k2048_lds_tail_takes(
    rows: int, width: int, k: int, device: int, stride0: int | None = None
) -> bool:
    """Whether uniform rows past sampled's width door go to plain's LDS tail.

    Only asked where `sampled` could serve, which is what it was measured
    against; see `_PLAIN_K2048_LDS_TAIL_MIN_ROWS`. Rows at a pitch past their
    width only where plain reads them in place; see `_plain_takes`.
    """
    return (
        k == 2048
        and _device_arch(device) == "gfx950"
        and rows >= _PLAIN_K2048_LDS_TAIL_MIN_ROWS
        and _sampled_min_width(device) <= width <= _plain_width("lds_tail_max")
        and (
            stride0 is None
            or stride0 == width
            or _plain_reads_pitch(rows, width, k, device)
        )
    )


def _stream_small_reject(rows: int, rejects: int) -> bool:
    """Whether stream serves a gfx950 k=2048 row as a reject-r problem."""
    return rejects > 0 and (
        rejects <= 4
        or (rows >= 1024 and rejects <= 7)
        or (rows >= 4096 and rejects <= _STREAM_REJECT_MAX)
    )


@lru_cache(maxsize=64)
def _device_cu_num(device: int) -> int:
    """The CU count of one GPU. Per device, like `_device_arch`."""
    return torch.cuda.get_device_properties(device).multi_processor_count


def _plain_takes(
    rows: int,
    width: int,
    k: int,
    device: int | None = None,
    ragged: bool = False,
    stride0: int | None = None,
) -> bool:
    """Enough rows for the row-scaling selector, on a width it is tuned for.

    k=2048 adds bands of its own on gfx950 and gfx942
    (`_plain_gfx950_k2048_takes`, `_plain_gfx942_k2048_takes`).

    Uniform rows at a pitch past their width (`stride0`) are read in place
    where `_plain_reads_pitch` holds -- gfx950's k=2048 one-block kernels and
    the register window -- or through plain's ranged form where
    `_plain_reads_pitch_ranged` does, and copied first elsewhere. On gfx950 the
    backends that read them in place measured faster than that copy and plain
    on every such cell plain was routed (165 of 165, 0.78x, at most 0.99x), so
    plain declines them there; its ranged form beat those backends on 136 of
    the 165 but ran up to 1.19x slower on the rest. Elsewhere plain keeps them.
    """
    if not ragged and stride0 is not None and stride0 != width:
        if device is None:
            device = torch.cuda.current_device()
        if _device_arch(device) == "gfx950" and not _plain_reads_pitch(
            rows, width, k, device
        ):
            return False
    lo, hi = _PLAIN_MANY_ROWS_BAND
    return rows >= _PLAIN_MANY_ROWS and k >= _PLAIN_MIN_K and lo <= width <= hi


def _plain_gfx950_k2048_takes(
    rows: int, width: int, device: int, ragged: bool, stride0: int | None
) -> bool:
    """plain's gfx950 k=2048 band: uniform rows its one-block kernels serve.

    For uniform rows only: those kernels serve calls without row bounds, and
    ragged rows run plain's generic kernel, which takes the general band
    (`_plain_takes`); see `_RAGGED_K2048_STREAM_MAX_WIDTH`. Rows `stream_reject`
    serves through its unmasked branch stay with it.
    """
    if ragged:
        return _plain_takes(rows, width, 2048, device, True, stride0)
    if (
        stride0 is not None
        and stride0 != width
        and not _plain_reads_pitch(rows, width, 2048, device)
    ):
        return False
    if width < _PLAIN_K2048_NARROW_WIDTH:
        rejects = width - 2048
        return width >= _plain_width("any_rows_min") and not (
            rejects <= _SMALL_REJECT_UNMASKED_MAX.value
            and _stream_small_reject(rows, rejects)
        )
    if rows <= _PLAIN_K2048_FEW_ROWS:
        return width <= _plain_width("any_rows_max")
    return width <= _PLAIN_K2048_MAX_WIDTH


def _plain_gfx942_k2048_takes(rows: int, width: int, ragged: bool, device: int) -> bool:
    """plain's gfx942 k=2048 bands; see `_PLAIN_K2048_SMALL_CU`."""
    if _device_cu_num(device) < _PLAIN_K2048_SMALL_CU:
        return (
            rows >= _PLAIN_K2048_SMALL_CU_MIN_ROWS
            and 2048 < width <= _PLAIN_K2048_SMALL_CU_MAX_WIDTH
        )
    return (
        not ragged
        and rows <= _PLAIN_K2048_REG_WINDOW_MAX_ROWS
        and any(lo <= width <= hi for lo, hi in _PLAIN_K2048_REG_WINDOW_BANDS)
        and _plain_width("reg_window_min") <= width <= _plain_width("reg_window_max")
    )


def _sampled_gfx950_k2048_takes(
    rows: int,
    width: int,
    plain_band: bool,
    yield_to_plain: bool,
    device: int,
    stride0: int | None,
) -> bool:
    """`_sampled_takes` for gfx950's k=2048 uniform rows, with that k's doors in
    front of it.

    Below sampled's width door plain's one-block kernels beat it on every cell
    the work door took (200 against 268us at 4096x64K), so there the work door
    yields to plain's band (`plain_band`) when `yield_to_plain` -- plain
    available. Two doors point the other way: rows in plain's band that its own
    dispatch hands to its multi-block kernel (`_plain_multiblock`, asked at the
    width, which sizes the launch at any pitch), and the few-row band's rows
    past its width, which decode served 1.47x-2.56x slower. See
    `_PLAIN_K2048_LDS_TAIL_MIN_ROWS`.
    """
    pitch = width if stride0 is None else stride0
    if pitch < width:
        return False
    if width < _sampled_min_width(device):
        if (plain_band and _plain_multiblock(rows, width, 2048, False, device)) or (
            rows <= _PLAIN_K2048_FEW_ROWS and width > _plain_width("any_rows_max")
        ):
            return _sampled_supports_cached(rows, pitch, 2048, device)
        if yield_to_plain and plain_band:
            return False
    return _sampled_takes(rows, width, 2048, device=device, stride0=stride0)


def _route_gfx950_k2048(
    rows: int,
    width: int,
    available: frozenset[str],
    ragged: bool,
    device: int,
    stride0: int | None,
) -> str:
    """Every gfx950 k=2048 band, in the order they are tried, then the general
    tail. Uniform rows past sampled's width door at 3000+ rows go to plain's LDS
    tail, ragged rows to `_ragged_k2048_backend`, and other uniform rows to
    `sampled` or plain where their k=2048 doors take them."""
    if (
        not ragged
        and "sampled" in available
        and "plain" in available
        and _plain_k2048_lds_tail_takes(rows, width, 2048, device, stride0)
    ):
        return "plain"
    if ragged and _ragged_k2048_backend(rows, width) in available:
        return _ragged_k2048_backend(rows, width)
    plain_band = _plain_gfx950_k2048_takes(rows, width, device, ragged, stride0)
    if (
        not ragged
        and "sampled" in available
        and _sampled_gfx950_k2048_takes(
            rows, width, plain_band, "plain" in available, device, stride0
        )
    ):
        return "sampled"
    if "plain" in available and plain_band:
        return "plain"
    return _route_tail(rows, width, 2048, available)


def _route_tail(rows: int, width: int, k: int, available: frozenset[str]) -> str:
    """The rules every call ends on once the bands that scale with work decline."""
    if "small_k" in available and k <= _SMALL_K_MAX_K:
        return "small_k"
    if "decode" in available and _decode_takes(rows, width, k):
        return "decode"
    if "stream" in available:
        return "stream"
    # Every rule declined: `tie` or `deterministic` narrowed the set to backends
    # the shape rules never reach. Naming one outside `available` breaks the
    # promise the narrowing was made to keep -- `tie='high'` once fell through
    # here and was served by decode, which ties the opposite way, silently.
    return next(b for b in _PREFERENCE if b in available)


def _stream_specialised(
    backend: str,
    rows: int,
    width: int,
    ragged: bool,
    with_values: bool,
    available: frozenset[str],
) -> str:
    """`stream`, or the gfx950 k=2048 kernel that serves its uniform rows faster
    where they select every column (`whole_row`) or all but a few
    (`stream_reject`); see `_WHOLE_ROW_INDEX_ROWS` and `_stream_small_reject`."""
    if backend != "stream" or ragged:
        return backend
    if "whole_row" in available and (
        with_values or _WHOLE_ROW_INDEX_ROWS[0] <= rows <= _WHOLE_ROW_INDEX_ROWS[1]
    ):
        return "whole_row"
    if "stream_reject" in available and _stream_small_reject(rows, width - 2048):
        return "stream_reject"
    return backend


def _ragged_k2048_backend(rows: int, width: int) -> str | None:
    """The backend for ragged k=2048 rows on gfx950, or None to leave them to
    the general rules; see `_RAGGED_K2048_STREAM_MAX_WIDTH`."""
    if width <= _RAGGED_K2048_STREAM_MAX_WIDTH:
        return "stream"
    if width <= _RAGGED_K2048_DECODE_MAX_WIDTH and rows < _PLAIN_MANY_ROWS:
        return "decode"
    return None


def _decode_takes(rows: int, width: int, k: int) -> bool:
    """Room to spread a grid across, and a grid that is not already full."""
    min_n, max_m = _DECODE_GATE_AMORTISED if k >= _DECODE_AMORTISING_K else _DECODE_GATE
    band_lo, band_hi = _DECODE_ANY_ROWS_BAND
    return (width >= min_n and rows <= max_m) or (
        k >= _DECODE_ANY_ROWS_K and band_lo <= width <= band_hi
    )


def topk_select_backend(
    rows: int,
    width: int,
    k: int,
    available: frozenset[str],
    ragged: bool = False,
    device: int | None = None,
    stride0: int | None = None,
    with_values: bool = False,
) -> str:
    """Name the backend to use for this shape among those that can serve it.

    `ragged` is whether the caller gave row ends. Ragged rows never go to
    `sampled`, whose time follows the pitch rather than where rows end; see
    `_sampled_takes`.

    `with_values` is whether the caller wants the values back, which `whole_row`
    writes in the same launch.

    k=2048 has bands of its own on gfx950 (`_route_gfx950_k2048`) and gfx942
    (`_plain_gfx942_k2048_takes`); every other call takes the rules below.

    `stride0` is the input's row pitch, the width for dense rows or a single
    row, and the default. `sampled` reads every row in place at it; plain's
    ranged form reads a row-strided view in place at it too, as do its uniform
    one-block kernels where `_plain_reads_pitch` holds, and plain's launcher
    sizes its multi-block choice by the length it launches with.

    The shape of the answer: `plain` takes the many-row middle, where it is the
    only one that scales with rows rather than against them; the small-k selector
    takes everything its narrowing still bites on; decode takes the rest of the
    low-row end, over any row wide enough to spread a grid across; and the
    streaming selector takes what is left, which is the many-row end outside
    plain's band.

    Fitted to a 565-cell sweep -- rows 1..16384, widths 2048..1M including the
    non-power-of-two widths a sparse indexer produces, k 16..4096 -- against the
    fastest backend measured at each cell. Re-swept twice since, because
    `_dispatch` twice changed what it hands a backend: the streaming selector's
    re-select trigger learned a row-count term, and then its `partial` split was
    retuned and went from unreachable to live, which moved `stream` by up to
    2.58x on the low-row end. A mean 1.014x and a p90 1.020x against the oracle,
    more than 1.2x off on 12 cells, and 1.009x of it by total time.

    The worst cell is 1.564x: `plain` takes 256 rows of 8192 and decode is
    faster there. Every way of trimming that band regresses somewhere else by
    0.631x..0.707x for a mean gain of at most 1.004x, so it stands.

    Two rules for changing any of this. Score candidates as an A/B against the
    rule in place, not only against the per-cell oracle -- the two disagreed
    four times over this table, and each time the oracle preferred a rule that
    made some shape markedly slower. And re-run the SWEEP, not just
    `topk_backend_fit.py`, whenever `_dispatch` changes what it hands a backend;
    twice now that has moved a boundary the fit alone would have kept.

    Every band here was timed with
    `op_tests/op_benchmarks/hip/bench_topk_select_routing.py`, which also
    compares two trees cell by cell; rerun it rather than nudging a number.

    The returned name is always one of `available`.
    """
    if not available:
        raise ValueError("topk_select_backend needs at least one backend")
    if device is None:
        device = torch.cuda.current_device()
    # k=1 first and unconditionally: the others answer it by building machinery
    # the answer does not need, and lose 1.3x to 12x doing so.
    if "argmax" in available:
        return "argmax"
    if k == 2048:
        arch = _device_arch(device)
        if arch == "gfx950":
            return _stream_specialised(
                _route_gfx950_k2048(rows, width, available, ragged, device, stride0),
                rows,
                width,
                ragged,
                with_values,
                available,
            )
        if (
            arch == "gfx942"
            and "plain" in available
            and _plain_gfx942_k2048_takes(rows, width, ragged, device)
        ):
            return "plain"
    if (
        not ragged
        and "sampled" in available
        and _sampled_takes(rows, width, k, device=device, stride0=stride0)
    ):
        return "sampled"
    if "plain" in available and _plain_takes(rows, width, k, device, ragged, stride0):
        return "plain"
    return _route_tail(rows, width, k, available)


def _reject_unsupported(
    *,
    input,
    topk,
    indices_type,
    idx_oob_fill_value,
    abort_when_nan_found,
    begin,
    hint,
    tie,
    sorted,
    sorted_index,
):
    """Refuse what this cannot do, rather than quietly doing something else.

    Returns `input.stride()`, which the entry also routes on: read once, as the
    tuple measured 70ns a call and `input.stride(dim)` 143ns each.
    """
    if begin is not None:
        raise NotImplementedError("`begin` is not supported (nor is it in DeepSelect)")
    if hint is not None:
        raise NotImplementedError("`hint` is not supported (nor is it in DeepSelect)")
    if input.dim() != 2 or input.dtype not in ARGMAX_DTYPES:
        raise ValueError(
            f"input must be 2-D and one of "
            f"{builtins.sorted(str(d) for d in ARGMAX_DTYPES)}; got "
            f"{tuple(input.shape)} {input.dtype}"
        )
    if input.dtype is not torch.float32 and topk != 1:
        # Named here rather than left to `_choose`, whose "no backend serves"
        # reads as a geometry problem. Only the k=1 reduction has a half-format
        # build: it folds int32 ordering keys, so it can widen each element as
        # it reads it. The selectors compare and re-read the scores themselves,
        # so for them a half format is a second set of kernels, not a load.
        raise NotImplementedError(
            f"{input.dtype} is served only at topk=1, the reduction; got "
            f"topk={topk}. Cast to float32 for a wider selection."
        )
    strides = input.stride()
    if strides[1] != 1:
        raise ValueError("input must have inner stride 1")
    if indices_type is not torch.int32:
        raise NotImplementedError(
            f"indices_type={indices_type}: the kernels emit int32. DeepSelect "
            "defaults to int64; cast the returned tensor if you need it."
        )
    if idx_oob_fill_value != -1:
        raise NotImplementedError(
            f"idx_oob_fill_value={idx_oob_fill_value}: the kernels pad short rows "
            "with -1. DeepSelect defaults to 2147483647."
        )
    if abort_when_nan_found:
        raise NotImplementedError(
            "abort_when_nan_found=True needs the NaN count read back on the host, "
            "and this path may not synchronise. NaN outranks +inf here, which is "
            "what torch.topk does on narrow rows; DeepSelect aborts instead."
        )
    if tie not in _BACKENDS_BY_TIE:
        raise ValueError(f"tie must be None, 'low' or 'high'; got {tie!r}")
    if sorted and sorted_index:
        # Descending by value and ascending by index are two different orders of
        # the same pairs; honouring both would mean returning a `values` that
        # does not line up with `idx`.
        raise ValueError(
            "sorted=True and sorted_index=True ask for two different orderings "
            "of the same (value, index) pairs; pick one"
        )
    return strides


def topk_select(
    input: torch.Tensor,
    topk: int,
    sorted: bool = False,
    begin: torch.Tensor | None = None,
    end: torch.Tensor | None = None,
    indices_type: torch.dtype = torch.int32,
    sorted_index: bool = False,
    hint: torch.Tensor | None = None,
    output_idx: torch.Tensor | None = None,
    output_idx_offset: torch.Tensor | None = None,
    idx_oob_fill_value: int = -1,
    value_oob_fill_value: float = float("-inf"),
    return_value: bool = False,
    abort_when_nan_found: bool = False,
    tie: str | None = None,
    deterministic: bool = False,
) -> tuple[torch.Tensor | None, torch.Tensor]:
    """Per-row top-k, dispatched across aiter's five selectors.

    Parameter names, order and return shape follow ``deep_select.topk``, and
    the return shape means the 2-tuple specifically: DeepSelect returns
    ``(values, indices)`` always, and its ``return_value=False`` sets the first
    element to None rather than changing the arity. So does this.

    Checked against DeepSelect v1.0.0 / main at 0f03b68 (2026-09-10): the first
    fourteen parameters match by name, order and default except where the table
    below says otherwise. Re-check before trusting it -- that repository is days
    old and moving. Two traps found while checking: its README's usage snippet
    passes ``sorted_index=True`` and ``indices_type=torch.int32`` explicitly, and
    neither is the default; and DeepWiki's generated page reports the
    ``indices_type`` default as int32, which is wrong.

    Four defaults differ, because this cannot honour DeepSelect's and will not
    pretend to -- each raises if you pass DeepSelect's value:

    ======================  ============  =========  =============================
    parameter               DeepSelect    here       why
    ======================  ============  =========  =============================
    ``abort_when_nan_found``  ``True``    ``False``  aborting needs a device-to-host
                                                     read, which this path may not do
    ``idx_oob_fill_value``    2147483647  ``-1``     what the kernels write
    ``indices_type``          int64       int32      what the kernels emit
    ``input`` dtype           bf16/fp32   see below  half formats only at ``topk=1``
    ======================  ============  =========  =============================

    fp32 is served at every k. bf16 and fp16 are served at ``topk=1`` and raise
    past it: the reduction folds int32 ordering keys, so it widens each element
    as it reads it, while the selectors compare and re-read the scores and would
    need a second set of kernels. At k=1 this is worth having rather than
    casting -- the row is half the bytes, and the cast is its own pass over it.

    That row is the one divergence that is not a narrowing. fp16 is not a
    DeepSelect dtype at all -- its ``csrc/api.cpp`` takes bfloat16 or float32 and
    rejects the rest -- so at ``topk=1`` this accepts an input DeepSelect will
    not, while above ``topk=1`` it accepts less. Do not read the table as "a
    subset of DeepSelect" in either direction.

    ``return_value`` also defaults the other way -- ``False`` here, ``True`` in
    DeepSelect -- and this one does not raise, since asking for the values back
    is still honoured. No backend produces the values: they are gathered from
    ``input`` afterwards, which is a separate kernel plus a compare and a select
    over ``[rows, topk]``, and it is pure overhead for the callers that only
    route on the indices. Measured at rows=1024, width=32768, topk=16: 39.8us of
    selection under 74.3us of call, so the gather and the buffers around it were
    most of the time spent. Pass ``return_value=True`` to get them.

    NaN: +nan ranks above +inf, as in ``torch.topk``, and a NaN with the sign
    bit set does too, on every backend.

    Args:
        input: ``[rows, width]``, inner stride 1, any row stride. float32 at
            any ``topk``; bfloat16 and float16 at ``topk=1`` only.
        topk: elements to select per row.
        sorted: sort the returned values descending. Done on the host.
        end: ``[rows]`` int32 exclusive right bound per row, DeepSelect's
            ``end``; this is each row's live length. Defaults to the full width,
            which every backend takes from the input's shape: no length tensor
            is built or read, so a captured graph holds no address for one.
        sorted_index: sort the returned indices ascending. Done on the host.
        output_idx: ``[rows, topk]`` int32 to write into, any layout; allocated
            if omitted.
        output_idx_offset: ``[rows]`` int32 added to every live index.
        return_value: gather the selected scores and return them. Off by
            default; see above. ``sorted`` still works without it -- the values
            are gathered to derive the order and then dropped.
        tie: ``None`` leaves which of several equal scores wins unspecified, as
            DeepSelect does, and lets every backend run. ``'low'`` and ``'high'``
            promise the smallest or largest column and restrict the backend set,
            which can cost up to 1.9x. Either implies ``deterministic``.
        deterministic: the *set* of selected columns is a function of the row
            alone -- the same input selects the same columns on every call, and a
            row selects the same columns wherever it sits in the batch. The order
            they are returned in is not promised, and does move between calls:
            the streaming selector takes its output slots from a shared counter,
            so a row with more ties at the cut than places permutes. Pass
            ``sorted=True`` or ``sorted_index=True`` if the returned tensor
            itself has to be reproducible, not just its contents.

            Weaker than ``tie`` and cheaper than it: it excludes ``plain`` and
            ``sampled`` (`_NONDETERMINISTIC`), keeping the small-k selector that
            ``tie='low'`` has to give up. Measured on gfx950, it costs up to 3.4x
            where ``plain`` would have won (k=2048, m=4096 n=16384: 75 against
            258us on ``stream``) and up to 3.7x on the wide rows where
            ``sampled`` would have (k=2048, m=64 n=1M: 57 against 210us on
            ``decode``).

    Returns:
        ``(values, indices)``; ``values`` is None when ``return_value`` is False.
    """
    strides = _reject_unsupported(
        input=input,
        topk=topk,
        indices_type=indices_type,
        idx_oob_fill_value=idx_oob_fill_value,
        abort_when_nan_found=abort_when_nan_found,
        begin=begin,
        hint=hint,
        tie=tie,
        sorted=sorted,
        sorted_index=sorted_index,
    )
    unsupported = _unsupported_arch()
    if unsupported is not None:
        raise RuntimeError(f"topk_select is not supported on {unsupported}")
    rows, width = input.shape
    if not 1 <= topk <= width:
        raise ValueError(f"topk must be in [1, {width}], got {topk}")
    # Read once: every `input.device` builds a new device object.
    device = input.device
    dev = device.index
    if dev is not None and dev != torch._C._cuda_getDevice():
        # The Triton and FlyDSL selectors launch on the current GPU.
        with torch.cuda.device(dev):
            return topk_select(
                input,
                topk,
                sorted=sorted,
                begin=begin,
                end=end,
                indices_type=indices_type,
                sorted_index=sorted_index,
                hint=hint,
                output_idx=output_idx,
                output_idx_offset=output_idx_offset,
                idx_oob_fill_value=idx_oob_fill_value,
                value_oob_fill_value=value_oob_fill_value,
                return_value=return_value,
                abort_when_nan_found=abort_when_nan_found,
                tie=tie,
                deterministic=deterministic,
            )

    if end is None:
        # Every row is full-width: each backend takes that from the input's
        # shape, or builds its own [start, end) pair where it needs one.
        row_lens = None
    else:
        if end.shape != (rows,) or end.dtype != torch.int32:
            raise ValueError(f"end must be int32 [{rows}], got {tuple(end.shape)}")
        # Every backend reads the row ends as one dense array.
        row_lens = end.contiguous()
    idx = (
        torch.empty((rows, topk), dtype=torch.int32, device=device)
        if output_idx is None
        else output_idx
    )
    if idx.shape != (rows, topk) or idx.dtype != torch.int32:
        raise ValueError(
            f"output_idx must be int32 [{rows}, {topk}], got {tuple(idx.shape)}"
        )
    if not idx.is_contiguous():
        # The backends write dense rows of `topk`; the answer is copied into the
        # caller's buffer at the end, like a reordered one.
        idx = torch.empty((rows, topk), dtype=torch.int32, device=device)

    # The row pitch the rows sit at in memory; a single row has none to speak
    # of and reads as dense. `topk_select_backend` derives from it the pitch
    # each backend that reads rows in place would launch at.
    stride0 = strides[0] if rows > 1 else width
    with_values = return_value or sorted
    backend = _choose(
        rows,
        width,
        stride0,
        topk,
        wave_size_of(dev),
        end is not None,
        tie,
        deterministic,
        input.dtype is torch.float32,
        dev,
        True,
        with_values=with_values,
    )
    # Read only when `sampled` won: the lookup costs more than the memo hit, and
    # every other call has no use for it.
    if (
        backend == "sampled"
        and os.environ.get("AITER_DISABLE_TOPK_SAMPLED", "0") == "1"
    ):
        backend = _choose(
            rows,
            width,
            stride0,
            topk,
            wave_size_of(dev),
            end is not None,
            tie,
            deterministic,
            input.dtype is torch.float32,
            dev,
            False,
            with_values=with_values,
        )
    gathered = _dispatch(
        backend,
        input,
        row_lens,
        idx,
        topk,
        rows,
        tie,
        deterministic,
        with_values=with_values,
    )

    values = None
    if return_value or sorted:
        # Gather before any offset is applied: the offset renumbers the output
        # for a sharded vocabulary, but the values still live in this tensor,
        # and gathering with shifted indices reads off the end of the row.
        #
        # `sorted` orders the pair by value, so the values are needed to derive
        # that order even when the caller does not want them back. Gathering
        # them and dropping them is the cost of asking for the order; returning
        # indices in an arbitrary order from `sorted=True` is not an option.
        if gathered is None:
            gathered = _gather_selected(input, idx, value_oob_fill_value)
        if sorted:
            gathered, order = torch.sort(gathered, dim=1, descending=True)
            idx = idx.gather(1, order)
        if return_value:
            values = gathered
    if sorted_index:
        # Reorder the values with it: `values[j]` is the score at `idx[j]`, and
        # sorting one of the pair alone silently breaks that. Note padded slots
        # carry -1 and so sort to the front; DeepSelect's 2147483647 default
        # sends them to the back instead.
        idx, order = torch.sort(idx, dim=1)
        if values is not None:
            values = values.gather(1, order)
    if output_idx_offset is not None:
        # A padded slot keeps its sentinel; only live indices move. Per-row
        # constant, so it cannot disturb an ordering applied above.
        idx = idx + (idx >= 0).to(torch.int32) * output_idx_offset.view(rows, 1)
    if output_idx is not None and idx.data_ptr() != output_idx.data_ptr():
        # Sorting and offsetting build new tensors; the caller asked for its own
        # buffer to hold the answer, so put it back.
        output_idx.copy_(idx)
        idx = output_idx
    return values, idx


def _dispatch(
    backend,
    input,
    row_lens,
    idx,
    topk,
    rows,
    tie=None,
    deterministic=False,
    with_values=False,
):
    """Run `backend` into `idx`; returns the selected values where the backend
    writes them itself (`whole_row` with `with_values`), else None."""
    # Every backend puts rows on a grid dimension, and HIP rejects a zero-sized
    # launch. An empty batch has nothing to select, so `idx` stays [0, topk].
    if rows == 0:
        return None
    # `row_lens` None means every row is full-width; each FlyDSL backend takes
    # that length from the input's shape and reads no length tensor.
    ragged = row_lens is not None
    if backend == "argmax":
        topk_per_row_argmax(input, row_lens, idx)
    elif backend == "sampled":
        # The kernel takes a [start, end) pair per row, both int32, and emits
        # indices only -- `topk_select` gathers the values itself further down,
        # so `values=None` here rather than scratch nobody reads.
        #
        # Its entry takes the pair even without an `end` -- it checks both have
        # `rows` entries, and reads them for a row-strided call -- so that case
        # gets `_default_bounds`; with one, the caller's ends beside the cached
        # zero starts. topk_select has already refused ends that are not int32
        # `[rows]`.
        if ragged:
            starts, ends = _zero_rows(rows, input.get_device()), row_lens
        else:
            starts, ends = _default_bounds(rows, input.shape[1], input.get_device())
        # `ragged` is `end is not None` -- whether the CALLER gave row bounds.
        # When it did not, the pair above is 0 and the full width for every row,
        # and saying so lets the entry run the kernels that do not bounds-check
        # every element. nan_high: NaN of either sign outranks +inf here, as in
        # stream and decode.
        #
        # topk_select built or checked every tensor here -- fp32 input with
        # inner stride 1, dense int32 [rows] bounds, a dense [rows, topk] output
        # -- and `_choose` asked topk_sampled_supports for this device at this
        # pitch, so the entry's validation would only repeat it. The one thing
        # not checked is the device of a caller's `end` (the default one is
        # built on input's); the entry reports that one.
        pitch = input.stride(0) if rows > 1 else input.shape[1]
        if not ragged or ends.get_device() == input.get_device():
            _sampled_run(
                input,
                starts,
                ends,
                idx,
                None,
                rows,
                pitch,
                input.stride(1),
                topk,
                None,
                ragged,
                True,
            )
        else:
            top_k_per_row_prefill_sampled(
                input,
                starts,
                ends,
                idx,
                None,
                rows,
                pitch,
                input.stride(1),
                k=topk,
                ragged=ragged,
                nan_high=True,
            )
    elif backend == "small_k":
        topk_per_row_small_k(input, row_lens, idx, topk)
    elif backend == "plain":
        # plain takes a [start, end) pair, not a length, and an empty
        # `rowStarts` with a real `rowEnds` reads as "no range" -- silently over
        # the whole row. Pass the pair only when the rows really differ, or
        # when uniform rows sit at a pitch the uniform form would copy them
        # past and the ranged form reads them in place on the same kernel
        # (`_plain_reads_pitch_ranged`). Off the radix path the ranged overload
        # loops per row -- 294918 launches against 25 -- but plain is admitted
        # only on it, and C++ decides that from the width, not the pitch.
        # The values are never read: whatever this entry returns is gathered
        # from `input` at `idx` below, because that is the only form that stays
        # consistent through the reorderings. So ask for the build that does not
        # write them, which `_available` admits plain only for, with NaN of
        # either sign ranked above +inf.
        width = input.shape[1]
        pitch = input.stride(0) if rows > 1 else width
        if ragged:
            starts = _zero_rows(rows, input.get_device())
            topk_plain(
                input, idx, None, topk, True, starts, row_lens, -1, 1, nan_high=True
            )
        elif pitch > width and _plain_reads_pitch_ranged(
            rows, width, pitch, topk, input.get_device()
        ):
            starts, ends = _default_bounds(rows, width, input.get_device())
            topk_plain(input, idx, None, topk, True, starts, ends, -1, 1, nan_high=True)
        else:
            empty = no_row_lens(input.device)
            topk_plain(input, idx, None, topk, True, empty, empty, -1, 1, nan_high=True)
    elif backend == "decode":
        # `stable` buys the smallest-index tie-break AND the selected set being
        # a function of the row, so both `tie='low'` and `deterministic` need
        # it. Only "no promise at all" may drop it.
        #
        # The second half of that was measured, not assumed: without `stable`
        # this backend selects a DIFFERENT set of tied columns from one call to
        # the next, which `test_invariants` catches at 16 rows of 262144, k=2048
        # as "selects the same set under {'deterministic': True}". A radix
        # select being a pure function of the row is the obvious guess and it is
        # wrong.
        #
        # Hardwired on, it cost 0-20% -- and the routing was fitted against a
        # decode that was always paying it, which is part of why the old rules
        # sent these shapes elsewhere.
        flydsl_top_k_per_row_decode(
            input,
            1,
            row_lens,
            idx,
            rows,
            input.stride(0),
            1,
            topk,
            stable=tie == "low" or deterministic,
        )
    elif backend == "stream":
        topk_per_row_radix_stream(
            input,
            row_lens,
            idx,
            topk,
            lds_budgets=_lds_budgets(_device_arch(input.device.index)),
        )
    elif backend == "stream_reject":
        # A dense row with a small N-K tail is a reject-r problem, not a
        # general selection problem.  Reduce the bottom tail and emit its
        # complement directly; this preserves the stream backend's
        # (value desc, column asc) selected set while avoiding its three
        # radix passes and candidate-buffer traffic.  Keep gfx942 on the
        # generic path until this geometry is measured there as well.
        _stream_small_reject_kernel[(rows,)](
            input,
            idx,
            input.stride(0),
            idx.stride(0),
            WIDTH=input.shape[1],
            REJECTS=input.shape[1] - topk,
            BLOCK_N=4096,
            BLOCK_K=2048,
            num_warps=4,
        )
    elif backend == "whole_row":
        return _select_whole_rows(input, idx, with_values)
    else:
        raise ValueError(f"unknown backend {backend!r}")
    return None
