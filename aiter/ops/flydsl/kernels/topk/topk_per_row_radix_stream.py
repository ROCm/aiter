# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Per-row TopK for a large k: read the row once, sort only what survives.

A row of N fp32 scores, k in the hundreds to low thousands, N up to a million.
The selectors this replaces walk the row once per radix pass and once more to
scatter -- four trips to HBM for a row that is otherwise perfectly streamable.

Here the row is read once. An initial window fills a candidate buffer and a
radix select over it gives a threshold; after that an element is kept only if it
beats the threshold, which almost none do, and the buffer is re-selected only
when it fills. The selection work is therefore proportional to the candidates,
not to N.

The threshold is sound at every point: it is the worst element of the k held so
far, so anything below it already has k better elements and cannot enter the
answer. It is a (key, column) pair rather than a key, because the answer is
defined to a finer order than the key alone -- see below.

The select is three radix passes of 11 / 11 / 10 bits on an order-preserving
32-bit key.

The answer is the k largest ordered by (value descending, column ascending),
which makes it a function of the input alone: identical across runs, and also
across block width, LDS budget and occupancy. That matters because those are
tuning knobs, and an answer that moved when they were turned would make a
retune a silent behaviour change.

Ties are what that order is for, and they cost nothing until they bite. When
more candidates share the cut value than there are places left, a second select
over ~column picks the smallest columns; columns are unique, so it lands on
exactly the right number and needs no tie rule of its own. On continuous input
the cut value occurs once, the branch is not taken, and the whole guarantee is
free. The alternative -- deterministic placement by prefix sum, as DeepSelect
does -- is reproducible but not canonical, since the answer it reproduces is
still a function of the thread geometry that placed it.

Output order is unspecified, matching ``torch.topk(sorted=False)``: the winners'
slots come from atomic counters, one filling from each end. Only the set is
canonical.

A row is one workgroup, which is the right shape only while there are enough
rows to fill the machine. Below that the `partial` mode splits one row across G
workgroups, each selecting its own slice; the survivors are then re-selected by
the same kernel in its ordinary mode, over a [rows, G*k] array of values.

G is set by how empty the machine is, not by balancing the two halves. The
balance argument gives N/G = G*k, i.e. G = sqrt(N/k), and measurement does not:
the best G puts `rows * G` near 256 at k=2048 and near 512 at k=512, whatever
the row width -- a smaller k claims less LDS, so more workgroups fit a CU and it
takes more of them to fill one. Which is the same thing the mode exists for,
stated in the units the hardware has.

`partial` writes the winning values rather than their keys so that the merge is
an ordinary fp32 selection; it re-reads each winner from the row to get that
value, which is k gathered loads against a slice of N/G.

The split does not weaken the guarantee above, and that takes one more piece.
A slot's position inside a part comes from an atomic counter, so a merge that
ordered by position would break ties by nothing in particular and differently
on a rerun. `labelled` is the fix: the merge is told each slot's ORIGINAL
column and orders by that, so a split answer is the unsplit answer, column for
column. The union it selects from is complete for the same reason the threshold
is sound -- a row's j-th largest has at most j-1 elements above it anywhere,
hence at most j-1 inside its own slice, so every global winner is a winner of
its slice, and the per-slice tie rule is the same one the whole row uses.
"""

from functools import cache, lru_cache

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import (
    Float32,
    Int32,
    arith,
    gpu,
    range_constexpr,
)
from flydsl.expr import rocdl as fly_rocdl
from flydsl.expr.typing import T

from aiter.jit.utils.chip_info import get_gfx_runtime, get_lds_capacity_bytes
from aiter.ops.flydsl.kernels.kernels_common import (
    atomic_add_i32,
    atomic_max_i32,
    kernel_signature,
)
from aiter.ops.flydsl.kernels.tensor_shim import buf_copy_atom

_VEC = 4
# Half the CU's threads, so two or three workgroups stay resident and one
# streams tiles through another's barriers. Worth 1.3x..1.5x over a full-width
# block across N=64K..1M and k=512/2048 -- in the regime where
# `topk_per_row_radix_stream_block_threads` keeps it, which is many rows below
# k=4096.
#
# Soak any change to this with `topk_stream_soak.py`. The width decides how many
# waves share the block's barriers and so how far one can drift ahead of another,
# which is what decided whether a race in the streaming loop was visible at all;
# the fault hit a few rows in ten thousand and only above a row count, so a smoke
# test passes on a broken build.
_BLOCK_THREADS = 512
# The full width wins in two regimes, and the rule is ROWS and k -- not width.
# Fitted over a 195-cell sweep of both widths (rows 16..16384, widths 16K..1M,
# k 16..4096, `bt_grid.csv`): `k >= 4096 or rows <= 256` costs a mean 1.001x and
# a worst 1.12x against picking the better width per cell, where holding 512
# everywhere costs 1.135x / 2.03x.
#
#   k = 4096      the full width wins 39 of 39 cells, ratio 0.76..0.85
#   rows <= 256   it wins 24 of 24 at every k below 4096, median 0.86
#   rows >= 512   it LOSES, median 1.47 at 512 rising to 2.12 at 16384
#
# A width term was tried and scores identically, so it is not in the rule: the
# apparent width effect was rows in disguise. A first pass at this shipped
# `width >= 262144` off two low-row samples and was 2.00x off at
# 4096x262144 k=256 -- the regime it had never measured.
#
# Re-checked after the placement passes were merged, the third radix pass learned
# to exit early, and `topk_per_row_radix_stream_lds_plan` arrived -- all three
# change what a width costs, and the rule was fitted against none of them. Over
# the 100 cells of `stream_knob_grid.csv`, which timed both widths across the
# knob space, it picks the better width on 100 of 100. That is only worth
# anything because the two widths are far apart there: they differ by a median
# 1.40x and by more than 1.2x on 86 of the 100, with no cell inside 1.02x. A
# perfect score on a comparison that cannot discriminate would mean nothing.
#
# Both block widths soak clean over 338484 rows at k=16..2048.
_WIDE_BLOCK_THREADS = 1024
_WIDE_BLOCK_MAX_ROWS = 256
_WIDE_BLOCK_MIN_K = 4096
# 11 + 11 + 10 covers the 32-bit key in three passes. 2048 buckets is 8 KiB of
# LDS, divided evenly across the block, so the scan is one wave prefix plus a
# fold over the wave totals at any block width.
#
# One bucket per thread (nine bits, four passes at 512 threads) makes each scan a
# single read and clear, and is slower: 39.5us became 54.5us at rows=1024,
# width=32768, k=16. The selection itself did not get cheaper -- 6.0us against
# 5.7 -- and 13us of the loss landed in the streaming loop, which the extra pass
# does not appear in. That 13us is unexplained; occupancy is not the answer (28
# VGPRs, 26.5 KiB LDS, 8 waves per SIMD -- the hardware cap -- and the four-pass
# build's LDS was smaller). Do not retry without an account of it.
_NUM_BUCKETS = 2048
_MID_SHIFT = 10
_LOW_MASK = (1 << _MID_SHIFT) - 1
_HIGH_SHIFT = 21

# Re-select once the arrivals reach this many. Larger trades LDS for fewer
# selects; k is the natural scale, capped so a large k still leaves room.
_SOFT_TRIGGER_MAX = 2048
# Below this the re-selects come often enough to cost more than the LDS they free.
_SOFT_TRIGGER_MIN = 256


# Loads a thread keeps in flight per group. Registers only, since arrivals are
# drained between tiles, so this is not a claim on LDS. A large k re-selects
# often and from a bigger buffer, so it has more latency to cover and pays for a
# deeper queue; a small k is nearly all streaming and a deep queue only adds
# barriers. Measured at N=64K..1M: depth 8 is worth 1.1x..1.3x at k=2048 and
# loses 1.2x at k=512, where depth 2 is best.
def _prefetch_tiles(k):
    return 8 if k >= 1024 else 2


# How much LDS one workgroup may claim: enough for two to be resident so one
# covers the other's barriers, and a solo ceiling for a k too large to hold
# twice. Asked of the DEVICE, not written down: `topk_select` advertises gfx942
# at 64 KiB against gfx950's 160 and routes without an arch term, so literals
# sized for the larger card build, above k=1024, a module the smaller one cannot
# launch -- loudly, but only after `serves` has promised the router otherwise.
# On gfx950 these reproduce the 76 KiB and 148 KiB they replace, which is where
# the prefetch depths below were tuned.
_LDS_RESIDENT_SLACK = 4 * 1024
_LDS_SOLO_SLACK = 12 * 1024


@cache
def _lds_budgets() -> tuple[int, int]:
    """(two-resident, solo) LDS bytes one workgroup may claim on this device."""
    # Runtime arch, not `get_gfx()`: that honours `GPU_ARCHS` while FlyDSL
    # compiles for the live device, so the two would size for different cards.
    cap = get_lds_capacity_bytes(get_gfx_runtime())
    return cap // 2 - _LDS_RESIDENT_SLACK, cap - _LDS_SOLO_SLACK


_INT32_MIN = -2147483648
_INF_BITS = 0x7F800000
# 0xFFFFFFFF: the top of the unsigned key space, one above +inf's key. Every NaN
# lands here, so NaN wins a selection it enters -- see `_ord_unsigned`.
_NAN_KEY = -1

_ST_CUT_HI = 0
_ST_CUT_MID = 1
_ST_CUT_LOW = 2
_ST_ABOVE = 3
_ST_ARRIVED = 4
_ST_KEPT = 5
# Largest column among the elements held at the cut value: the second half of
# the threshold, and the column cut that decides which ties are held at all.
_ST_THR_COL = 6
# The key half of the same threshold.
_ST_THR = 8
# Population of the bucket the pivot landed in. On the last radix pass that is
# the number of candidates sharing the cut key exactly.
_ST_BUCKET_CNT = 7
# Ties placed so far. The strict winners and the ties need separate counters to
# share a pass -- see `compact`.
_ST_TIED = 9
_ST_SLOTS = 16
_INT32_MAX = 2147483647


def _ugt(a, b):
    """`a > b` on int32 bit patterns read as unsigned.

    The radix walk is most-significant bit first, which only terminates on an
    unsigned order, and `>` on an fx integer is signed.
    """
    return arith.cmpi(arith.CmpIPredicate.ugt, a, b)


def _ord_unsigned(value):
    """Map fp32 to a uint32 bit pattern that compares the same way, NaN highest.

    fp32 is sign-magnitude, so flipping the magnitude bits of negatives yields a
    total order; the extra sign flip puts it in unsigned space, where the radix
    walk can start from the top bit. -0.0 and 0.0 are one value with two bit
    patterns and must not become two keys.

    NaN sorts above +inf, matching `torch.topk` and the decode selector. The
    dispatcher picks between those paths by shape, so a row holding a NaN would
    otherwise answer differently depending on a choice the caller did not make.
    Every NaN payload collapses onto the single key `_NAN_KEY`, so NaNs tie with
    each other and the column rule orders them -- `torch.topk` leaves that
    unspecified, and this is the stricter contract. Testing the bits rather than
    `x != x` keeps the whole thing in the integer domain and leaves the
    infinities where they belong.
    """
    bits = value.bitcast(Int32)
    bits = (bits == Int32(_INT32_MIN)).select(Int32(0), bits)
    ordered = (bits ^ ((bits >> Int32(31)) & Int32(0x7FFFFFFF))) ^ Int32(_INT32_MIN)
    is_nan = (bits & Int32(0x7FFFFFFF)) > Int32(_INF_BITS)
    return is_nan.select(Int32(_NAN_KEY), ordered)


def _wave_inclusive_prefix_i32(val, lane, wave_size):
    """Inclusive prefix sum across the wave: log depth, one swizzle per step."""
    distance = 1
    while distance < wave_size:
        remote = fly_rocdl.ds_bpermute(T.i32, (lane - Int32(distance)) * Int32(4), val)
        val = (lane >= Int32(distance)).select(val + Int32(remote), val)
        distance *= 2
    return val


@lru_cache(maxsize=64)
def topk_per_row_radix_stream_block_threads(rows: int, k: int) -> int:
    """The block width to build for a call of `rows` rows at this `k`.

    A build serves every row WIDTH -- the row length is a runtime value -- but
    not every row count or k, so the choice is the caller's. It lives here
    because the constants and the sweep behind them do.
    """
    if k >= _WIDE_BLOCK_MIN_K or rows <= _WIDE_BLOCK_MAX_ROWS:
        return _WIDE_BLOCK_THREADS
    return _BLOCK_THREADS


def _resolve_lds(
    k: int,
    block_threads: int,
    lds_budgets: tuple[int, int],
    vec: int,
    plan: tuple[int, int] | None = None,
):
    """The (unroll, soft_trigger) the LDS budget allows, or None if none does.

    Tiles are loaded in groups, all loads issued before any of the filtering, so
    a thread has `unroll` 128-bit reads in flight instead of one; with one tile
    per group a barrier follows every load and its latency is fully exposed. The
    arrivals region has to absorb a whole group, since the count is only checked
    between groups, so the group size is what LDS can pay for -- and it competes
    with the second resident workgroup for the same LDS.

    The budget is a preference, not a requirement: a large enough k cannot be
    held twice over on one CU at all, and one resident workgroup that runs beats
    a build that does not exist. Callers who pass a budget get it if it can be
    met and the solo ceiling if it cannot.

    `plan` is a preference one level up -- the pair a sweep found fastest for the
    caller's shape, taken only if it fits. One that could not be declined would
    be a table fitted on one card becoming a launch failure on a smaller one.

    Takes the budgets rather than reading them so it stays arithmetic: the
    sizing for a card this box does not have has to be testable without it.
    """
    tile = block_threads * vec

    def fits(unroll, soft, budget):
        cap = k + soft + unroll * tile
        return (cap + k) * 8 + _NUM_BUCKETS * 4 + 4096 <= budget

    budgets = dict.fromkeys(lds_budgets)
    if plan is not None and any(fits(*plan, budget) for budget in budgets):
        return plan
    for budget in budgets:
        for unroll in (_prefetch_tiles(k), 4, 2, 1):
            # Start at the floor, not at k: the arrivals region may hold more
            # than k candidates, and capping the trigger at k made every k below
            # `_SOFT_TRIGGER_MIN` unsatisfiable by construction -- reported, for
            # years, as an LDS shortage it never was.
            soft = min(max(k, _SOFT_TRIGGER_MIN), _SOFT_TRIGGER_MAX)
            while soft >= _SOFT_TRIGGER_MIN and not fits(unroll, soft, budget):
                soft //= 2
            if soft >= _SOFT_TRIGGER_MIN:
                return unroll, soft
    return None


def topk_per_row_radix_stream_lds_plan(
    rows: int, width: int, k: int
) -> tuple[int, int] | None:
    """The (unroll, soft_trigger) to prefer here, or None to let the budget decide.

    `_resolve_lds` starts from `_prefetch_tiles(k)` and takes the deepest
    prefetch the budget allows. That knows k and not the ROW COUNT, and the row
    count is what decides the depth wherever the kernel is mostly streaming: many
    rows keep the machine full and a deep queue only adds barriers, few rows need
    the queue to cover them. Supplying the missing term is the whole of this.

    Where it DECLINES is the other half. The tree behind it was fitted to
    minimise the distance to each cell's own best, which is not the same
    objective as not regressing: firing everywhere is a mean 1.126x as an A/B
    against the budget's own pick, with 11 of 100 cells below 0.95x. Both bounds
    below are ones the kernel already has rather than cuts fitted to the grid,
    and together they reach 1.118x with ONE.

      k >= 4096    the block width switches, so the tile doubles and a given
                   `unroll` stops meaning what it meant. Every regression in that
                   leaf was a k=4096 cell.
      wide rows    past `8 * k` columns the re-select fires rarely enough that
                   the budget's pick is already near the oracle; overriding it
                   buys 1.02x and six of the eleven regressions.

    A preference, not a decision: `_resolve_lds` declines a pair that does not
    fit, so a card with less LDS than this was fitted on falls back to its own
    search rather than failing to launch.
    """
    if k < 1024:
        return (4, 256) if rows < 1024 else (1, 512)
    if k < _WIDE_BLOCK_MIN_K and width <= 8 * k:
        return (1, 1024)
    return None


@lru_cache(maxsize=64)
def topk_per_row_radix_stream_serves(
    k: int,
    wave_size: int,
    block_threads: int = _BLOCK_THREADS,
    lds_budgets: tuple[int, int] | None = None,
    vec: int = _VEC,
) -> str | None:
    """Why this geometry cannot be built, or None if it can.

    What the build itself would hit, asked without building: a caller choosing
    between selectors needs the answer, not the module. The build shares this
    rather than restating the limits, so the two cannot drift.

    Asked without a plan, because a plan only ever narrows what is chosen from
    within what fits -- it can never make a buildable geometry unbuildable.
    """
    lds_budgets = lds_budgets or _lds_budgets()
    if wave_size not in (32, 64):
        return f"wave size must be 32 or 64, got {wave_size}"
    if k < 1:
        return f"k must be positive, got {k}"
    if block_threads % wave_size:
        return "block must be a whole number of waves"
    if _NUM_BUCKETS % block_threads:
        return (
            f"the bucket scan splits {_NUM_BUCKETS} buckets across the block, "
            f"so the block must divide it; got {block_threads}"
        )
    if _resolve_lds(k, block_threads, lds_budgets, vec) is None:
        return (
            f"k={k} with a {block_threads}-thread block needs more than "
            f"{lds_budgets[1]} bytes of LDS for its candidate buffer"
        )
    return None


@cache
def build_topk_per_row_radix_stream_module(
    k: int,
    wave_size: int,
    partial: bool = False,
    labelled: bool = False,
    block_threads: int = _BLOCK_THREADS,
    lds_budgets: tuple[int, int] | None = None,
    vec: int = _VEC,
    lds_plan: tuple[int, int] | None = None,
):
    """Compile the streaming selector for one k. The row width is a runtime value.

    LDS holds the candidates, not the row, so nothing here scales with N: the
    buffer is k survivors plus room for the arrivals between two selects, and
    the arrivals region must absorb a whole tile because a tile is filtered
    before the count is checked.

    `lds_budgets` is how much LDS one workgroup may claim -- the two-resident
    figure and the solo ceiling -- and so how many of them a CU holds at once.
    Spending all of it buys a longer prefetch group inside one workgroup;
    spending half buys a second resident workgroup whose loads cover this one's
    barriers. Which wins is a measurement, not a rule, and the measurement is
    `topk_per_row_radix_stream_lds_plan`, which callers pass as `lds_plan`. It
    does not appear in the cache key by accident: two shapes that prefer
    different pairs need different modules, and that is what the k, the width
    and the plan together are.

    `labelled` makes a candidate's identity a caller-supplied label rather than
    its position, which is what lets the merge half of a split selection order
    by the ORIGINAL column and so return the same answer as an unsplit one.
    A slice selector has positions that mean something and never needs it, so
    the two modes are exclusive.
    """
    if labelled and partial:
        raise ValueError(
            "[FlyDSL topk_per_row_radix_stream] `labelled` relabels the columns "
            "a block reports and `partial` renumbers them into the row's own "
            "coordinates; asking for both leaves the answer's columns undefined"
        )
    lds_budgets = lds_budgets or _lds_budgets()
    reason = topk_per_row_radix_stream_serves(
        k, wave_size, block_threads, lds_budgets, vec
    )
    if reason is not None:
        raise ValueError(f"[FlyDSL topk_per_row_radix_stream] {reason}")

    num_waves = block_threads // wave_size
    buckets_per_thread = _NUM_BUCKETS // block_threads
    tile = block_threads * vec
    unroll, soft_trigger = _resolve_lds(
        k, block_threads, lds_budgets, vec, plan=lds_plan
    )
    arrivals_cap = soft_trigger + unroll * tile
    capacity = k + arrivals_cap
    # Where the streaming loop starts, and it must be a whole number of vectors.
    # `absorb` takes the column of an element from its own base but the address
    # from `base // vec`, so an unaligned start reads one column and labels it
    # another -- the key and the column of every element after the window
    # disagree by `base % vec`. `arrivals_cap` is a multiple of `vec`, so this
    # was exactly the k whose `capacity` was not: k=1, 2, 3, 5, 6, 7 returned 56
    # to 64 wrong rows in 64 at 32768 columns, and k=4, 8, 12, 16 were clean.
    # Rounding down rather than up keeps it inside the buffer.
    window_cap = (capacity // vec) * vec

    @fx.struct
    class SharedStorage:
        cand_key: fx.Array[Int32, capacity, 16]
        cand_col: fx.Array[Int32, capacity, 16]
        keep_key: fx.Array[Int32, k, 16]
        keep_col: fx.Array[Int32, k, 16]
        hist: fx.Array[Int32, _NUM_BUCKETS, 16]
        scan: fx.Array[Int32, num_waves, 16]
        state: fx.Array[Int32, _ST_SLOTS, 16]

    @flyc.kernel(
        name="topk_per_row_radix_stream_"
        + kernel_signature(
            k=k,
            wave=wave_size,
            part=partial,
            lab=labelled,
            blk=block_threads,
            vec=vec,
            cap=capacity,
        ),
        known_block_size=[block_threads, 1, 1],
    )
    def topk_per_row_radix_stream_kernel(
        scores: fx.Tensor,
        row_lens: fx.Tensor,
        indices: fx.Tensor,
        part_val: fx.Tensor,
        col_label: fx.Tensor,
        num_parts: fx.Int32,
    ):
        block = fx.block_idx.x
        row = block // num_parts if partial else block
        part = block % num_parts if partial else Int32(0)
        tid = fx.thread_idx.x
        lane = tid % Int32(wave_size)
        wave = tid // Int32(wave_size)
        zero = Int32(0)
        one = Int32(1)
        top_k = Int32(k)

        storage = fx.SharedAllocator().allocate(SharedStorage)
        cand_key = storage.cand_key.peek().view(fx.make_layout(capacity, 1))
        cand_col = storage.cand_col.peek().view(fx.make_layout(capacity, 1))
        keep_key = storage.keep_key.peek().view(fx.make_layout(k, 1))
        keep_col = storage.keep_col.peek().view(fx.make_layout(k, 1))
        hist = storage.hist.peek().view(fx.make_layout(_NUM_BUCKETS, 1))
        scan = storage.scan.peek().view(fx.make_layout(num_waves, 1))
        state = storage.state.peek().view(fx.make_layout(_ST_SLOTS, 1))

        # In partial mode this block owns one slice of the row, and every column
        # it reports is still numbered in the row's own coordinates.
        full_len = row_lens[row]
        chunk = fx.ceildiv(fx.ceildiv(full_len, num_parts), Int32(vec)) * Int32(vec)
        col_base = chunk * part if partial else Int32(0)
        row_len = fx.min(chunk, full_len - col_base) if partial else full_len
        # Slice the row first, then build the descriptor over it. Built over the
        # whole tensor instead, `num_records` is a 32-bit BYTE count, so any
        # input past 4 GiB -- 1024 rows of a million fp32 -- wraps and the loads
        # come back silently wrong. One row is 4 MiB at the widest N here.
        score_full = fx.rocdl.make_buffer_tensor(
            fx.slice(scores, (row, None)), max_size=False
        )
        score_row = fx.logical_divide(score_full, fx.make_layout(vec, 1))
        vec_base = col_base // Int32(vec)
        row_indices = fx.slice(indices, (row, None))
        row_vals = fx.slice(part_val, (row, None))
        # In `labelled` mode a candidate's identity is not its position: the
        # caller supplies one label per column, laid out exactly like the scores,
        # and the answer is ordered by (value descending, LABEL ascending). The
        # merge half of a split selection needs this -- its row is the winners of
        # G slices, whose positions carry no meaning, while their original
        # columns do. Any unique labels work; nothing here assumes they ascend.
        label_row = (
            fx.logical_divide(
                fx.rocdl.make_buffer_tensor(
                    fx.slice(col_label, (row, None)), max_size=False
                ),
                fx.make_layout(vec, 1),
            )
            if labelled
            else None
        )

        def load_labels(vec_index):
            """The `vec` labels at one vector position, or their own columns."""
            src = fx.slice(label_row, (None, vec_index))
            fragment = fx.make_fragment_like(src)
            fx.copy(buf_copy_atom(vec * 4, Int32), src, fragment)
            return fx.Vector(fx.memref_load_vec(fragment))

        # Nested rather than module level: only code inside the kernel body is
        # AST-rewritten, and every one of these needs runtime `for` and `if`.
        # hist / scan / state are parameters, not captures, because the frontend
        # cannot yield a write to a closed-over name out of an `scf.if`.
        def pick_bucket(target, slot, hist, scan, state):
            # Each thread owns a contiguous run of buckets, so one wave prefix
            # over the run totals places every bucket without a second scan.
            first = tid * Int32(buckets_per_thread)
            counts = [
                hist[first + Int32(j)] for j in range_constexpr(buckets_per_thread)
            ]
            local = counts[0]
            for j in range_constexpr(buckets_per_thread - 1):
                local = local + counts[j + 1]
            inclusive = _wave_inclusive_prefix_i32(local, lane, wave_size)
            if lane == Int32(wave_size - 1):
                scan[wave] = inclusive
            # A thread's run has been read into `counts` and no other thread
            # touches it, so it can be zeroed here rather than at the top of the
            # next pass -- where it needed a barrier of its own. The histogram is
            # therefore clean on entry to and on exit from every pass, and only
            # the first pass of a row ever has to clear it.
            for j in range_constexpr(buckets_per_thread):
                hist[first + Int32(j)] = zero
            gpu.barrier()
            # Every wave folds the wave totals itself. Electing wave 0 to publish
            # a prefix is fewer instructions -- `num_waves` reads and adds per
            # thread against one log-depth swizzle in one wave -- but it costs a
            # second block barrier, and on this kernel the barriers are what the
            # fixed cost is made of: measured at rows=1024, width=32768, k=16,
            # selection is 14us of fixed cost over a stream running at the same
            # bandwidth as the selector it loses to, which has 7us.
            total = zero
            wave_prefix = zero
            for w in range_constexpr(num_waves):
                wave_total = scan[Int32(w)]
                total = total + wave_total
                wave_prefix = wave_prefix + (Int32(w) < wave).select(wave_total, zero)
            base = wave_prefix + (inclusive - local)
            # Buckets ascend, so the target-th largest sits where the prefix
            # count crosses total - target.
            want = total - target
            running = base
            for j in range_constexpr(buckets_per_thread):
                nxt = running + counts[j]
                if (running <= want) & (nxt > want):
                    state[slot] = first + Int32(j)
                    state[_ST_ABOVE] = total - nxt
                    state[_ST_BUCKET_CNT] = counts[j]
                running = nxt
            gpu.barrier()

        def radix_cut(count, target, key_of, hist, scan, state, allow_exit=True):
            """A separator with the target-th largest of `count` slots above it.

            The value returned is a KEY only when all three passes ran. The
            early exit below returns the pivot bucket's lower bound minus one
            instead -- the same separator for a strict test, but not a key any
            candidate need hold. A caller that inverts it back into something
            meaningful has to ask for `allow_exit=False`.

            `key_of` reads one candidate; the three passes each walk the buffer
            again rather than holding it in registers.

            Hoisting the buffer into registers -- two per slot, thirty-two at a
            half-width block -- is the obvious optimisation and it gains nothing
            measurable, which is reason enough not to.

            It was also long blamed for a fault of exactly this shape: the scan
            total short a few rows in ten thousand, the pivot then below the true
            k-th, and the placement quota discarding real winners at random,
            above a row count only. A fault answering that description was
            traced, by witnessing the waves of a workgroup read different
            arrivals counts, to the streaming loop's missing back-edge barrier
            -- see `absorb`. So register pressure is an UNPROVEN account of it,
            and the reason it survived a ballot, an ordered load, wait states,
            LDS padding on every side and four rewrites of this prefix is that
            each of those perturbs the schedule enough to hide a race. Nothing
            here may be explained by a change making the fault go away.
            """
            # `cut` accumulates the digits found so far, so after pass p it holds
            # exactly the top bits of the answer down to that pass's shift. That
            # makes the next pass's filter one comparison -- the candidate's own
            # top bits against it -- rather than one per digit already fixed.
            for i in range(tid, count, Int32(block_threads)):
                atomic_add_i32(
                    hist, one, key_of(i).shrui(Int32(_HIGH_SHIFT)), "workgroup"
                )
            gpu.barrier()
            pick_bucket(target, _ST_CUT_HI, hist, scan, state)
            cut_hi = state[_ST_CUT_HI]
            need_mid = target - state[_ST_ABOVE]

            for i in range(tid, count, Int32(block_threads)):
                key = key_of(i)
                if key.shrui(Int32(_HIGH_SHIFT)) == cut_hi:
                    atomic_add_i32(
                        hist,
                        one,
                        key.shrui(Int32(_MID_SHIFT)) & Int32(_NUM_BUCKETS - 1),
                        "workgroup",
                    )
            gpu.barrier()
            pick_bucket(need_mid, _ST_CUT_MID, hist, scan, state)
            cut_mid = state[_ST_CUT_MID]
            need_low = need_mid - state[_ST_ABOVE]
            mid_cnt = state[_ST_BUCKET_CNT]

            # When the pivot bucket holds exactly the candidates still wanted,
            # every one of them is in the answer and the last digit cannot
            # change which: the bucket's lower bound already separates `target`
            # candidates from the rest. `running == want` is the same statement,
            # which is why nothing here has to re-count.
            #
            # On continuous input 22 bits leave one candidate in the bucket and
            # one still wanted, so this is the usual case, and the third pass --
            # a walk of the buffer and a `pick_bucket`, three block barriers --
            # is skipped. 11 bits do not separate nearly as well, so the same
            # test after the FIRST pass almost never fires; measured, it is a
            # branch that only costs, and it is not here.
            #
            # Worth 1.19x at 16384x2048 k=1024, 1.04x at 4096x65536 k=256 and
            # 1.01x..1.02x elsewhere, against 0.99x at 1024x32768 k=512 -- the
            # saving scales with the buffer and the branch does not.
            cut = (
                (cut_hi << Int32(_HIGH_SHIFT)) | (cut_mid << Int32(_MID_SHIFT))
            ) - one
            n_eq = zero
            need = zero
            if (mid_cnt != need_low) if allow_exit else True:
                for i in range(tid, count, Int32(block_threads)):
                    key = key_of(i)
                    if (key.shrui(Int32(_HIGH_SHIFT)) == cut_hi) & (
                        key.shrui(Int32(_MID_SHIFT)) & Int32(_NUM_BUCKETS - 1)
                        == cut_mid
                    ):
                        atomic_add_i32(hist, one, key & Int32(_LOW_MASK), "workgroup")
                gpu.barrier()
                pick_bucket(need_low, _ST_CUT_LOW, hist, scan, state)
                cut = (
                    (cut_hi << Int32(_HIGH_SHIFT))
                    | (cut_mid << Int32(_MID_SHIFT))
                    | state[_ST_CUT_LOW]
                )
                # The last pass already knows both tie figures, so neither needs
                # a census of its own: the pivot bucket's population is the
                # number of candidates equal to the cut, and the rank left over
                # after the candidates strictly above it is how many of those
                # are wanted.
                n_eq = state[_ST_BUCKET_CNT]
                need = need_low - state[_ST_ABOVE]
            return cut, n_eq, need

        def compact(count, cand_key, cand_col, keep_key, keep_col, hist, scan, state):
            """Reduce cand[0:count] to its top k, and leave the cut in `state`.

            Winners go to a separate buffer before being copied back: writing
            them over the array the other threads are still reading is the one
            race this structure has, and a k-element copy is cheaper than the
            double buffering that would avoid it -- and far cheaper than holding
            the whole buffer in registers to dodge the reads.

            The answer is the k largest ordered by (key descending, column
            ascending). Only the second half of that order costs anything, and
            only when the cut value is shared by more candidates than there are
            places left: then a second select over ~column picks the `need`
            smallest. On continuous input the cut value occurs once and the
            branch is not taken.
            """
            # The placement counters belong to a stage that has not started, so
            # setting them here costs no barrier: the first radix pass ends in
            # one and nothing in between reads them. Set after the cut instead,
            # they need a barrier that exists only to publish three words.
            if tid == zero:
                state[_ST_KEPT] = zero
                state[_ST_TIED] = zero
                # No tie held yet, and a column is never negative.
                state[_ST_THR_COL] = Int32(-1)

            cut, n_eq, need = radix_cut(
                count, top_k, lambda i: cand_key[i], hist, scan, state
            )

            # `need == 0` means `radix_cut` took its early exit and `cut` is a
            # separator rather than a key. Some candidate BELOW the answer may
            # hold that exact value -- the bound has zeros in its low digits, so
            # the separator has ones, and about one key in 2^10 matches -- and
            # admitting it as a tie writes a k+1-th winner, whereupon the quota
            # discards a real one at random. Two rows in 16384 at 2048 columns,
            # fourteen at 65536: rare enough that only the soak found it.
            # Closing the tie branch on `need` is exact, where trusting no key
            # to equal the separator is not.
            #
            # Columns are unique, so the inner select has no ties of its own and
            # exactly `need` candidates clear it. Non-tied slots are given key 0,
            # far below any ~column, which shifts the running total and the
            # prefix at the answer's bucket by the same amount and so cannot
            # move the cut.
            cut_col = (need > zero).select(Int32(_INT32_MAX), Int32(-1))
            if n_eq > need:
                cut_col = (
                    Int32(-1)
                    - radix_cut(
                        count,
                        need,
                        lambda i: (cand_key[i] == cut).select(
                            Int32(-1) - cand_col[i], zero
                        ),
                        hist,
                        scan,
                        state,
                        # The result is inverted back into a column, so it has
                        # to be a key some candidate holds -- which a separator
                        # is not. The tie path is the rare one, so taking all
                        # three passes here costs nothing worth measuring.
                        allow_exit=False,
                    )[0]
                )

            # Strict winners fill from 0 up and ties fill from k-1 down, in one
            # pass. What forced two passes was the single counter, not the two
            # classes: a tie taking a slot ahead of a strict winner would push
            # that winner out, and the answer would hold the cut value in place
            # of something larger. Give the ties a counter of their own and the
            # classes cannot reach each other.
            #
            # They meet exactly, so neither range guard below can fire. When the
            # cut is a key, the S elements strictly above it number fewer than
            # k, at least k have key >= cut, so the n_eq at the cut satisfy
            # n_eq >= need = k - S, and `cut_col` admits exactly `need` of them:
            # S + need = k. When it is a separator, S is k by construction and
            # `need` is zero. The guards stay because a future change to the cut
            # would otherwise corrupt the answer in silence rather than loudly.
            #
            # One fewer walk of the buffer out of five, and one fewer block
            # barrier. Worth 1.016x..1.064x on random rows and 1.010x..1.043x on
            # descending ones, over six shapes from 512x131072 k=2048 to
            # 16384x32768 k=4096, against an `amax` control that moved under 1%
            # between the two builds.
            for i in range(tid, count, Int32(block_threads)):
                key = cand_key[i]
                col = cand_col[i]
                if _ugt(key, cut):
                    slot = atomic_add_i32(state, one, _ST_KEPT, "workgroup")
                    if slot < top_k:
                        keep_key[slot] = key
                        keep_col[slot] = col
                if (key == cut) & (col <= cut_col):
                    taken = atomic_add_i32(state, one, _ST_TIED, "workgroup")
                    slot = top_k - one - taken
                    if slot >= zero:
                        keep_key[slot] = key
                        keep_col[slot] = col
                        # The worst held element is the last tie by column, and
                        # a max is the same whatever order the lanes arrive in.
                        atomic_max_i32(state, col, _ST_THR_COL, "workgroup")
            gpu.barrier()
            for i in range(tid, top_k, Int32(block_threads)):
                cand_key[i] = keep_key[i]
                cand_col[i] = keep_col[i]
            if tid == zero:
                state[_ST_ARRIVED] = zero
                state[_ST_THR] = cut
            gpu.barrier()

        def absorb(base, cand_key, cand_col, state):
            """Filter one group of `unroll` tiles, re-selecting between them.

            Every load is issued before any of the filtering, so a thread has
            `unroll` reads in flight and their latencies overlap instead of each
            being exposed in turn. Past the end of the row the buffer descriptor
            bounds-check returns zero, which `live` then discards.

            Nothing crosses a re-select: the whole group is filtered, and only
            then is the arrivals count checked. Draining between tiles instead
            would keep the arrivals region one tile wide however deep the
            prefetch -- a real saving -- but a tile still waiting its turn does
            not reliably survive a compact, and no amount of pinning, reloading
            or scalarising made it. So the group is sized to what the arrivals
            region can absorb whole, and the prefetch depth is whatever that
            leaves.
            """
            keyed = []
            tagged = []
            for u in range_constexpr(unroll):
                pos = vec_base + (base + Int32(u * tile)) // Int32(vec) + tid
                src = fx.slice(score_row, (None, pos))
                fragment = fx.make_fragment_like(src)
                fx.copy(buf_copy_atom(vec * 4, Float32), src, fragment)
                loaded = fx.Vector(fx.memref_load_vec(fragment))
                keyed.append([_ord_unsigned(loaded[j]) for j in range_constexpr(vec)])
                if labelled:
                    tagged.append(load_labels(pos))
            # The loop's back edge carries no barrier, so without this one a
            # wave that has read the arrivals count runs into the next group and
            # adds to it while another has yet to look. That count gates
            # `compact`, and `compact` is barriers: one wave over the trigger and
            # another under it leaves the workgroup executing different numbers
            # of them. Measured at 161 rows in 338484 whose waves disagreed,
            # holding all 9 wrong ones; both go to zero with this. Placed after
            # the loads, whose latency covers it.
            gpu.barrier()
            thr = state[_ST_THR]
            thr_col = state[_ST_THR_COL]
            for u in range_constexpr(unroll):
                tile_base = base + Int32(u * tile)
                for j in range_constexpr(vec):
                    # The position decides whether the element is part of the
                    # row; the label decides what the answer calls it. They are
                    # the same number unless the caller supplied labels.
                    pos = tile_base + tid * Int32(vec) + Int32(j)
                    col = tagged[u][j] if labelled else pos
                    live = pos < row_len
                    key = keyed[u][j]
                    # The threshold is the pair (cut, worst held column), which
                    # is the order the answer is defined in. Testing only the
                    # key would drop an equal element that ought to displace a
                    # held one, and the answer would stop being canonical.
                    beats = _ugt(key, thr) | ((key == thr) & (col < thr_col))
                    if live & beats:
                        slot = atomic_add_i32(state, one, _ST_ARRIVED, "workgroup")
                        if slot < Int32(arrivals_cap):
                            cand_key[top_k + slot] = key
                            cand_col[top_k + slot] = col
            gpu.barrier()

        out_base = top_k * part if partial else Int32(0)

        if row_len <= top_k:
            # Every live column wins; nothing to select.
            for pad in range_constexpr((k + block_threads - 1) // block_threads):
                slot = tid + Int32(pad * block_threads)
                if slot < top_k:
                    live = slot < row_len
                    col = col_base + slot
                    reported = (
                        col_label[row, live.select(slot, zero)] if labelled else col
                    )
                    row_indices[out_base + slot] = live.select(reported, Int32(-1))
                    if partial:
                        # A dead slot must lose the merge, so it carries -inf.
                        row_vals[out_base + slot] = live.select(
                            scores[row, live.select(col, Int32(0))],
                            Float32(float("-inf")),
                        )

        if row_len > top_k:
            if tid < Int32(_ST_SLOTS):
                state[tid] = zero
            # The only histogram clear in the row: `pick_bucket` leaves it zero
            # behind every pass, so this rides the barrier the state init needs
            # anyway and no pass pays for one.
            for j in range_constexpr(buckets_per_thread):
                hist[tid * Int32(buckets_per_thread) + Int32(j)] = zero
            gpu.barrier()

            # The window that seeds the threshold: as much of the row as the
            # buffer holds, so a row that fits is selected once and never
            # streamed at all.
            window = fx.min(row_len, Int32(window_cap))
            window_vecs = fx.ceildiv(window, Int32(vec))
            for v_iv in range(tid, window_vecs, Int32(block_threads)):
                v = Int32(v_iv)
                src = fx.slice(score_row, (None, vec_base + v))
                fragment = fx.make_fragment_like(src)
                fx.copy(buf_copy_atom(vec * 4, Float32), src, fragment)
                loaded = fx.Vector(fx.memref_load_vec(fragment))
                # The labels share the scores' layout, so they ride the same
                # vector position rather than a gather.
                labels = load_labels(vec_base + v) if labelled else None
                for j in range_constexpr(vec):
                    col = v * Int32(vec) + Int32(j)
                    if col < window:
                        cand_key[col] = _ord_unsigned(loaded[j])
                        cand_col[col] = labels[j] if labelled else col
            gpu.barrier()

            compact(window, cand_key, cand_col, keep_key, keep_col, hist, scan, state)

            groups = fx.ceildiv(row_len - window, Int32(unroll * tile))
            for _t in range(zero, groups, one):
                absorb(
                    window + Int32(_t) * Int32(unroll * tile),
                    cand_key,
                    cand_col,
                    state,
                )
                arrived = state[_ST_ARRIVED]
                if arrived >= Int32(soft_trigger):
                    compact(
                        top_k + arrived,
                        cand_key,
                        cand_col,
                        keep_key,
                        keep_col,
                        hist,
                        scan,
                        state,
                    )

            arrived = state[_ST_ARRIVED]
            if arrived > zero:
                compact(
                    top_k + arrived,
                    cand_key,
                    cand_col,
                    keep_key,
                    keep_col,
                    hist,
                    scan,
                    state,
                )

            for i in range(tid, top_k, Int32(block_threads)):
                col = col_base + cand_col[i]
                row_indices[out_base + i] = col
                if partial:
                    # The merge selects on values, so re-read each winner; k
                    # gathered loads against a slice of N/G.
                    row_vals[out_base + i] = scores[row, col]

    @flyc.jit
    def launch_topk_per_row_radix_stream(
        scores: fx.Tensor,
        row_lens: fx.Tensor,
        indices: fx.Tensor,
        part_val: fx.Tensor,
        col_label: fx.Tensor,
        num_parts: fx.Int32,
        blocks: fx.Int32,
        stream: fx.Stream,
    ):
        topk_per_row_radix_stream_kernel(
            scores, row_lens, indices, part_val, col_label, num_parts
        ).launch(
            grid=(blocks, 1, 1),
            block=(block_threads, 1, 1),
            stream=stream,
        )

    # The shape the LDS budget resolved to. Derived here from several interacting
    # rules, so tests read it off the build rather than recomputing it and
    # risking a copy that drifts.
    launch_topk_per_row_radix_stream.topk_stream_config = {
        "block_threads": block_threads,
        "vec": vec,
        "unroll": unroll,
        "soft_trigger": soft_trigger,
        "capacity": capacity,
        "window_cap": window_cap,
        "arrivals_cap": arrivals_cap,
        "lds_bytes": (capacity + k) * 8 + _NUM_BUCKETS * 4,
        "labelled": labelled,
    }
    return launch_topk_per_row_radix_stream
