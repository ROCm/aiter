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
slots come from a shared counter. Only the set is canonical.

A row is one workgroup, which is the right shape only while there are enough
rows to fill the machine. Below that the `partial` mode splits one row across G
workgroups, each selecting its own slice; the survivors are then re-selected by
the same kernel in its ordinary mode, over a [rows, G*k] array of values. G
balances the two halves -- the slices run in parallel, the merge does not -- so
N/G = G*k, i.e. G = sqrt(N/k), which lands at 4..64 over the shapes here and is
the same order as the cluster size DeepSelect fixes at 16.

`partial` writes the winning values rather than their keys so that the merge is
an ordinary fp32 selection; it re-reads each winner from the row to get that
value, which is k gathered loads against a slice of N/G.
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

from aiter.ops.flydsl.kernels.kernels_common import (
    atomic_add_i32,
    atomic_max_i32,
    kernel_signature,
)
from aiter.ops.flydsl.kernels.tensor_shim import buf_copy_atom

_VEC = 4
# Half the CU's threads, so two or three workgroups stay resident and one
# streams tiles through another's barriers. Worth 1.3x..1.5x over a full-width
# block across N=64K..1M and k=512/2048.
#
# Read `radix_cut` before changing this: the width interacts with register
# pressure, and this kernel has been silently wrong once for that reason. Soak
# any change with `topk_stream_soak.py` -- the fault it caught hit a few rows in
# ten thousand, and only above a row count, so a smoke test passes on a broken
# build.
_BLOCK_THREADS = 512
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
# k=1 and k=2 return wrong rows, and not a few: at 64 rows of 32768 gaussian
# columns, 55 and 63 of them. Only once the row is wide enough to re-select --
# one tile group is right at every k -- and k>=4 is clean at every width tried.
# Declined rather than shipped while the cause is open; the dispatcher has three
# other selectors for a k this small, and `topk_per_row_argmax` owns k=1.
_MIN_SERVED_K = 3


# Loads a thread keeps in flight per group. Registers only, since arrivals are
# drained between tiles, so this is not a claim on LDS. A large k re-selects
# often and from a bigger buffer, so it has more latency to cover and pays for a
# deeper queue; a small k is nearly all streaming and a deep queue only adds
# barriers. Measured at N=64K..1M: depth 8 is worth 1.1x..1.3x at k=2048 and
# loses 1.2x at k=512, where depth 2 is best.
def _prefetch_tiles(k):
    return 8 if k >= 1024 else 2


# Two workgroups inside the 160 KiB of a gfx950 CU, with room left for the
# compiler's own LDS use; `_LDS_MAX` is the one-workgroup ceiling to fall back on
# when a k is too large to be held twice.
_LDS_BUDGET = 76 * 1024
_LDS_MAX = 148 * 1024

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
def _resolve_lds(k: int, block_threads: int, lds_budget: int, vec: int):
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
    met and the hardware ceiling if it cannot.
    """
    tile = block_threads * vec

    def fits(unroll, soft, budget):
        cap = k + soft + unroll * tile
        return (cap + k) * 8 + _NUM_BUCKETS * 4 + 4096 <= budget

    for budget in dict.fromkeys((lds_budget, _LDS_MAX)):
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


@lru_cache(maxsize=64)
def topk_per_row_radix_stream_serves(
    k: int,
    wave_size: int,
    block_threads: int = _BLOCK_THREADS,
    lds_budget: int = _LDS_BUDGET,
    vec: int = _VEC,
) -> str | None:
    """Why this geometry cannot be built, or None if it can.

    What the build itself would hit, asked without building: a caller choosing
    between selectors needs the answer, not the module. The build shares this
    rather than restating the limits, so the two cannot drift.
    """
    if wave_size not in (32, 64):
        return f"wave size must be 32 or 64, got {wave_size}"
    if k < _MIN_SERVED_K:
        return f"k must be at least {_MIN_SERVED_K}, got {k}"
    if block_threads % wave_size:
        return "block must be a whole number of waves"
    if _NUM_BUCKETS % block_threads:
        return (
            f"the bucket scan splits {_NUM_BUCKETS} buckets across the block, "
            f"so the block must divide it; got {block_threads}"
        )
    if _resolve_lds(k, block_threads, lds_budget, vec) is None:
        return (
            f"k={k} with a {block_threads}-thread block needs more than "
            f"{_LDS_MAX} bytes of LDS for its candidate buffer"
        )
    return None


@cache
def build_topk_per_row_radix_stream_module(
    k: int,
    wave_size: int,
    partial: bool = False,
    block_threads: int = _BLOCK_THREADS,
    lds_budget: int = _LDS_BUDGET,
    vec: int = _VEC,
):
    """Compile the streaming selector for one k. The row width is a runtime value.

    LDS holds the candidates, not the row, so nothing here scales with N: the
    buffer is k survivors plus room for the arrivals between two selects, and
    the arrivals region must absorb a whole tile because a tile is filtered
    before the count is checked.

    `lds_budget` sets how much LDS one workgroup may claim, and so how many of
    them a CU holds at once. Spending all of it buys a longer prefetch group
    inside one workgroup; spending half buys a second resident workgroup whose
    loads cover this one's barriers. Which wins is a measurement, not a rule.
    """
    reason = topk_per_row_radix_stream_serves(
        k, wave_size, block_threads, lds_budget, vec
    )
    if reason is not None:
        raise ValueError(f"[FlyDSL topk_per_row_radix_stream] {reason}")

    num_waves = block_threads // wave_size
    buckets_per_thread = _NUM_BUCKETS // block_threads
    tile = block_threads * vec
    unroll, soft_trigger = _resolve_lds(k, block_threads, lds_budget, vec)
    arrivals_cap = soft_trigger + unroll * tile
    capacity = k + arrivals_cap

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

        def radix_cut(count, target, key_of, hist, scan, state):
            """Key of the target-th largest of the block's first `count` slots.

            `key_of` reads one candidate; the three passes each walk the buffer
            again rather than holding it in registers.

            Hoisting the buffer into registers -- two per slot, thirty-two at a
            half-width block -- is the obvious optimisation and it is a trap. It
            gains nothing measurable, and it made this scan total short a few
            rows in ten thousand: the pivot then falls below the true k-th and
            the placement quota discards real winners at random. It only showed
            up above a row count, so a smoke test of a handful of rows passed
            throughout, and any added instruction moved the register allocation
            enough to hide it -- it survived a ballot, an ordered load, wait
            states, LDS padding on every side, four rewrites of this prefix and
            both cross-lane primitives before the pressure itself turned out to
            be the cause.
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

            for i in range(tid, count, Int32(block_threads)):
                key = key_of(i)
                if (key.shrui(Int32(_HIGH_SHIFT)) == cut_hi) & (
                    key.shrui(Int32(_MID_SHIFT)) & Int32(_NUM_BUCKETS - 1) == cut_mid
                ):
                    atomic_add_i32(hist, one, key & Int32(_LOW_MASK), "workgroup")
            gpu.barrier()
            pick_bucket(need_low, _ST_CUT_LOW, hist, scan, state)
            cut = (
                (cut_hi << Int32(_HIGH_SHIFT))
                | (state[_ST_CUT_MID] << Int32(_MID_SHIFT))
                | state[_ST_CUT_LOW]
            )
            # The last pass already knows both tie figures, so neither needs a
            # census of its own: the pivot bucket's population is the number of
            # candidates equal to the cut, and the rank left over after the
            # candidates strictly above it is how many of those are wanted.
            return cut, state[_ST_BUCKET_CNT], need_low - state[_ST_ABOVE]

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
            cut, n_eq, need = radix_cut(
                count, top_k, lambda i: cand_key[i], hist, scan, state
            )

            # Columns are unique, so the inner select has no ties of its own and
            # exactly `need` candidates clear it. Non-tied slots are given key 0,
            # far below any ~column, which shifts the running total and the
            # prefix at the answer's bucket by the same amount and so cannot
            # move the cut.
            cut_col = Int32(_INT32_MAX)
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
                    )[0]
                )
            if tid == zero:
                state[_ST_KEPT] = zero
                # No tie held yet, and a column is never negative.
                state[_ST_THR_COL] = Int32(-1)
            gpu.barrier()

            # Strict winners are fewer than k by construction, so the shared
            # counter doubles as the tie quota. The two passes cannot be merged
            # or overlapped: a tie taking a slot ahead of a strict winner would
            # push that winner out, and the answer would hold the cut value in
            # place of something larger.
            for i in range(tid, count, Int32(block_threads)):
                if _ugt(cand_key[i], cut):
                    slot = atomic_add_i32(state, one, _ST_KEPT, "workgroup")
                    if slot < top_k:
                        keep_key[slot] = cand_key[i]
                        keep_col[slot] = cand_col[i]
            gpu.barrier()
            for i in range(tid, count, Int32(block_threads)):
                if (cand_key[i] == cut) & (cand_col[i] <= cut_col):
                    slot = atomic_add_i32(state, one, _ST_KEPT, "workgroup")
                    if slot < top_k:
                        keep_key[slot] = cand_key[i]
                        keep_col[slot] = cand_col[i]
                        # The worst held element is the last tie by column, and
                        # a max is the same whatever order the lanes arrive in.
                        atomic_max_i32(state, cand_col[i], _ST_THR_COL, "workgroup")
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
            for u in range_constexpr(unroll):
                src = fx.slice(
                    score_row,
                    (None, vec_base + (base + Int32(u * tile)) // Int32(vec) + tid),
                )
                fragment = fx.make_fragment_like(src)
                fx.copy(buf_copy_atom(vec * 4, Float32), src, fragment)
                loaded = fx.Vector(fx.memref_load_vec(fragment))
                keyed.append([_ord_unsigned(loaded[j]) for j in range_constexpr(vec)])
            thr = state[_ST_THR]
            thr_col = state[_ST_THR_COL]
            for u in range_constexpr(unroll):
                tile_base = base + Int32(u * tile)
                for j in range_constexpr(vec):
                    col = tile_base + tid * Int32(vec) + Int32(j)
                    live = col < row_len
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
                    row_indices[out_base + slot] = live.select(col, Int32(-1))
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
            window = fx.min(row_len, Int32(capacity))
            window_vecs = fx.ceildiv(window, Int32(vec))
            for v_iv in range(tid, window_vecs, Int32(block_threads)):
                v = Int32(v_iv)
                src = fx.slice(score_row, (None, vec_base + v))
                fragment = fx.make_fragment_like(src)
                fx.copy(buf_copy_atom(vec * 4, Float32), src, fragment)
                loaded = fx.Vector(fx.memref_load_vec(fragment))
                for j in range_constexpr(vec):
                    col = v * Int32(vec) + Int32(j)
                    if col < window:
                        cand_key[col] = _ord_unsigned(loaded[j])
                        cand_col[col] = col
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
        num_parts: fx.Int32,
        blocks: fx.Int32,
        stream: fx.Stream,
    ):
        topk_per_row_radix_stream_kernel(
            scores, row_lens, indices, part_val, num_parts
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
        "arrivals_cap": arrivals_cap,
        "lds_bytes": (capacity + k) * 8 + _NUM_BUCKETS * 4,
    }
    return launch_topk_per_row_radix_stream
