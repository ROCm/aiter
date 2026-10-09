# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Per-row TopK for a small k, by narrowing to k chunks instead of sorting.

The rows this serves are short and wide-ish: a block-sparse indexer scores one
value per KV block, so even a 1M-token context is 8192 scores, and k is 16.
Sorting every tile of such a row costs 13-20x a pass that only reads it.

The narrowing:

    Partition the row into C >= k chunks and rank the chunks by their maximum,
    ties broken by chunk id. Every element of the row's true top-k lives in one
    of the top-k chunks.

    Why: let W be the chunks holding a true top-k element, so |W| <= k. Each
    chunk in W has a maximum >= v_k, the row's k-th largest. Every chunk outside
    W holds only elements < v_k, so its maximum is strictly below every maximum
    in W -- no tie can straddle the boundary, and W is inside the top-k chunks
    under any consistent tie-break.

    The k-th largest chunk maximum is therefore also a threshold no winner can
    fall below, and applying both -- chosen chunk AND above the cut -- is what
    makes the survivor set small as well as bounded.

A chunk is one lane's share of the row, so C is the wave width and the whole
selection is a ballot: no barrier, and the chunk maxima never leave registers.
Lane-strided, not sliced -- a slice would leave a short row inside fewer than k
chunks and collapse the cut. Measured survivors at k=16 over 1024- and 8192-wide
rows: 21-28 (gaussian, position-biased, heavily tied), 61 for a strictly
ascending row, 18-20 once truncated to a few hundred live columns. The buffer is
sized by the hard bound, k * elements-per-chunk.

Ties resolve toward the larger column, matching the Triton selector this
replaces: equal scores are common (a block score is a max over 128 keys from a
low-mantissa cache) and that is the order attention accumulates in.
"""

from functools import cache, reduce

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import (
    Float32,
    Int32,
    Int64,
    arith,
    const_expr,
    gpu,
    range_constexpr,
)
from flydsl.expr import rocdl as fly_rocdl
from flydsl.expr.typing import T

from aiter.jit.utils.chip_info import get_gfx_runtime, get_lds_capacity_bytes
from aiter.ops.flydsl.kernels.kernels_common import (
    atomic_add_i32,
    kernel_signature,
    ord_signed_f32,
    uint32_to_int32,
)
from aiter.ops.flydsl.kernels.tensor_shim import buf_copy_atom

_VEC = 4
_BLOCK_THREADS = 512
# A wide row is held across more threads, so each keeps a smaller register tile
# and the workgroup carries more waves through the same LDS -- the survivor
# buffer does not shrink when the block widens, but it stops being what caps
# occupancy. Measured at 1024 rows, k=16, against the 512-thread build: 1.21x at
# 16384 columns (8 waves per SIMD to 16) and 1.05x at 32768 (4 to 8).
#
# It is a loss below that: at 8192 columns the LDS already admits eight
# workgroups, occupancy is at the hardware cap either way, and the wider block
# only adds waves to every barrier -- 9.2us becomes 11.6us. Hence a
# threshold rather than a higher cap.
_WIDE_BLOCK_THREADS = 1024
_WIDE_ROW_VECTORS = 4096
# 128-bit reads a thread keeps in flight before it uses any of them. The row read
# is what a wide row costs and it was latency-bound, not bandwidth-bound: with
# k=1 at 1024 rows and 32768 columns -- an 8 KiB buffer and an empty selection --
# the kernel still ran at 4.1 TB/s where the other two selectors reach 5.8 on the
# same row. Registers, not LDS, so the cost of a deeper group is register
# pressure against the tile already held.
_PREFETCH_VECS = 4
# Scratch slots shared by the whole block.
_ST_COUNT = 0
_ST_CUT = 1
_ST_MASK_LO = 2
_ST_MASK_HI = 3
_ST_SLOTS = 16


# The threshold search walks the key MSB first, which only terminates on an
# unsigned order, and `>` / `>=` on an fx integer are signed. Raw `arith.cmpi` is
# the only way to name the unsigned predicate.
def _uge(a, b):
    """`a >= b` on int32 bit patterns read as unsigned."""
    return arith.cmpi(arith.CmpIPredicate.uge, a, b)


def _ugt(a, b):
    """`a > b` on int32 bit patterns read as unsigned."""
    return arith.cmpi(arith.CmpIPredicate.ugt, a, b)


# Pins for the forced leading / trailing blocks: above every real score, and
# above each other. Applied to the value before the order-preserving map, so a
# forced block then sorts like any other winner.
_INIT_PIN = 1e30
_LOCAL_PIN = 1e29
_INT32_MIN = -2147483648


@cache
def lds_limit() -> int:
    """LDS one workgroup may claim on this device.

    Asked of the device, not written down. `160 * 1024` was gfx950's figure
    under a comment claiming it for all of gfx9, which gfx942 answers at 64 KiB
    -- so `serves` promised the router a k=16 selector over a 32768-wide row
    that then failed to load with `local memory (69696) exceeds limit (65536)`.

    `get_gfx_runtime`, not `get_gfx`: the latter honours `GPU_ARCHS`, while
    FlyDSL compiles for the live device either way, so the build target and this
    budget would answer for different cards.
    """
    return get_lds_capacity_bytes(get_gfx_runtime())


def topk_per_row_small_k_shape(
    k: int, n_max: int, wave_size: int, block_threads: int | None = None
):
    """Resolve the block width, survivor bound and LDS cost for a build.

    The survivor buffer is the whole LDS cost and it grows with the row bound,
    not with k alone: a k=16 selector over a 128K-wide row wants 258 KiB and the
    build fails inside the compiler, which leaves the HIP context unusable
    rather than raising. So the host predicate has to be able to ask this
    question before building, and it has to get the same answer the builder
    would -- hence one function, called from both.
    """
    vectors = (n_max + _VEC - 1) // _VEC
    if block_threads is None:
        cap = _WIDE_BLOCK_THREADS if vectors >= _WIDE_ROW_VECTORS else _BLOCK_THREADS
        waves = max(1, min(cap // wave_size, -(-vectors // wave_size)))
        block_threads = waves * wave_size
    survivors = k * ((vectors + wave_size - 1) // wave_size) * _VEC
    lds_bytes = 4 * (block_threads + 2 * survivors + _ST_SLOTS)
    return block_threads, survivors, lds_bytes


@cache
def build_topk_per_row_small_k_module(
    k: int,
    n_max: int,
    wave_size: int,
    block_threads: int | None = None,
    tie_low: bool = False,
    forced_blocks: bool = True,
    prefetch_vecs: int = _PREFETCH_VECS,
):
    """Compile the selector for one (k, row-width bound) pair.

    `n_max` fixes the per-thread register tile and the survivor buffer, so it is
    compile-time. A row shorter than `n_max` costs less time but the same
    registers -- bucket `n_max` rather than passing an exact width per shape.

    `prefetch_vecs` is how many of a thread's 128-bit reads are issued before
    any of them is used; see `_PREFETCH_VECS`.

    `forced_blocks` compiles in the leading/trailing pins: six instructions per
    element, on every element, whether or not a caller uses them. The inner loop
    is the whole cost at a wide row -- its marginal rate is 3.4 TB/s against the
    5.7-5.8 the other two selectors reach.

    `block_threads` defaults to the narrowest block that covers the row in one
    round of 128-bit loads, capped at `_BLOCK_THREADS` -- or `_WIDE_BLOCK_THREADS`
    once the survivor buffer is what caps occupancy. A wide block over a short
    row buys nothing and pays the cross-wave barriers.
    """
    if wave_size != 64:
        # The chunk selection is a single i64 ballot over the lanes, so a
        # wave32 target would need a different mask width throughout.
        raise ValueError(f"wave size must be 64, got {wave_size}")
    if k < 1 or k > wave_size:
        raise ValueError(f"k must be in [1, {wave_size}] (one chunk per lane), got {k}")
    if n_max < k:
        raise ValueError(f"row bound {n_max} is below k={k}")

    vectors = (n_max + _VEC - 1) // _VEC
    block_threads, survivors, lds_bytes = topk_per_row_small_k_shape(
        k, n_max, wave_size, block_threads
    )
    if block_threads % wave_size:
        raise ValueError("block must be a whole number of waves")
    if lds_bytes > lds_limit():
        # Caught here rather than in the compiler: an over-budget build fails
        # the module load and leaves the HIP context in an error state, taking
        # the process with it instead of raising.
        raise ValueError(
            f"k={k} over a row bound of {n_max} needs {lds_bytes} bytes of LDS "
            f"for its survivor buffer, over the {lds_limit()} limit"
        )
    # Whole vectors per thread, so every load stays 128-bit.
    vec_per_thread = (vectors + block_threads - 1) // block_threads
    elems_per_thread = vec_per_thread * _VEC
    waves_per_block = block_threads // wave_size

    @fx.struct
    class SharedStorage:
        chunk_max: fx.Array[Int32, block_threads, 16]
        surv_ord: fx.Array[Int32, survivors, 16]
        surv_col: fx.Array[Int32, survivors, 16]
        state: fx.Array[Int32, _ST_SLOTS, 16]

    @flyc.kernel(
        name="topk_per_row_small_k_"
        + kernel_signature(
            k=k,
            n=n_max,
            wave=wave_size,
            blk=block_threads,
            tlow=tie_low,
            pins=forced_blocks,
            pf=min(prefetch_vecs, vec_per_thread),
        ),
        known_block_size=[block_threads, 1, 1],
    )
    def topk_per_row_small_k_kernel(
        scores: fx.Tensor,
        row_lens: fx.Tensor,
        indices: fx.Tensor,
        init_blocks: fx.Int32,
        local_blocks: fx.Int32,
    ):
        row = fx.block_idx.x
        tid = fx.thread_idx.x
        lane = tid % Int32(wave_size)
        wave = tid // Int32(wave_size)
        zero = Int32(0)
        one = Int32(1)
        top_k = Int32(k)
        neg_inf = Int32(_INT32_MIN)

        storage = fx.SharedAllocator().allocate(SharedStorage)
        chunk_max = storage.chunk_max.peek().view(fx.make_layout(block_threads, 1))
        surv_ord = storage.surv_ord.peek().view(fx.make_layout(survivors, 1))
        surv_col = storage.surv_col.peek().view(fx.make_layout(survivors, 1))
        state = storage.state.peek().view(fx.make_layout(_ST_SLOTS, 1))

        row_len = row_lens[row]
        local_start = fx.max(zero, row_len - local_blocks) if forced_blocks else zero
        # Slice the row first, then build the descriptor over it. Built over the
        # whole tensor and sliced afterwards, `num_records` is a 32-bit BYTE
        # count: at 4 GiB it wraps to zero and every load returns zero. Measured
        # at 32768 rows of 32768 columns -- exactly 4 GiB, and reachable inside
        # this selector's own limits -- all 32768 rows came back wrong, with
        # nothing raised. One row is 4 MiB at the widest bound it will build.
        score_row = fx.logical_divide(
            fx.rocdl.make_buffer_tensor(fx.slice(scores, (row, None)), max_size=False),
            fx.make_layout(_VEC, 1),
        )
        row_indices = fx.slice(indices, (row, None))

        # --- 1. The row into registers, and this thread's partial maximum. ---
        # Thread t takes vectors t, t + block_threads, ...: consecutive lanes
        # read consecutive 128-bit words, and every vector a thread touches is
        # congruent to its lane -- which is what makes a chunk a lane below.
        # Issued in groups so a thread holds `group` reads in flight rather than
        # exposing each latency in turn. Latency-bound, not bandwidth-bound: at
        # k=1 over 32768 columns this loop alone runs 4.1 TB/s against 5.8.
        ords = fx.make_rmem_tensor(elems_per_thread, Int32)
        # Thread t's slot (v, j) is column `tid * _VEC + (v * block_threads *
        # _VEC + j)`, and the bracket is a compile-time constant -- so the
        # columns are one add from `col_base` and need no register tile of their
        # own. That halves the tile, on a kernel whose row read is latency-bound
        # and therefore occupancy-bound: 64 registers of tile at the widest bound
        # become 32.
        col_base = tid * Int32(_VEC)

        def col_of(v, j):
            return col_base + Int32(v * block_threads * _VEC + j)

        # One accumulator per vector lane, so the running maximum chains once per
        # vector rather than once per element -- 32 deep at the widest bound
        # otherwise. Splitting costs the fold at the end, `_VEC - 1` extra
        # maxima, which is only earned back when a thread holds more than one
        # vector: at exactly one it shortens nothing and measured 7% slower.
        n_acc = _VEC if vec_per_thread > 1 else 1
        lane_max = [neg_inf] * n_acc
        group = min(prefetch_vecs, vec_per_thread)
        for base in range_constexpr((vec_per_thread + group - 1) // group):
            width = min(group, vec_per_thread - base * group)
            fragments = []
            for u in range_constexpr(width):
                # No bounds test on the address: the descriptor is built over
                # this row alone, so its `num_records` already clamps a read past
                # the end to zero, and the liveness test below discards it.
                src = fx.slice(
                    score_row, (None, tid + Int32((base * group + u) * block_threads))
                )
                fragment = fx.make_fragment_like(src)
                fx.copy(buf_copy_atom(16, Float32), src, fragment)
                fragments.append(fragment)
            for u in range_constexpr(width):
                v = base * group + u
                loaded = fx.Vector(fx.memref_load_vec(fragments[u]))
                for j in range_constexpr(_VEC):
                    col = col_of(v, j)
                    # `col < row_len` alone: a column past this thread's vectors
                    # is at least `n_max`, which already bounds `row_len`.
                    live = col < row_len
                    value = loaded[j]
                    if const_expr(forced_blocks):
                        value = (live & (col < init_blocks)).select(
                            Float32(_INIT_PIN), value
                        )
                        value = (live & (col >= local_start)).select(
                            Float32(_LOCAL_PIN), value
                        )
                    slot = v * _VEC + j
                    # A padding slot must lose to every real element, so it
                    # takes the bottom of the key space. A real NaN now takes the
                    # top, so the two no longer collide.
                    ords[slot] = live.select(ord_signed_f32(value), neg_inf)
                    lane_max[j % n_acc] = fx.max(lane_max[j % n_acc], ords[slot])
        my_max = reduce(fx.max, lane_max)

        chunk_max[tid] = my_max
        if tid == zero:
            state[_ST_COUNT] = zero
        gpu.barrier()

        # --- 2. One wave picks the k chunks and the threshold. ---------------
        # Folding the block's maxima onto one wave puts the whole selection in
        # lane registers: the count at each step is a ballot, so the search runs
        # without a barrier and without touching LDS again.
        if wave == zero:
            # A chunk is a lane, not a slice of the row: thread (w, l) reads
            # vectors congruent to l, so every chunk gets a share of any prefix.
            # Slicing instead would leave a short row inside the first few
            # chunks, fewer than k of them, and the cut would collapse to -inf
            # and pass nothing through -- rows of a few hundred columns are the
            # common case, not the corner.
            super_max = neg_inf
            for s in range_constexpr(waves_per_block):
                super_max = fx.max(super_max, chunk_max[Int32(s * wave_size) + lane])
            # Unsigned-ordered, so the search can build the key MSB first.
            key = super_max ^ neg_inf
            cut = zero
            for bit in range_constexpr(32):
                probe = cut | Int32(uint32_to_int32(1 << (31 - bit)))
                hits = fly_rocdl.ballot(T.i64, _uge(key, probe))
                cut = (Int32(fx.math.ctpop(hits)) >= top_k).select(probe, cut)
            # `cut` is the k-th largest chunk maximum. Chunks above it are in;
            # chunks equal to it are taken lowest-id first, so exactly k chunks
            # go through and the survivor count has a compile-time bound.
            above = _ugt(key, cut)
            tied = key == cut
            tie_rank = Int32(
                fx.math.ctpop(
                    fly_rocdl.ballot(T.i64, tied)
                    & ((Int64(1) << Int64(lane)) - Int64(1))
                )
            )
            quota = top_k - Int32(fx.math.ctpop(fly_rocdl.ballot(T.i64, above)))
            chosen = fly_rocdl.ballot(T.i64, above | (tied & (tie_rank < quota)))
            if lane == zero:
                state[_ST_CUT] = cut
                state[_ST_MASK_LO] = Int32(chosen)
                state[_ST_MASK_HI] = Int32(chosen >> Int64(32))
        gpu.barrier()

        # --- 3. The chosen chunks contribute their elements above the cut. ---
        # Both tests are needed: the chunk mask bounds the buffer, the cut keeps
        # the count near k so the ranking below stays short.
        cut = state[_ST_CUT]
        mask = (Int64(state[_ST_MASK_HI]) << Int64(32)) | (
            Int64(state[_ST_MASK_LO]) & Int64(0xFFFFFFFF)
        )
        in_chosen = ((mask >> Int64(lane)) & Int64(1)) == Int64(1)
        for slot in range_constexpr(elems_per_thread):
            col = col_of(slot // _VEC, slot % _VEC)
            if in_chosen & _uge(ords[slot] ^ neg_inf, cut) & (col < row_len):
                at = atomic_add_i32(state, one, _ST_COUNT, "workgroup")
                surv_ord[at] = ords[slot]
                surv_col[at] = col
        gpu.barrier()

        # --- 4. Rank the survivors; the rank is the output slot. -------------
        # Quadratic in the survivor count only, which the cut holds near k, and
        # ranking rather than sorting lands the result ordered.
        #
        # Strided, not one survivor per thread: there can be more of them than
        # threads. `if tid < found` left the tail unranked, and an unranked
        # winner is never written -- its slot keeps whatever the caller's buffer
        # held. That happens once k reaches the wave width and the cut stops
        # narrowing: at k=64, 300 survivors on average against 18 at k=16, worst
        # 885, over the block width on 2% of rows. At k <= 32 the worst is 68.
        found = state[_ST_COUNT]
        for mine in range(tid, found, Int32(block_threads)):
            my_ord = surv_ord[mine]
            my_col = surv_col[mine]
            place = zero
            for other in range(zero, found, one):
                peer_ord = surv_ord[other]
                peer_col = surv_col[other]
                # Which way equal scores break among the survivors. Larger
                # column first is this selector's native order, matching the
                # Triton selector it replaces.
                #
                # `tie_low` flips this ranking only, and that is NOT the same as
                # promising the smallest column: once more chunks tie at the cut
                # than there are places, step 2 has already dropped chunks by
                # lane id, and lane is `(col // 4) % wave`, which is not monotone
                # in the column. Measured on 64 tied chunks at k=16, this build
                # returns columns [208, 256, 260, ...] against a canonical [128,
                # 132, 136, ...]. Both orders are deterministic and independent
                # of the row's place in the batch; neither is canonical.
                col_ahead = (peer_col < my_col) if tie_low else (peer_col > my_col)
                ahead = (peer_ord > my_ord) | ((peer_ord == my_ord) & col_ahead)
                place = place + ahead.select(one, zero)
            if place < top_k:
                row_indices[place] = my_col

        # Rows with fewer than k real elements leave the tail unwritten.
        real = fx.min(top_k, row_len)
        if (tid >= real) & (tid < top_k):
            row_indices[tid] = Int32(-1)

    @flyc.jit
    def launch_topk_per_row_small_k(
        scores: fx.Tensor,
        row_lens: fx.Tensor,
        indices: fx.Tensor,
        init_blocks: fx.Int32,
        local_blocks: fx.Int32,
        rows: fx.Int32,
        stream: fx.Stream,
    ):
        topk_per_row_small_k_kernel(
            scores, row_lens, indices, init_blocks, local_blocks
        ).launch(
            grid=(rows, 1, 1),
            block=(block_threads, 1, 1),
            stream=stream,
        )

    return launch_topk_per_row_small_k
