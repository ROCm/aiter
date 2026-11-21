# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.

# NOTE: no ``from __future__ import annotations`` here.  ``fx.struct`` resolves
# the LDS field annotations eagerly, and stringified annotations break its
# layout computation.

"""Single-kernel MoonEP planner for gfx950.

``moonep_planning`` runs the plan as three launches and is the validated
baseline.  This module is an independent, latency-tuned implementation that puts
everything in **one** launch.  Both are checked against the same golden
(``build_reference_plan``) by ``op_tests/test_moonep_planning_gpu.py``.

Why one kernel is worth it
--------------------------
Measured on the three-launch version (MI355X, S=8192 E=384 K=8 R=8 B=48):

===================  ========  ==============================================
stage                     us   note
===================  ========  ==============================================
order_hist              21.7   grid-parallel
meta                    85.5   ONE workgroup, so nothing hides its latency
dst                     10.9   only S threads of work
sum of stages          118.3
measured total         112.2   launch overhead is below the noise floor
===================  ========  ==============================================

So the launches themselves cost nothing; the win from fusing is that ``meta``
(one workgroup, ~74 us of pure dependent-latency chains) and ``hist``
(grid-parallel) are *independent* -- ``meta`` reads only ``tokens_per_expert``
and ``hist`` reads only ``topk_experts`` -- yet the three-launch form runs them
back to back.  Block 0 runs ``meta`` here while every other block runs ``hist``,
hiding the histogram entirely.

The ``meta`` floor itself was attributed with a ``prefetch_slots`` sweep
(meta grew 0.568 us per slot, so top-B alone was ~27 us at B=48) and a
``num_vblocks`` sweep (the vblock prefix was only ~12 us at NV=128, leaving
~32 us for the greedy).  This module therefore also changes three algorithms:

* ``hist`` resolves each entry's rank with a software ``match_any`` (one ballot
  per expert-id bit) instead of one broadcast per lane: 9 cross-lane ops per
  64 entries instead of 64.  This is the primitive upstream MoonEP gets for free
  from ``match.any.sync``.
* ``meta``'s top-B prefetch selection ranks every candidate against the whole
  set in one parallel pass instead of running B dependent argmax passes.  The
  ordering is identical because ``sort(key=(alloc, e), reverse=True)`` is a
  total order, so "selected at slot j" == "exactly j candidates outrank it".
* ``meta``'s balancing greedy keeps its worst-case trip count but guards the
  body with an SGPR flag, so the wave branches over the (many) rounds that have
  no work left instead of predicating through them.

``dst`` also switches from one thread per token to one lane per routed entry
(K entries land in one power-of-two lane group, so the duplicate scan stays
in-wave), which raises its parallelism by ``K``.

Correctness contract is unchanged: every tie-break still matches
``build_reference_plan`` bit for bit.  See ``moonep_planning`` for the shared
device helpers and the FlyDSL gotchas.

Residency requirement
---------------------
The two phase separators are software grid barriers, so every block must be
co-resident: the host caps ``blocks`` at the device CU count, and this kernel
must not be run concurrently with other work on the same device.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir as _ir
from flydsl._mlir.dialects import llvm
from flydsl.expr import gpu, ptrtoint, range_constexpr
from flydsl.expr import rocdl as fly_rocdl
from flydsl.expr.typing import T

from aiter.ops.flydsl.kernels import buffer_ops
from aiter.ops.flydsl.kernels.communication_ops_utils import (
    atomic_add_global_at,
    fence_system_acquire,
)
from aiter.ops.flydsl.kernels.moonep_planning import (
    _INT_MAX,
    _INT_MIN,
    _KEY_NOT_CANDIDATE,
    _KEY_SELECTED,
    WAVE_SIZE,
    _addr_rsrc,
    _ceil_div,
    _false,
    _lds_atomic_add,
    _lds_load,
    _lds_store,
    _unwrap,
    _wave_argmax,
    _wave_inclusive_prefix_sum,
    _wave_sum,
    buffer_load_i32,
)
from aiter.ops.flydsl.kernels.tensor_shim import (
    AITER_FLYDSL_KERNARG_PRELOAD,
    AITER_FLYDSL_KERNARG_PRELOAD_COUNT,
)


def _next_pow2(n):
    p = 1
    while p < n:
        p *= 2
    return p


def _uniform_i32(v):
    """Force a wave-uniform value into an SGPR.

    Guarding a loop body with an SGPR predicate lets the wave skip it with a
    scalar branch, which is what makes the greedy's early exit actually save
    instructions rather than just predicate them away.
    """
    return fx.Int32(fly_rocdl.readfirstlane(T.i32, v))


def _popcount64(v):
    return fx.Uint64(llvm.call_intrinsic(T.i64, "llvm.ctpop.i64", [_unwrap(v)], [], []))


def _wave_match_any(value, value_bits):
    """Lane mask of the wavefront lanes carrying the same ``value``.

    Software equivalent of NVIDIA's ``match.any.sync``: one ballot per bit of the
    value narrows the candidate mask, so the cost is ``log2(range)`` cross-lane
    ops instead of one broadcast per lane.  Every lane must be active, so callers
    clamp their inputs rather than branch.

    The complement of a bit's ballot comes from ballotting the negated predicate
    rather than XOR-ing an all-ones constant: FlyDSL folds constant arithmetic at
    trace time, and a 64-bit all-ones value does not survive ``IntegerAttr.get``
    on a signless i64 (``std::bad_cast``).  Every lane is active here, so
    ``ballot(~p)`` is exactly ``~ballot(p)``.
    """

    mask = None
    for b in range_constexpr(value_bits):
        is_set = ((value >> fx.Int32(b)) & fx.Int32(1)) == fx.Int32(1)
        ones = fx.Uint64(fly_rocdl.ballot(T.i64, is_set))
        zeros = fx.Uint64(fly_rocdl.ballot(T.i64, ~is_set))
        selected = is_set.select(ones, zeros)
        mask = selected if mask is None else (mask & selected)
    return mask


def _lanemask_lt(lane):
    return (fx.Uint64(1) << fx.Uint64(lane)) - fx.Uint64(1)


def _wave_read_lane(val, lane_idx):
    """Read ``val`` from the wave lane given by ``lane_idx``."""
    return fx.Int32(fly_rocdl.ds_bpermute(T.i32, lane_idx * 4, val))


def _fence_system_release():
    """System-scope release fence, the write side of the grid barrier."""
    llvm.FenceOp(llvm.AtomicOrdering.release, syncscope="one-as")


def _load_i64_monotonic(addr_i64):
    """Relaxed *atomic* i64 load.

    ``communication_ops_utils.load_i64_global`` emits a plain load, which LLVM is
    free to hoist out of a spin loop; an atomic ordering forbids that.
    """

    ptr = llvm.IntToPtrOp(
        llvm.PointerType.get(address_space=1), _unwrap(addr_i64)
    ).result
    i64 = _ir.IntegerType.get_signless(64)
    return llvm.LoadOp(
        i64,
        ptr,
        alignment=8,
        ordering=llvm.AtomicOrdering.monotonic,
        syncscope="agent",
    ).result


class MoonEPFusedGeometry:
    """Compile-time shape/launch constants for the fused planner."""

    def __init__(
        self,
        *,
        rank,
        world_size,
        num_tokens,
        top_k,
        num_experts,
        prefetch_slots,
        token_padding,
        num_dispatch_rows,
        num_vblocks=128,
        blocks=64,
        waves_per_block=None,
    ):
        if num_experts % world_size != 0:
            raise ValueError("num_experts must be divisible by world_size")

        self.rank = rank
        self.R = world_size
        self.E = num_experts
        self.B = prefetch_slots
        self.S = num_tokens
        self.K = top_k
        self.epn = num_experts // world_size
        self.N = num_tokens * top_k
        self.CAP = self.N
        self.NvS = num_dispatch_rows
        self.token_padding = token_padding
        self.G = num_experts + prefetch_slots

        # One wavefront owns one EP rank throughout ``meta``, so the block must
        # be at least ``world_size`` waves wide.
        self.waves_per_block = waves_per_block or max(world_size, 4)
        self.block_threads = self.waves_per_block * WAVE_SIZE
        if self.block_threads > 1024:
            raise ValueError(
                f"world_size={world_size} needs {self.block_threads} threads per "
                "block (one wavefront per EP rank); the hardware cap is 1024"
            )
        if blocks < 2:
            raise ValueError("the fused planner needs at least 2 blocks")
        self.blocks = blocks

        self.NV = max(1, num_vblocks)
        self.EPV = _ceil_div(_ceil_div(self.N, self.NV), WAVE_SIZE) * WAVE_SIZE
        self.hist_batches = self.EPV // WAVE_SIZE
        # match_any narrows on the expert id plus the out-of-range sentinel E.
        self.expert_bits = max(1, int(num_experts).bit_length())
        self.vblocks_per_lane = _ceil_div(self.NV, WAVE_SIZE)

        self.experts_per_lane = _ceil_div(self.E, WAVE_SIZE)
        self.groups_per_lane = _ceil_div(self.G, WAVE_SIZE)
        self.local_experts_per_lane = _ceil_div(self.epn, WAVE_SIZE)
        self.rem_stride = self.local_experts_per_lane * WAVE_SIZE
        # Every greedy round retires either one local expert or one receiver.
        self.greedy_rounds = self.epn + self.R

        # ``dst`` groups a token's K entries into one power-of-two lane group so
        # the duplicate scan stays inside the wavefront.
        self.KG = _next_pow2(self.K)
        if self.KG > WAVE_SIZE:
            raise ValueError(f"top_k={self.K} exceeds the {WAVE_SIZE}-lane group")
        self.dst_slots = num_tokens * self.KG

        # Flat LDS arena.  ``fx.struct`` Storage objects are not valid @flyc.jit
        # arguments, so the phase functions take one base pointer and index it
        # with trace-time constant offsets.
        self.lds = {}
        cursor = 0
        for name, size in (
            # hist: one running expert count array per wave, plus a trash slot
            # per wave for the out-of-range sentinel expert.
            ("cnt", self.waves_per_block * (self.E + 1)),
            # alloc[e][d]: routed entries of expert e executed by destination d.
            ("alloc", self.E * self.R),
            # key[d][e] + a trash slot per destination: prefetch candidates.
            # -1 = not a candidate, -2 = selected.
            ("key", self.R * (self.E + 1)),
            ("ecount", self.E),
            ("rem", self.R * self.rem_stride),
            ("quota", self.R * self.R),
            ("bal", self.R),
            ("done", self.R),
            ("rstat0", self.R),
            ("rstat1", self.R),
            ("etc", self.R * self.B),
        ):
            self.lds[name] = cursor
            cursor += size
        self.lds_slots = cursor

    @property
    def key(self):
        return (
            self.rank,
            self.R,
            self.E,
            self.B,
            self.S,
            self.K,
            self.NvS,
            self.token_padding,
            self.NV,
            self.EPV,
            self.blocks,
            self.waves_per_block,
        )

    def scratch_sizes(self):
        return {
            "order": self.N,  # int32
            "local_hist": self.E * self.NV,  # int32, expert-major
            "tpe_prefix": self.E,  # int32
            "alloc_cumsum": self.E * self.R,  # int32
            "expert_off": self.R * self.E,  # int32
            "barrier": 4,  # int64 ticket counters
        }


def make_moonep_fused_plan_kernel(geo):
    """Build the single fused planning kernel for ``geo``."""

    R, E, B, K, KG = geo.R, geo.E, geo.B, geo.K, geo.KG
    epn, CAP, G = geo.epn, geo.CAP, geo.G
    NV, EPV, N, NvS = geo.NV, geo.EPV, geo.N, geo.NvS
    RANK, TP = geo.rank, geo.token_padding
    BLOCK, BLOCKS, WPB = geo.block_threads, geo.blocks, geo.waves_per_block
    EPL, GPL, LPL = (
        geo.experts_per_lane,
        geo.groups_per_lane,
        geo.local_experts_per_lane,
    )
    REM_STRIDE, ROUNDS = geo.rem_stride, geo.greedy_rounds
    CH = geo.vblocks_per_lane
    EXPERT_BITS = geo.expert_bits
    HIST_SLOTS = E + 1
    RANK_BITS = max(1, int(E).bit_length())
    RANK_MASK_BITS = (1 << RANK_BITS) - 1
    if RANK_BITS * EPL > 63:
        raise ValueError(
            f"packed top-B ranks need {RANK_BITS * EPL} bits for E={E}; "
            "the i64 accumulator holds 63"
        )
    RANK_MASK = [1 if r < RANK else 0 for r in range(R)]
    O = geo.lds  # noqa: E741 - trace-time constant LDS offsets

    ARENA_SLOTS = geo.lds_slots

    @fx.struct
    class _Arena:
        data: fx.Array[fx.Int32, ARENA_SLOTS, 16]

    # ------------------------------------------------------------------
    # Phase bodies.  These must be @flyc.jit: FlyDSL's AST rewriter only
    # transforms the body of a decorated function, so dynamic ``for`` / ``if`` /
    # ``while`` in a plain helper would stay ordinary Python and blow up.
    # ------------------------------------------------------------------

    @flyc.jit
    def _grid_barrier(bar_addr, slot, tid):
        """Ticket-based grid barrier over ``BLOCKS`` co-resident blocks.

        The leading release fence + workgroup barrier are not optional: thread 0
        signals arrival on behalf of the whole block, so every other thread's
        global stores must be complete *and* visible before the ticket is taken.
        Without them the phases race and the plan comes out non-deterministic.

        The wait is ">= end of my epoch" rather than "== target" so a later call
        of the same kernel cannot race past a slow waiter and hang it.
        """

        _fence_system_release()
        gpu.barrier()
        slot_addr = bar_addr + fx.Int64(slot * 8)
        if tid == fx.Int32(0):
            ticket = fx.Uint64(atomic_add_global_at(slot_addr, fx.Uint64(1)))
            target = (ticket // fx.Uint64(BLOCKS) + fx.Uint64(1)) * fx.Uint64(BLOCKS)
            current = fx.Uint64(_load_i64_monotonic(slot_addr))
            while current < target:
                current = fx.Uint64(_load_i64_monotonic(slot_addr))
        gpu.barrier()
        # Every thread, not just thread 0, is about to read peer blocks' writes.
        fence_system_acquire()

    @flyc.jit
    def _phase_hist(arena, topk_addr, order_addr, hist_addr, lane, wave, first_vblock):
        """Per-vblock histogram plus each entry's intra-vblock rank."""

        topk_rsrc = _addr_rsrc(topk_addr)
        order_rsrc = _addr_rsrc(order_addr)
        hist_rsrc = _addr_rsrc(hist_addr)
        cnt_base = fx.Int32(O["cnt"]) + wave * fx.Int32(HIST_SLOTS)

        for vblock in range(first_vblock, fx.Int32(NV), fx.Int32((BLOCKS - 1) * WPB)):
            for e in range(lane, fx.Int32(HIST_SLOTS), WAVE_SIZE):
                _lds_store(arena, fx.Int32(0), cnt_base + e)

            vblock_begin = vblock * fx.Int32(EPV)
            for batch in range_constexpr(geo.hist_batches):
                idx = vblock_begin + fx.Int32(batch * WAVE_SIZE) + lane
                in_range = idx < fx.Int32(N)
                # Every lane stays active so the ballots see a full wavefront;
                # the sentinel expert E gives out-of-range lanes their own group.
                safe_idx = in_range.select(idx, fx.Int32(0))
                expert = in_range.select(
                    buffer_load_i32(topk_rsrc, safe_idx), fx.Int32(E)
                )

                same = _wave_match_any(expert, EXPERT_BITS)
                rank_before = fx.Int32(_popcount64(same & _lanemask_lt(lane)))
                group_size = fx.Int32(_popcount64(same))

                base = _lds_load(arena, cnt_base + expert)
                if in_range:
                    buffer_ops.buffer_store(base + rank_before, order_rsrc, idx)
                # The group's last lane publishes the updated running count; it
                # read ``base`` in the same wave-synchronous step as everyone.
                if rank_before + fx.Int32(1) == group_size:
                    _lds_store(arena, base + group_size, cnt_base + expert)

            for e in range(lane, fx.Int32(E), WAVE_SIZE):
                buffer_ops.buffer_store(
                    _lds_load(arena, cnt_base + e),
                    hist_rsrc,
                    e * fx.Int32(NV) + vblock,
                )

    @flyc.jit
    def _phase_meta(
        arena,
        tpe_addr,
        tpe_prefix_addr,
        cumsum_addr,
        expert_off_addr,
        alloc_out_addr,
        etc_addr,
        cu_addr,
        zf_addr,
        gei_addr,
        stats_addr,
        tid,
        lane,
        wave,
    ):
        """Balancing greedy, top-B prefetch selection, and physical layout."""

        tpe_rsrc = _addr_rsrc(tpe_addr)
        tpe_prefix_rsrc = _addr_rsrc(tpe_prefix_addr)
        cumsum_rsrc = _addr_rsrc(cumsum_addr)
        expert_off_rsrc = _addr_rsrc(expert_off_addr)
        alloc_out_rsrc = _addr_rsrc(alloc_out_addr)
        etc_rsrc = _addr_rsrc(etc_addr)
        cu_rsrc = _addr_rsrc(cu_addr)
        zf_rsrc = _addr_rsrc(zf_addr)
        gei_rsrc = _addr_rsrc(gei_addr)
        stats_rsrc = _addr_rsrc(stats_addr)

        arena_base = fx.Int64(ptrtoint(arena))

        # -- zero the small LDS accumulators ------------------------------
        for i in range(tid, fx.Int32(R * R), BLOCK):
            _lds_store(arena, fx.Int32(0), fx.Int32(O["quota"]) + i)
        for i in range(tid, fx.Int32(R), BLOCK):
            _lds_store(arena, fx.Int32(0), fx.Int32(O["rstat0"]) + i)
            _lds_store(arena, fx.Int32(0), fx.Int32(O["rstat1"]) + i)
            _lds_store(arena, fx.Int32(0), fx.Int32(O["done"]) + i)
        for i in range(tid, fx.Int32(R * B), BLOCK):
            _lds_store(arena, fx.Int32(-1), fx.Int32(O["etc"]) + i)

        # -- expert_count and this rank's source-rank prefix ---------------
        for e in range(tid, fx.Int32(E), BLOCK):
            total = fx.Int32(0)
            prefix = fx.Int32(0)
            for r in range_constexpr(R):
                v = buffer_load_i32(tpe_rsrc, fx.Int32(r * E) + e)
                # RANK_MASK folds "source ranks before mine" into a trace-time
                # constant; FlyDSL owns every branch inside a kernel body.
                prefix = prefix + v * fx.Int32(RANK_MASK[r])
                total = total + v
            _lds_store(arena, total, fx.Int32(O["ecount"]) + e)
            buffer_ops.buffer_store(prefix, tpe_prefix_rsrc, e)

        gpu.barrier()

        # -- group_tokens / balance ---------------------------------------
        if tid < fx.Int32(R):
            group_total = fx.Int32(0)
            for j in range(fx.Int32(0), fx.Int32(epn), 1):
                group_total = group_total + _lds_load(
                    arena, fx.Int32(O["ecount"]) + tid * fx.Int32(epn) + j
                )
            _lds_store(
                arena, group_total - fx.Int32(CAP), fx.Int32(O["bal"]) + tid
            )

        # -- alloc[e][d] starts fully on the expert's home rank ------------
        for i in range(tid, fx.Int32(E * R), BLOCK):
            e = i // fx.Int32(R)
            d = i - e * fx.Int32(R)
            home_rank = e // fx.Int32(epn)
            _lds_store(
                arena,
                (d == home_rank).select(
                    _lds_load(arena, fx.Int32(O["ecount"]) + e), fx.Int32(0)
                ),
                fx.Int32(O["alloc"]) + i,
            )

        gpu.barrier()

        # -- receiver quotas: most overloaded home feeds the roomiest
        #    destination until no surplus is left.  O(R) rounds on thread 0.
        if tid == fx.Int32(0):
            for _round in range(fx.Int32(0), fx.Int32(R), 1):
                # Sentinel starts keep the scan branch-free; the strict > / <
                # then reproduce torch's first-extremum tie-break.
                best_h = fx.Int32(0)
                best_v = fx.Int32(_INT_MIN)
                worst_u = fx.Int32(0)
                worst_v = fx.Int32(_INT_MAX)
                for j in range_constexpr(R):
                    v = _lds_load(arena, fx.Int32(O["bal"] + j))
                    hi = v > best_v
                    best_v = hi.select(v, best_v)
                    best_h = hi.select(fx.Int32(j), best_h)
                    lo = v < worst_v
                    worst_v = lo.select(v, worst_v)
                    worst_u = lo.select(fx.Int32(j), worst_u)

                active = best_v > fx.Int32(0)
                move = active.select(fx.Int32(0) - worst_v, fx.Int32(0))
                q_slot = fx.Int32(O["quota"]) + best_h * fx.Int32(R) + worst_u
                _lds_store(
                    arena, active.select(move, _lds_load(arena, q_slot)), q_slot
                )
                # best_h != worst_u whenever active: the balances sum to zero,
                # so best_v > 0 forces worst_v < 0.
                _lds_store(arena, best_v - move, fx.Int32(O["bal"]) + best_h)
                _lds_store(
                    arena,
                    active.select(fx.Int32(0), worst_v),
                    fx.Int32(O["bal"]) + worst_u,
                )

        gpu.barrier()

        # -- resolve each home group's quotas into exact expert allocations -
        # Wave ``home`` owns home group ``home``; the homes are independent.
        home = wave
        rem_base = fx.Int32(O["rem"]) + home * fx.Int32(REM_STRIDE)
        for c in range_constexpr(LPL):
            local_e = fx.Int32(c * WAVE_SIZE) + lane
            in_range = local_e < fx.Int32(epn)
            value = _lds_load(
                arena,
                fx.Int32(O["ecount"])
                + home * fx.Int32(epn)
                + in_range.select(local_e, fx.Int32(0)),
            )
            _lds_store(
                arena,
                in_range.select(value, fx.Int32(_INT_MIN)),
                rem_base + fx.Int32(c * WAVE_SIZE) + lane,
            )

        for _round in range(fx.Int32(0), fx.Int32(ROUNDS), 1):
            # ROUNDS is the worst case (every round retires an expert or a
            # receiver).  In practice only a handful do work, so an SGPR guard
            # lets the wave branch over the rest instead of predicating through
            # them -- this is where most of the greedy's ~32 us went.
            still_running = _uniform_i32(
                _lds_load(arena, fx.Int32(O["done"]) + home)
            )
            if still_running == fx.Int32(0):
                q_in_range = lane < fx.Int32(R)
                q_val = q_in_range.select(
                    _lds_load(
                        arena,
                        fx.Int32(O["quota"])
                        + home * fx.Int32(R)
                        + q_in_range.select(lane, fx.Int32(0)),
                    ),
                    fx.Int32(_INT_MIN),
                )
                q_idx = q_in_range.select(lane, fx.Int32(1 << 30))
                quota, dest = _wave_argmax(q_val, q_idx, lane, prefer_low_index=True)

                best_v = fx.Int32(_INT_MIN)
                best_i = fx.Int32(1 << 30)
                for c in range_constexpr(LPL):
                    slot = fx.Int32(c * WAVE_SIZE) + lane
                    v = _lds_load(arena, rem_base + slot)
                    take = (v > best_v) | ((v == best_v) & (slot < best_i))
                    best_v = take.select(v, best_v)
                    best_i = take.select(slot, best_i)
                remaining, local_e = _wave_argmax(
                    best_v, best_i, lane, prefer_low_index=True
                )

                active = quota > fx.Int32(0)
                move = (remaining < quota).select(remaining, quota)
                move = active.select(move, fx.Int32(0))

                if lane == fx.Int32(0):
                    expert = home * fx.Int32(epn) + local_e
                    d_slot = fx.Int32(O["alloc"]) + expert * fx.Int32(R) + dest
                    h_slot = fx.Int32(O["alloc"]) + expert * fx.Int32(R) + home
                    _lds_store(arena, _lds_load(arena, d_slot) + move, d_slot)
                    _lds_store(arena, _lds_load(arena, h_slot) - move, h_slot)
                    _lds_store(
                        arena,
                        _lds_load(arena, rem_base + local_e) - move,
                        rem_base + local_e,
                    )
                    q_slot = fx.Int32(O["quota"]) + home * fx.Int32(R) + dest
                    _lds_store(arena, _lds_load(arena, q_slot) - move, q_slot)
                    _lds_store(
                        arena,
                        active.select(fx.Int32(0), fx.Int32(1)),
                        fx.Int32(O["done"]) + home,
                    )

        gpu.barrier()

        # -- publish alloc and its per-expert cumulative sum ----------------
        for e in range(tid, fx.Int32(E), BLOCK):
            acc = fx.Int32(0)
            for d in range_constexpr(R):
                acc = acc + _lds_load(
                    arena, fx.Int32(O["alloc"]) + e * fx.Int32(R) + fx.Int32(d)
                )
                buffer_ops.buffer_store(
                    acc, cumsum_rsrc, e * fx.Int32(R) + fx.Int32(d)
                )
        for i in range(tid, fx.Int32(E * R), BLOCK):
            e = i // fx.Int32(R)
            d = i - e * fx.Int32(R)
            buffer_ops.buffer_store(
                _lds_load(arena, fx.Int32(O["alloc"]) + i),
                alloc_out_rsrc,
                d * fx.Int32(E) + e,
            )

        # -- prefetch candidates: remote experts with a non-zero allocation -
        for i in range(tid, fx.Int32(R * E), BLOCK):
            d = i // fx.Int32(E)
            e = i - d * fx.Int32(E)
            is_local = (e // fx.Int32(epn)) == d
            a = _lds_load(arena, fx.Int32(O["alloc"]) + e * fx.Int32(R) + d)
            _lds_store(
                arena,
                (a > fx.Int32(0)).select(
                    is_local.select(fx.Int32(_KEY_NOT_CANDIDATE), a),
                    fx.Int32(_KEY_NOT_CANDIDATE),
                ),
                fx.Int32(O["key"]) + d * fx.Int32(E + 1) + e,
            )
        for d in range(tid, fx.Int32(R), BLOCK):
            _lds_store(
                arena,
                fx.Int32(_KEY_NOT_CANDIDATE),
                fx.Int32(O["key"]) + d * fx.Int32(E + 1) + fx.Int32(E),
            )

        gpu.barrier()

        # -- top-B prefetch selection, wave ``dest_rank`` owns destination d
        # A selection sort would be B dependent argmax passes.  Ranking each
        # candidate against the whole set is one parallel pass with the same
        # ordering, because sort(key=(alloc, e), reverse=True) is a total order:
        # a candidate lands in slot j exactly when j candidates outrank it.
        dest_rank = wave
        key_base = fx.Int32(O["key"]) + dest_rank * fx.Int32(E + 1)

        n_remote = fx.Int32(0)
        cand_vals = []
        cand_ids = []
        cand_ranks = []
        for c in range_constexpr(EPL):
            e = fx.Int32(c * WAVE_SIZE) + lane
            in_range = e < fx.Int32(E)
            value = _lds_load(arena, key_base + in_range.select(e, fx.Int32(E)))
            value = in_range.select(value, fx.Int32(_KEY_NOT_CANDIDATE))
            cand_vals.append(value)
            cand_ids.append(in_range.select(e, fx.Int32(-1)))
            n_remote = (value > fx.Int32(0)).select(n_remote + fx.Int32(1), n_remote)
        n_remote = _wave_sum(n_remote, lane)
        if lane == fx.Int32(0):
            _lds_store(arena, n_remote, fx.Int32(O["rstat0"]) + dest_rank)

        # One pass over the candidate set, not one pass per lane-chunk: the
        # EPL per-lane ranks are packed 10 bits apiece into a single i64 so the
        # dynamic loop carries one value.  That turns EPL*E dependent iterations
        # into E iterations with EPL-way ILP, which matters because a single
        # wavefront issues one VALU op every four cycles on CDNA.
        packed = fx.Uint64(0)
        for other in range(fx.Int32(0), fx.Int32(E), 1):
            a = _lds_load(arena, key_base + other)
            for c in range_constexpr(EPL):
                outranks = (a > cand_vals[c]) | (
                    (a == cand_vals[c]) & (other > cand_ids[c])
                )
                packed = packed + outranks.select(
                    fx.Uint64(1 << (RANK_BITS * c)), fx.Uint64(0)
                )
        for c in range_constexpr(EPL):
            cand_ranks.append(
                fx.Int32(
                    (packed >> fx.Uint64(RANK_BITS * c)) & fx.Uint64(RANK_MASK_BITS)
                )
            )

        # Apply only after every rank is known: writing _KEY_SELECTED earlier
        # would corrupt the comparisons of the later chunks.
        for c in range_constexpr(EPL):
            selected = (cand_vals[c] > fx.Int32(0)) & (cand_ranks[c] < fx.Int32(B))
            if selected:
                _lds_store(
                    arena,
                    cand_ids[c],
                    fx.Int32(O["etc"]) + dest_rank * fx.Int32(B) + cand_ranks[c],
                )
                _lds_store(arena, fx.Int32(_KEY_SELECTED), key_base + cand_ids[c])
                # remote_stats[:, 1] counts, per home rank, how many of its
                # experts some destination prefetches.
                _lds_atomic_add(
                    arena_base + fx.Int64(O["rstat1"] * 4),
                    cand_ids[c] // fx.Int32(epn),
                    fx.Int32(1),
                )

        # -- physical layout for destination ``dest_rank`` ------------------
        # Groups [0, E) are the global expert groups (minus the ones promoted to
        # a prefetch slot); groups [E, E + B) are the prefetch slots.
        counts = []
        paddeds = []
        experts = []
        lane_total = fx.Int32(0)
        for t in range_constexpr(GPL):
            g = lane * fx.Int32(GPL) + fx.Int32(t)
            in_range = g < fx.Int32(G)
            safe_g = in_range.select(g, fx.Int32(0))
            is_slot = safe_g >= fx.Int32(E)
            slot = is_slot.select(safe_g - fx.Int32(E), fx.Int32(0))
            slot_expert = _lds_load(
                arena, fx.Int32(O["etc"]) + dest_rank * fx.Int32(B) + slot
            )
            selected = _lds_load(
                arena, key_base + is_slot.select(fx.Int32(E), safe_g)
            ) == fx.Int32(_KEY_SELECTED)

            expert = is_slot.select(slot_expert, selected.select(fx.Int32(-1), safe_g))
            expert = in_range.select(expert, fx.Int32(-1))
            has_expert = expert >= fx.Int32(0)
            cnt = has_expert.select(
                _lds_load(
                    arena,
                    fx.Int32(O["alloc"])
                    + has_expert.select(expert, fx.Int32(0)) * fx.Int32(R)
                    + dest_rank,
                ),
                fx.Int32(0),
            )
            padded = (cnt > fx.Int32(0)).select(
                ((cnt + fx.Int32(TP - 1)) // fx.Int32(TP)) * fx.Int32(TP), fx.Int32(0)
            )
            counts.append(cnt)
            paddeds.append(padded)
            experts.append(expert)
            lane_total = lane_total + padded

        running = _wave_inclusive_prefix_sum(lane_total, lane) - lane_total

        for t in range_constexpr(GPL):
            g = lane * fx.Int32(GPL) + fx.Int32(t)
            in_range = g < fx.Int32(G)
            cnt = counts[t]
            padded = paddeds[t]
            expert = experts[t]
            has_expert = expert >= fx.Int32(0)
            nonempty = cnt > fx.Int32(0)

            start = running
            end = start + cnt
            padded_end = start + padded

            if in_range:
                if nonempty:
                    buffer_ops.buffer_store(
                        start,
                        expert_off_rsrc,
                        dest_rank * fx.Int32(E)
                        + has_expert.select(expert, fx.Int32(0)),
                    )
                if dest_rank == fx.Int32(RANK):
                    buffer_ops.buffer_store(padded_end, cu_rsrc, g)
                    buffer_ops.buffer_store(
                        nonempty.select(expert, fx.Int32(-1)), gei_rsrc, g
                    )
                    buffer_ops.buffer_store(
                        nonempty.select(end, fx.Int32(0)), zf_rsrc, g * fx.Int32(2)
                    )
                    buffer_ops.buffer_store(
                        nonempty.select(padded - cnt, fx.Int32(0)),
                        zf_rsrc,
                        g * fx.Int32(2) + fx.Int32(1),
                    )
            running = padded_end

        gpu.barrier()

        # -- publish experts_to_copy and this rank's remote_stats -----------
        for i in range(tid, fx.Int32(R * B), BLOCK):
            buffer_ops.buffer_store(
                _lds_load(arena, fx.Int32(O["etc"]) + i), etc_rsrc, i
            )
        if tid == fx.Int32(0):
            buffer_ops.buffer_store(
                _lds_load(arena, fx.Int32(O["rstat0"] + RANK)), stats_rsrc, 0
            )
            buffer_ops.buffer_store(
                _lds_load(arena, fx.Int32(O["rstat1"] + RANK)), stats_rsrc, 1
            )

    @flyc.jit
    def _phase_prefix(hist_addr, lane, first_expert):
        """Exclusive prefix of the per-vblock histograms, one wave per expert."""

        hist_rsrc = _addr_rsrc(hist_addr)
        for expert in range(first_expert, fx.Int32(E), fx.Int32(BLOCKS * WPB)):
            base = expert * fx.Int32(NV)
            counts = []
            lane_total = fx.Int32(0)
            for c in range_constexpr(CH):
                v = lane * fx.Int32(CH) + fx.Int32(c)
                in_range = v < fx.Int32(NV)
                value = in_range.select(
                    buffer_load_i32(
                        hist_rsrc, base + in_range.select(v, fx.Int32(0))
                    ),
                    fx.Int32(0),
                )
                counts.append(value)
                lane_total = lane_total + value

            running = _wave_inclusive_prefix_sum(lane_total, lane) - lane_total
            for c in range_constexpr(CH):
                v = lane * fx.Int32(CH) + fx.Int32(c)
                if v < fx.Int32(NV):
                    buffer_ops.buffer_store(running, hist_rsrc, base + v)
                running = running + counts[c]

    @flyc.jit
    def _phase_dst(
        topk_addr,
        order_addr,
        hist_addr,
        prefix_addr,
        cumsum_addr,
        expert_off_addr,
        dst_addr,
        lane,
        first_slot,
    ):
        """Resolve routed entries to destination rows, one lane per entry."""

        topk_rsrc = _addr_rsrc(topk_addr)
        order_rsrc = _addr_rsrc(order_addr)
        hist_rsrc = _addr_rsrc(hist_addr)
        prefix_rsrc = _addr_rsrc(prefix_addr)
        cumsum_rsrc = _addr_rsrc(cumsum_addr)
        expert_off_rsrc = _addr_rsrc(expert_off_addr)
        dst_rsrc = _addr_rsrc(dst_addr)

        group_base = lane & fx.Int32(~(KG - 1))
        k = lane & fx.Int32(KG - 1)

        for slot in range(
            first_slot, fx.Int32(geo.dst_slots), fx.Int32(BLOCKS * BLOCK)
        ):
            token = slot // fx.Int32(KG)
            active = k < fx.Int32(K)
            idx = token * fx.Int32(K) + k
            safe_idx = active.select(idx, fx.Int32(0))

            expert = buffer_load_i32(topk_rsrc, safe_idx)
            vblock = safe_idx // fx.Int32(EPV)
            global_index = (
                buffer_load_i32(prefix_rsrc, expert)
                + buffer_load_i32(hist_rsrc, expert * fx.Int32(NV) + vblock)
                + buffer_load_i32(order_rsrc, safe_idx)
            )

            # First destination whose cumulative allocation covers this entry;
            # equivalent to searchsorted(alloc_cumsum[e], g, right=True).
            dest = fx.Int32(R - 1)
            prev = fx.Int32(0)
            found = _false()
            acc_prev = fx.Int32(0)
            for d in range_constexpr(R):
                cum = buffer_load_i32(
                    cumsum_rsrc, expert * fx.Int32(R) + fx.Int32(d)
                )
                hit = (cum > global_index) & (~found)
                dest = hit.select(fx.Int32(d), dest)
                prev = hit.select(acc_prev, prev)
                found = found | hit
                acc_prev = cum

            local_offset = (
                buffer_load_i32(expert_off_rsrc, dest * fx.Int32(E) + expert)
                + global_index
                - prev
            )
            raw = dest * fx.Int32(NvS) + local_offset
            # Inactive lanes must never match a real destination below.
            dest = active.select(dest, fx.Int32(-1))

            # The first entry per destination rank keeps the payload; later ones
            # encode -raw - 1 and carry weights only.
            is_dup = _false()
            for j in range_constexpr(KG):
                peer_dest = _wave_read_lane(dest, group_base + fx.Int32(j))
                is_dup = is_dup | ((fx.Int32(j) < k) & active & (peer_dest == dest))

            if active:
                buffer_ops.buffer_store(
                    is_dup.select(fx.Int32(0) - raw - fx.Int32(1), raw), dst_rsrc, idx
                )

    suffix = f"r{R}_e{E}_b{B}_k{K}_nv{NV}_bl{BLOCKS}_rank{RANK}"

    @flyc.kernel(
        name=f"moonep_plan_fused_{suffix}", known_block_size=[BLOCK, 1, 1]
    )
    def fused_kernel(
        topk_experts: fx.Int64,
        tokens_per_expert: fx.Int64,
        order: fx.Int64,
        local_hist: fx.Int64,
        tpe_prefix: fx.Int64,
        alloc_cumsum: fx.Int64,
        expert_off: fx.Int64,
        alloc_out: fx.Int64,
        experts_to_copy: fx.Int64,
        cu_seqlens: fx.Int64,
        zero_fill: fx.Int64,
        group_expert_ids: fx.Int64,
        remote_stats: fx.Int64,
        dst: fx.Int64,
        barrier: fx.Int64,
    ):
        tid = fx.Int32(fx.thread_idx.x)
        lane = tid & fx.Int32(WAVE_SIZE - 1)
        wave = tid >> fx.Int32(6)
        blk = fx.Int32(fx.block_idx.x)

        arena = fx.SharedAllocator().allocate(_Arena).peek().data.ptr

        # Phase A.  meta reads only tokens_per_expert and hist reads only
        # topk_experts, so block 0's single-workgroup metadata work hides the
        # whole histogram.  The branch is workgroup-uniform, so the barriers
        # inside meta are fine.
        if blk == fx.Int32(0):
            _phase_meta(
                arena,
                tokens_per_expert,
                tpe_prefix,
                alloc_cumsum,
                expert_off,
                alloc_out,
                experts_to_copy,
                cu_seqlens,
                zero_fill,
                group_expert_ids,
                remote_stats,
                tid,
                lane,
                wave,
            )
        else:
            _phase_hist(
                arena,
                topk_experts,
                order,
                local_hist,
                lane,
                wave,
                (blk - fx.Int32(1)) * fx.Int32(WPB) + wave,
            )

        _grid_barrier(barrier, 0, tid)

        _phase_prefix(local_hist, lane, blk * fx.Int32(WPB) + wave)

        _grid_barrier(barrier, 1, tid)

        _phase_dst(
            topk_experts,
            order,
            local_hist,
            tpe_prefix,
            alloc_cumsum,
            expert_off,
            dst,
            lane,
            blk * fx.Int32(BLOCK) + tid,
        )

    return fused_kernel


def make_moonep_fused_plan_jit(geo):
    """Return the single fused JIT launcher for ``geo``."""

    fused_kernel = make_moonep_fused_plan_kernel(geo)
    _key = geo.key
    blocks = geo.blocks
    block_threads = geo.block_threads

    @flyc.jit
    def launch_fused(
        topk_experts: fx.Int64,
        tokens_per_expert: fx.Int64,
        order: fx.Int64,
        local_hist: fx.Int64,
        tpe_prefix: fx.Int64,
        alloc_cumsum: fx.Int64,
        expert_off: fx.Int64,
        alloc_out: fx.Int64,
        experts_to_copy: fx.Int64,
        cu_seqlens: fx.Int64,
        zero_fill: fx.Int64,
        group_expert_ids: fx.Int64,
        remote_stats: fx.Int64,
        dst: fx.Int64,
        barrier: fx.Int64,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
    ):
        _ = _key
        fused_kernel(
            topk_experts,
            tokens_per_expert,
            order,
            local_hist,
            tpe_prefix,
            alloc_cumsum,
            expert_off,
            alloc_out,
            experts_to_copy,
            cu_seqlens,
            zero_fill,
            group_expert_ids,
            remote_stats,
            dst,
            barrier,
        ).launch(grid=(blocks, 1, 1), block=(block_threads, 1, 1), stream=stream)

    launch_fused.compile_hints = {
        "llvm_options": {
            "amdgpu-kernarg-preload": AITER_FLYDSL_KERNARG_PRELOAD,
            "amdgpu-kernarg-preload-count": AITER_FLYDSL_KERNARG_PRELOAD_COUNT,
        },
    }
    return launch_fused


__all__ = [
    "MoonEPFusedGeometry",
    "make_moonep_fused_plan_jit",
    "make_moonep_fused_plan_kernel",
]
