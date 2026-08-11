# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""GPU planning kernels for the MoonEP gfx950 path (three-launch variant).

``aiter.ops.flydsl.moonep.build_reference_plan`` is a CPU port of the upstream
MoonEP torch *reference* (``tests/planning_reference.py``); it walks ``S * K``
routed entries in a Python loop and is a correctness baseline only.  Upstream's
production planner is a single cooperative CuTe launch running Phase A/B/C/D.

FlyDSL has no cooperative launch, so this module splits the same algorithm into
three ordinary launches.  Cross-launch ordering replaces the software grid
barrier, and the phase boundaries are the two points where the upstream kernel
calls ``grid_sync`` anyway:

  1. ``order_hist``  grid-parallel.  Per-vblock expert histogram plus, for every
     routed entry, its rank among earlier entries of the same expert *inside*
     that vblock.  One wavefront owns one vblock.
  2. ``meta``        one workgroup.  The balancing greedy, the top-B prefetch
     selection, the per-destination physical layout, and the exclusive prefix of
     the per-vblock histograms.
  3. ``dst``         grid-parallel.  One thread per token: resolves the K routed
     entries to destination rows and canonicalises duplicates in registers.

Unlike upstream this planner needs **no cross-rank traffic**: the caller already
supplies the all-gathered ``tokens_per_expert[R, E]``, every rank sees identical
inputs, and the algorithm is deterministic, so each rank reproduces the same
``alloc`` locally.  Upstream's ``multimem_st_v4`` broadcast and its in-kernel
``cross_rank_barrier`` have no counterpart here.

Every tie-break follows ``build_reference_plan`` exactly so the outputs are
bit-identical to it:

* ``torch.argmax`` / ``torch.argmin`` return the *first* extremum, so the
  wave-level reductions resolve ties to the smaller index.
* ``remote_experts.sort(key=(alloc, e), reverse=True)`` orders by descending
  allocation and then descending expert id, so the top-B scan resolves ties to
  the *larger* expert id.
* ``order`` must follow flat routed-entry order, so it is built with an
  in-register lane comparison rather than an atomic counter.
"""

# NOTE: no ``from __future__ import annotations`` here.  ``fx.struct`` resolves
# the LDS field annotations eagerly, and stringified annotations break its
# layout computation.

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import llvm
from flydsl.expr import gpu, ptrtoint, range_constexpr
from flydsl.expr import rocdl as fly_rocdl
from flydsl.expr.typing import T

from aiter.ops.flydsl.kernels import buffer_ops
from aiter.ops.flydsl.kernels.tensor_shim import (
    AITER_FLYDSL_KERNARG_PRELOAD,
    AITER_FLYDSL_KERNARG_PRELOAD_COUNT,
)

WAVE_SIZE = 64

# DPP controls for the intra-wave inclusive prefix sum (same constants as
# ``moe_sorting_kernel``).
_DPP_ROW_SHR_1 = 0x111
_DPP_ROW_SHR_2 = 0x112
_DPP_ROW_SHR_4 = 0x114
_DPP_ROW_SHR_8 = 0x118
_DPP_ROW_MASK = 0xF
_DPP_BANK_MASK = 0xF

# ``s_key`` sentinels for the top-B prefetch scan.
_KEY_NOT_CANDIDATE = -1
_KEY_SELECTED = -2

_INT_MIN = -(2**31)
_INT_MAX = 2**31 - 1


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _unwrap(v):
    return v.ir_value() if hasattr(v, "ir_value") else v


def _addr_rsrc(addr_i64):
    """Buffer resource for a raw global address kernel argument.

    The existing MoonEP kernels pass ``fx.Int64`` addresses rather than
    ``fx.Pointer``; this planner follows the same ABI.
    """
    return buffer_ops.create_buffer_resource_from_addr(addr_i64)


def _false():
    """A constant-false predicate (FlyDSL has no boolean literal)."""
    return fx.Int32(0) == fx.Int32(1)


def _lds_load(ptr, idx):
    """Scalar i32 load from an LDS pointer at element offset ``idx``."""
    return fx.ptr_load(ptr + fx.Int64(idx))


def _lds_store(ptr, val, idx):
    """Scalar i32 store to an LDS pointer at element offset ``idx``."""
    fx.ptr_store(val, ptr + fx.Int64(idx))


def _lds_atomic_add(base_i64, idx, val):
    """Workgroup-scope ``atomicrmw add`` on an i32 LDS slot."""
    ptr = buffer_ops.create_llvm_ptr(base_i64 + fx.Int64(idx) * 4, address_space=3)
    raw_ptr = ptr._value if hasattr(ptr, "_value") else ptr
    return llvm.AtomicRMWOp(
        llvm.AtomicBinOp.add,
        raw_ptr,
        _unwrap(val),
        llvm.AtomicOrdering.monotonic,
        syncscope="workgroup",
        alignment=4,
    ).result


def _lane_read(val, lane_idx):
    """Read ``val`` from wave lane ``lane_idx`` (byte-addressed ds_bpermute)."""
    return fx.Int32(fly_rocdl.ds_bpermute(T.i32, fx.Int32(lane_idx) * 4, val))


def _wave_rotate(val, stride, lane):
    """Read ``val`` from lane ``(lane + stride) % 64``."""
    peer = (lane + fx.Int32(stride)) & fx.Int32(WAVE_SIZE - 1)
    return fx.Int32(fly_rocdl.ds_bpermute(T.i32, peer * 4, val))


def _wave_sum(val, lane):
    """All-lanes sum reduction over one wavefront."""
    for stride in (1, 2, 4, 8, 16, 32):
        val = val + _wave_rotate(val, stride, lane)
    return val


def _wave_argmax(val, idx, lane, *, prefer_low_index: bool):
    """All-lanes argmax over one wavefront.

    ``prefer_low_index`` mirrors ``torch.argmax`` (first extremum wins).  The
    top-B prefetch scan needs the opposite tie-break because
    ``sort(key=(alloc, e), reverse=True)`` puts the larger expert id first.
    """

    for stride in (1, 2, 4, 8, 16, 32):
        o_val = _wave_rotate(val, stride, lane)
        o_idx = _wave_rotate(idx, stride, lane)
        if prefer_low_index:
            tie_wins = o_idx < idx
        else:
            tie_wins = o_idx > idx
        take = (o_val > val) | ((o_val == val) & tie_wins)
        val = take.select(o_val, val)
        idx = take.select(o_idx, idx)
    return val, idx


def _wave_inclusive_prefix_sum(val, lane):
    """Inclusive prefix sum inside one wavefront.

    Four DPP ``row_shr`` steps cover each 16-lane row, then two ``ds_bpermute``
    steps stitch the rows together.  Ported from ``moe_sorting_kernel``.
    """

    val_raw = _unwrap(val)
    zero_raw = _unwrap(fx.Int32(0))

    for shift, dpp_op in (
        (1, _DPP_ROW_SHR_1),
        (2, _DPP_ROW_SHR_2),
        (4, _DPP_ROW_SHR_4),
        (8, _DPP_ROW_SHR_8),
    ):
        remote = fly_rocdl.update_dpp(
            T.i32, zero_raw, val_raw, dpp_op, _DPP_ROW_MASK, _DPP_BANK_MASK, True
        )
        val = (lane >= fx.Int32(shift)).select(val + fx.Int32(remote), val)
        val_raw = _unwrap(val)

    src_lane_16 = (lane & fx.Int32(0x30)) - fx.Int32(1)
    remote16 = fly_rocdl.ds_bpermute(T.i32, src_lane_16 * fx.Int32(4), val)
    val = (lane >= fx.Int32(16)).select(val + fx.Int32(remote16), val)

    src_lane_32 = (lane & fx.Int32(0x30)) - fx.Int32(17)
    remote32 = fly_rocdl.ds_bpermute(T.i32, src_lane_32 * fx.Int32(4), val)
    val = (lane >= fx.Int32(32)).select(val + fx.Int32(remote32), val)

    return val


# ---------------------------------------------------------------------------
# Launch geometry
# ---------------------------------------------------------------------------


class MoonEPPlanGeometry:
    """Compile-time shape/launch constants shared by the three kernels."""

    def __init__(
        self,
        *,
        rank: int,
        world_size: int,
        num_tokens: int,
        top_k: int,
        num_experts: int,
        prefetch_slots: int,
        token_padding: int,
        num_dispatch_rows: int,
        no_migration: bool = False,
        num_vblocks: int = 128,
        hist_waves_per_block: int = 4,
        dst_block_threads: int = 256,
        dst_blocks: int = 256,
    ):
        if num_experts % world_size != 0:
            raise ValueError("num_experts must be divisible by world_size")
        if world_size * WAVE_SIZE > 1024:
            raise ValueError(
                f"world_size={world_size} needs {world_size * WAVE_SIZE} threads for "
                "the meta kernel (one wavefront per EP rank); the cap is 1024"
            )
        if num_experts > world_size * WAVE_SIZE:
            # The meta kernel spreads per-expert work over its own threads.
            # Everything below still works, it just loops; keep the check as a
            # reminder that the block is sized by world_size, not by E.
            pass

        self.rank = rank
        self.R = world_size
        self.E = num_experts
        self.B = prefetch_slots
        self.S = num_tokens
        self.K = top_k
        self.epn = num_experts // world_size
        self.N = num_tokens * top_k
        self.CAP = self.N
        # Decode mode: leave every expert on its home rank. The balancer exists
        # to even out a prefill step carrying thousands of tokens; with one
        # token per sequence there is nothing to even out, and the fixed-shape
        # plan it produces makes the experts step cost max_tokens*topk rows to
        # serve a handful. Skipping it also empties the migration groups, which
        # removes the second experts call and the prefetch entirely.
        self.NO_MIG = bool(no_migration)
        self.NvS = num_dispatch_rows
        self.token_padding = token_padding
        self.G = num_experts + prefetch_slots

        # One wavefront owns one vblock, so num_vblocks must tile the grid.
        self.hist_waves_per_block = hist_waves_per_block
        nv = min(num_vblocks, max(1, _ceil_div(self.N, WAVE_SIZE)))
        nv = max(1, (nv // hist_waves_per_block) * hist_waves_per_block)
        self.NV = nv
        self.hist_blocks = nv // hist_waves_per_block
        # Entries per vblock, rounded up to a whole wave batch.
        self.EPV = _ceil_div(_ceil_div(self.N, nv), WAVE_SIZE) * WAVE_SIZE
        self.hist_batches = self.EPV // WAVE_SIZE

        self.meta_threads = world_size * WAVE_SIZE
        # Experts / groups handled per lane inside one wavefront.
        self.experts_per_lane = _ceil_div(self.E, WAVE_SIZE)
        self.groups_per_lane = _ceil_div(self.G, WAVE_SIZE)
        self.local_experts_per_lane = _ceil_div(self.epn, WAVE_SIZE)
        self.rem_stride = self.local_experts_per_lane * WAVE_SIZE
        # Each greedy round retires either one local expert or one receiver.
        self.greedy_rounds = self.epn + self.R

        self.dst_block_threads = dst_block_threads
        self.dst_blocks = min(dst_blocks, max(1, _ceil_div(self.S, dst_block_threads)))

    @property
    def key(self) -> tuple:
        """Everything that changes the emitted IR, for the JIT cache key."""
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
            self.hist_waves_per_block,
            self.dst_block_threads,
            self.dst_blocks,
        )

    def scratch_sizes(self) -> dict:
        """Element counts of the int32 scratch buffers the caller must allocate."""
        return {
            "order": self.N,
            "local_hist": self.NV * self.E,
            "tpe_prefix": self.E,
            "alloc_cumsum": self.E * self.R,
            "expert_off": self.R * self.E,
        }


# ---------------------------------------------------------------------------
# Kernel 1: per-vblock histogram + intra-vblock routed-entry order
# ---------------------------------------------------------------------------


def make_moonep_plan_order_hist_kernel(geo: MoonEPPlanGeometry):
    """Build the per-vblock histogram / ``order`` kernel.

    One wavefront owns ``EPV`` consecutive routed entries and walks them in
    64-entry batches.  Within a batch, lane ``l`` counts the lanes ``j < l`` that
    carry the same expert; adding the wave-local running count gives the entry's
    rank inside the vblock.  Only the batch's last occurrence of an expert
    publishes the updated running count, which keeps the whole step
    wave-synchronous (no LDS atomics, no barriers, deterministic order).
    """

    E = geo.E
    EPV = geo.EPV
    N = geo.N
    BATCHES = geo.hist_batches
    WPB = geo.hist_waves_per_block
    BLOCK = WPB * WAVE_SIZE

    @fx.struct
    class _HistLDS:
        # Per-wave running expert counts; each wave owns its own E-slice, so no
        # workgroup barrier is ever needed.
        cnt: fx.Array[fx.Int32, WPB * E, 16]

    @flyc.kernel(
        name=f"moonep_plan_order_hist_e{E}_epv{EPV}_w{WPB}",
        known_block_size=[BLOCK, 1, 1],
    )
    def order_hist_kernel(
        topk_experts: fx.Int64,  # int32 [S, K] flattened
        order: fx.Int64,  # int32 [N] out
        local_hist: fx.Int64,  # int32 [NV, E] out
    ):
        tid = fx.Int32(fx.thread_idx.x)
        lane = tid & fx.Int32(WAVE_SIZE - 1)
        wave = tid >> fx.Int32(6)
        vblock = fx.Int32(fx.block_idx.x) * fx.Int32(WPB) + wave

        cnt_ptr = fx.SharedAllocator().allocate(_HistLDS).peek().cnt.ptr
        cnt_base = wave * fx.Int32(E)

        topk_rsrc = _addr_rsrc(topk_experts)
        order_rsrc = _addr_rsrc(order)
        hist_rsrc = _addr_rsrc(local_hist)

        for e in range(lane, fx.Int32(E), WAVE_SIZE):
            _lds_store(cnt_ptr, fx.Int32(0), cnt_base + e)

        vblock_begin = vblock * fx.Int32(EPV)
        for batch in range_constexpr(BATCHES):
            idx = vblock_begin + fx.Int32(batch * WAVE_SIZE) + lane
            in_range = idx < fx.Int32(N)
            # Keep every lane active for ds_bpermute: clamp the load and mask the
            # expert id instead of branching.
            safe_idx = in_range.select(idx, fx.Int32(0))
            loaded = buffer_load_i32(topk_rsrc, safe_idx)
            expert = in_range.select(loaded, fx.Int32(-1))

            rank_before = fx.Int32(0)
            count_after = fx.Int32(0)
            for j in range_constexpr(WAVE_SIZE):
                peer_expert = _lane_read(expert, j)
                same = peer_expert == expert
                # expert == -1 for out-of-range lanes, which never matches a
                # real expert id, so no extra masking is needed here.
                rank_before = (same & (lane > fx.Int32(j))).select(
                    rank_before + fx.Int32(1), rank_before
                )
                count_after = (same & (lane < fx.Int32(j))).select(
                    count_after + fx.Int32(1), count_after
                )

            safe_expert = in_range.select(expert, fx.Int32(0))
            base = _lds_load(cnt_ptr, cnt_base + safe_expert)
            if in_range:
                buffer_ops.buffer_store(base + rank_before, order_rsrc, idx)
            # The last occurrence in the batch owns the running-count update; it
            # read ``base`` in the same wave-synchronous step as everyone else.
            is_last = in_range & (count_after == fx.Int32(0))
            if is_last:
                _lds_store(
                    cnt_ptr, base + rank_before + fx.Int32(1), cnt_base + safe_expert
                )

        hist_base = vblock * fx.Int32(E)
        for e in range(lane, fx.Int32(E), WAVE_SIZE):
            buffer_ops.buffer_store(
                _lds_load(cnt_ptr, cnt_base + e), hist_rsrc, hist_base + e
            )

    return order_hist_kernel


def buffer_load_i32(rsrc, idx):
    return fx.Int32(buffer_ops.buffer_load(rsrc, idx, vec_width=1, dtype=T.i32))


# ---------------------------------------------------------------------------
# Kernel 2: balancing greedy, prefetch selection, physical layout, vblock prefix
# ---------------------------------------------------------------------------


def make_moonep_plan_meta_kernel(geo: MoonEPPlanGeometry):
    """Build the single-workgroup planning-metadata kernel.

    Wavefront ``w`` owns EP rank ``w`` throughout: it resolves home group ``w``'s
    migration quotas, picks destination ``w``'s top-B prefetch experts, and lays
    out destination ``w``'s physical groups.  The only serial part is the
    receiver-quota greedy, which is O(R) and runs on thread 0.
    """

    R = geo.R
    E = geo.E
    B = geo.B
    epn = geo.epn
    CAP = geo.CAP
    NO_MIG = geo.NO_MIG
    NV = geo.NV
    G = geo.G
    RANK = geo.rank
    TP = geo.token_padding
    BLOCK = geo.meta_threads
    EPL = geo.experts_per_lane
    GPL = geo.groups_per_lane
    LPL = geo.local_experts_per_lane
    REM_STRIDE = geo.rem_stride
    ROUNDS = geo.greedy_rounds
    RANK_MASK = [1 if r < RANK else 0 for r in range(R)]

    @fx.struct
    class _MetaLDS:
        # alloc[e][d]: routed entries of expert e executed by destination rank d.
        alloc: fx.Array[fx.Int32, E * R, 16]
        # key[d][e] with one trailing trash slot per destination: the remaining
        # prefetch candidates.  -1 = not a candidate, -2 = already selected.
        key: fx.Array[fx.Int32, R * (E + 1), 16]
        ecount: fx.Array[fx.Int32, E, 16]
        # Per-home remaining token counts, padded to whole lanes.
        rem: fx.Array[fx.Int32, R * REM_STRIDE, 16]
        quota: fx.Array[fx.Int32, R * R, 16]
        bal: fx.Array[fx.Int32, R, 16]
        rstat0: fx.Array[fx.Int32, R, 16]
        rstat1: fx.Array[fx.Int32, R, 16]
        etc: fx.Array[fx.Int32, R * B, 16]

    @flyc.kernel(
        name=f"moonep_plan_meta_r{R}_e{E}_b{B}_tp{TP}_rank{RANK}"
        + ("_nomig" if NO_MIG else ""),
        known_block_size=[BLOCK, 1, 1],
    )
    def meta_kernel(
        tokens_per_expert: fx.Int64,  # int32 [R, E]
        local_hist: fx.Int64,  # int32 [NV, E] in/out (counts -> exclusive prefix)
        tpe_prefix: fx.Int64,  # int32 [E] out scratch
        alloc_cumsum: fx.Int64,  # int32 [E, R] out scratch
        expert_off: fx.Int64,  # int32 [R, E] out scratch
        alloc_out: fx.Int64,  # int32 [R, E] out
        experts_to_copy: fx.Int64,  # int32 [R, B] out
        cu_seqlens: fx.Int64,  # int32 [G] out (this rank)
        zero_fill: fx.Int64,  # int32 [G, 2] out (this rank)
        group_expert_ids: fx.Int64,  # int32 [G] out (this rank)
        remote_stats: fx.Int64,  # int32 [2] out (this rank)
    ):
        tid = fx.Int32(fx.thread_idx.x)
        lane = tid & fx.Int32(WAVE_SIZE - 1)
        wave = tid >> fx.Int32(6)

        lds = fx.SharedAllocator().allocate(_MetaLDS).peek()
        p_alloc = lds.alloc.ptr
        p_key = lds.key.ptr
        p_ecount = lds.ecount.ptr
        p_rem = lds.rem.ptr
        p_quota = lds.quota.ptr
        p_bal = lds.bal.ptr
        p_rstat0 = lds.rstat0.ptr
        p_rstat1 = lds.rstat1.ptr
        p_etc = lds.etc.ptr
        rstat1_base = fx.Int64(ptrtoint(p_rstat1))

        tpe_rsrc = _addr_rsrc(tokens_per_expert)
        hist_rsrc = _addr_rsrc(local_hist)
        tpe_prefix_rsrc = _addr_rsrc(tpe_prefix)
        cumsum_rsrc = _addr_rsrc(alloc_cumsum)
        expert_off_rsrc = _addr_rsrc(expert_off)
        alloc_out_rsrc = _addr_rsrc(alloc_out)
        etc_rsrc = _addr_rsrc(experts_to_copy)
        cu_rsrc = _addr_rsrc(cu_seqlens)
        zf_rsrc = _addr_rsrc(zero_fill)
        gei_rsrc = _addr_rsrc(group_expert_ids)
        stats_rsrc = _addr_rsrc(remote_stats)

        # -- zero the small LDS accumulators ---------------------------------
        for i in range(tid, fx.Int32(R * R), BLOCK):
            _lds_store(p_quota, fx.Int32(0), i)
        for i in range(tid, fx.Int32(R), BLOCK):
            _lds_store(p_rstat0, fx.Int32(0), i)
            _lds_store(p_rstat1, fx.Int32(0), i)

        # -- expert_count and this rank's source-rank prefix ------------------
        for e in range(tid, fx.Int32(E), BLOCK):
            total = fx.Int32(0)
            prefix = fx.Int32(0)
            for r in range_constexpr(R):
                v = buffer_load_i32(tpe_rsrc, fx.Int32(r * E) + e)
                # RANK_MASK folds the "source ranks before mine" test into a
                # trace-time constant: FlyDSL's AST rewriter owns every branch
                # inside a kernel body, so plain Python control flow is out.
                prefix = prefix + v * fx.Int32(RANK_MASK[r])
                total = total + v
            _lds_store(p_ecount, total, e)
            buffer_ops.buffer_store(prefix, tpe_prefix_rsrc, e)

        # -- exclusive prefix of the per-vblock histograms --------------------
        # Independent of everything else in this kernel; folded in here so the
        # three-launch variant does not need a fourth launch.
        for e in range(tid, fx.Int32(E), BLOCK):
            acc = fx.Int32(0)
            for v in range(fx.Int32(0), fx.Int32(NV), 1):
                slot = v * fx.Int32(E) + e
                c = buffer_load_i32(hist_rsrc, slot)
                buffer_ops.buffer_store(acc, hist_rsrc, slot)
                acc = acc + c

        gpu.barrier()

        # -- group_tokens / balance ------------------------------------------
        if tid < fx.Int32(R):
            group_total = fx.Int32(0)
            for j in range(fx.Int32(0), fx.Int32(epn), 1):
                group_total = group_total + _lds_load(p_ecount, tid * fx.Int32(epn) + j)
            # NO_MIG is a Python bool, so this is a compile-time branch: the
            # balance is pinned to zero and the quota loop below never activates.
            _lds_store(
                p_bal,
                fx.Int32(0) if NO_MIG else group_total - fx.Int32(CAP),
                tid,
            )

        # -- alloc[e][d] starts fully on the expert's home rank ---------------
        for i in range(tid, fx.Int32(E * R), BLOCK):
            e = i // fx.Int32(R)
            d = i - e * fx.Int32(R)
            home = e // fx.Int32(epn)
            _lds_store(p_alloc, (d == home).select(_lds_load(p_ecount, e), fx.Int32(0)), i)

        gpu.barrier()

        # -- receiver quotas: pick the most overloaded home and the roomiest
        #    destination until no surplus is left.  O(R) rounds, so thread 0
        #    runs it serially; every write is predicated to avoid branches.
        if tid == fx.Int32(0):
            for _round in range(fx.Int32(0), fx.Int32(R), 1):
                # Sentinel starts keep the scan branch-free: the strict > / <
                # comparisons then reproduce torch's first-extremum tie-break
                # without special-casing j == 0.
                best_h = fx.Int32(0)
                best_v = fx.Int32(_INT_MIN)
                worst_u = fx.Int32(0)
                worst_v = fx.Int32(_INT_MAX)
                for j in range_constexpr(R):
                    v = _lds_load(p_bal, fx.Int32(j))
                    hi = v > best_v  # torch.argmax keeps the first maximum
                    best_v = hi.select(v, best_v)
                    best_h = hi.select(fx.Int32(j), best_h)
                    lo = v < worst_v  # torch.argmin keeps the first minimum
                    worst_v = lo.select(v, worst_v)
                    worst_u = lo.select(fx.Int32(j), worst_u)

                active = best_v > fx.Int32(0)
                move = active.select(fx.Int32(0) - worst_v, fx.Int32(0))
                q_slot = best_h * fx.Int32(R) + worst_u
                _lds_store(
                    p_quota,
                    active.select(move, _lds_load(p_quota, q_slot)),
                    q_slot,
                )
                # best_h != worst_u whenever active, because the balances sum to
                # zero and best_v > 0 forces worst_v < 0.
                _lds_store(p_bal, best_v - move, best_h)
                _lds_store(p_bal, active.select(fx.Int32(0), worst_v), worst_u)

        gpu.barrier()

        # -- resolve each home group's quotas into exact expert allocations ---
        # Wave ``h`` owns home group ``h``; the homes are independent.
        home = wave
        rem_base = home * fx.Int32(REM_STRIDE)
        for c in range_constexpr(LPL):
            local_e = fx.Int32(c * WAVE_SIZE) + lane
            in_range = local_e < fx.Int32(epn)
            safe_e = in_range.select(local_e, fx.Int32(0))
            v = _lds_load(p_ecount, home * fx.Int32(epn) + safe_e)
            _lds_store(
                p_rem,
                in_range.select(v, fx.Int32(_INT_MIN)),
                rem_base + fx.Int32(c * WAVE_SIZE) + lane,
            )

        for _round in range(fx.Int32(0), fx.Int32(ROUNDS), 1):
            # Largest remaining receiver quota (first maximum on ties).
            q_in_range = lane < fx.Int32(R)
            q_val = q_in_range.select(
                _lds_load(p_quota, home * fx.Int32(R) + q_in_range.select(lane, fx.Int32(0))),
                fx.Int32(_INT_MIN),
            )
            q_idx = q_in_range.select(lane, fx.Int32(1 << 30))
            quota, dest = _wave_argmax(q_val, q_idx, lane, prefer_low_index=True)

            # Local expert with the most remaining tokens (first maximum).
            best_v = fx.Int32(_INT_MIN)
            best_i = fx.Int32(1 << 30)
            for c in range_constexpr(LPL):
                slot = fx.Int32(c * WAVE_SIZE) + lane
                v = _lds_load(p_rem, rem_base + slot)
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
                d_slot = expert * fx.Int32(R) + dest
                h_slot = expert * fx.Int32(R) + home
                _lds_store(p_alloc, _lds_load(p_alloc, d_slot) + move, d_slot)
                _lds_store(p_alloc, _lds_load(p_alloc, h_slot) - move, h_slot)
                _lds_store(
                    p_rem,
                    _lds_load(p_rem, rem_base + local_e) - move,
                    rem_base + local_e,
                )
                q_slot = home * fx.Int32(R) + dest
                _lds_store(p_quota, _lds_load(p_quota, q_slot) - move, q_slot)

        gpu.barrier()

        # -- publish alloc and its per-expert cumulative sum over destinations -
        for e in range(tid, fx.Int32(E), BLOCK):
            acc = fx.Int32(0)
            for d in range_constexpr(R):
                acc = acc + _lds_load(p_alloc, e * fx.Int32(R) + fx.Int32(d))
                buffer_ops.buffer_store(acc, cumsum_rsrc, e * fx.Int32(R) + fx.Int32(d))
        for i in range(tid, fx.Int32(E * R), BLOCK):
            e = i // fx.Int32(R)
            d = i - e * fx.Int32(R)
            buffer_ops.buffer_store(
                _lds_load(p_alloc, i), alloc_out_rsrc, d * fx.Int32(E) + e
            )

        # -- prefetch candidates: remote experts with a non-zero allocation ----
        for i in range(tid, fx.Int32(R * E), BLOCK):
            d = i // fx.Int32(E)
            e = i - d * fx.Int32(E)
            is_local = (e // fx.Int32(epn)) == d
            a = _lds_load(p_alloc, e * fx.Int32(R) + d)
            _lds_store(
                p_key,
                (a > fx.Int32(0)).select(
                    is_local.select(fx.Int32(_KEY_NOT_CANDIDATE), a),
                    fx.Int32(_KEY_NOT_CANDIDATE),
                ),
                d * fx.Int32(E + 1) + e,
            )
        # Trash slot: absorbs the predicated "selected" write when no candidate
        # is left, so the selection loop needs no branch.
        for d in range(tid, fx.Int32(R), BLOCK):
            _lds_store(p_key, fx.Int32(_KEY_NOT_CANDIDATE), d * fx.Int32(E + 1) + fx.Int32(E))
        # Slots with no candidate are never written by the selection below.
        for i in range(tid, fx.Int32(R * B), BLOCK):
            _lds_store(p_etc, fx.Int32(-1), i)

        gpu.barrier()

        # -- top-B prefetch selection, wave ``d`` owns destination ``d`` -------
        dest_rank = wave
        key_base = dest_rank * fx.Int32(E + 1)

        # Rank every candidate against the whole set in one pass rather than
        # running B dependent argmax passes.  The orderings agree because
        # sort(key=(alloc, e), reverse=True) is a total order: a candidate
        # lands in slot j exactly when j candidates outrank it.  The B-pass
        # form was also wrong -- the _KEY_SELECTED written by pass b was not
        # visible to pass b+1, so every slot picked the same expert, which
        # only shows up once a destination has fewer than B remote candidates.
        n_remote = fx.Int32(0)
        cand_vals = []
        cand_ids = []
        for c in range_constexpr(EPL):
            e = fx.Int32(c * WAVE_SIZE) + lane
            in_range = e < fx.Int32(E)
            v = _lds_load(p_key, key_base + in_range.select(e, fx.Int32(E)))
            v = in_range.select(v, fx.Int32(_KEY_NOT_CANDIDATE))
            cand_vals.append(v)
            cand_ids.append(in_range.select(e, fx.Int32(-1)))
            n_remote = (v > fx.Int32(0)).select(n_remote + fx.Int32(1), n_remote)
        n_remote = _wave_sum(n_remote, lane)
        if lane == fx.Int32(0):
            _lds_store(p_rstat0, n_remote, dest_rank)

        ranks = [fx.Int32(0) for _ in range(EPL)]
        for other in range(fx.Int32(0), fx.Int32(E), 1):
            a = _lds_load(p_key, key_base + other)
            for c in range_constexpr(EPL):
                outranks = (a > cand_vals[c]) | (
                    (a == cand_vals[c]) & (other > cand_ids[c])
                )
                ranks[c] = ranks[c] + outranks.select(fx.Int32(1), fx.Int32(0))

        # Applied only once every rank is known: an earlier _KEY_SELECTED would
        # corrupt the comparisons of the later chunks.
        for c in range_constexpr(EPL):
            selected = (cand_vals[c] > fx.Int32(0)) & (ranks[c] < fx.Int32(B))
            if selected:
                _lds_store(p_etc, cand_ids[c], dest_rank * fx.Int32(B) + ranks[c])
                _lds_store(p_key, fx.Int32(_KEY_SELECTED), key_base + cand_ids[c])
                # remote_stats[:, 1] counts, per home rank, how many of its
                # experts some destination prefetches.
                _lds_atomic_add(
                    rstat1_base, cand_ids[c] // fx.Int32(epn), fx.Int32(1)
                )

        gpu.barrier()

        # -- physical layout for destination ``dest_rank`` ---------------------
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
            slot_expert = _lds_load(p_etc, dest_rank * fx.Int32(B) + slot)
            selected = _lds_load(p_key, key_base + is_slot.select(fx.Int32(E), safe_g)) == fx.Int32(
                _KEY_SELECTED
            )

            expert = is_slot.select(slot_expert, selected.select(fx.Int32(-1), safe_g))
            expert = in_range.select(expert, fx.Int32(-1))
            has_expert = expert >= fx.Int32(0)
            safe_expert = has_expert.select(expert, fx.Int32(0))
            cnt = has_expert.select(
                _lds_load(p_alloc, safe_expert * fx.Int32(R) + dest_rank), fx.Int32(0)
            )
            padded = (cnt > fx.Int32(0)).select(
                ((cnt + fx.Int32(TP - 1)) // fx.Int32(TP)) * fx.Int32(TP), fx.Int32(0)
            )
            counts.append(cnt)
            paddeds.append(padded)
            experts.append(expert)
            lane_total = lane_total + padded

        lane_base = _wave_inclusive_prefix_sum(lane_total, lane) - lane_total

        running = lane_base
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
                        dest_rank * fx.Int32(E) + has_expert.select(expert, fx.Int32(0)),
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

        # -- publish experts_to_copy and this rank's remote_stats --------------
        for i in range(tid, fx.Int32(R * B), BLOCK):
            buffer_ops.buffer_store(_lds_load(p_etc, i), etc_rsrc, i)
        if tid == fx.Int32(0):
            buffer_ops.buffer_store(_lds_load(p_rstat0, fx.Int32(RANK)), stats_rsrc, 0)
            buffer_ops.buffer_store(_lds_load(p_rstat1, fx.Int32(RANK)), stats_rsrc, 1)

    return meta_kernel


# ---------------------------------------------------------------------------
# Kernel 3: routed entry -> destination row, plus duplicate canonicalisation
# ---------------------------------------------------------------------------


def make_moonep_plan_dst_kernel(geo: MoonEPPlanGeometry):
    """Build the ``dst`` kernel.

    One thread owns one token and therefore all ``K`` of its routed entries, so
    the duplicate canonicalisation is a register-only ``O(K^2)`` comparison and
    needs no second pass.
    """

    R = geo.R
    E = geo.E
    K = geo.K
    S = geo.S
    EPV = geo.EPV
    NvS = geo.NvS
    BLOCK = geo.dst_block_threads
    GRID = geo.dst_blocks

    @flyc.kernel(
        name=f"moonep_plan_dst_r{R}_e{E}_k{K}_nvs{NvS}_epv{EPV}",
        known_block_size=[BLOCK, 1, 1],
    )
    def dst_kernel(
        topk_experts: fx.Int64,  # int32 [S, K] flattened
        order: fx.Int64,  # int32 [N]
        local_hist: fx.Int64,  # int32 [NV, E] exclusive vblock prefix
        tpe_prefix: fx.Int64,  # int32 [E]
        alloc_cumsum: fx.Int64,  # int32 [E, R]
        expert_off: fx.Int64,  # int32 [R, E]
        dst: fx.Int64,  # int32 [S, K] out
        num_tokens: fx.Int32,
    ):
        topk_rsrc = _addr_rsrc(topk_experts)
        order_rsrc = _addr_rsrc(order)
        hist_rsrc = _addr_rsrc(local_hist)
        prefix_rsrc = _addr_rsrc(tpe_prefix)
        cumsum_rsrc = _addr_rsrc(alloc_cumsum)
        expert_off_rsrc = _addr_rsrc(expert_off)
        dst_rsrc = _addr_rsrc(dst)

        gid = fx.Int32(fx.block_idx.x) * fx.Int32(BLOCK) + fx.Int32(fx.thread_idx.x)
        gstride = fx.Int32(GRID * BLOCK)

        for token in range(gid, num_tokens, gstride):
            raws = []
            dests = []
            for k in range_constexpr(K):
                idx = token * fx.Int32(K) + fx.Int32(k)
                expert = buffer_load_i32(topk_rsrc, idx)
                vblock = idx // fx.Int32(EPV)
                global_index = (
                    buffer_load_i32(prefix_rsrc, expert)
                    + buffer_load_i32(hist_rsrc, vblock * fx.Int32(E) + expert)
                    + buffer_load_i32(order_rsrc, idx)
                )

                # First destination whose cumulative allocation covers this
                # entry; equivalent to searchsorted(alloc_cumsum[e], g, right).
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
                raws.append(dest * fx.Int32(NvS) + local_offset)
                dests.append(dest)

            for k in range_constexpr(K):
                # The first entry per destination rank keeps the payload; later
                # ones encode -raw - 1 and carry weights only.
                is_dup = _false()
                for j in range_constexpr(k):
                    is_dup = is_dup | (dests[j] == dests[k])
                buffer_ops.buffer_store(
                    is_dup.select(fx.Int32(0) - raws[k] - fx.Int32(1), raws[k]),
                    dst_rsrc,
                    token * fx.Int32(K) + fx.Int32(k),
                )

    return dst_kernel


# ---------------------------------------------------------------------------
# JIT launchers
# ---------------------------------------------------------------------------


def make_moonep_plan_jit(geo: MoonEPPlanGeometry):
    """Return ``(order_hist, meta, dst)`` FlyDSL JIT launchers for ``geo``."""

    order_hist_kernel = make_moonep_plan_order_hist_kernel(geo)
    meta_kernel = make_moonep_plan_meta_kernel(geo)
    dst_kernel = make_moonep_plan_dst_kernel(geo)

    # Make every IR-affecting value visible to FlyDSL's closure-based cache key.
    _key = geo.key
    hist_blocks = geo.hist_blocks
    hist_block_threads = geo.hist_waves_per_block * WAVE_SIZE
    meta_threads = geo.meta_threads
    dst_blocks = geo.dst_blocks
    dst_block_threads = geo.dst_block_threads

    @flyc.jit
    def launch_order_hist(
        topk_experts: fx.Int64,
        order: fx.Int64,
        local_hist: fx.Int64,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
    ):
        _ = _key
        order_hist_kernel(topk_experts, order, local_hist).launch(
            grid=(hist_blocks, 1, 1),
            block=(hist_block_threads, 1, 1),
            stream=stream,
        )

    @flyc.jit
    def launch_meta(
        tokens_per_expert: fx.Int64,
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
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
    ):
        _ = _key
        meta_kernel(
            tokens_per_expert,
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
        ).launch(
            grid=(1, 1, 1),
            block=(meta_threads, 1, 1),
            stream=stream,
        )

    @flyc.jit
    def launch_dst(
        topk_experts: fx.Int64,
        order: fx.Int64,
        local_hist: fx.Int64,
        tpe_prefix: fx.Int64,
        alloc_cumsum: fx.Int64,
        expert_off: fx.Int64,
        dst: fx.Int64,
        num_tokens: fx.Int32,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
    ):
        _ = _key
        dst_kernel(
            topk_experts,
            order,
            local_hist,
            tpe_prefix,
            alloc_cumsum,
            expert_off,
            dst,
            num_tokens,
        ).launch(
            grid=(dst_blocks, 1, 1),
            block=(dst_block_threads, 1, 1),
            stream=stream,
        )

    for fn in (launch_order_hist, launch_meta, launch_dst):
        fn.compile_hints = {
            "llvm_options": {
                "amdgpu-kernarg-preload": AITER_FLYDSL_KERNARG_PRELOAD,
                "amdgpu-kernarg-preload-count": AITER_FLYDSL_KERNARG_PRELOAD_COUNT,
            },
        }

    return launch_order_hist, launch_meta, launch_dst


__all__ = [
    "MoonEPPlanGeometry",
    "make_moonep_plan_dst_kernel",
    "make_moonep_plan_jit",
    "make_moonep_plan_meta_kernel",
    "make_moonep_plan_order_hist_kernel",
]
