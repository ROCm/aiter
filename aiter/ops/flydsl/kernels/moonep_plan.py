# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
"""MoonEP route planning in front of MegaMoE.

Every rank counts its logical routes per expert, the group exchanges those
histograms once, and every rank runs the same deterministic placement on the
gathered matrix: balance towards the all-rank average, keep each
destination's top-B remote experts, return the rest to their owners, and
settle the prefetch slots stickily so a slot keeping its expert needs no
copy.  Each route is then rewritten to the virtual id of the replica it was
given, so an ordinary MegaMoEV2 over ``R * (EPR + B)`` experts runs it
unchanged.  Placement and slot tables are identical on every rank without
further communication.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl._mlir.dialects import llvm
from flydsl.expr import ptrtoint, range_constexpr
from flydsl.expr import rocdl as fly_rocdl
from flydsl.expr.typing import Stream, T

from . import communication_ops_utils as comm_ops
from .kernels_common import create_llvm_ptr
from .tensor_shim import _run_compiled, buf_copy_load, ptr_buf_tensor

WAVE_SIZE = 64
_INT_MIN = -(2**31)
_INT_MAX = 2**31 - 1
# Top-B candidate keys: -1 = not a candidate, -2 = already selected.
_KEY_NOT_CANDIDATE = -1
_KEY_SELECTED = -2


def _lds_load(ptr, idx):
    return fx.ptr_load(ptr + fx.Int64(idx))


def _lds_store(ptr, val, idx):
    fx.ptr_store(val, ptr + fx.Int64(idx))


def _lds_atomic_add(base_i64, idx, val):
    """Workgroup-scope ``atomicrmw add`` on an i32 LDS slot."""
    ptr = create_llvm_ptr(base_i64 + fx.Int64(idx) * 4, address_space=3)
    raw = val.ir_value() if hasattr(val, "ir_value") else val
    return llvm.AtomicRMWOp(
        llvm.AtomicBinOp.add,
        ptr,
        raw,
        llvm.AtomicOrdering.monotonic,
        syncscope="workgroup",
        alignment=4,
    ).result


def _wave_argmax(val, idx, lane):
    """All-lanes argmax over one wave; ties go to the lower index."""
    for stride in (1, 2, 4, 8, 16, 32):
        peer = ((lane + fx.Int32(stride)) & fx.Int32(WAVE_SIZE - 1)) * 4
        o_val = fx.Int32(fly_rocdl.ds_bpermute(T.i32, peer, val))
        o_idx = fx.Int32(fly_rocdl.ds_bpermute(T.i32, peer, idx))
        take = (o_val > val) | ((o_val == val) & (o_idx < idx))
        val = take.select(o_val, val)
        idx = take.select(o_idx, idx)
    return val, idx


def moonep_lds_fields(*, npes: int, experts: int, slots: int) -> dict:
    """Element counts of the int32 LDS arrays ``emit_moonep_placement`` uses."""

    epn = experts // npes
    rem_stride = ((epn + WAVE_SIZE - 1) // WAVE_SIZE) * WAVE_SIZE
    return {
        "mp_alloc": experts * npes,
        "mp_key": npes * (experts + 1),
        "mp_ecount": experts,
        "mp_rem": npes * rem_stride,
        "mp_quota": npes * npes,
        "mp_bal": npes,
        "mp_etc": npes * slots,
        "mp_target": npes * slots,
    }


@flyc.jit
def _resolve_quota_round(
    p_alloc,
    p_rem,
    p_quota,
    home,
    rem_base,
    lane,
    *,
    npes,
    experts_per_rank,
    lanes_per_home,
):
    """One greedy round for one home wave: largest quota, largest remaining expert."""
    R = npes
    epn = experts_per_rank
    q_in_range = lane < fx.Int32(R)
    q_val = q_in_range.select(
        _lds_load(p_quota, home * fx.Int32(R) + q_in_range.select(lane, fx.Int32(0))),
        fx.Int32(_INT_MIN),
    )
    q_idx = q_in_range.select(lane, fx.Int32(1 << 30))
    quota, dest = _wave_argmax(q_val, q_idx, lane)
    best_v = fx.Int32(_INT_MIN)
    best_i = fx.Int32(1 << 30)
    for c in range_constexpr(lanes_per_home):
        slot = fx.Int32(c * WAVE_SIZE) + lane
        v = _lds_load(p_rem, rem_base + slot)
        take = (v > best_v) | ((v == best_v) & (slot < best_i))
        best_v = take.select(v, best_v)
        best_i = take.select(slot, best_i)
    remaining, local_e = _wave_argmax(best_v, best_i, lane)
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
            p_rem, _lds_load(p_rem, rem_base + local_e) - move, rem_base + local_e
        )
        q_slot = home * fx.Int32(R) + dest
        _lds_store(p_quota, _lds_load(p_quota, q_slot) - move, q_slot)


@flyc.jit
def emit_moonep_placement(
    addr_count,
    addr_alloc_cumsum,
    addr_expert_to_slot,
    addr_slot_held,
    addr_slot_prev,
    addr_slot_placed,
    p_alloc,
    p_key,
    p_ecount,
    p_rem,
    p_quota,
    p_bal,
    p_etc,
    p_target,
    *,
    num_waves,
    npes,
    experts,
    slots,
    count_stride,
):
    """Place every expert's routes on destinations; all CTA threads call this.

    ``addr_count`` is the int32 ``[npes, count_stride]`` gathered histogram;
    its first ``experts`` columns are the logical route counts.  Writes
    ``alloc_cumsum[e, d]`` (routes of ``e`` on destinations ``<= d``) and
    ``expert_to_slot[d, e]``, and the ``[npes, slots]`` slot tables: ``prev``
    is what the slots held before this launch, ``placed`` what must be copied
    in (-1 for nothing), ``held`` what they hold afterwards.
    """

    R = npes
    E = experts
    B = slots
    epn = E // R
    block = num_waves * WAVE_SIZE
    lpl = (epn + WAVE_SIZE - 1) // WAVE_SIZE
    rem_stride = lpl * WAVE_SIZE
    epl = (E + WAVE_SIZE - 1) // WAVE_SIZE
    # Prepare keeps 8 waves whatever the EP size, like the native offset
    # derivation: wave d owns rank d in the per-rank phases, waves >= R only
    # take part in the CTA-wide work and the barriers.
    assert R <= num_waves, "one wave per destination rank"
    assert B <= WAVE_SIZE

    count = ptr_buf_tensor(addr_count, fx.Int32)
    alloc_cumsum = ptr_buf_tensor(addr_alloc_cumsum, fx.Int32)
    expert_to_slot = ptr_buf_tensor(addr_expert_to_slot, fx.Int32)
    slot_held = ptr_buf_tensor(addr_slot_held, fx.Int32)
    slot_prev = ptr_buf_tensor(addr_slot_prev, fx.Int32)
    slot_placed = ptr_buf_tensor(addr_slot_placed, fx.Int32)
    alloc_base = fx.Int64(ptrtoint(p_alloc))

    tid = fx.Int32(fx.thread_idx.x)
    lane = tid & fx.Int32(WAVE_SIZE - 1)
    wave = tid >> fx.Int32(6)

    for i in range(tid, fx.Int32(R * R), block):
        _lds_store(p_quota, fx.Int32(0), i)
    for e in range(tid, fx.Int32(E), block):
        total = fx.Int32(0)
        for r in range_constexpr(R):
            total = total + buf_copy_load(
                count, fx.Int32(r * count_stride) + e, fx.Int32, cache_modifier=2
            )
        _lds_store(p_ecount, total, e)
    fx.barrier()

    if tid < fx.Int32(R):
        group_total = fx.Int32(0)
        for j in range(fx.Int32(0), fx.Int32(epn), 1):
            group_total = group_total + _lds_load(p_ecount, tid * fx.Int32(epn) + j)
        routes_total = fx.Int32(0)
        for ej in range(fx.Int32(0), fx.Int32(E), 1):
            routes_total = routes_total + _lds_load(p_ecount, ej)
        rank_target = routes_total // fx.Int32(R) + (
            tid < routes_total % fx.Int32(R)
        ).select(fx.Int32(1), fx.Int32(0))
        _lds_store(p_bal, group_total - rank_target, tid)
    for i in range(tid, fx.Int32(E * R), block):
        e = i // fx.Int32(R)
        d = i - e * fx.Int32(R)
        home = e // fx.Int32(epn)
        _lds_store(p_alloc, (d == home).select(_lds_load(p_ecount, e), fx.Int32(0)), i)
    fx.barrier()

    # Receiver quotas: most overloaded home to the roomiest destination.
    if tid == fx.Int32(0):
        for _round in range(fx.Int32(0), fx.Int32(R), 1):
            best_h = fx.Int32(0)
            best_v = fx.Int32(_INT_MIN)
            worst_u = fx.Int32(0)
            worst_v = fx.Int32(_INT_MAX)
            for j in range_constexpr(R):
                v = _lds_load(p_bal, fx.Int32(j))
                hi = v > best_v
                best_v = hi.select(v, best_v)
                best_h = hi.select(fx.Int32(j), best_h)
                lo = v < worst_v
                worst_v = lo.select(v, worst_v)
                worst_u = lo.select(fx.Int32(j), worst_u)
            active = best_v > fx.Int32(0)
            move = active.select(fx.Int32(0) - worst_v, fx.Int32(0))
            q_slot = best_h * fx.Int32(R) + worst_u
            _lds_store(p_quota, active.select(move, _lds_load(p_quota, q_slot)), q_slot)
            _lds_store(p_bal, best_v - move, best_h)
            _lds_store(p_bal, active.select(fx.Int32(0), worst_v), worst_u)
    fx.barrier()

    # Wave h resolves home h's quotas into per-expert moves.  Waves without
    # a rank read rank 0's rows and store nothing.
    rank_wave = wave < fx.Int32(R)
    owned = rank_wave.select(wave, fx.Int32(0))
    home = owned
    rem_base = home * fx.Int32(rem_stride)
    for c in range_constexpr(lpl):
        local_e = fx.Int32(c * WAVE_SIZE) + lane
        in_range = local_e < fx.Int32(epn)
        safe_e = in_range.select(local_e, fx.Int32(0))
        v = _lds_load(p_ecount, home * fx.Int32(epn) + safe_e)
        if rank_wave:
            _lds_store(
                p_rem,
                in_range.select(v, fx.Int32(_INT_MIN)),
                rem_base + fx.Int32(c * WAVE_SIZE) + lane,
            )
    for _round in range(fx.Int32(0), fx.Int32(epn + R), 1):
        if rank_wave:
            _resolve_quota_round(
                p_alloc,
                p_rem,
                p_quota,
                home,
                rem_base,
                lane,
                npes=R,
                experts_per_rank=epn,
                lanes_per_home=lpl,
            )
        fx.barrier()
    fx.barrier()

    # Prefetch candidates are remote experts with a non-zero allocation.
    for i in range(tid, fx.Int32(R * E), block):
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
    for d in range(tid, fx.Int32(R), block):
        _lds_store(
            p_key, fx.Int32(_KEY_NOT_CANDIDATE), d * fx.Int32(E + 1) + fx.Int32(E)
        )
    for i in range(tid, fx.Int32(R * B), block):
        _lds_store(p_etc, fx.Int32(-1), i)
    fx.barrier()

    # Top-B by (alloc, expert) descending, wave d owns destination d.
    dest_rank = owned
    key_base = dest_rank * fx.Int32(E + 1)
    cand_vals = []
    cand_ids = []
    for c in range_constexpr(epl):
        e = fx.Int32(c * WAVE_SIZE) + lane
        in_range = e < fx.Int32(E)
        v = _lds_load(p_key, key_base + in_range.select(e, fx.Int32(E)))
        cand_vals.append(in_range.select(v, fx.Int32(_KEY_NOT_CANDIDATE)))
        cand_ids.append(in_range.select(e, fx.Int32(-1)))
    ranks = [fx.Int32(0) for _ in range(epl)]
    for other in range(fx.Int32(0), fx.Int32(E), 1):
        a = _lds_load(p_key, key_base + other)
        for c in range_constexpr(epl):
            outranks = (a > cand_vals[c]) | (
                (a == cand_vals[c]) & (other > cand_ids[c])
            )
            ranks[c] = ranks[c] + outranks.select(fx.Int32(1), fx.Int32(0))
    for c in range_constexpr(epl):
        selected = (cand_vals[c] > fx.Int32(0)) & (ranks[c] < fx.Int32(B))
        if selected & rank_wave:
            _lds_store(p_etc, cand_ids[c], dest_rank * fx.Int32(B) + ranks[c])
            _lds_store(p_key, fx.Int32(_KEY_SELECTED), key_base + cand_ids[c])
    fx.barrier()

    # Remote work beyond the top-B has no weight slot: return it to the owner.
    for i in range(tid, fx.Int32(R * E), block):
        dest = i // fx.Int32(E)
        expert = i - dest * fx.Int32(E)
        owner = expert // fx.Int32(epn)
        prefetched = _lds_load(p_key, dest * fx.Int32(E + 1) + expert) == fx.Int32(
            _KEY_SELECTED
        )
        alloc_slot = expert * fx.Int32(R) + dest
        moved = _lds_load(p_alloc, alloc_slot)
        if (owner != dest) & (~prefetched) & (moved > fx.Int32(0)):
            _lds_store(p_alloc, fx.Int32(0), alloc_slot)
            _lds_atomic_add(alloc_base, expert * fx.Int32(R) + owner, moved)

    fx.barrier()

    # Sticky settle, wave d settles destination d (every lane computes, lane
    # 0 stores): a wanted expert the row already holds keeps its slot; the
    # others fill the free slots in order.
    row = owned * fx.Int32(B)
    want = []
    held = []
    for s in range_constexpr(B):
        want.append(_lds_load(p_etc, row + fx.Int32(s)))
        held.append(slot_held[row + fx.Int32(s)])
    target = []
    # One bit per slot; B may reach a full wave (64).
    taken = fx.Int64(0)
    for s in range_constexpr(B):
        hit = fx.Int32(-1)
        for t in range_constexpr(B):
            t_rev = B - 1 - t
            hit = ((want[s] >= fx.Int32(0)) & (held[t_rev] == want[s])).select(
                fx.Int32(t_rev), hit
            )
        target.append(hit)
        safe_hit = (hit >= fx.Int32(0)).select(hit, fx.Int32(0))
        taken = (hit >= fx.Int32(0)).select(
            taken | (fx.Int64(1) << fx.Int64(safe_hit)), taken
        )
    for s in range_constexpr(B):
        free = fx.Int32(0)
        for t in range_constexpr(B):
            t_rev = B - 1 - t
            free = (((taken >> fx.Int64(t_rev)) & fx.Int64(1)) == fx.Int64(0)).select(
                fx.Int32(t_rev), free
            )
        mover = (want[s] >= fx.Int32(0)) & (target[s] < fx.Int32(0))
        target[s] = mover.select(free, target[s])
        taken = mover.select(taken | (fx.Int64(1) << fx.Int64(free)), taken)
    placed = []
    for t in range_constexpr(B):
        chosen = fx.Int32(-1)
        for s in range_constexpr(B):
            chosen = (target[s] == fx.Int32(t)).select(want[s], chosen)
        placed.append(chosen)
    fx.barrier()
    for t in range_constexpr(B):
        if (lane == fx.Int32(0)) & rank_wave:
            slot_prev[row + fx.Int32(t)] = held[t]
            slot_placed[row + fx.Int32(t)] = placed[t]
            slot_held[row + fx.Int32(t)] = (placed[t] >= fx.Int32(0)).select(
                placed[t], held[t]
            )
            _lds_store(p_target, target[t], row + fx.Int32(t))
    fx.rocdl.s_waitcnt(0)
    fx.barrier()

    for e in range(tid, fx.Int32(E), block):
        acc = fx.Int32(0)
        for d in range_constexpr(R):
            acc = acc + _lds_load(p_alloc, e * fx.Int32(R) + fx.Int32(d))
            alloc_cumsum[e * fx.Int32(R) + fx.Int32(d)] = acc
    for i in range(tid, fx.Int32(E * R), block):
        expert = i // fx.Int32(R)
        dest = i - expert * fx.Int32(R)
        owner = expert // fx.Int32(epn)
        slot = (owner == dest).select(expert - dest * fx.Int32(epn), fx.Int32(-1))
        for s in range_constexpr(B):
            is_chosen = _lds_load(p_etc, dest * fx.Int32(B) + fx.Int32(s)) == expert
            slot = is_chosen.select(
                fx.Int32(epn) + _lds_load(p_target, dest * fx.Int32(B) + fx.Int32(s)),
                slot,
            )
        expert_to_slot[dest * fx.Int32(E) + expert] = slot
    fx.rocdl.s_waitcnt(0)
    fx.barrier()


_BLOCK = 256


def make_moonep_plan_jit(
    *, rank: int, npes: int, experts: int, slots: int, grid: int, num_waves: int = 8
):
    """``(count, place, rewrite)`` launchers for one rank of one EP layout.

    ``count`` numbers each route within its expert on this rank, ``place``
    exchanges the per-rank histograms and runs ``emit_moonep_placement`` on
    the gathered matrix, ``rewrite`` turns every logical id into the virtual
    id ``dest * (EPR + B) + slot`` of the replica its route was given.
    """

    R, E, B = npes, experts, slots
    epn = E // R
    vs = epn + B
    block = num_waves * WAVE_SIZE
    sizes = moonep_lds_fields(npes=R, experts=E, slots=B)
    n_alloc, n_key, n_ecount, n_rem, n_quota, n_bal, n_etc, n_target = (
        sizes[f"mp_{k}"]
        for k in ("alloc", "key", "ecount", "rem", "quota", "bal", "etc", "target")
    )

    @fx.struct
    class PlacementLds:
        alloc: fx.Array[fx.Int32, n_alloc, 16]
        key: fx.Array[fx.Int32, n_key, 16]
        ecount: fx.Array[fx.Int32, n_ecount, 16]
        rem: fx.Array[fx.Int32, n_rem, 16]
        quota: fx.Array[fx.Int32, n_quota, 16]
        bal: fx.Array[fx.Int32, n_bal, 16]
        etc: fx.Array[fx.Int32, n_etc, 16]
        target: fx.Array[fx.Int32, n_target, 16]

    tag = f"r{rank}_n{R}_e{E}_b{B}_g{grid}"

    @flyc.kernel(name=f"moonep_plan_count_{tag}", known_block_size=[_BLOCK, 1, 1])
    def count_kernel(
        addr_ids: fx.Int64,  # INT32 [routes] logical ids, -1 = no route
        routes: fx.Int32,
        addr_hist: fx.Int64,  # INT32 [E] this rank's routes per expert
        addr_local_index: fx.Int64,  # INT32 [routes] index within the expert
    ):
        ids = ptr_buf_tensor(addr_ids, fx.Int32)
        local_index = ptr_buf_tensor(addr_local_index, fx.Int32)
        gid = fx.Int32(fx.block_idx.x) * fx.Int32(_BLOCK) + fx.Int32(fx.thread_idx.x)
        for route in range(gid, routes, grid * _BLOCK):
            expert = ids[route]
            if expert >= fx.Int32(0):
                local_index[route] = fx.Int32(
                    comm_ops.atomic_add_agent(
                        addr_hist + fx.Int64(expert) * fx.Int64(4), fx.Int32(1)
                    )
                )

    @flyc.kernel(name=f"moonep_plan_place_{tag}", known_block_size=[block, 1, 1])
    def place_kernel(
        addr_hist: fx.Int64,  # INT32 [E], zeroed again for the next launch
        addr_peer_counts: fx.Int64,  # INT64 [R] every rank's count matrix
        addr_counts: fx.Int64,  # INT32 [2, R, E] this rank's, symmetric
        addr_peer_arrive: fx.Int64,  # INT64 [R] every rank's arrival flags
        addr_arrive: fx.Int64,  # INT32 [2, R] this rank's, symmetric
        addr_epoch: fx.Int64,  # INT32 [1] launches so far
        addr_alloc_cumsum: fx.Int64,
        addr_expert_to_slot: fx.Int64,
        addr_source_prefix: fx.Int64,  # INT32 [E] lower ranks' routes per expert
        addr_slot_held: fx.Int64,
        addr_slot_prev: fx.Int64,
        addr_slot_placed: fx.Int64,
    ):
        lds = fx.SharedAllocator().allocate(PlacementLds).peek()
        tid = fx.Int32(fx.thread_idx.x)
        lane = tid & fx.Int32(WAVE_SIZE - 1)
        warp = tid >> fx.Int32(6)
        hist = ptr_buf_tensor(addr_hist, fx.Int32)
        epoch = ptr_buf_tensor(addr_epoch, fx.Int32)[fx.Int32(0)] + fx.Int32(1)
        # Two matrices alternate, so a rank one launch ahead never overwrites
        # the one a slower peer is still placing from.
        parity = epoch & fx.Int32(1)

        peer_counts = ptr_buf_tensor(addr_peer_counts, fx.Int64)
        for destination in range_constexpr(R):
            remote = ptr_buf_tensor(peer_counts[fx.Int32(destination)], fx.Int32)
            for e in range(tid, fx.Int32(E), block):
                remote[parity * fx.Int32(R * E) + fx.Int32(rank * E) + e] = hist[e]
        for e in range(tid, fx.Int32(E), block):
            hist[e] = fx.Int32(0)
        fx.rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0, expcnt=0)
        fx.barrier()

        if warp == fx.Int32(0):
            comm_ops.fence_system_release()
            peer_arrive = ptr_buf_tensor(addr_peer_arrive, fx.Int64)
            for destination in range(lane, fx.Int32(R), WAVE_SIZE):
                comm_ops.store_i32_system(
                    peer_arrive[destination],
                    parity * fx.Int32(R) + fx.Int32(rank),
                    epoch,
                )
            for source in range(lane, fx.Int32(R), WAVE_SIZE):
                comm_ops.wait_i32_until_equals(
                    addr_arrive + fx.Int64(parity * fx.Int32(R) + source) * fx.Int64(4),
                    epoch,
                )
        fx.rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0, expcnt=0)
        fx.barrier()
        comm_ops.fence_system_acquire()

        addr_matrix = addr_counts + fx.Int64(parity) * fx.Int64(R * E * 4)
        emit_moonep_placement(
            addr_matrix,
            addr_alloc_cumsum,
            addr_expert_to_slot,
            addr_slot_held,
            addr_slot_prev,
            addr_slot_placed,
            lds.alloc.ptr,
            lds.key.ptr,
            lds.ecount.ptr,
            lds.rem.ptr,
            lds.quota.ptr,
            lds.bal.ptr,
            lds.etc.ptr,
            lds.target.ptr,
            num_waves=num_waves,
            npes=R,
            experts=E,
            slots=B,
            count_stride=E,
        )
        matrix = ptr_buf_tensor(addr_matrix, fx.Int32)
        source_prefix = ptr_buf_tensor(addr_source_prefix, fx.Int32)
        for e in range(tid, fx.Int32(E), block):
            below = fx.Int32(0)
            for source in range_constexpr(rank):
                below = below + buf_copy_load(
                    matrix, fx.Int32(source * E) + e, fx.Int32, cache_modifier=2
                )
            source_prefix[e] = below
        if tid == fx.Int32(0):
            ptr_buf_tensor(addr_epoch, fx.Int32)[fx.Int32(0)] = epoch
        fx.rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0, expcnt=0)

    @flyc.kernel(name=f"moonep_plan_rewrite_{tag}", known_block_size=[_BLOCK, 1, 1])
    def rewrite_kernel(
        addr_ids: fx.Int64,
        routes: fx.Int32,
        addr_local_index: fx.Int64,
        addr_source_prefix: fx.Int64,
        addr_alloc_cumsum: fx.Int64,
        addr_expert_to_slot: fx.Int64,
        addr_virtual_ids: fx.Int64,  # INT32 [routes], -1 stays -1
    ):
        ids = ptr_buf_tensor(addr_ids, fx.Int32)
        local_index = ptr_buf_tensor(addr_local_index, fx.Int32)
        source_prefix = ptr_buf_tensor(addr_source_prefix, fx.Int32)
        alloc_cumsum = ptr_buf_tensor(addr_alloc_cumsum, fx.Int32)
        expert_to_slot = ptr_buf_tensor(addr_expert_to_slot, fx.Int32)
        virtual_ids = ptr_buf_tensor(addr_virtual_ids, fx.Int32)
        gid = fx.Int32(fx.block_idx.x) * fx.Int32(_BLOCK) + fx.Int32(fx.thread_idx.x)
        for route in range(gid, routes, grid * _BLOCK):
            expert = ids[route]
            valid = expert >= fx.Int32(0)
            safe = valid.select(expert, fx.Int32(0))
            # Expert e's routes are ranked source-major; destination d takes
            # the ranks [alloc_cumsum[e, d-1], alloc_cumsum[e, d]).
            rank_in_expert = source_prefix[safe] + valid.select(
                local_index[route], fx.Int32(0)
            )
            destination = fx.Int32(0)
            for d in range_constexpr(R - 1):
                passed = (
                    rank_in_expert >= alloc_cumsum[safe * fx.Int32(R) + fx.Int32(d)]
                )
                destination = destination + passed.select(fx.Int32(1), fx.Int32(0))
            slot = expert_to_slot[destination * fx.Int32(E) + safe]
            virtual_ids[route] = valid.select(destination * fx.Int32(vs) + slot, expert)

    @flyc.jit
    def count(
        addr_ids: fx.Int64,
        routes: fx.Int32,
        addr_hist: fx.Int64,
        addr_local_index: fx.Int64,
        stream: Stream = Stream(None),  # noqa: B008
    ):
        count_kernel(addr_ids, routes, addr_hist, addr_local_index).launch(
            grid=(grid, 1, 1), block=(_BLOCK, 1, 1), stream=stream
        )

    @flyc.jit
    def place(
        addr_hist: fx.Int64,
        addr_peer_counts: fx.Int64,
        addr_counts: fx.Int64,
        addr_peer_arrive: fx.Int64,
        addr_arrive: fx.Int64,
        addr_epoch: fx.Int64,
        addr_alloc_cumsum: fx.Int64,
        addr_expert_to_slot: fx.Int64,
        addr_source_prefix: fx.Int64,
        addr_slot_held: fx.Int64,
        addr_slot_prev: fx.Int64,
        addr_slot_placed: fx.Int64,
        stream: Stream = Stream(None),  # noqa: B008
    ):
        place_kernel(
            addr_hist,
            addr_peer_counts,
            addr_counts,
            addr_peer_arrive,
            addr_arrive,
            addr_epoch,
            addr_alloc_cumsum,
            addr_expert_to_slot,
            addr_source_prefix,
            addr_slot_held,
            addr_slot_prev,
            addr_slot_placed,
        ).launch(grid=(1, 1, 1), block=(block, 1, 1), stream=stream)

    @flyc.jit
    def rewrite(
        addr_ids: fx.Int64,
        routes: fx.Int32,
        addr_local_index: fx.Int64,
        addr_source_prefix: fx.Int64,
        addr_alloc_cumsum: fx.Int64,
        addr_expert_to_slot: fx.Int64,
        addr_virtual_ids: fx.Int64,
        stream: Stream = Stream(None),  # noqa: B008
    ):
        rewrite_kernel(
            addr_ids,
            routes,
            addr_local_index,
            addr_source_prefix,
            addr_alloc_cumsum,
            addr_expert_to_slot,
            addr_virtual_ids,
        ).launch(grid=(grid, 1, 1), block=(_BLOCK, 1, 1), stream=stream)

    return count, place, rewrite


class MoonEPPlanner:
    """Places one EP group's logical routes and rewrites them to virtual ids.

    The output feeds an ordinary MegaMoEV2 built over ``R * (EPR + B)``
    experts whose weight windows are a ``MoonEPWeightPool``'s: expert ``k`` of
    rank ``d``'s window is virtual id ``d * (EPR + B) + k``.  One planner
    serves every layer; each layer keeps its own slot tables
    (``new_slot_state``), since a slot holds that layer's expert.
    """

    def __init__(
        self,
        *,
        rank: int,
        world_size: int,
        experts: int,
        slots: int,
        max_routes: int,
    ) -> None:
        """``rank``/``world_size`` are the EP rank and size, also the mori PEs."""
        import mori.shmem as ms
        from mori.shmem import mori_shmem_create_tensor

        if world_size not in (4, 8):
            raise ValueError(f"MoonEP supports EP4 and EP8, got EP{world_size}")
        if experts % world_size:
            raise ValueError("experts must divide evenly over the ranks")
        if not 0 < slots <= WAVE_SIZE:
            raise ValueError(f"slots must be in [1, {WAVE_SIZE}], got {slots}")
        self.rank, self.world_size = rank, world_size
        self.experts, self.slots = experts, slots
        self.experts_per_rank = experts // world_size
        self.max_routes = max_routes
        self.device = torch.device("cuda", torch.cuda.current_device())

        R, E, dev = world_size, experts, self.device

        def i32(n, fill=0):
            return torch.full((n,), fill, dtype=torch.int32, device=dev)

        self._counts = mori_shmem_create_tensor((2 * R * E,), torch.int32)
        self._arrive = mori_shmem_create_tensor((2 * R,), torch.int32)
        self._counts.zero_()
        self._arrive.zero_()
        self._peer_counts = torch.tensor(
            [ms.shmem_ptr_p2p(self._counts.data_ptr(), rank, pe) for pe in range(R)],
            dtype=torch.int64,
            device=dev,
        )
        self._peer_arrive = torch.tensor(
            [ms.shmem_ptr_p2p(self._arrive.data_ptr(), rank, pe) for pe in range(R)],
            dtype=torch.int64,
            device=dev,
        )
        self._hist = i32(E)
        self._epoch = i32(1)
        self._local_index = i32(max_routes)
        self._source_prefix = i32(E)
        self._alloc_cumsum = i32(E * R)
        self._expert_to_slot = i32(R * E)
        self._virtual_ids = i32(max_routes)
        torch.cuda.synchronize(dev)
        ms.shmem_barrier_all()

        grid = max(1, min(304, -(-max_routes // _BLOCK)))
        self._jit = make_moonep_plan_jit(
            rank=rank, npes=R, experts=E, slots=slots, grid=grid
        )

    def new_slot_state(self) -> dict:
        """``held``/``prev``/``placed`` int32 ``[R, B]`` tables for one layer."""
        return {
            name: torch.full(
                (self.world_size, self.slots), -1, dtype=torch.int32, device=self.device
            )
            for name in ("held", "prev", "placed")
        }

    def home_ids(self, topk_ids: torch.Tensor) -> torch.Tensor:
        """Virtual ids that keep every route on its owner's own copy."""
        owner = topk_ids.clamp(min=0) // self.experts_per_rank
        return topk_ids + owner * self.slots

    def plan(self, topk_ids: torch.Tensor, state: dict) -> torch.Tensor:
        """Balance ``topk_ids`` (int32, -1 = none) across the group.

        Every EP rank calls this together.  Returns the virtual ids and leaves
        ``state["placed"][d, s]`` naming the expert slot ``s`` of rank ``d``
        must receive before the experts run (-1: keep), ``state["prev"]`` what
        it held before.
        """
        if topk_ids.dtype != torch.int32 or not topk_ids.is_contiguous():
            raise ValueError("topk_ids must be contiguous int32")
        routes = topk_ids.numel()
        if routes > self.max_routes:
            raise ValueError(f"{routes} routes exceed max_routes={self.max_routes}")
        stream = fx.Stream(torch.cuda.current_stream(self.device).cuda_stream)
        count, place, rewrite = self._jit
        _run_compiled(
            count,
            fx.Int64(topk_ids.data_ptr()),
            fx.Int32(routes),
            fx.Int64(self._hist.data_ptr()),
            fx.Int64(self._local_index.data_ptr()),
            stream,
        )
        _run_compiled(
            place,
            *(
                fx.Int64(t.data_ptr())
                for t in (
                    self._hist,
                    self._peer_counts,
                    self._counts,
                    self._peer_arrive,
                    self._arrive,
                    self._epoch,
                    self._alloc_cumsum,
                    self._expert_to_slot,
                    self._source_prefix,
                    state["held"],
                    state["prev"],
                    state["placed"],
                )
            ),
            stream,
        )
        _run_compiled(
            rewrite,
            fx.Int64(topk_ids.data_ptr()),
            fx.Int32(routes),
            fx.Int64(self._local_index.data_ptr()),
            fx.Int64(self._source_prefix.data_ptr()),
            fx.Int64(self._alloc_cumsum.data_ptr()),
            fx.Int64(self._expert_to_slot.data_ptr()),
            fx.Int64(self._virtual_ids.data_ptr()),
            stream,
        )
        return self._virtual_ids[:routes].view(topk_ids.shape)


__all__ = [
    "MoonEPPlanner",
    "emit_moonep_placement",
    "make_moonep_plan_jit",
    "moonep_lds_fields",
]
