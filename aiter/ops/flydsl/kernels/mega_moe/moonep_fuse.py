# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
"""MoonEP placement inside the MegaMoE compact prepare owner CTA.

The owner already holds the all-gathered logical route histogram
``count[src][expert]`` once COUNT_DONE completes.  These helpers turn it into
a MoonEP placement and then into the virtual ``[R, R * (EPR + B)]`` count
matrix that the unchanged destination plan, offset derivation and payload
kernels consume.

The placement follows ``build_prefill_reference_plan``: balance towards the
all-rank average, keep each destination's top-B remote experts, return the
rest to their owners.  Prefetch slots are then settled stickily: an expert a
slot already holds stays there, so the weight copy can skip it.

Every rank runs the same deterministic code on the same matrix, so the
placement and slot tables are identical on all ranks without communication.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import llvm
from flydsl.expr import const_expr, ptrtoint, range_constexpr
from flydsl.expr import rocdl as fly_rocdl
from flydsl.expr.typing import T

from ..kernels_common import create_llvm_ptr
from ..tensor_shim import buf_copy_load, ptr_buf_tensor

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
def _emit_home_placement(
    addr_count,
    addr_alloc_cumsum,
    addr_expert_to_slot,
    addr_slot_held,
    addr_slot_prev,
    addr_slot_placed,
    *,
    num_waves,
    npes,
    experts,
    slots,
    count_stride,
):
    """Every expert stays on its owner; the slots keep what they hold."""

    R = npes
    E = experts
    epn = E // R
    block = num_waves * WAVE_SIZE
    count = ptr_buf_tensor(addr_count, fx.Int32)
    alloc_cumsum = ptr_buf_tensor(addr_alloc_cumsum, fx.Int32)
    expert_to_slot = ptr_buf_tensor(addr_expert_to_slot, fx.Int32)
    slot_held = ptr_buf_tensor(addr_slot_held, fx.Int32)
    slot_prev = ptr_buf_tensor(addr_slot_prev, fx.Int32)
    slot_placed = ptr_buf_tensor(addr_slot_placed, fx.Int32)
    tid = fx.Int32(fx.thread_idx.x)

    for e in range(tid, fx.Int32(E), block):
        total = fx.Int32(0)
        for r in range_constexpr(R):
            total = total + buf_copy_load(
                count, fx.Int32(r * count_stride) + e, fx.Int32, cache_modifier=2
            )
        home = e // fx.Int32(epn)
        for d in range_constexpr(R):
            alloc_cumsum[e * fx.Int32(R) + fx.Int32(d)] = (
                fx.Int32(d) >= home
            ).select(total, fx.Int32(0))
    for i in range(tid, fx.Int32(R * E), block):
        dest = i // fx.Int32(E)
        expert = i - dest * fx.Int32(E)
        expert_to_slot[i] = ((expert // fx.Int32(epn)) == dest).select(
            expert - dest * fx.Int32(epn), fx.Int32(-1)
        )
    for i in range(tid, fx.Int32(R * slots), block):
        slot_prev[i] = slot_held[i]
        slot_placed[i] = fx.Int32(-1)
    fx.rocdl.s_waitcnt(0)
    fx.barrier()


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
    balance,
):
    """Place every expert's routes on destinations; all CTA threads call this.

    ``addr_count`` is the int32 ``[npes, count_stride]`` gathered histogram;
    its first ``experts`` columns are the logical route counts.  Writes
    ``alloc_cumsum[e, d]`` and ``expert_to_slot[d, e]`` for the virtual count
    derivation, and the ``[npes, slots]`` slot tables: ``prev`` is what the
    slots held before this launch, ``placed`` what must be copied in (-1 for
    nothing), ``held`` what they hold afterwards.

    ``balance=False`` keeps every expert on its owner and the slots untouched.
    """

    tables = (
        addr_count, addr_alloc_cumsum, addr_expert_to_slot,
        addr_slot_held, addr_slot_prev, addr_slot_placed,
    )
    shape = dict(
        num_waves=num_waves, npes=npes, experts=experts, slots=slots,
        count_stride=count_stride,
    )
    if const_expr(balance):
        _emit_balanced_placement(
            *tables, p_alloc, p_key, p_ecount, p_rem, p_quota, p_bal, p_etc,
            p_target, **shape,
        )
    else:
        _emit_home_placement(*tables, **shape)


@flyc.jit
def _emit_balanced_placement(
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
    """Balance, keep the top-B remote experts per destination, settle slots."""

    R = npes
    E = experts
    B = slots
    epn = E // R
    block = num_waves * WAVE_SIZE
    lpl = (epn + WAVE_SIZE - 1) // WAVE_SIZE
    rem_stride = lpl * WAVE_SIZE
    epl = (E + WAVE_SIZE - 1) // WAVE_SIZE
    assert R == num_waves, "one wave per destination rank"
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
        _lds_store(
            p_bal, group_total - rank_target, tid
        )
    for i in range(tid, fx.Int32(E * R), block):
        e = i // fx.Int32(R)
        d = i - e * fx.Int32(R)
        home = e // fx.Int32(epn)
        _lds_store(
            p_alloc, (d == home).select(_lds_load(p_ecount, e), fx.Int32(0)), i
        )
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
            _lds_store(
                p_quota, active.select(move, _lds_load(p_quota, q_slot)), q_slot
            )
            _lds_store(p_bal, best_v - move, best_h)
            _lds_store(p_bal, active.select(fx.Int32(0), worst_v), worst_u)
    fx.barrier()

    # Wave h resolves home h's quotas into per-expert moves.
    home = wave
    rem_base = home * fx.Int32(rem_stride)
    for c in range_constexpr(lpl):
        local_e = fx.Int32(c * WAVE_SIZE) + lane
        in_range = local_e < fx.Int32(epn)
        safe_e = in_range.select(local_e, fx.Int32(0))
        v = _lds_load(p_ecount, home * fx.Int32(epn) + safe_e)
        _lds_store(
            p_rem,
            in_range.select(v, fx.Int32(_INT_MIN)),
            rem_base + fx.Int32(c * WAVE_SIZE) + lane,
        )
    for _round in range(fx.Int32(0), fx.Int32(epn + R), 1):
        q_in_range = lane < fx.Int32(R)
        q_val = q_in_range.select(
            _lds_load(
                p_quota, home * fx.Int32(R) + q_in_range.select(lane, fx.Int32(0))
            ),
            fx.Int32(_INT_MIN),
        )
        q_idx = q_in_range.select(lane, fx.Int32(1 << 30))
        quota, dest = _wave_argmax(q_val, q_idx, lane)
        best_v = fx.Int32(_INT_MIN)
        best_i = fx.Int32(1 << 30)
        for c in range_constexpr(lpl):
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
        _lds_store(p_key, fx.Int32(_KEY_NOT_CANDIDATE), d * fx.Int32(E + 1) + fx.Int32(E))
    for i in range(tid, fx.Int32(R * B), block):
        _lds_store(p_etc, fx.Int32(-1), i)
    fx.barrier()

    # Top-B by (alloc, expert) descending, wave d owns destination d.
    dest_rank = wave
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
            outranks = (a > cand_vals[c]) | ((a == cand_vals[c]) & (other > cand_ids[c]))
            ranks[c] = ranks[c] + outranks.select(fx.Int32(1), fx.Int32(0))
    for c in range_constexpr(epl):
        selected = (cand_vals[c] > fx.Int32(0)) & (ranks[c] < fx.Int32(B))
        if selected:
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
    row = wave * fx.Int32(B)
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
        if lane == fx.Int32(0):
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


@flyc.jit
def emit_moonep_virtual_counts(
    addr_count,
    addr_alloc_cumsum,
    addr_expert_to_slot,
    addr_logical_pair_base,
    addr_virtual_count,
    addr_virtual_hist,
    addr_virtual_pair_base,
    *,
    num_waves,
    npes,
    experts,
    slots,
    rank,
    count_stride,
    virtual_stride,
):
    """Split each logical (source, expert) route range over its destinations.

    Expert ``e``'s routes are ordered source-major; destination ``d`` takes
    the global range ``[C[e][d-1], C[e][d])``.  Source ``s`` owns
    ``[P_s, P_s + c_s)``, so its count for virtual slot ``(d, slot)`` is the
    overlap, and this rank's sub-range starts that far into its logical
    ``pair_order`` segment.  All CTA threads call this.
    """

    R = npes
    E = experts
    epn = E // R
    vs = epn + slots
    block = num_waves * WAVE_SIZE
    count = ptr_buf_tensor(addr_count, fx.Int32)
    alloc_cumsum = ptr_buf_tensor(addr_alloc_cumsum, fx.Int32)
    expert_to_slot = ptr_buf_tensor(addr_expert_to_slot, fx.Int32)
    logical_pair_base = ptr_buf_tensor(addr_logical_pair_base, fx.Int32)
    virtual_count = ptr_buf_tensor(addr_virtual_count, fx.Int32)
    virtual_hist = ptr_buf_tensor(addr_virtual_hist, fx.Int32)
    virtual_pair_base = ptr_buf_tensor(addr_virtual_pair_base, fx.Int32)
    tid = fx.Int32(fx.thread_idx.x)

    for i in range(tid, fx.Int32(R * virtual_stride), block):
        virtual_count[i] = fx.Int32(0)
    for i in range(tid, fx.Int32(virtual_stride), block):
        virtual_hist[i] = fx.Int32(0)
        virtual_pair_base[i] = fx.Int32(0)
    fx.rocdl.s_waitcnt(0)
    fx.barrier()

    for e in range(tid, fx.Int32(E), block):
        lows = []
        highs = []
        columns = []
        previous = fx.Int32(0)
        for d in range_constexpr(R):
            cumulative = alloc_cumsum[e * fx.Int32(R) + fx.Int32(d)]
            slot = expert_to_slot[fx.Int32(d * E) + e]
            lows.append(previous)
            highs.append(cumulative)
            columns.append(fx.Int32(d * vs) + (slot >= fx.Int32(0)).select(slot, fx.Int32(0)))
            previous = cumulative
        source_begin = fx.Int32(0)
        base = logical_pair_base[e]
        for s in range_constexpr(R):
            source_end = source_begin + buf_copy_load(
                count, fx.Int32(s * count_stride) + e, fx.Int32, cache_modifier=2
            )
            for d in range_constexpr(R):
                begin = (source_begin > lows[d]).select(source_begin, lows[d])
                end = (source_end < highs[d]).select(source_end, highs[d])
                if end > begin:
                    virtual_count[fx.Int32(s * virtual_stride) + columns[d]] = end - begin
                    if const_expr(s == rank):
                        virtual_hist[columns[d]] = end - begin
                        virtual_pair_base[columns[d]] = base + begin - source_begin
            source_begin = source_end
    fx.rocdl.s_waitcnt(0)
    fx.barrier()
