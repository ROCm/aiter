# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Helpers shared verbatim by the gfx1250 m32x8 FMHA prefill kernels.

``fmha_fwd_prefill_a16w16_m32x8`` (16-bit) and ``fmha_fwd_prefill_a8w8_m32x8``
(8-bit) are deliberate siblings rather than one dtype-parameterized kernel, so
most of each file is dtype-specific. What lives here is the subset that was
byte-identical in both and whose whole dependency closure is identical too --
lane/wave indexing, LDS allocation and barriers, fragment packing, the LDS->VGPR
ring, and the reduction tree.

Sharing is gated on the dependency closure, not on matching source text. Five
more functions read identically but stay put for now: ``supported_num_q_tiles``,
``supports_gqa_ratio`` and ``_pick_num_q_tiles`` reach ``_o_fits``, which probes
the O manager with a different element type per kernel, and
``_packed_tile_indices`` and ``_wg_kv_span`` reach ``_m_tile_idx``, which differs.
Sharing those as-is would bind one sibling's behaviour into the other.

The buffer managers are already shared from ``fmha_buffer_managers``.
"""

import flydsl.expr as fx
from flydsl._mlir.dialects import llvm as llvm_dialect
from flydsl.expr import gpu, rocdl
from flydsl.expr.rocdl import tdm_ops
from flydsl.expr.typing import T
from flydsl.expr.utils.arith import _to_raw as _raw

from aiter.jit.utils.chip_info import get_lds_capacity_bytes

from .fmha_buffer_managers import _ir

# m32x8 geometry. Defined here rather than in each sibling so ``_block_m`` cannot
# drift from the kernel that calls it.
NUM_WAVES = 8  # "m32x8" -- 8 waves per threadgroup
WMMA_M = 16  # query rows per WMMA tile (the "m16" in m32x8)


def _lane_id():
    """Lane index within the wave (wave32), matching opus ``lane_id()``."""
    return fx.Int32(
        rocdl.mbcnt_lo(T.i32, fx.Int32(-1).ir_value(), fx.Int32(0).ir_value())
    )


def _warp_id():
    """Wave (warp) index within the workgroup, matching opus ``waveid_in_workgroup()``."""
    return fx.Int32(rocdl.wave_id())


def _block_m(num_q_tiles_per_wave):
    return WMMA_M * num_q_tiles_per_wave * NUM_WAVES


def _alloc_lds():
    """Allocate the full per-CU LDS and return its base. Called once per kernel body,
    before the warp-type dispatch, so both ``_core_attention`` traces share the single
    SharedAllocator flydsl permits; K/V, Q and the O epilogue all carve this base."""
    smem = fx.SharedAllocator().allocate(get_lds_capacity_bytes("gfx1250"))
    return fx.Int32(fx.ptrtoint(smem.peek().ptr))


def _bare_barrier():
    """Workgroup rendezvous with NO memory fence, for where the barrier is only a
    scheduling or counting rendezvous. ``gpu.barrier()`` prepends
    ``s_wait_storecnt_dscnt 0x0``, which would retire a ring head issued just above it.
    """
    rocdl.s_barrier_signal(-1)
    rocdl.s_barrier_wait(-1)


def _copy_fence(num_tensorcnt=-1, num_asynccnt=-1, num_dscnt=None):
    """``_copy_wait``, then a barrier to publish them; ``num_dscnt`` bounds in-flight LDS
    reads across it (None = gpu.barrier()'s full drain)."""
    _copy_wait(num_tensorcnt, num_asynccnt)
    rocdl.sched_barrier(0)
    if num_dscnt is None:
        gpu.barrier()
    else:
        rocdl.s_wait_dscnt(num_dscnt)
        _bare_barrier()
    rocdl.sched_barrier(0)


def _copy_wait(num_tensorcnt=-1, num_asynccnt=-1):
    """Retire this wave's copies to the given per-counter depths; -1 skips a counter."""
    if num_tensorcnt >= 0:
        tdm_ops.tensor_wait(num_tensorcnt)
    if num_asynccnt >= 0:
        rocdl.s_wait_asynccnt(num_asynccnt)


def _concat(parts):
    """Join a fragment's ds_load results into one WMMA operand. A pure shufflevector --
    the register allocator coalesces it into adjacent registers, so it costs no VALU."""
    cur = list(parts)
    while len(cur) > 1:
        cur = [
            cur[i].shuffle(cur[i + 1], list(range(2 * cur[i].numel)))
            for i in range(0, len(cur), 2)
        ]
    return cur[0]


def _keepalive(vals):
    """Empty side-effecting inline asm: emits no instruction but counts as a USE, so the
    operands stay live (VGPRs reserved) until this point. Load-bearing for the ring — a
    slot holding no in-flight value looks dead to regalloc, which then reclaims the pair
    the wmma just read, collapsing the WAR distance to 0 (LLVM falls back to
    ``s_wait_alu depctr_va_vdst(0)`` and the VGPR saving evaporates)."""
    ops = [_ir(v) for v in vals]
    llvm_dialect.inline_asm(
        None, ops, "", ",".join("v" for _ in ops), has_side_effects=True
    )


def _epi_state(*, m_list, d_list, o_list, o_lds_warp):
    """Flatten a core's finished softmax/O state into scf.if-yieldable raw values."""
    out = []
    for q_tile_idx in range(len(m_list)):
        out += [_raw(m_list[q_tile_idx]), _raw(d_list[q_tile_idx])] + [
            _raw(v) for v in o_list[q_tile_idx]
        ]
    return out + [_raw(o_lds_warp)]


def _o_pack(o_list):
    """Fold the per-d-tile O accumulators into ONE vector value."""
    cur = list(o_list)
    while len(cur) > 1:
        width = cur[0].numel
        cur = [
            cur[i].shuffle(cur[i + 1], list(range(2 * width)))
            for i in range(0, len(cur), 2)
        ]
    return cur[0]


def _o_unpack(wide, num_tiles):
    width = wide.numel // num_tiles
    return [
        schedule_bank(
            wide.shuffle(wide, list(range(i * width, (i + 1) * width))),
            bank=1,
            reason="oaccu",
        )
        for i in range(num_tiles)
    ]


def _load_seqlen_pair(ptr_tensor, idx):
    """Load the adjacent i32s ``ptr_tensor[idx:idx + 2]`` as ``(start, end)``. The
    address is uniform, so the 64-bit load should lower to one ``s_load_b64``."""
    p = fx.get_iter(ptr_tensor)
    pair = fx.ptr_load(p + fx.Int64(idx), result_type=fx.Vector.make_type(2, fx.Int32))
    return fx.Int32(pair[0]), fx.Int32(pair[1])


def _load_sink_logit(ptr_sink, q_head_idx, num_heads_q):
    """Load this lane's per-head sink logit from the 1-D fp32 ``sink[num_heads_q]`` — one
    extra ``exp(sink)`` term in the softmax denominator, in the scaled-score domain."""
    del num_heads_q  # in-bounds by construction; no buffer bounds check needed
    return fx.Float32(fx.ptr_load(fx.get_iter(ptr_sink) + fx.Int64(q_head_idx)))


def _kv_drain_depths(num_tensorcnt, num_asynccnt):
    """The same counters a steady-state fence names, but fully drained."""
    return (0 if num_tensorcnt >= 0 else -1, 0 if num_asynccnt >= 0 else -1)


def _ring_num_prefetch(num_ld, ring, ring_lag):
    """In-flight ds_load depth (NP) of a ``_ring_drive`` with this geometry. A tile that
    fits in the ring never wraps, so nothing is refilled and the WAR lag buys nothing --
    prefetch it whole. Only a wrapping ring pays for lag."""
    if num_ld <= ring:
        return num_ld
    return ring - 2 * ring_lag


def _ring_drive(*, widths, emit, consume, ring, ring_lag, head=None, dies=None):
    """Drive a fully-unrolled LDS->VGPR ring feeding a WMMA stream.

    ``widths[i]`` is how many ds_loads fragment ``i`` is built from -- 2 where one load
    covers half a 32-byte operand (b128), 4 where it covers a quarter (8-bit V, whose
    transpose load is the half-width ``tr8_b64``). ``emit(j)`` emits ds_load ``j`` of the
    flat stream; ``consume(i, *parts)`` emits fragment ``i``'s WMMA chain. Only ``ring`` loads are live at once and
    ``NP = ring - 2*lag`` are in flight, so a refill targets slots last read ``lag``
    fragments ago -- trading pipeline depth for WAR distance at fixed VGPR cost. Refills
    are hoisted ABOVE the wmma, which is what keeps the ring from reintroducing the
    wmma->ds_load issue bubble a burst-everything prefetch avoids.

    ``head`` optionally adopts an already-issued NP-deep prefetch (the PV case hoists it
    above softmax so the LDS latency hides under the softmax VALU). A tile of at most
    ``ring`` loads degenerates to that burst form (see ``_ring_num_prefetch``), as does
    ``lag=0`` with ``ring >= 2*num_frag``.

    ``dies(i)`` optionally names values *other* than the ring pair whose last read is
    fragment ``i`` -- the B operands, which the ring does not own. Without it those
    registers fall free the instant their last wmma issues and the next refill grabs them
    at a WAR distance of one VALU, instead of the ``2*rel`` the ring pairs enjoy.
    """
    num_frag = len(widths)
    starts = [0]
    for w in widths:
        starts.append(starts[-1] + w)
    num_ld = starts[-1]
    num_prefetch = _ring_num_prefetch(num_ld, ring, ring_lag)
    ring = min(ring, num_ld)
    assert (
        num_prefetch > 0
    ), f"ring={ring} too small for lag={ring_lag} (NP={num_prefetch})"

    def _waitn(n):
        rocdl.sched_barrier(0)
        rocdl.s_wait_dscnt(max(n, 0))
        rocdl.sched_barrier(0)

    a = [None] * ring
    if head is not None:
        assert (
            len(head) == num_prefetch
        ), f"head has {len(head)} loads, expected NP={num_prefetch}"
        for i, v in enumerate(head):
            a[i] = schedule_bank(v, bank=1, soft=True, reason="ring_frag")
    else:
        for i in range(num_prefetch):
            a[i] = schedule_bank(emit(i), bank=1, soft=True, reason="ring_frag")
            rocdl.sched_barrier(0)
        rocdl.sched_barrier(0)  # pin the num_prefetch-deep burst above the stream

    issued = num_prefetch
    held = []  # held[i] = the loads fragment i read; released rel fragments later
    rel = (
        ring_lag + 1
    )  # refill is hoisted, so hold one fragment PAST the refill of that slot

    def _refill(upto):
        nonlocal issued
        while issued < min(upto, num_ld):
            a[issued % ring] = schedule_bank(
                emit(issued), bank=1, soft=True, reason="ring_frag"
            )
            rocdl.sched_barrier(0)
            issued += 1

    def _release(i):
        j = i - rel
        if j >= 0 and held[j] is not None:
            # Fenced: the release point IS the WAR distance the refill below gets
            # measured against, so the scheduler must not slide a ds_load above it
            # or the keepalive below one.
            rocdl.sched_barrier(0)
            _keepalive(held[j])  # regs read by fragment j die HERE, not at the wmma
            rocdl.sched_barrier(0)
            held[j] = None

    def _record(i, parts):
        vals = list(parts) if parts is not None else []
        if dies is not None:
            vals.extend(dies(i))
        held.append(tuple(vals) if vals else None)

    for i in range(num_frag):
        # The wait is placed before the refill, so it is measured against what is already
        # outstanding: everything issued past the end of this fragment may still be in
        # flight, and the refill below only adds younger loads.
        _waitn(issued - starts[i + 1])
        _release(i)
        refilling = issued < num_ld
        _refill(starts[i + 1] + num_prefetch)
        parts = tuple(a[(starts[i] + k) % ring] for k in range(widths[i]))
        rocdl.sched_barrier(0)
        consume(i, *parts)
        rocdl.sched_barrier(0)
        # Past the last refill nothing can clobber the slots, so holding them buys nothing.
        _record(i, parts if (ring_lag and refilling) else None)
    for p in held:
        if p is not None:
            _keepalive(p)


def _tree_reduce_multi(lists, combine3, combine2):
    """Balanced 3-way tree reduction of num_q_tiles independent lists in lockstep, one result each.
    Critical path ~ceil(log3(N)) vs N-1 for a left-fold, and op3 = nested op2 so the
    backend fuses it (v_max3_f32 for max). Layers are emitted POSITION-MAJOR across the
    lists, putting the num_q_tiles independent ops adjacent in the IR so the backend can dual-issue
    them and hide one row's cross-lane latency behind the other's work."""
    curs = [list(v) for v in lists]
    while max(len(c) for c in curs) > 1:
        nxts = [[] for _ in curs]
        idxs = [0] * len(curs)
        while any(idxs[k] < len(curs[k]) for k in range(len(curs))):
            for k in range(len(curs)):
                cur, i, n = curs[k], idxs[k], len(curs[k])
                if i >= n:
                    continue
                if n - i >= 3:
                    nxts[k].append(combine3(cur[i], cur[i + 1], cur[i + 2]))
                    idxs[k] += 3
                elif n - i == 2:
                    nxts[k].append(combine2(cur[i], cur[i + 1]))
                    idxs[k] += 2
                else:
                    nxts[k].append(cur[i])
                    idxs[k] += 1
        curs = nxts
    return [c[0] for c in curs]


def schedule_bank(value, bank=0, soft=False, reason=None):
    """gfx1250 VGPR bank hint: ask the allocator to place ``value`` in ``bank``."""
    if schedule_bank.initialized is False:
        try:  # gfx1250 VGPR bank hint; absent on older FlyDSL
            from flydsl.expr.rocdl import schedule_bank as func
        except ImportError:  # pragma: no cover
            func = None
        schedule_bank.func = func
        schedule_bank.initialized = True

    reasons = {
        "oaccu": True,
        "ring_frag": True,
        "ring_lds_addr": False,
    }
    func = schedule_bank.func
    if func and (reason not in reasons or reasons[reason]):
        try:
            return func(value, bank, soft)
        except Exception:  # noqa: BLE001 - any binding failure degrades to identity
            schedule_bank.func = None
    return value
