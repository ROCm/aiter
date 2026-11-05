# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Helpers shared by the two gfx1250 m32x8 FMHA prefill kernels.

``fmha_fwd_prefill_a16w16_m32x8`` (16-bit) and ``fmha_fwd_prefill_a8w8_m32x8``
(8-bit) are deliberate siblings rather than one dtype-parameterized kernel, so
most of each file is dtype-specific. What lives here is what was common to both:
lane/wave indexing and tile ordering, LDS allocation and barriers, fragment
packing, the LDS->VGPR ring, the reduction tree, the O staging manager and the
Q-tile fit search, plus the geometry and LDS constants they read.

Everything here was byte-identical in both siblings, or differed only in
spelling of the same value -- ``_o_fits`` probed the O manager as
``_DTYPE_MAP[DEFAULT_DTYPE]`` on one side and ``O_DTYPE`` on the other, both
``fx.BFloat16``; ``_m_tile_idx`` wrote ``- 1`` against ``- fx.Int32(1)``.
Generated code is byte-identical before and after the move.

Constants live here rather than in each sibling so a helper cannot drift from
the kernel that calls it -- ``_block_m`` reading a stale ``NUM_WAVES`` would be
the same class of bug as a rescale threshold that no longer matches its dtype.

``DTYPE_MAP`` names EVERY dtype either sibling can express, 8-bit and 16-bit.
It is a lookup table, not a permission list: each kernel validates against its
own ``SUPPORTED_DTYPES``, because gating on ``in DTYPE_MAP`` would let the
16-bit kernel accept ``"fp8"`` and build a kernel that reads its inputs at the
wrong width.

What is NOT here is the dtype-specific body of each kernel: the attention
cores, softmax, the gemms, the prologue/epilogue and the host entry points.
The buffer managers are shared separately, from ``fmha_buffer_managers``.
"""

import functools

import flydsl.expr as fx
import torch
from flydsl._mlir.dialects import llvm as llvm_dialect
from flydsl.expr import gpu, rocdl
from flydsl.expr.rocdl import tdm_ops
from flydsl.expr.typing import T
from flydsl.expr.utils.arith import _to_raw as _raw

from aiter.jit.utils.chip_info import get_cu_num, get_lds_capacity_bytes

from .fmha_buffer_managers import (
    OManager16bV1,
    OManager16bV2,
    OManager16bV3,
    _ir,
)

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


# Q/K/V element types across both siblings. The map is shared so a dtype name means the
# same thing in each; what a given kernel actually SERVES is its own SUPPORTED_DTYPES,
# never this map's keys -- gating on `in DTYPE_MAP` would let the 16-bit kernel accept
# "fp8" and build a kernel that reads its inputs at the wrong width.
DTYPE_MAP = {
    "bf16": fx.BFloat16,
    "fp16": fx.Float16,
    "fp8": fx.Float8E4M3FN,
    "bf8": fx.Float8E5M2,
}
TORCH_DTYPE_MAP = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp8": torch.float8_e4m3fn,
    "bf8": torch.float8_e5m2,
}

# LDS budget for the 12-chunk map both siblings share.
LDS_QO_BYTES = 17 * 1024
LDS_CHUNK_BYTES = 26 * 1024
O_VARIANT = "v2"
NUM_Q_TILES_CHOICES = (1, 2)

# O stages through LDS as a 2-byte element in both siblings -- a16w16's O follows its
# bf16/fp16 input, a8w8 always writes bf16 -- and the staging size depends only on the
# element width, so one probe type answers the fit question for both.
_O_PROBE_DTYPE = fx.BFloat16


def _o_manager(*, v_hdim, gqa_ratio, num_q_tiles_per_wave, elem_dtype):
    """The O epilogue's LDS staging manager, plus the clearance check both cores need.

    Two waves share a chunk at 0 and LDS_QO_BYTES, and the upper one must stop short of
    the next chunk -- that clearance is what keeps O off the V rows a dead carried ring
    head may still be reading."""
    _o_manager_cls = {"v1": OManager16bV1, "v2": OManager16bV2, "v3": OManager16bV3}[
        O_VARIANT
    ]
    o_mgr = _o_manager_cls(
        v_hdim=v_hdim,
        gqa_ratio=gqa_ratio,
        num_waves=NUM_WAVES,
        q_tiles_per_wave=num_q_tiles_per_wave,
        elem_dtype=elem_dtype,
    )
    assert LDS_QO_BYTES + o_mgr.warp_lds_size_in_byte() <= LDS_CHUNK_BYTES, (
        f"O per-wave {o_mgr.warp_lds_size_in_byte()}B does not fit above the "
        f"{LDS_QO_BYTES}B slice inside a {LDS_CHUNK_BYTES}B chunk"
    )
    return o_mgr


@functools.cache
def _o_fits(v_hdim, gqa_ratio, num_q_tiles_per_wave):
    try:
        _o_manager(
            v_hdim=v_hdim,
            gqa_ratio=gqa_ratio,
            num_q_tiles_per_wave=num_q_tiles_per_wave,
            elem_dtype=_O_PROBE_DTYPE,
        )
    except AssertionError:
        return False
    return True


def supported_num_q_tiles(gqa_ratio, v_hdim=None):
    """The num_q_tiles choices whose packed-row tile never splits a head group and
    whose O epilogue staging fits its LDS chunk at ``v_hdim`` (256 fits one tile
    only). ``v_hdim=None`` skips the LDS check.

    ``_pick_num_q_tiles`` picks only from these, so a ratio is servable as long as ONE
    choice packs it -- gqa 48 straddles at rows_per_warp 32 but fits inside a group at 16.
    """
    return tuple(
        n
        for n in sorted(NUM_Q_TILES_CHOICES)
        if (gqa_ratio % (WMMA_M * n) == 0 or (WMMA_M * n) % gqa_ratio == 0)
        and (v_hdim is None or _o_fits(v_hdim, gqa_ratio, n))
    )


def supports_gqa_ratio(gqa_ratio, v_hdim=None):
    """True when some BLOCK_M can tile this ratio; the routers gate on it."""
    return bool(supported_num_q_tiles(gqa_ratio, v_hdim))


def _pick_num_q_tiles(num_kv_heads, batch, max_seqlen_q, gqa_ratio, v_hdim):
    """Smallest Q tile whose grid still fits one workgroup wave; the largest otherwise."""
    rows = int(max_seqlen_q) * int(gqa_ratio)
    per_tile = int(num_kv_heads) * int(batch)
    NUM_WGS_PER_CU = 1  # the >256 KB LDS footprint pins occupancy at one WG per CU
    resident = get_cu_num() * NUM_WGS_PER_CU
    ok = supported_num_q_tiles(gqa_ratio, v_hdim)
    for num_tiles in ok:
        block_m = _block_m(num_tiles)
        if ((rows + block_m - 1) // block_m) * per_tile <= resident:
            return num_tiles
    return max(ok)


# Causal work grows with the M index, so dispatch the heavy tiles first.
REVERSE_M_ORDER = True


def _m_tile_idx():
    x = fx.Int32(gpu.block_id("x"))
    if REVERSE_M_ORDER:
        x = fx.Int32(gpu.grid_dim.x) - 1 - x
    return x


def _packed_tile_indices(gqa_ratio, warp_idx, lane_idx, num_q_tiles_per_wave):
    """Map this lane's rows in the packed ``(seq, q_head_in_group)`` tile to global
    indices: ``(kv_head, q_head_idx, seq_idx)``, the latter two length-num_q_tiles lists (one per
    q-WMMA-tile this wave owns, num_q_tiles = num_q_tiles_per_wave).

    block_id y is the kv_head; block_id x tiles that head's ``(seq, q_head_in_group)``
    plane, with ``q_head_in_group`` the fast axis so the ``% / //`` use the small (often
    power-of-two) ``gqa_ratio``. The num_q_tiles tiles a wave owns are contiguous.
    """
    kv_head = fx.Int32(gpu.block_id("y"))
    warp_row0 = _m_tile_idx() * _block_m(num_q_tiles_per_wave) + warp_idx * (
        num_q_tiles_per_wave * WMMA_M
    )
    q_head_idx = []
    seq_idx = []
    for q_tile_idx in range(num_q_tiles_per_wave):
        row_idx = warp_row0 + q_tile_idx * WMMA_M + lane_idx % WMMA_M
        q_head_idx.append(kv_head * gqa_ratio + row_idx % gqa_ratio)
        seq_idx.append(row_idx // gqa_ratio)
    return kv_head, q_head_idx, seq_idx


def _wg_kv_span(
    *, n_block, mask_right, gqa_ratio, num_q_tiles_per_wave, q_len, kv_len, window_right
):
    """This WG's KV band: (m_tile_idx, causal_off, kv_len_wg, num_tiles).

    A query at seq s attends [s+causal_off-window_left, s+causal_off+window_right] with
    causal_off = kv_len-q_len; mask_right clips kv_len_wg to the WG's max query's
    attend-limit so no tile runs fully past the band. num_tiles is also the one-KV-tile
    dispatch predicate, so it is computed here once and shared with the kernel entry."""
    BLOCK_M = _block_m(num_q_tiles_per_wave)
    m_tile_idx = _m_tile_idx()
    causal_off = kv_len - q_len
    if mask_right:
        wg_max_seq = (
            m_tile_idx * fx.Int32(BLOCK_M) + fx.Int32(BLOCK_M - 1)
        ) // fx.Int32(gqa_ratio)
        wg_max_seq = fx.min(wg_max_seq, q_len - fx.Int32(1))
        kv_len_wg = wg_max_seq + causal_off + window_right + fx.Int32(1)
        kv_len_wg = fx.min(kv_len_wg, kv_len)
        kv_len_wg = fx.max(kv_len_wg, fx.Int32(1))
    else:
        kv_len_wg = kv_len
    return m_tile_idx, causal_off, kv_len_wg, fx.ceildiv(kv_len_wg, fx.Int32(n_block))
