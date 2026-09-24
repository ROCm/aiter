# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""MHA Forward Prefill kernel — ``m32x8`` design, gfx1250 (MI400 / mi450).

A clean FlyDSL kernel written in the high-level layout-algebra style
(tiled copy / tiled MMA + ``SharedAllocator``).

``m32x8`` names the threadgroup shape: **8 waves per threadgroup**, each wave
owning a **32-row** Q span (2 adjacent 16-row WMMA tiles). gfx1250 runs wave32,
so a threadgroup is ``8 * 32 = 256`` threads and ``BLOCK_M = 32 * 8 = 256`` Q
rows. (The leading ``32`` is per-wave Q rows; ``16`` is the WMMA M dimension.)

Layout support — two device kernels over one shared compute core (option B):
  - ``kn_fmha_fwd_prefill_a16w16_m32x8_thd``  — varlen THD, driven by ``cu_seqlens``.
  - ``kn_fmha_fwd_prefill_a16w16_m32x8_bshd`` — batched BSHD, uniform ``seq_len`` scalar
    (no ``cu_seqlens`` tensors → nothing transient to bake into a CUDA graph).
Both resolve their per-workgroup base offsets + sequence bounds, then call the
layout-agnostic ``_core_attention`` helper.

Scope — v1 (this file is intentionally config-agnostic in its name):
  - ``qk_hdim in {64, 128, 192, 256}`` (D_qk), ``v_hdim in {64, 128}`` (D_v); they are
    independent. ``n_block`` is picked by ``pick_n_block`` (128 at qk_hdim <= 128,
    64 at 192/256)
  - dtype: bf16 for Q/K/V/O
  - grouped-query attention (GQA): ``gqa = nheads_q // nheads_k``
  - causal and non-causal

``qk_hdim``, ``v_hdim`` and the dtype are compile-time (build-time) parameters
captured by the builder closure, so they never appear in the file name and can
be generalized later without changing the runtime kernel signatures.

Target: gfx1250, wave32, 8 waves per threadgroup (256 threads).
"""

import functools
from enum import IntEnum

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl._mlir.dialects import llvm as llvm_dialect
from flydsl.expr import arith, gpu, rocdl

# Q/K/V staging managers (own their LDS swizzles + async copy schedules). They are
# self-contained: this kernel maintains its own arch constants below and passes the
# config each manager needs through its constructor.
from flydsl.expr.rocdl import tdm_ops
from flydsl.expr.typing import T
from flydsl.expr.utils.arith import _to_raw as _raw

from aiter.jit.utils.chip_info import get_cu_num, get_lds_capacity_bytes
from aiter.ops.flydsl.kernels import buffer_ops

from ..kernels_common import LOG2E, create_llvm_ptr
from ..tensor_shim import _run_compiled

# Single source of truth for gfx1250 Expert Scheduling Mode 2 (DEP_MODE=2). Lives
# in fmha_b16_buffer_managers. Under mode 2 the LLVM setreg (via the
# amdgpu-expert-scheduling-mode hint, set in _ensure_*_kernel) makes LLVM insert all
# depctr covers itself for the plain intrinsics the kernel emits.
from .fmha_b16_buffer_managers import (
    ENABLE_SCHED_MODE2,
    KManager16bV1,
    KManager16bV2,
    OManager16bV1,
    OManager16bV2,
    OManager16bV3,
    ProducerCtx,
    QManager16bV1,
    QManager16bV2,
    VManager16bV1,
    VManager16bV2,
    _ir,
)

# ============================================================================
# Threadgroup / arch constants
# ============================================================================

WAVE_SIZE = 32  # gfx1250 kernels run wave32
NUM_WAVES = 8  # "m32x8" — 8 waves per threadgroup
BLOCK_SIZE = WAVE_SIZE * NUM_WAVES  # 256 threads

# "m32x8": each wave owns WMMA_ROW_PER_WAVE adjacent 16-row (WMMA M) Q sub-tiles →
# BLOCK_M = 16 * 2 * 8 = 256 Q rows per threadgroup. Each wave's 2 tiles are
# contiguous: warp i owns rows [i*32, i*32+32) = tiles 2i, 2i+1.
WMMA_M = 16  # query rows per WMMA tile (the "m16" in m32x8)
WMMA_N = 16  # kv rows per WMMA tile (the S^T=K@Q^T output's n_block-direction axis)
WMMA_K = 32  # WMMA contraction depth (bf16 v_wmma_f32_16x16x32); d-tile width
WMMA_ROW_PER_WAVE = 2  # Q WMMA tiles per wave (the "x2" step from m16x8 to m32x8)
BLOCK_M = WMMA_M * WMMA_ROW_PER_WAVE * NUM_WAVES  # 256
# 1 => BLOCK_M 128 ("m16x8"), 2 => BLOCK_M 256. Chosen per call by _pick_num_q_tiles.
NUM_Q_TILES_CHOICES = (1, 2)
NUM_WGS_PER_CU = 1  # the >256 KB LDS footprint pins occupancy at one WG per CU
# Causal work grows with the M index, so dispatch the heavy tiles first.
REVERSE_M_ORDER = True


def _block_m(num_q_tiles_per_wave):
    return WMMA_M * num_q_tiles_per_wave * NUM_WAVES


def _m_tile_idx():
    x = fx.Int32(gpu.block_id("x"))
    if REVERSE_M_ORDER:
        x = fx.Int32(gpu.grid_dim.x) - fx.Int32(1) - x
    return x


class WarpType(IntEnum):
    """Warp-specialization role (compile-time). gfx1250 pairs wave i with wave i+4 on
    one SIMD; the low half (waves 0..3) and high half (waves 4..7) run different
    main-loop preamble orderings so one wave drives memory while its SIMD-mate computes.

    SIMD parity (wave w -> SIMD w%4) is NOT a role: the swapped KV-half read order
    carries it as a runtime term instead, so the body stays 2-way.
    """

    LO = 0
    HI = 1

    @property
    def is_lo(self):
        return self is WarpType.LO


DEFAULT_QK_HDIM = 128
DEFAULT_V_HDIM = 128
DEFAULT_DTYPE = "bf16"
_DTYPE_MAP = {"bf16": fx.BFloat16, "fp16": fx.Float16}
_TORCH_DTYPE_MAP = {"bf16": torch.bfloat16, "fp16": torch.float16}
SUPPORTED_QK_HDIM = (64, 128, 192, 256)
SUPPORTED_V_HDIM = (64, 128)

# KV sequence block (columns of one QK GEMM tile). Configurable; 64 for now.
N_BLOCK_CHOICES = (32, 64, 128, 256)
# ---- LDS chunk layout: 12 x 26 KB = 312 KB of the 320 KB budget. ----
# Chunk i sits at LDS_CHUNK_BYTES * i, and every K|V tile is MANDATORILY split 2-way
# along n_block, the two halves landing in chunks 6 apart so they fall in different
# 64 KB LDS segments (segment = base // 64 KB):
#
#   K[0][0]   0..26  seg 0      K[0][1] 156..182  seg 2
#   V[0][0]  26..52  seg 0      V[0][1] 182..208  seg 2,3
#   K[1][0]  52..78  seg 0,1    K[1][1] 208..234  seg 3
#   V[1][0]  78..104 seg 1      V[1][1] 234..260  seg 3,4
#   K[2][0] 104..130 seg 1,2    K[2][1] 260..286  seg 4
#   V[2][0] 130..156 seg 2      V[2][1] 286..312  seg 4
#
# Q and O own no LDS of their own -- they time-share KV chunks (see _q_wave_base /
# _o_wave_base). Slot pp owns [52*pp, +52) low and [156 + 52*pp, +52) high.
LDS_CHUNK_BYTES = 26 * 1024
KV_LDS_SPLITS = 2  # halves one K (or V) tile is split into along n_block
# Per-wave Q / O region inside a chunk pair. 17 KB covers 32 rows x 256 hdim + pad.
LDS_QO_BYTES = 17 * 1024

DEFAULT_N_BLOCK = 64
# Preference order for the auto-picked n_block (widest first). A wider tile only fits if
# BOTH split halves still sit in one chunk, so the 12-chunk map is untouched.
N_BLOCK_PREF = (128, 64)
# ...and only if the body still fits the register file: 128 doubles the live fragment
# state, which spills above this hdim.
N_BLOCK_WIDE_MAX_QK_HDIM = 128

# K|V LDS slots the main loop rotates through. 3 is the exact minimum for the
# software-pipelined body: it reads V(u-1) from slot 0 and K(u) from slot 1 while the
# copy for tile u+1 is written into slot 2.
N_KV_PP = 3
# K runs one tile ahead of V. Body u reads K from slot 1 and V from slot 0, so landing
# both copies in slot 2 would leave K one body of latency cover against V's two. LO
# instead writes K(u+2) into slot 0 -- a chunk whose K half died at body u-1 and whose V
# half this body reads (the 12-chunk layout keeps them disjoint) -- which buys K the same
# two bodies, for one extra prologue copy (K(start+1) into slot 2).
#
# That head start is also what lets LO's steady-state fence be partial: it reaches the
# drain with the tile it is about to read already retired and the newest still in flight,
# so it waits to a depth of one tile instead of to 0. HI, one body of slack, drains fully.

# Softmax uses the native exp2 intrinsic and S reaches it already in log2 units
# (S' = S * softmax_scale * LOG2E), so the inner loop is a plain exp2(S' - m) and m/LSE
# live in the log2 domain. LOG2E is folded into Q by the bf16 multiply the Q loader
# already does -- free, but it rounds the scale to bf16's 8 mantissa bits. The exact
# alternative is to scale the f32 QK accumulator instead, at R*NKV*8 VALU per tile.

# Deferred oaccu rescale (FAv4 innovation, hk_mla spec §9.1.1). Rescaling O by
# corr = exp(m_prev - m_new) is a full-width VALU pass every tile, but corr == 1 when the
# running max doesn't move. So keep m STALE while the tile's row max stays within
# RESCALE_THRESHOLD logit units of it: P = exp2(S - m_stale) accumulates against the
# un-rescaled oaccu/denom, staying consistent. The per-lane test is promoted to
# wave-uniform via ballot, so the caller can gate the wide multiply with one
# non-divergent scf.if. The threshold is in NATURAL logits (m is log2-domain, so the
# compare scales by LOG2E): 8.0 defers until the max would move by e^8, far under the
# e^88 fp32 overflow wall. False (or threshold < 0) always rescales.
ENABLE_DEFER_RESCALE = True
RESCALE_THRESHOLD = 8.0

# Running-max seed: finite big-negative, not -inf, so a fully-masked row keeps m finite
# and softmax's (m_prev - m_new) / fma(s, .., -m) never do -inf arithmetic (NaN).
# exp2(big_neg - real) still underflows to 0. Masked scores themselves stay -inf.
BIG_NEG = -1.0e30

# Compile-time Q/K/V loader select. False = V1 (Q ring async + swizzled LDS; K/V
# cluster_load_async), True = V2 (Q per-warp TDM; K/V TDM global->LDS, HW OOB, far fewer
# address VGPRs). O is selected separately by O_VARIANT. For K and V the two differ ONLY
# in that transport -- same padded row-major LDS, same fragment reads, V2 a subclass of
# V1 -- so this picks a class pair and no call site below branches on it.
USE_TDM_LOADER = True

# K/V producer specialization: the LO half issues every K copy, the HI half every V copy,
# each wave of a half taking one dense n_block/KV_PRODUCER_WARPS row band, so a half's
# counter tracks one operand. The drain barrier still publishes both. The partition
# reaches the managers as a ProducerCtx, so it is transport-independent.
KV_PRODUCER_WARPS = NUM_WAVES // 2

# LO/HI anti-phase (FA3 ping-pong). Both halves run the same tile stream and the same
# number of bodies and barriers; the HI half runs its two phases in the opposite order,
#   LO body u:  gemm(u)                 | BAR | softmax(u)
#   HI body u:  softmax(u-1) + rescale  | BAR | gemm(u)
# so each barrier window has one half in the WMMA stream and the other in the softmax
# VALU stream. HI carries s_acc across the back edge instead of P (and needs no carried
# ring head -- its head hides under the softmax that opens its own body). Its dead
# leading softmax(start-1) is neutralized by seeding s_acc to -inf, which is exactly the
# fully-masked path (m_new = m_prev, corr = 1, P = 0, d unchanged). The lagging half also
# puts its KV drain barrier AFTER the gemm, so the loop's back-edge bookkeeping and the
# prefetch descriptor's VALU/SALU land behind it.
# O writer variant (decoupled from USE_TDM_LOADER): "v1" swizzled LDS + buffer_store (fastest so
# far), "v2" TDM store (padding ignored -> contiguous LDS -> bank conflict, slow), "v3" padded LDS +
# global_store_async_from_lds_b128.
O_VARIANT = "v3"

# LDS->VGPR ring for the fused PV+QK WMMA stream (``_pv_qk_gemm``). Bursting every
# ds_load of the resident KV tile into VGPRs before the first wmma makes the live cost
# scale with hdim; the ring instead holds only RING loads live at 4 VGPR each, NP =
# RING - 2*LAG in flight. Spanning both operand streams lets its refills pull K loads
# while the PV wmma stream is still running. Keep RING a multiple of 4 -- an odd wrap
# flips slot parity, costing a cycle on half the wmma. LAG=0 with RING >= num_ds_loads
# reproduces the un-ringed burst.
PVQK_RING = 24
PVQK_LAG = 2
# Ring-head loads issued at the END of a body (under the softmax VALU) and carried across
# the back edge as iter_args, instead of at the top of the body that consumes them. The V
# they read is tile u, already resident in the slot this body used for K. Clamped to NP;
# 0 restores the un-carried head. Costs 4 VGPR per carried load.
PVQK_HEAD_CARRY = 20

# NOTE: the remaining tiling constants (chunk sizes, K/V write-tile + V swizzle
# granularity) live inside fmha_b16_buffer_managers.py — they are intrinsic to the
# managers' LDS layouts, so the kernel no longer declares them here.


# ============================================================================
# Small device helpers
# ============================================================================


def _warp_id():
    """Wave (warp) index within the workgroup, matching opus ``waveid_in_workgroup()``."""
    return fx.Int32(rocdl.wave_id())


def _lane_id():
    """Lane index within the wave (wave32), matching opus ``lane_id()``."""
    return fx.Int32(
        rocdl.mbcnt_lo(T.i32, fx.Int32(-1).ir_value(), fx.Int32(0).ir_value())
    )


def _kv_wait(num_tensorcnt=-1, num_asynccnt=-1):
    """Retire this wave's K/V global->LDS copies down to the given per-counter depths.

    A counter is waited only if the caller names it (default -1 emits nothing for it), so
    a mixed loader pair -- a V1 K manager on ``asynccnt``, a V2 V manager on
    ``tensorcnt`` -- can share one fence."""
    if num_tensorcnt >= 0:
        tdm_ops.tensor_wait(num_tensorcnt)
    if num_asynccnt >= 0:
        rocdl.s_wait_asynccnt(num_asynccnt)


def _kv_drain_depths(num_tensorcnt, num_asynccnt):
    """The same counters a steady-state fence names, but fully drained."""
    return (0 if num_tensorcnt >= 0 else -1, 0 if num_asynccnt >= 0 else -1)


def _bare_barrier():
    """Workgroup rendezvous with NO memory fence, for where the barrier is only a
    scheduling or counting rendezvous. ``gpu.barrier()`` prepends
    ``s_wait_storecnt_dscnt 0x0``, which would retire a ring head issued just above it.
    """
    rocdl.s_barrier_signal(-1)
    rocdl.s_barrier_wait(-1)


def _kv_fence(num_tensorcnt=-1, num_asynccnt=-1, num_dscnt=None):
    """``_kv_wait``, then publish the retired copies workgroup-wide.

    The counters only bound the issuing wave's own copies, so the ``s_barrier`` is what
    publishes a wave's share of a tile to its peers; it doubles as the WAR wall for the
    slot about to be written. ``num_dscnt`` bounds this wave's in-flight LDS reads across
    it: ``None`` takes ``gpu.barrier()``'s full drain, a depth swaps that for a bare
    signal/wait plus a partial wait so a ring head issued just above survives."""
    _kv_wait(num_tensorcnt, num_asynccnt)
    rocdl.sched_barrier(0)
    if num_dscnt is None:
        gpu.barrier()
    else:
        rocdl.s_wait_dscnt(num_dscnt)
        _bare_barrier()
    rocdl.sched_barrier(0)


def _load_seqlen_pair(ptr_tensor, idx):
    """Load the adjacent i32s ``ptr_tensor[idx:idx + 2]`` as ``(start, end)``. The
    address is uniform, so the 64-bit load should lower to one ``s_load_b64``."""
    p = fx.get_iter(ptr_tensor)
    pair = fx.ptr_load(p + fx.Int64(idx), result_type=fx.Vector.make_type(2, fx.Int32))
    return fx.Int32(pair[0]), fx.Int32(pair[1])


def _load_sink_logit(ptr_sink, q_head_idx, num_heads_q):
    """Load this lane's per-head sink logit from the 1-D fp32 ``sink[num_heads_q]`` — one
    extra ``exp(sink)`` term in the softmax denominator, in the scaled-score domain.

    Flat ``llvm.load`` rather than ``buffer_load``, which would re-scale the offset
    internally; flat keeps the address arithmetic SSA-visible for LLVM to cover under
    sched mode 2. No bounds check needed: ``q_head_idx`` is in range by construction."""
    del num_heads_q  # in-bounds by construction; no buffer bounds check needed
    sink_base_i64 = fx.Int64(fx.ptrtoint(fx.get_iter(ptr_sink)))
    byte_off = fx.Int64(q_head_idx) * fx.Int64(4)
    addr = sink_base_i64 + byte_off
    gptr = create_llvm_ptr(addr, address_space=1)
    return fx.Float32(llvm_dialect.load(T.f32, gptr))


def _packed_tile_indices(gqa_ratio, warp_idx, lane_idx, num_q_tiles_per_wave):
    """Map this lane's rows in the packed ``(seq, q_head_in_group)`` tile to global
    indices: ``(kv_head, q_head_idx, seq_idx)``, the latter two length-R lists (one per
    q-WMMA-tile this wave owns, R = num_q_tiles_per_wave).

    block_id y is the kv_head; block_id x tiles that head's ``(seq, q_head_in_group)``
    plane, with ``q_head_in_group`` the fast axis so the ``% / //`` use the small (often
    power-of-two) ``gqa_ratio``. The R tiles a wave owns are contiguous.
    """
    kv_head = fx.Int32(gpu.block_id("y"))
    warp_row0 = _m_tile_idx() * _block_m(
        num_q_tiles_per_wave
    ) + warp_idx * (num_q_tiles_per_wave * WMMA_M)
    q_head_idx = []
    seq_idx = []
    for qt in range(num_q_tiles_per_wave):
        row_idx = warp_row0 + qt * WMMA_M + lane_idx % WMMA_M
        q_head_idx.append(kv_head * gqa_ratio + row_idx % gqa_ratio)
        seq_idx.append(row_idx // gqa_ratio)
    return kv_head, q_head_idx, seq_idx


# ============================================================================
# Compute stages
# ============================================================================


def _wmma(a, b, c):
    """v_wmma_f32_16x16x32_{bf16,f16} (gfx1250, wave32): C[16x16 f32] = A[16x32] @
    B[32x16] + C, with operand reuse off. ``a``/``b`` are v16 16-bit fragments, ``c`` a
    v8 f32 accumulator; returns the raw v8 f32 result, feedable straight back as ``c``.
    """
    v8f32 = fx.Vector.make_type(8, fx.Float32)
    wmma = (
        rocdl.wmma_f32_16x16x32_f16
        if a.dtype is fx.Float16
        else rocdl.wmma_f32_16x16x32_bf16
    )
    # modC defaults to WMMACModifier::none (== the old modC=0); omit it.
    d = wmma(v8f32, _ir(a), _ir(b), _ir(c), reuseA=False, reuseB=False).result
    # Every wmma is one tick of the ring's WAR clock: the depctr the hazard pass puts
    # in front of a refill counts the VALU between the last read of those registers
    # and the ds_load that reclaims them. Let the scheduler slide a wmma across a
    # ds_load and that count stops matching the geometry the ring was built for.
    rocdl.sched_barrier(0)
    return d


def _p_to_elem(p_list, elem_dtype):
    """Narrow softmax's f32 P^T to the wmma element type. Split out of ``_softmax`` so the
    caller can place it AFTER the next gemm's ring head -- nothing in the head depends on
    P, so the ds_loads issue first and the v_cvt batch fills their shadow. Costs P a wider
    live range (f32, not narrowed) across the head."""
    return [[pv.to(elem_dtype) for pv in pt] for pt in p_list]


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


def _ring_num_prefetch(num_frag, ring, lag):
    """In-flight ds_load depth (NP) of a ``_ring_drive`` with this geometry. A tile that
    fits in the ring never wraps, so nothing is refilled and the WAR lag buys nothing --
    prefetch it whole. Only a wrapping ring pays for lag."""
    num_ld = 2 * num_frag
    if num_ld <= ring:
        return num_ld
    return ring - 2 * lag


def _emit_ld(emit, j):
    """One ring ds_load, fenced. Pairs with the fence in ``_wmma``: together they hold
    the load/wmma interleave exactly as emitted, which is what makes the refill's WAR
    distance equal the ring geometry instead of whatever the scheduler settles on."""
    v = emit(j)
    rocdl.sched_barrier(0)
    return v


def _ring_drive(*, num_frag, emit, consume, ring, lag, head=None, dies=None):
    """Drive a fully-unrolled LDS->VGPR ring feeding a WMMA stream.

    ``emit(j)`` emits ds_load ``j`` (2 per WMMA fragment); ``consume(i, lo, hi)`` emits
    fragment ``i``'s WMMA chain. Only ``ring`` loads are live at once and
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
    num_ld = 2 * num_frag
    NP = _ring_num_prefetch(num_frag, ring, lag)
    ring = min(ring, num_ld)
    assert NP > 0, f"ring={ring} too small for lag={lag} (NP={NP})"

    def _waitn(n):
        rocdl.sched_barrier(0)
        rocdl.s_wait_dscnt(max(n, 0))
        rocdl.sched_barrier(0)

    a = [None] * ring
    if head is not None:
        assert len(head) == NP, f"head has {len(head)} loads, expected NP={NP}"
        for i, v in enumerate(head):
            a[i] = v
    else:
        for i in range(NP):
            a[i] = _emit_ld(emit, i)
        rocdl.sched_barrier(0)  # pin the NP-deep burst above the stream

    steady = num_frag - NP // 2
    held = []  # held[i] = the pair fragment i read; released rel fragments later
    rel = (
        lag + 1
    )  # refill is hoisted, so hold one fragment PAST the refill of that slot

    def _refill(i):
        for k in range(2):
            j = 2 * i + NP + k
            if j < num_ld:
                a[j % ring] = _emit_ld(emit, j)

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

    def _record(i, pair):
        vals = list(pair) if pair is not None else []
        if dies is not None:
            vals.extend(dies(i))
        held.append(tuple(vals) if vals else None)

    for i in range(steady):
        _waitn(NP - 2)
        _release(i)
        _refill(i)
        lo, hi = a[(2 * i) % ring], a[(2 * i + 1) % ring]
        rocdl.sched_barrier(0)
        consume(i, lo, hi)
        rocdl.sched_barrier(0)
        _record(i, (lo, hi) if lag else None)
    for i in range(steady, num_frag):
        _waitn(NP - 2 * (i - steady + 1))
        _release(i)
        lo, hi = a[(2 * i) % ring], a[(2 * i + 1) % ring]
        rocdl.sched_barrier(0)
        consume(i, lo, hi)
        rocdl.sched_barrier(0)
        _record(i, None)
    for p in held:
        if p is not None:
            _keepalive(p)


def _tree_reduce_multi(lists, op3, op2):
    """Balanced 3-way tree reduction of R independent lists in lockstep, one result each.
    Critical path ~ceil(log3(N)) vs N-1 for a left-fold, and op3 = nested op2 so the
    backend fuses it (v_max3_f32 for max). Layers are emitted POSITION-MAJOR across the
    lists, putting the R independent ops adjacent in the IR so the backend can dual-issue
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
                    nxts[k].append(op3(cur[i], cur[i + 1], cur[i + 2]))
                    idxs[k] += 3
                elif n - i == 2:
                    nxts[k].append(op2(cur[i], cur[i + 1]))
                    idxs[k] += 2
                else:
                    nxts[k].append(cur[i])
                    idxs[k] += 1
        curs = nxts
    return [c[0] for c in curs]


def _softmax(
    *,
    s_list,
    m_prev_list,
    d_prev_list,
    lane_idx,
    n_block,
    kv_swap_delta,
    kv_pos_base=None,
    q_max_list=None,
    q_min_list=None,
    kv_len=None,
):
    """Online-softmax update for one KV tile, for ALL R q-WMMA-tiles this wave owns.

    Every ``*_list`` argument and return is length R (= num_q_tiles_per_wave). The R rows are
    independent (each owns its S, running m/d and mask bounds) but share the tile's K/V,
    so processing them together lets ``_tree_reduce_multi`` interleave their max/sum trees
    and dual-issue the combines, hiding each other's permlanex16 latency. ``s_list`` is
    already in log2 units (the loader folds softmax_scale*LOG2E into Q), so exp is exp2.

    Layout (from ``_pv_qk_gemm``): ``s_list[r]`` is ``NKV = n_block//WMMA_N`` v8-f32
    accumulators; this lane owns query ``q = warp*16 + l%16`` and, in tile ``kvt``, kv rows
    ``kvt*16 + (l//16)*8 + [0..8)``. The peer lane ``l^16`` holds the other 8-row half of
    the same q, so rows reduce locally over (kvt, i) then across ``shuffle_xor(16)``.

    ``kv_swap_delta`` corrects the swapped half-order read: when the wave takes the tile's
    two halves in the other order, slot ``kvt`` holds physical kv-tile ``kvt ^ (NKV/2)``,
    so its absolute offset moves by +/- ``n_block/2``. A runtime Int32 (0 on even SIMDs)
    folded into two per-body mask bases rather than into the NKV per-slot constants.

    Masking (per element, sequence-relative ``kv_pos = kv_pos_base + (l//16)*8 + kvt*16 +
    i``; a None bound is skipped): ``q_max_list[r]`` masks ``kv_pos > q_max`` (band upper
    edge ``q_seq + (kv_len-q_len) + window_right``, clamped to ``kv_len-1`` on the last
    tile to fold the tail); ``q_min_list[r]`` masks ``kv_pos < q_min`` (lower edge, minus
    ``window_left``); ``kv_len`` masks ``kv_pos >= kv_len`` and applies only when q_max is
    None (the standalone non-causal tail).

    Returns ``(p, m_new, d_new, corr)`` plus ``rescale_masks``. p is NKV v8 **f32** P^T =
    exp(S^T - m_new), NOT narrowed to the wmma element type -- the caller places that with
    ``_p_to_elem``. m_new is STALE (== m_prev) when that row's ballot did not fire (FAv4
    §9.1.1), with corr == 1 and d_new = corr*d_prev + rowsum(p). ``rescale_masks`` is the R
    raw ballots (None when deferral is compiled out), for the caller to fold into its one
    branch condition at the use site.
    """
    NKV = n_block // WMMA_N
    f32 = T.f32
    fast = arith.FastMathFlags.fast
    neg_inf = fx.Float32(float("-inf"))
    _defer = ENABLE_DEFER_RESCALE and RESCALE_THRESHOLD >= 0.0

    def fmax(a, b):
        return fx.Float32(arith.MaxNumFOp(_raw(a), _raw(b), fastmath=fast).result)

    def fadd(a, b):
        return fx.Float32(arith.addf(_raw(a), _raw(b), fastmath=fast))

    # fast-math WITHOUT reassoc: LLVM's Reassociate pass otherwise re-linearizes the
    # sum tree back into a serial chain (max survives — Reassociate ignores maxnum).
    _FF = arith.FastMathFlags
    _no_reassoc = _FF.nnan | _FF.ninf | _FF.nsz | _FF.arcp | _FF.contract | _FF.afn

    def fadd_t(a, b):
        return fx.Float32(arith.addf(_raw(a), _raw(b), fastmath=_no_reassoc))

    def fsub(a, b):
        return fx.Float32(arith.subf(_raw(a), _raw(b), fastmath=fast))

    def fmul(a, b):
        return fx.Float32(arith.mulf(_raw(a), _raw(b), fastmath=fast))

    def fsub_inf(a, b):  # masked s is -inf and must stay -inf: no ninf fast-math here
        return fx.Float32(arith.subf(_raw(a), _raw(b)))

    def exp2(x):
        return fx.Float32(rocdl.exp2(f32, _raw(x)))

    # permlanex16 selectors: identity cross-16 gather (nibbles 0..15) => lane l<->l^16.
    sel_lo, sel_hi = _raw(fx.Int32(0x76543210)), _raw(fx.Int32(0xFEDCBA98))

    def peer(v):  # cross-lane reduce partner: lane l <-> l^16 (the other kv half)
        return fx.Float32(
            rocdl.permlanex16(
                f32,
                _raw(v),
                _raw(v),
                sel_lo,
                sel_hi,
                fi=False,
                bound_control=False,
            )
        )

    khalf = lane_idx // fx.Int32(WMMA_M)  # 0/1: which 8-row kv half this lane owns

    # Mask base: swapped reads move the low NKV/2 slots up by n_block/2 and the high slots
    # down, so one base per half absorbs the whole correction and the per-slot term stays
    # the compile-time kvt*WMMA_N + i.
    if kv_pos_base is not None:
        _pos0 = kv_pos_base + khalf * fx.Int32(8)
        _pos_half = [_pos0 + kv_swap_delta, _pos0 - kv_swap_delta]

    R = len(s_list)
    q_max_list = q_max_list if q_max_list is not None else [None] * R
    q_min_list = q_min_list if q_min_list is not None else [None] * R

    # ---- Pass 1: masked S, flattened (kvt, i) order. Built for every row first so the
    # max-trees below emit INTERLEAVED. ----
    s_masked_list = []
    for r in range(R):
        s = s_list[r]
        q_max, q_min = q_max_list[r], q_min_list[r]
        s_masked = []
        for kvt in range(NKV):
            svec = fx.Vector(_ir(s[kvt]))
            for i in range(8):
                sval = fx.Float32(svec[i])
                if q_max is not None or q_min is not None or kv_len is not None:
                    kv_pos = _pos_half[kvt >= NKV // 2] + fx.Int32(kvt * WMMA_N + i)
                    if q_max is not None:
                        ubound = (
                            q_max
                            if kv_len is None
                            else fx.min(q_max, kv_len - fx.Int32(1))
                        )
                        sval = (kv_pos > ubound).select(neg_inf, sval)
                    if q_min is not None:
                        sval = (kv_pos < q_min).select(neg_inf, sval)
                    if kv_len is not None and q_max is None:
                        sval = (kv_pos >= kv_len).select(neg_inf, sval)
                s_masked.append(sval)
        s_masked_list.append(s_masked)

    # ---- Row max: balanced max-trees, interleaved across rows. ----
    max3 = lambda a, b, c: fmax(fmax(a, b), c)
    local_max_list = _tree_reduce_multi(s_masked_list, max3, fmax)

    # ---- Per row: peer reduce + deferred-rescale decision + corr. ----
    m_new_list, corr_list = [], []
    rescale_masks = None if not _defer else []
    for r in range(R):
        m_prev, q_min = m_prev_list[r], q_min_list[r]
        row_max = fmax(local_max_list[r], peer(local_max_list[r]))
        m_full = fmax(m_prev, row_max)

        # Deferred oaccu rescale: keep m STALE while the running max barely moves, so the
        # caller SKIPS the wide `o_acc *= corr`. The ballot promotes the test to
        # wave-uniform for that branch. `>` lowers to ordered OGT, so a fully-masked
        # lane's -inf - -inf = NaN compares false and never forces a rescale.
        if _defer:
            need = fsub(row_max, m_prev) > fx.Float32(RESCALE_THRESHOLD * LOG2E)
            # Select on the per-lane `need`, not the ballot: every lane owns its own q row
            # (the l<->l^16 pair agrees after the peer reduce), so staleness is per-lane
            # and the ballot is only the caller's branch condition. That leaves the branch
            # as the compare's only other consumer, so the fold sinks to the s_cbranch
            # instead of sitting between the max tree and the exp chain.
            m_new = need.select(m_full, m_prev)
            rescale_masks.append(fx.Int32(rocdl.ballot(fx.Int32.ir_type, need)))
        else:
            m_new = m_full

        # log2-domain, so exp2 takes the difference directly. m is seeded to BIG_NEG
        # (finite), so a fully masked row keeps m_new = BIG_NEG -> corr = 1 and
        # p = exp2(-inf) = 0, with no (-inf)-(-inf) anywhere and no clamp needed.
        corr = exp2(fsub(m_prev, m_new))
        m_new_list.append(m_new)
        corr_list.append(corr)

    # ---- Pass 2: p = exp2(S - m_new), every row's subs before any exp. The subs pair
    # into v_dual_sub_f32 against their row's shared m_new, and the exp run stays pure
    # TRANS -- the coexecution hazard is TRANS followed by non-TRANS VALU, so an
    # uninterrupted run needs none of the v_nop padding an interleaved one does. ----
    diff_list = [
        [fsub_inf(sv, m_new_list[r]) for sv in s_masked_list[r]] for r in range(R)
    ]
    p_flat_list = [[exp2(d) for d in diff] for diff in diff_list]
    p_list = [
        [
            fx.Vector.from_elements(pf[kvt * 8 : (kvt + 1) * 8], fx.Float32)
            for kvt in range(NKV)
        ]
        for pf in p_flat_list
    ]

    # ---- Row sum: interleaved balanced sum-trees. fadd_t drops reassoc so LLVM's
    # Reassociate does not re-linearize the tree into a serial chain. ----
    add3 = lambda a, b, c: fadd_t(fadd_t(a, b), c)
    local_sum_list = _tree_reduce_multi(p_flat_list, add3, fadd_t)

    d_new_list = []
    for r in range(R):
        d_new_list.append(
            fadd(
                fmul(corr_list[r], d_prev_list[r]),
                fadd(local_sum_list[r], peer(local_sum_list[r])),
            )
        )
    return p_list, m_new_list, d_new_list, corr_list, rescale_masks


def _pv_qk_gemm(
    *,
    v_emit,
    k_emit,
    p_list,
    q_frags_list,
    v_hdim,
    n_block,
    warp_type,
    o_acc_list=None,
    head=None,
    ring=PVQK_RING,
    lag=PVQK_LAG,
    phase="pvqk",
):
    """GEMM2(u-1) then GEMM1(u) driven by ONE ring over the concatenated V-then-K
    fragment streams, for all R q-WMMA-tiles this wave owns.

    V and K are shared across the q-tiles, so each fragment is shuffled once and fed into
    R independent WMMA chains. Consumption stays strictly sequential -- a full PV pass
    then a full QK pass -- so P dies before the first QK wmma and the two accumulator sets
    are never co-live. What the single ring buys is the refill window: the K loads for
    QK(u) issue ~NP/2 fragments before the transition, while PV's wmma stream still runs.

    GEMM2, O^T = V^T @ P^T, D[M=d, N=q]: **A = V^T** (src_a, transpose-loaded via
    ds_load_tr16_b128, two V-tiles (kv, kv+16) shuffled) and **B = P^T** (src_b, two
    softmax tiles p[2kt], p[2kt+1]). Contracts kv in ``nkt = n_block//WMMA_K`` tiles and
    produces ``d_tiles = v_hdim//WMMA_M`` d-tiles on the M axis. Lane ``l`` element ``si``
    of tile ``dt`` holds O[q = l%16, d = dt*WMMA_M + (l//16)*8 + si] -- the OManager16b
    fragment layout.

    GEMM1, S^T = K @ Q^T: **K = A-operand**, **Q = B-operand**. Contracts d in
    ``NDT = qk_hdim//WMMA_K`` tiles and produces ``NKV = n_block//WMMA_N`` kv-tiles. Lane
    ``l`` element ``si`` holds S^T[kv = kv_tile*WMMA_N + (l//16)*8 + si, q = l%16] (kv on
    the C-row / M axis, q on the C-col / N axis; GPU-verified). Each K fragment is the two
    16-col halves of a d-tile shuffled into a v16 matching the Q frag layout.

    ``p_list``/``q_frags_list``/``o_acc_list`` are length-R lists: per q-tile its bf16 P^T
    B-operands, its NDT v16-bf16 Q fragments, and (or None) its running ``d_tiles`` v8-f32
    O accumulator, already rescaled by ``corr``, which PV accumulates onto online.
    ``v_emit(j)``/``k_emit(j)`` emit the j-th V transpose-load / K ds_load of the resident
    block in flat ``(dt, kt, half)`` and ``(kv, dt, half)`` order; ``_ring_drive`` calls
    them on demand. ``head`` adopts the NP-deep prefetch the caller issued before softmax,
    so that latency still hides under the softmax VALU.

    Returns ``(out_list, s_acc_list)``: the updated O accumulators, and per q-tile NKV
    v8-f32 S^T accumulators.
    """
    R = len(p_list)
    d_tiles = v_hdim // WMMA_M
    nkt = n_block // WMMA_K
    NKV = n_block // WMMA_N
    NDT = len(q_frags_list[0])

    # ``phase`` drops one half of the stream when the caller knows it is dead (the
    # single-KV-tile core runs QK and PV as separate passes around softmax).
    do_pv = phase != "qk"
    do_qk = phase != "pv"
    num_vfrag = d_tiles * nkt if do_pv else 0
    num_kfrag = NKV * NDT if do_qk else 0
    num_vld = 2 * num_vfrag

    out_list = [[None] * d_tiles for _ in range(R)]
    if not do_pv:
        out_list = [list(row) for row in o_acc_list]
    s_acc_list = [[None] * NKV for _ in range(R)]
    p_frags = {}  # (qt, kt) -> B operand, handed to _ring_drive on its last fragment

    def emit(j):
        return v_emit(j) if j < num_vld else k_emit(j - num_vld)

    def consume(i, lo, hi):
        if i < num_vfrag:
            dt, kt = divmod(i, nkt)
            v_frag = lo.shuffle(hi, list(range(16)))
            for qt in range(R):
                acc = out_list[qt][dt]
                if acc is None:
                    acc = (
                        o_acc_list[qt][dt]
                        if o_acc_list is not None
                        else fx.Vector.filled(8, 0.0, fx.Float32)
                    )
                p = p_list[qt]
                p_frag = p[2 * kt].shuffle(p[2 * kt + 1], list(range(16)))
                p_frags[(qt, kt)] = p_frag
                out_list[qt][dt] = _wmma(v_frag, p_frag, acc)
            return
        kv, dt = divmod(i - num_vfrag, NDT)
        k_frag = lo.shuffle(hi, list(range(16)))
        for qt in range(R):
            acc = s_acc_list[qt][kv] if dt > 0 else fx.Vector.filled(8, 0.0, fx.Float32)
            s_acc_list[qt][kv] = _wmma(k_frag, q_frags_list[qt][dt], acc)

    def dies(i):
        # The PV half's B operands die on the final d-tile row, which is where the ring is
        # already refilling for QK. Release them through the ring so the refill sees one
        # WAR distance, not two.
        if i >= num_vfrag:
            return ()
        dt, kt = divmod(i, nkt)
        if dt != d_tiles - 1:
            return ()
        return [p_frags.pop((qt, kt)) for qt in range(R)]

    # Raise wave priority for the whole WMMA stream: under anti-phase the other half is
    # in its softmax VALU here, and the gemm half must win issue arbitration.
    rocdl.s_setprio(2)
    _ring_drive(
        num_frag=num_vfrag + num_kfrag,
        emit=emit,
        consume=consume,
        ring=ring,
        lag=lag,
        head=head,
        dies=dies,
    )
    rocdl.s_setprio(0 if warp_type.is_lo else 1)
    return out_list, s_acc_list


# ============================================================================
# Shared, layout-agnostic compute core
# ============================================================================


def _wg_kv_span(
    *, n_block, mask_right, gqa_ratio, num_q_tiles_per_wave, q_len, kv_len, window_right
):
    """This WG's KV band: (block_x, causal_off, kv_len_wg, num_tiles).

    A query at seq s attends [s+causal_off-window_left, s+causal_off+window_right] with
    causal_off = kv_len-q_len; mask_right clips kv_len_wg to the WG's max query's
    attend-limit so no tile runs fully past the band. num_tiles is also the one-KV-tile
    dispatch predicate, so it is computed here once and shared with the kernel entry."""
    BLOCK_M = _block_m(num_q_tiles_per_wave)
    block_x = _m_tile_idx()
    causal_off = kv_len - q_len
    if mask_right:
        wg_max_seq = (block_x * fx.Int32(BLOCK_M) + fx.Int32(BLOCK_M - 1)) // fx.Int32(
            gqa_ratio
        )
        wg_max_seq = fx.min(wg_max_seq, q_len - fx.Int32(1))
        kv_len_wg = wg_max_seq + causal_off + window_right + fx.Int32(1)
        kv_len_wg = fx.min(kv_len_wg, kv_len)
        kv_len_wg = fx.max(kv_len_wg, fx.Int32(1))
    else:
        kv_len_wg = kv_len
    return block_x, causal_off, kv_len_wg, fx.ceildiv(kv_len_wg, fx.Int32(n_block))


def _alloc_lds():
    """Allocate the full per-CU LDS and return its base. Called once per kernel body,
    before the warp-type dispatch, so both ``_core_attention`` traces share the single
    SharedAllocator flydsl permits; K/V, Q and the O epilogue all carve this base."""
    smem = fx.SharedAllocator().allocate(get_lds_capacity_bytes("gfx1250"))
    return fx.Int32(fx.ptrtoint(smem.peek().ptr))


def _core_attention_multi_kv_tiles(
    *,
    qk_hdim,
    v_hdim,
    n_block,  # compile-time KV block width (columns of one QK GEMM tile)
    mask_left,  # compile-time: bound the left band edge (finite window_left)
    mask_right,  # compile-time: bound the right band edge (causal or finite window_right)
    return_lse,
    has_sink,  # compile-time: fold a per-head sink logit into the softmax denom
    gqa_ratio,  # compile-time GQA group size = nheads_q // nheads_kv
    num_q_tiles_per_wave,  # compile-time q-WMMA tiles per wave; BLOCK_M = 16*this*8
    ptr_O,
    ptr_Q,
    ptr_K,
    ptr_V,
    ptr_LSE,
    ptr_sink,  # [nheads_q] fp32 per-head sink logits; read only when has_sink
    softmax_scale,
    stride_q_seq,
    stride_k_seq,
    stride_v_seq,
    stride_o_seq,
    stride_q_head,
    stride_k_head,
    stride_v_head,
    stride_o_head,
    # LSE addressing (element strides + per-batch bound), resolved by the caller.
    # Only consumed when return_lse; the caller may pass anything otherwise.
    stride_lse_seq,
    stride_lse_head,
    lse_base_elems,  # first element offset of this batch's LSE slab
    lse_num_records_bytes,  # buffer-resource bound (below the 0x7FFFFFFF drop)
    # Per-batch token ranges (fx.Int32), resolved by the caller:
    q_start,  # first Q token index of this batch in the global tensor
    q_len,  # valid Q tokens in this batch
    kv_start,  # first K/V token index of this batch
    kv_len,  # valid K/V tokens in this batch
    # Sliding-window bounds (runtime fx.Int32, >= 0). window_left read only when
    # mask_left, window_right only when mask_right. Causal == mask_right, window_right=0.
    window_left,
    window_right,
    warp_idx,  # runtime fx.Int32 wave index
    warp_type,  # compile-time WarpType (LO/HI x SIMD parity)
    lds_base,  # LDS base (fx.Int32), allocated once by the caller (_alloc_lds)
    elem_dtype,  # compile-time fx.BFloat16 / fx.Float16 for Q/K/V/P/O fragments
):
    """Layout-agnostic m32x8 compute, shared by the THD and BSHD kernel entries.

    The caller resolves the per-batch token ranges (``q_start``/``q_len`` and
    ``kv_start``/``kv_len``) — the only part that differs between varlen and batched —
    and passes them here. It also dispatches on runtime ``warp_type``, tracing this body
    once per compile-time value; the two instantiations differ only in their main-loop
    phase ordering (LO drives the K load, HI shadows it).
    """
    BLOCK_M = _block_m(num_q_tiles_per_wave)
    lane_idx = _lane_id()
    kv_head, q_head_idx, seq_idx = _packed_tile_indices(
        gqa_ratio, warp_idx, lane_idx, num_q_tiles_per_wave
    )

    # softmax_scale*LOG2E, folded into Q by the loader's bf16 multiply.
    _q_scale = softmax_scale * fx.Float32(LOG2E)

    # K/V staging: N_KV_PP slots of 2 LDS_CHUNK_BYTES chunks each, every tile split 2-way
    # along n_block into chunks 6 apart (different 64 KB segments). Q time-shares slot 1's
    # chunks, O the two slots the loop has finished with -- see the 12-chunk map up top.
    if USE_TDM_LOADER:
        q_mgr = QManager16bV2(
            qk_hdim=qk_hdim,
            gqa_ratio=gqa_ratio,
            num_waves=NUM_WAVES,
            q_tiles_per_wave=num_q_tiles_per_wave,
            elem_dtype=elem_dtype,
        )
        k_mgr = KManager16bV2(
            qk_hdim=qk_hdim,
            n_block=n_block,
            num_waves=NUM_WAVES,
            elem_dtype=elem_dtype,
        )
        v_mgr = VManager16bV2(
            v_hdim=v_hdim, n_block=n_block, num_waves=NUM_WAVES, elem_dtype=elem_dtype
        )
    else:
        q_mgr = QManager16bV1(
            qk_hdim=qk_hdim,
            gqa_ratio=gqa_ratio,
            num_waves=NUM_WAVES,
            q_tiles_per_wave=num_q_tiles_per_wave,
            elem_dtype=elem_dtype,
        )
        k_mgr = KManager16bV1(
            qk_hdim=qk_hdim,
            n_block=n_block,
            num_waves=NUM_WAVES,
            elem_dtype=elem_dtype,
        )
        v_mgr = VManager16bV1(
            v_hdim=v_hdim, n_block=n_block, num_waves=NUM_WAVES, elem_dtype=elem_dtype
        )
    k_blk_bytes = k_mgr.get_lds_size_in_byte()
    v_blk_bytes = v_mgr.get_lds_size_in_byte()
    assert k_blk_bytes % KV_LDS_SPLITS == 0 and v_blk_bytes % KV_LDS_SPLITS == 0
    # Chunk stride between a tile's two n_block halves (K[pp][0] -> K[pp][1]) and between
    # consecutive slots. Slot pp occupies chunks 2pp, 2pp+1 low and 2pp+6, 2pp+7 high.
    _SPLIT_STRIDE = 6 * LDS_CHUNK_BYTES  # 156 KB
    slot_bytes = 2 * LDS_CHUNK_BYTES  # 52 KB: one slot's K|V pair in one half
    for _who, _b in (("K", k_blk_bytes), ("V", v_blk_bytes)):
        assert _b // KV_LDS_SPLITS <= LDS_CHUNK_BYTES, (
            f"{_who} split {_b // KV_LDS_SPLITS}B exceeds the {LDS_CHUNK_BYTES}B chunk "
            f"(qk_hdim={qk_hdim}, v_hdim={v_hdim}, n_block={n_block})"
        )
    assert 2 * N_KV_PP * slot_bytes <= get_lds_capacity_bytes(
        "gfx1250"
    ), "12-chunk K|V layout over LDS capacity"

    def _k_lds_buf(
        pp,
    ):  # slot ``pp``'s low-half base == K[pp][0] (int or fx.Int32; folds when const)
        if isinstance(pp, int):
            pp = fx.Int32(pp)
        return lds_base + pp * fx.Int32(slot_bytes)

    def _v_lds_buf(pp):  # V[pp][0] == the slot base one chunk in
        return _k_lds_buf(pp) + fx.Int32(LDS_CHUNK_BYTES)

    # Logical slot -> physical chunk pair. Q covers all of physical slot 1 and the low
    # 17 KB of slot 2's K chunks, so physical slot 0 is the only Q-disjoint one; putting
    # the first tile (logical slot 1) there lets the prologue issue it before Q is read.
    _PSLOT = [2, 0, 1]

    def _k_bufs_at(
        slot,
    ):  # K[.][0], K[.][1] of the slot whose low-half base is ``slot``
        return [slot, slot + fx.Int32(_SPLIT_STRIDE)]

    def _v_bufs_at(slot):
        v0 = slot + fx.Int32(LDS_CHUNK_BYTES)
        return [v0, v0 + fx.Int32(_SPLIT_STRIDE)]

    # READ side only -- the producer keeps writing split s to chunk s. Odd SIMDs take the
    # two bases in the other order, so the parities are never in the same 64 KB segment
    # set at the same point of a gemm. _split_bufs runs only in the prologue and its
    # results are carried as ds pointers, so this costs a few adds and nothing in-loop.
    assert KV_LDS_SPLITS == 2, "parity order needs a 2-way split"
    _odd = warp_idx & fx.Int32(1)
    _rd0 = _odd * fx.Int32(_SPLIT_STRIDE)
    _rd = [_rd0, fx.Int32(_SPLIT_STRIDE) - _rd0]
    # Slot kvt then holds physical kv-tile kvt ^ (NKV/2); the mask is the only consumer
    # of the absolute index, and it takes the correction as +/- this delta.
    _kv_swap_delta = _odd * fx.Int32(n_block // 2)

    def _split_bufs(b):
        return [b + o for o in _rd]

    def _k_lds_bufs(pp):
        return _split_bufs(_k_lds_buf(pp))

    def _v_lds_bufs(pp):
        return _split_bufs(_v_lds_buf(pp))

    # ---- Q and O own no LDS: both time-share KV chunks in LDS_QO_BYTES per-wave slices.
    # Q sits in the K[1][0] / K[1][1] chunk pairs (52 KB and 208 KB, 4 waves x 17 KB each);
    # it is drained into VGPR and dead before the prologue issues a tile into slot 1 or 2
    # (s_wait_dscnt + barrier below). Wave w -> chunk B iff (w&1) ^ ((w>>2)&1), so the two
    # waves sharing a SIMD (w and w+4) always land in different chunks. ----
    for _who, _b in (("Q", q_mgr.warp_lds_size_in_byte()),):
        assert (
            _b <= LDS_QO_BYTES
        ), f"{_who} per-wave {_b}B over the {LDS_QO_BYTES}B slice"
    _q_chunk_b = (warp_idx & fx.Int32(1)) ^ ((warp_idx >> fx.Int32(2)) & fx.Int32(1))
    q_lds_warp = (
        _k_lds_buf(1)
        + _q_chunk_b * fx.Int32(_SPLIT_STRIDE)
        + (warp_idx >> fx.Int32(1)) * fx.Int32(LDS_QO_BYTES)
    )

    q_mgr.load_q_to_vgpr_part1(
        ptr_Q=ptr_Q,
        stride_q_seq=stride_q_seq,
        stride_q_head=stride_q_head,
        q_start=q_start,
        q_len=q_len,
        kv_head=kv_head,
        block_x=_m_tile_idx(),
        warp_idx=warp_idx,
        lane_idx=lane_idx,
        ptr_lds_warp=q_lds_warp,
    )

    # ---- This WG's KV tiles span relative kv [start_tile*n_block, kv_len_wg); mask_left
    # moves start_tile past whole tiles before the WG's min query's band start. The
    # defensive min() keeps start_tile a valid buffer index even for an over-launched WG
    # whose whole band is empty (its per-element masks zero the work anyway).
    block_x, causal_off, kv_len_wg, num_tiles = _wg_kv_span(
        n_block=n_block,
        mask_right=mask_right,
        gqa_ratio=gqa_ratio,
        num_q_tiles_per_wave=num_q_tiles_per_wave,
        q_len=q_len,
        kv_len=kv_len,
        window_right=window_right,
    )
    last_tile = num_tiles - fx.Int32(1)  # always carries the kv_len tail
    if mask_left:
        wg_min_seq = (block_x * fx.Int32(BLOCK_M)) // fx.Int32(gqa_ratio)
        kv_lo = fx.max(wg_min_seq + causal_off - window_left, fx.Int32(0))
        start_tile = kv_lo // fx.Int32(n_block)
        start_tile = fx.min(start_tile, last_tile)
    else:
        start_tile = fx.Int32(0)

    def _tile_row0(t):  # clamped to the last tile, never guarded off the end
        return fx.min(t, last_tile) * fx.Int32(n_block)

    # How many rows of tile t are in-bounds: n_block for every tile but the last, whose
    # tail is loop-invariant. Tiles past last_tile read as the last one, matching
    # _tile_row0's clamp. A scalar select keeps this out of the VALU -- the min/max form
    # folds to v_med3_i32, which has no scalar counterpart, so a uniform value would go
    # out to a VGPR and come back through v_readfirstlane_b32 on every body.
    tail_valid = kv_len_wg - last_tile * fx.Int32(n_block)

    def _kv_valid(t):
        return (t < last_tile).select(fx.Int32(n_block), tail_valid)

    # ---- Prologue. All K/V address arithmetic is pure (no memory op until
    # ``.async_load()``), so it is hoisted into the Q global-load shadow and the copies
    # themselves issue last, leaving nothing between them and the prologue barrier.
    #
    # Slot rotation is LOCAL to this WG's tile stream: the prologue puts start_tile into
    # slot 1 and the loop carries the slot bases as iter_args (rotated in the yield), so
    # start_tile is irrelevant to the placement and no runtime "% N_KV_PP" is evaluated.
    # Body u reads K from slot 1 and V from slot 0, so the first body needs tile
    # start_tile's K AND V in slot 1 (K for QK(start_tile), V for the next body's PV) and
    # only FINITE data in slot 0's V region -- its PV is the dead leading one, p == 0, and
    # 0 * NaN would poison O. Loading start_tile's V there is the cheapest such filler.
    start_row0 = start_tile * fx.Int32(n_block)
    # This wave's slot among its half's producers; it owns one dense
    # n_block/KV_PRODUCER_WARPS row band and copies it alone.
    _producer_warp = warp_idx % fx.Int32(KV_PRODUCER_WARPS)
    _kv_ctx = ProducerCtx(
        producer_warp=_producer_warp,
        num_producer_warps=KV_PRODUCER_WARPS,
        lane_idx=lane_idx,
    )

    def _get_kv_desc(slot, row0, valid):
        """This half's BufferOpDescriptor for one tile: LO issues every K copy, HI every
        V copy. Pure until ``.async_load()``; which transport it carries (async vs TDM) is
        the manager's business, not this call site's."""
        if warp_type.is_lo:
            return k_mgr.load_descriptor(
                ptr_lds=_k_bufs_at(slot),
                ptr_src=ptr_K,
                stride_seq=stride_k_seq,
                stride_head=stride_k_head,
                head=kv_head,
                row0=kv_start + row0,
                valid=valid,
                ctx=_kv_ctx,
            )
        return v_mgr.load_descriptor(
            ptr_lds=_v_bufs_at(slot),
            ptr_src=ptr_V,
            stride_seq=stride_v_seq,
            stride_head=stride_v_head,
            head=kv_head,
            row0=kv_start + row0,
            valid=valid,
            ctx=_kv_ctx,
        )

    kv0 = _get_kv_desc(_k_lds_buf(_PSLOT[1]), start_row0, _kv_valid(start_tile))
    # LO's second copy is K(start+1) into slot 2, the tile the body no longer issues once
    # K runs ahead; HI's is V(start) into slot 0, read by body start's dead PV.
    if warp_type.is_lo:
        fill_tile = start_tile + fx.Int32(1)
        kv_fill = _get_kv_desc(
            _k_lds_buf(_PSLOT[2]), _tile_row0(fill_tile), _kv_valid(fill_tile)
        )
    else:
        kv_fill = _get_kv_desc(_k_lds_buf(_PSLOT[0]), start_row0, _kv_valid(start_tile))
    # Fence depths, straight off the descriptor's declared counter usage. A counter this
    # half's copies never touch gets -1 ("do not wait on it at all"), NOT 0. The one it
    # does touch gets this half's per-tile copy count -- the depth LO's partial
    # steady-state fence names to leave the newest tile in flight. Both counters take a
    # depth and retire in issue order, so it means the same on either transport. Per-HALF:
    # LO's K band and HI's V band differ whenever qk_hdim != v_hdim.
    num_tdm_copies = kv0.tensorcnt if kv0.tensorcnt else -1
    num_async_copies = kv0.asynccnt if kv0.asynccnt else -1
    _kv_drain = _kv_drain_depths(num_tdm_copies, num_async_copies)
    # What this half copies in the prologue, transport-independent: its tile plus the fill.
    _tiles = [kv0, kv_fill]
    # When each goes out is NOT transport-independent. Under V2, copies whose destination
    # misses Q go out BEFORE Q is read, so their global latency overlaps Q's and part2
    # waits tensorcnt down to them instead of to 0; only LO's K-ahead fill (logical slot
    # 2) lands on Q, so that one waits for the barrier. Under V1 there is no such overlap
    # to have: Q's own stage is on asynccnt too and its part2 waits that counter to a
    # depth counted over Q's loads ALONE, so a tile copy issued first would corrupt the
    # count. Everything goes after the barrier, and after part2, which ends at depth 0.
    if not USE_TDM_LOADER:
        _early, _late = [], _tiles
    elif warp_type.is_lo:
        _early, _late = _tiles[:1], _tiles[1:]
    else:
        _early, _late = _tiles, []
    for _d in _early:
        _d.async_load()
    if USE_TDM_LOADER:
        q_frags = q_mgr.load_q_to_vgpr_part2(
            scale=_q_scale, skip_tensorcnt=sum(_d.tensorcnt for _d in _early)
        )
    else:
        q_frags = q_mgr.load_q_to_vgpr_part2(scale=_q_scale)
    # Q's ds_loads must be RETIRED, not just issued, before the barrier that releases
    # the late copies onto Q's chunks: gpu.barrier() does not retire LDS reads, and Q's
    # atom is per-wave while a tile's is per-producer (wave A's Q region is written by
    # wave B's share of the tile).
    rocdl.s_wait_dscnt(0)
    gpu.barrier()
    # HI enters its resting priority at the first point both halves have reached: from
    # here on the lagging half must not lose arbitration to its SIMD-mate, and
    # _pv_qk_gemm's exit restores this same level after every gemm.
    if not warp_type.is_lo:
        rocdl.s_setprio(1)
    for _d in _late:
        _d.async_load()
    _kv_fence(*_kv_drain)

    # ---- Loop init: the online-softmax seed and the O accumulators, all iter_args.
    #
    # Seeding m=-inf, d=0, O=0 makes the first tile's corr = exp2(m_prev-m_new) = 0 zero
    # the (already-zero) O before its PV adds in — the standard flash seed.
    #
    # Attention sink (compile-time) is one extra ``exp(sink)`` term in the softmax denom.
    # Fold it in by seeding m = sink[q_head]*LOG2E (m is log2-domain) and d = 1.0
    # (= exp(sink-sink)); the rescales carry that d seed to exactly exp(sink - m_final).
    # Without a sink the first tile's corr zeroes the d seed, so d=1 would equal d=0 --
    # the no-sink path keeps d=0 to stay byte-for-byte.
    d_tiles = v_hdim // WMMA_M
    R = num_q_tiles_per_wave
    NKV = n_block // WMMA_N
    # Per-q-tile carried state: [m, d, O_0 .. O_{d_tiles-1}, P_0 .. P_{NKV-1}]. P is the
    # software pipeline -- body u's PV consumes the P body u-1's softmax produced. The HI
    # half runs softmax(u-1) before gemm(u), so its last NKV slots hold the f32 s_acc the
    # next body's softmax consumes instead of the bf16 P; the slot count is the same.
    _QS = 2 + d_tiles + NKV
    _lag_sm = not warp_type.is_lo
    if _lag_sm:
        # Rotating the drain to the body tail shifts this half's barrier stream by one:
        # it now opens with the phase barrier and closes with the drain. One filler here
        # (and its partner after the loop on the leading half) re-pairs the two streams
        # so phase barrier still meets phase barrier.
        _kv_fence(*_kv_drain)
    if has_sink:
        num_heads_q = gpu.grid_dim.y * fx.Int32(gqa_ratio)
        m_init = [
            _load_sink_logit(ptr_sink, q_head_idx[qt], num_heads_q) * fx.Float32(LOG2E)
            for qt in range(R)
        ]
        d_init = [fx.Float32(1.0) for _ in range(R)]
    else:
        m_init = [fx.Float32(BIG_NEG) for _ in range(R)]
        d_init = [fx.Float32(0.0) for _ in range(R)]
    # P == 0 makes the first body's PV the dead leading one (0 * V onto the zero O), so
    # no prologue QK/softmax trace is needed.
    _init = []
    for qt in range(R):
        _init += (
            [
                _raw(m_init[qt]),
                _raw(d_init[qt]),
            ]
            + [_raw(fx.Vector.filled(8, 0.0, fx.Float32)) for _ in range(d_tiles)]
            + [
                _raw(
                    fx.Vector.filled(8, float("-inf"), fx.Float32)
                    if _lag_sm
                    else fx.Vector.filled(8, 0.0, fx.Float32).to(elem_dtype)
                )
                for _ in range(NKV)
            ]
        )

    # ---- One ds_load base-pointer set per slot plus the slot's byte base, all carried as
    # iter_args and LEFT-ROTATED in the yield: iteration i reads set 0 (= tile i) and
    # writes tile i+2 at byte base 2. Rotation is pure register renaming, so no runtime
    # "% N_KV_PP" is ever evaluated. The per-manager base count is carried generically. ----
    k_lds_ld = [
        k_mgr.ds_load_ptrs(ptr_lds=_k_lds_bufs(_PSLOT[i]), lane_idx=lane_idx)
        for i in range(N_KV_PP)
    ]
    v_lds_ld = [
        v_mgr.ds_load_ptrs(ptr_lds=_v_lds_bufs(_PSLOT[i]), lane_idx=lane_idx)
        for i in range(N_KV_PP)
    ]
    _NKB = len(k_lds_ld[0])  # ds bases per K buffer (one per LDS split)
    _NVB = len(v_lds_ld[0])
    _PTR_BASE = len(_init)
    for i in range(N_KV_PP):
        _init = _init + k_lds_ld[i]
    _VB0 = len(_init)
    for i in range(N_KV_PP):
        _init = _init + v_lds_ld[i]
    _SLOT_BASE = len(_init)
    _init = _init + [_raw(_k_lds_buf(_PSLOT[i])) for i in range(N_KV_PP)]

    # Carried ring head: _NH of the first body's V loads, issued here (slot 0 is resident
    # -- the prologue fence just drained it) and thereafter at the end of each body.
    _num_kfrag = k_mgr.num_ds_loads() // 2
    _num_vfrag = v_mgr.num_ds_loads() // 2
    _NP = _ring_num_prefetch(_num_vfrag + _num_kfrag, PVQK_RING, PVQK_LAG)
    # The lagging half opens its body with softmax, so its ring head is issued there and
    # consumed after the barrier -- within one body, nothing to carry.
    _NH = 0 if _lag_sm else min(PVQK_HEAD_CARRY, _NP)
    assert _NH <= 2 * _num_vfrag, "carried head must be V loads only"
    _HEAD_BASE = len(_init)
    _seed_head = []
    for j in range(_NH):
        # Same per-pair pin as ``_issue_head`` in the body.
        _seed_head.append(_raw(v_mgr.load_one_to_reg(v_lds_ld[0], j)))
        if len(_seed_head) % 2 == 0:
            rocdl.sched_barrier(0)
    _init = _init + _seed_head

    # ========================================================================
    # Main KV loop -- SOFTWARE-PIPELINED by one tile. The loop variable ``u`` is the
    # SOFTMAX tile, and body u runs
    #
    #     PV(u-1)  ->  QK(u)  ->  softmax(u)  ->  rescale O by corr(u)
    #
    # carrying P across the back edge. That puts the two gemms back to back (a shared
    # ring feeds both) and leaves exactly three live slots: V(u-1) in slot 0, K(u) in
    # slot 1, tile u+1's copy landing in slot 2. Slots are selected by the carried ds
    # pointers / bases, left-rotated at the end of each `main_loop`.
    #
    # u runs [start_tile, num_tiles] -- ONE body more than there are tiles. Both extra
    # half-bodies are dead rather than peeled, an extra trace costing far more than an
    # extra iteration: body start_tile's PV multiplies the seeded P == 0, and body
    # num_tiles' QK/softmax sits at kv_pos_base >= kv_len so every element masks to -inf
    # (m unchanged, corr == 1, P == 0, d unchanged).
    # ========================================================================
    def main_loop(u, state, *, mask_left, mask_right, kv_len):
        # mask_left/mask_right/kv_len shadow the closure flags: the caller splits the
        # tile stream into a mask-free clean region + boundary loops and passes None for
        # any edge this sub-loop provably doesn't cross. The split points are already
        # expressed in SOFTMAX tiles, so they carry over unchanged. The lagging half runs
        # softmax one tile behind its gemm, hence u-1.
        sm_tile = u - fx.Int32(1) if _lag_sm else u
        kv_tile_start = sm_tile * fx.Int32(
            n_block
        )  # softmax tile's first (batch-relative) kv row

        # Unpack: R independent per-q-tile (m, d, O, P) groups, then the shared ds state.
        m_prev = [fx.Float32(state[qt * _QS + 0]) for qt in range(R)]
        d_prev = [fx.Float32(state[qt * _QS + 1]) for qt in range(R)]
        o_acc = [
            [fx.Vector(state[qt * _QS + 2 + dt]) for dt in range(d_tiles)]
            for qt in range(R)
        ]
        # Carried gemm/softmax hand-off: P(u-1) on the leading half, S(u-1) on the
        # lagging one (same slot count, different element type).
        carry_prev = [
            [fx.Vector(state[qt * _QS + 2 + d_tiles + kvt]) for kvt in range(NKV)]
            for qt in range(R)
        ]
        k_slots = [
            list(state[_PTR_BASE + i * _NKB : _PTR_BASE + (i + 1) * _NKB])
            for i in range(N_KV_PP)
        ]
        v_slots = [
            list(state[_VB0 + i * _NVB : _VB0 + (i + 1) * _NVB]) for i in range(N_KV_PP)
        ]
        slot_of = [fx.Int32(state[_SLOT_BASE + i]) for i in range(N_KV_PP)]
        head_carry = [fx.Vector(state[_HEAD_BASE + i]) for i in range(_NH)]
        v_curr = v_slots[0]  # tile u-1: this body's PV
        k_curr = k_slots[1]  # tile u:   this body's QK

        # Tile prefetched by this body, CLAMPED to the last tile rather than guarded off
        # past the end. Every body then issues exactly one tile's copies, so the fence
        # count is uniform and the last iteration needs no peel -- cheaper than the extra
        # trace a peel costs, against ~2 dead L2-resident tile loads per workgroup. The
        # clamped re-load lands in the slot body num_tiles reads as its dead K, and O
        # stages past all the slots. This half's copy: LO K(u+2) into slot 0 (its K half
        # died at body u-1; its V half is what this body reads, a different chunk),
        # HI V(u+1) into slot 2.
        _ahead = 2 if warp_type.is_lo else 1
        wr_slot = slot_of[0] if _ahead == 2 else slot_of[N_KV_PP - 1]
        pf_tile = u + fx.Int32(_ahead)
        pf_row0 = _tile_row0(pf_tile)
        pf_valid = _kv_valid(pf_tile)

        def _drain_barrier():
            # The producing half's own copy counter. K runs two bodies ahead, so LO reaches
            # this fence with the tile it is about to read second-oldest and can leave the
            # newest in flight. V only has one body of slack on the ring head this half
            # issues, so HI still drains to 0.
            #
            # dscnt is drained only to _NH: the leading half issues its ring head at the
            # tail of the previous body, so a full drain here would retire it right before
            # the gemm that wants it in flight. The gemm's own last ring fragment already
            # took dscnt to 0, so every read older than the head is retired regardless --
            # the WAR wall for the slot about to be written still holds.
            if warp_type.is_lo:
                _kv_fence(num_tdm_copies, num_async_copies, num_dscnt=_NH)
            else:
                _kv_fence(*_kv_drain, num_dscnt=_NH)

        # This half's descriptor for tile ``pf``'s K (LO) or V (HI) into the oldest slot.
        # Built up front: pure, so its address VALU overlaps the drain; only the copy
        # itself must stay after the barrier.
        pf_desc = _get_kv_desc(wr_slot, pf_row0, pf_valid)
        num_kfrag = k_mgr.num_ds_loads() // 2
        num_vfrag = v_mgr.num_ds_loads() // 2

        def _k_emit(j):
            return k_mgr.load_one_to_reg(k_curr, j)

        def _v_emit(j):
            return v_mgr.load_one_to_reg(v_curr, j)

        def _pvqk_emit(j):
            return _v_emit(j) if j < 2 * num_vfrag else _k_emit(j - 2 * num_vfrag)

        def _issue_head(emit, first, last, carried=()):
            # Pin every fragment's load PAIR in place. The ring consumes loads 2i and
            # 2i+1 for fragment i, but the scheduler scatters the 20-deep burst freely --
            # in the dumps a partner load drifted 12+ slots back, so the wmma waited on
            # most of the burst (s_wait_dscnt 0x9 / 0x7) instead of the ring's 0x12.
            # NOT _keepalive -- that is a USE, so it drags an s_wait_dscnt 0x0 in with it.
            out = list(carried)
            for j in range(first, last):
                out.append(emit(j))
                if len(out) % 2 == 0:
                    rocdl.sched_barrier(0)
            return out

        def _pvqk_head():
            # The first _NH loads came from the previous body (issued under its softmax);
            # only the remainder is issued here.
            return _issue_head(_v_emit, _NH, _NP, head_carry)

        def _softmax_phase(s_in, o_in):
            # This lane's query (tile qt) attends [q_min, q_max] in batch-relative kv.
            # None bounds are skipped, so a clean-region tile passes all-None and does
            # zero per-element masking. kv_len is passed only on the right-boundary
            # sub-loop, which is also what neutralizes the dead trailing softmax. All R
            # q-tiles go in ONE call so their tree reductions emit interleaved.
            q_max_list = [
                seq_idx[qt] + causal_off + window_right if mask_right else None
                for qt in range(R)
            ]
            q_min_list = [
                seq_idx[qt] + causal_off - window_left if mask_left else None
                for qt in range(R)
            ]
            p_list, m_new_list, d_new_list, corr_list, rescale_masks = _softmax(
                s_list=s_in,
                m_prev_list=m_prev,
                d_prev_list=d_prev,
                lane_idx=lane_idx,
                n_block=n_block,
                kv_pos_base=kv_tile_start,
                kv_swap_delta=_kv_swap_delta,
                q_max_list=q_max_list,
                q_min_list=q_min_list,
                kv_len=kv_len,
            )

            # Rescale each q-tile's running O by this tile's corr. The leading half
            # rescales at the END of the body and the next body's PV accumulates onto the
            # result; the lagging half rescales before its own PV. Same product either way.
            # Under deferral the wide multiply is gated behind a non-divergent scf.if that
            # fires only when a running max actually moved, with ONE branch covering all R
            # rows -- neither row moves in the common case, so the steady state is a single
            # not-taken s_cbranch_vccz, and a row that stayed stale has corr == 1 exactly,
            # making its rescale inside the taken branch the identity.
            corr_vecs = [
                fx.Vector.from_elements([corr_list[qt]], fx.Float32).broadcast_to(8)
                for qt in range(R)
            ]
            o_vecs = [
                [fx.Vector(_ir(o_in[qt][dt])) for dt in range(d_tiles)]
                for qt in range(R)
            ]
            if rescale_masks is None:
                o_resc_list = [
                    [ov * corr_vecs[qt] for ov in o_vecs[qt]] for qt in range(R)
                ]
            else:
                # Folded HERE, not in _softmax: the ballots are VALU-produced SGPRs, so
                # keeping the s_or/s_cmp at the use site leaves the whole row-max chain
                # between the v_cmp and the turnaround. OR the raw masks, not the per-row
                # booleans -- one s_or_b32 covers all R rows.
                mask_any = rescale_masks[0]
                for _m in rescale_masks[1:]:
                    mask_any = mask_any | _m

                @flyc.jit
                def _maybe_rescale_all(o_flat_in, corr_flat, do_rescale):
                    result = o_flat_in
                    if do_rescale:
                        result = [ov * cv for ov, cv in zip(o_flat_in, corr_flat)]
                    return result

                # One flat list: the R rows share the folded condition, so they rescale
                # (or pass through) together.
                o_flat = list(
                    _maybe_rescale_all(
                        [v for row in o_vecs for v in row],
                        [corr_vecs[qt] for qt in range(R) for _ in range(d_tiles)],
                        mask_any != fx.Int32(0),
                    )
                )
                o_resc_list = [
                    o_flat[qt * d_tiles : (qt + 1) * d_tiles] for qt in range(R)
                ]
            return p_list, m_new_list, d_new_list, o_resc_list

        # Only the drain barrier (the one before tensor_load_to_lds) is required; the wall
        # is pure scheduling. Dropping it halves the in-loop barriers and wins everywhere
        # except non-window-masked qk_hdim 256 (n_block 64), which regresses ~4%.
        # Compile-time constant, so both halves still agree on the barrier count.
        _keep_phase_wall = qk_hdim >= 256 and not (mask_left and mask_right)

        def _phase_barrier():
            # One half leaves the WMMA stream here as the other enters it; the
            # sched_barriers pin that split even when the s_barrier itself is gone.
            rocdl.sched_barrier(0)
            if _keep_phase_wall:
                _bare_barrier()
            rocdl.sched_barrier(0)

        # GEMM2(u-1) then GEMM1(u) on one ring: O += P^T(u-1) @ V(u-1), then
        # S^T = K(u) @ Q^T. sched_barrier fences the ring head out of the WMMA stream
        # (no wmma<-ds_load bubble); the ring itself issues the per-fragment s_wait_dscnt.
        if _lag_sm:
            pf_desc.async_load()
            p_f32, m_new_list, d_new_list, o_resc = _softmax_phase(carry_prev, o_acc)
            # Head out BEFORE P is narrowed: it is ds_loads with no dependence on P, so
            # issuing it first puts the conversion in its shadow rather than behind it.
            rocdl.sched_barrier(0)
            pvqk_head = _pvqk_head()
            rocdl.sched_barrier(0)
            p_list = _p_to_elem(p_f32, elem_dtype)
            # Anchor the conversion in THIS block. Its only real use is past the barrier,
            # so MachineSink (which ignores sched_barrier) would otherwise sink every
            # v_cvt_pk_bf16_f32 into the gemm and interleave it with the WMMA.
            _keepalive([v for pt in p_list for v in pt])
            _phase_barrier()
            o_out, s_acc = _pv_qk_gemm(
                v_emit=_v_emit,
                k_emit=_k_emit,
                p_list=p_list,
                q_frags_list=q_frags,
                v_hdim=v_hdim,
                n_block=n_block,
                warp_type=warp_type,
                o_acc_list=o_resc,
                head=pvqk_head,
            )
            carry_next = s_acc
            _drain_barrier()
            head_next = []
        else:
            _drain_barrier()
            pvqk_head = _pvqk_head()
            pf_desc.async_load()
            rocdl.sched_barrier(0)
            o_acc, s_list = _pv_qk_gemm(
                v_emit=_v_emit,
                k_emit=_k_emit,
                p_list=carry_prev,
                q_frags_list=q_frags,
                v_hdim=v_hdim,
                n_block=n_block,
                warp_type=warp_type,
                o_acc_list=o_acc,
                head=pvqk_head,
            )
            _phase_barrier()
            carry_f32, m_new_list, d_new_list, o_out = _softmax_phase(s_list, o_acc)
            # Next body's ring head: V(u) from the slot this body read K from, resident
            # and already fenced. Mirrors the lagging half -- both issue the head just
            # before the barrier preceding the gemm that consumes it, ahead of the
            # narrowing for the same reason.
            rocdl.sched_barrier(0)
            head_next = _issue_head(
                lambda j: v_mgr.load_one_to_reg(v_slots[1], j), 0, _NH
            )
            rocdl.sched_barrier(0)
            carry_next = _p_to_elem(carry_f32, elem_dtype)

        # Yield: R updated (m, d, O, P) groups, then the ds pointers and slot bases
        # left-rotated by one so slot 0 holds tile u (next body's PV) and the oldest
        # rotates into the write position.
        out = []
        for qt in range(R):
            out += (
                [_raw(m_new_list[qt]), _raw(d_new_list[qt])]
                + [_raw(o) for o in o_out[qt]]
                + [_raw(cv) for cv in carry_next[qt]]
            )
        rot = list(range(1, N_KV_PP)) + [0]
        for i in rot:
            out += k_slots[i]
        for i in rot:
            out += v_slots[i]
        out += [_raw(slot_of[i]) for i in rot]
        out += [_raw(h) for h in head_next]
        return out

    # ---- Stream softmax tiles [start_tile, num_tiles] through 3 sub-loops split by the
    # attention band, so interior tiles fully inside the band skip masking. clean_lo /
    # clean_hi are runtime split points, but each sub-loop's mask on/off is COMPILE-TIME.
    # The rotation state threads continuously through all three, leaving the slot
    # assignment intact, and the bounds are already softmax-tile indices, so the pipeline
    # shift only extends the last loop by one body (the dead-QK one).
    #   [start_tile, clean_lo)    left boundary   (emitted only when mask_left)
    #   [clean_lo,   clean_hi)    clean, no mask
    #   [clean_hi,   num_tiles+1) right boundary + kv_len tail + the dead trailing body
    num_iter = fx.Int32(num_tiles) - start_tile

    # clean_hi = first tile that could need RIGHT masking = the WG's earliest query's
    # diagonal tile ((min q_max + 1)//n_block). Kept <= last_tile so the tail tile stays in
    # the right loop, and >= start_tile for a valid partition.
    if mask_right:
        wg_min_seq = (block_x * fx.Int32(BLOCK_M)) // fx.Int32(gqa_ratio)
        qmax_min = fx.max(wg_min_seq + causal_off + window_right, fx.Int32(0))
        clean_hi = (qmax_min + fx.Int32(1)) // fx.Int32(n_block)
    else:
        clean_hi = fx.Int32(num_tiles)
    clean_hi = fx.max(fx.min(clean_hi, last_tile), start_tile)

    # clean_lo = first tile fully at/above the WG's latest query's window start
    # (ceildiv(max q_min, n_block)); clamped into [start_tile, clean_hi].
    if mask_left:
        wg_max_seq = fx.min(
            (block_x * fx.Int32(BLOCK_M) + fx.Int32(BLOCK_M - 1))
            // fx.Int32(gqa_ratio),
            q_len - fx.Int32(1),
        )
        qmin_max = fx.max(wg_max_seq + causal_off - window_left, fx.Int32(0))
        clean_lo = (qmin_max + fx.Int32(n_block - 1)) // fx.Int32(n_block)
    else:
        clean_lo = start_tile
    clean_lo = fx.min(fx.max(clean_lo, start_tile), clean_hi)
    if mask_left:
        # The lagging half's softmax tile is u-1, so the clean loop must start one body
        # later or tile clean_lo-1 would cross the left edge unmasked. The left loop
        # absorbs the extra body; over-masking a clean tile is a no-op. Only needed when
        # a left loop exists at all -- with mask_left off, clean_lo == start_tile and
        # shifting would drop the first body entirely.
        clean_lo = fx.min(clean_lo + fx.Int32(1), clean_hi)

    @flyc.jit
    def _run_tiles(state, lo_i32, hi_i32, *, mask_left, mask_right, kv_len):
        final_state = state
        for tile, carried in range(fx.Index(lo_i32), fx.Index(hi_i32), 1, init=state):
            next_state = main_loop(
                fx.Int32(tile),
                list(carried),
                mask_left=mask_left,
                mask_right=mask_right,
                kv_len=kv_len,
            )
            final_state = yield next_state
        return final_state

    state = _init
    if mask_left:
        state = _run_tiles(
            state,
            start_tile,
            clean_lo,
            mask_left=mask_left,
            mask_right=mask_right,
            kv_len=None,
        )
    state = _run_tiles(
        state, clean_lo, clean_hi, mask_left=None, mask_right=None, kv_len=None
    )
    state = _run_tiles(
        state,
        clean_hi,
        fx.Int32(num_tiles) + fx.Int32(1),
        mask_left=mask_left,
        mask_right=mask_right,
        kv_len=kv_len,
    )
    final = state
    if not _lag_sm:
        # Partner for the lagging half's prologue filler -- without it the two halves
        # disagree on the barrier count. It also orders this half's O stores against its
        # OWN last K reads: this half's last in-loop barrier is the PHASE one, which sits
        # BEFORE the gemm, so wave 1 could otherwise store O into final[0]'s K chunk while
        # wave 0 is still reading K out of it.
        _bare_barrier()

    # ========================================================================
    # Epilogue. The R q-tiles serialize through the same O ring.
    # ========================================================================
    o_mgr = _o_manager(
        v_hdim=v_hdim,
        gqa_ratio=gqa_ratio,
        num_q_tiles_per_wave=R,
        elem_dtype=elem_dtype,
    )
    _o_free = [
        fx.Int32(final[_SLOT_BASE + 2]),
        fx.Int32(final[_SLOT_BASE + 0]),
    ]
    o_lds_warp = (
        ((warp_idx & fx.Int32(1)) > fx.Int32(0)).select(_o_free[1], _o_free[0])
        + ((warp_idx >> fx.Int32(1)) & fx.Int32(1)) * fx.Int32(LDS_QO_BYTES)
        + (warp_idx >> fx.Int32(2)) * fx.Int32(_SPLIT_STRIDE)
    )
    # The trailing body's dead clamped tile copies are still in flight. HI's targets
    # final[1], a V chunk O never touches; LO's two K copies land in final[2] and final[1],
    # and final[2] IS an O chunk -- so the drain needs a rendezvous behind it, since a wave
    # only retires its own copies and O's chunk halves are shared by wave pairs.
    _kv_wait(*_kv_drain)
    _bare_barrier()
    _store_o_lse(
        o_mgr=o_mgr,
        m_list=[fx.Float32(final[qt * _QS + 0]) for qt in range(R)],
        d_list=[fx.Float32(final[qt * _QS + 1]) for qt in range(R)],
        o_list=[
            [fx.Vector(final[qt * _QS + 2 + dt]) for dt in range(d_tiles)]
            for qt in range(R)
        ],
        d_tiles=d_tiles,
        R=R,
        ptr_O=ptr_O,
        stride_o_seq=stride_o_seq,
        stride_o_head=stride_o_head,
        q_start=q_start,
        q_len=q_len,
        kv_head=kv_head,
        block_x=block_x,
        warp_idx=warp_idx,
        lane_idx=lane_idx,
        o_lds_warp=o_lds_warp,
        seq_idx=seq_idx,
        q_head_idx=q_head_idx,
        return_lse=return_lse,
        ptr_LSE=ptr_LSE,
        lse_base_elems=lse_base_elems,
        lse_num_records_bytes=lse_num_records_bytes,
        stride_lse_seq=stride_lse_seq,
        stride_lse_head=stride_lse_head,
    )


def _core_attention_one_kv_tile(
    *,
    qk_hdim,
    v_hdim,
    n_block,
    mask_left,
    mask_right,
    return_lse,
    has_sink,
    gqa_ratio,
    num_q_tiles_per_wave,
    ptr_O,
    ptr_Q,
    ptr_K,
    ptr_V,
    ptr_LSE,
    ptr_sink,
    softmax_scale,
    stride_q_seq,
    stride_k_seq,
    stride_v_seq,
    stride_o_seq,
    stride_q_head,
    stride_k_head,
    stride_v_head,
    stride_o_head,
    stride_lse_seq,
    stride_lse_head,
    lse_base_elems,
    lse_num_records_bytes,
    q_start,
    q_len,
    kv_start,
    kv_len,
    window_left,
    window_right,
    warp_idx,
    lds_base,
    elem_dtype,
):
    """Whole-attention compute for a workgroup whose band fits in ONE KV tile.

    Warp specialization buys nothing here: with a single tile there is no next tile to
    prefetch, so there is nothing for a producer half to run ahead on and no PV/QK
    ping-pong to anti-phase. All 8 waves run this one trace -- QK, softmax, PV -- and the
    only thing the wave's half still decides is which of K or V it copies from VRAM, a
    runtime branch rather than a second trace.

    Everything the pipelined core needs for a tile stream is gone: no loop, no slot
    rotation, no carried P/S hand-off, no clamped prefetch, no dead half-bodies. The one
    tile lands in physical slot 0 while Q sits in slot 1 and O stages through slots 1|2,
    so the tile is never overwritten and the whole body needs exactly ONE barrier -- the
    fence that publishes the tile.
    """
    lane_idx = _lane_id()
    kv_head, q_head_idx, seq_idx = _packed_tile_indices(
        gqa_ratio, warp_idx, lane_idx, num_q_tiles_per_wave
    )
    _q_scale = softmax_scale * fx.Float32(LOG2E)
    R = num_q_tiles_per_wave
    d_tiles = v_hdim // WMMA_M

    if USE_TDM_LOADER:
        q_mgr = QManager16bV2(
            qk_hdim=qk_hdim,
            gqa_ratio=gqa_ratio,
            num_waves=NUM_WAVES,
            q_tiles_per_wave=R,
            elem_dtype=elem_dtype,
        )
        k_mgr = KManager16bV2(
            qk_hdim=qk_hdim,
            n_block=n_block,
            num_waves=NUM_WAVES,
            elem_dtype=elem_dtype,
        )
        v_mgr = VManager16bV2(
            v_hdim=v_hdim, n_block=n_block, num_waves=NUM_WAVES, elem_dtype=elem_dtype
        )
    else:
        q_mgr = QManager16bV1(
            qk_hdim=qk_hdim,
            gqa_ratio=gqa_ratio,
            num_waves=NUM_WAVES,
            q_tiles_per_wave=R,
            elem_dtype=elem_dtype,
        )
        k_mgr = KManager16bV1(
            qk_hdim=qk_hdim,
            n_block=n_block,
            num_waves=NUM_WAVES,
            elem_dtype=elem_dtype,
        )
        v_mgr = VManager16bV1(
            v_hdim=v_hdim, n_block=n_block, num_waves=NUM_WAVES, elem_dtype=elem_dtype
        )
    _SPLIT_STRIDE = 6 * LDS_CHUNK_BYTES
    slot_bytes = 2 * LDS_CHUNK_BYTES
    for _who, _b in (("K", k_mgr.get_lds_size_in_byte()), ("V", v_mgr.get_lds_size_in_byte())):
        assert _b % KV_LDS_SPLITS == 0 and _b // KV_LDS_SPLITS <= LDS_CHUNK_BYTES, (
            f"{_who} split {_b // KV_LDS_SPLITS}B exceeds the {LDS_CHUNK_BYTES}B chunk "
            f"(qk_hdim={qk_hdim}, v_hdim={v_hdim}, n_block={n_block})"
        )
    assert q_mgr.warp_lds_size_in_byte() <= LDS_QO_BYTES

    # The tile is physical slot 0: K at chunks 0|6, V at 1|7.
    k_buf = lds_base
    v_buf = lds_base + fx.Int32(LDS_CHUNK_BYTES)
    assert KV_LDS_SPLITS == 2, "parity order needs a 2-way split"
    _odd = warp_idx & fx.Int32(1)
    _rd0 = _odd * fx.Int32(_SPLIT_STRIDE)
    _rd = [_rd0, fx.Int32(_SPLIT_STRIDE) - _rd0]
    _kv_swap_delta = _odd * fx.Int32(n_block // 2)

    def _split_bufs(b):
        return [b + o for o in _rd]

    _q_chunk_b = (warp_idx & fx.Int32(1)) ^ ((warp_idx >> fx.Int32(2)) & fx.Int32(1))
    q_lds_warp = (
        lds_base
        + fx.Int32(slot_bytes)
        + _q_chunk_b * fx.Int32(_SPLIT_STRIDE)
        + (warp_idx >> fx.Int32(1)) * fx.Int32(LDS_QO_BYTES)
    )
    q_mgr.load_q_to_vgpr_part1(
        ptr_Q=ptr_Q,
        stride_q_seq=stride_q_seq,
        stride_q_head=stride_q_head,
        q_start=q_start,
        q_len=q_len,
        kv_head=kv_head,
        block_x=_m_tile_idx(),
        warp_idx=warp_idx,
        lane_idx=lane_idx,
        ptr_lds_warp=q_lds_warp,
    )

    block_x, causal_off, kv_len_wg, _ = _wg_kv_span(
        n_block=n_block,
        mask_right=mask_right,
        gqa_ratio=gqa_ratio,
        num_q_tiles_per_wave=R,
        q_len=q_len,
        kv_len=kv_len,
        window_right=window_right,
    )

    # Both descriptors are pure; only the copy one of them issues is warp-dependent.
    _kv_ctx = ProducerCtx(
        producer_warp=warp_idx % fx.Int32(KV_PRODUCER_WARPS),
        num_producer_warps=KV_PRODUCER_WARPS,
        lane_idx=lane_idx,
    )
    k_desc = k_mgr.load_descriptor(
        ptr_lds=[k_buf, k_buf + fx.Int32(_SPLIT_STRIDE)],
        ptr_src=ptr_K,
        stride_seq=stride_k_seq,
        stride_head=stride_k_head,
        head=kv_head,
        row0=kv_start,
        valid=kv_len_wg,
        ctx=_kv_ctx,
    )
    v_desc = v_mgr.load_descriptor(
        ptr_lds=[v_buf, v_buf + fx.Int32(_SPLIT_STRIDE)],
        ptr_src=ptr_V,
        stride_seq=stride_v_seq,
        stride_head=stride_v_head,
        head=kv_head,
        row0=kv_start,
        valid=kv_len_wg,
        ctx=_kv_ctx,
    )

    @flyc.jit
    def _issue_kv(is_lo):
        if is_lo:
            k_desc.async_load()
        else:
            v_desc.async_load()

    _is_lo = warp_idx < fx.Int32(NUM_WAVES // 2)
    _num_tdm = max(k_desc.tensorcnt, v_desc.tensorcnt) or -1
    _num_async = max(k_desc.asynccnt, v_desc.asynccnt) or -1
    if USE_TDM_LOADER:
        # The tile misses Q's chunks entirely, so it goes out before Q is read and its
        # global latency overlaps Q's. Waves differ in how many copies they issued, so
        # part2 drains only down to the count EVERY wave has in flight.
        _issue_kv(_is_lo)
        q_frags = q_mgr.load_q_to_vgpr_part2(
            scale=_q_scale,
            skip_tensorcnt=min(k_desc.tensorcnt, v_desc.tensorcnt),
        )
    else:
        # V1 puts Q's own stage on asynccnt and part2 waits a depth counted over Q alone.
        q_frags = q_mgr.load_q_to_vgpr_part2(scale=_q_scale)
        _issue_kv(_is_lo)
    # The only barrier in the body: a wave's counters bound its own copies, so this is
    # what publishes the tile to the other seven.
    _kv_fence(*_kv_drain_depths(_num_tdm, _num_async))

    k_lds = k_mgr.ds_load_ptrs(ptr_lds=_split_bufs(k_buf), lane_idx=lane_idx)
    v_lds = v_mgr.ds_load_ptrs(ptr_lds=_split_bufs(v_buf), lane_idx=lane_idx)

    def _k_emit(j):
        return k_mgr.load_one_to_reg(k_lds, j)

    def _v_emit(j):
        return v_mgr.load_one_to_reg(v_lds, j)

    _, s_list = _pv_qk_gemm(
        v_emit=None,
        k_emit=_k_emit,
        p_list=[[] for _ in range(R)],
        q_frags_list=q_frags,
        v_hdim=v_hdim,
        n_block=n_block,
        warp_type=WarpType.LO,
        o_acc_list=[[] for _ in range(R)],
        phase="qk",
    )

    # PV's ring head goes out before softmax so its LDS latency hides under the VALU.
    _NP = _ring_num_prefetch(v_mgr.num_ds_loads() // 2, PVQK_RING, PVQK_LAG)
    rocdl.sched_barrier(0)
    pv_head = []
    for j in range(_NP):
        pv_head.append(_v_emit(j))
        if len(pv_head) % 2 == 0:
            rocdl.sched_barrier(0)
    rocdl.sched_barrier(0)

    if has_sink:
        num_heads_q = gpu.grid_dim.y * fx.Int32(gqa_ratio)
        m_prev = [
            _load_sink_logit(ptr_sink, q_head_idx[qt], num_heads_q) * fx.Float32(LOG2E)
            for qt in range(R)
        ]
        d_prev = [fx.Float32(1.0) for _ in range(R)]
    else:
        m_prev = [fx.Float32(BIG_NEG) for _ in range(R)]
        d_prev = [fx.Float32(0.0) for _ in range(R)]
    # This tile is both the first and the last, so it always carries the kv_len tail and
    # O needs no rescale: the seed is 0 and corr*0 == 0 whatever the running max does.
    p_f32, m_new_list, d_new_list, _corr, _masks = _softmax(
        s_list=s_list,
        m_prev_list=m_prev,
        d_prev_list=d_prev,
        lane_idx=lane_idx,
        n_block=n_block,
        kv_pos_base=fx.Int32(0),
        kv_swap_delta=_kv_swap_delta,
        q_max_list=[
            seq_idx[qt] + causal_off + window_right if mask_right else None
            for qt in range(R)
        ],
        q_min_list=[
            seq_idx[qt] + causal_off - window_left if mask_left else None
            for qt in range(R)
        ],
        kv_len=kv_len,
    )
    p_list = _p_to_elem(p_f32, elem_dtype)
    _keepalive([v for pt in p_list for v in pt])
    o_list, _ = _pv_qk_gemm(
        v_emit=_v_emit,
        k_emit=None,
        p_list=p_list,
        q_frags_list=q_frags,
        v_hdim=v_hdim,
        n_block=n_block,
        warp_type=WarpType.LO,
        o_acc_list=None,
        head=pv_head,
        phase="pv",
    )

    # O stages through physical slots 1 and 2 -- Q's chunks, dead since the prologue, and
    # never touched by the tile in slot 0. Nothing is in flight and no wave can still be
    # reading them, so the epilogue needs neither a drain nor a rendezvous.
    o_lds_warp = (
        (_odd > fx.Int32(0)).select(
            lds_base + fx.Int32(2 * slot_bytes), lds_base + fx.Int32(slot_bytes)
        )
        + ((warp_idx >> fx.Int32(1)) & fx.Int32(1)) * fx.Int32(LDS_QO_BYTES)
        + (warp_idx >> fx.Int32(2)) * fx.Int32(_SPLIT_STRIDE)
    )
    _store_o_lse(
        o_mgr=_o_manager(
            v_hdim=v_hdim,
            gqa_ratio=gqa_ratio,
            num_q_tiles_per_wave=R,
            elem_dtype=elem_dtype,
        ),
        m_list=m_new_list,
        d_list=d_new_list,
        o_list=o_list,
        d_tiles=d_tiles,
        R=R,
        ptr_O=ptr_O,
        stride_o_seq=stride_o_seq,
        stride_o_head=stride_o_head,
        q_start=q_start,
        q_len=q_len,
        kv_head=kv_head,
        block_x=block_x,
        warp_idx=warp_idx,
        lane_idx=lane_idx,
        o_lds_warp=o_lds_warp,
        seq_idx=seq_idx,
        q_head_idx=q_head_idx,
        return_lse=return_lse,
        ptr_LSE=ptr_LSE,
        lse_base_elems=lse_base_elems,
        lse_num_records_bytes=lse_num_records_bytes,
        stride_lse_seq=stride_lse_seq,
        stride_lse_head=stride_lse_head,
    )


def _o_manager(*, v_hdim, gqa_ratio, num_q_tiles_per_wave, elem_dtype):
    """The O epilogue's LDS staging manager, plus the clearance check both cores need.

    Two waves share a chunk at 0 and LDS_QO_BYTES, and the upper one must stop short of
    the next chunk -- that clearance is what keeps O off the V rows a dead carried ring
    head may still be reading."""
    _OMgr = {"v1": OManager16bV1, "v2": OManager16bV2, "v3": OManager16bV3}[O_VARIANT]
    o_mgr = _OMgr(
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


def _store_o_lse(
    *,
    o_mgr,
    m_list,
    d_list,
    o_list,
    d_tiles,
    R,
    ptr_O,
    stride_o_seq,
    stride_o_head,
    q_start,
    q_len,
    kv_head,
    block_x,
    warp_idx,
    lane_idx,
    o_lds_warp,
    seq_idx,
    q_head_idx,
    return_lse,
    ptr_LSE,
    lse_base_elems,
    lse_num_records_bytes,
    stride_lse_seq,
    stride_lse_head,
):
    """Normalize, stage and store O (and optionally LSE) for this wave's R q-tiles.

    Shared tail of both compute cores; ``o_lds_warp`` is the caller's choice of which
    finished KV chunk this wave stages through."""
    for qt in range(R):
        # Normalize this q-tile's O by its running denom d, then reshape+store to VRAM.
        # o_final[dt] lane l elem si = sum_kv P[q,kv] V[kv, dt*16+(l//16)*8+si]
        # (unnormalized); divide by the per-query denom d (peer-consistent across the
        # lane pair) to finish softmax. OManager16b masks rows with seq >= q_len.
        d_final = d_list[qt]
        o_final = o_list[qt]
        # Fully-masked row (d_final==0): 1/0=inf, o_final=0, 0*inf=NaN -> guard to O=0.
        inv = (d_final > fx.Float32(0.0)).select(
            fx.Float32(1.0) / d_final, fx.Float32(0.0)
        )
        inv_vec = fx.Vector.from_elements([inv], fx.Float32).broadcast_to(8)
        o_norm = [o_final[dt] * inv_vec for dt in range(d_tiles)]
        if qt > 0:
            rocdl.s_wait_dscnt(0)  # drain prev q-tile's O ring/DS ops before reuse
        o_mgr.store_o_to_vram(
            ptr_O=ptr_O,
            o_base_elems=fx.Int32(0),
            stride_o_seq=stride_o_seq,
            stride_o_head=stride_o_head,
            q_start=q_start,
            q_len=q_len,
            kv_head=kv_head,
            block_x=block_x,
            warp_idx=warp_idx,
            lane_idx=lane_idx,
            ptr_lds_warp=o_lds_warp,
            o_frags=o_norm,
            qtile=qt,
        )

    # ---- LSE store (optional). LSE = (m_final + log2(d_final)) / LOG2E: m is carried in
    # log2 units of the scaled score and d is domain-free, so one multiply converts both,
    # matching torch.logsumexp(scale * Q @ K^T, dim=kv). The lane pair (l, l^16) holds
    # each query identically, so store once from the khalf==0 lanes, masked by
    # seq < q_len; buffer_store redirects mask-drops to byte 0x7FFFFFFF.
    if return_lse:
        khalf0 = (lane_idx // fx.Int32(WMMA_M)) == fx.Int32(0)
        lse_rsrc = buffer_ops.create_buffer_resource(
            ptr_LSE, num_records_bytes=lse_num_records_bytes
        )
        for qt in range(R):
            m_final = m_list[qt]
            d_final = d_list[qt]
            # fx.log2 lowers to the HW v_log_f32 (base-2), matching m's log2 domain.
            lse_val = (m_final + fx.log2(d_final)) * fx.Float32(1.0 / LOG2E)
            lse_mask = khalf0 & (seq_idx[qt] < q_len)
            lse_off_el = (
                lse_base_elems
                + seq_idx[qt] * stride_lse_seq
                + q_head_idx[qt] * stride_lse_head
            )
            # Pre-mask the offset (OOB rows -> 0x7fffffff) and pass mask=None so the
            # store maps 1:1 to a single buffer_store with masking already SSA-visible.
            lse_off_masked = lse_mask.select(
                lse_off_el * fx.Int32(4), fx.Int32(0x7FFFFFFF)
            )
            buffer_ops.buffer_store(
                lse_val, lse_rsrc, lse_off_masked, mask=None, offset_is_bytes=True
            )


def _zero_fill_attention(
    *,
    v_hdim,
    gqa_ratio,
    num_q_tiles_per_wave,
    return_lse,
    has_sink,
    ptr_sink,
    ptr_O,
    ptr_LSE,
    stride_o_seq,
    stride_o_head,
    stride_lse_seq,
    stride_lse_head,
    lse_num_records_bytes,
    q_start,
    q_len,
    elem_dtype,
):
    """q_len>0 with kv_len==0 (cross-attention): softmax over an empty KV set, so O=0 for
    this WG's valid query rows and LSE=-inf -- or LSE=sink[head] with a sink, exp(sink)
    being the only surviving term. Flat coalesced b128 write, no WMMA layout."""
    BLOCK_M = _block_m(num_q_tiles_per_wave)
    tid = _warp_id() * fx.Int32(WAVE_SIZE) + _lane_id()
    kv_head = fx.Int32(gpu.block_id("y"))
    row0 = _m_tile_idx() * fx.Int32(BLOCK_M)
    g = fx.Int32(gqa_ratio)
    _CH = 8  # bf16 per b128 store
    cpr = v_hdim // _CH  # b128 chunks per O row

    # i64: an i32 product (large total_q * stride) can overflow negative, then the
    # descriptor sign-extends it to a huge bound, defeating the 0x7FFFFFFF OOB drop.
    o_num_records_bytes = (
        fx.Int64(q_start + q_len) * fx.Int64(stride_o_seq) * fx.Int64(2)
    )
    o_rsrc = buffer_ops.create_buffer_resource(
        ptr_O, num_records_bytes=o_num_records_bytes
    )
    zero_o = fx.Vector.filled(_CH, 0.0, elem_dtype)
    for r in range(BLOCK_M * cpr // BLOCK_SIZE):
        cix = fx.Int32(r * BLOCK_SIZE) + tid  # flat b128-chunk index this round
        prow = row0 + cix // fx.Int32(cpr)
        d = (cix % fx.Int32(cpr)) * fx.Int32(_CH)
        seq = prow // g
        head = kv_head * g + prow % g
        off = (q_start + seq) * stride_o_seq + head * stride_o_head + d
        off_masked = (seq < q_len).select(off * fx.Int32(2), fx.Int32(0x7FFFFFFF))
        buffer_ops.buffer_store(
            zero_o, o_rsrc, off_masked, mask=None, offset_is_bytes=True
        )

    if return_lse:
        lse_rsrc = buffer_ops.create_buffer_resource(
            ptr_LSE, num_records_bytes=lse_num_records_bytes
        )
        prow = row0 + tid  # one LSE per packed row, one row per thread
        seq = prow // g
        head = kv_head * g + prow % g
        if has_sink:
            num_heads_q = gpu.grid_dim.y * g
            lse_val = _load_sink_logit(ptr_sink, head, num_heads_q)
        else:
            lse_val = fx.Float32(float("-inf"))
        off = (q_start + seq) * stride_lse_seq + head * stride_lse_head
        off_masked = (seq < q_len).select(off * fx.Int32(4), fx.Int32(0x7FFFFFFF))
        if BLOCK_M < BLOCK_SIZE:
            off_masked = (tid < fx.Int32(BLOCK_M)).select(
                off_masked, fx.Int32(0x7FFFFFFF)
            )
        buffer_ops.buffer_store(
            lse_val, lse_rsrc, off_masked, mask=None, offset_is_bytes=True
        )


# ============================================================================
# Builder — one device kernel per (layout, config)
# ============================================================================


def kv_split_bytes(qk_hdim, v_hdim, n_block, elem_dtype):
    """Bytes one half of a 2-way-split K and V tile occupies, straight from the active
    managers -- n_block//KV_LDS_SPLITS rows of hdim*sizeof(elem) plus their per-row pad
    (16 B for K, 32 B for V)."""
    k_cls = KManager16bV2 if USE_TDM_LOADER else KManager16bV1
    v_cls = VManager16bV2 if USE_TDM_LOADER else VManager16bV1
    k_mgr = k_cls(
        qk_hdim=qk_hdim, n_block=n_block, num_waves=NUM_WAVES, elem_dtype=elem_dtype
    )
    v_mgr = v_cls(
        v_hdim=v_hdim, n_block=n_block, num_waves=NUM_WAVES, elem_dtype=elem_dtype
    )
    return (
        k_mgr.get_lds_size_in_byte() // KV_LDS_SPLITS,
        v_mgr.get_lds_size_in_byte() // KV_LDS_SPLITS,
    )


def pick_n_block(qk_hdim, v_hdim, elem_dtype):
    """Widest n_block from N_BLOCK_PREF whose split K and V tiles each still fit one
    LDS_CHUNK_BYTES chunk and whose hdim is within N_BLOCK_WIDE_MAX_QK_HDIM. At bf16 that
    is 128 for qk_hdim 128 and 64 for 192/256."""
    for nb in N_BLOCK_PREF:
        if nb > DEFAULT_N_BLOCK and qk_hdim > N_BLOCK_WIDE_MAX_QK_HDIM:
            continue
        if max(kv_split_bytes(qk_hdim, v_hdim, nb, elem_dtype)) <= LDS_CHUNK_BYTES:
            return nb
    raise ValueError(
        f"no n_block in {N_BLOCK_PREF} fits the {LDS_CHUNK_BYTES}B chunk "
        f"(qk_hdim={qk_hdim}, v_hdim={v_hdim})"
    )


@functools.cache
def build_fmha_fwd_prefill_a16w16_m32x8(
    *,
    layout: str = "thd",
    qk_hdim: int = DEFAULT_QK_HDIM,
    v_hdim: int = DEFAULT_V_HDIM,
    n_block: int | None = None,
    dtype_str: str = DEFAULT_DTYPE,
    mask_left: bool = False,
    mask_right: bool = False,
    one_kv_tile: bool = False,
    return_lse: bool = False,
    has_sink: bool = False,
    gqa_ratio: int = 1,
    num_q_tiles_per_wave: int = WMMA_ROW_PER_WAVE,
):
    """Build the m32x8 device kernel for a given layout ("thd" varlen or "bshd" batched)
    plus config; every parameter here is compile-time and baked into the trace.
    ``gqa_ratio`` (= ``nheads_q // nheads_kv``) is among them so the per-lane ``% / //``
    fold to shift/and when it is a power of two.
    """
    assert layout in ("thd", "bshd"), f"layout must be thd|bshd, got {layout!r}"
    # qk_hdim (D_qk) is a WMMA_K multiple; v_hdim (D_v) a WMMA_M multiple. Independent.
    assert (
        qk_hdim in SUPPORTED_QK_HDIM and v_hdim in SUPPORTED_V_HDIM
    ), f"supports qk_hdim in {SUPPORTED_QK_HDIM} x v_hdim in {SUPPORTED_V_HDIM}, got {qk_hdim}/{v_hdim}"
    assert (
        dtype_str in _DTYPE_MAP
    ), f"dtype_str must be in {list(_DTYPE_MAP)}, got {dtype_str!r}"
    ELEM_DTYPE = _DTYPE_MAP[dtype_str]
    assert gqa_ratio >= 1, f"gqa_ratio must be >= 1, got {gqa_ratio}"
    assert (
        num_q_tiles_per_wave in NUM_Q_TILES_CHOICES
    ), f"num_q_tiles_per_wave must be in {NUM_Q_TILES_CHOICES}, got {num_q_tiles_per_wave}"
    if n_block is None:
        n_block = pick_n_block(qk_hdim, v_hdim, ELEM_DTYPE)
    assert (
        n_block in N_BLOCK_CHOICES
    ), f"n_block must be in {N_BLOCK_CHOICES}, got {n_block}"

    QK_HDIM = qk_hdim
    V_HDIM = v_hdim
    N_BLOCK = int(n_block)
    MASK_LEFT = bool(mask_left)
    MASK_RIGHT = bool(mask_right)
    ONE_KV_TILE = bool(one_kv_tile)
    # thd cannot bound a WG's tile count at compile time -- max_seqlen_k is a batch max,
    # not a per-sequence one -- so when the gate is off it emits BOTH cores and each WG
    # dispatches on its own count. bshd's seq_len_k is exact, so the gate settles it.
    EMIT_BOTH_CORES = layout == "thd" and not ONE_KV_TILE
    RET_LSE = bool(return_lse)
    HAS_SINK = bool(has_sink)
    GQA_RATIO = int(gqa_ratio)
    NUM_Q_TILES = int(num_q_tiles_per_wave)

    if layout == "thd":

        @flyc.kernel(known_block_size=[BLOCK_SIZE, 1, 1])
        def kn_fmha_fwd_prefill_a16w16_m32x8_thd(
            ptr_O: fx.Pointer,
            ptr_Q: fx.Pointer,
            ptr_K: fx.Pointer,
            ptr_V: fx.Pointer,
            ptr_LSE: fx.Pointer,
            ptr_sink: fx.Pointer,
            ptr_cu_seqlens_q: fx.Pointer,
            ptr_cu_seqlens_k: fx.Pointer,
            softmax_scale: fx.Float32,
            stride_q_seq: fx.Int32,
            stride_k_seq: fx.Int32,
            stride_v_seq: fx.Int32,
            stride_o_seq: fx.Int32,
            stride_q_head: fx.Int32,
            stride_k_head: fx.Int32,
            stride_v_head: fx.Int32,
            stride_o_head: fx.Int32,
            stride_lse_seq: fx.Int32,
            stride_lse_head: fx.Int32,
            window_left: fx.Int32,
            window_right: fx.Int32,
            max_seqlen_q: fx.Int32,
            max_seqlen_k: fx.Int32,
        ):
            """Varlen THD entry: this batch's token ranges come from cu_seqlens
            (batch = grid.z)."""
            batch = fx.Int32(gpu.block_id("z"))
            q_start, q_end = _load_seqlen_pair(ptr_cu_seqlens_q, batch)
            kv_start, kv_end = _load_seqlen_pair(ptr_cu_seqlens_k, batch)
            q_len = q_end - q_start
            kv_len = kv_end - kv_start

            # LSE is [nheads_q, total_q]. Bound the buffer resource by the last element
            # this batch can touch over BOTH axes; a seq-only bound is exact only when
            # seq is the major axis.
            num_heads_q = gpu.grid_dim.y * fx.Int32(GQA_RATIO)
            lse_base_elems = q_start * stride_lse_seq
            lse_num_records_bytes = (
                fx.Int64(q_start + q_len - fx.Int32(1)) * fx.Int64(stride_lse_seq)
                + fx.Int64(num_heads_q - fx.Int32(1)) * fx.Int64(stride_lse_head)
                + fx.Int64(1)
            ) * fx.Int64(4)

            # An empty batch must NOT enter the core: kv_len==0 leaves the softmax denom
            # at 0 and the epilogue would write O/0 = NaN to that batch's query rows,
            # while q_len==0 has no rows to write at all.
            if (q_len > fx.Int32(0)) & (kv_len > fx.Int32(0)):
                _ca_kw = {
                    "qk_hdim": QK_HDIM,
                    "v_hdim": V_HDIM,
                    "n_block": N_BLOCK,
                    "mask_left": MASK_LEFT,
                    "mask_right": MASK_RIGHT,
                    "return_lse": RET_LSE,
                    "has_sink": HAS_SINK,
                    "gqa_ratio": GQA_RATIO,
                    "num_q_tiles_per_wave": NUM_Q_TILES,
                    "ptr_O": ptr_O,
                    "ptr_Q": ptr_Q,
                    "ptr_K": ptr_K,
                    "ptr_V": ptr_V,
                    "ptr_LSE": ptr_LSE,
                    "ptr_sink": ptr_sink,
                    "softmax_scale": softmax_scale,
                    "stride_q_seq": stride_q_seq,
                    "stride_k_seq": stride_k_seq,
                    "stride_v_seq": stride_v_seq,
                    "stride_o_seq": stride_o_seq,
                    "stride_q_head": stride_q_head,
                    "stride_k_head": stride_k_head,
                    "stride_v_head": stride_v_head,
                    "stride_o_head": stride_o_head,
                    "stride_lse_seq": stride_lse_seq,
                    "stride_lse_head": stride_lse_head,
                    "lse_base_elems": lse_base_elems,
                    "lse_num_records_bytes": lse_num_records_bytes,
                    "q_start": q_start,
                    "q_len": q_len,
                    "kv_start": kv_start,
                    "kv_len": kv_len,
                    "window_left": window_left,
                    "window_right": window_right,
                    "elem_dtype": ELEM_DTYPE,
                }
                # Which core(s) this build emits. ONE_KV_TILE means the host proved every WG
                # sees a single tile. Otherwise bshd knows seq_len_k exactly and needs only the
                # pipelined core, while thd does not: a ragged batch mixes short sequences with
                # long ones, so it carries both and each WG picks its own at runtime.
                lds_base = _alloc_lds()
                warp_idx = _warp_id()

                def _run_one_kv_tile():
                    _core_attention_one_kv_tile(
                        warp_idx=warp_idx, lds_base=lds_base, **_ca_kw
                    )

                def _run_multi_kv_tiles():
                    # Warp specialization: LO (waves 0..N/2-1) vs HI (N/2..N-1).
                    if warp_idx // fx.Int32(NUM_WAVES // 2) == fx.Int32(0):
                        _core_attention_multi_kv_tiles(
                            warp_idx=warp_idx,
                            warp_type=WarpType.LO,
                            lds_base=lds_base,
                            **_ca_kw,
                        )
                    else:
                        _core_attention_multi_kv_tiles(
                            warp_idx=warp_idx,
                            warp_type=WarpType.HI,
                            lds_base=lds_base,
                            **_ca_kw,
                        )

                if ONE_KV_TILE:
                    _run_one_kv_tile()
                elif not EMIT_BOTH_CORES:
                    _run_multi_kv_tiles()
                else:
                    _, _, _, _num_tiles = _wg_kv_span(
                        n_block=N_BLOCK,
                        mask_right=MASK_RIGHT,
                        gqa_ratio=GQA_RATIO,
                        num_q_tiles_per_wave=NUM_Q_TILES,
                        q_len=q_len,
                        kv_len=kv_len,
                        window_right=window_right,
                    )
                    # Multi first so the pipelined bodies keep the low code offsets.
                    if _num_tiles > fx.Int32(1):
                        _run_multi_kv_tiles()
                    else:
                        _run_one_kv_tile()
            elif q_len > fx.Int32(0):
                # Cross-attention tail: q_len>0 but kv_len==0 -> O=0, LSE=-inf (or sink).
                _zero_fill_attention(
                    v_hdim=V_HDIM,
                    gqa_ratio=GQA_RATIO,
                    num_q_tiles_per_wave=NUM_Q_TILES,
                    return_lse=RET_LSE,
                    has_sink=HAS_SINK,
                    ptr_sink=ptr_sink,
                    ptr_O=ptr_O,
                    ptr_LSE=ptr_LSE,
                    stride_o_seq=stride_o_seq,
                    stride_o_head=stride_o_head,
                    stride_lse_seq=stride_lse_seq,
                    stride_lse_head=stride_lse_head,
                    lse_num_records_bytes=lse_num_records_bytes,
                    q_start=q_start,
                    q_len=q_len,
                    elem_dtype=ELEM_DTYPE,
                )

        return kn_fmha_fwd_prefill_a16w16_m32x8_thd

    @flyc.kernel(known_block_size=[BLOCK_SIZE, 1, 1])
    def kn_fmha_fwd_prefill_a16w16_m32x8_bshd(
        ptr_O: fx.Pointer,
        ptr_Q: fx.Pointer,
        ptr_K: fx.Pointer,
        ptr_V: fx.Pointer,
        ptr_LSE: fx.Pointer,
        ptr_sink: fx.Pointer,
        softmax_scale: fx.Float32,
        stride_q_seq: fx.Int32,
        stride_k_seq: fx.Int32,
        stride_v_seq: fx.Int32,
        stride_o_seq: fx.Int32,
        stride_q_head: fx.Int32,
        stride_k_head: fx.Int32,
        stride_v_head: fx.Int32,
        stride_o_head: fx.Int32,
        stride_lse_seq: fx.Int32,
        stride_lse_head: fx.Int32,
        stride_lse_batch: fx.Int32,
        window_left: fx.Int32,
        window_right: fx.Int32,
        seq_len_q: fx.Int32,
        seq_len_k: fx.Int32,
    ):
        """Batched BSHD entry: uniform ``seq_len_q``/``seq_len_k`` scalars replace
        cu_seqlens, so nothing is transient and this path is CUDA-graph safe. Token base
        is batch_idx * seq_len (batch = grid.z)."""
        batch = fx.Int32(gpu.block_id("z"))

        # LSE is [B, nheads_q, seq_q]: base = batch*stride_lse_batch; every valid
        # element offset is < base + stride_lse_batch (< the 0x7FFFFFFF drop).
        lse_base_elems = batch * stride_lse_batch
        lse_num_records_bytes = fx.Int64(lse_base_elems + stride_lse_batch) * fx.Int64(
            4
        )

        _ca_kw = {
            "qk_hdim": QK_HDIM,
            "v_hdim": V_HDIM,
            "n_block": N_BLOCK,
            "mask_left": MASK_LEFT,
            "mask_right": MASK_RIGHT,
            "return_lse": RET_LSE,
            "has_sink": HAS_SINK,
            "gqa_ratio": GQA_RATIO,
            "num_q_tiles_per_wave": NUM_Q_TILES,
            "ptr_O": ptr_O,
            "ptr_Q": ptr_Q,
            "ptr_K": ptr_K,
            "ptr_V": ptr_V,
            "ptr_LSE": ptr_LSE,
            "ptr_sink": ptr_sink,
            "softmax_scale": softmax_scale,
            "stride_q_seq": stride_q_seq,
            "stride_k_seq": stride_k_seq,
            "stride_v_seq": stride_v_seq,
            "stride_o_seq": stride_o_seq,
            "stride_q_head": stride_q_head,
            "stride_k_head": stride_k_head,
            "stride_v_head": stride_v_head,
            "stride_o_head": stride_o_head,
            "stride_lse_seq": stride_lse_seq,
            "stride_lse_head": stride_lse_head,
            "lse_base_elems": lse_base_elems,
            "lse_num_records_bytes": lse_num_records_bytes,
            "q_start": batch * seq_len_q,
            "q_len": seq_len_q,
            "kv_start": batch * seq_len_k,
            "kv_len": seq_len_k,
            "window_left": window_left,
            "window_right": window_right,
            "elem_dtype": ELEM_DTYPE,
        }
        # Which core(s) this build emits. ONE_KV_TILE means the host proved every WG
        # sees a single tile. Otherwise bshd knows seq_len_k exactly and needs only the
        # pipelined core, while thd does not: a ragged batch mixes short sequences with
        # long ones, so it carries both and each WG picks its own at runtime.
        lds_base = _alloc_lds()
        warp_idx = _warp_id()

        def _run_one_kv_tile():
            _core_attention_one_kv_tile(
                warp_idx=warp_idx, lds_base=lds_base, **_ca_kw
            )

        def _run_multi_kv_tiles():
            # Warp specialization: LO (waves 0..N/2-1) vs HI (N/2..N-1).
            if warp_idx // fx.Int32(NUM_WAVES // 2) == fx.Int32(0):
                _core_attention_multi_kv_tiles(
                    warp_idx=warp_idx,
                    warp_type=WarpType.LO,
                    lds_base=lds_base,
                    **_ca_kw,
                )
            else:
                _core_attention_multi_kv_tiles(
                    warp_idx=warp_idx,
                    warp_type=WarpType.HI,
                    lds_base=lds_base,
                    **_ca_kw,
                )

        if ONE_KV_TILE:
            _run_one_kv_tile()
        elif not EMIT_BOTH_CORES:
            _run_multi_kv_tiles()
        else:
            _, _, _, _num_tiles = _wg_kv_span(
                n_block=N_BLOCK,
                mask_right=MASK_RIGHT,
                gqa_ratio=GQA_RATIO,
                num_q_tiles_per_wave=NUM_Q_TILES,
                q_len=q_len,
                kv_len=kv_len,
                window_right=window_right,
            )
            # Multi first so the pipelined bodies keep the low code offsets.
            if _num_tiles > fx.Int32(1):
                _run_multi_kv_tiles()
            else:
                _run_one_kv_tile()

    return kn_fmha_fwd_prefill_a16w16_m32x8_bshd


# ============================================================================
# Launch wrappers + host entries
# ============================================================================

_launch_fns = (
    {}
)  # {(layout, mask_left, mask_right, return_lse, has_sink, gqa_ratio): fn}


def _pick_one_kv_tile(max_seqlen_k, qk_hdim, v_hdim, dtype_str):
    """True when the launch bounds every WG to a single KV tile.

    kv_len_wg is floored at 1 and never exceeds kv_len, so max_seqlen_k <= n_block makes
    num_tiles == 1 everywhere -- causal and windowed included, since both only clip it.
    The kernel then drops the left and clean sub-loops, which cannot run.
    """
    n_block = pick_n_block(qk_hdim, v_hdim, _DTYPE_MAP[dtype_str])
    return int(max_seqlen_k) <= n_block


def _pick_num_q_tiles(num_kv_heads, batch, max_seqlen_q, gqa_ratio):
    """Smallest Q tile whose grid still fits one workgroup wave; the largest otherwise.

    The kernel is LDS-bound to one workgroup per CU, and halving BLOCK_M doubles KV
    traffic, so a smaller tile only pays while the bigger one leaves CUs idle.
    """
    rows = int(max_seqlen_q) * int(gqa_ratio)
    per_tile = int(num_kv_heads) * int(batch)
    resident = get_cu_num() * NUM_WGS_PER_CU
    for num_tiles in sorted(NUM_Q_TILES_CHOICES):
        block_m = _block_m(num_tiles)
        if ((rows + block_m - 1) // block_m) * per_tile <= resident:
            return num_tiles
    return max(NUM_Q_TILES_CHOICES)


def _ensure_thd_kernel(
    mask_left: bool,
    mask_right: bool,
    one_kv_tile: bool,
    return_lse: bool,
    has_sink: bool,
    gqa_ratio: int,
    qk_hdim: int = DEFAULT_QK_HDIM,
    v_hdim: int = DEFAULT_V_HDIM,
    dtype_str: str = DEFAULT_DTYPE,
    num_q_tiles_per_wave: int = WMMA_ROW_PER_WAVE,
):
    key = (
        "thd",
        bool(mask_left),
        bool(mask_right),
        bool(return_lse),
        bool(has_sink),
        int(gqa_ratio),
        int(qk_hdim),
        int(v_hdim),
        str(dtype_str),
        int(num_q_tiles_per_wave),
        bool(one_kv_tile),
    )
    if key in _launch_fns:
        return
    kernel = build_fmha_fwd_prefill_a16w16_m32x8(
        layout="thd",
        qk_hdim=qk_hdim,
        v_hdim=v_hdim,
        mask_left=mask_left,
        mask_right=mask_right,
        one_kv_tile=one_kv_tile,
        return_lse=return_lse,
        has_sink=has_sink,
        gqa_ratio=gqa_ratio,
        dtype_str=dtype_str,
        num_q_tiles_per_wave=num_q_tiles_per_wave,
    )
    block_m = _block_m(num_q_tiles_per_wave)

    @flyc.jit
    def _launch(
        ptr_O: fx.Pointer,
        ptr_Q: fx.Pointer,
        ptr_K: fx.Pointer,
        ptr_V: fx.Pointer,
        ptr_LSE: fx.Pointer,
        ptr_sink: fx.Pointer,
        ptr_cu_seqlens_q: fx.Pointer,
        ptr_cu_seqlens_k: fx.Pointer,
        softmax_scale: fx.Float32,
        stride_q_seq: fx.Int32,
        stride_k_seq: fx.Int32,
        stride_v_seq: fx.Int32,
        stride_o_seq: fx.Int32,
        stride_q_head: fx.Int32,
        stride_k_head: fx.Int32,
        stride_v_head: fx.Int32,
        stride_o_head: fx.Int32,
        stride_lse_seq: fx.Int32,
        stride_lse_head: fx.Int32,
        window_left: fx.Int32,
        window_right: fx.Int32,
        max_seqlen_q: fx.Int32,
        max_seqlen_k: fx.Int32,
        num_heads_kv: fx.Int32,
        batch_size: fx.Int32,
        stream: fx.Stream,
    ):
        # 3D grid: x = tiles over (seq, q_head_in_group) per kv-head,
        #          y = kv_head, z = batch. block = 256 (8 waves x wave32).
        grid_x = fx.Index(
            fx.ceildiv(fx.Uint32(max_seqlen_q * gqa_ratio), fx.Uint32(block_m))
        )
        grid_y = fx.Index(num_heads_kv)
        grid_z = fx.Index(batch_size)

        launcher = kernel(
            ptr_O,
            ptr_Q,
            ptr_K,
            ptr_V,
            ptr_LSE,
            ptr_sink,
            ptr_cu_seqlens_q,
            ptr_cu_seqlens_k,
            softmax_scale,
            stride_q_seq,
            stride_k_seq,
            stride_v_seq,
            stride_o_seq,
            stride_q_head,
            stride_k_head,
            stride_v_head,
            stride_o_head,
            stride_lse_seq,
            stride_lse_head,
            window_left,
            window_right,
            max_seqlen_q,
            max_seqlen_k,
        )
        launcher.launch(
            grid=(grid_x, grid_y, grid_z),
            block=(BLOCK_SIZE, 1, 1),
            stream=stream,
        )

    _launch.compile_hints["llvm_options"] = {
        "amdgpu-expert-scheduling-mode": ENABLE_SCHED_MODE2,
        # "amdgpu-sched-strategy": "coexec",  # gfx1250 co-exec sched for warp specialization
    }
    _launch.compile_hints["waves_per_eu"] = 2
    _launch_fns[key] = _launch


def _ensure_bshd_kernel(
    mask_left: bool,
    mask_right: bool,
    one_kv_tile: bool,
    return_lse: bool,
    has_sink: bool,
    gqa_ratio: int,
    qk_hdim: int = DEFAULT_QK_HDIM,
    v_hdim: int = DEFAULT_V_HDIM,
    dtype_str: str = DEFAULT_DTYPE,
    num_q_tiles_per_wave: int = WMMA_ROW_PER_WAVE,
):
    key = (
        "bshd",
        bool(mask_left),
        bool(mask_right),
        bool(return_lse),
        bool(has_sink),
        int(gqa_ratio),
        int(qk_hdim),
        int(v_hdim),
        str(dtype_str),
        int(num_q_tiles_per_wave),
        bool(one_kv_tile),
    )
    if key in _launch_fns:
        return
    kernel = build_fmha_fwd_prefill_a16w16_m32x8(
        layout="bshd",
        qk_hdim=qk_hdim,
        v_hdim=v_hdim,
        mask_left=mask_left,
        mask_right=mask_right,
        one_kv_tile=one_kv_tile,
        return_lse=return_lse,
        has_sink=has_sink,
        gqa_ratio=gqa_ratio,
        dtype_str=dtype_str,
        num_q_tiles_per_wave=num_q_tiles_per_wave,
    )
    block_m = _block_m(num_q_tiles_per_wave)

    @flyc.jit
    def _launch(
        ptr_O: fx.Pointer,
        ptr_Q: fx.Pointer,
        ptr_K: fx.Pointer,
        ptr_V: fx.Pointer,
        ptr_LSE: fx.Pointer,
        ptr_sink: fx.Pointer,
        softmax_scale: fx.Float32,
        stride_q_seq: fx.Int32,
        stride_k_seq: fx.Int32,
        stride_v_seq: fx.Int32,
        stride_o_seq: fx.Int32,
        stride_q_head: fx.Int32,
        stride_k_head: fx.Int32,
        stride_v_head: fx.Int32,
        stride_o_head: fx.Int32,
        stride_lse_seq: fx.Int32,
        stride_lse_head: fx.Int32,
        stride_lse_batch: fx.Int32,
        window_left: fx.Int32,
        window_right: fx.Int32,
        seq_len_q: fx.Int32,
        seq_len_k: fx.Int32,
        num_heads_kv: fx.Int32,
        batch_size: fx.Int32,
        stream: fx.Stream,
    ):
        # 3D grid: x = tiles over (seq, q_head_in_group) per kv-head,
        #          y = kv_head, z = batch. block = 256 (8 waves x wave32).
        grid_x = fx.Index(
            fx.ceildiv(fx.Uint32(seq_len_q * gqa_ratio), fx.Uint32(block_m))
        )
        grid_y = fx.Index(num_heads_kv)
        grid_z = fx.Index(batch_size)

        launcher = kernel(
            ptr_O,
            ptr_Q,
            ptr_K,
            ptr_V,
            ptr_LSE,
            ptr_sink,
            softmax_scale,
            stride_q_seq,
            stride_k_seq,
            stride_v_seq,
            stride_o_seq,
            stride_q_head,
            stride_k_head,
            stride_v_head,
            stride_o_head,
            stride_lse_seq,
            stride_lse_head,
            stride_lse_batch,
            window_left,
            window_right,
            seq_len_q,
            seq_len_k,
        )
        launcher.launch(
            grid=(grid_x, grid_y, grid_z),
            block=(BLOCK_SIZE, 1, 1),
            stream=stream,
        )

    _launch.compile_hints["llvm_options"] = {
        "amdgpu-expert-scheduling-mode": ENABLE_SCHED_MODE2,
        # "amdgpu-sched-strategy": "coexec",  # gfx1250 co-exec sched for warp specialization
    }
    _launch.compile_hints["waves_per_eu"] = 2
    _launch_fns[key] = _launch


def flash_attn_varlen_m32x8(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    out=None,
    return_lse=False,
    sink=None,
    lse=None,
    num_q_tiles_per_wave=None,
):
    """Host entry — varlen THD, qk_hdim in {64,128,192,256} / v_hdim in {64,128}, bf16 or fp16.

    ``window_size`` is ``(left, right)`` sliding-window bounds, ``-1`` meaning infinite
    on that side; ``causal`` forces ``right=0``. Only finiteness is baked into the kernel
    (compile-time ``mask_left``/``mask_right``), so one variant serves any magnitude.
    ``sink`` is an optional ``[nheads_q]`` fp32 per-head logit in the scaled-score domain,
    its presence likewise compile-time. ``lse`` is an optional caller-provided
    ``[nheads_q, total_q]`` fp32 buffer, allocated here when ``return_lse`` and None.
    """
    assert q.dtype in _TORCH_DTYPE_MAP.values(), f"Expected bf16 or fp16, got {q.dtype}"
    assert (
        k.dtype == q.dtype and v.dtype == q.dtype
    ), f"q/k/v dtype must match, got {q.dtype}/{k.dtype}/{v.dtype}"
    dtype_str = "bf16" if q.dtype == torch.bfloat16 else "fp16"
    qk_hdim = q.shape[-1]
    assert (
        qk_hdim in SUPPORTED_QK_HDIM
    ), f"Expected qk_hdim in {SUPPORTED_QK_HDIM}, got {qk_hdim}"
    v_hdim = v.shape[-1]
    assert (
        v_hdim in SUPPORTED_V_HDIM
    ), f"Expected v_hdim in {SUPPORTED_V_HDIM}, got {v_hdim}"

    total_q_tokens = q.shape[0]
    batch = cu_seqlens_q.shape[0] - 1
    nheads_q = q.shape[1]
    nheads_k = k.shape[1]
    assert (
        nheads_q % nheads_k == 0
    ), f"nheads_q={nheads_q} must be a multiple of nheads_k={nheads_k}"
    gqa = nheads_q // nheads_k

    has_sink = sink is not None
    if has_sink:
        assert sink.dtype == torch.float32, f"sink must be fp32, got {sink.dtype}"
        assert (
            sink.dim() == 1 and sink.shape[0] == nheads_q
        ), f"sink must be [nheads_q={nheads_q}], got {tuple(sink.shape)}"
    # ptr_sink is only read when has_sink; pass q as a valid placeholder otherwise.
    sink_ptr = sink if has_sink else q

    if softmax_scale is None:
        softmax_scale = 1.0 / (q.shape[-1] ** 0.5)

    # Sliding window: causal forces right=0. Finiteness (>=0) is compile-time
    # (mask_left/mask_right); the magnitudes ride along as runtime Int32 args.
    win_left, win_right = int(window_size[0]), int(window_size[1])
    if causal:
        win_right = 0
    mask_left = win_left >= 0
    mask_right = win_right >= 0
    window_left = max(win_left, 0)
    window_right = max(win_right, 0)

    if out is None:
        out = torch.empty(
            (total_q_tokens, nheads_q, v_hdim), dtype=q.dtype, device=q.device
        )
    if return_lse:
        if lse is None:
            # [nheads_q, total_q] is the aiter varlen LSE convention, what CK and the
            # gfx1250 ASM varlen kernel both return. The kernel is stride-driven, so the
            # layout lives entirely in the two strides below.
            lse = torch.empty(
                (nheads_q, total_q_tokens), dtype=torch.float32, device=q.device
            )
        lse_ptr = lse
        stride_lse_seq = lse.stride(1)
        stride_lse_head = lse.stride(0)
    else:
        lse_ptr = q
        stride_lse_seq = 0
        stride_lse_head = 0

    # Q/K/V/O strides in ELEMENTS (TDM loaders consume them directly).
    stride_q_seq = q.stride(0)
    stride_k_seq = k.stride(0)
    stride_v_seq = v.stride(0)
    stride_o_seq = out.stride(0)
    stride_q_head = q.stride(1)
    stride_k_head = k.stride(1)
    stride_v_head = v.stride(1)
    stride_o_head = out.stride(1)

    if num_q_tiles_per_wave is None:
        num_q_tiles_per_wave = _pick_num_q_tiles(nheads_k, batch, max_seqlen_q, gqa)
    num_q_tiles_per_wave = int(num_q_tiles_per_wave)

    one_kv_tile = _pick_one_kv_tile(max_seqlen_k, qk_hdim, v_hdim, dtype_str)

    _ensure_thd_kernel(
        mask_left,
        mask_right,
        one_kv_tile,
        bool(return_lse),
        has_sink,
        gqa,
        qk_hdim=qk_hdim,
        v_hdim=v_hdim,
        dtype_str=dtype_str,
        num_q_tiles_per_wave=num_q_tiles_per_wave,
    )

    _run_compiled(
        _launch_fns[
            (
                "thd",
                mask_left,
                mask_right,
                bool(return_lse),
                has_sink,
                gqa,
                qk_hdim,
                v_hdim,
                dtype_str,
                num_q_tiles_per_wave,
                one_kv_tile,
            )
        ],
        out,
        q,
        k,
        v,
        lse_ptr,
        sink_ptr,
        cu_seqlens_q,
        cu_seqlens_k,
        softmax_scale,
        stride_q_seq,
        stride_k_seq,
        stride_v_seq,
        stride_o_seq,
        stride_q_head,
        stride_k_head,
        stride_v_head,
        stride_o_head,
        stride_lse_seq,
        stride_lse_head,
        window_left,
        window_right,
        max_seqlen_q,
        max_seqlen_k,
        nheads_k,
        batch,
        torch.cuda.current_stream(),
    )

    if return_lse:
        return out, lse
    return out


def flash_attn_batch_m32x8(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    out=None,
    return_lse=False,
    sink=None,
    lse=None,
    num_q_tiles_per_wave=None,
):
    """Host entry — batched BSHD ``[B, S, H, D]``, qk_hdim in {64,128,192,256} / v_hdim in {64,128}, bf16 or fp16.

    Uses the dedicated BSHD kernel with a uniform ``seq_len`` scalar (no cu_seqlens), so
    there is nothing transient to bake into a CUDA graph. ``window_size``, ``causal``,
    ``sink`` and ``lse`` behave as in ``flash_attn_varlen_m32x8``; the LSE buffer is
    ``[B, nheads_q, S_q]`` here.
    """
    assert q.dtype in _TORCH_DTYPE_MAP.values(), f"Expected bf16 or fp16, got {q.dtype}"
    assert (
        k.dtype == q.dtype and v.dtype == q.dtype
    ), f"q/k/v dtype must match, got {q.dtype}/{k.dtype}/{v.dtype}"
    dtype_str = "bf16" if q.dtype == torch.bfloat16 else "fp16"
    assert q.dim() == 4, f"Expected 4D BSHD tensor, got rank {q.dim()}"
    qk_hdim = q.shape[-1]
    assert (
        qk_hdim in SUPPORTED_QK_HDIM
    ), f"Expected qk_hdim in {SUPPORTED_QK_HDIM}, got {qk_hdim}"
    v_hdim = v.shape[-1]
    assert (
        v_hdim in SUPPORTED_V_HDIM
    ), f"Expected v_hdim in {SUPPORTED_V_HDIM}, got {v_hdim}"

    batch, seq_len_q, nheads_q, _ = q.shape
    seq_len_k = k.shape[1]
    nheads_k = k.shape[2]
    assert (
        nheads_q % nheads_k == 0
    ), f"nheads_q={nheads_q} must be a multiple of nheads_k={nheads_k}"
    gqa = nheads_q // nheads_k

    has_sink = sink is not None
    if has_sink:
        assert sink.dtype == torch.float32, f"sink must be fp32, got {sink.dtype}"
        assert (
            sink.dim() == 1 and sink.shape[0] == nheads_q
        ), f"sink must be [nheads_q={nheads_q}], got {tuple(sink.shape)}"
    # ptr_sink is only read when has_sink; pass q as a valid placeholder otherwise.
    sink_ptr = sink if has_sink else q

    if softmax_scale is None:
        softmax_scale = 1.0 / (q.shape[-1] ** 0.5)

    # Sliding window: causal forces right=0. Finiteness (>=0) is compile-time
    # (mask_left/mask_right); the magnitudes ride along as runtime Int32 args.
    win_left, win_right = int(window_size[0]), int(window_size[1])
    if causal:
        win_right = 0
    mask_left = win_left >= 0
    mask_right = win_right >= 0
    window_left = max(win_left, 0)
    window_right = max(win_right, 0)

    if out is None:
        out = torch.empty(
            (batch, seq_len_q, nheads_q, v_hdim), dtype=q.dtype, device=q.device
        )
    if return_lse:
        if lse is None:
            lse = torch.empty(
                (batch, nheads_q, seq_len_q), dtype=torch.float32, device=q.device
            )
        lse_ptr = lse
        stride_lse_seq = lse.stride(2)
        stride_lse_head = lse.stride(1)
        stride_lse_batch = lse.stride(0)
    else:
        lse_ptr = q
        stride_lse_seq = 0
        stride_lse_head = 0
        stride_lse_batch = 0

    # Empty tensor — skip the launch (host-known dims, no device sync). No queries means
    # no rows to write; no keys means an empty softmax set, so O=0 and LSE=-inf, or
    # LSE=sink[head] with a sink, exp(sink) being the only surviving term.
    if seq_len_q == 0 or seq_len_k == 0:
        if seq_len_q > 0 and seq_len_k == 0:
            out.zero_()
            if return_lse:
                if sink is not None:
                    lse.copy_(
                        sink.to(device=lse.device, dtype=lse.dtype)
                        .view(1, -1, 1)
                        .expand_as(lse)
                    )
                else:
                    lse.fill_(float("-inf"))
        return (out, lse) if return_lse else out

    # BSHD: seq is dim 1, head dim 2; the per-batch base is derived in-kernel as
    # batch_idx * seq_len. Strides in ELEMENTS (TDM loaders consume them directly).
    stride_q_seq = q.stride(1)
    stride_k_seq = k.stride(1)
    stride_v_seq = v.stride(1)
    stride_o_seq = out.stride(1)
    stride_q_head = q.stride(2)
    stride_k_head = k.stride(2)
    stride_v_head = v.stride(2)
    stride_o_head = out.stride(2)

    if num_q_tiles_per_wave is None:
        num_q_tiles_per_wave = _pick_num_q_tiles(nheads_k, batch, seq_len_q, gqa)
    num_q_tiles_per_wave = int(num_q_tiles_per_wave)

    one_kv_tile = _pick_one_kv_tile(seq_len_k, qk_hdim, v_hdim, dtype_str)

    _ensure_bshd_kernel(
        mask_left,
        mask_right,
        one_kv_tile,
        bool(return_lse),
        has_sink,
        gqa,
        qk_hdim=qk_hdim,
        v_hdim=v_hdim,
        dtype_str=dtype_str,
        num_q_tiles_per_wave=num_q_tiles_per_wave,
    )

    _run_compiled(
        _launch_fns[
            (
                "bshd",
                mask_left,
                mask_right,
                bool(return_lse),
                has_sink,
                gqa,
                qk_hdim,
                v_hdim,
                dtype_str,
                num_q_tiles_per_wave,
                one_kv_tile,
            )
        ],
        out,
        q,
        k,
        v,
        lse_ptr,
        sink_ptr,
        softmax_scale,
        stride_q_seq,
        stride_k_seq,
        stride_v_seq,
        stride_o_seq,
        stride_q_head,
        stride_k_head,
        stride_v_head,
        stride_o_head,
        stride_lse_seq,
        stride_lse_head,
        stride_lse_batch,
        window_left,
        window_right,
        seq_len_q,
        seq_len_k,
        nheads_k,
        batch,
        torch.cuda.current_stream(),
    )

    if return_lse:
        return out, lse
    return out
