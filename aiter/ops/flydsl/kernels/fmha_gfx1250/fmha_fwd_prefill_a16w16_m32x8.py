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
  - ``qk_hdim in {128, 192, 256}`` (D_qk), ``v_hdim == 128`` (D_v); ``n_block`` is picked
    by ``pick_n_block`` (128 at qk_hdim 128/192, 64 at 256)
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

from aiter.jit.utils.chip_info import get_lds_capacity_bytes
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


class WarpType(IntEnum):
    """Warp-specialization role (compile-time). gfx1250 pairs wave i with wave i+4 on
    one SIMD; the low half (waves 0..3) and high half (waves 4..7) run different
    main-loop preamble orderings so one wave drives memory while its SIMD-mate computes.

    SIMD parity (wave w -> SIMD w%4) is NOT a role: ``KV_SPLIT_PARITY_ORDER`` carries it
    as a runtime term instead, so the body stays 2-way.
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
SUPPORTED_QK_HDIM = (128, 192, 256)

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
# ...and only if the body still fits the register file. 128 doubles the live fragment state;
# at qk_hdim 192 that is 512 VGPR / 49 spills and 11% slower, so cap the widening by hdim.
N_BLOCK_WIDE_MAX_QK_HDIM = 128

# K|V LDS slots the main loop rotates through. 3 is the exact minimum for the
# software-pipelined body: it reads V(u-1) from slot 0 and K(u) from slot 1 while the
# copy for tile u+1 is written into slot 2.
N_KV_PP = 3
# K one tile ahead of V. Body u reads K from slot 1 and V from slot 0, so with both copies
# landing in slot 2 the K ring wastes a slot on the already-dead K(u-1) and K gets only one
# body of latency cover against V's two. LO instead writes K(u+2) into slot 0 -- a chunk
# whose K half died at body u-1, and whose V half this body reads (12-chunk layout keeps
# them disjoint) -- which buys K the same two bodies. Costs one extra prologue copy
# (K(start+1) into slot 2) and makes the two halves' prologues symmetric.
KV_K_AHEAD = True
# Steady-state KV fence depth. With K one tile ahead both halves reach a fence with the
# newest tile still in flight and the one they are about to read already retired, so the
# fence can leave one tile outstanding instead of draining to 0.
KV_PARTIAL_FENCE = KV_K_AHEAD

# log2(e): exp(x) = exp2(x * LOG2E). Softmax uses the native ISA exp2 intrinsic.

# S reaches softmax already in log2 units (S' = S * softmax_scale * LOG2E), so its inner
# loop is a plain exp2(S' - m) with no LOG2E multiply and m/LSE live in the log2 domain.
# True folds that constant into Q via the bf16 multiply the Q loader already does: free,
# but it rounds the scale to bf16's 8 mantissa bits. False leaves Q raw and scales the f32
# QK accumulator instead -- exact, at R*NKV*8 VALU per tile, and paid on the gemm side of
# the anti-phase body (the shorter one). Expect to key this off return_lse once training
# wants the precision.
FOLD_SCALE_INTO_Q = True

# Deferred oaccu rescale (FAv4 innovation, hk_mla spec §9.1.1). Rescaling the
# running O accumulator by corr = exp(m_prev - m_new) is a full-width VALU pass
# (d_tiles*8 f32/lane) every tile, but corr == 1 when the running max doesn't
# move. So keep m STALE while the tile's row max stays within RESCALE_THRESHOLD
# logit units of it: P = exp2(S - m_stale) accumulates against the un-rescaled
# oaccu/denom, staying consistent. The per-lane test is promoted to wave-uniform
# via ballot (any lane over threshold => the whole wave rescales), so the caller
# can gate the wide multiply with one non-divergent scf.if. In NATURAL logits (m is
# log2-domain, so the compare scales this by LOG2E): threshold 8.0 => defer until the
# max would move by e^8 ~ 2981x, far under the e^88 fp32 exp overflow wall.
# Set ENABLE_DEFER_RESCALE=False (or threshold < 0) to always rescale.
ENABLE_DEFER_RESCALE = True
RESCALE_THRESHOLD = 8.0

# Running-max seed: a finite big-negative (not -inf) so a fully-masked row keeps m
# finite -> softmax's (m_prev - m_new) and fma(s, .., -m) never hit -inf arithmetic
# (NaN). exp2(big_neg - real) still underflows to 0, so it zeroes the empty seed like
# -inf did. Masked scores stay -inf (p = exp2(-inf) = 0); only the max seed changes.
BIG_NEG = -1.0e30

# Compile-time Q/K/V loader select. False = V1 (Q ring async + swizzled LDS; K/V
# cluster_load_async), True = V2 (Q per-warp TDM; K/V TDM global->LDS, HW OOB, far fewer
# address VGPRs). Gates all three loaders (Q, K and V); O is selected separately by
# O_VARIANT. For K and V the two differ ONLY in that transport -- same padded row-major
# LDS, same fragment reads, V2 a subclass of V1 -- so this flag picks a class pair and
# nothing at the call sites below branches on it.
USE_TDM_LOADER = True

# K/V producer specialization. The LO half issues every K copy, the HI half every V copy,
# and each of a half's KV_PRODUCER_WARPS waves copies one dense n_block/KV_PRODUCER_WARPS
# row band by itself, so each half's counter tracks one operand. The drain barrier still
# publishes both halves. Transport-independent: the partition reaches the managers as a
# ProducerCtx, and both build that band the same way.
KV_PRODUCER_WARPS = NUM_WAVES // 2
# Read a tile's two n_block halves in opposite order on odd SIMDs, so the two SIMD
# parities never sit in the same 64 KB LDS segment set at the same point of a gemm. Swaps
# the READ bases only (the producer still writes split s to chunk s); that permutes the
# kv-tile slots by ^(NKV/2) and the contraction slots by ^(nkt/2) TOGETHER, so S, P and
# the PV consumption stay mutually consistent and only the softmax mask sees the absolute
# index. Carried as a RUNTIME term off warp_idx, not a trace axis: the split bases are
# built once in the prologue and live in iter_args, so the swap is a prologue add, and the
# mask needs one extra base per body. (It used to split warp_type 4 ways, which doubled the
# kernel to 96 KB at n_block=128 -- past the 64 KB SQC I$ -- and cost 3.6% on case 10.)
KV_SPLIT_PARITY_ORDER = True

# LO/HI anti-phase (FA3 ping-pong). Both halves run the same tile stream and the same
# number of bodies and barriers; the HI half runs its two phases in the opposite order,
#   LO body u:  gemm(u)                 | BAR | softmax(u)
#   HI body u:  softmax(u-1) + rescale  | BAR | gemm(u)
# so each barrier window has one half in the WMMA stream and the other in the softmax
# VALU stream. HI carries s_acc across the back edge instead of P (and needs no carried
# ring head -- its head hides under the softmax that opens its own body). Its dead
# leading softmax(start-1) is neutralized by seeding s_acc to -inf, which is exactly the
# fully-masked path (m_new = m_prev, corr = 1, P = 0, d unchanged).
ANTI_PHASE = True
# Lagging half only: put the KV drain barrier AFTER the gemm instead of at the top of the
# body, so the loop's back-edge bookkeeping and the prefetch descriptor's VALU/SALU land
# BEHIND it.
LAG_DRAIN_AFTER_GEMM = True and ANTI_PHASE
# O writer variant (decoupled from USE_TDM_LOADER): "v1" swizzled LDS + buffer_store (fastest so
# far), "v2" TDM store (padding ignored -> contiguous LDS -> bank conflict, slow), "v3" padded LDS +
# global_store_async_from_lds_b128.
O_VARIANT = "v3"

# LDS->VGPR ring for the QK/PV WMMA streams. The gemms used to burst EVERY ds_load of
# the resident KV tile into VGPRs before the first wmma, so the live cost scaled with
# hdim (K: 32/48/64 loads = 128/192/256 VGPR at qk_hdim 128/192/256). A ring holds only
# RING loads live at a fixed 4 VGPR each, with NP = RING - 2*LAG in flight.
#
# Swept on case 10 at fixed init: QK 20/4 and PV 16/4 both beat the old burst
# (1255-1273us vs 1317us), and 256/128 goes from 223 spills to 0. The optimum is sharp --
# QK ring 16/24/28 all lose 3-6% to 20, and PV lag 2/6 lose to 4. Keep RING a multiple of 4
# (an odd wrap flips slot parity: +1 cyc on half the wmma).
# LAG=0 with RING >= num_ds_loads reproduces the old burst exactly (A/B without a revert).
QK_RING = 20
QK_LAG = 4
PV_RING = 16
PV_LAG = 4
# The fused PV+QK ring (_pv_qk_gemm). One ring spans both operand streams, so its
# refills pull K loads while the PV wmma stream is still running.
PVQK_RING = 24
PVQK_LAG = 2
# Ring-head loads issued at the END of a body (under the softmax VALU) and carried across
# the back edge as iter_args, instead of issued at the top of the body that consumes them.
# The V they read is tile u, already resident in the slot this body used for K. Clamped to
# NP; 0 restores the un-carried head. Costs 4 VGPR per carried load.
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
    """Retire outstanding K/V global->LDS copies down to the given per-counter depths.

    A counter is waited only if the caller names it; the default -1 emits nothing for it.
    Naming both is what lets a mixed loader pair (e.g. a V1 K manager on ``asynccnt`` with
    a V2 V manager on ``tensorcnt``) share one fence. The counts are the CALLER's: each is
    how many of THIS wave's copies may stay in flight past the wait."""
    if num_tensorcnt >= 0:
        tdm_ops.tensor_wait(num_tensorcnt)
    if num_asynccnt >= 0:
        rocdl.s_wait_asynccnt(num_asynccnt)


def _kv_drain_depths(num_tensorcnt, num_asynccnt):
    """The same counters a steady-state fence names, but fully drained."""
    return (0 if num_tensorcnt >= 0 else -1, 0 if num_asynccnt >= 0 else -1)


def _bare_barrier():
    """Workgroup rendezvous with NO memory fence, unlike ``gpu.barrier()``.

    ``gpu.barrier()`` lowers to ``s_wait_storecnt_dscnt 0x0`` + signal/wait, so it would
    retire a ring head issued just above it. Use this where the barrier is a scheduling
    or counting rendezvous, or where the caller has already named its own dscnt depth.
    """
    rocdl.s_barrier_signal(-1)
    rocdl.s_barrier_wait(-1)


def _kv_fence(num_tensorcnt=-1, num_asynccnt=-1, num_dscnt=None):
    """``_kv_wait``, then publish the retired copies workgroup-wide.

    The ``s_barrier`` is what publishes a wave's share of a tile to its peers -- the
    counters only bound the issuing wave's own copies -- and it doubles as the WAR wall
    for the slot about to be written.

    ``num_dscnt`` is how many of this wave's LDS reads may stay in flight past the
    barrier. ``None`` keeps ``gpu.barrier()``'s workgroup fence, which drains dscnt (and
    storecnt) to 0. Name a depth to swap that for a bare signal/wait plus an explicit
    partial wait, so a ring head issued just before the fence survives it."""
    _kv_wait(num_tensorcnt, num_asynccnt)
    rocdl.sched_barrier(0)
    if num_dscnt is None:
        gpu.barrier()
    else:
        rocdl.s_wait_dscnt(num_dscnt)
        _bare_barrier()
    rocdl.sched_barrier(0)


def _load_seqlen_pair(ptr_tensor, idx):
    """Load ``ptr_tensor[idx]`` and ``ptr_tensor[idx + 1]`` (adjacent i32s) as one
    ``vector<2xi32>``; returns ``(start, end)`` as ``fx.Int32``.

    The two values are contiguous and the address is uniform (derived from
    ``block_id``), so a single 64-bit load should lower to one ``s_load_b64``.
    """
    p = fx.get_iter(ptr_tensor)
    pair = fx.ptr_load(p + fx.Int64(idx), result_type=fx.Vector.make_type(2, fx.Int32))
    return fx.Int32(pair[0]), fx.Int32(pair[1])


def _load_sink_logit(ptr_sink, q_head_idx, num_heads_q):
    """Load this lane's per-head sink logit ``sink[q_head_idx]`` from the 1-D
    ``[num_heads_q]`` fp32 ``sink`` — one extra ``exp(sink)`` term in the softmax
    denominator, in the scaled-score domain (same units as S).

    Uses a flat ``llvm.load`` (not ``buffer_load``): ``buffer_load`` re-scales the
    offset (``offset * element_bytes``) INTERNALLY, so a flat load keeps the address
    arithmetic SSA-visible for LLVM to order/cover under sched mode 2. Safe without a
    HW bounds check because ``q_head_idx = kv_head*gqa_ratio + row_idx%gqa_ratio`` is
    always ``< num_heads_q`` (in-bounds by construction)."""
    del num_heads_q  # in-bounds by construction; no buffer bounds check needed
    sink_base_i64 = fx.Int64(fx.ptrtoint(fx.get_iter(ptr_sink)))
    byte_off = fx.Int64(q_head_idx) * fx.Int64(4)
    addr = sink_base_i64 + byte_off
    gptr = create_llvm_ptr(addr, address_space=1)
    return fx.Float32(llvm_dialect.load(T.f32, gptr))


def _packed_tile_indices(gqa_ratio, warp_idx, lane_idx):
    """Map this lane's rows in the packed ``(seq, q_head_in_group)`` tile to global
    indices; returns ``(kv_head, q_head_idx, seq_idx)`` where ``kv_head`` is a
    scalar ``fx.Int32`` (shared) and ``q_head_idx`` / ``seq_idx`` are length-R
    lists (one per q-WMMA-tile owned by this wave; R = WMMA_ROW_PER_WAVE).

    GQA head x seq packing:
      block_id x -> tile over one kv-head's ``(seq, q_head_in_group)`` plane
      block_id y -> kv_head
    ``q_head_in_group`` is the fast axis, so the ``% / //`` use the small (often
    power-of-two) ``gqa_ratio``. Each of the ``BLOCK_M`` rows is an independent
    query sharing this kv-head's K/V. The R tiles a wave owns are contiguous:
    ``warp_row0 = block_x*BLOCK_M + warp_idx*(R*WMMA_M)`` and tile ``qt`` starts
    at ``warp_row0 + qt*WMMA_M``.
    """
    kv_head = fx.Int32(gpu.block_id("y"))
    warp_row0 = fx.Int32(gpu.block_id("x")) * BLOCK_M + warp_idx * (
        WMMA_ROW_PER_WAVE * WMMA_M
    )
    q_head_idx = []
    seq_idx = []
    for qt in range(WMMA_ROW_PER_WAVE):
        row_idx = warp_row0 + qt * WMMA_M + lane_idx % WMMA_M
        q_head_idx.append(kv_head * gqa_ratio + row_idx % gqa_ratio)
        seq_idx.append(row_idx // gqa_ratio)
    return kv_head, q_head_idx, seq_idx


# ============================================================================
# Compute stages — EMPTY, unwired. Implemented and tested one at a time; the KV
# streaming driver below lands (and is tested) first with these left inert.
# ============================================================================


def _wmma(a, b, c):
    """v_wmma_f32_16x16x32_{bf16,f16} (gfx1250, wave32): C[16x16 f32] = A[16x32] @
    B[32x16] + C. Preserve the SSA-returning intrinsic and disable operand reuse.

    a/b: v16 16-bit fragments; c: v8 f32 accumulator; returns the v8 f32 result
    (raw MLIR value, feed straight back as ``c`` to accumulate)."""
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
    caller places it AFTER the next gemm's ring head: nothing in the head depends on P, so
    the ds_loads issue first and the v_cvt batch fills their shadow instead of the SP
    stalling on the cvts before the loads go out. Costs P a wider live range (f32, not the
    narrowed form) across the head on both halves."""
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
    """In-flight ds_load depth (NP) of a ``_ring_drive`` with this geometry.

    A tile that fits in the ring never wraps, so no slot is ever refilled and the WAR lag
    buys nothing: prefetch the whole tile rather than deferring loads behind wmma they
    gain nothing from. Only a wrapping ring pays for lag."""
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


def _ring_head(*, num_frag, emit, ring, lag):
    """Issue a ring's NP-deep prefetch early -- hoisting its LDS latency under unrelated
    work -- for a later ``_ring_drive`` called with the same ``ring``/``lag``."""
    return [
        _emit_ld(emit, j) for j in range(_ring_num_prefetch(num_frag, ring, lag))
    ]


def _ring_drive(*, num_frag, emit, consume, ring, lag, head=None, dies=None):
    """Drive a fully-unrolled LDS->VGPR ring feeding a WMMA stream.

    ``emit(j)`` emits ds_load ``j`` (2 per WMMA fragment); ``consume(i, lo, hi)`` emits
    fragment ``i``'s WMMA chain. Only ``ring`` loads are live at once and
    ``NP = ring - 2*lag`` are in flight, so a refill targets slots last read ``lag``
    fragments ago -- trading pipeline depth for WAR distance at fixed VGPR cost.

    Refills are hoisted ABOVE the wmma: that is what keeps the ring from reintroducing
    the wmma->ds_load issue bubble the old burst avoided.

    ``head`` optionally adopts an already-issued NP-deep prefetch (the PV case hoists it
    above softmax so the LDS latency hides under the softmax VALU). A tile of at most
    ``ring`` loads degenerates to the old burst-everything form (see
    ``_ring_num_prefetch``), as does ``lag=0`` with ``ring >= 2*num_frag``.

    ``dies(i)`` optionally names values *other* than the ring pair whose last read is
    fragment ``i`` -- the B operands, which the ring does not own. Without it those
    registers fall free the instant their last wmma issues, and the next refill grabs
    them at a WAR distance of one VALU (``s_wait_alu depctr_va_vdst(1)``) instead of the
    ``2*rel`` the ring pairs enjoy. Routing them through the same release queue puts
    every register the refill can see at the same distance.
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


def _scale_s(s_acc_list, s_scale):
    """Apply softmax_scale*LOG2E to a gemm's f32 QK accumulators (``FOLD_SCALE_INTO_Q
    =False`` path; None = already folded into Q, returns them untouched). The caller
    invokes this so the multiplies land where it wants them -- keep them on the gemm
    side of the anti-phase body, i.e. before the phase barrier."""
    if s_scale is None:
        return s_acc_list
    return [[acc * s_scale for acc in row] for row in s_acc_list]


def _qk_gemm(*, k_emit, q_frags_list, n_block, head=None, ring=QK_RING, lag=QK_LAG):
    """GEMM1: S^T = K @ Q^T for one resident KV tile, for all R q-WMMA-tiles this
    wave owns. K is **shared** across the q-tiles (loaded once), so each K fragment
    is shuffled once and fed into R independent WMMA chains.

    WMMA convention (gfx1250): S^T[kv,q] = K @ Q^T with **K = A-operand** (src_a)
    and **Q = B-operand** (src_b). Contract d in ``NDT = qk_hdim//WMMA_K`` tiles;
    produce ``NKV = n_block//WMMA_N`` kv-tiles. GPU-verified accumulator layout:
    lane ``l`` element ``si`` holds S^T[kv = kv_tile*WMMA_N + (l//16)*8 + si,
    q = l%16] (kv on the C-row / M axis, q on the C-col / N axis).

    ``q_frags_list`` is a length-R list; entry ``qt`` is that q-tile's NDT v16-bf16
    Q fragments. Returns ``s_acc_list``: a length-R list, each a list of NKV
    v8-f32 accumulators (== P^T for that q-tile).

    ``k_emit(j)`` emits the ``j``-th K ``ds_load`` of the resident block (see
    ``k_mgr.load_one_to_reg``) in flat ``(kv, dt, half)`` order; ``_ring_drive`` calls it on
    demand so only ``ring`` loads are live at a time instead of all 2*NKV*NDT. ``head`` adopts
    an NP-deep prefetch already issued by the caller (the warp-specialized preamble, which
    staggers it against the global->LDS prefetch). Each K fragment is the two 16-col halves of
    a d-tile shuffled into a v16 fragment matching the Q frag layout.
    """
    R = len(q_frags_list)
    NKV = n_block // WMMA_N  # output kv tiles (WMMA_N kv rows each)
    NDT = len(q_frags_list[0])  # contraction d-tiles (== qk_hdim // WMMA_K)

    # Consume in (kv, dt, half) order: a (half=0, half=1) pair shuffles into a v16
    # K fragment (shared by all q-tiles); NDT d-tiles accumulate into one kv-tile's
    # s_acc, independently per q-tile.
    s_acc_list = [[None] * NKV for _ in range(R)]

    def consume(i, lo, hi):
        kv, dt = divmod(i, NDT)
        k_frag = lo.shuffle(hi, list(range(16)))
        for qt in range(R):
            acc = s_acc_list[qt][kv] if dt > 0 else fx.Vector.filled(8, 0.0, fx.Float32)
            s_acc_list[qt][kv] = _wmma(k_frag, q_frags_list[qt][dt], acc)

    _ring_drive(
        num_frag=NKV * NDT,
        emit=k_emit,
        consume=consume,
        ring=ring,
        lag=lag,
        head=head,
    )
    return s_acc_list


def _tree_reduce_multi(lists, op3, op2):
    """Balanced 3-way tree reduction of R independent lists in lockstep, returning one result
    per list. Per list the critical path is ~ceil(log3(N)) vs N-1 for a left-fold, and op3 =
    nested op2 so the backend fuses it (v_max3_f32 for max). Each layer's combines are emitted
    POSITION-MAJOR across the lists (list0[pos], list1[pos], ...) so the R independent ops sit
    adjacent in the IR -> the backend can dual-issue them and hide one row's cross-lane /
    latency bubble behind the other's work."""
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
    kv_pos_base=None,
    kv_swap_delta=None,
    q_max_list=None,
    q_min_list=None,
    kv_len=None,
):
    """Online-softmax update for one KV tile, for ALL R q-WMMA-tiles this wave owns.

    The R rows are independent (each owns its S, running m/d, and mask bounds) but share
    the tile's K/V. Processing them together lets the two rows' balanced max-tree and
    sum-tree reductions emit INTERLEAVED (position-major across rows, via
    ``_tree_reduce_multi``) so the backend can dual-issue row0/row1 combines and hide each
    other's cross-lane permlanex16 latency. ``s_list[r]`` is already in log2 units
    (softmax_scale*LOG2E applied in Q or on the QK accumulator), so exp is a plain exp2.

    Layout (from ``_qk_gemm``): ``s_list[r]`` is a list of ``NKV = n_block//WMMA_N`` v8-f32
    accumulators; this lane owns query ``q = warp*16 + l%16`` and, in tile ``kvt``, the kv
    rows ``kvt*16 + (l//16)*8 + [0..8)`` (its half). The peer lane ``l^16`` holds the other
    8-row half of the same q, so the row max/sum reduce locally over (kvt, i) then across
    the ``shuffle_xor(16)`` partner.

    ``kv_swap_delta`` is ``KV_SPLIT_PARITY_ORDER``'s correction: when the wave reads the
    tile's two halves in swapped order, slot ``kvt`` holds physical kv-tile ``kvt ^ (NKV/2)``,
    i.e. its absolute offset moves by +/- ``n_block/2``. A runtime Int32 (0 on even SIMDs),
    folded into two per-body mask bases rather than into the NKV per-slot constants.

    Masking (per element, per row r, sequence-relative ``kv_pos = kv_pos_base + (l//16)*8 +
    kvt*16 + i``; all bounds fx.Int32): ``q_max_list[r]`` masks ``kv_pos > q_max`` (band
    upper edge = ``q_seq + (kv_len-q_len) + window_right``; clamped to ``kv_len-1`` on the
    last tile to fold the tail); ``q_min_list[r]`` masks ``kv_pos < q_min`` (band lower edge
    = ``q_seq + (kv_len-q_len) - window_left``); kv_len (only when q_max is None) masks
    ``kv_pos >= kv_len`` (standalone tail for the non-causal case). A None bound skips it.

    Args: ``s_list``/``m_prev_list``/``d_prev_list``/``q_max_list``/``q_min_list`` are
    length-R lists (R = WMMA_ROW_PER_WAVE); the q_*_list default to all-None. m_prev/d_prev
    are fx.Float32 shared by the l<->l^16 pair.

    Returns 4 length-R lists ``(p, m_new, d_new, corr)`` plus ``rescale_masks`` —
    per row: p = NKV v8 **f32** P^T = exp(S^T - m_new), NOT narrowed to the wmma element
    type -- the caller places that conversion with ``_p_to_elem``; m_new = updated running
    max, STALE (== m_prev) when that row's ballot did not fire (FAv4 §9.1.1);
    d_new = corr*d_prev + rowsum(p); corr = exp(m_prev - m_new) (== 1 on the stale path).
    ``rescale_masks`` is the R raw ballots (None when deferral is compiled out); the
    caller folds them into its one branch condition at the use site.
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

    # Mask base. Swapped reads move the low NKV/2 slots up by n_block/2 and the high slots
    # down by it, so one base per half absorbs the whole correction and the per-slot term
    # stays the compile-time kvt*WMMA_N + i.
    if kv_pos_base is not None:
        _pos0 = kv_pos_base + khalf * fx.Int32(8)
        _pos_half = (
            [_pos0, _pos0]
            if kv_swap_delta is None
            else [_pos0 + kv_swap_delta, _pos0 - kv_swap_delta]
        )

    R = len(s_list)
    q_max_list = q_max_list if q_max_list is not None else [None] * R
    q_min_list = q_min_list if q_min_list is not None else [None] * R

    # ---- Pass 1 (all R rows): masked S values, flattened (kvt, i) order. Built for every
    # row first so the row max-trees below emit INTERLEAVED. ----
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

    # ---- Row max: the R rows' balanced max-trees emitted INTERLEAVED (position-major
    # across rows) so the backend dual-issues row0/row1 combines and hides the cross-lane
    # permlanex16 latency. ----
    max3 = lambda a, b, c: fmax(fmax(a, b), c)
    local_max_list = _tree_reduce_multi(s_masked_list, max3, fmax)

    # ---- Per row: peer reduce + deferred-rescale decision + corr. ----
    m_new_list, corr_list = [], []
    rescale_masks = None if not _defer else []
    for r in range(R):
        m_prev, q_min = m_prev_list[r], q_min_list[r]
        row_max = fmax(local_max_list[r], peer(local_max_list[r]))
        m_full = fmax(m_prev, row_max)

        # Deferred oaccu rescale (FAv4, hk_mla spec 9.1.1): keep m STALE while the running
        # max barely moves (< RESCALE_THRESHOLD logits) so the caller SKIPS the wide
        # `o_acc *= corr` multiply. Ballot promotes the per-lane test to wave-uniform (non-
        # divergent branch). ORDERED OGT: a fully-masked lane's -inf - -inf = NaN never
        # forces a rescale. Safe stale path: row_max - m_prev <= 8 -> p <= e^8, no overflow.
        if _defer:
            # `>` lowers to ordered OGT, so a fully-masked lane's -inf - -inf = NaN
            # compares false and never forces a rescale.
            need = fsub(row_max, m_prev) > fx.Float32(RESCALE_THRESHOLD * LOG2E)
            # Select on the per-lane `need`, not on the ballot: every lane owns its own q
            # row (the l<->l^16 pair shares one and agrees after the peer reduce), so
            # staleness is a per-lane decision and the ballot is only the caller's branch
            # condition. This leaves that branch as the compare's only other consumer, so
            # the whole fold sinks to the s_cbranch instead of sitting between the max tree
            # and the exp chain as an s_cmp/s_cselect turnaround.
            m_new = need.select(m_full, m_prev)
            rescale_masks.append(fx.Int32(rocdl.ballot(fx.Int32.ir_type, need)))
        else:
            m_new = m_full

        # corr = exp(m_prev - m_new), log2-domain so exp2 takes the difference directly.
        # m is seeded to BIG_NEG (finite), so m_prev/m_new never reach -inf: a fully
        # masked row (row_max=-inf) keeps m_new=BIG_NEG, giving corr=exp2(0)=1 and a
        # finite p=exp2(-inf)=0. No (-inf)-(-inf) / -inf+inf, so no clamp needed.
        corr = exp2(fsub(m_prev, m_new))
        m_new_list.append(m_new)
        corr_list.append(corr)

    # ---- Pass 2 (all R rows): p = exp2(S - m_new), every row's subs emitted before any
    # exp. The subs then pair into v_dual_sub_f32 against their row's shared m_new, and the
    # exp run is pure TRANS -- the coexecution hazard is TRANS followed by a non-TRANS VALU,
    # so an uninterrupted run needs none of the ~36 v_nop the backend pads in when it
    # interleaves the two. ----
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

    # ---- Row sum: R rows' balanced sum-trees emitted INTERLEAVED. fadd_t (fast-math minus
    # reassoc) so LLVM's Reassociate does NOT re-linearize the tree into a serial chain. ----
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


def _pv_gemm(
    *,
    v_emit,
    p_list,
    v_hdim,
    n_block,
    o_acc_list=None,
    head=None,
    ring=PV_RING,
    lag=PV_LAG,
):
    """GEMM2: O^T = V^T @ P^T for one resident KV tile, for all R q-WMMA-tiles this
    wave owns. V is **shared** across the q-tiles (transpose-loaded once), so each
    V fragment is shuffled once and fed into R independent WMMA chains.

    WMMA convention (gfx1250): D[M=d, N=q] with **A = V^T** (src_a, transpose-loaded
    via ds_load_tr16_b128) and **B = P^T** (src_b, the bf16 softmax output). Contract
    kv in ``nkt = n_block//WMMA_K`` tiles (K=32); produce ``d_tiles = v_hdim//WMMA_M``
    output d-tiles (M axis). Lane ``l`` element ``si`` of tile ``dt`` holds
    O[q = l%16, d = dt*WMMA_M + (l//16)*8 + si] — the OManager16b frag layout.

    ``p_list`` is a length-R list; entry ``qt`` is that q-tile's list of softmax
    kv-tiles (bf16 P^T B-operands). ``o_acc_list`` is either None or a length-R
    list of running O accumulators (each ``d_tiles`` v8-f32, already rescaled by
    ``corr``). Returns ``out_list``: a length-R list of updated O accumulators.

    ``v_emit(j)`` emits the ``j``-th V transpose ds_load of the resident block (see
    ``v_mgr.load_one_to_reg``) in flat ``(dt, kt, half)`` order; ``_ring_drive`` calls it on
    demand so only ``ring`` loads are live at a time instead of all 2*d_tiles*nkt. ``head``
    adopts the NP-deep prefetch the caller issues before softmax, so that latency still hides
    under the softmax VALU. Online accumulation: each tile's PV adds onto the running o_acc.
    Each WMMA operand is a v16 bf16 fragment = two 16-wide halves shuffled: A from V-tiles
    (kv, kv+16), B from softmax tiles (p[2kt], p[2kt+1]).
    """
    R = len(p_list)
    d_tiles = v_hdim // WMMA_M  # output d-tiles (M axis, WMMA_M d rows each)
    nkt = n_block // WMMA_K  # kv contraction tiles (K=32 kv each)

    out_list = [[None] * d_tiles for _ in range(R)]
    p_frags = {}  # (qt, kt) -> B operand, handed to _ring_drive on its last fragment

    def consume(i, v_lo, v_hi):
        dt, kt = divmod(i, nkt)
        # A-operand: V^T frag = two 16-kv transpose-load tiles -> v16 bf16 (shared
        # across q-tiles).
        v_frag = v_lo.shuffle(v_hi, list(range(16)))
        for qt in range(R):
            acc = out_list[qt][dt]
            if acc is None:
                acc = (
                    o_acc_list[qt][dt]
                    if o_acc_list is not None
                    else fx.Vector.filled(8, 0.0, fx.Float32)
                )
            # B-operand: P^T frag = two consecutive softmax kv-tiles -> v16 bf16.
            p = p_list[qt]
            p_frag = p[2 * kt].shuffle(p[2 * kt + 1], list(range(16)))
            p_frags[(qt, kt)] = p_frag
            out_list[qt][dt] = _wmma(v_frag, p_frag, acc)

    def dies(i):
        # kt is reused by every d-tile, so a P^T fragment's last read is its row of
        # the final d-tile -- exactly where the ring's own refills land.
        dt, kt = divmod(i, nkt)
        if dt != d_tiles - 1:
            return ()
        return [p_frags.pop((qt, kt)) for qt in range(R)]

    _ring_drive(
        num_frag=d_tiles * nkt,
        emit=v_emit,
        consume=consume,
        ring=ring,
        lag=lag,
        head=head,
        dies=dies,
    )
    return out_list


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
):
    """GEMM2(u-1) then GEMM1(u) driven by ONE ring over the concatenated V-then-K
    fragment streams. Semantics are exactly ``_pv_gemm`` followed by ``_qk_gemm``
    -- consumption stays strictly sequential, so P dies before the first QK wmma and
    the two accumulator sets are never co-live. What the merge buys is the ring's
    refill window: the K loads for QK(u) issue ~NP/2 fragments before the transition,
    i.e. while PV's wmma stream is still running.

    Returns ``(out_list, s_acc_list)``.
    """
    R = len(p_list)
    d_tiles = v_hdim // WMMA_M
    nkt = n_block // WMMA_K
    NKV = n_block // WMMA_N
    NDT = len(q_frags_list[0])

    num_vfrag = d_tiles * nkt
    num_kfrag = NKV * NDT
    num_vld = 2 * num_vfrag

    out_list = [[None] * d_tiles for _ in range(R)]
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
        # The PV half's B operands die on the final d-tile row -- which is where the
        # ring is already refilling for the QK half. Release them through the ring so
        # the refill sees one distance, not two.
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


def _alloc_lds():
    """Allocate the full per-CU LDS once and return its base (fx.Int32). Called once per
    kernel body before the warp-type dispatch so both ``_core_attention`` traces share the
    single SharedAllocator flydsl permits; K/V, Q, and the O epilogue all carve this base.
    """
    smem = fx.SharedAllocator().allocate(get_lds_capacity_bytes("gfx1250"))
    return fx.Int32(fx.ptrtoint(smem.peek().ptr))


def _core_attention(
    *,
    qk_hdim,
    v_hdim,
    n_block,  # compile-time KV block width (columns of one QK GEMM tile)
    mask_left,  # compile-time: bound the left band edge (finite window_left)
    mask_right,  # compile-time: bound the right band edge (causal or finite window_right)
    return_lse,
    has_sink,  # compile-time: fold a per-head sink logit into the softmax denom
    gqa_ratio,  # compile-time GQA group size = nheads_q // nheads_kv
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
    """Layout-agnostic m32x8 compute — empty scaffold.

    Shared by the THD and BSHD kernel entries. The caller resolves the per-batch
    token ranges (``q_start``/``q_len`` and ``kv_start``/``kv_len``) — the only
    part that differs between varlen and batched layouts — and passes them here.

    Warp-specialized: the caller dispatches on runtime ``warp_type`` and traces this
    body TWICE (once per compile-time ``warp_type``); the two instantiations differ only
    in the ``main_loop`` preamble ordering (LO drives K load, HI shadows it).
    """
    lane_idx = _lane_id()
    kv_head, q_head_idx, seq_idx = _packed_tile_indices(gqa_ratio, warp_idx, lane_idx)

    # softmax_scale*LOG2E goes either into Q (bf16, free) or onto the f32 S (exact).
    _log2_scale = softmax_scale * fx.Float32(LOG2E)
    _q_scale = _log2_scale if FOLD_SCALE_INTO_Q else None
    _s_scale = None if FOLD_SCALE_INTO_Q else _log2_scale

    # K/V staging: N_KV_PP slots of 2 LDS_CHUNK_BYTES chunks each, every tile split 2-way
    # along n_block into chunks 6 apart (different 64 KB segments). Q time-shares slot 1's
    # chunks, O the two slots the loop has finished with -- see the 12-chunk map up top.
    if USE_TDM_LOADER:
        q_mgr = QManager16bV2(
            qk_hdim=qk_hdim,
            gqa_ratio=gqa_ratio,
            num_waves=NUM_WAVES,
            q_tiles_per_wave=WMMA_ROW_PER_WAVE,
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
            q_tiles_per_wave=WMMA_ROW_PER_WAVE,
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
    # two bases in the other order, so the parities are never in the same 64 KB segment set
    # at the same point of a gemm. Runtime, off warp_idx: _split_bufs runs N_KV_PP times in
    # the prologue and its results are carried as ds pointers, so this is a handful of
    # prologue adds and nothing in the loop.
    assert not (
        KV_SPLIT_PARITY_ORDER and KV_LDS_SPLITS != 2
    ), "parity order needs a 2-way split"
    if KV_SPLIT_PARITY_ORDER:
        _odd = warp_idx & fx.Int32(1)
        _rd0 = _odd * fx.Int32(_SPLIT_STRIDE)
        _rd = [_rd0, fx.Int32(_SPLIT_STRIDE) - _rd0]
        # Slot kvt then holds physical kv-tile kvt ^ (NKV/2); the mask is the only consumer
        # of the absolute index, and it takes the correction as +/- this delta.
        _kv_swap_delta = _odd * fx.Int32(n_block // 2)
    else:
        _rd = [fx.Int32(0), fx.Int32(_SPLIT_STRIDE)]
        _kv_swap_delta = None

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
        block_x=fx.Int32(gpu.block_id("x")),
        warp_idx=warp_idx,
        lane_idx=lane_idx,
        ptr_lds_warp=q_lds_warp,
    )

    # ---- This WG's KV tiles span relative kv [start_tile*n_block, kv_len_wg).
    # Packed row r maps to seq r//gqa_ratio; a query at seq s attends the band
    # [s+causal_off-window_left, s+causal_off+window_right] (causal_off=kv_len-q_len).
    #
    # Right edge (mask_right): kv_len_wg clips to the WG's max query's attend-limit so
    # we don't run tiles fully past the band. Non-mask_right: all kv (kv_len).
    # Left edge (mask_left): start_tile skips whole tiles before the WG's min query's
    # band start. Non-mask_left: start at tile 0.
    block_x = fx.Int32(gpu.block_id("x"))
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

    # Tile range [start_tile, num_tiles): num_tiles from the right-clipped kv_len_wg;
    # start_tile skips whole tiles before the WG's min query's band start. The
    # defensive min() keeps start_tile a valid buffer index even for an over-launched
    # WG whose whole band is empty (its per-element masks zero the work anyway).
    num_tiles = fx.ceildiv(kv_len_wg, fx.Int32(n_block))
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

    # ---- Prologue (reordered for the mode-2 hang investigation): compute all K/V
    # addresses AND the loop-init in the Q global-load shadow, then run part2 (Q
    # ds_load), then issue the K/V cluster_loads LAST — so NOTHING runs between the
    # loads and the prologue barrier below. Sequence: (1) part1 [above] -> (2) KMgr
    # param calc -> (3) loop init -> (4) part2 -> (5) K cluster_load -> (6) V
    # cluster_load.

    # (2) KMgr param calc — pure address arithmetic (no memory op), hoisted into the
    # Q global-load shadow.
    #
    # Slot rotation is LOCAL to this WG's tile stream: the prologue puts start_tile into
    # slot 1 and the loop carries the slot bases as iter_args (rotated in the yield), so
    # start_tile is irrelevant to the placement and no runtime "% N_KV_PP" is evaluated.
    # Body u reads K from slot 1 and V from slot 0, so the first body needs tile
    # start_tile's K AND V in slot 1 (K for QK(start_tile), V for the next body's PV) and
    # only FINITE data in slot 0's V region -- its PV is the dead leading one, p == 0, and
    # 0 * NaN would poison O. Loading start_tile's V there is the cheapest such filler.
    start_row0 = start_tile * fx.Int32(n_block)
    # This wave's slot among its half's KV_PRODUCER_WARPS producers: it owns a dense
    # n_block/KV_PRODUCER_WARPS row band and copies it alone (one tensor_load per pow2
    # hdim segment).
    _producer_warp = warp_idx % fx.Int32(KV_PRODUCER_WARPS)
    _kv_ctx = ProducerCtx(
        producer_warp=_producer_warp,
        num_producer_warps=KV_PRODUCER_WARPS,
        lane_idx=lane_idx,
    )

    def _get_kv_desc(slot, row0, valid):
        """This half's BufferOpDescriptor for one tile: LO issues every K copy, HI every
        V copy. Pure -- no memory op until ``.async_load()`` -- so the address VALU it
        costs can sit in an unrelated load's shadow. Which transport the descriptor carries
        (async vs TDM) is the manager's business, not this call site's."""
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
    # Built here, issued below: the descriptor is pure, so the address VALU stays in the
    # Q global-load shadow. LO's second copy is K(start+1) into slot 2, the tile the
    # body no longer issues once K runs ahead; HI's is V(start) into slot 0, read by
    # body start's dead PV.
    if KV_K_AHEAD and warp_type.is_lo:
        fill_tile = start_tile + fx.Int32(1)
        kv_fill = _get_kv_desc(
            _k_lds_buf(_PSLOT[2]), _tile_row0(fill_tile), _kv_valid(fill_tile)
        )
    else:
        kv_fill = _get_kv_desc(_k_lds_buf(_PSLOT[0]), start_row0, _kv_valid(start_tile))
    # Fence depths, straight off the descriptor's declared counter usage. A counter this
    # half's copies never touch gets -1 ("do not wait on it at all"), which is NOT the
    # same as waiting for 0. The one it does touch gets this half's per-tile copy count,
    # the depth a KV_PARTIAL_FENCE names to leave the newest tile in flight: both
    # counters take a depth (``tensor_wait(N)`` / ``s_wait_asynccnt(N)``) and both retire
    # in issue order, so it means the same thing on either transport. It is a per-HALF
    # count -- LO's K band and HI's V band differ whenever qk_hdim != v_hdim.
    num_tdm_copies = kv0.tensorcnt if kv0.tensorcnt else -1
    num_async_copies = kv0.asynccnt if kv0.asynccnt else -1
    _kv_drain = _kv_drain_depths(num_tdm_copies, num_async_copies)
    # What this half copies in the prologue, transport-independent: its tile, plus the
    # fill -- except on LO without K-ahead, where the fill is the same tile again and
    # ``kv_fill`` is built only to be dropped.
    _tiles = [kv0]
    if KV_K_AHEAD or not warp_type.is_lo:
        _tiles.append(kv_fill)
    # When each goes out is NOT transport-independent. Under V2, copies whose destination
    # misses Q go out BEFORE Q is read, so their global latency overlaps Q's and part2
    # waits tensorcnt down to them instead of to 0; only LO's K-ahead fill (logical slot
    # 2) lands on Q, so that one waits for the barrier. Under V1 there is no such overlap
    # to have: Q's own stage is on asynccnt too, and its part2 waits that counter to a
    # depth counted over Q's loads ALONE, so a tile copy issued first would corrupt the
    # count. Everything goes after the barrier -- and after part2, which ends at depth 0,
    # leaving asynccnt carrying this half's K/V copies alone from here on.
    if not USE_TDM_LOADER:
        _early, _late = [], _tiles
    elif KV_K_AHEAD and warp_type.is_lo:
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
    # HI enters its resting priority here, at the first point both halves have
    # reached. Before this the two are still in the symmetric Q/KV prologue; from
    # here on the lagging half is the one that must not lose arbitration to its
    # SIMD-mate, and _pv_qk_gemm's exit restores this same level after every gemm.
    # Compile-time: warp_type is a Python constant, so LO traces no instruction.
    if not warp_type.is_lo:
        rocdl.s_setprio(1)
    for _d in _late:
        _d.async_load()
    _kv_fence(*_kv_drain)

    # (7) Loop init — MOVED to after the prologue barrier (ordering experiment). Online-
    # softmax seed + O accumulators (iter_args) and loop bounds.
    #
    # Loop-carried state (the runtime `for ... init=` carry): the online-softmax running
    # max ``m`` and denom ``d`` (per-lane f32), then the ``d_tiles`` fp32 O accumulators. Seed
    # m=-inf, d=0, O=0: the first tile's corr=exp2(m_prev-m_new)=0 zeroes the
    # (already-zero) O before its PV adds in — the standard flash seed. (Fully-masked
    # leading tiles under a finite-left window would make exp2(-inf-(-inf))=NaN;
    # _softmax sanitizes that on the q_min path.)
    #
    # Attention sink (compile-time): the sink is one extra ``exp(sink)`` term in the
    # softmax denominator. Fold it in by seeding m=sink[q_head]*LOG2E (m is log2-domain)
    # and d=1.0 (=exp(sink-sink)); the rescales carry that d seed to exactly
    # exp(sink - m_final), the sink denom term. (Without a sink, m=-inf makes the first tile's corr zero the d seed,
    # so d=1 would equal d=0 — the no-sink path keeps d=0 to stay byte-for-byte.)
    d_tiles = v_hdim // WMMA_M
    R = WMMA_ROW_PER_WAVE
    NKV = n_block // WMMA_N
    # per-q-tile carried state: [m, d, O_0 .. O_{d_tiles-1}, P_0 .. P_{NKV-1}]. P is the
    # software pipeline: body u's PV consumes the P body u-1's softmax produced, so m/d/O
    # keep their offsets and the epilogue's indexing is untouched.
    _QS = 2 + d_tiles + NKV
    # HI half runs softmax(u-1) before gemm(u), so its per-q-tile carry slot holds the
    # f32 s_acc the next body's softmax consumes instead of the bf16 P. Same slot count.
    _lag_sm = ANTI_PHASE and not warp_type.is_lo
    if LAG_DRAIN_AFTER_GEMM and _lag_sm:
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
    # _init = R copies of [m, d, O_tile0 .. O_tile{d_tiles-1}, P_0 .. P_{NKV-1}] — per
    # q-tile running max, denom, then one v8-f32 O accumulator per 16-wide output-dim tile
    # (this lane's partial O[q, d]), then the carried softmax P, all zero. The R q-tiles
    # have independent online-softmax state. P == 0 makes the first body's PV the dead
    # leading one (0 * V onto the zero O), so no prologue QK/softmax trace is needed.
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
    # writes tile i+2 at byte base 2 (= slot i+2 == slot i-1). Rotation is pure register
    # renaming, so no runtime "% N_KV_PP" is ever evaluated. Base count per mgr is
    # manager-defined (== KV_LDS_SPLITS today) — carried generically. ----
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
    # carrying P across the back edge. That puts the two gemms back to back (Stage 3;
    # a shared ring feeds both) and leaves exactly three live slots: V(u-1) in slot 0,
    # K(u) in slot 1, tile u+1's copy landing in slot 2. Slots are selected by the
    # carried ds pointers / bases, left-rotated at the end of each `main_loop`.
    #
    # u runs [start_tile, num_tiles] -- ONE body more than there are tiles. Both extra
    # half-bodies are dead rather than peeled (an extra trace costs far more than an
    # extra iteration): body start_tile's PV multiplies the seeded P == 0, and body
    # num_tiles' QK/softmax sits at kv_pos_base >= kv_len so every element masks to -inf
    # (m unchanged, corr == 1, P == 0, d unchanged).
    #
    # TODO(perf): go finer still -- per-write-tile async_load interleaved between the
    # PV/QK/softmax ops (order tuned by thread trace) rather than one bulk burst.
    # ========================================================================
    def main_loop(u, state, *, mask_left, mask_right, kv_len):
        # mask_left/mask_right/kv_len shadow the closure flags: the caller splits the
        # tile stream into a mask-free clean region + boundary loops and passes None for
        # any edge this sub-loop provably doesn't cross (compile-time gate). The split
        # points are already expressed in SOFTMAX tiles, so they carry over unchanged.
        # This body's SOFTMAX tile. The lagging half runs softmax one tile behind its
        # gemm, so u-1 (never used past the masks; at u == 0 it only ever masks -inf).
        sm_tile = u - fx.Int32(1) if _lag_sm else u
        kv_tile_start = sm_tile * fx.Int32(
            n_block
        )  # softmax tile's first (batch-relative) kv row

        # Unpack loop-carried state — R independent per-q-tile (m, d, O, P) groups,
        # then the shared K/V ds pointers.
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

        # Warp-specialized preamble: same pieces, ordered so the SIMD-mate pair (i / i+4)
        # staggers the V ring head against the global->LDS prefetch. Correctness is
        # warp-type-independent (each wave reads its own resident tile under the workgroup
        # barrier); the stagger is perf-only. The READ (the ring head) and the slot
        # rotation are UNIFORM; only the global->LDS ISSUE branches by USE_TDM_LOADER
        # (V1 cluster_load_async / V2 TDM copy).
        # Tile prefetched by this body, CLAMPED to the last tile rather than guarded off
        # past the end. Every body then issues exactly one tile's copies, so the fence
        # count is uniform and the last iteration needs no peel -- peeling it cost 5.5% on
        # case 10 (a third trace of this body), against ~2 dead L2-resident tile loads per
        # workgroup here. The clamped re-load lands in the slot body num_tiles reads as its
        # dead K, and O stages past all the slots.
        # This half's copy: LO K(u+2) into slot 0 (its K half died at body u-1; its V half
        # is what this body reads, a different chunk), HI V(u+1) into slot 2 as before.
        _ahead = 2 if (KV_K_AHEAD and warp_type.is_lo) else 1
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
            if KV_PARTIAL_FENCE and warp_type.is_lo:
                _kv_fence(num_tdm_copies, num_async_copies, num_dscnt=_NH)
            else:
                _kv_fence(*_kv_drain, num_dscnt=_NH)

        # This half's descriptor for tile ``pf``'s K (LO) or V (HI) into the oldest slot.
        # Built up front: it is pure, so its address VALU has no barrier dependency and
        # overlaps the drain; only the copy itself (pf_desc.async_load()) must stay after
        # the barrier.
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
            # Online softmax over tile ``sm_tile``'s kv axis, INDEPENDENTLY per q-tile.
            # This lane's query (tile qt) attends [q_min, q_max] (batch-relative kv):
            # q_max = seq+causal_off+window_right, q_min = seq+causal_off-window_left
            # (causal_off = kv_len-q_len). None bounds are skipped -> a clean-region tile
            # passes all-None and does zero per-element masking. kv_len is passed on the
            # right-boundary sub-loop (folds the OOB tail into the q_max clamp / standalone
            # tail mask), which is also what neutralizes the dead trailing softmax.
            # K/V are shared, but each q-tile has its own S and running m/d. All R q-tiles
            # go in ONE call so the rows' max/sum tree reductions emit INTERLEAVED (ILP).
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

            # Rescale each q-tile's running O by this tile's corr. On the leading half the
            # rescale CLOSES the body (O already carries PV(u-1)) and the next body's PV
            # accumulates onto the rescaled O -- the same product as rescaling first. On
            # the lagging half it precedes this body's own PV, which is the same identity.
            # When deferral is active the wide `o_acc *= corr` multiply (R*d_tiles*8
            # f32/lane) is gated behind a non-divergent scf.if that fires only when a
            # running max actually moved. ONE branch covers all R rows, not one each:
            # neither row moves in the common case, so the steady state is a single
            # not-taken s_cbranch_vccz. A row that stayed stale has m_new == m_prev, hence
            # corr == 1 exactly, so rescaling it inside the taken branch is the identity.
            # rescale_masks is None -> deferral compiled out, keep the plain multiply.
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
                # The R ballots are folded HERE, not in _softmax: they are VALU-produced
                # SGPRs, so keeping the s_or/s_cmp that read them at the use site leaves
                # the whole row-max chain between the v_cmp and the turnaround. OR the raw
                # masks, not the per-row booleans -- one s_or_b32 covers all R rows.
                mask_any = rescale_masks[0]
                for _m in rescale_masks[1:]:
                    mask_any = mask_any | _m

                @flyc.jit
                def _maybe_rescale_all(o_flat_in, corr_flat, do_rescale):
                    result = o_flat_in
                    if do_rescale:
                        result = [ov * cv for ov, cv in zip(o_flat_in, corr_flat)]
                    return result

                # One flat list for the whole branch: the R rows share the single
                # folded condition, so they rescale (or pass through) together.
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
            if ANTI_PHASE:
                rocdl.sched_barrier(0)
                if _keep_phase_wall:
                    _bare_barrier()
                rocdl.sched_barrier(0)

        # GEMM2(u-1) then GEMM1(u) on one ring: O += P^T(u-1) @ V(u-1), then
        # S^T = K(u) @ Q^T. sched_barrier fences the ring head out of the WMMA stream
        # (no wmma<-ds_load bubble); the ring itself issues the per-fragment s_wait_dscnt.
        if _lag_sm:
            if not LAG_DRAIN_AFTER_GEMM:
                _drain_barrier()
            pf_desc.async_load()
            p_f32, m_new_list, d_new_list, o_resc = _softmax_phase(carry_prev, o_acc)
            # Ring head for the gemm below goes out BEFORE P is narrowed: the head is
            # ds_loads with no dependence on P, so issuing it first puts the conversion in
            # its shadow rather than behind it.
            rocdl.sched_barrier(0)
            pvqk_head = _pvqk_head()
            rocdl.sched_barrier(0)
            p_list = _p_to_elem(p_f32, elem_dtype)
            # Anchor the conversion in THIS block. Its only real use is the wmma stream
            # past the barrier, so MachineSink (which ignores sched_barrier) sinks all 32
            # v_cvt_pk_bf16_f32 into the gemm and interleaves them with the WMMA. A
            # side-effecting use here is a real use, so they stay put.
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
            carry_next = _scale_s(s_acc, _s_scale)
            if LAG_DRAIN_AFTER_GEMM:
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
            s_list = _scale_s(s_list, _s_scale)
            _phase_barrier()
            carry_f32, m_new_list, d_new_list, o_out = _softmax_phase(s_list, o_acc)
            # Next body's ring head: V(u) from the slot this body read K from -- resident
            # and already fenced. Issued behind the softmax, mirroring the lagging half:
            # both halves issue the head immediately before the barrier that precedes the
            # gemm consuming it, and ahead of the narrowing for the same reason as there.
            rocdl.sched_barrier(0)
            head_next = _issue_head(
                lambda j: v_mgr.load_one_to_reg(v_slots[1], j), 0, _NH
            )
            rocdl.sched_barrier(0)
            carry_next = _p_to_elem(carry_f32, elem_dtype)

        # Yield state — R updated (m, d, O, P) groups, then the K/V ds pointers and slot
        # bases left-rotated by one so slot 0 holds tile u (next body's PV) and the oldest
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
    # attention band so interior tiles fully inside the band skip masking. clean_lo/
    # clean_hi are runtime split points, but the mask on/off per sub-loop is COMPILE-TIME
    # (each loop traces main_loop once with fixed None-ness). The rotation state threads
    # continuously through all of them, so the split leaves the slot assignment intact.
    # The bounds are already softmax-tile indices, so the pipeline shift only extends the
    # last loop by one body (the dead-QK one, which needs the kv_len mask).
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
    if mask_left and ANTI_PHASE:
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
    if LAG_DRAIN_AFTER_GEMM and not _lag_sm:
        # Partner for the lagging half's prologue filler -- without it the two halves
        # disagree on the barrier count. It also orders this half's O stores against its
        # OWN last K reads: the lagging half's trailing drain retires those before its
        # last barrier, but this half's last in-loop barrier is the PHASE one, which sits
        # BEFORE the gemm, so wave 1 could otherwise store O into final[0]'s K chunk while
        # wave 0 is still reading K out of it.
        _bare_barrier()

    # ========================================================================
    # Epilogue: normalize O by the running denom d, then reshape+store to VRAM.
    # o_final[dt] lane l elem si = sum_kv P[q,kv] V[kv, dt*16+(l//16)*8+si]
    # (unnormalized); divide by the per-query denom d (peer-consistent across the
    # lane pair) to finish softmax. OManager16b masks rows with seq >= q_len.
    # ========================================================================
    # The R q-tiles serialize through the same O ring (s_wait_dscnt(0) between them).
    _OMgr = {"v1": OManager16bV1, "v2": OManager16bV2, "v3": OManager16bV3}[O_VARIANT]
    o_mgr = _OMgr(
        v_hdim=v_hdim,
        gqa_ratio=gqa_ratio,
        num_waves=NUM_WAVES,
        q_tiles_per_wave=R,
        elem_dtype=elem_dtype,
    )
    # Two waves share a chunk at 0 and LDS_QO_BYTES, and the upper one must stop short of
    # the next chunk -- that clearance is what keeps O off the V rows the dead carried
    # head is still reading (8704 + 17408 = 26112 of 26624 at v_hdim 128).
    assert LDS_QO_BYTES + o_mgr.warp_lds_size_in_byte() <= LDS_CHUNK_BYTES, (
        f"O per-wave {o_mgr.warp_lds_size_in_byte()}B does not fit above the "
        f"{LDS_QO_BYTES}B slice inside a {LDS_CHUNK_BYTES}B chunk"
    )
    # O strides are in ELEMENTS (OManager multiplies by _BF16_BYTES itself). Both V1/V2
    # take ptr_O and build their own store descriptor internally (V1 a bounded buffer
    # resource for the masked buffer_store; V2 the TDM store atom with HW OOB drop).
    # O reuses the two KV slots the loop is done with: body u writes slot_of[2], so after
    # the yield's left-rotation the last-written slot is final[1] and the free pair is
    # (final[2], final[0]) == ((k+1)%3, (k+2)%3). Wave w -> even/odd picks the slot, bit 1
    # the 17 KB half-slice, bit 2 the low/high 156 KB chunk group.
    assert LAG_DRAIN_AFTER_GEMM, "O's KV-chunk reuse needs the trailing rendezvous"
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
    # and final[2] IS an O chunk -- so under K-ahead the drain needs a rendezvous behind it,
    # since a wave only retires its own copies and O's chunk halves are shared wave pairs.
    _kv_wait(*_kv_drain)
    if KV_K_AHEAD:
        _bare_barrier()
    for qt in range(R):
        # Normalize this q-tile's O by its running denom d, then reshape+store to VRAM.
        # o_final[dt] lane l elem si = sum_kv P[q,kv] V[kv, dt*16+(l//16)*8+si]
        # (unnormalized); divide by the per-query denom d (peer-consistent across the
        # lane pair) to finish softmax. OManager16b masks rows with seq >= q_len.
        d_final = fx.Float32(final[qt * _QS + 1])
        o_final = [fx.Vector(final[qt * _QS + 2 + dt]) for dt in range(d_tiles)]
        # Fully-masked row (d_final==0): 1/0=inf, o_final=0, 0*inf=NaN -> guard to O=0.
        inv = (d_final > fx.Float32(0.0)).select(
            fx.Float32(1.0) / d_final, fx.Float32(0.0)
        )
        inv_vec = fx.Vector.from_elements([inv], fx.Float32).broadcast_to(8)
        # NOTE (mode-2): tying o_final through va_vdst here (to cover the final PV-wmma
        # writeback -> this normalize mul) was MEASURED HARMFUL: 8192nc 1/80 -> 9/80 with
        # the same PV fence present. Either va_vdst doesn't reliably track the wmma
        # writeback or the added drain reshuffles RA into a new race. Left uncovered.
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

    # ---- LSE store (optional). LSE = (m_final + log2(d_final)) / LOG2E: m is carried
    # in log2 units of the scaled score, d is domain-free (exp2 of a log2 difference is
    # the natural exp of the natural one), so one multiply converts both — matches
    # torch.logsumexp(scale * Q @ K^T, dim=kv). Each query q = warp*R*16 + qt*16 + l%16
    # is held identically by the lane pair (l, l^16); store once from the khalf==0
    # lanes, masked by seq < q_len. buffer_store redirects mask-drops to byte
    # 0x7FFFFFFF, so lse_rsrc is bounded. Emitted per q-tile.
    if return_lse:
        khalf0 = (lane_idx // fx.Int32(WMMA_M)) == fx.Int32(0)
        lse_rsrc = buffer_ops.create_buffer_resource(
            ptr_LSE, num_records_bytes=lse_num_records_bytes
        )
        for qt in range(R):
            m_final = fx.Float32(final[qt * _QS + 0])
            d_final = fx.Float32(final[qt * _QS + 1])
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
    this WG's valid query rows. LSE=-inf, or (with a sink) LSE=sink[head] since the only
    surviving softmax term is exp(sink) (sink value is 0, O stays 0). Flat coalesced b128
    write — consecutive lanes write consecutive 16-byte O chunks (no WMMA layout)."""
    tid = _warp_id() * fx.Int32(WAVE_SIZE) + _lane_id()
    kv_head = fx.Int32(gpu.block_id("y"))
    row0 = fx.Int32(gpu.block_id("x")) * fx.Int32(BLOCK_M)
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
        prow = row0 + tid  # one LSE per packed row (BLOCK_SIZE threads == BLOCK_M)
        seq = prow // g
        head = kv_head * g + prow % g
        if has_sink:
            num_heads_q = gpu.grid_dim.y * g
            lse_val = _load_sink_logit(ptr_sink, head, num_heads_q)
        else:
            lse_val = fx.Float32(float("-inf"))
        off = (q_start + seq) * stride_lse_seq + head * stride_lse_head
        off_masked = (seq < q_len).select(off * fx.Int32(4), fx.Int32(0x7FFFFFFF))
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
    return_lse: bool = False,
    has_sink: bool = False,
    gqa_ratio: int = 1,
):
    """Build the m32x8 device kernel for a given layout + config.

    ``layout`` is ``"thd"`` (varlen) or ``"bshd"`` (batched). Compile-time
    parameters are captured here and baked into the traced kernel. ``gqa_ratio``
    (= ``nheads_q // nheads_kv``) is compile-time so the per-lane ``% / //`` fold
    to shift/and when it is a power of two.
    """
    assert layout in ("thd", "bshd"), f"layout must be thd|bshd, got {layout!r}"
    # qk_hdim in {128,192,256} (D_qk, WMMA_K multiple); v_hdim fixed at 128 (D_v).
    assert (
        qk_hdim in SUPPORTED_QK_HDIM and v_hdim == 128
    ), f"supports qk_hdim in {SUPPORTED_QK_HDIM} with v_hdim==128, got {qk_hdim}/{v_hdim}"
    assert (
        dtype_str in _DTYPE_MAP
    ), f"dtype_str must be in {list(_DTYPE_MAP)}, got {dtype_str!r}"
    ELEM_DTYPE = _DTYPE_MAP[dtype_str]
    assert gqa_ratio >= 1, f"gqa_ratio must be >= 1, got {gqa_ratio}"
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
    RET_LSE = bool(return_lse)
    HAS_SINK = bool(has_sink)
    GQA_RATIO = int(gqa_ratio)

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
            """Varlen THD entry — empty scaffold.

            THD: this batch's token ranges come from cu_seqlens (batch = grid.z).
            """
            batch = fx.Int32(gpu.block_id("z"))
            q_start, q_end = _load_seqlen_pair(ptr_cu_seqlens_q, batch)
            kv_start, kv_end = _load_seqlen_pair(ptr_cu_seqlens_k, batch)
            q_len = q_end - q_start
            kv_len = kv_end - kv_start

            # LSE is [nheads_q, total_q]. Bound the buffer resource by the last
            # element this batch can touch over BOTH axes (seq = q_len-1, head =
            # nheads_q-1): a seq-only bound is exact only when seq is the major
            # axis. Exactly numel*4 on the last batch, under the 0x7FFFFFFF drop.
            num_heads_q = gpu.grid_dim.y * fx.Int32(GQA_RATIO)
            lse_base_elems = q_start * stride_lse_seq
            lse_num_records_bytes = (
                fx.Int64(q_start + q_len - fx.Int32(1)) * fx.Int64(stride_lse_seq)
                + fx.Int64(num_heads_q - fx.Int32(1)) * fx.Int64(stride_lse_head)
                + fx.Int64(1)
            ) * fx.Int64(4)

            # An empty batch (no queries OR no keys) must NOT enter the core:
            # kv_len==0 gives an empty softmax denom (d=0) and the epilogue would
            # write O/0 = NaN to that batch's query rows; q_len==0 has no rows to
            # write. Self-attn's kv_len==0 implies q_len==0, so this only skips
            # genuinely empty work. (varlen may carry a per-batch kv_len==0 tail.)
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
                # Warp specialization: LO (waves 0..N/2-1) vs HI (N/2..N-1).
                lds_base = _alloc_lds()
                warp_idx = _warp_id()

                def _run(wt):
                    _core_attention(
                        warp_idx=warp_idx, warp_type=wt, lds_base=lds_base, **_ca_kw
                    )

                if warp_idx // fx.Int32(NUM_WAVES // 2) == fx.Int32(0):
                    _run(WarpType.LO)
                else:
                    _run(WarpType.HI)
            elif q_len > fx.Int32(0):
                # Cross-attention tail: q_len>0 but kv_len==0 -> O=0, LSE=-inf (or sink).
                _zero_fill_attention(
                    v_hdim=V_HDIM,
                    gqa_ratio=GQA_RATIO,
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
        """Batched BSHD entry — empty scaffold.

        Uniform sequence lengths (``seq_len_q`` / ``seq_len_k``) replace
        cu_seqlens — nothing transient, so this path is CUDA-graph safe.
        Token base is batch_idx * seq_len (batch = grid.z).
        """
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
        # Warp specialization: LO (waves 0..N/2-1) vs HI (N/2..N-1).
        lds_base = _alloc_lds()
        warp_idx = _warp_id()

        def _run(wt):
            _core_attention(
                warp_idx=warp_idx, warp_type=wt, lds_base=lds_base, **_ca_kw
            )

        if warp_idx // fx.Int32(NUM_WAVES // 2) == fx.Int32(0):
            _run(WarpType.LO)
        else:
            _run(WarpType.HI)

    return kn_fmha_fwd_prefill_a16w16_m32x8_bshd


# ============================================================================
# Launch wrappers + host entries
# ============================================================================

_launch_fns = (
    {}
)  # {(layout, mask_left, mask_right, return_lse, has_sink, gqa_ratio): fn}


def _ensure_thd_kernel(
    mask_left: bool,
    mask_right: bool,
    return_lse: bool,
    has_sink: bool,
    gqa_ratio: int,
    qk_hdim: int = DEFAULT_QK_HDIM,
    dtype_str: str = DEFAULT_DTYPE,
):
    key = (
        "thd",
        bool(mask_left),
        bool(mask_right),
        bool(return_lse),
        bool(has_sink),
        int(gqa_ratio),
        int(qk_hdim),
        str(dtype_str),
    )
    if key in _launch_fns:
        return
    kernel = build_fmha_fwd_prefill_a16w16_m32x8(
        layout="thd",
        qk_hdim=qk_hdim,
        mask_left=mask_left,
        mask_right=mask_right,
        return_lse=return_lse,
        has_sink=has_sink,
        gqa_ratio=gqa_ratio,
        dtype_str=dtype_str,
    )

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
            fx.ceildiv(fx.Uint32(max_seqlen_q * gqa_ratio), fx.Uint32(BLOCK_M))
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
    return_lse: bool,
    has_sink: bool,
    gqa_ratio: int,
    qk_hdim: int = DEFAULT_QK_HDIM,
    dtype_str: str = DEFAULT_DTYPE,
):
    key = (
        "bshd",
        bool(mask_left),
        bool(mask_right),
        bool(return_lse),
        bool(has_sink),
        int(gqa_ratio),
        int(qk_hdim),
        str(dtype_str),
    )
    if key in _launch_fns:
        return
    kernel = build_fmha_fwd_prefill_a16w16_m32x8(
        layout="bshd",
        qk_hdim=qk_hdim,
        mask_left=mask_left,
        mask_right=mask_right,
        return_lse=return_lse,
        has_sink=has_sink,
        gqa_ratio=gqa_ratio,
        dtype_str=dtype_str,
    )

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
            fx.ceildiv(fx.Uint32(seq_len_q * gqa_ratio), fx.Uint32(BLOCK_M))
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
):
    """Host entry — varlen THD, qk_hdim in {128,192,256} / v_hdim=128, bf16 or fp16.

    ``window_size`` (optional): ``(left, right)`` sliding-window bounds. ``-1`` =
    infinite on that side; ``(-1, -1)`` = full attention. ``causal`` forces
    ``right=0``. Finiteness is baked into the kernel (compile-time ``mask_left`` /
    ``mask_right``); the window magnitudes are runtime args, so one variant serves
    any window value.

    ``sink`` (optional): 1-D ``[nheads_q]`` fp32 per-head sink logits in the
    scaled-score domain — one extra ``exp(sink)`` term in the softmax denominator.
    Presence is baked into the kernel at compile time (``has_sink``).

    ``lse`` (optional): caller-provided ``[nheads_q, total_q]`` fp32 output buffer,
    used only when ``return_lse``; allocated here when ``return_lse`` and None.
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
    assert v.shape[-1] == 128, f"Expected v_hdim=128, got {v.shape[-1]}"

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
            (total_q_tokens, nheads_q, 128), dtype=q.dtype, device=q.device
        )
    if return_lse:
        if lse is None:
            # [nheads_q, total_q] is the aiter varlen LSE convention: what
            # flash_attn_varlen_func documents, and what CK and the gfx1250 ASM
            # varlen kernel both return. The kernel is stride-driven, so the
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

    _ensure_thd_kernel(
        mask_left,
        mask_right,
        bool(return_lse),
        has_sink,
        gqa,
        qk_hdim=qk_hdim,
        dtype_str=dtype_str,
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
                dtype_str,
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
):
    """Host entry — batched BSHD ``[B, S, H, D]``, qk_hdim in {128,192,256} / v_hdim=128, bf16 or fp16.

    Uses the dedicated BSHD kernel with a uniform ``seq_len`` scalar (no
    cu_seqlens), so there is nothing transient to bake into a CUDA graph.

    ``window_size`` (optional): ``(left, right)`` sliding-window bounds. ``-1`` =
    infinite on that side; ``(-1, -1)`` = full attention. ``causal`` forces
    ``right=0``. Finiteness is baked into the kernel (compile-time ``mask_left`` /
    ``mask_right``); the window magnitudes are runtime args.

    ``sink`` (optional): 1-D ``[nheads_q]`` fp32 per-head sink logits in the
    scaled-score domain — one extra ``exp(sink)`` term in the softmax denominator.
    Presence is baked into the kernel at compile time (``has_sink``).

    ``lse`` (optional): caller-provided ``[B, nheads_q, S_q]`` fp32 output buffer,
    used only when ``return_lse``; allocated here when ``return_lse`` and None.
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
    assert v.shape[-1] == 128, f"Expected v_hdim=128, got {v.shape[-1]}"

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
            (batch, seq_len_q, nheads_q, 128), dtype=q.dtype, device=q.device
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

    # Empty tensor — skip the launch (host-known dims, no device sync). No queries: out
    # has no rows to write. No keys (seq_len_k==0, seq_len_q>0): softmax over an empty KV
    # set -> O=0. LSE=-inf, or (with a sink) LSE=sink[head] since the only surviving
    # softmax term is exp(sink) (sink value is 0, so O stays 0).
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

    # BSHD: seq is dim 1, head dim 2 — the per-batch base is derived in-kernel as
    # batch_idx * seq_len. Q/K/V/O strides in ELEMENTS (TDM loaders consume directly).
    stride_q_seq = q.stride(1)
    stride_k_seq = k.stride(1)
    stride_v_seq = v.stride(1)
    stride_o_seq = out.stride(1)
    stride_q_head = q.stride(2)
    stride_k_head = k.stride(2)
    stride_v_head = v.stride(2)
    stride_o_head = out.stride(2)

    _ensure_bshd_kernel(
        mask_left,
        mask_right,
        bool(return_lse),
        has_sink,
        gqa,
        qk_hdim=qk_hdim,
        dtype_str=dtype_str,
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
                dtype_str,
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
