# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""MHA Forward Prefill kernel — ``m16x8`` design, gfx1250 (MI400 / mi450).

A fresh, clean FlyDSL kernel written in the high-level layout-algebra style
(tiled copy / tiled MMA + ``SharedAllocator``) — deliberately independent of the
hand-tuned, assembly-mirroring ``fmha_kernel.py`` (inline ASM, raw TDM
descriptors, ``set_vgpr_bank`` hints, per-WMMA schedule tables).

``m16x8`` names the threadgroup shape: **8 waves per threadgroup**. gfx1250 runs
wave32, so a threadgroup is ``8 * 32 = 256`` threads. (The leading ``16`` is the
WMMA M dimension, one 16-row Q sub-tile per wave.)

Layout support — two device kernels over one shared compute core (option B):
  - ``kn_fmha_fwd_prefill_m16x8_thd``  — varlen THD, driven by ``cu_seqlens``.
  - ``kn_fmha_fwd_prefill_m16x8_bshd`` — batched BSHD, uniform ``seq_len`` scalar
    (no ``cu_seqlens`` tensors → nothing transient to bake into a CUDA graph).
Both resolve their per-workgroup base offsets + sequence bounds, then call the
layout-agnostic ``_core_attention`` helper.

Scope — v1 (this file is intentionally config-agnostic in its name):
  - ``qk_hdim == v_hdim == 128``
  - dtype: bf16 for Q/K/V/O
  - grouped-query attention (GQA): ``gqa = nheads_q // nheads_k``
  - causal and non-causal

``qk_hdim``, ``v_hdim`` and the dtype are compile-time (build-time) parameters
captured by the builder closure, so they never appear in the file name and can
be generalized later without changing the runtime kernel signatures.

Target: gfx1250, wave32, 8 waves per threadgroup (256 threads).
"""

import functools

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm as llvm_dialect
from flydsl._mlir.dialects import rocdl as rocdl_dialect
from flydsl._mlir.dialects import scf
from flydsl.expr import arith, buffer_ops, gpu, rocdl
from flydsl.expr import math as fmath
from flydsl.expr.typing import T
from flydsl.expr.utils.arith import _to_raw as _raw
from ..tensor_shim import _run_compiled

# Q/K/V staging managers (own their LDS swizzles + async copy schedules). They are
# self-contained: this kernel maintains its own arch constants below and passes the
# config each manager needs through its constructor.
from .mha_buffer_managers import QManager16b, KManager16b, VManager16b, OManager16b

# ============================================================================
# Threadgroup / arch constants
# ============================================================================

WAVE_SIZE = 32  # gfx1250 kernels run wave32
NUM_WAVES = 8  # "m16x8" — 8 waves per threadgroup
BLOCK_SIZE = WAVE_SIZE * NUM_WAVES  # 256 threads

# "m16x8": each wave owns a 16-row (WMMA M) Q sub-tile → BLOCK_M = 16 * 8 = 128
# Q rows per threadgroup.
WMMA_M = 16  # query rows per WMMA tile (the "m16" in m16x8; one tile per wave)
WMMA_N = 16  # kv rows per WMMA tile (the S^T=K@Q^T output's n_block-direction axis)
WMMA_K = 32  # WMMA contraction depth (bf16 v_wmma_f32_16x16x32); d-tile width
BLOCK_M = WMMA_M * NUM_WAVES  # 128

# v1 defaults (compile-time; see module docstring).
DEFAULT_QK_HDIM = 128
DEFAULT_V_HDIM = 128
DEFAULT_DTYPE = "bf16"

# KV sequence block (columns of one QK GEMM tile). Configurable; 64 for now.
N_BLOCK_CHOICES = (32, 64, 128, 256)
DEFAULT_N_BLOCK = 64

# Ping-pong K LDS buffers the main loop rotates through (double-buffered prefetch).
N_KV_PP = 2

# log2(e): exp(x) = exp2(x * LOG2E). Softmax uses the native ISA exp2 intrinsic.
LOG2E = 1.4426950408889634

# NOTE: the remaining tiling constants (chunk sizes, K/V write-tile + V swizzle
# granularity) live inside mha_buffer_managers.py — they are intrinsic to the
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
        rocdl_dialect.mbcnt_lo(T.i32, fx.Int32(-1).ir_value(), fx.Int32(0).ir_value())
    )


def _load_seqlen_pair(ptr_tensor, idx):
    """Load ``ptr_tensor[idx]`` and ``ptr_tensor[idx + 1]`` (adjacent i32s) as one
    ``vector<2xi32>``; returns ``(start, end)`` as ``fx.Int32``.

    The two values are contiguous and the address is uniform (derived from
    ``block_id``), so a single 64-bit load should lower to one ``s_load_b64``.
    """
    p = fx.get_iter(ptr_tensor)
    pair = fx.ptr_load(p + fx.Int64(idx), result_type=fx.Vector.make_type(2, fx.Int32))
    return fx.Int32(pair[0]), fx.Int32(pair[1])


def _packed_tile_indices(gqa_ratio, warp_idx, lane_idx):
    """Map this lane's row in the packed ``(seq, q_head_in_group)`` tile to global
    indices; returns ``(kv_head, q_head_idx, seq_idx)`` (all ``fx.Int32``).

    GQA head x seq packing:
      block_id x -> tile over one kv-head's ``(seq, q_head_in_group)`` plane
      block_id y -> kv_head
    ``q_head_in_group`` is the fast axis, so the ``% / //`` use the small (often
    power-of-two) ``gqa_ratio``. Each of the ``BLOCK_M`` rows is an independent
    query sharing this kv-head's K/V.
    """
    kv_head = fx.Int32(gpu.block_id("y"))
    row_idx = (
        warp_idx * WMMA_M + lane_idx % WMMA_M + fx.Int32(gpu.block_id("x")) * BLOCK_M
    )
    q_head_idx = kv_head * gqa_ratio + row_idx % gqa_ratio
    seq_idx = row_idx // gqa_ratio
    return kv_head, q_head_idx, seq_idx


def _min_i32(a, b):  # signed 32-bit min of two fx.Int32
    return fx.Int32(arith.minsi(arith.unwrap(a), arith.unwrap(b)))


def _max_i32(a, b):  # signed 32-bit max of two fx.Int32
    return fx.Int32(arith.maxsi(arith.unwrap(a), arith.unwrap(b)))


# ============================================================================
# Compute stages — EMPTY, unwired. Implemented and tested one at a time; the KV
# streaming driver below lands (and is tested) first with these left inert.
# ============================================================================


def _wmma_bf16(a, b, c):
    """v_wmma_f32_16x16x32_bf16 (gfx1250, wave32): C[16x16 f32] = A[16x32 bf16] @
    B[32x16 bf16] + C. No fdsl wrapper exists for this op (only mfma/fp8/f4), so
    we unwrap operands and call the raw ODS builder locally.

    a/b: v16 bf16 fragments; c: v8 f32 accumulator; returns the v8 f32 result
    (raw MLIR value, feed straight back as ``c`` to accumulate)."""
    v8f32 = fx.Vector.make_type(8, fx.Float32)
    return rocdl_dialect.wmma_f32_16x16x32_bf16(
        v8f32, _raw(a), _raw(b), _raw(c),
        signA=False, signB=False, modC=0, reuseA=False, reuseB=False,
    ).result


def _qk_gemm(*, k_mgr, k_lds, q_frags, n_block, lane_idx, n_inflight=3):
    """GEMM1: S^T = K @ Q^T for one resident KV tile (this wave's 16 Q rows).

    WMMA convention (gfx1250): S^T[kv,q] = K @ Q^T with **K = A-operand** (src_a)
    and **Q = B-operand** (src_b). Contract d in ``NDT = qk_hdim//WMMA_K`` tiles;
    produce ``NKV = n_block//WMMA_N`` kv-tiles. Returns ``s_acc``: a list of NKV
    v8-f32 accumulators (== P^T). GPU-verified accumulator layout: lane ``l``
    element ``si`` holds S^T[kv = kv_tile*WMMA_N + (l//16)*8 + si, q = l%16]
    (kv on the C-row / M axis, q on the C-col / N axis; kv is the 8-strided one).

    K LDS->VGPR reads are software-pipelined: ``n_inflight`` ds_load_b128 stay
    outstanding, drained in issue order with fine-grained ``s_wait_dscnt`` (always
    ``n_inflight-1`` outstanding in steady state, ramping down only for the final
    pair). Each K WMMA fragment is two ds_load_b128 (the two 16-col halves of a
    d-tile) shuffled into a v16 fragment matching the Q frag layout.
    """
    NKV = n_block // WMMA_N          # output kv tiles (WMMA_N kv rows each)
    NDT = len(q_frags)               # contraction d-tiles (== qk_hdim // WMMA_K)

    # Flat schedule of ds_load_b128: (kv, dt, half). kv outer / dt / half inner so
    # a kv-tile's dt accumulations stay contiguous. Two halves shuffle to a v16 frag.
    plan = [(kv, dt, half) for kv in range(NKV) for dt in range(NDT) for half in range(2)]
    n_loads = len(plan)

    def _issue(j):
        kv, dt, half = plan[j]
        col = dt * WMMA_K + half * WMMA_N       # 16-col half within d-tile dt
        return k_mgr.load_lds_to_vgpr_tile_as_k(
            ptr_lds=k_lds, row_idx=kv * WMMA_N, col_idx=col, lane_idx=lane_idx,
        )

    # Prime the pipeline: issue up to n_inflight loads ahead of consumption.
    inflight = []
    for issued in range(min(n_inflight, n_loads)):
        inflight.append(_issue(issued))
    issued = len(inflight)

    s_acc = [None] * NKV
    lo_hold = None
    for j in range(n_loads):
        # Keep n_inflight-1 loads outstanding so load j is the one that just retired;
        # only the final pair ramps this down (nothing left to refill with).
        rocdl.s_wait_dscnt(min(n_inflight - 1, n_loads - j - 1))
        val = inflight.pop(0)
        if issued < n_loads:                    # refill to keep n_inflight outstanding
            inflight.append(_issue(issued))
            issued += 1
        if j % 2 == 0:                          # half 0 -> hold, wait for half 1
            lo_hold = val
            continue
        kv, dt, _ = plan[j]
        k_frag = lo_hold.shuffle(val, list(range(16)))   # v16 bf16, concat(lo, hi)
        acc = s_acc[kv] if dt > 0 else fx.Vector.filled(8, 0.0, fx.Float32)
        s_acc[kv] = _wmma_bf16(k_frag, q_frags[dt], acc)
    return s_acc


def _softmax(*, s, m_prev, d_prev, lane_idx, n_block,
            is_causal=False, kv_pos_base=None, q_max=None, kv_len=None):
    """Online-softmax update for one KV tile. ``s`` already includes softmax_scale
    (folded into Q), so exp uses plain LOG2E.

    Layout (from ``_qk_gemm``): ``s`` is a list of ``NKV = n_block//WMMA_N`` v8-f32
    accumulators; this lane owns query ``q = warp*16 + l%16`` and, in tile ``kvt``,
    the kv rows ``kvt*16 + (l//16)*8 + [0..8)`` (its half). The peer lane ``l^16``
    holds the other 8-row half of the same q, so the row max/sum reduce locally over
    (kvt, i) then across the ``shuffle_xor(16)`` partner.

    Args:
      m_prev, d_prev: running max / denom (fx.Float32, shared by the l<->l^16 pair).
      is_causal: when True, mask element with sequence-relative kv position
        ``kv_pos_base + (l//16)*8 + kvt*16 + i`` greater than ``q_max`` (= q_seq +
        (kv_len - q_len)); both ``kv_pos_base`` and ``q_max`` are fx.Int32.
      kv_len: when set (non-causal), mask the same kv position when it is
        ``>= kv_len`` — the last tile's tail rows past the KV length (which the
        clamped OOB load left holding duplicate data). Redundant under causal
        (``q_max <= kv_len-1``), so pass None there to keep that path unchanged.

    Returns ``(p, m_new, d_new, corr)``:
      p:     list of NKV v8 **bf16** — P^T = exp(S^T - m_new) (PV B-operand).
      m_new: updated running max (fx.Float32).
      d_new: updated running denom = corr*d_prev + rowsum(p) (fx.Float32).
      corr:  exp(m_prev - m_new), the O-accumulator rescale factor (fx.Float32).
    """
    NKV = n_block // WMMA_N
    f32 = ir.F32Type.get()
    fast = arith.FastMathFlags.fast
    neg_inf = fx.Float32(float("-inf"))
    zero = fx.Float32(0.0)
    log2e = fx.Float32(LOG2E)

    def fmax(a, b):
        return fx.Float32(arith.MaxNumFOp(_raw(a), _raw(b), fastmath=fast).result)

    def fadd(a, b):
        return fx.Float32(arith.addf(_raw(a), _raw(b), fastmath=fast))

    def fsub(a, b):
        return fx.Float32(arith.subf(_raw(a), _raw(b), fastmath=fast))

    def fmul(a, b):
        return fx.Float32(arith.mulf(_raw(a), _raw(b), fastmath=fast))

    def exp2(x):
        return fx.Float32(rocdl.exp2(f32, _raw(x)))

    # permlanex16 selectors: identity cross-16 gather (nibbles 0..15) => lane l<->l^16.
    sel_lo, sel_hi = _raw(fx.Int32(0x76543210)), _raw(fx.Int32(0xFEDCBA98))

    def peer(v):  # cross-lane reduce partner: lane l <-> l^16 (the other kv half)
        return fx.Float32(rocdl_dialect.permlanex16(
            f32, _raw(v), _raw(v), sel_lo, sel_hi, fi=False, bound_control=False,
        ))

    khalf = lane_idx // fx.Int32(WMMA_M)  # 0/1: which 8-row kv half this lane owns

    # ---- Pass 1: masked S values + running row max ----
    s_masked = []  # flattened (kvt, i) order
    for kvt in range(NKV):
        svec = fx.Vector(s[kvt])
        for i in range(8):
            sval = fx.Float32(svec[i])
            if is_causal or kv_len is not None:
                kv_pos = kv_pos_base + khalf * fx.Int32(8) + fx.Int32(kvt * WMMA_N + i)
                if is_causal:
                    sval = (kv_pos > q_max).select(neg_inf, sval)
                if kv_len is not None:
                    sval = (kv_pos >= kv_len).select(neg_inf, sval)
            s_masked.append(sval)

    local_max = s_masked[0]
    for v in s_masked[1:]:
        local_max = fmax(local_max, v)
    row_max = fmax(local_max, peer(local_max))
    m_new = fmax(m_prev, row_max)

    # corr = exp(m_prev - m_new); neg_m = -(m_new * log2e) for the fused p exp.
    corr = exp2(fmul(fsub(m_prev, m_new), log2e))
    neg_m = fsub(zero, fmul(m_new, log2e))

    # ---- Pass 2: p = exp(S - m_new) (bf16, per tile) + running row sum ----
    p = []
    local_sum = zero
    idx = 0
    for kvt in range(NKV):
        pe = []
        for i in range(8):
            # exp2(s*log2e - m_new*log2e) via one fma.
            pj = exp2(fx.Float32(fmath.fma(_raw(s_masked[idx]), _raw(log2e), _raw(neg_m))))
            local_sum = fadd(local_sum, pj)
            pe.append(pj)
            idx += 1
        p.append(fx.Vector.from_elements(pe, fx.Float32).to(fx.BFloat16))

    d_new = fadd(fmul(corr, d_prev), fadd(local_sum, peer(local_sum)))
    return p, m_new, d_new, corr


def _pv_gemm(*, v_mgr, v_lds, p, v_hdim, n_block, lane_idx, o_acc=None):
    """GEMM2: O^T = V^T @ P^T for one resident KV tile — the second WMMA.

    WMMA convention (gfx1250): D[M=d, N=q] with **A = V^T** (src_a, transpose-loaded
    via ds_load_tr16_b128) and **B = P^T** (src_b, the bf16 softmax output). Contract
    kv in ``nkt = n_block//WMMA_K`` tiles (K=32); produce ``d_tiles = v_hdim//WMMA_M``
    output d-tiles (M axis). Returns ``o_acc``: a list of ``d_tiles`` v8-f32
    accumulators; lane ``l`` element ``si`` of tile ``dt`` holds
    O[q = l%16, d = dt*WMMA_M + (l//16)*8 + si] — the OManager16b frag layout.

    Online accumulation: pass the running ``o_acc`` (already rescaled by ``corr``);
    each tile's PV adds onto it. ``o_acc=None`` starts from zero (single tile / first
    tile). Each WMMA operand is a v16 bf16 fragment = two 16-wide halves shuffled:
    A from V-tiles (kv, kv+16), B from softmax tiles (p[2kt], p[2kt+1]).
    """
    d_tiles = v_hdim // WMMA_M       # output d-tiles (M axis, WMMA_M d rows each)
    nkt = n_block // WMMA_K          # kv contraction tiles (K=32 kv each)

    out = []
    for dt in range(d_tiles):
        acc = o_acc[dt] if o_acc is not None else fx.Vector.filled(8, 0.0, fx.Float32)
        for kt in range(nkt):
            # A-operand: V^T frag = two 16-kv transpose-load tiles -> v16 bf16.
            v_lo = v_mgr.load_lds_to_vgpr_tile_as_v(
                ptr_lds=v_lds, kv_idx=kt * WMMA_K, d_idx=dt * WMMA_M, lane_idx=lane_idx,
            )
            v_hi = v_mgr.load_lds_to_vgpr_tile_as_v(
                ptr_lds=v_lds, kv_idx=kt * WMMA_K + WMMA_N, d_idx=dt * WMMA_M,
                lane_idx=lane_idx,
            )
            rocdl.s_wait_dscnt(0)
            v_frag = v_lo.shuffle(v_hi, list(range(16)))
            # B-operand: P^T frag = two consecutive softmax kv-tiles -> v16 bf16.
            p_frag = p[2 * kt].shuffle(p[2 * kt + 1], list(range(16)))
            acc = _wmma_bf16(v_frag, p_frag, acc)
        out.append(acc)
    return out


# ============================================================================
# Shared, layout-agnostic compute core
# ============================================================================


def _core_attention(
    *,
    qk_hdim,
    v_hdim,
    n_block,  # compile-time KV block width (columns of one QK GEMM tile)
    is_causal,
    return_lse,
    gqa_ratio,  # compile-time GQA group size = nheads_q // nheads_kv
    ptr_O,
    ptr_Q,
    ptr_K,
    ptr_V,
    ptr_LSE,
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
):
    """Layout-agnostic m16x8 compute — empty scaffold.

    Shared by the THD and BSHD kernel entries. The caller resolves the per-batch
    token ranges (``q_start``/``q_len`` and ``kv_start``/``kv_len``) — the only
    part that differs between varlen and batched layouts — and passes them here.
    """
    warp_idx = _warp_id()
    lane_idx = _lane_id()
    kv_head, q_head_idx, seq_idx = _packed_tile_indices(gqa_ratio, warp_idx, lane_idx)

    # One SharedAllocator per kernel (flydsl constraint); Q/K/V carve their regions
    # from it. Each manager only reports the byte count it needs — the swizzled
    # layout inside each region is the manager's own business.
    smem = fx.SharedAllocator()

    # ---- Q staging in LDS. ----
    q_mgr = QManager16b(qk_hdim=qk_hdim, gqa_ratio=gqa_ratio, num_waves=NUM_WAVES)
    q_smem = smem.allocate(q_mgr.get_lds_size_in_byte())
    q_lds_base = fx.Int32(fx.ptrtoint(q_smem.peek().ptr))

    q_frags = q_mgr.load_q_to_vgpr(
        ptr_Q=ptr_Q,
        stride_q_seq=stride_q_seq,
        stride_q_head=stride_q_head,
        q_start=q_start,
        q_len=q_len,
        kv_head=kv_head,
        block_x=fx.Int32(gpu.block_id("x")),
        warp_idx=warp_idx,
        lane_idx=lane_idx,
        ptr_lds=q_lds_base,
        scale=softmax_scale,
    )

    # ---- K/V staging: N_KV_PP ping-pong slots, K and V interleaved so each slot is a
    # contiguous ``K.ppX | V.ppX``: [K.pp0 | V.pp0][K.pp1 | V.pp1]. The managers report
    # only their own footprint; the caller floors each block at MIN_KV_BLK_BYTES so a
    # slot (k_blk + v_blk) stays >= the O ring budget it later reuses (32KB). ----
    MIN_KV_BLK_BYTES = 16 * 1024
    k_mgr = KManager16b(qk_hdim=qk_hdim, n_block=n_block, num_waves=NUM_WAVES)
    v_mgr = VManager16b(v_hdim=v_hdim, n_block=n_block, num_waves=NUM_WAVES)
    k_blk_bytes = max(k_mgr.get_lds_size_in_byte(), MIN_KV_BLK_BYTES)
    v_blk_bytes = max(v_mgr.get_lds_size_in_byte(), MIN_KV_BLK_BYTES)
    slot_bytes = k_blk_bytes + v_blk_bytes
    kv_smem = smem.allocate(N_KV_PP * slot_bytes)
    kv_lds_base = fx.Int32(fx.ptrtoint(kv_smem.peek().ptr))

    def _k_lds_buf(pp):  # K base of ping-pong slot ``pp`` (int or fx.Int32; folds when const)
        if isinstance(pp, int):
            pp = fx.Int32(pp)
        return kv_lds_base + pp * fx.Int32(slot_bytes)

    def _v_lds_buf(pp):  # V base of ping-pong slot ``pp`` (== K base + k_blk_bytes)
        if isinstance(pp, int):
            pp = fx.Int32(pp)
        return kv_lds_base + pp * fx.Int32(slot_bytes) + fx.Int32(k_blk_bytes)

    # ---- This WG's KV tiles span relative kv [0, kv_len_wg). kv_len_wg is the
    # WG's effective KV length. Non-causal: all kv tokens (== kv_len). Causal: only
    # up to the last query row's attend-limit. Packed row r maps to seq r//gqa_ratio;
    # the WG's largest row is block_x*BLOCK_M+BLOCK_M-1, and a query at seq s attends
    # kv [0, s+causal_off] (causal_off=kv_len-q_len).
    block_x = fx.Int32(gpu.block_id("x"))
    if is_causal:
        causal_off = kv_len - q_len
        wg_max_seq = (
            block_x * fx.Int32(BLOCK_M) + fx.Int32(BLOCK_M - 1)
        ) // fx.Int32(gqa_ratio)
        wg_max_seq = _min_i32(wg_max_seq, q_len - fx.Int32(1))
        kv_len_wg = wg_max_seq + causal_off + fx.Int32(1)
        kv_len_wg = _min_i32(kv_len_wg, kv_len)
        kv_len_wg = _max_i32(kv_len_wg, fx.Int32(1))
    else:
        kv_len_wg = kv_len

    def _kv_valid(blk_row0):
        # How many rows of [blk_row0, blk_row0+n_block) are in-bounds, clamped to
        # the WG's effective KV length kv_len_wg (0..n_block). Past the end -> 0 (a
        # harmless clamped load that is never consumed).
        rem = _max_i32(kv_len_wg - blk_row0, fx.Int32(0))
        return _min_i32(rem, fx.Int32(n_block))

    # Prologue: bulk-load tile 0's K and V into ping-pong buffer 0 — this is the
    # ONLY use of the full-block ``async_load_vram_to_lds`` (issued once per wave).
    # Every later tile is prefetched INSIDE the compute via per-write-tile
    # ``async_load_vram_to_lds_wr_tile`` (interleaved with QK/softmax/PV, ordered by
    # thread-trace tuning) — never here, and never as a bulk call in the loop.
    k_mgr.async_load_vram_to_lds(
        ptr_lds=_k_lds_buf(0),
        ptr_K=ptr_K,
        stride_k_seq=stride_k_seq,
        stride_k_head=stride_k_head,
        kv_head=kv_head,
        kv_row0=kv_start,
        kv_valid=_kv_valid(fx.Int32(0)),
        warp_idx=warp_idx,
        lane_idx=lane_idx,
    )
    v_mgr.async_load_vram_to_lds(
        ptr_lds=_v_lds_buf(0),
        ptr_V=ptr_V,
        stride_v_seq=stride_v_seq,
        stride_v_head=stride_v_head,
        kv_head=kv_head,
        kv_row0=kv_start,
        kv_valid=_kv_valid(fx.Int32(0)),
        warp_idx=warp_idx,
        lane_idx=lane_idx,
    )

    # ========================================================================
    # Main KV loop — stream tiles [0, n_tiles) through the N_KV_PP ping-pong ring.
    #
    # v1 shape (raw scf.for_, per [[fdsl-ast-rewriter-scope]]): a single UNIFORM
    # loop, no first/last peel. Each iter opens by draining outstanding async and
    # barriering, after which the current tile's KV is GUARANTEED resident in LDS
    # (tile 0 from the prologue; every later tile from THIS loop's end-of-body
    # bulk prefetch into the other ping-pong buffer). The body computes on buffer
    # t % N_KV_PP, then prefetches tile t+1 into (t+1) % N_KV_PP.
    #
    # Loop-carried state (scf.for_ iter_args): the online-softmax running max
    # ``m`` and denom ``d`` (per-lane f32), followed by the ``d_tiles`` fp32 O
    # accumulators. Seed m=-inf, d=0, O=0: tile 0's corr=exp2(m_prev-m_new)=0
    # zeroes the (already-zero) O before its PV adds in — the standard flash seed.
    #
    # TODO(perf): replace the end-of-body bulk async_load with per-write-tile
    # async_load_vram_to_lds_wr_tile interleaved between QK/softmax/PV (order
    # tuned by thread trace) so the tile t+1 fetch overlaps tile t's compute.
    # ========================================================================
    d_tiles = v_hdim // WMMA_M
    _init = [
        _raw(fx.Float32(float("-inf"))),
        _raw(fx.Float32(0.0)),
    ] + [_raw(fx.Vector.filled(8, 0.0, fx.Float32)) for _ in range(d_tiles)]

    n_tiles = arith.ceildivui(
        arith.unwrap(kv_len_wg), arith.constant(n_block, type=T.i32)
    )
    _lo = arith.index(0)
    _hi = arith.index_cast(T.index, n_tiles)
    _step = arith.index(1)
    for _tile_iv, _iargs, _loop_res in scf.for_(_lo, _hi, _step, iter_args=_init):
        t = fx.Int32(arith.index_cast(T.i32, _tile_iv))

        # Current tile's KV is already in LDS. Drain the async that filled it, then
        # barrier so no wave still reads the buffer a later prefetch will overwrite.
        rocdl.s_wait_asynccnt(0)
        gpu.barrier()

        # Current tile lives in ping-pong buffer t % N_KV_PP.
        cur_pp = t % fx.Int32(N_KV_PP)
        k_cur = _k_lds_buf(cur_pp)
        v_cur = _v_lds_buf(cur_pp)

        kv_tile_start = t * fx.Int32(n_block)  # this tile's first (batch-relative) kv row

        # Unpack loop-carried state.
        m_prev = fx.Float32(_iargs[0])
        d_prev = fx.Float32(_iargs[1])
        o_acc = [fx.Vector(_iargs[2 + dt]) for dt in range(d_tiles)]

        # ---- GEMM1: S^T = K @ Q^T for this KV tile (== P^T pre-softmax). ----
        s = _qk_gemm(
            k_mgr=k_mgr,
            k_lds=k_cur,
            q_frags=q_frags,
            n_block=n_block,
            lane_idx=lane_idx,
        )

        # ---- Softmax: online update over this KV tile's kv axis. ----
        # Causal masks element kv position
        # kv_tile_start+(l//16)*8+kvt*16+i > seq_idx+causal_off
        # (causal_off = kv_len - q_len); q's sequence index is the GQA-aware seq_idx.
        #
        # TODO(perf): the per-element `.select` lowers to one v_cmp + v_cndmask_b32
        # per masked score (64 pairs for n_block=128). Split the KV loop into a
        # fully-below-diagonal prefix (no mask) + a boundary region (only diagonal
        # tiles masked) so the mask select is gone from the loop's steady state.
        if is_causal:
            q_max = seq_idx + (kv_len - q_len)
        else:
            q_max = None
        p, m_new, d_new, corr = _softmax(
            s=s,
            m_prev=m_prev,
            d_prev=d_prev,
            lane_idx=lane_idx,
            n_block=n_block,
            is_causal=is_causal,
            kv_pos_base=kv_tile_start,
            q_max=q_max,
            # Non-causal needs an explicit KV-length mask for the last tile's tail;
            # causal's q_max already excludes kv >= kv_len, so leave it None there.
            kv_len=(None if is_causal else kv_len),
        )

        # ---- Rescale the running O by corr, then GEMM2 accumulates this tile. ----
        corr_vec = fx.Vector.from_elements([corr], fx.Float32).broadcast_to(8)
        o_resc = [o_acc[dt] * corr_vec for dt in range(d_tiles)]
        o_new = _pv_gemm(
            v_mgr=v_mgr,
            v_lds=v_cur,
            p=p,
            v_hdim=v_hdim,
            n_block=n_block,
            lane_idx=lane_idx,
            o_acc=o_resc,
        )

        # ---- Prefetch tile t+1 into (t+1) % N_KV_PP. The final (past-the-end)
        # prefetch has kv_valid=0, so check_oob clamps every row in-bounds to a
        # harmless load the loop never consumes. No runtime guard needed. ----
        nxt = t + fx.Int32(1)
        nxt_pp = nxt % fx.Int32(N_KV_PP)
        nxt_row0 = nxt * fx.Int32(n_block)
        nxt_valid = _kv_valid(nxt_row0)
        k_mgr.async_load_vram_to_lds(
            ptr_lds=_k_lds_buf(nxt_pp),
            ptr_K=ptr_K,
            stride_k_seq=stride_k_seq,
            stride_k_head=stride_k_head,
            kv_head=kv_head,
            kv_row0=kv_start + nxt_row0,
            kv_valid=nxt_valid,
            warp_idx=warp_idx,
            lane_idx=lane_idx,
        )
        v_mgr.async_load_vram_to_lds(
            ptr_lds=_v_lds_buf(nxt_pp),
            ptr_V=ptr_V,
            stride_v_seq=stride_v_seq,
            stride_v_head=stride_v_head,
            kv_head=kv_head,
            kv_row0=kv_start + nxt_row0,
            kv_valid=nxt_valid,
            warp_idx=warp_idx,
            lane_idx=lane_idx,
        )

        scf.yield_([_raw(m_new), _raw(d_new)] + [_raw(o) for o in o_new])

    # ========================================================================
    # Epilogue: normalize O by the running denom d, then reshape+store to VRAM.
    # o_final[dt] lane l elem si = sum_kv P[q,kv] V[kv, dt*16+(l//16)*8+si]
    # (unnormalized); divide by the per-query denom d (peer-consistent across the
    # lane pair) to finish softmax. OManager16b masks rows with seq >= q_len.
    # ========================================================================
    d_final = fx.Float32(_loop_res[1])
    o_final = [fx.Vector(_loop_res[2 + dt]) for dt in range(d_tiles)]

    inv_vec = (
        fx.Vector.from_elements([fx.Float32(1.0) / d_final], fx.Float32)
        .broadcast_to(8)
    )
    o_norm = [o_final[dt] * inv_vec for dt in range(d_tiles)]

    # O staging rings through the NON-CURRENT K|V slot: iteration t prefetches tile
    # t+1 into slot (t+1)%N_KV_PP, so the last iter leaves a dead prefetch in slot
    # n_tiles%N_KV_PP that no wave consumes. Writing O there is per-warp/intra-wave, so
    # it needs no threadgroup barrier -- only the async drain below so the dead prefetch
    # can't land after O's LDS write.
    o_mgr = OManager16b(v_hdim=v_hdim, gqa_ratio=gqa_ratio, num_waves=NUM_WAVES)
    assert o_mgr.get_lds_size_in_byte() <= slot_bytes, (
        f"O ring budget {o_mgr.get_lds_size_in_byte()}B exceeds K|V slot {slot_bytes}B"
    )
    non_cur_pp = fx.Int32(n_tiles) % fx.Int32(N_KV_PP)
    rocdl.s_wait_asynccnt(0)  # drain the last (dead) tile prefetch before slot reuse
    # O strides are in ELEMENTS (OManager multiplies by _BF16_BYTES itself), unlike the
    # BYTE strides the K/V load path uses -- pass stride_o_* straight through. OManager
    # redirects mask-dropped rows to offset 0x7FFFFFFF, so o_rsrc must bound below that
    # (max_size=True would make the drop land in-bounds and fault); every valid write is
    # < (q_start+q_len)*stride_o_seq bytes, far below 0x7FFFFFFF.
    o_num_records_bytes = (q_start + q_len) * stride_o_seq * fx.Int32(2)
    o_rsrc = buffer_ops.create_buffer_resource(
        ptr_O, num_records_bytes=arith.unwrap(o_num_records_bytes)
    )
    o_mgr.store_o_to_vram(
        o_rsrc=o_rsrc,
        o_base_elems=fx.Int32(0),
        stride_o_seq=stride_o_seq,
        stride_o_head=stride_o_head,
        q_start=q_start,
        q_len=q_len,
        kv_head=kv_head,
        block_x=block_x,
        warp_idx=warp_idx,
        lane_idx=lane_idx,
        ptr_lds=_k_lds_buf(non_cur_pp),
        o_frags=o_norm,
    )

    # ---- LSE store (optional). LSE = m_final + ln(d_final) in the scaled-score
    # domain (softmax_scale is folded into Q, so S already carries it) — matches
    # torch.logsumexp(scale * Q @ K^T, dim=kv). Each query q = warp*16 + l%16 is
    # held identically by the lane pair (l, l^16); store once from the khalf==0
    # lanes (0-15 per warp = the BLOCK_M packed rows), masked by seq < q_len.
    # buffer_store redirects mask-drops to byte 0x7FFFFFFF, so lse_rsrc is bounded.
    if return_lse:
        m_final = fx.Float32(_loop_res[0])
        # fx.log2 lowers to the HW v_log_f32 (base-2), so scale by ln2 (= 1/LOG2E)
        # to get the natural log for LSE = m + ln(d).
        ln_d = fx.log2(d_final) * fx.Float32(1.0 / LOG2E)
        lse_val = m_final + ln_d
        khalf0 = (lane_idx // fx.Int32(WMMA_M)) == fx.Int32(0)
        lse_mask = khalf0 & (seq_idx < q_len)
        lse_off_el = (
            lse_base_elems + seq_idx * stride_lse_seq + q_head_idx * stride_lse_head
        )
        lse_rsrc = buffer_ops.create_buffer_resource(
            ptr_LSE, num_records_bytes=lse_num_records_bytes
        )
        buffer_ops.buffer_store(
            lse_val,
            lse_rsrc,
            lse_off_el * fx.Int32(4),
            mask=lse_mask,
            offset_is_bytes=True,
        )

    del q_head_idx, seq_idx


# ============================================================================
# Builder — one device kernel per (layout, config)
# ============================================================================


@functools.lru_cache(maxsize=None)
def build_fmha_fwd_prefill_m16x8(
    *,
    layout: str = "thd",
    qk_hdim: int = DEFAULT_QK_HDIM,
    v_hdim: int = DEFAULT_V_HDIM,
    n_block: int = DEFAULT_N_BLOCK,
    dtype_str: str = DEFAULT_DTYPE,
    is_causal: bool = False,
    return_lse: bool = False,
    gqa_ratio: int = 1,
):
    """Build the m16x8 device kernel for a given layout + config.

    ``layout`` is ``"thd"`` (varlen) or ``"bshd"`` (batched). Compile-time
    parameters are captured here and baked into the traced kernel. ``gqa_ratio``
    (= ``nheads_q // nheads_kv``) is compile-time so the per-lane ``% / //`` fold
    to shift/and when it is a power of two.
    """
    assert layout in ("thd", "bshd"), f"layout must be thd|bshd, got {layout!r}"
    # v1 supports a single configuration; generalize later.
    assert (
        qk_hdim == 128 and v_hdim == 128
    ), f"v1 supports qk_hdim == v_hdim == 128 only, got {qk_hdim}/{v_hdim}"
    assert dtype_str == "bf16", f"v1 supports bf16 only, got {dtype_str!r}"
    assert gqa_ratio >= 1, f"gqa_ratio must be >= 1, got {gqa_ratio}"
    assert n_block in N_BLOCK_CHOICES, f"n_block must be in {N_BLOCK_CHOICES}, got {n_block}"

    QK_HDIM = qk_hdim
    V_HDIM = v_hdim
    N_BLOCK = int(n_block)
    CAUSAL = bool(is_causal)
    RET_LSE = bool(return_lse)
    GQA_RATIO = int(gqa_ratio)

    if layout == "thd":

        @flyc.kernel(known_block_size=[BLOCK_SIZE, 1, 1])
        def kn_fmha_fwd_prefill_m16x8_thd(
            ptr_O: fx.Pointer,
            ptr_Q: fx.Pointer,
            ptr_K: fx.Pointer,
            ptr_V: fx.Pointer,
            ptr_LSE: fx.Pointer,
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

            # LSE is [total_q, nheads_q]: base = q_start*stride_lse_seq; every valid
            # element offset is < (q_start+q_len)*stride_lse_seq (< the 0x7FFFFFFF drop).
            lse_base_elems = q_start * stride_lse_seq
            lse_num_records_bytes = (q_start + q_len) * stride_lse_seq * fx.Int32(4)

            _core_attention(
                qk_hdim=QK_HDIM,
                v_hdim=V_HDIM,
                n_block=N_BLOCK,
                is_causal=CAUSAL,
                return_lse=RET_LSE,
                gqa_ratio=GQA_RATIO,
                ptr_O=ptr_O,
                ptr_Q=ptr_Q,
                ptr_K=ptr_K,
                ptr_V=ptr_V,
                ptr_LSE=ptr_LSE,
                softmax_scale=softmax_scale,
                stride_q_seq=stride_q_seq,
                stride_k_seq=stride_k_seq,
                stride_v_seq=stride_v_seq,
                stride_o_seq=stride_o_seq,
                stride_q_head=stride_q_head,
                stride_k_head=stride_k_head,
                stride_v_head=stride_v_head,
                stride_o_head=stride_o_head,
                stride_lse_seq=stride_lse_seq,
                stride_lse_head=stride_lse_head,
                lse_base_elems=lse_base_elems,
                lse_num_records_bytes=lse_num_records_bytes,
                q_start=q_start,
                q_len=q_len,
                kv_start=kv_start,
                kv_len=kv_len,
            )

        return kn_fmha_fwd_prefill_m16x8_thd

    @flyc.kernel(known_block_size=[BLOCK_SIZE, 1, 1])
    def kn_fmha_fwd_prefill_m16x8_bshd(
        ptr_O: fx.Pointer,
        ptr_Q: fx.Pointer,
        ptr_K: fx.Pointer,
        ptr_V: fx.Pointer,
        ptr_LSE: fx.Pointer,
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
        lse_num_records_bytes = (lse_base_elems + stride_lse_batch) * fx.Int32(4)

        _core_attention(
            qk_hdim=QK_HDIM,
            v_hdim=V_HDIM,
            n_block=N_BLOCK,
            is_causal=CAUSAL,
            return_lse=RET_LSE,
            gqa_ratio=GQA_RATIO,
            ptr_O=ptr_O,
            ptr_Q=ptr_Q,
            ptr_K=ptr_K,
            ptr_V=ptr_V,
            ptr_LSE=ptr_LSE,
            softmax_scale=softmax_scale,
            stride_q_seq=stride_q_seq,
            stride_k_seq=stride_k_seq,
            stride_v_seq=stride_v_seq,
            stride_o_seq=stride_o_seq,
            stride_q_head=stride_q_head,
            stride_k_head=stride_k_head,
            stride_v_head=stride_v_head,
            stride_o_head=stride_o_head,
            stride_lse_seq=stride_lse_seq,
            stride_lse_head=stride_lse_head,
            lse_base_elems=lse_base_elems,
            lse_num_records_bytes=lse_num_records_bytes,
            q_start=batch * seq_len_q,
            q_len=seq_len_q,
            kv_start=batch * seq_len_k,
            kv_len=seq_len_k,
        )

    return kn_fmha_fwd_prefill_m16x8_bshd


# ============================================================================
# Launch wrappers + host entries
# ============================================================================
# NOTE: v1 scaffold. The device kernel bodies are empty, so launching produces
# no output yet — this wiring exists so the dispatch paths in fmha_kernels.py can
# route qk_hdim==128 here while the kernel is being built out.

_launch_fns = {}  # {(layout, is_causal, return_lse, gqa_ratio): @flyc.jit launch fn}


def _ensure_thd_kernel(is_causal: bool, return_lse: bool, gqa_ratio: int):
    key = ("thd", bool(is_causal), bool(return_lse), int(gqa_ratio))
    if key in _launch_fns:
        return
    kernel = build_fmha_fwd_prefill_m16x8(
        layout="thd", is_causal=is_causal, return_lse=return_lse, gqa_ratio=gqa_ratio
    )

    @flyc.jit
    def _launch(
        ptr_O: fx.Pointer,
        ptr_Q: fx.Pointer,
        ptr_K: fx.Pointer,
        ptr_V: fx.Pointer,
        ptr_LSE: fx.Pointer,
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
        max_seqlen_q: fx.Int32,
        max_seqlen_k: fx.Int32,
        num_heads_kv: fx.Int32,
        batch_size: fx.Int32,
        stream: fx.Stream,
    ):
        # 3D grid: x = tiles over (seq, q_head_in_group) per kv-head,
        #          y = kv_head, z = batch. block = 256 (8 waves x wave32).
        grid_x = arith.index_cast(
            T.index,
            arith.ceildivui(
                arith.unwrap(max_seqlen_q * gqa_ratio),
                arith.constant(BLOCK_M, type=T.i32),
            ),
        )
        grid_y = arith.index_cast(T.index, num_heads_kv)
        grid_z = arith.index_cast(T.index, batch_size)

        launcher = kernel(
            ptr_O,
            ptr_Q,
            ptr_K,
            ptr_V,
            ptr_LSE,
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
            max_seqlen_q,
            max_seqlen_k,
        )
        launcher.launch(
            grid=(grid_x, grid_y, grid_z),
            block=(BLOCK_SIZE, 1, 1),
            stream=stream,
        )

    _launch_fns[key] = _launch


def _ensure_bshd_kernel(is_causal: bool, return_lse: bool, gqa_ratio: int):
    key = ("bshd", bool(is_causal), bool(return_lse), int(gqa_ratio))
    if key in _launch_fns:
        return
    kernel = build_fmha_fwd_prefill_m16x8(
        layout="bshd", is_causal=is_causal, return_lse=return_lse, gqa_ratio=gqa_ratio
    )

    @flyc.jit
    def _launch(
        ptr_O: fx.Pointer,
        ptr_Q: fx.Pointer,
        ptr_K: fx.Pointer,
        ptr_V: fx.Pointer,
        ptr_LSE: fx.Pointer,
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
        seq_len_q: fx.Int32,
        seq_len_k: fx.Int32,
        num_heads_kv: fx.Int32,
        batch_size: fx.Int32,
        stream: fx.Stream,
    ):
        # 3D grid: x = tiles over (seq, q_head_in_group) per kv-head,
        #          y = kv_head, z = batch. block = 256 (8 waves x wave32).
        grid_x = arith.index_cast(
            T.index,
            arith.ceildivui(
                arith.unwrap(seq_len_q * gqa_ratio),
                arith.constant(BLOCK_M, type=T.i32),
            ),
        )
        grid_y = arith.index_cast(T.index, num_heads_kv)
        grid_z = arith.index_cast(T.index, batch_size)

        launcher = kernel(
            ptr_O,
            ptr_Q,
            ptr_K,
            ptr_V,
            ptr_LSE,
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
            seq_len_q,
            seq_len_k,
        )
        launcher.launch(
            grid=(grid_x, grid_y, grid_z),
            block=(BLOCK_SIZE, 1, 1),
            stream=stream,
        )

    _launch_fns[key] = _launch


def flash_attn_varlen_m16x8(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    softmax_scale=None,
    causal=False,
    out=None,
    return_lse=False,
):
    """Host entry — varlen THD, qk_hdim=v_hdim=128, bf16.

    v1 scaffold: builds and launches the empty THD kernel and returns the
    (unwritten) output tensor.
    """
    assert q.dtype == torch.bfloat16, f"Expected bf16, got {q.dtype}"
    assert q.shape[-1] == 128, f"Expected qk_hdim=128, got {q.shape[-1]}"
    assert v.shape[-1] == 128, f"Expected v_hdim=128, got {v.shape[-1]}"

    total_q_tokens = q.shape[0]
    batch = cu_seqlens_q.shape[0] - 1
    nheads_q = q.shape[1]
    nheads_k = k.shape[1]
    gqa = nheads_q // nheads_k

    if softmax_scale is None:
        softmax_scale = 1.0 / (q.shape[-1] ** 0.5)

    if out is None:
        out = torch.empty(
            (total_q_tokens, nheads_q, 128), dtype=torch.bfloat16, device=q.device
        )
    if return_lse:
        lse = torch.empty(
            (total_q_tokens, nheads_q), dtype=torch.float32, device=q.device
        )
    else:
        lse = torch.empty(
            (batch, nheads_q, max_seqlen_q), dtype=torch.float32, device=q.device
        )

    # Byte strides for Q/K/V, element strides for O — matches the gfx1250 fmha
    # family convention; revisit when the clean kernel body is implemented.
    bpp = q.element_size()
    stride_q_seq = q.stride(0) * bpp
    stride_k_seq = k.stride(0) * bpp
    stride_v_seq = v.stride(0) * bpp
    stride_o_seq = out.stride(0)
    stride_q_head = q.stride(1) * bpp
    stride_k_head = k.stride(1) * bpp
    stride_v_head = v.stride(1) * bpp
    stride_o_head = out.stride(1)
    # LSE (element strides). When return_lse it is [total_q, nheads_q]; otherwise a
    # scratch tensor that the kernel never writes, so 0 strides are harmless.
    if return_lse:
        stride_lse_seq = lse.stride(0)
        stride_lse_head = lse.stride(1)
    else:
        stride_lse_seq = 0
        stride_lse_head = 0

    _ensure_thd_kernel(bool(causal), bool(return_lse), gqa)

    _run_compiled(
        _launch_fns[("thd", bool(causal), bool(return_lse), gqa)],
        out,
        q,
        k,
        v,
        lse,
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
        max_seqlen_q,
        max_seqlen_k,
        nheads_k,
        batch,
        torch.cuda.current_stream(),
    )

    if return_lse:
        return out, lse
    return out


def flash_attn_batch_m16x8(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale=None,
    causal=False,
    out=None,
    return_lse=False,
):
    """Host entry — batched BSHD ``[B, S, H, D]``, qk_hdim=v_hdim=128, bf16.

    Uses the dedicated BSHD kernel with a uniform ``seq_len`` scalar (no
    cu_seqlens), so there is nothing transient to bake into a CUDA graph.

    v1 scaffold: builds and launches the empty BSHD kernel and returns the
    (unwritten) output tensor.
    """
    assert q.dtype == torch.bfloat16, f"Expected bf16, got {q.dtype}"
    assert q.dim() == 4, f"Expected 4D BSHD tensor, got rank {q.dim()}"
    assert q.shape[-1] == 128, f"Expected qk_hdim=128, got {q.shape[-1]}"
    assert v.shape[-1] == 128, f"Expected v_hdim=128, got {v.shape[-1]}"

    batch, seq_len_q, nheads_q, _ = q.shape
    seq_len_k = k.shape[1]
    nheads_k = k.shape[2]
    gqa = nheads_q // nheads_k

    if softmax_scale is None:
        softmax_scale = 1.0 / (q.shape[-1] ** 0.5)

    if out is None:
        out = torch.empty(
            (batch, seq_len_q, nheads_q, 128), dtype=torch.bfloat16, device=q.device
        )
    # LSE is [B, H, S_q] (scratch when return_lse is False).
    lse = torch.empty(
        (batch, nheads_q, seq_len_q), dtype=torch.float32, device=q.device
    )

    # Byte strides for Q/K/V, element strides for O. BSHD: seq is dim 1, head
    # dim 2 — the per-batch base is derived in-kernel as batch_idx * seq_len.
    bpp = q.element_size()
    stride_q_seq = q.stride(1) * bpp
    stride_k_seq = k.stride(1) * bpp
    stride_v_seq = v.stride(1) * bpp
    stride_o_seq = out.stride(1)
    stride_q_head = q.stride(2) * bpp
    stride_k_head = k.stride(2) * bpp
    stride_v_head = v.stride(2) * bpp
    stride_o_head = out.stride(2)
    # LSE is [B, nheads_q, seq_q] (element strides; batch stride bounds the resource).
    stride_lse_seq = lse.stride(2)
    stride_lse_head = lse.stride(1)
    stride_lse_batch = lse.stride(0)

    _ensure_bshd_kernel(bool(causal), bool(return_lse), gqa)

    _run_compiled(
        _launch_fns[("bshd", bool(causal), bool(return_lse), gqa)],
        out,
        q,
        k,
        v,
        lse,
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
        seq_len_q,
        seq_len_k,
        nheads_k,
        batch,
        torch.cuda.current_stream(),
    )

    if return_lse:
        return out, lse
    return out
