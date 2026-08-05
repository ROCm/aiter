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
            is_causal=False, kv_pos0=None, q_max=None):
    """Online-softmax update for one KV tile. ``s`` already includes softmax_scale
    (folded into Q), so exp uses plain LOG2E.

    Layout (from ``_qk_gemm``): ``s`` is a list of ``NKV = n_block//WMMA_N`` v8-f32
    accumulators; this lane owns query ``q = warp*16 + l%16`` and, in tile ``kvt``,
    the kv rows ``kvt*16 + (l//16)*8 + [0..8)`` (its half). The peer lane ``l^16``
    holds the other 8-row half of the same q, so the row max/sum reduce locally over
    (kvt, si) then across the ``shuffle_xor(16)`` partner.

    Args:
      m_prev, d_prev: running max / denom (fx.Float32, shared by the l<->l^16 pair).
      is_causal: when True, mask element with sequence-relative kv position
        ``kv_pos0 + (l//16)*8 + kvt*16 + si`` greater than ``q_max`` (= q_seq +
        (kv_len - q_len)); both ``kv_pos0`` and ``q_max`` are fx.Int32.

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

    def peer(v):  # cross-lane reduce partner: lane l <-> l^16 (the other kv half)
        return fx.Float32(v).shuffle_xor(fx.Int32(16), fx.Int32(WAVE_SIZE))

    khalf = lane_idx // fx.Int32(WMMA_M)  # 0/1: which 8-row kv half this lane owns

    # ---- Pass 1: masked S values + running row max ----
    s_masked = []  # flattened (kvt, si) order
    for kvt in range(NKV):
        svec = fx.Vector(s[kvt])
        for si in range(8):
            sval = fx.Float32(svec[si])
            if is_causal:
                kpos = kv_pos0 + khalf * fx.Int32(8) + fx.Int32(kvt * WMMA_N + si)
                sval = (kpos > q_max).select(neg_inf, sval)
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
        for si in range(8):
            # exp2(s*log2e - m_new*log2e) via one fma.
            pj = exp2(fx.Float32(fmath.fma(_raw(s_masked[idx]), _raw(log2e), _raw(neg_m))))
            local_sum = fadd(local_sum, pj)
            pe.append(pj)
            idx += 1
        p.append(fx.Vector.from_elements(pe, fx.Float32).to(fx.BFloat16))

    d_new = fadd(fmul(corr, d_prev), fadd(local_sum, peer(local_sum)))
    return p, m_new, d_new, corr


def _pv_gemm(**kw):
    """GEMM2: O^T += V @ P^T for the current KV tile.

    Reads V^T (ds_load_tr16_b128, WMMA A-operand) and contracts it against the
    bf16 P^T B-frags, accumulating into the fp32 O accumulator. Empty for now.
    """
    raise NotImplementedError("PvGemm not implemented yet")


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

    # ---- K staging: N_KV_PP ping-pong buffers the main loop rotates through.
    # KManager owns only one block's swizzle/size; the ring is the caller's. ----
    k_mgr = KManager16b(qk_hdim=qk_hdim, n_block=n_block, num_waves=NUM_WAVES)
    k_blk_bytes = k_mgr.get_lds_size_in_byte()
    k_smem = smem.allocate(N_KV_PP * k_blk_bytes)
    k_lds_base = fx.Int32(fx.ptrtoint(k_smem.peek().ptr))

    def _k_lds_buf(pp):  # base of ping-pong buffer ``pp`` (int or fx.Int32; folds when const)
        if isinstance(pp, int):
            pp = fx.Int32(pp)
        return k_lds_base + pp * fx.Int32(k_blk_bytes)

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

    # ---- V staging: own swizzle (transpose-load friendly), N_KV_PP ping-pong. ----
    v_mgr = VManager16b(v_hdim=v_hdim, n_block=n_block, num_waves=NUM_WAVES)
    v_blk_bytes = v_mgr.get_lds_size_in_byte()
    v_smem = smem.allocate(N_KV_PP * v_blk_bytes)
    v_lds_base = fx.Int32(fx.ptrtoint(v_smem.peek().ptr))

    def _v_lds_buf(pp):  # base of ping-pong buffer ``pp`` (int or fx.Int32; folds when const)
        if isinstance(pp, int):
            pp = fx.Int32(pp)
        return v_lds_base + pp * fx.Int32(v_blk_bytes)

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
    # (tile 0 from the prologue; every later tile from the previous iter's
    # interleaved wr_tile prefetch). The body then just computes on buffer
    # t % N_KV_PP. Prefetch of tile t+1 (into (t+1)%N_KV_PP) is issued INSIDE the
    # compute stages via async_load_vram_to_lds_wr_tile (order tuned by thread
    # trace), NOT as a bulk load here.
    #
    # Compute is UNWIRED for now: QkGemm/Softmax/PvGemm land + are tested one at a
    # time. When they do, (O_accu, row_max, row_sum) become scf.for_ iter_args
    # (init before the loop, normalise after) and the per-wave causal predicates
    # below gate them.
    # ========================================================================
    n_tiles = arith.ceildivui(
        arith.unwrap(kv_len_wg), arith.constant(n_block, type=T.i32)
    )
    _lo = arith.index(0)
    _hi = arith.index_cast(T.index, n_tiles)
    _step = arith.index(1)
    for _tile_iv in scf.for_(_lo, _hi, _step):
        t = fx.Int32(arith.index_cast(T.i32, _tile_iv))

        # Current tile's KV is already in LDS. Drain the async that filled it, then
        # barrier so no wave still reads the buffer a later prefetch will overwrite.
        rocdl.s_wait_asynccnt(0)
        gpu.barrier()

        # Current tile lives in ping-pong buffer t % N_KV_PP.
        cur_pp = t % fx.Int32(N_KV_PP)
        k_cur = _k_lds_buf(cur_pp)

        n_start = t * fx.Int32(n_block)  # this tile's first (batch-relative) kv row

        # ---- GEMM1: S^T = K @ Q^T for this KV tile (== P^T pre-softmax). ----
        s = _qk_gemm(
            k_mgr=k_mgr,
            k_lds=k_cur,
            q_frags=q_frags,
            n_block=n_block,
            lane_idx=lane_idx,
        )

        # ---- Softmax: online update over this KV tile's kv axis. ----
        # v1 tests the SINGLE-tile path (m_prev=-inf, d_prev=0); the running
        # (m,d,O_acc) become scf.for_ iter_args once PV + epilogue land. Causal
        # masks element kv position n_start+(l//16)*8+kvt*16+si > seq_idx+causal_off
        # (causal_off = kv_len - q_len); q's sequence index is the GQA-aware seq_idx.
        if is_causal:
            q_max = seq_idx + (kv_len - q_len)
        else:
            q_max = None
        p, m_new, d_new, corr = _softmax(
            s=s,
            m_prev=fx.Float32(float("-inf")),
            d_prev=fx.Float32(0.0),
            lane_idx=lane_idx,
            n_block=n_block,
            is_causal=is_causal,
            kv_pos0=n_start,
            q_max=q_max,
        )

        # ---- TODO(PV/epilogue, still UNWIRED): _pv_gemm consumes (p, V) into an
        # O accumulator rescaled by corr; m_new/d_new become loop-carried state.
        # v_cur = _v_lds_buf(cur_pp)
        # Prefetch tile t+1 into (t+1)%N_KV_PP via async_load_vram_to_lds_wr_tile,
        # interleaved between QK/softmax/PV (order tuned by thread trace).
        del s, p, m_new, d_new, corr  # unused until PV lands (DCE'd for now)
        scf.yield_([])

    # TODO(epilogue): once PV lands, normalise O by row_sum and write via
    # OManager16b.store_o_to_vram using (q_head_idx, seq_idx); mask seq_idx>=q_len.
    del q_head_idx, seq_idx  # unused until the epilogue lands


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
        seq_len_q: fx.Int32,
        seq_len_k: fx.Int32,
    ):
        """Batched BSHD entry — empty scaffold.

        Uniform sequence lengths (``seq_len_q`` / ``seq_len_k``) replace
        cu_seqlens — nothing transient, so this path is CUDA-graph safe.
        Token base is batch_idx * seq_len (batch = grid.z).
        """
        batch = fx.Int32(gpu.block_id("z"))

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
        seq_len_q,
        seq_len_k,
        nheads_k,
        batch,
        torch.cuda.current_stream(),
    )

    if return_lse:
        return out, lse
    return out
