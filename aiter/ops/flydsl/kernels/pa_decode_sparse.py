# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""FlyDSL implementation of the sparse paged-decode attention kernel for gfx942.

Both KV_SPLITS==1 (direct output) and KV_SPLITS>1 paths are implemented.
The reduce part is not yet implemented (Triton kernel exist for this).

KV_SPLITS > 1
-------------
The Triton _pa_decode_sparse_reduce kernel is reused for the combine step. The FlyDSL
kernel writes (m, l, acc) partials to intermediate fp32 buffers with the same layout
that the Triton reduce kernel expects:
  m_partial   [T, KV_SPLITS, H]    fp32
  l_partial   [T, KV_SPLITS, H]    fp32
  acc_partial [T, KV_SPLITS, H, D] fp32
"""

from __future__ import annotations

import functools
from contextlib import contextmanager

import torch
import triton

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm as mlir_llvm, rocdl as mlir_rocdl, scf
from flydsl.expr import arith, buffer_ops, gpu, range_constexpr, vector
from flydsl.expr import rocdl
from flydsl.expr.arith import ArithValue, CmpFPredicate, CmpIPredicate
from flydsl.expr.typing import T, Int32, Stream as FlyStream
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.runtime.device import get_rocm_arch
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr

from .tensor_shim import STensor, _to_raw

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WAVE_SIZE = 64
_LOG2E = 1.4426950408889634
_NEG_INF = float("-inf")

# ---------------------------------------------------------------------------
# Helper: IfOp then-block context manager
# ---------------------------------------------------------------------------

@contextmanager
def _if_then(if_op):
    """SCF IfOp then-region context manager. Auto-yields empty if missing."""
    with ir.InsertionPoint(if_op.then_block):
        try:
            yield if_op.then_block
        finally:
            blk = if_op.then_block
            if (not blk.operations) or not isinstance(blk.operations[-1], scf.YieldOp):
                scf.YieldOp([])

# ---------------------------------------------------------------------------
# MFMA-based kernel (BLOCK_H=16, 4-wave 256-thread design)
# ---------------------------------------------------------------------------
#
# Architecture summary
# --------------------
# Grid : (T, ceil(H/BLOCK_H), KV_SPLITS)   BLOCK_H=16 heads per CTA
# Block: 256 threads (4 waves x 64 lanes)
# MFMA : mfma_f32_16x16x16bf16_1k — 16x16x16 BF16 accumulate into 4xf32 per lane
#
# Thread layout:
#   wave_id   = tid // 64        (0..3)
#   lane      = tid  % 64        (0..63)
#   lane_row  = lane % 16        — MFMA M/N index (head or kv slot, depending on step)
#   lane_cgrp = lane // 16       — column group within wave's D-slice (0..3)
#   d_wave_base = wave_id * (D // 4)  — D-column base for this wave
#
# MFMA C-output layout:
#   Lane l: c_frag[v] = C[M = (l//16)*4 + v,  N = l%16]
#   In other words: M-index = lane_cgrp*4+v,  N-index = lane_row
#
# LDS layout (D=576) — total 23168 bytes < 32 KB -> enables 2 CTAs/CU
# -----------------------------------------------------------------------
#   Region 0: scores_lds [N_WAVES=4, BLOCK_H=16, BLOCK_K+1=17] f32 = 4352 bytes
#             (cross-wave QK partial scores; aliased as p_lds [BLOCK_H, BLOCK_K+1])
#   Region 1: kv_lds     [BLOCK_K=16, D+KV_PAD=584]            bf16 = 18688 bytes
#   Region 2: slots_lds  [BLOCK_K=16]                          i32  =    64 bytes
#   Region 3: valids_lds [BLOCK_K=16]                          i32  =    64 bytes
#
# VGPR accumulator:
#   Each lane carries D_CHUNKS_W = (D//4)//16 acc chunks as ForOp iter-args.
#   acc_dc[k] = vec<4,f32>; acc_dc[k][v] = acc[head=lane_cgrp*4+v, d=d_wave_base+k*16+lane_row]
#   ForOp iter-args: [m_i×4, l_i×4, acc_dc0, ..., acc_dc{D_CHUNKS_W-1}] = 8 + D_CHUNKS_W values
#
# QK step: each wave computes a partial S[BLOCK_H, BLOCK_K] over its D-slice.
#   A-frag: Q[M=lane_row (head), K=d_wave_base + lane_cgrp*4..+3]    — from VRAM (L2-cached)
#   B-frag: KV[N=lane_row (kv_slot), K=d_wave_base + lane_cgrp*4..+3] — from VRAM
#   C-frag: c_frag[v] = partial S[head=lane_cgrp*4+v, kv_slot=lane_row] for this wave's D-slice
#   KV data also written to kv_lds[lane_row, col..+3] for the PV transpose step.
#
# Cross-wave reduce (QK scores):
#   Each wave writes its c_frag to scores_lds[wave_id, head, kv_slot]; barrier;
#   all waves then read and sum the 4 partials -> full score[head, kv_slot].
#
# Softmax: per-head max/exp across BLOCK_K KV columns.
#   Lane l holds scores for 4 heads (lane_cgrp*4..+3) vs 1 kv slot (lane_row).
#   Butterfly xor over lane_row dimension (xor 1,2,4,8) to get max/sum over all 16 kv.
#   Each lane carries 4 (m_i, l_i) pairs as ForOp iter-args.
#   Softmax runs redundantly on all 4 waves (identical results).
#
# LDS transpose (p-matrix):
#   Write p_lds[head=lane_cgrp*4+v, kv_slot=lane_row]; barrier;
#   read as A-frag: p_lds[head=lane_row, kv_slot=lane_cgrp*4..+3]
#   p_lds aliases the bottom BLOCK_H rows of scores_lds (wave_id==0 slice).
#
# PV step: acc[BLOCK_H, D] += p[BLOCK_H, BLOCK_K] x KV[BLOCK_K, D]
#   A-frag: p[M=lane_row (head), K=lane_cgrp*4..+3 (kv)]    — from p_lds after transpose
#   B-frag: KV[N=d_wave_base+lane_row (D_col), K=lane_cgrp*4..+3 (kv_slot)] — from kv_lds
#   C-frag: acc_dc[dc][v] = acc[head=lane_cgrp*4+v, d=d_wave_base+dc*16+lane_row]
#   Result stays in VGPR iter-args — no LDS write-back needed.
#
# Per-tile barrier sequence:
#   1. barrier after slot load      (lds_slots/lds_valids visible)
#   2. barrier after QK scores+KV   (scores_lds + kv_lds written by all waves)
#   3. barrier after p write        (p_lds visible for PV A-frag transpose read)

_MFMA_K = 16   # K-elements per MFMA call (gfx942 16x16x16 BF16)
_BLOCK_H = 16  # heads per CTA (matches MFMA M/N tile size)

def _get_mfma_bf16_k16():
    """Return the mfma_f32_16x16x16bf16_1k MLIR op, or None on non-gfx942."""
    fn = getattr(rocdl, "mfma_f32_16x16x16bf16_1k", None) or getattr(
        rocdl, "mfma_f32_16x16x16_bf16_1k", None
    )
    return fn

@functools.lru_cache(maxsize=256)
def _compile_pa_decode_sparse_mfma(
    *,
    H: int,          # actual number of heads (may be < _BLOCK_H)
    D: int,
    BLOCK_K: int,
    KV_SPLITS: int,
    softmax_scale: float,
):
    """JIT-compile the MFMA-based kernel: BLOCK_H=16 heads per CTA, 4 waves.

    Returns a flyc.jit launcher, cached on configuration.

    4-wave design (256 threads), VGPR accumulator:
    - 4 waves x 64 lanes = 256 threads per CTA.
    - Each wave handles D_PER_WAVE = D//4 D-columns for both QK and PV.
    - KV tile is stored to kv_lds during QK for transposed PV reads (restored).
    - Cross-wave QK partial scores are accumulated via scores_lds (a sub-region
      of p_lds): each wave writes its partial QK c_frag to scores_lds, all waves
      barrier, then all read and sum the 4 wave partials to get full scores.
    - Softmax runs redundantly on all 4 waves (same butterfly over lane_row dim
      after scores_lds reduce), so all waves write the same p_lds values.
    - PV uses p from p_lds (transposed) and KV from kv_lds.
    - Accumulator kept in VGPR ForOp iter-args (D_CHUNKS_W f32x4 values = 36 f32
      per lane), eliminating acc_lds entirely.

    LDS layout (23168 bytes < 32KB → enables 2 CTAs/CU):
    - scores_lds [N_WAVES, BLOCK_H, BLOCK_K+1] f32 = 4352 bytes
    - kv_lds     [BLOCK_K, D + KV_PAD] bf16 = 18688 bytes
    - slots_lds  [BLOCK_K] i32 = 64 bytes
    - valids_lds [BLOCK_K] i32 = 64 bytes
    - acc_lds ELIMINATED — accumulator in VGPR ForOp iter-args

    ForOp iter-args: [m_i x 4, l_i x 4, acc_dc0, acc_dc1, ..., acc_dc8] = 44 values
    where acc_dc[k] = vec<4,f32> carrying running accumulator for d-chunk k.
    """
    assert D % _MFMA_K == 0, f"D={D} must be divisible by MFMA_K={_MFMA_K}"
    assert D % 4 == 0, f"D={D} must be divisible by 4 waves"
    assert BLOCK_K == _BLOCK_H, (
        f"MFMA kernel requires BLOCK_K == BLOCK_H == {_BLOCK_H}; got BLOCK_K={BLOCK_K}"
    )

    mfma_fn = _get_mfma_bf16_k16()
    assert mfma_fn is not None, "mfma_f32_16x16x16bf16_1k not found in rocdl"

    fm = arith.FastMathFlags.fast
    N_WAVES       = 4
    D_PER_WAVE    = D // N_WAVES          # D-columns per wave (144 for D=576)
    D_CHUNKS_W    = D_PER_WAVE // _MFMA_K  # K-chunks per wave (9 for D=576)
    BLOCK_THREADS_MFMA = WAVE_SIZE * N_WAVES     # 256

    GPU_ARCH = get_rocm_arch()

    # LDS layout (23168 bytes < 32KB → enables 2 CTAs/CU):
    # scores_lds [N_WAVES, BLOCK_H, BLOCK_K+1] f32 = 4352 bytes (cross-wave QK + p-matrix)
    # kv_lds     [BLOCK_K, D + KV_PAD] bf16 = 18688 bytes (KV tile; QK B-frag + PV transpose)
    # slots_lds  [BLOCK_K] i32 = 64 bytes
    # valids_lds [BLOCK_K] i32 = 64 bytes
    # acc_lds    ELIMINATED — accumulator kept in VGPR ForOp iter-args (36 f32x4 per lane)
    #
    # VGPR accumulator: each lane carries D_CHUNKS_W=9 acc chunks as ForOp iter-args.
    # acc_dc[k] = vec<4,f32> = C-frag slice for d-chunk k.
    # acc_dc[k][v] = acc[head=lane_cgrp*4+v, d=d_wave_base+k*16+lane_row]
    # ForOp iter-args: [m_i×4, l_i×4, acc_dc0, acc_dc1, ..., acc_dc8] = 44 values

    KV_PAD  = 8
    KV_ROW_STRIDE  = D + KV_PAD
    ALIAS_ROW_STRIDE = BLOCK_K + 1

    SCORES_LDS_BYTES = N_WAVES * _BLOCK_H * ALIAS_ROW_STRIDE * 4
    KV_LDS_BYTES    = BLOCK_K * KV_ROW_STRIDE * 2

    assert SCORES_LDS_BYTES + KV_LDS_BYTES + 2 * BLOCK_K * 4 <= 65536, (
        f"LDS budget exceeded: {SCORES_LDS_BYTES+KV_LDS_BYTES+2*BLOCK_K*4} > 65536"
    )

    allocator = SmemAllocator(None, arch=GPU_ARCH,
        global_sym_name=f"pa_decode_sparse_mfma16x16x16_4w_v3_smem_D{D}_K{BLOCK_K}_S{KV_SPLITS}")
    lds_scores_off = allocator._align(allocator.ptr, 16)
    allocator.ptr  = lds_scores_off + SCORES_LDS_BYTES
    lds_kv_off     = allocator._align(allocator.ptr, 16)
    allocator.ptr  = lds_kv_off + KV_LDS_BYTES
    lds_slots_off  = allocator._align(allocator.ptr, 16)
    allocator.ptr  = lds_slots_off + BLOCK_K * 4
    lds_valids_off = allocator._align(allocator.ptr, 16)
    allocator.ptr  = lds_valids_off + BLOCK_K * 4
    lds_p_off = lds_scores_off

    # ---- Helpers ----
    def _fexp2(x):
        return mlir_llvm.call_intrinsic(T.f32, "llvm.amdgcn.exp2.f32", [x], [], [])

    def _mfma_bf16(a_v4bf16, b_v4bf16, acc_v4f32):
        """mfma_f32_16x16x16bf16_1k: A,B as v4i16 (bitcast from v4bf16)."""
        a_vi16 = vector.bitcast(T.vec(4, T.i16), a_v4bf16)
        b_vi16 = vector.bitcast(T.vec(4, T.i16), b_v4bf16)
        return mfma_fn(T.f32x4, [a_vi16, b_vi16, acc_v4f32, 0, 0, 0])

    _kname = f"pa_decode_sparse_mfma16x16x16_4w_v3_H{H}_D{D}_K{BLOCK_K}_S{KV_SPLITS}"

    @flyc.kernel(name=_kname, known_block_size=[BLOCK_THREADS_MFMA, 1, 1])
    def _kernel(
        q:           fx.Tensor,   # [T, H, D]            bf16
        unified_kv:  fx.Tensor,   # [total_pages, D]     bf16
        kv_indices:  fx.Tensor,   # [total_indices]      i32
        kv_indptr:   fx.Tensor,   # [T+1]                i32
        attn_sink:   fx.Tensor,   # [H]                  f32
        m_partial:   fx.Tensor,   # [T, KV_SPLITS, H]    f32
        l_partial:   fx.Tensor,   # [T, KV_SPLITS, H]    f32
        acc_partial: fx.Tensor,   # [T, KV_SPLITS, H, D] f32
        out:         fx.Tensor,   # [T, H, D]            bf16
    ):
        f32 = T.f32
        i32 = T.i32

        c_zero_f32  = arith.constant(0.0, type=f32)
        c_one_f32   = arith.constant(1.0, type=f32)
        c_neg_inf   = arith.constant(_NEG_INF, type=f32)
        c_eps       = arith.constant(1e-30, type=f32)
        c_log2e     = arith.constant(_LOG2E, type=f32)
        c_scale_l2  = arith.constant(softmax_scale * _LOG2E, type=f32)
        c0_i32  = arith.constant(0, type=i32)
        c1_i32  = arith.constant(1, type=i32)
        c2_i32  = arith.constant(2, type=i32)
        c4_i32  = arith.constant(4, type=i32)
        c64_i32 = arith.constant(64, type=i32)
        cD_i32  = arith.constant(D, type=i32)
        cH_i32  = arith.constant(H, type=i32)
        cHD_i32 = arith.constant(H * D, type=i32)
        cBH_i32 = arith.constant(_BLOCK_H, type=i32)
        cBK_i32 = arith.constant(BLOCK_K, type=i32)
        cKH_i32 = arith.constant(KV_SPLITS * H, type=i32)
        cDPW_i32 = arith.constant(D_PER_WAVE, type=i32)
        cKVRS_i32  = arith.constant(KV_ROW_STRIDE, type=i32)
        cARS_i32   = arith.constant(ALIAS_ROW_STRIDE, type=i32)

        # Block/thread indices
        t      = arith.index_cast(i32, _to_raw(fx.block_idx.x))
        bh_blk = arith.index_cast(i32, _to_raw(fx.block_idx.y))
        pid_k  = arith.index_cast(i32, _to_raw(fx.block_idx.z))
        tid    = arith.index_cast(i32, _to_raw(fx.thread_idx.x))  # 0..255

        # 4-wave decomposition
        wave_id  = arith.divsi(tid, c64_i32)   # 0..3
        lane     = arith.remsi(tid, c64_i32)   # 0..63
        lane_row = arith.remsi(lane, cBH_i32)  # 0..15 — MFMA M/N index (head/kv)
        lane_cgrp = arith.divsi(lane, cBH_i32) # 0..3  — column group within wave's D-slice

        # D-column base for this wave
        d_wave_base = arith.muli(wave_id, cDPW_i32)  # 0, D/4, D/2, 3*D/4

        h_base  = arith.muli(bh_blk, cBH_i32)
        h_lane  = arith.addi(h_base, lane_row)
        h_valid = arith.cmpi(CmpIPredicate.slt, h_lane, cH_i32)

        # ---- kv_indptr ----
        indptr_rsrc = buffer_ops.create_buffer_resource(kv_indptr, max_size=True)
        kv_start_v  = buffer_ops.buffer_load(indptr_rsrc, t, vec_width=1, dtype=i32)
        kv_end_v    = buffer_ops.buffer_load(
            indptr_rsrc, ArithValue(t) + c1_i32, vec_width=1, dtype=i32
        )
        kv_start = rocdl.readfirstlane(i32, kv_start_v)
        kv_end   = rocdl.readfirstlane(i32, kv_end_v)
        kv_len   = arith.subi(kv_end, kv_start)

        # ---- Tile range ----
        cKVSP_i32     = arith.constant(KV_SPLITS, type=i32)
        tiles_per_seg = arith.ceildivsi(kv_len, arith.muli(cBK_i32, cKVSP_i32))
        tile_start    = arith.muli(pid_k, tiles_per_seg)
        num_tiles     = arith.ceildivsi(kv_len, cBK_i32)
        tile_end_raw  = arith.muli(arith.addi(pid_k, c1_i32), tiles_per_seg)
        tile_end      = arith.minsi(tile_end_raw, num_tiles)

        has_work = arith.cmpi(CmpIPredicate.slt, tile_start, tile_end)
        _if_work = scf.IfOp(_to_raw(has_work), results_=[], has_else=False)
        with _if_then(_if_work):

            # ---- LDS regions ----
            lds_base = allocator.get_base()

            # scores_lds: [N_WAVES, BLOCK_H, ALIAS_ROW_STRIDE] fp32 — cross-wave partial QK scores
            # flat index: wave * BLOCK_H*ALIAS_ROW_STRIDE + head * ALIAS_ROW_STRIDE + kv
            lds_scores = STensor(SmemPtr(lds_base, lds_scores_off, T.f32, shape=(N_WAVES * _BLOCK_H * ALIAS_ROW_STRIDE,)), dtype=T.f32, shape=(N_WAVES * _BLOCK_H * ALIAS_ROW_STRIDE,))
            lds_p = STensor(SmemPtr(lds_base, lds_p_off, T.f32, shape=(_BLOCK_H * ALIAS_ROW_STRIDE,)), dtype=T.f32, shape=(_BLOCK_H * ALIAS_ROW_STRIDE,))
            lds_kv = STensor(SmemPtr(lds_base, lds_kv_off, T.bf16, shape=(BLOCK_K * KV_ROW_STRIDE,)), dtype=T.bf16, shape=(BLOCK_K * KV_ROW_STRIDE,))
            lds_slots = STensor(SmemPtr(lds_base, lds_slots_off, T.i32, shape=(BLOCK_K,)), dtype=T.i32, shape=(BLOCK_K,))
            lds_valids = STensor(SmemPtr(lds_base, lds_valids_off, T.i32, shape=(BLOCK_K,)), dtype=T.i32, shape=(BLOCK_K,))

            # ---- Init m_i[4], l_i[4] ----
            sink_rsrc_init = buffer_ops.create_buffer_resource(attn_sink, max_size=True) if KV_SPLITS == 1 else None
            m_inits = []
            l_inits = []
            for _v in range_constexpr(4):
                h_v = arith.addi(h_base, arith.addi(arith.muli(lane_cgrp, c4_i32), arith.constant(_v, type=i32)))
                h_v_valid = arith.cmpi(CmpIPredicate.slt, h_v, cH_i32)
                if KV_SPLITS == 1:
                    h_v_safe = arith.minsi(h_v, arith.subi(cH_i32, c1_i32))
                    sink_v = buffer_ops.buffer_load(sink_rsrc_init, h_v_safe, vec_width=1, dtype=f32)
                    sink_s = arith.select(h_v_valid, sink_v, c_neg_inf)
                    m_inits.append(arith.mulf(sink_s, c_log2e, fastmath=fm))
                    l_inits.append(c_one_f32)
                else:
                    m_inits.append(c_neg_inf)
                    l_inits.append(c_zero_f32)

            # ---- Accumulator VGPR init (zero vectors, one per D-chunk per wave) ----
            acc_inits = []
            for _dc in range_constexpr(D_CHUNKS_W):
                acc_inits.append(arith.constant_vector(0.0, T.f32x4))

            # ---- Buffer resources ----
            idx_rsrc = buffer_ops.create_buffer_resource(kv_indices, max_size=True)
            kv_rsrc  = buffer_ops.create_buffer_resource(unified_kv,  max_size=True)
            q_rsrc   = buffer_ops.create_buffer_resource(q, max_size=True)

            kv_end_m1 = arith.subi(arith.addi(kv_start, kv_len), c1_i32)

            # ---- KV tile loop ----
            for tile_idx, loop_state in range(
                _to_raw(tile_start), _to_raw(tile_end), 1,
                init=m_inits + l_inits + acc_inits,
            ):
                m_i   = [loop_state[v]     for v in range_constexpr(4)]
                l_i   = [loop_state[4 + v] for v in range_constexpr(4)]
                acc_i = [loop_state[8 + dc] for dc in range_constexpr(D_CHUNKS_W)]
                # acc_i[dc] is a vec<4,f32> carrying the running accumulator for d-chunk dc

                tile_i32    = arith.index_cast(i32, _to_raw(tile_idx))
                k_tile_base = arith.muli(tile_i32, cBK_i32)

                # -- Step 1: load BLOCK_K slot indices into lds_slots/lds_valids --
                # lane_cgrp==0 AND wave_id==0 lanes handle lane_row=0..15 (one slot each)
                is_cgrp0  = arith.cmpi(CmpIPredicate.eq, lane_cgrp, c0_i32)
                is_wave0  = arith.cmpi(CmpIPredicate.eq, wave_id, c0_i32)
                is_loader = arith.andi(is_cgrp0, is_wave0)
                _if_slot = scf.IfOp(_to_raw(is_loader), results_=[], has_else=False)
                with _if_then(_if_slot):
                    raw_pos   = arith.addi(kv_start, arith.addi(k_tile_base, lane_row))
                    in_rng    = arith.cmpi(CmpIPredicate.slt, raw_pos, arith.addi(kv_start, kv_len))
                    pos       = arith.minsi(raw_pos, kv_end_m1)
                    slot_v    = buffer_ops.buffer_load(idx_rsrc, pos, vec_width=1, dtype=i32)
                    slot      = arith.select(in_rng, slot_v, arith.constant(-1, type=i32))
                    valid_i32 = arith.select(in_rng, c1_i32, c0_i32)
                    lds_slots[fx.Index(lane_row)]  = slot
                    lds_valids[fx.Index(lane_row)] = valid_i32

                gpu.barrier()   # slots visible to all lanes

                # -- Step 2: QK MFMA (per wave, over this wave's D-slice) --
                # Each wave processes D_CHUNKS_W D-chunks.
                # kv slot for QK: lane_row selects the KV row (same across all waves).
                slot_j_qk    = lds_slots[fx.Index(lane_row)]
                safe_slot_qk = arith.maxsi(slot_j_qk, c0_i32)
                valid_kv_slot = arith.cmpi(CmpIPredicate.ne, lds_valids[fx.Index(lane_row)], c0_i32)

                c_zero_bf16  = arith.constant(0.0, type=T.bf16)
                c_zero_v4f32 = arith.constant_vector(0.0, T.f32x4)
                c_frag = c_zero_v4f32  # partial QK score for this wave's D-slice

                # --- Software pipelining: issue all VMEM loads before any MFMA ---
                # Since range_constexpr fully unrolls the loop at IR build time, we split
                # the QK loop into two phases:
                #   Phase A: issue all D_CHUNKS_W buffer_loads for Q and KV (VMEM in-flight)
                #   Phase B: write kv_lds, then issue MFMAs (GPU overlaps VMEM↔MFMA)
                # The GPU's VMEM instruction queue holds up to 16 outstanding requests;
                # all D_CHUNKS_W=9 Q and KV loads fit simultaneously, hiding VMEM latency 
                # behind the MFMA pipeline.

                qk_a_frags = []   # pre-loaded Q A-frags, one per D-chunk
                qk_kv_v4s  = []   # pre-loaded KV raw vectors, one per D-chunk
                qk_col_offs = []  # column offsets, one per D-chunk

                # Phase A: issue all loads
                for dc in range_constexpr(D_CHUNKS_W):
                    d_base_const = dc * _MFMA_K   # local offset within wave's D-slice
                    col_off = arith.addi(
                        d_wave_base,
                        arith.addi(arith.constant(d_base_const, type=i32),
                                   arith.muli(lane_cgrp, c4_i32)),
                    )
                    qk_col_offs.append(col_off)

                    # Q A-frag: Q[h_lane, col_off..+3]
                    flat_q = arith.addi(
                        arith.muli(t, cHD_i32),
                        arith.addi(arith.muli(h_lane, cD_i32), col_off),
                    )
                    dw_q  = arith.divsi(flat_q, c2_i32)
                    raw_q = buffer_ops.buffer_load(q_rsrc, dw_q, vec_width=2, dtype=i32)
                    q_v4  = vector.bitcast(T.vec(4, T.bf16), raw_q)
                    q_vals = []
                    for v in range_constexpr(4):
                        elem = vector.extract(q_v4, static_position=[v], dynamic_position=[])
                        q_vals.append(arith.select(h_valid, elem, c_zero_bf16))
                    qk_a_frags.append(vector.from_elements(T.vec(4, T.bf16), q_vals))

                    # KV B-frag for QK: KV[safe_slot_qk, col_off..+3]
                    flat_kv = arith.addi(arith.muli(safe_slot_qk, cD_i32), col_off)
                    dw_kv   = arith.divsi(flat_kv, c2_i32)
                    raw_kv  = buffer_ops.buffer_load(kv_rsrc, dw_kv, vec_width=2, dtype=i32)
                    qk_kv_v4s.append(vector.bitcast(T.vec(4, T.bf16), raw_kv))

                # Phase B: write kv_lds + issue MFMAs (VMEM latency already in-flight)
                for dc in range_constexpr(D_CHUNKS_W):
                    col_off = qk_col_offs[dc]
                    kv_v4   = qk_kv_v4s[dc]
                    a_frag  = qk_a_frags[dc]

                    c_frag = _mfma_bf16(a_frag, kv_v4, c_frag)

                    # Write KV to kv_lds[lane_row, col_off..+3] for PV transpose
                    for v in range_constexpr(4):
                        kv_elem = vector.extract(kv_v4, static_position=[v], dynamic_position=[])
                        kv_flat = arith.addi(arith.muli(lane_row, cKVRS_i32),
                                             arith.addi(col_off, arith.constant(v, type=i32)))
                        lds_kv[fx.Index(kv_flat)] = kv_elem

                # -- Step 3: Write partial QK scores to scores_lds --
                # c_frag[v] = partial S[head=lane_cgrp*4+v, kv_slot=lane_row] over D/4 columns.
                # scores_lds flat: wave*BLOCK_H*ALIAS_ROW_STRIDE + head*ALIAS_ROW_STRIDE + kv
                for v in range_constexpr(4):
                    head_idx  = arith.addi(arith.muli(lane_cgrp, c4_i32), arith.constant(v, type=i32))
                    scores_flat = arith.addi(
                        arith.muli(wave_id, arith.constant(_BLOCK_H * ALIAS_ROW_STRIDE, type=i32)),
                        arith.addi(arith.muli(head_idx, cARS_i32), lane_row),
                    )
                    s_v = vector.extract(c_frag, static_position=[v], dynamic_position=[])
                    lds_scores[fx.Index(scores_flat)] = s_v

                gpu.barrier()   # scores_lds + kv_lds written by all waves

                # -- Step 4: Cross-wave reduce + softmax (all waves, same result) --
                # Each lane reads the 4 wave partials for its (head, kv_slot) combination and sums.
                # scores_lds[w, head, kv_slot] for w=0..3 → full score[head, kv_slot]
                # Then butterfly max/sum over lane_row dim (xor 1,2,4,8) as in current 1-wave.

                c_frag_scaled = []
                for v in range_constexpr(4):
                    head_idx = arith.addi(arith.muli(lane_cgrp, c4_i32), arith.constant(v, type=i32))
                    # Sum 4 wave partials for this (head, kv_slot=lane_row)
                    partial_sum = c_zero_f32
                    for w in range_constexpr(N_WAVES):
                        alias_flat_w = arith.addi(
                            arith.constant(w * _BLOCK_H * ALIAS_ROW_STRIDE, type=i32),
                            arith.addi(arith.muli(head_idx, cARS_i32), lane_row),
                        )
                        partial_sum = arith.addf(partial_sum, lds_scores[fx.Index(alias_flat_w)], fastmath=fm)
                    # Scale and gate by validity
                    partial_scaled = arith.mulf(partial_sum, c_scale_l2, fastmath=fm)
                    c_frag_scaled.append(arith.select(valid_kv_slot, partial_scaled, c_neg_inf))

                # Online softmax (same as 1-wave: butterfly over lane_row / kv dimension)
                m_new     = []
                alpha     = []
                p_vals_v4 = []
                sum_p_v   = []

                for v in range_constexpr(4):
                    s_v = c_frag_scaled[v]

                    # Butterfly max over lane_row (kv) dim within this wave's 64 lanes
                    s_max_v = s_v
                    for xor_off in [1, 2, 4, 8]:
                        peer    = _to_raw(ArithValue(s_max_v).shuffle_xor(xor_off, BLOCK_THREADS_MFMA))
                        s_max_v = arith.maximumf(s_max_v, peer)

                    m_new_v      = arith.maximumf(m_i[v], s_max_v)
                    is_all_inf_v = arith.cmpf(CmpFPredicate.OEQ, m_new_v, c_neg_inf)
                    raw_alpha_v  = _fexp2(arith.subf(m_i[v], m_new_v))
                    alpha_v      = arith.select(is_all_inf_v, c_one_f32, raw_alpha_v)

                    pj_v      = _fexp2(arith.subf(c_frag_scaled[v], m_new_v))
                    pj_safe_v = arith.select(valid_kv_slot, pj_v, c_zero_f32)

                    # Butterfly sum_p over lane_row dim
                    sum_p_vv = pj_safe_v
                    for xor_off in [1, 2, 4, 8]:
                        peer     = _to_raw(ArithValue(sum_p_vv).shuffle_xor(xor_off, BLOCK_THREADS_MFMA))
                        sum_p_vv = arith.addf(sum_p_vv, peer, fastmath=fm)

                    m_new.append(m_new_v)
                    alpha.append(alpha_v)
                    p_vals_v4.append(pj_safe_v)
                    sum_p_v.append(sum_p_vv)

                l_new = []
                for v in range_constexpr(4):
                    l_new.append(arith.addf(
                        arith.mulf(l_i[v], alpha[v], fastmath=fm), sum_p_v[v], fastmath=fm
                    ))

                # -- Step 5: Write p to p_lds (= scores_lds base) for LDS transpose --
                # All 4 waves compute the same p values (same reduce, same butterfly).
                # All waves write p_lds[head*ALIAS_ROW_STRIDE+kv]; redundant writes are harmless.
                for v in range_constexpr(4):
                    p_head = arith.addi(arith.muli(lane_cgrp, c4_i32), arith.constant(v, type=i32))
                    p_flat = arith.addi(
                        arith.muli(p_head, cARS_i32), lane_row
                    )
                    lds_p[fx.Index(p_flat)] = p_vals_v4[v]

                gpu.barrier()   # p_lds visible for PV A-frag transposed read

                # -- Step 6: PV MFMA (per wave, using kv_lds for transposed B-frag) --
                # A-frag: p[M=lane_row (head), K=lane_cgrp*4..+3 (kv)] from p_lds (transposed)
                # B-frag: KV[kv_slot=lane_cgrp*4+v, d_col] from kv_lds (transposed read)
                # C-frag: acc_i[dc] = vec<4,f32> VGPR iter-arg (no LDS round-trip)
                #
                # Software pipelining for PV: pre-load all A-frags and B-frags from LDS
                # before issuing any MFMAs. LDS reads have have small latency - issuing them
                # all upfront lets early MFMAs run while later LDS reads complete.
                pv_a_frags = []   # pre-loaded P A-frags
                pv_b_frags = []   # pre-loaded KV B-frags
                pv_d_cols  = []   # d_col per D-chunk

                # The P A-frag is the same for all D-chunks (depends only on lane_row/lane_cgrp).
                # Pre-load it once and reuse across all D-chunks.
                a_vals_p_common = []
                for v in range_constexpr(4):
                    kv_k     = arith.addi(arith.muli(lane_cgrp, c4_i32), arith.constant(v, type=i32))
                    p_flat_r = arith.addi(arith.muli(lane_row, cARS_i32), kv_k)
                    a_vals_p_common.append(arith.truncf(T.bf16, lds_p[fx.Index(p_flat_r)]))
                a_frag_pv_common = vector.from_elements(T.vec(4, T.bf16), a_vals_p_common)

                # Phase A: issue all D-chunk LDS reads for KV B-frags
                for dc in range_constexpr(D_CHUNKS_W):
                    d_base_i32 = arith.addi(d_wave_base, arith.constant(dc * _MFMA_K, type=i32))
                    d_col_i32  = arith.addi(d_base_i32, lane_row)
                    pv_d_cols.append(d_col_i32)

                    b_vals_pv = []
                    for v in range_constexpr(4):
                        kv_slot_pv = arith.addi(arith.muli(lane_cgrp, c4_i32), arith.constant(v, type=i32))
                        kv_flat_pv = arith.addi(arith.muli(kv_slot_pv, cKVRS_i32), d_col_i32)
                        b_vals_pv.append(lds_kv[fx.Index(kv_flat_pv)])
                    pv_b_frags.append(vector.from_elements(T.vec(4, T.bf16), b_vals_pv))

                # Phase B: issue MFMAs (LDS reads from Phase A already in-flight / complete)
                new_acc = []
                for dc in range_constexpr(D_CHUNKS_W):
                    # Rescale acc by per-head alpha (online softmax correction)
                    acc_frag = acc_i[dc]   # vec<4,f32> from ForOp iter-arg
                    acc_scaled = []
                    for v in range_constexpr(4):
                        val = vector.extract(acc_frag, static_position=[v], dynamic_position=[])
                        acc_scaled.append(arith.mulf(val, alpha[v], fastmath=fm))
                    acc_frag_scaled = vector.from_elements(T.f32x4, acc_scaled)

                    new_acc.append(_mfma_bf16(a_frag_pv_common, pv_b_frags[dc], acc_frag_scaled))

                # acc stays in VGPRs — no barrier needed here.
                # The next tile's Step 3 barrier covers scores_lds reuse.

                loop_state = yield m_new + l_new + new_acc

            m_final     = [loop_state[v]      for v in range_constexpr(4)]
            l_final     = [loop_state[4 + v]  for v in range_constexpr(4)]
            acc_final   = [loop_state[8 + dc] for dc in range_constexpr(D_CHUNKS_W)]

            # ---- Output ----
            if KV_SPLITS == 1:
                out_rsrc = buffer_ops.create_buffer_resource(out, max_size=True)
                for v in range_constexpr(4):
                    h_v       = arith.addi(h_base, arith.addi(arith.muli(lane_cgrp, c4_i32), arith.constant(v, type=i32)))
                    h_v_valid = arith.cmpi(CmpIPredicate.slt, h_v, cH_i32)
                    l_safe_v  = arith.maximumf(l_final[v], c_eps)
                    for dc in range_constexpr(D_CHUNKS_W):
                        d_base_i32 = arith.addi(d_wave_base, arith.constant(dc * _MFMA_K, type=i32))
                        d_col_i32  = arith.addi(d_base_i32, lane_row)
                        acc_val    = vector.extract(acc_final[dc], static_position=[v], dynamic_position=[])
                        out_f32    = arith.divf(acc_val, l_safe_v)
                        out_bf16   = arith.truncf(T.bf16, out_f32)
                        out_flat   = arith.addi(
                            arith.muli(t, cHD_i32),
                            arith.addi(arith.muli(h_v, cD_i32), d_col_i32),
                        )
                        _if_h = scf.IfOp(_to_raw(h_v_valid), results_=[], has_else=False)
                        with _if_then(_if_h):
                            buffer_ops.buffer_store(out_bf16, out_rsrc, out_flat)
            else:
                ap_rsrc = buffer_ops.create_buffer_resource(acc_partial, max_size=True)
                mp_rsrc = buffer_ops.create_buffer_resource(m_partial, max_size=True)
                lp_rsrc = buffer_ops.create_buffer_resource(l_partial, max_size=True)
                for v in range_constexpr(4):
                    h_v       = arith.addi(h_base, arith.addi(arith.muli(lane_cgrp, c4_i32), arith.constant(v, type=i32)))
                    h_v_valid = arith.cmpi(CmpIPredicate.slt, h_v, cH_i32)
                    _if_hv    = scf.IfOp(_to_raw(h_v_valid), results_=[], has_else=False)
                    with _if_then(_if_hv):
                        ml_flat = arith.addi(
                            arith.muli(t, cKH_i32),
                            arith.addi(arith.muli(pid_k, cH_i32), h_v),
                        )
                        is_row0 = arith.cmpi(CmpIPredicate.eq, lane_row, c0_i32)
                        _if_r0  = scf.IfOp(_to_raw(is_row0), results_=[], has_else=False)
                        with _if_then(_if_r0):
                            buffer_ops.buffer_store(m_final[v], mp_rsrc, ml_flat)
                            buffer_ops.buffer_store(l_final[v], lp_rsrc, ml_flat)

                        for dc in range_constexpr(D_CHUNKS_W):
                            d_base_i32 = arith.addi(d_wave_base, arith.constant(dc * _MFMA_K, type=i32))
                            d_col_i32  = arith.addi(d_base_i32, lane_row)
                            acc_val    = vector.extract(acc_final[dc], static_position=[v], dynamic_position=[])
                            ap_flat    = arith.addi(arith.muli(ml_flat, cD_i32), d_col_i32)
                            buffer_ops.buffer_store(acc_val, ap_rsrc, ap_flat)

    @flyc.jit
    def _launcher(
        q, unified_kv, kv_indices, kv_indptr, attn_sink,
        m_partial, l_partial, acc_partial, out,
        T_size: Int32,
        stream: FlyStream = fx.Stream(None),
    ):
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()

        n_head_blocks = (H + _BLOCK_H - 1) // _BLOCK_H
        _kernel(
            q, unified_kv, kv_indices, kv_indptr, attn_sink,
            m_partial, l_partial, acc_partial, out,
        ).launch(
            grid=(T_size, n_head_blocks, KV_SPLITS),
            block=(BLOCK_THREADS_MFMA, 1, 1),
            stream=stream,
        )

    return _launcher


# ---------------------------------------------------------------------------
# Host wrapper
# ---------------------------------------------------------------------------

# Import the Triton reduce kernel for the KV_SPLITS>1 combine step.
# Imported lazily to avoid a circular dependency at module load time.
def _get_triton_reduce():
    from aiter.ops.triton._triton_kernels.attention.pa_decode_sparse import (
        _pa_decode_sparse_reduce,
    )
    return _pa_decode_sparse_reduce


def flydsl_pa_decode_sparse(
    q,
    unified_kv,
    kv_indices,
    kv_indptr,
    attn_sink,
    softmax_scale,
    *,
    kv_scales=None,
    block_h=None,    # unused (BLOCK_H=1 is fixed); kept for API parity
    kv_splits=None,
    has_invalid=False,   # sentinel support not yet implemented
    skip_reduce=False,
):
    """FlyDSL sparse paged-decode attention.

    Drop-in replacement for the Triton pa_decode_sparse wrapper. The FlyDSL
    main kernel runs the QK attention + online softmax; for KV_SPLITS>1 the
    existing Triton _pa_decode_sparse_reduce kernel is called to combine splits.

    Args
    ----
    q             : [T, H, D] bfloat16
    unified_kv    : [total_pages, D] bfloat16
    kv_indices    : [total_indices] int32 — flat scatter indices into unified_kv
    kv_indptr     : [T+1] int32 — CSR row pointers (kv_indptr[t]:kv_indptr[t+1]
                    gives token t's KV slice)
    attn_sink     : [H] float32 — per-head attention sink log-weight
    softmax_scale : float — 1/sqrt(D) scaling for QK dot products
    kv_scales     : not used (FP8 path not yet implemented)
    kv_splits     : int or None — number of K-splits (None = auto)
    has_invalid   : bool — whether kv_indices contains -1 sentinel values
    skip_reduce   : bool — if True return (acc_partial, m_partial, l_partial)
                    without running the reduce kernel (for testing)

    Returns
    -------
    Tensor [T, H, D] bfloat16, or tuple when skip_reduce=True.
    """
    if kv_scales is not None:
        raise NotImplementedError("FP8 KV cache not yet implemented in FlyDSL kernel")
    if has_invalid:
        raise NotImplementedError("has_invalid=True not yet implemented in FlyDSL kernel")

    T_val, H, D = q.shape
    device = q.device

    # ---- block_k selection ----
    # MFMA kernel: BLOCK_H=16, BLOCK_K=16. Requires H divisible by 16 and D divisible by 16.
    # Falls back to scalar kernel otherwise.
    _use_mfma = (
        D % _MFMA_K == 0
        and _get_mfma_bf16_k16() is not None
        and H % _BLOCK_H == 0
    )
    block_k = _BLOCK_H if _use_mfma else (16 if D >= 256 else 32)

    # ---- kv_splits auto-selection ----
    # Target: keep total CTAs near max_num_wg for good GPU utilization.
    # MFMA uses n_head_blocks = ceil(H/16) CTAs per token, vs H for scalar.
    # This reduces parallelism 16× → compensate with more kv_splits.
    max_num_wg = 256
    if kv_splits is None:
        max_kv_len    = kv_indices.shape[0]
        max_kv_splits = max(1, triton.cdiv(max_kv_len, block_k))
        n_head_blocks = (H + _BLOCK_H - 1) // _BLOCK_H if _use_mfma else H
        kv_splits     = max(1, max_num_wg // max(1, T_val * n_head_blocks))
        if _use_mfma:
            # MFMA groups BLOCK_H=16 heads per CTA, reducing head parallelism 16×.
            # Compensate with more kv_splits, but cap at 2 to avoid reduce overhead:
            # empirically kv_splits=2 is near-optimal for large T×H workloads.
            kv_splits = max(kv_splits, 2)
        kv_splits     = min(max_kv_splits, kv_splits)
        kv_splits     = triton.next_power_of_2(kv_splits)

    assert D % WAVE_SIZE == 0, (
        f"D={D} must be divisible by {WAVE_SIZE} (threads per wave); "
        f"got D % {WAVE_SIZE} = {D % WAVE_SIZE}"
    )

    out = torch.zeros((T_val, H, D), dtype=torch.bfloat16, device=device)

    if kv_splits == 1:
        m_partial = l_partial = acc_partial = out  # dummy (not written by kernel)
    else:
        m_partial   = torch.empty((T_val, kv_splits, H), dtype=torch.float32, device=device)
        l_partial   = torch.empty_like(m_partial)
        acc_partial = torch.empty((T_val, kv_splits, H, D), dtype=torch.float32, device=device)

    # ---- Compile (or retrieve cached) kernel ----
    if _use_mfma:
        launcher = _compile_pa_decode_sparse_mfma(
            H=H,
            D=D,
            BLOCK_K=block_k,
            KV_SPLITS=kv_splits,
            softmax_scale=float(softmax_scale),
        )
    else:
        launcher = _compile_pa_decode_sparse(
            H=H,
            D=D,
            BLOCK_K=block_k,
            KV_SPLITS=kv_splits,
            softmax_scale=float(softmax_scale),
        )

    # ---- Launch FlyDSL kernel ----
    # T_size is annotated as fx.Int32 in the launcher, which accepts Python ints.
    # stream must be passed explicitly so that flyc.compile's CallState (built with
    # apply_defaults including stream) and subsequent cf(*args) calls agree on arg count.
    from .tensor_shim import _run_compiled
    fly_stream = fx.Stream(torch.cuda.current_stream())
    _run_compiled(
        launcher,
        q, unified_kv, kv_indices, kv_indptr, attn_sink,
        m_partial, l_partial, acc_partial, out,
        T_val,
        fly_stream,
    )

    if kv_splits == 1:
        return out

    if skip_reduce:
        return acc_partial, m_partial, l_partial

    # ---- Triton reduce kernel: combine KV_SPLITS partial states ----
    block_d      = triton.next_power_of_2(D)
    block_h_red  = 1
    grid_reduce  = (T_val, triton.cdiv(H, block_h_red))

    _pa_decode_sparse_reduce = _get_triton_reduce()
    _pa_decode_sparse_reduce[grid_reduce](
        m_partial,
        l_partial,
        acc_partial,
        attn_sink,
        kv_indptr,
        out,
        m_partial.stride(0),
        m_partial.stride(1),
        m_partial.stride(2),
        l_partial.stride(0),
        l_partial.stride(1),
        l_partial.stride(2),
        acc_partial.stride(0),
        acc_partial.stride(1),
        acc_partial.stride(2),
        acc_partial.stride(3),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        H,
        D,
        kv_splits,
        BLOCK_H=block_h_red,
        BLOCK_D=block_d,
        BLOCK_K=block_k,
        USE_EXP2=True,   # FlyDSL kernel works in base-2 domain, same as Triton main
        num_warps=4,
    )
    return out
