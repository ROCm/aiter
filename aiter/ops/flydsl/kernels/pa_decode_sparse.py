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
from dataclasses import dataclass
from typing import Tuple

import torch

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
# MfmaAtom: compile-time MFMA shape descriptor
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MfmaAtom:
    name: str                              # e.g. "16x16x16", "32x32x8"
    MFMA_M: int                            # output tile M rows (= BLOCK_H per CTA)
    MFMA_N: int                            # output tile N cols (= BLOCK_K per CTA)
    MFMA_K: int                            # K-reduction elements per MFMA call
    ACC_ELEMS: int                         # accumulator f32 elements per lane (4 or 16)
    fn_name: str                           # rocdl attribute name
    shuffle_offsets: Tuple[int, ...]       # XOR offsets for softmax butterfly
    acc_head_static_offsets: Tuple[int, ...] # head index within M-tile per acc elem
    acc_head_group_stride: int             # lane_cgrp multiplier for head idx

MFMA_16x16x16 = MfmaAtom(
    name="16x16x16",
    MFMA_M=16, MFMA_N=16, MFMA_K=16,
    ACC_ELEMS=4,
    fn_name="mfma_f32_16x16x16bf16_1k",
    shuffle_offsets=(1, 2, 4, 8),
    acc_head_static_offsets=(0, 1, 2, 3),
    acc_head_group_stride=4,
)

MFMA_32x32x8 = MfmaAtom(
    name="32x32x8",
    MFMA_M=32, MFMA_N=32, MFMA_K=8,
    ACC_ELEMS=16,
    fn_name="mfma_f32_32x32x8bf16_1k",
    shuffle_offsets=(1, 2, 4, 8, 16),
    acc_head_static_offsets=(0,1,2,3, 8,9,10,11, 16,17,18,19, 24,25,26,27),
    acc_head_group_stride=4,
)

_MFMA_MAP: dict[str, MfmaAtom] = {
    "16x16x16": MFMA_16x16x16,
    "32x32x8":  MFMA_32x32x8,
}

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
# MFMA-based kernel (BLOCK_H=16, 4-wave 256-thread/6-wave 384-thread design)
# ---------------------------------------------------------------------------
#
# Architecture summary
# --------------------
# Grid : (T, ceil(H/BLOCK_H), KV_SPLITS)   BLOCK_H=16 heads per CTA
# Block: 256 threads (4 waves x 64 lanes) or 384 threads (6 waves x 64 lanes) depending on MFMA shape
# MFMA shapes: 16x16x16 or 32x32x8
#
# MFMA 16x16x16: mfma_f32_16x16x16bf16_1k — 16x16x16 BF16 accumulate into 4xf32 per lane
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

@functools.lru_cache(maxsize=256)
def _compile_pa_decode_sparse_mfma(
    *,
    H: int,          # actual number of heads
    D: int,
    BLOCK_K: int,
    KV_SPLITS: int,
    softmax_scale: float,
    quant_kv: bool = False,
    mfma: MfmaAtom = MFMA_16x16x16,
    n_waves: int = 4,
):
    """JIT-compile the MFMA-based sparse paged-decode attention kernel.

    Parameterized over MfmaAtom shape and number of wavefronts per CTA.

    Supported configurations:
    - 16x16x16 / 4-wave: BLOCK_H=16, 256 threads, ~23KB LDS → 2 CTAs/CU
    - 32x32x8  / 6-wave: BLOCK_H=32, 384 threads, ~63KB LDS → 1 CTA/CU

    Supports both bf16 KV (quant_kv=False) and fp8 KV with per-group scales
    (quant_kv=True). The two paths differ only in Step 2 (KV VRAM load).

    Returns a flyc.jit launcher, cached on (mfma, n_waves, quant_kv, ...).
    """
    _BLOCK_H  = mfma.MFMA_M
    _MFMA_K   = mfma.MFMA_K
    _MFMA_N   = mfma.MFMA_N   # N-dimension of MFMA (= number of D-cols per c_frag)
    ACC_ELEMS = mfma.ACC_ELEMS
    N_WAVES   = n_waves
    BLOCK_THREADS_MFMA = WAVE_SIZE * N_WAVES
    D_PER_WAVE = D // N_WAVES
    # D_CHUNKS_W: number of acc D-chunks per wave. Each acc chunk covers MFMA_N D-columns.
    # For QK: MFMA_N/MFMA_K inner calls per D-chunk reduce over that chunk's D-columns.
    # For PV: 1 MFMA call per D-chunk covers MFMA_N D-columns (lane_row = 0..MFMA_N-1).
    D_CHUNKS_W  = D_PER_WAVE // _MFMA_N
    # Number of QK MFMA calls per acc D-chunk (= MFMA_N // MFMA_K, always integer)
    QK_CALLS_PW = _MFMA_N // _MFMA_K

    assert D % N_WAVES == 0, f"D={D} must be divisible by N_WAVES={N_WAVES}"
    assert D_PER_WAVE % _MFMA_N == 0, f"D_PER_WAVE={D_PER_WAVE} must be divisible by MFMA_N={_MFMA_N}"
    assert _MFMA_N % _MFMA_K == 0, f"MFMA_N={_MFMA_N} must be divisible by MFMA_K={_MFMA_K}"
    assert BLOCK_K == _BLOCK_H, (
        f"MFMA kernel requires BLOCK_K == BLOCK_H == {_BLOCK_H}; got BLOCK_K={BLOCK_K}"
    )
    if quant_kv:
        assert D % 64 == 0, f"D={D} must be divisible by GROUP_SIZE=64 for fp8 quantization"

    mfma_fn = getattr(rocdl, mfma.fn_name, None)
    assert mfma_fn is not None, f"{mfma.fn_name} not found in rocdl"

    fm = arith.FastMathFlags.fast

    # FP8 quantization parameters (only relevant when quant_kv=True)
    GROUP_SIZE = 64
    N_GROUPS   = D // GROUP_SIZE   # 9 for D=576

    GPU_ARCH = get_rocm_arch()

    KV_PAD           = 8
    KV_ROW_STRIDE    = D + KV_PAD
    ALIAS_ROW_STRIDE = BLOCK_K + 1

    SCORES_LDS_BYTES = N_WAVES * _BLOCK_H * ALIAS_ROW_STRIDE * 4
    KV_LDS_BYTES     = BLOCK_K * KV_ROW_STRIDE * 2

    _kv_tag = "fp8" if quant_kv else "bf16"
    allocator = SmemAllocator(None, arch=GPU_ARCH,
        global_sym_name=f"pa_decode_sparse_{mfma.name}_{N_WAVES}w_{_kv_tag}_smem_D{D}_K{BLOCK_K}_S{KV_SPLITS}")
    lds_scores_off = allocator._align(allocator.ptr, 16)
    allocator.ptr  = lds_scores_off + SCORES_LDS_BYTES
    lds_kv_off     = allocator._align(allocator.ptr, 16)
    allocator.ptr  = lds_kv_off + KV_LDS_BYTES
    lds_slots_off  = allocator._align(allocator.ptr, 16)
    allocator.ptr  = lds_slots_off + BLOCK_K * 4
    lds_valids_off = allocator._align(allocator.ptr, 16)
    allocator.ptr  = lds_valids_off + BLOCK_K * 4
    # p_lds aliases the base of scores_lds (first wave's region).
    # Safe because all waves finish reading scores_lds before any wave writes p_lds
    # (enforced by the barrier inserted between the read and write phases below).
    lds_p_off      = lds_scores_off

    # ---- Helpers (defined inside kernel context to avoid MLIR type construction
    #      before the context is established) ----
    def _fexp2(x):
        return mlir_llvm.call_intrinsic(T.f32, "llvm.amdgcn.exp2.f32", [x], [], [])

    def _acc_vec_type():
        """vec<ACC_ELEMS, f32> — called inside kernel body where MLIR context is active."""
        return T.vec(ACC_ELEMS, T.f32)

    def _kfrag_vec_type():
        """vec<4, bf16> — the MFMA K-fragment is always 4 bf16 elements per lane group."""
        return T.vec(4, T.bf16)

    def _mfma_bf16(a_vbf16, b_vbf16, acc_vf32):
        """Call the MFMA instruction. A,B as v4i16 (bitcast from v4bf16); acc as vec<ACC_ELEMS,f32>."""
        a_vi16 = vector.bitcast(T.vec(4, T.i16), a_vbf16)
        b_vi16 = vector.bitcast(T.vec(4, T.i16), b_vbf16)
        return mfma_fn(_acc_vec_type(), [a_vi16, b_vi16, acc_vf32, 0, 0, 0])

    _kname = f"pa_decode_sparse_{mfma.name}_{N_WAVES}w_{_kv_tag}_H{H}_D{D}_K{BLOCK_K}_S{KV_SPLITS}"

    @flyc.kernel(name=_kname, known_block_size=[BLOCK_THREADS_MFMA, 1, 1])
    def _kernel(
        q:           fx.Tensor,   # [T, H, D]                bf16
        unified_kv:  fx.Tensor,   # [total_pages, D]         bf16 or fp8
        kv_scales:   fx.Tensor,   # [total_pages, N_GROUPS]  f32 (fp8 path only; dummy for bf16)
        kv_indices:  fx.Tensor,   # [total_indices]          i32
        kv_indptr:   fx.Tensor,   # [T+1]                    i32
        attn_sink:   fx.Tensor,   # [H]                      f32
        m_partial:   fx.Tensor,   # [T, KV_SPLITS, H]        f32
        l_partial:   fx.Tensor,   # [T, KV_SPLITS, H]        f32
        acc_partial: fx.Tensor,   # [T, KV_SPLITS, H, D]     f32
        out:         fx.Tensor,   # [T, H, D]                bf16
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
        cDPW_i32  = arith.constant(D_PER_WAVE, type=i32)
        cKVRS_i32 = arith.constant(KV_ROW_STRIDE, type=i32)
        cARS_i32  = arith.constant(ALIAS_ROW_STRIDE, type=i32)
        cNG_i32   = arith.constant(N_GROUPS,   type=i32)   # fp8 path; harmless for bf16
        cGS_i32   = arith.constant(GROUP_SIZE, type=i32)   # fp8 path; harmless for bf16

        # Block/thread indices
        t      = arith.index_cast(i32, _to_raw(fx.block_idx.x))
        bh_blk = arith.index_cast(i32, _to_raw(fx.block_idx.y))
        pid_k  = arith.index_cast(i32, _to_raw(fx.block_idx.z))
        tid    = arith.index_cast(i32, _to_raw(fx.thread_idx.x))  # 0..255

        # 4-wave decomposition
        wave_id   = arith.divsi(tid, c64_i32)   # 0..3
        lane      = arith.remsi(tid, c64_i32)   # 0..63
        lane_row  = arith.remsi(lane, cBH_i32)  # 0..15 — MFMA M/N index (head/kv)
        lane_cgrp = arith.divsi(lane, cBH_i32)  # 0..3  — column group within wave's D-slice

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

            # ---- LDS regions (v3 single-buffer) ----
            lds_base = allocator.get_base()

            # scores_lds: [N_WAVES, BLOCK_H, ALIAS_ROW_STRIDE] f32
            lds_scores = STensor(SmemPtr(lds_base, lds_scores_off, T.f32,
                shape=(N_WAVES * _BLOCK_H * ALIAS_ROW_STRIDE,)),
                dtype=T.f32, shape=(N_WAVES * _BLOCK_H * ALIAS_ROW_STRIDE,))
            lds_p = STensor(SmemPtr(lds_base, lds_p_off, T.f32,
                shape=(_BLOCK_H * ALIAS_ROW_STRIDE,)),
                dtype=T.f32, shape=(_BLOCK_H * ALIAS_ROW_STRIDE,))
            # kv_lds: [BLOCK_K, KV_ROW_STRIDE] bf16 (single buffer)
            lds_kv = STensor(SmemPtr(lds_base, lds_kv_off, T.bf16,
                shape=(BLOCK_K * KV_ROW_STRIDE,)),
                dtype=T.bf16, shape=(BLOCK_K * KV_ROW_STRIDE,))
            lds_slots  = STensor(SmemPtr(lds_base, lds_slots_off,  T.i32, shape=(BLOCK_K,)),
                dtype=T.i32, shape=(BLOCK_K,))
            lds_valids = STensor(SmemPtr(lds_base, lds_valids_off, T.i32, shape=(BLOCK_K,)),
                dtype=T.i32, shape=(BLOCK_K,))

            # ---- Init m_i[ACC_ELEMS], l_i[ACC_ELEMS] ----
            sink_rsrc_init = buffer_ops.create_buffer_resource(attn_sink, max_size=True) if KV_SPLITS == 1 else None
            m_inits = []
            l_inits = []
            for _v in range_constexpr(ACC_ELEMS):
                _head_local = mfma.acc_head_static_offsets[_v]
                h_v = arith.addi(h_base, arith.addi(
                    arith.muli(lane_cgrp, arith.constant(mfma.acc_head_group_stride, type=i32)),
                    arith.constant(_head_local, type=i32),
                ))
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
                acc_inits.append(arith.constant_vector(0.0, _acc_vec_type()))

            # ---- Buffer resources ----
            idx_rsrc = buffer_ops.create_buffer_resource(kv_indices, max_size=True)
            kv_rsrc  = buffer_ops.create_buffer_resource(unified_kv,  max_size=True)
            scl_rsrc = buffer_ops.create_buffer_resource(kv_scales,   max_size=True)
            q_rsrc   = buffer_ops.create_buffer_resource(q, max_size=True)

            kv_end_m1 = arith.subi(arith.addi(kv_start, kv_len), c1_i32)

            # ---- KV tile loop ----
            # ForOp iter-args: [m_i×ACC_ELEMS, l_i×ACC_ELEMS, acc_dc0..acc_dc{D_CHUNKS_W-1}]
            for tile_idx, loop_state in range(
                _to_raw(tile_start), _to_raw(tile_end), 1,
                init=m_inits + l_inits + acc_inits,
            ):
                m_i   = [loop_state[v]                for v  in range_constexpr(ACC_ELEMS)]
                l_i   = [loop_state[ACC_ELEMS + v]    for v  in range_constexpr(ACC_ELEMS)]
                acc_i = [loop_state[2*ACC_ELEMS + dc] for dc in range_constexpr(D_CHUNKS_W)]

                tile_i32    = arith.index_cast(i32, _to_raw(tile_idx))
                k_tile_base = arith.muli(tile_i32, cBK_i32)

                # -- Step 1: Load BLOCK_K slots and validity flags into LDS --
                # wave0/cgrp0 threads (16 lanes) load 16 slots.
                is_loader = arith.andi(
                    arith.cmpi(CmpIPredicate.eq, wave_id, c0_i32),
                    arith.cmpi(CmpIPredicate.eq, lane_cgrp, c0_i32),
                )
                _if_load = scf.IfOp(_to_raw(is_loader), results_=[], has_else=False)
                with _if_then(_if_load):
                    raw_pos   = arith.addi(kv_start, arith.addi(k_tile_base, lane_row))
                    kv_end_v2 = arith.addi(kv_start, kv_len)
                    in_range  = arith.cmpi(CmpIPredicate.slt, raw_pos, kv_end_v2)
                    pos       = arith.minsi(raw_pos, kv_end_m1)
                    slot_v    = buffer_ops.buffer_load(idx_rsrc, pos, vec_width=1, dtype=i32)
                    slot      = arith.select(in_range, slot_v, arith.constant(-1, type=i32))
                    valid_i32 = arith.select(in_range, c1_i32, c0_i32)
                    lds_slots[fx.Index(lane_row)]  = slot
                    lds_valids[fx.Index(lane_row)] = valid_i32

                gpu.barrier()   # lds_slots / lds_valids visible to all lanes

                # -- Step 2: QK MFMA — load Q from VRAM, KV from VRAM → kv_lds --
                slot_j       = lds_slots[fx.Index(lane_row)]
                safe_slot_qk = arith.maxsi(slot_j, c0_i32)
                valid_kv_slot = arith.cmpi(CmpIPredicate.ne, lds_valids[fx.Index(lane_row)], c0_i32)

                # NOTE: slot read before barrier is safe — same wave wrote it above.
                # Other waves will barrier before reading.

                c_zero_bf16 = arith.constant(0.0, type=T.bf16)
                c_frag = arith.constant_vector(0.0, _acc_vec_type())

                # -- QK MFMA: accumulate S[BLOCK_H, BLOCK_K] = Q[BLOCK_H, D] x KV[D, BLOCK_K]
                # Outer loop: dc over D_CHUNKS_W acc D-chunks (each chunk = MFMA_N D-columns).
                # Inner loop: kk over QK_CALLS_PW calls per chunk (each call covers MFMA_K D-cols).
                # kv_lds is also populated here (all MFMA_N D-cols per chunk row written).
                kv_row_base = arith.muli(lane_row, cKVRS_i32)

                for dc in range_constexpr(D_CHUNKS_W):
                    for kk in range_constexpr(QK_CALLS_PW):
                        # D-column base for this inner call: dc*MFMA_N + kk*MFMA_K
                        d_base_const = dc * _MFMA_N + kk * _MFMA_K
                        col_off = arith.addi(
                            d_wave_base,
                            arith.addi(arith.constant(d_base_const, type=i32),
                                       arith.muli(lane_cgrp, c4_i32)),
                        )

                        # Q A-frag: VRAM load (4 bf16 at col_off)
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
                        qk_a_frags = vector.from_elements(T.vec(4, T.bf16), q_vals)

                        # KV VRAM → kv_lds[lane_row, col_off:+4] and QK B-frag
                        qk_kv_v4s = None  # set in if/else below
                        if quant_kv:
                            # FP8 path: load 4 fp8 bytes as i32, dequantize to bf16
                            flat_kv_byte = arith.addi(arith.muli(safe_slot_qk, cD_i32), col_off)
                            raw_kv_i32   = buffer_ops.buffer_load(kv_rsrc,
                                               arith.divsi(flat_kv_byte, c4_i32),
                                               vec_width=1, dtype=i32)
                            dc_col_i32  = arith.addi(d_wave_base, arith.constant(d_base_const, type=i32))
                            group_g_i32 = arith.divsi(dc_col_i32, cGS_i32)
                            scale_v     = buffer_ops.buffer_load(scl_rsrc,
                                              arith.addi(arith.muli(safe_slot_qk, cNG_i32), group_g_i32),
                                              vec_width=1, dtype=f32)
                            bf16_vals = []
                            for v in range_constexpr(4):
                                f32_v  = rocdl.cvt_f32_fp8(T.f32, raw_kv_i32, v)
                                bf16_v = arith.truncf(T.bf16, arith.mulf(f32_v, scale_v, fastmath=fm))
                                kv_flat = arith.addi(kv_row_base, arith.addi(col_off, arith.constant(v, type=i32)))
                                lds_kv[fx.Index(kv_flat)] = bf16_v
                                bf16_vals.append(bf16_v)
                            qk_kv_v4s = vector.from_elements(T.vec(4, T.bf16), bf16_vals)
                        else:
                            # BF16 path: load 4 bf16 as i32×2, write to kv_lds
                            flat_kv = arith.addi(arith.muli(safe_slot_qk, cD_i32), col_off)
                            raw_kv  = buffer_ops.buffer_load(kv_rsrc, arith.divsi(flat_kv, c2_i32),
                                                             vec_width=2, dtype=i32)
                            kv_v4   = vector.bitcast(T.vec(4, T.bf16), raw_kv)
                            for v in range_constexpr(4):
                                kv_elem = vector.extract(kv_v4, static_position=[v], dynamic_position=[])
                                kv_flat = arith.addi(kv_row_base, arith.addi(col_off, arith.constant(v, type=i32)))
                                lds_kv[fx.Index(kv_flat)] = kv_elem
                            # Collect B-frag from just-written kv_lds
                            b_vals_qk = []
                            for v in range_constexpr(4):
                                kv_flat_r = arith.addi(kv_row_base, arith.addi(col_off, arith.constant(v, type=i32)))
                                b_vals_qk.append(lds_kv[fx.Index(kv_flat_r)])
                            qk_kv_v4s = vector.from_elements(T.vec(4, T.bf16), b_vals_qk)

                        # Execute QK MFMA for this D sub-chunk (accumulate into c_frag)
                        c_frag = _mfma_bf16(qk_a_frags, qk_kv_v4s, c_frag)

                # -- Step 3: Write partial QK scores to scores_lds --
                # c_frag[v] = partial S[head, kv_slot=lane_row] for this wave's D-slice.
                # scores_lds flat: wave*BLOCK_H*ALIAS_ROW_STRIDE + head*ALIAS_ROW_STRIDE + kv
                for v in range_constexpr(ACC_ELEMS):
                    _head_off_v = mfma.acc_head_static_offsets[v]
                    head_idx  = arith.addi(
                        arith.muli(lane_cgrp, arith.constant(mfma.acc_head_group_stride, type=i32)),
                        arith.constant(_head_off_v, type=i32),
                    )
                    scores_flat = arith.addi(
                        arith.muli(wave_id, arith.constant(_BLOCK_H * ALIAS_ROW_STRIDE, type=i32)),
                        arith.addi(arith.muli(head_idx, cARS_i32), lane_row),
                    )
                    s_v = vector.extract(c_frag, static_position=[v], dynamic_position=[])
                    lds_scores[fx.Index(scores_flat)] = s_v

                gpu.barrier()   # scores_lds + kv_lds written by all waves

                # -- Steps 4+5: Cross-wave reduce + softmax + p_lds write --
                # Phase A: Read all scores_lds partials and compute softmax values (VGPR only).
                # Phase B: Barrier to ensure all waves finish reading scores_lds before any
                #          wave writes p_lds (p_lds aliases scores_lds[wave=0] region).
                # Phase C: Write all p_lds values.
                # Phase D: Barrier before PV step reads p_lds.
                m_new    = []
                alpha    = []
                l_new    = []
                pj_vals  = []   # softmax probabilities; written to p_lds in phase C
                head_idxs = []  # head indices for p_lds write (reused in phase C)

                # Phase A: read scores_lds, compute softmax — no LDS writes yet
                for v in range_constexpr(ACC_ELEMS):
                    _head_off_v = mfma.acc_head_static_offsets[v]
                    head_idx = arith.addi(
                        arith.muli(lane_cgrp, arith.constant(mfma.acc_head_group_stride, type=i32)),
                        arith.constant(_head_off_v, type=i32),
                    )
                    head_idxs.append(head_idx)

                    # Sum N_WAVES partials from scores_lds
                    partial_sum = c_zero_f32
                    for w in range_constexpr(N_WAVES):
                        alias_flat_w = arith.addi(
                            arith.constant(w * _BLOCK_H * ALIAS_ROW_STRIDE, type=i32),
                            arith.addi(arith.muli(head_idx, cARS_i32), lane_row),
                        )
                        partial_sum = arith.addf(partial_sum, lds_scores[fx.Index(alias_flat_w)], fastmath=fm)
                    s_v = arith.select(valid_kv_slot,
                                       arith.mulf(partial_sum, c_scale_l2, fastmath=fm),
                                       c_neg_inf)

                    # Butterfly max over kv (lane_row) dim — shuffle within wavefront (width=WAVE_SIZE)
                    s_max_v = s_v
                    for xor_off in mfma.shuffle_offsets:
                        peer    = _to_raw(ArithValue(s_max_v).shuffle_xor(xor_off, WAVE_SIZE))
                        s_max_v = arith.maximumf(s_max_v, peer)

                    m_new_v  = arith.maximumf(m_i[v], s_max_v)
                    alpha_v  = arith.select(
                        arith.cmpf(CmpFPredicate.OEQ, m_new_v, c_neg_inf),
                        c_one_f32, _fexp2(arith.subf(m_i[v], m_new_v)),
                    )
                    pj_safe_v = arith.select(valid_kv_slot,
                                             _fexp2(arith.subf(s_v, m_new_v)),
                                             c_zero_f32)

                    # Butterfly sum_p over kv dim — shuffle within wavefront (width=WAVE_SIZE)
                    sum_p_vv = pj_safe_v
                    for xor_off in mfma.shuffle_offsets:
                        peer     = _to_raw(ArithValue(sum_p_vv).shuffle_xor(xor_off, WAVE_SIZE))
                        sum_p_vv = arith.addf(sum_p_vv, peer, fastmath=fm)

                    m_new.append(m_new_v)
                    alpha.append(alpha_v)
                    l_new.append(arith.addf(arith.mulf(l_i[v], alpha_v, fastmath=fm), sum_p_vv, fastmath=fm))
                    pj_vals.append(pj_safe_v)

                # Phase B: all waves finished reading scores_lds; safe to write p_lds
                gpu.barrier()

                # Phase C: write p_lds (aliases scores_lds[wave=0] region — now safe)
                for v in range_constexpr(ACC_ELEMS):
                    p_flat = arith.addi(arith.muli(head_idxs[v], cARS_i32), lane_row)
                    lds_p[fx.Index(p_flat)] = pj_vals[v]

                gpu.barrier()   # p_lds visible for PV

                # -- Step 6: PV MFMA (per wave, reading KV from kv_lds) --
                # For each acc D-chunk dc, accumulate QK_CALLS_PW MFMA calls (inner kk loop).
                # kk-th call:
                #   A-frag: p[M=lane_row, K=kk*MFMA_K + lane_cgrp*4..+3] from p_lds
                #   B-frag: KV[K=kk*MFMA_K + lane_cgrp*4..+3, N=d_wave_base+dc*MFMA_N+lane_row]

                new_acc = []
                for dc in range_constexpr(D_CHUNKS_W):
                    d_base_i32 = arith.addi(d_wave_base, arith.constant(dc * _MFMA_N, type=i32))
                    d_col_i32  = arith.addi(d_base_i32, lane_row)

                    # Scale acc from previous tile
                    acc_frag = acc_i[dc]
                    acc_scaled = []
                    for v in range_constexpr(ACC_ELEMS):
                        val = vector.extract(acc_frag, static_position=[v], dynamic_position=[])
                        acc_scaled.append(arith.mulf(val, alpha[v], fastmath=fm))
                    acc_cur = vector.from_elements(_acc_vec_type(), acc_scaled)

                    # QK_CALLS_PW inner MFMA calls, each covering MFMA_K kv-slots
                    for kk in range_constexpr(QK_CALLS_PW):
                        kk_off = kk * _MFMA_K  # kv-slot base for this inner call

                        # A-frag: p[head=lane_row, kv=kk_off + lane_cgrp*4..+3]
                        a_vals_pv = []
                        for v in range_constexpr(4):
                            kv_k     = arith.addi(arith.muli(lane_cgrp, c4_i32),
                                                  arith.constant(kk_off + v, type=i32))
                            p_flat_r = arith.addi(arith.muli(lane_row, cARS_i32), kv_k)
                            a_vals_pv.append(arith.truncf(T.bf16, lds_p[fx.Index(p_flat_r)]))
                        a_frag_pv = vector.from_elements(_kfrag_vec_type(), a_vals_pv)

                        # B-frag: KV[kv=kk_off + lane_cgrp*4..+3, d_col]
                        b_vals_pv = []
                        for v in range_constexpr(4):
                            kv_slot_pv = arith.addi(arith.muli(lane_cgrp, c4_i32),
                                                    arith.constant(kk_off + v, type=i32))
                            kv_flat_pv = arith.addi(arith.muli(kv_slot_pv, cKVRS_i32), d_col_i32)
                            b_vals_pv.append(lds_kv[fx.Index(kv_flat_pv)])
                        b_frag_pv = vector.from_elements(_kfrag_vec_type(), b_vals_pv)

                        acc_cur = _mfma_bf16(a_frag_pv, b_frag_pv, acc_cur)

                    new_acc.append(acc_cur)

                loop_state = yield m_new + l_new + new_acc

            m_final   = [loop_state[v]                for v  in range_constexpr(ACC_ELEMS)]
            l_final   = [loop_state[ACC_ELEMS + v]    for v  in range_constexpr(ACC_ELEMS)]
            acc_final = [loop_state[2*ACC_ELEMS + dc] for dc in range_constexpr(D_CHUNKS_W)]

            # ---- Output ----
            if KV_SPLITS == 1:
                out_rsrc = buffer_ops.create_buffer_resource(out, max_size=True)
                for v in range_constexpr(ACC_ELEMS):
                    _head_off_v = mfma.acc_head_static_offsets[v]
                    h_v       = arith.addi(h_base, arith.addi(
                        arith.muli(lane_cgrp, arith.constant(mfma.acc_head_group_stride, type=i32)),
                        arith.constant(_head_off_v, type=i32),
                    ))
                    h_v_valid = arith.cmpi(CmpIPredicate.slt, h_v, cH_i32)
                    l_safe_v  = arith.maximumf(l_final[v], c_eps)
                    for dc in range_constexpr(D_CHUNKS_W):
                        d_base_i32 = arith.addi(d_wave_base, arith.constant(dc * _MFMA_N, type=i32))
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
                for v in range_constexpr(ACC_ELEMS):
                    _head_off_v = mfma.acc_head_static_offsets[v]
                    h_v       = arith.addi(h_base, arith.addi(
                        arith.muli(lane_cgrp, arith.constant(mfma.acc_head_group_stride, type=i32)),
                        arith.constant(_head_off_v, type=i32),
                    ))
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
                            d_base_i32 = arith.addi(d_wave_base, arith.constant(dc * _MFMA_N, type=i32))
                            d_col_i32  = arith.addi(d_base_i32, lane_row)
                            acc_val    = vector.extract(acc_final[dc], static_position=[v], dynamic_position=[])
                            ap_flat    = arith.addi(arith.muli(ml_flat, cD_i32), d_col_i32)
                            buffer_ops.buffer_store(acc_val, ap_rsrc, ap_flat)

    @flyc.jit
    def _launcher(
        q, unified_kv, kv_scales, kv_indices, kv_indptr, attn_sink,
        m_partial, l_partial, acc_partial, out,
        T_size: Int32,
        stream: FlyStream = fx.Stream(None),
    ):
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()

        n_head_blocks = (H + _BLOCK_H - 1) // _BLOCK_H  # _BLOCK_H = mfma.MFMA_M, captured in closure
        _kernel(
            q, unified_kv, kv_scales, kv_indices, kv_indptr, attn_sink,
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
    block_h=None,        # unused (BLOCK_H fixed by mfma shape); kept for API parity
    kv_splits=None,
    has_invalid=False,   # sentinel support not yet implemented
    skip_reduce=False,
    mfma_shape = None,  # "16x16x16" or "32x32x8"
    n_waves = None,     # 4 or 6 are generally the supported wave configs.
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
    kv_scales     : [total_pages, D//64] float32 — per-group absmax scales for fp8 KV.
                    If provided, unified_kv must be float8_e4m3fnuz. If None, bf16 path.
    kv_splits     : int or None — number of K-splits (None = auto)
    has_invalid   : bool — whether kv_indices contains -1 sentinel values
    skip_reduce   : bool — if True return (acc_partial, m_partial, l_partial)
                    without running the reduce kernel (for testing)

    Returns
    -------
    Tensor [T, H, D] bfloat16, or tuple when skip_reduce=True.
    """
    if has_invalid:
        raise NotImplementedError("has_invalid=True not yet implemented in FlyDSL kernel")

    if mfma_shape is None:
        # For T=1, we use 32x32x8 shape, otherwise 16x16x16 shape is generally better for T>1.
        # 32x32x8 shape requires n_waves = 6
        if q.shape[0] == 1:
            assert n_waves is None or n_waves == 6 or n_waves == 4, (
                f"mfma_shape=None auto-selects 32x32x8 for T=1, which requires n_waves=6 or n_wave=4; "
                f"got n_waves={n_waves}"
            )
            # Select number of waves based on D. We must have  D % n_waves == 0.
            # Prefer 6 waves for better occupancy, but if D is not divisible by 6, fall back to 4 waves.
            if q.shape[2] % 6 == 0:
                n_waves = 6
            elif q.shape[2] % 4 == 0:
                n_waves = 4
            else:
                raise ValueError(
                    f"mfma_shape=None auto-selects 32x32x8 for T=1, which requires n_waves=6 or n_wave=4; "
                    f"got n_waves={n_waves} and D={q.shape[2]} is not divisible by 6 or 4"
                )
            mfma = _MFMA_MAP["32x32x8"]
        else:
            assert n_waves is None or n_waves == 4, (
                f"mfma_shape=None auto-selects 16x16x16 for T>1, which requires n_waves=4; "
                f"got n_waves={n_waves}"
            )
            n_waves = 4
            mfma = _MFMA_MAP["16x16x16"]
    else:
        mfma = _MFMA_MAP[mfma_shape]
        assert n_waves is not None, "n_waves must be specified when mfma_shape is specified"

    T_val, H, D = q.shape
    device = q.device

    assert D % WAVE_SIZE == 0, (
        f"D={D} must be divisible by {WAVE_SIZE} (threads per wave); "
        f"got D % {WAVE_SIZE} = {D % WAVE_SIZE}"
    )

    # ---- block_k selection ----
    # BLOCK_K == MFMA_N == MFMA_M for all supported shapes.
    block_k = mfma.MFMA_N
    assert D % mfma.MFMA_K == 0, f"D={D} must be divisible by MFMA_K={mfma.MFMA_K}"
    assert D % n_waves == 0, f"D={D} must be divisible by n_waves={n_waves}"
    assert H % mfma.MFMA_M == 0, (
        f"H={H} must be divisible by MFMA_M={mfma.MFMA_M} for mfma_shape={mfma_shape}"
    )

    # ---- kv_splits auto-selection ----
    # Target: keep total CTAs near max_num_wg for good GPU utilization.
    # Larger MFMA tiles reduce head parallelism; compensate with more kv_splits.
    lds_estimate = (n_waves * mfma.MFMA_M * (mfma.MFMA_N + 1) * 4 +
                    block_k * (D + 8) * 2 + 2 * block_k * 4)
    max_occupancy = 1 if lds_estimate > 32768 else 2
    num_cus = 304  # gfx942 has 304 CUs
    max_num_wg = max_occupancy * num_cus
    n_head_blocks = (H + mfma.MFMA_M - 1) // mfma.MFMA_M
    if kv_splits is None:
        max_kv_len    = kv_indices.shape[0]
        max_kv_splits = max(1, (max_kv_len + block_k - 1) // block_k)
        kv_splits     = max(1, max_num_wg // max(1, T_val * n_head_blocks))
        kv_splits     = max(kv_splits, 2)
        kv_splits     = min(max_kv_splits, kv_splits)
        kv_splits     = 1 << (kv_splits - 1).bit_length()

    out = torch.zeros((T_val, H, D), dtype=torch.bfloat16, device=device)

    if kv_splits == 1:
        m_partial = l_partial = acc_partial = out  # dummy (not written by kernel)
    else:
        m_partial   = torch.empty((T_val, kv_splits, H), dtype=torch.float32, device=device)
        l_partial   = torch.empty_like(m_partial)
        acc_partial = torch.empty((T_val, kv_splits, H, D), dtype=torch.float32, device=device)

    from .tensor_shim import _run_compiled
    fly_stream = fx.Stream(torch.cuda.current_stream())

    launcher = _compile_pa_decode_sparse_mfma(
        H=H,
        D=D,
        BLOCK_K=block_k,
        KV_SPLITS=kv_splits,
        softmax_scale=float(softmax_scale),
        quant_kv=kv_scales is not None,
        mfma=mfma,
        n_waves=n_waves,
    )
    # The merged kernel always takes kv_scales; pass a dummy 1-element tensor for bf16.
    kv_scales_arg = kv_scales if kv_scales is not None else q.new_empty(1, dtype=torch.float32)
    _run_compiled(
        launcher,
        q, unified_kv, kv_scales_arg, kv_indices, kv_indptr, attn_sink,
        m_partial, l_partial, acc_partial, out,
        T_val,
        fly_stream,
    )

    if kv_splits == 1:
        return out

    if skip_reduce:
        return acc_partial, m_partial, l_partial

    # ---- Triton reduce kernel: combine KV_SPLITS partial states ----
    block_d      = 1 << (D - 1).bit_length()
    block_h_red  = 1
    grid_reduce  = (T_val, (H + block_h_red - 1) // block_h_red)

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
