# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL Flash Attention Backward — dQ-only kernel (outer-Q, inner-K, no atomics).

Each WG owns BLOCK_M_DQ=64 Q-rows and loops over all K-tiles.
dQ accumulates in VGPR across the inner K-loop — no global atomic adds.

Geometry
────────
  Grid  : [ceil(Sq / BLOCK_M_DQ), H, B]
  Block : [256, 1, 1]  (4 wave64 warps)
  Warp  : owns M_Q-rows [wid*16..(wid+1)*16] of the Q-tile

MFMA
────
  S/dP  : mfma_f32_16x16x32_bf16  (K=32, 4 D-steps per K-group)
  dQ    : mfma_f32_16x16x16bf16_1k (K=16, 4 K-steps over BLOCK_N_DQ=64)
           A = ds_acc[k] truncated to bf16 (no LDS round-trip for dS!)
           B = K_LDS column access
           C = dq_acc[D-tile]  → written directly to global, NO atomic add

LDS layout (64 KB total; LDS_D=128, padding & transposed-B disabled — both cut
conflicts but cost VGPR/occupancy here, net slower. This kernel is occupancy-bound.)
──────────────────────────────────────────────────
  q_lds  [64, 128] bf16  16 KB  loaded once per Q-tile
  do_lds [64, 128] bf16  16 KB  loaded once per Q-tile
  k_lds  [64, 128] bf16  16 KB  reloaded each inner K-block
  v_lds  [64, 128] bf16  16 KB  reloaded each inner K-block
"""

from __future__ import annotations

import math

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, buffer_ops, gpu, range_constexpr, rocdl, vector
from flydsl.expr.typing import T, Int32
from flydsl.expr.arith import ArithValue, CmpIPredicate
from flydsl._mlir import ir
from flydsl._mlir.dialects import scf
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr
from flydsl.runtime.device import get_rocm_arch
from flydsl.compiler.kernel_function import CompilationContext

from .tensor_shim import STensor

_LOG2E = math.log2(math.e)

# ── geometry ──────────────────────────────────────────────────────────────────
BLOCK_M_DQ = 64  # Q-rows per WG
BLOCK_N_DQ = 64  # K-rows per inner loop step
WMMA_SIZE = 16
K_GROUPS = BLOCK_N_DQ // WMMA_SIZE  # 4  (groups of 16 K-rows for S/dP)
NUM_WARPS = 4
WARP_SIZE = 64
BLOCK_THREADS = NUM_WARPS * WARP_SIZE  # 256
D = 128
D_STEPS_K32 = D // 32  # 4  (K=32 MFMA steps for D-summation)
D_TILES = D // WMMA_SIZE  # 8  (D-tile outputs for dQ)
C_FRAG = 4  # f32 per lane for MFMA C
A_FRAG_K32 = 8  # bf16 per lane for K=32 MFMA A/B

# LDS bank-conflict padding: unpadded D=128 stride puts every row on bank 0, so
# both the K=32 row fragments and the _load_b_col_frag_k16 column gather collide.
# NOTE: padding is DISABLED here (LDS_PAD_D=0). Measured: LDS_D=136 cut conflicts
# 450M→52M but pushed VGPR 116→132, dropping occupancy 4→3 waves/SIMD and making
# the kernel SLOWER (313→491 ms) — this kernel keeps 8×f32x4 dQ accumulators live
# across the whole K-loop, so it is occupancy-bound, not conflict-bound. Padding
# stays wired up (set LDS_PAD_D=8 to re-enable) in case a VGPR-reduction change
# later moves it back below the 128-VGPR / 4-wave cliff.
LDS_PAD_D = 0
LDS_D = D + LDS_PAD_D  # 128 (padding disabled — see note above)

# NOTE: transposed-B (a K^T LDS copy to make the dQ-MFMA B read contiguous, like
# the dK/dV kernel) was tried and REGRESSED (313→504 ms): it pushed VGPR 128→156,
# dropping occupancy, while only cutting ~14% of conflicts — dQ's conflicts are
# dominated by the K=32 ROW reads (S, dP), not the column gather. This kernel is
# register-pressure/occupancy bound; conflict fixes that add VGPRs lose here.
# Reduce VGPR (or fuse dQ into the dK/dV pass) before revisiting.
_Q_LDS_OFF = 0
_DO_LDS_OFF = _Q_LDS_OFF + BLOCK_M_DQ * LDS_D * 2
_K_LDS_OFF = _DO_LDS_OFF + BLOCK_M_DQ * LDS_D * 2
_V_LDS_OFF = _K_LDS_OFF + BLOCK_N_DQ * LDS_D * 2
_LDS_TOTAL = _V_LDS_OFF + BLOCK_N_DQ * LDS_D * 2  # 64 KB


# ── helpers (duplicated from fmha_bwd_kernel.py to keep this file standalone) ─


def _mfma_k32(a_v8bf16, b_v8bf16, c_v4f32):
    return rocdl.mfma_f32_16x16x32_bf16(T.f32x4, [a_v8bf16, b_v8bf16, c_v4f32, 0, 0, 0])


def _mfma_k16(a_v4i16, b_v4i16, c_v4f32):
    return rocdl.mfma_f32_16x16x16bf16_1k(T.f32x4, [a_v4i16, b_v4i16, c_v4f32, 0, 0, 0])


def _f32x4_to_bf16x4_v4i16(v4f32):
    """f32x4 → v4i16 (bf16 bits) by extracting the upper 16 bits of each f32 element.

    Avoids arith.truncf (unreliable on raw MLIR Values from extractelement).
    bf16 is exactly the upper 16 bits of f32 (same exponent bias, truncated mantissa).
    """
    i32 = T.i32
    v4bf16_ty = ir.VectorType.get([4], T.bf16)
    v4i16_ty = ir.VectorType.get([4], T.i16)
    v = _llvm.mlir_undef(v4bf16_ty)
    for ci in range_constexpr(4):
        c_idx = arith.constant(ci, type=i32)
        elem_f32 = _llvm.extractelement(v4f32, arith.unwrap(c_idx))
        # Bitcast f32 bits to i32, take upper 16 bits = bf16 bits
        bits_i32 = arith.bitcast(i32, ArithValue(elem_f32))
        upper = arith.shrui(bits_i32, arith.constant(16, type=i32))
        i16_val = arith.trunci(T.i16, upper)
        bf16_val = arith.bitcast(T.bf16, i16_val)
        v = _llvm.insertelement(v, bf16_val, arith.unwrap(c_idx))
    return vector.bitcast(v4i16_ty, v)


def _buf_load_bf16_raw(rsrc, byte_off):
    i32 = T.i32
    dw = arith.shrui(byte_off, arith.constant(2, type=i32))
    hi = arith.andi(
        arith.shrui(byte_off, arith.constant(1, type=i32)), arith.constant(1, type=i32)
    )
    raw = buffer_ops.buffer_load(rsrc, dw, vec_width=1, dtype=i32)
    return arith.shli(
        arith.andi(
            arith.shrui(raw, arith.muli(hi, arith.constant(16, type=i32))),
            arith.constant(0xFFFF, type=i32),
        ),
        arith.constant(16, type=i32),
    )


def _buf_load_f32(rsrc, dw_off):
    return arith.bitcast(
        T.f32, buffer_ops.buffer_load(rsrc, dw_off, vec_width=1, dtype=T.i32)
    )


def _buf_store_f32(val, rsrc, dw_off):
    buffer_ops.buffer_store(arith.bitcast(T.i32, val), rsrc, dw_off)


class _LDS:
    def __init__(self, stensor, lds_cols):
        self.lds = stensor
        self._lds_cols = lds_cols


def _load_a_frag_k32(lds, row_base_i32, col_base_i32, lane):
    """8 consecutive bf16 per lane — MFMA A[16,32] row-major."""
    i32 = T.i32
    m = arith.remui(lane, arith.constant(WMMA_SIZE, type=i32))
    kg = arith.divui(lane, arith.constant(WMMA_SIZE, type=i32))
    row = arith.addi(row_base_i32, m)
    col = arith.addi(col_base_i32, arith.muli(kg, arith.constant(A_FRAG_K32, type=i32)))
    idx = arith.index_cast(
        T.index,
        arith.addi(arith.muli(row, arith.constant(lds._lds_cols, type=i32)), col),
    )
    return lds.lds.vec_load((idx,), A_FRAG_K32)  # v8bf16


def _load_b_row_frag_k32(lds, row_base_i32, col_base_i32, lane):
    """B^T semantics: loads Q/dO row-major → gives A @ B^T."""
    return _load_a_frag_k32(lds, row_base_i32, col_base_i32, lane)


def _load_b_col_frag_k16(lds, k_base_i32, n_col_i32, lane):
    """Column access — 4 scalar loads for B[K=16,N=16]."""
    i32 = T.i32
    kg = arith.divui(lane, arith.constant(WMMA_SIZE, type=i32))
    n_col = arith.addi(
        n_col_i32, arith.remui(lane, arith.constant(WMMA_SIZE, type=i32))
    )
    cols = arith.constant(lds._lds_cols, type=i32)
    v4bf16_ty = ir.VectorType.get([4], T.bf16)
    v = _llvm.mlir_undef(v4bf16_ty)
    for j in range_constexpr(C_FRAG):
        k_row = arith.addi(
            k_base_i32,
            arith.addi(
                arith.muli(kg, arith.constant(C_FRAG, type=i32)),
                arith.constant(j, type=i32),
            ),
        )
        idx = arith.index_cast(T.index, arith.addi(arith.muli(k_row, cols), n_col))
        v = _llvm.insertelement(
            v, lds.lds[idx], arith.unwrap(arith.constant(j, type=T.i32))
        )
    return vector.bitcast(ir.VectorType.get([4], T.i16), v)


def _load_b_rowT_frag_k16(lds, d_col_base_i32, k_base_i32, lane):
    """Load MFMA-B[K=16, N=16] from a TRANSPOSED [D, K-row] tile as a vec4 row.

    Same fragment as _load_b_col_frag_k16 on the row-major [K-row, D] tile, but
    the source K^T stores element (d, krow) at d*lds_cols + krow, so the 4 values
    B[k=(l//16)*4..+3, n=d_col_base+l%16] are CONTIGUOUS in krow — one vec4 load.
    """
    i32 = T.i32
    m16 = arith.remui(lane, arith.constant(WMMA_SIZE, type=i32))
    kg = arith.divui(lane, arith.constant(WMMA_SIZE, type=i32))
    d_row = arith.addi(d_col_base_i32, m16)  # n=D index → row of the [D,K] tile
    k_start = arith.addi(k_base_i32, arith.muli(kg, arith.constant(C_FRAG, type=i32)))
    idx = arith.index_cast(
        T.index,
        arith.addi(
            arith.muli(d_row, arith.constant(lds._lds_cols, type=i32)), k_start
        ),
    )
    return vector.bitcast(
        ir.VectorType.get([4], T.i16), lds.lds.vec_load((idx,), C_FRAG)
    )


# ── kernel ────────────────────────────────────────────────────────────────────


def build_fmha_bwd_dq_kernel_module(head_dim: int = 128, dtype: str = "bf16"):
    assert head_dim == 128
    assert dtype in ("bf16", "fp16")

    GPU_ARCH = get_rocm_arch()
    allocator = SmemAllocator(None, arch=GPU_ARCH, global_sym_name="fmha_bwd_dq_smem")
    allocator._align(allocator.ptr, 16)
    allocator.ptr = _LDS_TOTAL

    @flyc.kernel
    def fmha_bwd_dq_kernel(
        q_ptr: fx.Tensor,
        k_ptr: fx.Tensor,
        v_ptr: fx.Tensor,
        do_ptr: fx.Tensor,
        dq_ptr: fx.Tensor,
        lse_ptr: fx.Tensor,
        delta_ptr: fx.Tensor,
        sm_scale: fx.Float32,
        stride_qb: Int32,
        stride_qh: Int32,
        stride_qm: Int32,
        stride_kb: Int32,
        stride_kh: Int32,
        stride_kn: Int32,
        stride_vb: Int32,
        stride_vh: Int32,
        stride_vn: Int32,
        stride_dob: Int32,
        stride_doh: Int32,
        stride_dom: Int32,
        stride_dqb: Int32,
        stride_dqh: Int32,
        stride_dqm: Int32,
        stride_lb: Int32,
        stride_lh: Int32,
        stride_lm: Int32,
        stride_db: Int32,
        stride_dh: Int32,
        stride_dm: Int32,
        seqlen_q: Int32,
        seqlen_k: Int32,
        num_heads: Int32,
        batch: Int32,
    ):
        wg_id = ArithValue(fx.block_idx.x)
        tid = ArithValue(fx.thread_idx.x)
        i32 = T.i32
        f32 = T.f32

        c_bm = arith.constant(BLOCK_M_DQ, type=i32)
        c_bn = arith.constant(BLOCK_N_DQ, type=i32)
        sq_v = ArithValue(seqlen_q)
        sk_v = ArithValue(seqlen_k)
        nh_v = ArithValue(num_heads)
        bat_v = ArithValue(batch)

        # Decode (batch, head, q_tile) from wg_id
        num_qt = (sq_v + c_bm - arith.constant(1, type=i32)) // c_bm
        q_tile = wg_id % num_qt
        tmp = wg_id // num_qt
        head_idx = tmp % nh_v
        batch_idx = tmp // nh_v

        valid = arith.cmpi(CmpIPredicate.ult, batch_idx, bat_v)
        _if = scf.IfOp(valid)
        with ir.InsertionPoint(_if.then_block):

            wid = arith.divui(tid, arith.constant(WARP_SIZE, type=i32))
            lane = arith.remui(tid, arith.constant(WARP_SIZE, type=i32))
            # Warp wid: M_Q rows [wid*16..(wid+1)*16] in the Q-tile
            n_warp_q = arith.muli(wid, arith.constant(WMMA_SIZE, type=i32))
            # l%16 = Q M-row within warp (for lse/delta and dQ write)
            m_lane = arith.remui(lane, arith.constant(WMMA_SIZE, type=i32))
            kg = arith.divui(lane, arith.constant(WMMA_SIZE, type=i32))
            c4_i32 = arith.constant(C_FRAG, type=i32)

            q_base = arith.muli(q_tile, c_bm)  # first Q-row of this WG

            q_rsrc = buffer_ops.create_buffer_resource(q_ptr, max_size=True)
            k_rsrc = buffer_ops.create_buffer_resource(k_ptr, max_size=True)
            v_rsrc = buffer_ops.create_buffer_resource(v_ptr, max_size=True)
            do_rsrc = buffer_ops.create_buffer_resource(do_ptr, max_size=True)
            dq_rsrc = buffer_ops.create_buffer_resource(dq_ptr, max_size=True)
            lse_rsrc = buffer_ops.create_buffer_resource(lse_ptr, max_size=True)
            dlt_rsrc = buffer_ops.create_buffer_resource(delta_ptr, max_size=True)

            sb_q = ArithValue(stride_qb)
            sh_q = ArithValue(stride_qh)
            sm_q = ArithValue(stride_qm)
            sb_k = ArithValue(stride_kb)
            sh_k = ArithValue(stride_kh)
            sn_k = ArithValue(stride_kn)
            sb_v = ArithValue(stride_vb)
            sh_v = ArithValue(stride_vh)
            sn_v = ArithValue(stride_vn)
            sb_do = ArithValue(stride_dob)
            sh_do = ArithValue(stride_doh)
            sm_do = ArithValue(stride_dom)
            sb_dq = ArithValue(stride_dqb)
            sh_dq = ArithValue(stride_dqh)
            sm_dq = ArithValue(stride_dqm)
            sb_l = ArithValue(stride_lb)
            sh_l = ArithValue(stride_lh)
            sm_l = ArithValue(stride_lm)
            sb_d = ArithValue(stride_db)
            sh_d = ArithValue(stride_dh)
            sm_d = ArithValue(stride_dm)

            scale_v = ArithValue(sm_scale)
            scale_log2e = arith.mulf(arith.constant(_LOG2E, type=f32), scale_v)
            log2e_c = arith.constant(_LOG2E, type=f32)
            c2 = arith.constant(2, type=i32)
            zero_v4 = arith.constant_vector(0.0, T.f32x4)
            zero_f32 = arith.constant(0.0, type=f32)

            # ── LDS setup ──────────────────────────────────────────────────
            lds_base = allocator.get_base()
            _mk = lambda off, sz: STensor(
                SmemPtr(lds_base, off, T.bf16, shape=(sz,)), dtype=T.bf16, shape=(sz,)
            )
            q_lds_s = _mk(_Q_LDS_OFF, BLOCK_M_DQ * LDS_D)
            do_lds_s = _mk(_DO_LDS_OFF, BLOCK_M_DQ * LDS_D)
            k_lds_s = _mk(_K_LDS_OFF, BLOCK_N_DQ * LDS_D)
            v_lds_s = _mk(_V_LDS_OFF, BLOCK_N_DQ * LDS_D)

            Q_LDS = _LDS(q_lds_s, LDS_D)
            DO_LDS = _LDS(do_lds_s, LDS_D)
            K_LDS = _LDS(k_lds_s, LDS_D)
            V_LDS = _LDS(v_lds_s, LDS_D)

            # ── Load Q[BLOCK_M_DQ, D] and dO[BLOCK_M_DQ, D] into LDS ──────
            # Each loaded once per WG (Q-tile stays fixed for the whole kernel)
            for ei in range_constexpr(BLOCK_M_DQ * D // BLOCK_THREADS):
                lin = arith.addi(tid, arith.constant(ei * BLOCK_THREADS, type=i32))
                row = arith.divui(lin, arith.constant(D, type=i32))
                col = arith.remui(lin, arith.constant(D, type=i32))
                m_g = arith.addi(q_base, row)
                boff = arith.muli(
                    arith.addi(batch_idx * sb_q + head_idx * sh_q + m_g * sm_q, col), c2
                )
                q_lds_s[
                    arith.index_cast(
                        T.index,
                        arith.addi(
                            arith.muli(row, arith.constant(LDS_D, type=i32)), col
                        ),
                    )
                ] = arith.bitcast(
                    T.bf16,
                    arith.trunci(
                        T.i16,
                        arith.shrui(
                            _buf_load_bf16_raw(q_rsrc, boff),
                            arith.constant(16, type=i32),
                        ),
                    ),
                )

                do_boff = arith.muli(
                    arith.addi(batch_idx * sb_do + head_idx * sh_do + m_g * sm_do, col),
                    c2,
                )
                do_lds_s[
                    arith.index_cast(
                        T.index,
                        arith.addi(
                            arith.muli(row, arith.constant(LDS_D, type=i32)), col
                        ),
                    )
                ] = arith.bitcast(
                    T.bf16,
                    arith.trunci(
                        T.i16,
                        arith.shrui(
                            _buf_load_bf16_raw(do_rsrc, do_boff),
                            arith.constant(16, type=i32),
                        ),
                    ),
                )

            gpu.barrier()

            # ── Load lse and delta for this warp's M_Q rows ────────────────
            # One scalar per lane: m_q = q_base + wid*16 + l%16 = q_base + n_warp_q + m_lane
            m_q_lane = arith.addi(q_base, arith.addi(n_warp_q, m_lane))
            lse_val = _buf_load_f32(
                lse_rsrc, batch_idx * sb_l + head_idx * sh_l + m_q_lane * sm_l
            )
            dlt_val = _buf_load_f32(
                dlt_rsrc, batch_idx * sb_d + head_idx * sh_d + m_q_lane * sm_d
            )

            # ── dQ VGPR accumulators — one f32x4 per D-tile ───────────────
            dq_acc = [zero_v4] * D_TILES

            # ── Inner loop over K-blocks ──────────────────────────────────
            num_kb = (sk_v + c_bn - arith.constant(1, type=i32)) // c_bn
            lb = arith.constant(0, type=T.index)
            ub = arith.index_cast(T.index, num_kb)
            st = arith.constant(1, type=T.index)

            # Carry dq_acc as iter_args across K-blocks
            for_op = scf.ForOp(lb, ub, st, dq_acc)
            with ir.InsertionPoint(for_op.body):
                k_blk = for_op.body.arguments[0]
                dq_it = [for_op.body.arguments[1 + i] for i in range_constexpr(D_TILES)]

                k_blk_i32 = arith.index_cast(i32, k_blk)
                k_base = arith.muli(k_blk_i32, c_bn)  # first K-row of this K-block

                # ── Load K[BLOCK_N_DQ, D] into k_lds ──────────────────────
                for ei in range_constexpr(BLOCK_N_DQ * D // BLOCK_THREADS):
                    lin = arith.addi(tid, arith.constant(ei * BLOCK_THREADS, type=i32))
                    row = arith.divui(lin, arith.constant(D, type=i32))
                    col = arith.remui(lin, arith.constant(D, type=i32))
                    n_g = arith.addi(k_base, row)
                    boff = arith.muli(
                        arith.addi(
                            batch_idx * sb_k + head_idx * sh_k + n_g * sn_k, col
                        ),
                        c2,
                    )
                    k_lds_s[
                        arith.index_cast(
                            T.index,
                            arith.addi(
                                arith.muli(row, arith.constant(LDS_D, type=i32)), col
                            ),
                        )
                    ] = arith.bitcast(
                        T.bf16,
                        arith.trunci(
                            T.i16,
                            arith.shrui(
                                _buf_load_bf16_raw(k_rsrc, boff),
                                arith.constant(16, type=i32),
                            ),
                        ),
                    )

                # ── Load V[BLOCK_N_DQ, D] into v_lds ──────────────────────
                for ei in range_constexpr(BLOCK_N_DQ * D // BLOCK_THREADS):
                    lin = arith.addi(tid, arith.constant(ei * BLOCK_THREADS, type=i32))
                    row = arith.divui(lin, arith.constant(D, type=i32))
                    col = arith.remui(lin, arith.constant(D, type=i32))
                    n_g = arith.addi(k_base, row)
                    boff = arith.muli(
                        arith.addi(
                            batch_idx * sb_v + head_idx * sh_v + n_g * sn_v, col
                        ),
                        c2,
                    )
                    v_lds_s[
                        arith.index_cast(
                            T.index,
                            arith.addi(
                                arith.muli(row, arith.constant(LDS_D, type=i32)), col
                            ),
                        )
                    ] = arith.bitcast(
                        T.bf16,
                        arith.trunci(
                            T.i16,
                            arith.shrui(
                                _buf_load_bf16_raw(v_rsrc, boff),
                                arith.constant(16, type=i32),
                            ),
                        ),
                    )

                gpu.barrier()

                # ── For each K-group (16 K-rows): S, dP GEMMs → P, dS arithmetic → dQ MFMA
                new_dq = list(dq_it)
                for k_grp in range_constexpr(K_GROUPS):
                    k_grp_base = arith.constant(
                        k_grp * WMMA_SIZE, type=i32
                    )  # 0,16,32,48

                    # GEMM: S[K_grp=16, M_Q=16] = K[K_grp, D] @ Q^T[D, M_Q]  (K=32)
                    s_acc = zero_v4
                    for d in range_constexpr(D_STEPS_K32):
                        a_k = _load_a_frag_k32(
                            K_LDS, k_grp_base, arith.constant(d * 32, type=i32), lane
                        )
                        b_q = _load_b_row_frag_k32(
                            Q_LDS, n_warp_q, arith.constant(d * 32, type=i32), lane
                        )
                        s_acc = _mfma_k32(a_k, b_q, s_acc)

                    # GEMM: dP[K_grp=16, M_Q=16] = V[K_grp, D] @ dO^T[D, M_Q]  (K=32)
                    dp_acc = zero_v4
                    for d in range_constexpr(D_STEPS_K32):
                        a_v = _load_a_frag_k32(
                            V_LDS, k_grp_base, arith.constant(d * 32, type=i32), lane
                        )
                        b_do = _load_b_row_frag_k32(
                            DO_LDS, n_warp_q, arith.constant(d * 32, type=i32), lane
                        )
                        dp_acc = _mfma_k32(a_v, b_do, dp_acc)

                    # Compute P via exp2, then dS = P * (dP - delta)
                    # Use raw ir.Values (no ArithValue wrapping) matching the working dK/dV pattern.
                    zero_v4f32 = arith.constant_vector(0.0, T.f32x4)
                    p_acc = zero_v4f32
                    ds_acc = zero_v4f32
                    for ci in range_constexpr(C_FRAG):
                        c_idx_raw = arith.unwrap(arith.constant(ci, type=T.i32))
                        s_i = _llvm.extractelement(s_acc, c_idx_raw)
                        p_i = rocdl.exp2(
                            f32,
                            arith.subf(
                                arith.mulf(s_i, scale_log2e),
                                arith.mulf(lse_val, log2e_c),
                            ),
                        )
                        p_acc = _llvm.insertelement(p_acc, p_i, c_idx_raw)
                    for ci in range_constexpr(C_FRAG):
                        c_idx_raw = arith.unwrap(arith.constant(ci, type=T.i32))
                        p_i = _llvm.extractelement(p_acc, c_idx_raw)
                        dp_i = _llvm.extractelement(dp_acc, c_idx_raw)
                        ds_i = arith.mulf(p_i, arith.subf(dp_i, dlt_val))
                        ds_acc = _llvm.insertelement(ds_acc, ds_i, c_idx_raw)

                    # Build bf16 A-fragment: f32→bf16 via upper-16-bit extraction.
                    # arith.truncf(T.bf16, ...) silently produces zero on gfx950 —
                    # use bitcast+shrui+trunci+bitcast (round-toward-zero truncation) instead.
                    v4bf16_undef = ir.VectorType.get([4], T.bf16)
                    a_dq_raw = _llvm.mlir_undef(v4bf16_undef)
                    for ci in range_constexpr(C_FRAG):
                        c_idx_raw = arith.unwrap(arith.constant(ci, type=T.i32))
                        ds_val = _llvm.extractelement(ds_acc, c_idx_raw)
                        bits_i32 = arith.bitcast(T.i32, ArithValue(ds_val))
                        upper = arith.shrui(bits_i32, arith.constant(16, type=T.i32))
                        i16_val = arith.trunci(T.i16, upper)
                        bf16_val = arith.bitcast(T.bf16, i16_val)
                        a_dq_raw = _llvm.insertelement(a_dq_raw, bf16_val, c_idx_raw)
                    a_dq = vector.bitcast(ir.VectorType.get([4], T.i16), a_dq_raw)

                    # dQ MFMA: dQ[M_Q, D_tile] += dS^T[M_Q, K_grp] @ K[K_grp, D_tile]
                    for dt in range_constexpr(D_TILES):
                        b_dq = _load_b_col_frag_k16(
                            K_LDS,
                            k_grp_base,
                            arith.constant(dt * WMMA_SIZE, type=i32),
                            lane,
                        )
                        new_dq[dt] = _mfma_k16(a_dq, b_dq, new_dq[dt])

                gpu.barrier()  # K/V LDS safe for next K-block
                scf.YieldOp(new_dq)

            dq_final = [for_op.results[i] for i in range_constexpr(D_TILES)]

            # ── Write dQ to global (direct store, no atomics) ─────────────
            for dt in range_constexpr(D_TILES):
                acc = dq_final[dt]
                for ci in range_constexpr(C_FRAG):
                    val = _llvm.extractelement(
                        acc, arith.unwrap(arith.constant(ci, type=T.i32))
                    )
                    val_s = arith.mulf(val, scale_v)
                    m_q = arith.addi(
                        q_base,
                        arith.addi(
                            n_warp_q,
                            arith.addi(
                                arith.muli(kg, c4_i32), arith.constant(ci, type=i32)
                            ),
                        ),
                    )
                    d_col = arith.addi(arith.constant(dt * WMMA_SIZE, type=i32), m_lane)
                    dq_dw = batch_idx * sb_dq + head_idx * sh_dq + m_q * sm_dq + d_col
                    _buf_store_f32(val_s, dq_rsrc, dq_dw)

            scf.YieldOp([])

    @flyc.jit
    def launch_fmha_bwd_dq_kernel(
        q: fx.Tensor,
        k: fx.Tensor,
        v: fx.Tensor,
        do: fx.Tensor,
        dq: fx.Tensor,
        lse: fx.Tensor,
        delta: fx.Tensor,
        sm_scale: fx.Float32,
        stride_qb: fx.Int32,
        stride_qh: fx.Int32,
        stride_qm: fx.Int32,
        stride_kb: fx.Int32,
        stride_kh: fx.Int32,
        stride_kn: fx.Int32,
        stride_vb: fx.Int32,
        stride_vh: fx.Int32,
        stride_vn: fx.Int32,
        stride_dob: fx.Int32,
        stride_doh: fx.Int32,
        stride_dom: fx.Int32,
        stride_dqb: fx.Int32,
        stride_dqh: fx.Int32,
        stride_dqm: fx.Int32,
        stride_lb: fx.Int32,
        stride_lh: fx.Int32,
        stride_lm: fx.Int32,
        stride_db: fx.Int32,
        stride_dh: fx.Int32,
        stride_dm: fx.Int32,
        seqlen_q: fx.Int32,
        seqlen_k: fx.Int32,
        num_heads: fx.Int32,
        batch: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalized = False
            allocator.finalize()

        from flydsl.expr import arith as _a

        i32 = T.i32
        c_bm = arith.constant(BLOCK_M_DQ, type=i32)
        num_qt = arith.index_cast(
            T.index, (seqlen_q + c_bm - arith.constant(1, type=i32)) // c_bm
        )
        total = arith.muli(
            arith.muli(num_qt, arith.index_cast(T.index, num_heads)),
            arith.index_cast(T.index, batch),
        )

        launcher = fmha_bwd_dq_kernel(
            q,
            k,
            v,
            do,
            dq,
            lse,
            delta,
            sm_scale,
            stride_qb,
            stride_qh,
            stride_qm,
            stride_kb,
            stride_kh,
            stride_kn,
            stride_vb,
            stride_vh,
            stride_vn,
            stride_dob,
            stride_doh,
            stride_dom,
            stride_dqb,
            stride_dqh,
            stride_dqm,
            stride_lb,
            stride_lh,
            stride_lm,
            stride_db,
            stride_dh,
            stride_dm,
            seqlen_q,
            seqlen_k,
            num_heads,
            batch,
        )
        launcher.launch(grid=(total, 1, 1), block=(BLOCK_THREADS, 1, 1), stream=stream)

    return launch_fmha_bwd_dq_kernel
