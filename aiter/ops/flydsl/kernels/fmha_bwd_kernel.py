# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL Flash Attention Backward — MFMA kernel (gfx950).

CK kr_ktr_vr approach: K and V stay in LDS for the inner Q-block loop;
all four GEMMs (QK^T, dP, dV, dK) use mfma_f32_16x16x16bf16_1k.
dQ is accumulated per warp via MFMA and written with atomic-add.

Geometry
────────
  BLOCK_N = 64   K/V tile rows per WG  (4 warps × 16 rows each)
  BLOCK_M = 16   Q tile rows per inner Q-block iteration
  D = 128        head_dim (constexpr)

MFMA  mfma_f32_16x16x16bf16_1k (gfx950 wave64, K=16)
  A[16,16] : 4 bf16 per lane  →  row = l%16,  k-cols = (l//16)×4 .. +3
  B[16,16] : 4 bf16 per lane  →  k-rows = (l//16)×4 .. +3,  n-col = l%16
  C[16,16] : 4 f32  per lane  →  row = l%16,  n-cols = (l//16)×4 .. +3

Key layout insight
──────────────────
Loading Q_lds[M,D] (row-major) as MFMA-B gives A @ Q^T semantics, because
B[k=(l//16)×4..+3, n=l%16] is loaded from Q[l%16, D_step×16+(l//16)×4..+3]
which treats the M-index as MFMA-N and the D-index as MFMA-K.
The result C[n,m] = Σ_d K[n,d]·Q[m,d] = (K @ Q^T)[n,m] ✓

FUSED: this kernel produces dK, dV (VGPR accumulate → store) AND dQ (dS^T·K →
fp32 atomic-add into the dq workspace), CK-style. There is no separate dQ kernel.

LDS layout (~75 KB total; gfx950 limit 160 KB). Row strides padded (LDS_D=136,
LDS_M=20) for bank conflicts; transposed copies (Qᵀ/dOᵀ/Kᵀ) let the K=16 MFMA B
operand be read as a contiguous vec4 instead of a scalar column gather.
────────────────────────────────────────────────
  k_lds  [64, 136] bf16   loaded once per K-tile (resident)
  v_lds  [64, 136] bf16   loaded once per K-tile (resident)
  kt_lds [128, 68] bf16   Kᵀ, written once per K-tile → fused-dQ B read
  q_lds  [16, 136] bf16   reloaded each inner Q-block
  do_lds [16, 136] bf16   reloaded each inner Q-block
  qt_lds [128, 20] bf16   Qᵀ → dK MFMA B read
  dot_lds[128, 20] bf16   dOᵀ → dV MFMA B read
  p_lds  [64,  20] bf16   read by dV MFMA
  ds_lds [64,  20] bf16   read by dK & fused-dQ MFMA
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
# NOTE: BLOCK_N=128 (NUM_WARPS=8) was tried and gave 0% speedup on the canonical
# shape — this kernel is NOT bound by Q/dO HBM re-reads, so widening the K-tile
# does not help. Kept at 64 (compute/LDS/latency-bound; optimize those instead).
BLOCK_N = 64
BLOCK_M = 16
WMMA_SIZE = 16  # MFMA tile dimension (16×16×16)
A_FRAG = 4  # bf16 per lane for MFMA A/B (K=16 → 4 per lane)
C_FRAG = 4  # f32 per lane for MFMA C
NUM_WARPS = BLOCK_N // WMMA_SIZE  # 4 warps × 16 N-rows each
WARP_SIZE = 64
BLOCK_THREADS = NUM_WARPS * WARP_SIZE  # 256

D = 128
D_STEPS = D // WMMA_SIZE  # 8 K-steps (K=16 GEMMs)
D_STEPS_K32 = D // 32  # 4 K-steps (K=32 GEMMs: QK^T, dP)
D_TILES = D // WMMA_SIZE  # 8 D-output tiles (for dV / dK / dQ)

# ── LDS bank-conflict padding ──────────────────────────────────────────────
# Unpadded strides (D=128 bf16 = 64 dwords = 2×32 banks; BLOCK_M=16 = 8 dwords)
# make every LDS row start on bank 0, so the strided MFMA fragment reads (esp.
# _load_b_col_frag's scalar column gather) collide on the same banks — rocprof
# showed SQ_LDS_BANK_CONFLICT (807M) exceeding SQ_VALU_MFMA_BUSY_CYCLES (503M),
# i.e. the kernel is LDS-bank-conflict bound. Padding de-aliases the rows.
#   LDS_D = 136: keeps 16-byte alignment for the v8bf16 (K=32) frag loads and
#                cuts the 4-row column read from 8-way to 2-way conflicts.
#   LDS_M =  20: spreads the 16 P/dS rows across all 32 banks (conflict-free).
LDS_PAD_D = 8
LDS_PAD_M = 4
LDS_D = D + LDS_PAD_D  # 136  (K/V/Q/dO row stride, bf16)
LDS_M = BLOCK_M + LDS_PAD_M  # 20  (P/dS row stride, bf16)

# LDS byte offsets (padded strides)
_K_LDS_OFF = 0
_V_LDS_OFF = _K_LDS_OFF + BLOCK_N * LDS_D * 2
_Q_LDS_OFF = _V_LDS_OFF + BLOCK_N * LDS_D * 2
_DO_LDS_OFF = _Q_LDS_OFF + BLOCK_M * LDS_D * 2
_P_LDS_OFF = _DO_LDS_OFF + BLOCK_M * LDS_D * 2
_DS_LDS_OFF = _P_LDS_OFF + BLOCK_N * LDS_M * 2
# Transposed copies of Q and dO (stored [D, M]) so the dV/dK GEMMs read the B
# operand as contiguous vec4 rows instead of _load_b_col_frag's 4 scalar strided
# reads. Written during the Q/dO load (reusing the already-loaded value → no VGPR
# blowup, no extra barrier). LDS_MT=20 keeps the transposed rows off bank 0.
LDS_MT = BLOCK_M + 4  # 20  (Q^T/dO^T row stride, bf16; transposed M-dim)
# K^T copy ([D, N]) written ONCE per K-tile (K is loop-invariant) so the fused dQ
# reads its B=K operand as a contiguous vec4 instead of _load_b_col_frag's scalar
# strided reads. Amortized across all inner Q-blocks. LDS_NT_K=68 → conflict-free
# transposed read (bank=(d*2)%32 over 16 consecutive d) and 8-byte aligned.
LDS_NT_K = BLOCK_N + 4  # 68  (K^T row stride = N dim, bf16)
_QT_LDS_OFF = _DS_LDS_OFF + BLOCK_N * LDS_M * 2
_DOT_LDS_OFF = _QT_LDS_OFF + D * LDS_MT * 2
_KT_LDS_OFF = _DOT_LDS_OFF + D * LDS_MT * 2
_LDS_TOTAL = _KT_LDS_OFF + D * LDS_NT_K * 2  # ~75 KB (< 160 KB ✓)

ELEM_BYTES = 2  # bf16


# ── helpers ───────────────────────────────────────────────────────────────────


def _mfma_k32(a_v8bf16, b_v8bf16, c_v4f32):
    """mfma_f32_16x16x32_bf16  (gfx950, wave64, K=32) — preferred on CDNA3 gen2."""
    return rocdl.mfma_f32_16x16x32_bf16(T.f32x4, [a_v8bf16, b_v8bf16, c_v4f32, 0, 0, 0])


def _mfma_k16(a_v4i16, b_v4i16, c_v4f32):
    """mfma_f32_16x16x16bf16_1k  (K=16 fallback for M=16 summation)."""
    return rocdl.mfma_f32_16x16x16bf16_1k(T.f32x4, [a_v4i16, b_v4i16, c_v4f32, 0, 0, 0])


# Alias: D-summation GEMMs (QK^T, dP) use K=32; M-summation GEMMs use K=16
_mfma = _mfma_k16


def _bf16_to_v4i16(v4bf16):
    return vector.bitcast(ir.VectorType.get([4], T.i16), v4bf16)


def _load_a_frag(lds, row_base_i32, col_base_i32, lane):
    """Load MFMA-A[16,16] from row-major LDS (D=128 or BLOCK_M=16 stride).

    Lane l: A[l%16, (l//16)*4..+3]  from  LDS[row_base+l%16, col_base+(l//16)*4..+3].
    Returns v4i16.
    """
    i32 = T.i32
    m = arith.remui(lane, arith.constant(WMMA_SIZE, type=i32))
    kg = arith.divui(lane, arith.constant(WMMA_SIZE, type=i32))
    row = arith.addi(row_base_i32, m)
    col = arith.addi(col_base_i32, arith.muli(kg, arith.constant(A_FRAG, type=i32)))
    idx = arith.index_cast(
        T.index,
        arith.addi(arith.muli(row, arith.constant(lds._lds_cols, type=i32)), col),
    )
    return _bf16_to_v4i16(lds.lds.vec_load((idx,), A_FRAG))


def _load_b_row_frag(lds, row_base_i32, col_base_i32, lane):
    """Load MFMA-B treating row-major LDS as B^T (gives A @ B^T semantics).

    Lane l: reads LDS[row_base+l%16, col_base+(l//16)*4..+3]  (same as A-frag).
    The MFMA then computes  C += A @ LDS^T.
    """
    return _load_a_frag(lds, row_base_i32, col_base_i32, lane)


# ── K=32 variants (for D-summation GEMMs: QK^T and dP) ─────────────────────


def _load_a_frag_k32(lds, row_base_i32, col_base_i32, lane):
    """Load MFMA-A[16,32] from row-major LDS — 8 consecutive bf16 per lane.

    Lane l: A[l%16, (l//16)*8..+7] from LDS[row_base+l%16, col_base+(l//16)*8..+7].
    Returns v8bf16 (taken directly by mfma_f32_16x16x32_bf16).
    """
    i32 = T.i32
    m = arith.remui(lane, arith.constant(WMMA_SIZE, type=i32))
    kg = arith.divui(lane, arith.constant(WMMA_SIZE, type=i32))
    row = arith.addi(row_base_i32, m)
    col = arith.addi(col_base_i32, arith.muli(kg, arith.constant(8, type=i32)))
    idx = arith.index_cast(
        T.index,
        arith.addi(arith.muli(row, arith.constant(lds._lds_cols, type=i32)), col),
    )
    return lds.lds.vec_load((idx,), 8)  # v8bf16


def _load_b_row_frag_k32(lds, row_base_i32, col_base_i32, lane):
    """Load MFMA-B (row-major, B^T semantics) for K=32 — same pattern as A."""
    return _load_a_frag_k32(lds, row_base_i32, col_base_i32, lane)


def _load_b_col_frag(lds, k_base_i32, n_col_i32, lane):
    """Load MFMA-B[K=16, N=16] by column from row-major LDS.

    Lane l: B[(l//16)*4..+3, l%16] = LDS[k_base+(l//16)*4..+3, n_col_base+l%16].
    Uses 4 scalar LDS reads (column stride = lds._lds_cols).
    Returns v4i16.
    """
    i32 = T.i32
    kg = arith.divui(lane, arith.constant(WMMA_SIZE, type=i32))
    n_col = arith.addi(
        n_col_i32, arith.remui(lane, arith.constant(WMMA_SIZE, type=i32))
    )
    cols = arith.constant(lds._lds_cols, type=i32)

    v4bf16_ty = ir.VectorType.get([4], T.bf16)
    v = _llvm.mlir_undef(v4bf16_ty)
    for j in range_constexpr(A_FRAG):
        k_row = arith.addi(
            k_base_i32,
            arith.addi(
                arith.muli(kg, arith.constant(A_FRAG, type=i32)),
                arith.constant(j, type=i32),
            ),
        )
        idx = arith.index_cast(T.index, arith.addi(arith.muli(k_row, cols), n_col))
        val = lds.lds[idx]  # scalar bf16
        v = _llvm.insertelement(v, val, arith.unwrap(arith.constant(j, type=T.i32)))
    return _bf16_to_v4i16(v)


def _load_b_rowT_frag(lds, d_col_base_i32, lane):
    """Load MFMA-B[K=16, N=16] from a TRANSPOSED [D, M] LDS tile as a vec4 row.

    Equivalent fragment to _load_b_col_frag on the row-major [M, D] tile, but here
    the source is stored transposed (element (d, m) at d*lds_cols + m), so the 4
    values B[k=M=(l//16)*4..+3, n=D=d_col_base+l%16] are CONTIGUOUS in m — one
    vec4 load instead of 4 scalar strided reads. Returns v4i16.
    """
    i32 = T.i32
    m16 = arith.remui(lane, arith.constant(WMMA_SIZE, type=i32))
    kg = arith.divui(lane, arith.constant(WMMA_SIZE, type=i32))
    d_row = arith.addi(d_col_base_i32, m16)  # the n=D index → row of the [D,M] tile
    m_start = arith.muli(kg, arith.constant(A_FRAG, type=i32))  # k=M start (0/4/8/12)
    idx = arith.index_cast(
        T.index,
        arith.addi(arith.muli(d_row, arith.constant(lds._lds_cols, type=i32)), m_start),
    )
    return _bf16_to_v4i16(lds.lds.vec_load((idx,), A_FRAG))


def _load_b_rowT_frag_n(lds, d_col_base_i32, n_base_i32, lane):
    """Like _load_b_rowT_frag but with an N-group base — for Kᵀ [D, N] where N
    spans multiple 16-row groups. Reads B[k=N=n_base+(l//16)*4..+3, n=D=d_col_base
    +l%16] as a contiguous vec4 in N. Returns v4i16.
    """
    i32 = T.i32
    m16 = arith.remui(lane, arith.constant(WMMA_SIZE, type=i32))
    kg = arith.divui(lane, arith.constant(WMMA_SIZE, type=i32))
    d_row = arith.addi(d_col_base_i32, m16)
    n_start = arith.addi(n_base_i32, arith.muli(kg, arith.constant(A_FRAG, type=i32)))
    idx = arith.index_cast(
        T.index,
        arith.addi(arith.muli(d_row, arith.constant(lds._lds_cols, type=i32)), n_start),
    )
    return _bf16_to_v4i16(lds.lds.vec_load((idx,), A_FRAG))


# Wrapper carrying (STensor, lds_cols) so helpers can query the stride
class _LDS:
    def __init__(self, stensor, lds_cols):
        self.lds = stensor
        self._lds_cols = lds_cols


def _buf_load_bf16_raw(rsrc, byte_off):
    """Load one bf16 from global as i32 shifted to upper 16 bits (for bitcast to bf16)."""
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


def _buf_load_bf16_as_f32(rsrc, byte_off):
    i32 = T.i32
    f32 = T.f32
    dw = arith.shrui(byte_off, arith.constant(2, type=i32))
    hi = arith.andi(
        arith.shrui(byte_off, arith.constant(1, type=i32)), arith.constant(1, type=i32)
    )
    raw = buffer_ops.buffer_load(rsrc, dw, vec_width=1, dtype=i32)
    h16 = arith.andi(
        arith.shrui(raw, arith.muli(hi, arith.constant(16, type=i32))),
        arith.constant(0xFFFF, type=i32),
    )
    return arith.bitcast(f32, arith.shli(h16, arith.constant(16, type=i32)))


def _buf_load_f32(rsrc, dw_off):
    raw = buffer_ops.buffer_load(rsrc, dw_off, vec_width=1, dtype=T.i32)
    return arith.bitcast(T.f32, raw)


def _buf_atomic_add_f32(rsrc, dw_off, val):
    i32 = T.i32
    f32 = T.f32
    byte = arith.shli(dw_off, arith.constant(2, type=i32))
    zero = arith.constant(0, type=i32)
    _llvm.call_intrinsic(
        f32,
        "llvm.amdgcn.raw.ptr.buffer.atomic.fadd.f32",
        [val, rsrc, byte, zero, zero],
        [],
        [],
    )


def _buf_store_f32(val, rsrc, dw_off):
    buffer_ops.buffer_store(arith.bitcast(T.i32, val), rsrc, dw_off)


# ── kernel ────────────────────────────────────────────────────────────────────


def build_fmha_bwd_kernel_module(
    head_dim: int = 128,
    block_m: int = BLOCK_M,
    dtype: str = "bf16",
):
    assert head_dim == 128, "MFMA bwd requires head_dim=128"
    assert block_m == 16, "MFMA bwd requires block_m=16"
    # bf16 only: MFMA instrs and load path are bf16-hardcoded; fp16 would need the
    # f16 MFMA variants + an fp16 load/truncate path (not implemented → NaN).
    assert dtype == "bf16", "MFMA bwd is bf16-only (fp16 not implemented)"

    GPU_ARCH = get_rocm_arch()
    allocator = SmemAllocator(None, arch=GPU_ARCH, global_sym_name="fmha_bwd_mfma")
    allocator._align(allocator.ptr, 16)
    allocator.ptr = _LDS_TOTAL

    @flyc.kernel(known_block_size=[BLOCK_THREADS, 1, 1])
    def fmha_bwd_mfma_kernel(
        q_ptr: fx.Tensor,
        k_ptr: fx.Tensor,
        v_ptr: fx.Tensor,
        do_ptr: fx.Tensor,
        dq_ptr: fx.Tensor,
        dk_ptr: fx.Tensor,
        dv_ptr: fx.Tensor,
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
        stride_dkb: Int32,
        stride_dkh: Int32,
        stride_dkn: Int32,
        stride_dvb: Int32,
        stride_dvh: Int32,
        stride_dvn: Int32,
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

        sk_v = ArithValue(seqlen_k)
        sq_v = ArithValue(seqlen_q)
        nh_v = ArithValue(num_heads)
        bat_v = ArithValue(batch)
        c_bn = arith.constant(BLOCK_N, type=i32)

        num_kt = (sk_v + c_bn - arith.constant(1, type=i32)) // c_bn
        k_tile = wg_id % num_kt
        tmp = wg_id // num_kt
        head_idx = tmp % nh_v
        batch_idx = tmp // nh_v

        valid = arith.cmpi(CmpIPredicate.ult, batch_idx, bat_v)
        _if = scf.IfOp(valid)
        with ir.InsertionPoint(_if.then_block):

            wid = arith.divui(tid, arith.constant(WARP_SIZE, type=i32))
            lane = arith.remui(tid, arith.constant(WARP_SIZE, type=i32))
            # Warp wid handles N-rows [wid*16..(wid+1)*16] of the K-tile
            n_warp = arith.muli(wid, arith.constant(WMMA_SIZE, type=i32))
            k_row_base = arith.muli(k_tile, c_bn)

            # buffer resources
            q_rsrc = buffer_ops.create_buffer_resource(q_ptr, max_size=True)
            k_rsrc = buffer_ops.create_buffer_resource(k_ptr, max_size=True)
            v_rsrc = buffer_ops.create_buffer_resource(v_ptr, max_size=True)
            do_rsrc = buffer_ops.create_buffer_resource(do_ptr, max_size=True)
            dq_rsrc = buffer_ops.create_buffer_resource(dq_ptr, max_size=True)
            dk_rsrc = buffer_ops.create_buffer_resource(dk_ptr, max_size=True)
            dv_rsrc = buffer_ops.create_buffer_resource(dv_ptr, max_size=True)
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
            sb_dk = ArithValue(stride_dkb)
            sh_dk = ArithValue(stride_dkh)
            sn_dk = ArithValue(stride_dkn)
            sb_dv = ArithValue(stride_dvb)
            sh_dv = ArithValue(stride_dvh)
            sn_dv = ArithValue(stride_dvn)
            sb_l = ArithValue(stride_lb)
            sh_l = ArithValue(stride_lh)
            sm_l = ArithValue(stride_lm)
            sb_d = ArithValue(stride_db)
            sh_d = ArithValue(stride_dh)
            sm_d = ArithValue(stride_dm)

            scale_v = ArithValue(sm_scale)
            scale_log2e = arith.mulf(arith.constant(_LOG2E, type=f32), scale_v)
            log2e_c = arith.constant(_LOG2E, type=f32)
            c2 = arith.constant(ELEM_BYTES, type=i32)
            zero_v4 = arith.constant_vector(0.0, T.f32x4)

            # ── LDS setup ──────────────────────────────────────────────────
            lds_base = allocator.get_base()

            def _mk_lds(off, sz):
                return STensor(
                    SmemPtr(lds_base, off, T.bf16, shape=(sz,)),
                    dtype=T.bf16,
                    shape=(sz,),
                )

            k_lds_s = _mk_lds(_K_LDS_OFF, BLOCK_N * LDS_D)
            v_lds_s = _mk_lds(_V_LDS_OFF, BLOCK_N * LDS_D)
            q_lds_s = _mk_lds(_Q_LDS_OFF, BLOCK_M * LDS_D)
            do_lds_s = _mk_lds(_DO_LDS_OFF, BLOCK_M * LDS_D)
            p_lds_s = _mk_lds(_P_LDS_OFF, BLOCK_N * LDS_M)
            ds_lds_s = _mk_lds(_DS_LDS_OFF, BLOCK_N * LDS_M)
            qt_lds_s = _mk_lds(_QT_LDS_OFF, D * LDS_MT)  # Q^T [D, M]
            dot_lds_s = _mk_lds(_DOT_LDS_OFF, D * LDS_MT)  # dO^T [D, M]
            kt_lds_s = _mk_lds(_KT_LDS_OFF, D * LDS_NT_K)  # K^T [D, N]

            K_LDS = _LDS(k_lds_s, LDS_D)
            V_LDS = _LDS(v_lds_s, LDS_D)
            Q_LDS = _LDS(q_lds_s, LDS_D)
            DO_LDS = _LDS(do_lds_s, LDS_D)
            P_LDS = _LDS(p_lds_s, LDS_M)
            DS_LDS = _LDS(ds_lds_s, LDS_M)
            QT_LDS = _LDS(qt_lds_s, LDS_MT)
            DOT_LDS = _LDS(dot_lds_s, LDS_MT)
            KT_LDS = _LDS(kt_lds_s, LDS_NT_K)

            # ── Load K[BLOCK_N, D] into LDS  (256 threads, 32 elems each) ─
            for ei in range_constexpr(BLOCK_N * D // BLOCK_THREADS):
                lin = arith.addi(tid, arith.constant(ei * BLOCK_THREADS, type=i32))
                row = arith.divui(lin, arith.constant(D, type=i32))
                col = arith.remui(lin, arith.constant(D, type=i32))
                n_g = arith.addi(k_row_base, row)
                boff = arith.muli(
                    arith.addi(batch_idx * sb_k + head_idx * sh_k + n_g * sn_k, col), c2
                )
                k_val = arith.bitcast(
                    T.bf16,
                    arith.trunci(
                        T.i16,
                        arith.shrui(
                            _buf_load_bf16_raw(k_rsrc, boff),
                            arith.constant(16, type=i32),
                        ),
                    ),
                )
                k_lds_s[
                    arith.index_cast(
                        T.index,
                        arith.addi(
                            arith.muli(row, arith.constant(LDS_D, type=i32)), col
                        ),
                    )
                ] = k_val
                kt_lds_s[
                    arith.index_cast(
                        T.index,
                        arith.addi(
                            arith.muli(col, arith.constant(LDS_NT_K, type=i32)), row
                        ),
                    )
                ] = k_val

            # ── Load V[BLOCK_N, D] into LDS ────────────────────────────────
            for ei in range_constexpr(BLOCK_N * D // BLOCK_THREADS):
                lin = arith.addi(tid, arith.constant(ei * BLOCK_THREADS, type=i32))
                row = arith.divui(lin, arith.constant(D, type=i32))
                col = arith.remui(lin, arith.constant(D, type=i32))
                n_g = arith.addi(k_row_base, row)
                boff = arith.muli(
                    arith.addi(batch_idx * sb_v + head_idx * sh_v + n_g * sn_v, col), c2
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

            # ── dK, dV: accumulate over inner Q-block loop ─────────────────
            # Each warp accumulates [16 N-rows, D=128] → 8 D-tiles × f32x4 each
            # We carry iter_args as flat list of 2*D_TILES f32x4 values

            c_bm = arith.constant(BLOCK_M, type=i32)
            num_qb = (sq_v + c_bm - arith.constant(1, type=i32)) // c_bm
            lb = arith.constant(0, type=T.index)
            ub = arith.index_cast(T.index, num_qb)
            st = arith.constant(1, type=T.index)

            # Pre-compute per-lane indices. Defined before the for loop so they
            # dominate both the loop body AND the dK/dV write after the loop.
            kg = arith.divui(lane, arith.constant(WMMA_SIZE, type=i32))  # l//16
            m_lane = arith.remui(lane, arith.constant(WMMA_SIZE, type=i32))  # l%16
            c4_i32 = arith.constant(C_FRAG, type=i32)

            init_args = [zero_v4] * (2 * D_TILES)  # D_TILES dk + D_TILES dv
            for_op = scf.ForOp(lb, ub, st, init_args)
            with ir.InsertionPoint(for_op.body):
                q_blk = for_op.body.arguments[0]
                dk_it = [for_op.body.arguments[1 + i] for i in range_constexpr(D_TILES)]
                dv_it = [
                    for_op.body.arguments[1 + D_TILES + i]
                    for i in range_constexpr(D_TILES)
                ]

                q_blk_i32 = arith.index_cast(i32, q_blk)
                q_start = arith.muli(q_blk_i32, c_bm)

                # ── Load Q[BLOCK_M, D] into q_lds ────────────────────────
                for ei in range_constexpr(BLOCK_M * D // BLOCK_THREADS):
                    lin = arith.addi(tid, arith.constant(ei * BLOCK_THREADS, type=i32))
                    row = arith.divui(lin, arith.constant(D, type=i32))
                    col = arith.remui(lin, arith.constant(D, type=i32))
                    m_g = arith.addi(q_start, row)
                    boff = arith.muli(
                        arith.addi(
                            batch_idx * sb_q + head_idx * sh_q + m_g * sm_q, col
                        ),
                        c2,
                    )
                    q_val = arith.bitcast(
                        T.bf16,
                        arith.trunci(
                            T.i16,
                            arith.shrui(
                                _buf_load_bf16_raw(q_rsrc, boff),
                                arith.constant(16, type=i32),
                            ),
                        ),
                    )
                    q_lds_s[
                        arith.index_cast(
                            T.index,
                            arith.addi(
                                arith.muli(row, arith.constant(LDS_D, type=i32)), col
                            ),
                        )
                    ] = q_val
                    qt_lds_s[
                        arith.index_cast(
                            T.index,
                            arith.addi(
                                arith.muli(col, arith.constant(LDS_MT, type=i32)), row
                            ),
                        )
                    ] = q_val

                # ── Load dO[BLOCK_M, D] into do_lds ──────────────────────
                for ei in range_constexpr(BLOCK_M * D // BLOCK_THREADS):
                    lin = arith.addi(tid, arith.constant(ei * BLOCK_THREADS, type=i32))
                    row = arith.divui(lin, arith.constant(D, type=i32))
                    col = arith.remui(lin, arith.constant(D, type=i32))
                    m_g = arith.addi(q_start, row)
                    boff = arith.muli(
                        arith.addi(
                            batch_idx * sb_do + head_idx * sh_do + m_g * sm_do, col
                        ),
                        c2,
                    )
                    do_val = arith.bitcast(
                        T.bf16,
                        arith.trunci(
                            T.i16,
                            arith.shrui(
                                _buf_load_bf16_raw(do_rsrc, boff),
                                arith.constant(16, type=i32),
                            ),
                        ),
                    )
                    do_lds_s[
                        arith.index_cast(
                            T.index,
                            arith.addi(
                                arith.muli(row, arith.constant(LDS_D, type=i32)), col
                            ),
                        )
                    ] = do_val
                    dot_lds_s[
                        arith.index_cast(
                            T.index,
                            arith.addi(
                                arith.muli(col, arith.constant(LDS_MT, type=i32)), row
                            ),
                        )
                    ] = do_val

                gpu.barrier()

                # ── GEMM 0: S[N_warp=16, M=16] = K @ Q^T  ────────────────
                # Uses K=32 MFMA (gfx950 native): 4 steps of 32 D-elements
                s_acc = zero_v4
                for d in range_constexpr(D_STEPS_K32):
                    a = _load_a_frag_k32(
                        K_LDS, n_warp, arith.constant(d * 32, type=i32), lane
                    )
                    b = _load_b_row_frag_k32(
                        Q_LDS,
                        arith.constant(0, type=i32),
                        arith.constant(d * 32, type=i32),
                        lane,
                    )
                    s_acc = _mfma_k32(a, b, s_acc)

                # ── lse, delta for this M-block  ─────────────────────────
                # Correct C layout: C[(l//16)*4+ci, l%16] → all 4 ci share the
                # SAME Q M-row = l%16 = m_lane. Load one scalar per lane.
                m_abs_lse = arith.addi(q_start, m_lane)
                lse_val = _buf_load_f32(
                    lse_rsrc, batch_idx * sb_l + head_idx * sh_l + m_abs_lse * sm_l
                )
                dlt_val = _buf_load_f32(
                    dlt_rsrc, batch_idx * sb_d + head_idx * sh_d + m_abs_lse * sm_d
                )

                # ── Softmax: P = exp2(S * scale_log2e - lse * log2e) ─────
                p_acc = zero_v4
                for ci in range_constexpr(C_FRAG):
                    s_i = _llvm.extractelement(
                        s_acc, arith.unwrap(arith.constant(ci, type=T.i32))
                    )
                    p_i = rocdl.exp2(
                        f32,
                        arith.subf(
                            arith.mulf(s_i, scale_log2e), arith.mulf(lse_val, log2e_c)
                        ),
                    )
                    p_acc = _llvm.insertelement(
                        p_acc, p_i, arith.unwrap(arith.constant(ci, type=T.i32))
                    )

                # ── GEMM 2: dP[N,M] = V @ dO^T  (K=32) ──────────────────
                dp_acc = zero_v4
                for d in range_constexpr(D_STEPS_K32):
                    a = _load_a_frag_k32(
                        V_LDS, n_warp, arith.constant(d * 32, type=i32), lane
                    )
                    b = _load_b_row_frag_k32(
                        DO_LDS,
                        arith.constant(0, type=i32),
                        arith.constant(d * 32, type=i32),
                        lane,
                    )
                    dp_acc = _mfma_k32(a, b, dp_acc)

                # ── dS = P * (dP - delta) ─────────────────────────────────
                ds_acc = zero_v4
                for ci in range_constexpr(C_FRAG):
                    p_i = _llvm.extractelement(
                        p_acc, arith.unwrap(arith.constant(ci, type=T.i32))
                    )
                    dp_i = _llvm.extractelement(
                        dp_acc, arith.unwrap(arith.constant(ci, type=T.i32))
                    )
                    ds_i = arith.mulf(p_i, arith.subf(dp_i, dlt_val))
                    ds_acc = _llvm.insertelement(
                        ds_acc, ds_i, arith.unwrap(arith.constant(ci, type=T.i32))
                    )

                # ── Store P, dS to LDS for dV / dK / dQ GEMMs ───────────
                # Correct C layout: C[(l//16)*4+ci, l%16]
                # Row = n_warp+(l//16)*4+ci = K-tile N-row, Col = l%16 = Q M-row
                for ci in range_constexpr(C_FRAG):
                    n_abs_lds = arith.addi(
                        n_warp,
                        arith.addi(
                            arith.muli(kg, c4_i32), arith.constant(ci, type=i32)
                        ),
                    )  # K-tile N-row
                    m_col_lds = m_lane  # Q M-row = l%16
                    lds_idx = arith.index_cast(
                        T.index,
                        arith.addi(
                            arith.muli(n_abs_lds, arith.constant(LDS_M, type=i32)),
                            m_col_lds,
                        ),
                    )
                    p_lds_s[lds_idx] = arith.truncf(
                        T.bf16,
                        _llvm.extractelement(
                            p_acc, arith.unwrap(arith.constant(ci, type=T.i32))
                        ),
                    )
                    ds_lds_s[lds_idx] = arith.truncf(
                        T.bf16,
                        _llvm.extractelement(
                            ds_acc, arith.unwrap(arith.constant(ci, type=T.i32))
                        ),
                    )

                gpu.barrier()

                # ── GEMM 1: dV[N,D] += P[N,M] @ dO[M,D] ─────────────────
                # A[M=l%16, K=(l//16)*4+j] = P_LDS[n_warp+l%16, (l//16)*4+j]  (row access)
                # B[K=M_Q, N=D]: col access to DO_LDS
                new_dv = list(dv_it)
                for dt in range_constexpr(D_TILES):
                    a = _load_a_frag(P_LDS, n_warp, arith.constant(0, type=i32), lane)
                    b = _load_b_rowT_frag(
                        DOT_LDS, arith.constant(dt * WMMA_SIZE, type=i32), lane
                    )
                    new_dv[dt] = _mfma(a, b, new_dv[dt])

                # ── GEMM 3: dK[N,D] += dS[N,M] @ Q[M,D] ─────────────────
                # A[M=l%16, K=(l//16)*4+j] = DS_LDS[n_warp+l%16, (l//16)*4+j]  (row access)
                new_dk = list(dk_it)
                for dt in range_constexpr(D_TILES):
                    a = _load_a_frag(DS_LDS, n_warp, arith.constant(0, type=i32), lane)
                    b = _load_b_rowT_frag(
                        QT_LDS, arith.constant(dt * WMMA_SIZE, type=i32), lane
                    )
                    new_dk[dt] = _mfma(a, b, new_dk[dt])

                # ── Fused dQ: dQ[M,D] += dS^T @ K  → fp32 atomic-add ─────
                # CK-style single-pass dQ (no separate recompute kernel). dS is
                # already in DS_LDS [N,M] and K in K_LDS [N,D]; contract over N.
                # Each warp owns 2 of the 8 D-tiles (2*wid, 2*wid+1) and sums over
                # ALL N-groups → full dQ for those tiles, no cross-warp reduction.
                # MFMA: A=dS[k=N,i=M], B=K[k=N,j=D] (both via _load_b_col_frag).
                # C-layout: dQ[M=(l//16)*4+ci, D=l%16].
                N_GROUPS = BLOCK_N // WMMA_SIZE  # 4
                dqt0 = arith.muli(wid, arith.constant(2 * WMMA_SIZE, type=i32))
                dqt1 = arith.addi(dqt0, arith.constant(WMMA_SIZE, type=i32))
                dq0 = zero_v4
                dq1 = zero_v4
                for ng in range_constexpr(N_GROUPS):
                    ngb = arith.constant(ng * WMMA_SIZE, type=i32)
                    a_ds = _load_b_col_frag(
                        DS_LDS, ngb, arith.constant(0, type=i32), lane
                    )
                    b_k0 = _load_b_rowT_frag_n(KT_LDS, dqt0, ngb, lane)
                    b_k1 = _load_b_rowT_frag_n(KT_LDS, dqt1, ngb, lane)
                    dq0 = _mfma(a_ds, b_k0, dq0)
                    dq1 = _mfma(a_ds, b_k1, dq1)
                for tile_off, dq_v in ((dqt0, dq0), (dqt1, dq1)):
                    for ci in range_constexpr(C_FRAG):
                        val = _llvm.extractelement(
                            dq_v, arith.unwrap(arith.constant(ci, type=T.i32))
                        )
                        m_row = arith.addi(
                            q_start,
                            arith.addi(
                                arith.muli(kg, c4_i32), arith.constant(ci, type=i32)
                            ),
                        )
                        d_col = arith.addi(tile_off, m_lane)
                        dq_dw = (
                            batch_idx * sb_dq + head_idx * sh_dq + m_row * sm_dq + d_col
                        )
                        _buf_atomic_add_f32(dq_rsrc, dq_dw, arith.mulf(val, scale_v))

                gpu.barrier()
                scf.YieldOp(new_dk + new_dv)

            dk_final = [for_op.results[i] for i in range_constexpr(D_TILES)]
            dv_final = [for_op.results[D_TILES + i] for i in range_constexpr(D_TILES)]

            # ── Write dK, dV to global ─────────────────────────────────────
            # Correct C layout C[(l//16)*4+ci, l%16]:
            #   row = n_warp+(l//16)*4+ci = K-tile N-row
            #   col = l%16 = D-tile sub-column (m_lane is l%16, defined before for loop)
            for dt in range_constexpr(D_TILES):
                dk_acc = dk_final[dt]
                dv_acc = dv_final[dt]
                for ci in range_constexpr(C_FRAG):
                    dk_val = _llvm.extractelement(
                        dk_acc, arith.unwrap(arith.constant(ci, type=T.i32))
                    )
                    dv_val = _llvm.extractelement(
                        dv_acc, arith.unwrap(arith.constant(ci, type=T.i32))
                    )
                    n_row = arith.addi(
                        n_warp,
                        arith.addi(
                            arith.muli(kg, c4_i32), arith.constant(ci, type=i32)
                        ),
                    )
                    n_abs = arith.addi(k_row_base, n_row)
                    d_col = arith.addi(arith.constant(dt * WMMA_SIZE, type=i32), m_lane)
                    dk_dw = batch_idx * sb_dk + head_idx * sh_dk + n_abs * sn_dk + d_col
                    dv_dw = batch_idx * sb_dv + head_idx * sh_dv + n_abs * sn_dv + d_col
                    _buf_store_f32(arith.mulf(dk_val, scale_v), dk_rsrc, dk_dw)
                    _buf_store_f32(dv_val, dv_rsrc, dv_dw)

            scf.YieldOp([])

    # ── JIT launcher ─────────────────────────────────────────────────────────

    @flyc.jit
    def launch_fmha_bwd_kernel(
        q: fx.Tensor,
        k: fx.Tensor,
        v: fx.Tensor,
        do: fx.Tensor,
        dq: fx.Tensor,
        dk: fx.Tensor,
        dv: fx.Tensor,
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
        stride_dkb: fx.Int32,
        stride_dkh: fx.Int32,
        stride_dkn: fx.Int32,
        stride_dvb: fx.Int32,
        stride_dvh: fx.Int32,
        stride_dvn: fx.Int32,
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

        num_kt = arith.index_cast(
            T.index,
            (
                seqlen_k
                + arith.constant(BLOCK_N, type=T.i32)
                - arith.constant(1, type=T.i32)
            )
            // arith.constant(BLOCK_N, type=T.i32),
        )
        total_wgs = arith.muli(
            arith.muli(num_kt, arith.index_cast(T.index, num_heads)),
            arith.index_cast(T.index, batch),
        )

        launcher = fmha_bwd_mfma_kernel(
            q,
            k,
            v,
            do,
            dq,
            dk,
            dv,
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
            stride_dkb,
            stride_dkh,
            stride_dkn,
            stride_dvb,
            stride_dvh,
            stride_dvn,
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
        launcher.launch(
            grid=(total_wgs, 1, 1), block=(BLOCK_THREADS, 1, 1), stream=stream
        )

    return launch_fmha_bwd_kernel
