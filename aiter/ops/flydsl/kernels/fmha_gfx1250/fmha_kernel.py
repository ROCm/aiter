"""
 Forward Kernel — gfx1250, Unified Prologue + Core Loop + Dynamic KV Loop.

Integrates:
  - Prologue (fmha_prologue.py): HW setup, Q load, K/V addr gen
  - Core loop (fmha_core_loop.py): GEMM1(QK) + GEMM2(PV) interleaved
  - Dynamic scf.for_ loop over KV tiles (tile_n=128, variable kv_seq_len)

Target: gfx1250 (MI450), wave32, 4 waves per TG (1TG), 1024 shared VGPRs.
Causal mask always on. num_tiles = bx + 1 (triangular).

    core_loop
    tile n = 128
    4 stages  perstage 1msb vgprbank

    1 stage -> 32 tilen wmma

    gemm1 next tile 0-32
    | wmma 24 per stage
    gemm1 next tile 32-64
    | wmma 24 per stage
    gemm1 next tile 64-96
    | wmma 24 per stage
    gemm1 next tile 96-128
    | wmma 24 per stage
    |
    |
    gemm2 cur tile 0-32
    | wmma 16 per stage
    gemm2 cur tile 32-64
    | wmma 16 per stage
    gemm2 cur tile 64-96
    | wmma 16 per stage
    gemm2 cur tile 96-128
    | wmma 16 per stage

    stage_schedule

    tdm 192
"""

from __future__ import annotations

import functools
from contextlib import contextmanager

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm as llvm_dialect
from flydsl._mlir.dialects import scf
from flydsl.expr import arith, buffer_ops, gpu, rocdl, vector
from flydsl.expr.primitive import const_expr
from flydsl.expr.rocdl import tdm_ops
from flydsl.expr.typing import T
from flydsl.utils.smem_allocator import SmemAllocator
from ..tensor_shim import _run_compiled
from flydsl.compiler.kernel_function import (
    CompilationContext,
)

from .fmha_prologue import (
    _emit_void,
    _setreg,
    _build_tdm_dgroup1,
    _split_i64_to_lo_hi,
    _phase4_q_load_flydsl,
    _phase5_head_index_div_flydsl,
    _compute_k_global_addr,
    _compute_v_global_addr,
    BLOCK_SIZE,
    K_TILE_N,
    _K_TDM_CONFIG,
    _V_TDM_CONFIG,
    K_ROW_BYTES,
    V_ROW_BYTES,
    K_SU_HALF_OFFSET,
    V_SU_HALF_OFFSET,
)

from .fmha_core_loop import (
    _get_types,
    _make_v2f32,
    _pair_k_tiles_for_wmma,
    _load_v_two_sus_from_lds,
    _pair_v_tiles_for_wmma,
    _qk_pure_su,
    _pv_pure_su,
    _load_k_su_from_lds,
    _sp_tiles_to_sp_pairs,
    _softmax_part01_only,
    _build_p_tiles_from_softmax,
    _cl_su_v3_stage,
    _cl_su_v3_stage_gemm2,
    _build_all_softmax_part2_ops,
    _build_all_softmax_gemm2_ops,
    _atom_s_wait_dscnt,
    NUM_MSB,
    WAVE_SIZE,
    Q_WMMA_PER_MSB,
    N_WMMA_K_TILES,
    N_LDS_PER_MSB,
    N_LDS_V_PER_MSB,
    N_SP_PAIRS,
    N_PV_WMMA_N,
    CNT_SU,
    SU_K_N,
    LDS_K_SU_P_SIZE,
    LDS_V_SU_P_SIZE,
    KV_K,
    KV_V,
    KV_NONE,
    LDS_INST_COUNT,
    ALU_STAGES,
    ALU_PER_STAGE,
    RLTS_LEN,
    QK_HDIM,
    V_HDIM,
    KV_BPP,
    PART2_SPLIT,
    PART2_SETUP_A,
    _rocdl_permlanex16,
    set_vgpr_bank,
)


@contextmanager
def _if_then(if_op):
    """SCF IfOp then-region helper (same as moe_gemm_2stage)."""
    with ir.InsertionPoint(if_op.then_block):
        try:
            yield if_op.then_block
        finally:
            blk = if_op.then_block
            if (not blk.operations) or not isinstance(blk.operations[-1], scf.YieldOp):
                scf.YieldOp([])


# ============================================================================
# Constants
# ============================================================================

TILE_N = K_TILE_N  # 128 — KV tile width

# ============================================================================
# SmemAllocators — 4 separate LDS regions for K/V ping-pong
# ============================================================================
# K per tile: CNT_SU(4) * LDS_K_SU_P_SIZE(0x3200)
# = 0xC800 = 51200 bytes (for QK_HDIM=192)
# V per tile: CNT_SU(4) × LDS_V_SU_P_SIZE(0x2400) = 0x9000 = 36864 bytes
#
# K_a, K_b, V_a are padded to 64KB segment boundary to prevent TDM cross-segment.
# V_b is last — no padding needed. D output reuses V_a (PV done before D store).
LDS_SEGMENT = 0x10000  # 64KB

_lds_alloc_k_a = SmemAllocator(None, arch="gfx1250", global_sym_name="smem_k_a")
_lds_alloc_k_a.ptr = LDS_SEGMENT

_lds_alloc_k_b = SmemAllocator(None, arch="gfx1250", global_sym_name="smem_k_b")
_lds_alloc_k_b.ptr = LDS_SEGMENT

_lds_alloc_v_a = SmemAllocator(None, arch="gfx1250", global_sym_name="smem_v_a")
_lds_alloc_v_a.ptr = LDS_SEGMENT

_lds_alloc_v_b = SmemAllocator(None, arch="gfx1250", global_sym_name="smem_v_b")
# 0x9000, last buffer — no segment padding
_lds_alloc_v_b.ptr = CNT_SU * LDS_V_SU_P_SIZE

TDM_D_TILE_DIM0 = 128 * 2  # 256 bytes per LDS row
TDM_D_TENSOR_DIM0 = 128 * 2
WV_SUBQD = 32
LDS_D_WV_SIZE = WV_SUBQD * TDM_D_TILE_DIM0 + 1024  # 9216 bytes per wave

_lds_allocator = _lds_alloc_k_a


# ============================================================================
# LDS base extraction helper
# ============================================================================


def _extract_lds_base_i32(memref_base):
    """Extract i32 LDS address from SmemAllocator memref base."""
    from flydsl._mlir.dialects import memref as _memref_d

    idx = _memref_d.extract_aligned_pointer_as_index(memref_base)
    return arith.unwrap(arith.index_cast(T.i32, idx))


def _build_kv_lds_addrs(lane_id, k_base_i32, v_base_i32):
    """Build kv_lds_addrs[12] from allocator i32 bases + per-lane offsets.

    Layout:
      [0..3]  K addresses: k_dh0(b0), k_dh1(b1), k_dh0_hi(b2), k_dh1_hi(b3)
      [4..11] V addresses: 2 per MSB, BOTH in the SAME bank as that MSB.
                msb=0: [4]=v_dh0(b0), [5]=v_dh1(b0)
                msb=1: [6]=v_dh0(b1), [7]=v_dh1(b1)
                msb=2: [8]=v_dh0(b2), [9]=v_dh1(b2)
                msb=3: [10]=v_dh0(b3),[11]=v_dh1(b3)

    When dst, addr_dh0 and addr_dh1 are all in bank_msb, cal_set_msb produces
    MSB=0x00/0x55/0xAA/0xFF (全-bankN) → zero s_set_vgpr_msb within each MSB group.
    Previous layout put dh0 and dh1 in DIFFERENT banks, causing alternating
    0x02↔0x03 / 0x42↔0x43 etc. switches on every consecutive V load pair.
    """
    lane_lo = arith.unwrap(arith.andi(lane_id, arith.constant(0xF, type=T.i32)))
    lane_hi = arith.unwrap(arith.shrui(lane_id, arith.constant(4, type=T.i32)))

    k_lane_off = arith.unwrap(
        arith.addi(
            arith.muli(lane_lo, arith.constant(K_ROW_BYTES, type=T.i32)),
            arith.muli(lane_hi, arith.constant(16, type=T.i32)),
        )
    )

    k_dh0 = arith.unwrap(arith.addi(k_base_i32, k_lane_off))
    k_dh1 = arith.unwrap(
        arith.addi(
            k_dh0,
            arith.constant(K_SU_HALF_OFFSET, type=T.i32),
        )
    )

    lane_and_7 = arith.unwrap(arith.andi(lane_id, arith.constant(7, type=T.i32)))
    lane_shr4 = arith.unwrap(arith.shrui(lane_id, arith.constant(4, type=T.i32)))
    v_row = arith.unwrap(
        arith.addi(lane_and_7, arith.shli(lane_shr4, arith.constant(3, type=T.i32)))
    )

    lane_shr3 = arith.unwrap(arith.shrui(lane_id, arith.constant(3, type=T.i32)))
    v_sub_col = arith.unwrap(
        arith.shli(
            arith.andi(lane_shr3, arith.constant(1, type=T.i32)),
            arith.constant(4, type=T.i32),
        )
    )

    v_lane_off = arith.unwrap(
        arith.addi(
            arith.muli(v_row, arith.constant(V_ROW_BYTES, type=T.i32)), v_sub_col
        )
    )

    v_dh0 = arith.unwrap(arith.addi(v_base_i32, v_lane_off))
    v_dh1 = arith.unwrap(
        arith.addi(
            v_dh0,
            arith.constant(V_SU_HALF_OFFSET, type=T.i32),
        )
    )

    K_COL_D_HALF = QK_HDIM * KV_BPP // 2
    k_dh0_hi = arith.unwrap(arith.addi(k_dh0, arith.constant(K_COL_D_HALF, type=T.i32)))
    k_dh1_hi = arith.unwrap(arith.addi(k_dh1, arith.constant(K_COL_D_HALF, type=T.i32)))

    rocdl.sched_barrier(0)

    # MSB-specific column-group offsets folded into each address register so
    # all 8 V SSA values are genuinely distinct → one bank hint each → no
    # allocator conflict.  _build_lds_v_schedule omits these from offset field.
    _V_COL_GROUP = (N_LDS_V_PER_MSB // 2) * 32  # 64 bytes (V_HDIM=128)
    _V_D_HALF = V_HDIM * KV_BPP // 2  # 128 bytes
    _V_MSB_EXTRA = [0, _V_D_HALF, _V_COL_GROUP, _V_D_HALF + _V_COL_GROUP]

    v_addrs = []
    for msb in range(NUM_MSB):
        extra = _V_MSB_EXTRA[msb]
        if extra == 0:
            v_dh0_b, v_dh1_b = v_dh0, v_dh1
        else:
            v_dh0_b = arith.unwrap(arith.addi(v_dh0, arith.constant(extra, type=T.i32)))
            v_dh1_b = arith.unwrap(arith.addi(v_dh1, arith.constant(extra, type=T.i32)))
        v_addrs += [set_vgpr_bank(v_dh0_b, msb), set_vgpr_bank(v_dh1_b, msb)]

    return [
        # K addresses [0..3]: each is a distinct SSA value → one bank hint each
        set_vgpr_bank(k_dh0, 0),
        set_vgpr_bank(k_dh1, 1),
        set_vgpr_bank(k_dh0_hi, 2),
        set_vgpr_bank(k_dh1_hi, 3),
        # V addresses [4..11]: 8 distinct SSA values, two per MSB bank
    ] + v_addrs


# ============================================================================
# TDM Priming — Load full tile_n=128 of K+V into LDS
# ============================================================================


def _build_tdm_descs(dg1, addr_i64, stride_adv_i64, lds_base, su_p_size, n_su):
    """Build per-SU TDM descriptors [(dg0, dg1)] without issuing loads.

    dg1 can be a single v8i32 (shared across SUs) or a list of n_su v8i32.
    """
    _dg1_list = dg1 if isinstance(dg1, list) else [dg1] * n_su
    pred = arith.constant(1, type=T.i32)
    cur_addr = addr_i64
    descs = []
    for su in range(n_su):
        lds_off = arith.unwrap(
            arith.addi(
                lds_base,
                arith.constant(su * su_p_size, type=T.i32),
            )
        )
        addr_lo, addr_hi = _split_i64_to_lo_hi(cur_addr)
        dg0 = vector.from_elements(T.vec(4, T.i32), [pred, lds_off, addr_lo, addr_hi])
        descs.append((dg0, _dg1_list[su]))
        if su < n_su - 1:
            cur_addr = arith.addi(cur_addr, stride_adv_i64)
    return descs


def _issue_tdm_from_descs(descs):
    """Issue TDM loads from pre-built descriptors, with per-SU barriers."""
    for dg0, dg1 in descs:
        tdm_ops.tensor_load_2d(tdm_ops.TDMDescriptor2D(dg0, dg1))
        rocdl.s_barrier_signal(-1)
        rocdl.s_barrier_wait(-1)


def _per_warp_oob_dim1(total_rows_i32, wave_id, rows_per_warp=8):
    wave_off = arith.unwrap(
        arith.muli(
            wave_id,
            arith.constant(rows_per_warp, type=T.i32),
        )
    )
    remaining = arith.subi(total_rows_i32, wave_off)
    clamped_lo = arith.maxsi(
        remaining,
        arith.unwrap(arith.constant(0, type=T.i32)),
    )
    return arith.minsi(
        clamped_lo,
        arith.unwrap(arith.constant(rows_per_warp, type=T.i32)),
    )


def _make_kv_dg1_with_oob(
    config_bf16, dim0_elems, dim1_rows, stride_seq_elems, oob_dim1_raw, dim0_stride=None
):
    _i32 = ir.IntegerType.get_signless(32)
    _td1_lo = arith.andi(oob_dim1_raw, arith.constant(0xFFFF, type=T.i32))
    _sgpr2 = arith.shli(_td1_lo, arith.constant(16, type=T.i32))
    if dim0_stride is None:
        dim0_stride = dim0_elems
    return vector.from_elements(
        T.vec(8, T.i32),
        [
            arith.constant(config_bf16, type=T.i32),
            arith.constant(dim0_elems << 16, type=T.i32),
            _sgpr2,
            arith.constant(dim0_stride << 16, type=T.i32),
            arith.constant(dim1_rows, type=T.i32),
            stride_seq_elems,
            arith.constant(0, type=T.i32),
            arith.constant(0, type=T.i32),
        ],
    )


def _tdm_load_kv_blk(kv_type, dg1, addr_i64, stride_adv_i64, lds_base, su_p_size, n_su):
    """Issue n_su TDM loads for one blk of K or V data.

    Each TDM load covers one SU. After all loads, issues barrier.
    dg1 can be a single v8i32 or a list of n_su v8i32 (per-SU OOB).
    """
    descs = _build_tdm_descs(dg1, addr_i64, stride_adv_i64, lds_base, su_p_size, n_su)
    _issue_tdm_from_descs(descs)


def _tdm_load_k_only(
    ptr_K,
    k_offset,
    stride_k_seq,
    stride_k_32,
    wave_id,
    lds_base_i32,
    oob_dg1_list=None,
):
    """Load one tile_n=128 of K into LDS via TDM (K only, no V).

    Single TDM: dim0=QK_HDIM=192 elements, no padding.
    QK_HDIM=192 is not a multiple of any power-of-2 pad_interval that fits
    exactly one pad per row, so padding is disabled to avoid the
    continuous-stream rotation bug that corrupts K elements.
    K_ROW_BYTES = 384 (= 192 * 2, flat, no padding).

    Per-warp: 4 warps × 8 rows = 32 rows per SU, 4 SUs total.
    """
    i64 = ir.IntegerType.get_signless(64)

    # dim0_valid=QK_HDIM(192): only read 192 bf16 from global per row.
    # dim0_stride=200: LDS inner stride = 200*2=400B = K_ROW_BYTES.
    # Previous dim0_valid=200 caused OOB reads for the last head of the last
    # token (16 extra bytes past valid K allocation).
    _DIM0_VALID = QK_HDIM  # 192 — global read width
    _DIM0_STRIDE = 200  # LDS row stride in elements
    _DIM1_ROWS = 8  # rows per warp

    _K_CONFIG_BF16 = (1 << 16) | _K_TDM_CONFIG
    stride_k_seq_elems = arith.unwrap(
        arith.shrui(stride_k_seq, arith.constant(1, type=T.i32))
    )

    k_dg1 = (
        oob_dg1_list
        if oob_dg1_list is not None
        else vector.from_elements(
            T.vec(8, T.i32),
            [
                arith.constant(_K_CONFIG_BF16, type=T.i32),
                arith.constant(_DIM0_VALID << 16, type=T.i32),
                arith.constant(_DIM1_ROWS << 16, type=T.i32),
                arith.constant(_DIM0_STRIDE << 16, type=T.i32),
                arith.constant(_DIM1_ROWS, type=T.i32),
                stride_k_seq_elems,
                arith.constant(0, type=T.i32),
                arith.constant(0, type=T.i32),
            ],
        )
    )

    # Per-warp global offset: wave_id * 8 * stride_k_seq bytes
    k_addr = _compute_k_global_addr(
        ptr_K,
        k_offset,
        wave_id,
        arith.unwrap(arith.muli(arith.constant(8, type=T.i32), stride_k_seq)),
    )

    # Per-warp LDS offset: wave_id * 8 * K_ROW_BYTES
    wid_i32 = arith.unwrap(wave_id)
    lds_warp_off = arith.muli(
        wid_i32, arith.unwrap(arith.constant(8 * K_ROW_BYTES, type=T.i32))
    )
    lds_base_with_warp = arith.addi(lds_base_i32, lds_warp_off)

    k_stride_adv = arith.extsi(i64, arith.unwrap(stride_k_32))

    _tdm_load_kv_blk(
        KV_K, k_dg1, k_addr, k_stride_adv, lds_base_with_warp, LDS_K_SU_P_SIZE, CNT_SU
    )

    rocdl.s_wait_tensorcnt(0)
    rocdl.s_barrier_signal(-1)
    rocdl.s_barrier_wait(-1)


def _tdm_load_v_only(
    ptr_V,
    v_offset,
    stride_v_seq,
    stride_v_32,
    wave_id,
    lds_base_i32,
    oob_dg1_list=None,
):
    """Load one tile_n=128 of V into LDS via TDM (V only, no K).

    4 TDM loads (SU 0-3). Per-warp distribution: 4 warps each load
    8 rows of the 32-row SU, writing to their own LDS sub-region.
    """
    i64 = ir.IntegerType.get_signless(64)

    _V_CONFIG_BF16 = (1 << 16) | _V_TDM_CONFIG
    _DIM0_ELEMS = 128  # D dimension in bf16 elements
    _DIM1_ROWS = 8  # rows per warp
    stride_v_seq_elems = arith.unwrap(
        arith.shrui(stride_v_seq, arith.constant(1, type=T.i32))
    )
    v_dg1 = (
        oob_dg1_list
        if oob_dg1_list is not None
        else vector.from_elements(
            T.vec(8, T.i32),
            [
                arith.constant(_V_CONFIG_BF16, type=T.i32),
                arith.constant(_DIM0_ELEMS << 16, type=T.i32),
                arith.constant(_DIM1_ROWS << 16, type=T.i32),
                arith.constant(_DIM0_ELEMS << 16, type=T.i32),
                arith.constant(_DIM1_ROWS, type=T.i32),
                stride_v_seq_elems,
                arith.constant(0, type=T.i32),
                arith.constant(0, type=T.i32),
            ],
        )
    )

    v_addr = _compute_v_global_addr(
        ptr_V,
        v_offset,
        wave_id,
        arith.unwrap(arith.muli(arith.constant(8, type=T.i32), stride_v_seq)),
    )

    wid_i32 = arith.unwrap(wave_id)
    lds_warp_off = arith.muli(
        wid_i32, arith.unwrap(arith.constant(8 * V_ROW_BYTES, type=T.i32))
    )
    lds_base_with_warp = arith.addi(lds_base_i32, lds_warp_off)

    v_stride_adv = arith.extsi(i64, arith.unwrap(stride_v_32))

    _tdm_load_kv_blk(
        KV_V, v_dg1, v_addr, v_stride_adv, lds_base_with_warp, LDS_V_SU_P_SIZE, CNT_SU
    )

    rocdl.s_wait_tensorcnt(0)
    rocdl.s_barrier_signal(-1)
    rocdl.s_barrier_wait(-1)


def _manual_load_k_to_lds(
    ptr_K,
    k_offset_raw,
    stride_k_seq,
    lane_id,
    wave_id,
    lds_base_i32,
):
    """Load K tile (4 SUs × 32 rows × K_ROW_BYTES) to LDS via flat_load + ds_write.

    Replaces tensor_load_to_lds for FFM-lite which doesn't support TDM.
    128 threads cooperate: each iteration covers BLOCK_SIZE chunks of 16 bytes.
    N_CHUNKS_PER_ROW = QK_HDIM * KV_BPP / 16 (24 for QK_HDIM=192).
    LDS row stride = K_ROW_BYTES (400 for QK_HDIM=192, includes 16B padding).
    """
    from flydsl._mlir.dialects import fly as _fly_d

    i32_ty = ir.IntegerType.get_signless(32)
    i64 = ir.IntegerType.get_signless(64)
    v4i32_ty = ir.VectorType.get([4], i32_ty)
    glb_ptr_type = ir.Type.parse("!llvm.ptr<1>")
    lds_ptr_type = ir.Type.parse("!llvm.ptr<3>")

    a_raw = ptr_K.__extract_to_ir_values__()[0]
    glb_ptr = _fly_d.extract_aligned_pointer_as_index(glb_ptr_type, a_raw)
    base_i64 = llvm_dialect.ptrtoint(i64, glb_ptr)
    k_off_i64 = arith.extsi(i64, k_offset_raw)
    k_base_i64 = arith.addi(base_i64, k_off_i64)

    tid = arith.addi(
        arith.muli(
            arith.unwrap(wave_id), arith.unwrap(arith.constant(WAVE_SIZE, type=T.i32))
        ),
        arith.unwrap(lane_id),
    )
    stride_i64 = arith.extsi(i64, arith.unwrap(stride_k_seq))
    c16 = arith.unwrap(arith.constant(16, type=T.i32))
    k_row_c = arith.unwrap(arith.constant(K_ROW_BYTES, type=T.i32))

    N_CHUNKS_PER_ROW = (QK_HDIM * KV_BPP) // 16
    c_chunks = arith.unwrap(arith.constant(N_CHUNKS_PER_ROW, type=T.i32))
    TOTAL_CHUNKS = SU_K_N * N_CHUNKS_PER_ROW
    N_ITERS = (TOTAL_CHUNKS + BLOCK_SIZE - 1) // BLOCK_SIZE

    for su in fx.range_constexpr(CNT_SU):
        su_row_c = arith.unwrap(arith.constant(su * SU_K_N, type=T.i32))
        su_lds_c = arith.unwrap(arith.constant(su * LDS_K_SU_P_SIZE, type=T.i32))
        for j in fx.range_constexpr(N_ITERS):
            p = arith.addi(
                tid, arith.unwrap(arith.constant(j * BLOCK_SIZE, type=T.i32))
            )
            row = arith.divui(p, c_chunks)
            chunk = arith.remui(p, c_chunks)

            global_row = arith.addi(su_row_c, row)
            g_row_off = arith.muli(arith.extsi(i64, global_row), stride_i64)
            g_chunk_off = arith.extsi(i64, arith.muli(chunk, c16))
            g_addr = arith.addi(k_base_i64, arith.addi(g_row_off, g_chunk_off))
            gptr = llvm_dialect.inttoptr(glb_ptr_type, g_addr)
            data = llvm_dialect.load(v4i32_ty, gptr)

            lds_row_off = arith.muli(row, k_row_c)
            lds_chunk_off = arith.muli(chunk, c16)
            lds_off = arith.addi(su_lds_c, arith.addi(lds_row_off, lds_chunk_off))
            lds_addr = arith.addi(lds_base_i32, lds_off)
            lptr = llvm_dialect.inttoptr(lds_ptr_type, lds_addr)
            llvm_dialect.store(data, lptr)

    _atom_s_wait_dscnt(0)
    rocdl.s_barrier_signal(-1)
    rocdl.s_barrier_wait(-1)


def _manual_load_v_to_lds(
    ptr_V,
    v_offset_raw,
    stride_v_seq,
    lane_id,
    wave_id,
    lds_base_i32,
):
    """Load V tile (4 SUs × 32 rows × 256B) to LDS via flat_load + ds_write.

    Same as K but with V_ROW_BYTES (288) row stride and LDS_V_SU_P_SIZE.
    """
    from flydsl._mlir.dialects import fly as _fly_d

    i32_ty = ir.IntegerType.get_signless(32)
    i64 = ir.IntegerType.get_signless(64)
    v4i32_ty = ir.VectorType.get([4], i32_ty)
    glb_ptr_type = ir.Type.parse("!llvm.ptr<1>")
    lds_ptr_type = ir.Type.parse("!llvm.ptr<3>")

    a_raw = ptr_V.__extract_to_ir_values__()[0]
    glb_ptr = _fly_d.extract_aligned_pointer_as_index(glb_ptr_type, a_raw)
    base_i64 = llvm_dialect.ptrtoint(i64, glb_ptr)
    v_off_i64 = arith.extsi(i64, v_offset_raw)
    v_base_i64 = arith.addi(base_i64, v_off_i64)

    tid = arith.addi(
        arith.muli(
            arith.unwrap(wave_id), arith.unwrap(arith.constant(WAVE_SIZE, type=T.i32))
        ),
        arith.unwrap(lane_id),
    )
    stride_i64 = arith.extsi(i64, arith.unwrap(stride_v_seq))
    c16 = arith.unwrap(arith.constant(16, type=T.i32))
    c4_shift = arith.unwrap(arith.constant(4, type=T.i32))
    c15_mask = arith.unwrap(arith.constant(15, type=T.i32))
    v_row_c = arith.unwrap(arith.constant(V_ROW_BYTES, type=T.i32))

    for su in fx.range_constexpr(CNT_SU):
        su_row_c = arith.unwrap(arith.constant(su * SU_K_N, type=T.i32))
        su_lds_c = arith.unwrap(arith.constant(su * LDS_V_SU_P_SIZE, type=T.i32))
        for j in fx.range_constexpr(4):
            p = arith.addi(
                tid, arith.unwrap(arith.constant(j * BLOCK_SIZE, type=T.i32))
            )
            row = arith.shrui(p, c4_shift)
            chunk = arith.andi(p, c15_mask)

            global_row = arith.addi(su_row_c, row)
            g_row_off = arith.muli(arith.extsi(i64, global_row), stride_i64)
            g_chunk_off = arith.extsi(i64, arith.muli(chunk, c16))
            g_addr = arith.addi(v_base_i64, arith.addi(g_row_off, g_chunk_off))
            gptr = llvm_dialect.inttoptr(glb_ptr_type, g_addr)
            data = llvm_dialect.load(v4i32_ty, gptr)

            lds_row_off = arith.muli(row, v_row_c)
            lds_chunk_off = arith.muli(chunk, c16)
            lds_off = arith.addi(su_lds_c, arith.addi(lds_row_off, lds_chunk_off))
            lds_addr = arith.addi(lds_base_i32, lds_off)
            lptr = llvm_dialect.inttoptr(lds_ptr_type, lds_addr)
            llvm_dialect.store(data, lptr)

    _atom_s_wait_dscnt(0)
    rocdl.s_barrier_signal(-1)
    rocdl.s_barrier_wait(-1)


def _tdm_prime_full_tile(
    ptr_K,
    ptr_V,
    k_offset,
    v_offset,
    stride_k_seq,
    stride_v_seq,
    stride_k_32,
    stride_v_32,
    wave_id,
    k_lds_base_i32,
    v_lds_base_i32,
):
    """Load one full tile_n=128 of K and V into LDS via TDM.

    K: 4 TDM loads (SU 0-3). V: 4 TDM loads (SU 0-3).
    Blocking — waits for all TDM to complete before returning.
    k_lds_base_i32/v_lds_base_i32: i32 LDS base addresses from SmemAllocator.
    """
    i64 = ir.IntegerType.get_signless(64)

    k_dg1 = _build_tdm_dgroup1(_K_TDM_CONFIG, stride_k_seq)
    v_dg1 = _build_tdm_dgroup1(_V_TDM_CONFIG, stride_v_seq)

    k_addr = _compute_k_global_addr(ptr_K, k_offset, wave_id, stride_k_32)
    v_addr = _compute_v_global_addr(ptr_V, v_offset, wave_id, stride_v_32)

    k_stride_adv = arith.extsi(i64, arith.unwrap(stride_k_32))
    v_stride_adv = arith.extsi(i64, arith.unwrap(stride_v_32))

    # K: 4 SUs (tile_n=128)
    _tdm_load_kv_blk(
        KV_K, k_dg1, k_addr, k_stride_adv, k_lds_base_i32, LDS_K_SU_P_SIZE, CNT_SU
    )

    # V: 4 SUs
    _tdm_load_kv_blk(
        KV_V, v_dg1, v_addr, v_stride_adv, v_lds_base_i32, LDS_V_SU_P_SIZE, CNT_SU
    )

    # Wait for all TDM loads
    rocdl.s_wait_tensorcnt(0)
    rocdl.s_barrier_signal(-1)
    rocdl.s_barrier_wait(-1)


# ============================================================================
# Initial K load from LDS — provides kv_tiles for core_loop entry
# ============================================================================


def _issue_k_loads(ty, kv_lds_addrs, blk, su):
    """Issue ds_load_b128 for one SU of K. Does NOT wait.

    Returns raw kv_raw[4 msb][N_LDS_PER_MSB] — caller must wait
    (rocdl.s_wait_dscnt(0)) before using the results.
    """
    from .fmha_core_loop import _atom_ds_load_b128

    su_off = (blk * CNT_SU + su) * LDS_K_SU_P_SIZE
    kv_raw = [[None] * N_LDS_PER_MSB for _ in range(NUM_MSB)]
    for msb in range(NUM_MSB):
        for v_idx in range(N_LDS_PER_MSB):
            offset = v_idx * 32 + su_off
            kv_raw[msb][v_idx] = _atom_ds_load_b128(ty, kv_lds_addrs[msb], offset, msb)
    return kv_raw


def _wait_and_pair_k(ty, kv_raw):
    """Wait for issued K loads and pair into WMMA-ready v16bf16."""
    rocdl.s_wait_dscnt(0)
    return _pair_k_tiles_for_wmma(kv_raw, ty)


def _load_initial_kv_tiles(ty, kv_lds_addrs, blk, su):
    """Load K data for one SU from LDS → kv_tiles[4 msb][2] v16bf16.

    Issues N_LDS_PER_MSB ds_load_b128 per MSB, waits, then pairs
    into WMMA-ready v16bf16 fragments.
    """
    kv_raw = _issue_k_loads(ty, kv_lds_addrs, blk, su)
    return _wait_and_pair_k(ty, kv_raw)


# ============================================================================
# Unified FMHA Kernel
# ============================================================================


@functools.lru_cache(maxsize=None)
def compile_fmha_fwd(*, is_causal: bool = False, return_lse: bool = False):
    """Compile FMHA kernel variant. Cached per (is_causal, return_lse)."""
    IS_CAUSAL = int(is_causal)
    RETURN_LSE = int(return_lse)

    @flyc.kernel(known_block_size=[BLOCK_SIZE, 1, 1])
    def fmha_fwd_kernel(
        ptr_O: fx.Pointer,
        ptr_Q: fx.Pointer,
        ptr_K: fx.Pointer,
        ptr_V: fx.Pointer,
        ptr_LSE: fx.Pointer,
        ptr_cu_seqlens_q: fx.Pointer,
        ptr_cu_seqlens_k: fx.Pointer,
        scalar_f: fx.Float32,
        stride_q_seq: fx.Int32,
        stride_k_seq: fx.Int32,
        stride_v_seq: fx.Int32,
        stride_o_seq: fx.Int32,
        stride_q_head: fx.Int32,
        stride_k_head: fx.Int32,
        stride_v_head: fx.Int32,
        stride_o_head: fx.Int32,
        gqa: fx.Int32,
        max_seqlen_q: fx.Int32,
        max_seqlen_k: fx.Int32,
    ):
        """D128 BF16 FMHA Forward — full kernel with dynamic KV loop.

        iter_args through scf.for_:
          [0..15]  o_tiles[4][4] v8f32
          [16..19] old_max[4]    f32
          [20..23] row_sums[4]   f32
          [24..31] kv_tiles[4][2] v16bf16
          [32..35] local_max[4]  f32
          [36..39] delta[4]      f32
          [40..55] per-SU sp_tiles[16] v8f32
          [56..59] ping-pong bases: k_cur, v_cur, k_next, v_next (i32)
        Total: 60 SSA values carried across iterations.
        """
        scalar_f = arith.unwrap(scalar_f)
        stride_q_seq = arith.unwrap(stride_q_seq)
        stride_k_seq = arith.unwrap(stride_k_seq)
        stride_v_seq = arith.unwrap(stride_v_seq)
        stride_o_seq = arith.unwrap(stride_o_seq)
        stride_q_head = arith.unwrap(stride_q_head)
        stride_k_head = arith.unwrap(stride_k_head)
        stride_v_head = arith.unwrap(stride_v_head)
        stride_o_head = arith.unwrap(stride_o_head)
        gqa = arith.unwrap(gqa)

        # Per-head byte strides are now runtime parameters (stride_q/k/v/o_head).
        # actual_q_len / actual_kv_len are derived later from cu_seqlens (THD).

        ty = _get_types()

        # ================================================================
        # SECTION 1: Prologue — HW Setup + Q Load + Address Gen
        # ================================================================

        _setreg(2074, 2)  # WAVE_SCHED_MODE = 2
        rocdl.s_nop(0)

        tx = arith.index_cast(T.i32, gpu.thread_id("x"))
        lane_id = arith.andi(tx, arith.constant(31, type=T.i32))
        wave_id = arith.shrui(tx, arith.constant(5, type=T.i32))

        # Grid layout: [B, num_m, H] — batch on x, m-block on y, head on z.
        # Software XCD remap (HipKittens style):
        #   flat wgid = raw_x + gdx*(raw_y + gdy*raw_z)
        # This converts hardware round-robin XCD assignment to chunked assignment:
        #   XCD i gets wgids [i*(NUM_WGS/8) .. (i+1)*(NUM_WGS/8)-1]
        # → workgroups with nearby new_wgid share the same XCD → K/V cache locality.
        _NUM_XCDS = arith.constant(8, type=T.i32)
        _raw_bx = arith.index_cast(T.i32, gpu.block_id("x"))  # raw batch
        _raw_by = arith.index_cast(T.i32, gpu.block_id("y"))  # raw m-block
        _raw_bz = arith.index_cast(T.i32, gpu.block_id("z"))  # raw head
        _gdx = arith.unwrap(gpu.grid_dim.x)  # B
        _gdy = arith.unwrap(gpu.grid_dim.y)  # M
        _gdz = arith.unwrap(gpu.grid_dim.z)  # H
        # flat wgid
        _wgid = arith.addi(
            arith.addi(_raw_bx, arith.muli(_gdx, _raw_by)),
            arith.muli(arith.muli(_gdx, _gdy), _raw_bz),
        )
        # total workgroups
        _num_wgs = arith.muli(arith.muli(_gdx, _gdy), _gdz)
        # Guard: only remap when num_wgs is a positive multiple of NUM_XCDS.
        # Otherwise (num_wgs/8 truncates), the formula maps distinct wgids to
        # the same new_wgid → workgroup collision and skipped tiles.
        _wgs_per_xcd = arith.divui(_num_wgs, _NUM_XCDS)
        _num_wgs_rem = arith.remui(_num_wgs, _NUM_XCDS)
        _is_gt = arith.cmpi(arith.CmpIPredicate.ugt, _num_wgs, _NUM_XCDS)
        _is_mul = arith.cmpi(
            arith.CmpIPredicate.eq, _num_wgs_rem, arith.constant(0, type=T.i32)
        )
        _do_remap = arith.andi(_is_gt, _is_mul)
        _new_wgid_remapped = arith.addi(
            arith.muli(arith.remui(_wgid, _NUM_XCDS), _wgs_per_xcd),
            arith.divui(_wgid, _NUM_XCDS),
        )
        _new_wgid = arith.select(_do_remap, _new_wgid_remapped, _wgid)
        # decompose back to 3D: x = new%gdx, y = (new/gdx)%gdy, z = new/(gdx*gdy)
        _new_bx = arith.remui(_new_wgid, _gdx)
        _new_tmp = arith.divui(_new_wgid, _gdx)
        _new_by = arith.remui(_new_tmp, _gdy)
        _new_bz = arith.divui(_new_tmp, _gdy)
        bz = _new_bx  # batch    (grid.x)
        bx = _new_by  # m-block  (grid.y)
        by = _new_bz  # head     (grid.z)

        m_start = arith.muli(bx, arith.constant(TILE_N, type=T.i32))

        # THD Step 2: load cu_seqlens -> q_start_tok,
        # k_start_tok (in tokens, broadcast as SGPR)
        _i32_ty = ir.IntegerType.get_signless(32)
        _i64_ty = ir.IntegerType.get_signless(64)
        _glb_ptr_ty = ir.Type.parse("!llvm.ptr<1>")
        from flydsl._mlir.dialects import fly as _fly_cu

        def _load_cu_seqlen(ptr_tensor, idx_i32):
            _raw_t = ptr_tensor.__extract_to_ir_values__()[0]
            _base_ptr = _fly_cu.extract_aligned_pointer_as_index(_glb_ptr_ty, _raw_t)
            _base_i64 = llvm_dialect.ptrtoint(_i64_ty, _base_ptr)
            _byte_off = arith.muli(idx_i32, arith.unwrap(arith.constant(4, type=T.i32)))
            _byte_off_64 = arith.extsi(_i64_ty, _byte_off)
            _addr_i64 = arith.addi(_base_i64, _byte_off_64)
            _addr_ptr = llvm_dialect.inttoptr(_glb_ptr_ty, _addr_i64)
            return llvm_dialect.load(_i32_ty, _addr_ptr)

        _bz_raw = arith.unwrap(bz)
        _bz1_raw = arith.addi(_bz_raw, arith.unwrap(arith.constant(1, type=T.i32)))
        q_start_tok = arith.unwrap(
            rocdl.readfirstlane(
                T.i32,
                _load_cu_seqlen(ptr_cu_seqlens_q, _bz_raw),
            )
        )
        q_end_tok = arith.unwrap(
            rocdl.readfirstlane(
                T.i32,
                _load_cu_seqlen(ptr_cu_seqlens_q, _bz1_raw),
            )
        )
        k_start_tok = arith.unwrap(
            rocdl.readfirstlane(
                T.i32,
                _load_cu_seqlen(ptr_cu_seqlens_k, _bz_raw),
            )
        )
        k_end_tok = arith.unwrap(
            rocdl.readfirstlane(
                T.i32,
                _load_cu_seqlen(ptr_cu_seqlens_k, _bz1_raw),
            )
        )
        actual_q_len = arith.subi(q_end_tok, q_start_tok)
        actual_kv_len = arith.subi(k_end_tok, k_start_tok)

        def _apply_causal_mask(su_sp_tiles, n_start_fx):
            _lane_lo = arith.andi(lane_id, arith.constant(15, type=T.i32))
            _lane_hi_x8 = arith.muli(
                arith.shrui(lane_id, arith.constant(4, type=T.i32)),
                arith.constant(8, type=T.i32),
            )
            _wave_x32 = arith.muli(wave_id, arith.constant(32, type=T.i32))
            _base = arith.addi(
                arith.addi(arith.subi(m_start, n_start_fx), _wave_x32),
                arith.subi(_lane_lo, _lane_hi_x8),
            )
            _neg_inf_c = arith.constant(float("-inf"), type=T.f32)
            for _su in fx.range_constexpr(CNT_SU):
                for _msb in fx.range_constexpr(NUM_MSB):
                    _off = (_msb // 2) * 16 - _su * 32 - (_msb % 2) * 16
                    _bnd_fx = arith.addi(_base, arith.constant(_off, type=T.i32))
                    _v8 = su_sp_tiles[_su][_msb][0]
                    for _e in fx.range_constexpr(8):
                        _e_idx = arith.unwrap(arith.constant(_e, type=T.i32))
                        _cmp_fx = arith.cmpi(
                            arith.CmpIPredicate.slt,
                            _bnd_fx,
                            arith.constant(_e, type=T.i32),
                        )
                        _elem_raw = llvm_dialect.extractelement(_v8, _e_idx)
                        _elem_fx = arith.bitcast(T.f32, _elem_raw)
                        _mval_fx = arith.select(_cmp_fx, _neg_inf_c, _elem_fx)
                        _v8 = llvm_dialect.insertelement(
                            _v8,
                            arith.unwrap(_mval_fx),
                            _e_idx,
                        )
                    su_sp_tiles[_su][_msb][0] = _v8

        def _apply_kv_oob_mask(su_sp_tiles, kv_remain_raw):
            _i32 = ir.IntegerType.get_signless(32)
            _f32 = ir.F32Type.get()

            def _c(v):
                return arith.constant(v, type=T.i32)

            _lane_hi = arith.shrui(arith.unwrap(lane_id), _c(4))
            _lane_hi_x8 = arith.muli(_lane_hi, _c(8))
            _base = arith.subi(arith.subi(kv_remain_raw, _c(1)), _lane_hi_x8)
            _neg_inf = arith.constant(float("-inf"), type=T.f32)
            for _su in fx.range_constexpr(CNT_SU):
                for _msb in fx.range_constexpr(NUM_MSB):
                    _col_base_val = _su * 32 + (_msb % 2) * 16
                    _bnd = arith.subi(_base, _c(_col_base_val))
                    _v8 = su_sp_tiles[_su][_msb][0]
                    for _e in fx.range_constexpr(8):
                        _ev = _c(_e)
                        _cmp = arith.cmpi(arith.CmpIPredicate.slt, _bnd, _ev)
                        _elem = llvm_dialect.extractelement(_v8, _ev)
                        _mval = arith.select(_cmp, _neg_inf, _elem)
                        _v8 = llvm_dialect.insertelement(_v8, _mval, _ev)
                    su_sp_tiles[_su][_msb][0] = _v8

        # ================================================================
        # OOB protection: pre-compute per-block dg1 lists for TDM loads
        # ================================================================
        _stride_k_elems_oob = arith.unwrap(
            arith.shrui(stride_k_seq, arith.constant(1, type=T.i32))
        )
        _stride_v_elems_oob = arith.unwrap(
            arith.shrui(stride_v_seq, arith.constant(1, type=T.i32))
        )
        _K_CFG_OOB = (1 << 16) | _K_TDM_CONFIG
        _V_CFG_OOB = (1 << 16) | _V_TDM_CONFIG
        m_start_raw = arith.unwrap(m_start)

        _k_oob_dg1 = [
            _make_kv_dg1_with_oob(
                _K_CFG_OOB,
                QK_HDIM,
                8,
                _stride_k_elems_oob,
                _per_warp_oob_dim1(
                    arith.subi(
                        actual_kv_len,
                        arith.unwrap(arith.constant(_su * 32, type=T.i32)),
                    ),
                    wave_id,
                    8,
                ),
                dim0_stride=200,
            )
            for _su in range(CNT_SU)
        ]
        _v_oob_dg1 = [
            _make_kv_dg1_with_oob(
                _V_CFG_OOB,
                128,
                8,
                _stride_v_elems_oob,
                _per_warp_oob_dim1(
                    arith.subi(
                        actual_kv_len,
                        arith.unwrap(arith.constant(_su * 32, type=T.i32)),
                    ),
                    wave_id,
                    8,
                ),
            )
            for _su in range(CNT_SU)
        ]
        # THD: clamp q_remain to ≥ 0 so excess workgroups (m_start >= actual_q_len)
        # write nothing.
        _q_remain_raw = arith.subi(actual_q_len, m_start_raw)
        _q_remain_o = arith.maxsi(
            _q_remain_raw, arith.unwrap(arith.constant(0, type=T.i32))
        )
        _o_oob_dim1 = _per_warp_oob_dim1(_q_remain_o, wave_id, 32)

        # THD: skip workgroups whose m_start >= actual_q_len (no valid tokens for this
        # tile).
        # Everything below (Q load / K/V load / core loop / O writeback) is gated by
        # _wg_valid.
        _wg_valid = arith.cmpi(arith.CmpIPredicate.slt, m_start_raw, actual_q_len)
        # ---- seqlen_k == 0: zero-fill output for valid Q rows, then skip compute ----
        _kv_is_zero = arith.cmpi(
            arith.CmpIPredicate.eq,
            actual_kv_len,
            arith.unwrap(arith.constant(0, type=T.i32)),
        )
        _need_zero = arith.andi(_wg_valid, _kv_is_zero)
        _if_zero_fill = scf.IfOp(arith.unwrap(_need_zero))
        with _if_then(_if_zero_fill):
            _i64z = ir.IntegerType.get_signless(64)
            _glbpz = ir.Type.parse("!llvm.ptr<1>")
            _i32z = ir.IntegerType.get_signless(32)
            _v4i32z = ir.VectorType.get([4], _i32z)
            from flydsl._mlir.dialects import fly as _fly_z

            _o_raw_z = ptr_O.__extract_to_ir_values__()[0]
            _o_gp_z = _fly_z.extract_aligned_pointer_as_index(_glbpz, _o_raw_z)
            _o_base_z = llvm_dialect.ptrtoint(_i64z, _o_gp_z)
            _tid_z = arith.addi(
                arith.muli(
                    arith.unwrap(wave_id),
                    arith.unwrap(arith.constant(WAVE_SIZE, type=T.i32)),
                ),
                arith.unwrap(lane_id),
            )
            _zero_v4 = arith.unwrap(arith.constant_vector(0, T.vec(4, T.i32)))
            _q_rows = arith.subi(actual_q_len, m_start_raw)
            _q_rows_clamped = arith.maxsi(
                _q_rows, arith.unwrap(arith.constant(0, type=T.i32))
            )
            _q_tok_z = arith.addi(q_start_tok, arith.addi(m_start_raw, _tid_z))
            _valid_z = arith.cmpi(arith.CmpIPredicate.slt, _tid_z, _q_rows_clamped)
            _if_valid_z = scf.IfOp(_valid_z)
            with _if_then(_if_valid_z):
                _elem_off_z = arith.addi(
                    arith.muli(arith.unwrap(by), stride_o_head),
                    arith.muli(_q_tok_z, stride_o_seq),
                )
                _byte_off_z = arith.muli(
                    _elem_off_z, arith.unwrap(arith.constant(2, type=T.i32))
                )
                _byte_off_z64 = arith.extsi(_i64z, _byte_off_z)
                _o_addr_z = arith.addi(_o_base_z, _byte_off_z64)
                for _chunk_z in fx.range_constexpr(V_HDIM // 8):
                    _chunk_addr = arith.addi(
                        _o_addr_z,
                        arith.extsi(
                            _i64z,
                            arith.unwrap(arith.constant(_chunk_z * 16, type=T.i32)),
                        ),
                    )
                    _ptr_z = llvm_dialect.inttoptr(_glbpz, _chunk_addr)
                    llvm_dialect.store(_zero_v4, _ptr_z)
                # LSE = -inf for seqlen_k==0 rows
                if const_expr(RETURN_LSE):
                    from flydsl._mlir.dialects import fly as _fly_zl
                    from flydsl._mlir.dialects import llvm as _llvm_zl

                    _neg_inf_zl = arith.unwrap(
                        arith.constant(float("-inf"), type=T.f32)
                    )
                    _lse_raw_z = ptr_LSE.__extract_to_ir_values__()[0]
                    _lse_gp_z = _fly_zl.extract_aligned_pointer_as_index(
                        _glbpz, _lse_raw_z
                    )
                    _lse_base_z = llvm_dialect.ptrtoint(_i64z, _lse_gp_z)
                    # lse layout: (total_q, nheads), elem_off = tok * nheads + head
                    _lse_elem_z = arith.addi(
                        arith.muli(_q_tok_z, _gdz), arith.unwrap(by)
                    )
                    _lse_byte_z = arith.muli(
                        _lse_elem_z,
                        arith.unwrap(arith.constant(4, type=T.i32)),
                    )
                    _lse_byte_z64 = arith.extsi(_i64z, _lse_byte_z)
                    _lse_addr_z = arith.addi(_lse_base_z, _lse_byte_z64)
                    _lse_ptr_z = _llvm_zl.inttoptr(_glbpz, _lse_addr_z)
                    _llvm_zl.store(_neg_inf_zl, _lse_ptr_z)

        # Gate the main compute path: need valid Q rows AND non-zero K/V length.
        _kv_nonzero = arith.cmpi(
            arith.CmpIPredicate.sgt,
            actual_kv_len,
            arith.unwrap(arith.constant(0, type=T.i32)),
        )
        _wg_valid_compute = arith.andi(_wg_valid, _kv_nonzero)
        _if_wg = scf.IfOp(arith.unwrap(_wg_valid_compute))
        with _if_then(_if_wg):
            # Q resource descriptor with OOB protection (THD: batch is implicit in
            # q_start_tok)
            _q_tok = arith.addi(
                q_start_tok,
                arith.muli(
                    arith.unwrap(bx),
                    arith.unwrap(arith.constant(128, type=T.i32)),
                ),
            )
            q_offset = arith.addi(
                arith.muli(_q_tok, stride_q_seq),
                arith.muli(arith.unwrap(by), stride_q_head),
            )
            _q_num_bytes = arith.muli(q_end_tok, stride_q_seq)
            q_rsrc = buffer_ops.create_buffer_resource(
                ptr_Q, num_records_bytes=_q_num_bytes
            )

            # Q load → q_frags[4 bank][2 frag] v16bf16
            # Pass q_offset (byte offset for this workgroup's Q tile) so WG>0 reads
            # correct rows
            q_frags_raw = _phase4_q_load_flydsl(
                lane_id,
                arith.unwrap(q_rsrc),
                stride_q_seq,
                wave_id,
                q_tile_offset_bytes=q_offset,
            )
            rocdl.sched_barrier(0)

            # Bridge prologue q_frags[4 bank][frags_per_bank] → core loop q_tiles[4
            # msb][Q_WMMA_PER_MSB]
            # q_msb only takes values {0, 2} (k//Q_WMMA_PER_MSB = 0 for all k <
            # total_k_iters).
            # Each q_msb merges TWO adjacent banks (lo+hi K-col halves):
            #   q_msb=0 (sp_msb∈{0,1}): banks[0] (K cols 0..QK_HDIM/2) + banks[1] (K
            #   cols QK_HDIM/2..QK_HDIM)
            #   q_msb=2 (sp_msb∈{2,3}): banks[2] + banks[3]
            # frags_per_bank = len(q_frags_raw[0]) = Q_WMMA_PER_MSB//2 (3 for 192-dim)
            _frags_per_bank = len(q_frags_raw[0])  # e.g. 3 for 192-dim, 2 for 128-dim
            q_frags = [[None] * Q_WMMA_PER_MSB for _ in range(NUM_MSB)]
            _pad = [None] * (Q_WMMA_PER_MSB - 2 * _frags_per_bank)
            q_frags[0] = q_frags_raw[0] + q_frags_raw[1] + _pad
            q_frags[2] = q_frags_raw[2] + q_frags_raw[3] + _pad

            # Head index (for GQA)
            head_index = _phase5_head_index_div_flydsl(by, gqa)

            # K/V base offsets
            # THD: K/V batch offset = k_start_tok * stride_{k,v}_seq
            k_offset = arith.addi(
                arith.muli(k_start_tok, stride_k_seq),
                arith.muli(arith.unwrap(head_index), stride_k_head),
            )
            v_offset = arith.addi(
                arith.muli(k_start_tok, stride_v_seq),
                arith.muli(arith.unwrap(head_index), stride_v_head),
            )

            # SmemAllocator bases → i32 LDS addresses
            k_a_base_i32 = _extract_lds_base_i32(_lds_alloc_k_a.get_base())
            k_b_base_i32 = _extract_lds_base_i32(_lds_alloc_k_b.get_base())
            v_a_base_i32 = _extract_lds_base_i32(_lds_alloc_v_a.get_base())
            v_b_base_i32 = _extract_lds_base_i32(_lds_alloc_v_b.get_base())

            # K/V LDS address generation — from SmemAllocator bases
            # kv_lds_addrs_a[0..3]=K_a, [4..7]=V_a  (ping / blk=0)
            # kv_lds_addrs_b[0..3]=K_b, [4..7]=V_b  (pong / blk=1)
            rocdl.sched_barrier(0)
            kv_lds_addrs_a = _build_kv_lds_addrs(lane_id, k_a_base_i32, v_a_base_i32)
            kv_lds_addrs_b = _build_kv_lds_addrs(lane_id, k_b_base_i32, v_b_base_i32)

            stride_k_32 = arith.muli(arith.constant(32, type=T.i32), stride_k_seq)
            stride_v_32 = arith.muli(arith.constant(32, type=T.i32), stride_v_seq)

            # SGPR state: softmax scale
            log2e_val = arith.constant(1.4426950408889634, type=T.f32)
            scale = arith.mulf(log2e_val, scalar_f)
            idx_0 = arith.unwrap(arith.constant(0, type=T.i32))
            idx_1 = arith.unwrap(arith.constant(1, type=T.i32))
            v2f32_ty = ty["v2f32"]
            pair_undef = llvm_dialect.mlir_undef(v2f32_ty)
            pair_v = llvm_dialect.insertelement(pair_undef, scale, idx_0)
            scale_pair = llvm_dialect.insertelement(pair_v, scale, idx_1)

            sgpr_state = {
                "s_log2e_scl": scale,
                "s_log2e_scl_pair": scale_pair,
            }

            # ================================================================
            # SECTION 2: Prologue — Tile 0 QK + Partial Softmax (no PV)
            # ================================================================
            #
            #   1. TDM K(tile 0, 4 SUs) → wait → QK_pure (64 WMMAs)
            #   2. TDM V(tile 0, 4 SUs) — overlapped with softmax
            #   3. sp_tiles → sp_pairs → softmax PART0+PART1 only (no PART2)
            #   4. TDM K(tile 1) — prefetch K only, V(tile 0) stays in LDS
            #   5. Wait K(tile 1) → LDS K(su=0) — preload for core_loop entry
            #
            # No PV in prologue. PART2 + PV run in core_loop iterations.

            zero_f32 = arith.unwrap(arith.constant(0.0, type=T.f32))
            neg_inf = arith.unwrap(arith.constant(float("-inf"), type=T.f32))
            zero_v8f32 = arith.unwrap(arith.constant_vector(0.0, T.vec(8, T.f32)))

            # -- 2a: Load K(tile 0) → K_a (ping buffer) --
            rocdl.sched_barrier(0)
            _tdm_load_k_only(
                ptr_K,
                k_offset,
                stride_k_seq,
                stride_k_32,
                wave_id,
                k_a_base_i32,
                oob_dg1_list=_k_oob_dg1,
            )
            rocdl.sched_barrier(0)

            # -- 2b: QK_pure for all 4 SUs --
            all_su_sp_tiles = []
            for su in fx.range_constexpr(CNT_SU):
                kv_tiles_su = _load_k_su_from_lds(ty, kv_lds_addrs_a, 0, su)
                fresh_sp = []
                for msb in fx.range_constexpr(NUM_MSB):
                    fresh_sp.append([zero_v8f32])
                fresh_sp = _qk_pure_su(ty, 0, su, q_frags, kv_tiles_su, fresh_sp)
                all_su_sp_tiles.append(fresh_sp)

            # -- 2c: Load V(tile 0) → V_a (ping buffer) --
            _tdm_load_v_only(
                ptr_V,
                v_offset,
                stride_v_seq,
                stride_v_32,
                wave_id,
                v_a_base_i32,
                oob_dg1_list=_v_oob_dg1,
            )

            # -- 2d': causal mask on prologue tile (n_start=0) --
            # Bottom-right aligned: shift n_start by -(sk - sq) so the diagonal
            # is anchored at the bottom-right corner of the QK matrix.
            _causal_offset = arith.subi(actual_kv_len, actual_q_len)
            if const_expr(IS_CAUSAL):
                _pro_causal_n = arith.subi(
                    arith.unwrap(arith.constant(0, type=T.i32)), _causal_offset
                )
                _apply_causal_mask(all_su_sp_tiles, _pro_causal_n)

            # -- 2d'': KV OOB mask on prologue tile --
            _apply_kv_oob_mask(all_su_sp_tiles, actual_kv_len)

            # -- 2d: sp_tiles → sp_pairs --
            sp_pairs_all_pro = _sp_tiles_to_sp_pairs(all_su_sp_tiles)

            # -- 2e: Softmax PART0+PART1 only (no PART2) --
            # Pin each MSB's scalar state to its own VGPR bank for the
            # "全-bankN" (0x00/0x55/0xAA/0xFF) MSB allocation pattern.
            softmax_state_pro = {
                "old_max": [set_vgpr_bank(neg_inf, m) for m in range(NUM_MSB)],
                "local_max": [set_vgpr_bank(neg_inf, m) for m in range(NUM_MSB)],
                "delta": [set_vgpr_bank(zero_f32, m) for m in range(NUM_MSB)],
                "exp_delta": [None] * NUM_MSB,
                "cur_max_log2e": [None] * NUM_MSB,
                "cur_max_log2e_1": [None] * NUM_MSB,
                "cur_max_log2e_scalar": [None] * NUM_MSB,
                "cur_max_log2e_dup": [None] * NUM_MSB,
                "vgpr_log2e_scl_pair": [None] * NUM_MSB,
                "exp_delta_dup": [None] * NUM_MSB,
                "row_sums": [set_vgpr_bank(zero_f32, m) for m in range(NUM_MSB)],
                "p_bf16": [[], [], [], []],
                "sp_pairs_prev": sp_pairs_all_pro,
            }
            _softmax_part01_only(ty, 0, sp_pairs_all_pro, softmax_state_pro, sgpr_state)

            # -- 2e': PART2 first half for tile 0 --
            # Runs ops 0..PART2_SPLIT-1: setup(7)+pkfma(16)+pair_exp(8) = 31 ops/MSB.
            # 4 MSBs × 8 pair_exp = 32 pair_exp + 4 exp_delta (unavoidable pipeline
            # overhead).
            # sp_pairs_all_pro[m][0..15] are partially modified (pkfma+8-exp applied).
            # pro_exp_delta seeds the O-rescale iter_arg for the first core_loop
            # iteration.
            _pro_part2_ops = _build_all_softmax_part2_ops(
                ty, 0, sp_pairs_all_pro, softmax_state_pro, sgpr_state
            )
            for _m in fx.range_constexpr(NUM_MSB):
                for _op in _pro_part2_ops[_m][:PART2_SPLIT]:
                    _op()

            # -- 2f: Prefetch K(tile 1) → K_b if available --
            tile_n_const = arith.unwrap(arith.constant(TILE_N, type=T.i32))
            _kv_tiles_avail = arith.divui(
                arith.addi(
                    actual_kv_len, arith.unwrap(arith.constant(TILE_N - 1, type=T.i32))
                ),
                tile_n_const,
            )
            if const_expr(IS_CAUSAL):
                # Causal (bottom-right aligned): the last Q row (m_start + TILE_N - 1)
                # attends to the last K row (actual_kv_len - 1).  The first Q row in
                # this tile attends to K starting at position
                #   (actual_kv_len - actual_q_len) + m_start.
                # So the number of KV tiles needed is:
                _sk_sq_diff = arith.subi(actual_kv_len, actual_q_len)
                _sk_sq_tiles = arith.divui(
                    arith.addi(
                        _sk_sq_diff,
                        arith.unwrap(arith.constant(TILE_N - 1, type=T.i32)),
                    ),
                    tile_n_const,
                )
                _bx_plus_1 = arith.addi(
                    arith.unwrap(bx), arith.unwrap(arith.constant(1, type=T.i32))
                )
                _causal_tiles = arith.addi(_bx_plus_1, _sk_sq_tiles)
                num_tiles = arith.minui(_causal_tiles, _kv_tiles_avail)
            else:
                # Non-causal: iterate over all KV tiles.
                num_tiles = _kv_tiles_avail
            num_tiles_idx = arith.index_cast(T.index, num_tiles)
            # Loop runs N-2 iterations (tiles 1..N-2); endtile handled in epilogue.
            _one_i32_loop = arith.unwrap(arith.constant(1, type=T.i32))
            num_tiles_minus1 = arith.subi(num_tiles, _one_i32_loop)
            num_tiles_minus1_idx = arith.index_cast(T.index, num_tiles_minus1)

            # Load K(tile 1) → K_b for core_loop first iteration.
            # For num_tiles=1 (128x128), K_b is never used (loop runs 0 iterations).
            rocdl.sched_barrier(0)
            _k_tile1_stride = arith.muli(tile_n_const, stride_k_seq)
            _k_tile1_offset = arith.addi(arith.unwrap(k_offset), _k_tile1_stride)
            _kv_remain_t1 = arith.subi(
                actual_kv_len, arith.unwrap(arith.constant(TILE_N, type=T.i32))
            )
            _k_tile1_oob_dg1 = [
                _make_kv_dg1_with_oob(
                    _K_CFG_OOB,
                    QK_HDIM,
                    8,
                    _stride_k_elems_oob,
                    _per_warp_oob_dim1(
                        arith.subi(
                            _kv_remain_t1,
                            arith.unwrap(arith.constant(_su * 32, type=T.i32)),
                        ),
                        wave_id,
                        8,
                    ),
                    dim0_stride=200,
                )
                for _su in range(CNT_SU)
            ]
            _tdm_load_k_only(
                ptr_K,
                _k_tile1_offset,
                stride_k_seq,
                stride_k_32,
                wave_id,
                k_b_base_i32,
                oob_dg1_list=_k_tile1_oob_dg1,
            )
            rocdl.sched_barrier(0)

            # -- 2g: Load K(su=0) from K_b for core_loop entry --
            kv_tiles_init = _load_initial_kv_tiles(ty, kv_lds_addrs_b, blk=0, su=0)

            # Prologue results: PART0+PART1+PART2 first half done.
            # old_max = local_max (set by PART2 setup op0); row_sums rescaled
            # (×exp_delta).
            pro_old_max = [
                softmax_state_pro["old_max"][m] for m in fx.range_constexpr(NUM_MSB)
            ]
            pro_row_sums = [
                softmax_state_pro["row_sums"][m] for m in fx.range_constexpr(NUM_MSB)
            ]
            pro_local_max = [
                softmax_state_pro["local_max"][m] for m in fx.range_constexpr(NUM_MSB)
            ]
            pro_delta = [
                softmax_state_pro["delta"][m] for m in fx.range_constexpr(NUM_MSB)
            ]

            # Partial sp_pairs after first half: yield as separate lo+hi f32 scalars.
            # Prologue runs pair_exp sequentially (correct v2f32), so extractelement is
            # safe here.
            pro_partial_sp_lo_flat = []
            pro_partial_sp_hi_flat = []
            for _m in fx.range_constexpr(NUM_MSB):
                for _i in fx.range_constexpr(N_SP_PAIRS):
                    _idx0 = arith.unwrap(arith.constant(0, type=T.i32))
                    pro_partial_sp_lo_flat.append(
                        llvm_dialect.extractelement(
                            sp_pairs_all_pro[_m][_i],
                            _idx0,
                        )
                    )
                    _idx1 = arith.unwrap(arith.constant(1, type=T.i32))
                    pro_partial_sp_hi_flat.append(
                        llvm_dialect.extractelement(
                            sp_pairs_all_pro[_m][_i],
                            _idx1,
                        )
                    )
            pro_partial_sp_flat = pro_partial_sp_lo_flat + pro_partial_sp_hi_flat
            # exp_delta from PART2 setup (used by first core_loop iteration's O
            # rescale).
            pro_exp_delta = [
                softmax_state_pro["exp_delta"][_m] for _m in fx.range_constexpr(NUM_MSB)
            ]

            # Flatten kv_tiles_init[4 msb][2] → 8 v16bf16
            kv_flat_init = []
            for msb in fx.range_constexpr(NUM_MSB):
                for k in fx.range_constexpr(N_WMMA_K_TILES):
                    kv_flat_init.append(kv_tiles_init[msb][k])

            # Flatten prologue sp_tiles per SU [CNT_SU][NUM_MSB][1] → 16 v8f32
            sp_flat_init = []
            for su in fx.range_constexpr(CNT_SU):
                for msb in fx.range_constexpr(NUM_MSB):
                    sp_flat_init.append(all_su_sp_tiles[su][msb][0])

            # ================================================================
            # SECTION 3: Dynamic KV Loop — scf.for_ from tile 1
            # ================================================================
            #
            # Pipeline layout:
            #   - K in LDS is one tile AHEAD of V in LDS
            #   - Prologue loaded K(tile 0)+V(tile 0), did QK, prefetched K(tile 1)
            #   - Iteration i: GEMM1 on K(tile i+1), GEMM2 on V(tile i)
            #   - After core_loop: TDM V(tile i+1) + K(tile i+2)
            #
            # O tiles start as zeros (no PV in prologue).

            def _core_loop(
                ty,
                memload,
                q_tiles,  # [4 msb][Q_WMMA_PER_MSB] v16bf16 — Q data
                kv_tiles,  # [4 msb][2] v16bf16 — paired K tiles for WMMA
                sp_tiles,  # [4 msb][1] v8f32 — QK accumulators
                # [4 d_msb][N_PV_WMMA_N] v8f32 -- O accum
                o_tiles,
                # [8] i32 -- [K_cur[0:4] + V_cur[4:8]]
                kv_lds_addrs,
                tdm_state,  # TDM SGPR descriptors
                # Softmax state (old_max, local_max, etc.)
                softmax_state,
                sgpr_state,  # SGPR refs (s_log2e_scl)
                gemm2=True,  # run GEMM2 stages?
                # V global offset for TDM (cur -> next V)
                tdm_v_offset=None,
                # K global offset for TDM (next -> next K)
                tdm_k_offset=None,
                tdm_k_target=None,  # i32 LDS base K
                tdm_v_target=None,  # i32 LDS base V
                # [8] [K_next + V_next] for K reload
                kv_lds_addrs_next=None,
                # False=main(GEMM1 K) True=epi(GEMM1 V)
                gemm1_tdm_is_v=False,
                # i32 n_start for causal mask (None=skip)
                causal_n_start=None,
                # OOB dg1 list for GEMM1 V TDM
                endtile_v_dg1=None,
                # raw i32: valid K cols (None = all 128)
                kv_oob_cols=None,
                # list[CNT_SU] v8i32 OOB dg1 K prefetch
                loop_k_oob_dg1=None,
                # list[CNT_SU] v8i32 OOB dg1 V load
                loop_v_oob_dg1=None,
            ):
                """Full core loop: GEMM1 (QK) + softmax + GEMM2 (PV).

                tile_n=128: single pass with 4 SUs (no pi/half loops).
                4 GEMM1 stages (64 QK WMMAs) + 4 GEMM2 stages (64 PV WMMAs) = 128 total.

                Pipeline per call:
                  GEMM1: QK on current K (in LDS, from kv_lds_addrs[0:4])
                  PART2: run on sp_pairs_prev (from previous tile)
                  PART0+PART1: run on current sp_tiles
                  GEMM2: PV using P tiles × V (in LDS, from kv_lds_addrs[4:8])
                  TDM: GEMM1 loads K(i+1)->K_next; GEMM2 loads V(i)->V_next (main loop)
                       GEMM1 loads V(endtile)->V_next (epilogue, gemm1_tdm_is_v=True)

                kv_lds_addrs: [K_cur[0:4] + V_cur[4:8]] — built from mixed allocator
                bases via _build_kv_lds_addrs(lane_id, k_cur_base, v_cur_base).
                kv_lds_addrs_next: same structure for next ping-pong buffer, used for
                GEMM2 stage 3 K preload (K_next already filled by GEMM1 TDM).

                Returns: (sp_tiles, kv_tiles, o_tiles, su_sp_tiles_list).
                """
                _atom_s_wait_dscnt(LDS_INST_COUNT // 2)  # s_wait_dscnt 0x8

                v_tiles_out = None
                blk = 0

                # ================================================================
                # GEMM1 (QK): 4 stages (SU 0..3)
                # Interleave PART2 on sp_pairs_prev during GEMM1 stages.
                # ================================================================
                sp_pairs_all = softmax_state.get("sp_pairs_prev", None)
                if const_expr(sp_pairs_all is None):
                    sp_pairs_all = [[None] * N_SP_PAIRS for _ in range(NUM_MSB)]

                # DBG: print sp_pairs_prev pairs 0-3 of MSBs 2,3 at core_loop entry
                # if const_expr(sp_pairs_all[2][0] is not None):

                softmax_ops_by_msb = _build_all_softmax_part2_ops(
                    ty, 0, sp_pairs_all, softmax_state, sgpr_state
                )
                # Second half starts at PART2_SPLIT; first half ran in previous GEMM2.
                softmax_idx_by_msb = [PART2_SPLIT] * NUM_MSB

                # Pre-run setup ops 0..PART2_SETUP_A-2 (skip last = row_sums rescale,
                # already applied).
                # Sets cur_max_log2e (needed by pkfma), exp_delta, old_max, etc.
                # Row_sums is NOT re-rescaled here — ia_row_sums already carries the
                # correctly-rescaled value from the previous GEMM2's PART2 first half.
                for _m in fx.range_constexpr(NUM_MSB):
                    # ops 0..5, skip op 6
                    for _i in fx.range_constexpr(PART2_SETUP_A - 1):
                        softmax_ops_by_msb[_m][_i]()

                # Build TDM descriptors for GEMM1 stages 0-1.
                # Main-loop (gemm1_tdm_is_v=False): GEMM1 loads K(i+1) -> K_next
                # Epilogue (gemm1_tdm_is_v=True):  GEMM1 loads V(endtile) -> V_next
                has_tdm_k_g1 = (not gemm1_tdm_is_v) and (tdm_k_offset is not None)
                has_tdm_v_g1 = gemm1_tdm_is_v and (tdm_v_offset is not None)

                if has_tdm_k_g1:
                    i64 = ir.IntegerType.get_signless(64)
                    _K_CFG = (1 << 16) | _K_TDM_CONFIG
                    _stride_k_elems = arith.unwrap(
                        arith.shrui(stride_k_seq, arith.constant(1, type=T.i32))
                    )
                    if const_expr(loop_k_oob_dg1 is not None):
                        k_dg1 = loop_k_oob_dg1
                    else:
                        k_dg1 = vector.from_elements(
                            T.vec(8, T.i32),
                            [
                                arith.constant(_K_CFG, type=T.i32),
                                arith.constant(QK_HDIM << 16, type=T.i32),
                                arith.constant(8 << 16, type=T.i32),
                                arith.constant(200 << 16, type=T.i32),
                                arith.constant(8, type=T.i32),
                                _stride_k_elems,
                                arith.constant(0, type=T.i32),
                                arith.constant(0, type=T.i32),
                            ],
                        )
                    k_addr = _compute_k_global_addr(
                        ptr_K,
                        tdm_k_offset,
                        wave_id,
                        arith.unwrap(
                            arith.muli(arith.constant(8, type=T.i32), stride_k_seq)
                        ),
                    )
                    k_stride_adv = arith.extsi(i64, arith.unwrap(stride_k_32))
                    _wid_k = arith.unwrap(wave_id)
                    _k_warp_off = arith.muli(
                        _wid_k,
                        arith.unwrap(arith.constant(8 * K_ROW_BYTES, type=T.i32)),
                    )
                    _k_lds_base = arith.addi(tdm_k_target, _k_warp_off)
                    k_descs = _build_tdm_descs(
                        k_dg1,
                        k_addr,
                        k_stride_adv,
                        _k_lds_base,
                        LDS_K_SU_P_SIZE,
                        CNT_SU,
                    )
                    tdm_state["k_descs"] = k_descs
                    tdm_state["k_desc_idx"] = 0

                if has_tdm_v_g1:
                    i64 = ir.IntegerType.get_signless(64)
                    _V_CFG = (1 << 16) | _V_TDM_CONFIG
                    _stride_v_elems = arith.unwrap(
                        arith.shrui(stride_v_seq, arith.constant(1, type=T.i32))
                    )
                    if const_expr(endtile_v_dg1 is not None):
                        v_dg1 = endtile_v_dg1
                    else:
                        v_dg1 = vector.from_elements(
                            T.vec(8, T.i32),
                            [
                                arith.constant(_V_CFG, type=T.i32),
                                arith.constant(128 << 16, type=T.i32),
                                arith.constant(8 << 16, type=T.i32),
                                arith.constant(128 << 16, type=T.i32),
                                arith.constant(8, type=T.i32),
                                _stride_v_elems,
                                arith.constant(0, type=T.i32),
                                arith.constant(0, type=T.i32),
                            ],
                        )
                    v_addr = _compute_v_global_addr(
                        ptr_V,
                        tdm_v_offset,
                        wave_id,
                        arith.unwrap(
                            arith.muli(arith.constant(8, type=T.i32), stride_v_seq)
                        ),
                    )
                    v_stride_adv = arith.extsi(i64, arith.unwrap(stride_v_32))
                    _wid_v = arith.unwrap(wave_id)
                    _v_warp_off = arith.muli(
                        _wid_v,
                        arith.unwrap(arith.constant(8 * V_ROW_BYTES, type=T.i32)),
                    )
                    _v_lds_base = arith.addi(tdm_v_target, _v_warp_off)
                    v_descs = _build_tdm_descs(
                        v_dg1,
                        v_addr,
                        v_stride_adv,
                        _v_lds_base,
                        LDS_V_SU_P_SIZE,
                        CNT_SU,
                    )
                    tdm_state["v_descs"] = v_descs
                    tdm_state["v_desc_idx"] = 0

                _g1_tdm_type = (
                    KV_K if has_tdm_k_g1 else KV_V if has_tdm_v_g1 else KV_NONE
                )
                stage_configs = [
                    (0, _g1_tdm_type, KV_K, blk, 1),
                    (1, _g1_tdm_type, KV_K, blk, 2),
                    (2, KV_NONE, KV_K, blk, 3),
                    (3, KV_NONE, KV_V, blk, 0),
                ]

                su_sp_tiles_list = []

                # O rescale: all 4 MSBs across GEMM1 stages 0-3.
                # Each stage dispatches N_PV_WMMA_N closures (1 tile per MSB, 4 MSBs
                # total
                # = N_PV_WMMA_N * NUM_MSB closures per stage), running at the last N
                # WMMAs.
                # This removes O_resc from GEMM2 stage 0 entirely — exp+O_resc+cvt
                # interleave throughout GEMM1 (QK) stages.
                #
                # Stage assignment: stage s handles tile n=s for all 4 MSBs.
                #   stage 0 → n=0 (all MSBs), stage 1 → n=1, stage 2 → n=2, stage 3 →
                #   n=3
                # Build broadcast v8f32 for each MSB's exp_delta
                _ed_v8 = []
                for _dm in fx.range_constexpr(NUM_MSB):
                    _edv = llvm_dialect.mlir_undef(ty["v8f32"])
                    for _ii in fx.range_constexpr(8):
                        _edv = llvm_dialect.insertelement(
                            _edv,
                            ia_exp_delta[_dm],
                            arith.unwrap(arith.constant(_ii, type=T.i32)),
                        )
                    _ed_v8.append(_edv)
                # _o_rescale_by_stage[s] = list of closures for GEMM1 stage s
                _o_rescale_by_stage = []
                # s = tile index = stage index
                for _s in fx.range_constexpr(N_PV_WMMA_N):
                    _stage_closures = []
                    for _dm in fx.range_constexpr(NUM_MSB):  # 4 closures per stage

                        def _mk_rescale(dm=_dm, nn=_s, ev8=_ed_v8[_dm]):
                            def _op():
                                o_tiles[dm][nn] = arith.mulf(o_tiles[dm][nn], ev8)

                            return _op

                        _stage_closures.append(_mk_rescale())
                    _o_rescale_by_stage.append(_stage_closures)

                for stage_idx, (g_su, t_type, l_type, l_blk, l_su) in enumerate(
                    stage_configs
                ):

                    _n_lds = N_LDS_V_PER_MSB if l_type == KV_V else N_LDS_PER_MSB
                    kv_tiles_next_raw = [[None] * _n_lds for _ in range(NUM_MSB)]

                    softmax_stage = (stage_idx + 4) % ALU_STAGES
                    budget_per_msb = ALU_PER_STAGE[softmax_stage] // NUM_MSB
                    # MSBs 0,1: dispatch ops during GEMM1 stages (register pressure
                    # manageable).
                    # MSBs 2,3: budget=0 → all ops 32+ run in sequential inter-GEMM
                    # flush instead,
                    #           avoiding WMMA-induced register collision for banks 2,3.
                    softmax_budget = [budget_per_msb, budget_per_msb, 0, 0]

                    is_barrier_stage = stage_idx == 2 and (has_tdm_k_g1 or has_tdm_v_g1)

                    # Distribute O_resc across all 4 GEMM1 stages (1
                    # tile/MSB per stage).
                    stage_o_rescale = _o_rescale_by_stage[stage_idx]

                    sp_tiles, kv_tiles_next_raw = _cl_su_v3_stage(
                        ty,
                        stage_idx,
                        blk,
                        g_su,
                        t_type,
                        blk,
                        g_su,
                        l_type,
                        l_blk,
                        l_su,
                        q_tiles,
                        kv_tiles,
                        sp_tiles,
                        kv_lds_addrs,
                        kv_tiles_next_raw,
                        softmax_ops_by_msb,
                        softmax_idx_by_msb,
                        softmax_budget,
                        tdm_state,
                        tdm_barrier=is_barrier_stage,
                        o_rescale_ops=stage_o_rescale,
                    )

                    su_sp_tiles_list.append(
                        [[sp_tiles[msb][0]] for msb in range(NUM_MSB)]
                    )

                    if const_expr(l_type == KV_K):
                        kv_tiles = _pair_k_tiles_for_wmma(kv_tiles_next_raw, ty)
                    else:
                        v_tiles_out = kv_tiles_next_raw

                if const_expr(not gemm2):
                    return sp_tiles, kv_tiles, o_tiles, su_sp_tiles_list

                # ================================================================
                # Between GEMM1 and GEMM2: complete softmax pipeline
                # ================================================================

                # 1. Flush any remaining PART2 second-half ops
                for msb in fx.range_constexpr(NUM_MSB):
                    for op in softmax_ops_by_msb[msb][softmax_idx_by_msb[msb] :]:
                        op()

                # DBG: row_sums after GEMM1 second-half (sum_accum done), before GEMM2
                # PART0+1 rescale

                # O_resc moved entirely to GEMM1 stages — no O_resc in GEMM2.
                _o_rescale_exp_delta = None

                # ================================================================
                # Causal mask on current QK tile (before sp_pairs conversion)
                # ================================================================
                if const_expr(causal_n_start is not None):
                    _apply_causal_mask(su_sp_tiles_list, causal_n_start)
                if const_expr(kv_oob_cols is not None):
                    _apply_kv_oob_mask(su_sp_tiles_list, kv_oob_cols)

                # 2. Build sp_pairs for current tile (all 4 SUs).
                sp_pairs_current = _sp_tiles_to_sp_pairs(su_sp_tiles_list)

                # 3. Build all PART0+PART1+PART2 closures for current tile.
                # g2_sp_lo/hi_cache: f32 scalar caches populated by EXP token dispatch,
                # bypassing unreliable v2f32 insertelement under WMMA register pressure.
                (
                    g2_ops_by_rid,
                    _,
                    g2_sp_lo_cache,
                    g2_sp_hi_cache,
                ) = _build_all_softmax_gemm2_ops(
                    ty,
                    blk,
                    sp_pairs_current,
                    softmax_state,
                    sgpr_state,
                )
                # P0/P1 dispatched via schedule tokens (P0 tokens in stages 0-1).
                # Starts from op 0; token dispatch happens AFTER each WMMA's
                # sched_barrier(0)
                # so LLVM places ops between WMMAs rather than clustering before the
                # loop.
                # PART2 (P2/EXP) starts from stage 2 (after PART1 updates local_max).
                g2_rid_idx = [0] * RLTS_LEN

                # 4. Build P tiles from p_bf16 (produced by PART2 second half on prev
                # tile)
                p_tiles_computed = _build_p_tiles_from_softmax(ty, softmax_state)

                # ================================================================
                # GEMM2 (PV): 4 stages (SU 0..3)
                # PART0 distributed in two chunks so LLVM interleaves them with WMMAs:
                #   Chunk A (ops 0..10): emitted above → LLVM places in G2-su0 WMMA
                #   slots
                #   Chunk B (ops 11..21) + PART1: emitted between G2-su0 and G2-su1
                #                                  → LLVM places in G2-su1 WMMA slots
                # ================================================================

                v_tiles_paired = _pair_v_tiles_for_wmma(v_tiles_out, ty)

                # Build V TDM descriptors for GEMM2 stages 0-1.
                # Main-loop mode: GEMM2 loads V(i) -> V_next
                has_tdm_v_g2 = (not gemm1_tdm_is_v) and (tdm_v_offset is not None)
                if has_tdm_v_g2:
                    i64 = ir.IntegerType.get_signless(64)
                    _V_CFG = (1 << 16) | _V_TDM_CONFIG
                    _stride_v_elems = arith.unwrap(
                        arith.shrui(stride_v_seq, arith.constant(1, type=T.i32))
                    )
                    if const_expr(loop_v_oob_dg1 is not None):
                        v_dg1 = loop_v_oob_dg1
                    else:
                        v_dg1 = vector.from_elements(
                            T.vec(8, T.i32),
                            [
                                arith.constant(_V_CFG, type=T.i32),
                                arith.constant(128 << 16, type=T.i32),
                                arith.constant(8 << 16, type=T.i32),
                                arith.constant(128 << 16, type=T.i32),
                                arith.constant(8, type=T.i32),
                                _stride_v_elems,
                                arith.constant(0, type=T.i32),
                                arith.constant(0, type=T.i32),
                            ],
                        )
                    v_addr = _compute_v_global_addr(
                        ptr_V,
                        tdm_v_offset,
                        wave_id,
                        arith.unwrap(
                            arith.muli(arith.constant(8, type=T.i32), stride_v_seq)
                        ),
                    )
                    v_stride_adv = arith.extsi(i64, arith.unwrap(stride_v_32))
                    _wid_v = arith.unwrap(wave_id)
                    _v_warp_off = arith.muli(
                        _wid_v,
                        arith.unwrap(arith.constant(8 * V_ROW_BYTES, type=T.i32)),
                    )
                    _v_lds_base = arith.addi(tdm_v_target, _v_warp_off)
                    v_descs = _build_tdm_descs(
                        v_dg1,
                        v_addr,
                        v_stride_adv,
                        _v_lds_base,
                        LDS_V_SU_P_SIZE,
                        CNT_SU,
                    )
                    tdm_state["v_descs"] = v_descs
                    tdm_state["v_desc_idx"] = 0

                # stage tuple: (gemm_su, lds_type, lds_blk, lds_su, tdm_type,
                # tdm_barrier)
                g2_stage_configs = [
                    (0, KV_V, blk, 1, KV_V if has_tdm_v_g2 else KV_NONE, False),
                    (1, KV_V, blk, 2, KV_V if has_tdm_v_g2 else KV_NONE, False),
                    (2, KV_V, blk, 3, KV_NONE, has_tdm_v_g2),
                    (3, KV_K, blk, 0, KV_NONE, False),
                ]

                for stage_idx, (
                    g_su,
                    l_type,
                    l_blk,
                    l_su,
                    t_type,
                    barrier,
                ) in enumerate(g2_stage_configs):

                    p_tiles_su = p_tiles_computed[g_su]

                    _n_lds = N_LDS_V_PER_MSB if l_type == KV_V else N_LDS_PER_MSB
                    kv_tiles_next_raw = [[None] * _n_lds for _ in range(NUM_MSB)]

                    if const_expr(l_type == KV_K):
                        g2_addrs = (
                            kv_lds_addrs_next
                            if kv_lds_addrs_next is not None
                            else kv_lds_addrs
                        )
                    else:
                        g2_addrs = kv_lds_addrs

                    o_tiles, kv_tiles_next_raw = _cl_su_v3_stage_gemm2(
                        ty,
                        stage_idx,
                        blk,
                        g_su,
                        l_type,
                        l_blk,
                        l_su,
                        v_tiles_paired,
                        p_tiles_su,
                        o_tiles,
                        g2_addrs,
                        kv_tiles_next_raw,
                        g2_ops_by_rid,
                        g2_rid_idx,
                        tdm_state=tdm_state,
                        tdm_type=t_type,
                        tdm_barrier=barrier,
                        o_rescale_exp_delta=(
                            _o_rescale_exp_delta if stage_idx == 0 else None
                        ),
                    )

                    if const_expr(l_type == KV_V):
                        v_tiles_paired = _pair_v_tiles_for_wmma(kv_tiles_next_raw, ty)
                    else:
                        kv_tiles = _pair_k_tiles_for_wmma(kv_tiles_next_raw, ty)

                # DBG: GEMM2 (O acc) — lane 0, wave 0

                rocdl.sched_barrier(0)

                # Build partial_sp lo+hi flat lists for yield.
                # For pairs 0-3: use f32 cache from EXP token dispatch (correct
                # scalars).
                # For pairs 4-15: extractelement from sp_pairs_current (pkfma, to be
                # exp'd later).
                partial_sp_lo_out = []
                partial_sp_hi_out = []
                for _psm in fx.range_constexpr(NUM_MSB):
                    for _psi in fx.range_constexpr(N_SP_PAIRS):
                        _lo_c = g2_sp_lo_cache[_psm][_psi]
                        _hi_c = g2_sp_hi_cache[_psm][_psi]
                        if const_expr(_lo_c is None):
                            _c0 = arith.unwrap(arith.constant(0, type=T.i32))
                            _lo_c = llvm_dialect.extractelement(
                                sp_pairs_current[_psm][_psi],
                                _c0,
                            )
                        if const_expr(_hi_c is None):
                            _c1 = arith.unwrap(arith.constant(1, type=T.i32))
                            _hi_c = llvm_dialect.extractelement(
                                sp_pairs_current[_psm][_psi],
                                _c1,
                            )
                        partial_sp_lo_out.append(_lo_c)
                        partial_sp_hi_out.append(_hi_c)

                partial_ed_out = [
                    softmax_state["exp_delta"][_m] for _m in fx.range_constexpr(NUM_MSB)
                ]
                return (
                    sp_tiles,
                    kv_tiles,
                    o_tiles,
                    su_sp_tiles_list,
                    partial_sp_lo_out,
                    partial_sp_hi_out,
                    partial_ed_out,
                )

            # ================================================================
            # Init iter_args layout (dynamic — offsets depend on N_WMMA_K_TILES):
            #   [0..15]            o_tiles[4][4] v8f32                  = 16 values
            #   [16..19]           old_max[4] f32                        = 4 values
            #   [20..23]           row_sums[4] f32                       = 4 values
            #   [24..24+KV-1]      kv_tiles[NUM_MSB][N_WMMA_K_TILES]    = KV values
            #   [24+KV..+3]        local_max[4] f32                      = 4 values
            #   [24+KV+4..+7]      delta[4] f32                          = 4 values
            #   [24+KV+8..+23]     sp_tiles[CNT_SU*NUM_MSB] v8f32       = 16 values
            #   [24+KV+24..+27]    ping-pong bases (i32)                 = 4 values
            #   [24+KV+28..+91]    partial_sp_pairs[4][16] v2f32        = 64 values
            #                      (PART2 first half output, double-buffered pipeline)
            #   [24+KV+92..+95]    exp_delta[4] f32                     = 4 values
            # ================================================================
            _KV_SIZE = NUM_MSB * N_WMMA_K_TILES  # 12 for 192-dim, 8 for 128-dim
            _OFF_LOCAL_MAX = 24 + _KV_SIZE  # 36 for 192-dim
            _OFF_DELTA = _OFF_LOCAL_MAX + NUM_MSB
            _OFF_SP = _OFF_DELTA + NUM_MSB
            _OFF_PP = _OFF_SP + CNT_SU * NUM_MSB
            _OFF_PSP = _OFF_PP + 4  # partial_sp lo: 64 f32
            _PSP_SIZE = NUM_MSB * N_SP_PAIRS  # = 64 (lo half)
            _OFF_PSP_HI = _OFF_PSP + _PSP_SIZE  # partial_sp hi: 64 f32
            _OFF_PED = _OFF_PSP_HI + _PSP_SIZE  # exp_delta: 4 f32
            o_flat_init = [zero_v8f32] * (NUM_MSB * N_PV_WMMA_N)

            # Ping-pong bases for iteration 1:
            #   K_cur = K_b (tile 1 K), V_cur = V_a (tile 0 V)
            #   K_next = K_a (TDM K target), V_next = V_b (TDM V target)
            pp_init = [k_b_base_i32, v_a_base_i32, k_a_base_i32, v_b_base_i32]

            init_args = (
                o_flat_init
                + pro_old_max
                + pro_row_sums
                + kv_flat_init
                + pro_local_max
                + pro_delta
                + sp_flat_init
                + pp_init
                + pro_partial_sp_flat
                + pro_exp_delta
            )

            # ---- Split point for causal loops (chunked-prefill: sq < sk) ----
            # Tiles below _first_causal_tile are fully under the diagonal and
            # need no causal mask.  Tiles at or above it cross the diagonal.
            # When sq == sk, _first_causal_tile == num_tiles so loop 1 covers
            # everything and loop 2 runs zero iterations — no regression.
            if const_expr(IS_CAUSAL):
                # Tile t is fully below the diagonal when
                #   (t+1)*TILE_N - 1 <= _causal_offset + m_start
                # i.e. t < floor((_causal_offset + m_start) / TILE_N) + 1
                # Since m_start = bx * TILE_N this equals bx + floor(_causal_offset /
                # TILE_N).
                # Clamp to [1, num_tiles-1) since loop 1 starts at tile 1.
                _first_causal_tile = arith.addi(
                    arith.unwrap(bx), arith.divui(_causal_offset, tile_n_const)
                )
                _first_causal_tile = arith.maxsi(
                    _first_causal_tile, arith.unwrap(arith.constant(1, type=T.i32))
                )
                _first_causal_tile = arith.minui(_first_causal_tile, num_tiles_minus1)
                _first_causal_tile_idx = arith.index_cast(T.index, _first_causal_tile)
            else:
                _first_causal_tile_idx = num_tiles_minus1_idx

            # ================================================================
            # SECTION 3a: Main KV loop — non-causal tiles [1, _first_causal_tile)
            # ================================================================
            for tile_idx, iter_args, loop1_results in scf.for_(
                arith.index(1),
                _first_causal_tile_idx,
                arith.index(1),
                iter_args=init_args,
            ):
                # ---- Unpack iter_args ----
                # Pin each MSB's values to bank=d/msb for full-bank MSB
                # pattern.
                o_tiles_flat = [iter_args[i] for i in fx.range_constexpr(16)]
                o_tiles = []
                for d in fx.range_constexpr(NUM_MSB):
                    row = []
                    for n in fx.range_constexpr(N_PV_WMMA_N):
                        _idx = d * N_PV_WMMA_N + n
                        row.append(set_vgpr_bank(o_tiles_flat[_idx], d))
                    o_tiles.append(row)

                ia_old_max = [
                    set_vgpr_bank(iter_args[16 + i], i)
                    for i in fx.range_constexpr(NUM_MSB)
                ]
                ia_row_sums = [
                    set_vgpr_bank(iter_args[20 + i], i)
                    for i in fx.range_constexpr(NUM_MSB)
                ]

                kv_tiles_flat = [
                    iter_args[24 + i] for i in fx.range_constexpr(_KV_SIZE)
                ]
                kv_tiles = []
                for msb in fx.range_constexpr(NUM_MSB):
                    row = []
                    for k in fx.range_constexpr(N_WMMA_K_TILES):
                        _ki = msb * N_WMMA_K_TILES + k
                        row.append(set_vgpr_bank(kv_tiles_flat[_ki], msb))
                    kv_tiles.append(row)

                ia_local_max = [
                    set_vgpr_bank(iter_args[_OFF_LOCAL_MAX + i], i)
                    for i in fx.range_constexpr(NUM_MSB)
                ]
                ia_delta = [
                    set_vgpr_bank(iter_args[_OFF_DELTA + i], i)
                    for i in fx.range_constexpr(NUM_MSB)
                ]

                ia_sp_flat = [
                    iter_args[_OFF_SP + i] for i in fx.range_constexpr(CNT_SU * NUM_MSB)
                ]
                prev_su_sp_tiles = []
                for su in fx.range_constexpr(CNT_SU):
                    msb_list = []
                    for msb in fx.range_constexpr(NUM_MSB):
                        _si = su * NUM_MSB + msb
                        msb_list.append([set_vgpr_bank(ia_sp_flat[_si], msb)])
                    prev_su_sp_tiles.append(msb_list)

                # Unpack ping-pong bases
                ia_k_cur_base = iter_args[_OFF_PP]
                ia_v_cur_base = iter_args[_OFF_PP + 1]
                ia_k_next_base = iter_args[_OFF_PP + 2]
                ia_v_next_base = iter_args[_OFF_PP + 3]

                # Build per-lane LDS addresses from ping-pong bases
                kv_lds_addrs_cur = _build_kv_lds_addrs(
                    lane_id,
                    ia_k_cur_base,
                    ia_v_cur_base,
                )
                kv_lds_addrs_next = _build_kv_lds_addrs(
                    lane_id,
                    ia_k_next_base,
                    ia_v_next_base,
                )

                # ---- Unpack partial_sp_pairs: reconstruct
                # v2f32 from separate lo+hi f32 scalars ----
                ia_partial_sp_lo = [
                    iter_args[_OFF_PSP + i] for i in fx.range_constexpr(_PSP_SIZE)
                ]
                ia_partial_sp_hi = [
                    iter_args[_OFF_PSP_HI + i] for i in fx.range_constexpr(_PSP_SIZE)
                ]
                ia_exp_delta = [
                    set_vgpr_bank(iter_args[_OFF_PED + i], i)
                    for i in fx.range_constexpr(NUM_MSB)
                ]

                ia_partial_sp_pairs = []
                for _m in fx.range_constexpr(NUM_MSB):
                    msb_pairs = [
                        _make_v2f32(
                            ia_partial_sp_lo[_m * N_SP_PAIRS + _i],
                            ia_partial_sp_hi[_m * N_SP_PAIRS + _i],
                            _m,
                        )
                        for _i in fx.range_constexpr(N_SP_PAIRS)
                    ]
                    ia_partial_sp_pairs.append(msb_pairs)

                # ---- SP tiles: zero accumulators (fresh each iteration) ----
                sp_tiles = []
                for msb in fx.range_constexpr(NUM_MSB):
                    sp_tiles.append([set_vgpr_bank(zero_v8f32, msb)])

                softmax_state = {
                    "old_max": list(ia_old_max),
                    "local_max": list(ia_local_max),
                    "delta": list(ia_delta),
                    "exp_delta": [None] * NUM_MSB,
                    "cur_max_log2e": [None] * NUM_MSB,
                    "cur_max_log2e_1": [None] * NUM_MSB,
                    "cur_max_log2e_scalar": [None] * NUM_MSB,
                    "cur_max_log2e_dup": [None] * NUM_MSB,
                    "vgpr_log2e_scl_pair": [None] * NUM_MSB,
                    "exp_delta_dup": [None] * NUM_MSB,
                    "row_sums": list(ia_row_sums),
                    "p_bf16": [[], [], [], []],
                    "sp_pairs_prev": ia_partial_sp_pairs,
                }

                # ---- Build TDM state (zero placeholder for memload=False) ----
                _zero_i32 = arith.unwrap(arith.constant(0, type=T.i32))

                def _mk_zero_v4i32():
                    return vector.broadcast(ty["v4i32"], _zero_i32)

                def _mk_zero_v8i32():
                    return vector.broadcast(ty["v8i32"], _zero_i32)

                tdm_state = {
                    "v_g0": _mk_zero_v4i32(),
                    "v_g1": _mk_zero_v8i32(),
                    "k_g0": _mk_zero_v4i32(),
                    "k_g1": _mk_zero_v8i32(),
                    "v_salu_queue": [],
                    "k_salu_queue": [],
                }

                # ---- Compute TDM offsets ----
                tile_idx_i32 = arith.index_cast(T.i32, tile_idx)

                tile_n_stride_v = arith.muli(tile_n_const, stride_v_seq)
                cur_v_advance = arith.muli(tile_idx_i32, tile_n_stride_v)
                cur_v_offset = arith.addi(arith.unwrap(v_offset), cur_v_advance)

                next_tile = arith.addi(
                    tile_idx_i32, arith.unwrap(arith.constant(1, type=T.i32))
                )
                tile_n_stride_k = arith.muli(tile_n_const, stride_k_seq)
                next_k_advance = arith.muli(next_tile, tile_n_stride_k)
                next_k_offset = arith.addi(arith.unwrap(k_offset), next_k_advance)

                # ---- Per-tile OOB dg1 for K prefetch and V load ----
                _loop_v_remain = arith.subi(
                    actual_kv_len, arith.muli(tile_idx_i32, tile_n_const)
                )
                _loop_v_oob_dg1 = [
                    _make_kv_dg1_with_oob(
                        _V_CFG_OOB,
                        128,
                        8,
                        _stride_v_elems_oob,
                        _per_warp_oob_dim1(
                            arith.subi(
                                _loop_v_remain,
                                arith.unwrap(arith.constant(_su * 32, type=T.i32)),
                            ),
                            wave_id,
                            8,
                        ),
                    )
                    for _su in range(CNT_SU)
                ]
                _loop_k_remain = arith.subi(
                    actual_kv_len, arith.muli(next_tile, tile_n_const)
                )
                _loop_k_oob_dg1 = [
                    _make_kv_dg1_with_oob(
                        _K_CFG_OOB,
                        QK_HDIM,
                        8,
                        _stride_k_elems_oob,
                        _per_warp_oob_dim1(
                            arith.subi(
                                _loop_k_remain,
                                arith.unwrap(arith.constant(_su * 32, type=T.i32)),
                            ),
                            wave_id,
                            8,
                        ),
                        dim0_stride=200,
                    )
                    for _su in range(CNT_SU)
                ]

                # ---- Core loop: no causal mask ----
                (
                    sp_out,
                    kv_out,
                    o_tiles,
                    su_sp_tiles_out,
                    _partial_sp_lo_out,
                    _partial_sp_hi_out,
                    _partial_ed_out,
                ) = _core_loop(
                    ty,
                    False,
                    q_frags,
                    kv_tiles,
                    sp_tiles,
                    o_tiles,
                    kv_lds_addrs_cur,
                    tdm_state,
                    softmax_state,
                    sgpr_state,
                    gemm2=True,
                    tdm_v_offset=cur_v_offset,
                    tdm_v_target=ia_v_next_base,
                    tdm_k_offset=next_k_offset,
                    tdm_k_target=ia_k_next_base,
                    kv_lds_addrs_next=kv_lds_addrs_next,
                    gemm1_tdm_is_v=False,
                    loop_k_oob_dg1=_loop_k_oob_dg1,
                    loop_v_oob_dg1=_loop_v_oob_dg1,
                )

                # ---- Yield updated state with ping-pong swap ----
                new_o = []
                for d in fx.range_constexpr(NUM_MSB):
                    for n in fx.range_constexpr(N_PV_WMMA_N):
                        new_o.append(o_tiles[d][n])

                new_max = [
                    softmax_state["old_max"][i] for i in fx.range_constexpr(NUM_MSB)
                ]
                new_sums = [
                    softmax_state["row_sums"][i] for i in fx.range_constexpr(NUM_MSB)
                ]

                kv_out_flat = []
                for msb in fx.range_constexpr(NUM_MSB):
                    for k in fx.range_constexpr(N_WMMA_K_TILES):
                        kv_out_flat.append(kv_out[msb][k])

                new_local_max = [
                    softmax_state["local_max"][i] for i in fx.range_constexpr(NUM_MSB)
                ]
                new_delta = [
                    softmax_state["delta"][i] for i in fx.range_constexpr(NUM_MSB)
                ]

                sp_out_flat = []
                for su in fx.range_constexpr(CNT_SU):
                    for msb in fx.range_constexpr(NUM_MSB):
                        sp_out_flat.append(su_sp_tiles_out[su][msb][0])

                pp_swapped = [
                    ia_k_next_base,
                    ia_v_next_base,
                    ia_k_cur_base,
                    ia_v_cur_base,
                ]

                new_partial_sp_flat = _partial_sp_lo_out + _partial_sp_hi_out
                new_exp_delta = [
                    _partial_ed_out[_m] for _m in fx.range_constexpr(NUM_MSB)
                ]

                yield (
                    new_o
                    + new_max
                    + new_sums
                    + kv_out_flat
                    + new_local_max
                    + new_delta
                    + sp_out_flat
                    + pp_swapped
                    + new_partial_sp_flat
                    + new_exp_delta
                )

            # ================================================================
            # SECTION 3b: Main KV loop — causal tiles [_first_causal_tile, num_tiles-1)
            # ================================================================
            for tile_idx, iter_args, loop_results in scf.for_(
                _first_causal_tile_idx,
                num_tiles_minus1_idx,
                arith.index(1),
                iter_args=loop1_results,
            ):
                # ---- Unpack iter_args ----
                o_tiles_flat = [iter_args[i] for i in fx.range_constexpr(16)]
                o_tiles = []
                for d in fx.range_constexpr(NUM_MSB):
                    row = []
                    for n in fx.range_constexpr(N_PV_WMMA_N):
                        _idx = d * N_PV_WMMA_N + n
                        row.append(set_vgpr_bank(o_tiles_flat[_idx], d))
                    o_tiles.append(row)

                ia_old_max = [
                    set_vgpr_bank(iter_args[16 + i], i)
                    for i in fx.range_constexpr(NUM_MSB)
                ]
                ia_row_sums = [
                    set_vgpr_bank(iter_args[20 + i], i)
                    for i in fx.range_constexpr(NUM_MSB)
                ]

                kv_tiles_flat = [
                    iter_args[24 + i] for i in fx.range_constexpr(_KV_SIZE)
                ]
                kv_tiles = []
                for msb in fx.range_constexpr(NUM_MSB):
                    row = []
                    for k in fx.range_constexpr(N_WMMA_K_TILES):
                        _ki = msb * N_WMMA_K_TILES + k
                        row.append(set_vgpr_bank(kv_tiles_flat[_ki], msb))
                    kv_tiles.append(row)

                ia_local_max = [
                    set_vgpr_bank(iter_args[_OFF_LOCAL_MAX + i], i)
                    for i in fx.range_constexpr(NUM_MSB)
                ]
                ia_delta = [
                    set_vgpr_bank(iter_args[_OFF_DELTA + i], i)
                    for i in fx.range_constexpr(NUM_MSB)
                ]

                ia_sp_flat = [
                    iter_args[_OFF_SP + i] for i in fx.range_constexpr(CNT_SU * NUM_MSB)
                ]
                prev_su_sp_tiles = []
                for su in fx.range_constexpr(CNT_SU):
                    msb_list = []
                    for msb in fx.range_constexpr(NUM_MSB):
                        _si = su * NUM_MSB + msb
                        msb_list.append([set_vgpr_bank(ia_sp_flat[_si], msb)])
                    prev_su_sp_tiles.append(msb_list)

                ia_k_cur_base = iter_args[_OFF_PP]
                ia_v_cur_base = iter_args[_OFF_PP + 1]
                ia_k_next_base = iter_args[_OFF_PP + 2]
                ia_v_next_base = iter_args[_OFF_PP + 3]

                kv_lds_addrs_cur = _build_kv_lds_addrs(
                    lane_id,
                    ia_k_cur_base,
                    ia_v_cur_base,
                )
                kv_lds_addrs_next = _build_kv_lds_addrs(
                    lane_id,
                    ia_k_next_base,
                    ia_v_next_base,
                )

                ia_partial_sp_lo = [
                    iter_args[_OFF_PSP + i] for i in fx.range_constexpr(_PSP_SIZE)
                ]
                ia_partial_sp_hi = [
                    iter_args[_OFF_PSP_HI + i] for i in fx.range_constexpr(_PSP_SIZE)
                ]
                ia_exp_delta = [
                    set_vgpr_bank(iter_args[_OFF_PED + i], i)
                    for i in fx.range_constexpr(NUM_MSB)
                ]

                ia_partial_sp_pairs = []
                for _m in fx.range_constexpr(NUM_MSB):
                    msb_pairs = [
                        _make_v2f32(
                            ia_partial_sp_lo[_m * N_SP_PAIRS + _i],
                            ia_partial_sp_hi[_m * N_SP_PAIRS + _i],
                            _m,
                        )
                        for _i in fx.range_constexpr(N_SP_PAIRS)
                    ]
                    ia_partial_sp_pairs.append(msb_pairs)

                sp_tiles = []
                for msb in fx.range_constexpr(NUM_MSB):
                    sp_tiles.append([set_vgpr_bank(zero_v8f32, msb)])

                softmax_state = {
                    "old_max": list(ia_old_max),
                    "local_max": list(ia_local_max),
                    "delta": list(ia_delta),
                    "exp_delta": [None] * NUM_MSB,
                    "cur_max_log2e": [None] * NUM_MSB,
                    "cur_max_log2e_1": [None] * NUM_MSB,
                    "cur_max_log2e_scalar": [None] * NUM_MSB,
                    "cur_max_log2e_dup": [None] * NUM_MSB,
                    "vgpr_log2e_scl_pair": [None] * NUM_MSB,
                    "exp_delta_dup": [None] * NUM_MSB,
                    "row_sums": list(ia_row_sums),
                    "p_bf16": [[], [], [], []],
                    "sp_pairs_prev": ia_partial_sp_pairs,
                }

                _zero_i32 = arith.unwrap(arith.constant(0, type=T.i32))

                def _mk_zero_v4i32():
                    return vector.broadcast(ty["v4i32"], _zero_i32)

                def _mk_zero_v8i32():
                    return vector.broadcast(ty["v8i32"], _zero_i32)

                tdm_state = {
                    "v_g0": _mk_zero_v4i32(),
                    "v_g1": _mk_zero_v8i32(),
                    "k_g0": _mk_zero_v4i32(),
                    "k_g1": _mk_zero_v8i32(),
                    "v_salu_queue": [],
                    "k_salu_queue": [],
                }

                tile_idx_i32 = arith.index_cast(T.i32, tile_idx)

                tile_n_stride_v = arith.muli(tile_n_const, stride_v_seq)
                cur_v_advance = arith.muli(tile_idx_i32, tile_n_stride_v)
                cur_v_offset = arith.addi(arith.unwrap(v_offset), cur_v_advance)

                next_tile = arith.addi(
                    tile_idx_i32, arith.unwrap(arith.constant(1, type=T.i32))
                )
                tile_n_stride_k = arith.muli(tile_n_const, stride_k_seq)
                next_k_advance = arith.muli(next_tile, tile_n_stride_k)
                next_k_offset = arith.addi(arith.unwrap(k_offset), next_k_advance)

                _loop_v_remain = arith.subi(
                    actual_kv_len, arith.muli(tile_idx_i32, tile_n_const)
                )
                _loop_v_oob_dg1 = [
                    _make_kv_dg1_with_oob(
                        _V_CFG_OOB,
                        128,
                        8,
                        _stride_v_elems_oob,
                        _per_warp_oob_dim1(
                            arith.subi(
                                _loop_v_remain,
                                arith.unwrap(arith.constant(_su * 32, type=T.i32)),
                            ),
                            wave_id,
                            8,
                        ),
                    )
                    for _su in range(CNT_SU)
                ]
                _loop_k_remain = arith.subi(
                    actual_kv_len, arith.muli(next_tile, tile_n_const)
                )
                _loop_k_oob_dg1 = [
                    _make_kv_dg1_with_oob(
                        _K_CFG_OOB,
                        QK_HDIM,
                        8,
                        _stride_k_elems_oob,
                        _per_warp_oob_dim1(
                            arith.subi(
                                _loop_k_remain,
                                arith.unwrap(arith.constant(_su * 32, type=T.i32)),
                            ),
                            wave_id,
                            8,
                        ),
                        dim0_stride=200,
                    )
                    for _su in range(CNT_SU)
                ]

                # ---- Causal mask for this tile ----
                _loop_tile_n_start = arith.muli(tile_idx_i32, tile_n_const)
                _loop_causal_ns = arith.subi(_loop_tile_n_start, _causal_offset)

                # ---- Core loop: with causal mask ----
                (
                    sp_out,
                    kv_out,
                    o_tiles,
                    su_sp_tiles_out,
                    _partial_sp_lo_out,
                    _partial_sp_hi_out,
                    _partial_ed_out,
                ) = _core_loop(
                    ty,
                    False,
                    q_frags,
                    kv_tiles,
                    sp_tiles,
                    o_tiles,
                    kv_lds_addrs_cur,
                    tdm_state,
                    softmax_state,
                    sgpr_state,
                    gemm2=True,
                    tdm_v_offset=cur_v_offset,
                    tdm_v_target=ia_v_next_base,
                    tdm_k_offset=next_k_offset,
                    tdm_k_target=ia_k_next_base,
                    kv_lds_addrs_next=kv_lds_addrs_next,
                    gemm1_tdm_is_v=False,
                    causal_n_start=_loop_causal_ns,
                    loop_k_oob_dg1=_loop_k_oob_dg1,
                    loop_v_oob_dg1=_loop_v_oob_dg1,
                )

                # ---- Yield updated state with ping-pong swap ----
                new_o = []
                for d in fx.range_constexpr(NUM_MSB):
                    for n in fx.range_constexpr(N_PV_WMMA_N):
                        new_o.append(o_tiles[d][n])

                new_max = [
                    softmax_state["old_max"][i] for i in fx.range_constexpr(NUM_MSB)
                ]
                new_sums = [
                    softmax_state["row_sums"][i] for i in fx.range_constexpr(NUM_MSB)
                ]

                kv_out_flat = []
                for msb in fx.range_constexpr(NUM_MSB):
                    for k in fx.range_constexpr(N_WMMA_K_TILES):
                        kv_out_flat.append(kv_out[msb][k])

                new_local_max = [
                    softmax_state["local_max"][i] for i in fx.range_constexpr(NUM_MSB)
                ]
                new_delta = [
                    softmax_state["delta"][i] for i in fx.range_constexpr(NUM_MSB)
                ]

                sp_out_flat = []
                for su in fx.range_constexpr(CNT_SU):
                    for msb in fx.range_constexpr(NUM_MSB):
                        sp_out_flat.append(su_sp_tiles_out[su][msb][0])

                pp_swapped = [
                    ia_k_next_base,
                    ia_v_next_base,
                    ia_k_cur_base,
                    ia_v_cur_base,
                ]

                new_partial_sp_flat = _partial_sp_lo_out + _partial_sp_hi_out
                new_exp_delta = [
                    _partial_ed_out[_m] for _m in fx.range_constexpr(NUM_MSB)
                ]

                yield (
                    new_o
                    + new_max
                    + new_sums
                    + kv_out_flat
                    + new_local_max
                    + new_delta
                    + sp_out_flat
                    + pp_swapped
                    + new_partial_sp_flat
                    + new_exp_delta
                )

            # ================================================================
            # SECTION 4: Epilogue — post_process + div_cvt + write_out
            # ================================================================
            #
            #   fmha_post_process(is_odd):
            #     softmax stages 4..7 (PART2) → complete last tile's softmax
            #     LDS V(su=0..3) → PV_pure (64 WMMAs)
            #   fmha_div_cvt():
            #     row_sums cross-MSB reduce → LSE = max*scale + log(sum)
            #     O = O * rcp(row_sum) → cvt_pk_bf16
            #   lds_store_D_LSE() + TDM_store_D_LSE()
            #
            # For tile_n=128: blk=0 always, no is_odd branching.

            # ---- 4a: Unpack loop results ----
            # Pin each MSB's values to bank=d/msb for full-bank MSB pattern.
            ep_o_tiles = []
            for d in fx.range_constexpr(NUM_MSB):
                row = []
                for n in fx.range_constexpr(N_PV_WMMA_N):
                    row.append(set_vgpr_bank(loop_results[d * N_PV_WMMA_N + n], d))
                ep_o_tiles.append(row)

            ep_old_max = [
                set_vgpr_bank(loop_results[16 + i], i)
                for i in fx.range_constexpr(NUM_MSB)
            ]
            ep_row_sums = [
                set_vgpr_bank(loop_results[20 + i], i)
                for i in fx.range_constexpr(NUM_MSB)
            ]
            ep_local_max = [
                set_vgpr_bank(loop_results[_OFF_LOCAL_MAX + i], i)
                for i in fx.range_constexpr(NUM_MSB)
            ]
            ep_delta = [
                set_vgpr_bank(loop_results[_OFF_DELTA + i], i)
                for i in fx.range_constexpr(NUM_MSB)
            ]

            # Epilogue ping-pong: V_cur after swap has last tile's V data
            ep_k_cur_base = loop_results[_OFF_PP]
            ep_v_cur_base = loop_results[_OFF_PP + 1]
            ep_kv_lds_addrs = _build_kv_lds_addrs(
                lane_id,
                ep_k_cur_base,
                ep_v_cur_base,
            )

            # ---- 4b: Unpack partial_sp_pairs:
            # reconstruct v2f32 from separate lo+hi f32 ----
            ep_partial_sp_lo = [
                loop_results[_OFF_PSP + i] for i in fx.range_constexpr(_PSP_SIZE)
            ]
            ep_partial_sp_hi = [
                loop_results[_OFF_PSP_HI + i] for i in fx.range_constexpr(_PSP_SIZE)
            ]
            ep_partial_sp_pairs = []
            for _m in fx.range_constexpr(NUM_MSB):
                ep_pairs = [
                    _make_v2f32(
                        ep_partial_sp_lo[_m * N_SP_PAIRS + _i],
                        ep_partial_sp_hi[_m * N_SP_PAIRS + _i],
                        _m,
                    )
                    for _i in fx.range_constexpr(N_SP_PAIRS)
                ]
                ep_partial_sp_pairs.append(ep_pairs)

            # ---- 4b': Extra state for endtile (N>=2) ----
            # K(N-1) fragments from loop (GEMM2 stage 3 loaded from K_next).
            ep_kv_tiles_flat = [
                loop_results[24 + _i] for _i in fx.range_constexpr(_KV_SIZE)
            ]
            ep_kv_tiles = []
            for _m in fx.range_constexpr(NUM_MSB):
                _row = [
                    set_vgpr_bank(
                        ep_kv_tiles_flat[_m * N_WMMA_K_TILES + _k],
                        _m,
                    )
                    for _k in fx.range_constexpr(N_WMMA_K_TILES)
                ]
                ep_kv_tiles.append(_row)

            # K_next / V_next LDS bases for endtile
            # core_loop kv_lds_addrs_next.
            ep_k_next_base = loop_results[_OFF_PP + 2]
            ep_v_next_base = loop_results[_OFF_PP + 3]
            ep_kv_lds_addrs_next = _build_kv_lds_addrs(
                lane_id,
                ep_k_next_base,
                ep_v_next_base,
            )

            # V(N-1) global offset for endtile GEMM1 TDM.
            _num_tiles_m1_ep = arith.subi(
                num_tiles,
                arith.unwrap(arith.constant(1, type=T.i32)),
            )
            ep_v_endtile_offset = arith.addi(
                arith.unwrap(v_offset),
                arith.muli(arith.muli(_num_tiles_m1_ep, tile_n_const), stride_v_seq),
            )

            # ia_exp_delta for endtile core_loop: exp_delta from last loop GEMM2.
            ia_exp_delta = [
                set_vgpr_bank(loop_results[_OFF_PED + _m], _m)
                for _m in fx.range_constexpr(NUM_MSB)
            ]

            # ---- 4c: s_wait_idle + barrier ----
            _emit_void("s_wait_idle")
            rocdl.s_barrier_signal(-1)
            rocdl.s_barrier_wait(-1)

            # ---- 4c': _ep_finish — PART2 + PV_pure + div_cvt + TDM store D ----
            # o_tiles:           [[v8f32]*N_PV_WMMA_N]*NUM_MSB — accumulated O
            # sp_pairs_in:       [[v2f32]*N_SP_PAIRS]*NUM_MSB  — PART2 first-half input
            # exp_delta_rescale: [f32]*NUM_MSB                 — exp_delta for O
            # rescale before PV
            # v_base_for_pv:     i32                           — V LDS base for PV_pure
            # old_max_in:        [f32]*NUM_MSB                 — max across all tiles
            # seen so far
            # local_max_in:      [f32]*NUM_MSB                 — local max (same as
            # old_max after PART0+1)
            # delta_in:          [f32]*NUM_MSB                 — delta (for PART2 setup)
            # row_sums_in:       [f32]*NUM_MSB                 — row_sums accumulated
            # to this point
            def _ep_finish(
                o_tiles,
                sp_pairs_in,
                exp_delta_rescale,
                v_base_for_pv,
                old_max_in,
                local_max_in,
                delta_in,
                row_sums_in,
            ):
                sfx = {
                    "old_max": list(old_max_in),
                    "local_max": list(local_max_in),
                    "delta": list(delta_in),
                    "exp_delta": [None] * NUM_MSB,
                    "cur_max_log2e": [None] * NUM_MSB,
                    "cur_max_log2e_1": [None] * NUM_MSB,
                    "cur_max_log2e_scalar": [None] * NUM_MSB,
                    "cur_max_log2e_dup": [None] * NUM_MSB,
                    "vgpr_log2e_scl_pair": [None] * NUM_MSB,
                    "exp_delta_dup": [None] * NUM_MSB,
                    "row_sums": list(row_sums_in),
                    "p_bf16": [[], [], [], []],
                    "sp_pairs_prev": sp_pairs_in,
                }
                # DBG: print row_sums_in at ep_finish entry

                # PART2 second half: setup ops + pair_exp+cvt+sum.
                # MSBs 0,1: pairs 0-3 already exp'd by GEMM2 EXP tokens → start from
                # PART2_SPLIT.
                _p2ops = _build_all_softmax_part2_ops(
                    ty,
                    0,
                    sp_pairs_in,
                    sfx,
                    sgpr_state,
                )
                for _m in fx.range_constexpr(NUM_MSB):
                    for _i in fx.range_constexpr(PART2_SETUP_A - 1):  # ops 0..6
                        _p2ops[_m][_i]()

                for _m in fx.range_constexpr(NUM_MSB):
                    for _op in _p2ops[_m][PART2_SPLIT:]:
                        _op()

                p_tiles = _build_p_tiles_from_softmax(ty, sfx)

                # O rescale
                _vf8 = ir.VectorType.get([8], ir.F32Type.get())
                for _msb in fx.range_constexpr(NUM_MSB):
                    _ed = exp_delta_rescale[_msb]
                    _edv8 = llvm_dialect.mlir_undef(_vf8)
                    for _i in fx.range_constexpr(8):
                        _ci = arith.unwrap(arith.constant(_i, type=T.i32))
                        _edv8 = llvm_dialect.insertelement(
                            _edv8,
                            _ed,
                            _ci,
                        )
                    for _n in fx.range_constexpr(N_PV_WMMA_N):
                        o_tiles[_msb][_n] = arith.mulf(o_tiles[_msb][_n], _edv8)

                # PV_pure
                _kv_pv = _build_kv_lds_addrs(lane_id, ep_k_cur_base, v_base_for_pv)
                for _sp in fx.range_constexpr(2):
                    _sb = _sp * 2
                    _vr0, _vr1 = _load_v_two_sus_from_lds(ty, _kv_pv, 0, _sb, _sb + 1)
                    o_tiles = _pv_pure_su(
                        ty,
                        0,
                        _sb,
                        _pair_v_tiles_for_wmma(_vr0, ty),
                        p_tiles[_sb],
                        o_tiles,
                    )
                    o_tiles = _pv_pure_su(
                        ty,
                        0,
                        _sb + 1,
                        _pair_v_tiles_for_wmma(_vr1, ty),
                        p_tiles[_sb + 1],
                        o_tiles,
                    )

                # 4d: div_cvt
                _v8f32 = ir.VectorType.get([8], ir.F32Type.get())
                _v8bf16 = ir.VectorType.get([8], ir.BF16Type.get())
                _rsf = list(sfx["row_sums"])
                _lmf = list(sfx["local_max"])
                for _mb in fx.range_constexpr(0, NUM_MSB, 2):
                    _sm = arith.addf(_rsf[_mb], _rsf[_mb + 1])
                    _slo = arith.unwrap(arith.constant(0x76543210, type=T.i32))
                    _shi = arith.unwrap(arith.constant(0xFEDCBA98, type=T.i32))
                    _pm = _rocdl_permlanex16(
                        ty["f32"],
                        _sm,
                        _sm,
                        _slo,
                        _shi,
                        False,
                        False,
                    )
                    _sf = arith.addf(_sm, _pm)
                    _rsf[_mb] = _sf
                    _rsf[_mb + 1] = _sf
                _l2e = arith.constant(0.6931471805599453, type=T.f32)
                _lse_vals = [None] * NUM_MSB
                for _msb in fx.range_constexpr(NUM_MSB):
                    _mxs = arith.mulf(_lmf[_msb], scalar_f)
                    _lgs = rocdl.log(ty["f32"], _rsf[_msb])
                    _lse_vals[_msb] = arith.addf(arith.mulf(_lgs, _l2e), _mxs)

                # Store LSE to global: ptr_LSE layout (total_q, nheads) fp32
                if const_expr(RETURN_LSE):
                    # lse[tok, head] where tok = q_start_tok + bx*128 + wave*32 +
                    # lane_lo + msb_off
                    from flydsl._mlir.dialects import fly as _fly_lse
                    from flydsl._mlir.dialects import llvm as _llvm_lse

                    _i64_lse = ir.IntegerType.get_signless(64)
                    _glbpt_lse = ir.Type.parse("!llvm.ptr<1>")
                    _lse_raw = ptr_LSE.__extract_to_ir_values__()[0]
                    _lse_base = _fly_lse.extract_aligned_pointer_as_index(
                        _glbpt_lse, _lse_raw
                    )
                    _lse_base_i64 = _llvm_lse.ptrtoint(_i64_lse, _lse_base)

                    _wv_lse = rocdl.wave_id()
                    _lane_lo_lse = arith.unwrap(
                        arith.andi(
                            lane_id,
                            arith.constant(15, type=T.i32),
                        )
                    )

                    _lse_bx128 = arith.muli(
                        arith.unwrap(bx), arith.unwrap(arith.constant(128, type=T.i32))
                    )
                    _lse_wv32 = arith.muli(
                        _wv_lse, arith.unwrap(arith.constant(WV_SUBQD, type=T.i32))
                    )
                    _lse_base_row = arith.addi(_lse_bx128, _lse_wv32)

                    for _msb_lse in [0, 2]:
                        _msb_off = 0 if _msb_lse == 0 else 16
                        _seq_pos = arith.addi(
                            _lse_base_row,
                            arith.addi(
                                _lane_lo_lse,
                                arith.unwrap(arith.constant(_msb_off, type=T.i32)),
                            ),
                        )
                        _lse_valid = arith.cmpi(
                            arith.CmpIPredicate.slt,
                            _seq_pos,
                            actual_q_len,
                        )
                        _lse_if = scf.IfOp(_lse_valid)
                        with _if_then(_lse_if):
                            # tok = q_start_tok + seq_pos
                            _lse_tok = arith.addi(q_start_tok, _seq_pos)
                            # elem_off = tok * nheads + head
                            _lse_elem_off = arith.addi(
                                arith.muli(_lse_tok, _gdz), arith.unwrap(by)
                            )
                            _lse_byte_off = arith.muli(
                                _lse_elem_off,
                                arith.unwrap(arith.constant(4, type=T.i32)),
                            )
                            _lse_byte_off_i64 = arith.extsi(_i64_lse, _lse_byte_off)
                            _lse_addr = arith.addi(_lse_base_i64, _lse_byte_off_i64)
                            _lse_ptr = _llvm_lse.inttoptr(_glbpt_lse, _lse_addr)
                            _llvm_lse.store(_lse_vals[_msb_lse], _lse_ptr)
                _obf16 = []
                for _msb in fx.range_constexpr(NUM_MSB):
                    _rcp = rocdl.rcp(ty["f32"], _rsf[_msb])
                    _rv8 = llvm_dialect.mlir_undef(_v8f32)
                    for _i in fx.range_constexpr(8):
                        _ci = arith.unwrap(arith.constant(_i, type=T.i32))
                        _rv8 = llvm_dialect.insertelement(
                            _rv8,
                            _rcp,
                            _ci,
                        )
                    _mb16 = []
                    for _n in fx.range_constexpr(N_PV_WMMA_N):
                        _mb16.append(
                            arith.truncf(
                                _v8bf16,
                                arith.mulf(o_tiles[_msb][_n], _rv8),
                            )
                        )
                    _obf16.append(_mb16)

                # Barrier: ensure all waves finish PV before any wave writes D to V_a
                # LDS.
                # Without this, wave 0 can overwrite V_a SU data still being read by
                # waves 1-3.
                rocdl.s_barrier_signal(-1)
                rocdl.s_barrier_wait(-1)

                # 4e: TDM store D (VGPR -> LDS -> Global)
                _i32t = ir.IntegerType.get_signless(32)
                _ldst = ir.Type.parse("!llvm.ptr<3>")
                _v4i32t = ir.VectorType.get([4], _i32t)
                _db32 = _extract_lds_base_i32(_lds_alloc_v_a.get_base())
                _dw_wv = arith.muli(
                    arith.unwrap(wave_id),
                    arith.unwrap(arith.constant(LDS_D_WV_SIZE, type=T.i32)),
                )
                _dw = arith.addi(_db32, _dw_wv)
                _llo = arith.unwrap(
                    arith.andi(
                        lane_id,
                        arith.constant(15, type=T.i32),
                    )
                )
                _lhi = arith.unwrap(
                    arith.shrui(
                        lane_id,
                        arith.constant(4, type=T.i32),
                    )
                )
                _loff = arith.addi(
                    arith.muli(
                        _llo,
                        arith.unwrap(arith.constant(TDM_D_TILE_DIM0, type=T.i32)),
                    ),
                    arith.muli(
                        _lhi,
                        arith.unwrap(arith.constant(16, type=T.i32)),
                    ),
                )
                for _msb in fx.range_constexpr(NUM_MSB):
                    for _n in fx.range_constexpr(N_PV_WMMA_N):
                        _ioff = (
                            (_msb // 2) * 16 * TDM_D_TILE_DIM0
                            + (_msb % 2) * 128
                            + _n * 32
                        )
                        _la = arith.addi(
                            arith.addi(_dw, _loff),
                            arith.unwrap(arith.constant(_ioff, type=T.i32)),
                        )
                        llvm_dialect.store(
                            vector.bitcast(_v4i32t, _obf16[_msb][_n]),
                            llvm_dialect.inttoptr(_ldst, _la),
                            volatile_=True,
                        )
                _emit_void("s_wait_dscnt 0x0")
                _wsgpr = rocdl.wave_id()
                from flydsl._mlir.dialects import fly as _fly2
                from flydsl._mlir.dialects import llvm as _llvm2

                _i64t = ir.IntegerType.get_signless(64)
                _glbpt = ir.Type.parse("!llvm.ptr<1>")
                # THD O element offset:
                #   elem_off = by*stride_o_head + _o_tok*stride_o_seq
                _o_tok = arith.addi(
                    arith.addi(
                        q_start_tok,
                        arith.muli(
                            bx,
                            arith.unwrap(arith.constant(128, type=T.i32)),
                        ),
                    ),
                    arith.muli(
                        _wsgpr,
                        arith.unwrap(arith.constant(WV_SUBQD, type=T.i32)),
                    ),
                )
                _o_elem_off = arith.addi(
                    arith.muli(by, stride_o_head),
                    arith.muli(_o_tok, stride_o_seq),
                )
                _o_raw = ptr_O.__extract_to_ir_values__()[0]
                _o_gp = _fly2.extract_aligned_pointer_as_index(_glbpt, _o_raw)
                _o64 = _llvm2.ptrtoint(_i64t, _o_gp)
                _boff32 = arith.muli(
                    _o_elem_off,
                    arith.unwrap(arith.constant(2, type=T.i32)),
                )
                _boff64 = arith.extsi(_i64t, _boff32)
                _oadr64 = arith.addi(_o64, _boff64)
                _alo, _ahi = _split_i64_to_lo_hi(_oadr64)
                _olds2 = arith.addi(
                    _extract_lds_base_i32(_lds_alloc_v_a.get_base()),
                    arith.muli(
                        _wsgpr,
                        arith.unwrap(arith.constant(LDS_D_WV_SIZE, type=T.i32)),
                    ),
                )
                _dg0 = vector.from_elements(
                    T.vec(4, T.i32),
                    [
                        arith.unwrap(arith.constant(1, type=T.i32)),
                        _olds2,
                        _alo,
                        _ahi,
                    ],
                )
                _g0 = arith.unwrap(arith.constant((1 << 16) | 0, type=T.i32))
                _g1 = arith.unwrap(arith.constant((128 & 0xFFFF) << 16, type=T.i32))
                _i32_o = ir.IntegerType.get_signless(32)
                _td1_lo_o = arith.andi(_o_oob_dim1, arith.constant(0xFFFF, type=T.i32))
                _g2 = arith.ori(
                    arith.shli(_td1_lo_o, arith.constant(16, type=T.i32)),
                    arith.constant((128 >> 16) & 0xFFFF, type=T.i32),
                )
                _g3_val = ((32 >> 16) & 0xFFFF) | ((128 & 0xFFFF) << 16)
                _g3 = arith.unwrap(arith.constant(_g3_val, type=T.i32))
                _g4 = arith.unwrap(arith.constant(32 & 0xFFFF, type=T.i32))
                _g5 = stride_o_seq
                _g6 = arith.unwrap(arith.constant(0, type=T.i32))
                _g7 = arith.unwrap(arith.constant(0, type=T.i32))
                _dg1 = vector.from_elements(
                    T.vec(8, T.i32),
                    [
                        _g0,
                        _g1,
                        _g2,
                        _g3,
                        _g4,
                        _g5,
                        _g6,
                        _g7,
                    ],
                )
                tdm_ops.tensor_store_2d(tdm_ops.TDMDescriptor2D(_dg0, _dg1))
                tdm_ops.tensor_wait(0)

            # ---- 4c'': endtile dispatch — if N>=2: core_loop + ep_finish, else:
            # ep_finish ----
            _two_ep = arith.constant(2, type=T.i32)
            _is_multi = arith.cmpi(arith.CmpIPredicate.uge, num_tiles, _two_ep)

            if _is_multi:  # N>=2: endtile core_loop then ep_finish
                # All variables defined fresh inside THEN — not state variables.
                _et_sp_t = [
                    [set_vgpr_bank(zero_v8f32, _m)]
                    for _m in fx.range_constexpr(NUM_MSB)
                ]
                _et_sfx = {
                    "old_max": list(ep_old_max),
                    "local_max": list(ep_local_max),
                    "delta": list(ep_delta),
                    "exp_delta": [None] * NUM_MSB,
                    "cur_max_log2e": [None] * NUM_MSB,
                    "cur_max_log2e_1": [None] * NUM_MSB,
                    "cur_max_log2e_scalar": [None] * NUM_MSB,
                    "cur_max_log2e_dup": [None] * NUM_MSB,
                    "vgpr_log2e_scl_pair": [None] * NUM_MSB,
                    "exp_delta_dup": [None] * NUM_MSB,
                    "row_sums": list(ep_row_sums),
                    "p_bf16": [[], [], [], []],
                    "sp_pairs_prev": [
                        [
                            ep_partial_sp_pairs[_m][_i]
                            for _i in fx.range_constexpr(N_SP_PAIRS)
                        ]
                        for _m in fx.range_constexpr(NUM_MSB)
                    ],
                }
                # DBG: row_sums entering endtile core_loop (= ep_row_sums from
                # loop_results)

                _et_z = arith.unwrap(arith.constant(0, type=T.i32))
                _et_tdm = {
                    "v_g0": vector.broadcast(ty["v4i32"], _et_z),
                    "v_g1": vector.broadcast(ty["v8i32"], _et_z),
                    "k_g0": vector.broadcast(ty["v4i32"], _et_z),
                    "k_g1": vector.broadcast(ty["v8i32"], _et_z),
                    "v_salu_queue": [],
                    "k_salu_queue": [],
                }
                _et_o = [
                    [ep_o_tiles[_d][_n] for _n in range(N_PV_WMMA_N)]
                    for _d in range(NUM_MSB)
                ]
                if const_expr(IS_CAUSAL):
                    _et_tile_n_start = arith.muli(
                        arith.subi(
                            arith.index_cast(T.i32, num_tiles_idx),
                            arith.constant(1, type=T.i32),
                        ),
                        arith.constant(TILE_N, type=T.i32),
                    )
                    _et_causal_ns = arith.subi(_et_tile_n_start, _causal_offset)
                else:
                    _et_causal_ns = None
                _et_kv_remain = arith.subi(
                    actual_kv_len,
                    arith.muli(
                        arith.subi(
                            num_tiles,
                            arith.unwrap(arith.constant(1, type=T.i32)),
                        ),
                        tile_n_const,
                    ),
                )
                _et_v_oob_dg1 = [
                    _make_kv_dg1_with_oob(
                        _V_CFG_OOB,
                        128,
                        8,
                        _stride_v_elems_oob,
                        _per_warp_oob_dim1(
                            arith.subi(
                                _et_kv_remain,
                                arith.unwrap(arith.constant(_su * 32, type=T.i32)),
                            ),
                            wave_id,
                            8,
                        ),
                    )
                    for _su in range(CNT_SU)
                ]
                _, _, _et_o, _, _et_psp_lo, _et_psp_hi, _et_ped = _core_loop(
                    ty,
                    False,
                    q_frags,
                    ep_kv_tiles,
                    _et_sp_t,
                    _et_o,
                    ep_kv_lds_addrs,
                    _et_tdm,
                    _et_sfx,
                    sgpr_state,
                    gemm2=True,
                    tdm_v_offset=ep_v_endtile_offset,
                    tdm_v_target=ep_v_next_base,
                    tdm_k_offset=None,
                    kv_lds_addrs_next=ep_kv_lds_addrs_next,
                    gemm1_tdm_is_v=True,
                    causal_n_start=_et_causal_ns,
                    endtile_v_dg1=_et_v_oob_dg1,
                    kv_oob_cols=_et_kv_remain,
                )
                # Pass updated softmax state (old_max/local_max/delta/row_sums after
                # PART0+1
                # for the endtile tile) so _ep_finish can correctly run PART2 second
                # half.
                # Reconstruct v2f32 sp_pairs for _ep_finish from safe f32 lo+hi scalars.
                _et_psp = []
                for _rpsm in fx.range_constexpr(NUM_MSB):
                    _rpairs = [
                        _make_v2f32(
                            _et_psp_lo[_rpsm * N_SP_PAIRS + _rpi],
                            _et_psp_hi[_rpsm * N_SP_PAIRS + _rpi],
                            _rpsm,
                        )
                        for _rpi in fx.range_constexpr(N_SP_PAIRS)
                    ]
                    _et_psp.append(_rpairs)
                rocdl.s_wait_tensorcnt(0)
                rocdl.s_barrier_signal(-1)
                rocdl.s_barrier_wait(-1)
                _ep_finish(
                    _et_o,
                    _et_psp,
                    _et_ped,
                    ep_v_next_base,
                    _et_sfx["old_max"],
                    _et_sfx["local_max"],
                    _et_sfx["delta"],
                    _et_sfx["row_sums"],
                )
            else:  # N=1: original epilogue flow
                _ep_finish(
                    [
                        [ep_o_tiles[_d][_n] for _n in range(N_PV_WMMA_N)]
                        for _d in range(NUM_MSB)
                    ],
                    ep_partial_sp_pairs,
                    [loop_results[_OFF_PED + _m] for _m in fx.range_constexpr(NUM_MSB)],
                    ep_v_cur_base,
                    list(ep_old_max),
                    list(ep_local_max),
                    list(ep_delta),
                    list(ep_row_sums),
                )

    return fmha_fwd_kernel


# ============================================================================
# Launch wrapper + PyTorch entry point
# ============================================================================

HEAD_DIM_QK = 192
HEAD_DIM_V = 128
BLOCK_M = 128
KV_TILE_N = 128
BPP = 2  # bytes per element (bf16)

_launch_fns = {}  # {(is_causal, return_lse): launch_fn}


def _patch_reusable_slot_specs():
    import ctypes
    from flydsl.expr.numeric import Float32, Float64

    if not hasattr(Float32, "_reusable_slot_spec"):

        @classmethod
        def _f32_slot_spec(cls, arg):
            return ctypes.c_float, lambda a: a.value if hasattr(a, "value") else a

        Float32._reusable_slot_spec = _f32_slot_spec
        Float32._reusable_ctype = ctypes.c_float

    if not hasattr(Float64, "_reusable_slot_spec"):

        @classmethod
        def _f64_slot_spec(cls, arg):
            return ctypes.c_double, lambda a: a.value if hasattr(a, "value") else a

        Float64._reusable_slot_spec = _f64_slot_spec
        Float64._reusable_ctype = ctypes.c_double


def _ensure_kernel(is_causal: bool, return_lse: bool = False):
    key = (is_causal, return_lse)
    if key in _launch_fns:
        return

    _patch_reusable_slot_specs()

    kernel = compile_fmha_fwd(is_causal=is_causal, return_lse=return_lse)

    @flyc.jit
    def _launch(
        ptr_O: fx.Pointer,
        ptr_Q: fx.Pointer,
        ptr_K: fx.Pointer,
        ptr_V: fx.Pointer,
        ptr_LSE: fx.Pointer,
        ptr_cu_seqlens_q: fx.Pointer,
        ptr_cu_seqlens_k: fx.Pointer,
        scalar_f: fx.Float32,
        stride_q_seq: fx.Int32,
        stride_k_seq: fx.Int32,
        stride_v_seq: fx.Int32,
        stride_o_seq: fx.Int32,
        stride_q_head: fx.Int32,
        stride_k_head: fx.Int32,
        stride_v_head: fx.Int32,
        stride_o_head: fx.Int32,
        gqa: fx.Int32,
        max_seqlen_q: fx.Int32,
        max_seqlen_k: fx.Int32,
        num_heads: fx.Int32,
        batch_size: fx.Int32,
        stream: fx.Stream,
    ):
        _lds_alloc_k_a.finalized = False
        _lds_alloc_k_b.finalized = False
        _lds_alloc_v_a.finalized = False
        _lds_alloc_v_b.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            _lds_alloc_k_a.finalize()
            _lds_alloc_k_b.finalize()
            _lds_alloc_v_a.finalize()
            _lds_alloc_v_b.finalize()

        num_tg = arith.index_cast(
            T.index,
            arith.ceildivui(
                arith.unwrap(max_seqlen_q), arith.constant(BLOCK_M, type=T.i32)
            ),
        )
        grid_x = arith.index_cast(T.index, batch_size)
        grid_z = arith.index_cast(T.index, num_heads)

        launcher = kernel(
            ptr_O,
            ptr_Q,
            ptr_K,
            ptr_V,
            ptr_LSE,
            ptr_cu_seqlens_q,
            ptr_cu_seqlens_k,
            scalar_f,
            stride_q_seq,
            stride_k_seq,
            stride_v_seq,
            stride_o_seq,
            stride_q_head,
            stride_k_head,
            stride_v_head,
            stride_o_head,
            gqa,
            max_seqlen_q,
            max_seqlen_k,
        )
        launcher.launch(
            grid=(grid_x, num_tg, grid_z),
            block=(BLOCK_SIZE, 1, 1),
            stream=stream,
        )

    _launch.compile_hints["llvm_options"] = {"amdgpu-expert-scheduling-mode": True}
    _launch_fns[key] = _launch


def flash_attn_varlen_d192_gfx1250(
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
    assert q.dtype == torch.bfloat16, f"Expected bf16, got {q.dtype}"
    assert q.shape[-1] == HEAD_DIM_QK, (
        f"Expected headdim_qk={HEAD_DIM_QK}," f" got {q.shape[-1]}"
    )
    assert v.shape[-1] == HEAD_DIM_V, (
        f"Expected headdim_v={HEAD_DIM_V}," f" got {v.shape[-1]}"
    )

    total_q_tokens = q.shape[0]
    batch = cu_seqlens_q.shape[0] - 1
    nheads_q = q.shape[1]
    nheads_k = k.shape[1]
    gqa = nheads_q // nheads_k

    if softmax_scale is None:
        softmax_scale = 1.0 / (HEAD_DIM_QK**0.5)

    if out is None:
        out = torch.empty(
            (total_q_tokens, nheads_q, HEAD_DIM_V),
            dtype=torch.bfloat16,
            device=q.device,
        )
    if return_lse:
        lse = torch.empty(
            (total_q_tokens, nheads_q), dtype=torch.float32, device=q.device
        )
    else:
        lse = torch.empty(
            (batch, nheads_q, max_seqlen_q), dtype=torch.float32, device=q.device
        )

    stride_q_seq = q.stride(0) * BPP
    stride_k_seq = k.stride(0) * BPP
    stride_v_seq = v.stride(0) * BPP
    stride_o_seq = out.stride(0)
    stride_q_head = q.stride(1) * BPP
    stride_k_head = k.stride(1) * BPP
    stride_v_head = v.stride(1) * BPP
    stride_o_head = out.stride(1)

    _ensure_kernel(bool(causal), bool(return_lse))

    _run_compiled(
        _launch_fns[(bool(causal), bool(return_lse))],
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
        gqa,
        max_seqlen_q,
        max_seqlen_k,
        nheads_q,
        batch,
        torch.cuda.current_stream(),
    )

    if return_lse:
        return out, lse
    return out
