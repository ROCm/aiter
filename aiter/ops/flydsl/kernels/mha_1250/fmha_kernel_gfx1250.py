"""
 Forward Kernel — gfx1250, Unified Prologue + Core Loop + Dynamic KV Loop.

Integrates:
  - Prologue (fmha_prologue_gfx1250.py): HW setup, Q load, K/V addr gen
  - Core loop (fmha_core_loop_gfx1250.py): GEMM1(QK) + GEMM2(PV) interleaved
  - Dynamic scf.for_ loop over KV tiles (tile_n=128, variable kv_seq_len)

Target: gfx1250 (MI450), wave32, 4 waves per TG (1TG), 1024 shared VGPRs.
Causal mask via FMHA_CAUSAL=1 env var (compile-time). kv_seq_len must be aligned to tile_n=128.

    core_loop: tile_n=128, 4 stages (1 MSB per VGPR bank)

    GEMM1 (QK): 24 WMMAs per stage × 4 stages = 96
    GEMM2 (PV): 16 WMMAs per stage × 4 stages = 64

    TDM loads 192 bytes per row
"""


from __future__ import annotations

import functools
import os
import sys

_REPO = os.path.join(os.path.dirname(__file__), "FlyDSL")
_BUILD_PKGS = os.path.join(_REPO, "build-fly", "python_packages")
for p in [_BUILD_PKGS, os.path.join(_REPO, "python"), os.path.dirname(__file__)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm as llvm_dialect
from flydsl._mlir.dialects import rocdl as _rocdl
from flydsl._mlir.dialects import scf
from flydsl._mlir.dialects import vector as vector_dialect
from flydsl.expr import arith, buffer_ops, gpu, rocdl, vector
from flydsl.expr.primitive import const_expr
from flydsl.expr.rocdl import tdm_ops
from flydsl.expr.typing import T
from flydsl.utils.smem_allocator import SmemAllocator

from fmha_prologue_gfx1250 import (
    _raw, _asm_void, set_vgpr_bank, set_vgpr_bank_offset,
    _setreg, _build_tdm_dgroup1, _split_i64_to_lo_hi,
    _phase4_q_load_flydsl,
    _phase5_head_index_div_flydsl,
    _compute_k_global_addr, _compute_v_global_addr,
    BLOCK_SIZE, K_TILE_N,
    _K_TDM_CONFIG, _V_TDM_CONFIG,
    K_ROW_BYTES, V_ROW_BYTES,
    K_SU_HALF_OFFSET, V_SU_HALF_OFFSET,
)

from fmha_core_loop_gfx1250 import (
    _get_types, SP_PAIR_BASE,
    _make_v2f32,
    _pair_k_tiles_for_wmma,
    _load_v_two_sus_from_lds,
    _pair_v_tiles_for_wmma,
    _broadcast_f32_to_v2f32,
    _qk_pure_su, _pv_pure_su,
    _load_k_su_from_lds, _load_v_su_from_lds,
    _sp_tiles_to_sp_pairs,
    _fill_sp_pairs_for_su,
    _softmax_pure,
    _softmax_part01_only,
    _dispatch_part0_chunk,
    _dispatch_part1,
    _softmax_part2_only,
    _build_p_tiles_from_softmax,
    _cl_su_v3_stage, _cl_su_v3_stage_gemm2,
    _build_all_softmax_part2_ops,
    _build_all_softmax_gemm2_ops,
    _sched_barrier, _asm_void as _cl_asm_void,
    _atom_s_wait_dscnt,
    _build_lds_k_schedule, _emit_lds_load, make_wmma_frag_bf16,
    NUM_MSB, NUM_WAVES, WAVE_SIZE,
    Q_WMMA_PER_MSB, N_WMMA_K_TILES, N_LDS_PER_MSB, N_LDS_V_PER_MSB,
    N_SP_PAIRS, N_PV_WMMA_N, N_V_MSB, CNT_SU,
    SU_K_N, LDS_K_SU_P_SIZE, LDS_V_SU_P_SIZE,
    KV_K, KV_V, KV_NONE,
    LDS_INST_COUNT, LDS_V_INST_COUNT, ALU_STAGES, ALU_PER_STAGE, RLTS_LEN,
    QK_HDIM, V_HDIM, KV_BPP,
    PART2_SPLIT,
    PART2_EXP_START,
    PART2_SETUP_A,
    PART0_INSTS,
    PART1_INSTS,
    N_VALID_GROUPS,
    N_SP_PAIRS,
)

# ============================================================================
# Constants
# ============================================================================

TILE_N = K_TILE_N      # 128 — KV tile width


# ============================================================================
# SmemAllocators — 4 separate LDS regions for K/V ping-pong
# ============================================================================
# K per tile: CNT_SU(4) × LDS_K_SU_P_SIZE(0x3200) = 0xC800 = 51200 bytes (for QK_HDIM=192)
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
_lds_alloc_v_b.ptr = CNT_SU * LDS_V_SU_P_SIZE   # 0x9000, last buffer — no segment padding

TDM_D_TILE_DIM0 = 128 * 2   # 256 bytes per LDS row
TDM_D_TENSOR_DIM0 = 128 * 2
WV_SUBQD = 32
LDS_D_WV_SIZE = WV_SUBQD * TDM_D_TILE_DIM0 + 1024   # 9216 bytes per wave

_lds_allocator = _lds_alloc_k_a


# ============================================================================
# LDS base extraction helper
# ============================================================================

def _extract_lds_base_i32(memref_base):
    """Extract i32 LDS address from SmemAllocator memref base."""
    from flydsl._mlir.dialects import memref as _memref_d
    idx = _memref_d.extract_aligned_pointer_as_index(memref_base)
    return _raw(arith.index_cast(T.i32, idx))


def _build_kv_lds_addrs(lane_id, k_base_i32, v_base_i32):
    """Build kv_lds_addrs[12] from allocator i32 bases + per-lane offsets.

    Layout:
      [0..3]  K addresses: k_dh0(b0), k_dh1(b1), k_dh0_hi(b2), k_dh1_hi(b3)
      [4..11] V addresses: 2 per MSB, BOTH in the SAME bank as that MSB.
                msb=0: [4]=v_dh0(b0), [5]=v_dh1(b0)
                msb=1: [6]=v_dh0(b1), [7]=v_dh1(b1)
                msb=2: [8]=v_dh0(b2), [9]=v_dh1(b2)
                msb=3: [10]=v_dh0(b3),[11]=v_dh1(b3)

    _v_V_lds_addr[msb] and _v_V_lds_addr1[msb] are both in bank msb:
    When dst, addr_dh0 and addr_dh1 are all in bank_msb, cal_set_msb produces
    MSB=0x00/0x55/0xAA/0xFF (全-bankN) → zero s_set_vgpr_msb within each MSB group.
    Previous layout put dh0 and dh1 in DIFFERENT banks, causing alternating
    0x02↔0x03 / 0x42↔0x43 etc. switches on every consecutive V load pair.
    """
    lane_lo = _raw(arith.andi(lane_id, arith.constant(0xF, type=T.i32)))
    lane_hi = _raw(arith.shrui(lane_id, arith.constant(4, type=T.i32)))

    k_lane_off = _raw(arith.addi(
        arith.muli(lane_lo, arith.constant(K_ROW_BYTES, type=T.i32)),
        arith.muli(lane_hi, arith.constant(16, type=T.i32))))

    k_dh0 = _raw(arith.addi(k_base_i32, k_lane_off))
    k_dh1 = _raw(arith.addi(k_dh0, arith.constant(K_SU_HALF_OFFSET, type=T.i32)))

    lane_and_7 = _raw(arith.andi(lane_id, arith.constant(7, type=T.i32)))
    lane_shr4 = _raw(arith.shrui(lane_id, arith.constant(4, type=T.i32)))
    v_row = _raw(arith.addi(
        lane_and_7,
        arith.shli(lane_shr4, arith.constant(3, type=T.i32))))

    lane_shr3 = _raw(arith.shrui(lane_id, arith.constant(3, type=T.i32)))
    v_sub_col = _raw(arith.shli(
        arith.andi(lane_shr3, arith.constant(1, type=T.i32)),
        arith.constant(4, type=T.i32)))

    v_lane_off = _raw(arith.addi(
        arith.muli(v_row, arith.constant(V_ROW_BYTES, type=T.i32)),
        v_sub_col))

    v_dh0 = _raw(arith.addi(v_base_i32, v_lane_off))
    v_dh1 = _raw(arith.addi(v_dh0, arith.constant(V_SU_HALF_OFFSET, type=T.i32)))

    K_COL_D_HALF = QK_HDIM * KV_BPP // 2
    k_dh0_hi = _raw(arith.addi(k_dh0, arith.constant(K_COL_D_HALF, type=T.i32)))
    k_dh1_hi = _raw(arith.addi(k_dh1, arith.constant(K_COL_D_HALF, type=T.i32)))

    rocdl.sched_barrier(0)

    # MSB-specific column-group offsets folded into each address register so
    # all 8 V SSA values are genuinely distinct → one bank hint each → no
    # allocator conflict.  _build_lds_v_schedule omits these from offset field.
    _V_COL_GROUP = (N_LDS_V_PER_MSB // 2) * 32  # 64 bytes (V_HDIM=128)
    _V_D_HALF    = V_HDIM * KV_BPP // 2         # 128 bytes
    _V_MSB_EXTRA = [0, _V_D_HALF, _V_COL_GROUP, _V_D_HALF + _V_COL_GROUP]

    v_addrs = []
    for msb in range(NUM_MSB):
        extra = _V_MSB_EXTRA[msb]
        if extra == 0:
            v_dh0_b, v_dh1_b = v_dh0, v_dh1
        else:
            v_dh0_b = _raw(arith.addi(v_dh0, arith.constant(extra, type=T.i32)))
            v_dh1_b = _raw(arith.addi(v_dh1, arith.constant(extra, type=T.i32)))
        v_addrs += [set_vgpr_bank(v_dh0_b, msb), set_vgpr_bank(v_dh1_b, msb)]

    return [
        # K addresses [0..3]: each is a distinct SSA value → one bank hint each
        set_vgpr_bank(k_dh0,    0),
        set_vgpr_bank(k_dh1,    1),
        set_vgpr_bank(k_dh0_hi, 2),
        set_vgpr_bank(k_dh1_hi, 3),
        # V addresses [4..11]: 8 distinct SSA values, two per MSB bank
    ] + v_addrs


# ============================================================================
# TDM Priming — Load full tile_n=128 of K+V into LDS
# ============================================================================

def _build_tdm_descs(dg1, addr_i64, stride_adv_i64,
                     lds_base, su_p_size, n_su):
    """Build per-SU TDM descriptors [(dg0, dg1)] without issuing loads.

    dg1 can be a single v8i32 (shared across SUs) or a list of n_su v8i32.
    """
    from flydsl._mlir.dialects import arith as std_arith
    pred = arith.constant(1, type=T.i32)
    cur_addr = addr_i64
    _dg1_list = dg1 if isinstance(dg1, list) else [dg1] * n_su
    descs = []
    for su in range(n_su):
        lds_off = _raw(arith.addi(lds_base, arith.constant(su * su_p_size, type=T.i32)))
        addr_lo, addr_hi = _split_i64_to_lo_hi(cur_addr)
        dg0 = vector.from_elements(T.vec(4, T.i32), [pred, lds_off, addr_lo, addr_hi])
        descs.append((dg0, _dg1_list[su]))
        if su < n_su - 1:
            cur_addr = std_arith.AddIOp(cur_addr, stride_adv_i64).result
    return descs


def _issue_tdm_from_descs(descs):
    """Issue TDM loads from pre-built descriptors, with per-SU barriers."""
    for dg0, dg1 in descs:
        tdm_ops.tensor_load_2d(tdm_ops.TDMDescriptor2D(dg0, dg1))
        rocdl.s_barrier_signal(-1)
        rocdl.s_barrier_wait(-1)


def _tdm_load_kv_blk(kv_type, dg1, addr_i64, stride_adv_i64,
                      lds_base, su_p_size, n_su):
    """Issue n_su TDM loads for one blk of K or V data.

    Each TDM load covers one SU. After all loads, issues barrier.
    """
    descs = _build_tdm_descs(dg1, addr_i64, stride_adv_i64,
                             lds_base, su_p_size, n_su)
    _issue_tdm_from_descs(descs)


def _per_warp_oob_dim1(total_rows_i32, wave_id, rows_per_warp=8):
    """Compute per-warp OOB tensor_dim1 = clamp(total_rows - wave_id*rows, 0, rows)."""
    from flydsl._mlir.dialects import arith as std_arith
    wave_off = std_arith.MulIOp(
        _raw(wave_id), _raw(arith.constant(rows_per_warp, type=T.i32))).result
    remaining = std_arith.SubIOp(total_rows_i32, wave_off).result
    clamped_lo = std_arith.MaxSIOp(
        remaining, _raw(arith.constant(0, type=T.i32))).result
    return std_arith.MinSIOp(
        clamped_lo, _raw(arith.constant(rows_per_warp, type=T.i32))).result


def _make_kv_dg1_with_oob(config_bf16, dim0_elems, dim1_rows,
                          stride_seq_elems, oob_dim1_raw):
    """Build TDM dgroup1 with runtime tensor_dim1 for OOB.

    oob_dim1_raw: raw i32 MLIR Value — per-warp clamped valid rows.
    """
    from flydsl._mlir.dialects import arith as _da
    _i32 = ir.IntegerType.get_signless(32)
    _td1_lo = _da.andi(oob_dim1_raw,
        _da.constant(_i32, value=ir.IntegerAttr.get(_i32, 0xFFFF)))
    _sgpr2 = _da.shli(_td1_lo,
        _da.constant(_i32, value=ir.IntegerAttr.get(_i32, 16)))
    return vector.from_elements(
        T.vec(8, T.i32),
        [arith.constant(config_bf16, type=T.i32),
         arith.constant(dim0_elems << 16, type=T.i32),
         _sgpr2,
         arith.constant(dim0_elems << 16, type=T.i32),
         arith.constant(dim1_rows, type=T.i32),
         stride_seq_elems,
         arith.constant(0, type=T.i32),
         arith.constant(0, type=T.i32)])


def _tdm_load_k_only(
    ptr_K, k_offset,
    stride_k_seq, stride_k_32,
    wave_id, lds_base_i32,
    oob_dg1_list=None,
):
    """Load one tile_n=128 of K into LDS via TDM (K only, no V).

    oob_dg1_list: pre-built list of CNT_SU dg1 vectors with OOB tensor_dim1.
                  Computed once per block and reused. None = full tile (no OOB).
    """
    from flydsl._mlir.dialects import arith as std_arith
    from flydsl.expr.arith import _to_raw

    i64 = ir.IntegerType.get_signless(64)

    _DIM0_ELEMS = 200
    _DIM1_ROWS = 8
    _K_CONFIG_BF16 = (1 << 16) | _K_TDM_CONFIG
    stride_k_seq_elems = _to_raw(arith.shrui(
        stride_k_seq, arith.constant(1, type=T.i32)))

    k_dg1 = oob_dg1_list if oob_dg1_list is not None else vector.from_elements(
        T.vec(8, T.i32),
        [arith.constant(_K_CONFIG_BF16, type=T.i32),
         arith.constant(_DIM0_ELEMS << 16, type=T.i32),
         arith.constant(_DIM1_ROWS << 16, type=T.i32),
         arith.constant(_DIM0_ELEMS << 16, type=T.i32),
         arith.constant(_DIM1_ROWS, type=T.i32),
         stride_k_seq_elems,
         arith.constant(0, type=T.i32),
         arith.constant(0, type=T.i32)])

    # Per-warp global offset: wave_id * 8 * stride_k_seq bytes
    k_addr = _compute_k_global_addr(ptr_K, k_offset, wave_id,
                                     _to_raw(arith.muli(
                                         arith.constant(8, type=T.i32),
                                         stride_k_seq)))

    # Per-warp LDS offset: wave_id * 8 * K_ROW_BYTES
    wid_i32 = _to_raw(wave_id)
    lds_warp_off = std_arith.MulIOp(
        wid_i32,
        _raw(arith.constant(8 * K_ROW_BYTES, type=T.i32))).result
    lds_base_with_warp = std_arith.AddIOp(lds_base_i32, lds_warp_off).result

    k_stride_adv = std_arith.ExtSIOp(i64, _to_raw(stride_k_32)).result

    _tdm_load_kv_blk(KV_K, k_dg1, k_addr, k_stride_adv,
                      lds_base_with_warp, LDS_K_SU_P_SIZE, CNT_SU)

    rocdl.s_wait_tensorcnt(0)
    rocdl.s_barrier_signal(-1)
    rocdl.s_barrier_wait(-1)


def _tdm_load_v_only(
    ptr_V, v_offset,
    stride_v_seq, stride_v_32,
    wave_id, lds_base_i32,
    oob_dg1_list=None,
):
    """Load one tile_n=128 of V into LDS via TDM (V only, no K).

    oob_dg1_list: pre-built list of CNT_SU dg1 vectors with OOB tensor_dim1.
                  None = full tile (no OOB).
    """
    from flydsl._mlir.dialects import arith as std_arith
    from flydsl.expr.arith import _to_raw

    i32 = ir.IntegerType.get_signless(32)
    i64 = ir.IntegerType.get_signless(64)

    _V_CONFIG_BF16 = (1 << 16) | _V_TDM_CONFIG
    _DIM0_ELEMS = 128
    _DIM1_ROWS = 8
    stride_v_seq_elems = _to_raw(arith.shrui(
        stride_v_seq, arith.constant(1, type=T.i32)))

    v_dg1 = oob_dg1_list if oob_dg1_list is not None else vector.from_elements(
        T.vec(8, T.i32),
        [arith.constant(_V_CONFIG_BF16, type=T.i32),
         arith.constant(_DIM0_ELEMS << 16, type=T.i32),
         arith.constant(_DIM1_ROWS << 16, type=T.i32),
         arith.constant(_DIM0_ELEMS << 16, type=T.i32),
         arith.constant(_DIM1_ROWS, type=T.i32),
         stride_v_seq_elems,
         arith.constant(0, type=T.i32),
         arith.constant(0, type=T.i32)])

    v_addr = _compute_v_global_addr(ptr_V, v_offset, wave_id,
                                     _to_raw(arith.muli(
                                         arith.constant(8, type=T.i32),
                                         stride_v_seq)))

    wid_i32 = _to_raw(wave_id)
    lds_warp_off = std_arith.MulIOp(
        wid_i32,
        _raw(arith.constant(8 * V_ROW_BYTES, type=T.i32))).result
    lds_base_with_warp = std_arith.AddIOp(lds_base_i32, lds_warp_off).result

    v_stride_adv = std_arith.ExtSIOp(i64, _to_raw(stride_v_32)).result

    _tdm_load_kv_blk(KV_V, v_dg1, v_addr, v_stride_adv,
                      lds_base_with_warp, LDS_V_SU_P_SIZE, CNT_SU)

    rocdl.s_wait_tensorcnt(0)
    rocdl.s_barrier_signal(-1)
    rocdl.s_barrier_wait(-1)


def _manual_load_k_to_lds(
    ptr_K, k_offset_raw, stride_k_seq,
    lane_id, wave_id, lds_base_i32,
):
    """Load K tile (4 SUs × 32 rows × K_ROW_BYTES) to LDS via flat_load + ds_write.

    Replaces tensor_load_to_lds for FFM-lite which doesn't support TDM.
    128 threads cooperate: each iteration covers BLOCK_SIZE chunks of 16 bytes.
    N_CHUNKS_PER_ROW = QK_HDIM * KV_BPP / 16 (24 for QK_HDIM=192).
    LDS row stride = K_ROW_BYTES (400 for QK_HDIM=192, includes 16B padding).
    """
    from flydsl._mlir.dialects import arith as std_arith
    from flydsl._mlir.dialects import fly as _fly_d

    i32 = ir.IntegerType.get_signless(32)
    i64 = ir.IntegerType.get_signless(64)
    v4i32_ty = ir.VectorType.get([4], i32)
    glb_ptr_type = ir.Type.parse("!llvm.ptr<1>")
    lds_ptr_type = ir.Type.parse("!llvm.ptr<3>")

    a_raw = ptr_K.__extract_to_ir_values__()[0]
    glb_ptr = _fly_d.extract_aligned_pointer_as_index(glb_ptr_type, a_raw)
    base_i64 = llvm_dialect.ptrtoint(i64, glb_ptr)
    k_off_i64 = std_arith.ExtSIOp(i64, k_offset_raw).result
    k_base_i64 = std_arith.AddIOp(base_i64, k_off_i64).result

    tid = std_arith.AddIOp(
        std_arith.MulIOp(
            _raw(wave_id),
            _raw(arith.constant(WAVE_SIZE, type=T.i32))).result,
        _raw(lane_id)).result
    stride_i64 = std_arith.ExtSIOp(i64, _raw(stride_k_seq)).result
    c16 = _raw(arith.constant(16, type=T.i32))
    k_row_c = _raw(arith.constant(K_ROW_BYTES, type=T.i32))

    N_CHUNKS_PER_ROW = (QK_HDIM * KV_BPP) // 16
    c_chunks = _raw(arith.constant(N_CHUNKS_PER_ROW, type=T.i32))
    TOTAL_CHUNKS = SU_K_N * N_CHUNKS_PER_ROW
    N_ITERS = (TOTAL_CHUNKS + BLOCK_SIZE - 1) // BLOCK_SIZE

    for su in fx.range_constexpr(CNT_SU):
        su_row_c = _raw(arith.constant(su * SU_K_N, type=T.i32))
        su_lds_c = _raw(arith.constant(su * LDS_K_SU_P_SIZE, type=T.i32))
        for j in fx.range_constexpr(N_ITERS):
            p = std_arith.AddIOp(
                tid, _raw(arith.constant(j * BLOCK_SIZE, type=T.i32))).result
            row = std_arith.DivUIOp(p, c_chunks).result
            chunk = std_arith.RemUIOp(p, c_chunks).result

            global_row = std_arith.AddIOp(su_row_c, row).result
            g_row_off = std_arith.MulIOp(
                std_arith.ExtSIOp(i64, global_row).result, stride_i64).result
            g_chunk_off = std_arith.ExtSIOp(
                i64, std_arith.MulIOp(chunk, c16).result).result
            g_addr = std_arith.AddIOp(
                k_base_i64,
                std_arith.AddIOp(g_row_off, g_chunk_off).result).result
            gptr = llvm_dialect.inttoptr(glb_ptr_type, g_addr)
            data = llvm_dialect.load(v4i32_ty, gptr)

            lds_row_off = std_arith.MulIOp(row, k_row_c).result
            lds_chunk_off = std_arith.MulIOp(chunk, c16).result
            lds_off = std_arith.AddIOp(
                su_lds_c,
                std_arith.AddIOp(lds_row_off, lds_chunk_off).result).result
            lds_addr = std_arith.AddIOp(lds_base_i32, lds_off).result
            lptr = llvm_dialect.inttoptr(lds_ptr_type, lds_addr)
            llvm_dialect.store(data, lptr)

    _atom_s_wait_dscnt(0)
    rocdl.s_barrier_signal(-1)
    rocdl.s_barrier_wait(-1)


def _manual_load_v_to_lds(
    ptr_V, v_offset_raw, stride_v_seq,
    lane_id, wave_id, lds_base_i32,
):
    """Load V tile (4 SUs × 32 rows × 256B) to LDS via flat_load + ds_write.

    Same as K but with V_ROW_BYTES (288) row stride and LDS_V_SU_P_SIZE.
    """
    from flydsl._mlir.dialects import arith as std_arith
    from flydsl._mlir.dialects import fly as _fly_d

    i32 = ir.IntegerType.get_signless(32)
    i64 = ir.IntegerType.get_signless(64)
    v4i32_ty = ir.VectorType.get([4], i32)
    glb_ptr_type = ir.Type.parse("!llvm.ptr<1>")
    lds_ptr_type = ir.Type.parse("!llvm.ptr<3>")

    a_raw = ptr_V.__extract_to_ir_values__()[0]
    glb_ptr = _fly_d.extract_aligned_pointer_as_index(glb_ptr_type, a_raw)
    base_i64 = llvm_dialect.ptrtoint(i64, glb_ptr)
    v_off_i64 = std_arith.ExtSIOp(i64, v_offset_raw).result
    v_base_i64 = std_arith.AddIOp(base_i64, v_off_i64).result

    tid = std_arith.AddIOp(
        std_arith.MulIOp(
            _raw(wave_id),
            _raw(arith.constant(WAVE_SIZE, type=T.i32))).result,
        _raw(lane_id)).result
    stride_i64 = std_arith.ExtSIOp(i64, _raw(stride_v_seq)).result
    c16 = _raw(arith.constant(16, type=T.i32))
    c4_shift = _raw(arith.constant(4, type=T.i32))
    c15_mask = _raw(arith.constant(15, type=T.i32))
    v_row_c = _raw(arith.constant(V_ROW_BYTES, type=T.i32))

    for su in fx.range_constexpr(CNT_SU):
        su_row_c = _raw(arith.constant(su * SU_K_N, type=T.i32))
        su_lds_c = _raw(arith.constant(su * LDS_V_SU_P_SIZE, type=T.i32))
        for j in fx.range_constexpr(4):
            p = std_arith.AddIOp(
                tid, _raw(arith.constant(j * BLOCK_SIZE, type=T.i32))).result
            row = std_arith.ShRUIOp(p, c4_shift).result
            chunk = std_arith.AndIOp(p, c15_mask).result

            global_row = std_arith.AddIOp(su_row_c, row).result
            g_row_off = std_arith.MulIOp(
                std_arith.ExtSIOp(i64, global_row).result, stride_i64).result
            g_chunk_off = std_arith.ExtSIOp(
                i64, std_arith.MulIOp(chunk, c16).result).result
            g_addr = std_arith.AddIOp(
                v_base_i64,
                std_arith.AddIOp(g_row_off, g_chunk_off).result).result
            gptr = llvm_dialect.inttoptr(glb_ptr_type, g_addr)
            data = llvm_dialect.load(v4i32_ty, gptr)

            lds_row_off = std_arith.MulIOp(row, v_row_c).result
            lds_chunk_off = std_arith.MulIOp(chunk, c16).result
            lds_off = std_arith.AddIOp(
                su_lds_c,
                std_arith.AddIOp(lds_row_off, lds_chunk_off).result).result
            lds_addr = std_arith.AddIOp(lds_base_i32, lds_off).result
            lptr = llvm_dialect.inttoptr(lds_ptr_type, lds_addr)
            llvm_dialect.store(data, lptr)

    _atom_s_wait_dscnt(0)
    rocdl.s_barrier_signal(-1)
    rocdl.s_barrier_wait(-1)


def _tdm_prime_full_tile(
    ptr_K, ptr_V,
    k_offset, v_offset,
    stride_k_seq, stride_v_seq,
    stride_k_32, stride_v_32,
    wave_id,
    k_lds_base_i32, v_lds_base_i32,
):
    """Load one full tile_n=128 of K and V into LDS via TDM.

    K: 4 TDM loads (SU 0-3). V: 4 TDM loads (SU 0-3).
    Blocking — waits for all TDM to complete before returning.
    k_lds_base_i32/v_lds_base_i32: i32 LDS base addresses from SmemAllocator.
    """
    from flydsl._mlir.dialects import arith as std_arith
    from flydsl.expr.arith import _to_raw

    i64 = ir.IntegerType.get_signless(64)

    k_dg1 = _build_tdm_dgroup1(_K_TDM_CONFIG, stride_k_seq)
    v_dg1 = _build_tdm_dgroup1(_V_TDM_CONFIG, stride_v_seq)

    k_addr = _compute_k_global_addr(ptr_K, k_offset, wave_id, stride_k_32)
    v_addr = _compute_v_global_addr(ptr_V, v_offset, wave_id, stride_v_32)

    k_stride_adv = std_arith.ExtSIOp(i64, _to_raw(stride_k_32)).result
    v_stride_adv = std_arith.ExtSIOp(i64, _to_raw(stride_v_32)).result

    # K: 4 SUs (tile_n=128)
    _tdm_load_kv_blk(KV_K, k_dg1, k_addr, k_stride_adv,
                      k_lds_base_i32, LDS_K_SU_P_SIZE, CNT_SU)

    # V: 4 SUs
    _tdm_load_kv_blk(KV_V, v_dg1, v_addr, v_stride_adv,
                      v_lds_base_i32, LDS_V_SU_P_SIZE, CNT_SU)

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
    from fmha_core_loop_gfx1250 import _atom_ds_load_b128

    su_off = (blk * CNT_SU + su) * LDS_K_SU_P_SIZE
    kv_raw = [[None] * N_LDS_PER_MSB for _ in range(NUM_MSB)]
    for msb in range(NUM_MSB):
        for v_idx in range(N_LDS_PER_MSB):
            offset = v_idx * 32 + su_off
            kv_raw[msb][v_idx] = _atom_ds_load_b128(
                ty, kv_lds_addrs[msb], offset, msb)
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
def compile_fmha_fwd(*, is_causal: bool = False):
    """Compile FMHA kernel variant. Cached per is_causal value."""
    IS_CAUSAL = int(is_causal)

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
        gqa: fx.Int32,
        max_seqlen_q: fx.Int32,
        max_seqlen_k: fx.Int32,
    ):
        """BF16 FMHA Forward — full kernel with dynamic KV loop.
    
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
        from flydsl.expr.arith import _to_raw
        from flydsl._mlir.dialects import arith as _mlir_arith
    
        scalar_f = _to_raw(scalar_f)
        stride_q_seq = _to_raw(stride_q_seq)
        stride_k_seq = _to_raw(stride_k_seq)
        stride_v_seq = _to_raw(stride_v_seq)
        stride_o_seq = _to_raw(stride_o_seq)
        gqa = _to_raw(gqa)
        max_seqlen_q_raw = _to_raw(max_seqlen_q)
        max_seqlen_k_raw = _to_raw(max_seqlen_k)
    
        ty = _get_types()
    
        # ================================================================
        # SECTION 1: Prologue — HW Setup + Q Load + Address Gen
        # ================================================================
    
        _setreg(2074, 2)   # WAVE_SCHED_MODE = 2
        _rocdl.s_nop(0)
    
        tx = arith.index_cast(T.i32, gpu.thread_id("x"))
        lane_id = arith.andi(tx, arith.constant(31, type=T.i32))
        wave_id = arith.shrui(tx, arith.constant(5, type=T.i32))
    
        # bx = arith.index_cast(T.i32, gpu.block_id("y"))
        # by = arith.index_cast(T.i32, gpu.block_id("x"))
        # bz = arith.index_cast(T.i32, gpu.block_id("z"))
    
    
        # Grid layout: [B, num_m, H] — batch on x, m-block on y, head on z.
        # XCD remap: reorder wgids so same-(batch, head) tiles are XCD-local.
        #
        # new_wgid = batch + B*head + (B*H)*tile
        #
        # Hardware assigns XCD = new_wgid % 8. When B*H is divisible by 8,
        # all tiles of the same (batch, head) land on the same XCD → K/V cache
        # locality. For any B*H the mapping is a bijection over [0, total-1].
        _raw_bx = arith.index_cast(T.i32, gpu.block_id("x"))   # raw batch
        _raw_by = arith.index_cast(T.i32, gpu.block_id("y"))   # raw m-block
        _raw_bz = arith.index_cast(T.i32, gpu.block_id("z"))   # raw head
        _gdx = _to_raw(gpu.grid_dim.x)      # B
        _gdy = _to_raw(gpu.grid_dim.y)      # M (num_tg)
        _gdz = _to_raw(gpu.grid_dim.z)      # H
        _num_pairs = arith.muli(_gdx, _gdz)  # B * H
        # new_wgid = batch + B*head + (B*H)*tile
        _new_wgid = arith.addi(
            arith.addi(_raw_bx, arith.muli(_gdx, _raw_bz)),  # batch + B*head
            arith.muli(_num_pairs, _raw_by))                   # + (B*H)*tile
        # decompose → (bz=batch, bx=tile, by=head)
        _pair_idx = arith.remui(_new_wgid, _num_pairs)
        bz = arith.remui(_pair_idx, _gdx)          # batch
        by = arith.divui(_pair_idx, _gdx)          # head
        bx = arith.divui(_new_wgid, _num_pairs)    # m-block (tile)
    
        m_start = arith.muli(bx, arith.constant(TILE_N, type=T.i32))
        m_start_raw = _to_raw(m_start)
    
        # ================================================================
        # Per-batch seq_len from cu_seqlens (THD varlen layout)
        # ================================================================
        # THD: Q[total_q, nheads_q, D_qk], K[total_k, nheads_k, D_qk], ...
        # cu_seqlens_q/k: [batch+1] i32 — cumulative token counts.
        # Load via global pointer + llvm.load (scalar i32).
        _i32_ty = ir.IntegerType.get_signless(32)
        _i64_ty = ir.IntegerType.get_signless(64)
        _glb_ptr_ty = ir.Type.parse("!llvm.ptr<1>")
        from flydsl._mlir.dialects import fly as _fly_cu
    
        def _load_cu_seqlen(ptr_tensor, idx_i32):
            """Load cu_seqlens[idx] as scalar i32 via global pointer."""
            _raw_t = ptr_tensor.__extract_to_ir_values__()[0]
            _base_ptr = _fly_cu.extract_aligned_pointer_as_index(_glb_ptr_ty, _raw_t)
            _base_i64 = llvm_dialect.ptrtoint(_i64_ty, _base_ptr)
            _byte_off = _mlir_arith.MulIOp(idx_i32, _to_raw(arith.constant(4, type=T.i32))).result
            _byte_off_64 = _mlir_arith.ExtSIOp(_i64_ty, _byte_off).result
            _addr_i64 = _mlir_arith.AddIOp(_base_i64, _byte_off_64).result
            _addr_ptr = llvm_dialect.inttoptr(_glb_ptr_ty, _addr_i64)
            return llvm_dialect.load(_i32_ty, _addr_ptr)
    
        _bz_raw = _to_raw(bz)
        _bz1_raw = _mlir_arith.AddIOp(_bz_raw, _to_raw(arith.constant(1, type=T.i32))).result
        q_start_tok = _load_cu_seqlen(ptr_cu_seqlens_q, _bz_raw)
        q_end_tok = _load_cu_seqlen(ptr_cu_seqlens_q, _bz1_raw)
        k_start_tok = _load_cu_seqlen(ptr_cu_seqlens_k, _bz_raw)
        k_end_tok = _load_cu_seqlen(ptr_cu_seqlens_k, _bz1_raw)
        actual_q_len = _mlir_arith.SubIOp(q_end_tok, q_start_tok).result
        actual_kv_len = _mlir_arith.SubIOp(k_end_tok, k_start_tok).result

        # ================================================================
        # Pre-compute per-block OOB dg1 lists — constant for this workgroup.
        # Computed once here; reused by all TDM loads without rebuilding.
        #
        # K OOB: actual_kv_len rows valid from token k_start_tok.
        # V OOB: same as K (kv layout is identical).
        # O OOB: actual_q_len - m_start rows valid for this WG's Q tile.
        # ================================================================
        _stride_k_elems_oob = _to_raw(arith.shrui(
            stride_k_seq, arith.constant(1, type=T.i32)))
        _stride_v_elems_oob = _to_raw(arith.shrui(
            stride_v_seq, arith.constant(1, type=T.i32)))

        _K_CFG_OOB = (1 << 16) | _K_TDM_CONFIG
        _V_CFG_OOB = (1 << 16) | _V_TDM_CONFIG

        # per-SU K OOB dg1: tensor_dim1 = clamp(actual_kv_len - su*32 - wave*8, 0, 8)
        _k_oob_dg1 = [
            _make_kv_dg1_with_oob(
                _K_CFG_OOB, 200, 8, _stride_k_elems_oob,
                _per_warp_oob_dim1(
                    _mlir_arith.SubIOp(
                        actual_kv_len,
                        _to_raw(arith.constant(_su * 32, type=T.i32))).result,
                    wave_id, 8))
            for _su in range(CNT_SU)
        ]
        # per-SU V OOB dg1: for tile 0 (full kv_len)
        _v_oob_dg1 = [
            _make_kv_dg1_with_oob(
                _V_CFG_OOB, 128, 8, _stride_v_elems_oob,
                _per_warp_oob_dim1(
                    _mlir_arith.SubIOp(
                        actual_kv_len,
                        _to_raw(arith.constant(_su * 32, type=T.i32))).result,
                    wave_id, 8))
            for _su in range(CNT_SU)
        ]
        # per-SU K OOB dg1 for tile 1 prefetch: kv_remain = actual_kv_len - TILE_N
        _kv_remain_t1_raw = _mlir_arith.SubIOp(
            actual_kv_len, _to_raw(arith.constant(TILE_N, type=T.i32))).result
        _k_tile1_oob_dg1 = [
            _make_kv_dg1_with_oob(
                _K_CFG_OOB, 200, 8, _stride_k_elems_oob,
                _per_warp_oob_dim1(
                    _mlir_arith.SubIOp(
                        _kv_remain_t1_raw,
                        _to_raw(arith.constant(_su * 32, type=T.i32))).result,
                    wave_id, 8))
            for _su in range(CNT_SU)
        ]
        # O store OOB: per-warp dim1 based on remaining Q rows for this WG
        _q_remain_o = _mlir_arith.SubIOp(actual_q_len, m_start_raw).result
        _o_oob_dim1 = _per_warp_oob_dim1(_q_remain_o, wave_id, 32)

        # Head index (for GQA)
        head_index = _phase5_head_index_div_flydsl(by, gqa)
    
        # ================================================================
        # Q/K/V offsets for THD layout
        # ================================================================
        # Q offset = (q_start_tok + bx*128) * stride_q_seq + head * QK_HDIM * BPP
        _HEAD_BYTES_QK = QK_HDIM * KV_BPP    # compile-time
        _HEAD_BYTES_V = V_HDIM * KV_BPP
        _q_tok_off = _mlir_arith.AddIOp(
            q_start_tok, _to_raw(arith.muli(bx, arith.constant(128, type=T.i32)))).result
        q_offset = _mlir_arith.AddIOp(
            _mlir_arith.MulIOp(_q_tok_off, stride_q_seq).result,
            _mlir_arith.MulIOp(_to_raw(by), _to_raw(arith.constant(_HEAD_BYTES_QK, type=T.i32))).result).result
    
        # Q buffer resource with OOB
        _q_end_byte = _mlir_arith.AddIOp(
            _mlir_arith.MulIOp(q_end_tok, stride_q_seq).result,
            _mlir_arith.MulIOp(_to_raw(by), _to_raw(arith.constant(_HEAD_BYTES_QK, type=T.i32))).result).result
        _q_num_bytes = _mlir_arith.AddIOp(
            _q_end_byte, _to_raw(arith.constant(_HEAD_BYTES_QK, type=T.i32))).result
        q_rsrc = buffer_ops.create_buffer_resource(
            ptr_Q, num_records_bytes=_q_num_bytes)
    
        # Q load
        q_frags_raw = _phase4_q_load_flydsl(
            lane_id, _raw(q_rsrc), stride_q_seq, wave_id,
            q_tile_offset_bytes=q_offset)
        rocdl.sched_barrier(0)
    
        _frags_per_bank = len(q_frags_raw[0])
        q_frags = [[None] * Q_WMMA_PER_MSB for _ in range(NUM_MSB)]
        q_frags[0] = q_frags_raw[0] + q_frags_raw[1] + [None] * (Q_WMMA_PER_MSB - 2 * _frags_per_bank)
        q_frags[2] = q_frags_raw[2] + q_frags_raw[3] + [None] * (Q_WMMA_PER_MSB - 2 * _frags_per_bank)
    
        # K offset = k_start_tok * stride_k_seq + head_index * D_qk * BPP
        k_offset = _mlir_arith.AddIOp(
            _mlir_arith.MulIOp(k_start_tok, stride_k_seq).result,
            _mlir_arith.MulIOp(_to_raw(head_index), _to_raw(arith.constant(_HEAD_BYTES_QK, type=T.i32))).result).result
    
        # V offset = k_start_tok * stride_v_seq + head_index * D_v * BPP
        v_offset = _mlir_arith.AddIOp(
            _mlir_arith.MulIOp(k_start_tok, stride_v_seq).result,
            _mlir_arith.MulIOp(_to_raw(head_index), _to_raw(arith.constant(_HEAD_BYTES_V, type=T.i32))).result).result
    
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
    
        stride_k_32 = _mlir_arith.MulIOp(
            _to_raw(arith.constant(32, type=T.i32)), stride_k_seq).result
        stride_v_32 = _mlir_arith.MulIOp(
            _to_raw(arith.constant(32, type=T.i32)), stride_v_seq).result
    
        # SGPR state: softmax scale
        log2e_val = _mlir_arith.constant(ir.F32Type.get(), 1.4426950408889634)
        scale = _mlir_arith.mulf(log2e_val, scalar_f)
        idx_0 = _to_raw(arith.constant(0, type=T.i32))
        idx_1 = _to_raw(arith.constant(1, type=T.i32))
        v2f32_ty = ty['v2f32']
        pair_undef = llvm_dialect.mlir_undef(v2f32_ty)
        pair_v = llvm_dialect.insertelement(pair_undef, scale, idx_0)
        scale_pair = llvm_dialect.insertelement(pair_v, scale, idx_1)
    
        sgpr_state = {
            's_log2e_scl': scale,
            's_log2e_scl_pair': scale_pair,
        }
    
        # ================================================================
        # SECTION 2: Prologue — Tile 0 QK + Partial Softmax (no PV)
        # ================================================================
        #
        # Init adapted for tile_n=128:
        #   1. TDM K(tile 0, 4 SUs) → wait → QK_pure (64 WMMAs)
        #   2. TDM V(tile 0, 4 SUs) — overlapped with softmax
        #   3. sp_tiles → sp_pairs → softmax PART0+PART1 only (no PART2)
        #   4. TDM K(tile 1) — prefetch K only, V(tile 0) stays in LDS
        #   5. Wait K(tile 1) → LDS K(su=0) — preload for core_loop entry
        #
        # No PV in prologue. PART2 + PV run in core_loop iterations.
    
        # ----------------------------------------------------------------
        # Causal mask helper — branchless, operates on v8f32 sp_tiles
        # ----------------------------------------------------------------
        def _apply_causal_mask(su_sp_tiles, n_start_fx):
            """Apply causal mask to QK result tiles (in-place).
    
            Column-major WMMA layout (gfx1250):
              Q_pos = m_start + wave_id*32 + (msb//2)*16 + lane_id%16
              K_pos = n_start + su*32 + (msb%2)*16 + elem + 8*(lane_id//16)
            Mask element e with -inf when K_pos > Q_pos, i.e. boundary < e.
    
            n_start_fx: FlyDSL-wrapped i32 value.
            """
            _lane_lo = arith.andi(lane_id, arith.constant(15, type=T.i32))
            _lane_hi_x8 = arith.muli(
                arith.shrui(lane_id, arith.constant(4, type=T.i32)),
                arith.constant(8, type=T.i32))
            _wave_x32 = arith.muli(wave_id, arith.constant(32, type=T.i32))
            _base = arith.addi(
                arith.addi(arith.subi(m_start, n_start_fx), _wave_x32),
                arith.subi(_lane_lo, _lane_hi_x8))
    
            _neg_inf_c = arith.constant(float('-inf'), type=T.f32)
    
            for _su in fx.range_constexpr(CNT_SU):
                for _msb in fx.range_constexpr(NUM_MSB):
                    _off = (_msb // 2) * 16 - _su * 32 - (_msb % 2) * 16
                    _bnd_fx = arith.addi(_base, arith.constant(_off, type=T.i32))
                    _v8 = su_sp_tiles[_su][_msb][0]
                    for _e in fx.range_constexpr(8):
                        _e_idx = _to_raw(arith.constant(_e, type=T.i32))
                        _cmp_fx = arith.cmpi(arith.CmpIPredicate.slt,
                                             _bnd_fx, arith.constant(_e, type=T.i32))
                        _elem_raw = llvm_dialect.extractelement(_v8, _e_idx)
                        _elem_fx = arith.bitcast(T.f32, _elem_raw)
                        _mval_fx = arith.select(_cmp_fx, _neg_inf_c, _elem_fx)
                        _v8 = llvm_dialect.insertelement(_v8, _to_raw(_mval_fx), _e_idx)
                    su_sp_tiles[_su][_msb][0] = _v8

        def _apply_kv_oob_mask(su_sp_tiles, kv_remain_raw):
            """Mask QK result columns >= kv_remain with -inf (in-place).

            Column-major WMMA layout (gfx1250):
              K_col_in_tile = su*32 + (msb%2)*16 + elem + 8*(lane_id>>4)
            Mask elem e when K_col_in_tile >= kv_remain:
              boundary = kv_remain - 1 - col_base - lane_hi*8
              slt(boundary, e) → e > boundary → K_col >= kv_remain → mask.
            If kv_remain >= 128, boundary >= 7 → nothing masked (no-op).

            kv_remain_raw: raw i32 MLIR Value.
            """
            from flydsl._mlir.dialects import arith as _da
            _i32 = ir.IntegerType.get_signless(32)
            _f32 = ir.F32Type.get()

            def _c(v):
                return _da.constant(_i32, value=ir.IntegerAttr.get(_i32, v))

            _lane_hi = _da.shrui(_to_raw(lane_id), _c(4))
            _lane_hi_x8 = _da.muli(_lane_hi, _c(8))
            # base = kv_remain - 1 - lane_hi_x8
            _base = _da.subi(_da.subi(kv_remain_raw, _c(1)), _lane_hi_x8)
            _neg_inf = _da.constant(_f32, value=ir.FloatAttr.get(_f32, float('-inf')))

            for _su in fx.range_constexpr(CNT_SU):
                for _msb in fx.range_constexpr(NUM_MSB):
                    _col_base_val = _su * 32 + (_msb % 2) * 16
                    _bnd = _da.subi(_base, _c(_col_base_val))
                    _v8 = su_sp_tiles[_su][_msb][0]
                    for _e in fx.range_constexpr(8):
                        _ev = _c(_e)
                        _cmp = _da.cmpi(_da.CmpIPredicate.slt, _bnd, _ev)
                        _elem = llvm_dialect.extractelement(_v8, _ev)
                        _mval = _da.select(_cmp, _neg_inf, _elem)
                        _v8 = llvm_dialect.insertelement(_v8, _mval, _ev)
                    su_sp_tiles[_su][_msb][0] = _v8

        zero_f32 = _to_raw(arith.constant(0.0, type=T.f32))
        neg_inf = _to_raw(arith.constant(float('-inf'), type=T.f32))
        zero_v8f32 = _to_raw(arith.constant_vector(0.0, T.vec(8, T.f32)))
    
        # -- 2a: Load K(tile 0) → K_a (ping buffer) --
        rocdl.sched_barrier(0)
        _tdm_load_k_only(
            ptr_K, k_offset, stride_k_seq, stride_k_32,
            wave_id, k_a_base_i32,
            oob_dg1_list=_k_oob_dg1)
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
    
        # # DBG: Prologue GEMM1 (QK) — lane 0, wave 0, block 0
        # _dbg_pg1_l0 = arith.cmpi(arith.CmpIPredicate.eq, lane_id, arith.constant(0, type=T.i32))
        # _dbg_pg1_w0 = arith.cmpi(arith.CmpIPredicate.eq, wave_id, arith.constant(0, type=T.i32))
        # _dbg_pg1_b0 = arith.cmpi(arith.CmpIPredicate.eq, bx, arith.constant(0, type=T.i32))
        # _dbg_pg1_cond = arith.andi(arith.andi(_dbg_pg1_l0, _dbg_pg1_w0), _dbg_pg1_b0)
        # if _dbg_pg1_cond:
        #     for _pg1su in fx.range_constexpr(CNT_SU):
        #         for _pg1m in fx.range_constexpr(NUM_MSB):
        #             for _pg1e in range(8):
        #                 _pg1v = llvm_dialect.extractelement(all_su_sp_tiles[_pg1su][_pg1m][0], _raw(arith.constant(_pg1e, type=T.i32)))
        #                 fx.printf(f"PRO_G1 l=0 w=0 su={_pg1su} m={_pg1m} e={_pg1e} v={{}}\n", _pg1v)
    
        # -- 2c: Load V(tile 0) → V_a (ping buffer) --
        _tdm_load_v_only(
            ptr_V, v_offset, stride_v_seq, stride_v_32,
            wave_id, v_a_base_i32,
            oob_dg1_list=_v_oob_dg1)
    
        # -- 2d': causal mask on prologue tile (n_start=0) --
        if const_expr(IS_CAUSAL):
            _cm_n = arith.constant(0, type=T.i32)
            _cm_lane_lo = arith.andi(lane_id, arith.constant(15, type=T.i32))
            _cm_lane_hi_x8 = arith.muli(
                arith.shrui(lane_id, arith.constant(4, type=T.i32)),
                arith.constant(8, type=T.i32))
            _cm_wave_x32 = arith.muli(wave_id, arith.constant(32, type=T.i32))
            _cm_base = arith.addi(
                arith.addi(arith.subi(m_start, _cm_n), _cm_wave_x32),
                arith.subi(_cm_lane_lo, _cm_lane_hi_x8))
            _cm_neg_inf = arith.constant(float('-inf'), type=T.f32)
            for _cm_su in fx.range_constexpr(CNT_SU):
                for _cm_msb in fx.range_constexpr(NUM_MSB):
                    _cm_off = (_cm_msb // 2) * 16 - _cm_su * 32 - (_cm_msb % 2) * 16
                    _cm_bnd = arith.addi(_cm_base, arith.constant(_cm_off, type=T.i32))
                    _cm_v8 = all_su_sp_tiles[_cm_su][_cm_msb][0]
                    for _cm_e in fx.range_constexpr(8):
                        _cm_ec = _to_raw(arith.constant(_cm_e, type=T.i32))
                        _cm_cmp = arith.cmpi(arith.CmpIPredicate.slt,
                                             _cm_bnd, arith.constant(_cm_e, type=T.i32))
                        _cm_elem = llvm_dialect.extractelement(_cm_v8, _cm_ec)
                        _cm_sel = arith.select(_cm_cmp, _cm_neg_inf,
                                               arith.bitcast(T.f32, _cm_elem))
                        _cm_v8 = llvm_dialect.insertelement(_cm_v8, _to_raw(_cm_sel), _cm_ec)
                    all_su_sp_tiles[_cm_su][_cm_msb][0] = _cm_v8
    
        # -- 2d'': KV OOB mask on prologue tile (no-op when actual_kv_len >= 128) --
        _apply_kv_oob_mask(all_su_sp_tiles, actual_kv_len)

        # -- 2d: sp_tiles → sp_pairs --
        sp_pairs_all_pro = _sp_tiles_to_sp_pairs(all_su_sp_tiles)
    
        # -- 2e: Softmax PART0+PART1 only (no PART2, matches init stages 0..3) --
        # Pin each MSB's scalar state to its own VGPR bank to match the
        # full-bank (0x00/0x55/0xAA/0xFF) MSB allocation pattern.
        softmax_state_pro = {
            'old_max': [set_vgpr_bank(neg_inf, m) for m in range(NUM_MSB)],
            'local_max': [set_vgpr_bank(neg_inf, m) for m in range(NUM_MSB)],
            'delta': [set_vgpr_bank(zero_f32, m) for m in range(NUM_MSB)],
            'exp_delta': [None] * NUM_MSB,
            'cur_max_log2e': [None] * NUM_MSB,
            'cur_max_log2e_1': [None] * NUM_MSB,
            'cur_max_log2e_scalar': [None] * NUM_MSB,
            'cur_max_log2e_dup': [None] * NUM_MSB,
            'vgpr_log2e_scl_pair': [None] * NUM_MSB,
            'exp_delta_dup': [None] * NUM_MSB,
            'row_sums': [set_vgpr_bank(zero_f32, m) for m in range(NUM_MSB)],
            'p_bf16': [[], [], [], []],
            'sp_pairs_prev': sp_pairs_all_pro,
        }
        _softmax_part01_only(
            ty, 0, sp_pairs_all_pro, softmax_state_pro, sgpr_state)
    
        # -- 2e': PART2 first half for tile 0 (mirrors prologue stages 0-3) --
        # Runs ops 0..PART2_SPLIT-1: setup(7)+pkfma(16)+pair_exp(8) = 31 ops/MSB.
        # 4 MSBs × 8 pair_exp = 32 pair_exp + 4 exp_delta (unavoidable pipeline overhead).
        # sp_pairs_all_pro[m][0..15] are partially modified (pkfma+8-exp applied).
        # pro_exp_delta seeds the O-rescale iter_arg for the first core_loop iteration.
        _pro_part2_ops = _build_all_softmax_part2_ops(
            ty, 0, sp_pairs_all_pro, softmax_state_pro, sgpr_state)
        for _m in fx.range_constexpr(NUM_MSB):
            for _op in _pro_part2_ops[_m][:PART2_SPLIT]:
                _op()
    
        # -- 2f: Prefetch K(tile 1) → K_b if available --
        tile_n_const = _to_raw(arith.constant(TILE_N, type=T.i32))
        if const_expr(IS_CAUSAL):
            num_tiles = _mlir_arith.addi(
                _to_raw(bx), _to_raw(arith.constant(1, type=T.i32)))
        else:
            _kv_plus = _mlir_arith.AddIOp(
                actual_kv_len,
                _to_raw(arith.constant(TILE_N - 1, type=T.i32))).result
            num_tiles = _mlir_arith.divui(_kv_plus, tile_n_const)
        num_tiles_idx = arith.index_cast(T.index, num_tiles)
        # Loop runs N-2 iterations (tiles 1..N-2); endtile handled in epilogue.
        _one_i32_loop = _to_raw(arith.constant(1, type=T.i32))
        num_tiles_minus1 = _mlir_arith.subi(num_tiles, _one_i32_loop)
        num_tiles_minus1_idx = arith.index_cast(T.index, num_tiles_minus1)
    
        # Load K(tile 1) → K_b for core_loop first iteration.
        # For num_tiles=1 (128x128), K_b is never used (loop runs 0 iterations).
        rocdl.sched_barrier(0)
        _k_tile1_stride = _mlir_arith.muli(tile_n_const, stride_k_seq)
        _k_tile1_offset = _mlir_arith.addi(_to_raw(k_offset), _k_tile1_stride)
        _tdm_load_k_only(
            ptr_K, _k_tile1_offset, stride_k_seq, stride_k_32,
            wave_id, k_b_base_i32,
            oob_dg1_list=_k_tile1_oob_dg1)
        rocdl.sched_barrier(0)
    
        # -- 2g: Load K(su=0) from K_b for core_loop entry --
        kv_tiles_init = _load_initial_kv_tiles(ty, kv_lds_addrs_b, blk=0, su=0)
    
        # Prologue results: PART0+PART1+PART2 first half done.
        # old_max = local_max (set by PART2 setup op0); row_sums rescaled (×exp_delta).
        pro_old_max = [softmax_state_pro['old_max'][m]
                       for m in fx.range_constexpr(NUM_MSB)]
        pro_row_sums = [softmax_state_pro['row_sums'][m]
                        for m in fx.range_constexpr(NUM_MSB)]
        pro_local_max = [softmax_state_pro['local_max'][m]
                         for m in fx.range_constexpr(NUM_MSB)]
        pro_delta = [softmax_state_pro['delta'][m]
                     for m in fx.range_constexpr(NUM_MSB)]
    
        # Partial sp_pairs after first half: yield as separate lo+hi f32 scalars.
        # Prologue runs pair_exp sequentially (correct v2f32), so extractelement is safe here.
        pro_partial_sp_lo_flat = []
        pro_partial_sp_hi_flat = []
        for _m in fx.range_constexpr(NUM_MSB):
            for _i in fx.range_constexpr(N_SP_PAIRS):
                pro_partial_sp_lo_flat.append(llvm_dialect.extractelement(
                    sp_pairs_all_pro[_m][_i], _raw(arith.constant(0, type=T.i32))))
                pro_partial_sp_hi_flat.append(llvm_dialect.extractelement(
                    sp_pairs_all_pro[_m][_i], _raw(arith.constant(1, type=T.i32))))
        pro_partial_sp_flat = pro_partial_sp_lo_flat + pro_partial_sp_hi_flat
        # exp_delta from PART2 setup (used by first core_loop iteration's O rescale).
        pro_exp_delta = [softmax_state_pro['exp_delta'][_m]
                         for _m in fx.range_constexpr(NUM_MSB)]
    
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
            q_tiles,        # [4 msb][Q_WMMA_PER_MSB] v16bf16 — Q data
            kv_tiles,       # [4 msb][2] v16bf16 — paired K tiles for WMMA
            sp_tiles,       # [4 msb][1] v8f32 — QK accumulators
            o_tiles,        # [4 d_msb][N_PV_WMMA_N] v8f32 — O accumulators (or None)
            kv_lds_addrs,   # [8] i32 — [K_cur[0:4] + V_cur[4:8]] LDS addresses
            tdm_state,      # TDM SGPR descriptors
            softmax_state,  # Softmax state (old_max, local_max, row_sums, delta, etc.)
            sgpr_state,     # SGPR references (s_log2e_scl, etc.)
            gemm2=True,     # Whether to run GEMM2 stages
            tdm_v_offset=None,  # V global offset for TDM (current tile → next V buf)
            tdm_k_offset=None,  # K global offset for TDM (next tile → next K buf)
            tdm_k_target=None,  # i32 LDS base for TDM K writes (next K buf)
            tdm_v_target=None,  # i32 LDS base for TDM V writes (next V buf)
            kv_lds_addrs_next=None,  # [8] [K_next[0:4] + V_next[4:8]] for K reload
            gemm1_tdm_is_v=False,  # False=main-loop(GEMM1 loads K) True=epilogue(GEMM1 loads V)
            causal_n_start=None,   # i32 n_start for causal mask (None = skip)
            endtile_v_dg1=None,    # pre-built OOB dg1 list for GEMM1 V TDM (endtile only)
            kv_oob_cols=None,      # raw i32: valid K columns in this tile (None = all 128)
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
            _atom_s_wait_dscnt(LDS_INST_COUNT // 2)   # s_wait_dscnt 0x8
    
            v_tiles_out = None
            blk = 0
    
            # ================================================================
            # GEMM1 (QK): 4 stages (SU 0..3)
            # Interleave PART2 on sp_pairs_prev during GEMM1 stages.
            # ================================================================
            sp_pairs_all = softmax_state.get('sp_pairs_prev', None)
            if const_expr(sp_pairs_all is None):
                sp_pairs_all = [[None] * N_SP_PAIRS for _ in range(NUM_MSB)]
    
            # DBG: print sp_pairs_prev pairs 0-3 of MSBs 2,3 at core_loop entry
            # if const_expr(sp_pairs_all[2][0] is not None):
                # _spp_w0 = arith.cmpi(arith.CmpIPredicate.eq, wave_id, arith.constant(0, type=T.i32))
                # _spp_c0 = arith.andi(arith.cmpi(arith.CmpIPredicate.eq, lane_id, arith.constant(0, type=T.i32)), _spp_w0)
                # for _spm in [2, 3]:
                    # for _spp in [0, 1, 2, 3]:
                        # _spp_lo = llvm_dialect.extractelement(sp_pairs_all[_spm][_spp], _raw(arith.constant(0, type=T.i32)))
                        # _spp_hi = llvm_dialect.extractelement(sp_pairs_all[_spm][_spp], _raw(arith.constant(1, type=T.i32)))
                        # if _spp_c0:
                            # fx.printf(f"SPP m={_spm} p={_spp} lo={{}}\n", _spp_lo)
                            # fx.printf(f"SPP m={_spm} p={_spp} hi={{}}\n", _spp_hi)
    
            softmax_ops_by_msb = _build_all_softmax_part2_ops(
                ty, 0, sp_pairs_all, softmax_state, sgpr_state)
            # Second half starts at PART2_SPLIT; first half ran in previous GEMM2.
            softmax_idx_by_msb = [PART2_SPLIT] * NUM_MSB
    
            # Pre-run setup ops 0..PART2_SETUP_A-2 (skip last = row_sums rescale, already applied).
            # Sets cur_max_log2e (needed by pkfma), exp_delta, old_max, etc.
            # Row_sums is NOT re-rescaled here — ia_row_sums already carries the
            # correctly-rescaled value from the previous GEMM2's PART2 first half.
            for _m in fx.range_constexpr(NUM_MSB):
                for _i in fx.range_constexpr(PART2_SETUP_A - 1):  # ops 0..5, skip op 6
                    softmax_ops_by_msb[_m][_i]()
    
            # Build TDM descriptors for GEMM1 stages 0-1.
            # Main-loop (gemm1_tdm_is_v=False): GEMM1 loads K(i+1) -> K_next
            # Epilogue (gemm1_tdm_is_v=True):  GEMM1 loads V(endtile) -> V_next
            has_tdm_k_g1 = (not gemm1_tdm_is_v) and (tdm_k_offset is not None)
            has_tdm_v_g1 = gemm1_tdm_is_v and (tdm_v_offset is not None)
    
            if has_tdm_k_g1:
                from flydsl._mlir.dialects import arith as _mlir_arith
                from flydsl.expr.arith import _to_raw
                i64 = ir.IntegerType.get_signless(64)
                _K_CFG = (1 << 16) | _K_TDM_CONFIG
                _stride_k_elems = _to_raw(arith.shrui(
                    stride_k_seq, arith.constant(1, type=T.i32)))
                k_dg1 = vector.from_elements(
                    T.vec(8, T.i32),
                    [arith.constant(_K_CFG, type=T.i32),
                     arith.constant(200 << 16, type=T.i32),  # dim0=200, LDS stride=400B
                     arith.constant(8 << 16, type=T.i32),
                     arith.constant(200 << 16, type=T.i32),  # tile_dim0=200
                     arith.constant(8, type=T.i32),
                     _stride_k_elems,
                     arith.constant(0, type=T.i32),
                     arith.constant(0, type=T.i32)])
                k_addr = _compute_k_global_addr(
                    ptr_K, tdm_k_offset, wave_id,
                    _to_raw(arith.muli(
                        arith.constant(8, type=T.i32), stride_k_seq)))
                k_stride_adv = _mlir_arith.ExtSIOp(
                    i64, _to_raw(stride_k_32)).result
                _wid_k = _to_raw(wave_id)
                _k_warp_off = _mlir_arith.MulIOp(
                    _wid_k,
                    _raw(arith.constant(8 * K_ROW_BYTES, type=T.i32))).result
                _k_lds_base = _mlir_arith.AddIOp(tdm_k_target, _k_warp_off).result
                k_descs = _build_tdm_descs(
                    k_dg1, k_addr, k_stride_adv,
                    _k_lds_base, LDS_K_SU_P_SIZE, CNT_SU)
                tdm_state['k_descs'] = k_descs
                tdm_state['k_desc_idx'] = 0
    
            if has_tdm_v_g1:
                from flydsl._mlir.dialects import arith as _mlir_arith
                from flydsl.expr.arith import _to_raw
                i64 = ir.IntegerType.get_signless(64)
                _V_CFG = (1 << 16) | _V_TDM_CONFIG
                _stride_v_elems = _to_raw(arith.shrui(
                    stride_v_seq, arith.constant(1, type=T.i32)))
                v_dg1 = endtile_v_dg1 if endtile_v_dg1 is not None else vector.from_elements(
                    T.vec(8, T.i32),
                    [arith.constant(_V_CFG, type=T.i32),
                     arith.constant(128 << 16, type=T.i32),
                     arith.constant(8 << 16, type=T.i32),
                     arith.constant(128 << 16, type=T.i32),
                     arith.constant(8, type=T.i32),
                     _stride_v_elems,
                     arith.constant(0, type=T.i32),
                     arith.constant(0, type=T.i32)])
                v_addr = _compute_v_global_addr(
                    ptr_V, tdm_v_offset, wave_id,
                    _to_raw(arith.muli(
                        arith.constant(8, type=T.i32), stride_v_seq)))
                v_stride_adv = _mlir_arith.ExtSIOp(
                    i64, _to_raw(stride_v_32)).result
                _wid_v = _to_raw(wave_id)
                _v_warp_off = _mlir_arith.MulIOp(
                    _wid_v,
                    _raw(arith.constant(8 * V_ROW_BYTES, type=T.i32))).result
                _v_lds_base = _mlir_arith.AddIOp(tdm_v_target, _v_warp_off).result
                v_descs = _build_tdm_descs(
                    v_dg1, v_addr, v_stride_adv,
                    _v_lds_base, LDS_V_SU_P_SIZE, CNT_SU)
                tdm_state['v_descs'] = v_descs
                tdm_state['v_desc_idx'] = 0
    
            _g1_tdm_type = (KV_K if has_tdm_k_g1 else KV_V if has_tdm_v_g1 else KV_NONE)
            stage_configs = [
                (0, _g1_tdm_type, KV_K, blk, 1),
                (1, _g1_tdm_type, KV_K, blk, 2),
                (2, KV_NONE,      KV_K, blk, 3),
                (3, KV_NONE,      KV_V, blk, 0),
            ]
    
            su_sp_tiles_list = []
    
            # O rescale: all 4 MSBs across GEMM1 stages 0-3.
            # Each stage dispatches N_PV_WMMA_N closures (1 tile per MSB, 4 MSBs total
            # = N_PV_WMMA_N * NUM_MSB closures per stage), running at the last N WMMAs.
            # This removes O_resc from GEMM2 stage 0 entirely — exp+O_resc+cvt
            # interleave throughout GEMM1 (QK) stages.
            #
            # Stage assignment: stage s handles tile n=s for all 4 MSBs.
            #   stage 0 → n=0 (all MSBs), stage 1 → n=1, stage 2 → n=2, stage 3 → n=3
            from flydsl._mlir.dialects import arith as _rescale_arith
            # Build broadcast v8f32 for each MSB's exp_delta
            _ed_v8 = []
            for _dm in fx.range_constexpr(NUM_MSB):
                _edv = llvm_dialect.mlir_undef(ty['v8f32'])
                for _ii in fx.range_constexpr(8):
                    _edv = llvm_dialect.insertelement(
                        _edv, ia_exp_delta[_dm],
                        _raw(arith.constant(_ii, type=T.i32)))
                _ed_v8.append(_edv)
            # _o_rescale_by_stage[s] = list of closures for GEMM1 stage s
            _o_rescale_by_stage = []
            for _s in fx.range_constexpr(N_PV_WMMA_N):      # s = tile index = stage index
                _stage_closures = []
                for _dm in fx.range_constexpr(NUM_MSB):      # 4 closures per stage
                    def _mk_rescale(dm=_dm, nn=_s, ev8=_ed_v8[_dm]):
                        def _op():
                            o_tiles[dm][nn] = _rescale_arith.mulf(o_tiles[dm][nn], ev8)
                        return _op
                    _stage_closures.append(_mk_rescale())
                _o_rescale_by_stage.append(_stage_closures)
    
            for stage_idx, (g_su, t_type, l_type, l_blk, l_su) in \
                    enumerate(stage_configs):
    
                _n_lds = N_LDS_V_PER_MSB if l_type == KV_V else N_LDS_PER_MSB
                kv_tiles_next_raw = [
                    [None] * _n_lds for _ in range(NUM_MSB)]
    
                softmax_stage = (stage_idx + 4) % ALU_STAGES
                budget_per_msb = ALU_PER_STAGE[softmax_stage] // NUM_MSB
                # MSBs 0,1: dispatch ops during GEMM1 stages (register pressure manageable).
                # MSBs 2,3: budget=0 → all ops 32+ run in sequential inter-GEMM flush instead,
                #           avoiding WMMA-induced register collision for banks 2,3.
                softmax_budget = [budget_per_msb, budget_per_msb, 0, 0]
    
                is_barrier_stage = (stage_idx == 2 and (has_tdm_k_g1 or has_tdm_v_g1))
    
                # Distribute O_resc across all 4 GEMM1 stages (1 tile/MSB per stage).
                stage_o_rescale = _o_rescale_by_stage[stage_idx]
    
                sp_tiles, kv_tiles_next_raw = _cl_su_v3_stage(
                    ty, stage_idx,
                    blk, g_su,
                    t_type, blk, g_su,
                    l_type, l_blk, l_su,
                    q_tiles, kv_tiles, sp_tiles,
                    kv_lds_addrs, kv_tiles_next_raw,
                    softmax_ops_by_msb, softmax_idx_by_msb,
                    softmax_budget,
                    tdm_state,
                    tdm_barrier=is_barrier_stage,
                    o_rescale_ops=stage_o_rescale,
                )
    
                su_sp_tiles_list.append([
                    [sp_tiles[msb][0]] for msb in range(NUM_MSB)
                ])
    
                if const_expr(l_type == KV_K):
                    kv_tiles = _pair_k_tiles_for_wmma(kv_tiles_next_raw, ty)
                else:
                    v_tiles_out = kv_tiles_next_raw
    
            # # DBG: GEMM1 (QK) — lane 0, wave 0
            # _dbg_g1_l0 = arith.cmpi(arith.CmpIPredicate.eq, lane_id, arith.constant(0, type=T.i32))
            # _dbg_g1_w0 = arith.cmpi(arith.CmpIPredicate.eq, wave_id, arith.constant(0, type=T.i32))
            # _dbg_g1_b0 = arith.cmpi(arith.CmpIPredicate.eq, bx, arith.constant(0, type=T.i32))
            # _dbg_g1_cond = arith.andi(arith.andi(_dbg_g1_l0, _dbg_g1_w0), _dbg_g1_b0)
            # if _dbg_g1_cond:
            #     for _g1m in fx.range_constexpr(NUM_MSB):
            #         for _g1e in range(8):
            #             _g1v = llvm_dialect.extractelement(sp_tiles[_g1m][0], _raw(arith.constant(_g1e, type=T.i32)))
            #             fx.printf(f"G1 l=0 w=0 m={_g1m} e={_g1e} v={{}}\n", _g1v)
    
            if const_expr(not gemm2):
                return sp_tiles, kv_tiles, o_tiles, su_sp_tiles_list
    
            # ================================================================
            # Between GEMM1 and GEMM2: complete softmax pipeline
            # ================================================================
    
            # 1. Flush any remaining PART2 second-half ops
            for msb in fx.range_constexpr(NUM_MSB):
                for op in softmax_ops_by_msb[msb][softmax_idx_by_msb[msb]:]:
                    op()
    
            # # CMP: GEMM1 output — row_sums after GEMM1, wave 0 lane 0
            # _cg1_w0 = arith.cmpi(arith.CmpIPredicate.eq, wave_id, arith.constant(0, type=T.i32))
            # _cg1_c0 = arith.andi(arith.cmpi(arith.CmpIPredicate.eq, lane_id, arith.constant(0, type=T.i32)), _cg1_w0)
            # for _cg1m in fx.range_constexpr(NUM_MSB):
            #     if _cg1_c0:
            #         fx.printf(f"CG1 m={_cg1m} rs={{}}\n", softmax_state['row_sums'][_cg1m])
    
            # DBG: row_sums after GEMM1 second-half (sum_accum done), before GEMM2 PART0+1 rescale
            # _g1_w0 = arith.cmpi(arith.CmpIPredicate.eq, wave_id, arith.constant(0, type=T.i32))
            # for _g1m in fx.range_constexpr(NUM_MSB):
                # _g1c0 = arith.andi(arith.cmpi(arith.CmpIPredicate.eq, lane_id, arith.constant(0, type=T.i32)), _g1_w0)
                # _g1c16 = arith.andi(arith.cmpi(arith.CmpIPredicate.eq, lane_id, arith.constant(16, type=T.i32)), _g1_w0)
                # if _g1c0:
                    # fx.printf(f"G1_RS l=0 m={_g1m} rs={{}}\n", softmax_state['row_sums'][_g1m])
                # if _g1c16:
                    # fx.printf(f"G1_RS l=16 m={_g1m} rs={{}}\n", softmax_state['row_sums'][_g1m])
    
            # O_resc moved entirely to GEMM1 stages — no O_resc in GEMM2.
            _o_rescale_exp_delta = None
    
            # 2. Masks on current QK tile (before sp_pairs conversion).
            if const_expr(causal_n_start is not None):
                _apply_causal_mask(su_sp_tiles_list, causal_n_start)
            if const_expr(kv_oob_cols is not None):
                _apply_kv_oob_mask(su_sp_tiles_list, kv_oob_cols)

            # 3. Build sp_pairs for current tile (all 4 SUs).
            sp_pairs_current = _sp_tiles_to_sp_pairs(su_sp_tiles_list)
    
            # 4. Build all PART0+PART1+PART2 closures for current tile.
            g2_ops_by_rid, _, g2_sp_lo_cache, g2_sp_hi_cache = _build_all_softmax_gemm2_ops(
                ty, blk, sp_pairs_current, softmax_state, sgpr_state)
            g2_rid_idx = [0] * RLTS_LEN
    
            # 5. Build P tiles from p_bf16 (produced by PART2 second half on prev tile)
            p_tiles_computed = _build_p_tiles_from_softmax(ty, softmax_state)
    
            # ================================================================
            # GEMM2 (PV): 4 stages (SU 0..3)
            # PART0 distributed in two chunks so LLVM interleaves them with WMMAs:
            #   Chunk A (ops 0..10): emitted above → LLVM places in G2-su0 WMMA slots
            #   Chunk B (ops 11..21) + PART1: emitted between G2-su0 and G2-su1
            #                                  → LLVM places in G2-su1 WMMA slots
            # ================================================================
    
            v_tiles_paired = _pair_v_tiles_for_wmma(v_tiles_out, ty)
    
            # Build V TDM descriptors for GEMM2 stages 0-1.
            # Main-loop mode: GEMM2 loads V(i) -> V_next
            has_tdm_v_g2 = (not gemm1_tdm_is_v) and (tdm_v_offset is not None)
            if has_tdm_v_g2:
                from flydsl._mlir.dialects import arith as _mlir_arith
                from flydsl.expr.arith import _to_raw
                i64 = ir.IntegerType.get_signless(64)
                _V_CFG = (1 << 16) | _V_TDM_CONFIG
                _stride_v_elems = _to_raw(arith.shrui(
                    stride_v_seq, arith.constant(1, type=T.i32)))
                v_dg1 = vector.from_elements(
                    T.vec(8, T.i32),
                    [arith.constant(_V_CFG, type=T.i32),
                     arith.constant(128 << 16, type=T.i32),
                     arith.constant(8 << 16, type=T.i32),
                     arith.constant(128 << 16, type=T.i32),
                     arith.constant(8, type=T.i32),
                     _stride_v_elems,
                     arith.constant(0, type=T.i32),
                     arith.constant(0, type=T.i32)])
                v_addr = _compute_v_global_addr(
                    ptr_V, tdm_v_offset, wave_id,
                    _to_raw(arith.muli(
                        arith.constant(8, type=T.i32), stride_v_seq)))
                v_stride_adv = _mlir_arith.ExtSIOp(
                    i64, _to_raw(stride_v_32)).result
                _wid_v = _to_raw(wave_id)
                _v_warp_off = _mlir_arith.MulIOp(
                    _wid_v,
                    _raw(arith.constant(8 * V_ROW_BYTES, type=T.i32))).result
                _v_lds_base = _mlir_arith.AddIOp(tdm_v_target, _v_warp_off).result
                v_descs = _build_tdm_descs(
                    v_dg1, v_addr, v_stride_adv,
                    _v_lds_base, LDS_V_SU_P_SIZE, CNT_SU)
                tdm_state['v_descs'] = v_descs
                tdm_state['v_desc_idx'] = 0
    
            # stage tuple: (gemm_su, lds_type, lds_blk, lds_su, tdm_type, tdm_barrier)
            g2_stage_configs = [
                (0, KV_V, blk, 1, KV_V if has_tdm_v_g2 else KV_NONE, False),
                (1, KV_V, blk, 2, KV_V if has_tdm_v_g2 else KV_NONE, False),
                (2, KV_V, blk, 3, KV_NONE, has_tdm_v_g2),
                (3, KV_K, blk, 0, KV_NONE, False),
            ]
    
            for stage_idx, (g_su, l_type, l_blk, l_su, t_type, barrier) in \
                    enumerate(g2_stage_configs):
    
                p_tiles_su = p_tiles_computed[g_su]
    
                _n_lds = N_LDS_V_PER_MSB if l_type == KV_V else N_LDS_PER_MSB
                kv_tiles_next_raw = [
                    [None] * _n_lds for _ in range(NUM_MSB)]
    
                if const_expr(l_type == KV_K):
                    g2_addrs = kv_lds_addrs_next if kv_lds_addrs_next is not None else kv_lds_addrs
                else:
                    g2_addrs = kv_lds_addrs
    
                o_tiles, kv_tiles_next_raw = _cl_su_v3_stage_gemm2(
                    ty, stage_idx,
                    blk, g_su,
                    l_type, l_blk, l_su,
                    v_tiles_paired, p_tiles_su, o_tiles,
                    g2_addrs, kv_tiles_next_raw,
                    g2_ops_by_rid, g2_rid_idx,
                    tdm_state=tdm_state,
                    tdm_type=t_type,
                    tdm_barrier=barrier,
                    o_rescale_exp_delta=_o_rescale_exp_delta if stage_idx == 0 else None,
                )
    
                if const_expr(l_type == KV_V):
                    v_tiles_paired = _pair_v_tiles_for_wmma(
                        kv_tiles_next_raw, ty)
                else:
                    kv_tiles = _pair_k_tiles_for_wmma(kv_tiles_next_raw, ty)
    
            # # CMP: GEMM2 output — sp_pairs[2][3] and [3][2] hi after GEMM2, wave 0 lane 0
            # _cg2_w0 = arith.cmpi(arith.CmpIPredicate.eq, wave_id, arith.constant(0, type=T.i32))
            # _cg2_c0 = arith.andi(arith.cmpi(arith.CmpIPredicate.eq, lane_id, arith.constant(0, type=T.i32)), _cg2_w0)
            # for _cg2m, _cg2p in [(2, 3), (3, 2), (3, 3)]:
            #     _cg2_hi = llvm_dialect.extractelement(sp_pairs_current[_cg2m][_cg2p], _raw(arith.constant(1, type=T.i32)))
            #     _cg2_lo = llvm_dialect.extractelement(sp_pairs_current[_cg2m][_cg2p], _raw(arith.constant(0, type=T.i32)))
            #     if _cg2_c0:
            #         fx.printf(f"CG2 m={_cg2m} p={_cg2p} lo={{}}\n", _cg2_lo)
            #         fx.printf(f"CG2 m={_cg2m} p={_cg2p} hi={{}}\n", _cg2_hi)
    
            # DBG: GEMM2 (O acc) — lane 0, wave 0
            # _dbg_g2_l0 = arith.cmpi(arith.CmpIPredicate.eq, lane_id, arith.constant(0, type=T.i32))
            # _dbg_g2_w0 = arith.cmpi(arith.CmpIPredicate.eq, wave_id, arith.constant(0, type=T.i32))
            # _dbg_g2_b0 = arith.cmpi(arith.CmpIPredicate.eq, bx, arith.constant(0, type=T.i32))
            # _dbg_g2_cond = arith.andi(arith.andi(_dbg_g2_l0, _dbg_g2_w0), _dbg_g2_b0)
            # if _dbg_g2_cond:
            #     for _g2d in fx.range_constexpr(NUM_MSB):
            #         for _g2n in fx.range_constexpr(N_PV_WMMA_N):
            #             for _g2e in range(8):
            #                 _g2v = llvm_dialect.extractelement(o_tiles[_g2d][_g2n], _raw(arith.constant(_g2e, type=T.i32)))
            #                 fx.printf(f"G2 l=0 w=0 d={_g2d} n={_g2n} e={_g2e} v={{}}\n", _g2v)
    
            rocdl.sched_barrier(0)
    
            # Build partial_sp lo+hi flat lists for yield.
            # For pairs 0-3: use f32 cache from EXP token dispatch (correct scalars).
            # For pairs 4-15: extractelement from sp_pairs_current (pkfma, to be exp'd later).
            partial_sp_lo_out = []
            partial_sp_hi_out = []
            for _psm in fx.range_constexpr(NUM_MSB):
                for _psi in fx.range_constexpr(N_SP_PAIRS):
                    _lo_c = g2_sp_lo_cache[_psm][_psi]
                    _hi_c = g2_sp_hi_cache[_psm][_psi]
                    if const_expr(_lo_c is None):
                        _lo_c = llvm_dialect.extractelement(
                            sp_pairs_current[_psm][_psi], _raw(arith.constant(0, type=T.i32)))
                    if const_expr(_hi_c is None):
                        _hi_c = llvm_dialect.extractelement(
                            sp_pairs_current[_psm][_psi], _raw(arith.constant(1, type=T.i32)))
                    partial_sp_lo_out.append(_lo_c)
                    partial_sp_hi_out.append(_hi_c)
    
            partial_ed_out = [softmax_state['exp_delta'][_m]
                              for _m in fx.range_constexpr(NUM_MSB)]
            return sp_tiles, kv_tiles, o_tiles, su_sp_tiles_list, partial_sp_lo_out, partial_sp_hi_out, partial_ed_out
    
    
    
    
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
        _KV_SIZE = NUM_MSB * N_WMMA_K_TILES   # 12 for 192-dim, 8 for 128-dim
        _OFF_LOCAL_MAX = 24 + _KV_SIZE        # 36 for 192-dim
        _OFF_DELTA     = _OFF_LOCAL_MAX + NUM_MSB
        _OFF_SP        = _OFF_DELTA + NUM_MSB
        _OFF_PP        = _OFF_SP + CNT_SU * NUM_MSB
        _OFF_PSP       = _OFF_PP + 4                  # partial_sp lo: 64 f32
        _PSP_SIZE      = NUM_MSB * N_SP_PAIRS         # = 64 (lo half)
        _OFF_PSP_HI    = _OFF_PSP + _PSP_SIZE         # partial_sp hi: 64 f32
        _OFF_PED       = _OFF_PSP_HI + _PSP_SIZE      # exp_delta: 4 f32
        o_flat_init = [zero_v8f32] * (NUM_MSB * N_PV_WMMA_N)
    
        # Ping-pong bases for iteration 1:
        #   K_cur = K_b (tile 1 K), V_cur = V_a (tile 0 V)
        #   K_next = K_a (TDM K target), V_next = V_b (TDM V target)
        pp_init = [k_b_base_i32, v_a_base_i32,
                   k_a_base_i32, v_b_base_i32]
    
        init_args = (o_flat_init + pro_old_max + pro_row_sums
                     + kv_flat_init + pro_local_max + pro_delta
                     + sp_flat_init + pp_init
                     + pro_partial_sp_flat + pro_exp_delta)
    
        for tile_idx, iter_args, loop_results in scf.for_(
            arith.index(1),
            num_tiles_minus1_idx,   # tiles 1..N-2; endtile handled in epilogue
            arith.index(1),
            iter_args=init_args,
        ):
            # ---- Unpack iter_args ----
            # Pin each MSB's values to bank=d/msb to match the full-bank MSB pattern.
            o_tiles_flat = [iter_args[i] for i in fx.range_constexpr(16)]
            o_tiles = []
            for d in fx.range_constexpr(NUM_MSB):
                row = []
                for n in fx.range_constexpr(N_PV_WMMA_N):
                    row.append(set_vgpr_bank(o_tiles_flat[d * N_PV_WMMA_N + n], d))
                o_tiles.append(row)
    
            ia_old_max = [set_vgpr_bank(iter_args[16 + i], i) for i in fx.range_constexpr(NUM_MSB)]
            ia_row_sums = [set_vgpr_bank(iter_args[20 + i], i) for i in fx.range_constexpr(NUM_MSB)]
    
            kv_tiles_flat = [iter_args[24 + i]
                             for i in fx.range_constexpr(_KV_SIZE)]
            kv_tiles = []
            for msb in fx.range_constexpr(NUM_MSB):
                row = []
                for k in fx.range_constexpr(N_WMMA_K_TILES):
                    row.append(set_vgpr_bank(kv_tiles_flat[msb * N_WMMA_K_TILES + k], msb))
                kv_tiles.append(row)
    
            ia_local_max = [set_vgpr_bank(iter_args[_OFF_LOCAL_MAX + i], i) for i in fx.range_constexpr(NUM_MSB)]
            ia_delta = [set_vgpr_bank(iter_args[_OFF_DELTA + i], i) for i in fx.range_constexpr(NUM_MSB)]
    
            ia_sp_flat = [iter_args[_OFF_SP + i]
                          for i in fx.range_constexpr(CNT_SU * NUM_MSB)]
            prev_su_sp_tiles = []
            for su in fx.range_constexpr(CNT_SU):
                msb_list = []
                for msb in fx.range_constexpr(NUM_MSB):
                    msb_list.append([set_vgpr_bank(ia_sp_flat[su * NUM_MSB + msb], msb)])
                prev_su_sp_tiles.append(msb_list)
    
            # Unpack ping-pong bases
            ia_k_cur_base = iter_args[_OFF_PP]
            ia_v_cur_base = iter_args[_OFF_PP + 1]
            ia_k_next_base = iter_args[_OFF_PP + 2]
            ia_v_next_base = iter_args[_OFF_PP + 3]
    
            # Build per-lane LDS addresses from ping-pong bases
            kv_lds_addrs_cur = _build_kv_lds_addrs(lane_id, ia_k_cur_base, ia_v_cur_base)
            kv_lds_addrs_next = _build_kv_lds_addrs(lane_id, ia_k_next_base, ia_v_next_base)
    
            # ---- Unpack partial_sp_pairs: reconstruct v2f32 from separate lo+hi f32 scalars ----
            # lo values at [_OFF_PSP .. _OFF_PSP+_PSP_SIZE), hi at [_OFF_PSP_HI .. _OFF_PSP_HI+_PSP_SIZE)
            ia_partial_sp_lo = [iter_args[_OFF_PSP + i]     for i in fx.range_constexpr(_PSP_SIZE)]
            ia_partial_sp_hi = [iter_args[_OFF_PSP_HI + i]  for i in fx.range_constexpr(_PSP_SIZE)]
            ia_exp_delta = [set_vgpr_bank(iter_args[_OFF_PED + i], i) for i in fx.range_constexpr(NUM_MSB)]
    
            # Reconstruct v2f32 from f32 scalars in sequential context (no WMMA pressure → correct).
            ia_partial_sp_pairs = []
            for _m in fx.range_constexpr(NUM_MSB):
                msb_pairs = [_make_v2f32(ia_partial_sp_lo[_m * N_SP_PAIRS + _i],
                                         ia_partial_sp_hi[_m * N_SP_PAIRS + _i], _m)
                             for _i in fx.range_constexpr(N_SP_PAIRS)]
                ia_partial_sp_pairs.append(msb_pairs)
    
            # ---- SP tiles: zero accumulators (fresh each iteration) ----
            sp_tiles = []
            for msb in fx.range_constexpr(NUM_MSB):
                sp_tiles.append([set_vgpr_bank(zero_v8f32, msb)])
    
            softmax_state = {
                'old_max': list(ia_old_max),
                'local_max': list(ia_local_max),
                'delta': list(ia_delta),
                'exp_delta': [None] * NUM_MSB,
                'cur_max_log2e': [None] * NUM_MSB,
                'cur_max_log2e_1': [None] * NUM_MSB,
                'cur_max_log2e_scalar': [None] * NUM_MSB,
                'cur_max_log2e_dup': [None] * NUM_MSB,
                'vgpr_log2e_scl_pair': [None] * NUM_MSB,
                'exp_delta_dup': [None] * NUM_MSB,
                'row_sums': list(ia_row_sums),
                'p_bf16': [[], [], [], []],
                'sp_pairs_prev': ia_partial_sp_pairs,  # partial (after PART2 first half)
            }
    
            # ---- Build TDM state (zero placeholder for memload=False) ----
            _zero_i32 = _to_raw(arith.constant(0, type=T.i32))
    
            def _mk_zero_v4i32():
                return vector_dialect.broadcast(ty['v4i32'], _zero_i32)
    
            def _mk_zero_v8i32():
                return vector_dialect.broadcast(ty['v8i32'], _zero_i32)
    
            tdm_state = {
                'v_g0': _mk_zero_v4i32(),
                'v_g1': _mk_zero_v8i32(),
                'k_g0': _mk_zero_v4i32(),
                'k_g1': _mk_zero_v8i32(),
                'v_salu_queue': [],
                'k_salu_queue': [],
            }
    
            # ---- Compute TDM offsets ----
            # GEMM1 loads K(tile_idx+1) -> K_next (stage 2 wait + stage 3 ds_load)
            # GEMM2 loads V(tile_idx) -> V_next (stage 2 wait)
            tile_idx_i32 = arith.index_cast(T.i32, tile_idx)
    
            tile_n_stride_v = _mlir_arith.muli(tile_n_const, stride_v_seq)
            cur_v_advance = _mlir_arith.muli(tile_idx_i32, tile_n_stride_v)
            cur_v_offset = _mlir_arith.addi(_to_raw(v_offset), cur_v_advance)
    
            next_tile = _mlir_arith.addi(
                tile_idx_i32, _to_raw(arith.constant(1, type=T.i32)))
            tile_n_stride_k = _mlir_arith.muli(tile_n_const, stride_k_seq)
            next_k_advance = _mlir_arith.muli(next_tile, tile_n_stride_k)
            next_k_offset = _mlir_arith.addi(_to_raw(k_offset), next_k_advance)
    
            # ---- Core loop: GEMM1(QK)+TDM_K + softmax + GEMM2(PV)+TDM_V ----
            sp_out, kv_out, o_tiles, su_sp_tiles_out, _partial_sp_lo_out, _partial_sp_hi_out, _partial_ed_out = _core_loop(
                ty, False,
                q_frags, kv_tiles, sp_tiles,
                o_tiles,
                kv_lds_addrs_cur, tdm_state, softmax_state, sgpr_state,
                gemm2=True,
                tdm_v_offset=cur_v_offset,   # GEMM2 loads V(tile_idx)
                tdm_v_target=ia_v_next_base,
                tdm_k_offset=next_k_offset,  # GEMM1 loads K(tile_idx+1)
                tdm_k_target=ia_k_next_base,
                kv_lds_addrs_next=kv_lds_addrs_next,
                gemm1_tdm_is_v=False,
            )
    
            # ---- Yield updated state with ping-pong swap ----
            new_o = []
            for d in fx.range_constexpr(NUM_MSB):
                for n in fx.range_constexpr(N_PV_WMMA_N):
                    new_o.append(o_tiles[d][n])
    
            new_max = [softmax_state['old_max'][i]
                       for i in fx.range_constexpr(NUM_MSB)]
            new_sums = [softmax_state['row_sums'][i]
                        for i in fx.range_constexpr(NUM_MSB)]
    
    
            kv_out_flat = []
            for msb in fx.range_constexpr(NUM_MSB):
                for k in fx.range_constexpr(N_WMMA_K_TILES):
                    kv_out_flat.append(kv_out[msb][k])
    
            new_local_max = [softmax_state['local_max'][i]
                             for i in fx.range_constexpr(NUM_MSB)]
            new_delta = [softmax_state['delta'][i]
                         for i in fx.range_constexpr(NUM_MSB)]
    
            sp_out_flat = []
            for su in fx.range_constexpr(CNT_SU):
                for msb in fx.range_constexpr(NUM_MSB):
                    sp_out_flat.append(su_sp_tiles_out[su][msb][0])
    
            # Ping-pong swap: next→cur, cur→next
            pp_swapped = [ia_k_next_base, ia_v_next_base,
                          ia_k_cur_base, ia_v_cur_base]
    
            # DBG: print sp_pairs just before yield, wave 0 lane 0
            # _yld_w0 = arith.cmpi(arith.CmpIPredicate.eq, wave_id, arith.constant(0, type=T.i32))
            # _yld_c0 = arith.andi(arith.cmpi(arith.CmpIPredicate.eq, lane_id, arith.constant(0, type=T.i32)), _yld_w0)
            # for _ym in [2, 3]:
                # for _yp in [2, 3]:
                    # _yhi = llvm_dialect.extractelement(_partial_sp_out[_ym][_yp], _raw(arith.constant(1, type=T.i32)))
                    # _ylo = llvm_dialect.extractelement(_partial_sp_out[_ym][_yp], _raw(arith.constant(0, type=T.i32)))
                    # if _yld_c0:
                        # fx.printf(f"YLD m={_ym} p={_yp} lo={{}}\n", _ylo)
                        # fx.printf(f"YLD m={_ym} p={_yp} hi={{}}\n", _yhi)
    
            # Use lo+hi flat lists returned from _core_loop (safe f32 scalars from EXP token cache)
            new_partial_sp_flat = _partial_sp_lo_out + _partial_sp_hi_out
            new_exp_delta = [_partial_ed_out[_m]
                             for _m in fx.range_constexpr(NUM_MSB)]
    
            yield (new_o + new_max + new_sums
                   + kv_out_flat + new_local_max + new_delta
                   + sp_out_flat + pp_swapped
                   + new_partial_sp_flat + new_exp_delta)
    
        # ================================================================
        # SECTION 4: Epilogue — post_process + div_cvt + write_out
        # ================================================================
        #
        # post_process(is_odd):
        #   softmax stages 4..7 (PART2) → complete last tile's softmax
        #   LDS V(su=0..3) → PV_pure (64 WMMAs)
        # div_cvt():
        #   row_sums cross-MSB reduce → LSE = max*scale + log(sum)
        #   O = O * rcp(row_sum) → cvt_pk_bf16
        # lds_store_D_LSE() + TDM_store_D_LSE()
        #
        # For tile_n=128: blk=0 always, no is_odd branching.
    
        # ---- 4a: Unpack loop results ----
        # Pin each MSB's values to bank=d/msb to match the full-bank MSB pattern.
        ep_o_tiles = []
        for d in fx.range_constexpr(NUM_MSB):
            row = []
            for n in fx.range_constexpr(N_PV_WMMA_N):
                row.append(set_vgpr_bank(loop_results[d * N_PV_WMMA_N + n], d))
            ep_o_tiles.append(row)
    
        ep_old_max = [set_vgpr_bank(loop_results[16 + i], i) for i in fx.range_constexpr(NUM_MSB)]
        ep_row_sums = [set_vgpr_bank(loop_results[20 + i], i) for i in fx.range_constexpr(NUM_MSB)]
        ep_local_max = [set_vgpr_bank(loop_results[_OFF_LOCAL_MAX + i], i) for i in fx.range_constexpr(NUM_MSB)]
        ep_delta = [set_vgpr_bank(loop_results[_OFF_DELTA + i], i) for i in fx.range_constexpr(NUM_MSB)]
    
        # Epilogue ping-pong: V_cur after swap has last tile's V data
        ep_k_cur_base = loop_results[_OFF_PP]
        ep_v_cur_base = loop_results[_OFF_PP + 1]
        ep_kv_lds_addrs = _build_kv_lds_addrs(lane_id, ep_k_cur_base, ep_v_cur_base)
    
        # ---- 4b: Unpack partial_sp_pairs: reconstruct v2f32 from separate lo+hi f32 ----
        ep_partial_sp_lo = [loop_results[_OFF_PSP    + i] for i in fx.range_constexpr(_PSP_SIZE)]
        ep_partial_sp_hi = [loop_results[_OFF_PSP_HI + i] for i in fx.range_constexpr(_PSP_SIZE)]
        ep_partial_sp_pairs = []
        for _m in fx.range_constexpr(NUM_MSB):
            ep_pairs = [_make_v2f32(ep_partial_sp_lo[_m * N_SP_PAIRS + _i],
                                    ep_partial_sp_hi[_m * N_SP_PAIRS + _i], _m)
                        for _i in fx.range_constexpr(N_SP_PAIRS)]
            ep_partial_sp_pairs.append(ep_pairs)
    
        # ---- 4b': Extra state for endtile (N>=2) ----
        # K(N-1) fragments from loop (GEMM2 stage 3 loaded from K_next).
        ep_kv_tiles_flat = [loop_results[24 + _i] for _i in fx.range_constexpr(_KV_SIZE)]
        ep_kv_tiles = []
        for _m in fx.range_constexpr(NUM_MSB):
            _row = [set_vgpr_bank(ep_kv_tiles_flat[_m * N_WMMA_K_TILES + _k], _m)
                    for _k in fx.range_constexpr(N_WMMA_K_TILES)]
            ep_kv_tiles.append(_row)
    
        # K_next / V_next LDS bases (for endtile core_loop kv_lds_addrs_next).
        ep_k_next_base = loop_results[_OFF_PP + 2]
        ep_v_next_base = loop_results[_OFF_PP + 3]
        ep_kv_lds_addrs_next = _build_kv_lds_addrs(lane_id, ep_k_next_base, ep_v_next_base)
    
        # V(N-1) global offset for endtile GEMM1 TDM.
        from flydsl._mlir.dialects import arith as _ep_arith_off
        _num_tiles_m1_ep = _ep_arith_off.subi(num_tiles, _to_raw(arith.constant(1, type=T.i32)))
        ep_v_endtile_offset = _ep_arith_off.addi(
            _to_raw(v_offset),
            _ep_arith_off.muli(_ep_arith_off.muli(_num_tiles_m1_ep, tile_n_const), stride_v_seq))
    
        # ia_exp_delta for endtile core_loop: exp_delta from last loop GEMM2.
        ia_exp_delta = [set_vgpr_bank(loop_results[_OFF_PED + _m], _m)
                        for _m in fx.range_constexpr(NUM_MSB)]
    
        # ---- 4c: s_wait_idle + barrier ----
        _asm_void("s_wait_idle")
        rocdl.s_barrier_signal(-1)
        rocdl.s_barrier_wait(-1)
    
    
        # ---- 4c': _ep_finish — PART2 + PV_pure + div_cvt + TDM store D ----
        # o_tiles:           [[v8f32]*N_PV_WMMA_N]*NUM_MSB — accumulated O
        # sp_pairs_in:       [[v2f32]*N_SP_PAIRS]*NUM_MSB  — PART2 first-half input
        # exp_delta_rescale: [f32]*NUM_MSB                 — exp_delta for O rescale before PV
        # v_base_for_pv:     i32                           — V LDS base for PV_pure
        # old_max_in:        [f32]*NUM_MSB                 — max across all tiles seen so far
        # local_max_in:      [f32]*NUM_MSB                 — local max (same as old_max after PART0+1)
        # delta_in:          [f32]*NUM_MSB                 — delta (for PART2 setup)
        # row_sums_in:       [f32]*NUM_MSB                 — row_sums accumulated to this point
        def _ep_finish(o_tiles, sp_pairs_in, exp_delta_rescale, v_base_for_pv,
                       old_max_in, local_max_in, delta_in, row_sums_in):
            sfx = {
                'old_max':          list(old_max_in),
                'local_max':        list(local_max_in),
                'delta':            list(delta_in),
                'exp_delta':        [None] * NUM_MSB,
                'cur_max_log2e':    [None] * NUM_MSB,
                'cur_max_log2e_1':  [None] * NUM_MSB,
                'cur_max_log2e_scalar': [None] * NUM_MSB,
                'cur_max_log2e_dup':[None] * NUM_MSB,
                'vgpr_log2e_scl_pair': [None] * NUM_MSB,
                'exp_delta_dup':    [None] * NUM_MSB,
                'row_sums':         list(row_sums_in),
                'p_bf16':           [[], [], [], []],
                'sp_pairs_prev':    sp_pairs_in,
            }
            # DBG: print row_sums_in at ep_finish entry
            # _epf_w0 = arith.cmpi(arith.CmpIPredicate.eq, wave_id, arith.constant(0, type=T.i32))
            # for _efm in fx.range_constexpr(NUM_MSB):
                # _efcl0 = arith.andi(arith.cmpi(arith.CmpIPredicate.eq, lane_id, arith.constant(0, type=T.i32)), _epf_w0)
                # _efcl16 = arith.andi(arith.cmpi(arith.CmpIPredicate.eq, lane_id, arith.constant(16, type=T.i32)), _epf_w0)
                # if _efcl0:
                    # fx.printf(f"EPF_RS l=0 m={_efm} rs={{}}\n", sfx['row_sums'][_efm])
                # if _efcl16:
                    # fx.printf(f"EPF_RS l=16 m={_efm} rs={{}}\n", sfx['row_sums'][_efm])
    
            # PART2 second half: setup ops + pair_exp+cvt+sum.
            # MSBs 0,1: pairs 0-3 already exp'd by GEMM2 EXP tokens → start from PART2_SPLIT.
            _p2ops = _build_all_softmax_part2_ops(ty, 0, sp_pairs_in, sfx, sgpr_state)
            for _m in fx.range_constexpr(NUM_MSB):
                for _i in fx.range_constexpr(PART2_SETUP_A - 1):  # ops 0..6
                    _p2ops[_m][_i]()
    
            for _m in fx.range_constexpr(NUM_MSB):
                for _op in _p2ops[_m][PART2_SPLIT:]:
                    _op()
    
            p_tiles = _build_p_tiles_from_softmax(ty, sfx)
    
            # O rescale
            from flydsl._mlir.dialects import arith as _fin_a
            _vf8 = ir.VectorType.get([8], ir.F32Type.get())
            for _msb in fx.range_constexpr(NUM_MSB):
                _ed = exp_delta_rescale[_msb]
                _edv8 = llvm_dialect.mlir_undef(_vf8)
                for _i in fx.range_constexpr(8):
                    _edv8 = llvm_dialect.insertelement(
                        _edv8, _ed, _raw(arith.constant(_i, type=T.i32)))
                for _n in fx.range_constexpr(N_PV_WMMA_N):
                    o_tiles[_msb][_n] = _fin_a.mulf(o_tiles[_msb][_n], _edv8)
    
            # PV_pure
            _kv_pv = _build_kv_lds_addrs(lane_id, ep_k_cur_base, v_base_for_pv)
            for _sp in fx.range_constexpr(2):
                _sb = _sp * 2
                _vr0, _vr1 = _load_v_two_sus_from_lds(ty, _kv_pv, 0, _sb, _sb + 1)
                o_tiles = _pv_pure_su(
                    ty, 0, _sb, _pair_v_tiles_for_wmma(_vr0, ty), p_tiles[_sb], o_tiles)
                o_tiles = _pv_pure_su(
                    ty, 0, _sb + 1, _pair_v_tiles_for_wmma(_vr1, ty), p_tiles[_sb + 1], o_tiles)
    
            # # DBG: Epilogue GEMM2 (O acc after PV_pure) — lane 0, wave 0, block 0
            # _dbg_eg2_l0 = arith.cmpi(arith.CmpIPredicate.eq, lane_id, arith.constant(0, type=T.i32))
            # _dbg_eg2_w0 = arith.cmpi(arith.CmpIPredicate.eq, wave_id, arith.constant(0, type=T.i32))
            # _dbg_eg2_b0 = arith.cmpi(arith.CmpIPredicate.eq, bx, arith.constant(0, type=T.i32))
            # _dbg_eg2_cond = arith.andi(arith.andi(_dbg_eg2_l0, _dbg_eg2_w0), _dbg_eg2_b0)
            # if _dbg_eg2_cond:
            #     for _eg2d in fx.range_constexpr(NUM_MSB):
            #         for _eg2n in fx.range_constexpr(N_PV_WMMA_N):
            #             for _eg2e in range(8):
            #                 _eg2v = llvm_dialect.extractelement(o_tiles[_eg2d][_eg2n], _raw(arith.constant(_eg2e, type=T.i32)))
            #                 fx.printf(f"EPI_G2 l=0 w=0 d={_eg2d} n={_eg2n} e={_eg2e} v={{}}\n", _eg2v)
    
            # 4d: div_cvt
            _v8f32 = ir.VectorType.get([8], ir.F32Type.get())
            _v8bf16 = ir.VectorType.get([8], ir.BF16Type.get())
            _rsf = list(sfx['row_sums'])
            _lmf = list(sfx['local_max'])
            for _mb in fx.range_constexpr(0, NUM_MSB, 2):
                _sm = _mlir_arith.addf(_rsf[_mb], _rsf[_mb + 1])
                _slo = _to_raw(arith.constant(0x76543210, type=T.i32))
                _shi = _to_raw(arith.constant(0xfedcba98, type=T.i32))
                _pm = _rocdl.permlanex16(ty['f32'], _sm, _sm, _slo, _shi, False, False)
                _sf = _mlir_arith.addf(_sm, _pm)
                _rsf[_mb] = _sf
                _rsf[_mb + 1] = _sf
            _l2e = _mlir_arith.constant(ir.F32Type.get(), 0.6931471805599453)
            for _msb in fx.range_constexpr(NUM_MSB):
                _mxs = _mlir_arith.mulf(_lmf[_msb], scalar_f)
                _lgs = _rocdl.log(ty['f32'], _rsf[_msb])
                _mlir_arith.addf(_mlir_arith.mulf(_lgs, _l2e), _mxs)  # LSE (not stored)
            from flydsl._mlir.dialects import arith as _fin_a2
            _obf16 = []
            for _msb in fx.range_constexpr(NUM_MSB):
                _rcp = _rocdl.rcp(ty['f32'], _rsf[_msb])
                _rv8 = llvm_dialect.mlir_undef(_v8f32)
                for _i in fx.range_constexpr(8):
                    _rv8 = llvm_dialect.insertelement(
                        _rv8, _rcp, _raw(arith.constant(_i, type=T.i32)))
                _mb16 = []
                for _n in fx.range_constexpr(N_PV_WMMA_N):
                    _mb16.append(_fin_a2.truncf(_v8bf16, _fin_a2.mulf(o_tiles[_msb][_n], _rv8)))
                _obf16.append(_mb16)
    
            # Barrier: ensure all waves finish PV before any wave writes D to V_a LDS.
            # Without this, wave 0 can overwrite V_a SU data still being read by waves 1-3.
            rocdl.s_barrier_signal(-1)
            rocdl.s_barrier_wait(-1)
    
            # 4e: TDM store D (VGPR -> LDS -> Global)
            from flydsl._mlir.dialects import arith as _sa
            _i32t = ir.IntegerType.get_signless(32)
            _ldst = ir.Type.parse("!llvm.ptr<3>")
            _v4i32t = ir.VectorType.get([4], _i32t)
            _db32 = _extract_lds_base_i32(_lds_alloc_v_a.get_base())
            _dw = _sa.AddIOp(_db32,
                _sa.MulIOp(_to_raw(wave_id),
                    _raw(arith.constant(LDS_D_WV_SIZE, type=T.i32))).result).result
            _llo = _raw(arith.andi(lane_id, arith.constant(15, type=T.i32)))
            _lhi = _raw(arith.shrui(lane_id, arith.constant(4, type=T.i32)))
            _loff = _sa.AddIOp(
                _sa.MulIOp(_llo, _raw(arith.constant(TDM_D_TILE_DIM0, type=T.i32))).result,
                _sa.MulIOp(_lhi, _raw(arith.constant(16, type=T.i32))).result).result
            for _msb in fx.range_constexpr(NUM_MSB):
                for _n in fx.range_constexpr(N_PV_WMMA_N):
                    _ioff = ((_msb // 2) * 16 * TDM_D_TILE_DIM0 + (_msb % 2) * 128 + _n * 32)
                    _la = _sa.AddIOp(
                        _sa.AddIOp(_dw, _loff).result,
                        _raw(arith.constant(_ioff, type=T.i32))).result
                    llvm_dialect.store(
                        vector.bitcast(_v4i32t, _obf16[_msb][_n]),
                        llvm_dialect.inttoptr(_ldst, _la), volatile_=True)
            _asm_void("s_wait_dscnt 0x0")
            _wsgpr = rocdl.wave_id()
            _D = 128
            _LOG2_D = 7
            from flydsl._mlir.dialects import arith as _sa2
            from flydsl._mlir.dialects import fly as _fly2
            from flydsl._mlir.dialects import llvm as _llvm2
            _i64t = ir.IntegerType.get_signless(64)
            _glbpt = ir.Type.parse("!llvm.ptr<1>")
            # THD: O token = q_start_tok + bx*128 + wave*32
            _o_tok = _sa2.AddIOp(
                _sa2.AddIOp(q_start_tok,
                    _sa2.MulIOp(bx, _raw(arith.constant(128, type=T.i32))).result).result,
                _sa2.MulIOp(_wsgpr, _raw(arith.constant(WV_SUBQD, type=T.i32))).result).result
            # Element offset = tok * stride_o_seq + head * D_v
            _o_elem_off = _sa2.AddIOp(
                _sa2.MulIOp(_o_tok, stride_o_seq).result,
                _sa2.MulIOp(_to_raw(by), _raw(arith.constant(_D, type=T.i32))).result).result
            _o_raw = ptr_O.__extract_to_ir_values__()[0]
            _o_gp = _fly2.extract_aligned_pointer_as_index(_glbpt, _o_raw)
            _o64 = _llvm2.ptrtoint(_i64t, _o_gp)
            _boff32 = _sa2.MulIOp(_o_elem_off, _raw(arith.constant(2, type=T.i32))).result
            _boff64 = _sa2.ExtSIOp(_i64t, _boff32).result
            _oadr64 = _sa2.AddIOp(_o64, _boff64).result
            _alo, _ahi = _split_i64_to_lo_hi(_oadr64)
            _olds2 = _sa2.AddIOp(
                _extract_lds_base_i32(_lds_alloc_v_a.get_base()),
                _sa2.MulIOp(_wsgpr, _raw(arith.constant(LDS_D_WV_SIZE, type=T.i32))).result).result
            _dg0 = vector.from_elements(T.vec(4, T.i32),
                [_raw(arith.constant(1, type=T.i32)), _olds2, _alo, _ahi])
            _g0 = _raw(arith.constant((1 << 16) | 0, type=T.i32))
            _g1 = _raw(arith.constant((128 & 0xFFFF) << 16, type=T.i32))
            # tensor_dim1 OOB: _o_oob_dim1 = clamp(actual_q_len-m_start-wave*32, 0, 32)
            # Use function-style _da.* ops (same as _make_kv_dg1_with_oob) to ensure emission.
            from flydsl._mlir.dialects import arith as _da_o
            _i32_o = ir.IntegerType.get_signless(32)
            _td1_lo_o = _da_o.andi(_o_oob_dim1,
                _da_o.constant(_i32_o, value=ir.IntegerAttr.get(_i32_o, 0xFFFF)))
            _g2 = _da_o.ori(
                _da_o.shli(_td1_lo_o,
                    _da_o.constant(_i32_o, value=ir.IntegerAttr.get(_i32_o, 16))),
                _da_o.constant(_i32_o, value=ir.IntegerAttr.get(_i32_o, (128>>16)&0xFFFF)))
            _g3 = _raw(arith.constant(((32>>16)&0xFFFF)|((128&0xFFFF)<<16), type=T.i32))
            _g4 = _raw(arith.constant(32 & 0xFFFF, type=T.i32))  # tile_dim1 stays 32
            _g5 = stride_o_seq  # THD: elements per token row
            _g6 = _raw(arith.constant(0, type=T.i32))
            _g7 = _raw(arith.constant(0, type=T.i32))
            _dg1 = vector.from_elements(T.vec(8, T.i32), [_g0,_g1,_g2,_g3,_g4,_g5,_g6,_g7])
            tdm_ops.tensor_store_2d(tdm_ops.TDMDescriptor2D(_dg0, _dg1))
            tdm_ops.tensor_wait(0)
    
        # ---- 4c'': endtile dispatch — if N>=2: core_loop + ep_finish, else: ep_finish ----
        from flydsl._mlir.dialects import arith as _ep_cmp_a
        _two_ep = _ep_cmp_a.ConstantOp(
            ir.IntegerType.get_signless(32),
            ir.IntegerAttr.get(ir.IntegerType.get_signless(32), 2)).result
        _is_multi = _ep_cmp_a.CmpIOp(
            _ep_cmp_a.CmpIPredicate.uge, num_tiles, _two_ep).result
    
        if _is_multi:  # N>=2: endtile core_loop then ep_finish
            # All variables defined fresh inside THEN — not state variables.
            _et_sp_t = [[set_vgpr_bank(zero_v8f32, _m)] for _m in fx.range_constexpr(NUM_MSB)]
            _et_sfx = {
                'old_max':          list(ep_old_max),
                'local_max':        list(ep_local_max),
                'delta':            list(ep_delta),
                'exp_delta':        [None] * NUM_MSB,
                'cur_max_log2e':    [None] * NUM_MSB,
                'cur_max_log2e_1':  [None] * NUM_MSB,
                'cur_max_log2e_scalar': [None] * NUM_MSB,
                'cur_max_log2e_dup':[None] * NUM_MSB,
                'vgpr_log2e_scl_pair': [None] * NUM_MSB,
                'exp_delta_dup':    [None] * NUM_MSB,
                'row_sums':         list(ep_row_sums),
                'p_bf16':           [[], [], [], []],
                'sp_pairs_prev':    [[ep_partial_sp_pairs[_m][_i]
                                      for _i in fx.range_constexpr(N_SP_PAIRS)]
                                     for _m in fx.range_constexpr(NUM_MSB)],
            }
            # DBG: row_sums entering endtile core_loop (= ep_row_sums from loop_results)
            # _etcl_w0 = arith.cmpi(arith.CmpIPredicate.eq, wave_id, arith.constant(0, type=T.i32))
            # for _etm in fx.range_constexpr(NUM_MSB):
                # _etcl0 = arith.andi(arith.cmpi(arith.CmpIPredicate.eq, lane_id, arith.constant(0, type=T.i32)), _etcl_w0)
                # _etcl16 = arith.andi(arith.cmpi(arith.CmpIPredicate.eq, lane_id, arith.constant(16, type=T.i32)), _etcl_w0)
                # if _etcl0:
                    # fx.printf(f"ETCL_RS l=0 m={_etm} rs={{}}\n", ep_row_sums[_etm])
                # if _etcl16:
                    # fx.printf(f"ETCL_RS l=16 m={_etm} rs={{}}\n", ep_row_sums[_etm])
    
            _et_z = _to_raw(arith.constant(0, type=T.i32))
            _et_tdm = {
                'v_g0': vector_dialect.broadcast(ty['v4i32'], _et_z),
                'v_g1': vector_dialect.broadcast(ty['v8i32'], _et_z),
                'k_g0': vector_dialect.broadcast(ty['v4i32'], _et_z),
                'k_g1': vector_dialect.broadcast(ty['v8i32'], _et_z),
                'v_salu_queue': [], 'k_salu_queue': [],
            }
            _et_o = [[ep_o_tiles[_d][_n] for _n in range(N_PV_WMMA_N)]
                     for _d in range(NUM_MSB)]
            _et_causal_ns = None
            if const_expr(IS_CAUSAL):
                _et_causal_ns = arith.muli(
                    arith.subi(arith.index_cast(T.i32, num_tiles_idx),
                               arith.constant(1, type=T.i32)),
                    arith.constant(TILE_N, type=T.i32))
            # Pre-build endtile V OOB dg1: actual_kv_len - (num_tiles-1)*TILE_N
            _et_kv_remain = _mlir_arith.SubIOp(
                actual_kv_len,
                _mlir_arith.MulIOp(
                    _mlir_arith.SubIOp(num_tiles, _to_raw(arith.constant(1, type=T.i32))).result,
                    tile_n_const).result).result
            _stride_v_elems_et = _to_raw(arith.shrui(
                stride_v_seq, arith.constant(1, type=T.i32)))
            _et_v_oob_dg1 = [
                _make_kv_dg1_with_oob(
                    (1 << 16) | _V_TDM_CONFIG, 128, 8, _stride_v_elems_et,
                    _per_warp_oob_dim1(
                        _mlir_arith.SubIOp(
                            _et_kv_remain,
                            _to_raw(arith.constant(_su * 32, type=T.i32))).result,
                        wave_id, 8))
                for _su in range(CNT_SU)
            ]
            _, _, _et_o, _, _et_psp_lo, _et_psp_hi, _et_ped = _core_loop(
                ty, False,
                q_frags, ep_kv_tiles, _et_sp_t,
                _et_o,
                ep_kv_lds_addrs, _et_tdm, _et_sfx, sgpr_state,
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
            # Pass updated softmax state (old_max/local_max/delta/row_sums after PART0+1
            # for the endtile tile) so _ep_finish can correctly run PART2 second half.
            # Reconstruct v2f32 sp_pairs for _ep_finish from safe f32 lo+hi scalars.
            _et_psp = []
            for _rpsm in fx.range_constexpr(NUM_MSB):
                _rpairs = [_make_v2f32(_et_psp_lo[_rpsm * N_SP_PAIRS + _rpi],
                                       _et_psp_hi[_rpsm * N_SP_PAIRS + _rpi], _rpsm)
                           for _rpi in fx.range_constexpr(N_SP_PAIRS)]
                _et_psp.append(_rpairs)
            _ep_finish(
                _et_o, _et_psp, _et_ped, ep_v_next_base,
                _et_sfx['old_max'], _et_sfx['local_max'],
                _et_sfx['delta'], _et_sfx['row_sums'],
            )
        else:  # N=1: original epilogue flow
            _ep_finish(
                [[ep_o_tiles[_d][_n] for _n in range(N_PV_WMMA_N)] for _d in range(NUM_MSB)],
                ep_partial_sp_pairs,
                [loop_results[_OFF_PED + _m] for _m in fx.range_constexpr(NUM_MSB)],
                ep_v_cur_base,
                list(ep_old_max), list(ep_local_max), list(ep_delta), list(ep_row_sums),
            )

    return fmha_fwd_kernel


