"""D128 BF16 FMHA Forward Core Loop — GEMM1 (QK) Only — gfx1250 FlyDSL.

Translates core_loop stages 0-3 (GEMM1/QK) + cl_su_V3 interleaving.
Target: align with BF16_FMHA_FWD_D128_1TG_4W_32mx4_256nx1_cas_brd_rxy.s L1701-~2400.

GEMM2 (PV) stages 4-7 are handled by a separate agent.

Architecture:
  1. Atomic primitives: one instruction per function (no per-atom barriers)
  2. Schedule builders: compile-time lists of WMMA / LDS / softmax ops
  3. cl_su_V3 interleaving engine: dispatches ops following slot tables
  4. core_loop: stages 0-3 (QK) + mask placeholder

Target: gfx1250, wave32, 4 waves (1TG), 1024 shared VGPRs (256 per bank).
"""

from __future__ import annotations

from .fmha_schedule import (
    GEMM1_SCHEDULE,
    GEMM2_SCHEDULE,
    g1_row_idx,
    g2_row_idx,
    PART2_EXP_START,  # ops index where pair_exp starts (=23)
)
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm as llvm_dialect
from flydsl._mlir.dialects import rocdl as rocdl_dialect
from flydsl.expr import arith, rocdl, vector
from flydsl.expr.typing import T
from flydsl.expr.primitive import const_expr, range_constexpr

# run_id offset for PART2 pkfma (rid 5..8 → MSB 0..3)
_P2_BASE = 5
# token base for EXP_Mx (19..22 → MSB 0..3, draws from pair_exp)
_EXP_BASE = 19

# ============================================================================
# Local rocdl primitive wrappers (until merged into upstream FlyDSL)
# ============================================================================


def _rocdl_exp2(res, arg, **kw):
    return rocdl_dialect.exp2(res=res, arg=arith.unwrap(arg), **kw)


def _rocdl_permlanex16(res, old, src0, src1, src2, fi, bound_control, **kw):
    return rocdl_dialect.permlanex16(
        res=res,
        old=arith.unwrap(old),
        src0=arith.unwrap(src0),
        src1=arith.unwrap(src1),
        src2=arith.unwrap(src2),
        fi=fi,
        bound_control=bound_control,
        **kw,
    )


def _rocdl_fmax3(a, b, c):
    m = llvm_dialect.intr_maxnum(arith.unwrap(a), arith.unwrap(b))
    return llvm_dialect.intr_maxnum(m, arith.unwrap(c))


# ============================================================================
# Constants
# ============================================================================

WAVE_SIZE = 32
NUM_WAVES = 4
NUM_MSB = 4
BLOCK_SIZE = WAVE_SIZE * NUM_WAVES  # 128

QK_HDIM = 192
V_HDIM = 128
Q_BPP = 2
KV_BPP = 2
TG_SUBQD = 128
TG_SUBKV = 128
WV_SUBKV = TG_SUBKV
SU_K_N = 32
SU_K_K = QK_HDIM
CNT_SU = WV_SUBKV // SU_K_N  # 4
TDM_LD_CNT = CNT_SU  # 4

COMPUTE_N = 128
COMPUTE_CNT_SU = COMPUTE_N // SU_K_N  # 4

VPS_Q = TG_SUBQD * QK_HDIM * Q_BPP // 128  # 64
VPS_MSB_Q = VPS_Q // NUM_MSB  # 16
VTS_MSB_Q = VPS_MSB_Q  # 16
Q_WMMA_PER_MSB = VTS_MSB_Q // 8  # 2

VPS_KV = SU_K_K * SU_K_N * KV_BPP // 128  # 128
VPS_MSB_KV = VPS_KV // NUM_MSB  # 32

VPS_MSB_SP = 32  # 4 SUs * 8 f32/lane = 32 f32/lane real per MSB (no padding)

WMMA_M, WMMA_N, WMMA_K = 16, 16, 32
SP_MSB_M = 16
SP_MSB_N = 16
SP_MSB_K = QK_HDIM  # 128

GEMM_INST_COUNT = 24  # 2 × (QK_HDIM/WMMA_K=6) × 2 = 24 for QK_HDIM=192
LDS_INST_COUNT = 24  # NUM_MSB × N_LDS_PER_MSB = 4 × 6
N_LDS_PER_MSB = VPS_MSB_KV // 4  # 6 for QK_HDIM=192
N_LDS_V_PER_MSB = 4  # V tile loads per MSB (V_HDIM=128)
LDS_V_INST_COUNT = NUM_MSB * N_LDS_V_PER_MSB  # 16

ALU_STAGES = 8

GEMM1 = 0  # QK
KV_K = 0
KV_V = 1
KV_NONE = 2

LDS_K_SU_P_SIZE = 0x3200  # 4 waves × 8 rows × 400 bytes = 12800
LDS_V_SU_P_SIZE = 0x2400
LDS_K_BLK_SIZE = 0x32000  # 16 × LDS_K_SU_P_SIZE
LDS_PINGPONG_OFF = 0x46000

DSWAIT_OCCUPY_WHOLE_WMMA = 1
WAIT_DSCNT0 = 0
TDM_ALL_WAVES = 1
BARRIER_SIGNAL_AHEAD = 0
TDM_LOADS_PER_STAGE = 2
TDM_WAIT_CNT = TDM_LOADS_PER_STAGE * 2  # total loads across 2 stages
QK_WMMA_INTERLEAVE = 1

# PART2 double-buffer pipeline split (ops per MSB):
#   FIRST HALF  [0..PART2_SPLIT-1] = setup(8) + pkfma(16) + exp(EXP_PER_MSB_TO_G2) ops
#     All pkfmas run in GEMM2. The 16-op gap between each pkfma and its exp provides
#     >4 cycles of transcendental latency hiding without interleaving.
#     8 exp ops per MSB (pairs 0..3, lo+hi) are also folded into GEMM2 first half,
#     moving 32 exp total (4 MSBs × 8) from GEMM1 to GEMM2 for load balance.
#   SECOND HALF [PART2_SPLIT..] = exp(GEMM1_EXP_OPS)+pkadd(8)+cvt(16)+sum(9) = 57 ops
#     cvt and sum-tree interleaved to break s_delay_alu dependency chains.
#     dispatched in GEMM1 stages 0+1+2 (42/MSB/stage × 3 = 126 ≥ 57 ✓)
EXP_PER_MSB_TO_G2 = 8  # pair_exp ops per MSB moved to GEMM2 (pairs 0..3 lo+hi)
# Note: exp_delta (1 exp/MSB) is at op[2] in setup and is unavoidably included in
# GEMM2 first half — it MUST be computed in GEMM2 because partial_ed_out carries
# it as an iter_arg for the next tile's O-rescale. Cannot be deferred to GEMM1.
# So GEMM2 has 4 exp_delta + 32 pair_exp = 36 exp total (4 is necessary overhead).
# total setup ops (save_old_max..broadcast_dup..rescale_sum)
PART2_SETUP_A = 8
PART2_SPLIT = 24 + EXP_PER_MSB_TO_G2  # = 32: setup(8)+pkfma(16)+pair_exp(8)
PART2_G2_SPLIT = PART2_SPLIT  # GEMM2 truncation limit (mirrors PART2_SPLIT)
GEMM1_EXP_OPS = VPS_MSB_SP - EXP_PER_MSB_TO_G2  # = 24: pair_exp remaining in GEMM1

# GEMM2 (PV) softmax budget — stage 0-1 spread PART0, stage 2-3 run PART1+PART2:
#   Stage 0: 40  (10/MSB; O-rescale moved to GEMM1 stages, frees slot for PART0)
#   Stage 1: 52  (13/MSB, finishes PART0 remainder = 11/MSB; PART1 deferred to stage 2)
#   Stage 2: 56  (PART1=8 + PART2 first 12/MSB)
#   Stage 3: 168 (PART2 overflow → interleaves with K ds_loads)
# PART2 first half now has 32 ops/MSB (was 24): budget 4+14+42=60/MSB ≥ 32 ✓.
#
# GEMM1 (QK) PART2 second-half budget (cycle-weighted: exp=3 cycles, others=1 cycle):
#   Second-half = 24 exp (72 cy/MSB) + 33 pkadd/cvt/sum (33 cy/MSB) = 105 cy/MSB.
#   Stages 4-5 throttled to 84 (21/MSB) to spread exp across all 4 stages instead of
#   exhausting in su0-su2. Stage 7 (G1-su3) then receives all 33 cheap ops/MSB,
#   filling its WMMAs that were previously naked (2.3 cy/W → ~6 cy/W target).
# #   Stage 4: 120 → 30/MSB → 10 exp (30 cy) — fills 10/12 lds_done=True phases
#     Stage 5: 120 → 30/MSB → 10 exp
#     Stage 6: 132 → 33/MSB →  4 exp + cheap
#     Stage 7: 132 → 33/MSB →  0 exp + cheap (G1-su3 cheap fill)
#   10 exp/MSB × 2 phases/WMMA = 20 WMMAs with exp + O-rescale at last 4 = all covered.
#   With phase cycle budget = 3: exactly 1 exp per phase, or 3 cheap ops per phase.
ALU_PER_STAGE = [40, 52, 56, 168, 120, 120, 132, 132]

# Cycle budget per VALU phase in GEMM1 dispatch.
# exp_f32 costs 3 cycles, all other VALU (pk_add, cvt, mov) costs 1 cycle.
# Budget=3 → exactly 1 exp op per phase, or up to 3 cheap ops.
GEMM1_VALU_PER_PHASE = 3

N_SP_PAIRS = VPS_MSB_SP // 2  # 16 v2f32 pairs per MSB (compact, no -inf padding)

# GEMM2 PV constants
GEMM2 = 1  # PV
N_V_MSB = 2  # v_msb = d_msb % 2, only 2 V banks needed
N_PV_WMMA_N = 4  # D_MSB_N / WMMA_N = 64 / 16
D_MSB_K = SU_K_N  # 32 (one WMMA-K per stage)
# With compact (no-padding) P layout, each PV WMMA src_b is built by
# concatenating 2 sibling sp_msbs (sp_msb=2*m_tile and 2*m_tile+1) along the
# K_pv (=N_qk) axis, giving a full 16 bf16/lane src_b = full K=32 real.
# Each (d_msb, n) accumulator therefore receives exactly CNT_SU(=4) WMMAs
# across the 4 PV stages, covering tile_n=128 K-reduction (4 * 32 = 128).
PV_K_ITERS = 1  # 1 WMMA per (d_msb, n) per SU
PV_GEMM_INST_COUNT = NUM_MSB * PV_K_ITERS * N_PV_WMMA_N  # 16


# ============================================================================
# Low-level Helpers
# ============================================================================


def _emit_void(inst_str, operands=None, constraints="", **kwargs):
    llvm_dialect.inline_asm(
        None, operands or [], inst_str, constraints, has_side_effects=True, **kwargs
    )


def _emit_result(result_type, inst_str, operands=None, constraints="", **kwargs):
    return llvm_dialect.inline_asm(
        result_type,
        operands or [],
        inst_str,
        constraints,
        has_side_effects=True,
        **kwargs,
    )


def _sched_barrier(mask=0):
    """Emit llvm.amdgcn.sched.barrier.

    Prevents instruction reordering across this point.
    mask=0: barrier for ALL instruction types
    (VMEM, VALU, LDS, SALU, etc.).
    """
    mask_val = arith.unwrap(arith.constant(mask, type=T.i32))
    llvm_dialect.call_intrinsic(None, "llvm.amdgcn.sched.barrier", [mask_val], [], [])


# ============================================================================
# VGPR Bank Hint
# ============================================================================

_USE_BANK_HINTS = False

# Starting physical-register offset (within each bank) reserved for sp_pairs.
# sp_pairs[i] (v2f32) lands at HWIdx = bank*256 + SP_PAIR_BASE + i*2.
#
# Offset selection is driven by per-bank free-range analysis of the ISA:
#   Bank0: saturated (198/256 used), NO contiguous 32-slot range → skip offset hint
#   Bank1: free from offset 174 (V/K tiles occupy 0-173)
#   Bank2: free from offset 127 (→ use 128 for even alignment)
#   Bank3: free from offset 121 (→ use 122 for even alignment)
# Using 174 is safe for all of banks 1-3 (all have free range ≥174).
# Bank0 sp_pairs use only BankHint=0 (no offset constraint) since bank0 is full.
SP_PAIR_BASE = 174

# Per-bank VGPR copy of s_log2e_scl_pair (v2f32) for pk_fma src1.
# Offset 206 = SP_PAIR_BASE(174) + N_SP_PAIRS(16)*2, just past sp_pairs.
LOG2E_PAIR_OFFSET = 206


def set_vgpr_bank(raw_val, bank: int):
    if const_expr(not _USE_BANK_HINTS):
        return raw_val
    val_type = raw_val.type
    bank_val = arith.unwrap(arith.constant(bank, type=T.i32))
    return llvm_dialect.call_intrinsic(
        val_type, "llvm.amdgcn.set.vgpr.bank", [raw_val, bank_val], [], []
    )


def set_vgpr_bank_offset(raw_val, bank: int, offset: int):
    """Pin raw_val to HWIdx = bank*256+offset (single-candidate BankOffsetHint)."""
    if const_expr(not _USE_BANK_HINTS):
        return raw_val
    val_type = raw_val.type
    bank_val = arith.unwrap(arith.constant(bank, type=T.i32))
    offset_val = arith.unwrap(arith.constant(offset, type=T.i32))
    return llvm_dialect.call_intrinsic(
        val_type,
        "llvm.amdgcn.set.vgpr.bank.offset",
        [raw_val, bank_val, offset_val],
        [],
        [],
    )


# ============================================================================
# MLIR Types
# ============================================================================


def _get_types():
    f32_ty = ir.F32Type.get()
    bf16_ty = ir.BF16Type.get()
    i32_ty = ir.IntegerType.get_signless(32)
    return {
        "f32": f32_ty,
        "bf16": bf16_ty,
        "i32": i32_ty,
        "v2f32": ir.VectorType.get([2], f32_ty),
        "v2bf16": ir.VectorType.get([2], bf16_ty),
        "v8f32": ir.VectorType.get([8], f32_ty),
        "v8bf16": ir.VectorType.get([8], bf16_ty),
        "v16bf16": ir.VectorType.get([16], bf16_ty),
        "v4i32": ir.VectorType.get([4], i32_ty),
        "v8i32": ir.VectorType.get([8], i32_ty),
        "lds_ptr": ir.Type.parse("!llvm.ptr<3>"),
    }


# ============================================================================
# V2F32 Helpers (from epilogue)
# ============================================================================


def _make_v2f32(lo, hi, bank=0):
    v2f32_ty = ir.VectorType.get([2], ir.F32Type.get())
    idx_0 = arith.unwrap(arith.constant(0, type=T.i32))
    idx_1 = arith.unwrap(arith.constant(1, type=T.i32))
    undef = llvm_dialect.mlir_undef(v2f32_ty)
    v = llvm_dialect.insertelement(undef, lo, idx_0)
    v = llvm_dialect.insertelement(v, hi, idx_1)
    return set_vgpr_bank(v, bank)


def _split_v2f32(pair):
    idx_0 = arith.unwrap(arith.constant(0, type=T.i32))
    idx_1 = arith.unwrap(arith.constant(1, type=T.i32))
    lo = llvm_dialect.extractelement(pair, idx_0)
    hi = llvm_dialect.extractelement(pair, idx_1)
    return lo, hi


def _broadcast_f32_to_v2f32(val, bank=0):
    return _make_v2f32(val, val, bank)


# ============================================================================
# WMMA Fragment Pairing (from prologue — concat two v4i32 → v16bf16)
# ============================================================================


def make_wmma_frag_bf16(vec4_lo, vec4_hi):
    bf16_ty = ir.BF16Type.get()
    vec8_bf16_ty = ir.VectorType.get([8], bf16_ty)
    v0 = vector.bitcast(vec8_bf16_ty, vec4_lo)
    v1 = vector.bitcast(vec8_bf16_ty, vec4_hi)
    return vector.shuffle(v0, v1, list(range_constexpr(16)))


def _pair_k_tiles_for_wmma(kv_tiles_raw, ty):
    """Pair 4 raw v4i32 K loads per MSB into 2 v16bf16 WMMA fragments.

    kv_tiles_raw: [4 msb][4] v4i32 from ds_load_b128
    Returns: [4 msb][2] v16bf16 ready for WMMA consumption.
    """
    kv_paired = []
    for msb in range_constexpr(NUM_MSB):
        msb_frags = []
        for k in range_constexpr(N_WMMA_K_TILES):
            lo = kv_tiles_raw[msb][k * 2]
            hi = kv_tiles_raw[msb][k * 2 + 1]
            frag = make_wmma_frag_bf16(lo, hi)
            frag = set_vgpr_bank(frag, msb)
            msb_frags.append(frag)
        kv_paired.append(msb_frags)
    return kv_paired


N_WMMA_K_TILES = (QK_HDIM // WMMA_K) // 2  # 3 for QK_HDIM=192, 2 for 128


def _pack_v2bf16_to_v16bf16(ty, v2bf16_list, bank):
    """Pack 8 v2bf16 into one v16bf16 for PV WMMA src_b.

    Uses vector.shuffle to concatenate pairs pairwise until we reach v16bf16.
    """
    # Stage 1: 8 v2bf16 → 4 v4bf16
    v4s = []
    for i in range_constexpr(4):
        v4s.append(
            vector.shuffle(
                v2bf16_list[i * 2], v2bf16_list[i * 2 + 1], list(range_constexpr(4))
            )
        )

    # Stage 2: 4 v4bf16 → 2 v8bf16
    v8s = []
    for i in range_constexpr(2):
        v8s.append(vector.shuffle(v4s[i * 2], v4s[i * 2 + 1], list(range_constexpr(8))))

    # Stage 3: 2 v8bf16 → 1 v16bf16
    result = vector.shuffle(v8s[0], v8s[1], list(range_constexpr(16)))
    return set_vgpr_bank(result, bank)


# ============================================================================
# Atomic Instruction Primitives
# Each emits exactly ONE instruction (scheduling controlled at stage level).
# ============================================================================

# --- WMMA ---


def _atom_wmma_init(ty, src_a, src_b, bank_dst):
    _sched_barrier(0)
    zero = vector.broadcast(ty["v8f32"], arith.unwrap(arith.constant(0.0, type=T.f32)))

    result = rocdl_dialect.wmma_f32_16x16x32_bf16(
        ty["v8f32"],
        src_a,
        src_b,
        zero,
        signA=False,
        signB=False,
        modC=0,
        reuseA=False,
        reuseB=False,
    )
    banked = set_vgpr_bank(result.result, bank_dst)
    _sched_barrier(0)

    return banked


def _atom_wmma_accum(ty, src_a, src_b, acc, bank_dst):

    _sched_barrier(0)
    result = rocdl_dialect.wmma_f32_16x16x32_bf16(
        ty["v8f32"],
        src_a,
        src_b,
        acc,
        signA=False,
        signB=False,
        modC=0,
        reuseA=False,
        reuseB=False,
    )
    banked = set_vgpr_bank(result.result, bank_dst)
    _sched_barrier(0)

    return banked


# --- LDS Loads ---


def _atom_ds_load_b128(ty, addr, offset_val, bank):

    _sched_barrier(0)
    off = arith.unwrap(arith.constant(offset_val, type=T.i32))
    ptr = llvm_dialect.inttoptr(ty["lds_ptr"], arith.unwrap(arith.addi(addr, off)))
    raw = llvm_dialect.load(ty["v4i32"], ptr)
    banked = set_vgpr_bank(raw, bank)
    _sched_barrier(0)

    return banked


def _atom_ds_load_tr16_b128(ty, addr, offset_val, bank):

    _sched_barrier(0)
    off = arith.unwrap(arith.constant(offset_val, type=T.i32))
    ptr = llvm_dialect.inttoptr(ty["lds_ptr"], arith.unwrap(arith.addi(addr, off)))
    raw = rocdl.ds_load_tr16_b128(ty["v8bf16"], ptr)
    banked = set_vgpr_bank(raw, bank)
    _sched_barrier(0)

    return banked


# --- TDM (Tensor Descriptor Memory) ---


def _atom_tdm_load(ty, s_g0, s_g1):

    _sched_barrier(0)
    zero_i32 = arith.unwrap(arith.constant(0, type=T.i32))
    null_v4 = llvm_dialect.mlir_undef(ty["v4i32"])
    for i in range_constexpr(4):
        null_v4 = llvm_dialect.insertelement(
            null_v4, zero_i32, arith.unwrap(arith.constant(i, type=T.i32))
        )
    null_v8 = llvm_dialect.mlir_undef(ty["v8i32"])
    for i in range_constexpr(8):
        null_v8 = llvm_dialect.insertelement(
            null_v8, zero_i32, arith.unwrap(arith.constant(i, type=T.i32))
        )
    rocdl.tensor_load_to_lds(s_g0, s_g1, null_v4, null_v4, null_v8, 0)
    _sched_barrier(0)


def _atom_tdm_addr_inc(s_g0_2, s_inc_bytes):

    _emit_void("s_lshl2_add_u32 $0, $1, $0", [s_g0_2, s_inc_bytes], "=s,s,0")


def _atom_tdm_addr_carry(s_g0_3):

    _emit_void("s_add_co_ci_u32 $0, $0, 0", [s_g0_3], "=s,0")


def _atom_tdm_dim1_dec(s_g1_2, dec_amount):

    _emit_void(f"s_sub_co_i32 $0, $0, {dec_amount}", [s_g1_2], "=s,0")


def _atom_tdm_dim1_clamp(s_g1_2):

    _emit_void("s_max_i32 $0, $0, 0", [s_g1_2], "=s,0")


def _atom_tdm_next_lds_addr(s_lds_part_offset, next_offset):

    result = _emit_result(
        ir.IntegerType.get_signless(32),
        f"s_add_co_i32 $0, $1, {next_offset}",
        [s_lds_part_offset],
        "=s,s",
    )

    return result


def _atom_tdm_set_lds_addr(s_g0, s_addr):

    _emit_void("s_mov_b32 $0, $1", [s_g0, s_addr], "=s,s")


# --- Scalar Softmax VALU (f32) ---
# Use MLIR native ops to give LLVM full visibility
# into register lifetimes, avoiding unnecessary spills to scratch.


def _atom_exp_f32(src, bank):

    _sched_barrier(0)
    f32 = ir.F32Type.get()
    result = _rocdl_exp2(f32, src)
    banked = set_vgpr_bank(result, bank)
    _sched_barrier(0)

    return banked


def _atom_mul_f32(src0, src1, bank):

    _sched_barrier(0)
    result = arith.mulf(src0, src1)
    banked = set_vgpr_bank(result, bank)
    _sched_barrier(0)

    return banked


def _atom_fma_f32(src0, src1, src2, bank):

    _sched_barrier(0)
    result = llvm_dialect.intr_fma(src0, src1, src2)
    banked = set_vgpr_bank(result, bank)
    _sched_barrier(0)

    return banked


def _atom_fma_f32_neg_src0(src0, src1, src2, bank):

    _sched_barrier(0)
    neg_src0 = llvm_dialect.fneg(src0)
    result = llvm_dialect.intr_fma(neg_src0, src1, src2)
    banked = set_vgpr_bank(result, bank)
    _sched_barrier(0)

    return banked


def _atom_mov_b32(src, bank):

    _sched_barrier(0)
    """v_mov_b32 — just a bank-change copy. LLVM inserts the copy itself."""
    banked = set_vgpr_bank(src, bank)
    _sched_barrier(0)

    return banked


def _atom_add_f32(src0, src1, bank):

    _sched_barrier(0)
    r = arith.addf(src0, src1)
    banked = set_vgpr_bank(r, bank)
    _sched_barrier(0)

    return banked


def _atom_max3_num_f32(src0, src1, src2, bank):

    _sched_barrier(0)
    """v_max3_num_f32 via rocdl.fmax3 — native."""
    result = _rocdl_fmax3(src0, src1, src2)
    banked = set_vgpr_bank(result, bank)
    _sched_barrier(0)

    return banked


def _atom_max_num_f32(src0, src1, bank):
    _sched_barrier(0)
    result = llvm_dialect.intr_maxnum(src0, src1)
    banked = set_vgpr_bank(result, bank)
    _sched_barrier(0)
    return banked


def _atom_permlanex16(src, s_sel0, s_sel1, bank):

    _sched_barrier(0)
    i32_ty = ir.IntegerType.get_signless(32)
    f32_ty = ir.F32Type.get()
    src_i32 = llvm_dialect.bitcast(i32_ty, src)
    r = _rocdl_permlanex16(i32_ty, src_i32, src_i32, s_sel0, s_sel1, False, False)
    r_f32 = llvm_dialect.bitcast(f32_ty, r)
    # Use bank+2 so tmps[1] is in a DIFFERENT bank from tmps[0] (which is in `bank`).
    # This prevents LLVM from doing in-place permlanex16 (same VGPR for src and dst),
    # which would cause v_max_num_f32 v,v,v (self-max) in the subsequent max3.
    banked = set_vgpr_bank(r_f32, (bank + 2) % NUM_MSB)
    _sched_barrier(0)

    return banked


# --- Packed Softmax VALU (v2f32) ---


def _atom_pk_fma_f32_neg_c(a, b, c, bank):
    """v_pk_fma_f32 with neg_lo/neg_hi on src2: a*b - c."""
    _sched_barrier(0)
    neg_c = llvm_dialect.fneg(c)
    r = llvm_dialect.intr_fma(a, b, neg_c)
    banked = set_vgpr_bank(r, bank)
    _sched_barrier(0)

    return banked


def _atom_pk_add_f32(a, b, bank):
    """v_pk_add_f32 via arith.addf on v2f32."""
    _sched_barrier(0)
    r = arith.addf(a, b)
    banked = set_vgpr_bank(r, bank)
    _sched_barrier(0)

    return banked


def _atom_pk_mul_f32(a, b, bank):
    """v_pk_mul_f32 via arith.mulf on v2f32."""
    _sched_barrier(0)
    r = arith.mulf(a, b)
    banked = set_vgpr_bank(r, bank)
    _sched_barrier(0)

    return banked


def _atom_cvt_pk_bf16_f32(a, bank):
    """v_cvt_pk_bf16_f32 via arith.truncf v2f32 → v2bf16."""
    _sched_barrier(0)
    v2bf16_ty = ir.VectorType.get([2], ir.BF16Type.get())
    r = arith.truncf(v2bf16_ty, a)
    banked = set_vgpr_bank(r, bank)
    _sched_barrier(0)

    return banked


# --- Synchronization ---


def _atom_s_wait_dscnt(cnt):

    _sched_barrier(0)
    rocdl.s_wait_dscnt(cnt)
    _sched_barrier(0)


def _atom_s_wait_tensorcnt(cnt):
    _sched_barrier(0)
    rocdl.s_wait_tensorcnt(cnt)
    _sched_barrier(0)


def _atom_wgp_barrier():

    _sched_barrier(0)
    rocdl.s_barrier_signal(-1)
    rocdl.s_barrier_wait(0xFFFF)
    _sched_barrier(0)


# ============================================================================
# Schedule Builders (compile-time, no IR emitted)
# ============================================================================


def _build_qk_wmma_schedule(blk, su):
    """Build ordered list of GEMM_INST_COUNT WMMA parameter tuples for QK GEMM.

    SU_K_N=32: 2 N-tiles distributed across 4 MSBs via D-dim.
    For QK_HDIM=192: loop order:
      msb_idx(2) x k(6) x sp_msb(2) x n(1) x m(1) = 24.
    KV MSB remapping:
      k_msb = (k//K_FRAGS_PER_MSB)*2 + sp_msb%2,
      k_frag = k%K_FRAGS_PER_MSB.
    K_FRAGS_PER_MSB = (QK_HDIM // WMMA_K) // 2 = 3 for 192.
    """
    sp_pingpong = blk % 2
    k_pingpong = su % 2
    sp_MSBOFF = sp_pingpong * VPS_MSB_SP
    k_MSBOFF = k_pingpong * VPS_MSB_KV
    sp_off = sp_MSBOFF + 8 * su  # compact SP region: 8 VGPRs/lane per SU

    # K_FRAGS_PER_MSB: how many K-tiles fit in one KV MSB = total_k_tiles / 2_msb_pairs
    _K_FRAGS_PER_MSB = (SP_MSB_K // WMMA_K) // 2  # =2 for 128-dim, =3 for 192-dim

    schedule = []
    for msb_idx in range_constexpr(2):
        for k in range_constexpr(SP_MSB_K // WMMA_K):  # 0..5 for 192-dim
            for sp_msb in [msb_idx, 2 + msb_idx]:
                for n in range_constexpr(SP_MSB_N // WMMA_N):  # 0..0 (1 N-tile)
                    for m in range_constexpr(SP_MSB_M // WMMA_M):  # 0..0
                        q_msb = (sp_msb // 2) * 2 + k // Q_WMMA_PER_MSB
                        n_tile = sp_msb % 2
                        k_msb = (k // _K_FRAGS_PER_MSB) * 2 + n_tile
                        k_frag = k % _K_FRAGS_PER_MSB
                        is_init = k == 0
                        schedule.append(
                            {
                                "sp_msb": sp_msb,
                                "k_msb": k_msb,
                                "q_msb": q_msb,
                                "k_iter": k,
                                "k_frag": k_frag,
                                "n_iter": n,
                                "m_iter": m,
                                "is_init": is_init,
                                "sp_off": sp_off,
                                "k_MSBOFF": k_MSBOFF,
                            }
                        )
    assert len(schedule) == GEMM_INST_COUNT
    return schedule


def _build_pv_wmma_schedule(blk, su):
    """Build 16 PV WMMA parameter tuples per SU stage (no padding, no K-loop).

    Layout: d_msb(4) × n(4) = 16 WMMAs per stage. Across 4 SU stages = 64 WMMAs
    per warp total — matching tile_m=32 × tile_n=128 / (16M*32K_real) for the
    M-side and × N_pv_per_warp(=128)/16 for the N-side accumulators.

    d_msb encodes (m_tile, v_bank):
      d_msb=0 → (m_tile=0, v_bank=0)  → o_tiles[0]
      d_msb=1 → (m_tile=0, v_bank=1)  → o_tiles[1]
      d_msb=2 → (m_tile=1, v_bank=0)  → o_tiles[2]
      d_msb=3 → (m_tile=1, v_bank=1)  → o_tiles[3]

    src_b layout: p_tiles[su][m_tile] is a v16bf16 holding *all real* P data
    for (this SU's K_pv chunk, m_tile half of M), built by concatenating the
    two sibling sp_msbs (2*m_tile and 2*m_tile+1) along K_pv (= N_qk).
    """
    sp_pingpong = blk % 2
    sp_off = sp_pingpong * VPS_MSB_SP + 8 * su

    schedule = []
    for d_msb in range_constexpr(NUM_MSB):
        m_tile = d_msb // 2
        v_msb = d_msb % N_V_MSB
        for n in range_constexpr(N_PV_WMMA_N):
            schedule.append(
                {
                    "d_msb": d_msb,
                    "n": n,
                    "sp_msb": m_tile,
                    "v_msb": v_msb,
                    "sp_off": sp_off,
                }
            )
    assert len(schedule) == PV_GEMM_INST_COUNT
    return schedule


def _build_lds_k_schedule(blk, su):
    """Build 32 LDS load descriptors for K data (ds_load_b128)."""
    schedule = []
    for msb in range_constexpr(NUM_MSB):
        su_off = (blk * CNT_SU + su) * LDS_K_SU_P_SIZE
        for v_idx in range_constexpr(N_LDS_PER_MSB):
            schedule.append(
                {
                    "msb": msb,
                    "offset": v_idx * 32 + su_off,
                    "v_idx": v_idx,
                    "load_type": "b128",
                }
            )
    assert len(schedule) == LDS_INST_COUNT
    return schedule


def _build_lds_v_schedule(blk, su):
    """Build 16 LDS load descriptors for V data (ds_load_tr16_b128).

    The reference kernel uses 2 MSBs x 8 loads each (VPS_MSB_KV=32).
    FlyDSL uses 4 MSBs x 4
    loads each (VPS_MSB_KV=16).  MSBs 0,2 map to V-bank 0 (D cols 0-63) and
    MSBs 1,3 map to V-bank 1 (D cols 64-127).  Within each bank, MSBs 0/1
    cover the first 32 columns and MSBs 2/3 cover the second 32 columns via
    an additional (msb//2)*N_LDS_PER_MSB//2*32 byte offset.
    """
    schedule = []
    # MSB-specific column-group offsets are now folded into kv_lds_addrs[4+msb*2+half_p]
    # (see _build_kv_lds_addrs), so only the per-SU part remains in the offset field.
    su_base_off = (blk * CNT_SU + su) * LDS_V_SU_P_SIZE
    for msb in range_constexpr(NUM_MSB):
        for v_idx in range_constexpr(N_LDS_V_PER_MSB):
            half_p = v_idx & 1
            schedule.append(
                {
                    "msb": msb,
                    "offset": (v_idx // 2) * 32 + su_base_off,
                    "v_idx": v_idx,
                    "half_p": half_p,
                    "load_type": "tr16_b128",
                }
            )
    assert len(schedule) == LDS_V_INST_COUNT
    return schedule


# ============================================================================
# Slot Scheduling Tables
# ============================================================================


def _get_lds_slots(stage, gemm_idx, cycle23, qpoint=GEMM_INST_COUNT // 4 - 1):
    # qpoint: the WMMA index where s_wait_dscnt occupies the slot.
    # GEMM1 (QK): GEMM_INST_COUNT//4-1 = 5. GEMM2 (PV): PV_GEMM_INST_COUNT//4-1 = 3.
    if const_expr(DSWAIT_OCCUPY_WHOLE_WMMA):
        if const_expr(stage >= 1):
            if const_expr(gemm_idx == qpoint and not WAIT_DSCNT0):
                return 0
    if const_expr(cycle23 == 1):
        if const_expr(gemm_idx == qpoint and not WAIT_DSCNT0):
            return 0
        return 1
    return 2


def _get_valu_slots(
    stage, gemm_idx, cycle23, default=2, qpoint=GEMM_INST_COUNT // 4 - 1
):
    # qpoint: same as _get_lds_slots. Pass PV_GEMM_INST_COUNT//4-1 for GEMM2.
    if const_expr(DSWAIT_OCCUPY_WHOLE_WMMA):
        if const_expr(stage >= 1):
            if const_expr(gemm_idx == qpoint and not WAIT_DSCNT0):
                return 0
    if const_expr(stage == 0 and cycle23 == 1):
        if const_expr(gemm_idx in (4, 8, 12)):
            return 0
    return default


def _get_salu_slots(stage, gemm_idx, cycle23):
    if const_expr(cycle23 == 1):
        return 1
    return 2


# ============================================================================
# Softmax PART2 Builder
# ============================================================================


def _build_softmax_part2_ops(
    ty,
    msb,
    blk,
    sp_pairs,
    ss,
    sgpr,
    skip_rescale_sum=False,
    sp_lo_cache=None,
    sp_hi_cache=None,
):
    """Build PART2 softmax ops for one MSB as a flat list of closures.

    Each closure emits one atomic instruction.
    Must be called in order (data dependencies between ops within MSB).

    PART2 (rescale + exp + sum + cvt) runs during GEMM1 stages 0-3,
    which correspond to softmax_stages 4-7.

    Args:
        ty: MLIR type dict
        msb: MSB index (0-3), also used as bank hint
        blk: block index for SP pingpong
        sp_pairs: mutable list[N_SP_PAIRS] of v2f32 (SP accumulator pairs)
        ss: mutable softmax state dict with per-MSB SSA values
        sgpr: dict with SGPR references (s_log2e_scl, s_log2e_scl_pair)
    """
    ops = []
    bank = msb

    # --- Setup phase (8 ops) ---
    # exp_delta is at op[2] — included in GEMM2 first half (< PART2_SPLIT=32).
    # This is required: partial_ed_out carries exp_delta as iter_arg for the next
    # tile's O-rescale. 4 exp_delta total (1/MSB) are unavoidable pipeline overhead.

    # 0. old_max[msb] = local_max[msb]
    # Pre-materialize s_log2e_scl into a per-bank VGPR v2f32
    # early (GEMM1 start) so pk_fma doesn't need a lazy
    # SGPR->VGPR copy (which would cause s_delay_alu or
    # v_nop).
    #
    # IMPORTANT: use plain MLIR insertelement so that the
    # set_vgpr_bank_offset hint can freely direct the register allocator to the target
    # physical register.
    #
    # CSE-safety: each bank (b) passes a different argument to set_vgpr_bank_offset /
    # set_vgpr_bank, so CSE never merges the four copies.  LLVM allocates each at its
    # bank-specific physical register (bank×256 + LOG2E_PAIR_OFFSET for banks 1-3).
    def op_save_old_max(b=bank):
        ss["old_max"][b] = _atom_mov_b32(ss["local_max"][b], b)
        _sched_barrier(0)
        v2f32_ty = ir.VectorType.get([2], ir.F32Type.get())
        idx_0 = arith.unwrap(arith.constant(0, type=T.i32))
        idx_1 = arith.unwrap(arith.constant(1, type=T.i32))
        undef = llvm_dialect.mlir_undef(v2f32_ty)
        _scl = sgpr["s_log2e_scl"]
        v = llvm_dialect.insertelement(undef, _scl, idx_0)
        v = llvm_dialect.insertelement(v, _scl, idx_1)
        if const_expr(b > 0):
            ss["vgpr_log2e_scl_pair"][b] = set_vgpr_bank_offset(v, b, LOG2E_PAIR_OFFSET)
        else:
            ss["vgpr_log2e_scl_pair"][b] = set_vgpr_bank(v, b)
        _sched_barrier(0)

    ops.append(op_save_old_max)

    # 1. curMaxLog2eScl = local_max * s_log2e_scl
    def op_cur_max(b=bank):
        ss["cur_max_log2e"][b] = _atom_mul_f32(
            ss["local_max"][b], sgpr["s_log2e_scl"], b
        )

    ops.append(op_cur_max)

    # 2. exp_delta = exp(delta)  — transcendental, 4-cycle latency
    def op_exp_delta(b=bank):
        ss["exp_delta"][b] = _atom_exp_f32(ss["delta"][b], b)

    ops.append(op_exp_delta)

    # 3. curMaxLog2eScl_1 = local_max * s_log2e_scl (duplicate)
    def op_cur_max_1(b=bank):
        ss["cur_max_log2e_1"][b] = _atom_mul_f32(
            ss["local_max"][b], sgpr["s_log2e_scl"], b
        )

    ops.append(op_cur_max_1)

    # 4. curMaxLog2eScl = old_max * log2e_scl (scalar v_mul_f32)
    def op_mul_old_max(b=bank):
        ss["cur_max_log2e_scalar"][b] = _atom_mul_f32(
            ss["old_max"][b], sgpr["s_log2e_scl"], b
        )

    ops.append(op_mul_old_max)

    # 5. curMaxLog2eScl_dup = broadcast scalar → v2f32 (v_mov_b32)
    # Separated from op4 so scheduler can place distance between
    # the mul producer, the broadcast, and the pk_fma consumer.
    def op_broadcast_dup(b=bank):
        ss["cur_max_log2e_dup"][b] = _broadcast_f32_to_v2f32(
            ss["cur_max_log2e_scalar"][b], b
        )

    ops.append(op_broadcast_dup)

    # 6. exp_delta_dup = exp_delta (duplicate via mov)
    def op_exp_delta_dup(b=bank):
        ss["exp_delta_dup"][b] = _atom_mov_b32(ss["exp_delta"][b], b)

    ops.append(op_exp_delta_dup)

    # 7. row_sums *= exp_delta
    if not skip_rescale_sum:

        def op_rescale_sum(b=bank):
            ss["row_sums"][b] = _atom_mul_f32(ss["exp_delta"][b], ss["row_sums"][b], b)

        ops.append(op_rescale_sum)

    # --- Rescale phase: ALL pk_fma first, then ALL exp ---
    # Separating pkfma and exp into two consecutive groups eliminates the
    # EXP(bank0)↔PK_FMA(bank_msb) alternating MSB switches caused by escaped
    # sp_pairs (pairs 0,1 land in bank0 instead of bank_msb).
    # Latency hiding: 16 pkfmas separate each pair's pkfma from its exp,
    # providing >4 cycles of delay to hide the transcendental exp latency.

    # Phase A: all 16 pk_fma ops.
    # cur_max_log2e_dup (v2f32, lo==hi) produced by op5 broadcast.
    for i in range_constexpr(N_SP_PAIRS):
        _sp_offset = SP_PAIR_BASE + i * 2
        _escaped = i < 2  # first 2 pairs can escape to bank0 when bank>0

        def op_pkfma(idx=i, b=bank, sp_off=_sp_offset, escaped=_escaped):
            src = sp_pairs[idx]
            if const_expr(b > 0 and escaped):
                src = set_vgpr_bank(src, b)  # move escaped pair to bank b
            result = _atom_pk_fma_f32_neg_c(
                src, ss["vgpr_log2e_scl_pair"][b], ss["cur_max_log2e_dup"][b], b
            )
            if const_expr(b > 0):
                sp_pairs[idx] = set_vgpr_bank_offset(result, b, sp_off)
            else:
                sp_pairs[idx] = result

        ops.append(op_pkfma)

    # Phase B: all 32 exp ops (pairs 0..15 in order, lo then hi per pair)
    # ops[24..31] = pair_exp[0..7] → GEMM2 first half (< PART2_SPLIT=32)
    # ops[32..55] = pair_exp[8..15] → GEMM1 second half (>= PART2_SPLIT)
    sum_tmps = [None] * (N_SP_PAIRS // 2)

    for _eidx in range_constexpr(VPS_MSB_SP):  # 32 iters
        _pidx = _eidx // 2
        _is_hi = _eidx % 2
        _ep_offset = SP_PAIR_BASE + _pidx * 2
        if const_expr(_is_hi == 0):

            def op_exp_lo(pidx=_pidx, b=bank, _clo=sp_lo_cache):
                lo, hi = _split_v2f32(sp_pairs[pidx])
                v2f32_ty = ir.VectorType.get([2], ir.F32Type.get())
                _sched_barrier(0)
                exp_lo = _rocdl_exp2(ir.F32Type.get(), lo)
                _sched_barrier(0)
                undef = llvm_dialect.mlir_undef(v2f32_ty)
                _idx0 = arith.unwrap(arith.constant(0, type=T.i32))
                _idx1 = arith.unwrap(arith.constant(1, type=T.i32))
                v = llvm_dialect.insertelement(undef, exp_lo, _idx0)
                v = llvm_dialect.insertelement(v, hi, _idx1)
                sp_pairs[pidx] = v
                if const_expr(_clo is not None):
                    _clo[pidx] = exp_lo  # standalone f32 scalar for yield

            ops.append(op_exp_lo)
        else:

            def op_exp_hi(pidx=_pidx, b=bank, _chi=sp_hi_cache):
                lo, hi = _split_v2f32(sp_pairs[pidx])
                v2f32_ty = ir.VectorType.get([2], ir.F32Type.get())
                _sched_barrier(0)
                exp_hi = _rocdl_exp2(ir.F32Type.get(), hi)
                _sched_barrier(0)
                undef = llvm_dialect.mlir_undef(v2f32_ty)
                _idx0 = arith.unwrap(arith.constant(0, type=T.i32))
                _idx1 = arith.unwrap(arith.constant(1, type=T.i32))
                v = llvm_dialect.insertelement(undef, lo, _idx0)
                v = llvm_dialect.insertelement(v, exp_hi, _idx1)
                sp_pairs[pidx] = v
                if const_expr(_chi is not None):
                    _chi[pidx] = exp_hi  # standalone f32 scalar for yield

            ops.append(op_exp_hi)

    # All cvt_pk first (16 ops), then all pk_add (sum tree).
    # cvt reads sp_pairs (written by exp phase), independent of pk_add tree.
    # The 16 cvt ops provide latency coverage across the pk_add dependency chain
    # when these tokens are dispatched across WMMA boundaries.
    sum_l0 = [None] * (N_SP_PAIRS // 4)
    sum_l1 = [None] * (N_SP_PAIRS // 8)
    sum_l2 = [None]
    final_sum = [None]

    # --- Phase C1: all 16 cvt_pk_bf16_f32 ops ---
    for i in range_constexpr(N_SP_PAIRS):

        def op_cvt(cidx=i, b=bank):
            src = set_vgpr_bank(sp_pairs[cidx], b)
            ss["p_bf16"][b].append(_atom_cvt_pk_bf16_f32(src, b))

        ops.append(op_cvt)

    # --- Phase C2: pk_add level-0 — 8 ops ---
    for i in range_constexpr(N_SP_PAIRS // 2):

        def op_pkadd(idx=i, b=bank):
            sum_tmps[idx] = _atom_pk_add_f32(
                sp_pairs[idx * 2], sp_pairs[idx * 2 + 1], b
            )

        ops.append(op_pkadd)

    # --- Phase C3: sum tree (level-1, level-2, split, accum) ---
    for j in range_constexpr(N_SP_PAIRS // 4):

        def op_sum_l0(j_val=j, b=bank):
            sum_l0[j_val] = _atom_pk_add_f32(
                sum_tmps[j_val * 2], sum_tmps[j_val * 2 + 1], b
            )

        ops.append(op_sum_l0)

    for j in range_constexpr(2):

        def op_sum_l1(j_val=j, b=bank):
            sum_l1[j_val] = _atom_pk_add_f32(
                sum_l0[j_val * 2], sum_l0[j_val * 2 + 1], b
            )

        ops.append(op_sum_l1)

    def op_sum_l2(b=bank):
        sum_l2[0] = _atom_pk_add_f32(sum_l1[0], sum_l1[1], b)

    ops.append(op_sum_l2)

    def op_sum_split(b=bank):
        lo, hi = _split_v2f32(sum_l2[0])
        final_sum[0] = _atom_add_f32(lo, hi, b)

    ops.append(op_sum_split)

    def op_sum_accum(b=bank):
        ss["row_sums"][b] = _atom_add_f32(ss["row_sums"][b], final_sum[0], b)

    ops.append(op_sum_accum)

    return ops


def _build_all_softmax_part2_ops(ty, blk, sp_pairs_all, softmax_state, sgpr_state):
    """Build ALL PART2 softmax ops for all 4 MSBs (built once, consumed
    incrementally across stages with per-stage budgets).

    Returns: list of 4 lists (one per MSB), each containing closures.
    Each closure emits exactly one instruction.
    """
    ops_by_msb = [[] for _ in range_constexpr(NUM_MSB)]
    for m in range_constexpr(NUM_MSB):
        sp_pairs = sp_pairs_all[m]
        msb_ops = _build_softmax_part2_ops(
            ty, m, blk, sp_pairs, softmax_state, sgpr_state
        )
        ops_by_msb[m] = msb_ops
    return ops_by_msb


# ============================================================================
# Softmax PART0 + PART1 Builder (runs during GEMM2 stages)
# ============================================================================

# per MSB: tree-max(16) + merge2(1) + perm-prep(1)
#   + perm-exec(1) + mul(1) + cur-max(1) + merge1(1)
PART0_INSTS = 22
PART1_INSTS = 8  # cross-MSB merge + delta (total, not per MSB)
RLTS_LEN = 9  # run-ids: 0-3 PART0, 4 PART1, 5-8 PART2

N_VALID_GROUPS = CNT_SU  # 4 groups of valid data per MSB
VALID_GROUP_SIZE = 8  # f32 per valid group (one QK accum per SU)
VALID_GROUP_STRIDE = 8  # stride between groups (no padding)


def _build_softmax_part0_ops(ty, msb, sp_pairs, ss, sgpr):
    """Build 22 PART0 closures for one MSB (tree-max of 32 valid values).

    With tile_n=128, each SU provides 4 contiguous valid pairs (= 8 f32) per
    MSB.  All 32 f32 (= 16 pairs) of the compact layout are valid; no padding.
    All values stay in bank=msb → minimal s_set_vgpr_msb transitions.

    Layout per MSB: 4 groups at stride 8, each with 8 valid f32:
      group 0: sp[0..7],   group 1: sp[8..15],
      group 2: sp[16..23], group 3: sp[24..31]

    op 0-3:  Phase 1 — 4 initial max3 (3 elements each)
    op 4-11: Phase 2 — 8 cross-column max3 (2 rounds × 4 groups)
    op 12-15:Phase 3 — 4 last-element max3 (with sp[0] as 3rd arg to force VOP3)
    op 16:   merge1 (max3)
    op 17:   merge2 (max3)
    op 18:   perm_prep (add_f32+0 bank change)
    op 19:   perm_exec (permlanex16)
    op 20:   mul (preMaxLog2eScl) — INDEPENDENT, used as filler in dispatch
    op 21:   cur_max (max3)
    Total: 4 + 8 + 4 + 1 + 1 + 1 + 1 + 1 + 1 = 22
    """
    ops = []
    bank = msb

    sp_f32 = [None] * VPS_MSB_SP
    tmps = [None] * N_VALID_GROUPS

    def _get_sp(offset):
        if const_expr(sp_f32[offset] is None):
            sp_f32[offset] = llvm_dialect.extractelement(
                sp_pairs[offset // 2],
                arith.unwrap(arith.constant(offset % 2, type=T.i32)),
            )
        return sp_f32[offset]

    # Phase 1: initial max3 per valid group (4 ops)
    for k in range_constexpr(N_VALID_GROUPS):

        def op_init_max3(k_=k, b=bank):
            base = k_ * VALID_GROUP_STRIDE
            tmps[k_] = _atom_max3_num_f32(
                _get_sp(base), _get_sp(base + 1), _get_sp(base + 2), b
            )

        ops.append(op_init_max3)

    # Phase 2: cross-column merge (2 rounds × 4 groups = 8 ops)
    for j in range_constexpr(2):
        for k in range_constexpr(N_VALID_GROUPS):

            def op_cross_col(k_=k, j_=j, b=bank):
                base = k_ * VALID_GROUP_STRIDE
                src0_off = base + 3 + j_ * 2
                src1_off = src0_off + 1
                tmps[k_] = _atom_max3_num_f32(
                    _get_sp(src0_off), _get_sp(src1_off), tmps[k_], b
                )

            ops.append(op_cross_col)

    # Phase 3: bring in last element per group (4 ops)
    # Use max3(sp[7], tmp, sp[0]) instead of max3(sp[7], tmp, old_max).
    # sp[0] is guaranteed in bank=msb (same as the other operands here) so no
    # s_set_vgpr_msb switch is needed.  old_max was used purely to prevent LLVM
    # from collapsing max3→max2 AND to force VOP3 encoding; sp[0] achieves
    # both without the cross-bank penalty.
    # Correctness: sp[0] is already included in tmps[k_] via Phase 1
    # (max3(sp[0],sp[1],sp[2])→tmps[k_]), so max3(sp[7], tmps[k_], sp[0])
    # = max(sp[7], tmps[k_]).  old_max is still included via Phase 7.
    for k in range_constexpr(N_VALID_GROUPS):

        def op_last_elem(k_=k, b=bank):
            base = k_ * VALID_GROUP_STRIDE
            tmps[k_] = _atom_max3_num_f32(_get_sp(base + 7), tmps[k_], _get_sp(base), b)

        ops.append(op_last_elem)

    # Phase 4: merge 4 groups (2 ops)
    def op_merge1(b=bank):
        tmps[0] = _atom_max3_num_f32(tmps[0], tmps[1], tmps[2], b)

    ops.append(op_merge1)

    # Phase 4b merge2 (op17): reads tmps[0] from merge1
    def op_merge2(b=bank):
        tmps[0] = _atom_max3_num_f32(tmps[0], tmps[3], tmps[1], b)

    ops.append(op_merge2)

    # Phase 5 perm_prep (op18): bank-change add_f32+0, reads tmps[0] from merge2
    tmps_perm = [None]
    _zero_f32 = arith.unwrap(arith.constant(0.0, type=T.f32))

    def op_perm_prep(b=bank, z=_zero_f32):
        dst_bank = (b + 2) % NUM_MSB
        tmps_perm[0] = _atom_add_f32(tmps[0], z, dst_bank)

    ops.append(op_perm_prep)

    # Phase 6 perm_exec (op19): permlanex16, reads tmps_perm[0] from perm_prep
    def op_perm(b=bank):
        sel_lo = arith.unwrap(arith.constant(0x76543210, type=T.i32))
        sel_hi = arith.unwrap(arith.constant(0xFEDCBA98, type=T.i32))
        dst_bank = (b + 2) % NUM_MSB
        tmps[1] = _atom_permlanex16(tmps_perm[0], sel_lo, sel_hi, dst_bank)

    ops.append(op_perm)

    # Phase 7 mul (op20): preMaxLog2eScl = old_max * log2e_scl — FULLY INDEPENDENT.
    # This op has no data dependency on any op in the merge1→merge2→perm chain.
    # The dispatch uses it as a 1-per-gap filler to provide 4 intervening VALU
    # between each consecutive dependent pair.
    def op_pre_max(b=bank):
        ss["pre_max_log2e_scl"][b] = _atom_mul_f32(
            ss["old_max"][b], sgpr["s_log2e_scl"], b
        )

    ops.append(op_pre_max)

    # Phase 8 cur_max (op21): max3(tmps[0], tmps[1], old_max), reads perm_exec result
    def op_cur_max(b=bank):
        ss["local_max"][b] = _atom_max3_num_f32(tmps[0], tmps[1], ss["old_max"][b], b)

    ops.append(op_cur_max)

    assert len(ops) == PART0_INSTS
    return ops


def _build_softmax_part1_ops(ty, ss, sgpr):
    """Build 8 PART1 closures for cross-MSB merge + delta.

    Layout: 2 max3 (pair merge, old_max 3rd arg forces VOP3) + 2 mov + 4 fma

    Returns (ops_list, per_msb_assignment):
      ops_list: 8 closures
      per_msb_assignment: list of MSB bank hints per op (for dispatch)
    """
    ops = []
    msb_assign = []

    # max_num(local_max[0], local_max[1]) → local_max[0]
    # Use max3 with pre_max_log2e_scl[0] (bank0, already computed) as 3rd arg
    # to force VOP3 without cross-bank penalty from old_max (which may be in
    # wrong bank due to regalloc).  pre_max_log2e_scl[0] ≤ any local_max so
    # it doesn't change the result; it also prevents LLVM from folding to max2.
    def op_max01():
        ss["local_max"][0] = _atom_max3_num_f32(
            ss["local_max"][0], ss["local_max"][1], ss["pre_max_log2e_scl"][0], 0
        )

    ops.append(op_max01)
    msb_assign.append(0)

    # max_num(local_max[2], local_max[3]) → local_max[2]
    def op_max23():
        ss["local_max"][2] = _atom_max3_num_f32(
            ss["local_max"][2], ss["local_max"][3], ss["pre_max_log2e_scl"][2], 2
        )

    ops.append(op_max23)
    msb_assign.append(2)

    # mov local_max[1] = local_max[0]
    def op_mov1():
        ss["local_max"][1] = _atom_mov_b32(ss["local_max"][0], 1)

    ops.append(op_mov1)
    msb_assign.append(1)

    # mov local_max[3] = local_max[2]
    def op_mov3():
        ss["local_max"][3] = _atom_mov_b32(ss["local_max"][2], 3)

    ops.append(op_mov3)
    msb_assign.append(3)

    # fma delta[msb] = preMaxLog2eScl[msb] - local_max[msb] * log2e_scl
    for msb in [0, 2, 1, 3]:

        def op_fma_delta(b=msb):
            ss["delta"][b] = _atom_fma_f32_neg_src0(
                ss["local_max"][b], sgpr["s_log2e_scl"], ss["pre_max_log2e_scl"][b], b
            )

        ops.append(op_fma_delta)
        msb_assign.append(msb)

    assert len(ops) == PART1_INSTS
    return ops, msb_assign


def _build_all_softmax_gemm2_ops(
    ty, blk, sp_pairs_all, softmax_state, sgpr_state, skip_rescale_sum=False
):
    """Build all softmax ops for GEMM2 stages: PART0 + PART1 + PART2.

    Organized by run-id (9 groups):
      run_id 0-3: PART0 per MSB (22 ops each)
      run_id 4:   PART1 cross-MSB (8 ops)
      run_id 5-8: PART2 per MSB (beginning of PART2 for current SP)

    Returns: (ops_by_rid, rid_budget_by_stage)
      ops_by_rid: list of 9 lists of closures
      rid_budget_by_stage: [4 stages][9 run-ids] int budget
    """
    if const_expr("pre_max_log2e_scl" not in softmax_state):
        softmax_state["pre_max_log2e_scl"] = [None] * NUM_MSB

    # PART0: 22 ops per MSB
    ops_by_rid = [[] for _ in range_constexpr(RLTS_LEN)]
    for m in range_constexpr(NUM_MSB):
        sp_pairs = sp_pairs_all[m]
        p0_ops = _build_softmax_part0_ops(ty, m, sp_pairs, softmax_state, sgpr_state)
        ops_by_rid[m] = p0_ops

    # PART1: 8 ops (cross-MSB)
    p1_ops, p1_msb_assign = _build_softmax_part1_ops(ty, softmax_state, sgpr_state)
    ops_by_rid[4] = p1_ops

    # PART2: first half only (ops 0..PART2_SPLIT-1) for each MSB.
    # sp_lo_cache / sp_hi_cache: per-MSB lists of f32 scalars to capture exp results
    # from EXP token dispatch. Avoids unreliable v2f32 packing under WMMA pressure.
    sp_lo_cache = [[None] * N_SP_PAIRS for _ in range_constexpr(NUM_MSB)]
    sp_hi_cache = [[None] * N_SP_PAIRS for _ in range_constexpr(NUM_MSB)]
    for m in range_constexpr(NUM_MSB):
        sp_pairs = sp_pairs_all[m]
        p2_ops = _build_softmax_part2_ops(
            ty,
            m,
            blk,
            sp_pairs,
            softmax_state,
            sgpr_state,
            skip_rescale_sum=skip_rescale_sum,
            sp_lo_cache=sp_lo_cache[m],
            sp_hi_cache=sp_hi_cache[m],
        )
        ops_by_rid[5 + m] = p2_ops[:PART2_G2_SPLIT]

    # Budget layout (mirrors v0, adapted for 16-WMMA stages):
    # Budget layout: spread PART0 across stages 0-1 to stay within slot capacity.
    # Each stage has 16 WMMAs × 4 slots = 64 (−4 for dswait) ≈ 60 usable slots.
    # PART0: 10/MSB in stage 0 (40 total), 11/MSB in stage 1 (44 total) → fits in 60.
    # PART1: budget opened in stage 1 (dispatches once all PART0 rids exhaust, filling
    #   the last 2-4 WMMAs of stage 1 that were previously
    #   naked); stage 2 catches overflow.
    # PART2: stages 2-3. In stage 2, lds_done=False also
    #   dispatches PART2 per-MSB (rids 5..8) in parallel
    #   with V LDS loads (see stage>=2 check in
    #   _dispatch_softmax_gemm2), spreading PART2 across
    #   8 lds phases instead of crowding lds_done=True.
    rid_budget = [[0] * RLTS_LEN for _ in range_constexpr(4)]

    # Stage 0: PART0 first chunk (10/MSB = 40 total).
    # No PART1 here: budget=10 exhausts each rid before all 21 PART0 ops complete,
    # so ss['local_max'] is stale when budget reaches 0. PART1 requires fully-completed
    # PART0 (all 21 ops/MSB), which only holds after stage 1 finishes the remaining 11.
    part0_left = PART0_INSTS
    p0c = min(part0_left, ALU_PER_STAGE[0] // NUM_MSB)
    for m in range_constexpr(NUM_MSB):
        rid_budget[0][m] = p0c
    part0_left -= p0c

    # Stage 1: PART0 remainder (11/MSB = 44 total) + PART1 overflow + small PART2 seed.
    # PART1 overflow catches any PART1 ops not dispatched in stage 0.
    # PART2 seed (4/MSB) fills WMMAs 14-15 after PART1 exhausts, via drain-one-rid.
    p0c = min(part0_left, ALU_PER_STAGE[1] // NUM_MSB)
    for m in range_constexpr(NUM_MSB):
        rid_budget[1][m] = p0c
    part0_left -= p0c
    rid_budget[1][4] = PART1_INSTS  # fills naked tail WMMAs of stage 1
    for m in range_constexpr(NUM_MSB):
        rid_budget[1][5 + m] = 4  # seed: ~1 WMMA of PART2 per MSB at stage 1 tail

    # Stage 2: remaining PART0 (if any) + PART1 fallback + PART2 start.
    # PART1 budget kept for safety (catches any ops not dispatched in stage 1),
    # but p2_budget_2 is computed from the full ALU_PER_STAGE[2] without subtracting
    # PART1_INSTS: since PART1 is normally done in stage 1, those 8 slots are free
    # for PART2 in practice. This gives p2_budget_2=14 instead of 12 per MSB, which
    # provides 2 extra WMMAs of PART2 coverage in lds_done=True of stage 2, eliminating
    # the 3-WMMA naked cluster at the end of stage 2.
    if const_expr(part0_left > 0):
        p0c = min(part0_left, ALU_PER_STAGE[2] // NUM_MSB)
        for m in range_constexpr(NUM_MSB):
            rid_budget[2][m] = p0c
        part0_left -= p0c
        remaining_budget = ALU_PER_STAGE[2] - p0c * NUM_MSB
    else:
        remaining_budget = ALU_PER_STAGE[2]
    # fallback in case stage 1 left some
    rid_budget[2][4] = min(PART1_INSTS, remaining_budget)
    # Do NOT subtract PART1 from remaining_budget: PART1 is typically done in stage 1,
    # so those budget slots are available to PART2 dispatch at runtime.
    p2_budget_2 = remaining_budget // NUM_MSB
    for m in range_constexpr(NUM_MSB):
        rid_budget[2][5 + m] = p2_budget_2

    # Stage 3: PART2 overflow (budget generous; ops_by_rid truncated to PART2_SPLIT
    # so rid exhaustion prevents double-processing regardless of budget).
    p2_budget_3 = ALU_PER_STAGE[3] // NUM_MSB
    for m in range_constexpr(NUM_MSB):
        rid_budget[3][5 + m] = p2_budget_3

    return ops_by_rid, rid_budget, sp_lo_cache, sp_hi_cache


# ============================================================================
# Emit Helpers for Interleaving Engine
# ============================================================================


def _emit_qk_wmma(ty, wmma_op, q_tiles, kv_tiles, sp_tiles):
    """Emit one QK WMMA. Returns updated SP tiles.

    Pre-fence on src_a and post-fence on result tie the pure WMMA
    intrinsic to the LLVM scheduling barrier chain, preventing the
    scheduler from clustering WMMAs across beats.
    """
    sp_msb = wmma_op["sp_msb"]
    k_msb = wmma_op["k_msb"]
    q_msb = wmma_op["q_msb"]
    k_iter = wmma_op["k_iter"]
    k_frag = wmma_op["k_frag"]
    n_iter = wmma_op["n_iter"]

    src_a = kv_tiles[k_msb][k_frag]
    src_b = q_tiles[q_msb][k_iter % Q_WMMA_PER_MSB]

    if const_expr(wmma_op["is_init"]):
        result = _atom_wmma_init(ty, src_a, src_b, sp_msb)
    else:
        acc = sp_tiles[sp_msb][n_iter]
        result = _atom_wmma_accum(ty, src_a, src_b, acc, sp_msb)

    sp_tiles[sp_msb][n_iter] = result
    return sp_tiles


def _emit_pv_wmma(ty, wmma_op, v_tiles, p_tiles, o_tiles):
    """Emit one PV WMMA. Returns updated O tiles.

    src_a = V tile (v16bf16), src_b = P tile (v16bf16), acc = O tile (v8f32).
    PV always accumulates (no init) — O persists across the entire core loop.
    """
    d_msb = wmma_op["d_msb"]
    n = wmma_op["n"]
    sp_msb = wmma_op["sp_msb"]
    v_msb = wmma_op["v_msb"]

    src_a = v_tiles[v_msb][n]
    src_b = p_tiles[sp_msb]
    acc = o_tiles[d_msb][n]

    result = _atom_wmma_accum(ty, src_a, src_b, acc, d_msb)
    o_tiles[d_msb][n] = result
    return o_tiles


def _emit_lds_load(ty, lds_op, kv_lds_addrs, kv_tiles_out):
    """Emit one LDS load (K: ds_load_b128 or V: ds_load_tr16_b128)."""
    msb = lds_op["msb"]
    offset = lds_op["offset"]
    v_idx = lds_op["v_idx"]
    load_type = lds_op["load_type"]

    if const_expr(load_type == "b128"):
        addr = kv_lds_addrs[msb]
        tile = _atom_ds_load_b128(ty, addr, offset, msb)
    else:
        half_p = lds_op["half_p"]
        # kv_lds_addrs[4 + msb*2 + half_p] — both dh0 and dh1 for
        # this MSB are in bank=msb, so dst/addr/addr are all same bank →
        # s_set_vgpr_msb stays constant within each MSB's load group.
        addr = kv_lds_addrs[NUM_MSB + msb * 2 + (1 if half_p else 0)]
        tile = _atom_ds_load_tr16_b128(ty, addr, offset, msb)

    kv_tiles_out[msb][v_idx] = tile
    return kv_tiles_out


def _emit_tdm_salu(tdm_type, tdm_state, n_slots):
    """Emit up to n_slots SALU instructions for TDM address update.

    Sequence: s_lshl2_add_u32 → s_addc_u32 → s_sub_i32 → s_max_i32 →
              s_add_i32 → s_mov_b32
    """
    key = "v" if tdm_type == KV_V else "k"
    salu_queue = tdm_state.get(f"{key}_salu_queue", [])
    emitted = 0
    while emitted < n_slots and salu_queue:
        op = salu_queue.pop(0)
        op()
        emitted += 1
    tdm_state[f"{key}_salu_queue"] = salu_queue


def _first_nonempty(counts):
    """Return index of first non-zero element, or len(counts) if all zero."""
    for i, c in enumerate(counts):
        if const_expr(c > 0):
            return i
    return len(counts)


# ============================================================================
# Core Loop Stage Composer — cl_su_V3 Interleaving Engine (GEMM1 only)
# ============================================================================


def _cl_su_v3_stage(
    ty,
    stage,
    gemm_blk,
    gemm_su,
    tdm_type,
    tdm_blk,
    tdm_su,
    lds_type,
    lds_blk,
    lds_su,
    q_tiles,
    kv_tiles,
    sp_tiles,
    kv_lds_addrs,
    kv_tiles_next,
    softmax_ops_by_msb,
    softmax_idx_by_msb,
    softmax_budget,
    tdm_state,
    tdm_barrier=False,
    # optional list of O-tile rescale closures
    # (16 per stage, 1 pk_mul each)
    o_rescale_ops=None,
):
    """One fully interleaved GEMM1 (QK) pipeline stage (SU_K_N=32).

    Interleaves 16 WMMAs, 16 LDS loads, and softmax VALU ops
    using per-MSB softmax budgets from ALU_CNT_STAGE tables.
    """
    has_tdm = tdm_type != KV_NONE

    wmma_schedule = _build_qk_wmma_schedule(gemm_blk, gemm_su)

    if const_expr(lds_type == KV_K):
        lds_schedule = _build_lds_k_schedule(lds_blk, lds_su)
    else:
        lds_schedule = _build_lds_v_schedule(lds_blk, lds_su)

    lds_idx = 0
    ds_issued = 0
    _o_resc_idx = 0  # O-rescale closure index (for O_RESC token dispatch)

    for gemm_idx in range_constexpr(GEMM_INST_COUNT):
        # === Emit WMMA ===
        wmma_op = wmma_schedule[gemm_idx]
        sp_tiles = _emit_qk_wmma(ty, wmma_op, q_tiles, kv_tiles, sp_tiles)
        _sched_barrier(0)

        # DS wait at 1/4 point (mandatory, non-scheduled)
        _sched_barrier(0)
        if const_expr(gemm_idx == GEMM_INST_COUNT // 4 - 1 and not WAIT_DSCNT0):
            _atom_s_wait_dscnt(ds_issued)
        _sched_barrier(0)

        # TDM loads at WMMA 0 (non-scheduled: tensor_load_to_lds setup is SALU+load)
        if const_expr(gemm_idx == 0 and has_tdm):
            tdm_key = "v" if tdm_type == KV_V else "k"
            descs = tdm_state.get(f"{tdm_key}_descs", None)
            _sched_barrier(0)
            if const_expr(descs is not None):
                _tdm_di = tdm_state[f"{tdm_key}_desc_idx"]
                for _ in range_constexpr(TDM_LOADS_PER_STAGE):
                    if const_expr(_tdm_di < len(descs)):
                        _atom_tdm_load(ty, descs[_tdm_di][0], descs[_tdm_di][1])
                        _tdm_di += 1
                tdm_state[f"{tdm_key}_desc_idx"] = _tdm_di
            else:
                if const_expr(tdm_type == KV_V):
                    _atom_tdm_load(ty, tdm_state["v_g0"], tdm_state["v_g1"])
                else:
                    _atom_tdm_load(ty, tdm_state["k_g0"], tdm_state["k_g1"])
            _sched_barrier(0)

        # TDM barrier signal (mandatory, non-scheduled)
        _g1_barrier_idx = GEMM_INST_COUNT - BARRIER_SIGNAL_AHEAD - 1
        if const_expr(tdm_barrier and gemm_idx == _g1_barrier_idx):
            _atom_s_wait_tensorcnt(4)
            rocdl.s_barrier_signal(-1)

        # === Schedule-driven dispatch: ALL tokens between this WMMA and next ===
        # Tokens 5-8: PART2 softmax  Tokens 9-12: K_tile ds_load_b128 (by MSB)
        # Tokens 13-16: V_tile ds_load_tr16_b128 (by MSB)
        # Token 17: O_RESC pk_mul   Token 18: TDM (handled above, no-op here)
        _g1_row = GEMM1_SCHEDULE[g1_row_idx(stage, gemm_idx)]
        _g1_half = len(_g1_row) // 2

        # Phase 1: first half of row
        for _i in range_constexpr(len(_g1_row)):
            if const_expr(_i < _g1_half):
                _tok = _g1_row[_i]
                if const_expr(5 <= _tok <= 8):  # PART2 softmax op
                    _msb = _tok - _P2_BASE
                    if softmax_idx_by_msb[_msb] < len(softmax_ops_by_msb[_msb]):
                        softmax_ops_by_msb[_msb][softmax_idx_by_msb[_msb]]()
                        softmax_idx_by_msb[_msb] += 1
                    _sched_barrier(0)
                elif const_expr(9 <= _tok <= 12):  # K tile ds_load_b128
                    if lds_idx < LDS_INST_COUNT:
                        kv_tiles_next = _emit_lds_load(
                            ty, lds_schedule[lds_idx], kv_lds_addrs, kv_tiles_next
                        )
                        lds_idx += 1
                        ds_issued += 1
                elif const_expr(13 <= _tok <= 16):  # V tile ds_load_tr16_b128
                    if lds_idx < LDS_V_INST_COUNT:
                        kv_tiles_next = _emit_lds_load(
                            ty, lds_schedule[lds_idx], kv_lds_addrs, kv_tiles_next
                        )
                        lds_idx += 1
                        ds_issued += 1
                elif const_expr(_tok == 17):  # O_RESC: pk_mul closure
                    if const_expr(o_rescale_ops is not None):
                        _sched_barrier(0)
                        o_rescale_ops[_o_resc_idx]()
                        _o_resc_idx += 1
                        _sched_barrier(0)
                # Token 18 (TDM): handled above, nothing here

        # DS wait at end of stage (mandatory, non-scheduled)
        _sched_barrier(0)
        if const_expr(gemm_idx == GEMM_INST_COUNT - 1):
            if const_expr(WAIT_DSCNT0):
                _atom_s_wait_dscnt(0)
            else:
                _atom_s_wait_dscnt(LDS_INST_COUNT // 2)
        _sched_barrier(0)

        # Phase 2: second half of row
        for _i in range_constexpr(len(_g1_row)):
            if const_expr(_g1_half <= _i < len(_g1_row)):
                _tok = _g1_row[_i]
                if const_expr(5 <= _tok <= 8):
                    _msb = _tok - _P2_BASE
                    if softmax_idx_by_msb[_msb] < len(softmax_ops_by_msb[_msb]):
                        softmax_ops_by_msb[_msb][softmax_idx_by_msb[_msb]]()
                        softmax_idx_by_msb[_msb] += 1
                    _sched_barrier(0)
                elif const_expr(9 <= _tok <= 12):
                    if lds_idx < LDS_INST_COUNT:
                        kv_tiles_next = _emit_lds_load(
                            ty, lds_schedule[lds_idx], kv_lds_addrs, kv_tiles_next
                        )
                        lds_idx += 1
                        ds_issued += 1
                elif const_expr(13 <= _tok <= 16):
                    if lds_idx < LDS_V_INST_COUNT:
                        kv_tiles_next = _emit_lds_load(
                            ty, lds_schedule[lds_idx], kv_lds_addrs, kv_tiles_next
                        )
                        lds_idx += 1
                        ds_issued += 1
                elif const_expr(_tok == 17):
                    if const_expr(o_rescale_ops is not None):
                        _sched_barrier(0)
                        o_rescale_ops[_o_resc_idx]()
                        _o_resc_idx += 1
                        _sched_barrier(0)

        # TDM barrier_wait at last WMMA (mandatory, non-scheduled)
        if const_expr(tdm_barrier and gemm_idx == GEMM_INST_COUNT - 1):
            rocdl.s_barrier_wait(-1)

        _sched_barrier(0)

    return sp_tiles, kv_tiles_next


# ============================================================================
# Core Loop Stage Composer — cl_su_V3 Interleaving Engine (GEMM2 / PV)
# ============================================================================


def _cl_su_v3_stage_gemm2(
    ty,
    stage,
    gemm_blk,
    gemm_su,
    lds_type,
    lds_blk,
    lds_su,
    v_tiles,
    p_tiles,
    o_tiles,
    kv_lds_addrs,
    kv_tiles_next,
    ops_by_rid,
    rid_idx,
    tdm_state=None,
    tdm_type=KV_NONE,
    tdm_barrier=False,
    o_rescale_exp_delta=None,
):
    """GEMM2 (PV) stage — same structure as GEMM1.

    Dispatch is 100% schedule-driven (GEMM2_SCHEDULE).
    V/K tokens (13-16, 9-12) call _emit_lds_load directly in the dispatch loop,
    NOT via _get_lds_slots. This matches GEMM1's approach and prevents LLVM hoisting.

    Token dispatch (per schedule row):
      0-3   P0_Mx : draw from ops_by_rid[0..3]
      4     P1    : draw from ops_by_rid[4]
      5-8   P2_Mx : draw from ops_by_rid[5..8], capped at PART2_EXP_START
      9-12  K_Mx  : _emit_lds_load from K tile schedule
      13-16 V_Mx  : _emit_lds_load from V tile schedule
      17    O_RESC: (not used in GEMM2)
      18    TDM   : handled non-scheduled (emitted before the loop)
      19-22 EXP_Mx: draw pair_exp from ops_by_rid[5..8] via exp_rid_idx
    """
    has_tdm = tdm_type != KV_NONE
    wmma_schedule = _build_pv_wmma_schedule(gemm_blk, gemm_su)

    if const_expr(lds_type == KV_K):
        lds_schedule = _build_lds_k_schedule(lds_blk, lds_su)
    else:
        lds_schedule = _build_lds_v_schedule(lds_blk, lds_su)

    lds_idx = 0
    ds_issued = 0

    # exp_rid_idx: separate tracking for EXP_Mx tokens (pair_exp ops).
    # Starts at PART2_EXP_START so EXP dispatch draws from pair_exp portion only.
    exp_rid_idx = [PART2_EXP_START] * NUM_MSB

    # ---- O tile rescaling (stage=0 only, interleaved with WMMAs) ----
    _o_rescale_ed_v8 = {}

    def _build_o_rescale_ed_v8(d_msb):
        if d_msb not in _o_rescale_ed_v8:
            _ed = o_rescale_exp_delta[d_msb]
            if _ed is None:
                _o_rescale_ed_v8[d_msb] = None
                return
            _v = llvm_dialect.mlir_undef(ty["v8f32"])
            for _i in range_constexpr(8):
                _v = llvm_dialect.insertelement(
                    _v, _ed, arith.unwrap(arith.constant(_i, type=T.i32))
                )
            _o_rescale_ed_v8[d_msb] = _v

    def _emit_o_rescale_tile(d_msb, n):
        if const_expr(o_rescale_exp_delta is None):
            return
        _build_o_rescale_ed_v8(d_msb)
        _ed_v8 = _o_rescale_ed_v8[d_msb]
        if _ed_v8 is None:
            return
        o_tiles[d_msb][n] = arith.mulf(o_tiles[d_msb][n], _ed_v8)

    if const_expr(stage == 0):
        _sched_barrier(0)
        for _n0 in range_constexpr(N_PV_WMMA_N):
            _emit_o_rescale_tile(0, _n0)
        _sched_barrier(0)

    for gemm_idx in range_constexpr(PV_GEMM_INST_COUNT):
        if const_expr(stage == 0 and 3 * N_PV_WMMA_N <= gemm_idx < PV_GEMM_INST_COUNT):
            _sched_barrier(0)
            _emit_o_rescale_tile(3, gemm_idx - 3 * N_PV_WMMA_N)
            _sched_barrier(0)

        wmma_op = wmma_schedule[gemm_idx]
        o_tiles = _emit_pv_wmma(ty, wmma_op, v_tiles, p_tiles, o_tiles)
        _sched_barrier(0)

        if const_expr(stage == 0):
            if const_expr(0 <= gemm_idx < N_PV_WMMA_N):
                _sched_barrier(0)
                _emit_o_rescale_tile(1, gemm_idx)
                _sched_barrier(0)
            elif const_expr(N_PV_WMMA_N <= gemm_idx < 2 * N_PV_WMMA_N):
                _sched_barrier(0)
                _emit_o_rescale_tile(2, gemm_idx - N_PV_WMMA_N)
                _sched_barrier(0)

        _sched_barrier(0)
        if const_expr(gemm_idx == PV_GEMM_INST_COUNT // 4 - 1 and not WAIT_DSCNT0):
            _atom_s_wait_dscnt(ds_issued)
        _sched_barrier(0)

        # --- Phase 1 (cycle23=1) ---
        # TDM at WMMA 0 (non-scheduled: tensor_load_to_lds setup is SALU)
        if const_expr(gemm_idx == 0 and has_tdm):
            tdm_key = "v" if tdm_type == KV_V else "k"
            descs = tdm_state.get(f"{tdm_key}_descs", None)
            if const_expr(descs is not None):
                _sched_barrier(0)
                _tdm_di = tdm_state[f"{tdm_key}_desc_idx"]
                for _ in range_constexpr(TDM_LOADS_PER_STAGE):
                    if const_expr(_tdm_di < len(descs)):
                        _atom_tdm_load(ty, descs[_tdm_di][0], descs[_tdm_di][1])
                        _tdm_di += 1
                tdm_state[f"{tdm_key}_desc_idx"] = _tdm_di
                _sched_barrier(0)

        _pv_barrier_idx = PV_GEMM_INST_COUNT - BARRIER_SIGNAL_AHEAD - 1
        if const_expr(tdm_barrier and gemm_idx == _pv_barrier_idx):
            _atom_s_wait_tensorcnt(4)
            rocdl.s_barrier_signal(-1)

        # Schedule-driven dispatch — same structure as GEMM1
        _g2_row = GEMM2_SCHEDULE[g2_row_idx(stage, gemm_idx)]
        _g2_half = len(_g2_row) // 2

        for _i in range_constexpr(len(_g2_row)):
            if const_expr(_i < _g2_half):
                _tok = _g2_row[_i]
                if const_expr(0 <= _tok < RLTS_LEN):  # P0/P1/P2_pkfma
                    _rid = _tok
                    if const_expr(5 <= _rid <= 8):
                        if rid_idx[_rid] < PART2_EXP_START and rid_idx[_rid] < len(
                            ops_by_rid[_rid]
                        ):
                            ops_by_rid[_rid][rid_idx[_rid]]()
                            rid_idx[_rid] += 1
                    else:
                        if rid_idx[_rid] < len(ops_by_rid[_rid]):
                            ops_by_rid[_rid][rid_idx[_rid]]()
                            rid_idx[_rid] += 1
                    _sched_barrier(0)
                elif const_expr(9 <= _tok <= 12):  # K tile ds_load_b128
                    if lds_idx < LDS_INST_COUNT:
                        kv_tiles_next = _emit_lds_load(
                            ty, lds_schedule[lds_idx], kv_lds_addrs, kv_tiles_next
                        )
                        lds_idx += 1
                        ds_issued += 1
                elif const_expr(13 <= _tok <= 16):  # V tile ds_load_tr16_b128
                    if lds_idx < LDS_V_INST_COUNT:
                        kv_tiles_next = _emit_lds_load(
                            ty, lds_schedule[lds_idx], kv_lds_addrs, kv_tiles_next
                        )
                        lds_idx += 1
                        ds_issued += 1
                elif const_expr(19 <= _tok <= 22):  # EXP_Mx pair_exp (3cy)
                    _msb = _tok - _EXP_BASE
                    _erid = _msb + _P2_BASE
                    if exp_rid_idx[_msb] < len(ops_by_rid[_erid]):
                        ops_by_rid[_erid][exp_rid_idx[_msb]]()
                        exp_rid_idx[_msb] += 1
                    _sched_barrier(0)

        _sched_barrier(0)
        if const_expr(gemm_idx == PV_GEMM_INST_COUNT - 1):
            if const_expr(WAIT_DSCNT0):
                _atom_s_wait_dscnt(0)
            else:
                _atom_s_wait_dscnt(LDS_INST_COUNT // 2)
        _sched_barrier(0)

        # --- Phase 2 (cycle23=0) ---
        for _i in range_constexpr(len(_g2_row)):
            if const_expr(_g2_half <= _i < len(_g2_row)):
                _tok = _g2_row[_i]
                if const_expr(0 <= _tok < RLTS_LEN):
                    _rid = _tok
                    if const_expr(5 <= _rid <= 8):
                        if rid_idx[_rid] < PART2_EXP_START and rid_idx[_rid] < len(
                            ops_by_rid[_rid]
                        ):
                            ops_by_rid[_rid][rid_idx[_rid]]()
                            rid_idx[_rid] += 1
                    else:
                        if rid_idx[_rid] < len(ops_by_rid[_rid]):
                            ops_by_rid[_rid][rid_idx[_rid]]()
                            rid_idx[_rid] += 1
                    _sched_barrier(0)
                elif const_expr(9 <= _tok <= 12):
                    if lds_idx < LDS_INST_COUNT:
                        kv_tiles_next = _emit_lds_load(
                            ty, lds_schedule[lds_idx], kv_lds_addrs, kv_tiles_next
                        )
                        lds_idx += 1
                        ds_issued += 1
                elif const_expr(13 <= _tok <= 16):
                    if lds_idx < LDS_V_INST_COUNT:
                        kv_tiles_next = _emit_lds_load(
                            ty, lds_schedule[lds_idx], kv_lds_addrs, kv_tiles_next
                        )
                        lds_idx += 1
                        ds_issued += 1
                elif const_expr(19 <= _tok <= 22):
                    _msb = _tok - _EXP_BASE
                    _erid = _msb + _P2_BASE
                    if exp_rid_idx[_msb] < len(ops_by_rid[_erid]):
                        ops_by_rid[_erid][exp_rid_idx[_msb]]()
                        exp_rid_idx[_msb] += 1
                    _sched_barrier(0)

        if const_expr(tdm_barrier and gemm_idx == PV_GEMM_INST_COUNT - 1):
            rocdl.s_barrier_wait(-1)

        _sched_barrier(0)

    return o_tiles, kv_tiles_next


# ============================================================================
# Core Loop — GEMM1 Stages 0-3 + GEMM2 Stages 4-7
# ============================================================================


def _core_loop_gemm1(
    ty,
    memload,
    q_tiles,  # [4 msb][Q_WMMA_PER_MSB] v16bf16 — Q data
    kv_tiles,  # [4 msb][2] v16bf16 — paired K tiles for WMMA
    sp_tiles,  # [4 msb][1] v8f32 — QK accumulators
    kv_lds_addrs,  # [4 K_addr + 4 V_addr] i32 — LDS address VGPRs
    tdm_state,  # TDM SGPR descriptors
    softmax_state,  # Softmax state (old_max, local_max, row_sums, delta, etc.)
    sgpr_state,  # SGPR references (s_log2e_scl, etc.)
):
    """GEMM1 (QK) only — kept for backward compatibility with test harness."""
    sp_out, kv_out, _, _ = _core_loop(
        ty,
        memload,
        q_tiles,
        kv_tiles,
        sp_tiles,
        None,  # o_tiles (no GEMM2)
        kv_lds_addrs,
        tdm_state,
        softmax_state,
        sgpr_state,
        gemm2=False,
    )
    return sp_out, kv_out


def _pair_v_tiles_for_wmma(v_tiles_raw, ty):
    """Pair raw V loads [NUM_MSB][N_LDS_PER_MSB] into [N_V_MSB][N_PV_WMMA_N] v16bf16.

    V loads are ds_load_tr16_b128 producing v8bf16. MSBs map to V banks
    via v_bank = msb % N_V_MSB: MSBs {0,2}→bank 0, MSBs {1,3}→bank 1.
    Each bank collects 2×N_LDS_PER_MSB raw loads, paired into N_PV_WMMA_N
    v16bf16 fragments for WMMA consumption.
    """
    v_paired = []
    for bank in range_constexpr(N_V_MSB):
        bank_raw = []
        for msb in range_constexpr(NUM_MSB):
            if const_expr(msb % N_V_MSB == bank):
                bank_raw.extend(v_tiles_raw[msb])
        frags = []
        for n in range_constexpr(N_PV_WMMA_N):
            lo = bank_raw[n * 2]
            hi = bank_raw[n * 2 + 1]
            frag = make_wmma_frag_bf16(lo, hi)
            frags.append(frag)
        v_paired.append(frags)
    return v_paired


# ============================================================================
# Pure QK / PV / Softmax Helpers (for prologue — no interleaving)
# ============================================================================


def _load_k_su_from_lds(ty, kv_lds_addrs, blk, su):
    """Load K data for one SU from LDS, wait, return paired WMMA tiles."""
    kv_raw = [[None] * N_LDS_PER_MSB for _ in range_constexpr(NUM_MSB)]
    lds_schedule = _build_lds_k_schedule(blk, su)
    for lds_op in lds_schedule:
        kv_raw = _emit_lds_load(ty, lds_op, kv_lds_addrs, kv_raw)
    _atom_s_wait_dscnt(0)
    _sched_barrier(0)
    return _pair_k_tiles_for_wmma(kv_raw, ty)


def _load_v_su_from_lds(ty, kv_lds_addrs, blk, su):
    """Load V data for one SU from LDS, wait, return raw tiles."""
    kv_raw = [[None] * N_LDS_V_PER_MSB for _ in range_constexpr(NUM_MSB)]
    lds_schedule = _build_lds_v_schedule(blk, su)
    for lds_op in lds_schedule:
        kv_raw = _emit_lds_load(ty, lds_op, kv_lds_addrs, kv_raw)
    _atom_s_wait_dscnt(0)
    _sched_barrier(0)
    return kv_raw


def _load_v_two_sus_from_lds(ty, kv_lds_addrs, blk, su0, su1):
    """Load V data for two SUs simultaneously, sorted by MSB.

    Issues all 2×NUM_MSB×N_LDS_PER_MSB ds_load_tr16_b128 instructions
    grouped by MSB (all MSB-0 loads first, then MSB-1, …, then MSB-3),
    followed by a single s_wait_dscnt.

    Within each MSB group every load uses the same address
    bank (= bank_msb) and the same dst bank (= bank_msb), so
    s_set_vgpr_msb only changes at the three MSB-group boundaries →
    4 MSB contexts total instead of 8 with two back-to-back single-SU calls.
    """
    sched0 = _build_lds_v_schedule(blk, su0)
    sched1 = _build_lds_v_schedule(blk, su1)

    raw0 = [[None] * N_LDS_V_PER_MSB for _ in range_constexpr(NUM_MSB)]
    raw1 = [[None] * N_LDS_V_PER_MSB for _ in range_constexpr(NUM_MSB)]

    # Interleave by MSB: for each MSB, emit su0's loads then su1's loads.
    # All loads in one MSB group share the same addr bank and dst bank,
    # so no s_set_vgpr_msb switch is needed within the group.
    for msb in range_constexpr(NUM_MSB):
        for op in sched0:
            if const_expr(op["msb"] == msb):
                raw0 = _emit_lds_load(ty, op, kv_lds_addrs, raw0)
        for op in sched1:
            if const_expr(op["msb"] == msb):
                raw1 = _emit_lds_load(ty, op, kv_lds_addrs, raw1)

    _atom_s_wait_dscnt(0)
    _sched_barrier(0)
    return raw0, raw1


def _qk_pure_su(ty, blk, su, q_tiles, kv_tiles, sp_tiles):
    """Pure QK GEMM for one SU: GEMM_INST_COUNT WMMAs, no interleaving."""
    schedule = _build_qk_wmma_schedule(blk, su)
    for wmma_op in schedule:
        sp_tiles = _emit_qk_wmma(ty, wmma_op, q_tiles, kv_tiles, sp_tiles)
        _sched_barrier(0)
    return sp_tiles


def _pv_pure_su(ty, blk, su, v_tiles, p_tiles_su, o_tiles):
    """Pure PV GEMM for one SU: 16 WMMAs (4 d_msb × 4 n), no interleaving.

    Returns updated o_tiles.
    v_tiles: [N_V_MSB][N_PV_WMMA_N] v16bf16 (paired V data)
    p_tiles_su: [2 m_tile] v16bf16 (compact softmax output for this SU, each
        v16bf16 is a full no-padding PV WMMA src_b for one (m_tile, K_pv chunk)).
    """
    schedule = _build_pv_wmma_schedule(blk, su)
    for wmma_op in schedule:
        o_tiles = _emit_pv_wmma(ty, wmma_op, v_tiles, p_tiles_su, o_tiles)
        _sched_barrier(0)
    return o_tiles


def _fill_sp_pairs_for_su(sp_pairs_all, su_sp_tiles_one, su_idx):
    """Fill sp_pairs_all[msb][su_idx*4 : su_idx*4+4] from one SU's sp_tiles.

    Identical to one SU's iteration inside _sp_tiles_to_sp_pairs.
    Call this after each GEMM1 stage so the corresponding op_init_max3(k_=su_idx)
    can be dispatched immediately after — LLVM schedules it in the NEXT stage's
    tail WMMAs rather than as a monolithic burst between GEMM1 and GEMM2.

    Args:
        sp_pairs_all: [NUM_MSB][N_SP_PAIRS] mutable list pre-filled with None
        su_sp_tiles_one: [NUM_MSB][1] list — one SU's v8f32 QK accumulators
        su_idx: which SU (0..CNT_SU-1)
    """
    _pairs_per_su = N_SP_PAIRS // CNT_SU  # 4
    for msb in range_constexpr(NUM_MSB):
        v8 = su_sp_tiles_one[msb][0]
        for i in range_constexpr(_pairs_per_su):
            lo = llvm_dialect.extractelement(
                v8, arith.unwrap(arith.constant(i * 2, type=T.i32))
            )
            hi = llvm_dialect.extractelement(
                v8, arith.unwrap(arith.constant(i * 2 + 1, type=T.i32))
            )
            pair_idx = su_idx * _pairs_per_su + i
            v2 = _make_v2f32(lo, hi, bank=msb)
            if const_expr(msb > 0):
                sp_pairs_all[msb][pair_idx] = set_vgpr_bank_offset(
                    v2, msb, SP_PAIR_BASE + pair_idx * 2
                )
            else:
                sp_pairs_all[msb][pair_idx] = v2


def _sp_tiles_to_sp_pairs(su_sp_tiles_list):
    """Convert sp_tiles from all SUs to sp_pairs for softmax (compact layout).

    Layout per MSB (16 v2f32 pairs = 32 f32/lane, all real, no padding):
      SU 0: pairs 0..3
      SU 1: pairs 4..7
      SU 2: pairs 8..11
      SU 3: pairs 12..15

    Each SU contributes exactly one v8f32 (= 8 f32/lane = 4 v2f32) per MSB
    from its QK accumulator, packed contiguously.

    Args:
        su_sp_tiles_list: [CNT_SU][NUM_MSB][1] — sp_tiles after each SU's QK.
    Returns:
        sp_pairs: [NUM_MSB][N_SP_PAIRS=16] v2f32
    """
    sp_pairs = []
    for msb in range_constexpr(NUM_MSB):
        pairs = [None] * N_SP_PAIRS
        for su in range_constexpr(CNT_SU):
            v8 = su_sp_tiles_list[su][msb][0]
            for i in range_constexpr(4):
                lo = llvm_dialect.extractelement(
                    v8, arith.unwrap(arith.constant(i * 2, type=T.i32))
                )
                hi = llvm_dialect.extractelement(
                    v8, arith.unwrap(arith.constant(i * 2 + 1, type=T.i32))
                )
                pair_idx = su * 4 + i
                v2 = _make_v2f32(lo, hi, bank=msb)
                if const_expr(msb > 0):
                    pairs[pair_idx] = set_vgpr_bank_offset(
                        v2, msb, SP_PAIR_BASE + pair_idx * 2
                    )
                else:
                    pairs[pair_idx] = v2
        sp_pairs.append(pairs)
    return sp_pairs


def _softmax_pure(ty, blk, sp_pairs_all, softmax_state, sgpr_state):
    """Run full softmax sequentially (all PART0 + PART1 + PART2).

    No interleaving with GEMM. Used in prologue and post-process.
    Modifies sp_pairs_all and softmax_state in place.

    Execution order by run_id 0..8:
      0-3: PART0 per MSB (max-tree reduction)
      4:   PART1 (cross-MSB merge + delta)
      5-8: PART2 per MSB (rescale + exp + sum + cvt_bf16)
    """
    if const_expr("pre_max_log2e_scl" not in softmax_state):
        softmax_state["pre_max_log2e_scl"] = [None] * NUM_MSB

    ops_by_rid, _, _, _ = _build_all_softmax_gemm2_ops(
        ty, blk, sp_pairs_all, softmax_state, sgpr_state
    )

    for rid in range_constexpr(RLTS_LEN):
        for op in ops_by_rid[rid]:
            op()


def _softmax_part01_only(ty, blk, sp_pairs_all, softmax_state, sgpr_state):
    """Run PART0 + PART1 only (no PART2), for init stages 0..3 core work.

    After this call:
      - softmax_state['local_max'][msb] = max of sp_pairs for each MSB
      - softmax_state['delta'][msb] = old_max*log2e - local_max*log2e
      - softmax_state['pre_max_log2e_scl'][msb] = old_max * log2e_scl
    PART2 (pkfma, exp, cvt, rescale) runs later in core_loop's GEMM1 stages.
    """
    if const_expr("pre_max_log2e_scl" not in softmax_state):
        softmax_state["pre_max_log2e_scl"] = [None] * NUM_MSB

    ops_by_rid, _, _, _ = _build_all_softmax_gemm2_ops(
        ty, blk, sp_pairs_all, softmax_state, sgpr_state
    )

    # Op layout:
    #   op 0..15: Phase 1-3 (tree max, block-major)
    #   op 16:    merge1   (max3)
    #   op 17:    merge2   (max3, reads merge1 result)
    #   op 18:    perm_prep (add_f32+0 bank change, reads merge2 result)
    #   op 19:    perm_exec (permlanex16, reads perm_prep result)
    #   op 20:    mul      (independent: old_max * log2e_scl)
    #   op 21:    cur_max  (max3, reads perm_exec result)
    #
    # To eliminate all s_delay_alu in the chain
    # merge1->merge2->perm_prep->perm_exec->cur_max,
    # each consecutive pair needs 4+ intervening VALU
    # (= dep at position 5+, not tracked).
    # Strategy: dispatch 5 column-major groups separated by a SINGLE mul from ONE MSB
    # as filler. This gives exactly 4 intervening between each dep pair:
    #
    #   merge1(M0..M3), [B], mul(M0), [B], merge2(M0..M3), [B], mul(M1), [B],
    #   perm_prep(M0..M3), [B], mul(M2), [B], perm_exec(M0..M3), [B], mul(M3), [B],
    #   cur_max(M0..M3)
    #
    # Verification (worst case = same-MSB chain):
    #   merge1(Mi) at pos P -> merge2(Mi) at pos P+5:
    #     4 intervening -> DEP_5 (not tracked)
    #   merge2(Mi) at pos Q → perm_prep(Mi) at pos Q+5: 4 intervening ✓
    #   perm_prep(Mi) at pos R → perm_exec(Mi) at pos R+5: 4 intervening ✓
    #   perm_exec(Mi) at pos S → cur_max(Mi) at pos S+5: 4 intervening ✓
    #   → zero s_delay_alu for the entire Part0 chain
    #
    _MUL = 20  # op index of mul (the filler)
    _BLK = 4  # ops per MSB per block

    # Ops 0..15 in blocks of _BLK, MSB-major within each block
    for _b in range_constexpr(16 // _BLK):  # blocks 0,1,2,3 (ops 0-15)
        for rid in range_constexpr(NUM_MSB):
            for _j in range_constexpr(_BLK):
                ops_by_rid[rid][_b * _BLK + _j]()

    # merge1 column-major
    for rid in range_constexpr(NUM_MSB):
        ops_by_rid[rid][16]()
    _sched_barrier(0)
    # mul(M0) filler: 4 intervening between merge1/merge2
    ops_by_rid[0][_MUL]()
    _sched_barrier(0)

    # merge2 column-major
    for rid in range_constexpr(NUM_MSB):
        ops_by_rid[rid][17]()
    _sched_barrier(0)
    # mul(M1) filler: 4 between merge2/perm_prep
    ops_by_rid[1][_MUL]()
    _sched_barrier(0)

    # perm_prep column-major
    for rid in range_constexpr(NUM_MSB):
        ops_by_rid[rid][18]()
    _sched_barrier(0)
    # mul(M2) filler: 4 between perm_prep/perm_exec
    ops_by_rid[2][_MUL]()
    _sched_barrier(0)

    # perm_exec column-major
    for rid in range_constexpr(NUM_MSB):
        ops_by_rid[rid][19]()
    _sched_barrier(0)
    # mul(M3) filler: 4 between perm_exec/cur_max
    ops_by_rid[3][_MUL]()
    _sched_barrier(0)

    # cur_max column-major
    for rid in range_constexpr(NUM_MSB):
        ops_by_rid[rid][21]()

    # PART1 (cross-MSB, sequential after all PART0):
    for op in ops_by_rid[4]:
        op()


def _dispatch_part0_chunk(ops_by_rid, op_start: int, op_end: int):
    """Dispatch PART0 closures ops[op_start..op_end) for all 4 MSBs.

    Column-major execution: emit op[i] for all MSBs before op[i+1].
    Used to split PART0 into chunks for interleaving across GEMM2 stages.
    Caller must emit sched_barrier(0) before and after each chunk to prevent
    LLVM from hoisting ops across GEMM2 WMMA boundaries.
    """
    for op_idx in range_constexpr(op_end - op_start):
        for rid in range_constexpr(NUM_MSB):
            actual = op_start + op_idx
            if const_expr(actual < len(ops_by_rid[rid])):
                ops_by_rid[rid][actual]()


def _dispatch_part1(ops_by_rid):
    """Dispatch PART1 closures (cross-MSB merge). Must run after PART0 is done."""
    for op in ops_by_rid[4]:
        op()


def _softmax_part2_only(
    ty, blk, sp_pairs_all, softmax_state, sgpr_state, skip_rescale_sum=False
):
    """Run PART2 only (run_ids 5-8): rescale, exp, sum, cvt → p_bf16.

    Runs post-process softmax stages 4..7 (is_first=0).
    After this call, softmax_state['p_bf16'] has the bf16 softmax output
    ready for PV GEMM.
    """
    if const_expr("pre_max_log2e_scl" not in softmax_state):
        softmax_state["pre_max_log2e_scl"] = [None] * NUM_MSB

    ops_by_rid, _, _, _ = _build_all_softmax_gemm2_ops(
        ty,
        blk,
        sp_pairs_all,
        softmax_state,
        sgpr_state,
        skip_rescale_sum=skip_rescale_sum,
    )

    for rid in range_constexpr(5, RLTS_LEN):  # run_ids 5-8 (PART2)
        for op in ops_by_rid[rid]:
            op()


def _build_p_tiles_from_softmax(ty, softmax_state):
    """Build P tiles (v16bf16) for PV GEMM from softmax p_bf16 output.

    Compact (no-padding) layout:
      softmax_state['p_bf16'][msb] holds N_SP_PAIRS (=16) v2bf16 per MSB,
      all real, with SU `su` writing to indices [su*4 : su*4+4].

    Each PV WMMA src_b needs 16 bf16/lane = 8 v2bf16/lane covering K_pv=0..31
    for one (M_tile, this SU's K range).  We build it by concatenating the two
    sibling sp_msbs (2*m_tile, 2*m_tile+1) along the K_pv axis:
      first octet  (8 bf16) ← sp_msb=2*m_tile     (K_pv low  half of SU range)
      second octet (8 bf16) ← sp_msb=2*m_tile+1   (K_pv high half of SU range)

    Returns: p_tiles[CNT_SU][2] v16bf16  (2 = m_tile in {top, bot})
    """
    p_bf16_all = softmax_state["p_bf16"]
    p_tiles = []
    for su in range_constexpr(CNT_SU):
        su_tiles = []
        p_start = su * 4
        for m_tile in range_constexpr(2):
            msb_lo = 2 * m_tile  # K_pv low half (e.g. sp_msb 0 or 2)
            msb_hi = 2 * m_tile + 1  # K_pv high half (e.g. sp_msb 1 or 3)
            combined = (
                p_bf16_all[msb_lo][p_start : p_start + 4]
                + p_bf16_all[msb_hi][p_start : p_start + 4]
            )
            su_tiles.append(_pack_v2bf16_to_v16bf16(ty, combined, m_tile))
        p_tiles.append(su_tiles)
    return p_tiles


def _core_loop(
    ty,
    memload,
    q_tiles,  # [4 msb][Q_WMMA_PER_MSB] v16bf16 — Q data
    kv_tiles,  # [4 msb][2] v16bf16 — paired K tiles for WMMA
    sp_tiles,  # [4 msb][1] v8f32 — QK accumulators
    o_tiles,  # [4 d_msb][N_PV_WMMA_N] v8f32 — O accumulators (or None)
    kv_lds_addrs,  # [4 K_addr + 4 V_addr] i32 — LDS address VGPRs
    tdm_state,  # TDM SGPR descriptors
    softmax_state,  # Softmax state (old_max, local_max, row_sums, delta, etc.)
    sgpr_state,  # SGPR references (s_log2e_scl, etc.)
    gemm2=True,  # Whether to run GEMM2 stages
):
    """Full core loop: GEMM1 (QK) + softmax + GEMM2 (PV).

    tile_n=128: single pass with 4 SUs (no pi/half loops).
    4 GEMM1 stages (96 QK WMMAs for QK_HDIM=192) + 4 GEMM2 stages (64 PV WMMAs
    via the compact, no-padding schedule = 16 WMMAs/stage × 4 SUs).

    Pipeline per call:
      GEMM1: QK on current K (in LDS) → sp_tiles per SU
      PART2: run on sp_pairs_prev (from previous tile) → P tiles + rescale O
      PART0+PART1: run on current sp_tiles → update max/delta for next PART2
      GEMM2: PV using P tiles × V (in LDS) → O accumulates

    Returns: (sp_tiles, kv_tiles, o_tiles, su_sp_tiles_list).
      su_sp_tiles_list: [CNT_SU][NUM_MSB][1] v8f32 — per-SU QK output for
                        next iteration's sp_pairs_prev.
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
        sp_pairs_all = [[None] * N_SP_PAIRS for _ in range_constexpr(NUM_MSB)]

    softmax_ops_by_msb = _build_all_softmax_part2_ops(
        ty, 0, sp_pairs_all, softmax_state, sgpr_state
    )
    softmax_idx_by_msb = [0] * NUM_MSB

    stage_configs = [
        (0, KV_V if memload else KV_NONE, KV_K, blk, 1),
        (1, KV_V if memload else KV_NONE, KV_K, blk, 2),
        (2, KV_NONE, KV_K, blk, 3),
        (3, KV_NONE, KV_V, blk, 0),
    ]

    su_sp_tiles_list = []

    for stage_idx, (g_su, t_type, l_type, l_blk, l_su) in enumerate(stage_configs):

        _n_lds = N_LDS_V_PER_MSB if l_type == KV_V else N_LDS_PER_MSB
        kv_tiles_next_raw = [[None] * _n_lds for _ in range_constexpr(NUM_MSB)]

        softmax_stage = (stage_idx + 4) % ALU_STAGES
        budget_per_msb = ALU_PER_STAGE[softmax_stage] // NUM_MSB
        softmax_budget = [budget_per_msb] * NUM_MSB

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
        )

        su_sp_tiles_list.append(
            [[sp_tiles[msb][0]] for msb in range_constexpr(NUM_MSB)]
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

    # 1. Run remaining PART2 ops (budget was insufficient during GEMM1)
    for msb in range_constexpr(NUM_MSB):
        for op in softmax_ops_by_msb[msb][softmax_idx_by_msb[msb] :]:
            op()

    # 2. Convert per-SU sp_tiles → sp_pairs for current tile
    sp_pairs_current = _sp_tiles_to_sp_pairs(su_sp_tiles_list)

    # 3. Run PART0+PART1 on current tile's sp_pairs → updates max/delta
    _softmax_part01_only(ty, blk, sp_pairs_current, softmax_state, sgpr_state)

    # 4. Build P tiles from p_bf16 (produced by PART2)
    p_tiles_computed = _build_p_tiles_from_softmax(ty, softmax_state)

    # ================================================================
    # fmha_mask placeholder (no causal mask)
    # ================================================================
    _emit_void("s_nop 0")

    # ================================================================
    # GEMM2 (PV): 4 stages (SU 0..3)
    # No softmax interleaving — PART0+PART1 already ran above.
    # ================================================================

    v_tiles_paired = _pair_v_tiles_for_wmma(v_tiles_out, ty)

    empty_ops = [[] for _ in range_constexpr(RLTS_LEN)]
    empty_idx = [0] * RLTS_LEN

    g2_stage_configs = [
        (0, KV_V, blk, 1),
        (1, KV_V, blk, 2),
        (2, KV_V, blk, 3),
        (3, KV_K, blk, 0),
    ]

    for stage_idx, (g_su, l_type, l_blk, l_su) in enumerate(g2_stage_configs):

        p_tiles_su = p_tiles_computed[g_su]

        _n_lds = N_LDS_V_PER_MSB if l_type == KV_V else N_LDS_PER_MSB
        kv_tiles_next_raw = [[None] * _n_lds for _ in range_constexpr(NUM_MSB)]

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
            kv_lds_addrs,
            kv_tiles_next_raw,
            empty_ops,
            empty_idx,
        )

        if const_expr(l_type == KV_V):
            v_tiles_paired = _pair_v_tiles_for_wmma(kv_tiles_next_raw, ty)
        else:
            kv_tiles = _pair_k_tiles_for_wmma(kv_tiles_next_raw, ty)

    return sp_tiles, kv_tiles, o_tiles, su_sp_tiles_list
