"""BF16 FMHA Forward Prologue — gfx1250 Pure FlyDSL Implementation.

FlyDSL-native implementation of the FMHA prologue with compiler-managed
VGPR bank allocation via @llvm.amdgcn.set.vgpr.bank intrinsic.

Target: gfx1250 (MI450), wave32, 4 waves per thread-group, 128 threads.
All phases use FlyDSL — zero inline ASM. 128×128 compute (TDM loads 256).
"""

from __future__ import annotations

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm as llvm_dialect
from flydsl.expr import arith, buffer_ops, gpu, rocdl, vector
from flydsl.expr.rocdl import tdm_ops
from flydsl.expr.typing import T
from flydsl._mlir.dialects import rocdl as _rocdl
from fmha_core_loop_gfx1250 import QK_HDIM, Q_WMMA_PER_MSB, KV_BPP

# ============================================================================
# Constants
# ============================================================================

WAVE_SIZE = 32
NUM_WAVES = 4
BLOCK_SIZE = WAVE_SIZE * NUM_WAVES

Q_TILE_M = 128
Q_TILE_D = QK_HDIM  # = 192
K_TILE_N = 128       # TDM load tile = compute tile (N=128)
K_COMPUTE_N = 128    # compute tile (QK WMMA uses SU0+SU1)
K_SU_SIZE = 64
NUM_K_SU = 2         # TDM loads 2 SUs (= compute)
NUM_K_SU_COMPUTE = 2

K_ROW_BYTES = 400   # TDM dim0=200 → LDS inner stride = 200*2 = 400B (2-way bank conflicts)
V_ROW_BYTES = 288
K_SU_HALF_OFFSET = 0x1900   # 16 * K_ROW_BYTES = 16 * 400 = 6400
V_SU_HALF_OFFSET = 0x1200
V_LDS_OFFSET = 0x8800       # after 2 K SUs: 2×0x4400
PINGPONG_OFFSET = 0x11800   # one-ping: K(0x8800)+V(0x9000)
K_SU1_OFFSET = 0x4400
V_SU1_OFFSET = 0x4800

WMMA_M, WMMA_N, WMMA_K = 16, 16, 32


K_D_HALF_OFFSET = 0x2200
K_LOAD_STRIDE = 32
NUM_K_LOADS_PER_HALF = 8

# WMMA tiling: 4 groups, each with (k_bank_pair, k_frag_indices)
GROUP_CONFIG = [
    ((0, 1), (0, 1, 2, 3)),
    ((2, 3), (0, 1, 2, 3)),
    ((0, 1), (4, 5, 6, 7)),
    ((2, 3), (4, 5, 6, 7)),
]

# Accumulator column base for causal mask
ACC_COL_BASE = {
    (0, 0): 0, (0, 1): 16, (2, 0): 64, (2, 1): 80,
    (1, 0): 8, (1, 1): 24, (3, 0): 72, (3, 1): 88,
    (0, 2): 32, (0, 3): 48, (2, 2): 96, (2, 3): 112,
    (1, 2): 40, (1, 3): 56, (3, 2): 104, (3, 3): 120,
}

# TDM config constants.
# K: dim0=192 (QK_HDIM), no padding. QK_HDIM=192 is not a multiple of any
# power-of-2 pad_interval that fits in one pad per row, so we skip padding to
# avoid the continuous-stream rotation bug. No bank-conflict padding for K.
_K_TDM_CONFIG = (1 << 16)  # data_size=1 (bf16), pad_enable=0
# V: dim0=128, pad_interval=128 elems=64dwords → enc_interval=5, 32B pad → enc_amount=7
_V_TDM_CONFIG = (1 << 20) | (5 << 22) | (7 << 25)

# ============================================================================
# Core Helpers
# ============================================================================


def _raw(val):
    return arith.unwrap(val)


_NUW_ATTR = None

def _get_nuw():
    global _NUW_ATTR
    if _NUW_ATTR is None:
        _NUW_ATTR = ir.Attribute.parse("#arith.overflow<nuw>")
    return _NUW_ATTR

def _add_nuw(a, b):
    """arith.addi with nuw flag — enables gfx1250 buffer offset folding."""
    return _raw(arith.addi(a, b, overflow_flags=_get_nuw()))

def _mul_nuw(a, b):
    """arith.muli with nuw flag — preserves nuw through constant folding."""
    return _raw(arith.muli(a, b, overflow_flags=_get_nuw()))


def set_vgpr_bank(raw_val, bank: int):
    val_type = raw_val.type
    bank_val = _raw(arith.constant(bank, type=T.i32))
    return llvm_dialect.call_intrinsic(
        val_type, "llvm.amdgcn.set.vgpr.bank",
        [raw_val, bank_val], [], [])


def set_vgpr_bank_offset(raw_val, bank: int, offset: int):
    """Pin raw_val to a specific physical register: HWIdx = bank*256 + offset.

    Emits llvm.amdgcn.set.vgpr.bank.offset which the pre-RA pass converts to a
    BankOffsetHint — a single-candidate hint that is much stronger than the
    256-candidate BankHint produced by set_vgpr_bank.
    """
    val_type = raw_val.type
    bank_val   = _raw(arith.constant(bank,   type=T.i32))
    offset_val = _raw(arith.constant(offset, type=T.i32))
    return llvm_dialect.call_intrinsic(
        val_type, "llvm.amdgcn.set.vgpr.bank.offset",
        [raw_val, bank_val, offset_val], [], [])


def _acc_bank(g_idx, tile):
    return (g_idx & 1) + 2 * (1 if tile >= 2 else 0)


def _permlanex16_f32(src):
    """v_permlanex16_b32 via inline asm (intrinsic lacks SelectionDAG lowering)."""
    i32_ty = ir.IntegerType.get_signless(32)
    from flydsl._mlir.dialects import arith as arith_d
    src_i32 = arith_d.bitcast(i32_ty, src)
    result_i32 = llvm_dialect.inline_asm(
        i32_ty, [src_i32],
        "v_permlanex16_b32 $0, $1, 0, 0",
        "=v,0", has_side_effects=True,
    )
    return arith_d.bitcast(ir.F32Type.get(), result_i32)


def _setreg(hwreg_enc, value):
    """s_setreg_imm32_b32 via llvm.amdgcn.s.setreg intrinsic.
    hwreg_enc = id | (offset << 6) | ((size-1) << 11)"""
    imm = _raw(arith.constant(hwreg_enc, type=T.i32))
    val = _raw(arith.constant(value, type=T.i32))
    llvm_dialect.call_intrinsic(None, "llvm.amdgcn.s.setreg", [imm, val], [], [])


def _asm_void(asm_str, operands=None, constraints="", **kwargs):
    llvm_dialect.inline_asm(
        None, operands or [], asm_str, constraints,
        has_side_effects=True, **kwargs)


def _s_wait_tensorcnt(cnt):
    from flydsl.expr import rocdl as _rocdl_expr
    _rocdl_expr.s_wait_tensorcnt(cnt)


# ============================================================================
# LDS / Fragment Helpers
# ============================================================================


def lds_load_b128(lds_base_raw, byte_offset_raw):
    lds_ptr_ty = ir.Type.parse("!llvm.ptr<3>")
    total = _raw(arith.addi(lds_base_raw, byte_offset_raw))
    ptr = llvm_dialect.inttoptr(lds_ptr_ty, total)
    vec_ty = ir.VectorType.get([4], ir.IntegerType.get_signless(32))
    return llvm_dialect.load(vec_ty, ptr)


def make_wmma_frag_bf16(vec4_lo, vec4_hi):
    vec8bf16_ty = ir.VectorType.get([8], ir.BF16Type.get())
    v0 = vector.bitcast(vec8bf16_ty, vec4_lo)
    v1 = vector.bitcast(vec8bf16_ty, vec4_hi)
    return vector.shuffle(v0, v1, list(range(16)))


# ============================================================================
# FlyDSL Phase Functions
# ============================================================================


def _phase4_q_load_flydsl(lane_id, q_rsrc, stride_q_seq, wave_id, q_tile_offset_bytes=None):
    """Load Q tile (QK_HDIM × TG_Q_ROWS bf16) → q_frags[4][Q_FRAGS_PER_BANK] with bank hints.

    For QK_HDIM=192: 6 loads per bank (3 frags × 2 loads each), bank1 offset=192 bytes.
    For QK_HDIM=128: 4 loads per bank (2 frags × 2 loads each), bank1 offset=128 bytes.
    """
    lane_lo = arith.andi(lane_id, arith.constant(15, type=T.i32))
    lane_hi = arith.shrui(lane_id, arith.constant(4, type=T.i32))
    base = arith.addi(
        arith.muli(lane_lo, stride_q_seq),
        arith.muli(lane_hi, arith.constant(16, type=T.i32)))
    wave_off = arith.muli(
        arith.muli(wave_id, arith.constant(32, type=T.i32)),
        stride_q_seq)
    q_byte_off = arith.addi(base, wave_off)

    q_elem_off = arith.shrui(q_byte_off, arith.constant(2, type=T.i32))

    vec4i32_ty = ir.VectorType.get([4], ir.IntegerType.get_signless(32))
    soff_zero = _raw(arith.constant(0, type=T.i32))
    aux_zero = _raw(arith.constant(0, type=T.i32))

    q_base_bytes = _mul_nuw(_raw(q_elem_off), _raw(arith.constant(4, type=T.i32)))
    if q_tile_offset_bytes is not None:
        q_base_bytes = _add_nuw(q_tile_offset_bytes, q_base_bytes)
    stride_16_bytes = _raw(arith.muli(stride_q_seq, arith.constant(16, type=T.i32)))

    # K-half byte offset = QK_HDIM bytes (splits K cols in half per bank pair)
    _K_HALF_BYTES = QK_HDIM               # 192 for 192-dim, 128 for 128-dim
    # Each bank covers half of QK_HDIM: QK_HDIM/2 elements / WMMA_K(32) = frags_per_bank
    _FRAGS_PER_BANK = (QK_HDIM // 2) // 32  # = 3 for 192-dim (96 K cols / 32 = 3), 2 for 128-dim
    _LOADS_PER_BANK = _FRAGS_PER_BANK * 2   # 2 raw loads per v16bf16 frag

    bank_offsets_bytes = [
        _raw(arith.constant(0, type=T.i32)),
        _raw(arith.constant(_K_HALF_BYTES, type=T.i32)),
        stride_16_bytes,
        _add_nuw(stride_16_bytes, _raw(arith.constant(_K_HALF_BYTES, type=T.i32))),
    ]

    q_frags = []
    for bank in fx.range_constexpr(4):
        if bank == 0:
            bank_voff = q_base_bytes
        else:
            bank_voff = _add_nuw(q_base_bytes, bank_offsets_bytes[bank])
        bank_loads = []
        for i in fx.range_constexpr(_LOADS_PER_BANK):
            if i == 0:
                voff = bank_voff
            else:
                voff = _add_nuw(bank_voff, _raw(arith.constant(i * 32, type=T.i32)))
            loaded = _rocdl.raw_ptr_buffer_load(
                vec4i32_ty, q_rsrc, voff, soff_zero, aux_zero)
            bank_loads.append(set_vgpr_bank(loaded, bank))
        rocdl.sched_barrier(0)
        bank_frags = []
        for f in fx.range_constexpr(_FRAGS_PER_BANK):
            frag = make_wmma_frag_bf16(bank_loads[2 * f], bank_loads[2 * f + 1])
            bank_frags.append(set_vgpr_bank(frag, bank))
        q_frags.append(bank_frags)
        rocdl.sched_barrier(0)

    return q_frags


def _phase9a_k_lds_addr_gen(lane_id, wave_id):
    """K LDS addresses — pure FlyDSL arith + bank hints."""
    lane_lo = arith.andi(lane_id, arith.constant(0xF, type=T.i32))
    lane_hi = arith.shrui(lane_id, arith.constant(4, type=T.i32))
    base = arith.addi(
        arith.muli(lane_lo, arith.constant(K_ROW_BYTES, type=T.i32)),
        arith.muli(lane_hi, arith.constant(16, type=T.i32)))

    seg1 = arith.addi(base, arith.constant(K_SU_HALF_OFFSET, type=T.i32))
    seg2 = arith.addi(base, arith.constant(PINGPONG_OFFSET, type=T.i32))
    seg3 = arith.addi(seg1, arith.constant(PINGPONG_OFFSET, type=T.i32))

    wave_is_odd = arith.cmpi(
        arith.CmpIPredicate.ne,
        arith.andi(wave_id, arith.constant(1, type=T.i32)),
        arith.constant(0, type=T.i32))

    a0 = _raw(arith.select(wave_is_odd, seg2, base))
    a1 = _raw(arith.select(wave_is_odd, seg3, seg1))
    a2 = _raw(arith.select(wave_is_odd, base, seg2))
    a3 = _raw(arith.select(wave_is_odd, seg1, seg3))

    rocdl.sched_barrier(0)

    return [
        set_vgpr_bank(a0, 0),
        set_vgpr_bank(a1, 1),
        set_vgpr_bank(a2, 2),
        set_vgpr_bank(a3, 3),
    ]


def _phase9b_v_lds_addr_gen(lane_id, wave_id):
    """V LDS addresses — pure FlyDSL arith + bank hints."""
    lane_and_7 = arith.andi(lane_id, arith.constant(7, type=T.i32))
    lane_shr4 = arith.shrui(lane_id, arith.constant(4, type=T.i32))
    row = arith.addi(lane_and_7, arith.shli(lane_shr4, arith.constant(3, type=T.i32)))

    lane_shr3 = arith.shrui(lane_id, arith.constant(3, type=T.i32))
    sub_col = arith.shli(
        arith.andi(lane_shr3, arith.constant(1, type=T.i32)),
        arith.constant(4, type=T.i32))

    addr_base = arith.addi(
        arith.addi(
            arith.muli(row, arith.constant(V_ROW_BYTES, type=T.i32)),
            sub_col),
        arith.constant(V_LDS_OFFSET, type=T.i32))
    addr_half = arith.addi(addr_base, arith.constant(V_SU_HALF_OFFSET, type=T.i32))

    seg2_a = arith.addi(addr_base, arith.constant(PINGPONG_OFFSET, type=T.i32))
    seg2_h = arith.addi(addr_half, arith.constant(PINGPONG_OFFSET, type=T.i32))

    wave_is_odd = arith.cmpi(
        arith.CmpIPredicate.ne,
        arith.andi(wave_id, arith.constant(1, type=T.i32)),
        arith.constant(0, type=T.i32))

    s0_a = _raw(arith.select(wave_is_odd, seg2_a, addr_base))
    s0_h = _raw(arith.select(wave_is_odd, seg2_h, addr_half))
    s2_a = _raw(arith.select(wave_is_odd, addr_base, seg2_a))
    s2_h = _raw(arith.select(wave_is_odd, addr_half, seg2_h))

    rocdl.sched_barrier(0)

    return [
        set_vgpr_bank(s0_a, 0),
        set_vgpr_bank(s0_h, 1),
        set_vgpr_bank(s2_a, 2),
        set_vgpr_bank(s2_h, 3),
    ]


def _phase9d_k_lds_load_flydsl(k_addrs, lds_offset=0):
    """Load K from LDS → k_frags[4][8].

    Ordering: half(SU) outer → bank inner, matching lds_K_blk_su pattern.
    tile0_bank0(8) → tile0_bank1(8) → ... → tile1_bank0(8) → ...

    Two-pass: all 64 ds_loads → wait + hw barrier → frag building.
    The s_barrier fence prevents LLVM from scheduling WMMAs between bank loads.
    """
    # Pass 1: issue all 64 ds_loads
    all_bank_loads = [[[] for _ in range(4)] for _ in range(2)]
    for half_idx in fx.range_constexpr(2):
        half_off = half_idx * K_D_HALF_OFFSET + lds_offset
        for bank in fx.range_constexpr(4):
            for i in fx.range_constexpr(NUM_K_LOADS_PER_HALF):
                byte_off = _raw(arith.constant(half_off + i * K_LOAD_STRIDE, type=T.i32))
                loaded = lds_load_b128(k_addrs[bank], byte_off)
                all_bank_loads[half_idx][bank].append(set_vgpr_bank(loaded, bank))

    rocdl.s_wait_dscnt(0)
    rocdl.s_wait_loadcnt(0)
    rocdl.s_barrier_signal(-1)
    rocdl.s_barrier_wait(-1)
    rocdl.sched_barrier(0)

    # Pass 2: build frags from waited data
    k_frags = [[] for _ in range(4)]
    for half_idx in fx.range_constexpr(2):
        for bank in fx.range_constexpr(4):
            bank_loads = all_bank_loads[half_idx][bank]
            for f in fx.range_constexpr(4):
                frag = make_wmma_frag_bf16(bank_loads[2 * f], bank_loads[2 * f + 1])
                k_frags[bank].append(set_vgpr_bank(frag, bank))
    rocdl.sched_barrier(0)
    return k_frags


def _qk_wmma_64_flydsl(k_frags, q_frags):
    """64 WMMAs → accs dict {(g_idx, tile): vec<8xf32>} with bank hints.

    QK_WMMA_INTERLEAVE=1 ordering: g_idx → k_step → tile(0,1,2,3).
    g_idx 0,1 = SU0 (frags 0-3), g_idx 2,3 = SU1 (frags 4-7).
    Tile ordering: (acc_pair0,n=0), (acc_pair0,n=1), (acc_pair1,n=0), (acc_pair1,n=1)
    avoids consecutive WMMAs with same SRCC/SRCD bank.
    """
    vec8f32_ty = ir.VectorType.get([8], ir.F32Type.get())
    zero_acc = _raw(arith.constant_vector(0.0, T.vec(8, T.f32)))

    accs = {}
    rocdl.sched_barrier(0)
    for su in fx.range_constexpr(2):
        for su_g in fx.range_constexpr(2):
            g_idx = su * 2 + su_g
            k_bank_pair, k_frag_indices = GROUP_CONFIG[g_idx]
            for k_step in fx.range_constexpr(4):
                k_frag_idx = k_frag_indices[k_step]
                for tile in fx.range_constexpr(4):
                    k_bank = k_bank_pair[tile & 1]
                    q_base = 0 if tile < 2 else 2
                    q_bank = q_base + (k_step >> 1)
                    q_frag = k_step & 1
                    acc_bank = _acc_bank(g_idx, tile)

                    key = (g_idx, tile)
                    c_operand = zero_acc if k_step == 0 else accs[key]

                    result = rocdl.wmma_f32_16x16x32_bf16(
                        vec8f32_ty,
                        k_frags[k_bank][k_frag_idx],
                        q_frags[q_bank][q_frag],
                        c_operand,
                        signA=False, signB=False, modC=0,
                        reuseA=False, reuseB=False)
                    accs[key] = set_vgpr_bank(result.result, acc_bank)
                rocdl.sched_barrier(0)
    return accs


def _causal_mask_flydsl(accs, row_pos, su_col_offset=0):
    """Apply causal mask: if row < col, set to -inf."""
    from flydsl._mlir.dialects import vector as vector_dialect

    neg_inf_raw = _raw(arith.constant(float('-inf'), type=T.f32))

    for key in accs:
        g_idx, tile = key
        bank = _acc_bank(g_idx, tile)
        col_base = ACC_COL_BASE[key] + su_col_offset
        acc = accs[key]

        for i in fx.range_constexpr(8):
            col = col_base + i
            elem = vector_dialect.extract(acc, [], static_position=[i])
            cmp = _raw(arith.cmpi(arith.CmpIPredicate.slt, row_pos,
                                   arith.constant(col, type=T.i32)))
            masked = llvm_dialect.select(cmp, neg_inf_raw, elem)
            acc = vector_dialect.insert(masked, acc, [], static_position=[i])

        accs[key] = set_vgpr_bank(acc, bank)
    return accs


def _softmax_complete_flydsl(accs, softmax_scale_raw):
    """Softmax for tile_n=128: single accs dict (4 tiles/bank, 32 elements/bank).

    1. Per-bank max tree (arith.maxnumf)
    2. Cross-lane permlanex16 + cross-bank max reduction
    3. Per-bank: fma(acc, scale, -max_scaled) → exp2 → row_sum accumulate

    Returns (row_max, row_sum).
    """
    from flydsl._mlir.dialects import arith as arith_d
    from flydsl._mlir.dialects import vector as vector_dialect

    f32 = ir.F32Type.get()
    neg_inf_raw = _raw(arith.constant(float('-inf'), type=T.f32))
    zero_raw = _raw(arith.constant(0.0, type=T.f32))
    s_zero = _raw(arith.constant(0, type=T.i32))

    tiles_by_bank = [[] for _ in range(4)]
    for key in accs:
        g_idx, tile = key
        bank = _acc_bank(g_idx, tile)
        tiles_by_bank[bank].append(key)

    # ---- Max reduction: per-bank max ----
    max_per_bank = {}
    for bank in fx.range_constexpr(4):
        running_max = set_vgpr_bank(neg_inf_raw, bank)
        for key in tiles_by_bank[bank]:
            acc = accs[key]
            for i in fx.range_constexpr(8):
                elem = vector_dialect.extract(acc, [], static_position=[i])
                running_max = arith_d.maxnumf(running_max, elem)
        max_per_bank[bank] = set_vgpr_bank(running_max, bank)
        rocdl.sched_barrier(0)

    # ---- Cross-lane permlanex16 + cross-bank max → global_max ----
    for bank in fx.range_constexpr(4):
        val = max_per_bank[bank]
        xchg = _rocdl.permlanex16(f32, val, val, s_zero, s_zero, False, False)
        max_per_bank[bank] = set_vgpr_bank(arith_d.maxnumf(val, xchg), bank)

    cross01 = arith_d.maxnumf(max_per_bank[0], max_per_bank[1])
    cross23 = arith_d.maxnumf(max_per_bank[2], max_per_bank[3])
    global_max = arith_d.maxnumf(cross01, cross23)
    rocdl.sched_barrier(0)

    # ---- Per-bank: fma(elem, scale, -ms) → exp2 → row_sum ----
    row_sum = {}
    for bank in fx.range_constexpr(4):
        neg_ms = set_vgpr_bank(arith_d.negf(arith_d.mulf(global_max, softmax_scale_raw)), bank)
        scale_b = set_vgpr_bank(softmax_scale_raw, bank)
        running_sum = set_vgpr_bank(zero_raw, bank)
        for key in tiles_by_bank[bank]:
            acc = accs[key]
            for i in fx.range_constexpr(8):
                elem = vector_dialect.extract(acc, [], static_position=[i])
                x = llvm_dialect.intr_fma(elem, scale_b, neg_ms)
                exp_val = _rocdl.exp2(f32, x)
                running_sum = arith_d.addf(running_sum, exp_val)
        row_sum[bank] = set_vgpr_bank(running_sum, bank)
        rocdl.sched_barrier(0)

    row_max = {}
    for bank in fx.range_constexpr(4):
        row_max[bank] = set_vgpr_bank(global_max, bank)

    return row_max, row_sum


def _phase5_head_index_div_flydsl(workgroup_id, num_heads):
    """head_index = workgroup_id / num_heads — LLVM auto-generates Newton-Raphson."""
    quotient = arith.divui(workgroup_id, num_heads)
    return _raw(rocdl.readfirstlane(T.i32, quotient))


def _phase6_compute_lds_offsets(wave_id):
    """Per-wave LDS base offsets for K and V TDM descriptors."""
    wid_odd = arith.andi(wave_id, arith.constant(1, type=T.i32))
    wid_half = arith.shrui(wave_id, arith.constant(1, type=T.i32))

    k_lds_base = arith.addi(
        arith.muli(wid_odd, arith.constant(PINGPONG_OFFSET, type=T.i32)),
        arith.muli(wid_half, arith.constant(0x2200, type=T.i32)))

    v_lds_base = arith.addi(
        arith.addi(
            arith.constant(V_LDS_OFFSET, type=T.i32),
            arith.muli(wid_odd, arith.constant(PINGPONG_OFFSET, type=T.i32))),
        arith.muli(wid_half, arith.constant(0x2400, type=T.i32)))

    return k_lds_base, v_lds_base


def _build_tdm_dgroup1(config_val, stride_i32):
    """Build TDM GROUP1 descriptor (vec<8xi32>)."""
    return vector.from_elements(
        T.vec(8, T.i32),
        [arith.constant(config_val, type=T.i32),
         arith.constant(256 << 16, type=T.i32),
         arith.constant(0, type=T.i32),
         arith.constant(256 << 16, type=T.i32),
         arith.constant(32, type=T.i32),
         stride_i32,
         arith.constant(0, type=T.i32),
         arith.constant(0, type=T.i32)])



def _split_i64_to_lo_hi(val_i64):
    from flydsl._mlir.dialects import arith as std_arith
    i32 = ir.IntegerType.get_signless(32)
    i64 = ir.IntegerType.get_signless(64)
    lo = std_arith.TruncIOp(i32, val_i64).result
    hi_shifted = std_arith.ShRUIOp(
        val_i64,
        std_arith.ConstantOp(i64, ir.IntegerAttr.get(i64, 32)).result).result
    hi_raw = std_arith.TruncIOp(i32, hi_shifted).result
    hi = std_arith.OrIOp(
        hi_raw,
        std_arith.ConstantOp(i32, ir.IntegerAttr.get(i32, -2147483648)).result).result
    return lo, hi


def _compute_k_global_addr(arg_K, k_offset, wave_id, stride_k_32):
    from flydsl._mlir.dialects import fly as _fly_d, arith as std_arith
    from flydsl.expr.arith import _to_raw

    i64 = ir.IntegerType.get_signless(64)
    glb_ptr_type = ir.Type.parse("!llvm.ptr<1>")

    a_raw = arg_K.__extract_to_ir_values__()[0]
    glb_ptr = _fly_d.extract_aligned_pointer_as_index(glb_ptr_type, a_raw)
    base_i64 = llvm_dialect.ptrtoint(i64, glb_ptr)

    k_off_i64 = std_arith.ExtSIOp(i64, _to_raw(k_offset)).result
    addr_i64 = std_arith.AddIOp(base_i64, k_off_i64).result

    wave_off = std_arith.MulIOp(_to_raw(wave_id), _to_raw(stride_k_32)).result
    addr_i64 = std_arith.AddIOp(addr_i64, std_arith.ExtSIOp(i64, wave_off).result).result

    return addr_i64


def _compute_v_global_addr(arg_V, v_offset, wave_id, stride_v_32):
    from flydsl._mlir.dialects import fly as _fly_d, arith as std_arith
    from flydsl.expr.arith import _to_raw

    i64 = ir.IntegerType.get_signless(64)
    glb_ptr_type = ir.Type.parse("!llvm.ptr<1>")

    a_raw = arg_V.__extract_to_ir_values__()[0]
    glb_ptr = _fly_d.extract_aligned_pointer_as_index(glb_ptr_type, a_raw)
    base_i64 = llvm_dialect.ptrtoint(i64, glb_ptr)

    v_off_i64 = std_arith.ExtSIOp(i64, _to_raw(v_offset)).result
    addr_i64 = std_arith.AddIOp(base_i64, v_off_i64).result

    wave_off = std_arith.MulIOp(_to_raw(wave_id), _to_raw(stride_v_32)).result
    addr_i64 = std_arith.AddIOp(addr_i64, std_arith.ExtSIOp(i64, wave_off).result).result

    return addr_i64


def _k_tdm_setup(arg_K, k_offset, stride_k_seq, stride_k_32, wave_id):
    """Common K TDM setup: returns (dgroup1, addr_i64, stride_adv_i64)."""
    from flydsl._mlir.dialects import arith as std_arith
    from flydsl.expr.arith import _to_raw

    i64 = ir.IntegerType.get_signless(64)
    k_dgroup1 = _build_tdm_dgroup1(_K_TDM_CONFIG, stride_k_seq)
    k_addr_i64 = _compute_k_global_addr(arg_K, k_offset, wave_id, stride_k_32)
    stride_adv_i64 = std_arith.ExtSIOp(i64, _to_raw(stride_k_32)).result
    return k_dgroup1, k_addr_i64, stride_adv_i64


def _k_tdm_issue_pair(k_dgroup1, addr_i64, stride_adv_i64, lds_off_0, lds_off_1, wait_count=1):
    """Issue 2 K TDM loads. Returns addr after the 2nd load (for next pair)."""
    from flydsl._mlir.dialects import arith as std_arith

    pred = arith.constant(1, type=T.i32)
    cur_addr = addr_i64
    for i, lds_off in enumerate([lds_off_0, lds_off_1]):
        addr_lo, addr_hi = _split_i64_to_lo_hi(cur_addr)
        dg0 = vector.from_elements(T.vec(4, T.i32), [pred, lds_off, addr_lo, addr_hi])
        tdm_ops.tensor_load_2d(tdm_ops.TDMDescriptor2D(dg0, k_dgroup1))
        rocdl.s_barrier_signal(-1)
        rocdl.s_barrier_wait(-1)
        if i == 0:
            cur_addr = std_arith.AddIOp(cur_addr, stride_adv_i64).result

    _s_wait_tensorcnt(wait_count)
    rocdl.s_barrier_signal(-1)
    rocdl.s_barrier_wait(-1)

    return std_arith.AddIOp(cur_addr, stride_adv_i64).result


def _phase_first_v_tdm_flydsl(arg_V, v_offset, v_lds_base, stride_v_seq, stride_v_32, wave_id):
    """Issue 2 V TDM copies (Global → LDS) for V block 0."""
    from flydsl._mlir.dialects import arith as std_arith
    from flydsl.expr.arith import _to_raw

    i64 = ir.IntegerType.get_signless(64)
    v_dgroup1 = _build_tdm_dgroup1(_V_TDM_CONFIG, stride_v_seq)
    v_addr_i64 = _compute_v_global_addr(arg_V, v_offset, wave_id, stride_v_32)
    stride_adv_i64 = std_arith.ExtSIOp(i64, _to_raw(stride_v_32)).result
    pred = arith.constant(1, type=T.i32)

    lds_offsets = [
        v_lds_base,
        arith.addi(v_lds_base, arith.constant(V_SU1_OFFSET, type=T.i32)),
    ]

    cur_addr = v_addr_i64
    for i, lds_off in enumerate(lds_offsets):
        addr_lo, addr_hi = _split_i64_to_lo_hi(cur_addr)
        dg0 = vector.from_elements(T.vec(4, T.i32), [pred, lds_off, addr_lo, addr_hi])
        tdm_ops.tensor_load_2d(tdm_ops.TDMDescriptor2D(dg0, v_dgroup1))
        rocdl.s_barrier_signal(-1)
        rocdl.s_barrier_wait(-1)
        if i < len(lds_offsets) - 1:
            cur_addr = std_arith.AddIOp(cur_addr, stride_adv_i64).result


def _phase7_softmax_init_flydsl():
    """Init softmax state: row_max=-inf, row_sum=0 across 4 banks."""
    neg_inf = _raw(arith.constant(float('-inf'), type=T.f32))
    zero = _raw(arith.constant(0.0, type=T.f32))
    row_max = {}
    row_sum = {}
    for bank in range(4):
        row_max[bank] = set_vgpr_bank(neg_inf, bank)
        row_sum[bank] = set_vgpr_bank(zero, bank)
    return row_max, row_sum


def _phase8_zero_o_accum_flydsl():
    """Zero O output accumulators across 4 banks, 4 tiles each."""
    zero_vec = _raw(arith.constant_vector(0.0, T.vec(8, T.f32)))
    o_accs = {}
    rocdl.sched_barrier(0)
    for bank in range(4):
        for tile in range(4):
            o_accs[(bank, tile)] = set_vgpr_bank(zero_vec, bank)
    rocdl.sched_barrier(0)
    return o_accs








def _phase_v_tdm_blk1_flydsl(arg_V, v_offset, v_lds_base, stride_v_seq, stride_v_32, wave_id):
    """V TDM block 1 (2 copies)."""
    from flydsl._mlir.dialects import arith as std_arith
    from flydsl.expr.arith import _to_raw

    i64 = ir.IntegerType.get_signless(64)
    v_dgroup1 = _build_tdm_dgroup1(_V_TDM_CONFIG, stride_v_seq)
    pred = arith.constant(1, type=T.i32)

    v_blk_inc = arith.muli(arith.constant(K_TILE_N, type=T.i32), stride_v_seq)
    v_offset_blk1 = arith.addi(v_offset, v_blk_inc)
    v_addr_i64 = _compute_v_global_addr(arg_V, v_offset_blk1, wave_id, stride_v_32)
    stride_adv_i64 = std_arith.ExtSIOp(i64, _to_raw(stride_v_32)).result

    lds_offsets = [
        arith.addi(v_lds_base, arith.constant(2 * V_SU1_OFFSET, type=T.i32)),
        arith.addi(v_lds_base, arith.constant(3 * V_SU1_OFFSET, type=T.i32)),
    ]

    cur_addr = v_addr_i64
    for i, lds_off in enumerate(lds_offsets):
        addr_lo, addr_hi = _split_i64_to_lo_hi(cur_addr)
        dg0 = vector.from_elements(T.vec(4, T.i32), [pred, lds_off, addr_lo, addr_hi])
        tdm_ops.tensor_load_2d(tdm_ops.TDMDescriptor2D(dg0, v_dgroup1))
        rocdl.s_barrier_signal(-1)
        rocdl.s_barrier_wait(-1)
        if i < len(lds_offsets) - 1:
            cur_addr = std_arith.AddIOp(cur_addr, stride_adv_i64).result

    _s_wait_tensorcnt(4)
    rocdl.s_barrier_signal(-1)
    rocdl.s_barrier_wait(-1)


# ============================================================================
# Main Kernel
# ============================================================================

@flyc.kernel(known_block_size=[BLOCK_SIZE, 1, 1])
def fmha_fwd_prologue(
    ptr_O: fx.Tensor,
    ptr_Q: fx.Tensor,
    ptr_K: fx.Tensor,
    ptr_V: fx.Tensor,
    ptr_LSE: fx.Tensor,
    scalar_f: fx.Float32,
    q_seq_len: fx.Int32,
    stride_q_seq: fx.Int32,
    stride_q_tg: fx.Int32,
    stride_q_head: fx.Int32,
    stride_q_batch: fx.Int32,
    gqa: fx.Int32,
    stride_k_seq: fx.Int32,
    stride_k_head: fx.Int32,
    stride_k_batch: fx.Int32,
    opt: fx.Int32,
    lse: fx.Int32,
    kv_seq_len: fx.Int32,
    qk_head_dim: fx.Int32,
    v_head_dim: fx.Int32,
    q_head_num: fx.Int32,
    stride_v_seq: fx.Int32,
    stride_v_head: fx.Int32,
    stride_v_batch: fx.Int32,
    stride_o_seq: fx.Int32,
    stride_o_head: fx.Int32,
    stride_o_batch: fx.Int32,
):
    from flydsl.expr.arith import _to_raw

    scalar_f = _to_raw(scalar_f)
    q_seq_len = _to_raw(q_seq_len)
    stride_q_seq = _to_raw(stride_q_seq)
    stride_q_tg = _to_raw(stride_q_tg)
    stride_q_head = _to_raw(stride_q_head)
    stride_q_batch = _to_raw(stride_q_batch)
    gqa = _to_raw(gqa)
    stride_k_seq = _to_raw(stride_k_seq)
    stride_k_head = _to_raw(stride_k_head)
    stride_k_batch = _to_raw(stride_k_batch)
    opt = _to_raw(opt)
    lse = _to_raw(lse)
    kv_seq_len = _to_raw(kv_seq_len)
    qk_head_dim = _to_raw(qk_head_dim)
    v_head_dim = _to_raw(v_head_dim)
    q_head_num = _to_raw(q_head_num)
    stride_v_seq = _to_raw(stride_v_seq)
    stride_v_head = _to_raw(stride_v_head)
    stride_v_batch = _to_raw(stride_v_batch)
    stride_o_seq = _to_raw(stride_o_seq)
    stride_o_head = _to_raw(stride_o_head)
    stride_o_batch = _to_raw(stride_o_batch)

    # ---- Phase 1-3: Hardware setup + IDs + Q address ----

    _setreg(2074, 2)  # WAVE_SCHED_MODE[1:0] = 2
    _rocdl.s_nop(0)

    tx = arith.index_cast(T.i32, gpu.thread_id("x"))
    lane_id = arith.andi(tx, arith.constant(31, type=T.i32))
    wave_id = arith.shrui(tx, arith.constant(5, type=T.i32))

    bx = arith.index_cast(T.i32, gpu.block_id("y"))
    by = arith.index_cast(T.i32, gpu.block_id("x"))
    bz = arith.index_cast(T.i32, gpu.block_id("z"))

    q_offset = arith.addi(
        arith.addi(arith.muli(bz, stride_q_batch), arith.muli(by, stride_q_head)),
        arith.muli(bx, stride_q_tg))

    q_nbytes = arith.muli(q_seq_len, stride_q_seq)
    q_rsrc = buffer_ops.create_buffer_resource(
        ptr_Q, max_size=False,
        num_records_bytes=arith.index_cast(T.index, q_nbytes))

    # ---- Phase 4: Q Load — FlyDSL ----

    q_frags = _phase4_q_load_flydsl(lane_id, _raw(q_rsrc), stride_q_seq, wave_id)
    rocdl.sched_barrier(0)

    # ---- Phase 5: Head Index Division ----

    head_index = _phase5_head_index_div_flydsl(by, gqa)

    # ---- Phase 6: K/V Base + TDM Descriptors ----

    k_offset = arith.addi(arith.muli(bz, stride_k_batch),
                           arith.muli(head_index, stride_k_head))
    v_offset = arith.addi(arith.muli(bz, stride_v_batch),
                           arith.muli(head_index, stride_v_head))

    k_lds_base, v_lds_base = _phase6_compute_lds_offsets(wave_id)

    # ---- Phase 7/8: Softmax Init + O Zero — FlyDSL ----

    _row_max_init, _row_sum_init = _phase7_softmax_init_flydsl()
    _o_accs = _phase8_zero_o_accum_flydsl()

    # ---- Phase 9: K Addr Gen + TDM + LDS Load + QK WMMA ----

    rocdl.sched_barrier(0)
    k_addrs = _phase9a_k_lds_addr_gen(lane_id, wave_id)
    _phase9b_v_lds_addr_gen(lane_id, wave_id)

    stride_k_32 = arith.muli(arith.constant(32, type=T.i32), stride_k_seq)
    rocdl.sched_barrier(0)

    # ---- Phase 9c-0: K TDM pair 0 (tiles 0+1) ----

    k_dgroup1, k_addr_i64, stride_adv_i64 = _k_tdm_setup(
        ptr_K, k_offset, stride_k_seq, stride_k_32, wave_id)
    lds_off_0 = _raw(k_lds_base)
    lds_off_1 = _raw(arith.addi(k_lds_base, arith.constant(K_SU1_OFFSET, type=T.i32)))
    next_addr = _k_tdm_issue_pair(
        k_dgroup1, k_addr_i64, stride_adv_i64, lds_off_0, lds_off_1, wait_count=1)

    # ---- Phase 9d: K ds_load SU0+SU1 + WMMA (64 WMMAs, tile_n=128) ----

    rocdl.sched_barrier(0)
    k_frags = _phase9d_k_lds_load_flydsl(k_addrs, lds_offset=0)
    accs = _qk_wmma_64_flydsl(k_frags, q_frags)
    rocdl.sched_barrier(0)

    # ---- Phase 11: V TDM Block 0 — FlyDSL ----

    stride_v_32 = arith.muli(arith.constant(32, type=T.i32), stride_v_seq)
    rocdl.sched_barrier(0)
    _phase_first_v_tdm_flydsl(ptr_V, v_offset, v_lds_base, stride_v_seq, stride_v_32, wave_id)
    rocdl.sched_barrier(0)

    # ---- Softmax — FlyDSL (tile_n=128, single accs) ----

    from flydsl._mlir.dialects import arith as _mlir_arith
    log2e = _mlir_arith.constant(ir.F32Type.get(), 1.4426950408889634)
    softmax_scale_raw = _mlir_arith.mulf(log2e, scalar_f)

    row_max, row_sum = _softmax_complete_flydsl(accs, softmax_scale_raw)

    # N=128: V block 0 already covers all 128 V columns, no block 1 needed



if __name__ == "__main__":
    import os
    os.environ.setdefault("FLYDSL_GPU_ARCH", "gfx1250")
    os.environ.setdefault("FLYDSL_RUNTIME_ENABLE_CACHE", "0")
    os.environ.setdefault("COMPILE_ONLY", "1")

    import torch
    arch = os.environ.get("FLYDSL_GPU_ARCH", "gfx1250")

    B, H, S_q, S_kv, D = 1, 1, 128, 256, 128
    dtype = torch.bfloat16

    Q = torch.zeros(B * H * S_q * D, dtype=dtype)
    K = torch.zeros(B * H * S_kv * D, dtype=dtype)
    V = torch.zeros(B * H * S_kv * D, dtype=dtype)
    O = torch.zeros(B * H * S_q * D, dtype=dtype)
    LSE = torch.zeros(B * H * S_q, dtype=torch.float32)

    stride_q_seq = D
    stride_q_head = S_q * D
    stride_q_batch = H * S_q * D
    stride_q_tg = stride_q_batch
    stride_k_seq = D
    stride_k_head = S_kv * D
    stride_k_batch = H * S_kv * D
    stride_v_seq = D
    stride_v_head = S_kv * D
    stride_v_batch = H * S_kv * D
    stride_o_seq = D
    stride_o_head = S_q * D
    stride_o_batch = H * S_q * D

    @flyc.jit
    def launch_prologue(
        ptr_O: fx.Tensor, ptr_Q: fx.Tensor, ptr_K: fx.Tensor,
        ptr_V: fx.Tensor, ptr_LSE: fx.Tensor,
        scalar_f: fx.Float32, q_seq_len: fx.Int32,
        stride_q_seq: fx.Int32, stride_q_tg: fx.Int32,
        stride_q_head: fx.Int32, stride_q_batch: fx.Int32,
        gqa: fx.Int32,
        stride_k_seq: fx.Int32, stride_k_head: fx.Int32,
        stride_k_batch: fx.Int32,
        opt: fx.Int32, lse: fx.Int32,
        kv_seq_len: fx.Int32, qk_head_dim: fx.Int32,
        v_head_dim: fx.Int32, q_head_num: fx.Int32,
        stride_v_seq: fx.Int32, stride_v_head: fx.Int32,
        stride_v_batch: fx.Int32,
        stride_o_seq: fx.Int32, stride_o_head: fx.Int32,
        stride_o_batch: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        fmha_fwd_prologue(
            ptr_O, ptr_Q, ptr_K, ptr_V, ptr_LSE,
            scalar_f, q_seq_len,
            stride_q_seq, stride_q_tg, stride_q_head, stride_q_batch,
            gqa,
            stride_k_seq, stride_k_head, stride_k_batch,
            opt, lse,
            kv_seq_len, qk_head_dim, v_head_dim, q_head_num,
            stride_v_seq, stride_v_head, stride_v_batch,
            stride_o_seq, stride_o_head, stride_o_batch,
        ).launch(grid=(1, H, B), block=(BLOCK_SIZE, 1, 1), stream=stream)

    launch_prologue(
        O, Q, K, V, LSE,
        1.0 / (D ** 0.5),
        S_q,
        stride_q_seq, stride_q_tg, stride_q_head, stride_q_batch,
        1,
        stride_k_seq, stride_k_head, stride_k_batch,
        0, 0,
        S_kv, D, D, H,
        stride_v_seq, stride_v_head, stride_v_batch,
        stride_o_seq, stride_o_head, stride_o_batch,
    )
    print(f"Done ({arch})")

