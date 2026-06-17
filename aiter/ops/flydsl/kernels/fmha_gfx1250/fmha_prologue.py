"""D128 BF16 FMHA Forward Prologue — gfx1250 Pure FlyDSL Implementation.

FlyDSL-native implementation of the FMHA prologue with compiler-managed
VGPR bank allocation via @llvm.amdgcn.set.vgpr.bank intrinsic.

Target: gfx1250 (MI450), wave32, 4 waves per thread-group, 128 threads.
All phases use FlyDSL — zero inline ASM. 128×128 compute (TDM loads 256).

Reference: BF16_FMHA_FWD_D128_1TG_4W_32mx4_256nx1_cas_brd_rxy.s
"""

from __future__ import annotations

import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm as llvm_dialect
from flydsl.expr import arith, rocdl, vector
from flydsl.expr.rocdl import tdm_ops
from flydsl.expr.typing import T
from .fmha_core_loop import (
    QK_HDIM,
    _rocdl_permlanex16,
    _rocdl_exp2,
    set_vgpr_bank,
)

# ============================================================================
# Constants
# ============================================================================

WAVE_SIZE = 32
NUM_WAVES = 4
BLOCK_SIZE = WAVE_SIZE * NUM_WAVES

Q_TILE_M = 128
Q_TILE_D = QK_HDIM  # = 192
K_TILE_N = 128  # TDM load tile = compute tile (N=128)
K_COMPUTE_N = 128  # compute tile (QK WMMA uses SU0+SU1)
K_SU_SIZE = 64
NUM_K_SU = 2  # TDM loads 2 SUs (= compute)
NUM_K_SU_COMPUTE = 2

# TDM dim0=200 -> LDS inner stride = 200*2 = 400B
# (2-way bank conflicts)
K_ROW_BYTES = 400
V_ROW_BYTES = 288
K_SU_HALF_OFFSET = 0x1900  # 16 * K_ROW_BYTES = 16 * 400 = 6400
V_SU_HALF_OFFSET = 0x1200
V_LDS_OFFSET = 0x8800  # after 2 K SUs: 2×0x4400
PINGPONG_OFFSET = 0x11800  # one-ping: K(0x8800)+V(0x9000)
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
    (0, 0): 0,
    (0, 1): 16,
    (2, 0): 64,
    (2, 1): 80,
    (1, 0): 8,
    (1, 1): 24,
    (3, 0): 72,
    (3, 1): 88,
    (0, 2): 32,
    (0, 3): 48,
    (2, 2): 96,
    (2, 3): 112,
    (1, 2): 40,
    (1, 3): 56,
    (3, 2): 104,
    (3, 3): 120,
}

# TDM config constants.
# K: dim0=192 (QK_HDIM), no padding. QK_HDIM=192 is not a multiple of any
# power-of-2 pad_interval that fits in one pad per row, so we skip padding to
# avoid the continuous-stream rotation bug. No bank-conflict padding for K.
_K_TDM_CONFIG = 1 << 16  # data_size=1 (bf16), pad_enable=0
# V: dim0=128, pad_interval=128 elems=64dwords → enc_interval=5, 32B pad → enc_amount=7
_V_TDM_CONFIG = (1 << 20) | (5 << 22) | (7 << 25)

# ============================================================================
# Core Helpers
# ============================================================================


_NUW_ATTR = None


def _get_nuw():
    global _NUW_ATTR
    if _NUW_ATTR is None:
        _NUW_ATTR = ir.Attribute.parse("#arith.overflow<nuw>")
    return _NUW_ATTR


def _add_nuw(a, b):
    """arith.addi with nuw flag — enables gfx1250 buffer offset folding."""
    return arith.unwrap(arith.addi(a, b, overflow_flags=_get_nuw()))


def _mul_nuw(a, b):
    """arith.muli with nuw flag — preserves nuw through constant folding."""
    return arith.unwrap(arith.muli(a, b, overflow_flags=_get_nuw()))


def _acc_bank(g_idx, tile):
    return (g_idx & 1) + 2 * (1 if tile >= 2 else 0)


def _permlanex16_f32(src):
    """v_permlanex16_b32 via inline asm (intrinsic lacks SelectionDAG lowering)."""
    i32_ty = ir.IntegerType.get_signless(32)
    src_i32 = arith.bitcast(i32_ty, src)
    result_i32 = llvm_dialect.inline_asm(
        i32_ty,
        [src_i32],
        "v_permlanex16_b32 $0, $1, 0, 0",
        "=v,0",
        has_side_effects=True,
    )
    return arith.bitcast(ir.F32Type.get(), result_i32)


def _setreg(hwreg_enc, value):
    """s_setreg_imm32_b32 via llvm.amdgcn.s.setreg intrinsic.
    hwreg_enc = id | (offset << 6) | ((size-1) << 11)"""
    imm = arith.unwrap(arith.constant(hwreg_enc, type=T.i32))
    val = arith.unwrap(arith.constant(value, type=T.i32))
    llvm_dialect.call_intrinsic(None, "llvm.amdgcn.s.setreg", [imm, val], [], [])


def _emit_void(inst_str, operands=None, constraints="", **kwargs):
    llvm_dialect.inline_asm(
        None, operands or [], inst_str, constraints, has_side_effects=True, **kwargs
    )


def _s_wait_tensorcnt(cnt):
    from flydsl.expr import rocdl as _rocdl_expr

    _rocdl_expr.s_wait_tensorcnt(cnt)


# ============================================================================
# LDS / Fragment Helpers
# ============================================================================


def lds_load_b128(lds_base_raw, byte_offset_raw):
    lds_ptr_ty = ir.Type.parse("!llvm.ptr<3>")
    total = arith.unwrap(arith.addi(lds_base_raw, byte_offset_raw))
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


def _phase4_q_load_flydsl(
    lane_id,
    q_rsrc,
    stride_q_seq,
    wave_id,
    q_tile_offset_bytes=None,
):
    """Load Q tile (QK_HDIM x TG_Q_ROWS bf16) ->
    q_frags[4][Q_FRAGS_PER_BANK] with bank hints.

    For QK_HDIM=192: 6 loads per bank (3 frags × 2 loads each), bank1 offset=192 bytes.
    For QK_HDIM=128: 4 loads per bank (2 frags × 2 loads each), bank1 offset=128 bytes.
    """
    lane_lo = arith.andi(lane_id, arith.constant(15, type=T.i32))
    lane_hi = arith.shrui(lane_id, arith.constant(4, type=T.i32))
    base = arith.addi(
        arith.muli(lane_lo, stride_q_seq),
        arith.muli(lane_hi, arith.constant(16, type=T.i32)),
    )
    wave_off = arith.muli(
        arith.muli(wave_id, arith.constant(32, type=T.i32)), stride_q_seq
    )
    q_byte_off = arith.addi(base, wave_off)

    q_elem_off = arith.shrui(q_byte_off, arith.constant(2, type=T.i32))

    vec4i32_ty = ir.VectorType.get([4], ir.IntegerType.get_signless(32))
    soff_zero = arith.unwrap(arith.constant(0, type=T.i32))
    aux_zero = arith.unwrap(arith.constant(0, type=T.i32))

    four_i32 = arith.unwrap(arith.constant(4, type=T.i32))
    q_base_bytes = _mul_nuw(arith.unwrap(q_elem_off), four_i32)
    if q_tile_offset_bytes is not None:
        q_base_bytes = _add_nuw(q_tile_offset_bytes, q_base_bytes)
    stride_16_bytes = arith.unwrap(
        arith.muli(stride_q_seq, arith.constant(16, type=T.i32))
    )

    # K-half byte offset = QK_HDIM bytes (splits K cols in half per bank pair)
    _K_HALF_BYTES = QK_HDIM  # 192 for 192-dim, 128 for 128-dim
    # Each bank covers half of QK_HDIM: QK_HDIM/2 elements / WMMA_K(32) = frags_per_bank
    # = 3 for 192-dim (96 K cols / 32 = 3), 2 for 128-dim
    _FRAGS_PER_BANK = (QK_HDIM // 2) // 32
    _LOADS_PER_BANK = _FRAGS_PER_BANK * 2  # 2 raw loads per v16bf16 frag

    bank_offsets_bytes = [
        arith.unwrap(arith.constant(0, type=T.i32)),
        arith.unwrap(arith.constant(_K_HALF_BYTES, type=T.i32)),
        stride_16_bytes,
        _add_nuw(
            stride_16_bytes,
            arith.unwrap(arith.constant(_K_HALF_BYTES, type=T.i32)),
        ),
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
                voff = _add_nuw(
                    bank_voff,
                    arith.unwrap(arith.constant(i * 32, type=T.i32)),
                )
            loaded = rocdl.raw_ptr_buffer_load(
                vec4i32_ty, q_rsrc, voff, soff_zero, aux_zero
            )
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
        arith.muli(lane_hi, arith.constant(16, type=T.i32)),
    )

    seg1 = arith.addi(base, arith.constant(K_SU_HALF_OFFSET, type=T.i32))
    seg2 = arith.addi(base, arith.constant(PINGPONG_OFFSET, type=T.i32))
    seg3 = arith.addi(seg1, arith.constant(PINGPONG_OFFSET, type=T.i32))

    wave_is_odd = arith.cmpi(
        arith.CmpIPredicate.ne,
        arith.andi(wave_id, arith.constant(1, type=T.i32)),
        arith.constant(0, type=T.i32),
    )

    a0 = arith.unwrap(arith.select(wave_is_odd, seg2, base))
    a1 = arith.unwrap(arith.select(wave_is_odd, seg3, seg1))
    a2 = arith.unwrap(arith.select(wave_is_odd, base, seg2))
    a3 = arith.unwrap(arith.select(wave_is_odd, seg1, seg3))

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
        arith.constant(4, type=T.i32),
    )

    addr_base = arith.addi(
        arith.addi(arith.muli(row, arith.constant(V_ROW_BYTES, type=T.i32)), sub_col),
        arith.constant(V_LDS_OFFSET, type=T.i32),
    )
    addr_half = arith.addi(addr_base, arith.constant(V_SU_HALF_OFFSET, type=T.i32))

    seg2_a = arith.addi(addr_base, arith.constant(PINGPONG_OFFSET, type=T.i32))
    seg2_h = arith.addi(addr_half, arith.constant(PINGPONG_OFFSET, type=T.i32))

    wave_is_odd = arith.cmpi(
        arith.CmpIPredicate.ne,
        arith.andi(wave_id, arith.constant(1, type=T.i32)),
        arith.constant(0, type=T.i32),
    )

    s0_a = arith.unwrap(arith.select(wave_is_odd, seg2_a, addr_base))
    s0_h = arith.unwrap(arith.select(wave_is_odd, seg2_h, addr_half))
    s2_a = arith.unwrap(arith.select(wave_is_odd, addr_base, seg2_a))
    s2_h = arith.unwrap(arith.select(wave_is_odd, addr_half, seg2_h))

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
                byte_off = arith.unwrap(
                    arith.constant(
                        half_off + i * K_LOAD_STRIDE,
                        type=T.i32,
                    )
                )
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
    zero_acc = arith.unwrap(arith.constant_vector(0.0, T.vec(8, T.f32)))

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
                        signA=False,
                        signB=False,
                        modC=0,
                        reuseA=False,
                        reuseB=False,
                    )
                    accs[key] = set_vgpr_bank(result.result, acc_bank)
                rocdl.sched_barrier(0)
    return accs


def _causal_mask_flydsl(accs, row_pos, su_col_offset=0):
    """Apply causal mask: if row < col, set to -inf."""

    neg_inf_raw = arith.unwrap(arith.constant(float("-inf"), type=T.f32))

    for key in accs:
        g_idx, tile = key
        bank = _acc_bank(g_idx, tile)
        col_base = ACC_COL_BASE[key] + su_col_offset
        acc = accs[key]

        for i in fx.range_constexpr(8):
            col = col_base + i
            elem = vector.extract(acc, [], static_position=[i])
            cmp = arith.unwrap(
                arith.cmpi(
                    arith.CmpIPredicate.slt, row_pos, arith.constant(col, type=T.i32)
                )
            )
            masked = llvm_dialect.select(cmp, neg_inf_raw, elem)
            acc = vector.insert(masked, acc, [], static_position=[i])

        accs[key] = set_vgpr_bank(acc, bank)
    return accs


def _softmax_complete_flydsl(accs, softmax_scale_raw):
    """Softmax for tile_n=128: single accs dict (4 tiles/bank, 32 elements/bank).

    1. Per-bank max tree (arith.maxnumf)
    2. Cross-lane permlanex16 + cross-bank max reduction
    3. Per-bank: fma(acc, scale, -max_scaled) → exp2 → row_sum accumulate

    Returns (row_max, row_sum).
    """

    f32 = ir.F32Type.get()
    neg_inf_raw = arith.unwrap(arith.constant(float("-inf"), type=T.f32))
    zero_raw = arith.unwrap(arith.constant(0.0, type=T.f32))
    s_zero = arith.unwrap(arith.constant(0, type=T.i32))

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
                elem = vector.extract(acc, [], static_position=[i])
                running_max = arith.maxnumf(running_max, elem)
        max_per_bank[bank] = set_vgpr_bank(running_max, bank)
        rocdl.sched_barrier(0)

    # ---- Cross-lane permlanex16 + cross-bank max → global_max ----
    for bank in fx.range_constexpr(4):
        val = max_per_bank[bank]
        xchg = _rocdl_permlanex16(f32, val, val, s_zero, s_zero, False, False)
        max_per_bank[bank] = set_vgpr_bank(arith.maxnumf(val, xchg), bank)

    cross01 = arith.maxnumf(max_per_bank[0], max_per_bank[1])
    cross23 = arith.maxnumf(max_per_bank[2], max_per_bank[3])
    global_max = arith.maxnumf(cross01, cross23)
    rocdl.sched_barrier(0)

    # ---- Per-bank: fma(elem, scale, -ms) → exp2 → row_sum ----
    row_sum = {}
    for bank in fx.range_constexpr(4):
        neg_ms = set_vgpr_bank(
            arith.negf(arith.mulf(global_max, softmax_scale_raw)),
            bank,
        )
        scale_b = set_vgpr_bank(softmax_scale_raw, bank)
        running_sum = set_vgpr_bank(zero_raw, bank)
        for key in tiles_by_bank[bank]:
            acc = accs[key]
            for i in fx.range_constexpr(8):
                elem = vector.extract(acc, [], static_position=[i])
                x = llvm_dialect.intr_fma(elem, scale_b, neg_ms)
                exp_val = _rocdl_exp2(f32, x)
                running_sum = arith.addf(running_sum, exp_val)
        row_sum[bank] = set_vgpr_bank(running_sum, bank)
        rocdl.sched_barrier(0)

    row_max = {}
    for bank in fx.range_constexpr(4):
        row_max[bank] = set_vgpr_bank(global_max, bank)

    return row_max, row_sum


def _phase5_head_index_div_flydsl(workgroup_id, num_heads):
    """head_index = workgroup_id / num_heads — LLVM auto-generates Newton-Raphson."""
    quotient = arith.divui(workgroup_id, num_heads)
    return arith.unwrap(rocdl.readfirstlane(T.i32, quotient))


def _phase6_compute_lds_offsets(wave_id):
    """Per-wave LDS base offsets for K and V TDM descriptors."""
    wid_odd = arith.andi(wave_id, arith.constant(1, type=T.i32))
    wid_half = arith.shrui(wave_id, arith.constant(1, type=T.i32))

    k_lds_base = arith.addi(
        arith.muli(wid_odd, arith.constant(PINGPONG_OFFSET, type=T.i32)),
        arith.muli(wid_half, arith.constant(0x2200, type=T.i32)),
    )

    v_lds_base = arith.addi(
        arith.addi(
            arith.constant(V_LDS_OFFSET, type=T.i32),
            arith.muli(wid_odd, arith.constant(PINGPONG_OFFSET, type=T.i32)),
        ),
        arith.muli(wid_half, arith.constant(0x2400, type=T.i32)),
    )

    return k_lds_base, v_lds_base


def _build_tdm_dgroup1(config_val, stride_i32):
    """Build TDM GROUP1 descriptor (vec<8xi32>)."""
    return vector.from_elements(
        T.vec(8, T.i32),
        [
            arith.constant(config_val, type=T.i32),
            arith.constant(256 << 16, type=T.i32),
            arith.constant(0, type=T.i32),
            arith.constant(256 << 16, type=T.i32),
            arith.constant(32, type=T.i32),
            stride_i32,
            arith.constant(0, type=T.i32),
            arith.constant(0, type=T.i32),
        ],
    )


def _split_i64_to_lo_hi(val_i64):
    i32 = ir.IntegerType.get_signless(32)
    lo = arith.trunci(i32, val_i64)
    hi_shifted = arith.shrui(val_i64, arith.constant(32, type=T.i64))
    hi_raw = arith.trunci(i32, hi_shifted)
    hi = arith.ori(hi_raw, arith.constant(-2147483648, type=T.i32))
    return lo, hi


def _compute_k_global_addr(arg_K, k_offset, wave_id, stride_k_32):
    from flydsl._mlir.dialects import fly as _fly_d

    i64 = ir.IntegerType.get_signless(64)
    glb_ptr_type = ir.Type.parse("!llvm.ptr<1>")

    a_raw = arg_K.__extract_to_ir_values__()[0]
    glb_ptr = _fly_d.extract_aligned_pointer_as_index(glb_ptr_type, a_raw)
    base_i64 = llvm_dialect.ptrtoint(i64, glb_ptr)

    k_off_i64 = arith.extsi(i64, arith.unwrap(k_offset))
    addr_i64 = arith.addi(base_i64, k_off_i64)

    wave_off = arith.muli(arith.unwrap(wave_id), arith.unwrap(stride_k_32))
    addr_i64 = arith.addi(addr_i64, arith.extsi(i64, wave_off))

    return addr_i64


def _compute_v_global_addr(arg_V, v_offset, wave_id, stride_v_32):
    from flydsl._mlir.dialects import fly as _fly_d

    i64 = ir.IntegerType.get_signless(64)
    glb_ptr_type = ir.Type.parse("!llvm.ptr<1>")

    a_raw = arg_V.__extract_to_ir_values__()[0]
    glb_ptr = _fly_d.extract_aligned_pointer_as_index(glb_ptr_type, a_raw)
    base_i64 = llvm_dialect.ptrtoint(i64, glb_ptr)

    v_off_i64 = arith.extsi(i64, arith.unwrap(v_offset))
    addr_i64 = arith.addi(base_i64, v_off_i64)

    wave_off = arith.muli(arith.unwrap(wave_id), arith.unwrap(stride_v_32))
    addr_i64 = arith.addi(addr_i64, arith.extsi(i64, wave_off))

    return addr_i64


def _k_tdm_setup(arg_K, k_offset, stride_k_seq, stride_k_32, wave_id):
    """Common K TDM setup: returns (dgroup1, addr_i64, stride_adv_i64)."""
    i64 = ir.IntegerType.get_signless(64)
    k_dgroup1 = _build_tdm_dgroup1(_K_TDM_CONFIG, stride_k_seq)
    k_addr_i64 = _compute_k_global_addr(arg_K, k_offset, wave_id, stride_k_32)
    stride_adv_i64 = arith.extsi(i64, arith.unwrap(stride_k_32))
    return k_dgroup1, k_addr_i64, stride_adv_i64


def _k_tdm_issue_pair(
    k_dgroup1,
    addr_i64,
    stride_adv_i64,
    lds_off_0,
    lds_off_1,
    wait_count=1,
):
    """Issue 2 K TDM loads. Returns addr after the 2nd load (for next pair)."""
    pred = arith.constant(1, type=T.i32)
    cur_addr = addr_i64
    for i, lds_off in enumerate([lds_off_0, lds_off_1]):
        addr_lo, addr_hi = _split_i64_to_lo_hi(cur_addr)
        dg0 = vector.from_elements(T.vec(4, T.i32), [pred, lds_off, addr_lo, addr_hi])
        tdm_ops.tensor_load_2d(tdm_ops.TDMDescriptor2D(dg0, k_dgroup1))
        rocdl.s_barrier_signal(-1)
        rocdl.s_barrier_wait(-1)
        if i == 0:
            cur_addr = arith.addi(cur_addr, stride_adv_i64)

    _s_wait_tensorcnt(wait_count)
    rocdl.s_barrier_signal(-1)
    rocdl.s_barrier_wait(-1)

    return arith.addi(cur_addr, stride_adv_i64)


def _phase_first_v_tdm_flydsl(
    arg_V,
    v_offset,
    v_lds_base,
    stride_v_seq,
    stride_v_32,
    wave_id,
):
    """Issue 2 V TDM copies (Global → LDS) for V block 0."""
    i64 = ir.IntegerType.get_signless(64)
    v_dgroup1 = _build_tdm_dgroup1(_V_TDM_CONFIG, stride_v_seq)
    v_addr_i64 = _compute_v_global_addr(arg_V, v_offset, wave_id, stride_v_32)
    stride_adv_i64 = arith.extsi(i64, arith.unwrap(stride_v_32))
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
            cur_addr = arith.addi(cur_addr, stride_adv_i64)


def _phase7_softmax_init_flydsl():
    """Init softmax state: row_max=-inf, row_sum=0 across 4 banks."""
    neg_inf = arith.unwrap(arith.constant(float("-inf"), type=T.f32))
    zero = arith.unwrap(arith.constant(0.0, type=T.f32))
    row_max = {}
    row_sum = {}
    for bank in range(4):
        row_max[bank] = set_vgpr_bank(neg_inf, bank)
        row_sum[bank] = set_vgpr_bank(zero, bank)
    return row_max, row_sum


def _phase8_zero_o_accum_flydsl():
    """Zero O output accumulators across 4 banks, 4 tiles each."""
    zero_vec = arith.unwrap(arith.constant_vector(0.0, T.vec(8, T.f32)))
    o_accs = {}
    rocdl.sched_barrier(0)
    for bank in range(4):
        for tile in range(4):
            o_accs[(bank, tile)] = set_vgpr_bank(zero_vec, bank)
    rocdl.sched_barrier(0)
    return o_accs


def _phase_v_tdm_blk1_flydsl(
    arg_V,
    v_offset,
    v_lds_base,
    stride_v_seq,
    stride_v_32,
    wave_id,
):
    """V TDM block 1 (2 copies)."""
    i64 = ir.IntegerType.get_signless(64)
    v_dgroup1 = _build_tdm_dgroup1(_V_TDM_CONFIG, stride_v_seq)
    pred = arith.constant(1, type=T.i32)

    v_blk_inc = arith.muli(arith.constant(K_TILE_N, type=T.i32), stride_v_seq)
    v_offset_blk1 = arith.addi(v_offset, v_blk_inc)
    v_addr_i64 = _compute_v_global_addr(arg_V, v_offset_blk1, wave_id, stride_v_32)
    stride_adv_i64 = arith.extsi(i64, arith.unwrap(stride_v_32))

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
            cur_addr = arith.addi(cur_addr, stride_adv_i64)

    _s_wait_tensorcnt(4)
    rocdl.s_barrier_signal(-1)
    rocdl.s_barrier_wait(-1)


# ============================================================================
# Main Kernel
