from __future__ import annotations

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm as llvm_dialect
from flydsl._mlir.dialects import rocdl as rocdl_dialect
from flydsl._mlir.dialects import scf
from flydsl.expr import arith, rocdl
from aiter.ops.flydsl.kernels import vector
from flydsl.expr.primitive import const_expr, range_constexpr
from flydsl.expr.rocdl import tdm_ops
from flydsl.expr.typing import T, Float32, Vector as Vec
from flydsl.utils.smem_allocator import SmemAllocator

from flydsl._mlir.dialects import fly as fly_d
from ..layout_utils import idx2crd as idx2crd

def glb_ptr_ty():
    return ir.Type.parse("!llvm.ptr<1>")
def lds_ptr_ty():
    return ir.Type.parse("!llvm.ptr<3>")

# Schedule token IDs
P1 = 4              # PART1 cross-MSB merge
O_RESC0 = 17        # O-rescale pk_mul
TDM_TOKEN = 18      # tensor_load_to_lds
PART2_EXP_START = 24 # ops[0..23]=setup+pkfma, ops[24..]=exp
# Token base offsets: token_id = BASE + msb
P2_BASE = 5         # softmax PART2
EXP_BASE = 19       # pair_exp (3-cycle transcendental)
K_BASE = 9          # ds_load_b128 K
V_BASE = 13         # ds_load_tr16_b128 V

# Schedule row helpers: n copies of token for MSB m
def lds_k(m, n=1):    return [K_BASE + m] * n
def lds_v(m, n=1):    return [V_BASE + m] * n
def tree_max(m, n=1): return [m] * n
def cross_max(n=1):   return [P1] * n
def sm_ops(m, n=1):   return [P2_BASE + m] * n
def pair_exp(m, n=1): return [EXP_BASE + m] * n
def o_rescale(n=1):   return [O_RESC0] * n
def tdm_load(n=1):    return [TDM_TOKEN] * n

# GEMM1 (QK) schedule: 96 rows = 4 stages x 24 WMMAs.
# Each row = ops dispatched between consecutive WMMA instructions.
#   lds_k/lds_v(msb, n) = n LDS loads for MSB bank
#   sm_ops(msb, n)       = n softmax PART2 ops for MSB
#   tree_max(msb, n)     = n PART0 max-reduction ops
#   cross_max(n)         = n PART1 cross-MSB merge ops
#   pair_exp(msb, n)     = n exp2 ops (3-cycle transcendental)
#   o_rescale(n)         = n O *= exp_delta rescale ops
#   tdm_load(n)          = n TDM prefetch ops
GEMM1_SCHEDULE: list[list[int]] = [
    # Stage 0: TDM K prefetch + K LDS loads + softmax
    tdm_load(2) + sm_ops(0),                        # wmma 0
    lds_k(0, 3) + o_rescale(),                      # wmma 1
    lds_k(0, 3) + sm_ops(0, 2),                     # wmma 2
    lds_k(1, 3) + sm_ops(1, 2),                     # wmma 3
    lds_k(1, 3) + sm_ops(1, 2),                     # wmma 4
    lds_k(2, 3) + sm_ops(2, 2),                     # wmma 5 (ds_wait)
    lds_k(2, 3) + sm_ops(2) + sm_ops(0),            # wmma 6
    lds_k(3, 3) + sm_ops(3, 2),                     # wmma 7
    lds_k(3, 3) + sm_ops(3),                         # wmma 8
    sm_ops(0, 2), sm_ops(0, 2),                      # wmma 9-10
    sm_ops(1, 2), sm_ops(1, 2),                      # wmma 11-12
    sm_ops(1) + o_rescale(), sm_ops(1) + sm_ops(0),  # wmma 13-14
    sm_ops(2) + sm_ops(3),                           # wmma 15
    sm_ops(2) + o_rescale(), sm_ops(2) + o_rescale(), # wmma 16-17
    sm_ops(0, 2), sm_ops(0, 2),                      # wmma 18-19
    sm_ops(0, 2), sm_ops(0, 2),                      # wmma 20-21
    sm_ops(0, 2), sm_ops(0) + sm_ops(2),             # wmma 22-23
    # Stage 1
    tdm_load(2) + sm_ops(2),
    lds_k(0, 3) + o_rescale(),
    lds_k(0, 3) + sm_ops(0),
    lds_k(1, 3) + sm_ops(1, 2),
    lds_k(1, 3) + sm_ops(1, 2),
    lds_k(2, 3) + sm_ops(2, 2),
    lds_k(2, 3) + sm_ops(2, 2),
    lds_k(3, 3) + sm_ops(3),
    lds_k(3, 3) + sm_ops(3),
    sm_ops(3, 2), sm_ops(3, 2),
    sm_ops(3, 2), sm_ops(1, 2),
    sm_ops(1, 2), sm_ops(1) + sm_ops(2),
    sm_ops(1) + o_rescale(),
    sm_ops(2) + o_rescale(), sm_ops(2) + o_rescale(),
    sm_ops(2, 2), sm_ops(2, 2),
    sm_ops(3, 2), sm_ops(3, 2),
    sm_ops(3, 2), sm_ops(3) + sm_ops(0),
    # Stage 2: K LDS loads + heavy softmax
    lds_k(0, 3) + sm_ops(0) + sm_ops(3),
    lds_k(0, 3) + sm_ops(0) + sm_ops(2),
    lds_k(1, 3) + sm_ops(1, 2),
    lds_k(1, 3) + sm_ops(1, 2),
    lds_k(2, 3) + sm_ops(2, 2),
    lds_k(2, 3) + sm_ops(2, 2),
    lds_k(3, 3) + sm_ops(3, 2),
    lds_k(3, 3) + sm_ops(3, 2),
    sm_ops(0, 6), sm_ops(0, 7), sm_ops(0, 7),
    sm_ops(1, 7), sm_ops(1, 7), sm_ops(1, 7),
    sm_ops(2, 2) + o_rescale(), sm_ops(2, 2) + o_rescale(),
    sm_ops(2, 2) + o_rescale(), sm_ops(3, 2) + o_rescale(),
    sm_ops(2, 6), sm_ops(2, 6),
    sm_ops(2, 3) + sm_ops(3, 3),
    sm_ops(3, 6), sm_ops(3, 6),
    sm_ops(0, 4) + sm_ops(3, 2),
    # Stage 3: V LDS loads + remaining softmax
    lds_v(0, 2) + sm_ops(0, 2),
    lds_v(0, 2) + sm_ops(0, 2),
    lds_v(1, 2) + sm_ops(1, 2),
    lds_v(1, 2) + sm_ops(1) + sm_ops(1) + sm_ops(2),
    lds_v(2, 2) + sm_ops(2, 2),
    lds_v(2, 2) + sm_ops(2, 2),
    lds_v(3, 2) + sm_ops(3, 2),
    lds_v(3, 2) + sm_ops(3) + sm_ops(1),
    sm_ops(1, 2) + sm_ops(3), sm_ops(1) + sm_ops(0),
    sm_ops(1) + sm_ops(3), sm_ops(1) + sm_ops(0) + sm_ops(3),
    sm_ops(2) + sm_ops(3), sm_ops(2, 2),
    sm_ops(3, 2), sm_ops(3, 2) + sm_ops(1),
    o_rescale() + sm_ops(2), o_rescale() + sm_ops(2),
    o_rescale() + sm_ops(2) + sm_ops(0),
    o_rescale() + sm_ops(2) + sm_ops(3),
    sm_ops(0) + sm_ops(3) + sm_ops(1),
    sm_ops(0), sm_ops(3), [],
]
# GEMM2 (PV) schedule: 64 rows = 4 stages x 16 WMMAs
GEMM2_SCHEDULE: list[list[int]] = [
    # Stage 0: TDM + V LDS loads + tree_max (PART0)
    tdm_load(2) + tree_max(0, 3),
    lds_v(0, 2) + tree_max(0, 3),
    lds_v(0, 2) + tree_max(0, 3),
    lds_v(1, 2) + tree_max(1, 4),
    lds_v(1, 2) + tree_max(1, 4),
    lds_v(2, 2) + tree_max(2, 4),
    lds_v(2, 2) + tree_max(2, 4),
    lds_v(3, 2) + tree_max(3, 4),
    lds_v(3, 2) + tree_max(3, 4),
    tree_max(0) + tree_max(2, 3) + tree_max(3),
    tree_max(0) + tree_max(1, 2) + tree_max(3),
    tree_max(0, 4) + tree_max(1),
    tree_max(0) + tree_max(1, 2) + tree_max(2, 2),
    tree_max(1) + tree_max(2, 2) + tree_max(3, 2),
    tree_max(2) + tree_max(3, 2) + tree_max(1, 2),
    tree_max(3, 2),
    # Stage 1: V LDS loads + cross_max (PART1) + softmax PART2
    tdm_load(2) + tree_max(0),
    lds_v(0, 2) + [0, 1, 2, 3],
    lds_v(0, 2) + [0, 1, 2, 3],
    lds_v(1, 2) + [1, 0, 2, 3],
    lds_v(1, 2) + [1, 0],
    lds_v(2, 2) + [2, 2, 1],
    lds_v(2, 2) + [2, 0, 1],
    lds_v(3, 2) + [3],
    lds_v(3, 2) + [3, 3],
    cross_max(4), cross_max(4),
    sm_ops(1, 4) + sm_ops(2, 4),
    sm_ops(0, 4) + sm_ops(3, 4),
    sm_ops(0) + sm_ops(1) + sm_ops(2) + sm_ops(3),
    (sm_ops(0) + sm_ops(1) + sm_ops(2) + sm_ops(3)) * 2,
    sm_ops(0) + sm_ops(1) + sm_ops(2) + sm_ops(3),
    # Stage 2: V LDS loads + softmax PART2
    lds_v(0, 2) + sm_ops(0, 3),
    lds_v(0, 2) + sm_ops(0, 3),
    lds_v(1, 2) + sm_ops(1) + sm_ops(2, 2),
    lds_v(1, 2) + sm_ops(1) + sm_ops(2, 2),
    lds_v(2, 2) + sm_ops(2) + sm_ops(1, 2),
    lds_v(2, 2) + sm_ops(2) + sm_ops(1, 2),
    lds_v(3, 2) + sm_ops(3, 4),
    lds_v(3, 2) + sm_ops(3, 3),
    sm_ops(0, 6), sm_ops(0, 3) + sm_ops(1),
    sm_ops(1, 6), sm_ops(1, 3) + sm_ops(2, 2),
    sm_ops(2, 6), sm_ops(3, 3),
    sm_ops(3, 4), sm_ops(0, 2) + sm_ops(1, 2),
    # Stage 3: K LDS loads + pair_exp
    lds_k(0, 3) + pair_exp(0, 2),
    lds_k(0, 3) + pair_exp(0),
    lds_k(1, 3) + pair_exp(1, 2),
    lds_k(1, 3) + pair_exp(1, 2),
    lds_k(2, 3) + sm_ops(2) + pair_exp(2),
    lds_k(2, 3) + sm_ops(2) + pair_exp(2),
    lds_k(3, 3) + sm_ops(3) + pair_exp(3),
    lds_k(3, 3) + sm_ops(3) + pair_exp(3),
    pair_exp(0, 3), pair_exp(0, 2) + pair_exp(1),
    pair_exp(1, 3), pair_exp(3, 3),
    pair_exp(2, 3), pair_exp(2, 3),
    pair_exp(3, 3), [],
]

def g1_row_idx(stage: int, wmma: int) -> int:
    return stage * GEMM_INST_COUNT + wmma
def g2_row_idx(stage: int, wmma: int) -> int:
    return stage * PV_GEMM_INST_COUNT + wmma
def rocdl_exp2(res, arg, **kw):
    return rocdl_dialect.exp2(res=res, arg=arg, **kw)
def rocdl_permlanex16(res, old, src0, src1, src2, fi, bound_control, **kw):
    return rocdl_dialect.permlanex16(
        res=res,
        old=old,
        src0=src0,
        src1=src1,
        src2=src2,
        fi=fi,
        bound_control=bound_control,
        **kw,
    )
def rocdl_fmax3(a, b, c):
    m = llvm_dialect.intr_maxnum(a, b)
    return llvm_dialect.intr_maxnum(m, c)
WAVE_SIZE = 32
NUM_WAVES = 4
NUM_MSB = 4
BLOCK_SIZE = WAVE_SIZE * NUM_WAVES

QK_HDIM = 192
V_HDIM = 128
Q_BPP = 2
KV_BPP = 2
TG_SUBQD = 128
TG_SUBKV = 128
WV_SUBKV = TG_SUBKV
SU_K_N = 32
SU_K_K = QK_HDIM
CNT_SU = WV_SUBKV // SU_K_N

WMMA_M, WMMA_N, WMMA_K = 16, 16, 32

VPS_Q = TG_SUBQD * QK_HDIM * Q_BPP // 128
VPS_MSB_Q = VPS_Q // NUM_MSB
VTS_MSB_Q = VPS_MSB_Q
Q_WMMA_PER_MSB = VTS_MSB_Q // 8
VPS_KV = SU_K_K * SU_K_N * KV_BPP // 128
VPS_MSB_KV = VPS_KV // NUM_MSB
VPS_MSB_SP = 32
SP_MSB_M = 16
SP_MSB_N = 16
SP_MSB_K = QK_HDIM

GEMM_INST_COUNT = 24
LDS_INST_COUNT = 24
N_LDS_PER_MSB = VPS_MSB_KV // 4
N_LDS_V_PER_MSB = 4
LDS_V_INST_COUNT = NUM_MSB * N_LDS_V_PER_MSB

ALU_STAGES = 8
GEMM1 = 0
KV_K = 0
KV_V = 1
KV_NONE = 2
BARRIER_SIGNAL_AHEAD = 0
TDM_LOADS_PER_STAGE = 2

LDS_K_SU_P_SIZE = 0x3200
LDS_V_SU_P_SIZE = 0x2400

EXP_PER_MSB_TO_G2 = 8
PART2_SETUP_A = 8
PART2_SPLIT = 24 + EXP_PER_MSB_TO_G2
PART2_G2_SPLIT = PART2_SPLIT
GEMM1_EXP_OPS = VPS_MSB_SP - EXP_PER_MSB_TO_G2

ALU_PER_STAGE = [40, 52, 56, 168, 120, 120, 132, 132]

N_SP_PAIRS = VPS_MSB_SP // 2

GEMM2 = 1
N_V_MSB = 2
N_PV_WMMA_N = 4
D_MSB_K = SU_K_N
PV_K_ITERS = 1
PV_GEMM_INST_COUNT = NUM_MSB * PV_K_ITERS * N_PV_WMMA_N

def emit_void(inst_str, operands=None, constraints="", **kwargs):
    llvm_dialect.inline_asm(
        None, operands or [], inst_str, constraints, has_side_effects=True, **kwargs
    )
def sched_barrier(mask=0):
    mask_val = arith.constant(mask, type=T.i32)
    llvm_dialect.call_intrinsic(None, "llvm.amdgcn.sched.barrier", [mask_val], [], [])
USE_BANK_HINTS = False

# sp_pairs[i] (v2f32) lands at HWIdx = bank*256 + SP_PAIR_BASE + i*2.
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
    if const_expr(not USE_BANK_HINTS):
        return raw_val
    val_type = raw_val.type
    bank_val = arith.constant(bank, type=T.i32)
    return llvm_dialect.call_intrinsic(
        val_type, "llvm.amdgcn.set.vgpr.bank", [raw_val, bank_val], [], []
    )
def set_vgpr_bank_offset(raw_val, bank: int, offset: int):
    """Pin raw_val to HWIdx = bank*256+offset (single-candidate BankOffsetHint)."""
    if const_expr(not USE_BANK_HINTS):
        return raw_val
    val_type = raw_val.type
    bank_val = arith.constant(bank, type=T.i32)
    offset_val = arith.constant(offset, type=T.i32)
    return llvm_dialect.call_intrinsic(
        val_type,
        "llvm.amdgcn.set.vgpr.bank.offset",
        [raw_val, bank_val, offset_val],
        [],
        [],
    )
def get_types():
    return {
        "f32": T.f32,
        "bf16": T.bf16,
        "i32": T.i32,
        "v2f32": T.vec(2, T.f32),
        "v2bf16": T.vec(2, T.bf16),
        "v8f32": T.vec(8, T.f32),
        "v8bf16": T.vec(8, T.bf16),
        "v16bf16": T.vec(16, T.bf16),
        "v4i32": T.vec(4, T.i32),
        "v8i32": T.vec(8, T.i32),
        "lds_ptr": lds_ptr_ty(),
    }
def make_v2f32(lo, hi, bank=0):
    v = Vec.from_elements([lo, hi], Float32)
    return set_vgpr_bank(v, bank)
def split_v2f32(pair):
    lo = Vec(pair, dtype=Float32)[0].ir_value()
    hi = Vec(pair, dtype=Float32)[1].ir_value()
    return lo, hi
def broadcast_f32_to_v2f32(val, bank=0):
    return make_v2f32(val, val, bank)
N_WMMA_K_TILES = (QK_HDIM // WMMA_K) // 2

def _banked_op(result, bank):
    sched_barrier(0)
    b = set_vgpr_bank(result, bank)
    sched_barrier(0)
    return b
def _wmma_bf16(ty, src_a, src_b, acc, bank_dst):
    sched_barrier(0)
    r = rocdl_dialect.wmma_f32_16x16x32_bf16(
        ty["v8f32"], src_a, src_b, acc,
        signA=False, signB=False, modC=0, reuseA=False, reuseB=False)
    return _banked_op(r.result, bank_dst)
class Atom:
    @staticmethod
    def wmma_init(ty, src_a, src_b, bank_dst):
        return _wmma_bf16(ty, src_a, src_b, fx.constant_vector(0.0, T.vec(8, T.f32)), bank_dst)
    @staticmethod
    def wmma_accum(ty, src_a, src_b, acc, bank_dst):
        return _wmma_bf16(ty, src_a, src_b, acc, bank_dst)
    @staticmethod
    def ds_load_b128(ty, addr, offset_val, bank):
        sched_barrier(0)
        ptr = llvm_dialect.inttoptr(ty["lds_ptr"], (addr + offset_val))
        return _banked_op(llvm_dialect.load(ty["v4i32"], ptr), bank)
    @staticmethod
    def ds_load_tr16_b128(ty, addr, offset_val, bank):
        sched_barrier(0)
        ptr = llvm_dialect.inttoptr(ty["lds_ptr"], (addr + offset_val))
        return _banked_op(rocdl.ds_load_tr16_b128(ty["v8bf16"], ptr), bank)
    @staticmethod
    def tdm_load(ty, s_g0, s_g1):
        sched_barrier(0)
        null_v4 = fx.constant_vector(0, T.vec(4, T.i32))
        rocdl.tensor_load_to_lds(s_g0, s_g1, null_v4, null_v4, fx.constant_vector(0, T.vec(8, T.i32)), 0)
        sched_barrier(0)
    @staticmethod
    def exp_f32(src, bank):
        sched_barrier(0)
        return _banked_op(rocdl_exp2(T.f32, src), bank)
    @staticmethod
    def mul_f32(src0, src1, bank):
        sched_barrier(0)
        return _banked_op(src0 * src1, bank)
    @staticmethod
    def fma_f32_neg_src0(src0, src1, src2, bank):
        sched_barrier(0)
        return _banked_op(llvm_dialect.intr_fma(llvm_dialect.fneg(src0), src1, src2), bank)
    @staticmethod
    def mov_b32(src, bank):
        sched_barrier(0)
        return _banked_op(src, bank)
    @staticmethod
    def add_f32(src0, src1, bank):
        sched_barrier(0)
        return _banked_op(src0 + src1, bank)
    @staticmethod
    def max3_num_f32(src0, src1, src2, bank):
        sched_barrier(0)
        return _banked_op(rocdl_fmax3(src0, src1, src2), bank)
    @staticmethod
    def permlanex16(src, s_sel0, s_sel1, bank):
        sched_barrier(0)
        src_i32 = llvm_dialect.bitcast(T.i32, src)
        r_f32 = llvm_dialect.bitcast(T.f32, rocdl_permlanex16(T.i32, src_i32, src_i32, s_sel0, s_sel1, False, False))
        return _banked_op(r_f32, (bank + 2) % NUM_MSB)
    @staticmethod
    def pk_fma_f32_neg_c(a, b, c, bank):
        sched_barrier(0)
        return _banked_op(llvm_dialect.intr_fma(a, b, llvm_dialect.fneg(c)), bank)
    @staticmethod
    def pk_add_f32(a, b, bank):
        sched_barrier(0)
        return _banked_op(a + b, bank)
    @staticmethod
    def cvt_pk_bf16_f32(a, bank):
        sched_barrier(0)
        return _banked_op(arith.truncf(T.vec(2, T.bf16), a), bank)
    @staticmethod
    def s_wait_dscnt(cnt):
        sched_barrier(0); rocdl.s_wait_dscnt(cnt); sched_barrier(0)
    @staticmethod
    def s_wait_tensorcnt(cnt):
        sched_barrier(0); rocdl.s_wait_tensorcnt(cnt); sched_barrier(0)
def tdm_wait_and_barrier():
    rocdl.s_wait_tensorcnt(0)
    rocdl.s_barrier_signal(-1)
    rocdl.s_barrier_wait(-1)
def _none_list():
    return [None] * NUM_MSB
def make_softmax_state(old_max, local_max, delta, row_sums, sp_pairs_prev=None):
    return {"old_max": list(old_max), "local_max": list(local_max), "delta": list(delta),
            "exp_delta": _none_list(), "cur_max_log2e": _none_list(),
            "cur_max_log2e_1": _none_list(), "cur_max_log2e_scalar": _none_list(),
            "cur_max_log2e_dup": _none_list(), "vgpr_log2e_scl_pair": _none_list(),
            "exp_delta_dup": _none_list(), "row_sums": list(row_sums),
            "p_bf16": [[] for _ in range(NUM_MSB)], "sp_pairs_prev": sp_pairs_prev}
class Softmax:
    @staticmethod
    def build_part2_ops(
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
        ops = []
        bank = msb

        # CSE-safety: each bank (b) passes a different argument to set_vgpr_bank_offset /
        # bank-specific physical register (bank×256 + LOG2E_PAIR_OFFSET for banks 1-3).
        def op_save_old_max(b=bank):
            ss["old_max"][b] = Atom.mov_b32(ss["local_max"][b], b)
            sched_barrier(0)
            _scl = sgpr["s_log2e_scl"]
            v = Vec.from_elements([_scl, _scl], Float32)
            ss["vgpr_log2e_scl_pair"][b] = set_vgpr_bank_offset(v, b, LOG2E_PAIR_OFFSET) if const_expr(b > 0) else set_vgpr_bank(v, b)
            sched_barrier(0)
        def op_cur_max(b=bank):
            ss["cur_max_log2e"][b] = Atom.mul_f32(ss["local_max"][b], sgpr["s_log2e_scl"], b)
        def op_exp_delta(b=bank):
            ss["exp_delta"][b] = Atom.exp_f32(ss["delta"][b], b)
        def op_cur_max_1(b=bank):
            ss["cur_max_log2e_1"][b] = Atom.mul_f32(ss["local_max"][b], sgpr["s_log2e_scl"], b)
        def op_mul_old_max(b=bank):
            ss["cur_max_log2e_scalar"][b] = Atom.mul_f32(ss["old_max"][b], sgpr["s_log2e_scl"], b)
        def op_broadcast_dup(b=bank):
            ss["cur_max_log2e_dup"][b] = broadcast_f32_to_v2f32(ss["cur_max_log2e_scalar"][b], b)
        def op_exp_delta_dup(b=bank):
            ss["exp_delta_dup"][b] = Atom.mov_b32(ss["exp_delta"][b], b)
        ops += [op_save_old_max, op_cur_max, op_exp_delta, op_cur_max_1,
                op_mul_old_max, op_broadcast_dup, op_exp_delta_dup]
        if not skip_rescale_sum:
            def op_rescale_sum(b=bank):
                ss["row_sums"][b] = Atom.mul_f32(ss["exp_delta"][b], ss["row_sums"][b], b)
            ops.append(op_rescale_sum)
        for i in range_constexpr(N_SP_PAIRS):
            _sp_offset = SP_PAIR_BASE + i * 2
            _escaped = i < 2
            def op_pkfma(idx=i, b=bank, sp_off=_sp_offset, escaped=_escaped):
                src = sp_pairs[idx]
                if const_expr(b > 0 and escaped):
                    src = set_vgpr_bank(src, b)
                result = Atom.pk_fma_f32_neg_c(src, ss["vgpr_log2e_scl_pair"][b], ss["cur_max_log2e_dup"][b], b)
                sp_pairs[idx] = set_vgpr_bank_offset(result, b, sp_off) if const_expr(b > 0) else result
            ops.append(op_pkfma)
        sum_tmps = [None] * (N_SP_PAIRS // 2)
        for _eidx in range_constexpr(VPS_MSB_SP):
            _pidx, _is_hi = _eidx // 2, _eidx % 2
            if const_expr(_is_hi == 0):
                def op_exp_lo(pidx=_pidx, b=bank, _clo=sp_lo_cache):
                    lo, hi = split_v2f32(sp_pairs[pidx])
                    sched_barrier(0); exp_lo = rocdl_exp2(T.f32, lo); sched_barrier(0)
                    sp_pairs[pidx] = Vec.from_elements([exp_lo, hi], Float32)
                    if const_expr(_clo is not None): _clo[pidx] = exp_lo
                ops.append(op_exp_lo)
            else:
                def op_exp_hi(pidx=_pidx, b=bank, _chi=sp_hi_cache):
                    lo, hi = split_v2f32(sp_pairs[pidx])
                    sched_barrier(0); exp_hi = rocdl_exp2(T.f32, hi); sched_barrier(0)
                    sp_pairs[pidx] = Vec.from_elements([lo, exp_hi], Float32)
                    if const_expr(_chi is not None): _chi[pidx] = exp_hi
                ops.append(op_exp_hi)
        sum_l0 = [None] * (N_SP_PAIRS // 4)
        sum_l1 = [None] * (N_SP_PAIRS // 8)
        sum_l2 = [None]
        final_sum = [None]
        for i in range_constexpr(N_SP_PAIRS):
            def op_cvt(cidx=i, b=bank):
                ss["p_bf16"][b].append(Atom.cvt_pk_bf16_f32(set_vgpr_bank(sp_pairs[cidx], b), b))
            ops.append(op_cvt)
        for i in range_constexpr(N_SP_PAIRS // 2):
            def op_pkadd(idx=i, b=bank):
                sum_tmps[idx] = Atom.pk_add_f32(sp_pairs[idx * 2], sp_pairs[idx * 2 + 1], b)
            ops.append(op_pkadd)
        for j in range_constexpr(N_SP_PAIRS // 4):
            def op_sum_l0(j_val=j, b=bank):
                sum_l0[j_val] = Atom.pk_add_f32(sum_tmps[j_val * 2], sum_tmps[j_val * 2 + 1], b)
            ops.append(op_sum_l0)
        for j in range_constexpr(2):
            def op_sum_l1(j_val=j, b=bank):
                sum_l1[j_val] = Atom.pk_add_f32(sum_l0[j_val * 2], sum_l0[j_val * 2 + 1], b)
            ops.append(op_sum_l1)
        def op_sum_l2(b=bank):
            sum_l2[0] = Atom.pk_add_f32(sum_l1[0], sum_l1[1], b)
        def op_sum_split(b=bank):
            lo, hi = split_v2f32(sum_l2[0])
            final_sum[0] = Atom.add_f32(lo, hi, b)
        def op_sum_accum(b=bank):
            ss["row_sums"][b] = Atom.add_f32(ss["row_sums"][b], final_sum[0], b)
        ops += [op_sum_l2, op_sum_split, op_sum_accum]
        return ops
    @staticmethod
    def build_all_part2_ops(ty, blk, sp_pairs_all, softmax_state, sgpr_state):
        ops_by_msb = [[] for _ in range_constexpr(NUM_MSB)]
        for m in range_constexpr(NUM_MSB):
            sp_pairs = sp_pairs_all[m]
            msb_ops = Softmax.build_part2_ops(
                ty, m, blk, sp_pairs, softmax_state, sgpr_state
            )
            ops_by_msb[m] = msb_ops
        return ops_by_msb
    @staticmethod
    def build_part0_ops(ty, msb, sp_pairs, ss, sgpr):
        ops = []
        bank = msb
        sp_f32 = [None] * VPS_MSB_SP
        tmps = [None] * N_VALID_GROUPS
        def _get_sp(offset):
            if const_expr(sp_f32[offset] is None):
                sp_f32[offset] = Vec(sp_pairs[offset // 2], dtype=Float32)[offset % 2].ir_value()
            return sp_f32[offset]
        for k in range_constexpr(N_VALID_GROUPS):
            def op_init_max3(k_=k, b=bank):
                base = k_ * VALID_GROUP_STRIDE
                tmps[k_] = Atom.max3_num_f32(_get_sp(base), _get_sp(base + 1), _get_sp(base + 2), b)
            ops.append(op_init_max3)
        for j in range_constexpr(2):
            for k in range_constexpr(N_VALID_GROUPS):
                def op_cross_col(k_=k, j_=j, b=bank):
                    base = k_ * VALID_GROUP_STRIDE
                    s0 = base + 3 + j_ * 2
                    tmps[k_] = Atom.max3_num_f32(_get_sp(s0), _get_sp(s0 + 1), tmps[k_], b)
                ops.append(op_cross_col)
        for k in range_constexpr(N_VALID_GROUPS):
            def op_last_elem(k_=k, b=bank):
                base = k_ * VALID_GROUP_STRIDE
                tmps[k_] = Atom.max3_num_f32(_get_sp(base + 7), tmps[k_], _get_sp(base), b)
            ops.append(op_last_elem)
        def op_merge1(b=bank):
            tmps[0] = Atom.max3_num_f32(tmps[0], tmps[1], tmps[2], b)
        def op_merge2(b=bank):
            tmps[0] = Atom.max3_num_f32(tmps[0], tmps[3], tmps[1], b)
        tmps_perm = [None]
        _zero_f32 = arith.constant(0.0, type=T.f32)
        def op_perm_prep(b=bank, z=_zero_f32):
            tmps_perm[0] = Atom.add_f32(tmps[0], z, (b + 2) % NUM_MSB)
        def op_perm(b=bank):
            tmps[1] = Atom.permlanex16(tmps_perm[0],
                arith.constant(0x76543210, type=T.i32),
                arith.constant(0xFEDCBA98, type=T.i32), (b + 2) % NUM_MSB)
        def op_pre_max(b=bank):
            ss["pre_max_log2e_scl"][b] = Atom.mul_f32(ss["old_max"][b], sgpr["s_log2e_scl"], b)
        def op_cur_max(b=bank):
            ss["local_max"][b] = Atom.max3_num_f32(tmps[0], tmps[1], ss["old_max"][b], b)
        ops += [op_merge1, op_merge2, op_perm_prep, op_perm, op_pre_max, op_cur_max]
        assert len(ops) == PART0_INSTS
        return ops
    @staticmethod
    def build_part1_ops(ty, ss, sgpr):
        ops = []
        def op_max01():
            ss["local_max"][0] = Atom.max3_num_f32(ss["local_max"][0], ss["local_max"][1], ss["pre_max_log2e_scl"][0], 0)
        def op_max23():
            ss["local_max"][2] = Atom.max3_num_f32(ss["local_max"][2], ss["local_max"][3], ss["pre_max_log2e_scl"][2], 2)
        def op_mov1():
            ss["local_max"][1] = Atom.mov_b32(ss["local_max"][0], 1)
        def op_mov3():
            ss["local_max"][3] = Atom.mov_b32(ss["local_max"][2], 3)
        ops += [op_max01, op_max23, op_mov1, op_mov3]
        msb_assign = [0, 2, 1, 3]
        for msb in [0, 2, 1, 3]:
            def op_fma_delta(b=msb):
                ss["delta"][b] = Atom.fma_f32_neg_src0(ss["local_max"][b], sgpr["s_log2e_scl"], ss["pre_max_log2e_scl"][b], b)
            ops.append(op_fma_delta)
            msb_assign.append(msb)
        assert len(ops) == PART1_INSTS
        return ops, msb_assign
    @staticmethod
    def build_all_gemm2_ops(ty, blk, sp_pairs_all, softmax_state, sgpr_state, skip_rescale_sum=False):
        if const_expr("pre_max_log2e_scl" not in softmax_state):
            softmax_state["pre_max_log2e_scl"] = _none_list()
        ops_by_rid = [[] for _ in range_constexpr(RLTS_LEN)]
        for m in range_constexpr(NUM_MSB):
            ops_by_rid[m] = Softmax.build_part0_ops(ty, m, sp_pairs_all[m], softmax_state, sgpr_state)
        ops_by_rid[4] = Softmax.build_part1_ops(ty, softmax_state, sgpr_state)[0]
        sp_lo_cache = [[None] * N_SP_PAIRS for _ in range_constexpr(NUM_MSB)]
        sp_hi_cache = [[None] * N_SP_PAIRS for _ in range_constexpr(NUM_MSB)]
        for m in range_constexpr(NUM_MSB):
            p2_ops = Softmax.build_part2_ops(ty, m, blk, sp_pairs_all[m], softmax_state,
                sgpr_state, skip_rescale_sum=skip_rescale_sum,
                sp_lo_cache=sp_lo_cache[m], sp_hi_cache=sp_hi_cache[m])
            ops_by_rid[5 + m] = p2_ops[:PART2_G2_SPLIT]
        rid_budget = [[0] * RLTS_LEN for _ in range_constexpr(4)]
        p0c = min(PART0_INSTS, ALU_PER_STAGE[0] // NUM_MSB)
        for m in range_constexpr(NUM_MSB): rid_budget[0][m] = p0c
        p0c2 = min(PART0_INSTS - p0c, ALU_PER_STAGE[1] // NUM_MSB)
        for m in range_constexpr(NUM_MSB): rid_budget[1][m] = p0c2
        rid_budget[1][4] = PART1_INSTS
        for m in range_constexpr(NUM_MSB): rid_budget[1][5 + m] = 4
        rid_budget[2][4] = min(PART1_INSTS, ALU_PER_STAGE[2])
        for m in range_constexpr(NUM_MSB): rid_budget[2][5 + m] = ALU_PER_STAGE[2] // NUM_MSB
        for m in range_constexpr(NUM_MSB): rid_budget[3][5 + m] = ALU_PER_STAGE[3] // NUM_MSB
        return ops_by_rid, rid_budget, sp_lo_cache, sp_hi_cache
    @staticmethod
    def tiles_to_pairs(su_sp_tiles_list):
        sp_pairs = []
        for msb in range_constexpr(NUM_MSB):
            pairs = [None] * N_SP_PAIRS
            for su in range_constexpr(CNT_SU):
                v8w = Vec(su_sp_tiles_list[su][msb][0], dtype=Float32)
                for i in range_constexpr(4):
                    pair_idx = su * 4 + i
                    v2 = make_v2f32(v8w[i * 2].ir_value(), v8w[i * 2 + 1].ir_value(), bank=msb)
                    pairs[pair_idx] = set_vgpr_bank_offset(v2, msb, SP_PAIR_BASE + pair_idx * 2) if const_expr(msb > 0) else v2
            sp_pairs.append(pairs)
        return sp_pairs
    @staticmethod
    def part01_only(ty, blk, sp_pairs_all, softmax_state, sgpr_state):
        if const_expr("pre_max_log2e_scl" not in softmax_state):
            softmax_state["pre_max_log2e_scl"] = _none_list()
        ops_by_rid, _, _, _ = Softmax.build_all_gemm2_ops(ty, blk, sp_pairs_all, softmax_state, sgpr_state)
        for _b in range_constexpr(4):
            for rid in range_constexpr(NUM_MSB):
                for _j in range_constexpr(4):
                    ops_by_rid[rid][_b * 4 + _j]()
        for step in range_constexpr(NUM_MSB):
            for rid in range_constexpr(NUM_MSB):
                ops_by_rid[rid][16 + step]()
            sched_barrier(0)
            ops_by_rid[step][20]()
            sched_barrier(0)
        for rid in range_constexpr(NUM_MSB):
            ops_by_rid[rid][21]()
        for op in ops_by_rid[4]:
            op()
    @staticmethod
    def build_p_tiles(ty, softmax_state):
        p_bf16_all = softmax_state["p_bf16"]
        p_tiles = []
        for su in range_constexpr(CNT_SU):
            ps = su * 4
            p_tiles.append([Fragment.pack_v2bf16(ty, p_bf16_all[2 * mt][ps:ps + 4] + p_bf16_all[2 * mt + 1][ps:ps + 4], mt) for mt in range_constexpr(2)])
        return p_tiles
PART0_INSTS = 22
PART1_INSTS = 8
RLTS_LEN = 9

N_VALID_GROUPS = CNT_SU
VALID_GROUP_STRIDE = 8

class Fragment:
    @staticmethod
    def wmma_bf16(vec4_lo, vec4_hi):
        vec8_bf16_ty = T.vec(8, T.bf16)
        v0 = vector.bitcast(vec8_bf16_ty, vec4_lo)
        v1 = vector.bitcast(vec8_bf16_ty, vec4_hi)
        return vector.shuffle(v0, v1, list(range_constexpr(16)))
    @staticmethod
    def pair_k_tiles(kv_tiles_raw, ty):
        kv_paired = []
        for msb in range_constexpr(NUM_MSB):
            msb_frags = []
            for k in range_constexpr(N_WMMA_K_TILES):
                lo = kv_tiles_raw[msb][k * 2]
                hi = kv_tiles_raw[msb][k * 2 + 1]
                frag = Fragment.wmma_bf16(lo, hi)
                frag = set_vgpr_bank(frag, msb)
                msb_frags.append(frag)
            kv_paired.append(msb_frags)
        return kv_paired
    @staticmethod
    def pack_v2bf16(ty, v2bf16_list, bank):
        v4s = []
        for i in range_constexpr(4):
            v4s.append(
                vector.shuffle(
                    v2bf16_list[i * 2], v2bf16_list[i * 2 + 1], list(range_constexpr(4))
                )
            )
        v8s = []
        for i in range_constexpr(2):
            v8s.append(
                vector.shuffle(v4s[i * 2], v4s[i * 2 + 1], list(range_constexpr(8)))
            )
        result = vector.shuffle(v8s[0], v8s[1], list(range_constexpr(16)))
        return set_vgpr_bank(result, bank)
    @staticmethod
    def pair_v_tiles(v_tiles_raw, ty):
        v_paired = []
        for bank in range_constexpr(N_V_MSB):
            bank_raw = []
            for msb in range_constexpr(NUM_MSB):
                if const_expr(msb % N_V_MSB == bank):
                    bank_raw.extend(v_tiles_raw[msb])
            v_paired.append([Fragment.wmma_bf16(bank_raw[n * 2], bank_raw[n * 2 + 1]) for n in range_constexpr(N_PV_WMMA_N)])
        return v_paired
    @staticmethod
    def load_k_su(ty, kv_lds_addrs, blk, su):
        kv_raw = [[None] * N_LDS_PER_MSB for _ in range_constexpr(NUM_MSB)]
        lds_schedule = Pipeline.build_lds_k_schedule(blk, su)
        for lds_op in lds_schedule:
            kv_raw = Pipeline.emit_lds_load(ty, lds_op, kv_lds_addrs, kv_raw)
        Atom.s_wait_dscnt(0)
        sched_barrier(0)
        return Fragment.pair_k_tiles(kv_raw, ty)
    @staticmethod
    def load_v_two_sus(ty, kv_lds_addrs, blk, su0, su1):
        sched0 = Pipeline.build_lds_v_schedule(blk, su0)
        sched1 = Pipeline.build_lds_v_schedule(blk, su1)

        raw0 = [[None] * N_LDS_V_PER_MSB for _ in range_constexpr(NUM_MSB)]
        raw1 = [[None] * N_LDS_V_PER_MSB for _ in range_constexpr(NUM_MSB)]

        for msb in range_constexpr(NUM_MSB):
            for op in sched0:
                if const_expr(op["msb"] == msb):
                    raw0 = Pipeline.emit_lds_load(ty, op, kv_lds_addrs, raw0)
            for op in sched1:
                if const_expr(op["msb"] == msb):
                    raw1 = Pipeline.emit_lds_load(ty, op, kv_lds_addrs, raw1)
        Atom.s_wait_dscnt(0)
        sched_barrier(0)
        return raw0, raw1
class Pipeline:
    @staticmethod
    def build_qk_schedule(blk, su):
        k_MSBOFF = (su % 2) * VPS_MSB_KV
        sp_off = (blk % 2) * VPS_MSB_SP + 8 * su
        _K_FRAGS_PER_MSB = (SP_MSB_K // WMMA_K) // 2
        schedule = []
        for msb_idx in range_constexpr(2):
            for k in range_constexpr(SP_MSB_K // WMMA_K):
                for sp_msb in [msb_idx, 2 + msb_idx]:
                    for n in range_constexpr(SP_MSB_N // WMMA_N):
                        for m in range_constexpr(SP_MSB_M // WMMA_M):
                            schedule.append({
                                "sp_msb": sp_msb, "k_msb": (k // _K_FRAGS_PER_MSB) * 2 + sp_msb % 2,
                                "q_msb": (sp_msb // 2) * 2 + k // Q_WMMA_PER_MSB, "k_iter": k,
                                "k_frag": k % _K_FRAGS_PER_MSB, "n_iter": n, "m_iter": m,
                                "is_init": k == 0, "sp_off": sp_off, "k_MSBOFF": k_MSBOFF})
        assert len(schedule) == GEMM_INST_COUNT
        return schedule
    @staticmethod
    def build_pv_schedule(blk, su):
        sp_off = (blk % 2) * VPS_MSB_SP + 8 * su
        schedule = []
        for d_msb in range_constexpr(NUM_MSB):
            for n in range_constexpr(N_PV_WMMA_N):
                schedule.append({"d_msb": d_msb, "n": n, "sp_msb": d_msb // 2,
                                 "v_msb": d_msb % N_V_MSB, "sp_off": sp_off})
        assert len(schedule) == PV_GEMM_INST_COUNT
        return schedule
    @staticmethod
    def build_lds_k_schedule(blk, su):
        su_off = (blk * CNT_SU + su) * LDS_K_SU_P_SIZE
        schedule = [{"msb": msb, "offset": v_idx * 32 + su_off, "v_idx": v_idx, "load_type": "b128"}
                    for msb in range_constexpr(NUM_MSB) for v_idx in range_constexpr(N_LDS_PER_MSB)]
        assert len(schedule) == LDS_INST_COUNT
        return schedule
    @staticmethod
    def build_lds_v_schedule(blk, su):
        su_base_off = (blk * CNT_SU + su) * LDS_V_SU_P_SIZE
        schedule = [{"msb": msb, "offset": (v_idx // 2) * 32 + su_base_off, "v_idx": v_idx,
                      "half_p": v_idx & 1, "load_type": "tr16_b128"}
                    for msb in range_constexpr(NUM_MSB) for v_idx in range_constexpr(N_LDS_V_PER_MSB)]
        assert len(schedule) == LDS_V_INST_COUNT
        return schedule
    @staticmethod
    def emit_qk_wmma(ty, wmma_op, q_tiles, kv_tiles, sp_tiles):
        sp_msb, k_frag, n_iter = wmma_op["sp_msb"], wmma_op["k_frag"], wmma_op["n_iter"]
        src_a = kv_tiles[wmma_op["k_msb"]][k_frag]
        src_b = q_tiles[wmma_op["q_msb"]][wmma_op["k_iter"] % Q_WMMA_PER_MSB]
        if const_expr(wmma_op["is_init"]):
            sp_tiles[sp_msb][n_iter] = Atom.wmma_init(ty, src_a, src_b, sp_msb)
        else:
            sp_tiles[sp_msb][n_iter] = Atom.wmma_accum(ty, src_a, src_b, sp_tiles[sp_msb][n_iter], sp_msb)
        return sp_tiles
    @staticmethod
    def emit_pv_wmma(ty, wmma_op, v_tiles, p_tiles, o_tiles):
        d_msb, n = wmma_op["d_msb"], wmma_op["n"]
        o_tiles[d_msb][n] = Atom.wmma_accum(ty, v_tiles[wmma_op["v_msb"]][n], p_tiles[wmma_op["sp_msb"]], o_tiles[d_msb][n], d_msb)
        return o_tiles
    @staticmethod
    def emit_lds_load(ty, lds_op, kv_lds_addrs, kv_tiles_out):
        msb, offset, v_idx = lds_op["msb"], lds_op["offset"], lds_op["v_idx"]
        if const_expr(lds_op["load_type"] == "b128"):
            kv_tiles_out[msb][v_idx] = Atom.ds_load_b128(ty, kv_lds_addrs[msb], offset, msb)
        else:
            hp = lds_op["half_p"]
            kv_tiles_out[msb][v_idx] = Atom.ds_load_tr16_b128(ty, kv_lds_addrs[NUM_MSB + msb * 2 + (1 if hp else 0)], offset, msb)
        return kv_tiles_out
K_TILE_N = 128

# TDM dim0=200 -> LDS inner stride = 200*2 = 400B
# (2-way bank conflicts)
K_ROW_BYTES = 400
V_ROW_BYTES = 288
K_SU_HALF_OFFSET = 0x1900
V_SU_HALF_OFFSET = 0x1200

# K: dim0=192 (QK_HDIM), no padding. QK_HDIM=192 is not a multiple of any
# power-of-2 pad_interval that fits in one pad per row, so we skip padding to
# avoid the continuous-stream rotation bug. No bank-conflict padding for K.
K_TDM_CONFIG = 1 << 16  # data_size=1 (bf16), pad_enable=0
# V: dim0=128, pad_interval=128 elems=64dwords → enc_interval=5, 32B pad → enc_amount=7
V_TDM_CONFIG = (1 << 20) | (5 << 22) | (7 << 25)

TILE_N = K_TILE_N

# K_a, K_b, V_a are padded to 64KB segment boundary to prevent TDM cross-segment.
LDS_SEGMENT = 0x10000

lds_alloc_k_a = SmemAllocator(None, arch="gfx1250", global_sym_name="smem_k_a")
lds_alloc_k_a.ptr = LDS_SEGMENT

lds_alloc_k_b = SmemAllocator(None, arch="gfx1250", global_sym_name="smem_k_b")
lds_alloc_k_b.ptr = LDS_SEGMENT

lds_alloc_v_a = SmemAllocator(None, arch="gfx1250", global_sym_name="smem_v_a")
lds_alloc_v_a.ptr = LDS_SEGMENT

lds_alloc_v_b = SmemAllocator(None, arch="gfx1250", global_sym_name="smem_v_b")
# 0x9000, last buffer — no segment padding
lds_alloc_v_b.ptr = CNT_SU * LDS_V_SU_P_SIZE

TDM_D_TILE_DIM0 = 128 * 2
WV_SUBQD = 32
LDS_D_WV_SIZE = WV_SUBQD * TDM_D_TILE_DIM0 + 1024

NUW_ATTR = None

def get_nuw():
    global NUW_ATTR
    if NUW_ATTR is None:
        NUW_ATTR = ir.Attribute.parse("#arith.overflow<nuw>")
    return NUW_ATTR
def _ensure_ir_value(v):
    if isinstance(v, ir.Value):
        return v
    if hasattr(v, "ir_value"):
        return v.ir_value()
    return v
def add_nuw(a, b):
    return arith.addi(_ensure_ir_value(a), _ensure_ir_value(b), overflow_flags=get_nuw())
def mul_nuw(a, b):
    return arith.muli(_ensure_ir_value(a), _ensure_ir_value(b), overflow_flags=get_nuw())
def setreg(hwreg_enc, value):
    llvm_dialect.call_intrinsic(None, "llvm.amdgcn.s.setreg",
        [arith.constant(hwreg_enc, type=T.i32), arith.constant(value, type=T.i32)], [], [])
def phase4_q_load(lane_id, q_rsrc, stride_q_seq, wave_id, q_tile_offset_bytes=None):
    q_crd = idx2crd(lane_id, fx.make_layout((16, 2), (1, 16)))
    lane_lo = arith.index_cast(T.i32, q_crd[0])
    lane_hi = arith.index_cast(T.i32, q_crd[1])
    q_elem_off = (lane_lo * stride_q_seq + lane_hi * 16 + (wave_id * 32) * stride_q_seq) >> 2
    vec4i32_ty = T.vec(4, T.i32)
    soff_zero = arith.constant(0, type=T.i32)
    q_base_bytes = mul_nuw(q_elem_off, arith.constant(4, type=T.i32))
    if q_tile_offset_bytes is not None:
        q_base_bytes = add_nuw(q_tile_offset_bytes, q_base_bytes)
    stride_16_bytes = stride_q_seq * 16
    _k_half_c = arith.constant(QK_HDIM, type=T.i32)
    bank_offsets = [arith.constant(0, type=T.i32), _k_half_c, stride_16_bytes, add_nuw(stride_16_bytes, _k_half_c)]
    _FRAGS_PER_BANK = (QK_HDIM // 2) // 32
    _LOADS_PER_BANK = _FRAGS_PER_BANK * 2
    q_frags = []
    for bank in fx.range_constexpr(4):
        bank_voff = q_base_bytes if bank == 0 else add_nuw(q_base_bytes, bank_offsets[bank])
        bank_loads = []
        for i in fx.range_constexpr(_LOADS_PER_BANK):
            voff = bank_voff if i == 0 else add_nuw(bank_voff, arith.constant(i * 32, type=T.i32))
            bank_loads.append(set_vgpr_bank(rocdl.raw_ptr_buffer_load(vec4i32_ty, q_rsrc, voff, soff_zero, soff_zero), bank))
        rocdl.sched_barrier(0)
        q_frags.append([set_vgpr_bank(Fragment.wmma_bf16(bank_loads[2 * f], bank_loads[2 * f + 1]), bank) for f in fx.range_constexpr(_FRAGS_PER_BANK)])
        rocdl.sched_barrier(0)
    return q_frags
def head_index_div(workgroup_id, num_heads):
    return rocdl.readfirstlane(T.i32, workgroup_id // num_heads)
def split_i64_to_lo_hi(val_i64):
    return arith.trunci(T.i32, val_i64), arith.trunci(T.i32, val_i64 >> 32) | -2147483648
def load_scalar_from_tensor(ptr_tensor, idx_i32):
    """Load cu_seqlens[idx] as i32 SGPR (uniform across wavefront)."""
    gp = glb_ptr_ty()
    return rocdl.readfirstlane(T.i32, llvm_dialect.load(T.i32,
        llvm_dialect.inttoptr(gp, ptr_base_i64(ptr_tensor) + fx.Int64(idx_i32 * 4))))
def ptr_base_i64(tensor):
    return llvm_dialect.ptrtoint(T.i64, fly_d.extract_aligned_pointer_as_index(glb_ptr_ty(), tensor.__extract_to_ir_values__()[0]))
def compute_global_addr(tensor, byte_offset, wave_id, stride_32):
    return ptr_base_i64(tensor) + fx.Int64(byte_offset) + fx.Int64(wave_id * stride_32)
def extract_lds_base_i32(memref_base):
    from flydsl._mlir.dialects import memref as _memref_d
    return arith.index_cast(T.i32, _memref_d.extract_aligned_pointer_as_index(memref_base))
def build_kv_lds_addrs(lane_id, k_base_i32, v_base_i32):
    k_crd = idx2crd(lane_id, fx.make_layout((16, 2), (1, 16)))
    k_row = arith.index_cast(T.i32, k_crd[0])
    k_col = arith.index_cast(T.i32, k_crd[1])
    k_dh0 = k_base_i32 + k_row * K_ROW_BYTES + k_col * 16
    k_dh1 = k_dh0 + K_SU_HALF_OFFSET
    K_COL_D_HALF = QK_HDIM * KV_BPP // 2
    v_crd = idx2crd(lane_id, fx.make_layout((8, 2, 2), (1, 8, 16)))
    v_row = arith.index_cast(T.i32, v_crd[0]) + arith.index_cast(T.i32, v_crd[2]) * 8
    v_dh0 = v_base_i32 + v_row * V_ROW_BYTES + arith.index_cast(T.i32, v_crd[1]) * 16
    v_dh1 = v_dh0 + V_SU_HALF_OFFSET
    rocdl.sched_barrier(0)
    _V_D_HALF = V_HDIM * KV_BPP // 2
    _V_COL_GROUP = (N_LDS_V_PER_MSB // 2) * 32
    _V_MSB_EXTRA = [0, _V_D_HALF, _V_COL_GROUP, _V_D_HALF + _V_COL_GROUP]
    v_addrs = []
    for msb in range(NUM_MSB):
        e = _V_MSB_EXTRA[msb]
        d0 = v_dh0 if e == 0 else v_dh0 + e
        d1 = v_dh1 if e == 0 else v_dh1 + e
        v_addrs += [set_vgpr_bank(d0, msb), set_vgpr_bank(d1, msb)]
    return [set_vgpr_bank(k_dh0, 0), set_vgpr_bank(k_dh1, 1),
            set_vgpr_bank(k_dh0 + K_COL_D_HALF, 2), set_vgpr_bank(k_dh1 + K_COL_D_HALF, 3)] + v_addrs
def issue_k_loads(ty, kv_lds_addrs, blk, su):
    su_off = (blk * CNT_SU + su) * LDS_K_SU_P_SIZE
    kv_raw = [[None] * N_LDS_PER_MSB for _ in range(NUM_MSB)]
    for msb in range(NUM_MSB):
        for v_idx in range(N_LDS_PER_MSB):
            offset = v_idx * 32 + su_off
            kv_raw[msb][v_idx] = Atom.ds_load_b128(ty, kv_lds_addrs[msb], offset, msb)
    return kv_raw
def wait_and_pair_k(ty, kv_raw):
    rocdl.s_wait_dscnt(0)
    return Fragment.pair_k_tiles(kv_raw, ty)
def load_initial_kv_tiles(ty, kv_lds_addrs, blk, su):
    kv_raw = issue_k_loads(ty, kv_lds_addrs, blk, su)
    return wait_and_pair_k(ty, kv_raw)
def _dispatch_tdm_at_wmma0(ty, tdm_type, tdm_state, has_fallback=True):
    tdm_key = "v" if tdm_type == KV_V else "k"
    descs = tdm_state.get(f"{tdm_key}_descs", None)
    sched_barrier(0)
    if const_expr(descs is not None):
        _tdm_di = tdm_state[f"{tdm_key}_desc_idx"]
        for _ in range_constexpr(TDM_LOADS_PER_STAGE):
            if const_expr(_tdm_di < len(descs)):
                Atom.tdm_load(ty, descs[_tdm_di][0], descs[_tdm_di][1])
                _tdm_di += 1
        tdm_state[f"{tdm_key}_desc_idx"] = _tdm_di
    elif const_expr(has_fallback):
        if const_expr(tdm_type == KV_V):
            Atom.tdm_load(ty, tdm_state["v_g0"], tdm_state["v_g1"])
        else:
            Atom.tdm_load(ty, tdm_state["k_g0"], tdm_state["k_g1"])
    sched_barrier(0)
def _dispatch_lds_tok(ty, _tok, lds_schedule, kv_lds_addrs, kv_tiles_next, lds_idx, ds_issued):
    if const_expr(9 <= _tok <= 12):
        if lds_idx < LDS_INST_COUNT:
            kv_tiles_next = Pipeline.emit_lds_load(
                ty, lds_schedule[lds_idx], kv_lds_addrs, kv_tiles_next
            )
            lds_idx += 1
            ds_issued += 1
    elif const_expr(13 <= _tok <= 16):
        if lds_idx < LDS_V_INST_COUNT:
            kv_tiles_next = Pipeline.emit_lds_load(
                ty, lds_schedule[lds_idx], kv_lds_addrs, kv_tiles_next
            )
            lds_idx += 1
            ds_issued += 1
    return kv_tiles_next, lds_idx, ds_issued
def gemm1_interleaved_stage(
    ty, stage, gemm_blk, gemm_su, tdm_type, tdm_blk, tdm_su,
    lds_type, lds_blk, lds_su, q_tiles, kv_tiles, sp_tiles,
    kv_lds_addrs, kv_tiles_next, softmax_ops_by_msb, softmax_idx_by_msb,
    softmax_budget, tdm_state, tdm_barrier=False, o_rescale_ops=None,
):
    has_tdm = tdm_type != KV_NONE
    wmma_schedule = Pipeline.build_qk_schedule(gemm_blk, gemm_su)
    if const_expr(lds_type == KV_K):
        lds_schedule = Pipeline.build_lds_k_schedule(lds_blk, lds_su)
    else:
        lds_schedule = Pipeline.build_lds_v_schedule(lds_blk, lds_su)
    lds_idx = 0
    ds_issued = 0
    _o_resc_idx = 0
    for gemm_idx in range_constexpr(GEMM_INST_COUNT):
        wmma_op = wmma_schedule[gemm_idx]
        sp_tiles = Pipeline.emit_qk_wmma(ty, wmma_op, q_tiles, kv_tiles, sp_tiles)
        sched_barrier(0)
        sched_barrier(0)
        if const_expr(gemm_idx == GEMM_INST_COUNT // 4 - 1):
            Atom.s_wait_dscnt(ds_issued)
        sched_barrier(0)
        if const_expr(gemm_idx == 0 and has_tdm):
            _dispatch_tdm_at_wmma0(ty, tdm_type, tdm_state, has_fallback=True)
        _g1_barrier_idx = GEMM_INST_COUNT - BARRIER_SIGNAL_AHEAD - 1
        if const_expr(tdm_barrier and gemm_idx == _g1_barrier_idx):
            Atom.s_wait_tensorcnt(4)
            rocdl.s_barrier_signal(-1)
        _g1_row = GEMM1_SCHEDULE[g1_row_idx(stage, gemm_idx)]
        _g1_half = len(_g1_row) // 2
        for _i in range_constexpr(len(_g1_row)):
            if const_expr(_i < _g1_half):
                _tok = _g1_row[_i]
                if const_expr(5 <= _tok <= 8):
                    _msb = _tok - P2_BASE
                    if softmax_idx_by_msb[_msb] < len(softmax_ops_by_msb[_msb]):
                        softmax_ops_by_msb[_msb][softmax_idx_by_msb[_msb]]()
                        softmax_idx_by_msb[_msb] += 1
                    sched_barrier(0)
                elif const_expr(_tok == O_RESC0):
                    if const_expr(o_rescale_ops is not None):
                        sched_barrier(0)
                        o_rescale_ops[_o_resc_idx]()
                        _o_resc_idx += 1
                        sched_barrier(0)
                else:
                    kv_tiles_next, lds_idx, ds_issued = _dispatch_lds_tok(
                        ty, _tok, lds_schedule, kv_lds_addrs, kv_tiles_next, lds_idx, ds_issued)
        sched_barrier(0)
        if const_expr(gemm_idx == GEMM_INST_COUNT - 1):
            Atom.s_wait_dscnt(LDS_INST_COUNT // 2)
        sched_barrier(0)
        for _i in range_constexpr(len(_g1_row)):
            if const_expr(_g1_half <= _i < len(_g1_row)):
                _tok = _g1_row[_i]
                if const_expr(5 <= _tok <= 8):
                    _msb = _tok - P2_BASE
                    if softmax_idx_by_msb[_msb] < len(softmax_ops_by_msb[_msb]):
                        softmax_ops_by_msb[_msb][softmax_idx_by_msb[_msb]]()
                        softmax_idx_by_msb[_msb] += 1
                    sched_barrier(0)
                elif const_expr(_tok == O_RESC0):
                    if const_expr(o_rescale_ops is not None):
                        sched_barrier(0)
                        o_rescale_ops[_o_resc_idx]()
                        _o_resc_idx += 1
                        sched_barrier(0)
                else:
                    kv_tiles_next, lds_idx, ds_issued = _dispatch_lds_tok(
                        ty, _tok, lds_schedule, kv_lds_addrs, kv_tiles_next, lds_idx, ds_issued)
        if const_expr(tdm_barrier and gemm_idx == GEMM_INST_COUNT - 1):
            rocdl.s_barrier_wait(-1)
        sched_barrier(0)
    return sp_tiles, kv_tiles_next
def _dispatch_g2_tok(ty, _tok, ops_by_rid, rid_idx, exp_rid_idx, lds_schedule,
                     kv_lds_addrs, kv_tiles_next, lds_idx, ds_issued):
    if const_expr(0 <= _tok < RLTS_LEN):
        _rid = _tok
        if const_expr(5 <= _rid <= 8):
            if rid_idx[_rid] < PART2_EXP_START and rid_idx[_rid] < len(ops_by_rid[_rid]):
                ops_by_rid[_rid][rid_idx[_rid]]()
                rid_idx[_rid] += 1
        else:
            if rid_idx[_rid] < len(ops_by_rid[_rid]):
                ops_by_rid[_rid][rid_idx[_rid]]()
                rid_idx[_rid] += 1
        sched_barrier(0)
    elif const_expr(19 <= _tok <= 22):
        _msb = _tok - EXP_BASE
        _erid = _msb + P2_BASE
        if exp_rid_idx[_msb] < len(ops_by_rid[_erid]):
            ops_by_rid[_erid][exp_rid_idx[_msb]]()
            exp_rid_idx[_msb] += 1
        sched_barrier(0)
    else:
        kv_tiles_next, lds_idx, ds_issued = _dispatch_lds_tok(
            ty, _tok, lds_schedule, kv_lds_addrs, kv_tiles_next, lds_idx, ds_issued)
    return kv_tiles_next, lds_idx, ds_issued
def gemm2_interleaved_stage(
    ty, stage, gemm_blk, gemm_su, lds_type, lds_blk, lds_su,
    v_tiles, p_tiles, o_tiles, kv_lds_addrs, kv_tiles_next,
    ops_by_rid, rid_idx, tdm_state=None, tdm_type=KV_NONE,
    tdm_barrier=False, o_rescale_exp_delta=None,
):
    has_tdm = tdm_type != KV_NONE
    wmma_schedule = Pipeline.build_pv_schedule(gemm_blk, gemm_su)
    if const_expr(lds_type == KV_K):
        lds_schedule = Pipeline.build_lds_k_schedule(lds_blk, lds_su)
    else:
        lds_schedule = Pipeline.build_lds_v_schedule(lds_blk, lds_su)
    lds_idx = 0
    ds_issued = 0
    exp_rid_idx = [PART2_EXP_START] * NUM_MSB
    _o_rescale_ed_v8 = {}
    def _build_o_rescale_ed_v8(d_msb):
        if d_msb not in _o_rescale_ed_v8:
            _ed = o_rescale_exp_delta[d_msb]
            if _ed is None:
                _o_rescale_ed_v8[d_msb] = None
                return
            _o_rescale_ed_v8[d_msb] = fx.vector.broadcast(T.vec(8, T.f32), _ed)
    def _emit_o_rescale_tile(d_msb, n):
        if const_expr(o_rescale_exp_delta is None):
            return
        _build_o_rescale_ed_v8(d_msb)
        _ed_v8 = _o_rescale_ed_v8[d_msb]
        if _ed_v8 is None:
            return
        o_tiles[d_msb][n] = o_tiles[d_msb][n] * _ed_v8
    if const_expr(stage == 0):
        sched_barrier(0)
        for _n0 in range_constexpr(N_PV_WMMA_N):
            _emit_o_rescale_tile(0, _n0)
        sched_barrier(0)
    for gemm_idx in range_constexpr(PV_GEMM_INST_COUNT):
        if const_expr(stage == 0 and 3 * N_PV_WMMA_N <= gemm_idx < PV_GEMM_INST_COUNT):
            sched_barrier(0)
            _emit_o_rescale_tile(3, gemm_idx - 3 * N_PV_WMMA_N)
            sched_barrier(0)
        wmma_op = wmma_schedule[gemm_idx]
        o_tiles = Pipeline.emit_pv_wmma(ty, wmma_op, v_tiles, p_tiles, o_tiles)
        sched_barrier(0)
        if const_expr(stage == 0):
            if const_expr(0 <= gemm_idx < N_PV_WMMA_N):
                sched_barrier(0)
                _emit_o_rescale_tile(1, gemm_idx)
                sched_barrier(0)
            elif const_expr(N_PV_WMMA_N <= gemm_idx < 2 * N_PV_WMMA_N):
                sched_barrier(0)
                _emit_o_rescale_tile(2, gemm_idx - N_PV_WMMA_N)
                sched_barrier(0)
        sched_barrier(0)
        if const_expr(gemm_idx == PV_GEMM_INST_COUNT // 4 - 1):
            Atom.s_wait_dscnt(ds_issued)
        sched_barrier(0)
        if const_expr(gemm_idx == 0 and has_tdm):
            _dispatch_tdm_at_wmma0(ty, tdm_type, tdm_state, has_fallback=False)
        _pv_barrier_idx = PV_GEMM_INST_COUNT - BARRIER_SIGNAL_AHEAD - 1
        if const_expr(tdm_barrier and gemm_idx == _pv_barrier_idx):
            Atom.s_wait_tensorcnt(4)
            rocdl.s_barrier_signal(-1)
        _g2_row = GEMM2_SCHEDULE[g2_row_idx(stage, gemm_idx)]
        _g2_half = len(_g2_row) // 2
        for _i in range_constexpr(len(_g2_row)):
            if const_expr(_i < _g2_half):
                _tok = _g2_row[_i]
                kv_tiles_next, lds_idx, ds_issued = _dispatch_g2_tok(
                    ty, _tok, ops_by_rid, rid_idx, exp_rid_idx, lds_schedule,
                    kv_lds_addrs, kv_tiles_next, lds_idx, ds_issued)
        sched_barrier(0)
        if const_expr(gemm_idx == PV_GEMM_INST_COUNT - 1):
            Atom.s_wait_dscnt(LDS_INST_COUNT // 2)
        sched_barrier(0)
        for _i in range_constexpr(len(_g2_row)):
            if const_expr(_g2_half <= _i < len(_g2_row)):
                _tok = _g2_row[_i]
                kv_tiles_next, lds_idx, ds_issued = _dispatch_g2_tok(
                    ty, _tok, ops_by_rid, rid_idx, exp_rid_idx, lds_schedule,
                    kv_lds_addrs, kv_tiles_next, lds_idx, ds_issued)
        if const_expr(tdm_barrier and gemm_idx == PV_GEMM_INST_COUNT - 1):
            rocdl.s_barrier_wait(-1)
        sched_barrier(0)
    return o_tiles, kv_tiles_next
def qk_gemm_pure(ty, blk, su, q_tiles, kv_tiles, sp_tiles):
    schedule = Pipeline.build_qk_schedule(blk, su)
    for wmma_op in schedule:
        sp_tiles = Pipeline.emit_qk_wmma(ty, wmma_op, q_tiles, kv_tiles, sp_tiles)
        sched_barrier(0)
    return sp_tiles
def pv_gemm_pure(ty, blk, su, v_tiles, p_tiles_su, o_tiles):
    schedule = Pipeline.build_pv_schedule(blk, su)
    for wmma_op in schedule:
        o_tiles = Pipeline.emit_pv_wmma(ty, wmma_op, v_tiles, p_tiles_su, o_tiles)
        sched_barrier(0)
    return o_tiles
class TDM:
    @staticmethod
    def build_descs(dg1, addr_i64, stride_adv_i64, lds_base, su_p_size, n_su):
        _dg1_list = dg1 if isinstance(dg1, list) else [dg1] * n_su
        pred = fx.Int32(1)
        cur_addr = addr_i64
        descs = []
        for su in range(n_su):
            lds_off = lds_base + (su * su_p_size)
            addr_lo, addr_hi = split_i64_to_lo_hi(cur_addr)
            dg0 = Vec.from_elements([pred, lds_off, addr_lo, addr_hi], fx.Int32)
            descs.append((dg0, _dg1_list[su]))
            if su < n_su - 1:
                cur_addr = cur_addr + stride_adv_i64
        return descs
    @staticmethod
    def issue_from_descs(descs):
        for dg0, dg1 in descs:
            tdm_ops.tensor_load_2d(tdm_ops.TDMDescriptor2D(dg0, dg1))
            rocdl.s_barrier_signal(-1)
            rocdl.s_barrier_wait(-1)
    @staticmethod
    def per_warp_oob_dim1(total_rows_i32, wave_id, rows_per_warp=8):
        remaining = total_rows_i32 - wave_id * rows_per_warp
        return arith.minsi(arith.maxsi(remaining, arith.constant(0, type=T.i32)),
                           arith.constant(rows_per_warp, type=T.i32))
    @staticmethod
    def make_kv_dg1_with_oob(config_bf16, dim0_elems, dim1_rows, stride_seq_elems,
                              oob_dim1_raw, dim0_stride=None):
        _sgpr2 = arith.shli(arith.andi(oob_dim1_raw, arith.constant(0xFFFF, type=T.i32)),
                            arith.constant(16, type=T.i32))
        if dim0_stride is None: dim0_stride = dim0_elems
        return Vec.from_elements(
            [fx.Int32(config_bf16), fx.Int32(dim0_elems << 16), _sgpr2,
             fx.Int32(dim0_stride << 16), fx.Int32(dim1_rows), stride_seq_elems,
             fx.Int32(0), fx.Int32(0)], fx.Int32)
    @staticmethod
    def build_oob_dg1_list(config, dim0_elems, stride_elems, remain, wave_id, dim0_stride=None):
        return [TDM.make_kv_dg1_with_oob(config, dim0_elems, 8, stride_elems,
                TDM.per_warp_oob_dim1(remain - su * 32, wave_id, 8), dim0_stride=dim0_stride)
                for su in range(CNT_SU)]
    @staticmethod
    def load_kv_blk(kv_type, dg1, addr_i64, stride_adv_i64, lds_base, su_p_size, n_su):
        TDM.issue_from_descs(TDM.build_descs(dg1, addr_i64, stride_adv_i64, lds_base, su_p_size, n_su))
    @staticmethod
    def _load_kv(is_k, ptr_tensor, offset, stride_seq, stride_32, wave_id, lds_base_i32, oob_dg1_list=None):
        config = (1 << 16) | (K_TDM_CONFIG if is_k else V_TDM_CONFIG)
        dim0 = QK_HDIM if is_k else 128
        dim0_stride = 200 if is_k else 128
        dg1 = oob_dg1_list if oob_dg1_list is not None else Vec.from_elements(
            [fx.Int32(config), fx.Int32(dim0 << 16), fx.Int32(8 << 16),
             fx.Int32(dim0_stride << 16), fx.Int32(8), stride_seq >> 1,
             fx.Int32(0), fx.Int32(0)], fx.Int32)
        row_bytes = K_ROW_BYTES if is_k else V_ROW_BYTES
        su_p_size = LDS_K_SU_P_SIZE if is_k else LDS_V_SU_P_SIZE
        TDM.load_kv_blk(KV_K if is_k else KV_V, dg1,
            compute_global_addr(ptr_tensor, offset, wave_id, 8 * stride_seq),
            fx.Int64(stride_32), lds_base_i32 + wave_id * (8 * row_bytes), su_p_size, CNT_SU)
        tdm_wait_and_barrier()
    @staticmethod
    def load_k_only(ptr_K, k_offset, stride_k_seq, stride_k_32, wave_id, lds_base_i32, oob_dg1_list=None):
        TDM._load_kv(True, ptr_K, k_offset, stride_k_seq, stride_k_32, wave_id, lds_base_i32, oob_dg1_list)
    @staticmethod
    def load_v_only(ptr_V, v_offset, stride_v_seq, stride_v_32, wave_id, lds_base_i32, oob_dg1_list=None):
        TDM._load_kv(False, ptr_V, v_offset, stride_v_seq, stride_v_32, wave_id, lds_base_i32, oob_dg1_list)

_KV_SIZE = NUM_MSB * N_WMMA_K_TILES
_OFF_LOCAL_MAX = 24 + _KV_SIZE
_OFF_DELTA = _OFF_LOCAL_MAX + NUM_MSB
_OFF_SP = _OFF_DELTA + NUM_MSB
_OFF_PP = _OFF_SP + CNT_SU * NUM_MSB
_OFF_PSP = _OFF_PP + 4
_PSP_SIZE = NUM_MSB * N_SP_PAIRS
_OFF_PSP_HI = _OFF_PSP + _PSP_SIZE
_OFF_PED = _OFF_PSP_HI + _PSP_SIZE

def xcd_remap(raw_bx, raw_by, raw_bz, gdx, gdy, gdz):
    """Software XCD remap: flat wgid -> chunked assignment for K/V cache locality."""
    _NUM_XCDS = 8
    wgid = raw_bx + gdx * raw_by + gdx * gdy * raw_bz
    num_wgs = gdx * gdy * gdz
    wgs_per_xcd = num_wgs // _NUM_XCDS
    do_remap = (num_wgs > _NUM_XCDS) & (num_wgs % _NUM_XCDS == 0)
    new_wgid = do_remap.select(
        (wgid % _NUM_XCDS) * wgs_per_xcd + wgid // _NUM_XCDS, wgid)
    new_bx = new_wgid % gdx
    new_tmp = new_wgid // gdx
    return new_bx, new_tmp % gdy, new_tmp // gdy


def build_init_args(zero_v8f32, softmax_state_pro, sp_pairs_all_pro,
                    kv_tiles_init, all_su_sp_tiles,
                    k_b_base, v_a_base, k_a_base, v_b_base):
    """Pack prologue results into scf.for_ iter_args list."""
    _ssp = softmax_state_pro
    pro_old_max = [_ssp["old_max"][m] for m in fx.range_constexpr(NUM_MSB)]
    pro_row_sums = [_ssp["row_sums"][m] for m in fx.range_constexpr(NUM_MSB)]
    pro_local_max = [_ssp["local_max"][m] for m in fx.range_constexpr(NUM_MSB)]
    pro_delta = [_ssp["delta"][m] for m in fx.range_constexpr(NUM_MSB)]
    pro_partial_sp_lo_flat, pro_partial_sp_hi_flat = [], []
    for m in fx.range_constexpr(NUM_MSB):
        for i in fx.range_constexpr(N_SP_PAIRS):
            pair = Vec(sp_pairs_all_pro[m][i], dtype=Float32)
            pro_partial_sp_lo_flat.append(pair[0].ir_value())
            pro_partial_sp_hi_flat.append(pair[1].ir_value())
    pro_exp_delta = [_ssp["exp_delta"][m] for m in fx.range_constexpr(NUM_MSB)]
    kv_flat = [kv_tiles_init[msb][k]
               for msb in fx.range_constexpr(NUM_MSB)
               for k in fx.range_constexpr(N_WMMA_K_TILES)]
    sp_flat = [all_su_sp_tiles[su][msb][0]
               for su in fx.range_constexpr(CNT_SU)
               for msb in fx.range_constexpr(NUM_MSB)]
    return ([zero_v8f32] * (NUM_MSB * N_PV_WMMA_N) + pro_old_max + pro_row_sums
            + kv_flat + pro_local_max + pro_delta + sp_flat
            + [k_b_base, v_a_base, k_a_base, v_b_base]
            + pro_partial_sp_lo_flat + pro_partial_sp_hi_flat + pro_exp_delta)


def unpack_loop_results(lr, lane_id):
    """Unpack scf.for_ loop results into epilogue state dict."""
    ep_partial_sp_lo = [lr[_OFF_PSP + i] for i in fx.range_constexpr(_PSP_SIZE)]
    ep_partial_sp_hi = [lr[_OFF_PSP_HI + i] for i in fx.range_constexpr(_PSP_SIZE)]
    return {
        "o_tiles": [[set_vgpr_bank(lr[d * N_PV_WMMA_N + n], d)
                      for n in fx.range_constexpr(N_PV_WMMA_N)]
                     for d in fx.range_constexpr(NUM_MSB)],
        "old_max": [set_vgpr_bank(lr[16 + i], i)
                    for i in fx.range_constexpr(NUM_MSB)],
        "row_sums": [set_vgpr_bank(lr[20 + i], i)
                     for i in fx.range_constexpr(NUM_MSB)],
        "kv_tiles": [[set_vgpr_bank(lr[24 + m * N_WMMA_K_TILES + k], m)
                       for k in fx.range_constexpr(N_WMMA_K_TILES)]
                      for m in fx.range_constexpr(NUM_MSB)],
        "local_max": [set_vgpr_bank(lr[_OFF_LOCAL_MAX + i], i)
                      for i in fx.range_constexpr(NUM_MSB)],
        "delta": [set_vgpr_bank(lr[_OFF_DELTA + i], i)
                  for i in fx.range_constexpr(NUM_MSB)],
        "k_cur_base": lr[_OFF_PP],
        "v_cur_base": lr[_OFF_PP + 1],
        "k_next_base": lr[_OFF_PP + 2],
        "v_next_base": lr[_OFF_PP + 3],
        "kv_lds_addrs": build_kv_lds_addrs(lane_id, lr[_OFF_PP], lr[_OFF_PP + 1]),
        "kv_lds_addrs_next": build_kv_lds_addrs(
            lane_id, lr[_OFF_PP + 2], lr[_OFF_PP + 3]),
        "partial_sp_pairs": [
            [make_v2f32(ep_partial_sp_lo[m * N_SP_PAIRS + i],
                        ep_partial_sp_hi[m * N_SP_PAIRS + i], m)
             for i in fx.range_constexpr(N_SP_PAIRS)]
            for m in fx.range_constexpr(NUM_MSB)],
        "exp_delta": [set_vgpr_bank(lr[_OFF_PED + m], m)
                      for m in fx.range_constexpr(NUM_MSB)],
    }


def prologue_tile0(ctx, ty, q_frags, kv_lds_addrs_a, k_a_base_i32, v_a_base_i32,
                   k_oob_dg1, v_oob_dg1, IS_CAUSAL, sgpr_state):
    """Prologue: K(tile0) load -> QK GEMM -> V(tile0) load -> masks -> softmax PART0/1/2."""
    wave_id = ctx["wave_id"]
    zero_v8f32 = fx.constant_vector(0.0, T.vec(8, T.f32))
    zero_f32 = arith.constant(0.0, type=T.f32)
    neg_inf = arith.constant(float("-inf"), type=T.f32)
    rocdl.sched_barrier(0)
    TDM.load_k_only(ctx["ptr_K"], ctx["k_offset"], ctx["stride_k_seq"], ctx["stride_k_32"],
                     wave_id, k_a_base_i32, oob_dg1_list=k_oob_dg1)
    rocdl.sched_barrier(0)
    all_su_sp_tiles = []
    for su in fx.range_constexpr(CNT_SU):
        fresh_sp = qk_gemm_pure(ty, 0, su, q_frags, Fragment.load_k_su(ty, kv_lds_addrs_a, 0, su),
                                [[zero_v8f32] for msb in fx.range_constexpr(NUM_MSB)])
        all_su_sp_tiles.append(fresh_sp)
    TDM.load_v_only(ctx["ptr_V"], ctx["v_offset"], ctx["stride_v_seq"], ctx["stride_v_32"],
                     wave_id, v_a_base_i32, oob_dg1_list=v_oob_dg1)
    causal_offset = ctx["actual_kv_len"] - ctx["actual_q_len"]
    if const_expr(IS_CAUSAL):
        apply_causal_mask(ctx, all_su_sp_tiles, -causal_offset)
    apply_kv_oob_mask(ctx, all_su_sp_tiles, ctx["actual_kv_len"])
    sp_pairs_all_pro = Softmax.tiles_to_pairs(all_su_sp_tiles)
    softmax_state_pro = make_softmax_state(
        [set_vgpr_bank(neg_inf, m) for m in range(NUM_MSB)],
        [set_vgpr_bank(neg_inf, m) for m in range(NUM_MSB)],
        [set_vgpr_bank(zero_f32, m) for m in range(NUM_MSB)],
        [set_vgpr_bank(zero_f32, m) for m in range(NUM_MSB)],
        sp_pairs_prev=sp_pairs_all_pro)
    Softmax.part01_only(ty, 0, sp_pairs_all_pro, softmax_state_pro, sgpr_state)
    pro_part2_ops = Softmax.build_all_part2_ops(ty, 0, sp_pairs_all_pro, softmax_state_pro, sgpr_state)
    for m in fx.range_constexpr(NUM_MSB):
        for op in pro_part2_ops[m][:PART2_SPLIT]:
            op()
    return softmax_state_pro, sp_pairs_all_pro, all_su_sp_tiles, causal_offset, zero_v8f32


def compute_num_tiles(actual_kv_len, actual_q_len, bx, tile_n_const, causal_offset, IS_CAUSAL):
    """Compute number of KV tiles, first causal tile index, and num_tiles_minus1 index."""
    kv_tiles_avail = (actual_kv_len + (tile_n_const - 1)) // tile_n_const
    if const_expr(IS_CAUSAL):
        sk_sq_diff = actual_kv_len - actual_q_len
        sk_sq_tiles = (sk_sq_diff + (tile_n_const - 1)) // tile_n_const
        bx_plus_1 = bx + 1
        causal_tiles = bx_plus_1 + sk_sq_tiles
        num_tiles = arith.minui(causal_tiles.ir_value(), kv_tiles_avail)
    else:
        num_tiles = kv_tiles_avail
    num_tiles_idx = arith.index_cast(T.index, num_tiles)
    num_tiles_minus1 = num_tiles - 1
    num_tiles_minus1_idx = arith.index_cast(T.index, num_tiles_minus1)
    if const_expr(IS_CAUSAL):
        first_causal_tile = bx + causal_offset // tile_n_const
        first_causal_tile = arith.maxsi(
            first_causal_tile.ir_value(), arith.constant(1, type=T.i32)
        )
        first_causal_tile = arith.minui(first_causal_tile, num_tiles_minus1)
        first_causal_tile_idx = arith.index_cast(T.index, first_causal_tile)
    else:
        first_causal_tile_idx = num_tiles_minus1_idx
    return num_tiles, num_tiles_idx, num_tiles_minus1_idx, first_causal_tile_idx


def endtile_pipeline(ctx, ty, ep, q_frags, sgpr_state, num_tiles, num_tiles_idx,
                     tile_n_const, causal_offset, IS_CAUSAL, _V_CFG, zero_v8f32):
    """Endtile: run fmha_pipeline on last tile + ep_finish (for num_tiles >= 2)."""
    v_offset = ctx["v_offset"]
    stride_v_seq = ctx["stride_v_seq"]
    wave_id = ctx["wave_id"]
    actual_kv_len = ctx["actual_kv_len"]
    et_sp_t = [[set_vgpr_bank(zero_v8f32, m)] for m in fx.range_constexpr(NUM_MSB)]
    et_sfx = make_softmax_state(ep["old_max"], ep["local_max"], ep["delta"], ep["row_sums"],
        sp_pairs_prev=[[ep["partial_sp_pairs"][m][i] for i in fx.range_constexpr(N_SP_PAIRS)]
                       for m in fx.range_constexpr(NUM_MSB)])
    et_tdm = {"v_g0": fx.constant_vector(0, T.vec(4, T.i32)),
              "v_g1": fx.constant_vector(0, T.vec(8, T.i32)),
              "k_g0": fx.constant_vector(0, T.vec(4, T.i32)),
              "k_g1": fx.constant_vector(0, T.vec(8, T.i32)),
              "v_salu_queue": [], "k_salu_queue": []}
    et_o = [[ep["o_tiles"][d][n] for n in range(N_PV_WMMA_N)] for d in range(NUM_MSB)]
    et_causal_ns = (arith.index_cast(T.i32, num_tiles_idx) - 1) * TILE_N - causal_offset if const_expr(IS_CAUSAL) else None
    et_kv_remain = actual_kv_len - (num_tiles - 1) * tile_n_const
    et_v_oob_dg1 = TDM.build_oob_dg1_list(_V_CFG, 128, stride_v_seq >> 1, et_kv_remain, wave_id)
    ep_v_endtile_offset = v_offset + (num_tiles - 1) * tile_n_const * stride_v_seq
    _, _, et_o, _, et_psp_lo, et_psp_hi, et_ped = fmha_pipeline_ctx(
        ctx, ty, False, q_frags, ep["kv_tiles"], et_sp_t, et_o, ep["kv_lds_addrs"], et_tdm,
        et_sfx, sgpr_state, gemm2=True, tdm_v_offset=ep_v_endtile_offset,
        tdm_v_target=ep["v_next_base"], tdm_k_offset=None,
        kv_lds_addrs_next=ep["kv_lds_addrs_next"], gemm1_tdm_is_v=True,
        ia_exp_delta=ep["exp_delta"], causal_n_start=et_causal_ns,
        endtile_v_oob_dg1=et_v_oob_dg1, kv_oob_cols=et_kv_remain)
    et_psp = [[make_v2f32(et_psp_lo[m * N_SP_PAIRS + i], et_psp_hi[m * N_SP_PAIRS + i], m)
               for i in fx.range_constexpr(N_SP_PAIRS)] for m in fx.range_constexpr(NUM_MSB)]
    tdm_wait_and_barrier()
    _ep_finish(ctx, et_o, et_psp, et_ped, ep["v_next_base"],
               et_sfx["old_max"], et_sfx["local_max"], et_sfx["delta"], et_sfx["row_sums"],
               ep["k_cur_base"])


def apply_causal_mask(ctx, su_sp_tiles, n_start_fx):
    lane_id = ctx["lane_id"]
    wave_id = ctx["wave_id"]
    m_start = ctx["m_start"]
    lane_lo = lane_id & 15
    lane_hi_x8 = (lane_id >> 4) * 8
    wave_x32 = wave_id * 32
    base = (m_start - n_start_fx) + wave_x32 + (lane_lo - lane_hi_x8)
    neg_inf_c = arith.constant(float("-inf"), type=T.f32)
    for su in fx.range_constexpr(CNT_SU):
        for msb in fx.range_constexpr(NUM_MSB):
            off = (msb // 2) * 16 - su * 32 - (msb % 2) * 16
            bnd_fx = base + off
            v8w = Vec(su_sp_tiles[su][msb][0], dtype=fx.Float32)
            masked_elems = []
            for e in fx.range_constexpr(8):
                cmp_fx = bnd_fx < e
                elem_fx = v8w[e]
                mval_fx = cmp_fx.select(neg_inf_c, elem_fx)
                masked_elems.append(mval_fx)
            su_sp_tiles[su][msb][0] = Vec.from_elements(masked_elems, fx.Float32)
def apply_kv_oob_mask(ctx, su_sp_tiles, kv_remain_raw):
    lane_id = ctx["lane_id"]
    lane_hi = lane_id >> 4
    lane_hi_x8 = lane_hi * 8
    base = (kv_remain_raw - 1) - lane_hi_x8
    neg_inf = arith.constant(float("-inf"), type=T.f32)
    for su in fx.range_constexpr(CNT_SU):
        for msb in fx.range_constexpr(NUM_MSB):
            col_base_val = su * 32 + (msb % 2) * 16
            bnd = base - col_base_val
            v8w = Vec(su_sp_tiles[su][msb][0], dtype=fx.Float32)
            masked_elems = []
            for e in fx.range_constexpr(8):
                cmp = bnd < e
                elem = v8w[e]
                mval = cmp.select(neg_inf, elem)
                masked_elems.append(mval)
            su_sp_tiles[su][msb][0] = Vec.from_elements(masked_elems, fx.Float32)
def _setup_tdm_descs(tdm_state, kv_key, ptr_tensor, offset, stride_seq,
                     stride_32, wave_id, lds_target, oob_dg1_override):
    is_k = kv_key == "k"
    _CFG = (1 << 16) | (K_TDM_CONFIG if is_k else V_TDM_CONFIG)
    _stride_elems = stride_seq >> 1
    if const_expr(oob_dg1_override is not None):
        dg1 = oob_dg1_override
    else:
        dim0 = QK_HDIM if is_k else 128
        dim0_stride = 200 if is_k else 128
        dg1 = Vec.from_elements(
            [fx.Int32(_CFG), fx.Int32(dim0 << 16), fx.Int32(8 << 16),
             fx.Int32(dim0_stride << 16), fx.Int32(8), _stride_elems,
             fx.Int32(0), fx.Int32(0)], fx.Int32)
    row_bytes = K_ROW_BYTES if is_k else V_ROW_BYTES
    su_p_size = LDS_K_SU_P_SIZE if is_k else LDS_V_SU_P_SIZE
    addr = compute_global_addr(ptr_tensor, offset, wave_id, 8 * stride_seq)
    stride_adv = fx.Int64(stride_32)
    warp_off = wave_id * (8 * row_bytes)
    lds_base = lds_target + warp_off
    descs = TDM.build_descs(dg1, addr, stride_adv, lds_base, su_p_size, CNT_SU)
    tdm_state[f"{kv_key}_descs"] = descs
    tdm_state[f"{kv_key}_desc_idx"] = 0
def fmha_pipeline_ctx(
    ctx,
    ty,
    memload,
    q_tiles,
    kv_tiles,
    sp_tiles,
    o_tiles,
    kv_lds_addrs,
    tdm_state,
    softmax_state,
    sgpr_state,
    gemm2=True,
    tdm_v_offset=None,
    tdm_k_offset=None,
    tdm_k_target=None,
    tdm_v_target=None,
    kv_lds_addrs_next=None,
    gemm1_tdm_is_v=False,
    ia_exp_delta=None,
    causal_n_start=None,
    endtile_v_oob_dg1=None,
    kv_oob_cols=None,
    loop_k_oob_dg1=None,
    loop_v_oob_dg1=None,
):
    stride_k_seq = ctx["stride_k_seq"]
    stride_v_seq = ctx["stride_v_seq"]
    stride_k_32 = ctx["stride_k_32"]
    stride_v_32 = ctx["stride_v_32"]
    ptr_K = ctx["ptr_K"]
    ptr_V = ctx["ptr_V"]
    wave_id = ctx["wave_id"]

    Atom.s_wait_dscnt(LDS_INST_COUNT // 2)

    v_tiles_out = None
    blk = 0

    sp_pairs_all = softmax_state.get("sp_pairs_prev", None)
    if const_expr(sp_pairs_all is None):
        sp_pairs_all = [[None] * N_SP_PAIRS for _ in range(NUM_MSB)]
    softmax_ops_by_msb = Softmax.build_all_part2_ops(
        ty, 0, sp_pairs_all, softmax_state, sgpr_state
    )
    softmax_idx_by_msb = [PART2_SPLIT] * NUM_MSB

    for m in fx.range_constexpr(NUM_MSB):
        for i in fx.range_constexpr(PART2_SETUP_A - 1):
            softmax_ops_by_msb[m][i]()
    has_tdm_k_g1 = (not gemm1_tdm_is_v) and (tdm_k_offset is not None)
    has_tdm_v_g1 = gemm1_tdm_is_v and (tdm_v_offset is not None)
    if has_tdm_k_g1:
        _setup_tdm_descs(tdm_state, "k", ptr_K, tdm_k_offset, stride_k_seq,
                         stride_k_32, wave_id, tdm_k_target, loop_k_oob_dg1)
    if has_tdm_v_g1:
        _setup_tdm_descs(tdm_state, "v", ptr_V, tdm_v_offset, stride_v_seq,
                         stride_v_32, wave_id, tdm_v_target, endtile_v_oob_dg1)
    g1_tdm_type = KV_K if has_tdm_k_g1 else KV_V if has_tdm_v_g1 else KV_NONE
    stage_configs = [
        (0, g1_tdm_type, KV_K, blk, 1),
        (1, g1_tdm_type, KV_K, blk, 2),
        (2, KV_NONE, KV_K, blk, 3),
        (3, KV_NONE, KV_V, blk, 0),
    ]

    su_sp_tiles_list = []

    ed_v8 = []
    for dm in fx.range_constexpr(NUM_MSB):
        ed_v8.append(fx.vector.broadcast(T.vec(8, T.f32), ia_exp_delta[dm]))
    o_rescale_by_stage = []
    for s in fx.range_constexpr(N_PV_WMMA_N):
        stage_closures = []
        for dm in fx.range_constexpr(NUM_MSB):
            def mk_rescale(dm=dm, nn=s, ev8=ed_v8[dm]):
                def op():
                    o_tiles[dm][nn] = o_tiles[dm][nn] * ev8
                return op
            stage_closures.append(mk_rescale())
        o_rescale_by_stage.append(stage_closures)
    for stage_idx, (g_su, t_type, l_type, l_blk, l_su) in enumerate(stage_configs):
        n_lds = N_LDS_V_PER_MSB if l_type == KV_V else N_LDS_PER_MSB
        kv_tiles_next_raw = [[None] * n_lds for _ in range(NUM_MSB)]
        bpm = ALU_PER_STAGE[(stage_idx + 4) % ALU_STAGES] // NUM_MSB
        is_barrier_stage = stage_idx == 2 and (has_tdm_k_g1 or has_tdm_v_g1)
        sp_tiles, kv_tiles_next_raw = gemm1_interleaved_stage(
            ty, stage_idx, blk, g_su, t_type, blk, g_su, l_type, l_blk, l_su,
            q_tiles, kv_tiles, sp_tiles, kv_lds_addrs, kv_tiles_next_raw,
            softmax_ops_by_msb, softmax_idx_by_msb, [bpm, bpm, 0, 0],
            tdm_state, tdm_barrier=is_barrier_stage,
            o_rescale_ops=o_rescale_by_stage[stage_idx],
        )
        su_sp_tiles_list.append([[sp_tiles[msb][0]] for msb in range(NUM_MSB)])
        if const_expr(l_type == KV_K):
            kv_tiles = Fragment.pair_k_tiles(kv_tiles_next_raw, ty)
        else:
            v_tiles_out = kv_tiles_next_raw
    if const_expr(not gemm2):
        return sp_tiles, kv_tiles, o_tiles, su_sp_tiles_list
    for msb in fx.range_constexpr(NUM_MSB):
        for op in softmax_ops_by_msb[msb][softmax_idx_by_msb[msb] :]:
            op()
    o_rescale_exp_delta = None

    if const_expr(causal_n_start is not None):
        apply_causal_mask(ctx, su_sp_tiles_list, causal_n_start)
    if const_expr(kv_oob_cols is not None):
        apply_kv_oob_mask(ctx, su_sp_tiles_list, kv_oob_cols)
    sp_pairs_current = Softmax.tiles_to_pairs(su_sp_tiles_list)

    (
        g2_ops_by_rid,
        _,
        g2_sp_lo_cache,
        g2_sp_hi_cache,
    ) = Softmax.build_all_gemm2_ops(
        ty,
        blk,
        sp_pairs_current,
        softmax_state,
        sgpr_state,
    )
    g2_rid_idx = [0] * RLTS_LEN

    p_tiles_computed = Softmax.build_p_tiles(ty, softmax_state)

    v_tiles_paired = Fragment.pair_v_tiles(v_tiles_out, ty)

    has_tdm_v_g2 = (not gemm1_tdm_is_v) and (tdm_v_offset is not None)
    if has_tdm_v_g2:
        _setup_tdm_descs(tdm_state, "v", ptr_V, tdm_v_offset, stride_v_seq,
                         stride_v_32, wave_id, tdm_v_target, loop_v_oob_dg1)
    g2_stage_configs = [
        (0, KV_V, blk, 1, KV_V if has_tdm_v_g2 else KV_NONE, False),
        (1, KV_V, blk, 2, KV_V if has_tdm_v_g2 else KV_NONE, False),
        (2, KV_V, blk, 3, KV_NONE, has_tdm_v_g2),
        (3, KV_K, blk, 0, KV_NONE, False),
    ]

    for stage_idx, (g_su, l_type, l_blk, l_su, t_type, barrier) in enumerate(g2_stage_configs):
        n_lds = N_LDS_V_PER_MSB if l_type == KV_V else N_LDS_PER_MSB
        kv_tiles_next_raw = [[None] * n_lds for _ in range(NUM_MSB)]
        g2_addrs = (kv_lds_addrs_next if kv_lds_addrs_next is not None else kv_lds_addrs) if l_type == KV_K else kv_lds_addrs
        o_tiles, kv_tiles_next_raw = gemm2_interleaved_stage(
            ty, stage_idx, blk, g_su, l_type, l_blk, l_su,
            v_tiles_paired, p_tiles_computed[g_su], o_tiles, g2_addrs,
            kv_tiles_next_raw, g2_ops_by_rid, g2_rid_idx,
            tdm_state=tdm_state, tdm_type=t_type, tdm_barrier=barrier,
            o_rescale_exp_delta=(o_rescale_exp_delta if stage_idx == 0 else None),
        )
        if const_expr(l_type == KV_V):
            v_tiles_paired = Fragment.pair_v_tiles(kv_tiles_next_raw, ty)
        else:
            kv_tiles = Fragment.pair_k_tiles(kv_tiles_next_raw, ty)
    rocdl.sched_barrier(0)

    partial_sp_lo_out = []
    partial_sp_hi_out = []
    for psm in fx.range_constexpr(NUM_MSB):
        for psi in fx.range_constexpr(N_SP_PAIRS):
            lo_c = g2_sp_lo_cache[psm][psi]
            hi_c = g2_sp_hi_cache[psm][psi]
            if const_expr(lo_c is None or hi_c is None):
                pair = Vec(sp_pairs_current[psm][psi], dtype=fx.Float32)
            if const_expr(lo_c is None):
                lo_c = pair[0].ir_value()
            if const_expr(hi_c is None):
                hi_c = pair[1].ir_value()
            partial_sp_lo_out.append(lo_c)
            partial_sp_hi_out.append(hi_c)
    partial_ed_out = [
        softmax_state["exp_delta"][m] for m in fx.range_constexpr(NUM_MSB)
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
def tile_iteration(ctx, tile_idx, iter_args, causal_n_start=None):
    lane_id, wave_id = ctx["lane_id"], ctx["wave_id"]
    actual_kv_len = ctx["actual_kv_len"]
    stride_k_seq, stride_v_seq = ctx["stride_k_seq"], ctx["stride_v_seq"]
    k_offset, v_offset = ctx["k_offset"], ctx["v_offset"]
    tile_n_const, zero_v8f32 = ctx["tile_n_const"], ctx["zero_v8f32"]
    q_frags, sgpr_state, ty = ctx["q_frags"], ctx["sgpr_state"], ctx["ty"]
    o_tiles = [
        [set_vgpr_bank(iter_args[d * N_PV_WMMA_N + n], d) for n in fx.range_constexpr(N_PV_WMMA_N)]
        for d in fx.range_constexpr(NUM_MSB)
    ]
    ia_old_max = [set_vgpr_bank(iter_args[16 + i], i) for i in fx.range_constexpr(NUM_MSB)]
    ia_row_sums = [set_vgpr_bank(iter_args[20 + i], i) for i in fx.range_constexpr(NUM_MSB)]
    kv_tiles = [
        [set_vgpr_bank(iter_args[24 + msb * N_WMMA_K_TILES + k], msb) for k in fx.range_constexpr(N_WMMA_K_TILES)]
        for msb in fx.range_constexpr(NUM_MSB)
    ]
    ia_local_max = [set_vgpr_bank(iter_args[_OFF_LOCAL_MAX + i], i) for i in fx.range_constexpr(NUM_MSB)]
    ia_delta = [set_vgpr_bank(iter_args[_OFF_DELTA + i], i) for i in fx.range_constexpr(NUM_MSB)]
    ia_sp_flat = [iter_args[_OFF_SP + i] for i in fx.range_constexpr(CNT_SU * NUM_MSB)]
    prev_su_sp_tiles = [
        [[set_vgpr_bank(ia_sp_flat[su * NUM_MSB + msb], msb)] for msb in fx.range_constexpr(NUM_MSB)]
        for su in fx.range_constexpr(CNT_SU)
    ]
    ia_k_cur_base, ia_v_cur_base = iter_args[_OFF_PP], iter_args[_OFF_PP + 1]
    ia_k_next_base, ia_v_next_base = iter_args[_OFF_PP + 2], iter_args[_OFF_PP + 3]
    kv_lds_addrs_cur = build_kv_lds_addrs(lane_id, ia_k_cur_base, ia_v_cur_base)
    kv_lds_addrs_next = build_kv_lds_addrs(lane_id, ia_k_next_base, ia_v_next_base)
    ia_partial_sp_lo = [iter_args[_OFF_PSP + i] for i in fx.range_constexpr(_PSP_SIZE)]
    ia_partial_sp_hi = [iter_args[_OFF_PSP_HI + i] for i in fx.range_constexpr(_PSP_SIZE)]
    ia_exp_delta = [set_vgpr_bank(iter_args[_OFF_PED + i], i) for i in fx.range_constexpr(NUM_MSB)]
    ia_partial_sp_pairs = [
        [make_v2f32(ia_partial_sp_lo[m * N_SP_PAIRS + i], ia_partial_sp_hi[m * N_SP_PAIRS + i], m)
         for i in fx.range_constexpr(N_SP_PAIRS)]
        for m in fx.range_constexpr(NUM_MSB)
    ]
    sp_tiles = [[set_vgpr_bank(zero_v8f32, msb)] for msb in fx.range_constexpr(NUM_MSB)]
    softmax_state = make_softmax_state(ia_old_max, ia_local_max, ia_delta, ia_row_sums,
                                       sp_pairs_prev=ia_partial_sp_pairs)
    tdm_state = {
        "v_g0": fx.constant_vector(0, T.vec(4, T.i32)),
        "v_g1": fx.constant_vector(0, T.vec(8, T.i32)),
        "k_g0": fx.constant_vector(0, T.vec(4, T.i32)),
        "k_g1": fx.constant_vector(0, T.vec(8, T.i32)),
        "v_salu_queue": [], "k_salu_queue": [],
    }
    tile_idx_i32 = arith.index_cast(T.i32, tile_idx)
    cur_v_offset = v_offset + tile_idx_i32 * (tile_n_const * stride_v_seq)
    next_tile = tile_idx_i32 + 1
    next_k_offset = k_offset + next_tile * (tile_n_const * stride_k_seq)
    _loop_stride_k_elems = stride_k_seq >> 1
    _loop_stride_v_elems = stride_v_seq >> 1
    _loop_K_CFG_OOB = (1 << 16) | K_TDM_CONFIG
    _loop_V_CFG_OOB = (1 << 16) | V_TDM_CONFIG
    loop_v_oob_dg1 = TDM.build_oob_dg1_list(
        _loop_V_CFG_OOB, 128, _loop_stride_v_elems,
        actual_kv_len - tile_idx_i32 * tile_n_const, wave_id)
    loop_k_oob_dg1 = TDM.build_oob_dg1_list(
        _loop_K_CFG_OOB, QK_HDIM, _loop_stride_k_elems,
        actual_kv_len - next_tile * tile_n_const, wave_id, dim0_stride=200)
    (sp_out, kv_out, o_tiles, su_sp_tiles_out,
     partial_sp_lo_out, partial_sp_hi_out, partial_ed_out,
    ) = fmha_pipeline_ctx(
        ctx, ty, False, q_frags, kv_tiles, sp_tiles, o_tiles,
        kv_lds_addrs_cur, tdm_state, softmax_state, sgpr_state,
        gemm2=True, tdm_v_offset=cur_v_offset, tdm_v_target=ia_v_next_base,
        tdm_k_offset=next_k_offset, tdm_k_target=ia_k_next_base,
        kv_lds_addrs_next=kv_lds_addrs_next, gemm1_tdm_is_v=False,
        ia_exp_delta=ia_exp_delta, causal_n_start=causal_n_start,
        loop_k_oob_dg1=loop_k_oob_dg1, loop_v_oob_dg1=loop_v_oob_dg1,
    )
    new_o = [o_tiles[d][n] for d in fx.range_constexpr(NUM_MSB) for n in fx.range_constexpr(N_PV_WMMA_N)]
    new_max = [softmax_state["old_max"][i] for i in fx.range_constexpr(NUM_MSB)]
    new_sums = [softmax_state["row_sums"][i] for i in fx.range_constexpr(NUM_MSB)]
    kv_out_flat = [kv_out[msb][k] for msb in fx.range_constexpr(NUM_MSB) for k in fx.range_constexpr(N_WMMA_K_TILES)]
    new_local_max = [softmax_state["local_max"][i] for i in fx.range_constexpr(NUM_MSB)]
    new_delta = [softmax_state["delta"][i] for i in fx.range_constexpr(NUM_MSB)]
    sp_out_flat = [su_sp_tiles_out[su][msb][0] for su in fx.range_constexpr(CNT_SU) for msb in fx.range_constexpr(NUM_MSB)]
    pp_swapped = [ia_k_next_base, ia_v_next_base, ia_k_cur_base, ia_v_cur_base]
    new_exp_delta = [partial_ed_out[m] for m in fx.range_constexpr(NUM_MSB)]
    return (new_o + new_max + new_sums + kv_out_flat + new_local_max + new_delta
            + sp_out_flat + pp_swapped + partial_sp_lo_out + partial_sp_hi_out + new_exp_delta)
def _ep_finish(
    ctx, o_tiles, sp_pairs_in, exp_delta_rescale, v_base_for_pv,
    old_max_in, local_max_in, delta_in, row_sums_in, ep_k_cur_base,
):
    ty, sgpr_state, lane_id = ctx["ty"], ctx["sgpr_state"], ctx["lane_id"]
    scalar_f, RETURN_LSE = ctx["scalar_f"], ctx["RETURN_LSE"]
    ptr_LSE, bx, actual_q_len = ctx["ptr_LSE"], ctx["bx"], ctx["actual_q_len"]
    q_start_tok, gdz, by = ctx["q_start_tok"], ctx["gdz"], ctx["by"]
    wave_id, o_oob_dim1 = ctx["wave_id"], ctx["o_oob_dim1"]
    stride_o_seq, stride_o_head, ptr_O = ctx["stride_o_seq"], ctx["stride_o_head"], ctx["ptr_O"]
    sfx = make_softmax_state(old_max_in, local_max_in, delta_in, row_sums_in,
                             sp_pairs_prev=sp_pairs_in)
    p2ops = Softmax.build_all_part2_ops(ty, 0, sp_pairs_in, sfx, sgpr_state)
    for m in fx.range_constexpr(NUM_MSB):
        for i in fx.range_constexpr(PART2_SETUP_A - 1):
            p2ops[m][i]()
    for m in fx.range_constexpr(NUM_MSB):
        for op in p2ops[m][PART2_SPLIT:]:
            op()
    p_tiles = Softmax.build_p_tiles(ty, sfx)
    for msb in fx.range_constexpr(NUM_MSB):
        edv8 = fx.vector.broadcast(T.vec(8, T.f32), exp_delta_rescale[msb])
        for n in fx.range_constexpr(N_PV_WMMA_N):
            o_tiles[msb][n] = o_tiles[msb][n] * edv8
    kv_pv = build_kv_lds_addrs(lane_id, ep_k_cur_base, v_base_for_pv)
    for sp in fx.range_constexpr(2):
        sb = sp * 2
        vr0, vr1 = Fragment.load_v_two_sus(ty, kv_pv, 0, sb, sb + 1)
        o_tiles = pv_gemm_pure(ty, 0, sb, Fragment.pair_v_tiles(vr0, ty), p_tiles[sb], o_tiles)
        o_tiles = pv_gemm_pure(ty, 0, sb + 1, Fragment.pair_v_tiles(vr1, ty), p_tiles[sb + 1], o_tiles)
    v8bf16 = T.vec(8, T.bf16)
    rsf = list(sfx["row_sums"])
    lmf = list(sfx["local_max"])
    slo = arith.constant(0x76543210, type=T.i32)
    shi = arith.constant(0xFEDCBA98, type=T.i32)
    for mb in fx.range_constexpr(0, NUM_MSB, 2):
        sm = rsf[mb] + rsf[mb + 1]
        pm = rocdl_permlanex16(ty["f32"], sm, sm, slo, shi, False, False)
        sf = sm + pm
        rsf[mb] = sf
        rsf[mb + 1] = sf
    l2e = 0.6931471805599453
    lse_vals = [None] * NUM_MSB
    for msb in fx.range_constexpr(NUM_MSB):
        lse_vals[msb] = rocdl.log(ty["f32"], rsf[msb]) * l2e + lmf[msb] * scalar_f
    if const_expr(RETURN_LSE):
        glbpt_lse = glb_ptr_ty()
        lse_base_i64 = ptr_base_i64(ptr_LSE)
        wv_lse = rocdl.wave_id()
        lane_lo_lse = lane_id & 15
        lse_base_row = bx * 128 + wv_lse * WV_SUBQD
        for msb_lse in [0, 2]:
            msb_off = 0 if msb_lse == 0 else 16
            seq_pos = lse_base_row + lane_lo_lse + msb_off
            lse_valid = seq_pos < actual_q_len
            _if_lse = scf.IfOp(lse_valid.ir_value())
            with ir.InsertionPoint(_if_lse.then_block):
                lse_tok = q_start_tok + seq_pos
                lse_addr = ptr_base_i64(ptr_LSE) + fx.Int64((lse_tok * gdz + by) * 4)
                llvm_dialect.store(lse_vals[msb_lse], llvm_dialect.inttoptr(glbpt_lse, lse_addr))
                scf.YieldOp([])
    obf16 = []
    for msb in fx.range_constexpr(NUM_MSB):
        rcp = rocdl.rcp(ty["f32"], rsf[msb])
        rv8 = fx.vector.broadcast(T.vec(8, T.f32), rcp)
        obf16.append([fx.trunc_f(v8bf16, o_tiles[msb][n] * rv8) for n in fx.range_constexpr(N_PV_WMMA_N)])
    rocdl.s_barrier_signal(-1)
    rocdl.s_barrier_wait(-1)
    ldst = lds_ptr_ty()
    v4i32t = T.vec(4, T.i32)
    db32 = extract_lds_base_i32(lds_alloc_v_a.get_base())
    dw = db32 + wave_id * LDS_D_WV_SIZE
    d_crd = idx2crd(lane_id, fx.make_layout((16, 2), (1, 16)))
    llo = arith.index_cast(T.i32, d_crd[0])
    lhi = arith.index_cast(T.i32, d_crd[1])
    loff = llo * TDM_D_TILE_DIM0 + lhi * 16
    for msb in fx.range_constexpr(NUM_MSB):
        for n in fx.range_constexpr(N_PV_WMMA_N):
            ioff = (msb // 2) * 16 * TDM_D_TILE_DIM0 + (msb % 2) * 128 + n * 32
            la = dw + loff + ioff
            llvm_dialect.store(fx.vector.bitcast(v4i32t, obf16[msb][n]),
                               llvm_dialect.inttoptr(ldst, la), volatile_=True)
    emit_void("s_wait_dscnt 0x0")
    wsgpr = rocdl.wave_id()
    o_tok = q_start_tok + bx * 128 + wsgpr * WV_SUBQD
    o_elem_off = by * stride_o_head + o_tok * stride_o_seq
    oadr64 = ptr_base_i64(ptr_O) + fx.Int64(o_elem_off * 2)
    alo, ahi = split_i64_to_lo_hi(oadr64)
    olds2 = extract_lds_base_i32(lds_alloc_v_a.get_base()) + wsgpr * LDS_D_WV_SIZE
    _dg0 = Vec.from_elements([fx.Int32(1), olds2, alo, ahi], fx.Int32)
    _td1_lo_o = arith.andi(o_oob_dim1, arith.constant(0xFFFF, type=T.i32))
    _g2 = arith.ori(arith.shli(_td1_lo_o, arith.constant(16, type=T.i32)),
                     arith.constant(0, type=T.i32))
    _dg1 = Vec.from_elements(
        [fx.Int32(1 << 16), fx.Int32(128 << 16), _g2,
         fx.Int32(128 << 16), fx.Int32(32), stride_o_seq,
         fx.Int32(0), fx.Int32(0)], fx.Int32)
    tdm_ops.tensor_store_2d(tdm_ops.TDMDescriptor2D(_dg0, _dg1))
    tdm_ops.tensor_wait(0)
