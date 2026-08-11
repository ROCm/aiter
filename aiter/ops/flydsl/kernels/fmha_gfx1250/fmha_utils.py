"""
Forward Kernel Utilities -- gfx1250, FMHA Schedule Tables & Namespace Classes.

Contains:
  - Schedule tables and helpers (GEMM1/GEMM2 schedules)
  - Core loop constants and low-level helpers
  - Namespace classes: Atom, Softmax, Fragment, Pipeline

Used by fmha_kernel.py which implements the top-level kernel
(prologue, core loop, epilogue, public API).
"""

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


# SECTION 1: Schedule Tables

# Schedule token IDs (used by schedule helpers and dispatch engine)
P1 = 4  # PART1 cross-MSB merge
O_RESC0 = 17  # O-rescale pk_mul
TDM_TOKEN = 18  # tensor_load_to_lds
PART2_EXP_START = 24  # ops[0..23] = setup+pkfma (cheap), ops[24..] = exp

# Token base offsets for per-MSB helpers: token_id = BASE + msb
P2_BASE = 5  # P2_M0..P2_M3 = 5..8 (softmax PART2)
EXP_BASE = 19  # EXP_M0..EXP_M3 = 19..22 (pair_exp, 3cy)
K_BASE = 9  # K_M0..K_M3 = 9..12 (ds_load_b128)
V_BASE = 13  # V_M0..V_M3 = 13..16 (ds_load_tr16_b128)

# Schedule dimensions


# GEMM1 schedule: 96 rows
# Budget derivation (ALU_PER_STAGE = [40,52,56,168, 120,120,132,132]):
#   GEMM1_EXP_OPS=24, cheap=33 per MSB
#   Stage 0 (softmax_stage 4, budget=120): 10 exp/MSB, 0 cheap
#   Stage 1 (softmax_stage 5, budget=120): 10 exp/MSB, 0 cheap
#   Stage 2 (softmax_stage 6, budget=132):  4 exp/MSB, 21 cheap
#   Stage 3 (softmax_stage 7, budget=132):  0 exp/MSB, 33 cheap
# Stages 0,1 have TDM K loads (main loop). Stage 2 has K loads, no TDM.
# Stage 3 has V tile loads (lds_type='V'), no TDM.


# Schedule row helpers: generate n copies of a token for MSB bank m.
# Each helper name describes the hardware operation it schedules.
def lds_k(m, n=1):
    return [K_BASE + m] * n  # ds_load_b128 K tile


def lds_v(m, n=1):
    return [V_BASE + m] * n  # ds_load_tr16_b128 V tile


def tree_max(m, n=1):
    return [m] * n  # PART0: per-bank max reduction


def cross_max(n=1):
    return [P1] * n  # PART1: cross-MSB max merge + delta


def sm_ops(m, n=1):
    return [P2_BASE + m] * n  # PART2: rescale / pkfma / cvt / sum


def pair_exp(m, n=1):
    return [EXP_BASE + m] * n  # PART2 exp2 (3-cycle transcendental)


def o_rescale(n=1):
    return [O_RESC0] * n  # O *= exp_delta (pk_mul)


def tdm_load(n=1):
    return [TDM_TOKEN] * n  # tensor_load_to_lds (DMA prefetch)


def build_gemm1_schedule() -> list[list[int]]:
    """GEMM1 (QK) interleaved schedule: 96 rows = 4 stages × 24 WMMAs.

    Each row lists ops to issue between two consecutive WMMA instructions.
    Ops within a row execute in order; same-MSB grouping minimizes bank switches.

    Pipeline: GEMM1 computes Q×K^T while interleaving:
      - K/V tile LDS loads (ds_load_b128 / ds_load_tr16_b128)
      - Softmax PART2 (rescale, pkfma, exp, cvt, sum-tree)
      - O-rescale (O *= exp_delta)
      - TDM prefetch (next K tile Global→LDS)
    """
    # ── Stage 0: K loads + TDM prefetch, 10 exp/MSB ──────────────────────
    # w00:      TDM issues 2 Global→LDS copies; 1 softmax op fills the gap
    # w01-w08:  K loads grouped by MSB (3 loads/WMMA); softmax same-bank
    # w09-w23:  K done → pure softmax drain + 4 O-rescale scattered
    s0 = [
        tdm_load(2) + sm_ops(0),  # w00
        lds_k(0, 3) + o_rescale(),  # w01
        lds_k(0, 3) + sm_ops(0, 2),  # w02
        lds_k(1, 3) + sm_ops(1, 2),
        lds_k(1, 3) + sm_ops(1, 2),  # w03-04
        lds_k(2, 3) + sm_ops(2, 2),
        lds_k(2, 3) + sm_ops(2) + sm_ops(0),  # w05-06
        lds_k(3, 3) + sm_ops(3, 2),
        lds_k(3, 3) + sm_ops(3),  # w07-08
        sm_ops(0, 2),  # w09
        sm_ops(0, 2),
        sm_ops(1, 2),
        sm_ops(1, 2),  # w10-12
        sm_ops(1) + o_rescale(),  # w13
        sm_ops(1) + sm_ops(0),
        sm_ops(2) + sm_ops(3),  # w14-15
        sm_ops(2) + o_rescale(),
        sm_ops(2) + o_rescale(),  # w16-17
        sm_ops(0, 2),
        sm_ops(0, 2),
        sm_ops(0, 2),
        sm_ops(0, 2),
        sm_ops(0, 2),  # w18-22
        sm_ops(0) + sm_ops(2),  # w23
    ]

    # ── Stage 1: K loads + TDM prefetch, 10 exp/MSB ──────────────────────
    # Same structure as stage 0 with different MSB distribution
    s1 = [
        tdm_load(2) + sm_ops(2),  # w00
        lds_k(0, 3) + o_rescale(),
        lds_k(0, 3) + sm_ops(0),  # w01-02
        lds_k(1, 3) + sm_ops(1, 2),
        lds_k(1, 3) + sm_ops(1, 2),  # w03-04
        lds_k(2, 3) + sm_ops(2, 2),
        lds_k(2, 3) + sm_ops(2, 2),  # w05-06
        lds_k(3, 3) + sm_ops(3),
        lds_k(3, 3) + sm_ops(3),  # w07-08
        sm_ops(3, 2),
        sm_ops(3, 2),
        sm_ops(3, 2),  # w09-11
        sm_ops(1, 2),
        sm_ops(1, 2),  # w12-13
        sm_ops(1) + sm_ops(2),
        sm_ops(1) + o_rescale(),  # w14-15
        sm_ops(2) + o_rescale(),
        sm_ops(2) + o_rescale(),  # w16-17
        sm_ops(2, 2),
        sm_ops(2, 2),  # w18-19
        sm_ops(3, 2),
        sm_ops(3, 2),
        sm_ops(3, 2),  # w20-22
        sm_ops(3) + sm_ops(0),  # w23
    ]

    # ── Stage 2: K loads only (no TDM), 4 exp + 21 cheap/MSB ─────────────
    # Larger softmax budget (132 cy) → bulk cheap ops fill w08-w23
    s2 = [
        lds_k(0, 3) + sm_ops(0) + sm_ops(3),  # w00
        lds_k(0, 3) + sm_ops(0) + sm_ops(2),  # w01
        lds_k(1, 3) + sm_ops(1, 2),
        lds_k(1, 3) + sm_ops(1, 2),  # w02-03
        lds_k(2, 3) + sm_ops(2, 2),
        lds_k(2, 3) + sm_ops(2, 2),  # w04-05
        lds_k(3, 3) + sm_ops(3, 2),
        lds_k(3, 3) + sm_ops(3, 2),  # w06-07
        sm_ops(0, 6),  # w08
        sm_ops(0, 7),
        sm_ops(0, 7),  # w09-10
        sm_ops(1, 7),
        sm_ops(1, 7),
        sm_ops(1, 7),  # w11-13
        sm_ops(2, 2) + o_rescale(),  # w14
        sm_ops(2, 2) + o_rescale(),
        sm_ops(2, 2) + o_rescale(),  # w15-16
        sm_ops(3, 2) + o_rescale(),  # w17
        sm_ops(2, 6),
        sm_ops(2, 6),  # w18-19
        sm_ops(2, 3) + sm_ops(3, 3),  # w20
        sm_ops(3, 6),
        sm_ops(3, 6),  # w21-22
        sm_ops(0, 4) + sm_ops(3, 2),  # w23
    ]

    # ── Stage 3: V loads (not K), 33 cheap/MSB ───────────────────────────
    # V uses ds_load_tr16_b128 (transposed); 2 loads/WMMA instead of 3
    s3 = [
        lds_v(0, 2) + sm_ops(0, 2),
        lds_v(0, 2) + sm_ops(0, 2),  # w00-01
        lds_v(1, 2) + sm_ops(1, 2),
        lds_v(1, 2) + sm_ops(1, 2) + sm_ops(2),  # w02-03
        lds_v(2, 2) + sm_ops(2, 2),
        lds_v(2, 2) + sm_ops(2, 2),  # w04-05
        lds_v(3, 2) + sm_ops(3, 2),
        lds_v(3, 2) + sm_ops(3) + sm_ops(1),  # w06-07
        sm_ops(1, 2) + sm_ops(3),  # w08
        sm_ops(1) + sm_ops(0),
        sm_ops(1) + sm_ops(3),  # w09-10
        sm_ops(1) + sm_ops(0) + sm_ops(3),  # w11
        sm_ops(2) + sm_ops(3),
        sm_ops(2, 2),  # w12-13
        sm_ops(3, 2),
        sm_ops(3, 2) + sm_ops(1),  # w14-15
        o_rescale() + sm_ops(2),
        o_rescale() + sm_ops(2),  # w16-17
        o_rescale() + sm_ops(2) + sm_ops(0),  # w18
        o_rescale() + sm_ops(2) + sm_ops(3),  # w19
        sm_ops(0) + sm_ops(3) + sm_ops(1),  # w20
        sm_ops(0),
        sm_ops(3),
        [],  # w21-23
    ]

    sched = s0 + s1 + s2 + s3
    assert len(sched) == 96
    return sched


# GEMM2 schedule: 64 rows
# Strict ordering: P0 (max) → P1 (cross-MSB delta) → P2 (pkfma) → EXP (pair_exp)
#
# Key dependency: PART2 setup op[2] reads delta[b] = PART1 output.
# → P1 must finish BEFORE any P2 token fires.
# → P1 placed in stage 1 post-load (after all P0 overflow), ensuring
#   PART1 completes before stage 2 where P2 companions start. ✓
#
# Stage 0 (V+TDM): P0=10/MSB
# Stage 1 (V+TDM): P0=12/MSB + P1=8 (P1 after all P0 overflow in post-load)
# Stage 2 (V):     ALL P2=24/MSB (load-WMMAs give 8/MSB; post-load 64 tokens, row_cap=8)
# Stage 3 (K):     ALL EXP=8/MSB at 2/WMMA
# Totals: P0=22/MSB ✓  P1=8 ✓  P2=24/MSB ✓  EXP=8/MSB=32 total ✓


def build_gemm2_schedule() -> list[list[int]]:
    """GEMM2 (PV) interleaved schedule: 64 rows = 4 stages x 16 WMMAs.

    Pipeline: GEMM2 computes P*V while interleaving softmax for the NEXT tile:
      Stage 0-1: PART0 (per-bank tree-max) + V loads + TDM
      Stage 1->2: PART1 (cross-MSB merge) then PART2 (pkfma/rescale)
      Stage 3:   pair_exp (3-cycle transcendental) + K prefetch

    Strict ordering: P0 -> P1 -> P2 -> EXP
    Key dep: PART2 setup reads delta = PART1 output -> P1 finishes in stage 1.
    Totals: P0=22/MSB  P1=8  P2=24/MSB  EXP=8/MSB=32
    """
    # -- Stage 0: V+TDM, PART0 tree-max 10/MSB --
    # V loads paired with same-MSB tree-max ops; overflow fills w09-w15
    s0 = [
        tdm_load(2) + tree_max(0, 3),  # w00
        lds_v(0, 2) + tree_max(0, 3),
        lds_v(0, 2) + tree_max(0, 3),  # w01-02
        lds_v(1, 2) + tree_max(1, 4),
        lds_v(1, 2) + tree_max(1, 4),  # w03-04
        lds_v(2, 2) + tree_max(2, 4),
        lds_v(2, 2) + tree_max(2, 4),  # w05-06
        lds_v(3, 2) + tree_max(3, 4),
        lds_v(3, 2) + tree_max(3, 4),  # w07-08
        tree_max(0) + tree_max(2, 3) + tree_max(3),  # w09
        tree_max(0) + tree_max(1, 2) + tree_max(3),  # w10
        tree_max(0, 4) + tree_max(1),  # w11
        tree_max(0) + tree_max(1, 2) + tree_max(2, 2),  # w12
        tree_max(1) + tree_max(2, 2) + tree_max(3, 2),  # w13
        tree_max(2) + tree_max(3, 2) + tree_max(1, 2),  # w14
        tree_max(3, 2),  # w15
    ]

    # -- Stage 1: V+TDM, PART0 12/MSB + PART1 8 + PART2 start --
    # P0 interleaved across MSBs; P1 at w09-10; P2 seeds w11-15
    s1 = [
        tdm_load(2) + tree_max(0),  # w00
        lds_v(0, 2) + tree_max(0) + tree_max(1) + tree_max(2) + tree_max(3),  # w01
        lds_v(0, 2) + tree_max(0) + tree_max(1) + tree_max(2) + tree_max(3),  # w02
        lds_v(1, 2) + tree_max(1) + tree_max(0) + tree_max(2) + tree_max(3),  # w03
        lds_v(1, 2) + tree_max(1) + tree_max(0),  # w04
        lds_v(2, 2) + tree_max(2, 2) + tree_max(1),  # w05
        lds_v(2, 2) + tree_max(2) + tree_max(0) + tree_max(1),  # w06
        lds_v(3, 2) + tree_max(3),
        lds_v(3, 2) + tree_max(3, 2),  # w07-08
        cross_max(4),
        cross_max(4),  # w09-10: P1 merge
        sm_ops(1, 4) + sm_ops(2, 4),
        sm_ops(0, 4) + sm_ops(3, 4),  # w11-12: P2 start
        sm_ops(0) + sm_ops(1) + sm_ops(2) + sm_ops(3),  # w13
        sm_ops(0)
        + sm_ops(1)
        + sm_ops(2)
        + sm_ops(3)  # w14
        + sm_ops(0)
        + sm_ops(1)
        + sm_ops(2)
        + sm_ops(3),
        sm_ops(0) + sm_ops(1) + sm_ops(2) + sm_ops(3),  # w15
    ]

    # -- Stage 2: V only, PART2 bulk 24/MSB --
    # V loads + pkfma at w00-07; pure softmax drain at w08-15
    s2 = [
        lds_v(0, 2) + sm_ops(0, 3),
        lds_v(0, 2) + sm_ops(0, 3),  # w00-01
        lds_v(1, 2) + sm_ops(1) + sm_ops(2, 2),  # w02
        lds_v(1, 2) + sm_ops(1) + sm_ops(2, 2),  # w03
        lds_v(2, 2) + sm_ops(2) + sm_ops(1, 2),  # w04
        lds_v(2, 2) + sm_ops(2) + sm_ops(1, 2),  # w05
        lds_v(3, 2) + sm_ops(3, 4),
        lds_v(3, 2) + sm_ops(3, 3),  # w06-07
        sm_ops(0, 6),
        sm_ops(0, 3) + sm_ops(1),  # w08-09
        sm_ops(1, 6),
        sm_ops(1, 3) + sm_ops(2, 2),  # w10-11
        sm_ops(2, 6),
        sm_ops(3, 3),  # w12-13
        sm_ops(3, 4),
        sm_ops(0, 2) + sm_ops(1, 2),  # w14-15
    ]

    # -- Stage 3: K prefetch, pair_exp 8/MSB (3cy each) --
    # K loads for NEXT tile; pair_exp <=2/WMMA to stay within cycle budget
    s3 = [
        lds_k(0, 3) + pair_exp(0, 2),
        lds_k(0, 3) + pair_exp(0),  # w00-01
        lds_k(1, 3) + pair_exp(1, 2),
        lds_k(1, 3) + pair_exp(1, 2),  # w02-03
        lds_k(2, 3) + sm_ops(2) + pair_exp(2),  # w04
        lds_k(2, 3) + sm_ops(2) + pair_exp(2),  # w05
        lds_k(3, 3) + sm_ops(3) + pair_exp(3),  # w06
        lds_k(3, 3) + sm_ops(3) + pair_exp(3),  # w07
        pair_exp(0, 3),
        pair_exp(0, 2) + pair_exp(1),  # w08-09
        pair_exp(1, 3),
        pair_exp(3, 3),  # w10-11
        pair_exp(2, 3),
        pair_exp(2, 3),  # w12-13
        pair_exp(3, 3),
        [],  # w14-15
    ]

    sched = s0 + s1 + s2 + s3
    assert len(sched) == 64
    return sched


# Pre-built tables
GEMM1_SCHEDULE: list[list[int]] = build_gemm1_schedule()
GEMM2_SCHEDULE: list[list[int]] = build_gemm2_schedule()


# Index helpers
def g1_row_idx(stage: int, wmma: int) -> int:
    return stage * GEMM_INST_COUNT + wmma


def g2_row_idx(stage: int, wmma: int) -> int:
    return stage * PV_GEMM_INST_COUNT + wmma


# SECTION 2: Core Loop


# Local rocdl primitive wrappers (until merged into upstream FlyDSL)


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


# Hardware / tile geometry
WAVE_SIZE = 32
NUM_WAVES = 4
NUM_MSB = 4  # VGPR banks
BLOCK_SIZE = WAVE_SIZE * NUM_WAVES  # 128 threads per workgroup

QK_HDIM = 192  # Q/K head dimension (bf16 elements)
V_HDIM = 128  # V head dimension
Q_BPP = 2  # bytes per element (bf16)
KV_BPP = 2
TG_SUBQD = 128  # tile_m (Q rows per workgroup)
TG_SUBKV = 128  # tile_n (KV cols per tile)
WV_SUBKV = TG_SUBKV
SU_K_N = 32  # rows per SU (sub-unit)
SU_K_K = QK_HDIM
CNT_SU = WV_SUBKV // SU_K_N  # 4 SUs per tile

# WMMA instruction shape
WMMA_M, WMMA_N, WMMA_K = 16, 16, 32

# VGPR register file layout (per-MSB)
VPS_Q = TG_SUBQD * QK_HDIM * Q_BPP // 128  # 64
VPS_MSB_Q = VPS_Q // NUM_MSB  # 16
VTS_MSB_Q = VPS_MSB_Q  # 16
Q_WMMA_PER_MSB = VTS_MSB_Q // 8  # 2
VPS_KV = SU_K_K * SU_K_N * KV_BPP // 128  # 128
VPS_MSB_KV = VPS_KV // NUM_MSB  # 32
VPS_MSB_SP = 32  # 4 SUs × 8 f32/lane per MSB
SP_MSB_M = 16
SP_MSB_N = 16
SP_MSB_K = QK_HDIM

# GEMM instruction counts
GEMM_INST_COUNT = 24  # WMMAs per GEMM1 stage
LDS_INST_COUNT = 24  # K LDS loads per stage (4 MSBs × 6)
N_LDS_PER_MSB = VPS_MSB_KV // 4  # 6 K loads per MSB
N_LDS_V_PER_MSB = 4  # V loads per MSB
LDS_V_INST_COUNT = NUM_MSB * N_LDS_V_PER_MSB  # 16

# Pipeline control
ALU_STAGES = 8
GEMM1 = 0  # QK gemm index
KV_K = 0
KV_V = 1
KV_NONE = 2
BARRIER_SIGNAL_AHEAD = 0
TDM_LOADS_PER_STAGE = 2

# LDS layout (byte offsets)
LDS_K_SU_P_SIZE = 0x3200  # per-SU K region: 12800 bytes
LDS_V_SU_P_SIZE = 0x2400  # per-SU V region: 9216 bytes

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


# Low-level Helpers


def emit_void(inst_str, operands=None, constraints="", **kwargs):
    llvm_dialect.inline_asm(
        None, operands or [], inst_str, constraints, has_side_effects=True, **kwargs
    )


def sched_barrier(mask=0):
    """Emit llvm.amdgcn.sched.barrier.

    Prevents instruction reordering across this point.
    mask=0: barrier for ALL instruction types
    (VMEM, VALU, LDS, SALU, etc.).
    """
    mask_val = arith.constant(mask, type=T.i32)
    llvm_dialect.call_intrinsic(None, "llvm.amdgcn.sched.barrier", [mask_val], [], [])


# VGPR Bank Hint

USE_BANK_HINTS = False

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


# MLIR Types


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


# V2F32 Helpers (from epilogue)


def make_v2f32(lo, hi, bank=0):
    v = Vec.from_elements([lo, hi], Float32)
    return set_vgpr_bank(v, bank)


def split_v2f32(pair):
    lo = Vec(pair, dtype=Float32)[0].ir_value()
    hi = Vec(pair, dtype=Float32)[1].ir_value()
    return lo, hi


def broadcast_f32_to_v2f32(val, bank=0):
    return make_v2f32(val, val, bank)


# WMMA Fragment Pairing (from prologue — concat two v4i32 → v16bf16)


N_WMMA_K_TILES = (QK_HDIM // WMMA_K) // 2  # 3 for QK_HDIM=192, 2 for 128


# Atomic Instruction Primitives
# Each emits exactly ONE instruction (scheduling controlled at stage level).

# --- WMMA ---


class Atom:
    """Atomic GPU instruction primitives. Each method emits exactly one
    instruction, fenced by sched_barrier(0) to prevent reordering."""

    @staticmethod
    def wmma_init(ty, src_a, src_b, bank_dst):
        sched_barrier(0)
        zero = fx.constant_vector(0.0, T.vec(8, T.f32))
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
        sched_barrier(0)
        return banked

    @staticmethod
    def wmma_accum(ty, src_a, src_b, acc, bank_dst):
        sched_barrier(0)
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
        sched_barrier(0)
        return banked

    @staticmethod
    def ds_load_b128(ty, addr, offset_val, bank):
        sched_barrier(0)
        ptr = llvm_dialect.inttoptr(ty["lds_ptr"], (addr + offset_val))
        raw = llvm_dialect.load(ty["v4i32"], ptr)
        banked = set_vgpr_bank(raw, bank)
        sched_barrier(0)
        return banked

    @staticmethod
    def ds_load_tr16_b128(ty, addr, offset_val, bank):
        sched_barrier(0)
        ptr = llvm_dialect.inttoptr(ty["lds_ptr"], (addr + offset_val))
        raw = rocdl.ds_load_tr16_b128(ty["v8bf16"], ptr)
        banked = set_vgpr_bank(raw, bank)
        sched_barrier(0)
        return banked

    @staticmethod
    def tdm_load(ty, s_g0, s_g1):
        sched_barrier(0)
        null_v4 = fx.constant_vector(0, T.vec(4, T.i32))
        null_v8 = fx.constant_vector(0, T.vec(8, T.i32))
        rocdl.tensor_load_to_lds(s_g0, s_g1, null_v4, null_v4, null_v8, 0)
        sched_barrier(0)

    # --- Scalar softmax VALU (f32) ---

    @staticmethod
    def exp_f32(src, bank):
        sched_barrier(0)
        result = rocdl_exp2(T.f32, src)
        banked = set_vgpr_bank(result, bank)
        sched_barrier(0)
        return banked

    @staticmethod
    def mul_f32(src0, src1, bank):
        sched_barrier(0)
        result = src0 * src1
        banked = set_vgpr_bank(result, bank)
        sched_barrier(0)
        return banked

    @staticmethod
    def fma_f32_neg_src0(src0, src1, src2, bank):
        sched_barrier(0)
        neg_src0 = llvm_dialect.fneg(src0)
        result = llvm_dialect.intr_fma(neg_src0, src1, src2)
        banked = set_vgpr_bank(result, bank)
        sched_barrier(0)
        return banked

    @staticmethod
    def mov_b32(src, bank):
        """v_mov_b32 — bank-change copy."""
        sched_barrier(0)
        banked = set_vgpr_bank(src, bank)
        sched_barrier(0)
        return banked

    @staticmethod
    def add_f32(src0, src1, bank):
        sched_barrier(0)
        r = src0 + src1
        banked = set_vgpr_bank(r, bank)
        sched_barrier(0)
        return banked

    @staticmethod
    def max3_num_f32(src0, src1, src2, bank):
        """v_max3_num_f32 via rocdl.fmax3."""
        sched_barrier(0)
        result = rocdl_fmax3(src0, src1, src2)
        banked = set_vgpr_bank(result, bank)
        sched_barrier(0)
        return banked

    @staticmethod
    def permlanex16(src, s_sel0, s_sel1, bank):
        sched_barrier(0)
        src_i32 = llvm_dialect.bitcast(T.i32, src)
        r = rocdl_permlanex16(T.i32, src_i32, src_i32, s_sel0, s_sel1, False, False)
        r_f32 = llvm_dialect.bitcast(T.f32, r)
        # bank+2: different bank from src to prevent in-place permlanex16
        banked = set_vgpr_bank(r_f32, (bank + 2) % NUM_MSB)
        sched_barrier(0)
        return banked

    # --- Packed softmax VALU (v2f32) ---

    @staticmethod
    def pk_fma_f32_neg_c(a, b, c, bank):
        """v_pk_fma_f32 with negated src2: a*b - c."""
        sched_barrier(0)
        neg_c = llvm_dialect.fneg(c)
        r = llvm_dialect.intr_fma(a, b, neg_c)
        banked = set_vgpr_bank(r, bank)
        sched_barrier(0)
        return banked

    @staticmethod
    def pk_add_f32(a, b, bank):
        """v_pk_add_f32 via arith.addf on v2f32."""
        sched_barrier(0)
        r = a + b
        banked = set_vgpr_bank(r, bank)
        sched_barrier(0)
        return banked

    @staticmethod
    def cvt_pk_bf16_f32(a, bank):
        """v_cvt_pk_bf16_f32 via arith.truncf v2f32 → v2bf16."""
        sched_barrier(0)
        r = arith.truncf(T.vec(2, T.bf16), a)
        banked = set_vgpr_bank(r, bank)
        sched_barrier(0)
        return banked

    # --- Synchronization ---

    @staticmethod
    def s_wait_dscnt(cnt):
        sched_barrier(0)
        rocdl.s_wait_dscnt(cnt)
        sched_barrier(0)

    @staticmethod
    def s_wait_tensorcnt(cnt):
        sched_barrier(0)
        rocdl.s_wait_tensorcnt(cnt)
        sched_barrier(0)


# Schedule Builders (compile-time, no IR emitted)

# Slot Scheduling Tables


def tdm_wait_and_barrier():
    rocdl.s_wait_tensorcnt(0)
    rocdl.s_barrier_signal(-1)
    rocdl.s_barrier_wait(-1)


def make_softmax_state(old_max, local_max, delta, row_sums, sp_pairs_prev=None):
    return {
        "old_max": list(old_max),
        "local_max": list(local_max),
        "delta": list(delta),
        "exp_delta": [None] * NUM_MSB,
        "cur_max_log2e": [None] * NUM_MSB,
        "cur_max_log2e_1": [None] * NUM_MSB,
        "cur_max_log2e_scalar": [None] * NUM_MSB,
        "cur_max_log2e_dup": [None] * NUM_MSB,
        "vgpr_log2e_scl_pair": [None] * NUM_MSB,
        "exp_delta_dup": [None] * NUM_MSB,
        "row_sums": list(row_sums),
        "p_bf16": [[] for _ in range(NUM_MSB)],
        "sp_pairs_prev": sp_pairs_prev,
    }


class Softmax:
    """Softmax operation builders for the FMHA pipeline."""

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
            ss["old_max"][b] = Atom.mov_b32(ss["local_max"][b], b)
            sched_barrier(0)
            _scl = sgpr["s_log2e_scl"]
            v = Vec.from_elements([_scl, _scl], Float32)
            if const_expr(b > 0):
                ss["vgpr_log2e_scl_pair"][b] = set_vgpr_bank_offset(
                    v, b, LOG2E_PAIR_OFFSET
                )
            else:
                ss["vgpr_log2e_scl_pair"][b] = set_vgpr_bank(v, b)
            sched_barrier(0)

        ops.append(op_save_old_max)

        # 1. curMaxLog2eScl = local_max * s_log2e_scl
        def op_cur_max(b=bank):
            ss["cur_max_log2e"][b] = Atom.mul_f32(
                ss["local_max"][b], sgpr["s_log2e_scl"], b
            )

        ops.append(op_cur_max)

        # 2. exp_delta = exp(delta)  — transcendental, 4-cycle latency
        def op_exp_delta(b=bank):
            ss["exp_delta"][b] = Atom.exp_f32(ss["delta"][b], b)

        ops.append(op_exp_delta)

        # 3. curMaxLog2eScl_1 = local_max * s_log2e_scl (duplicate)
        def op_cur_max_1(b=bank):
            ss["cur_max_log2e_1"][b] = Atom.mul_f32(
                ss["local_max"][b], sgpr["s_log2e_scl"], b
            )

        ops.append(op_cur_max_1)

        # 4. curMaxLog2eScl = old_max * log2e_scl (scalar v_mul_f32)
        def op_mul_old_max(b=bank):
            ss["cur_max_log2e_scalar"][b] = Atom.mul_f32(
                ss["old_max"][b], sgpr["s_log2e_scl"], b
            )

        ops.append(op_mul_old_max)

        # 5. curMaxLog2eScl_dup = broadcast scalar → v2f32 (v_mov_b32)
        # Separated from op4 so scheduler can place distance between
        # the mul producer, the broadcast, and the pk_fma consumer.
        def op_broadcast_dup(b=bank):
            ss["cur_max_log2e_dup"][b] = broadcast_f32_to_v2f32(
                ss["cur_max_log2e_scalar"][b], b
            )

        ops.append(op_broadcast_dup)

        # 6. exp_delta_dup = exp_delta (duplicate via mov)
        def op_exp_delta_dup(b=bank):
            ss["exp_delta_dup"][b] = Atom.mov_b32(ss["exp_delta"][b], b)

        ops.append(op_exp_delta_dup)

        # 7. row_sums *= exp_delta
        if not skip_rescale_sum:

            def op_rescale_sum(b=bank):
                ss["row_sums"][b] = Atom.mul_f32(
                    ss["exp_delta"][b], ss["row_sums"][b], b
                )

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
                result = Atom.pk_fma_f32_neg_c(
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
                    lo, hi = split_v2f32(sp_pairs[pidx])
                    sched_barrier(0)
                    exp_lo = rocdl_exp2(T.f32, lo)
                    sched_barrier(0)
                    sp_pairs[pidx] = Vec.from_elements([exp_lo, hi], Float32)
                    if const_expr(_clo is not None):
                        _clo[pidx] = exp_lo  # standalone f32 scalar for yield

                ops.append(op_exp_lo)
            else:

                def op_exp_hi(pidx=_pidx, b=bank, _chi=sp_hi_cache):
                    lo, hi = split_v2f32(sp_pairs[pidx])
                    sched_barrier(0)
                    exp_hi = rocdl_exp2(T.f32, hi)
                    sched_barrier(0)
                    sp_pairs[pidx] = Vec.from_elements([lo, exp_hi], Float32)
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
                ss["p_bf16"][b].append(Atom.cvt_pk_bf16_f32(src, b))

            ops.append(op_cvt)

        # --- Phase C2: pk_add level-0 — 8 ops ---
        for i in range_constexpr(N_SP_PAIRS // 2):

            def op_pkadd(idx=i, b=bank):
                sum_tmps[idx] = Atom.pk_add_f32(
                    sp_pairs[idx * 2], sp_pairs[idx * 2 + 1], b
                )

            ops.append(op_pkadd)

        # --- Phase C3: sum tree (level-1, level-2, split, accum) ---
        for j in range_constexpr(N_SP_PAIRS // 4):

            def op_sum_l0(j_val=j, b=bank):
                sum_l0[j_val] = Atom.pk_add_f32(
                    sum_tmps[j_val * 2], sum_tmps[j_val * 2 + 1], b
                )

            ops.append(op_sum_l0)

        for j in range_constexpr(2):

            def op_sum_l1(j_val=j, b=bank):
                sum_l1[j_val] = Atom.pk_add_f32(
                    sum_l0[j_val * 2], sum_l0[j_val * 2 + 1], b
                )

            ops.append(op_sum_l1)

        def op_sum_l2(b=bank):
            sum_l2[0] = Atom.pk_add_f32(sum_l1[0], sum_l1[1], b)

        ops.append(op_sum_l2)

        def op_sum_split(b=bank):
            lo, hi = split_v2f32(sum_l2[0])
            final_sum[0] = Atom.add_f32(lo, hi, b)

        ops.append(op_sum_split)

        def op_sum_accum(b=bank):
            ss["row_sums"][b] = Atom.add_f32(ss["row_sums"][b], final_sum[0], b)

        ops.append(op_sum_accum)

        return ops

    @staticmethod
    def build_all_part2_ops(ty, blk, sp_pairs_all, softmax_state, sgpr_state):
        """Build ALL PART2 softmax ops for all 4 MSBs (built once, consumed
        incrementally across stages with per-stage budgets).

        Returns: list of 4 lists (one per MSB), each containing closures.
        Each closure emits exactly one instruction.
        """
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
                sp_f32[offset] = Vec(sp_pairs[offset // 2], dtype=Float32)[
                    offset % 2
                ].ir_value()
            return sp_f32[offset]

        # Phase 1: initial max3 per valid group (4 ops)
        for k in range_constexpr(N_VALID_GROUPS):

            def op_init_max3(k_=k, b=bank):
                base = k_ * VALID_GROUP_STRIDE
                tmps[k_] = Atom.max3_num_f32(
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
                    tmps[k_] = Atom.max3_num_f32(
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
                tmps[k_] = Atom.max3_num_f32(
                    _get_sp(base + 7), tmps[k_], _get_sp(base), b
                )

            ops.append(op_last_elem)

        # Phase 4: merge 4 groups (2 ops)
        def op_merge1(b=bank):
            tmps[0] = Atom.max3_num_f32(tmps[0], tmps[1], tmps[2], b)

        ops.append(op_merge1)

        # Phase 4b merge2 (op17): reads tmps[0] from merge1
        def op_merge2(b=bank):
            tmps[0] = Atom.max3_num_f32(tmps[0], tmps[3], tmps[1], b)

        ops.append(op_merge2)

        # Phase 5 perm_prep (op18): bank-change add_f32+0, reads tmps[0] from merge2
        tmps_perm = [None]
        _zero_f32 = arith.constant(0.0, type=T.f32)

        def op_perm_prep(b=bank, z=_zero_f32):
            dst_bank = (b + 2) % NUM_MSB
            tmps_perm[0] = Atom.add_f32(tmps[0], z, dst_bank)

        ops.append(op_perm_prep)

        # Phase 6 perm_exec (op19): permlanex16, reads tmps_perm[0] from perm_prep
        def op_perm(b=bank):
            sel_lo = arith.constant(0x76543210, type=T.i32)
            sel_hi = arith.constant(0xFEDCBA98, type=T.i32)
            dst_bank = (b + 2) % NUM_MSB
            tmps[1] = Atom.permlanex16(tmps_perm[0], sel_lo, sel_hi, dst_bank)

        ops.append(op_perm)

        # Phase 7 mul (op20): preMaxLog2eScl = old_max * log2e_scl — FULLY INDEPENDENT.
        # This op has no data dependency on any op in the merge1→merge2→perm chain.
        # The dispatch uses it as a 1-per-gap filler to provide 4 intervening VALU
        # between each consecutive dependent pair.
        def op_pre_max(b=bank):
            ss["pre_max_log2e_scl"][b] = Atom.mul_f32(
                ss["old_max"][b], sgpr["s_log2e_scl"], b
            )

        ops.append(op_pre_max)

        # Phase 8 cur_max (op21): max3(tmps[0], tmps[1], old_max), reads perm_exec result
        def op_cur_max(b=bank):
            ss["local_max"][b] = Atom.max3_num_f32(
                tmps[0], tmps[1], ss["old_max"][b], b
            )

        ops.append(op_cur_max)

        assert len(ops) == PART0_INSTS
        return ops

    @staticmethod
    def build_part1_ops(ty, ss, sgpr):
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
            ss["local_max"][0] = Atom.max3_num_f32(
                ss["local_max"][0], ss["local_max"][1], ss["pre_max_log2e_scl"][0], 0
            )

        ops.append(op_max01)
        msb_assign.append(0)

        # max_num(local_max[2], local_max[3]) → local_max[2]
        def op_max23():
            ss["local_max"][2] = Atom.max3_num_f32(
                ss["local_max"][2], ss["local_max"][3], ss["pre_max_log2e_scl"][2], 2
            )

        ops.append(op_max23)
        msb_assign.append(2)

        # mov local_max[1] = local_max[0]
        def op_mov1():
            ss["local_max"][1] = Atom.mov_b32(ss["local_max"][0], 1)

        ops.append(op_mov1)
        msb_assign.append(1)

        # mov local_max[3] = local_max[2]
        def op_mov3():
            ss["local_max"][3] = Atom.mov_b32(ss["local_max"][2], 3)

        ops.append(op_mov3)
        msb_assign.append(3)

        # fma delta[msb] = preMaxLog2eScl[msb] - local_max[msb] * log2e_scl
        for msb in [0, 2, 1, 3]:

            def op_fma_delta(b=msb):
                ss["delta"][b] = Atom.fma_f32_neg_src0(
                    ss["local_max"][b],
                    sgpr["s_log2e_scl"],
                    ss["pre_max_log2e_scl"][b],
                    b,
                )

            ops.append(op_fma_delta)
            msb_assign.append(msb)

        assert len(ops) == PART1_INSTS
        return ops, msb_assign

    @staticmethod
    def build_all_gemm2_ops(
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
            p0_ops = Softmax.build_part0_ops(ty, m, sp_pairs, softmax_state, sgpr_state)
            ops_by_rid[m] = p0_ops

        # PART1: 8 ops (cross-MSB)
        p1_ops, p1_msb_assign = Softmax.build_part1_ops(ty, softmax_state, sgpr_state)
        ops_by_rid[4] = p1_ops

        # PART2: first half only (ops 0..PART2_SPLIT-1) for each MSB.
        # sp_lo_cache / sp_hi_cache: per-MSB lists of f32 scalars to capture exp results
        # from EXP token dispatch. Avoids unreliable v2f32 packing under WMMA pressure.
        sp_lo_cache = [[None] * N_SP_PAIRS for _ in range_constexpr(NUM_MSB)]
        sp_hi_cache = [[None] * N_SP_PAIRS for _ in range_constexpr(NUM_MSB)]
        for m in range_constexpr(NUM_MSB):
            sp_pairs = sp_pairs_all[m]
            p2_ops = Softmax.build_part2_ops(
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

    @staticmethod
    def tiles_to_pairs(su_sp_tiles_list):
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
                v8w = Vec(v8, dtype=Float32)
                for i in range_constexpr(4):
                    lo = v8w[i * 2].ir_value()
                    hi = v8w[i * 2 + 1].ir_value()
                    pair_idx = su * 4 + i
                    v2 = make_v2f32(lo, hi, bank=msb)
                    if const_expr(msb > 0):
                        pairs[pair_idx] = set_vgpr_bank_offset(
                            v2, msb, SP_PAIR_BASE + pair_idx * 2
                        )
                    else:
                        pairs[pair_idx] = v2
            sp_pairs.append(pairs)
        return sp_pairs

    @staticmethod
    def part01_only(ty, blk, sp_pairs_all, softmax_state, sgpr_state):
        """Run PART0 + PART1 only (no PART2), for init stages 0..3 core work.

        After this call:
          - softmax_state['local_max'][msb] = max of sp_pairs for each MSB
          - softmax_state['delta'][msb] = old_max*log2e - local_max*log2e
          - softmax_state['pre_max_log2e_scl'][msb] = old_max * log2e_scl
        PART2 (pkfma, exp, cvt, rescale) runs later in core_loop's GEMM1 stages.
        """
        if const_expr("pre_max_log2e_scl" not in softmax_state):
            softmax_state["pre_max_log2e_scl"] = [None] * NUM_MSB

        ops_by_rid, _, _, _ = Softmax.build_all_gemm2_ops(
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
        sched_barrier(0)
        # mul(M0) filler: 4 intervening between merge1/merge2
        ops_by_rid[0][_MUL]()
        sched_barrier(0)

        # merge2 column-major
        for rid in range_constexpr(NUM_MSB):
            ops_by_rid[rid][17]()
        sched_barrier(0)
        # mul(M1) filler: 4 between merge2/perm_prep
        ops_by_rid[1][_MUL]()
        sched_barrier(0)

        # perm_prep column-major
        for rid in range_constexpr(NUM_MSB):
            ops_by_rid[rid][18]()
        sched_barrier(0)
        # mul(M2) filler: 4 between perm_prep/perm_exec
        ops_by_rid[2][_MUL]()
        sched_barrier(0)

        # perm_exec column-major
        for rid in range_constexpr(NUM_MSB):
            ops_by_rid[rid][19]()
        sched_barrier(0)
        # mul(M3) filler: 4 between perm_exec/cur_max
        ops_by_rid[3][_MUL]()
        sched_barrier(0)

        # cur_max column-major
        for rid in range_constexpr(NUM_MSB):
            ops_by_rid[rid][21]()

        # PART1 (cross-MSB, sequential after all PART0):
        for op in ops_by_rid[4]:
            op()

    @staticmethod
    def build_p_tiles(ty, softmax_state):
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
                su_tiles.append(Fragment.pack_v2bf16(ty, combined, m_tile))
            p_tiles.append(su_tiles)
        return p_tiles


# Softmax PART2 Builder


# Softmax PART0 + PART1 Builder (runs during GEMM2 stages)

# per MSB: tree-max(16) + merge2(1) + perm-prep(1)
#   + perm-exec(1) + mul(1) + cur-max(1) + merge1(1)
PART0_INSTS = 22
PART1_INSTS = 8  # cross-MSB merge + delta (total, not per MSB)
RLTS_LEN = 9  # run-ids: 0-3 PART0, 4 PART1, 5-8 PART2

N_VALID_GROUPS = CNT_SU  # 4 groups of valid data per MSB
VALID_GROUP_STRIDE = 8  # stride between groups (no padding)


# Unified FMHA Kernel


class Fragment:
    """WMMA fragment construction, pairing, and LDS tile loading."""

    @staticmethod
    def wmma_bf16(vec4_lo, vec4_hi):
        vec8_bf16_ty = T.vec(8, T.bf16)
        v0 = vector.bitcast(vec8_bf16_ty, vec4_lo)
        v1 = vector.bitcast(vec8_bf16_ty, vec4_hi)
        return vector.shuffle(v0, v1, list(range_constexpr(16)))

    @staticmethod
    def pair_k_tiles(kv_tiles_raw, ty):
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
                frag = Fragment.wmma_bf16(lo, hi)
                frag = set_vgpr_bank(frag, msb)
                msb_frags.append(frag)
            kv_paired.append(msb_frags)
        return kv_paired

    @staticmethod
    def pack_v2bf16(ty, v2bf16_list, bank):
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
            v8s.append(
                vector.shuffle(v4s[i * 2], v4s[i * 2 + 1], list(range_constexpr(8)))
            )

        # Stage 3: 2 v8bf16 → 1 v16bf16
        result = vector.shuffle(v8s[0], v8s[1], list(range_constexpr(16)))
        return set_vgpr_bank(result, bank)

    @staticmethod
    def pair_v_tiles(v_tiles_raw, ty):
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
                frag = Fragment.wmma_bf16(lo, hi)
                frags.append(frag)
            v_paired.append(frags)
        return v_paired

    @staticmethod
    def load_k_su(ty, kv_lds_addrs, blk, su):
        """Load K data for one SU from LDS, wait, return paired WMMA tiles."""
        kv_raw = [[None] * N_LDS_PER_MSB for _ in range_constexpr(NUM_MSB)]
        lds_schedule = Pipeline.build_lds_k_schedule(blk, su)
        for lds_op in lds_schedule:
            kv_raw = Pipeline.emit_lds_load(ty, lds_op, kv_lds_addrs, kv_raw)
        Atom.s_wait_dscnt(0)
        sched_barrier(0)
        return Fragment.pair_k_tiles(kv_raw, ty)

    @staticmethod
    def load_v_two_sus(ty, kv_lds_addrs, blk, su0, su1):
        """Load V data for two SUs simultaneously, sorted by MSB.

        Issues all 2×NUM_MSB×N_LDS_PER_MSB ds_load_tr16_b128 instructions
        grouped by MSB (all MSB-0 loads first, then MSB-1, …, then MSB-3),
        followed by a single s_wait_dscnt.

        Within each MSB group every load uses the same address
        bank (= bank_msb) and the same dst bank (= bank_msb), so
        s_set_vgpr_msb only changes at the three MSB-group boundaries →
        4 MSB contexts total instead of 8 with two back-to-back single-SU calls.
        """
        sched0 = Pipeline.build_lds_v_schedule(blk, su0)
        sched1 = Pipeline.build_lds_v_schedule(blk, su1)

        raw0 = [[None] * N_LDS_V_PER_MSB for _ in range_constexpr(NUM_MSB)]
        raw1 = [[None] * N_LDS_V_PER_MSB for _ in range_constexpr(NUM_MSB)]

        # Interleave by MSB: for each MSB, emit su0's loads then su1's loads.
        # All loads in one MSB group share the same addr bank and dst bank,
        # so no s_set_vgpr_msb switch is needed within the group.
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
    """GEMM schedule builders and per-WMMA dispatch helpers."""

    @staticmethod
    def build_qk_schedule(blk, su):
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

    @staticmethod
    def build_pv_schedule(blk, su):
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

    @staticmethod
    def build_lds_k_schedule(blk, su):
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

    @staticmethod
    def build_lds_v_schedule(blk, su):
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
        # (see build_kv_lds_addrs), so only the per-SU part remains in the offset field.
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

    @staticmethod
    def emit_qk_wmma(ty, wmma_op, q_tiles, kv_tiles, sp_tiles):
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
            result = Atom.wmma_init(ty, src_a, src_b, sp_msb)
        else:
            acc = sp_tiles[sp_msb][n_iter]
            result = Atom.wmma_accum(ty, src_a, src_b, acc, sp_msb)

        sp_tiles[sp_msb][n_iter] = result
        return sp_tiles

    @staticmethod
    def emit_pv_wmma(ty, wmma_op, v_tiles, p_tiles, o_tiles):
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

        result = Atom.wmma_accum(ty, src_a, src_b, acc, d_msb)
        o_tiles[d_msb][n] = result
        return o_tiles

    @staticmethod
    def emit_lds_load(ty, lds_op, kv_lds_addrs, kv_tiles_out):
        """Emit one LDS load (K: ds_load_b128 or V: ds_load_tr16_b128)."""
        msb = lds_op["msb"]
        offset = lds_op["offset"]
        v_idx = lds_op["v_idx"]
        load_type = lds_op["load_type"]

        if const_expr(load_type == "b128"):
            addr = kv_lds_addrs[msb]
            tile = Atom.ds_load_b128(ty, addr, offset, msb)
        else:
            half_p = lds_op["half_p"]
            # kv_lds_addrs[4 + msb*2 + half_p] — both dh0 and dh1 for
            # this MSB are in bank=msb, so dst/addr/addr are all same bank →
            # s_set_vgpr_msb stays constant within each MSB's load group.
            addr = kv_lds_addrs[NUM_MSB + msb * 2 + (1 if half_p else 0)]
            tile = Atom.ds_load_tr16_b128(ty, addr, offset, msb)

        kv_tiles_out[msb][v_idx] = tile
        return kv_tiles_out


# SECTION 5: Prologue Constants & Helpers (moved from fmha_kernel.py)

# Prologue Constants

K_TILE_N = 128  # TDM load tile = compute tile (N=128)

# TDM dim0=200 -> LDS inner stride = 200*2 = 400B
# (2-way bank conflicts)
K_ROW_BYTES = 400
V_ROW_BYTES = 288
K_SU_HALF_OFFSET = 0x1900  # 16 * K_ROW_BYTES = 16 * 400 = 6400
V_SU_HALF_OFFSET = 0x1200

# TDM config constants.
# K: dim0=192 (QK_HDIM), no padding. QK_HDIM=192 is not a multiple of any
# power-of-2 pad_interval that fits in one pad per row, so we skip padding to
# avoid the continuous-stream rotation bug. No bank-conflict padding for K.
K_TDM_CONFIG = 1 << 16  # data_size=1 (bf16), pad_enable=0
# V: dim0=128, pad_interval=128 elems=64dwords → enc_interval=5, 32B pad → enc_amount=7
V_TDM_CONFIG = (1 << 20) | (5 << 22) | (7 << 25)

TILE_N = K_TILE_N  # 128 — KV tile width

# SmemAllocators — 4 separate LDS regions for K/V ping-pong
# K per tile: CNT_SU(4) * LDS_K_SU_P_SIZE(0x3200)
# = 0xC800 = 51200 bytes (for QK_HDIM=192)
# V per tile: CNT_SU(4) × LDS_V_SU_P_SIZE(0x2400) = 0x9000 = 36864 bytes
#
# K_a, K_b, V_a are padded to 64KB segment boundary to prevent TDM cross-segment.
# V_b is last — no padding needed. D output reuses V_a (PV done before D store).
LDS_SEGMENT = 0x10000  # 64KB

lds_alloc_k_a = SmemAllocator(None, arch="gfx1250", global_sym_name="smem_k_a")
lds_alloc_k_a.ptr = LDS_SEGMENT

lds_alloc_k_b = SmemAllocator(None, arch="gfx1250", global_sym_name="smem_k_b")
lds_alloc_k_b.ptr = LDS_SEGMENT

lds_alloc_v_a = SmemAllocator(None, arch="gfx1250", global_sym_name="smem_v_a")
lds_alloc_v_a.ptr = LDS_SEGMENT

lds_alloc_v_b = SmemAllocator(None, arch="gfx1250", global_sym_name="smem_v_b")
# 0x9000, last buffer — no segment padding
lds_alloc_v_b.ptr = CNT_SU * LDS_V_SU_P_SIZE

TDM_D_TILE_DIM0 = 128 * 2  # 256 bytes per LDS row
WV_SUBQD = 32
LDS_D_WV_SIZE = WV_SUBQD * TDM_D_TILE_DIM0 + 1024  # 9216 bytes per wave

# Prologue Helpers


NUW_ATTR = None


def get_nuw():
    global NUW_ATTR
    if NUW_ATTR is None:
        NUW_ATTR = ir.Attribute.parse("#arith.overflow<nuw>")
    return NUW_ATTR


def _ensure_ir_value(v):
    """Convert Numeric (fx.Int32 etc.) to ir.Value for raw MLIR ops."""
    if isinstance(v, ir.Value):
        return v
    if hasattr(v, "ir_value"):
        return v.ir_value()
    return v


def add_nuw(a, b):
    """arith.addi with nuw flag — enables gfx1250 buffer offset folding."""
    return arith.addi(
        _ensure_ir_value(a), _ensure_ir_value(b), overflow_flags=get_nuw()
    )


def mul_nuw(a, b):
    """arith.muli with nuw flag — preserves nuw through constant folding."""
    return arith.muli(
        _ensure_ir_value(a), _ensure_ir_value(b), overflow_flags=get_nuw()
    )


def setreg(hwreg_enc, value):
    """s_setreg_imm32_b32 via llvm.amdgcn.s.setreg intrinsic.
    hwreg_enc = id | (offset << 6) | ((size-1) << 11)"""
    imm = arith.constant(hwreg_enc, type=T.i32)
    val = arith.constant(value, type=T.i32)
    llvm_dialect.call_intrinsic(None, "llvm.amdgcn.s.setreg", [imm, val], [], [])


# FlyDSL Phase Functions


def phase4_q_load(
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
    # Q uses same (16,2):(1,16) lane decomposition as K LDS
    q_thr = fx.make_layout((16, 2), (1, 16))
    q_crd = idx2crd(lane_id, q_thr)
    lane_lo = arith.index_cast(T.i32, q_crd[0])
    lane_hi = arith.index_cast(T.i32, q_crd[1])
    base = lane_lo * stride_q_seq + lane_hi * 16
    wave_off = (wave_id * 32) * stride_q_seq
    q_byte_off = base + wave_off

    q_elem_off = q_byte_off >> 2

    vec4i32_ty = T.vec(4, T.i32)
    soff_zero = arith.constant(0, type=T.i32)
    aux_zero = arith.constant(0, type=T.i32)

    four_i32 = arith.constant(4, type=T.i32)
    q_base_bytes = mul_nuw(q_elem_off, four_i32)
    if q_tile_offset_bytes is not None:
        q_base_bytes = add_nuw(q_tile_offset_bytes, q_base_bytes)
    stride_16_bytes = stride_q_seq * 16

    # K-half byte offset = QK_HDIM bytes (splits K cols in half per bank pair)
    _K_HALF_BYTES = QK_HDIM  # 192 for 192-dim, 128 for 128-dim
    # Each bank covers half of QK_HDIM: QK_HDIM/2 elements / WMMA_K(32) = frags_per_bank
    # = 3 for 192-dim (96 K cols / 32 = 3), 2 for 128-dim
    _FRAGS_PER_BANK = (QK_HDIM // 2) // 32
    _LOADS_PER_BANK = _FRAGS_PER_BANK * 2  # 2 raw loads per v16bf16 frag

    _k_half_c = arith.constant(_K_HALF_BYTES, type=T.i32)
    bank_offsets_bytes = [
        arith.constant(0, type=T.i32),
        _k_half_c,
        stride_16_bytes,
        add_nuw(stride_16_bytes, _k_half_c),
    ]

    q_frags = []
    for bank in fx.range_constexpr(4):
        if bank == 0:
            bank_voff = q_base_bytes
        else:
            bank_voff = add_nuw(q_base_bytes, bank_offsets_bytes[bank])
        bank_loads = []
        for i in fx.range_constexpr(_LOADS_PER_BANK):
            if i == 0:
                voff = bank_voff
            else:
                voff = add_nuw(
                    bank_voff,
                    arith.constant(i * 32, type=T.i32),
                )
            loaded = rocdl.raw_ptr_buffer_load(
                vec4i32_ty, q_rsrc, voff, soff_zero, aux_zero
            )
            bank_loads.append(set_vgpr_bank(loaded, bank))
        rocdl.sched_barrier(0)
        bank_frags = []
        for f in fx.range_constexpr(_FRAGS_PER_BANK):
            frag = Fragment.wmma_bf16(bank_loads[2 * f], bank_loads[2 * f + 1])
            bank_frags.append(set_vgpr_bank(frag, bank))
        q_frags.append(bank_frags)
        rocdl.sched_barrier(0)

    return q_frags


def head_index_div(workgroup_id, num_heads):
    """head_index = workgroup_id / num_heads — LLVM auto-generates Newton-Raphson."""
    quotient = workgroup_id // num_heads
    return rocdl.readfirstlane(T.i32, quotient)


def split_i64_to_lo_hi(val_i64):
    lo = arith.trunci(T.i32, val_i64)
    hi_shifted = val_i64 >> 32
    hi_raw = arith.trunci(T.i32, hi_shifted)
    hi = hi_raw | -2147483648
    return lo, hi


def ptr_base_i64(tensor):
    """Extract the i64 base address from an fx.Pointer / fly tensor."""
    raw = tensor.__extract_to_ir_values__()[0]
    glb_ptr = fly_d.extract_aligned_pointer_as_index(glb_ptr_ty(), raw)
    return llvm_dialect.ptrtoint(T.i64, glb_ptr)


def compute_global_addr(tensor, byte_offset, wave_id, stride_32):
    """Compute global i64 address: tensor base + byte_offset + wave_id * stride_32."""
    base_i64 = ptr_base_i64(tensor)
    off_i64 = fx.Int64(byte_offset)
    wave_off = fx.Int64(wave_id * stride_32)
    return base_i64 + off_i64 + wave_off


# LDS base extraction helper


def extract_lds_base_i32(memref_base):
    """Extract i32 LDS address from SmemAllocator memref base."""
    from flydsl._mlir.dialects import memref as _memref_d

    idx = _memref_d.extract_aligned_pointer_as_index(memref_base)
    return arith.index_cast(T.i32, idx)


def build_kv_lds_addrs(lane_id, k_base_i32, v_base_i32):
    """Build kv_lds_addrs[12] from allocator i32 bases + per-lane offsets.

    Returns [K_dh0(b0), K_dh1(b1), K_dh0_hi(b2), K_dh1_hi(b3),
             V_dh0(b0), V_dh1(b0), ..., V_dh0(b3), V_dh1(b3)].
    All V addresses for one MSB share the same bank → zero s_set_vgpr_msb switches.
    """
    # K LDS thread layout: (16,2):(1,16) → lane%16=row, lane//16=col_half
    # byte_off = row * K_ROW_BYTES + col_half * 16
    k_thr = fx.make_layout((16, 2), (1, 16))
    k_crd = idx2crd(lane_id, k_thr)
    k_row = arith.index_cast(T.i32, k_crd[0])
    k_col = arith.index_cast(T.i32, k_crd[1])
    k_lane_off = k_row * K_ROW_BYTES + k_col * 16

    k_dh0 = k_base_i32 + k_lane_off
    k_dh1 = k_dh0 + K_SU_HALF_OFFSET
    K_COL_D_HALF = QK_HDIM * KV_BPP // 2
    k_dh0_hi = k_dh0 + K_COL_D_HALF
    k_dh1_hi = k_dh1 + K_COL_D_HALF

    # V LDS thread layout: (8,2,2):(1,8,16) → (row_lo, sub_col, row_hi)
    # byte_off = (row_lo + row_hi*8) * V_ROW_BYTES + sub_col * 16
    v_thr = fx.make_layout((8, 2, 2), (1, 8, 16))
    v_crd = idx2crd(lane_id, v_thr)
    v_row_lo = arith.index_cast(T.i32, v_crd[0])
    v_sub_col = arith.index_cast(T.i32, v_crd[1])
    v_row_hi = arith.index_cast(T.i32, v_crd[2])
    v_row = v_row_lo + v_row_hi * 8
    v_lane_off = v_row * V_ROW_BYTES + v_sub_col * 16

    v_dh0 = v_base_i32 + v_lane_off
    v_dh1 = v_dh0 + V_SU_HALF_OFFSET

    rocdl.sched_barrier(0)

    _V_D_HALF = V_HDIM * KV_BPP // 2
    _V_COL_GROUP = (N_LDS_V_PER_MSB // 2) * 32
    _V_MSB_EXTRA = [0, _V_D_HALF, _V_COL_GROUP, _V_D_HALF + _V_COL_GROUP]

    v_addrs = []
    for msb in range(NUM_MSB):
        extra = _V_MSB_EXTRA[msb]
        if extra == 0:
            v_dh0_b, v_dh1_b = v_dh0, v_dh1
        else:
            v_dh0_b = v_dh0 + extra
            v_dh1_b = v_dh1 + extra
        v_addrs += [set_vgpr_bank(v_dh0_b, msb), set_vgpr_bank(v_dh1_b, msb)]

    return [
        set_vgpr_bank(k_dh0, 0),
        set_vgpr_bank(k_dh1, 1),
        set_vgpr_bank(k_dh0_hi, 2),
        set_vgpr_bank(k_dh1_hi, 3),
    ] + v_addrs


# TDM Priming — Load full tile_n=128 of K+V into LDS
# Initial K load from LDS — provides kv_tiles for core_loop entry
def issue_k_loads(ty, kv_lds_addrs, blk, su):
    """Issue ds_load_b128 for one SU of K. Does NOT wait.

    Returns raw kv_raw[4 msb][N_LDS_PER_MSB] — caller must wait
    (rocdl.s_wait_dscnt(0)) before using the results.
    """

    su_off = (blk * CNT_SU + su) * LDS_K_SU_P_SIZE
    kv_raw = [[None] * N_LDS_PER_MSB for _ in range(NUM_MSB)]
    for msb in range(NUM_MSB):
        for v_idx in range(N_LDS_PER_MSB):
            offset = v_idx * 32 + su_off
            kv_raw[msb][v_idx] = Atom.ds_load_b128(ty, kv_lds_addrs[msb], offset, msb)
    return kv_raw


def wait_and_pair_k(ty, kv_raw):
    """Wait for issued K loads and pair into WMMA-ready v16bf16."""
    rocdl.s_wait_dscnt(0)
    return Fragment.pair_k_tiles(kv_raw, ty)


def load_initial_kv_tiles(ty, kv_lds_addrs, blk, su):
    """Load K data for one SU from LDS → kv_tiles[4 msb][2] v16bf16.

    Issues N_LDS_PER_MSB ds_load_b128 per MSB, waits, then pairs
    into WMMA-ready v16bf16 fragments.
    """
    kv_raw = issue_k_loads(ty, kv_lds_addrs, blk, su)
    return wait_and_pair_k(ty, kv_raw)


def gemm1_interleaved_stage(
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

    wmma_schedule = Pipeline.build_qk_schedule(gemm_blk, gemm_su)

    if const_expr(lds_type == KV_K):
        lds_schedule = Pipeline.build_lds_k_schedule(lds_blk, lds_su)
    else:
        lds_schedule = Pipeline.build_lds_v_schedule(lds_blk, lds_su)

    lds_idx = 0
    ds_issued = 0
    _o_resc_idx = 0  # O-rescale closure index (for O_RESC token dispatch)

    for gemm_idx in range_constexpr(GEMM_INST_COUNT):
        # === Emit WMMA ===
        wmma_op = wmma_schedule[gemm_idx]
        sp_tiles = Pipeline.emit_qk_wmma(ty, wmma_op, q_tiles, kv_tiles, sp_tiles)
        sched_barrier(0)

        # DS wait at 1/4 point (mandatory, non-scheduled)
        sched_barrier(0)
        if const_expr(gemm_idx == GEMM_INST_COUNT // 4 - 1):
            Atom.s_wait_dscnt(ds_issued)
        sched_barrier(0)

        # TDM loads at WMMA 0 (non-scheduled: tensor_load_to_lds setup is SALU+load)
        if const_expr(gemm_idx == 0 and has_tdm):
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
            else:
                if const_expr(tdm_type == KV_V):
                    Atom.tdm_load(ty, tdm_state["v_g0"], tdm_state["v_g1"])
                else:
                    Atom.tdm_load(ty, tdm_state["k_g0"], tdm_state["k_g1"])
            sched_barrier(0)

        # TDM barrier signal (mandatory, non-scheduled)
        _g1_barrier_idx = GEMM_INST_COUNT - BARRIER_SIGNAL_AHEAD - 1
        if const_expr(tdm_barrier and gemm_idx == _g1_barrier_idx):
            Atom.s_wait_tensorcnt(4)
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
                    _msb = _tok - P2_BASE
                    if softmax_idx_by_msb[_msb] < len(softmax_ops_by_msb[_msb]):
                        softmax_ops_by_msb[_msb][softmax_idx_by_msb[_msb]]()
                        softmax_idx_by_msb[_msb] += 1
                    sched_barrier(0)
                elif const_expr(9 <= _tok <= 12):  # K tile ds_load_b128
                    if lds_idx < LDS_INST_COUNT:
                        kv_tiles_next = Pipeline.emit_lds_load(
                            ty, lds_schedule[lds_idx], kv_lds_addrs, kv_tiles_next
                        )
                        lds_idx += 1
                        ds_issued += 1
                elif const_expr(13 <= _tok <= 16):  # V tile ds_load_tr16_b128
                    if lds_idx < LDS_V_INST_COUNT:
                        kv_tiles_next = Pipeline.emit_lds_load(
                            ty, lds_schedule[lds_idx], kv_lds_addrs, kv_tiles_next
                        )
                        lds_idx += 1
                        ds_issued += 1
                elif const_expr(_tok == 17):  # O_RESC: pk_mul closure
                    if const_expr(o_rescale_ops is not None):
                        sched_barrier(0)
                        o_rescale_ops[_o_resc_idx]()
                        _o_resc_idx += 1
                        sched_barrier(0)
                # Token 18 (TDM): handled above, nothing here

        # DS wait at end of stage (mandatory, non-scheduled)
        sched_barrier(0)
        if const_expr(gemm_idx == GEMM_INST_COUNT - 1):
            Atom.s_wait_dscnt(LDS_INST_COUNT // 2)
        sched_barrier(0)

        # Phase 2: second half of row
        for _i in range_constexpr(len(_g1_row)):
            if const_expr(_g1_half <= _i < len(_g1_row)):
                _tok = _g1_row[_i]
                if const_expr(5 <= _tok <= 8):
                    _msb = _tok - P2_BASE
                    if softmax_idx_by_msb[_msb] < len(softmax_ops_by_msb[_msb]):
                        softmax_ops_by_msb[_msb][softmax_idx_by_msb[_msb]]()
                        softmax_idx_by_msb[_msb] += 1
                    sched_barrier(0)
                elif const_expr(9 <= _tok <= 12):
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
                elif const_expr(_tok == 17):
                    if const_expr(o_rescale_ops is not None):
                        sched_barrier(0)
                        o_rescale_ops[_o_resc_idx]()
                        _o_resc_idx += 1
                        sched_barrier(0)

        # TDM barrier_wait at last WMMA (mandatory, non-scheduled)
        if const_expr(tdm_barrier and gemm_idx == GEMM_INST_COUNT - 1):
            rocdl.s_barrier_wait(-1)

        sched_barrier(0)

    return sp_tiles, kv_tiles_next


# Core Loop Stage Composer — cl_su_V3 Interleaving Engine (GEMM2 / PV)
def gemm2_interleaved_stage(
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
    wmma_schedule = Pipeline.build_pv_schedule(gemm_blk, gemm_su)

    if const_expr(lds_type == KV_K):
        lds_schedule = Pipeline.build_lds_k_schedule(lds_blk, lds_su)
    else:
        lds_schedule = Pipeline.build_lds_v_schedule(lds_blk, lds_su)

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

        # --- Phase 1 (cycle23=1) ---
        # TDM at WMMA 0 (non-scheduled: tensor_load_to_lds setup is SALU)
        if const_expr(gemm_idx == 0 and has_tdm):
            tdm_key = "v" if tdm_type == KV_V else "k"
            descs = tdm_state.get(f"{tdm_key}_descs", None)
            if const_expr(descs is not None):
                sched_barrier(0)
                _tdm_di = tdm_state[f"{tdm_key}_desc_idx"]
                for _ in range_constexpr(TDM_LOADS_PER_STAGE):
                    if const_expr(_tdm_di < len(descs)):
                        Atom.tdm_load(ty, descs[_tdm_di][0], descs[_tdm_di][1])
                        _tdm_di += 1
                tdm_state[f"{tdm_key}_desc_idx"] = _tdm_di
                sched_barrier(0)

        _pv_barrier_idx = PV_GEMM_INST_COUNT - BARRIER_SIGNAL_AHEAD - 1
        if const_expr(tdm_barrier and gemm_idx == _pv_barrier_idx):
            Atom.s_wait_tensorcnt(4)
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
                    sched_barrier(0)
                elif const_expr(9 <= _tok <= 12):  # K tile ds_load_b128
                    if lds_idx < LDS_INST_COUNT:
                        kv_tiles_next = Pipeline.emit_lds_load(
                            ty, lds_schedule[lds_idx], kv_lds_addrs, kv_tiles_next
                        )
                        lds_idx += 1
                        ds_issued += 1
                elif const_expr(13 <= _tok <= 16):  # V tile ds_load_tr16_b128
                    if lds_idx < LDS_V_INST_COUNT:
                        kv_tiles_next = Pipeline.emit_lds_load(
                            ty, lds_schedule[lds_idx], kv_lds_addrs, kv_tiles_next
                        )
                        lds_idx += 1
                        ds_issued += 1
                elif const_expr(19 <= _tok <= 22):  # EXP_Mx pair_exp (3cy)
                    _msb = _tok - EXP_BASE
                    _erid = _msb + P2_BASE
                    if exp_rid_idx[_msb] < len(ops_by_rid[_erid]):
                        ops_by_rid[_erid][exp_rid_idx[_msb]]()
                        exp_rid_idx[_msb] += 1
                    sched_barrier(0)

        sched_barrier(0)
        if const_expr(gemm_idx == PV_GEMM_INST_COUNT - 1):
            Atom.s_wait_dscnt(LDS_INST_COUNT // 2)
        sched_barrier(0)

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
                    sched_barrier(0)
                elif const_expr(9 <= _tok <= 12):
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
                elif const_expr(19 <= _tok <= 22):
                    _msb = _tok - EXP_BASE
                    _erid = _msb + P2_BASE
                    if exp_rid_idx[_msb] < len(ops_by_rid[_erid]):
                        ops_by_rid[_erid][exp_rid_idx[_msb]]()
                        exp_rid_idx[_msb] += 1
                    sched_barrier(0)

        if const_expr(tdm_barrier and gemm_idx == PV_GEMM_INST_COUNT - 1):
            rocdl.s_barrier_wait(-1)

        sched_barrier(0)

    return o_tiles, kv_tiles_next


# Pure QK / PV / Softmax Helpers (for prologue — no interleaving)
def qk_gemm_pure(ty, blk, su, q_tiles, kv_tiles, sp_tiles):
    """Pure QK GEMM for one SU: GEMM_INST_COUNT WMMAs, no interleaving."""
    schedule = Pipeline.build_qk_schedule(blk, su)
    for wmma_op in schedule:
        sp_tiles = Pipeline.emit_qk_wmma(ty, wmma_op, q_tiles, kv_tiles, sp_tiles)
        sched_barrier(0)
    return sp_tiles


def pv_gemm_pure(ty, blk, su, v_tiles, p_tiles_su, o_tiles):
    """Pure PV GEMM for one SU: 16 WMMAs (4 d_msb × 4 n), no interleaving.

    Returns updated o_tiles.
    v_tiles: [N_V_MSB][N_PV_WMMA_N] v16bf16 (paired V data)
    p_tiles_su: [2 m_tile] v16bf16 (compact softmax output for this SU, each
        v16bf16 is a full no-padding PV WMMA src_b for one (m_tile, K_pv chunk)).
    """
    schedule = Pipeline.build_pv_schedule(blk, su)
    for wmma_op in schedule:
        o_tiles = Pipeline.emit_pv_wmma(ty, wmma_op, v_tiles, p_tiles_su, o_tiles)
        sched_barrier(0)
    return o_tiles


class TDM:
    """TDM (Tensor Data Mover) descriptor building and load dispatch."""

    @staticmethod
    def build_descs(dg1, addr_i64, stride_adv_i64, lds_base, su_p_size, n_su):
        """Build per-SU TDM descriptors [(dg0, dg1)] without issuing loads.

        dg1 can be a single v8i32 (shared across SUs) or a list of n_su v8i32.
        """
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
        """Issue TDM loads from pre-built descriptors, with per-SU barriers."""
        for dg0, dg1 in descs:
            tdm_ops.tensor_load_2d(tdm_ops.TDMDescriptor2D(dg0, dg1))
            rocdl.s_barrier_signal(-1)
            rocdl.s_barrier_wait(-1)

    @staticmethod
    def per_warp_oob_dim1(total_rows_i32, wave_id, rows_per_warp=8):
        wave_off = wave_id * rows_per_warp
        remaining = total_rows_i32 - wave_off
        clamped_lo = arith.maxsi(
            remaining,
            arith.constant(0, type=T.i32),
        )
        return arith.minsi(
            clamped_lo,
            arith.constant(rows_per_warp, type=T.i32),
        )

    @staticmethod
    def make_kv_dg1_with_oob(
        config_bf16,
        dim0_elems,
        dim1_rows,
        stride_seq_elems,
        oob_dim1_raw,
        dim0_stride=None,
    ):
        _td1_lo = arith.andi(oob_dim1_raw, arith.constant(0xFFFF, type=T.i32))
        _sgpr2 = arith.shli(_td1_lo, arith.constant(16, type=T.i32))
        if dim0_stride is None:
            dim0_stride = dim0_elems
        return Vec.from_elements(
            [
                fx.Int32(config_bf16),
                fx.Int32(dim0_elems << 16),
                _sgpr2,
                fx.Int32(dim0_stride << 16),
                fx.Int32(dim1_rows),
                stride_seq_elems,
                fx.Int32(0),
                fx.Int32(0),
            ],
            fx.Int32,
        )

    @staticmethod
    def build_oob_dg1_list(
        config, dim0_elems, stride_elems, remain, wave_id, dim0_stride=None
    ):
        return [
            TDM.make_kv_dg1_with_oob(
                config,
                dim0_elems,
                8,
                stride_elems,
                TDM.per_warp_oob_dim1(remain - su * 32, wave_id, 8),
                dim0_stride=dim0_stride,
            )
            for su in range(CNT_SU)
        ]

    @staticmethod
    def load_kv_blk(kv_type, dg1, addr_i64, stride_adv_i64, lds_base, su_p_size, n_su):
        """Issue n_su TDM loads for one blk of K or V data.

        Each TDM load covers one SU. After all loads, issues barrier.
        dg1 can be a single v8i32 or a list of n_su v8i32 (per-SU OOB).
        """
        descs = TDM.build_descs(
            dg1, addr_i64, stride_adv_i64, lds_base, su_p_size, n_su
        )
        TDM.issue_from_descs(descs)

    @staticmethod
    def load_k_only(
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
        K_ROW_BYTES = 384 (= 192 * 2, flat, no padding).
        Per-warp: 4 warps x 8 rows = 32 rows per SU, 4 SUs total.
        """
        _DIM0_VALID = QK_HDIM  # 192
        _DIM0_STRIDE = 200  # LDS row stride in elements
        _DIM1_ROWS = 8

        _K_CONFIG_BF16 = (1 << 16) | K_TDM_CONFIG
        stride_k_seq_elems = stride_k_seq >> 1

        k_dg1 = (
            oob_dg1_list
            if oob_dg1_list is not None
            else Vec.from_elements(
                [
                    fx.Int32(_K_CONFIG_BF16),
                    fx.Int32(_DIM0_VALID << 16),
                    fx.Int32(_DIM1_ROWS << 16),
                    fx.Int32(_DIM0_STRIDE << 16),
                    fx.Int32(_DIM1_ROWS),
                    stride_k_seq_elems,
                    fx.Int32(0),
                    fx.Int32(0),
                ],
                fx.Int32,
            )
        )

        k_addr = compute_global_addr(ptr_K, k_offset, wave_id, 8 * stride_k_seq)

        lds_warp_off = wave_id * (8 * K_ROW_BYTES)
        lds_base_with_warp = lds_base_i32 + lds_warp_off

        k_stride_adv = fx.Int64(stride_k_32)

        TDM.load_kv_blk(
            KV_K,
            k_dg1,
            k_addr,
            k_stride_adv,
            lds_base_with_warp,
            LDS_K_SU_P_SIZE,
            CNT_SU,
        )

        tdm_wait_and_barrier()

    @staticmethod
    def load_v_only(
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
        _V_CONFIG_BF16 = (1 << 16) | V_TDM_CONFIG
        _DIM0_ELEMS = 128
        _DIM1_ROWS = 8
        stride_v_seq_elems = stride_v_seq >> 1

        v_dg1 = (
            oob_dg1_list
            if oob_dg1_list is not None
            else Vec.from_elements(
                [
                    fx.Int32(_V_CONFIG_BF16),
                    fx.Int32(_DIM0_ELEMS << 16),
                    fx.Int32(_DIM1_ROWS << 16),
                    fx.Int32(_DIM0_ELEMS << 16),
                    fx.Int32(_DIM1_ROWS),
                    stride_v_seq_elems,
                    fx.Int32(0),
                    fx.Int32(0),
                ],
                fx.Int32,
            )
        )

        v_addr = compute_global_addr(ptr_V, v_offset, wave_id, 8 * stride_v_seq)

        lds_warp_off = wave_id * (8 * V_ROW_BYTES)
        lds_base_with_warp = lds_base_i32 + lds_warp_off

        v_stride_adv = fx.Int64(stride_v_32)

        TDM.load_kv_blk(
            KV_V,
            v_dg1,
            v_addr,
            v_stride_adv,
            lds_base_with_warp,
            LDS_V_SU_P_SIZE,
            CNT_SU,
        )

        tdm_wait_and_barrier()


# ================================================================
# Kernel pipeline functions
# (extracted from fmha_fwd_kernel -- take ctx dict for kernel state)
# ================================================================

# Iter-arg layout constants (must match fmha_kernel.py SECTION 3 init_args)
_KV_SIZE = NUM_MSB * N_WMMA_K_TILES
_OFF_LOCAL_MAX = 24 + _KV_SIZE
_OFF_DELTA = _OFF_LOCAL_MAX + NUM_MSB
_OFF_SP = _OFF_DELTA + NUM_MSB
_OFF_PP = _OFF_SP + CNT_SU * NUM_MSB
_OFF_PSP = _OFF_PP + 4
_PSP_SIZE = NUM_MSB * N_SP_PAIRS
_OFF_PSP_HI = _OFF_PSP + _PSP_SIZE
_OFF_PED = _OFF_PSP_HI + _PSP_SIZE


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


def fmha_pipeline_ctx(
    ctx,
    ty,
    memload,
    q_tiles,  # [4 msb][Q_WMMA_PER_MSB] v16bf16 -- Q data
    kv_tiles,  # [4 msb][2] v16bf16 -- paired K tiles for WMMA
    sp_tiles,  # [4 msb][1] v8f32 -- QK accumulators
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
    # exp_delta from previous tile (for O-rescale)
    ia_exp_delta=None,
    # i32 n_start for causal mask (None=skip)
    causal_n_start=None,
    # list[CNT_SU] v8i32 OOB dg1 for GEMM1 V TDM
    endtile_v_oob_dg1=None,
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
      GEMM2: PV using P tiles x V (in LDS, from kv_lds_addrs[4:8])
      TDM: GEMM1 loads K(i+1)->K_next; GEMM2 loads V(i)->V_next (main loop)
           GEMM1 loads V(endtile)->V_next (epilogue, gemm1_tdm_is_v=True)

    kv_lds_addrs: [K_cur[0:4] + V_cur[4:8]] -- built from mixed allocator
    bases via build_kv_lds_addrs(lane_id, k_cur_base, v_cur_base).
    kv_lds_addrs_next: same structure for next ping-pong buffer, used for
    GEMM2 stage 3 K preload (K_next already filled by GEMM1 TDM).

    Returns: (sp_tiles, kv_tiles, o_tiles, su_sp_tiles_list).
    """
    stride_k_seq = ctx["stride_k_seq"]
    stride_v_seq = ctx["stride_v_seq"]
    stride_k_32 = ctx["stride_k_32"]
    stride_v_32 = ctx["stride_v_32"]
    ptr_K = ctx["ptr_K"]
    ptr_V = ctx["ptr_V"]
    wave_id = ctx["wave_id"]

    Atom.s_wait_dscnt(LDS_INST_COUNT // 2)  # s_wait_dscnt 0x8

    v_tiles_out = None
    blk = 0

    # ================================================================
    # GEMM1 (QK): 4 stages (SU 0..3)
    # Interleave PART2 on sp_pairs_prev during GEMM1 stages.
    # ================================================================
    sp_pairs_all = softmax_state.get("sp_pairs_prev", None)
    if const_expr(sp_pairs_all is None):
        sp_pairs_all = [[None] * N_SP_PAIRS for _ in range(NUM_MSB)]

    softmax_ops_by_msb = Softmax.build_all_part2_ops(
        ty, 0, sp_pairs_all, softmax_state, sgpr_state
    )
    # Second half starts at PART2_SPLIT; first half ran in previous GEMM2.
    softmax_idx_by_msb = [PART2_SPLIT] * NUM_MSB

    # Pre-run setup ops 0..PART2_SETUP_A-2 (skip last = row_sums rescale,
    # already applied).
    for m in fx.range_constexpr(NUM_MSB):
        # ops 0..5, skip op 6
        for i in fx.range_constexpr(PART2_SETUP_A - 1):
            softmax_ops_by_msb[m][i]()

    # Build TDM descriptors for GEMM1 stages 0-1.
    has_tdm_k_g1 = (not gemm1_tdm_is_v) and (tdm_k_offset is not None)
    has_tdm_v_g1 = gemm1_tdm_is_v and (tdm_v_offset is not None)

    if has_tdm_k_g1:
        _K_CFG = (1 << 16) | K_TDM_CONFIG
        _stride_k_elems = stride_k_seq >> 1
        if const_expr(loop_k_oob_dg1 is not None):
            k_dg1 = loop_k_oob_dg1
        else:
            k_dg1 = Vec.from_elements(
                [
                    fx.Int32(_K_CFG),
                    fx.Int32(QK_HDIM << 16),
                    fx.Int32(8 << 16),
                    fx.Int32(200 << 16),
                    fx.Int32(8),
                    _stride_k_elems,
                    fx.Int32(0),
                    fx.Int32(0),
                ],
                fx.Int32,
            )
        k_addr = compute_global_addr(
            ptr_K,
            tdm_k_offset,
            wave_id,
            8 * stride_k_seq,
        )
        k_stride_adv = fx.Int64(stride_k_32)
        _k_warp_off = wave_id * (8 * K_ROW_BYTES)
        _k_lds_base = tdm_k_target + _k_warp_off
        k_descs = TDM.build_descs(
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
        _V_CFG = (1 << 16) | V_TDM_CONFIG
        _stride_v_elems = stride_v_seq >> 1
        if const_expr(endtile_v_oob_dg1 is not None):
            v_dg1 = endtile_v_oob_dg1
        else:
            v_dg1 = Vec.from_elements(
                [
                    fx.Int32(_V_CFG),
                    fx.Int32(128 << 16),
                    fx.Int32(8 << 16),
                    fx.Int32(128 << 16),
                    fx.Int32(8),
                    _stride_v_elems,
                    fx.Int32(0),
                    fx.Int32(0),
                ],
                fx.Int32,
            )
        v_addr = compute_global_addr(
            ptr_V,
            tdm_v_offset,
            wave_id,
            8 * stride_v_seq,
        )
        v_stride_adv = fx.Int64(stride_v_32)
        _v_warp_off = wave_id * (8 * V_ROW_BYTES)
        _v_lds_base = tdm_v_target + _v_warp_off
        v_descs = TDM.build_descs(
            v_dg1,
            v_addr,
            v_stride_adv,
            _v_lds_base,
            LDS_V_SU_P_SIZE,
            CNT_SU,
        )
        tdm_state["v_descs"] = v_descs
        tdm_state["v_desc_idx"] = 0

    g1_tdm_type = KV_K if has_tdm_k_g1 else KV_V if has_tdm_v_g1 else KV_NONE
    stage_configs = [
        (0, g1_tdm_type, KV_K, blk, 1),
        (1, g1_tdm_type, KV_K, blk, 2),
        (2, KV_NONE, KV_K, blk, 3),
        (3, KV_NONE, KV_V, blk, 0),
    ]

    su_sp_tiles_list = []

    # O rescale: all 4 MSBs across GEMM1 stages 0-3.
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

        softmax_stage = (stage_idx + 4) % ALU_STAGES
        budget_per_msb = ALU_PER_STAGE[softmax_stage] // NUM_MSB
        softmax_budget = [budget_per_msb, budget_per_msb, 0, 0]

        is_barrier_stage = stage_idx == 2 and (has_tdm_k_g1 or has_tdm_v_g1)

        stage_o_rescale = o_rescale_by_stage[stage_idx]

        sp_tiles, kv_tiles_next_raw = gemm1_interleaved_stage(
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

        su_sp_tiles_list.append([[sp_tiles[msb][0]] for msb in range(NUM_MSB)])

        if const_expr(l_type == KV_K):
            kv_tiles = Fragment.pair_k_tiles(kv_tiles_next_raw, ty)
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

    # O_resc moved entirely to GEMM1 stages -- no O_resc in GEMM2.
    o_rescale_exp_delta = None

    # ================================================================
    # Causal mask on current QK tile (before sp_pairs conversion)
    # ================================================================
    if const_expr(causal_n_start is not None):
        apply_causal_mask(ctx, su_sp_tiles_list, causal_n_start)
    if const_expr(kv_oob_cols is not None):
        apply_kv_oob_mask(ctx, su_sp_tiles_list, kv_oob_cols)

    # 2. Build sp_pairs for current tile (all 4 SUs).
    sp_pairs_current = Softmax.tiles_to_pairs(su_sp_tiles_list)

    # 3. Build all PART0+PART1+PART2 closures for current tile.
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

    # 4. Build P tiles from p_bf16
    p_tiles_computed = Softmax.build_p_tiles(ty, softmax_state)

    # ================================================================
    # GEMM2 (PV): 4 stages (SU 0..3)
    # ================================================================

    v_tiles_paired = Fragment.pair_v_tiles(v_tiles_out, ty)

    # Build V TDM descriptors for GEMM2 stages 0-1.
    has_tdm_v_g2 = (not gemm1_tdm_is_v) and (tdm_v_offset is not None)
    if has_tdm_v_g2:
        _V_CFG = (1 << 16) | V_TDM_CONFIG
        _stride_v_elems = stride_v_seq >> 1
        if const_expr(loop_v_oob_dg1 is not None):
            v_dg1 = loop_v_oob_dg1
        else:
            v_dg1 = Vec.from_elements(
                [
                    fx.Int32(_V_CFG),
                    fx.Int32(128 << 16),
                    fx.Int32(8 << 16),
                    fx.Int32(128 << 16),
                    fx.Int32(8),
                    _stride_v_elems,
                    fx.Int32(0),
                    fx.Int32(0),
                ],
                fx.Int32,
            )
        v_addr = compute_global_addr(
            ptr_V,
            tdm_v_offset,
            wave_id,
            8 * stride_v_seq,
        )
        v_stride_adv = fx.Int64(stride_v_32)
        _v_warp_off = wave_id * (8 * V_ROW_BYTES)
        _v_lds_base = tdm_v_target + _v_warp_off
        v_descs = TDM.build_descs(
            v_dg1,
            v_addr,
            v_stride_adv,
            _v_lds_base,
            LDS_V_SU_P_SIZE,
            CNT_SU,
        )
        tdm_state["v_descs"] = v_descs
        tdm_state["v_desc_idx"] = 0

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

        n_lds = N_LDS_V_PER_MSB if l_type == KV_V else N_LDS_PER_MSB
        kv_tiles_next_raw = [[None] * n_lds for _ in range(NUM_MSB)]

        if const_expr(l_type == KV_K):
            g2_addrs = (
                kv_lds_addrs_next if kv_lds_addrs_next is not None else kv_lds_addrs
            )
        else:
            g2_addrs = kv_lds_addrs

        o_tiles, kv_tiles_next_raw = gemm2_interleaved_stage(
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
            o_rescale_exp_delta=(o_rescale_exp_delta if stage_idx == 0 else None),
        )

        if const_expr(l_type == KV_V):
            v_tiles_paired = Fragment.pair_v_tiles(kv_tiles_next_raw, ty)
        else:
            kv_tiles = Fragment.pair_k_tiles(kv_tiles_next_raw, ty)

    rocdl.sched_barrier(0)

    # Build partial_sp lo+hi flat lists for yield.
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
    """One iteration of the KV loop: unpack -> fmha_pipeline -> pack."""
    lane_id = ctx["lane_id"]
    wave_id = ctx["wave_id"]
    actual_kv_len = ctx["actual_kv_len"]
    stride_k_seq = ctx["stride_k_seq"]
    stride_v_seq = ctx["stride_v_seq"]
    k_offset = ctx["k_offset"]
    v_offset = ctx["v_offset"]
    tile_n_const = ctx["tile_n_const"]
    zero_v8f32 = ctx["zero_v8f32"]
    q_frags = ctx["q_frags"]
    sgpr_state = ctx["sgpr_state"]
    ty = ctx["ty"]

    # ---- Unpack iter_args ----
    o_tiles_flat = [iter_args[i] for i in fx.range_constexpr(16)]
    o_tiles = []
    for d in fx.range_constexpr(NUM_MSB):
        row = []
        for n in fx.range_constexpr(N_PV_WMMA_N):
            idx = d * N_PV_WMMA_N + n
            row.append(set_vgpr_bank(o_tiles_flat[idx], d))
        o_tiles.append(row)

    ia_old_max = [
        set_vgpr_bank(iter_args[16 + i], i) for i in fx.range_constexpr(NUM_MSB)
    ]
    ia_row_sums = [
        set_vgpr_bank(iter_args[20 + i], i) for i in fx.range_constexpr(NUM_MSB)
    ]

    kv_tiles_flat = [iter_args[24 + i] for i in fx.range_constexpr(_KV_SIZE)]
    kv_tiles = []
    for msb in fx.range_constexpr(NUM_MSB):
        row = []
        for k in fx.range_constexpr(N_WMMA_K_TILES):
            ki = msb * N_WMMA_K_TILES + k
            row.append(set_vgpr_bank(kv_tiles_flat[ki], msb))
        kv_tiles.append(row)

    ia_local_max = [
        set_vgpr_bank(iter_args[_OFF_LOCAL_MAX + i], i)
        for i in fx.range_constexpr(NUM_MSB)
    ]
    ia_delta = [
        set_vgpr_bank(iter_args[_OFF_DELTA + i], i) for i in fx.range_constexpr(NUM_MSB)
    ]

    ia_sp_flat = [iter_args[_OFF_SP + i] for i in fx.range_constexpr(CNT_SU * NUM_MSB)]
    prev_su_sp_tiles = []
    for su in fx.range_constexpr(CNT_SU):
        msb_list = []
        for msb in fx.range_constexpr(NUM_MSB):
            si = su * NUM_MSB + msb
            msb_list.append([set_vgpr_bank(ia_sp_flat[si], msb)])
        prev_su_sp_tiles.append(msb_list)

    # Unpack ping-pong bases
    ia_k_cur_base = iter_args[_OFF_PP]
    ia_v_cur_base = iter_args[_OFF_PP + 1]
    ia_k_next_base = iter_args[_OFF_PP + 2]
    ia_v_next_base = iter_args[_OFF_PP + 3]

    # Build per-lane LDS addresses from ping-pong bases
    kv_lds_addrs_cur = build_kv_lds_addrs(
        lane_id,
        ia_k_cur_base,
        ia_v_cur_base,
    )
    kv_lds_addrs_next = build_kv_lds_addrs(
        lane_id,
        ia_k_next_base,
        ia_v_next_base,
    )

    # Unpack partial_sp_pairs: reconstruct v2f32 from separate lo+hi f32 scalars
    ia_partial_sp_lo = [iter_args[_OFF_PSP + i] for i in fx.range_constexpr(_PSP_SIZE)]
    ia_partial_sp_hi = [
        iter_args[_OFF_PSP_HI + i] for i in fx.range_constexpr(_PSP_SIZE)
    ]
    ia_exp_delta = [
        set_vgpr_bank(iter_args[_OFF_PED + i], i) for i in fx.range_constexpr(NUM_MSB)
    ]

    ia_partial_sp_pairs = []
    for m in fx.range_constexpr(NUM_MSB):
        msb_pairs = [
            make_v2f32(
                ia_partial_sp_lo[m * N_SP_PAIRS + i],
                ia_partial_sp_hi[m * N_SP_PAIRS + i],
                m,
            )
            for i in fx.range_constexpr(N_SP_PAIRS)
        ]
        ia_partial_sp_pairs.append(msb_pairs)

    # SP tiles: zero accumulators (fresh each iteration)
    sp_tiles = []
    for msb in fx.range_constexpr(NUM_MSB):
        sp_tiles.append([set_vgpr_bank(zero_v8f32, msb)])

    softmax_state = make_softmax_state(
        ia_old_max,
        ia_local_max,
        ia_delta,
        ia_row_sums,
        sp_pairs_prev=ia_partial_sp_pairs,
    )

    # Build TDM state (zero placeholder for memload=False)
    tdm_state = {
        "v_g0": fx.constant_vector(0, T.vec(4, T.i32)),
        "v_g1": fx.constant_vector(0, T.vec(8, T.i32)),
        "k_g0": fx.constant_vector(0, T.vec(4, T.i32)),
        "k_g1": fx.constant_vector(0, T.vec(8, T.i32)),
        "v_salu_queue": [],
        "k_salu_queue": [],
    }

    # Compute TDM offsets
    tile_idx_i32 = arith.index_cast(T.i32, tile_idx)

    tile_n_stride_v = tile_n_const * stride_v_seq
    cur_v_advance = tile_idx_i32 * tile_n_stride_v
    cur_v_offset = v_offset + cur_v_advance

    next_tile = tile_idx_i32 + 1
    tile_n_stride_k = tile_n_const * stride_k_seq
    next_k_advance = next_tile * tile_n_stride_k
    next_k_offset = k_offset + next_k_advance

    # Per-tile OOB dg1 for K prefetch and V load
    _loop_stride_k_elems = stride_k_seq >> 1
    _loop_stride_v_elems = stride_v_seq >> 1
    _loop_K_CFG_OOB = (1 << 16) | K_TDM_CONFIG
    _loop_V_CFG_OOB = (1 << 16) | V_TDM_CONFIG

    loop_v_remain = actual_kv_len - tile_idx_i32 * tile_n_const
    loop_v_oob_dg1 = TDM.build_oob_dg1_list(
        _loop_V_CFG_OOB,
        128,
        _loop_stride_v_elems,
        loop_v_remain,
        wave_id,
    )
    loop_k_remain = actual_kv_len - next_tile * tile_n_const
    loop_k_oob_dg1 = TDM.build_oob_dg1_list(
        _loop_K_CFG_OOB,
        QK_HDIM,
        _loop_stride_k_elems,
        loop_k_remain,
        wave_id,
        dim0_stride=200,
    )

    # Core fmha_pipeline call
    (
        sp_out,
        kv_out,
        o_tiles,
        su_sp_tiles_out,
        partial_sp_lo_out,
        partial_sp_hi_out,
        partial_ed_out,
    ) = fmha_pipeline_ctx(
        ctx,
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
        ia_exp_delta=ia_exp_delta,
        causal_n_start=causal_n_start,
        loop_k_oob_dg1=loop_k_oob_dg1,
        loop_v_oob_dg1=loop_v_oob_dg1,
    )

    # Pack yield tuple with ping-pong swap
    new_o = []
    for d in fx.range_constexpr(NUM_MSB):
        for n in fx.range_constexpr(N_PV_WMMA_N):
            new_o.append(o_tiles[d][n])

    new_max = [softmax_state["old_max"][i] for i in fx.range_constexpr(NUM_MSB)]
    new_sums = [softmax_state["row_sums"][i] for i in fx.range_constexpr(NUM_MSB)]

    kv_out_flat = []
    for msb in fx.range_constexpr(NUM_MSB):
        for k in fx.range_constexpr(N_WMMA_K_TILES):
            kv_out_flat.append(kv_out[msb][k])

    new_local_max = [softmax_state["local_max"][i] for i in fx.range_constexpr(NUM_MSB)]
    new_delta = [softmax_state["delta"][i] for i in fx.range_constexpr(NUM_MSB)]

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

    new_partial_sp_flat = partial_sp_lo_out + partial_sp_hi_out
    new_exp_delta = [partial_ed_out[m] for m in fx.range_constexpr(NUM_MSB)]

    return (
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


def _ep_finish(
    ctx,
    o_tiles,
    sp_pairs_in,
    exp_delta_rescale,
    v_base_for_pv,
    old_max_in,
    local_max_in,
    delta_in,
    row_sums_in,
    ep_k_cur_base,
):
    ty = ctx["ty"]
    sgpr_state = ctx["sgpr_state"]
    lane_id = ctx["lane_id"]
    scalar_f = ctx["scalar_f"]
    RETURN_LSE = ctx["RETURN_LSE"]
    ptr_LSE = ctx["ptr_LSE"]
    bx = ctx["bx"]
    actual_q_len = ctx["actual_q_len"]
    q_start_tok = ctx["q_start_tok"]
    gdz = ctx["gdz"]
    by = ctx["by"]
    wave_id = ctx["wave_id"]
    o_oob_dim1 = ctx["o_oob_dim1"]
    stride_o_seq = ctx["stride_o_seq"]
    stride_o_head = ctx["stride_o_head"]
    ptr_O = ctx["ptr_O"]
    zero_v8f32 = ctx["zero_v8f32"]
    q_frags = ctx["q_frags"]
    actual_kv_len = ctx["actual_kv_len"]
    stride_v_seq = ctx["stride_v_seq"]

    sfx = make_softmax_state(
        old_max_in,
        local_max_in,
        delta_in,
        row_sums_in,
        sp_pairs_prev=sp_pairs_in,
    )

    # PART2 second half: setup ops + pair_exp+cvt+sum.
    p2ops = Softmax.build_all_part2_ops(
        ty,
        0,
        sp_pairs_in,
        sfx,
        sgpr_state,
    )
    for m in fx.range_constexpr(NUM_MSB):
        for i in fx.range_constexpr(PART2_SETUP_A - 1):  # ops 0..6
            p2ops[m][i]()

    for m in fx.range_constexpr(NUM_MSB):
        for op in p2ops[m][PART2_SPLIT:]:
            op()

    p_tiles = Softmax.build_p_tiles(ty, sfx)

    # O rescale
    for msb in fx.range_constexpr(NUM_MSB):
        edv8 = fx.vector.broadcast(T.vec(8, T.f32), exp_delta_rescale[msb])
        for n in fx.range_constexpr(N_PV_WMMA_N):
            o_tiles[msb][n] = o_tiles[msb][n] * edv8

    # PV_pure
    kv_pv = build_kv_lds_addrs(lane_id, ep_k_cur_base, v_base_for_pv)
    for sp in fx.range_constexpr(2):
        sb = sp * 2
        vr0, vr1 = Fragment.load_v_two_sus(ty, kv_pv, 0, sb, sb + 1)
        o_tiles = pv_gemm_pure(
            ty,
            0,
            sb,
            Fragment.pair_v_tiles(vr0, ty),
            p_tiles[sb],
            o_tiles,
        )
        o_tiles = pv_gemm_pure(
            ty,
            0,
            sb + 1,
            Fragment.pair_v_tiles(vr1, ty),
            p_tiles[sb + 1],
            o_tiles,
        )

    # 4d: div_cvt
    v8f32 = T.vec(8, T.f32)
    v8bf16 = T.vec(8, T.bf16)
    rsf = list(sfx["row_sums"])
    lmf = list(sfx["local_max"])
    for mb in fx.range_constexpr(0, NUM_MSB, 2):
        sm = rsf[mb] + rsf[mb + 1]
        slo = arith.constant(0x76543210, type=T.i32)
        shi = arith.constant(0xFEDCBA98, type=T.i32)
        pm = rocdl_permlanex16(
            ty["f32"],
            sm,
            sm,
            slo,
            shi,
            False,
            False,
        )
        sf = sm + pm
        rsf[mb] = sf
        rsf[mb + 1] = sf
    l2e = 0.6931471805599453
    lse_vals = [None] * NUM_MSB
    for msb in fx.range_constexpr(NUM_MSB):
        mxs = lmf[msb] * scalar_f
        lgs = rocdl.log(ty["f32"], rsf[msb])
        lse_vals[msb] = lgs * l2e + mxs

    # Store LSE to global: ptr_LSE layout (total_q, nheads) fp32
    if const_expr(RETURN_LSE):
        i64_lse = T.i64
        glbpt_lse = glb_ptr_ty()
        lse_base_i64 = ptr_base_i64(ptr_LSE)

        wv_lse = rocdl.wave_id()
        lane_lo_lse = lane_id & 15

        lse_bx128 = bx * 128
        lse_wv32 = wv_lse * WV_SUBQD
        lse_base_row = lse_bx128 + lse_wv32

        for msb_lse in [0, 2]:
            msb_off = 0 if msb_lse == 0 else 16
            seq_pos = lse_base_row + lane_lo_lse + msb_off
            lse_valid = seq_pos < actual_q_len
            _if_lse = scf.IfOp(lse_valid.ir_value())
            with ir.InsertionPoint(_if_lse.then_block):
                lse_tok = q_start_tok + seq_pos
                lse_elem_off = lse_tok * gdz + by
                lse_byte_off = lse_elem_off * 4
                lse_byte_off_i64 = fx.Int64(lse_byte_off)
                lse_addr = lse_base_i64 + lse_byte_off_i64
                lse_ptr = llvm_dialect.inttoptr(glbpt_lse, lse_addr)
                llvm_dialect.store(lse_vals[msb_lse], lse_ptr)
                scf.YieldOp([])
    obf16 = []
    for msb in fx.range_constexpr(NUM_MSB):
        rcp = rocdl.rcp(ty["f32"], rsf[msb])
        rv8 = fx.vector.broadcast(T.vec(8, T.f32), rcp)
        mb16 = []
        for n in fx.range_constexpr(N_PV_WMMA_N):
            mb16.append(
                fx.trunc_f(
                    v8bf16,
                    o_tiles[msb][n] * rv8,
                )
            )
        obf16.append(mb16)

    # Barrier: ensure all waves finish PV before any wave writes D to V_a LDS.
    rocdl.s_barrier_signal(-1)
    rocdl.s_barrier_wait(-1)

    # 4e: TDM store D (VGPR -> LDS -> Global)
    i32t = T.i32
    ldst = lds_ptr_ty()
    v4i32t = T.vec(4, T.i32)
    db32 = extract_lds_base_i32(lds_alloc_v_a.get_base())
    dw_wv = wave_id * LDS_D_WV_SIZE
    dw = db32 + dw_wv
    d_thr = fx.make_layout((16, 2), (1, 16))
    d_crd = idx2crd(lane_id, d_thr)
    llo = arith.index_cast(T.i32, d_crd[0])
    lhi = arith.index_cast(T.i32, d_crd[1])
    loff = llo * TDM_D_TILE_DIM0 + lhi * 16
    for msb in fx.range_constexpr(NUM_MSB):
        for n in fx.range_constexpr(N_PV_WMMA_N):
            ioff = (msb // 2) * 16 * TDM_D_TILE_DIM0 + (msb % 2) * 128 + n * 32
            la = dw + loff + ioff
            llvm_dialect.store(
                fx.vector.bitcast(v4i32t, obf16[msb][n]),
                llvm_dialect.inttoptr(ldst, la),
                volatile_=True,
            )
    emit_void("s_wait_dscnt 0x0")
    wsgpr = rocdl.wave_id()

    i64t = T.i64
    o_tok = q_start_tok + bx * 128 + wsgpr * WV_SUBQD
    o_elem_off = by * stride_o_head + o_tok * stride_o_seq
    o64 = ptr_base_i64(ptr_O)
    boff32 = o_elem_off * 2
    boff64 = fx.Int64(boff32)
    oadr64 = o64 + boff64

    alo, ahi = split_i64_to_lo_hi(oadr64)
    olds2 = extract_lds_base_i32(lds_alloc_v_a.get_base()) + wsgpr * LDS_D_WV_SIZE
    _dg0 = Vec.from_elements(
        [fx.Int32(1), olds2, alo, ahi],
        fx.Int32,
    )
    _g0 = fx.Int32((1 << 16) | 0)
    _g1 = fx.Int32((128 & 0xFFFF) << 16)
    _td1_lo_o = arith.andi(o_oob_dim1, arith.constant(0xFFFF, type=T.i32))
    _g2 = arith.ori(
        arith.shli(_td1_lo_o, arith.constant(16, type=T.i32)),
        arith.constant((128 >> 16) & 0xFFFF, type=T.i32),
    )
    _g3_val = ((32 >> 16) & 0xFFFF) | ((128 & 0xFFFF) << 16)
    _g3 = fx.Int32(_g3_val)
    _g4 = fx.Int32(32 & 0xFFFF)
    _g5 = stride_o_seq
    _g6 = fx.Int32(0)
    _g7 = fx.Int32(0)
    _dg1 = Vec.from_elements(
        [_g0, _g1, _g2, _g3, _g4, _g5, _g6, _g7],
        fx.Int32,
    )
    tdm_ops.tensor_store_2d(tdm_ops.TDMDescriptor2D(_dg0, _dg1))
    tdm_ops.tensor_wait(0)
