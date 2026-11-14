# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""The gfx1250 MegaMoE TP kernel (ag_rs): one persistent launch (all CTAs
co-resident) runs, per rank,

  AG   own rows -> MXFP4 (+ routing) pushed into every rank's symmetric arena
  R1   wait every rank's rows; routes per expert (grid histogram)
  R2   expert starts (tile aligned), route placement, A rows gathered in route order
  G1   GEMM1 tiles: gate/up, activation, MXFP4 requant (the gfx1250 TDM grouped GEMM)
  G2   GEMM2 tiles: route rows (bf16)
  RS   per token: weighted sum of its route rows -> bf16 row pushed to its owner rank
  Y    wait every rank's partials; y = their sum

with grid barriers between the local phases and epoch flags between ranks."""

from __future__ import annotations

import functools
import os

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import gpu, range_constexpr
from flydsl.expr.typing import T

from ..tensor_shim import AITER_FLYDSL_KERNARG_PRELOAD, AITER_FLYDSL_KERNARG_PRELOAD_COUNT
from flydsl.expr import tdm_ops

from ..gemm_common_gfx1250 import make_lds_copy_ops, workgroup_barrier
from .gemm_tile import make_gemm_a8w4_tdm_task
from ..mxfp4_preshuffle_gfx1250_tdm import build_gemm_a8w4_tdm_task
from .config import (
    CT_CNT,
    CT_EPOCH,
    CT_ERR,
    CT_FILL,
    ERR_AG,
    SYNC_GS,
    SYNC_NG,
    LINE,
    ERR_BAR,
    ERR_DF,
    ERR_RS,
    FL_AG,
    FL_RS,
    FL_ID,
    FL_Y,
    CT_SYNC,
    CTRL_INTS,
    MAX_TP,
    SY_AG,
    SY_BAR,
    SY_DONE,
    SY_RS,
    SYNC_INTS,
    N_SYNC,
    TS_N,
    WAVE,
    LaunchCfg,
    q_off,
    pl_off,
    E_MAX,
    tile_counts_stride,
)
from .device import _EXP_NOFENCE
from .dense_tile import make_dense_task
from .config import DEADLINE, ERR_LL_GATE, ERR_LL_HIST, ERR_LL_IDS, ERR_LL_ROWS, ERR_LL_TW, ERR_LL_Y
from .device import AGENT, SPIN_SLEEP, gptr, ld_sys64, now
from flydsl.expr import rocdl
from .device import (
    st_sys_wt,
    acq_agent,
    acq_sys,
    add_agent,
    bf16x8_f32,
    grid_barrier,
    i32,
    i64,
    ld_agent,
    ld_f32,
    ld_i32,
    ld_v4,
    lds_add,
    lds_ld,
    lds_st,
    quant_fp4_block32,
    rel_agent,
    rel_wg,
    acq_wg,
    rel_sys,
    spin_agent,
    spin_sys,
    add_sys,
    st_sys,
    st_agent,
    st_i32,
    st_i8,
    st_sys_rel,
    st_v4,
    st_v2,
    ld_v2,
    ld_u8,
    q8_block,
    q8_unpack,
    stamp,
    sync_arrive,
    rdlane,
    uni,
)

ACTS = {"silu": 1, "swiglu": 2, "situv2": 3}


def ceildiv(a, b):
    return (a + b - 1) // b


def fp8x8_f32(v2, e8):
    """8 fp8 e4m3 (2 x i32) and their e8m0 (i32) -> 8 f32: the native unscaled
    convert, then the exact power of two (combine.py's _dequant_pk8)."""
    from flydsl.expr import arith
    from flydsl.expr.rocdl import cvt_scale_pk8_f32_fp8

    u = cvt_scale_pk8_f32_fp8(
        T.vec(8, T.f32), fx.Vector(v2).ir_value(), arith.constant(127), 0
    )
    sc = ((fx.Int32(e8) & i32(0xFF)) << i32(23)).bitcast(fx.Float32)
    vu = fx.Vector(u)
    return [fx.Float32(vu[q]) * sc for q in range(8)]


def mtile_of(tk, n_mt, n_nt):
    """The m-tile of grouped-GEMM task tk (gemm_tile: groups of 16 m-tiles, n
    slowest inside a group; cluster 1x1) of n_mt m-tiles x n_nt n-tiles."""
    G = 16
    bpg = i32(n_nt * G)
    group = tk // bpg
    first = group * i32(G)
    in_group = tk - group * bpg
    gt = fx.min(n_mt - first, i32(G))
    return first + in_group % gt


@functools.cache
def compile_mega_moe_tp_1250(
    *,
    H: int,
    I: int,
    E: int,
    TOPK: int,
    tp: int,
    mmax: int,
    act: str,
    n_cu: int,
    cfg: LaunchCfg,
    Ip: int = 0,
):
    """The launcher (flyc.jit) of one kernel variant; see engine.py for the arguments.
    I is the inter dim the kernel runs (a multiple of 128), Ip <= I the weights' own
    (e.g. 576): their N / K past Ip is never read (gemm_tile b_n / b_k)"""
    Ip = Ip or I
    assert Ip <= I and Ip % 64 == 0 and I % 128 == 0 and I - Ip < 128
    g1, g2 = cfg.g1, cfg.g2
    NCTA = n_cu * cfg.cta_per_cu
    NW = g1.m_warp * g1.n_warp
    assert NW == g2.m_warp * g2.n_warp
    BLK = NW * WAVE
    # Every CTA must be co-resident (grid waits): a wave's VGPRs (measured: ~1.3 x its
    # f32 accumulators (tile_m x tile_n over its waves) + 100; 32x512: 260, 128x256: 424)
    # must fit the 1024 a SIMD has, shared by cta_per_cu * NW / 4 waves (64x1024 or
    # 128x512 tiles at 2 CTAs per CU: the second CTA never became resident, a hang)
    for _g in (g1, g2):
        _acc = _g.tile_m * _g.tile_n // (NW * WAVE)
        _wps = max(1, cfg.cta_per_cu * NW // 4)
        assert _acc * 13 // 10 + 100 <= 1024 // _wps, (
            f"tile {_g.tile_m}x{_g.tile_n}: ~{_acc * 13 // 10 + 100} VGPRs per wave do not fit "
            f"{cfg.cta_per_cu} CTAs per CU ({1024 // _wps} VGPRs per wave)")
    # the non-GEMM phases alone need ~260 VGPRs (measured): from 4 waves per SIMD the
    # compiler must be capped (waves_per_eu), else the last CTA is not resident (hang)
    _wps = cfg.cta_per_cu * NW // 4
    assert _wps <= 3 or cfg.waves_per_eu >= _wps, (
        f"{cfg.cta_per_cu} CTAs per CU need waves_per_eu >= {_wps}")
    GW = NCTA * NW
    AM = cfg.align_m
    TMAX = tp * mmax
    XB, XS = H // 2, H // 32  # MXFP4 row bytes, e8m0 scale bytes
    XSC = (XS + WAVE * 4 - 1) // (WAVE * 4)  # a wave's 4 B-a-lane accesses of a scale row
    assert XS % 4 == 0 and XB % 16 == 0 and H % 512 == 0
    assert E <= 1024 and TOPK <= WAVE
    NBLK = H // 32  # MX blocks of a row
    # lane l quantizes / copies blocks b * WAVE + l; H % 1024 != 0 (e.g. Kimi 3584): the last
    # round is partial (its lanes past the row's blocks load block 0's address, store nothing)
    NB_PER_LANE = ceildiv(NBLK, WAVE)
    PARTB = NBLK % WAVE != 0
    CCN = ceildiv(XB, WAVE * 16)  # 512 B row copies (FP4 row bytes, a wave's 16 B a lane)
    PARTC = XB % (WAVE * 16) != 0
    wrep2 = g2.tile_m // g2.m_warp // 16
    stage1_act = ACTS[act]
    LDS_BYTES = {6: 50, 5: 60, 4: 72, 3: 100, 2: 158, 1: 300}[cfg.cta_per_cu] * 1024  # of 320 KB per CU
    if not cfg.expert_hint:  # no resident psum copy: the GEMM arena gets the rest of the CU's share
        LDS_BYTES = (320 * 1024 // cfg.cta_per_cu - 16) // 1024 * 1024
    else:
        # the resident psum copy (E ints) must fit the CU's share too: every CTA must be
        # co-resident (Kimi E896 at 2 CTAs per CU overflowed 320 KB: barrier timeouts)
        LDS_BYTES = min(LDS_BYTES, (320 * 1024 // cfg.cta_per_cu - 16 - E * 4) // 1024 * 1024)
    _lds_cta = LDS_BYTES + 16 + (E * 4 if cfg.expert_hint else 0)
    assert cfg.cta_per_cu * ((_lds_cta + 1023) // 1024 * 1024) <= 320 * 1024, (
        f"{cfg.cta_per_cu} CTAs x {_lds_cta} B of LDS exceed the CU's 320 KB")
    # LDS of the routing phases (inside the GEMM arena: phases never overlap)
    L_HIST = 0
    L_SCAN0 = L_HIST + E * 4
    L_SCAN1 = L_SCAN0 + E * 4
    L_CNT = L_SCAN1 + E * 4
    assert L_CNT + E * 4 <= LDS_BYTES
    EPT = (E + BLK - 1) // BLK  # experts a thread scans
    DF1 = cfg.df1
    Q8 = cfg.rs_q8
    LH = cfg.local_hist
    BF = cfg.b_first
    EHINT = cfg.expert_hint
    CPC = cfg.cta_per_cu
    SPREAD = cfg.spread
    AR = cfg.ar
    assert not AR or (not cfg.rs_pull and not cfg.ag_split)
    AR2 = AR and cfg.ar_two
    assert not AR2 or not cfg.y_tdm
    assert not cfg.y_tdm or H % 1024 == 0
    assert not cfg.self_place or not PARTB
    DF0 = cfg.df0
    TM1 = g1.tile_m
    G2Q8 = cfg.g2_q8
    AGS = cfg.ag_split
    PULL = cfg.rs_pull
    YU = cfg.y_unroll  # Y units a wave has in flight
    YT = cfg.y_tdm
    YT_T, YC = (8 if Q8 else 4), 1024  # Y TDM tile: rows x columns (LDS budget)
    YT_CK = H // YC
    YT_PB = YC if Q8 else YC * 2  # payload bytes of a tile row
    YT_SB = YC // 32 if Q8 else 0  # e8m0 bytes of a tile row
    YT_SRC = YT_T * (YT_PB + YT_SB)  # one source's tile
    YT_IN = tp * YT_SRC  # an input buffer
    YT_OB = YT_T * YC * 2  # an output buffer
    YT_OUT0 = 2 * YT_IN
    YT_ROUNDS = YT_T * YC // 16 // BLK
    YT_WAIT = (2 if Q8 else 1) * tp + 1  # the prefetch's loads + the last store
    assert not YT or (YT_OUT0 + 2 * YT_OB <= LDS_BYTES and YT_T * YC // 16 % BLK == 0)
    DF2 = cfg.df2
    assert not DF2 or DF1
    # AR two-step: the reduced rows peers pull (Y2) as int8 + e8m0 per 32 (Q8 layout);
    # the owner keeps the dequantized values too, so every rank's output is identical
    YQ8 = cfg.y_q8
    Y2U = cfg.y2_unroll  # Y2 units a wave has in flight
    assert not YQ8 or (AR2 and Q8 and not YT)
    assert H % g2.tile_n == 0, f"G2 tile_n {g2.tile_n} must divide H {H} (else columns are dropped)"
    # a partial last G1 n-tile (2 Ip % tile_n) is right only up to 256 wide (measured: 512
    # over DSv4.1's 1152 gives logits_diff ~0.9)
    assert (2 * Ip) % g1.tile_n == 0 or g1.tile_n <= 256, (
        f"G1 tile_n {g1.tile_n}: a partial last n-tile of 2 x Ip {2 * Ip} needs tile_n <= 256")
    NT2 = H // g2.tile_n  # G2 n-tiles of an m-tile
    DYN = cfg.dyn
    assert not DYN or DF1
    assert not DF1 or g1.tile_m == g2.tile_m
    NT1 = 2 * I // g1.tile_n  # G1 n-tiles of an m-tile
    CM_C = (tp * mmax * TOPK + E * 256 + 255) // 256 * 256  # engine.cm
    assert AM <= 256
    MTS = tile_counts_stride(CM_C)  # ints per counter set
    QOFF = q_off(CM_C)  # the task queue counters in tcnt
    PLOFF = pl_off(CM_C)  # per-expert placed counts (df0)

    def rup(x):
        return (x + i32(AM - 1)) // i32(AM) * i32(AM)

    def expert_of(row0, psum_addr):
        """the expert whose (AM-aligned) rows hold row0: the first e with
        rup(psum[e]) > row0 (psum: an expert's start + count, starts aligned)"""
        lo, hi = i32(0), i32(E - 1)
        for _ in range_constexpr(max(1, (E - 1).bit_length())):
            mid = (lo + hi) // i32(2)
            above = rup(ld_i32(psum_addr + i64(mid) * 4)) > row0
            hi = above.select(mid, hi)
            lo = above.select(lo, mid + i32(1))
        return lo
    # RS / Y column chunks of CW elements: a lane's NJ accesses of 8 elements (bf16: 512 B
    # apart; q8: pairs of 16 consecutive); 1024 when H allows, else 512 (H % 1024 != 0)
    CW = 1024 if H % 1024 == 0 else 512
    NCH = H // CW
    NJ = CW // 256
    NHH = NJ // 2
    RPW = cfg.place_rpw  # routes a wave places per batch
    CH = min(4, RPW)  # A rows a wave copies at once
    QO = cfg.q_once
    SP = cfg.self_place
    WT = cfg.rs_wt
    # WT (rows a peer reads written through at system scope, so their signal needs no
    # system-scope writeback): AR (own arena) and ag_rs (the AG rows and RS partials pushed
    # to the peers, or kept in the own arena under rs_pull); gpu.barrier drains the stores
    # (their acks) before the signal in every case
    SPB = 12  # self_place: routes a thread ranks (<= SPB * BLK routes)
    assert not SP or (cfg.ar and cfg.local_hist and not cfg.dyn and not cfg.df0
                      and cfg.expert_hint)
    assert not QO or (cfg.ar and cfg.local_hist)
    # one route a wave: its x row early
    XEARLY = cfg.ar and cfg.local_hist and RPW == 1 and not QO
    # AR (not LH): no AG quantization pass; placement takes whole tokens, quantizes a
    # token's x row once and stores it to its TOPK slots (no xq round trip)
    QP = cfg.q_place
    DENSE = cfg.dense
    assert not DENSE or QP, "dense tiles: placement by q_place"
    # dense tiles take ~570-700 VGPRs uncapped (measured): from 2 waves per SIMD the
    # compiler must be capped, else the second CTA is not resident (hang)
    assert not DENSE or cfg.cta_per_cu * NW // 4 <= 1 or cfg.waves_per_eu >= cfg.cta_per_cu * NW // 4, (
        "dense tiles at >= 2 waves per SIMD need waves_per_eu")
    assert not QP or (cfg.ar and not cfg.local_hist and not QO and not cfg.self_place)
    LL = cfg.ag_ll
    assert not LL or not (cfg.ar or AGS or QO or SP or QP), "ag_ll: the ag_rs AG only"
    # LL row of global token t in the yred region: the FP4 chunks (a wave's 16 B chunks c
    # (block b = c // WAVE, NLB(b) lanes) at b * WAVE * 32: (w0, ep, w1, ep) at lane * 16,
    # (w2, ep, w3, ep) NLB(b) * 16 B after (a wave's store: contiguous bytes)), the e8m0
    # words (4 blocks a word), ids, weights
    LLS = 2 * XB + 2 * XS
    LLU = 4  # routes a thread's hist polls at once
    RSL = cfg.rs_ll
    UG = cfg.ug
    assert not UG or (Ip == I and g1.tile_m == g2.tile_m == cfg.align_m and not cfg.dense
                      and not cfg.self_place and not cfg.g2_q8), "ug"
    # (AR: each rank's partials in its own arena, every rank's Y polls them remotely)
    assert not RSL or ((LL or (cfg.ar and not cfg.ar_two)) and not cfg.rs_pull and not cfg.rs_q8
                       and not cfg.y_tdm), "rs_ll"

    def NLB(b):
        return min(WAVE, NBLK - b * WAVE)
    LLW = LLS + 8 * TOPK
    RLL = (LLW + 8 * TOPK + 127) // 128 * 128
    assert RPW <= WAVE and RPW % CH == 0
    name = (
        f"mega_moe_tp1250_h{H}_i{I}_e{E}_k{TOPK}_tp{tp}_mm{mmax}_{act}_{cfg.path}"
        f"_g1m{g1.tile_m}n{g1.tile_n}b{g1.num_buffers}"
        f"_g2m{g2.tile_m}n{g2.tile_n}b{g2.num_buffers}_c{NCTA}"
    )
    if g1.b_th or g2.b_th:
        name += f"_th{g1.b_th}x{g2.b_th}"
    if Ip < I:
        name += f"_ip{Ip}"
    if cfg.dense:
        name += f"_dense_k{g1.tile_k}x{g2.tile_k}"
    if cfg.b_first:
        name += "_bf"
    if cfg.q_once:
        name += "_qo"
    if cfg.self_place:
        name += "_sp"
    if cfg.rs_wt:
        name += "_wt"
    if LL:
        name += "_ll"
    if RSL:
        name += "_rsll"
    if UG:
        name += f"_ug{g1.soa}{g2.soa}w{g1.tdm_w}{g2.tdm_w}"
    TS = os.environ.get("MEGA_TS", "0") == "1"  # debug phase stamps
    EXP = os.environ.get("MEGA_EXP", "")  # perf experiments (device.py _EXP)
    # gemmonly: G1 / G2 on the routing a previous launch left (same inputs): every
    # phase before G1 and after barrier 4 is skipped
    GEMMONLY = EXP == "gemmonly"
    if EXP:
        name += "_" + EXP
    if not _EXP_NOFENCE:
        name += "_devwb"
    from .device import SPIN_SLEEP

    if SPIN_SLEEP != 1:
        name += f"_sl{SPIN_SLEEP}"
    if TS:
        name += "_ts"
    Q_LDS = LDS_BYTES  # the claimed task's broadcast word
    # [E] psum (expert ends) for the tasks' expert bisect, past the GEMM arena; without the
    # hint only the scan writes it, in the arena (the routing phase's)
    LEXP = LDS_BYTES + 16 if cfg.expert_hint else 4 * E * 4
    lds_slot = {}

    def lds_arena(n):
        assert n <= LDS_BYTES, f"GEMM tile needs {n} B of LDS > {LDS_BYTES}"
        return lds_slot["ptr"]

    def gemm_kw(g):
        return dict(
            tile_m=g.tile_m,
            tile_n=g.tile_n,
            tile_k=g.tile_k,
            m_warp=g.m_warp,
            n_warp=g.n_warp,
            out_is_f16=0,
            a_is_fp4=1,
            n_experts=E,
            has_bias=0,
            persistent_workers=NCTA,
            lds_arena=lds_arena,
            next_stage_prefetch=g.next_stage_prefetch,
            mma_group=g.mma_group or None,
            mma_first_group=g.mma_first_group or None,
            ds_first_n=None if g.ds_first_n < 0 else g.ds_first_n,
            c_sep=g.c_sep,
            tdm_b_th=g.b_th,
            expert_given=int(cfg.expert_hint),
        )

    @flyc.jit
    def launch(
        w1: fx.Pointer,
        w1s: fx.Pointer,
        w2: fx.Pointer,
        w2s: fx.Pointer,
        a1: fx.Pointer,
        a1s: fx.Pointer,
        a2: fx.Pointer,
        a2s: fx.Pointer,
        gout: fx.Pointer,
        psum: fx.Pointer,
        psum_addr: fx.Int64,
        a1_addr: fx.Int64,
        a1s_addr: fx.Int64,
        gout_addr: fx.Int64,
        x_in: fx.Int64,
        ids_in: fx.Int64,
        tw_in: fx.Int64,
        y_out: fx.Int64,
        ctrl: fx.Int64,
        rpos: fx.Int64,
        tcnt: fx.Int64,
        p0: fx.Int64,
        p1: fx.Int64,
        p2: fx.Int64,
        p3: fx.Int64,
        p4: fx.Int64,
        p5: fx.Int64,
        p6: fx.Int64,
        p7: fx.Int64,
        off_xq: fx.Int64,
        off_xs: fx.Int64,
        off_ids: fx.Int64,
        off_tw: fx.Int64,
        off_recv: fx.Int64,
        off_flag: fx.Int64,
        off_yred: fx.Int64,
        rank: fx.Int32,
        m: fx.Int32,
        cm: fx.Int32,
        swiglu_limit: fx.Float32,
        situ_beta: fx.Float32,
        situ_linear_beta: fx.Float32,
        stream: fx.Stream,
    ):
        ug_lds, ug_meta1, ug_ready1, ug_meta2, ug_ready2 = [None], [None], [None], [None], [None]
        task1g, b1g, _ = (None, BLK, None) if UG else make_gemm_a8w4_tdm_task(
            K=H,
            stage1_act=stage1_act,
            stage1_quant_out=1,
            quant_wmma_rep=wrep2,
            row_major_ascale=1,
            num_buffers=min(g1.num_buffers, ceildiv(H, g1.tile_k)),
            b_n=2 * Ip if Ip < I else 0,
            a_wait=int(cfg.b_first and cfg.df0 and not cfg.dyn),
            **gemm_kw(g1),
        )
        task2g, b2g, _ = (None, BLK, None) if UG else make_gemm_a8w4_tdm_task(
            K=I,
            stage1_act=0,
            ep_quant_bits=8 if cfg.g2_q8 else 0,
            q8_rows=int(cfg.g2_q8),
            num_buffers=min(g2.num_buffers, ceildiv(I, g2.tile_k)),
            b_k=Ip if Ip < I else 0,
            a_wait=int(cfg.b_first and cfg.df1),
            **gemm_kw(g2),
        )
        dtask1, db1 = (make_dense_task(
            K=H, tile_m=g1.tile_m, tile_n=g1.tile_n, tile_k=g1.tile_k, m_warp=g1.m_warp,
            n_warp=g1.n_warp, num_buffers=min(g1.num_buffers, ceildiv(H, g1.tile_k)),
            n_experts=E, lds_arena=lds_arena, stage1=True, expert_given=bool(cfg.expert_hint), b_n=2 * Ip if Ip < I else 0,
        ) if DENSE else (None, None))
        dtask2, db2 = (make_dense_task(
            K=I, tile_m=g2.tile_m, tile_n=g2.tile_n, tile_k=g2.tile_k, m_warp=g2.m_warp,
            n_warp=g2.n_warp, num_buffers=min(g2.num_buffers, ceildiv(Ip, g2.tile_k)),
            n_experts=E, lds_arena=lds_arena, stage1=False, expert_given=bool(cfg.expert_hint), a_row_bytes=I // 2,
            b_k=Ip if Ip < I else 0,
        ) if DENSE else (None, None))
        task1, b1 = (dtask1, db1) if DENSE else (task1g, b1g)
        task2, b2 = (dtask2, db2) if DENSE else (task2g, b2g)
        assert b1 == b2 == BLK
        ug_t1 = ug_t2 = None
        if UG:
            def ug_build(g, K, stage1, meta, ready):
                t_, b_, _, _, lay_ = build_gemm_a8w4_tdm_task(
                    K=K, tile_m=g.tile_m, tile_n=g.tile_n, tile_k=g.tile_k, m_warp=g.m_warp,
                    n_warp=g.n_warp, out_is_f16=0,
                    num_buffers=min(g.num_buffers, ceildiv(K, g.tile_k)), a_is_fp4=1,
                    n_experts=E, stage1_act=stage1_act if stage1 else 0, has_bias=0,
                    stage1_quant_out=int(stage1), quant_wmma_rep=wrep2 if stage1 else 1,
                    cluster_n=1, cluster_m=1, next_stage_prefetch=g.next_stage_prefetch,
                    num_waves_per_tensor_tdm=g.tdm_w, tdm_as_in_prologue=0, tdm_b_th=g.b_th,
                    enable_ep_scatter=0, ep_arena_handle=0, ep_combine_input_offset=0,
                    ep_slot_stride_bytes=0, ep_destination_stride=0, ep_world_size=0,
                    ep_quant_bits=0, row_major_ascale=int(stage1), a_row_stride_bytes=0,
                    a_scale_row_stride_bytes=0, lds_soa_load_interleave=g.soa,
                    persistent_workers=NCTA, rolled_drain=0, a_preshuffled=0, m_align=0,
                    explicit_tile=1, lds_base=ug_lds, a_ready=ready, tile_meta=meta,
                )
                assert b_ == BLK and lay_["arena"] <= LDS_BYTES, (
                    f"ug tile: {b_} threads, {lay_['arena']} B LDS > {LDS_BYTES}")
                return t_
            ug_t1 = ug_build(g1, H, True, ug_meta1, ug_ready1)
            ug_t2 = ug_build(g2, I, False, ug_meta2, ug_ready2)

        @flyc.kernel(name=name, known_block_size=[BLK, 1, 1])
        def kernel(
            w1: fx.Pointer,
            w1s: fx.Pointer,
            w2: fx.Pointer,
            w2s: fx.Pointer,
            a1: fx.Pointer,
            a1s: fx.Pointer,
            a2: fx.Pointer,
            a2s: fx.Pointer,
            gout: fx.Pointer,
            psum: fx.Pointer,
            psum_addr: fx.Int64,
            a1_addr: fx.Int64,
            a1s_addr: fx.Int64,
            gout_addr: fx.Int64,
            x_in: fx.Int64,
            ids_in: fx.Int64,
            tw_in: fx.Int64,
            y_out: fx.Int64,
            ctrl: fx.Int64,
            rpos: fx.Int64,
            tcnt: fx.Int64,
            p0: fx.Int64,
            p1: fx.Int64,
            p2: fx.Int64,
            p3: fx.Int64,
            p4: fx.Int64,
            p5: fx.Int64,
            p6: fx.Int64,
            p7: fx.Int64,
            off_xq: fx.Int64,
            off_xs: fx.Int64,
            off_ids: fx.Int64,
            off_tw: fx.Int64,
            off_recv: fx.Int64,
            off_flag: fx.Int64,
            off_yred: fx.Int64,
            rank: fx.Int32,
            m: fx.Int32,
            cm: fx.Int32,
            swiglu_limit: fx.Float32,
            situ_beta: fx.Float32,
            situ_linear_beta: fx.Float32,
        ):
            tid = fx.Int32(gpu.thread_id("x"))
            bid = fx.Int32(gpu.block_id("x"))
            wave = uni(tid // WAVE)
            lane = tid % WAVE
            lptr = fx.SharedAllocator().allocate(LDS_BYTES + 16 + (E * 4 if EHINT else 0), alignment=1024)._ptr
            lds_slot["ptr"] = lptr
            ug_lds[0] = lptr  # (ug: upstream's tile body runs in this kernel's LDS)
            L = uni(fx.Int32(fx.ptrtoint(lptr)))
            peers = [p0, p1, p2, p3, p4, p5, p6, p7]

            def peer(r):
                v = peers[0]
                for j in range_constexpr(1, tp):
                    v = (r == i32(j)).select(peers[j], v)
                return fx.Int64(v)

            mine = peer(rank)
            peers_sel = peer
            err = ctrl + CT_ERR * 4
            ep = ld_agent(ctrl + CT_EPOCH * 4) + i32(1)  # this launch's epoch

            def sync(k):
                return ctrl + i64(i32(CT_SYNC + k * SYNC_INTS) + par * i32(N_SYNC * SYNC_INTS)) * 4

            def barrier(j):
                grid_barrier(
                    tid, bid, sync(SY_BAR + j - 1), ep, NCTA, err, ERR_BAR << (j - 1)
                )

            ts_base = ctrl + CTRL_INTS * 4 + i64(bid) * (TS_N * 8)

            def ts(j):
                if fx.const_expr(TS):
                    stamp(tid, ts_base + j * 8)

            ts(0)
            par = ep & i32(1)
            cnt1 = tcnt + i64(par * i32(2 * MTS)) * 4
            cnt2 = cnt1 + MTS * 4
            placed = tcnt + i64(PLOFF * 4) + i64(par * i32(E_MAX)) * 4
            wgc = tcnt + (4 * MTS) * 4 + i64(par * i32(3 * SYNC_NG)) * 4
            # the next launch's counters (the other parity), zeroed write-through
            oth = tcnt + i64((i32(1) - par) * i32(2 * MTS)) * 4
            for k in range(bid * i32(BLK) + tid, 2 * MTS, NCTA * BLK):
                st_agent(oth + i64(k) * 4, 0)
            # the routing counters of this launch's parity; the other parity's zeroed
            cntb = ctrl + (CT_CNT * 4) + i64(par * i32(E_MAX)) * 4
            fillb = ctrl + (CT_FILL * 4) + i64(par * i32(E_MAX)) * 4
            for k in range(bid * i32(BLK) + tid, 2 * E, NCTA * BLK):
                kk_ = k // i32(E)
                st_agent(ctrl + (CT_CNT * 4) + i64(kk_ * i32(2 * E_MAX) + (i32(1) - par) * i32(E_MAX)
                                                  + k - kk_ * i32(E)) * 4, 0)
            oths = ctrl + i64(i32(CT_SYNC) + (i32(1) - par) * i32(N_SYNC * SYNC_INTS)) * 4
            for k in range(bid * i32(BLK) + tid, N_SYNC * SYNC_INTS, NCTA * BLK):
                st_agent(oths + i64(k) * 4, 0)
            if fx.const_expr(DF0):
                if bid == i32(2 % NCTA):
                    for k in range(tid, E, BLK):
                        st_agent(tcnt + i64(PLOFF * 4) + i64((i32(1) - par) * i32(E_MAX) + k) * 4, 0)
            if bid == i32(1 % NCTA):
                if tid == i32(0):
                    st_agent(tcnt + i64(QOFF * 4) + i64(i32(1) - par) * (LINE * 4), 0)
                for k in range(tid, 3 * SYNC_NG, BLK):
                    st_agent(
                        tcnt + (4 * MTS) * 4 + i64((i32(1) - par) * i32(3 * SYNC_NG) + k) * 4, 0
                    )
            gw = bid * i32(NW) + wave
            # static GEMM task order by a CU-spread CTA id: consecutive CTAs share a CU
            # (CTA_PER_CU of them), so task i goes to CTA (i % n_cu) * CPC + i // n_cu
            vbid = (bid % i32(CPC)) * i32(NCTA // CPC) + bid // i32(CPC) if SPREAD else bid

            def bok(b):
                """lane's block b * WAVE + lane is in the row (None: always)"""
                if fx.const_expr(not PARTB or b < NB_PER_LANE - 1):
                    return None
                return lane < i32(NBLK - b * WAVE)

            def bofs(b):
                """bf16 byte offset of block b's loads; past the row: block 0's (a valid address)"""
                ok = bok(b)
                return i64(b * WAVE * 64) if ok is None else ok.select(i64(b * WAVE * 64), i64(0))

            def cok(cc):
                """lane's 16 B of row copy cc is in the FP4 row (None: always)"""
                if fx.const_expr(not PARTC or cc < CCN - 1):
                    return None
                return lane < i32((XB - cc * WAVE * 16) // 16)

            def eoff(cu, j):
                """element offset of a lane's access j (8 elements) in RS / Y chunk cu:
                bf16: j * 256 + lane * 8 (a wave's 16 B accesses: 512 contiguous B);
                q8: lane's 16 consecutive elements per access pair (its int8 store:
                16 B, a wave's: 512 contiguous B)"""
                if fx.const_expr(Q8):
                    return i64(cu) * CW + (j // 2) * 512 + i64(lane) * 16 + (j % 2) * 8
                return i64(cu) * CW + j * 256 + i64(lane) * 8
            nw_ag = fx.min(ceildiv(m, i32(NW)), i32(NCTA))
            nw_rs = fx.min(ceildiv(m * i32((1 if AR else tp) * NCH), i32(NW)), i32(NCTA))
            if bid == i32(0):
                if tid < i32(4):  # FL_AG / FL_RS / FL_ID / FL_Y of the other parity
                    st_sys(
                        mine + off_flag + i64(tid * i32(2 * LINE * 4))
                        + i64(i32(1) - par) * (LINE * 4),
                        0,
                    )
            Tt = m if AR else m * i32(tp)  # AR: m is every token
            notar = i32(0 if AR else 1)  # AR: no AG signal / wait

            def arow(t):
                """arena row of global token t (rank t // m, its row t % m)"""
                o = t // m
                return o * i32(mmax) + (t - o * m)

            def ll_wait(addrs, bit):
                """LL items (8 B: a data word, the writer's epoch) at addrs (per lane):
                their system-scope loads issued at once; if one is not there yet, polled
                until each carries this launch's epoch (a timeout sets bit: the site's ERR_LL_*) and loaded
                again. The data words"""
                na_ = len(addrs)
                v_ = [ld_sys64(addrs[a_]) for a_ in range_constexpr(na_)]
                nok = i32(0)
                for a_ in range_constexpr(na_):
                    nok = nok + (fx.Int32(v_[a_] >> 32) != ep).select(i32(1), i32(0))
                lo_ = fx.Vector.from_elements([fx.Int32(v_[a_]) for a_ in range_constexpr(na_)], fx.Int32)
                if nok > i32(0):
                    t0 = now()
                    n = i32(0)
                    alive = fx.Boolean(True)
                    while (nok > i32(0)) & alive:
                        rocdl.s_sleep(SPIN_SLEEP)
                        nok = i32(0)
                        for a_ in range_constexpr(na_):
                            nok = nok + (fx.Int32(ld_sys64(addrs[a_]) >> 32) != ep).select(i32(1), i32(0))
                        n = n + i32(1)
                        if (n & i32(1023)) == i32(0):
                            alive = (now() - t0) < fx.Int64(DEADLINE)
                            acq_sys()  # a long wait: drop cached lines (see spin_sys)
                    if nok > i32(0):
                        fx.atomic_or(gptr(err), fx.Int32(bit), syncscope=AGENT)
                    lo_ = fx.Vector.from_elements(
                        [fx.Int32(ld_sys64(addrs[a_])) for a_ in range_constexpr(na_)], fx.Int32)
                return [lo_[a_] for a_ in range_constexpr(na_)]

            # LL items of this launch's epoch parity (a rank may be a launch ahead)
            llp = off_yred + i64(par * Tt) * (RLL + 4 * H)
            llrs = llp + i64(Tt) * RLL  # RS partials: [sender rank][owner row] 4H B
            if fx.const_expr(AR):  # RS partials of every token pushed to every rank
                llrs = off_yred + i64(par * Tt) * (tp * 4 * H)

            def llrow(t):
                """LL row of global token t (own arena)"""
                return mine + llp + i64(t) * RLL

            def ids_at(t, k):
                """route k of token t's expert id (AR: the local input)"""
                if fx.const_expr(AR):
                    return ld_i32(ids_in + i64(t * i32(TOPK) + k) * 4)
                if fx.const_expr(LL):  # (a timed-out poll's word: clamped, no wild address)
                    e_ = ll_wait([llrow(t) + LLS + i64(k) * 8], ERR_LL_IDS)[0]
                    return ((e_ < i32(0)) | (e_ >= i32(E))).select(i32(-1), e_)
                return ld_i32(mine + off_ids + (i64(arow(t)) * TOPK + i64(k)) * 4)

            if fx.const_expr(not GEMMONLY):  # experiment: GEMM phases only
                # ---------------- AG: own rows -> MXFP4, pushed to every rank ----------
                if fx.const_expr(AGS):
                    # ids / weights first, signalled on their own: routing needs only them
                    for i in range(gw, m, GW):
                        if lane < i32(TOPK):
                            idv = ld_i32(ids_in + i64(i * i32(TOPK) + lane) * 4)
                            twv = ld_f32(tw_in + i64(i * i32(TOPK) + lane) * 4)
                            bad = (idv < i32(0)) | (idv >= i32(E))
                            idv = bad.select(i32(-1), idv)
                            dr = i64(rank * i32(mmax) + i)
                            for p in range_constexpr(tp):
                                st_i32(peers[p] + off_ids + (dr * TOPK + i64(lane)) * 4, idv)
                                st_i32(peers[p] + off_tw + (dr * TOPK + i64(lane)) * 4,
                                       twv.bitcast(fx.Int32))
                    rel_wg()
                    gpu.barrier()
                    if tid == i32(0):
                        if bid * i32(NW) < m:
                            rel_sys()
                            gi = bid // i32(SYNC_GS)
                            gsz = fx.min(nw_ag - gi * i32(SYNC_GS), i32(SYNC_GS))
                            old = add_agent(wgc + i64(i32(2 * SYNC_NG) + gi) * 4, 1)
                            if old + i32(1) == gsz:
                                for p in range_constexpr(tp):
                                    add_sys(peers[p] + off_flag + i64(FL_ID * 4) + i64(par) * (LINE * 4), 1)
                # lane l quantizes MX blocks b * WAVE + l: a wave's store b is 512
                # contiguous bytes of FP4 (and 32 of scales) on the xGMI
                # AR (not LH): every token quantized once into the own arena (no push);
                # barrier 1 orders it before the placement copy
                for i in range(gw, m if not ((AR and LH) or QP) or QO else 0, GW):
                    src = x_in + i64(i) * (H * 2) + i64(lane) * 64
                    q, scs = [], []
                    for b in range_constexpr(NB_PER_LANE):
                        v4s = [ld_v4(src + bofs(b) + j * 16) for j in range_constexpr(4)]
                        qb, e8 = quant_fp4_block32(v4s)
                        q.append(qb)
                        scs.append(e8)
                    dr = i64(i) if AR else i64(rank * i32(mmax) + i)
                    idv = i32(-1)
                    twv = fx.Float32(0.0)
                    if lane < i32(TOPK):
                        idv = ld_i32(ids_in + i64(i * i32(TOPK) + lane) * 4)
                        twv = ld_f32(tw_in + i64(i * i32(TOPK) + lane) * 4)
                        bad = (idv < i32(0)) | (idv >= i32(E))
                        idv = bad.select(i32(-1), idv)
                    if fx.const_expr(LL):  # a lane's 4 blocks' e8m0 as one word (lane 4j's)
                        swd = []
                        for b in range_constexpr(NB_PER_LANE):
                            sv = fx.Int32(scs[b]) & i32(0xFF)
                            for k_ in range_constexpr(1, 4):
                                sv = sv | ((fx.Int32(gpu.shuffle_down(fx.Int32(scs[b]), k_, WAVE))
                                            & i32(0xFF)) << i32(8 * k_))
                            swd.append(sv)
                    for p in range_constexpr(1 if AR else tp):
                        base = mine if AR else peers[p]
                        dq = base + off_xq + dr * XB + i64(lane) * 16
                        ds = base + off_xs + dr * XS + i64(lane)
                        if fx.const_expr(LL):
                            lrow = base + llp + i64(rank * m + i) * RLL
                            for b in range_constexpr(NB_PER_LANE):
                                w_ = fx.Vector(q[b])
                                c16 = lane * i32(16) + i32(b * WAVE * 32)
                                v0 = fx.Vector.from_elements([w_[0], ep, w_[1], ep], fx.Int32)
                                v1 = fx.Vector.from_elements([w_[2], ep, w_[3], ep], fx.Int32)
                                sw = fx.Vector.from_elements([swd[b], ep], fx.Int32)
                                sco = i32(2 * XB + b * WAVE * 2) + lane * i32(2)  # 4 blocks: 8 B
                                if fx.const_expr(bok(b) is None):
                                    st_sys_wt(lrow, c16, v0)
                                    st_sys_wt(lrow, c16 + i32(NLB(b) * 16), v1)
                                    if (lane & i32(3)) == i32(0):
                                        st_sys_wt(lrow, sco, sw)
                                else:
                                    if bok(b):
                                        st_sys_wt(lrow, c16, v0)
                                        st_sys_wt(lrow, c16 + i32(NLB(b) * 16), v1)
                                        if (lane & i32(3)) == i32(0):
                                            st_sys_wt(lrow, sco, sw)
                            if lane < i32(TOPK):
                                st_sys_wt(lrow, i32(LLS) + lane * i32(8),
                                          fx.Vector.from_elements([idv, ep], fx.Int32))
                                st_sys_wt(lrow, i32(LLW) + lane * i32(8),
                                          fx.Vector.from_elements([twv.bitcast(fx.Int32), ep], fx.Int32))
                        if fx.const_expr(WT and not AR and not LL):
                            # rows pushed to the peers, written through (no writeback before
                            # the AG signal): row-uniform bases, the lane offsets in the stores
                            qrow = base + off_xq + dr * XB
                            srow = base + off_xs + dr * XS
                            for b in range_constexpr(NB_PER_LANE):
                                if fx.const_expr(bok(b) is None):
                                    st_sys_wt(qrow, lane * i32(16) + i32(b * WAVE * 16), q[b])
                                    st_sys_wt(srow, lane + i32(b * WAVE), fx.Int8(scs[b]))
                                else:
                                    if bok(b):
                                        st_sys_wt(qrow, lane * i32(16) + i32(b * WAVE * 16), q[b])
                                        st_sys_wt(srow, lane + i32(b * WAVE), fx.Int8(scs[b]))
                            if fx.const_expr(not AGS):
                                if lane < i32(TOPK):
                                    st_sys_wt(base + off_ids + dr * (TOPK * 4), lane * i32(4), idv)
                                    st_sys_wt(base + off_tw + dr * (TOPK * 4), lane * i32(4),
                                              twv.bitcast(fx.Int32))
                        if fx.const_expr(not (WT and not AR) and not LL):
                            for b in range_constexpr(NB_PER_LANE):
                                if fx.const_expr(bok(b) is None):
                                    st_v4(dq + b * WAVE * 16, q[b])
                                    st_i8(ds + b * WAVE, scs[b])
                                else:
                                    if bok(b):
                                        st_v4(dq + b * WAVE * 16, q[b])
                                        st_i8(ds + b * WAVE, scs[b])
                            if fx.const_expr(not AGS and not AR):
                                if lane < i32(TOPK):
                                    st_i32(base + off_ids + (dr * TOPK + i64(lane)) * 4, idv)
                                    st_i32(
                                        base + off_tw + (dr * TOPK + i64(lane)) * 4,
                                        twv.bitcast(fx.Int32),
                                    )
                ts(15)
                if fx.const_expr(not LL):  # (LL: no drain, no signal; readers poll)
                    rel_wg()
                    gpu.barrier()
                if tid == i32(0):
                    if fx.const_expr(QO):  # this CTA's quantized tokens are out
                        if bid * i32(NW) < m:
                            rel_agent()
                            add_agent(wgc, 1)
                    if (bid * i32(NW) < m) & (notar == i32(1) - i32(int(LL))):  # pushed rows
                        if fx.const_expr(not WT):  # (WT: written through, drained above)
                            rel_sys()
                        # the group's last writer signals every rank
                        gi = bid // i32(SYNC_GS)
                        gsz = fx.min(nw_ag - gi * i32(SYNC_GS), i32(SYNC_GS))
                        old = add_agent(wgc + i64(i32(0 * SYNC_NG) + gi) * 4, 1)
                        if old + i32(1) == gsz:
                            for p in range_constexpr(tp):
                                add_sys(peers[p] + off_flag + i64(FL_AG * 4) + i64(par) * (LINE * 4), 1)
                    ts(16)
                    ts(17)
                ts(1)
                # ---------------- R1: wait every rank; routes per expert ----------------
                if fx.const_expr(LL):  # a CTA's readers start once every rank's last ids landed
                    if tid == i32(0):
                        ll_wait([llrow(i32(o_) * m + m - i32(1)) + LLS + (TOPK - 1) * 8
                                 for o_ in range_constexpr(tp)], ERR_LL_GATE)
                if (tid == i32(0)) & (notar == i32(1) - i32(int(LL))):  # every rank's writers done (AGS: ids)
                    spin_sys(
                        mine + off_flag + i64((FL_ID if AGS else FL_AG) * 4) + i64(par) * (LINE * 4),
                        i32(tp) * ceildiv(nw_ag, i32(SYNC_GS)), err, ERR_AG,
                    )
                    acq_sys()
                gpu.barrier()
                acq_wg()
                ts(2)
                for e in range(tid, E, BLK):
                    lds_st(L + L_HIST + e * 4, 0)
                gpu.barrier()
                nr = Tt * i32(TOPK)
                # LH: every CTA counts every route itself (no grid count, no barrier 1)
                # LL: a thread's LLU routes' ids polled at once
                hst = BLK if LH else NCTA * BLK
                for r0 in range(
                    (i32(0) if LH else bid * i32(BLK)) + tid, nr if LL else 0, hst * LLU
                ):
                    rr_ = [fx.min(r0 + i32(u * hst), nr - i32(1)) for u in range_constexpr(LLU)]
                    tt_ = [x_ // i32(TOPK) for x_ in rr_]
                    ev_ = ll_wait([llrow(tt_[u]) + LLS + i64(rr_[u] - tt_[u] * i32(TOPK)) * 8
                                   for u in range_constexpr(LLU)], ERR_LL_HIST)
                    for u in range_constexpr(LLU):
                        if (r0 + i32(u * hst) < nr) & (ev_[u] >= i32(0)) & (ev_[u] < i32(E)):
                            lds_add(L + L_HIST + ev_[u] * 4, 1)
                for r in range(
                    (i32(0) if LH else bid * i32(BLK)) + tid, 0 if LL else nr, hst
                ):
                    t = r // i32(TOPK)
                    eid = ids_at(t, r - t * i32(TOPK))
                    if eid >= i32(0):
                        lds_add(L + L_HIST + eid * 4, 1)
                    if fx.const_expr(SP):  # a dropped route has no row
                        if (eid < i32(0)) & (bid == i32(0)):
                            st_i32(rpos + i64(r) * 4, i32(-1))
                gpu.barrier()
                if fx.const_expr(not LH):
                    for e in range(tid, E, BLK):
                        c = lds_ld(L + L_HIST + e * 4)
                        if c > i32(0):
                            add_agent(cntb + i64(e) * 4, c)
                    ts(12)
                    barrier(1)
                ts(3)
                # ---------------- R2: expert starts, placement, A rows -----------------
                # expert starts: an exclusive scan of the AM-aligned counts. A thread
                # owns EPT consecutive experts; a wave scans its threads' sums by
                # shuffles, one barrier joins the waves' totals
                cs, als = [], []
                for j in range_constexpr(EPT):
                    sx_e = tid * i32(EPT) + i32(j)
                    sx_ec = fx.min(sx_e, i32(E - 1))
                    if fx.const_expr(LH):
                        sx_c = lds_ld(L + L_HIST + sx_ec * 4)
                    else:
                        sx_c = ld_i32(cntb + i64(sx_ec) * 4)
                    sx_c = (sx_e < i32(E)).select(sx_c, i32(0))
                    cs.append(sx_c)
                    als.append(rup(sx_c))
                incl = []
                run = i32(0)
                for j in range_constexpr(EPT):
                    run = run + als[j]
                    incl.append(run)
                wsc = run
                for k in range_constexpr(5):
                    up = fx.Int32(gpu.shuffle_up(wsc, 1 << k, WAVE))
                    wsc = wsc + (lane >= i32(1 << k)).select(up, i32(0))
                if lane == i32(WAVE - 1):
                    lds_st(L + L_SCAN0 + wave * 4, wsc)
                gpu.barrier()
                base = wsc - run  # the threads before this one in its wave
                for w in range_constexpr(NW - 1):
                    base = base + (wave > i32(w)).select(lds_ld(L + L_SCAN0 + w * 4), i32(0))
                for j in range_constexpr(EPT):
                    sx_e = tid * i32(EPT) + i32(j)
                    if sx_e < i32(E):
                        sx_st = base + incl[j] - als[j]
                        sx_c = cs[j]
                        lds_st(L + L_HIST + sx_e * 4, sx_st)
                        lds_st(L + LEXP + sx_e * 4, sx_st + sx_c)  # outside the GEMM arena
                gpu.barrier()
                # psum from LDS: a wave's store is 128 contiguous bytes (every CTA
                # stores it under DF0, so scattered 4-byte stores contend)
                # every CTA (same values) when nothing grid-wide orders it before its own
                # tasks read it (no barrier 2: DF0, self_place)
                if fx.const_expr(DF0 or SP):
                    for pe_ in range(tid, E, BLK):
                        st_i32(psum_addr + i64(pe_) * 4, lds_ld(L + LEXP + pe_ * 4))
                else:
                    if bid == i32(0):
                        for pe0 in range(tid, E, BLK):
                            st_i32(psum_addr + i64(pe0) * 4, lds_ld(L + LEXP + pe0 * 4))
                gpu.barrier()
                ts(13)
                if fx.const_expr(AGS):  # the A rows: every rank's AG rows landed
                    if tid == i32(0):
                        spin_sys(
                            mine + off_flag + i64(FL_AG * 4) + i64(par) * (LINE * 4),
                            i32(tp) * ceildiv(nw_ag, i32(SYNC_GS)), err, ERR_AG,
                        )
                        acq_sys()
                    gpu.barrier()
                    acq_wg()
                # each wave places RPW routes (lane j: route j's slot), then copies their
                # A rows CH at a time, loads first (no branch: a dropped route's row
                # goes to the spare row cm - 1)
                spare = cm - i32(1)
                if fx.const_expr(QO):  # every token quantized (by the writer CTAs)
                    if tid == i32(0):
                        spin_agent(wgc, nw_ag, err, ERR_DF)
                        acq_agent()
                    gpu.barrier()
                    acq_wg()
                for tq in range(gw, Tt if QP else 0, GW):
                    # one token: its x row's loads go out before its slots' atomics
                    srcq = x_in + i64(tq) * (H * 2) + i64(lane) * 64
                    xq_ = [[ld_v4(srcq + bofs(b) + j * 16) for j in range_constexpr(4)]
                           for b in range_constexpr(NB_PER_LANE)]
                    pos = i32(-1)
                    eidl = i32(0)
                    if lane < i32(TOPK):
                        eid = ids_at(tq, lane)
                        eidl = eid
                        if eid >= i32(0):
                            pos = lds_ld(L + L_HIST + eid * 4) + add_agent(
                                fillb + i64(eid) * 4, 1
                            )
                        st_i32(rpos + i64(tq * i32(TOPK) + lane) * 4, pos)
                    qq_ = [quant_fp4_block32(xq_[b]) for b in range_constexpr(NB_PER_LANE)]
                    for k in range_constexpr(TOPK):
                        pk = rdlane(pos, k)
                        drow = i64((pk >= i32(0)).select(pk, spare))
                        dq = a1_addr + drow * XB + i64(lane) * 16
                        dsc = a1s_addr + drow * XS + i64(lane)
                        for b in range_constexpr(NB_PER_LANE):
                            if fx.const_expr(bok(b) is None):
                                st_v4(dq + b * WAVE * 16, qq_[b][0])
                                st_i8(dsc + b * WAVE, qq_[b][1])
                            else:
                                if bok(b):
                                    st_v4(dq + b * WAVE * 16, qq_[b][0])
                                    st_i8(dsc + b * WAVE, qq_[b][1])
                    if fx.const_expr(DF0):  # this token's rows are in a1: count them
                        rel_wg()
                        if pos >= i32(0):
                            add_agent(placed + i64(eidl) * 4, 1)
                for pb in range(gw, ceildiv(nr, i32(RPW)) if not (SP or QP) else 0, GW):
                    r = pb * i32(RPW) + lane
                    if fx.const_expr(XEARLY):  # the route's x row loads go out before
                        # its slot's atomic (they overlap); quantized once both land
                        srcx0 = x_in + i64(fx.min(pb, nr - i32(1)) // i32(TOPK)) * (H * 2) \
                            + i64(lane) * 64
                        xv = [[ld_v4(srcx0 + bofs(b) + j * 16) for j in range_constexpr(4)]
                              for b in range_constexpr(NB_PER_LANE)]
                    pos = i32(-1)
                    ar = i32(0)
                    tg = i32(0)
                    eidl = i32(0)
                    if (lane < i32(RPW)) & (r < nr):
                        t = r // i32(TOPK)
                        tg = t
                        ar = t if AR else arow(t)  # AR: the token (its x row)
                        eid = ids_at(t, r - t * i32(TOPK))
                        eidl = eid
                        if eid >= i32(0):
                            pos = lds_ld(L + L_HIST + eid * 4) + add_agent(
                                fillb + i64(eid) * 4, 1
                            )
                        st_i32(rpos + i64(r) * 4, pos)
                    for j0 in range_constexpr(0, RPW, CH):
                        pss = [rdlane(pos, j0 + jj) for jj in range_constexpr(CH)]
                        ars = [i64(rdlane(ar, j0 + jj)) for jj in range_constexpr(CH)]
                        if fx.const_expr(AR and LH and not QO):  # route rows straight from x
                            for jj in range_constexpr(CH):
                                drow = i64(((pss[jj] >= i32(0)) & (pss[jj] < spare)).select(pss[jj], spare))
                                srcx = x_in + ars[jj] * (H * 2) + i64(lane) * 64
                                dq = a1_addr + drow * XB + i64(lane) * 16
                                dsc = a1s_addr + drow * XS + i64(lane)
                                for b in range_constexpr(NB_PER_LANE):
                                    if fx.const_expr(XEARLY):
                                        v4s = xv[b]
                                    else:
                                        v4s = [ld_v4(srcx + bofs(b) + j * 16)
                                               for j in range_constexpr(4)]
                                    qb, e8 = quant_fp4_block32(v4s)
                                    if fx.const_expr(bok(b) is None):
                                        st_v4(dq + b * WAVE * 16, qb)
                                        st_i8(dsc + b * WAVE, e8)
                                    else:
                                        if bok(b):
                                            st_v4(dq + b * WAVE * 16, qb)
                                            st_i8(dsc + b * WAVE, e8)
                        vs, ss = [], []
                        for jj in range_constexpr(CH if not (AR and LH) or QO else 0):
                            # access cc of a lane: 16 B at cc * 512 + lane * 16 (a wave's
                            # access is 512 contiguous bytes)
                            if fx.const_expr(LL):
                                lr = llrow(rdlane(tg, j0 + jj))
                                its = []
                                for cc in range_constexpr(CCN):
                                    c16 = i64(lane) * 16 + (cc * WAVE * 32 if cok(cc) is None
                                                            else cok(cc).select(i64(cc * WAVE * 32), i64(0)))
                                    for h_ in range_constexpr(4):
                                        its.append(lr + c16 + (h_ // 2) * (NLB(cc) * 16) + (h_ % 2) * 8)
                                for c4 in range_constexpr(XSC):
                                    its.append(lr + 2 * XB + i64(fx.min(lane + i32(c4 * WAVE), i32(XS // 4 - 1))) * 8)
                                ws = ll_wait(its, ERR_LL_ROWS)
                                vs.append([fx.Vector.from_elements(ws[4 * cc : 4 * cc + 4], fx.Int32)
                                           for cc in range_constexpr(CCN)])
                                ss.append([ws[4 * CCN + c4] for c4 in range_constexpr(XSC)])
                            else:
                                s0 = mine + off_xq + ars[jj] * XB + i64(lane) * 16
                                vs.append([ld_v4(s0 + (cc * 512 if cok(cc) is None else cok(cc).select(i64(cc * 512), i64(0))))
                                           for cc in range_constexpr(CCN)])
                                # the e8m0 row (XS bytes): 4 B a lane, WAVE * 4 B per access
                                ss.append([ld_i32(mine + off_xs + ars[jj] * XS
                                                  + i64(fx.min(lane * i32(4) + i32(c4 * WAVE * 4), i32(XS - 4))))
                                           for c4 in range_constexpr(XSC)])
                        for jj in range_constexpr(CH if not (AR and LH) or QO else 0):
                            drow = i64(((pss[jj] >= i32(0)) & (pss[jj] < spare)).select(pss[jj], spare))
                            d0 = a1_addr + drow * XB + i64(lane) * 16
                            for cc in range_constexpr(CCN):
                                if fx.const_expr(cok(cc) is None):
                                    st_v4(d0 + cc * 512, vs[jj][cc])
                                else:
                                    if cok(cc):
                                        st_v4(d0 + cc * 512, vs[jj][cc])
                            for c4 in range_constexpr(XSC):
                                if fx.const_expr((c4 + 1) * WAVE * 4 <= XS):
                                    st_i32(a1s_addr + drow * XS + i64(lane) * 4 + c4 * WAVE * 4,
                                           ss[jj][c4])
                                else:  # the row's tail (H % 4096)
                                    if lane * i32(4) + i32(c4 * WAVE * 4) < i32(XS):
                                        st_i32(a1s_addr + drow * XS + i64(lane) * 4 + c4 * WAVE * 4,
                                               ss[jj][c4])
                    if fx.const_expr(DF0):  # this batch's rows are in a1: count them
                        rel_wg()
                        if pos >= i32(0):
                            add_agent(placed + i64(eidl) * 4, 1)
                ts(14)
                if fx.const_expr(not DF0 and not SP):
                    barrier(2)
                ts(4)
            # ---------------- G1 / G2: the grouped GEMM tiles ----------------------
            def self_place(tk_):
                """SP: G1 task tk_'s A rows. Its expert's routes in token order (rank of a
                route: a block scan of per-thread counts over SPB-route runs), the m-tile's
                16 of them quantized from x into a1 (n-tile 0 also writes their rpos)"""
                sp_mt = mtile_of(tk_, n_mt, NT1)
                sp_row0 = sp_mt * i32(TM1)
                sp_e = ehint(tk_, NT1)
                sp_st = (sp_e > i32(0)).select(
                    rup(lds_ld(L + LEXP + fx.max(sp_e - i32(1), i32(0)) * 4)), i32(0))
                sp_k0 = sp_row0 - sp_st
                sp_nv = fx.min(lds_ld(L + LEXP + sp_e * 4) - sp_row0, i32(TM1))  # valid rows
                sp_gr = tk_ // i32(NT1 * 16)
                sp_ig = tk_ - sp_gr * i32(NT1 * 16)
                sp_nt = sp_ig // fx.min(n_mt - sp_gr * i32(16), i32(16))
                sp_b = ceildiv(nr, i32(BLK))
                sp_ms, sp_ts = [], []
                sp_c = i32(0)
                for q in range_constexpr(SPB):
                    sp_r = tid * sp_b + i32(q)
                    sp_ok = (i32(q) < sp_b) & (sp_r < nr)
                    sp_rr = fx.min(sp_r, nr - i32(1))
                    sp_t = sp_rr // i32(TOPK)
                    sp_m = sp_ok & (ids_at(sp_t, sp_rr - sp_t * i32(TOPK)) == sp_e)
                    sp_ms.append(sp_m)
                    sp_ts.append(sp_t)
                    sp_c = sp_c + sp_m.select(i32(1), i32(0))
                # exclusive block scan of the per-thread counts
                sp_w = sp_c
                for k in range_constexpr(5):
                    sp_up = fx.Int32(gpu.shuffle_up(sp_w, 1 << k, WAVE))
                    sp_w = sp_w + (lane >= i32(1 << k)).select(sp_up, i32(0))
                if lane == i32(WAVE - 1):
                    lds_st(L + L_SCAN0 + wave * 4, sp_w)
                gpu.barrier()
                sp_rank = sp_w - sp_c
                for w in range_constexpr(NW - 1):
                    sp_rank = sp_rank + (wave > i32(w)).select(lds_ld(L + L_SCAN0 + w * 4), i32(0))
                for q in range_constexpr(SPB):
                    sp_j = sp_rank - sp_k0
                    if sp_ms[q] & (sp_j >= i32(0)) & (sp_j < i32(TM1)):
                        lds_st(L + L_CNT + sp_j * 4, sp_ts[q])
                        if sp_nt == i32(0):
                            st_i32(rpos + i64(tid * sp_b + i32(q)) * 4, sp_st + sp_rank)
                    sp_rank = sp_rank + sp_ms[q].select(i32(1), i32(0))
                gpu.barrier()
                # each wave quantizes rows wave, wave + NW, ... (2 rows' loads at once)
                for j0 in range_constexpr(0, TM1 // NW, 2):
                    sp_js = [wave + i32((j0 + jj) * NW) for jj in range_constexpr(2)]
                    sp_tok = [lds_ld(L + L_CNT + fx.min(jx, sp_nv - i32(1)) * 4) for jx in sp_js]
                    sp_xv = [[[ld_v4(x_in + i64(tq) * (H * 2) + i64(lane) * 64 + (b * WAVE * 64 + j * 16))
                               for j in range_constexpr(4)] for b in range_constexpr(NB_PER_LANE)]
                             for tq in sp_tok]
                    for jj in range_constexpr(2):
                        if sp_js[jj] < sp_nv:
                            sp_row = i64(sp_row0 + sp_js[jj])
                            for b in range_constexpr(NB_PER_LANE):
                                qb, e8 = quant_fp4_block32(sp_xv[jj][b])
                                st_v4(a1_addr + sp_row * XB + i64(lane) * 16 + b * WAVE * 16, qb)
                                st_i8(a1s_addr + sp_row * XS + i64(lane) + b * WAVE, e8)
                gpu.barrier()  # (waits the stores) the tile's A loads read them

            def ug_task(ut, meta, ready, TM, NT, c_, a_, b_, sa_, sb_, map_, bias_, qs_, erm_,
                        m_, n_, sl_, sb1_, sb2_, tk_, rows_, eh_, w_addr, w_tgt, err_):
                """ug: task tk_ as an explicit tile of upstream's body. Its m- / n-tile by
                gemm_tile's order (groups of 16 m-tiles, n slowest: the completion counts
                index the same m-tiles), its expert and valid rows (tile_meta), and the
                dataflow wait the A owners run after the B prologue (a_ready)"""
                n_mt_ = ceildiv(rows_, i32(TM))
                bpg = i32(NT * 16)
                grp = tk_ // bpg
                first = grp * i32(16)
                ing = tk_ - grp * bpg
                gt = fx.min(n_mt_ - first, i32(16))
                mt = first + ing % gt
                nt = ing // gt
                row0 = mt * i32(TM)
                ex = eh_ if EHINT else expert_of(row0, psum_addr)
                ex = fx.min(fx.max(ex, i32(0)), i32(E - 1))
                if fx.const_expr(EHINT):
                    end_ = lds_ld(L + LEXP + ex * 4)
                else:
                    end_ = ld_i32(psum_addr + i64(ex) * 4)
                meta[0] = (ex, fx.min(end_ - row0, i32(TM)))

                def ready_fn():
                    if fx.const_expr(w_addr is not None):
                        spin_agent(w_addr, w_tgt, err_, ERR_DF)
                        acq_agent()

                ready[0] = ready_fn
                ut(c_, a_, b_, sa_, sb_, map_, bias_, qs_, erm_, m_, n_, sl_, sb1_, sb2_, mt, nt)

            def T1(*a_):
                if fx.const_expr(UG):
                    return ug_task(ug_t1, ug_meta1, ug_ready1, g1.tile_m, NT1, *a_)
                if fx.const_expr(a_[-3] is None):
                    a_ = a_[:-3] + (i64(0), i32(0), a_[-1])
                return task1(*a_)

            def T2(*a_):
                if fx.const_expr(UG):
                    return ug_task(ug_t2, ug_meta2, ug_ready2, g2.tile_m, NT2, *a_)
                if fx.const_expr(a_[-3] is None):
                    a_ = a_[:-3] + (i64(0), i32(0), a_[-1])
                return task2(*a_)

            def ehint(tk_, n_nt):
                """the task's expert (gemm_tile's bisect, over the LDS psum copy)"""
                if not EHINT:
                    return i32(-1)
                row0 = mtile_of(tk_, n_mt, n_nt) * i32(g1.tile_m)
                lo, hi = i32(0), i32(E)
                for _ in range_constexpr(max(1, (max(2, E) - 1).bit_length() + 1)):
                    mid = (lo + hi) >> i32(1)
                    midc = (mid < i32(E - 1)).select(mid, i32(E - 1))
                    right = lds_ld(L + LEXP + midc * 4) <= row0
                    lo = right.select(mid + i32(1), lo)
                    hi = right.select(hi, mid)
                return lo
            rows = ld_i32(psum_addr + (E - 1) * 4)
            n_mt = ceildiv(rows, i32(g1.tile_m))
            n1 = n_mt * i32(NT1)
            if fx.const_expr(DYN):
                # one queue: tasks [0, n1) G1, [n1, n1 + n2) G2. A G2 task is claimed
                # only once every G1 task is (claimed by a running CTA): no deadlock
                n2q = n_mt * i32(H // g2.tile_n)
                ntot = n1 + n2q
                qc = tcnt + i64(QOFF * 4) + i64(par) * (LINE * 4)
                if tid == i32(0):
                    lds_st(L + Q_LDS, add_agent(qc, 1))
                gpu.barrier()
                tq = uni(lds_ld(L + Q_LDS))
                gpu.barrier()
                while tq < ntot:
                    nxt = i32(0)
                    if tid == i32(0):  # the next claim, in flight during this task
                        nxt = add_agent(qc, 1)
                    if tq < n1:
                        if fx.const_expr(DF0):
                            if tid == i32(0):
                                e_mt = expert_of(mtile_of(tq, n_mt, NT1) * i32(TM1), psum_addr)
                                st0 = (e_mt > i32(0)).select(
                                    rup(ld_i32(psum_addr + i64(fx.max(e_mt - i32(1), i32(0))) * 4)),
                                    i32(0),
                                )
                                spin_agent(placed + i64(e_mt) * 4,
                                           ld_i32(psum_addr + i64(e_mt) * 4) - st0, err, ERR_DF)
                                acq_agent()
                            gpu.barrier()
                            acq_wg()
                        T1(a2, a1, w1, a1s, w1s, psum, a1, a2s, a2, cm, i32(2 * I),
                              swiglu_limit, situ_beta, situ_linear_beta, tq, rows,
                              ehint(tq, NT1), None, None, err)
                        if fx.const_expr(g1.c_sep or UG):
                            tdm_ops.tensor_wait(0)
                        rel_wg()
                        gpu.barrier()
                        if tid == i32(0):
                            rel_agent()
                            add_agent(cnt1 + i64(mtile_of(tq, n_mt, NT1)) * 4, 1)
                    else:
                        t2 = tq - n1
                        cnt_t2 = cnt1 + i64(mtile_of(t2, n_mt, H // g2.tile_n)) * 4
                        if fx.const_expr(BF):  # the tile waits for the m-tile after B
                            T2(gout, a2, w2, a2s, w2s, psum, a2, a2, a2, cm, i32(H),
                                  swiglu_limit, situ_beta, situ_linear_beta, t2, rows,
                                  ehint(t2, NT2), cnt_t2, i32(NT1), err)
                        else:
                            if tid == i32(0):  # every G1 tile of this m-tile done
                                spin_agent(cnt_t2, NT1, err, ERR_DF)
                                acq_agent()
                            gpu.barrier()
                            acq_wg()
                            T2(gout, a2, w2, a2s, w2s, psum, a2, a2, a2, cm, i32(H),
                                  swiglu_limit, situ_beta, situ_linear_beta, t2, rows,
                                  ehint(t2, NT2), None, None, err)
                        if fx.const_expr(DF2):  # this m-tile's G2 tile done
                            tdm_ops.tensor_wait(0)
                            rel_wg()
                            gpu.barrier()
                            if tid == i32(0):
                                rel_agent()
                                add_agent(cnt2 + i64(mtile_of(t2, n_mt, NT2)) * 4, 1)
                    if tid == i32(0):
                        lds_st(L + Q_LDS, nxt)
                    gpu.barrier()
                    tq = uni(lds_ld(L + Q_LDS))
                    gpu.barrier()
            for tk in range(vbid, n1 if not DYN else 0, NCTA):
                if fx.const_expr(DF0 and BF and not DYN):  # the tile waits for its rows after B
                    bf_e = expert_of(mtile_of(tk, n_mt, NT1) * i32(TM1), psum_addr)
                    bf_s = (bf_e > i32(0)).select(
                        rup(ld_i32(psum_addr + i64(fx.max(bf_e - i32(1), i32(0))) * 4)),
                        i32(0),
                    )
                    T1(a2, a1, w1, a1s, w1s, psum, a1, a2s, a2, cm, i32(2 * I),
                          swiglu_limit, situ_beta, situ_linear_beta, tk, rows, ehint(tk, NT1),
                          placed + i64(bf_e) * 4, ld_i32(psum_addr + i64(bf_e) * 4) - bf_s, err)
                elif fx.const_expr(DF0):
                    if tid == i32(0):
                        e_mt = expert_of(mtile_of(tk, n_mt, NT1) * i32(TM1), psum_addr)
                        st0 = (e_mt > i32(0)).select(
                            rup(ld_i32(psum_addr + i64(fx.max(e_mt - i32(1), i32(0))) * 4)),
                            i32(0),
                        )
                        spin_agent(placed + i64(e_mt) * 4,
                                   ld_i32(psum_addr + i64(e_mt) * 4) - st0, err, ERR_DF)
                        acq_agent()
                    gpu.barrier()
                    acq_wg()
                if fx.const_expr(SP):
                    self_place(tk)
                if fx.const_expr(not (DF0 and BF and not DYN)):
                    T1(a2, a1, w1, a1s, w1s, psum, a1, a2s, a2, cm, i32(2 * I),
                          swiglu_limit, situ_beta, situ_linear_beta, tk, rows, ehint(tk, NT1),
                          None, None, err)
                if fx.const_expr(DF1):  # this m-tile's G1 tile done
                    if fx.const_expr(g1.c_sep or UG):
                        tdm_ops.tensor_wait(0)
                    rel_wg()
                    gpu.barrier()
                    if tid == i32(0):
                        rel_agent()
                        add_agent(cnt1 + i64(mtile_of(tk, n_mt, NT1)) * 4, 1)
            ts(5)
            if fx.const_expr(not DF1 and not DYN):
                if fx.const_expr(UG):  # the G1 tiles' TDM output stores
                    tdm_ops.tensor_wait(0)
                barrier(3)
            ts(6)
            n2 = n_mt * i32(H // g2.tile_n)
            for tk in range(vbid, n2 if not DYN else 0, NCTA):
                if fx.const_expr(DF1 and BF):  # the tile waits for the m-tile after B
                    T2(gout, a2, w2, a2s, w2s, psum, a2, a2, a2, cm, i32(H),
                          swiglu_limit, situ_beta, situ_linear_beta, tk, rows, ehint(tk, NT2),
                          cnt1 + i64(mtile_of(tk, n_mt, H // g2.tile_n)) * 4, i32(NT1), err)
                else:
                    if fx.const_expr(DF1):  # wait for every G1 tile of this m-tile
                        if tid == i32(0):
                            mt2 = mtile_of(tk, n_mt, H // g2.tile_n)
                            spin_agent(cnt1 + i64(mt2) * 4, NT1, err, ERR_DF)
                            acq_agent()
                        gpu.barrier()
                        acq_wg()
                    T2(gout, a2, w2, a2s, w2s, psum, a2, a2, a2, cm, i32(H),
                          swiglu_limit, situ_beta, situ_linear_beta, tk, rows, ehint(tk, NT2),
                          None, None, err)
                if fx.const_expr(DF2):
                    tdm_ops.tensor_wait(0)
                    rel_wg()
                    gpu.barrier()
                    if tid == i32(0):
                        rel_agent()
                        add_agent(cnt2 + i64(mtile_of(tk, n_mt, NT2)) * 4, 1)
            ts(7)
            tdm_ops.tensor_wait(0)  # the last task's output store (c_sep)
            if fx.const_expr(not DF2):
                barrier(4)
            ts(8)
            if fx.const_expr(not GEMMONLY):  # experiment: GEMM phases only
                # ---------------- RS: topk sums pushed to the owner rank ----------------
                # unit (token t, column chunk cu of CW elements): lane k < TOPK fetches
                # route k's slot and weight, then all TOPK rows' loads are in flight at
                # once; access j of a lane is 16 B at cu * CW * 2 + j * 512 + lane * 16, so
                # a wave's store is 512 contiguous bytes (one xGMI burst, not 32 pieces)
                for u in range(gw, Tt * i32(NCH), GW):
                    t = u // i32(NCH)
                    cu = u - t * i32(NCH)
                    o = i32(0) if AR else t // m
                    i = t if AR else t - o * m
                    ar = o * i32(mmax) + i
                    psl = i32(-1)
                    wl = fx.Float32(0.0)
                    if lane < i32(TOPK):
                        psl = ld_i32(rpos + i64(t * i32(TOPK) + lane) * 4)
                        if fx.const_expr(AR):
                            wl = ld_f32(tw_in + i64(t * i32(TOPK) + lane) * 4)
                        else:
                            if fx.const_expr(LL):
                                wl = ll_wait([llrow(t) + LLW + i64(lane) * 8], ERR_LL_TW)[0].bitcast(fx.Float32)
                            else:
                                wl = ld_f32(mine + off_tw + (i64(ar) * TOPK + i64(lane)) * 4)
                    # a dropped route: row 0 read, weight 0 (no branch); a slot past the
                    # routed rows (never written: a stale rpos) clamped
                    psl = fx.min(psl, cm - i32(1))
                    if fx.const_expr(DF2):  # every G2 tile of this token's routes' m-tiles
                        if (lane < i32(TOPK)) & (psl >= i32(0)):
                            spin_agent(cnt2 + i64(psl // i32(g2.tile_m)) * 4, NT2, err, ERR_DF)
                        acq_agent()
                    wl = (psl >= i32(0)).select(wl, fx.Float32(0.0))
                    psl = fx.max(psl, i32(0))
                    fs = []
                    for k in range_constexpr(TOPK):
                        srow_k = rdlane(psl, k)
                        if fx.const_expr(EXP == "rs_contig"):  # experiment: contiguous rows
                            srow_k = t * i32(TOPK) + i32(k)
                        src = gout_addr + i64(srow_k) * (H * 2)
                        if fx.const_expr(G2Q8):  # 8 fp8 + their block's e8m0 per access
                            fs.append([
                                (ld_v2(src + eoff(cu, j)), ld_u8(src + H + eoff(cu, j) // 32))
                                for j in range_constexpr(NJ)
                            ])
                        else:
                            fs.append([ld_v4(src + eoff(cu, j) * 2) for j in range_constexpr(NJ)])
                    acc = [fx.Float32(0.0)] * (NJ * 8)
                    for k in range_constexpr(TOPK):
                        w = rdlane(wl.bitcast(fx.Int32), k).bitcast(fx.Float32)
                        for j in range_constexpr(NJ):
                            if fx.const_expr(G2Q8):
                                f = fp8x8_f32(fs[k][j][0], fs[k][j][1])
                            else:
                                f = bf16x8_f32(fs[k][j])
                            for q in range_constexpr(8):
                                acc[j * 8 + q] = acc[j * 8 + q] + w * f[q]
                    dst_rank = peer(o)
                    if fx.const_expr(EXP == "rs_local"):  # experiment: RS rows to own arena
                        dst_rank = mine
                    drow = dst_rank + off_recv + i64(rank * i32(mmax) + i) * (H * 2)
                    if fx.const_expr(PULL):  # own arena, slot [owner][row]: owners read it
                        drow = mine + off_recv + i64(o * i32(mmax) + i) * (H * 2)
                    if fx.const_expr(AR):  # own arena, slot t: every rank reads it
                        drow = mine + off_recv + i64(t) * (H * 2)
                    if fx.const_expr(Q8):
                        # 16 int8 per lane at element offsets (a wave's store: 512
                        # contiguous bytes); block e8m0s after the H data bytes, by the
                        # block's first (even) lane
                        for hh in range_constexpr(NHH):
                            words, e8 = q8_block(acc[16 * hh : 16 * hh + 16])
                            el = eoff(cu, 2 * hh)
                            if fx.const_expr(WT):
                                st_sys_wt(drow, el, fx.Vector.from_elements(words, fx.Int32))
                                if (lane & i32(1)) == i32(0):
                                    st_sys_wt(drow, H + el // 32, fx.Int8(e8))
                            else:
                                st_v4(drow + el, fx.Vector.from_elements(words, fx.Int32))
                                if (lane & i32(1)) == i32(0):
                                    st_i8(drow + H + el // 32, e8)
                    else:
                        out = []
                        for q in range_constexpr(NJ * 4):
                            lo = acc[2 * q].to(fx.BFloat16)
                            hi = acc[2 * q + 1].to(fx.BFloat16)
                            out.append(
                                fx.Vector.from_elements([lo, hi], fx.BFloat16).bitcast(fx.Int32)[0]
                            )
                        # RSL: access j's 16 B as LL items (w0, ep, w1, ep) at j * 1024 +
                        # lane * 16, (w2, ep, w3, ep) 512 B after, of chunk cu's 4 CW bytes
                        # (AR: to every rank, slot [rank][t]: each rank's Y polls locally)
                        for p_ in range_constexpr(tp if AR else 1):
                            rsl_row = dst_rank + llrs + i64(rank * m + i) * (4 * H)
                            if fx.const_expr(AR):
                                rsl_row = peers[p_] + llrs + i64(rank * m + t) * (4 * H)
                            for j in range_constexpr(NJ if RSL else 0):
                                c_ = cu * i32(CW * 4) + i32(j * 1024) + lane * i32(16)
                                st_sys_wt(rsl_row, c_, fx.Vector.from_elements(
                                    [out[4 * j], ep, out[4 * j + 1], ep], fx.Int32))
                                st_sys_wt(rsl_row, c_ + i32(512), fx.Vector.from_elements(
                                    [out[4 * j + 2], ep, out[4 * j + 3], ep], fx.Int32))
                        for j in range_constexpr(0 if RSL else NJ):
                            if fx.const_expr(WT):
                                st_sys_wt(drow, eoff(cu, j) * 2,
                                          fx.Vector.from_elements(out[4 * j : 4 * j + 4], fx.Int32))
                            else:
                                st_v4(
                                    drow + eoff(cu, j) * 2,
                                    fx.Vector.from_elements(out[4 * j : 4 * j + 4], fx.Int32),
                                )
                ts(18)
                if fx.const_expr(not RSL):  # (RSL: no drain, no signal; Y polls)
                    rel_wg()
                    gpu.barrier()
                if tid == i32(0):
                    if (bid * i32(NW) < Tt * i32(NCH)) & (i32(int(RSL)) == i32(0)):  # pushed rows
                        if fx.const_expr(not WT):  # (WT: written through, waited above)
                            rel_sys()
                        # the group's last writer signals every rank
                        gi = bid // i32(SYNC_GS)
                        gsz = fx.min(nw_rs - gi * i32(SYNC_GS), i32(SYNC_GS))
                        old = add_agent(wgc + i64(i32(1 * SYNC_NG) + gi) * 4, 1)
                        if old + i32(1) == gsz:
                            for p in range_constexpr(tp):
                                add_sys(peers[p] + off_flag + i64(FL_RS * 4) + i64(par) * (LINE * 4), 1)
                    ts(19)
                    ts(20)
                ts(9)
                # ---------------- Y: own rows = sum of every rank's partial ------------
                if (tid == i32(0)) & (i32(int(RSL)) == i32(0)):  # every rank's writer CTAs done
                    spin_sys(
                        mine + off_flag + i64(FL_RS * 4) + i64(par) * (LINE * 4),
                        i32(tp) * ceildiv(nw_rs, i32(SYNC_GS)), err, ERR_RS,
                    )
                    acq_sys()
                gpu.barrier()
                acq_wg()
                ts(10)
                # unit (row i, column chunk cu): the RS layout, every rank's partial loaded
                # at once
                # Y by TDM (YT): helpers (no dynamic control flow; the loop is below)
                if fx.const_expr(True):
                    lds_load_b32, _ = make_lds_copy_ops(32)
                    lds_load_b128, lds_store_b128 = make_lds_copy_ops(128)
                    p8s = fx.PointerType.get(
                        elem_ty=fx.Int8.ir_type, address_space=fx.AddressSpace.Shared, alignment=16
                    )
                    i8g = fx.PointerType.get(
                        elem_ty=fx.Int8.ir_type, address_space=fx.AddressSpace.Global, alignment=16
                    )

                    def gview(base, off, shape, stride):
                        return fx.Tensor(fx.make_view(base + off, fx.make_layout(shape, stride)))

                    def lview(off, shape, stride):
                        return fx.Tensor(
                            fx.make_view(fx.recast_iter(p8s, lptr + off), fx.make_layout(shape, stride))
                        )

                    srcs = []
                    for p in range_constexpr(tp):
                        if fx.const_expr(PULL):
                            a_ = peers[p] + off_recv + i64(rank * i32(mmax)) * (H * 2)
                        else:
                            a_ = mine + off_recv + i64(i32(p * mmax)) * (H * 2)
                        srcs.append(fx.inttoptr(i8g, a_))
                    ydst = fx.inttoptr(i8g, y_out)
                    n_rt = ceildiv(m, i32(YT_T))
                    y_total = n_rt * i32(YT_CK)
                    y_last = y_total - i32(1)
                    in_idx0 = fx.index_cast(T.index, fx.ptrtoint(lptr))

                    def issue_load(work, buf, live):
                        work = (work > y_last).select(y_last, work)
                        rt = work // i32(YT_CK)
                        ck = work - rt * i32(YT_CK)
                        i0 = rt * i32(YT_T)
                        left = m - i0
                        oob = (left > i32(YT_T)).select(i32(YT_T), left)
                        oob = live.select(oob, i32(1))
                        for p in range_constexpr(tp):
                            gt = gview(srcs[p], i64(i0) * (H * 2) + i64(ck) * YT_PB, (YT_T, YT_PB),
                                       (H * 2, 1))
                            at = fx.rocdl.make_tdm_atom(gt, [oob, None], strides=[H * 2, None],
                                                        num_warps=NW)
                            fx.copy(at, gt, lview(buf * YT_IN + p * YT_SRC, (YT_T, YT_PB), (YT_PB, 1)))
                            if fx.const_expr(Q8):
                                gs = gview(srcs[p], i64(i0) * (H * 2) + H + i64(ck) * YT_SB,
                                           (YT_T, YT_SB), (H * 2, 1))
                                ats = fx.rocdl.make_tdm_atom(gs, [oob, None], strides=[H * 2, None],
                                                             num_warps=NW)
                                fx.copy(ats, gs, lview(buf * YT_IN + p * YT_SRC + YT_T * YT_PB,
                                                       (YT_T, YT_SB), (YT_SB, 1)))

                    def reduce_tile(buf):
                        for r in range_constexpr(YT_ROUNDS):
                            slot = tid + i32(r * BLK)
                            row = slot // i32(YC // 16)
                            e16 = (slot - row * i32(YC // 16)) * i32(16)
                            acc = None
                            for p in range_constexpr(tp):
                                sb = buf * YT_IN + p * YT_SRC
                                if fx.const_expr(Q8):
                                    pay = fx.Vector(lds_load_b128(in_idx0, sb + row * YT_PB + e16))
                                    sc_i = e16 // i32(32)
                                    dw = fx.Int32(lds_load_b32(
                                        in_idx0, sb + YT_T * YT_PB + row * YT_SB + (sc_i // i32(4)) * i32(4)
                                    )[0])
                                    e8 = (dw >> ((sc_i % i32(4)) * i32(8))) & i32(0xFF)
                                    f = q8_unpack(pay, e8, 4)
                                else:
                                    f = []
                                    for j in range_constexpr(2):
                                        f += bf16x8_f32(
                                            fx.Vector(lds_load_b128(in_idx0, sb + row * YT_PB + e16 * 2 + j * 16))
                                        )
                                acc = f if acc is None else [a + b for a, b in zip(acc, f)]
                            for j in range_constexpr(2):
                                words = []
                                for q in range_constexpr(4):
                                    words.append(
                                        fx.Vector.from_elements(
                                            [acc[8 * j + 2 * q].to(fx.BFloat16),
                                             acc[8 * j + 2 * q + 1].to(fx.BFloat16)],
                                            fx.BFloat16,
                                        ).bitcast(fx.Int32)[0]
                                    )
                                lds_store_b128(
                                    in_idx0, YT_OUT0 + buf * YT_OB + row * (YC * 2) + e16 * 2 + j * 16,
                                    fx.Vector.from_elements(words, fx.Int32).ir_value(),
                                )

                    def store_tile(work, buf):
                        rt = work // i32(YT_CK)
                        ck = work - rt * i32(YT_CK)
                        i0 = rt * i32(YT_T)
                        left = m - i0
                        gt = gview(ydst, i64(i0) * (H * 2) + i64(ck) * (YC * 2), (YT_T, YC * 2),
                                   (H * 2, 1))
                        at = fx.rocdl.make_tdm_atom(
                            gt, [(left > i32(YT_T)).select(i32(YT_T), left), None],
                            strides=[H * 2, None], num_warps=NW,
                        )
                        fx.copy(at, lview(YT_OUT0 + buf * YT_OB, (YT_T, YC * 2), (YC * 2, 1)), gt)


                # YU units per wave at a time: every unit's (remote, under PULL) loads are
                # in flight together; a unit past the end reloads the first one, unstored
                mt = m // i32(tp)  # AR2: this rank's rows [rank * mt, (rank + 1) * mt)
                n_y = mt * i32(NCH) if AR2 else m * i32(NCH)
                y_row0 = rank * mt if AR2 else i32(0)

                def y_loads(i, cu):
                    fs, es = [], []
                    if fx.const_expr(RSL):  # every rank's LL items of row i, chunk cu
                        its = []
                        for p in range_constexpr(tp):
                            lr = mine + llrs + i64(i32(p) * m + i) * (4 * H) + i64(cu) * (CW * 4) \
                                + i64(lane) * 16
                            for j in range_constexpr(NJ):
                                for h_ in range_constexpr(4):
                                    its.append(lr + j * 1024 + (h_ // 2) * 512 + (h_ % 2) * 8)
                        ws = ll_wait(its, ERR_LL_Y)
                        for p in range_constexpr(tp):
                            fs.append([fx.Vector.from_elements(ws[(p * NJ + j) * 4 : (p * NJ + j) * 4 + 4],
                                                               fx.Int32) for j in range_constexpr(NJ)])
                    for p in range_constexpr(0 if RSL else tp):
                        srow = mine + off_recv + i64(i32(p * mmax) + i) * (H * 2)
                        if fx.const_expr(PULL):  # rank p's partial of my row, in its arena
                            srow = peers[p] + off_recv + i64(rank * i32(mmax) + i) * (H * 2)
                        if fx.const_expr(AR):  # rank p's partial of token i, in its arena
                            srow = peers[p] + off_recv + i64(i) * (H * 2)
                        if fx.const_expr(Q8):
                            fs.append([ld_v4(srow + eoff(cu, 2 * hh)) for hh in range_constexpr(NHH)])
                            es.append(
                                [ld_u8(srow + H + eoff(cu, 2 * hh) // 32) for hh in range_constexpr(NHH)]
                            )
                        else:
                            fs.append([ld_v4(srow + eoff(cu, j) * 2) for j in range_constexpr(NJ)])
                    return fs, es

                def y_rows(fs, es, f32_out=None):
                    """the 4 accesses' bf16 output (4 x i32 each); f32_out: gets the
                    4 accesses' f32 sums (8 each)"""
                    if fx.const_expr(Q8):
                        deq = [
                            [q8_unpack(fs[p][hh], es[p][hh], 4) for hh in range_constexpr(NHH)]
                            for p in range_constexpr(tp)
                        ]
                    yrs = []
                    for j in range_constexpr(NJ):
                        if fx.const_expr(Q8):
                            h0 = 8 * (j % 2)
                            acc = deq[0][j // 2][h0 : h0 + 8]
                            for p in range_constexpr(1, tp):
                                acc = [a + b for a, b in zip(acc, deq[p][j // 2][h0 : h0 + 8])]
                        else:
                            acc = bf16x8_f32(fs[0][j])
                            for p in range_constexpr(1, tp):
                                acc = [a + b for a, b in zip(acc, bf16x8_f32(fs[p][j]))]
                        if f32_out is not None:
                            f32_out.append(acc)
                        out = []
                        for q in range_constexpr(4):
                            out.append(
                                fx.Vector.from_elements(
                                    [acc[2 * q].to(fx.BFloat16), acc[2 * q + 1].to(fx.BFloat16)],
                                    fx.BFloat16,
                                ).bitcast(fx.Int32)[0]
                            )
                        yrs.append(fx.Vector.from_elements(out, fx.Int32))
                    return yrs

                if fx.const_expr(YT):
                    # combine.py's pipeline: prime the first tile twice (a uniform wait
                    # count), then per trip: prefetch the next, wait for this one, sum,
                    # store
                    issue_load(bid, i32(0), bid <= y_last)
                    issue_load(bid, i32(0), bid <= y_last)
                    for work in range(bid, y_total, NCTA):
                        ybuf = (work // i32(NCTA)) % i32(2)
                        ynxt = work + i32(NCTA)
                        issue_load(ynxt, i32(1) - ybuf, ynxt <= y_last)
                        tdm_ops.tensor_wait(YT_WAIT)
                        workgroup_barrier()
                        reduce_tile(ybuf)
                        workgroup_barrier()
                        store_tile(work, ybuf)
                    tdm_ops.tensor_wait(0)
                for u0 in range(gw, n_y if not YT else 0, GW * YU):
                    us_ = [u0 + i32(kk * GW) for kk in range_constexpr(YU)]
                    oks = [uu < n_y for uu in us_]
                    us_ = [oks[kk].select(us_[kk], u0) for kk in range_constexpr(YU)]
                    iis = [uu // i32(NCH) for uu in us_]
                    cus = [us_[kk] - iis[kk] * i32(NCH) for kk in range_constexpr(YU)]
                    iis = [y_row0 + ii for ii in iis]
                    lds_ = [y_loads(iis[kk], cus[kk]) for kk in range_constexpr(YU)]
                    for kk in range_constexpr(YU):
                        if fx.const_expr(YQ8):
                            # quantize the sums (every lane: the block amax is a lane-pair
                            # shuffle), keep their dequantized values as this rank's rows
                            accs = []
                            y_rows(*lds_[kk], f32_out=accs)
                            yq = []
                            yrows = []
                            for hh in range_constexpr(NHH):
                                words, e8 = q8_block(accs[2 * hh] + accs[2 * hh + 1])
                                yq.append((words, e8))
                                dq = q8_unpack(words, e8, 4)
                                for s_ in range_constexpr(2):
                                    o4 = []
                                    for q in range_constexpr(4):
                                        o4.append(
                                            fx.Vector.from_elements(
                                                [dq[8 * s_ + 2 * q].to(fx.BFloat16),
                                                 dq[8 * s_ + 2 * q + 1].to(fx.BFloat16)],
                                                fx.BFloat16,
                                            ).bitcast(fx.Int32)[0]
                                        )
                                    yrows.append(fx.Vector.from_elements(o4, fx.Int32))
                        else:
                            yrows = y_rows(*lds_[kk])
                        if fx.const_expr(YQ8):
                            if oks[kk]:
                                yb = mine + off_yred + i64(iis[kk]) * (H * 2)
                                for j in range_constexpr(NJ):
                                    st_v4(y_out + i64(iis[kk]) * (H * 2) + eoff(cus[kk], j) * 2,
                                          yrows[j])
                                for hh in range_constexpr(NHH):
                                    words, e8 = yq[hh]
                                    el = eoff(cus[kk], 2 * hh)
                                    if fx.const_expr(WT):
                                        st_sys_wt(yb, el, fx.Vector.from_elements(words, fx.Int32))
                                        if (lane & i32(1)) == i32(0):
                                            st_sys_wt(yb, H + el // 32, fx.Int8(e8))
                                    else:
                                        st_v4(yb + el, fx.Vector.from_elements(words, fx.Int32))
                                        if (lane & i32(1)) == i32(0):
                                            st_i8(yb + H + el // 32, e8)
                        elif oks[kk]:
                            for j in range_constexpr(NJ):
                                st_v4(
                                    y_out + i64(iis[kk]) * (H * 2) + eoff(cus[kk], j) * 2, yrows[j]
                                )
                                if fx.const_expr(AR2 and WT):
                                    st_sys_wt(mine + off_yred + i64(iis[kk]) * (H * 2),
                                              eoff(cus[kk], j) * 2, yrows[j])
                                elif fx.const_expr(AR2):
                                    st_v4(
                                        mine + off_yred + i64(iis[kk]) * (H * 2) + eoff(cus[kk], j) * 2,
                                        yrows[j],
                                    )
                if fx.const_expr(AR2):
                    # every rank's reduced rows ready, then pull the other ranks' rows
                    nw_y = fx.min(ceildiv(mt * i32(NCH), i32(NW)), i32(NCTA))
                    ts(21)
                    rel_wg()
                    gpu.barrier()
                    if tid == i32(0):
                        if bid * i32(NW) < mt * i32(NCH):
                            if fx.const_expr(not WT):
                                rel_sys()
                            gi = bid // i32(SYNC_GS)
                            gsz = fx.min(nw_y - gi * i32(SYNC_GS), i32(SYNC_GS))
                            old = add_agent(wgc + i64(i32(2 * SYNC_NG) + gi) * 4, 1)
                            if old + i32(1) == gsz:
                                for p in range_constexpr(tp):
                                    add_sys(peers[p] + off_flag + i64(FL_Y * 4) + i64(par) * (LINE * 4), 1)
                        spin_sys(
                            mine + off_flag + i64(FL_Y * 4) + i64(par) * (LINE * 4),
                            i32(tp) * ceildiv(nw_y, i32(SYNC_GS)), err, ERR_RS,
                        )
                        acq_sys()
                    gpu.barrier()
                    acq_wg()
                    ts(22)
                    n_y2 = (m - mt) * i32(NCH)
                    for u0 in range(gw, n_y2, GW * Y2U):
                        # Y2U units per wave: all their (remote) loads in flight at once
                        uu_ = [u0 + i32(kk * GW) for kk in range_constexpr(Y2U)]
                        ok_ = [uu < n_y2 for uu in uu_]
                        uu_ = [ok_[kk].select(uu_[kk], u0) for kk in range_constexpr(Y2U)]
                        trows, cus_, vys = [], [], []
                        for kk in range_constexpr(Y2U):
                            u = uu_[kk]
                            rr = u // i32(NCH)
                            cu = u - rr * i32(NCH)
                            q_ = rr // mt
                            o_ = (rank + i32(1) + q_) % i32(tp)
                            trow = o_ * mt + (rr - q_ * mt)
                            srcy = peers_sel(o_) + off_yred + i64(trow) * (H * 2)
                            if fx.const_expr(YQ8):
                                vys.append((
                                    [ld_v4(srcy + eoff(cu, 2 * hh)) for hh in range_constexpr(NHH)],
                                    [ld_u8(srcy + H + eoff(cu, 2 * hh) // 32) for hh in range_constexpr(NHH)],
                                ))
                            else:
                                vys.append([ld_v4(srcy + eoff(cu, j) * 2) for j in range_constexpr(NJ)])
                            trows.append(trow)
                            cus_.append(cu)
                        for kk in range_constexpr(Y2U):
                            if fx.const_expr(YQ8):
                                qw, qe = vys[kk]
                                vy = []
                                for hh in range_constexpr(NHH):
                                    dq = q8_unpack(qw[hh], qe[hh], 4)
                                    for s_ in range_constexpr(2):
                                        o4 = []
                                        for q in range_constexpr(4):
                                            o4.append(
                                                fx.Vector.from_elements(
                                                    [dq[8 * s_ + 2 * q].to(fx.BFloat16),
                                                     dq[8 * s_ + 2 * q + 1].to(fx.BFloat16)],
                                                    fx.BFloat16,
                                                ).bitcast(fx.Int32)[0]
                                            )
                                        vy.append(fx.Vector.from_elements(o4, fx.Int32))
                            else:
                                vy = vys[kk]
                            if ok_[kk]:
                                for j in range_constexpr(NJ):
                                    st_v4(y_out + i64(trows[kk]) * (H * 2) + eoff(cus_[kk], j) * 2, vy[j])
                ts(11)
            # ---------------- done: the last CTA advances the epoch ----------------
            # (orders no data: only every CTA's read of the epoch precedes it)
            if tid == i32(0):
                if sync_arrive(sync(SY_DONE), bid, ep, NCTA) == i32(1):
                    # an atomic, not a plain store: it must not sit in this XCD's L2
                    # when the next launch's CTAs on the other XCDs read it
                    add_agent(ctrl + CT_EPOCH * 4, 1)

        kernel(
            w1, w1s, w2, w2s, a1, a1s, a2, a2s, gout, psum, psum_addr, a1_addr,
            a1s_addr, gout_addr, x_in, ids_in, tw_in, y_out, ctrl, rpos,
            tcnt, p0, p1, p2,
            p3, p4, p5, p6, p7, off_xq, off_xs, off_ids, off_tw, off_recv, off_flag,
            off_yred,
            rank, m, cm, swiglu_limit, situ_beta, situ_linear_beta,
        ).launch(grid=(NCTA, 1, 1), block=(BLK, 1, 1), stream=stream)

    # the grouped GEMM's own launch hints (mxfp4_preshuffle_gfx1250_tdm)
    launch.compile_hints["llvm_options"] = {
        "amdgpu-expert-scheduling-mode": bool(cfg.expert_sched),
        "amdgpu-kernarg-preload": AITER_FLYDSL_KERNARG_PRELOAD,
        "amdgpu-kernarg-preload-count": AITER_FLYDSL_KERNARG_PRELOAD_COUNT,
    }
    if cfg.waves_per_eu:  # cap VGPRs so that many waves (CTAs per CU) fit a SIMD
        launch.compile_hints["waves_per_eu"] = cfg.waves_per_eu
    return launch, NCTA
