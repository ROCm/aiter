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
from .device import (
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
):
    """The launcher (flyc.jit) of one kernel variant; see engine.py for the arguments."""
    g1, g2 = cfg.g1, cfg.g2
    NCTA = n_cu * cfg.cta_per_cu
    NW = g1.m_warp * g1.n_warp
    assert NW == g2.m_warp * g2.n_warp
    BLK = NW * WAVE
    GW = NCTA * NW
    AM = cfg.align_m
    TMAX = tp * mmax
    XB, XS = H // 2, H // 32  # MXFP4 row bytes, e8m0 scale bytes
    LPR = H // WAVE  # elements per lane of a row
    assert LPR % 32 == 0 and E <= 1024 and TOPK <= WAVE
    NB_PER_LANE = LPR // 32  # MX blocks per lane
    wrep2 = g2.tile_m // g2.m_warp // 16
    stage1_act = ACTS[act]
    LDS_BYTES = {6: 50, 5: 60, 4: 72, 3: 100, 2: 158, 1: 300}[cfg.cta_per_cu] * 1024  # of 320 KB per CU
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
    NCH = LPR // 32  # RS / Y column chunks of CW elements (32 per lane)
    CW = H // NCH
    assert CW == 4 * 256  # a lane's 4 accesses of 8 elements, 512 B apart
    RPW = cfg.place_rpw  # routes a wave places per batch
    CH = min(4, RPW)  # A rows a wave copies at once
    QO = cfg.q_once
    assert not QO or (cfg.ar and cfg.local_hist)
    # one route a wave: its x row early
    XEARLY = cfg.ar and cfg.local_hist and RPW == 1 and not QO
    assert RPW <= WAVE and RPW % CH == 0
    name = (
        f"mega_moe_tp1250_h{H}_i{I}_e{E}_k{TOPK}_tp{tp}_mm{mmax}_{act}_{cfg.path}"
        f"_g1m{g1.tile_m}n{g1.tile_n}b{g1.num_buffers}"
        f"_g2m{g2.tile_m}n{g2.tile_n}b{g2.num_buffers}_c{NCTA}"
    )
    if g1.b_th or g2.b_th:
        name += f"_th{g1.b_th}x{g2.b_th}"
    if cfg.b_first:
        name += "_bf"
    if cfg.q_once:
        name += "_qo"
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
    LEXP = LDS_BYTES + 16  # [E] psum (expert ends) for the tasks' expert bisect
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
        task1, b1, _ = make_gemm_a8w4_tdm_task(
            K=H,
            stage1_act=stage1_act,
            stage1_quant_out=1,
            quant_wmma_rep=wrep2,
            row_major_ascale=1,
            num_buffers=min(g1.num_buffers, ceildiv(H, g1.tile_k)),
            a_wait=int(cfg.b_first and cfg.df0 and not cfg.dyn),
            **gemm_kw(g1),
        )
        task2, b2, _ = make_gemm_a8w4_tdm_task(
            K=I,
            stage1_act=0,
            ep_quant_bits=8 if cfg.g2_q8 else 0,
            q8_rows=int(cfg.g2_q8),
            num_buffers=min(g2.num_buffers, ceildiv(I, g2.tile_k)),
            a_wait=int(cfg.b_first and cfg.df1),
            **gemm_kw(g2),
        )
        assert b1 == b2 == BLK

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
            lptr = fx.SharedAllocator().allocate(LDS_BYTES + 16 + E * 4, alignment=1024)._ptr
            lds_slot["ptr"] = lptr
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

            def ids_at(t, k):
                """route k of token t's expert id (AR: the local input)"""
                if fx.const_expr(AR):
                    return ld_i32(ids_in + i64(t * i32(TOPK) + k) * 4)
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
                for i in range(gw, m if not (AR and LH) or QO else 0, GW):
                    src = x_in + i64(i) * (H * 2) + i64(lane) * 64
                    q, scs = [], []
                    for b in range_constexpr(NB_PER_LANE):
                        v4s = [ld_v4(src + (b * WAVE * 64 + j * 16)) for j in range_constexpr(4)]
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
                    for p in range_constexpr(1 if AR else tp):
                        base = mine if AR else peers[p]
                        dq = base + off_xq + dr * XB + i64(lane) * 16
                        ds = base + off_xs + dr * XS + i64(lane)
                        for b in range_constexpr(NB_PER_LANE):
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
                rel_wg()
                gpu.barrier()
                if tid == i32(0):
                    if fx.const_expr(QO):  # this CTA's quantized tokens are out
                        if bid * i32(NW) < m:
                            rel_agent()
                            add_agent(wgc, 1)
                    if (bid * i32(NW) < m) & (notar == i32(1)):  # this CTA pushed rows
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
                if (tid == i32(0)) & (notar == i32(1)):  # every rank's writers done (AGS: ids)
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
                for r in range(
                    (i32(0) if LH else bid * i32(BLK)) + tid, nr, BLK if LH else NCTA * BLK
                ):
                    t = r // i32(TOPK)
                    eid = ids_at(t, r - t * i32(TOPK))
                    if eid >= i32(0):
                        lds_add(L + L_HIST + eid * 4, 1)
                gpu.barrier()
                if fx.const_expr(not LH):
                    for e in range(tid, E, BLK):
                        c = lds_ld(L + L_HIST + e * 4)
                        if c > i32(0):
                            add_agent(ctrl + (CT_CNT * 4) + i64(e) * 4, c)
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
                        sx_c = ld_i32(ctrl + (CT_CNT * 4) + i64(sx_ec) * 4)
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
                if fx.const_expr(DF0):  # every CTA (same values): its own tasks read it
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
                for pb in range(gw, ceildiv(nr, i32(RPW)), GW):
                    r = pb * i32(RPW) + lane
                    if fx.const_expr(XEARLY):  # the route's x row loads go out before
                        # its slot's atomic (they overlap); quantized once both land
                        srcx0 = x_in + i64(fx.min(pb, nr - i32(1)) // i32(TOPK)) * (H * 2) \
                            + i64(lane) * 64
                        xv = [[ld_v4(srcx0 + (b * WAVE * 64 + j * 16)) for j in range_constexpr(4)]
                              for b in range_constexpr(NB_PER_LANE)]
                    pos = i32(-1)
                    ar = i32(0)
                    eidl = i32(0)
                    if (lane < i32(RPW)) & (r < nr):
                        t = r // i32(TOPK)
                        ar = t if AR else arow(t)  # AR: the token (its x row)
                        eid = ids_at(t, r - t * i32(TOPK))
                        eidl = eid
                        if eid >= i32(0):
                            pos = lds_ld(L + L_HIST + eid * 4) + add_agent(
                                ctrl + (CT_FILL * 4) + i64(eid) * 4, 1
                            )
                        st_i32(rpos + i64(r) * 4, pos)
                    for j0 in range_constexpr(0, RPW, CH):
                        pss = [rdlane(pos, j0 + jj) for jj in range_constexpr(CH)]
                        ars = [i64(rdlane(ar, j0 + jj)) for jj in range_constexpr(CH)]
                        if fx.const_expr(AR and LH and not QO):  # route rows straight from x
                            for jj in range_constexpr(CH):
                                drow = i64((pss[jj] >= i32(0)).select(pss[jj], spare))
                                srcx = x_in + ars[jj] * (H * 2) + i64(lane) * 64
                                dq = a1_addr + drow * XB + i64(lane) * 16
                                dsc = a1s_addr + drow * XS + i64(lane)
                                for b in range_constexpr(NB_PER_LANE):
                                    if fx.const_expr(XEARLY):
                                        v4s = xv[b]
                                    else:
                                        v4s = [ld_v4(srcx + (b * WAVE * 64 + j * 16))
                                               for j in range_constexpr(4)]
                                    qb, e8 = quant_fp4_block32(v4s)
                                    st_v4(dq + b * WAVE * 16, qb)
                                    st_i8(dsc + b * WAVE, e8)
                        vs, ss = [], []
                        for jj in range_constexpr(CH if not (AR and LH) or QO else 0):
                            # access cc of a lane: 16 B at cc * 512 + lane * 16 (a wave's
                            # access is 512 contiguous bytes)
                            s0 = mine + off_xq + ars[jj] * XB + i64(lane) * 16
                            vs.append([ld_v4(s0 + cc * 512) for cc in range_constexpr(XB // WAVE // 16)])
                            ss.append(ld_i32(mine + off_xs + ars[jj] * XS + i64(lane) * 4))
                        for jj in range_constexpr(CH if not (AR and LH) or QO else 0):
                            drow = i64((pss[jj] >= i32(0)).select(pss[jj], spare))
                            d0 = a1_addr + drow * XB + i64(lane) * 16
                            for cc in range_constexpr(XB // WAVE // 16):
                                st_v4(d0 + cc * 512, vs[jj][cc])
                            st_i32(a1s_addr + drow * XS + i64(lane) * 4, ss[jj])
                    if fx.const_expr(DF0):  # this batch's rows are in a1: count them
                        rel_wg()
                        if pos >= i32(0):
                            add_agent(placed + i64(eidl) * 4, 1)
                ts(14)
                if fx.const_expr(not DF0):
                    barrier(2)
                ts(4)
                if fx.const_expr(not DF0):
                    if bid == i32(0):
                        for e in range(tid, E, BLK):
                            st_i32(ctrl + (CT_CNT * 4) + i64(e) * 4, 0)
                            st_i32(ctrl + (CT_FILL * 4) + i64(e) * 4, 0)
            # ---------------- G1 / G2: the grouped GEMM tiles ----------------------
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
                        task1(a2, a1, w1, a1s, w1s, psum, a1, a2s, a2, cm, i32(2 * I),
                              swiglu_limit, situ_beta, situ_linear_beta, tq, rows,
                              ehint(tq, NT1), i64(0), i32(0), err)
                        if fx.const_expr(g1.c_sep):
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
                            task2(gout, a2, w2, a2s, w2s, psum, a2, a2, a2, cm, i32(H),
                                  swiglu_limit, situ_beta, situ_linear_beta, t2, rows,
                                  ehint(t2, NT2), cnt_t2, i32(NT1), err)
                        else:
                            if tid == i32(0):  # every G1 tile of this m-tile done
                                spin_agent(cnt_t2, NT1, err, ERR_DF)
                                acq_agent()
                            gpu.barrier()
                            acq_wg()
                            task2(gout, a2, w2, a2s, w2s, psum, a2, a2, a2, cm, i32(H),
                                  swiglu_limit, situ_beta, situ_linear_beta, t2, rows,
                                  ehint(t2, NT2), i64(0), i32(0), err)
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
                    task1(a2, a1, w1, a1s, w1s, psum, a1, a2s, a2, cm, i32(2 * I),
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
                if fx.const_expr(not (DF0 and BF and not DYN)):
                    task1(a2, a1, w1, a1s, w1s, psum, a1, a2s, a2, cm, i32(2 * I),
                          swiglu_limit, situ_beta, situ_linear_beta, tk, rows, ehint(tk, NT1),
                          i64(0), i32(0), err)
                if fx.const_expr(DF1):  # this m-tile's G1 tile done
                    if fx.const_expr(g1.c_sep):
                        tdm_ops.tensor_wait(0)
                    rel_wg()
                    gpu.barrier()
                    if tid == i32(0):
                        rel_agent()
                        add_agent(cnt1 + i64(mtile_of(tk, n_mt, NT1)) * 4, 1)
            ts(5)
            if fx.const_expr(not DF1 and not DYN):
                barrier(3)
            ts(6)
            n2 = n_mt * i32(H // g2.tile_n)
            for tk in range(vbid, n2 if not DYN else 0, NCTA):
                if fx.const_expr(DF1 and BF):  # the tile waits for the m-tile after B
                    task2(gout, a2, w2, a2s, w2s, psum, a2, a2, a2, cm, i32(H),
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
                    task2(gout, a2, w2, a2s, w2s, psum, a2, a2, a2, cm, i32(H),
                          swiglu_limit, situ_beta, situ_linear_beta, tk, rows, ehint(tk, NT2),
                          i64(0), i32(0), err)
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
            if fx.const_expr(DF0 and not DF2):  # every CTA's placement is long done
                if bid == i32(0):
                    for e in range(tid, E, BLK):
                        st_i32(ctrl + (CT_CNT * 4) + i64(e) * 4, 0)
                        st_i32(ctrl + (CT_FILL * 4) + i64(e) * 4, 0)
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
                            wl = ld_f32(mine + off_tw + (i64(ar) * TOPK + i64(lane)) * 4)
                    # a dropped route: row 0 read, weight 0 (no branch)
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
                                for j in range_constexpr(4)
                            ])
                        else:
                            fs.append([ld_v4(src + eoff(cu, j) * 2) for j in range_constexpr(4)])
                    acc = [fx.Float32(0.0)] * 32
                    for k in range_constexpr(TOPK):
                        w = rdlane(wl.bitcast(fx.Int32), k).bitcast(fx.Float32)
                        for j in range_constexpr(4):
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
                        for hh in range_constexpr(2):
                            words, e8 = q8_block(acc[16 * hh : 16 * hh + 16])
                            el = eoff(cu, 2 * hh)
                            st_v4(drow + el, fx.Vector.from_elements(words, fx.Int32))
                            if (lane & i32(1)) == i32(0):
                                st_i8(drow + H + el // 32, e8)
                    else:
                        out = []
                        for q in range_constexpr(16):
                            lo = acc[2 * q].to(fx.BFloat16)
                            hi = acc[2 * q + 1].to(fx.BFloat16)
                            out.append(
                                fx.Vector.from_elements([lo, hi], fx.BFloat16).bitcast(fx.Int32)[0]
                            )
                        for j in range_constexpr(4):
                            st_v4(
                                drow + eoff(cu, j) * 2,
                                fx.Vector.from_elements(out[4 * j : 4 * j + 4], fx.Int32),
                            )
                ts(18)
                rel_wg()
                gpu.barrier()
                if tid == i32(0):
                    if bid * i32(NW) < Tt * i32(NCH):  # this CTA pushed rows
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
                if tid == i32(0):  # every rank's writer CTAs done
                    spin_sys(
                        mine + off_flag + i64(FL_RS * 4) + i64(par) * (LINE * 4),
                        i32(tp) * ceildiv(nw_rs, i32(SYNC_GS)), err, ERR_RS,
                    )
                    acq_sys()
                gpu.barrier()
                acq_wg()
                ts(10)
                if fx.const_expr(DF0 and DF2):
                    # (no barrier 4) past the RS wait every G1 tile is done, and a G1 tile
                    # ran only once its expert's rows were all placed: the routing counts
                    # are dead
                    if bid == i32(0):
                        for e in range(tid, E, BLK):
                            st_i32(ctrl + (CT_CNT * 4) + i64(e) * 4, 0)
                            st_i32(ctrl + (CT_FILL * 4) + i64(e) * 4, 0)
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
                    for p in range_constexpr(tp):
                        srow = mine + off_recv + i64(i32(p * mmax) + i) * (H * 2)
                        if fx.const_expr(PULL):  # rank p's partial of my row, in its arena
                            srow = peers[p] + off_recv + i64(rank * i32(mmax) + i) * (H * 2)
                        if fx.const_expr(AR):  # rank p's partial of token i, in its arena
                            srow = peers[p] + off_recv + i64(i) * (H * 2)
                        if fx.const_expr(Q8):
                            fs.append([ld_v4(srow + eoff(cu, 2 * hh)) for hh in range_constexpr(2)])
                            es.append(
                                [ld_u8(srow + H + eoff(cu, 2 * hh) // 32) for hh in range_constexpr(2)]
                            )
                        else:
                            fs.append([ld_v4(srow + eoff(cu, j) * 2) for j in range_constexpr(4)])
                    return fs, es

                def y_rows(fs, es):
                    """the 4 accesses' bf16 output (4 x i32 each)"""
                    if fx.const_expr(Q8):
                        deq = [
                            [q8_unpack(fs[p][hh], es[p][hh], 4) for hh in range_constexpr(2)]
                            for p in range_constexpr(tp)
                        ]
                    yrs = []
                    for j in range_constexpr(4):
                        if fx.const_expr(Q8):
                            h0 = 8 * (j % 2)
                            acc = deq[0][j // 2][h0 : h0 + 8]
                            for p in range_constexpr(1, tp):
                                acc = [a + b for a, b in zip(acc, deq[p][j // 2][h0 : h0 + 8])]
                        else:
                            acc = bf16x8_f32(fs[0][j])
                            for p in range_constexpr(1, tp):
                                acc = [a + b for a, b in zip(acc, bf16x8_f32(fs[p][j]))]
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
                        yrows = y_rows(*lds_[kk])
                        if oks[kk]:
                            for j in range_constexpr(4):
                                st_v4(
                                    y_out + i64(iis[kk]) * (H * 2) + eoff(cus[kk], j) * 2, yrows[j]
                                )
                                if fx.const_expr(AR2):
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
                    for u in range(gw, (m - mt) * i32(NCH), GW):
                        rr = u // i32(NCH)
                        cu = u - rr * i32(NCH)
                        q_ = rr // mt
                        o_ = (rank + i32(1) + q_) % i32(tp)
                        trow = o_ * mt + (rr - q_ * mt)
                        srcy = peers_sel(o_) + off_yred + i64(trow) * (H * 2)
                        vy = [ld_v4(srcy + eoff(cu, j) * 2) for j in range_constexpr(4)]
                        for j in range_constexpr(4):
                            st_v4(y_out + i64(trow) * (H * 2) + eoff(cu, j) * 2, vy[j])
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
