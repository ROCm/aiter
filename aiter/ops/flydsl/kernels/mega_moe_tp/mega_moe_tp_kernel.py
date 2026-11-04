# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Fused TP MegaMoE kernel (a4w4, gfx950): the fused tail norm, the wave roles, the
kernel entry and its launcher. The device code is built per instance by the build_*
parts (communication, gemm, schedule) over one shared KernelCtx."""

from __future__ import annotations

import functools
import inspect
import struct

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import gpu, range_constexpr, rocdl
from flydsl.expr import math as fmath
from flydsl.expr.typing import T

from ..communication_ops_utils import (
    atomic_add_agent,
    fence_release,
    load_i32_global_agent,
)
from ..kernels_common import ceildiv
from ..mxfp4_gemm_common import _e8m0_from_amax
from .common import (
    AUX_SYS,
    MAX_TP,
    V4I,
    alive,
    amax,
    before,
    bf16x8_to_f32,
    bld,
    bst,
    f32_bits_add,
    f32_bits_max,
    fp4_pack,
    fp8x4_pack,
    fp8x8_decode,
    g_add_agent,
    g_ld_sys,
    g_st_sys,
    i4x8_pack,
    i4x8_unpack,
    i8x4_pack,
    i8x4_unpack,
    i32,
    lds_atomic_add,
    lds_ld_acq,
    lds_ld_i32,
    lds_st,
    now,
    pack_bf16x8,
    rsrc,
    sum_live,
    uni,
    wait_vm,
    wave_red,
)
from .communication import build_communication
from .gemm import build_bigm_unit, build_gemm
from .mega_moe_tp_config import (
    C_ECNT,
    C_EPOCH,
    C_ERR,
    C_FILL,
    C_OFF,
    C_PH,
    C_SB,
    C_SBH,
    C_TK,
    C_TNL,
    C_TURN,
    C_UN,
    C_XW,
    CTRL_TNC,
    ERR_YAG,
    FLAG_TN,
    NKIND,
    NQ,
    NT,
    NTT,
    NW,
    P_RB,
    P_RF,
    P_RQ,
    P_S1,
    P_S2,
    P_S3,
    P_U,
    P_UB,
    QE,
    QID,
    QQ,
    QS,
    QUEUES,
    QX,
    QXS,
    SRT,
    TN_MAX,
    UNIT_ROWS,
    UQ,
    ZQ,
    i8_row_bytes,
    kernel_config,
)
from .schedule import build_schedule

__all__ = ["compile_bigm_kernel", "compile_mega_moe_tp"]


@functools.cache
def compile_mega_moe_tp(
    *,
    H: int,
    I: int,
    TOPK: int,
    MT: int,
    TMAX: int,
    E: int,
    act: str = "silu",
    situ_beta: float = 1.0,
    situ_linear_beta: float = 1.0,
    swiglu_limit: float | None = None,
    route_fp8: bool = False,
    agr: int = 1,
    tp: int = MAX_TP,
    ar: bool = False,
    nab: int = 4,
    npp: int = 1,
    ll_rs: bool = False,
    ll_route: bool = False,
    comm_bf16: bool = False,
    ag8: bool = False,
    rch: int = 16,
    nch: int = 0,
    a8: bool = False,
    lb: bool = False,
    lbq: int = 1,
    tn: int = 0,
    tn_eps: float = 1e-6,
    tn_gemma: bool = True,
    tn_local: bool = False,
    lbpf: bool = False,
    lbp: int = 0,
):
    """Build the launcher of one (shape, variant) instance; LaunchCfg (host) picks them.

    Units are planned in the kernel over the active experts: an expert routed by R
    tokens runs as ceil(R / rch) row chunks (chunk c takes its tokens t with
    t % chunks == c; nch bounds the chunk table), each in P equal inter pieces; piece k
    writes its GEMM2 partial into route region k.
    ar: every rank holds the same input and routing of all tokens (each rank quantizes
        and all-gathers its 1/tp of the rows) and gets every token's sum (all-reduce);
        else (ag_rs) each rank holds its own rows and gets their sums (reduce-scatter).
    ll_rs: LL ReduceScatter packets; ll_route: LL route rows too.
    comm_bf16: bf16 (not MXFP8) ReduceScatter / all-reduce partials (no LL).
    ag8: all-reduce whose second hop carries MXFP8 rows (else bf16).
    a8: activations as MXFP8 (E4M3 + E8M0 per 32) instead of MXFP4.
    lb: (rch = 16 * MT) large batches in two phases without partial sums:
        GEMM1 units (row chunk x inter piece) export their quantized intermediate; GEMM2
        units (column group x row chunk) take the full inter dim and are claimed by
        whichever CTA is free; a column chunk is pushed once all its units are done.
        lbq: output column chunks per GEMM2 unit; lbpf: the A loader wave prefetches the
        GEMM2 units; lbp: forced GEMM1 inter pieces (0: picked per batch).
    tn: (ag_rs, lb) the next layer's input fused in: each output row r of this rank,
        once final, becomes res_out[r] = y[r] + res_in[r] and the per-token FP8 quant of
        GemmaRMSNorm(res_out[r]; w, tn_eps) (RMSNorm unless tn_gemma), gathered by every
        rank into qall / sall; tn 2: the bf16 normed rows too (qall after the FP8 rows).
    """
    kc = kernel_config(**locals())
    kc.add(build_communication(kc))
    kc.add(build_gemm(kc))
    kc.add(build_schedule(kc))
    AG8, ARLL, DLL, L_CTL, L_RING, LBPF = kc.get("AG8 ARLL DLL L_CTL L_RING LBPF")
    LDS_BYTES, MLL, NCHA, TN, TPC = kc.get("LDS_BYTES MLL NCHA TN TPC")
    a_loader, ag8_convert, ag_send = kc.get("a_loader ag8_convert ag_send")
    _ag_send_meta, _ag_send_meta_ll = kc.get("_ag_send_meta _ag_send_meta_ll")
    claim_addr, comm_help, comm_wave = kc.get("claim_addr comm_help comm_wave")
    compute_units, const_expr, ctrl_at = kc.get("compute_units const_expr ctrl_at")
    (TNL,) = kc.get("TNL")
    _drop_stale, _dyn_zero, fin_key = kc.get("_drop_stale _dyn_zero fin_key")
    fin_owned, fin_split, finish = kc.get("fin_owned fin_split finish")
    init_lds, layout_tag, lb_finish = kc.get("init_lds layout_tag lb_finish")
    lb_g2_prefetch, name, peer_rs = kc.get("lb_g2_prefetch name peer_rs")
    peer_sel, poll_loop, poll_zero = kc.get("peer_sel poll_loop poll_zero")
    push_dll, quant_chunks, signal_loop = kc.get("push_dll quant_chunks signal_loop")
    yag_flush, yag_wait, zero_masked = kc.get("yag_flush yag_wait zero_masked")

    TN_V = H // 8
    TN_IT = ceildiv(TN_V, NTT)

    @flyc.jit
    def blk_red(L, tid, v, op, slot):
        r = wave_red(v, tid % i32(64), op)
        if (tid % i32(64)) == i32(0):
            lds_st(
                L, i32(L_RING) + (i32(slot * (NTT // 64)) + tid // i32(64)) * i32(4), r
            )
        gpu.barrier()
        t = lds_ld_i32(L, i32(L_RING) + i32(slot * (NTT // 64) * 4))
        for k in range_constexpr(1, NTT // 64):
            t = op(t, lds_ld_i32(L, i32(L_RING) + i32((slot * (NTT // 64) + k) * 4)))
        return t

    @flyc.jit
    def tn_rows(L, tid, a):
        m = a["m"]
        S = fin_split(a)
        step = (S > i32(1)).select(m, i32(gpu.grid_dim.x))
        mine = fin_owned(a) != i32(0)
        start = ((S > i32(1)) & (i32(gpu.block_id("x")) >= m * S)).select(m, fin_key(a))
        for r_ in range(start, m, step):
            r = i32(r_)
            if tid == i32(0):
                last = i32(1)
                if S > i32(1):
                    ca = ctrl_at(a, i32(CTRL_TNC) + r)
                    old = g_add_agent(ca, 1)
                    last = (old == S - i32(1)).select(i32(1), i32(0))
                    if last == i32(1):
                        g_st_sys(ca, i32(0))
                lds_st(L, L_CTL + C_TNL * 4, mine.select(last, i32(0)))
            gpu.barrier()
            if lds_ld_i32(L, L_CTL + C_TNL * 4) == i32(1):
                tn_row(L, tid, a, r, True)
            gpu.barrier()

    def own_q(a):
        return rsrc(a["mine"] + fx.Int64(a["off_qall"]))

    @flyc.jit
    def tn_row(L, tid, a, r, sysy):
        H8 = i32(TN_V)
        ry = rsrc(a["y"])
        rres = rsrc(a["tn_res"])
        rout = rsrc(a["tn_out"])
        rw = rsrc(a["tn_w"])
        f, ss = [], fx.Float32(0.0)
        for k in range_constexpr(TN_IT):
            q = tid + i32(k * NTT)
            qc = fx.min(q, H8 - i32(1))
            off = (r * i32(H) + qc * i32(8)) * i32(2)
            yv = bf16x8_to_f32(bld(ry, off, 0, V4I, AUX_SYS if sysy else 0))
            rv = bf16x8_to_f32(bld(rres, off, 0, V4I, 0))
            fk = [x + y for x, y in zip(yv, rv)]
            live = q < H8
            if live:
                bst(pack_bf16x8(fk), rout, off, 0, 0)
            fk = [live.select(x, fx.Float32(0.0)) for x in fk]
            for x in fk:
                ss = ss + x * x
            f.append(fk)
        tot = blk_red(
            L,
            tid,
            ss.bitcast(fx.Int32),
            lambda x, y: (x.bitcast(fx.Float32) + y.bitcast(fx.Float32)).bitcast(
                fx.Int32
            ),
            0,
        )
        rcp = fx.Float32(
            fmath.rsqrt(
                tot.bitcast(fx.Float32) / fx.Float32(float(H))
                + fx.Float32(float(tn_eps))
            )
        )
        xs, am = [], fx.Float32(0.0)
        for k in range_constexpr(TN_IT):
            q = fx.min(tid + i32(k * NTT), H8 - i32(1))
            wv = bf16x8_to_f32(bld(rw, q * i32(16), 0, V4I, 0))
            xk = [
                (
                    fx.Float32(x * rcp * ((w + fx.Float32(1.0)) if tn_gemma else w))
                    .to(fx.BFloat16)
                    .to(fx.Float32)
                )
                for x, w in zip(f[k], wv)
            ]
            am = am.maximumf(amax(xk))
            xs.append(xk)
        amx = blk_red(
            L,
            tid,
            am.bitcast(fx.Int32),
            lambda x, y: x.bitcast(fx.Float32)
            .maximumf(y.bitcast(fx.Float32))
            .bitcast(fx.Int32),
            1,
        )
        scale = amx.bitcast(fx.Float32).maximumf(fx.Float32(1e-10)) / fx.Float32(448.0)
        inv = fx.Float32(1.0) / scale
        one = fx.Float32(1.0)
        grow = a["rank"] * a["m"] + r
        for k in range_constexpr(TN_IT):
            q = tid + i32(k * NTT)
            qv = [x * inv for x in xs[k]]
            d = fx.Vector.from_elements(
                [fp8x4_pack(qv[0:4], one), fp8x4_pack(qv[4:8], one)], fx.Int32
            )
            if q < H8:
                for p in range_constexpr(1 if TNL else TPC):
                    bst(
                        d,
                        own_q(a) if TNL else peer_rs(a, p, "off_qall"),
                        grow * i32(H) + q * i32(8),
                        0,
                        0 if TNL else AUX_SYS,
                    )
        b0 = a["mmax"] * i32(TPC * H)
        for k in range_constexpr(TN_IT if tn == 2 else 0):
            q = tid + i32(k * NTT)
            d = pack_bf16x8(xs[k])
            if q < H8:
                for p in range_constexpr(TPC):
                    bst(
                        d,
                        peer_rs(a, p, "off_qall"),
                        b0 + (grow * i32(H) + q * i32(8)) * i32(2),
                        0,
                        AUX_SYS,
                    )
        if tid == i32(0):
            for p in range_constexpr(1 if TNL else TPC):
                bst(
                    scale.bitcast(fx.Int32),
                    (
                        rsrc(a["mine"] + fx.Int64(a["off_sall"]))
                        if TNL
                        else peer_rs(a, p, "off_sall")
                    ),
                    grow * i32(4),
                    0,
                    0 if TNL else AUX_SYS,
                )
        if const_expr(TNL):
            return
        wait_vm(0)
        gpu.barrier()
        if tid < i32(TPC):
            fo = fx.Int64((i32(FLAG_TN) + a["rank"] * i32(TN_MAX) + r) * i32(4))
            g_st_sys(peer_sel(a, tid) + fx.Int64(a["off_flag"]) + fo, a["epoch"])

    @flyc.jit
    def tn_wait(tid, a):
        if tid < i32(64):
            nblk = i32(gpu.grid_dim.x)
            t = i32(gpu.block_id("x")) + tid * nblk
            live = t < a["ttot"]
            tc = fx.min(t, a["ttot"] - i32(1))
            src = tc // a["m"]
            row = tc - src * a["m"]
            addr = (
                a["mine"]
                + fx.Int64(a["off_flag"])
                + fx.Int64((i32(FLAG_TN) + src * i32(TN_MAX) + row) * i32(4))
            )
            poll_zero(
                lambda: wave_red(
                    (live & before(g_ld_sys(addr), a["epoch"])).select(i32(1), i32(0)),
                    tid,
                    fx.max,
                ),
                a,
                tid,
                ERR_YAG,
            )

    @flyc.jit
    def roles(L, tid, a, epoch):
        if tid < i32(NT):
            compute_units(L, tid, a)
            comm_help(L, tid, a, epoch)
        else:
            if tid < i32(NT + 64):
                ag_send(L, tid % i32(64), a)
                if const_expr(DLL):
                    push_dll(L, tid, a)
                    comm_wave(L, tid, a, epoch)
                else:
                    poll_loop(L, tid, a, epoch)
            elif tid < i32(NT + 128):
                a_loader(L, tid, a)
                if const_expr(LBPF):
                    lb_g2_prefetch(L, tid % i32(64), a)
                if const_expr(DLL):
                    push_dll(L, tid, a, 1)
                    signal_loop(L, tid, a, epoch)
                comm_help(L, tid, a, epoch)
            else:
                if const_expr(DLL):
                    if tid < i32(NT + 192):
                        zero_masked(L, tid % i32(64), a, 0, 1)
                        push_dll(L, tid, a, 2)
                    else:
                        push_dll(L, tid, a, 3)
                comm_help(L, tid, a, epoch)

    Shared = fx.struct(
        type(
            "Shared", (), {"__annotations__": {"buf": fx.Array[fx.Int8, LDS_BYTES, 16]}}
        )
    )

    def mega_moe_tp_kernel(
        w1: fx.Int64,
        w1s: fx.Int64,
        w2: fx.Int64,
        w2s: fx.Int64,
        x: fx.Int64,
        ids_in: fx.Int64,
        tw_in: fx.Int64,
        y: fx.Int64,
        routes: fx.Int64,
        proutes: fx.Int64,
        ctrl: fx.Int64,
        p0: fx.Int64,
        p1: fx.Int64,
        p2: fx.Int64,
        p3: fx.Int64,
        p4: fx.Int64,
        p5: fx.Int64,
        p6: fx.Int64,
        p7: fx.Int64,
        off_x: fx.Int64,
        off_xs: fx.Int64,
        off_ids: fx.Int64,
        off_w: fx.Int64,
        off_part: fx.Int64,
        off_flag: fx.Int64,
        off_yall: fx.Int64,
        rank: fx.Int32,
        tp: fx.Int32,
        m: fx.Int32,
        mmax: fx.Int32,
        xg: fx.Int64,
        tn_res: fx.Int64,
        tn_out: fx.Int64,
        tn_w: fx.Int64,
        off_qall: fx.Int64,
        off_sall: fx.Int64,
    ):
        a = dict(locals())
        tid = fx.Int32(gpu.thread_id("x"))
        assert layout_tag and name
        lds = fx.SharedAllocator().allocate(Shared).peek()
        L = uni(fx.Int32(fx.ptrtoint(lds.buf.ptr)))
        peers = [p0, p1, p2, p3, p4, p5, p6, p7]
        mine = peers[0]
        for j in range_constexpr(1, MAX_TP):
            mine = (rank == i32(j)).select(peers[j], mine)
        a.update(peer=peers, mine=fx.Int64(mine), ttot=tp * m)
        _dyn_zero(L, tid)
        init_lds(L, tid, a)
        gpu.barrier()
        epoch = lds_ld_i32(L, L_CTL + C_EPOCH * 4)
        a["epoch"] = epoch
        _drop_stale(tid, a)
        if (tid == i32(0)) & (i32(gpu.block_id("x")) == i32(0)):
            g_st_sys(claim_addr(a, 1), i32(0))
        if const_expr(ar):
            a["ids"] = fx.Int64(ids_in)
            a["tw"] = fx.Int64(tw_in)
        else:
            par = fx.Int64(epoch & i32(1)) * fx.Int64(a["tp"] * a["mmax"] * i32(TOPK))
            a["off_ids"] = off_ids + par * 16
            a["off_w"] = off_w + par * 4
            a["ids"] = fx.Int64(mine) + a["off_ids"]
            a["tw"] = fx.Int64(mine) + a["off_w"]
            if (tid >= i32(NT)) & (tid < i32(NT + 64)):
                if const_expr(MLL):
                    _ag_send_meta_ll(tid % i32(64), a)
                else:
                    _ag_send_meta(tid % i32(64), a)
        quant_chunks(L, tid, NTT, a, i32(0), i32(NCHA))
        gpu.barrier()
        a["ax"] = fx.Int64(mine) + off_x
        a["axs"] = fx.Int64(mine) + off_xs
        roles(L, tid, a, epoch)
        if const_expr(ar and not ARLL):
            wait_vm(0)
        if const_expr(TN):
            wait_vm(0)
        gpu.barrier()
        if const_expr(ar and not ARLL):
            yag_flush(L, tid, a)
            yag_wait(tid, a, epoch)
        if const_expr(TN):
            tn_rows(L, tid, a)
            if const_expr(not TNL):
                tn_wait(tid, a)
        if const_expr(AG8):
            gpu.barrier()
            ag8_convert(tid, a)
        if const_expr(lb):
            lb_finish(tid, a, L)
        finish(tid, a, epoch)

    kargs = list(inspect.signature(mega_moe_tp_kernel).parameters)
    mega_moe_tp_kernel = flyc.kernel(name=name, known_block_size=[NTT, 1, 1])(
        mega_moe_tp_kernel
    )

    def launch(
        w1: fx.Int64,
        w1s: fx.Int64,
        w2: fx.Int64,
        w2s: fx.Int64,
        x: fx.Int64,
        ids_in: fx.Int64,
        tw_in: fx.Int64,
        y: fx.Int64,
        routes: fx.Int64,
        proutes: fx.Int64,
        ctrl: fx.Int64,
        p0: fx.Int64,
        p1: fx.Int64,
        p2: fx.Int64,
        p3: fx.Int64,
        p4: fx.Int64,
        p5: fx.Int64,
        p6: fx.Int64,
        p7: fx.Int64,
        off_x: fx.Int64,
        off_xs: fx.Int64,
        off_ids: fx.Int64,
        off_w: fx.Int64,
        off_part: fx.Int64,
        off_flag: fx.Int64,
        off_yall: fx.Int64,
        rank: fx.Int32,
        tp: fx.Int32,
        m: fx.Int32,
        mmax: fx.Int32,
        xg: fx.Int64,
        tn_res: fx.Int64,
        tn_out: fx.Int64,
        tn_w: fx.Int64,
        off_qall: fx.Int64,
        off_sall: fx.Int64,
        i32_grid: fx.Int32,
        stream: fx.Stream,
    ):
        loc = locals()
        mega_moe_tp_kernel(*[loc[k] for k in kargs]).launch(
            grid=(fx.Int64(i32_grid), 1, 1), block=(NTT, 1, 1), stream=stream
        )

    # launch forwards its arguments to the kernel by name
    if list(inspect.signature(launch).parameters) != kargs + ["i32_grid", "stream"]:
        raise ValueError("compile_mega_moe_tp: kernel and launch signatures differ")
    return flyc.jit(launch)


# ---- big-M kernel ----
# Large-batch (big-M) path of MegaMoeTP.forward: one persistent launch runs the MoE of T
# tokens (m = T / tp rows per rank, ag_rs: every rank's routing of its own rows), in C chunks:
#
#   RB  this rank's rows -> MXFP4 rows, + their routing        -> SDMA gather of rows + routing (C)
#   S1  wait C; routes per expert       S2  route-list offsets, per-XCD unit queues
#   S3  route lists (route index, weight) per expert
#   U   fused units (build_bigm_unit): GEMM1 -> act + MXFP4 in LDS -> GEMM2 -> route rows
#   RQ  topk sum: own rows bf16, peer rows int8                -> SDMA MoE-output RS (E)
#   RF  wait E; y = the sum (+ tail: res_out = y + res, next-layer norm, per-token FP8
#       -> SDMA gather (G); or the y gather of the ar mode)
#   GW  wait every G; advance the persistent signal base
#
# All communication is MORI SDMA (puts with remote signals) issued and awaited inside the
# kernel; the GEMM1 -> GEMM2 intermediate stays in LDS. CTAs (one per CU, 4 waves) claim
# items with a per-XCD atomic ticket in a fixed phase order; an item only waits on items
# before it (and on peers' puts of earlier items), so the schedule cannot deadlock.
# Data another XCD reads in the same launch is stored write-through (sc0 | sc1).
const_expr = fx.const_expr


@functools.cache
def compile_bigm_kernel(
    H: int,
    I: int,
    E: int,
    K: int,
    tp: int,
    C: int,
    eps: float,
    gemma: bool,
    swiglu_limit: float,
    NAB: int = 4,
    order: int = 3,
    opts: tuple = (),
    tailq: int = 1,
    act: str = "swiglu",
):
    """act: the units' activation ("swiglu", "silu"); E < 512 routed experts (+ the shared one).
    tailq: 0 no tail, 1 the tail with the q gather, 2 the local tail (this rank's q rows),
    3 no tail, y of every rank's rows (bf16, the "q" region: the y gather of the ar mode).
    The MoE interface (MegaMoeTP.forward, ag_rs): the caller's normed own rows x [m, H] and their
    routing (ids / tw [m, K1], the shared expert's column included) are gathered; the tail sums the
    MoE output (y), and with tailq res_out = y + res_in, the next-layer norm + FP8 rows of every
    rank (q gather).
    """
    # every compile argument (the disk cache key, _bigm_cached)
    spec = repr(sorted(locals().items()))
    assert H % 256 == 0 and 0 < E < 512 and K <= 64 and act in ("swiglu", "silu")
    RG = UNIT_ROWS  # (one CTA of NW waves per CU)
    K1, NE = K + 1, E + 1
    HV = H // 8  # 16 B (8 bf16) columns of a row
    # row iterations per thread (the last one partial unless NT | HV)
    PIT = ceildiv(HV, NT)
    # per-expert ctrl / LDS entries (one thread per expert, NEB rounds of NT)
    NEB = ceildiv(NE, NT)
    NES = NEB * NT
    if NE <= 256:
        XST, CC_CNT, CC_OFF, CC_FILL = 256, C_ECNT, C_OFF, C_FILL
    else:
        XST, CC_CNT, CC_OFF, CC_FILL = 512, C_XW, C_XW + 8 * 512, C_XW + 16 * 512

    od = dict(opts)  # tuned AITER_BIGM_* options (configs/mega_moe_tp_bigm_tuned.csv)

    def _opt(name, default):
        return od.get(name, default)

    # cache policy of the data one CTA writes and another (on another XCD: separate L2s) reads in
    # this launch -- route rows, own MoE rows, received E rows: stored cached and this XCD's L2
    # written back (buffer_wbl2 sc1) before a unit counts as done (release), loaded sc1
    CAUX = 16
    # GEMM2 B ring depth (6 = a whole group: the route stores of a group never sit between a
    # GEMM2 load and its wait)
    BS2 = int(_opt("AITER_BIGM_BS2", "3"))
    # the B rings' depths divide the unit's GEMM1 chunks (H / 256) / GEMM2 k-steps (I / 128): the
    # largest divisor up to the requested depth (H 4096: BST 3 -> 2; I 512: BS2 3 -> 2, 6 -> 4)
    BST = max(d for d in range(1, 4) if (H // 256) % d == 0)
    BS2 = max(d for d in range(1, BS2 + 1) if (I // 128) % d == 0)
    # item order (see ORDER below); the host picks 3 (one unit round per chunk) or 4 (several)
    ORD = int(_opt("AITER_BIGM_ORD", str(order)))
    # tail sub-blocks: each chunk's topk sums / drains post their E / q puts per row sub-block
    NSB = 4
    # head sub-blocks: each chunk's gather of the input rows + routing (C) per row sub-block
    NSBA = int(_opt("AITER_BIGM_NSBA", "1"))
    assert NSBA in (1, 2, 4)
    UA = int(_opt("AITER_BIGM_UA", "32"))  # ORD 4: first-round unit slots per XCD
    # R8: route rows as MXFP8 (half the bytes of bf16 GEMM2 stores / topk-sum loads)
    R8 = int(_opt("AITER_BIGM_R8", "0"))
    # MoE-output RS rows as int4 (+ bf16 scale per 32)
    E4 = int(_opt("AITER_BIGM_E4", "0"))
    # SXW 0: the routing plan waits only for the gathered routing; units wait for the gathered rows
    SXW = int(_opt("AITER_BIGM_SXW", "1"))
    # topk sum: token pipeline depth (0: ZQ, a batch: every load first). bf16 route rows of many
    # routes (GLM-5: 9 x 6144): a batch's topk-sum loads alone take the register file (512 VGPRs
    # + scratch spills): pipelined instead
    RQD = int(bool(R8) or K1 * PIT > 16)
    # puts per chunk and queue; no q gather without the tail or with the local tail (own rows'
    # q only); the y gather (tailq 3): bf16 rows on QQ, no scales
    PPC = {QE: NSB, QQ: NSB, QS: NSB, QX: NSBA, QXS: NSBA, QID: 2 * NSBA}
    if tailq not in (1, 3):
        PPC[QQ] = PPC[QS] = 0
    if tailq == 3:
        PPC[QS] = 0
    ZX = 16  # io: rows per input item (quant + routing copy)
    EPL = ceildiv(E, 64)  # experts per lane of the plan's wave
    # int8 rows of the attention / MoE-output RS: one fp32 scale per 32 values
    RB = i8_row_bytes(H)
    SCO = H
    lay, unit_body = build_bigm_unit(H, I, K1, RG, NAB, BST, BS2, act, swiglu_limit, R8)
    # row-phase / plan scratch after the unit's LDS (also reused ranges)
    LBASE = lay["BYTES"]
    LSC = LBASE  # 64 B scratch for block reductions (NW x 4 words)
    LT = LBASE + 64  # ticket broadcast
    LH = LBASE + 128  # [NE] histogram / ranks
    LB2 = LH + 4 * NES  # [NE] bases
    LP = LB2 + 4 * NES  # [SRT * K1] route positions of a sort item ([NE] plan scratch)
    LTID = LP + max(4 * SRT * K1, 64 + 4 * max(NE, 64 * EPL)) + 64
    LDS_BYTES = LTID + 4 * NT  # [NT] each thread's id (own_tid)
    assert LDS_BYTES <= 160 * 1024, LDS_BYTES

    def cpart(kk):  # row iteration kk is the partial last one
        return HV % NT != 0 and kk == PIT - 1

    def col(tid, kk):
        """this thread's column of row iteration kk; past the row (a partial last iteration) the
        same lane of the last full 4-column group: those threads redo its loads, math and stores
        (identical values, its per-32 scales included; the fused tail: see own_col)"""
        p = tid + i32(kk * NT)
        if not cpart(kk):
            return p
        return (p < i32(HV)).select(p, i32(HV - 4) + (p & i32(3)))

    def own_col(tid, kk):
        """None, else (a partial iteration) whether this thread's column is its own, not redone.
        The fused tail updates res in place (res_in == res_out): a redone column's res load
        can follow its owner's store, so there the redone columns store nothing and stay out
        of the row's sums."""
        if not cpart(kk):
            return None
        return tid + i32(kk * NT) < i32(HV)

    def at_col(base, tid, kk):
        """base + this thread's first element (bf16) of iteration kk"""
        if not cpart(kk):
            return base + tid * i32(8) + i32(kk * NT * 8)
        return base + col(tid, kk) * i32(8)

    # item order: chunk k+1's prep (input rows RB, plan) ahead of chunk k's units and drain
    # (a CTA spinning on an item cannot take later ones: items that wait for a transfer come
    # after the work that overlaps it) chunk k's units, then chunk k+1's prep (claimed as
    # chunk k's units drain), chunk k's topk sum, then chunk k-1's drain after k's units
    def prep(k):
        return [(k, P_RB), (k, P_S1), (k, P_S2), (k, P_S3)]

    assert ORD in (3, 4), ORD
    if ORD == 4:
        # U(k) = the first unit slot of every CTA, UB(k) the rest: chunk k+1's input rows run
        # between them and its routing plan right after, so chunk k+1's units fill the CTAs
        # chunk k's last unit round leaves idle
        ORDER = prep(0)
        for k in range(C):
            ORDER += [(k, P_U)]
            if k >= 1:
                ORDER += [(k - 1, P_RQ)]
            if k + 1 < C:
                ORDER += [(k + 1, P_RB)]
            ORDER += [(k, P_UB)]
            if k >= 1:
                ORDER += [(k - 1, P_RF)]
            if k + 1 < C:
                ORDER += [(k + 1, P_S1), (k + 1, P_S2), (k + 1, P_S3)]
        ORDER += [(C - 1, P_RQ), (C - 1, P_RF)]
    elif ORD == 3:
        # chunk k's topk sum right after chunk k+1's prep (it fills the gap while chunk k+1's
        # routing lands, on the whole machine) so E(k), RF(k), G(k) overlap chunk k+1's units
        ORDER = prep(0)
        for k in range(C):
            ORDER += [(k, P_U)]
            if k >= 1:  # on the CTAs chunk k's units leave idle (E(k-1) is on its way)
                ORDER += [(k - 1, P_RF)]
            if k + 1 < C:
                ORDER += prep(k + 1)
            ORDER += [(k, P_RQ)]
        ORDER += [(C - 1, P_RF)]
        ORDER += [(k, P_UB) for k in range(C)]  # empty: U(k) takes every unit slot
    USPLIT = ORD == 4

    def fb(v):
        return f"{struct.unpack('<I', struct.pack('<f', v))[0]:x}"

    name = (
        f"bigm_kernel_h{H}_i{I}_e{E}k{K}_tp{tp}_c{C}_rg{RG}_{fb(eps)}{'g' if gemma else 'r'}"
        + f"_v23_t{tailq}_c1h{NSBA}_o{ORD}s{NSB}a{UA}d{RQD}r{R8}w{SXW}e{E4}b{BS2}"
        + ("" if act == "swiglu" else f"_{act}{fb(swiglu_limit)}")
    )

    Shared = fx.struct(
        type(
            "Shared",
            (),
            {"__annotations__": {"raw": fx.Array[fx.Uint8, LDS_BYTES, 16]}},
        )
    )

    @flyc.jit
    def blk_red_vec(L, tid, vals, op):
        """block reduction of len(vals) <= 4 values with one barrier pair"""
        n = len(vals)
        rs_ = [wave_red(v, tid % i32(64), op) for v in vals]
        if (tid % i32(64)) == i32(0):
            for i in range_constexpr(n):
                lds_st(
                    L, i32(LSC) + ((tid // i32(64)) * i32(n) + i32(i)) * i32(4), rs_[i]
                )
        gpu.barrier()
        out = []
        for i in range_constexpr(n):
            t = lds_ld_i32(L, i32(LSC + i * 4))
            for k in range_constexpr(1, NW):
                t = op(t, lds_ld_i32(L, i32(LSC + (k * n + i) * 4)))
            out.append(t)
        gpu.barrier()
        return out

    def i8_scale_ld(rx, row, p):
        """the fp32 bits of the scale of values p*8.. of an int8 row"""
        g = p // i32(4)  # scale group (32 values)
        return fx.Int32(bld(rx, row * i32(RB) + i32(SCO) + g * i32(4), 0, T.i32, CAUX))

    def i8_rows_ld(rx, row, p):
        return (
            fx.Vector(bld(rx, row * i32(RB) + p * i32(8), 0, T.vec(2, T.i32), CAUX)),
            i8_scale_ld(rx, row, p),
        )

    def i8_rows_val(ld):
        d, sc = ld
        return i8x4_unpack(d[0], sc.bitcast(fx.Float32)) + i8x4_unpack(
            d[1], sc.bitcast(fx.Float32)
        )

    # E (MoE-output RS) rows: int8 + f32 scale per 32 (RB bytes), or E4: int4 + bf16 scale per 32
    # (QuickReduce INT4's code: ATOM's MoE-output reduce-scatter)
    RBE = (H // 2 + H // 16) if E4 else RB

    def e_ld(rx, row, p):
        if const_expr(E4):
            d = fx.Int32(bld(rx, row * i32(RBE) + p * i32(4), 0, T.i32, CAUX))
            sb = fx.Int32(
                bld(
                    rx,
                    row * i32(RBE) + i32(H // 2) + (p // i32(4)) * i32(2),
                    0,
                    T.i16,
                    CAUX,
                )
            ) & i32(0xFFFF)
            return (d, sb)
        return i8_rows_ld(rx, row, p)

    def e_val(ld):
        if const_expr(E4):
            d, sb = ld
            return i4x8_unpack(d, (sb << i32(16)).bitcast(fx.Float32))
        return i8_rows_val(ld)

    @flyc.jit
    def e_store(f, rd, drow, p, lane):
        if const_expr(E4):
            am = amax(f)
            am = am.maximumf(am.shuffle_xor(i32(1), i32(64)))
            am = am.maximumf(am.shuffle_xor(i32(2), i32(64)))
            # QuickReduce CodecQ4's code (ATOM's INT4 MoE-output reduce-scatter): scale -amax / 8,
            # all 16 levels [-8, 7] (amax -> -8, -amax -> 8 clamped to 7); MoE output rel_l2 vs
            # torch 0.0856 (symmetric [-7, 7]: 0.0882)
            sc = fx.Float32(
                fx.Float32(am.maximumf(fx.Float32(1e-30)) / fx.Float32(-8.0))
                .to(fx.BFloat16)
                .to(fx.Float32)
            )
            bst(
                i4x8_pack(f, fx.Float32(1.0) / sc),
                rd,
                drow * i32(RBE) + p * i32(4),
                0,
                AUX_SYS,
            )
            if (lane & i32(3)) == i32(0):
                bst(
                    fx.Int16(sc.bitcast(fx.Int32) >> i32(16)),
                    rd,
                    drow * i32(RBE) + i32(H // 2) + (p // i32(4)) * i32(2),
                    0,
                    AUX_SYS,
                )
        else:
            i8_row_store(f, rd, drow, p, lane)

    @flyc.jit
    def i8_row_store(f, rd, drow, p, lane):
        am = amax(f)
        am = am.maximumf(am.shuffle_xor(i32(1), i32(64)))
        am = am.maximumf(am.shuffle_xor(i32(2), i32(64)))
        g = p // i32(4)
        sc = am.maximumf(fx.Float32(1e-30)) / fx.Float32(127.0)
        inv = fx.Float32(1.0) / sc
        dv = fx.Vector.from_elements(
            [i8x4_pack(f[0:4], inv), i8x4_pack(f[4:8], inv)], fx.Int32
        )
        bst(dv, rd, drow * i32(RB) + p * i32(8), 0, AUX_SYS)
        if (lane & i32(3)) == i32(0):
            bst(
                sc.bitcast(fx.Int32),
                rd,
                drow * i32(RB) + i32(SCO) + g * i32(4),
                0,
                AUX_SYS,
            )

    def kernel(
        wb: fx.Int64,  # this rank's data arena (SdmaComm: hipIpc, not a MORI window)
        pbases: fx.Int64,  # [tp] int64 every rank's arena as mapped here (the put destinations)
        dc: fx.Int64,  # MORI device communicator
        consumed: fx.Int64,  # [tp][NQ] int32 puts awaited before this launch (persistent;
        # the next launch's at consumed ^ (tp * NQ * 4): two buffers in turn)
        offs: fx.Int64,  # [8] int64 arena region offsets
        xin: fx.Int64,  # [m, H] bf16 own rows
        ids: fx.Int64,  # [m, K1] int32
        tw: fx.Int64,  # [m, K1] f32
        normw: fx.Int64,  # [H] the next layer's norm weight (tail)
        resin: fx.Int64,  # [m, H] bf16 (tail)
        yout: fx.Int64,  # [m, H] bf16 out (the MoE output)
        resout: fx.Int64,  # [m, H] bf16 out
        ybuf: fx.Int64,  # [m, H] bf16 own MoE rows
        routes: fx.Int64,  # [T * K1, H] bf16 route rows (chunk k's from route tp * o_k * K1)
        lrix: fx.Int64,  # [C][T * K1] route lists
        lwt: fx.Int64,
        utab: fx.Int64,  # [C][8][ucap][4] units
        w1: fx.Int64,
        w1s: fx.Int64,
        w2: fx.Int64,
        w2s: fx.Int64,
        ctrl: fx.Int64,
        chunks: fx.Int64,  # [C][2] int32: (rows offset o, rows s) per rank
        rank: fx.Int32,
        m: fx.Int32,
        ucap: fx.Int32,
    ):
        sa = fx.SharedAllocator()
        st = sa.allocate(Shared).peek()
        L = fx.Int32(fx.ptrtoint(st.raw.ptr))
        tid = fx.Int32(gpu.thread_id("x"))
        lds_st(L, i32(LTID) + tid * i32(4), tid)

        def own_tid():
            # the thread id through a volatile LDS load, at the start of every item: values
            # derived from it stay inside the item (not hoisted / CSE'd across items and live
            # through the whole loop: register spills)
            return lds_ld_acq(L, i32(LTID) + tid * i32(4))

        lane = tid % i32(64)
        bid = i32(gpu.block_id("x"))
        xc = bid % i32(8)
        roffs = rsrc(offs)

        def off_of(i):
            return fx.Int64(bld(roffs, i32(i * 8), 0, T.i64))

        a = {
            "wb": wb,
            "off_xq": off_of(0),
            "off_xs": off_of(1),
            "off_ids": off_of(2),
            "off_tw": off_of(3),
            "off_eq": off_of(4),
            "off_erecv": off_of(5),
            "off_q": off_of(6),
            "off_qs": off_of(7),
        }

        def ca(i):
            return ctrl + fx.Int64(i * i32(4))

        def wp(name):  # local window address of a region
            return wb + a["off_" + name]

        rch = rsrc(chunks)

        def chunk(k):  # (rows offset o, rows s) of chunk k (traced)
            return fx.Int32(bld(rch, k * i32(8), 0, T.i32)), fx.Int32(
                bld(rch, k * i32(8) + i32(4), 0, T.i32)
            )

        def lbase(k):  # chunk k's first route (route lists, route rows: chunks packed)
            return i32(tp) * chunk(k)[0] * i32(K1)

        def rbase(k):  # chunk k's route rows (bytes)
            return fx.Int64(lbase(k)) * fx.Int64(H * 2)

        # this launch's signal base per (peer, queue) (GW writes the next launch's into the
        # other buffer)
        rcons = rsrc(consumed)

        def sig_base(p, q):
            return fx.Int32(
                bld(rcons, (p * i32(NQ) + i32(q)) * i32(4), 0, T.i32, AUX_SYS)
            )

        # ---- SDMA (MORI device functions) ----
        from flydsl.compiler.extern_link import link_extern
        from flydsl.expr.extern import ffi

        from ....communication import sdma_bitcode

        bc = sdma_bitcode()
        put_ptr = link_extern(
            ffi(
                "cco_sdma_put_ptr_rsig__thread",
                ["uint64", "int32", "uint64", "uint64", "uint64", "int32"],
                "int32",
            ),
            bitcode_path=bc,
        )
        sig_slot = link_extern(
            ffi("cco_sdma_signal_slot", ["uint64", "int32", "int32"], "uint64"),
            bitcode_path=bc,
        )

        @flyc.jit
        def take_turn(q, nth):
            # posts on one queue in a fixed order, one poster at a time (MORI loses remote
            # signals of concurrent puts on the same queue)
            cur = fx.Int32(load_i32_global_agent(ca(i32(C_TURN + 32 * q))))
            while cur < nth:
                rocdl.s_sleep(1)
                cur = fx.Int32(load_i32_global_agent(ca(i32(C_TURN + 32 * q))))

        def give_turn(q, nth):
            g_st_sys(ca(i32(C_TURN + 32 * q)), nth + i32(1))

        rpb = rsrc(pbases)

        def put(peer, dst_region, dst_off, src_region, src_off, nbytes, q):
            # region + offset of the peer's arena <- the same of ours (remote-signalled)
            dst = (
                fx.Int64(bld(rpb, peer * i32(8), 0, T.i64))
                + a["off_" + dst_region]
                + fx.Int64(dst_off)
            )
            src = wb + a["off_" + src_region] + fx.Int64(src_off)
            put_ptr(dc, peer, dst, src, fx.Int64(nbytes), i32(q))

        def wait_puts(q, n_per, k):
            # every peer's first (k + 1) * n_per puts on queue q of this launch landed
            wait_puts_n(q, (k + i32(1)) * i32(n_per))

        @flyc.jit
        def wait_puts_n(q, n):
            if lane < i32(tp):
                p = lane
                if p != rank:
                    slot = fx.Int64(sig_slot(dc, p, i32(q)))
                    tgt = sig_base(p, q) + n
                    cur = g_ld_sys(slot)
                    t0 = now()
                    while before(cur, tgt) & alive(t0):
                        rocdl.s_sleep(2)
                        cur = g_ld_sys(slot)
                    if before(
                        cur, tgt
                    ):  # gave up on a peer: flag it (the results are not valid)
                        g_st_sys(ca(i32(C_ERR)), i32(1))

        def ph(k, kind):
            return ca(i32(C_PH) + (k * i32(NKIND) + i32(kind)) * i32(32))

        @flyc.jit
        def done_last(k, kind, n):
            """this CTA finished an item of (k, kind): True (uniform) for the last of n"""
            wait_vm(0)
            gpu.barrier()
            if tid == i32(0):
                v = fx.Int32(atomic_add_agent(ph(k, kind), i32(1)))
                lds_st(L, i32(LT) + i32(4), (v == n - i32(1)).select(i32(1), i32(0)))
            gpu.barrier()
            return lds_ld_i32(L, i32(LT) + i32(4)) == i32(1)

        @flyc.jit
        def wait_ph(k, kind, n):
            wait_at(ph(k, kind), n)

        @flyc.jit
        def wait_at(addr, n):
            if tid == i32(0):
                cur = fx.Int32(load_i32_global_agent(addr))
                while cur < n:
                    rocdl.s_sleep(1)
                    cur = fx.Int32(load_i32_global_agent(addr))
            gpu.barrier()

        def sbc(k, w, b):  # sub-block done counter (w 0: RQ items, 1: RF items)
            return ca(i32(C_SB) + ((k * i32(2) + i32(w)) * i32(16) + b) * i32(8))

        @flyc.jit
        def done_at(addr, n):
            """like done_last, on a counter address (callers ran done_last first: stores drained)"""
            if tid == i32(0):
                v = fx.Int32(atomic_add_agent(addr, i32(1)))
                lds_st(L, i32(LT) + i32(4), (v == n - i32(1)).select(i32(1), i32(0)))
            gpu.barrier()
            return lds_ld_i32(L, i32(LT) + i32(4)) == i32(1)

        def il(j, n):
            # XCD-interleaved item order (XCD x's contiguous slice -> every 8th item) so the
            # sub-blocks complete in order; identity when n is not a multiple of 8
            q = n // i32(8)
            qq = fx.max(q, i32(1))
            x = j // qq
            jn = (j - x * qq) * i32(8) + x
            return ((n % i32(8) == i32(0)) & (q > i32(0))).select(jn, j)

        def claim():
            if tid == i32(0):
                lds_st(
                    L,
                    i32(LT),
                    fx.Int32(atomic_add_agent(ca(i32(C_TK) + xc * i32(32)), i32(1))),
                )
            gpu.barrier()
            v = lds_ld_i32(L, i32(LT))
            gpu.barrier()
            return v

        def part8(n):
            q = n // i32(8)
            r = n - q * i32(8)
            return xc * q + (xc < r).select(xc, r), q + (xc < r).select(i32(1), i32(0))

        # ---- item counts per (chunk, kind) (rows s per rank of the chunk) ----

        def cnt(kind, s_):
            if const_expr(kind == P_RB):  # the input rows (item_rx)
                return s_ // i32(ZX)
            if const_expr(kind == P_RF):
                return s_ // i32(ZQ)
            if const_expr(kind in (P_S1, P_S3)):
                return (i32(tp) * s_ + i32(SRT - 1)) // i32(SRT)
            if const_expr(kind == P_S2):
                return i32(1)
            if const_expr(kind == P_U):
                return ucap * i32(8)
            return (i32(tp) * s_) // i32(ZQ)

        def chunk_v(k):  # per-claim (not hoisted: system-scope load)
            return fx.Int32(bld(rch, k * i32(8) + i32(4), 0, T.i32, AUX_SYS))

        def slices_of(sk):
            out = []
            tot_ = i32(0)
            for oi in range_constexpr(len(ORDER)):
                k, kind = ORDER[oi]
                n = cnt(kind, sk[k])
                if const_expr(kind == P_S2):
                    lo_, c_ = i32(0), (xc == i32(0)).select(i32(1), i32(0))
                elif const_expr(kind == P_U):
                    lo_, c_ = i32(0), (fx.min(ucap, i32(UA)) if USPLIT else ucap)
                elif const_expr(kind == P_UB):
                    lo_, c_ = i32(UA), (
                        fx.max(ucap - i32(UA), i32(0)) if USPLIT else i32(0)
                    )
                else:
                    lo_, c_ = part8(n)
                out.append((k, kind, lo_, tot_, tot_ + c_))
                tot_ = tot_ + c_
            return out, tot_

        _, gw_lo = slices_of([chunk(i32(k))[1] for k in range(C)])
        nall = gw_lo + (xc == i32(0)).select(i32(1), i32(0))

        r_xin, r_resin, r_ids = rsrc(xin), rsrc(resin), rsrc(ids)
        r_tw, r_normw, r_y = rsrc(tw), rsrc(normw), rsrc(ybuf)
        r_yout, r_out = rsrc(yout), rsrc(resout)

        def norm_w(rw, p):
            return bf16x8_to_f32(bld(rw, p * i32(16), 0, T.vec(4, T.i32)))

        def gate_w(x, wk):
            return x * ((wk + fx.Float32(1.0)) if gemma else wk)

        # ======================= items =======================

        def sbh(k, w, b):  # head sub-block done counter (w 1: input row items)
            return ca(i32(C_SBH) + ((k * i32(2) + i32(w)) * i32(16) + b) * i32(8))

        @flyc.jit
        def post_c(k, bh):
            # thread 0: this head sub-block's quantized rows + routing -> every peer
            o, s = chunk(k)
            B = i32(tp) * o
            sa = s // i32(NSBA)
            r0b = B + rank * s + bh * sa
            t1 = k * i32(NSBA) + bh
            CPUTS = (
                (QX, "xq", H // 2),
                (QXS, "xs", H // 32),
                (QID, "ids", K1 * 4),
                (QID, "tw", K1 * 4),
            )
            for ci in range_constexpr(4):
                q_, reg, rowb = CPUTS[ci]
                tq = t1 * i32(2) if const_expr(q_ == QID) else t1
                if const_expr(reg != "tw"):
                    take_turn(q_, tq)
                for p_ in range_constexpr(tp):
                    if i32(p_) != rank:
                        put(
                            i32(p_),
                            reg,
                            r0b * i32(rowb),
                            reg,
                            r0b * i32(rowb),
                            sa * i32(rowb),
                            q_,
                        )
                if const_expr(reg in ("xq", "xs")):
                    give_turn(q_, tq)
                if const_expr(reg == "tw"):
                    give_turn(q_, tq + i32(1))

        @flyc.jit
        def item_rx(k, j):
            # io: ZX of this rank's (normed) rows -> MXFP4 rows + routing in the window; the last
            # item of a head sub-block posts it to every peer (gather C)
            tid = own_tid()
            lane = tid % i32(64)
            o, s = chunk(k)
            B = i32(tp) * o
            jx = il(j, cnt(P_RB, s))
            rq, rqs = rsrc(wp("xq")), rsrc(wp("xs"))
            i0 = jx * i32(ZX)
            ld = [
                [
                    bld(
                        r_xin,
                        ((o + i0 + i32(ii)) * i32(H) + col(tid, kk) * i32(8)) * i32(2),
                        0,
                        T.vec(4, T.i32),
                    )
                    for kk in range(PIT)
                ]
                for ii in range(ZX)
            ]
            if tid < i32(ZX * K1):
                rr_ = tid // i32(K1)
                src = (o + i0 + rr_) * i32(K1) + (tid - rr_ * i32(K1))
                dst = (B + rank * s + i0) * i32(K1) + tid
                bst(
                    fx.Int32(bld(r_ids, src * i32(4), 0, T.i32)),
                    rsrc(wp("ids")),
                    dst * i32(4),
                    0,
                    AUX_SYS,
                )
                bst(
                    fx.Int32(bld(r_tw, src * i32(4), 0, T.i32)),
                    rsrc(wp("tw")),
                    dst * i32(4),
                    0,
                    AUX_SYS,
                )
            for ii in range_constexpr(ZX):
                row0 = B + rank * s + i0 + i32(ii)
                for kk in range_constexpr(PIT):
                    p = col(tid, kk)
                    y = bf16x8_to_f32(ld[ii][kk])
                    am = amax(y)
                    am = am.maximumf(am.shuffle_xor(i32(1), i32(64)))
                    am = am.maximumf(am.shuffle_xor(i32(2), i32(64)))
                    e8, qs = _e8m0_from_amax(am, max_norm=6.0)
                    bst(
                        fp4_pack(y, qs), rq, row0 * i32(H // 2) + p * i32(4), 0, AUX_SYS
                    )
                    if (lane & i32(3)) == i32(0):
                        bst(
                            fx.Int8(e8),
                            rqs,
                            row0 * i32(H // 32) + p // i32(4),
                            0,
                            AUX_SYS,
                        )
            done_last(k, P_RB, cnt(P_RB, s))
            sa = s // i32(NSBA)
            bh = i0 // sa
            if done_at(  # noqa: SIM102 - preserve FlyDSL branches
                sbh(k, 1, bh), sa // i32(ZX)
            ):
                if tid == i32(0):
                    post_c(k, bh)

        @flyc.jit
        def item_s1(k, j):
            tid = own_tid()
            o, s = chunk(k)
            B = i32(tp) * o
            if const_expr(SXW):
                wait_puts(QX, NSBA, k)
                wait_puts(QXS, NSBA, k)
            wait_puts(QID, 2 * NSBA, k)
            wait_ph(k, P_RB, cnt(P_RB, s))  # own rows' routing copied into the window
            for b_ in range_constexpr(NEB):
                if tid + i32(b_ * NT) < i32(NE):
                    lds_st(L, i32(LH) + (tid + i32(b_ * NT)) * i32(4), i32(0))
            gpu.barrier()
            rids = rsrc(wp("ids"))
            r0 = (B + j * i32(SRT)) * i32(K1)
            nr = fx.min((B + i32(tp) * s) * i32(K1) - r0, i32(SRT * K1))
            for kk in range_constexpr((SRT * K1 + NT - 1) // NT):
                r = tid + i32(kk * NT)
                if r < nr:
                    e = fx.Int32(bld(rids, (r0 + r) * i32(4), 0, T.i32, AUX_SYS))
                    if (e >= i32(0)) & (e < i32(NE)):  # masked / invalid ids: no route
                        lds_atomic_add(L, i32(LH) + e * i32(4), i32(1))
            gpu.barrier()
            for b_ in range_constexpr(NEB):
                e = tid + i32(b_ * NT)
                if e < i32(NE):
                    c = lds_ld_i32(L, i32(LH) + e * i32(4))
                    if c > i32(0):
                        atomic_add_agent(ca(i32(CC_CNT) + k * i32(XST) + e), c)
            done_last(k, P_S1, cnt(P_S1, s))

        @flyc.jit
        def item_s2(k):
            tid = own_tid()
            _o, s = chunk(k)
            wait_ph(k, P_S1, cnt(P_S1, s))
            # route-list offsets (prefix of the counts, each thread its own expert)
            for b_ in range_constexpr(NEB):
                e = tid + i32(b_ * NT)
                if e < i32(NE):
                    lds_st(
                        L,
                        i32(LH) + e * i32(4),
                        fx.Int32(
                            load_i32_global_agent(ca(i32(CC_CNT) + k * i32(XST) + e))
                        ),
                    )
            gpu.barrier()
            accs = [i32(0)] * NEB
            for e_ in range_constexpr(NE - 1):
                v = lds_ld_i32(L, i32(LH + 4 * e_))
                for b_ in range_constexpr(NEB):
                    accs[b_] = accs[b_] + (i32(e_) < tid + i32(b_ * NT)).select(
                        v, i32(0)
                    )
            for b_ in range_constexpr(NEB):
                e = tid + i32(b_ * NT)
                if e < i32(NE):
                    g_st_sys(ca(i32(CC_OFF) + k * i32(XST) + e), accs[b_])
                    lds_st(L, i32(LB2) + e * i32(4), accs[b_])
            gpu.barrier()
            s2_greedy(k)
            done_last(k, P_S2, i32(1))

        @flyc.jit
        def s2_greedy(k):
            tid = own_tid()
            # units -> per-XCD queues: unit g of one round-robin sequence (the shared expert's
            # units, then those of the routed experts with more than UQ units) to queue g % 8,
            # slot g / 8; every other routed expert's units whole to the least-loaded queue (its
            # weights shared in that XCD's L2)
            rut = rsrc(utab)
            ns = (lds_ld_i32(L, i32(LH + 4 * E)) + i32(RG - 1)) // i32(RG)
            soff = lds_ld_i32(L, i32(LB2 + 4 * E))
            cs_ = lds_ld_i32(L, i32(LH + 4 * E))
            for u_ in range(tid, ns, i32(NT)):
                u = i32(u_)
                ent = ((k * i32(8) + u % i32(8)) * ucap + u // i32(8)) * i32(16)
                bst(i32(E), rut, ent, 0, AUX_SYS)
                bst(soff + u * i32(RG), rut, ent + i32(4), 0, AUX_SYS)
                bst(fx.min(cs_ - u * i32(RG), i32(RG)), rut, ent + i32(8), 0, AUX_SYS)
            # the routed experts' queue / first slot to LDS (wave 0), then every expert's units
            LEQ = LP + 64
            # wave 0: the queue loads and unit counts wave-uniform; key = load * 8 + queue is
            # unique per queue, so the least-loaded (lowest on ties) queue is the one whose key is
            # the minimum; each expert's (queue, first slot) is kept by lane e % 64
            # (a spread expert's entry: bit 31 | its first g)
            if tid < i32(64):
                nuv = [
                    (
                        lds_ld_i32(L, i32(LH) + (tid + i32(64 * j)) * i32(4))
                        + i32(RG - 1)
                    )
                    // i32(RG)
                    for j in range(EPL)
                ]
                g_ = uni(ns)
                qb = [i32(0)] * EPL
                for e_ in range_constexpr(E):
                    nu = i32(rocdl.readlane(T.i32, nuv[e_ // 64], i32(e_ % 64)))
                    sp = nu > i32(UQ)
                    qb[e_ // 64] = ((tid == i32(e_ % 64)) & sp).select(
                        g_ | i32(-(1 << 31)), qb[e_ // 64]
                    )
                    g_ = g_ + sp.select(nu, i32(0))
                ky = [
                    ((g_ - i32(x) + i32(7)) // i32(8)) * i32(8) + i32(x)
                    for x in range(8)
                ]
                for e_ in range_constexpr(E):
                    nu = i32(rocdl.readlane(T.i32, nuv[e_ // 64], i32(e_ % 64)))
                    sp = nu > i32(UQ)
                    nu8 = sp.select(i32(0), nu * i32(8))
                    m_ = fx.min(
                        fx.min(fx.min(ky[0], ky[1]), fx.min(ky[2], ky[3])),
                        fx.min(fx.min(ky[4], ky[5]), fx.min(ky[6], ky[7])),
                    )
                    qb[e_ // 64] = (tid == i32(e_ % 64)).select(
                        sp.select(
                            qb[e_ // 64], (m_ & i32(7)) | ((m_ >> i32(3)) << i32(8))
                        ),
                        qb[e_ // 64],
                    )
                    for x in range_constexpr(8):
                        ky[x] = ky[x] + (ky[x] == m_).select(nu8, i32(0))
                for j in range_constexpr(EPL):
                    lds_st(L, i32(LEQ) + (tid + i32(64 * j)) * i32(4), qb[j])
                if tid == i32(0):
                    for x in range_constexpr(8):
                        lds_st(L, i32(LP + 4 * x), ky[x] >> i32(3))
            gpu.barrier()
            for b_ in range_constexpr(
                ceildiv(E, NT)
            ):  # (one round per NT routed experts)
                e = tid if b_ == 0 else tid + i32(b_ * NT)
                if e < i32(E):
                    c = lds_ld_i32(L, i32(LH) + e * i32(4))
                    off = lds_ld_i32(L, i32(LB2) + e * i32(4))
                    nu = (c + i32(RG - 1)) // i32(RG)
                    qb = lds_ld_i32(L, i32(LEQ) + e * i32(4))
                    spr = qb < i32(0)
                    g0 = qb & i32(0x7FFFFFFF)
                    x_ = qb & i32(0xFF)
                    b0 = (qb >> i32(8)) & i32(0xFFFFFF)
                    for u_ in range(i32(0), nu, i32(1)):
                        u = i32(u_)
                        xs = spr.select((g0 + u) % i32(8), x_)
                        slot = fx.min(
                            spr.select((g0 + u) // i32(8), b0 + u), ucap - i32(1)
                        )
                        ent = ((k * i32(8) + xs) * ucap + slot) * i32(16)
                        bst(e, rut, ent, 0, AUX_SYS)
                        bst(off + u * i32(RG), rut, ent + i32(4), 0, AUX_SYS)
                        bst(
                            fx.min(c - u * i32(RG), i32(RG)),
                            rut,
                            ent + i32(8),
                            0,
                            AUX_SYS,
                        )
            gpu.barrier()
            if tid < i32(8):
                g_st_sys(
                    ca(i32(C_UN) + tid * i32(32) + k * i32(256)),
                    fx.min(lds_ld_i32(L, i32(LP) + tid * i32(4)), ucap),
                )

        @flyc.jit
        def item_s3(k, j):
            tid = own_tid()
            o, s = chunk(k)
            B = i32(tp) * o
            wait_ph(k, P_S2, i32(1))  # the route-list offsets
            for b_ in range_constexpr(NEB):
                if tid + i32(b_ * NT) < i32(NE):
                    lds_st(L, i32(LH) + (tid + i32(b_ * NT)) * i32(4), i32(0))
            gpu.barrier()
            rids, rtw = rsrc(wp("ids")), rsrc(wp("tw"))
            r0 = (B + j * i32(SRT)) * i32(K1)
            nr = fx.min((B + i32(tp) * s) * i32(K1) - r0, i32(SRT * K1))
            NRI = (SRT * K1 + NT - 1) // NT
            es, rk, oks = [], [], []
            for kk in range_constexpr(NRI):
                r = tid + i32(kk * NT)
                e = fx.Int32(
                    bld(rids, (fx.min(r, nr - i32(1)) + r0) * i32(4), 0, T.i32, AUX_SYS)
                )
                ok = (
                    (r < nr) & (e >= i32(0)) & (e < i32(NE))
                )  # masked / invalid ids: no route
                e = ok.select(e, i32(0))
                es.append(e)
                oks.append(ok)
                rk.append(
                    lds_atomic_add(L, i32(LH) + e * i32(4), ok.select(i32(1), i32(0)))
                )
            gpu.barrier()
            for b_ in range_constexpr(NEB):
                e = tid + i32(b_ * NT)
                if e < i32(NE):
                    c = lds_ld_i32(L, i32(LH) + e * i32(4))
                    base = fx.Int32(
                        atomic_add_agent(ca(i32(CC_FILL) + k * i32(XST) + e), c)
                    )
                    lds_st(
                        L,
                        i32(LB2) + e * i32(4),
                        fx.Int32(
                            load_i32_global_agent(ca(i32(CC_OFF) + k * i32(XST) + e))
                        )
                        + base,
                    )
            gpu.barrier()
            rl, rlw = rsrc(lrix), rsrc(lwt)
            for kk in range_constexpr(NRI):
                if oks[kk]:
                    r = tid + i32(kk * NT)
                    pos = lds_ld_i32(L, i32(LB2) + es[kk] * i32(4)) + rk[kk]
                    lo_ = lbase(k) + pos
                    bst(
                        (r0 + r) - B * i32(K1), rl, lo_ * i32(4), 0, AUX_SYS
                    )  # chunk-local route index
                    bst(
                        fx.Int32(bld(rtw, (r0 + r) * i32(4), 0, T.i32, AUX_SYS)),
                        rlw,
                        lo_ * i32(4),
                        0,
                        AUX_SYS,
                    )
            done_last(k, P_S3, cnt(P_S3, s))

        @flyc.jit
        def item_u_own(k, j0):
            tid = own_tid()
            o, s = chunk(k)
            B = i32(tp) * o
            wait_ph(k, P_S3, cnt(P_S3, s))
            if const_expr(
                not SXW
            ):  # the plan only waited for the routing: the rows now
                wait_puts(QX, NSBA, k)
                wait_puts(QXS, NSBA, k)
                gpu.barrier()
            nu = fx.Int32(
                load_i32_global_agent(ca(i32(C_UN) + k * i32(256) + xc * i32(32)))
            )
            j = j0
            if j < nu:
                rut = rsrc(utab)
                ent = ((k * i32(8) + xc) * ucap + j) * i32(16)
                e = uni(fx.Int32(bld(rut, ent, 0, T.i32, AUX_SYS)))
                r0 = uni(fx.Int32(bld(rut, ent + i32(4), 0, T.i32, AUX_SYS)))
                rows = uni(fx.Int32(bld(rut, ent + i32(8), 0, T.i32, AUX_SYS)))
                rl = rsrc(lrix + fx.Int64(lbase(k) * i32(4)))
                rlw = rsrc(lwt + fx.Int64(lbase(k) * i32(4)))
                rx = rsrc(wp("xq") + fx.Int64(B * i32(H // 2)))
                rxs = rsrc(wp("xs") + fx.Int64(B * i32(H // 32)))
                nrt = i32(tp) * s * i32(K1)
                r_rt = rsrc(routes + rbase(k), nrt * i32(H * 2))
                unit_body(
                    L,
                    tid,
                    e,
                    r0,
                    rows,
                    rl,
                    rlw,
                    rx,
                    rxs,
                    w1,
                    w2,
                    rsrc(w1s),
                    rsrc(w2s),
                    r_rt,
                    nrt * i32(H * 2),
                )
                # release: the unit's route rows out of this XCD's L2 (agent-scope release
                # fence: buffer_wbl2 sc1 + wait)
                wait_vm(0)
                gpu.barrier()
                if tid == i32(0):
                    fence_release(fx.rocdl.SyncScope.Agent)
                done_last(k, P_U, i32(-1))  # counts units done (no single last)

        @flyc.jit
        def item_rq(k, j):
            tid = own_tid()
            lane = tid % i32(64)
            o, s = chunk(k)
            B = i32(tp) * o
            wait_ph(k, P_S2, i32(1))  # the unit plan (C_UN)
            un = i32(0)
            for x in range_constexpr(8):
                un = un + fx.Int32(
                    load_i32_global_agent(ca(i32(C_UN + 32 * x) + k * i32(256)))
                )
            wait_ph(k, P_U, un)
            rx = rsrc(routes + rbase(k))
            rd = rsrc(wp("eq"))
            rids_ = rsrc(wp("ids"))

            def rvalid(
                wrow, t
            ):  # (token, slot) routes to an expert: else its route row is not written
                e_ = fx.Int32(
                    bld(
                        rids_,
                        ((B + wrow) * i32(K1) + i32(t)) * i32(4),
                        0,
                        T.i32,
                        AUX_SYS,
                    )
                )
                return (e_ >= i32(0)) & (e_ < i32(NE))

            sb = s // i32(NSB)
            nb = (i32(tp) * sb) // i32(ZQ)
            jx = il(j, cnt(P_RQ, s))
            b = jx // nb
            rem = (jx - b * nb) * i32(ZQ)
            pd = rem // sb
            w0 = pd * s + b * sb + (rem - pd * sb)  # first (dest rank, row) of the item
            own = pd == rank

            nrt_ = i32(tp) * s * i32(K1)

            def rq_load(
                ii,
            ):  # token ii of the item: its K1 route rows (this thread's columns)
                if const_expr(R8):
                    return [
                        [
                            (
                                bld(
                                    rx,
                                    ((w0 + i32(ii)) * i32(K1) + i32(t)) * i32(H)
                                    + col(tid, kk) * i32(8),
                                    0,
                                    T.vec(2, T.i32),
                                    CAUX,
                                ),
                                bld(
                                    rx,
                                    nrt_ * i32(H)
                                    + ((w0 + i32(ii)) * i32(K1) + i32(t)) * i32(H // 32)
                                    + col(tid, kk) // i32(4),
                                    0,
                                    T.i8,
                                    CAUX,
                                ),
                            )
                            for t in range(K1)
                        ]
                        for kk in range(PIT)
                    ]
                return [
                    [
                        bld(
                            rx,
                            (
                                at_col(
                                    ((w0 + i32(ii)) * i32(K1) + i32(t)) * i32(H),
                                    tid,
                                    kk,
                                )
                            )
                            * i32(2),
                            0,
                            T.vec(4, T.i32),
                            CAUX,
                        )
                        for t in range(K1)
                    ]
                    for kk in range(PIT)
                ]

            def rq_store(ii, ld):
                wrow = w0 + i32(ii)
                vt = [rvalid(wrow, t) for t in range(K1)]
                for kk in range_constexpr(PIT):
                    p = col(tid, kk)
                    acc = [fx.Float32(0.0)] * 8
                    for t in range_constexpr(K1):
                        v = (
                            fp8x8_decode(ld[kk][t])
                            if const_expr(R8)
                            else bf16x8_to_f32(ld[kk][t])
                        )
                        acc = [vt[t].select(a_ + b_, a_) for a_, b_ in zip(acc, v)]
                    yb = pack_bf16x8(acc)
                    if own:
                        bst(
                            yb,
                            r_y,
                            ((o + wrow - rank * s) * i32(H) + p * i32(8)) * i32(2),
                            0,
                            AUX_SYS,
                        )
                    else:
                        e_store(bf16x8_to_f32(yb), rd, B + wrow, p, lane)

            # token pipeline: the next D tokens' loads in flight while one is summed (D = ZQ:
            # every load of the item first)
            D = RQD or ZQ
            rqp = [rq_load(ii) for ii in range(min(D, ZQ))]
            for ii in range_constexpr(ZQ):
                if const_expr(ii + D < ZQ):
                    rqp.append(rq_load(ii + D))
                rq_store(ii, rqp[ii])
            done_last(k, P_RQ, cnt(P_RQ, s))
            if done_at(sbc(k, 0, b), nb):  # noqa: SIM102 - preserve FlyDSL branches
                if tid == i32(0):
                    tn = k * i32(NSB) + b
                    take_turn(QE, tn)
                    for p_ in range_constexpr(tp):
                        if i32(p_) != rank:
                            put(
                                i32(p_),
                                "erecv",
                                (B + rank * s + b * sb) * i32(RBE),
                                "eq",
                                (B + i32(p_) * s + b * sb) * i32(RBE),
                                sb * i32(RBE),
                                QE,
                            )
                    give_turn(QE, tn)

        def rf_head(k, j):
            # an RF item (ZQ rows of chunk k, from i0, in sub-block b), once every peer's E
            # rows of its sub-block and this rank's topk sums of it are in:
            # (tid, o, s, B, sb, i0, b, the received E rows)
            tid = own_tid()
            o, s = chunk(k)
            B = i32(tp) * o
            sb = s // i32(NSB)
            jx = il(j, cnt(P_RF, s))
            i0 = jx * i32(ZQ)
            b = i0 // sb
            wait_puts_n(QE, k * i32(NSB) + b + i32(1))
            wait_at(sbc(k, 0, b), (i32(tp) * sb) // i32(ZQ))
            return tid, o, s, B, sb, i0, b, rsrc(wp("erecv"))

        def y_sum(ly, lsrc):
            # y of a row's 8 columns: this rank's partial (ly) + every peer's (lsrc(src): its
            # E value, loaded here or earlier)
            f = bf16x8_to_f32(ly)
            for src_ in range_constexpr(tp):
                f = sum_live(f, e_val(lsrc(src_)), i32(src_) != rank)
            return f

        @flyc.jit
        def item_rf(k, j):
            tid, o, s, B, sb, i0, b, rv = rf_head(k, j)
            rq, rqs = rsrc(wp("q")), rsrc(wp("qs"))
            one = fx.Float32(1.0)
            # every load of the item first (one memory round trip), then the math
            ld = []
            for ii in range_constexpr(ZQ):
                i = i0 + i32(ii)
                lo = ((o + i) * i32(H)) * i32(2)
                row = []
                for kk in range_constexpr(PIT):
                    off = lo + col(tid, kk) * i32(16)
                    row.append(
                        (
                            bld(r_y, off, 0, T.vec(4, T.i32), CAUX),
                            [
                                e_ld(rv, B + i32(src_) * s + i, col(tid, kk))
                                for src_ in range(tp)
                            ],
                            bld(r_resin, off, 0, T.vec(4, T.i32), 0),
                        )
                    )
                ld.append(row)
            fss, sss = [], []
            for ii in range_constexpr(ZQ):
                i = i0 + i32(ii)
                lo = ((o + i) * i32(H)) * i32(2)
                fs = []
                ss = fx.Float32(0.0)
                for kk in range_constexpr(PIT):
                    off = lo + col(tid, kk) * i32(16)
                    ly, lr, lres = ld[ii][kk]
                    f = y_sum(ly, lambda src_, lr=lr: lr[src_])
                    # the MoE output y (this rank's rows)
                    f = [fx.Float32(x.to(fx.BFloat16).to(fx.Float32)) for x in f]
                    live = own_col(tid, kk)
                    if const_expr(live is None):
                        bst(pack_bf16x8(f), r_yout, off, 0)
                    else:
                        yv = pack_bf16x8(f)
                    f = [x + yy for x, yy in zip(f, bf16x8_to_f32(lres))]
                    f = [fx.Float32(x.to(fx.BFloat16).to(fx.Float32)) for x in f]
                    if const_expr(live is None):
                        bst(pack_bf16x8(f), r_out, off, 0)
                        for x in f:
                            ss = ss + x * x
                    else:
                        if live:
                            bst(yv, r_yout, off, 0)
                            bst(pack_bf16x8(f), r_out, off, 0)
                        for x in f:
                            ss = ss + live.select(x * x, fx.Float32(0.0))
                    fs.append(f)
                fss.append(fs)
                sss.append(ss.bitcast(fx.Int32))
            tots = blk_red_vec(L, tid, sss, f32_bits_add)
            xss, ams = [], []
            for ii in range_constexpr(ZQ):
                rcp = fx.Float32(
                    fmath.rsqrt(
                        tots[ii].bitcast(fx.Float32) / fx.Float32(float(H))
                        + fx.Float32(float(eps))
                    )
                )
                xs_, am = [], fx.Float32(0.0)
                for kk in range_constexpr(PIT):
                    wv = norm_w(r_normw, col(tid, kk))
                    xk = [
                        fx.Float32(
                            fx.Float32(gate_w(x * rcp, wk))
                            .to(fx.BFloat16)
                            .to(fx.Float32)
                        )
                        for x, wk in zip(fss[ii][kk], wv)
                    ]
                    live = own_col(tid, kk)
                    if const_expr(live is None):
                        am = am.maximumf(amax(xk))
                    else:
                        am = am.maximumf(live.select(amax(xk), fx.Float32(0.0)))
                    xs_.append(xk)
                xss.append(xs_)
                ams.append(am.bitcast(fx.Int32))
            amxs = blk_red_vec(L, tid, ams, f32_bits_max)
            for ii in range_constexpr(ZQ):
                i = i0 + i32(ii)
                scl = amxs[ii].bitcast(fx.Float32).maximumf(
                    fx.Float32(1e-10)
                ) / fx.Float32(448.0)
                inv = one / scl
                g = rank * m + o + i
                for kk in range_constexpr(PIT):
                    p = col(tid, kk)
                    qv = [x * inv for x in xss[ii][kk]]
                    d = fx.Vector.from_elements(
                        [fp8x4_pack(qv[0:4], one), fp8x4_pack(qv[4:8], one)],
                        fx.Int32,
                    )
                    live = own_col(tid, kk)
                    if const_expr(live is None):
                        bst(d, rq, g * i32(H) + p * i32(8), 0, AUX_SYS)
                    else:
                        if live:
                            bst(d, rq, g * i32(H) + p * i32(8), 0, AUX_SYS)
                if tid == i32(0):
                    bst(scl.bitcast(fx.Int32), rqs, g * i32(4), 0, AUX_SYS)
            done_last(k, P_RF, cnt(P_RF, s))
            # (nested ifs: preserve FlyDSL branches)
            if done_at(sbc(k, 1, b), sb // i32(ZQ)) & (  # noqa: SIM102
                i32(int(tailq == 1)) == i32(1)
            ):
                if tid == i32(0):
                    g0 = rank * m + o + b * sb
                    tn = k * i32(NSB) + b
                    take_turn(QQ, tn)
                    take_turn(QS, tn)
                    for p_ in range_constexpr(tp):
                        if i32(p_) != rank:
                            put(
                                i32(p_),
                                "q",
                                g0 * i32(H),
                                "q",
                                g0 * i32(H),
                                sb * i32(H),
                                QQ,
                            )
                            put(
                                i32(p_),
                                "qs",
                                g0 * i32(4),
                                "qs",
                                g0 * i32(4),
                                sb * i32(4),
                                QS,
                            )
                    give_turn(QQ, tn)
                    give_turn(QS, tn)

        @flyc.jit
        def item_rf_y(k, j):
            # io without tail: y = own partial + the peers' MoE-output rows (this rank's rows);
            # tailq 3: y rows to the "q" region (global row rank * m + ..), each sub-block then
            # put to every peer
            tid, o, s, B, sb, i0, b, rv = rf_head(k, j)
            ryg = rsrc(wp("q"))
            for ii in range_constexpr(ZQ):
                i = i0 + i32(ii)
                lo = ((o + i) * i32(H)) * i32(2)
                for kk in range_constexpr(PIT):
                    off = lo + col(tid, kk) * i32(16)
                    f = y_sum(
                        bld(r_y, off, 0, T.vec(4, T.i32), CAUX),
                        lambda src_, i=i, kk=kk: e_ld(
                            rv, B + i32(src_) * s + i, col(tid, kk)
                        ),
                    )
                    if const_expr(tailq == 3):
                        bst(
                            pack_bf16x8(f),
                            ryg,
                            (rank * m + o + i) * i32(H * 2) + col(tid, kk) * i32(16),
                            0,
                            AUX_SYS,
                        )
                    else:
                        bst(pack_bf16x8(f), r_yout, off, 0)
            done_last(k, P_RF, cnt(P_RF, s))
            if const_expr(tailq == 3):  # noqa: SIM102 - preserve FlyDSL branches
                if done_at(sbc(k, 1, b), sb // i32(ZQ)):  # noqa: SIM102
                    if tid == i32(0):
                        g0 = rank * m + o + b * sb
                        tn = k * i32(NSB) + b
                        take_turn(QQ, tn)
                        for p_ in range_constexpr(tp):
                            if i32(p_) != rank:
                                put(
                                    i32(p_),
                                    "q",
                                    g0 * i32(H * 2),
                                    "q",
                                    g0 * i32(H * 2),
                                    sb * i32(H * 2),
                                    QQ,
                                )
                        give_turn(QQ, tn)

        @flyc.jit
        def item_gw():
            for k in range_constexpr(C):
                wait_ph(i32(k), P_RF, cnt(P_RF, chunk(i32(k))[1]))
            if const_expr(tailq in (1, 3)):
                wait_puts(QQ, NSB, i32(C - 1))
            if const_expr(tailq == 1):
                wait_puts(QS, NSB, i32(C - 1))
            gpu.barrier()
            # every put of this launch landed: the next launch's per-(peer, queue) base, into
            # the other buffer (a late wait of this launch still reads this launch's base).
            # One thread per peer (tid, not lane: every wave would store it)
            if tid < i32(tp):  # noqa: SIM102 - preserve FlyDSL branches
                if tid != rank:
                    for qi in range_constexpr(len(QUEUES)):
                        q_ = QUEUES[qi]
                        addr = (consumed ^ fx.Int64(tp * NQ * 4)) + fx.Int64(
                            (tid * i32(NQ) + i32(q_)) * i32(4)
                        )
                        g_st_sys(addr, sig_base(tid, q_) + i32(C * PPC[q_]))

        # ======================= the work loop =======================
        def decode(slices, t):
            kind_r = i32(-1)
            k_r = i32(0)
            jj = i32(0)
            for si in range_constexpr(len(slices)):
                k, kind, lo_, b0, b1 = slices[si]
                hit = (t >= b0) & (t < b1)
                kind_r = hit.select(i32(kind), kind_r)
                k_r = hit.select(i32(k), k_r)
                jj = hit.select(lo_ + (t - b0), jj)
            return kind_r, k_r, jj

        rf_item = item_rf_y if tailq in (0, 3) else item_rf

        t = claim()
        alive_ = t < nall
        while alive_:
            kind_r, k_r, jj = decode(
                slices_of([chunk_v(i32(k)) for k in range(C)])[0], t
            )
            kind_r = uni(kind_r)
            k_r = uni(k_r)
            jj = uni(jj)
            if kind_r == i32(P_RB):
                item_rx(k_r, jj)
            elif kind_r == i32(P_S1):
                item_s1(k_r, jj)
            elif kind_r == i32(P_S2):
                item_s2(k_r)
            elif kind_r == i32(P_S3):
                item_s3(k_r, jj)
            elif (kind_r == i32(P_U)) | (kind_r == i32(P_UB)):
                item_u_own(k_r, jj)
            elif kind_r == i32(P_RQ):
                item_rq(k_r, jj)
            elif kind_r == i32(P_RF):
                rf_item(k_r, jj)
            if t == gw_lo:
                item_gw()
            gpu.barrier()
            t = claim()
            alive_ = t < nall

    kargs = list(inspect.signature(kernel).parameters)
    kernel = flyc.kernel(name=name, known_block_size=[NT, 1, 1])(kernel)

    def launch(
        wb: fx.Int64,
        pbases: fx.Int64,
        dc: fx.Int64,
        consumed: fx.Int64,
        offs: fx.Int64,
        xin: fx.Int64,
        ids: fx.Int64,
        tw: fx.Int64,
        normw: fx.Int64,
        resin: fx.Int64,
        yout: fx.Int64,
        resout: fx.Int64,
        ybuf: fx.Int64,
        routes: fx.Int64,
        lrix: fx.Int64,
        lwt: fx.Int64,
        utab: fx.Int64,
        w1: fx.Int64,
        w1s: fx.Int64,
        w2: fx.Int64,
        w2s: fx.Int64,
        ctrl: fx.Int64,
        chunks: fx.Int64,
        rank: fx.Int32,
        m: fx.Int32,
        ucap: fx.Int32,
        grid: fx.Int32,
        stream: fx.Stream,
    ):
        loc = locals()
        kernel(*[loc[k] for k in kargs]).launch(
            grid=(fx.Int64(grid), 1, 1), block=(NT, 1, 1), stream=stream
        )

    # launch forwards its arguments to the kernel by name
    if list(inspect.signature(launch).parameters) != kargs + ["grid", "stream"]:
        raise ValueError("compile_bigm_kernel: kernel and launch signatures differ")
    launch = flyc.jit(launch)
    launch._bigm_spec = spec
    return launch
