# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Fused TP MegaMoE kernel (a4w4, gfx950): the fused tail norm, the wave roles, the
kernel entry and its launcher. The device code is built per instance by the build_*
parts (comm, gemm, schedule) over one shared KernelCtx."""

from __future__ import annotations

import functools
import inspect

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import gpu, range_constexpr
from flydsl.expr import math as fmath

from ..kernels_common import ceildiv
from .comm import build_comm
from .common import (
    AUX_SYS,
    MAX_TP,
    V4I,
    amax,
    before,
    bf16x8_to_f32,
    bld,
    bst,
    fp8x4_pack,
    g_add_agent,
    g_ld_sys,
    g_st_sys,
    i32,
    lds_ld_i32,
    lds_st,
    pack_bf16x8,
    rsrc,
    traced,
    uni,
    wait_vm,
    wave_red,
)
from .gemm import build_gemm
from .mega_moe_tp_config import (
    C_EPOCH,
    C_TNL,
    CTRL_TNC,
    ERR_YAG,
    FLAG_TN,
    NT,
    NTT,
    TN_MAX,
    kernel_config,
)
from .schedule import build_schedule

__all__ = ["compile_mega_moe_tp"]


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
    kc.add(build_comm(kc))
    kc.add(build_gemm(kc))
    kc.add(build_schedule(kc))
    AG8, ARLL, DLL, L_CTL, L_RING, LBPF = kc.get("AG8 ARLL DLL L_CTL L_RING LBPF")
    LDS_BYTES, MLL, NCHA, TN, TPC = kc.get("LDS_BYTES MLL NCHA TN TPC")
    a_loader, ag8_convert, ag_send = kc.get("a_loader ag8_convert ag_send")
    _ag_send_meta, _ag_send_meta_ll = kc.get("_ag_send_meta _ag_send_meta_ll")
    claim_addr, comm_help, comm_wave = kc.get("claim_addr comm_help comm_wave")
    compute_units, const_expr, ctrl_at = kc.get("compute_units const_expr ctrl_at")
    _drop_stale, _dyn_zero, fin_key = kc.get("_drop_stale _dyn_zero fin_key")
    fin_owned, fin_split, finish = kc.get("fin_owned fin_split finish")
    init_lds, layout_tag, lb_finish = kc.get("init_lds layout_tag lb_finish")
    lb_g2_prefetch, name, peer_rs = kc.get("lb_g2_prefetch name peer_rs")
    peer_sel, poll_loop, poll_zero = kc.get("peer_sel poll_loop poll_zero")
    push_dll, quant_chunks, signal_loop = kc.get("push_dll quant_chunks signal_loop")
    yag_flush, yag_wait, zero_masked = kc.get("yag_flush yag_wait zero_masked")

    TN_V = H // 8
    TN_IT = ceildiv(TN_V, NTT)

    @traced
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

    @traced
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

    @traced
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
                fx.Float32(
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
                for p in range_constexpr(TPC):
                    bst(
                        d,
                        peer_rs(a, p, "off_qall"),
                        grow * i32(H) + q * i32(8),
                        0,
                        AUX_SYS,
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
            for p in range_constexpr(TPC):
                bst(
                    scale.bitcast(fx.Int32),
                    peer_rs(a, p, "off_sall"),
                    grow * i32(4),
                    0,
                    AUX_SYS,
                )
        wait_vm(0)
        gpu.barrier()
        if tid < i32(TPC):
            fo = fx.Int64((i32(FLAG_TN) + a["rank"] * i32(TN_MAX) + r) * i32(4))
            g_st_sys(peer_sel(a, tid) + fx.Int64(a["off_flag"]) + fo, a["epoch"])

    @traced
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

    @traced
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
            a["off_ids"] = off_ids + par * fx.Int64(16)
            a["off_w"] = off_w + par * fx.Int64(4)
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
