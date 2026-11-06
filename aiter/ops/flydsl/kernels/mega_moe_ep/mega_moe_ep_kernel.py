# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Fused EP MegaMoE kernel (gfx950): the kernel entry and its launcher.

One persistent CTA per CU (grid=#CU, block=512 = 8 waves, all of them compute).
``compile_mega_moe_ep`` assembles the device parts over one shared ``KernelCtx``:

* ``schedule.build_schedule``       -> dyn_plan + compute_units + gather_recv_chunk
* ``communication.build_communication`` -> arena/flag primitives, dispatch sends,
  combine_final, watchdog
* ``gemm.build_gemm``               -> the inline-A GEMM1 / GEMM2 (+ fused combine)

``compute_units`` runs the a2a dispatch, the planner, the GEMM passes and the combine
scatter; the finish phase does the source-side top-k weighted sum (combine_final).
"""

from __future__ import annotations

import functools
import inspect

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr

from .common import g_ld_sys, i32, lds_st, uni
from .communication import build_communication
from .gemm import build_gemm
from .mega_moe_ep_config import (
    C_BARCNT,
    C_BARGEN,
    EPOCH_SLOT,
    kernel_config,
)
from .schedule import build_schedule

__all__ = ["compile_mega_moe_ep"]


@functools.cache
def compile_mega_moe_ep(
    *,
    H: int,
    I: int,
    TOPK: int,
    MT: int,
    TMAX: int,
    E: int,
    EPR: int,
    NPES: int,
    act: str = "silu",
    situ_beta: float = 1.0,
    situ_linear_beta: float = 1.0,
    swiglu_limit: float | None = None,
    a8: bool = False,
    SB: bool = False,
    WNT: bool = False,
):
    """Build the launcher of one (shape, variant) EP instance (see the module docstring)."""
    assert TMAX % NPES == 0
    kc = kernel_config(
        H=H,
        I=I,
        TOPK=TOPK,
        MT=MT,
        TMAX=TMAX,
        E=E,
        EPR=EPR,
        NPES=NPES,
        act=act,
        situ_beta=situ_beta,
        situ_linear_beta=situ_linear_beta,
        swiglu_limit=swiglu_limit,
        a8=a8,
        SB=SB,
        WNT=WNT,
    )
    kc.add(build_communication(kc))
    kc.add(build_gemm(kc))
    kc.add(build_schedule(kc))
    name, LDS_BYTES = kc.get("name LDS_BYTES")
    NBW, L_DYN, L_DCNT, L_CTL = kc.get("NBW L_DYN L_DCNT L_CTL")
    compute_units = kc.get("compute_units")[0]
    combine_final, barrier_init = kc.get("combine_final barrier_init")
    NTT = kc.get("NTT")[0]
    SBXP = kc.get("SBXP")[0]

    @flyc.jit
    def init_lds(L, tid, a):
        # zero the dynamic planner's LDS tables (active-expert bitmap + prefix + dense
        # list, per-local-expert counts) and the barrier control slots.
        for s_ in range(tid, i32(2 * NBW + EPR), i32(NTT)):
            lds_st(L, L_DYN + i32(s_) * i32(4), i32(0))
        for s_ in range(tid, i32(EPR), i32(NTT)):
            lds_st(L, L_DCNT + i32(s_) * i32(4), i32(0))
        if tid == i32(0):
            lds_st(L, L_CTL + C_BARCNT * 4, i32(0))
            lds_st(L, L_CTL + C_BARGEN * 4, i32(0))

    Shared = fx.struct(
        type(
            "Shared", (), {"__annotations__": {"buf": fx.Array[fx.Int8, LDS_BYTES, 16]}}
        )
    )

    def mega_moe_ep_kernel(
        w1: fx.Int64,
        w1s: fx.Int64,
        w2: fx.Int64,
        w2s: fx.Int64,
        x: fx.Int64,
        xs: fx.Int64,
        ids: fx.Int64,
        wts: fx.Int64,
        out: fx.Int64,
        ctrl: fx.Int64,
        p0: fx.Int64,
        p1: fx.Int64,
        p2: fx.Int64,
        p3: fx.Int64,
        p4: fx.Int64,
        p5: fx.Int64,
        p6: fx.Int64,
        p7: fx.Int64,
        off_payload: fx.Int64,
        off_pscale: fx.Int64,
        off_recv: fx.Int64,
        off_part: fx.Int64,
        off_flag: fx.Int64,
        off_out: fx.Int64,
        inter: fx.Int64,
        inters: fx.Int64,
        rowmap: fx.Int64,
        rowidx: fx.Int64,
        rowlist: fx.Int64,
        ecnt: fx.Int64,
        rank: fx.Int32,
        npes: fx.Int32,
        epr: fx.Int32,
        m: fx.Int32,
        mtpr: fx.Int32,
        dc: fx.Int64,
        sdst: fx.Int64,
        spos: fx.Int64,
        off_stage: fx.Int64,
        off_sstage: fx.Int64,
        off_rpos: fx.Int64,
        off_sdum: fx.Int64,
    ):
        a = dict(locals())
        # npes / epr / mtpr are the compiled-in NPES / EPR / TMAX // NPES (the host passes
        # exactly those): as constants the per-peer guards fold and e // epr, sk // mtpr
        # become multiply-shifts instead of runtime divisions with spilled SGPRs
        npes, epr, mtpr = i32(NPES), i32(EPR), i32(TMAX // NPES)
        a.update(npes=npes, epr=epr, mtpr=mtpr)
        a["epoch"] = uni(g_ld_sys(fx.Int64(ctrl) + fx.Int64(EPOCH_SLOT * 4)))
        tid = fx.Int32(gpu.thread_id("x"))
        assert name
        lds = fx.SharedAllocator().allocate(Shared).peek()
        L = uni(fx.Int32(fx.ptrtoint(lds.buf.ptr)))
        peers = [p0, p1, p2, p3, p4, p5, p6, p7]
        mine = peers[0]
        for j in range_constexpr(1, 8):
            mine = (rank == i32(j)).select(peers[j], mine)
        a["peer"] = peers
        a["mine"] = fx.Int64(mine)
        a["recv_le"] = fx.Int64(mine) + off_recv  # own received-routing region
        a["nrecv"] = (
            npes * mtpr * i32(TOPK)
        )  # dynamic-planner scan bound (indexed recv)
        if const_expr(SBXP):  # this launch's part bank
            a["off_part"] = off_part + fx.Int64(a["epoch"] & i32(1)) * fx.Int64(
                mtpr * i32(TOPK * (H + H // 8))
            )

        init_lds(L, tid, a)
        gpu.barrier()
        # recv_le is sentinel'd on the host on THIS stream before the launch (stream
        # order); barrier_init is the cross-rank wait so a peer only dispatches once every
        # rank's fill is done -- no host sync/barrier, so forward() stays a stream pipeline.
        if const_expr(not SBXP):
            barrier_init(L, tid, a)
        compute_units(L, tid, a)
        gpu.barrier()
        combine_final(L, tid, a)

    kargs = list(inspect.signature(mega_moe_ep_kernel).parameters)
    mega_moe_ep_kernel = flyc.kernel(name=name, known_block_size=[NTT, 1, 1])(
        mega_moe_ep_kernel
    )

    def launch(
        w1: fx.Int64,
        w1s: fx.Int64,
        w2: fx.Int64,
        w2s: fx.Int64,
        x: fx.Int64,
        xs: fx.Int64,
        ids: fx.Int64,
        wts: fx.Int64,
        out: fx.Int64,
        ctrl: fx.Int64,
        p0: fx.Int64,
        p1: fx.Int64,
        p2: fx.Int64,
        p3: fx.Int64,
        p4: fx.Int64,
        p5: fx.Int64,
        p6: fx.Int64,
        p7: fx.Int64,
        off_payload: fx.Int64,
        off_pscale: fx.Int64,
        off_recv: fx.Int64,
        off_part: fx.Int64,
        off_flag: fx.Int64,
        off_out: fx.Int64,
        inter: fx.Int64,
        inters: fx.Int64,
        rowmap: fx.Int64,
        rowidx: fx.Int64,
        rowlist: fx.Int64,
        ecnt: fx.Int64,
        rank: fx.Int32,
        npes: fx.Int32,
        epr: fx.Int32,
        m: fx.Int32,
        mtpr: fx.Int32,
        dc: fx.Int64,
        sdst: fx.Int64,
        spos: fx.Int64,
        off_stage: fx.Int64,
        off_sstage: fx.Int64,
        off_rpos: fx.Int64,
        off_sdum: fx.Int64,
        i32_grid: fx.Int32,
        stream: fx.Stream,
    ):
        loc = locals()
        mega_moe_ep_kernel(*[loc[k] for k in kargs]).launch(
            grid=(fx.Int64(i32_grid), 1, 1), block=(NTT, 1, 1), stream=stream
        )

    if list(inspect.signature(launch).parameters) != kargs + ["i32_grid", "stream"]:
        raise ValueError("compile_mega_moe_ep: kernel and launch signatures differ")
    return flyc.jit(launch)
