# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Fused EP MegaMoE kernel (gfx950): the wave roles, the kernel entry and its launcher.

One persistent CTA per CU (grid=#CU, block=512 = 8 waves). ``compile_mega_moe_ep``
assembles the device parts over one shared ``KernelCtx``:

* ``schedule.build_schedule``       -> dyn_plan + compute_units + gather_recv_chunk
* ``communication.build_communication`` -> arena/flag primitives, dispatch_send,
  combine_send / combine_final, watchdog

roles() splits the block by tid: compute (local-expert GEMM over the planned units) /
dispatch_send (a2a scatter) / a_loader / comm_help. init sentinels the received-routing
region; the finish phase does the source-side top-k weighted sum (combine_final).

SCAFFOLD STATE: the GEMM1+act+GEMM2 math (``schedule.unit_tile``), its A-loader and the
comm_help drain are no-op stubs pending the compute milestone + GPU bring-up; everything
else (planner, dispatch, combine addressing, flags) is wired. The per-unit GEMM2-epilog
scatter (``combine_send``) is called from compute once ``unit_tile`` produces output.
"""

from __future__ import annotations

import functools
import inspect
import os

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr

from .common import bst, i32, lds_st, now, rsrc, uni

# Device-phase profiling: when EP_PROF_DEV=1, each CTA's tid0 stamps now() into a
# per-CTA global `dbg` buffer at the kernel phase boundaries (entry / post-barrier_init
# / post-roles / post-combine_final). Off by default (zero runtime cost -- the writes
# are compile-time gated and the `dbg` arg is just an unused pointer).
_PROF = bool(int(os.environ.get("EP_PROF_DEV", "0")))
_PROF_SLOTS = 19  # 0:entry 1:post-barrier_init 2:post-roles 3:post-combine_final
#   4:post-dispatch_wait 5:post-dyn_plan 7:post-gemm1 8:post-gemm2 (schedule)
#   6: EP_OVL payload-wait tick sum | sender send ticks << 32 (schedule)
#   9:post-dispatch_send(NW8) 10:post-gather 11:post-payload-wait (schedule)
#   12:post-combine_wait (communication.combine_final)
#   13..18: gemm1 {kloop, pre-epi cbar, epi}, gemm2 {..} tick sums (gemm.py _pmark)
from .communication import build_communication
from .gemm import build_gemm
from .mega_moe_ep_config import (
    C_BARCNT,
    C_BARGEN,
    C_PLAN,
    NLOAD,
    RC_INTS,
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
    indexed_payload: bool = False,
    NW: int = 4,
):
    """Build the launcher of one (shape, variant) EP instance (see the module docstring)."""
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
        indexed_payload=indexed_payload,
        NW=NW,
    )
    kc.add(build_communication(kc))
    kc.add(build_gemm(kc))
    kc.add(build_schedule(kc))
    name, LDS_BYTES = kc.get("name LDS_BYTES")
    NBW, L_DYN, L_DCNT, L_CTL = kc.get("NBW L_DYN L_DCNT L_CTL")
    L_RCTL, L_PF = kc.get("L_RCTL L_PF")
    compute_units, dispatch_send = kc.get("compute_units dispatch_send")
    a_loader_units, w_lds_units = kc.get("a_loader_units w_lds_units")
    WL = kc.get("WL")[0]
    combine_final, barrier_init = kc.get("combine_final barrier_init")
    NT, NTT, B_LOADER = kc.get("NT NTT B_LOADER")  # per-launch warp-spec bands (not globals)

    @flyc.jit
    def init_lds(L, tid, a):
        # zero the dynamic planner's LDS tables (active-expert bitmap + prefix + dense
        # list, per-local-expert counts) and the barrier / plan-ready control slots.
        for s_ in range(tid, i32(2 * NBW + EPR), i32(NTT)):
            lds_st(L, L_DYN + i32(s_) * i32(4), i32(0))
        for s_ in range(tid, i32(EPR), i32(NTT)):
            lds_st(L, L_DCNT + i32(s_) * i32(4), i32(0))
        # zero the A-loader ring counters (RC_ASEQ / RC_AFREE per bank + the go flags)
        for s_ in range(tid, i32(RC_INTS), i32(NTT)):
            lds_st(L, L_RCTL + i32(s_) * i32(4), i32(0))
        if const_expr(_PROF):  # noqa: SIM102 - compile-time gate, not dynamic
            if tid < i32(16):  # gemm.py _pmark tick sums + last mark, 8/9: EP_OVL waits
                lds_st(L, L_PF + tid * i32(4), i32(0))
        if tid == i32(0):
            lds_st(L, L_CTL + C_BARCNT * 4, i32(0))
            lds_st(L, L_CTL + C_BARGEN * 4, i32(0))
            lds_st(L, L_CTL + C_PLAN * 4, i32(0))

    @flyc.jit
    def comm_help(L, tid, a, epoch):
        # TODO(tune): assist dispatch / combine (drain pending a2a packets, poll flags).
        del L, tid, a, epoch

    @flyc.jit
    def roles(L, tid, a, epoch):
        del epoch
        if tid < i32(NT):  # compute band: planner + GEMM + combine (fused into GEMM2 at
            compute_units(L, tid, a)  # P==1; pass 3 in-band at P>1)
        elif tid < i32(B_LOADER):  # [NT, NT+64): a2a dispatch scatter (input)
            dispatch_send(L, tid % i32(64), a)
        elif tid < i32(B_LOADER + NLOAD):  # A-loader: stage GEMM1 A into the NAB-bank ring
            a_loader_units(L, tid, a)
        else:  # [B_LOADER+NLOAD, NTT): W-loader into the LDS W ring (EP_WLDS), else idle
            if const_expr(WL):  # noqa: SIM102 - compile-time gate, not dynamic
                w_lds_units(L, tid, a)

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
        gout: fx.Int64,
        rowmap: fx.Int64,
        rowidx: fx.Int64,
        rowlist: fx.Int64,
        ecnt: fx.Int64,
        dbg: fx.Int64,
        rank: fx.Int32,
        npes: fx.Int32,
        epr: fx.Int32,
        m: fx.Int32,
        mtpr: fx.Int32,
        epoch: fx.Int32,
    ):
        a = dict(locals())
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

        def _ts(slot):
            if const_expr(_PROF):  # noqa: SIM102 - compile-time gate, not dynamic
                if tid == i32(0):
                    bst(
                        now(),
                        rsrc(dbg),
                        (i32(gpu.block_id("x")) * i32(_PROF_SLOTS) + i32(slot))
                        * i32(8),
                        0,
                        0,
                    )

        init_lds(L, tid, a)
        gpu.barrier()
        _ts(0)
        # recv_le is sentinel'd on the host on THIS stream before the launch (stream
        # order); barrier_init is the cross-rank wait so a peer only dispatches once every
        # rank's fill is done -- no host sync/barrier, so forward() stays a stream pipeline.
        barrier_init(L, tid, a)
        _ts(1)
        roles(L, tid, a, epoch)
        _ts(2)
        gpu.barrier()
        combine_final(L, tid, a)
        _ts(3)

    kargs = list(inspect.signature(mega_moe_ep_kernel).parameters)
    mega_moe_ep_kernel = flyc.kernel(
        name=name + ("_prof" if _PROF else ""), known_block_size=[NTT, 1, 1]
    )(mega_moe_ep_kernel)

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
        gout: fx.Int64,
        rowmap: fx.Int64,
        rowidx: fx.Int64,
        rowlist: fx.Int64,
        ecnt: fx.Int64,
        dbg: fx.Int64,
        rank: fx.Int32,
        npes: fx.Int32,
        epr: fx.Int32,
        m: fx.Int32,
        mtpr: fx.Int32,
        epoch: fx.Int32,
        i32_grid: fx.Int32,
        stream: fx.Stream,
    ):
        loc = locals()
        _wpe = int(os.environ.get("EP_WPE", "0"))
        _va = {"value_attrs": {"rocdl.waves_per_eu": _wpe}} if _wpe else {}
        mega_moe_ep_kernel(*[loc[k] for k in kargs], **_va).launch(
            grid=(fx.Int64(i32_grid), 1, 1), block=(NTT, 1, 1), stream=stream
        )

    if list(inspect.signature(launch).parameters) != kargs + ["i32_grid", "stream"]:
        raise ValueError("compile_mega_moe_ep: kernel and launch signatures differ")
    return flyc.jit(launch)
