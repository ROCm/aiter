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

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import gpu

from .common import i32, lds_st, uni
from .communication import build_communication
from .mega_moe_ep_config import (
    B_COMBINE,
    B_LOADER,
    C_BARCNT,
    C_BARGEN,
    C_PLAN,
    NT,
    NTT,
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
    )
    kc.add(build_communication(kc))
    kc.add(build_schedule(kc))
    name, LDS_BYTES = kc.get("name LDS_BYTES")
    NBW, L_DYN, L_DCNT, L_CTL = kc.get("NBW L_DYN L_DCNT L_CTL")
    compute_units, dispatch_send = kc.get("compute_units dispatch_send")
    init_recv, combine_final = kc.get("init_recv combine_final")

    @flyc.jit
    def init_lds(L, tid, a):
        # zero the dynamic planner's LDS tables (active-expert bitmap + prefix + dense
        # list, per-local-expert counts) and the barrier / plan-ready control slots.
        for s_ in range(tid, i32(2 * NBW + EPR), i32(NTT)):
            lds_st(L, L_DYN + i32(s_) * i32(4), i32(0))
        for s_ in range(tid, i32(EPR), i32(NTT)):
            lds_st(L, L_DCNT + i32(s_) * i32(4), i32(0))
        if tid == i32(0):
            lds_st(L, L_CTL + C_BARCNT * 4, i32(0))
            lds_st(L, L_CTL + C_BARGEN * 4, i32(0))
            lds_st(L, L_CTL + C_PLAN * 4, i32(0))

    @flyc.jit
    def a_loader(L, tid, a):
        # TODO(compute): stage the gathered payload rows (L_RIX -> a["payload"]) into
        # LDS for the GEMM, like TP a_loader.
        del L, tid, a

    @flyc.jit
    def comm_help(L, tid, a, epoch):
        # TODO(tune): assist dispatch / combine (drain pending a2a packets, poll flags).
        del L, tid, a, epoch

    @flyc.jit
    def roles(L, tid, a, epoch):
        if tid < i32(NT):
            compute_units(L, tid, a)
            comm_help(L, tid, a, epoch)
        elif tid < i32(B_LOADER):  # [NT, NT+64)
            dispatch_send(L, tid % i32(64), a)
            comm_help(L, tid, a, epoch)
        elif tid < i32(B_COMBINE):  # [NT+64, NT+128)
            a_loader(L, tid, a)
            comm_help(L, tid, a, epoch)
        else:  # [NT+128, NTT)
            comm_help(L, tid, a, epoch)

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
        for j in range(1, 8):
            mine = (rank == i32(j)).select(peers[j], mine)
        a["peer"] = peers
        a["mine"] = fx.Int64(mine)
        a["recv_le"] = fx.Int64(mine) + off_recv  # own received-routing region
        a["nrecv"] = (
            npes * mtpr * i32(TOPK)
        )  # dynamic-planner scan bound (indexed recv)
        init_lds(L, tid, a)
        init_recv(L, tid, a)
        gpu.barrier()
        # TODO(bring-up): a cross-rank "inited" barrier here so a peer's dispatch_send
        # cannot race this rank's init_recv (sentinel) write.
        roles(L, tid, a, epoch)
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
        mega_moe_ep_kernel(*[loc[k] for k in kargs]).launch(
            grid=(fx.Int64(i32_grid), 1, 1), block=(NTT, 1, 1), stream=stream
        )

    if list(inspect.signature(launch).parameters) != kargs + ["i32_grid", "stream"]:
        raise ValueError("compile_mega_moe_ep: kernel and launch signatures differ")
    return flyc.jit(launch)
