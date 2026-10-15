# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Fused EP MegaMoE kernel (gfx950): the wave roles, the kernel entry and its
launcher.

SCAFFOLD STATE: the role bodies (``dispatch_send`` / ``compute_units`` / ``a_loader`` /
``combine_send`` / ``combine_final`` / ``comm_help``) are stubs. They are filled in by
the ``build_*`` parts over one shared ``KernelCtx`` in the follow-up milestones:

* ``schedule.py``       -> in-kernel dispatch planner + ``compute_units``   (planner, compute)
* ``communication.py``  -> arena a2a dispatch / combine + epoch flags        (arena_flags, dispatch_send, combine)
* ``gemm.py``           -> GEMM1 + act + GEMM2 device math                   (compute)

``compile_mega_moe_ep`` is only invoked by the engine once those exist; importing this
module does not trace any device code (the decorators run when the function is called).
The structure here is the one-launch, grid=#CU, block=512 skeleton the parts plug into.
"""

from __future__ import annotations

import functools
import inspect

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import gpu

from .common import i32, uni
from .mega_moe_ep_config import (
    B_COMBINE,
    B_DISPATCH,
    B_LOADER,
    NT,
    NTT,
    kernel_config,
)

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
    """Build the launcher of one (shape, variant) EP instance.

    The warp roles run concurrently in one persistent CTA per CU: dispatch_send
    all-to-all scatters each token's payload to its top-k experts' owner ranks and
    publishes DISPATCH_READY; compute_units waits on DISPATCH_READY and runs the local
    experts' GEMM1 + act + GEMM2; combine_send scatters the weighted GEMM2 partials
    back to the source ranks and publishes COMBINE_READY; combine_final sums the top-k
    partials per source token. a_loader stages dispatched tokens into LDS for compute.
    """
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
    # TODO(planner, dispatch_send, compute, combine): populate kc with the device
    # functions built by the parts, then pull them out like the TP kernel does:
    #   kc.add(build_schedule(kc)); kc.add(build_communication(kc)); kc.add(build_gemm(kc))
    #   compute_units, comm_help, dispatch_send, a_loader, combine_send, combine_final, \
    #       init_lds, quant_chunks, finish = kc.get("...")
    name = kc.name  # kc.LDS_BYTES is wired into the Shared allocator in roles_entry

    # --- wave roles (SCAFFOLD stubs) -------------------------------------------------
    @flyc.jit
    def dispatch_send(L, tid, a):
        # TODO(dispatch_send): a2a scatter token payload to expert-owner ranks' arena,
        # atomic_add_system -> publish DISPATCH_READY[tile].
        pass

    @flyc.jit
    def a_loader(L, tid, a):
        # TODO(compute): stage dispatched tokens (per-source-row indexed) into LDS.
        pass

    @flyc.jit
    def compute_units(L, tid, a):
        # TODO(compute): wait DISPATCH_READY, local-expert GEMM1 + act + GEMM2 on the
        # planned units; 4 compute waves cooperate via cbar.
        pass

    @flyc.jit
    def combine_send(L, tid, a):
        # TODO(combine): GEMM2 epilog scatters weighted partials back to source rank,
        # publish COMBINE_READY[tile].
        pass

    @flyc.jit
    def combine_final(L, tid, a):
        # TODO(combine): poll COMBINE_READY, top-k weighted sum per source token.
        pass

    @flyc.jit
    def comm_help(L, tid, a, epoch):
        # TODO(arena_flags): drain pending a2a packets / poll flags with watchdog.
        pass

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
            combine_send(L, tid, a)
            comm_help(L, tid, a, epoch)
        assert B_DISPATCH == NT  # band constants wired consistently

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
        # TODO(scaffold->roles_entry): allocate Shared LDS, resolve peers p0..p7, read
        # the epoch flag, run init/quant, then roles(L, tid, a, epoch), then finish.
        peers = [p0, p1, p2, p3, p4, p5, p6, p7]
        mine = peers[0]
        for j in range(1, 8):
            mine = (rank == i32(j)).select(peers[j], mine)
        a["mine"] = fx.Int64(mine)
        a["recv_le"] = fx.Int64(mine) + off_recv  # own received-routing region
        a["nrecv"] = (
            npes * mtpr * i32(TOPK)
        )  # dynamic-planner scan bound (indexed recv)
        # TODO(roles_entry): allocate Shared LDS (real base) + init / quant before roles.
        L = uni(i32(0))  # placeholder LDS base
        roles(L, tid, a, epoch)

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
