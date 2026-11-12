# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Fused EP MegaMoE kernel (gfx950): the kernel entry and its launcher.

One persistent CTA per CU (grid=#CU, block=512 = 8 waves, all of them compute).
``compile_mega_moe_ep`` assembles the device parts over one shared ``KernelCtx``:

* ``schedule.build_schedule``       -> plan + compute_units + gather_recv_chunk
* ``communication.build_communication`` -> arena/flag primitives, dispatch sends,
  combine_final, cross-rank waits
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
    CTL_BAR_COUNT,
    CTL_BAR_GEN,
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
    TOKEN_SLOTS: int,
    E: int,
    EPR: int,
    N_RANKS: int,
    act: str = "silu",
    situ_beta: float = 1.0,
    situ_linear_beta: float = 1.0,
    swiglu_limit: float | None = None,
    a8: bool = False,
    SMALL_BATCH: bool = False,
    STREAM_WEIGHTS: bool = False,
):
    """Build the launcher of one (shape, variant) EP instance (see the module docstring)."""
    assert TOKEN_SLOTS % N_RANKS == 0
    kc = kernel_config(
        H=H,
        I=I,
        TOPK=TOPK,
        MT=MT,
        TOKEN_SLOTS=TOKEN_SLOTS,
        E=E,
        EPR=EPR,
        N_RANKS=N_RANKS,
        act=act,
        situ_beta=situ_beta,
        situ_linear_beta=situ_linear_beta,
        swiglu_limit=swiglu_limit,
        a8=a8,
        SMALL_BATCH=SMALL_BATCH,
        STREAM_WEIGHTS=STREAM_WEIGHTS,
    )
    kc.add(build_communication(kc))
    kc.add(build_gemm(kc))
    kc.add(build_schedule(kc))
    name, LDS_BYTES = kc.get("name LDS_BYTES")
    BITMAP_WORDS, LDS_ACTIVE, LDS_EXPERT_ROWS, LDS_CTRL = kc.get(
        "BITMAP_WORDS LDS_ACTIVE LDS_EXPERT_ROWS LDS_CTRL"
    )
    compute_units = kc.get("compute_units")[0]
    combine_final, barrier_init = kc.get("combine_final barrier_init")
    NT = kc.get("NT")[0]
    SMALL_BATCH = kc.get("SMALL_BATCH")[0]

    @flyc.jit
    def init_lds(L, tid):
        # zero the dynamic planner's LDS tables (active-expert bitmap + prefix + dense
        # list, per-local-expert counts) and the barrier control slots.
        for s_ in range(tid, i32(2 * BITMAP_WORDS + EPR), i32(NT)):
            lds_st(L, LDS_ACTIVE + i32(s_) * i32(4), i32(0))
        for s_ in range(tid, i32(EPR), i32(NT)):
            lds_st(L, LDS_EXPERT_ROWS + i32(s_) * i32(4), i32(0))
        if tid == i32(0):
            lds_st(L, LDS_CTRL + CTL_BAR_COUNT * 4, i32(0))
            lds_st(L, LDS_CTRL + CTL_BAR_GEN * 4, i32(0))

    Shared = fx.struct(
        type(
            "Shared", (), {"__annotations__": {"buf": fx.Array[fx.Int8, LDS_BYTES, 16]}}
        )
    )

    def mega_moe_ep_kernel(
        w1: fx.Int64,
        w1_scale: fx.Int64,
        w2: fx.Int64,
        w2_scale: fx.Int64,
        x: fx.Int64,
        x_scale: fx.Int64,
        topk_ids: fx.Int64,
        topk_weights: fx.Int64,
        out: fx.Int64,
        ctrl: fx.Int64,
        # rank j's symmetric-arena base as mapped into this process (0 past n_ranks); a
        # region lives at arena base + its off_* offset on every rank
        arena_rank0: fx.Int64,
        arena_rank1: fx.Int64,
        arena_rank2: fx.Int64,
        arena_rank3: fx.Int64,
        arena_rank4: fx.Int64,
        arena_rank5: fx.Int64,
        arena_rank6: fx.Int64,
        arena_rank7: fx.Int64,
        off_payload: fx.Int64,
        off_pscale: fx.Int64,
        off_recv: fx.Int64,
        off_part: fx.Int64,
        off_flag: fx.Int64,
        inter: fx.Int64,
        inter_scale: fx.Int64,
        chunk_payload_row: fx.Int64,
        chunk_recv_slot: fx.Int64,
        expert_recv_slots: fx.Int64,
        expert_count: fx.Int64,
        rank: fx.Int32,
        n_ranks: fx.Int32,
        epr: fx.Int32,
        m: fx.Int32,
        mtpr: fx.Int32,
    ):
        a = dict(locals())
        # n_ranks / epr / mtpr are the compiled-in N_RANKS / EPR / TOKEN_SLOTS //
        # N_RANKS (the host passes exactly those): as constants the per-peer guards fold
        # and e // epr, source_key // mtpr become multiply-shifts instead of runtime
        # divisions with spilled SGPRs
        n_ranks, epr, mtpr = i32(N_RANKS), i32(EPR), i32(TOKEN_SLOTS // N_RANKS)
        a.update(n_ranks=n_ranks, epr=epr, mtpr=mtpr)
        a["epoch"] = uni(g_ld_sys(fx.Int64(ctrl) + fx.Int64(EPOCH_SLOT * 4)))
        tid = fx.Int32(gpu.thread_id("x"))
        assert name
        lds = fx.SharedAllocator().allocate(Shared).peek()
        L = uni(fx.Int32(fx.ptrtoint(lds.buf.ptr)))
        arena_bases = [
            arena_rank0,
            arena_rank1,
            arena_rank2,
            arena_rank3,
            arena_rank4,
            arena_rank5,
            arena_rank6,
            arena_rank7,
        ]
        own_arena = arena_bases[0]
        for j in range_constexpr(1, 8):
            own_arena = (rank == i32(j)).select(arena_bases[j], own_arena)
        a["arena_bases"] = arena_bases
        a["own_arena"] = fx.Int64(own_arena)
        a["recv_expert"] = fx.Int64(own_arena) + off_recv  # own received-routing region
        # dynamic-planner scan bound (indexed recv)
        a["n_recv"] = n_ranks * mtpr * i32(TOPK)
        if const_expr(SMALL_BATCH):  # this launch's part bank
            a["off_part"] = off_part + fx.Int64(a["epoch"] & i32(1)) * fx.Int64(
                mtpr * i32(TOPK * (H + H // 8))
            )

        init_lds(L, tid)
        gpu.barrier()
        # recv_expert is sentinel'd on the host on THIS stream before the launch (stream
        # order); barrier_init is the cross-rank wait so a peer only dispatches once every
        # rank's fill is done -- no host sync/barrier, so forward() stays a stream pipeline.
        if const_expr(not SMALL_BATCH):
            barrier_init(tid, a)
        compute_units(L, tid, a)
        gpu.barrier()
        combine_final(tid, a)

    kargs = list(inspect.signature(mega_moe_ep_kernel).parameters)
    mega_moe_ep_kernel = flyc.kernel(name=name, known_block_size=[NT, 1, 1])(
        mega_moe_ep_kernel
    )

    def launch(
        w1: fx.Int64,
        w1_scale: fx.Int64,
        w2: fx.Int64,
        w2_scale: fx.Int64,
        x: fx.Int64,
        x_scale: fx.Int64,
        topk_ids: fx.Int64,
        topk_weights: fx.Int64,
        out: fx.Int64,
        ctrl: fx.Int64,
        arena_rank0: fx.Int64,
        arena_rank1: fx.Int64,
        arena_rank2: fx.Int64,
        arena_rank3: fx.Int64,
        arena_rank4: fx.Int64,
        arena_rank5: fx.Int64,
        arena_rank6: fx.Int64,
        arena_rank7: fx.Int64,
        off_payload: fx.Int64,
        off_pscale: fx.Int64,
        off_recv: fx.Int64,
        off_part: fx.Int64,
        off_flag: fx.Int64,
        inter: fx.Int64,
        inter_scale: fx.Int64,
        chunk_payload_row: fx.Int64,
        chunk_recv_slot: fx.Int64,
        expert_recv_slots: fx.Int64,
        expert_count: fx.Int64,
        rank: fx.Int32,
        n_ranks: fx.Int32,
        epr: fx.Int32,
        m: fx.Int32,
        mtpr: fx.Int32,
        n_cta: fx.Int32,
        stream: fx.Stream,
    ):
        loc = locals()
        mega_moe_ep_kernel(*[loc[k] for k in kargs]).launch(
            grid=(fx.Int64(n_cta), 1, 1), block=(NT, 1, 1), stream=stream
        )

    if list(inspect.signature(launch).parameters) != kargs + ["n_cta", "stream"]:
        raise ValueError("compile_mega_moe_ep: kernel and launch signatures differ")
    return flyc.jit(launch)
