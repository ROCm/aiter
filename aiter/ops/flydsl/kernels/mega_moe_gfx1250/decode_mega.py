# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Decode MegaMoE: quant, dispatch, GEMM1, GEMM2 and combine as ONE kernel (gfx1250).

The small-M counterpart of the staged compact path (plan, quant, dispatch,
GEMM1, GEMM2 + EP scatter, combine sync, reduce: seven launches), after the
gfx950 small-M MegaMoE (ROCm/aiter#6881). Every CTA of an all-resident grid of
4-wave CTAs runs the same phases:

1. PRODUCE. One wave per local route (token, k). Lane 0 reserves a slot of the
   destination expert with one SCOPE_SYS buffer atomic on that rank's
   ``running[parity][expert]`` (expert ``e`` owns rows ``[e * cap, (e + 1) *
   cap)`` of ``dmk_disp``; ``cap`` covers every token of every rank); the
   token's bf16 row, loaded first, is quantized to the fp4 wire in registers
   meanwhile (per-1x32 MX with the HIP quant's RoundUp scale and amax floor,
   so the bytes match the staged wire); the row and its ``(meta, w)`` rowmap
   pair then go straight into the slot with SCOPE_SYS stores. The wave that
   finishes this rank's last route stores the generation into every
   destination's ``src_done[rank]``.
2. PLAN. CTA 0 alone polls the peers' ``src_done`` (hundreds of CTAs polling
   one uncached line starve every access queued behind it), reads the expert
   counts, publishes them in ``PLAN_REPS`` replicas and opens every CTA's own
   gate line. Each CTA lays the m-tiles (expert-major, ``tile_m`` rows) out in
   LDS from its replica.
3. GEMM. One dynamic queue: every GEMM1 item ``(m, n)``, then every GEMM2
   item. A CTA's first item is static, later ones come from one agent claim
   each. GEMM1 is the stage-1 body of ``build_gemm_a8w4_tdm_task`` (act + fp4
   quant into a2, stored SCOPE_DEV); the item completing an m-tile's
   ``g1_cnt`` publishes ``g1_ready[m] = gen``. A GEMM2 item's A owners wait for
   ``g1_ready`` while its B owners stream W2; its EP-scatter epilogue writes
   each weighted row into its token owner's ``comb_inp`` slot with SCOPE_SYS
   gather stores. All GEMM1 items are claimed before any GEMM2 item.
4. CLOSE. Each CTA counts its GEMM2 units (CTA 0 one more, for resetting the
   next launch's counters); the count completing the total stores the
   generation into every peer's ``comb_flag[rank]``.
5. REDUCE. CTA 0 polls the peers' ``comb_flag`` and opens the gates; each CTA
   then sums its tokens' top-k slots in k order in fp32, the staged reduce's
   arithmetic, so the output is bitwise the staged one.

Forward progress: producers wait on nothing; the plan waits only on producers
(all ranks run theirs first); a GEMM1 item waits on nothing; a GEMM2 item waits
only on GEMM1 items already claimed; the reduce waits on the peers' closes,
which depend only on the above. All CTAs must be resident at once (the grid is
the CU count times ``DMK_CTAS_PER_CU``, and no other kernel may hold CUs).
Every spin traps past its limit.

No launch handshake: rank s writes d's ``dmk_disp`` / ``dmk_rowmap`` /
``running`` in launch L+1 only after s's launch L ended, which needed d's close
of L, which comes after every read d made of them in L (and after CTA 0 reset
the counters launch L+1 uses). Rows of ``comb_inp`` on s are written in L+1
only for tokens s produced in L+1, i.e. after s's reduce of L.

Generation and parity come from u64 entry tickets (``G`` per launch over
``N_TICKET`` counters): every rank launches once per layer, so they agree.
"""

from __future__ import annotations

import os

import flydsl.compiler as flyc
import flydsl.expr as fx
import mori.cco.device.flydsl as cco
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.expr import arith, const_expr, range_constexpr, rocdl
from flydsl.expr.rocdl import ballot, ds_bpermute, readlane
from flydsl.expr.typing import T

from aiter.ops.flydsl.kernels import communication_ops_utils as comm_ops
from aiter.ops.flydsl.kernels.buffer_ops import (
    buffer_load,
    buffer_store,
    create_buffer_resource_from_addr,
)
from aiter.ops.flydsl.kernels.moe_fused_route_quant_scatter import (
    _cvt_scalef32_pk8_fp4_bf16 as emit_cvt_scalef32_pk8_fp4_bf16,
)
from aiter.ops.flydsl.kernels.mxfp4_preshuffle_gfx1250_tdm import (
    build_gemm_a8w4_tdm_task,
    launch_gemm_a8w4_tdm,
)
from aiter.ops.flydsl.kernels.quant_utils import emit_mx_e8m0_scale
from aiter.utility.mx_types import MX_DEFAULT_ROUND_MODE as _MX_ROUND_UP
from aiter.utility.mx_types import MxDtypeInt as _MxDtype

from .compact_plan import _wave32_inclusive_scan_i32
from .config import _WAVE_SIZE as WAVE
from .s1f_stage1 import (
    _OOB,
    GEN_MOD,
    SCOPE_DEV,
    SCOPE_SYS,
    _buffer_atomic_add_ret,
    _i32_lo,
    _now,
    _store_i64_sys,
    _trap,
    _uniform,
)

#: HIP ``kAmaxFloor`` (csrc/include/quant_amax_floor.h): the staged wire quant
#: seeds every group's amax with it, so an all-zero group gets the same e8m0.
AMAX_FLOOR = 1e-10

#: Local control block (int32 words, zeroed once by the host). Each counter on
#: its own 4 KB page: every CTA hits the entry ticket at launch, and an atomic
#: queued behind those waits tens of us (S1F's CTL_PHEAD note).
CTL_ENTRY = 0  # u64 entry tickets: N_TICKET of them, TICKET_STRIDE words apart
N_TICKET = 64
TICKET_STRIDE = 16
CTL_PROD = 2048  # + 32 * parity: routes produced
CTL_CLOSE = 3072  # + 32 * parity: GEMM2 units closed (+1 for the counter reset)
CTL_QHEAD = 4096  # + 32 * parity: the GEMM work queue head
CTL_ERR = 5120  # routes dropped for a full expert slot (never expected)
#: Per m-tile: g1_cnt at word 0 and g1_ready at word 32, each on its own
#: 128 B line, so the ready polls do not queue the completion atomics.
CTL_G1 = 6144
G1_STRIDE = 64
G1_READY = 32
#: Words between two CTAs' gate words (one 128 B line each), and where the
#: packed plan starts inside a plan replica.
GATE_STRIDE = 32
PLAN_PK = 32
#: Plan replicas: CTA c reads replica c % PLAN_REPS, so a line is read by
#: G / PLAN_REPS CTAs, not by all of them (32 readers on a line cost ~10 us).
#: A replica is the experts' row counts, 16 bits each.
PLAN_REPS = 128

#: Words between two experts' slot counters: each on its own 128 B line, so
#: the peers' atomics on different experts do not serialize on one line.
RUN_STRIDE = 32

#: Arena control region (``dmk_ctl``), bytes. ``running[2][EPR_PAD]`` takes the
#: peers' slot atomics; ``src_done`` / ``comb_flag`` one 256 B line per source.
FLAG_STRIDE = 256

#: Polls (each with an s_sleep) before a wait traps, and the sleep between them.
SPIN_LIMIT = 1 << 22
POLL_SLEEP = int(os.environ.get("AITER_DMK_POLL_SLEEP", "32"))
#: CTA 0 is the only poller of the peers' flags; it can poll tightly.
LEAD_SLEEP = 1
#: A CTA polls only its own gate line, so a short sleep costs no one.
GATE_SLEEP = 32

#: Phase stamps per CTA under AITER_DMK_TRACE: start, ticket, produced,
#: planned, GEMM1 done, GEMM2 done, combine flags seen, end.
TR_N = 24
RT_N = 1024

#: Elements a reduce wave owns per item: 32 lanes x 16.
RED_ELEMS = WAVE * 16


def dmk_ctl_arena_bytes(*, npes: int, experts_per_rank: int) -> int:
    """Bytes of the arena's ``dmk_ctl`` region: slot counters, then src_done,
    then comb_flag (4 KB apart)."""
    return dmk_src_off(experts_per_rank) + (4 << 10) + npes * FLAG_STRIDE


def dmk_epr_pad(experts_per_rank: int) -> int:
    """Experts rounded up to whole wave chunks, an even number of them past
    one (the plan packs two 16-bit counts per dword across chunk pairs)."""
    ch = (int(experts_per_rank) + WAVE - 1) // WAVE
    if ch > 1 and ch % 2:
        ch += 1
    return ch * WAVE


def dmk_src_off(experts_per_rank: int) -> int:
    """The peer flags start 64 KB past the slot counters: CTA 0 polls them
    while the peers' slot atomics land on the counters."""
    run_bytes = 2 * dmk_epr_pad(experts_per_rank) * RUN_STRIDE * 4
    return (run_bytes + (64 << 10) - 1) // (64 << 10) * (64 << 10)


def dmk_ctl_layout(*, experts_per_rank: int, cap: int, tile_m: int, workers: int):
    """Word offsets of the local control block: the per-m-tile GEMM1 lines,
    then per CTA a plan slot (gate word, n_mt, and its own copy of the plan:
    per expert ``inclusive m-tile end << 16 | rows``) and a combine gate line.
    No line is read or polled by more than one CTA: hundreds of CTAs on one
    uncached line serialize at its DRAM bank."""
    mt_cap = int(experts_per_rank) * (int(cap) // int(tile_m))
    epr_pad = dmk_epr_pad(experts_per_rank)
    plan = CTL_G1 + mt_cap * G1_STRIDE
    plan_slot = (epr_pad // 2 + 31) // 32 * 32
    gate_p = plan + PLAN_REPS * plan_slot
    gate_c = gate_p + int(workers) * GATE_STRIDE
    return {
        "plan": plan,
        "plan_slot": plan_slot,
        "gate_p": gate_p,
        "gate_c": gate_c,
        "words": gate_c + int(workers) * GATE_STRIDE,
    }


def dmk_ctl_words(*, experts_per_rank: int, cap: int, tile_m: int, workers: int) -> int:
    """i32 words of the local control block."""
    return dmk_ctl_layout(
        experts_per_rank=experts_per_rank, cap=cap, tile_m=tile_m, workers=workers
    )["words"]


def _load_i32_sys(rsrc, word, *, volatile=True):
    return fx.Int32(
        buffer_load(
            rsrc,
            word,
            vec_width=1,
            dtype=T.i32,
            cache_modifier=SCOPE_SYS | ((1 << 31) if volatile else 0),
        )
    )


def _load_i32_dev(rsrc, word):
    """Volatile SCOPE_DEV i32 load (a device-local word another CTA stores)."""
    return fx.Int32(
        buffer_load(
            rsrc,
            word,
            vec_width=1,
            dtype=T.i32,
            cache_modifier=SCOPE_DEV | (1 << 31),
        )
    )


def make_decode_mega(
    *,
    rank,
    npes,
    experts,
    experts_per_rank,
    topk,
    model_dim,
    inter_dim,
    max_tok_per_rank,
    cap,
    row_stride,
    off_disp,
    off_rowmap,
    off_ctl,
    off_comb_inp,
    comb_slot_stride,
    arena_handle,
    stage1_act,
    workers,
    tile_m=16,
    tile_n=512,
    tile_k=128,
    num_buffers=2,
    tile_n2=512,
    tile_k2=128,
    num_buffers2=2,
    ctas_per_cu=2,
):
    """Build the decode megakernel; returns a ``@flyc.jit`` launcher.

    Launcher arguments (positional): ``arg_a2``, ``arg_disp``,
    ``arg_disp_scale``, ``arg_w1``, ``arg_w1_scale``, ``arg_a2_scale``,
    ``arg_w2``, ``arg_w2_scale``, ``arg_rowmap`` (pointers), then i64
    ``arena`` (window handle), ``x``, ``ids``, ``wts``, ``out``, ``ctl``, then
    ``i32_tok`` (this rank's tokens), ``f32_swiglu_limit``, ``stream``.
    """
    if WAVE != 32:
        raise ValueError("the decode megakernel requires gfx1250 wave32")
    H, I = int(model_dim), int(inter_dim)
    E, EPR, K = int(experts), int(experts_per_rank), int(topk)
    TM, G = int(tile_m), int(workers)
    CAP, ROW = int(cap), int(row_stride)
    PAYLOAD = H // 2
    SCALE_B = H // 32
    NG = H // 32 // WAVE  # 32-column groups per lane
    if H % (32 * WAVE) or H % RED_ELEMS:
        raise ValueError(f"model_dim={H} must be a multiple of {32 * WAVE}")
    if PAYLOAD + SCALE_B > ROW or ROW % 16:
        raise ValueError("wire row must hold the payload and its e8m0 row")
    if CAP % TM:
        raise ValueError(f"cap={CAP} must be a multiple of tile_m={TM}")
    if G % WAVE or G % N_TICKET:
        raise ValueError(f"workers={G} must be a multiple of {WAVE} and {N_TICKET}")
    if npes > WAVE or E != npes * EPR:
        raise ValueError("experts must split evenly over at most 32 ranks")
    EPR_PAD = dmk_epr_pad(EPR)
    EPR_CH = EPR_PAD // WAVE
    SRC_OFF = dmk_src_off(EPR)
    CF_OFF = SRC_OFF + (4 << 10)
    MT_PER_E = CAP // TM
    MT_CAP = EPR * MT_PER_E
    N1 = 2 * I
    NT1 = N1 // int(tile_n)
    NT2 = H // int(tile_n2)
    if N1 % int(tile_n) or H % int(tile_n2):
        raise ValueError("tile_n / tile_n2 must divide 2*inter_dim / model_dim")
    DEST_STRIDE = int(max_tok_per_rank) * K
    RED_CH = H // RED_ELEMS
    ROWS = EPR * CAP
    _lay = dmk_ctl_layout(experts_per_rank=EPR, cap=CAP, tile_m=TM, workers=G)
    CTL_WORDS = _lay["words"]
    PLAN_W, PLAN_SLOT = _lay["plan"], _lay["plan_slot"]
    GATE_P, GATE_C = _lay["gate_p"], _lay["gate_c"]
    if MT_CAP > 0xFFFF or CAP > 0xFFFF:
        raise ValueError("the packed plan holds m-tile ends and rows in 16 bits")
    TR_BASE = CTL_WORDS  # i64 stamps [G][TR_N] after the control words
    RT_BASE = TR_BASE + 2 * G * TR_N  # i64 [RT_N][4] per-route stamps
    CTL_ARENA_B = dmk_ctl_arena_bytes(npes=npes, experts_per_rank=EPR)

    from aiter.ops.flydsl.grouped_moe_gfx1250 import get_wmma_m_rep

    rep2 = get_wmma_m_rep(TM, int(tile_n2), 1, 4, "gemm2")
    # Debug only: AITER_DMK_PHASES masks phases out (1 produce routes, 2 GEMM1,
    # 4 GEMM2, 8 reduce); the protocol (tickets, flags, close) always runs.
    PH = int(os.environ.get("AITER_DMK_PHASES", "15"))
    # Debug only: AITER_DMK_TRACE=1 has tid 0 of every CTA stamp the phase
    # boundaries into the trace words after the control block.
    TRACE = int(os.environ.get("AITER_DMK_TRACE", "0"))
    # Debug only: the hand-off cache policies (A loads of GEMM2, GEMM1 output,
    # EP scatter).
    # A at the default policy: only the m-tile's own GEMM2 units read its a2
    # rows, after g1_ready, and the launch started with no line of them cached.
    POL_A = int(os.environ.get("AITER_DMK_POL_A", "0"))
    POL_C = int(os.environ.get("AITER_DMK_POL_C", str(SCOPE_DEV)))
    POL_S = int(os.environ.get("AITER_DMK_POL_S", str(SCOPE_SYS)))

    @flyc.jit
    def launch(
        arg_a2: fx.Pointer,
        arg_disp: fx.Pointer,
        arg_disp_scale: fx.Pointer,
        arg_w1: fx.Pointer,
        arg_w1_scale: fx.Pointer,
        arg_a2_scale: fx.Pointer,
        arg_w2: fx.Pointer,
        arg_w2_scale: fx.Pointer,
        arg_rowmap: fx.Pointer,
        arena: fx.Int64,
        addr_x: fx.Int64,
        addr_ids: fx.Int64,
        addr_wts: fx.Int64,
        addr_out: fx.Int64,
        addr_ctl: fx.Int64,
        i32_tok: fx.Int32,
        f32_swiglu_limit: fx.Float32,
        stream: fx.Stream,
    ):
        lds_holder = [None]
        meta1_holder = [None]
        meta2_holder = [None]
        ready2_holder = [None]
        common = dict(  # noqa: C408 - the builder's keyword names
            m_warp=1,
            n_warp=4,
            out_is_f16=0,
            a_is_fp4=1,
            n_experts=EPR,
            has_bias=0,
            cluster_n=1,
            cluster_m=1,
            next_stage_prefetch=0,
            num_waves_per_tensor_tdm=2,
            tdm_as_in_prologue=0,
            tdm_b_th=0,
            lds_soa_load_interleave=0,
            persistent_workers=G,
            rolled_drain=0,
            a_preshuffled=0,
            m_align=0,
            explicit_tile=1,
            lds_base=lds_holder,
            raw_drain=True,
            defer_c_wait=False,
            tdm_snop=True,
        )
        g1_task, block, g1_name, _, lay1 = build_gemm_a8w4_tdm_task(
            K=H,
            tile_m=TM,
            tile_n=tile_n,
            tile_k=tile_k,
            num_buffers=min(num_buffers, max(1, H // tile_k)),
            stage1_act=stage1_act,
            stage1_quant_out=1,
            quant_wmma_rep=rep2,
            enable_ep_scatter=0,
            ep_arena_handle=0,
            ep_combine_input_offset=0,
            ep_slot_stride_bytes=0,
            ep_destination_stride=0,
            ep_world_size=0,
            ep_quant_bits=0,
            row_major_ascale=1,
            a_row_stride_bytes=ROW,
            a_scale_row_stride_bytes=ROW,
            tile_meta=meta1_holder,
            c_cache_modifier=POL_C,
            **common,
        )
        g2_task, block2, g2_name, _, lay2 = build_gemm_a8w4_tdm_task(
            K=I,
            tile_m=TM,
            tile_n=tile_n2,
            tile_k=tile_k2,
            num_buffers=min(num_buffers2, max(1, I // tile_k2)),
            stage1_act=0,
            stage1_quant_out=0,
            quant_wmma_rep=1,
            enable_ep_scatter=1,
            ep_arena_handle=arena_handle,
            ep_combine_input_offset=off_comb_inp,
            ep_slot_stride_bytes=comb_slot_stride,
            ep_destination_stride=DEST_STRIDE,
            ep_world_size=npes,
            ep_quant_bits=0,
            row_major_ascale=0,
            a_row_stride_bytes=0,
            a_scale_row_stride_bytes=0,
            tile_meta=meta2_holder,
            a_ready=ready2_holder,
            a_cache_modifier=POL_A,
            scatter_cache_policy=POL_S,
            **common,
        )
        if block != 4 * WAVE or block2 != block:
            raise ValueError(f"expected 4-wave GEMM blocks, got {block}/{block2}")
        ARENA = max(int(lay1["arena"]), int(lay2["arena"]))
        # LDS beyond the GEMM arena: broadcast words, then the plan's per-expert
        # inclusive m-tile ends and row counts.
        BC_B = 64
        PLAN_B = 2 * EPR_PAD * 4
        CTRL_B = BC_B + PLAN_B
        if int(ctas_per_cu) * (((ARENA + CTRL_B) + 2047) // 2048 * 2048) > 327680:
            raise ValueError(
                f"{ctas_per_cu} CTAs of {ARENA + CTRL_B} B LDS do not fit one CU"
            )
        kname = (
            f"decode_mega_w{G}_r{rank}of{npes}_c{CAP}"
            + (f"_ph{PH}" if PH != 15 else "")
            + f"__{g1_name}__{g2_name}"
        )
        # Keep the separate-kernel launcher in this cache key's dependency set.
        _ = launch_gemm_a8w4_tdm

        @flyc.kernel(name=kname, known_block_size=[block, 1, 1])
        def kernel(
            arg_a2: fx.Pointer,
            arg_disp: fx.Pointer,
            arg_disp_scale: fx.Pointer,
            arg_w1: fx.Pointer,
            arg_w1_scale: fx.Pointer,
            arg_a2_scale: fx.Pointer,
            arg_w2: fx.Pointer,
            arg_w2_scale: fx.Pointer,
            arg_rowmap: fx.Pointer,
            arena: fx.Int64,
            addr_x: fx.Int64,
            addr_ids: fx.Int64,
            addr_wts: fx.Int64,
            addr_out: fx.Int64,
            addr_ctl: fx.Int64,
            i32_tok: fx.Int32,
            f32_swiglu_limit: fx.Float32,
        ):
            tid = fx.Int32(fx.thread_idx.x)
            lane = tid & fx.Int32(WAVE - 1)
            wave = _uniform(tid >> fx.Int32(5))
            bid = fx.Int32(fx.block_idx.x)
            is_t0 = tid == fx.Int32(0)

            smem = fx.SharedAllocator(static=False)
            arena_ptr = smem.allocate(ARENA, 128)._ptr
            lds_holder[0] = arena_ptr
            ctrl_ptr = smem.allocate(CTRL_B, 16)._ptr
            bc = fx.Int32(
                arith.index_cast(T.i32, fx.index_cast(T.index, fx.ptrtoint(ctrl_ptr)))
            )
            lds_end = bc + fx.Int32(BC_B)
            lds_cnt = lds_end + fx.Int32(EPR_PAD * 4)

            win = cco.CachedWindow(arena)
            win_base = win.base
            win_stride = win.stride

            def peer_va(pe, off):
                return win_base + fx.Int64(pe) * win_stride + fx.Int64(off)

            my_ctl = peer_va(rank, off_ctl)
            rsrc_ctl = create_buffer_resource_from_addr(
                addr_ctl, num_records_bytes=4 * CTL_WORDS
            )
            rsrc_actl = create_buffer_resource_from_addr(
                my_ctl, num_records_bytes=CTL_ARENA_B
            )
            tok = fx.Int32(i32_tok)

            def stamp(k):
                if const_expr(TRACE):  # noqa: SIM102 - preserve DSL staging
                    if is_t0:
                        # Relaxed: a release store would flush L2 per stamp.
                        _store_i64_sys(
                            addr_ctl
                            + fx.Int64(TR_BASE * 4)
                            + fx.Int64(bid * fx.Int32(TR_N * 8) + fx.Int32(k * 8)),
                            _now(),
                        )

            stamp(0)

            # ---- entry ticket (u64) -> generation, parity ----
            # CTA b takes its ticket on counter b % N_TICKET, which G / N_TICKET
            # CTAs share and every launch advances by exactly that much: one
            # counter for all G CTAs queued the last ones ~15 us.
            if is_t0:
                t64 = fx.Int64(
                    comm_ops.atomic_add_agent(
                        addr_ctl
                        + fx.Int64(CTL_ENTRY * 4)
                        + fx.Int64(
                            (bid & fx.Int32(N_TICKET - 1)) * fx.Int32(TICKET_STRIDE * 4)
                        ),
                        fx.Int64(1),
                    )
                )
                l64 = t64 // fx.Int64(G // N_TICKET)
                comm_ops.store_i32_lds(
                    fx.Int64(bc), _i32_lo(l64 % fx.Int64(GEN_MOD)) + fx.Int32(1)
                )
                comm_ops.store_i32_lds(
                    fx.Int64(bc + fx.Int32(4)), _i32_lo(l64) & fx.Int32(1)
                )
            rocdl.s_wait_dscnt(0)
            rocdl.s_barrier_signal(-1)
            rocdl.s_barrier_wait(-1)
            gen = _uniform(comm_ops.load_i32_lds(fx.Int64(bc)))
            par = _uniform(comm_ops.load_i32_lds(fx.Int64(bc + fx.Int32(4))))
            other = fx.Int32(1) - par

            stamp(1)
            # ---- CTA 0: reset the next launch's counters ----
            # The peers' slot atomics of the next launch land in
            # running[other] only after this rank's close, which counts this
            # reset as one unit once its stores have retired.
            if (bid == fx.Int32(0)) & (wave == fx.Int32(0)):
                for j in range_constexpr(EPR_CH):
                    buffer_store(
                        fx.Int32(0),
                        rsrc_actl,
                        (other * fx.Int32(EPR_PAD) + lane + fx.Int32(j * WAVE))
                        * fx.Int32(RUN_STRIDE),
                        cache_modifier=SCOPE_SYS,
                    )
                if lane < fx.Int32(3):
                    buffer_store(
                        fx.Int32(0),
                        rsrc_ctl,
                        fx.Int32(CTL_PROD)
                        + lane * fx.Int32(CTL_CLOSE - CTL_PROD)
                        + other * fx.Int32(32),
                        cache_modifier=SCOPE_DEV,
                    )
                rocdl.s_wait_storecnt(0)

            # ======================== 1. PRODUCE ========================
            rsrc_x = create_buffer_resource_from_addr(addr_x)
            rsrc_ids = create_buffer_resource_from_addr(addr_ids)
            rsrc_wts = create_buffer_resource_from_addr(addr_wts)
            wave_gid = bid * fx.Int32(4) + wave
            n_mine = fx.Int32(0)
            # One wave per route (t, k): the token's quant is redone per route
            # (cheap next to the remote round trip) so the routes go out in
            # parallel. Its lane 0 reserves the destination slot first; the
            # atomic is in flight across the quant.
            n_routes = tok * fx.Int32(K)
            for r in range(wave_gid, n_routes, G * 4):
                n_mine = n_mine + fx.Int32(1)
                t = r // fx.Int32(K)
                gid = fx.Int32(
                    buffer_load(rsrc_ids, r, vec_width=1, dtype=T.i32, is_scalar=True)
                )
                w_bits = fx.Int32(
                    buffer_load(rsrc_wts, r, vec_width=1, dtype=T.i32, is_scalar=True)
                )
                valid = (gid >= fx.Int32(0)) & (gid < fx.Int32(E if PH & 1 else 0))
                gid_s = fx.Int32(valid.select(gid, fx.Int32(0)))
                dst = gid_s // fx.Int32(EPR)
                le = gid_s - dst * fx.Int32(EPR)
                live0 = valid & (lane == fx.Int32(0))
                # The row loads go out before the slot atomic: loads retire in
                # issue order, so behind the remote round trip the quant could
                # not start until the atomic returned.
                raws = []
                for jj in range_constexpr(NG):
                    in_dw = t * fx.Int32(H // 2) + (
                        lane + fx.Int32(jj * WAVE)
                    ) * fx.Int32(16)
                    for c in range_constexpr(4):
                        raws.append(
                            buffer_load(
                                rsrc_x,
                                in_dw + fx.Int32(4 * c),
                                vec_width=4,
                                dtype=T.i32,
                            )
                        )
                # Branch-free buffer atomic: idle lanes address past the
                # resource and touch no memory (an add of 0 by each idle lane
                # on one scratch word serialized ~1700 atomics per launch), and
                # the return is waited on only at its first use.
                rsrc_run = create_buffer_resource_from_addr(
                    peer_va(dst, off_ctl), num_records_bytes=SRC_OFF
                )
                if const_expr(TRACE):
                    t_issue = _now()
                off_l = _buffer_atomic_add_ret(
                    rsrc_run,
                    fx.Int32(
                        live0.select(
                            (par * fx.Int32(EPR_PAD) + le) * fx.Int32(RUN_STRIDE * 4),
                            fx.Int32(_OOB),
                        )
                    ),
                    1,
                    SCOPE_SYS,
                )
                if const_expr(TRACE):
                    rocdl.s_wait_loadcnt(0)
                    t_back = _now()
                    if (lane == fx.Int32(0)) & (r < fx.Int32(RT_N)):
                        rt_va = (
                            addr_ctl
                            + fx.Int64(RT_BASE * 4)
                            + fx.Int64(r * fx.Int32(32))
                        )
                        _store_i64_sys(rt_va, t_issue)
                        _store_i64_sys(rt_va + fx.Int64(8), t_back)
                        _store_i64_sys(rt_va + fx.Int64(24), fx.Int64(dst))
                # Lane l owns 32-column groups l + 32 * jj: 64 B of bf16 in,
                # 16 B of fp4 and one e8m0 byte out (compact_plan._wire_quant).
                packed = []
                scales = []
                for jj in range_constexpr(NG):
                    amax = fx.Float32(AMAX_FLOOR)
                    chunks = []
                    for c in range_constexpr(4):
                        bf = fx.Vector(raws[4 * jj + c]).bitcast(fx.BFloat16)
                        chunks.append(bf)
                        f32x8 = bf.to(fx.Float32)
                        for e_ in range_constexpr(8):
                            amax = fx.max(amax, abs(f32x8[e_]))
                    e8m0 = emit_mx_e8m0_scale(
                        amax, mode=_MX_ROUND_UP, dtype=_MxDtype.FP4_E2M1
                    )
                    scale_f32 = (fx.Int32(e8m0) << fx.Int32(23)).bitcast(fx.Float32)
                    for c in range_constexpr(4):
                        packed.append(
                            fx.Int32(
                                emit_cvt_scalef32_pk8_fp4_bf16(
                                    chunks[c], scale_f32, i32_ty=T.i32
                                )
                            )
                        )
                    scales.append(fx.Int32(e8m0) & fx.Int32(0xFF))
                # Lane q < 8 packs the e8m0 bytes of groups 4q .. 4q+3 of each
                # 32-group slab into one dword.
                sc_dw = []
                for jj in range_constexpr(NG):
                    w_ = fx.Int32(0)
                    for b in range_constexpr(4):
                        src = (
                            (lane & fx.Int32(7)) * fx.Int32(4) + fx.Int32(b)
                        ) * fx.Int32(4)
                        w_ = w_ | (
                            fx.Int32(ds_bpermute(T.i32, src, scales[jj]))
                            << fx.Int32(8 * b)
                        )
                    sc_dw.append(w_)
                off = _uniform(readlane(T.i32, arith.unwrap(off_l), 0))
                if const_expr(TRACE):
                    stamp(8)
                if valid & (off < fx.Int32(CAP)):
                    row = le * fx.Int32(CAP) + off
                    # Sized resources: a masked buffer store sends its idle
                    # lanes to offset 0x7FFFFFFF, which must be OOB.
                    rsrc_row = create_buffer_resource_from_addr(
                        peer_va(dst, off_disp) + fx.Int64(row) * fx.Int64(ROW),
                        num_records_bytes=ROW,
                    )
                    for jj in range_constexpr(NG):
                        buffer_store(
                            fx.Vector.from_elements(
                                packed[4 * jj : 4 * jj + 4], dtype=fx.Int32
                            ),
                            rsrc_row,
                            (lane + fx.Int32(jj * WAVE)) * fx.Int32(4),
                            cache_modifier=SCOPE_SYS,
                        )
                    for jj in range_constexpr(NG):
                        buffer_store(
                            sc_dw[jj],
                            rsrc_row,
                            fx.Int32(PAYLOAD // 4 + jj * 8) + lane,
                            mask=lane < fx.Int32(8),
                            cache_modifier=SCOPE_SYS,
                        )
                    rsrc_rm = create_buffer_resource_from_addr(
                        peer_va(dst, off_rowmap) + fx.Int64(row) * fx.Int64(8),
                        num_records_bytes=8,
                    )
                    if const_expr(TRACE):  # noqa: SIM102 - preserve DSL staging
                        if (lane == fx.Int32(0)) & (r < fx.Int32(RT_N)):
                            _store_i64_sys(
                                addr_ctl
                                + fx.Int64(RT_BASE * 4)
                                + fx.Int64(r * fx.Int32(32))
                                + fx.Int64(16),
                                _now(),
                            )
                    meta = fx.Int32(rank * DEST_STRIDE) + r
                    buffer_store(
                        fx.Vector.from_elements([meta, w_bits], dtype=fx.Int32),
                        rsrc_rm,
                        fx.Int32(0),
                        mask=lane == fx.Int32(0),
                        cache_modifier=SCOPE_SYS,
                    )
                if valid & (off >= fx.Int32(CAP)) & (lane == fx.Int32(0)):
                    comm_ops.atomic_add_agent(
                        addr_ctl + fx.Int64(CTL_ERR * 4), fx.Int32(1)
                    )
            stamp(10)
            # Publish once this rank's last token has every route in place:
            # the stores above retire (SCOPE_SYS) before the count, and the
            # wave completing the count stores gen into every destination.
            rocdl.s_wait_storecnt(0)
            stamp(11)
            done_before = fx.Int32(0)
            if (n_mine > fx.Int32(0)) & (lane == fx.Int32(0)):
                done_before = fx.Int32(
                    comm_ops.atomic_add_agent(
                        addr_ctl + fx.Int64(CTL_PROD * 4) + fx.Int64(par) * 128, n_mine
                    )
                )
            done_before = _uniform(readlane(T.i32, arith.unwrap(done_before), 0))
            last = (n_mine > fx.Int32(0)) & (done_before + n_mine == n_routes)
            last = last | (
                (n_routes == fx.Int32(0)) & (bid == fx.Int32(0)) & (wave == fx.Int32(0))
            )
            if last & (lane < fx.Int32(npes)):
                comm_ops.store_i32_global_system_monotonic(
                    peer_va(lane, off_ctl) + fx.Int64(SRC_OFF + rank * FLAG_STRIDE),
                    gen,
                )

            stamp(2)

            # ======================== 2. PLAN ========================
            # CTA 0 alone polls the peers' src_done (hundreds of CTAs polling
            # the same uncached lines starve every access queued behind them),
            # lays out the plan, publishes it to the local control block and
            # then opens each CTA's own gate line.
            def poll_peer_flags(flag_off):
                src = fx.Int32((lane < fx.Int32(npes)).select(lane, fx.Int32(0)))
                flag_w = fx.Int32(flag_off // 4) + src * fx.Int32(FLAG_STRIDE // 4)
                cur = _load_i32_sys(rsrc_actl, flag_w)
                bad = fx.Int32(ballot(T.i32, (lane < fx.Int32(npes)) & (cur != gen)))
                it = fx.Int32(0)
                while (bad != fx.Int32(0)) & (it < fx.Int32(SPIN_LIMIT)):
                    fx.rocdl.s_sleep(LEAD_SLEEP)
                    cur = _load_i32_sys(rsrc_actl, flag_w)
                    bad = fx.Int32(
                        ballot(T.i32, (lane < fx.Int32(npes)) & (cur != gen))
                    )
                    it = it + fx.Int32(1)
                if it >= fx.Int32(SPIN_LIMIT):
                    _trap()

            def open_gates(gate_w, stride):
                """Wave 0 of CTA 0: once its earlier stores retired, open
                every CTA's gate."""
                rocdl.s_wait_storecnt(0)
                for c in range_constexpr(G // WAVE):
                    buffer_store(
                        gen,
                        rsrc_ctl,
                        fx.Int32(gate_w)
                        + (lane + fx.Int32(c * WAVE)) * fx.Int32(stride),
                        cache_modifier=SCOPE_DEV,
                    )

            def wait_gate(gate_w, stride):
                rdy_w = fx.Int32(gate_w) + bid * fx.Int32(stride)
                cur = _uniform(_load_i32_dev(rsrc_ctl, rdy_w))
                it = fx.Int32(0)
                while (cur != gen) & (it < fx.Int32(SPIN_LIMIT)):
                    fx.rocdl.s_sleep(GATE_SLEEP)
                    cur = _uniform(_load_i32_dev(rsrc_ctl, rdy_w))
                    it = it + fx.Int32(1)
                if it >= fx.Int32(SPIN_LIMIT):
                    _trap()

            def plan_from_counts(cnts):
                """Wave 0: lay the m-tiles out from the per-expert row counts
                (lane l, chunk j: expert l + 32 j) into the LDS plan."""
                carry = fx.Int32(0)
                for j in range_constexpr(EPR_CH):
                    e_ = lane + fx.Int32(j * WAVE)
                    c_ = cnts[j]
                    mt = (c_ + fx.Int32(TM - 1)) // fx.Int32(TM)
                    incl = _wave32_inclusive_scan_i32(mt, lane) + carry
                    comm_ops.store_i32_lds(
                        fx.Int64(lds_end + e_ * fx.Int32(4)),
                        (incl << fx.Int32(16)) | c_,
                    )
                    carry = _uniform(readlane(T.i32, arith.unwrap(incl), WAVE - 1))
                if lane == fx.Int32(0):
                    comm_ops.store_i32_lds(fx.Int64(bc + fx.Int32(8)), carry)

            if bid == fx.Int32(0):
                if wave == fx.Int32(0):
                    poll_peer_flags(SRC_OFF)
                    stamp(14)
                    cnts = []
                    for j in range_constexpr(EPR_CH):
                        e_ = lane + fx.Int32(j * WAVE)
                        c_ = _load_i32_sys(
                            rsrc_actl,
                            (par * fx.Int32(EPR_PAD) + e_) * fx.Int32(RUN_STRIDE),
                            volatile=False,
                        )
                        c_ = fx.Int32((e_ < fx.Int32(EPR)).select(c_, fx.Int32(0)))
                        c_ = fx.Int32((c_ > fx.Int32(CAP)).select(fx.Int32(CAP), c_))
                        cnts.append(c_)
                    plan_from_counts(cnts)
                    # Pack 16-bit counts: replica dword w holds experts 2w, 2w+1.
                    # Lane l of half h builds dword 16 h + l / 2 ... via
                    # bpermute: dword w = cnt(2w) | cnt(2w + 1) << 16.
                    for j2 in range_constexpr(EPR_CH // 2 if EPR_CH > 1 else 1):
                        w_ = lane + fx.Int32(j2 * WAVE)
                        lo_e = w_ * fx.Int32(2)
                        hi_e = lo_e + fx.Int32(1)
                        lo = fx.Int32(0)
                        hi = fx.Int32(0)
                        for j in range_constexpr(EPR_CH):
                            src_lo = (lo_e - fx.Int32(j * WAVE)) & fx.Int32(WAVE - 1)
                            src_hi = (hi_e - fx.Int32(j * WAVE)) & fx.Int32(WAVE - 1)
                            v_lo = fx.Int32(
                                ds_bpermute(T.i32, src_lo * fx.Int32(4), cnts[j])
                            )
                            v_hi = fx.Int32(
                                ds_bpermute(T.i32, src_hi * fx.Int32(4), cnts[j])
                            )
                            in_lo = (lo_e >= fx.Int32(j * WAVE)) & (
                                lo_e < fx.Int32((j + 1) * WAVE)
                            )
                            in_hi = (hi_e >= fx.Int32(j * WAVE)) & (
                                hi_e < fx.Int32((j + 1) * WAVE)
                            )
                            lo = fx.Int32(in_lo.select(v_lo, lo))
                            hi = fx.Int32(in_hi.select(v_hi, hi))
                        comm_ops.store_i32_lds(
                            fx.Int64(lds_cnt + w_ * fx.Int32(4)),
                            lo | (hi << fx.Int32(16)),
                        )
                rocdl.s_wait_dscnt(0)
                rocdl.s_barrier_signal(-1)
                rocdl.s_barrier_wait(-1)
                stamp(15)
                # Every wave of CTA 0 writes a quarter of the replicas; wave 0
                # then opens the gates.
                pk = []
                for j2 in range_constexpr(EPR_CH // 2 if EPR_CH > 1 else 1):
                    pk.append(
                        fx.Int32(
                            comm_ops.load_i32_lds(
                                fx.Int64(
                                    lds_cnt + (lane + fx.Int32(j2 * WAVE)) * fx.Int32(4)
                                )
                            )
                        )
                    )
                for c in range_constexpr(PLAN_REPS // 4):
                    slot_w = fx.Int32(PLAN_W) + (
                        wave * fx.Int32(PLAN_REPS // 4) + fx.Int32(c)
                    ) * fx.Int32(PLAN_SLOT)
                    for j2 in range_constexpr(len(pk)):
                        buffer_store(
                            pk[j2],
                            rsrc_ctl,
                            slot_w + fx.Int32(j2 * WAVE) + lane,
                            cache_modifier=SCOPE_DEV,
                        )
                rocdl.s_wait_storecnt(0)
                rocdl.s_barrier_signal(-1)
                rocdl.s_barrier_wait(-1)
                stamp(16)
                if wave == fx.Int32(0):
                    open_gates(GATE_P, GATE_STRIDE)
                    stamp(17)
            else:
                if wave == fx.Int32(0):
                    wait_gate(GATE_P, GATE_STRIDE)
                    stamp(9)
                    slot_w = fx.Int32(PLAN_W) + (
                        bid & fx.Int32(PLAN_REPS - 1)
                    ) * fx.Int32(PLAN_SLOT)
                    pkw = []
                    for j2 in range_constexpr(EPR_CH // 2 if EPR_CH > 1 else 1):
                        pkw.append(
                            _load_i32_dev(rsrc_ctl, slot_w + fx.Int32(j2 * WAVE) + lane)
                        )
                    # Unpack: expert e = lane + 32 j sits in dword e / 2.
                    cnts = []
                    for j in range_constexpr(EPR_CH):
                        e_ = lane + fx.Int32(j * WAVE)
                        w_ = e_ >> fx.Int32(1)
                        j2 = j // 2
                        src = (w_ - fx.Int32(j2 * WAVE)) & fx.Int32(WAVE - 1)
                        d_ = fx.Int32(ds_bpermute(T.i32, src * fx.Int32(4), pkw[j2]))
                        c_ = fx.Int32(
                            ((e_ & fx.Int32(1)) == fx.Int32(0)).select(
                                d_ & fx.Int32(0xFFFF), d_ >> fx.Int32(16)
                            )
                        )
                        cnts.append(
                            fx.Int32((e_ < fx.Int32(EPR)).select(c_, fx.Int32(0)))
                        )
                    plan_from_counts(cnts)
            rocdl.s_wait_dscnt(0)
            rocdl.s_barrier_signal(-1)
            rocdl.s_barrier_wait(-1)
            n_mt = _uniform(comm_ops.load_i32_lds(fx.Int64(bc + fx.Int32(8))))
            stamp(3)

            def decode_mtile(i):
                """(expert, GEMM m-tile index, valid rows) of plan m-tile i: the
                expert is the number of inclusive m-tile ends <= i."""
                e_ = fx.Int32(0)
                for j in range_constexpr(EPR_CH):
                    e_j = lane + fx.Int32(j * WAVE)
                    end_j = fx.Int32(
                        comm_ops.load_i32_lds(fx.Int64(lds_end + e_j * fx.Int32(4)))
                    ) >> fx.Int32(16)
                    e_ = e_ + _popc_i32(
                        fx.Int32(ballot(T.i32, (e_j < fx.Int32(EPR)) & (end_j <= i)))
                    )
                e_ = _uniform(e_)
                e_prev = fx.Int32((e_ > fx.Int32(0)).select(e_ - fx.Int32(1), e_))
                prev = fx.Int32(
                    comm_ops.load_i32_lds(fx.Int64(lds_end + e_prev * fx.Int32(4)))
                ) >> fx.Int32(16)
                prev = _uniform(fx.Int32((e_ > fx.Int32(0)).select(prev, fx.Int32(0))))
                jt = i - prev
                cnt = _uniform(
                    fx.Int32(
                        comm_ops.load_i32_lds(fx.Int64(lds_end + e_ * fx.Int32(4)))
                    )
                    & fx.Int32(0xFFFF)
                )
                rows = cnt - jt * fx.Int32(TM)
                rows = fx.Int32((rows > fx.Int32(TM)).select(fx.Int32(TM), rows))
                return e_, e_ * fx.Int32(MT_PER_E) + jt, rows

            def set_meta1(meta):
                meta1_holder[0] = meta

            def set_meta2(meta, ready_fn):
                meta2_holder[0] = meta
                ready2_holder[0] = ready_fn

            n1 = fx.Int32(N1)
            n2 = fx.Int32(H)
            rows_cap = fx.Int32(ROWS)

            total_units = n_mt * fx.Int32(NT2 if PH & 4 else 0) + fx.Int32(1)

            def close_units(n_units):
                """Wave 0: count this CTA's closed units. The count completing
                ``total_units`` (every unit's scatter retired, SCOPE_SYS, before
                it was counted) tells every peer this rank's rows are in its
                comb_inp."""
                old = fx.Int32(0)
                if lane == fx.Int32(0):
                    old = fx.Int32(
                        comm_ops.atomic_add_agent(
                            addr_ctl + fx.Int64(CTL_CLOSE * 4) + fx.Int64(par) * 128,
                            n_units,
                        )
                    )
                old = _uniform(readlane(T.i32, arith.unwrap(old), 0))
                if (old + n_units == total_units) & (lane < fx.Int32(npes)):
                    comm_ops.store_i32_global_system_monotonic(
                        peer_va(lane, off_ctl) + fx.Int64(CF_OFF + rank * FLAG_STRIDE),
                        gen,
                    )

            # ================== 3./4. GEMM1 then GEMM2 items ==================
            # One queue: every GEMM1 item (m-major), then every GEMM2 item. A
            # CTA's first item is static (its block id); each later one comes
            # from one agent claim, issued before the current item's body and
            # resolved after it. All GEMM1 items are claimed before any GEMM2
            # item, so a GEMM2 item waits only on GEMM1 items already running.
            U1 = n_mt * fx.Int32(NT1 if PH & 2 else 0)
            U = U1 + n_mt * fx.Int32(NT2 if PH & 4 else 0)
            bc_next = fx.Int64(bc + fx.Int32(12))
            u = bid
            n2_done = fx.Int32(0)
            while u < U:
                claim = _buffer_atomic_add_ret(
                    rsrc_ctl,
                    fx.Int32(
                        is_t0.select(
                            fx.Int32(CTL_QHEAD * 4) + par * fx.Int32(128),
                            fx.Int32(_OOB),
                        )
                    ),
                    1,
                    SCOPE_DEV,
                )
                if u < U1:
                    i = u // fx.Int32(NT1)
                    n = u - i * fx.Int32(NT1)
                    e_, mt_idx, rows = decode_mtile(i)
                    set_meta1((e_, rows))
                    g1_task(
                        arg_a2,
                        arg_disp,
                        arg_w1,
                        arg_disp_scale,
                        arg_w1_scale,
                        arg_disp,
                        arg_disp,
                        arg_a2_scale,
                        arg_disp,
                        rows_cap,
                        n1,
                        f32_swiglu_limit,
                        fx.Float32(1.0),
                        fx.Float32(1.0),
                        mt_idx,
                        n,
                    )
                    # Every wave waited its C / e8m0 stores (SCOPE_DEV) before
                    # the body's closing barrier: count the item.
                    if is_t0:
                        cnt_va = (
                            addr_ctl
                            + fx.Int64(CTL_G1 * 4)
                            + fx.Int64(i) * fx.Int64(G1_STRIDE * 4)
                        )
                        old = fx.Int32(comm_ops.atomic_add_agent(cnt_va, fx.Int32(1)))
                        if old == fx.Int32(NT1 - 1):
                            buffer_store(
                                gen,
                                rsrc_ctl,
                                fx.Int32(CTL_G1 + G1_READY) + i * fx.Int32(G1_STRIDE),
                                cache_modifier=SCOPE_DEV,
                            )
                            comm_ops.atomic_add_agent(cnt_va, fx.Int32(-NT1))
                else:
                    n2_done = n2_done + fx.Int32(1)
                    v = u - U1
                    i = v // fx.Int32(NT2)
                    n = v - i * fx.Int32(NT2)
                    e_, mt_idx, rows = decode_mtile(i)

                    def a_ready(i=i):
                        if const_expr(PH & 16):
                            return
                        stamp(12)
                        rdy_w = fx.Int32(CTL_G1 + G1_READY) + i * fx.Int32(G1_STRIDE)
                        cur = _uniform(_load_i32_dev(rsrc_ctl, rdy_w))
                        it = fx.Int32(0)
                        while (cur != gen) & (it < fx.Int32(SPIN_LIMIT)):
                            fx.rocdl.s_sleep(POLL_SLEEP)
                            cur = _uniform(_load_i32_dev(rsrc_ctl, rdy_w))
                            it = it + fx.Int32(1)
                        if it >= fx.Int32(SPIN_LIMIT):
                            _trap()
                        stamp(13)

                    set_meta2((e_, rows), a_ready)
                    g2_task(
                        arg_disp,
                        arg_a2,
                        arg_w2,
                        arg_a2_scale,
                        arg_w2_scale,
                        arg_disp,
                        arg_disp,
                        arg_disp,
                        arg_rowmap,
                        rows_cap,
                        n2,
                        f32_swiglu_limit,
                        fx.Float32(1.0),
                        fx.Float32(1.0),
                        mt_idx,
                        n,
                    )
                if is_t0:
                    comm_ops.store_i32_lds(bc_next, fx.Int32(G) + claim)
                rocdl.s_wait_dscnt(0)
                rocdl.s_barrier_signal(-1)
                rocdl.s_barrier_wait(-1)
                u = _uniform(comm_ops.load_i32_lds(bc_next))
            stamp(4)

            # CTA 0's counter reset (its stores retired above) is one unit.
            n2_done = n2_done + fx.Int32(
                (bid == fx.Int32(0)).select(fx.Int32(1), fx.Int32(0))
            )
            if (wave == fx.Int32(0)) & (n2_done > fx.Int32(0)):
                close_units(n2_done)
            stamp(5)
            # ======================== 5. REDUCE ========================
            # CTA 0 alone polls the peers' comb_flag, then opens the gates.
            if (bid == fx.Int32(0)) & (wave == fx.Int32(0)):
                poll_peer_flags(CF_OFF)
                stamp(18)
                open_gates(GATE_C, GATE_STRIDE)
                stamp(19)
            if wave == fx.Int32(0):
                wait_gate(GATE_C, GATE_STRIDE)
            rocdl.s_wait_dscnt(0)
            rocdl.s_barrier_signal(-1)
            rocdl.s_barrier_wait(-1)
            stamp(6)
            rsrc_comb = create_buffer_resource_from_addr(peer_va(rank, off_comb_inp))
            rsrc_out = create_buffer_resource_from_addr(addr_out)
            for w in range(wave_gid, tok * fx.Int32(RED_CH if PH & 8 else 0), G * 4):
                t = w // fx.Int32(RED_CH)
                ch = w - t * fx.Int32(RED_CH)
                accs = [fx.full(8, 0.0, fx.Float32) for _ in range_constexpr(2)]
                for k in range_constexpr(K):
                    gid = fx.Int32(
                        buffer_load(
                            rsrc_ids,
                            t * fx.Int32(K) + fx.Int32(k),
                            vec_width=1,
                            dtype=T.i32,
                            is_scalar=True,
                        )
                    )
                    valid = (gid >= fx.Int32(0)) & (gid < fx.Int32(E))
                    slot_b = (t * fx.Int32(K) + fx.Int32(k)) * fx.Int32(
                        comb_slot_stride
                    )
                    for j in range_constexpr(2):
                        raw = buffer_load(
                            rsrc_comb,
                            (
                                slot_b
                                + ch * fx.Int32(RED_ELEMS * 2)
                                + lane * fx.Int32(32)
                            )
                            // fx.Int32(4)
                            + fx.Int32(4 * j),
                            vec_width=4,
                            dtype=T.i32,
                            cache_modifier=SCOPE_SYS,
                        )
                        v = fx.Vector(raw).bitcast(fx.BFloat16).to(fx.Float32)
                        v = valid.select(v, fx.full(8, 0.0, fx.Float32))
                        accs[j] = accs[j] + v
                o_dw = (
                    t * fx.Int32(H) + ch * fx.Int32(RED_ELEMS) + lane * fx.Int32(16)
                ) // fx.Int32(2)
                for j in range_constexpr(2):
                    buffer_store(
                        accs[j].to(fx.BFloat16).bitcast(fx.Int32),
                        rsrc_out,
                        o_dw + fx.Int32(4 * j),
                    )
            stamp(7)

        kernel(
            arg_a2,
            arg_disp,
            arg_disp_scale,
            arg_w1,
            arg_w1_scale,
            arg_a2_scale,
            arg_w2,
            arg_w2_scale,
            arg_rowmap,
            arena,
            addr_x,
            addr_ids,
            addr_wts,
            addr_out,
            addr_ctl,
            i32_tok,
            f32_swiglu_limit,
        ).launch(grid=(G, 1, 1), block=(block, 1, 1), stream=stream)

    launch.compile_hints = dict(launch_gemm_a8w4_tdm.compile_hints)
    return launch


def _popc_i32(x):
    return fx.Int32(
        _llvm.call_intrinsic(
            T.i32, "llvm.ctpop.i32", [arith.unwrap(fx.Int32(x))], [], []
        )
    )


# ----------------------------------------------------------------------------
# Host side
# ----------------------------------------------------------------------------

#: The decode GEMM tiles: the a4w4 CSV's decode row for this shape family
#: (t16x256x256 b3 / t16x512x128 b3), which also measured best inside the
#: megakernel at tpr 8 (t16x512x128 b2 GEMM1, the staged optimum, is 15%
#: slower here: its m-tiles finish later, and GEMM2 waits on them). The tile
#: does not change the bits (each output element's K order is the same for
#: every tile), so it is a pure speed choice.
DMK_TILE = dict(  # noqa: C408 - the builder's keyword names
    tile_m=16,
    tile_n=256,
    tile_k=256,
    num_buffers=3,
    tile_n2=512,
    tile_k2=128,
    num_buffers2=3,
)
#: Past this many tokens per rank the 32-row tile wins (tpr 128: 303 vs
#: 312 us/layer; tpr 64: 306 vs 301).
DMK_TILE_M32_TOKENS = 64
DMK_CTAS_PER_CU = 2
# Debug/tuning only: AITER_DMK_TILE="tm,tn,tk,nb,tn2,tk2,nb2" and
# AITER_DMK_CTAS_PER_CU override the above.
_DMK_TILE_ENV = os.environ.get("AITER_DMK_TILE")


def dmk_tile(max_tokens_per_rank: int) -> dict:
    """The decode tiles for an instance serving up to this many tokens."""
    if _DMK_TILE_ENV:
        v = [int(x) for x in _DMK_TILE_ENV.split(",")]
        return dict(
            zip(
                (
                    "tile_m",
                    "tile_n",
                    "tile_k",
                    "num_buffers",
                    "tile_n2",
                    "tile_k2",
                    "num_buffers2",
                ),
                v,
            )
        )
    tile = dict(DMK_TILE)
    if int(max_tokens_per_rank) > DMK_TILE_M32_TOKENS:
        tile["tile_m"] = 32
    return tile


if os.environ.get("AITER_DMK_CTAS_PER_CU"):
    DMK_CTAS_PER_CU = int(os.environ["AITER_DMK_CTAS_PER_CU"])


def read_decode_mega_tokens() -> int:
    """``AITER_MEGA_DECODE``: tokens per rank up to which a step runs the decode
    megakernel (0 or unset: off). A step qualifies when its receive bound
    (``recv_token_bound``, else ``world * max_tokens_per_rank``) is at most
    ``world`` times this, so every rank picks the same path."""
    v = os.environ.get("AITER_MEGA_DECODE", "")
    return int(v) if v.strip() else 0


def decode_mega_reason(config) -> str | None:
    """Why this MegaMoE cannot run the decode megakernel, or None if it can."""
    from aiter import ActivationType, QuantType

    from .recipe import CombineKind, DispatchKind

    bound = read_decode_mega_tokens()
    recipe, geo = config.recipe, config.geometry
    if bound <= 0:
        return "AITER_MEGA_DECODE is off"
    if recipe.dispatch is not DispatchKind.COMPACT or config.dispatch_wire != "fp4":
        return "needs the compact dispatch on the fp4 wire"
    if recipe.combine is not CombineKind.FUSED_SCATTER:
        return "needs the fused-scatter combine"
    comp = recipe.compute
    if comp.activation not in (ActivationType.Silu, ActivationType.Swiglu):
        return f"activation {comp.activation} not supported"
    if (
        comp.quant_type != QuantType.per_1x32
        or comp.hidden_pad
        or comp.intermediate_pad
    ):
        return "needs per_1x32 quant and no padding"
    H, I = config.hidden_dim, geo.inter_dim
    t = dmk_tile(dmk_small_tokens(config))
    if H % (32 * WAVE) or (2 * I) % t["tile_n"] or H % t["tile_n2"]:
        return f"hidden {H} / inter {I} do not tile"
    if config.world_size > WAVE or geo.experts % config.world_size:
        return "experts must split evenly over at most 32 ranks"
    return None


def dmk_small_tokens(config) -> int:
    """Tokens per rank the megakernel serves: ``AITER_MEGA_DECODE``, at most the
    instance's ``max_tokens_per_rank``."""
    return min(read_decode_mega_tokens(), int(config.geometry.max_tokens_per_rank))


def dmk_recv_cap(config) -> int:
    """The largest receive bound (tokens) a step may have to run the megakernel."""
    return config.world_size * dmk_small_tokens(config)


def dmk_cap(config) -> int:
    """Rows each local expert reserves. A token's top-k experts are distinct, so
    an expert gets at most one row per token its rank receives, and a served
    step receives at most ``dmk_recv_cap`` tokens."""
    tm = dmk_tile(dmk_small_tokens(config))["tile_m"]
    rows = dmk_recv_cap(config)
    return (rows + tm - 1) // tm * tm


def dmk_wire_row(config) -> int:
    h = config.hidden_dim
    return (h // 2 + h // 32 + 127) // 128 * 128


def decode_mega_regions(config) -> dict:
    """The arena regions the decode megakernel adds ({} when it is off)."""
    if decode_mega_reason(config) is not None:
        return {}
    epr, cap = config.experts_per_rank, dmk_cap(config)
    return {
        "dmk_disp": epr * cap * dmk_wire_row(config),
        "dmk_rowmap": epr * cap * 8,
        "dmk_ctl": dmk_ctl_arena_bytes(npes=config.world_size, experts_per_rank=epr),
    }


class DecodeMega:
    """The decode megakernel bound to one MegaMoE's arena and combine."""

    def __init__(self, config, res, device):
        import torch

        from aiter.jit.utils.chip_info import get_cu_num

        from ..tensor_shim import _run_compiled  # noqa: F401  (import check)

        self.config = config
        self.res = res
        geo = config.geometry
        arena = res.arena
        epr, cap = config.experts_per_rank, dmk_cap(config)
        self.cap = cap
        self.row = dmk_wire_row(config)
        t = self.tile = dmk_tile(dmk_small_tokens(config))
        self.recv_cap = dmk_recv_cap(config)
        from aiter.ops.flydsl.grouped_moe_gfx1250 import get_wmma_m_rep

        rep2 = get_wmma_m_rep(t["tile_m"], t["tile_n2"], 1, 4, "gemm2")
        rows = epr * cap
        inter = geo.inter_dim
        # Zeroed once: dmk_ctl's counters, parity words and generation flags.
        # Done before construction returns: the peers' first slot atomics may
        # land as soon as every rank is past its construction barrier.
        arena.zero("dmk_ctl")
        torch.cuda.synchronize(device)
        self.a2 = torch.empty((rows, inter // 2), dtype=torch.uint8, device=device)
        self.a2_scale = torch.empty(
            (rows // rep2, inter // 32 * rep2), dtype=torch.uint8, device=device
        )
        self.workers = int(get_cu_num()) * DMK_CTAS_PER_CU
        # Per-call switch for in-process A/B against the staged path. All
        # ranks must flip it together (the paths share no protocol state).
        self.enabled = True
        self._ctl_words = dmk_ctl_words(
            experts_per_rank=epr, cap=cap, tile_m=t["tile_m"], workers=self.workers
        )
        self.ctl = torch.zeros(
            self._ctl_words + 2 * TR_N * self.workers + 8 * RT_N,
            dtype=torch.int32,
            device=device,
        )
        self._traced = 0
        self._launch = None
        self._act_key = None

    def serves(self, recv_token_bound) -> bool:
        """Whether a step with this receive bound runs the megakernel: the
        same answer on every rank, since every rank passes the same bound."""
        if not self.enabled:
            return False
        bound = (
            self.config.max_recv if recv_token_bound is None else int(recv_token_bound)
        )
        return bound <= self.recv_cap

    def _launcher(self, stage1_act):
        if self._launch is None or self._act_key != stage1_act:
            config, arena = self.config, self.res.arena
            geo = config.geometry
            self._launch = make_decode_mega(
                rank=config.rank,
                npes=config.world_size,
                experts=geo.experts,
                experts_per_rank=config.experts_per_rank,
                topk=config.topk,
                model_dim=config.hidden_dim,
                inter_dim=geo.inter_dim,
                max_tok_per_rank=geo.max_tokens_per_rank,
                cap=self.cap,
                row_stride=self.row,
                off_disp=arena.offset("dmk_disp"),
                off_rowmap=arena.offset("dmk_rowmap"),
                off_ctl=arena.offset("dmk_ctl"),
                off_comb_inp=arena.offset("comb_inp"),
                comb_slot_stride=config.combine_slot_stride_for(0),
                arena_handle=arena.handle,
                stage1_act=stage1_act,
                workers=self.workers,
                ctas_per_cu=DMK_CTAS_PER_CU,
                **self.tile,
            )
            self._act_key = stage1_act
        return self._launch

    def forward(
        self, hidden_states, topk_weights, topk_ids, *, w1, w2, w1_scale, w2_scale
    ):
        import flydsl.compiler as flyc
        import flydsl.expr as fx
        import torch

        from aiter import ActivationType
        from aiter.ops.flydsl.grouped_moe_gfx1250 import _grouped_weight_uint8

        from ..tensor_shim import _run_compiled

        comp = self.config.recipe.compute
        swiglu = comp.activation == ActivationType.Swiglu
        stage1_act = 2 if swiglu else 1
        sl = (
            float(comp.swiglu_limit)
            if comp.swiglu_limit
            else (7.0 if swiglu else float("inf"))
        )
        tokens = int(hidden_states.shape[0])
        arena = self.res.arena
        out = self.res.combine_output

        def ptr(addr):
            # Every buffer here is at least 16 B aligned (torch, arena regions,
            # and the e8m0 plane one 3584 B payload into the wire row).
            return flyc.from_c_void_p(fx.Uint8, int(addr), assumed_align=16)

        tracing = (
            os.environ.get("AITER_DMK_TRACE", "0") != "0"
            and self._traced < 6
            and not torch.cuda.is_current_stream_capturing()
        )
        if tracing:
            self.ctl[self._ctl_words :].zero_()
        disp = arena.local_ptr("dmk_disp")
        w1_u8 = _grouped_weight_uint8(w1)
        w2_u8 = _grouped_weight_uint8(w2)
        if os.environ.get("AITER_DMK_DEBUG") and not getattr(self, "_dbg_done", 0):
            self._dbg_done = 1
            regs = {
                n: (arena.local_ptr(n), arena.size(n)) for n, _ in arena.layout.regions
            }
            print(
                f"[dmk r{self.config.rank}] regions="
                + " ".join(f"{n}:{a:#x}+{z:#x}" for n, (a, z) in regs.items())
                + f" a2={self.a2.data_ptr():#x}+{self.a2.nbytes:#x}"
                + f" a2s={self.a2_scale.data_ptr():#x}+{self.a2_scale.nbytes:#x}"
                + f" ctl={self.ctl.data_ptr():#x}+{self.ctl.nbytes:#x}"
                + f" w1={w1_u8.data_ptr():#x}+{w1_u8.nbytes:#x}"
                + f" w1s={w1_scale.data_ptr():#x}+{w1_scale.nbytes:#x}"
                + f" w2={w2_u8.data_ptr():#x}+{w2_u8.nbytes:#x}"
                + f" w2s={w2_scale.data_ptr():#x}+{w2_scale.nbytes:#x}"
                + f" x={hidden_states.data_ptr():#x} out={out.data_ptr():#x}+{out.nbytes:#x}"
                + f" cap={self.cap} row={self.row} G={self.workers}",
                flush=True,
            )
        _run_compiled(
            self._launcher(stage1_act),
            ptr(self.a2.data_ptr()),
            ptr(disp),
            ptr(disp + self.config.hidden_dim // 2),
            ptr(w1_u8.data_ptr()),
            ptr(w1_scale.data_ptr()),
            ptr(self.a2_scale.data_ptr()),
            ptr(w2_u8.data_ptr()),
            ptr(w2_scale.data_ptr()),
            ptr(arena.local_ptr("dmk_rowmap")),
            arena.handle,
            hidden_states.data_ptr(),
            topk_ids.data_ptr(),
            topk_weights.data_ptr(),
            out.data_ptr(),
            self.ctl.data_ptr(),
            tokens,
            sl,
            fx.Stream(torch.cuda.current_stream()),
        )
        if tracing:
            self._traced += 1
            self._print_trace(tokens)
        return self.res.output(tokens)

    def report(self):
        """AITER_DMK_TRACE: the last launch's phase stamps (graph replays too)."""
        if os.environ.get("AITER_DMK_TRACE", "0") != "0":
            self._print_trace(-1)

    def _print_trace(self, tokens):
        import torch

        torch.cuda.synchronize()
        tr = self.ctl[self._ctl_words : self._ctl_words + 2 * TR_N * self.workers]
        tr = tr.view(torch.int64).view(self.workers, TR_N)
        tr = tr.cpu().double()
        t0 = tr[:, 0].min()
        rel = (tr - t0) / 100.0  # steady counter: 100 MHz -> us
        names = [
            "start",
            "ticket",
            "produced",
            "planned",
            "gemm1",
            "gemm2",
            "flags",
            "end",
            "p_ready",
            "gate_seen",
            "p_issued",
            "p_retired",
            "g2_wait0",
            "g2_wait1",
        ]
        lead = ["src_seen", "plan_lds", "reps_done", "gates_p", "cf_seen", "gates_c"]
        cols = []
        for k, n in enumerate(names):
            c = rel[:, k]
            # Stamps only some CTAs take may be stale from an earlier launch.
            c = c[tr[:, k] >= t0] if k >= 8 else c
            if c.numel() == 0:
                continue
            cols.append(f"{n}={c.min():.1f}/{c.median():.1f}/{c.max():.1f}")
        rt = self.ctl[self._ctl_words + 2 * TR_N * self.workers :].view(torch.int64)
        rt = rt.view(RT_N, 4).cpu().double()
        ok = (rt[:, 0] >= t0) & (rt[:, 1] >= rt[:, 0])
        if ok.any():
            r_ = rt[ok]
            lat = (r_[:, 1] - r_[:, 0]) / 100.0
            iss = (r_[:, 0] - t0) / 100.0
            for d in sorted(set(r_[:, 3].long().tolist())):
                m = r_[:, 3] == d
                cols.append(
                    f"atom->r{d}:n={int(m.sum())},issue={iss[m].min():.1f}-{iss[m].max():.1f},"
                    f"lat={lat[m].min():.1f}/{lat[m].median():.1f}/{lat[m].max():.1f}"
                )
        cols.append(
            "cta0:" + ",".join(f"{n}={rel[0, 14 + k]:.1f}" for k, n in enumerate(lead))
        )
        # The CTAs that finish GEMM2 last: their last unit's wait.
        late = rel[:, 5].argsort(descending=True)[:3].tolist()
        cols.append(
            "late="
            + ",".join(
                f"b{b}:g1={rel[b, 4]:.0f},w0={rel[b, 12]:.0f},w1={rel[b, 13]:.0f},g2={rel[b, 5]:.0f}"
                for b in late
            )
        )
        print(
            f"[dmk trace r{self.config.rank} T={tokens}] (min/med/max us) "
            + " ".join(cols),
            flush=True,
        )

    def errors(self) -> int:
        """Routes dropped for a full expert slot since construction."""
        return int(self.ctl[CTL_ERR].item())


__all__ = [
    "DecodeMega",
    "decode_mega_reason",
    "decode_mega_regions",
    "dmk_ctl_arena_bytes",
    "dmk_ctl_words",
    "dmk_epr_pad",
    "make_decode_mega",
    "read_decode_mega_tokens",
]
