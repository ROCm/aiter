# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""make_megakernel: the build constants, the GEMM builders, the LDS and
the kernel body, which runs one front end (frontend_slots or
frontend_compact) and then the core."""

from __future__ import annotations

import os

import flydsl.compiler as flyc
import flydsl.expr as fx
import mori.cco.device.flydsl as cco
from flydsl.expr import arith, const_expr
from flydsl.expr.typing import T

from aiter.ops.flydsl.kernels.buffer_ops import (
    create_buffer_resource_from_addr,
)
from aiter.ops.flydsl.kernels.mxfp4_preshuffle_gfx1250_tdm import (
    build_gemm_a8w4_tdm_task,
    launch_gemm_a8w4_tdm,
)

from ..config import _WAVE_SIZE as WAVE
from ..s1f_stage1 import (
    SCOPE_DEV,
    SCOPE_SYS,
    _uniform,
)
from . import core, frontend_compact, frontend_slots
from .core import (
    FLAG_SPAN,
    FLAG_STRIDE,
    N_TICKET,
    NOW_HZ,
    POLL_SLEEP,
    RED_ELEMS,
    SEG_BITS,
    TALLY_CHUNK,
    TALLY_MAX,
    TALLY_MAXR,
    TIMEOUT_S,
    TR_N,
    KCtx,
    compact_scratch_bytes,
    ctl_arena_bytes,
    ctl_layout,
    epr_pad,
    segs_pad,
    src_off,
)


def make_megakernel(
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
    layout="slots",
    mt_cap=None,
):
    """Build a megakernel; returns a ``@flyc.jit`` launcher.

    ``layout="slots"``: expert ``e`` owns rows ``[e*cap, (e+1)*cap)``
    and a route reserves its slot with a remote atomic.
    ``layout="compact"``: the staged compact layout (expert-major, each expert padded to
    ``tile_m``, sources in rank order inside it), laid out from one tagged
    count exchange (X1); ``mt_cap`` m-tiles at most.

    Launcher arguments (positional): ``arg_a2``, ``arg_disp``,
    ``arg_disp_scale``, ``arg_w1``, ``arg_w1_scale``, ``arg_a2_scale``,
    ``arg_w2``, ``arg_w2_scale``, ``arg_rowmap`` (pointers), then i64
    ``arena`` (window handle), ``x``, ``ids``, ``wts``, ``out``, ``ctl``, then
    ``i32_tok`` (this rank's tokens), ``i32_rows`` (rows the GEMMs may see),
    ``addr_route`` (compact: the route slots, i32 per route),
    ``f32_swiglu_limit``, ``stream``.
    """
    COMPACT = layout == "compact"
    if layout not in ("slots", "compact"):
        raise ValueError(f"layout must be slots or compact, got {layout!r}")
    if WAVE != 32:
        raise ValueError("the megakernel requires gfx1250 wave32")
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
    if COMPACT and int(max_tok_per_rank) > 0xFFFF:
        raise ValueError("X1 holds a source's rows of an expert in 16 bits")
    if G % WAVE or G % N_TICKET:
        raise ValueError(f"workers={G} must be a multiple of {WAVE} and {N_TICKET}")
    if npes > WAVE or E != npes * EPR:
        raise ValueError("experts must split evenly over at most 32 ranks")
    EPR_PAD = epr_pad(EPR)
    EPR_CH = EPR_PAD // WAVE
    SRC_OFF = src_off(EPR)
    CF_OFF = SRC_OFF + FLAG_SPAN
    ST_OFF = SRC_OFF + 2 * FLAG_SPAN
    RS_OFF = ST_OFF + npes * FLAG_STRIDE
    MT_PER_E = CAP // TM
    MT_CAP = int(mt_cap) if COMPACT else EPR * MT_PER_E
    N1 = 2 * I
    NT1 = N1 // int(tile_n)
    NT2 = H // int(tile_n2)
    if N1 % int(tile_n) or H % int(tile_n2):
        raise ValueError("tile_n / tile_n2 must divide 2*inter_dim / model_dim")
    DEST_STRIDE = int(max_tok_per_rank) * K
    RED_CH = H // RED_ELEMS
    _lay = ctl_layout(
        experts_per_rank=EPR, mt_cap=MT_CAP, workers=G, npes=npes, compact=COMPACT
    )
    CTL_WORDS = _lay["words"]
    PLAN_W, PLAN_SLOT = _lay["plan"], _lay["plan_slot"]
    GATE_P, GATE_C, GATE_S = _lay["gate_p"], _lay["gate_c"], _lay["gate_s"]
    GATE_R = _lay["gate_r"]
    if not COMPACT and CAP > 0xFFFF:
        raise ValueError(
            "the slots layout's plan replicas hold an expert's rows in 16 bits"
        )
    # Plan replica dwords per lane: slots packs two 16-bit counts a dword,
    # compact stores one count a dword (an expert may receive past 65535 rows).
    REP_W = EPR_CH if COMPACT else max(1, EPR_CH // 2)
    SEGS_PAD = segs_pad(npes=npes, experts_per_rank=EPR)
    SEG_CNT_W = SREP_W = MAXR = T_MAX = T_CHUNK = None
    if COMPACT:
        SEG_CNT_W, SREP_W = _lay["seg_cnt"], _lay["srep"]
        if npes * SEGS_PAD * 4 > src_off(EPR) or SEGS_PAD > (1 << SEG_BITS):
            raise ValueError("the X1 rows do not fit the control region")
        MAXR = int(max_tok_per_rank) * K
        T_MAX = min(TALLY_MAX, G - 1)
        if MAXR > T_MAX * TALLY_MAXR:
            raise ValueError("more routes per rank than the tally CTAs count")
        T_CHUNK = int(os.environ.get("AITER_LM_TALLY_CHUNK", TALLY_CHUNK))
        if not 1 <= T_CHUNK <= TALLY_MAXR:
            raise ValueError(f"AITER_LM_TALLY_CHUNK must be in 1..{TALLY_MAXR}")
    TR_BASE = CTL_WORDS  # i64 stamps [G][TR_N] after the control words
    RT_BASE = TR_BASE + 2 * G * TR_N  # i64 [RT_N][4] per-route stamps
    CTL_ARENA_B = ctl_arena_bytes(npes=npes, experts_per_rank=EPR)

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
    # Debug only: GEMM1's A / A-scale loads (the dispatched rows).
    POL_A1 = int(os.environ.get("AITER_DMK_POL_A1", "0"))
    POL_C = int(os.environ.get("AITER_DMK_POL_C", str(SCOPE_DEV)))
    POL_S = int(os.environ.get("AITER_DMK_POL_S", str(SCOPE_SYS)))
    # Debug only: AITER_DMK_NOTRAP=1 has a timed-out wait count itself in
    # ctl[CTL_FAIL + site] and go on instead of trapping (with a short
    # AITER_DMK_TIMEOUT_S, a hung protocol then shows where it stops).
    NOTRAP = int(os.environ.get("AITER_DMK_NOTRAP", "0"))
    # Debug only: AITER_DMK_INV=1 has wave 0 of every CTA invalidate the caches
    # at system scope at launch (a stale cross-launch L2 line test).
    INV = int(os.environ.get("AITER_DMK_INV", "0"))
    # Debug only: every wave idles this many us after its ticket (lets every
    # rank's launch start before anyone sends).
    DELAY_US = int(os.environ.get("AITER_DMK_DELAY_US", "0"))
    # The slots layout's start handshake is off by default; AITER_DMK_NO_HS=0 restores it.
    HS = not int(os.environ.get("AITER_DMK_NO_HS", "1"))
    TIMEOUT_TICKS = int(
        float(os.environ.get("AITER_DMK_TIMEOUT_S", TIMEOUT_S)) * NOW_HZ
    )
    POLL_SL = int(os.environ.get("AITER_DMK_POLL_SLEEP", POLL_SLEEP))
    # AITER_DMK_PLAN_TAG: each plan replica dword is stored with the launch's
    # generation in the same 64-bit store, so a CTA polls its replica instead
    # of a gate CTA 0 opens after the replicas. On by default for 16-row tiles
    # (V4.1-Flash DEP4: tpr 8 -2.8, tpr 32 -1.5 us/layer; +1 to +4 past 128).
    PLAN_TAG = (
        0
        if COMPACT
        else int(os.environ.get("AITER_DMK_PLAN_TAG", "1" if TM <= 16 else "0"))
    )
    if REP_W * WAVE * (2 if PLAN_TAG else 1) > PLAN_SLOT:
        raise ValueError("a plan replica does not fit its slot")
    # AITER_DMK_DEFER_SCATTER (default 1): a GEMM2 item does not wait for its
    # EP-scatter stores; they retire under the next item, and the queue's end
    # drains them before the close counts the CTA's units. Its separate C stage
    # does not fit next to 64-row tiles at 2 CTAs per CU.
    DEFER_SC = 0 if COMPACT else int(os.environ.get("AITER_DMK_DEFER_SCATTER", "1"))
    if TM >= 64:
        DEFER_SC = 0
    # AITER_DMK_EARLY_PLAN (default 1): a source signals "reserved" once its
    # slot atomics have returned, before its rows retire. CTA 0 plans from the
    # counters as soon as every source reserved; GEMM items start and stream
    # weights while their A owners wait for the rows-landed gate (every
    # source's src_done). V4.1-Flash DEP4: -1.8 to -3.7 us/layer.
    EARLY = 0 if COMPACT else int(os.environ.get("AITER_DMK_EARLY_PLAN", "1"))
    # AITER_DMK_PREFETCH_KB (default 256, 0 = off): while the plan is pending,
    # a step of at least PREF_MIN_TOK tokens has every CTA but 0 read the first
    # N KB of the W1 slice its first item most likely needs, so that item finds
    # it in cache. V4.1-Flash DEP4: -1.4 (tpr 32) to -4.9 (tpr 128) us/layer;
    # below 32 tokens most guessed experts get no rows and it costs.
    PREF_B = (
        0 if COMPACT else int(os.environ.get("AITER_DMK_PREFETCH_KB", "256")) * 1024
    )
    PREF_B = min(PREF_B, int(tile_n) * (H // 2))
    # Every CTA of the grid must be resident: a 4-wave CTA puts one wave on
    # each SIMD, so ctas_per_cu CTAs need that many waves per EU, which caps
    # the VGPRs (compact at 4 CTAs per CU: 128). AITER_DMK_WPE=0 drops the cap.
    WPE = int(os.environ.get("AITER_DMK_WPE", str(int(ctas_per_cu))))
    # AITER_DMK_RED_UNROLL: (token, chunk) items a reduce wave loads at once.
    # Its passes wait out their loads; compact's waves own up to ~5 items each
    # (V4.1-Flash DEP4, 2048 tokens/rank: the reduce 19 -> 14 us at 2).
    RED_UN_DEF = 2 if COMPACT else 1
    RED_UN = int(os.environ.get("AITER_DMK_RED_UNROLL", str(RED_UN_DEF)))
    knob_sfx = f"_wpe{WPE}" if WPE else ""
    knob_sfx += "".join(
        f"_{name}{val}"
        for name, val, default in (
            ("tc", T_CHUNK if COMPACT else TALLY_CHUNK, TALLY_CHUNK),
            ("ru", RED_UN, RED_UN_DEF),
            ("ps", POLL_SL, POLL_SLEEP),
            ("hs", int(HS and not COMPACT), 0),
            ("pt", PLAN_TAG, 1 if TM <= 16 else 0),
            ("ds", DEFER_SC, 1),
            ("ep", EARLY, 1),
            ("pf", PREF_B // 1024, min(256 * 1024, int(tile_n) * (H // 2)) // 1024),
        )
        if val != default
    )

    # The kernel's build constants, for the core and the front ends.
    static = KCtx(
        CAP=CAP,
        CF_OFF=CF_OFF,
        COMPACT=COMPACT,
        DEFER_SC=DEFER_SC,
        DELAY_US=DELAY_US,
        DEST_STRIDE=DEST_STRIDE,
        E=E,
        EARLY=EARLY,
        EPR=EPR,
        EPR_CH=EPR_CH,
        EPR_PAD=EPR_PAD,
        G=G,
        GATE_C=GATE_C,
        GATE_P=GATE_P,
        GATE_R=GATE_R,
        GATE_S=GATE_S,
        H=H,
        HS=HS,
        INV=INV,
        K=K,
        MAXR=MAXR,
        MT_PER_E=MT_PER_E,
        N1=N1,
        NG=NG,
        NOTRAP=NOTRAP,
        NT1=NT1,
        NT2=NT2,
        PAYLOAD=PAYLOAD,
        PH=PH,
        PLAN_SLOT=PLAN_SLOT,
        PLAN_TAG=PLAN_TAG,
        PLAN_W=PLAN_W,
        POLL_SL=POLL_SL,
        PREF_B=PREF_B,
        RED_CH=RED_CH,
        RED_UN=RED_UN,
        REP_W=REP_W,
        ROW=ROW,
        RS_OFF=RS_OFF,
        RT_BASE=RT_BASE,
        SEGS_PAD=SEGS_PAD,
        SEG_CNT_W=SEG_CNT_W,
        SRC_OFF=SRC_OFF,
        SREP_W=SREP_W,
        ST_OFF=ST_OFF,
        TIMEOUT_TICKS=TIMEOUT_TICKS,
        TM=TM,
        TRACE=TRACE,
        TR_BASE=TR_BASE,
        T_CHUNK=T_CHUNK,
        T_MAX=T_MAX,
        comb_slot_stride=comb_slot_stride,
        npes=npes,
        off_comb_inp=off_comb_inp,
        off_ctl=off_ctl,
        off_disp=off_disp,
        off_rowmap=off_rowmap,
        rank=rank,
        tile_n=tile_n,
    )

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
        i32_rows: fx.Int32,
        addr_route: fx.Int64,
        f32_swiglu_limit: fx.Float32,
        stream: fx.Stream,
    ):
        lds_holder = [None]
        meta1_holder = [None]
        ready1_holder = [None]
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
            a_ready=ready1_holder if (COMPACT or EARLY) else None,
            a_cache_modifier=POL_A1,
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
            defer_ep_wait=DEFER_SC,
            **common,
        )
        if block != 4 * WAVE or block2 != block:
            raise ValueError(f"expected 4-wave GEMM blocks, got {block}/{block2}")
        ARENA = max(int(lay1["arena"]), int(lay2["arena"]))
        # LDS beyond the GEMM arena: broadcast words, then per expert the
        # plan's inclusive m-tile end, CTA 0's replica counts and the plan's
        # row count, a dword each.
        BC_B = 64
        PLAN_B = 3 * EPR_PAD * 4
        CTRL_B = BC_B + PLAN_B
        if COMPACT and compact_scratch_bytes(npes=npes, experts_per_rank=EPR) > ARENA:
            raise ValueError("the compact front end's LDS does not fit the GEMM arena")
        if int(ctas_per_cu) * (((ARENA + CTRL_B) + 2047) // 2048 * 2048) > 327680:
            raise ValueError(
                f"{ctas_per_cu} CTAs of {ARENA + CTRL_B} B LDS do not fit one CU"
            )
        kname = (
            ("mega_compact" if COMPACT else "mega_slots")
            + f"_w{G}_r{rank}of{npes}_c{CAP if not COMPACT else MT_CAP * TM}"
            + (f"_ph{PH}" if PH != 15 else "")
            + knob_sfx
            + f"__{g1_name}__{g2_name}"
        )
        # Keep the separate-kernel launcher in this cache key's dependency set.
        _ = launch_gemm_a8w4_tdm

        lctx = KCtx(
            static,
            g1_task=g1_task,
            g2_task=g2_task,
            meta1_holder=meta1_holder,
            ready1_holder=ready1_holder,
            meta2_holder=meta2_holder,
            ready2_holder=ready2_holder,
        )

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
            i32_rows: fx.Int32,
            addr_route: fx.Int64,
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
            lds_rows = lds_cnt + fx.Int32(EPR_PAD * 4)
            # Compact front-end scratch in the GEMM arena (CTA 0's X1 matrix,
            # send table and receive counts), unused once GEMM items start.
            ar = fx.Int32(
                arith.index_cast(T.i32, fx.index_cast(T.index, fx.ptrtoint(arena_ptr)))
            )
            l_m = ar
            l_sb = ar + fx.Int32(npes * SEGS_PAD * 4)
            l_rc = l_sb + fx.Int32(SEGS_PAD * 4)

            win = cco.CachedWindow(arena)
            win_base = win.base
            win_stride = win.stride
            kc = KCtx(
                lctx,
                tid=tid,
                lane=lane,
                wave=wave,
                bid=bid,
                is_t0=is_t0,
                bc=bc,
                lds_end=lds_end,
                lds_cnt=lds_cnt,
                lds_rows=lds_rows,
                ar=ar,
                l_m=l_m,
                l_sb=l_sb,
                l_rc=l_rc,
                win_base=win_base,
                win_stride=win_stride,
                arg_a2=arg_a2,
                arg_disp=arg_disp,
                arg_disp_scale=arg_disp_scale,
                arg_w1=arg_w1,
                arg_w1_scale=arg_w1_scale,
                arg_a2_scale=arg_a2_scale,
                arg_w2=arg_w2,
                arg_w2_scale=arg_w2_scale,
                arg_rowmap=arg_rowmap,
                arena=arena,
                addr_x=addr_x,
                addr_ids=addr_ids,
                addr_wts=addr_wts,
                addr_out=addr_out,
                addr_ctl=addr_ctl,
                i32_tok=i32_tok,
                i32_rows=i32_rows,
                addr_route=addr_route,
                f32_swiglu_limit=f32_swiglu_limit,
            )
            core.bind_base(kc)
            peer_va, stamp = kc.peer_va, kc.stamp

            my_ctl = peer_va(rank, off_ctl)
            rsrc_ctl = create_buffer_resource_from_addr(
                addr_ctl, num_records_bytes=4 * CTL_WORDS
            )
            rsrc_actl = create_buffer_resource_from_addr(
                my_ctl, num_records_bytes=CTL_ARENA_B
            )
            tok = fx.Int32(i32_tok)
            kc.rsrc_ctl, kc.rsrc_actl, kc.tok = rsrc_ctl, rsrc_actl, tok
            kc.gen, kc.par, kc.tag16 = core.start(kc)
            core.bind_protocol(kc)
            if const_expr(not COMPACT and HS):
                frontend_slots.start_handshake(kc)

            # ======================== 1. PRODUCE ========================
            rsrc_x = create_buffer_resource_from_addr(addr_x)
            rsrc_ids = create_buffer_resource_from_addr(addr_ids)
            rsrc_wts = create_buffer_resource_from_addr(addr_wts)
            wave_gid = bid * fx.Int32(4) + wave
            n_mine = fx.Int32(0)
            n_routes = tok * fx.Int32(K)
            kc.rsrc_x, kc.rsrc_ids, kc.rsrc_wts = rsrc_x, rsrc_ids, rsrc_wts
            kc.wave_gid = wave_gid
            if const_expr(COMPACT):
                n_mine = frontend_compact.produce(kc, n_mine, n_routes)
            else:
                n_mine, n_routes = frontend_slots.produce(kc, n_mine)
            stamp(10)
            if const_expr(EARLY):
                frontend_slots.signal_reserved(kc, n_mine, n_routes)
            core.publish_produced(kc, n_mine, n_routes)
            stamp(2)
            if const_expr(PREF_B):
                frontend_slots.prefetch_w1(kc)
            n_mt = core.plan(kc)
            n2_done, total_units = core.gemm_queue(kc, n_mt)
            core.close(kc, n2_done, total_units)
            core.reduce(kc)

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
            i32_rows,
            addr_route,
            f32_swiglu_limit,
            value_attrs={"rocdl.waves_per_eu": str(WPE)} if WPE else {},
        ).launch(grid=(G, 1, 1), block=(block, 1, 1), stream=stream)

    launch.compile_hints = dict(launch_gemm_a8w4_tdm.compile_hints)
    return launch
