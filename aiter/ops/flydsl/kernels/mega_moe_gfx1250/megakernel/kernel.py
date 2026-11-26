# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""make_megakernel: the build constants, the GEMM builders, the LDS and the
kernel body, which runs one layout's dispatch (dispatch_slots or
dispatch_compact) and then the shared protocol."""

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
from . import dispatch_compact, dispatch_slots, protocol
from .protocol import (
    FLAG_SPAN,
    N_TICKET,
    NOTRAP_TIMEOUT_S,
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
    stream=False,
    producers=64,
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
    ``addr_route`` (compact: the route slots, i32 per route; STREAM: its
    scratch, config.stream_layout), ``f32_swiglu_limit``, ``stream``.

    ``stream`` (compact only): the dispatch is S1F's producer pipeline
    (config.STREAM).
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
    RS_OFF = SRC_OFF + 2 * FLAG_SPAN
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
    SEG_CNT_W = SREP_W = MAXR = T_MAX = None
    if COMPACT:
        SEG_CNT_W, SREP_W = _lay["seg_cnt"], _lay["srep"]
        if npes * SEGS_PAD * 4 > src_off(EPR) or SEGS_PAD > (1 << SEG_BITS):
            raise ValueError("the X1 rows do not fit the control region")
        MAXR = int(max_tok_per_rank) * K
        T_MAX = min(TALLY_MAX, G - 1)
        if MAXR > T_MAX * TALLY_MAXR:
            raise ValueError("more routes per rank than the tally CTAs count")
    STREAM = bool(stream) and COMPACT
    # The accepted DSR1 EP4 prefill profile writes activation/scale directly
    # in APRE layout and consumes it without an intervening reshuffle.
    APRE = bool(
        STREAM and TM == 128 and npes == 4
        and H == 7168 and I == 2048 and EPR == 65 and K == 9
        and int(max_tok_per_rank) == 8192
    )
    st_kw = {}
    if STREAM:
        from ..compact_plan import S1F_SHARED_FIRST
        from ..s1f_stage1 import s1f_block_runs, s1f_flag_m, s1f_producer_lds
        from .config import (
            STREAM_RUN_ROWS,
            stream_flag_rel,
            stream_layout,
        )

        FM = s1f_flag_m(TM)
        SUB = TM // FM
        P = int(producers)
        if not 1 <= P < G - 1:
            raise ValueError(f"producers={P} must leave consumers of workers={G}")
        TILE_CAP = (CAP + FM - 1) // FM
        lay_s = stream_layout(
            npes=npes,
            experts_per_rank=EPR,
            max_tok_per_rank=max_tok_per_rank,
            row=ROW,
            tile_cap=TILE_CAP,
        )
        mb = s1f_producer_lds(run_rows=STREAM_RUN_ROWS, row_stride=ROW)
        TBL_OFF = (mb["end"] + 127) // 128 * 128
        NPOS2 = lay_s["npos"]
        RBK = int(
            os.environ.get(
                "AITER_MEGAKERNEL_BLOCK_RUNS", str(s1f_block_runs(max_tok_per_rank))
            )
        )
        # GEMM1 claimed per XCC (protocol.gemm_queue): faster with two
        # n-tiles an XCC (DSR1 4096 tpr -12 us); three (Pro) read a little
        # slower, so they keep the one queue unless asked.
        NXCC = 8
        _xq = os.environ.get("AITER_MEGAKERNEL_XQ", "1" if NT1 == 2 * NXCC else "0")
        XQ = int(_xq == "1" and NT1 % NXCC == 0)
        # Nine interleaved full-layer runs accepted this profile. Preserve
        # producer progress and G1 XCC ownership; consume only ready M tiles.
        mix_supported = bool(
            APRE and G == 512 and XQ
            and int(tile_k2) == 128 and int(num_buffers2) == 3
        )
        mix_slots = int(os.environ.get(
            "AITER_MEGAKERNEL_READY_MIX", "4" if mix_supported else "0"
        ))
        mix_burst = int(os.environ.get(
            "AITER_MEGAKERNEL_READY_BURST", "2" if mix_supported else "4"
        ))
        if not mix_supported:
            mix_slots = 0
        if not 0 <= mix_slots <= 4 or not 1 <= mix_burst <= 8:
            raise ValueError("ready mix needs 0..4 slots and a burst of 1..8")
        st_kw = dict(  # noqa: C408 - KCtx fields
            READY_MIX=mix_slots,
            READY_BURST=mix_burst,
            XQ=XQ,
            NXCC=NXCC,
            GMX=int(os.environ.get("AITER_MEGAKERNEL_GMX", "32")),
            GM1=int(os.environ.get("AITER_MEGAKERNEL_GM1", "32")),
            GM2=int(os.environ.get(
                "AITER_MEGAKERNEL_GM2", "64" if mix_supported and mix_slots else "32"
            )),
            FM=FM,
            FSH=FM.bit_length() - 1,
            SUB=SUB,
            P=P,
            RBK=RBK,
            R_RUN=STREAM_RUN_ROWS,
            TILE_CAP=TILE_CAP,
            FLAG_OFF=off_ctl + stream_flag_rel(npes=npes, experts_per_rank=EPR),
            SEGCAP=int(max_tok_per_rank),
            SEND_N=lay_s["send_n"],
            NPOS=npes * EPR,
            NPOS2=NPOS2,
            SF=S1F_SHARED_FIRST if S1F_SHARED_FIRST < EPR else 0,
            TBL_OFF=TBL_OFF,
            WIRE_B=int(max_tok_per_rank) * ROW,
            ST_LAY=lay_s,
        )
        PROD_LDS = TBL_OFF + (4 * NPOS2 + 32) * 4
    TR_BASE = CTL_WORDS  # i64 stamps [G][TR_N] after the control words
    RT_BASE = TR_BASE + 2 * G * TR_N  # i64 [RT_N][4] per-route stamps
    CTL_ARENA_B = ctl_arena_bytes(npes=npes, experts_per_rank=EPR)

    from aiter.ops.flydsl.grouped_moe_gfx1250 import get_wmma_m_rep

    rep2 = get_wmma_m_rep(TM, int(tile_n2), 1, 4, "gemm2")
    # Debug: AITER_DMK_TRACE=1 stamps each CTA's phase boundaries into the
    # trace words after the control block.
    TRACE = int(os.environ.get("AITER_DMK_TRACE", "0"))
    # Debug: AITER_DMK_NOTRAP=1 counts a timed-out wait in ctl[CTL_FAIL + site]
    # and goes on instead of trapping.
    NOTRAP = int(os.environ.get("AITER_DMK_NOTRAP", "0"))
    TIMEOUT_TICKS = int((NOTRAP_TIMEOUT_S if NOTRAP else TIMEOUT_S) * NOW_HZ)
    # Plan replica dwords carry the launch's generation in the same 64-bit
    # store, so a CTA polls its replica instead of a gate CTA 0 opens after
    # them; faster for 16-row tiles only.
    PLAN_TAG = int(not COMPACT and TM <= 16)
    if REP_W * WAVE * (2 if PLAN_TAG else 1) > PLAN_SLOT:
        raise ValueError("a plan replica does not fit its slot")
    # slots: a GEMM2 item does not wait for its EP-scatter stores; they retire
    # under the next item, and the queue's end drains them before the close.
    # Its separate C stage does not fit next to 64-row tiles at 2 CTAs per CU.
    # Three interleaved accepted rounds retain scatter across consecutive G2
    # items. The protocol drains every wave before G1 can reuse their LDS.
    stream_defer_supported = bool(
        APRE and G == 512 and not st_kw.get("LOC", 0)
        and int(tile_n) == int(tile_n2) == 256
        and int(tile_k) == 256 and int(num_buffers) == 3
        and int(tile_k2) == 128 and int(num_buffers2) == 3
    )
    stream_defer = bool(
        stream_defer_supported
        and os.environ.get("AITER_MEGAKERNEL_STREAM_DEFER_SC", "1") == "1"
    )
    DEFER_SC = int((not COMPACT and TM < 64) or stream_defer)
    # Experimental direct LDS->peer stores keep the existing native STREAM
    # geometry and retirement protocol, but move scatter to ASYNCcnt.
    scatter_async_env = os.environ.get("AITER_MEGAKERNEL_SCATTER_ASYNC", "0")
    if scatter_async_env not in ("0", "1"):
        raise ValueError("AITER_MEGAKERNEL_SCATTER_ASYNC must be 0 or 1")
    SCATTER_ASYNC = int(stream_defer_supported and scatter_async_env == "1")
    # Three interleaved full-layer rounds accepted packed APRE output
    # scales. Keep the explicit builder opt-in within the verified profile.
    scale_pack = (int(os.environ.get("AITER_NATIVE_SCALE_OUTPUT_PACK", "2"))
                  if stream_defer_supported and int(stage1_act) == 1 else 0)
    if scale_pack not in (0, 1, 2):
        raise ValueError("native scale output packing needs mode 0, 1, or 2")
    # slots: a source signals "reserved" once its slot atomics returned, before
    # its rows retire; CTA 0 plans from the counters then, and GEMM items start
    # and stream weights while their A owners wait for the rows-landed gate.
    EARLY = int(not COMPACT)
    # slots: while the plan is pending, a step of at least PREF_MIN_TOK tokens
    # has every CTA but 0 read the first 256 KB of the W1 slice its first item
    # most likely needs.
    PREF_B = 0 if COMPACT else min(256 * 1024, int(tile_n) * (H // 2))
    # Every CTA of the grid must be resident: a 4-wave CTA puts one wave on
    # each SIMD, so ctas_per_cu CTAs need that many waves per EU, which caps
    # the VGPRs.
    WPE = int(ctas_per_cu)
    # (token, chunk) items a reduce wave loads at once; compact's waves own
    # several items each.
    RED_UN = 2 if COMPACT else 1
    # Three interleaved eight-layer rounds retained adjacent-chunk ID reuse
    # within the accepted native APRE profile, with k-order sums unchanged.
    RED_ROW2 = int(
        stream_defer_supported and DEFER_SC and scale_pack == 2
        and os.environ.get("AITER_NATIVE_REDUCE_ROW2", "1") == "1"
    )
    # The reduce reads comb_inp through one resource and i32 offsets while the
    # region fits them, else through a 64-bit base per token.
    RED_B64 = int(int(max_tok_per_rank) * K * int(comb_slot_stride) >= 1 << 31)
    knob_sfx = f"_wpe{WPE}" + "".join(
        f"_{name}{val}"
        for name, val, default in (
            ("st", (st_kw["P"] * 100 + st_kw["RBK"]) if STREAM else 0, 0),
            ("gm", (st_kw["GM1"] * 100 + st_kw["GM2"]) if STREAM else 3232, 3232),
            ("xq", st_kw["XQ"] * st_kw["GMX"] if STREAM else 32, 32),
            ("rr", RED_ROW2, 0),
            ("as", SCATTER_ASYNC, 0),
            ("ds", DEFER_SC, 1),
            ("ep", EARLY, 1),
            ("pf", PREF_B // 1024, min(256 * 1024, int(tile_n) * (H // 2)) // 1024),
        )
        if val != default
    )

    # The kernel's build constants, for the protocol and the dispatch phases.
    static = KCtx(
        CAP=CAP,
        CF_OFF=CF_OFF,
        COMPACT=COMPACT,
        SCATTER_ASYNC=SCATTER_ASYNC,
        DEFER_SC=DEFER_SC,
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
        K=K,
        MAXR=MAXR,
        MT_PER_E=MT_PER_E,
        N1=N1,
        NG=NG,
        NOTRAP=NOTRAP,
        NT1=NT1,
        NT2=NT2,
        PAYLOAD=PAYLOAD,
        PLAN_SLOT=PLAN_SLOT,
        PLAN_TAG=PLAN_TAG,
        PLAN_W=PLAN_W,
        POLL_SL=POLL_SLEEP,
        PREF_B=PREF_B,
        RED_CH=RED_CH,
        RED_UN=RED_UN,
        RED_ROW2=RED_ROW2,
        RED_B64=RED_B64,
        REP_W=REP_W,
        ROW=ROW,
        RS_OFF=RS_OFF,
        RT_BASE=RT_BASE,
        SEGS_PAD=SEGS_PAD,
        SEG_CNT_W=SEG_CNT_W,
        SRC_OFF=SRC_OFF,
        SREP_W=SREP_W,
        STREAM=STREAM,
        TIMEOUT_TICKS=TIMEOUT_TICKS,
        TM=TM,
        TRACE=TRACE,
        TR_BASE=TR_BASE,
        T_CHUNK=TALLY_CHUNK,
        T_MAX=T_MAX,
        comb_slot_stride=comb_slot_stride,
        npes=npes,
        off_comb_inp=off_comb_inp,
        off_ctl=off_ctl,
        off_disp=off_disp,
        off_rowmap=off_rowmap,
        rank=rank,
        tile_n=tile_n,
        **st_kw,
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
            m_align=0,
            explicit_tile=1,
            # Independent three-round acceptance retained the TDM address
            # preparation for this native APRE profile. Other layouts opt out.
            tdm_native_coordinates=int(APRE),
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
            a_cache_modifier=0,
            c_cache_modifier=SCOPE_DEV,
            a_preshuffled=0,
            out_preshuffled=int(APRE),
            tdm_native_scale_output_pack=scale_pack,
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
            # A at the default policy: only the m-tile's own GEMM2 units read
            # its a2 rows, after g1_ready, and the launch started with no line
            # of them cached.
            a_cache_modifier=0,
            scatter_cache_policy=SCOPE_SYS,
            a_preshuffled=int(APRE),
            ep_scatter_async=SCATTER_ASYNC,
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
        # STREAM packs the row counts into the m-tile ends and keeps CTA 0's
        # replica counts in its dispatch scratch.
        PLAN_B = (1 if STREAM else 3) * EPR_PAD * 4
        CTRL_B = BC_B + PLAN_B
        if COMPACT and compact_scratch_bytes(npes=npes, experts_per_rank=EPR) > ARENA:
            raise ValueError("compact's dispatch LDS does not fit the GEMM arena")
        # STREAM: a producer's ring, mailboxes and segment table share the
        # GEMM arena, which grows to hold them.
        ARENA_LDS = max(ARENA, PROD_LDS) if STREAM else ARENA
        if int(ctas_per_cu) * (((ARENA_LDS + CTRL_B) + 2047) // 2048 * 2048) > 327680:
            raise ValueError(
                f"{ctas_per_cu} CTAs of {ARENA_LDS + CTRL_B} B LDS do not fit one CU"
            )
        kname = (
            ("mega_compact" if COMPACT else "mega_slots")
            + f"_w{G}_r{rank}of{npes}_c{CAP if not COMPACT else MT_CAP * TM}"
            + knob_sfx
            + (
                f"_ready{st_kw['READY_MIX']}b{st_kw['READY_BURST']}"
                if STREAM and st_kw["READY_MIX"] else ""
            )
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
            arena_ptr = smem.allocate(ARENA_LDS, 128)._ptr
            lds_holder[0] = arena_ptr
            ctrl_ptr = smem.allocate(CTRL_B, 16)._ptr
            bc = fx.Int32(
                arith.index_cast(T.i32, fx.index_cast(T.index, fx.ptrtoint(ctrl_ptr)))
            )
            lds_end = bc + fx.Int32(BC_B)
            lds_cnt = lds_end + fx.Int32(EPR_PAD * 4)
            lds_rows = lds_cnt + fx.Int32(EPR_PAD * 4)
            # compact's dispatch scratch in the GEMM arena (CTA 0's X1 matrix,
            # send table and receive counts), unused once GEMM items start.
            ar = fx.Int32(
                arith.index_cast(T.i32, fx.index_cast(T.index, fx.ptrtoint(arena_ptr)))
            )
            l_m = ar
            l_sb = ar + fx.Int32(npes * SEGS_PAD * 4)
            l_rc = l_sb + fx.Int32(SEGS_PAD * 4)
            l_st = l_rc + fx.Int32(EPR_PAD * 4)
            l_scan = l_st + fx.Int32(EPR_PAD * 4)
            if const_expr(STREAM):
                lds_cnt = l_scan + fx.Int32(64)

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
                l_st=l_st,
                l_scan=l_scan,
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
            protocol.bind_base(kc)
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
            kc.gen, kc.par, kc.tag16 = protocol.start(kc)
            protocol.bind_protocol(kc)

            # ======================== 1. DISPATCH ========================
            rsrc_x = create_buffer_resource_from_addr(addr_x)
            rsrc_ids = create_buffer_resource_from_addr(addr_ids)
            rsrc_wts = create_buffer_resource_from_addr(addr_wts)
            wave_gid = bid * fx.Int32(4) + wave
            n_mine = fx.Int32(0)
            n_routes = tok * fx.Int32(K)
            kc.rsrc_x, kc.rsrc_ids, kc.rsrc_wts = rsrc_x, rsrc_ids, rsrc_wts
            kc.wave_gid = wave_gid
            if const_expr(STREAM):
                lay_s = st_kw["ST_LAY"]
                kc.addr_send = addr_route + fx.Int64(lay_s["send"])
                kc.addr_segt = addr_route + fx.Int64(lay_s["segt"])
                kc.addr_tsrc = addr_route + fx.Int64(lay_s["tsrc"])
                kc.addr_wire = addr_route + fx.Int64(lay_s["wire"])
                kc.is_prod = (bid >= fx.Int32(1)) & (bid <= fx.Int32(st_kw["P"]))
            if const_expr(COMPACT):
                n_mine = dispatch_compact.produce(kc, n_mine, n_routes)
            else:
                n_mine, n_routes = dispatch_slots.produce(kc, n_mine)
            stamp(10)
            if const_expr(EARLY):
                dispatch_slots.signal_reserved(kc, n_mine, n_routes)
            if const_expr(not STREAM):
                protocol.publish_produced(kc, n_mine, n_routes)
            stamp(2)
            if const_expr(PREF_B):
                dispatch_slots.prefetch_w1(kc)
            n_mt = protocol.plan(kc)
            n2_done, total_units = protocol.gemm_queue(kc, n_mt)
            protocol.close(kc, n2_done, total_units)
            protocol.reduce(kc)

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
