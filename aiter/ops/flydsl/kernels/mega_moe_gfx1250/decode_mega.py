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
   pair then go straight into the slot with SCOPE_SYS stores. Once every slot
   atomic of this rank has returned, the wave completing that count stores the
   generation into every destination's ``reserved[rank]`` (its counters from
   this rank are final); once every row has retired, the wave completing that
   count stores it into ``src_done[rank]``. A step of at least
   ``PREF_MIN_TOK`` tokens then has every CTA but 0 read the W1 slice its
   first item most likely needs into cache (HBM is idle until the plan).
2. PLAN. CTA 0 alone polls the peers' ``reserved`` (hundreds of CTAs polling
   one uncached line starve every access queued behind it), reads the expert
   counts and publishes them in ``PLAN_REPS`` replicas, each dword tagged with
   the generation in one 64-bit store (16-row tiles) or followed by every
   CTA's own gate line. Each CTA lays the m-tiles (expert-major, ``tile_m``
   rows) out in LDS from its replica. CTA 0 then polls the peers'
   ``src_done`` and opens the rows-landed gates.
3. GEMM. One dynamic queue: every GEMM1 item ``(m, n)``, then every GEMM2
   item. A CTA's first item is static, later ones come from one agent claim
   each. GEMM1 is the stage-1 body of ``build_gemm_a8w4_tdm_task`` (act + fp4
   quant into a2, stored SCOPE_DEV); its A owners wait for the rows-landed
   gate while its B owners stream W1, and the item completing an m-tile's
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

large_m (``layout="compact"``, AITER_MEGA_LARGE) replaces 1. with a count-first
front end on the staged compact layout (expert-major, each expert padded to
``tile_m``, sources in rank order inside it), so its rows, m-tiles and GEMMs
are the staged path's, without per-expert slots sized for every token:

1a. TALLY. CTAs 1..T each count a chunk of this rank's routes in an LDS
    histogram, then take one device atomic per (CTA, segment) for the chunk's
    base in the segment (a segment is a destination rank's expert); each
    route's (segment, rank in segment) goes to a route table. The last tally
    CTA sends the segment totals, every dword tagged with the launch's 16-bit
    tag, into each peer's X1 row for this source.
1b. LAYOUT. CTA 0 polls the X1 rows until every dword carries this launch's
    tag, lays every destination's segments out from the matrix (all ranks
    compute the same layout), keeps this rank's receive counts, publishes its
    send bases in ``N_SREP`` replicas and opens every CTA's send gate.
1c. SEND. One wave per token quantizes the row in registers, waits for its
    CTA's send gate and stores the row and its rowmap pair at each route's row.
    Then 2. to 5., the plan from the receive counts CTA 0 already holds.

Forward progress: producers wait on nothing (large_m's on CTA 0's layout, which
waits only on the peers' tallies); the plan waits only on producers (all ranks
run theirs first); a GEMM1 item waits on nothing; a GEMM2 item waits only on
GEMM1 items already claimed; the reduce waits on the peers' closes, which
depend only on the above. All CTAs must be resident at once (the grid is the CU
count times ``DMK_CTAS_PER_CU`` for small_m, ``LM_CTAS_PER_CU`` for large_m, and
no other kernel may hold CUs). A wait
traps after ``TIMEOUT_S``: a peer's launch may start seconds late.

Cross-launch reuse: rank s writes d's ``dmk_disp`` / ``dmk_rowmap`` /
``running`` in launch L+1 only after s's launch L ended, which needed d's close
of L, which comes after every read d made of them in L (and after CTA 0 reset
the counters launch L+1 uses). Rows of ``comb_inp`` on s are written in L+1
only for tokens s produced in L+1, i.e. after s's reduce of L. large_m sends to
d only once d's X1 row of the same launch arrived, i.e. once d's previous
launch ended; that holds whichever path (large_m or staged, which share
``disp_out``, ``ep_rowmap`` and ``comb_inp``) ran it, as the staged dispatch
likewise waits for d's counts.

Start handshake (off by default, ``AITER_DMK_NO_HS=0`` turns it on): small_m
then sends only once every peer's ``started`` flag carries this launch's
generation, which CTA 0 polls and passes on through each CTA's send gate. It
was added when writes that reached a rank still between two launches came out
corrupted now and then (about 1 in 400 calls had one token wrong). That no
longer reproduces: without it, ~72k calls with up to 1 ms of random launch skew
between ranks and staged calls interleaved matched the staged path bit for
bit (4x gfx1250), and it costs ~5 us per launch. large_m needs no such step:
its rows go out after every peer's X1.

Generation and parity come from u64 entry tickets (``G`` per launch over
``N_TICKET`` counters): every rank launches once per layer, so they agree.
"""

from __future__ import annotations

import os

import flydsl.compiler as flyc
import flydsl.expr as fx
import mori.cco.device.flydsl as cco
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.expr import arith, const_expr, range_constexpr, rocdl, tdm_ops
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
    _rawbar,
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
CTL_FAIL = 5121  # + wait site: AITER_DMK_NOTRAP's timed-out waits (debug)
CTL_TALLY = 5248  # + 32 * parity: compact layout, tally CTAs done
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
#: peers' slot atomics; ``src_done`` / ``comb_flag`` one 256 B line per source,
#: each array room for the most ranks the kernels admit (one wave of them).
FLAG_STRIDE = 256
FLAG_SPAN = WAVE * FLAG_STRIDE

#: Compact layout (large_m): routes one tally CTA counts (its LDS route table),
#: the most tally CTAs, the words between two segments' global counters, the
#: send-table replicas, and the bits of a route slot's segment field.
TALLY_MAXR = 2048
TALLY_MAX = 64
SEG_STRIDE = 32
N_SREP = 16
SEG_BITS = 10
#: LDS the compact front end may take out of the GEMM arena on the host's
#: admission check (the kernel checks against the arena it actually builds;
#: large_m's default GEMM arena is ~70 KB).
LM_SCRATCH_MAX = 48 << 10


def lm_scratch_bytes(*, npes: int, experts_per_rank: int) -> int:
    """LDS the compact front end uses in the GEMM arena: CTA 0's X1 matrix,
    send table and receive counts, or a tally CTA's histogram, bases and
    route table, whichever is larger."""
    segs = dmk_segs_pad(npes=npes, experts_per_rank=experts_per_rank)
    lead = (int(npes) + 1) * segs * 4 + dmk_epr_pad(experts_per_rank) * 4
    tally = 2 * segs * 4 + TALLY_MAXR * 4
    return max(lead, tally)


#: Seconds a wait on another CTA or rank spins before it traps. The ranks'
#: launches can be seconds apart (each rank JIT-compiles its own kernel before
#: its first one), so a peer's late arrival must not look like a hang; the
#: staged path's waits never give up at all. AITER_DMK_TIMEOUT_S overrides
#: (read when the kernel is built, so it is part of the JIT cache key).
TIMEOUT_S = 120.0
NOW_HZ = 100_000_000  # _now() ticks per second
#: s_sleep units between polls (AITER_DMK_POLL_SLEEP overrides).
POLL_SLEEP = 32
#: CTA 0 is the only poller of the peers' flags; it can poll tightly.
LEAD_SLEEP = 1
#: A CTA polls only its own gate line, so a short sleep costs no one.
GATE_SLEEP = 32

#: Phase stamps per CTA under AITER_DMK_TRACE: start, ticket, produced,
#: planned, GEMM1 done, GEMM2 done, combine flags seen, end. Slots 24-26 hold
#: the CTA's GEMM item count, its last item's start and its first GEMM2
#: item's start; slot 27 CTA 0's every-row-landed time (EARLY).
TR_N = 40
TR_ITEMS, TR_LAST_ITEM, TR_FIRST_G2, TR_LANDED = 24, 25, 26, 27
#: Per-item time sums past each CTA's first item (100 MHz ticks): GEMM1
#: fill / mainloop / epilogue / tail and count, then the same for GEMM2.
TR_IT = 28
RT_N = 1024

#: W1 prefetch (AITER_DMK_PREFETCH_KB): the fewest tokens of a step that runs
#: it, and the 16 B loads each lane keeps in flight (deeper ran slower).
PREF_MIN_TOK = 32
PREF_DEPTH = 8

#: Elements a reduce wave owns per item: 32 lanes x 16.
RED_ELEMS = WAVE * 16


def dmk_ctl_arena_bytes(*, npes: int, experts_per_rank: int) -> int:
    """Bytes of the arena's ``dmk_ctl`` region: slot counters, then src_done,
    comb_flag (``FLAG_SPAN`` apart), started, reserved and expected (one line
    per source each)."""
    return dmk_src_off(experts_per_rank) + 2 * FLAG_SPAN + 3 * npes * FLAG_STRIDE


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


def dmk_segs_pad(*, npes: int, experts_per_rank: int) -> int:
    """Compact layout: (destination, local expert) segments, padded to 128 so
    one wave covers a row of them with a b128 per lane."""
    return (int(npes) * int(experts_per_rank) + 127) // 128 * 128


def dmk_ctl_layout(
    *, experts_per_rank: int, mt_cap: int, workers: int, npes: int = 0, compact=False
):
    """Word offsets of the local control block: the per-m-tile GEMM1 lines,
    the plan replicas (per expert ``rows``, 16 bits each), one plan, one
    combine and one send gate line per CTA, and on the compact layout the
    segment counters and the send-table replicas. No line is read or
    polled by more than a few CTAs: hundreds of CTAs on one uncached line
    serialize at its DRAM bank."""
    epr_pad = dmk_epr_pad(experts_per_rank)
    plan = CTL_G1 + int(mt_cap) * G1_STRIDE
    # A replica holds WAVE packed-count dwords per chunk pair (one pair at
    # least); the tagged replicas store each as a (counts, generation) i64.
    plan_slot = 2 * WAVE * max(1, epr_pad // WAVE // 2)
    gate_p = plan + PLAN_REPS * plan_slot
    gate_c = gate_p + int(workers) * GATE_STRIDE
    gate_s = gate_c + int(workers) * GATE_STRIDE
    gate_r = gate_s + int(workers) * GATE_STRIDE
    end = gate_r + int(workers) * GATE_STRIDE
    lay = {
        "plan": plan,
        "plan_slot": plan_slot,
        "gate_p": gate_p,
        "gate_c": gate_c,
        "gate_s": gate_s,
        "gate_r": gate_r,
    }
    if compact:
        segs = dmk_segs_pad(npes=npes, experts_per_rank=experts_per_rank)
        lay["seg_cnt"] = end
        lay["srep"] = end + segs * SEG_STRIDE
        end = lay["srep"] + N_SREP * segs
    lay["words"] = end
    return lay


def dmk_ctl_words(**kw) -> int:
    """i32 words of the local control block."""
    return dmk_ctl_layout(**kw)["words"]


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
    layout="slots",
    mt_cap=None,
):
    """Build the decode megakernel; returns a ``@flyc.jit`` launcher.

    ``layout="slots"`` (small_m): expert ``e`` owns rows ``[e*cap, (e+1)*cap)``
    and a route reserves its slot with a remote atomic. ``layout="compact"``
    (large_m): the staged compact layout (expert-major, each expert padded to
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
    if COMPACT and npes * int(max_tok_per_rank) > 0xFFFF:
        raise ValueError("the compact plan holds an expert's rows in 16 bits")
    if G % WAVE or G % N_TICKET:
        raise ValueError(f"workers={G} must be a multiple of {WAVE} and {N_TICKET}")
    if npes > WAVE or E != npes * EPR:
        raise ValueError("experts must split evenly over at most 32 ranks")
    EPR_PAD = dmk_epr_pad(EPR)
    EPR_CH = EPR_PAD // WAVE
    SRC_OFF = dmk_src_off(EPR)
    CF_OFF = SRC_OFF + FLAG_SPAN
    ST_OFF = SRC_OFF + 2 * FLAG_SPAN
    RS_OFF = ST_OFF + npes * FLAG_STRIDE
    EXP_OFF = RS_OFF + npes * FLAG_STRIDE
    MT_PER_E = CAP // TM
    MT_CAP = int(mt_cap) if COMPACT else EPR * MT_PER_E
    N1 = 2 * I
    NT1 = N1 // int(tile_n)
    NT2 = H // int(tile_n2)
    if N1 % int(tile_n) or H % int(tile_n2):
        raise ValueError("tile_n / tile_n2 must divide 2*inter_dim / model_dim")
    DEST_STRIDE = int(max_tok_per_rank) * K
    RED_CH = H // RED_ELEMS
    _lay = dmk_ctl_layout(
        experts_per_rank=EPR, mt_cap=MT_CAP, workers=G, npes=npes, compact=COMPACT
    )
    CTL_WORDS = _lay["words"]
    PLAN_W, PLAN_SLOT = _lay["plan"], _lay["plan_slot"]
    GATE_P, GATE_C, GATE_S = _lay["gate_p"], _lay["gate_c"], _lay["gate_s"]
    GATE_R = _lay["gate_r"]
    if MT_CAP > 0xFFFF or (not COMPACT and CAP > 0xFFFF):
        raise ValueError("the packed plan holds m-tile ends and rows in 16 bits")
    SEGS_PAD = dmk_segs_pad(npes=npes, experts_per_rank=EPR)
    if COMPACT:
        SEG_CNT_W, SREP_W = _lay["seg_cnt"], _lay["srep"]
        if npes * SEGS_PAD * 4 > dmk_src_off(EPR) or SEGS_PAD > (1 << SEG_BITS):
            raise ValueError("the X1 rows do not fit the control region")
        MAXR = int(max_tok_per_rank) * K
        if MAXR > TALLY_MAX * TALLY_MAXR:
            raise ValueError("more routes per rank than the tally CTAs count")
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
    # small_m's start handshake is off by default; AITER_DMK_NO_HS=0 restores it.
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
    PREF_B = 0 if COMPACT else int(os.environ.get("AITER_DMK_PREFETCH_KB", "256")) * 1024
    PREF_B = min(PREF_B, int(tile_n) * (H // 2))
    GATE_ONCE = int(os.environ.get("AITER_DMK_GATE_ONCE", "1"))
    # AITER_DMK_LOOKAHEAD: a GEMM item's drain issues the next item's first
    # k-tiles (same GEMM only), so the pipeline stays full across items. The
    # drain frees stage j for k-tile j only when the K tiles are a multiple of
    # the stages, so GEMM1 takes 4 stages where that divides and 3 does not.
    LA = 0 if COMPACT else int(os.environ.get("AITER_DMK_LOOKAHEAD", "0"))
    post_mc = TM <= 32
    k_tiles1 = -(-H // int(tile_k))
    k_tiles2 = -(-I // int(tile_k2))
    nb1 = min(int(num_buffers), max(1, H // int(tile_k)))
    nb2 = min(int(num_buffers2), max(1, I // int(tile_k2)))
    # Debug only: AITER_DMK_LA_DBG bit 0 GEMM1, bit 1 GEMM2, bit 2 the A side.
    LA_DBG = int(os.environ.get("AITER_DMK_LA_DBG", "7"))
    if LA and LA_DBG & 1 and post_mc and k_tiles1 % nb1 and not k_tiles1 % 4:
        nb1 = 4
    LA1 = bool(LA and post_mc and not k_tiles1 % nb1 and LA_DBG & 1)
    LA2 = bool(LA and post_mc and DEFER_SC and not k_tiles2 % nb2 and LA_DBG & 2)
    LA_A = bool(LA_DBG & 4)
    # AITER_DMK_EXPECT (needs EARLY): each source sends every destination its
    # route count up front, and CTA 0 plans once its slot counters add up to
    # the sum -- no reserved tally (one device atomic per route wave on one
    # word) and no reserved flag hop.
    EXPECT = 0 if (COMPACT or not EARLY) else int(os.environ.get("AITER_DMK_EXPECT", "0"))
    if EXPECT and int(max_tok_per_rank) * K > 0xFFFF:
        EXPECT = 0  # the count rides the low 16 bits of the tagged word
    # AITER_DMK_ROWFLAG (needs EARLY): each route stores the generation into
    # its slot's rows-landed word once its row retired, and a GEMM item checks
    # its own m-tile's rows -- no produced tally, src_done or rows-landed gate.
    ROWFLAG = (
        0 if (COMPACT or not EARLY) else int(os.environ.get("AITER_DMK_ROWFLAG", "0"))
    )
    # AITER_DMK_LEAD_FREE: CTA 0 takes no routes, so its wave 0 polls the
    # peers as soon as its ticket is in, not after its own rows retired.
    LEAD_FREE = 0 if COMPACT else int(os.environ.get("AITER_DMK_LEAD_FREE", "0"))
    # Producer waves, and the most routes one of them takes (ROWFLAG keeps
    # each route's (destination, row) in LDS until the rows retired).
    PROD_W = (G - 1) * 4 if LEAD_FREE else G * 4
    R_MAX = -(-int(max_tok_per_rank) * K // PROD_W)
    knob_sfx = "".join(
        f"_{name}{val}"
        for name, val, default in (
            ("ps", POLL_SL, POLL_SLEEP),
            ("hs", int(HS and not COMPACT), 0),
            ("pt", PLAN_TAG, 1 if TM <= 16 else 0),
            ("ds", DEFER_SC, 1),
            ("ep", EARLY, 1),
            ("pf", PREF_B // 1024, min(256 * 1024, int(tile_n) * (H // 2)) // 1024),
            ("go", GATE_ONCE, 1),
            ("la", LA, 0),
            ("lad", LA_DBG, 7),
            ("ex", EXPECT, 0),
            ("rf", ROWFLAG, 0),
            ("lf", LEAD_FREE, 0),
        )
        if val != default
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
        hooks1_holder = [None] if TRACE else None
        hooks2_holder = [None] if TRACE else None
        la1_holder = [None] if LA1 else None
        la2_holder = [None] if LA2 else None
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
            num_buffers=nb1,
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
            item_hooks=hooks1_holder,
            lookahead=la1_holder,
            a_cache_modifier=POL_A1,
            c_cache_modifier=POL_C,
            **common,
        )
        g2_task, block2, g2_name, _, lay2 = build_gemm_a8w4_tdm_task(
            K=I,
            tile_m=TM,
            tile_n=tile_n2,
            tile_k=tile_k2,
            num_buffers=nb2,
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
            item_hooks=hooks2_holder,
            lookahead=la2_holder,
            a_cache_modifier=POL_A,
            scatter_cache_policy=POL_S,
            defer_ep_wait=DEFER_SC,
            **common,
        )
        if block != 4 * WAVE or block2 != block:
            raise ValueError(f"expected 4-wave GEMM blocks, got {block}/{block2}")
        ARENA = max(int(lay1["arena"]), int(lay2["arena"]))
        # LDS beyond the GEMM arena: broadcast words, then the plan's per-expert
        # inclusive m-tile ends and row counts.
        BC_B = 128 if TRACE else 64
        BC_GATE_R = 24  # this CTA has seen the rows-landed gate open
        BC_LA = 28  # sides of this CTA's next item its drain issued (lookahead)
        PLAN_B = 2 * EPR_PAD * 4
        CTRL_B = BC_B + PLAN_B
        if COMPACT and lm_scratch_bytes(npes=npes, experts_per_rank=EPR) > ARENA:
            raise ValueError("the compact front end's LDS does not fit the GEMM arena")
        if int(ctas_per_cu) * (((ARENA + CTRL_B) + 2047) // 2048 * 2048) > 327680:
            raise ValueError(
                f"{ctas_per_cu} CTAs of {ARENA + CTRL_B} B LDS do not fit one CU"
            )
        kname = (
            ("large_mega" if COMPACT else "decode_mega")
            + f"_w{G}_r{rank}of{npes}_c{CAP if not COMPACT else MT_CAP * TM}"
            + (f"_ph{PH}" if PH != 15 else "")
            + knob_sfx
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

            def stamp(k, value=None):
                if const_expr(TRACE):  # noqa: SIM102 - preserve DSL staging
                    if is_t0:
                        # Relaxed: a release store would flush L2 per stamp.
                        _store_i64_sys(
                            addr_ctl
                            + fx.Int64(TR_BASE * 4)
                            + fx.Int64(bid * fx.Int32(TR_N * 8) + fx.Int32(k * 8)),
                            _now() if value is None else value,
                        )

            def spin_end():
                return _now() + fx.Int64(TIMEOUT_TICKS)

            def alive(t_end):
                return fx.Int32((_now() < t_end).select(fx.Int32(1), fx.Int32(0)))

            def spin_fail(site):
                if const_expr(NOTRAP):
                    if lane == fx.Int32(0):
                        comm_ops.atomic_add_agent(
                            addr_ctl + fx.Int64((CTL_FAIL + site) * 4), fx.Int32(1)
                        )
                else:
                    _trap()

            stamp(0)
            if const_expr(INV):  # noqa: SIM102 - preserve DSL staging
                if wave == fx.Int32(0):
                    comm_ops.fence_acquire(fx.rocdl.SyncScope.OneAs)

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
                # X1 tag: 16 bits, never 0, differs from the previous launch's.
                comm_ops.store_i32_lds(
                    fx.Int64(bc + fx.Int32(16)),
                    _i32_lo(l64 % fx.Int64(65535)) + fx.Int32(1),
                )
                comm_ops.store_i32_lds(fx.Int64(bc + fx.Int32(BC_GATE_R)), fx.Int32(0))
                comm_ops.store_i32_lds(fx.Int64(bc + fx.Int32(BC_LA)), fx.Int32(0))
            rocdl.s_wait_dscnt(0)
            rocdl.s_barrier_signal(-1)
            rocdl.s_barrier_wait(-1)
            gen = _uniform(comm_ops.load_i32_lds(fx.Int64(bc)))
            par = _uniform(comm_ops.load_i32_lds(fx.Int64(bc + fx.Int32(4))))
            tag16 = _uniform(comm_ops.load_i32_lds(fx.Int64(bc + fx.Int32(16))))
            other = fx.Int32(1) - par
            if const_expr(DELAY_US):
                d_end = _now() + fx.Int64(DELAY_US * (NOW_HZ // 1_000_000))
                d_live = fx.Int32(1)
                while d_live != fx.Int32(0):
                    fx.rocdl.s_sleep(32)
                    d_live = alive(d_end)

            stamp(1)
            # ---- CTA 0: reset the next launch's counters ----
            # The peers' slot atomics of the next launch land in
            # running[other] only after this rank's close, which counts this
            # reset as one unit once its stores have retired.
            if (bid == fx.Int32(0)) & (wave == fx.Int32(0)):
                if const_expr(not COMPACT):
                    for j in range_constexpr(EPR_CH):
                        buffer_store(
                            fx.Int32(0),
                            rsrc_actl,
                            (other * fx.Int32(EPR_PAD) + lane + fx.Int32(j * WAVE))
                            * fx.Int32(RUN_STRIDE),
                            cache_modifier=SCOPE_SYS,
                        )
                # Lane 3 resets the tally (large_m) or reserved (EARLY) count.
                if lane < fx.Int32(4 if (COMPACT or EARLY) else 3):
                    buffer_store(
                        fx.Int32(0),
                        rsrc_ctl,
                        fx.Int32(
                            (lane == fx.Int32(3)).select(
                                fx.Int32(CTL_TALLY),
                                fx.Int32(CTL_PROD)
                                + lane * fx.Int32(CTL_CLOSE - CTL_PROD),
                            )
                        )
                        + other * fx.Int32(32),
                        cache_modifier=SCOPE_DEV,
                    )
                rocdl.s_wait_storecnt(0)

            def poll_peer_flags(flag_off, site):
                src = fx.Int32((lane < fx.Int32(npes)).select(lane, fx.Int32(0)))
                flag_w = fx.Int32(flag_off // 4) + src * fx.Int32(FLAG_STRIDE // 4)
                cur = _load_i32_sys(rsrc_actl, flag_w)
                bad = fx.Int32(ballot(T.i32, (lane < fx.Int32(npes)) & (cur != gen)))
                t_end = spin_end()
                live = fx.Int32(1)
                while (bad != fx.Int32(0)) & (live != fx.Int32(0)):
                    fx.rocdl.s_sleep(LEAD_SLEEP)
                    cur = _load_i32_sys(rsrc_actl, flag_w)
                    bad = fx.Int32(
                        ballot(T.i32, (lane < fx.Int32(npes)) & (cur != gen))
                    )
                    live = alive(t_end)
                if bad != fx.Int32(0):
                    spin_fail(site)

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

            def wait_gate(gate_w, stride, site):
                rdy_w = fx.Int32(gate_w) + bid * fx.Int32(stride)
                cur = _uniform(_load_i32_dev(rsrc_ctl, rdy_w))
                t_end = spin_end()
                live = fx.Int32(1)
                while (cur != gen) & (live != fx.Int32(0)):
                    fx.rocdl.s_sleep(GATE_SLEEP)
                    cur = _uniform(_load_i32_dev(rsrc_ctl, rdy_w))
                    live = alive(t_end)
                if cur != gen:
                    spin_fail(site)

            def wait_landed():
                """The rows-landed gate, polled until any wave of this CTA has
                seen it open; later items read that from LDS."""
                seen_w = fx.Int64(bc + fx.Int32(BC_GATE_R))
                if const_expr(not GATE_ONCE):
                    wait_gate(GATE_P if COMPACT else GATE_R, GATE_STRIDE, 7)
                else:
                    if _uniform(comm_ops.load_i32_lds(seen_w)) == fx.Int32(0):
                        wait_gate(GATE_P if COMPACT else GATE_R, GATE_STRIDE, 7)
                        if lane == fx.Int32(0):
                            comm_ops.store_i32_lds(seen_w, fx.Int32(1))

            # AITER_DMK_ROWFLAG: rows-landed words, one per slot, past the
            # rowmap pairs; a route stores the generation into its slot's
            # once its row and rowmap pair retired (SCOPE_SYS).
            LAND_OFF = EPR * CAP * 8
            rsrc_land = create_buffer_resource_from_addr(
                peer_va(rank, off_rowmap) + fx.Int64(LAND_OFF),
                num_records_bytes=EPR * CAP * 4,
            )

            def rows_pending(mt_idx, rows):
                """Loads of m-tile ``mt_idx``'s first ``rows`` landed words, and
                a function giving the mask of lanes still waiting on them."""
                base = mt_idx * fx.Int32(TM)
                parts = []
                for c in range_constexpr((TM + WAVE - 1) // WAVE):
                    slot = lane + fx.Int32(c * WAVE)
                    need = slot < rows
                    word = fx.Int32(need.select(base + slot, fx.Int32(EPR * CAP)))
                    parts.append((need, _load_i32_sys(rsrc_land, word)))

                def pending():
                    bad = fx.Int32(0)
                    for need, f in parts:
                        bad = bad | fx.Int32(ballot(T.i32, need & (f != gen)))
                    return bad

                return pending

            def wait_rows(mt_idx, rows, site):
                """Until every row of m-tile ``mt_idx`` has landed."""
                bad = rows_pending(mt_idx, rows)()
                t_end = spin_end()
                live = fx.Int32(1)
                while (bad != fx.Int32(0)) & (live != fx.Int32(0)):
                    fx.rocdl.s_sleep(GATE_SLEEP)
                    bad = rows_pending(mt_idx, rows)()
                    live = alive(t_end)
                if bad != fx.Int32(0):
                    spin_fail(site)

            def poll_expected(site):
                """Wave 0 of CTA 0 (EXPECT): wait for every source's route count
                for this rank, then for the slot counters to add up to their
                sum; the counters are final then. Returns them, clamped."""
                src = fx.Int32((lane < fx.Int32(npes)).select(lane, fx.Int32(0)))
                flag_w = fx.Int32(EXP_OFF // 4) + src * fx.Int32(FLAG_STRIDE // 4)

                def stale(cur):
                    return fx.Int32(
                        ballot(
                            T.i32,
                            (lane < fx.Int32(npes))
                            & (((cur >> fx.Int32(16)) & fx.Int32(0xFFFF)) != tag16),
                        )
                    )

                cur = _load_i32_sys(rsrc_actl, flag_w)
                bad = stale(cur)
                t_end = spin_end()
                live = fx.Int32(1)
                while (bad != fx.Int32(0)) & (live != fx.Int32(0)):
                    fx.rocdl.s_sleep(LEAD_SLEEP)
                    cur = _load_i32_sys(rsrc_actl, flag_w)
                    bad = stale(cur)
                    live = alive(t_end)
                if bad != fx.Int32(0):
                    spin_fail(site)
                mine = fx.Int32(
                    (lane < fx.Int32(npes)).select(cur & fx.Int32(0xFFFF), fx.Int32(0))
                )
                total = _uniform(
                    readlane(
                        T.i32, arith.unwrap(_wave32_inclusive_scan_i32(mine, lane)), WAVE - 1
                    )
                )

                def counted():
                    # The raw counters go to the LDS plan words (rewritten by
                    # the plan right after), so the last poll's values survive.
                    s_ = fx.Int32(0)
                    for j in range_constexpr(EPR_CH):
                        e_ = lane + fx.Int32(j * WAVE)
                        c_ = _load_i32_sys(
                            rsrc_actl,
                            (par * fx.Int32(EPR_PAD) + e_) * fx.Int32(RUN_STRIDE),
                        )
                        c_ = fx.Int32((e_ < fx.Int32(EPR)).select(c_, fx.Int32(0)))
                        comm_ops.store_i32_lds(fx.Int64(lds_end + e_ * fx.Int32(4)), c_)
                        s_ = s_ + c_
                    return _uniform(
                        readlane(
                            T.i32, arith.unwrap(_wave32_inclusive_scan_i32(s_, lane)), WAVE - 1
                        )
                    )

                got = counted()
                live = fx.Int32(1)
                while (got != total) & (live != fx.Int32(0)):
                    fx.rocdl.s_sleep(LEAD_SLEEP)
                    got = counted()
                    live = alive(t_end)
                if got != total:
                    spin_fail(site)
                rocdl.s_wait_dscnt(0)
                cnts = []
                for j in range_constexpr(EPR_CH):
                    e_ = lane + fx.Int32(j * WAVE)
                    c_ = fx.Int32(comm_ops.load_i32_lds(fx.Int64(lds_end + e_ * fx.Int32(4))))
                    cnts.append(fx.Int32((c_ > fx.Int32(CAP)).select(fx.Int32(CAP), c_)))
                return cnts

            def lead_plan_lds(cnts):
                """CTA 0, wave 0: its own LDS plan from the per-expert counts,
                and the counts packed two per dword (dword w = cnt(2w) |
                cnt(2w + 1) << 16, built by bpermute) for the replicas."""
                plan_from_counts(cnts)
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

            def lead_plan_replicas():
                """CTA 0, every wave: a quarter of the plan replicas from the
                packed counts in LDS (after a barrier past lead_plan_lds)."""
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
                        if const_expr(PLAN_TAG):
                            buffer_store(
                                fx.Vector.from_elements([pk[j2], gen], dtype=fx.Int32),
                                rsrc_ctl,
                                slot_w + (fx.Int32(j2 * WAVE) + lane) * fx.Int32(2),
                                cache_modifier=SCOPE_DEV,
                            )
                        else:
                            buffer_store(
                                pk[j2],
                                rsrc_ctl,
                                slot_w + fx.Int32(j2 * WAVE) + lane,
                                cache_modifier=SCOPE_DEV,
                            )

            def load_tagged_replica(slot_w):
                """PLAN_TAG: (packed counts, all tags == gen) of one replica."""
                pkw, ok = [], fx.Int32(1)
                for j2 in range_constexpr(EPR_CH // 2 if EPR_CH > 1 else 1):
                    vv = fx.Vector(
                        buffer_load(
                            rsrc_ctl,
                            slot_w + (fx.Int32(j2 * WAVE) + lane) * fx.Int32(2),
                            vec_width=2,
                            dtype=T.i32,
                            cache_modifier=SCOPE_DEV | (1 << 31),
                        )
                    )
                    pkw.append(fx.Int32(vv[0]))
                    ok = ok & fx.Int32(
                        (fx.Int32(vv[1]) == gen).select(fx.Int32(1), fx.Int32(0))
                    )
                return pkw, ok

            def follower_plan(gate_w, site):
                """Wave 0 of every other CTA: once its gate opens, its LDS plan
                from its replica."""
                slot_w = fx.Int32(PLAN_W) + (bid & fx.Int32(PLAN_REPS - 1)) * fx.Int32(
                    PLAN_SLOT
                )
                if const_expr(PLAN_TAG):
                    _, ok = load_tagged_replica(slot_w)
                    bad = fx.Int32(ballot(T.i32, ok == fx.Int32(0)))
                    t_end = spin_end()
                    live = fx.Int32(1)
                    while (bad != fx.Int32(0)) & (live != fx.Int32(0)):
                        fx.rocdl.s_sleep(GATE_SLEEP)
                        _, ok = load_tagged_replica(slot_w)
                        bad = fx.Int32(ballot(T.i32, ok == fx.Int32(0)))
                        live = alive(t_end)
                    if bad != fx.Int32(0):
                        spin_fail(site)
                    stamp(9)
                    pkw, _ = load_tagged_replica(slot_w)
                else:
                    wait_gate(gate_w, GATE_STRIDE, site)
                    stamp(9)
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
                            d_ & fx.Int32(0xFFFF),
                            (d_ >> fx.Int32(16)) & fx.Int32(0xFFFF),
                        )
                    )
                    cnts.append(fx.Int32((e_ < fx.Int32(EPR)).select(c_, fx.Int32(0))))
                plan_from_counts(cnts)

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

            if const_expr(not COMPACT and HS):  # noqa: SIM102 - preserve DSL staging
                # ---- start handshake (small_m, AITER_DMK_NO_HS=0) ----
                # A peer writes this rank's counters, rows and rowmap only once
                # this launch has started (see the module docstring).
                if (bid == fx.Int32(0)) & (wave == fx.Int32(0)):
                    if lane < fx.Int32(npes):
                        comm_ops.store_i32_global_system_monotonic(
                            peer_va(lane, off_ctl)
                            + fx.Int64(ST_OFF + rank * FLAG_STRIDE),
                            gen,
                        )
                    poll_peer_flags(ST_OFF, 8)
                    open_gates(GATE_S, GATE_STRIDE)

            # ======================== 1. PRODUCE ========================
            rsrc_x = create_buffer_resource_from_addr(addr_x)
            rsrc_ids = create_buffer_resource_from_addr(addr_ids)
            rsrc_wts = create_buffer_resource_from_addr(addr_wts)
            wave_gid = bid * fx.Int32(4) + wave
            n_mine = fx.Int32(0)
            n_routes = tok * fx.Int32(K)
            if const_expr(COMPACT):
                # ======== 1. FRONT-END (compact layout, large_m) ========
                # LDS scratch in the GEMM arena (no GEMM item runs before the
                # plan gate): a tally CTA's histogram, bases and route table;
                # CTA 0's X1 matrix, send table and receive counts.
                rsrc_route = create_buffer_resource_from_addr(
                    addr_route, num_records_bytes=4 * MAXR
                )
                T_ = (n_routes + fx.Int32(TALLY_MAXR - 1)) // fx.Int32(TALLY_MAXR)
                T_ = fx.Int32((T_ < fx.Int32(1)).select(fx.Int32(1), T_))
                T_ = fx.Int32(
                    (T_ > fx.Int32(TALLY_MAX)).select(fx.Int32(TALLY_MAX), T_)
                )
                # ---- tally: CTAs 1..T_ count a chunk of routes each. LDS
                # first, then one device atomic per (CTA, segment): the
                # segment's global base for this chunk, so a hot segment (the
                # shared expert) takes T_ atomics, not one per route.
                if (bid >= fx.Int32(1)) & (bid <= T_):
                    l_hist = ar
                    l_hb = ar + fx.Int32(SEGS_PAD * 4)
                    l_rt = ar + fx.Int32(2 * SEGS_PAD * 4)
                    C_ = (n_routes + T_ - fx.Int32(1)) // T_
                    r0 = (bid - fx.Int32(1)) * C_
                    nloc = n_routes - r0
                    nloc = fx.Int32((nloc > C_).select(C_, nloc))
                    nloc = fx.Int32((nloc < fx.Int32(0)).select(fx.Int32(0), nloc))
                    for i in range(tid, SEGS_PAD, 4 * WAVE):
                        comm_ops.store_i32_lds(fx.Int64(l_hist + i * fx.Int32(4)), 0)
                    _rawbar()
                    for i in range(tid, nloc, 4 * WAVE):
                        gid = fx.Int32(
                            buffer_load(rsrc_ids, r0 + i, vec_width=1, dtype=T.i32)
                        )
                        ok = (gid >= fx.Int32(0)) & (gid < fx.Int32(E))
                        g_s = fx.Int32(ok.select(gid, fx.Int32(0)))
                        loc = fx.Int32(
                            comm_ops.atomic_add_lds(
                                fx.Int64(l_hist + g_s * fx.Int32(4)),
                                fx.Int32(ok.select(fx.Int32(1), fx.Int32(0))),
                            )
                        )
                        comm_ops.store_i32_lds(
                            fx.Int64(l_rt + i * fx.Int32(4)),
                            fx.Int32(
                                ok.select(
                                    g_s | (loc << fx.Int32(SEG_BITS)), fx.Int32(-1)
                                )
                            ),
                        )
                    _rawbar()
                    # Every segment's atomic in flight before any return is
                    # used: a loop storing each return serialized the round
                    # trips.
                    bases = []
                    for c in range_constexpr(SEGS_PAD // (4 * WAVE)):
                        i = tid + fx.Int32(c * 4 * WAVE)
                        n_ = fx.Int32(
                            comm_ops.load_i32_lds(fx.Int64(l_hist + i * fx.Int32(4)))
                        )
                        bases.append(
                            _buffer_atomic_add_ret(
                                rsrc_ctl,
                                fx.Int32(
                                    (n_ > fx.Int32(0)).select(
                                        (fx.Int32(SEG_CNT_W) + i * fx.Int32(SEG_STRIDE))
                                        * fx.Int32(4),
                                        fx.Int32(_OOB),
                                    )
                                ),
                                n_,
                                SCOPE_DEV,
                            )
                        )
                    for c in range_constexpr(SEGS_PAD // (4 * WAVE)):
                        i = tid + fx.Int32(c * 4 * WAVE)
                        comm_ops.store_i32_lds(
                            fx.Int64(l_hb + i * fx.Int32(4)), bases[c]
                        )
                    _rawbar()
                    for i in range(tid, nloc, 4 * WAVE):
                        v = fx.Int32(
                            comm_ops.load_i32_lds(fx.Int64(l_rt + i * fx.Int32(4)))
                        )
                        sg = v & fx.Int32((1 << SEG_BITS) - 1)
                        hb = fx.Int32(
                            comm_ops.load_i32_lds(fx.Int64(l_hb + sg * fx.Int32(4)))
                        )
                        slot = fx.Int32(
                            (v >= fx.Int32(0)).select(
                                sg
                                | (
                                    (hb + (v >> fx.Int32(SEG_BITS)))
                                    << fx.Int32(SEG_BITS)
                                ),
                                fx.Int32(-1),
                            )
                        )
                        buffer_store(slot, rsrc_route, r0 + i, cache_modifier=SCOPE_DEV)
                    rocdl.s_wait_storecnt(0)
                    _rawbar()
                    if is_t0:
                        old_t = fx.Int32(
                            comm_ops.atomic_add_agent(
                                addr_ctl
                                + fx.Int64(CTL_TALLY * 4)
                                + fx.Int64(par) * 128,
                                fx.Int32(1),
                            )
                        )
                        comm_ops.store_i32_lds(fx.Int64(bc + fx.Int32(20)), old_t)
                    _rawbar()
                    old_t = _uniform(comm_ops.load_i32_lds(fx.Int64(bc + fx.Int32(20))))
                    # The last tally CTA reads (and clears) the segment totals and
                    # sends them, tagged per dword, to every peer: X1.
                    if (old_t == T_ - fx.Int32(1)) & (wave == fx.Int32(0)):
                        # Every count load in flight at once (SCOPE_SYS, not
                        # volatile: a volatile load waits before the next).
                        # They sit in this branch, after the tally atomic that
                        # chose the completer, so none can move above it.
                        cws = []
                        cvs = []
                        for j in range_constexpr(SEGS_PAD // 128):
                            for q in range_constexpr(4):
                                w_ = fx.Int32(SEG_CNT_W) + (
                                    fx.Int32(128 * j + q) + lane * fx.Int32(4)
                                ) * fx.Int32(SEG_STRIDE)
                                cws.append(w_)
                                cvs.append(
                                    fx.Int32(
                                        buffer_load(
                                            rsrc_ctl,
                                            w_,
                                            vec_width=1,
                                            dtype=T.i32,
                                            cache_modifier=SCOPE_SYS,
                                        )
                                    )
                                )
                        for j in range_constexpr(SEGS_PAD // 128):
                            cs = []
                            cw = cws[4 * j : 4 * j + 4]
                            for q in range_constexpr(4):
                                c_ = cvs[4 * j + q]
                                c_ = fx.Int32(
                                    (c_ > fx.Int32(0xFFFF)).select(fx.Int32(0xFFFF), c_)
                                )
                                cs.append((tag16 << fx.Int32(16)) | c_)
                            vec = fx.Vector.from_elements(cs, dtype=fx.Int32)
                            for pe in range_constexpr(npes):
                                rsrc_x1 = create_buffer_resource_from_addr(
                                    peer_va(pe, off_ctl),
                                    num_records_bytes=npes * SEGS_PAD * 4,
                                )
                                buffer_store(
                                    vec,
                                    rsrc_x1,
                                    fx.Int32(rank * SEGS_PAD + 128 * j)
                                    + lane * fx.Int32(4),
                                    cache_modifier=SCOPE_SYS,
                                )
                            for q in range_constexpr(4):
                                buffer_store(
                                    fx.Int32(0),
                                    rsrc_ctl,
                                    cw[q],
                                    cache_modifier=SCOPE_DEV,
                                )
                        stamp(23)
                # ---- CTA 0: X1 from every source -> every destination's
                # layout (the staged compact one: expert-major, each expert
                # padded to tile_m, sources in rank order inside it) -> the
                # send table (replicated) -> send gates. Every rank lays out
                # from the same matrix, so senders and receivers agree.
                if bid == fx.Int32(0):
                    tag_hi = tag16 << fx.Int32(16)

                    def x1_load(w0):
                        return fx.Vector(
                            buffer_load(
                                rsrc_actl,
                                w0,
                                vec_width=4,
                                dtype=T.i32,
                                cache_modifier=SCOPE_SYS | (1 << 31),
                            )
                        )

                    def x1_bad(v4):
                        """Ballot of the lanes holding another launch's tag."""
                        badl = fx.Int32(0)
                        for q in range_constexpr(4):
                            badl = badl | fx.Int32(
                                ((fx.Int32(v4[q]) & fx.Int32(-65536)) != tag_hi).select(
                                    fx.Int32(1), fx.Int32(0)
                                )
                            )
                        return fx.Int32(ballot(T.i32, badl != fx.Int32(0)))

                    for sj in range_constexpr((npes + 3) // 4):
                        s_ = wave + fx.Int32(4 * sj)
                        if s_ < fx.Int32(npes):
                            for j in range_constexpr(SEGS_PAD // 128):
                                w0 = (
                                    s_ * fx.Int32(SEGS_PAD)
                                    + fx.Int32(128 * j)
                                    + lane * fx.Int32(4)
                                )
                                v4 = x1_load(w0)
                                bad = x1_bad(v4)
                                t_end = spin_end()
                                live = fx.Int32(1)
                                while (bad != fx.Int32(0)) & (live != fx.Int32(0)):
                                    fx.rocdl.s_sleep(LEAD_SLEEP)
                                    v4 = x1_load(w0)
                                    bad = x1_bad(v4)
                                    live = alive(t_end)
                                if bad != fx.Int32(0):
                                    spin_fail(0)
                                for q in range_constexpr(4):
                                    comm_ops.store_i32_lds(
                                        fx.Int64(
                                            l_m + (w0 + fx.Int32(q)) * fx.Int32(4)
                                        ),
                                        fx.Int32(v4[q]) & fx.Int32(0xFFFF),
                                    )
                    _rawbar()
                    stamp(20)
                    for dj in range_constexpr((npes + 3) // 4):
                        d_ = wave + fx.Int32(4 * dj)
                        if d_ < fx.Int32(npes):
                            carry = fx.Int32(0)
                            for ch in range_constexpr(EPR_CH):
                                le = lane + fx.Int32(ch * WAVE)
                                live = le < fx.Int32(EPR)
                                sg = fx.Int32(
                                    live.select(d_ * fx.Int32(EPR) + le, fx.Int32(0))
                                )
                                tot = fx.Int32(0)
                                pre = fx.Int32(0)
                                for s2 in range_constexpr(npes):
                                    m_ = fx.Int32(
                                        comm_ops.load_i32_lds(
                                            fx.Int64(
                                                l_m
                                                + (fx.Int32(s2 * SEGS_PAD) + sg)
                                                * fx.Int32(4)
                                            )
                                        )
                                    )
                                    tot = tot + m_
                                    if const_expr(s2 < rank):
                                        pre = pre + m_
                                tot = fx.Int32(live.select(tot, fx.Int32(0)))
                                al = (
                                    (tot + fx.Int32(TM - 1))
                                    // fx.Int32(TM)
                                    * fx.Int32(TM)
                                )
                                incl = _wave32_inclusive_scan_i32(al, lane) + carry
                                start = incl - al
                                carry = _uniform(
                                    readlane(T.i32, arith.unwrap(incl), WAVE - 1)
                                )
                                if live:
                                    comm_ops.store_i32_lds(
                                        fx.Int64(l_sb + sg * fx.Int32(4)), start + pre
                                    )
                                # This rank's own receive side: rows at or past
                                # i32_rows are dropped by senders and receiver
                                # alike (an expert starts on a tile, so whole
                                # tiles go).
                                if (d_ == fx.Int32(rank)) & live:
                                    room = i32_rows - start
                                    room = fx.Int32(
                                        (room < fx.Int32(0)).select(fx.Int32(0), room)
                                    )
                                    comm_ops.store_i32_lds(
                                        fx.Int64(l_rc + le * fx.Int32(4)),
                                        fx.Int32((tot > room).select(room, tot)),
                                    )
                    # The plan is known now (this rank's receive counts, which
                    # the wave laying out this rank wrote): wave 0 lays it
                    # out, every wave publishes it with the send table below.
                    _rawbar()
                    if wave == fx.Int32(0):
                        cnts = []
                        for j in range_constexpr(EPR_CH):
                            e_ = lane + fx.Int32(j * WAVE)
                            c_ = fx.Int32(
                                comm_ops.load_i32_lds(fx.Int64(l_rc + e_ * fx.Int32(4)))
                            )
                            cnts.append(
                                fx.Int32((e_ < fx.Int32(EPR)).select(c_, fx.Int32(0)))
                            )
                        lead_plan_lds(cnts)
                    _rawbar()
                    lead_plan_replicas()
                    sbv = []
                    for j in range_constexpr(SEGS_PAD // 128):
                        for q in range_constexpr(4):
                            sbv.append(
                                fx.Int32(
                                    comm_ops.load_i32_lds(
                                        fx.Int64(
                                            l_sb
                                            + (
                                                fx.Int32(128 * j + q)
                                                + lane * fx.Int32(4)
                                            )
                                            * fx.Int32(4)
                                        )
                                    )
                                )
                            )
                    for c in range_constexpr(N_SREP // 4):
                        rep = wave * fx.Int32(N_SREP // 4) + fx.Int32(c)
                        for j in range_constexpr(SEGS_PAD // 128):
                            buffer_store(
                                fx.Vector.from_elements(
                                    sbv[4 * j : 4 * j + 4], dtype=fx.Int32
                                ),
                                rsrc_ctl,
                                fx.Int32(SREP_W + 128 * j)
                                + rep * fx.Int32(SEGS_PAD)
                                + lane * fx.Int32(4),
                                cache_modifier=SCOPE_DEV,
                            )
                    rocdl.s_wait_storecnt(0)
                    _rawbar()
                    stamp(21)
                    if wave == fx.Int32(0):
                        open_gates(GATE_S, GATE_STRIDE)
                    stamp(22)
                # ---- every CTA but 0: one wave per token. The row is
                # quantized into registers once, then sent to each of its
                # routes' rows once the send gate is open.
                if bid > fx.Int32(0):
                    rep_w = fx.Int32(SREP_W) + (bid & fx.Int32(N_SREP - 1)) * fx.Int32(
                        SEGS_PAD
                    )
                    for t in range(
                        (bid - fx.Int32(1)) * fx.Int32(4) + wave, tok, (G - 1) * 4
                    ):
                        n_mine = n_mine + fx.Int32(K)
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
                            scale_f32 = (fx.Int32(e8m0) << fx.Int32(23)).bitcast(
                                fx.Float32
                            )
                            for c in range_constexpr(4):
                                packed.append(
                                    fx.Int32(
                                        emit_cvt_scalef32_pk8_fp4_bf16(
                                            chunks[c], scale_f32, i32_ty=T.i32
                                        )
                                    )
                                )
                            scales.append(fx.Int32(e8m0) & fx.Int32(0xFF))
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
                        wait_gate(GATE_S, GATE_STRIDE, 1)
                        rk = fx.Int32((lane < fx.Int32(K)).select(lane, fx.Int32(0)))
                        rs = _load_i32_dev(rsrc_route, t * fx.Int32(K) + rk)
                        w_l = fx.Int32(
                            buffer_load(
                                rsrc_wts, t * fx.Int32(K) + rk, vec_width=1, dtype=T.i32
                            )
                        )
                        okr = (lane < fx.Int32(K)) & (rs >= fx.Int32(0))
                        sg = fx.Int32(
                            okr.select(rs & fx.Int32((1 << SEG_BITS) - 1), fx.Int32(0))
                        )
                        sb = _load_i32_dev(rsrc_ctl, rep_w + sg)
                        row_l = sb + (rs >> fx.Int32(SEG_BITS))
                        okr = okr & (row_l < i32_rows)
                        d_l = sg // fx.Int32(EPR)
                        okv = fx.Int32(okr.select(fx.Int32(1), fx.Int32(0)))
                        for k in range_constexpr(K):
                            okk = _uniform(
                                readlane(T.i32, arith.unwrap(okv), k)
                            ) != fx.Int32(0)
                            row = _uniform(readlane(T.i32, arith.unwrap(row_l), k))
                            dst = _uniform(readlane(T.i32, arith.unwrap(d_l), k))
                            w_bits = _uniform(readlane(T.i32, arith.unwrap(w_l), k))
                            if okk:
                                rsrc_row = create_buffer_resource_from_addr(
                                    peer_va(dst, off_disp)
                                    + fx.Int64(row) * fx.Int64(ROW),
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
                                    peer_va(dst, off_rowmap)
                                    + fx.Int64(row) * fx.Int64(8),
                                    num_records_bytes=8,
                                )
                                meta = (
                                    fx.Int32(rank * DEST_STRIDE)
                                    + t * fx.Int32(K)
                                    + fx.Int32(k)
                                )
                                buffer_store(
                                    fx.Vector.from_elements(
                                        [meta, w_bits], dtype=fx.Int32
                                    ),
                                    rsrc_rm,
                                    fx.Int32(0),
                                    mask=lane == fx.Int32(0),
                                    cache_modifier=SCOPE_SYS,
                                )
            else:
                if const_expr(EXPECT):  # noqa: SIM102 - preserve DSL staging
                    # The last wave of the grid (a route of its own only past
                    # G * 4 - 1 routes) counts this rank's routes per
                    # destination and sends each its count, tagged with the
                    # launch's 16-bit tag in one word.
                    if (bid == fx.Int32(G - 1)) & (wave == fx.Int32(3)):
                        n_all = tok * fx.Int32(K)
                        # Sized to this step's routes: the unrolled loads run to
                        # the instance's most routes, and past the ids tensor
                        # they must read zeros, not the next page.
                        rsrc_ids_n = create_buffer_resource_from_addr(
                            addr_ids, num_records_bytes=n_all * fx.Int32(4)
                        )
                        per_dst = [fx.Int32(0)] * npes
                        for base in range_constexpr(
                            0, int(max_tok_per_rank) * K, 4 * WAVE
                        ):
                            # One dword per load: a raw buffer access partly
                            # past num_records reads zeros for all of it.
                            r0 = fx.Int32(base) + lane * fx.Int32(4)
                            for c in range_constexpr(4):
                                gid = fx.Int32(
                                    buffer_load(
                                        rsrc_ids_n,
                                        r0 + fx.Int32(c),
                                        vec_width=1,
                                        dtype=T.i32,
                                    )
                                )
                                ok = (
                                    (r0 + fx.Int32(c) < n_all)
                                    & (gid >= fx.Int32(0))
                                    & (gid < fx.Int32(E if PH & 1 else 0))
                                )
                                d_ = gid // fx.Int32(EPR)
                                for dd in range_constexpr(npes):
                                    per_dst[dd] = per_dst[dd] + _popc_i32(
                                        fx.Int32(
                                            ballot(T.i32, ok & (d_ == fx.Int32(dd)))
                                        )
                                    )
                        mine = per_dst[0]
                        for dd in range_constexpr(1, npes):
                            mine = fx.Int32(
                                (lane == fx.Int32(dd)).select(per_dst[dd], mine)
                            )
                        if lane < fx.Int32(npes):
                            comm_ops.store_i32_global_system_monotonic(
                                peer_va(lane, off_ctl)
                                + fx.Int64(EXP_OFF + rank * FLAG_STRIDE),
                                (tag16 << fx.Int32(16)) | mine,
                            )
                # One wave per route (t, k): the token's quant is redone per route
                # (cheap next to the remote round trip) so the routes go out in
                # parallel. Its lane 0 reserves the destination slot first; the
                # atomic is in flight across the quant.
                n_routes = tok * fx.Int32(K)
                if const_expr(LEAD_FREE):
                    r_first = fx.Int32(
                        (bid == fx.Int32(0)).select(n_routes, wave_gid - fx.Int32(4))
                    )
                else:
                    r_first = wave_gid
                for r in range(r_first, n_routes, PROD_W):
                    n_mine = n_mine + fx.Int32(1)
                    if const_expr(ROWFLAG):  # noqa: SIM102
                        # No row (invalid route, or past its expert's slots).
                        if lane == fx.Int32(0):
                            comm_ops.store_i32_lds(
                                fx.Int64(
                                    ar
                                    + (wave * fx.Int32(R_MAX) + n_mine - fx.Int32(1))
                                    * fx.Int32(8)
                                    + fx.Int32(4)
                                ),
                                fx.Int32(-1),
                            )
                    t = r // fx.Int32(K)
                    gid = fx.Int32(
                        buffer_load(
                            rsrc_ids, r, vec_width=1, dtype=T.i32, is_scalar=True
                        )
                    )
                    w_bits = fx.Int32(
                        buffer_load(
                            rsrc_wts, r, vec_width=1, dtype=T.i32, is_scalar=True
                        )
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
                    # Every peer has started this launch (the row loads above
                    # are local and go out first).
                    if const_expr(HS):
                        wait_gate(GATE_S, GATE_STRIDE, 1)
                    rsrc_run = create_buffer_resource_from_addr(
                        peer_va(dst, off_ctl), num_records_bytes=SRC_OFF
                    )
                    if const_expr(TRACE):
                        t_issue = _now()
                    off_l = _buffer_atomic_add_ret(
                        rsrc_run,
                        fx.Int32(
                            live0.select(
                                (par * fx.Int32(EPR_PAD) + le)
                                * fx.Int32(RUN_STRIDE * 4),
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
                        if const_expr(ROWFLAG):  # noqa: SIM102
                            # Marked once every row of this wave retired, past
                            # the reserved signal (see below the loop).
                            if lane == fx.Int32(0):
                                comm_ops.store_i32_lds(
                                    fx.Int64(
                                        ar
                                        + (wave * fx.Int32(R_MAX) + n_mine - fx.Int32(1))
                                        * fx.Int32(8)
                                    ),
                                    dst,
                                )
                                comm_ops.store_i32_lds(
                                    fx.Int64(
                                        ar
                                        + (wave * fx.Int32(R_MAX) + n_mine - fx.Int32(1))
                                        * fx.Int32(8)
                                        + fx.Int32(4)
                                    ),
                                    row,
                                )
                    if valid & (off >= fx.Int32(CAP)) & (lane == fx.Int32(0)):
                        comm_ops.atomic_add_agent(
                            addr_ctl + fx.Int64(CTL_ERR * 4), fx.Int32(1)
                        )
            stamp(10)
            if const_expr(EARLY and not EXPECT):
                # Every slot atomic of this wave has returned (its rows used
                # the offsets), so the peers' counters hold this wave's routes:
                # the wave completing the count tells every destination its
                # counters from this rank are final, before any row retires.
                res_before = fx.Int32(0)
                if (n_mine > fx.Int32(0)) & (lane == fx.Int32(0)):
                    res_before = fx.Int32(
                        comm_ops.atomic_add_agent(
                            addr_ctl + fx.Int64(CTL_TALLY * 4) + fx.Int64(par) * 128,
                            n_mine,
                        )
                    )
                res_before = _uniform(readlane(T.i32, arith.unwrap(res_before), 0))
                res_last = (n_mine > fx.Int32(0)) & (res_before + n_mine == n_routes)
                res_last = res_last | (
                    (n_routes == fx.Int32(0))
                    & (bid == fx.Int32(0))
                    & (wave == fx.Int32(0))
                )
                if res_last & (lane < fx.Int32(npes)):
                    comm_ops.store_i32_global_system_monotonic(
                        peer_va(lane, off_ctl) + fx.Int64(RS_OFF + rank * FLAG_STRIDE),
                        gen,
                    )
            # Publish once this rank's last token has every route in place:
            # the stores above retire (SCOPE_SYS) before the count, and the
            # wave completing the count stores gen into every destination.
            # ROWFLAG: each route marked its own slot instead.
            rocdl.s_wait_storecnt(0)
            stamp(11)
            if const_expr(ROWFLAG):
                # Every row and rowmap pair of this wave retired: mark their
                # slots landed.
                rocdl.s_wait_dscnt(0)
                for j in range(fx.Int32(0), n_mine, 1):
                    rec = ar + (wave * fx.Int32(R_MAX) + j) * fx.Int32(8)
                    dst_j = _uniform(fx.Int32(comm_ops.load_i32_lds(fx.Int64(rec))))
                    row_j = _uniform(
                        fx.Int32(comm_ops.load_i32_lds(fx.Int64(rec + fx.Int32(4))))
                    )
                    if (row_j >= fx.Int32(0)) & (lane == fx.Int32(0)):
                        comm_ops.store_i32_global_system_monotonic(
                            peer_va(dst_j, off_rowmap)
                            + fx.Int64(LAND_OFF)
                            + fx.Int64(row_j) * fx.Int64(4),
                            gen,
                        )
            if const_expr(not ROWFLAG):
                done_before = fx.Int32(0)
                if (n_mine > fx.Int32(0)) & (lane == fx.Int32(0)):
                    done_before = fx.Int32(
                        comm_ops.atomic_add_agent(
                            addr_ctl + fx.Int64(CTL_PROD * 4) + fx.Int64(par) * 128,
                            n_mine,
                        )
                    )
                done_before = _uniform(readlane(T.i32, arith.unwrap(done_before), 0))
                last = (n_mine > fx.Int32(0)) & (done_before + n_mine == n_routes)
                last = last | (
                    (n_routes == fx.Int32(0))
                    & (bid == fx.Int32(0))
                    & (wave == fx.Int32(0))
                )
                if last & (lane < fx.Int32(npes)):
                    comm_ops.store_i32_global_system_monotonic(
                        peer_va(lane, off_ctl) + fx.Int64(SRC_OFF + rank * FLAG_STRIDE),
                        gen,
                    )

            stamp(2)
            if const_expr(PREF_B):  # noqa: SIM102 - preserve DSL staging
                # While the plan is pending HBM is idle: read the W1 slice this
                # CTA's first item most likely needs (m-tile bid // NT1 is
                # expert bid // NT1 when every expert before it has rows), so
                # that item finds it in cache. Real loads, PREF_DEPTH x 16 B in
                # flight per lane, folded into one word that is stored only on
                # an impossible value (global_prefetch hints measured no gain).
                e_p = bid // fx.Int32(NT1)
                n_p = bid - e_p * fx.Int32(NT1)
                if (
                    (bid != fx.Int32(0))
                    & (tok >= fx.Int32(PREF_MIN_TOK))
                    & (e_p < fx.Int32(EPR))
                ):
                    w1_base = fx.Int64(fx.ptrtoint(fx.recast_iter(fx.Int8, arg_w1)))
                    rsrc_w1p = create_buffer_resource_from_addr(
                        w1_base
                        + fx.Int64(e_p * fx.Int32(N1) + n_p * fx.Int32(tile_n))
                        * fx.Int64(H // 2)
                    )
                    acc_p = fx.Int32(0)
                    for p_off in range(
                        tid * fx.Int32(16), PREF_B, 4 * WAVE * 16 * PREF_DEPTH
                    ):
                        vs = [
                            fx.Vector(
                                buffer_load(
                                    rsrc_w1p,
                                    (p_off + fx.Int32(q * 4 * WAVE * 16)) // fx.Int32(4),
                                    vec_width=4,
                                    dtype=T.i32,
                                )
                            )
                            for q in range_constexpr(PREF_DEPTH)
                        ]
                        for v4 in vs:
                            acc_p = acc_p ^ fx.Int32(v4[0])
                    if acc_p == i32_rows + fx.Int32(0x5A5A5A5A):
                        comm_ops.atomic_add_agent(
                            addr_ctl + fx.Int64(CTL_ERR * 4), fx.Int32(0)
                        )

            # ======================== 2. PLAN ========================
            # CTA 0 alone polls the peers' src_done (hundreds of CTAs polling
            # the same uncached lines starve every access queued behind them).
            # small_m: it then lays out the plan, publishes it to the local
            # control block and opens each CTA's own gate line. large_m laid
            # the plan out from X1 and published it with the send gates: the
            # gate opened here only tells GEMM1's A owners the rows landed.
            if bid == fx.Int32(0):
                if const_expr(COMPACT):
                    if wave == fx.Int32(0):
                        poll_peer_flags(SRC_OFF, 2)
                        stamp(14)
                        open_gates(GATE_P, GATE_STRIDE)
                        stamp(17)
                else:
                    if wave == fx.Int32(0):
                        # EARLY: the counters are final once every source
                        # reserved; the rows-landed gate opens further down.
                        if const_expr(EXPECT):
                            cnts = poll_expected(2)
                            stamp(14)
                        else:
                            poll_peer_flags(RS_OFF if EARLY else SRC_OFF, 2)
                            stamp(14)
                            cnts = []
                            for j in range_constexpr(EPR_CH):
                                e_ = lane + fx.Int32(j * WAVE)
                                c_ = _load_i32_sys(
                                    rsrc_actl,
                                    (par * fx.Int32(EPR_PAD) + e_)
                                    * fx.Int32(RUN_STRIDE),
                                    volatile=False,
                                )
                                c_ = fx.Int32(
                                    (c_ > fx.Int32(CAP)).select(fx.Int32(CAP), c_)
                                )
                                c_ = fx.Int32(
                                    (e_ < fx.Int32(EPR)).select(c_, fx.Int32(0))
                                )
                                cnts.append(c_)
                        lead_plan_lds(cnts)
                    rocdl.s_wait_dscnt(0)
                    rocdl.s_barrier_signal(-1)
                    rocdl.s_barrier_wait(-1)
                    stamp(15)
                    # Every wave of CTA 0 writes a quarter of the replicas; wave
                    # 0 then opens the gates (tagged replicas need none).
                    lead_plan_replicas()
                    if const_expr(not PLAN_TAG):
                        rocdl.s_wait_storecnt(0)
                    rocdl.s_barrier_signal(-1)
                    rocdl.s_barrier_wait(-1)
                    stamp(16)
                    if const_expr(not PLAN_TAG):  # noqa: SIM102 - preserve DSL staging
                        if wave == fx.Int32(0):
                            open_gates(GATE_P, GATE_STRIDE)
                    stamp(17)
                    if const_expr(EARLY and not ROWFLAG):  # noqa: SIM102
                        if wave == fx.Int32(0):
                            poll_peer_flags(SRC_OFF, 2)
                            stamp(TR_LANDED)
                            open_gates(GATE_R, GATE_STRIDE)
            else:
                if wave == fx.Int32(0):
                    follower_plan(GATE_S if COMPACT else GATE_P, 3)
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
                    end_j = (
                        fx.Int32(
                            comm_ops.load_i32_lds(fx.Int64(lds_end + e_j * fx.Int32(4)))
                        )
                        >> fx.Int32(16)
                    ) & fx.Int32(0xFFFF)
                    e_ = e_ + _popc_i32(
                        fx.Int32(ballot(T.i32, (e_j < fx.Int32(EPR)) & (end_j <= i)))
                    )
                e_ = _uniform(e_)
                e_prev = fx.Int32((e_ > fx.Int32(0)).select(e_ - fx.Int32(1), e_))
                prev = (
                    fx.Int32(
                        comm_ops.load_i32_lds(fx.Int64(lds_end + e_prev * fx.Int32(4)))
                    )
                    >> fx.Int32(16)
                ) & fx.Int32(0xFFFF)
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
                # Compact: experts start on a tile, so plan m-tile i is GEMM
                # m-tile i; slots: expert e's tiles start at e * MT_PER_E.
                gidx = i if COMPACT else e_ * fx.Int32(MT_PER_E) + jt
                return e_, gidx, rows

            def set_meta1(meta, ready_fn=None):
                meta1_holder[0] = meta
                ready1_holder[0] = ready_fn

            def set_meta2(meta, ready_fn):
                meta2_holder[0] = meta
                ready2_holder[0] = ready_fn

            def set_la(holder, d):
                holder[0] = d

            # Lookahead: an item's drain issues the next item's first k-tiles
            # when it is of the same GEMM. The claim made at the item's start
            # names it; BC_LA hands the sides issued (bit 0 B, bit 1 A) to it.
            la_w = fx.Int64(bc + fx.Int32(BC_LA))

            def la_publish(claim):
                if is_t0:
                    comm_ops.store_i32_lds(bc_next, fx.Int32(G) + claim)
                rocdl.s_wait_dscnt(0)

            def la_store(b, a):
                if is_t0:
                    comm_ops.store_i32_lds(la_w, b + a * fx.Int32(2))

            def la_next(gemm1):
                """The next item's tile when it is of this item's GEMM (b=1)."""
                u_n = _uniform(fx.Int32(comm_ops.load_i32_lds(bc_next)))
                if const_expr(gemm1):
                    same = u_n < U1
                    v_n = u_n
                    nt = fx.Int32(NT1)
                else:
                    same = u_n < U
                    v_n = u_n - U1
                    nt = fx.Int32(NT2)
                i_raw = v_n // nt
                i_n = fx.Int32(same.select(i_raw, fx.Int32(0)))
                n_n = fx.Int32(same.select(v_n - i_raw * nt, fx.Int32(0)))
                e_n, mt_n, rows_n = decode_mtile(i_n)
                b = fx.Int32(same.select(fx.Int32(1), fx.Int32(0)))
                return b, i_n, {
                    "b": b,
                    "m_tile": mt_n,
                    "n_tile": n_n,
                    "expert": e_n,
                    "rows": rows_n,
                }

            # The A-side verdicts: an i32, or a function the builder calls past
            # a draining fence (a flag load issued here, compared there). Each
            # stores the sides for the next item once known. Branches only
            # return: the AST rewriter turns a branch's assignments into
            # branch results.
            def la_deferred(b, ready):
                def a_fn(b=b, ready=ready):
                    a = b * fx.Int32(ready().select(fx.Int32(1), fx.Int32(0)))
                    la_store(b, a)
                    return a

                return a_fn

            def la_now(b, a):
                la_store(b, a)
                return a

            def la_a1(b, d):
                if const_expr(not LA_A):
                    return la_now(b, fx.Int32(0))
                if const_expr(ROWFLAG):
                    pending = rows_pending(d["m_tile"], d["rows"])
                    return la_deferred(b, lambda p=pending: p() == fx.Int32(0))
                if const_expr(COMPACT or EARLY):
                    # A runs ahead once this CTA saw the rows-landed gate,
                    # which this GEMM1 item's own a_ready waited for.
                    seen = _uniform(
                        fx.Int32(
                            comm_ops.load_i32_lds(fx.Int64(bc + fx.Int32(BC_GATE_R)))
                        )
                    )
                    return la_now(
                        b,
                        b
                        * fx.Int32(
                            (seen != fx.Int32(0)).select(fx.Int32(1), fx.Int32(0))
                        ),
                    )
                return la_now(b, b)

            def la_a2(b, i_n):
                if const_expr(not LA_A):
                    return la_now(b, fx.Int32(0))
                if const_expr(PH & 16):
                    return la_now(b, b)
                rdy = _load_i32_dev(
                    rsrc_ctl, fx.Int32(CTL_G1 + G1_READY) + i_n * fx.Int32(G1_STRIDE)
                )
                return la_deferred(b, lambda r=rdy: _uniform(r) == gen)

            def la_plan1():
                b, _, d = la_next(True)
                d["a"] = la_a1(b, d)
                return d

            def la_plan2():
                b, i_n, d = la_next(False)
                d["a"] = la_a2(b, i_n)
                return d

            # AITER_DMK_TRACE: per-item phase sums in LDS words past the
            # broadcast ones: [0] the last mark, [1 + k] TR_IT slot k.
            it_lds = bc + fx.Int32(64)

            def it_word(k):
                return fx.Int64(it_lds + fx.Int32(4) + k * fx.Int32(4))

            def it_mark():
                if is_t0:
                    comm_ops.store_i32_lds(fx.Int64(it_lds), _i32_lo(_now()))

            def it_acc(k):
                """Add the time since the last mark to slot ``k``; re-mark."""
                if is_t0:
                    now = _i32_lo(_now())
                    prev = fx.Int32(comm_ops.load_i32_lds(fx.Int64(it_lds)))
                    old = fx.Int32(comm_ops.load_i32_lds(it_word(fx.Int32(k))))
                    comm_ops.store_i32_lds(it_word(fx.Int32(k)), old + now - prev)
                    comm_ops.store_i32_lds(fx.Int64(it_lds), now)

            def it_clear():
                if is_t0:
                    for k in range_constexpr(10):
                        comm_ops.store_i32_lds(it_word(fx.Int32(k)), fx.Int32(0))

            if const_expr(TRACE):
                for holder, k0 in ((hooks1_holder, 0), (hooks2_holder, 5)):
                    holder[0] = {
                        "first_ready": lambda k0=k0: it_acc(k0),
                        "epi_start": lambda k0=k0: it_acc(k0 + 1),
                        "epi_end": lambda k0=k0: it_acc(k0 + 2),
                    }
                it_clear()

            n1 = fx.Int32(N1)
            n2 = fx.Int32(H)
            rows_cap = i32_rows

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
            n_items = fx.Int32(0)
            while u < U:
                n_items = n_items + fx.Int32(1)
                stamp(TR_LAST_ITEM)
                if const_expr(TRACE):
                    it_mark()
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
                if const_expr(LA1 or LA2):
                    # Written by the previous item's plan, read before this
                    # item's plan rewrites it (past the loop's barrier).
                    la_pre = _uniform(fx.Int32(comm_ops.load_i32_lds(la_w)))
                if u < U1:
                    i = u // fx.Int32(NT1)
                    n = u - i * fx.Int32(NT1)
                    e_, mt_idx, rows = decode_mtile(i)
                    if const_expr(LA1):

                        def la_pub1(claim=claim):
                            la_publish(claim)

                        set_la(
                            la1_holder,
                            {"pre": la_pre, "publish": la_pub1, "plan": la_plan1},
                        )
                    if const_expr(COMPACT or EARLY):

                        def a_ready1(mt_idx=mt_idx, rows=rows):
                            # The rows have landed (ROWFLAG: this m-tile's own
                            # rows; else CTA 0 saw every src_done): only A
                            # owners wait, the W1 prologue streams meanwhile.
                            if const_expr(ROWFLAG):
                                wait_rows(mt_idx, rows, 7)
                            else:
                                wait_landed()

                        set_meta1((e_, rows), a_ready1)
                    else:
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
                    if const_expr(TRACE):  # noqa: SIM102 - preserve DSL staging
                        if n2_done == fx.Int32(0):
                            stamp(TR_FIRST_G2)
                    n2_done = n2_done + fx.Int32(1)
                    v = u - U1
                    i = v // fx.Int32(NT2)
                    n = v - i * fx.Int32(NT2)
                    e_, mt_idx, rows = decode_mtile(i)
                    if const_expr(COMPACT or EARLY):
                        # GEMM2's epilogue loads its rows' rowmap pairs before
                        # a_ready, and large_m / EARLY start the queue before
                        # the peers' rows land: every wave waits for them
                        # first.
                        if const_expr(ROWFLAG):
                            wait_rows(mt_idx, rows, 7)
                        else:
                            wait_landed()

                    def a_ready(i=i):
                        if const_expr(PH & 16):
                            return
                        stamp(12)
                        rdy_w = fx.Int32(CTL_G1 + G1_READY) + i * fx.Int32(G1_STRIDE)
                        cur = _uniform(_load_i32_dev(rsrc_ctl, rdy_w))
                        t_end = spin_end()
                        live = fx.Int32(1)
                        while (cur != gen) & (live != fx.Int32(0)):
                            fx.rocdl.s_sleep(POLL_SL)
                            cur = _uniform(_load_i32_dev(rsrc_ctl, rdy_w))
                            live = alive(t_end)
                        if cur != gen:
                            spin_fail(4)
                        stamp(13)

                    set_meta2((e_, rows), a_ready)
                    if const_expr(LA2):

                        def la_pub2(claim=claim):
                            la_publish(claim)

                        set_la(
                            la2_holder,
                            {"pre": la_pre, "publish": la_pub2, "plan": la_plan2},
                        )
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
                if const_expr(TRACE):
                    k_it = (u < U1).select(fx.Int32(0), fx.Int32(5))
                    it_acc(k_it + fx.Int32(3))
                    if is_t0:
                        c_old = fx.Int32(
                            comm_ops.load_i32_lds(it_word(k_it + fx.Int32(4)))
                        )
                        comm_ops.store_i32_lds(
                            it_word(k_it + fx.Int32(4)), c_old + fx.Int32(1)
                        )
                    if n_items == fx.Int32(1):
                        it_clear()
                u = _uniform(comm_ops.load_i32_lds(bc_next))
            if const_expr(DEFER_SC):
                # The last GEMM2 item's scatter: every wave retires its own
                # before the barrier that precedes the close.
                tdm_ops.tensor_wait(0)
                rocdl.s_barrier_signal(-1)
                rocdl.s_barrier_wait(-1)
            stamp(4)
            stamp(TR_ITEMS, fx.Int64(n_items))
            if const_expr(TRACE):
                for k in range_constexpr(10):
                    stamp(
                        TR_IT + k,
                        fx.Int64(
                            fx.Int32(comm_ops.load_i32_lds(it_word(fx.Int32(k))))
                        ),
                    )

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
                poll_peer_flags(CF_OFF, 5)
                stamp(18)
                open_gates(GATE_C, GATE_STRIDE)
                stamp(19)
            if wave == fx.Int32(0):
                wait_gate(GATE_C, GATE_STRIDE, 6)
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
            i32_rows,
            addr_route,
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
#: Past this many the 64-row tile wins (V4.1-Flash DEP4, 97 experts a rank:
#: tpr 512 172 vs 189 us/layer; tpr 256 158 vs 147).
DMK_TILE_M64_TOKENS = 256
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
    if int(max_tokens_per_rank) > DMK_TILE_M64_TOKENS:
        tile["tile_m"] = 64
    elif int(max_tokens_per_rank) > DMK_TILE_M32_TOKENS:
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
    if read_decode_mega_tokens() <= 0:
        return "AITER_MEGA_DECODE is off"
    return _mega_reason(config, dmk_tile(dmk_small_tokens(config)))


def read_large_mega() -> bool:
    """``AITER_MEGA_LARGE=1``: steps the decode megakernel does not serve run
    the compact-layout megakernel (large_m) instead of the staged path."""
    return os.environ.get("AITER_MEGA_LARGE", "").strip() in ("1", "true", "True")


def large_mega_reason(config) -> str | None:
    """Why this MegaMoE cannot run the large_m megakernel, or None if it can."""
    if not read_large_mega():
        return "AITER_MEGA_LARGE is off"
    geo = config.geometry
    if config.world_size * geo.max_tokens_per_rank > 0xFFFF:
        return "the compact plan holds an expert's rows in 16 bits"
    if geo.max_tokens_per_rank * config.topk > TALLY_MAX * TALLY_MAXR:
        return "more routes per rank than the tally CTAs count"
    npes, epr = config.world_size, config.experts_per_rank
    segs = dmk_segs_pad(npes=npes, experts_per_rank=epr)
    if segs > (1 << SEG_BITS):
        return "too many (rank, expert) segments for a route slot"
    if npes * segs * 4 > dmk_src_off(epr):
        return "the X1 rows do not fit the control region"
    if lm_scratch_bytes(npes=npes, experts_per_rank=epr) > LM_SCRATCH_MAX:
        return "the compact front end's LDS does not fit the GEMM arena"
    return _mega_reason(config, lm_tile())


def _mega_reason(config, t) -> str | None:
    from aiter import ActivationType, QuantType

    from .recipe import CombineKind, DispatchKind

    recipe, geo = config.recipe, config.geometry
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


#: large_m GEMM tiles: 64x256x128 with 3 buffers for both GEMMs, the staged
#: GEMMs' tile. Its ~70 KB arena lets 4 CTAs share a CU (LM_CTAS_PER_CU): the
#: weight-bound GEMMs need that occupancy. At 2 CTAs per CU with the 64x256x256
#: tile (134 KB) the GEMM phase ran ~85-115 us slower at 2048 tokens/rank.
#: AITER_LM_TILE="tm,tn,tk,nb,tn2,tk2,nb2" and AITER_LM_CTAS_PER_CU override.
LM_TILE = dict(  # noqa: C408 - the builder's keyword names
    tile_m=64,
    tile_n=256,
    tile_k=128,
    num_buffers=3,
    tile_n2=256,
    tile_k2=128,
    num_buffers2=3,
)
LM_CTAS_PER_CU = 4


def lm_ctas_per_cu() -> int:
    return int(os.environ.get("AITER_LM_CTAS_PER_CU", LM_CTAS_PER_CU))


def lm_tile() -> dict:
    v = os.environ.get("AITER_LM_TILE")
    if v:
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
                (int(x) for x in v.split(",")),
            )
        )
    return dict(LM_TILE)


def lm_caps(config, recv_bound: int) -> int:
    """Compact rows a step with this receive bound may fill: the staged
    plan's capacity at large_m's alignment."""
    from .compact_plan import compact_row_capacity

    return compact_row_capacity(
        max_recv=int(recv_bound),
        topk=config.topk,
        experts_per_rank=config.experts_per_rank,
        tile_m=lm_tile()["tile_m"],
    )


def large_mega_regions(config) -> dict:
    """The arena regions large_m needs ({} when it is off): its control
    region, and the staged compact rows / rowmap at least as large as its
    alignment needs (merged with the staged sizes)."""
    if large_mega_reason(config) is not None:
        return {}
    caps = lm_caps(config, config.max_recv)
    rows, rowmap = lm_row_regions()
    return {
        rows: caps * dmk_wire_row(config),
        rowmap: (caps + 1) * 8,
        "lm_ctl": dmk_ctl_arena_bytes(
            npes=config.world_size, experts_per_rank=config.experts_per_rank
        ),
    }


def lm_row_regions() -> tuple:
    """large_m's wire rows and rowmap: the staged path's, or (debug,
    AITER_LM_OWN_ROWS=1) regions of its own."""
    if os.environ.get("AITER_LM_OWN_ROWS", "0") == "1":
        return "lm_disp", "lm_rowmap"
    return "disp_out", "ep_rowmap"


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
        # Rowmap pairs, then one rows-landed word per slot (AITER_DMK_ROWFLAG).
        "dmk_rowmap": epr * cap * 12,
        "dmk_ctl": dmk_ctl_arena_bytes(npes=config.world_size, experts_per_rank=epr),
    }


class DecodeMega:
    """A single-kernel MegaMoE bound to one MegaMoE's arena and combine:
    ``layout="slots"`` is small_m (fixed expert slots, AITER_MEGA_DECODE),
    ``layout="compact"`` is large_m (the staged compact layout, AITER_MEGA_LARGE).
    Each has its own control block, so they share no protocol state."""

    def __init__(self, config, res, device, layout="slots"):
        import torch

        from aiter.jit.utils.chip_info import get_cu_num

        from ..tensor_shim import _run_compiled  # noqa: F401  (import check)

        self.config = config
        self.res = res
        self.layout = layout
        compact = layout == "compact"
        geo = config.geometry
        arena = res.arena
        epr = config.experts_per_rank
        self.row = dmk_wire_row(config)
        if compact:
            t = self.tile = lm_tile()
            rows = lm_caps(config, config.max_recv)
            self.cap = rows
            self.mt_cap = rows // t["tile_m"]
            # Steps above what small_m serves (all of them without it).
            self.recv_lo = (
                dmk_recv_cap(config) if decode_mega_reason(config) is None else 0
            )
            self.regions = (*lm_row_regions(), "lm_ctl")
            self.route = torch.empty(
                geo.max_tokens_per_rank * config.topk, dtype=torch.int32, device=device
            )
        else:
            cap = self.cap = dmk_cap(config)
            t = self.tile = dmk_tile(dmk_small_tokens(config))
            self.recv_cap = dmk_recv_cap(config)
            rows = epr * cap
            self.mt_cap = epr * (cap // t["tile_m"])
            self.regions = ("dmk_disp", "dmk_rowmap", "dmk_ctl")
            self.route = None
        self.rows = rows
        from aiter.ops.flydsl.grouped_moe_gfx1250 import get_wmma_m_rep

        rep2 = get_wmma_m_rep(t["tile_m"], t["tile_n2"], 1, 4, "gemm2")
        inter = geo.inter_dim
        # Zeroed once: the control region's counters, parity words and
        # generation flags. Done before construction returns: the peers' first
        # writes may land as soon as every rank is past its construction barrier.
        arena.zero(self.regions[2])
        if not compact:
            # The rows-landed words hold a launch's generation, never 0.
            arena.zero(self.regions[1])
        torch.cuda.synchronize(device)
        self.a2 = torch.empty((rows, inter // 2), dtype=torch.uint8, device=device)
        self.a2_scale = torch.empty(
            (rows // rep2, inter // 32 * rep2), dtype=torch.uint8, device=device
        )
        self.ctas_per_cu = lm_ctas_per_cu() if compact else DMK_CTAS_PER_CU
        self.workers = int(get_cu_num()) * self.ctas_per_cu
        # Per-call switch for in-process A/B against the staged path. All
        # ranks must flip it together (the paths share no protocol state).
        self.enabled = True
        self._ctl_words = dmk_ctl_words(
            experts_per_rank=epr,
            mt_cap=self.mt_cap,
            workers=self.workers,
            npes=config.world_size,
            compact=compact,
        )
        self.ctl = torch.zeros(
            self._ctl_words + 2 * TR_N * self.workers + 8 * RT_N,
            dtype=torch.int32,
            device=device,
        )
        self._traced = 0
        self._launch = None
        self._act_key = None
        self._lined_up = None

    def serves(self, recv_token_bound) -> bool:
        """Whether a step with this receive bound runs the megakernel: the
        same answer on every rank, since every rank passes the same bound."""
        if not self.enabled:
            return False
        bound = self._bound(recv_token_bound)
        if self.layout == "compact":
            return self.recv_lo < bound
        return bound <= self.recv_cap

    def _bound(self, recv_token_bound) -> int:
        return (
            self.config.max_recv if recv_token_bound is None else int(recv_token_bound)
        )

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
                off_disp=arena.offset(self.regions[0]),
                off_rowmap=arena.offset(self.regions[1]),
                off_ctl=arena.offset(self.regions[2]),
                off_comb_inp=arena.offset("comb_inp"),
                comb_slot_stride=config.combine_slot_stride_for(0),
                arena_handle=arena.handle,
                stage1_act=stage1_act,
                workers=self.workers,
                ctas_per_cu=self.ctas_per_cu,
                layout=self.layout,
                mt_cap=self.mt_cap,
                **self.tile,
            )
            self._act_key = stage1_act
        return self._launch

    def forward(
        self,
        hidden_states,
        topk_weights,
        topk_ids,
        *,
        w1,
        w2,
        w1_scale,
        w2_scale,
        recv_token_bound=None,
    ):
        import flydsl.compiler as flyc
        import flydsl.expr as fx
        import torch

        from aiter import ActivationType
        from aiter.ops.flydsl.grouped_moe_gfx1250 import _grouped_weight_uint8

        from ..tensor_shim import _preload_compiled, _run_compiled

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
        disp = arena.local_ptr(self.regions[0])
        if self.layout == "compact":
            # The GEMMs see the rows this step's bound can fill; senders and
            # receiver drop anything past it alike.
            rows = min(lm_caps(self.config, self._bound(recv_token_bound)), self.rows)
            route = self.route.data_ptr()
        else:
            rows, route = self.rows, 0
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
        launch = self._launcher(stage1_act)
        args = (
            ptr(self.a2.data_ptr()),
            ptr(disp),
            ptr(disp + self.config.hidden_dim // 2),
            ptr(w1_u8.data_ptr()),
            ptr(w1_scale.data_ptr()),
            ptr(self.a2_scale.data_ptr()),
            ptr(w2_u8.data_ptr()),
            ptr(w2_scale.data_ptr()),
            ptr(arena.local_ptr(self.regions[1])),
            arena.handle,
            hidden_states.data_ptr(),
            topk_ids.data_ptr(),
            topk_weights.data_ptr(),
            out.data_ptr(),
            self.ctl.data_ptr(),
            tokens,
            rows,
            route,
            sl,
            fx.Stream(torch.cuda.current_stream()),
        )
        if launch is not self._lined_up:
            # Every rank compiles its own kernel (the rank is baked in), seconds
            # apart: line the ranks up before the first launch, so its waits do
            # not start long before a peer's kernel does.
            _preload_compiled(launch, *args)
            arena._communicator.barrier()
            self._lined_up = launch
        _run_compiled(launch, *args)
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
        if self.layout == "compact":
            x1 = rel[:, 23][tr[:, 23] >= t0]
            cols.append(
                f"x1:tally={x1.max() if x1.numel() else -1:.1f},"
                + ",".join(
                    f"{n}={rel[0, 20 + k]:.1f}"
                    for k, n in enumerate(("x1_seen", "srep", "gates_s"))
                )
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

    def fails(self) -> list:
        """AITER_DMK_NOTRAP: timed-out waits per site (X1, send gate, source
        flags, plan gate, g1_ready, combine flags, reduce gate, rows gate,
        started flags)."""
        return self.ctl[CTL_FAIL : CTL_FAIL + 9].tolist()


__all__ = [
    "DecodeMega",
    "decode_mega_reason",
    "decode_mega_regions",
    "dmk_ctl_arena_bytes",
    "dmk_ctl_words",
    "dmk_epr_pad",
    "large_mega_reason",
    "large_mega_regions",
    "make_decode_mega",
    "read_decode_mega_tokens",
]
