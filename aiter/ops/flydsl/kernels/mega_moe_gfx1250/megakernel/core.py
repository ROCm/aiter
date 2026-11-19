# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The megakernel's core, shared by both layouts: the control block, the
entry ticket, the waits and gates, the plan, the GEMM queue, the close and
the reduce (phases 2. to 5. of the package docstring)."""

from __future__ import annotations

import flydsl.expr as fx
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.compiler.ast_rewriter import ASTRewriter
from flydsl.expr import arith, const_expr, range_constexpr, rocdl, tdm_ops
from flydsl.expr.rocdl import ballot, ds_bpermute, readlane
from flydsl.expr.typing import T

from aiter.ops.flydsl.kernels import communication_ops_utils as comm_ops
from aiter.ops.flydsl.kernels.buffer_ops import (
    buffer_load,
    buffer_store,
    create_buffer_resource_from_addr,
)

from ..compact_plan import _wave32_inclusive_scan_i32
from ..config import _WAVE_SIZE as WAVE
from ..s1f_stage1 import (
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
#: A replica is the experts' row counts: 16 bits each for slots, whose slots
#: cap an expert at 65535 rows; 32 for compact, whose experts may receive more.
PLAN_REPS = 128

#: Words between two experts' slot counters: each on its own 128 B line, so
#: the peers' atomics on different experts do not serialize on one line.
RUN_STRIDE = 32

#: Arena control region (``slots_ctl``), bytes. ``running[2][EPR_PAD]`` takes the
#: peers' slot atomics; ``src_done`` / ``comb_flag`` one 256 B line per source,
#: each array room for the most ranks the kernels admit (one wave of them).
FLAG_STRIDE = 256
FLAG_SPAN = WAVE * FLAG_STRIDE

#: Compact layout: routes one tally CTA counts at most (its LDS route
#: table), the most tally CTAs, the words between two segments' global
#: counters, the send-table replicas, and the bits of a route slot's segment
#: field.
TALLY_MAXR = 2048
TALLY_MAX = 256
#: Routes a tally CTA aims to count (AITER_LM_TALLY_CHUNK overrides): its loops
#: wait out a global load per pass, so fewer routes per CTA end the tally
#: sooner; more CTAs put more base atomics on a hot segment's line.
#: V4.1-Flash DEP4: X1 goes out 5-6.5 us sooner than at 2048.
TALLY_CHUNK = 128
SEG_STRIDE = 32
N_SREP = 16
SEG_BITS = 10
#: LDS the compact front end may take out of the GEMM arena on the host's
#: admission check (the kernel checks against the arena it actually builds;
#: compact's default GEMM arena is ~70 KB).
COMPACT_SCRATCH_MAX = 48 << 10


def compact_scratch_bytes(*, npes: int, experts_per_rank: int) -> int:
    """LDS the compact front end uses in the GEMM arena: CTA 0's X1 matrix,
    send table and receive counts, or a tally CTA's histogram, bases and
    route table, whichever is larger."""
    segs = segs_pad(npes=npes, experts_per_rank=experts_per_rank)
    lead = (int(npes) + 1) * segs * 4 + epr_pad(experts_per_rank) * 4
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
TR_N = 28
TR_ITEMS, TR_LAST_ITEM, TR_FIRST_G2, TR_LANDED = 24, 25, 26, 27
RT_N = 1024

#: W1 prefetch (AITER_DMK_PREFETCH_KB): the fewest tokens of a step that runs
#: it, and the 16 B loads each lane keeps in flight (deeper ran slower).
PREF_MIN_TOK = 32
PREF_DEPTH = 8

#: Elements a reduce wave owns per item: 32 lanes x 16.
RED_ELEMS = WAVE * 16


def vdst_fence(v):
    """``v`` once every VALU write has landed. The fp4 converts are inline
    asm, so no VALU -> VMEM wait guards a store that reads their results;
    without one the store can send a stale dword."""
    return fx.Vector(
        _llvm.inline_asm(
            v.type,
            [v.ir_value()],
            "s_wait_alu depctr_va_vdst(0)",
            "=v,0",
            has_side_effects=True,
        )
    )


def ctl_arena_bytes(*, npes: int, experts_per_rank: int) -> int:
    """Bytes of the arena's ``slots_ctl`` region: slot counters, then src_done,
    comb_flag (``FLAG_SPAN`` apart), started and reserved (one line per
    source each)."""
    return src_off(experts_per_rank) + 2 * FLAG_SPAN + 2 * npes * FLAG_STRIDE


def epr_pad(experts_per_rank: int) -> int:
    """Experts rounded up to whole wave chunks, an even number of them past
    one (the plan packs two 16-bit counts per dword across chunk pairs)."""
    ch = (int(experts_per_rank) + WAVE - 1) // WAVE
    if ch > 1 and ch % 2:
        ch += 1
    return ch * WAVE


def src_off(experts_per_rank: int) -> int:
    """The peer flags start 64 KB past the slot counters: CTA 0 polls them
    while the peers' slot atomics land on the counters."""
    run_bytes = 2 * epr_pad(experts_per_rank) * RUN_STRIDE * 4
    return (run_bytes + (64 << 10) - 1) // (64 << 10) * (64 << 10)


def segs_pad(*, npes: int, experts_per_rank: int) -> int:
    """Compact layout: (destination, local expert) segments, padded to 128 so
    one wave covers a row of them with a b128 per lane."""
    return (int(npes) * int(experts_per_rank) + 127) // 128 * 128


def ctl_layout(
    *, experts_per_rank: int, mt_cap: int, workers: int, npes: int = 0, compact=False
):
    """Word offsets of the local control block: the per-m-tile GEMM1 lines,
    the plan replicas (per expert ``rows``, 16 or 32 bits each), one plan, one
    combine and one send gate line per CTA, and on the compact layout the
    segment counters and the send-table replicas. No line is read or
    polled by more than a few CTAs: hundreds of CTAs on one uncached line
    serialize at its DRAM bank."""
    pad = epr_pad(experts_per_rank)
    plan = CTL_G1 + int(mt_cap) * G1_STRIDE
    # A replica holds WAVE packed-count dwords per chunk pair (one pair at
    # least); the tagged replicas store each as a (counts, generation) i64.
    plan_slot = 2 * WAVE * max(1, pad // WAVE // 2)
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
        segs = segs_pad(npes=npes, experts_per_rank=experts_per_rank)
        lay["seg_cnt"] = end
        lay["srep"] = end + segs * SEG_STRIDE
        end = lay["srep"] + N_SREP * segs
    lay["words"] = end
    return lay


def ctl_words(**kw) -> int:
    """i32 words of the local control block."""
    return ctl_layout(**kw)["words"]


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


def _popc_i32(x):
    return fx.Int32(
        _llvm.call_intrinsic(
            T.i32, "llvm.ctpop.i32", [arith.unwrap(fx.Int32(x))], [], []
        )
    )


class KCtx:
    """The megakernel's build constants and, while its body is traced, the
    per-thread values and protocol helpers that the core and the front ends
    share. They read its fields into locals first: a method of it called
    inside a traced ``if`` or loop would have the DSL carry the whole object
    through that region."""

    def __init__(self, base=None, **fields):
        if base is not None:
            self.__dict__.update(base.__dict__)
        self.__dict__.update(fields)


def device_fn(f):
    """Trace ``f`` the way @flyc.kernel traces the kernel body and the
    functions nested in it: its Python ``if`` / ``for`` / ``while`` on traced
    values become scf regions (const_expr ones stay Python)."""
    return ASTRewriter.transform(f)


@device_fn
def bind_base(kc):
    """kc.peer_va / stamp / spin_end / alive / spin_fail."""
    NOTRAP, TIMEOUT_TICKS, TRACE = kc.NOTRAP, kc.TIMEOUT_TICKS, kc.TRACE
    TR_BASE, addr_ctl, bid, is_t0 = kc.TR_BASE, kc.addr_ctl, kc.bid, kc.is_t0
    lane, win_base, win_stride = kc.lane, kc.win_base, kc.win_stride

    def peer_va(pe, off):
        return win_base + fx.Int64(pe) * win_stride + fx.Int64(off)

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

    kc.peer_va, kc.stamp, kc.spin_end = peer_va, stamp, spin_end
    kc.alive, kc.spin_fail = alive, spin_fail


@device_fn
def start(kc):
    """Entry ticket -> (generation, parity, X1 tag); CTA 0 resets the next
    launch's counters."""
    COMPACT, DELAY_US, EARLY, EPR_CH = kc.COMPACT, kc.DELAY_US, kc.EARLY, kc.EPR_CH
    EPR_PAD, G, INV, addr_ctl, alive = kc.EPR_PAD, kc.G, kc.INV, kc.addr_ctl, kc.alive
    bc, bid, is_t0, lane, rsrc_actl = kc.bc, kc.bid, kc.is_t0, kc.lane, kc.rsrc_actl
    rsrc_ctl, stamp, wave = kc.rsrc_ctl, kc.stamp, kc.wave
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
        comm_ops.store_i32_lds(fx.Int64(bc + fx.Int32(4)), _i32_lo(l64) & fx.Int32(1))
        # X1 tag: 16 bits, never 0, differs from the previous launch's.
        comm_ops.store_i32_lds(
            fx.Int64(bc + fx.Int32(16)),
            _i32_lo(l64 % fx.Int64(65535)) + fx.Int32(1),
        )
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
        # Lane 3 resets the tally (compact) or reserved (EARLY) count.
        if lane < fx.Int32(4 if (COMPACT or EARLY) else 3):
            buffer_store(
                fx.Int32(0),
                rsrc_ctl,
                fx.Int32(
                    (lane == fx.Int32(3)).select(
                        fx.Int32(CTL_TALLY),
                        fx.Int32(CTL_PROD) + lane * fx.Int32(CTL_CLOSE - CTL_PROD),
                    )
                )
                + other * fx.Int32(32),
                cache_modifier=SCOPE_DEV,
            )
        rocdl.s_wait_storecnt(0)
    return gen, par, tag16


@device_fn
def bind_protocol(kc):
    """kc.poll_peer_flags / open_gates / wait_gate and the plan's
    lead_plan_lds / lead_plan_replicas / follower_plan (after start)."""
    COMPACT, EPR, EPR_CH, G = kc.COMPACT, kc.EPR, kc.EPR_CH, kc.G
    PLAN_SLOT, PLAN_TAG, PLAN_W, REP_W = kc.PLAN_SLOT, kc.PLAN_TAG, kc.PLAN_W, kc.REP_W
    TM, alive, bc, bid, gen, lane = kc.TM, kc.alive, kc.bc, kc.bid, kc.gen, kc.lane
    lds_cnt, lds_end, lds_rows, npes = kc.lds_cnt, kc.lds_end, kc.lds_rows, kc.npes
    rsrc_actl, rsrc_ctl, spin_end = kc.rsrc_actl, kc.rsrc_ctl, kc.spin_end
    spin_fail, stamp, wave = kc.spin_fail, kc.stamp, kc.wave

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
            bad = fx.Int32(ballot(T.i32, (lane < fx.Int32(npes)) & (cur != gen)))
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
                fx.Int32(gate_w) + (lane + fx.Int32(c * WAVE)) * fx.Int32(stride),
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

    def lead_plan_lds(cnts):
        """CTA 0, wave 0: its own LDS plan from the per-expert counts,
        and the counts for the replicas: compact's one per dword
        (dword e = cnt(e)), slots' packed two per dword (dword w =
        cnt(2w) | cnt(2w + 1) << 16, built by bpermute)."""
        plan_from_counts(cnts)
        if const_expr(COMPACT):
            for j in range_constexpr(EPR_CH):
                comm_ops.store_i32_lds(
                    fx.Int64(lds_cnt + (lane + fx.Int32(j * WAVE)) * fx.Int32(4)),
                    cnts[j],
                )
            return
        for j2 in range_constexpr(EPR_CH // 2 if EPR_CH > 1 else 1):
            w_ = lane + fx.Int32(j2 * WAVE)
            lo_e = w_ * fx.Int32(2)
            hi_e = lo_e + fx.Int32(1)
            lo = fx.Int32(0)
            hi = fx.Int32(0)
            for j in range_constexpr(EPR_CH):
                src_lo = (lo_e - fx.Int32(j * WAVE)) & fx.Int32(WAVE - 1)
                src_hi = (hi_e - fx.Int32(j * WAVE)) & fx.Int32(WAVE - 1)
                v_lo = fx.Int32(ds_bpermute(T.i32, src_lo * fx.Int32(4), cnts[j]))
                v_hi = fx.Int32(ds_bpermute(T.i32, src_hi * fx.Int32(4), cnts[j]))
                in_lo = (lo_e >= fx.Int32(j * WAVE)) & (lo_e < fx.Int32((j + 1) * WAVE))
                in_hi = (hi_e >= fx.Int32(j * WAVE)) & (hi_e < fx.Int32((j + 1) * WAVE))
                lo = fx.Int32(in_lo.select(v_lo, lo))
                hi = fx.Int32(in_hi.select(v_hi, hi))
            comm_ops.store_i32_lds(
                fx.Int64(lds_cnt + w_ * fx.Int32(4)),
                lo | (hi << fx.Int32(16)),
            )

    def lead_plan_replicas():
        """CTA 0, every wave: a quarter of the plan replicas from the
        replica counts in LDS (after a barrier past lead_plan_lds)."""
        pk = []
        for j2 in range_constexpr(REP_W):
            pk.append(
                fx.Int32(
                    comm_ops.load_i32_lds(
                        fx.Int64(lds_cnt + (lane + fx.Int32(j2 * WAVE)) * fx.Int32(4))
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
            for j2 in range_constexpr(REP_W):
                pkw.append(_load_i32_dev(rsrc_ctl, slot_w + fx.Int32(j2 * WAVE) + lane))
        if const_expr(COMPACT):
            # One count per dword: expert e = lane + 32 j is dword e.
            cnts = [
                fx.Int32(
                    (lane + fx.Int32(j * WAVE) < fx.Int32(EPR)).select(
                        pkw[j], fx.Int32(0)
                    )
                )
                for j in range_constexpr(EPR_CH)
            ]
            plan_from_counts(cnts)
            return
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
            comm_ops.store_i32_lds(fx.Int64(lds_end + e_ * fx.Int32(4)), incl)
            comm_ops.store_i32_lds(fx.Int64(lds_rows + e_ * fx.Int32(4)), c_)
            carry = _uniform(readlane(T.i32, arith.unwrap(incl), WAVE - 1))
        if lane == fx.Int32(0):
            comm_ops.store_i32_lds(fx.Int64(bc + fx.Int32(8)), carry)

    kc.poll_peer_flags, kc.open_gates = poll_peer_flags, open_gates
    kc.wait_gate, kc.follower_plan = wait_gate, follower_plan
    kc.lead_plan_lds, kc.lead_plan_replicas = lead_plan_lds, lead_plan_replicas


@device_fn
def publish_produced(kc, n_mine, n_routes):
    """Once every route of this rank is in place, tell every destination
    (``src_done``)."""
    SRC_OFF, addr_ctl, bid, gen, lane = kc.SRC_OFF, kc.addr_ctl, kc.bid, kc.gen, kc.lane
    npes, off_ctl, par, peer_va, rank = kc.npes, kc.off_ctl, kc.par, kc.peer_va, kc.rank
    stamp, wave = kc.stamp, kc.wave
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


@device_fn
def plan(kc):
    """2. PLAN: every CTA's LDS plan; returns the m-tile count."""
    CAP, COMPACT, EARLY, EPR, EPR_CH = kc.CAP, kc.COMPACT, kc.EARLY, kc.EPR, kc.EPR_CH
    EPR_PAD, GATE_P, GATE_R, GATE_S = kc.EPR_PAD, kc.GATE_P, kc.GATE_R, kc.GATE_S
    PLAN_TAG, RS_OFF, SRC_OFF, bc = kc.PLAN_TAG, kc.RS_OFF, kc.SRC_OFF, kc.bc
    bid, follower_plan, lane = kc.bid, kc.follower_plan, kc.lane
    lead_plan_lds, lead_plan_replicas = kc.lead_plan_lds, kc.lead_plan_replicas
    open_gates, par, poll_peer_flags = kc.open_gates, kc.par, kc.poll_peer_flags
    rsrc_actl, stamp, wave = kc.rsrc_actl, kc.stamp, kc.wave
    # ======================== 2. PLAN ========================
    # CTA 0 alone polls the peers' src_done (hundreds of CTAs polling
    # the same uncached lines starve every access queued behind them).
    # slots: it then lays out the plan, publishes it to the local
    # control block and opens each CTA's own gate line. compact laid
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
                poll_peer_flags(RS_OFF if EARLY else SRC_OFF, 2)
                stamp(14)
                cnts = []
                for j in range_constexpr(EPR_CH):
                    e_ = lane + fx.Int32(j * WAVE)
                    c_ = _load_i32_sys(
                        rsrc_actl,
                        (par * fx.Int32(EPR_PAD) + e_) * fx.Int32(RUN_STRIDE),
                        volatile=False,
                    )
                    c_ = fx.Int32((c_ > fx.Int32(CAP)).select(fx.Int32(CAP), c_))
                    c_ = fx.Int32((e_ < fx.Int32(EPR)).select(c_, fx.Int32(0)))
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
            if const_expr(EARLY):  # noqa: SIM102 - preserve DSL staging
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
    return n_mt


@device_fn
def gemm_queue(kc, n_mt):
    """3./4. the GEMM1 then GEMM2 items; returns (this CTA's GEMM2 units,
    the rank's total units)."""
    COMPACT, DEFER_SC, EARLY, EPR = kc.COMPACT, kc.DEFER_SC, kc.EARLY, kc.EPR
    EPR_CH, G, GATE_P, GATE_R, H = kc.EPR_CH, kc.G, kc.GATE_P, kc.GATE_R, kc.H
    MT_PER_E, N1, NT1, NT2, PH = kc.MT_PER_E, kc.N1, kc.NT1, kc.NT2, kc.PH
    POLL_SL, TM, TRACE, addr_ctl = kc.POLL_SL, kc.TM, kc.TRACE, kc.addr_ctl
    alive, arg_a2, arg_a2_scale = kc.alive, kc.arg_a2, kc.arg_a2_scale
    arg_disp, arg_disp_scale, arg_rowmap = kc.arg_disp, kc.arg_disp_scale, kc.arg_rowmap
    arg_w1, arg_w1_scale, arg_w2 = kc.arg_w1, kc.arg_w1_scale, kc.arg_w2
    arg_w2_scale, bc, bid = kc.arg_w2_scale, kc.bc, kc.bid
    f32_swiglu_limit, g1_task, g2_task = kc.f32_swiglu_limit, kc.g1_task, kc.g2_task
    gen, i32_rows, is_t0, lane = kc.gen, kc.i32_rows, kc.is_t0, kc.lane
    lds_end, lds_rows, meta1_holder = kc.lds_end, kc.lds_rows, kc.meta1_holder
    meta2_holder, par, ready1_holder = kc.meta2_holder, kc.par, kc.ready1_holder
    ready2_holder, rsrc_ctl, spin_end = kc.ready2_holder, kc.rsrc_ctl, kc.spin_end
    spin_fail, stamp, wait_gate = kc.spin_fail, kc.stamp, kc.wait_gate

    def decode_mtile(i):
        """(expert, GEMM m-tile index, valid rows) of plan m-tile i: the
        expert is the number of inclusive m-tile ends <= i."""
        e_ = fx.Int32(0)
        for j in range_constexpr(EPR_CH):
            e_j = lane + fx.Int32(j * WAVE)
            end_j = fx.Int32(
                comm_ops.load_i32_lds(fx.Int64(lds_end + e_j * fx.Int32(4)))
            )
            e_ = e_ + _popc_i32(
                fx.Int32(ballot(T.i32, (e_j < fx.Int32(EPR)) & (end_j <= i)))
            )
        e_ = _uniform(e_)
        e_prev = fx.Int32((e_ > fx.Int32(0)).select(e_ - fx.Int32(1), e_))
        prev = fx.Int32(comm_ops.load_i32_lds(fx.Int64(lds_end + e_prev * fx.Int32(4))))
        prev = _uniform(fx.Int32((e_ > fx.Int32(0)).select(prev, fx.Int32(0))))
        jt = i - prev
        cnt = _uniform(
            fx.Int32(comm_ops.load_i32_lds(fx.Int64(lds_rows + e_ * fx.Int32(4))))
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

    n1 = fx.Int32(N1)
    n2 = fx.Int32(H)
    rows_cap = i32_rows

    total_units = n_mt * fx.Int32(NT2 if PH & 4 else 0) + fx.Int32(1)

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
            if const_expr(COMPACT or EARLY):

                def a_ready1():
                    # The rows of every rank have landed (CTA 0 saw
                    # every src_done): only A owners wait, the W1
                    # prologue streams meanwhile.
                    wait_gate(GATE_P if COMPACT else GATE_R, GATE_STRIDE, 7)

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
                # a_ready, and compact / EARLY start the queue before
                # the peers' rows land: every wave waits for them
                # first. Once open, the gate costs one load per item.
                wait_gate(GATE_P if COMPACT else GATE_R, GATE_STRIDE, 7)

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
    if const_expr(DEFER_SC):
        # The last GEMM2 item's scatter: every wave retires its own
        # before the barrier that precedes the close.
        tdm_ops.tensor_wait(0)
        rocdl.s_barrier_signal(-1)
        rocdl.s_barrier_wait(-1)
    stamp(4)
    stamp(TR_ITEMS, fx.Int64(n_items))
    return n2_done, total_units


@device_fn
def close(kc, n2_done, total_units):
    """4. CLOSE: count this CTA's units; the last tells every peer."""
    CF_OFF, addr_ctl, bid, gen, lane = kc.CF_OFF, kc.addr_ctl, kc.bid, kc.gen, kc.lane
    npes, off_ctl, par, peer_va, rank = kc.npes, kc.off_ctl, kc.par, kc.peer_va, kc.rank
    stamp, wave = kc.stamp, kc.wave

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

    # CTA 0's counter reset (its stores retired above) is one unit.
    n2_done = n2_done + fx.Int32((bid == fx.Int32(0)).select(fx.Int32(1), fx.Int32(0)))
    if (wave == fx.Int32(0)) & (n2_done > fx.Int32(0)):
        close_units(n2_done)
    stamp(5)


@device_fn
def reduce(kc):
    """5. REDUCE: once every peer closed, sum each token's top-k slots."""
    CF_OFF, E, G, GATE_C, H, K, PH = kc.CF_OFF, kc.E, kc.G, kc.GATE_C, kc.H, kc.K, kc.PH
    RED_CH, RED_UN, addr_out, bid = kc.RED_CH, kc.RED_UN, kc.addr_out, kc.bid
    RED_B64 = kc.RED_B64
    comb_slot_stride, lane, off_comb_inp = kc.comb_slot_stride, kc.lane, kc.off_comb_inp
    open_gates, peer_va, poll_peer_flags = kc.open_gates, kc.peer_va, kc.poll_peer_flags
    rank, rsrc_ids, stamp, tok = kc.rank, kc.rsrc_ids, kc.stamp, kc.tok
    wait_gate, wave, wave_gid = kc.wait_gate, kc.wave, kc.wave_gid
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
    # RED_B64: each token's top-k slots get their own resource from its 64-bit
    # base (past ~14.6 k tokens x top-9 the region outgrows i32 offsets);
    # otherwise one resource and i32 offsets.
    comb_base = peer_va(rank, off_comb_inp)
    if const_expr(not RED_B64):
        rsrc_comb_all = create_buffer_resource_from_addr(comb_base)
    rsrc_out = create_buffer_resource_from_addr(addr_out)
    n_red = tok * fx.Int32(RED_CH if PH & 8 else 0)
    for w in range(wave_gid, n_red, G * 4 * RED_UN):
        # Every item's loads first, then each item's sum in k order (the
        # staged reduce's arithmetic). An item past the end reloads the
        # first one and skips its store.
        items = []
        for u in range_constexpr(RED_UN):
            w_u = w + fx.Int32(u * G * 4)
            in_u = w_u < n_red
            w_u = fx.Int32(in_u.select(w_u, w))
            t = w_u // fx.Int32(RED_CH)
            ch = w_u - t * fx.Int32(RED_CH)
            if const_expr(RED_B64):
                rsrc_comb = create_buffer_resource_from_addr(
                    comb_base + fx.Int64(t * fx.Int32(K)) * fx.Int64(comb_slot_stride)
                )
                tok_b = fx.Int32(0)
            else:
                rsrc_comb = rsrc_comb_all
                tok_b = t * fx.Int32(K * int(comb_slot_stride))
            vals = []
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
                slot_b = tok_b + fx.Int32(k * int(comb_slot_stride))
                for j in range_constexpr(2):
                    raw = buffer_load(
                        rsrc_comb,
                        (slot_b + ch * fx.Int32(RED_ELEMS * 2) + lane * fx.Int32(32))
                        // fx.Int32(4)
                        + fx.Int32(4 * j),
                        vec_width=4,
                        dtype=T.i32,
                        cache_modifier=SCOPE_SYS,
                    )
                    vals.append((valid, raw))
            items.append((t, ch, in_u, vals))
        for u in range_constexpr(RED_UN):
            t, ch, in_u, vals = items[u]
            accs = [fx.full(8, 0.0, fx.Float32) for _ in range_constexpr(2)]
            for k in range_constexpr(K):
                for j in range_constexpr(2):
                    valid, raw = vals[2 * k + j]
                    v = fx.Vector(raw).bitcast(fx.BFloat16).to(fx.Float32)
                    v = valid.select(v, fx.full(8, 0.0, fx.Float32))
                    accs[j] = accs[j] + v
            o_dw = (
                t * fx.Int32(H) + ch * fx.Int32(RED_ELEMS) + lane * fx.Int32(16)
            ) // fx.Int32(2)
            if in_u:
                for j in range_constexpr(2):
                    buffer_store(
                        accs[j].to(fx.BFloat16).bitcast(fx.Int32),
                        rsrc_out,
                        o_dw + fx.Int32(4 * j),
                    )
    stamp(7)
