# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""MegaMoEV2 Stage2 + combine with the transfer done by the SDMA copy engines.

Ported from mori's ``examples/cco/python/11_flydsl_mega_moe_sdma`` (ROCm/mori#859),
with the 8-wave GEMM2 of ``gemm2_8wave.py``, an exact send/landing layout from the
gathered route counts, and a fallback for rows the send buffer cannot hold.

The default Stage2 scatters every GEMM2 row straight into the origin rank's
``comb_inp``, so the GEMM's stores wait on the XGMI links. Here a row goes to a
local send buffer ``S`` grouped by origin rank, and a copy engine moves one
contiguous run per peer while the GEMM is still computing::

    S = [ origin rank 0: rows... | origin rank 1: rows... | ... ]   (exact counts, packed)

Within a rank's run rows are in sorted-row (expert-major) order, so a prefix of
every run is finished as soon as the first experts are.

Three kernels per call, on one stream:

    1. plan     ``out_row`` per sorted row, ``idx_send``, ``tile_base``, ``cursor``,
                and ``inv`` for the rows that stay on this rank; zeroes ``cdone``
    2. gemm2    (``gemm2_8wave``) writes ``S[out_row]``; workgroup 0's waves, one per
                peer, fill that peer's ``inv`` and put each finished chunk with SDMA
    3. combine  waits for the peers, reduces the top-k rows, publishes "consumed"

Everything the peers address lives in ONE cco window, so a put names the same
window for source and destination and only the byte offsets differ::

    [ S | R | inv | sig | flags | done | counter ]

``inv[slot]`` (``slot = origin_lid * topk + k``) is the landing row of that slot's
row, counted from the start of ``S`` (``S`` rows ``[0, send_rows)``, ``R`` rows
``[send_rows, send_rows + recv_rows)``; my own run is read from ``S`` in place), or
``INV_P2P`` for a row that came through ``comb_inp``, or -1. The sender writes it,
so no index travels with the data. ``flags[src]`` says "src's data for call n has
landed", ``done[dst]`` "dst has finished reading call n" (a sender waits for it
before overwriting that peer's ``R`` / ``inv``), and ``counter`` is the call number,
advanced only by the combine. All three are i64, as is the copy engine's signal
counter ``wait_signal`` compares against, so none of them wraps in practice; at
1200 calls a second an i32 call number would have gone negative in 21 days.

The layout is exact. Prepare all-gathers every rank's per-expert route counts
(``bigcnt``), so the plan kernel knows how many rows every rank sends every other
rank before anything moves: my run for ``d`` starts at ``send_base[d]`` (the rows I
send lower-numbered ranks), and lands in ``d``'s ``R`` at ``land_off[d]`` (the rows
lower-numbered ranks other than ``d`` send ``d``). ``R`` holds what one rank gets
back, at most ``max_tok * topk`` rows, so it never overflows. ``S`` holds what one
rank sends -- everything it received in the dispatch, which skewed routing can make
several times the even share -- and is sized ``send_slack`` times ``max_tok * topk``
(capped by the 32-bit buffer-resource range). Only rows past it take the direct
path: the plan marks them ``OUT_ROW_P2P`` and GEMM2 writes them straight into the
origin rank's ``comb_inp[slot]`` (the default Stage2's layout) and sets the peer's
``inv[slot]`` to ``INV_P2P``.

Rows are fp8 with one e8m0 scale per 32 values, payload then scales, as the
default Stage2's P2P scatter lays them out, so one contiguous put carries a row.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.expr import const_expr, range_constexpr, rocdl
from flydsl.expr.rocdl import cvt_pk_f32_fp8
from flydsl.expr.typing import Int8, T

from .. import communication_ops_utils as comm_ops
from ..mxfp4_gemm_common import global_typed_ptr, lds_typed_ptr
from ..tensor_shim import _run_compiled, buf_copy_load, buf_copy_store, ptr_buf_tensor
from . import envs as _envs

__all__ = ["MegaMoESdmaStage2", "sdma_stage2_unsupported_reason"]

#: i32 slots between consecutive chunk-completion counters (one 128 B line each:
#: every unit of a chunk bumps its counter).
CDONE_STRIDE = 32
#: Rows per lane per loop trip when a comm wave pushes inv entries to a peer.
PUSH_U = 8
#: ``out_row`` of a valid row past the end of ``S``: written through ``comb_inp``.
OUT_ROW_P2P = -2
#: ``inv`` of a slot whose row is in this rank's ``comb_inp``.
INV_P2P = -2

BLOCK = 256
WAVE = 64
BAR_STRIDE = 16  # i64 slots per tree_barrier counter: one 128 B cache line
BAR_GROUP = 32  # blocks per tree_barrier group
MIN_SBM = 16
_ALIGN = 128


def _align(n):
    return (n + _ALIGN - 1) // _ALIGN * _ALIGN


def arena_rows(*, max_tok, topk, hidden, send_slack):
    """``(send_rows, recv_rows, row_bytes)`` of the window; raises RuntimeError
    when they do not fit.

    R holds what one rank gets back -- at most its own routes -- so it never
    overflows. S holds what one rank sends, everything it received; it is
    ``send_slack`` times the even share, capped so that S and R stay inside the
    32-bit buffer-resource offset GEMM2 and the combine address them through.
    """
    row_bytes = hidden + hidden // 32
    recv_rows = _align(max_tok * topk)
    fit = ((1 << 31) - 1 - recv_rows * row_bytes) // row_bytes // _ALIGN * _ALIGN
    send_rows = min(_align(int(max_tok * topk * send_slack)), fit)
    if send_rows < _ALIGN:
        raise RuntimeError(
            f"max_tok_per_rank={max_tok}: the landing buffer alone "
            f"({recv_rows * row_bytes} B) leaves no 32-bit room for a send buffer"
        )
    return send_rows, recv_rows, row_bytes


class ArenaLayout:
    """Byte offsets inside the cco window; identical on every rank."""

    def __init__(self, npes, send_rows, recv_rows, row_bytes, slot_cap):
        self.npes, self.send_rows, self.recv_rows = npes, send_rows, recv_rows
        self.row_bytes, self.slot_cap = row_bytes, slot_cap
        self.off_s = 0
        self.off_r = send_rows * row_bytes
        self.off_inv = self.off_r + recv_rows * row_bytes
        self.off_sig = self.off_inv + _align(slot_cap * 4)
        self.off_flags = self.off_sig + _align(npes * 4)
        self.off_done = self.off_flags + _align(npes * 8)
        self.off_counter = self.off_done + _align(npes * 8)
        self.total_bytes = self.off_counter + _ALIGN


def _uniform_i64(x):
    """A wave-uniform i64 (SGPR) copy of ``x``, as a buffer-resource base needs."""
    return fx.Int64(rocdl.readfirstlane(T.i64, fx.Int64(x).ir_value()))


def _decode(p, *, log2_max_tok, recv_cap, topk, npes):
    """Packed srcmap word -> (valid, dest_pe, dest_lid, slot_k), as the Stage2 epilogue does."""
    t = p & fx.Int32(0x00FFFFFF)
    s = p.shrui(fx.Int32(24)) & fx.Int32(0xFF)
    dest_pe = t >> fx.Int32(log2_max_tok)
    dest_lid = t & fx.Int32((1 << log2_max_tok) - 1)
    valid = (t < fx.Int32(recv_cap)) & (s < fx.Int32(topk)) & (dest_pe < fx.Int32(npes))
    return valid, dest_pe, dest_lid, s


# --- Cross-rank waits -----------------------------------------------------------

#: Polls before a cross-rank wait traps instead of hanging (0: wait forever). Read
#: when a kernel is compiled; a trap aborts the dispatch with the wait's location.
SPIN_POLLS = int(_envs.AITER_MEGA_SDMA_SPIN_POLLS)


def load_i64(addr, *, scope, acquire=False):
    """Volatile i64 load at ``addr`` with ``scope`` visibility, for a spin-wait."""
    return fx.Int64(
        fx.generic_load(
            comm_ops._to_ptr_global(addr, fx.Int64, 8),
            volatile=True,
            memory_order=(
                fx.AtomicOrdering.Acquire if acquire else fx.AtomicOrdering.Monotonic
            ),
            syncscope=scope,
        )
    )


@flyc.jit
def spin_until_ge_i64(addr, val, *, scope, acquire=False):
    """Spin until the i64 at ``addr`` reaches ``val``; traps after ``SPIN_POLLS`` polls
    when that is set."""
    cur = load_i64(addr, scope=scope, acquire=acquire)
    if const_expr(SPIN_POLLS > 0):
        polls = fx.Int32(0)
        while cur < fx.Int64(val):
            polls = polls + fx.Int32(1)
            if polls >= fx.Int32(SPIN_POLLS):
                comm_ops._device_trap()
            rocdl.s_sleep(1)
            cur = load_i64(addr, scope=scope, acquire=acquire)
    else:
        while cur < fx.Int64(val):
            rocdl.s_sleep(1)
            cur = load_i64(addr, scope=scope, acquire=acquire)
    return cur


@flyc.jit
def spin_until_ge_i32_agent(addr, val):
    """``comm_ops.spin_until_ge_i32_agent`` with the ``SPIN_POLLS`` trap. For counters
    that restart from zero every call, so an i32 cannot wrap."""
    cur = fx.Int32(comm_ops.load_i32_global_agent(addr))
    if const_expr(SPIN_POLLS > 0):
        polls = fx.Int32(0)
        while cur < fx.Int32(val):
            polls = polls + fx.Int32(1)
            if polls >= fx.Int32(SPIN_POLLS):
                comm_ops._device_trap()
            rocdl.s_sleep(1)
            cur = fx.Int32(comm_ops.load_i32_global_agent(addr))
    else:
        while cur < fx.Int32(val):
            rocdl.s_sleep(1)
            cur = fx.Int32(comm_ops.load_i32_global_agent(addr))
    return cur


def call_number(w, *, layout, rank):
    """The completed-call count (``counter``), as an i64."""
    return load_i64(fx.Int64(w.lsa_ptr(rank, layout.off_counter)), scope=SYS)


SYS = fx.rocdl.SyncScope.OneAs
AGENT = fx.rocdl.SyncScope.AgentOneAs


# --- The sender side of the SDMA protocol (called by the GEMM2 comm waves) ------


def sdma_put_run(
    sdma,
    win,
    peer,
    queue,
    *,
    layout,
    send_base,
    land_off,
    lo,
    hi,
    cco,
    remote_signal=False,
):
    """Ship rows ``[lo, hi)`` of my run for ``peer`` (``S`` row ``send_base`` on) with one
    put, to ``peer``'s ``R`` row ``land_off`` on.

    No local signal (nothing waits on it). ``remote_signal`` hangs the completion
    signal on this put: the copy engine bumps the peer's counter once the bytes
    are there, and the peer's combine polls it. A queue is a FIFO its engine
    drains in order, so a signal on the last put covers every put before it.
    """
    rb = layout.row_bytes
    sdma.put(
        peer,
        win,
        fx.Int64(layout.off_r) + (fx.Int64(land_off) + fx.Int64(lo)) * fx.Int64(rb),
        win,
        fx.Int64(layout.off_s) + (fx.Int64(send_base) + fx.Int64(lo)) * fx.Int64(rb),
        fx.Int64(hi - lo) * fx.Int64(rb),
        queue,
        coop=cco.CoopScope.THREAD,
        local_signal=False,
        remote_signal=remote_signal,
    )


def sdma_signal_peer(sdma, win, peer, queue, *, layout, rank, cco):
    """A 4-byte put that only carries the completion signal (a peer whose last chunk was empty)."""
    sdma.put(
        peer,
        win,
        fx.Int64(layout.off_sig + rank * 4),
        win,
        fx.Int64(layout.off_sig + peer * 4),
        fx.Int64(4),
        queue,
        coop=cco.CoopScope.THREAD,
        remote_signal=True,
    )


def publish_flag(w, peer, phase, *, layout, rank):
    """``flags[rank]`` on ``peer`` = call number (the data itself is covered by the SDMA signal)."""
    comm_ops.store_i64_global_system(
        fx.Int64(w.lsa_ptr(peer, layout.off_flags)) + fx.Int64(rank * 8), phase
    )


# --- Kernel 1: plan -------------------------------------------------------------


def tree_barrier(bar_addr, nblocks, tid, bid):
    """All blocks of a resident grid rendezvous with two levels of small atomics.

    Blocks are grouped ``BAR_GROUP`` to a counter; the last block of a group bumps a
    top counter; everyone polls only the top one. Counters are never reset: a
    group's counter advances by its size per launch, so the launch number is
    ``old // size``. They are i64 for that reason: an i32 group counter goes negative
    after 2^26 launches, and the signed wait on the top counter then passes at once.
    Each counter sits on its own cache line.
    """
    group = bid // fx.Int32(BAR_GROUP)
    n_groups = (nblocks + fx.Int32(BAR_GROUP - 1)) // fx.Int32(BAR_GROUP)
    size = nblocks - group * fx.Int32(BAR_GROUP)
    size = fx.Int64((size < fx.Int32(BAR_GROUP)).select(size, fx.Int32(BAR_GROUP)))
    fx.rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0, expcnt=0)
    fx.barrier()

    @flyc.jit
    def arrive_and_wait():
        if tid == fx.Int32(0):
            top = bar_addr + fx.Int64(n_groups) * fx.Int64(BAR_STRIDE * 8)
            comm_ops.fence_agent_release()
            old = fx.Int64(
                comm_ops.atomic_add_agent(
                    bar_addr + fx.Int64(group) * fx.Int64(BAR_STRIDE * 8), fx.Int64(1)
                )
            )
            launch = old // size
            if old - launch * size == size - fx.Int64(1):
                comm_ops.atomic_add_agent(top, fx.Int64(1))
            spin_until_ge_i64(
                top, (launch + fx.Int64(1)) * fx.Int64(n_groups), scope=AGENT
            )
            comm_ops.fence_agent_acquire()

    arrive_and_wait()
    fx.barrier()


def build_plan(
    *, SBM, topk, log2_max_tok, recv_cap, npes, rank, epr, total_segments, send_rows
):
    """Kernel 1: give every valid sorted row a stable index in its rank's run.

    * offsets: from the gathered route counts ``bigcnt[src][segment]`` (per expert, then
              one group column per rank for the dispatch's fanout pair), per pe:
              ``n_send`` (rows I send it), ``send_base`` (rows I send lower pes),
              ``land_off`` (rows lower ranks other than pe send it) and ``avail``
              (rows of its run that fit in ``S``: ``send_rows - send_base``, clamped)
    * count:  ``tile_cnt[tile][pe]`` = rows of the sort tile bound for ``pe``
    * prefix: ``tile_base[tile][pe]`` = rows for ``pe`` in earlier tiles (so the rows
              of a tile range are contiguous in the run), ``cursor[pe]`` = total
    * assign: ``out_row = send_base[pe] + tile_base + (earlier rows of this tile for pe)``
              and ``idx_send[out_row] = origin_lid*topk + k``. Rows bound for this rank
              also get their ``inv`` entry here; the peers' entries are pushed by
              the GEMM2 comm waves. A row past ``avail`` gets ``OUT_ROW_P2P``.

    ``cursor`` = [rows in the run (clamped to avail) | send_base | land_off | avail]
    per pe, then a counter of pes whose counted rows differ from ``bigcnt``'s (must
    stay 0). One grid-wide rendezvous, between count and prefix. Before writing
    anything the kernel waits until every peer has consumed the previous call: with
    the copy engine's FIFO that means this rank's puts have finished reading ``S``.
    """
    dec = {
        "log2_max_tok": log2_max_tok,
        "recv_cap": recv_cap,
        "topk": topk,
        "npes": npes,
    }
    if SBM > BLOCK:
        raise ValueError(
            f"plan needs one thread per row of a sort tile: SBM={SBM} > {BLOCK}"
        )
    if BLOCK % npes:
        raise ValueError(f"plan needs npes to divide {BLOCK}, got {npes}")
    GROUPS = BLOCK // npes
    WAVES = BLOCK // WAVE
    # LDS words: per-wave per-pe counts | per-thread partial sums | the tile's prefix.
    OFF_PART = WAVES * npes
    OFF_PREF = OFF_PART + BLOCK
    # Per pe, for the whole kernel: rows I send it, my run's base in S, rows that fit.
    OFF_NSEND = OFF_PREF + npes
    OFF_SBASE = OFF_NSEND + npes
    OFF_AVAIL = OFF_SBASE + npes
    OFF_LAND = OFF_AVAIL + npes
    lds_bytes = (OFF_LAND + npes) * 4

    @fx.struct
    class PlanShared:
        buf: fx.Array[Int8, lds_bytes, 16]

    # fmt: off
    @flyc.kernel(name=f"mega_moe_sdma_plan_sbm{SBM}_k{topk}_p{npes}_s{send_rows}_e{epr}x{total_segments}"
                      f"_r{rank}", known_block_size=[BLOCK, 1, 1])
    def plan(arg_stids: fx.Int64, arg_trb: fx.Int64, arg_num_valid: fx.Int64,
        arg_tile_cnt: fx.Int64, arg_tile_base: fx.Int64, arg_cursor: fx.Int64,
        arg_out_row: fx.Int64, arg_idx: fx.Int64, arg_inv: fx.Int64, arg_cdone: fx.Int64,
        arg_bar: fx.Int64, arg_over: fx.Int64, arg_done: fx.Int64, arg_counter: fx.Int64,
        arg_bigcnt: fx.Int64, arg_pair_cfg: fx.Int64, arg_parity: fx.Int64, i32_nblocks: fx.Int32,
        i32_cdone_n: fx.Int32):
    # fmt: on
        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        gid = bid * fx.Int32(BLOCK) + tid
        stride = i32_nblocks * fx.Int32(BLOCK)
        num_valid = global_typed_ptr(arg_num_valid, T.i32)[0]
        tiles = (num_valid + fx.Int32(SBM - 1)) // fx.Int32(SBM)
        trb = ptr_buf_tensor(arg_trb, fx.Int32)
        stids = ptr_buf_tensor(arg_stids, fx.Int32)
        tile_cnt = ptr_buf_tensor(arg_tile_cnt, fx.Int32)
        tile_base = ptr_buf_tensor(arg_tile_base, fx.Int32)
        out_buf = ptr_buf_tensor(arg_out_row, fx.Int32)
        cdone = ptr_buf_tensor(arg_cdone, fx.Int32)

        lds = fx.SharedAllocator().allocate(PlanShared).peek()
        lds_i32 = fx.Int32(fx.ptrtoint(lds.buf.ptr))

        def lw(i):
            """LDS i32 word ``i`` of the block's scratch."""
            return lds_typed_ptr(lds_i32 + i * fx.Int32(4), T.i32, align=4)

        # S and the peers' inv and comb_inp are rewritten from here on: wait until
        # every peer has read what I sent it last call. GEMM2 runs after this kernel
        # on the same stream, so this covers its comm waves and direct rows too.
        consumed = load_i64(arg_counter, scope=SYS)
        if (tid < fx.Int32(npes)) & (tid != fx.Int32(rank)):
            spin_until_ge_i64(
                fx.Int64(arg_done) + fx.Int64(tid) * fx.Int64(8),
                consumed,
                scope=SYS,
                acquire=True,
            )
        fx.barrier()

        # The GEMM2 kernel's per-chunk completion counters start from zero every call.
        for g in range(gid, i32_cdone_n, stride):
            cdone[g] = fx.Int32(0)

        # Exact offsets from the gathered route counts. Thread pe reads bigcnt row pe
        # (the routes pe dispatched) and, per rank s, the rows pe sent s: s's expert
        # counts plus, for the fanout pair the dispatch picked at s this call, one row per
        # pair member for every token sent once as a group (bigcnt column TE + s).
        # Rows I send pe = the rows pe sent me; they land in pe's R after the rows of
        # the ranks below me other than pe itself.
        cursor_buf = ptr_buf_tensor(arg_cursor, fx.Int32)
        bc = ptr_buf_tensor(arg_bigcnt, fx.Int32)
        parity = fx.Int32(comm_ops.load_i32_system(arg_parity, fx.Int32(0)))
        if tid < fx.Int32(npes):
            row = tid * fx.Int32(total_segments)
            n_send = fx.Int32(0)
            below = fx.Int32(0)
            for dst in range_constexpr(npes):
                if const_expr(dst <= rank):
                    rows = fx.Int32(0)
                    for e in range(fx.Int32(dst * epr), fx.Int32(dst * epr + epr), fx.Int32(1)):
                        rows = rows + fx.Int32(bc[row + e])
                    packed = fx.Int32(comm_ops.load_i32_system(arg_pair_cfg, parity * fx.Int32(npes) + fx.Int32(dst)))
                    enabled = (packed & fx.Int32(1 << 16)) != fx.Int32(0)
                    same = (packed & fx.Int32(0xFF)) == ((packed >> fx.Int32(8)) & fx.Int32(0xFF))
                    members = enabled.select(same.select(fx.Int32(1), fx.Int32(2)), fx.Int32(0))
                    rows = rows + members * fx.Int32(bc[row + fx.Int32(total_segments - npes + dst)])
                    if const_expr(dst == rank):
                        n_send = rows
                    else:
                        below = below + (tid != fx.Int32(dst)).select(rows, fx.Int32(0))
            fx.ptr_store(n_send, lw(fx.Int32(OFF_NSEND) + tid))
            fx.ptr_store(below, lw(fx.Int32(OFF_LAND) + tid))
        fx.barrier()
        if tid == fx.Int32(0):
            sbase = fx.Int32(0)
            for p in range_constexpr(npes):
                n_p = fx.Int32(fx.ptr_load(lw(fx.Int32(OFF_NSEND + p))))
                room = fx.Int32(send_rows) - sbase
                room = (room > fx.Int32(0)).select(room, fx.Int32(0))
                fx.ptr_store(sbase, lw(fx.Int32(OFF_SBASE + p)))
                fx.ptr_store((n_p < room).select(n_p, room), lw(fx.Int32(OFF_AVAIL + p)))
                sbase = sbase + n_p
        fx.barrier()
        if (bid == fx.Int32(0)) & (tid < fx.Int32(npes)):
            cursor_buf[fx.Int32(npes) + tid] = fx.Int32(fx.ptr_load(lw(fx.Int32(OFF_SBASE) + tid)))
            cursor_buf[fx.Int32(2 * npes) + tid] = fx.Int32(fx.ptr_load(lw(fx.Int32(OFF_LAND) + tid)))
            cursor_buf[fx.Int32(3 * npes) + tid] = fx.Int32(fx.ptr_load(lw(fx.Int32(OFF_AVAIL) + tid)))

        lane = tid % fx.Int32(WAVE)
        wave = tid // fx.Int32(WAVE)
        row_ok = tid < fx.Int32(SBM)

        def tile_masks(tile):
            """One thread per row of ``tile``: decode it, ballot per pe, publish the
            per-wave counts to LDS (visible after the barrier)."""
            base = trb[tile]
            valid, dest_pe, dest_lid, s = _decode(
                stids[base + row_ok.select(tid, fx.Int32(0))], **dec
            )
            valid = valid & row_ok
            masks = [
                fx.Int64(rocdl.ballot(T.i64, valid & (dest_pe == fx.Int32(p))))
                for p in range_constexpr(npes)
            ]
            if lane == fx.Int32(0):
                for p in range_constexpr(npes):
                    fx.ptr_store(
                        fx.Int32(fx.math.ctpop(masks[p])), lw(wave * fx.Int32(npes) + fx.Int32(p))
                    )
            fx.barrier()
            return valid, dest_pe, dest_lid, s, masks

        for tile in range(bid, tiles, i32_nblocks):
            tile_masks(tile)
            if tid < fx.Int32(npes):
                total = fx.Int32(0)
                for w in range_constexpr(WAVES):
                    total = total + fx.Int32(fx.ptr_load(lw(fx.Int32(w * npes) + tid)))
                tile_cnt[tile * fx.Int32(npes) + tid] = total
            fx.barrier()
        tree_barrier(arg_bar, i32_nblocks, tid, bid)

        # No rows at all: the loop below never runs, so say so explicitly.
        if (tiles == fx.Int32(0)) & (bid == fx.Int32(0)) & (tid < fx.Int32(npes)):
            tile_base[tid] = fx.Int32(0)
            cursor_buf[tid] = fx.Int32(0)
            if fx.Int32(fx.ptr_load(lw(fx.Int32(OFF_NSEND) + tid))) != fx.Int32(0):
                comm_ops.atomic_add_agent(fx.Int64(arg_cursor) + fx.Int64(16 * npes), fx.Int32(1))

        idx_buf = ptr_buf_tensor(arg_idx, fx.Int32)
        inv_buf = ptr_buf_tensor(arg_inv, fx.Int32)
        lane_lt = (fx.Int64(1) << fx.Int64(lane)) - fx.Int64(1)
        for tile in range(bid, tiles, i32_nblocks):
            # Rows for pe in the tiles before this one; element e of tile_cnt is
            # (tile e // npes, pe e % npes) and BLOCK % npes == 0, so a thread sums one pe.
            part = fx.Int32(0)
            for e in range(tid, tile * fx.Int32(npes), fx.Int32(BLOCK)):
                part = part + tile_cnt[e]
            fx.ptr_store(part, lw(fx.Int32(OFF_PART) + tid))
            fx.barrier()
            if tid < fx.Int32(npes):
                pre = fx.Int32(0)
                for k in range_constexpr(GROUPS):
                    pre = pre + fx.Int32(fx.ptr_load(lw(fx.Int32(OFF_PART + k * npes) + tid)))
                fx.ptr_store(pre, lw(fx.Int32(OFF_PREF) + tid))
                avail = fx.Int32(fx.ptr_load(lw(fx.Int32(OFF_AVAIL) + tid)))
                tile_base[tile * fx.Int32(npes) + tid] = (pre < avail).select(pre, avail)
                # The extra row holds the totals the comm waves read as a chunk's end.
                if tile == tiles - fx.Int32(1):
                    total = pre + tile_cnt[tile * fx.Int32(npes) + tid]
                    capped = (total < avail).select(total, avail)
                    tile_base[tiles * fx.Int32(npes) + tid] = capped
                    cursor_buf[tid] = capped
                    # The rows counted here must be the ones the gathered counts promised
                    # (the peers placed my rows in their R from those).
                    if total != fx.Int32(fx.ptr_load(lw(fx.Int32(OFF_NSEND) + tid))):
                        comm_ops.atomic_add_agent(
                            fx.Int64(arg_cursor) + fx.Int64(16 * npes), fx.Int32(1)
                        )
            fx.barrier()

            valid, dest_pe, dest_lid, s, masks = tile_masks(tile)
            g = tile * fx.Int32(SBM) + tid
            if row_ok:
                out_buf[g] = fx.Int32(-1)
            if valid:
                # Rank among this wave's rows for my pe, plus the earlier waves' rows for it.
                mine = masks[0]
                for p in range_constexpr(1, npes):
                    mine = (dest_pe == fx.Int32(p)).select(masks[p], mine)
                within = fx.Int32(fx.math.ctpop(mine & lane_lt))
                for w in range_constexpr(WAVES):
                    within = within + (fx.Int32(w) < wave).select(
                        fx.Int32(fx.ptr_load(lw(fx.Int32(w * npes) + dest_pe))), fx.Int32(0)
                    )
                place = fx.Int32(fx.ptr_load(lw(fx.Int32(OFF_PREF) + dest_pe))) + within
                if place >= fx.Int32(fx.ptr_load(lw(fx.Int32(OFF_AVAIL) + dest_pe))):
                    # Past the end of S: GEMM2 writes it through comb_inp.
                    out_buf[g] = fx.Int32(OUT_ROW_P2P)
                    comm_ops.atomic_add_agent(arg_over, fx.Int32(1))
                else:
                    out_row = fx.Int32(fx.ptr_load(lw(fx.Int32(OFF_SBASE) + dest_pe))) + place
                    out_buf[g] = out_row
                    slot = dest_lid * fx.Int32(topk) + s
                    idx_buf[out_row] = slot
                    if dest_pe == fx.Int32(rank):
                        inv_buf[slot] = out_row
            fx.barrier()

    # fmt: off
    @flyc.jit
    def launch(arg_stids: fx.Int64, arg_trb: fx.Int64, arg_num_valid: fx.Int64,
        arg_tile_cnt: fx.Int64, arg_tile_base: fx.Int64, arg_cursor: fx.Int64,
        arg_out_row: fx.Int64, arg_idx: fx.Int64, arg_inv: fx.Int64, arg_cdone: fx.Int64,
        arg_bar: fx.Int64, arg_over: fx.Int64, arg_done: fx.Int64, arg_counter: fx.Int64,
        arg_bigcnt: fx.Int64, arg_pair_cfg: fx.Int64, arg_parity: fx.Int64, i32_nblocks: fx.Int32,
        i32_cdone_n: fx.Int32, stream: fx.Stream):
    # fmt: on
        plan(arg_stids, arg_trb, arg_num_valid, arg_tile_cnt, arg_tile_base, arg_cursor,
             arg_out_row, arg_idx, arg_inv, arg_cdone, arg_bar, arg_over, arg_done, arg_counter,
             arg_bigcnt, arg_pair_cfg, arg_parity, i32_nblocks, i32_cdone_n).launch(
            grid=(i32_nblocks, 1, 1), block=(BLOCK, 1, 1), stream=stream)

    return launch


# --- Kernel 3: combine ------------------------------------------------------------


def build_combine(*, layout, rank, hidden, topk):
    """Kernel 3: finish the transfer and reduce.

    A. wait for every peer's flag and SDMA completion signal for this call;
    B. ``out[t] = sum_k row(inv[t*topk+k])`` in f32, one block per token in turn: a
       landing row of ``S``/``R``, or ``comb_inp[slot]`` for ``INV_P2P``. A slot's
       ``inv`` is reset to -1 once read; slots no route filled stay -1 and add zero;
    C. the last block publishes ``done`` to every peer and advances ``counter``.
    """
    import mori.cco.device.flydsl as cco

    if hidden % 32:
        raise ValueError(f"fp8 1x32 rows need hidden % 32 == 0, got {hidden}")
    npes = layout.npes
    units = hidden // 8
    chunks = (units + BLOCK - 1) // BLOCK
    row_bytes = layout.row_bytes

    # fmt: off
    @flyc.kernel(name=f"mega_moe_sdma_combine_p{npes}_s{layout.send_rows}r{layout.recv_rows}_h{hidden}_k{topk}_r{rank}",
                 known_block_size=[BLOCK, 1, 1])
    def combine(win: fx.Int64, arg_inv: fx.Int64, arg_out: fx.Int64, arg_comb_inp: fx.Int64,
        arg_fin: fx.Int64, arg_dev_comm: fx.Int64, i32_nblocks: fx.Int32, i32_tokens: fx.Int32):
    # fmt: on
        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        w = cco.Window(win)
        counter_addr = fx.Int64(w.lsa_ptr(rank, layout.off_counter))
        # Advanced only by the last block below, after every block has read it.
        phase = call_number(w, layout=layout, rank=rank) + fx.Int64(1)
        flags_base = fx.Int64(w.lsa_ptr(rank, layout.off_flags))

        # A. the peers' data has landed
        if (tid < fx.Int32(npes)) & (tid != fx.Int32(rank)):
            spin_until_ge_i64(
                flags_base + fx.Int64(tid) * fx.Int64(8), phase, scope=SYS, acquire=True
            )
            # The sender's copy engine bumps this u64 once per call when its last put
            # lands, so the call number is compared at its full width.
            cco.DevComm(arg_dev_comm).sdma().wait_signal(tid, 0, phase)
            comm_ops.fence_system_acquire()
        fx.barrier()

        inv_buf = ptr_buf_tensor(arg_inv, fx.Int32)

        # B. top-k reduce. Payload is 8 fp8 bytes (2 i32) per thread; its 32-value
        # block is u // 4, so each thread reads its own scale byte.
        bases = [_uniform_i64(w.lsa_ptr(rank, layout.off_s)), _uniform_i64(arg_comb_inp)]
        rows = [ptr_buf_tensor(b, fx.Int32, unit_elems=2) for b in bases]
        scales = [ptr_buf_tensor(b, fx.Int8) for b in bases]
        out = ptr_buf_tensor(arg_out, fx.BFloat16, unit_elems=8)
        v2f32 = T.vec(2, T.f32)

        def row_vals(src, row0, u):
            raw = fx.Vector(
                buf_copy_load(rows[src], (row0 + u * fx.Int32(8)) // fx.Int32(8), fx.Int32, unit_elems=2)
            )
            sc = scales[src][row0 + fx.Int32(hidden) + u // fx.Int32(4)]
            scale = (fx.Uint8(sc).to(fx.Uint32).bitcast(fx.Int32) << fx.Int32(23)).bitcast(fx.Float32)
            parts = []
            for word in range_constexpr(2):
                lo = cvt_pk_f32_fp8(res=v2f32, src=raw[word], word_sel=False)
                hi = cvt_pk_f32_fp8(res=v2f32, src=raw[word], word_sel=True)
                parts.append(lo.shuffle(hi, [0, 1, 2, 3]))
            vals = fx.Vector(parts[0]).shuffle(fx.Vector(parts[1]), [0, 1, 2, 3, 4, 5, 6, 7])
            return fx.Vector(vals), scale

        for tok in range(bid, i32_tokens, i32_nblocks):
            slots = []
            for k in range_constexpr(topk):
                slots.append(inv_buf[tok * fx.Int32(topk) + fx.Int32(k)])
            for chunk in range_constexpr(chunks):
                u = fx.Int32(chunk * BLOCK) + tid
                if u < fx.Int32(units):
                    # Branch-free over k, so all top-k row loads are in flight together.
                    acc = fx.Vector.filled(8, 0.0, fx.Float32)
                    for k in range_constexpr(topk):
                        landed = slots[k] >= fx.Int32(0)
                        keep = landed.select(fx.Float32(1.0), fx.Float32(0.0))
                        vals, scale = row_vals(
                            0, landed.select(slots[k], fx.Int32(0)) * fx.Int32(row_bytes), u
                        )
                        acc = acc + vals * (scale * keep)
                    # Rare: rows past the end of S, in comb_inp[slot]. The
                    # slots are the same for the whole block, so this branch is uniform.
                    direct = [slots[k] == fx.Int32(INV_P2P) for k in range_constexpr(topk)]
                    any_direct = direct[0]
                    for k in range_constexpr(1, topk):
                        any_direct = any_direct | direct[k]
                    if any_direct:
                        for k in range_constexpr(topk):
                            slot = direct[k].select(tok * fx.Int32(topk) + fx.Int32(k), fx.Int32(0))
                            pvals, pscale = row_vals(1, slot * fx.Int32(row_bytes), u)
                            keep = direct[k].select(fx.Float32(1.0), fx.Float32(0.0))
                            acc = acc + pvals * (pscale * keep)
                    buf_copy_store(
                        out, tok * fx.Int32(units) + u, acc.to(fx.BFloat16), fx.BFloat16, unit_elems=8
                    )
            # Every thread has the slots in registers; only then may they be reset.
            fx.barrier()
            if tid < fx.Int32(topk):
                inv_buf[tok * fx.Int32(topk) + tid] = fx.Int32(-1)

        # C. the last block out tells the peers this call's data has been consumed.
        # It also puts the arrival count back to zero: every block of this call has
        # counted, and none of the next call's can before this kernel ends, so the
        # count needs no width and no divisibility of 2^32 by the grid.
        fx.rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0, expcnt=0)
        fx.barrier()
        if tid == fx.Int32(0):
            comm_ops.fence_agent_release()
            old = fx.Int32(comm_ops.atomic_add_agent(arg_fin, fx.Int32(1)))
            if old + fx.Int32(1) == i32_nblocks:
                comm_ops.atomic_add_agent(arg_fin, fx.Int32(0) - i32_nblocks)
                comm_ops.fence_agent_acquire()
                comm_ops.store_i64_global(counter_addr, phase)
                comm_ops.fence_system_release()
                for peer in range_constexpr(npes):
                    if peer != rank:
                        comm_ops.store_i64_global_system(
                            fx.Int64(w.lsa_ptr(peer, layout.off_done)) + fx.Int64(rank * 8),
                            phase,
                        )

    # fmt: off
    @flyc.jit
    def launch(win: fx.Int64, arg_inv: fx.Int64, arg_out: fx.Int64, arg_comb_inp: fx.Int64,
        arg_fin: fx.Int64, arg_dev_comm: fx.Int64, i32_nblocks: fx.Int32, i32_tokens: fx.Int32,
        stream: fx.Stream):
    # fmt: on
        combine(win, arg_inv, arg_out, arg_comb_inp, arg_fin, arg_dev_comm,
                i32_nblocks, i32_tokens).launch(
            grid=(i32_nblocks, 1, 1), block=(BLOCK, 1, 1), stream=stream)

    return launch


# --- The host side -----------------------------------------------------------------


def _torch_view(ptr, nbytes, dtype):
    """A torch tensor over ``nbytes`` of device memory at ``ptr`` (not owned)."""

    class _View:
        def __init__(self):
            self.__cuda_array_interface__ = {
                "shape": (nbytes,),
                "typestr": "|u1",
                "data": (ptr, False),
                "version": 3,
            }

    return torch.as_tensor(_View(), device=torch.cuda.current_device()).view(dtype)


def sdma_stage2_unsupported_reason(moe):
    """Why ``moe`` cannot run the SDMA Stage2, or None. Checks the static configuration
    only; a mori without working SDMA queues is detected when the stage is built."""
    arch = torch.cuda.get_device_properties(moe.dev).gcnArchName.split(":")[0]
    if not arch.startswith("gfx95"):
        return f"needs gfx95x, got {arch}"
    if moe._a_dtype != "fp8":
        return "needs a8w4 (fp8 activations)"
    if moe._s1_fixed_slot:
        return "needs the compact (indexed) Stage1 dispatch"
    if moe.world_size > 8 or BLOCK % moe.world_size:
        return f"needs world_size <= 8 dividing {BLOCK}, got {moe.world_size}"
    if moe.model_dim % 256 or moe.inter_dim % 256:
        return "needs model_dim and inter_dim multiples of 256"
    if any(
        e.config.p2p_quant != "fp8_blockwise_1x32" for e in moe._bundle_plan.entries
    ):
        return "needs the fp8_blockwise_1x32 Stage2 P2P format"
    try:
        arena_rows(
            max_tok=moe.mtpr,
            topk=moe.topk,
            hidden=moe.model_dim,
            send_slack=float(_envs.AITER_MEGA_SDMA_SEND_SLACK),
        )
    except RuntimeError as exc:
        return str(exc)
    try:
        import mori.cco.device.flydsl  # noqa: F401
    except ImportError as exc:
        return f"mori cco device API unavailable: {exc}"
    return None


class MegaMoESdmaStage2:
    """Stage2 + combine of one ``MegaMoEV2`` with the transfer on the SDMA engines.

    Owns a cco communicator and window; reads the instance's Stage1 state (sorted
    rows, srcmap, weights, ``expert_tile_end``, the gathered route counts) and its P2P
    ``comb_inp`` for rows past the end of ``S``. ``run`` returns the combined
    ``[max_tok_per_rank, hidden]`` bf16 output. ``send_slack``: rows of ``S`` as a
    multiple of ``max_tok * topk`` (what one rank sends in an even split), capped so
    that ``S`` and ``R`` fit the 32-bit buffer-resource range. Needs
    ``MORI_ENABLE_SDMA=1``; construction raises RuntimeError when the communicator
    connected no SDMA queue.
    """

    def __init__(self, moe, *, chunk_tiles=16, send_slack=4.0):
        import torch.distributed as dist

        if not dist.is_initialized():
            raise RuntimeError(
                "the SDMA Stage2 needs an initialized torch.distributed group"
            )
        if chunk_tiles < 1 or send_slack <= 0:
            raise ValueError(
                f"chunk_tiles={chunk_tiles} and send_slack={send_slack} must be positive"
            )
        self.moe = moe
        self.rank, self.npes = moe.rank, moe.world_size
        self.hidden, self.topk, self.max_tok = moe.model_dim, moe.topk, moe.mtpr
        self.chunk_tiles = int(chunk_tiles)
        self.block_m = 256  # GEMM2 tile rows
        self.slot_cap = self.max_tok * self.topk
        send_rows, recv_rows, row_bytes = arena_rows(
            max_tok=self.max_tok,
            topk=self.topk,
            hidden=self.hidden,
            send_slack=send_slack,
        )
        self.layout = ArenaLayout(
            self.npes, send_rows, recv_rows, row_bytes, self.slot_cap
        )
        self.epr = moe.epr
        self.bigcnt = moe._s1_dispatch_workspace["bigcnt"]
        self.total_segments = self.bigcnt.numel() // self.npes
        self.comm = None

        # The peers' SDMA handles exist only within a host: a put to a rank on another
        # node would dereference a null handle. Every rank sees the same list.
        import socket

        hosts = [None] * self.npes
        dist.all_gather_object(hosts, socket.gethostname())
        if len(set(hosts)) != 1:
            raise RuntimeError(
                f"needs every rank on one host, got {sorted(set(hosts))}"
            )

        # Whatever fails on one rank from here on, every rank learns it before the
        # communicator's barrier, so all of them fall back together instead of the
        # healthy ones waiting there for the one that left.
        error = None
        try:
            self._connect()
        except Exception as exc:  # noqa: BLE001 - reported on every rank below
            error = f"rank {self.rank}: {type(exc).__name__}: {exc}"
        errors = [None] * self.npes
        dist.all_gather_object(errors, error)
        errors = [e for e in errors if e is not None]
        if errors:
            self._release()
            raise RuntimeError("; ".join(errors))
        self._build_state()
        torch.cuda.synchronize()
        self.comm.barrier()

    def _connect(self):
        """Communicator, window and dev-comm. Raises if no SDMA queue connected: a put
        on a communicator without one moves no bytes."""
        import torch.distributed as dist
        from mori.cco import GDA_CONNECTION_NONE, CCODevCommRequirements, Communicator

        lay = self.layout
        uid = [Communicator.get_unique_id() if self.rank == 0 else None]
        dist.broadcast_object_list(uid, src=0)
        per_rank_vmm = max(
            Communicator.DEFAULT_PER_RANK_VMM, _align(lay.total_bytes) + (256 << 20)
        )
        self.comm = Communicator.init(
            self.npes, self.rank, uid[0], per_rank_vmm=per_rank_vmm
        )
        self.mem = self.comm.alloc_mem(lay.total_bytes)
        self.win = self.comm.register_window(self.mem.ptr, self.mem.size)
        _torch_view(self.mem.ptr, lay.total_bytes, torch.uint8).zero_()
        reqs = CCODevCommRequirements()
        reqs.gda_connection_type = GDA_CONNECTION_NONE
        reqs.gda_signal_count = 0
        reqs.gda_counter_count = 0
        # One channel per peer: KFD recommends one engine per peer link, and the
        # completion signal must ride the channel the data went out on.
        reqs.sdma_queue_count = 1
        self.dev_comm = self.comm.create_dev_comm(reqs)
        # mori before DevCommHandle.sdma_num_queue cannot tell, and is treated as
        # unsupported.
        queues = getattr(self.dev_comm, "sdma_num_queue", None)
        if queues is None or queues < 1:
            raise RuntimeError(
                "mori too old to report its SDMA queues"
                if queues is None
                else "cco connected no SDMA queue (mori built with BUILD_CCO_SDMA=OFF, "
                "MORI_ENABLE_SDMA unset, or no peer reports SDMA)"
            )

    def _release(self):
        if self.comm is not None:
            self.comm.close_all_resources()
            self.comm = None

    def _build_state(self):
        moe, lay, dev = self.moe, self.layout, self.moe.dev
        nvm = int(moe._s1_nvm)
        self.max_tiles = nvm // MIN_SBM + 2
        i32 = {"dtype": torch.int32, "device": dev}
        self.out_row = torch.full((nvm + 4 * 128,), -1, **i32)
        self.tile_cnt = torch.zeros(self.max_tiles * self.npes, **i32)
        self.tile_base = torch.zeros((self.max_tiles + 1) * self.npes, **i32)
        # [run rows | send_base | land_off | avail] per pe, then the count-mismatch counter.
        self.cursor = torch.zeros(4 * self.npes + 1, **i32)
        self.cdone = torch.zeros((self.max_tiles + 1) * CDONE_STRIDE, **i32)
        n_cu = torch.cuda.get_device_properties(dev).multi_processor_count
        # The plan's never-reset barrier counters (i64), and the combine's arrival
        # count, which its last block puts back to zero.
        self.plan_bar = torch.zeros(
            (n_cu // BAR_GROUP + 2) * BAR_STRIDE, dtype=torch.int64, device=dev
        )
        self.combine_fin = torch.zeros(1, **i32)
        # Rows written through comb_inp because they did not fit in S.
        self.overflow = torch.zeros(1, **i32)
        self.inv = _torch_view(
            self.mem.ptr + lay.off_inv, self.slot_cap * 4, torch.int32
        )
        self.inv.fill_(-1)
        self.idx_send = torch.zeros(lay.send_rows, **i32)
        self.out = torch.empty(
            self.max_tok, self.hidden, dtype=torch.bfloat16, device=dev
        )
        # The plan barriers across its grid, so it stays resident: one block per CU.
        # The combine has no rendezvous; two blocks per CU keep more row loads in flight.
        self.plan_blocks = n_cu
        self.comb_blocks = 2 * n_cu
        self.gemm_ctas = n_cu
        self._plans = {}
        self._gemm2 = {}
        self._combine = build_combine(
            layout=lay, rank=self.rank, hidden=self.hidden, topk=self.topk
        )

    def _plan_for(self, sbm):
        if sbm < MIN_SBM:
            raise ValueError(
                f"sort_block_m={sbm} is below the supported minimum {MIN_SBM}"
            )
        if sbm not in self._plans:
            self._plans[sbm] = build_plan(
                SBM=sbm,
                topk=self.topk,
                log2_max_tok=self.max_tok.bit_length() - 1,
                recv_cap=int(self.moe.max_recv),
                npes=self.npes,
                rank=self.rank,
                epr=self.epr,
                total_segments=self.total_segments,
                send_rows=self.layout.send_rows,
            )
        return self._plans[sbm]

    def _gemm2_for(self, sbm):
        from .gemm2_8wave import compile_gemm2_8wave_sdma

        if sbm not in self._gemm2:
            moe = self.moe
            self._gemm2[sbm] = compile_gemm2_8wave_sdma(
                K=moe.inter_dim,
                N=moe.model_dim,
                experts=moe.epr,
                topk=moe.topk,
                npes=moe.world_size,
                max_tok=moe.mtpr,
                recv_cap=int(moe.max_recv),
                SBM=sbm,
                rank=self.rank,
                layout=self.layout,
                chunk_tiles=self.chunk_tiles,
                grid_ctas=self.gemm_ctas,
                comb_bytes=self.max_tok * self.topk * self.layout.row_bytes,
            )
        return self._gemm2[sbm]

    def _plan_args(self, s_fx):
        moe, op, lay = self.moe, self.moe._s1_op, self.layout
        n_chunks = (self.max_tiles + self.chunk_tiles - 1) // self.chunk_tiles + 1
        return (
            fx.Int64(op.srcmap_em.data_ptr()),
            fx.Int64(op.tile_row_base.data_ptr()),
            fx.Int64(op.num_valid.data_ptr()),
            fx.Int64(self.tile_cnt.data_ptr()),
            fx.Int64(self.tile_base.data_ptr()),
            fx.Int64(self.cursor.data_ptr()),
            fx.Int64(self.out_row.data_ptr()),
            fx.Int64(self.idx_send.data_ptr()),
            fx.Int64(self.inv.data_ptr()),
            fx.Int64(self.cdone.data_ptr()),
            fx.Int64(self.plan_bar.data_ptr()),
            fx.Int64(self.overflow.data_ptr()),
            fx.Int64(self.mem.ptr + lay.off_done),
            fx.Int64(self.mem.ptr + lay.off_counter),
            fx.Int64(self.bigcnt.data_ptr()),
            fx.Int64(moe._s1_dispatch_workspace["fanout_pair_config"].data_ptr()),
            fx.Int64(moe._s1_epoch_parity.data_ptr()),
            fx.Int32(self.plan_blocks),
            fx.Int32(min(self.cdone.numel(), n_chunks * CDONE_STRIDE)),
            s_fx,
        )

    def _gemm2_args(self, s_fx):
        moe, op, lay = self.moe, self.moe._s1_op, self.layout
        nvm = int(moe._s1_nvm)
        return (
            # A and B by address: the Stage1 output can pass the 2^31 elements a
            # tensor argument's i32 size holds (see gemm2_8wave.fp8_buffer_at).
            fx.Int64(moe._s1_out.data_ptr()),
            fx.Int64(moe.w2.data_ptr()),
            moe._s1_osd.view(-1),
            moe.w2_scale.view(torch.uint8).view(-1),
            (nvm + 255) // 256 * 256,
            fx.Int64(moe._s1_dispatch_workspace["expert_tile_end"].data_ptr()),
            fx.Int64(op.srcmap_em.data_ptr()),
            fx.Int64(op.wts_em.data_ptr()),
            fx.Int64(op.tile_row_base.data_ptr()),
            moe.comb_op._fx_p2p_comb_inp,
            fx.Int64(op.num_valid.data_ptr()),
            fx.Int64(self.out_row.data_ptr()),
            fx.Int64(self.mem.ptr + lay.off_s),
            fx.Int64(self.dev_comm.ptr),
            fx.Int64(self.win.handle),
            fx.Int64(self.cdone.data_ptr()),
            fx.Int64(self.tile_base.data_ptr()),
            fx.Int64(self.cursor.data_ptr()),
            fx.Int64(self.idx_send.data_ptr()),
            s_fx,
        )

    def _combine_args(self, run_tokens, s_fx):
        return (
            fx.Int64(self.win.handle),
            fx.Int64(self.inv.data_ptr()),
            fx.Int64(self.out.data_ptr()),
            fx.Int64(self.moe.comb_op.shmem_comb_inp_tok.data_ptr()),
            fx.Int64(self.combine_fin.data_ptr()),
            fx.Int64(self.dev_comm.ptr),
            fx.Int32(self.comb_blocks),
            fx.Int32(run_tokens),
            s_fx,
        )

    def preload(self):
        """Compile every kernel this instance can launch -- the plan and GEMM2 for each
        sort block of the bundle plan, and the combine -- without dispatching any.

        Otherwise each compiles on the first call that reaches its sort block: a
        multi-second stall while serving, at a time that differs per rank, possibly
        inside a CUDA graph capture.
        """
        from ..tensor_shim import _preload_compiled

        s_fx = fx.Stream(torch.cuda.current_stream().cuda_stream)
        for sbm in sorted(
            {int(e.config.stage1.sort_block_m) for e in self.moe._bundle_plan.entries}
        ):
            _preload_compiled(self._plan_for(sbm), *self._plan_args(s_fx))
            _preload_compiled(self._gemm2_for(sbm), *self._gemm2_args(s_fx))
        _preload_compiled(self._combine, *self._combine_args(self.max_tok, s_fx))

    def run(self, run_tokens, config, stream):
        """Plan, GEMM2 + SDMA, combine on ``stream`` (a torch stream). Returns the
        ``[max_tok_per_rank, hidden]`` output; rows ``[0, run_tokens)`` are valid.

        Reads only ``config.stage1.sort_block_m``: the GEMM2 tile is fixed at 256x256,
        so ``config.stage2``'s tile and persistence settings do not apply. Top-k slots
        with id -1 need no mask: they are never dispatched, so they have no row and
        their ``inv`` stays -1.
        """
        sbm = int(config.stage1.sort_block_m)
        s_fx = fx.Stream(stream.cuda_stream)
        _run_compiled(self._plan_for(sbm), *self._plan_args(s_fx))
        _run_compiled(self._gemm2_for(sbm), *self._gemm2_args(s_fx))
        _run_compiled(self._combine, *self._combine_args(run_tokens, s_fx))
        return self.out

    def take_overflow_rows(self):
        """Rows that took the ``comb_inp`` path since the last call (synchronizes).
        Raises if a plan found its row counts differ from the gathered ones (the exact
        layout would then be wrong)."""
        torch.cuda.synchronize()
        rows = int(self.overflow.item())
        self.overflow.zero_()
        bad = int(self.cursor[4 * self.npes].item())
        if bad:
            raise RuntimeError(
                f"rank {self.rank}: {bad} per-rank row counts differ from the gathered route counts"
            )
        return rows

    def close(self):
        """Free the window once every peer has read what this rank last sent.

        Collective: every rank must call it, at the same point, after its last call.
        A rank that freed its window while a peer's copy engine still wrote into it
        would take a page fault on that peer.
        """
        import logging
        import time

        import torch.distributed as dist

        torch.cuda.synchronize()
        dist.barrier()
        lay = self.layout
        calls = int(_torch_view(self.mem.ptr + lay.off_counter, 8, torch.int64).item())
        done = _torch_view(self.mem.ptr + lay.off_done, self.npes * 8, torch.int64)
        deadline = time.monotonic() + 30.0
        pending = []
        while True:
            pending = [
                r for r, f in enumerate(done.tolist()) if r != self.rank and f < calls
            ]
            if not pending or time.monotonic() >= deadline:
                break
            time.sleep(0.001)
        if pending:
            logging.getLogger(__name__).warning(
                "MegaMoE SDMA Stage2 rank %d: peers %s have not read call %d after 30 s; "
                "freeing the window anyway",
                self.rank,
                pending,
                calls,
            )
        self.comm.barrier()
        self._release()
