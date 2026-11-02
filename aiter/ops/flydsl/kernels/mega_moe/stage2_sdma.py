# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""MegaMoEV2 Stage2 + combine with the transfer done by the SDMA copy engines.

Ported from mori's ``examples/cco/python/11_flydsl_mega_moe_sdma`` (ROCm/mori#859),
with the 8-wave GEMM2 of ``gemm2_8wave.py`` and a fallback for rows the per-pair
reservation cannot hold.

The default Stage2 scatters every GEMM2 row straight into the origin rank's
``comb_inp``, so the GEMM's stores wait on the XGMI links. Here a row goes to a
local send buffer ``S`` grouped by origin rank, and a copy engine moves one
contiguous run per peer while the GEMM is still computing::

    S = [ origin rank 0: rows... | origin rank 1: rows... | ... ]   (cap rows each)

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
row, counted from the start of ``S`` (``S`` rows ``[0, npes*cap)``, ``R`` rows
``[npes*cap, 2*npes*cap)``; my own run is read from ``S`` in place), or
``INV_P2P`` for a row that came through ``comb_inp``, or -1. The sender writes it,
so no index travels with the data. ``flags[src]`` says "src's data for call n has
landed", ``done[dst]`` "dst has finished reading call n" (a sender waits for it
before overwriting that peer's ``R`` / ``inv``), and ``counter`` is the call number,
advanced only by the combine.

``cap`` is the rows reserved for ONE (src, dst) pair: a sender writes into the
peer's ``R`` at a statically known offset with no handshake. Reserving the worst
case (every token of a peer routing all of top-k here) costs ``npes`` times the
window, so ``cap`` is the even share times ``row_slack``. A row past it is not
dropped: the plan marks it ``OUT_ROW_P2P`` and GEMM2 writes it straight into the
origin rank's ``comb_inp[slot]`` (the default Stage2's layout) and sets the peer's
``inv[slot]`` to ``INV_P2P``. Skewed routing therefore stays correct; only the
rows past the reservation pay for the direct path.

Rows are fp8 with one e8m0 scale per 32 values, payload then scales, as the
default Stage2's P2P scatter lays them out, so one contiguous put carries a row.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.expr import range_constexpr, rocdl
from flydsl.expr.rocdl import cvt_pk_f32_fp8
from flydsl.expr.typing import Int8, T

from .. import communication_ops_utils as comm_ops
from ..buffer_ops import buffer_load, buffer_store, create_buffer_resource_from_addr
from ..mxfp4_gemm_common import global_typed_ptr, lds_typed_ptr
from ..tensor_shim import _run_compiled, buf_copy_load, buf_copy_store, ptr_buf_tensor

__all__ = ["MegaMoESdmaStage2", "sdma_stage2_unsupported_reason"]

#: i32 slots between consecutive chunk-completion counters (one 128 B line each:
#: every unit of a chunk bumps its counter).
CDONE_STRIDE = 32
#: Rows per lane per loop trip when a comm wave pushes inv entries to a peer.
PUSH_U = 8
#: ``out_row`` of a valid row past the per-pair reservation: written through ``comb_inp``.
OUT_ROW_P2P = -2
#: ``inv`` of a slot whose row is in this rank's ``comb_inp``.
INV_P2P = -2

BLOCK = 256
WAVE = 64
BAR_STRIDE = 32  # ints per tree_barrier counter: one 128 B cache line
BAR_GROUP = 32  # blocks per tree_barrier group
MIN_SBM = 16
_ALIGN = 128


def _align(n):
    return (n + _ALIGN - 1) // _ALIGN * _ALIGN


class ArenaLayout:
    """Byte offsets inside the cco window; identical on every rank."""

    def __init__(self, npes, cap, row_bytes, slot_cap):
        self.npes, self.cap, self.row_bytes, self.slot_cap = (
            npes,
            cap,
            row_bytes,
            slot_cap,
        )
        self.n_rows = npes * cap
        self.off_s = 0
        self.off_r = self.n_rows * row_bytes
        self.off_inv = self.off_r + self.n_rows * row_bytes
        self.off_sig = self.off_inv + _align(slot_cap * 4)
        self.off_flags = self.off_sig + _align(npes * 4)
        self.off_done = self.off_flags + _align(npes * 4)
        self.off_counter = self.off_done + _align(npes * 4)
        self.total_bytes = self.off_counter + _ALIGN
        # The combine addresses S and R through one 32-bit buffer-resource offset.
        if 2 * self.n_rows * row_bytes >= 1 << 31:
            raise ValueError(
                f"SDMA Stage2 send+landing buffers ({2 * self.n_rows * row_bytes} B) exceed "
                "the 32-bit buffer-resource range; lower row_slack or max_tok_per_rank"
            )

    def key(self):
        return (self.npes, self.cap, self.row_bytes, self.slot_cap)


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


# --- The sender side of the SDMA protocol (called by the GEMM2 comm waves) ------


def sdma_wait_consumed(w, peer, phase, *, layout, rank):
    """Block until ``peer`` has finished reading the previous call's data from my sends."""
    comm_ops.spin_until_ge_i32_system(
        fx.Int64(w.lsa_ptr(rank, layout.off_done)) + fx.Int64(peer) * fx.Int64(4),
        phase - fx.Int32(1),
        acquire=True,
    )


def sdma_put_run(
    sdma, win, peer, queue, *, layout, rank, lo, hi, cco, remote_signal=False
):
    """Ship rows ``[lo, hi)`` of my run for ``peer`` with one put.

    No local signal (nothing waits on it). ``remote_signal`` hangs the completion
    signal on this put: the copy engine bumps the peer's counter once the bytes
    are there, and the peer's combine polls it. A queue is a FIFO its engine
    drains in order, so a signal on the last put covers every put before it.
    """
    rb, cap = layout.row_bytes, layout.cap
    sdma.put(
        peer,
        win,
        fx.Int64(layout.off_r + rank * cap * rb) + fx.Int64(lo) * fx.Int64(rb),
        win,
        fx.Int64(layout.off_s)
        + (fx.Int64(peer) * fx.Int64(cap) + fx.Int64(lo)) * fx.Int64(rb),
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
    comm_ops.store_i32_global_system_release(
        fx.Int64(w.lsa_ptr(peer, layout.off_flags)) + fx.Int64(rank * 4), phase
    )


# --- Kernel 1: plan -------------------------------------------------------------


def tree_barrier(bar_addr, nblocks, tid, bid):
    """All blocks of a resident grid rendezvous with two levels of small atomics.

    Blocks are grouped ``BAR_GROUP`` to a counter; the last block of a group bumps a
    top counter; everyone polls only the top one. Counters are never reset: a
    group's counter advances by its size per launch, so the launch number is
    ``old // size``. Each counter sits on its own cache line.
    """
    group = bid // fx.Int32(BAR_GROUP)
    n_groups = (nblocks + fx.Int32(BAR_GROUP - 1)) // fx.Int32(BAR_GROUP)
    size = nblocks - group * fx.Int32(BAR_GROUP)
    size = (size < fx.Int32(BAR_GROUP)).select(size, fx.Int32(BAR_GROUP))
    rocdl.s_waitcnt(0)
    fx.barrier()

    @flyc.jit
    def arrive_and_wait():
        if tid == fx.Int32(0):
            comm_ops.fence_agent_release()
            old = fx.Int32(
                comm_ops.atomic_add_agent(
                    bar_addr + fx.Int64(group) * fx.Int64(BAR_STRIDE * 4), fx.Int32(1)
                )
            )
            launch = old // size
            if old - launch * size == size - fx.Int32(1):
                comm_ops.atomic_add_agent(
                    bar_addr + fx.Int64(n_groups) * fx.Int64(BAR_STRIDE * 4),
                    fx.Int32(1),
                )
            comm_ops.spin_until_ge_i32_agent(
                bar_addr + fx.Int64(n_groups) * fx.Int64(BAR_STRIDE * 4),
                (launch + fx.Int32(1)) * n_groups,
            )
            comm_ops.fence_agent_acquire()

    arrive_and_wait()
    fx.barrier()


def build_plan(*, SBM, topk, log2_max_tok, recv_cap, npes, cap, rank):
    """Kernel 1: give every valid sorted row a stable index in its rank's run.

    * count:  ``tile_cnt[tile][pe]`` = rows of the sort tile bound for ``pe``
    * prefix: ``tile_base[tile][pe]`` = rows for ``pe`` in earlier tiles (so the rows
              of a tile range are contiguous in the run), ``cursor[pe]`` = total
    * assign: ``out_row = pe*cap + tile_base + (earlier rows of this tile for pe)``
              and ``idx_send[out_row] = origin_lid*topk + k``. Rows bound for this rank
              also get their ``inv`` entry here; the peers' entries are pushed by
              the GEMM2 comm waves. A row past ``cap`` gets ``OUT_ROW_P2P``.

    One grid-wide rendezvous, between count and prefix. Before writing anything the
    kernel waits until every peer has consumed the previous call: with the copy
    engine's FIFO that means this rank's puts have finished reading ``S``.
    ``tile_base`` is clamped to ``cap``, so a chunk never ships past the reservation.
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
    lds_bytes = (OFF_PREF + npes) * 4

    @fx.struct
    class PlanShared:
        buf: fx.Array[Int8, lds_bytes, 16]

    # fmt: off
    @flyc.kernel(name=f"mega_moe_sdma_plan_sbm{SBM}_k{topk}_p{npes}_c{cap}_r{rank}",
                 known_block_size=[BLOCK, 1, 1])
    def plan(arg_stids: fx.Int64, arg_trb: fx.Int64, arg_num_valid: fx.Int64,
        arg_tile_cnt: fx.Int64, arg_tile_base: fx.Int64, arg_cursor: fx.Int64,
        arg_out_row: fx.Int64, arg_idx: fx.Int64, arg_inv: fx.Int64, arg_cdone: fx.Int64,
        arg_bar: fx.Int64, arg_over: fx.Int64, arg_done: fx.Int64, arg_counter: fx.Int64,
        i32_nblocks: fx.Int32, i32_cdone_n: fx.Int32):
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

        # S and the peers' inv are rewritten from here on: wait until every peer has
        # read what I sent it last call.
        consumed = global_typed_ptr(arg_counter, T.i32)[0]
        if (tid < fx.Int32(npes)) & (tid != fx.Int32(rank)):
            comm_ops.spin_until_ge_i32_system(
                fx.Int64(arg_done) + fx.Int64(tid) * fx.Int64(4), consumed, acquire=True
            )
        fx.barrier()

        # The GEMM2 kernel's per-chunk completion counters start from zero every call.
        for g in range(gid, i32_cdone_n, stride):
            cdone[g] = fx.Int32(0)

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
            ptr_buf_tensor(arg_cursor, fx.Int32)[tid] = fx.Int32(0)

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
                tile_base[tile * fx.Int32(npes) + tid] = (pre < fx.Int32(cap)).select(
                    pre, fx.Int32(cap)
                )
                # The extra row holds the totals the comm waves read as a chunk's end.
                if tile == tiles - fx.Int32(1):
                    total = pre + tile_cnt[tile * fx.Int32(npes) + tid]
                    capped = (total < fx.Int32(cap)).select(total, fx.Int32(cap))
                    tile_base[tiles * fx.Int32(npes) + tid] = capped
                    ptr_buf_tensor(arg_cursor, fx.Int32)[tid] = capped
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
                if place >= fx.Int32(cap):
                    # Past the per-pair reservation: GEMM2 writes it through comb_inp.
                    out_buf[g] = fx.Int32(OUT_ROW_P2P)
                    comm_ops.atomic_add_agent(arg_over, fx.Int32(1))
                else:
                    out_row = dest_pe * fx.Int32(cap) + place
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
        i32_nblocks: fx.Int32, i32_cdone_n: fx.Int32, stream: fx.Stream):
    # fmt: on
        plan(arg_stids, arg_trb, arg_num_valid, arg_tile_cnt, arg_tile_base, arg_cursor,
             arg_out_row, arg_idx, arg_inv, arg_cdone, arg_bar, arg_over, arg_done, arg_counter,
             i32_nblocks, i32_cdone_n).launch(
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
    @flyc.kernel(name=f"mega_moe_sdma_combine_p{npes}_c{layout.cap}_h{hidden}_k{topk}_r{rank}",
                 known_block_size=[BLOCK, 1, 1])
    def combine(win: fx.Int64, arg_inv: fx.Int64, arg_out: fx.Int64, arg_comb_inp: fx.Int64,
        arg_fin: fx.Int64, arg_dev_comm: fx.Int64, i32_nblocks: fx.Int32, i32_tokens: fx.Int32):
    # fmt: on
        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        w = cco.Window(win)
        rsrc_counter = create_buffer_resource_from_addr(
            _uniform_i64(w.lsa_ptr(rank, layout.off_counter))
        )
        # Advanced only by the last block below, after every block has read it.
        phase = fx.Int32(buffer_load(rsrc_counter, 0, vec_width=1, dtype=T.i32)) + fx.Int32(1)
        flags_base = fx.Int64(w.lsa_ptr(rank, layout.off_flags))

        # A. the peers' data has landed
        if (tid < fx.Int32(npes)) & (tid != fx.Int32(rank)):
            comm_ops.spin_until_ge_i32_system(
                flags_base + fx.Int64(tid) * fx.Int64(4), phase, acquire=True
            )
            # The sender's copy engine bumps this once per call when its last put lands.
            cco.DevComm(arg_dev_comm).sdma().wait_signal(tid, 0, fx.Int64(phase))
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
                    # Rare: rows past the per-pair reservation, in comb_inp[slot]. The
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

        # C. the last block out tells the peers this call's data has been consumed
        rocdl.s_waitcnt(0)
        fx.barrier()
        if tid == fx.Int32(0):
            comm_ops.fence_agent_release()
            old = fx.Int32(comm_ops.atomic_add_agent(arg_fin, fx.Int32(1)))
            if (old + fx.Int32(1)) % i32_nblocks == fx.Int32(0):
                comm_ops.fence_agent_acquire()
                buffer_store(phase, rsrc_counter, 0)
                comm_ops.fence_system_release()
                for peer in range_constexpr(npes):
                    if peer != rank:
                        comm_ops.store_i32_global_system_release(
                            fx.Int64(w.lsa_ptr(peer, layout.off_done)) + fx.Int64(rank * 4), phase
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
        import mori.cco.device.flydsl  # noqa: F401
    except ImportError as exc:
        return f"mori cco device API unavailable: {exc}"
    return None


class MegaMoESdmaStage2:
    """Stage2 + combine of one ``MegaMoEV2`` with the transfer on the SDMA engines.

    Owns a cco communicator and window; reads the instance's Stage1 state (sorted
    rows, srcmap, weights, ``expert_tile_end``) and its P2P ``comb_inp`` for rows
    past the reservation. ``run`` returns the combined ``[max_tok_per_rank, hidden]``
    bf16 output. Needs ``MORI_ENABLE_SDMA=1``; construction raises RuntimeError
    when the communicator connected no SDMA queue.
    """

    def __init__(self, moe, *, chunk_tiles=16, row_slack=2.0):
        import torch.distributed as dist
        from mori.cco import GDA_CONNECTION_NONE, CCODevCommRequirements, Communicator

        if not dist.is_initialized():
            raise RuntimeError(
                "the SDMA Stage2 needs an initialized torch.distributed group"
            )
        if chunk_tiles < 1 or row_slack <= 0:
            raise ValueError(
                f"chunk_tiles={chunk_tiles} and row_slack={row_slack} must be positive"
            )
        self.moe = moe
        self.rank, self.npes = moe.rank, moe.world_size
        self.hidden, self.topk, self.max_tok = moe.model_dim, moe.topk, moe.mtpr
        self.chunk_tiles = int(chunk_tiles)
        self.block_m = 256  # GEMM2 tile rows
        self.slot_cap = self.max_tok * self.topk
        share = -(-self.slot_cap // self.npes)
        self.cap = -(-int(share * row_slack) // 128) * 128
        self.layout = ArenaLayout(
            self.npes, self.cap, self.hidden + self.hidden // 32, self.slot_cap
        )
        lay = self.layout
        dev = moe.dev

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
        # A put on a communicator without SDMA queues moves no bytes, so this must hold
        # before anything is launched. mori before DevCommHandle.sdma_num_queue cannot
        # tell, and is treated as unsupported.
        queues = getattr(self.dev_comm, "sdma_num_queue", None)
        if queues is None or queues < 1:
            self.comm.close_all_resources()
            raise RuntimeError(
                "mori too old to report its SDMA queues"
                if queues is None
                else "cco connected no SDMA queue (mori built with BUILD_CCO_SDMA=OFF, "
                "MORI_ENABLE_SDMA unset, or no peer reports SDMA)"
            )

        nvm = int(moe._s1_nvm)
        self.max_tiles = nvm // MIN_SBM + 2
        i32 = {"dtype": torch.int32, "device": dev}
        self.out_row = torch.full((nvm + 4 * 128,), -1, **i32)
        self.tile_cnt = torch.zeros(self.max_tiles * self.npes, **i32)
        self.tile_base = torch.zeros((self.max_tiles + 1) * self.npes, **i32)
        self.cursor = torch.zeros(self.npes, **i32)
        self.cdone = torch.zeros((self.max_tiles + 1) * CDONE_STRIDE, **i32)
        n_cu = torch.cuda.get_device_properties(dev).multi_processor_count
        # Never-reset counters: the plan's barrier and the combine's last-block count.
        self.plan_bar = torch.zeros(n_cu * BAR_STRIDE, **i32)
        self.combine_fin = torch.zeros(1, **i32)
        # Rows written through comb_inp because they did not fit the reservation.
        self.overflow = torch.zeros(1, **i32)
        self.inv = _torch_view(
            self.mem.ptr + lay.off_inv, self.slot_cap * 4, torch.int32
        )
        self.inv.fill_(-1)
        self.idx_send = torch.zeros(self.npes * self.cap, **i32)
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
        torch.cuda.synchronize()
        self.comm.barrier()

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
                cap=self.cap,
                rank=self.rank,
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

    def run(self, run_tokens, config, stream):
        """Plan, GEMM2 + SDMA, combine on ``stream`` (a torch stream). Returns the
        ``[max_tok_per_rank, hidden]`` output; rows ``[0, run_tokens)`` are valid."""
        moe, op, lay = self.moe, self.moe._s1_op, self.layout
        sbm = int(config.stage1.sort_block_m)
        s_fx = fx.Stream(stream.cuda_stream)
        n_chunks = (self.max_tiles + self.chunk_tiles - 1) // self.chunk_tiles + 1
        _run_compiled(
            self._plan_for(sbm),
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
            fx.Int32(self.plan_blocks),
            fx.Int32(min(self.cdone.numel(), n_chunks * CDONE_STRIDE)),
            s_fx,
        )
        nvm = int(moe._s1_nvm)
        _run_compiled(
            self._gemm2_for(sbm),
            moe._s1_out.view(torch.uint8).view(-1),
            moe.w2.view(torch.uint8).view(-1),
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
        _run_compiled(
            self._combine,
            fx.Int64(self.win.handle),
            fx.Int64(self.inv.data_ptr()),
            fx.Int64(self.out.data_ptr()),
            fx.Int64(moe.comb_op.shmem_comb_inp_tok.data_ptr()),
            fx.Int64(self.combine_fin.data_ptr()),
            fx.Int64(self.dev_comm.ptr),
            fx.Int32(self.comb_blocks),
            fx.Int32(run_tokens),
            s_fx,
        )
        return self.out

    def take_overflow_rows(self):
        """Rows that took the ``comb_inp`` path since the last call (synchronizes)."""
        torch.cuda.synchronize()
        rows = int(self.overflow.item())
        self.overflow.zero_()
        return rows

    def close(self):
        """Wait until every peer has read what this rank last sent, then free the window."""
        import time

        torch.cuda.synchronize()
        lay = self.layout
        calls = int(_torch_view(self.mem.ptr + lay.off_counter, 4, torch.int32).item())
        done = _torch_view(self.mem.ptr + lay.off_done, self.npes * 4, torch.int32)
        deadline = time.monotonic() + 30.0
        while time.monotonic() < deadline:
            if all(f >= calls for r, f in enumerate(done.tolist()) if r != self.rank):
                break
            time.sleep(0.001)
        self.comm.close_all_resources()
