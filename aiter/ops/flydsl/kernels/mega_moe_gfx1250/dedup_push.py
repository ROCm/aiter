# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Dedup'd combine push for the gemm2-fused EP path (AITER_EP_DEDUP).

gemm2 stages every route-weighted row in this rank's ``grouped_out``. For a recv
slot ``t`` its ``c`` local rows are ``topids_to_rows[t*topk + k] >= 0``; their sum
goes out as ONE row, to the origin's ``comb_inp[origin_lid*topk + lead_k]`` where
``lead_k`` is the first local ``k``. The receiving reduce sums only those lead
slots, so a (token, rank) pair costs one row on the wire instead of ``c``.

One wave pushes one token: no LDS and no barrier, so the four waves of a block
and the pushers living in the gemm2 grid all run independently.
"""

import functools
import os

import torch
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl, tdm_ops
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec
from flydsl.expr.arith import _to_raw as _raw

from ..buffer_ops import buffer_load, buffer_store, create_buffer_resource_from_addr
from ..gemm_common_gfx1250 import make_lds_copy_ops
from .tdm_gather_shim import make_tensor_gather_descriptor, tensor_store_gather
from ..communication_ops_utils import (
    atomic_add_agent,
    atomic_add_lds,
    load_i32_global_agent,
    load_i32_lds,
    store_i32_lds,
    traced,
)

WAVE = 32
# Elements one wave moves per b128 instruction: 32 lanes x 8 bf16.
WAVE_CHUNK = WAVE * 8
# cpol SCOPE_DEV: the rows were staged by other XCDs, so miss this XCD's L2.
SCOPE_DEV = 16
# Rows of one token loaded together; c <= 4 covers almost every token at EP4.
ROW_BATCH = int(os.environ.get("AITER_EP_DEDUP_ROW_BATCH", "4"))
# Staging rows per pusher wave, i.e. its sends in flight: a 14 KB peer write
# takes long to retire, and two in flight leave the wave waiting on them.
NBUF = int(os.environ.get("AITER_EP_DEDUP_NBUF", "5"))
# Timing-only knobs: skip the pusher's writes / its whole per-token work.
NO_STORE = int(os.environ.get("AITER_EP_DEDUP_NO_STORE", "0"))
NO_PUSH = int(os.environ.get("AITER_EP_DEDUP_NO_PUSH", "0"))
NO_POLL = int(os.environ.get("AITER_EP_DEDUP_NO_POLL", "0"))
# TDM stores per pushed row, sent in a per-token rotated order.
PIECES = int(os.environ.get("AITER_EP_DEDUP_PIECES", "4"))
# Timing-only: send the summed row back into this rank's own staging row.
STORE_LOCAL = int(os.environ.get("AITER_EP_DEDUP_STORE_LOCAL", "0"))
# 0: count tiles, 2: timing-only, no atomic.
COUNT_MODE = int(os.environ.get("AITER_EP_DEDUP_COUNT_MODE", "0"))
# Pusher read cpol; an experiment knob, SCOPE_DEV is the one that is coherent.
LOAD_CPOL = int(os.environ.get("AITER_EP_DEDUP_LOAD_CPOL", str(SCOPE_DEV)))
# The knobs above shape the emitted code; kernel names carry them so a cached
# binary built under other settings is never picked up.
KNOBS_TAG = (
    f"nb{NBUF}rb{ROW_BATCH}c{LOAD_CPOL}{'ns' if NO_STORE else ''}{'np' if NO_PUSH else ''}"
    f"{'npl' if NO_POLL else ''}{f'cm{COUNT_MODE}' if COUNT_MODE else ''}"
    f"{'sl' if STORE_LOCAL else ''}pc{os.environ.get('AITER_EP_DEDUP_PIECES', '4')}"
    f"cb{os.environ.get('AITER_EP_DEDUP_COL_NBUF', '4')}"
    f"cp{os.environ.get('AITER_EP_DEDUP_COLS_PER_PUSH', '4')}"
    f"bo{os.environ.get('AITER_EP_DEDUP_BACKOFF', '32')}"
)

# dedup state words, one i32 buffer per device:
#   [MTILE_OFF + m]   N tiles of gemm2 M tile m staged so far
#   [KEY_OFF + t]     recv slot t's last M tile, the one it waits for
#   [ORDER_OFF + p]   recv slots sorted by that key, and [SKEY_OFF + p] their keys
MTILE_OFF = 16
MAX_MTILES = 4096
MAX_RECV_CAP = 1 << 16
KEY_OFF = MTILE_OFF + MAX_MTILES
ORDER_OFF = KEY_OFF + MAX_RECV_CAP
SKEY_OFF = ORDER_OFF + MAX_RECV_CAP
STATE_WORDS = SKEY_OFF + MAX_RECV_CAP


def ep_dedup_enabled() -> bool:
    return os.environ.get("AITER_EP_DEDUP", "0") in ("1", "true", "True")


def ep_dedup_pushers() -> int:
    """Pusher workgroups in the gemm2 grid; 0 pushes from a kernel after gemm2."""
    return int(os.environ.get("AITER_EP_DEDUP_PUSHERS", "96"))


@functools.cache
def dedup_state_buffer(device: str) -> torch.Tensor:
    """The order kernel zeroes the tile counters each layer, so no memset."""
    return torch.zeros(STATE_WORDS, dtype=torch.int32, device=device)


def _I16_GLOBAL():
    return fx.PointerType.get(T.i16, address_space=fx.AddressSpace.Global, alignment=16)


def _I16_SHARED():
    return fx.PointerType.get(T.i16, address_space=fx.AddressSpace.Shared, alignment=16)


def push_lds_bytes(hidden, waves):
    """LDS the pushers of one workgroup need: NBUF staging rows per wave."""
    return waves * NBUF * hidden * 2


def emit_push_token(
    t,
    lane,
    lds_ptr,
    lds_buf,
    *,
    out_rsrc,
    out_base,
    rows_rsrc,
    tis_rsrc,
    lsa0,
    peer_stride,
    combine_input_offset,
    slot_stride_bytes,
    topk,
    max_tok,
    hidden,
):
    """Sum recv slot ``t``'s local rows and write the result to its lead slot.

    ``t`` must be wave-uniform. ``lsa0`` is peer 0's window base and
    ``peer_stride`` the distance between two peers' windows. ``lds_ptr`` is this
    wave's NBUF-row LDS staging area and ``lds_buf`` the row this token uses;
    rotate it per token so a row is reused only after its store retired.
    """
    n_chunks = hidden // WAVE_CHUNK
    lds_idx = fx.Int32(fx.ptrtoint(lds_ptr))
    accs = [fx.make_rmem_tensor(8, fx.Float32) for _ in range_constexpr(n_chunks)]
    for acc in accs:
        acc.store(fx.constant_vector(0.0, T.vec(8, T.f32)))
    lane_elem = lane * fx.Int32(8)
    # Compact the local rows into rows[0..c) with scalar selects, so a whole batch
    # of rows can be in flight at once instead of one row per round trip.
    raw_rows = [
        fx.Int32(
            buffer_load(rows_rsrc, t * fx.Int32(topk) + fx.Int32(k), vec_width=1,
                        dtype=T.i32, is_scalar=True)
        )
        for k in range_constexpr(topk)
    ]
    rows = [fx.Int32(0) for _ in range_constexpr(topk)]
    c = fx.Int32(0)
    lead = fx.Int32(topk)
    for k in range_constexpr(topk):
        local = raw_rows[k] >= fx.Int32(0)
        for j in range_constexpr(k + 1):
            rows[j] = (local & (c == fx.Int32(j))).select(raw_rows[k], rows[j])
        lead = (local & (lead == fx.Int32(topk))).select(fx.Int32(k), lead)
        c = c + local.select(fx.Int32(1), fx.Int32(0))

    def add_batch(j0):
        """Rows j0.. j0+ROW_BATCH: issue every load, then accumulate. A row past c
        is masked, and a masked buffer load returns zeros."""
        raws = [
            [
                buffer_load(
                    out_rsrc,
                    rows[j] * fx.Int32(hidden // 2)
                    + (fx.Int32(ch * WAVE_CHUNK) + lane_elem) // fx.Int32(2),
                    vec_width=4,
                    dtype=T.i32,
                    mask=fx.Int32(j) < c,
                    cache_modifier=LOAD_CPOL,
                )
                for ch in range_constexpr(n_chunks)
            ]
            for j in range_constexpr(j0, min(j0 + ROW_BATCH, topk))
        ]
        for raw_row in raws:
            for ch in range_constexpr(n_chunks):
                accs[ch].store(
                    accs[ch].load() + Vec(raw_row[ch]).bitcast(fx.BFloat16).to(fx.Float32)
                )

    add_batch(0)
    for j0 in range_constexpr(ROW_BATCH, topk, ROW_BATCH):
        if c > fx.Int32(j0):
            add_batch(j0)
    # Uniform by construction; say so, or the dst descriptor becomes a VGPR value
    # and every store below turns into a waterfall loop.
    lead = fx.Int32(rocdl.readfirstlane(T.i32, lead))
    enc = fx.Int32(buffer_load(tis_rsrc, t, vec_width=1, dtype=T.i32, is_scalar=True))
    origin_pe = enc // fx.Int32(max_tok)
    origin_lid = enc - origin_pe * fx.Int32(max_tok)
    dst = (
        lsa0
        + fx.Int64(origin_pe) * peer_stride
        + fx.Int64(combine_input_offset)
        + fx.Int64(origin_lid * fx.Int32(topk) + lead) * fx.Int64(slot_stride_bytes)
    )
    if const_expr(STORE_LOCAL):
        dst = out_base + fx.Int64(rows[0]) * fx.Int64(hidden * 2)
    dst_lo = fx.Int32(rocdl.readfirstlane(T.i32, fx.Int32(dst & fx.Int64(0xFFFFFFFF))))
    dst_hi = fx.Int32(rocdl.readfirstlane(T.i32, fx.Int32(dst >> fx.Int64(32))))
    dst = (fx.Int64(dst_hi) << fx.Int64(32)) | (fx.Int64(dst_lo) & fx.Int64(0xFFFFFFFF))
    # A slot with no local route has lead == topk, i.e. the next token's slot 0.
    # NO_STORE gates on t < -1: never true, but not provably so, so the loads stay.
    if (lead < fx.Int32(topk)) & ((fx.Int32(NO_STORE) == fx.Int32(0)) | (t < fx.Int32(-1))):
        # Stage the row in this wave's LDS buffer and send it with one TDM store:
        # per-lane VMEM stores to a peer run several times slower than TDM.
        # Stores retire in order, so <= NBUF-1 outstanding means the store that
        # last read this buffer, NBUF tokens back, is done.
        tdm_ops.tensor_wait((NBUF - 1) * PIECES)
        _, lds_store_b128 = make_lds_copy_ops(128)
        row_b = lds_buf * fx.Int32(hidden * 2)
        for ch in range_constexpr(n_chunks):
            lds_store_b128(
                lds_idx,
                row_b + (fx.Int32(ch * WAVE_CHUNK) + lane_elem) * fx.Int32(2),
                Vec(accs[ch].load()).to(fx.BFloat16).bitcast(fx.Int32).ir_value(),
            )
        rocdl.s_wait_dscnt(0)
        # The slot stride is a power of two, so whole rows sent front to back
        # would have every pusher hitting the same peer channels at once; split
        # the row and rotate the piece order per token to spread them.
        piece = hidden // PIECES
        rot = t % fx.Int32(PIECES)
        for q in range_constexpr(PIECES):
            pc = (rot + fx.Int32(q)) % fx.Int32(PIECES)
            gt = fx.Tensor(
                fx.make_view(
                    fx.inttoptr(_I16_GLOBAL(), dst + fx.Int64(pc * fx.Int32(piece * 2))),
                    fx.make_layout((1, piece), (piece, 1)),
                )
            )
            lv = fx.Tensor(
                fx.make_view(
                    fx.recast_iter(_I16_SHARED(), lds_ptr)
                    + lds_buf * fx.Int32(hidden)
                    + pc * fx.Int32(piece),
                    fx.make_layout((1, piece), (piece, 1)),
                )
            )
            atom = fx.rocdl.make_tdm_atom(gt, [1, None], strides=[piece, None], num_warps=1)
            fx.copy(atom, lv, gt)


emit_push_token = traced(emit_push_token)


# A pusher gives up on a token after this many polls. Its tiles do complete;
# the cap only turns a broken invariant into wrong output, not a hung kernel.
POLL_LIMIT = 1 << 22
# s_sleep(BACKOFF) between polls of a tile that is not done (~27 ns per unit).
BACKOFF = int(os.environ.get("AITER_EP_DEDUP_BACKOFF", "32"))


def emit_tile_done(m_tile, *, state_addr):
    """One (M tile, N tile) of gemm2 output is staged; call once per workgroup,
    after the whole tile's device-scope stores have completed."""
    if const_expr(COUNT_MODE == 2):
        return
    atomic_add_agent(state_addr + fx.Int64(MTILE_OFF + m_tile) * fx.Int64(4), fx.Int32(1))


emit_tile_done = traced(emit_tile_done)


def _wait_tile(addr_i64, n_tiles):
    v = fx.Int32(load_i32_global_agent(addr_i64))
    n = fx.Int32(0)
    while (v < n_tiles) & (n < fx.Int32(POLL_LIMIT)):
        rocdl.s_sleep(BACKOFF)
        v = fx.Int32(load_i32_global_agent(addr_i64))
        n = n + fx.Int32(1)


_wait_tile = traced(_wait_tile)


def _wait_token(t, *, state_addr, rows_rsrc, topk, tile_m, n_tiles):
    """Every M tile holding one of ``t``'s rows has all its N tiles staged."""
    for k in range_constexpr(topk):
        row = fx.Int32(
            buffer_load(rows_rsrc, t * fx.Int32(topk) + fx.Int32(k), vec_width=1,
                        dtype=T.i32, is_scalar=True)
        )
        if row >= fx.Int32(0):
            _wait_tile(
                state_addr + fx.Int64(fx.Int32(MTILE_OFF) + row // fx.Int32(tile_m)) * fx.Int64(4),
                n_tiles,
            )


_wait_token = traced(_wait_token)


def emit_pusher_loop(
    wave_id, n_waves, lane, lds_ptr, nvt, *, state_addr, tile_m, n_tiles, **token_kw
):
    """Push the recv slots in the order the GEMM completes them.

    The order kernel sorted the slots by the last M tile each waits for, and
    wave ``wave_id`` takes sorted positions wave_id, +n_waves, ...: every wave
    walks the keys upward, and a crowded key is spread over all of them. Before
    a slot it waits for that key's tile, then confirms its other tiles, which
    the GEMM has almost always finished by then.
    """
    if const_expr(NO_POLL):
        return
    state_rsrc = create_buffer_resource_from_addr(state_addr.ir_value())
    for p in range(wave_id, nvt, n_waves):
        t = fx.Int32(
            buffer_load(state_rsrc, fx.Int32(ORDER_OFF) + p, vec_width=1, dtype=T.i32,
                        is_scalar=True)
        )
        key = fx.Int32(
            buffer_load(state_rsrc, fx.Int32(SKEY_OFF) + p, vec_width=1, dtype=T.i32,
                        is_scalar=True)
        )
        _wait_tile(state_addr + fx.Int64(fx.Int32(MTILE_OFF) + key) * fx.Int64(4), n_tiles)
        _wait_token(
            t,
            state_addr=state_addr,
            rows_rsrc=token_kw["rows_rsrc"],
            topk=token_kw["topk"],
            tile_m=tile_m,
            n_tiles=n_tiles,
        )
        if const_expr(not NO_PUSH):
            emit_push_token(
                t, lane, lds_ptr, ((p - wave_id) // fx.Int32(n_waves)) % fx.Int32(NBUF),
                **token_kw,
            )
    tdm_ops.tensor_wait(0)


emit_pusher_loop = traced(emit_pusher_loop)


# ---------------------------------------------------------------------------
# Column pushers. gemm2 walks its tiles column by column (all M tiles of N tile
# 0, then N tile 1, ...), so N tile j of EVERY recv slot is complete as soon as
# column j is: the pushers sum and send each slot's 256-column segment j right
# behind the GEMM, eight slots per TDM gather store, instead of waiting for a
# slot's last expert the way whole-row pushing must.
# ---------------------------------------------------------------------------
COL_OFF = MTILE_OFF  # one counter per gemm2 N tile, zeroed by the last pusher
STATE_DONE = 0
GROUP = 8  # slots per gather store: its index limit at 32-bit indices
COL_NBUF = int(os.environ.get("AITER_EP_DEDUP_COL_NBUF", "4"))
COLS_PER_PUSH = int(os.environ.get("AITER_EP_DEDUP_COLS_PER_PUSH", "4"))


def column_push_lds_bytes(*, waves, max_tokens_per_wave, topk, seg_elems):
    """Per wave: its slots' metadata (rows, count, dest index) and COL_NBUF
    staging tiles of GROUP segments."""
    meta = max_tokens_per_wave * (topk + 2) * 4
    return waves * (meta + COL_NBUF * GROUP * seg_elems * 2)


def emit_column_tile_done(n_tile, *, state_addr):
    """One (M tile, N tile) is staged; once per workgroup, after its stores."""
    atomic_add_agent(state_addr + fx.Int64(COL_OFF + n_tile) * fx.Int64(4), fx.Int32(1))


emit_column_tile_done = traced(emit_column_tile_done)


def emit_column_pusher(
    wave_id,
    n_waves,
    lane,
    lds_meta,
    lds_out,
    nvt,
    n_tiles,
    full_col,
    *,
    state_addr,
    out_rsrc,
    rows_rsrc,
    tis_rsrc,
    lsa0,
    peer_stride,
    combine_input_offset,
    slot_stride_bytes,
    world_size,
    topk,
    max_tok,
    hidden,
    seg_elems,
    max_tokens_per_wave,
):
    """Push this wave's recv slots t = wave_id + k*n_waves, column by column.

    ``lds_meta`` (i32) and ``lds_out`` (bf16) are this wave's LDS areas.
    ``full_col`` is how many tiles a column has, the gemm2 M tiles with work.
    """
    if const_expr(NO_POLL):
        return
    meta_w = topk + 2
    n_mine = (nvt - wave_id + fx.Int32(n_waves - 1)) // fx.Int32(n_waves)
    n_mine = (n_mine < fx.Int32(0)).select(fx.Int32(0), n_mine)
    n_mine = (n_mine > fx.Int32(max_tokens_per_wave)).select(
        fx.Int32(max_tokens_per_wave), n_mine
    )
    meta_idx = fx.Int32(fx.ptrtoint(lds_meta))
    out_idx = fx.Int32(fx.ptrtoint(lds_out))
    lds_load_b32, lds_store_b32 = make_lds_copy_ops(32)
    _, lds_store_b128 = make_lds_copy_ops(128)
    k_slots = fx.Int32((peer_stride) // fx.Int64(slot_stride_bytes))
    oob = k_slots * fx.Int32(world_size)

    # -- once, lane-parallel: lane L describes slot k=L of this wave (its local
    # rows, their count, its lead destination); readlane hands them out later.
    k_lane = lane
    live_lane = k_lane < n_mine
    t_lane = live_lane.select(wave_id + k_lane * fx.Int32(n_waves), fx.Int32(0))
    raw_rows = [
        fx.Int32(
            buffer_load(rows_rsrc, t_lane * fx.Int32(topk) + fx.Int32(kk), vec_width=1,
                        dtype=T.i32)
        )
        for kk in range_constexpr(topk)
    ]
    rows_lane = [fx.Int32(0) for _ in range_constexpr(topk)]
    c_lane = fx.Int32(0)
    lead = fx.Int32(topk)
    for kk in range_constexpr(topk):
        local = live_lane & (raw_rows[kk] >= fx.Int32(0))
        for j in range_constexpr(kk + 1):
            rows_lane[j] = (local & (c_lane == fx.Int32(j))).select(raw_rows[kk], rows_lane[j])
        lead = (local & (lead == fx.Int32(topk))).select(fx.Int32(kk), lead)
        c_lane = c_lane + local.select(fx.Int32(1), fx.Int32(0))
    enc = fx.Int32(buffer_load(tis_rsrc, t_lane, vec_width=1, dtype=T.i32))
    origin_pe = enc // fx.Int32(max_tok)
    origin_lid = enc - origin_pe * fx.Int32(max_tok)
    dst_lane = (live_lane & (lead < fx.Int32(topk))).select(
        origin_pe * k_slots + origin_lid * fx.Int32(topk) + lead, oob
    )

    def from_lane(v, k):
        return fx.Int32(rocdl.readlane(T.i32, _raw(v), _raw(k)))

    comb_view = fx.Tensor(
        fx.make_view(
            fx.inttoptr(_I16_GLOBAL(), lsa0 + fx.Int64(combine_input_offset)),
            fx.make_layout((oob, seg_elems * COLS_PER_PUSH), (slot_stride_bytes // 2, 1)),
        )
    )
    lds_c = fx.Tensor(
        fx.make_view(
            fx.recast_iter(_I16_SHARED(), lds_out),
            fx.make_layout(
                (COL_NBUF * GROUP, seg_elems * COLS_PER_PUSH), (seg_elems * COLS_PER_PUSH, 1)
            ),
        )
    )
    n_groups = (n_mine + fx.Int32(GROUP - 1)) // fx.Int32(GROUP)
    if const_expr(NO_PUSH):
        n_groups = fx.Int32(0)
    lane_elem = lane * fx.Int32(8)
    # A push covers COLS_PER_PUSH adjacent N tiles: bigger gather stores, at the
    # cost of trailing the GEMM by that many columns.
    span = seg_elems * COLS_PER_PUSH
    n_spans = n_tiles // fx.Int32(COLS_PER_PUSH)
    for sp in range(n_spans):
        for q in range_constexpr(COLS_PER_PUSH):
            _wait_tile(
                state_addr
                + fx.Int64(fx.Int32(COL_OFF) + sp * fx.Int32(COLS_PER_PUSH) + fx.Int32(q))
                * fx.Int64(4),
                full_col,
            )
        col_e = sp * fx.Int32(span)
        for g in range(n_groups):
            buf = (sp * n_groups + g) % fx.Int32(COL_NBUF)
            # In-order retirement: <= COL_NBUF-1 left means this buffer's
            # previous store, COL_NBUF groups back, has read it.
            tdm_ops.tensor_wait(COL_NBUF - 1)
            slots = []
            for i in range_constexpr(GROUP):
                k = g * fx.Int32(GROUP) + fx.Int32(i)
                ks = (k < n_mine).select(k, fx.Int32(0))
                c = (k < n_mine).select(from_lane(c_lane, ks), fx.Int32(0))
                row_dw = [
                    from_lane(rows_lane[r], ks) * fx.Int32(hidden // 2)
                    for r in range_constexpr(topk)
                ]
                slots.append((c, row_dw))
            for q in range_constexpr(COLS_PER_PUSH):
                e = col_e + fx.Int32(q * seg_elems) + lane_elem
                # The whole group's rows for this column are issued before any
                # is summed, so they are all in flight together.
                raws = [
                    [
                        buffer_load(
                            out_rsrc,
                            slots[i][1][r] + e // fx.Int32(2),
                            vec_width=4,
                            dtype=T.i32,
                            mask=fx.Int32(r) < slots[i][0],
                            cache_modifier=LOAD_CPOL,
                        )
                        for r in range_constexpr(min(ROW_BATCH, topk))
                    ]
                    for i in range_constexpr(GROUP)
                ]
                for i in range_constexpr(GROUP):
                    c, row_dw = slots[i]
                    acc_t = fx.make_rmem_tensor(8, fx.Float32)
                    acc = Vec(raws[i][0]).bitcast(fx.BFloat16).to(fx.Float32)
                    for r in range_constexpr(1, len(raws[i])):
                        acc = acc + Vec(raws[i][r]).bitcast(fx.BFloat16).to(fx.Float32)
                    acc_t.store(acc)
                    # Rare: a slot with more local rows than one batch.
                    for r in range_constexpr(ROW_BATCH, topk):
                        if c > fx.Int32(r):
                            raw = buffer_load(
                                out_rsrc,
                                row_dw[r] + e // fx.Int32(2),
                                vec_width=4,
                                dtype=T.i32,
                                cache_modifier=LOAD_CPOL,
                            )
                            acc_t.store(
                                acc_t.load() + Vec(raw).bitcast(fx.BFloat16).to(fx.Float32)
                            )
                    lds_store_b128(
                        out_idx,
                        (
                            (buf * fx.Int32(GROUP) + fx.Int32(i)) * fx.Int32(span)
                            + fx.Int32(q * seg_elems)
                            + lane_elem
                        )
                        * fx.Int32(2),
                        Vec(acc_t.load()).to(fx.BFloat16).bitcast(fx.Int32).ir_value(),
                    )
            rocdl.s_wait_dscnt(0)
            idx = []
            for i in range_constexpr(GROUP):
                k = g * fx.Int32(GROUP) + fx.Int32(i)
                idx.append(
                    (k < n_mine).select(
                        from_lane(dst_lane, (k < n_mine).select(k, fx.Int32(0))), oob
                    )
                )
            desc = make_tensor_gather_descriptor(
                comb_view,
                lds_c,
                idx,
                row_width=span,
                tensor_dim0=span,
                tensor_dim1=oob.ir_value(),
                stride=slot_stride_bytes // 2,
                elem_bytes=2,
                index_size=32,
                lds_byte_offset=fx.index_cast(T.index, buf * fx.Int32(GROUP * span * 2)),
                global_byte_offset=col_e * fx.Int32(2),
            )
            if const_expr(not NO_STORE):
                tensor_store_gather(desc)
    tdm_ops.tensor_wait(0)
    if lane == fx.Int32(0):
        done = fx.Int32(atomic_add_agent(state_addr + fx.Int64(STATE_DONE * 4), fx.Int32(1)))
        if done == fx.Int32(n_waves - 1):
            for i in range(n_tiles):
                _store_i32_agent(state_addr + fx.Int64(fx.Int32(COL_OFF) + i) * fx.Int64(4), 0)
            _store_i32_agent(state_addr + fx.Int64(STATE_DONE * 4), 0)


emit_column_pusher = traced(emit_column_pusher)


def _store_i32_agent(addr_i64, val):
    from ..communication_ops_utils import _llvm_d, _to_ptr_global

    _llvm_d.StoreOp(
        fx.Int32(val).ir_value(),
        _to_ptr_global(addr_i64),
        alignment=4,
        ordering=_llvm_d.AtomicOrdering.monotonic,
        syncscope=fx.rocdl.SyncScope.AgentOneAs,
    )


# ---------------------------------------------------------------------------
# Reduce-shaped push: the combine reduce's pipeline (TDM in, sum in LDS, TDM
# out, both sides double buffered), with the input rows gathered by
# topids_to_rows instead of read as contiguous slots, and the output gathered
# out to each token's lead slot on its origin rank instead of stored locally.
# ---------------------------------------------------------------------------
RP_CHUNK = 256  # elements of one N tile; a staged row spans RP_CHUNKS of them
RP_TOKS = 4  # tokens per trip, one per wave
RP_CHUNKS = int(os.environ.get("AITER_EP_DEDUP_RP_CHUNKS", "4"))
# Waves 0-1 only load and waves 2-3 only store: TDM ops retire in order per
# wave, so a wave that did both would wait on its slow peer write before it
# could see its next load. RP_NIN input tiles keep RP_NIN-1 trips of loads
# ahead; RP_NOUT output tiles let a store retire RP_NOUT trips later.
RP_NIN = int(os.environ.get("AITER_EP_DEDUP_RP_NIN", "3"))
RP_NOUT = int(os.environ.get("AITER_EP_DEDUP_RP_NOUT", "8"))


def build_dedup_reduce_push_module(
    *,
    arena_handle,
    combine_input_offset,
    slot_stride_bytes,
    topk,
    max_tok,
    hidden,
    world_size,
    block_num,
):
    import mori.cco.device.flydsl as _cco
    from .tdm_gather_shim import tensor_load_gather

    waves = RP_TOKS
    lanes = waves * WAVE
    row_e = RP_CHUNKS * RP_CHUNK  # elements of one staged row
    in_rows = RP_TOKS * topk
    in_b = in_rows * row_e * 2
    out_b = RP_TOKS * row_e * 2
    n_chunks = hidden // RP_CHUNK
    assert hidden % RP_CHUNK == 0 and n_chunks % RP_CHUNKS == 0
    iters = n_chunks // RP_CHUNKS
    slots = RP_TOKS * row_e // 16  # a lane sums 16 elements per round
    assert slots % lanes == 0
    rounds = slots // lanes
    slots_per_tok = row_e // 16
    # Two gathers per token: a 32-bit-index gather carries at most 8 rows.
    gathers = [(r0, min(r0 + 8, topk)) for r0 in range(0, topk, 8)]
    loads_per_trip = len(gathers)
    knobs = KNOBS_TAG

    @flyc.kernel(
        name=f"ep_dedup_rpush_{knobs}_c{RP_CHUNKS}i{RP_NIN}o{RP_NOUT}_b{block_num}",
        known_block_size=[lanes, 1, 1],
    )
    def ep_dedup_rpush(
        addr_out: fx.Int64,
        out_rows: fx.Int32,
        addr_rows: fx.Int64,
        addr_nvt: fx.Int64,
        addr_tis: fx.Int64,
    ):
        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        wave = fx.Int32(rocdl.readfirstlane(T.i32, tid // WAVE))
        lds_load_b128, lds_store_b128 = make_lds_copy_ops(128)
        smem = fx.SharedAllocator(static=False)
        in_base = smem.allocate(RP_NIN * in_b)._ptr
        out_base = smem.allocate(RP_NOUT * out_b)._ptr
        in_idx = fx.Int32(fx.ptrtoint(in_base))
        out_idx = fx.Int32(fx.ptrtoint(out_base))

        win = _cco.Window(fx.Int64(arena_handle))
        lsa0 = fx.Int64(win.lsa_ptr(fx.Int32(0), 0))
        peer_stride = fx.Int64(win.lsa_ptr(fx.Int32(1), 0)) - lsa0
        k_slots = fx.Int32(peer_stride // fx.Int64(slot_stride_bytes))
        oob = k_slots * fx.Int32(world_size)
        rows_rsrc = create_buffer_resource_from_addr(addr_rows.ir_value())
        tis_rsrc = create_buffer_resource_from_addr(addr_tis.ir_value())
        nvt_rsrc = create_buffer_resource_from_addr(addr_nvt.ir_value())
        nvt = fx.Int32(buffer_load(nvt_rsrc, fx.Int32(0), vec_width=1, dtype=T.i32, is_scalar=True))

        src_view = fx.Tensor(
            fx.make_view(
                fx.inttoptr(_I16_GLOBAL(), addr_out),
                fx.make_layout((out_rows, row_e), (hidden, 1)),
            )
        )
        comb_view = fx.Tensor(
            fx.make_view(
                fx.inttoptr(_I16_GLOBAL(), lsa0 + fx.Int64(combine_input_offset)),
                fx.make_layout((oob, row_e), (slot_stride_bytes // 2, 1)),
            )
        )

        def lds_in(buf):
            return fx.Tensor(
                fx.make_view(
                    fx.recast_iter(_I16_SHARED(), in_base) + buf * fx.Int32(in_rows * row_e),
                    fx.make_layout((in_rows, row_e), (row_e, 1)),
                )
            )

        lds_out = fx.Tensor(
            fx.make_view(
                fx.recast_iter(_I16_SHARED(), out_base),
                fx.make_layout((RP_NOUT * RP_TOKS, row_e), (row_e, 1)),
            )
        )

        n_groups = (nvt + fx.Int32(RP_TOKS - 1)) // fx.Int32(RP_TOKS)
        total_work = n_groups * fx.Int32(iters)

        def coords(work, ti):
            grp = work // fx.Int32(iters)
            it = work - grp * fx.Int32(iters)
            tok = grp * fx.Int32(RP_TOKS) + ti
            return tok, it * fx.Int32(RP_CHUNKS * RP_CHUNK)

        lane = fx.Int32(tid) % fx.Int32(WAVE)
        half = RP_TOKS // 2  # tokens per loader / store wave
        is_loader = wave < fx.Int32(2)
        role_ti = (wave % fx.Int32(2)) * fx.Int32(half)

        def prefetch_meta(work):
            """Lane h*topk+k: row of route k of this wave's h-th token of trip
            ``work`` (out_rows if remote or past the end); lane h: its origin.
            Issued a trip ahead and consumed with readlane, so no gather waits
            on a metadata load."""
            h = lane // fx.Int32(topk)
            k = lane - h * fx.Int32(topk)
            tok, _ = coords(work, role_ti + h)
            valid = (work < total_work) & (tok < nvt) & (lane < fx.Int32(half * topk))
            safe = valid.select(tok, fx.Int32(0))
            r = fx.Int32(
                buffer_load(rows_rsrc, safe * fx.Int32(topk) + k, vec_width=1, dtype=T.i32)
            )
            idxv = (valid & (r >= fx.Int32(0))).select(r, out_rows)
            tok_e, _ = coords(work, role_ti + lane)
            valid_e = (work < total_work) & (tok_e < nvt) & (lane < fx.Int32(half))
            encv = fx.Int32(
                buffer_load(tis_rsrc, valid_e.select(tok_e, fx.Int32(0)), vec_width=1, dtype=T.i32)
            )
            encv = valid_e.select(encv, fx.Int32(-1))
            return idxv, encv

        def lane_val(v, i):
            return fx.Int32(rocdl.readlane(T.i32, _raw(v), _raw(fx.Int32(i))))

        def token_rows(idxv, h):
            return [lane_val(idxv, h * topk + k) for k in range_constexpr(topk)]

        def issue_load(work, h, buf, idxv):
            """This wave's h-th token of the trip: its topk rows of this column
            span, gathered."""
            ti = role_ti + fx.Int32(h)
            _, e0 = coords(work, ti)
            rows = token_rows(idxv, h)
            for r0, r1 in gathers:
                desc = make_tensor_gather_descriptor(
                    src_view,
                    lds_in(buf),
                    rows[r0:r1],
                    row_width=row_e,
                    tensor_dim0=row_e,
                    tensor_dim1=out_rows.ir_value(),
                    stride=hidden,
                    elem_bytes=2,
                    index_size=32,
                    lds_byte_offset=fx.index_cast(
                        T.index, (ti * fx.Int32(topk) + fx.Int32(r0)) * fx.Int32(row_e * 2)
                    ),
                    global_byte_offset=fx.Int64(e0) * fx.Int64(2),
                )
                tensor_load_gather(desc, cache_policy=LOAD_CPOL)

        def store_tile(work, h, buf, idxv, encv):
            """This wave's h-th token's summed row, to its lead slot on the origin."""
            ti = role_ti + fx.Int32(h)
            _, e0 = coords(work, ti)
            rows = token_rows(idxv, h)
            lead = fx.Int32(topk)
            for k in range_constexpr(topk):
                lead = ((rows[k] < out_rows) & (lead == fx.Int32(topk))).select(fx.Int32(k), lead)
            enc = lane_val(encv, h)
            origin_pe = enc // fx.Int32(max_tok)
            origin_lid = enc - origin_pe * fx.Int32(max_tok)
            dst = ((enc >= fx.Int32(0)) & (lead < fx.Int32(topk))).select(
                origin_pe * k_slots + origin_lid * fx.Int32(topk) + lead, oob
            )
            if const_expr(STORE_LOCAL):
                # Timing only: the same gather store, aimed at this rank's own
                # staging rows instead of a peer.
                dst = rows[0]
            desc = make_tensor_gather_descriptor(
                src_view if STORE_LOCAL else comb_view,
                lds_out,
                [dst],
                row_width=row_e,
                tensor_dim0=row_e,
                tensor_dim1=(out_rows if STORE_LOCAL else oob).ir_value(),
                stride=hidden if STORE_LOCAL else slot_stride_bytes // 2,
                elem_bytes=2,
                index_size=32,
                lds_byte_offset=fx.index_cast(
                    T.index, (buf * fx.Int32(RP_TOKS) + ti) * fx.Int32(row_e * 2)
                ),
                global_byte_offset=fx.Int64(e0) * fx.Int64(2),
            )
            if const_expr(not NO_STORE):
                tensor_store_gather(desc)

        def reduce_tile(buf, obuf):
            for rd in range_constexpr(rounds):
                slot = fx.Int32(tid) + fx.Int32(rd * lanes)
                ti = slot // fx.Int32(slots_per_tok)
                e = (slot - ti * fx.Int32(slots_per_tok)) * fx.Int32(16)
                accs = [Vec.filled(8, 0.0, fx.Float32) for _ in range_constexpr(2)]
                for k in range_constexpr(topk):
                    row_b = (
                        buf * fx.Int32(in_rows) + ti * fx.Int32(topk) + fx.Int32(k)
                    ) * fx.Int32(row_e * 2)
                    for j in range_constexpr(2):
                        raw = Vec(lds_load_b128(in_idx, row_b + e * fx.Int32(2) + fx.Int32(j * 16)))
                        accs[j] = accs[j] + raw.bitcast(fx.BFloat16).to(fx.Float32)
                out_b_off = ((obuf * fx.Int32(RP_TOKS) + ti) * fx.Int32(row_e) + e) * fx.Int32(2)
                for j in range_constexpr(2):
                    lds_store_b128(
                        out_idx,
                        out_b_off + fx.Int32(j * 16),
                        accs[j].to(fx.BFloat16).bitcast(fx.Int32).ir_value(),
                    )

        loads_per_wave_trip = half * loads_per_trip

        def load_trip(work, slot, idxv):
            for h in range_constexpr(half):
                issue_load(work, h, slot, idxv)

        # Loaders run RP_NIN-1 trips ahead; stores consume the current trip.
        lead_trips = is_loader.select(fx.Int32(RP_NIN - 1), fx.Int32(0))
        if is_loader:
            for p in range_constexpr(RP_NIN - 1):
                w = bid + fx.Int32(p * block_num)
                pidx, _ = prefetch_meta(w)
                load_trip(w, fx.Int32(p), pidx)
        idxv, encv = prefetch_meta(bid + lead_trips * fx.Int32(block_num))
        trip = fx.Int32(0)
        for work in range(bid, total_work, block_num):
            ib = trip % fx.Int32(RP_NIN)
            ob = trip % fx.Int32(RP_NOUT)
            nidx, nenc = prefetch_meta(work + (lead_trips + fx.Int32(1)) * fx.Int32(block_num))
            if is_loader:
                # Always issued, as zero rows past the end, so the count is uniform.
                load_trip(
                    work + fx.Int32((RP_NIN - 1) * block_num),
                    (trip + fx.Int32(RP_NIN - 1)) % fx.Int32(RP_NIN),
                    idxv,
                )
                # Newer than this trip's loads: the RP_NIN-1 trips issued after.
                tdm_ops.tensor_wait((RP_NIN - 1) * loads_per_wave_trip)
            else:
                # The store that last read output tile ob, RP_NOUT trips back.
                tdm_ops.tensor_wait((RP_NOUT - 1) * half)
            gpu.barrier()
            reduce_tile(ib, ob)
            rocdl.s_wait_dscnt(0)
            gpu.barrier()
            if not is_loader:
                for h in range_constexpr(half):
                    store_tile(work, h, ob, idxv, encv)
            idxv = nidx
            encv = nenc
            trip = trip + fx.Int32(1)
        tdm_ops.tensor_wait(0)

    @flyc.jit
    def launch(
        addr_out: fx.Int64,
        out_rows: fx.Int32,
        addr_rows: fx.Int64,
        addr_nvt: fx.Int64,
        addr_tis: fx.Int64,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
    ):
        ep_dedup_rpush(addr_out, out_rows, addr_rows, addr_nvt, addr_tis).launch(
            grid=(block_num, 1, 1), block=[lanes, 1, 1], stream=stream
        )

    return launch


# ---------------------------------------------------------------------------
# Batched wave pusher. A wave owns slots t = wave_gid + k*n_waves and takes them
# eight at a time. Lane L describes row L%4 of slot L//4 of the batch (a slot's
# first four local rows; more is the rare slow path), so the batch's metadata
# is one lane-parallel load. The batch then walks the row 256 columns at a
# time: 32 VMEM loads (one 512 B segment of each row) per column, summed per
# slot, and ONE 8-row TDM gather store of the eight 512 B results -- the
# p2p_bw store shape. Loads (vmcnt) and stores (tensorcnt) never wait on each
# other; columns go in pairs so one pair's loads fly while the other is summed.
# ---------------------------------------------------------------------------
BP_ROWS = int(os.environ.get("AITER_EP_DEDUP_BP_ROWS", "4"))  # local rows per slot in lanes
BP_SLOTS = 32 // BP_ROWS  # slots per batch = rows of one gather store
BP_NBUF = int(os.environ.get("AITER_EP_DEDUP_BP_NBUF", "4"))  # stores in flight
BP_SEG = 256  # elements of one column segment: 32 lanes x 8 bf16


def batch_push_lds_bytes():
    """Per wave: BP_NBUF staging tiles of BP_SLOTS 512 B rows."""
    return BP_NBUF * BP_SLOTS * BP_SEG * 2


def emit_batched_pusher(
    wave_gid,
    n_waves,
    lane,
    lds_out,
    nvt,
    *,
    out_rsrc,
    out_rows,
    rows_rsrc,
    tis_rsrc,
    lsa0,
    peer_stride,
    combine_input_offset,
    slot_stride_bytes,
    world_size,
    topk,
    max_tok,
    hidden,
    col_wait=None,
):
    """Push this wave's recv slots; ``col_wait(col)`` (if given) blocks until
    gemm2 column ``col`` is staged, which is what lets this run inside gemm2."""
    n_cols = hidden // BP_SEG
    assert n_cols % 2 == 0
    k_slots = fx.Int32(peer_stride // fx.Int64(slot_stride_bytes))
    oob = k_slots * fx.Int32(world_size)
    n_mine = (nvt - wave_gid + fx.Int32(n_waves - 1)) // fx.Int32(n_waves)
    n_mine = (n_mine < fx.Int32(0)).select(fx.Int32(0), n_mine)
    n_batches = (n_mine + fx.Int32(BP_SLOTS - 1)) // fx.Int32(BP_SLOTS)
    out_idx = fx.Int32(fx.ptrtoint(lds_out))
    _, lds_store_b128 = make_lds_copy_ops(128)
    lane_elem = lane * fx.Int32(8)
    comb_view = fx.Tensor(
        fx.make_view(
            fx.inttoptr(_I16_GLOBAL(), lsa0 + fx.Int64(combine_input_offset)),
            fx.make_layout((oob, BP_SEG), (slot_stride_bytes // 2, 1)),
        )
    )
    lds_c = fx.Tensor(
        fx.make_view(
            fx.recast_iter(_I16_SHARED(), lds_out),
            fx.make_layout((BP_NBUF * BP_SLOTS, BP_SEG), (BP_SEG, 1)),
        )
    )
    step = fx.Int32(0)
    for b in range(n_batches):
        # -- the batch's metadata, lane-parallel --
        si = lane // fx.Int32(BP_ROWS)
        rj = lane - si * fx.Int32(BP_ROWS)
        k = b * fx.Int32(BP_SLOTS) + si
        live = k < n_mine
        t = live.select(wave_gid + k * fx.Int32(n_waves), fx.Int32(0))
        raw = [
            fx.Int32(
                buffer_load(rows_rsrc, t * fx.Int32(topk) + fx.Int32(kk), vec_width=1,
                            dtype=T.i32)
            )
            for kk in range_constexpr(topk)
        ]
        row_l = fx.Int32(0)  # this lane's row: the rj-th local row of its slot
        c = fx.Int32(0)
        lead = fx.Int32(topk)
        for kk in range_constexpr(topk):
            local = live & (raw[kk] >= fx.Int32(0))
            row_l = (local & (c == rj)).select(raw[kk], row_l)
            lead = (local & (lead == fx.Int32(topk))).select(fx.Int32(kk), lead)
            c = c + local.select(fx.Int32(1), fx.Int32(0))
        valid_l = rj < c
        enc = fx.Int32(buffer_load(tis_rsrc, t, vec_width=1, dtype=T.i32))
        origin_pe = enc // fx.Int32(max_tok)
        origin_lid = enc - origin_pe * fx.Int32(max_tok)
        dst_l = (live & (lead < fx.Int32(topk))).select(
            origin_pe * k_slots + origin_lid * fx.Int32(topk) + lead, oob
        )

        def lane_i(v, i):
            return fx.Int32(rocdl.readlane(T.i32, _raw(v), _raw(fx.Int32(i))))

        row_dw = [lane_i(row_l, i) * fx.Int32(hidden // 2) for i in range_constexpr(32)]
        row_ok = [lane_i(valid_l.select(fx.Int32(1), fx.Int32(0)), i) != fx.Int32(0)
                  for i in range_constexpr(32)]
        slot_c = [lane_i(c, BP_ROWS * s) for s in range_constexpr(BP_SLOTS)]
        slot_dst = [lane_i(dst_l, BP_ROWS * s) for s in range_constexpr(BP_SLOTS)]
        slot_raw = [[lane_i(raw[kk], BP_ROWS * s) for kk in range_constexpr(topk)]
                    for s in range_constexpr(BP_SLOTS)]

        def issue_col(col):
            e_dw = (col * fx.Int32(BP_SEG) + lane_elem) // fx.Int32(2)
            return [
                buffer_load(out_rsrc, row_dw[i] + e_dw, vec_width=4, dtype=T.i32,
                            mask=row_ok[i], cache_modifier=LOAD_CPOL)
                for i in range_constexpr(32)
            ]

        def finish_col(col, raws):
            buf = step % fx.Int32(BP_NBUF)
            # Peer stores do not retire in issue order, so a partial count says
            # nothing about one tile. Drain at each wrap: every tile's previous
            # store is then known done for the whole round.
            if buf == fx.Int32(0):
                tdm_ops.tensor_wait(0)
            for s in range_constexpr(BP_SLOTS):
                acc = Vec(raws[BP_ROWS * s]).bitcast(fx.BFloat16).to(fx.Float32)
                for j in range_constexpr(1, BP_ROWS):
                    acc = acc + Vec(raws[BP_ROWS * s + j]).bitcast(fx.BFloat16).to(fx.Float32)
                acc_t = fx.make_rmem_tensor(8, fx.Float32)
                acc_t.store(acc)
                # Slow path: a slot with more local rows than the lanes carry.
                if slot_c[s] > fx.Int32(BP_ROWS):
                    n_seen = fx.Int32(0)
                    for kk in range_constexpr(topk):
                        r = slot_raw[s][kk]
                        if r >= fx.Int32(0):
                            if n_seen >= fx.Int32(BP_ROWS):
                                extra = buffer_load(
                                    out_rsrc,
                                    r * fx.Int32(hidden // 2)
                                    + (col * fx.Int32(BP_SEG) + lane_elem) // fx.Int32(2),
                                    vec_width=4, dtype=T.i32, cache_modifier=LOAD_CPOL,
                                )
                                acc_t.store(acc_t.load()
                                            + Vec(extra).bitcast(fx.BFloat16).to(fx.Float32))
                        n_seen = n_seen + (r >= fx.Int32(0)).select(fx.Int32(1), fx.Int32(0))
                lds_store_b128(
                    out_idx,
                    ((buf * fx.Int32(BP_SLOTS) + fx.Int32(s)) * fx.Int32(BP_SEG) + lane_elem)
                    * fx.Int32(2),
                    Vec(acc_t.load()).to(fx.BFloat16).bitcast(fx.Int32).ir_value(),
                )
            rocdl.s_wait_dscnt(0)
            desc = make_tensor_gather_descriptor(
                comb_view,
                lds_c,
                slot_dst,
                row_width=BP_SEG,
                tensor_dim0=BP_SEG,
                tensor_dim1=oob.ir_value(),
                stride=slot_stride_bytes // 2,
                elem_bytes=2,
                index_size=32,
                lds_byte_offset=fx.index_cast(T.index, buf * fx.Int32(BP_SLOTS * BP_SEG * 2)),
                global_byte_offset=col * fx.Int32(BP_SEG * 2),
            )
            if const_expr(not NO_STORE):
                tensor_store_gather(desc)

        for cp in range(n_cols // 2):
            c0 = cp * fx.Int32(2)
            if const_expr(col_wait is not None):
                col_wait(c0)
            ra = issue_col(c0)
            if const_expr(col_wait is not None):
                col_wait(c0 + fx.Int32(1))
            rb = issue_col(c0 + fx.Int32(1))
            finish_col(c0, ra)
            step = step + fx.Int32(1)
            finish_col(c0 + fx.Int32(1), rb)
            step = step + fx.Int32(1)
    tdm_ops.tensor_wait(0)


emit_batched_pusher = traced(emit_batched_pusher)


def build_dedup_batch_push_module(
    *,
    arena_handle,
    combine_input_offset,
    slot_stride_bytes,
    topk,
    max_tok,
    hidden,
    world_size,
    block_num,
):
    """The batched wave pusher as a standalone kernel, run after gemm2."""
    import mori.cco.device.flydsl as _cco

    waves = 4
    knobs = KNOBS_TAG

    @flyc.kernel(name=f"ep_dedup_bpush_{knobs}_n{BP_NBUF}r{BP_ROWS}_b{block_num}",
                 known_block_size=[waves * WAVE, 1, 1])
    def ep_dedup_bpush(
        addr_out: fx.Int64,
        out_rows: fx.Int32,
        addr_rows: fx.Int64,
        addr_nvt: fx.Int64,
        addr_tis: fx.Int64,
    ):
        tid = fx.thread_idx.x
        wave = fx.Int32(rocdl.readfirstlane(T.i32, tid // WAVE))
        lane = fx.Int32(tid) % fx.Int32(WAVE)
        base = fx.SharedAllocator(static=False).allocate(waves * batch_push_lds_bytes())._ptr
        lds_out = fx.recast_iter(_I16_SHARED(), base) + wave * fx.Int32(batch_push_lds_bytes() // 2)
        win = _cco.Window(fx.Int64(arena_handle))
        lsa0 = fx.Int64(win.lsa_ptr(fx.Int32(0), 0))
        peer_stride = fx.Int64(win.lsa_ptr(fx.Int32(1), 0)) - lsa0
        nvt = fx.Int32(buffer_load(create_buffer_resource_from_addr(addr_nvt.ir_value()),
                                   fx.Int32(0), vec_width=1, dtype=T.i32, is_scalar=True))
        emit_batched_pusher(
            fx.block_idx.x * fx.Int32(waves) + wave,
            block_num * waves,
            lane,
            lds_out,
            nvt,
            out_rsrc=create_buffer_resource_from_addr(
                addr_out.ir_value(),
                num_records_bytes=(fx.Int64(out_rows) * fx.Int64(hidden * 2)).ir_value(),
            ),
            out_rows=out_rows,
            rows_rsrc=create_buffer_resource_from_addr(addr_rows.ir_value()),
            tis_rsrc=create_buffer_resource_from_addr(addr_tis.ir_value()),
            lsa0=lsa0,
            peer_stride=peer_stride,
            combine_input_offset=combine_input_offset,
            slot_stride_bytes=slot_stride_bytes,
            world_size=world_size,
            topk=topk,
            max_tok=max_tok,
            hidden=hidden,
        )

    @flyc.jit
    def launch(
        addr_out: fx.Int64,
        out_rows: fx.Int32,
        addr_rows: fx.Int64,
        addr_nvt: fx.Int64,
        addr_tis: fx.Int64,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
    ):
        ep_dedup_bpush(addr_out, out_rows, addr_rows, addr_nvt, addr_tis).launch(
            grid=(block_num, 1, 1), block=[waves * WAVE, 1, 1], stream=stream
        )

    return launch


# ---------------------------------------------------------------------------
# LDS-metadata push. On a CU with peer TDM stores in flight, every VMEM load
# queues behind them (p2p_bw: 4 consumed loads per store take a 64-CU push
# from 1.23 to 0.27 TB/s; local stores barely notice). So this push does all
# of its VMEM reads -- the metadata -- before its first store, into LDS, and
# moves the data with TDM only: waves 0-1 gather each token's rows, waves 2-3
# take turns sending a whole trip (LP_TOKS tokens) as one gather store.
# ---------------------------------------------------------------------------
LP_TOKS = 4
LP_CHUNKS = int(os.environ.get("AITER_EP_DEDUP_LP_CHUNKS", "4"))
LP_NIN = int(os.environ.get("AITER_EP_DEDUP_LP_NIN", "3"))
LP_NOUTW = int(os.environ.get("AITER_EP_DEDUP_LP_NOUTW", "4"))  # out tiles per store wave


def build_dedup_lds_push_module(
    *,
    arena_handle,
    combine_input_offset,
    slot_stride_bytes,
    topk,
    max_tok,
    hidden,
    world_size,
    block_num,
):
    import mori.cco.device.flydsl as _cco
    from .tdm_gather_shim import tensor_load_gather

    waves = 4
    lanes = waves * WAVE
    row_e = LP_CHUNKS * RP_CHUNK
    in_rows = LP_TOKS * topk
    in_b = in_rows * row_e * 2
    out_b = LP_TOKS * row_e * 2
    assert hidden % row_e == 0
    iters = hidden // row_e
    slots = LP_TOKS * row_e // 16
    assert slots % lanes == 0
    rounds = slots // lanes
    slots_per_tok = row_e // 16
    gathers = [(r0, min(r0 + 8, topk)) for r0 in range(0, topk, 8)]
    max_recv = max_tok * world_size
    max_groups = (max_recv + LP_TOKS - 1) // LP_TOKS
    groups_per_block = (max_groups + block_num - 1) // block_num
    meta_w = topk + 1
    meta_b = ((groups_per_block * LP_TOKS * meta_w * 4 + 511) // 512) * 512
    loads_per_wave_trip = (LP_TOKS // 2) * len(gathers)
    knobs = KNOBS_TAG

    @flyc.kernel(
        name=f"ep_dedup_lpush_{knobs}_c{LP_CHUNKS}i{LP_NIN}o{LP_NOUTW}_b{block_num}",
        known_block_size=[lanes, 1, 1],
    )
    def ep_dedup_lpush(
        addr_out: fx.Int64,
        out_rows: fx.Int32,
        addr_rows: fx.Int64,
        addr_nvt: fx.Int64,
        addr_tis: fx.Int64,
    ):
        tid = fx.Int32(fx.thread_idx.x)
        bid = fx.block_idx.x
        wave = fx.Int32(rocdl.readfirstlane(T.i32, tid // fx.Int32(WAVE)))
        lds_load_b32, lds_store_b32 = make_lds_copy_ops(32)
        lds_load_b128, lds_store_b128 = make_lds_copy_ops(128)
        smem = fx.SharedAllocator(static=False)
        in_base = smem.allocate(LP_NIN * in_b)._ptr
        out_base = smem.allocate(2 * LP_NOUTW * out_b)._ptr
        meta_base = smem.allocate(meta_b)._ptr
        in_idx = fx.Int32(fx.ptrtoint(in_base))
        out_idx = fx.Int32(fx.ptrtoint(out_base))
        meta_idx = fx.Int32(fx.ptrtoint(meta_base))

        win = _cco.Window(fx.Int64(arena_handle))
        lsa0 = fx.Int64(win.lsa_ptr(fx.Int32(0), 0))
        peer_stride = fx.Int64(win.lsa_ptr(fx.Int32(1), 0)) - lsa0
        k_slots = fx.Int32(peer_stride // fx.Int64(slot_stride_bytes))
        oob = k_slots * fx.Int32(world_size)
        rows_rsrc = create_buffer_resource_from_addr(addr_rows.ir_value())
        tis_rsrc = create_buffer_resource_from_addr(addr_tis.ir_value())
        nvt = fx.Int32(buffer_load(create_buffer_resource_from_addr(addr_nvt.ir_value()),
                                   fx.Int32(0), vec_width=1, dtype=T.i32, is_scalar=True))
        n_groups = (nvt + fx.Int32(LP_TOKS - 1)) // fx.Int32(LP_TOKS)
        my_groups = (n_groups - bid + fx.Int32(block_num - 1)) // fx.Int32(block_num)
        my_groups = (my_groups < fx.Int32(0)).select(fx.Int32(0), my_groups)
        n_trips = my_groups * fx.Int32(iters)

        # -- every VMEM read of the kernel, before its first peer store --
        for x in range(tid, my_groups * fx.Int32(LP_TOKS), lanes):
            j = x // fx.Int32(LP_TOKS)
            ti = x - j * fx.Int32(LP_TOKS)
            tok = (bid + j * fx.Int32(block_num)) * fx.Int32(LP_TOKS) + ti
            valid = tok < nvt
            safe = valid.select(tok, fx.Int32(0))
            lead = fx.Int32(topk)
            for k in range_constexpr(topk):
                r = fx.Int32(buffer_load(rows_rsrc, safe * fx.Int32(topk) + fx.Int32(k),
                                         vec_width=1, dtype=T.i32))
                local = valid & (r >= fx.Int32(0))
                lead = (local & (lead == fx.Int32(topk))).select(fx.Int32(k), lead)
                lds_store_b32(meta_idx, (x * fx.Int32(meta_w) + fx.Int32(k)) * fx.Int32(4),
                              Vec.from_elements([local.select(r, out_rows)], fx.Int32).ir_value())
            enc = fx.Int32(buffer_load(tis_rsrc, safe, vec_width=1, dtype=T.i32))
            origin_pe = enc // fx.Int32(max_tok)
            origin_lid = enc - origin_pe * fx.Int32(max_tok)
            dst = (valid & (lead < fx.Int32(topk))).select(
                origin_pe * k_slots + origin_lid * fx.Int32(topk) + lead, oob
            )
            lds_store_b32(meta_idx, (x * fx.Int32(meta_w) + fx.Int32(topk)) * fx.Int32(4),
                          Vec.from_elements([dst], fx.Int32).ir_value())
        rocdl.s_wait_dscnt(0)
        gpu.barrier()

        def meta_at(x, field):
            v = lds_load_b32(meta_idx, (x * fx.Int32(meta_w) + fx.Int32(field)) * fx.Int32(4))[0]
            return fx.Int32(rocdl.readfirstlane(T.i32, fx.Int32(v)))

        src_view = fx.Tensor(
            fx.make_view(
                fx.inttoptr(_I16_GLOBAL(), addr_out),
                fx.make_layout((out_rows, row_e), (hidden, 1)),
            )
        )
        comb_view = fx.Tensor(
            fx.make_view(
                fx.inttoptr(_I16_GLOBAL(), lsa0 + fx.Int64(combine_input_offset)),
                fx.make_layout((oob, row_e), (slot_stride_bytes // 2, 1)),
            )
        )

        def lds_in(buf):
            return fx.Tensor(
                fx.make_view(
                    fx.recast_iter(_I16_SHARED(), in_base) + buf * fx.Int32(in_rows * row_e),
                    fx.make_layout((in_rows, row_e), (row_e, 1)),
                )
            )

        lds_out = fx.Tensor(
            fx.make_view(
                fx.recast_iter(_I16_SHARED(), out_base),
                fx.make_layout((2 * LP_NOUTW * LP_TOKS, row_e), (row_e, 1)),
            )
        )

        def trip_coords(trip):
            j = trip // fx.Int32(iters)
            return j, (trip - j * fx.Int32(iters)) * fx.Int32(row_e)

        def issue_load(trip, ti, ib):
            j, e0 = trip_coords(trip)
            live = trip < n_trips
            x = live.select(j * fx.Int32(LP_TOKS) + ti, fx.Int32(0))
            rows = [live.select(meta_at(x, k), out_rows) for k in range_constexpr(topk)]
            for r0, r1 in gathers:
                desc = make_tensor_gather_descriptor(
                    src_view,
                    lds_in(ib),
                    rows[r0:r1],
                    row_width=row_e,
                    tensor_dim0=row_e,
                    tensor_dim1=out_rows.ir_value(),
                    stride=hidden,
                    elem_bytes=2,
                    index_size=32,
                    lds_byte_offset=fx.index_cast(
                        T.index, (ti * fx.Int32(topk) + fx.Int32(r0)) * fx.Int32(row_e * 2)
                    ),
                    global_byte_offset=fx.Int64(e0) * fx.Int64(2),
                )
                tensor_load_gather(desc, cache_policy=LOAD_CPOL)

        def load_trip(trip, ib):
            for h in range_constexpr(LP_TOKS // 2):
                issue_load(trip, wave * fx.Int32(LP_TOKS // 2) + fx.Int32(h), ib)

        def store_trip(trip, ob):
            j, e0 = trip_coords(trip)
            # STORE_LOCAL (timing only): the same store aimed at local rows.
            idx = [meta_at(j * fx.Int32(LP_TOKS) + fx.Int32(ti), 0 if STORE_LOCAL else topk)
                   for ti in range_constexpr(LP_TOKS)]
            desc = make_tensor_gather_descriptor(
                src_view if STORE_LOCAL else comb_view,
                lds_out,
                idx,
                row_width=row_e,
                tensor_dim0=row_e,
                tensor_dim1=(out_rows if STORE_LOCAL else oob).ir_value(),
                stride=hidden if STORE_LOCAL else slot_stride_bytes // 2,
                elem_bytes=2,
                index_size=32,
                lds_byte_offset=fx.index_cast(T.index, ob * fx.Int32(out_b)),
                global_byte_offset=fx.Int64(e0) * fx.Int64(2),
            )
            if const_expr(not NO_STORE):
                tensor_store_gather(desc)

        def reduce_tile(ib, ob):
            for rd in range_constexpr(rounds):
                slot = tid + fx.Int32(rd * lanes)
                ti = slot // fx.Int32(slots_per_tok)
                e = (slot - ti * fx.Int32(slots_per_tok)) * fx.Int32(16)
                accs = [Vec.filled(8, 0.0, fx.Float32) for _ in range_constexpr(2)]
                for k in range_constexpr(topk):
                    row_b = (ib * fx.Int32(in_rows) + ti * fx.Int32(topk) + fx.Int32(k)) * fx.Int32(row_e * 2)
                    for q in range_constexpr(2):
                        raw = Vec(lds_load_b128(in_idx, row_b + e * fx.Int32(2) + fx.Int32(q * 16)))
                        accs[q] = accs[q] + raw.bitcast(fx.BFloat16).to(fx.Float32)
                ob_off = ((ob * fx.Int32(LP_TOKS) + ti) * fx.Int32(row_e) + e) * fx.Int32(2)
                for q in range_constexpr(2):
                    lds_store_b128(out_idx, ob_off + fx.Int32(q * 16),
                                   accs[q].to(fx.BFloat16).bitcast(fx.Int32).ir_value())

        is_loader = wave < fx.Int32(2)
        if is_loader:
            for p in range_constexpr(LP_NIN - 1):
                load_trip(fx.Int32(p), fx.Int32(p))
        for trip in range(n_trips):
            ib = trip % fx.Int32(LP_NIN)
            ob = trip % fx.Int32(2 * LP_NOUTW)
            sender = fx.Int32(2) + trip % fx.Int32(2)
            if is_loader:
                # Zero-row gathers past the end keep the count uniform.
                load_trip(trip + fx.Int32(LP_NIN - 1), (trip + fx.Int32(LP_NIN - 1)) % fx.Int32(LP_NIN))
                # Local loads retire in order: the newer LP_NIN-1 trips may fly.
                tdm_ops.tensor_wait((LP_NIN - 1) * loads_per_wave_trip)
            if wave == sender:
                # Peer stores need not retire in order, so drain at each wrap of
                # this wave's LP_NOUTW tiles instead of trusting a partial count.
                if (trip // fx.Int32(2)) % fx.Int32(LP_NOUTW) == fx.Int32(0):
                    tdm_ops.tensor_wait(0)
            gpu.barrier()
            reduce_tile(ib, ob)
            rocdl.s_wait_dscnt(0)
            gpu.barrier()
            if wave == sender:
                store_trip(trip, ob)
        tdm_ops.tensor_wait(0)

    @flyc.jit
    def launch(
        addr_out: fx.Int64,
        out_rows: fx.Int32,
        addr_rows: fx.Int64,
        addr_nvt: fx.Int64,
        addr_tis: fx.Int64,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
    ):
        ep_dedup_lpush(addr_out, out_rows, addr_rows, addr_nvt, addr_tis).launch(
            grid=(block_num, 1, 1), block=[lanes, 1, 1], stream=stream
        )

    return launch


# ---------------------------------------------------------------------------
# Paired push: the LDS-metadata push without workgroup barriers. Waves (0, 2)
# and (1, 3) are two independent pipelines over alternate trips. In each, the
# loader gathers and sums the trip's tokens and flags the output tile ready in
# LDS; the sender waits for the flag, sends the tile, and only after draining
# its stores flags tiles free again. A sender waiting on peer stores then
# stalls nothing but itself, until its loader runs out of free tiles.
# ---------------------------------------------------------------------------
PP_TOKS = 4
PP_CHUNKS = int(os.environ.get("AITER_EP_DEDUP_PP_CHUNKS", "2"))
PP_NIN = int(os.environ.get("AITER_EP_DEDUP_PP_NIN", "2"))
PP_NOUT = int(os.environ.get("AITER_EP_DEDUP_PP_NOUT", "6"))
PP_NO_LOAD = os.environ.get("AITER_EP_DEDUP_PP_NO_LOAD", "0") == "1"
PP_NO_REDUCE = os.environ.get("AITER_EP_DEDUP_PP_NO_REDUCE", "0") == "1"
# Sender waves per pair; sender s sends rounds s, s+PP_SEND, ... of PP_NOUT tiles,
# so one drains its round while the next sends.
PP_SEND = int(os.environ.get("AITER_EP_DEDUP_PP_SEND", "2"))
# Rows every token sums unconditionally, all loads of a trip issued at once;
# rows past c are zero-filled, rows past PP_RROWS take a branch per row.
PP_RROWS = int(os.environ.get("AITER_EP_DEDUP_PP_RROWS", "3"))
# Rows every token gathers up front; a trip with a token past this loads the
# rest on demand.
PP_GROWS = int(os.environ.get("AITER_EP_DEDUP_PP_GROWS", "4"))
# A reducer wave per pair, so the loader only keeps TDM loads in flight.
PP_SPLIT = int(os.environ.get("AITER_EP_DEDUP_PP_SPLIT", "1"))


def _lds_poll_ge(addr_i64, target):
    """Spin until the LDS word reaches ``target``; an atomic add of 0 is a load
    the compiler cannot hoist out of the loop."""
    v = fx.Int32(atomic_add_lds(addr_i64, fx.Int32(0)))
    while v < target:
        rocdl.s_sleep(1)
        v = fx.Int32(atomic_add_lds(addr_i64, fx.Int32(0)))


_lds_poll_ge = traced(_lds_poll_ge)


def build_dedup_pair_push_module(
    *,
    arena_handle,
    combine_input_offset,
    slot_stride_bytes,
    topk,
    max_tok,
    hidden,
    world_size,
    block_num,
):
    import mori.cco.device.flydsl as _cco
    from .tdm_gather_shim import tensor_load_gather

    waves = 2 * (1 + PP_SPLIT + PP_SEND)
    lanes = waves * WAVE
    n_out = PP_NOUT * PP_SEND  # output tiles per pair
    row_e = PP_CHUNKS * RP_CHUNK
    in_rows = PP_TOKS * topk
    in_b = in_rows * row_e * 2
    out_b = PP_TOKS * row_e * 2
    assert hidden % row_e == 0
    iters = hidden // row_e
    slots = PP_TOKS * row_e // 16
    assert slots % WAVE == 0
    rounds = slots // WAVE  # one loader wave sums a whole trip
    slots_per_tok = row_e // 16
    max_recv = max_tok * world_size
    assert slots_per_tok == WAVE and rounds == PP_TOKS  # round rd sums token rd
    max_groups = (max_recv + PP_TOKS - 1) // PP_TOKS
    groups_per_block = (max_groups + block_num - 1) // block_num
    # Per token: its c local rows first then out_rows, dst, c, a scratch word.
    meta_w = topk + 3
    f_dst, f_cnt, f_junk = topk, topk + 1, topk + 2
    meta_b = ((groups_per_block * PP_TOKS * meta_w * 4 + 511) // 512) * 512
    pair_b = PP_NIN * in_b + n_out * out_b
    # Per pair: ready[n_out], free[n_out], in_ready[PP_NIN], in_free[PP_NIN].
    fl_w = 2 * n_out + 2 * PP_NIN
    flags_b = 2 * fl_w * 4
    grows = min(PP_GROWS, topk)
    ovf = topk - grows
    assert ovf <= 8
    main_rows = PP_TOKS * grows
    main_gathers = [(r0, min(r0 + 8, main_rows)) for r0 in range(0, main_rows, 8)]
    loads_per_trip = len(main_gathers)
    rrows = min(PP_RROWS, grows)
    knobs = KNOBS_TAG

    @flyc.kernel(
        name=f"ep_dedup_ppush5_{knobs}_c{PP_CHUNKS}i{PP_NIN}o{PP_NOUT}s{PP_SEND}r{rrows}g{grows}"
        f"x{PP_SPLIT}"
        f"l{int(PP_NO_LOAD)}r{int(PP_NO_REDUCE)}_b{block_num}",
        known_block_size=[lanes, 1, 1],
    )
    def ep_dedup_ppush(
        addr_out: fx.Int64,
        out_rows: fx.Int32,
        addr_rows: fx.Int64,
        addr_nvt: fx.Int64,
        addr_tis: fx.Int64,
    ):
        tid = fx.Int32(fx.thread_idx.x)
        bid = fx.block_idx.x
        wave = fx.Int32(rocdl.readfirstlane(T.i32, tid // fx.Int32(WAVE)))
        lane = tid % fx.Int32(WAVE)
        pair = wave % fx.Int32(2)
        role = wave // fx.Int32(2)
        is_loader = role == fx.Int32(0)
        is_reducer = role == fx.Int32(1 if PP_SPLIT else -1)
        sender = role - fx.Int32(1 + PP_SPLIT)
        lds_load_b32, lds_store_b32 = make_lds_copy_ops(32)
        lds_load_b128, lds_store_b128 = make_lds_copy_ops(128)
        smem = fx.SharedAllocator(static=False)
        pair_base = smem.allocate(2 * pair_b)._ptr
        meta_base = smem.allocate(meta_b)._ptr
        flag_base = smem.allocate(flags_b)._ptr
        meta_idx = fx.Int32(fx.ptrtoint(meta_base))
        flag_addr = fx.Int64(fx.ptrtoint(flag_base))
        my_base_b = pair * fx.Int32(pair_b)  # this pair's tiles, bytes from pair_base
        in_idx = fx.Int32(fx.ptrtoint(pair_base)) + my_base_b
        out_idx = in_idx + fx.Int32(PP_NIN * in_b)

        def flag_at(off, i):
            return flag_addr + fx.Int64((pair * fx.Int32(fl_w) + fx.Int32(off) + i) * fx.Int32(4))

        def ready_addr(ob):
            return flag_at(0, ob)

        def free_addr(ob):
            return flag_at(n_out, ob)

        def in_ready_addr(ib):
            return flag_at(2 * n_out, ib)

        def in_free_addr(ib):
            return flag_at(2 * n_out + PP_NIN, ib)

        win = _cco.Window(fx.Int64(arena_handle))
        lsa0 = fx.Int64(win.lsa_ptr(fx.Int32(0), 0))
        peer_stride = fx.Int64(win.lsa_ptr(fx.Int32(1), 0)) - lsa0
        k_slots = fx.Int32(peer_stride // fx.Int64(slot_stride_bytes))
        oob = k_slots * fx.Int32(world_size)
        rows_rsrc = create_buffer_resource_from_addr(addr_rows.ir_value())
        tis_rsrc = create_buffer_resource_from_addr(addr_tis.ir_value())
        nvt = fx.Int32(buffer_load(create_buffer_resource_from_addr(addr_nvt.ir_value()),
                                   fx.Int32(0), vec_width=1, dtype=T.i32, is_scalar=True))
        n_groups = (nvt + fx.Int32(PP_TOKS - 1)) // fx.Int32(PP_TOKS)
        my_groups = (n_groups - bid + fx.Int32(block_num - 1)) // fx.Int32(block_num)
        my_groups = (my_groups < fx.Int32(0)).select(fx.Int32(0), my_groups)
        n_trips = my_groups * fx.Int32(iters)
        # This pair's trips: pair, pair+2, ...
        n_mine = (n_trips - pair + fx.Int32(1)) // fx.Int32(2)
        n_mine = (n_mine < fx.Int32(0)).select(fx.Int32(0), n_mine)

        # -- every VMEM read of the kernel, before its first peer store --
        for x in range(tid, my_groups * fx.Int32(PP_TOKS), lanes):
            j = x // fx.Int32(PP_TOKS)
            ti = x - j * fx.Int32(PP_TOKS)
            tok = (bid + j * fx.Int32(block_num)) * fx.Int32(PP_TOKS) + ti
            valid = tok < nvt
            safe = valid.select(tok, fx.Int32(0))

            def put(field, v):
                lds_store_b32(meta_idx, (x * fx.Int32(meta_w) + field) * fx.Int32(4),
                              Vec.from_elements([v], fx.Int32).ir_value())

            rs = [fx.Int32(buffer_load(rows_rsrc, safe * fx.Int32(topk) + fx.Int32(k),
                                       vec_width=1, dtype=T.i32))
                  for k in range_constexpr(topk)]
            for k in range_constexpr(topk):
                put(fx.Int32(k), out_rows)
            lead = fx.Int32(topk)
            cnt = fx.Int32(0)
            for k in range_constexpr(topk):
                local = valid & (rs[k] >= fx.Int32(0))
                lead = (local & (lead == fx.Int32(topk))).select(fx.Int32(k), lead)
                put(local.select(cnt, fx.Int32(f_junk)), rs[k])
                cnt = cnt + local.select(fx.Int32(1), fx.Int32(0))
            enc = fx.Int32(buffer_load(tis_rsrc, safe, vec_width=1, dtype=T.i32))
            origin_pe = enc // fx.Int32(max_tok)
            origin_lid = enc - origin_pe * fx.Int32(max_tok)
            dst = (valid & (lead < fx.Int32(topk))).select(
                origin_pe * k_slots + origin_lid * fx.Int32(topk) + lead, oob
            )
            put(fx.Int32(f_dst), dst)
            put(fx.Int32(f_cnt), cnt)
        for i in range(tid, 2 * fl_w, lanes):
            store_i32_lds(flag_addr + fx.Int64(i * fx.Int32(4)), fx.Int32(0))
        rocdl.s_wait_dscnt(0)
        gpu.barrier()

        def meta_at(x, field):
            v = lds_load_b32(meta_idx, (x * fx.Int32(meta_w) + fx.Int32(field)) * fx.Int32(4))[0]
            return fx.Int32(rocdl.readfirstlane(T.i32, fx.Int32(v)))

        src_view = fx.Tensor(
            fx.make_view(
                fx.inttoptr(_I16_GLOBAL(), addr_out),
                fx.make_layout((out_rows, row_e), (hidden, 1)),
            )
        )
        comb_view = fx.Tensor(
            fx.make_view(
                fx.inttoptr(_I16_GLOBAL(), lsa0 + fx.Int64(combine_input_offset)),
                fx.make_layout((oob, row_e), (slot_stride_bytes // 2, 1)),
            )
        )
        in_view = fx.Tensor(
            fx.make_view(
                fx.recast_iter(_I16_SHARED(), pair_base) + my_base_b // fx.Int32(2),
                fx.make_layout((PP_NIN * in_rows, row_e), (row_e, 1)),
            )
        )
        out_view = fx.Tensor(
            fx.make_view(
                fx.recast_iter(_I16_SHARED(), pair_base)
                + (my_base_b + fx.Int32(PP_NIN * in_b)) // fx.Int32(2),
                fx.make_layout((n_out * PP_TOKS, row_e), (row_e, 1)),
            )
        )

        def trip_coords(m):
            """m-th trip of this pair."""
            trip = pair + m * fx.Int32(2)
            j = trip // fx.Int32(iters)
            return j, (trip - j * fx.Int32(iters)) * fx.Int32(row_e)

        def row_off(ti, k):
            """Byte offset in an input tile: every token's first grows rows packed
            together, so the regular gathers carry no padding, then the rest."""
            row = ti * grows + k if k < grows else main_rows + ti * ovf + (k - grows)
            return row * row_e * 2

        def gather_in(rows, ib, row0, e0):
            desc = make_tensor_gather_descriptor(
                src_view, in_view, rows,
                row_width=row_e, tensor_dim0=row_e, tensor_dim1=out_rows.ir_value(),
                stride=hidden, elem_bytes=2, index_size=32,
                lds_byte_offset=fx.index_cast(
                    T.index, ib * fx.Int32(in_b) + fx.Int32(row0 * row_e * 2)
                ),
                global_byte_offset=fx.Int64(e0) * fx.Int64(2),
            )
            if const_expr(not PP_NO_LOAD):
                tensor_load_gather(desc, cache_policy=LOAD_CPOL)

        def load_trip(m, ib):
            j, e0 = trip_coords(m)
            live = m < n_mine
            rows = []
            for ti in range_constexpr(PP_TOKS):
                x = live.select(j * fx.Int32(PP_TOKS) + fx.Int32(ti), fx.Int32(0))
                rows += [live.select(meta_at(x, k), out_rows) for k in range_constexpr(grows)]
            for r0, r1 in main_gathers:
                gather_in(rows[r0:r1], ib, r0, e0)

        def load_overflow(m, ib, cs):
            j, e0 = trip_coords(m)
            for ti in range_constexpr(PP_TOKS):
                if cs[ti] > fx.Int32(grows):
                    x = j * fx.Int32(PP_TOKS) + fx.Int32(ti)
                    gather_in([meta_at(x, k) for k in range_constexpr(grows, topk)],
                              ib, main_rows + ti * ovf, e0)

        accs = [fx.make_rmem_tensor(8, fx.Float32) for _ in range_constexpr(2)]
        lane_b = lane * fx.Int32(32)  # a lane sums 16 elements of each row

        def add_row(base, ti, k, first):
            for q in range_constexpr(2):
                raw = Vec(lds_load_b128(in_idx, base + fx.Int32(row_off(ti, k) + q * 16)))
                v = raw.bitcast(fx.BFloat16).to(fx.Float32)
                accs[q].store(v if first else accs[q].load() + v)

        def reduce_trip(ib, ob, cs):
            # A token's c local rows come first, the rest zero-filled.
            base = ib * fx.Int32(in_b) + lane_b
            raws = [[[lds_load_b128(in_idx, base + fx.Int32(row_off(ti, k) + q * 16))
                      for q in range_constexpr(2)]
                     for k in range_constexpr(rrows)]
                    for ti in range_constexpr(PP_TOKS)]
            for ti in range_constexpr(PP_TOKS):
                for q in range_constexpr(2):
                    acc = Vec(raws[ti][0][q]).bitcast(fx.BFloat16).to(fx.Float32)
                    for k in range_constexpr(1, rrows):
                        acc = acc + Vec(raws[ti][k][q]).bitcast(fx.BFloat16).to(fx.Float32)
                    accs[q].store(acc)
                if const_expr(rrows < topk):
                    if cs[ti] > fx.Int32(rrows):
                        for k in range_constexpr(rrows, topk):
                            if cs[ti] > fx.Int32(k):
                                add_row(base, ti, k, False)
                ob_off = (ob * fx.Int32(PP_TOKS) + fx.Int32(ti)) * fx.Int32(row_e * 2) + lane_b
                for q in range_constexpr(2):
                    lds_store_b128(out_idx, ob_off + fx.Int32(q * 16),
                                   Vec(accs[q].load()).to(fx.BFloat16).bitcast(fx.Int32).ir_value())

        def store_trip(m, ob):
            j, e0 = trip_coords(m)
            idx = [meta_at(j * fx.Int32(PP_TOKS) + fx.Int32(ti), 0 if STORE_LOCAL else f_dst)
                   for ti in range_constexpr(PP_TOKS)]
            desc = make_tensor_gather_descriptor(
                src_view if STORE_LOCAL else comb_view, out_view, idx,
                row_width=row_e, tensor_dim0=row_e,
                tensor_dim1=(out_rows if STORE_LOCAL else oob).ir_value(),
                stride=hidden if STORE_LOCAL else slot_stride_bytes // 2,
                elem_bytes=2, index_size=32,
                lds_byte_offset=fx.index_cast(T.index, ob * fx.Int32(out_b)),
                global_byte_offset=fx.Int64(e0) * fx.Int64(2),
            )
            if const_expr(not NO_STORE):
                tensor_store_gather(desc)

        def trip_counts(m):
            j, _ = trip_coords(m)
            return [meta_at(j * fx.Int32(PP_TOKS) + fx.Int32(ti), f_cnt)
                    for ti in range_constexpr(PP_TOKS)]

        def reduce_step(m, ib, ob, cs):
            # The tile's previous use, n_out trips back, must have been sent.
            _lds_poll_ge(free_addr(ob), m // fx.Int32(n_out))
            if const_expr(not PP_NO_REDUCE):
                reduce_trip(ib, ob, cs)
            rocdl.s_wait_dscnt(0)

        if is_loader:
            for p in range_constexpr(PP_NIN - 1):
                load_trip(fx.Int32(p), fx.Int32(p))
            for m in range(n_mine):
                ib = m % fx.Int32(PP_NIN)
                t = m + fx.Int32(PP_NIN - 1)
                if const_expr(PP_SPLIT):
                    # Trip t's tile last held trip t - PP_NIN; the reducer frees it.
                    _lds_poll_ge(in_free_addr(t % fx.Int32(PP_NIN)), t // fx.Int32(PP_NIN))
                # Zero-row gathers past the end keep the count uniform.
                load_trip(t, t % fx.Int32(PP_NIN))
                tdm_ops.tensor_wait((PP_NIN - 1) * loads_per_trip)
                cs = trip_counts(m)
                if const_expr(ovf > 0):
                    big = cs[0] > fx.Int32(grows)
                    for ti in range_constexpr(1, PP_TOKS):
                        big = big | (cs[ti] > fx.Int32(grows))
                    if big:
                        # Rare: rows past grows, loaded on demand; drains the prefetch too.
                        load_overflow(m, ib, cs)
                        tdm_ops.tensor_wait(0)
                if const_expr(PP_SPLIT):
                    if lane == fx.Int32(0):
                        store_i32_lds(in_ready_addr(ib), m + fx.Int32(1))
                else:
                    ob = m % fx.Int32(n_out)
                    reduce_step(m, ib, ob, cs)
                    if lane == fx.Int32(0):
                        store_i32_lds(ready_addr(ob), m + fx.Int32(1))
        elif is_reducer:
            for m in range(n_mine):
                ib = m % fx.Int32(PP_NIN)
                ob = m % fx.Int32(n_out)
                _lds_poll_ge(in_ready_addr(ib), m + fx.Int32(1))
                reduce_step(m, ib, ob, trip_counts(m))
                if lane == fx.Int32(0):
                    store_i32_lds(in_free_addr(ib), m // fx.Int32(PP_NIN) + fx.Int32(1))
                    store_i32_lds(ready_addr(ob), m + fx.Int32(1))
        else:
            # Round r is trips r*PP_NOUT.. of tiles (r % PP_SEND)*PP_NOUT..; its
            # tiles' u-th use, u = r // PP_SEND, ends once its stores drain.
            n_rounds = (n_mine + fx.Int32(PP_NOUT - 1)) // fx.Int32(PP_NOUT)
            for r in range(sender, n_rounds, PP_SEND):
                for i in range_constexpr(PP_NOUT):
                    m = r * fx.Int32(PP_NOUT) + fx.Int32(i)
                    if m < n_mine:
                        ob = m % fx.Int32(n_out)
                        _lds_poll_ge(ready_addr(ob), m + fx.Int32(1))
                        store_trip(m, ob)
                tdm_ops.tensor_wait(0)
                if lane < fx.Int32(PP_NOUT):
                    store_i32_lds(
                        free_addr((r % fx.Int32(PP_SEND)) * fx.Int32(PP_NOUT) + lane),
                        r // fx.Int32(PP_SEND) + fx.Int32(1),
                    )
        tdm_ops.tensor_wait(0)

    @flyc.jit
    def launch(
        addr_out: fx.Int64,
        out_rows: fx.Int32,
        addr_rows: fx.Int64,
        addr_nvt: fx.Int64,
        addr_tis: fx.Int64,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
    ):
        ep_dedup_ppush(addr_out, out_rows, addr_rows, addr_nvt, addr_tis).launch(
            grid=(block_num, 1, 1), block=[lanes, 1, 1], stream=stream
        )

    return launch


# ---------------------------------------------------------------------------
# Step push: the shape of an in-grid pusher block (4 waves). The recv slots are
# split evenly over the blocks and, within a block, over its waves; each wave
# runs its own slots through load -> sum -> send, column chunk by column chunk
# (a chunk is the 2 N tiles one flag would release). A step is up to 8 slots
# whose local rows, packed back to back, fit SP_RESV rows: one wave packs its
# slots into steps once, up front. Waves never wait on each other; LDS rings of
# SP_NIN input and SP_NOUT output tiles let a wave's loads, sums and peer stores
# of different steps overlap.
#
# A wave's loads and stores share its tensorcnt, so "this step's loads have
# landed" is a count of the TDM ops issued after them -- correct only if TDM
# ops retire in issue order. The same wait then also proves every store at
# least SP_NIN steps old has retired, which frees the output tile once
# SP_NOUT >= SP_NIN.
# ---------------------------------------------------------------------------
SP_NIN = int(os.environ.get("AITER_EP_DEDUP_SP_NIN", "2"))
SP_NOUT = int(os.environ.get("AITER_EP_DEDUP_SP_NOUT", str(SP_NIN)))
SP_RROWS = int(os.environ.get("AITER_EP_DEDUP_SP_RROWS", "3"))
SP_RESV = int(os.environ.get("AITER_EP_DEDUP_SP_RESV", "28"))
# SP_SENDER=1: waves 0-2 load and sum, wave 3 only sends. Then a worker's
# tensorcnt holds just its loads, and the sender frees a worker's output tile
# once its store, SW_LAG rounds back, has retired; SW_NOUT tiles per worker.
SP_SENDER = int(os.environ.get("AITER_EP_DEDUP_SP_SENDER", "0"))
SW_NOUT = int(os.environ.get("AITER_EP_DEDUP_SW_NOUT", "4"))
SW_LAG = int(os.environ.get("AITER_EP_DEDUP_SW_LAG", "2"))
# SP_FIXED=F: a slot's first F rows sit at fixed tile rows (constant offsets in
# the sum); rows past F go to a per-step overflow area of SP_OV rows.
SP_FIXED = int(os.environ.get("AITER_EP_DEDUP_SP_FIXED", "0"))
SP_OV = int(os.environ.get("AITER_EP_DEDUP_SP_OV", "8"))
# Timing only: sum just the first SP_RTOK slots of each step.
SP_RTOK = int(os.environ.get("AITER_EP_DEDUP_SP_RTOK", "8"))
# Sum a slot's first rows in packed bf16 (rounding at every add) instead of f32.
SP_BF16ACC = int(os.environ.get("AITER_EP_DEDUP_SP_BF16ACC", "0"))
# Lane layout in a 1 KB row: stride 16 (q step 512) or the old stride 32 (q step 16).
SP_LSTRIDE = int(os.environ.get("AITER_EP_DEDUP_SP_LSTRIDE", "16"))
SP_QSTEP = 512 if SP_LSTRIDE == 16 else 16
# Timing only: per-phase realtime ticks of each worker wave, printed at exit.
SP_PROF = int(os.environ.get("AITER_EP_DEDUP_SP_PROF", "0"))
# Slots whose rows are loaded together in the sum (8: the whole step at once).
SP_SUMBATCH = int(os.environ.get("AITER_EP_DEDUP_SP_SUMBATCH", "4"))
# SP_WMMA=1 (with SP_FIXED=3, 32 rows a step): the sum is S[16 x 32] @ X[32 x 512]
# on the matrix cores, S the 0/1 slot-of-row matrix built in registers, X read
# with transposing LDS loads from input tiles padded 16 B a row.
SP_WMMA = int(os.environ.get("AITER_EP_DEDUP_SP_WMMA", "0"))
SP_WMMA_BATCH = int(os.environ.get("AITER_EP_DEDUP_SP_WMMA_BATCH", "8"))  # 16-column blocks per batch
_SP_PROF_PHASES = ("issue loads", "wait loads", "wait free tile", "sum", "lds drain+flag")


def _realtime_lo():
    from flydsl._mlir.dialects import llvm as _llvm

    t = _llvm.InlineAsmOp(
        res=T.i64, operands_=[],
        asm_string="s_sendmsg_rtn_b64 $0, sendmsg(MSG_RTN_GET_REALTIME)\ns_wait_kmcnt 0",
        constraints="=s", has_side_effects=True, is_align_stack=False).result
    return fx.Int32(fx.Int64(t) & fx.Int64(0x7FFFFFFF))


def _sp_prof_buffer(n_words):
    import atexit

    buf = torch.zeros(n_words, dtype=torch.int32, device="cuda")

    def report():
        t = buf.view(-1, 4, 8).cpu().double()
        work = t[:, :3]  # the worker waves
        alive = work[..., 7] > 0
        if not alive.any():
            return
        w = work[alive]
        parts = ", ".join(f"{name} {w[:, i].mean() / 100:.1f}" for i, name in enumerate(_SP_PROF_PHASES))
        print(f"[step-push prof] rank={os.environ.get('RANK')} worker waves={int(alive.sum())} "
              f"steps/wave {w[:, 7].mean():.1f}; mean us per wave: {parts}; "
              f"total {w[:, :5].sum(1).mean() / 100:.1f} (max {w[:, :5].sum(1).max() / 100:.1f})", flush=True)

    atexit.register(report)
    return buf


def build_dedup_step_push_module(
    *,
    arena_handle,
    combine_input_offset,
    slot_stride_bytes,
    topk,
    max_tok,
    hidden,
    world_size,
    block_num,
):
    import mori.cco.device.flydsl as _cco
    from .tdm_gather_shim import tensor_load_gather

    assert SP_NIN == SP_NOUT == 2  # tiles alternate by step parity
    assert SP_SENDER in (0, 1) and 1 <= SW_LAG < SW_NOUT
    waves = 4
    workers = waves - SP_SENDER
    nouts = SW_NOUT if SP_SENDER else SP_NOUT  # output tiles per worker
    lanes = waves * WAVE
    tmax = 8  # slots per step: the rows of one gather store
    fixed = SP_FIXED
    resv = tmax * fixed + SP_OV if fixed else SP_RESV
    ov0 = tmax * fixed  # first overflow row
    assert resv <= WAVE and resv >= topk and (not fixed or topk - fixed <= SP_OV)
    row_e = 2 * RP_CHUNK  # 2 N tiles, 1 KB
    row_b = row_e * 2
    assert hidden % row_e == 0 and row_e // 16 == WAVE
    iters = hidden // row_e
    gathers = [(r0, min(r0 + 8, resv)) for r0 in range(0, resv, 8)]
    loads_per_step = len(gathers)
    ops_per_step = loads_per_step + 1  # + the step's store
    if SP_WMMA:
        assert fixed and resv == 32 and tmax == 8 and SP_SENDER
    in_pad = 8 if SP_WMMA else 0  # elements after every input row
    in_pitch = (row_e + in_pad) * 2
    in_b = resv * in_pitch
    out_b = tmax * row_b
    wave_b = SP_NIN * in_b + nouts * out_b
    rrows = fixed if fixed else SP_RROWS
    max_recv = max_tok * world_size
    tb_max = (max_recv + block_num - 1) // block_num
    tw_max = (tb_max + workers - 1) // workers
    # Every step holds at least this many slots, which bounds the step count.
    per_step_min = max(1, SP_OV // (topk - fixed)) if fixed else resv // topk
    maxs = (tw_max + per_step_min - 1) // per_step_min
    # Per slot: its local rows first then out_rows, count, dest, a scratch word.
    tm_w = topk + 3
    t_cnt, t_dst, t_junk = topk, topk + 1, topk + 2
    # Per step: slot count, packed rows, then per slot offset, count, dest.
    sm_w = 64
    s_n, s_pack = 0, 1
    s_off = s_pack + resv
    s_cnt = s_off + tmax
    s_dst = s_cnt + tmax
    s_maxc = sm_w - 2  # the step's largest count
    s_junk = sm_w - 1
    assert s_dst + tmax <= s_maxc
    tm_b = ((tb_max * tm_w * 4 + 511) // 512) * 512
    sm_b = workers * maxs * sm_w * 4
    # Per worker ready[nouts], free[nouts]; then each wave's step count.
    f_ready, f_free, f_nsw = 0, nouts, 2 * nouts
    fl_w = 2 * nouts
    flags_b = ((workers * fl_w + waves) * 4 + 15) // 16 * 16
    knobs = KNOBS_TAG
    prof_addr = _sp_prof_buffer(block_num * waves * 8).data_ptr() if SP_PROF else 0

    @flyc.kernel(
        name=f"ep_dedup_spushb_{knobs}_i{SP_NIN}o{nouts}r{rrows}v{resv}"
        f"s{SP_SENDER}g{SW_LAG}f{fixed}t{SP_RTOK}h{SP_BF16ACC}w{SP_LSTRIDE}p{SP_PROF}"
        f"b{SP_SUMBATCH}m{SP_WMMA}x{SP_WMMA_BATCH}"
        f"l{int(PP_NO_LOAD)}r{int(PP_NO_REDUCE)}_b{block_num}",
        known_block_size=[lanes, 1, 1],
    )
    def ep_dedup_spush(
        addr_out: fx.Int64,
        out_rows: fx.Int32,
        addr_rows: fx.Int64,
        addr_nvt: fx.Int64,
        addr_tis: fx.Int64,
    ):
        tid = fx.Int32(fx.thread_idx.x)
        bid = fx.block_idx.x
        wave = fx.Int32(rocdl.readfirstlane(T.i32, tid // fx.Int32(WAVE)))
        lane = tid % fx.Int32(WAVE)
        lds_load_b32, lds_store_b32 = make_lds_copy_ops(32)
        lds_load_b128, lds_store_b128 = make_lds_copy_ops(128)
        smem = fx.SharedAllocator(static=False)
        tiles_base = smem.allocate(workers * wave_b)._ptr
        zero_base = smem.allocate(row_b)._ptr
        tm_base = smem.allocate(tm_b)._ptr
        sm_base = smem.allocate(sm_b)._ptr
        flag_base = smem.allocate(flags_b)._ptr
        tm_idx = fx.Int32(fx.ptrtoint(tm_base))
        sm_idx = fx.Int32(fx.ptrtoint(sm_base))
        zero_idx = fx.Int32(fx.ptrtoint(zero_base))
        flag_idx = fx.Int32(fx.ptrtoint(flag_base))
        is_worker = wave < fx.Int32(workers)
        # The sender owns no tiles; clamp so its (unused) addresses stay in range.
        wk = is_worker.select(wave, fx.Int32(workers - 1))
        my_b = wk * fx.Int32(wave_b)  # this wave's tiles, bytes from tiles_base
        in_idx = fx.Int32(fx.ptrtoint(tiles_base)) + my_b
        out_idx = in_idx + fx.Int32(SP_NIN * in_b)

        win = _cco.Window(fx.Int64(arena_handle))
        lsa0 = fx.Int64(win.lsa_ptr(fx.Int32(0), 0))
        peer_stride = fx.Int64(win.lsa_ptr(fx.Int32(1), 0)) - lsa0
        k_slots = fx.Int32(peer_stride // fx.Int64(slot_stride_bytes))
        oob = k_slots * fx.Int32(world_size)
        rows_rsrc = create_buffer_resource_from_addr(addr_rows.ir_value())
        tis_rsrc = create_buffer_resource_from_addr(addr_tis.ir_value())
        nvt = fx.Int32(buffer_load(create_buffer_resource_from_addr(addr_nvt.ir_value()),
                                   fx.Int32(0), vec_width=1, dtype=T.i32, is_scalar=True))
        tb = (nvt + fx.Int32(block_num - 1)) // fx.Int32(block_num)
        tok0 = bid * tb
        n_tb = nvt - tok0
        n_tb = (n_tb < fx.Int32(0)).select(fx.Int32(0), n_tb)
        n_tb = (n_tb > tb).select(tb, n_tb)

        def put(base, word, v):
            lds_store_b32(base, word * fx.Int32(4), Vec.from_elements([v], fx.Int32).ir_value())

        def get(base, word):
            v = lds_load_b32(base, word * fx.Int32(4))[0]
            return fx.Int32(rocdl.readfirstlane(T.i32, fx.Int32(v)))

        # -- every VMEM read of the kernel, before its first peer store --
        for x in range(tid, n_tb, lanes):
            tok = tok0 + x
            w0 = x * fx.Int32(tm_w)
            rs = [fx.Int32(buffer_load(rows_rsrc, tok * fx.Int32(topk) + fx.Int32(k),
                                       vec_width=1, dtype=T.i32))
                  for k in range_constexpr(topk)]
            for k in range_constexpr(topk):
                put(tm_idx, w0 + fx.Int32(k), out_rows)
            lead = fx.Int32(topk)
            cnt = fx.Int32(0)
            for k in range_constexpr(topk):
                local = rs[k] >= fx.Int32(0)
                lead = (local & (lead == fx.Int32(topk))).select(fx.Int32(k), lead)
                put(tm_idx, w0 + local.select(cnt, fx.Int32(t_junk)), rs[k])
                cnt = cnt + local.select(fx.Int32(1), fx.Int32(0))
            enc = fx.Int32(buffer_load(tis_rsrc, tok, vec_width=1, dtype=T.i32))
            origin_pe = enc // fx.Int32(max_tok)
            origin_lid = enc - origin_pe * fx.Int32(max_tok)
            dst = (lead < fx.Int32(topk)).select(
                origin_pe * k_slots + origin_lid * fx.Int32(topk) + lead, oob
            )
            put(tm_idx, w0 + fx.Int32(t_dst), dst)
            put(tm_idx, w0 + fx.Int32(t_cnt), cnt)
        for i in range(tid, flags_b // 4, lanes):
            put(flag_idx, i, fx.Int32(0))
        # Empty step slots: out-of-range rows and dests, zero counts.
        for i in range(tid, workers * maxs * sm_w, lanes):
            f = i % fx.Int32(sm_w)
            in_pack = (f >= fx.Int32(s_pack)) & (f < fx.Int32(s_off))
            in_dst = (f >= fx.Int32(s_dst)) & (f < fx.Int32(s_dst + tmax))
            put(sm_idx, i, in_pack.select(out_rows, in_dst.select(oob, fx.Int32(0))))
        for i in range(tid, row_b // 4, lanes):
            put(zero_idx, i, fx.Int32(0))
        rocdl.s_wait_dscnt(0)
        gpu.barrier()

        # -- each wave packs its own slots into steps (every lane alike) --
        tw = (tb + fx.Int32(workers - 1)) // fx.Int32(workers)
        x0 = wave * tw
        n_w = n_tb - x0
        n_w = (n_w < fx.Int32(0)).select(fx.Int32(0), n_w)
        n_w = (n_w > tw).select(tw, n_w)
        n_w = is_worker.select(n_w, fx.Int32(0))
        my_sm = sm_idx + wk * fx.Int32(maxs * sm_w * 4)
        st = fx.Int32(0)
        nt = fx.Int32(0)
        used = fx.Int32(0)
        mc = fx.Int32(0)
        for i in range_constexpr(tw_max):
            live = fx.Int32(i) < n_w
            x = live.select(x0 + fx.Int32(i), fx.Int32(0))
            w0 = x * fx.Int32(tm_w)
            c = get(tm_idx, w0 + fx.Int32(t_cnt))
            # Packed: all c rows take step rows. Fixed: rows past `fixed` do.
            if const_expr(fixed):
                takes = (c > fx.Int32(fixed)).select(c - fx.Int32(fixed), fx.Int32(0))
                cap = SP_OV
            else:
                takes = c
                cap = resv
            full = (nt == fx.Int32(tmax)) | (used + takes > fx.Int32(cap))
            new = live & full & (nt > fx.Int32(0))
            st = new.select(st + fx.Int32(1), st)
            nt = new.select(fx.Int32(0), nt)
            used = new.select(fx.Int32(0), used)
            mc = new.select(fx.Int32(0), mc)
            mc = (live & (c > mc)).select(c, mc)
            sw = my_sm + st * fx.Int32(sm_w * 4)
            junk = fx.Int32(s_junk)
            put(sw, live.select(fx.Int32(s_maxc), junk), mc)
            put(sw, live.select(fx.Int32(s_off) + nt, junk), used)
            put(sw, live.select(fx.Int32(s_cnt) + nt, junk), c)
            put(sw, live.select(fx.Int32(s_dst) + nt, junk), get(tm_idx, w0 + fx.Int32(t_dst)))
            for k in range_constexpr(topk):
                if const_expr(fixed and k < fixed):
                    row_at = fx.Int32(s_pack + k) + nt * fx.Int32(fixed)
                elif const_expr(fixed):
                    row_at = fx.Int32(s_pack + ov0 + k - fixed) + used
                else:
                    row_at = fx.Int32(s_pack + k) + used
                put(sw, (live & (fx.Int32(k) < c)).select(row_at, junk),
                    get(tm_idx, w0 + fx.Int32(k)))
            put(sw, live.select(fx.Int32(s_n), junk), nt + fx.Int32(1))
            used = live.select(used + takes, used)
            nt = live.select(nt + fx.Int32(1), nt)
        n_sw = (n_w > fx.Int32(0)).select(st + fx.Int32(1), fx.Int32(0))
        n_steps = n_sw * fx.Int32(iters)
        put(flag_idx, fx.Int32(workers * fl_w) + wave, n_sw)
        rocdl.s_wait_dscnt(0)
        if const_expr(SP_SENDER):
            # The sender reads every worker's steps.
            gpu.barrier()

        src_view = fx.Tensor(
            fx.make_view(
                fx.inttoptr(_I16_GLOBAL(), addr_out),
                fx.make_layout((out_rows, row_e), (hidden, 1)),
            )
        )
        comb_view = fx.Tensor(
            fx.make_view(
                fx.inttoptr(_I16_GLOBAL(), lsa0 + fx.Int64(combine_input_offset)),
                fx.make_layout((oob, row_e), (slot_stride_bytes // 2, 1)),
            )
        )
        lds_i16 = fx.recast_iter(_I16_SHARED(), tiles_base) + my_b // fx.Int32(2)
        in_view = fx.Tensor(fx.make_view(
            lds_i16, fx.make_layout((SP_NIN * resv, row_e), (row_e, 1))))
        out_view = fx.Tensor(fx.make_view(
            lds_i16 + fx.Int32(SP_NIN * in_b // 2),
            fx.make_layout((nouts * tmax, row_e), (row_e, 1))))
        tiles_view = fx.Tensor(fx.make_view(
            fx.recast_iter(_I16_SHARED(), tiles_base),
            fx.make_layout((workers * wave_b // row_b, row_e), (row_e, 1))))
        flag_addr = fx.Int64(fx.ptrtoint(flag_base))

        def flag(w, f, i):
            return flag_addr + fx.Int64((fx.Int32(w) * fx.Int32(fl_w) + fx.Int32(f) + i) * fx.Int32(4))

        n_sw_safe = (n_sw > fx.Int32(0)).select(n_sw, fx.Int32(1))

        def step_coords(s):
            """Step s: column chunk s // n_sw (the order flags would come in),
            packed step s % n_sw of this wave."""
            ch = s // n_sw_safe
            return s - ch * n_sw_safe, ch * fx.Int32(row_e)

        def pick(v, i):
            return fx.Int32(rocdl.readlane(T.i32, _raw(v), _raw(fx.Int32(i))))

        def lane_words(k, f0, n, base=None):
            """Words f0.. f0+n-1 of step k, word i in lane i; readlane picks them."""
            f = fx.Int32(f0) + (lane < fx.Int32(n)).select(lane, fx.Int32(0))
            return fx.Int32(lds_load_b32(my_sm if base is None else base,
                                         (k * fx.Int32(sm_w) + f) * fx.Int32(4))[0])

        def load_step(s):
            # Past the end it gathers out-of-range rows, keeping the op count.
            live = s < n_steps
            k, e0 = step_coords(live.select(s, fx.Int32(0)))
            ib = s % fx.Int32(SP_NIN)
            packed = lane_words(k, s_pack, resv)
            rows = [live.select(pick(packed, r), out_rows) for r in range_constexpr(resv)]
            for r0, r1 in gathers:
                desc = make_tensor_gather_descriptor(
                    src_view, in_view, rows[r0:r1],
                    row_width=row_e, tensor_dim0=row_e, tensor_dim1=out_rows.ir_value(),
                    stride=hidden, elem_bytes=2, index_size=32,
                    pad_interval=row_e if in_pad else 0, pad_amount=in_pad,
                    lds_byte_offset=fx.index_cast(
                        T.index, ib * fx.Int32(in_b) + fx.Int32(r0 * in_pitch)),
                    global_byte_offset=fx.Int64(e0) * fx.Int64(2),
                )
                if const_expr(not PP_NO_LOAD):
                    tensor_load_gather(desc, cache_policy=LOAD_CPOL)

        accs = [fx.make_rmem_tensor(8, fx.Float32) for _ in range_constexpr(2)]
        # A lane takes 16 B at a 16 B stride (q = 0) and again a half row on
        # (q = 1): every b128 covers 512 contiguous bytes, free of bank conflicts.
        lane_b = lane * fx.Int32(SP_LSTRIDE)

        def add_row(row):
            for q in range_constexpr(2):
                v = Vec(lds_load_b128(row, lane_b + fx.Int32(q * SP_QSTEP)))
                accs[q].store(accs[q].load() + v.bitcast(fx.BFloat16).to(fx.Float32))

        def add_rows_from(base, off, c, r):
            # Nested, so a slot with c <= r tests once.
            if const_expr(r < topk):
                if c > fx.Int32(r):
                    add_row(base + (off + fx.Int32(r)) * fx.Int32(row_b))
                    add_rows_from(base, off, c, r + 1)

        def reduce_step(s, ob):
            k, _ = step_coords(s)
            ib = s % fx.Int32(SP_NIN)
            oc = lane_words(k, s_off, 2 * tmax)  # offsets, then counts
            mxc = get(my_sm, k * fx.Int32(sm_w) + fx.Int32(s_maxc))
            base = in_idx + ib * fx.Int32(in_b)
            for half in range_constexpr(tmax // SP_SUMBATCH):
                ts = [half * SP_SUMBATCH + i for i in range_constexpr(SP_SUMBATCH)
                      if half * SP_SUMBATCH + i < SP_RTOK]
                if const_expr(not ts):
                    continue
                offs = {t: pick(oc, t) for t in ts}
                cs = {t: pick(oc, tmax + t) for t in ts}
                if const_expr(fixed):
                    # Fixed rows, zero-filled past the count by the gather.
                    addrs = {(t, r): base + fx.Int32((t * fixed + r) * row_b)
                             for t in ts for r in range_constexpr(rrows)}
                    # add_rows_from's row off + r is overflow row off - fixed + r.
                    offs = {t: offs[t] + fx.Int32(ov0 - fixed) for t in ts}
                else:
                    # A row past the slot's count reads the zero row: no branch.
                    addrs = {
                        (t, r): (fx.Int32(r) < cs[t]).select(
                            base + (offs[t] + fx.Int32(r)) * fx.Int32(row_b), zero_idx)
                        for t in ts for r in range_constexpr(rrows)
                    }
                raws = {(t, r, q): lds_load_b128(addrs[(t, r)], lane_b + fx.Int32(q * SP_QSTEP))
                        for t in ts for r in range_constexpr(rrows) for q in range_constexpr(2)}
                for t in ts:
                    for q in range_constexpr(2):
                        if const_expr(SP_BF16ACC):
                            acc16 = Vec(raws[(t, 0, q)]).bitcast(fx.BFloat16)
                            for r in range_constexpr(1, rrows):
                                acc16 = acc16 + Vec(raws[(t, r, q)]).bitcast(fx.BFloat16)
                            acc = acc16.to(fx.Float32)
                        else:
                            acc = Vec(raws[(t, 0, q)]).bitcast(fx.BFloat16).to(fx.Float32)
                            for r in range_constexpr(1, rrows):
                                acc = acc + Vec(raws[(t, r, q)]).bitcast(fx.BFloat16).to(fx.Float32)
                        accs[q].store(acc)
                    if const_expr(rrows < topk):
                        if mxc > fx.Int32(rrows):
                            add_rows_from(base, offs[t], cs[t], rrows)
                    o = out_idx + (ob * fx.Int32(tmax) + fx.Int32(t)) * fx.Int32(row_b)
                    for q in range_constexpr(2):
                        lds_store_b128(o, lane_b + fx.Int32(q * SP_QSTEP),
                                       Vec(accs[q].load()).to(fx.BFloat16).bitcast(fx.Int32).ir_value())

        if const_expr(SP_WMMA):
            wmma_atom = fx.make_mma_atom(fx.rocdl.WMMA(16, 16, 32, fx.BFloat16, fx.Float32))
            lane16 = lane % fx.Int32(16)
            kgrp = lane // fx.Int32(16)
            tr_ptr_ty = fx.PointerType.get(
                elem_ty=fx.BFloat16.ir_type, address_space=fx.AddressSpace.Shared, alignment=2)

        def reduce_wmma(s, ob):
            """out[slot m] = sum over the step's rows k of S[m][k] * X[k], 16 columns
            per WMMA. Lane (m = lane16, kgrp) holds S[m][kgrp*8 + j] and
            S[m][16 + kgrp*8 + j]; the result lane holds slot m's columns
            kgrp*8 .. +7 of the 16."""
            k, _ = step_coords(s)
            ib = s % fx.Int32(SP_NIN)
            real = lane16 < fx.Int32(tmax)
            m = real.select(lane16, fx.Int32(0))
            w_off = (k * fx.Int32(sm_w) + m) * fx.Int32(4)
            off_m = fx.Int32(lds_load_b32(my_sm, w_off + fx.Int32(s_off * 4))[0])
            c_m = fx.Int32(lds_load_b32(my_sm, w_off + fx.Int32(s_cnt * 4))[0])
            ov_m = (c_m > fx.Int32(fixed)).select(c_m - fixed, fx.Int32(0))
            # A row is slot m's: one of its fixed rows, or in its overflow run.
            words = []
            for half in range_constexpr(2):
                for jp in range_constexpr(4):
                    pair = []
                    for jj in range_constexpr(2):
                        kk = kgrp * fx.Int32(8) + fx.Int32(half * 16 + jp * 2 + jj)
                        mine = (kk < fx.Int32(ov0)) & (kk // fx.Int32(fixed) == m)
                        ovk = kk - fx.Int32(ov0)
                        mine = mine | ((kk >= fx.Int32(ov0)) & (ovk >= off_m) & (ovk < off_m + ov_m))
                        pair.append((real & mine).select(fx.Int32(0x3F80), fx.Int32(0)))
                    words.append(pair[0] | (pair[1] << fx.Int32(16)))
            a_t = fx.make_rmem_tensor(16, fx.BFloat16)
            a_t.store(Vec.from_elements(words, fx.Int32).bitcast(fx.BFloat16))
            tile = in_idx + ib * fx.Int32(in_b)
            row_in = kgrp * fx.Int32(8) + lane16 % fx.Int32(8)
            col_in = (lane16 // fx.Int32(8)) * fx.Int32(8)
            o_row = out_idx + (ob * fx.Int32(tmax) + m) * fx.Int32(row_b) + kgrp * fx.Int32(16)
            in_lane = tile + row_in * fx.Int32(in_pitch) + col_in * fx.Int32(2)
            nbs = row_e // 16
            for g0 in range_constexpr(0, nbs, SP_WMMA_BATCH):
                group = list(range(g0, min(g0 + SP_WMMA_BATCH, nbs)))
                # Every block's operand loads first, then the WMMAs, then stores.
                frags = {}
                for nb in group:
                    halves = []
                    for kh in range_constexpr(2):
                        addr = in_lane + fx.Int32(kh * 16 * in_pitch + nb * 32)
                        ptr = fx.to_llvm_ptr(fx.inttoptr(tr_ptr_ty, addr))
                        halves.append(Vec(rocdl.ds_load_tr16_b128(T.vec(8, T.bf16), ptr)))
                    frags[nb] = halves
                accs_w = {}
                for nb in group:
                    b_t = fx.make_rmem_tensor(16, fx.BFloat16)
                    b_t.store(frags[nb][0].shuffle(frags[nb][1], list(range(16))))
                    acc_t = fx.make_rmem_tensor(8, fx.Float32)
                    acc_t.store(fx.constant_vector(0.0, T.vec(8, T.f32)))
                    fx.gemm(wmma_atom, acc_t, b_t, a_t, acc_t)
                    accs_w[nb] = acc_t
                if real:
                    for nb in group:
                        lds_store_b128(o_row, fx.Int32(nb * 32),
                                       Vec(accs_w[nb].load()).to(fx.BFloat16).bitcast(fx.Int32).ir_value())

        def store_step(s):
            k, e0 = step_coords(s)
            ob = s % fx.Int32(SP_NOUT)
            if const_expr(STORE_LOCAL):
                # Timing only: back into each slot's first staged row.
                packed = lane_words(k, s_pack, resv)
                oc = lane_words(k, s_off, 2 * tmax)
                idx = [(pick(oc, tmax + t) > fx.Int32(0)).select(
                    pick(packed, t * fixed) if fixed else
                    fx.Int32(rocdl.readlane(T.i32, _raw(packed), _raw(pick(oc, t)))), out_rows)
                    for t in range_constexpr(tmax)]
            else:
                dsts = lane_words(k, s_dst, tmax)
                idx = [pick(dsts, t) for t in range_constexpr(tmax)]
            desc = make_tensor_gather_descriptor(
                src_view if STORE_LOCAL else comb_view, out_view, idx,
                row_width=row_e, tensor_dim0=row_e,
                tensor_dim1=(out_rows if STORE_LOCAL else oob).ir_value(),
                stride=hidden if STORE_LOCAL else slot_stride_bytes // 2,
                elem_bytes=2, index_size=32,
                lds_byte_offset=fx.index_cast(T.index, ob * fx.Int32(out_b)),
                global_byte_offset=fx.Int64(e0) * fx.Int64(2),
            )
            if const_expr(not NO_STORE):
                tensor_store_gather(desc)

        def run_step(s, after):
            """``after``: TDM ops this wave issued after step s's loads."""
            load_step(s + fx.Int32(1))
            tdm_ops.tensor_wait(after)
            if const_expr(not PP_NO_REDUCE):
                reduce_step(s, s % fx.Int32(nouts))
            # The sums are in LDS before TDM reads them, and this step's input
            # reads are done before a later gather overwrites the tile.
            rocdl.s_wait_dscnt(0)
            store_step(s)

        def send(w, n_sw_w, t, ob, live):
            """Worker w's step t (of its n_sw_w per chunk), from its output tile ob."""
            nsafe = (n_sw_w > fx.Int32(0)).select(n_sw_w, fx.Int32(1))
            ch = t // nsafe
            k = t - ch * nsafe
            e0 = ch * fx.Int32(row_e)
            wsm = sm_idx + fx.Int32(w * maxs * sm_w * 4)
            if const_expr(STORE_LOCAL):
                packed = lane_words(k, s_pack, resv, wsm)
                oc = lane_words(k, s_off, 2 * tmax, wsm)
                idx = [(live & (pick(oc, tmax + j) > fx.Int32(0))).select(
                    pick(packed, j * fixed) if fixed else
                    fx.Int32(rocdl.readlane(T.i32, _raw(packed), _raw(pick(oc, j)))), out_rows)
                    for j in range_constexpr(tmax)]
            else:
                dsts = lane_words(k, s_dst, tmax, wsm)
                idx = [live.select(pick(dsts, j), oob) for j in range_constexpr(tmax)]
            desc = make_tensor_gather_descriptor(
                src_view if STORE_LOCAL else comb_view, tiles_view, idx,
                row_width=row_e, tensor_dim0=row_e,
                tensor_dim1=(out_rows if STORE_LOCAL else oob).ir_value(),
                stride=hidden if STORE_LOCAL else slot_stride_bytes // 2,
                elem_bytes=2, index_size=32,
                lds_byte_offset=fx.index_cast(
                    T.index, fx.Int32(w * wave_b + SP_NIN * in_b) + ob * fx.Int32(out_b)),
                global_byte_offset=fx.Int64(e0) * fx.Int64(2),
            )
            if const_expr(not NO_STORE):
                tensor_store_gather(desc)

        if const_expr(SP_SENDER):
            if is_worker:
                if const_expr(SP_PROF):
                    prof = fx.make_rmem_tensor(8, fx.Int32)
                    prof.store(fx.constant_vector(0, T.vec(8, T.i32)))

                def tick(prev, i):
                    if const_expr(not SP_PROF):
                        return prev
                    now = _realtime_lo()
                    v = Vec(prof.load())
                    prof.store(Vec.from_elements(
                        [v[j] + (now - prev if j == i else fx.Int32(0)) for j in range(8)],
                        fx.Int32).ir_value())
                    return now

                load_step(fx.Int32(0))
                for s in range(n_steps):
                    t_ = _realtime_lo() if SP_PROF else None
                    load_step(s + fx.Int32(1))
                    t_ = tick(t_, 0)
                    # Only loads in this wave's tensorcnt: step s's, then s+1's.
                    tdm_ops.tensor_wait(loads_per_step)
                    t_ = tick(t_, 1)
                    ob = s % fx.Int32(nouts)
                    # The tile's previous use, nouts steps back, has been sent.
                    _lds_poll_ge(flag(wave, f_free, ob), s // fx.Int32(nouts))
                    t_ = tick(t_, 2)
                    if const_expr(PP_NO_REDUCE):
                        pass
                    elif const_expr(SP_WMMA):
                        reduce_wmma(s, ob)
                    else:
                        reduce_step(s, ob)
                    t_ = tick(t_, 3)
                    rocdl.s_wait_dscnt(0)
                    if lane == fx.Int32(0):
                        store_i32_lds(flag(wave, f_ready, ob), s + fx.Int32(1))
                    t_ = tick(t_, 4)
                tdm_ops.tensor_wait(0)
                if const_expr(SP_PROF):
                    v = Vec(prof.load())
                    if lane < fx.Int32(8):
                        pv = (lane == fx.Int32(7)).select(n_steps, fx.Int32(0))
                        for j in range_constexpr(5):
                            pv = (lane == fx.Int32(j)).select(v[j], pv)
                        buffer_store(pv.ir_value(),
                                     create_buffer_resource_from_addr(fx.Int64(prof_addr).ir_value()),
                                     (bid * fx.Int32(waves) + wave) * fx.Int32(8) + lane)
            else:
                nsw = [get(flag_idx, fx.Int32(workers * fl_w + w)) for w in range_constexpr(workers)]
                nst = [x * fx.Int32(iters) for x in nsw]
                rounds = nst[0]
                for w in range_constexpr(1, workers):
                    rounds = (nst[w] > rounds).select(nst[w], rounds)
                # Round t sends every worker's step t (a dead one sends no rows,
                # keeping the count); stores retire in order, so after round t
                # the ones SW_LAG rounds back have, and their tiles go free.
                for t in range(rounds):
                    t_ob = t % fx.Int32(nouts)
                    for w in range_constexpr(workers):
                        if t < nst[w]:
                            _lds_poll_ge(flag(w, f_ready, t_ob), t + fx.Int32(1))
                        send(w, nsw[w], t, t_ob, t < nst[w])
                    if t >= fx.Int32(SW_LAG):
                        tdm_ops.tensor_wait(workers * SW_LAG)
                        t_done = t - fx.Int32(SW_LAG)
                        for w in range_constexpr(workers):
                            if (lane == fx.Int32(0)) & (t_done < nst[w]):
                                store_i32_lds(flag(w, f_free, t_done % fx.Int32(nouts)),
                                              t_done // fx.Int32(nouts) + fx.Int32(1))
                tdm_ops.tensor_wait(0)
        else:
            load_step(fx.Int32(0))
            # In issue order, step s's loads are followed by the previous step's
            # store and the next step's loads; step 0 has no previous store.
            if fx.Int32(0) < n_steps:
                run_step(fx.Int32(0), loads_per_step)
            for s in range(fx.Int32(1), n_steps):
                run_step(s, ops_per_step)
            tdm_ops.tensor_wait(0)

    @flyc.jit
    def launch(
        addr_out: fx.Int64,
        out_rows: fx.Int32,
        addr_rows: fx.Int64,
        addr_nvt: fx.Int64,
        addr_tis: fx.Int64,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
    ):
        ep_dedup_spush(addr_out, out_rows, addr_rows, addr_nvt, addr_tis).launch(
            grid=(block_num, 1, 1), block=[lanes, 1, 1], stream=stream
        )

    return launch


ORDER_THREADS = 1024


def build_dedup_order_module(*, topk, tile_m):
    """Sort this layer's recv slots by the last gemm2 M tile they wait for.

    One workgroup: per-slot key = max(row // tile_m) over its local rows, an LDS
    histogram, a block scan, then a scatter into ORDER/SKEY. It also zeroes the
    M tile counters the coming gemm2 counts into.
    """
    bins_per_thread = MAX_MTILES // ORDER_THREADS
    lds_words = MAX_MTILES + 2 * ORDER_THREADS

    @flyc.kernel(name=f"ep_dedup_order_k{topk}_m{tile_m}", known_block_size=[ORDER_THREADS, 1, 1])
    def ep_dedup_order(addr_rows: fx.Int64, addr_nvt: fx.Int64, addr_state: fx.Int64):
        tid = fx.Int32(fx.thread_idx.x)
        base = fx.SharedAllocator(static=False).allocate(lds_words * 4)._ptr
        lds_addr = fx.Int64(fx.ptrtoint(base))
        hist = lambda i: lds_addr + fx.Int64(i) * fx.Int64(4)  # noqa: E731
        scan = lambda b, i: lds_addr + fx.Int64(  # noqa: E731
            fx.Int32(MAX_MTILES + b * ORDER_THREADS) + i
        ) * fx.Int64(4)
        rows_rsrc = create_buffer_resource_from_addr(addr_rows.ir_value())
        nvt_rsrc = create_buffer_resource_from_addr(addr_nvt.ir_value())
        state_rsrc = create_buffer_resource_from_addr(addr_state.ir_value())
        nvt = fx.Int32(buffer_load(nvt_rsrc, fx.Int32(0), vec_width=1, dtype=T.i32, is_scalar=True))

        for i in range(tid, MAX_MTILES, ORDER_THREADS):
            store_i32_lds(hist(i), fx.Int32(0))
            buffer_store(fx.Int32(0).ir_value(), state_rsrc, fx.Int32(MTILE_OFF) + i)
        gpu.barrier()
        for t in range(tid, nvt, ORDER_THREADS):
            key = fx.Int32(0)
            for k in range_constexpr(topk):
                row = fx.Int32(
                    buffer_load(rows_rsrc, t * fx.Int32(topk) + fx.Int32(k), vec_width=1,
                                dtype=T.i32)
                )
                mt = (row >= fx.Int32(0)).select(row // fx.Int32(tile_m), fx.Int32(0))
                key = (mt > key).select(mt, key)
            buffer_store(key.ir_value(), state_rsrc, fx.Int32(KEY_OFF) + t)
            atomic_add_lds(hist(key), fx.Int32(1))
        gpu.barrier()
        # Exclusive scan: each thread owns bins_per_thread consecutive bins.
        counts = [fx.Int32(load_i32_lds(hist(tid * fx.Int32(bins_per_thread) + fx.Int32(j))))
                  for j in range_constexpr(bins_per_thread)]
        total = fx.Int32(0)
        for j in range_constexpr(bins_per_thread):
            total = total + counts[j]
        store_i32_lds(scan(0, tid), total)
        gpu.barrier()
        src = 0
        for off in range_constexpr(10):
            step = 1 << off
            v = fx.Int32(load_i32_lds(scan(src, tid)))
            if tid >= fx.Int32(step):
                v = v + fx.Int32(load_i32_lds(scan(src, tid - fx.Int32(step))))
            store_i32_lds(scan(1 - src, tid), v)
            gpu.barrier()
            src = 1 - src
        run = fx.Int32(load_i32_lds(scan(src, tid))) - total
        for j in range_constexpr(bins_per_thread):
            store_i32_lds(hist(tid * fx.Int32(bins_per_thread) + fx.Int32(j)), run)
            run = run + counts[j]
        gpu.barrier()
        for t in range(tid, nvt, ORDER_THREADS):
            key = fx.Int32(buffer_load(state_rsrc, fx.Int32(KEY_OFF) + t, vec_width=1, dtype=T.i32))
            pos = fx.Int32(atomic_add_lds(hist(key), fx.Int32(1)))
            buffer_store(fx.Int32(t).ir_value(), state_rsrc, fx.Int32(ORDER_OFF) + pos)
            buffer_store(key.ir_value(), state_rsrc, fx.Int32(SKEY_OFF) + pos)

    @flyc.jit
    def launch(addr_rows: fx.Int64, addr_nvt: fx.Int64, addr_state: fx.Int64,
               stream: fx.Stream = fx.Stream(None)):  # noqa: B008
        ep_dedup_order(addr_rows, addr_nvt, addr_state).launch(
            grid=(1, 1, 1), block=[ORDER_THREADS, 1, 1], stream=stream
        )

    return launch


def build_dedup_push_module(
    *,
    arena_handle,
    combine_input_offset,
    slot_stride_bytes,
    topk,
    max_tok,
    hidden,
    block_num=256,
):
    """Standalone push over every recv slot, launched after gemm2 has finished."""
    import mori.cco.device.flydsl as _cco

    if hidden % WAVE_CHUNK:
        raise ValueError(f"hidden_dim must be a multiple of {WAVE_CHUNK}, got {hidden}")
    waves = 4

    knobs = KNOBS_TAG

    @flyc.kernel(name=f"ep_dedup_push_{knobs}_b{block_num}", known_block_size=[waves * WAVE, 1, 1])
    def ep_dedup_push(
        addr_out: fx.Int64,
        out_bytes: fx.Int64,
        addr_rows: fx.Int64,
        addr_nvt: fx.Int64,
        addr_tis: fx.Int64,
    ):
        tid = fx.thread_idx.x
        wave = rocdl.readfirstlane(T.i32, tid // WAVE)
        lane = tid % fx.Int32(WAVE)
        base = fx.SharedAllocator(static=False).allocate(push_lds_bytes(hidden, waves))._ptr
        lds_ptr = fx.recast_iter(_I16_SHARED(), base) + wave * fx.Int32(NBUF * hidden)
        win = _cco.Window(fx.Int64(arena_handle))
        lsa0 = fx.Int64(win.lsa_ptr(fx.Int32(0), 0))
        peer_stride = fx.Int64(win.lsa_ptr(fx.Int32(1), 0)) - lsa0
        out_rsrc = create_buffer_resource_from_addr(addr_out.ir_value(), num_records_bytes=out_bytes)
        out_base = addr_out
        rows_rsrc = create_buffer_resource_from_addr(addr_rows.ir_value())
        tis_rsrc = create_buffer_resource_from_addr(addr_tis.ir_value())
        nvt_rsrc = create_buffer_resource_from_addr(addr_nvt.ir_value())
        nvt = fx.Int32(buffer_load(nvt_rsrc, fx.Int32(0), vec_width=1, dtype=T.i32, is_scalar=True))
        gw = fx.block_idx.x * fx.Int32(waves) + wave
        for t in range(gw, nvt, block_num * waves):
            emit_push_token(
                fx.Int32(t),
                lane,
                lds_ptr,
                ((t - gw) // fx.Int32(block_num * waves)) % fx.Int32(NBUF),
                out_rsrc=out_rsrc,
                out_base=out_base,
                rows_rsrc=rows_rsrc,
                tis_rsrc=tis_rsrc,
                lsa0=lsa0,
                peer_stride=peer_stride,
                combine_input_offset=combine_input_offset,
                slot_stride_bytes=slot_stride_bytes,
                topk=topk,
                max_tok=max_tok,
                hidden=hidden,
            )
        tdm_ops.tensor_wait(0)

    @flyc.jit
    def launch(
        addr_out: fx.Int64,
        out_bytes: fx.Int64,
        addr_rows: fx.Int64,
        addr_nvt: fx.Int64,
        addr_tis: fx.Int64,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
    ):
        ep_dedup_push(addr_out, out_bytes, addr_rows, addr_nvt, addr_tis).launch(
            grid=(block_num, 1, 1), block=[waves * WAVE, 1, 1], stream=stream
        )

    return launch
