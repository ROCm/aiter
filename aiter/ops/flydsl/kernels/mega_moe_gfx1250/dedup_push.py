# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Dedup'd combine push for the gemm2 EP path (AITER_EP_DEDUP).

gemm2 stages every route-weighted row in this rank's ``grouped_out``. For a recv
slot ``t`` its ``c`` local rows are ``topids_to_rows[t*topk + k] >= 0``; their sum
goes out as ONE row, to the origin's ``comb_inp[origin_lid*topk + lead_k]`` where
``lead_k`` is the first local ``k``. The receiving reduce sums only those lead
slots, so a (token, rank) pair costs one row on the wire instead of ``c``.

The push runs on PUSH_BLOCKS workgroups leading the gemm2 grid
(``mxfp4_preshuffle_gfx1250_tdm.py``), one column chunk at a time as gemm2
completes the columns.
"""

import functools
import hashlib
import os
from types import SimpleNamespace

import torch
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl, tdm_ops
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec
from flydsl.expr.arith import _to_raw as _raw

from ..buffer_ops import buffer_load, create_buffer_resource_from_addr
from ..gemm_common_gfx1250 import make_lds_copy_ops
from .tdm_gather_shim import (
    make_tensor_gather_descriptor,
    tensor_load_gather,
    tensor_store_gather,
)
from ..communication_ops_utils import (
    atomic_add_agent,
    atomic_add_lds,
    load_i32_global_agent,
    store_i32_global_agent_release,
    store_i32_lds,
    traced,
)

WAVE = 32
# cpol SCOPE_DEV: the rows were staged by other XCDs, so miss this XCD's L2.
SCOPE_DEV = 16
# The flydsl cache key of the gemm2 that runs the push covers neither this
# file's constants nor its code, so gemm2 takes this hash as a Constexpr.
with open(__file__, "rb") as _f:
    SRC_TAG = int(hashlib.sha256(_f.read()).hexdigest()[:7], 16)


def ep_dedup_enabled() -> bool:
    return os.environ.get("AITER_EP_DEDUP", "0") in ("1", "true", "True")


def _I16_GLOBAL():
    return fx.PointerType.get(T.i16, address_space=fx.AddressSpace.Global, alignment=16)


def _I16_SHARED():
    return fx.PointerType.get(T.i16, address_space=fx.AddressSpace.Shared, alignment=16)


def _lds_poll_ge(addr_i64, target):
    """Spin until the LDS word reaches ``target``; an atomic add of 0 is a load
    the compiler cannot hoist out of the loop."""
    v = fx.Int32(atomic_add_lds(addr_i64, fx.Int32(0)))
    while v < target:
        rocdl.s_sleep(1)
        v = fx.Int32(atomic_add_lds(addr_i64, fx.Int32(0)))


_lds_poll_ge = traced(_lds_poll_ge)


# ---------------------------------------------------------------------------
# In-grid column counters, one i32 buffer per device: [n] counts the staged
# tiles of gemm2 N tile n, [DONE_WORD] the pusher blocks that are done. The last
# pusher block zeroes them all for the next layer.
# ---------------------------------------------------------------------------
COL_WORDS = 64
DONE_WORD = COL_WORDS - 1
# A wait gives up after this many polls: a broken invariant then shows up as
# wrong output instead of a hung kernel.
POLL_LIMIT = 1 << 22


@functools.cache
def dedup_col_state(device: str) -> torch.Tensor:
    return torch.zeros(COL_WORDS, dtype=torch.int32, device=device)


def emit_col_tile_done(n_tile, *, state_addr):
    """One gemm2 tile of N tile ``n_tile`` is staged; call once per workgroup,
    after every wave's stores of the tile have completed."""
    atomic_add_agent(state_addr + fx.Int64(n_tile) * fx.Int64(4), fx.Int32(1))


def _wait_col(addr_i64, target):
    v = fx.Int32(load_i32_global_agent(addr_i64))
    n = fx.Int32(0)
    while (v < target) & (n < fx.Int32(POLL_LIMIT)):
        rocdl.s_sleep(2)
        v = fx.Int32(load_i32_global_agent(addr_i64))
        n = n + fx.Int32(1)


_wait_col = traced(_wait_col)


def emit_col_drain(tid, *, state_addr, target, n_cols, block_num):
    """End of every pusher block: wait for all ``n_cols`` columns, then the last
    block to arrive zeroes the counters. Every counter is at ``target`` by then,
    so no gemm2 tile can still add to one."""
    gpu.barrier()
    if tid == fx.Int32(0):
        for c in range_constexpr(n_cols):
            _wait_col(state_addr + fx.Int64(c * 4), target)
        old = fx.Int32(atomic_add_agent(state_addr + fx.Int64(DONE_WORD * 4), fx.Int32(1)))
        if old == fx.Int32(block_num - 1):
            for c in range_constexpr(n_cols):
                store_i32_global_agent_release(state_addr + fx.Int64(c * 4), fx.Int32(0))
            store_i32_global_agent_release(state_addr + fx.Int64(DONE_WORD * 4), fx.Int32(0))


emit_col_drain = traced(emit_col_drain)


# ---------------------------------------------------------------------------
# Step push. The recv slots are split evenly over the blocks and, within a
# block, over waves 0-2 (the workers); wave 3 only sends. A worker packs its
# slots into steps of up to TMAX (the rows of one gather store) and runs every
# step through load -> sum, column chunk by column chunk (a chunk is 2 N tiles,
# 1 KB of a row). A slot's first FIXED rows sit at fixed input tile rows, so
# the sum reads them at constant offsets; rows past FIXED go to the step's
# OV-row overflow area. A step's 32 input rows are 4 TDM gathers.
#
# Workers and sender hand off over LDS flags, per worker output tile:
# ready = steps summed into it, free = its sends retired. A worker's tensorcnt
# then holds only its own loads, the sender's only its stores. The sender sends
# round by round, each worker's step t in round t; stores retire in issue
# order, so after round t those of round t - LAG have, and their tiles go free.
# ---------------------------------------------------------------------------
PUSH_BLOCKS = 64
WORKERS = 3
TMAX = 8  # slots per step
NIN = 2  # input tiles per worker; steps alternate between them
NOUT = 4  # output tiles per worker
LAG = 3  # sender rounds in flight
FIXED = 3
OV = 8
CHUNK_ELEMS = 512  # 2 N tiles, 1 KB


def _step_layout(*, topk, max_tok, world_size, block_num):
    assert 1 <= LAG < NOUT
    L = SimpleNamespace()
    L.resv = TMAX * FIXED + OV  # rows of an input tile
    L.ov0 = TMAX * FIXED  # first overflow row
    assert L.resv <= WAVE and topk - FIXED <= OV
    L.row_b = CHUNK_ELEMS * 2
    L.in_b = L.resv * L.row_b
    L.out_b = TMAX * L.row_b
    L.wave_b = NIN * L.in_b + NOUT * L.out_b
    L.tb_max = (max_tok * world_size + block_num - 1) // block_num
    L.tw_max = (L.tb_max + WORKERS - 1) // WORKERS
    # Every step holds at least this many slots, which bounds the step count.
    per_step_min = max(1, OV // max(1, topk - FIXED))
    L.maxs = (L.tw_max + per_step_min - 1) // per_step_min
    # Per slot: its local rows first then out_rows, count, dest, a scratch word.
    L.tm_w = topk + 3
    # Per step: slot count, packed rows, then per slot offset, count, dest.
    L.sm_w = 64
    L.s_n, L.s_pack = 0, 1
    L.s_off = L.s_pack + L.resv
    L.s_cnt = L.s_off + TMAX
    L.s_dst = L.s_cnt + TMAX
    L.s_maxc = L.sm_w - 2  # the step's largest count
    L.s_junk = L.sm_w - 1
    assert L.s_dst + TMAX <= L.s_maxc
    tiles_b = WORKERS * L.wave_b
    tm_b = ((L.tb_max * L.tm_w * 4 + 511) // 512) * 512
    # A barrier ends the packing before any tile is loaded, so the slot
    # metadata can live in the tiles.
    tm_in_tiles = tm_b <= tiles_b
    L.tm_off = 0 if tm_in_tiles else tiles_b
    L.sm_off = tiles_b + (0 if tm_in_tiles else tm_b)
    sm_b = WORKERS * L.maxs * L.sm_w * 4
    # Per worker ready[NOUT], free[NOUT]; then each wave's step count.
    L.f_ready, L.f_free = 0, NOUT
    L.fl_w = 2 * NOUT
    L.f_nsw = WORKERS * L.fl_w
    L.flag_off = L.sm_off + sm_b
    L.lds_b = L.flag_off + ((L.f_nsw + WORKERS + 1) * 4 + 15) // 16 * 16
    return L


def step_push_lds_bytes(*, topk, max_tok, world_size, block_num=PUSH_BLOCKS):
    return _step_layout(topk=topk, max_tok=max_tok, world_size=world_size,
                        block_num=block_num).lds_b


def emit_step_push(
    bid,
    tid,
    lds_base,
    *,
    addr_out,
    out_rows,
    addr_rows,
    addr_nvt,
    addr_tis,
    lsa0,
    peer_stride,
    combine_input_offset,
    slot_stride_bytes,
    topk,
    max_tok,
    hidden,
    world_size,
    col_state,
    col_target,
    tile_n,
    block_num=PUSH_BLOCKS,
):
    """One push block, ``bid`` of ``block_num``: 4 waves, ``lds_base`` holding
    step_push_lds_bytes(). Before its first load of a column chunk, a worker
    waits for both of the chunk's N tile counters in ``col_state`` to reach
    ``col_target``."""
    L = _step_layout(topk=topk, max_tok=max_tok, world_size=world_size, block_num=block_num)
    waves = WORKERS + 1
    lanes = waves * WAVE
    row_e = CHUNK_ELEMS
    row_b = L.row_b
    resv, ov0, in_b, out_b, wave_b = L.resv, L.ov0, L.in_b, L.out_b, L.wave_b
    maxs, sm_w, tm_w = L.maxs, L.sm_w, L.tm_w
    t_cnt, t_dst, t_junk = topk, topk + 1, topk + 2
    assert hidden % row_e == 0 and row_e == 2 * tile_n
    iters = hidden // row_e
    gathers = [(r0, min(r0 + 8, resv)) for r0 in range(0, resv, 8)]
    loads_per_step = len(gathers)

    wave = fx.Int32(rocdl.readfirstlane(T.i32, tid // fx.Int32(WAVE)))
    lane = tid % fx.Int32(WAVE)
    lds_load_b32, lds_store_b32 = make_lds_copy_ops(32)
    lds_load_b128, lds_store_b128 = make_lds_copy_ops(128)
    base_idx = fx.Int32(fx.ptrtoint(lds_base))
    tm_idx = base_idx + fx.Int32(L.tm_off)
    sm_idx = base_idx + fx.Int32(L.sm_off)
    flag_idx = base_idx + fx.Int32(L.flag_off)
    is_worker = wave < fx.Int32(WORKERS)
    # The sender owns no tiles; clamp so its (unused) addresses stay in range.
    wk = is_worker.select(wave, fx.Int32(WORKERS - 1))
    my_b = wk * fx.Int32(wave_b)  # this wave's tiles, bytes from lds_base
    in_idx = base_idx + my_b
    out_idx = in_idx + fx.Int32(NIN * in_b)

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

    # -- every VMEM read of the push, before its first peer store --
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
    for i in range(tid, (L.lds_b - L.flag_off) // 4, lanes):
        put(flag_idx, i, fx.Int32(0))
    # Empty step slots: out-of-range rows and dests, zero counts.
    for i in range(tid, WORKERS * maxs * sm_w, lanes):
        f = i % fx.Int32(sm_w)
        in_pack = (f >= fx.Int32(L.s_pack)) & (f < fx.Int32(L.s_off))
        in_dst = (f >= fx.Int32(L.s_dst)) & (f < fx.Int32(L.s_dst + TMAX))
        put(sm_idx, i, in_pack.select(out_rows, in_dst.select(oob, fx.Int32(0))))
    rocdl.s_wait_dscnt(0)
    gpu.barrier()

    # -- each worker packs its own slots into steps (every lane alike) --
    tw = (tb + fx.Int32(WORKERS - 1)) // fx.Int32(WORKERS)
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
    for i in range_constexpr(L.tw_max):
        live = fx.Int32(i) < n_w
        x = live.select(x0 + fx.Int32(i), fx.Int32(0))
        w0 = x * fx.Int32(tm_w)
        c = get(tm_idx, w0 + fx.Int32(t_cnt))
        takes = (c > fx.Int32(FIXED)).select(c - fx.Int32(FIXED), fx.Int32(0))
        full = (nt == fx.Int32(TMAX)) | (used + takes > fx.Int32(OV))
        new = live & full & (nt > fx.Int32(0))
        st = new.select(st + fx.Int32(1), st)
        nt = new.select(fx.Int32(0), nt)
        used = new.select(fx.Int32(0), used)
        mc = new.select(fx.Int32(0), mc)
        mc = (live & (c > mc)).select(c, mc)
        sw = my_sm + st * fx.Int32(sm_w * 4)
        junk = fx.Int32(L.s_junk)
        put(sw, live.select(fx.Int32(L.s_maxc), junk), mc)
        put(sw, live.select(fx.Int32(L.s_off) + nt, junk), used)
        put(sw, live.select(fx.Int32(L.s_cnt) + nt, junk), c)
        put(sw, live.select(fx.Int32(L.s_dst) + nt, junk), get(tm_idx, w0 + fx.Int32(t_dst)))
        for k in range_constexpr(topk):
            if const_expr(k < FIXED):
                row_at = fx.Int32(L.s_pack + k) + nt * fx.Int32(FIXED)
            else:
                row_at = fx.Int32(L.s_pack + ov0 + k - FIXED) + used
            put(sw, (live & (fx.Int32(k) < c)).select(row_at, junk),
                get(tm_idx, w0 + fx.Int32(k)))
        put(sw, live.select(fx.Int32(L.s_n), junk), nt + fx.Int32(1))
        used = live.select(used + takes, used)
        nt = live.select(nt + fx.Int32(1), nt)
    n_sw = (n_w > fx.Int32(0)).select(st + fx.Int32(1), fx.Int32(0))
    n_steps = n_sw * fx.Int32(iters)
    put(flag_idx, fx.Int32(L.f_nsw) + wave, n_sw)
    rocdl.s_wait_dscnt(0)
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
    lds_i16 = fx.recast_iter(_I16_SHARED(), lds_base)
    in_view = fx.Tensor(fx.make_view(
        lds_i16 + my_b // fx.Int32(2), fx.make_layout((NIN * resv, row_e), (row_e, 1))))
    tiles_view = fx.Tensor(fx.make_view(
        lds_i16, fx.make_layout((WORKERS * wave_b // row_b, row_e), (row_e, 1))))
    flag_addr = fx.Int64(flag_idx)

    def flag(w, f, i):
        return flag_addr + fx.Int64((fx.Int32(w) * fx.Int32(L.fl_w) + fx.Int32(f) + i) * fx.Int32(4))

    n_sw_safe = (n_sw > fx.Int32(0)).select(n_sw, fx.Int32(1))

    def step_coords(s):
        """Step s: column chunk s // n_sw, the order gemm2 completes them in,
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
        ib = s % fx.Int32(NIN)
        packed = lane_words(k, L.s_pack, resv)
        rows = [live.select(pick(packed, r), out_rows) for r in range_constexpr(resv)]
        if live & (k == fx.Int32(0)):
            n0 = e0 // fx.Int32(tile_n)
            for j in range_constexpr(2):
                _wait_col(col_state + fx.Int64(n0 + fx.Int32(j)) * fx.Int64(4), col_target)
        for r0, r1 in gathers:
            desc = make_tensor_gather_descriptor(
                src_view, in_view, rows[r0:r1],
                row_width=row_e, tensor_dim0=row_e, tensor_dim1=out_rows.ir_value(),
                stride=hidden, elem_bytes=2, index_size=32,
                lds_byte_offset=fx.index_cast(
                    T.index, ib * fx.Int32(in_b) + fx.Int32(r0 * row_b)),
                global_byte_offset=fx.Int64(e0) * fx.Int64(2),
            )
            tensor_load_gather(desc, cache_policy=SCOPE_DEV)

    accs = [fx.make_rmem_tensor(8, fx.Float32) for _ in range_constexpr(2)]
    # A lane takes 16 B at a 16 B stride and again half a row on: every b128
    # covers 512 contiguous bytes, free of bank conflicts.
    lane_b = lane * fx.Int32(16)
    half_b = row_b // 2

    def add_row(row):
        for q in range_constexpr(2):
            v = Vec(lds_load_b128(row, lane_b + fx.Int32(q * half_b)))
            accs[q].store(accs[q].load() + v.bitcast(fx.BFloat16).to(fx.Float32))

    def add_rows_from(base, off, c, r):
        # Nested, so a slot with c <= r tests once.
        if const_expr(r < topk):
            if c > fx.Int32(r):
                add_row(base + (off + fx.Int32(r)) * fx.Int32(row_b))
                add_rows_from(base, off, c, r + 1)

    def reduce_step(s, ob):
        k, _ = step_coords(s)
        base = in_idx + (s % fx.Int32(NIN)) * fx.Int32(in_b)
        oc = lane_words(k, L.s_off, 2 * TMAX)  # offsets, then counts
        mxc = get(my_sm, k * fx.Int32(sm_w) + fx.Int32(L.s_maxc))
        # add_rows_from's row off + r is overflow row off - FIXED + r.
        offs = [pick(oc, t) + fx.Int32(ov0 - FIXED) for t in range_constexpr(TMAX)]
        cs = [pick(oc, TMAX + t) for t in range_constexpr(TMAX)]
        # Every slot's fixed rows (zero-filled past its count by the gather)
        # are read before the first add.
        raws = {(t, r, q): lds_load_b128(base + fx.Int32((t * FIXED + r) * row_b),
                                         lane_b + fx.Int32(q * half_b))
                for t in range_constexpr(TMAX) for r in range_constexpr(FIXED)
                for q in range_constexpr(2)}
        for t in range_constexpr(TMAX):
            for q in range_constexpr(2):
                # Packed bf16 adds, rounding at every add.
                acc16 = Vec(raws[(t, 0, q)]).bitcast(fx.BFloat16)
                for r in range_constexpr(1, FIXED):
                    acc16 = acc16 + Vec(raws[(t, r, q)]).bitcast(fx.BFloat16)
                accs[q].store(acc16.to(fx.Float32))
            if const_expr(FIXED < topk):
                if mxc > fx.Int32(FIXED):
                    add_rows_from(base, offs[t], cs[t], FIXED)
            o = out_idx + (ob * fx.Int32(TMAX) + fx.Int32(t)) * fx.Int32(row_b)
            for q in range_constexpr(2):
                lds_store_b128(o, lane_b + fx.Int32(q * half_b),
                               Vec(accs[q].load()).to(fx.BFloat16).bitcast(fx.Int32).ir_value())

    def send(w, n_sw_w, t, ob, live):
        """Worker w's step t (of its n_sw_w per chunk), from its output tile ob."""
        nsafe = (n_sw_w > fx.Int32(0)).select(n_sw_w, fx.Int32(1))
        ch = t // nsafe
        k = t - ch * nsafe
        e0 = ch * fx.Int32(row_e)
        dsts = lane_words(k, L.s_dst, TMAX, sm_idx + fx.Int32(w * maxs * sm_w * 4))
        desc = make_tensor_gather_descriptor(
            comb_view, tiles_view,
            [live.select(pick(dsts, j), oob) for j in range_constexpr(TMAX)],
            row_width=row_e, tensor_dim0=row_e, tensor_dim1=oob.ir_value(),
            stride=slot_stride_bytes // 2, elem_bytes=2, index_size=32,
            lds_byte_offset=fx.index_cast(
                T.index, fx.Int32(w * wave_b + NIN * in_b) + ob * fx.Int32(out_b)),
            global_byte_offset=fx.Int64(e0) * fx.Int64(2),
        )
        tensor_store_gather(desc)

    if is_worker:
        load_step(fx.Int32(0))
        for s in range(n_steps):
            load_step(s + fx.Int32(1))
            # Only loads in this wave's tensorcnt: step s's, then s+1's.
            tdm_ops.tensor_wait(loads_per_step)
            ob = s % fx.Int32(NOUT)
            # The tile's previous use, NOUT steps back, has been sent.
            _lds_poll_ge(flag(wave, L.f_free, ob), s // fx.Int32(NOUT))
            reduce_step(s, ob)
            rocdl.s_wait_dscnt(0)
            if lane == fx.Int32(0):
                store_i32_lds(flag(wave, L.f_ready, ob), s + fx.Int32(1))
        tdm_ops.tensor_wait(0)
    else:
        nsw = [get(flag_idx, fx.Int32(L.f_nsw + w)) for w in range_constexpr(WORKERS)]
        nst = [x * fx.Int32(iters) for x in nsw]
        rounds = fx.Int32(0)
        for w in range_constexpr(WORKERS):
            rounds = (nst[w] > rounds).select(nst[w], rounds)
        # A worker out of steps still sends (no rows), keeping WORKERS stores
        # per round.
        for t in range(rounds):
            t_ob = t % fx.Int32(NOUT)
            for w in range_constexpr(WORKERS):
                if t < nst[w]:
                    _lds_poll_ge(flag(w, L.f_ready, t_ob), t + fx.Int32(1))
                send(w, nsw[w], t, t_ob, t < nst[w])
            if t >= fx.Int32(LAG):
                tdm_ops.tensor_wait(WORKERS * LAG)
                t_done = t - fx.Int32(LAG)
                for w in range_constexpr(WORKERS):
                    if (lane == fx.Int32(0)) & (t_done < nst[w]):
                        store_i32_lds(flag(w, L.f_free, t_done % fx.Int32(NOUT)),
                                      t_done // fx.Int32(NOUT) + fx.Int32(1))
        tdm_ops.tensor_wait(0)


emit_step_push = traced(emit_step_push)
