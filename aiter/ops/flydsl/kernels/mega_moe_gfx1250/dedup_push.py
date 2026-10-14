# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Dedup'd combine push for the gemm2 EP path (AITER_EP_DEDUP).

gemm2 stages every route-weighted row in this rank's ``grouped_out``. For a recv
slot ``t`` its ``c`` local rows are ``topids_to_rows[t*topk + k] >= 0``; their sum
goes out as ONE row, to the origin's ``comb_inp[origin_lid*topk + lead_k]`` where
``lead_k`` is the first local ``k``. The receiving reduce sums only those lead
slots, so a (token, rank) pair costs one row on the wire instead of ``c``.
"""

import os

import flydsl.compiler as flyc
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
from ..communication_ops_utils import atomic_add_lds, store_i32_lds, traced

WAVE = 32
# cpol SCOPE_DEV: the rows were staged by other XCDs, so miss this XCD's L2.
SCOPE_DEV = 16


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
# Step push, run after gemm2. The recv slots are split evenly over the blocks
# and, within a block, over waves 0-2 (the workers); wave 3 only sends. A worker
# packs its slots into steps of up to TMAX (the rows of one gather store) and
# runs every step through load -> sum, column chunk by column chunk (a chunk is
# 2 N tiles, 1 KB of a row). A slot's first FIXED rows sit at fixed input tile
# rows, so the sum reads them at constant offsets; rows past FIXED go to the
# step's OV-row overflow area. A step's 32 input rows are 4 TDM gathers.
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


def build_dedup_step_push_module(
    *,
    arena_handle,
    combine_input_offset,
    slot_stride_bytes,
    topk,
    max_tok,
    hidden,
    world_size,
    block_num=PUSH_BLOCKS,
):
    import mori.cco.device.flydsl as _cco

    assert 1 <= LAG < NOUT
    waves = WORKERS + 1
    lanes = waves * WAVE
    resv = TMAX * FIXED + OV  # rows of an input tile
    ov0 = TMAX * FIXED  # first overflow row
    assert resv <= WAVE and topk - FIXED <= OV
    row_e = 512  # 2 N tiles, 1 KB
    row_b = row_e * 2
    assert hidden % row_e == 0
    iters = hidden // row_e
    gathers = [(r0, min(r0 + 8, resv)) for r0 in range(0, resv, 8)]
    loads_per_step = len(gathers)
    in_b = resv * row_b
    out_b = TMAX * row_b
    wave_b = NIN * in_b + NOUT * out_b
    max_recv = max_tok * world_size
    tb_max = (max_recv + block_num - 1) // block_num
    tw_max = (tb_max + WORKERS - 1) // WORKERS
    # Every step holds at least this many slots, which bounds the step count.
    per_step_min = max(1, OV // max(1, topk - FIXED))
    maxs = (tw_max + per_step_min - 1) // per_step_min
    # Per slot: its local rows first then out_rows, count, dest, a scratch word.
    tm_w = topk + 3
    t_cnt, t_dst, t_junk = topk, topk + 1, topk + 2
    # Per step: slot count, packed rows, then per slot offset, count, dest.
    sm_w = 64
    s_n, s_pack = 0, 1
    s_off = s_pack + resv
    s_cnt = s_off + TMAX
    s_dst = s_cnt + TMAX
    s_maxc = sm_w - 2  # the step's largest count
    s_junk = sm_w - 1
    assert s_dst + TMAX <= s_maxc
    tm_b = ((tb_max * tm_w * 4 + 511) // 512) * 512
    sm_b = WORKERS * maxs * sm_w * 4
    # A barrier ends the packing before any tile is loaded, so the slot
    # metadata can live in the tiles.
    tm_in_tiles = tm_b <= WORKERS * wave_b
    # Per worker ready[NOUT], free[NOUT]; then each wave's step count.
    f_ready, f_free = 0, NOUT
    fl_w = 2 * NOUT
    f_nsw = WORKERS * fl_w
    flags_b = ((f_nsw + waves) * 4 + 15) // 16 * 16

    @flyc.kernel(
        name=f"ep_dedup_step_push_k{topk}_h{hidden}_w{world_size}_b{block_num}",
        known_block_size=[lanes, 1, 1],
    )
    def ep_dedup_step_push(
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
        tiles_base = smem.allocate(WORKERS * wave_b)._ptr
        tm_base = tiles_base if tm_in_tiles else smem.allocate(tm_b)._ptr
        sm_base = smem.allocate(sm_b)._ptr
        flag_base = smem.allocate(flags_b)._ptr
        tm_idx = fx.Int32(fx.ptrtoint(tm_base))
        sm_idx = fx.Int32(fx.ptrtoint(sm_base))
        flag_idx = fx.Int32(fx.ptrtoint(flag_base))
        is_worker = wave < fx.Int32(WORKERS)
        # The sender owns no tiles; clamp so its (unused) addresses stay in range.
        wk = is_worker.select(wave, fx.Int32(WORKERS - 1))
        my_b = wk * fx.Int32(wave_b)  # this wave's tiles, bytes from tiles_base
        in_idx = fx.Int32(fx.ptrtoint(tiles_base)) + my_b
        out_idx = in_idx + fx.Int32(NIN * in_b)

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
        for i in range(tid, WORKERS * maxs * sm_w, lanes):
            f = i % fx.Int32(sm_w)
            in_pack = (f >= fx.Int32(s_pack)) & (f < fx.Int32(s_off))
            in_dst = (f >= fx.Int32(s_dst)) & (f < fx.Int32(s_dst + TMAX))
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
        for i in range_constexpr(tw_max):
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
            junk = fx.Int32(s_junk)
            put(sw, live.select(fx.Int32(s_maxc), junk), mc)
            put(sw, live.select(fx.Int32(s_off) + nt, junk), used)
            put(sw, live.select(fx.Int32(s_cnt) + nt, junk), c)
            put(sw, live.select(fx.Int32(s_dst) + nt, junk), get(tm_idx, w0 + fx.Int32(t_dst)))
            for k in range_constexpr(topk):
                if const_expr(k < FIXED):
                    row_at = fx.Int32(s_pack + k) + nt * fx.Int32(FIXED)
                else:
                    row_at = fx.Int32(s_pack + ov0 + k - FIXED) + used
                put(sw, (live & (fx.Int32(k) < c)).select(row_at, junk),
                    get(tm_idx, w0 + fx.Int32(k)))
            put(sw, live.select(fx.Int32(s_n), junk), nt + fx.Int32(1))
            used = live.select(used + takes, used)
            nt = live.select(nt + fx.Int32(1), nt)
        n_sw = (n_w > fx.Int32(0)).select(st + fx.Int32(1), fx.Int32(0))
        n_steps = n_sw * fx.Int32(iters)
        put(flag_idx, fx.Int32(f_nsw) + wave, n_sw)
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
        lds_i16 = fx.recast_iter(_I16_SHARED(), tiles_base) + my_b // fx.Int32(2)
        in_view = fx.Tensor(fx.make_view(
            lds_i16, fx.make_layout((NIN * resv, row_e), (row_e, 1))))
        tiles_view = fx.Tensor(fx.make_view(
            fx.recast_iter(_I16_SHARED(), tiles_base),
            fx.make_layout((WORKERS * wave_b // row_b, row_e), (row_e, 1))))
        flag_addr = fx.Int64(fx.ptrtoint(flag_base))

        def flag(w, f, i):
            return flag_addr + fx.Int64((fx.Int32(w) * fx.Int32(fl_w) + fx.Int32(f) + i) * fx.Int32(4))

        n_sw_safe = (n_sw > fx.Int32(0)).select(n_sw, fx.Int32(1))

        def chunk_col(ch):
            """First column of the ch-th chunk this block visits. Each block
            starts at a different chunk, so the blocks' peer stores spread over
            the slot rows instead of all hitting one 1 KB column slice."""
            ch = (ch + fx.Int32(bid)) % fx.Int32(iters)
            return ch * fx.Int32(row_e)

        def step_coords(s):
            """Step s: column chunk s // n_sw, packed step s % n_sw of this wave."""
            ch = s // n_sw_safe
            return s - ch * n_sw_safe, chunk_col(ch)

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
            packed = lane_words(k, s_pack, resv)
            rows = [live.select(pick(packed, r), out_rows) for r in range_constexpr(resv)]
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
            oc = lane_words(k, s_off, 2 * TMAX)  # offsets, then counts
            mxc = get(my_sm, k * fx.Int32(sm_w) + fx.Int32(s_maxc))
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
            e0 = chunk_col(ch)
            dsts = lane_words(k, s_dst, TMAX, sm_idx + fx.Int32(w * maxs * sm_w * 4))
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
                _lds_poll_ge(flag(wave, f_free, ob), s // fx.Int32(NOUT))
                reduce_step(s, ob)
                rocdl.s_wait_dscnt(0)
                if lane == fx.Int32(0):
                    store_i32_lds(flag(wave, f_ready, ob), s + fx.Int32(1))
            tdm_ops.tensor_wait(0)
        else:
            nsw = [get(flag_idx, fx.Int32(f_nsw + w)) for w in range_constexpr(WORKERS)]
            nst = [x * fx.Int32(iters) for x in nsw]
            rounds = fx.Int32(0)
            for w in range_constexpr(WORKERS):
                rounds = (nst[w] > rounds).select(nst[w], rounds)
            # A worker out of steps still sends (no rows), keeping WORKERS
            # stores per round.
            for t in range(rounds):
                t_ob = t % fx.Int32(NOUT)
                for w in range_constexpr(WORKERS):
                    if t < nst[w]:
                        _lds_poll_ge(flag(w, f_ready, t_ob), t + fx.Int32(1))
                    send(w, nsw[w], t, t_ob, t < nst[w])
                if t >= fx.Int32(LAG):
                    tdm_ops.tensor_wait(WORKERS * LAG)
                    t_done = t - fx.Int32(LAG)
                    for w in range_constexpr(WORKERS):
                        if (lane == fx.Int32(0)) & (t_done < nst[w]):
                            store_i32_lds(flag(w, f_free, t_done % fx.Int32(NOUT)),
                                          t_done // fx.Int32(NOUT) + fx.Int32(1))
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
        ep_dedup_step_push(addr_out, out_rows, addr_rows, addr_nvt, addr_tis).launch(
            grid=(block_num, 1, 1), block=[lanes, 1, 1], stream=stream
        )

    return launch
