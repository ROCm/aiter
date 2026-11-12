# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""MiniMax-M3 NVFP4 sparse paged attention, decode (FlyDSL, gfx950).

Takes what the index selector emits: ``topk_idx [Hkv, rows, topk]`` logical 128-token
block ids (-1 padded, any order) and the request-level ``block_table``; the logical
block -> physical page lookup happens inside the kernel, so no sparse block table is
built between the selector and this kernel. A row's tail block (the one holding its
position) is masked positionally, so a partial tail needs no special placement.

One workgroup = one query row x the (<= 16) query heads of one kv head x one share of
the row's top-k slots. A row never has more than ``topk`` x 128 keys, so lane = top-k
slot: one ballot gives the valid slots and their order, and splitting a row over
workgroups is a static partition of those slots merged in split order (the last
workgroup to arrive merges, so the result does not depend on arrival order).

K and V are the exact e2m1 x e4m3 products in f16 (see the _common file), Q and P are
f16, scores / softmax / accumulation are fp32, the output is bf16.
"""

import ctypes
import functools
import math

import flydsl.compiler as flyc
import flydsl.expr as fx
import numpy as np
import torch
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from aiter.ops.flydsl.kernels import buffer_ops
from aiter.ops.flydsl.kernels.minimax_m3_sparse_pa_nvfp4_common import (
    BLK,
    DATA_B,
    HEAD_B,
    KPW,
    LOG2E,
    NDT,
    NKS,
    NW,
    PRELOAD_HINTS,
    RAW,
    SC1,
    SCALE_B,
    D,
    as_f32,
    bf16x8_to_f16,
    bits,
    cvt_fp4x2_f16,
    cvt_fp8x2_f16,
    exp2,
    global_f32_ptr,
    global_i32_ptr,
    maxf_nn,
    mfma_f16,
    pk_bf16,
    pk_f16,
    xlane4,
)
from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled

# Cache-policy bit of buffer loads: non-temporal (streaming). The K data is read once, so it
# streams; V stays cacheable (measured faster than streaming both).
NT = 0x2
PREFETCH = 1  # pages in flight while one is consumed; deeper rings measured slower
PART_F = 36  # fp32 per (row, kv head, split, lane) partial: O (NDT x 4), max, sum, pad
MAX_SPLITS = 16
MAX_SPLIT_ROWS = 1024  # (row, kv head) pairs a split launch can hold


def _atomic_add(addr_i64, value):
    """Relaxed agent-scope atomic add on a global i32; returns the old value."""
    return fx.Int32(
        _llvm.AtomicRMWOp(
            _llvm.AtomicBinOp.add,
            global_i32_ptr(addr_i64).llvm_ptr,
            fx.Int32(value).ir_value(),
            _llvm.AtomicOrdering.monotonic,
            syncscope="agent",
            alignment=4,
        ).result
    )


def _page_loader(arg_cache, page_bytes, kvh, hkv, w, n, g):
    """``(open_page, load_k, load_v)``: ``open_page(page, valid)`` is the buffer resource of a
    page and ``load_k`` / ``load_v`` read this lane's share of its K / V side as raw dwords:

      K: raw[0:4], raw[4:8]  K data of key rows st = 0, 1 (dims 32 g .. 32 g + 31)
         raw[8]              K scale bytes [st][group] of those rows
      V: raw[9:17]           V data, dword dt = 8 keys (8 g .. 8 g + 7) of dim 8 n + dt
         raw[17:19]          V scale bytes of keys 8 g .. 8 g + 7, dim group n // 2

    An invalid page is read through a zero-size buffer: no memory traffic, zeros back.
    """
    side_b = hkv * HEAD_B
    k_data = kvh * DATA_B
    k_scl = hkv * DATA_B + kvh * SCALE_B
    v_data = side_b + kvh * DATA_B
    v_scl = side_b + hkv * DATA_B + kvh * SCALE_B
    k_rows = [w * KPW + 8 * (n // 4) + 4 * st + n % 4 for st in range_constexpr(2)]
    k_load = [(k_data + r * (D // 2) + g * 16) // 4 for r in k_rows]
    k_scale_load = (k_scl + ((w * 4 + g) * 16 + n) * 4) // 4
    v_load = (v_data + ((w * 4 + g) * BLK + 8 * n) * 4) // 4
    v_scale_load = (v_scl + (n // 2) * BLK + w * KPW + 8 * g) // 4

    def open_page(page, valid):
        return buffer_ops.create_buffer_resource_from_addr(
            arg_cache + fx.Int64(page) * fx.Int64(page_bytes),
            num_records_bytes=valid.select(fx.Int32(page_bytes), fx.Int32(0)),
        )

    def load_k(pr):
        raw = []
        for st in range_constexpr(2):
            kd = fx.Vector(
                buffer_ops.buffer_load(
                    pr, k_load[st], vec_width=4, dtype=fx.Int32, cache_modifier=NT
                )
            )
            raw += [kd[i] for i in range_constexpr(4)]
        raw.append(
            fx.Int32(
                buffer_ops.buffer_load(pr, k_scale_load, vec_width=1, dtype=fx.Int32)
            )
        )
        return raw

    def load_v(pr):
        raw = []
        for h in range_constexpr(2):
            vd = fx.Vector(
                buffer_ops.buffer_load(pr, v_load + 4 * h, vec_width=4, dtype=fx.Int32)
            )
            raw += [vd[i] for i in range_constexpr(4)]
        vs = fx.Vector(
            buffer_ops.buffer_load(pr, v_scale_load, vec_width=2, dtype=fx.Int32)
        )
        raw += [vs[i] for i in range_constexpr(2)]
        return raw

    return open_page, load_k, load_v


def _dq4(word, scales):
    """4 dwords of 8 exact f16 products: e2m1 pair i of ``word`` times ``scales[i]`` (an f16 pair)."""
    return Vec.from_elements(
        [
            (cvt_fp4x2_f16(word, i) * scales[i]).bitcast(fx.Int32)[0]
            for i in range_constexpr(4)
        ],
        fx.Int32,
    )


def compile_decode(
    splits: int = 1,
    topk: int = 16,
    xcds: int = 0,
    late_v: bool = False,
):
    """Build the decode kernel for ``splits`` workgroups per (row, kv head). ``late_v`` issues
    the next page's V loads after the QK of the page in hand instead of together with its K
    loads: fewer bytes in flight, which pays when several
    workgroups share a CU."""
    assert 1 <= topk <= 64 and 1 <= splits <= topk
    SPLITS, TOPK, XCDS, LATE_V = splits, topk, xcds, late_v
    # Split partials need sc1 unless all splits of a row run on one XCD.
    PART_SCOPE = 0 if XCDS else SC1
    name = (
        f"m3_nvfp4_sparse_pa_decode_k{TOPK}_s{SPLITS}"
        + (f"_x{XCDS}" if XCDS else "")
        + ("_lv" if late_v else "")
    )

    @fx.struct
    class Shared:
        o: fx.Array[fx.Int32, NW * NDT * 64 * 4, 16]
        ml: fx.Array[fx.Float32, NW * 2 * 64]
        ticket: fx.Array[fx.Int32, 4]

    @flyc.kernel(name=name, known_block_size=[64 * NW, 1, 1])
    def kernel(
        i32_rows: fx.Int32,
        i32_ql: fx.Int32,  # rows per request
        i32_topk_hs: fx.Int32,  # topk_idx stride over kv heads (elements)
        i32_topk_rs: fx.Int32,  # topk_idx stride over rows (elements)
        i32_bt_s: fx.Int32,
        i32_H: fx.Int32,
        i32_hkv: fx.Int32,
        i32_group: fx.Int32,
        i32_reqs: fx.Int32,
        arg_seq: fx.Int64,  # [reqs] int32
        arg_topk: fx.Int64,  # [Hkv, >= rows, TOPK] int32 logical blocks, -1 = none
        arg_bt: fx.Int64,  # [reqs, stride] int32
        arg_q: fx.Int64,  # [rows, H, 128] bf16
        arg_cache: fx.Int64,  # K/V pages, see the _common file
        arg_ks: fx.Int64,  # fp32 global K scale
        arg_vs: fx.Int64,  # fp32 global V scale
        arg_out: fx.Int64,  # [rows, H, 128] bf16
        arg_part: fx.Int64,  # [(row * Hkv + kvh) * S + split, 64, PART_F] fp32
        arg_cnt: fx.Int64,  # [row * Hkv + kvh] int32 tickets, zero
        i64_part_bytes: fx.Int64,
        i32_page_byte: fx.Int32,
        f32_c: fx.Float32,  # sm_scale * log2(e)
    ):
        smem = fx.SharedAllocator().allocate(Shared).peek()
        tx = fx.thread_idx.x
        lane = tx % 64
        w = fx.Int32(fx.rocdl.readfirstlane(T.i32, tx // 64))
        n, g = lane % 16, lane // 16
        topk = global_i32_ptr(arg_topk)
        bt = global_i32_ptr(arg_bt)
        seq_p = global_i32_ptr(arg_seq)
        c = f32_c * fx.Float32(global_f32_ptr(arg_ks)[0])
        v_scale = fx.Float32(global_f32_ptr(arg_vs)[0])
        neg_inf = fx.Float32(float("-inf"))

        if const_expr(XCDS > 0):
            # 1-D grid; workgroup b runs on XCD b % XCDS. Unit u = (row, kv head); all
            # splits of a unit sit on XCD u % XCDS, so split partials only cross CUs of
            # one L2.
            xcd = fx.block_idx.x % XCDS
            slot = fx.block_idx.x // XCDS
            zz = slot % SPLITS
            unit = (slot // SPLITS) * XCDS + xcd
            row_ok = unit < i32_rows * i32_hkv
            unit = row_ok.select(unit, fx.Int32(0))
            kvh = unit % i32_hkv
            row = unit // i32_hkv
        else:
            row, kvh, zz = fx.block_idx.x, fx.block_idx.y, fx.block_idx.z
            row_ok = row < i32_rows
        req = row // i32_ql
        pos = fx.Int32(seq_p[req]) - i32_ql + row % i32_ql
        pos = row_ok.select(pos, fx.Int32(-1))
        pos = (pos >= 0).select(pos, fx.Int32(-1))  # padding row: sees nothing
        pos = fx.Int32(fx.rocdl.readfirstlane(T.i32, pos))
        req = fx.Int32(fx.rocdl.readfirstlane(T.i32, req))
        self_blk = (pos >= 0).select(pos // BLK, fx.Int32(-1))

        # Lane = top-k slot: its block id, the page behind it, and validity.
        slot_ok = lane < TOPK
        blk = fx.Int32(
            topk[
                kvh * i32_topk_hs
                + row * i32_topk_rs
                + slot_ok.select(lane, fx.Int32(0))
            ]
        )
        ok = slot_ok & (blk >= 0) & (blk <= self_blk)
        page = fx.Int32(bt[req * i32_bt_s + ok.select(blk, fx.Int32(0))])
        page = ok.select(page, fx.Int32(-1))
        blk = ok.select(blk, fx.Int32(-1))
        vmask = fx.Int64(fx.rocdl.ballot(T.i64, ok))
        n_v = fx.Int32(fx.math.ctpop(vmask).to(fx.Int32))
        if const_expr(SPLITS > 1):
            below = (fx.Int64(1) << fx.Int64(lane)) - fx.Int64(1)
            rank = fx.Int32(fx.math.ctpop(vmask & below).to(fx.Int32))
            chunk = (n_v + (SPLITS - 1)) // SPLITS
            lo = zz * chunk
            vmask = fx.Int64(
                fx.rocdl.ballot(T.i64, ok & (rank >= lo) & (rank < lo + chunk))
            )
            n_v = fx.Int32(fx.math.ctpop(vmask).to(fx.Int32))

        lim = pos % BLK - w * KPW  # last key of this wave's 32 a diagonal block shows
        qr = buffer_ops.create_buffer_resource_from_addr(
            arg_q, num_records_bytes=fx.Int64(i32_rows) * fx.Int64(i32_H) * (D * 2)
        )
        head = (n < i32_group).select(n, fx.Int32(0))  # masked heads read head 0
        q_row = (row * i32_H + kvh * i32_group + head) * (D // 2) + g * 16
        qf = [
            bf16x8_to_f16(
                fx.Vector(
                    buffer_ops.buffer_load(
                        qr, q_row + ks * 4, vec_width=4, dtype=fx.Int32
                    )
                )
            )
            for ks in range_constexpr(NKS)
        ]

        open_page, load_k, load_v = _page_loader(
            arg_cache, i32_page_byte, kvh, i32_hkv, w, n, g
        )
        zero4 = Vec.filled(4, 0.0, fx.Float32)
        key_off = [
            8 * g + 4 * st + e for st in range_constexpr(2) for e in range_constexpr(4)
        ]
        first = (n_v > 0).select(
            fx.Int32(fx.math.cttz(vmask).to(fx.Int32)), fx.Int32(0)
        )

        def slot_at(mask):
            """The lowest slot left in ``mask``, the first one when empty."""
            return (mask != fx.Int64(0)).select(
                fx.Int32(fx.math.cttz(mask).to(fx.Int32)), first
            )

        def page_of(slot):
            pg = fx.Int32(fx.rocdl.readlane(T.i32, page, slot))
            return (pg >= 0).select(pg, fx.Int32(0))

        def attend(raw, cur_blk, m_t, l_t, o_t, after_qk=None):
            """Folds one block into the running (max, sum, O^T). Each K / V fragment is
            dequantized right before the MFMA that consumes it, so few are live."""
            a = []
            for st in range_constexpr(2):
                ksc = cvt_fp8x2_f16(
                    raw[8], st == 1
                )  # [group 2g, group 2g + 1] of row st
                acc = zero4
                for ks in range_constexpr(NKS):
                    sv = Vec.from_elements([ksc[ks // 2], ksc[ks // 2]], fx.Float16)
                    acc = mfma_f16(_dq4(raw[st * 4 + ks], [sv] * 4), qf[ks], acc)
                a += [acc[e] for e in range_constexpr(4)]
            if after_qk is not None:
                after_qk()
            diag = cur_blk == self_blk
            # Only the tail block and the empty slots of the unrolled tail need masking.
            if diag | (cur_blk < 0):
                a = [
                    ((cur_blk < 0) | (diag & (key_off[i] > lim))).select(neg_inf, a[i])
                    for i in range_constexpr(8)
                ]
            mx = a[0]
            for i in range_constexpr(1, 8):
                mx = maxf_nn(mx, a[i])
            # The running max is in scaled units (c > 0).
            m_new = maxf_nn(m_t, xlane4(mx, maxf_nn) * c)
            m_use = (m_new == neg_inf).select(fx.Float32(0.0), m_new)
            p = [exp2(fx.math.fma(x, c, -m_use)) for x in a]
            ps = p[0]
            for i in range_constexpr(1, 8):
                ps = ps + p[i]
            alpha = exp2(m_t - m_use)
            l_t = l_t * alpha + ps
            pf = Vec.from_elements(
                [pk_f16(p[2 * j], p[2 * j + 1]) for j in range_constexpr(4)], fx.Int32
            )
            # alpha is exactly 1 unless some head's max grew.
            if fx.Int64(fx.rocdl.ballot(T.i64, m_new != m_t)) != fx.Int64(0):
                av = Vec.filled(4, alpha, fx.Float32)
                o_t = [x * av for x in o_t]
            vs = []
            for h in range_constexpr(2):
                vs += [
                    cvt_fp8x2_f16(raw[17 + h], False),
                    cvt_fp8x2_f16(raw[17 + h], True),
                ]
            o_t = [
                mfma_f16(_dq4(raw[9 + dt], vs), pf, o_t[dt])
                for dt in range_constexpr(NDT)
            ]
            return m_new, l_t, o_t

        def open_next(mask):
            """The page of the lowest slot left in ``mask``; nothing is read when empty."""
            return open_page(page_of(slot_at(mask)), mask != fx.Int64(0))

        def load_next(mask):
            pr = open_next(mask)
            return load_k(pr) + load_v(pr)

        # PREFETCH pages are in flight while one is consumed. The body is unrolled
        # PREFETCH + 1 times so the page buffers rotate by renaming, not by register
        # copies (a copy would wait for the data of every load in flight).
        U = PREFETCH + 1
        ahead = vmask
        ring = []
        for _ in range_constexpr(PREFETCH):
            ring += load_next(ahead)
            ahead = ahead & (ahead - fx.Int64(1))
        init = [vmask, ahead, neg_inf, fx.Float32(0.0)]
        init += [zero4 for _ in range_constexpr(NDT)]
        init += ring
        n_st = 4 + NDT
        results = init
        for u, st_ in range(  # type: ignore[call-overload]
            fx.Index(0),
            fx.Index((n_v + (U - 1)) // U),
            fx.Index(1),
            init=[fx.as_ir_value(x) for x in init],
        ):
            rem = fx.Int64(st_[0])
            ahead = fx.Int64(st_[1])
            m_t = fx.Float32(st_[2])
            l_t = fx.Float32(st_[3])
            o_t = [Vec(st_[4 + dt]) for dt in range_constexpr(NDT)]
            bufs = [
                [fx.Int32(st_[n_st + j * RAW + i]) for i in range_constexpr(RAW)]
                for j in range_constexpr(PREFETCH)
            ]
            for _ in range_constexpr(U):
                if const_expr(LATE_V):
                    cur = bufs.pop(0)
                    nxt_pr = open_next(ahead)
                else:
                    bufs.append(load_next(ahead))
                    cur = bufs.pop(0)
                ahead = ahead & (ahead - fx.Int64(1))
                cur_blk = (rem != fx.Int64(0)).select(
                    fx.Int32(fx.rocdl.readlane(T.i32, blk, slot_at(rem))), fx.Int32(-1)
                )
                rem = rem & (rem - fx.Int64(1))
                fx.rocdl.sched_barrier(0)
                if const_expr(LATE_V):
                    # K of the next page now, its V after the QK of this one.
                    nxt = load_k(nxt_pr)

                    def after_qk(nxt=nxt, nxt_pr=nxt_pr):
                        nxt.extend(load_v(nxt_pr))

                    bufs.append(nxt)
                else:
                    after_qk = None
                m_t, l_t, o_t = attend(cur, cur_blk, m_t, l_t, o_t, after_qk)
            state = [rem, ahead, m_t, l_t] + o_t
            for b_ in bufs:
                state += b_
            results = yield [fx.as_ir_value(x) for x in state]

        def store_out(acc, inv):
            outr = buffer_ops.create_buffer_resource_from_addr(
                arg_out,
                num_records_bytes=fx.Int64(i32_rows) * fx.Int64(i32_H) * (D * 2),
            )
            # Lane (n, g) holds dims [32 g, 32 g + 32) of head n in order.
            o_row = (row * i32_H + kvh * i32_group + n) * (D // 2) + g * 16
            for e in range_constexpr(4):
                val = Vec.from_elements(
                    [
                        pk_bf16(acc[2 * j][e] * inv, acc[2 * j + 1][e] * inv)
                        for j in range_constexpr(4)
                    ],
                    fx.Int32,
                )
                buffer_ops.buffer_store(
                    val, outr, o_row + e * 4, mask=(n < i32_group) & row_ok
                )

        # Combine the waves' partials in wave 0.
        o_lds = fx.logical_divide(
            fx.make_view(
                fx.recast_iter(fx.Int32, smem.o.ptr),
                fx.make_layout(NW * NDT * 64 * 4, 1),
            ),
            fx.make_layout(4, 1),
        )
        lds_atom = fx.make_copy_atom(fx.UniversalCopy128b(), fx.Int32)
        ml = smem.ml.ptr
        ml[(w * 2) * 64 + lane] = fx.Float32(results[2])
        ml[(w * 2 + 1) * 64 + lane] = fx.Float32(results[3])
        for dt in range_constexpr(NDT):
            r = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Int32)
            r.store(Vec(results[4 + dt]).bitcast(fx.Int32))
            fx.copy(lds_atom, r, fx.slice(o_lds, (None, (w * NDT + dt) * 64 + lane)))
        fx.gpu.barrier()
        if w == 0:
            ms = [fx.Float32(ml[(ww * 2) * 64 + lane]) for ww in range_constexpr(NW)]
            big = ms[0]
            for ww in range_constexpr(1, NW):
                big = maxf_nn(big, ms[ww])
            big_raw = big
            big = (big == neg_inf).select(fx.Float32(0.0), big)
            wt = [fx.Float32(fx.math.exp2(ms[ww] - big)) for ww in range_constexpr(NW)]
            l_tot = fx.Float32(0.0)
            for ww in range_constexpr(NW):
                l_tot = l_tot + fx.Float32(ml[(ww * 2 + 1) * 64 + lane]) * wt[ww]
            l_tot = xlane4(l_tot, lambda x, y: x + y)
            acc = [zero4 for _ in range_constexpr(NDT)]
            for ww in range_constexpr(NW):
                wv = Vec.filled(4, wt[ww], fx.Float32)
                for dt in range_constexpr(NDT):
                    r = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Int32)
                    fx.copy(
                        lds_atom,
                        fx.slice(o_lds, (None, (ww * NDT + dt) * 64 + lane)),
                        r,
                    )
                    acc[dt] = acc[dt] + Vec(r.load()).bitcast(fx.Float32) * wv
            if const_expr(SPLITS > 1):
                # Padding workgroups (row_ok false) work on a spare unit, never a real one.
                tid = row_ok.select(row * i32_hkv + kvh, fx.Int32(MAX_SPLIT_ROWS))
                pr = buffer_ops.create_buffer_resource_from_addr(
                    arg_part, num_records_bytes=i64_part_bytes
                )
                row_f = (tid * SPLITS + zz) * 64 + lane
                for dt in range_constexpr(NDT):
                    buffer_ops.buffer_store(
                        acc[dt].bitcast(fx.Int32),
                        pr,
                        row_f * PART_F + dt * 4,
                        cache_modifier=PART_SCOPE,
                    )
                buffer_ops.buffer_store(
                    Vec.from_elements([bits(big_raw), bits(l_tot)], fx.Int32),
                    pr,
                    row_f * PART_F + NDT * 4,
                    cache_modifier=PART_SCOPE,
                )
                # The partial is stored before the ticket is taken.
                fx.rocdl.s_waitcnt(vmcnt=0)
                if lane == 0:
                    smem.ticket.ptr[0] = _atomic_add(arg_cnt + fx.Int64(tid) * 4, 1)
                fx.rocdl.s_waitcnt(lgkmcnt=0)
                if (
                    fx.Int32(
                        fx.rocdl.readfirstlane(T.i32, fx.Int32(smem.ticket.ptr[0]))
                    )
                    == SPLITS - 1
                ):
                    rows_f = [
                        (tid * SPLITS + sp) * 64 + lane
                        for sp in range_constexpr(SPLITS)
                    ]
                    mls = [
                        Vec(
                            buffer_ops.buffer_load(
                                pr,
                                rf * PART_F + NDT * 4,
                                vec_width=2,
                                dtype=fx.Int32,
                                cache_modifier=PART_SCOPE,
                            )
                        )
                        for rf in rows_f
                    ]
                    bigs = [as_f32(x[0]) for x in mls]
                    top = bigs[0]
                    for sp in range_constexpr(1, SPLITS):
                        top = maxf_nn(top, bigs[sp])
                    top = (top == neg_inf).select(fx.Float32(0.0), top)
                    ws = [fx.Float32(fx.math.exp2(b - top)) for b in bigs]
                    l_all = fx.Float32(0.0)
                    for sp in range_constexpr(SPLITS):
                        l_all = l_all + as_f32(mls[sp][1]) * ws[sp]
                    inv_all = (l_all > fx.Float32(0.0)).select(
                        v_scale / l_all, fx.Float32(0.0)
                    )
                    acc_all = [zero4 for _ in range_constexpr(NDT)]
                    for sp in range_constexpr(SPLITS):
                        wv = Vec.filled(4, ws[sp], fx.Float32)
                        for dt in range_constexpr(NDT):
                            part = Vec(
                                buffer_ops.buffer_load(
                                    pr,
                                    rows_f[sp] * PART_F + dt * 4,
                                    vec_width=4,
                                    dtype=fx.Int32,
                                    cache_modifier=PART_SCOPE,
                                )
                            ).bitcast(fx.Float32)
                            acc_all[dt] = acc_all[dt] + part * wv
                    store_out(acc_all, inv_all)
                    if lane == 0:
                        global_i32_ptr(arg_cnt + fx.Int64(tid) * 4)[0] = fx.Int32(0)
            else:
                inv = (l_tot > fx.Float32(0.0)).select(v_scale / l_tot, fx.Float32(0.0))
                store_out(acc, inv)

    @flyc.jit
    def launch(
        arg_q: fx.Int64,
        arg_cache: fx.Int64,
        arg_topk: fx.Int64,
        arg_bt: fx.Int64,
        arg_seq: fx.Int64,
        arg_out: fx.Int64,
        arg_ks: fx.Int64,
        arg_vs: fx.Int64,
        arg_part: fx.Int64,
        arg_cnt: fx.Int64,
        i64_part_bytes: fx.Int64,
        i32_rows: fx.Int32,
        i32_ql: fx.Int32,
        i32_H: fx.Int32,
        i32_hkv: fx.Int32,
        i32_group: fx.Int32,
        i32_reqs: fx.Int32,
        i32_grid_x: fx.Int32,
        i32_topk_hs: fx.Int32,
        i32_topk_rs: fx.Int32,
        i32_bt_s: fx.Int32,
        i32_page_b: fx.Int32,
        f32_c: fx.Float32,
        stream: fx.Stream,
    ):
        kernel(
            i32_rows,
            i32_ql,
            i32_topk_hs,
            i32_topk_rs,
            i32_bt_s,
            i32_H,
            i32_hkv,
            i32_group,
            i32_reqs,
            arg_seq,
            arg_topk,
            arg_bt,
            arg_q,
            arg_cache,
            arg_ks,
            arg_vs,
            arg_out,
            arg_part,
            arg_cnt,
            i64_part_bytes,
            i32_page_b,
            f32_c,
        ).launch(
            grid=(i32_grid_x, 1 if XCDS else i32_hkv, 1 if XCDS else SPLITS),
            block=(64 * NW, 1, 1),
            stream=stream,
        )

    launch.kernel_name = name
    launch.compile_hints = PRELOAD_HINTS
    return launch


# ---------------------------------------------------------------------------
# host side
# ---------------------------------------------------------------------------
@functools.cache
def _get_kernel(splits: int, topk: int, xcds: int, late_v: bool, device_index: int):
    # device_index only keys the cache: the launcher memoises its compiled handle, which
    # is bound to the device it was compiled on.
    return compile_decode(splits, topk, xcds, late_v)


@functools.cache
def _gfx_arch(device_index: int) -> str:
    return torch.cuda.get_device_properties(device_index).gcnArchName.split(":")[0]


@functools.cache
def _cu_count(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).multi_processor_count


@functools.cache
def _unit_scale(device_index: int) -> torch.Tensor:
    return torch.ones(1, dtype=torch.float32, device=f"cuda:{device_index}")


@functools.cache
def _split_buffers(device_index: int):
    """Split partials and tickets, one set per device; allocated on first use, so warm up
    before capture. Launches that split a row share them, so they must not overlap on
    different streams of one device (see ``minimax_m3_sparse_pa_nvfp4_decode``)."""
    dev = f"cuda:{device_index}"
    part = torch.empty(
        (MAX_SPLIT_ROWS + 1) * MAX_SPLITS * 64 * PART_F, dtype=torch.float32, device=dev
    )
    return part, torch.zeros(MAX_SPLIT_ROWS + 1, dtype=torch.int32, device=dev)


_HIP_ATTR_NUM_XCCS = 10018


@functools.cache
def _xcd_count(device_index: int) -> int:
    """XCDs of the device when HIP reports 8 (gfx950's topology, which the XCD-aware
    grid assumes: workgroup b is dispatched to XCD b % 8), else 0 (grid off)."""
    try:
        get_attr = ctypes.CDLL("libamdhip64.so").hipDeviceGetAttribute
        get_attr.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int]
        get_attr.restype = ctypes.c_int
        v = ctypes.c_int()
        ok = get_attr(ctypes.byref(v), _HIP_ATTR_NUM_XCCS, device_index) == 0
        return 8 if ok and v.value == 8 else 0
    except (AttributeError, OSError, TypeError, ctypes.ArgumentError):
        return 0


def decode_splits(units: int, topk: int, cus: int) -> int:
    """Splits of a row's top-k slots for ``units`` = rows x kv heads workgroup rows.

    Measured on MI355X (rows 1..256, Hkv 1 and 4, 16 valid blocks): one workgroup walks
    its blocks serially (~0.6 us each), so splitting pays until the grid reaches the CU
    count, and the best split is the largest power of two with units * S <= CUs, at
    most 8 (4 from 16 units up: the merge's fixed cost grows with S). Each split keeps
    at least 2 slots.
    """
    cap = 8 if units < 16 else 4
    s = 1
    while s * 2 <= cap and units * s * 2 <= cus and topk // (s * 2) >= 2:
        s *= 2
    return s


_WG_PER_CU = 4  # resident workgroups per CU at 122 VGPR and 34.8 KB of LDS


def decode_late_v(workgroups: int, cus: int) -> bool:
    """Whether the V loads should trail the K loads (``late_v``).

    Measured on MI355X (cold 8 GB pool, 16 valid blocks per row): with 1.5 or more
    workgroups per CU and a single round of workgroups (<= 4 per CU) the kernel is 1-3.5%
    faster with fewer bytes in flight; with 1 per CU it needs the bytes in flight (-6..-10%
    when late), and a partial second round (e.g. 1280 workgroups) is slowed the same way
    by the straggler workgroups (-1..-6%).
    """
    return 3 * cus <= 2 * workgroups <= 2 * _WG_PER_CU * cus


def minimax_m3_sparse_pa_nvfp4_decode(
    q: torch.Tensor,  # [rows, Hq, 128] bf16
    kv_cache: torch.Tensor,  # uint8 pages, [pages, 2 * Hkv * 9216] (any view with page stride)
    topk_idx: torch.Tensor,  # [Hkv, >= rows, topk] int32 logical blocks, -1 = none
    block_table: torch.Tensor,  # [reqs, max_blocks] int32, logical block -> page
    seq_lens: torch.Tensor,  # [reqs] int32 kv length including this step's query tokens
    sm_scale: float,
    out: torch.Tensor | None = None,  # [rows, Hq, 128] bf16
    k_scale: torch.Tensor | None = None,  # [1] fp32 global K scale (default 1)
    v_scale: torch.Tensor | None = None,  # [1] fp32 global V scale (default 1)
    query_len: int = 1,
    splits: int | None = None,
    xcds: int | None = None,
    late_v: bool | None = None,
) -> torch.Tensor:
    """Row i is query token i % query_len of request i // query_len and attends to
    the keys of its ``topk_idx`` blocks up to its position.

    gfx950 only. Every operand must be on ``q``'s device; ``k_scale`` / ``v_scale`` are
    one positive float32 element each (a device value, not checked) and ``sm_scale``
    must be finite and positive. One launch in flight per device: the split partials and
    arrival counters are one shared buffer per device, so two launches must not overlap
    on different streams of the same GPU (launches on one stream, and graph replays, are
    serialized).
    """
    if q.device.type != "cuda":
        raise ValueError(f"q must be a CUDA tensor, got {q.device}")
    if _gfx_arch(q.device.index) != "gfx950":
        raise ValueError(
            f"NVFP4 sparse PA decode requires gfx950, got {_gfx_arch(q.device.index)}"
        )
    rows, num_heads, head_dim = q.shape
    hkv = topk_idx.shape[0]
    topk = topk_idx.shape[-1]
    if out is None:
        out = torch.empty_like(q)
    if rows == 0:
        return out
    group = num_heads // hkv
    assert head_dim == D and group * hkv == num_heads and group in (8, 16)
    assert q.dtype == torch.bfloat16 and out.dtype == torch.bfloat16
    assert q.is_contiguous() and out.is_contiguous() and out.shape == q.shape
    # Q and out are addressed with 32-bit byte offsets.
    assert q.numel() * q.element_size() < 2**31, "q must be below 2 GiB"
    page_b = 2 * hkv * (BLK * D // 2 + BLK * D // 16)
    assert kv_cache.dtype == torch.uint8 and kv_cache.stride(0) < 2**31
    assert (
        kv_cache.dim() == 2 and kv_cache.stride(1) == 1
    ), "kv_cache must be 2-D with a unit last stride"
    assert (
        kv_cache.stride(0) >= page_b and kv_cache.size(1) >= page_b
    ), "kv_cache page is smaller than its payload"
    assert topk_idx.dtype == torch.int32 and topk_idx.shape[1] >= rows
    assert topk_idx.stride(2) == 1
    for t in (block_table, seq_lens):
        assert t.dtype == torch.int32 and t.stride(-1) == 1
    assert rows == seq_lens.shape[0] * query_len
    for t in (out, kv_cache, topk_idx, block_table, seq_lens):
        assert t.device == q.device, "all operands must be on q's device"
    dev = q.device.index
    if splits is None:
        splits = 1
        if rows * hkv <= MAX_SPLIT_ROWS:
            splits = decode_splits(rows * hkv, topk, _cu_count(dev))
    assert 1 <= splits <= MAX_SPLITS, f"splits must be in [1, {MAX_SPLITS}]"
    assert splits == 1 or rows * hkv <= MAX_SPLIT_ROWS
    if xcds is None:
        # The XCD-aware 1-D grid only pays when splits of a row must share an L2; with one
        # workgroup per unit the plain (row, kv head) grid skips its runtime divisions.
        xcds = _xcd_count(dev) if splits > 1 else 0
    if late_v is None:
        late_v = decode_late_v(rows * hkv * splits, _cu_count(dev))
    part, cnt = _split_buffers(dev)
    unit = _unit_scale(dev)
    ks = unit if k_scale is None else k_scale
    vs = unit if v_scale is None else v_scale
    for t in (ks, vs):
        assert (
            t.device == q.device and t.dtype == torch.float32 and t.numel() == 1
        ), "k_scale and v_scale must be one float32 element on q's device"
    # The softmax runs in scaled units with masked scores at -inf, so it needs c > 0.
    c = float(np.float32(sm_scale) * np.float32(LOG2E))
    assert math.isfinite(c) and c > 0, "sm_scale must be finite and positive"
    reqs = seq_lens.shape[0]
    if xcds:
        grid_x = -(-(rows * hkv) // xcds) * xcds * splits
    else:
        grid_x = rows
    with torch.cuda.device(q.device):
        _run_compiled(
            _get_kernel(splits, topk, xcds, late_v, dev),
            q.data_ptr(),
            kv_cache.data_ptr(),
            topk_idx.data_ptr(),
            block_table.data_ptr(),
            seq_lens.data_ptr(),
            out.data_ptr(),
            ks.data_ptr(),
            vs.data_ptr(),
            part.data_ptr(),
            cnt.data_ptr(),
            part.numel() * 4,
            rows,
            query_len,
            num_heads,
            hkv,
            group,
            reqs,
            grid_x,
            topk_idx.stride(0),
            topk_idx.stride(1),
            block_table.stride(0),
            kv_cache.stride(0),
            c,
            torch.cuda.current_stream(q.device),
        )
    return out
