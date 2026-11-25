# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""MoE router GEMM + biased sigmoid top-k + MXFP4 token rows for one tile of at most 8 rows (gfx1250, FlyDSL).

The bytes of ``moe_router_topk_gfx1250`` at ``split_k`` 8 with ``quant_rows`` 1 (same chains, operands, fragment
slots and split-order sum; the same gate arithmetic; the same MX helpers), laid out for the decode tile:

* one cluster of 16 workgroups: workgroup ``c`` = (K split ``c // 2``, expert group ``c % 2``), wave ``w`` runs the
  24-step chain of 16-expert slice ``4 * (c % 2) + w`` -- production's geometry at ``split_k`` 8;
* the split's x rows are loaded once per workgroup and staged in LDS as WMMA B fragments (production loads them per
  wave and step, the same lines four times), and each workgroup writes half of the split's MXFP4 blocks while
  staging (production quantizes after the cluster barrier, on the gating waves);
* all 24 K steps of a wave's weights are issued at entry (production keeps 8 in flight, three dependent windows);
* the gate's bias / num_valid_rows loads are issued before the cluster barrier.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, range_constexpr, rocdl
from flydsl.expr.rocdl import cluster
from flydsl.expr.typing import Int32, T
from flydsl.expr.typing import Vector as Vec

from aiter.ops.flydsl.kernels.buffer_ops import (
    buffer_load,
    buffer_store,
    create_buffer_resource_from_addr,
)
from aiter.ops.flydsl.kernels.gemm_common_gfx1250 import make_lds_copy_ops

EXPERTS = 128
WAVE = 32
ROWS = 8  # token rows of the tile (m <= ROWS)
SPLIT = 8
GROUPS = 2
CSIZE = SPLIT * GROUPS
WAVES = 4
LOG2E = 1.4426950408889634
RENORM_SUM_FLOOR = 1e-20
FLT_MIN = 1.1754943508222875e-38


def _rsrc(addr, nbytes=None):
    return create_buffer_resource_from_addr(fx.Int64(addr), num_records_bytes=nbytes)


def _ld(rsrc, idx, words, cpol=0):
    return buffer_load(
        rsrc, fx.Int32(idx), vec_width=words, dtype=T.i32, cache_modifier=cpol
    )


def _uni(v):
    return fx.Int32(rocdl.readfirstlane(T.i32, fx.as_ir_value(fx.Int32(v))))


def _rmem(v, n, dtype):
    t = fx.make_rmem_tensor(n, dtype)
    t.store(v if isinstance(v, fx.Vector) else fx.Vector(v))
    return t


def _cat2(lo, hi):
    return Vec(lo).shuffle(Vec(hi), list(range(8)))


def _amax_bf16(chunks):
    """max |x| (fp32) over the bf16 pairs of ``chunks`` (vec4 i32 each) on the bit patterns: |x| of a bf16 is its low
    15 bits and non-negative floats order as their bits, so this is the fp32 max of the values.
    """
    m = fx.Int32(0)
    for v in chunks:
        for d in range(4):
            w = fx.Int32(Vec(v)[d])
            m = fx.max(m, w & fx.Int32(0x7FFF))
            m = fx.max(m, (w >> fx.Int32(16)) & fx.Int32(0x7FFF))
    return (m << fx.Int32(16)).bitcast(fx.Float32)


def build_router8_module(
    K: int = 6144,
    topk: int = 4,
    need_renorm: bool = True,
    routed_scaling_factor: float = 1.0,
    w_cpol: int = 0,
    xpf: int = 3,
):
    """Launcher ``(x, w, bias, num_valid_rows, ws, topk_weights, topk_ids, m, lda, has_nvr, xq, xs, stream)``;
    ``m <= 8``, ``ws`` holds ``8 * 8 * 128`` fp32, ``xq`` / ``xs`` receive the MXFP4 rows (``quant_rows``).
    """
    from aiter.ops.flydsl.kernels.moe_fused_route_quant_scatter import (
        _ROUND_MODE,
        _cvt_scalef32_pk8_fp4_bf16,
    )
    from aiter.ops.flydsl.kernels.quant_utils import emit_mx_e8m0_scale
    from aiter.utility.mx_types import MxDtypeInt

    chunk = K // SPLIT
    steps = chunk // 32
    assert chunk * SPLIT == K and steps * 32 == chunk
    blocks = ROWS * steps  # MX blocks of 32 values of the split's x
    xr = -(-blocks // (WAVES * WAVE))  # blocks per thread
    lds_bytes = steps * 2 * 16 * 16
    scale = float(routed_scaling_factor)
    neg_inf = float("-inf")
    name = (
        f"moe_router8_topk_k{K}_t{topk}_r{int(need_renorm)}_w{w_cpol}_xp{xpf}"
        f"{'' if scale == 1.0 else '_sc'}"
    )

    @fx.struct
    class SharedStorage:
        buf: fx.Array[fx.Int32, lds_bytes // 4, 16]

    def gate_row(ws_rs, row, lane, bias_v, has_nvr, nvr):
        """(weight, id) of top-k slot ``lane`` (lanes < topk) of token ``row``: moe_router_topk's _gate_row."""
        ept = EXPERTS // WAVE
        parts = [
            [
                fx.Float32(
                    buffer_load(
                        ws_rs,
                        (sidx * ROWS + row) * EXPERTS + lane + WAVE * i,
                        vec_width=1,
                        dtype=T.f32,
                    )
                )
                for i in range_constexpr(ept)
            ]
            for sidx in range_constexpr(SPLIT)
        ]
        vals, orig, idxs = [], [], []
        for i in range_constexpr(ept):
            total = parts[0][i]
            for sidx in range_constexpr(1, SPLIT):
                total = total + parts[sidx][i]
            logit = total.to(fx.BFloat16).to(fx.Float32)
            e2 = fx.exp2((-logit) * fx.Float32(LOG2E))
            score = fx.Float32(rocdl.rcp(T.f32, fx.Float32(1.0) + e2))
            score = (score < fx.Float32(FLT_MIN)).select(fx.Float32(0.0), score)
            is_nan = score != score  # noqa: PLR0124 -- traced value: the NaN test
            orig.append(is_nan.select(fx.Float32(0.0), score))
            v = score + bias_v[i]
            v = (v != v).select(fx.Float32(neg_inf), v)  # noqa: PLR0124
            vals.append(v)
            idxs.append(lane + WAVE * i)

        def cas(a, b):
            sw = vals[a] < vals[b]
            vals[a], vals[b] = sw.select(vals[b], vals[a]), sw.select(vals[a], vals[b])
            orig[a], orig[b] = sw.select(orig[b], orig[a]), sw.select(orig[a], orig[b])
            idxs[a], idxs[b] = sw.select(idxs[b], idxs[a]), sw.select(idxs[a], idxs[b])

        for a, b in ((0, 1), (2, 3), (0, 2), (1, 3), (1, 2)):
            cas(a, b)

        cursor = fx.Int32(0)
        wsum = fx.Float32(0.0)
        my_id = fx.Int32(0)
        my_w = fx.Float32(0.0)
        for k in range_constexpr(topk):
            head_v = fx.Float32(neg_inf)
            head_i = fx.Int32(0)
            head_o = fx.Float32(0.0)
            for i in range_constexpr(ept - 1, -1, -1):
                at = cursor == i
                head_v = at.select(vals[i], head_v)
                head_i = at.select(idxs[i], head_i)
                head_o = at.select(orig[i], head_o)
            mx = head_v
            for off in (16, 8, 4, 2, 1):
                mx = fx.maxnumf(mx, mx.shuffle_xor(fx.Int32(off), fx.Int32(WAVE)))
            mask = fx.Int32(rocdl.ballot(T.i32, head_v == mx))
            win = (mask != 0).select(fx.Int32(fx.cttz(mask)), fx.Int32(0))
            win_id = fx.Int32(rocdl.readlane(T.i32, head_i, win))
            won = (cursor < ept) & (head_i == win_id)
            mine = won.select(head_o, fx.Float32(0.0))
            cursor = cursor + won.select(fx.Int32(1), fx.Int32(0))
            weight = fx.Int32(
                rocdl.readlane(T.i32, mine.bitcast(fx.Int32), win_id & (WAVE - 1))
            ).bitcast(fx.Float32)
            is_k = lane == k
            my_id = is_k.select(win_id, my_id)
            my_w = is_k.select(weight, my_w)
            if const_expr(need_renorm):
                wsum = wsum + weight
        if const_expr(need_renorm):
            inv = fx.Float32(scale) / fx.maxnumf(wsum, fx.Float32(RENORM_SUM_FLOOR))
        else:
            inv = fx.Float32(scale)
        out_w = my_w * inv
        pad = (has_nvr != 0) & (row >= nvr)
        out_w = pad.select(fx.Float32(0.0), out_w)
        out_id = pad.select(fx.Int32(-1), my_id)
        return out_w, out_id

    @flyc.kernel(name=name, known_block_size=[WAVES * WAVE, 1, 1])
    def router8_kernel(
        x: fx.Pointer,
        w: fx.Pointer,
        bias: fx.Pointer,
        num_valid_rows: fx.Pointer,
        ws: fx.Pointer,
        topk_weights: fx.Pointer,
        topk_ids: fx.Pointer,
        m: Int32,
        lda: Int32,
        has_nvr: Int32,
        xq_p: fx.Pointer,
        xs_p: fx.Pointer,
    ):
        tid = fx.Int32(fx.thread_idx.x)
        c = fx.Int32(fx.block_idx.x)
        wave = _uni(tid // WAVE)
        lane = tid % WAVE
        lane16 = lane % 16
        kgrp = lane // 16
        s = c // GROUPS
        ng = c % GROUPS
        lds_all = fx.SharedAllocator().allocate(SharedStorage).peek()
        lds_idx = fx.Int64(fx.ptrtoint(fx.recast_iter(fx.Uint8, lds_all.buf.ptr)))
        lds_load_b128, lds_store_b128 = make_lds_copy_ops(128)

        # x: MX block b = tok * steps + j of the split (32 values, 64 B) -> thread tid + 128 r. Rows >= m read row
        # m - 1 (production's clamped tail lanes).
        x_rs = _rsrc(fx.Int64(fx.ptrtoint(x)))
        rb = [tid + WAVES * WAVE * r for r in range(xr)]
        rlive = [rb[r] < blocks for r in range(xr)]
        rtok = [rlive[r].select(rb[r], fx.Int32(0)) // steps for r in range(xr)]
        rj = [rlive[r].select(rb[r], fx.Int32(0)) % steps for r in range(xr)]
        rtl = [(rtok[r] < m).select(rtok[r], m - 1) for r in range(xr)]
        rdw = [(rtl[r] * lda + s * chunk + rj[r] * 32) // 2 for r in range(xr)]
        xc = [
            [Vec(_ld(x_rs, rdw[r] + cc * 4, 4)) for cc in range_constexpr(4)]
            for r in range_constexpr(xr)
        ]
        rocdl.sched_barrier(0)
        # this wave's 24 K steps of weights (production's A fragment: K [32 j + 8 kgrp, +8) and [32 j + 16 + 8 kgrp, +8)
        # of row 16 slice + lane16), all in flight at once
        w_rs = _rsrc(fx.Int64(fx.ptrtoint(w)), EXPERTS * K * 2)
        slc = ng * WAVES + wave
        a_dw = ((slc * 16 + lane16) * K + s * chunk + kgrp * 8) // 2
        wa = [
            [
                Vec(_ld(w_rs, a_dw + j * 16 + h * 8, 4, w_cpol))
                for h in range_constexpr(2)
            ]
            for j in range_constexpr(steps)
        ]
        rocdl.sched_barrier(0)
        bias_rs = _rsrc(fx.Int64(fx.ptrtoint(bias)))
        bias_v = [
            fx.Float32(buffer_load(bias_rs, lane + WAVE * i, vec_width=1, dtype=T.f32))
            for i in range_constexpr(EXPERTS // WAVE)
        ]
        nvr = fx.Int32(
            buffer_load(
                _rsrc(fx.Int64(fx.ptrtoint(num_valid_rows))),
                fx.Int32(0),
                vec_width=1,
                dtype=T.i32,
            )
        )
        xq_rs = _rsrc(fx.Int64(fx.ptrtoint(xq_p)), m * (K // 2))
        xs_rs = _rsrc(fx.Int64(fx.ptrtoint(xs_p)), m * (K // 32))
        for r in range_constexpr(xr):
            rocdl.sched_barrier(0)
            tok = rtok[r]
            j = rj[r]
            ch = xc[r]
            if rlive[r]:
                # B fragment slots: (step j, half h, kgrp, token) 16 B each; chunk cc = values [8 cc, +8)
                for cc in range_constexpr(4):
                    lds_store_b128(
                        lds_idx,
                        ((j * 2 + cc // 2) * 16 + (cc % 2) * 8 + tok) * 16,
                        ch[cc],
                    )
                # MXFP4 block (tok, s * steps + j): group 0 writes the blocks of r = 0, group 1 the rest
                if ((ng == 0) == (r == 0)) & (tok < m):
                    bfs = [ch[cc].bitcast(fx.BFloat16) for cc in range_constexpr(4)]
                    amax = _amax_bf16(ch)
                    e8m0 = fx.Int32(
                        emit_mx_e8m0_scale(
                            amax, mode=_ROUND_MODE, dtype=MxDtypeInt.FP4_E2M1
                        )
                    )
                    sc = (e8m0 << fx.Int32(23)).bitcast(fx.Float32)
                    pq = [
                        fx.Int32(_cvt_scalef32_pk8_fp4_bf16(bfs[cc], sc, i32_ty=T.i32))
                        for cc in range_constexpr(4)
                    ]
                    buffer_store(
                        Vec.from_elements(pq, fx.Int32),
                        xq_rs,
                        tok * (K // 2) + (s * steps + j) * 16,
                        offset_is_bytes=True,
                    )
                    buffer_store(
                        fx.Uint8(e8m0),
                        xs_rs,
                        tok * (K // 32) + s * steps + j,
                        offset_is_bytes=True,
                    )
        rocdl.s_wait_dscnt(0)
        rocdl.s_barrier_signal(-1)
        rocdl.s_barrier_wait(-1)

        tok16 = lane16 % ROWS

        def x_frag(j):
            return _cat2(
                lds_load_b128(lds_idx, ((j * 2 + 0) * 16 + kgrp * 8 + tok16) * 16),
                lds_load_b128(lds_idx, ((j * 2 + 1) * 16 + kgrp * 8 + tok16) * 16),
            ).bitcast(fx.BFloat16)

        wmma = fx.make_mma_atom(fx.rocdl.WMMA(16, 16, 32, fx.BFloat16, fx.Float32))
        acc = fx.full(8, 0.0, fx.Float32)
        pend = [x_frag(j) for j in range_constexpr(xpf)]
        for j in range_constexpr(steps):
            if const_expr(j + xpf < steps):
                pend.append(x_frag(j + xpf))
            rocdl.sched_barrier(0)
            ta = _rmem(acc, 8, fx.Float32)
            fx.gemm(
                wmma,
                ta,
                _rmem(_cat2(wa[j][0], wa[j][1]).bitcast(fx.BFloat16), 16, fx.BFloat16),
                _rmem(pend[j], 16, fx.BFloat16),
                ta,
            )
            acc = ta.load()
            pend[j] = None
            wa[j] = None
            rocdl.sched_barrier(0)
        # acc[i]: expert 16 slc + 8 kgrp + i of token lane16 -> ws [split][token][expert]
        ws_rs = _rsrc(fx.Int64(fx.ptrtoint(ws)))
        if lane16 < ROWS:
            av = Vec(acc)
            for half in range_constexpr(2):
                part = Vec.from_elements(
                    [av[half * 4 + i] for i in range_constexpr(4)], fx.Float32
                )
                buffer_store(
                    part,
                    ws_rs,
                    (s * ROWS + lane16) * EXPERTS + slc * 16 + kgrp * 8 + half * 4,
                )
        fx.memory_fence(syncscope="cluster", ordering=fx.AtomicOrdering.Release)
        cluster.cluster_barrier()
        fx.memory_fence(syncscope="cluster", ordering=fx.AtomicOrdering.Acquire)
        if (wave == 0) & (c < m):
            out_w, out_id = gate_row(ws_rs, c, lane, bias_v, has_nvr, nvr)
            if lane < topk:
                buffer_store(
                    out_w, _rsrc(fx.Int64(fx.ptrtoint(topk_weights))), c * topk + lane
                )
                buffer_store(
                    out_id, _rsrc(fx.Int64(fx.ptrtoint(topk_ids))), c * topk + lane
                )

    @flyc.jit
    def launch(
        x: fx.Pointer,
        w: fx.Pointer,
        bias: fx.Pointer,
        num_valid_rows: fx.Pointer,
        ws: fx.Pointer,
        topk_weights: fx.Pointer,
        topk_ids: fx.Pointer,
        m: fx.Int32,
        lda: fx.Int32,
        has_nvr: fx.Int32,
        xq: fx.Pointer,
        xs: fx.Pointer,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
    ):
        router8_kernel(
            x,
            w,
            bias,
            num_valid_rows,
            ws,
            topk_weights,
            topk_ids,
            m,
            lda,
            has_nvr,
            xq,
            xs,
        ).launch(
            grid=(CSIZE, 1, 1),
            block=(WAVES * WAVE, 1, 1),
            cluster=(CSIZE, 1, 1),
            stream=stream,
        )

    return launch


__all__ = ["build_router8_module", "ROWS"]
