# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""MoE router GEMM + biased sigmoid top-k in one gfx1250 launch (FlyDSL).

Replaces the decode pair ``tgemm.mm(x, w)`` (the FlyDSL gfx1250 A16W16 GEMM,
split-K, bf16 logits) + ``topk_gating(..., score_func="sigmoid")`` and gives
their bytes:

* Logits. The FlyDSL GEMM (``gemm_a16w16_kernel_gfx1250``) computes every
  16x16 (expert, token) block as ``split_k`` WMMA chains: chain ``s`` covers K
  ``[s*K/split_k, (s+1)*K/split_k)`` in 32-wide steps from a zero
  accumulator, router weight as WMMA operand A and activation as operand B;
  the last split to arrive sums the chains in split order (p0 + p1 + ...) and
  rounds to bf16. This kernel runs the same chains with the same operands in
  the same fragment slots, and sums them in the same order.
* Top-k. ``topk_gating_kernel_opt`` (SCORE_SIGMOID): one wave per token row,
  lane L holds experts L, L+32, L+64, L+96; score ``rcp(1 + exp2(-x*log2e))``
  (the reference is built with -fgpu-flush-denormals-to-zero, so a denormal
  score is flushed here too), selection value ``score + bias`` (NaN -> -inf),
  the same 5-comparator sorting network per lane, and ``topk`` rounds of a wave
  argmax whose ties go to the lowest lane. Weights are the unbiased scores in
  selection order, renormalized with the same sum order and floor.

Each 16-row token tile is one workgroup cluster of ``split_k * n_groups``
(<= 16) workgroups, one per (K split, expert group); a workgroup's waves each
own ``CHAINS`` 16-expert slices, run as interleaved chains on one activation
fragment. The chains' partials go to a cluster-private workspace; after the
cluster barrier, wave ``w`` of cluster member ``c`` reduces and gates the
tile's row ``c + C * w``. The hand-off is the cluster barrier (measured on
MI455X: ~0.1 us over an empty launch, where a software grid barrier costs
~2.5 us): no flags, no polling, no cross-tile dependency.

``with_counts`` also leaves, per 8-row group, how many of its routes go to
each expert (the A1 route/quant of the grouped MoE reads these instead of
re-counting every id in every block), counted after a second cluster barrier.

``quant_rows`` (1: MXFP4, 2: MXFP4 + MXFP8) also writes every token row of
``x`` quantized, token-major: MXFP4 payload ``[m, K/2]`` + E8M0 ``[m, K/32]``
(the bytes the grouped MoE's A1 quant writes per routed row) and, with 2, the
MXFP8 copy ``[m, K]`` + ``[m, K/32]`` (its ``mxfp8_copy_out``). Lane (kgrp,
lane16) already holds columns ``kgrp*8 + 0..7`` and ``16 + kgrp*8 + 0..7`` of
every 32-wide K step of its token as the WMMA B fragment, so one MX block is
two lanes: the block amax is a max over both (order-free), the e8m0 rule and
the pk8 converts are the A1 kernel's own helpers, and the two lanes swap one
half so each stores contiguous bytes. The K steps of a (tile, K split) are
spread over the ``n_groups * waves`` waves that all load them.
"""

import struct

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, range_constexpr, rocdl
from flydsl.expr.rocdl import cluster
from flydsl.expr.typing import Int32, T

from aiter.ops.flydsl.kernels import buffer_ops

EXPERTS = 128
WAVE = 32
TILE = 16
WMMA_K = 32
GROUP = 8  # rows per route-count group: the A1 route/quant block's tokens
MAX_CLUSTER = 16
LOG2E = 1.4426950408889634
RENORM_SUM_FLOOR = 1e-20
FLT_MIN = 1.1754943508222875e-38
CHAINS = 1  # 16-expert slices per wave


def router_topk_geometry(split_k: int):
    """(expert groups, experts per workgroup, waves per workgroup, cluster size)."""
    n_groups = min(EXPERTS // TILE, MAX_CLUSTER // split_k)
    while n_groups & (n_groups - 1):  # a power of two divides the 8 expert slices
        n_groups -= 1
    if n_groups < 1:
        raise ValueError(
            f"split_k {split_k} exceeds the {MAX_CLUSTER}-workgroup cluster"
        )
    tile_n = EXPERTS // n_groups
    if tile_n * n_groups != EXPERTS:
        raise ValueError(f"split_k {split_k} gives no even expert split")
    return n_groups, tile_n, tile_n // TILE, split_k * n_groups


def _rmem_vec(v, n, dtype):
    t = fx.make_rmem_tensor(n, dtype)
    t.store(v if isinstance(v, fx.Vector) else fx.Vector(v))
    return t


def build_moe_router_topk_module(
    K: int,
    split_k: int,
    topk: int = 4,
    need_renorm: bool = True,
    routed_scaling_factor: float = 1.0,
    has_bias: bool = True,
    prefetch: int = 8,
    with_counts: bool = False,
    quant_rows: int = 0,
):
    """Launcher ``(x, w, bias, num_valid_rows, ws, topk_weights, topk_ids,
    group_counts, m, lda, has_nvr, stream)``; ``ws`` holds
    ``ceil(m/16) * split_k * 16 * 128`` fp32. ``with_counts`` also writes
    ``group_counts`` int32 ``[2 * ceil(m/16), 128]``: per 8-row group, its
    routes per expert (ids outside [0, 128) and rows >= m count nowhere).
    ``quant_rows`` > 0: four more pointers before ``stream`` -- the MXFP4
    payload and E8M0 rows, then the MXFP8 payload and E8M0 rows (unread at
    ``quant_rows == 1``), see the module docstring."""
    if K % (split_k * WMMA_K):
        raise ValueError(f"K {K} is not a multiple of split_k * 32")
    if not 1 <= topk <= 4:
        raise ValueError("topk must be in [1, 4] (one slot per merge round)")
    if quant_rows not in (0, 1, 2):
        raise ValueError(f"quant_rows must be 0, 1 or 2, got {quant_rows}")
    n_groups, tile_n, slices, csize = router_topk_geometry(split_k)
    waves = slices // CHAINS
    chunk = K // split_k
    k_steps = chunk // WMMA_K
    block = waves * WAVE
    row_iters = -(-TILE // (csize * waves))
    pf = min(prefetch, k_steps)
    scale = float(routed_scaling_factor)
    scale_tag = format(struct.unpack("<I", struct.pack("<f", scale))[0], "08x")
    name = (
        f"moe_router_topk_k{K}_s{split_k}_t{topk}_r{int(need_renorm)}"
        f"_b{int(has_bias)}_pf{pf}_sc{scale_tag}{'_cnt' if with_counts else ''}"
        f"{f'_q{quant_rows}' if quant_rows else ''}"
    )
    neg_inf = float("-inf")
    # Quantized rows: the (tile, K split)'s k_steps MX blocks go to its
    # n_groups * waves waves, q_steps consecutive blocks each.
    q_workers = n_groups * waves
    q_steps = -(-k_steps // q_workers)

    def _quant_step(xf, blk, tok_row, kgrp, qrows):
        """MX block ``blk`` of token row ``tok_row`` from this lane's 16 bf16:
        MXFP4 (and MXFP8) bytes and E8M0 of the A1 kernel, see the docstring."""
        from aiter.ops.flydsl.kernels.moe_fused_route_quant_scatter import (
            _MXFP8_COPY_AMAX_FLOOR,
            _ROUND_MODE,
            _cvt_scalef32_pk8_fp4_bf16,
            _cvt_scalef32_pk8_fp8_bf16,
        )
        from aiter.ops.flydsl.kernels.quant_utils import emit_mx_e8m0_scale
        from aiter.utility.mx_types import MxDtypeInt

        xq_rsrc, xs_rsrc, x8_rsrc, x8s_rsrc = qrows
        halves = [xf.shuffle(xf, list(range(h * 8, h * 8 + 8))) for h in range(2)]
        amax = fx.Float32(0.0)
        for h in range_constexpr(2):
            f32x8 = halves[h].to(fx.Float32)
            for i in range_constexpr(8):
                amax = fx.max(amax, abs(f32x8[i]))
        amax = fx.max(amax, amax.shuffle_xor(fx.Int32(TILE), fx.Int32(WAVE)))
        is_k0 = kgrp == 0
        e8m0 = fx.Int32(
            emit_mx_e8m0_scale(amax, mode=_ROUND_MODE, dtype=MxDtypeInt.FP4_E2M1)
        )
        sc = (e8m0 << fx.Int32(23)).bitcast(fx.Float32)
        p = [
            fx.Int32(_cvt_scalef32_pk8_fp4_bf16(halves[h], sc, i32_ty=T.i32))
            for h in range(2)
        ]
        # kgrp0 owns bytes 0-3 / 8-11 of the block, kgrp1 4-7 / 12-15: swap
        # one dword so kgrp0 stores bytes 0-7 and kgrp1 bytes 8-15.
        recv = is_k0.select(p[1], p[0]).shuffle_xor(fx.Int32(TILE), fx.Int32(WAVE))
        out = fx.Vector.from_elements(
            [is_k0.select(p[0], recv), is_k0.select(recv, p[1])], fx.Int32
        )
        buffer_ops.buffer_store(
            out, xq_rsrc, tok_row * (K // 2) + blk * 16 + kgrp * 8, offset_is_bytes=True
        )
        buffer_ops.buffer_store(
            fx.Uint8(e8m0),
            xs_rsrc,
            tok_row * (K // 32) + blk,
            mask=is_k0,
            offset_is_bytes=True,
        )
        if const_expr(quant_rows == 2):
            e8 = fx.Int32(
                emit_mx_e8m0_scale(
                    fx.max(amax, fx.Float32(_MXFP8_COPY_AMAX_FLOOR)),
                    mode=_ROUND_MODE,
                    dtype=MxDtypeInt.FP8_E4M3,
                )
            )
            sc8 = (e8 << fx.Int32(23)).bitcast(fx.Float32)
            q = [
                fx.Vector(
                    _cvt_scalef32_pk8_fp8_bf16(halves[h], sc8, v2i32_ty=T.vec(2, T.i32))
                )
                for h in range(2)
            ]
            # kgrp0 owns bytes 0-7 / 16-23, kgrp1 8-15 / 24-31: kgrp0 stores
            # 0-15, kgrp1 16-31.
            send = [is_k0.select(q[1][d], q[0][d]) for d in range(2)]
            got = [v.shuffle_xor(fx.Int32(TILE), fx.Int32(WAVE)) for v in send]
            out8 = fx.Vector.from_elements(
                [is_k0.select(q[0][d], got[d]) for d in range(2)]
                + [is_k0.select(got[d], q[1][d]) for d in range(2)],
                fx.Int32,
            )
            buffer_ops.buffer_store(
                out8, x8_rsrc, tok_row * K + blk * 32 + kgrp * 16, offset_is_bytes=True
            )
            buffer_ops.buffer_store(
                fx.Uint8(e8),
                x8s_rsrc,
                tok_row * (K // 32) + blk,
                mask=is_k0,
                offset_is_bytes=True,
            )

    @flyc.kernel(name=name, known_block_size=[block, 1, 1])
    def router_topk_kernel(
        x: fx.Pointer,
        w: fx.Pointer,
        bias: fx.Pointer,
        num_valid_rows: fx.Pointer,
        ws: fx.Pointer,
        topk_weights: fx.Pointer,
        topk_ids: fx.Pointer,
        group_counts: fx.Pointer,
        m: Int32,
        lda: Int32,
        has_nvr: Int32,
        qrows=(),  # quant_rows: (xq, xs, x8, x8s); empty, so no kernel args, otherwise
    ):
        tid = fx.Int32(fx.thread_idx.x)
        bid = fx.Int32(fx.block_idx.x)
        wave = fx.Int32(rocdl.readfirstlane(T.i32, tid // WAVE))
        lane = tid % WAVE
        lane16 = lane % TILE
        kgrp = lane // TILE
        mt = bid // csize
        c = bid % csize
        s = c // n_groups
        ng = c % n_groups
        ntile0 = (ng * waves + wave) * CHAINS
        tok = fx.Int32(((mt * TILE + lane16) < m).select(mt * TILE + lane16, m - 1))

        x_rsrc = buffer_ops.create_buffer_resource_from_addr(fx.Int64(fx.ptrtoint(x)))
        w_rsrc = buffer_ops.create_buffer_resource_from_addr(fx.Int64(fx.ptrtoint(w)))
        # Dword offsets of this lane's fragment rows: K [kgrp*8, +8) and
        # [16 + kgrp*8, +8) of each 32-wide step.
        kbase = s * chunk + kgrp * 8
        x_dw = (tok * lda + kbase) // 2
        w_dws = [
            (((ntile0 + ch) * TILE + lane16) * K + kbase) // 2
            for ch in range_constexpr(CHAINS)
        ]

        def load_frag(rsrc, base, j):
            lo = buffer_ops.buffer_load(rsrc, base + j * 16, vec_width=4, dtype=T.i32)
            hi = buffer_ops.buffer_load(
                rsrc, base + j * 16 + 8, vec_width=4, dtype=T.i32
            )
            return (
                fx.Vector(lo)
                .shuffle(fx.Vector(hi), list(range(8)))
                .bitcast(fx.BFloat16)
            )

        def load_step(j):
            return [load_frag(w_rsrc, wd, j) for wd in w_dws], load_frag(
                x_rsrc, x_dw, j
            )

        q_rsrcs = None
        q_wid = None
        if const_expr(quant_rows):
            # Rows >= m (the clamped tail lanes) fall outside the descriptors.
            xq_p, xs_p, x8_p, x8s_p = qrows
            q_rsrcs = (
                buffer_ops.create_buffer_resource_from_addr(
                    fx.Int64(fx.ptrtoint(xq_p)), num_records_bytes=m * (K // 2)
                ),
                buffer_ops.create_buffer_resource_from_addr(
                    fx.Int64(fx.ptrtoint(xs_p)), num_records_bytes=m * (K // 32)
                ),
                buffer_ops.create_buffer_resource_from_addr(
                    fx.Int64(fx.ptrtoint(x8_p)), num_records_bytes=m * K
                ),
                buffer_ops.create_buffer_resource_from_addr(
                    fx.Int64(fx.ptrtoint(x8s_p)), num_records_bytes=m * (K // 32)
                ),
            )
            q_wid = ng * waves + wave
            q_tok_row = mt * TILE + lane16
            # The K loop only keeps this wave's fragments (one conditional
            # copy per step); they are quantized and stored after the cluster
            # barrier, next to the gating. Quantizing inside the unrolled loop
            # emits the convert once per K step and the code outgrows the
            # instruction cache (measured +0.85 us at split_k 4).
            q_x = [fx.make_rmem_tensor(8, fx.Int32) for _ in range(q_steps)]

        wmma = fx.make_mma_atom(fx.rocdl.WMMA(16, 16, 32, fx.BFloat16, fx.Float32))
        accs = [fx.full(8, 0.0, fx.Float32) for _ in range_constexpr(CHAINS)]
        pending = [load_step(j) for j in range_constexpr(pf)]
        for j in range_constexpr(k_steps):
            if j + pf < k_steps:
                pending.append(load_step(j + pf))
            rocdl.sched_barrier(0)
            wfs, xf = pending[j]
            x_t = _rmem_vec(xf, 16, fx.BFloat16)
            for ch in range_constexpr(CHAINS):
                acc_t = _rmem_vec(accs[ch], 8, fx.Float32)
                fx.gemm(wmma, acc_t, _rmem_vec(wfs[ch], 16, fx.BFloat16), x_t, acc_t)
                accs[ch] = acc_t.load()
            if const_expr(quant_rows):
                # Wave-uniform: this wave owns K steps [q_wid * q_steps, +q_steps).
                if q_wid == j // q_steps:
                    q_x[j % q_steps].store(xf.bitcast(fx.Int32))
            pending[j] = None
            rocdl.sched_barrier(0)

        # acc[i]: expert ntile*16 + kgrp*8 + i of token row mt*16 + lane16.
        ws_i = fx.Int64(fx.ptrtoint(ws))
        ws_rsrc = buffer_ops.create_buffer_resource_from_addr(ws_i)
        for ch in range_constexpr(CHAINS):
            ws_off = (
                ((mt * split_k + s) * TILE + lane16) * EXPERTS
                + (ntile0 + ch) * TILE
                + kgrp * 8
            )
            accv = fx.Vector(accs[ch])
            for half in range_constexpr(2):
                part = fx.Vector.from_elements(
                    [accv[half * 4 + i] for i in range_constexpr(4)], fx.Float32
                )
                buffer_ops.buffer_store(part, ws_rsrc, ws_off + half * 4)
        cnt_i = fx.Int64(fx.ptrtoint(group_counts))
        fx.memory_fence(syncscope="cluster", ordering=fx.AtomicOrdering.Release)
        cluster.cluster_barrier()
        fx.memory_fence(syncscope="cluster", ordering=fx.AtomicOrdering.Acquire)
        if const_expr(quant_rows):
            for i in range_constexpr(q_steps):
                blk_i = q_wid * q_steps + i
                if blk_i < k_steps:
                    _quant_step(
                        fx.Vector(q_x[i].load()).bitcast(fx.BFloat16),
                        s * k_steps + blk_i,
                        q_tok_row,
                        kgrp,
                        q_rsrcs,
                    )

        bias_rsrc = buffer_ops.create_buffer_resource_from_addr(
            fx.Int64(fx.ptrtoint(bias))
        )
        tw_rsrc = buffer_ops.create_buffer_resource_from_addr(
            fx.Int64(fx.ptrtoint(topk_weights))
        )
        ti_rsrc = buffer_ops.create_buffer_resource_from_addr(
            fx.Int64(fx.ptrtoint(topk_ids))
        )
        nvr_rsrc = buffer_ops.create_buffer_resource_from_addr(
            fx.Int64(fx.ptrtoint(num_valid_rows))
        )
        for rr in range_constexpr(row_iters):
            row_local = c + csize * (wave + waves * rr)
            row = mt * TILE + row_local
            if (row_local < TILE) & (row < m):
                out_w, out_id = _gate_row(
                    ws_rsrc,
                    (mt * split_k) * TILE * EXPERTS + row_local * EXPERTS,
                    bias_rsrc,
                    nvr_rsrc,
                    has_nvr,
                    row,
                    lane,
                )
                if lane < topk:
                    buffer_ops.buffer_store(out_w, tw_rsrc, row * topk + lane)
                    buffer_ops.buffer_store(out_id, ti_rsrc, row * topk + lane)

        if const_expr(with_counts):
            # Per 8-row group: how many of its routes go to each expert, for
            # the A1 route of the next launch (one 8-token block per group).
            fx.memory_fence(syncscope="cluster", ordering=fx.AtomicOrdering.Release)
            cluster.cluster_barrier()
            fx.memory_fence(syncscope="cluster", ordering=fx.AtomicOrdering.Acquire)
            if (c < 2) & (wave == 0):
                grow0 = mt * TILE + c * GROUP
                route_row = grow0 + lane // topk
                live = (lane < GROUP * topk) & (route_row < m)
                rid = buffer_ops.buffer_load(
                    ti_rsrc,
                    live.select(grow0 * topk + lane, fx.Int32(0)),
                    vec_width=1,
                    dtype=T.i32,
                )
                rid = live.select(fx.Int32(rid), fx.Int32(-1))
                cnts = [fx.Int32(0)] * 4
                for r in range_constexpr(GROUP * topk):
                    e_r = fx.Int32(rocdl.readlane(T.i32, rid, r))
                    for j in range_constexpr(4):
                        cnts[j] = cnts[j] + (e_r == lane * 4 + j).select(
                            fx.Int32(1), fx.Int32(0)
                        )
                cnt_rsrc = buffer_ops.create_buffer_resource_from_addr(cnt_i)
                buffer_ops.buffer_store(
                    fx.Vector.from_elements(cnts, fx.Int32),
                    cnt_rsrc,
                    (mt * 2 + c) * EXPERTS + lane * 4,
                )

    def _gate_row(ws_rsrc, ws_row, bias_rsrc, nvr_rsrc, has_nvr, row, lane):
        """(weight, id) of top-k slot ``lane`` (lanes < topk) of one row."""
        ept = EXPERTS // WAVE
        parts = [
            [
                fx.Float32(
                    buffer_ops.buffer_load(
                        ws_rsrc,
                        ws_row + sidx * TILE * EXPERTS + lane + WAVE * i,
                        vec_width=1,
                        dtype=T.f32,
                    )
                )
                for i in range_constexpr(ept)
            ]
            for sidx in range_constexpr(split_k)
        ]
        bias_v = [
            (
                fx.Float32(
                    buffer_ops.buffer_load(
                        bias_rsrc, lane + WAVE * i, vec_width=1, dtype=T.f32
                    )
                )
                if has_bias
                else None
            )
            for i in range_constexpr(ept)
        ]
        nvr = fx.Int32(
            buffer_ops.buffer_load(nvr_rsrc, fx.Int32(0), vec_width=1, dtype=T.i32)
        )
        vals, orig, idxs = [], [], []
        for i in range_constexpr(ept):
            total = parts[0][i]
            for sidx in range_constexpr(1, split_k):
                total = total + parts[sidx][i]
            logit = total.to(fx.BFloat16).to(fx.Float32)
            e2 = fx.exp2((-logit) * fx.Float32(LOG2E))
            score = fx.Float32(rocdl.rcp(T.f32, fx.Float32(1.0) + e2))
            # The reference flushes denormals (score >= 0 or NaN).
            score = (score < fx.Float32(FLT_MIN)).select(fx.Float32(0.0), score)
            is_nan = score != score
            orig.append(is_nan.select(fx.Float32(0.0), score))
            v = score
            if has_bias:
                v = v + bias_v[i]
            v = (v != v).select(fx.Float32(neg_inf), v)
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
            if need_renorm:
                wsum = wsum + weight
        if need_renorm:
            inv = fx.Float32(scale) / fx.maxnumf(wsum, fx.Float32(RENORM_SUM_FLOOR))
        else:
            inv = fx.Float32(scale)
        out_w = my_w * inv
        pad = (has_nvr != 0) & (row >= nvr)
        out_w = pad.select(fx.Float32(0.0), out_w)
        out_id = pad.select(fx.Int32(-1), my_id)
        return out_w, out_id

    grid_per_tile = csize

    if quant_rows:

        @flyc.jit
        def launch_q(
            x: fx.Pointer,
            w: fx.Pointer,
            bias: fx.Pointer,
            num_valid_rows: fx.Pointer,
            ws: fx.Pointer,
            topk_weights: fx.Pointer,
            topk_ids: fx.Pointer,
            group_counts: fx.Pointer,
            m: fx.Int32,
            lda: fx.Int32,
            has_nvr: fx.Int32,
            xq: fx.Pointer,
            xs: fx.Pointer,
            x8: fx.Pointer,
            x8s: fx.Pointer,
            stream: fx.Stream = fx.Stream(None),  # noqa: B008
        ):
            tiles = (m + (TILE - 1)) // TILE
            router_topk_kernel(
                x,
                w,
                bias,
                num_valid_rows,
                ws,
                topk_weights,
                topk_ids,
                group_counts,
                m,
                lda,
                has_nvr,
                (xq, xs, x8, x8s),
            ).launch(
                grid=(fx.Int64(tiles * grid_per_tile), 1, 1),
                block=(block, 1, 1),
                cluster=(csize, 1, 1),
                stream=stream,
            )

        return launch_q

    @flyc.jit
    def launch(
        x: fx.Pointer,
        w: fx.Pointer,
        bias: fx.Pointer,
        num_valid_rows: fx.Pointer,
        ws: fx.Pointer,
        topk_weights: fx.Pointer,
        topk_ids: fx.Pointer,
        group_counts: fx.Pointer,
        m: fx.Int32,
        lda: fx.Int32,
        has_nvr: fx.Int32,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
    ):
        tiles = (m + (TILE - 1)) // TILE
        router_topk_kernel(
            x,
            w,
            bias,
            num_valid_rows,
            ws,
            topk_weights,
            topk_ids,
            group_counts,
            m,
            lda,
            has_nvr,
        ).launch(
            grid=(fx.Int64(tiles * grid_per_tile), 1, 1),
            block=(block, 1, 1),
            cluster=(csize, 1, 1),
            stream=stream,
        )

    return launch


__all__ = ["build_moe_router_topk_module", "router_topk_geometry"]
