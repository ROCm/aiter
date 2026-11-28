# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""MoE router GEMM + top-k gating in one gfx1250 launch (FlyDSL): what
``tgemm.mm(x, w)`` (bf16 logits) + ``topk_gating`` compute, with the logits
summed in this kernel's own order:

* Logits: each 16x16 (expert, token) block is ``split_k`` WMMA chains over
  consecutive K chunks from a zero accumulator (weight as operand A, activation
  as B), summed in split order in fp32, rounded to bf16.
* Top-k: as ``topk_gating_kernel_opt``: lane L holds experts L + 32i, scored
  by ``score_func`` (``SCORE_FUNCS``); selection value ``score + bias`` (NaN ->
  -inf), the reference's sorting network per lane (``EXPERT_COUNTS``), ``topk``
  wave-argmax rounds with ties to the lowest lane, the same renorm.

Each 16-row tile is one cluster of ``split_k * n_groups`` (<= 16) workgroups,
one per (K split, expert group). Partials go to a cluster-private workspace;
after the cluster barrier (~0.1 us on MI455X, a grid barrier ~2.5 us), wave
``w`` of member ``c`` reduces and gates row ``c + C * w``.
"""

import struct

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import range_constexpr, rocdl
from flydsl.expr.rocdl import cluster
from flydsl.expr.typing import Int32, T

from aiter.ops.flydsl.kernels import buffer_ops

WAVE = 32
TILE = 16
WMMA_K = 32
MAX_CLUSTER = 16
LOG2E = 1.4426950408889634
RENORM_SUM_FLOOR = 1e-20
FLT_MIN = 1.1754943508222875e-38
CHAINS = 1  # 16-expert slices per wave
MAX_TOPK = WAVE  # lane k holds top-k slot k

# Experts per lane -> topk_gating's sort_network_desc for that count (descending,
# ties kept in place); an expert count is WAVE times a key.
_SORT_NETWORKS = {4: ((0, 1), (2, 3), (0, 2), (1, 3), (1, 2))}
EXPERT_COUNTS = tuple(WAVE * n for n in _SORT_NETWORKS)


def _sigmoid(logit):
    """topk_gating's SCORE_SIGMOID: ``rcp(1 + exp2(-x*log2e))``, denormals
    flushed (the reference builds with -fgpu-flush-denormals-to-zero)."""
    e2 = fx.exp2((-logit) * fx.Float32(LOG2E))
    score = fx.Float32(rocdl.rcp(T.f32, fx.Float32(1.0) + e2))
    return (score < fx.Float32(FLT_MIN)).select(fx.Float32(0.0), score)


# score_func -> score of one bf16-rounded logit. A row-wise one (softmax) would
# also need the row's max and sum in _gate_row.
_SCORES = {"sigmoid": _sigmoid}
SCORE_FUNCS = tuple(_SCORES)


def router_topk_geometry(split_k: int, experts: int = 128):
    """(expert groups, experts per workgroup, waves per workgroup, cluster size)."""
    n_groups = min(experts // TILE, MAX_CLUSTER // split_k)
    while n_groups & (n_groups - 1):  # a power of two divides the expert slices
        n_groups -= 1
    if n_groups < 1:
        raise ValueError(
            f"split_k {split_k} exceeds the {MAX_CLUSTER}-workgroup cluster"
        )
    tile_n = experts // n_groups
    return n_groups, tile_n, tile_n // TILE, split_k * n_groups


def _rsrc(ptr):
    return buffer_ops.create_buffer_resource_from_addr(fx.Int64(fx.ptrtoint(ptr)))


def _rmem_vec(v, n, dtype):
    t = fx.make_rmem_tensor(n, dtype)
    t.store(v if isinstance(v, fx.Vector) else fx.Vector(v))
    return t


def build_moe_router_topk_module(
    K: int,
    split_k: int,
    topk: int,
    need_renorm: bool = True,
    routed_scaling_factor: float = 1.0,
    has_bias: bool = True,
    prefetch: int = 8,
    score_func: str = "sigmoid",
    experts: int = 128,
):
    """Launcher ``(x, w, bias, num_valid_rows, ws, topk_weights, topk_ids, m,
    lda, has_nvr, stream)``; ``ws`` holds ``ceil(m/16) * split_k * 16 *
    experts`` fp32."""
    if score_func not in SCORE_FUNCS:
        raise ValueError(f"score_func {score_func!r} is not one of {SCORE_FUNCS}")
    if experts not in EXPERT_COUNTS:
        raise ValueError(f"{experts} experts is not one of {EXPERT_COUNTS}")
    if not 1 <= topk <= MAX_TOPK:
        raise ValueError(f"topk {topk} is not in [1, {MAX_TOPK}]")
    if K % (split_k * WMMA_K):
        raise ValueError(f"K {K} is not a multiple of split_k * 32")
    score_fn = _SCORES[score_func]
    network = _SORT_NETWORKS[experts // WAVE]
    n_groups, _, slices, csize = router_topk_geometry(split_k, experts)
    waves = slices // CHAINS
    chunk = K // split_k
    k_steps = chunk // WMMA_K
    block = waves * WAVE
    row_iters = -(-TILE // (csize * waves))
    pf = min(prefetch, k_steps)
    scale = float(routed_scaling_factor)
    scale_tag = format(struct.unpack("<I", struct.pack("<f", scale))[0], "08x")
    name = (
        f"moe_router_topk_{score_func}_e{experts}_k{K}_s{split_k}_t{topk}"
        f"_r{int(need_renorm)}_b{int(has_bias)}_pf{pf}_sc{scale_tag}"
    )
    neg_inf = float("-inf")

    @flyc.kernel(name=name, known_block_size=[block, 1, 1])
    def router_topk_kernel(
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

        x_rsrc, w_rsrc = _rsrc(x), _rsrc(w)
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
            pending[j] = None
            rocdl.sched_barrier(0)

        # acc[i]: expert ntile*16 + kgrp*8 + i of token row mt*16 + lane16.
        ws_rsrc = _rsrc(ws)
        for ch in range_constexpr(CHAINS):
            ws_off = (
                ((mt * split_k + s) * TILE + lane16) * experts
                + (ntile0 + ch) * TILE
                + kgrp * 8
            )
            accv = fx.Vector(accs[ch])
            for half in range_constexpr(2):
                part = fx.Vector.from_elements(
                    [accv[half * 4 + i] for i in range_constexpr(4)], fx.Float32
                )
                buffer_ops.buffer_store(part, ws_rsrc, ws_off + half * 4)
        fx.memory_fence(syncscope="cluster", ordering=fx.AtomicOrdering.Release)
        cluster.cluster_barrier()
        fx.memory_fence(syncscope="cluster", ordering=fx.AtomicOrdering.Acquire)

        bias_rsrc, nvr_rsrc = _rsrc(bias), _rsrc(num_valid_rows)
        tw_rsrc, ti_rsrc = _rsrc(topk_weights), _rsrc(topk_ids)
        for rr in range_constexpr(row_iters):
            row_local = c + csize * (wave + waves * rr)
            row = mt * TILE + row_local
            if (row_local < TILE) & (row < m):
                out_w, out_id = _gate_row(
                    ws_rsrc,
                    (mt * split_k) * TILE * experts + row_local * experts,
                    bias_rsrc,
                    nvr_rsrc,
                    has_nvr,
                    row,
                    lane,
                )
                if lane < topk:
                    buffer_ops.buffer_store(out_w, tw_rsrc, row * topk + lane)
                    buffer_ops.buffer_store(out_id, ti_rsrc, row * topk + lane)

    def _gate_row(ws_rsrc, ws_row, bias_rsrc, nvr_rsrc, has_nvr, row, lane):
        """(weight, id) of top-k slot ``lane`` (lanes < topk) of one row."""
        ept = experts // WAVE
        parts = [
            [
                fx.Float32(
                    buffer_ops.buffer_load(
                        ws_rsrc,
                        ws_row + sidx * TILE * experts + lane + WAVE * i,
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
            score = score_fn(total.to(fx.BFloat16).to(fx.Float32))
            is_nan = score != score  # noqa: PLR0124 -- traced value: the NaN test
            orig.append(is_nan.select(fx.Float32(0.0), score))
            v = score
            if has_bias:
                v = v + bias_v[i]
            v = (v != v).select(fx.Float32(neg_inf), v)  # noqa: PLR0124
            vals.append(v)
            idxs.append(lane + WAVE * i)

        def cas(a, b):
            sw = vals[a] < vals[b]
            vals[a], vals[b] = sw.select(vals[b], vals[a]), sw.select(vals[a], vals[b])
            orig[a], orig[b] = sw.select(orig[b], orig[a]), sw.select(orig[a], orig[b])
            idxs[a], idxs[b] = sw.select(idxs[b], idxs[a]), sw.select(idxs[a], idxs[b])

        for a, b in network:
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


__all__ = [
    "EXPERT_COUNTS",
    "MAX_TOPK",
    "SCORE_FUNCS",
    "build_moe_router_topk_module",
    "router_topk_geometry",
]
