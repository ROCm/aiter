# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""MiniMax-M3 NVFP4 sparse paged attention, prefill (FlyDSL, gfx950).

Same inputs as the decode kernel: ``topk_idx [Hkv, rows, 16]`` logical block ids and the
request-level ``block_table`` (no sparse block table is built), plus, per row, its
request (``row_req_id``) and its causal key count (``row_kv_lens`` = position + 1), so
ragged batches and chunked prefill (q_len < kv_len) need nothing else.

Q-outer with a token tile: one workgroup = ``TQ`` consecutive rows x the (<= 16) query
heads of one kv head. It walks the union of the pages those rows selected; each page is
fetched and dequantized once and every row that selected it runs its own QK / softmax /
PV on the shared K / V fragments (rows that did not, skip it with a wave-uniform
branch). Neighbouring rows share their init / local / tail blocks, so the union is much
smaller than TQ x 16. Every (row, page) pair is one online-softmax step, merged across
the 4 waves (each owns 32 keys of every page) through LDS at the end; no atomics, no
partial buffers, nothing written but the output.

Numerics as the decode kernel: exact e2m1 x e4m3 products in f16, f16 Q and P, fp32
scores / softmax / accumulation, bf16 output.
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
import numpy as np
import torch
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from aiter.ops.flydsl.kernels import buffer_ops
from aiter.ops.flydsl.kernels.minimax_m3_sparse_pa_nvfp4_common import (
    BLK,
    HG,
    KPW,
    LOG2E,
    NDT,
    NKS,
    NW,
    RAW,
    D,
    bf16x8_to_f16,
    dequant_page,
    exp2,
    global_f32_ptr,
    global_i32_ptr,
    maxf_nn,
    mfma_f16,
    page_loader,
    pk_bf16,
    pk_f16,
    xlane4,
)

TOPK = 16  # selected blocks per row (one lane per (row, slot): TQ x 16 <= 64 lanes)


def compile_prefill(tq: int = 4, rounds: int = 2, q_lds: bool | None = None):
    """Build the prefill kernel for ``tq`` rows per workgroup (1, 2 or 4).

    ``rounds`` splits the cross-wave combine over that many passes (tq must divide);
    2 halves the LDS so two workgroups fit a CU.
    """
    assert tq in (1, 2, 4) and tq % rounds == 0 and rounds in (1, 2)
    TQ, R = tq, rounds
    # Q^T lives in LDS (aliasing the combine buffer) when 4 rows would not fit the
    # 256 VGPRs two waves per SIMD allow.
    Q_LDS = (TQ == 4 and R == 2) if q_lds is None else q_lds
    Q_ROW = D * 2 // 16 + 1  # 16 B LDS tiles per Q^T row, padded by one
    tpr = TQ // R  # rows per combine round
    name = f"m3_nvfp4_sparse_pa_prefill_tq{TQ}_r{R}" + ("_ql" if Q_LDS else "")

    @fx.struct
    class Shared:
        o: fx.Array[fx.Int32, NW * tpr * NDT * 64 * 4, 16]
        ml: fx.Array[fx.Float32, NW * TQ * 2 * 64]

    @flyc.kernel(name=name, known_block_size=[64 * NW, 1, 1])
    def kernel(
        i32_T: fx.Int32,
        i32_H: fx.Int32,
        i32_hkv: fx.Int32,
        i32_group: fx.Int32,
        i32_topk_hs: fx.Int32,
        i32_topk_rs: fx.Int32,
        i32_bt_s: fx.Int32,
        arg_q: fx.Int64,  # [T, H, 128] bf16
        arg_cache: fx.Int64,
        arg_topk: fx.Int64,  # [Hkv, >= T, 16] int32 logical blocks, -1 = none
        arg_bt: fx.Int64,  # [reqs, stride] int32
        arg_req: fx.Int64,  # [T] int32
        arg_kv: fx.Int64,  # [T] int32 causal key count (position + 1)
        arg_out: fx.Int64,  # [T, H, 128] bf16
        arg_ks: fx.Int64,
        arg_vs: fx.Int64,
        i32_page_byte: fx.Int32,
        f32_c: fx.Float32,  # sm_scale * log2(e)
    ):
        smem = fx.SharedAllocator().allocate(Shared).peek()
        tx = fx.thread_idx.x
        bidx, kvh = fx.block_idx.x, fx.block_idx.y
        lane = tx % 64
        w = fx.Int32(fx.rocdl.readfirstlane(T.i32, tx // 64))
        n, g = lane % 16, lane // 16
        topk = global_i32_ptr(arg_topk)
        bt = global_i32_ptr(arg_bt)
        req_p = global_i32_ptr(arg_req)
        kv_p = global_i32_ptr(arg_kv)
        c = f32_c * fx.Float32(global_f32_ptr(arg_ks)[0])
        v_scale = fx.Float32(global_f32_ptr(arg_vs)[0])
        neg_inf = fx.Float32(float("-inf"))

        o_lds = fx.logical_divide(
            fx.make_view(
                fx.recast_iter(fx.Int32, smem.o.ptr),
                fx.make_layout(NW * tpr * NDT * 64 * 4, 1),
            ),
            fx.make_layout(4, 1),
        )
        lds_atom = fx.make_copy_atom(fx.UniversalCopy128b(), fx.Int32)
        # Per row (wave-uniform): the page of its own block and its causal limit.
        qr = buffer_ops.create_buffer_resource_from_addr(
            arg_q, num_records_bytes=fx.Int64(i32_T) * fx.Int64(i32_H) * (D * 2)
        )
        head = (n < i32_group).select(n, fx.Int32(0))  # masked heads read head 0
        if const_expr(Q_LDS):
            # Q^T in LDS: thread (row, head, quarter) stages 64 B as f16.
            tq_, hq, qq = tx // 64, (tx // 4) % HG, tx % 4
            tok_q = bidx * TQ + tq_
            tok_q = (tok_q < i32_T).select(tok_q, fx.Int32(0))
            hq_src = (hq < i32_group).select(hq, fx.Int32(0))
            q_src = (tok_q * i32_H + kvh * i32_group + hq_src) * (D // 2) + qq * 16
            for i in range_constexpr(4):
                r = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Int32)
                r.store(
                    bf16x8_to_f16(
                        fx.Vector(
                            buffer_ops.buffer_load(
                                qr, q_src + 4 * i, vec_width=4, dtype=fx.Int32
                            )
                        )
                    )
                )
                fx.copy(
                    lds_atom,
                    r,
                    fx.slice(o_lds, (None, (tq_ * HG + hq) * Q_ROW + qq * 4 + i)),
                )

        def q_frags(t):
            """Row t's Q^T, one B operand per k-step."""
            if const_expr(not Q_LDS):
                return q_regs[t]
            out = []
            for ks in range_constexpr(NKS):
                r = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Int32)
                fx.copy(
                    lds_atom,
                    fx.slice(o_lds, (None, (t * HG + n) * Q_ROW + g * 4 + ks)),
                    r,
                )
                out.append(Vec(r.load()))
            return out

        diag, lim, q_regs = [], [], []
        for t in range_constexpr(TQ):
            tok = bidx * TQ + t
            tc = (tok < i32_T).select(tok, fx.Int32(0))
            kv = fx.Int32(fx.rocdl.readfirstlane(T.i32, fx.Int32(kv_p[tc])))
            kv = (tok < i32_T).select(kv, fx.Int32(0))
            pos = kv - 1  # -1: nothing to attend
            req = fx.Int32(fx.rocdl.readfirstlane(T.i32, fx.Int32(req_p[tc])))
            own = (pos >= 0).select(pos // BLK, fx.Int32(0))
            diag.append(
                fx.Int32(
                    fx.rocdl.readfirstlane(T.i32, fx.Int32(bt[req * i32_bt_s + own]))
                )
            )
            lim.append(pos % BLK - w * KPW)
            if const_expr(not Q_LDS):
                q_row = (tc * i32_H + kvh * i32_group + head) * (D // 2) + g * 16
            if const_expr(not Q_LDS):
                q_regs.append(
                    [
                        bf16x8_to_f16(
                            fx.Vector(
                                buffer_ops.buffer_load(
                                    qr, q_row + ks * 4, vec_width=4, dtype=fx.Int32
                                )
                            )
                        )
                        for ks in range_constexpr(NKS)
                    ]
                )

        load_page = page_loader(arg_cache, i32_page_byte, kvh, i32_hkv, w, n, g)

        # Lane (row lane // 16, slot lane % 16): the page of the block it selected.
        lane_tok = bidx * TQ + lane // TOPK
        lane_in = (lane // TOPK < TQ) & (lane_tok < i32_T)
        lane_tc = lane_in.select(lane_tok, fx.Int32(0))
        kv_l = fx.Int32(kv_p[lane_tc])
        blk = fx.Int32(topk[kvh * i32_topk_hs + lane_tc * i32_topk_rs + lane % TOPK])
        lane_ok = lane_in & (kv_l > 0) & (blk >= 0) & (blk <= (kv_l - 1) // BLK)
        lane_page = fx.Int32(
            bt[fx.Int32(req_p[lane_tc]) * i32_bt_s + lane_ok.select(blk, fx.Int32(0))]
        )
        lane_page = lane_ok.select(lane_page, fx.Int32(-1))
        valid = fx.Int64(fx.rocdl.ballot(T.i64, lane_page >= 0))
        first = (valid != fx.Int64(0)).select(
            fx.Int32(fx.math.cttz(valid).to(fx.Int32)), fx.Int32(0)
        )

        def page_at(mask):
            """Page of the lowest lane left in ``mask``, the first page when empty."""
            idx = (mask != fx.Int64(0)).select(
                fx.Int32(fx.math.cttz(mask).to(fx.Int32)), first
            )
            pg = fx.Int32(fx.rocdl.readlane(T.i32, lane_page, idx))
            return (pg >= 0).select(pg, fx.Int32(0))

        raw0 = load_page(page_at(valid))
        fx.rocdl.sched_barrier(0)
        # Union: the first lane of each distinct page.
        n_dup = fx.Int32(0)
        for j in range_constexpr(TOPK * (TQ - 1)):
            pj = fx.Int32(fx.rocdl.readlane(T.i32, lane_page, j))
            n_dup = n_dup + ((lane_page == pj) & (lane > j)).select(
                fx.Int32(1), fx.Int32(0)
            )
        umask = fx.Int64(fx.rocdl.ballot(T.i64, (lane_page >= 0) & (n_dup == 0)))
        n_u = fx.Int32(fx.math.ctpop(umask).to(fx.Int32))

        if const_expr(Q_LDS):
            fx.gpu.barrier()  # Q^T staged
        zero4 = Vec.filled(4, 0.0, fx.Float32)
        key_off = [
            8 * g + 4 * st + e for st in range_constexpr(2) for e in range_constexpr(4)
        ]

        init = [umask]
        init += [neg_inf for _ in range_constexpr(TQ)]
        init += [fx.Float32(0.0) for _ in range_constexpr(TQ)]
        init += [zero4 for _ in range_constexpr(TQ * NDT)]
        init += raw0
        n_st = 1 + 2 * TQ + TQ * NDT
        results = init
        # Iteration u dequantizes page u and loads page u + 1.
        for u, st_ in range(  # type: ignore[call-overload]
            fx.Index(0),
            fx.Index(n_u),
            fx.Index(1),
            init=[fx.as_ir_value(x) for x in init],
        ):
            rem = fx.Int64(st_[0])
            m = [fx.Float32(st_[1 + t]) for t in range_constexpr(TQ)]
            l_run = [fx.Float32(st_[1 + TQ + t]) for t in range_constexpr(TQ)]
            o = [
                [Vec(st_[1 + 2 * TQ + t * NDT + dt]) for dt in range_constexpr(NDT)]
                for t in range_constexpr(TQ)
            ]
            raw = [fx.Int32(st_[n_st + i]) for i in range_constexpr(RAW)]
            page = page_at(rem)
            rem = rem & (rem - fx.Int64(1))
            raw_next = load_page(page_at(rem))
            kf, vf = dequant_page(raw)
            hits = fx.Int64(fx.rocdl.ballot(T.i64, lane_page == page))
            for t in range_constexpr(TQ):
                m_t, l_t, o_t = m[t], l_run[t], o[t]
                if (hits.shrui(fx.Int64(TOPK * t)) & fx.Int64(0xFFFF)) != fx.Int64(0):
                    qf = q_frags(t)
                    a = []
                    for st in range_constexpr(2):
                        acc = zero4
                        for ks in range_constexpr(NKS):
                            acc = mfma_f16(kf[st][ks], qf[ks], acc)
                        a += [acc[e] for e in range_constexpr(4)]
                    if page == diag[t]:
                        a = [
                            (key_off[i] <= lim[t]).select(a[i], neg_inf)
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
                        [pk_f16(p[2 * j], p[2 * j + 1]) for j in range_constexpr(4)],
                        fx.Int32,
                    )
                    # alpha is exactly 1 unless some head's max grew.
                    if fx.Int64(fx.rocdl.ballot(T.i64, m_new != m_t)) != fx.Int64(0):
                        av = Vec.filled(4, alpha, fx.Float32)
                        o_t = [x * av for x in o_t]
                    o_t = [mfma_f16(vf[dt], pf, o_t[dt]) for dt in range_constexpr(NDT)]
                    m_t = m_new
                m[t], l_run[t], o[t] = m_t, l_t, o_t
            state = [rem] + m + l_run + [x for tok_o in o for x in tok_o] + raw_next
            results = yield [fx.as_ir_value(x) for x in state]

        # Combine the waves' partials, tpr rows per round; wave w finalizes row w.
        ml = smem.ml.ptr
        for t in range_constexpr(TQ):
            ml[((w * TQ + t) * 2) * 64 + lane] = fx.Float32(results[1 + t])
            ml[((w * TQ + t) * 2 + 1) * 64 + lane] = fx.Float32(results[1 + TQ + t])
        outr = buffer_ops.create_buffer_resource_from_addr(
            arg_out, num_records_bytes=fx.Int64(i32_T) * fx.Int64(i32_H) * (D * 2)
        )
        for rnd in range_constexpr(R):
            if const_expr(rnd > 0 or Q_LDS):
                fx.gpu.barrier()  # Q^T / the previous round's partials are dead
            for tt in range_constexpr(tpr):
                for dt in range_constexpr(NDT):
                    r = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Int32)
                    r.store(
                        Vec(results[1 + 2 * TQ + (tpr * rnd + tt) * NDT + dt]).bitcast(
                            fx.Int32
                        )
                    )
                    fx.copy(
                        lds_atom,
                        r,
                        fx.slice(
                            o_lds, (None, ((w * tpr + tt) * NDT + dt) * 64 + lane)
                        ),
                    )
            fx.gpu.barrier()
            if w // tpr == rnd:
                t = w
                ms = [
                    fx.Float32(ml[((ww * TQ + t) * 2) * 64 + lane])
                    for ww in range_constexpr(NW)
                ]
                big = ms[0]
                for ww in range_constexpr(1, NW):
                    big = maxf_nn(big, ms[ww])
                big = (big == neg_inf).select(fx.Float32(0.0), big)
                wt = [
                    fx.Float32(fx.math.exp2(ms[ww] - big)) for ww in range_constexpr(NW)
                ]
                l_tot = fx.Float32(0.0)
                for ww in range_constexpr(NW):
                    l_w = ml[((ww * TQ + t) * 2 + 1) * 64 + lane]
                    l_tot = l_tot + fx.Float32(l_w) * wt[ww]
                l_tot = xlane4(l_tot, lambda x, y: x + y)
                inv = (l_tot > fx.Float32(0.0)).select(v_scale / l_tot, fx.Float32(0.0))
                acc = [zero4 for _ in range_constexpr(NDT)]
                for ww in range_constexpr(NW):
                    wv = Vec.filled(4, wt[ww], fx.Float32)
                    for dt in range_constexpr(NDT):
                        r = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Int32)
                        o_tile = ((ww * tpr + t - tpr * rnd) * NDT + dt) * 64 + lane
                        fx.copy(lds_atom, fx.slice(o_lds, (None, o_tile)), r)
                        acc[dt] = acc[dt] + Vec(r.load()).bitcast(fx.Float32) * wv
                # Lane (n, g) holds dims [32 g, 32 g + 32) of head n in order.
                tok = bidx * TQ + t
                o_row = (tok * i32_H + kvh * i32_group + n) * (D // 2) + g * 16
                for e in range_constexpr(4):
                    val = Vec.from_elements(
                        [
                            pk_bf16(acc[2 * j][e] * inv, acc[2 * j + 1][e] * inv)
                            for j in range_constexpr(4)
                        ],
                        fx.Int32,
                    )
                    buffer_ops.buffer_store(
                        val, outr, o_row + e * 4, mask=(tok < i32_T) & (n < i32_group)
                    )

    @flyc.jit
    def launch(
        arg_q: fx.Int64,
        arg_cache: fx.Int64,
        arg_topk: fx.Int64,
        arg_bt: fx.Int64,
        arg_req: fx.Int64,
        arg_kv: fx.Int64,
        arg_out: fx.Int64,
        arg_ks: fx.Int64,
        arg_vs: fx.Int64,
        i32_T: fx.Int32,
        i32_H: fx.Int32,
        i32_hkv: fx.Int32,
        i32_group: fx.Int32,
        i32_topk_hs: fx.Int32,
        i32_topk_rs: fx.Int32,
        i32_bt_s: fx.Int32,
        i32_page_b: fx.Int32,
        f32_c: fx.Float32,
        stream: fx.Stream,
    ):
        kernel(
            i32_T,
            i32_H,
            i32_hkv,
            i32_group,
            i32_topk_hs,
            i32_topk_rs,
            i32_bt_s,
            arg_q,
            arg_cache,
            arg_topk,
            arg_bt,
            arg_req,
            arg_kv,
            arg_out,
            arg_ks,
            arg_vs,
            i32_page_b,
            f32_c,
            value_attrs={
                "rocdl.waves_per_eu": R,
                "rocdl.flat_work_group_size": "256,256",
            },
        ).launch(
            grid=((i32_T + TQ - 1) // TQ, i32_hkv, 1),
            block=(64 * NW, 1, 1),
            stream=stream,
        )

    launch.kernel_name = name
    return launch


# ---------------------------------------------------------------------------
# host side
# ---------------------------------------------------------------------------
@functools.cache
def _get_kernel(tq: int, rounds: int):
    return compile_prefill(tq, rounds)


@functools.cache
def _unit_scale(device_index: int) -> torch.Tensor:
    return torch.ones(1, dtype=torch.float32, device=f"cuda:{device_index}")


def _run(exe, *args):
    cf = getattr(exe, "_cf", None)
    if cf is None:
        exe._cf = flyc.compile(exe, *args)
    else:
        cf(*args)


@functools.cache
def _cu_count(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).multi_processor_count


def pick_tile(rows: int, hkv: int, cus: int) -> tuple[int, int]:
    """(rows per workgroup, combine rounds) from the grid size.

    A bigger tile shares each page's dequantization across more rows but needs LDS and
    registers that cap occupancy at 2 waves per SIMD, so it only pays once the grid has
    at least two workgroups per CU (measured on MI355X, 256 CUs, Hkv 1 / 4, q_len 256 ..
    8192): fewer than one CU's worth of rows runs 1 row per workgroup, below two
    workgroups per CU at 4 rows runs 2.
    """
    if rows * hkv < 2 * cus:
        return 1, 1
    if -(-rows // 4) * hkv < 2 * cus:
        return 2, 1
    return 4, 2


def minimax_m3_sparse_pa_nvfp4_prefill(
    q: torch.Tensor,  # [rows, Hq, 128] bf16
    kv_cache: torch.Tensor,  # uint8 pages, see the _common file
    topk_idx: torch.Tensor,  # [Hkv, >= rows, 16] int32 logical blocks, -1 = none
    block_table: torch.Tensor,  # [reqs, max_blocks] int32, logical block -> page
    row_req_id: torch.Tensor,  # [rows] int32 request of each row
    row_kv_lens: torch.Tensor,  # [rows] int32 causal key count of each row (position + 1)
    sm_scale: float,
    out: torch.Tensor | None = None,  # [rows, Hq, 128] bf16
    k_scale: torch.Tensor | None = None,  # [1] fp32 global K scale (default 1)
    v_scale: torch.Tensor | None = None,  # [1] fp32 global V scale (default 1)
    tq: int | None = None,
    rounds: int | None = None,
) -> torch.Tensor:
    """Row i attends to the keys of its ``topk_idx`` blocks up to its position."""
    rows, num_heads, head_dim = q.shape
    hkv = topk_idx.shape[0]
    if out is None:
        out = torch.empty_like(q)
    if rows == 0:
        return out
    group = num_heads // hkv
    assert head_dim == D and group * hkv == num_heads and group in (8, 16)
    assert topk_idx.shape[-1] == TOPK
    assert q.dtype == torch.bfloat16 and out.dtype == torch.bfloat16
    assert q.is_contiguous() and out.is_contiguous() and out.shape == q.shape
    assert kv_cache.dtype == torch.uint8 and kv_cache.stride(0) < 2**31
    assert kv_cache.stride(0) >= 2 * hkv * (BLK * D // 2 + BLK * D // 16)
    assert topk_idx.dtype == torch.int32 and topk_idx.shape[1] >= rows
    assert topk_idx.stride(2) == 1
    for t in (block_table, row_req_id, row_kv_lens):
        assert t.dtype == torch.int32 and t.stride(-1) == 1
    assert row_req_id.shape[0] >= rows and row_kv_lens.shape[0] >= rows
    dev = q.device.index
    if tq is None:
        tq, rounds = pick_tile(rows, hkv, _cu_count(dev))
    elif rounds is None:
        rounds = 2 if tq == 4 else 1
    unit = _unit_scale(dev)
    ks = unit if k_scale is None else k_scale
    vs = unit if v_scale is None else v_scale
    c = float(np.float32(sm_scale) * np.float32(LOG2E))
    with torch.cuda.device(q.device):
        _run(
            _get_kernel(tq, rounds),
            q.data_ptr(),
            kv_cache.data_ptr(),
            topk_idx.data_ptr(),
            block_table.data_ptr(),
            row_req_id.data_ptr(),
            row_kv_lens.data_ptr(),
            out.data_ptr(),
            ks.data_ptr(),
            vs.data_ptr(),
            rows,
            num_heads,
            hkv,
            group,
            topk_idx.stride(0),
            topk_idx.stride(1),
            block_table.stride(0),
            kv_cache.stride(0),
            c,
            torch.cuda.current_stream(q.device),
        )
    return out
