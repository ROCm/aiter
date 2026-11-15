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

Every slot of a row is one attended block, as in the decode kernel: a block id repeated
in a row (a selector never emits one) counts once per slot, and so does a table that maps
several logical blocks of a request to one page. The union shares the page loads
(deduplicated over every (row, slot) lane of the tile, so the result does not depend on
the tile size); each row weights a page by the number of its slots that selected it, and
a row's own block keeps its causal limit even when another slot of the row reaches the
same page.
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.ops.flydsl.kernels.minimax_m3_index_score import _cu_count
from aiter.ops.flydsl.kernels.sparse_pa_nvfp4_common import (
    BLK,
    HG,
    KPW,
    NDT,
    NKS,
    NW,
    OOB_DWORD,
    RAW,
    D,
    bf16x8_to_f16,
    check_operands,
    dequant_page,
    exp2,
    global_f32_ptr,
    global_i32_ptr,
    host_scales,
    maxf_nn,
    mfma_f16,
    page_loader,
    pk_bf16,
    pk_f16,
    require,
    scales_up_to,
    xlane4,
)
from aiter.ops.flydsl.kernels.tensor_shim import (
    _run_compiled,
    buf_copy_load,
    buf_copy_store,
    ptr_buf_tensor,
)

TOPK = 16  # selected blocks per row (one lane per (row, slot): TQ x 16 <= 64 lanes)


def compile_prefill(tq: int = 4, rounds: int = 2):
    """Build the prefill kernel for ``tq`` rows per workgroup (1, 2 or 4).

    ``rounds`` splits the cross-wave combine over that many passes (tq must divide);
    2 halves the LDS so two workgroups fit a CU.
    """
    require(tq in (1, 2, 4), f"tq must be 1, 2 or 4, got {tq}")
    require(
        rounds in (1, 2) and tq % rounds == 0,
        f"rounds must be 1 or 2 and divide tq, got {rounds}",
    )
    TQ, R = tq, rounds
    # Q^T lives in LDS (aliasing the combine buffer) when 4 rows would not fit the
    # 256 VGPRs two waves per SIMD allow.
    Q_LDS = TQ == 4 and R == 2
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
        # Raw rocdl ops for the wave-uniform / SGPR primitives (readfirstlane here, ballot,
        # readlane, sched_barrier): FlyDSL 0.5.0 has no typed spelling, as in the decode kernel.
        w = fx.Int32(fx.rocdl.readfirstlane(T.i32, tx // 64))
        n, g = lane % 16, lane // 16
        topk = global_i32_ptr(arg_topk)
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
        # Per row (wave-uniform): its causal limit inside its own block.
        qr = ptr_buf_tensor(
            arg_q,
            fx.Int32,
            unit_elems=4,
            unit_stride=1,
            num_records_bytes=i32_T * i32_H * (D * 2),
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
                r.store(bf16x8_to_f16(buf_copy_load(qr, q_src + 4 * i, unit_elems=4)))
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

        lim, q_regs = [], []
        for t in range_constexpr(TQ):
            tok = bidx * TQ + t
            tc = (tok < i32_T).select(tok, fx.Int32(0))
            kv = fx.Int32(fx.rocdl.readfirstlane(T.i32, fx.Int32(kv_p[tc])))
            kv = (tok < i32_T).select(kv, fx.Int32(0))
            pos = kv - 1  # -1: nothing to attend
            lim.append(pos % BLK - w * KPW)
            if const_expr(not Q_LDS):
                q_row = (tc * i32_H + kvh * i32_group + head) * (D // 2) + g * 16
                q_regs.append(
                    [
                        bf16x8_to_f16(buf_copy_load(qr, q_row + ks * 4, unit_elems=4))
                        for ks in range_constexpr(NKS)
                    ]
                )

        load_page = page_loader(arg_cache, i32_page_byte, kvh, i32_hkv, w, n, g)

        # Lane (row lane // 16, slot lane % 16): the block it selected, as the key
        # 2 * page + (1 if it is the row's own, partially visible, block). One load of a
        # page serves every row that selected it; a page reached through two logical
        # blocks of one row (a table that maps both to it) is two keys, so each block keeps
        # its own causal limit.
        lane_tok = bidx * TQ + lane // TOPK
        lane_in = (lane // TOPK < TQ) & (lane_tok < i32_T)
        lane_tc = lane_in.select(lane_tok, fx.Int32(0))
        kv_l = fx.Int32(kv_p[lane_tc])
        blk = fx.Int32(topk[kvh * i32_topk_hs + lane_tc * i32_topk_rs + lane % TOPK])
        own_l = (kv_l - 1) // BLK
        lane_ok = lane_in & (kv_l > 0) & (blk >= 0) & (blk <= own_l)
        # A slot that is not valid does not read the table (it may have no columns): its
        # lane loads the row's request id instead, which is discarded.
        bt_at = (
            arg_bt
            + fx.Int64(
                fx.Int32(req_p[lane_tc]) * i32_bt_s + lane_ok.select(blk, fx.Int32(0))
            )
            * 4
        )
        lane_page = fx.Int32(global_i32_ptr(lane_ok.select(bt_at, arg_req))[0])
        lane_key = lane_ok.select(
            lane_page * 2 + (blk == own_l).select(fx.Int32(1), fx.Int32(0)),
            fx.Int32(-1),
        )
        valid = fx.Int64(fx.rocdl.ballot(T.i64, lane_key >= 0))
        first = (valid != fx.Int64(0)).select(
            fx.Int32(fx.math.cttz(valid).to(fx.Int32)), fx.Int32(0)
        )

        def key_at(mask):
            """Key of the lowest lane left in ``mask``, the first lane's when empty."""
            idx = (mask != fx.Int64(0)).select(
                fx.Int32(fx.math.cttz(mask).to(fx.Int32)), first
            )
            key = fx.Int32(fx.rocdl.readlane(T.i32, lane_key, idx))
            return (key >= 0).select(key, fx.Int32(0))

        raw0 = load_page(key_at(valid) // 2, valid != fx.Int64(0))
        fx.rocdl.sched_barrier(0)
        # Union: the first lane of each distinct key, over all TQ x TOPK lanes (the pairs
        # of the last row included; with TQ = 1 this is the only dedup there is).
        n_dup = fx.Int32(0)
        for j in range_constexpr(TOPK * TQ - 1):
            kj = fx.Int32(fx.rocdl.readlane(T.i32, lane_key, j))
            n_dup = n_dup + ((lane_key == kj) & (lane > j)).select(
                fx.Int32(1), fx.Int32(0)
            )
        umask = fx.Int64(fx.rocdl.ballot(T.i64, (lane_key >= 0) & (n_dup == 0)))
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
            key = key_at(rem)
            rem = rem & (rem - fx.Int64(1))
            raw_next = load_page(key_at(rem) // 2)
            hits = fx.Int64(fx.rocdl.ballot(T.i64, lane_key == key))
            # An own-block key: the rows hitting it see only the keys up to their limit.
            own_blk = (key & 1) != 0
            v_scl = [raw[17], raw[18]]
            if own_blk:
                # Masking P is not enough: the keys past every hitting row's position may be
                # unwritten and hold a NaN V scale (e4m3 0x7f), and 0 * NaN is NaN in the PV
                # MFMA. Zero the scale bytes of those keys.
                lim_hit = fx.Int32(-1 << 20)
                for t in range_constexpr(TQ):
                    row_hit = (
                        hits.shrui(fx.Int64(TOPK * t)) & fx.Int64(0xFFFF)
                    ) != fx.Int64(0)
                    lim_hit = (row_hit & (lim[t] > lim_hit)).select(lim[t], lim_hit)
                v_scl = [
                    scales_up_to(raw[17 + h], h, g, lim_hit) for h in range_constexpr(2)
                ]
            kf, vf = dequant_page(raw[:17] + v_scl)
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
                    if own_blk:
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
                    # A page reached through n slots of this row (a table that maps n
                    # logical blocks to it) counts n times, as in the decode kernel.
                    n_slots = fx.Int32(
                        fx.math.ctpop(
                            hits.shrui(fx.Int64(TOPK * t)) & fx.Int64(0xFFFF)
                        ).to(fx.Int32)
                    )
                    if n_slots > 1:
                        p = [x * n_slots.to(fx.Float32) for x in p]
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
        outr = ptr_buf_tensor(
            arg_out,
            fx.Int32,
            unit_elems=4,
            unit_stride=1,
            num_records_bytes=i32_T * i32_H * (D * 2),
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
                    # Masked lanes store past the buffer end, which drops the write.
                    keep = (tok < i32_T) & (n < i32_group)
                    buf_copy_store(
                        outr,
                        keep.select(o_row + e * 4, fx.Int32(OOB_DWORD)),
                        val,
                        unit_elems=4,
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
def _get_kernel(tq: int, rounds: int, device_index: int):
    # device_index only keys the cache: the launcher memoises its compiled handle, which
    # is bound to the device it was compiled on.
    return compile_prefill(tq, rounds)


def pick_tile(rows: int, hkv: int, cus: int) -> tuple[int, int]:
    """(rows per workgroup, combine rounds) from the grid size.

    A bigger tile shares each page's dequantization across more rows but needs LDS and
    registers that cap occupancy at 2 waves per SIMD, so it only pays once the grid has
    enough workgroups (measured on MI355X, 256 CUs, Hkv 1 / 4, q_len 256 .. 8192): with
    fewer than 2 workgroups per CU at 1 row per workgroup it runs 1 row; with fewer than 2
    per CU at 4 rows per workgroup it runs 2 rows; otherwise 4 rows, 2 combine rounds.
    """
    if rows * hkv < 2 * cus:
        return 1, 1
    if -(-rows // 4) * hkv < 2 * cus:
        return 2, 1
    return 4, 2


def sparse_pa_nvfp4_prefill(
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
    """Row i attends to the keys of its ``topk_idx`` blocks up to its position.

    gfx950 only. Every operand must be on ``q``'s device; ``k_scale`` / ``v_scale`` are
    one positive float32 element each (a device value, not checked) and ``sm_scale``
    must be finite and positive. ``row_req_id`` values index ``block_table`` rows (a
    device value, not checked). Block ids within a row should be distinct; a repeated id
    counts once per slot, as in decode (see the module docstring). ``rounds`` needs ``tq``.
    """
    if q.device.type != "cuda":
        raise ValueError(f"q must be a CUDA tensor, got {q.device}")
    arch = get_gfx_runtime()
    if arch != "gfx950":
        raise ValueError(f"NVFP4 sparse PA prefill requires gfx950, got {arch}")
    require(q.dim() == 3 and topk_idx.dim() == 3, "q and topk_idx must be 3-D")
    rows, num_heads = q.shape[:2]
    hkv = topk_idx.shape[0]
    if out is None:
        out = torch.empty_like(q)
    if rows == 0:
        return out
    group = check_operands(q, out, kv_cache, topk_idx, block_table)
    require(topk_idx.shape[-1] == TOPK, f"topk_idx must have {TOPK} slots per row")
    require(block_table.shape[0] >= 1, "block_table needs at least one request row")
    for t in (row_req_id, row_kv_lens):
        require(
            t.dtype == torch.int32 and t.dim() == 1 and t.stride(-1) == 1,
            "row_req_id and row_kv_lens must be 1-D int32",
        )
        require(t.shape[0] >= rows, "row_req_id and row_kv_lens need a value per row")
        require(t.device == q.device, "all operands must be on q's device")
    dev = q.device.index
    if tq is None:
        require(rounds is None, "rounds needs tq")
        tq, rounds = pick_tile(rows, hkv, _cu_count(dev))
    elif rounds is None:
        rounds = 2 if tq == 4 else 1
    ks, vs, c = host_scales(q, k_scale, v_scale, sm_scale)
    with torch.cuda.device(q.device):
        _run_compiled(
            _get_kernel(tq, rounds, dev),
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
