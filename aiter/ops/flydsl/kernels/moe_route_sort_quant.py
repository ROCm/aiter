# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""MoE routing for DeepSeek-V4.1-Flash TP4 verify steps (257-2048 rows) on gfx950, in two launches and without a
grid barrier.

Same outputs as the unfused chain (topk_gating, the shared-expert append, moe_sorting, the MXFP8 quant and the
sorted e8m0 scales fused_moe stage 1 reads); the fused_moe output is bitwise equal.
One wave per token, 8 waves per workgroup, ceil(M / 8) workgroups in each launch:

  launch 1: top-6 of the 384 routed experts (sqrt-softplus + correction bias, topk_gating's order and ties), renorm x
     routed scaling factor, shared expert 384 in slot 7; rows >= n_valid (CUDA-graph padding) get ids -1, weight 0;
     per-expert count atomics whose old value is the route's slot (kept in `slots`); MXFP8 quant of x into a1 and
     the token's 160 e8m0 bytes into `tsc` (token order).
  launch 2: every workgroup reads the 385 counts, builds the padded (block_m) exclusive prefix in LDS, writes its
     owned experts' sorted_expert_ids blocks and pad rows (workgroup 0: num_valid_ids), then for its tokens the sorted
     ids (slot << 24 | token), sorted weights and the e8m0 bytes in the shuffled sorted layout stage 1 reads. A ticket
     atomic lets the last workgroup that has read the counts zero them for the next call.
Rows within one expert come out in atomic order, not token order; stage 1 and stage 2 compute each row on its own.
Pad rows' scale bytes are not written (stage 1 computes pad rows and drops them; a1_scale is zeroed once).
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import arith, gpu, ptrtoint, range_constexpr, rocdl
from flydsl.expr.typing import T

from aiter.ops.flydsl.kernels import buffer_ops
from aiter.ops.flydsl.kernels.kernels_common import create_llvm_ptr
from aiter.ops.flydsl.kernels.quant_utils import emit_mx_e8m0_scale
from aiter.ops.flydsl.kernels.tensor_shim import _to_raw as _raw
from aiter.ops.flydsl.kernels.tensor_shim import ptr_buf_tensor
from aiter.utility.mx_types import MxDtypeInt, MxScaleRoundModeInt

WAVE = 64
WAVES = 8
BLOCK = WAVE * WAVES
E_ROUTED = 384
E_ALL = 385
TOPK_R = 6
TOPK = 7
EPT = E_ROUTED // WAVE  # 6 experts per lane, expert = lane + 64 * i
DIM = 5120
GROUPS = DIM // 32  # 160 e8m0 scales per row
LANE_ELEMS = 8  # bf16 per lane per quant step (one dwordx4)
QUANT_ITERS = DIM // (WAVE * LANE_ELEMS)  # 10
BLOCK_M = 32
MAX_TOKENS = 2048
# workspace (int32, zeroed once at allocation): ws[0:384] per-expert counts, ws[WS_DONE] launch 2's ticket. Launch 2's
# last workgroup to read the counts (ticket G - 1) zeroes them for the next call: no wait, no co-residency needed.
WS_DONE, WS_INTS = 448, 512
LOG2E, LN2 = 1.4426950408889634, 0.6931471805599453
FLT_MAX = 3.4028234663852886e38

# 12-comparator 6-sort network of topk_gating_kernel_opt (sort_network_desc<6>)
_NET6 = (
    (0, 1),
    (2, 3),
    (4, 5),
    (0, 2),
    (1, 4),
    (3, 5),
    (0, 1),
    (2, 3),
    (4, 5),
    (1, 2),
    (3, 4),
    (2, 3),
)


@fx.struct
class _Lds:
    start: fx.Array[
        fx.Int32, 512, 16
    ]  # padded exclusive prefix per expert (row of its first block)
    cnt: fx.Array[fx.Int32, 512, 16]  # rows per expert (384: n_valid)
    misc: fx.Array[fx.Int32, 16, 16]  # [0] padded total, [1] last-workgroup flag


def workers(m_tokens: int, waves: int = WAVES) -> int:
    return (m_tokens + waves - 1) // waves


def _helpers(i32):
    def _elem_ptr(tt, idx, nbytes=4):
        p = create_llvm_ptr(fx.Int64(ptrtoint(tt)) + fx.Int64(idx) * nbytes)
        return p._value if hasattr(p, "_value") else p

    def _atomic_add(tt, idx, val):
        return fx.Int32(
            llvm.AtomicRMWOp(
                llvm.AtomicBinOp.add,
                _elem_ptr(tt, idx),
                _raw(fx.Int32(val)),
                llvm.AtomicOrdering.monotonic,
                syncscope="agent",
                alignment=4,
            ).result
        )

    return _atomic_add


def build_route_p1(waves: int = WAVES):
    """Launch 1: top-k, ids / weights, count atomics (slots), MXFP8 a1 and token-order e8m0 bytes."""
    WAVES_ = waves
    BLOCK_ = WAVE * waves
    name = f"moe_route_sort_quant_p1_e385_k7_d{DIM}_w{waves}"

    @flyc.kernel(name=name, known_block_size=[BLOCK_, 1, 1])
    def p1_kernel(
        logits: fx.Pointer,  # f32 [M, 384]
        bias: fx.Pointer,  # f32 [384]
        x: fx.Pointer,  # bf16 [M, 5120]
        topk_ids: fx.Pointer,  # i32 [M, 7] out
        topk_w: fx.Pointer,  # f32 [M, 7] out
        a1: fx.Pointer,  # u8 (fp8 e4m3) [M, 5120] out
        slots: fx.Pointer,  # i32 [M, 7] out: each route's row inside its expert
        tsc: fx.Pointer,  # u8 [M, 160] out: the token's e8m0 bytes
        ws: fx.Pointer,  # i32 [WS_INTS]: counts at 0..383 (zero on entry)
        n_valid: fx.Pointer,  # i32 [1]
        m_tokens: fx.Int32,
        rsf: fx.Float32,
        shared_w: fx.Float32,
    ):
        i32 = T.i32
        c0 = fx.Int32(0)
        c1 = fx.Int32(1)
        tid = fx.Int32(fx.thread_idx.x)
        wg = fx.Int32(fx.block_idx.x)
        wave = tid // fx.Int32(WAVE)
        lane = tid - wave * fx.Int32(WAVE)
        M = fx.Int32(m_tokens)
        t = wg * fx.Int32(WAVES_) + wave
        active = t < M
        tc = active.select(t, M - c1)
        NV = fx.min(fx.Int32(ptr_buf_tensor(n_valid, fx.Int32)[c0]), M)
        routed = t < NV
        _atomic_add = _helpers(i32)

        def _readlane_f32(v, ln):
            return fx.Int32(
                rocdl.readlane(i32, _raw(v.bitcast(fx.Int32)), _raw(ln))
            ).bitcast(fx.Float32)

        logits_t = ptr_buf_tensor(logits, fx.Float32)
        bias_t = ptr_buf_tensor(bias, fx.Float32)
        ids_t = ptr_buf_tensor(topk_ids, fx.Int32)
        w_t = ptr_buf_tensor(topk_w, fx.Float32)
        slots_t = ptr_buf_tensor(slots, fx.Int32)
        tsc_t = ptr_buf_tensor(tsc, fx.Int8)
        x_rsrc = buffer_ops.create_buffer_resource_from_addr(
            fx.Int64(ptrtoint(x)) + fx.Int64(tc) * (DIM * 2), num_records_bytes=DIM * 2
        )
        a1_rsrc = buffer_ops.create_buffer_resource_from_addr(
            fx.Int64(ptrtoint(a1)) + fx.Int64(tc) * DIM, num_records_bytes=DIM
        )

        def score_of(v):
            v = (v > fx.Float32(FLT_MAX)).select(fx.Float32(FLT_MAX), v)
            sp = fx.math.log2(
                fx.Float32(1.0) + fx.math.exp2(v * fx.Float32(LOG2E))
            ) * fx.Float32(LN2)
            sp = (v > fx.Float32(20.0)).select(v, sp)
            return fx.math.sqrt(sp)

        xdw = [
            buffer_ops.buffer_load(
                x_rsrc,
                _raw(
                    (fx.Int32(it * WAVE * LANE_ELEMS) + lane * fx.Int32(LANE_ELEMS))
                    >> c1
                ),
                vec_width=4,
                dtype=i32,
            )
            for it in range_constexpr(QUANT_ITERS)
        ]
        vals, origs, idxs = [], [], []
        for i in range_constexpr(EPT):
            e = lane + fx.Int32(WAVE * i)
            s = score_of(fx.Float32(logits_t[tc * fx.Int32(E_ROUTED) + e]))
            # `s == s` is false only for NaN: a NaN score becomes 0.
            origs.append((s == s).select(s, fx.Float32(0.0)))  # noqa: PLR0124
            vals.append(
                fx.maxnumf(s + fx.Float32(bias_t[e]), fx.Float32(-float("inf")))
            )
            idxs.append(e)
        for a, b in _NET6:
            swap = vals[a] < vals[b]
            vals[a], vals[b] = swap.select(vals[b], vals[a]), swap.select(
                vals[a], vals[b]
            )
            origs[a], origs[b] = swap.select(origs[b], origs[a]), swap.select(
                origs[a], origs[b]
            )
            idxs[a], idxs[b] = swap.select(idxs[b], idxs[a]), swap.select(
                idxs[a], idxs[b]
            )
        cursor = fx.Int32(0)
        wsum = fx.Float32(0.0)
        my_id = fx.Int32(0)
        my_w = fx.Float32(0.0)
        for k in range_constexpr(TOPK_R):
            cv = fx.Float32(-float("inf"))
            ci = fx.Int32(0)
            co = fx.Float32(0.0)
            for j in range_constexpr(EPT):
                hit = cursor == fx.Int32(j)
                cv = hit.select(vals[j], cv)
                ci = hit.select(idxs[j], ci)
                co = hit.select(origs[j], co)
            mx = cv
            for d in (32, 16, 8, 4, 2, 1):
                mx = fx.maxnumf(
                    mx, fx.Float32(gpu.shuffle_xor(mx, fx.Int32(d), fx.Int32(WAVE)))
                )
            mask = fx.Int64(rocdl.ballot(T.i64, _raw(cv == mx)))
            win_lane = fx.Int32(arith.trunci(i32, _raw(fx.math.cttz(mask))))
            win_lane = (mask == fx.Int64(0)).select(c0, win_lane)
            win_id = fx.Int32(rocdl.readlane(i32, _raw(ci), _raw(win_lane)))
            i_won = (cursor < fx.Int32(EPT)) & (ci == win_id)
            my_orig = i_won.select(co, fx.Float32(0.0))
            cursor = i_won.select(cursor + c1, cursor)
            weight = _readlane_f32(my_orig, win_id & fx.Int32(WAVE - 1))
            is_k = lane == fx.Int32(k)
            my_id = is_k.select(win_id, my_id)
            my_w = is_k.select(weight, my_w)
            wsum = wsum + weight
        my_w = my_w * (fx.Float32(rsf) / fx.maxnumf(wsum, fx.Float32(1e-20)))
        my_id = (lane == fx.Int32(TOPK_R)).select(fx.Int32(E_ROUTED), my_id)
        my_w = (lane == fx.Int32(TOPK_R)).select(fx.Float32(shared_w), my_w)
        out_id = routed.select(my_id, fx.Int32(-1))
        out_w = routed.select(my_w, fx.Float32(0.0))
        if active & (lane < fx.Int32(TOPK)):
            ids_t[t * fx.Int32(TOPK) + lane] = out_id
            w_t[t * fx.Int32(TOPK) + lane] = out_w
        slot = t
        if routed & (lane < fx.Int32(TOPK_R)):
            slot = _atomic_add(ws, my_id, c1)
        if active & (lane < fx.Int32(TOPK)):
            slots_t[t * fx.Int32(TOPK) + lane] = slot
        for it in range_constexpr(QUANT_ITERS):
            col = fx.Int32(it * WAVE * LANE_ELEMS) + lane * fx.Int32(LANE_ELEMS)
            bf = fx.Vector(xdw[it]).bitcast(fx.Numeric.from_ir_type(T.bf16))
            xs = [
                fx.Float32(fx.Vector(bf)[j].to(fx.Float32))
                for j in range_constexpr(LANE_ELEMS)
            ]
            amax = fx.Float32(0.0)
            for j in range_constexpr(LANE_ELEMS):
                amax = fx.maxnumf(amax, abs(xs[j]))
            for d in (1, 2):
                amax = fx.maxnumf(
                    amax, fx.Float32(gpu.shuffle_xor(amax, fx.Int32(d), fx.Int32(WAVE)))
                )
            amax = fx.minnumf(amax, fx.Float32(FLT_MAX))
            e8 = fx.Int32(
                emit_mx_e8m0_scale(
                    _raw(amax),
                    mode=MxScaleRoundModeInt.RoundUp,
                    dtype=MxDtypeInt.FP8_E4M3,
                )
            )
            rcp = ((fx.Int32(254) - e8) << fx.Int32(23)).bitcast(fx.Float32)
            q = [_raw(xs[j] * rcp) for j in range_constexpr(LANE_ELEMS)]
            w0 = rocdl.cvt_pk_fp8_f32(i32, q[0], q[1], _raw(c0), 0)
            w0 = rocdl.cvt_pk_fp8_f32(i32, q[2], q[3], w0, 1)
            w1 = rocdl.cvt_pk_fp8_f32(i32, q[4], q[5], _raw(c0), 0)
            w1 = rocdl.cvt_pk_fp8_f32(i32, q[6], q[7], w1, 1)
            pair = fx.Vector.from_elements(
                [fx.Int32(w0), fx.Int32(w1)], fx.Numeric.from_ir_type(i32)
            )
            if active:
                buffer_ops.buffer_store(_raw(pair), a1_rsrc, _raw(col >> fx.Int32(2)))
            # lane l (l % 4 == 0) holds the e8m0 of group it * 16 + l // 4
            if active & ((lane & fx.Int32(3)) == c0):
                tsc_t[
                    t * fx.Int32(GROUPS)
                    + fx.Int32(it * (WAVE // 4))
                    + (lane >> fx.Int32(2))
                ] = fx.Int8(arith.trunci(T.i8, _raw(e8)))

    @flyc.jit
    def launch(
        logits: fx.Pointer,
        bias: fx.Pointer,
        x: fx.Pointer,
        topk_ids: fx.Pointer,
        topk_w: fx.Pointer,
        a1: fx.Pointer,
        slots: fx.Pointer,
        tsc: fx.Pointer,
        ws: fx.Pointer,
        n_valid: fx.Pointer,
        m_tokens: fx.Int32,
        rsf: fx.Float32,
        shared_w: fx.Float32,
        n_workers: fx.Int32,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
    ):
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            pass
        p1_kernel(
            logits,
            bias,
            x,
            topk_ids,
            topk_w,
            a1,
            slots,
            tsc,
            ws,
            n_valid,
            m_tokens,
            rsf,
            shared_w,
        ).launch(
            grid=(arith.index_cast(T.index, _raw(fx.Int32(n_workers))), 1, 1),
            block=(BLOCK_, 1, 1),
            stream=stream,
        )

    return launch


def build_route_p2(block_m: int = BLOCK_M, waves: int = WAVES):
    """Launch 2: prefix, block ids, pad rows, sorted ids / weights, sorted e8m0 bytes; the last reader zeroes counts."""
    WAVES_ = waves
    BLOCK_ = WAVE * waves
    assert block_m in (32, 64)
    BM, BM_SHIFT = block_m, block_m.bit_length() - 1
    name = f"moe_route_sort_quant_p2_e385_k7_d{DIM}_bm{block_m}_w{waves}"

    @flyc.kernel(name=name, known_block_size=[BLOCK_, 1, 1])
    def p2_kernel(
        topk_ids: fx.Pointer,  # i32 [M, 7] (launch 1)
        topk_w: fx.Pointer,  # f32 [M, 7] (launch 1)
        slots: fx.Pointer,  # i32 [M, 7] (launch 1)
        tsc: fx.Pointer,  # u8 [M, 160] (launch 1)
        sorted_ids: fx.Pointer,  # i32 [S] out
        sorted_w: fx.Pointer,  # f32 [S] out
        sorted_eids: fx.Pointer,  # i32 [B] out
        num_valid: fx.Pointer,  # i32 [2] out
        a1_scale: fx.Pointer,  # u8 (e8m0) [S, 160] out, shuffled sorted layout
        ws: fx.Pointer,  # i32 [WS_INTS]
        n_valid: fx.Pointer,  # i32 [1]
        m_tokens: fx.Int32,
        n_workers: fx.Int32,
    ):
        i32 = T.i32
        c0 = fx.Int32(0)
        c1 = fx.Int32(1)
        tid = fx.Int32(fx.thread_idx.x)
        wg = fx.Int32(fx.block_idx.x)
        wave = tid // fx.Int32(WAVE)
        lane = tid - wave * fx.Int32(WAVE)
        M = fx.Int32(m_tokens)
        G = fx.Int32(n_workers)
        t = wg * fx.Int32(WAVES_) + wave
        active = t < M
        tc = active.select(t, M - c1)
        NV = fx.min(fx.Int32(ptr_buf_tensor(n_valid, fx.Int32)[c0]), M)
        routed = t < NV
        _atomic_add = _helpers(i32)

        ids_t = ptr_buf_tensor(topk_ids, fx.Int32)
        w_t = ptr_buf_tensor(topk_w, fx.Float32)
        slots_t = ptr_buf_tensor(slots, fx.Int32)
        tsc_t = ptr_buf_tensor(tsc, fx.Int8)
        ws_t = ptr_buf_tensor(ws, fx.Int32)
        sid_t = ptr_buf_tensor(sorted_ids, fx.Int32)
        sw_t = ptr_buf_tensor(sorted_w, fx.Float32)
        seid_t = ptr_buf_tensor(sorted_eids, fx.Int32)
        nv_t = ptr_buf_tensor(num_valid, fx.Int32)
        a1s_t = ptr_buf_tensor(a1_scale, fx.Int8)
        lds = fx.SharedAllocator().allocate(_Lds).peek()
        start_l, cnt_l, misc_l = lds.start.ptr, lds.cnt.ptr, lds.misc.ptr

        def shuf_addr(row, j):
            rem = j & fx.Int32(7)
            return (
                (row >> fx.Int32(5)) * fx.Int32(GROUPS * 32)
                + (j >> fx.Int32(3)) * fx.Int32(256)
                + (rem & fx.Int32(3)) * fx.Int32(64)
                + (row & fx.Int32(15)) * fx.Int32(4)
                + ((row >> fx.Int32(4)) & c1)
                + (rem >> fx.Int32(2)) * fx.Int32(2)
            )

        # the token's routes (launch 1 wrote them; a kernel boundary separates the launches)
        my_id = fx.Int32(0)
        my_w = fx.Float32(0.0)
        slot = fx.Int32(0)
        if lane < fx.Int32(TOPK):
            my_id = fx.Int32(ids_t[tc * fx.Int32(TOPK) + lane])
            my_w = fx.Float32(w_t[tc * fx.Int32(TOPK) + lane])
            slot = fx.Int32(slots_t[tc * fx.Int32(TOPK) + lane])

        # ---------------- counts -> LDS, then the ticket: the last reader zeroes the counts ----------------
        e = tid
        if e < fx.Int32(E_ALL):
            cnt_l[e] = (e < fx.Int32(E_ROUTED)).select(
                fx.Int32(ws_t[(e < fx.Int32(E_ROUTED)).select(e, c0)]), NV
            )
        gpu.barrier()
        # no fence: this workgroup's count loads have returned (their values are in LDS) before its ticket atomic, so
        # the last ticket holder cannot zero a count another workgroup still has to read; launch 1's atomics reach
        # launch 2 across the kernel boundary (a release fence here costs an L2 writeback per wave on gfx950)
        ticket = fx.Int32(0)
        if tid == c0:
            ticket = _atomic_add(ws, fx.Int32(WS_DONE), c1)
        if tid == c0:
            misc_l[c1] = (ticket == G - c1).select(c1, c0)
        gpu.barrier()
        if fx.Int32(misc_l[c1]) == c1:
            if tid < fx.Int32(E_ROUTED):
                ws_t[tid] = c0
            if tid == c0:
                ws_t[fx.Int32(WS_DONE)] = c0

        # ---------------- padded exclusive prefix over the 385 experts, in LDS ----------------
        if wave == c0:
            base = lane * fx.Int32(EPT)
            loc = []
            run = fx.Int32(0)
            for j in range_constexpr(EPT + 1):
                ej = base + fx.Int32(j)
                ok = ((fx.Int32(j) < fx.Int32(EPT)) | (lane == fx.Int32(WAVE - 1))) & (
                    ej < fx.Int32(E_ALL)
                )
                cj = ok.select(fx.Int32(cnt_l[ok.select(ej, c0)]), c0)
                loc.append((ej, ok, run))
                run = run + ((cj + fx.Int32(BM - 1)) & fx.Int32(-BM))
            incl = run
            for d in (1, 2, 4, 8, 16, 32):
                o = fx.Int32(gpu.shuffle_up(incl, fx.Int32(d), fx.Int32(WAVE)))
                incl = incl + (lane >= fx.Int32(d)).select(o, c0)
            excl = incl - run
            for ej, ok, r0 in loc:
                if ok:
                    start_l[ej] = excl + r0
            if lane == fx.Int32(WAVE - 1):
                misc_l[c0] = incl
        gpu.barrier()
        if (wg == c0) & (tid == c0):
            nv_t[c0] = fx.Int32(misc_l[c0])
            nv_t[c1] = M
        for eo in range(wg, fx.Int32(E_ALL), G):
            st = fx.Int32(start_l[eo])
            c = fx.Int32(cnt_l[eo])
            pad = ((c + fx.Int32(BM - 1)) & fx.Int32(-BM)) - c
            nblk = (c + fx.Int32(BM - 1)) >> fx.Int32(BM_SHIFT)
            for b in range(tid, nblk, fx.Int32(BLOCK_)):
                seid_t[(st >> fx.Int32(BM_SHIFT)) + b] = eo
            prow = st + c + tid
            if tid < pad:
                sid_t[prow] = (fx.Int32(TOPK) << fx.Int32(24)) | M
                sw_t[prow] = fx.Float32(0.0)

        # ---------------- rows: sorted ids / weights, e8m0 bytes into the sorted layout ----------------
        pos = fx.Int32(0)
        if routed & (lane < fx.Int32(TOPK)):
            pos = fx.Int32(start_l[my_id]) + slot
        if routed & (lane < fx.Int32(TOPK)):
            sid_t[pos] = (lane << fx.Int32(24)) | t
            sw_t[pos] = my_w
        rows = [
            fx.Int32(rocdl.readlane(i32, _raw(pos), k)) for k in range_constexpr(TOPK)
        ]
        if routed & ((lane & fx.Int32(3)) == c0):
            for it in range_constexpr(QUANT_ITERS):
                j = fx.Int32(it * (WAVE // 4)) + (lane >> fx.Int32(2))
                sb = fx.Int8(tsc_t[t * fx.Int32(GROUPS) + j])
                for k in range_constexpr(TOPK):
                    a1s_t[shuf_addr(rows[k], j)] = sb

    @flyc.jit
    def launch(
        topk_ids: fx.Pointer,
        topk_w: fx.Pointer,
        slots: fx.Pointer,
        tsc: fx.Pointer,
        sorted_ids: fx.Pointer,
        sorted_w: fx.Pointer,
        sorted_eids: fx.Pointer,
        num_valid: fx.Pointer,
        a1_scale: fx.Pointer,
        ws: fx.Pointer,
        n_valid: fx.Pointer,
        m_tokens: fx.Int32,
        n_workers: fx.Int32,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
    ):
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            pass
        p2_kernel(
            topk_ids,
            topk_w,
            slots,
            tsc,
            sorted_ids,
            sorted_w,
            sorted_eids,
            num_valid,
            a1_scale,
            ws,
            n_valid,
            m_tokens,
            n_workers,
        ).launch(
            grid=(arith.index_cast(T.index, _raw(fx.Int32(n_workers))), 1, 1),
            block=(BLOCK_, 1, 1),
            stream=stream,
        )

    return launch
