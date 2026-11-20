# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""MoE router: bf16 gate GEMM (fp32 accumulate) and the sqrt-softplus biased top-k.

GEMM (``build_router_gemm``): a workgroup owns ``16 a`` tokens x ``16 b``
experts of one of ``ksp`` k-slices. Its ``kw`` waves run MFMA 16x16x32 bf16
(gate rows as the A operand, x as B; both loaded straight from global memory,
k contiguous per lane) over consecutive k-steps and are summed in wave order
through LDS; the slice's partial tile goes to an fp32 scratch
``[ksp, m, experts]``. Each launch starts with cold L2, so slices and tiles are
chosen to put every CU to work on a few KB. Workgroup ``i`` runs on XCD
``i % 8``, one of ``xm`` token slabs x ``xn`` expert slabs, so an XCD's L2
serves its slab's re-reads.

Top-k (``build_router_topk``): one wave per token, ``experts / 64`` consecutive
experts per lane. It sums the ``ksp`` partials in slice order, then ``topk``
rounds of a wave argmax over ``score + bias`` (a DPP max tree; the lowest lane
among equal values, so ties go to the lowest expert id). A pick's weight is
``(score + bias) - bias`` in fp32 (as vLLM's ``topk_softplus_sqrt``, which
decodes it from the argmax key), scaled to sum ``routed_scaling_factor`` in
slot order; padding rows get id -1 and weight 0.

No atomics; each output has one writer; the result is bitwise repeatable.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from . import buffer_ops

H = 5120
STEPS = H // 32  # MFMA k-steps
LOG2E = 1.4426950408889634
# DPP max tree over a wave: quad_perm [1, 0, 3, 2], quad_perm [2, 3, 0, 1],
# row_half_mirror, row_mirror, row_bcast:15, row_bcast:31. Lane 63 ends with
# the wave maximum.
DPP_TREE = (0xB1, 0x4E, 0x141, 0x140, 0x142, 0x143)
COMPILE_HINTS = {
    "llvm_options": {
        "amdgpu-kernarg-preload": True,
        "amdgpu-kernarg-preload-count": 16,
    }
}


def build_router_gemm(
    experts: int,
    a: int,
    b: int,
    ksp: int,
    kw: int,
    chunk: int,
    xm: int,
    xn: int,
    wshuf: bool = False,
):
    """Launcher of the gate GEMM: ``a`` x ``b`` 16-token x 16-expert MFMA tiles
    per workgroup, k split ``ksp`` ways over workgroups and ``kw`` ways over a
    workgroup's waves, ``chunk`` k-steps loaded per batch. ``wshuf``: the gate in
    the load-time layout ``[expert tile, k step, lane, 8]``
    (``shuffle_router_gate``), so a wave's weight fragment is one 1 KB run."""
    E = experts
    assert E % (16 * b) == 0 and STEPS % (ksp * kw) == 0 and xm * xn == 8
    assert E // (16 * b) % xn == 0
    SPW = STEPS // ksp // kw  # k-steps per wave
    assert SPW % chunk == 0
    TM, TN = 16 * a, 16 * b
    NGX = E // TN // xn  # expert groups per XCD
    name = f"router_gemm_bf16_h{H}_e{E}_t{TM}x{TN}_s{ksp}_w{kw}_c{chunk}_x{xm}{xn}" + (
        "_ws" if wshuf else ""
    )

    @fx.struct
    class Shared:
        part: fx.Array[fx.Float32, kw * TM * TN]

    def mfma(wv, xv, acc):
        av = Vec(wv).bitcast(fx.BFloat16)
        bv = Vec(xv).bitcast(fx.BFloat16)
        return Vec(
            fx.rocdl.mfma_f32_16x16x32_bf16(
                Vec.make_type(4, fx.Float32), [av, bv, acc, 0, 0, 0]
            )
        )

    @flyc.kernel(name=name, known_block_size=[64 * kw, 1, 1])
    def kernel(
        arg_x: fx.Int64,  # [m, H] bf16, row stride i32_sx elements
        arg_w: fx.Int64,  # [E, H] bf16
        arg_p: fx.Int64,  # partials [ksp, m, E] fp32
        i32_m: fx.Int32,
        i32_sx: fx.Int32,
    ):
        tx, bid = fx.thread_idx.x, fx.block_idx.x
        lane = tx % 64
        wave = fx.Int32(fx.rocdl.readfirstlane(T.i32, tx // 64))
        n_mg = (i32_m + TM - 1) // TM
        mgs = (n_mg + xm - 1) // xm  # token groups per slab
        xcd = bid % 8
        local = bid // 8
        mgl = local % mgs
        rest = local // mgs
        s = rest % ksp
        ngl = rest // ksp
        mg = (xcd // xn) * mgs + mgl
        ng = (xcd % xn) * NGX + ngl
        if (mg < n_mg) & (ngl < NGX):
            m0 = mg * TM
            n0 = ng * TN
            smem = fx.SharedAllocator().allocate(Shared).peek()
            part = smem.part.ptr
            xr = buffer_ops.create_buffer_resource_from_addr(
                arg_x, num_records_bytes=fx.Int64(i32_m) * fx.Int64(i32_sx) * 2
            )
            wr = buffer_ops.create_buffer_resource_from_addr(
                arg_w, num_records_bytes=E * H * 2
            )
            q = lane // 16
            rows = [m0 + ai * 16 + lane % 16 for ai in range_constexpr(a)]
            nrow = [n0 + bj * 16 + lane % 16 for bj in range_constexpr(b)]

            def load(step):
                k0 = step * 32 + q * 8
                xs = [
                    Vec(
                        buffer_ops.buffer_load(
                            xr,
                            (rows[ai] * i32_sx + k0) >> 1,
                            vec_width=4,
                            dtype=fx.Int32,
                            mask=rows[ai] < i32_m,
                        )
                    )
                    for ai in range_constexpr(a)
                ]
                if const_expr(wshuf):
                    ws = [
                        Vec(
                            buffer_ops.buffer_load(
                                wr,
                                (((n0 // 16 + bj) * STEPS + step) * 64 + lane) * 4,
                                vec_width=4,
                                dtype=fx.Int32,
                            )
                        )
                        for bj in range_constexpr(b)
                    ]
                else:
                    ws = [
                        Vec(
                            buffer_ops.buffer_load(
                                wr,
                                (nrow[bj] * H + k0) >> 1,
                                vec_width=4,
                                dtype=fx.Int32,
                            )
                        )
                        for bj in range_constexpr(b)
                    ]
                return xs, ws

            acc = [
                [Vec.filled(4, 0.0, fx.Float32) for _ in range_constexpr(b)]
                for _ in range_constexpr(a)
            ]
            base = (s * kw + wave) * SPW
            nch = SPW // chunk
            cur = [load(base + j) for j in range_constexpr(chunk)]
            for c in range_constexpr(nch):
                nxt = (
                    [load(base + (c + 1) * chunk + j) for j in range_constexpr(chunk)]
                    if c + 1 < nch
                    else None
                )
                for j in range_constexpr(chunk):
                    xs, ws = cur[j]
                    for ai in range_constexpr(a):
                        for bj in range_constexpr(b):
                            acc[ai][bj] = mfma(ws[bj], xs[ai], acc[ai][bj])
                cur = nxt
            pr = buffer_ops.create_buffer_resource_from_addr(
                arg_p, num_records_bytes=fx.Int64(ksp) * fx.Int64(i32_m) * (E * 4)
            )
            if kw == 1:
                for ai in range_constexpr(a):
                    for bj in range_constexpr(b):
                        tok = rows[ai]
                        buffer_ops.buffer_store(
                            acc[ai][bj],
                            pr,
                            (s * i32_m + tok) * E + n0 + bj * 16 + q * 4,
                            mask=tok < i32_m,
                        )
            else:
                for ai in range_constexpr(a):
                    for bj in range_constexpr(b):
                        for i in range_constexpr(4):
                            part[
                                wave * (TM * TN)
                                + (ai * 16 + lane % 16) * TN
                                + bj * 16
                                + q * 4
                                + i
                            ] = acc[ai][bj][i]
                fx.gpu.barrier()
                for r in range_constexpr((TM * TN // 4 + 64 * kw - 1) // (64 * kw)):
                    e = (r * 64 * kw + tx) * 4  # four consecutive experts of one token
                    if e < TM * TN:
                        v = [part[e + i] for i in range_constexpr(4)]
                        for w in range_constexpr(1, kw):
                            v = [
                                v[i] + part[w * (TM * TN) + e + i]
                                for i in range_constexpr(4)
                            ]
                        tok = m0 + e // TN
                        buffer_ops.buffer_store(
                            Vec.from_elements(v, fx.Float32),
                            pr,
                            (s * i32_m + tok) * E + n0 + e % TN,
                            mask=tok < i32_m,
                        )

    @flyc.jit
    def launch_router_gemm(
        arg_x: fx.Int64,
        arg_w: fx.Int64,
        arg_p: fx.Int64,
        i32_m: fx.Int32,
        i32_sx: fx.Int32,
        stream: fx.Stream,
    ):
        n_slabs = (((i32_m + TM - 1) // TM) + xm - 1) // xm
        kernel(arg_x, arg_w, arg_p, i32_m, i32_sx).launch(
            grid=(fx.Int64(8 * n_slabs * ksp * NGX), 1, 1),
            block=(64 * kw, 1, 1),
            stream=stream,
        )

    launch_router_gemm.compile_hints = COMPILE_HINTS
    return launch_router_gemm


def _dpp(v, ctrl):
    """v_mov_b32_dpp with ``old = v``: an invalid source lane keeps its value
    (the lane-63 result never reads one)."""
    return fx.Int32(
        fx.rocdl.update_dpp(
            T.i32, fx.as_ir_value(v), fx.as_ir_value(v), ctrl, 0xF, 0xF, False
        )
    )


def _score(x):
    """sqrt(softplus(x)), x itself above 20. The exp is v_exp_f32 on
    ``x * log2(e)`` (``__expf``), the log and sqrt the accurate ones, as vLLM's
    ``topk_softplus_sqrt``."""
    e = fx.Float32(fx.rocdl.exp2(T.f32, fx.as_ir_value(x * fx.Float32(LOG2E))))
    sp = (x > fx.Float32(20.0)).select(x, fx.math.log(fx.Float32(1.0) + e))
    return fx.Float32(fx.math.sqrt(sp))


def build_router_topk(experts: int, topk: int, ksp: int, has_padding: bool):
    """Launcher of the top-k over the ``ksp`` fp32 partials."""
    E, TOPK = experts, topk
    assert E % 128 == 0 and TOPK <= 64
    EPL = E // 64  # experts per lane; lane l holds EPL l .. EPL l + EPL - 1
    name = f"router_sqrtsoftplus_top{TOPK}_e{E}_s{ksp}" + (
        "_padding" if has_padding else ""
    )

    @flyc.kernel(name=name, known_block_size=[64, 1, 1])
    def kernel(
        arg_p: fx.Int64,  # partials [ksp, m, E] fp32
        arg_bias: fx.Int64,  # [E] fp32
        arg_pad: fx.Int64,  # [m] bool (padding rows), unused without padding
        arg_w: fx.Int64,  # [m, TOPK] fp32
        arg_ids: fx.Int64,  # [m, TOPK] int32
        i32_m: fx.Int32,
        i32_scale: fx.Int32,  # routed_scaling_factor (fp32 bits)
    ):
        lane = fx.thread_idx.x
        t = fx.block_idx.x
        if t < i32_m:
            lr = buffer_ops.create_buffer_resource_from_addr(
                arg_p, num_records_bytes=fx.Int64(ksp) * fx.Int64(i32_m) * (E * 4)
            )
            br = buffer_ops.create_buffer_resource_from_addr(
                arg_bias, num_records_bytes=E * 4
            )
            wr = buffer_ops.create_buffer_resource_from_addr(
                arg_w, num_records_bytes=fx.Int64(i32_m) * (TOPK * 4)
            )
            ir_ = buffer_ops.create_buffer_resource_from_addr(
                arg_ids, num_records_bytes=fx.Int64(i32_m) * (TOPK * 4)
            )
            neg_inf = fx.Float32(float("-inf"))

            def load_lane(res, off):
                v = [
                    Vec(
                        buffer_ops.buffer_load(
                            res, off + 2 * j, vec_width=2, dtype=fx.Float32
                        )
                    )
                    for j in range_constexpr(EPL // 2)
                ]
                return [fx.Float32(v[i // 2][i % 2]) for i in range_constexpr(EPL)]

            parts = [
                load_lane(lr, (p * i32_m + t) * E + lane * EPL)
                for p in range_constexpr(ksp)
            ]
            bias = load_lane(br, lane * EPL)
            ch = []
            for i in range_constexpr(EPL):
                x = parts[0][i]
                for p in range_constexpr(1, ksp):
                    x = x + parts[p][i]
                c = _score(x) + bias[i]
                # A NaN score (c != c) loses to every finite one.
                ch.append((c == c).select(c, fx.Float32(-1e30)))  # noqa: PLR0124
            total = fx.Float32(0.0)
            out_id = fx.Int32(0)
            out_w = fx.Float32(0.0)
            for k in range_constexpr(TOPK):
                mv = ch[0]
                bi = fx.Int32(0)
                for i in range_constexpr(1, EPL):
                    take = ch[i] > mv
                    mv = take.select(ch[i], mv)
                    bi = take.select(fx.Int32(i), bi)
                r = mv
                for ctrl in DPP_TREE:
                    d = _dpp(r.bitcast(fx.Int32), ctrl).bitcast(fx.Float32)
                    r = (d > r).select(d, r)
                vmax = fx.Int32(
                    fx.rocdl.readlane(T.i32, r.bitcast(fx.Int32), fx.Int32(63))
                ).bitcast(fx.Float32)
                win = fx.Int32(
                    fx.math.cttz(fx.Int64(fx.rocdl.ballot(T.i64, mv == vmax))).to(
                        fx.Int32
                    )
                )
                idx = win * EPL + fx.Int32(fx.rocdl.readlane(T.i32, bi, win))
                bsel = bias[0]
                for i in range_constexpr(1, EPL):
                    bsel = (bi == i).select(bias[i], bsel)
                w = fx.Int32(
                    fx.rocdl.readlane(T.i32, (mv - bsel).bitcast(fx.Int32), win)
                ).bitcast(fx.Float32)
                hit = lane == win
                for i in range_constexpr(EPL):
                    ch[i] = (hit & (bi == i)).select(neg_inf, ch[i])
                total = total + w
                out_id = (lane == k).select(idx, out_id)
                out_w = (lane == k).select(w, out_w)
            factor = i32_scale.bitcast(fx.Float32) / (total > fx.Float32(0.0)).select(
                total, fx.Float32(1.0)
            )
            out_w = out_w * factor
            keep = lane < TOPK
            if has_padding:
                pr = buffer_ops.create_buffer_resource_from_addr(
                    arg_pad, num_records_bytes=(i32_m + 3) // 4 * 4
                )
                word = fx.Int32(
                    buffer_ops.buffer_load(pr, t // 4, vec_width=1, dtype=fx.Int32)
                )
                pad = ((word >> ((t % 4) * 8)) & 0xFF) != 0
                out_w = pad.select(fx.Float32(0.0), out_w)
                out_id = pad.select(fx.Int32(-1), out_id)
            buffer_ops.buffer_store(out_w, wr, t * TOPK + lane, mask=keep)
            buffer_ops.buffer_store(out_id, ir_, t * TOPK + lane, mask=keep)

    @flyc.jit
    def launch_router_topk(
        arg_p: fx.Int64,
        arg_bias: fx.Int64,
        arg_pad: fx.Int64,
        arg_w: fx.Int64,
        arg_ids: fx.Int64,
        i32_m: fx.Int32,
        i32_scale: fx.Int32,
        stream: fx.Stream,
    ):
        kernel(arg_p, arg_bias, arg_pad, arg_w, arg_ids, i32_m, i32_scale).launch(
            grid=(fx.Int64(i32_m), 1, 1), block=(64, 1, 1), stream=stream
        )

    launch_router_topk.compile_hints = COMPILE_HINTS
    return launch_router_topk
