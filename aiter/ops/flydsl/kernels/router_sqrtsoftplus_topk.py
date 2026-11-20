# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""MoE router for decode rows: bf16 gate GEMM, then sqrt-softplus biased top-6.

Shapes: x bf16 ``[m, 5120]``, gate bf16 ``[384, 5120]``, fp32 bias ``[384]``
(DeepSeek-V4.1-Flash). Two launches (gfx950):

GEMM. A workgroup owns ``16 a`` tokens x ``16 b`` experts of one of ``ksp``
k-slices. Its ``kw`` waves run MFMA 16x16x32 bf16 (gate rows as the A operand,
x as B; both loaded straight from global memory, k contiguous per lane) over
consecutive k-steps and are summed in wave order through LDS; the slice's fp32
partial tile goes to a ``[ksp, m, 384]`` scratch. Workgroup ``i`` runs on XCD
``i % 8``, which owns one of ``xm`` token slabs x ``xn`` expert slabs, so the
XCD's L2 serves its slab's re-reads. With ``wshuf`` the gate is in the
load-time layout ``[expert tile, k step, lane, 8]`` and a wave's weight
fragment is one contiguous 1 KB run.

Top-6. One wave per token, six consecutive experts per lane. It sums the
``ksp`` partials in slice order, scores ``sqrt(softplus(logit))`` (``logit``
itself above 20; fast ``exp2``, accurate ``log`` and ``sqrt``), then six rounds
of a wave argmax over ``score + bias`` (DPP max tree; the lowest lane among
equal values, so ties go to the lowest expert id). A pick's weight is
``(score + bias) - bias`` in fp32, as vLLM's ``topk_softplus_sqrt`` decodes it
from its argmax key, scaled to sum ``routed_scaling_factor`` in slot order.
Padding rows get id -1 and weight 0.

Global memory goes through buffer-resource views (``ptr_buf_tensor``) and
``fx.copy`` atoms (``buf_copy_load`` / ``buf_copy_store``); the views' byte
bounds make out-of-range loads return 0 and out-of-range stores no-ops.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from .dpp_utils import update_dpp_i32
from .kernels_common import LOG2E
from .tensor_shim import buf_copy_load, buf_copy_store, ptr_buf_tensor

HIDDEN = 5120
EXPERTS = 384
TOPK = 6
_STEPS = HIDDEN // 32  # MFMA k-steps
# A masked-off access goes to this byte offset. Every buffer here is smaller
# (the wrapper keeps x under 1 GiB; partials are at most 16 x 65536 x 1536 B),
# so the hardware bound check drops the store or returns 0 for the load.
_OOB_BYTES = 0x7FFFFFF0
# quad_perm [1,0,3,2], quad_perm [2,3,0,1], row_half_mirror, row_mirror,
# row_bcast 15, row_bcast 31: after these the wave max is in lane 63.
_DPP_MAX_TREE = (0xB1, 0x4E, 0x141, 0x140, 0x142, 0x143)
# Arguments arrive in SGPRs without a kernarg load: both launches are short.
_KERNARG_PRELOAD_HINTS = {
    "llvm_options": {
        "amdgpu-kernarg-preload": True,
        "amdgpu-kernarg-preload-count": 16,
    }
}


def _at(valid, index, unit_bytes):
    """``index`` (in ``unit_bytes`` units) where ``valid``, else past the
    buffer's bound."""
    return valid.select(index, fx.Int32(_OOB_BYTES // unit_bytes))


def _mfma_bf16(a_i32x4, b_i32x4, acc):
    """``acc + A B`` of one MFMA 16x16x32 bf16 on this lane's operands (eight
    bf16 of k each, as four dwords)."""
    atom = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 32, fx.BFloat16))
    a = fx.make_rmem_tensor(8, fx.BFloat16)
    a.store(Vec(a_i32x4).bitcast(fx.BFloat16))
    b = fx.make_rmem_tensor(8, fx.BFloat16)
    b.store(Vec(b_i32x4).bitcast(fx.BFloat16))
    c = fx.make_rmem_tensor(4, fx.Float32)
    c.store(acc)
    fx.gemm(atom, c, a, b, c)
    return Vec(c.load())


def build_router_gemm_module(
    a: int, b: int, ksp: int, kw: int, chunk: int, xm: int, xn: int, wshuf: bool
):
    """JIT launcher ``(x, w, partials, m, x_row_stride, stream)``.

    ``a`` x ``b`` 16x16 MFMA tiles per workgroup, k split ``ksp`` ways over
    workgroups and ``kw`` ways over a workgroup's waves, ``chunk`` k-steps
    loaded per batch, XCD grid ``xm`` x ``xn``.
    """
    assert EXPERTS % (16 * b) == 0 and _STEPS % (ksp * kw) == 0
    assert xm * xn == 8 and EXPERTS // (16 * b) % xn == 0
    spw = _STEPS // ksp // kw  # k-steps per wave
    assert spw % chunk == 0
    tm, tn = 16 * a, 16 * b
    ng_total = EXPERTS // tn  # expert groups
    ngx = ng_total // xn  # expert groups per XCD
    name = (
        f"router_gemm_bf16_h{HIDDEN}_e{EXPERTS}_t{tm}x{tn}_s{ksp}_w{kw}_c{chunk}"
        f"_x{xm}{xn}" + ("_ws" if wshuf else "")
    )

    @fx.struct
    class Shared:
        part: fx.Array[fx.Float32, kw * tm * tn]

    @flyc.kernel(name=name, known_block_size=[64 * kw, 1, 1])
    def kernel(
        arg_x: fx.Pointer,  # [m, 5120] bf16, row stride i32_sx (a multiple of 8)
        arg_w: fx.Pointer,  # [384, 5120] bf16
        arg_p: fx.Pointer,  # partials [ksp, m, 384] fp32
        i32_m: fx.Int32,
        i32_sx: fx.Int32,
    ):
        tx, bid = fx.thread_idx.x, fx.block_idx.x
        lane = tx % 64
        # Raw rocdl: FlyDSL has no wave-uniform cast. Without it the wave's
        # k-step base and the operand offsets built on it stay per-lane VALU
        # math: the 32x64 tile with kw=2 grows by 64 v_add_u32 and 34
        # v_lshl_or_b32 (gfx950 ISA, FlyDSL 0.5.0).
        wave = fx.Int32(fx.rocdl.readfirstlane(T.i32, tx // 64))
        n_mg = (i32_m + tm - 1) // tm
        mgs = (n_mg + xm - 1) // xm  # token groups per slab
        xcd = bid % 8
        local = bid // 8
        mgl = local % mgs
        rest = local // mgs
        s = rest % ksp
        ngl = rest // ksp
        mg = (xcd // xn) * mgs + mgl
        ng = (xcd % xn) * ngx + ngl
        if (mg < n_mg) & (ngl < ngx):
            m0 = mg * tm
            n0 = ng * tn
            part = fx.SharedAllocator().allocate(Shared).peek().part.ptr
            # 16 B units: eight bf16 of k (one MFMA operand) or four fp32.
            xb = ptr_buf_tensor(
                arg_x,
                fx.Int32,
                unit_elems=4,
                num_records_bytes=fx.Int64(i32_m) * fx.Int64(i32_sx * 2),
            )
            wb = ptr_buf_tensor(
                arg_w, fx.Int32, unit_elems=4, num_records_bytes=EXPERTS * HIDDEN * 2
            )
            pb = ptr_buf_tensor(
                arg_p,
                fx.Float32,
                unit_elems=4,
                num_records_bytes=fx.Int64(ksp) * fx.Int64(i32_m) * (EXPERTS * 4),
            )
            q = lane // 16
            sx16 = i32_sx // 8
            rows = [m0 + ai * 16 + lane % 16 for ai in range_constexpr(a)]
            nrow = [n0 + bj * 16 + lane % 16 for bj in range_constexpr(b)]

            def load(step):
                k16 = step * 4 + q  # k = 32 step + 8 q
                xs = [
                    buf_copy_load(
                        xb,
                        _at(rows[ai] < i32_m, rows[ai] * sx16 + k16, 16),
                        fx.Int32,
                        unit_elems=4,
                    )
                    for ai in range_constexpr(a)
                ]
                if const_expr(wshuf):
                    ws = [
                        buf_copy_load(
                            wb,
                            ((n0 // 16 + bj) * _STEPS + step) * 64 + lane,
                            fx.Int32,
                            unit_elems=4,
                        )
                        for bj in range_constexpr(b)
                    ]
                else:
                    ws = [
                        buf_copy_load(
                            wb, nrow[bj] * (HIDDEN // 8) + k16, fx.Int32, unit_elems=4
                        )
                        for bj in range_constexpr(b)
                    ]
                return xs, ws

            acc = [
                [Vec.filled(4, 0.0, fx.Float32) for _ in range_constexpr(b)]
                for _ in range_constexpr(a)
            ]
            base = (s * kw + wave) * spw
            nch = spw // chunk
            # Each load batch is fenced off (sched_barrier) so it issues ahead
            # of the MFMAs of the batch before it. The 16 B copy atoms load
            # i128 values, which the gfx950 scheduler otherwise sinks next to
            # their MFMAs, losing the prefetch (up to 3.6 us at 768 rows).
            cur = [load(base + j) for j in range_constexpr(chunk)]
            fx.rocdl.sched_barrier(0)
            for c in range_constexpr(nch):
                nxt = None
                if c + 1 < nch:
                    nxt = [
                        load(base + (c + 1) * chunk + j) for j in range_constexpr(chunk)
                    ]
                    fx.rocdl.sched_barrier(0)
                for j in range_constexpr(chunk):
                    xs, ws = cur[j]
                    for ai in range_constexpr(a):
                        for bj in range_constexpr(b):
                            acc[ai][bj] = _mfma_bf16(ws[bj], xs[ai], acc[ai][bj])
                cur = nxt
            if const_expr(kw == 1):
                for ai in range_constexpr(a):
                    for bj in range_constexpr(b):
                        tok = rows[ai]
                        buf_copy_store(
                            pb,
                            _at(
                                tok < i32_m,
                                (s * i32_m + tok) * (EXPERTS // 4)
                                + (n0 + bj * 16) // 4
                                + q,
                                16,
                            ),
                            acc[ai][bj],
                            fx.Float32,
                            unit_elems=4,
                        )
            else:
                for ai in range_constexpr(a):
                    for bj in range_constexpr(b):
                        for i in range_constexpr(4):
                            part[
                                wave * (tm * tn)
                                + (ai * 16 + lane % 16) * tn
                                + bj * 16
                                + q * 4
                                + i
                            ] = acc[ai][bj][i]
                fx.gpu.barrier()
                for r in range_constexpr((tm * tn // 4 + 64 * kw - 1) // (64 * kw)):
                    e = (r * 64 * kw + tx) * 4  # four consecutive experts of a token
                    if e < tm * tn:
                        v = [part[e + i] for i in range_constexpr(4)]
                        for w in range_constexpr(1, kw):
                            v = [
                                v[i] + part[w * (tm * tn) + e + i]
                                for i in range_constexpr(4)
                            ]
                        tok = m0 + e // tn
                        buf_copy_store(
                            pb,
                            _at(
                                tok < i32_m,
                                (s * i32_m + tok) * (EXPERTS // 4) + (n0 + e % tn) // 4,
                                16,
                            ),
                            Vec.from_elements(v, fx.Float32),
                            fx.Float32,
                            unit_elems=4,
                        )

    @flyc.jit
    def launch_router_gemm(
        arg_x: fx.Pointer,
        arg_w: fx.Pointer,
        arg_p: fx.Pointer,
        i32_m: fx.Int32,
        i32_sx: fx.Int32,
        stream: fx.Stream,
    ):
        n_slab = ((i32_m + tm - 1) // tm + xm - 1) // xm
        kernel(arg_x, arg_w, arg_p, i32_m, i32_sx).launch(
            grid=(fx.Int64(8 * n_slab * ksp * ngx), 1, 1),
            block=(64 * kw, 1, 1),
            stream=stream,
        )

    launch_router_gemm.kernel_name = name
    launch_router_gemm.compile_hints = _KERNARG_PRELOAD_HINTS
    return launch_router_gemm


def _dpp_keep(v, ctrl):
    """``v_mov_b32_dpp``; an invalid source lane keeps its own value (the
    lane-63 result never reads one)."""
    return fx.Int32(update_dpp_i32(v, v, ctrl, 0xF, 0xF, False))


def _score(x):
    """sqrt(softplus(x)), ``x`` itself above 20; fast exp2 (``__expf``),
    accurate log and sqrt, as vLLM's ``topk_softplus_sqrt``."""
    # Raw rocdl: fx.math.exp2 lowers to llvm.exp2.f32, which gfx950 codegen
    # wraps in a denormal range fix-up (v_cmp, two v_cndmask, v_add_f32 and
    # v_ldexp_f32 around each v_exp_f32; +42 instructions in this kernel).
    # vLLM's __expf is the bare v_exp_f32, which the bitwise match needs.
    e = fx.Float32(fx.rocdl.exp2(T.f32, (x * fx.Float32(LOG2E)).ir_value()))
    sp = (x > fx.Float32(20.0)).select(x, fx.math.log(fx.Float32(1.0) + e))
    return fx.Float32(fx.math.sqrt(sp))


def build_router_topk_module(ksp: int, has_padding: bool):
    """JIT launcher ``(partials, bias, is_padding, topk_weights, topk_ids, m,
    scale_bits, stream)``; ``is_padding`` is read only with ``has_padding``."""
    name = f"router_sqrtsoftplus_top{TOPK}_e{EXPERTS}_s{ksp}" + (
        "_padding" if has_padding else ""
    )
    # Lane l holds experts 6 l .. 6 l + 5, so the lowest lane is the lowest id.
    epl = EXPERTS // 64

    @flyc.kernel(name=name, known_block_size=[64, 1, 1])
    def kernel(
        arg_p: fx.Pointer,  # partials [ksp, m, 384] fp32
        arg_bias: fx.Pointer,  # [384] fp32
        arg_pad: fx.Pointer,  # [m] bool
        arg_w: fx.Pointer,  # [m, 6] fp32
        arg_ids: fx.Pointer,  # [m, 6] int32
        i32_m: fx.Int32,
        i32_scale: fx.Int32,  # routed_scaling_factor (fp32 bits)
    ):
        lane = fx.thread_idx.x
        t = fx.block_idx.x
        if t < i32_m:
            # 8 B units: two of a lane's six experts.
            pb = ptr_buf_tensor(
                arg_p,
                fx.Float32,
                unit_elems=2,
                num_records_bytes=fx.Int64(ksp) * fx.Int64(i32_m) * (EXPERTS * 4),
            )
            bb = ptr_buf_tensor(
                arg_bias, fx.Float32, unit_elems=2, num_records_bytes=EXPERTS * 4
            )
            wb = ptr_buf_tensor(
                arg_w, fx.Float32, num_records_bytes=fx.Int64(i32_m) * (TOPK * 4)
            )
            ib = ptr_buf_tensor(
                arg_ids, fx.Int32, num_records_bytes=fx.Int64(i32_m) * (TOPK * 4)
            )
            neg_inf = fx.Float32(float("-inf"))

            def load6(buf, unit):
                v = [
                    buf_copy_load(buf, unit + j, fx.Float32, unit_elems=2)
                    for j in range_constexpr(3)
                ]
                return [fx.Float32(v[i // 2][i % 2]) for i in range_constexpr(epl)]

            parts = [
                load6(pb, (p * i32_m + t) * (EXPERTS // 2) + lane * (epl // 2))
                for p in range_constexpr(ksp)
            ]
            bias = load6(bb, lane * (epl // 2))
            ch = []
            for i in range_constexpr(epl):
                x = parts[0][i]
                for p in range_constexpr(1, ksp):
                    x = x + parts[p][i]
                c = _score(x) + bias[i]
                # A NaN logit ranks below every finite candidate (the
                # ordered compare is false only for NaN).
                not_nan = c == c  # noqa: PLR0124
                ch.append(not_nan.select(c, fx.Float32(-1e30)))
            total = fx.Float32(0.0)
            out_id = fx.Int32(0)
            out_w = fx.Float32(0.0)
            for k in range_constexpr(TOPK):
                mv = ch[0]
                bi = fx.Int32(0)
                for i in range_constexpr(1, epl):
                    take = ch[i] > mv
                    mv = take.select(ch[i], mv)
                    bi = take.select(fx.Int32(i), bi)
                # Raw rocdl (DPP): the stable gpu.shuffle_xor butterfly lowers
                # to four ds_swizzle_b32 and two v_permlane*_swap per round,
                # with LDS waits (+24 ds_swizzle, +27 s_waitcnt per kernel);
                # this tree is six v_mov_b32_dpp per round.
                r = mv
                for ctrl in _DPP_MAX_TREE:
                    d = _dpp_keep(r.bitcast(fx.Int32), ctrl).bitcast(fx.Float32)
                    r = (d > r).select(d, r)
                vmax = fx.Float32(fx.gpu.shuffle_idx(r, fx.Int32(63), 64))
                # Raw rocdl (ballot): FlyDSL has no wave vote.
                win = fx.Int32(
                    fx.math.cttz(fx.Int64(fx.rocdl.ballot(T.i64, mv == vmax))).to(
                        fx.Int32
                    )
                )
                # Raw rocdl (readlane): gpu.shuffle_idx from a runtime lane
                # lowers to ds_bpermute_b32 (+12 ds_bpermute, +15 s_waitcnt
                # per kernel); the constant lane 63 above becomes v_readlane.
                idx = win * epl + fx.Int32(fx.rocdl.readlane(T.i32, bi, win))
                bsel = bias[0]
                for i in range_constexpr(1, epl):
                    bsel = (bi == i).select(bias[i], bsel)
                w = fx.Int32(
                    fx.rocdl.readlane(T.i32, (mv - bsel).bitcast(fx.Int32), win)
                ).bitcast(fx.Float32)
                hit = lane == win
                for i in range_constexpr(epl):
                    ch[i] = (hit & (bi == i)).select(neg_inf, ch[i])
                total = total + w
                out_id = (lane == k).select(idx, out_id)
                out_w = (lane == k).select(w, out_w)
            factor = i32_scale.bitcast(fx.Float32) / (total > fx.Float32(0.0)).select(
                total, fx.Float32(1.0)
            )
            out_w = out_w * factor
            if const_expr(has_padding):
                # One byte per row (torch.bool): no alignment or length
                # assumption beyond the m bytes of is_padding.
                padb = ptr_buf_tensor(
                    arg_pad, fx.Int8, num_records_bytes=fx.Int64(i32_m)
                )
                pad = buf_copy_load(padb, t, fx.Int8) != fx.Int8(0)
                out_w = pad.select(fx.Float32(0.0), out_w)
                out_id = pad.select(fx.Int32(-1), out_id)
            slot = _at(lane < TOPK, t * TOPK + lane, 4)
            buf_copy_store(wb, slot, out_w, fx.Float32)
            buf_copy_store(ib, slot, out_id, fx.Int32)

    @flyc.jit
    def launch_router_topk(
        arg_p: fx.Pointer,
        arg_bias: fx.Pointer,
        arg_pad: fx.Pointer,
        arg_w: fx.Pointer,
        arg_ids: fx.Pointer,
        i32_m: fx.Int32,
        i32_scale: fx.Int32,
        stream: fx.Stream,
    ):
        kernel(arg_p, arg_bias, arg_pad, arg_w, arg_ids, i32_m, i32_scale).launch(
            grid=(fx.Int64(i32_m), 1, 1), block=(64, 1, 1), stream=stream
        )

    launch_router_topk.kernel_name = name
    launch_router_topk.compile_hints = _KERNARG_PRELOAD_HINTS
    return launch_router_topk
