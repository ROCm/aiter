# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL sqrtsoftplus top-k router for the DeepSeek-V4 / V4.1 MoE gates (gfx950).

Bit for bit the HIP kernel ``topk_gating_kernel_opt<bf16, float, E, renorm,
sqrtsoftplus>`` (csrc/include/topk_gating_kernels.cuh), for the two shapes it is
verified on:

- 384 experts, top-6: the V4 / V4.1 target model's router;
- 128 experts, top-3: the V4.1 DSpark draft's router.

One wave a row. Lane ``l`` holds experts ``l + 64 j``, sorts them with the HIP
kernel's ``sort_network_desc<N>``, and the wave picks the k-th winner with the
same DPP max and lowest-lane tie break. The renorm sum runs in selection order,
rows at or past ``num_valid_rows[0]`` get id -1 / weight 0.

The row count and the row strides are launch arguments, so one binary serves
every row count (each CUDA graph batch size and every eager step).

Ported from the plugin_poc ``ops/router_top6`` kernel
(https://github.com/coderfeli/plugin_poc/pull/129).
"""

from functools import cache

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import llvm
from flydsl.expr import Float32, Int32, Int64, const_expr, range_constexpr
from flydsl.expr.typing import T

from aiter.ops.flydsl.kernels.kernels_common import kernel_signature
from aiter.ops.flydsl.kernels.tensor_shim import ptr_buf_tensor

WAVE = 64
# (experts, top-k) verified against topk_gating_kernel_opt
SQRTSOFTPLUS_TOPK_SHAPES = ((384, 6), (128, 3))

# sort_network_desc<N> of topk_gating_kernels.cuh, N = experts a lane
_SORT6 = ((0, 1), (2, 3), (4, 5), (0, 2), (1, 4), (3, 5),
          (0, 1), (2, 3), (4, 5), (1, 2), (3, 4), (2, 3))  # fmt: skip
_SORT_NETWORKS = {2: ((0, 1),), 6: _SORT6}
_FLT_MAX = 3.4028234663852886e38


def _hw(x, name):
    # Raw intrinsic: the HIP kernel's exp2f / log2f / sqrtf compile to these single
    # hardware instructions; the fx math ops expand to the IEEE-accurate sequences,
    # whose results differ in the last bit.
    return Float32(llvm.call_intrinsic(T.f32, name, [x.ir_value()], [], []))


def _dpp_reduce_max(value):
    # Raw asm: multithread_reduce_max_dpp<64>'s six v_max_f32 DPP controls; fx has no
    # DPP max with row_bcast. Keep the `s_nop 1` before each: without the wait a DPP
    # read can see the previous instruction's stale lane.
    for control in ("quad_perm:[1,0,3,2]", "quad_perm:[2,3,0,1]",
                    "row_half_mirror", "row_mirror", "row_bcast:15", "row_bcast:31"):  # fmt: skip
        value = Float32(llvm.inline_asm(
            T.f32, [value.ir_value()],
            "s_nop 1\nv_max_f32 $0, $1, $1 " + control + " row_mask:0xf bank_mask:0xf bound_ctrl:1",
            "=v,v", has_side_effects=False, is_align_stack=False))  # fmt: skip
    return Int32(
        fx.rocdl.readlane(T.i32, value.bitcast(Int32).ir_value(), Int32(63).ir_value())
    ).bitcast(Float32)


def _writelane(value, lane, into):
    # Raw intrinsic: fx exposes readlane but not writelane.
    return Int32(
        llvm.call_intrinsic(
            T.i32,
            "llvm.amdgcn.writelane",
            [value.ir_value(), Int32(lane).ir_value(), into.ir_value()],
            [],
            [],
        )
    )


def _sqrt_normal_nonnegative(x):
    # LLVM's correctly rounded f32 sqrt on AMDGPU, for the zero / positive normal softplus
    y = _hw(x, "llvm.amdgcn.sqrt.f32")
    lower = (y.bitcast(Int32) - Int32(1)).bitcast(Float32)
    upper = (y.bitcast(Int32) + Int32(1)).bitcast(Float32)
    lo_residual = fx.fma(-lower, y, x)
    hi_residual = fx.fma(-upper, y, x)
    result = (lo_residual <= Float32(0.0)).select(lower, y)
    result = (hi_residual > Float32(0.0)).select(upper, result)
    return result


def _select(x_buf, bias_buf, scores, row, x_stride, lane, experts, topk, scale, renorm):
    """One live row: -> (ow, oi, factor). Lane k < topk holds the k-th selected
    expert's unbiased score ``ow`` and id ``oi``; ``factor`` is the row's weight factor.
    """
    per_lane = experts // WAVE
    vals, ids = [], []
    for j in range_constexpr(per_lane):
        e = lane + WAVE * j
        xv = x_buf[row * x_stride + e].to(Float32)
        # only +Inf is clamped; a NaN stays one until the guards below
        xv = (xv > Float32(_FLT_MAX)).select(Float32(_FLT_MAX), xv)
        sp = _hw(
            Float32(1.0)
            + _hw(xv * Float32(1.4426950408889634), "llvm.amdgcn.exp2.f32"),
            "llvm.amdgcn.log.f32",
        ) * Float32(0.6931471805599453)
        sp = (xv > Float32(20.0)).select(xv, sp)
        score = _sqrt_normal_nonnegative(sp)
        scores[e] = fx.isnan(score).select(Float32(0.0), score)
        vals.append(fx.maxnumf(score + bias_buf[e], Float32(float("-inf"))))
        ids.append(e)
    for p, q in _SORT_NETWORKS[per_lane]:
        swap = vals[p] < vals[q]
        vals[p], vals[q] = swap.select(vals[q], vals[p]), swap.select(vals[p], vals[q])
        ids[p], ids[q] = swap.select(ids[q], ids[p]), swap.select(ids[p], ids[q])
    # A lane can run out only when it holds fewer experts than are selected (the draft:
    # 2 a lane, top-3). The HIP kernel then offers (-inf, id 0) for it, and a round such
    # an offer wins takes its weight from lane 0's head only if that head IS expert 0,
    # else 0.0 (`i_won ? orig[cursor] : 0`, read at lane `id & 63`). Reachable when a
    # row has fewer than top-k experts above -inf (NaN logits). Not emitted where no
    # lane can run out.
    can_run_out = topk > per_lane
    cursor = Int32(0)
    oi = Int32(0)
    genuine = Int32(0)
    for k in range_constexpr(topk):
        cv, ci = vals[0], ids[0]
        best = _dpp_reduce_max(cv)
        # the first winning lane, as the HIP kernel's ballot + ctz (lane 0 if none)
        mask = Int64(fx.rocdl.ballot(T.i64, (cv == best).ir_value()))
        win = Int32(fx.cttz(mask))
        win = (mask == Int64(0)).select(Int32(0), win)
        wi = Int32(fx.rocdl.readlane(T.i32, ci.ir_value(), win.ir_value()))
        won = (cursor < per_lane) & (ci == wi)
        if const_expr(can_run_out):
            took = Int32(
                fx.rocdl.readlane(
                    T.i32,
                    won.select(Int32(1), Int32(0)).ir_value(),
                    (wi & Int32(WAVE - 1)).ir_value(),
                )
            )
            genuine = _writelane(took, k, genuine)
        cursor = cursor + won.select(Int32(1), Int32(0))
        for j in range_constexpr(per_lane - 1):
            vals[j] = won.select(vals[j + 1], vals[j])
            ids[j] = won.select(ids[j + 1], ids[j])
        vals[per_lane - 1] = won.select(Float32(float("-inf")), vals[per_lane - 1])
        ids[per_lane - 1] = won.select(Int32(0), ids[per_lane - 1])
        oi = _writelane(wi, k, oi)
    # the winners' unbiased scores (lanes >= topk hold id 0: a harmless read)
    ow = scores[oi]
    if const_expr(can_run_out):
        ow = (genuine != Int32(0)).select(ow, Float32(0.0))
    total = Float32(0.0)
    for k in range_constexpr(topk):
        weight = Int32(
            fx.rocdl.readlane(T.i32, ow.bitcast(Int32).ir_value(), Int32(k).ir_value())
        ).bitcast(Float32)
        total = total + weight
    factor = Float32(scale)
    if const_expr(renorm):
        factor = factor / fx.maxnumf(total, Float32(1e-20))
    return ow, oi, factor


def _flush_fp32_denormals():
    # Raw intrinsic: MODE.fp_denorm single = 0, as the HIP kernel runs; fx has no
    # spelling for the MODE register.
    llvm.call_intrinsic(
        None,
        "llvm.amdgcn.s.setreg",
        [Int32(1 | (4 << 6) | (1 << 11)).ir_value(), Int32(0).ir_value()],
        [],
        [],
    )


@cache
def build_sqrtsoftplus_topk_module(
    experts: int, topk: int, has_valid: bool, scale: float, renorm: bool
):
    """The launcher of one configuration. Arguments: gating (bf16 [rows, experts],
    row stride ``x_stride``), bias (fp32 [experts]), weights (fp32) and ids (int32),
    both [rows, >= topk] with row stride ``out_stride``, num_valid_rows (int32,
    read when ``has_valid``), rows."""
    if (experts, topk) not in SQRTSOFTPLUS_TOPK_SHAPES:
        raise ValueError(f"unsupported sqrtsoftplus top-k shape {(experts, topk)}")

    @fx.struct
    class RawScores:
        scores: fx.Array[Float32, experts, 16]

    name = "sqrtsoftplus_topk_" + kernel_signature(
        e=experts,
        k=topk,
        nv=has_valid,
        s=round(scale * 1000),
        rn=renorm,
    )

    @flyc.kernel(name=name, known_block_size=[WAVE, 1, 1])
    def sqrtsoftplus_topk_kernel(
        x: Int64,
        bias: Int64,
        out_w: Int64,
        out_i: Int64,
        nvr: Int64,
        x_stride: Int32,
        out_stride: Int32,
    ):
        _flush_fp32_denormals()
        scores = (
            fx.SharedAllocator()
            .allocate(RawScores)
            .scores.peek()
            .view(fx.make_layout(experts, 1))
        )
        x_buf = ptr_buf_tensor(x, elem=fx.BFloat16)
        bias_buf = ptr_buf_tensor(bias, elem=Float32)
        w_buf = ptr_buf_tensor(out_w, elem=Float32)
        i_buf = ptr_buf_tensor(out_i, elem=Int32)
        lane = fx.thread_idx.x
        row = fx.block_idx.x
        live = row >= 0
        if const_expr(has_valid):
            live = row < ptr_buf_tensor(nvr, elem=Int32)[0]
        out_at = row * out_stride + lane
        if live:
            ow, oi, factor = _select(
                x_buf,
                bias_buf,
                scores,
                row,
                x_stride,
                lane,
                experts,
                topk,
                scale,
                renorm,
            )
            if lane < topk:
                w_buf[out_at] = ow * factor
                i_buf[out_at] = oi
        else:
            if lane < topk:
                w_buf[out_at] = Float32(0.0)
                i_buf[out_at] = Int32(-1)

    @flyc.jit
    def launch(
        x: Int64,
        bias: Int64,
        out_w: Int64,
        out_i: Int64,
        nvr: Int64,
        x_stride: Int32,
        out_stride: Int32,
        rows: Int32,
        stream: fx.Stream,
    ):
        sqrtsoftplus_topk_kernel(
            x, bias, out_w, out_i, nvr, x_stride, out_stride
        ).launch(grid=(rows, 1, 1), block=(WAVE, 1, 1), stream=stream)

    launch.func.__name__ = f"launch_{name}"
    return launch
