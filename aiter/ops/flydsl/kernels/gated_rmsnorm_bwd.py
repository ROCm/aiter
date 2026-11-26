# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Gated RMSNorm backward FlyDSL kernel (Kimi Delta Attention output norm).

``out = rmsnorm(o) * w * sigmoid(gp)`` over the last dim. One wave per row of
``dim`` (a multiple of 64); each wave walks ``rows_per_wave`` rows and writes its
norm-weight gradient partial. Every tensor is contiguous fp32 and addressed
through a flat unit-stride view; the launch wrapper in ``../gated_rmsnorm_bwd.py``
guarantees both. The device index is only a cache key.
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import range_constexpr

_DEFAULT_STREAM = fx.Stream(None)
WAVE = 64


def _flat(t, span):
    return fx.Tensor(fx.make_view(fx.get_iter(t), fx.make_layout(span, 1)))


def _sigmoid(x):
    return fx.Float32(1.0) / (fx.Float32(1.0) + fx.exp(-x))


def _wave_sum(v):
    for sh in [32, 16, 8, 4, 2, 1]:
        v = v + v.shuffle_xor(fx.Int32(sh), fx.Int32(WAVE))
    return v


@functools.lru_cache(maxsize=32)
def create_gated_rmsnorm_bwd_kernel(
    dim: int, eps: float, waves: int, rows_per_wave: int, device: int | None = None
):
    """o, gp, dy, do, dgp: [rows, dim]; w: [dim]; dw_part: [n_groups, dim].

    out = o * r * w * sigmoid(gp), r = rsqrt(mean(o^2) + eps).
    """
    E = dim // WAVE
    block = waves * WAVE
    RPW = rows_per_wave

    @flyc.kernel(known_block_size=[block, 1, 1])
    def gated_rmsnorm_bwd_kernel(
        o: fx.Tensor,
        gp: fx.Tensor,
        w: fx.Tensor,
        dy: fx.Tensor,
        do: fx.Tensor,
        dgp: fx.Tensor,
        dw_part: fx.Tensor,
        rows: fx.Int64,
        n_groups: fx.Int64,
        span: fx.Int64,
        dw_span: fx.Int64,
    ):
        o = _flat(o, span)
        gp = _flat(gp, span)
        dy = _flat(dy, span)
        do = _flat(do, span)
        dgp = _flat(dgp, span)
        w = _flat(w, fx.Int64(dim))
        dw_part = _flat(dw_part, dw_span)
        tid = fx.Int64(fx.thread_idx.x)
        group = fx.Int64(fx.block_idx.x) * waves + tid // WAVE
        lane = tid % WAVE
        wv = [fx.Float32(w[lane + e * WAVE]) for e in range_constexpr(E)]
        dwacc = [fx.Float32(0.0) for _ in range_constexpr(E)]
        for r in range_constexpr(RPW):
            row = group * RPW + r
            valid = row < rows
            safe = valid.select(row, fx.Int64(0))
            offs = [safe * dim + lane + e * WAVE for e in range_constexpr(E)]
            ov = [fx.Float32(o[x]) for x in offs]
            gv = [_sigmoid(fx.Float32(gp[x])) for x in offs]
            # rows past the end contribute nothing to the weight gradient
            dv = [valid.select(fx.Float32(dy[x]), fx.Float32(0.0)) for x in offs]
            ss = fx.Float32(0.0)
            for e in range_constexpr(E):
                ss = ss + ov[e] * ov[e]
            rs = fx.Float32(1.0) / fx.sqrt(
                _wave_sum(ss) * fx.Float32(1.0 / dim) + fx.Float32(eps)
            )
            dh = [dv[e] * wv[e] * gv[e] for e in range_constexpr(E)]
            dot = fx.Float32(0.0)
            for e in range_constexpr(E):
                dot = dot + dh[e] * ov[e]
            coef = _wave_sum(dot) * rs * rs * rs * fx.Float32(1.0 / dim)
            for e in range_constexpr(E):
                on = ov[e] * rs
                dwacc[e] = dwacc[e] + dv[e] * on * gv[e]
                if valid:
                    do[offs[e]] = dh[e] * rs - ov[e] * coef
                    dgp[offs[e]] = (
                        dv[e] * on * wv[e] * gv[e] * (fx.Float32(1.0) - gv[e])
                    )
        if group < n_groups:
            for e in range_constexpr(E):
                dw_part[group * dim + lane + e * WAVE] = dwacc[e]

    @flyc.jit
    def launch(
        o: fx.Tensor,
        gp: fx.Tensor,
        w: fx.Tensor,
        dy: fx.Tensor,
        do: fx.Tensor,
        dgp: fx.Tensor,
        dw_part: fx.Tensor,
        rows: fx.Int64,
        n_groups: fx.Int64,
        span: fx.Int64,
        dw_span: fx.Int64,
        stream: fx.Stream = _DEFAULT_STREAM,
    ):
        gated_rmsnorm_bwd_kernel(
            o, gp, w, dy, do, dgp, dw_part, rows, n_groups, span, dw_span
        ).launch(
            grid=((n_groups + waves - 1) // waves, 1, 1),
            block=(block, 1, 1),
            stream=stream,
        )

    return launch
