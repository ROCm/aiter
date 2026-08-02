# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2025-2026 FlyDSL Project Contributors

"""Shared device-side SiLU / SiTUv2 activation helpers for FlyDSL MoE kernels.

Elementwise f32-register helpers (exp2/rcp-based sigmoid, sign-restored tanh, and the
gate*up batch forms) usable by any FlyDSL gemm1 fused gate+up epilog. Leaf module:
depends only on flydsl and ``tensor_shim._to_raw``.
"""

import flydsl.expr as fx
from flydsl.expr import arith, rocdl
from flydsl.expr.typing import T

from aiter.ops.flydsl.kernels.tensor_shim import _to_raw as _raw

LOG2E = 1.4426950408889634


def _silu_mul_batch(gs, us):
    e = [fx.Float32(rocdl.exp2(T.f32, _raw(g * fx.Float32(-LOG2E)))) for g in gs]
    sig = [fx.Float32(rocdl.rcp(T.f32, _raw(fx.Float32(1.0) + ei))) for ei in e]
    return [gs[i] * sig[i] * us[i] for i in range(len(gs))]


def _sigmoid_f32(g):
    e = fx.Float32(rocdl.exp2(T.f32, _raw(g * fx.Float32(-LOG2E))))
    return fx.Float32(rocdl.rcp(T.f32, _raw(fx.Float32(1.0) + e)))


def _tanh_f32(x):
    # tanh via exp2/rcp, sign-restored (aiter mixed_moe tanh_elem):
    #   t = (1-exp(-2|x|))/(1+exp(-2|x|)),  tanh(x) = sign(x)*t
    neg_two_log2e = fx.Float32(-2.0 * LOG2E)
    abs_x = x.maximumf(-x)
    e = fx.Float32(rocdl.exp2(T.f32, _raw(abs_x * neg_two_log2e)))
    recip = fx.Float32(rocdl.rcp(T.f32, _raw(fx.Float32(1.0) + e)))
    tanh_abs = (fx.Float32(1.0) - e) * recip
    is_pos = arith.cmpf(arith.CmpFPredicate.OGT, _raw(x), _raw(fx.Float32(0.0)))
    return fx.Float32(arith.select(is_pos, _raw(tanh_abs), _raw(-tanh_abs)))


def _situ_mul_batch(gs, us, situ_beta=1.0, situ_linear_beta=1.0, clamp_limit=7.0):
    """SiTUv2 activation (aiter mixed_moe situ_mul_vec4):
        situ(g)    = beta * tanh(g / beta) * sigmoid(g)
        situ_up(u) = linear_beta * tanh(u / linear_beta)
        out        = situ(clamp_gate(g)) * situ_up(clamp_lin(u))
    clamp_gate: g <= +limit (upper only); clamp_lin: u in [-limit, +limit].
    """
    neg_lim = fx.Float32(-clamp_limit)
    beta = fx.Float32(situ_beta)
    beta_rcp = fx.Float32(1.0 / situ_beta)
    lbeta = fx.Float32(situ_linear_beta)
    lbeta_rcp = fx.Float32(1.0 / situ_linear_beta)

    out = []
    for i in range(len(gs)):
        # clamp_gate: g <= +lim (upper only, via -max(-g,-lim)); clamp_lin: u in [-lim,+lim].
        g = -((-gs[i]).maximumf(neg_lim))
        u = (-((-us[i]).maximumf(neg_lim))).maximumf(neg_lim)
        situ_g = beta * _tanh_f32(g * beta_rcp) * _sigmoid_f32(g)
        situ_u = lbeta * _tanh_f32(u * lbeta_rcp)
        out.append(situ_g * situ_u)
    return out
