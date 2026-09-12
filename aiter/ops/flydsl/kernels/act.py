# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2025-2026 FlyDSL Project Contributors

"""Shared device-side gate/up activation helpers for FlyDSL MoE kernels.

Elementwise f32-register helpers (exp2/rcp-based sigmoid, sign-restored tanh, and the
gate*up batch forms) usable by any FlyDSL gemm1 fused gate+up epilog, plus
:func:`gate_up_act`, which picks between them so kernels carry no ``act`` branch.
Leaf module: depends only on flydsl and ``tensor_shim._to_raw``.
"""

from typing import NamedTuple

import flydsl.expr as fx
from flydsl.expr import const_expr, rocdl
from flydsl.expr.typing import T

from aiter.ops.flydsl.kernels.tensor_shim import _to_raw as _raw

LOG2E = 1.4426950408889634


def sigmoid_batch(xs, *, alpha=1.0):
    """Emit all exponentials before their reciprocals to preserve batch scheduling."""
    e = [
        fx.Float32(rocdl.exp2(T.f32, _raw(x * fx.Float32(-alpha * LOG2E)))) for x in xs
    ]
    return [fx.Float32(rocdl.rcp(T.f32, _raw(fx.Float32(1.0) + ei))) for ei in e]


def sigmoid_f32(g, *, alpha=1.0):
    return sigmoid_batch([g], alpha=alpha)[0]


def clamp_gate_up(g, u, neg_limit):
    """Upper-bound the gate and symmetrically clamp the up value."""
    return -fx.max(-g, neg_limit), fx.max(-fx.max(-u, neg_limit), neg_limit)


def silu_mul_batch(gs, us):
    sig = sigmoid_batch(gs)
    return [gs[i] * sig[i] * us[i] for i in range(len(gs))]


def swiglu_mul_batch(gs, us, neg_clamp_limit):
    out = []
    for i in range(len(gs)):
        gate, up = clamp_gate_up(gs[i], us[i], neg_clamp_limit)
        out.append(
            gate * sigmoid_f32(fx.Float32(1.702) * gate) * (up + fx.Float32(1.0))
        )
    return out


def tanh_batch(xs):
    """Sign-restored tanh with exp2 and reciprocal operations grouped by stage."""
    neg_two_log2e = fx.Float32(-2.0 * LOG2E)
    es = []
    for x in xs:
        abs_x = fx.max(x, -x)
        es.append(fx.Float32(rocdl.exp2(T.f32, _raw(abs_x * neg_two_log2e))))
    recips = [fx.Float32(rocdl.rcp(T.f32, _raw(fx.Float32(1.0) + e))) for e in es]
    zero = fx.Float32(0.0)
    out = []
    for i, x in enumerate(xs):
        tanh_abs = (fx.Float32(1.0) - es[i]) * recips[i]
        out.append((x > zero).select(tanh_abs, -tanh_abs))
    return out


def tanh_f32(x):
    return tanh_batch([x])[0]


def tanh_via_sigmoid_f32(x):
    """Tanh identity used by split-K activation, preserving its rounding order."""
    two = fx.Float32(2.0)
    return two * sigmoid_f32(two * x) - fx.Float32(1.0)


def situ_mul(g, u, beta, beta_rcp, lbeta, lbeta_rcp, *, tanh=tanh_f32):
    gate = beta * tanh(g * beta_rcp) * sigmoid_f32(g)
    up = lbeta * tanh(u * lbeta_rcp)
    return gate * up


def situ_mul_batch(gs, us, beta, beta_rcp, lbeta, lbeta_rcp, neg_clamp_limit):
    """SiTUv2 activation (aiter mixed_moe situ_mul_vec4):
        situ(g)    = beta * tanh(g / beta) * sigmoid(g)
        situ_up(u) = linear_beta * tanh(u / linear_beta)
        out        = situ(clamp_gate(g)) * situ_up(clamp_lin(u))
    clamp_gate: g <= +limit (upper only); clamp_lin: u in [-limit, +limit].

    beta/beta_rcp/lbeta/lbeta_rcp and neg_clamp_limit are runtime fx.Float32
    scalars (nothing baked; one kernel serves any beta/limit). neg_clamp_limit is
    -swiglu_limit (host-negated), so the clamp matches the a8w4/mixed_moe situv2
    path exactly: a +inf limit -> -inf -> maximumf no-op = no clamp; a finite
    limit clamps. Do NOT drop the clamp -- at large linear_beta the model expects
    the clamp and no-clamp diverges badly (a16w4 must match a8w4).
    """
    out = []
    for i in range(len(gs)):
        # clamp_gate: g <= +lim (upper only); clamp_lin: u in [-lim, +lim].
        g, u = clamp_gate_up(gs[i], us[i], neg_clamp_limit)
        out.append(situ_mul(g, u, beta, beta_rcp, lbeta, lbeta_rcp))
    return out


class SituParams(NamedTuple):
    """Runtime SiTUv2 scalars, as :func:`situ_mul_batch` wants them.

    Built by :func:`situ_params` so the ``-swiglu_limit`` negation happens once, next
    to the reason for it, instead of inside every caller's epilogue loop.
    """

    beta: object
    beta_rcp: object
    linear_beta: object
    linear_beta_rcp: object
    neg_clamp_limit: object


def situ_params(beta, beta_rcp, linear_beta, linear_beta_rcp, swiglu_limit):
    """Bundle the five runtime SiTUv2 f32 scalars (all fx.Float32, nothing baked)."""
    return SituParams(beta, beta_rcp, linear_beta, linear_beta_rcp, -swiglu_limit)


def gate_up_act(act, gs, us, situ=None):
    """Fused gate+up epilogue activation: ``act(gate) * up``.

    ``act`` is a compile-time string ("silu", "swiglu", or "situv2");
    ``situ`` carries runtime activation parameters for the latter two modes.
    """
    if const_expr(act == "situv2"):
        return situ_mul_batch(gs, us, *situ)
    if const_expr(act == "swiglu"):
        return swiglu_mul_batch(gs, us, situ.neg_clamp_limit)
    return silu_mul_batch(gs, us)
